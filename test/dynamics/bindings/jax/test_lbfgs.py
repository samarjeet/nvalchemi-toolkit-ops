# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the JAX L-BFGS binding.

Tests cover:

- The registration contract: input-output aliasing, argument order and graph
  mode. These need no GPU and catch ABI drift cheaply.
- Relaxation results, checked against the Warp layer rather than restating the
  algorithm.
- Donation and CUDA-graph replay, including that the capture count stays
  bounded rather than growing with the step count.
- That the step is not differentiable, which is a contract rather than an
  oversight.
"""

from __future__ import annotations

import functools
import inspect
import warnings

import numpy as np
import pytest

from nvalchemiops.dynamics.optimizers.lbfgs import LBFGSCellState, LBFGSState

from .conftest import requires_gpu

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from nvalchemiops.jax.lbfgs import (  # noqa: E402
    LBFGS_CONVERGED,
    LBFGS_NEED_EVAL,
    lbfgs_allocate_state,
    lbfgs_converged,
    lbfgs_step_coord,
)

STIFFNESS = np.array([1.0, 4.0, 9.0])


def _cluster(num_atoms, seed=42, scale=2.0):
    return np.random.default_rng(seed).normal(size=(num_atoms, 3)) * scale


class JaxDriver:
    """Relaxes an anisotropic quadratic through the JAX binding."""

    def __init__(self, positions, num_systems=1, dtype=jnp.float64, history_size=6):
        self.num_dofs = positions.shape[0]
        self.num_systems = num_systems
        per_system = self.num_dofs // num_systems
        self.dtype = dtype
        self.positions = jnp.asarray(positions, dtype)
        self.batch_idx = jnp.repeat(
            jnp.arange(num_systems, dtype=jnp.int32), per_system
        )
        self.n_particles = jnp.full((num_systems,), per_system, jnp.int32)
        self.state = lbfgs_allocate_state(
            self.num_dofs, num_systems, dtype=dtype, history_size=history_size
        )
        self.stiffness = jnp.asarray(STIFFNESS)
        self.n_evals = 0

    def model(self, positions):
        x = positions.astype(jnp.float64)
        per_atom = 0.5 * (self.stiffness * x**2).sum(axis=1)
        energy = jax.ops.segment_sum(per_atom, self.batch_idx, self.num_systems)
        return energy, (-(self.stiffness * x)).astype(self.dtype)

    def step(self, **kwargs):
        energy, forces = self.model(self.positions)
        self.n_evals += 1
        self.positions, self.state = lbfgs_step_coord(
            self.positions,
            self.state,
            forces,
            energy,
            self.batch_idx,
            self.n_particles,
            **kwargs,
        )

    def run(self, max_evals=200, **kwargs):
        for _ in range(max_evals):
            self.step(**kwargs)
            if bool(lbfgs_converged(self.state)):
                break
        return self


class TestLBFGSJaxRegistration:
    """The registration contract. These run without a GPU."""

    def test_in_out_argnames_match_the_state_fields(self):
        """The aliased arrays are exactly ``positions`` plus every state field."""
        from nvalchemiops.jax.lbfgs import _LBFGS_IN_OUT_ARGS

        assert _LBFGS_IN_OUT_ARGS == ("positions",) + LBFGSState._fields

    @pytest.mark.parametrize("suffix", ["f32", "f64"])
    def test_body_parameter_order_matches_the_state(self, suffix):
        """A reordering would silently swap two arrays; nothing else catches it."""
        import nvalchemiops.jax.lbfgs as module

        body = getattr(module, f"_lbfgs_body_{suffix}")
        params = tuple(inspect.signature(body).parameters)
        offset = 5  # forces, energy, batch_idx, n_particles, positions
        assert params[offset : offset + len(LBFGSState._fields)] == LBFGSState._fields

    @pytest.mark.parametrize("graph_mode", ["none", "warp", "warp_staged"])
    def test_callable_has_no_pure_outputs(self, graph_mode):
        """Every output is an alias of an input.

        With no pure outputs, warp's "in-out before output" ordering rule
        cannot bind. Any future diagnostic array must therefore be added
        *after* every aliased one.
        """
        from nvalchemiops.jax.lbfgs import _LBFGS_IN_OUT_ARGS, _get_callable

        call = _get_callable(jnp.float64, graph_mode)
        assert call.num_in_out == len(_LBFGS_IN_OUT_ARGS)
        assert call.num_outputs == len(_LBFGS_IN_OUT_ARGS)
        assert len(call.output_args) == 0

    def test_unknown_graph_mode_is_rejected(self):
        from nvalchemiops.jax.lbfgs import _get_callable

        with pytest.raises(ValueError, match="graph_mode must be one of"):
            _get_callable(jnp.float64, "cuda-graphs-please")

    def test_unsupported_dtype_is_rejected(self):
        with pytest.raises(ValueError, match="float32 or float64"):
            lbfgs_allocate_state(4, 1, dtype=jnp.float16)

    def test_state_is_a_pytree_with_one_leaf_per_field(self):
        """Donating the state as a single object depends on this."""
        state = lbfgs_allocate_state(5, 2, dtype=jnp.float64)
        leaves, treedef = jax.tree_util.tree_flatten(state)
        assert len(leaves) == len(LBFGSState._fields)
        assert isinstance(jax.tree_util.tree_unflatten(treedef, leaves), LBFGSState)

    def test_allocation_initial_values(self):
        """Three fields start non-zero; a blanket zeros() would be wrong."""
        state = lbfgs_allocate_state(7, 3, dtype=jnp.float64, history_size=4)
        assert state.s_history.shape == (4, 7, 3)
        assert state.ys.shape == (4, 3)
        assert state.gg.dtype == jnp.float64
        np.testing.assert_array_equal(state.iteration, np.full(3, -1))
        np.testing.assert_array_equal(state.alpha_step, np.ones(3))
        np.testing.assert_array_equal(state.status, np.zeros(3, np.int32))


@pytest.mark.parametrize("_gpu", [pytest.param(None, marks=requires_gpu)])
class TestLBFGSJax:
    """Behaviour on device."""

    def test_reaches_the_minimum(self, _gpu):
        d = JaxDriver(_cluster(8)).run(force_tol=1e-8, maxstep=0.5)
        assert int(d.state.status[0]) == LBFGS_CONVERGED
        assert float(jnp.abs(d.positions).max()) < 1e-7

    def test_batched_systems_converge_independently(self, _gpu):
        rng = np.random.default_rng(17)
        blocks = [rng.normal(size=(4, 3)) * s for s in (0.001, 1.0, 5.0)]
        d = JaxDriver(np.vstack(blocks), num_systems=3).run(force_tol=1e-8, maxstep=0.5)
        np.testing.assert_array_equal(
            np.asarray(d.state.status), np.full(3, LBFGS_CONVERGED)
        )

    def test_matches_the_warp_layer(self, _gpu):
        """A thin adapter must agree with the Warp layer step for step."""
        import warp as wp

        from nvalchemiops.dynamics.optimizers.lbfgs import (
            lbfgs_reset as warp_reset,
        )
        from nvalchemiops.dynamics.optimizers.lbfgs import lbfgs_step as warp_step

        from ...conftest import make_lbfgs_state

        start = _cluster(5, seed=17)
        d = JaxDriver(start)

        n = start.shape[0]
        device = "cuda:0"
        wp_pos = wp.array(start.copy(), dtype=wp.vec3d, device=device)
        wp_forces = wp.zeros(n, dtype=wp.vec3d, device=device)
        wp_energy = wp.zeros(1, dtype=wp.float64, device=device)
        wp_batch = wp.zeros(n, dtype=wp.int32, device=device)
        wp_nparts = wp.array(np.array([n], np.int32), dtype=wp.int32, device=device)
        wp_state = make_lbfgs_state(n, 1, 6, wp.vec3d, device)
        warp_reset(**wp_state)

        for _ in range(30):
            d.step(force_tol=1e-8, maxstep=0.5)
            x = wp_pos.numpy()
            wp_forces.assign(-(STIFFNESS * x))
            wp_energy.assign(np.array([(0.5 * STIFFNESS * x**2).sum()]))
            warp_step(
                positions=wp_pos,
                forces=wp_forces,
                energy=wp_energy,
                batch_idx=wp_batch,
                n_particles=wp_nparts,
                force_tol=1e-8,
                maxstep=0.5,
                **wp_state,
            )
            wp.synchronize()
            np.testing.assert_allclose(
                np.asarray(d.positions), wp_pos.numpy(), rtol=1e-12, atol=1e-14
            )
            if bool(lbfgs_converged(d.state)):
                break
        np.testing.assert_array_equal(
            np.asarray(d.state.status), wp_state["status"].numpy()
        )

    @pytest.mark.parametrize("graph_mode", ["none", "warp", "warp_staged"])
    def test_graph_modes_agree(self, _gpu, graph_mode):
        """Capture must not change the answer.

        Compared bit-for-bit against the ungraphed baseline from an identical
        start: replay that silently drops work would show up here.
        """
        start = _cluster(6, seed=23)
        reference = JaxDriver(start).run(
            force_tol=1e-10, maxstep=0.5, graph_mode="none"
        )
        candidate = JaxDriver(start).run(
            force_tol=1e-10, maxstep=0.5, graph_mode=graph_mode
        )
        np.testing.assert_array_equal(
            np.asarray(candidate.positions), np.asarray(reference.positions)
        )
        np.testing.assert_array_equal(
            np.asarray(candidate.state.status), np.asarray(reference.state.status)
        )

    def test_donated_replay_keeps_the_capture_count_bounded(self, _gpu):
        """The graph working set must plateau, not grow with the step count.

        ``forces`` arrives from the model with a fresh buffer every step, and
        the capture is keyed on input addresses, so a small set of graphs is
        expected rather than exactly one. What must not happen is a capture per
        call, which would make graph mode slower than no graph at all.
        """
        from nvalchemiops.jax.lbfgs import _get_callable

        d = JaxDriver(_cluster(6, seed=31))
        batch_idx, n_particles = d.batch_idx, d.n_particles

        @functools.partial(jax.jit, donate_argnums=(0, 1))
        def relax_step(positions, state, forces, energy):
            return lbfgs_step_coord(
                positions,
                state,
                forces,
                energy,
                batch_idx,
                n_particles,
                force_tol=1e-12,
                maxstep=0.5,
            )

        call = _get_callable(jnp.float64, "warp")
        # The callable is cached module-wide, so earlier tests may already have
        # populated it. Measure growth from here rather than absolute counts.
        baseline = len(getattr(call, "captures", {}))
        counts = []
        with warnings.catch_warnings():
            # A refused donation is only a warning; make it fail the test.
            warnings.simplefilter("error", UserWarning)
            for i in range(40):
                energy, forces = d.model(d.positions)
                d.positions, d.state = relax_step(d.positions, d.state, forces, energy)
                if i in (9, 19, 39):
                    jax.block_until_ready(d.positions)
                    counts.append(len(getattr(call, "captures", {})) - baseline)

        # A little growth is expected as JAX cycles through a pool of buffers;
        # what would mean replay is not happening is growth proportional to the
        # step count. Measured working set on this system is three to five.
        steps = 40
        assert counts[-1] <= steps // 5, (
            f"capture count {counts} is growing with the step count; "
            "replay is not happening, so graph_mode='warp_staged' is needed"
        )
        assert counts[-1] - counts[0] <= 2, (
            f"working set still growing between steps 10 and 40: {counts}"
        )

    def test_step_is_not_differentiable(self, _gpu):
        """Differentiating must fail loudly rather than return zeros."""
        d = JaxDriver(_cluster(4))
        energy, forces = d.model(d.positions)

        def loss(positions):
            out, _ = lbfgs_step_coord(
                positions,
                d.state,
                forces,
                energy,
                d.batch_idx,
                d.n_particles,
                force_tol=1e-8,
            )
            return out.sum()

        with pytest.raises(Exception):
            jax.grad(loss)(d.positions)


@pytest.mark.parametrize("_gpu", [pytest.param(None, marks=requires_gpu)])
class TestLBFGSJaxErrors:
    """Input validation."""

    def test_force_shape_mismatch(self, _gpu):
        d = JaxDriver(_cluster(3))
        energy, _ = d.model(d.positions)
        with pytest.raises(ValueError, match="forces shape"):
            lbfgs_step_coord(
                d.positions,
                d.state,
                jnp.zeros((2, 3)),
                energy,
                d.batch_idx,
                d.n_particles,
            )

    def test_energy_must_be_float64(self, _gpu):
        d = JaxDriver(_cluster(3))
        energy, forces = d.model(d.positions)
        with pytest.raises(ValueError, match="energy must be float64"):
            lbfgs_step_coord(
                d.positions,
                d.state,
                forces,
                energy.astype(jnp.float32),
                d.batch_idx,
                d.n_particles,
            )

    def test_history_size_mismatch(self, _gpu):
        d = JaxDriver(_cluster(3))
        energy, forces = d.model(d.positions)
        other = lbfgs_allocate_state(9, 1, dtype=jnp.float64)
        with pytest.raises(ValueError, match="history buffers"):
            lbfgs_step_coord(
                d.positions,
                other,
                forces,
                energy,
                d.batch_idx,
                d.n_particles,
            )


@pytest.mark.parametrize("_gpu", [pytest.param(None, marks=requires_gpu)])
class TestLBFGSJaxCoordCell:
    """The variable-cell binding."""

    @staticmethod
    def _setup(num_atoms=6, seed=11):
        from nvalchemiops.jax.lbfgs import (
            lbfgs_allocate_cell_state,
            lbfgs_set_reference_cell,
        )

        from ...conftest import CellPotential

        rng = np.random.default_rng(seed)
        s0 = np.array([0.1, -0.2, 0.05])
        potential = CellPotential(s0)
        cell_np = np.diag([6.0, 6.5, 7.0])
        frac = s0 + rng.normal(size=(num_atoms, 3)) * 0.05
        positions = jnp.asarray(np.ascontiguousarray((cell_np @ frac.T).T))
        cell = jnp.asarray(cell_np[None])

        cell_state = lbfgs_allocate_cell_state(num_atoms, 1, dtype=jnp.float64)
        cell_state = lbfgs_set_reference_cell(cell, cell_state)
        state = lbfgs_allocate_state(num_atoms + 2, 1, dtype=jnp.float64)
        return positions, cell, state, cell_state, potential

    def test_relaxes_cell_and_coordinates(self, _gpu):
        """A compressed cell expands to the target volume while atoms relax."""
        from nvalchemiops.jax.lbfgs import lbfgs_step_coord_cell

        n = 6
        positions, cell, state, cell_state, potential = self._setup(n)
        batch_idx = jnp.zeros(n, jnp.int32)
        n_particles = jnp.full((1,), n, jnp.int32)

        for _ in range(400):
            e, f, s = potential.energy_forces_stress(
                np.asarray(positions), np.asarray(cell)[0]
            )
            positions, cell, state, cell_state = lbfgs_step_coord_cell(
                positions,
                cell,
                state,
                cell_state,
                jnp.asarray(f),
                jnp.asarray(s[None]),
                jnp.asarray([e]),
                batch_idx,
                n_particles,
                force_tol=1e-6,
                stress_tol=1e-6,
                maxstep=0.2,
            )
            if int(state.status[0]) != LBFGS_NEED_EVAL:
                break

        assert int(state.status[0]) == LBFGS_CONVERGED, int(state.status[0])
        volume = abs(np.linalg.det(np.asarray(cell)[0]))
        np.testing.assert_allclose(volume, potential.target_volume, rtol=1e-4)

    def test_matches_the_warp_layer(self, _gpu):
        """A thin adapter must agree with the Warp layer step for step."""
        import warp as wp

        from nvalchemiops.dynamics.optimizers.lbfgs import (
            LBFGSState as WarpState,
        )
        from nvalchemiops.dynamics.optimizers.lbfgs import (
            lbfgs_reset as warp_reset,
        )
        from nvalchemiops.dynamics.optimizers.lbfgs import (
            lbfgs_set_reference_cell as warp_set_ref,
        )
        from nvalchemiops.dynamics.optimizers.lbfgs import (
            lbfgs_step_coord_cell as warp_step_cell,
        )
        from nvalchemiops.jax.lbfgs import lbfgs_step_coord_cell

        from ...conftest import make_lbfgs_cell_state, make_lbfgs_state

        n, device = 6, "cuda:0"
        positions, cell, state, cell_state, potential = self._setup(n)
        batch_idx = jnp.zeros(n, jnp.int32)
        n_particles = jnp.full((1,), n, jnp.int32)

        start_pos = np.asarray(positions).copy()
        start_cell = np.asarray(cell)[0].copy()
        wp_pos = wp.array(start_pos, dtype=wp.vec3d, device=device)
        wp_cell = wp.array(start_cell[None], dtype=wp.mat33d, device=device)
        wp_forces = wp.zeros(n, dtype=wp.vec3d, device=device)
        wp_stress = wp.zeros(1, dtype=wp.mat33d, device=device)
        wp_energy = wp.zeros(1, dtype=wp.float64, device=device)
        wp_batch = wp.zeros(n, dtype=wp.int32, device=device)
        wp_nparts = wp.array(np.array([n], np.int32), dtype=wp.int32, device=device)
        wp_cell_state = make_lbfgs_cell_state(n, 1, wp.vec3d, device)
        warp_set_ref(wp_cell, wp_cell_state.ref_cell, wp_cell_state.ref_cell_inv)
        wp_state = make_lbfgs_state(n + 2, 1, 6, wp.vec3d, device)
        warp_reset(**wp_state)

        for _ in range(25):
            e, f, s = potential.energy_forces_stress(
                np.asarray(positions), np.asarray(cell)[0]
            )
            positions, cell, state, cell_state = lbfgs_step_coord_cell(
                positions,
                cell,
                state,
                cell_state,
                jnp.asarray(f),
                jnp.asarray(s[None]),
                jnp.asarray([e]),
                batch_idx,
                n_particles,
                force_tol=1e-8,
                stress_tol=1e-8,
                maxstep=0.2,
            )

            e2, f2, s2 = potential.energy_forces_stress(
                wp_pos.numpy(), wp_cell.numpy()[0]
            )
            wp_forces.assign(f2)
            wp_stress.assign(s2[None])
            wp_energy.assign(np.array([e2]))
            warp_step_cell(
                wp_pos,
                wp_forces,
                wp_cell,
                wp_stress,
                wp_energy,
                wp_batch,
                wp_nparts,
                wp_cell_state,
                WarpState(**wp_state),
                force_tol=1e-8,
                stress_tol=1e-8,
                maxstep=0.2,
            )
            wp.synchronize()
            np.testing.assert_allclose(
                np.asarray(positions), wp_pos.numpy(), rtol=1e-12, atol=1e-14
            )
            np.testing.assert_allclose(
                np.asarray(cell), wp_cell.numpy(), rtol=1e-12, atol=1e-14
            )
            if int(state.status[0]) != LBFGS_NEED_EVAL:
                break
        np.testing.assert_array_equal(
            np.asarray(state.status), wp_state["status"].numpy()
        )

    def test_cell_callable_aliases_every_mutable_array(self, _gpu):
        """All 42 outputs are aliases; a future pure output must come last."""
        from nvalchemiops.jax.lbfgs import _CELL_IN_OUT_ARGS, _get_cell_callable

        call = _get_cell_callable(jnp.float64, "warp")
        assert call.num_in_out == len(_CELL_IN_OUT_ARGS)
        assert call.num_outputs == len(_CELL_IN_OUT_ARGS)
        assert len(call.output_args) == 0
        assert _CELL_IN_OUT_ARGS == (
            ("positions", "cell") + LBFGSState._fields + LBFGSCellState._fields
        )

    def test_state_sized_for_the_wrong_dof_count_is_rejected(self, _gpu):
        from nvalchemiops.jax.lbfgs import lbfgs_step_coord_cell

        n = 6
        positions, cell, _, cell_state, potential = self._setup(n)
        wrong = lbfgs_allocate_state(n, 1, dtype=jnp.float64)
        e, f, s = potential.energy_forces_stress(
            np.asarray(positions), np.asarray(cell)[0]
        )
        with pytest.raises(ValueError, match="num_atoms \\+ 2 \\* num_systems"):
            lbfgs_step_coord_cell(
                positions,
                cell,
                wrong,
                cell_state,
                jnp.asarray(f),
                jnp.asarray(s[None]),
                jnp.asarray([e]),
                jnp.zeros(n, jnp.int32),
                jnp.full((1,), n, jnp.int32),
            )

    def test_jit_with_donation_matches_uncompiled(self, _gpu):
        """The variable-cell step runs under jit with everything donated.

        Donating ``cell`` and ``cell_state`` together only works because the
        reference cell is stored as a copy; sharing the caller's buffer would
        make XLA reject the same buffer being donated twice.
        """
        from nvalchemiops.jax.lbfgs import lbfgs_step_coord_cell

        n = 6
        batch_idx = jnp.zeros(n, jnp.int32)
        n_particles = jnp.full((1,), n, jnp.int32)
        opts = dict(force_tol=1e-8, stress_tol=1e-8, maxstep=0.2)

        def one_run(jit):
            positions, cell, state, cell_state, potential = self._setup(n)
            e, f, s = potential.energy_forces_stress(
                np.asarray(positions), np.asarray(cell)[0]
            )

            def body(p, c, st, cs, f_, s_, e_):
                return lbfgs_step_coord_cell(
                    p, c, st, cs, f_, s_, e_, batch_idx, n_particles, **opts
                )

            fn = jax.jit(body, donate_argnums=(0, 1, 2, 3)) if jit else body
            out = fn(
                positions,
                cell,
                state,
                cell_state,
                jnp.asarray(f),
                jnp.asarray(s[None]),
                jnp.asarray([e]),
            )
            jax.block_until_ready(out[0])
            return np.asarray(out[0]), np.asarray(out[1]), np.asarray(out[2].status)

        eager = one_run(False)
        jitted = one_run(True)
        np.testing.assert_array_equal(jitted[0], eager[0])
        np.testing.assert_array_equal(jitted[1], eager[1])
        np.testing.assert_array_equal(jitted[2], eager[2])

    def test_reference_cell_is_an_independent_copy(self, _gpu):
        """``ref_cell`` must not share a buffer with the caller's ``cell``.

        If it did, donating both to one jitted step would fail outright.
        """
        from nvalchemiops.jax.lbfgs import (
            lbfgs_allocate_cell_state,
            lbfgs_set_reference_cell,
        )

        cell = jnp.asarray(np.diag([6.0, 6.5, 7.0])[None])
        cell_state = lbfgs_allocate_cell_state(6, 1, dtype=jnp.float64)
        cell_state = lbfgs_set_reference_cell(cell, cell_state)
        np.testing.assert_array_equal(np.asarray(cell_state.ref_cell), np.asarray(cell))
        assert cell_state.ref_cell is not cell, (
            "ref_cell aliases the caller's cell; donating both would be rejected"
        )
