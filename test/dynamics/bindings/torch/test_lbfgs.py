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

"""Tests for the PyTorch L-BFGS binding.

Tests cover:

- The registered operator's schema, and that it stays in step with the state
  container it is generated from.
- Tracing under ``make_fx`` and compilation under ``torch.compile``.
- CUDA-graph capture and replay, and that a step allocates nothing.
- Relaxation results, checked against the Warp layer rather than restating the
  algorithm.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest
import torch

from nvalchemiops.dynamics.optimizers.lbfgs import LBFGSState
from nvalchemiops.torch.lbfgs import (
    LBFGS_CONVERGED,
    LBFGS_NEED_EVAL,
    lbfgs_allocate_state,
    lbfgs_reduce_energy,
    lbfgs_reset,
    lbfgs_step_coord,
)

DEVICES = ["cuda:0"]
DTYPES = [
    pytest.param(torch.float32, id="dof_f32"),
    pytest.param(torch.float64, id="dof_f64"),
]
STIFFNESS = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)


class TorchDriver:
    """Relaxes an anisotropic quadratic through the Torch binding."""

    def __init__(self, positions, num_systems, dtype, device, history_size=6):
        self.device = device
        self.dtype = dtype
        self.num_dofs = positions.shape[0]
        self.num_systems = num_systems
        per_system = self.num_dofs // num_systems

        self.positions = torch.tensor(positions, dtype=dtype, device=device)
        self.forces = torch.zeros_like(self.positions)
        self.per_atom_energy = torch.zeros(
            self.num_dofs, dtype=torch.float64, device=device
        )
        self.energy = torch.zeros(num_systems, dtype=torch.float64, device=device)
        self.batch_idx = torch.repeat_interleave(
            torch.arange(num_systems, dtype=torch.int32, device=device), per_system
        )
        self.n_particles = torch.full(
            (num_systems,), per_system, dtype=torch.int32, device=device
        )
        self.state = lbfgs_allocate_state(
            self.num_dofs,
            num_systems,
            dtype=dtype,
            device=device,
            history_size=history_size,
        )
        self.stiffness = STIFFNESS.to(device)
        self.n_evals = 0

    def evaluate(self):
        x = self.positions.to(torch.float64)
        self.per_atom_energy.copy_(0.5 * (self.stiffness * x**2).sum(dim=1))
        self.forces.copy_((-(self.stiffness * x)).to(self.dtype))
        lbfgs_reduce_energy(self.per_atom_energy, self.batch_idx, self.energy)
        self.n_evals += 1

    def step(self, **kwargs):
        lbfgs_step_coord(
            self.positions,
            self.state,
            self.forces,
            self.energy,
            self.batch_idx,
            self.n_particles,
            **kwargs,
        )

    def run(self, max_evals=200, **kwargs):
        for _ in range(max_evals):
            self.evaluate()
            self.step(**kwargs)
            torch.cuda.synchronize()
            if not (self.state.status == LBFGS_NEED_EVAL).any():
                break
        return self


def _cluster(num_systems, atoms_per_system, seed=42, scale=2.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(num_systems * atoms_per_system, 3)) * scale


class TestLBFGSTorchState:
    """The state container and its relationship to the registered operator."""

    def test_operator_parameters_match_state_fields_in_order(self):
        """A reordering here would silently swap two tensors at the boundary.

        Registration already rejects a name in ``mutates_args`` that does not
        exist as a parameter, but only an ordered comparison catches a swap.
        """
        from nvalchemiops.torch.lbfgs import _lbfgs_step_op

        params = tuple(inspect.signature(_lbfgs_step_op).parameters)
        offset = 5  # positions, forces, energy, batch_idx, n_particles
        assert params[offset : offset + len(LBFGSState._fields)] == LBFGSState._fields

    def test_mutates_args_covers_every_state_field(self):
        """Every field is declared mutable, and nothing else is."""
        from nvalchemiops.torch.lbfgs import _MUTATED

        assert set(_MUTATED) == {"positions"} | set(LBFGSState._fields)
        assert len(_MUTATED) == len(LBFGSState._fields) + 1

    def test_schema_arity_matches_the_signature(self):
        """The registered schema sees every argument the implementation takes."""
        from nvalchemiops.torch.lbfgs import _lbfgs_step_op

        schema = torch.ops.nvalchemiops.lbfgs_step.default._schema
        assert len(schema.arguments) == len(
            inspect.signature(_lbfgs_step_op).parameters
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_allocation_shapes_and_initial_values(self, device, dtype):
        state = lbfgs_allocate_state(7, 3, dtype=dtype, device=device, history_size=4)
        assert state.x_base.shape == (7, 3)
        assert state.s_history.shape == (4, 7, 3)
        assert state.ys.shape == (4, 3)
        assert state.x_base.dtype == dtype
        # Per-system scalars are float64 whatever the coordinate precision.
        assert state.gg.dtype == torch.float64
        assert state.status.dtype == torch.int32
        torch.testing.assert_close(
            state.iteration, torch.full((3,), -1, dtype=torch.int32, device=device)
        )
        torch.testing.assert_close(
            state.alpha_step, torch.ones(3, dtype=torch.float64, device=device)
        )
        assert (state.status == LBFGS_NEED_EVAL).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_reset_restores_the_non_zero_defaults(self, device):
        """Reset is not a blanket ``zero_()``; three fields start non-zero."""
        d = TorchDriver(_cluster(1, 4), 1, torch.float64, device).run(
            force_tol=1e-6, maxstep=0.5
        )
        assert d.state.history_count.item() > 0
        lbfgs_reset(d.state)
        assert d.state.history_count.item() == 0
        assert d.state.iteration.item() == -1
        assert d.state.alpha_step.item() == 1.0
        assert d.state.status.item() == LBFGS_NEED_EVAL

    def test_bad_dtype_is_rejected(self):
        with pytest.raises(ValueError, match="float32 or float64"):
            lbfgs_allocate_state(4, 1, dtype=torch.float16, device="cuda:0")


class TestLBFGSTorchCoord:
    """Relaxation through the binding."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_reaches_the_minimum(self, device, dtype):
        force_tol = 1e-3 if dtype == torch.float32 else 1e-8
        d = TorchDriver(_cluster(2, 4), 2, dtype, device).run(
            force_tol=force_tol, maxstep=0.5
        )
        assert (d.state.status == LBFGS_CONVERGED).all(), d.state.status
        assert d.positions.abs().max().item() < 10 * force_tol

    @pytest.mark.parametrize("device", DEVICES)
    def test_matches_the_warp_layer(self, device):
        """The binding is a thin adapter, so it must agree exactly.

        Compared step by step against the Warp implementation on identical
        inputs. Both run the same kernels in the same order, so this is an
        exact comparison rather than a tolerance check.
        """
        import warp as wp

        from nvalchemiops.dynamics.optimizers.lbfgs import lbfgs_step as warp_step

        from ...conftest import make_lbfgs_state

        start = _cluster(1, 5, seed=17)
        torch_driver = TorchDriver(start, 1, torch.float64, device)

        num_dofs = start.shape[0]
        wp_positions = wp.array(start.copy(), dtype=wp.vec3d, device=device)
        wp_forces = wp.zeros(num_dofs, dtype=wp.vec3d, device=device)
        wp_energy = wp.zeros(1, dtype=wp.float64, device=device)
        wp_batch = wp.zeros(num_dofs, dtype=wp.int32, device=device)
        wp_nparts = wp.array(
            np.array([num_dofs], np.int32), dtype=wp.int32, device=device
        )
        wp_state = make_lbfgs_state(num_dofs, 1, 6, wp.vec3d, device)
        from nvalchemiops.dynamics.optimizers.lbfgs import lbfgs_reset as warp_reset

        warp_reset(**wp_state)
        stiffness = STIFFNESS.numpy()

        for _ in range(30):
            torch_driver.evaluate()
            torch_driver.step(force_tol=1e-8, maxstep=0.5)

            x = wp_positions.numpy()
            wp_forces.assign(-(stiffness * x))
            wp_energy.assign(np.array([(0.5 * stiffness * x**2).sum()]))
            warp_step(
                positions=wp_positions,
                forces=wp_forces,
                energy=wp_energy,
                batch_idx=wp_batch,
                n_particles=wp_nparts,
                force_tol=1e-8,
                maxstep=0.5,
                **wp_state,
            )
            wp.synchronize()
            torch.cuda.synchronize()
            np.testing.assert_array_equal(
                torch_driver.positions.cpu().numpy(), wp_positions.numpy()
            )
            if not (torch_driver.state.status == LBFGS_NEED_EVAL).any():
                break
        np.testing.assert_array_equal(
            torch_driver.state.status.cpu().numpy(), wp_state["status"].numpy()
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_per_atom_energy_reduction_is_float64(self, device):
        """The reduction must accumulate in float64 even from float32 input.

        A float32 accumulation loses the resolution the Armijo test depends on,
        so this asserts the result is strictly closer to an exact sum than a
        float32 accumulation would be.
        """
        rng = np.random.default_rng(5)
        n = 4096
        values = (rng.normal(size=n) - 10.0) * 1000.0
        per_atom = torch.tensor(values, dtype=torch.float32, device=device)
        batch_idx = torch.zeros(n, dtype=torch.int32, device=device)
        energy = torch.zeros(1, dtype=torch.float64, device=device)
        lbfgs_reduce_energy(per_atom, batch_idx, energy)

        exact = np.float64(values.astype(np.float32)).sum()
        got = energy.item()
        naive_f32 = float(per_atom.sum().item())
        assert abs(got - exact) <= abs(naive_f32 - exact), (
            f"float64 accumulation ({got}) is no better than float32 ({naive_f32})"
        )


class TestLBFGSTorchRegistration:
    """Tracing, compilation and graph capture."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_traces_as_a_custom_op(self, device):
        """``make_fx`` must see one opaque call with the full argument list.

        This is what catches a missing or wrong fake registration, and any
        drift between the schema and the call site.
        """
        from torch.fx.experimental.proxy_tensor import make_fx

        d = TorchDriver(_cluster(1, 3), 1, torch.float64, device)
        d.evaluate()

        def run(positions, forces, energy, batch_idx, n_particles, *state):
            lbfgs_step_coord(
                positions,
                LBFGSState(*state),
                forces,
                energy,
                batch_idx,
                n_particles,
                force_tol=1e-8,
                maxstep=0.5,
            )
            return positions

        graph = make_fx(run, tracing_mode="fake")(
            d.positions,
            d.forces,
            d.energy,
            d.batch_idx,
            d.n_particles,
            *d.state,
        )
        target = torch.ops.nvalchemiops.lbfgs_step.default
        nodes = [n for n in graph.graph.nodes if n.target is target]
        assert len(nodes) == 1, "the step did not trace as a single custom op"
        assert len(nodes[0].args) == len(target._schema.arguments)

    @pytest.mark.parametrize("device", DEVICES)
    def test_compiles_fullgraph_without_breaks(self, device):
        """``torch.compile(fullgraph=True)`` must succeed and match eager."""
        eager = TorchDriver(_cluster(1, 4), 1, torch.float64, device)
        eager.evaluate()
        eager.step(force_tol=1e-8, maxstep=0.5)
        torch.cuda.synchronize()

        compiled_driver = TorchDriver(_cluster(1, 4), 1, torch.float64, device)
        compiled_driver.evaluate()

        torch._dynamo.reset()

        def run(positions, forces):
            lbfgs_step_coord(
                positions,
                compiled_driver.state,
                forces,
                compiled_driver.energy,
                compiled_driver.batch_idx,
                compiled_driver.n_particles,
                force_tol=1e-8,
                maxstep=0.5,
            )

        torch.compile(run, fullgraph=True)(
            compiled_driver.positions, compiled_driver.forces
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(compiled_driver.positions, eager.positions)

    @pytest.mark.parametrize("device", DEVICES)
    def test_compiles_with_zero_graph_breaks(self, device):
        """Assert the break count directly, not just that fullgraph succeeded.

        Also compiles a region containing a toy force model, which is how a
        caller would actually use it: the whole loop body should be one graph.
        """
        d = TorchDriver(_cluster(1, 4), 1, torch.float64, device)
        d.evaluate()

        def step_only(positions, forces, energy):
            lbfgs_step_coord(
                positions,
                d.state,
                forces,
                energy,
                d.batch_idx,
                d.n_particles,
                force_tol=1e-8,
                maxstep=0.5,
            )

        def step_with_model(positions, forces, energy):
            forces.copy_(-(d.stiffness * positions))
            energy.copy_((0.5 * (d.stiffness * positions**2).sum()).reshape(1))
            lbfgs_step_coord(
                positions,
                d.state,
                forces,
                energy,
                d.batch_idx,
                d.n_particles,
                force_tol=1e-8,
                maxstep=0.5,
            )

        for fn in (step_only, step_with_model):
            torch._dynamo.reset()
            explanation = torch._dynamo.explain(fn)(d.positions, d.forces, d.energy)
            assert explanation.graph_break_count == 0, (
                f"{fn.__name__} broke the graph {explanation.graph_break_count} times"
            )
            assert explanation.graph_count == 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_step_allocates_nothing(self, device):
        """A pre-allocated state must not grow memory across steps."""
        d = TorchDriver(_cluster(1, 6), 1, torch.float64, device)
        for _ in range(3):  # warm up caching allocator and kernel cache
            d.evaluate()
            d.step(force_tol=1e-10, maxstep=0.5)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(device)
        for _ in range(10):
            d.evaluate()
            d.step(force_tol=1e-10, maxstep=0.5)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated(device) == before

    @pytest.mark.parametrize("device", DEVICES)
    def test_cuda_graph_replay_matches_eager(self, device):
        """The step captures in a non-empty CUDA graph and replays correctly.

        Only holds because every buffer is pre-allocated, all zeroing happens
        inside the operator, and Warp launches are bound to PyTorch's stream.
        Without the stream binding the capture would silently record nothing.
        """
        import warp as wp

        start = _cluster(1, 5, seed=23)
        eager = TorchDriver(start, 1, torch.float64, device)
        graphed = TorchDriver(start, 1, torch.float64, device)
        opts = dict(force_tol=1e-12, maxstep=0.5)

        # Identical warm-up on both so they enter the comparison in step.
        for driver in (eager, graphed):
            driver.evaluate()
            driver.step(**opts)
        torch.cuda.synchronize()
        wp.synchronize()

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                graphed.step(**opts)
        torch.cuda.current_stream().wait_stream(side)
        for _ in range(3):
            eager.step(**opts)
        torch.cuda.synchronize()
        torch.testing.assert_close(graphed.positions, eager.positions)

        graph = torch.cuda.CUDAGraph()
        with warnings.catch_warnings():
            # An empty capture is only a warning, so make it fail the test.
            warnings.simplefilter("error", UserWarning)
            # Warp launches must be bound to the capture stream by the caller.
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
                with torch.cuda.graph(graph):
                    graphed.step(**opts)
        torch.cuda.synchronize()

        eager.step(**opts)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graphed.positions, eager.positions)
        assert torch.isfinite(graphed.positions).all()


class TestLBFGSTorchErrors:
    """Input validation."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_force_shape_mismatch(self, device):
        d = TorchDriver(_cluster(1, 3), 1, torch.float64, device)
        d.evaluate()
        with pytest.raises(ValueError, match="forces shape"):
            lbfgs_step_coord(
                d.positions,
                d.state,
                torch.zeros(2, 3, dtype=torch.float64, device=device),
                d.energy,
                d.batch_idx,
                d.n_particles,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_non_contiguous_input_is_rejected(self, device):
        """A non-contiguous tensor must raise, not be silently copied.

        These operations write through a zero-copy view, so copying the input
        would discard every update and leave the caller watching an optimizer
        that never moves. Loud beats silent.
        """
        d = TorchDriver(_cluster(1, 4), 1, torch.float64, device)
        d.evaluate()
        # A strided view is the usual way to end up here by accident.
        strided = torch.zeros(4, 6, dtype=torch.float64, device=device)[:, ::2]
        assert not strided.is_contiguous()
        with pytest.raises(ValueError, match="must be contiguous"):
            lbfgs_step_coord(
                strided,
                d.state,
                d.forces,
                d.energy,
                d.batch_idx,
                d.n_particles,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_energy_must_be_float64(self, device):
        d = TorchDriver(_cluster(1, 3), 1, torch.float64, device)
        d.evaluate()
        with pytest.raises(ValueError, match="energy must be float64"):
            lbfgs_step_coord(
                d.positions,
                d.state,
                d.forces,
                d.energy.float(),
                d.batch_idx,
                d.n_particles,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_history_size_mismatch(self, device):
        d = TorchDriver(_cluster(1, 3), 1, torch.float64, device)
        d.evaluate()
        other = lbfgs_allocate_state(9, 1, dtype=torch.float64, device=device)
        with pytest.raises(ValueError, match="history buffers"):
            lbfgs_step_coord(
                d.positions,
                other,
                d.forces,
                d.energy,
                d.batch_idx,
                d.n_particles,
            )


class TestLBFGSTorchCoordCell:
    """The variable-cell binding."""

    @staticmethod
    def _setup(device, num_atoms=6, seed=11):
        """A compressed cell with perturbed fractional coordinates."""
        from nvalchemiops.torch.lbfgs import (
            lbfgs_allocate_cell_state,
            lbfgs_set_reference_cell,
        )

        from ...conftest import CellPotential

        rng = np.random.default_rng(seed)
        s0 = np.array([0.1, -0.2, 0.05])
        potential = CellPotential(s0)
        cell_np = np.diag([6.0, 6.5, 7.0])
        frac = s0 + rng.normal(size=(num_atoms, 3)) * 0.05
        positions_np = np.ascontiguousarray((cell_np @ frac.T).T)

        positions = torch.tensor(positions_np, dtype=torch.float64, device=device)
        cell = torch.tensor(cell_np[None], dtype=torch.float64, device=device)
        cell_state = lbfgs_allocate_cell_state(
            num_atoms, 1, dtype=torch.float64, device=device
        )
        lbfgs_set_reference_cell(cell, cell_state)
        state = lbfgs_allocate_state(
            num_atoms + 2, 1, dtype=torch.float64, device=device
        )
        return positions, cell, state, cell_state, potential

    @staticmethod
    def _evaluate(positions, cell, potential, device):
        e, f, s = potential.energy_forces_stress(
            positions.cpu().numpy(), cell.cpu().numpy()[0]
        )
        return (
            torch.tensor(f, dtype=torch.float64, device=device),
            torch.tensor(s[None], dtype=torch.float64, device=device),
            torch.tensor([e], dtype=torch.float64, device=device),
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_relaxes_cell_and_coordinates(self, device):
        """A compressed cell expands to the target volume while atoms relax."""
        from nvalchemiops.torch.lbfgs import lbfgs_step_coord_cell

        n = 6
        positions, cell, state, cell_state, potential = self._setup(device, n)
        batch_idx = torch.zeros(n, dtype=torch.int32, device=device)
        n_particles = torch.full((1,), n, dtype=torch.int32, device=device)

        for _ in range(400):
            forces, stress, energy = self._evaluate(positions, cell, potential, device)
            lbfgs_step_coord_cell(
                positions,
                cell,
                state,
                cell_state,
                forces,
                stress,
                energy,
                batch_idx,
                n_particles,
                force_tol=1e-6,
                stress_tol=1e-6,
                maxstep=0.2,
            )
            torch.cuda.synchronize()
            if state.status.item() != LBFGS_NEED_EVAL:
                break

        assert state.status.item() == LBFGS_CONVERGED, state.status.item()
        volume = abs(np.linalg.det(cell.cpu().numpy()[0]))
        np.testing.assert_allclose(volume, potential.target_volume, rtol=1e-4)

    @pytest.mark.parametrize("device", DEVICES)
    def test_matches_the_warp_layer(self, device):
        """The binding is a thin adapter, so it must agree exactly."""
        import warp as wp

        from nvalchemiops.dynamics.optimizers.lbfgs import LBFGSState as WarpState
        from nvalchemiops.dynamics.optimizers.lbfgs import (
            lbfgs_reset as warp_reset,
        )
        from nvalchemiops.dynamics.optimizers.lbfgs import (
            lbfgs_set_reference_cell as warp_set_ref,
        )
        from nvalchemiops.dynamics.optimizers.lbfgs import (
            lbfgs_step_coord_cell as warp_step_cell,
        )
        from nvalchemiops.torch.lbfgs import lbfgs_step_coord_cell

        from ...conftest import make_lbfgs_cell_state, make_lbfgs_state

        n = 6
        positions, cell, state, cell_state, potential = self._setup(device, n)
        batch_idx = torch.zeros(n, dtype=torch.int32, device=device)
        n_particles = torch.full((1,), n, dtype=torch.int32, device=device)

        start_pos = positions.cpu().numpy().copy()
        start_cell = cell.cpu().numpy()[0].copy()
        wp_pos = wp.array(start_pos, dtype=wp.vec3d, device=device)
        wp_cell = wp.array(start_cell[None], dtype=wp.mat33d, device=device)
        wp_forces = wp.zeros(n, dtype=wp.vec3d, device=device)
        wp_stress = wp.zeros(1, dtype=wp.mat33d, device=device)
        wp_energy = wp.zeros(1, dtype=wp.float64, device=device)
        wp_batch = wp.zeros(n, dtype=wp.int32, device=device)
        wp_nparts = wp.array(np.array([n], np.int32), dtype=wp.int32, device=device)
        wp_cell_state = make_lbfgs_cell_state(n, 1, wp.vec3d, device)
        warp_set_ref(wp_cell, wp_cell_state.ref_cell, wp_cell_state.ref_cell_inv)
        wp_state_dict = make_lbfgs_state(n + 2, 1, 6, wp.vec3d, device)
        warp_reset(**wp_state_dict)

        for _ in range(25):
            forces, stress, energy = self._evaluate(positions, cell, potential, device)
            lbfgs_step_coord_cell(
                positions,
                cell,
                state,
                cell_state,
                forces,
                stress,
                energy,
                batch_idx,
                n_particles,
                force_tol=1e-8,
                stress_tol=1e-8,
                maxstep=0.2,
            )

            e, f, s = potential.energy_forces_stress(wp_pos.numpy(), wp_cell.numpy()[0])
            wp_forces.assign(f)
            wp_stress.assign(s[None])
            wp_energy.assign(np.array([e]))
            warp_step_cell(
                wp_pos,
                wp_forces,
                wp_cell,
                wp_stress,
                wp_energy,
                wp_batch,
                wp_nparts,
                wp_cell_state,
                WarpState(**wp_state_dict),
                force_tol=1e-8,
                stress_tol=1e-8,
                maxstep=0.2,
            )
            wp.synchronize()
            torch.cuda.synchronize()
            np.testing.assert_array_equal(positions.cpu().numpy(), wp_pos.numpy())
            np.testing.assert_array_equal(cell.cpu().numpy(), wp_cell.numpy())
            if state.status.item() != LBFGS_NEED_EVAL:
                break
        np.testing.assert_array_equal(
            state.status.cpu().numpy(), wp_state_dict["status"].numpy()
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_state_sized_for_the_wrong_dof_count_is_rejected(self, device):
        """The cell adds two degrees of freedom per system; a coordinate-sized
        state would silently under-cover the packed array."""
        from nvalchemiops.torch.lbfgs import lbfgs_step_coord_cell

        n = 6
        positions, cell, _, cell_state, potential = self._setup(device, n)
        wrong = lbfgs_allocate_state(n, 1, dtype=torch.float64, device=device)
        forces, stress, energy = self._evaluate(positions, cell, potential, device)
        with pytest.raises(ValueError, match="num_atoms \\+ 2 \\* num_systems"):
            lbfgs_step_coord_cell(
                positions,
                cell,
                wrong,
                cell_state,
                forces,
                stress,
                energy,
                torch.zeros(n, dtype=torch.int32, device=device),
                torch.full((1,), n, dtype=torch.int32, device=device),
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_cell_operator_schema_matches_the_state_containers(self, device):
        """Both containers must line up with the registered operator, in order."""
        from nvalchemiops.dynamics.optimizers.lbfgs import LBFGSCellState
        from nvalchemiops.torch.lbfgs import _lbfgs_step_coord_cell_op

        params = tuple(inspect.signature(_lbfgs_step_coord_cell_op).parameters)
        offset = 7  # forces, stress, energy, batch_idx, n_particles, positions, cell
        n_state = len(LBFGSState._fields)
        assert params[offset : offset + n_state] == LBFGSState._fields
        assert (
            params[offset + n_state : offset + n_state + len(LBFGSCellState._fields)]
            == LBFGSCellState._fields
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_compiles_fullgraph_and_matches_eager(self, device):
        """The variable-cell step compiles as one graph and agrees with eager.

        Each run gets its own freshly allocated state, since the step mutates
        it; comparing against a run whose state had already advanced would be
        meaningless.
        """
        from nvalchemiops.torch.lbfgs import lbfgs_step_coord_cell

        n = 6
        opts = dict(force_tol=1e-8, stress_tol=1e-8, maxstep=0.2)
        batch_idx = torch.zeros(n, dtype=torch.int32, device=device)
        n_particles = torch.full((1,), n, dtype=torch.int32, device=device)

        def one_run(compiled):
            positions, cell, state, cell_state, potential = self._setup(device, n)
            forces, stress, energy = self._evaluate(positions, cell, potential, device)

            def body(p, c, f, s, e):
                lbfgs_step_coord_cell(
                    p,
                    c,
                    state,
                    cell_state,
                    f,
                    s,
                    e,
                    batch_idx,
                    n_particles,
                    **opts,
                )

            torch._dynamo.reset()
            fn = torch.compile(body, fullgraph=True) if compiled else body
            fn(positions, cell, forces, stress, energy)
            torch.cuda.synchronize()
            return positions.clone(), cell.clone(), state.status.clone()

        eager = one_run(False)
        compiled = one_run(True)
        torch.testing.assert_close(compiled[0], eager[0], rtol=0, atol=0)
        torch.testing.assert_close(compiled[1], eager[1], rtol=0, atol=0)
        torch.testing.assert_close(compiled[2], eager[2])
