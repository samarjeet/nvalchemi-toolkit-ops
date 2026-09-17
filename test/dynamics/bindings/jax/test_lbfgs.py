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

"""JAX state-object and x64 guard tests for L-BFGS."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from nvalchemiops.jax import lbfgs  # noqa: E402

from .conftest import requires_gpu  # noqa: E402


def _state_leaves(state):
    """Return all public optimizer arrays through the pytree contract."""
    return jax.tree_util.tree_leaves(state)


def test_public_surface_and_pytree_state():
    """Prepared state participates in JAX tree reconstruction."""
    if not jax.config.jax_enable_x64:
        pytest.skip("the current JAX process has x64 disabled")
    positions = jnp.zeros((2, 3), jnp.float32)
    state = lbfgs.prepare_lbfgs_state(positions, 1)
    leaves, rebuilt = jax.tree_util.tree_flatten(state), None
    rebuilt = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(state), leaves[0]
    )
    assert isinstance(rebuilt, lbfgs.LBFGSState)
    assert lbfgs.__all__ == [
        "LBFGSState",
        "LBFGSCellState",
        "prepare_lbfgs_state",
        "prepare_lbfgs_cell_state",
        "lbfgs_step_coord",
        "lbfgs_step_coord_cell",
    ]


def test_functional_coordinate_step_reconstructs_state():
    """The JAX entry point returns updated coordinates and a new state object."""
    if not jax.config.jax_enable_x64:
        pytest.skip("the current JAX process has x64 disabled")
    positions = jnp.array([[1.0, 0.0, 0.0]], jnp.float32)
    state = lbfgs.prepare_lbfgs_state(positions, 1)
    updated, next_state = lbfgs.lbfgs_step_coord(
        positions, -positions, jnp.zeros((1,), jnp.int32), state
    )
    assert isinstance(next_state, lbfgs.LBFGSState)
    assert float(updated[0, 0]) < float(positions[0, 0])
    assert int(next_state.initialized[0]) == 1


def test_functional_cell_step_reconstructs_cell_state():
    """The JAX cell entry point returns the mutable scratch through state."""
    if not jax.config.jax_enable_x64:
        pytest.skip("the current JAX process has x64 disabled")
    positions = jnp.array([[1.0, 0.0, 0.0]], jnp.float32)
    cell = jnp.eye(3, dtype=jnp.float32)[None]
    batch_idx = jnp.zeros((1,), jnp.int32)
    state = lbfgs.prepare_lbfgs_cell_state(positions, cell, batch_idx)
    updated_positions, updated_cell, next_state = lbfgs.lbfgs_step_coord_cell(
        positions, -positions, cell, jnp.zeros_like(cell), batch_idx, state
    )
    assert isinstance(next_state, lbfgs.LBFGSCellState)
    assert float(updated_positions[0, 0]) < float(positions[0, 0])
    assert updated_cell.shape == cell.shape


@pytest.mark.parametrize("labels", [[1, 0], [0, 2], [0, 0]])
def test_jax_cell_preparation_rejects_invalid_batch_topology(labels):
    """JAX cell preparation enforces the same topology as Core."""
    if not jax.config.jax_enable_x64:
        pytest.skip("the current JAX process has x64 disabled")
    positions = jnp.zeros((2, 3), jnp.float32)
    cell = jnp.zeros((2, 3, 3), jnp.float32)
    batch_idx = jnp.asarray(labels, jnp.int32)
    with pytest.raises(ValueError, match="batch_idx"):
        lbfgs.prepare_lbfgs_cell_state(positions, cell, batch_idx)


def test_x64_disabled_is_rejected_before_state_creation():
    """The required float64 reduction mode fails before dtype normalization."""
    script = """
import jax.numpy as jnp
from nvalchemiops.jax.lbfgs import prepare_lbfgs_state
try:
    prepare_lbfgs_state(jnp.zeros((1, 3), jnp.float32), 1)
except ValueError as error:
    assert 'jax_enable_x64=True' in str(error)
else:
    raise AssertionError('x64-disabled state preparation succeeded')
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ | {"JAX_ENABLE_X64": "False", "JAX_PLATFORMS": "cpu"},
    )  # noqa: S603
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("_gpu", [pytest.param(None, marks=requires_gpu)])
def test_jit_donation_replays_and_supports_both_coordinate_dtypes(dtype, _gpu):
    """JIT donation preserves the functional state across repeated replays."""
    initial_values = np.array([[1.5, -0.2, 0.4], [-0.7, 1.1, -0.3]])
    initial = jnp.array(initial_values, dtype)
    batch_idx = jnp.zeros((2,), dtype=jnp.int32)

    def step(positions, forces, batch, state):
        return lbfgs.lbfgs_step_coord(positions, forces, batch, state, maxstep=0.2)

    donated_step = jax.jit(step, donate_argnums=(0, 3))
    state = lbfgs.prepare_lbfgs_state(initial, 1, history_size=2)
    positions = initial
    for _ in range(2):
        positions, state = donated_step(positions, -positions, batch_idx, state)
        jax.block_until_ready(positions)

    eager_positions = jnp.array(initial_values, dtype)
    eager_state = lbfgs.prepare_lbfgs_state(eager_positions, 1, history_size=2)
    for _ in range(2):
        eager_positions, eager_state = step(
            eager_positions, -eager_positions, batch_idx, eager_state
        )
        jax.block_until_ready(eager_positions)

    np.testing.assert_array_equal(np.asarray(positions), np.asarray(eager_positions))
    for actual, expected in zip(
        _state_leaves(state), _state_leaves(eager_state), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("_gpu", [pytest.param(None, marks=requires_gpu)])
def test_cell_step_returns_the_complete_updated_state_including_scratch(_gpu):
    """Functional cell stepping returns all mutable packing and chart scratch."""
    positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], jnp.float32)
    cell = jnp.eye(3, dtype=jnp.float32)[None] * 3
    forces = jnp.zeros_like(positions)
    cell_force = jnp.eye(3, dtype=jnp.float32)[None] * 20
    batch_idx = jnp.zeros((2,), dtype=jnp.int32)
    state = lbfgs.prepare_lbfgs_cell_state(
        positions, cell, batch_idx, cell_force_scale=0.5
    )
    updated_positions, updated_cell, updated_state = lbfgs.lbfgs_step_coord_cell(
        positions, forces, cell, cell_force, batch_idx, state, maxstep=0.2
    )
    jax.block_until_ready(updated_positions)
    assert updated_positions.shape == positions.shape
    assert updated_cell.shape == cell.shape
    assert isinstance(updated_state, lbfgs.LBFGSCellState)
    assert len(_state_leaves(updated_state)) == 21
    for name in ("phi", "phi_inv", "d_phi", "ext_positions", "ext_forces"):
        actual = np.asarray(getattr(updated_state, name))
        before = np.asarray(getattr(state, name))
        assert actual.shape == before.shape
        assert not np.array_equal(actual, before), f"{name} was not returned updated"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jit_cell_donation_replays_complete_state_without_aliasing_cell(dtype):
    """Donated cell geometry and complete state replay like an eager path."""
    initial_values = np.array([[1.1, -0.7, 0.3], [-0.4, 1.6, -0.8]])
    positions = jnp.asarray(initial_values, dtype)
    cell = jnp.asarray([[[2.3, 0.0, 0.0], [0.4, 1.7, 0.0], [-0.2, 0.3, 2.1]]], dtype)
    batch_idx = jnp.zeros((2,), dtype=jnp.int32)

    def step(current_positions, current_cell, forces, cell_force, batch, state):
        return lbfgs.lbfgs_step_coord_cell(
            current_positions,
            forces,
            current_cell,
            cell_force,
            batch,
            state,
            maxstep=0.2,
        )

    donated_step = jax.jit(step, donate_argnums=(0, 1, 5))
    state = lbfgs.prepare_lbfgs_cell_state(positions, cell, batch_idx, history_size=2)
    for _ in range(2):
        positions, cell, state = donated_step(
            positions,
            cell,
            -positions,
            jnp.zeros_like(cell),
            batch_idx,
            state,
        )
        jax.block_until_ready(positions)

    eager_positions = jnp.asarray(initial_values, dtype)
    eager_cell = jnp.asarray(
        [[[2.3, 0.0, 0.0], [0.4, 1.7, 0.0], [-0.2, 0.3, 2.1]]], dtype
    )
    eager_state = lbfgs.prepare_lbfgs_cell_state(
        eager_positions, eager_cell, batch_idx, history_size=2
    )
    for _ in range(2):
        eager_positions, eager_cell, eager_state = step(
            eager_positions,
            eager_cell,
            -eager_positions,
            jnp.zeros_like(eager_cell),
            batch_idx,
            eager_state,
        )
        jax.block_until_ready(eager_positions)

    np.testing.assert_array_equal(np.asarray(positions), np.asarray(eager_positions))
    np.testing.assert_array_equal(np.asarray(cell), np.asarray(eager_cell))
    for actual, expected in zip(
        _state_leaves(state), _state_leaves(eager_state), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
