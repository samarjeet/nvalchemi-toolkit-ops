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

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from nvalchemiops.jax import lbfgs  # noqa: E402


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
