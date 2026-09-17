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

"""CPU contract tests for the one-step L-BFGS optimizer."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.optimizers import lbfgs


@pytest.fixture(
    params=[(wp.vec3f, wp.mat33f, np.float32), (wp.vec3d, wp.mat33d, np.float64)]
)
def dtypes(request):
    """Provide coordinate and cell dtypes for CPU coverage."""
    return request.param


def _coordinate_problem(
    vector_dtype, numpy_dtype, *, systems: int = 1, history_size: int = 3
):
    positions = wp.array(
        np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], numpy_dtype),
        dtype=vector_dtype,
        device="cpu",
    )
    forces = wp.zeros(2, dtype=vector_dtype, device="cpu")
    batch_idx = wp.array(
        np.array([0, systems - 1], np.int32), dtype=wp.int32, device="cpu"
    )
    return (
        positions,
        forces,
        batch_idx,
        lbfgs.prepare_lbfgs_state(positions, systems, history_size=history_size),
    )


def test_public_surface_is_the_approved_state_api():
    """Only state-object preparation and one-step functions are exported."""
    assert lbfgs.__all__ == [
        "LBFGSState",
        "LBFGSCellState",
        "prepare_lbfgs_state",
        "prepare_lbfgs_cell_state",
        "lbfgs_step_coord",
        "lbfgs_step_coord_cell",
    ]


def test_prepare_state_uses_canonical_shapes_and_initialization(dtypes):
    """Preparation creates caller-owned vectors, coefficients, and controls."""
    vector_dtype, _, numpy_dtype = dtypes
    positions, _, _, state = _coordinate_problem(vector_dtype, numpy_dtype, systems=2)
    assert state.x_base.shape == positions.shape
    assert state.s_history.shape == (3, positions.shape[0])
    assert state.ys.shape == (3, 2)
    assert state.ys.dtype == wp.float64
    assert state.initialized.dtype == wp.int32
    for array in state._buffers():
        np.testing.assert_array_equal(array.numpy(), 0)


def test_first_step_normalizes_force_and_caps_each_atom(dtypes):
    """The first lifecycle step is a normalized Cartesian-force restart."""
    vector_dtype, _, numpy_dtype = dtypes
    positions, forces, batch_idx, state = _coordinate_problem(vector_dtype, numpy_dtype)
    before = positions.numpy().copy()
    forces.assign(np.array([[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], numpy_dtype))
    lbfgs.lbfgs_step_coord(positions, forces, batch_idx, state, maxstep=0.2)
    np.testing.assert_allclose(
        np.linalg.norm(positions.numpy() - before, axis=1).max(), 0.2, rtol=3e-6
    )
    np.testing.assert_allclose(state.direction.numpy(), forces.numpy() / 5.0, rtol=3e-6)
    np.testing.assert_array_equal(state.initialized.numpy(), 1)


def test_positive_curvature_uses_newest_h0(dtypes):
    """A usable pair supplies the newest ``ys / yy`` two-loop scale."""
    vector_dtype, _, numpy_dtype = dtypes
    positions, forces, batch_idx, state = _coordinate_problem(
        vector_dtype, numpy_dtype, history_size=2
    )
    for _ in range(2):
        forces.assign((-2.0 * positions.numpy()).astype(numpy_dtype))
        lbfgs.lbfgs_step_coord(positions, forces, batch_idx, state)
    np.testing.assert_allclose(
        state.direction.numpy()[:, 0], [-0.8, 0.8], rtol=4e-6, atol=2e-7
    )
    np.testing.assert_array_equal(state.history_count.numpy(), [1])


def test_unusable_curvature_clears_history_and_restarts(dtypes):
    """A zero-curvature pair resets all history before force normalization."""
    vector_dtype, _, numpy_dtype = dtypes
    positions, forces, batch_idx, state = _coordinate_problem(vector_dtype, numpy_dtype)
    forces.assign(np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], numpy_dtype))
    lbfgs.lbfgs_step_coord(positions, forces, batch_idx, state)
    for array in (
        state.s_history,
        state.y_history,
        state.ys,
        state.yy,
        state.two_loop_alpha,
    ):
        array.fill_(3.0)
    state.history_count.fill_(2)
    state.history_end.fill_(2)
    lbfgs.lbfgs_step_coord(positions, forces, batch_idx, state)
    np.testing.assert_array_equal(state.history_count.numpy(), 0)
    np.testing.assert_array_equal(state.history_end.numpy(), 0)
    for array in (
        state.s_history,
        state.y_history,
        state.ys,
        state.yy,
        state.two_loop_alpha,
    ):
        np.testing.assert_array_equal(array.numpy(), 0)


def test_zero_force_records_base_without_motion(dtypes):
    """Zero force has a finite zero direction and leaves coordinates unchanged."""
    vector_dtype, _, numpy_dtype = dtypes
    positions, forces, batch_idx, state = _coordinate_problem(vector_dtype, numpy_dtype)
    before = positions.numpy().copy()
    lbfgs.lbfgs_step_coord(positions, forces, batch_idx, state)
    np.testing.assert_array_equal(positions.numpy(), before)
    np.testing.assert_array_equal(state.x_base.numpy(), before)
    np.testing.assert_array_equal(state.direction.numpy(), 0)


def test_ragged_batch_steps_each_system(dtypes):
    """Batch membership, not a uniform atom count, defines reductions."""
    vector_dtype, _, numpy_dtype = dtypes
    positions = wp.array(
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], numpy_dtype),
        dtype=vector_dtype,
        device="cpu",
    )
    forces = wp.array(
        np.array([[-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], numpy_dtype),
        dtype=vector_dtype,
        device="cpu",
    )
    batch_idx = wp.array(np.array([0, 0, 1], np.int32), dtype=wp.int32, device="cpu")
    state = lbfgs.prepare_lbfgs_state(positions, 2)
    lbfgs.lbfgs_step_coord(positions, forces, batch_idx, state, maxstep=0.2)
    np.testing.assert_array_equal(state.initialized.numpy(), [1, 1])
    np.testing.assert_allclose(
        np.linalg.norm(
            positions.numpy()
            - np.array(
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], numpy_dtype
            ),
            axis=1,
        ).max(),
        0.2,
        rtol=3e-6,
    )


def test_cell_state_scales_raw_cell_force_and_caps_physical_motion(dtypes):
    """Cell packing divides raw cell force by atom count and applies a physical cap."""
    vector_dtype, matrix_dtype, numpy_dtype = dtypes
    positions, forces, batch_idx, _ = _coordinate_problem(vector_dtype, numpy_dtype)
    cell = wp.array(
        np.array([np.eye(3) * 3.0], numpy_dtype), dtype=matrix_dtype, device="cpu"
    )
    cell_force = wp.array(
        np.array([np.eye(3) * 6.0], numpy_dtype), dtype=matrix_dtype, device="cpu"
    )
    state = lbfgs.prepare_lbfgs_cell_state(
        positions, cell, batch_idx, cell_force_scale=0.5
    )
    np.testing.assert_allclose(state.cell_scale.numpy(), [1.0], rtol=3e-6)
    before = positions.numpy().copy()
    forces.assign(np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], numpy_dtype))
    lbfgs.lbfgs_step_coord_cell(
        positions, forces, cell, cell_force, batch_idx, state, maxstep=0.2
    )
    assert np.linalg.norm(positions.numpy() - before, axis=1).max() <= 0.2 * (
        1.0 + 3e-6
    )


@pytest.mark.parametrize(
    "labels",
    [
        np.array([1, 0], np.int32),
        np.array([0, 2], np.int32),
        np.array([0, 0], np.int32),
    ],
)
def test_cell_preparation_rejects_invalid_batch_topology(labels):
    """Cell state requires sorted, contiguous, nonempty system segments."""
    positions = wp.zeros(2, dtype=wp.vec3f, device="cpu")
    cell = wp.zeros(2, dtype=wp.mat33f, device="cpu")
    batch_idx = wp.array(labels, dtype=wp.int32, device="cpu")
    with pytest.raises(ValueError, match="batch_idx"):
        lbfgs.prepare_lbfgs_cell_state(positions, cell, batch_idx)


@pytest.mark.parametrize("maximum", [0.0, -0.1, float("inf")])
def test_invalid_maxstep_is_rejected(maximum):
    """Public scalar validation rejects nonpositive and nonfinite caps."""
    positions, forces, batch_idx, state = _coordinate_problem(wp.vec3f, np.float32)
    with pytest.raises(ValueError, match="maxstep"):
        lbfgs.lbfgs_step_coord(positions, forces, batch_idx, state, maxstep=maximum)
