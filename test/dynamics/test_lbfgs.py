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

from dataclasses import fields

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


def _state_fields(state):
    """Return state arrays through public dataclass fields."""
    return tuple(getattr(state, field.name) for field in fields(state))


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
    for array in _state_fields(state):
        np.testing.assert_array_equal(array.numpy(), 0)


def test_first_step_normalizes_force_and_caps_each_atom(dtypes):
    """The first lifecycle step is a normalized Cartesian-force restart."""
    vector_dtype, _, numpy_dtype = dtypes
    positions, forces, batch_idx, state = _coordinate_problem(vector_dtype, numpy_dtype)
    before = positions.numpy().copy()
    forces.assign(np.array([[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], numpy_dtype))
    lbfgs.lbfgs_step_coord(positions, forces, batch_idx, state, maxstep=0.2)
    displacements = np.linalg.norm(positions.numpy() - before, axis=1)
    assert np.all(displacements <= 0.2 * (1.0 + 3e-6))
    np.testing.assert_allclose(displacements.max(), 0.2, rtol=3e-6)
    assert displacements.min() < displacements.max()
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


def _history_slots(history_end, history_count, history_size):
    """Return committed ring slots from newest to oldest."""
    return [
        (int(history_end) - 1 - back) % history_size
        for back in range(int(history_count))
    ]


def _numpy_two_loop(s_vecs, y_vecs, ys, yy, q):
    """Apply the public L-BFGS history with an independent NumPy recursion."""
    alpha = []
    q = np.array(q, copy=True)
    for s, y, sy in zip(s_vecs, y_vecs, ys, strict=True):
        coefficient = np.sum(s * q) / sy
        alpha.append(coefficient)
        q = q - coefficient * y
    result = (ys[0] / yy[0]) * q
    for s, y, sy, coefficient in zip(
        reversed(s_vecs), reversed(y_vecs), reversed(ys), reversed(alpha), strict=True
    ):
        beta = np.sum(y * result) / sy
        result = result + s * (coefficient - beta)
    return result


@pytest.mark.parametrize(
    "vector_dtype,numpy_dtype",
    [(wp.vec3f, np.float32), (wp.vec3d, np.float64)],
)
def test_ragged_history_ring_wraps_without_cross_system_state(
    vector_dtype, numpy_dtype
):
    """Ragged systems retain the right pairs when their shared ring wraps."""
    positions_np = np.array(
        [[2.0, 0.5, -0.2], [-1.0, 0.2, 0.3], [0.8, -1.5, 0.4], [1.2, 0.7, -0.9]],
        dtype=numpy_dtype,
    )
    stiffness = np.array(
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [1.5, 2.5, 3.5], [3.0, 1.0, 2.0]],
        dtype=numpy_dtype,
    )
    positions = wp.array(positions_np, dtype=vector_dtype, device="cpu")
    forces = wp.zeros(4, dtype=vector_dtype, device="cpu")
    batch_idx = wp.array([0, 0, 1, 1], dtype=wp.int32, device="cpu")
    state = lbfgs.prepare_lbfgs_state(positions, 2, history_size=2)
    expected = [[], []]

    for _ in range(8):
        positions_before = positions.numpy().copy()
        force_base_before = state.force_base.numpy().copy()
        x_base_before = state.x_base.numpy().copy()
        initialized_before = state.initialized.numpy().copy()
        end_before = state.history_end.numpy().copy()
        forces_np = (-stiffness * positions_before).astype(numpy_dtype)
        forces.assign(forces_np)
        lbfgs.lbfgs_step_coord(positions, forces, batch_idx, state, maxstep=0.15)

        for system, mask in enumerate((slice(0, 2), slice(2, 4))):
            if initialized_before[system]:
                slot = int(end_before[system])
                expected[system].append(
                    (
                        positions_before[mask] - x_base_before[mask],
                        force_base_before[mask] - forces_np[mask],
                    )
                )
                expected[system] = expected[system][-2:]
                slots = _history_slots(
                    state.history_end.numpy()[system],
                    state.history_count.numpy()[system],
                    2,
                )
                assert slot in slots
            else:
                slots = []
            assert int(state.history_count.numpy()[system]) == len(expected[system])
            for expected_pair, slot in zip(expected[system][::-1], slots, strict=True):
                expected_s, expected_y = expected_pair
                np.testing.assert_allclose(
                    state.s_history.numpy()[slot, mask], expected_s, rtol=3e-5
                )
                np.testing.assert_allclose(
                    state.y_history.numpy()[slot, mask], expected_y, rtol=3e-5
                )

    assert np.any(state.history_end.numpy() != 0), "the ring did not wrap"


@pytest.mark.parametrize(
    "vector_dtype,numpy_dtype,tolerance",
    [(wp.vec3f, np.float32, 4e-5), (wp.vec3d, np.float64, 1e-12)],
)
def test_direction_matches_independent_numpy_two_loop(
    vector_dtype, numpy_dtype, tolerance
):
    """The direct Core direction agrees with a NumPy two-loop reference."""
    positions = wp.array(
        np.array([[2.0, 0.5, -0.2], [-1.0, 0.2, 0.3]], dtype=numpy_dtype),
        dtype=vector_dtype,
        device="cpu",
    )
    forces = wp.zeros(2, dtype=vector_dtype, device="cpu")
    batch_idx = wp.zeros(2, dtype=wp.int32, device="cpu")
    state = lbfgs.prepare_lbfgs_state(positions, 1, history_size=3)
    stiffness = np.array([1.0, 2.0, 4.0], dtype=numpy_dtype)

    for _ in range(7):
        force_np = (-stiffness * positions.numpy()).astype(numpy_dtype)
        forces.assign(force_np)
        lbfgs.lbfgs_step_coord(positions, forces, batch_idx, state, maxstep=0.15)
        count = int(state.history_count.numpy()[0])
        if count == 0:
            continue
        slots = _history_slots(state.history_end.numpy()[0], count, 3)
        mask = np.ones(2, dtype=bool)
        reference = _numpy_two_loop(
            [state.s_history.numpy()[slot][mask] for slot in slots],
            [state.y_history.numpy()[slot][mask] for slot in slots],
            [state.ys.numpy()[slot, 0] for slot in slots],
            [state.yy.numpy()[slot, 0] for slot in slots],
            force_np,
        )
        np.testing.assert_allclose(
            state.direction.numpy()[mask], reference, rtol=tolerance, atol=tolerance
        )


def test_checkpointed_state_can_restart_a_coordinate_sequence():
    """A copied public state resumes with the same observable trajectory."""
    initial = np.array([[1.7, -0.4, 0.3], [-0.6, 1.2, -0.8]], np.float64)
    positions = wp.array(initial, dtype=wp.vec3d, device="cpu")
    forces = wp.zeros(2, dtype=wp.vec3d, device="cpu")
    batch_idx = wp.zeros(2, dtype=wp.int32, device="cpu")
    state = lbfgs.prepare_lbfgs_state(positions, 1, history_size=2)
    stiffness = np.array([1.0, 2.0, 3.0])
    for _ in range(3):
        forces.assign((-stiffness * positions.numpy()).astype(np.float64))
        lbfgs.lbfgs_step_coord(positions, forces, batch_idx, state, maxstep=0.2)

    resumed_positions = wp.array(positions.numpy(), dtype=wp.vec3d, device="cpu")
    resumed_state = lbfgs.LBFGSState(
        *(
            wp.array(value.numpy(), dtype=value.dtype, device="cpu")
            for value in (
                state.x_base,
                state.force_base,
                state.direction,
                state.s_history,
                state.y_history,
                state.ys,
                state.yy,
                state.two_loop_alpha,
                state.initialized,
                state.history_end,
                state.history_count,
            )
        )
    )
    for _ in range(3):
        left_forces = (-stiffness * positions.numpy()).astype(np.float64)
        right_forces = (-stiffness * resumed_positions.numpy()).astype(np.float64)
        forces.assign(left_forces)
        resumed_forces = wp.array(right_forces, dtype=wp.vec3d, device="cpu")
        lbfgs.lbfgs_step_coord(positions, forces, batch_idx, state, maxstep=0.2)
        lbfgs.lbfgs_step_coord(
            resumed_positions,
            resumed_forces,
            batch_idx,
            resumed_state,
            maxstep=0.2,
        )
        np.testing.assert_array_equal(positions.numpy(), resumed_positions.numpy())
        for left, right in zip(
            (
                state.x_base,
                state.force_base,
                state.direction,
                state.s_history,
                state.y_history,
                state.ys,
                state.yy,
                state.two_loop_alpha,
                state.initialized,
                state.history_end,
                state.history_count,
            ),
            (
                resumed_state.x_base,
                resumed_state.force_base,
                resumed_state.direction,
                resumed_state.s_history,
                resumed_state.y_history,
                resumed_state.ys,
                resumed_state.yy,
                resumed_state.two_loop_alpha,
                resumed_state.initialized,
                resumed_state.history_end,
                resumed_state.history_count,
            ),
            strict=True,
        ):
            np.testing.assert_array_equal(left.numpy(), right.numpy())


@pytest.mark.parametrize(
    "vector_dtype,matrix_dtype,numpy_dtype",
    [(wp.vec3f, wp.mat33f, np.float32), (wp.vec3d, wp.mat33d, np.float64)],
)
def test_cell_force_scaling_and_coupled_cap(vector_dtype, matrix_dtype, numpy_dtype):
    """Raw cell forces are scaled and atom/cell coupling reaches one cap."""
    positions = wp.array(
        np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=numpy_dtype),
        dtype=vector_dtype,
        device="cpu",
    )
    forces = wp.zeros(2, dtype=vector_dtype, device="cpu")
    batch_idx = wp.zeros(2, dtype=wp.int32, device="cpu")
    cell = wp.array(
        np.array(
            [[[3.0, 0.0, 0.0], [0.4, 2.6, 0.0], [0.2, -0.3, 3.2]]],
            dtype=numpy_dtype,
        ),
        dtype=matrix_dtype,
        device="cpu",
    )
    raw_cell_force = wp.array(
        np.array([np.eye(3) * 20.0], dtype=numpy_dtype),
        dtype=matrix_dtype,
        device="cpu",
    )
    state = lbfgs.prepare_lbfgs_cell_state(
        positions, cell, batch_idx, cell_force_scale=0.5
    )
    before_positions = positions.numpy().copy()
    before_cell = cell.numpy().copy()
    lbfgs.lbfgs_step_coord_cell(
        positions,
        forces,
        cell,
        raw_cell_force,
        batch_idx,
        state,
        maxstep=0.2,
    )
    assert np.linalg.norm(cell.numpy() - before_cell) > 0.0
    displacements = np.linalg.norm(positions.numpy() - before_positions, axis=1)
    assert np.max(displacements) <= 0.2
    np.testing.assert_allclose(np.max(displacements), 0.2, rtol=3e-5, atol=3e-7)
    # The first cell block is the raw force transformed by the reference cell
    # and divided by atom_count * scale.
    np.testing.assert_allclose(state.ext_forces.numpy()[2], [60.0, 0.0, 0.0], rtol=3e-5)


@pytest.mark.parametrize(
    "vector_dtype,matrix_dtype,numpy_dtype",
    [(wp.vec3f, wp.mat33f, np.float32), (wp.vec3d, wp.mat33d, np.float64)],
)
def test_cell_zero_direction_is_a_noop_for_a_deformed_cell(
    vector_dtype, matrix_dtype, numpy_dtype
):
    """A zero coupled direction does not reconstruct or move float32 atoms."""
    positions = wp.array(
        np.array([[1.1, -0.7, 0.3], [-0.4, 1.6, -0.8]], dtype=numpy_dtype),
        dtype=vector_dtype,
        device="cpu",
    )
    cell = wp.array(
        np.array(
            [[[2.3, 0.0, 0.0], [0.4, 1.7, 0.0], [-0.2, 0.3, 2.1]]],
            dtype=numpy_dtype,
        ),
        dtype=matrix_dtype,
        device="cpu",
    )
    batch_idx = wp.zeros(2, dtype=wp.int32, device="cpu")
    state = lbfgs.prepare_lbfgs_cell_state(positions, cell, batch_idx)
    before_positions = positions.numpy().copy()
    before_cell = cell.numpy().copy()
    lbfgs.lbfgs_step_coord_cell(
        positions,
        wp.zeros(2, dtype=vector_dtype, device="cpu"),
        cell,
        wp.zeros(1, dtype=matrix_dtype, device="cpu"),
        batch_idx,
        state,
        maxstep=1.0e-9,
    )
    displacements = np.linalg.norm(positions.numpy() - before_positions, axis=1)
    assert np.all(displacements <= 1.0e-9)
    np.testing.assert_array_equal(positions.numpy(), before_positions)
    np.testing.assert_array_equal(cell.numpy(), before_cell)


@pytest.mark.parametrize(
    "position_value,maxstep,vector_dtype,matrix_dtype,numpy_dtype",
    [
        (-1.0e-9, 0.125, wp.vec3f, wp.mat33f, np.float32),
        (1.0, 1.0e-7, wp.vec3f, wp.mat33f, np.float32),
        (100000.0, 0.2, wp.vec3f, wp.mat33f, np.float32),
        (1.0e15, 0.2, wp.vec3d, wp.mat33d, np.float64),
    ],
)
def test_cell_cap_uses_realized_coordinate_displacement(
    position_value, maxstep, vector_dtype, matrix_dtype, numpy_dtype
):
    """Rounded coordinate updates never exceed the requested Cartesian cap."""
    positions = wp.array(
        np.array([[position_value, 0.0, 0.0]], dtype=numpy_dtype),
        dtype=vector_dtype,
        device="cpu",
    )
    cell = wp.array(
        np.eye(3, dtype=numpy_dtype)[None], dtype=matrix_dtype, device="cpu"
    )
    batch_idx = wp.zeros(1, dtype=wp.int32, device="cpu")
    state = lbfgs.prepare_lbfgs_cell_state(positions, cell, batch_idx)
    before = positions.numpy().copy()
    lbfgs.lbfgs_step_coord_cell(
        positions,
        wp.array([[1.0, 0.0, 0.0]], dtype=vector_dtype, device="cpu"),
        cell,
        wp.zeros(1, dtype=matrix_dtype, device="cpu"),
        batch_idx,
        state,
        maxstep=maxstep,
    )
    displacement = positions.numpy().astype(np.float64) - before.astype(np.float64)
    assert np.linalg.norm(displacement) <= maxstep


@pytest.mark.parametrize(
    "vector_dtype,matrix_dtype,numpy_dtype",
    [(wp.vec3f, wp.mat33f, np.float32), (wp.vec3d, wp.mat33d, np.float64)],
)
def test_cell_direction_moves_cell_when_atoms_are_at_the_origin(
    vector_dtype, matrix_dtype, numpy_dtype
):
    """A cell-only packed direction is not mistaken for a zero step."""
    positions = wp.zeros(1, dtype=vector_dtype, device="cpu")
    cell = wp.array(
        np.eye(3, dtype=numpy_dtype)[None] * 3,
        dtype=matrix_dtype,
        device="cpu",
    )
    batch_idx = wp.zeros(1, dtype=wp.int32, device="cpu")
    state = lbfgs.prepare_lbfgs_cell_state(positions, cell, batch_idx)
    before_positions = positions.numpy().copy()
    before_cell = cell.numpy().copy()
    lbfgs.lbfgs_step_coord_cell(
        positions,
        wp.zeros(1, dtype=vector_dtype, device="cpu"),
        cell,
        wp.array(np.eye(3, dtype=numpy_dtype)[None], dtype=matrix_dtype, device="cpu"),
        batch_idx,
        state,
        maxstep=0.2,
    )
    assert np.linalg.norm(cell.numpy() - before_cell) > 0.0
    assert np.linalg.norm(positions.numpy() - before_positions) <= 0.2
