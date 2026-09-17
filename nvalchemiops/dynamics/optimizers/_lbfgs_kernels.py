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

"""Private Warp kernels for fixed-radius L-BFGS updates."""

from __future__ import annotations

from typing import Any

import warp as wp


@wp.func
def _dot64(left: Any, right: Any) -> wp.float64:
    return (
        wp.float64(left[0]) * wp.float64(right[0])
        + wp.float64(left[1]) * wp.float64(right[1])
        + wp.float64(left[2]) * wp.float64(right[2])
    )


@wp.func
def _length64(value: Any) -> wp.float64:
    return wp.sqrt(_dot64(value, value))


@wp.func
def _distance64(left: Any, right: Any) -> wp.float64:
    dx = wp.float64(left[0]) - wp.float64(right[0])
    dy = wp.float64(left[1]) - wp.float64(right[1])
    dz = wp.float64(left[2]) - wp.float64(right[2])
    return wp.sqrt(dx * dx + dy * dy + dz * dz)


@wp.func
def _slot(end: wp.int32, back: wp.int32, size: wp.int32) -> wp.int32:
    return ((end - wp.int32(1) - back) % size + size) % size


@wp.func
def _alpha_cap(
    linear: wp.float64, quadratic: wp.float64, maximum: wp.float64
) -> wp.float64:
    if linear > wp.float64(0.0):
        discriminant = linear * linear + wp.float64(4.0) * quadratic * maximum
        return (wp.float64(2.0) * maximum) / (linear + wp.sqrt(discriminant))
    if quadratic > wp.float64(0.0):
        return wp.sqrt(maximum / quadratic)
    return wp.float64(1.0)


@wp.kernel(enable_backward=False)
def lbfgs_step_kernel(
    positions: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    x_base: wp.array(dtype=Any),
    force_base: wp.array(dtype=Any),
    direction: wp.array(dtype=Any),
    s_history: wp.array(dtype=Any, ndim=2),
    y_history: wp.array(dtype=Any, ndim=2),
    ys: wp.array(dtype=wp.float64, ndim=2),
    yy: wp.array(dtype=wp.float64, ndim=2),
    two_loop_alpha: wp.array(dtype=wp.float64, ndim=2),
    initialized: wp.array(dtype=wp.int32),
    history_end: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    maxstep: wp.float64,
    curvature_eps: wp.float64,
    apply_cartesian: wp.bool,
):
    """Update one system's history and optionally apply its Cartesian step."""
    system = wp.tid()
    history_size = wp.int32(s_history.shape[0])

    force_sq = wp.float64(0.0)
    for atom in range(positions.shape[0]):
        if batch_idx[atom] == system:
            force_sq += _dot64(forces[atom], forces[atom])

    if initialized[system] == wp.int32(0):
        initialized[system] = wp.int32(1)
        norm = wp.sqrt(force_sq)
        largest = wp.float64(0.0)
        for atom in range(positions.shape[0]):
            if batch_idx[atom] == system:
                x_base[atom] = positions[atom]
                force_base[atom] = forces[atom]
                direction[atom] = forces[atom] - forces[atom]
                if wp.isfinite(norm) and norm > wp.float64(0.0):
                    direction[atom] = forces[atom] / type(forces[atom][0])(norm)
                    largest = wp.max(largest, _length64(direction[atom]))
        if apply_cartesian and largest > wp.float64(0.0):
            alpha = wp.min(wp.float64(1.0), maxstep / largest)
            for atom in range(positions.shape[0]):
                if batch_idx[atom] == system:
                    positions[atom] = (
                        positions[atom]
                        + type(positions[atom][0])(alpha) * direction[atom]
                    )
        return

    slot = history_end[system]
    curvature = wp.float64(0.0)
    displacement_sq = wp.float64(0.0)
    gradient_sq = wp.float64(0.0)
    for atom in range(positions.shape[0]):
        if batch_idx[atom] == system:
            displacement = positions[atom] - x_base[atom]
            gradient_change = force_base[atom] - forces[atom]
            curvature += _dot64(displacement, gradient_change)
            displacement_sq += _dot64(displacement, displacement)
            gradient_sq += _dot64(gradient_change, gradient_change)
            s_history[slot, atom] = displacement
            y_history[slot, atom] = gradient_change
            x_base[atom] = positions[atom]
            force_base[atom] = forces[atom]

    ys[slot, system] = curvature
    yy[slot, system] = gradient_sq
    usable = (
        wp.isfinite(curvature)
        and wp.isfinite(displacement_sq)
        and wp.isfinite(gradient_sq)
        and gradient_sq > wp.float64(0.0)
        and curvature > curvature_eps * wp.sqrt(displacement_sq * gradient_sq)
    )
    if usable:
        history_end[system] = (slot + wp.int32(1)) % history_size
        if history_count[system] < history_size:
            history_count[system] = history_count[system] + wp.int32(1)
    else:
        history_end[system] = wp.int32(0)
        history_count[system] = wp.int32(0)
        for history in range(history_size):
            ys[history, system] = wp.float64(0.0)
            yy[history, system] = wp.float64(0.0)
            two_loop_alpha[history, system] = wp.float64(0.0)
            for atom in range(positions.shape[0]):
                if batch_idx[atom] == system:
                    s_history[history, atom] = (
                        s_history[history, atom] - s_history[history, atom]
                    )
                    y_history[history, atom] = (
                        y_history[history, atom] - y_history[history, atom]
                    )

    restart = history_count[system] == wp.int32(0)
    if not restart:
        for atom in range(positions.shape[0]):
            if batch_idx[atom] == system:
                direction[atom] = forces[atom]

        count = history_count[system]
        for back in range(history_size):
            if back < count:
                index = _slot(history_end[system], wp.int32(back), history_size)
                projection = wp.float64(0.0)
                for atom in range(positions.shape[0]):
                    if batch_idx[atom] == system:
                        projection += _dot64(s_history[index, atom], direction[atom])
                coefficient = projection / ys[index, system]
                two_loop_alpha[index, system] = coefficient
                for atom in range(positions.shape[0]):
                    if batch_idx[atom] == system:
                        direction[atom] = (
                            direction[atom]
                            - type(direction[atom][0])(coefficient)
                            * y_history[index, atom]
                        )

        newest = _slot(history_end[system], wp.int32(0), history_size)
        h0 = ys[newest, system] / yy[newest, system]
        for atom in range(positions.shape[0]):
            if batch_idx[atom] == system:
                direction[atom] = type(direction[atom][0])(h0) * direction[atom]

        for back in range(history_size):
            reverse = history_size - wp.int32(1) - wp.int32(back)
            if reverse < count:
                index = _slot(history_end[system], reverse, history_size)
                projection = wp.float64(0.0)
                for atom in range(positions.shape[0]):
                    if batch_idx[atom] == system:
                        projection += _dot64(y_history[index, atom], direction[atom])
                beta = projection / ys[index, system]
                coefficient = two_loop_alpha[index, system] - beta
                for atom in range(positions.shape[0]):
                    if batch_idx[atom] == system:
                        direction[atom] = (
                            direction[atom]
                            + type(direction[atom][0])(coefficient)
                            * s_history[index, atom]
                        )

        force_dot_direction = wp.float64(0.0)
        largest = wp.float64(0.0)
        for atom in range(positions.shape[0]):
            if batch_idx[atom] == system:
                force_dot_direction += _dot64(forces[atom], direction[atom])
                largest = wp.max(largest, _length64(direction[atom]))
        restart = (
            not wp.isfinite(force_dot_direction)
            or force_dot_direction <= wp.float64(0.0)
            or not wp.isfinite(largest)
        )

    if restart:
        history_end[system] = wp.int32(0)
        history_count[system] = wp.int32(0)
        norm = wp.sqrt(force_sq)
        for history in range(history_size):
            ys[history, system] = wp.float64(0.0)
            yy[history, system] = wp.float64(0.0)
            two_loop_alpha[history, system] = wp.float64(0.0)
            for atom in range(positions.shape[0]):
                if batch_idx[atom] == system:
                    s_history[history, atom] = (
                        s_history[history, atom] - s_history[history, atom]
                    )
                    y_history[history, atom] = (
                        y_history[history, atom] - y_history[history, atom]
                    )
        for atom in range(positions.shape[0]):
            if batch_idx[atom] == system:
                direction[atom] = forces[atom] - forces[atom]
                if wp.isfinite(norm) and norm > wp.float64(0.0):
                    direction[atom] = forces[atom] / type(forces[atom][0])(norm)

    largest = wp.float64(0.0)
    for atom in range(positions.shape[0]):
        if batch_idx[atom] == system:
            largest = wp.max(largest, _length64(direction[atom]))
    if apply_cartesian and largest > wp.float64(0.0):
        alpha = wp.min(wp.float64(1.0), maxstep / largest)
        for atom in range(positions.shape[0]):
            if batch_idx[atom] == system:
                positions[atom] = (
                    positions[atom] + type(positions[atom][0])(alpha) * direction[atom]
                )


@wp.kernel(enable_backward=False)
def prepare_reference_cell_kernel(
    cell: wp.array(dtype=Any),
    ref_cell: wp.array(dtype=Any),
    ref_cell_inv: wp.array(dtype=Any),
):
    """Capture the caller-validated reference cell and its inverse."""
    system = wp.tid()
    ref_cell[system] = cell[system]
    ref_cell_inv[system] = wp.inverse(cell[system])


@wp.kernel(enable_backward=False)
def cell_scale_kernel(
    atom_counts: wp.array(dtype=wp.int32),
    cell_force_scale: wp.float64,
    cell_scale: wp.array(dtype=Any),
):
    """Store the FIRE2-compatible cell-force divisor."""
    system = wp.tid()
    cell_scale[system] = type(cell_scale[system])(
        wp.float64(atom_counts[system]) * cell_force_scale
    )


@wp.kernel(enable_backward=False)
def pack_cell_kernel(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    cell_force: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    ref_cell: wp.array(dtype=Any),
    ref_cell_inv: wp.array(dtype=Any),
    cell_scale: wp.array(dtype=Any),
    ext_atom_ptr: wp.array(dtype=wp.int32),
    phi: wp.array(dtype=Any),
    phi_inv: wp.array(dtype=Any),
    d_phi: wp.array(dtype=Any),
    ext_positions: wp.array(dtype=Any),
    ext_forces: wp.array(dtype=Any),
):
    """Pack atomic and cell coordinates into the shared L-BFGS space."""
    system = wp.tid()
    chart = cell[system] * ref_cell_inv[system]
    inverse = wp.inverse(chart)
    phi[system] = chart
    phi_inv[system] = inverse
    d_phi[system] = chart - chart

    scale = cell_scale[system]
    cell_start = ext_atom_ptr[system + wp.int32(1)] - wp.int32(2)
    first = ext_positions[cell_start] - ext_positions[cell_start]
    second = first
    first[0] = scale * chart[0, 0]
    first[1] = scale * chart[1, 0]
    first[2] = scale * chart[2, 0]
    second[0] = scale * chart[1, 1]
    second[1] = scale * chart[2, 1]
    second[2] = scale * chart[2, 2]
    ext_positions[cell_start] = first
    ext_positions[cell_start + wp.int32(1)] = second

    conjugate = (cell_force[system] * wp.transpose(ref_cell[system])) / scale
    first_force = first - first
    second_force = first - first
    first_force[0] = conjugate[0, 0]
    first_force[1] = conjugate[1, 0]
    first_force[2] = conjugate[2, 0]
    second_force[0] = conjugate[1, 1]
    second_force[1] = conjugate[2, 1]
    second_force[2] = conjugate[2, 2]
    ext_forces[cell_start] = first_force
    ext_forces[cell_start + wp.int32(1)] = second_force

    for atom in range(positions.shape[0]):
        if batch_idx[atom] == system:
            packed = atom + wp.int32(2) * system
            ext_positions[packed] = inverse * positions[atom]
            ext_forces[packed] = wp.transpose(chart) * forces[atom]


@wp.kernel(enable_backward=False)
def apply_cell_step_kernel(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    ref_cell: wp.array(dtype=Any),
    cell_scale: wp.array(dtype=Any),
    ext_atom_ptr: wp.array(dtype=wp.int32),
    direction: wp.array(dtype=Any),
    phi: wp.array(dtype=Any),
    phi_inv: wp.array(dtype=Any),
    d_phi: wp.array(dtype=Any),
    ext_positions: wp.array(dtype=Any),
    maxstep: wp.float64,
):
    """Apply one coupled atom/cell step with a Cartesian atom cap."""
    system = wp.tid()
    chart = phi[system]
    scale = cell_scale[system]
    cell_start = ext_atom_ptr[system + wp.int32(1)] - wp.int32(2)
    first = direction[cell_start] / scale
    second = direction[cell_start + wp.int32(1)] / scale
    delta = chart - chart
    delta[0, 0] = first[0]
    delta[1, 0] = first[1]
    delta[2, 0] = first[2]
    delta[1, 1] = second[0]
    delta[2, 1] = second[1]
    delta[2, 2] = second[2]

    direction_is_zero = wp.int32(1)
    if (
        first[0] != type(first[0])(0.0)
        or first[1] != type(first[1])(0.0)
        or first[2] != type(first[2])(0.0)
        or second[0] != type(second[0])(0.0)
        or second[1] != type(second[1])(0.0)
        or second[2] != type(second[2])(0.0)
    ):
        direction_is_zero = wp.int32(0)

    linear_max = wp.float64(0.0)
    quadratic_max = wp.float64(0.0)
    for atom in range(positions.shape[0]):
        if batch_idx[atom] == system:
            packed = atom + wp.int32(2) * system
            base_coordinate = ext_positions[packed]
            coordinate_direction = direction[packed]
            if (
                coordinate_direction[0] != type(coordinate_direction[0])(0.0)
                or coordinate_direction[1] != type(coordinate_direction[1])(0.0)
                or coordinate_direction[2] != type(coordinate_direction[2])(0.0)
            ):
                direction_is_zero = wp.int32(0)
            linear = chart * coordinate_direction + delta * base_coordinate
            quadratic = delta * coordinate_direction
            linear_max = wp.max(linear_max, _length64(linear))
            quadratic_max = wp.max(quadratic_max, _length64(quadratic))

    # Preserve caller geometry for a genuinely zero packed direction.  This
    # cannot be inferred from Cartesian coefficients: a cell direction can be
    # nonzero while all current atoms lie at the chart origin.
    if direction_is_zero == wp.int32(1):
        return

    alpha = wp.min(wp.float64(1.0), _alpha_cap(linear_max, quadratic_max, maxstep))
    # The analytic cap bounds ideal arithmetic.  Check the coordinate-dtype
    # candidate against actual caller coordinates before committing any state.
    # A bounded reduction avoids nontermination when no nonzero float proposal
    # is representable within a very small physical cap.
    alpha_is_safe = wp.int32(0)
    for _ in range(32):
        if alpha_is_safe == wp.int32(0):
            rounded_max = wp.float64(0.0)
            for atom in range(positions.shape[0]):
                if batch_idx[atom] == system:
                    packed = atom + wp.int32(2) * system
                    base_coordinate = ext_positions[packed]
                    coordinate_direction = direction[packed]
                    linear = chart * coordinate_direction + delta * base_coordinate
                    quadratic = delta * coordinate_direction
                    coordinate_alpha = type(positions[atom][0])(alpha)
                    candidate_position = (
                        positions[atom]
                        + coordinate_alpha * linear
                        + (coordinate_alpha * coordinate_alpha * quadratic)
                    )
                    rounded_max = wp.max(
                        rounded_max, _distance64(candidate_position, positions[atom])
                    )
            if rounded_max <= maxstep:
                alpha_is_safe = wp.int32(1)
            else:
                alpha = alpha * maxstep / rounded_max
    if alpha_is_safe == wp.int32(0):
        alpha = wp.float64(0.0)

    d_phi[system] = delta
    candidate_chart = chart + type(chart[0, 0])(alpha) * delta
    for atom in range(positions.shape[0]):
        if batch_idx[atom] == system:
            packed = atom + wp.int32(2) * system
            base_coordinate = ext_positions[packed]
            coordinate_direction = direction[packed]
            linear = chart * coordinate_direction + delta * base_coordinate
            quadratic = delta * coordinate_direction
            coordinate_alpha = type(positions[atom][0])(alpha)
            packed_candidate = base_coordinate + coordinate_alpha * coordinate_direction
            ext_positions[packed] = packed_candidate
            positions[atom] = (
                positions[atom]
                + coordinate_alpha * linear
                + (coordinate_alpha * coordinate_alpha * quadratic)
            )

    first_candidate = first - first
    second_candidate = first - first
    first_candidate[0] = scale * candidate_chart[0, 0]
    first_candidate[1] = scale * candidate_chart[1, 0]
    first_candidate[2] = scale * candidate_chart[2, 0]
    second_candidate[0] = scale * candidate_chart[1, 1]
    second_candidate[1] = scale * candidate_chart[2, 1]
    second_candidate[2] = scale * candidate_chart[2, 2]
    ext_positions[cell_start] = first_candidate
    ext_positions[cell_start + wp.int32(1)] = second_candidate
    cell[system] = candidate_chart * ref_cell[system]
    phi[system] = candidate_chart
    phi_inv[system] = wp.inverse(candidate_chart)
