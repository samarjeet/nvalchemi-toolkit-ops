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

"""Caller-owned, single-step fixed-radius L-BFGS optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from nvalchemiops.batch_utils import atom_ptr_to_batch_idx, batch_idx_to_atom_ptr
from nvalchemiops.dynamics.optimizers import _lbfgs_kernels as _kernels
from nvalchemiops.dynamics.utils.cell_filter import extend_atom_ptr

__all__ = [
    "LBFGSState",
    "LBFGSCellState",
    "prepare_lbfgs_state",
    "prepare_lbfgs_cell_state",
    "lbfgs_step_coord",
    "lbfgs_step_coord_cell",
]

_CURVATURE_EPS_F32 = 1.0e-6
_CURVATURE_EPS_F64 = 1.0e-10
_OPTIMIZER_BUFFERS: tuple[str, ...] = (
    "x_base",
    "force_base",
    "direction",
    "s_history",
    "y_history",
    "ys",
    "yy",
    "two_loop_alpha",
    "initialized",
    "history_end",
    "history_count",
)
_CELL_BUFFERS: tuple[str, ...] = (
    "ref_cell",
    "ref_cell_inv",
    "cell_scale",
    "ext_batch_idx",
    "ext_atom_ptr",
    "phi",
    "phi_inv",
    "d_phi",
    "ext_positions",
    "ext_forces",
)
_CELL_SCRATCH: tuple[str, ...] = _CELL_BUFFERS[5:]


@dataclass
class LBFGSState:
    """Persistent history and scratch arrays for batched L-BFGS."""

    x_base: Any
    force_base: Any
    direction: Any
    s_history: Any
    y_history: Any
    ys: Any
    yy: Any
    two_loop_alpha: Any
    initialized: Any
    history_end: Any
    history_count: Any

    def _buffers(self) -> tuple[Any, ...]:
        """Return fields in the private binding order."""
        return tuple(getattr(self, name) for name in _OPTIMIZER_BUFFERS)


@dataclass
class LBFGSCellState:
    """L-BFGS state plus variable-cell topology and packing scratch."""

    optimizer: LBFGSState
    ref_cell: Any
    ref_cell_inv: Any
    cell_scale: Any
    ext_batch_idx: Any
    ext_atom_ptr: Any
    phi: Any
    phi_inv: Any
    d_phi: Any
    ext_positions: Any
    ext_forces: Any

    def _cell_buffers(self) -> tuple[Any, ...]:
        """Return cell fields in the private binding order."""
        return tuple(getattr(self, name) for name in _CELL_BUFFERS)


def _same_device(*arrays: wp.array) -> None:
    if not arrays:
        return
    device = arrays[0].device
    if any(array.device != device for array in arrays[1:]):
        raise ValueError("all L-BFGS arrays must be on the same device")


def _validate_cell_topology(batch_idx: wp.array, num_systems: int) -> None:
    """Require one sorted, nonempty contiguous atom segment per cell."""
    labels = np.asarray(batch_idx.numpy())
    if num_systems == 0:
        if labels.size:
            raise ValueError("batch_idx must match the number of cell systems")
        return
    if labels.size == 0 or np.any(labels < 0) or np.any(labels >= num_systems):
        raise ValueError("batch_idx must contain one in-range atom for every system")
    counts = np.bincount(labels, minlength=num_systems)
    expected = np.repeat(np.arange(num_systems, dtype=np.int32), counts)
    if np.any(counts == 0) or not np.array_equal(labels, expected):
        raise ValueError("batch_idx must be sorted and contiguous by system")


def _allocate_state(
    num_dofs: int,
    num_systems: int,
    history_size: int,
    coordinate_dtype,
    device,
) -> LBFGSState:
    return LBFGSState(
        x_base=wp.zeros(num_dofs, dtype=coordinate_dtype, device=device),
        force_base=wp.zeros(num_dofs, dtype=coordinate_dtype, device=device),
        direction=wp.zeros(num_dofs, dtype=coordinate_dtype, device=device),
        s_history=wp.zeros(
            (history_size, num_dofs), dtype=coordinate_dtype, device=device
        ),
        y_history=wp.zeros(
            (history_size, num_dofs), dtype=coordinate_dtype, device=device
        ),
        ys=wp.zeros((history_size, num_systems), dtype=wp.float64, device=device),
        yy=wp.zeros((history_size, num_systems), dtype=wp.float64, device=device),
        two_loop_alpha=wp.zeros(
            (history_size, num_systems), dtype=wp.float64, device=device
        ),
        initialized=wp.zeros(num_systems, dtype=wp.int32, device=device),
        history_end=wp.zeros(num_systems, dtype=wp.int32, device=device),
        history_count=wp.zeros(num_systems, dtype=wp.int32, device=device),
    )


def prepare_lbfgs_state(
    positions: wp.array,
    num_systems: int,
    *,
    history_size: int = 6,
) -> LBFGSState:
    """Allocate L-BFGS history and scratch for a coordinate batch.

    Parameters
    ----------
    positions : wp.array
        Coordinate array whose length, dtype, and device define the degrees of
        freedom.
    num_systems : int
        Number of systems represented by the batch.
    history_size : int, default=6
        Number of curvature pairs retained per system.

    Returns
    -------
    LBFGSState
        Zero-initialized caller-owned optimizer state.
    """
    if positions.dtype not in (wp.vec3f, wp.vec3d) or positions.ndim != 1:
        raise ValueError("positions must be a rank-one wp.vec3f or wp.vec3d array")
    if not isinstance(num_systems, int) or num_systems < 0:
        raise ValueError("num_systems must be a nonnegative integer")
    if not isinstance(history_size, int) or history_size < 1:
        raise ValueError("history_size must be a positive integer")
    return _allocate_state(
        positions.shape[0],
        num_systems,
        history_size,
        positions.dtype,
        positions.device,
    )


def _validate_state(
    positions: wp.array,
    forces: wp.array,
    batch_idx: wp.array,
    state: LBFGSState,
) -> int:
    if not isinstance(state, LBFGSState):
        raise ValueError("state must be an LBFGSState")
    if positions.dtype not in (wp.vec3f, wp.vec3d) or positions.ndim != 1:
        raise ValueError("positions must be a rank-one wp.vec3f or wp.vec3d array")
    if forces.shape != positions.shape or forces.dtype != positions.dtype:
        raise ValueError("forces must match positions in shape and dtype")
    if batch_idx.dtype != wp.int32 or batch_idx.shape != positions.shape:
        raise ValueError("batch_idx must have one int32 entry per coordinate")

    buffers = state._buffers()
    x_base, force_base, direction = buffers[:3]
    for name, value in (
        ("x_base", x_base),
        ("force_base", force_base),
        ("direction", direction),
    ):
        if value.shape != positions.shape or value.dtype != positions.dtype:
            raise ValueError(f"{name} must match positions in shape and dtype")
    s_history, y_history = state.s_history, state.y_history
    if (
        s_history.ndim != 2
        or y_history.shape != s_history.shape
        or s_history.shape[1] != positions.shape[0]
        or s_history.dtype != positions.dtype
        or y_history.dtype != positions.dtype
        or s_history.shape[0] < 1
    ):
        raise ValueError("history arrays must match positions and have depth >= 1")
    history_size = s_history.shape[0]
    if state.initialized.dtype != wp.int32 or state.initialized.ndim != 1:
        raise ValueError("initialized must be a rank-one int32 array")
    num_systems = state.initialized.shape[0]
    for name, value in (
        ("ys", state.ys),
        ("yy", state.yy),
        ("two_loop_alpha", state.two_loop_alpha),
    ):
        if value.dtype != wp.float64 or value.shape != (history_size, num_systems):
            raise ValueError(
                f"{name} must be float64 with shape ({history_size}, {num_systems})"
            )
    for name, value in (
        ("initialized", state.initialized),
        ("history_end", state.history_end),
        ("history_count", state.history_count),
    ):
        if value.dtype != wp.int32 or value.shape != (num_systems,):
            raise ValueError(f"{name} must be int32 with one entry per system")
    _same_device(positions, forces, batch_idx, *buffers)
    return num_systems


def lbfgs_step_coord(
    positions: wp.array,
    forces: wp.array,
    batch_idx: wp.array,
    state: LBFGSState,
    *,
    maxstep: float = 0.2,
) -> None:
    """Update L-BFGS history and apply one capped coordinate step.

    The caller owns model evaluation, convergence, batch membership, and the
    surrounding optimization loop. ``batch_idx`` must be sorted and aligned
    with ``state``.
    """
    if not math.isfinite(maxstep) or maxstep <= 0.0:
        raise ValueError("maxstep must be finite and positive")
    num_systems = _validate_state(positions, forces, batch_idx, state)
    if num_systems == 0:
        return
    curvature_eps = (
        _CURVATURE_EPS_F32 if positions.dtype == wp.vec3f else _CURVATURE_EPS_F64
    )
    wp.launch(
        _kernels._lbfgs_step_kernel,
        dim=num_systems,
        inputs=[
            positions,
            forces,
            batch_idx,
            *state._buffers(),
            wp.float64(maxstep),
            wp.float64(curvature_eps),
            wp.bool(True),
        ],
        device=positions.device,
    )


def prepare_lbfgs_cell_state(
    positions: wp.array,
    cell: wp.array,
    batch_idx: wp.array,
    *,
    history_size: int = 6,
    cell_force_scale: float = 1.0,
) -> LBFGSCellState:
    """Allocate variable-cell L-BFGS state for a fixed batch topology."""
    if positions.dtype not in (wp.vec3f, wp.vec3d) or positions.ndim != 1:
        raise ValueError("positions must be a rank-one wp.vec3f or wp.vec3d array")
    expected_matrix = wp.mat33f if positions.dtype == wp.vec3f else wp.mat33d
    expected_scalar = wp.float32 if positions.dtype == wp.vec3f else wp.float64
    if cell.dtype != expected_matrix or cell.ndim != 1:
        raise ValueError("cell must match the coordinate precision and be rank one")
    if batch_idx.dtype != wp.int32 or batch_idx.shape != positions.shape:
        raise ValueError("batch_idx must have one int32 entry per coordinate")
    if not isinstance(history_size, int) or history_size < 1:
        raise ValueError("history_size must be a positive integer")
    if not math.isfinite(cell_force_scale) or cell_force_scale <= 0.0:
        raise ValueError("cell_force_scale must be finite and positive")
    _same_device(positions, cell, batch_idx)

    num_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    _validate_cell_topology(batch_idx, num_systems)
    num_extended = num_atoms + 2 * num_systems
    optimizer = _allocate_state(
        num_extended,
        num_systems,
        history_size,
        positions.dtype,
        positions.device,
    )
    ref_cell = wp.zeros(num_systems, dtype=expected_matrix, device=positions.device)
    ref_cell_inv = wp.zeros(num_systems, dtype=expected_matrix, device=positions.device)
    cell_scale = wp.zeros(num_systems, dtype=expected_scalar, device=positions.device)
    atom_counts = wp.zeros(num_systems, dtype=wp.int32, device=positions.device)
    atom_ptr = wp.zeros(num_systems + 1, dtype=wp.int32, device=positions.device)
    ext_atom_ptr = wp.zeros(num_systems + 1, dtype=wp.int32, device=positions.device)
    ext_batch_idx = wp.zeros(num_extended, dtype=wp.int32, device=positions.device)
    batch_idx_to_atom_ptr(batch_idx, atom_counts, atom_ptr)
    extend_atom_ptr(atom_ptr, ext_atom_ptr, device=positions.device)
    atom_ptr_to_batch_idx(ext_atom_ptr, ext_batch_idx)
    if num_systems:
        wp.launch(
            _kernels._prepare_reference_cell_kernel,
            dim=num_systems,
            inputs=[cell, ref_cell, ref_cell_inv],
            device=positions.device,
        )
        wp.launch(
            _kernels._cell_scale_kernel,
            dim=num_systems,
            inputs=[atom_counts, wp.float64(cell_force_scale), cell_scale],
            device=positions.device,
        )
    return LBFGSCellState(
        optimizer=optimizer,
        ref_cell=ref_cell,
        ref_cell_inv=ref_cell_inv,
        cell_scale=cell_scale,
        ext_batch_idx=ext_batch_idx,
        ext_atom_ptr=ext_atom_ptr,
        phi=wp.zeros(num_systems, dtype=expected_matrix, device=positions.device),
        phi_inv=wp.zeros(num_systems, dtype=expected_matrix, device=positions.device),
        d_phi=wp.zeros(num_systems, dtype=expected_matrix, device=positions.device),
        ext_positions=wp.zeros(
            num_extended, dtype=positions.dtype, device=positions.device
        ),
        ext_forces=wp.zeros(
            num_extended, dtype=positions.dtype, device=positions.device
        ),
    )


def lbfgs_step_coord_cell(
    positions: wp.array,
    forces: wp.array,
    cell: wp.array,
    cell_force: wp.array,
    batch_idx: wp.array,
    state: LBFGSCellState,
    *,
    maxstep: float = 0.2,
) -> None:
    """Update L-BFGS history and apply one coupled atom/cell step."""
    if not isinstance(state, LBFGSCellState):
        raise ValueError("state must be an LBFGSCellState")
    if not math.isfinite(maxstep) or maxstep <= 0.0:
        raise ValueError("maxstep must be finite and positive")
    if positions.dtype not in (wp.vec3f, wp.vec3d) or positions.ndim != 1:
        raise ValueError("positions must be a rank-one wp.vec3f or wp.vec3d array")
    expected_matrix = wp.mat33f if positions.dtype == wp.vec3f else wp.mat33d
    expected_scalar = wp.float32 if positions.dtype == wp.vec3f else wp.float64
    if forces.shape != positions.shape or forces.dtype != positions.dtype:
        raise ValueError("forces must match positions in shape and dtype")
    if batch_idx.dtype != wp.int32 or batch_idx.shape != positions.shape:
        raise ValueError("batch_idx must have one int32 entry per coordinate")
    if cell.dtype != expected_matrix or cell_force.dtype != expected_matrix:
        raise ValueError("cell and cell_force must match coordinate precision")
    if cell.shape != cell_force.shape or cell.ndim != 1:
        raise ValueError("cell and cell_force must have one matrix per system")
    num_systems = cell.shape[0]
    num_extended = positions.shape[0] + 2 * num_systems
    if state.cell_scale.dtype != expected_scalar or state.cell_scale.shape != (
        num_systems,
    ):
        raise ValueError("cell_scale must match the coordinate precision and systems")
    for name, value in (
        ("ref_cell", state.ref_cell),
        ("ref_cell_inv", state.ref_cell_inv),
        ("phi", state.phi),
        ("phi_inv", state.phi_inv),
        ("d_phi", state.d_phi),
    ):
        if value.dtype != expected_matrix or value.shape != cell.shape:
            raise ValueError(f"{name} must match cell in shape and dtype")
    if state.ext_batch_idx.dtype != wp.int32 or state.ext_batch_idx.shape != (
        num_extended,
    ):
        raise ValueError("ext_batch_idx must match the extended topology")
    if state.ext_atom_ptr.dtype != wp.int32 or state.ext_atom_ptr.shape != (
        num_systems + 1,
    ):
        raise ValueError("ext_atom_ptr must match the extended topology")
    if (
        state.ext_positions.dtype != positions.dtype
        or state.ext_positions.shape != (num_extended,)
        or state.ext_forces.dtype != positions.dtype
        or state.ext_forces.shape != (num_extended,)
    ):
        raise ValueError("extended buffers must match the packed coordinate layout")
    _same_device(positions, forces, cell, cell_force, batch_idx, *state._cell_buffers())
    _validate_state(
        state.ext_positions,
        state.ext_forces,
        state.ext_batch_idx,
        state.optimizer,
    )
    if num_systems == 0:
        return

    wp.launch(
        _kernels._pack_cell_kernel,
        dim=num_systems,
        inputs=[
            positions,
            cell,
            forces,
            cell_force,
            batch_idx,
            state.ref_cell,
            state.ref_cell_inv,
            state.cell_scale,
            state.ext_atom_ptr,
            state.phi,
            state.phi_inv,
            state.d_phi,
            state.ext_positions,
            state.ext_forces,
        ],
        device=positions.device,
    )
    curvature_eps = (
        _CURVATURE_EPS_F32 if positions.dtype == wp.vec3f else _CURVATURE_EPS_F64
    )
    wp.launch(
        _kernels._lbfgs_step_kernel,
        dim=num_systems,
        inputs=[
            state.ext_positions,
            state.ext_forces,
            state.ext_batch_idx,
            *state.optimizer._buffers(),
            wp.float64(maxstep),
            wp.float64(curvature_eps),
            wp.bool(False),
        ],
        device=positions.device,
    )
    wp.launch(
        _kernels._apply_cell_step_kernel,
        dim=num_systems,
        inputs=[
            positions,
            cell,
            batch_idx,
            state.ref_cell,
            state.cell_scale,
            state.ext_atom_ptr,
            state.optimizer.direction,
            state.phi,
            state.phi_inv,
            state.d_phi,
            state.ext_positions,
            wp.float64(maxstep),
        ],
        device=positions.device,
    )
