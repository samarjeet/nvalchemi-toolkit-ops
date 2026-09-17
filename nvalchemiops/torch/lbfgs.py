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

"""PyTorch bindings for caller-owned, single-step L-BFGS."""

from __future__ import annotations

import math

import torch
import warp as wp

from nvalchemiops.dynamics.optimizers import lbfgs as _core
from nvalchemiops.torch._warp_op_helpers import (
    register_noop_fake,
    scoped_warp_stream,
    torch_custom_op,
)

__all__ = [
    "LBFGSState",
    "LBFGSCellState",
    "prepare_lbfgs_state",
    "prepare_lbfgs_cell_state",
    "lbfgs_step_coord",
    "lbfgs_step_coord_cell",
]

LBFGSState = _core.LBFGSState
LBFGSCellState = _core.LBFGSCellState
_COORD_MUTATED = ("positions",) + _core._OPTIMIZER_BUFFERS
_CELL_MUTATED = ("positions", "cell") + _core._OPTIMIZER_BUFFERS + _core._CELL_SCRATCH
_TORCH_TO_WP_VEC = {torch.float32: wp.vec3f, torch.float64: wp.vec3d}
_TORCH_TO_WP_MAT = {torch.float32: wp.mat33f, torch.float64: wp.mat33d}
_TORCH_TO_WP_SCALAR = {torch.float32: wp.float32, torch.float64: wp.float64}


def _wp(tensor: torch.Tensor, dtype):
    if not tensor.is_contiguous():
        raise ValueError("L-BFGS tensors must be contiguous")
    return wp.from_torch(tensor.detach(), dtype=dtype)


def _allocate_state(
    num_dofs: int,
    num_systems: int,
    history_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> LBFGSState:
    def coordinate(*shape):
        return torch.zeros(shape, dtype=dtype, device=device)

    def scalar(*shape):
        return torch.zeros(shape, dtype=torch.float64, device=device)

    def control(*shape):
        return torch.zeros(shape, dtype=torch.int32, device=device)

    return LBFGSState(
        x_base=coordinate(num_dofs, 3),
        force_base=coordinate(num_dofs, 3),
        direction=coordinate(num_dofs, 3),
        s_history=coordinate(history_size, num_dofs, 3),
        y_history=coordinate(history_size, num_dofs, 3),
        ys=scalar(history_size, num_systems),
        yy=scalar(history_size, num_systems),
        two_loop_alpha=scalar(history_size, num_systems),
        initialized=control(num_systems),
        history_end=control(num_systems),
        history_count=control(num_systems),
    )


def prepare_lbfgs_state(
    positions: torch.Tensor,
    num_systems: int,
    *,
    history_size: int = 6,
) -> LBFGSState:
    """Allocate L-BFGS history and scratch for a Torch coordinate batch."""
    if positions.dtype not in _TORCH_TO_WP_VEC or positions.ndim != 2:
        raise ValueError("positions must be a rank-2 float32 or float64 tensor")
    if positions.shape[1] != 3:
        raise ValueError("positions must have shape (num_dofs, 3)")
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
    positions: torch.Tensor,
    forces: torch.Tensor,
    batch_idx: torch.Tensor,
    state: LBFGSState,
) -> int:
    if not isinstance(state, LBFGSState):
        raise ValueError("state must be an LBFGSState")
    if positions.dtype not in _TORCH_TO_WP_VEC or positions.ndim != 2:
        raise ValueError("positions must be a rank-2 float32 or float64 tensor")
    if positions.shape[1] != 3:
        raise ValueError("positions must have shape (num_dofs, 3)")
    if forces.shape != positions.shape or forces.dtype != positions.dtype:
        raise ValueError("forces must match positions in shape and dtype")
    if batch_idx.dtype != torch.int32 or batch_idx.shape != (positions.shape[0],):
        raise ValueError("batch_idx must have one int32 entry per coordinate")
    for name in ("x_base", "force_base", "direction"):
        value = getattr(state, name)
        if value.shape != positions.shape or value.dtype != positions.dtype:
            raise ValueError(f"{name} must match positions in shape and dtype")
    if (
        state.s_history.ndim != 3
        or state.y_history.shape != state.s_history.shape
        or state.s_history.shape[1:] != positions.shape
        or state.s_history.dtype != positions.dtype
        or state.y_history.dtype != positions.dtype
        or state.s_history.shape[0] < 1
    ):
        raise ValueError("history arrays must match positions and have depth >= 1")
    history_size = state.s_history.shape[0]
    if state.initialized.dtype != torch.int32 or state.initialized.ndim != 1:
        raise ValueError("initialized must be a rank-1 int32 tensor")
    num_systems = state.initialized.shape[0]
    for name in ("ys", "yy", "two_loop_alpha"):
        value = getattr(state, name)
        if value.dtype != torch.float64 or value.shape != (
            history_size,
            num_systems,
        ):
            raise ValueError(
                f"{name} must be float64 with shape ({history_size}, {num_systems})"
            )
    for name in ("initialized", "history_end", "history_count"):
        value = getattr(state, name)
        if value.dtype != torch.int32 or value.shape != (num_systems,):
            raise ValueError(f"{name} must be int32 with one entry per system")
    tensors = (forces, batch_idx, *state._buffers())
    if any(value.device != positions.device for value in tensors):
        raise ValueError("all L-BFGS tensors must share a device")
    if any(not value.is_contiguous() for value in (positions, *tensors)):
        raise ValueError("all L-BFGS tensors must be contiguous")
    return num_systems


@torch_custom_op("nvalchemiops::lbfgs_step_coord", mutates_args=_COORD_MUTATED)
def _lbfgs_step_coord_op(
    positions: torch.Tensor,
    forces: torch.Tensor,
    batch_idx: torch.Tensor,
    x_base: torch.Tensor,
    force_base: torch.Tensor,
    direction: torch.Tensor,
    s_history: torch.Tensor,
    y_history: torch.Tensor,
    ys: torch.Tensor,
    yy: torch.Tensor,
    two_loop_alpha: torch.Tensor,
    initialized: torch.Tensor,
    history_end: torch.Tensor,
    history_count: torch.Tensor,
    maxstep: float,
) -> None:
    """Launch the shared Core coordinate step on Torch storage."""
    vector = _TORCH_TO_WP_VEC[positions.dtype]
    with scoped_warp_stream(positions.device):
        state = LBFGSState(
            _wp(x_base, vector),
            _wp(force_base, vector),
            _wp(direction, vector),
            _wp(s_history, vector),
            _wp(y_history, vector),
            _wp(ys, wp.float64),
            _wp(yy, wp.float64),
            _wp(two_loop_alpha, wp.float64),
            _wp(initialized, wp.int32),
            _wp(history_end, wp.int32),
            _wp(history_count, wp.int32),
        )
        _core.lbfgs_step_coord(
            _wp(positions, vector),
            _wp(forces, vector),
            _wp(batch_idx, wp.int32),
            state,
            maxstep=maxstep,
        )


register_noop_fake(_lbfgs_step_coord_op)


def lbfgs_step_coord(
    positions: torch.Tensor,
    forces: torch.Tensor,
    batch_idx: torch.Tensor,
    state: LBFGSState,
    *,
    maxstep: float = 0.2,
) -> None:
    """Update L-BFGS history and apply one capped Torch coordinate step."""
    _validate_state(positions, forces, batch_idx, state)
    if not math.isfinite(maxstep) or maxstep <= 0.0:
        raise ValueError("maxstep must be finite and positive")
    _lbfgs_step_coord_op(
        positions,
        forces,
        batch_idx,
        *state._buffers(),
        maxstep,
    )


def prepare_lbfgs_cell_state(
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor,
    *,
    history_size: int = 6,
    cell_force_scale: float = 1.0,
) -> LBFGSCellState:
    """Allocate variable-cell L-BFGS state for a fixed Torch batch."""
    if positions.dtype not in _TORCH_TO_WP_VEC or positions.ndim != 2:
        raise ValueError("positions must be a rank-2 float32 or float64 tensor")
    if positions.shape[1] != 3:
        raise ValueError("positions must have shape (num_atoms, 3)")
    num_systems = cell.shape[0] if cell.ndim == 3 else -1
    if cell.dtype != positions.dtype or cell.ndim != 3 or cell.shape[1:] != (3, 3):
        raise ValueError("cell must match positions and have shape (M, 3, 3)")
    if batch_idx.dtype != torch.int32 or batch_idx.shape != (positions.shape[0],):
        raise ValueError("batch_idx must have one int32 entry per atom")
    if any(value.device != positions.device for value in (cell, batch_idx)):
        raise ValueError("positions, cell, and batch_idx must share a device")
    if any(not value.is_contiguous() for value in (positions, cell, batch_idx)):
        raise ValueError("preparation inputs must be contiguous")
    if not isinstance(history_size, int) or history_size < 1:
        raise ValueError("history_size must be a positive integer")
    if not math.isfinite(cell_force_scale) or cell_force_scale <= 0.0:
        raise ValueError("cell_force_scale must be finite and positive")

    if num_systems == 0:
        if batch_idx.numel():
            raise ValueError("batch_idx must match the number of cell systems")
    elif (
        batch_idx.numel() == 0
        or torch.any(batch_idx < 0).item()
        or torch.any(batch_idx >= num_systems).item()
    ):
        raise ValueError("batch_idx must contain one in-range atom for every system")
    atom_counts = torch.bincount(batch_idx.to(torch.int64), minlength=num_systems)
    if atom_counts.shape[0] != num_systems or torch.any(atom_counts <= 0).item():
        raise ValueError("batch_idx must contain one in-range atom for every system")
    expected = torch.repeat_interleave(
        torch.arange(num_systems, dtype=torch.int32, device=positions.device),
        atom_counts,
    )
    if not torch.equal(batch_idx, expected):
        raise ValueError("batch_idx must be sorted and contiguous by system")
    atom_counts = atom_counts.to(torch.int32)
    num_extended = positions.shape[0] + 2 * num_systems
    optimizer = _allocate_state(
        num_extended,
        num_systems,
        history_size,
        positions.dtype,
        positions.device,
    )
    ext_atom_ptr = torch.zeros(
        num_systems + 1, dtype=torch.int32, device=positions.device
    )
    ext_atom_ptr[1:] = torch.cumsum(atom_counts + 2, dim=0)
    ext_batch_idx = torch.repeat_interleave(
        torch.arange(num_systems, dtype=torch.int32, device=positions.device),
        atom_counts + 2,
    )
    return LBFGSCellState(
        optimizer=optimizer,
        ref_cell=cell.clone(),
        ref_cell_inv=torch.linalg.inv(cell).contiguous(),
        cell_scale=atom_counts.to(positions.dtype) * cell_force_scale,
        ext_batch_idx=ext_batch_idx,
        ext_atom_ptr=ext_atom_ptr,
        phi=torch.zeros_like(cell),
        phi_inv=torch.zeros_like(cell),
        d_phi=torch.zeros_like(cell),
        ext_positions=torch.zeros(
            (num_extended, 3), dtype=positions.dtype, device=positions.device
        ),
        ext_forces=torch.zeros(
            (num_extended, 3), dtype=positions.dtype, device=positions.device
        ),
    )


@torch_custom_op(
    "nvalchemiops::lbfgs_step_coord_cell",
    mutates_args=_CELL_MUTATED,
)
def _lbfgs_step_coord_cell_op(
    positions: torch.Tensor,
    forces: torch.Tensor,
    cell: torch.Tensor,
    cell_force: torch.Tensor,
    batch_idx: torch.Tensor,
    x_base: torch.Tensor,
    force_base: torch.Tensor,
    direction: torch.Tensor,
    s_history: torch.Tensor,
    y_history: torch.Tensor,
    ys: torch.Tensor,
    yy: torch.Tensor,
    two_loop_alpha: torch.Tensor,
    initialized: torch.Tensor,
    history_end: torch.Tensor,
    history_count: torch.Tensor,
    ref_cell: torch.Tensor,
    ref_cell_inv: torch.Tensor,
    cell_scale: torch.Tensor,
    ext_batch_idx: torch.Tensor,
    ext_atom_ptr: torch.Tensor,
    phi: torch.Tensor,
    phi_inv: torch.Tensor,
    d_phi: torch.Tensor,
    ext_positions: torch.Tensor,
    ext_forces: torch.Tensor,
    maxstep: float,
) -> None:
    """Launch the shared Core variable-cell step on Torch storage."""
    vector = _TORCH_TO_WP_VEC[positions.dtype]
    matrix = _TORCH_TO_WP_MAT[positions.dtype]
    scalar = _TORCH_TO_WP_SCALAR[positions.dtype]
    with scoped_warp_stream(positions.device):
        optimizer = LBFGSState(
            _wp(x_base, vector),
            _wp(force_base, vector),
            _wp(direction, vector),
            _wp(s_history, vector),
            _wp(y_history, vector),
            _wp(ys, wp.float64),
            _wp(yy, wp.float64),
            _wp(two_loop_alpha, wp.float64),
            _wp(initialized, wp.int32),
            _wp(history_end, wp.int32),
            _wp(history_count, wp.int32),
        )
        state = LBFGSCellState(
            optimizer,
            _wp(ref_cell, matrix),
            _wp(ref_cell_inv, matrix),
            _wp(cell_scale, scalar),
            _wp(ext_batch_idx, wp.int32),
            _wp(ext_atom_ptr, wp.int32),
            _wp(phi, matrix),
            _wp(phi_inv, matrix),
            _wp(d_phi, matrix),
            _wp(ext_positions, vector),
            _wp(ext_forces, vector),
        )
        _core.lbfgs_step_coord_cell(
            _wp(positions, vector),
            _wp(forces, vector),
            _wp(cell, matrix),
            _wp(cell_force, matrix),
            _wp(batch_idx, wp.int32),
            state,
            maxstep=maxstep,
        )


register_noop_fake(_lbfgs_step_coord_cell_op)


def lbfgs_step_coord_cell(
    positions: torch.Tensor,
    forces: torch.Tensor,
    cell: torch.Tensor,
    cell_force: torch.Tensor,
    batch_idx: torch.Tensor,
    state: LBFGSCellState,
    *,
    maxstep: float = 0.2,
) -> None:
    """Update L-BFGS history and apply one coupled Torch atom/cell step."""
    if not isinstance(state, LBFGSCellState):
        raise ValueError("state must be an LBFGSCellState")
    if positions.dtype not in _TORCH_TO_WP_VEC or positions.ndim != 2:
        raise ValueError("positions must be a rank-2 float32 or float64 tensor")
    if positions.shape[1] != 3:
        raise ValueError("positions must have shape (num_atoms, 3)")
    num_systems = cell.shape[0] if cell.ndim == 3 else -1
    if forces.shape != positions.shape or forces.dtype != positions.dtype:
        raise ValueError("forces must match positions in shape and dtype")
    if (
        cell.dtype != positions.dtype
        or cell.ndim != 3
        or cell.shape[1:] != (3, 3)
        or cell_force.shape != cell.shape
        or cell_force.dtype != cell.dtype
    ):
        raise ValueError("cell and cell_force must match the coordinate precision")
    if batch_idx.dtype != torch.int32 or batch_idx.shape != (positions.shape[0],):
        raise ValueError("batch_idx must have one int32 entry per atom")
    num_extended = positions.shape[0] + 2 * num_systems
    _validate_state(
        state.ext_positions,
        state.ext_forces,
        state.ext_batch_idx,
        state.optimizer,
    )
    if state.ext_positions.shape != (num_extended, 3):
        raise ValueError("extended buffers must match the packed coordinate layout")
    for name, value in (
        ("ref_cell", state.ref_cell),
        ("ref_cell_inv", state.ref_cell_inv),
        ("phi", state.phi),
        ("phi_inv", state.phi_inv),
        ("d_phi", state.d_phi),
    ):
        if value.shape != cell.shape or value.dtype != cell.dtype:
            raise ValueError(f"{name} must match cell in shape and dtype")
    if state.cell_scale.shape != (num_systems,) or state.cell_scale.dtype != cell.dtype:
        raise ValueError("cell_scale must match the coordinate precision and systems")
    if state.ext_atom_ptr.dtype != torch.int32 or state.ext_atom_ptr.shape != (
        num_systems + 1,
    ):
        raise ValueError("ext_atom_ptr must match the extended topology")
    tensors = (
        forces,
        cell,
        cell_force,
        batch_idx,
        *state.optimizer._buffers(),
        *state._cell_buffers(),
    )
    if any(value.device != positions.device for value in tensors):
        raise ValueError("all variable-cell L-BFGS tensors must share a device")
    if any(not value.is_contiguous() for value in (positions, *tensors)):
        raise ValueError("all variable-cell L-BFGS tensors must be contiguous")
    if not math.isfinite(maxstep) or maxstep <= 0.0:
        raise ValueError("maxstep must be finite and positive")
    _lbfgs_step_coord_cell_op(
        positions,
        forces,
        cell,
        cell_force,
        batch_idx,
        *state.optimizer._buffers(),
        *state._cell_buffers(),
        maxstep,
    )
