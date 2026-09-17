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

"""Functional JAX bindings for the one-step L-BFGS optimizer."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from warp import jax_callable

from nvalchemiops.dynamics.optimizers import lbfgs as _core

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
_STATE_NAMES = _core._OPTIMIZER_BUFFERS
_CELL_MUTABLE_NAMES = _core._CELL_SCRATCH


def _state_flatten(state: LBFGSState):
    return tuple(getattr(state, name) for name in _STATE_NAMES), None


def _state_unflatten(_aux, children):
    return LBFGSState(*children)


def _cell_flatten(state: LBFGSCellState):
    return (
        state.optimizer,
        state.ref_cell,
        state.ref_cell_inv,
        state.cell_scale,
        state.ext_batch_idx,
        state.ext_atom_ptr,
        state.phi,
        state.phi_inv,
        state.d_phi,
        state.ext_positions,
        state.ext_forces,
    ), None


def _cell_unflatten(_aux, children):
    return LBFGSCellState(*children)


jax.tree_util.register_pytree_node(LBFGSState, _state_flatten, _state_unflatten)
jax.tree_util.register_pytree_node(LBFGSCellState, _cell_flatten, _cell_unflatten)


def _validate_x64() -> None:
    if not jax.config.jax_enable_x64:
        raise ValueError("L-BFGS requires jax_enable_x64=True for float64 reductions")


def _validate_coordinates(
    positions: jax.Array, forces: jax.Array, batch_idx: jax.Array
) -> None:
    if (
        positions.dtype not in (jnp.float32, jnp.float64)
        or positions.ndim != 2
        or positions.shape[1] != 3
    ):
        raise ValueError(
            "positions must have shape (num_atoms, 3) and float32 or float64 dtype"
        )
    if forces.shape != positions.shape or forces.dtype != positions.dtype:
        raise ValueError("forces must match positions in shape and dtype")
    if batch_idx.dtype != jnp.int32 or batch_idx.shape != (positions.shape[0],):
        raise ValueError("batch_idx must have one int32 entry per coordinate")


def _validate_state(positions: jax.Array, state: LBFGSState) -> None:
    if not isinstance(state, LBFGSState):
        raise ValueError("state must be an LBFGSState")
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
    systems = state.initialized.shape[0]
    for name in ("ys", "yy", "two_loop_alpha"):
        value = getattr(state, name)
        if value.dtype != jnp.float64 or value.shape != (history_size, systems):
            raise ValueError(
                f"{name} must be float64 with shape (history_size, num_systems)"
            )
    for name in ("initialized", "history_end", "history_count"):
        value = getattr(state, name)
        if value.dtype != jnp.int32 or value.shape != (systems,):
            raise ValueError(f"{name} must be int32 with one entry per system")


def _validate_cell_topology(batch_idx: jax.Array, num_systems: int) -> None:
    """Require one sorted, nonempty contiguous atom segment per cell."""
    labels = np.asarray(batch_idx)
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


def prepare_lbfgs_state(
    positions: jax.Array, num_systems: int, *, history_size: int = 6
) -> LBFGSState:
    """Allocate the explicit functional state for a coordinate batch."""
    _validate_x64()
    if (
        positions.dtype not in (jnp.float32, jnp.float64)
        or positions.ndim != 2
        or positions.shape[1] != 3
    ):
        raise ValueError(
            "positions must have shape (num_atoms, 3) and float32 or float64 dtype"
        )
    if not isinstance(num_systems, int) or num_systems < 0:
        raise ValueError("num_systems must be a nonnegative integer")
    if not isinstance(history_size, int) or history_size < 1:
        raise ValueError("history_size must be a positive integer")

    def coordinates(*shape):
        return jnp.zeros(shape, positions.dtype)

    def coefficients(*shape):
        return jnp.zeros(shape, jnp.float64)

    def controls(*shape):
        return jnp.zeros(shape, jnp.int32)

    return LBFGSState(
        coordinates(*positions.shape),
        coordinates(*positions.shape),
        coordinates(*positions.shape),
        coordinates(history_size, *positions.shape),
        coordinates(history_size, *positions.shape),
        coefficients(history_size, num_systems),
        coefficients(history_size, num_systems),
        coefficients(history_size, num_systems),
        controls(num_systems),
        controls(num_systems),
        controls(num_systems),
    )


def prepare_lbfgs_cell_state(
    positions: jax.Array,
    cell: jax.Array,
    batch_idx: jax.Array,
    *,
    history_size: int = 6,
    cell_force_scale: float = 1.0,
) -> LBFGSCellState:
    """Allocate the explicit functional state for a fixed cell topology."""
    _validate_x64()
    _validate_coordinates(positions, positions, batch_idx)
    if cell.dtype != positions.dtype or cell.ndim != 3 or cell.shape[1:] != (3, 3):
        raise ValueError("cell must match positions and have shape (num_systems, 3, 3)")
    if not isinstance(history_size, int) or history_size < 1:
        raise ValueError("history_size must be a positive integer")
    if not math.isfinite(cell_force_scale) or cell_force_scale <= 0.0:
        raise ValueError("cell_force_scale must be finite and positive")
    systems = cell.shape[0]
    _validate_cell_topology(batch_idx, systems)
    counts = jnp.bincount(batch_idx, length=systems).astype(jnp.int32)
    atom_ptr = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(counts)))
    ext_atom_ptr = atom_ptr + 2 * jnp.arange(systems + 1, dtype=jnp.int32)
    ext_batch_idx = jnp.repeat(jnp.arange(systems, dtype=jnp.int32), counts + 2)
    extended = positions.shape[0] + 2 * systems
    optimizer = prepare_lbfgs_state(
        jnp.zeros((extended, 3), positions.dtype), systems, history_size=history_size
    )
    # Keep reference geometry independent so a caller can donate ``cell`` and
    # the complete state to a JIT-compiled variable-cell step.
    ref_cell = jnp.array(cell, copy=True)
    return LBFGSCellState(
        optimizer,
        ref_cell,
        jnp.linalg.inv(ref_cell),
        counts.astype(positions.dtype) * cell_force_scale,
        ext_batch_idx,
        ext_atom_ptr,
        jnp.zeros_like(cell),
        jnp.zeros_like(cell),
        jnp.zeros_like(cell),
        jnp.zeros((extended, 3), positions.dtype),
        jnp.zeros((extended, 3), positions.dtype),
    )


def _body_f32(
    forces: wp.array(dtype=wp.vec3f),
    batch_idx: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3f),
    x_base: wp.array(dtype=wp.vec3f),
    force_base: wp.array(dtype=wp.vec3f),
    direction: wp.array(dtype=wp.vec3f),
    s_history: wp.array(dtype=wp.vec3f, ndim=2),
    y_history: wp.array(dtype=wp.vec3f, ndim=2),
    ys: wp.array(dtype=wp.float64, ndim=2),
    yy: wp.array(dtype=wp.float64, ndim=2),
    two_loop_alpha: wp.array(dtype=wp.float64, ndim=2),
    initialized: wp.array(dtype=wp.int32),
    history_end: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    maxstep: wp.float64,
) -> None:
    _core.lbfgs_step_coord(
        positions,
        forces,
        batch_idx,
        LBFGSState(
            x_base,
            force_base,
            direction,
            s_history,
            y_history,
            ys,
            yy,
            two_loop_alpha,
            initialized,
            history_end,
            history_count,
        ),
        maxstep=maxstep,
    )


def _body_f64(
    forces: wp.array(dtype=wp.vec3d),
    batch_idx: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3d),
    x_base: wp.array(dtype=wp.vec3d),
    force_base: wp.array(dtype=wp.vec3d),
    direction: wp.array(dtype=wp.vec3d),
    s_history: wp.array(dtype=wp.vec3d, ndim=2),
    y_history: wp.array(dtype=wp.vec3d, ndim=2),
    ys: wp.array(dtype=wp.float64, ndim=2),
    yy: wp.array(dtype=wp.float64, ndim=2),
    two_loop_alpha: wp.array(dtype=wp.float64, ndim=2),
    initialized: wp.array(dtype=wp.int32),
    history_end: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    maxstep: wp.float64,
) -> None:
    _core.lbfgs_step_coord(
        positions,
        forces,
        batch_idx,
        LBFGSState(
            x_base,
            force_base,
            direction,
            s_history,
            y_history,
            ys,
            yy,
            two_loop_alpha,
            initialized,
            history_end,
            history_count,
        ),
        maxstep=maxstep,
    )


_COORD_CALLABLES: dict[object, object] = {}


def _coord_callable(dtype):
    key = jnp.dtype(dtype).type
    if key not in _COORD_CALLABLES:
        body = {jnp.float32: _body_f32, jnp.float64: _body_f64}.get(key)
        if body is None:
            raise ValueError("positions must have float32 or float64 dtype")
        _COORD_CALLABLES[key] = jax_callable(
            body,
            num_outputs=1 + len(_STATE_NAMES),
            in_out_argnames=["positions", *_STATE_NAMES],
        )
    return _COORD_CALLABLES[key]


def lbfgs_step_coord(
    positions: jax.Array,
    forces: jax.Array,
    batch_idx: jax.Array,
    state: LBFGSState,
    *,
    maxstep: float = 0.2,
) -> tuple[jax.Array, LBFGSState]:
    """Return the positions and explicit state after one L-BFGS step."""
    _validate_x64()
    _validate_coordinates(positions, forces, batch_idx)
    _validate_state(positions, state)
    if not math.isfinite(maxstep) or maxstep <= 0.0:
        raise ValueError("maxstep must be finite and positive")
    outputs = _coord_callable(positions.dtype)(
        forces, batch_idx, positions, *state._buffers(), maxstep
    )
    return outputs[0], LBFGSState(*outputs[1:])


def _cell_body_f32(
    forces: wp.array(dtype=wp.vec3f),
    cell_force: wp.array(dtype=wp.mat33f),
    batch_idx: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    x_base: wp.array(dtype=wp.vec3f),
    force_base: wp.array(dtype=wp.vec3f),
    direction: wp.array(dtype=wp.vec3f),
    s_history: wp.array(dtype=wp.vec3f, ndim=2),
    y_history: wp.array(dtype=wp.vec3f, ndim=2),
    ys: wp.array(dtype=wp.float64, ndim=2),
    yy: wp.array(dtype=wp.float64, ndim=2),
    two_loop_alpha: wp.array(dtype=wp.float64, ndim=2),
    initialized: wp.array(dtype=wp.int32),
    history_end: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    ref_cell: wp.array(dtype=wp.mat33f),
    ref_cell_inv: wp.array(dtype=wp.mat33f),
    cell_scale: wp.array(dtype=wp.float32),
    ext_batch_idx: wp.array(dtype=wp.int32),
    ext_atom_ptr: wp.array(dtype=wp.int32),
    phi: wp.array(dtype=wp.mat33f),
    phi_inv: wp.array(dtype=wp.mat33f),
    d_phi: wp.array(dtype=wp.mat33f),
    ext_positions: wp.array(dtype=wp.vec3f),
    ext_forces: wp.array(dtype=wp.vec3f),
    maxstep: wp.float64,
) -> None:
    _core.lbfgs_step_coord_cell(
        positions,
        forces,
        cell,
        cell_force,
        batch_idx,
        LBFGSCellState(
            LBFGSState(
                x_base,
                force_base,
                direction,
                s_history,
                y_history,
                ys,
                yy,
                two_loop_alpha,
                initialized,
                history_end,
                history_count,
            ),
            ref_cell,
            ref_cell_inv,
            cell_scale,
            ext_batch_idx,
            ext_atom_ptr,
            phi,
            phi_inv,
            d_phi,
            ext_positions,
            ext_forces,
        ),
        maxstep=maxstep,
    )


def _cell_body_f64(
    forces: wp.array(dtype=wp.vec3d),
    cell_force: wp.array(dtype=wp.mat33d),
    batch_idx: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    x_base: wp.array(dtype=wp.vec3d),
    force_base: wp.array(dtype=wp.vec3d),
    direction: wp.array(dtype=wp.vec3d),
    s_history: wp.array(dtype=wp.vec3d, ndim=2),
    y_history: wp.array(dtype=wp.vec3d, ndim=2),
    ys: wp.array(dtype=wp.float64, ndim=2),
    yy: wp.array(dtype=wp.float64, ndim=2),
    two_loop_alpha: wp.array(dtype=wp.float64, ndim=2),
    initialized: wp.array(dtype=wp.int32),
    history_end: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    ref_cell: wp.array(dtype=wp.mat33d),
    ref_cell_inv: wp.array(dtype=wp.mat33d),
    cell_scale: wp.array(dtype=wp.float64),
    ext_batch_idx: wp.array(dtype=wp.int32),
    ext_atom_ptr: wp.array(dtype=wp.int32),
    phi: wp.array(dtype=wp.mat33d),
    phi_inv: wp.array(dtype=wp.mat33d),
    d_phi: wp.array(dtype=wp.mat33d),
    ext_positions: wp.array(dtype=wp.vec3d),
    ext_forces: wp.array(dtype=wp.vec3d),
    maxstep: wp.float64,
) -> None:
    _core.lbfgs_step_coord_cell(
        positions,
        forces,
        cell,
        cell_force,
        batch_idx,
        LBFGSCellState(
            LBFGSState(
                x_base,
                force_base,
                direction,
                s_history,
                y_history,
                ys,
                yy,
                two_loop_alpha,
                initialized,
                history_end,
                history_count,
            ),
            ref_cell,
            ref_cell_inv,
            cell_scale,
            ext_batch_idx,
            ext_atom_ptr,
            phi,
            phi_inv,
            d_phi,
            ext_positions,
            ext_forces,
        ),
        maxstep=maxstep,
    )


_CELL_CALLABLES: dict[object, object] = {}


def _cell_callable(dtype):
    key = jnp.dtype(dtype).type
    if key not in _CELL_CALLABLES:
        body = {jnp.float32: _cell_body_f32, jnp.float64: _cell_body_f64}.get(key)
        if body is None:
            raise ValueError("positions must have float32 or float64 dtype")
        _CELL_CALLABLES[key] = jax_callable(
            body,
            num_outputs=2 + len(_STATE_NAMES) + len(_CELL_MUTABLE_NAMES),
            in_out_argnames=["positions", "cell", *_STATE_NAMES, *_CELL_MUTABLE_NAMES],
        )
    return _CELL_CALLABLES[key]


def lbfgs_step_coord_cell(
    positions: jax.Array,
    forces: jax.Array,
    cell: jax.Array,
    cell_force: jax.Array,
    batch_idx: jax.Array,
    state: LBFGSCellState,
    *,
    maxstep: float = 0.2,
) -> tuple[jax.Array, jax.Array, LBFGSCellState]:
    """Return positions, cell, and explicit state after one coupled step."""
    _validate_x64()
    _validate_coordinates(positions, forces, batch_idx)
    if not isinstance(state, LBFGSCellState):
        raise ValueError("state must be an LBFGSCellState")
    if (
        cell.dtype != positions.dtype
        or cell.ndim != 3
        or cell.shape[1:] != (3, 3)
        or cell_force.shape != cell.shape
        or cell_force.dtype != cell.dtype
    ):
        raise ValueError(
            "cell and cell_force must match positions with shape (num_systems, 3, 3)"
        )
    _validate_state(state.ext_positions, state.optimizer)
    systems = cell.shape[0]
    extended = positions.shape[0] + 2 * systems
    if (
        state.ext_positions.shape != (extended, 3)
        or state.ext_forces.shape != state.ext_positions.shape
    ):
        raise ValueError("extended buffers must match the packed coordinate layout")
    if (
        state.ext_batch_idx.dtype != jnp.int32
        or state.ext_batch_idx.shape != (extended,)
        or state.ext_atom_ptr.dtype != jnp.int32
        or state.ext_atom_ptr.shape != (systems + 1,)
    ):
        raise ValueError("extended topology must match positions and cell")
    for name in ("ref_cell", "ref_cell_inv", "phi", "phi_inv", "d_phi"):
        value = getattr(state, name)
        if value.shape != cell.shape or value.dtype != cell.dtype:
            raise ValueError(f"{name} must match cell in shape and dtype")
    if (
        state.cell_scale.shape != (systems,)
        or state.cell_scale.dtype != positions.dtype
    ):
        raise ValueError("cell_scale must match the coordinate dtype and systems")
    if not math.isfinite(maxstep) or maxstep <= 0.0:
        raise ValueError("maxstep must be finite and positive")
    outputs = _cell_callable(positions.dtype)(
        forces,
        cell_force,
        batch_idx,
        positions,
        cell,
        *state.optimizer._buffers(),
        state.ref_cell,
        state.ref_cell_inv,
        state.cell_scale,
        state.ext_batch_idx,
        state.ext_atom_ptr,
        state.phi,
        state.phi_inv,
        state.d_phi,
        state.ext_positions,
        state.ext_forces,
        maxstep,
    )
    mutable_start = 2 + len(_STATE_NAMES)
    optimizer = LBFGSState(*outputs[2:mutable_start])
    phi, phi_inv, d_phi, ext_positions, ext_forces = outputs[mutable_start:]
    return (
        outputs[0],
        outputs[1],
        LBFGSCellState(
            optimizer,
            state.ref_cell,
            state.ref_cell_inv,
            state.cell_scale,
            state.ext_batch_idx,
            state.ext_atom_ptr,
            phi,
            phi_inv,
            d_phi,
            ext_positions,
            ext_forces,
        ),
    )
