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

"""PyTorch bindings for the batched L-BFGS geometry optimizer.

L-BFGS reaches a given force tolerance in far fewer energy/force evaluations
than the FIRE optimizers, which is the cost that dominates relaxation with a
machine-learned potential.

You own the loop. Each :func:`lbfgs_step_coord` call consumes exactly one
energy/force evaluation and mutates the state in place::

    from nvalchemiops.torch.lbfgs import (
        LBFGS_NEED_EVAL, lbfgs_allocate_state, lbfgs_step_coord,
    )

    state = lbfgs_allocate_state(
        num_dofs=positions.shape[0], num_systems=num_systems,
        dtype=positions.dtype, device=positions.device,
    )

    while True:
        energy, forces = model(positions)     # energy per system, float64
        lbfgs_step_coord(
            positions, state, forces, energy, batch_idx, n_particles,
            force_tol=0.05, maxstep=0.2,
        )
        if not (state.status == LBFGS_NEED_EVAL).any():
            break

``state.status`` is the only value you need to inspect: ``LBFGS_NEED_EVAL``
means keep going, ``LBFGS_CONVERGED`` means ``positions`` hold the answer, and
``LBFGS_LS_FAILED`` means the line search stalled even from a steepest-descent
direction and ``positions`` were restored to the last accepted point.

These operations mutate their inputs and are not differentiable; they are
registered as PyTorch custom operators so they trace correctly under
``torch.compile``.

CUDA graphs
-----------
A step captures in a CUDA graph, which is worth doing for a loop that runs
thousands of times. Warp launches have to be bound to the capture stream by the
caller, so wrap the capture::

    import warp as wp

    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
        with torch.cuda.graph(graph):
            lbfgs_step_coord(...)

Without that scope the capture records nothing and replay silently does no
work. Every buffer must be pre-allocated and reused, which
:func:`lbfgs_allocate_state` gives you; the step itself allocates nothing and
does all of its zeroing on the device.

See Also
--------
nvalchemiops.dynamics.optimizers.lbfgs : the underlying Warp implementation,
    which documents the algorithm, the sign convention and the precision
    policy in detail.
"""

from __future__ import annotations

import inspect

import torch
import warp as wp

from nvalchemiops.dynamics.optimizers.lbfgs import (
    LBFGS_CONVERGED,
    LBFGS_LS_FAILED,
    LBFGS_NEED_EVAL,
    LBFGSCellState,
    LBFGSState,
)
from nvalchemiops.dynamics.optimizers.lbfgs import lbfgs_cell_kappa as _wp_cell_kappa
from nvalchemiops.dynamics.optimizers.lbfgs import (
    lbfgs_reduce_energy as _wp_reduce_energy,
)
from nvalchemiops.dynamics.optimizers.lbfgs import lbfgs_reset as _wp_reset
from nvalchemiops.dynamics.optimizers.lbfgs import (
    lbfgs_set_reference_cell as _wp_set_reference_cell,
)
from nvalchemiops.dynamics.optimizers.lbfgs import lbfgs_step as _wp_step
from nvalchemiops.dynamics.optimizers.lbfgs import (
    lbfgs_step_coord_cell as _wp_step_cell,
)
from nvalchemiops.torch._warp_op_helpers import (
    register_noop_fake,
    scoped_warp_stream,
    torch_custom_op,
)

__all__ = [
    "LBFGSCellState",
    "LBFGSState",
    "LBFGS_CONVERGED",
    "LBFGS_LS_FAILED",
    "LBFGS_NEED_EVAL",
    "lbfgs_allocate_state",
    "lbfgs_reduce_energy",
    "lbfgs_reset",
    "lbfgs_allocate_cell_state",
    "lbfgs_set_reference_cell",
    "lbfgs_step_coord",
    "lbfgs_step_coord_cell",
    "lbfgs_step_extended",
]

_TORCH_TO_WP_VEC = {torch.float32: wp.vec3f, torch.float64: wp.vec3d}
_TORCH_TO_WP_MAT = {torch.float32: wp.mat33f, torch.float64: wp.mat33d}

#: Every tensor the step operator writes to. Derived from the state container
#: so the two cannot drift apart.
_MUTATED = ("positions",) + LBFGSState._fields


def _wp(tensor: torch.Tensor, dtype):
    """View a Torch tensor as a Warp array, without copying.

    Deliberately refuses non-contiguous input rather than calling
    ``.contiguous()`` for you. These operations write through the view, so a
    silent copy would discard every update and leave the caller watching an
    optimizer that never moves. Call ``.contiguous()`` yourself if you need to.
    """
    if not tensor.is_contiguous():
        raise ValueError(
            "L-BFGS tensors must be contiguous, because the optimizer writes "
            "through them in place; a non-contiguous tensor would be copied and "
            "the updates lost. Call .contiguous() on the argument first."
        )
    return wp.from_torch(tensor.detach(), dtype=dtype)


@torch_custom_op("nvalchemiops::lbfgs_step", mutates_args=_MUTATED)
def _lbfgs_step_op(
    positions: torch.Tensor,
    forces: torch.Tensor,
    energy: torch.Tensor,
    batch_idx: torch.Tensor,
    n_particles: torch.Tensor,
    x_base: torch.Tensor,
    force_base: torch.Tensor,
    direction: torch.Tensor,
    s_history: torch.Tensor,
    y_history: torch.Tensor,
    ys: torch.Tensor,
    yy: torch.Tensor,
    alpha_hist: torch.Tensor,
    beta_hist: torch.Tensor,
    ss: torch.Tensor,
    f_base: torch.Tensor,
    gg: torch.Tensor,
    gd: torch.Tensor,
    fmax: torch.Tensor,
    frms_sq: torch.Tensor,
    smax: torch.Tensor,
    d0: torch.Tensor,
    dmax: torch.Tensor,
    dquad: torch.Tensor,
    alpha_step: torch.Tensor,
    status: torch.Tensor,
    iteration: torch.Tensor,
    end: torch.Tensor,
    n_loop: torch.Tensor,
    ls_trials: torch.Tensor,
    history_count: torch.Tensor,
    force_tol: float,
    rms_tol: float,
    stress_tol: float,
    ftol: float,
    wolfe: float,
    step_scale_down: float,
    step_scale_up: float,
    min_step: float,
    max_step: float,
    max_ls_iter: int,
    maxstep: float,
    curvature_eps: float,
    compute_reductions: bool,
) -> None:
    """Run one registered L-BFGS step. All tensors are passed positionally.

    Launches are bound to PyTorch's current stream so that the step can be
    captured in a CUDA graph. Without this Warp would use its own stream and a
    capture would record nothing.
    """
    vec = _TORCH_TO_WP_VEC[positions.dtype]
    with scoped_warp_stream(positions.device):
        _wp_step(
            positions=_wp(positions, vec),
            forces=_wp(forces, vec),
            energy=_wp(energy, wp.float64),
            batch_idx=_wp(batch_idx, wp.int32),
            n_particles=_wp(n_particles, wp.int32),
            x_base=_wp(x_base, vec),
            force_base=_wp(force_base, vec),
            direction=_wp(direction, vec),
            s_history=_wp(s_history, vec),
            y_history=_wp(y_history, vec),
            ys=_wp(ys, wp.float64),
            yy=_wp(yy, wp.float64),
            alpha_hist=_wp(alpha_hist, wp.float64),
            beta_hist=_wp(beta_hist, wp.float64),
            ss=_wp(ss, wp.float64),
            f_base=_wp(f_base, wp.float64),
            gg=_wp(gg, wp.float64),
            gd=_wp(gd, wp.float64),
            fmax=_wp(fmax, wp.float64),
            frms_sq=_wp(frms_sq, wp.float64),
            smax=_wp(smax, wp.float64),
            d0=_wp(d0, wp.float64),
            dmax=_wp(dmax, wp.float64),
            dquad=_wp(dquad, wp.float64),
            alpha_step=_wp(alpha_step, wp.float64),
            status=_wp(status, wp.int32),
            iteration=_wp(iteration, wp.int32),
            end=_wp(end, wp.int32),
            n_loop=_wp(n_loop, wp.int32),
            ls_trials=_wp(ls_trials, wp.int32),
            history_count=_wp(history_count, wp.int32),
            force_tol=force_tol,
            rms_tol=rms_tol,
            stress_tol=stress_tol,
            ftol=ftol,
            wolfe=wolfe,
            step_scale_down=step_scale_down,
            step_scale_up=step_scale_up,
            min_step=min_step,
            max_step=max_step,
            max_ls_iter=max_ls_iter,
            maxstep=maxstep,
            curvature_eps=curvature_eps,
            compute_reductions=compute_reductions,
        )


register_noop_fake(_lbfgs_step_op)

# The operator signature is written out by hand, so pin its parameter names to
# the state container. Registration already rejects a name in ``mutates_args``
# that does not exist, but only this catches a *reordering*, which would
# silently swap two tensors.
_STATE_SLICE = slice(5, 5 + len(LBFGSState._fields))
_op_state_params = tuple(inspect.signature(_lbfgs_step_op).parameters)[_STATE_SLICE]
if _op_state_params != LBFGSState._fields:
    # Raised rather than asserted so the guard survives `python -O`.
    raise RuntimeError(
        "LBFGSState fields and _lbfgs_step_op parameters have diverged:\n"
        f"  state:    {LBFGSState._fields}\n"
        f"  operator: {_op_state_params}"
    )


def lbfgs_allocate_state(
    num_dofs: int,
    num_systems: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    history_size: int = 6,
) -> LBFGSState:
    """Allocate a ready-to-use optimizer state.

    Field initialization is exactly:

    - ``iteration`` to ``-1``, the "never evaluated" marker;
    - ``alpha_step`` to ``1.0``;
    - ``status`` to ``LBFGS_NEED_EVAL``;
    - everything else to zero.

    Only those three are non-zero, which is why neither this nor
    :func:`lbfgs_reset` can be a blanket ``zero_()``.

    Parameters
    ----------
    num_dofs : int
        Degrees of freedom the optimizer moves. For coordinate-only relaxation
        this is the atom count.
    num_systems : int
        Number of independent systems in the batch.
    dtype : torch.dtype
        Coordinate precision, ``float32`` or ``float64``. Per-system scalars
        are always float64 regardless, because the Armijo test compares a
        difference of total energies.
    history_size : int, optional
        Number of stored curvature pairs. Memory is dominated by
        ``2 * history_size`` arrays of ``num_dofs`` vectors; 3 to 7 is typical.

    Returns
    -------
    LBFGSState
        Freshly reset state.
    """
    if dtype not in _TORCH_TO_WP_VEC:
        raise ValueError(f"dtype must be float32 or float64; got {dtype}")
    if history_size < 1:
        raise ValueError(f"history_size must be >= 1; got {history_size}")
    fields = {
        "x_base": torch.zeros(num_dofs, 3, dtype=dtype, device=device),
        "force_base": torch.zeros(num_dofs, 3, dtype=dtype, device=device),
        "direction": torch.zeros(num_dofs, 3, dtype=dtype, device=device),
        "s_history": torch.zeros(history_size, num_dofs, 3, dtype=dtype, device=device),
        "y_history": torch.zeros(history_size, num_dofs, 3, dtype=dtype, device=device),
        "ys": torch.zeros(
            history_size, num_systems, dtype=torch.float64, device=device
        ),
        "yy": torch.zeros(
            history_size, num_systems, dtype=torch.float64, device=device
        ),
        "alpha_hist": torch.zeros(
            history_size, num_systems, dtype=torch.float64, device=device
        ),
        "beta_hist": torch.zeros(
            history_size, num_systems, dtype=torch.float64, device=device
        ),
        "ss": torch.zeros(num_systems, dtype=torch.float64, device=device),
        "f_base": torch.zeros(num_systems, dtype=torch.float64, device=device),
        "gg": torch.zeros(num_systems, dtype=torch.float64, device=device),
        "gd": torch.zeros(num_systems, dtype=torch.float64, device=device),
        "fmax": torch.zeros(num_systems, dtype=torch.float64, device=device),
        "frms_sq": torch.zeros(num_systems, dtype=torch.float64, device=device),
        "smax": torch.zeros(num_systems, dtype=torch.float64, device=device),
        "d0": torch.zeros(num_systems, dtype=torch.float64, device=device),
        "dmax": torch.zeros(num_systems, dtype=torch.float64, device=device),
        "dquad": torch.zeros(num_systems, dtype=torch.float64, device=device),
        "alpha_step": torch.zeros(num_systems, dtype=torch.float64, device=device),
        "status": torch.zeros(num_systems, dtype=torch.int32, device=device),
        "iteration": torch.zeros(num_systems, dtype=torch.int32, device=device),
        "end": torch.zeros(num_systems, dtype=torch.int32, device=device),
        "n_loop": torch.zeros(num_systems, dtype=torch.int32, device=device),
        "ls_trials": torch.zeros(num_systems, dtype=torch.int32, device=device),
        "history_count": torch.zeros(num_systems, dtype=torch.int32, device=device),
    }
    state = LBFGSState(**fields)
    lbfgs_reset(state)
    return state


def lbfgs_reset(state: LBFGSState) -> None:
    """Return a state to its pre-first-step condition, in place.

    Call this to discard the accumulated history, for instance after changing
    the potential or moving the atoms behind the optimizer's back.
    """
    _wp_reset(
        x_base=_wp(state.x_base, _TORCH_TO_WP_VEC[state.x_base.dtype]),
        force_base=_wp(state.force_base, _TORCH_TO_WP_VEC[state.x_base.dtype]),
        direction=_wp(state.direction, _TORCH_TO_WP_VEC[state.x_base.dtype]),
        s_history=_wp(state.s_history, _TORCH_TO_WP_VEC[state.x_base.dtype]),
        y_history=_wp(state.y_history, _TORCH_TO_WP_VEC[state.x_base.dtype]),
        ys=_wp(state.ys, wp.float64),
        yy=_wp(state.yy, wp.float64),
        alpha_hist=_wp(state.alpha_hist, wp.float64),
        beta_hist=_wp(state.beta_hist, wp.float64),
        ss=_wp(state.ss, wp.float64),
        f_base=_wp(state.f_base, wp.float64),
        gg=_wp(state.gg, wp.float64),
        gd=_wp(state.gd, wp.float64),
        fmax=_wp(state.fmax, wp.float64),
        frms_sq=_wp(state.frms_sq, wp.float64),
        smax=_wp(state.smax, wp.float64),
        d0=_wp(state.d0, wp.float64),
        dmax=_wp(state.dmax, wp.float64),
        dquad=_wp(state.dquad, wp.float64),
        alpha_step=_wp(state.alpha_step, wp.float64),
        status=_wp(state.status, wp.int32),
        iteration=_wp(state.iteration, wp.int32),
        end=_wp(state.end, wp.int32),
        n_loop=_wp(state.n_loop, wp.int32),
        ls_trials=_wp(state.ls_trials, wp.int32),
        history_count=_wp(state.history_count, wp.int32),
    )


def lbfgs_reduce_energy(
    per_atom_energy: torch.Tensor,
    batch_idx: torch.Tensor,
    energy: torch.Tensor,
) -> None:
    """Sum per-atom energies into per-system totals, accumulating in float64.

    Use this when your model returns per-atom energies, which is the common
    case and the recommended one. The Armijo test compares a difference of
    *total* energies, and at ``E ~ -1e4 eV`` a float32 total is already rounded
    to about ``1e-3 eV`` before it reaches the optimizer -- enough to make the
    line search unreliable near convergence. Summing in float64 avoids that;
    the per-atom values may be single precision.

    Parameters
    ----------
    per_atom_energy : torch.Tensor, shape (num_atoms,)
    batch_idx : torch.Tensor, shape (num_atoms,), dtype int32
        Sorted system index per atom.
    energy : torch.Tensor, shape (num_systems,), dtype float64
        OUTPUT. Zeroed internally.
    """
    if energy.dtype != torch.float64:
        raise ValueError(f"energy must be float64; got {energy.dtype}")
    wp_dtype = wp.float32 if per_atom_energy.dtype == torch.float32 else wp.float64
    _wp_reduce_energy(
        _wp(per_atom_energy, wp_dtype),
        _wp(batch_idx, wp.int32),
        _wp(energy, wp.float64),
    )


def lbfgs_step_coord(
    positions: torch.Tensor,
    state: LBFGSState,
    forces: torch.Tensor,
    energy: torch.Tensor,
    batch_idx: torch.Tensor,
    n_particles: torch.Tensor,
    *,
    force_tol: float = 0.05,
    rms_tol: float = 0.0,
    stress_tol: float = 0.0,
    ftol: float = 1e-4,
    wolfe: float = 0.9,
    step_scale_down: float = 0.5,
    step_scale_up: float = 2.1,
    min_step: float = 1e-20,
    max_step: float = 1e20,
    max_ls_iter: int = 40,
    maxstep: float = 0.2,
    curvature_eps: float = 1e-10,
    compute_reductions: bool = True,
) -> None:
    """Advance one batched L-BFGS step, consuming one force evaluation.

    Mutates ``positions`` and every field of ``state`` in place. Progress is
    reported through ``state.status``; see the module docstring.

    Parameters
    ----------
    positions : torch.Tensor, shape (num_atoms, 3)
        Current geometry. Advanced to the next trial point.
    state : LBFGSState
        From :func:`lbfgs_allocate_state`.
    forces : torch.Tensor, shape (num_atoms, 3)
        Forces at ``positions``. Forces, not gradients.
    energy : torch.Tensor, shape (num_systems,), dtype float64
        Per-system total energy. Use :func:`lbfgs_reduce_energy` if your model
        returns per-atom energies.
    batch_idx : torch.Tensor, shape (num_atoms,), dtype int32
        Sorted system index per atom.
    n_particles : torch.Tensor, shape (num_systems,), dtype int32
        Atom count per system, for the optional RMS convergence criterion.
    force_tol : float, optional
        Convergence threshold on the largest per-atom force magnitude, in the
        force units you supplied. Zero disables it.
    rms_tol, stress_tol : float, optional
        Additional convergence thresholds, disabled by default. All enabled
        criteria must hold.
    maxstep : float, optional
        Largest distance any atom may move in one step. Zero disables the
        trust region.
    max_ls_iter : int, optional
        Trials allowed in one line search before the history is discarded and
        the search restarts from steepest descent.
    curvature_eps : float, optional
        Relative threshold below which a curvature pair is judged unusable and
        discarded.

    Raises
    ------
    ValueError
        If dtypes or shapes are inconsistent.

    See Also
    --------
    lbfgs_step_extended : the same operator on caller-packed degrees of freedom.
    """
    _validate(positions, forces, energy, batch_idx, n_particles, state)
    _lbfgs_step_op(
        positions,
        forces,
        energy,
        batch_idx,
        n_particles,
        *state,
        force_tol,
        rms_tol,
        stress_tol,
        ftol,
        wolfe,
        step_scale_down,
        step_scale_up,
        min_step,
        max_step,
        max_ls_iter,
        maxstep,
        curvature_eps,
        compute_reductions,
    )


def lbfgs_step_extended(
    ext_positions: torch.Tensor,
    state: LBFGSState,
    ext_forces: torch.Tensor,
    energy: torch.Tensor,
    ext_batch_idx: torch.Tensor,
    n_particles: torch.Tensor,
    **kwargs,
) -> None:
    """Advance one step on caller-packed degrees of freedom.

    Identical to :func:`lbfgs_step_coord` and backed by the same registered
    operator; only the meaning of the arrays differs. Use it when you have
    packed extra degrees of freedom alongside the atoms, as variable-cell
    relaxation does.

    Note that convergence is evaluated on whatever ``ext_forces`` contains. If
    those are not Cartesian atomic forces, ``force_tol`` will not mean a force
    per atom, and you should drive convergence yourself from ``state.status``
    and your own reductions.
    """
    lbfgs_step_coord(
        ext_positions, state, ext_forces, energy, ext_batch_idx, n_particles, **kwargs
    )


def _validate(positions, forces, energy, batch_idx, n_particles, state) -> None:
    """Check the shapes and dtypes that would otherwise fail deep in a kernel."""
    num_dofs = positions.shape[0]
    if positions.dtype not in _TORCH_TO_WP_VEC:
        raise ValueError(f"positions must be float32 or float64; got {positions.dtype}")
    if forces.shape != positions.shape:
        raise ValueError(
            f"forces shape {tuple(forces.shape)} != positions shape "
            f"{tuple(positions.shape)}"
        )
    if forces.dtype != positions.dtype:
        raise ValueError(
            f"forces dtype {forces.dtype} != positions dtype {positions.dtype}"
        )
    if batch_idx.shape[0] != num_dofs:
        raise ValueError(
            f"batch_idx length {batch_idx.shape[0]} != positions length {num_dofs}"
        )
    if energy.dtype != torch.float64:
        raise ValueError(f"energy must be float64; got {energy.dtype}")
    num_systems = state.status.shape[0]
    if energy.shape[0] != num_systems:
        raise ValueError(
            f"energy length {energy.shape[0]} != number of systems {num_systems}"
        )
    if n_particles.shape[0] != num_systems:
        raise ValueError(
            f"n_particles length {n_particles.shape[0]} != number of systems "
            f"{num_systems}"
        )
    if state.s_history.shape[1] != num_dofs:
        raise ValueError(
            f"history buffers hold {state.s_history.shape[1]} degrees of freedom, "
            f"but positions has {num_dofs}"
        )


# =============================================================================
# Variable-cell relaxation
# =============================================================================

#: Tensors the variable-cell step writes to. ``ext_batch_idx`` and
#: ``ext_atom_ptr`` are topology, written once at allocation, so they are
#: inputs rather than mutated state.
_CELL_MUTATED = (
    ("positions", "cell")
    + LBFGSState._fields
    + (
        "ref_cell",
        "ref_cell_inv",
        "kappa",
        "phi",
        "phi_inv",
        "d_phi",
        "cell_dof_a",
        "cell_dof_b",
        "cell_force_a",
        "cell_force_b",
        "ext_positions",
        "ext_forces",
    )
)


@torch_custom_op("nvalchemiops::lbfgs_step_coord_cell", mutates_args=_CELL_MUTATED)
def _lbfgs_step_coord_cell_op(
    forces: torch.Tensor,
    stress: torch.Tensor,
    energy: torch.Tensor,
    batch_idx: torch.Tensor,
    n_particles: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    x_base: torch.Tensor,
    force_base: torch.Tensor,
    direction: torch.Tensor,
    s_history: torch.Tensor,
    y_history: torch.Tensor,
    ys: torch.Tensor,
    yy: torch.Tensor,
    alpha_hist: torch.Tensor,
    beta_hist: torch.Tensor,
    ss: torch.Tensor,
    f_base: torch.Tensor,
    gg: torch.Tensor,
    gd: torch.Tensor,
    fmax: torch.Tensor,
    frms_sq: torch.Tensor,
    smax: torch.Tensor,
    d0: torch.Tensor,
    dmax: torch.Tensor,
    dquad: torch.Tensor,
    alpha_step: torch.Tensor,
    status: torch.Tensor,
    iteration: torch.Tensor,
    end: torch.Tensor,
    n_loop: torch.Tensor,
    ls_trials: torch.Tensor,
    history_count: torch.Tensor,
    ref_cell: torch.Tensor,
    ref_cell_inv: torch.Tensor,
    kappa: torch.Tensor,
    ext_batch_idx: torch.Tensor,
    ext_atom_ptr: torch.Tensor,
    phi: torch.Tensor,
    phi_inv: torch.Tensor,
    d_phi: torch.Tensor,
    cell_dof_a: torch.Tensor,
    cell_dof_b: torch.Tensor,
    cell_force_a: torch.Tensor,
    cell_force_b: torch.Tensor,
    ext_positions: torch.Tensor,
    ext_forces: torch.Tensor,
    force_tol: float,
    rms_tol: float,
    stress_tol: float,
    ftol: float,
    wolfe: float,
    step_scale_down: float,
    step_scale_up: float,
    min_step: float,
    max_step: float,
    max_ls_iter: int,
    maxstep: float,
    curvature_eps: float,
) -> None:
    """Run one registered variable-cell L-BFGS step."""
    vec = _TORCH_TO_WP_VEC[positions.dtype]
    mat = _TORCH_TO_WP_MAT[positions.dtype]
    with scoped_warp_stream(positions.device):
        _wp_step_cell(
            positions=_wp(positions, vec),
            forces=_wp(forces, vec),
            cell=_wp(cell, mat),
            stress=_wp(stress, mat),
            energy=_wp(energy, wp.float64),
            batch_idx=_wp(batch_idx, wp.int32),
            n_particles=_wp(n_particles, wp.int32),
            cell_state=LBFGSCellState(
                ref_cell=_wp(ref_cell, mat),
                ref_cell_inv=_wp(ref_cell_inv, mat),
                kappa=_wp(kappa, wp.float64),
                ext_batch_idx=_wp(ext_batch_idx, wp.int32),
                ext_atom_ptr=_wp(ext_atom_ptr, wp.int32),
                phi=_wp(phi, mat),
                phi_inv=_wp(phi_inv, mat),
                d_phi=_wp(d_phi, mat),
                cell_dof_a=_wp(cell_dof_a, vec),
                cell_dof_b=_wp(cell_dof_b, vec),
                cell_force_a=_wp(cell_force_a, vec),
                cell_force_b=_wp(cell_force_b, vec),
                ext_positions=_wp(ext_positions, vec),
                ext_forces=_wp(ext_forces, vec),
            ),
            state=LBFGSState(
                x_base=_wp(x_base, vec),
                force_base=_wp(force_base, vec),
                direction=_wp(direction, vec),
                s_history=_wp(s_history, vec),
                y_history=_wp(y_history, vec),
                ys=_wp(ys, wp.float64),
                yy=_wp(yy, wp.float64),
                alpha_hist=_wp(alpha_hist, wp.float64),
                beta_hist=_wp(beta_hist, wp.float64),
                ss=_wp(ss, wp.float64),
                f_base=_wp(f_base, wp.float64),
                gg=_wp(gg, wp.float64),
                gd=_wp(gd, wp.float64),
                fmax=_wp(fmax, wp.float64),
                frms_sq=_wp(frms_sq, wp.float64),
                smax=_wp(smax, wp.float64),
                d0=_wp(d0, wp.float64),
                dmax=_wp(dmax, wp.float64),
                dquad=_wp(dquad, wp.float64),
                alpha_step=_wp(alpha_step, wp.float64),
                status=_wp(status, wp.int32),
                iteration=_wp(iteration, wp.int32),
                end=_wp(end, wp.int32),
                n_loop=_wp(n_loop, wp.int32),
                ls_trials=_wp(ls_trials, wp.int32),
                history_count=_wp(history_count, wp.int32),
            ),
            force_tol=force_tol,
            rms_tol=rms_tol,
            stress_tol=stress_tol,
            ftol=ftol,
            wolfe=wolfe,
            step_scale_down=step_scale_down,
            step_scale_up=step_scale_up,
            min_step=min_step,
            max_step=max_step,
            max_ls_iter=max_ls_iter,
            maxstep=maxstep,
            curvature_eps=curvature_eps,
        )


register_noop_fake(_lbfgs_step_coord_cell_op)

_CELL_STATE_SLICE = slice(
    7 + len(LBFGSState._fields),
    7 + len(LBFGSState._fields) + len(LBFGSCellState._fields),
)
_cell_params = tuple(inspect.signature(_lbfgs_step_coord_cell_op).parameters)
if _cell_params[7 : 7 + len(LBFGSState._fields)] != LBFGSState._fields:
    raise RuntimeError("LBFGSState fields and the cell operator have diverged")
if _cell_params[_CELL_STATE_SLICE] != LBFGSCellState._fields:
    raise RuntimeError("LBFGSCellState fields and the cell operator have diverged")


def lbfgs_allocate_cell_state(
    num_atoms: int,
    num_systems: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    cell_force_scale: float | None = None,
) -> LBFGSCellState:
    """Allocate the working arrays for variable-cell relaxation.

    Assumes every system has the same atom count, ``num_atoms // num_systems``.

    The reference cell is left zeroed; capture it with
    :func:`lbfgs_set_reference_cell` once the starting geometry is known.
    Everything else that depends only on topology -- the coordinate scaling and
    the packed index arrays -- is filled here.

    Parameters
    ----------
    cell_force_scale : float, optional
        Multiplier on the atom count giving the cell coordinate scaling.
        Defaults to ``1 / atoms_per_system``, which makes the scaling one and
        puts cell and atom degrees of freedom on a comparable footing. Raise it
        to make the cell move less per step.

    Returns
    -------
    LBFGSCellState
    """
    if dtype not in _TORCH_TO_WP_VEC:
        raise ValueError(f"dtype must be float32 or float64; got {dtype}")
    if num_atoms % num_systems:
        raise ValueError(
            f"num_atoms {num_atoms} is not divisible by num_systems {num_systems}; "
            "ragged batches are not supported by this allocator"
        )
    per_system = num_atoms // num_systems
    num_ext = num_atoms + 2 * num_systems
    if cell_force_scale is None:
        cell_force_scale = 1.0 / per_system

    ext_atom_ptr = torch.tensor(
        [s * per_system + 2 * s for s in range(num_systems + 1)],
        dtype=torch.int32,
        device=device,
    )
    ext_batch_idx = torch.repeat_interleave(
        torch.arange(num_systems, dtype=torch.int32, device=device), per_system + 2
    )
    fields = {
        "ref_cell": torch.zeros(num_systems, 3, 3, dtype=dtype, device=device),
        "ref_cell_inv": torch.zeros(num_systems, 3, 3, dtype=dtype, device=device),
        "kappa": torch.zeros(num_systems, dtype=dtype, device=device),
        "ext_batch_idx": ext_batch_idx,
        "ext_atom_ptr": ext_atom_ptr,
        "phi": torch.zeros(num_systems, 3, 3, dtype=dtype, device=device),
        "phi_inv": torch.zeros(num_systems, 3, 3, dtype=dtype, device=device),
        "d_phi": torch.zeros(num_systems, 3, 3, dtype=dtype, device=device),
        "cell_dof_a": torch.zeros(num_systems, 3, dtype=dtype, device=device),
        "cell_dof_b": torch.zeros(num_systems, 3, dtype=dtype, device=device),
        "cell_force_a": torch.zeros(num_systems, 3, dtype=dtype, device=device),
        "cell_force_b": torch.zeros(num_systems, 3, dtype=dtype, device=device),
        "ext_positions": torch.zeros(num_ext, 3, dtype=dtype, device=device),
        "ext_forces": torch.zeros(num_ext, 3, dtype=dtype, device=device),
    }
    cell_state = LBFGSCellState(**fields)
    n_atoms_per_system = torch.full(
        (num_systems,), per_system, dtype=torch.int32, device=device
    )
    _wp_cell_kappa(
        _wp(n_atoms_per_system, wp.int32),
        _wp(cell_state.kappa, wp.float32 if dtype == torch.float32 else wp.float64),
        cell_force_scale=cell_force_scale,
    )
    return cell_state


def lbfgs_set_reference_cell(cell: torch.Tensor, cell_state: LBFGSCellState) -> None:
    """Capture the reference cell that defines the variable-cell chart.

    Coordinates are measured relative to a cell held fixed for the whole
    relaxation, which is what makes history pairs from different iterations
    comparable. Call this once, before the first step.

    Calling it again re-references the chart and invalidates every stored
    curvature pair, so follow it with :func:`lbfgs_reset`.
    """
    mat = _TORCH_TO_WP_MAT[cell.dtype]
    _wp_set_reference_cell(
        _wp(cell, mat),
        _wp(cell_state.ref_cell, mat),
        _wp(cell_state.ref_cell_inv, mat),
    )


def lbfgs_step_coord_cell(
    positions: torch.Tensor,
    cell: torch.Tensor,
    state: LBFGSState,
    cell_state: LBFGSCellState,
    forces: torch.Tensor,
    stress: torch.Tensor,
    energy: torch.Tensor,
    batch_idx: torch.Tensor,
    n_particles: torch.Tensor,
    *,
    force_tol: float = 0.05,
    rms_tol: float = 0.0,
    stress_tol: float = 0.0,
    ftol: float = 1e-4,
    wolfe: float = 0.9,
    step_scale_down: float = 0.5,
    step_scale_up: float = 2.1,
    min_step: float = 1e-20,
    max_step: float = 1e20,
    max_ls_iter: int = 40,
    maxstep: float = 0.2,
    curvature_eps: float = 1e-10,
) -> None:
    """Advance one variable-cell step, relaxing coordinates and cell together.

    Mutates ``positions``, ``cell`` and both state objects in place. Consumes
    exactly one energy/force/stress evaluation, and reports progress through
    ``state.status`` exactly like the coordinate-only path.

    ``state`` must be sized for ``num_atoms + 2 * num_systems`` degrees of
    freedom, since the cell contributes two entries per system.

    Parameters
    ----------
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Lattice vectors as columns. Kept lower-triangular, so the cell cannot
        drift into a rotation.
    stress : torch.Tensor, shape (num_systems, 3, 3)
        Cauchy stress. Drives the cell degrees of freedom and is what
        ``stress_tol`` is compared against; the packed cell force cannot be
        used for that, since it carries units of energy rather than stress.
    force_tol, rms_tol, stress_tol : float, optional
        Convergence thresholds, always evaluated on the Cartesian forces and
        the stress, so ``force_tol`` keeps its meaning as a force per atom
        however far the cell deforms.
    maxstep : float, optional
        Largest Cartesian distance an atom may move in one step.

    Examples
    --------
    >>> cell_state = lbfgs_allocate_cell_state(
    ...     num_atoms, num_systems, dtype=positions.dtype, device=positions.device
    ... )
    >>> lbfgs_set_reference_cell(cell, cell_state)
    >>> state = lbfgs_allocate_state(
    ...     num_atoms + 2 * num_systems, num_systems,
    ...     dtype=positions.dtype, device=positions.device,
    ... )

    See Also
    --------
    lbfgs_set_reference_cell : must be called first.
    lbfgs_step_coord : the coordinate-only equivalent.
    """
    _validate_cell(positions, forces, cell, stress, energy, batch_idx, state)
    _lbfgs_step_coord_cell_op(
        forces,
        stress,
        energy,
        batch_idx,
        n_particles,
        positions,
        cell,
        *state,
        *cell_state,
        force_tol,
        rms_tol,
        stress_tol,
        ftol,
        wolfe,
        step_scale_down,
        step_scale_up,
        min_step,
        max_step,
        max_ls_iter,
        maxstep,
        curvature_eps,
    )


def _validate_cell(positions, forces, cell, stress, energy, batch_idx, state) -> None:
    """Check the variable-cell shapes that would otherwise fail in a kernel."""
    num_atoms = positions.shape[0]
    num_systems = state.status.shape[0]
    if positions.dtype not in _TORCH_TO_WP_VEC:
        raise ValueError(f"positions must be float32 or float64; got {positions.dtype}")
    if forces.shape != positions.shape:
        raise ValueError(
            f"forces shape {tuple(forces.shape)} != positions shape "
            f"{tuple(positions.shape)}"
        )
    if cell.shape != (num_systems, 3, 3):
        raise ValueError(
            f"cell must have shape ({num_systems}, 3, 3); got {tuple(cell.shape)}"
        )
    if stress.shape != cell.shape:
        raise ValueError(
            f"stress shape {tuple(stress.shape)} != cell shape {tuple(cell.shape)}"
        )
    if cell.dtype != positions.dtype or stress.dtype != positions.dtype:
        raise ValueError("cell and stress must share the dtype of positions")
    if energy.dtype != torch.float64:
        raise ValueError(f"energy must be float64; got {energy.dtype}")
    if batch_idx.shape[0] != num_atoms:
        raise ValueError(
            f"batch_idx length {batch_idx.shape[0]} != number of atoms {num_atoms}"
        )
    expected = num_atoms + 2 * num_systems
    if state.s_history.shape[1] != expected:
        raise ValueError(
            f"state is sized for {state.s_history.shape[1]} degrees of freedom, but "
            f"the variable-cell path needs {expected} "
            f"(num_atoms + 2 * num_systems)"
        )
