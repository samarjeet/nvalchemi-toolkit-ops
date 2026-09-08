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

"""JAX bindings for the batched L-BFGS geometry optimizer.

L-BFGS reaches a given force tolerance in far fewer energy/force evaluations
than the FIRE optimizers, which is the cost that dominates relaxation with a
machine-learned potential.

JAX arrays are immutable, so unlike the PyTorch binding these entry points
**return** a new state rather than mutating one::

    import functools, jax
    from nvalchemiops.jax.lbfgs import (
        LBFGS_NEED_EVAL, lbfgs_allocate_state, lbfgs_step_coord,
    )

    state = lbfgs_allocate_state(num_dofs, num_systems, dtype=jnp.float64)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def relax_step(positions, state, forces, energy):
        return lbfgs_step_coord(
            positions, state, forces, energy, batch_idx, n_particles,
        )

    while True:
        energy, forces = model(positions)
        positions, state = relax_step(positions, state, forces, energy)
        if not (state.status == LBFGS_NEED_EVAL).any():
            break

Donation and pointer stability
------------------------------
Every mutable array is declared as an input-output alias, so XLA may reuse each
input buffer for the matching output. Two things follow, and both matter for
performance rather than correctness:

- **Donate the state.** ``jax.jit(donate_argnums=...)`` over ``positions`` and
  ``state`` lets the buffers round-trip at stable addresses. Without donation
  JAX copies, which doubles peak memory and moves the pointers.
- **Keep the topology out of the state.** ``batch_idx`` and ``n_particles``
  never change, so close over them rather than threading them through as
  donated leaves.

Under ``GraphMode.WARP`` the step is captured and replayed as a CUDA graph. The
capture is keyed on the input buffer addresses, so a fresh ``forces`` array each
step produces a small working set of graphs rather than one; measurements on a
six-atom system settle at four or five captures and stay there, well inside the
default cache. If you see the count grow without bound, pass
``graph_mode="warp_staged"``, which keys the capture on the call instead and
patches the changing buffers in, at the cost of one copy per staged array.

Scalars are baked into the compiled call, so changing ``force_tol`` or
``maxstep`` between steps triggers a recompilation. Hold them fixed for the
duration of a relaxation.

These operations are **not differentiable**. ``jax.grad`` through a step will
fail rather than return a silently wrong answer. Callers that relax and then
differentiate the relaxed energy should stop the gradient at the relaxed
positions themselves.

See Also
--------
nvalchemiops.dynamics.optimizers.lbfgs : the underlying Warp implementation,
    which documents the algorithm, the sign convention and the precision policy.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from warp.jax_experimental import GraphMode, jax_callable

from nvalchemiops.dynamics.optimizers.lbfgs import (
    LBFGS_CONVERGED,
    LBFGS_LS_FAILED,
    LBFGS_NEED_EVAL,
    LBFGSCellState,
    LBFGSState,
)
from nvalchemiops.dynamics.optimizers.lbfgs import lbfgs_cell_kappa as _warp_cell_kappa
from nvalchemiops.dynamics.optimizers.lbfgs import lbfgs_step as _warp_step
from nvalchemiops.dynamics.optimizers.lbfgs import (
    lbfgs_step_coord_cell as _warp_step_cell,
)

__all__ = [
    "LBFGSCellState",
    "LBFGSState",
    "LBFGS_CONVERGED",
    "LBFGS_LS_FAILED",
    "LBFGS_NEED_EVAL",
    "lbfgs_allocate_state",
    "lbfgs_allocate_cell_state",
    "lbfgs_converged",
    "lbfgs_set_reference_cell",
    "lbfgs_step_coord",
    "lbfgs_step_coord_cell",
]

#: The mutable arrays, in the order the callable takes them. Derived from the
#: shared state container so the two cannot drift apart.
_LBFGS_IN_OUT_ARGS: tuple[str, ...] = ("positions",) + LBFGSState._fields

_GRAPH_MODES = {
    "none": GraphMode.NONE,
    "warp": GraphMode.WARP,
    "warp_staged": GraphMode.WARP_STAGED,
}


def _lbfgs_body_f32(
    forces: wp.array(dtype=wp.vec3f),
    energy: wp.array(dtype=wp.float64),
    batch_idx: wp.array(dtype=wp.int32),
    n_particles: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3f),
    x_base: wp.array(dtype=wp.vec3f),
    force_base: wp.array(dtype=wp.vec3f),
    direction: wp.array(dtype=wp.vec3f),
    s_history: wp.array(dtype=wp.vec3f, ndim=2),
    y_history: wp.array(dtype=wp.vec3f, ndim=2),
    ys: wp.array(dtype=wp.float64, ndim=2),
    yy: wp.array(dtype=wp.float64, ndim=2),
    alpha_hist: wp.array(dtype=wp.float64, ndim=2),
    beta_hist: wp.array(dtype=wp.float64, ndim=2),
    ss: wp.array(dtype=wp.float64),
    f_base: wp.array(dtype=wp.float64),
    gg: wp.array(dtype=wp.float64),
    gd: wp.array(dtype=wp.float64),
    fmax: wp.array(dtype=wp.float64),
    frms_sq: wp.array(dtype=wp.float64),
    smax: wp.array(dtype=wp.float64),
    d0: wp.array(dtype=wp.float64),
    dmax: wp.array(dtype=wp.float64),
    dquad: wp.array(dtype=wp.float64),
    alpha_step: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    iteration: wp.array(dtype=wp.int32),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    ls_trials: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    force_tol: wp.float64,
    rms_tol: wp.float64,
    stress_tol: wp.float64,
    ftol: wp.float64,
    wolfe: wp.float64,
    step_scale_down: wp.float64,
    step_scale_up: wp.float64,
    min_step: wp.float64,
    max_step: wp.float64,
    max_ls_iter: wp.int32,
    maxstep: wp.float64,
    curvature_eps: wp.float64,
) -> None:
    """Advance one L-BFGS step on f32 coordinates.

    Every mutable array is listed before the static scalars and named in
    ``_LBFGS_IN_OUT_ARGS``, so XLA aliases each one to the matching output and
    the step can be written as if it mutated in place.
    """
    _warp_step(
        positions=positions,
        forces=forces,
        energy=energy,
        batch_idx=batch_idx,
        n_particles=n_particles,
        x_base=x_base,
        force_base=force_base,
        direction=direction,
        s_history=s_history,
        y_history=y_history,
        ys=ys,
        yy=yy,
        alpha_hist=alpha_hist,
        beta_hist=beta_hist,
        ss=ss,
        f_base=f_base,
        gg=gg,
        gd=gd,
        fmax=fmax,
        frms_sq=frms_sq,
        smax=smax,
        d0=d0,
        dmax=dmax,
        dquad=dquad,
        alpha_step=alpha_step,
        status=status,
        iteration=iteration,
        end=end,
        n_loop=n_loop,
        ls_trials=ls_trials,
        history_count=history_count,
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


def _lbfgs_body_f64(
    forces: wp.array(dtype=wp.vec3d),
    energy: wp.array(dtype=wp.float64),
    batch_idx: wp.array(dtype=wp.int32),
    n_particles: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3d),
    x_base: wp.array(dtype=wp.vec3d),
    force_base: wp.array(dtype=wp.vec3d),
    direction: wp.array(dtype=wp.vec3d),
    s_history: wp.array(dtype=wp.vec3d, ndim=2),
    y_history: wp.array(dtype=wp.vec3d, ndim=2),
    ys: wp.array(dtype=wp.float64, ndim=2),
    yy: wp.array(dtype=wp.float64, ndim=2),
    alpha_hist: wp.array(dtype=wp.float64, ndim=2),
    beta_hist: wp.array(dtype=wp.float64, ndim=2),
    ss: wp.array(dtype=wp.float64),
    f_base: wp.array(dtype=wp.float64),
    gg: wp.array(dtype=wp.float64),
    gd: wp.array(dtype=wp.float64),
    fmax: wp.array(dtype=wp.float64),
    frms_sq: wp.array(dtype=wp.float64),
    smax: wp.array(dtype=wp.float64),
    d0: wp.array(dtype=wp.float64),
    dmax: wp.array(dtype=wp.float64),
    dquad: wp.array(dtype=wp.float64),
    alpha_step: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    iteration: wp.array(dtype=wp.int32),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    ls_trials: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    force_tol: wp.float64,
    rms_tol: wp.float64,
    stress_tol: wp.float64,
    ftol: wp.float64,
    wolfe: wp.float64,
    step_scale_down: wp.float64,
    step_scale_up: wp.float64,
    min_step: wp.float64,
    max_step: wp.float64,
    max_ls_iter: wp.int32,
    maxstep: wp.float64,
    curvature_eps: wp.float64,
) -> None:
    """Advance one L-BFGS step on f64 coordinates.

    Every mutable array is listed before the static scalars and named in
    ``_LBFGS_IN_OUT_ARGS``, so XLA aliases each one to the matching output and
    the step can be written as if it mutated in place.
    """
    _warp_step(
        positions=positions,
        forces=forces,
        energy=energy,
        batch_idx=batch_idx,
        n_particles=n_particles,
        x_base=x_base,
        force_base=force_base,
        direction=direction,
        s_history=s_history,
        y_history=y_history,
        ys=ys,
        yy=yy,
        alpha_hist=alpha_hist,
        beta_hist=beta_hist,
        ss=ss,
        f_base=f_base,
        gg=gg,
        gd=gd,
        fmax=fmax,
        frms_sq=frms_sq,
        smax=smax,
        d0=d0,
        dmax=dmax,
        dquad=dquad,
        alpha_step=alpha_step,
        status=status,
        iteration=iteration,
        end=end,
        n_loop=n_loop,
        ls_trials=ls_trials,
        history_count=history_count,
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


_BODIES = {jnp.float32: _lbfgs_body_f32, jnp.float64: _lbfgs_body_f64}

# The bodies are written out by hand, so pin their parameter names to the state
# container. A reordering would silently swap two arrays, and neither warp nor
# XLA would notice.
_STATE_SLICE = slice(5, 5 + len(LBFGSState._fields))
for _name, _body in (("f32", _lbfgs_body_f32), ("f64", _lbfgs_body_f64)):
    _params = tuple(inspect.signature(_body).parameters)[_STATE_SLICE]
    if _params != LBFGSState._fields:
        raise RuntimeError(
            f"LBFGSState fields and _lbfgs_body_{_name} parameters have diverged:\n"
            f"  state: {LBFGSState._fields}\n"
            f"  body:  {_params}"
        )

_CALLABLES: dict[tuple, object] = {}


def _get_callable(dtype, graph_mode: str):
    """Return the registered callable for a dtype, building it on first use.

    Registration is deferred so that importing this module does not compile
    anything or touch the GPU.
    """
    key = (jnp.dtype(dtype).type, graph_mode)
    if key not in _CALLABLES:
        body = _BODIES.get(key[0])
        if body is None:
            raise ValueError(
                f"positions must be float32 or float64; got {jnp.dtype(dtype)}"
            )
        if graph_mode not in _GRAPH_MODES:
            raise ValueError(
                f"graph_mode must be one of {sorted(_GRAPH_MODES)}; got {graph_mode!r}"
            )
        kwargs = {}
        if graph_mode == "warp_staged":
            # Stage only the arrays that change identity every step; staging the
            # history buffers would copy megabytes per call for no benefit.
            kwargs["stage_in_argnames"] = ["forces", "energy"]
        _CALLABLES[key] = jax_callable(
            body,
            num_outputs=len(_LBFGS_IN_OUT_ARGS),
            in_out_argnames=list(_LBFGS_IN_OUT_ARGS),
            graph_mode=_GRAPH_MODES[graph_mode],
            **kwargs,
        )
    return _CALLABLES[key]


def lbfgs_allocate_state(
    num_dofs: int,
    num_systems: int,
    *,
    dtype=jnp.float64,
    history_size: int = 6,
) -> LBFGSState:
    """Allocate a state ready for the first step.

    Field initialization is exactly: ``iteration`` to ``-1``, the "never
    evaluated" marker; ``alpha_step`` to ``1.0``; ``status`` to
    ``LBFGS_NEED_EVAL``; everything else to zero. Only those three are
    non-zero.

    There is no separate reset in the JAX binding: in a functional setting
    resetting and allocating are the same operation, so call this again to
    discard the history.

    Parameters
    ----------
    num_dofs : int
        Degrees of freedom the optimizer moves; the atom count for
        coordinate-only relaxation.
    num_systems : int
        Number of independent systems in the batch.
    dtype : optional
        Coordinate precision, ``float32`` or ``float64``. Per-system scalars
        are float64 regardless, because the Armijo test compares a difference
        of total energies. ``float64`` requires ``JAX_ENABLE_X64``.
    history_size : int, optional
        Stored curvature pairs; 3 to 7 is typical. Memory is dominated by
        ``2 * history_size`` arrays of ``num_dofs`` vectors.

    Returns
    -------
    LBFGSState
    """
    dtype = jnp.dtype(dtype).type
    if dtype not in _BODIES:
        raise ValueError(f"dtype must be float32 or float64; got {dtype}")
    if history_size < 1:
        raise ValueError(f"history_size must be >= 1; got {history_size}")
    if dtype is jnp.float64 and jnp.zeros(1, jnp.float64).dtype != jnp.float64:
        raise RuntimeError(
            "float64 requested but JAX x64 is disabled; set JAX_ENABLE_X64=1 "
            "or call jax.config.update('jax_enable_x64', True) before use"
        )
    fields = {
        "x_base": jnp.zeros((num_dofs, 3), dtype),
        "force_base": jnp.zeros((num_dofs, 3), dtype),
        "direction": jnp.zeros((num_dofs, 3), dtype),
        "s_history": jnp.zeros((history_size, num_dofs, 3), dtype),
        "y_history": jnp.zeros((history_size, num_dofs, 3), dtype),
        "ys": jnp.zeros((history_size, num_systems), jnp.float64),
        "yy": jnp.zeros((history_size, num_systems), jnp.float64),
        "alpha_hist": jnp.zeros((history_size, num_systems), jnp.float64),
        "beta_hist": jnp.zeros((history_size, num_systems), jnp.float64),
        "ss": jnp.zeros((num_systems,), jnp.float64),
        "f_base": jnp.zeros((num_systems,), jnp.float64),
        "gg": jnp.zeros((num_systems,), jnp.float64),
        "gd": jnp.zeros((num_systems,), jnp.float64),
        "fmax": jnp.zeros((num_systems,), jnp.float64),
        "frms_sq": jnp.zeros((num_systems,), jnp.float64),
        "smax": jnp.zeros((num_systems,), jnp.float64),
        "d0": jnp.zeros((num_systems,), jnp.float64),
        "dmax": jnp.zeros((num_systems,), jnp.float64),
        "dquad": jnp.zeros((num_systems,), jnp.float64),
        "alpha_step": jnp.ones((num_systems,), jnp.float64),
        "status": jnp.zeros((num_systems,), jnp.int32),
        "iteration": jnp.full((num_systems,), -1, jnp.int32),
        "end": jnp.zeros((num_systems,), jnp.int32),
        "n_loop": jnp.zeros((num_systems,), jnp.int32),
        "ls_trials": jnp.zeros((num_systems,), jnp.int32),
        "history_count": jnp.zeros((num_systems,), jnp.int32),
    }
    return LBFGSState(**fields)


def lbfgs_step_coord(
    positions: jax.Array,
    state: LBFGSState,
    forces: jax.Array,
    energy: jax.Array,
    batch_idx: jax.Array,
    n_particles: jax.Array,
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
    graph_mode: str = "warp",
) -> tuple[jax.Array, LBFGSState]:
    """Advance one batched L-BFGS step, consuming one force evaluation.

    Returns new arrays; nothing is mutated in place. Donate ``positions`` and
    ``state`` so XLA can reuse their buffers.

    Parameters
    ----------
    positions : jax.Array, shape (num_atoms, 3)
        Current geometry.
    state : LBFGSState
        From :func:`lbfgs_allocate_state`.
    forces : jax.Array, shape (num_atoms, 3)
        Forces at ``positions``. Forces, not gradients.
    energy : jax.Array, shape (num_systems,), dtype float64
        Per-system total energy. Sum per-atom energies in float64; a float32
        total is already too coarse for the Armijo test near convergence.
    batch_idx : jax.Array, shape (num_atoms,), dtype int32
        Sorted system index per atom. Static topology, so close over it.
    n_particles : jax.Array, shape (num_systems,), dtype int32
        Atom count per system, for the optional RMS criterion.
    force_tol : float, optional
        Threshold on the largest per-atom force magnitude. Zero disables it.
    rms_tol, stress_tol : float, optional
        Additional criteria, disabled by default; all enabled ones must hold.
    maxstep : float, optional
        Largest distance an atom may move in one step. Zero disables the trust
        region.
    graph_mode : {"warp", "warp_staged", "none"}, optional
        How the step is captured. ``"warp"`` replays a CUDA graph and is the
        default; ``"warp_staged"`` keys the capture on the call rather than on
        buffer addresses, which helps if the capture count grows without bound;
        ``"none"`` disables capture.

    Returns
    -------
    positions, state
        The advanced geometry and the new optimizer state.

    Raises
    ------
    ValueError
        If dtypes or shapes are inconsistent.
    """
    _validate(positions, forces, energy, batch_idx, n_particles, state)
    call = _get_callable(positions.dtype, graph_mode)
    outputs = call(
        forces,
        energy,
        batch_idx,
        n_particles,
        positions,
        *state,
        float(force_tol),
        float(rms_tol),
        float(stress_tol),
        float(ftol),
        float(wolfe),
        float(step_scale_down),
        float(step_scale_up),
        float(min_step),
        float(max_step),
        int(max_ls_iter),
        float(maxstep),
        float(curvature_eps),
    )
    return outputs[0], LBFGSState(*outputs[1:])


def lbfgs_converged(state: LBFGSState) -> jax.Array:
    """Whether every system has finished, as a device-side boolean.

    Reading this back costs a host synchronization, so a caller that wants to
    amortize it can check every few steps instead of every step.
    """
    return jnp.all(state.status != LBFGS_NEED_EVAL)


def _validate(positions, forces, energy, batch_idx, n_particles, state) -> None:
    """Check the shapes and dtypes that would otherwise fail inside the FFI."""
    num_dofs = positions.shape[0]
    if jnp.dtype(positions.dtype).type not in _BODIES:
        raise ValueError(f"positions must be float32 or float64; got {positions.dtype}")
    if forces.shape != positions.shape:
        raise ValueError(
            f"forces shape {forces.shape} != positions shape {positions.shape}"
        )
    if forces.dtype != positions.dtype:
        raise ValueError(
            f"forces dtype {forces.dtype} != positions dtype {positions.dtype}"
        )
    if batch_idx.shape[0] != num_dofs:
        raise ValueError(
            f"batch_idx length {batch_idx.shape[0]} != positions length {num_dofs}"
        )
    if jnp.dtype(energy.dtype) != jnp.dtype(jnp.float64):
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


def _lbfgs_cell_body_f32(
    forces: wp.array(dtype=wp.vec3f),
    stress: wp.array(dtype=wp.mat33f),
    energy: wp.array(dtype=wp.float64),
    batch_idx: wp.array(dtype=wp.int32),
    n_particles: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    x_base: wp.array(dtype=wp.vec3f),
    force_base: wp.array(dtype=wp.vec3f),
    direction: wp.array(dtype=wp.vec3f),
    s_history: wp.array(dtype=wp.vec3f, ndim=2),
    y_history: wp.array(dtype=wp.vec3f, ndim=2),
    ys: wp.array(dtype=wp.float64, ndim=2),
    yy: wp.array(dtype=wp.float64, ndim=2),
    alpha_hist: wp.array(dtype=wp.float64, ndim=2),
    beta_hist: wp.array(dtype=wp.float64, ndim=2),
    ss: wp.array(dtype=wp.float64),
    f_base: wp.array(dtype=wp.float64),
    gg: wp.array(dtype=wp.float64),
    gd: wp.array(dtype=wp.float64),
    fmax: wp.array(dtype=wp.float64),
    frms_sq: wp.array(dtype=wp.float64),
    smax: wp.array(dtype=wp.float64),
    d0: wp.array(dtype=wp.float64),
    dmax: wp.array(dtype=wp.float64),
    dquad: wp.array(dtype=wp.float64),
    alpha_step: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    iteration: wp.array(dtype=wp.int32),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    ls_trials: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    ref_cell: wp.array(dtype=wp.mat33f),
    ref_cell_inv: wp.array(dtype=wp.mat33f),
    kappa: wp.array(dtype=wp.float32),
    ext_batch_idx: wp.array(dtype=wp.int32),
    ext_atom_ptr: wp.array(dtype=wp.int32),
    phi: wp.array(dtype=wp.mat33f),
    phi_inv: wp.array(dtype=wp.mat33f),
    d_phi: wp.array(dtype=wp.mat33f),
    cell_dof_a: wp.array(dtype=wp.vec3f),
    cell_dof_b: wp.array(dtype=wp.vec3f),
    cell_force_a: wp.array(dtype=wp.vec3f),
    cell_force_b: wp.array(dtype=wp.vec3f),
    ext_positions: wp.array(dtype=wp.vec3f),
    ext_forces: wp.array(dtype=wp.vec3f),
    force_tol: wp.float64,
    rms_tol: wp.float64,
    stress_tol: wp.float64,
    ftol: wp.float64,
    wolfe: wp.float64,
    step_scale_down: wp.float64,
    step_scale_up: wp.float64,
    min_step: wp.float64,
    max_step: wp.float64,
    max_ls_iter: wp.int32,
    maxstep: wp.float64,
    curvature_eps: wp.float64,
) -> None:
    """Advance one variable-cell L-BFGS step on f32 coordinates."""
    _warp_step_cell(
        positions=positions,
        forces=forces,
        cell=cell,
        stress=stress,
        energy=energy,
        batch_idx=batch_idx,
        n_particles=n_particles,
        cell_state=LBFGSCellState(
            ref_cell=ref_cell,
            ref_cell_inv=ref_cell_inv,
            kappa=kappa,
            ext_batch_idx=ext_batch_idx,
            ext_atom_ptr=ext_atom_ptr,
            phi=phi,
            phi_inv=phi_inv,
            d_phi=d_phi,
            cell_dof_a=cell_dof_a,
            cell_dof_b=cell_dof_b,
            cell_force_a=cell_force_a,
            cell_force_b=cell_force_b,
            ext_positions=ext_positions,
            ext_forces=ext_forces,
        ),
        state=LBFGSState(
            x_base=x_base,
            force_base=force_base,
            direction=direction,
            s_history=s_history,
            y_history=y_history,
            ys=ys,
            yy=yy,
            alpha_hist=alpha_hist,
            beta_hist=beta_hist,
            ss=ss,
            f_base=f_base,
            gg=gg,
            gd=gd,
            fmax=fmax,
            frms_sq=frms_sq,
            smax=smax,
            d0=d0,
            dmax=dmax,
            dquad=dquad,
            alpha_step=alpha_step,
            status=status,
            iteration=iteration,
            end=end,
            n_loop=n_loop,
            ls_trials=ls_trials,
            history_count=history_count,
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


def _lbfgs_cell_body_f64(
    forces: wp.array(dtype=wp.vec3d),
    stress: wp.array(dtype=wp.mat33d),
    energy: wp.array(dtype=wp.float64),
    batch_idx: wp.array(dtype=wp.int32),
    n_particles: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    x_base: wp.array(dtype=wp.vec3d),
    force_base: wp.array(dtype=wp.vec3d),
    direction: wp.array(dtype=wp.vec3d),
    s_history: wp.array(dtype=wp.vec3d, ndim=2),
    y_history: wp.array(dtype=wp.vec3d, ndim=2),
    ys: wp.array(dtype=wp.float64, ndim=2),
    yy: wp.array(dtype=wp.float64, ndim=2),
    alpha_hist: wp.array(dtype=wp.float64, ndim=2),
    beta_hist: wp.array(dtype=wp.float64, ndim=2),
    ss: wp.array(dtype=wp.float64),
    f_base: wp.array(dtype=wp.float64),
    gg: wp.array(dtype=wp.float64),
    gd: wp.array(dtype=wp.float64),
    fmax: wp.array(dtype=wp.float64),
    frms_sq: wp.array(dtype=wp.float64),
    smax: wp.array(dtype=wp.float64),
    d0: wp.array(dtype=wp.float64),
    dmax: wp.array(dtype=wp.float64),
    dquad: wp.array(dtype=wp.float64),
    alpha_step: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    iteration: wp.array(dtype=wp.int32),
    end: wp.array(dtype=wp.int32),
    n_loop: wp.array(dtype=wp.int32),
    ls_trials: wp.array(dtype=wp.int32),
    history_count: wp.array(dtype=wp.int32),
    ref_cell: wp.array(dtype=wp.mat33d),
    ref_cell_inv: wp.array(dtype=wp.mat33d),
    kappa: wp.array(dtype=wp.float64),
    ext_batch_idx: wp.array(dtype=wp.int32),
    ext_atom_ptr: wp.array(dtype=wp.int32),
    phi: wp.array(dtype=wp.mat33d),
    phi_inv: wp.array(dtype=wp.mat33d),
    d_phi: wp.array(dtype=wp.mat33d),
    cell_dof_a: wp.array(dtype=wp.vec3d),
    cell_dof_b: wp.array(dtype=wp.vec3d),
    cell_force_a: wp.array(dtype=wp.vec3d),
    cell_force_b: wp.array(dtype=wp.vec3d),
    ext_positions: wp.array(dtype=wp.vec3d),
    ext_forces: wp.array(dtype=wp.vec3d),
    force_tol: wp.float64,
    rms_tol: wp.float64,
    stress_tol: wp.float64,
    ftol: wp.float64,
    wolfe: wp.float64,
    step_scale_down: wp.float64,
    step_scale_up: wp.float64,
    min_step: wp.float64,
    max_step: wp.float64,
    max_ls_iter: wp.int32,
    maxstep: wp.float64,
    curvature_eps: wp.float64,
) -> None:
    """Advance one variable-cell L-BFGS step on f64 coordinates."""
    _warp_step_cell(
        positions=positions,
        forces=forces,
        cell=cell,
        stress=stress,
        energy=energy,
        batch_idx=batch_idx,
        n_particles=n_particles,
        cell_state=LBFGSCellState(
            ref_cell=ref_cell,
            ref_cell_inv=ref_cell_inv,
            kappa=kappa,
            ext_batch_idx=ext_batch_idx,
            ext_atom_ptr=ext_atom_ptr,
            phi=phi,
            phi_inv=phi_inv,
            d_phi=d_phi,
            cell_dof_a=cell_dof_a,
            cell_dof_b=cell_dof_b,
            cell_force_a=cell_force_a,
            cell_force_b=cell_force_b,
            ext_positions=ext_positions,
            ext_forces=ext_forces,
        ),
        state=LBFGSState(
            x_base=x_base,
            force_base=force_base,
            direction=direction,
            s_history=s_history,
            y_history=y_history,
            ys=ys,
            yy=yy,
            alpha_hist=alpha_hist,
            beta_hist=beta_hist,
            ss=ss,
            f_base=f_base,
            gg=gg,
            gd=gd,
            fmax=fmax,
            frms_sq=frms_sq,
            smax=smax,
            d0=d0,
            dmax=dmax,
            dquad=dquad,
            alpha_step=alpha_step,
            status=status,
            iteration=iteration,
            end=end,
            n_loop=n_loop,
            ls_trials=ls_trials,
            history_count=history_count,
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


_CELL_IN_OUT_ARGS: tuple[str, ...] = (
    ("positions", "cell") + LBFGSState._fields + LBFGSCellState._fields
)

_CELL_BODIES = {jnp.float32: _lbfgs_cell_body_f32, jnp.float64: _lbfgs_cell_body_f64}

for _name, _body in (("f32", _lbfgs_cell_body_f32), ("f64", _lbfgs_cell_body_f64)):
    _p = tuple(inspect.signature(_body).parameters)
    _n = len(LBFGSState._fields)
    if _p[7 : 7 + _n] != LBFGSState._fields:
        raise RuntimeError(f"LBFGSState and _lbfgs_cell_body_{_name} have diverged")
    if _p[7 + _n : 7 + _n + len(LBFGSCellState._fields)] != LBFGSCellState._fields:
        raise RuntimeError(f"LBFGSCellState and _lbfgs_cell_body_{_name} have diverged")

_CELL_CALLABLES: dict[tuple, object] = {}


def _get_cell_callable(dtype, graph_mode: str):
    """Return the registered variable-cell callable, building it on first use."""
    key = (jnp.dtype(dtype).type, graph_mode)
    if key not in _CELL_CALLABLES:
        body = _CELL_BODIES.get(key[0])
        if body is None:
            raise ValueError(
                f"positions must be float32 or float64; got {jnp.dtype(dtype)}"
            )
        if graph_mode not in _GRAPH_MODES:
            raise ValueError(
                f"graph_mode must be one of {sorted(_GRAPH_MODES)}; got {graph_mode!r}"
            )
        kwargs = {}
        if graph_mode == "warp_staged":
            kwargs["stage_in_argnames"] = ["forces", "stress", "energy"]
        _CELL_CALLABLES[key] = jax_callable(
            body,
            num_outputs=len(_CELL_IN_OUT_ARGS),
            in_out_argnames=list(_CELL_IN_OUT_ARGS),
            graph_mode=_GRAPH_MODES[graph_mode],
            **kwargs,
        )
    return _CELL_CALLABLES[key]


def lbfgs_allocate_cell_state(
    num_atoms: int,
    num_systems: int,
    *,
    dtype=jnp.float64,
    cell_force_scale: float | None = None,
) -> LBFGSCellState:
    """Allocate the working arrays for variable-cell relaxation.

    Assumes every system has the same atom count, ``num_atoms // num_systems``.

    The reference cell is left zeroed; capture it with
    :func:`lbfgs_set_reference_cell` once the starting geometry is known.
    Everything that depends only on topology is filled here.

    Parameters
    ----------
    cell_force_scale : float, optional
        Multiplier on the atom count giving the cell coordinate scaling.
        Defaults to ``1 / atoms_per_system``, which puts cell and atom degrees
        of freedom on a comparable footing. Raise it to make the cell move less
        per step.

    Returns
    -------
    LBFGSCellState
    """
    dtype = jnp.dtype(dtype).type
    if dtype not in _CELL_BODIES:
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

    fields = {
        "ref_cell": jnp.zeros((num_systems, 3, 3), dtype),
        "ref_cell_inv": jnp.zeros((num_systems, 3, 3), dtype),
        "kappa": jnp.zeros((num_systems,), dtype),
        "ext_batch_idx": jnp.repeat(
            jnp.arange(num_systems, dtype=jnp.int32), per_system + 2
        ),
        "ext_atom_ptr": jnp.asarray(
            [s * per_system + 2 * s for s in range(num_systems + 1)], jnp.int32
        ),
        "phi": jnp.zeros((num_systems, 3, 3), dtype),
        "phi_inv": jnp.zeros((num_systems, 3, 3), dtype),
        "d_phi": jnp.zeros((num_systems, 3, 3), dtype),
        "cell_dof_a": jnp.zeros((num_systems, 3), dtype),
        "cell_dof_b": jnp.zeros((num_systems, 3), dtype),
        "cell_force_a": jnp.zeros((num_systems, 3), dtype),
        "cell_force_b": jnp.zeros((num_systems, 3), dtype),
        "ext_positions": jnp.zeros((num_ext, 3), dtype),
        "ext_forces": jnp.zeros((num_ext, 3), dtype),
    }
    cell_state = LBFGSCellState(**fields)
    # kappa depends only on topology, so fill it once here via the Warp helper.
    kappa = wp.zeros(
        num_systems,
        dtype=wp.float32 if dtype is jnp.float32 else wp.float64,
        device="cuda:0",
    )
    n_atoms_per_system = wp.array(
        np.full(num_systems, per_system, np.int32), dtype=wp.int32, device="cuda:0"
    )
    _warp_cell_kappa(n_atoms_per_system, kappa, cell_force_scale=cell_force_scale)
    return cell_state._replace(kappa=jnp.asarray(kappa.numpy(), dtype))


def lbfgs_set_reference_cell(
    cell: jax.Array, cell_state: LBFGSCellState
) -> LBFGSCellState:
    """Capture the reference cell that defines the variable-cell chart.

    Returns an updated state rather than mutating, as everything here does.
    Coordinates are measured relative to a cell held fixed for the whole
    relaxation, which is what makes history pairs from different iterations
    comparable, so call this once before the first step. Calling it again
    re-references the chart and invalidates every stored curvature pair.

    The reference is stored as an independent copy. Sharing a buffer with the
    caller's live ``cell`` would make it impossible to donate both to the same
    jitted step, because XLA rejects the same buffer being donated twice.
    """
    return cell_state._replace(
        ref_cell=jnp.array(cell, copy=True), ref_cell_inv=jnp.linalg.inv(cell)
    )


def lbfgs_step_coord_cell(
    positions: jax.Array,
    cell: jax.Array,
    state: LBFGSState,
    cell_state: LBFGSCellState,
    forces: jax.Array,
    stress: jax.Array,
    energy: jax.Array,
    batch_idx: jax.Array,
    n_particles: jax.Array,
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
    graph_mode: str = "warp",
) -> tuple[jax.Array, jax.Array, LBFGSState, LBFGSCellState]:
    """Advance one variable-cell step, relaxing coordinates and cell together.

    Returns new arrays; nothing is mutated. Donate all four of the returned
    objects so XLA can reuse their buffers.

    ``state`` must be sized for ``num_atoms + 2 * num_systems`` degrees of
    freedom, since the cell contributes two entries per system.

    Parameters
    ----------
    cell : jax.Array, shape (num_systems, 3, 3)
        Lattice vectors as columns. Kept lower-triangular, so the cell cannot
        drift into a rotation.
    stress : jax.Array, shape (num_systems, 3, 3)
        Cauchy stress. Drives the cell degrees of freedom, and is what
        ``stress_tol`` is compared against.
    force_tol, rms_tol, stress_tol : float, optional
        Convergence thresholds, always evaluated on the Cartesian forces and
        the stress rather than on packed norms.

    Returns
    -------
    positions, cell, state, cell_state

    See Also
    --------
    lbfgs_set_reference_cell : must be called first.
    lbfgs_step_coord : the coordinate-only equivalent.
    """
    num_systems = state.status.shape[0]
    if cell.shape != (num_systems, 3, 3):
        raise ValueError(
            f"cell must have shape ({num_systems}, 3, 3); got {cell.shape}"
        )
    if stress.shape != cell.shape:
        raise ValueError(f"stress shape {stress.shape} != cell shape {cell.shape}")
    expected = positions.shape[0] + 2 * num_systems
    if state.s_history.shape[1] != expected:
        raise ValueError(
            f"state is sized for {state.s_history.shape[1]} degrees of freedom, but "
            f"the variable-cell path needs {expected} (num_atoms + 2 * num_systems)"
        )
    call = _get_cell_callable(positions.dtype, graph_mode)
    outputs = call(
        forces,
        stress,
        energy,
        batch_idx,
        n_particles,
        positions,
        cell,
        *state,
        *cell_state,
        float(force_tol),
        float(rms_tol),
        float(stress_tol),
        float(ftol),
        float(wolfe),
        float(step_scale_down),
        float(step_scale_up),
        float(min_step),
        float(max_step),
        int(max_ls_iter),
        float(maxstep),
        float(curvature_eps),
    )
    n = len(LBFGSState._fields)
    return (
        outputs[0],
        outputs[1],
        LBFGSState(*outputs[2 : 2 + n]),
        LBFGSCellState(*outputs[2 + n :]),
    )
