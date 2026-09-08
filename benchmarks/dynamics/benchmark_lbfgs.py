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

"""Compare L-BFGS and FIRE2 by energy/force evaluations to convergence.

Evaluation count is the metric that matters for relaxation driven by a
machine-learned potential: the model call dominates the optimizer's own kernel
time by orders of magnitude, so the optimizer that reaches a given force
tolerance in fewer evaluations wins regardless of its per-step cost.

The test system is a Lennard-Jones cluster in reduced units, which is cheap to
evaluate on the host and anharmonic enough to be a fair test. FIRE2's timestep
and step cap are swept and its **best** configuration is reported, so the
baseline is not handicapped by a poor choice of hyperparameters.

Usage
-----
    python -m benchmarks.dynamics.benchmark_lbfgs [--sizes 13 32 55]
                                                  [--seeds 5]
                                                  [--force-tol 1e-4]
                                                  [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np
import warp as wp

from nvalchemiops.dynamics.optimizers import (
    LBFGS_CONVERGED,
    LBFGS_NEED_EVAL,
    fire2_step,
    lbfgs_reset,
    lbfgs_step,
)

DEVICE = "cuda:0"
EVAL_CAP = 20000


def lennard_jones(positions):
    """All-pairs Lennard-Jones in reduced units (epsilon = sigma = 1)."""
    delta = positions[:, None, :] - positions[None, :, :]
    r2 = (delta**2).sum(-1)
    np.fill_diagonal(r2, np.inf)
    inv6 = r2**-3
    inv12 = inv6**2
    per_atom_energy = 0.5 * (4.0 * (inv12 - inv6)).sum(1)
    coefficient = 24.0 * (2.0 * inv12 - inv6) / r2
    forces = (coefficient[..., None] * delta).sum(1)
    return per_atom_energy, forces


def _allocate_lbfgs_state(num_dofs, num_systems, history_size):
    def f64(n):
        return wp.zeros(n, dtype=wp.float64, device=DEVICE)

    def f64_2d(a, b):
        return wp.zeros((a, b), dtype=wp.float64, device=DEVICE)

    def vec(n):
        return wp.zeros(n, dtype=wp.vec3d, device=DEVICE)

    def i32(n):
        return wp.zeros(n, dtype=wp.int32, device=DEVICE)

    return {
        "x_base": vec(num_dofs),
        "force_base": vec(num_dofs),
        "direction": vec(num_dofs),
        "s_history": wp.zeros((history_size, num_dofs), dtype=wp.vec3d, device=DEVICE),
        "y_history": wp.zeros((history_size, num_dofs), dtype=wp.vec3d, device=DEVICE),
        "ys": f64_2d(history_size, num_systems),
        "yy": f64_2d(history_size, num_systems),
        "alpha_hist": f64_2d(history_size, num_systems),
        "beta_hist": f64_2d(history_size, num_systems),
        "ss": f64(num_systems),
        "f_base": f64(num_systems),
        "gg": f64(num_systems),
        "gd": f64(num_systems),
        "fmax": f64(num_systems),
        "frms_sq": f64(num_systems),
        "smax": f64(num_systems),
        "d0": f64(num_systems),
        "dmax": f64(num_systems),
        "dquad": f64(num_systems),
        "alpha_step": f64(num_systems),
        "status": i32(num_systems),
        "iteration": i32(num_systems),
        "end": i32(num_systems),
        "n_loop": i32(num_systems),
        "ls_trials": i32(num_systems),
        "history_count": i32(num_systems),
    }


def run_lbfgs(start, force_tol, history_size=6, maxstep=0.2):
    """Relax with L-BFGS; return (evaluations, converged, final max force)."""
    num_atoms = start.shape[0]
    positions = wp.array(start.copy(), dtype=wp.vec3d, device=DEVICE)
    forces = wp.zeros(num_atoms, dtype=wp.vec3d, device=DEVICE)
    energy = wp.zeros(1, dtype=wp.float64, device=DEVICE)
    batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=DEVICE)
    n_particles = wp.array(
        np.array([num_atoms], np.int32), dtype=wp.int32, device=DEVICE
    )
    state = _allocate_lbfgs_state(num_atoms, 1, history_size)
    lbfgs_reset(**state)

    for n_evals in range(1, EVAL_CAP + 1):
        per_atom_energy, f = lennard_jones(positions.numpy())
        forces.assign(f)
        energy.assign(np.array([per_atom_energy.sum()]))
        lbfgs_step(
            positions=positions,
            forces=forces,
            energy=energy,
            batch_idx=batch_idx,
            n_particles=n_particles,
            force_tol=force_tol,
            maxstep=maxstep,
            **state,
        )
        wp.synchronize()
        if state["status"].numpy()[0] != LBFGS_NEED_EVAL:
            final = np.linalg.norm(lennard_jones(positions.numpy())[1], axis=1).max()
            return n_evals, state["status"].numpy()[0] == LBFGS_CONVERGED, final
    final = np.linalg.norm(lennard_jones(positions.numpy())[1], axis=1).max()
    return EVAL_CAP, False, final


def run_fire2(start, force_tol, dt_start=0.02, maxstep=0.05, tmax=0.1):
    """Relax with FIRE2; return (evaluations, converged, final max force)."""
    num_atoms = start.shape[0]
    positions = wp.array(start.copy(), dtype=wp.vec3d, device=DEVICE)
    velocities = wp.zeros(num_atoms, dtype=wp.vec3d, device=DEVICE)
    forces = wp.zeros(num_atoms, dtype=wp.vec3d, device=DEVICE)
    batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=DEVICE)
    alpha = wp.array(np.array([0.09]), dtype=wp.float64, device=DEVICE)
    dt = wp.array(np.array([dt_start]), dtype=wp.float64, device=DEVICE)
    nsteps_inc = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    scratch = [wp.zeros(1, dtype=wp.float64, device=DEVICE) for _ in range(4)]

    for n_evals in range(1, EVAL_CAP + 1):
        f = lennard_jones(positions.numpy())[1]
        current = np.linalg.norm(f, axis=1).max()
        if current <= force_tol:
            return n_evals, True, current
        forces.assign(f)
        fire2_step(
            positions,
            velocities,
            forces,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            *scratch,
            maxstep=maxstep,
            tmax=tmax,
            tmin=0.002,
            dtgrow=1.1,
            dtshrink=0.5,
            delaystep=5,
            alpha0=0.09,
            alphashrink=0.99,
        )
        wp.synchronize()
    return EVAL_CAP, False, current


def best_fire2(start, force_tol):
    """FIRE2 at its best over a small hyperparameter sweep.

    Comparing against an untuned baseline would overstate the result; FIRE2 is
    sensitive to its timestep on this system.
    """
    best = (EVAL_CAP + 1, False, np.inf, None)
    for dt_start in (0.005, 0.01, 0.02, 0.05):
        for maxstep in (0.05, 0.1, 0.2):
            evals, converged, final = run_fire2(
                start, force_tol, dt_start=dt_start, maxstep=maxstep
            )
            if converged and evals < best[0]:
                best = (evals, converged, final, (dt_start, maxstep))
    if best[3] is None:
        evals, converged, final = run_fire2(start, force_tol)
        return evals, converged, final, "none converged"
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[13, 32, 55])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--force-tol", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    rows = []
    print(
        f"{'atoms':>6} {'seed':>5} {'lbfgs':>7} {'fire2':>7} "
        f"{'ratio':>7}  {'fire2 config':>14}"
    )
    for num_atoms in args.sizes:
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            start = rng.normal(size=(num_atoms, 3)) * (num_atoms ** (1 / 3)) * 0.55

            lb_evals, lb_ok, lb_force = run_lbfgs(start, args.force_tol)
            f2_evals, f2_ok, f2_force, f2_cfg = best_fire2(start, args.force_tol)
            ratio = lb_evals / f2_evals
            rows.append(
                {
                    "num_atoms": num_atoms,
                    "seed": seed,
                    "lbfgs_evals": lb_evals,
                    "lbfgs_converged": lb_ok,
                    "lbfgs_fmax": lb_force,
                    "fire2_evals": f2_evals,
                    "fire2_converged": f2_ok,
                    "fire2_fmax": f2_force,
                    "fire2_config": str(f2_cfg),
                    "ratio": ratio,
                }
            )
            print(
                f"{num_atoms:>6} {seed:>5} {lb_evals:>7} {f2_evals:>7} "
                f"{ratio:>7.3f}  {str(f2_cfg):>14}"
                f"{'' if (lb_ok and f2_ok) else '  (capped)'}"
            )

    ratios = [r["ratio"] for r in rows]
    geo_mean = float(np.exp(np.mean(np.log(ratios))))
    all_lbfgs_ok = all(r["lbfgs_converged"] for r in rows)
    print(f"\ngeometric mean evaluation ratio (L-BFGS / FIRE2): {geo_mean:.3f}")
    print(f"worst individual ratio:                          {max(ratios):.3f}")
    print(f"every L-BFGS run converged:                      {all_lbfgs_ok}")
    capped = [r for r in rows if not r["fire2_converged"]]
    if capped:
        print(
            f"note: FIRE2 hit the {EVAL_CAP}-evaluation cap in {len(capped)} case(s), "
            "so those ratios are upper bounds on L-BFGS's advantage."
        )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "lbfgs_vs_fire2_evaluations.csv"
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
