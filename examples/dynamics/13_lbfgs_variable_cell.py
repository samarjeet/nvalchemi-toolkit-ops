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

"""Run a caller-owned variable-cell L-BFGS loop on an analytic target."""

import torch

from nvalchemiops.torch.lbfgs import (
    lbfgs_step_coord_cell,
    prepare_lbfgs_cell_state,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
positions = torch.zeros((1, 3), dtype=torch.float64, device=device)
cell = torch.diag(torch.tensor([3.0, 2.8, 2.6], dtype=torch.float64, device=device))[
    None
].contiguous()
target_cell = torch.eye(3, dtype=torch.float64, device=device)[None] * 2.5
batch_idx = torch.zeros(1, dtype=torch.int32, device=device)
state = prepare_lbfgs_cell_state(
    positions,
    cell,
    batch_idx,
    history_size=6,
    cell_force_scale=1.0,
)

cell_force_tolerance = 1.0e-6
maximum_evaluations = 200

for evaluation in range(1, maximum_evaluations + 1):
    # This quadratic example has no atomic force. In a model-driven loop,
    # convert evaluated stress to raw cell force with stress_to_cell_force().
    forces = torch.zeros_like(positions)
    cell_force = (target_cell - cell).contiguous()
    maximum_cell_force = cell_force.abs().max().item()
    if maximum_cell_force < cell_force_tolerance:
        print(f"Converged after {evaluation} evaluations")
        break

    lbfgs_step_coord_cell(
        positions,
        forces,
        cell,
        cell_force,
        batch_idx,
        state,
        maxstep=0.2,
    )
    if not torch.isfinite(positions).all() or not torch.isfinite(cell).all():
        raise RuntimeError("L-BFGS proposed nonfinite geometry")
    if torch.linalg.det(cell).min().item() <= 0.0:
        raise RuntimeError("L-BFGS proposed a non-positive cell")
else:
    raise RuntimeError("L-BFGS reached the caller-selected evaluation cap")

print(f"Final maximum cell force: {maximum_cell_force:.3e}")
print("Final cell:")
print(cell[0])
