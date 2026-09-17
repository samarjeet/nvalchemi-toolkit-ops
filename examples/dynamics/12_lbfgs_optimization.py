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

"""Run a caller-owned fixed-cell L-BFGS loop on a quadratic system."""

import torch

from nvalchemiops.torch.lbfgs import lbfgs_step_coord, prepare_lbfgs_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
positions = torch.tensor(
    [[1.0, -0.5, 0.25], [-0.75, 0.4, -0.2]],
    dtype=torch.float64,
    device=device,
)
batch_idx = torch.zeros(len(positions), dtype=torch.int32, device=device)
state = prepare_lbfgs_state(positions, num_systems=1, history_size=6)

force_tolerance = 1.0e-6
maximum_evaluations = 200

for evaluation in range(1, maximum_evaluations + 1):
    # Analytic force for E(x) = 0.5 * sum(x**2). Replace this with a model call.
    forces = -positions
    maximum_force = torch.linalg.vector_norm(forces, dim=1).max().item()
    if maximum_force < force_tolerance:
        print(f"Converged after {evaluation} evaluations")
        break

    lbfgs_step_coord(positions, forces, batch_idx, state, maxstep=0.2)
    if not torch.isfinite(positions).all():
        raise RuntimeError("L-BFGS proposed nonfinite coordinates")
else:
    raise RuntimeError("L-BFGS reached the caller-selected evaluation cap")

print(f"Final maximum force: {maximum_force:.3e}")
