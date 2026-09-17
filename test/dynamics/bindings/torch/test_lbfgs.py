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

"""Torch contract tests for the state-object L-BFGS interface."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

torch = pytest.importorskip("torch")

from nvalchemiops.dynamics.optimizers import lbfgs as core  # noqa: E402
from nvalchemiops.torch import lbfgs as binding  # noqa: E402


@pytest.mark.parametrize(
    "dtype,vector", [(torch.float32, wp.vec3f), (torch.float64, wp.vec3d)]
)
def test_prepare_and_coordinate_step_match_core(dtype, vector):
    """Torch mutation matches Core for every explicit state field."""
    initial = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=np.float32 if dtype == torch.float32 else np.float64,
    )
    torch_positions = torch.tensor(initial, dtype=dtype)
    torch_forces = -torch_positions.clone()
    torch_batch = torch.zeros(2, dtype=torch.int32)
    torch_state = binding.prepare_lbfgs_state(torch_positions, 1, history_size=2)
    core_positions = wp.array(initial, dtype=vector, device="cpu")
    core_forces = wp.array(-initial, dtype=vector, device="cpu")
    core_batch = wp.zeros(2, dtype=wp.int32, device="cpu")
    core_state = core.prepare_lbfgs_state(core_positions, 1, history_size=2)
    binding.lbfgs_step_coord(torch_positions, torch_forces, torch_batch, torch_state)
    core.lbfgs_step_coord(core_positions, core_forces, core_batch, core_state)
    tolerance = 3e-6 if dtype == torch.float32 else 1e-12
    np.testing.assert_allclose(
        torch_positions.numpy(), core_positions.numpy(), rtol=tolerance, atol=tolerance
    )
    for torch_value, core_value in zip(
        torch_state._buffers(), core_state._buffers(), strict=True
    ):
        np.testing.assert_allclose(
            torch_value.numpy(), core_value.numpy(), rtol=tolerance, atol=tolerance
        )


def test_torch_public_surface_and_cell_allocator():
    """The Torch namespace exposes only state preparation and step functions."""
    assert binding.__all__ == [
        "LBFGSState",
        "LBFGSCellState",
        "prepare_lbfgs_state",
        "prepare_lbfgs_cell_state",
        "lbfgs_step_coord",
        "lbfgs_step_coord_cell",
    ]
    positions = torch.zeros((3, 3), dtype=torch.float32)
    cell = torch.eye(3, dtype=torch.float32)[None]
    batch_idx = torch.zeros(3, dtype=torch.int32)
    state = binding.prepare_lbfgs_cell_state(positions, cell, batch_idx)
    assert state.optimizer.x_base.shape == (5, 3)
    assert state.ext_atom_ptr.tolist() == [0, 5]
    binding.lbfgs_step_coord_cell(
        positions,
        torch.zeros_like(positions),
        cell,
        torch.zeros_like(cell),
        batch_idx,
        state,
    )


@pytest.mark.parametrize("labels", [[1, 0], [0, 2], [0, 0]])
def test_torch_cell_preparation_rejects_invalid_batch_topology(labels):
    """Torch cell preparation enforces the same topology as Core."""
    positions = torch.zeros((2, 3), dtype=torch.float32)
    cell = torch.zeros((2, 3, 3), dtype=torch.float32)
    batch_idx = torch.tensor(labels, dtype=torch.int32)
    with pytest.raises(ValueError, match="batch_idx"):
        binding.prepare_lbfgs_cell_state(positions, cell, batch_idx)
