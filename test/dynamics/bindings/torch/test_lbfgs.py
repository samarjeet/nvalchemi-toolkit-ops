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


def _state_fields(state):
    """Return the public mutable coordinate state fields in order."""
    return (
        state.x_base,
        state.force_base,
        state.direction,
        state.s_history,
        state.y_history,
        state.ys,
        state.yy,
        state.two_loop_alpha,
        state.initialized,
        state.history_end,
        state.history_count,
    )


def _cell_state_fields(state):
    """Return all public mutable variable-cell state fields in order."""
    return (
        *_state_fields(state.optimizer),
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
    )


def _cuda_case(dtype):
    """Create a deterministic CUDA coordinate problem."""
    positions = torch.tensor(
        [[1.5, -0.2, 0.4], [-0.7, 1.1, -0.3]], dtype=dtype, device="cuda"
    )
    forces = -positions.clone()
    batch_idx = torch.zeros(2, dtype=torch.int32, device="cuda")
    state = binding.prepare_lbfgs_state(positions, 1, history_size=2)
    return positions, forces, batch_idx, state


CUDA = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


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
        _state_fields(torch_state), _state_fields(core_state), strict=True
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("_cuda", [pytest.param(None, marks=CUDA)])
def test_cuda_eager_mutates_coordinates_and_state(dtype, _cuda):
    """Eager CUDA execution makes the public coordinate state observable."""
    positions, forces, batch_idx, state = _cuda_case(dtype)
    before = tuple(value.clone() for value in _state_fields(state))
    binding.lbfgs_step_coord(positions, forces, batch_idx, state, maxstep=0.2)
    torch.cuda.synchronize()
    assert not torch.equal(
        positions,
        torch.tensor(
            [[1.5, -0.2, 0.4], [-0.7, 1.1, -0.3]],
            dtype=dtype,
            device="cuda",
        ),
    )
    assert int(state.initialized.item()) == 1
    assert not torch.equal(state.x_base, before[0])
    assert not torch.equal(state.force_base, before[1])
    assert not torch.equal(state.direction, before[2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("_cuda", [pytest.param(None, marks=CUDA)])
def test_cuda_eager_cell_step_mutates_cell_and_cell_scratch(dtype, _cuda):
    """Variable-cell CUDA execution mutates both outputs and packing scratch."""
    positions, forces, batch_idx, _ = _cuda_case(dtype)
    cell = torch.eye(3, dtype=dtype, device="cuda")[None] * 3
    cell_force = torch.eye(3, dtype=dtype, device="cuda")[None] * 20
    state = binding.prepare_lbfgs_cell_state(
        positions, cell, batch_idx, cell_force_scale=0.5
    )
    before_cell = cell.clone()
    before_scratch = tuple(
        value.clone()
        for value in (
            state.phi,
            state.phi_inv,
            state.d_phi,
            state.ext_positions,
            state.ext_forces,
        )
    )
    binding.lbfgs_step_coord_cell(
        positions, forces, cell, cell_force, batch_idx, state, maxstep=0.2
    )
    torch.cuda.synchronize()
    assert not torch.equal(cell, before_cell)
    assert all(
        not torch.equal(value, old)
        for value, old in zip(
            (
                state.phi,
                state.phi_inv,
                state.d_phi,
                state.ext_positions,
                state.ext_forces,
            ),
            before_scratch,
            strict=True,
        )
    )


@pytest.mark.parametrize("_cuda", [pytest.param(None, marks=CUDA)])
def test_cuda_coordinate_step_compiles_fullgraph_and_preserves_state(_cuda):
    """A fullgraph caller observes the same mutations as eager execution."""
    positions, forces, batch_idx, state = _cuda_case(torch.float32)
    eager_positions = positions.clone()
    eager_forces = forces.clone()
    eager_batch_idx = batch_idx.clone()
    eager_fields = tuple(value.clone() for value in _state_fields(state))

    def run(pos, frc, batch, *fields):
        binding.lbfgs_step_coord(
            pos, frc, batch, binding.LBFGSState(*fields), maxstep=0.2
        )
        return (pos, *fields)

    eager = run(eager_positions, eager_forces, eager_batch_idx, *eager_fields)
    positions, forces, batch_idx, state = _cuda_case(torch.float32)
    torch._dynamo.reset()
    compiled = torch.compile(run, fullgraph=True)
    actual = compiled(positions, forces, batch_idx, *_state_fields(state))
    torch.cuda.synchronize()
    for value, expected in zip(actual, eager, strict=True):
        torch.testing.assert_close(value, expected)


@pytest.mark.parametrize("_cuda", [pytest.param(None, marks=CUDA)])
def test_cuda_cell_step_compiles_fullgraph_and_preserves_state(_cuda):
    """A fullgraph variable-cell caller observes all eager mutations."""
    positions, forces, batch_idx, _ = _cuda_case(torch.float32)
    cell = torch.eye(3, dtype=torch.float32, device="cuda")[None] * 3
    cell_force = torch.eye(3, dtype=torch.float32, device="cuda")[None] * 20
    state = binding.prepare_lbfgs_cell_state(
        positions, cell, batch_idx, cell_force_scale=0.5
    )

    def run(pos, frc, current_cell, raw_cell_force, batch, *fields):
        optimizer = binding.LBFGSState(*fields[:11])
        variable_cell = binding.LBFGSCellState(optimizer, *fields[11:])
        binding.lbfgs_step_coord_cell(
            pos,
            frc,
            current_cell,
            raw_cell_force,
            batch,
            variable_cell,
            maxstep=0.2,
        )
        return (pos, current_cell, *fields)

    eager_args = (
        positions.clone(),
        forces.clone(),
        cell.clone(),
        cell_force.clone(),
        batch_idx.clone(),
        *tuple(value.clone() for value in _cell_state_fields(state)),
    )
    compiled_args = tuple(value.clone() for value in eager_args)
    expected = run(*eager_args)
    torch._dynamo.reset()
    compiled = torch.compile(run, fullgraph=True)
    actual = compiled(*compiled_args)
    torch.cuda.synchronize()
    for value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, expected_value)


@pytest.mark.parametrize("_cuda", [pytest.param(None, marks=CUDA)])
def test_cuda_step_uses_the_current_non_default_stream(_cuda):
    """A selected stream orders the force producer, step, and consumer."""
    positions, _, batch_idx, state = _cuda_case(torch.float32)
    before = positions.clone()
    prepared = torch.cuda.Event()
    prepared.record(torch.cuda.current_stream())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        assert torch.cuda.current_stream() == stream
        stream.wait_event(prepared)
        forces = -2.0 * positions
        producer_done = torch.cuda.Event()
        producer_done.record(stream)
        binding.lbfgs_step_coord(positions, forces, batch_idx, state, maxstep=0.2)
        consumer = torch.sum((positions - before) * forces)
        consumer_done = torch.cuda.Event()
        consumer_done.record(stream)
    consumer_done.synchronize()
    assert producer_done.query()
    assert int(state.initialized.item()) == 1
    assert consumer.item() > 0.0


@pytest.mark.parametrize("_cuda", [pytest.param(None, marks=CUDA)])
def test_cuda_graph_replay_repeats_the_mutating_step(_cuda):
    """CUDA graph capture records Warp work and replay mutates the same state."""
    positions, forces, batch_idx, state = _cuda_case(torch.float32)
    initial_positions = positions.clone()
    initial_fields = tuple(value.clone() for value in _state_fields(state))
    warmup_stream = torch.cuda.Stream()
    with torch.cuda.stream(warmup_stream):
        binding.lbfgs_step_coord(positions, forces, batch_idx, state, maxstep=0.2)
    warmup_stream.synchronize()
    expected_positions = positions.clone()
    expected_fields = tuple(value.clone() for value in _state_fields(state))
    positions.copy_(initial_positions)
    for value, initial in zip(_state_fields(state), initial_fields, strict=True):
        value.copy_(initial)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.lbfgs_step_coord(positions, forces, batch_idx, state, maxstep=0.2)
    # CUDA stream capture records the Warp launch; replay performs the update.
    positions.copy_(initial_positions)
    for value, initial in zip(_state_fields(state), initial_fields, strict=True):
        value.copy_(initial)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(positions, expected_positions)
    for value, expected in zip(_state_fields(state), expected_fields, strict=True):
        torch.testing.assert_close(value, expected)
