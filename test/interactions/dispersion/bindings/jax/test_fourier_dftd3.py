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

"""JAX binding tests for FourierD3.

The Warp layer is covered against a NumPy reference and a direct lattice sum elsewhere. These
tests check the binding: the transforms, the ``jax_kernel`` plumbing, both neighbour formats,
validation, and tracing under ``jax.jit``.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="No JAX installed.")
jnp = jax.numpy

from nvalchemiops.jax.interactions.dispersion import (  # noqa: E402
    FourierD3Parameters,
    fourier_dftd3,
)
from test.interactions.dispersion.test_fourier_dftd3 import (  # noqa: E402
    _neighbour_list,
    _reference_tables,
    _to_dense,
)

DAMPING = dict(a1=0.4289, a2=4.4407, s8=0.7875, s6=1.0)
R_CUT = 4.0
MESH = (32, 32, 32)
TRICLINIC_CELL = np.array(
    [[9.0, 0.0, 0.0], [1.7, 8.4, 0.0], [-0.8, 1.1, 8.7]],
    dtype=np.float64,
)

_FROZEN_TRICLINIC_OUTPUTS = {
    jnp.float32: (
        np.array([-0.042421203], dtype=np.float32),
        np.array(
            [
                [7.32706030e-05, 2.59222317e-04, -7.62797499e-05],
                [-1.29488518e-03, 1.93061540e-03, 3.51413013e-03],
                [3.60277714e-04, -1.40953547e-04, 1.40327495e-04],
                [-1.49265805e-04, -1.06874497e-04, 1.78414659e-04],
                [-1.57069109e-04, 8.88683062e-05, -2.09908729e-04],
                [6.16686375e-05, -1.09744695e-04, -4.29300067e-04],
                [5.28315832e-06, 4.07289452e-04, 9.36904617e-05],
                [1.10058126e-03, -2.32846173e-03, -3.21100932e-03],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [0.04543469, 0.000643458, 0.0036931634],
                [0.000643458, 0.04162443, -0.0028540797],
                [0.0036931634, -0.0028540797, 0.043026395],
            ],
            dtype=np.float32,
        ),
    ),
    jnp.float64: (
        np.array([-0.04242115847512903], dtype=np.float64),
        np.array(
            [
                [
                    7.3272755576328852e-05,
                    2.5922826286811689e-04,
                    -7.6292473440352427e-05,
                ],
                [
                    -1.2948776569518261e-03,
                    1.9306240367039824e-03,
                    3.5141324796775053e-03,
                ],
                [
                    3.6028863663117617e-04,
                    -1.4096899420555642e-04,
                    1.4033575353610551e-04,
                ],
                [
                    -1.4926656250950653e-04,
                    -1.0686814511734562e-04,
                    1.7845377942733487e-04,
                ],
                [
                    -1.5707016602808395e-04,
                    8.8849679514372521e-05,
                    -2.0989386911122226e-04,
                ],
                [
                    6.1647100556776177e-05,
                    -1.0973333666189772e-04,
                    -4.2931902627551508e-04,
                ],
                [
                    5.2917249519913725e-06,
                    4.0729532280844519e-04,
                    9.3659491793599909e-05,
                ],
                [
                    1.1005803784227147e-03,
                    -2.3284506151441341e-03,
                    -3.2110181605304994e-03,
                ],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [0.04543464167545261, 0.0006434603706614028, 0.0036931676833316836],
                [0.0006434603706614028, 0.04162438083939287, -0.0028540861723742816],
                [0.0036931676833316836, -0.0028540861723742816, 0.04302633387731889],
            ],
            dtype=np.float64,
        ),
    ),
}


@pytest.fixture()
def device():
    """GPU device fixture.

    ``jax_kernel`` wrappers are CUDA-only, a Warp JAX FFI limitation, so these tests do not
    run on CPU.
    """
    try:
        if len(jax.devices("gpu")) == 0:
            pytest.skip("No CUDA device available.")
    except RuntimeError:
        pytest.skip("No CUDA device available.")
    return "gpu"


@pytest.fixture(scope="module")
def system():
    """A small periodic cell with its neighbour list in both formats."""
    rng = np.random.default_rng(0)
    c6ab, cn_ref, species = _reference_tables()
    max_z = c6ab.shape[0]
    rcov = np.zeros(max_z)
    rcov[[1, 6, 8]] = [0.6, 1.2, 1.1]
    r4r2 = np.zeros(max_z)
    r4r2[[1, 6, 8]] = [1.0, 1.4, 1.2]

    n_atoms, box = 8, 9.0
    positions = rng.uniform(0.0, box, (n_atoms, 3))
    numbers = rng.choice(species, n_atoms)
    cell = np.eye(3) * box
    targets, pointer, shifts, _ = _neighbour_list(positions, cell, R_CUT)
    sources = np.repeat(np.arange(n_atoms), np.diff(pointer))
    matrix, matrix_shifts = _to_dense(targets, pointer, shifts, n_atoms)

    return {
        "positions": jnp.asarray(positions),
        "numbers": jnp.asarray(numbers, dtype=jnp.int32),
        "cell": jnp.asarray(cell),
        "params": FourierD3Parameters.from_tables(rcov, r4r2, c6ab, cn_ref, species),
        "neighbor_list": jnp.asarray(np.stack([sources, targets]), dtype=jnp.int32),
        "neighbor_ptr": jnp.asarray(pointer, dtype=jnp.int32),
        "unit_shifts": jnp.asarray(shifts, dtype=jnp.int32),
        "neighbor_matrix": jnp.asarray(matrix, dtype=jnp.int32),
        "neighbor_matrix_shifts": jnp.asarray(matrix_shifts, dtype=jnp.int32),
        "numpy": {
            "positions": positions,
            "numbers": numbers,
            "cell": cell,
            "rcov": rcov,
            "r4r2": r4r2,
            "targets": targets,
            "pointer": pointer,
            "shifts": shifts,
            "species": species,
            "c6ab": c6ab,
            "cn_ref": cn_ref,
            "n_atoms": n_atoms,
        },
    }


def _evaluate(system, **kwargs):
    """Call the public API with the CSR neighbour list unless told otherwise."""
    arguments = dict(
        fd3_params=system["params"],
        cell=system["cell"],
        r_cut=R_CUT,
        mesh_dimensions=MESH,
        neighbor_list=system["neighbor_list"],
        neighbor_ptr=system["neighbor_ptr"],
        unit_shifts=system["unit_shifts"],
        **DAMPING,
    )
    arguments.update(kwargs)
    positions = arguments.pop("positions", system["positions"])
    return fourier_dftd3(positions, system["numbers"], **arguments)


def _fixed_triclinic_system(dtype):
    """Build the deterministic mixed-species triclinic public regression fixture."""
    rng = np.random.default_rng(0)
    c6ab, cn_ref, species = _reference_tables()
    max_z = c6ab.shape[0]
    rcov = np.zeros(max_z)
    rcov[[1, 6, 8]] = [0.6, 1.2, 1.1]
    r4r2 = np.zeros(max_z)
    r4r2[[1, 6, 8]] = [1.0, 1.4, 1.2]
    positions = rng.uniform(0.1, 0.9, (8, 3)) @ TRICLINIC_CELL
    numbers = rng.choice(species, 8)
    targets, pointer, shifts, _ = _neighbour_list(positions, TRICLINIC_CELL, R_CUT)
    return {
        "positions": jnp.asarray(positions, dtype=dtype),
        "numbers": jnp.asarray(numbers, dtype=jnp.int32),
        "cell": jnp.asarray(TRICLINIC_CELL, dtype=dtype),
        "params": FourierD3Parameters.from_tables(
            rcov, r4r2, c6ab, cn_ref, species, dtype=dtype
        ),
        "neighbor_list": jnp.asarray(
            np.stack([np.repeat(np.arange(8), np.diff(pointer)), targets]),
            dtype=jnp.int32,
        ),
        "neighbor_ptr": jnp.asarray(pointer, dtype=jnp.int32),
        "unit_shifts": jnp.asarray(shifts, dtype=jnp.int32),
    }


def _single(box, seed, cell=None):
    """One periodic cell, as plain NumPy, with its neighbour list in both formats."""
    rng = np.random.default_rng(seed)
    c6ab, cn_ref, species = _reference_tables()
    max_z = c6ab.shape[0]
    rcov = np.zeros(max_z)
    rcov[[1, 6, 8]] = [0.6, 1.2, 1.1]
    r4r2 = np.zeros(max_z)
    r4r2[[1, 6, 8]] = [1.0, 1.4, 1.2]
    n_atoms = 8
    if cell is None:
        cell = np.eye(3) * box
        positions = rng.uniform(0.0, box, (n_atoms, 3))
    else:
        cell = np.asarray(cell, dtype=np.float64)
        positions = rng.uniform(0.1, 0.9, (n_atoms, 3)) @ cell
    numbers = rng.choice(species, n_atoms)
    targets, pointer, shifts, _ = _neighbour_list(positions, cell, R_CUT)
    matrix, matrix_shifts = _to_dense(targets, pointer, shifts, n_atoms)
    return dict(
        positions=positions,
        numbers=numbers,
        cell=cell,
        matrix=matrix,
        matrix_shifts=matrix_shifts,
        n_atoms=n_atoms,
        params=FourierD3Parameters.from_tables(rcov, r4r2, c6ab, cn_ref, species),
    )


def _dense_call(
    parts, cells, matrix, matrix_shifts, batch_idx, num_systems, fill_value, **kwargs
):
    """Evaluate in the dense neighbour format."""
    arguments = dict(
        mesh_dimensions=MESH,
        neighbor_matrix=jnp.asarray(matrix, dtype=jnp.int32),
        neighbor_matrix_shifts=jnp.asarray(matrix_shifts, dtype=jnp.int32),
        fill_value=fill_value,
        batch_idx=None
        if batch_idx is None
        else jnp.asarray(batch_idx, dtype=jnp.int32),
        num_systems=num_systems,
    )
    arguments.update(kwargs)
    return fourier_dftd3(
        jnp.asarray(parts["positions"]),
        jnp.asarray(parts["numbers"], dtype=jnp.int32),
        **DAMPING,
        fd3_params=parts["params"],
        cell=jnp.asarray(cells),
        r_cut=R_CUT,
        **arguments,
    )


@pytest.mark.gpu
class TestMeshSizing:
    """Mesh sizing and validation at the public JAX binding."""

    @pytest.mark.parametrize(
        ("spline_order", "mesh_dimensions"),
        [(2, (2, 3, 3)), (3, (2, 3, 3)), (5, (4, 5, 5)), (6, (5, 6, 6))],
    )
    def test_rejects_mesh_dimension_below_spline_floor(
        self, device, system, spline_order, mesh_dimensions
    ):
        """Every mesh axis must fit the requested interpolation stencil."""
        minimum = max(spline_order, 3)
        with pytest.raises(
            ValueError,
            match=rf"at least max\(spline_order, 3\) = {minimum}",
        ):
            _evaluate(
                system,
                mesh_dimensions=mesh_dimensions,
                spline_order=spline_order,
            )

    @pytest.mark.parametrize(
        ("spline_order", "mesh_dimensions"),
        [(2, (3, 4, 5)), (5, (5, 6, 7)), (6, (6, 7, 8))],
    )
    def test_accepts_floor_mesh_with_odd_and_even_axes(
        self, device, system, spline_order, mesh_dimensions
    ):
        """Exact-bound meshes remain valid across odd and even axes."""
        energy, forces = _evaluate(
            system,
            mesh_dimensions=mesh_dimensions,
            spline_order=spline_order,
        )
        assert jnp.isfinite(energy).all()
        assert jnp.isfinite(forces).all()

    @pytest.mark.parametrize("spline_order", [2, 5, 6])
    def test_coarse_spacing_is_clamped_to_spline_floor(
        self, device, system, spline_order
    ):
        """Automatic sizing must apply the same lower bound as explicit sizing."""
        minimum = max(spline_order, 3)
        by_spacing = _evaluate(
            system,
            mesh_dimensions=None,
            mesh_spacing=100.0,
            spline_order=spline_order,
        )
        by_dimensions = _evaluate(
            system,
            mesh_dimensions=(minimum, minimum, minimum),
            spline_order=spline_order,
        )
        np.testing.assert_allclose(
            np.asarray(by_spacing[0]), np.asarray(by_dimensions[0]), rtol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(by_spacing[1]),
            np.asarray(by_dimensions[1]),
            atol=1e-11 * float(jnp.abs(by_dimensions[1]).max()),
        )

    def test_batched_spacing_matches_smooth_explicit_mesh(self, device):
        """Automatic sizing uses all batched axes and matches the rounded public result."""
        cells = [
            np.diag([17.0, 6.0, 5.0]),
            np.diag([5.0, 11.0, 6.0]),
            np.diag([7.0, 8.0, 13.0]),
        ]
        systems = [_single(0.0, index, cell=cell) for index, cell in enumerate(cells)]
        counts = [system["n_atoms"] for system in systems]
        total = sum(counts)
        offsets = np.cumsum([0] + counts[:-1])
        width = max(system["matrix"].shape[1] for system in systems)
        matrices, matrix_shifts = [], []
        for system, offset in zip(systems, offsets, strict=True):
            matrix = np.full((system["n_atoms"], width), total, dtype=np.int32)
            shifts = np.zeros((system["n_atoms"], width, 3), dtype=np.int32)
            padded = system["matrix"] >= system["n_atoms"]
            matrix[:, : system["matrix"].shape[1]] = np.where(
                padded, total, system["matrix"] + int(offset)
            )
            shifts[:, : system["matrix_shifts"].shape[1]] = system["matrix_shifts"]
            matrices.append(matrix)
            matrix_shifts.append(shifts)
        batch = dict(
            positions=np.concatenate([system["positions"] for system in systems]),
            numbers=np.concatenate([system["numbers"] for system in systems]),
            params=systems[0]["params"],
        )
        batch_idx = np.concatenate(
            [
                np.full(count, index, dtype=np.int32)
                for index, count in enumerate(counts)
            ]
        )
        common = dict(
            parts=batch,
            cells=np.stack(cells),
            matrix=np.concatenate(matrices),
            matrix_shifts=np.concatenate(matrix_shifts),
            batch_idx=batch_idx,
            num_systems=len(systems),
            fill_value=total,
            compute_virial=True,
        )
        automatic = _dense_call(**common, mesh_dimensions=None, mesh_spacing=1.0)
        explicit = _dense_call(
            **common, mesh_dimensions=(18, 12, 14), mesh_spacing=None
        )
        for actual, expected in zip(automatic, explicit, strict=True):
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(expected), rtol=1e-12
            )
        lengths = np.linalg.norm(np.stack(cells), axis=-1).max(axis=0)
        assert all(
            length / dimension <= 1.0
            for length, dimension in zip(lengths, (18, 12, 14), strict=True)
        )

    def test_explicit_non_smooth_mesh_is_accepted_publicly(self, device, system):
        """A valid non-smooth explicit mesh remains usable through the public API."""
        energy, forces, virial = _evaluate(
            system,
            mesh_dimensions=(17, 19, 23),
            mesh_spacing=None,
            compute_virial=True,
        )
        assert jnp.isfinite(energy).all()
        assert jnp.isfinite(forces).all()
        assert jnp.isfinite(virial).all()


@pytest.mark.gpu
class TestBatching:
    """Several systems in one call, with different cells.

    The binding builds the Cartesian shifts itself, so this is not covered by the Warp-layer
    batching tests. The boxes are small enough relative to ``R_CUT`` that both systems have
    periodic neighbours well inside it; a system whose image shifts are all zero cannot
    detect which cell they were multiplied by.
    """

    def test_dense_batch_keeps_each_systems_cell(self, device):
        """A system's periodic images must be built from its own lattice."""
        systems = [_single(5.0, 0), _single(7.0, 1)]
        counts = [s["n_atoms"] for s in systems]
        total = int(sum(counts))
        offsets = np.cumsum([0] + counts[:-1]).astype(np.int64)
        width = max(s["matrix"].shape[1] for s in systems)

        rows, row_shifts = [], []
        for system, offset in zip(systems, offsets):
            own = system["matrix"]
            row = np.full((system["n_atoms"], width), total, dtype=np.int32)
            shift = np.zeros((system["n_atoms"], width, 3), dtype=np.int32)
            padded = own >= system["n_atoms"]
            row[:, : own.shape[1]] = np.where(padded, total, own + int(offset))
            shift[:, : own.shape[1]] = system["matrix_shifts"]
            rows.append(row)
            row_shifts.append(shift)

        batch = dict(
            positions=np.concatenate([s["positions"] for s in systems]),
            numbers=np.concatenate([s["numbers"] for s in systems]),
            params=systems[0]["params"],
        )
        batch_idx = np.concatenate(
            [np.full(count, index) for index, count in enumerate(counts)]
        )
        together = _dense_call(
            batch,
            np.stack([s["cell"] for s in systems]),
            np.concatenate(rows),
            np.concatenate(row_shifts),
            batch_idx,
            len(systems),
            total,
        )

        start = 0
        for index, system in enumerate(systems):
            alone = _dense_call(
                system,
                system["cell"][None],
                system["matrix"],
                system["matrix_shifts"],
                None,
                None,
                system["n_atoms"],
            )
            stop = start + system["n_atoms"]
            np.testing.assert_allclose(
                float(together[0][index]), float(alone[0][0]), rtol=1e-11
            )
            np.testing.assert_allclose(
                np.asarray(together[1][start:stop]),
                np.asarray(alone[1]),
                atol=1e-11 * float(jnp.abs(alone[1]).max()),
            )
            start = stop


@pytest.mark.gpu
class TestAgreementWithWarpLayer:
    """The binding must reproduce what the Warp layer already validated."""

    def test_matches_the_warp_pipeline(self, device, system):
        """Energy and forces agree with the NumPy-driven harness."""
        from nvalchemiops.interactions.dispersion._c6_decomposition import (
            decompose_c6_reference,
        )
        from test.interactions.dispersion._fourier_harness import fourier_d3_energy

        raw = system["numpy"]
        decomposition = decompose_c6_reference(
            raw["c6ab"], raw["cn_ref"], raw["species"]
        )
        reference = fourier_d3_energy(
            raw["positions"],
            raw["numbers"],
            decomposition.species_map[raw["numbers"]],
            np.zeros(raw["n_atoms"], dtype=np.int32),
            raw["cell"][None],
            raw["rcov"],
            decomposition,
            raw["r4r2"][decomposition.species],
            raw["targets"],
            raw["pointer"],
            raw["shifts"] @ raw["cell"],
            R_CUT,
            MESH,
            (DAMPING["s6"], DAMPING["s8"], DAMPING["a1"], DAMPING["a2"]),
        )
        energy, forces = _evaluate(system)
        np.testing.assert_allclose(np.asarray(energy), reference["energy"], rtol=1e-12)
        np.testing.assert_allclose(
            np.asarray(forces),
            reference["forces"],
            atol=1e-11 * np.abs(reference["forces"]).max(),
        )

    def test_forces_match_finite_differences(self, device, system):
        """The returned forces are the gradient of the returned energy."""
        base = system["positions"]
        analytic = np.asarray(_evaluate(system)[1])
        step = 1e-5
        numerical = np.zeros_like(analytic)
        for atom in range(base.shape[0]):
            for axis in range(3):
                for sign in (1.0, -1.0):
                    moved = base.at[atom, axis].add(sign * step)
                    energy = _evaluate({**system, "positions": moved})[0]
                    numerical[atom, axis] -= sign * float(energy[0]) / (2.0 * step)
        np.testing.assert_allclose(
            analytic, numerical, atol=1e-6 * np.abs(numerical).max()
        )

    def test_virial_is_returned_and_symmetric(self, device, system):
        """The virial is available and symmetric."""
        energy, forces, virial = _evaluate(system, compute_virial=True)
        virial = np.asarray(virial)[0]
        assert np.abs(virial).max() > 0.0
        np.testing.assert_allclose(virial, virial.T, atol=1e-12 * np.abs(virial).max())

    def test_triclinic_forces_match_finite_differences(self, device, system):
        """Cartesian forces remain the negative energy gradient in a triclinic cell."""
        raw = system["numpy"]
        rng = np.random.default_rng(3)
        positions = rng.uniform(0.1, 0.9, raw["positions"].shape) @ TRICLINIC_CELL
        targets, pointer, shifts, _ = _neighbour_list(positions, TRICLINIC_CELL, R_CUT)
        triclinic = {
            **system,
            "positions": jnp.asarray(positions),
            "cell": jnp.asarray(TRICLINIC_CELL),
            "neighbor_list": jnp.asarray(
                np.stack(
                    [np.repeat(np.arange(raw["n_atoms"]), np.diff(pointer)), targets]
                ),
                dtype=jnp.int32,
            ),
            "neighbor_ptr": jnp.asarray(pointer, dtype=jnp.int32),
            "unit_shifts": jnp.asarray(shifts, dtype=jnp.int32),
        }
        analytic = np.asarray(_evaluate(triclinic)[1])

        step = 1e-5
        base = triclinic["positions"]
        numerical = np.zeros_like(analytic)
        for atom in range(base.shape[0]):
            for axis in range(3):
                for sign in (1.0, -1.0):
                    moved = base.at[atom, axis].add(sign * step)
                    energy = _evaluate({**triclinic, "positions": moved})[0]
                    numerical[atom, axis] -= sign * float(energy[0]) / (2.0 * step)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-8)

    def test_triclinic_virial_matches_six_strain_derivatives(self, device, system):
        """The six independent virial components match triclinic strain differences."""
        raw = system["numpy"]
        rng = np.random.default_rng(3)
        positions = rng.uniform(0.1, 0.9, raw["positions"].shape) @ TRICLINIC_CELL
        targets, pointer, shifts, _ = _neighbour_list(positions, TRICLINIC_CELL, R_CUT)
        triclinic = {
            **system,
            "positions": jnp.asarray(positions),
            "cell": jnp.asarray(TRICLINIC_CELL),
            "neighbor_list": jnp.asarray(
                np.stack(
                    [np.repeat(np.arange(raw["n_atoms"]), np.diff(pointer)), targets]
                ),
                dtype=jnp.int32,
            ),
            "neighbor_ptr": jnp.asarray(pointer, dtype=jnp.int32),
            "unit_shifts": jnp.asarray(shifts, dtype=jnp.int32),
        }
        analytic = np.asarray(_evaluate(triclinic, compute_virial=True)[2])[0]

        step = 1e-6
        base_positions = triclinic["positions"]
        base_cell = triclinic["cell"]
        components = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
        numerical = np.zeros(len(components))
        for index, (row, column) in enumerate(components):
            energies = []
            for sign in (1.0, -1.0):
                strain = (
                    jnp.zeros((3, 3), dtype=base_cell.dtype)
                    .at[row, column]
                    .set(sign * step)
                )
                deformation = jnp.eye(3, dtype=base_cell.dtype) + strain
                energy = _evaluate(
                    {
                        **triclinic,
                        "positions": base_positions @ deformation.T,
                        "cell": base_cell @ deformation.T,
                    }
                )[0]
                energies.append(float(energy[0]))
            numerical[index] = (energies[0] - energies[1]) / (2.0 * step)
        np.testing.assert_allclose(
            analytic[
                [row for row, _ in components], [column for _, column in components]
            ],
            numerical,
            rtol=1e-6,
            atol=1e-8,
        )


@pytest.mark.gpu
class TestFrozenReciprocalContraction:
    """The public binding preserves the ordered reciprocal contraction result."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_fixed_mixed_species_triclinic_outputs(self, device, dtype):
        """Energy, forces, and virial match the pre-rewrite public outputs."""
        system = _fixed_triclinic_system(dtype)
        np.testing.assert_array_equal(
            np.asarray(system["numbers"]), np.array([1, 6, 8, 6, 6, 8, 8, 8])
        )
        assert system["params"].rank > 1
        energy, forces, virial = _evaluate(system, compute_virial=True)
        expected_energy, expected_forces, expected_virial = _FROZEN_TRICLINIC_OUTPUTS[
            dtype
        ]
        tolerance = (2e-5, 2e-6) if dtype == jnp.float32 else (1e-10, 1e-12)
        np.testing.assert_allclose(
            np.asarray(energy), expected_energy, rtol=tolerance[0], atol=tolerance[1]
        )
        np.testing.assert_allclose(
            np.asarray(forces), expected_forces, rtol=tolerance[0], atol=tolerance[1]
        )
        np.testing.assert_allclose(
            np.asarray(virial)[0], expected_virial, rtol=tolerance[0], atol=tolerance[1]
        )


@pytest.mark.gpu
class TestModulusConventions:
    """The public JAX binding supports both Fourier modulus conventions."""

    def test_default_matches_explicit_exact_moduli(self, device, system):
        """The default retains the discrete modulus behaviour."""
        default = _evaluate(system, compute_virial=True)
        explicit = _evaluate(system, exact_moduli=True, compute_virial=True)
        for actual, expected in zip(default, explicit, strict=True):
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(expected), rtol=1e-12, atol=1e-12
            )

    @pytest.mark.parametrize("exact_moduli", [True, False])
    def test_both_conventions_return_finite_outputs(self, device, system, exact_moduli):
        """Both modulus constructions produce finite energy, force, and virial outputs."""
        energy, forces, virial = _evaluate(
            system, exact_moduli=exact_moduli, compute_virial=True
        )
        assert jnp.isfinite(energy).all()
        assert jnp.isfinite(forces).all()
        assert jnp.isfinite(virial).all()

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("exact_moduli", [True, False])
    def test_matches_torch_public_call(self, device, system, dtype, exact_moduli):
        """Matched public inputs agree with the Torch binding in each convention."""
        torch = pytest.importorskip("torch", reason="PyTorch not installed.")
        if not torch.cuda.is_available():
            pytest.skip("No CUDA device available for the Torch binding.")
        from nvalchemiops.torch.interactions.dispersion import (  # noqa: PLC0415
            FourierD3Parameters as TorchFourierD3Parameters,
        )
        from nvalchemiops.torch.interactions.dispersion import (
            fourier_dftd3 as torch_fourier_dftd3,
        )

        raw = system["numpy"]
        jax_system = {
            **system,
            "positions": jnp.asarray(raw["positions"], dtype=dtype),
            "numbers": jnp.asarray(raw["numbers"], dtype=jnp.int32),
            "cell": jnp.asarray(raw["cell"], dtype=dtype),
            "params": FourierD3Parameters.from_tables(
                raw["rcov"],
                raw["r4r2"],
                raw["c6ab"],
                raw["cn_ref"],
                raw["species"],
                dtype=dtype,
            ),
        }
        jax_outputs = _evaluate(
            jax_system, exact_moduli=exact_moduli, compute_virial=True
        )

        torch_dtype = torch.float32 if dtype == jnp.float32 else torch.float64

        def tensor(array, tensor_dtype=torch_dtype):
            return torch.as_tensor(
                np.ascontiguousarray(array), dtype=tensor_dtype, device="cuda:0"
            )

        torch_params = TorchFourierD3Parameters.from_tables(
            tensor(raw["rcov"]),
            tensor(raw["r4r2"]),
            tensor(raw["c6ab"]),
            tensor(raw["cn_ref"]),
            raw["species"],
            device="cuda:0",
            dtype=torch_dtype,
        )
        torch_outputs = torch_fourier_dftd3(
            tensor(raw["positions"]),
            tensor(raw["numbers"], torch.int32),
            **DAMPING,
            fd3_params=torch_params,
            cell=tensor(raw["cell"]),
            r_cut=R_CUT,
            mesh_dimensions=MESH,
            neighbor_list=tensor(
                np.stack(
                    [
                        np.repeat(np.arange(raw["n_atoms"]), np.diff(raw["pointer"])),
                        raw["targets"],
                    ]
                ),
                torch.int32,
            ),
            neighbor_ptr=tensor(raw["pointer"], torch.int32),
            unit_shifts=tensor(raw["shifts"], torch.int32),
            exact_moduli=exact_moduli,
            compute_virial=True,
        )

        rtol, atol = (2e-5, 2e-6) if dtype == jnp.float32 else (1e-10, 1e-11)
        for actual, expected in zip(
            jax_outputs,
            (output.detach().cpu().numpy() for output in torch_outputs),
            strict=True,
        ):
            np.testing.assert_allclose(
                np.asarray(actual), expected, rtol=rtol, atol=atol
            )


@pytest.mark.gpu
class TestRankChunking:
    """Rank-chunked reciprocal passes preserve the public outputs."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_chunk_sizes_match_unchunked(self, device, dtype):
        """Unit, non-dividing, equal and oversized chunks agree with the fast path."""
        system = _fixed_triclinic_system(dtype)
        rank = system["params"].rank
        assert rank > 1
        nondivisor = next((size for size in range(1, rank) if rank % size), rank - 1)
        expected = _evaluate(system, compute_virial=True)
        rtol, atol = (2e-5, 2e-6) if dtype == jnp.float32 else (1e-9, 1e-10)
        for chunk_size in (1, nondivisor, rank, rank + 1):
            actual = _evaluate(
                system,
                compute_virial=True,
                rank_chunk_size=chunk_size,
            )
            for got, want in zip(actual, expected, strict=True):
                np.testing.assert_allclose(
                    np.asarray(got), np.asarray(want), rtol=rtol, atol=atol
                )

    def test_chunked_path_supports_both_modulus_conventions(self, device):
        """A non-dividing chunk remains valid with both modulus conventions."""
        system = _fixed_triclinic_system(jnp.float64)
        rank = system["params"].rank
        assert rank > 1
        chunk_size = next((size for size in range(1, rank) if rank % size), rank - 1)
        for exact_moduli in (True, False):
            expected = _evaluate(
                system,
                compute_virial=True,
                exact_moduli=exact_moduli,
            )
            actual = _evaluate(
                system,
                compute_virial=True,
                exact_moduli=exact_moduli,
                rank_chunk_size=chunk_size,
            )
            for got, want in zip(actual, expected, strict=True):
                np.testing.assert_allclose(
                    np.asarray(got), np.asarray(want), rtol=1e-9, atol=1e-10
                )

    @pytest.mark.parametrize(
        "invalid", [True, 0, -1, 1.5, "1", np.int64(1), jnp.asarray(1)]
    )
    def test_rejects_invalid_chunk_sizes(self, device, system, invalid):
        """Chunking is a positive host-static Python integer configuration."""
        with pytest.raises(ValueError, match="rank_chunk_size"):
            _evaluate(system, rank_chunk_size=invalid)

    def test_empty_system_is_zero_for_chunked_path(self, device, system):
        """Empty inputs preserve output shapes and zero initialization."""
        empty = {
            **system,
            "positions": jnp.empty((0, 3), dtype=system["positions"].dtype),
            "numbers": jnp.empty((0,), dtype=jnp.int32),
            "neighbor_list": jnp.empty((2, 0), dtype=jnp.int32),
            "neighbor_ptr": jnp.zeros((1,), dtype=jnp.int32),
            "unit_shifts": jnp.empty((0, 3), dtype=jnp.int32),
        }
        rank = empty["params"].rank
        assert rank > 1
        energy, forces, virial = _evaluate(
            empty,
            compute_virial=True,
            rank_chunk_size=rank - 1,
        )
        assert energy.shape == (1,)
        assert forces.shape == (0, 3)
        assert virial.shape == (1, 3, 3)
        np.testing.assert_array_equal(np.asarray(energy), 0.0)
        np.testing.assert_array_equal(np.asarray(forces), 0.0)
        np.testing.assert_array_equal(np.asarray(virial), 0.0)

    def test_rejects_traced_runtime_chunk_size(self, device, system):
        """A runtime JAX chunk size must be host-static under ``jax.jit``."""

        def evaluate(rank_chunk_size):
            return _evaluate(system, rank_chunk_size=rank_chunk_size)

        with pytest.raises(ValueError, match="host-static Python integer"):
            jax.jit(evaluate)(jnp.asarray(1))

    def test_nondividing_chunk_matches_eager_under_jit(self, device, system):
        """A closed-over non-dividing chunk preserves energy, force, and virial under JIT."""
        rank = system["params"].rank
        assert rank > 1
        chunk_size = next((size for size in range(1, rank) if rank % size), rank - 1)

        def evaluate(positions):
            return fourier_dftd3(
                positions,
                system["numbers"],
                fd3_params=system["params"],
                cell=system["cell"],
                r_cut=R_CUT,
                mesh_dimensions=MESH,
                neighbor_list=system["neighbor_list"],
                neighbor_ptr=system["neighbor_ptr"],
                unit_shifts=system["unit_shifts"],
                rank_chunk_size=chunk_size,
                compute_virial=True,
                **DAMPING,
            )

        eager = evaluate(system["positions"])
        traced = jax.jit(evaluate)(system["positions"])
        for actual, expected in zip(traced, eager, strict=True):
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                rtol=1e-9,
                atol=1e-10,
            )


@pytest.mark.gpu
class TestNeighbourFormats:
    """Both neighbour representations, and the validation around them."""

    def test_dense_and_csr_agree(self, device, system):
        """The two formats give the same answer."""
        csr = _evaluate(system)
        dense = _evaluate(
            system,
            neighbor_list=None,
            neighbor_ptr=None,
            unit_shifts=None,
            neighbor_matrix=system["neighbor_matrix"],
            neighbor_matrix_shifts=system["neighbor_matrix_shifts"],
        )
        np.testing.assert_allclose(np.asarray(csr[0]), np.asarray(dense[0]), rtol=1e-12)
        np.testing.assert_allclose(
            np.asarray(csr[1]),
            np.asarray(dense[1]),
            atol=1e-11 * float(jnp.abs(csr[1]).max()),
        )

    def test_rejects_a_half_filled_list(self, device, system):
        """Half of the coordination contributions would simply go missing."""
        numpy = system["numpy"]
        sources = np.repeat(np.arange(numpy["n_atoms"]), np.diff(numpy["pointer"]))
        targets, shifts = numpy["targets"], numpy["shifts"]
        lexicographic = np.where(
            shifts[:, 0] != 0,
            shifts[:, 0],
            np.where(shifts[:, 1] != 0, shifts[:, 1], shifts[:, 2]),
        )
        keep = (sources < targets) | ((sources == targets) & (lexicographic > 0))
        order = np.argsort(sources[keep], kind="stable")
        half_sources = sources[keep][order]
        pointer = np.zeros(numpy["n_atoms"] + 1, dtype=np.int32)
        np.add.at(pointer, half_sources + 1, 1)
        with pytest.raises(ValueError, match="both directions of every pair"):
            _evaluate(
                system,
                neighbor_list=jnp.asarray(
                    np.stack([half_sources, targets[keep][order]]), dtype=jnp.int32
                ),
                neighbor_ptr=jnp.asarray(np.cumsum(pointer), dtype=jnp.int32),
                unit_shifts=jnp.asarray(shifts[keep][order], dtype=jnp.int32),
            )

    def test_rejects_both_formats(self, device, system):
        """Supplying both neighbour formats is an error."""
        with pytest.raises(ValueError, match="Cannot provide both"):
            _evaluate(system, neighbor_matrix=system["neighbor_matrix"])

    def test_rejects_neither_format(self, device, system):
        """Supplying no neighbour format is an error."""
        with pytest.raises(ValueError, match="Must provide either"):
            _evaluate(system, neighbor_list=None, neighbor_ptr=None, unit_shifts=None)


@pytest.mark.gpu
class TestPrecision:
    """Both floating precisions are dispatched."""

    def test_float32_tracks_float64(self, device, system):
        """The single-precision path reproduces the double-precision one to its own accuracy.

        The Warp kernels are dtype-overloaded, so a missing or mismatched overload shows up
        as a dispatch failure or a silently different answer rather than as a type error.
        """
        numpy = system["numpy"]
        outputs = {}
        for dtype in (jnp.float64, jnp.float32):
            parameters = FourierD3Parameters.from_tables(
                numpy["rcov"],
                numpy["r4r2"],
                numpy["c6ab"],
                numpy["cn_ref"],
                numpy["species"],
                dtype=dtype,
            )
            outputs[dtype] = _evaluate(
                system,
                positions=jnp.asarray(numpy["positions"], dtype=dtype),
                cell=jnp.asarray(numpy["cell"], dtype=dtype),
                fd3_params=parameters,
            )
        double, single = outputs[jnp.float64], outputs[jnp.float32]
        assert single[0].dtype == jnp.float32
        assert single[1].dtype == jnp.float32
        # Single precision, so forces are compared against the magnitude of the result
        # rather than component by component: the small components carry the cancellation.
        np.testing.assert_allclose(float(single[0][0]), float(double[0][0]), rtol=1e-4)
        np.testing.assert_allclose(
            np.asarray(single[1]),
            np.asarray(double[1]),
            rtol=0.0,
            atol=1e-4 * float(jnp.abs(double[1]).max()),
        )


@pytest.mark.gpu
class TestJit:
    """Tracing behaviour."""

    def test_matches_eager_under_jit(self, device, system):
        """Compilation does not change the result."""
        eager = _evaluate(system)

        def evaluate(positions):
            return fourier_dftd3(
                positions,
                system["numbers"],
                fd3_params=system["params"],
                cell=system["cell"],
                r_cut=R_CUT,
                mesh_dimensions=MESH,
                neighbor_list=system["neighbor_list"],
                neighbor_ptr=system["neighbor_ptr"],
                unit_shifts=system["unit_shifts"],
                **DAMPING,
            )

        traced = jax.jit(evaluate)(system["positions"])
        np.testing.assert_allclose(
            np.asarray(traced[0]), np.asarray(eager[0]), rtol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(traced[1]),
            np.asarray(eager[1]),
            atol=1e-12 * float(jnp.abs(eager[1]).max()),
        )

    @pytest.mark.parametrize("exact_moduli", [True, False])
    def test_both_modulus_conventions_match_eager_under_jit(
        self, device, system, exact_moduli
    ):
        """A host-static modulus convention can be closed over by ``jax.jit``."""
        eager = _evaluate(system, exact_moduli=exact_moduli, compute_virial=True)

        def evaluate(positions):
            return fourier_dftd3(
                positions,
                system["numbers"],
                fd3_params=system["params"],
                cell=system["cell"],
                r_cut=R_CUT,
                mesh_dimensions=MESH,
                neighbor_list=system["neighbor_list"],
                neighbor_ptr=system["neighbor_ptr"],
                unit_shifts=system["unit_shifts"],
                exact_moduli=exact_moduli,
                compute_virial=True,
                **DAMPING,
            )

        traced = jax.jit(evaluate)(system["positions"])
        for actual, expected in zip(traced, eager, strict=True):
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                rtol=1e-12,
                atol=1e-12 * max(1.0, float(jnp.abs(expected).max())),
            )

    @pytest.mark.parametrize("invalid", [None, 1, 0.0, "true", jnp.asarray(True)])
    def test_rejects_non_boolean_modulus_configuration(self, device, system, invalid):
        """Only Python and NumPy booleans are valid modulus configuration values."""
        with pytest.raises(TypeError, match="Python or NumPy boolean"):
            _evaluate(system, exact_moduli=invalid)

    def test_rejects_traced_runtime_modulus_configuration(self, device, system):
        """A runtime JAX boolean must be static for ``jax.jit``."""

        def evaluate(exact_moduli):
            return _evaluate(system, exact_moduli=exact_moduli)

        with pytest.raises(TypeError, match="close over the flag|static argument"):
            jax.jit(evaluate)(jnp.asarray(True))

    def test_mesh_spacing_is_rejected_while_tracing(self, device, system):
        """``mesh_spacing`` reads cell lengths, so it cannot be used inside ``jax.jit``.

        Failing with an explanation beats silently baking in whatever the tracer produced.
        """

        def evaluate(cell):
            return fourier_dftd3(
                system["positions"],
                system["numbers"],
                fd3_params=system["params"],
                cell=cell,
                r_cut=R_CUT,
                mesh_dimensions=None,
                mesh_spacing=0.3,
                neighbor_list=system["neighbor_list"],
                neighbor_ptr=system["neighbor_ptr"],
                unit_shifts=system["unit_shifts"],
                **DAMPING,
            )

        with pytest.raises(ValueError, match="not possible inside jax.jit"):
            jax.jit(evaluate)(system["cell"])
