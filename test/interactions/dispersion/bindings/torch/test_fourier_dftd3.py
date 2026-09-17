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

"""PyTorch binding tests for FourierD3.

The Warp layer is already covered against a NumPy reference and a direct lattice sum in
``test/interactions/dispersion/test_fourier_dftd3.py``. These tests check what the binding
itself adds: the two Fourier transforms, tensor plumbing, neighbour-format handling, unit and
mesh validation, and the parameter object.
"""

from __future__ import annotations

import numpy as np
import pytest

# Imported rather than guarded by a flag, so that the module body cannot name ``torch``
# while it is undefined. Default arguments, class bodies and decorator arguments all run at
# import time, which is before a module-level skipif can fire, so a flag would turn a missing
# optional dependency into a NameError during collection.
torch = pytest.importorskip("torch", reason="PyTorch not installed.")

from nvalchemiops.torch.interactions.dispersion import (  # noqa: E402
    FourierD3Parameters,
    FourierD3Setup,
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
    torch.float32: (
        np.array([-0.04242118448019028], dtype=np.float32),
        np.array(
            [
                [7.327698403969407e-05, 0.0002592335222288966, -7.628148887306452e-05],
                [-0.0012948603834956884, 0.0019306077156215906, 0.003514127107337117],
                [
                    0.00036028135218657553,
                    -0.00014095740334596485,
                    0.00014032777107786387,
                ],
                [
                    -0.00014926462608855218,
                    -0.00010687720350688323,
                    0.0001784134074114263,
                ],
                [
                    -0.00015706736303400248,
                    8.886829891707748e-05,
                    -0.00020990778284613043,
                ],
                [
                    6.168545223772526e-05,
                    -0.00010974753240589052,
                    -0.0004292989906389266,
                ],
                [5.288507054501679e-06, 0.0004072889860253781, 9.369157487526536e-05],
                [0.001100582187063992, -0.002328458707779646, -0.0032110046595335007],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [0.04543466866016388, 0.0006434561219066381, 0.003693157806992531],
                [0.00064345623832196, 0.04162442311644554, -0.002854075748473406],
                [0.003693157806992531, -0.002854075748473406, 0.043026383966207504],
            ],
            dtype=np.float32,
        ),
    ),
    torch.float64: (
        np.array([-0.04242115847512903], dtype=np.float64),
        np.array(
            [
                [
                    7.3272755576330587e-05,
                    2.5922826286811515e-04,
                    -7.6292473440352861e-05,
                ],
                [
                    -1.294877656951826e-03,
                    1.9306240367039824e-03,
                    3.5141324796775053e-03,
                ],
                [
                    3.602886366311727e-04,
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
                    8.884967951437258e-05,
                    -2.0989386911122313e-04,
                ],
                [
                    6.164710055677618e-05,
                    -1.0973333666189772e-04,
                    -4.2931902627551334e-04,
                ],
                [5.2917249519913725e-06, 4.072953228084452e-04, 9.365949183359991e-05],
                [
                    1.1005803784227147e-03,
                    -2.328450615144134e-03,
                    -3.2110181605304994e-03,
                ],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [0.04543464167545261, 0.0006434603706614, 0.00369316768333168],
                [0.0006434603706614, 0.04162438083939287, -0.00285408617237428],
                [0.00369316768333168, -0.00285408617237428, 0.04302633387731889],
            ],
            dtype=np.float64,
        ),
    ),
}


def _system(device, dtype=None, n_atoms=8, box=9.0, seed=0, cell=None):
    """A small periodic cell with its neighbour list in both formats.

    ``dtype`` defaults to ``torch.float64``, resolved on the call rather than written into
    the signature. Default arguments are evaluated when the module body runs, which is
    before pytest can apply the module-level skip, so naming ``torch`` there would raise a
    NameError during collection wherever Torch is not installed instead of skipping.
    """
    dtype = torch.float64 if dtype is None else dtype
    rng = np.random.default_rng(seed)
    c6ab, cn_ref, species = _reference_tables()
    max_z = c6ab.shape[0]
    rcov = np.zeros(max_z)
    rcov[[1, 6, 8]] = [0.6, 1.2, 1.1]
    r4r2 = np.zeros(max_z)
    r4r2[[1, 6, 8]] = [1.0, 1.4, 1.2]

    if cell is None:
        cell = np.eye(3) * box
        positions = rng.uniform(0.0, box, (n_atoms, 3))
    else:
        cell = np.asarray(cell, dtype=np.float64)
        positions = rng.uniform(0.1, 0.9, (n_atoms, 3)) @ cell
    numbers = rng.choice(species, n_atoms)
    targets, pointer, shifts, _ = _neighbour_list(positions, cell, R_CUT)
    sources = np.repeat(np.arange(n_atoms), np.diff(pointer))

    def tensor(array, torch_dtype=dtype):
        return torch.as_tensor(
            np.ascontiguousarray(array), dtype=torch_dtype, device=device
        )

    parameters = FourierD3Parameters.from_tables(
        tensor(rcov),
        tensor(r4r2),
        tensor(c6ab),
        tensor(cn_ref),
        species,
        device=device,
        dtype=dtype,
    )
    matrix, matrix_shifts = _to_dense(targets, pointer, shifts, n_atoms)
    return {
        "positions": tensor(positions),
        "numbers": tensor(numbers, torch.int32),
        "cell": tensor(cell),
        "params": parameters,
        "neighbor_list": torch.stack(
            [tensor(sources, torch.int32), tensor(targets, torch.int32)]
        ),
        "neighbor_ptr": tensor(pointer, torch.int32),
        "unit_shifts": tensor(shifts, torch.int32),
        "neighbor_matrix": tensor(matrix, torch.int32),
        "neighbor_matrix_shifts": tensor(matrix_shifts, torch.int32),
        "n_atoms": n_atoms,
    }


def _halve(system):
    """Keep one direction of each pair, the shape a ``half_fill=True`` builder produces."""
    sources = system["neighbor_list"][0].cpu().numpy()
    targets = system["neighbor_list"][1].cpu().numpy()
    shifts = system["unit_shifts"].cpu().numpy()
    # Between distinct atoms keep the ascending direction; an atom paired with its own
    # periodic image appears as (i, i, s) and (i, i, -s), so break that tie on the shift.
    lexicographic = np.where(
        shifts[:, 0] != 0,
        shifts[:, 0],
        np.where(shifts[:, 1] != 0, shifts[:, 1], shifts[:, 2]),
    )
    keep = (sources < targets) | ((sources == targets) & (lexicographic > 0))
    sources, targets, shifts = sources[keep], targets[keep], shifts[keep]
    order = np.argsort(sources, kind="stable")
    sources, targets, shifts = sources[order], targets[order], shifts[order]
    n_atoms = system["n_atoms"]
    pointer = np.zeros(n_atoms + 1, dtype=np.int32)
    np.add.at(pointer, sources + 1, 1)
    pointer = np.cumsum(pointer).astype(np.int32)

    def tensor(array, dtype):
        return torch.as_tensor(
            np.ascontiguousarray(array), dtype=dtype, device=system["positions"].device
        )

    matrix, matrix_shifts = _to_dense(targets, pointer, shifts, n_atoms)
    return {
        "neighbor_list": torch.stack(
            [tensor(sources, torch.int32), tensor(targets, torch.int32)]
        ),
        "neighbor_ptr": tensor(pointer, torch.int32),
        "unit_shifts": tensor(shifts, torch.int32),
        "neighbor_matrix": tensor(matrix, torch.int32),
        "neighbor_matrix_shifts": tensor(matrix_shifts, torch.int32),
    }


def _batched(systems):
    """Concatenate single-system dictionaries into one batch, in both neighbour formats.

    The systems are expected to have different cells. A batch whose cells are identical
    cannot detect a shift conversion that uses the wrong one, which is the whole point of
    exercising the bindings here rather than only at the Warp layer.
    """
    device = systems[0]["positions"].device
    counts = [system["n_atoms"] for system in systems]
    offsets = np.cumsum([0] + counts[:-1]).astype(np.int64)
    total = int(sum(counts))

    batch_idx = torch.cat(
        [
            torch.full((count,), index, dtype=torch.int32, device=device)
            for index, count in enumerate(counts)
        ]
    )

    sources, targets, unit_shifts = [], [], []
    pointer = [torch.zeros(1, dtype=torch.int32, device=device)]
    edges_so_far = 0
    for system, offset in zip(systems, offsets):
        sources.append(system["neighbor_list"][0] + int(offset))
        targets.append(system["neighbor_list"][1] + int(offset))
        unit_shifts.append(system["unit_shifts"])
        pointer.append(system["neighbor_ptr"][1:] + edges_so_far)
        edges_so_far += int(system["neighbor_ptr"][-1])

    # Dense rows are padded to a common width, and padding must point past every atom.
    width = max(int(system["neighbor_matrix"].shape[1]) for system in systems)
    matrices, matrix_shifts = [], []
    for system, offset in zip(systems, offsets):
        own = system["neighbor_matrix"]
        matrix = torch.full(
            (system["n_atoms"], width), total, dtype=torch.int32, device=device
        )
        shifts = torch.zeros(
            (system["n_atoms"], width, 3), dtype=torch.int32, device=device
        )
        padded = own >= system["n_atoms"]
        matrix[:, : own.shape[1]] = torch.where(
            padded, torch.full_like(own, total), own + int(offset)
        )
        shifts[:, : own.shape[1]] = system["neighbor_matrix_shifts"]
        matrices.append(matrix)
        matrix_shifts.append(shifts)

    return {
        "positions": torch.cat([system["positions"] for system in systems]),
        "numbers": torch.cat([system["numbers"] for system in systems]),
        "cell": torch.stack([system["cell"] for system in systems]),
        "params": systems[0]["params"],
        "batch_idx": batch_idx,
        "num_systems": len(systems),
        "neighbor_list": torch.stack([torch.cat(sources), torch.cat(targets)]),
        "neighbor_ptr": torch.cat(pointer),
        "unit_shifts": torch.cat(unit_shifts),
        "neighbor_matrix": torch.cat(matrices),
        "neighbor_matrix_shifts": torch.cat(matrix_shifts),
        "fill_value": total,
        "n_atoms": total,
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
    return fourier_dftd3(system["positions"], system["numbers"], **arguments)


@pytest.mark.gpu
class TestAgreementWithWarpLayer:
    """The binding must reproduce what the Warp layer already validated."""

    def test_matches_the_warp_pipeline(self):
        """Energy and forces agree with the NumPy-driven harness.

        The harness runs the same launchers with NumPy transforms, so this isolates the
        binding's own plumbing and its use of ``torch.fft``.
        """
        from test.interactions.dispersion._fourier_harness import fourier_d3_energy

        device = "cuda:0"
        system = _system(device)
        energy, forces = _evaluate(system)

        parameters = system["params"]
        reference = fourier_d3_energy(
            system["positions"].cpu().numpy(),
            system["numbers"].cpu().numpy(),
            parameters.species_map.cpu().numpy()[system["numbers"].cpu().numpy()],
            np.zeros(system["n_atoms"], dtype=np.int32),
            system["cell"].cpu().numpy()[None],
            parameters.rcov.cpu().numpy(),
            _decomposition_view(parameters),
            parameters.sqrt_q.cpu().numpy(),
            system["neighbor_list"][1].cpu().numpy(),
            system["neighbor_ptr"].cpu().numpy(),
            system["unit_shifts"].cpu().numpy() @ system["cell"].cpu().numpy(),
            R_CUT,
            MESH,
            (DAMPING["s6"], DAMPING["s8"], DAMPING["a1"], DAMPING["a2"]),
            device=device,
        )
        np.testing.assert_allclose(
            energy.cpu().numpy(), reference["energy"], rtol=1e-12
        )
        np.testing.assert_allclose(
            forces.cpu().numpy(),
            reference["forces"],
            atol=1e-11 * np.abs(reference["forces"]).max(),
        )

    def test_forces_match_finite_differences(self):
        """The returned forces are the gradient of the returned energy."""
        device = "cuda:0"
        system = _system(device, n_atoms=6, seed=3)
        analytic = _evaluate(system)[1].cpu().numpy()

        step = 1e-5
        base = system["positions"].clone()
        numerical = np.zeros_like(analytic)
        for atom in range(system["n_atoms"]):
            for axis in range(3):
                for sign in (1.0, -1.0):
                    system["positions"] = base.clone()
                    system["positions"][atom, axis] += sign * step
                    energy = _evaluate(system)[0]
                    numerical[atom, axis] -= sign * float(energy) / (2.0 * step)
        system["positions"] = base
        np.testing.assert_allclose(
            analytic, numerical, atol=1e-6 * np.abs(numerical).max()
        )

    def test_virial_matches_finite_strain(self):
        """The returned virial is the strain derivative of the returned energy."""
        device = "cuda:0"
        system = _system(device, n_atoms=6, seed=3)
        analytic = _evaluate(system, compute_virial=True)[2][0].cpu().numpy()

        step = 1e-6
        base_positions = system["positions"].clone()
        base_cell = system["cell"].clone()
        numerical = np.zeros((3, 3))
        for row in range(3):
            for column in range(3):
                energies = []
                for sign in (1.0, -1.0):
                    strain = torch.zeros(3, 3, dtype=base_cell.dtype, device=device)
                    strain[row, column] = sign * step
                    deformation = (
                        torch.eye(3, dtype=base_cell.dtype, device=device) + strain
                    )
                    system["positions"] = base_positions @ deformation.T
                    system["cell"] = base_cell @ deformation.T
                    energies.append(float(_evaluate(system)[0]))
                numerical[row, column] = (energies[0] - energies[1]) / (2.0 * step)
        system["positions"], system["cell"] = base_positions, base_cell
        np.testing.assert_allclose(
            analytic, numerical, atol=1e-6 * np.abs(numerical).max()
        )

    def test_triclinic_forces_match_finite_differences(self):
        """Cartesian forces remain the negative energy gradient in a triclinic cell."""
        device = "cuda:0"
        system = _system(device, n_atoms=6, seed=3, cell=TRICLINIC_CELL)
        analytic = _evaluate(system)[1].cpu().numpy()

        step = 1e-5
        base = system["positions"].clone()
        numerical = np.zeros_like(analytic)
        for atom in range(system["n_atoms"]):
            for axis in range(3):
                for sign in (1.0, -1.0):
                    system["positions"] = base.clone()
                    system["positions"][atom, axis] += sign * step
                    energy = _evaluate(system)[0]
                    numerical[atom, axis] -= sign * float(energy) / (2.0 * step)
        system["positions"] = base
        np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-8)

    def test_triclinic_virial_matches_six_strain_derivatives(self):
        """The six independent virial components match triclinic strain differences."""
        device = "cuda:0"
        system = _system(device, n_atoms=6, seed=3, cell=TRICLINIC_CELL)
        analytic = _evaluate(system, compute_virial=True)[2][0].cpu().numpy()

        step = 1e-6
        base_positions = system["positions"].clone()
        base_cell = system["cell"].clone()
        components = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
        numerical = np.zeros(len(components))
        for index, (row, column) in enumerate(components):
            energies = []
            for sign in (1.0, -1.0):
                strain = torch.zeros(3, 3, dtype=base_cell.dtype, device=device)
                strain[row, column] = sign * step
                deformation = (
                    torch.eye(3, dtype=base_cell.dtype, device=device) + strain
                )
                system["positions"] = base_positions @ deformation.T
                system["cell"] = base_cell @ deformation.T
                energies.append(float(_evaluate(system)[0]))
            numerical[index] = (energies[0] - energies[1]) / (2.0 * step)
        system["positions"], system["cell"] = base_positions, base_cell
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

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_fixed_mixed_species_triclinic_outputs(self, dtype):
        """Energy, forces, and virial match the pre-rewrite public outputs."""
        system = _system("cuda:0", dtype=dtype, n_atoms=8, seed=0, cell=TRICLINIC_CELL)
        np.testing.assert_array_equal(
            system["numbers"].cpu().numpy(), np.array([1, 6, 8, 6, 6, 8, 8, 8])
        )
        assert system["params"].rank > 1
        energy, forces, virial = _evaluate(system, compute_virial=True)
        expected_energy, expected_forces, expected_virial = _FROZEN_TRICLINIC_OUTPUTS[
            dtype
        ]
        tolerance = (2e-5, 2e-6) if dtype == torch.float32 else (1e-10, 1e-12)
        np.testing.assert_allclose(
            energy.cpu().numpy(), expected_energy, rtol=tolerance[0], atol=tolerance[1]
        )
        np.testing.assert_allclose(
            forces.cpu().numpy(), expected_forces, rtol=tolerance[0], atol=tolerance[1]
        )
        np.testing.assert_allclose(
            virial[0].cpu().numpy(),
            expected_virial,
            rtol=tolerance[0],
            atol=tolerance[1],
        )


def _decomposition_view(parameters):
    """Adapt a parameter object back to what the NumPy harness expects."""

    class _View:
        species = None
        eigs = parameters.eigs.cpu().numpy()
        v_q = parameters.v_q.cpu().numpy()
        cnref = parameters.cnref.cpu().numpy()
        species_map = parameters.species_map.cpu().numpy()
        n_species = parameters.n_species
        rank = parameters.rank

    return _View()


@pytest.mark.gpu
class TestNeighbourFormats:
    """Both neighbour representations, and the validation around them."""

    def test_dense_and_csr_agree(self):
        """The two formats describe the same neighbourhood and give the same answer."""
        system = _system("cuda:0")
        csr = _evaluate(system)
        dense = _evaluate(
            system,
            neighbor_list=None,
            neighbor_ptr=None,
            unit_shifts=None,
            neighbor_matrix=system["neighbor_matrix"],
            neighbor_matrix_shifts=system["neighbor_matrix_shifts"],
        )
        np.testing.assert_allclose(
            csr[0].cpu().numpy(), dense[0].cpu().numpy(), rtol=1e-12
        )
        np.testing.assert_allclose(
            csr[1].cpu().numpy(),
            dense[1].cpu().numpy(),
            atol=1e-11 * float(csr[1].abs().max()),
        )

    def test_rejects_a_half_filled_list(self):
        """A half-filled list silently loses coordination, so it must not be accepted.

        Each atom's coordination number is accumulated from its own row alone, with the
        reverse edge walked by the other atom. Half of the contributions simply go missing,
        which shifts the energy and leaves the forces non-conservative rather than raising.
        """
        system = _system("cuda:0", box=5.0)
        half = _halve(system)
        with pytest.raises(ValueError, match="both directions of every pair"):
            _evaluate(
                system,
                neighbor_list=half["neighbor_list"],
                neighbor_ptr=half["neighbor_ptr"],
                unit_shifts=half["unit_shifts"],
            )

    def test_rejects_a_half_filled_matrix(self):
        """The dense format carries the same requirement."""
        system = _system("cuda:0", box=5.0)
        half = _halve(system)
        with pytest.raises(ValueError, match="both directions of every pair"):
            _evaluate(
                system,
                neighbor_list=None,
                neighbor_ptr=None,
                unit_shifts=None,
                neighbor_matrix=half["neighbor_matrix"],
                neighbor_matrix_shifts=half["neighbor_matrix_shifts"],
            )

    def test_rejects_both_formats(self):
        """Supplying both neighbour formats is an error."""
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="Cannot provide both"):
            _evaluate(system, neighbor_matrix=system["neighbor_matrix"])

    def test_rejects_neither_format(self):
        """Supplying no neighbour format is an error."""
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="Must provide either"):
            _evaluate(system, neighbor_list=None, neighbor_ptr=None, unit_shifts=None)

    def test_rejects_mismatched_shifts(self):
        """Each format needs its own shift representation."""
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="unit_shifts is for neighbor_list"):
            _evaluate(
                system,
                neighbor_list=None,
                neighbor_ptr=None,
                neighbor_matrix=system["neighbor_matrix"],
                neighbor_matrix_shifts=system["neighbor_matrix_shifts"],
                unit_shifts=system["unit_shifts"],
            )

    def test_requires_shifts(self):
        """Periodic images are mandatory, since the method is periodic."""
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="unit_shifts is required"):
            _evaluate(system, unit_shifts=None)


@pytest.mark.gpu
class TestBatching:
    """Several systems in one call.

    The bindings build the Cartesian shifts themselves, so the Warp-layer batching tests do
    not cover this; the cells must differ for the coverage to mean anything.
    """

    @staticmethod
    def _systems():
        # Both boxes must be small enough relative to R_CUT to have periodic neighbours
        # well inside the cutoff. At box 13 there are none at all, and a system whose image
        # shifts are all zero cannot detect which cell they were multiplied by.
        return [
            _system("cuda:0", box=5.0, seed=0),
            _system("cuda:0", box=7.0, seed=1),
        ]

    @staticmethod
    def _dense(arguments, system):
        """Swap the CSR arguments for the dense matrix ones."""
        arguments.update(
            neighbor_list=None,
            neighbor_ptr=None,
            unit_shifts=None,
            neighbor_matrix=system["neighbor_matrix"],
            neighbor_matrix_shifts=system["neighbor_matrix_shifts"],
            fill_value=system.get("fill_value"),
        )
        return arguments

    @pytest.mark.parametrize("dense", [False, True])
    def test_each_system_keeps_its_own_cell(self, dense):
        """A system's periodic images must be built from its own lattice.

        Converting every image shift with the first system's cell leaves systems after the
        first with neighbours in the wrong places, which corrupts their coordination numbers
        and so their energies, forces and virial. It is invisible in a batch of identical
        cells.
        """
        systems = self._systems()
        batch = _batched(systems)
        extra = {"compute_virial": True}
        together = _evaluate(
            batch,
            batch_idx=batch["batch_idx"],
            num_systems=batch["num_systems"],
            **(self._dense(dict(extra), batch) if dense else extra),
        )
        start = 0
        for index, system in enumerate(systems):
            alone = _evaluate(
                system, **(self._dense(dict(extra), system) if dense else extra)
            )
            stop = start + system["n_atoms"]
            np.testing.assert_allclose(
                together[0][index].item(), alone[0][0].item(), rtol=1e-11
            )
            np.testing.assert_allclose(
                together[1][start:stop].cpu().numpy(),
                alone[1].cpu().numpy(),
                atol=1e-11 * float(alone[1].abs().max()),
            )
            np.testing.assert_allclose(
                together[2][index].cpu().numpy(),
                alone[2][0].cpu().numpy(),
                atol=1e-11 * float(alone[2].abs().max()),
            )
            start = stop


@pytest.mark.gpu
class TestMeshAndUnits:
    """Mesh sizing and the unit contract, both of which fail silently if left implicit."""

    def test_every_spline_order_reaches_the_same_energy(self):
        """The lattice sum does not depend on the interpolation order used to reach it.

        The B-spline attenuation factors are order-dependent and an odd order places its
        stencil differently from an even one. A wrong modulus for a single order would still
        give a finite, plausible, self-consistent answer -- it would just converge somewhere
        else -- so the orders have to be checked against each other rather than themselves.

        Accuracy at a fixed mesh must also improve with order, which is the property that
        makes a low order a cost/accuracy trade rather than a mistake.
        """
        system = _system("cuda:0")
        fine = (96, 96, 96)
        reference = _evaluate(system, mesh_dimensions=fine, spline_order=6)[0].item()
        errors = {
            order: abs(
                _evaluate(system, mesh_dimensions=fine, spline_order=order)[0].item()
                - reference
            )
            / abs(reference)
            for order in (2, 3, 4, 5)
        }
        # Every order lands on the same number; order 2 is linear interpolation and gets
        # there far more slowly, and order 3 is noticeably noisier than the even orders.
        assert errors[2] < 1e-3, errors
        assert errors[3] < 1e-5, errors
        assert errors[4] < 1e-7, errors
        assert errors[5] < 1e-8, errors
        ordered = [errors[o] for o in (2, 3, 4, 5)]
        assert ordered == sorted(ordered, reverse=True), (
            f"accuracy should improve with spline order, got {errors}"
        )

    def test_refining_the_mesh_improves_every_spline_order(self):
        """A stencil offset would leave a residual the mesh cannot reduce.

        Comparing one order against another at a single mesh cannot tell a constant offset
        from ordinary discretisation error; only refining can.
        """
        system = _system("cuda:0")
        reference = _evaluate(system, mesh_dimensions=(128, 128, 128), spline_order=6)[
            0
        ].item()
        for order in (2, 4, 5):
            coarse, fine = (
                abs(
                    _evaluate(system, mesh_dimensions=(m, m, m), spline_order=order)[
                        0
                    ].item()
                    - reference
                )
                / abs(reference)
                for m in (24, 96)
            )
            assert fine < coarse, f"order {order} not converging: {coarse} -> {fine}"

    def test_requires_exactly_one_mesh_option(self):
        """Neither or both of the two ways to size the mesh is an error.

        There is no accuracy-based estimator to fall back on, so guessing would be worse
        than refusing.
        """
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="exactly one of mesh_dimensions"):
            _evaluate(system, mesh_dimensions=None)
        with pytest.raises(ValueError, match="exactly one of mesh_dimensions"):
            _evaluate(system, mesh_spacing=0.3)

    def test_mesh_spacing_sizes_from_the_cell(self):
        """A spacing gives the same answer as the dimensions it implies."""
        system = _system("cuda:0")
        spacing = 9.0 / 32.0
        by_spacing = _evaluate(system, mesh_dimensions=None, mesh_spacing=spacing)
        by_dimensions = _evaluate(system)
        np.testing.assert_allclose(
            by_spacing[0].cpu().numpy(), by_dimensions[0].cpu().numpy(), rtol=1e-12
        )

    @pytest.mark.parametrize(
        ("spline_order", "mesh_dimensions"),
        [(2, (2, 3, 3)), (3, (2, 3, 3)), (5, (4, 5, 5)), (6, (5, 6, 6))],
    )
    def test_rejects_mesh_dimension_below_spline_floor(
        self, spline_order, mesh_dimensions
    ):
        """Every mesh axis must fit the requested interpolation stencil."""
        system = _system("cuda:0")
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

    @pytest.mark.parametrize("exact_moduli", [True, False])
    @pytest.mark.parametrize(
        ("spline_order", "mesh_dimensions"),
        [(2, (3, 4, 5)), (5, (5, 6, 7)), (6, (6, 7, 8))],
    )
    def test_accepts_floor_mesh_for_both_moduli(
        self, exact_moduli, spline_order, mesh_dimensions
    ):
        """Exact-bound meshes remain valid for both exposed modulus conventions."""
        system = _system("cuda:0")
        energy, forces = _evaluate(
            system,
            mesh_dimensions=mesh_dimensions,
            spline_order=spline_order,
            exact_moduli=exact_moduli,
        )
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()

    @pytest.mark.parametrize("spline_order", [2, 5, 6])
    def test_coarse_spacing_is_clamped_to_spline_floor(self, spline_order):
        """Automatic sizing must apply the same lower bound as explicit sizing."""
        system = _system("cuda:0")
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
            by_spacing[0].cpu().numpy(), by_dimensions[0].cpu().numpy(), rtol=1e-12
        )
        np.testing.assert_allclose(
            by_spacing[1].cpu().numpy(),
            by_dimensions[1].cpu().numpy(),
            atol=1e-11 * float(by_dimensions[1].abs().max()),
        )

    def test_batched_spacing_matches_smooth_explicit_mesh(self):
        """Automatic sizing uses all batched axes and matches the rounded public result."""
        cells = [
            np.diag([17.0, 6.0, 5.0]),
            np.diag([5.0, 11.0, 6.0]),
            np.diag([7.0, 8.0, 13.0]),
        ]
        systems = [
            _system("cuda:0", seed=index, cell=cell) for index, cell in enumerate(cells)
        ]
        batch = _batched(systems)
        automatic = _evaluate(
            batch,
            batch_idx=batch["batch_idx"],
            num_systems=batch["num_systems"],
            mesh_dimensions=None,
            mesh_spacing=1.0,
            exact_moduli=True,
            compute_virial=True,
        )
        explicit = _evaluate(
            batch,
            batch_idx=batch["batch_idx"],
            num_systems=batch["num_systems"],
            mesh_dimensions=(18, 12, 14),
            mesh_spacing=None,
            exact_moduli=True,
            compute_virial=True,
        )
        for actual, expected in zip(automatic, explicit, strict=True):
            np.testing.assert_allclose(
                actual.cpu().numpy(), expected.cpu().numpy(), rtol=1e-12, atol=1e-12
            )
        lengths = torch.linalg.norm(batch["cell"], dim=-1).amax(dim=0)
        assert all(
            float(length) / dimension <= 1.0
            for length, dimension in zip(lengths, (18, 12, 14), strict=True)
        )

    def test_explicit_non_smooth_mesh_is_accepted_publicly(self):
        """A valid non-smooth explicit mesh remains usable through the public API."""
        system = _system("cuda:0")
        energy, forces, virial = _evaluate(
            system,
            mesh_dimensions=(17, 19, 23),
            mesh_spacing=None,
            compute_virial=True,
        )
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(virial).all()

    def test_rejects_bad_mesh_arguments(self):
        """Degenerate mesh requests are rejected rather than clamped."""
        system = _system("cuda:0")
        with pytest.raises(ValueError, match="three positive integers"):
            _evaluate(system, mesh_dimensions=(0, 8, 8))
        with pytest.raises(ValueError, match="mesh_spacing must be positive"):
            _evaluate(system, mesh_dimensions=None, mesh_spacing=-1.0)

    @pytest.mark.parametrize("scale", [1.5, 3.0])
    def test_energy_is_invariant_under_a_consistent_unit_change(self, scale):
        """Restating the same physical system in another length unit changes nothing.

        This is the check that a unit mistake would fail. Every dimensioned quantity has to
        move together: with ``[C6] = energy * length**6`` and
        ``R0 = a1 * sqrt(3 * sqrt_q_A * sqrt_q_B) + a2``, the length-carrying quantities are
        ``positions``, ``cell``, ``rcov``, ``r_cut``, ``sqrt_q`` and ``a2``, while ``eigs``
        carries ``length**6`` and ``s6``, ``s8`` and ``a1`` are dimensionless.

        Agreement is close but not exact because the counting function carries one absolute
        regulariser, which is the single scale-dependent constant in the method.
        """
        device = "cuda:0"
        base = _system(device)
        expected = float(_evaluate(base)[0])

        parameters = base["params"]
        rescaled = _system(device)
        rescaled["positions"] = base["positions"] * scale
        rescaled["cell"] = base["cell"] * scale
        rescaled["params"] = FourierD3Parameters(
            rcov=parameters.rcov * scale,
            sqrt_q=parameters.sqrt_q * scale,
            cnref=parameters.cnref,
            v_q=parameters.v_q,
            eigs=parameters.eigs * scale**6,
            species_map=parameters.species_map,
            max_relative_error=parameters.max_relative_error,
        )
        actual = float(
            _evaluate(
                rescaled,
                r_cut=R_CUT * scale,
                a1=DAMPING["a1"],
                a2=DAMPING["a2"] * scale,
                s8=DAMPING["s8"],
                s6=DAMPING["s6"],
            )[0]
        )
        assert abs(actual - expected) < 1e-9 * abs(expected)

    def test_rejects_uncovered_species(self):
        """An atom the decomposition does not cover is reported, not silently zeroed."""
        system = _system("cuda:0")
        numbers = system["numbers"].clone()
        numbers[0] = 7
        system["numbers"] = numbers
        with pytest.raises(ValueError, match="not covered by fd3_params"):
            _evaluate(system)


@pytest.mark.gpu
class TestParameters:
    """The parameter object."""

    def test_carries_no_damping_parameters(self):
        """Damping is supplied per call, so a stored copy cannot go stale.

        Keeping a derived self-energy term alongside call-time damping would let a caller mix
        one functional's reciprocal sum with another's self-energy and get a plausible but
        wrong number.
        """
        fields = set(FourierD3Parameters.__dataclass_fields__)
        assert not (fields & {"s6", "s8", "a1", "a2", "selfcont", "phi_zero"})

    def test_changing_damping_changes_the_energy(self):
        """The same parameter object under two functionals gives two answers."""
        system = _system("cuda:0")
        first = float(_evaluate(system)[0])
        second = float(_evaluate(system, a1=0.35, a2=5.0, s8=1.2)[0])
        assert abs(second - first) > 1e-6 * abs(first)

    def test_reports_its_truncation_error(self):
        """The achieved reconstruction error is available to the caller."""
        system = _system("cuda:0")
        assert 0.0 <= system["params"].max_relative_error < 1e-3

    def test_to_moves_device_and_dtype(self):
        """``to`` converts the floating fields and leaves the channel map integral."""
        system = _system("cuda:0")
        moved = system["params"].to(device="cpu", dtype=torch.float32)
        assert moved.rcov.device.type == "cpu"
        assert moved.eigs.dtype == torch.float32
        assert moved.species_map.dtype == torch.int32

    def test_rejects_inconsistent_shapes(self):
        """Mismatched factor shapes are caught at construction."""
        system = _system("cuda:0")
        parameters = system["params"]
        with pytest.raises(ValueError, match="eigs has rank"):
            FourierD3Parameters(
                rcov=parameters.rcov,
                sqrt_q=parameters.sqrt_q,
                cnref=parameters.cnref,
                v_q=parameters.v_q,
                eigs=parameters.eigs[:-1],
                species_map=parameters.species_map,
                max_relative_error=0.0,
            )

    def test_rejects_mixed_devices(self):
        """All parameter tensors must live together."""
        system = _system("cuda:0")
        parameters = system["params"]
        with pytest.raises(ValueError, match="must share one device"):
            FourierD3Parameters(
                rcov=parameters.rcov.cpu(),
                sqrt_q=parameters.sqrt_q,
                cnref=parameters.cnref,
                v_q=parameters.v_q,
                eigs=parameters.eigs,
                species_map=parameters.species_map,
                max_relative_error=0.0,
            )


@pytest.mark.gpu
class TestPrecision:
    """Both floating precisions are dispatched."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_outputs_follow_the_input_dtype(self, dtype):
        """Energy, forces and virial come back in the precision they went in with."""
        system = _system("cuda:0", dtype=dtype)
        energy, forces, virial = _evaluate(system, compute_virial=True)
        assert energy.dtype == dtype
        assert forces.dtype == dtype
        assert virial.dtype == dtype
        assert torch.isfinite(energy).all()

    def test_single_and_double_agree_to_single_precision(self):
        """The float32 path tracks the float64 one to its own accuracy."""
        double = _system("cuda:0", dtype=torch.float64)
        single = _system("cuda:0", dtype=torch.float32)
        reference = float(_evaluate(double)[0])
        assert abs(float(_evaluate(single)[0]) - reference) < 1e-4 * abs(reference)


@pytest.mark.gpu
class TestDeviceAgreement:
    """The same pipeline on CPU and on CUDA.

    The Warp layer's end-to-end classes are GPU-only, so nothing else runs the mesh passes
    on CPU. That matters because the two devices take different paths through the block
    reductions: a block is a warp on CUDA and a single thread on CPU, and a reduction
    written against the wrong width silences most of the mesh while leaving the forces --
    which come from the cotangent field rather than a reduction -- untouched.
    """

    @pytest.mark.parametrize("mesh", [(16, 16, 16), (32, 32, 32)])
    def test_energy_forces_and_virial_match_between_devices(self, mesh):
        """Every output, not just the forces, has to agree across devices."""
        results = {}
        for device in ("cpu", "cuda:0"):
            system = _system(device)
            results[device] = _evaluate(
                system, mesh_dimensions=mesh, compute_virial=True
            )
        cpu, gpu = results["cpu"], results["cuda:0"]
        np.testing.assert_allclose(
            cpu[0].cpu().numpy(), gpu[0].cpu().numpy(), rtol=1e-11
        )
        np.testing.assert_allclose(
            cpu[1].cpu().numpy(),
            gpu[1].cpu().numpy(),
            atol=1e-11 * float(gpu[1].abs().max()),
        )
        np.testing.assert_allclose(
            cpu[2].cpu().numpy(),
            gpu[2].cpu().numpy(),
            atol=1e-11 * float(gpu[2].abs().max()),
        )


@pytest.mark.gpu
class TestTorchCompile:
    """The op has to survive tracing, and do so without falling out of the graph."""

    @staticmethod
    def _callable(system):
        """Close over everything but the positions, as an MD step would."""

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

        return evaluate

    def test_compiled_matches_eager(self):
        """Compilation does not change the result."""
        system = _system("cuda:0")
        evaluate = self._callable(system)
        eager_energy, eager_forces = evaluate(system["positions"])
        energy, forces = torch.compile(evaluate)(system["positions"])
        np.testing.assert_allclose(
            energy.cpu().numpy(), eager_energy.cpu().numpy(), rtol=1e-12
        )
        np.testing.assert_allclose(
            forces.cpu().numpy(),
            eager_forces.cpu().numpy(),
            atol=1e-12 * float(eager_forces.abs().max()),
        )

    def test_traces_without_graph_breaks(self):
        """No graph breaks.

        A break here would mean a device synchronisation inside the molecular-dynamics step
        this op exists to make cheap, which is the reason the species-coverage check is
        skipped while tracing.
        """
        import torch._dynamo as dynamo

        system = _system("cuda:0")
        explanation = dynamo.explain(self._callable(system))(system["positions"])
        assert explanation.graph_break_count == 0

    def test_repeated_calls_are_stable(self):
        """Calling the compiled function repeatedly keeps giving the same answer.

        Output buffers are freshly allocated and zeroed each call; a stale-buffer bug would
        show up as drift here.
        """
        system = _system("cuda:0")
        compiled = torch.compile(self._callable(system))
        first = float(compiled(system["positions"])[0])
        for _ in range(3):
            assert abs(float(compiled(system["positions"])[0]) - first) < 1e-12 * abs(
                first
            )


@pytest.mark.gpu
class TestPrecomputedSetup:
    """Cell- and mesh-derived quantities reused across steps."""

    def test_setup_rejects_mesh_dimension_below_spline_floor(self):
        """Precomputed setups enforce the same mesh floor as the call-time API."""
        system = _system("cuda:0")
        with pytest.raises(ValueError, match=r"at least max\(spline_order, 3\) = 5"):
            FourierD3Setup.build(
                system["cell"], system["params"].n_species, (4, 5, 5), spline_order=5
            )

    def test_matches_computing_them_inline(self):
        """Supplying the setup gives the same answer as letting the call derive it."""
        system = _system("cuda:0")
        setup = FourierD3Setup.build(system["cell"], system["params"].n_species, MESH)
        inline = _evaluate(system)
        reused = _evaluate(system, setup=setup)
        np.testing.assert_allclose(
            reused[0].cpu().numpy(), inline[0].cpu().numpy(), rtol=1e-12
        )
        np.testing.assert_allclose(
            reused[1].cpu().numpy(),
            inline[1].cpu().numpy(),
            atol=1e-12 * float(inline[1].abs().max()),
        )

    def test_enables_cuda_graph_capture(self):
        """A CUDA graph can be captured only when the setup is precomputed.

        ``torch.linalg.inv`` cannot be recorded into a graph, so deriving the cell inverse
        inside the call makes ``torch.compile(mode="reduce-overhead")`` fail. This is the
        test that pins that down; without it the failure would only appear to a user trying
        to speed up an MD loop.
        """
        system = _system("cuda:0")
        setup = FourierD3Setup.build(system["cell"], system["params"].n_species, MESH)

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
                setup=setup,
                **DAMPING,
            )

        expected = evaluate(system["positions"])
        compiled = torch.compile(evaluate, mode="reduce-overhead")
        for _ in range(3):
            actual = compiled(system["positions"])
        np.testing.assert_allclose(
            actual[0].cpu().numpy(), expected[0].cpu().numpy(), rtol=1e-10
        )

    def test_records_what_it_was_built_for(self):
        """The setup carries its mesh and spline order, so the call cannot disagree."""
        system = _system("cuda:0")
        setup = FourierD3Setup.build(
            system["cell"], system["params"].n_species, (16, 16, 16), spline_order=5
        )
        assert setup.mesh_dimensions == (16, 16, 16)
        assert setup.spline_order == 5
        # The call follows the setup rather than its own arguments.
        result = _evaluate(system, setup=setup, mesh_dimensions=MESH)
        coarse = _evaluate(system, mesh_dimensions=(16, 16, 16), spline_order=5)
        np.testing.assert_allclose(
            result[0].cpu().numpy(), coarse[0].cpu().numpy(), rtol=1e-12
        )
