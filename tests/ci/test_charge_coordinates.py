"""Charge-only coordinates have no spectator spin block."""

from math import comb

import numpy as np
import pytest

from c3l.cc.component_tensor import (
    ComponentCoordinateShell,
    ComponentCoordinates,
    _sample_charge_shell,
)
from c3l.ci.basis import ness_basis, response_basis
from c3l.ci.sectors import (
    canonical_mode_layout,
    full_ness_dimension,
    full_response_dimension,
    mode_layout_from_references,
    ness_dimension,
    response_dimension,
)
from c3l.ci.tilde import tilde_map_from_references
from c3l.superfermion.normal_modes import normal_mode_reference_from_arrays


@pytest.mark.parametrize("num_orbitals", [1, 2, 3, 4])
def test_charge_basis_rank_dimensions_and_tilde_closure(num_orbitals):
    layout = canonical_mode_layout(num_orbitals, symmetry="charge")
    assert layout.num_modes == 2 * num_orbitals
    np.testing.assert_array_equal(
        layout.mode_charges,
        [(1, 0)] * num_orbitals + [(-1, 0)] * num_orbitals,
    )
    neutral = ness_basis(num_orbitals, num_orbitals, layout)
    response = response_basis(num_orbitals, num_orbitals, layout=layout)
    for rank in range(num_orbitals + 1):
        assert neutral.ranks.count(rank) == comb(num_orbitals, rank) ** 2
        assert response.ranks.count(rank) == (
            2 * comb(num_orbitals, rank + 1) * comb(num_orbitals, rank)
        )
    assert neutral.dimension == comb(2 * num_orbitals, num_orbitals)
    assert response.dimension == 2 * comb(2 * num_orbitals, num_orbitals + 1)
    assert neutral.is_tilde_closed()
    assert response.is_tilde_closed()
    retained = response_basis(num_orbitals, 1, layout=layout)
    assert retained.dimension == 2 * (
        num_orbitals + comb(num_orbitals, 2) * num_orbitals
    )
    for basis in (neutral, response):
        assert max(basis.masks) < 1 << (2 * num_orbitals)
        coordinates = ComponentCoordinates.from_masks(num_orbitals, basis.masks)
        assert all(block.key[2:] == (0, 0) for block in coordinates.blocks)
        values = np.arange(basis.dimension) + 0.25j
        np.testing.assert_array_equal(coordinates.pack(coordinates.unpack(values)), values)


@pytest.mark.parametrize("num_orbitals", [1, 2, 3, 4])
def test_charge_dimension_metadata(num_orbitals):
    for rank in range(num_orbitals + 2):
        assert ness_dimension(num_orbitals, rank, symmetry="charge") == sum(
            comb(num_orbitals, count) ** 2
            for count in range(min(rank, num_orbitals) + 1)
        )
        assert response_dimension(num_orbitals, rank, symmetry="charge") == 2 * sum(
            comb(num_orbitals, count + 1) * comb(num_orbitals, count)
            for count in range(min(rank, num_orbitals) + 1)
        )
    assert full_ness_dimension(num_orbitals, symmetry="charge") == comb(
        2 * num_orbitals, num_orbitals
    )
    assert full_response_dimension(num_orbitals, symmetry="charge") == 2 * comb(
        2 * num_orbitals, num_orbitals + 1
    )


def test_charge_response_rejects_second_spin():
    layout = canonical_mode_layout(3, symmetry="charge")
    with pytest.raises(ValueError, match="response_spin.*0"):
        response_basis(3, 1, response_spin=1, layout=layout)
    with pytest.raises(ValueError, match="response_spin.*0"):
        ComponentCoordinateShell(3, 1, 1, 5, symmetry="charge")


@pytest.mark.parametrize("num_orbitals", [1, 2, 3])
def test_single_reference_layout_and_antilinear_tilde(num_orbitals):
    energies = np.diag(np.linspace(-0.3, 0.7, num_orbitals)).astype(complex)
    if num_orbitals > 1:
        energies[0, 1] = 0.13j
        energies[1, 0] = -0.13j
    reference = normal_mode_reference_from_arrays(
        energies,
        np.diag(np.linspace(0.2, 0.5, num_orbitals)),
        np.diag(np.linspace(0.4, 0.1, num_orbitals)),
    )
    layout = mode_layout_from_references(reference)
    tilde = tilde_map_from_references(reference)
    assert layout.symmetry == tilde.symmetry == "charge"
    assert layout.num_modes == tilde.num_modes == 2 * num_orbitals
    np.testing.assert_array_equal(layout.tilde_partners, tilde.partners)
    for mask in range(1 << layout.num_modes):
        partner, factor = tilde.determinant(mask)
        original, reverse_factor = tilde.determinant(partner)
        assert partner == layout.tilde_mask(mask)
        assert original == mask
        assert factor.conjugate() * reverse_factor == pytest.approx(1.0)
    spin_layout = mode_layout_from_references(reference, reference)
    spin_tilde = tilde_map_from_references(reference, reference)
    assert spin_layout.symmetry == spin_tilde.symmetry == "spin"
    assert spin_layout.num_modes == spin_tilde.num_modes == 4 * num_orbitals


@pytest.mark.parametrize("num_orbitals", [1, 2, 3, 4])
@pytest.mark.parametrize("charge", [-1, 1])
def test_charge_omitted_response_shell_has_only_real_modes(num_orbitals, charge):
    shell = ComponentCoordinateShell(num_orbitals, 0, charge, 5, symmetry="charge")
    assert shell.dimension == comb(num_orbitals, 3) * comb(num_orbitals, 2)
    masks = set()
    for tile in shell.tiles(2):
        assert tile.key[2:] == (0, 0)
        groups = [group for group, count in enumerate(tile.key) for _ in range(count)]
        for indices in tile.indices:
            masks.add(sum(1 << (group * num_orbitals + index) for group, index in zip(groups, indices)))
    layout = canonical_mode_layout(num_orbitals, symmetry="charge")
    expected = {
        mask for mask in range(1 << (2 * num_orbitals))
        if mask.bit_count() == 5 and layout.charge(mask) == (charge, 0)
    }
    assert masks == expected


def test_charge_sampled_neutral_shell_counts_actual_coordinates():
    sample, dimension = _sample_charge_shell(4, 0, 6, 16, 71, symmetry="charge")
    assert dimension == sample.dimension == 16
    assert len(sample.blocks) == 1
    assert sample.blocks[0].key == (3, 3, 0, 0)
    assert np.unique(sample.blocks[0].indices, axis=0).shape[0] == 16


def test_default_spin_coordinates_keep_existing_dimensions():
    layout = canonical_mode_layout(2)
    assert layout.symmetry == "spin"
    assert layout.num_modes == 8
    assert ness_basis(2, 1).dimension == ness_dimension(2, 1) == 9
    assert response_basis(2, 1).dimension == response_dimension(2, 1) == 24
    assert full_ness_dimension(2) == 36
    assert full_response_dimension(2) == 48
    assert ComponentCoordinateShell(2, 0, 1, 5).dimension == 10
