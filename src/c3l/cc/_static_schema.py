"""Separate fixed contraction structure from numerical model coefficients.

A schema declares the site count, interacting sites, spin-conserving operator
patterns, and tile capacities. lower() fills these arrays with the current
interaction and HF coefficients, including zero padding. Parameter changes
within the same shapes can reuse JAX compilation; changes to the structure,
dtype, or device may require compilation again. This is an internal numerical
layout, not an additional model that a scientist needs to construct.
"""

from collections import Counter
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from c3l import SparseTwoBodyInteraction
from c3l.cc import component_tensor
from ._static_layout import _tilde_coordinate_maps
from c3l.cc.newton_krylov import rapidity_diagonal
from c3l.cc.s135_jax import _assembly_gather_lookup, _coordinate_gather_maps, _lower_coordinates
from c3l.ci.matrix_free import physical_mode_operator
from ._static_layout import StaticLayout


CHANNELS = ("c", "c_dagger", "tilde_c", "tilde_c_dagger")


def _factor_signature(spin, channel):
    sharp = channel in ("c_dagger", "tilde_c")
    creator_group = 2 * spin + (0 if sharp else 1)
    annihilator_group = 2 * spin + (1 if sharp else 0)
    creator = tuple(int(group == creator_group) for group in range(4))
    return (creator,), (annihilator_group,)


@dataclass(frozen=True)
class StructuralSchema:
    n_sites: int
    interacting_sites: tuple
    tile_size: int
    signatures: tuple
    capacities: tuple
    symmetry: str = "spin"


def make_schema(n_sites, interacting_sites, tile_size=4, *, symmetry="spin"):
    if symmetry not in ("spin", "charge"):
        raise ValueError("symmetry must be 'spin' or 'charge'")
    if isinstance(n_sites, bool) or not isinstance(n_sites, int) or n_sites < 1:
        raise ValueError("site count must be a positive integer")
    sites = tuple(interacting_sites)
    if any(isinstance(site, bool) or not isinstance(site, int) or not 0 <= site < n_sites for site in sites) or len(set(sites)) != len(sites):
        raise ValueError("interacting sites must be distinct in-range integers")
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size not in (1, 2, 4, 8):
        raise ValueError("tile size must be one of 1, 2, 4, 8")
    counts = Counter()
    block_count = 1 if symmetry == "charge" else 2
    for spin in range(block_count):
        sharp = _factor_signature(spin, "c_dagger")
        dual = _factor_signature(spin, "c")
        counts[(dual, sharp)] = 4 * n_sites ** 2 + 2
        counts[(sharp, dual)] = 4 * n_sites ** 2 + 2
    pairs = tuple(combinations(range(block_count * len(sites)), 2))
    for creation, annihilation in product(pairs, repeat=2):
        if symmetry == "spin" and sorted(orbital % 2 for orbital in creation) != sorted(orbital % 2 for orbital in annihilation):
            continue
        for creator_channel, annihilator_channel in (("c_dagger", "c"), ("tilde_c_dagger", "tilde_c")):
            sequence = ((creation[0], creator_channel), (creation[1], creator_channel), (annihilation[1], annihilator_channel), (annihilation[0], annihilator_channel))
            signature = tuple(_factor_signature(orbital % block_count, channel) for orbital, channel in reversed(sequence))
            counts[signature] += 1
    signatures = tuple(sorted(counts))
    capacities = tuple(tile_size * ((counts[signature] + tile_size - 1) // tile_size) for signature in signatures)
    return StructuralSchema(n_sites, tuple(sorted(sites)), tile_size, signatures, capacities, symmetry)


def interaction_from_spatial_tensor(values):
    tensor = np.asarray(values, dtype=np.complex128)
    if tensor.ndim != 4 or len(set(tensor.shape)) != 1 or tensor.shape[0] < 1:
        raise ValueError("spatial interaction must have shape (K,K,K,K)")
    if not np.all(np.isfinite(tensor)):
        raise ValueError("spatial interaction must be finite")
    terms = []
    for indices in np.ndindex(tensor.shape):
        first, second, third, fourth = indices
        for spin, other_spin in product((0, 1), repeat=2):
            creation = (2 * first + spin, 2 * second + other_spin)
            annihilation = (2 * fourth + spin, 2 * third + other_spin)
            if creation[0] != creation[1] and annihilation[0] != annihilation[1]:
                terms.append((*creation, *annihilation, complex(tensor[indices])))
    return SparseTwoBodyInteraction(2 * tensor.shape[0], tuple(terms))


def lower(problem, schema):
    oracle = problem.oracle
    if oracle.model.symmetry != schema.symmetry:
        raise ValueError("problem symmetry differs from the declared schema")
    if problem.coordinates.n != schema.n_sites:
        raise ValueError("problem site count differs from the declared schema")
    allowed = set(schema.interacting_sites)
    if any(oracle.model.orbital_site_spin(orbital)[0] not in allowed for term in oracle.model.interaction.terms for orbital in term[:4]):
        raise ValueError("interaction lies outside the declared interacting sites")
    expected_vacuum = np.arange(2 * schema.n_sites) >= schema.n_sites
    for reference in oracle.reference.mode_references:
        if not np.array_equal(reference.vacuum_mask, expected_vacuum):
            raise ValueError("reference must use the canonical spin-major particle/hole layout")
    coordinates = _lower_coordinates(problem.coordinates)
    layout = StaticLayout(schema.n_sites, oracle.num_amplitudes, oracle.num_variables, tuple(block.key for block in coordinates), tuple(len(block.indices) for block in coordinates), schema.signatures, schema.capacities, schema.tile_size)
    block_count = len(oracle.reference.mode_references)
    factor_bank = np.zeros((4 * block_count * schema.n_sites, 2, 4, schema.n_sites), dtype=np.complex128)
    factor_records = {}
    for site, spin, channel_index in product(range(schema.n_sites), range(block_count), range(4)):
        channel = CHANNELS[channel_index]
        factor = physical_mode_operator(oracle.reference, site, spin, channel)
        signature = _factor_signature(spin, channel)
        bank_index = (block_count * site + spin) * 4 + channel_index
        linear = component_tensor._linear_data(factor, schema.n_sites)
        if not set(linear.creator.blocks).issubset(signature[0]) or not set(linear.annihilator).issubset(signature[1]):
            raise ValueError("factor support is outside the declared spin/operator family")
        for key, values in linear.creator.blocks.items():
            factor_bank[bank_index, 0, key.index(1)] = values
        for group, values in linear.annihilator.items():
            factor_bank[bank_index, 1, group] = values
        factor_records[factor.label] = (bank_index, signature, factor.components)
    grouped = {signature: [] for signature in schema.signatures}
    for record in oracle.products:
        factors = []
        for factor in reversed(record.factors):
            if factor.label not in factor_records:
                raise ValueError("unknown physical operator identity")
            bank_index, signature, components = factor_records[factor.label]
            if factor.components != components:
                raise ValueError("physical factor identity and numerical components disagree")
            factors.append((bank_index, signature))
        signature = tuple(item[1] for item in factors)
        if signature not in grouped:
            raise ValueError("product is outside the declared spin/operator family")
        grouped[signature].append((complex(record.coefficient), tuple(item[0] for item in factors)))
    groups = []
    for signature, capacity in zip(schema.signatures, schema.capacities, strict=True):
        entries = grouped[signature]
        if len(entries) > capacity:
            raise ValueError("product count exceeds the declared structural capacity")
        coefficients = np.zeros(capacity, dtype=np.complex128)
        indices = np.zeros((capacity, len(signature)), dtype=np.int32)
        for position, (coefficient, factor_ids) in enumerate(entries):
            coefficients[position] = coefficient
            indices[position] = factor_ids
        active_tiles = np.asarray((len(entries) + schema.tile_size - 1) // schema.tile_size, dtype=np.int32)
        groups.append((coefficients.reshape(-1, schema.tile_size), indices.reshape(-1, schema.tile_size, len(signature)), active_tiles))
    data = dict(
        groups=tuple(groups), factor_bank=factor_bank,
        gather=tuple(_coordinate_gather_maps(block, schema.n_sites) for block in coordinates),
        indices=tuple(block.indices for block in coordinates),
        assembly=_assembly_gather_lookup(tuple(block.vector_positions for block in coordinates), problem.coordinates.dimension),
        tilde=_tilde_coordinate_maps(oracle.tilde), diagonal=np.asarray(rapidity_diagonal(oracle), dtype=np.complex128),
    )
    return layout, data
