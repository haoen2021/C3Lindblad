"""Host-only static layout and tilde coordinate maps."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StaticLayout:
    mode_count: int
    dimension: int
    real_dimension: int
    coordinate_keys: tuple
    coordinate_sizes: tuple
    signatures: tuple
    capacities: tuple
    tile_size: int


def _tilde_coordinate_maps(tilde):
    """Return sparse vectorized maps between real and full complex coordinates."""

    unpack_first = np.empty(tilde.dimension, dtype=np.int64)
    unpack_second = np.zeros(tilde.dimension, dtype=np.int64)
    unpack_first_coefficient = np.empty(tilde.dimension, dtype=np.complex128)
    unpack_second_coefficient = np.zeros(tilde.dimension, dtype=np.complex128)
    pack_indices = np.empty(tilde.real_size, dtype=np.int64)
    pack_weights = np.empty(tilde.real_size, dtype=np.complex128)
    cursor = 0
    for index in tilde.independent:
        partner = tilde.partner[index]
        if index == partner:
            scale = tilde._self_scale(index)
            unpack_first[index] = cursor
            unpack_first_coefficient[index] = scale
            pack_indices[cursor] = index
            pack_weights[cursor] = 1.0 / scale
            cursor += 1
        else:
            unpack_first[index] = cursor
            unpack_second[index] = cursor + 1
            unpack_first_coefficient[index] = 1.0
            unpack_second_coefficient[index] = 1.0j
            unpack_first[partner] = cursor
            unpack_second[partner] = cursor + 1
            unpack_first_coefficient[partner] = tilde.phase[index]
            unpack_second_coefficient[partner] = -1.0j * tilde.phase[index]
            pack_indices[cursor : cursor + 2] = index
            pack_weights[cursor : cursor + 2] = (1.0, -1.0j)
            cursor += 2
    if cursor != tilde.real_size:
        raise RuntimeError("tilde coordinate lowering did not consume all variables")
    return (
        unpack_first,
        unpack_second,
        unpack_first_coefficient,
        unpack_second_coefficient,
        pack_indices,
        pack_weights,
    )
