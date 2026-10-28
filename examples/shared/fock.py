"""Small physical-Fock operators for the paper molecular observables."""

import numpy as np


def fock_annihilators(count):
    ops = []
    for p in range(count):
        a = np.zeros((1 << count, 1 << count), complex)
        for state in range(1 << count):
            if state >> p & 1:
                a[state ^ (1 << p), state] = (-1) ** (
                    (state & ((1 << p) - 1)).bit_count()
                )
        ops.append(a)
    return ops
