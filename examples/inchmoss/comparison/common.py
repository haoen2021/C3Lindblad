"""Share the InchMOSS orbital rotation, continuum bath quadrature and local observables between comparison calculations."""

from pathlib import Path
from ...parameters import parameters

PARAMETERS = parameters("inchmoss")
import numpy as np

ROOT = Path.cwd() / "inchmoss/comparison"
BETA = PARAMETERS["beta"]
ROT = np.asarray(PARAMETERS["comparison"]["rotation"]) / np.sqrt(PARAMETERS["comparison"]["rotation_norm"])

BATH_MATRIX = ROT.T @ (PARAMETERS["bath"]["identical_leads"] *
                      np.asarray(PARAMETERS["bath"]["matrix_per_lead"])) @ ROT


def bath_quadrature(size=1024):
    from scipy.special import roots_legendre

    # Split at the steep band edges rather than fit the continuum.
    energies = []
    weights = []
    for lo, hi in [
        (-30.0, -26.0),
        (-26.0, -24.0),
        (-24.0, 24.0),
        (24.0, 26.0),
        (26.0, 30.0),
    ]:
        x, w = roots_legendre(size // 5)
        e = (hi + lo) / 2 + (hi - lo) * x / 2
        shape = np.exp(
            -np.logaddexp(0, PARAMETERS["bath"]["edge_steepness"] * (e - PARAMETERS["bath"]["omega_c"])) - np.logaddexp(0, -PARAMETERS["bath"]["edge_steepness"] * (e + PARAMETERS["bath"]["omega_c"]))
        )
        energies.extend(e)
        weights.extend(w * (hi - lo) / 2 * shape / (2 * np.pi))
    return np.array(energies), np.array(weights)


def operators():
    from triqs.operators import c, n, dagger, Operator

    ann = [
        sum((float(ROT[i, k]) * c(s, k) for k in range(2)), Operator())
        for i in range(2)
        for s in ("up", "dn")
    ]
    w = PARAMETERS["comparison"]["rotated_density_interaction"]
    h = PARAMETERS["comparison"]["rotated_one_body"]
    hint = sum(w[i][i] * n("up", i) * n("dn", i) for i in range(2))
    hint += w[0][1] * (n("up", 0) + n("dn", 0)) * (n("up", 1) + n("dn", 1))
    h0 = sum(h[i] * (n("up", i) + n("dn", i)) for i in range(2))

    def obs(a):
        occ = [dagger(x) * x for x in a]
        z = [(occ[2 * i] - occ[2 * i + 1]) / 2 for i in range(2)]
        plus = [dagger(a[2 * i]) * a[2 * i + 1] for i in range(2)]
        dot = z[0] * z[1] + (plus[0] * dagger(plus[1]) + dagger(plus[0]) * plus[1]) / 2
        return dict(
            occupations=[occ[0] + occ[1], occ[2] + occ[3]],
            double_occupations=[occ[0] * occ[1], occ[2] * occ[3]],
            spin_correlation=dot,
            spin_zz=z[0] * z[1],
        )

    rotated = [c(s, i) for i in range(2) for s in ("up", "dn")]
    quantum = [sum((n(s, i) for i in range(2)), Operator()) for s in ("up", "dn")]
    return hint, h0, dict(original=obs(ann), rotated=obs(rotated)), quantum


def expectations(mean, obs):
    return {
        basis: {
            k: [mean(x) for x in v] if isinstance(v, list) else mean(v)
            for k, v in values.items()
        }
        for basis, values in obs.items()
    }
