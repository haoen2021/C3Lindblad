"""Supplement S5: exact, second-Born and GW exchange-split Green functions.

Implements Eqs. S97, S104, S111–S113 directly; no impurity solver is required.
"""

from pathlib import Path
import argparse

import numpy as np
from .parameters import parameters

PARAMETERS = parameters("exchange_model")


def responses(z, epsilon=PARAMETERS["epsilon"], U=PARAMETERS["U"], J=PARAMETERS["J"]):
    return {
        name: 1.0 / (z - epsilon - J**2 / (z - epsilon - shift))
        for name, shift in (("Exact", 0.0), ("2ndB", U - J), ("GW", U))
    }


def main(output):
    omega = np.linspace(*PARAMETERS["frequency"])
    z = omega + 1j * PARAMETERS["eta"]
    values = responses(z)
    # Independently check the exact two-state resolvent of Eq. S95.
    h = np.array([[PARAMETERS["epsilon"], -PARAMETERS["J"]], [-PARAMETERS["J"], PARAMETERS["epsilon"]]])
    exact = np.linalg.inv(z[:, None, None] * np.eye(2) - h)[:, 0, 0]
    np.testing.assert_allclose(values["Exact"], exact, rtol=1e-12, atol=1e-12)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "responses.npz", omega=omega, **values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/exchange_model"))
    main(parser.parse_args().output)
