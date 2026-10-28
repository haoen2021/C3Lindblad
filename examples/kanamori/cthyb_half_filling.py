"""TRIQS/CT-HYB control for the full two-orbital Kanamori model at mu=1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray


D = 2.0
U = 2.0
J = 0.2
CHEMICAL_POTENTIAL = 1.0
INTERACTION = "full_kanamori_spin_exchange_and_pair_hopping"


def bethe_green(z: ArrayLike) -> NDArray[np.complex128]:
    """Return the normalized causal semicircle transform."""

    values = np.asarray(z, dtype=np.complex128)
    root = np.sqrt(values - D) * np.sqrt(values + D)
    root = np.where((root / values).real < 0.0, -root, root)
    return np.asarray(2.0 / (values + root), dtype=np.complex128)


def weiss_green(z: ArrayLike) -> NDArray[np.complex128]:
    """Return the two-orbital noninteracting impurity Green function."""

    values = np.asarray(z, dtype=np.complex128)
    bath = bethe_green(values)
    identity = np.eye(2, dtype=np.complex128)
    orbital = np.ones((2, 2), dtype=np.complex128)
    return np.asarray(
        [
            np.linalg.inv(
                (value + CHEMICAL_POTENTIAL) * identity - orbital * bath[index]
            )
            for index, value in enumerate(values)
        ]
    )


def _interaction():
    from triqs.operators import c, c_dag, n

    h_int = 0
    for i in (0, 1):
        h_int += U * n("dn", i) * n("up", i)
    for i in (0, 1):
        for j in (0, 1):
            if i != j:
                h_int += (U - 2.0 * J) * n("dn", i) * n("up", j)
    for spin in ("up", "dn"):
        h_int += (U - 3.0 * J) * n(spin, 1) * n(spin, 0)
    for i in (0, 1):
        for j in (0, 1):
            if i == j:
                continue
            exchange = c_dag("up", i) * c_dag("dn", j) * c("dn", i) * c("up", j)
            pair = c_dag("up", i) * c_dag("dn", i) * c("dn", j) * c("up", j)
            h_int += J * (exchange + pair)
    return h_int


def run(beta: float, cycles: int, seed: int, output: Path) -> dict[str, object]:
    """Run one independent serial CT-HYB chain."""

    from triqs_cthyb import Solver

    inverse_temperature = float(beta)
    if inverse_temperature not in (8.0, 64.0):
        raise ValueError("the CT-HYB control is fixed to beta 8 or 64")
    if int(cycles) <= 0:
        raise ValueError("cycles must be positive")

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    solver = Solver(
        beta=inverse_temperature,
        gf_struct=[("up", 2), ("dn", 2)],
        n_iw=256,
        n_tau=2001,
    )
    frequencies = np.asarray([complex(point) for point in solver.G0_iw.mesh])
    noninteracting = weiss_green(frequencies)
    for block in ("up", "dn"):
        for index in range(frequencies.size):
            solver.G0_iw[block].data[index] = noninteracting[index]

    started = perf_counter()
    solver.solve(
        h_int=_interaction(),
        n_cycles=int(cycles),
        n_warmup_cycles=max(5000, int(cycles) // 20),
        length_cycle=50,
        measure_G_tau=True,
        measure_pert_order=True,
        random_seed=int(seed),
        verbosity=2,
    )
    wall_seconds = perf_counter() - started

    tau = np.asarray([float(point) for point in solver.G_tau.mesh])
    G_tau_up = np.asarray(solver.G_tau["up"].data)
    G_tau_dn = np.asarray(solver.G_tau["dn"].data)
    G_iw_up = np.asarray(solver.G_iw["up"].data)
    G_iw_dn = np.asarray(solver.G_iw["dn"].data)
    endpoint_occupation = float(-0.5 * np.trace(G_tau_up[-1]).real)
    np.savez_compressed(
        destination / "compact.npz",
        beta=np.asarray(inverse_temperature),
        cycles=np.asarray(cycles, dtype=np.int64),
        seed=np.asarray(seed, dtype=np.int64),
        tau=tau,
        iw=frequencies,
        G_tau_up=G_tau_up,
        G_tau_dn=G_tau_dn,
        G_iw_up=G_iw_up,
        G_iw_dn=G_iw_dn,
    )

    record: dict[str, object] = {
        "beta": inverse_temperature,
        "chemical_potential": CHEMICAL_POTENTIAL,
        "interaction": INTERACTION,
        "sampling_weight": "trace",
        "cycles": int(cycles),
        "warmup_cycles": max(5000, int(cycles) // 20),
        "length_cycle": 50,
        "seed": int(seed),
        "average_sign": float(solver.average_sign),
        "average_order": float(solver.average_order),
        "auto_corr_time": float(solver.auto_corr_time),
        "solve_status": int(solver.solve_status),
        "G_tau_endpoint_occupation": endpoint_occupation,
        "endpoint_sum_defect": float(
            abs(
                0.5 * np.trace(G_tau_up[0]).real
                + 0.5 * np.trace(G_tau_up[-1]).real
                + 1.0
            )
        ),
        "spin_defect": float(np.max(abs(G_tau_up - G_tau_dn))),
        "orbital_diagonal_defect": float(
            np.max(abs(G_tau_up[:, 0, 0] - G_tau_up[:, 1, 1]))
        ),
        "reciprocity_defect": float(np.max(abs(G_tau_up[:, 0, 1] - G_tau_up[:, 1, 0]))),
        "maximum_prehermitization_asymmetry": float(
            max(
                np.max(abs(solver.asymmetry_G_tau["up"].data)),
                np.max(abs(solver.asymmetry_G_tau["dn"].data)),
            )
        ),
        "wall_seconds": wall_seconds,
    }
    (destination / "run.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--cycles", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.beta,
                arguments.cycles,
                arguments.seed,
                arguments.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
