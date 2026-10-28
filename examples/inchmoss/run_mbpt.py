"""Continuum self-consistent 2B/GW Fig. 6 comparison, reusing wire solvers."""

from pathlib import Path
import argparse
from ..parameters import parameters

PARAMETERS = parameters("inchmoss")
import json
from dataclasses import asdict
from time import perf_counter
import numpy as np
from scipy.signal import hilbert
from scipy.special import expit
from scipy.integrate import quad


REPO = Path.cwd()
from ..shared.second_born import Solver, Config, adjoint
from ..shared.gw import GWSolver
from .run import wide_band_shape
from .comparison.common import BATH_MATRIX

OUT = REPO / "results/inchmoss_fig6_mbpt_v1"
ROT = np.asarray(PARAMETERS["comparison"]["rotation"]) / np.sqrt(PARAMETERS["comparison"]["rotation_norm"])


class BathAdapter:
    def __init__(self, config):
        super().__init__(config, [0.0],
                         np.diag(PARAMETERS["comparison"]["rotated_one_body"]),
                         PARAMETERS["comparison"]["rotated_density_interaction"])
        xp = self.xp
        self.eye = xp.eye(2, dtype=xp.complex128)
        self.h = xp.asarray(np.diag(PARAMETERS["comparison"]["rotated_one_body"]))
        self.w = xp.asarray(PARAMETERS["comparison"]["rotated_density_interaction"])

    def embedding(self):
        xp = self.xp
        d = wide_band_shape(self.omega)
        # Large zero padding suppresses periodic Hilbert images of the finite band.
        pad = 16 * self.config.n
        real = hilbert(np.pad(d, (pad, pad))).imag[pad : pad + d.size] / 2
        self.embedding_check = max(
            abs(
                np.interp(w, self.omega, real)
                + quad(
                    lambda x: float(wide_band_shape(np.array(x))),
                    -32.0,
                    32.0,
                    weight="cauchy",
                    wvar=w,
                    epsabs=1e-10,
                    limit=300,
                )[0]
                / (2 * np.pi)
            )
            for w in (0.0, 2.0, 8.0, 24.0)
        )
        ret = (real - 0.5j * d)[:, None, None] * BATH_MATRIX
        gamma = d[:, None, None] * BATH_MATRIX
        f = expit(-self.config.beta * self.omega)
        return (
            xp.asarray(ret[None]),
            xp.asarray((1j * f[:, None, None] * gamma)[None]),
            xp.asarray((-1j * (1 - f[:, None, None]) * gamma)[None]),
            xp.asarray(gamma[None, None]),
            xp.asarray(f[None, None]),
        )

    def initial_state(self):
        xp = self.xp
        return (
            xp.asarray(np.diag([0.0, 4.0])[None], dtype=xp.complex128),
            *(
                xp.zeros((1, self.config.n, 2, 2), dtype=xp.complex128)
                for _ in range(3)
            ),
        )


class Born(BathAdapter, Solver):
    pass


class GW(BathAdapter, GWSolver):
    pass


def rotate(a):
    return ROT @ a @ ROT.T


def time_green(w, a, time):
    return np.stack(
        [
            -1j * (w[1] - w[0]) * np.einsum("w,wij->ij", np.exp(-1j * w * t), a)
            for t in time
        ]
    )


def solve(method, level, n, window, padding, *, device="gpu"):
    path = OUT / f"{method}_{level}.npz"
    if path.exists():
        return dict(np.load(path)), json.loads(path.with_suffix(".json").read_text())
    c = Config(
        n=n,
        window=window,
        padding=padding,
        beta=PARAMETERS["beta"],
        mu=0.0,
        backend=device,
        mixing=0.1,
        tolerance=1e-8,
        max_iterations=4000,
    )
    start = perf_counter()
    solver = ({"2B": Born, "GW": GW}[method])(c)

    def progress(i, state, defects):
        print(method, level, i, float(defects.max()), flush=True)

    state, greens, defects, history = solver.solve(callback=progress)
    if defects.max() > c.tolerance:
        raise RuntimeError(f"{method} did not converge: {defects}")
    gr, gl, gg = map(np.asarray, greens)
    spectral = -(gr - adjoint(gr)) / (2j * np.pi)
    rho = -1j * np.sum(gl[0], axis=0) * solver.dw / (2 * np.pi)
    f = expit(-c.beta * solver.omega)
    fdt = np.max(abs(gl + f[None, :, None, None] * (gr - adjoint(gr))))
    a = rotate(spectral[0])
    g = rotate(gr[0])
    rec = dict(
        method=method,
        level=level,
        config=asdict(c),
        device=solver.device,
        seconds=perf_counter() - start,
        iterations=len(history),
        fixed_point_defect=float(defects.max()),
        bath_quadrature_max_error=float(solver.embedding_check),
        weight=np.diag(np.sum(a, axis=0) * solver.dw).real.tolist(),
        density_eigenvalues=np.linalg.eigvalsh(rho).tolist(),
        minimum_spectral_eigenvalue=float(np.linalg.eigvalsh(a).min()),
        fdt_max_error=float(fdt),
        free_channel_self_energy_max=float(
            max(abs(np.asarray(s)[:, :, 0, 0]).max() for s in state[1:])
        ),
    )
    if (
        rec["bath_quadrature_max_error"] > 1e-4
        or np.max(abs(np.array(rec["weight"]) - 1)) > 5e-4
        or fdt > 1e-5
    ):
        raise RuntimeError(f"Validation failed: {rec}")
    np.savez_compressed(
        path,
        omega=solver.omega,
        A=a,
        G_R=g,
        rho_rotated=rho,
        hf=np.asarray(state[0]),
        sigma_R=np.asarray(state[1]),
        defect_history=history,
    )
    path.with_suffix(".json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec), flush=True)
    return dict(np.load(path)), rec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    omega = np.linspace(*PARAMETERS["frequency"])
    time = np.linspace(*PARAMETERS["comparison"]["time"])
    records = {}
    arrays = {}
    for method in ("2B", "GW"):
        base, rb = solve(method, "base", 32768, 48.0, 4, device=args.device)
        fine, rf = solve(method, "refined", 65536, 64.0, 8, device=args.device)

        def sample(d):
            return np.stack(
                [
                    np.interp(omega, d["omega"], d["A"][:, i, j].real)
                    for i, j in ((0, 0), (0, 1), (1, 1))
                ],
                axis=1,
            )

        a = sample(fine)
        b = sample(base)
        delta = float(np.linalg.norm(a - b) / np.linalg.norm(a))
        if delta > 2e-3:
            raise RuntimeError(f"{method} refinement error {delta}")
        gt = time_green(fine["omega"], fine["A"], time)
        records[method] = {"base": rb, "refined": rf, "refinement_relative_L2": delta}
        arrays[f"A_{method}"] = a
        arrays[f"G_time_{method}"] = gt
    np.savez_compressed(
        OUT / "comparison.npz",
        omega=omega,
        time=time,
        **arrays,
    )
    (OUT / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
    print("COMPLETE", json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
