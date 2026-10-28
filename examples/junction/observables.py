"""Fig. 4 observables: neutral-transfer strengths, currents, and removal sticks.

All covariances use gamma[p,q] = <a_q^dagger a_p>, in site-major spin order.
Only paper observables and necessary convergence checks are retained.
No bath Hilbert-space state or density matrix is constructed.
"""

from ..parameters import parameters

import json
import argparse
from scipy.special import expit
from .model import ROOT, load_input, assemble_model, RESPONSE_BIASES
from ..shared.fock import fock_annihilators
from ..shared.cc_io import save_npz, write_json
import time

import numpy as np


def _diagnostics(gamma_full_spin, four):
    """Required h1/l1 transfer moments and necessary CC physicality checks."""
    gamma = np.asarray(gamma_full_spin, dtype=complex)
    impurity = gamma[:8, :8]
    n = np.diag(impurity)  # n[p] = <n_p>, with p = (site, spin).
    nn = np.diag(n)  # nn[p,q] = <n_p n_q>; n_p^2 = n_p.
    for p in range(8):
        for q in range(p + 1, 8):
            nn[p, q] = nn[q, p] = four(p, p, q, q)
    # <S_h1 . S_l1> = <Sz_h1 Sz_l1> + ( <S+_h1 S-_l1> + h.c. ) / 2.
    spin = (nn[0, 4] - nn[0, 5] - nn[1, 4] + nn[1, 5]) / 4 + (
        four(0, 1, 5, 4) + four(1, 0, 4, 5)
    ) / 2
    return dict(
        site_occupations=n.reshape(4, 2).sum(axis=1),
        density_pairs=nn,
        h1_l1_spin_correlation=spin,
        maximum_observable_imaginary_defect=float(
            np.max(abs(np.r_[n, nn.ravel(), spin].imag))
        ),
    )


def cc_observables(problem_factorized, amplitudes):
    """Return full auxiliary covariance and rank-four-exact CCSD diagnostics.

    Only the creator-rank 0/2/4 prefix of exp(T)|rho_ref> contributes to the
    at-most-four physical operators used here; no higher ranks are needed.
    """
    from c3l.cc.observables import low_rank_cc_coefficients
    from c3l.ci.matrix_free import (
        FactorizedModeProduct,
        factorized_mode_product_expectation,
        physical_mode_operator,
    )

    started = time.monotonic()
    reference = problem_factorized.reference
    state = low_rank_cc_coefficients(
        reference, problem_factorized.cluster_basis, amplitudes
    )
    count = 2 * reference.n_sites
    operators = {
        (p, dagger): physical_mode_operator(
            reference, p // 2, p % 2, "c_dagger" if dagger else "c"
        )
        for p in range(count)
        for dagger in (False, True)
    }

    def expectation(indices):
        product = FactorizedModeProduct(
            "radical_dimer_static_observable",
            1.0 + 0j,
            tuple(operators[key] for key in indices),
        )
        return factorized_mode_product_expectation(
            state.basis, state.coefficients, product, target_mask=0
        )

    gamma = np.empty((count, count), dtype=complex)
    for p in range(count):
        for q in range(count):
            gamma[p, q] = expectation(((q, True), (p, False)))

    def four(p, q, r, s):
        return expectation(((p, True), (q, False), (r, True), (s, False)))

    result = _diagnostics(gamma, four)
    result.update(
        method="factorized CCSD rank-0/2/4 expectations",
        low_rank_coefficient_dimension=state.basis.dimension,
        observables_wall_seconds=time.monotonic() - started,
    )
    return gamma, result


def strengths(nh, nl, nn, spin):
    """W0 and W1 for scalar/vector l1 -> h1 transfer (transfer operators in Supplement S4).

    nh=<n_h1>, nl=<n_l1>, nn=<n_h1 n_l1>, spin=<S_h1 . S_l1>.
    R=W1/(3 W0); W1 sums all three vector components.
    Direct operator contraction gives the -3*nn/4 term in W1.
    """
    x = 2 * nl - nn  # <(2 - n_h1) n_l1>: occupied l1, available h1.
    w0, w1 = x / 4 - spin, 3 * x / 4 + spin
    return {
        "W0": float(w0),
        "W1": float(w1),
        "R": float(w1 / (3 * w0)) if w0 != 0 else float("nan"),
    }


def molecule():
    data = parameters("junction")["model"]
    a = fock_annihilators(8)
    n = [v.T @ v for v in a]
    sz = [(n[2 * i] - n[2 * i + 1]) / 2 for i in range(4)]
    sp = [a[2 * i].T @ a[2 * i + 1] for i in range(4)]
    h = np.asarray(data["one_body_matrix"])
    H = sum(
        h[i, j] * a[2 * i + s].T @ a[2 * j + s]
        for i, j in np.ndindex(4, 4)
        for s in range(2)
    )
    for p, q, r, s, v in data["canonical_two_body_terms"]:
        H += v * a[p].T @ a[q].T @ a[s] @ a[r]
    z = sum(sz)
    p = sum(sp)
    S2 = z @ z + (p @ p.T + p.T @ p) / 2
    np.testing.assert_allclose(H, H.T.conj(), atol=1e-12)
    np.testing.assert_allclose(H @ S2, S2 @ H, atol=1e-11)
    return data, a, H, S2


def diagonalize(H, S2):
    es = []
    vs = []
    number = []
    for up in range(5):
        for dn in range(5):
            if up + dn not in (3, 4, 5):
                continue
            ids = [
                m
                for m in range(256)
                if (m & 85).bit_count() == up and (m & 170).bit_count() == dn
            ]
            e, v = np.linalg.eigh(H[np.ix_(ids, ids)])
            # Resolve accidental degeneracies by total spin before assigning labels.
            for value in np.unique(np.round(e, 9)):
                ix = np.flatnonzero(abs(e - value) < 1e-8)
                if len(ix) > 1:
                    block = v[:, ix].conj().T @ S2[np.ix_(ids, ids)] @ v[:, ix]
                    _, u = np.linalg.eigh(block)
                    v[:, ix] = v[:, ix] @ u
            full = np.zeros((256, len(e)), complex)
            full[ids] = v
            es.extend(e)
            vs.extend(full.T)
            number.extend([up + dn] * len(e))
    e = np.array(es)
    v = np.array(vs).T
    number = np.array(number)
    order = np.lexsort((e, number))
    e, v, number = e[order], v[:, order], number[order]
    spin2 = np.einsum("ij,ij->j", v.conj(), S2 @ v).real
    np.testing.assert_allclose(spin2, np.round(4 * spin2) / 4, atol=1e-9)
    return e, v, number, spin2


def dump_currents():
    out = ROOT / "results/bias_production"
    out.mkdir(parents=True, exist_ok=True)
    data = load_input()
    us = np.asarray(data["leads"]["u_substrate"])
    ut = np.asarray(data["leads"]["tip_geometries"]["center"])
    phis = RESPONSE_BIASES[1:]
    config = parameters("junction")
    diagnostics = {}

    def read(path):
        with np.load(path) as f:
            return {k: f[k].copy() for k in f.files}

    def adj(x):
        return x.conj().swapaxes(-1, -2)

    def bath_noise(w, phi, pm=False):
        if not pm:
            result = {}
            for name, mu, strength, u in [("s", 0.0, 0.8, us), ("t", phi, 0.08, ut)]:
                gamma = strength * np.sqrt(np.maximum(0.0, 1 - (w - mu) ** 2 / 16)) / 4
                f = expit(-10 * (w - mu))
                result[name] = tuple(
                    (gamma * x)[:, None, None] * np.outer(u, u.conj())
                    for x in (f, 1 - f)
                )
            return result
        model = assemble_model(ROOT / config["bath_files"]["4"]["path"], phi)
        width = (model.n_sites - 4) // 2
        result = {}
        for name, lo in [("s", 4), ("t", 4 + width)]:
            ix = slice(lo, lo + width)
            T = model.E[:4, ix]
            gain, loss = model.gamma_gain[ix, ix], model.gamma_loss[ix, ix]
            g = np.linalg.inv(
                w[:, None, None] * np.eye(width) - model.E[ix, ix] + 1j * (gain + loss)
            )
            result[name] = tuple(
                2 * T @ g @ d @ adj(g) @ T.conj().T for d in (gain, loss)
            )
        return result

    def currents(
        phi, name, w, ar, aa, bath, converged, static_current=None, gamma=None
    ):
        influx, outflux, net = {}, {}, {}
        for lead, (bin_, bout) in bath.items():
            influx[lead] = 2 * np.trace(bin_ @ aa, axis1=-2, axis2=-1).real
            outflux[lead] = 2 * np.trace(bout @ ar, axis1=-2, axis2=-1).real
            net[lead] = influx[lead] - outflux[lead]
        integrated = {k: float(np.trapezoid(v, w)) for k, v in net.items()}
        info = dict(
            residual_converged=converged,
            integrated_currents=integrated,
            continuity_integral=sum(integrated.values()),
            removal_min_eigenvalue=float(np.linalg.eigvalsh((ar + adj(ar)) / 2).min()),
            addition_min_eigenvalue=float(np.linalg.eigvalsh((aa + adj(aa)) / 2).min()),
            sum_rule_matrix_error=float(
                np.max(abs(np.trapezoid(ar + aa, w, axis=0) - np.eye(4)))
            ),
        )
        if static_current is not None:
            info["static_current_max_error"] = max(
                abs(integrated[k] - static_current[k]) for k in ("s", "t")
            )
        if gamma is not None:
            info["removal_occupation_max_error"] = float(
                np.max(abs(np.trapezoid(ar, w, axis=0) - gamma))
            )
        info["current_consistency_pass"] = bool(
            converged
            and abs(info["continuity_integral"]) < 1e-5
            and info.get("static_current_max_error", 0.0) < 1e-5
        )
        diagnostics[f"{phi:+.2f}_{name}"] = info
        save_npz(
            out / f"currents_phi_{phi:+.2f}_{name.replace(' ', '_')}.npz",
            omega=w,
            J_tip=-net["t"],
            I_tip=-integrated["t"],
            phi=phi,
            converged=info["current_consistency_pass"],
        )

    for phi in phis:
        base = out / f"phi_{phi:+.2f}"
        for method, label in [("GW", "scGW"), ("2B", "sc2ndB")]:
            folder = base / method
            if (
                not (folder / "spectra.npz").exists()
                or not (folder / "result.json").exists()
            ):
                continue
            f = read(folder / "spectra.npz")
            meta = json.loads((folder / "result.json").read_text())
            assert float(f["phi"]) == phi and float(f["beta"]) == 10.0
            w = f["omega"]
            rho = -1j * f["G_lesser"].sum(axis=0) * (w[1] - w[0]) / (2 * np.pi)
            currents(
                phi,
                label,
                w,
                -1j * f["G_lesser"] / (2 * np.pi),
                1j * f["G_greater"] / (2 * np.pi),
                bath_noise(w, phi),
                meta["converged"],
                gamma=rho,
            )
        for rank in ("S13", "S135"):
            name = f"CC {rank}"
            folder = base / f"CC_N4_{rank}"
            if (
                not (folder / "branches.npz").exists()
                or not (folder / "result.json").exists()
            ):
                continue
            meta = json.loads((folder / "result.json").read_text())
            branch = read(folder / "branches.npz")
            ar = -(branch["R_rem"] - adj(branch["R_rem"])) / (2j * np.pi)
            aa = -(branch["R_add"] - adj(branch["R_add"])) / (2j * np.pi)
            state = read(base / "cc_N4/state.npz")
            obs = json.loads((base / "cc_N4/observables.json").read_text())
            cur = obs["currents"]
            currents(
                phi,
                name,
                branch["omega"],
                ar,
                aa,
                bath_noise(branch["omega"], phi, pm=True),
                meta["physical_converged"],
                {"t": cur["I_tip"], "s": cur["I_substrate"]},
                state["gamma"][::2, ::2][:4, :4],
            )
    write_json(out / "current_checks.json", diagnostics)


def dump_atomic():
    """Tip-contact removal sticks from the neutral singlet and anionic doublet."""
    data, ann, h, s2 = molecule()
    energy, vectors, number, spin2 = diagonalize(h, s2)
    contact = np.asarray(data["leads"]["tip_geometries"]["center"])
    tip = [sum(contact[i] * ann[2 * i + spin] for i in range(4)) for spin in (0, 1)]
    rows = []
    # N=4 singlet and N=5 doublet; average equally over the initial multiplet.
    for n, spin in ((4, 0.0), (5, 0.75)):
        candidates = np.flatnonzero((number == n) & np.isclose(spin2, spin, atol=1e-8))
        ground = candidates[
            np.isclose(energy[candidates], energy[candidates].min(), atol=1e-8, rtol=0)
        ]
        assert len(ground) == (1 if n == 4 else 2)
        final = np.flatnonzero(number == n - 1)
        weights = sum(
            np.sum(
                abs(vectors[:, final].conj().T @ b @ vectors[:, ground]) ** 2, axis=1
            )
            for b in tip
        ) / len(ground)
        expected = sum(np.sum(abs(b @ vectors[:, ground]) ** 2) for b in tip) / len(
            ground
        )
        np.testing.assert_allclose(weights.sum(), expected, rtol=1e-12, atol=1e-12)
        groups = {}
        for f, weight in zip(final, weights):
            key = (
                round(float(energy[ground].mean() - energy[f]), 9),
                round(float(spin2[f]), 8),
            )
            groups[key] = groups.get(key, 0.0) + float(weight)
        for (omega, final_spin2), weight in sorted(groups.items()):
            if weight > 1e-14:
                rows.append((n, omega, final_spin2, weight))
    rows = np.asarray(rows)
    visible = np.ones(len(rows), dtype=bool)
    for family in (
        (rows[:, 0] == 4),
        ((rows[:, 0] == 5) & np.isclose(rows[:, 2], 2.0)),
    ):
        if family.any():
            visible[family] = rows[family, 3] >= 0.1 * rows[family, 3].max()
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    save_npz(
        ROOT / "results/atomic_sticks.npz",
        initial_N=rows[:, 0].astype(int),
        omega=rows[:, 1],
        final_S2=rows[:, 2],
        weight=rows[:, 3],
        paper_visible=visible,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dump Fig. 4 current densities or isolated-molecule sticks"
    )
    parser.add_argument("quantity", choices=("currents", "atomic"))
    args = parser.parse_args()
    if args.quantity == "currents":
        dump_currents()
    else:
        dump_atomic()
