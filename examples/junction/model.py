"""Molecular Hamiltonian and two fitted reservoirs, Supplement S4.

Spatial order: (h1, h2, l1, l2), substrate bath, tip bath.
The adjacent parameters.json stores h_ij, Coulomb coefficients, and contacts.
The draft orders its first two orbitals as (h2, h1); all stored arrays use
the order stated above to preserve the published checkpoint basis.
Each bath has filled (lesser) and empty (greater) pseudomodes.
"""

from __future__ import annotations

from ..parameters import parameters

import hashlib
import json
from pathlib import Path

import numpy as np

from c3l.interactions import SparseTwoBodyInteraction
from c3l.models import FermionicLindbladModel


ROOT = Path.cwd() / "junction"
BIASES = tuple(parameters("junction")["biases"])
RESPONSE_BIASES = tuple(parameters("junction")["response_biases"])


def case_directory(phi):
    if phi not in BIASES:
        raise ValueError(
            "Use equilibrium or a bias in the paper's -0.1 to -1.6 static sweep"
        )
    return ROOT / ("results" if phi == 0 else f"results/bias_production/phi_{phi:+.2f}")


def load_input() -> dict:
    """Read the paper Hamiltonian and contact vectors."""
    return parameters("junction")["model"]


def input_fingerprint():
    """Bind restarts to the actual Hamiltonian, temperature, and contact strengths."""
    p = parameters("junction")
    definition = {k: p[k] for k in
                  ("model", "beta", "substrate_coupling_squared", "tip_coupling_squared")}
    digest = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    # Preserve the archived paper checkpoints only for the same physical model.
    return p["archived_input_sha256"] if digest == p["published_model_sha256"] else digest


def _contacts(geometry):
    data = load_input()
    if geometry not in data["leads"]["tip_geometries"]:
        raise ValueError(f"unknown tip geometry: {geometry}")
    return (
        np.asarray(data["leads"]["u_substrate"], dtype=complex),
        np.asarray(data["leads"]["tip_geometries"][geometry], dtype=complex),
    )


def _bath(path):
    branches = []
    with np.load(path, allow_pickle=False) as saved:
        for branch in ("lesser", "greater"):
            h = np.asarray(saved[f"H_{branch}"], dtype=complex)
            d = np.asarray(saved[f"damping_{branch}"], dtype=complex)
            g = np.asarray(saved[f"g_{branch}"], dtype=complex)
            if (
                g.ndim != 1
                or not g.size
                or h.shape != (g.size, g.size)
                or d.shape != h.shape
            ):
                raise ValueError(f"invalid {branch} bath dimensions")
            if not all(np.all(np.isfinite(a)) for a in (h, d, g)):
                raise ValueError(f"nonfinite {branch} bath")
            if not np.allclose(h, h.conj().T, atol=1e-12, rtol=1e-12):
                raise ValueError(f"{branch} Hamiltonian must be Hermitian")
            damping_tolerance = 1e-10 * max(1.0, float(np.linalg.norm(d, 2)))
            if (
                not np.allclose(d, d.conj().T, atol=1e-12, rtol=1e-12)
                or np.linalg.eigvalsh(d).min() < -damping_tolerance
            ):
                raise ValueError(
                    f"{branch} damping must be positive semidefinite Hermitian"
                )
            branches.append((h, d, g))
    return branches


def assemble_model(
    bath_path, phi, geometry="center", interaction_scale=1.0, kappa=None
):
    """Build H, Gamma_gain, and Gamma_loss in the spatial-orbital basis.

    Phi shifts the tip bath energies and its chemical potential together.
    The two reservoirs retain separate occupations and coupling vectors.
    """
    if kappa is None:
        p = parameters("junction")
        kappa = np.sqrt(p["tip_coupling_squared"] / p["substrate_coupling_squared"])
    phi, interaction_scale, kappa = map(float, (phi, interaction_scale, kappa))
    if (
        not np.all(np.isfinite([phi, interaction_scale, kappa]))
        or min(interaction_scale, kappa) < 0
    ):
        raise ValueError("phi must be finite and interaction_scale/kappa nonnegative")
    data = load_input()
    us, ut = _contacts(geometry)
    branches = _bath(bath_path)
    reservoir_size = sum(len(g) for _, _, g in branches)
    n = 4 + 2 * reservoir_size
    h = np.zeros((n, n), dtype=complex)
    loss = np.zeros_like(h)
    gain = np.zeros_like(h)
    h[:4, :4] = np.asarray(data["one_body_matrix"], dtype=complex)
    # Append substrate and tip baths after the four interacting orbitals.
    offset = 4
    for u, amplitude, shift in (
        (us, np.sqrt(parameters("junction")["substrate_coupling_squared"]), 0.0),
        (ut, np.sqrt(parameters("junction")["substrate_coupling_squared"]) * kappa, phi),
    ):
        for branch, (hb, damping, g) in enumerate(branches):
            block = slice(offset, offset + len(g))
            h[block, block] = hb + shift * np.eye(len(g))
            h[:4, block] = amplitude * np.outer(u, g.conj())
            h[block, :4] = h[:4, block].conj().T
            (gain if branch == 0 else loss)[block, block] = damping
            offset += len(g)
    interaction = SparseTwoBodyInteraction(
        2 * n,
        tuple(
            (int(p), int(q), int(r), int(s), interaction_scale * complex(c))
            for p, q, r, s, c in data["canonical_two_body_terms"]
        ),
    )
    return FermionicLindbladModel(
        h,
        loss,
        gain,
        U=0.0,
        interaction_sites=(0, 1, 2, 3),
        density_interaction=None,
        two_body_interaction=interaction,
        units="E0",
    )


def contact_currents(model, gamma):
    """Return independent impurity injection and dissipative reservoir fluxes.

    Positive current flows from the named reservoir into the impurity.
    Reservoir boundaries follow this module's assembly order.
    """
    n = model.n_sites
    if n <= 4 or (n - 4) % 2:
        raise ValueError("model must contain two equally sized reservoir blocks")
    gamma = np.asarray(gamma, dtype=complex)
    if gamma.shape != (2 * n, 2 * n) or not np.all(np.isfinite(gamma)):
        raise ValueError("gamma must use the complete finite spin-orbital space")
    h = np.kron(model.E, np.eye(2))
    loss = np.kron(model.gamma_loss, np.eye(2))
    gain = np.kron(model.gamma_gain, np.eye(2))
    width = n - 4  # Spin-orbital size of either reservoir.
    result = {}
    for label, start in (("substrate", 8), ("tip", 8 + width)):
        q = np.arange(start, start + width)
        result[f"I_{label}"] = float(2 * np.sum(h[:8, q] * gamma[q, :8].T).imag)
        flux = np.trace(gain[np.ix_(q, q)]) - np.sum(
            (loss + gain)[q, :] * gamma[:, q].T
        )
        result[f"F_{label}"] = float(2 * flux.real)
    return result
