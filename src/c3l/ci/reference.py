"""Spin- or charge-conserving Hartree-Fock normal-mode references.

Each spin block diagonalizes a quadratic 2n by 2n matrix. The full HF
constructor includes all Wick contractions of the general interaction;
the half-filled constructor supplies the fixed reference used by the SIAM
paper calculations. Spin mode orders all up modes, then down. Charge mode
uses one full physical-orbital block, allowing spin-orbit coupling.
No doubled many-body matrices are constructed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from c3l.conventions import TolerancePolicy
from c3l.gaussian_mean_field import wick_fock
from c3l.models import FermionicLindbladModel, QuadraticOpenSystem
from c3l.models import spin_lift as expand_spin_diagonal
from c3l.quadratic_ness import finite_pm_covariance
from c3l.superfermion.generator import quadratic_mode_matrix
from c3l.superfermion.normal_modes import NormalModeReference, normal_mode_reference

ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]


def _immutable_complex(array: ArrayLike) -> NDArray[np.complex128]:
    contiguous = np.ascontiguousarray(array, dtype=np.complex128)
    immutable = np.frombuffer(contiguous.tobytes(), dtype=np.complex128).reshape(
        contiguous.shape
    )
    immutable.flags.writeable = False
    return immutable


def _hartree_shifts(
    model: FermionicLindbladModel, m_up: float, m_down: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the spin-dependent diagonal Hartree contractions."""

    up = np.zeros(model.n_sites, dtype=np.float64)
    down = np.zeros(model.n_sites, dtype=np.float64)
    if model.density_interaction is None:
        for site in model.interaction_sites:
            up[site] = model.U * m_down
            down[site] = model.U * m_up
        return up, down
    U = model.interaction_matrix
    for a, site in enumerate(model.interaction_sites):
        offsite = float(np.sum(U[a]) - U[a, a])
        up[site] = U[a, a] * m_down + offsite * (m_up + m_down)
        down[site] = U[a, a] * m_up + offsite * (m_up + m_down)
    return up, down


@dataclass(frozen=True, slots=True)
class HartreeFockReference:
    """Quadratic reference and optional full HF density/Fock matrices.

    Each spin has 2n particle-hole modes; the global order is up then down.
    For an onsite half-filled reference only, the diagonal shifts reduce to
    U*m_down for up spin and U*m_up for down spin. General multiorbital
    references use the complete Fock contraction, including exchange.
    In charge mode, ``up`` and ``E_up`` hold the full physical reference;
    ``down`` and ``E_down`` are absent.
    """

    model: FermionicLindbladModel
    m_up: np.float64
    m_down: np.float64
    E_up: ComplexMatrix
    E_down: ComplexMatrix | None
    up: NormalModeReference
    down: NormalModeReference | None
    tolerance_policy: TolerancePolicy
    gamma: ComplexMatrix | None = None
    fock: ComplexMatrix | None = None
    reference_gamma_loss: ComplexMatrix | None = None
    reference_gamma_gain: ComplexMatrix | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, FermionicLindbladModel):
            raise TypeError("model must be a FermionicLindbladModel")
        if not isinstance(self.tolerance_policy, TolerancePolicy):
            raise TypeError("tolerance_policy must be a TolerancePolicy")
        if self.model.input_tolerance != self.tolerance_policy.input_abs_rel:
            raise ValueError("model input tolerance must match the reference policy")
        charge = self.model.symmetry == "charge"
        if charge:
            if self.down is not None or self.E_down is not None:
                raise ValueError("charge references must have only one active block")
            if self.gamma is None or self.fock is None:
                raise ValueError("charge references require full gamma and fock")
        elif self.down is None or self.E_down is None:
            raise ValueError("spin references require both spin blocks")
        if any(mode.policy != self.tolerance_policy for mode in self.mode_references):
            raise ValueError("spin normal modes must use the Hartree--Fock policy")
        m_up = np.float64(self.m_up)
        m_down = np.float64(self.m_down)
        general = self.gamma is not None or self.fock is not None
        custom_dissipation = (
            self.reference_gamma_loss is not None
            or self.reference_gamma_gain is not None
        )
        if general:
            if self.gamma is None or self.fock is None:
                raise ValueError("general HF references require both gamma and fock")
            if (
                not np.isfinite(m_up)
                or not np.isfinite(m_down)
                or not 0.0 <= m_up <= 1.0
                or not 0.0 <= m_down <= 1.0
            ):
                raise ValueError("general HF occupations must lie in [0,1]")
        elif (
            not np.isfinite(m_up)
            or not np.isfinite(m_down)
            or not 0.0 <= m_up <= 0.5
            or m_down != np.float64(1.0 - m_up)
        ):
            raise ValueError("Hartree--Fock occupations must obey half filling")
        if custom_dissipation and not general:
            raise ValueError("reference-only dissipation requires a general HF reference")
        if custom_dissipation and (
            self.reference_gamma_loss is None or self.reference_gamma_gain is None
        ):
            raise ValueError("reference-only loss and gain must be supplied together")
        if np.asarray(self.E_up).shape != self.model.E.shape:
            raise ValueError("E_up shape does not match the auxiliary model")
        if not charge and np.asarray(self.E_down).shape != self.model.E.shape:
            raise ValueError("E_down shape does not match the auxiliary model")
        if any(not np.all(np.isfinite(block)) for block in self.one_body_blocks):
            raise ValueError("Hartree--Fock one-body matrices must be finite")
        if not general and self.model.density_interaction is None:
            half_filled_scale = max(
                float(np.linalg.norm(self.model.E, ord="fro")),
                abs(float(self.model.U)),
                1.0,
            )
            half_filled_threshold = self.tolerance_policy.input_abs_rel.threshold(
                half_filled_scale
            )
            for site in self.model.interaction_sites:
                entry = complex(self.model.E[site, site])
                if abs(entry + 0.5 * self.model.U) > half_filled_threshold:
                    raise ValueError("Hartree--Fock reference model is not half filled")
        if self.up.n_orbitals != self.model.n_sites:
            raise ValueError("up-spin normal modes do not match the site count")
        if not charge and self.down.n_orbitals != self.model.n_sites:
            raise ValueError("down-spin normal modes do not match the site count")
        if general:
            gamma = np.asarray(self.gamma, dtype=np.complex128)
            fock = np.asarray(self.fock, dtype=np.complex128)
            expected_shape = (self.model.n_spin_orbitals,) * 2
            if gamma.shape != expected_shape or fock.shape != expected_shape:
                raise ValueError("general gamma and fock must use the spin-orbital space")
            if not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(fock)):
                raise ValueError("general gamma and fock must be finite")
            scale = max(
                float(np.linalg.norm(gamma, ord="fro")),
                float(np.linalg.norm(fock, ord="fro")),
                1.0,
            )
            threshold = self.tolerance_policy.algebra_abs_rel.threshold(scale)
            if np.linalg.norm(gamma - gamma.conj().T, ord="fro") > threshold:
                raise ValueError("general gamma must be Hermitian")
            if np.linalg.norm(fock - fock.conj().T, ord="fro") > threshold:
                raise ValueError("general fock must be Hermitian")
            reconstructed_fock = wick_fock(
                self.model.physical_E,
                self.model.interaction,
                gamma,
            )
            fock_scale = max(
                float(np.linalg.norm(reconstructed_fock, ord="fro")),
                float(np.linalg.norm(fock, ord="fro")),
                1.0,
            )
            if np.linalg.norm(fock - reconstructed_fock, ord="fro") > (
                self.tolerance_policy.algebra_abs_rel.threshold(fock_scale)
            ):
                raise ValueError("general Fock reconstruction disagrees with h plus vbar gamma")
            reference_loss = np.asarray(
                self.model.gamma_loss
                if self.reference_gamma_loss is None
                else self.reference_gamma_loss,
                dtype=np.complex128,
            )
            reference_gain = np.asarray(
                self.model.gamma_gain
                if self.reference_gamma_gain is None
                else self.reference_gamma_gain,
                dtype=np.complex128,
            )
            if (
                reference_loss.shape != self.model.E.shape
                or reference_gain.shape != self.model.E.shape
                or not np.all(np.isfinite(reference_loss))
                or not np.all(np.isfinite(reference_gain))
            ):
                raise ValueError("reference-only loss and gain must match the spatial model")
            for name, matrix in (
                ("reference_gamma_loss", reference_loss),
                ("reference_gamma_gain", reference_gain),
            ):
                if np.linalg.norm(matrix - matrix.conj().T, ord="fro") > threshold:
                    raise ValueError(f"{name} must be Hermitian")
                if np.linalg.eigvalsh(matrix)[0] < -threshold:
                    raise ValueError(f"{name} must be positive semidefinite")
            if not charge:
                spin_mixing = max(
                    float(np.linalg.norm(gamma[0::2, 1::2], ord="fro")),
                    float(np.linalg.norm(gamma[1::2, 0::2], ord="fro")),
                    float(np.linalg.norm(fock[0::2, 1::2], ord="fro")),
                    float(np.linalg.norm(fock[1::2, 0::2], ord="fro")),
                )
                if spin_mixing > threshold:
                    raise ValueError(
                        "the current component coordinates require a spin-conserving HF reference"
                    )
            eigenvalues = np.linalg.eigvalsh(gamma)
            if eigenvalues[0] < -threshold or eigenvalues[-1] > 1.0 + threshold:
                raise ValueError("general gamma has occupations outside [0,1]")
            spin_restriction = np.inf if charge else max(
                float(np.linalg.norm(gamma[0::2, 0::2] - gamma[1::2, 1::2])),
                float(np.linalg.norm(fock[0::2, 0::2] - fock[1::2, 1::2])),
            )
            if charge:
                target, lyapunov = finite_pm_covariance(
                    fock, reference_loss, reference_gain
                )
            elif spin_restriction <= threshold:
                spatial_target, lyapunov = finite_pm_covariance(
                    fock[0::2, 0::2],
                    reference_loss,
                    reference_gain,
                )
                target = expand_spin_diagonal(spatial_target)
            else:
                target, lyapunov = finite_pm_covariance(
                    fock,
                    expand_spin_diagonal(reference_loss),
                    expand_spin_diagonal(reference_gain),
                )
            covariance_defect = np.linalg.norm(target - gamma, ord="fro") / max(
                np.linalg.norm(target, ord="fro"), 1.0
            )
            covariance_threshold = self.tolerance_policy.nonlinear_abs_rel.threshold(
                max(np.linalg.norm(target, ord="fro"), np.linalg.norm(gamma, ord="fro"), 1.0)
            )
            if lyapunov > threshold or covariance_defect > covariance_threshold:
                raise ValueError("general gamma is not the covariance of its fock matrix")
            expected_up = fock if charge else np.asarray(fock[0::2, 0::2])
            expected_down = None if charge else np.asarray(fock[1::2, 1::2])
        else:
            gamma = None
            fock = None
            reference_loss = np.asarray(self.model.gamma_loss)
            reference_gain = np.asarray(self.model.gamma_gain)
            expected_up = np.array(self.model.E, dtype=np.complex128, copy=True)
            expected_down = np.array(self.model.E, dtype=np.complex128, copy=True)
            shift_up, shift_down = _hartree_shifts(self.model, m_up, m_down)
            expected_up[np.diag_indices(self.model.n_sites)] += shift_up
            expected_down[np.diag_indices(self.model.n_sites)] += shift_down
        scale = max(
            float(np.linalg.norm(expected_up, ord="fro")),
            0.0 if charge else float(np.linalg.norm(expected_down, ord="fro")),
            1.0,
        )
        threshold = self.tolerance_policy.algebra_abs_rel.threshold(scale)
        if np.linalg.norm(np.asarray(self.E_up) - expected_up, ord="fro") > threshold:
            raise ValueError("E_up disagrees with the Hartree shift")
        if not charge and (
            np.linalg.norm(np.asarray(self.E_down) - expected_down, ord="fro")
            > threshold
        ):
            raise ValueError("E_down disagrees with the Hartree shift")
        expected_blocks = (expected_up,) if charge else (expected_up, expected_down)
        for label, spin_reference, E_sigma in zip(
            ("up",) if charge else ("up", "down"),
            self.mode_references,
            expected_blocks,
            strict=True,
        ):
            expected_h = quadratic_mode_matrix(
                E_sigma, reference_loss, reference_gain
            )
            h_scale = max(float(np.linalg.norm(expected_h, ord="fro")), 1.0)
            if np.linalg.norm(spin_reference.h - expected_h, ord="fro") > (
                self.tolerance_policy.algebra_abs_rel.threshold(h_scale)
            ):
                raise ValueError(f"{label}-spin normal modes disagree with E_sigma")
        object.__setattr__(self, "m_up", m_up)
        object.__setattr__(self, "m_down", m_down)
        object.__setattr__(self, "E_up", _immutable_complex(self.E_up))
        object.__setattr__(
            self, "E_down", None if charge else _immutable_complex(self.E_down)
        )
        object.__setattr__(
            self,
            "gamma",
            None if gamma is None else _immutable_complex(gamma),
        )
        object.__setattr__(
            self,
            "fock",
            None if fock is None else _immutable_complex(fock),
        )
        object.__setattr__(
            self,
            "reference_gamma_loss",
            None if not custom_dissipation else _immutable_complex(reference_loss),
        )
        object.__setattr__(
            self,
            "reference_gamma_gain",
            None if not custom_dissipation else _immutable_complex(reference_gain),
        )

    @property
    def n_sites(self) -> int:
        return self.model.n_sites

    @property
    def num_ph_modes(self) -> int:
        return 2 * self.model.n_spin_orbitals

    @property
    def mode_references(self) -> tuple[NormalModeReference, ...]:
        """Active normal-mode blocks in global particle-hole order."""

        return (self.up,) if self.model.symmetry == "charge" else (self.up, self.down)

    @property
    def one_body_blocks(self) -> tuple[ComplexMatrix, ...]:
        """Active reference one-body blocks in the same order as the modes."""

        return (self.E_up,) if self.model.symmetry == "charge" else (self.E_up, self.E_down)


def half_filled_hartree_fock_reference(
    model: FermionicLindbladModel,
    m_up: float | np.floating[Any],
    policy: TolerancePolicy | None = None,
) -> HartreeFockReference:
    """Construct the paper's half-filled, possibly spin-broken reference.

    This deliberately narrow constructor requires the benchmark convention
    ``E[f,f] = -U/2`` at every interacting site.  ``m_up`` is one common
    reference-orbital parameter in ``[0, 0.5]`` and ``m_down = 1-m_up``; it is
    not a physical magnetization or a site-dependent Hartree field.
    """

    if not isinstance(model, FermionicLindbladModel):
        raise TypeError("model must be a FermionicLindbladModel")
    if model.symmetry == "charge":
        raise ValueError("charge symmetry requires full_hartree_fock_reference")
    tolerance_policy = TolerancePolicy.first() if policy is None else policy
    if not isinstance(tolerance_policy, TolerancePolicy):
        raise TypeError("policy must be a TolerancePolicy")
    if model.input_tolerance != tolerance_policy.input_abs_rel:
        raise ValueError("model input tolerance must match the reference policy")
    if isinstance(m_up, (bool, np.bool_)):
        raise TypeError("m_up must be a real scalar")
    raw_m = np.asarray(m_up)
    if raw_m.ndim != 0 or (np.iscomplexobj(raw_m) and np.imag(raw_m) != 0.0):
        raise ValueError("m_up must be a real scalar")
    m_up64 = np.float64(np.real(raw_m))
    if not np.isfinite(m_up64) or not 0.0 <= m_up64 <= 0.5:
        raise ValueError("m_up must lie in [0, 0.5]")
    m_down64 = np.float64(1.0 - m_up64)

    if model.density_interaction is None:
        expected_entry = np.float64(-0.5 * model.U)
        input_scale = max(
            float(np.linalg.norm(model.E, ord="fro")), abs(float(model.U)), 1.0
        )
        threshold = tolerance_policy.input_abs_rel.threshold(input_scale)
        for site in model.interaction_sites:
            if abs(complex(model.E[site, site]) - expected_entry) > threshold:
                raise ValueError(
                    "half-filled Hartree--Fock reference requires E[f,f] = -U/2"
                )

    E_up = np.array(model.E, dtype=np.complex128, copy=True)
    E_down = np.array(model.E, dtype=np.complex128, copy=True)
    shift_up, shift_down = _hartree_shifts(model, m_up64, m_down64)
    E_up[np.diag_indices(model.n_sites)] += shift_up
    E_down[np.diag_indices(model.n_sites)] += shift_down
    up_model = QuadraticOpenSystem(
        E_up,
        model.gamma_loss,
        model.gamma_gain,
        units=model.units,
        input_tolerance=tolerance_policy.input_abs_rel,
    )
    down_model = QuadraticOpenSystem(
        E_down,
        model.gamma_loss,
        model.gamma_gain,
        units=model.units,
        input_tolerance=tolerance_policy.input_abs_rel,
    )
    up = normal_mode_reference(up_model, tolerance_policy)
    down = normal_mode_reference(down_model, tolerance_policy)
    return HartreeFockReference(
        model=model,
        m_up=m_up64,
        m_down=m_down64,
        E_up=E_up,
        E_down=E_down,
        up=up,
        down=down,
        tolerance_policy=tolerance_policy,
    )


def full_hartree_fock_reference(
    model: FermionicLindbladModel,
    fock: ArrayLike,
    gamma: ArrayLike,
    policy: TolerancePolicy | None = None,
    *,
    reference_gamma_loss: ArrayLike | None = None,
    reference_gamma_gain: ArrayLike | None = None,
) -> HartreeFockReference:
    """Build active PH mode blocks from one solved full-HF fixed point."""

    if not isinstance(model, FermionicLindbladModel):
        raise TypeError("model must be a FermionicLindbladModel")
    tolerance_policy = TolerancePolicy.first() if policy is None else policy
    if not isinstance(tolerance_policy, TolerancePolicy):
        raise TypeError("policy must be a TolerancePolicy")
    if model.input_tolerance != tolerance_policy.input_abs_rel:
        raise ValueError("model input tolerance must match the reference policy")
    F = np.asarray(fock, dtype=np.complex128)
    density = np.asarray(gamma, dtype=np.complex128)
    shape = (model.n_spin_orbitals, model.n_spin_orbitals)
    if F.shape != shape or density.shape != shape:
        raise ValueError("fock and gamma must use the complete spin-orbital space")
    charge = model.symmetry == "charge"
    E_up = F if charge else np.asarray(F[0::2, 0::2])
    E_down = None if charge else np.asarray(F[1::2, 1::2])
    if (reference_gamma_loss is None) != (reference_gamma_gain is None):
        raise ValueError("reference-only loss and gain must be supplied together")
    loss = (
        np.asarray(model.gamma_loss)
        if reference_gamma_loss is None
        else np.asarray(reference_gamma_loss, dtype=np.complex128)
    )
    gain = (
        np.asarray(model.gamma_gain)
        if reference_gamma_gain is None
        else np.asarray(reference_gamma_gain, dtype=np.complex128)
    )
    up = normal_mode_reference(
        QuadraticOpenSystem(
            E_up,
            loss,
            gain,
            units=model.units,
            input_tolerance=tolerance_policy.input_abs_rel,
        ),
        tolerance_policy,
    )
    down = None if charge else normal_mode_reference(
        QuadraticOpenSystem(
            E_down,
            loss,
            gain,
            units=model.units,
            input_tolerance=tolerance_policy.input_abs_rel,
        ),
        tolerance_policy,
    )
    up_indices = np.asarray(
        [model.orbital_index(site, 0) for site in model.interaction_sites],
        dtype=np.int64,
    )
    down_indices = up_indices if charge else up_indices + 1
    m_up = np.float64(np.mean(np.diag(density)[up_indices]).real)
    m_down = np.float64(np.mean(np.diag(density)[down_indices]).real)
    return HartreeFockReference(
        model=model,
        m_up=m_up,
        m_down=m_down,
        E_up=E_up,
        E_down=E_down,
        up=up,
        down=down,
        tolerance_policy=tolerance_policy,
        gamma=density,
        fock=F,
        reference_gamma_loss=(None if reference_gamma_loss is None else loss),
        reference_gamma_gain=(None if reference_gamma_gain is None else gain),
    )
