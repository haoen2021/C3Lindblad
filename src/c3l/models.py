"""Physical one-body matrices, Lindblad gain/loss, and general two-body interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import AbsRelTolerance, TolerancePolicy
from .interactions import SparseTwoBodyInteraction, density_density_interaction

ComplexMatrix = NDArray[np.complex128]


def _immutable_complex_matrix(matrix: ArrayLike) -> ComplexMatrix:
    contiguous = np.ascontiguousarray(matrix, dtype=np.complex128)
    immutable = np.frombuffer(contiguous.tobytes(), dtype=np.complex128).reshape(
        contiguous.shape
    )
    immutable.flags.writeable = False
    return immutable


def _immutable_real_matrix(matrix: ArrayLike) -> NDArray[np.float64]:
    contiguous = np.ascontiguousarray(matrix, dtype=np.float64)
    immutable = np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(
        contiguous.shape
    )
    immutable.flags.writeable = False
    return immutable


@dataclass(frozen=True, slots=True)
class InputCorrection:
    """Recorded roundoff-level input canonicalization."""

    field: str
    kind: str
    original_defect: np.float64
    tolerance: np.float64
    correction_norm: np.float64

    def to_dict(self) -> dict[str, str | float]:
        return {
            "field": self.field,
            "kind": self.kind,
            "original_defect": float(self.original_defect),
            "tolerance": float(self.tolerance),
            "correction_norm": float(self.correction_norm),
        }


def _canonical_matrix(
    name: str,
    value: ArrayLike,
    tolerance: AbsRelTolerance,
    *,
    positive_semidefinite: bool,
) -> tuple[ComplexMatrix, tuple[InputCorrection, ...]]:
    matrix = np.array(value, dtype=np.complex128, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if matrix.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")

    corrections: list[InputCorrection] = []
    scale = np.linalg.norm(matrix, ord="fro")
    threshold = tolerance.threshold(scale)
    defect = np.linalg.norm(matrix - matrix.conj().T, ord="fro")
    if defect > threshold:
        raise ValueError(
            f"{name} Hermiticity defect {defect:.3e} exceeds {threshold:.3e}"
        )
    if defect > 0.0:
        canonical = (matrix + matrix.conj().T) / np.complex128(2.0)
        corrections.append(
            InputCorrection(
                field=name,
                kind="Hermitian_symmetrization",
                original_defect=np.float64(defect),
                tolerance=threshold,
                correction_norm=np.float64(
                    np.linalg.norm(canonical - matrix, ord="fro")
                ),
            )
        )
        matrix = canonical

    if positive_semidefinite:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        psd_threshold = tolerance.threshold(np.linalg.norm(matrix, ord=2))
        minimum = np.float64(eigenvalues[0])
        if minimum < -psd_threshold:
            raise ValueError(
                f"{name} minimum eigenvalue {minimum:.3e} is below {-psd_threshold:.3e}"
            )
        if minimum < 0.0:
            clipped = np.maximum(eigenvalues, 0.0)
            canonical = (eigenvectors * clipped) @ eigenvectors.conj().T
            canonical = (canonical + canonical.conj().T) / np.complex128(2.0)
            corrections.append(
                InputCorrection(
                    field=name,
                    kind="roundoff_PSD_eigenvalue_clipping",
                    original_defect=np.float64(-minimum),
                    tolerance=psd_threshold,
                    correction_norm=np.float64(
                        np.linalg.norm(canonical - matrix, ord="fro")
                    ),
                )
            )
            matrix = canonical

    matrix = _immutable_complex_matrix(matrix)
    return matrix, tuple(corrections)


def _validated_matrices(
    E: ArrayLike,
    gamma_loss: ArrayLike,
    gamma_gain: ArrayLike,
    tolerance: AbsRelTolerance,
) -> tuple[ComplexMatrix, ComplexMatrix, ComplexMatrix, tuple[InputCorrection, ...]]:
    E_canonical, E_corrections = _canonical_matrix(
        "E", E, tolerance, positive_semidefinite=False
    )
    loss_canonical, loss_corrections = _canonical_matrix(
        "gamma_loss", gamma_loss, tolerance, positive_semidefinite=True
    )
    gain_canonical, gain_corrections = _canonical_matrix(
        "gamma_gain", gamma_gain, tolerance, positive_semidefinite=True
    )
    if (
        E_canonical.shape != loss_canonical.shape
        or E_canonical.shape != gain_canonical.shape
    ):
        raise ValueError("E, gamma_loss, and gamma_gain must have identical shapes")
    corrections = E_corrections + loss_corrections + gain_corrections
    return E_canonical, loss_canonical, gain_canonical, corrections


def _matrix_to_dict(matrix: ComplexMatrix) -> dict[str, list[list[float]]]:
    return {"real": matrix.real.tolist(), "imag": matrix.imag.tolist()}


def _matrix_from_dict(data: Mapping[str, Any]) -> ComplexMatrix:
    return np.asarray(data["real"], dtype=np.float64) + 1j * np.asarray(
        data["imag"], dtype=np.float64
    )


@dataclass(frozen=True, slots=True)
class QuadraticOpenSystem:
    """Validated quadratic model on an explicitly ordered physical-orbital space."""

    E: ArrayLike
    gamma_loss: ArrayLike
    gamma_gain: ArrayLike
    units: str = "arbitrary"
    input_tolerance: AbsRelTolerance = field(
        default_factory=lambda: TolerancePolicy.first().input_abs_rel
    )
    corrections: tuple[InputCorrection, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.input_tolerance, AbsRelTolerance):
            raise TypeError("input_tolerance must be an AbsRelTolerance")
        if not isinstance(self.units, str) or not self.units:
            raise ValueError("units must be a nonempty string")
        E, gamma_loss, gamma_gain, corrections = _validated_matrices(
            self.E, self.gamma_loss, self.gamma_gain, self.input_tolerance
        )
        object.__setattr__(self, "E", E)
        object.__setattr__(self, "gamma_loss", gamma_loss)
        object.__setattr__(self, "gamma_gain", gamma_gain)
        object.__setattr__(self, "corrections", corrections)

    @property
    def n_orbitals(self) -> int:
        return int(self.E.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "QuadraticOpenSystem",
            "E": _matrix_to_dict(self.E),
            "gamma_loss": _matrix_to_dict(self.gamma_loss),
            "gamma_gain": _matrix_to_dict(self.gamma_gain),
            "units": self.units,
            "input_tolerance": self.input_tolerance.to_dict(),
            "corrections": [correction.to_dict() for correction in self.corrections],
        }


@dataclass(frozen=True, slots=True)
class FermionicLindbladModel:
    """Number-conserving fermionic Lindblad model with selectable symmetry.

    In spin mode the physical orbital map is ``p = 2 * site + spin``; ``E``
    and both Gamma matrices are replicated without spin mixing.
    Without ``density_interaction``, the common onsite interaction ``U`` acts
    between up/down occupations at every site in ``interaction_sites``.  When
    supplied, ``density_interaction[a,b]`` is the complete spin-summed density
    matrix in that site order and ``U`` must be zero.  A
    ``two_body_interaction`` instead supplies the canonical sparse
    antisymmetrized spin-orbital interaction and is mutually exclusive with
    both legacy forms.  ``impurity_site`` remains the default site used by
    legacy observables and response sources. In ``symmetry="charge"``, matrices
    and interaction sites use physical orbital indices, with no spin lifting;
    interactions must use ``two_body_interaction`` and ``U=0``.
    """

    E: ArrayLike
    gamma_loss: ArrayLike
    gamma_gain: ArrayLike
    U: float | np.floating[Any]
    impurity_site: int = 0
    units: str = "Gamma"
    input_tolerance: AbsRelTolerance = field(
        default_factory=lambda: TolerancePolicy.first().input_abs_rel
    )
    interaction_sites: tuple[int, ...] | None = None
    density_interaction: ArrayLike | None = None
    two_body_interaction: SparseTwoBodyInteraction | None = None
    symmetry: str = "spin"
    corrections: tuple[InputCorrection, ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.symmetry not in ("spin", "charge"):
            raise ValueError("symmetry must be 'spin' or 'charge'")
        if not isinstance(self.input_tolerance, AbsRelTolerance):
            raise TypeError("input_tolerance must be an AbsRelTolerance")
        if not isinstance(self.units, str) or not self.units:
            raise ValueError("units must be a nonempty string")
        E, gamma_loss, gamma_gain, corrections = _validated_matrices(
            self.E, self.gamma_loss, self.gamma_gain, self.input_tolerance
        )
        raw_U = np.asarray(self.U)
        if raw_U.ndim != 0 or (np.iscomplexobj(raw_U) and np.imag(raw_U) != 0.0):
            raise ValueError("U must be a real scalar")
        U = np.float64(np.real(raw_U))
        if not np.isfinite(U):
            raise ValueError("U must be finite")
        if self.symmetry == "charge" and (
            U != 0.0 or self.density_interaction is not None
        ):
            raise ValueError(
                "charge symmetry requires U=0 and a general two_body_interaction"
            )
        if isinstance(self.impurity_site, bool) or not isinstance(
            self.impurity_site, int
        ):
            raise TypeError("impurity_site must be an integer")
        if not 0 <= self.impurity_site < E.shape[0]:
            raise ValueError("impurity_site is outside the spatial-site range")
        if self.interaction_sites is None:
            interaction_sites = (self.impurity_site,)
            if self.symmetry == "charge" and isinstance(
                self.two_body_interaction, SparseTwoBodyInteraction
            ):
                support = {
                    index
                    for term in self.two_body_interaction.terms
                    for index in term[:4]
                }
                interaction_sites += tuple(sorted(support - {self.impurity_site}))
        else:
            if not isinstance(self.interaction_sites, tuple):
                raise TypeError("interaction_sites must be a tuple of integers")
            interaction_sites = self.interaction_sites
        if not interaction_sites:
            raise ValueError("interaction_sites must contain at least one site")
        if any(isinstance(site, bool) or not isinstance(site, int) for site in interaction_sites):
            raise TypeError("interaction_sites must contain only integers")
        if any(not 0 <= site < E.shape[0] for site in interaction_sites):
            raise ValueError("an interaction site is outside the spatial-site range")
        if len(set(interaction_sites)) != len(interaction_sites):
            raise ValueError("interaction_sites must not contain duplicates")
        if interaction_sites[0] != self.impurity_site:
            raise ValueError("impurity_site must be the first interaction site")
        density_interaction = self.density_interaction
        two_body_interaction = self.two_body_interaction
        if two_body_interaction is not None:
            if not isinstance(two_body_interaction, SparseTwoBodyInteraction):
                raise TypeError(
                    "two_body_interaction must be a SparseTwoBodyInteraction"
                )
            if U != 0.0 or density_interaction is not None:
                raise ValueError(
                    "two_body_interaction is mutually exclusive with U and density_interaction"
                )
            num_physical = E.shape[0] if self.symmetry == "charge" else 2 * E.shape[0]
            if two_body_interaction.n_spin_orbitals != num_physical:
                raise ValueError(
                    "two_body_interaction does not match the model spin-orbital space"
                )
            allowed = set(interaction_sites) if self.symmetry == "charge" else {
                2 * site + spin
                for site in interaction_sites
                for spin in (0, 1)
            }
            if any(
                index not in allowed
                for term in two_body_interaction.terms
                for index in term[:4]
            ):
                raise ValueError(
                    "two_body_interaction must be confined to interaction_sites"
                )
        if density_interaction is not None:
            if U != 0.0:
                raise ValueError(
                    "U must be zero when density_interaction supplies the full matrix"
                )
            raw_density = np.asarray(density_interaction)
            shape = (len(interaction_sites), len(interaction_sites))
            if raw_density.shape != shape:
                raise ValueError(
                    "density_interaction must be indexed by interaction_sites"
                )
            if np.iscomplexobj(raw_density) and np.any(np.imag(raw_density) != 0.0):
                raise ValueError("density_interaction must be real")
            density = np.asarray(np.real(raw_density), dtype=np.float64)
            if not np.all(np.isfinite(density)):
                raise ValueError("density_interaction must be finite")
            scale = max(float(np.linalg.norm(density, ord="fro")), 1.0)
            threshold = self.input_tolerance.threshold(scale)
            defect = np.linalg.norm(density - density.T, ord="fro")
            if defect > threshold:
                raise ValueError("density_interaction must be symmetric")
            canonical_density = 0.5 * (density + density.T)
            if defect > 0.0:
                corrections += (
                    InputCorrection(
                        field="density_interaction",
                        kind="symmetric_canonicalization",
                        original_defect=np.float64(defect),
                        tolerance=np.float64(threshold),
                        correction_norm=np.float64(
                            np.linalg.norm(canonical_density - density, ord="fro")
                        ),
                    ),
                )
            density_interaction = _immutable_real_matrix(canonical_density)
        object.__setattr__(self, "E", E)
        object.__setattr__(self, "gamma_loss", gamma_loss)
        object.__setattr__(self, "gamma_gain", gamma_gain)
        object.__setattr__(self, "U", U)
        object.__setattr__(self, "interaction_sites", tuple(interaction_sites))
        object.__setattr__(self, "density_interaction", density_interaction)
        object.__setattr__(self, "two_body_interaction", two_body_interaction)
        object.__setattr__(self, "corrections", corrections)

    @property
    def n_sites(self) -> int:
        """Input block size: spatial sites in spin mode, physical modes in charge."""

        return int(self.E.shape[0])

    @property
    def n_spin_orbitals(self) -> int:
        return self.n_sites if self.symmetry == "charge" else 2 * self.n_sites

    @property
    def physical_E(self) -> ComplexMatrix:
        """One-body matrix in the complete physical orbital space."""

        return self.E if self.symmetry == "charge" else spin_lift(self.E)

    @property
    def physical_gamma_loss(self) -> ComplexMatrix:
        """Loss matrix in the complete physical orbital space."""

        return self.gamma_loss if self.symmetry == "charge" else spin_lift(self.gamma_loss)

    @property
    def physical_gamma_gain(self) -> ComplexMatrix:
        """Gain matrix in the complete physical orbital space."""

        return self.gamma_gain if self.symmetry == "charge" else spin_lift(self.gamma_gain)

    def orbital_index(self, site: int, spin: int) -> int:
        """Map a block index to a physical orbital; charge has only block zero."""

        spins = (0,) if self.symmetry == "charge" else (0, 1)
        if not 0 <= site < self.n_sites or spin not in spins:
            raise ValueError("site or spin index is outside the declared ordering")
        return site if self.symmetry == "charge" else 2 * site + spin

    def orbital_site_spin(self, orbital: int) -> tuple[int, int]:
        """Invert the physical orbital map without assuming spin pairs."""

        if not 0 <= orbital < self.n_spin_orbitals:
            raise ValueError("orbital index is outside the declared ordering")
        return (orbital, 0) if self.symmetry == "charge" else divmod(orbital, 2)

    @property
    def interaction_matrix(self) -> NDArray[np.float64]:
        """Density interaction in the declared ``interaction_sites`` order."""

        if self.density_interaction is not None:
            return self.density_interaction
        return _immutable_real_matrix(
            np.eye(len(self.interaction_sites), dtype=np.float64) * self.U
        )

    @property
    def interaction(self) -> SparseTwoBodyInteraction:
        """Canonical sparse spin-orbital form of the physical interaction."""

        if self.two_body_interaction is not None:
            return self.two_body_interaction
        if self.symmetry == "charge":
            return SparseTwoBodyInteraction(self.n_spin_orbitals, ())
        return density_density_interaction(
            self.n_sites,
            self.interaction_sites,
            self.interaction_matrix,
        )

    @property
    def has_interactions(self) -> bool:
        return bool(self.interaction.terms)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "model_type": "FermionicLindbladModel",
            "symmetry": self.symmetry,
            "E": _matrix_to_dict(self.E),
            "gamma_loss": _matrix_to_dict(self.gamma_loss),
            "gamma_gain": _matrix_to_dict(self.gamma_gain),
            "U": float(self.U),
            "impurity_site": self.impurity_site,
            "units": self.units,
            "input_tolerance": self.input_tolerance.to_dict(),
            "corrections": [correction.to_dict() for correction in self.corrections],
        }
        if self.interaction_sites != (self.impurity_site,):
            result["interaction_sites"] = list(self.interaction_sites)
        if self.density_interaction is not None:
            result["density_interaction"] = self.density_interaction.tolist()
        if self.two_body_interaction is not None:
            result["two_body_interaction"] = {
                "n_spin_orbitals": self.two_body_interaction.n_spin_orbitals,
                "terms": [
                    {
                        "indices": [p, q, r, s],
                        "coefficient": [float(value.real), float(value.imag)],
                    }
                    for p, q, r, s, value in self.two_body_interaction.terms
                ],
            }
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FermionicLindbladModel:
        """Read a canonical model dictionary without guessing historical types."""

        model_type = data.get("model_type", "FermionicLindbladModel")
        if model_type != "FermionicLindbladModel":
            raise ValueError(f"unsupported model_type {model_type!r}")
        tolerance_data = data.get("input_tolerance")
        tolerance = (
            TolerancePolicy.first().input_abs_rel
            if tolerance_data is None
            else AbsRelTolerance.from_dict(tolerance_data)
        )
        interaction_sites = data.get("interaction_sites")
        two_body_data = data.get("two_body_interaction")
        two_body = None
        if two_body_data is not None:
            two_body = SparseTwoBodyInteraction(
                n_spin_orbitals=int(two_body_data["n_spin_orbitals"]),
                terms=tuple(
                    (
                        *(int(index) for index in record["indices"]),
                        complex(*record["coefficient"]),
                    )
                    for record in two_body_data["terms"]
                ),
            )
        return cls(
            E=_matrix_from_dict(data["E"]),
            gamma_loss=_matrix_from_dict(data["gamma_loss"]),
            gamma_gain=_matrix_from_dict(data["gamma_gain"]),
            U=data["U"],
            impurity_site=int(data.get("impurity_site", 0)),
            units=str(data.get("units", "Gamma")),
            input_tolerance=tolerance,
            interaction_sites=(
                None
                if interaction_sites is None
                else tuple(int(site) for site in interaction_sites)
            ),
            density_interaction=data.get("density_interaction"),
            two_body_interaction=two_body,
            symmetry=data.get("symmetry", "spin"),
        )


def spin_lift(matrix: ArrayLike) -> ComplexMatrix:
    """Lift a spatial matrix to site-major, spin-minor ordering."""

    a = np.asarray(matrix, dtype=np.complex128)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    out = np.zeros((2 * a.shape[0], 2 * a.shape[1]), dtype=np.complex128)
    out[0::2, 0::2] = a
    out[1::2, 1::2] = a
    return out


def homotopy_model(
    physical: FermionicLindbladModel, K_ref: ArrayLike, s: float
) -> FermionicLindbladModel:
    r"""Return ``L0(K_ref) + s [L-L0(K_ref)]`` in physical coefficients."""

    K = np.asarray(K_ref, dtype=np.complex128)
    value = float(s)
    if K.shape != physical.E.shape or not 0.0 <= value <= 1.0:
        raise ValueError("K_ref must match the model and s must lie in [0,1]")
    if physical.two_body_interaction is None and physical.symmetry == "spin":
        density_interaction = value * physical.interaction_matrix
        two_body_interaction = None
    else:
        density_interaction = None
        two_body_interaction = SparseTwoBodyInteraction(
            physical.n_spin_orbitals,
            tuple(
                (p, q, r, s, value * coefficient)
                for p, q, r, s, coefficient in physical.interaction.terms
            ),
        )
    return FermionicLindbladModel(
        K + value * (physical.E - K),
        physical.gamma_loss,
        physical.gamma_gain,
        U=0.0,
        impurity_site=physical.impurity_site,
        interaction_sites=physical.interaction_sites,
        density_interaction=density_interaction,
        two_body_interaction=two_body_interaction,
        units=physical.units,
        input_tolerance=physical.input_tolerance,
        symmetry=physical.symmetry,
    )
