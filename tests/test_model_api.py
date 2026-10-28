"""Model serialization, spin lifting and paper bath conventions."""

from __future__ import annotations

import importlib
from pathlib import Path
import pytest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_generic_model_helpers_are_available_without_benchmarks() -> None:
    from c3l import models

    assert callable(getattr(models, "homotopy_model", None))
    assert callable(getattr(models, "spin_lift", None))
    matrix = np.asarray([[0.3, 0.2j], [-0.2j, -0.5]])
    np.testing.assert_array_equal(models.spin_lift(matrix), np.kron(matrix, np.eye(2)))
    with pytest.raises(ValueError, match="square"):
        models.spin_lift(np.ones((2, 3)))


def test_core_does_not_import_benchmark_or_script_modules() -> None:
    import ast

    for path in (ROOT / "src" / "c3l").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert not any(
                module.split(".")[0] in {"benchmarks", "scripts"}
                for module in modules
            ), path


def test_response_uses_paper_sector_names() -> None:
    assert importlib.util.find_spec("c3l.cc.s135_full") is not None
    assert importlib.util.find_spec("c3l.cc.s135_jax") is not None
    from c3l.cc.s135_full import CompiledFullS135Action, FullS135ResponseOperator
    from c3l.cc.s135_jax import JaxFullS135ResponseOperator, build_full_s135_jax_plan

    assert all(callable(value) for value in (
        CompiledFullS135Action, FullS135ResponseOperator,
        JaxFullS135ResponseOperator, build_full_s135_jax_plan,
    ))


def test_semicircle_recipe_uses_explicit_physical_parameters() -> None:
    from examples.shared import bath_targets as hybridization

    assert callable(getattr(hybridization, "semicircle_bcf", None))
    times = np.asarray([0.0, 0.2])
    lesser = hybridization.semicircle_bcf(
        times, lesser=True, W=2.0, Gamma_bath=0.4, beta=3.0, quadrature_order=128
    )
    greater = hybridization.semicircle_bcf(
        times, lesser=False, W=2.0, Gamma_bath=0.4, beta=3.0, quadrature_order=128
    )
    np.testing.assert_allclose(lesser, greater.conj(), atol=1e-14)
    np.testing.assert_allclose(lesser[0], 0.2, atol=1e-7)
    with pytest.raises(TypeError):
        hybridization.semicircle_bcf(times, lesser=True)

def _inputs() -> dict[str, object]:
    return {
        "E": np.asarray([[0.2, -0.3], [-0.3, 0.7]], dtype=np.complex128),
        "gamma_loss": np.asarray([[0.4, 0.0], [0.0, 0.1]]),
        "gamma_gain": np.asarray([[0.2, 0.0], [0.0, 0.3]]),
        "U": 1.5,
        "impurity_site": 0,
    }


def test_canonical_model_roundtrip_and_unknown_tag_rejection() -> None:
    from c3l import FermionicLindbladModel

    model = FermionicLindbladModel(**_inputs())
    data = model.to_dict()
    assert data["model_type"] == "FermionicLindbladModel"
    restored = FermionicLindbladModel.from_dict(data)
    np.testing.assert_array_equal(restored.E, model.E)
    np.testing.assert_array_equal(restored.interaction_matrix, model.interaction_matrix)
    with pytest.raises(ValueError, match="model_type"):
        FermionicLindbladModel.from_dict(dict(data, model_type="AuxiliaryAIM"))
