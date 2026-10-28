import numpy as np
import pytest

from examples.kanamori import run
from tests.reference.physical_fock import annihilation_operators, interaction_hamiltonian


def test_full_kanamori_matches_physical_operators_and_atomic_multiplets():
    annihilation = annihilation_operators(4)
    creation = tuple(operator.conj().T for operator in annihilation)
    number = tuple(creator @ operator for creator, operator in zip(creation, annihilation))
    exchange = creation[0] @ creation[3] @ annihilation[1] @ annihilation[2]
    pair = creation[0] @ creation[1] @ annihilation[3] @ annihilation[2]
    expected = (
        2.0 * (number[0] @ number[1] + number[2] @ number[3])
        + 1.6 * (number[0] @ number[3] + number[1] @ number[2])
        + 1.4 * (number[0] @ number[2] + number[1] @ number[3])
        + 0.2 * (exchange + exchange.conj().T + pair + pair.conj().T)
    )
    actual = interaction_hamiltonian(run.interaction(), annihilation)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
    sector = [state for state in range(16) if state.bit_count() == 2]
    np.testing.assert_allclose(
        np.linalg.eigvalsh(actual[np.ix_(sector, sector)]),
        [1.4, 1.4, 1.4, 1.8, 1.8, 2.2], atol=1e-14, rtol=0,
    )


def test_requested_chemical_potential_is_distinct_from_half_filling():
    np.testing.assert_allclose(run.impurity_one_body("chemical_potential"), -np.eye(2))
    np.testing.assert_allclose(run.impurity_one_body("particle_hole_symmetric"), -2.5 * np.eye(2))


def test_corrected_results_use_a_new_namespace():
    assert run.results_directory(8, "chemical_potential").name == "kanamori_full_beta8_chemical_potential_v2"


class DenseOperator:
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return DenseOperator(self.value + (other.value if isinstance(other, DenseOperator) else other))

    __radd__ = __add__

    def __mul__(self, other):
        return DenseOperator(self.value @ other.value if isinstance(other, DenseOperator) else self.value * other)

    __rmul__ = __mul__


def test_cthyb_builder_matches_the_same_full_kanamori(monkeypatch):
    import sys
    from types import ModuleType
    from examples.kanamori import cthyb_half_filling

    annihilation = annihilation_operators(4)
    operators = ModuleType("triqs.operators")
    operators.c = lambda spin, orbital: DenseOperator(annihilation[2 * orbital + (spin == "dn")])
    operators.c_dag = lambda spin, orbital: DenseOperator(operators.c(spin, orbital).value.conj().T)
    operators.n = lambda spin, orbital: operators.c_dag(spin, orbital) * operators.c(spin, orbital)
    triqs = ModuleType("triqs")
    triqs.operators = operators
    monkeypatch.setitem(sys.modules, "triqs", triqs)
    monkeypatch.setitem(sys.modules, "triqs.operators", operators)
    np.testing.assert_allclose(
        cthyb_half_filling._interaction().value,
        interaction_hamiltonian(run.interaction(), annihilation), atol=1e-14, rtol=0,
    )
    assert cthyb_half_filling.CHEMICAL_POTENTIAL == 1.0


def test_cthyb_campaign_rejects_old_model_data(tmp_path):
    import json
    from examples.kanamori import cthyb_long_campaign

    beta, cycles, seed = cthyb_long_campaign.task_spec(0)
    folder = tmp_path / f"beta{int(beta)}" / f"seed_{seed}"
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps(dict(
        solve_status=0, cycles=cycles, seed=seed, chemical_potential=2.5,
        interaction="literal_printed_eq5_double_exchange_no_pair_hopping",
    )))
    (folder / "compact.npz").touch()
    with pytest.raises(RuntimeError, match="different Hamiltonian"):
        cthyb_long_campaign.run_task(0, tmp_path)


@pytest.fixture
def cthyb_chain_archive(tmp_path, monkeypatch):
    import json
    from examples.kanamori import cthyb_long_campaign

    monkeypatch.setattr(cthyb_long_campaign, "CHAINS_PER_BETA", 2)
    folders = []
    for task in range(2):
        beta, cycles, seed = cthyb_long_campaign.task_spec(task)
        folder = tmp_path / f"beta{int(beta)}" / f"seed_{seed}"
        folder.mkdir(parents=True)
        record = dict(
            beta=beta, solve_status=0, cycles=cycles, seed=seed,
            chemical_potential=1.0,
            interaction="full_kanamori_spin_exchange_and_pair_hopping",
            average_sign=.5, average_order=2., auto_corr_time=1.,
            wall_seconds=2., spin_defect=0., orbital_diagonal_defect=0.,
            reciprocity_defect=0.,
        )
        (folder / "run.json").write_text(json.dumps(record))
        tau = np.linspace(0., beta, 3)
        iw = 1j * np.array([1., 3.]) * np.pi / beta
        g_tau = np.repeat((-.5 * np.eye(2))[None], tau.size, axis=0)
        g_iw = np.eye(2)[None] / (iw[:, None, None] + 1.)
        np.savez(folder / "compact.npz", tau=tau, iw=iw,
                 G_tau_up=g_tau, G_tau_dn=g_tau, G_iw_up=g_iw, G_iw_dn=g_iw)
        folders.append(folder)
    return tmp_path, folders


@pytest.mark.parametrize("mismatch", [
    {"chemical_potential": 2.5},
    {"interaction": "literal_printed_eq5_double_exchange_no_pair_hopping"},
    {"chemical_potential": None},
    {"interaction": None},
])
def test_cthyb_aggregation_rejects_mixed_or_unidentified_models(cthyb_chain_archive, mismatch):
    import json
    from examples.kanamori import cthyb_long_campaign

    root, folders = cthyb_chain_archive
    path = folders[-1] / "run.json"
    record = json.loads(path.read_text())
    record.update(mismatch)
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="different Hamiltonian"):
        cthyb_long_campaign._load_temperature(root, 8.)


def test_cthyb_curated_data_retain_model_identity(cthyb_chain_archive):
    from examples.kanamori import cthyb_long_campaign

    root, _ = cthyb_chain_archive
    chains = cthyb_long_campaign._load_temperature(root, 8.)
    result = cthyb_long_campaign._curated_arrays(8., chains)
    assert result["chemical_potential"].item() == 1.0
    assert result["interaction"].item() == "full_kanamori_spin_exchange_and_pair_hopping"
    np.testing.assert_allclose(result["G_tau_00_mean"], -.5)
