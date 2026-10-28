import importlib
import json
import logging
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def jax_logging():
    jax = pytest.importorskip("jax")
    logger = logging.getLogger("jax")
    handlers, propagate = list(logger.handlers), logger.propagate
    settings = {name: getattr(jax.config, name) for name in (
        "jax_enable_x64", "jax_log_compiles", "jax_enable_compilation_cache",
    )}
    yield jax
    for handler in list(logger.handlers):
        if handler not in handlers:
            logger.removeHandler(handler)
            handler.close()
    logger.propagate = propagate
    for name, value in settings.items():
        jax.config.update(name, value)


@pytest.mark.parametrize("model", ["inchmoss", "junction"])
@pytest.mark.parametrize("device", [None, "cpu", "gpu"])
def test_timing_cli_selects_requested_device(monkeypatch, tmp_path, jax_logging, model, device):
    timing = importlib.import_module(f"examples.{model}.gpu_timing")
    arguments = ["gpu_timing", "--rank", "S13", "--bath", "bath.npz",
                 "--reference", "reference.npz", "--output", str(tmp_path / "timing")]
    if device is not None:
        arguments += ["--device", device]
    monkeypatch.setattr(sys, "argv", arguments)
    selections, runs = [], []

    def devices(platform=None):
        selections.append(platform)
        return [SimpleNamespace(platform=platform)]

    monkeypatch.setattr(jax_logging, "devices", devices)
    monkeypatch.setattr(timing, "benchmark", runs.append)
    timing.main()
    assert selections == [device or "gpu"]
    assert len(runs) == 1
    assert runs[0].device == (device or "gpu")


@pytest.mark.parametrize("stage", ["static", "eom"])
@pytest.mark.parametrize("device", [None, "cpu", "gpu"])
def test_junction_cli_forwards_device(monkeypatch, stage, device):
    from examples.junction import cc

    arguments = ["junction.cc", stage]
    if device is not None:
        arguments += ["--device", device]
    monkeypatch.setattr(sys, "argv", arguments)
    calls = []
    monkeypatch.setattr(cc, "run_static", lambda phi, *, device="gpu": calls.append(("static", phi, device)))
    monkeypatch.setattr(cc, "run_eom", lambda args: calls.append(("eom", args.phi, args.device)))
    cc.main()
    assert calls == [(stage, 0., device or "gpu")]


@pytest.mark.parametrize("module,arguments", [
    ("examples.siam_ed.gpu", ["--output", "unused"]),
    ("examples.inchmoss.gpu_timing", ["--rank", "S13", "--bath", "unused", "--reference", "unused", "--output", "unused"]),
    ("examples.junction.gpu_timing", ["--rank", "S13", "--bath", "unused", "--reference", "unused", "--output", "unused"]),
    ("examples.junction.cc", ["static"]),
])
def test_cli_rejects_unknown_device(monkeypatch, module, arguments):
    driver = importlib.import_module(module)
    monkeypatch.setattr(sys, "argv", [module, *arguments, "--device", "tpu"])
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2


@pytest.mark.parametrize("builder", ["siam_ed", "inchmoss", "junction", "shared"])
@pytest.mark.parametrize("rank", ["S13", "S135"])
@pytest.mark.parametrize("device", [None, "cpu", "gpu"])
def test_response_builders_use_selected_jax_factory(monkeypatch, builder, rank, device):
    from c3l.cc import component_jax, component_response, component_tensor, eom, s135_full, s135_jax
    from examples.shared import response

    context = object()
    source = SimpleNamespace(addition=np.array([1., 2.]), removal=np.array([3., 4.]), sink=np.array([5., 6.]))
    static = SimpleNamespace(oracle=SimpleNamespace(model=SimpleNamespace(impurity_site=0)))
    monkeypatch.setattr(eom, "eom_ccsd_problem", lambda *args, **kwargs: object())
    monkeypatch.setattr(component_tensor.ComponentEOMProblem, "compile", lambda *args, **kwargs: object())
    monkeypatch.setattr(component_tensor, "component_eom_context", lambda *args: context)
    monkeypatch.setattr(component_tensor, "component_eom_sources", lambda *args, **kwargs: source)
    monkeypatch.setattr(response, "response_context", lambda *args: (context, (source,)))
    monkeypatch.setattr(component_response, "ComponentSigmaOperator", lambda context: SimpleNamespace(dimension=2))
    monkeypatch.setattr(s135_full, "CompiledFullS135Action", lambda context: context)
    native = SimpleNamespace(dimension=3, p_dimension=2, s5_dimension=1)
    monkeypatch.setattr(s135_full, "FullS135ResponseOperator", lambda *args, **kwargs: native)
    operator = SimpleNamespace(dimension=2 if rank == "S13" else 3)
    factories = []

    def retained_factory(value, **kwargs):
        factories.append(("S13", value, kwargs))
        return operator

    def full_factory(value, **kwargs):
        factories.append(("S135", value, kwargs))
        return operator

    monkeypatch.setattr(component_jax.JaxRetainedEOMOperator, "from_cpu", staticmethod(retained_factory))
    monkeypatch.setattr(s135_jax.JaxFullS135ResponseOperator, "from_cpu", staticmethod(full_factory))
    if builder == "shared":
        observed, sources = response.response_operator(context, (source,), device or "gpu", rank)
    else:
        driver = importlib.import_module(f"examples.{builder}.{'gpu' if builder == 'siam_ed' else 'gpu_timing'}")
        observed, sources = driver.build_response(static, np.zeros(2), rank,
                                                  **({} if device is None else {"device": device}))
        if builder == "siam_ed":
            sources = (sources,)
    assert observed is operator
    assert len(factories) == 1
    used_rank, argument, options = factories[0]
    assert used_rank == rank
    assert argument is (context if rank == "S13" else native)
    assert options["platform"] == (device or "gpu")
    assert options["production"] is True
    for embedded in sources:
        for name in ("addition", "removal", "sink"):
            expected = getattr(source, name)
            if rank == "S135":
                expected = np.r_[expected, 0.]
            np.testing.assert_array_equal(getattr(embedded, name), expected)


@pytest.mark.parametrize("driver", ["siam_ed", "inchmoss", "junction_timing", "junction_cc"])
@pytest.mark.parametrize("operator_has_device", [False, True])
def test_resident_drivers_place_true_residual_on_chain_device(monkeypatch, tmp_path, jax_logging, driver, operator_has_device):
    from c3l.green import resident_arnoldi

    jax = jax_logging
    jax.config.update("jax_enable_x64", True)
    selected = jax.devices("cpu")[0]
    real_device_put = jax.device_put
    placements, chains = [], []
    rhs = np.ones(1, dtype=complex)
    sinks = np.ones((1, 1), dtype=complex)
    shifts = np.array([.2j, .4j])
    operator = SimpleNamespace(dimension=1, device_matvec=lambda vector: -1j * vector)
    if operator_has_device:
        operator.device = selected

    class Resident:
        def __init__(self, action, values, projections, cap, *, device):
            assert device is (selected if operator_has_device else None)
            self.device = selected
            self.current = 1
            self.breakdown = True
            self.hessenberg = np.array([[-1j], [0.]])
            self.projected = np.array([[1., 0.]])
            self.action_seconds = []
            self.orthogonalization_seconds = []
            chains.append(self)

        def advance(self, target):
            assert target == 1

        def reconstruct(self, coefficients):
            return real_device_put(coefficients, self.device)

    def device_put(value, device=None, **kwargs):
        placements.append((np.asarray(value), device))
        return real_device_put(value, device, **kwargs)

    monkeypatch.setattr(resident_arnoldi, "ResidentArnoldi", Resident)
    monkeypatch.setattr(jax, "device_put", device_put)
    if driver == "siam_ed":
        from examples.siam_ed import gpu

        source = SimpleNamespace(addition=rhs, removal=np.zeros(1), sink=sinks[0])
        observed, record = gpu.resident_response(operator, source, shifts, 1, tmp_path / "source")
        observed = observed[:, None]
        assert record["basis_storage"] == "cpu_only"
    elif driver == "junction_cc":
        from examples.junction import cc

        observed, record = cc.solve_source(operator, rhs, sinks, shifts, tmp_path / "source", 1)
        assert record["orthogonalization_backend"] == "cpu_resident_cgs2"
    else:
        model = "junction" if driver == "junction_timing" else driver
        timing = importlib.import_module(f"examples.{model}.gpu_timing")
        observed, record = timing.source_response(operator, rhs, sinks, shifts, 1, 1, tmp_path / "source.npz")
    assert len(chains) == 1
    assert any(device is selected and np.array_equal(value, rhs) for value, device in placements)
    np.testing.assert_allclose(observed[:, 0], 1 / (shifts + 1j), atol=1e-12, rtol=1e-12)
    assert record["converged"]
    assert not list(tmp_path.rglob("*.npy"))


@pytest.mark.parametrize("device", [None, "cpu", "gpu"])
def test_junction_static_forwards_device_to_jax_solver(monkeypatch, tmp_path, device):
    from examples.junction import cc

    bath = tmp_path / "bath.npz"
    bath.write_bytes(b"test bath")
    bath_hash = cc.hashlib.sha256(bath.read_bytes()).hexdigest()
    monkeypatch.setattr(cc, "paper_parameters", lambda model: {"bath_files": {"4": {"path": str(bath), "sha256": bath_hash}}})
    monkeypatch.setattr(cc, "case_directory", lambda phi: tmp_path)
    monkeypatch.setattr(cc, "input_fingerprint", lambda: "test inputs")
    monkeypatch.setattr(cc, "assemble_model", lambda *args: object())
    monkeypatch.setattr(cc, "build_problem", lambda *args, **kwargs: (object(), object()))
    folder = tmp_path / "cc_N4"
    folder.mkdir()
    np.savez(folder / "reference.npz", fock=np.zeros((1, 1)), gamma_reference=np.zeros((1, 1)))

    class SolverReached(Exception):
        pass

    def solver(problem, initial, **kwargs):
        assert kwargs["platform"] == (device or "gpu")
        assert kwargs["production"] is True
        assert kwargs["policy"].tolerance == cc.cc_tolerance
        raise SolverReached

    monkeypatch.setattr(cc, "solve_component_ccsd_jax", solver)
    with pytest.raises(SolverReached):
        cc.run_static(0., **({} if device is None else {"device": device}))


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_junction_eom_forwards_device_to_response_factory(monkeypatch, tmp_path, device):
    from examples.junction import cc
    from examples.shared import response

    bath = tmp_path / "bath.npz"
    bath.write_bytes(b"test bath")
    bath_hash = cc.hashlib.sha256(bath.read_bytes()).hexdigest()
    monkeypatch.setattr(cc, "paper_parameters", lambda model: {"bath_files": {"4": {"path": str(bath), "sha256": bath_hash}}})
    monkeypatch.setattr(cc, "case_directory", lambda phi: tmp_path)
    monkeypatch.setattr(cc, "input_fingerprint", lambda: "test inputs")
    monkeypatch.setattr(cc, "assemble_model", lambda *args: object())
    monkeypatch.setattr(cc, "build_problem", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr(cc, "fresh_residual", lambda *args: 1e-12)
    monkeypatch.setattr(response, "response_context", lambda *args: (object(), ()))
    folder = tmp_path / "cc_N4"
    folder.mkdir()
    for filename, record in (
        ("configuration.json", {"input_sha256": "test inputs"}),
        ("solve.json", {"converged": True, "acceptance_tolerance": 1e-10}),
        ("observables.json", {"checks": {"passed": True}}),
    ):
        (folder / filename).write_text(json.dumps(record))
    np.savez(folder / "static.npz", amplitudes=np.ones(1), fock=np.zeros((1, 1)),
             gamma_reference=np.zeros((1, 1)), phi=0., beta=cc.beta)

    class FactoryReached(Exception):
        pass

    def factory(context, sources, platform, rank):
        assert platform == device
        assert rank == "S13"
        raise FactoryReached

    monkeypatch.setattr(response, "response_operator", factory)
    with pytest.raises(FactoryReached):
        cc.run_eom(SimpleNamespace(phi=0., rank="S13", cap=1, device=device))


@pytest.fixture
def junction_static_case(monkeypatch, tmp_path):
    from examples.junction import cc

    bath = tmp_path / "bath.npz"
    bath.write_bytes(b"test bath")
    bath_hash = cc.hashlib.sha256(bath.read_bytes()).hexdigest()
    monkeypatch.setattr(cc, "paper_parameters", lambda model: {"bath_files": {"4": {"path": str(bath), "sha256": bath_hash}}})
    monkeypatch.setattr(cc, "case_directory", lambda phi: tmp_path)
    monkeypatch.setattr(cc, "input_fingerprint", lambda: "test inputs")
    folder = tmp_path / "cc_N4"
    folder.mkdir()
    parameters = dict(beta=cc.beta, phi=0., input_sha256="test inputs", bath_sha256=bath_hash)
    return cc, folder, parameters


@pytest.mark.parametrize("previous_device", [None, "gpu"])
def test_junction_fresh_cpu_records_execution_metadata(monkeypatch, junction_static_case, previous_device):
    cc, folder, parameters = junction_static_case
    if previous_device is not None:
        parameters["execution"] = dict(backend="jax", device=previous_device)
        (folder / "configuration.json").write_text(json.dumps(parameters))
    np.savez(folder / "reference.npz", fock=np.zeros((1, 1)), gamma_reference=np.zeros((1, 1)))
    basis = SimpleNamespace(V=np.eye(1))
    problem = SimpleNamespace(reference=SimpleNamespace(up=basis, down=basis))
    monkeypatch.setattr(cc, "assemble_model", lambda *args: object())
    monkeypatch.setattr(cc, "build_problem", lambda *args, **kwargs: (problem, object()))
    monkeypatch.setattr(cc, "interaction_steps", (1.,))
    monkeypatch.setattr(cc, "fresh_residual", lambda *args: 1e-12)
    solves = []

    def solver(component, initial, **kwargs):
        solves.append(kwargs)
        return SimpleNamespace(amplitudes=np.ones(1), iterations=1)

    monkeypatch.setattr(cc, "solve_component_ccsd_jax", solver)
    observables = dict(
        density_pairs=.25 * np.ones((8, 8)) + .25 * np.eye(8),
        maximum_observable_imaginary_defect=0., site_occupations=np.ones(4),
        h1_l1_spin_correlation=0.,
    )
    monkeypatch.setattr(cc, "cc_observables", lambda *args: (.5 * np.eye(8), observables))
    monkeypatch.setattr(cc, "contact_currents", lambda *args: dict(I_tip=0., I_substrate=0., F_tip=0., F_substrate=0.))
    monkeypatch.setattr(cc, "strengths", lambda *args: dict(R=0.))
    record = cc.run_static(0., device="cpu")
    saved = json.loads((folder / "solve.json").read_text())
    configuration = json.loads((folder / "configuration.json").read_text())
    assert saved["execution"] == dict(backend="jax", device="cpu")
    assert configuration["execution"] == dict(backend="jax", device=previous_device or "cpu")
    assert record == dict(saved, cached=False)
    assert len(solves) == 1
    assert solves[0]["platform"] == "cpu"
    assert solves[0]["production"] is True


@pytest.mark.parametrize("original_device", [None, "gpu", "cpu"])
def test_junction_cached_static_preserves_original_timing(monkeypatch, capsys, junction_static_case, original_device):
    cc, folder, parameters = junction_static_case
    saved = dict(converged=True, residual=1e-12, seconds=123.456, continuation=[])
    if original_device is not None:
        saved["execution"] = dict(backend="jax", device=original_device)
        parameters["execution"] = saved["execution"].copy()
    originals = {}
    for filename, contents in (
        ("configuration.json", parameters),
        ("solve.json", saved),
        ("observables.json", {"checks": {"passed": True}, "seconds": 124.}),
    ):
        originals[filename] = json.dumps(contents)
        (folder / filename).write_text(originals[filename])

    def forbidden(*args, **kwargs):
        pytest.fail("accepted static cache must not rerun model construction or numerical work")

    monkeypatch.setattr(cc, "assemble_model", forbidden)
    monkeypatch.setattr(cc, "solve_component_ccsd_jax", forbidden)
    monkeypatch.setattr(cc, "solve_ghf", forbidden)
    monkeypatch.setattr(cc, "cc_observables", forbidden)
    record = cc.run_static(0., device="cpu")
    execution = dict(backend="unknown", device="unknown") if original_device is None else saved["execution"]
    assert record == dict(saved, execution=execution, cached=True)
    message = capsys.readouterr().out.lower()
    assert "cached saved static" in message
    assert "no new" in message and "timing" in message
    assert execution["device"] in message
    for filename, contents in originals.items():
        assert (folder / filename).read_text() == contents


def test_junction_failed_cpu_solve_records_execution_metadata(monkeypatch, junction_static_case):
    cc, folder, parameters = junction_static_case
    np.savez(folder / "reference.npz", fock=np.zeros((1, 1)), gamma_reference=np.zeros((1, 1)))
    monkeypatch.setattr(cc, "assemble_model", lambda *args: object())
    monkeypatch.setattr(cc, "build_problem", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr(cc, "interaction_steps", (1.,))
    monkeypatch.setattr(cc, "gmres_attempts", ((4, .5),))
    monkeypatch.setattr(cc, "fresh_residual", lambda *args: 1e-5)
    monkeypatch.setattr(cc, "solve_component_ccsd_jax", lambda *args, **kwargs: SimpleNamespace(amplitudes=np.ones(1), iterations=1))
    with pytest.raises(RuntimeError, match="CCSD did not converge"):
        cc.run_static(0., device="cpu")
    saved = json.loads((folder / "solve.json").read_text())
    assert saved["converged"] is False
    assert saved["execution"] == dict(backend="jax", device="cpu")
