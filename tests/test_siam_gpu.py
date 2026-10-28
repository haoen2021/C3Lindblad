from types import SimpleNamespace

import numpy as np
import pytest

from examples.siam_ed import run


def small_fit():
    from examples.shared.bath import CoupledPseudomodeFit

    return CoupledPseudomodeFit(
        N=1, H_lesser=np.array([[-.7]]), Gamma_lesser=np.array([[.2]]),
        g_lesser=np.array([.3]), H_greater=np.array([[.7]]),
        Gamma_greater=np.array([[.2]]), g_greater=np.array([.3]),
    )


def test_new_interaction_preserves_archived_bath(monkeypatch):
    monkeypatch.setattr(run, "load_fit", small_fit)
    old = run.build_model(5.)
    new = run.build_model(6.)
    assert new.U == 6.
    assert new.E[0, 0] == -3.
    np.testing.assert_array_equal(new.E[1:], old.E[1:])
    np.testing.assert_array_equal(new.E[0, 1:], old.E[0, 1:])
    np.testing.assert_array_equal(new.gamma_gain, old.gamma_gain)
    np.testing.assert_array_equal(new.gamma_loss, old.gamma_loss)


@pytest.mark.parametrize("interaction", [np.nan, np.inf, -1.])
def test_invalid_interaction_rejected_before_loading_bath(monkeypatch, interaction):
    def forbidden():
        pytest.fail("invalid interaction must not load the bath")

    monkeypatch.setattr(run, "load_fit", forbidden)
    with pytest.raises(ValueError, match="finite nonnegative"):
        run.build_model(interaction)


@pytest.mark.parametrize("device", [None, "cpu", "gpu"])
def test_continuation_uses_seed_then_previous_root(monkeypatch, tmp_path, device):
    from examples.siam_ed import gpu

    calls = []
    basis = SimpleNamespace(V=np.eye(2))
    reference = SimpleNamespace(up=basis, down=basis)
    problem = SimpleNamespace(oracle=SimpleNamespace(reference=reference, num_variables=875))
    monkeypatch.setattr(run, "build_static_problem", lambda interaction: problem)

    def solver(problem, initial, **kwargs):
        calls.append((initial.copy(), kwargs))
        return SimpleNamespace(
            amplitudes=initial + .1, converged=True, packed_residual_norm=1e-12,
            residual_norm=1e-12, iterations=2, history=[],
        )

    result = gpu.continue_static(
        problem, np.array([.2 + .1j]), [5.5, 6.], gpu.static_policy(320),
        tmp_path, solver=solver, **({} if device is None else {"device": device}),
    )
    np.testing.assert_allclose(calls[0][0], [.2 + .1j])
    np.testing.assert_allclose(calls[1][0], [.3 + .1j])
    np.testing.assert_allclose(result[1], [.4 + .1j])
    assert all(kwargs["platform"] == (device or "gpu") for _, kwargs in calls)
    assert all(kwargs["production"] is True for _, kwargs in calls)
    assert calls[0][1]["policy"].gmres_restart == 320
    assert (tmp_path / "U6/static.npz").exists()


@pytest.mark.parametrize("residual", [np.nan, np.inf, 1e-5])
def test_failed_static_never_becomes_an_accepted_checkpoint(monkeypatch, tmp_path, residual):
    from examples.siam_ed import gpu

    basis = SimpleNamespace(V=np.eye(2))
    problem = SimpleNamespace(oracle=SimpleNamespace(
        reference=SimpleNamespace(up=basis, down=basis), num_variables=875,
    ))
    monkeypatch.setattr(run, "build_static_problem", lambda interaction: problem)

    def solver(problem, initial, **kwargs):
        return SimpleNamespace(
            amplitudes=initial, converged=True, packed_residual_norm=residual,
            residual_norm=residual, iterations=1, history=[],
        )

    with pytest.raises(RuntimeError, match="Static solve failed"):
        gpu.continue_static(problem, np.ones(1), [6.], gpu.static_policy(320), tmp_path, solver=solver)
    assert not (tmp_path / "U6/static.npz").exists()


def test_resident_response_matches_dense_and_stores_no_basis(tmp_path):
    from examples.siam_ed import gpu
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    generator = np.random.default_rng(63)
    dimension = 8
    matrix = .1 * generator.normal(size=(dimension, dimension)) - 1j * np.eye(dimension)
    rhs = generator.normal(size=dimension).astype(complex)
    sink = generator.normal(size=dimension).astype(complex)
    device_matrix = jnp.asarray(matrix)
    operator = SimpleNamespace(
        dimension=dimension, device_matvec=lambda vector: device_matrix @ vector,
    )
    source = SimpleNamespace(addition=.4 * rhs, removal=.6 * rhs, sink=sink)
    shifts = np.array([-1 + .1j, .1j, 1 + .1j])
    observed, record = gpu.resident_response(operator, source, shifts, 1024, tmp_path / "S13")
    expected = np.array([sink @ np.linalg.solve(shift * np.eye(dimension) - matrix, rhs) for shift in shifts])
    np.testing.assert_allclose(observed, expected, rtol=1e-11, atol=1e-11)
    assert record["converged"]
    assert record["dimension"] <= dimension
    assert not list(tmp_path.rglob("*.npy"))


def test_failed_response_is_recorded_but_not_accepted(tmp_path):
    from examples.siam_ed import gpu
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    operator = SimpleNamespace(
        dimension=8, device_matvec=lambda vector: (jnp.arange(8) - 1j) * vector,
    )
    source = SimpleNamespace(addition=np.ones(8), removal=np.zeros(8), sink=np.ones(8))
    with pytest.raises(RuntimeError, match="EOM did not converge"):
        gpu.resident_response(operator, source, np.array([.1j]), 1, tmp_path / "S13")
    assert (tmp_path / "S13.json").exists()


def test_ph_reference_coordinates_match_across_interactions(monkeypatch):
    monkeypatch.setattr(run, "load_fit", small_fit)
    old = run.build_static_problem(5.)
    new = run.build_static_problem(6.)
    for spin in ("up", "down"):
        np.testing.assert_allclose(
            getattr(old.oracle.reference, spin).V,
            getattr(new.oracle.reference, spin).V, rtol=0, atol=1e-13,
        )


@pytest.mark.parametrize("interactions", [None, [3., 5., 8.]])
@pytest.mark.parametrize("device", [None, "cpu"])
def test_gpu_campaign_computes_both_responses_at_every_interaction(monkeypatch, tmp_path, interactions, device):
    import json
    import logging
    import sys
    import jax
    from examples.siam_ed import gpu

    expected = [2., 4., 6., 8., 10.] if interactions is None else interactions
    output = tmp_path / "campaign"
    arguments = ["siam_ed.gpu", "--output", str(output)]
    if interactions is not None:
        arguments += ["--U", *map(str, interactions)]
    if device is not None:
        arguments += ["--device", device]
    monkeypatch.setattr(sys, "argv", arguments)
    selected = device or "gpu"

    def devices(platform=None):
        assert platform == selected
        return [SimpleNamespace(platform=selected, device_kind=f"test {selected}")]

    monkeypatch.setattr(jax, "devices", devices)
    events = []
    seeds = []

    def load_seed(path):
        seeds.append(path)
        return SimpleNamespace(interaction=2.), np.array([2.]), {"U": 2.}

    def static(previous, initial, points, policy, destination, *, device="gpu"):
        assert device == selected
        assert len(points) == 1
        interaction = points[0]
        expected_previous = 2. if interaction == expected[0] else expected[expected.index(interaction) - 1]
        np.testing.assert_array_equal(initial, [expected_previous])
        (destination / f"U{interaction:g}").mkdir()
        events.append((interaction, "static"))
        return SimpleNamespace(interaction=interaction), np.array([interaction]), [{"U": interaction}]

    def build_response(problem, amplitudes, rank, *, device="gpu"):
        assert device == selected
        np.testing.assert_array_equal(amplitudes, [problem.interaction])
        return SimpleNamespace(interaction=problem.interaction, rank=rank), None

    def response(operator, source, shifts, cap, destination):
        assert destination == output / f"U{operator.interaction:g}" / operator.rank
        events.append((operator.interaction, operator.rank))
        np.savez(destination.with_suffix(".npz"), retarded=np.ones(len(shifts)))
        return np.ones(len(shifts)), {"converged": True}

    monkeypatch.setattr(gpu, "load_seed", load_seed)
    monkeypatch.setattr(gpu, "continue_static", static)
    monkeypatch.setattr(gpu, "build_response", build_response)
    monkeypatch.setattr(gpu, "resident_response", response)
    logger = logging.getLogger("jax")
    handlers, propagate = list(logger.handlers), logger.propagate
    logging_compiles = jax.config.jax_log_compiles
    try:
        gpu.main()
    finally:
        for handler in list(logger.handlers):
            if handler not in handlers:
                logger.removeHandler(handler)
                handler.close()
        logger.propagate = propagate
        jax.config.update("jax_log_compiles", logging_compiles)
    assert seeds[0].name == "U2.npz"
    assert events == [(interaction, stage) for interaction in expected for stage in ("static", "S13", "S135")]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["device"] == f"test {selected}"
    assert [record["U"] for record in summary["static"]] == expected
    for interaction in expected:
        assert set(summary["responses"][f"U{interaction:g}"]) == {"S13", "S135"}
        for rank in ("S13", "S135"):
            assert (output / f"U{interaction:g}/{rank}.npz").exists()
