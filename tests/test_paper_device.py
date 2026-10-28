import argparse
import importlib
import os
import runpy
import sys
from types import SimpleNamespace

import pytest


EXAMPLES = (
    ("examples.shared.anderson_response", ["pilot"]),
    ("examples.shared.multiorbital_response", ["all"]),
    ("examples.kanamori.realtimebath_n5_campaign", ["eom", "--space", "S13"]),
    ("examples.siam_modes.run_s135", []),
    ("examples.siam_modes.run_s135_ladder", ["--modes", "1"]),
)


def _parse_example_arguments(monkeypatch, module_name, arguments):
    parsed = []
    original = argparse.ArgumentParser.parse_args

    class ParsedArguments(Exception):
        pass

    def parse_args(parser, *args, **kwargs):
        parsed.append(original(parser, *args, **kwargs))
        raise ParsedArguments

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_args)
    monkeypatch.setattr(sys, "argv", [module_name, *arguments])
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    with pytest.raises(ParsedArguments):
        runpy.run_module(module_name, run_name="__main__")
    return parsed[0]


@pytest.mark.parametrize("module_name,arguments", EXAMPLES)
@pytest.mark.parametrize("selection,expected", [([], "gpu"), (["--device", "cpu"], "cpu"), (["--device", "gpu"], "gpu")])
def test_example_device_selection(monkeypatch, module_name, arguments, selection, expected):
    parsed = _parse_example_arguments(monkeypatch, module_name, arguments + selection)
    assert parsed.device == expected


@pytest.mark.parametrize("module_name,arguments", (EXAMPLES[1], EXAMPLES[3]))
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_backend_remains_a_device_alias(monkeypatch, module_name, arguments, device):
    parsed = _parse_example_arguments(monkeypatch, module_name, arguments + ["--backend", device])
    assert parsed.device == device


@pytest.mark.parametrize("module_name,arguments", EXAMPLES)
def test_example_rejects_unknown_device(monkeypatch, module_name, arguments):
    with pytest.raises(SystemExit) as raised:
        _parse_example_arguments(monkeypatch, module_name, arguments + ["--device", "tpu"])
    assert raised.value.code == 2


@pytest.mark.parametrize("module_name", [EXAMPLES[0][0], EXAMPLES[1][0]])
def test_jax_configuration_tracks_selected_device(monkeypatch, tmp_path, module_name):
    runner = importlib.import_module(module_name)
    for name in ("JAX_PLATFORMS", "JAX_ENABLE_X64", "JAX_COMPILATION_CACHE_DIR", "XLA_PYTHON_CLIENT_PREALLOCATE"):
        monkeypatch.delenv(name, raising=False)
    gpu_cache = runner.configure_jax(tmp_path, device="gpu")
    assert os.environ["JAX_PLATFORMS"] in ("cuda", "gpu")
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == str(gpu_cache)
    cpu_cache = runner.configure_jax(tmp_path, device="cpu")
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == str(cpu_cache)
    assert cpu_cache != gpu_cache
    assert cpu_cache.is_dir() and gpu_cache.is_dir()
    assert os.environ["JAX_ENABLE_X64"].lower() == "true"


@pytest.mark.parametrize("available", [None, 16 * 2**30])
def test_resource_gate_checks_selected_cpu(monkeypatch, available):
    from c3l.green import resident_arnoldi
    from examples.shared import multiorbital_response

    selected = SimpleNamespace(platform="cpu", memory_stats=lambda: None)
    calls = []

    def check_memory(device, required_bytes, workspace_bytes=8 * 2**30):
        calls.append((device, required_bytes, workspace_bytes))
        return available

    monkeypatch.setattr(resident_arnoldi, "check_resident_memory", check_memory, raising=False)
    record = multiorbital_response._resource_gate(8, device=selected)
    assert calls == [(selected, 4096, 8 * 2**30)]
    assert record["required_device_bytes"] == 4096 + 8 * 2**30
    assert record["available_device_bytes"] == available


def test_resource_gate_propagates_insufficient_cpu_memory(monkeypatch):
    from c3l.green import resident_arnoldi
    from examples.shared import multiorbital_response

    def check_memory(device, required_bytes, workspace_bytes=8 * 2**30):
        raise MemoryError("insufficient CPU RAM")

    monkeypatch.setattr(resident_arnoldi, "check_resident_memory", check_memory, raising=False)
    with pytest.raises(MemoryError, match="insufficient CPU RAM"):
        multiorbital_response._resource_gate(8, device=SimpleNamespace(platform="cpu"))
