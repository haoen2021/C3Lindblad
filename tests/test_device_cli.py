import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module,arguments", [
    ("examples.siam_ed.gpu", []),
    ("examples.inchmoss.gpu_timing", []),
    ("examples.junction.gpu_timing", []),
    ("examples.junction.cc", ["static"]),
    ("examples.junction.cc", ["eom"]),
    ("examples.shared.anderson_response", []),
    ("examples.shared.multiorbital_response", []),
    ("examples.kanamori.realtimebath_n5_campaign", []),
    ("examples.siam_modes.run_s135", []),
    ("examples.siam_modes.run_s135_ladder", []),
    ("examples.inchmoss.comparison.cc_static", []),
    ("examples.inchmoss.comparison.eom_green", []),
    ("examples.inchmoss.run_mbpt", []),
    ("examples.junction.mbpt", []),
])
def test_device_option_available_without_gpu_initialization(module, arguments):
    environment = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
                       PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run(
        [sys.executable, "-m", module, *arguments, "--device", "cpu", "--help"],
        capture_output=True, text=True, env=environment, timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "--device {cpu,gpu}" in result.stdout
