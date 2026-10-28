"""Full S135 response for the saved beta=1, N=6+6 RealTimeBath SIAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from ..parameters import parameters

PARAMETERS = parameters("siam_modes")
from .example04 import siam
from time import perf_counter


import numpy as np

from c3l.cc.component_tensor import (
    ComponentStaticProblem,
    component_static_residual,
)
from c3l.cc.factorized import factorized_cc_problem
from ..shared.anderson_response import (
    _build_operator,
    _run_source,
    _source_final,
    _source_metadata,
    configure_jax,
    portable_checkpoint_amplitudes,
)


REPO = Path.cwd()
OUTPUT = REPO / "siam_modes/results"


def main(output: Path = OUTPUT, device: str = "gpu") -> None:
    configure_jax(output / "s135", device=device)
    started = perf_counter()
    saved = output / "cc_reference"
    fit = json.loads((saved / "bath_N6.json").read_text())
    static_record = json.loads((saved / "cc_N6.json").read_text())
    if float(fit["beta"]) != PARAMETERS["beta"] or float(static_record["beta"]) != PARAMETERS["beta"]:
        raise ValueError("saved bath or CCSD state does not have beta=1")
    with np.load(saved / "bath_N6.npz", allow_pickle=False) as bath:
        model = siam(bath["H"], bath["D"], bath["g"])
    cc = factorized_cc_problem(
        model,
        "CCSD",
        m_up=0.5,
        build_next_rank_masks=False,
        reference_mode="legacy_half_filled",
    )
    static = ComponentStaticProblem.compile(cc)
    with np.load(saved / "cc_N6_latest.npz", allow_pickle=False) as checkpoint:
        amplitudes = np.asarray(checkpoint["amplitudes"], dtype=np.complex128)
    if amplitudes.shape != (cc.num_amplitudes,):
        raise ValueError(
            "saved CCSD amplitude dimension differs from the rebuilt model"
        )
    residual = float(
        np.linalg.norm(component_static_residual(static, amplitudes).retained)
    )
    aligned = False
    if residual > 1e-8:
        adjustment = portable_checkpoint_amplitudes(static, amplitudes)
        amplitudes = adjustment.amplitudes
        residual = float(
            np.linalg.norm(component_static_residual(static, amplitudes).retained)
        )
        aligned = bool(adjustment.resolved)
    if residual > 1e-8:
        raise RuntimeError(
            f"saved CCSD amplitudes fail the fresh residual gate: {residual:.3e}"
        )
    print(
        json.dumps(
            {"event": "static_gate", "residual": residual, "gauge_aligned": aligned}
        ),
        flush=True,
    )

    operator, sources, compiled, setup_seconds = _build_operator(
        static, amplitudes, (model.impurity_site,), device=device
    )
    device = getattr(getattr(operator, "device", None), "platform", device)
    if (operator.p_dimension, operator.s5_dimension) != (3224, 272766):
        raise RuntimeError(
            f"unexpected response dimensions: {(operator.p_dimension, operator.s5_dimension)}"
        )
    print(
        json.dumps(
            {
                "event": "response_operator",
                "device": operator.runtime.device_kind,
                "backend": device,
                "P": operator.p_dimension,
                "Q5": operator.s5_dimension,
                "setup_seconds": setup_seconds,
            }
        ),
        flush=True,
    )

    q13_data = np.loadtxt(saved / "spectrum_N6.csv", delimiter=",", skiprows=1)
    omega = np.asarray(q13_data[:, 0], dtype=np.float64)
    q13 = np.asarray(q13_data[:, 3], dtype=np.float64)
    eta = PARAMETERS["eta"]
    from c3l.green.checkpointed_arnoldi import eom_checkpoint_identity

    checkpoint_identity = eom_checkpoint_identity(compiled.context)
    source = _run_source(
        operator, sources[0], sources, omega, "frequency",
        output / "s135" / f"source_0_{device}", eta=eta,
        provenance=f"siam-beta1-n6-s135:d67c534:{checkpoint_identity}",
    )
    green = _source_final(source)[:, 0]
    spectrum = -green.imag / np.pi
    result_dir = output / "s135"
    result_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        result_dir / "spectrum_N6_S135.csv",
        np.column_stack((omega, green.real, green.imag, spectrum)),
        delimiter=",",
        header="omega,Re_G_up,Im_G_up,A_up",
        comments="",
    )
    metadata = _source_metadata(source)
    metadata.update(
        beta=PARAMETERS["beta"],
        U=PARAMETERS["U"],
        N_lesser=6,
        N_greater=6,
        eta=eta,
        static_residual=residual,
        checkpoint_gauge_aligned=aligned,
        P_dimension=operator.p_dimension,
        Q5_dimension=operator.s5_dimension,
        backend=device,
        device=operator.runtime.device_kind,
        operator_setup_seconds=setup_seconds,
        compiled_group_count=compiled.group_count,
        A_0=float(spectrum[len(spectrum) // 2]),
        integral_minus15_plus15=float(np.trapezoid(spectrum, x=omega)),
        relative_L2_S135_vs_S13=float(
            np.linalg.norm(spectrum - q13) / np.linalg.norm(q13)
        ),
        total_wall_seconds=perf_counter() - started,
    )
    (result_dir / "spectrum_N6_S135.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "event": "response_complete",
                "final_m": metadata["final_m"],
                "observable_converged": metadata["observable_converged"],
                "observable_change": metadata["observable_change_relative_l2"],
                "relative_L2_S135_vs_S13": metadata["relative_L2_S135_vs_S13"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", "--backend", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    main(args.output, args.device)
