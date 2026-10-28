"""S135 mode ladder for the paper beta=1 SIAM checkpoints."""

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

from .run_s135 import REPO


def output_for_beta(beta: float) -> Path:
    if beta == PARAMETERS["beta"]:
        return REPO / "siam_modes/results"
    raise ValueError("the paper mode ladder uses beta=1")


def main(beta: float, modes: int, device: str = "gpu") -> None:
    if modes not in range(1, 7):
        raise ValueError("modes must be N=1,...,6")
    started = perf_counter()
    output = output_for_beta(beta)
    configure_jax(output / "s135", device=device)
    saved = output / "cc_reference"
    fit = json.loads((saved / f"bath_N{modes}.json").read_text())
    static_record = json.loads((saved / f"cc_N{modes}.json").read_text())
    if float(fit["beta"]) != beta or float(static_record["beta"]) != beta:
        raise ValueError("saved bath or CCSD checkpoint has the wrong beta")
    with np.load(saved / f"bath_N{modes}.npz", allow_pickle=False) as bath:
        model = siam(bath["H"], bath["D"], bath["g"])
    options = dict(m_up=0.5, build_next_rank_masks=False)
    if beta == PARAMETERS["beta"]:
        options["reference_mode"] = "legacy_half_filled"
    cc = factorized_cc_problem(model, "CCSD", **options)
    static = ComponentStaticProblem.compile(cc)
    with np.load(saved / f"cc_N{modes}_latest.npz", allow_pickle=False) as checkpoint:
        amplitudes = np.asarray(checkpoint["amplitudes"], dtype=np.complex128)
    if amplitudes.shape != (cc.num_amplitudes,):
        raise ValueError("saved CCSD amplitudes have the wrong dimension")
    residual = float(
        np.linalg.norm(component_static_residual(static, amplitudes).retained)
    )
    aligned = False
    if residual > 1e-8:
        alignment = portable_checkpoint_amplitudes(static, amplitudes)
        amplitudes = alignment.amplitudes
        residual = float(
            np.linalg.norm(component_static_residual(static, amplitudes).retained)
        )
        aligned = bool(alignment.resolved)
    if residual > 1e-8:
        raise RuntimeError(
            f"N={modes} saved CCSD state fails its fresh residual gate: {residual:.3e}"
        )
    print(
        json.dumps(
            {
                "event": "static_gate",
                "beta": beta,
                "N": modes,
                "residual": residual,
                "gauge_aligned": aligned,
            }
        ),
        flush=True,
    )

    operator, sources, compiled, setup_seconds = _build_operator(
        static, amplitudes, (model.impurity_site,), device=device
    )
    device = getattr(getattr(operator, "device", None), "platform", device)
    q13_data = np.loadtxt(saved / f"spectrum_N{modes}.csv", delimiter=",", skiprows=1)
    omega = np.asarray(q13_data[:, 0], dtype=np.float64)
    q13 = np.asarray(q13_data[:, 3], dtype=np.float64)
    result_dir = output / "s135"
    result_dir.mkdir(parents=True, exist_ok=True)
    from c3l.green.checkpointed_arnoldi import eom_checkpoint_identity

    identity = eom_checkpoint_identity(compiled.context)
    root = result_dir / f"source_N{modes}_{device}"
    record = _run_source(
        operator, sources[0], sources, omega, "frequency", root,
        eta=PARAMETERS["eta"],
        provenance=f"siam-beta{beta:g}-n{modes}-s135:{identity}",
    )
    green = _source_final(record)[:, 0]
    spectrum = -green.imag / np.pi
    np.savetxt(
        result_dir / f"spectrum_N{modes}_S135.csv",
        np.column_stack((omega, green.real, green.imag, spectrum)),
        delimiter=",",
        header="omega,Re_G_up,Im_G_up,A_up",
        comments="",
    )
    metadata = _source_metadata(record)
    metadata.update(
        beta=beta,
        U=PARAMETERS["U"],
        N_lesser=modes,
        N_greater=modes,
        eta=PARAMETERS["eta"],
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
    (result_dir / f"spectrum_N{modes}_S135.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "event": "response_complete",
                "beta": beta,
                "N": modes,
                "final_m": metadata["final_m"],
                "observable_converged": metadata["observable_converged"],
                "observable_change": metadata["observable_change_relative_l2"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, choices=(PARAMETERS["beta"],), default=PARAMETERS["beta"])
    parser.add_argument("--modes", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    main(args.beta, args.modes, args.device)
