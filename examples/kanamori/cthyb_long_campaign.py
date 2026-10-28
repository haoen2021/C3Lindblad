"""Run and curate the full-Kanamori CT-HYB campaign at mu=1."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
from pathlib import Path

import numpy as np


CHAINS_PER_BETA = 120
SEED_STRIDE = 10_007


def task_spec(task_id: int) -> tuple[float, int, int]:
    """Map a Slurm array index to beta, cycles, and an independent seed."""

    index = int(task_id)
    if not 0 <= index < 2 * CHAINS_PER_BETA:
        raise ValueError("task_id must be between 0 and 239")
    if index < CHAINS_PER_BETA:
        return 8.0, 180_000_000, 8_000_000 + SEED_STRIDE * index
    local_index = index - CHAINS_PER_BETA
    return 64.0, 75_000_000, 64_000_000 + SEED_STRIDE * local_index


def _sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(values.shape[0])
    return (
        np.std(values.real, axis=0, ddof=1) / scale,
        np.std(values.imag, axis=0, ddof=1) / scale,
    )


def _chain_directory(root: Path, beta: float, seed: int) -> Path:
    return root / f"beta{int(beta)}" / f"seed_{seed}"


def run_task(task_id: int, output_root: Path) -> dict[str, object]:
    """Run one array task and append the minimal Slurm provenance."""

    from .cthyb_half_filling import CHEMICAL_POTENTIAL, INTERACTION, run

    beta, cycles, seed = task_spec(task_id)
    destination = _chain_directory(Path(output_root), beta, seed)
    record_path = destination / "run.json"
    compact_path = destination / "compact.npz"
    if record_path.is_file() and compact_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("interaction") != INTERACTION or record.get("chemical_potential") != CHEMICAL_POTENTIAL:
            raise RuntimeError("Existing CT-HYB data use a different Hamiltonian; choose a new output root")
        if (
            record.get("solve_status") == 0
            and record.get("cycles") == cycles
            and record.get("seed") == seed
        ):
            return record

    record = run(beta=beta, cycles=cycles, seed=seed, output=destination)
    record.update(
        {
            "slurm_array_task_id": int(task_id),
            "slurm_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
            "compute_node": socket.gethostname(),
        }
    )
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _load_temperature(input_root: Path, beta: float) -> dict[str, np.ndarray]:
    from .cthyb_half_filling import CHEMICAL_POTENTIAL, INTERACTION

    tau = None
    iw = None
    tau_00 = []
    tau_01 = []
    iw_00 = []
    iw_01 = []
    signs = []
    orders = []
    autocorrelations = []
    wall_seconds = []
    seeds = []
    spin_defects = []
    orbital_defects = []
    reciprocity_defects = []

    beta_offset = 0 if beta == 8.0 else CHAINS_PER_BETA
    for local_index in range(CHAINS_PER_BETA):
        _, expected_cycles, seed = task_spec(beta_offset + local_index)
        directory = _chain_directory(input_root, beta, seed)
        record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if record.get("interaction") != INTERACTION or record.get("chemical_potential") != CHEMICAL_POTENTIAL:
            raise RuntimeError(f"CT-HYB chain uses a different Hamiltonian: {directory}")
        if (
            record.get("solve_status") != 0
            or record.get("cycles") != expected_cycles
            or record.get("seed") != seed
            or float(record.get("beta")) != beta
        ):
            raise ValueError(f"incomplete or inconsistent chain: {directory}")
        with np.load(directory / "compact.npz", allow_pickle=False) as saved:
            local_tau = np.asarray(saved["tau"], dtype=np.float64)
            local_iw = np.asarray(saved["iw"], dtype=np.complex128)
            up_tau = np.asarray(saved["G_tau_up"], dtype=np.complex128)
            dn_tau = np.asarray(saved["G_tau_dn"], dtype=np.complex128)
            up_iw = np.asarray(saved["G_iw_up"], dtype=np.complex128)
            dn_iw = np.asarray(saved["G_iw_dn"], dtype=np.complex128)
        if tau is None:
            tau = local_tau
            iw = local_iw
        elif not np.array_equal(tau, local_tau) or not np.array_equal(iw, local_iw):
            raise ValueError(f"mesh mismatch: {directory}")
        if up_tau.shape != (tau.size, 2, 2) or up_iw.shape != (iw.size, 2, 2):
            raise ValueError(f"Green-function shape mismatch: {directory}")
        if not all(
            np.all(np.isfinite(value)) for value in (up_tau, dn_tau, up_iw, dn_iw)
        ):
            raise ValueError(f"non-finite Green function: {directory}")

        spin_tau = 0.5 * (up_tau + dn_tau)
        spin_iw = 0.5 * (up_iw + dn_iw)
        tau_00.append(0.5 * (spin_tau[:, 0, 0] + spin_tau[:, 1, 1]))
        tau_01.append(0.5 * (spin_tau[:, 0, 1] + spin_tau[:, 1, 0]))
        iw_00.append(0.5 * (spin_iw[:, 0, 0] + spin_iw[:, 1, 1]))
        iw_01.append(0.5 * (spin_iw[:, 0, 1] + spin_iw[:, 1, 0]))
        signs.append(float(record["average_sign"]))
        orders.append(float(record["average_order"]))
        autocorrelations.append(float(record["auto_corr_time"]))
        wall_seconds.append(float(record["wall_seconds"]))
        seeds.append(seed)
        spin_defects.append(float(record["spin_defect"]))
        orbital_defects.append(float(record["orbital_diagonal_defect"]))
        reciprocity_defects.append(float(record["reciprocity_defect"]))

    return {
        "tau": np.asarray(tau),
        "iw": np.asarray(iw),
        "G_tau_00_chains": np.asarray(tau_00),
        "G_tau_01_chains": np.asarray(tau_01),
        "G_iw_00_chains": np.asarray(iw_00),
        "G_iw_01_chains": np.asarray(iw_01),
        "average_sign": np.asarray(signs),
        "average_order": np.asarray(orders),
        "auto_corr_time": np.asarray(autocorrelations),
        "wall_seconds": np.asarray(wall_seconds),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "spin_defect": np.asarray(spin_defects),
        "orbital_diagonal_defect": np.asarray(orbital_defects),
        "reciprocity_defect": np.asarray(reciprocity_defects),
    }


def _curated_arrays(
    beta: float, chains: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    from .cthyb_half_filling import CHEMICAL_POTENTIAL, INTERACTION

    result = dict(chains)
    result["beta"] = np.asarray(beta)
    result["chemical_potential"] = np.asarray(CHEMICAL_POTENTIAL)
    result["interaction"] = np.asarray(INTERACTION)
    result["tau_over_beta"] = chains["tau"] / beta
    result["omega_n"] = chains["iw"].imag
    for domain in ("tau", "iw"):
        for element in ("00", "01"):
            key = f"G_{domain}_{element}_chains"
            values = chains[key]
            real_sem, imag_sem = _sem(values)
            result[f"G_{domain}_{element}_mean"] = np.mean(values, axis=0)
            result[f"G_{domain}_{element}_sem_real"] = real_sem
            result[f"G_{domain}_{element}_sem_imag"] = imag_sem
    return result


def _write_csv(
    path: Path, coordinate: np.ndarray, arrays: dict[str, np.ndarray], domain: str
) -> None:
    columns = [coordinate]
    names = ["tau" if domain == "tau" else "omega_n"]
    for element in ("00", "01"):
        mean = arrays[f"G_{domain}_{element}_mean"]
        columns.extend(
            [
                mean.real,
                mean.imag,
                arrays[f"G_{domain}_{element}_sem_real"],
                arrays[f"G_{domain}_{element}_sem_imag"],
            ]
        )
        names.extend(
            [
                f"G{element}_real",
                f"G{element}_imag",
                f"G{element}_sem_real",
                f"G{element}_sem_imag",
            ]
        )
    np.savetxt(
        path,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(names),
        comments="",
    )


def aggregate_campaign(
    input_root: Path,
    output_dir: Path,
    delete_inputs: bool = False,
) -> dict[float, dict[str, float]]:
    """Aggregate chains into numerical arrays, CSV files, and diagnostics."""

    source = Path(input_root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summaries: dict[float, dict[str, float]] = {}
    for beta in (8.0, 64.0):
        arrays = _curated_arrays(beta, _load_temperature(source, beta))
        np.savez_compressed(
            destination / f"beta{int(beta)}_cthyb_plot_data.npz", **arrays
        )
        _write_csv(
            destination / f"beta{int(beta)}_G_tau.csv", arrays["tau"], arrays, "tau"
        )
        _write_csv(
            destination / f"beta{int(beta)}_G_iw.csv", arrays["omega_n"], arrays, "iw"
        )
        sign_sem = float(
            np.std(arrays["average_sign"], ddof=1) / np.sqrt(CHAINS_PER_BETA)
        )
        split = CHAINS_PER_BETA // 2
        split_00 = float(
            np.max(
                np.abs(
                    np.mean(arrays["G_tau_00_chains"][:split], axis=0)
                    - np.mean(arrays["G_tau_00_chains"][split:], axis=0)
                )
            )
        )
        split_01 = float(
            np.max(
                np.abs(
                    np.mean(arrays["G_tau_01_chains"][:split], axis=0)
                    - np.mean(arrays["G_tau_01_chains"][split:], axis=0)
                )
            )
        )
        summaries[beta] = {
            "sign_mean": float(np.mean(arrays["average_sign"])),
            "sign_sem": sign_sem,
            "autocorrelation_mean": float(np.mean(arrays["auto_corr_time"])),
            "autocorrelation_max": float(np.max(arrays["auto_corr_time"])),
            "actual_core_hours": float(np.sum(arrays["wall_seconds"]) / 3600.0),
            "split_half_G00_max_abs": split_00,
            "split_half_G01_max_abs": split_01,
            "G00_tau_sem_max": float(np.max(arrays["G_tau_00_sem_real"])),
            "G01_tau_sem_max": float(np.max(arrays["G_tau_01_sem_real"])),
            "spin_defect_max": float(np.max(arrays["spin_defect"])),
            "orbital_defect_max": float(np.max(arrays["orbital_diagonal_defect"])),
            "reciprocity_defect_max": float(np.max(arrays["reciprocity_defect"])),
        }

    (destination / "diagnostics.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    if delete_inputs:
        if source.resolve().name != "raw":
            raise ValueError("refusing to delete an input directory not named 'raw'")
        for beta in (8, 64):
            with np.load(
                destination / f"beta{beta}_cthyb_plot_data.npz",
                allow_pickle=False,
            ) as saved:
                if saved["G_tau_00_chains"].shape[0] != CHAINS_PER_BETA:
                    raise ValueError("curated output validation failed")
        shutil.rmtree(source)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    task_parser = subparsers.add_parser("run-task")
    task_parser.add_argument("--task-id", type=int, required=True)
    task_parser.add_argument("--output-root", type=Path, required=True)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--input-root", type=Path, required=True)
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)
    aggregate_parser.add_argument("--delete-inputs", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "run-task":
        print(json.dumps(run_task(arguments.task_id, arguments.output_root), indent=2))
    else:
        print(
            json.dumps(
                aggregate_campaign(
                    arguments.input_root,
                    arguments.output_dir,
                    delete_inputs=arguments.delete_inputs,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
