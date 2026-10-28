"""Serialization and residual checks shared by paper calculations."""

from pathlib import Path
import json
import numpy as np


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def write_json(path, record):
    path = Path(path)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(jsonable(record), indent=2) + "\n")
    tmp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def fresh_residual(problem, component, amplitudes, platform="cpu"):
    # Use CPU for the independent reload check even after a GPU solve.
    from c3l.cc.component_tensor import component_static_residual

    retained = component_static_residual(component, amplitudes).retained
    return float(np.linalg.norm(problem.tilde.pack(retained, check=False)))
