#!/usr/bin/env python3
"""Run a calculation with all relative data paths rooted in --work-dir."""

import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("module", help="Calculation module, e.g. siam_modes.example04")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    work = args.work_dir.resolve()
    if work == source or source in work.parents:
        parser.error("--work-dir must be outside the script directory")
    if not all(part.isidentifier() for part in args.module.split(".")):
        parser.error("module must be a dotted Python module name")
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    sys.dont_write_bytecode = True
    # The paper tolerances require double precision in JAX calculations.
    os.environ["JAX_ENABLE_X64"] = "true"
    # A private package alias also supports source directory names with spaces/hyphens.
    spec = importlib.util.spec_from_file_location(
        "_calculations",
        source / "__init__.py",
        submodule_search_locations=[str(source)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    sys.argv = [args.module, *args.arguments]
    runpy.run_module(f"{spec.name}.{args.module}", run_name="__main__")


if __name__ == "__main__":
    main()
