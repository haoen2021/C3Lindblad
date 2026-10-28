"""Paper SIAM, U=10 and beta=1; data are written under --work-dir."""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    # The paper launcher handles portable imports and the external data directory.
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Calculation options: --modes 1 2 ... 6, --fit-only, --output PATH.",
    )
    parser.add_argument("--work-dir", required=True)
    args, calculation_arguments = parser.parse_known_args()
    launcher = Path(__file__).resolve().parents[1] / "run.py"
    sys.argv = [str(launcher), "--work-dir", args.work_dir,
                "siam_modes.example04", *calculation_arguments]
    runpy.run_path(str(launcher), run_name="__main__")
