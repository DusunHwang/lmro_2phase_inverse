"""DFN 2-phase fitting + report wrapper."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _append_default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _append_default("--fit-model", "DFN")
    _append_default("--out-dir", "data/fit_results/DFN_2phase_dynamic_bounds_dqdv")
    runpy.run_path(str(HERE / "run_lmr_fit_and_report.py"), run_name="__main__")
