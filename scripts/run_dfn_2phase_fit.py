"""DFN 2상 역추정 실행 wrapper.

run_lmr_2phase_fit.py의 공통 fitting 엔진을 사용하되,
fit model을 PyBaMM DFN으로 고정한다.
"""
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
    _append_default("--out-dir", "data/fit_results/DFN_2phase_병렬_optun_dqdv")
    runpy.run_path(str(HERE / "run_lmr_2phase_fit.py"), run_name="__main__")
