"""Thin wrappers around the stage scripts.

The numbered scripts remain the canonical stage definitions for now. These
entry points make editable installs easier to use while keeping behavior
identical to running ``python scripts/<stage>.py`` directly.
"""

from __future__ import annotations

import runpy
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _ROOT / "scripts"


def _run_script(name: str) -> None:
    runpy.run_path(str(_SCRIPTS / name), run_name="__main__")


def smoke() -> None:
    _run_script("00_smoke_test_pybamm_halfcell.py")


def parse_toyo() -> None:
    _run_script("01_parse_toyo_ascii.py")


def fit_ocp() -> None:
    _run_script("02_fit_tanh_ocp.py")


def generate_synthetic() -> None:
    _run_script("03_generate_synthetic_dataset.py")


def train_inverse() -> None:
    _run_script("04_train_inverse_model.py")


def infer_profile() -> None:
    _run_script("05_infer_lmr_profile.py")


def forward_validate() -> None:
    _run_script("06_forward_validate.py")


def generate_report() -> None:
    _run_script("07_generate_report.py")
