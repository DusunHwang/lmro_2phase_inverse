"""PyBaMM Experiment 생성 및 측정 전류 drive cycle 변환."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..io.profile_schema import BatteryProfile, ProfileSegment, StepMode

log = logging.getLogger(__name__)


def make_current_interpolant(profile: BatteryProfile, segment: Optional[ProfileSegment] = None):
    """
    BatteryProfile (또는 특정 세그먼트)의 측정 전류를 PyBaMM Interpolant로 변환.

    - 시간은 0에서 시작하도록 shift
    - rest 구간 전류 노이즈는 이미 profile_cleaning에서 0으로 처리됨
    """
    import pybamm

    if segment is not None:
        t = segment.time_s
        i = segment.current_a
    else:
        t = profile.time_s
        i = profile.current_a

    t = t - t[0]  # 0 기준 shift

    return pybamm.Interpolant(t, i, pybamm.t, extrapolate=False)


def build_experiment_from_profile(profile: BatteryProfile,
                                   cycle: Optional[int] = None,
                                   voltage_upper: float = 4.8,
                                   voltage_lower: float = 2.0,
                                   cv_cutoff_c_rate: float = 0.05) -> object:
    """
    BatteryProfile의 세그먼트를 PyBaMM Experiment string으로 변환.

    synthetic dataset 생성 시 사용합니다.
    """
    import pybamm

    if cycle is not None:
        segs = [s for s in profile.segments if s.cycle_index == cycle]
    else:
        segs = profile.segments

    steps = []
    for seg in segs:
        mean_i = abs(seg.mean_current_a)
        nominal_cap = max(abs(seg.delta_capacity_ah), 1e-6)
        c_rate = mean_i / (nominal_cap / (seg.duration_s / 3600.0 + 1e-6) + 1e-10)
        c_str = f"C/{max(1, int(round(1.0 / (c_rate + 1e-10))))}" if c_rate < 0.9 else f"{c_rate:.2f}C"

        if seg.mode == StepMode.CC_CHARGE:
            steps.append(f"Charge at {c_str} until {voltage_upper} V")
        elif seg.mode == StepMode.CV_CHARGE:
            steps.append(f"Hold at {voltage_upper} V until {cv_cutoff_c_rate}C")
        elif seg.mode == StepMode.CC_DISCHARGE:
            steps.append(f"Discharge at {c_str} until {voltage_lower} V")
        elif seg.mode == StepMode.CV_DISCHARGE:
            steps.append(f"Hold at {voltage_lower} V until {cv_cutoff_c_rate}C")
        elif seg.mode == StepMode.REST:
            rest_min = max(1, int(seg.duration_s / 60))
            steps.append(f"Rest for {rest_min} minutes")
        else:
            # DYNAMIC / UNKNOWN은 해당 구간 무시
            log.debug(f"  세그먼트 mode={seg.mode} → skip")

    if not steps:
        log.warning("Experiment step이 없습니다. protocol 분리를 확인하세요.")
        return None

    log.debug(f"  Experiment steps: {steps}")
    return pybamm.Experiment(steps)


def build_standard_experiment(voltage_upper: float = 4.8,
                               voltage_lower: float = 2.0,
                               c_rates: list[float] = (0.05, 0.1, 0.2),
                               rest_min: int = 30) -> object:
    """표준 CC-CV-rest-multirate Experiment 생성."""
    import pybamm

    steps = []
    for cr in c_rates:
        c_str = f"C/{int(round(1.0/cr))}" if cr < 0.5 else f"{cr:.1f}C"
        steps += [
            f"Charge at {c_str} until {voltage_upper} V",
            f"Hold at {voltage_upper} V until C/50",
            f"Rest for {rest_min} minutes",
            f"Discharge at {c_str} until {voltage_lower} V",
            f"Rest for {rest_min} minutes",
        ]
    return pybamm.Experiment(steps)
