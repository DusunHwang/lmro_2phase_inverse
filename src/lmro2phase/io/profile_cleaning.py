"""BatteryProfile 데이터 정제 유틸리티."""

from __future__ import annotations

import logging

import numpy as np

from .profile_schema import BatteryProfile

log = logging.getLogger(__name__)


def clean_profile(profile: BatteryProfile,
                  voltage_min: float = 1.5,
                  voltage_max: float = 5.5,
                  current_max_abs: float = 100.0,
                  drop_zero_time_dup: bool = True) -> BatteryProfile:
    """물리적으로 비현실적인 데이터 포인트를 제거합니다."""
    n0 = len(profile)
    mask = np.ones(n0, dtype=bool)

    # 전압 범위
    mask &= (profile.voltage_v >= voltage_min) & (profile.voltage_v <= voltage_max)
    # 전류 절댓값
    mask &= np.abs(profile.current_a) <= current_max_abs
    # NaN 제거
    mask &= ~np.isnan(profile.voltage_v)
    mask &= ~np.isnan(profile.current_a)
    mask &= ~np.isnan(profile.time_s)

    # 시간 중복 제거 (time_s가 0 증분인 행)
    if drop_zero_time_dup and len(profile.time_s) > 1:
        dt = np.diff(profile.time_s, prepend=profile.time_s[0] - 1)
        mask &= dt > 0

    n1 = int(mask.sum())
    if n1 < n0:
        log.info(f"  정제: {n0} → {n1}포인트 ({n0 - n1}개 제거)")

    def apply(arr):
        return arr[mask] if arr is not None else None

    profile.time_s = profile.time_s[mask]
    profile.voltage_v = profile.voltage_v[mask]
    profile.current_a = profile.current_a[mask]
    profile.capacity_ah = apply(profile.capacity_ah)
    profile.cycle_index = apply(profile.cycle_index)
    profile.step_index = apply(profile.step_index)
    profile.mode = apply(profile.mode)
    profile.temperature_c = apply(profile.temperature_c)
    return profile


def shift_time_to_zero(time_s: np.ndarray) -> np.ndarray:
    """시간 배열을 0에서 시작하도록 shift."""
    return time_s - time_s[0]


def compute_capacity_from_current(time_s: np.ndarray,
                                  current_a: np.ndarray) -> np.ndarray:
    """전류 적분으로 용량(Ah) 계산 (사이클 내 상대 용량)."""
    dt = np.diff(time_s, prepend=time_s[0])
    dt[0] = 0.0
    return np.cumsum(np.abs(current_a) * dt / 3600.0)
