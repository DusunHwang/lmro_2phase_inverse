"""Synthetic 시뮬레이션 품질 필터."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..physics.simulator import SimulationResult


@dataclass
class QualityConfig:
    voltage_min_v: float = 1.5
    voltage_max_v: float = 5.5
    capacity_min_fraction: float = 0.2
    capacity_max_fraction: float = 2.0
    max_voltage_oscillation: float = 0.1
    max_dqdv_spike: float = 50.0


def is_quality_ok(result: Optional[SimulationResult],
                   cfg: QualityConfig) -> bool:
    """시뮬레이션 결과가 품질 기준을 통과하는지 확인."""
    if result is None or not result.ok:
        return False

    v = result.voltage_v
    if v is None or len(v) < 10:
        return False

    # 전압 범위
    if v.min() < cfg.voltage_min_v or v.max() > cfg.voltage_max_v:
        return False

    # 전압 진동 감지 (연속 차분의 부호 변환 횟수)
    dv = np.diff(v)
    sign_changes = int(np.sum(np.diff(np.sign(dv)) != 0))
    if sign_changes > len(v) * 0.3:
        return False

    # 최대 전압 변동
    rolling_std = np.std(v[:min(10, len(v))])
    if rolling_std > cfg.max_voltage_oscillation:
        return False

    return True
