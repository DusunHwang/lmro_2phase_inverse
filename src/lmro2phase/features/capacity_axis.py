"""용량 정규화 및 격자 보간."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


FEATURE_GRID = 512   # 정규화 격자 길이


def normalize_to_capacity_grid(time_s: np.ndarray,
                                 voltage_v: np.ndarray,
                                 current_a: np.ndarray,
                                 n_grid: int = FEATURE_GRID
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    시간 → 정규화 용량 축으로 변환.
    Returns: (q_norm [n_grid], V_on_grid [n_grid], I_on_grid [n_grid])
    """
    cap = np.cumsum(np.abs(current_a) * np.gradient(time_s) / 3600.0)
    if cap[-1] < 1e-10:
        q = np.linspace(0, 1, n_grid)
        return q, np.full(n_grid, np.nan), np.full(n_grid, np.nan)

    cap_norm = cap / cap[-1]
    q_grid = np.linspace(0, 1, n_grid)

    v_grid = interp1d(cap_norm, voltage_v, bounds_error=False, fill_value=np.nan)(q_grid)
    i_grid = interp1d(cap_norm, current_a, bounds_error=False, fill_value=0.0)(q_grid)
    return q_grid, v_grid, i_grid
