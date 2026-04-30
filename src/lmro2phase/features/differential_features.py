"""dV/dQ, dQ/dV 특징 추출."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def compute_dvdq(q: np.ndarray, v: np.ndarray,
                  smooth: bool = True, window: int = 15, poly: int = 3) -> np.ndarray:
    if smooth:
        v = savgol_filter(v, window_length=min(window, len(v) - (len(v) % 2 == 0)),
                           polyorder=poly, mode="nearest")
    dv = np.gradient(v, q)
    return dv


def compute_dqdv(q: np.ndarray, v: np.ndarray,
                  smooth: bool = True, clip: float = 200.0, **kwargs) -> np.ndarray:
    dv = compute_dvdq(q, v, smooth=smooth, **kwargs)
    with np.errstate(divide="ignore", invalid="ignore"):
        dqdv = np.where(np.abs(dv) > 1e-6, 1.0 / dv, 0.0)
    return np.clip(dqdv, -clip, clip)


def build_feature_tensor(q_grid: np.ndarray,
                          v_chg: np.ndarray, i_chg: np.ndarray,
                          v_dchg: np.ndarray, i_dchg: np.ndarray,
                          rest_relaxation: np.ndarray | None = None
                          ) -> np.ndarray:
    """
    [channels, grid_length] feature tensor 생성.

    채널 순서:
        0: V_charge(Q)
        1: I_charge(Q)
        2: dV/dQ_charge
        3: dQ/dV_charge
        4: V_discharge(Q)
        5: I_discharge(Q)
        6: dV/dQ_discharge
        7: dQ/dV_discharge
        8: rest_relaxation (없으면 0)
        9: valid mask (NaN이면 0)
    """
    nan_mask = (
        ~np.isnan(v_chg) & ~np.isnan(v_dchg)
    ).astype(float)

    # NaN을 0으로 채움
    def _fill(arr):
        return np.where(np.isnan(arr), 0.0, arr)

    dvdq_chg  = _fill(compute_dvdq(q_grid, _fill(v_chg)))
    dqdv_chg  = _fill(compute_dqdv(q_grid, _fill(v_chg)))
    dvdq_dchg = _fill(compute_dvdq(q_grid, _fill(v_dchg)))
    dqdv_dchg = _fill(compute_dqdv(q_grid, _fill(v_dchg)))

    rest = rest_relaxation if rest_relaxation is not None else np.zeros_like(q_grid)

    tensor = np.stack([
        _fill(v_chg), _fill(i_chg),
        dvdq_chg, dqdv_chg,
        _fill(v_dchg), _fill(i_dchg),
        dvdq_dchg, dqdv_dchg,
        rest,
        nan_mask,
    ], axis=0)   # [10, L]

    return tensor.astype(np.float32)
