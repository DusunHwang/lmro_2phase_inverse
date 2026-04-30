"""Stage 2 synthetic generation용 OCP 변동 생성기."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .ocp_grid import OCPGrid, STO_GRID


def _gp_smooth_noise(sto: np.ndarray, amplitude: float,
                     length_scale: float, rng: np.random.Generator) -> np.ndarray:
    """RBF kernel로 근사한 GP-like smooth 노이즈."""
    n = len(sto)
    K = np.exp(-0.5 * ((sto[:, None] - sto[None, :]) / length_scale) ** 2)
    K += np.eye(n) * 1e-8  # jitter
    L = np.linalg.cholesky(K)
    noise = amplitude * (L @ rng.standard_normal(n))
    return noise


def perturb_ocp_grid(base: OCPGrid, rng: np.random.Generator,
                     mode: str = "random") -> OCPGrid:
    """
    base OCP grid에 물리적으로 타당한 변동을 추가합니다.

    mode:
        random     - 아래 중 무작위 선택
        gp         - GP smooth noise
        spline     - 랜덤 cubic spline 변동
        plateau    - 전압 plateau shift
        transition - phase transition 위치 이동
        shoulder   - 국소 shoulder 추가
        tanh       - tanh 항 추가
    """
    if mode == "random":
        mode = rng.choice(["gp", "spline", "plateau", "transition", "shoulder", "tanh"])

    v = base.voltage.copy()
    sto = base.sto

    if mode == "gp":
        amp = float(rng.uniform(0.005, 0.08))
        ls  = float(rng.uniform(0.05, 0.3))
        v += _gp_smooth_noise(sto, amp, ls, rng)

    elif mode == "spline":
        n_knots = int(rng.integers(3, 8))
        knot_x = np.sort(rng.uniform(0.05, 0.95, n_knots))
        knot_y = rng.uniform(-0.05, 0.05, n_knots)
        cs = CubicSpline(knot_x, knot_y, bc_type="not-a-knot", extrapolate=True)
        v += cs(sto)

    elif mode == "plateau":
        center = float(rng.uniform(0.2, 0.8))
        width  = float(rng.uniform(0.05, 0.2))
        shift  = float(rng.uniform(-0.1, 0.1))
        v += shift * np.exp(-0.5 * ((sto - center) / width) ** 2)

    elif mode == "transition":
        # 기존 transition 위치를 약간 이동
        center = float(rng.uniform(0.1, 0.9))
        new_center = center + float(rng.uniform(-0.1, 0.1))
        new_center = np.clip(new_center, 0.05, 0.95)
        shift = float(rng.uniform(-0.15, 0.15))
        v += shift * np.tanh((sto - new_center) / 0.05)

    elif mode == "shoulder":
        center = float(rng.uniform(0.1, 0.9))
        width  = float(rng.uniform(0.02, 0.08))
        height = float(rng.uniform(-0.08, 0.08))
        v += height * np.exp(-0.5 * ((sto - center) / width) ** 2)

    elif mode == "tanh":
        A = float(rng.uniform(-0.1, 0.1))
        c = float(rng.uniform(0.1, 0.9))
        w = float(rng.uniform(0.03, 0.15))
        v += A * np.tanh((sto - c) / w)

    v = np.clip(v, base.voltage_lower, base.voltage_upper)
    return OCPGrid(sto=sto.copy(), voltage=v,
                   voltage_lower=base.voltage_lower,
                   voltage_upper=base.voltage_upper)
