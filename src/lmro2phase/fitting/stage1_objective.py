"""Stage 1 fitting 목적함수.

측정 전류 drive cycle로 PyBaMM을 구동하고,
V(Q), V(t), dV/dQ, dQ/dV, rest 잔차를 가중합산합니다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

from ..io.profile_schema import BatteryProfile, StepMode
from ..physics.ocp_tanh import TanhOCPParams, make_tanh_ocp_pybamm, tanh_ocp_second_derivative_penalty
from ..physics.positive_2phase_factory import TwoPhaseOCPParams, make_effective_ocp

log = logging.getLogger(__name__)


@dataclass
class Stage1Params:
    """Stage 1 최적화 대상 파라미터."""
    # R3m phase
    log10_D_R3m: float = -14.0
    log10_R_R3m: float = -5.3
    frac_R3m: float = 0.6
    tanh_R3m: Optional[TanhOCPParams] = None

    # C2m phase
    log10_D_C2m: float = -15.0
    log10_R_C2m: float = -5.5
    # frac_C2m = 1 - frac_R3m (constraint)
    tanh_C2m: Optional[TanhOCPParams] = None

    # shared
    log10_contact_resistance: float = -3.0
    capacity_scale: float = 1.0
    initial_stoichiometry_shift: float = 0.0

    @property
    def frac_C2m(self) -> float:
        return 1.0 - self.frac_R3m

    @property
    def D_R3m(self) -> float:
        return 10.0 ** self.log10_D_R3m

    @property
    def D_C2m(self) -> float:
        return 10.0 ** self.log10_D_C2m

    @property
    def R_R3m(self) -> float:
        return 10.0 ** self.log10_R_R3m

    @property
    def R_C2m(self) -> float:
        return 10.0 ** self.log10_R_C2m

    @property
    def contact_resistance(self) -> float:
        return 10.0 ** self.log10_contact_resistance


@dataclass
class LossWeights:
    w_v_q: float = 1.0
    w_v_t: float = 0.3
    w_dvdq: float = 2.0
    w_dqdv: float = 2.0
    w_rest: float = 0.5
    w_rate: float = 1.0
    w_prior: float = 0.1
    w_ocp_smooth: float = 0.5
    w_bounds: float = 10.0

    @classmethod
    def from_dict(cls, d: dict) -> "LossWeights":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _normalize_to_capacity_grid(time_s, voltage_v, current_a,
                                  n_grid: int = 512):
    """(time, V, I) → 정규화 용량 격자 위의 V, I 반환."""
    cap = np.cumsum(np.abs(current_a) * np.gradient(time_s) / 3600.0)
    if cap[-1] < 1e-8:
        return np.linspace(0, 1, n_grid), np.full(n_grid, np.nan), np.full(n_grid, np.nan)
    cap_norm = cap / cap[-1]
    q_grid = np.linspace(0, 1, n_grid)
    V_interp = interp1d(cap_norm, voltage_v, bounds_error=False, fill_value="extrapolate")(q_grid)
    I_interp = interp1d(cap_norm, current_a, bounds_error=False, fill_value="extrapolate")(q_grid)
    return q_grid, V_interp, I_interp


def _dVdQ(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.gradient(v, q)


def _dQdV(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    dv = np.gradient(v, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        dqdv = np.where(np.abs(dv) > 1e-6, 1.0 / dv, 0.0)
    return np.clip(dqdv, -200, 200)


def compute_loss(params: Stage1Params,
                  profile: BatteryProfile,
                  pybamm_model,
                  weights: LossWeights,
                  n_grid: int = 512) -> float:
    """
    Stage 1 전체 loss 계산.
    PyBaMM 시뮬레이션을 실행하고 측정값과 비교합니다.
    """
    from ..physics.lmr_parameter_set import build_pybamm_halfcell_params
    from ..physics.simulator import run_current_drive
    from ..physics.positive_2phase_factory import TwoPhaseOCPParams, make_effective_ocp

    # --- OCP 구성 ---
    ocp_R3m = make_tanh_ocp_pybamm(params.tanh_R3m) if params.tanh_R3m else None
    ocp_C2m = make_tanh_ocp_pybamm(params.tanh_C2m) if params.tanh_C2m else None

    if ocp_R3m is None or ocp_C2m is None:
        return 1e9

    two_phase = TwoPhaseOCPParams(
        frac_R3m=params.frac_R3m,
        frac_C2m=params.frac_C2m,
        U_R3m=ocp_R3m,
        U_C2m=ocp_C2m,
        D_R3m=params.D_R3m,
        D_C2m=params.D_C2m,
        R_R3m=params.R_R3m,
        R_C2m=params.R_C2m,
    )
    eff_ocp = make_effective_ocp(two_phase)

    # --- PyBaMM 파라미터 ---
    from ..physics.positive_2phase_factory import make_effective_diffusivity, make_effective_radius
    pv = build_pybamm_halfcell_params(
        ocp_fn=eff_ocp,
        D_s=make_effective_diffusivity(two_phase),
        R_particle=make_effective_radius(two_phase),
        contact_resistance=params.contact_resistance,
        capacity_scale=params.capacity_scale,
        c_s_0_fraction=0.5 + params.initial_stoichiometry_shift,
    )

    # --- 시뮬레이션 ---
    result = run_current_drive(pybamm_model, pv, profile.time_s, profile.current_a)
    if not result.ok:
        return 1e8

    t_sim = result.time_s
    v_sim = result.voltage_v
    i_sim = result.current_a

    t_exp = profile.time_s
    v_exp = profile.voltage_v
    i_exp = profile.current_a

    # --- V(t) 보간 비교 ---
    v_sim_t = interp1d(t_sim, v_sim, bounds_error=False, fill_value="extrapolate")(t_exp)
    loss_vt = float(np.mean((v_sim_t - v_exp) ** 2))

    # --- V(Q) 비교 ---
    _, v_sim_q, _ = _normalize_to_capacity_grid(t_sim, v_sim, i_sim, n_grid)
    _, v_exp_q, _ = _normalize_to_capacity_grid(t_exp, v_exp, i_exp, n_grid)
    valid = ~(np.isnan(v_sim_q) | np.isnan(v_exp_q))
    loss_vq = float(np.mean((v_sim_q[valid] - v_exp_q[valid]) ** 2)) if valid.any() else 1e6

    # --- dV/dQ, dQ/dV ---
    q_grid = np.linspace(0, 1, n_grid)
    dvdq_sim = _dVdQ(q_grid, v_sim_q)
    dvdq_exp = _dVdQ(q_grid, v_exp_q)
    dqdv_sim = _dQdV(q_grid, v_sim_q)
    dqdv_exp = _dQdV(q_grid, v_exp_q)
    loss_dvdq = float(np.mean((dvdq_sim[valid] - dvdq_exp[valid]) ** 2)) if valid.any() else 1e6
    loss_dqdv = float(np.mean((dqdv_sim[valid] - dqdv_exp[valid]) ** 2)) if valid.any() else 1e6

    # --- Rest 잔차 ---
    loss_rest = 0.0
    rest_segs = profile.get_segments_by_mode(StepMode.REST)
    for rseg in rest_segs:
        v_rest_sim = interp1d(t_sim, v_sim, bounds_error=False, fill_value="extrapolate")(rseg.time_s)
        loss_rest += float(np.mean((v_rest_sim - rseg.voltage_v) ** 2))
    if rest_segs:
        loss_rest /= len(rest_segs)

    # --- OCP smoothness 페널티 ---
    ocp_smooth = 0.0
    if params.tanh_R3m:
        ocp_smooth += tanh_ocp_second_derivative_penalty(params.tanh_R3m)
    if params.tanh_C2m:
        ocp_smooth += tanh_ocp_second_derivative_penalty(params.tanh_C2m)

    # --- Bounds 페널티 ---
    bounds_penalty = 0.0
    if not (0.05 < params.frac_R3m < 0.95):
        bounds_penalty += 1.0
    if not (-18 < params.log10_D_R3m < -12):
        bounds_penalty += 1.0
    if not (-18 < params.log10_D_C2m < -12):
        bounds_penalty += 1.0

    total = (
        weights.w_v_q * loss_vq
        + weights.w_v_t * loss_vt
        + weights.w_dvdq * loss_dvdq
        + weights.w_dqdv * loss_dqdv
        + weights.w_rest * loss_rest
        + weights.w_ocp_smooth * ocp_smooth
        + weights.w_bounds * bounds_penalty
    )
    return float(total)
