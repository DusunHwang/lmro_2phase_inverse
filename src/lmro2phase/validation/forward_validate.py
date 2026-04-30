"""Stage 4: 예측 파라미터로 forward validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def validate_forward(predicted_params: dict,
                      profile,
                      model_type: str = "SPMe") -> dict:
    """
    predicted_params (DL 추론 결과) → PyBaMM forward 시뮬레이션 → 잔차 계산.

    Returns:
        dict: rmse_v, max_err_v, residual 통계
    """
    from ..physics.lmr_parameter_set import build_pybamm_halfcell_params
    from ..physics.positive_2phase_factory import (
        TwoPhaseOCPParams, make_effective_ocp,
        make_effective_diffusivity, make_effective_radius
    )
    from ..physics.ocp_grid import OCPGrid, STO_GRID
    from ..physics.halfcell_model_factory import build_halfcell_model, ModelType
    from ..physics.simulator import run_current_drive
    from scipy.interpolate import interp1d

    # OCP 복원
    ocp_R3m_v = np.array(predicted_params["ocp_R3m_grid"])
    ocp_C2m_v = np.array(predicted_params["ocp_C2m_grid"])
    n = len(ocp_R3m_v)
    sto = np.linspace(0.005, 0.995, n)

    ocp_R3m_grid = OCPGrid(sto=sto, voltage=ocp_R3m_v)
    ocp_C2m_grid = OCPGrid(sto=sto, voltage=ocp_C2m_v)

    two_phase = TwoPhaseOCPParams(
        frac_R3m=float(predicted_params["frac_R3m"]),
        frac_C2m=float(predicted_params["frac_C2m"]),
        U_R3m=ocp_R3m_grid.to_pybamm_interpolant(),
        U_C2m=ocp_C2m_grid.to_pybamm_interpolant(),
        D_R3m=10.0 ** float(predicted_params["log10_D_R3m"]),
        D_C2m=10.0 ** float(predicted_params["log10_D_C2m"]),
        R_R3m=10.0 ** float(predicted_params["log10_R_R3m"]),
        R_C2m=10.0 ** float(predicted_params["log10_R_C2m"]),
    )

    pv = build_pybamm_halfcell_params(
        ocp_fn=make_effective_ocp(two_phase),
        D_s=make_effective_diffusivity(two_phase),
        R_particle=make_effective_radius(two_phase),
        contact_resistance=10.0 ** float(predicted_params["log10_contact_resistance"]),
        capacity_scale=float(predicted_params["capacity_scale"]),
        c_s_0_fraction=0.5 + float(predicted_params["initial_stoichiometry_shift"]),
    )

    model = build_halfcell_model(ModelType(model_type))
    result = run_current_drive(model, pv, profile.time_s, profile.current_a)

    if not result.ok:
        log.warning(f"Forward validation 시뮬레이션 실패: {result.error}")
        return {"ok": False, "error": result.error}

    # 잔차 계산
    t_exp = profile.time_s
    v_exp = profile.voltage_v
    v_sim_interp = interp1d(result.time_s, result.voltage_v,
                             bounds_error=False, fill_value="extrapolate")(t_exp)

    residual = v_sim_interp - v_exp
    return {
        "ok": True,
        "rmse_v": float(np.sqrt(np.mean(residual ** 2))),
        "mae_v": float(np.mean(np.abs(residual))),
        "max_err_v": float(np.max(np.abs(residual))),
        "time_s": result.time_s.tolist(),
        "voltage_sim": result.voltage_v.tolist(),
        "voltage_exp": v_exp.tolist(),
        "residual": residual.tolist(),
    }
