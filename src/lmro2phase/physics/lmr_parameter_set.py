"""LMR half-cell base 파라미터 셋 생성.

halfcell_lmr_base.yaml 설정을 읽어 PyBaMM parameter set을 구성합니다.
2-phase FallbackA 전략 기준으로 effective OCP/D/R을 주입합니다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

# 기본 파라미터 값 (halfcell_lmr_base.yaml 없을 때 fallback)
_DEFAULTS = {
    "particle_radius_m": 5.0e-6,
    "D_s_m2_s": 1.0e-14,
    "active_material_volume_fraction": 0.665,
    "porosity": 0.3,
    "electrode_thickness_m": 75.0e-6,
    "contact_resistance_ohm_m2": 0.005,
    "c_s_max_mol_m3": 47500.0,
    "nominal_capacity_ah": 0.005,
}


def load_lmr_base_params(config_path: str | Path = "configs/halfcell_lmr_base.yaml") -> dict:
    """yaml에서 base 파라미터 로드."""
    cfg_path = Path(config_path)
    if cfg_path.exists():
        cfg = OmegaConf.load(str(cfg_path))
        return OmegaConf.to_container(cfg, resolve=True)
    log.warning(f"{config_path} 없음, 기본값 사용")
    return {}


def build_pybamm_halfcell_params(
    ocp_fn: Callable,                # effective OCP (FallbackA 또는 native)
    D_s: float = 1.0e-14,
    R_particle: float = 5.0e-6,
    contact_resistance: float = 0.005,
    capacity_scale: float = 1.0,
    c_s_max: float = 47500.0,
    c_s_0_fraction: float = 0.5,
    electrode_thickness: float = 75.0e-6,
    active_material_fraction: float = 0.665,
    config_path: str | Path = "configs/halfcell_lmr_base.yaml",
):
    """
    PyBaMM positive half-cell parameter set 생성.

    Returns:
        pybamm.ParameterValues
    """
    import pybamm

    param = pybamm.ParameterValues("Chen2020")

    # Li metal 상대 전극 설정
    param["Lithium counter electrode exchange-current density [A.m-2]"] = 1.0
    param["Lithium counter electrode conductivity [S.m-1]"] = 1e7
    param["Lithium counter electrode thickness [m]"] = 250e-6

    # Positive electrode 기하학
    param["Positive electrode thickness [m]"] = electrode_thickness
    param["Positive particle radius [m]"] = R_particle
    param["Positive electrode active material volume fraction"] = active_material_fraction
    param["Maximum concentration in positive electrode [mol.m3]"] = c_s_max

    # OCP
    param["Positive electrode OCP [V]"] = ocp_fn

    # Diffusivity
    param["Positive particle diffusivity [m2.s-1]"] = D_s

    # 용량
    nominal_cap = _DEFAULTS["nominal_capacity_ah"] * capacity_scale
    param["Nominal cell capacity [A.h]"] = nominal_cap

    # 접촉 저항
    try:
        param["Contact resistance [Ohm]"] = contact_resistance
    except Exception:
        pass  # 일부 PyBaMM 버전에서 없을 수 있음

    # 초기 stoichiometry
    param["Initial concentration in positive electrode [mol.m3]"] = (
        c_s_max * c_s_0_fraction
    )

    log.debug(f"파라미터 셋 생성: D_s={D_s:.2e}, R={R_particle:.2e}, "
              f"cap_scale={capacity_scale:.3f}")
    return param


def param_vector_to_kwargs(v: np.ndarray, param_names: list[str]) -> dict:
    """최적화 벡터를 PyBaMM 파라미터 dict로 변환."""
    return dict(zip(param_names, v.tolist()))
