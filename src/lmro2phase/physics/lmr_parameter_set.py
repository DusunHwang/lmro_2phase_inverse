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


FARADAY = 96485.0  # C/mol


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

    Electrode height/width are set so the electrode area matches the nominal
    capacity; this ensures the applied current (in A) produces the correct
    stoichiometry swing over the experiment.

    Returns:
        pybamm.ParameterValues
    """
    import pybamm

    param = pybamm.ParameterValues("Chen2020")

    # Li metal 상대 전극 설정 (PyBaMM 26.x 파라미터명)
    param["Exchange-current density for lithium metal electrode [A.m-2]"] = 1.0

    # Positive electrode 기하학
    param["Positive electrode thickness [m]"] = electrode_thickness
    param["Positive particle radius [m]"] = R_particle
    param["Positive electrode active material volume fraction"] = active_material_fraction
    param["Maximum concentration in positive electrode [mol.m-3]"] = c_s_max

    # OCP
    param["Positive electrode OCP [V]"] = ocp_fn

    # Diffusivity
    param["Positive particle diffusivity [m2.s-1]"] = D_s

    # 용량 및 전극 면적 스케일링
    # 실제 전극 면적을 nominal capacity에 맞게 설정해야 전류가 올바른
    # stoichiometry 변화를 만들어냄.
    # area = nominal_cap_C / (c_s_max * thickness * active_frac * F)
    nominal_cap = _DEFAULTS["nominal_capacity_ah"] * capacity_scale
    nominal_cap_c = nominal_cap * 3600.0  # Ah → C
    electrode_area = nominal_cap_c / (
        c_s_max * electrode_thickness * active_material_fraction * FARADAY
    )
    electrode_side = float(np.sqrt(electrode_area))
    param["Electrode height [m]"] = electrode_side
    param["Electrode width [m]"] = electrode_side
    param["Nominal cell capacity [A.h]"] = nominal_cap

    # 전압 범위 — LMR half-cell 범위로 설정 (Chen2020 4.2V 상한을 4.6V로 확장)
    param["Open-circuit voltage at 0% SOC [V]"] = 2.5
    param["Open-circuit voltage at 100% SOC [V]"] = 4.6
    # 시뮬레이션 termination event 기준 전압 (LMR 범위 커버)
    param["Upper voltage cut-off [V]"] = 4.8
    param["Lower voltage cut-off [V]"] = 2.2

    # 접촉 저항
    try:
        param["Contact resistance [Ohm]"] = contact_resistance
    except Exception:
        pass  # 일부 PyBaMM 버전에서 없을 수 있음

    # 초기 stoichiometry (c_s_0_fraction: 0=fully delithiated/high V, 1=fully lithiated/low V)
    param["Initial concentration in positive electrode [mol.m-3]"] = (
        c_s_max * c_s_0_fraction
    )

    log.debug(f"파라미터 셋 생성: D_s={D_s:.2e}, R={R_particle:.2e}, "
              f"cap_scale={capacity_scale:.3f}, electrode_area={electrode_area:.3e}m²")
    return param


def param_vector_to_kwargs(v: np.ndarray, param_names: list[str]) -> dict:
    """최적화 벡터를 PyBaMM 파라미터 dict로 변환."""
    return dict(zip(param_names, v.tolist()))
