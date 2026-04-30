"""2-phase positive electrode 모델 팩토리.

FallbackA: 가중 혼합 OCP surrogate
  - PyBaMM single-phase half-cell 유지
  - effective OCP = frac_R3m * U_R3m(sto) + frac_C2m * U_C2m(sto)
  - 각 phase diffusion은 effective D_eff = frac * D_R3m + frac * D_C2m

FallbackB: 직접 ODE wrapper (향후 구현)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class TwoPhaseOCPParams:
    """2-phase OCP 파라미터 컨테이너."""
    frac_R3m: float   # rhombohedral phase 분율 (0~1)
    frac_C2m: float   # monoclinic phase 분율 (= 1 - frac_R3m)
    U_R3m: Callable   # stoichiometry → voltage (PyBaMM 호환)
    U_C2m: Callable
    D_R3m: float      # m^2/s
    D_C2m: float
    R_R3m: float      # m
    R_C2m: float

    def __post_init__(self):
        # 분율 합이 1에 가깝도록 normalize
        total = self.frac_R3m + self.frac_C2m
        if abs(total - 1.0) > 0.01:
            log.warning(f"Phase fraction 합={total:.4f} ≠ 1, normalize")
        self.frac_R3m = self.frac_R3m / total
        self.frac_C2m = self.frac_C2m / total


def make_effective_ocp(params: TwoPhaseOCPParams) -> Callable:
    """
    FallbackA: 가중 혼합 effective OCP 함수 반환.
    PyBaMM parameter set의 "Positive electrode OCP [V]" 에 직접 주입 가능.
    """
    import pybamm

    fR = params.frac_R3m
    fC = params.frac_C2m

    def effective_ocp(sto):
        # stoichiometry는 두 phase가 공유하는 양극 전체 기준
        return fR * params.U_R3m(sto) + fC * params.U_C2m(sto)

    return effective_ocp


def make_effective_diffusivity(params: TwoPhaseOCPParams) -> float:
    """
    FallbackA: 가중 조화 평균 effective 확산계수.
    병렬 경로 모델 기반 (직렬이면 조화 평균 사용).
    """
    # 가중 산술 평균 (병렬 diffusion 경로 근사)
    D_eff = params.frac_R3m * params.D_R3m + params.frac_C2m * params.D_C2m
    return D_eff


def make_effective_radius(params: TwoPhaseOCPParams) -> float:
    """가중 평균 effective 입자 반경."""
    R_eff = params.frac_R3m * params.R_R3m + params.frac_C2m * params.R_C2m
    return R_eff


def inject_fallback_a_params(pybamm_params, ocp_params: TwoPhaseOCPParams) -> None:
    """
    FallbackA 파라미터를 PyBaMM parameter set에 주입.
    single-phase half-cell 모델 기준.
    """
    pybamm_params["Positive electrode OCP [V]"] = make_effective_ocp(ocp_params)
    pybamm_params["Positive particle diffusivity [m2.s-1]"] = (
        make_effective_diffusivity(ocp_params)
    )
    pybamm_params["Positive particle radius [m]"] = make_effective_radius(ocp_params)
    log.debug("FallbackA OCP/D_eff/R_eff 주입 완료")
