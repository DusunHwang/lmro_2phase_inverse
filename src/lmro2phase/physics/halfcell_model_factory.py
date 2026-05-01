"""PyBaMM positive half-cell 모델 팩토리.

2-phase positive electrode 지원 여부를 smoke test에서 판단하고,
실패 시 fallback 전략을 자동 선택합니다.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

log = logging.getLogger(__name__)


class ModelType(str, Enum):
    SPM = "SPM"
    SPMe = "SPMe"
    DFN = "DFN"


class TwoPhaseStrategy(str, Enum):
    NATIVE = "native"          # PyBaMM particle_phases 옵션 사용
    FALLBACK_A = "fallback_a"  # 가중 혼합 OCP surrogate
    FALLBACK_B = "fallback_b"  # 직접 diffusion ODE wrapper


def _pybamm_halfcell_options(two_phase: bool = False,
                              strategy: TwoPhaseStrategy = TwoPhaseStrategy.NATIVE) -> dict:
    opts: dict[str, Any] = {
        "working electrode": "positive",
    }
    if two_phase and strategy == TwoPhaseStrategy.NATIVE:
        # (negative_phases, positive_phases)
        # half-cell에서 negative는 Li metal이므로 "1"
        opts["particle phases"] = ("1", "2")
        opts["particle"] = ("Fickian diffusion", "Fickian diffusion")
        opts["open-circuit potential"] = ("single", "single")
    return opts


def build_halfcell_model(model_type: ModelType | str,
                          two_phase: bool = False,
                          strategy: TwoPhaseStrategy = TwoPhaseStrategy.NATIVE):
    """PyBaMM half-cell 모델 객체 반환."""
    import pybamm

    mtype = ModelType(model_type)
    opts = _pybamm_halfcell_options(two_phase, strategy)

    log.debug(f"모델 생성: {mtype}, two_phase={two_phase}, strategy={strategy}, opts={opts}")

    if mtype == ModelType.SPM:
        return pybamm.lithium_ion.SPM(opts)
    if mtype == ModelType.SPMe:
        return pybamm.lithium_ion.SPMe(opts)
    if mtype == ModelType.DFN:
        return pybamm.lithium_ion.DFN(opts)
    raise ValueError(f"Unknown model type: {mtype}")


def probe_two_phase_support() -> TwoPhaseStrategy:
    """
    positive 2-phase half-cell이 현재 PyBaMM에서 바로 동작하는지 확인하고
    사용할 TwoPhaseStrategy를 반환합니다.
    """
    import pybamm

    log.info("PyBaMM positive 2-phase half-cell 지원 여부 확인 중...")
    try:
        model = build_halfcell_model(ModelType.SPMe, two_phase=True,
                                     strategy=TwoPhaseStrategy.NATIVE)
        # 기본 파라미터로 매우 짧은 시뮬레이션 시도
        param = pybamm.ParameterValues("Chen2020")
        # half-cell용 수정 (positive working electrode)
        _patch_halfcell_params(param)

        sim = pybamm.Simulation(model, parameter_values=param)
        sim.solve([0, 10], initial_soc=0.5)

        log.info("  → native 2-phase 지원 확인 (TwoPhaseStrategy.NATIVE)")
        return TwoPhaseStrategy.NATIVE

    except Exception as e:
        log.warning(f"  → native 2-phase 실패: {e}")
        log.warning("  → FallbackA (가중 혼합 OCP surrogate) 사용")
        return TwoPhaseStrategy.FALLBACK_A


def _patch_halfcell_params(param) -> None:
    """Chen2020 파라미터를 positive half-cell용으로 최소 패치."""
    param["Exchange-current density for lithium metal electrode [A.m-2]"] = 1.0
