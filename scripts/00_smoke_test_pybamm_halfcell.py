"""
Stage 0: PyBaMM positive half-cell smoke test

실행:
    cd lmro_2phase_inverse
    python scripts/00_smoke_test_pybamm_halfcell.py
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPORT_PATH = ROOT / "data" / "reports" / "smoke_test_report.json"


def test_pybamm_version() -> dict:
    import pybamm
    v = pybamm.__version__
    log.info(f"[1] PyBaMM 버전: {v}")
    return {"pybamm_version": v, "ok": True}


def test_build_single_phase_models() -> dict:
    """SPM / SPMe / DFN single-phase positive half-cell 빌드."""
    import pybamm

    results = {}
    for name in ["SPM", "SPMe", "DFN"]:
        try:
            opts = {"working electrode": "positive"}
            model = getattr(pybamm.lithium_ion, name)(opts)
            results[name] = "built"
            log.info(f"[2] {name} single-phase half-cell: OK")
        except Exception as e:
            results[name] = f"FAILED: {e}"
            log.warning(f"[2] {name} single-phase half-cell: FAILED → {e}")
    return results


def test_build_two_phase_model() -> dict:
    """positive 2-phase half-cell 빌드 시도."""
    import pybamm

    result = {}
    try:
        opts = {
            "working electrode": "positive",
            "particle phases": ("1", "2"),
            "particle": ("Fickian diffusion", "Fickian diffusion"),
            "open-circuit potential": ("single", "single"),
        }
        model = pybamm.lithium_ion.SPMe(opts)
        result["two_phase_build"] = "ok"
        log.info("[3] positive 2-phase SPMe 빌드: OK")
    except Exception as e:
        result["two_phase_build"] = f"FAILED: {e}"
        log.warning(f"[3] positive 2-phase SPMe 빌드: FAILED → {e}")
        log.warning("    → FallbackA (weighted mixture OCP) 전략으로 진행합니다.")

    # Primary/Secondary 파라미터 키 탐색
    try:
        from pybamm import ParameterValues
        pv = ParameterValues("Chen2020")
        relevant_keys = [
            k for k in pv.keys()
            if any(kw.lower() in k.lower()
                   for kw in ["Primary", "Secondary", "Positive", "open-circuit", "OCP"])
        ]
        result["relevant_param_keys"] = relevant_keys[:20]
        log.info(f"[3] 관련 파라미터 키 예시: {relevant_keys[:5]}")
    except Exception as e:
        result["param_key_search"] = f"FAILED: {e}"

    return result


def _make_simple_halfcell_params():
    """smoke test용 최소 positive half-cell 파라미터 반환."""
    import pybamm

    param = pybamm.ParameterValues("Chen2020")
    # Li metal counter electrode 최소 설정 (PyBaMM 26.x 파라미터명)
    param["Exchange-current density for lithium metal electrode [A.m-2]"] = 1.0
    return param


def test_tanh_ocp_injection() -> dict:
    """tanh OCP 함수를 PyBaMM에 주입하는 테스트."""
    import pybamm

    result = {}
    try:
        def tanh_ocp(sto):
            return (
                3.5
                + 0.5 * pybamm.tanh((sto - 0.5) / 0.1)
                + 0.2 * pybamm.tanh((sto - 0.3) / 0.08)
            )

        # PyBaMM symbolic 표현식으로 호출 테스트
        sto_test = pybamm.Scalar(0.5)
        val = tanh_ocp(sto_test)
        result["tanh_ocp"] = "ok"
        result["test_value_type"] = type(val).__name__
        log.info(f"[4] tanh OCP 주입: OK (type={type(val).__name__})")
    except Exception as e:
        result["tanh_ocp"] = f"FAILED: {e}"
        log.warning(f"[4] tanh OCP 주입: FAILED → {e}")
    return result


def test_current_drive_simulation() -> dict:
    """측정 전류 drive cycle로 single-phase half-cell 시뮬레이션."""
    import pybamm
    import numpy as np

    result = {}
    try:
        # 간단한 전류 프로파일 (CC + rest)
        t = np.array([0, 100, 200, 300, 400])
        i = np.array([0.5e-3, 0.5e-3, 0.0, 0.0, -0.5e-3])  # A

        current_fn = pybamm.Interpolant(t, i, pybamm.t, extrapolate=False)

        opts = {"working electrode": "positive"}
        model = pybamm.lithium_ion.SPMe(opts)
        param = _make_simple_halfcell_params()
        param["Current function [A]"] = current_fn
        param["Nominal cell capacity [A.h]"] = 0.005

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 350], initial_soc=0.5)
        t_out = sol["Time [s]"].entries
        v_out = sol["Voltage [V]"].entries

        result["current_drive"] = "ok"
        result["n_time_points"] = int(len(t_out))
        result["voltage_range"] = [float(v_out.min()), float(v_out.max())]
        log.info(f"[5] 전류 drive 시뮬레이션: OK ({len(t_out)}포인트, "
                 f"V=[{v_out.min():.3f}, {v_out.max():.3f}] V)")
    except Exception as e:
        result["current_drive"] = f"FAILED: {e}"
        log.warning(f"[5] 전류 drive 시뮬레이션: FAILED → {e}")
    return result


def test_experiment_cc_cv_rest() -> dict:
    """PyBaMM Experiment CC-CV-rest 시뮬레이션."""
    import pybamm

    result = {}
    try:
        opts = {"working electrode": "positive"}
        model = pybamm.lithium_ion.SPMe(opts)
        param = _make_simple_halfcell_params()
        param["Nominal cell capacity [A.h]"] = 0.005

        experiment = pybamm.Experiment([
            "Charge at C/20 until 4.5 V",
            "Hold at 4.5 V until C/50",
            "Rest for 10 minutes",
            "Discharge at C/20 until 2.5 V",
        ])
        sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
        sol = sim.solve(initial_soc=0.2)

        cycles = len(sol.cycles)
        result["cc_cv_rest_experiment"] = "ok"
        result["n_cycles_solved"] = cycles
        log.info(f"[6] CC-CV-rest Experiment: OK ({cycles}사이클)")
    except Exception as e:
        result["cc_cv_rest_experiment"] = f"FAILED: {e}"
        log.warning(f"[6] CC-CV-rest Experiment: FAILED → {e}")
    return result


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {}

    log.info("=" * 60)
    log.info("Stage 0: PyBaMM Positive Half-Cell Smoke Test")
    log.info("=" * 60)

    report["pybamm_version"] = test_pybamm_version()
    report["single_phase_models"] = test_build_single_phase_models()
    report["two_phase"] = test_build_two_phase_model()
    report["tanh_ocp"] = test_tanh_ocp_injection()
    report["current_drive"] = test_current_drive_simulation()
    report["experiment"] = test_experiment_cc_cv_rest()

    # 전략 결정
    two_phase_ok = report["two_phase"].get("two_phase_build") == "ok"
    report["selected_strategy"] = "native" if two_phase_ok else "fallback_a"

    log.info("=" * 60)
    log.info(f"선택된 2-phase 전략: {report['selected_strategy']}")
    log.info(f"리포트 저장: {REPORT_PATH}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info("Smoke test 완료.")


if __name__ == "__main__":
    main()
