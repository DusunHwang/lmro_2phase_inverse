"""PyBaMM 시뮬레이터 공통 인터페이스.

측정 전류 drive cycle 모드와 PyBaMM Experiment 모드를 동일한 인터페이스로 실행합니다.
실패 케이스를 로깅하고 SimulationResult로 반환합니다.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class SimMode(str, Enum):
    CURRENT_DRIVE = "current_drive"   # 측정 전류 직접 입력
    EXPERIMENT = "experiment"          # PyBaMM Experiment string


@dataclass
class SimulationResult:
    ok: bool
    time_s: Optional[np.ndarray] = None
    voltage_v: Optional[np.ndarray] = None
    current_a: Optional[np.ndarray] = None
    capacity_ah: Optional[np.ndarray] = None
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def is_physical(self, v_min: float = 1.5, v_max: float = 5.5) -> bool:
        if not self.ok or self.voltage_v is None:
            return False
        return bool((self.voltage_v >= v_min).all() and (self.voltage_v <= v_max).all())


def run_current_drive(pybamm_model, pybamm_params,
                       time_s: np.ndarray,
                       current_a: np.ndarray,
                       initial_soc: float = 0.5,
                       t_eval: Optional[np.ndarray] = None,
                       solver_kwargs: Optional[dict] = None) -> SimulationResult:
    """측정 전류 profile로 PyBaMM 시뮬레이션."""
    import pybamm

    try:
        t_rel = time_s - time_s[0]
        current_interp = pybamm.Interpolant(t_rel, current_a, pybamm.t, extrapolate=False)
        pybamm_params["Current function [A]"] = current_interp

        sim = pybamm.Simulation(pybamm_model, parameter_values=pybamm_params)
        t_span = [0, float(t_rel[-1])]
        sol = sim.solve(t_span, initial_soc=initial_soc,
                        t_eval=t_eval, **(solver_kwargs or {}))

        t_out = sol["Time [s]"].entries
        v_out = sol["Voltage [V]"].entries
        i_out = sol["Current [A]"].entries
        q_out = np.trapz(np.abs(i_out), t_out / 3600.0) * np.ones_like(t_out)

        return SimulationResult(ok=True, time_s=t_out, voltage_v=v_out,
                                current_a=i_out, capacity_ah=q_out)
    except Exception as e:
        msg = traceback.format_exc()
        log.debug(f"current_drive 시뮬레이션 실패: {e}")
        return SimulationResult(ok=False, error=str(e))


def run_experiment(pybamm_model, pybamm_params,
                    experiment,
                    initial_soc: float = 0.5,
                    solver_kwargs: Optional[dict] = None) -> SimulationResult:
    """PyBaMM Experiment 시뮬레이션."""
    import pybamm

    try:
        sim = pybamm.Simulation(pybamm_model, parameter_values=pybamm_params,
                                 experiment=experiment)
        sol = sim.solve(initial_soc=initial_soc, **(solver_kwargs or {}))

        t_out = sol["Time [s]"].entries
        v_out = sol["Voltage [V]"].entries
        i_out = sol["Current [A]"].entries
        q_out = np.cumsum(np.abs(i_out) * np.gradient(t_out) / 3600.0)

        return SimulationResult(ok=True, time_s=t_out, voltage_v=v_out,
                                current_a=i_out, capacity_ah=q_out)
    except Exception as e:
        log.debug(f"experiment 시뮬레이션 실패: {e}")
        return SimulationResult(ok=False, error=str(e))
