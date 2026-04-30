"""Stage 2/3용 자유형 OCP grid representation (256점).

stoichiometry 격자 위의 전압값을 직접 저장하고,
PyBaMM Interpolant로 변환하여 시뮬레이터에 주입합니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator


GRID_LENGTH = 256
STO_GRID = np.linspace(0.005, 0.995, GRID_LENGTH)


@dataclass
class OCPGrid:
    """자유형 OCP 격자."""
    sto: np.ndarray       # [GRID_LENGTH]
    voltage: np.ndarray   # [GRID_LENGTH]
    voltage_lower: float = 1.5
    voltage_upper: float = 5.2

    def __post_init__(self):
        self.sto = np.asarray(self.sto, dtype=float)
        self.voltage = np.clip(np.asarray(self.voltage, dtype=float),
                               self.voltage_lower, self.voltage_upper)

    @classmethod
    def from_tanh_params(cls, params, voltage_lower=1.5, voltage_upper=5.2) -> "OCPGrid":
        from .ocp_tanh import make_tanh_ocp_numpy
        fn = make_tanh_ocp_numpy(params)
        voltage = fn(STO_GRID)
        return cls(sto=STO_GRID.copy(), voltage=voltage,
                   voltage_lower=voltage_lower, voltage_upper=voltage_upper)

    def to_pybamm_interpolant(self):
        """PyBaMM Interpolant 함수 반환 (parameter set 주입용)."""
        import pybamm

        sto_arr = self.sto.copy()
        v_arr = self.voltage.copy()

        def ocp(sto):
            return pybamm.Interpolant(sto_arr, v_arr, sto, extrapolate=True)
        return ocp

    def smooth(self, sigma: float = 2.0) -> "OCPGrid":
        """Gaussian filter로 OCP grid 스무딩."""
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(self.voltage, sigma=sigma)
        smoothed = np.clip(smoothed, self.voltage_lower, self.voltage_upper)
        return OCPGrid(sto=self.sto.copy(), voltage=smoothed,
                       voltage_lower=self.voltage_lower,
                       voltage_upper=self.voltage_upper)

    def derivative(self) -> np.ndarray:
        """dU/dsto 수치 미분."""
        return np.gradient(self.voltage, self.sto)

    def second_derivative_penalty(self) -> float:
        d2 = np.gradient(np.gradient(self.voltage, self.sto), self.sto)
        return float(np.mean(d2 ** 2))

    def to_csv(self, path: str) -> None:
        import pandas as pd
        pd.DataFrame({"stoichiometry": self.sto, "voltage_v": self.voltage}).to_csv(
            path, index=False
        )

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "OCPGrid":
        import pandas as pd
        df = pd.read_csv(path)
        return cls(sto=df["stoichiometry"].to_numpy(),
                   voltage=df["voltage_v"].to_numpy(), **kwargs)
