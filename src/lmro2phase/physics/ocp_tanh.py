"""Stage 1용 tanh basis OCP representation.

U_phase(x) = b0 + b1*x + Σ Ai * tanh((x - ci) / wi)

PyBaMM symbolic expression 안에서 pybamm.tanh를 사용합니다.
numpy 도메인에서도 동일한 수식을 평가할 수 있도록 양쪽 구현을 제공합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TanhOCPParams:
    """tanh basis OCP 파라미터."""
    b0: float                      # 기준 전압 (V)
    b1: float                      # 선형 기울기
    amps: np.ndarray               # tanh 항 진폭 배열  [n_terms]
    centers: np.ndarray            # tanh 항 중심 (0~1) [n_terms]
    widths: np.ndarray             # tanh 항 너비 (>0)  [n_terms]
    voltage_lower: float = 1.5
    voltage_upper: float = 5.2

    def __post_init__(self):
        self.amps = np.asarray(self.amps, dtype=float)
        self.centers = np.asarray(self.centers, dtype=float)
        self.widths = np.asarray(self.widths, dtype=float)

    @property
    def n_terms(self) -> int:
        return len(self.amps)

    def to_vector(self) -> np.ndarray:
        """최적화용 1D 벡터 변환 [b0, b1, A0..An, c0..cn, w0..wn]."""
        return np.concatenate([[self.b0, self.b1], self.amps, self.centers, self.widths])

    @classmethod
    def from_vector(cls, v: np.ndarray, n_terms: int,
                    voltage_lower: float = 1.5,
                    voltage_upper: float = 5.2) -> "TanhOCPParams":
        b0, b1 = float(v[0]), float(v[1])
        amps    = v[2:2 + n_terms]
        centers = v[2 + n_terms:2 + 2 * n_terms]
        widths  = np.abs(v[2 + 2 * n_terms:2 + 3 * n_terms])  # 항상 양수
        return cls(b0=b0, b1=b1, amps=amps, centers=centers, widths=widths,
                   voltage_lower=voltage_lower, voltage_upper=voltage_upper)

    def n_params(self) -> int:
        return 2 + 3 * self.n_terms

    @classmethod
    def default_init(cls, n_terms: int = 5,
                     voltage_center: float = 3.8,
                     voltage_lower: float = 1.5,
                     voltage_upper: float = 5.2) -> "TanhOCPParams":
        """LMR 양극 기준 초기값 생성."""
        centers = np.linspace(0.15, 0.85, n_terms)
        return cls(
            b0=voltage_center,
            b1=-1.0,
            amps=np.full(n_terms, 0.2),
            centers=centers,
            widths=np.full(n_terms, 0.1),
            voltage_lower=voltage_lower,
            voltage_upper=voltage_upper,
        )


def make_tanh_ocp_numpy(params: TanhOCPParams):
    """numpy 배열을 받아 전압을 반환하는 함수 생성."""
    def ocp(sto: np.ndarray) -> np.ndarray:
        y = params.b0 + params.b1 * sto
        for A, c, w in zip(params.amps, params.centers, params.widths):
            y = y + A * np.tanh((sto - c) / (w + 1e-10))
        return np.clip(y, params.voltage_lower, params.voltage_upper)
    return ocp


def make_tanh_ocp_pybamm(params: TanhOCPParams):
    """PyBaMM symbolic expression을 반환하는 함수 생성.
    PyBaMM parameter set에 직접 주입 가능.
    """
    import pybamm

    b0  = float(params.b0)
    b1  = float(params.b1)
    amps    = [float(a) for a in params.amps]
    centers = [float(c) for c in params.centers]
    widths  = [max(float(w), 1e-10) for w in params.widths]

    def ocp(sto):
        y = b0 + b1 * sto
        for A, c, w in zip(amps, centers, widths):
            y = y + A * pybamm.tanh((sto - c) / w)
        return y
    return ocp


def tanh_ocp_second_derivative_penalty(params: TanhOCPParams,
                                        n_points: int = 200) -> float:
    """OCP 2차 미분 L2 페널티 (smoothness 보장용)."""
    sto = np.linspace(0.01, 0.99, n_points)
    ocp_fn = make_tanh_ocp_numpy(params)
    u = ocp_fn(sto)
    d2u = np.diff(u, n=2)
    return float(np.mean(d2u ** 2))
