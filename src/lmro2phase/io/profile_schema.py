"""BatteryProfile 데이터 스키마 정의."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class StepMode(str, Enum):
    CC_CHARGE = "cc_charge"
    CV_CHARGE = "cv_charge"
    CC_DISCHARGE = "cc_discharge"
    CV_DISCHARGE = "cv_discharge"
    REST = "rest"
    DYNAMIC = "dynamic"
    UNKNOWN = "unknown"


@dataclass
class ProfileSegment:
    """단일 프로토콜 구간."""
    mode: StepMode
    cycle_index: int
    step_index: int
    time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    capacity_ah: Optional[np.ndarray] = None

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0]) if len(self.time_s) > 1 else 0.0

    @property
    def mean_current_a(self) -> float:
        return float(np.mean(self.current_a))

    @property
    def delta_capacity_ah(self) -> float:
        if self.capacity_ah is not None and len(self.capacity_ah) > 1:
            return float(self.capacity_ah[-1] - self.capacity_ah[0])
        return float(np.trapezoid(self.current_a, self.time_s / 3600.0))


@dataclass
class BatteryProfile:
    """파싱된 전체 배터리 프로파일."""
    source_file: str
    time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    capacity_ah: Optional[np.ndarray] = None
    cycle_index: Optional[np.ndarray] = None
    step_index: Optional[np.ndarray] = None
    mode: Optional[np.ndarray] = None          # StepMode 문자열 배열
    temperature_c: Optional[np.ndarray] = None
    segments: list[ProfileSegment] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.time_s)

    def get_cycles(self) -> list[int]:
        if self.cycle_index is not None:
            return sorted(set(self.cycle_index.astype(int).tolist()))
        return []

    def select_cycle(self, cycle: int) -> "BatteryProfile":
        if self.cycle_index is None:
            raise ValueError("cycle_index가 없습니다.")
        mask = self.cycle_index == cycle
        return BatteryProfile(
            source_file=self.source_file,
            time_s=self.time_s[mask],
            voltage_v=self.voltage_v[mask],
            current_a=self.current_a[mask],
            capacity_ah=self.capacity_ah[mask] if self.capacity_ah is not None else None,
            cycle_index=self.cycle_index[mask],
            step_index=self.step_index[mask] if self.step_index is not None else None,
            mode=self.mode[mask] if self.mode is not None else None,
            temperature_c=self.temperature_c[mask] if self.temperature_c is not None else None,
            segments=[s for s in self.segments if s.cycle_index == cycle],
            metadata=self.metadata,
        )

    def get_segments_by_mode(self, mode: StepMode) -> list[ProfileSegment]:
        return [s for s in self.segments if s.mode == mode]

    @classmethod
    def from_dataframe(cls, df, source_file: str = "") -> "BatteryProfile":
        """pandas DataFrame에서 BatteryProfile 생성."""
        import pandas as pd

        def get_col(names: list[str]) -> Optional[np.ndarray]:
            for name in names:
                if name in df.columns:
                    return df[name].to_numpy(dtype=float, na_value=np.nan)
            return None

        time_s = get_col(["time_s"])
        voltage_v = get_col(["voltage_v"])
        current_a = get_col(["current_a"])

        if time_s is None or voltage_v is None or current_a is None:
            raise ValueError("필수 컬럼 time_s / voltage_v / current_a 누락")

        mode_arr = None
        if "mode" in df.columns:
            mode_arr = df["mode"].to_numpy(dtype=str)

        return cls(
            source_file=source_file,
            time_s=time_s,
            voltage_v=voltage_v,
            current_a=current_a,
            capacity_ah=get_col(["capacity_ah"]),
            cycle_index=get_col(["cycle_index"]),
            step_index=get_col(["step_index"]),
            mode=mode_arr,
            temperature_c=get_col(["temperature_c"]),
        )
