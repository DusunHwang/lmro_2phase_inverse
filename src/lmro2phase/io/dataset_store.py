"""처리된 프로파일의 저장/로드 유틸리티."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_profile_parquet(profile, out_path: str | Path) -> None:
    """BatteryProfile을 parquet으로 저장."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "time_s": profile.time_s,
        "voltage_v": profile.voltage_v,
        "current_a": profile.current_a,
    }
    if profile.capacity_ah is not None:
        data["capacity_ah"] = profile.capacity_ah
    if profile.cycle_index is not None:
        data["cycle_index"] = profile.cycle_index.astype(int)
    if profile.step_index is not None:
        data["step_index"] = profile.step_index.astype(int)
    if profile.mode is not None:
        data["mode"] = profile.mode
    if profile.temperature_c is not None:
        data["temperature_c"] = profile.temperature_c

    pd.DataFrame(data).to_parquet(str(out_path), index=False)


def load_profile_parquet(path: str | Path):
    """parquet에서 BatteryProfile 복원."""
    from .profile_schema import BatteryProfile

    df = pd.read_parquet(str(path))
    return BatteryProfile.from_dataframe(df, source_file=str(path))
