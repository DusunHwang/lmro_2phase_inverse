"""TOYO SYSTEM / TOSCAT ASCII/CSV 파서.

컬럼 순서, 인코딩, 구분자를 자동 감지합니다.
configs/toyo_ascii.yaml 의 alias 매핑으로 다양한 파일 형식에 대응합니다.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from .profile_schema import BatteryProfile, ProfileSegment, StepMode

log = logging.getLogger(__name__)


# ─── 인코딩 자동 감지 ───────────────────────────────────────────────────────

def _decode_bytes(raw: bytes, candidates: list[str]) -> str:
    for enc in candidates:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    # charset-normalizer fallback
    try:
        from charset_normalizer import from_bytes
        result = from_bytes(raw).best()
        if result:
            return str(result)
    except ImportError:
        pass
    return raw.decode("utf-8", errors="replace")


# ─── 구분자 자동 감지 ────────────────────────────────────────────────────────

def _detect_delimiter(line: str, candidates: list[str]) -> str:
    counts = {}
    for d in candidates:
        if d == "whitespace":
            counts[d] = len(line.split()) - 1
        else:
            counts[d] = line.count(d)
    best = max(counts, key=lambda k: counts[k])
    return None if counts[best] == 0 else best


# ─── 헤더 행 감지 ────────────────────────────────────────────────────────────

def _find_header_row(lines: list[str], required_keywords: list[str],
                     max_metadata_rows: int) -> int:
    """required_keywords 중 하나라도 포함된 첫 번째 행 인덱스를 반환."""
    for i, line in enumerate(lines[:max_metadata_rows + 5]):
        up = line.upper()
        if any(kw.upper() in up for kw in required_keywords):
            return i
    raise ValueError(
        f"헤더 행을 찾지 못했습니다. 첫 {max_metadata_rows}행 내에 "
        f"{required_keywords} 중 하나가 있어야 합니다."
    )


# ─── 컬럼명 정규화 ────────────────────────────────────────────────────────────

def _normalize_columns(df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    """DataFrame 컬럼을 alias 맵핑 기준으로 표준 이름으로 변환."""
    rename_map = {}
    for std_name, alias_list in aliases.items():
        for alias in alias_list:
            for col in df.columns:
                if col.strip().lower() == alias.strip().lower():
                    rename_map[col] = std_name
                    break
    return df.rename(columns=rename_map)


# ─── 단위 변환 ────────────────────────────────────────────────────────────────

def _apply_unit_conversion(df: pd.DataFrame, unit_cfg: dict) -> pd.DataFrame:
    # mA → A
    if unit_cfg.get("current_ma_to_a") and "current_ma" in df.columns:
        df["current_a"] = df["current_ma"].astype(float) * 1e-3
        df = df.drop(columns=["current_ma"])
        log.debug("current_ma → current_a 변환 완료")

    # mAh → Ah
    if unit_cfg.get("capacity_mah_to_ah") and "capacity_mah" in df.columns:
        df["capacity_ah"] = df["capacity_mah"].astype(float) * 1e-3
        df = df.drop(columns=["capacity_mah"])
        log.debug("capacity_mah → capacity_ah 변환 완료")

    return df


# ─── 전류 부호 변환 ───────────────────────────────────────────────────────────

def _apply_current_sign(df: pd.DataFrame, sign_cfg: dict) -> pd.DataFrame:
    """PyBaMM half-cell convention에 맞게 전류 부호를 정렬."""
    # 현재 파일: 충전=양수, 방전=음수 (discharge_positive_in_file=False)
    # PyBaMM half-cell: 방전=양수 (pybamm_discharge_positive=False → 충전=양수 유지)
    # 두 convention이 같으면 그대로, 다르면 부호 반전
    file_dchg_pos = sign_cfg.get("discharge_positive_in_file", False)
    pybamm_dchg_pos = sign_cfg.get("pybamm_discharge_positive", False)

    if file_dchg_pos != pybamm_dchg_pos:
        df["current_a"] = -df["current_a"]
        log.debug("전류 부호 반전 적용")
    return df


# ─── Mode 컬럼 정규화 ─────────────────────────────────────────────────────────

def _normalize_mode_column(df: pd.DataFrame, mode_map: dict) -> pd.DataFrame:
    if "mode" not in df.columns:
        return df
    reverse = {}
    for std, labels in mode_map.items():
        for lab in labels:
            reverse[lab.lower()] = std
    df["mode"] = df["mode"].astype(str).str.strip().str.lower().map(
        lambda x: reverse.get(x, StepMode.UNKNOWN.value)
    )
    return df


# ─── 프로토콜 자동 분리 ───────────────────────────────────────────────────────

def _infer_step_mode(current_a: np.ndarray, voltage_v: np.ndarray,
                     cfg: dict) -> StepMode:
    rest_thr = cfg.get("rest_current_threshold_a", 1e-4)
    cc_ratio_thr = cfg.get("cc_std_ratio_threshold", 0.05)
    cv_v_thr = cfg.get("cv_voltage_std_threshold_v", 0.002)

    mean_i = float(np.mean(current_a))
    if abs(mean_i) < rest_thr:
        return StepMode.REST

    std_i = float(np.std(current_a))
    if abs(mean_i) > 0 and std_i / abs(mean_i) < cc_ratio_thr:
        return StepMode.CC_CHARGE if mean_i > 0 else StepMode.CC_DISCHARGE

    std_v = float(np.std(voltage_v))
    if std_v < cv_v_thr:
        return StepMode.CV_CHARGE if mean_i > 0 else StepMode.CV_DISCHARGE

    return StepMode.DYNAMIC


def _build_segments(profile: BatteryProfile, proto_cfg: dict) -> list[ProfileSegment]:
    """cycle/step 경계로 ProfileSegment 목록 생성."""
    segments = []
    df_idx = np.arange(len(profile.time_s))

    if profile.cycle_index is not None and profile.step_index is not None:
        cycle_arr = profile.cycle_index.astype(int)
        step_arr = profile.step_index.astype(int)
        # (cycle, step) 그룹
        keys = np.stack([cycle_arr, step_arr], axis=1)
        boundaries = np.where(np.any(np.diff(keys, axis=0) != 0, axis=1))[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(df_idx)]])
    else:
        starts, ends = np.array([0]), np.array([len(df_idx)])
        cycle_arr = np.zeros(len(df_idx), dtype=int)
        step_arr = np.zeros(len(df_idx), dtype=int)

    for s, e in zip(starts, ends):
        sl = slice(s, e)
        cur = profile.current_a[sl]
        vol = profile.voltage_v[sl]

        # Mode 컬럼이 있으면 우선 사용
        if profile.mode is not None:
            raw_mode = profile.mode[s]
            try:
                mode = StepMode(str(raw_mode))
            except ValueError:
                mode = _infer_step_mode(cur, vol, proto_cfg)
        else:
            mode = _infer_step_mode(cur, vol, proto_cfg)

        segments.append(ProfileSegment(
            mode=mode,
            cycle_index=int(cycle_arr[s]),
            step_index=int(step_arr[s]),
            time_s=profile.time_s[sl].copy(),
            voltage_v=vol.copy(),
            current_a=cur.copy(),
            capacity_ah=profile.capacity_ah[sl].copy() if profile.capacity_ah is not None else None,
        ))
    return segments


# ─── 메인 파서 ───────────────────────────────────────────────────────────────

class ToyoAsciiParser:
    """TOYO ASCII/CSV 파일을 BatteryProfile로 변환하는 파서."""

    def __init__(self, config_path: str | Path = "configs/toyo_ascii.yaml"):
        self.cfg = OmegaConf.load(str(config_path))

    def parse(self, path: str | Path) -> BatteryProfile:
        path = Path(path)
        log.info(f"파일 파싱 시작: {path}")

        raw = path.read_bytes()

        # 1. 인코딩 감지
        text = _decode_bytes(raw, list(self.cfg.encoding_candidates))
        lines = text.splitlines()
        log.debug(f"  인코딩 감지 완료, 총 {len(lines)}행")

        # 2. 헤더 행 감지
        header_idx = _find_header_row(
            lines,
            list(self.cfg.header.required_keywords),
            int(self.cfg.header.max_metadata_rows),
        )
        metadata_lines = lines[:header_idx]
        log.debug(f"  헤더 행: {header_idx}번째")

        # 3. 구분자 감지
        delimiter = _detect_delimiter(lines[header_idx], list(self.cfg.delimiter_candidates))
        if delimiter is None:
            delimiter = ","
            log.warning("구분자를 감지하지 못해 ','로 fallback")
        log.debug(f"  구분자: {repr(delimiter)}")

        # 4. 단위 행 skip 처리
        skip_unit = bool(self.cfg.header.skip_unit_row)
        data_start = header_idx + (2 if skip_unit else 1)

        # 5. DataFrame 읽기
        sep = r"\s+" if delimiter == "whitespace" else re.escape(delimiter)
        content = "\n".join(lines[header_idx:])
        df = pd.read_csv(
            io.StringIO(content),
            sep=sep,
            engine="python",
            skiprows=1 if skip_unit else 0,
        )
        df = df.dropna(how="all")
        log.debug(f"  DataFrame shape: {df.shape}")

        # 6. 컬럼 정규화
        all_aliases = dict(self.cfg.column_aliases)
        df = _normalize_columns(df, all_aliases)

        # 7. 단위 변환
        df = _apply_unit_conversion(df, dict(self.cfg.unit_conversion))

        # 8. 전류 부호 변환
        df = _apply_current_sign(df, dict(self.cfg.current_sign))

        # 9. rest 구간 노이즈 clipping
        if "current_a" in df.columns:
            rest_thr = float(self.cfg.protocol_detection.rest_current_threshold_a)
            df["current_a"] = df["current_a"].where(
                df["current_a"].abs() >= rest_thr, other=0.0
            )

        # 10. Mode 컬럼 정규화
        df = _normalize_mode_column(df, dict(self.cfg.mode_label_map))

        # 11. BatteryProfile 생성
        profile = BatteryProfile.from_dataframe(df, source_file=str(path))
        profile.metadata["source_lines"] = len(lines)
        profile.metadata["header_row"] = header_idx
        profile.metadata["delimiter"] = delimiter
        for i, mline in enumerate(metadata_lines):
            profile.metadata[f"meta_{i}"] = mline

        # 12. 세그먼트 분리
        proto_cfg = dict(self.cfg.protocol_detection)
        profile.segments = _build_segments(profile, proto_cfg)
        log.info(f"  파싱 완료: {len(profile)}포인트, {len(profile.segments)}세그먼트, "
                 f"사이클 {profile.get_cycles()}")
        return profile


def parse_toyo(path: str | Path,
               config_path: str | Path = "configs/toyo_ascii.yaml") -> BatteryProfile:
    """단일 함수 편의 인터페이스."""
    return ToyoAsciiParser(config_path).parse(path)
