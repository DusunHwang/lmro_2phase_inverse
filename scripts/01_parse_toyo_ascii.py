"""
Stage 1a: TOYO ASCII 파싱 및 전처리

실행:
    cd lmro_2phase_inverse
    python scripts/01_parse_toyo_ascii.py [--input DATA_FILE] [--config CONFIG]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/toyo/Toyo_LMR_HalfCell_Sample_50cycles.csv")
    parser.add_argument("--config", default="configs/toyo_ascii.yaml")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()

    input_path = ROOT / args.input
    config_path = ROOT / args.config
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"입력 파일: {input_path}")

    from lmro2phase.io.toyo_ascii import ToyoAsciiParser
    from lmro2phase.io.profile_cleaning import clean_profile
    from lmro2phase.io.dataset_store import save_profile_parquet

    parser_obj = ToyoAsciiParser(config_path)
    profile = parser_obj.parse(input_path)

    log.info(f"파싱 완료: {len(profile)}포인트, 사이클={profile.get_cycles()}")
    log.info(f"세그먼트 수: {len(profile.segments)}")

    # 세그먼트 통계
    from lmro2phase.io.profile_schema import StepMode
    for mode in StepMode:
        segs = profile.get_segments_by_mode(mode)
        if segs:
            log.info(f"  {mode.value}: {len(segs)}개 세그먼트")

    # 정제
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(str(config_path))
    profile = clean_profile(
        profile,
        voltage_min=float(cfg.cleaning.voltage_min_v),
        voltage_max=float(cfg.cleaning.voltage_max_v),
        current_max_abs=float(cfg.cleaning.current_max_abs_a),
    )

    # 저장
    out_file = out_dir / (Path(args.input).stem + "_processed.parquet")
    save_profile_parquet(profile, out_file)
    log.info(f"저장 완료: {out_file}")

    # 간단한 summary 출력
    import numpy as np
    log.info(f"  시간 범위: {profile.time_s[0]:.0f}~{profile.time_s[-1]:.0f} s")
    log.info(f"  전압 범위: {profile.voltage_v.min():.3f}~{profile.voltage_v.max():.3f} V")
    log.info(f"  전류 범위: {profile.current_a.min():.4f}~{profile.current_a.max():.4f} A")


if __name__ == "__main__":
    main()
