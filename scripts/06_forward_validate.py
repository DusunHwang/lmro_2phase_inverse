"""
Stage 4b: 예측 파라미터 → PyBaMM forward validation

실행:
    cd lmro_2phase_inverse
    python scripts/06_forward_validate.py [--config configs/stage4_infer_validate.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage4_infer_validate.yaml")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ROOT / args.config)

    from lmro2phase.io.toyo_ascii import ToyoAsciiParser
    from lmro2phase.validation.forward_validate import validate_forward
    from lmro2phase.validation.plots import plot_forward_validation

    out_dir = ROOT / cfg.output.inference_dir
    profile_parser = ToyoAsciiParser(ROOT / cfg.inference.toyo_config)
    profile = profile_parser.parse(ROOT / cfg.inference.input_data)

    model_type = str(cfg.forward_validation.model_type)
    all_summaries = []

    cycle_dirs = sorted(out_dir.glob("cycle_*"))
    if not cycle_dirs:
        log.error(f"추론 결과 없음: {out_dir}. 05_infer_lmr_profile.py를 먼저 실행하세요.")
        sys.exit(1)

    for cycle_dir in cycle_dirs:
        cycle_num = int(cycle_dir.name.split("_")[1])
        params_file = cycle_dir / "predicted_params.json"
        if not params_file.exists():
            continue

        with open(params_file) as f:
            params = json.load(f)

        # OCP 로드
        import numpy as np, pandas as pd
        for phase in ["R3m", "C2m"]:
            ocp_file = cycle_dir / f"predicted_ocp_{phase}.csv"
            if ocp_file.exists():
                df_ocp = pd.read_csv(ocp_file)
                params[f"ocp_{phase}_grid"] = df_ocp["voltage_v"].tolist()

        if "ocp_R3m_grid" not in params or "ocp_C2m_grid" not in params:
            log.warning(f"사이클 {cycle_num}: OCP 파일 없음, skip")
            continue

        cp = profile.select_cycle(cycle_num)
        log.info(f"사이클 {cycle_num} forward validation ({model_type})...")

        val_result = validate_forward(params, cp, model_type=model_type)

        if val_result["ok"]:
            log.info(f"  RMSE={val_result['rmse_v']*1000:.1f} mV, "
                     f"MAE={val_result['mae_v']*1000:.1f} mV")
        else:
            log.warning(f"  실패: {val_result.get('error', 'unknown')}")

        # 잔차 플롯
        plot_path = cycle_dir / "forward_validation.png"
        plot_forward_validation(val_result, plot_path)

        # 잔차 summary JSON
        summary = {k: v for k, v in val_result.items()
                   if k not in ("time_s", "voltage_sim", "voltage_exp", "residual")}
        summary["cycle"] = cycle_num
        with open(cycle_dir / "residual_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        all_summaries.append(summary)

    # 전체 summary
    if all_summaries:
        import numpy as np
        all_rmse = [s["rmse_v"] for s in all_summaries if s.get("ok")]
        log.info(f"전체 사이클 RMSE: mean={np.mean(all_rmse)*1000:.1f} mV, "
                 f"max={np.max(all_rmse)*1000:.1f} mV")

        with open(ROOT / cfg.output.residual_summary, "w") as f:
            json.dump(all_summaries, f, indent=2)

    log.info("Forward validation 완료.")


if __name__ == "__main__":
    main()
