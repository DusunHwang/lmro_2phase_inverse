"""
Stage 4a: 실측 TOYO 프로파일 → DL inverse model 추론

실행:
    cd lmro_2phase_inverse
    python scripts/05_infer_lmr_profile.py [--config configs/stage4_infer_validate.yaml]
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
    parser.add_argument("--config", default="configs/stage4_infer_validate.yaml")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ROOT / args.config)

    from lmro2phase.io.toyo_ascii import ToyoAsciiParser
    from lmro2phase.features.capacity_axis import normalize_to_capacity_grid
    from lmro2phase.features.differential_features import build_feature_tensor
    from lmro2phase.learning.infer import load_model, infer_profile, save_inference_results
    from omegaconf import OmegaConf
    import numpy as np

    model_cfg_yaml = OmegaConf.load(ROOT / "configs/stage3_train_inverse.yaml")
    model_cfg = {
        "in_channels": int(model_cfg_yaml.model.encoder.in_channels),
        "base_channels": int(model_cfg_yaml.model.encoder.base_channels),
        "n_blocks": int(model_cfg_yaml.model.encoder.n_blocks),
        "kernel_size": int(model_cfg_yaml.model.encoder.kernel_size),
        "dropout": float(model_cfg_yaml.model.encoder.dropout),
        "use_transformer": bool(model_cfg_yaml.model.encoder.use_transformer),
        "transformer_heads": int(model_cfg_yaml.model.encoder.transformer_heads),
        "transformer_layers": int(model_cfg_yaml.model.encoder.transformer_layers),
        "n_scalar": int(model_cfg_yaml.model.heads.scalar_dim),
        "ocp_grid_len": int(model_cfg_yaml.model.heads.ocp_grid_length),
    }

    model, device = load_model(ROOT / cfg.inference.model_checkpoint, model_cfg)

    profile_parser = ToyoAsciiParser(ROOT / cfg.inference.toyo_config)
    profile = profile_parser.parse(ROOT / cfg.inference.input_data)

    log.info(f"프로파일: {len(profile)}포인트, 사이클={profile.get_cycles()}")

    use_cycles = list(cfg.inference.use_cycles) if cfg.inference.use_cycles else profile.get_cycles()

    out_dir = ROOT / cfg.output.inference_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for cycle in use_cycles:
        cp = profile.select_cycle(cycle)
        log.info(f"사이클 {cycle} 추론 중...")

        # feature 추출
        t = cp.time_s
        v = cp.voltage_v
        i = cp.current_a
        n_grid = int(model_cfg_yaml.dataset.grid_length)

        chg = i > 0
        dchg = i < 0
        q = np.linspace(0, 1, n_grid)

        if chg.sum() > 5:
            _, v_chg, i_chg = normalize_to_capacity_grid(t[chg], v[chg], i[chg], n_grid)
        else:
            v_chg = np.full(n_grid, np.nan)
            i_chg = np.zeros(n_grid)

        if dchg.sum() > 5:
            _, v_dchg, i_dchg = normalize_to_capacity_grid(
                t[dchg], v[dchg], np.abs(i[dchg]), n_grid)
        else:
            v_dchg = np.full(n_grid, np.nan)
            i_dchg = np.zeros(n_grid)

        feat = build_feature_tensor(q, v_chg, i_chg, v_dchg, i_dchg)  # [10, 512]
        result = infer_profile(model, feat, device)

        cycle_dir = out_dir / f"cycle_{cycle:03d}"
        save_inference_results(result, cycle_dir)
        log.info(f"  → 저장: {cycle_dir}")

    log.info("추론 완료.")


if __name__ == "__main__":
    main()
