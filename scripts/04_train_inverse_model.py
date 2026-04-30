"""
Stage 3: DL inverse model 학습

실행:
    cd lmro_2phase_inverse
    python scripts/04_train_inverse_model.py [--config configs/stage3_train_inverse.yaml]
    python scripts/04_train_inverse_model.py --overfit_test  # 100개 overfit 확인
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
    parser.add_argument("--config", default="configs/stage3_train_inverse.yaml")
    parser.add_argument("--overfit_test", action="store_true",
                        help="100개 샘플 overfit 테스트 실행")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ROOT / args.config)

    from lmro2phase.learning.dataset import SyntheticDataset
    from lmro2phase.learning.model_inverse import InverseModel
    from lmro2phase.learning.train import train
    from torch.utils.data import Subset

    dataset = SyntheticDataset(
        params_path=ROOT / cfg.dataset.synthetic_params,
        profiles_path=ROOT / cfg.dataset.synthetic_profiles,
        ocp_path=ROOT / cfg.dataset.synthetic_ocp,
        n_grid=int(cfg.dataset.grid_length),
    )
    log.info(f"데이터셋 크기: {len(dataset)}")

    model_cfg = dict(cfg.model.encoder)
    model_cfg.update({
        "n_scalar": int(cfg.model.heads.scalar_dim),
        "ocp_grid_len": int(cfg.model.heads.ocp_grid_length),
    })

    model = InverseModel(
        in_channels=model_cfg.get("in_channels", 10),
        base_channels=model_cfg.get("base_channels", 64),
        n_blocks=model_cfg.get("n_blocks", 6),
        kernel_size=model_cfg.get("kernel_size", 7),
        dropout=model_cfg.get("dropout", 0.1),
        use_transformer=model_cfg.get("use_transformer", True),
        transformer_heads=model_cfg.get("transformer_heads", 8),
        transformer_layers=model_cfg.get("transformer_layers", 2),
        n_scalar=int(cfg.model.heads.scalar_dim),
        ocp_grid_len=int(cfg.model.heads.ocp_grid_length),
    )

    if args.overfit_test:
        log.info("=== Overfit test (100 samples) ===")
        n_over = int(cfg.training.get("overfit_test", {}).get("n_samples", 100))
        subset = Subset(dataset, list(range(min(n_over, len(dataset)))))
        trained = train(
            model, subset,
            weights=dict(cfg.training.loss_weights),
            batch_size=16,
            max_epochs=int(cfg.training.get("overfit_test", {}).get("max_epochs", 500)),
            lr=float(cfg.training.learning_rate),
            device=str(cfg.training.device),
            checkpoint_dir=None,
        )
        log.info("Overfit test 완료.")
    else:
        log.info("=== Full training ===")
        trained = train(
            model, dataset,
            weights=dict(cfg.training.loss_weights),
            batch_size=int(cfg.training.batch_size),
            max_epochs=int(cfg.training.max_epochs),
            lr=float(cfg.training.learning_rate),
            weight_decay=float(cfg.training.weight_decay),
            train_frac=float(cfg.dataset.train_val_split),
            device=str(cfg.training.device),
            checkpoint_dir=ROOT / cfg.output.checkpoint_dir,
            use_permutation=bool(cfg.training.loss_weights.permutation_invariant),
        )


if __name__ == "__main__":
    main()
