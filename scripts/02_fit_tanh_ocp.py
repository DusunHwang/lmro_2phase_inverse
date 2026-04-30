"""
Stage 1b: tanh OCP 초기 피팅

실행:
    cd lmro_2phase_inverse
    python scripts/02_fit_tanh_ocp.py [--config configs/stage1_fit_tanh.yaml]
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
    parser.add_argument("--config", default="configs/stage1_fit_tanh.yaml")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ROOT / args.config)

    from lmro2phase.io.toyo_ascii import ToyoAsciiParser
    from lmro2phase.io.profile_cleaning import clean_profile
    from lmro2phase.io.profile_schema import StepMode
    from lmro2phase.physics.halfcell_model_factory import (
        ModelType, TwoPhaseStrategy, build_halfcell_model, probe_two_phase_support
    )
    from lmro2phase.fitting.stage1_objective import LossWeights, compute_loss
    from lmro2phase.fitting.stage1_optimizer import (
        run_optuna_search, run_scipy_refinement, save_best_params
    )

    # --- 데이터 로드 ---
    log.info("=== Stage 1: tanh OCP fitting ===")
    data_path = ROOT / cfg.input.data_file
    toyo_cfg = ROOT / cfg.input.toyo_config
    profile = ToyoAsciiParser(toyo_cfg).parse(data_path)

    # 사이클 선택
    use_cycles = list(cfg.input.use_cycles) if cfg.input.use_cycles else profile.get_cycles()
    log.info(f"사용 사이클: {use_cycles}")

    # 단일 사이클 사용 (첫 번째)
    target_cycle = use_cycles[0]
    cycle_profile = profile.select_cycle(target_cycle)
    log.info(f"  사이클 {target_cycle}: {len(cycle_profile)}포인트")

    # --- 모델 빌드 ---
    strategy = probe_two_phase_support()
    log.info(f"  2-phase 전략: {strategy}")

    model = build_halfcell_model(ModelType.SPMe,
                                  two_phase=(strategy.value == "native"),
                                  strategy=strategy)

    # --- loss 함수 ---
    weights = LossWeights.from_dict(dict(cfg.fitting.loss_weights))
    n_terms = int(cfg.fitting.tanh_ocp.n_terms)

    def loss_fn(params):
        return compute_loss(params, cycle_profile, model, weights)

    # --- Global search ---
    log.info(f"Optuna global search ({cfg.fitting.optimizer.global_.n_trials} trials)...")
    best = run_optuna_search(
        loss_fn,
        n_trials=int(cfg.fitting.optimizer.global_.n_trials),
        timeout_s=float(cfg.fitting.optimizer.global_.timeout_s),
        n_tanh_terms=n_terms,
    )
    log.info(f"  Optuna best loss: {loss_fn(best):.6f}")

    # --- Local refinement ---
    log.info("scipy L-BFGS-B refinement...")
    best = run_scipy_refinement(loss_fn, best,
                                 max_iter=int(cfg.fitting.optimizer.local.max_iter))
    log.info(f"  Refined loss: {loss_fn(best):.6f}")

    # --- 저장 ---
    out_dir = ROOT / cfg.output.report_dir
    save_best_params(best, out_dir)
    log.info(f"결과 저장: {out_dir}")

    # --- 플롯 ---
    _plot_results(best, cycle_profile, model, out_dir)


def _plot_results(best, profile, model, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from lmro2phase.physics.ocp_tanh import make_tanh_ocp_numpy

        sto = np.linspace(0.01, 0.99, 300)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for ax, (name, tanh_p) in zip(axes, [("R3m", best.tanh_R3m), ("C2m", best.tanh_C2m)]):
            if tanh_p is None:
                continue
            fn = make_tanh_ocp_numpy(tanh_p)
            ax.plot(sto, fn(sto))
            ax.set_xlabel("Stoichiometry")
            ax.set_ylabel("Voltage (V)")
            ax.set_title(f"OCP {name}")
            ax.grid(True)

        plt.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "ocp_phase_plot.png", dpi=150)
        plt.close(fig)
        log.info("OCP 플롯 저장 완료")
    except Exception as e:
        log.warning(f"플롯 생성 실패: {e}")


if __name__ == "__main__":
    main()
