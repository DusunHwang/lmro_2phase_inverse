"""
Stage 2: Synthetic dataset generation

실행:
    cd lmro_2phase_inverse
    python scripts/03_generate_synthetic_dataset.py [--config configs/stage2_generate_synthetic.yaml]
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
    parser.add_argument("--config", default="configs/stage2_generate_synthetic.yaml")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="샘플 수 override (없으면 config 사용)")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="병렬 프로세스 수 override (없으면 config 사용)")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ROOT / args.config)

    # best params 로드
    best_params_path = ROOT / "data/reports/stage1_fit/best_params.json"
    if not best_params_path.exists():
        log.error("Stage 1 best_params.json 없음. 02_fit_tanh_ocp.py를 먼저 실행하세요.")
        sys.exit(1)

    with open(best_params_path) as f:
        center = json.load(f)

    from lmro2phase.generation.sampler import (
        SampleRecord, sample_parameters, records_to_dataframe
    )
    from lmro2phase.generation.quality_filter import QualityConfig
    from lmro2phase.physics.ocp_grid import OCPGrid
    from lmro2phase.physics.ocp_tanh import TanhOCPParams

    # Stage 1 OCP 로드
    def load_tanh_ocp(phase: str) -> TanhOCPParams:
        p = ROOT / f"data/reports/stage1_fit/best_ocp_tanh_{phase}.json"
        with open(p) as f:
            d = json.load(f)
        import numpy as np
        return TanhOCPParams(
            b0=d["b0"], b1=d["b1"],
            amps=np.array(d["amps"]),
            centers=np.array(d["centers"]),
            widths=np.array(d["widths"]),
        )

    base_ocp_R3m = OCPGrid.from_tanh_params(load_tanh_ocp("R3m"))
    base_ocp_C2m = OCPGrid.from_tanh_params(load_tanh_ocp("C2m"))

    # 샘플 수
    n_key = cfg.generation.current_n_samples
    n_total = args.n_samples or int(cfg.generation.n_samples[n_key])
    log.info(f"총 샘플 수: {n_total}")

    strat_cfg = cfg.generation.sampling_strategy
    n_local  = int(n_total * float(strat_cfg.local_perturbation))
    n_broad  = int(n_total * float(strat_cfg.broad_prior))
    n_edge   = n_total - n_local - n_broad

    import numpy as np
    rng = np.random.default_rng(42)

    records = []
    records += sample_parameters(n_local, center, cfg.generation, rng, "local")
    records += sample_parameters(n_broad, center, cfg.generation, rng, "broad")
    records += sample_parameters(n_edge,  center, cfg.generation, rng, "edge")

    log.info(f"파라미터 샘플링 완료: {len(records)}개")

    # Experiment 생성 — 실제 데이터 전압 범위(2.5~4.6V)에 맞춤
    from lmro2phase.physics.protocol_builder import build_standard_experiment
    experiment = build_standard_experiment(voltage_upper=4.5, voltage_lower=2.5)

    # Job 생성
    from lmro2phase.generation.batch_simulate import SimJob, run_batch
    jobs = []
    for i, rec in enumerate(records):
        model_type = "DFN" if rng.random() < float(cfg.generation.model_mix.DFN) else "SPMe"
        jobs.append(SimJob(
            record=rec,
            base_ocp_R3m=base_ocp_R3m,
            base_ocp_C2m=base_ocp_C2m,
            model_type=model_type,
            experiment=experiment,
            rng_seed=int(rng.integers(0, 2**31)),
        ))

    quality_cfg = QualityConfig(
        voltage_min_v=float(cfg.quality_filter.voltage_min_v),
        voltage_max_v=float(cfg.quality_filter.voltage_max_v),
    )

    n_workers = args.n_workers or int(getattr(cfg.generation, "n_workers", 1))
    log.info(f"병렬 workers: {n_workers}")
    good, failed = run_batch(jobs, quality_cfg, n_workers=n_workers)

    # --- 저장 ---
    _save_results(good, failed, records, cfg, ROOT)


def _save_results(good, failed, records, cfg, root: Path):
    import numpy as np
    import pandas as pd
    import zarr

    out = cfg.output
    for key in [out.params_file, out.profiles_file, out.ocp_profiles_file,
                out.metadata_file, out.failed_cases_file]:
        (root / key).parent.mkdir(parents=True, exist_ok=True)

    # params parquet — SampleRecord 필드만 저장 (OCPGrid 등 직렬화 불가 객체 제외)
    _RECORD_FIELDS = {
        "sample_id", "log10_D_R3m", "log10_R_R3m", "frac_R3m",
        "log10_D_C2m", "log10_R_C2m", "log10_contact_resistance",
        "capacity_scale", "initial_stoichiometry_shift", "ocp_mode",
    }
    # sample_id → record 매핑을 records 리스트로부터 생성
    record_map = {r.sample_id: r for r in records}
    params_df = pd.DataFrame([
        {f: getattr(record_map[g.sample_id], f) for f in _RECORD_FIELDS
         if hasattr(record_map[g.sample_id], f)}
        for g in good if g.sample_id in record_map
    ])
    params_df.to_parquet(root / out.params_file, index=False)

    # profiles zarr
    max_len = max((len(g.sim_result.time_s) for g in good if g.sim_result), default=0)
    if max_len > 0:
        zroot = zarr.open(str(root / out.profiles_file), mode="w")
        for i, g in enumerate(good):
            if g.sim_result:
                grp = zroot.require_group(str(g.sample_id))
                grp["time_s"] = g.sim_result.time_s
                grp["voltage_v"] = g.sim_result.voltage_v
                grp["current_a"] = g.sim_result.current_a

    # OCP zarr
    zroot_ocp = zarr.open(str(root / out.ocp_profiles_file), mode="w")
    for g in good:
        if g.ocp_R3m and g.ocp_C2m:
            grp = zroot_ocp.require_group(str(g.sample_id))
            grp["ocp_R3m"] = g.ocp_R3m.voltage
            grp["ocp_C2m"] = g.ocp_C2m.voltage

    # failed parquet
    failed_df = pd.DataFrame([
        {"sample_id": f.sample_id, "error": str(f.error)[:500]} for f in failed
    ])
    failed_df.to_parquet(root / out.failed_cases_file, index=False)

    log.info(f"저장 완료: {len(good)}개 성공, {len(failed)}개 실패 로깅")


if __name__ == "__main__":
    main()
