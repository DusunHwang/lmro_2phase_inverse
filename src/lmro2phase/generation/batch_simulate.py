"""Stage 2: 병렬 배치 시뮬레이션."""

from __future__ import annotations

import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .sampler import SampleRecord
from .quality_filter import is_quality_ok, QualityConfig
from ..physics.ocp_grid import OCPGrid
from ..physics.ocp_perturbation import perturb_ocp_grid
from ..physics.simulator import SimulationResult

log = logging.getLogger(__name__)


@dataclass
class SimJob:
    record: SampleRecord
    base_ocp_R3m: OCPGrid
    base_ocp_C2m: OCPGrid
    model_type: str   # "SPMe" | "DFN"
    experiment: object
    rng_seed: int


@dataclass
class SimJobResult:
    sample_id: int
    ok: bool
    sim_result: Optional[SimulationResult] = None
    ocp_R3m: Optional[OCPGrid] = None
    ocp_C2m: Optional[OCPGrid] = None
    error: Optional[str] = None


def _run_single_job(job: SimJob) -> SimJobResult:
    """단일 시뮬레이션 (subprocess에서 실행)."""
    try:
        import pybamm
        from ..physics.lmr_parameter_set import build_pybamm_halfcell_params
        from ..physics.positive_2phase_factory import (
            TwoPhaseOCPParams, make_effective_ocp,
            make_effective_diffusivity, make_effective_radius
        )
        from ..physics.halfcell_model_factory import build_halfcell_model, ModelType
        from ..physics.simulator import run_experiment

        rng = np.random.default_rng(job.rng_seed)

        # OCP 변동
        ocp_R3m = perturb_ocp_grid(job.base_ocp_R3m, rng)
        ocp_C2m = perturb_ocp_grid(job.base_ocp_C2m, rng)

        # 2-phase effective
        rec = job.record
        two_phase = TwoPhaseOCPParams(
            frac_R3m=rec.frac_R3m,
            frac_C2m=rec.frac_C2m,
            U_R3m=ocp_R3m.to_pybamm_interpolant(),
            U_C2m=ocp_C2m.to_pybamm_interpolant(),
            D_R3m=10.0 ** rec.log10_D_R3m,
            D_C2m=10.0 ** rec.log10_D_C2m,
            R_R3m=10.0 ** rec.log10_R_R3m,
            R_C2m=10.0 ** rec.log10_R_C2m,
        )

        pv = build_pybamm_halfcell_params(
            ocp_fn=make_effective_ocp(two_phase),
            D_s=make_effective_diffusivity(two_phase),
            R_particle=make_effective_radius(two_phase),
            contact_resistance=10.0 ** rec.log10_contact_resistance,
            capacity_scale=rec.capacity_scale,
            c_s_0_fraction=0.5 + rec.initial_stoichiometry_shift,
        )

        model = build_halfcell_model(ModelType(job.model_type))
        result = run_experiment(model, pv, job.experiment)

        return SimJobResult(
            sample_id=rec.sample_id,
            ok=result.ok,
            sim_result=result if result.ok else None,
            ocp_R3m=ocp_R3m if result.ok else None,
            ocp_C2m=ocp_C2m if result.ok else None,
            error=result.error,
        )
    except Exception as e:
        return SimJobResult(
            sample_id=job.record.sample_id,
            ok=False,
            error=traceback.format_exc(),
        )


def run_batch(jobs: list[SimJob],
               quality_cfg: QualityConfig,
               n_workers: int = 4) -> tuple[list[SimJobResult], list[SimJobResult]]:
    """
    병렬 배치 실행.
    Returns: (good_results, failed_results)
    """
    good, failed = [], []
    total = len(jobs)

    log.info(f"배치 시뮬레이션: {total}개, workers={n_workers}")

    if n_workers <= 1:
        # 단일 프로세스 (디버깅용)
        for i, job in enumerate(jobs):
            r = _run_single_job(job)
            if r.ok and is_quality_ok(r.sim_result, quality_cfg):
                good.append(r)
            else:
                failed.append(r)
            if (i + 1) % 50 == 0:
                log.info(f"  진행: {i+1}/{total} (성공={len(good)}, 실패={len(failed)})")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run_single_job, job): job for job in jobs}
            for i, future in enumerate(as_completed(futures)):
                r = future.result()
                if r.ok and is_quality_ok(r.sim_result, quality_cfg):
                    good.append(r)
                else:
                    failed.append(r)
                if (i + 1) % 50 == 0:
                    log.info(f"  진행: {i+1}/{total} (성공={len(good)}, 실패={len(failed)})")

    log.info(f"완료: 성공={len(good)}, 실패={len(failed)} ({len(failed)/total*100:.1f}%)")
    return good, failed
