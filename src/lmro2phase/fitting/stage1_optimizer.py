"""Stage 1 optimizer: Optuna global search → scipy local refinement."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .stage1_objective import LossWeights, Stage1Params
from ..physics.ocp_tanh import TanhOCPParams

log = logging.getLogger(__name__)


def build_optuna_objective(loss_fn: Callable[[Stage1Params], float],
                            n_tanh_terms: int = 5) -> Callable:
    """Optuna trial → Stage1Params → loss 변환 함수 생성."""

    def objective(trial):
        p = Stage1Params(
            log10_D_R3m=trial.suggest_float("log10_D_R3m", -17, -12),
            log10_R_R3m=trial.suggest_float("log10_R_R3m", -7.0, -4.5),
            frac_R3m=trial.suggest_float("frac_R3m", 0.1, 0.9),
            log10_D_C2m=trial.suggest_float("log10_D_C2m", -18, -12),
            log10_R_C2m=trial.suggest_float("log10_R_C2m", -7.0, -4.5),
            log10_contact_resistance=trial.suggest_float("log10_contact_R", -5, -1),
            capacity_scale=trial.suggest_float("capacity_scale", 0.7, 1.5),
            # LMR 사이클은 x≈0.55에서 시작 → shift 범위 확대
            initial_stoichiometry_shift=trial.suggest_float("stoich_shift", -0.2, 0.3),
        )

        # R3m tanh OCP: LMR R3m은 b0≈4.5V (고전압 상한)
        p.tanh_R3m = TanhOCPParams(
            b0=trial.suggest_float("R3m_b0", 3.5, 4.8),
            b1=trial.suggest_float("R3m_b1", -2.5, -0.5),
            amps=np.array([trial.suggest_float(f"R3m_A{i}", -0.4, 0.4)
                           for i in range(n_tanh_terms)]),
            centers=np.array([trial.suggest_float(f"R3m_c{i}", 0.05, 0.95)
                               for i in range(n_tanh_terms)]),
            widths=np.abs(np.array([trial.suggest_float(f"R3m_w{i}", 0.01, 0.4)
                                     for i in range(n_tanh_terms)])),
        )
        # C2m tanh OCP
        p.tanh_C2m = TanhOCPParams(
            b0=trial.suggest_float("C2m_b0", 3.0, 4.8),
            b1=trial.suggest_float("C2m_b1", -2.5, -0.3),
            amps=np.array([trial.suggest_float(f"C2m_A{i}", -0.4, 0.4)
                           for i in range(n_tanh_terms)]),
            centers=np.array([trial.suggest_float(f"C2m_c{i}", 0.05, 0.95)
                               for i in range(n_tanh_terms)]),
            widths=np.abs(np.array([trial.suggest_float(f"C2m_w{i}", 0.01, 0.4)
                                     for i in range(n_tanh_terms)])),
        )

        return loss_fn(p)

    return objective


def run_optuna_search(loss_fn: Callable[[Stage1Params], float],
                       n_trials: int = 200,
                       n_jobs: int = 1,
                       timeout_s: Optional[float] = None,
                       n_tanh_terms: int = 5,
                       seed: int = 42) -> Stage1Params:
    """Optuna TPE로 global search."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    obj = build_optuna_objective(loss_fn, n_tanh_terms)
    study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs, timeout=timeout_s)

    best_trial = study.best_trial
    log.info(f"Optuna 완료: best loss={best_trial.value:.6f}, trial={best_trial.number}")

    # best params 재구성
    p = best_trial.params
    best = Stage1Params(
        log10_D_R3m=p["log10_D_R3m"],
        log10_R_R3m=p["log10_R_R3m"],
        frac_R3m=p["frac_R3m"],
        log10_D_C2m=p["log10_D_C2m"],
        log10_R_C2m=p["log10_R_C2m"],
        log10_contact_resistance=p["log10_contact_R"],
        capacity_scale=p["capacity_scale"],
        initial_stoichiometry_shift=p["stoich_shift"],
    )
    best.tanh_R3m = TanhOCPParams(
        b0=p["R3m_b0"], b1=p["R3m_b1"],
        amps=np.array([p[f"R3m_A{i}"] for i in range(n_tanh_terms)]),
        centers=np.array([p[f"R3m_c{i}"] for i in range(n_tanh_terms)]),
        widths=np.abs(np.array([p[f"R3m_w{i}"] for i in range(n_tanh_terms)])),
    )
    best.tanh_C2m = TanhOCPParams(
        b0=p["C2m_b0"], b1=p["C2m_b1"],
        amps=np.array([p[f"C2m_A{i}"] for i in range(n_tanh_terms)]),
        centers=np.array([p[f"C2m_c{i}"] for i in range(n_tanh_terms)]),
        widths=np.abs(np.array([p[f"C2m_w{i}"] for i in range(n_tanh_terms)])),
    )
    return best


def run_scipy_refinement(loss_fn: Callable[[Stage1Params], float],
                          init_params: Stage1Params,
                          max_iter: int = 500) -> Stage1Params:
    """scipy L-BFGS-B로 local refinement."""
    from scipy.optimize import minimize

    n_terms = init_params.tanh_R3m.n_terms if init_params.tanh_R3m else 5

    def pack(p: Stage1Params) -> np.ndarray:
        v = [p.log10_D_R3m, p.log10_R_R3m, p.frac_R3m,
             p.log10_D_C2m, p.log10_R_C2m,
             p.log10_contact_resistance, p.capacity_scale,
             p.initial_stoichiometry_shift]
        if p.tanh_R3m:
            v += p.tanh_R3m.to_vector().tolist()
        if p.tanh_C2m:
            v += p.tanh_C2m.to_vector().tolist()
        return np.array(v)

    def unpack(v: np.ndarray) -> Stage1Params:
        offset = 8
        p = Stage1Params(
            log10_D_R3m=float(v[0]), log10_R_R3m=float(v[1]),
            frac_R3m=float(np.clip(v[2], 0.05, 0.95)),
            log10_D_C2m=float(v[3]), log10_R_C2m=float(v[4]),
            log10_contact_resistance=float(v[5]),
            capacity_scale=float(np.clip(v[6], 0.5, 2.0)),
            initial_stoichiometry_shift=float(np.clip(v[7], -0.3, 0.4)),
        )
        np_tanh = 2 + 3 * n_terms
        p.tanh_R3m = TanhOCPParams.from_vector(v[offset:offset + np_tanh], n_terms)
        p.tanh_C2m = TanhOCPParams.from_vector(v[offset + np_tanh:], n_terms)
        return p

    x0 = pack(init_params)

    def scipy_obj(x):
        try:
            return loss_fn(unpack(x))
        except Exception:
            return 1e9

    result = minimize(scipy_obj, x0, method="L-BFGS-B",
                      options={"maxiter": max_iter, "ftol": 1e-10})
    log.info(f"scipy refinement 완료: loss={result.fun:.6f}, success={result.success}")
    return unpack(result.x)


def save_best_params(params: Stage1Params, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _np_to_list(v):
        return v.tolist() if hasattr(v, "tolist") else v

    d = {
        "log10_D_R3m": params.log10_D_R3m,
        "log10_R_R3m": params.log10_R_R3m,
        "frac_R3m": params.frac_R3m,
        "frac_C2m": params.frac_C2m,
        "log10_D_C2m": params.log10_D_C2m,
        "log10_R_C2m": params.log10_R_C2m,
        "log10_contact_resistance": params.log10_contact_resistance,
        "capacity_scale": params.capacity_scale,
        "initial_stoichiometry_shift": params.initial_stoichiometry_shift,
    }
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(d, f, indent=2)

    for name, tanh_p in [("R3m", params.tanh_R3m), ("C2m", params.tanh_C2m)]:
        if tanh_p is None:
            continue
        td = {
            "b0": tanh_p.b0, "b1": tanh_p.b1,
            "amps": _np_to_list(tanh_p.amps),
            "centers": _np_to_list(tanh_p.centers),
            "widths": _np_to_list(tanh_p.widths),
        }
        with open(out_dir / f"best_ocp_tanh_{name}.json", "w") as f:
            json.dump(td, f, indent=2)

    log.info(f"파라미터 저장: {out_dir}")
