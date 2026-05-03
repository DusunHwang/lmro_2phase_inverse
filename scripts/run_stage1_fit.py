"""Stage 1 역추정 실행 스크립트.

최종 가상 시뮬레이션 결과(native 2-phase Gaussian C2m LowBroad 2x D100x slow)를
측정치로 가정하고, effective 2-phase SPM 모델로 R, diffusivity, OCP를 추정합니다.

추정 전략:
  - 피팅 모델: effective single-phase SPM (FallbackA)
    - effective OCP = frac_R3m*U_R3m + frac_C2m*U_C2m
    - effective D = frac_R3m*D_R3m + frac_C2m*D_C2m
    - effective R = frac_R3m*R_R3m + frac_C2m*R_C2m
  - 데이터: cycle 1 (0.1C) 전체 (충전+휴지+방전+휴지), 5분 격자 서브샘플링
  - 초기 농도: c_s_0_fraction=0.99 (고정 — 방전 직전 완충 상태)
  - 전역 탐색: Optuna TPE
  - 지역 정제: scipy L-BFGS-B

출력:
  --out-dir 폴더에
    fit_history.jsonl   : 각 trial의 파라미터 및 loss
    best_params.json    : 최종 추정값
    comparison.json     : 추정값 vs 진실값 비교
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── 프로젝트 패키지 경로 ─────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "src"))


# ─── 전역 피팅 설정 ───────────────────────────────────────────────────────────

DATA_CSV = (
    "data/raw/toyo/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/"
    "Toyo_LMR_native2phase_PyBaMM_0p1C_0p33C_0p5C_1C.csv"
)
TRUE_PARAMS_JSON = (
    "data/raw/toyo/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/"
    "true_native_2phase_parameters.json"
)

# 서브샘플링: 원본 0.5s → 300s 해상도 (every 600th row)
SUBSAMPLE_STEP = 600
USE_CYCLE = 1
C_S_0_FRACTION = 0.99   # 고정: 충전 직전 완방전 상태

N_TANH = 5              # 각 phase tanh 항 수

# Optuna
N_OPTUNA_TRIALS = 80
OPTUNA_SEED = 42

# scipy L-BFGS-B
SCIPY_MAX_ITER = 300


# ─── 검색 공간 ────────────────────────────────────────────────────────────────

BOUNDS = {
    "log10_D_R3m":   (-18.0, -14.0),
    "log10_R_R3m":   (-7.5,  -5.5),
    "frac_R3m":      (0.10,   0.90),
    "log10_D_C2m":   (-19.0, -15.0),
    "log10_R_C2m":   (-7.5,  -5.5),
    "log10_contact": (-5.0,  -1.0),
    # OCP R3m
    "R3m_b0":  (3.5,  4.8),
    "R3m_b1":  (-2.5, -0.3),
    # OCP C2m
    "C2m_b0":  (3.0,  4.8),
    "C2m_b1":  (-2.5, -0.3),
}
for i in range(N_TANH):
    BOUNDS[f"R3m_A{i}"] = (-0.5, 0.5)
    BOUNDS[f"R3m_c{i}"] = (0.05, 0.95)
    BOUNDS[f"R3m_w{i}"] = (0.01, 0.5)
    BOUNDS[f"C2m_A{i}"] = (-0.5, 0.5)
    BOUNDS[f"C2m_c{i}"] = (0.05, 0.95)
    BOUNDS[f"C2m_w{i}"] = (0.01, 0.5)

PARAM_KEYS = list(BOUNDS.keys())
LOWER = np.array([BOUNDS[k][0] for k in PARAM_KEYS])
UPPER = np.array([BOUNDS[k][1] for k in PARAM_KEYS])


# ─── 데이터 로드 & 서브샘플링 ─────────────────────────────────────────────────

def load_profile(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TOYO CSV → (time_s, current_a, voltage_v) for the fitting cycle."""
    from lmro2phase.io.toyo_ascii import parse_toyo

    profile = parse_toyo(csv_path)
    c = profile.select_cycle(USE_CYCLE)
    idx = np.arange(0, len(c), SUBSAMPLE_STEP)
    t = c.time_s[idx]
    i = c.current_a[idx]
    v = c.voltage_v[idx]
    log.info(f"데이터 로드: cycle={USE_CYCLE}, n={len(t)} 포인트, span={t[-1]/3600:.1f}h")
    return t, i, v


# ─── 모델 빌드 ────────────────────────────────────────────────────────────────

def build_model():
    from lmro2phase.physics.halfcell_model_factory import (
        build_halfcell_model, ModelType, TwoPhaseStrategy
    )
    return build_halfcell_model(ModelType.SPM, two_phase=False,
                                 strategy=TwoPhaseStrategy.FALLBACK_A)


# ─── 파라미터 벡터 → 모델 파라미터 변환 ────────────────────────────────────────

def vec_to_pybamm_params(v: np.ndarray):
    """파라미터 벡터를 PyBaMM parameter set으로 변환."""
    from lmro2phase.physics.lmr_parameter_set import build_pybamm_halfcell_params
    from lmro2phase.physics.ocp_tanh import TanhOCPParams, make_tanh_ocp_pybamm
    from lmro2phase.physics.positive_2phase_factory import (
        TwoPhaseOCPParams, make_effective_ocp, make_effective_diffusivity, make_effective_radius
    )

    d = {k: float(v[i]) for i, k in enumerate(PARAM_KEYS)}

    log10_D_R3m = d["log10_D_R3m"]
    log10_R_R3m = d["log10_R_R3m"]
    frac_R3m    = float(np.clip(d["frac_R3m"], 0.05, 0.95))
    frac_C2m    = 1.0 - frac_R3m
    log10_D_C2m = d["log10_D_C2m"]
    log10_R_C2m = d["log10_R_C2m"]
    contact     = 10.0 ** d["log10_contact"]

    tanh_R3m = TanhOCPParams(
        b0=d["R3m_b0"], b1=d["R3m_b1"],
        amps   =np.array([d[f"R3m_A{i}"] for i in range(N_TANH)]),
        centers=np.array([d[f"R3m_c{i}"] for i in range(N_TANH)]),
        widths =np.abs([d[f"R3m_w{i}"] for i in range(N_TANH)]),
    )
    tanh_C2m = TanhOCPParams(
        b0=d["C2m_b0"], b1=d["C2m_b1"],
        amps   =np.array([d[f"C2m_A{i}"] for i in range(N_TANH)]),
        centers=np.array([d[f"C2m_c{i}"] for i in range(N_TANH)]),
        widths =np.abs([d[f"C2m_w{i}"] for i in range(N_TANH)]),
    )

    ocp_R3m = make_tanh_ocp_pybamm(tanh_R3m)
    ocp_C2m = make_tanh_ocp_pybamm(tanh_C2m)

    two_phase = TwoPhaseOCPParams(
        frac_R3m=frac_R3m, frac_C2m=frac_C2m,
        U_R3m=ocp_R3m, U_C2m=ocp_C2m,
        D_R3m=10.0 ** log10_D_R3m, D_C2m=10.0 ** log10_D_C2m,
        R_R3m=10.0 ** log10_R_R3m, R_C2m=10.0 ** log10_R_C2m,
    )
    eff_ocp = make_effective_ocp(two_phase)
    eff_D   = make_effective_diffusivity(two_phase)
    eff_R   = make_effective_radius(two_phase)

    pv = build_pybamm_halfcell_params(
        ocp_fn=eff_ocp,
        D_s=eff_D,
        R_particle=eff_R,
        contact_resistance=contact,
        capacity_scale=1.0,
        c_s_0_fraction=C_S_0_FRACTION,
    )
    return pv, d


# ─── loss 계산 ────────────────────────────────────────────────────────────────

N_GRID = 512

def _norm_grid(t, v, i, n=N_GRID):
    cap = np.cumsum(np.abs(i) * np.gradient(t) / 3600.0)
    if cap[-1] < 1e-9:
        return np.linspace(0, 1, n), np.full(n, np.nan)
    q = cap / cap[-1]
    fn = interp1d(q, v, bounds_error=False, fill_value="extrapolate")
    return np.linspace(0, 1, n), fn(np.linspace(0, 1, n))


def compute_loss(v: np.ndarray, model, time_s, current_a, voltage_v_ref) -> float:
    from lmro2phase.physics.simulator import run_current_drive

    try:
        pv, _ = vec_to_pybamm_params(np.clip(v, LOWER, UPPER))
    except Exception:
        return 1e9

    t_rel = time_s - time_s[0]
    result = run_current_drive(model, pv, time_s, current_a, t_eval=t_rel)
    if not result.ok:
        return 1e8

    # V(t) MSE
    v_sim_t = interp1d(result.time_s, result.voltage_v,
                        bounds_error=False, fill_value="extrapolate")(t_rel)
    loss_vt = float(np.mean((v_sim_t - voltage_v_ref) ** 2))

    # V(Q) MSE
    _, vg_sim = _norm_grid(result.time_s, result.voltage_v, result.current_a)
    _, vg_ref = _norm_grid(t_rel, voltage_v_ref, current_a)
    ok = ~(np.isnan(vg_sim) | np.isnan(vg_ref))
    loss_vq = float(np.mean((vg_sim[ok] - vg_ref[ok]) ** 2)) if ok.any() else 1e6

    return 0.3 * loss_vt + 1.0 * loss_vq


# ─── Optuna 전역 탐색 ─────────────────────────────────────────────────────────

def run_optuna(model, time_s, current_a, voltage_v, out_dir: Path,
               n_trials: int = N_OPTUNA_TRIALS, seed: int = OPTUNA_SEED):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    hist_path = out_dir / "fit_history.jsonl"
    trial_counter = [0]
    t_start = time.time()

    def objective(trial):
        v = np.array([
            trial.suggest_float(k, BOUNDS[k][0], BOUNDS[k][1])
            for k in PARAM_KEYS
        ])
        loss = compute_loss(v, model, time_s, current_a, voltage_v)
        trial_counter[0] += 1
        elapsed = time.time() - t_start
        log.info(f"  trial {trial_counter[0]:3d}/{n_trials}  loss={loss:.6f}  "
                 f"D_R3m=1e{v[0]:.2f}  D_C2m=1e{v[3]:.2f}  "
                 f"frac_R3m={v[2]:.3f}  elapsed={elapsed:.0f}s")
        # 이력 저장
        row = {"trial": trial_counter[0], "loss": loss, "elapsed_s": round(elapsed, 1)}
        row.update({k: round(float(v[i]), 6) for i, k in enumerate(PARAM_KEYS)})
        with open(hist_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return loss

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    log.info(f"Optuna 완료: best_loss={best.value:.6f}, trial={best.number+1}")
    return np.array([best.params[k] for k in PARAM_KEYS])


# ─── scipy 지역 정제 ──────────────────────────────────────────────────────────

def run_scipy(init_v: np.ndarray, model, time_s, current_a, voltage_v,
              out_dir: Path, max_iter: int = SCIPY_MAX_ITER):
    hist_path = out_dir / "fit_history.jsonl"
    iter_counter = [0]
    t_start = time.time()

    def obj(x):
        x_clipped = np.clip(x, LOWER, UPPER)
        loss = compute_loss(x_clipped, model, time_s, current_a, voltage_v)
        iter_counter[0] += 1
        if iter_counter[0] % 10 == 0:
            elapsed = time.time() - t_start
            log.info(f"  scipy iter {iter_counter[0]:4d}  loss={loss:.6f}  "
                     f"D_R3m=1e{x_clipped[0]:.2f}  D_C2m=1e{x_clipped[3]:.2f}  "
                     f"frac_R3m={x_clipped[2]:.3f}  elapsed={elapsed:.0f}s")
            row = {"scipy_iter": iter_counter[0], "loss": loss,
                   "elapsed_s": round(elapsed, 1)}
            row.update({k: round(float(x_clipped[i]), 6) for i, k in enumerate(PARAM_KEYS)})
            with open(hist_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        return loss

    bounds = list(zip(LOWER, UPPER))
    result = minimize(obj, init_v, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": max_iter, "ftol": 1e-12})
    log.info(f"scipy 완료: loss={result.fun:.6f}, success={result.success}")
    return np.clip(result.x, LOWER, UPPER)


# ─── 결과 저장 ────────────────────────────────────────────────────────────────

def save_results(best_v: np.ndarray, true_json_path: str | Path, out_dir: Path) -> None:
    _, best_d = vec_to_pybamm_params(best_v)

    est = {
        "frac_R3m":        round(float(np.clip(best_d["frac_R3m"], 0.05, 0.95)), 6),
        "frac_C2m":        round(1.0 - float(np.clip(best_d["frac_R3m"], 0.05, 0.95)), 6),
        "log10_D_R3m":     round(best_d["log10_D_R3m"], 4),
        "D_R3m_m2_s":      float(10.0 ** best_d["log10_D_R3m"]),
        "log10_R_R3m":     round(best_d["log10_R_R3m"], 4),
        "R_R3m_m":         float(10.0 ** best_d["log10_R_R3m"]),
        "log10_D_C2m":     round(best_d["log10_D_C2m"], 4),
        "D_C2m_m2_s":      float(10.0 ** best_d["log10_D_C2m"]),
        "log10_R_C2m":     round(best_d["log10_R_C2m"], 4),
        "R_C2m_m":         float(10.0 ** best_d["log10_R_C2m"]),
        "log10_contact_R": round(best_d["log10_contact"], 4),
        "ocp_R3m": {
            "b0": round(best_d["R3m_b0"], 4),
            "b1": round(best_d["R3m_b1"], 4),
            "amps":    [round(best_d[f"R3m_A{i}"], 5) for i in range(N_TANH)],
            "centers": [round(best_d[f"R3m_c{i}"], 5) for i in range(N_TANH)],
            "widths":  [round(abs(best_d[f"R3m_w{i}"]), 5) for i in range(N_TANH)],
        },
        "ocp_C2m": {
            "b0": round(best_d["C2m_b0"], 4),
            "b1": round(best_d["C2m_b1"], 4),
            "amps":    [round(best_d[f"C2m_A{i}"], 5) for i in range(N_TANH)],
            "centers": [round(best_d[f"C2m_c{i}"], 5) for i in range(N_TANH)],
            "widths":  [round(abs(best_d[f"C2m_w{i}"]), 5) for i in range(N_TANH)],
        },
    }
    (out_dir / "best_params.json").write_text(json.dumps(est, indent=2))

    # 진실값과 비교
    true_d = json.loads(Path(true_json_path).read_text())
    t = true_d.get("truth", {})
    eff_D_true = 0.333 * 10.0 ** t["log10_D_R3m"] + 0.667 * 10.0 ** t["log10_D_C2m"]
    eff_R_true = 1.5e-7  # both phases equal

    comp = {
        "note": "effective model fitting vs native 2-phase truth",
        "frac_R3m": {
            "estimated": est["frac_R3m"],
            "true": t.get("frac_R3m"),
        },
        "log10_D_R3m": {
            "estimated": est["log10_D_R3m"],
            "true": round(t.get("log10_D_R3m", 0), 4),
        },
        "log10_D_C2m": {
            "estimated": est["log10_D_C2m"],
            "true": round(t.get("log10_D_C2m", 0), 4),
        },
        "log10_R_R3m": {
            "estimated": est["log10_R_R3m"],
            "true": round(t.get("log10_R_R3m", 0), 4),
        },
        "log10_R_C2m": {
            "estimated": est["log10_R_C2m"],
            "true": round(t.get("log10_R_C2m", 0), 4),
        },
        "effective_D_eff": {
            "estimated": est["frac_R3m"] * 10.0 ** est["log10_D_R3m"]
                         + est["frac_C2m"] * 10.0 ** est["log10_D_C2m"],
            "true_arithmetic": eff_D_true,
            "note": "frac*D_R3m + (1-frac)*D_C2m",
        },
    }
    (out_dir / "comparison.json").write_text(json.dumps(comp, indent=2))

    # 콘솔 출력
    print("\n" + "=" * 60)
    print("추정 결과 vs 진실값")
    print("=" * 60)
    print(f"{'파라미터':<22} {'추정값':>14}  {'진실값':>14}")
    print("-" * 60)
    rows = [
        ("frac_R3m",    est["frac_R3m"],           t.get("frac_R3m")),
        ("log10_D_R3m", est["log10_D_R3m"],         round(t.get("log10_D_R3m",0),4)),
        ("log10_D_C2m", est["log10_D_C2m"],         round(t.get("log10_D_C2m",0),4)),
        ("log10_R_R3m", est["log10_R_R3m"],         round(t.get("log10_R_R3m",0),4)),
        ("log10_R_C2m", est["log10_R_C2m"],         round(t.get("log10_R_C2m",0),4)),
    ]
    for name, est_val, true_val in rows:
        print(f"  {name:<20} {str(est_val):>14}  {str(true_val):>14}")
    print("=" * 60)
    print(f"  effective D_eff (추정) : {comp['effective_D_eff']['estimated']:.4e}")
    print(f"  effective D_eff (진실) : {comp['effective_D_eff']['true_arithmetic']:.4e}")
    print("=" * 60)


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Stage 1 역추정 시작 ===")
    log.info(f"데이터: {args.data_csv}")
    log.info(f"출력:   {out_dir}")

    # 데이터 로드
    time_s, current_a, voltage_v = load_profile(args.data_csv)

    # 모델 빌드 (한 번만)
    log.info("PyBaMM SPM 모델 초기화...")
    model = build_model()

    # 기존 이력 파일 초기화
    hist_path = out_dir / "fit_history.jsonl"
    hist_path.write_text("")

    # 1. Optuna 전역 탐색
    log.info(f"\n[1/2] Optuna TPE 전역 탐색 (n_trials={args.n_optuna}) ...")
    best_optuna = run_optuna(model, time_s, current_a, voltage_v,
                             out_dir, n_trials=args.n_optuna)

    # 2. scipy 지역 정제
    log.info(f"\n[2/2] scipy L-BFGS-B 지역 정제 (max_iter={args.n_scipy}) ...")
    best_final = run_scipy(best_optuna, model, time_s, current_a, voltage_v,
                           out_dir, max_iter=args.n_scipy)

    # 결과 저장 및 출력
    save_results(best_final, args.true_params, out_dir)
    log.info(f"\n결과 저장 완료: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 역추정 실행")
    parser.add_argument("--data-csv",    default=DATA_CSV)
    parser.add_argument("--true-params", default=TRUE_PARAMS_JSON)
    parser.add_argument("--out-dir",     default="data/fit_results/stage1_gaussianc2m")
    parser.add_argument("--n-optuna",    type=int, default=N_OPTUNA_TRIALS)
    parser.add_argument("--n-scipy",     type=int, default=SCIPY_MAX_ITER)
    args = parser.parse_args()
    main(args)
