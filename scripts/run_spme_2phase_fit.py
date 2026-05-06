"""LMR 2상 SPMe 역추정 스크립트.

DFN으로 생성된 가상 측정 데이터를 입력받아 PyBaMM SPMe 모델로
frac, D, R, Gaussian OCP(center/sigma)를 추정한다.

모델-형식 불일치(model-form mismatch):
  truth (data source) : DFN  — 전해액 농도 분포, 전극 내 전류 분포 포함
  fit model           : SPMe — 전해액 1차 근사 (평균 전해액 농도 사용)

이 불일치는 의도적이다:
  실제 실험 데이터는 DFN 수준의 복잡한 물리를 포함하지만,
  역추정에는 계산 비용이 저렴한 SPMe를 사용하는 것이 현실적이다.
  SPMe는 SPM보다 전해액 저항/농도 효과를 1차 근사로 포함하여
  고율(high C-rate)에서의 전압 강하를 더 정확히 모사한다.

탐색 공간 (문헌 기반 LMR):
  R3m D: 10⁻¹⁸~10⁻¹⁵ m²/s  (문헌: 10⁻¹⁶~10⁻¹⁷)
  C2m D: 10⁻²⁰~10⁻¹⁷ m²/s  (문헌: 10⁻¹⁸~10⁻¹⁹)
  R3m OCP center: 3.40~4.30 V (문헌: 3.6~3.8V 방전 피크)
  C2m OCP center: 2.60~3.30 V (문헌: 3.2~3.3V 방전 피크)

사용법:
  .venv/bin/python scripts/run_spme_2phase_fit.py \\
      --data-csv  data/raw/toyo/lmr_dfn_2phase_sample/Toyo_LMR_DFN_2phase_*.csv \\
      --true-params data/raw/toyo/lmr_dfn_2phase_sample/true_lmr_dfn_parameters.json \\
      --out-dir   data/fit_results/spme_2phase \\
      [--cycles 1,2,3,4]  [--n-optuna 80]  [--n-scipy 500]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import multiprocessing
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))


# ─── 의존 모듈 로드 ───────────────────────────────────────────────────────────

def _load_gen():
    """generate_lmr_dfn_2phase_sample 의 OCP/params 빌더를 임포트."""
    spec = importlib.util.spec_from_file_location(
        "lmr_dfn_gen",
        _HERE / "generate_lmr_dfn_2phase_sample.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── 파라미터 공간 ────────────────────────────────────────────────────────────

PARAM_DEFS = [
    # (key, lo, hi, description)
    ("frac_R3m",         0.10,  0.90,  "R3m 활물질 분율"),
    ("log10_D_R3m",     -18.0, -15.0,  "R3m log10 D [m²/s]  (문헌: -17~-16)"),
    ("log10_R_R3m",      -7.5,  -5.5,  "R3m log10 입자반경 [m]"),
    ("log10_D_C2m",     -20.0, -17.0,  "C2m log10 D [m²/s]  (문헌: -19~-18)"),
    ("log10_R_C2m",      -7.5,  -5.5,  "C2m log10 입자반경 [m]"),
    # 전압 범위를 R3m(고전압)/C2m(저전압)으로 분리하여 위상 swap 방지
    ("R3m_center_v",     3.40,  4.30,  "R3m OCP Gaussian 중심 [V]  (문헌: 3.6~3.8V)"),
    ("R3m_sigma_v",      0.03,  0.45,  "R3m OCP Gaussian σ [V]"),
    ("C2m_center_v",     2.60,  3.30,  "C2m OCP Gaussian 중심 [V]  (문헌: 3.2~3.3V)"),
    ("C2m_sigma_v",      0.05,  0.50,  "C2m OCP Gaussian σ [V]"),
    ("log10_contact_R", -5.00, -1.00,  "log10 접촉저항 [Ω·m²]"),
]

PARAM_KEYS = [d[0] for d in PARAM_DEFS]
LOWER = np.array([d[1] for d in PARAM_DEFS])
UPPER = np.array([d[2] for d in PARAM_DEFS])

# 고정 물리 상수
NOM_CAP_AH    = 0.005
C_S_0_FRAC    = 0.99
N_GRID        = 512

# SPMe 모델 옵션 (스레드 워커가 모델을 독립적으로 초기화할 때 사용)
SPME_OPTIONS = {
    "working electrode": "positive",
    "particle phases":   ("1", "2"),
    "particle":          "Fickian diffusion",
    "particle size":     "single",
    "surface form":      "differential",
}


# ─── 실시간 모니터 ────────────────────────────────────────────────────────────

class Monitor:
    def __init__(self, model_desc: str, data_desc: str,
                 n_optuna: int, n_scipy: int, true_params: dict):
        self.model_desc  = model_desc
        self.data_desc   = data_desc
        self.n_optuna    = n_optuna
        self.n_scipy     = n_scipy
        self.true_params = true_params or {}
        self.best_loss   = float("inf")
        self.best_params: dict = {}
        self.n_trial     = 0
        self.phase       = "Optuna"
        self.t_start     = time.time()
        self._lock       = threading.Lock()
        self._print_header()

    def _print_header(self):
        sep = "=" * 76
        print(sep)
        print(f"  피팅 모델  : {self.model_desc}")
        print(f"  데이터 원천: DFN (truth)  →  SPMe (fit model)  [model-form mismatch]")
        print(f"  데이터     : {self.data_desc}")
        print()
        print(f"  탐색 공간 (문헌 기반 LMR):")
        for key, lo, hi, desc in PARAM_DEFS:
            tv = self.true_params.get(key)
            tv_str = f"  ← 진실값={tv:.4f}" if isinstance(tv, float) else ""
            print(f"    {key:<22} [{lo:>7.2f}, {hi:>7.2f}]   {desc}{tv_str}")
        print(sep)
        print(f"  {'단계':<10} {'trial':>6}  {'loss':>12}  "
              f"{'D_R3m':>10}  {'D_C2m':>10}  "
              f"{'R3m_ctr':>8}  {'C2m_ctr':>8}  {'frac':>7}")
        print("-" * 76)

    def update(self, loss: float, params: dict):
        with self._lock:
            self.n_trial += 1
            improved = loss < self.best_loss
            if improved:
                self.best_loss   = loss
                self.best_params = params.copy()

            D_r  = 10.0 ** params.get("log10_D_R3m", -16)
            D_c  = 10.0 ** params.get("log10_D_C2m", -18)
            cr   = params.get("R3m_center_v", 0.0)
            cc   = params.get("C2m_center_v", 0.0)
            frac = params.get("frac_R3m", 0.0)
            star = "★" if improved else " "
            elapsed = time.time() - self.t_start

            print(
                f"  {self.phase:<10} {self.n_trial:>6}  {loss:>12.6f}  "
                f"{D_r:>10.3e}  {D_c:>10.3e}  "
                f"{cr:>8.3f}V  {cc:>8.3f}V  {frac:>7.4f}  "
                f"{star}  [{elapsed:5.0f}s]",
                flush=True,
            )

    def set_phase(self, phase: str):
        self.phase = phase
        elapsed = time.time() - self.t_start
        print(f"\n{'─'*76}")
        print(f"  >> {phase} 시작  (경과 {elapsed:.0f}s,  현재 최저 loss={self.best_loss:.6f})")
        print(f"{'─'*76}")

    def print_summary(self, true_params: dict | None = None):
        elapsed = time.time() - self.t_start
        tp = true_params or self.true_params
        print(f"\n{'='*76}")
        print(f"  피팅 완료  총 경과 = {elapsed:.0f}s  최종 loss = {self.best_loss:.6f}")
        print(f"\n  {'파라미터':<22} {'추정값':>14}  {'진실값':>14}  {'오차':>10}")
        print(f"  {'-'*60}")
        for key, *_ in PARAM_DEFS:
            est  = self.best_params.get(key, float("nan"))
            true = tp.get(key)
            ts   = f"{true:.4f}" if isinstance(true, float) else "—"
            err  = f"{est - true:+.4f}" if isinstance(true, float) else "—"
            print(f"  {key:<22} {est:>14.4f}  {ts:>14}  {err:>10}")
        print(f"\n  파생 파라미터:")
        b = self.best_params
        D_r = 10.0 ** b.get("log10_D_R3m", -16)
        D_c = 10.0 ** b.get("log10_D_C2m", -18)
        R_r = 10.0 ** b.get("log10_R_R3m", -7)
        R_c = 10.0 ** b.get("log10_R_C2m", -7)
        print(f"    D_R3m = {D_r:.3e} m²/s   (D/R² = {D_r/R_r**2:.3e} s⁻¹)")
        print(f"    D_C2m = {D_c:.3e} m²/s   (D/R² = {D_c/R_c**2:.3e} s⁻¹)")
        print(f"    R_R3m = {R_r:.3e} m")
        print(f"    R_C2m = {R_c:.3e} m")
        if tp.get("log10_D_R3m") and tp.get("log10_D_C2m"):
            tD_r = 10.0 ** tp["log10_D_R3m"]
            tD_c = 10.0 ** tp["log10_D_C2m"]
            tR_r = 10.0 ** tp.get("log10_R_R3m", -6.824)
            tR_c = 10.0 ** tp.get("log10_R_C2m", -6.824)
            print(f"\n  D/R² 비교  (확산 시간스케일):")
            print(f"    R3m  추정={D_r/R_r**2:.3e}  진실={tD_r/tR_r**2:.3e}  "
                  f"비율={D_r/R_r**2 / (tD_r/tR_r**2):.2f}x")
            print(f"    C2m  추정={D_c/R_c**2:.3e}  진실={tD_c/tR_c**2:.3e}  "
                  f"비율={D_c/R_c**2 / (tD_c/tR_c**2):.2f}x")
        print(f"{'='*76}")


# ─── 병렬 워커 ──────────────────────────────────────────────────────────────────

# ── Optuna 스레드 병렬: 각 스레드가 독립 SPMe 모델을 보유 ──
_thread_local = threading.local()
_hist_lock    = threading.Lock()   # fit_history.jsonl 동시 쓰기 보호


def _get_thread_model_gen():
    """스레드-로컬 SPMe 모델과 gen 모듈 반환 (스레드당 1회만 초기화)."""
    if not hasattr(_thread_local, "spme_model"):
        import pybamm as _pybamm
        _thread_local.spme_model = _pybamm.lithium_ion.SPMe(SPME_OPTIONS)
        _thread_local.gen        = _load_gen()
    return _thread_local.spme_model, _thread_local.gen


# ── scipy 사이클 프로세스 풀 (fork): CasADi 컴파일 함수 상속 ──
_w_model = None   # 워커 프로세스에서 사용할 SPMe 모델 (fork 전 부모에서 설정)
_w_gen   = None   # 워커 프로세스에서 사용할 gen 모듈
_POOL: "concurrent.futures.ProcessPoolExecutor | None" = None


def _cycle_worker(payload: tuple) -> float:
    """워커 태스크: 포크된 전역 _w_model/_w_gen 으로 1 사이클 loss 계산."""
    v_clip, time_s, current_a, voltage_ref = payload
    phys, d = _vec_to_phys(v_clip)
    pv = _build_pybamm_params(_w_gen, phys, d)
    return _cycle_loss(_w_model, pv, time_s, current_a, voltage_ref)


def _init_pool(n_jobs: int, spme_model, gen) -> None:
    """fork 기반 프로세스 풀을 초기화한다.

    첫 PyBaMM solve 전에 fork해야 CasADi JIT 컴파일 함수를
    자식 프로세스가 clean하게 상속한다.
    """
    global _w_model, _w_gen, _POOL
    if n_jobs <= 1:
        return
    _w_model = spme_model
    _w_gen   = gen
    ctx  = multiprocessing.get_context("fork")
    _POOL = concurrent.futures.ProcessPoolExecutor(
        max_workers=n_jobs, mp_context=ctx
    )
    print(f"  프로세스 풀 초기화: {n_jobs} workers  (fork, PID={os.getpid()})", flush=True)


# ─── 시뮬레이션 & Loss ────────────────────────────────────────────────────────

@dataclass
class PhysParams:
    frac_R3m: float = 0.40
    frac_C2m: float = 0.60
    D_R3m:    float = 5e-17
    D_C2m:    float = 5e-19
    R_R3m:    float = 1.5e-7
    R_C2m:    float = 1.5e-7
    nominal_capacity_ah: float = NOM_CAP_AH


def _vec_to_phys(v: np.ndarray) -> tuple[PhysParams, dict]:
    d = {k: float(v[i]) for i, k in enumerate(PARAM_KEYS)}
    f = float(np.clip(d["frac_R3m"], 0.05, 0.95))
    return PhysParams(
        frac_R3m=f, frac_C2m=1.0 - f,
        D_R3m=10.0 ** d["log10_D_R3m"],
        D_C2m=10.0 ** d["log10_D_C2m"],
        R_R3m=10.0 ** d["log10_R_R3m"],
        R_C2m=10.0 ** d["log10_R_C2m"],
    ), d


def _build_pybamm_params(gen, phys: PhysParams, d: dict):
    ocp_R3m, ocp_C2m = gen.build_gaussian_ocps(
        r3m_center_v=d["R3m_center_v"],
        c2m_center_v=d["C2m_center_v"],
        r3m_sigma_v=abs(d["R3m_sigma_v"]),
        c2m_sigma_v=abs(d["C2m_sigma_v"]),
    )
    pv = gen.build_params(
        frac_R3m=phys.frac_R3m,
        D_R3m=phys.D_R3m, D_C2m=phys.D_C2m,
        R_R3m=phys.R_R3m, R_C2m=phys.R_C2m,
        ocp_R3m=ocp_R3m, ocp_C2m=ocp_C2m,
        nominal_capacity_ah=NOM_CAP_AH,
        initial_fraction=C_S_0_FRAC,
    )
    # contact resistance
    try:
        pv.update(
            {"Contact resistance [Ohm]": 10.0 ** d["log10_contact_R"]},
            check_already_exists=False,
        )
    except Exception:
        pass
    return pv


def _norm_vq_grid(t, v, i, n=N_GRID) -> np.ndarray:
    cap = np.cumsum(np.abs(i) * np.gradient(t) / 3600.0)
    if cap[-1] < 1e-10:
        return np.full(n, np.nan)
    fn = interp1d(cap / cap[-1], v, bounds_error=False, fill_value="extrapolate")
    return fn(np.linspace(0.0, 1.0, n))


def _cycle_loss(spme_model, pv,
                time_s: np.ndarray,
                current_a: np.ndarray,
                voltage_ref: np.ndarray) -> float:
    from lmro2phase.physics.simulator import run_current_drive
    t_rel  = time_s - time_s[0]
    result = run_current_drive(spme_model, pv, time_s, current_a, t_eval=t_rel)
    if not result.ok:
        return 1e8

    v_sim = interp1d(result.time_s, result.voltage_v,
                     bounds_error=False, fill_value="extrapolate")(t_rel)
    loss_vt = float(np.mean((v_sim - voltage_ref) ** 2))

    vg_sim = _norm_vq_grid(result.time_s, result.voltage_v, result.current_a)
    vg_ref = _norm_vq_grid(t_rel, voltage_ref, current_a)
    ok = ~(np.isnan(vg_sim) | np.isnan(vg_ref))
    loss_vq = float(np.mean((vg_sim[ok] - vg_ref[ok]) ** 2)) if ok.any() else 1e6

    return 0.3 * loss_vt + 1.0 * loss_vq


def compute_loss(v_vec: np.ndarray,
                  gen, spme_model,
                  cycles_data: list[tuple],
                  use_pool: bool = True) -> float:
    """멀티 C-rate 동시 loss (사이클 평균).

    use_pool=True  이고 _POOL 이 초기화된 경우 각 사이클을 별도 프로세스에서 병렬 실행한다.
    use_pool=False (Optuna 스레드 경로): 전달된 gen/spme_model 로 순차 실행한다.
    """
    v_clip = np.clip(v_vec, LOWER, UPPER)

    if use_pool and _POOL is not None:
        try:
            payloads = [(v_clip, t, i, v) for t, i, v in cycles_data]
            losses   = list(_POOL.map(_cycle_worker, payloads))
        except Exception:
            return 1e8
    else:
        try:
            phys, d = _vec_to_phys(v_clip)
            pv = _build_pybamm_params(gen, phys, d)
        except Exception:
            return 1e9
        losses = [_cycle_loss(spme_model, pv, t, i, v) for t, i, v in cycles_data]

    return sum(losses) / max(len(losses), 1)


# ─── Optuna 탐색 ──────────────────────────────────────────────────────────────

def run_optuna(gen, spme_model, cycles_data, monitor, n_trials, seed, out_dir,
               n_jobs: int = 1) -> np.ndarray:
    """Optuna 탐색.

    n_jobs > 1: 각 trial을 별도 스레드에서 실행 (joblib prefer='threads').
    CasADi/SUNDIALS 는 GIL을 해제하므로 실제 병렬 연산이 이뤄진다.
    각 스레드는 _get_thread_model_gen() 을 통해 독립 SPMe 모델을 보유한다.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    hist = out_dir / "fit_history.jsonl"

    phase_label = f"Optuna(×{n_jobs})" if n_jobs > 1 else "Optuna"
    monitor.set_phase(phase_label)
    if n_jobs > 1:
        print(f"  Optuna 스레드 병렬: {n_jobs} workers  (TPE batch mode)", flush=True)

    def objective(trial):
        v = np.array([trial.suggest_float(k, lo, hi) for k, lo, hi, _ in PARAM_DEFS])
        # n_jobs>1 이면 스레드-로컬 모델 사용, 아니면 전달된 모델 사용
        if n_jobs > 1:
            t_model, t_gen = _get_thread_model_gen()
            loss = compute_loss(v, t_gen, t_model, cycles_data, use_pool=False)
        else:
            loss = compute_loss(v, gen, spme_model, cycles_data)
        params = {k: float(v[i]) for i, k in enumerate(PARAM_KEYS)}
        monitor.update(loss, params)
        with _hist_lock, open(hist, "a") as f:
            f.write(json.dumps({"type": "optuna", "loss": loss, **params}) + "\n")
        return loss

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    best = study.best_trial
    return np.array([best.params[k] for k in PARAM_KEYS])


# ─── scipy 정제 ───────────────────────────────────────────────────────────────

def run_scipy(init_v, gen, spme_model, cycles_data, monitor, max_fun, out_dir) -> np.ndarray:
    hist = out_dir / "fit_history.jsonl"
    monitor.set_phase("scipy L-BFGS-B")
    n_eval = [0]

    def obj(x):
        xc = np.clip(x, LOWER, UPPER)
        loss = compute_loss(xc, gen, spme_model, cycles_data)
        n_eval[0] += 1
        params = {k: float(xc[i]) for i, k in enumerate(PARAM_KEYS)}
        if loss < monitor.best_loss or n_eval[0] % 20 == 0:
            monitor.update(loss, params)
        elif loss < monitor.best_loss:
            monitor.best_loss = loss
            monitor.best_params = params.copy()
        with open(hist, "a") as f:
            f.write(json.dumps({"type": "scipy", "loss": loss, **params}) + "\n")
        return loss

    res = minimize(
        obj, init_v, method="L-BFGS-B",
        bounds=list(zip(LOWER, UPPER)),
        options={"maxfun": max_fun, "ftol": 1e-12, "gtol": 1e-8},
    )
    return np.clip(res.x, LOWER, UPPER)


# ─── 결과 저장 ────────────────────────────────────────────────────────────────

def save_results(best_v, monitor, true_json, out_dir):
    d = {k: float(best_v[i]) for i, k in enumerate(PARAM_KEYS)}
    d["frac_R3m"] = float(np.clip(d["frac_R3m"], 0.05, 0.95))
    d["frac_C2m"] = 1.0 - d["frac_R3m"]

    result = {
        "loss":             monitor.best_loss,
        "frac_R3m":         round(d["frac_R3m"], 6),
        "frac_C2m":         round(d["frac_C2m"], 6),
        "log10_D_R3m":      round(d["log10_D_R3m"], 4),
        "D_R3m_m2_s":       10.0 ** d["log10_D_R3m"],
        "log10_R_R3m":      round(d["log10_R_R3m"], 4),
        "R_R3m_m":          10.0 ** d["log10_R_R3m"],
        "log10_D_C2m":      round(d["log10_D_C2m"], 4),
        "D_C2m_m2_s":       10.0 ** d["log10_D_C2m"],
        "log10_R_C2m":      round(d["log10_R_C2m"], 4),
        "R_C2m_m":          10.0 ** d["log10_R_C2m"],
        "R3m_center_v":     round(d["R3m_center_v"], 4),
        "R3m_sigma_v":      round(abs(d["R3m_sigma_v"]), 4),
        "C2m_center_v":     round(d["C2m_center_v"], 4),
        "C2m_sigma_v":      round(abs(d["C2m_sigma_v"]), 4),
        "log10_contact_R":  round(d["log10_contact_R"], 4),
    }
    (out_dir / "best_params.json").write_text(json.dumps(result, indent=2))

    # 진실값 비교
    true_d: dict = {}
    if true_json and Path(true_json).exists():
        raw = json.loads(Path(true_json).read_text())
        t  = raw.get("truth", {})
        tp = raw.get("target_peaks", raw.get("ocp", {}))
        true_d = {
            "frac_R3m":    t.get("frac_R3m"),
            "log10_D_R3m": t.get("log10_D_R3m"),
            "log10_R_R3m": t.get("log10_R_R3m"),
            "log10_D_C2m": t.get("log10_D_C2m"),
            "log10_R_C2m": t.get("log10_R_C2m"),
            "R3m_center_v": tp.get("R3m_primary_redox_feature_v") or tp.get("R3m_center_v"),
            "R3m_sigma_v":  tp.get("R3m_sigma_v"),
            "C2m_center_v": tp.get("C2m_secondary_redox_feature_v") or tp.get("C2m_center_v"),
            "C2m_sigma_v":  tp.get("C2m_sigma_v"),
        }

    comp = {
        "truth_model": "PyBaMM DFN (half-cell, 2-phase positive)",
        "fit_model":   "PyBaMM SPMe (half-cell, 2-phase positive)",
        "note":        "model-form mismatch: DFN truth → SPMe fit",
        "final_loss":  monitor.best_loss,
        "estimated":   result,
        "true":        true_d,
    }
    if true_d:
        comp["errors"] = {
            k: round((result.get(k) or 0) - (true_d.get(k) or 0), 4)
            for k in ["frac_R3m","log10_D_R3m","log10_D_C2m",
                       "R3m_center_v","R3m_sigma_v","C2m_center_v","C2m_sigma_v"]
            if true_d.get(k) is not None
        }
    (out_dir / "comparison.json").write_text(json.dumps(comp, indent=2))
    monitor.print_summary(true_d)


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fit_history.jsonl").write_text("")

    gen = _load_gen()

    # 사이클 파싱 & 데이터 로드
    cycle_ids = [int(c.strip()) for c in str(args.cycles).split(",") if c.strip()]
    from lmro2phase.io.toyo_ascii import parse_toyo
    profile = parse_toyo(args.data_csv)

    cycles_data, cycle_descs = [], []
    for cid in cycle_ids:
        cyc = profile.select_cycle(cid)
        idx = np.arange(0, len(cyc), args.subsample)
        t_s, i_a, v_v = cyc.time_s[idx], cyc.current_a[idx], cyc.voltage_v[idx]
        cycles_data.append((t_s, i_a, v_v))
        cycle_descs.append(f"cycle{cid}(n={len(t_s)}, span={t_s[-1]/3600:.1f}h)")

    data_desc = (
        f"{Path(args.data_csv).parent.name}  "
        f"cycles={cycle_ids}  sub=1/{args.subsample}  "
        + "  ".join(cycle_descs)
    )

    # 진실값 로드
    true_params: dict = {}
    if args.true_params and Path(args.true_params).exists():
        raw = json.loads(Path(args.true_params).read_text())
        t  = raw.get("truth", {})
        tp = raw.get("target_peaks", raw.get("ocp", {}))
        true_params = {
            "frac_R3m":    t.get("frac_R3m"),
            "log10_D_R3m": t.get("log10_D_R3m"),
            "log10_R_R3m": t.get("log10_R_R3m"),
            "log10_D_C2m": t.get("log10_D_C2m"),
            "log10_R_C2m": t.get("log10_R_C2m"),
            "R3m_center_v": tp.get("R3m_primary_redox_feature_v") or tp.get("R3m_center_v"),
            "R3m_sigma_v":  tp.get("R3m_sigma_v"),
            "C2m_center_v": tp.get("C2m_secondary_redox_feature_v") or tp.get("C2m_center_v"),
            "C2m_sigma_v":  tp.get("C2m_sigma_v"),
        }

    model_desc = (
        "PyBaMM SPMe  |  particle phases: ('1','2')  |  "
        "Primary=R3m (고전압), Secondary=C2m (저전압)  |  "
        "Gaussian dQ/dV OCP"
    )
    monitor = Monitor(model_desc, data_desc, args.n_optuna, args.n_scipy, true_params)

    print("  PyBaMM SPMe 2상 모델 초기화...", flush=True)
    spme_model = pybamm.lithium_ion.SPMe(SPME_OPTIONS)
    print("  모델 초기화 완료.")

    # scipy 단계용 사이클 병렬 풀 초기화 (첫 solve 전 fork)
    _init_pool(args.n_cycle_workers, spme_model, gen)
    print()

    try:
        best_optuna = run_optuna(gen, spme_model, cycles_data,
                                  monitor, args.n_optuna, args.seed, out_dir,
                                  n_jobs=args.n_jobs)
        best_final  = run_scipy(best_optuna, gen, spme_model, cycles_data,
                                 monitor, args.n_scipy, out_dir)
        save_results(best_final, monitor, args.true_params, out_dir)
    finally:
        if _POOL is not None:
            _POOL.shutdown(wait=False)

    print(f"\n  출력 경로: {out_dir}")


# ─── pybamm 임포트 (메인에서만) ───────────────────────────────────────────────

import pybamm  # noqa: E402  (generate 스크립트와 동일 환경 보장)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LMR 2상 SPMe 역추정 (DFN truth → SPMe fit)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-csv",
        default="data/raw/toyo/lmr_dfn_2phase_sample/Toyo_LMR_DFN_2phase_0p1C_0p33C_0p5C_1C.csv",
    )
    parser.add_argument(
        "--true-params",
        default="data/raw/toyo/lmr_dfn_2phase_sample/true_lmr_dfn_parameters.json",
    )
    parser.add_argument("--out-dir",   default="data/fit_results/spme_2phase")
    parser.add_argument("--cycles",    default="1,2,3,4",
                        help="피팅 사이클 (쉼표 구분)")
    parser.add_argument("--subsample", type=int, default=600)
    parser.add_argument("--n-optuna",  type=int, default=80)
    parser.add_argument("--n-scipy",   type=int, default=500)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Optuna 병렬 스레드 수 (trial 동시 실행). 1=순차, 20=최대 활용",
    )
    parser.add_argument(
        "--n-cycle-workers", type=int, default=4,
        help="scipy 단계 사이클 병렬 프로세스 수. 1=순차, 4=기본",
    )
    args = parser.parse_args()
    main(args)
