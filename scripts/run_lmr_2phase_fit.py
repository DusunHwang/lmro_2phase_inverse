"""LMR 2상 PyBaMM 역추정 스크립트.

DFN으로 생성된 가상 측정 데이터를 입력받아 PyBaMM SPMe 또는 DFN 모델로
frac, D, R, Gaussian OCP(center/sigma)를 추정한다.

모델 형식:
  truth (data source) : DFN  — 전해액 농도 분포, 전극 내 전류 분포 포함
  fit model           : SPMe 또는 DFN (--fit-model)

탐색 공간 (문헌 기반 LMR):
  R3m D: 10⁻¹⁸~10⁻¹⁵ m²/s  (문헌: 10⁻¹⁶~10⁻¹⁷)
  C2m D: 10⁻²⁰~10⁻¹⁷ m²/s  (문헌: 10⁻¹⁸~10⁻¹⁹)
  R3m OCP center: 3.40~4.30 V (문헌: 3.6~3.8V 방전 피크)
  C2m OCP center: 2.60~3.30 V (문헌: 3.2~3.3V 방전 피크)

사용법:
  .venv/bin/python scripts/run_lmr_2phase_fit.py \\
      --data-csv  data/raw/toyo/lmr_dfn_2phase_sample/Toyo_LMR_DFN_2phase_*.csv \\
      --true-params data/raw/toyo/lmr_dfn_2phase_sample/true_lmr_dfn_parameters.json \\
      --out-dir   data/fit_results/lmr_2phase \\
      --fit-model SPMe \\
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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import re

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
    ("frac_R3m",         0.15,  0.85,  "R3m 활물질 분율"),
    ("log10_D_R3m",     -18.0, -15.0,  "R3m log10 D [m²/s]  (문헌: -17~-16)"),
    ("log10_R_R3m",      -7.5,  -5.8,  "R3m log10 입자반경 [m]"),
    ("log10_D_C2m",     -19.5, -17.0,  "C2m log10 D [m²/s]  (문헌: -19~-18)"),
    ("log10_R_C2m",      -7.5,  -6.0,  "C2m log10 입자반경 [m]"),
    # 전압 범위를 R3m(고전압)/C2m(저전압)으로 분리하여 위상 swap 방지
    ("R3m_center_v",     3.40,  4.00,  "R3m OCP Gaussian 중심 [V]  (문헌: 3.6~3.8V)"),
    ("R3m_sigma_v",      0.06,  0.50,  "R3m OCP Gaussian σ [V]"),
    ("C2m_center_v",     2.90,  3.35,  "C2m OCP Gaussian 중심 [V]  (문헌: 3.2~3.3V)"),
    ("C2m_sigma_v",      0.07,  0.55,  "C2m OCP Gaussian σ [V]"),
    ("log10_contact_R", -4.50, -1.20,  "log10 접촉저항 [Ω·m²]"),
]

PARAM_KEYS = [d[0] for d in PARAM_DEFS]
LOWER = np.array([d[1] for d in PARAM_DEFS])
UPPER = np.array([d[2] for d in PARAM_DEFS])

# 고정 물리 상수
NOM_CAP_AH    = 0.005
C_S_0_FRAC    = 0.99
N_GRID        = 512
DQDV_N_GRID   = 256
LOSS_W_VT     = 0.1
LOSS_W_VQ     = 0.5
LOSS_W_DQDV   = 5.0
FIT_MODEL_TYPE = "SPMe"
KST = timezone(timedelta(hours=9), name="KST")

# PyBaMM fitting 모델 옵션 (스레드 워커가 모델을 독립적으로 초기화할 때 사용)
SPME_OPTIONS = {
    "working electrode": "positive",
    "particle phases":   ("1", "2"),
    "particle":          "Fickian diffusion",
    "particle size":     "single",
    "surface form":      "differential",
}


def _timestamped_out_dir(path: str | Path) -> Path:
    """결과 폴더명 앞에 실행 시점 YYMMDD_hhmm_ prefix를 붙인다."""
    p = Path(path)
    if re.match(r"^\d{6}_\d{4}_", p.name):
        return p
    return p.with_name(datetime.now(KST).strftime("%y%m%d_%H%M_") + p.name)


def _build_fit_model():
    if FIT_MODEL_TYPE == "DFN":
        return pybamm.lithium_ion.DFN(SPME_OPTIONS)
    return pybamm.lithium_ion.SPMe(SPME_OPTIONS)


def _clip_trial_params(params: dict, param_defs: list[tuple] | None = None) -> dict:
    defs = param_defs or PARAM_DEFS
    clipped = {}
    for key, lo, hi, _desc in defs:
        value = params.get(key)
        if value is None:
            continue
        clipped[key] = float(np.clip(float(value), lo, hi))
    return clipped


def _unique_complete_trials(trials: list[dict], param_defs: list[tuple] | None = None) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple] = set()
    for trial in trials:
        clipped = _clip_trial_params(trial, param_defs)
        if any(key not in clipped for key in PARAM_KEYS):
            continue
        sig = tuple(round(clipped[key], 10) for key in PARAM_KEYS)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(clipped)
    return out


def _param_bounds_dict(param_defs: list[tuple]) -> dict:
    return {
        key: {"low": lo, "high": hi, "description": desc}
        for key, lo, hi, desc in param_defs
    }


def _derive_dynamic_param_defs(
    trials,
    base_defs: list[tuple],
    penalty_threshold: float,
    top_fraction: float,
    margin_fraction: float,
    min_width_fraction: float = 0.12,
) -> tuple[list[tuple], dict]:
    valid = [
        t for t in trials
        if t.value is not None
        and np.isfinite(float(t.value))
        and float(t.value) < penalty_threshold
    ]
    valid.sort(key=lambda t: float(t.value))
    if len(valid) < 3:
        return base_defs, {
            "enabled": False,
            "reason": "not_enough_valid_trials",
            "valid_trials": len(valid),
        }

    n_top = max(3, int(np.ceil(len(valid) * float(top_fraction))))
    top = valid[:n_top]
    narrowed = []
    details = {
        "enabled": True,
        "valid_trials": len(valid),
        "top_trials": len(top),
        "penalty_threshold": penalty_threshold,
        "top_fraction": top_fraction,
        "margin_fraction": margin_fraction,
        "bounds": {},
    }

    for key, lo, hi, desc in base_defs:
        vals = np.array([float(t.params[key]) for t in top if key in t.params], dtype=float)
        if len(vals) == 0:
            narrowed.append((key, lo, hi, desc))
            continue

        span = hi - lo
        best_center = float(valid[0].params[key])
        v_lo = float(np.min(vals))
        v_hi = float(np.max(vals))
        std = float(np.std(vals)) if len(vals) > 1 else 0.0
        width = max(std * 3.0, (v_hi - v_lo) * (1.0 + margin_fraction), span * min_width_fraction)
        width = min(width, span * 0.55)
        new_lo = max(lo, best_center - 0.5 * width)
        new_hi = min(hi, best_center + 0.5 * width)

        if new_hi - new_lo < span * min_width_fraction:
            center = 0.5 * (new_lo + new_hi)
            half = 0.5 * span * min_width_fraction
            new_lo = max(lo, center - half)
            new_hi = min(hi, center + half)

        narrowed.append((key, float(new_lo), float(new_hi), desc))
        details["bounds"][key] = {
            "base_low": lo,
            "base_high": hi,
            "dynamic_low": float(new_lo),
            "dynamic_high": float(new_hi),
            "best_center": best_center,
            "top_min": v_lo,
            "top_max": v_hi,
        }

    return narrowed, details


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
        fit_name = "DFN" if "PyBaMM DFN" in self.model_desc else "SPMe"
        print(sep)
        print(f"  피팅 모델  : {self.model_desc}")
        mismatch = "" if fit_name == "DFN" else "  [model-form mismatch]"
        print(f"  데이터 원천: DFN (truth)  →  {fit_name} (fit model){mismatch}")
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

# ── Optuna 스레드 병렬: 각 스레드가 독립 fitting 모델을 보유 ──
_thread_local = threading.local()
_hist_lock    = threading.Lock()   # fit_history.jsonl 동시 쓰기 보호


def _get_thread_model_gen():
    """스레드-로컬 fitting 모델과 gen 모듈 반환 (스레드당 1회만 초기화)."""
    attr = f"{FIT_MODEL_TYPE.lower()}_model"
    if not hasattr(_thread_local, attr):
        setattr(_thread_local, attr, _build_fit_model())
        _thread_local.gen        = _load_gen()
    return getattr(_thread_local, attr), _thread_local.gen


# ── scipy 사이클 프로세스 풀 (fork): CasADi 컴파일 함수 상속 ──
_w_model = None   # 워커 프로세스에서 사용할 fitting 모델 (fork 전 부모에서 설정)
_w_gen   = None   # 워커 프로세스에서 사용할 gen 모듈
_POOL: "concurrent.futures.ProcessPoolExecutor | None" = None


def _cycle_worker(payload: tuple) -> float:
    """워커 태스크: 포크된 전역 _w_model/_w_gen 으로 1 사이클 loss 계산."""
    v_clip, time_s, current_a, voltage_ref = payload
    phys, d = _vec_to_phys(v_clip)
    pv = _build_pybamm_params(_w_gen, phys, d)
    return _cycle_loss(_w_model, pv, time_s, current_a, voltage_ref)


def _thread_cycle_worker(payload: tuple) -> float:
    """Optuna trial 내부 C-rate 병렬용 스레드 태스크."""
    v_clip, time_s, current_a, voltage_ref = payload
    t_model, t_gen = _get_thread_model_gen()
    phys, d = _vec_to_phys(v_clip)
    pv = _build_pybamm_params(t_gen, phys, d)
    return _cycle_loss(t_model, pv, time_s, current_a, voltage_ref)


def _init_pool(n_jobs: int, fit_model, gen) -> None:
    """fork 기반 프로세스 풀을 초기화한다.

    첫 PyBaMM solve 전에 fork해야 CasADi JIT 컴파일 함수를
    자식 프로세스가 clean하게 상속한다.
    """
    global _w_model, _w_gen, _POOL
    if n_jobs <= 1:
        return
    _w_model = fit_model
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


def _branch_dqdv_grid(
    t, v, i, branch: str, n=DQDV_N_GRID
) -> tuple[np.ndarray, np.ndarray]:
    """충전/방전 branch dQ/dV(V)를 공통 전압 grid로 반환한다.

    charge는 양수, discharge는 음수 dQ/dV로 반환한다.
    grid 밖 영역은 NaN으로 두어 overlap 구간만 loss에 사용한다.
    """
    if branch == "charge":
        mask = i > 1e-5
        sign = 1.0
    elif branch == "discharge":
        mask = i < -1e-5
        sign = -1.0
    else:
        raise ValueError(f"unknown branch: {branch}")
    min_pts = 5
    if int(mask.sum()) < min_pts:
        return np.linspace(2.5, 4.65, n), np.full(n, np.nan)

    t_b = np.asarray(t[mask], dtype=float)
    v_b = np.asarray(v[mask], dtype=float)
    i_b = np.asarray(i[mask], dtype=float)
    q_b = np.cumsum(np.abs(i_b) * np.gradient(t_b) / 3600.0) * 1000.0
    if not np.isfinite(q_b[-1]) or q_b[-1] < 1e-6:
        return np.linspace(2.5, 4.65, n), np.full(n, np.nan)

    order = np.argsort(v_b)
    v_s = v_b[order]
    q_s = q_b[order]
    uniq_v, uniq_idx = np.unique(v_s, return_index=True)
    uniq_q = q_s[uniq_idx]
    if len(uniq_v) < min_pts or uniq_v[-1] - uniq_v[0] < 0.05:
        return np.linspace(2.5, 4.65, n), np.full(n, np.nan)

    v_grid = np.linspace(2.5, 4.65, n)
    q_grid = interp1d(
        uniq_v, uniq_q, bounds_error=False, fill_value=np.nan
    )(v_grid)
    ok = np.isfinite(q_grid)
    if int(ok.sum()) < min_pts:
        return v_grid, np.full(n, np.nan)

    q_valid = q_grid[ok]
    win = min(19, max(5, (len(q_valid) // 3) * 2 - 1))
    if len(q_valid) >= 7 and win >= 5 and win > 3:
        q_grid[ok] = savgol_filter(q_valid, win, 3 if win > 3 else 2)

    dqdv = np.full(n, np.nan)
    dqdv[ok] = sign * np.gradient(q_grid[ok], v_grid[ok])
    return v_grid, dqdv


def _discharge_dqdv_grid(t, v, i, n=DQDV_N_GRID) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible discharge-only dQ/dV helper."""
    return _branch_dqdv_grid(t, v, i, "discharge", n=n)


def _dqdv_loss(t_sim, v_sim, i_sim, t_ref, v_ref, i_ref) -> float:
    losses = []
    for branch in ("charge", "discharge"):
        _vg_ref, dqdv_ref = _branch_dqdv_grid(t_ref, v_ref, i_ref, branch)
        _vg_sim, dqdv_sim = _branch_dqdv_grid(t_sim, v_sim, i_sim, branch)
        ok = np.isfinite(dqdv_ref) & np.isfinite(dqdv_sim)
        if int(ok.sum()) < 5:
            losses.append(1e6)
            continue
        scale = float(np.mean(dqdv_ref[ok] ** 2))
        if not np.isfinite(scale) or scale < 1e-12:
            losses.append(1e6)
            continue
        losses.append(float(np.mean((dqdv_sim[ok] - dqdv_ref[ok]) ** 2) / scale))
    return float(np.mean(losses)) if losses else 1e6


def _branch_balanced_indices(time_s, current_a, points_per_branch: int) -> np.ndarray:
    """충전/방전 branch별로 균등 point를 확보하고 rest 경계점을 보존한다."""
    n = len(time_s)
    keep: set[int] = {0, n - 1}
    for mask in (current_a > 1e-5, current_a < -1e-5):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        take = min(points_per_branch, len(idx))
        sel = np.linspace(0, len(idx) - 1, take).round().astype(int)
        keep.update(idx[sel].tolist())

    rest_idx = np.flatnonzero(np.abs(current_a) <= 1e-5)
    if len(rest_idx):
        keep.update(rest_idx[np.linspace(0, len(rest_idx) - 1, min(12, len(rest_idx))).round().astype(int)].tolist())

    # current sign transition 주변점을 보존해 Experiment branch 경계가 흐려지지 않게 한다.
    sign = np.sign(current_a)
    edges = np.flatnonzero(np.diff(sign) != 0)
    for e in edges:
        for j in range(max(0, e - 1), min(n, e + 3)):
            keep.add(j)
    return np.array(sorted(keep), dtype=int)


def _dqdv_point_summary(t_s, v_v, i_a, cycle_id: int, c_rate: float | None) -> dict:
    summary = {
        "cycle": int(cycle_id),
        "c_rate": c_rate,
        "raw_sample_points": int(len(t_s)),
        "charge_points": int((i_a > 1e-5).sum()),
        "discharge_points": int((i_a < -1e-5).sum()),
        "rest_points": int((np.abs(i_a) <= 1e-5).sum()),
        "dqdv_grid_points": int(DQDV_N_GRID),
    }
    for branch in ("charge", "discharge"):
        _vg, dqdv = _branch_dqdv_grid(t_s, v_v, i_a, branch)
        summary[f"{branch}_finite_dqdv_points"] = int(np.isfinite(dqdv).sum())
    return summary


def _cycle_loss(fit_model, pv,
                time_s: np.ndarray,
                current_a: np.ndarray,
                voltage_ref: np.ndarray) -> float:
    from lmro2phase.physics.simulator import run_current_drive
    t_rel  = time_s - time_s[0]
    result = run_current_drive(fit_model, pv, time_s, current_a, t_eval=t_rel)
    if not result.ok:
        return 1e8

    v_sim = interp1d(result.time_s, result.voltage_v,
                     bounds_error=False, fill_value="extrapolate")(t_rel)
    loss_vt = float(np.mean((v_sim - voltage_ref) ** 2))

    vg_sim = _norm_vq_grid(result.time_s, result.voltage_v, result.current_a)
    vg_ref = _norm_vq_grid(t_rel, voltage_ref, current_a)
    ok = ~(np.isnan(vg_sim) | np.isnan(vg_ref))
    loss_vq = float(np.mean((vg_sim[ok] - vg_ref[ok]) ** 2)) if ok.any() else 1e6

    loss_dqdv = _dqdv_loss(
        result.time_s, result.voltage_v, result.current_a,
        t_rel, voltage_ref, current_a,
    )

    return LOSS_W_VT * loss_vt + LOSS_W_VQ * loss_vq + LOSS_W_DQDV * loss_dqdv


def compute_loss(v_vec: np.ndarray,
                  gen, fit_model,
                  cycles_data: list[tuple],
                  use_pool: bool = True) -> float:
    """멀티 C-rate 동시 loss (사이클 평균).

    use_pool=True  이고 _POOL 이 초기화된 경우 각 사이클을 별도 프로세스에서 병렬 실행한다.
    use_pool=False (Optuna 스레드 경로): 전달된 gen/fit_model 로 순차 실행한다.
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
        losses = [_cycle_loss(fit_model, pv, t, i, v) for t, i, v in cycles_data]

    return sum(losses) / max(len(losses), 1)


def compute_loss_threaded_cycles(v_vec: np.ndarray,
                                 cycles_data: list[tuple],
                                 cycle_workers: int) -> float:
    """Optuna trial 내부에서 C-rate별 loss를 스레드로 병렬 계산한다.

    Optuna 자체가 스레드 병렬로 trial을 돌리므로, 여기서는 프로세스 풀을
    공유하지 않고 각 사이클 스레드가 독립 fitting 모델을 보유하도록 한다.
    """
    v_clip = np.clip(v_vec, LOWER, UPPER)
    if cycle_workers <= 1 or len(cycles_data) <= 1:
        t_model, t_gen = _get_thread_model_gen()
        return compute_loss(v_clip, t_gen, t_model, cycles_data, use_pool=False)

    try:
        payloads = [(v_clip, t, i, v) for t, i, v in cycles_data]
        workers = min(int(cycle_workers), len(payloads))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            losses = list(ex.map(_thread_cycle_worker, payloads))
    except Exception:
        return 1e8
    return sum(losses) / max(len(losses), 1)


# ─── Optuna 탐색 ──────────────────────────────────────────────────────────────

def run_optuna(gen, fit_model, cycles_data, monitor, n_trials, seed, out_dir,
               n_jobs: int = 1, cycle_workers: int = 1,
               early_stop_loss: float = 0.0,
               early_stop_patience: int = 0,
               early_stop_min_delta: float = 0.0,
               dynamic_bounds: bool = True,
               dynamic_warmup_trials: int = 30,
               dynamic_top_fraction: float = 0.35,
               dynamic_margin_fraction: float = 0.35,
               penalty_loss_threshold: float = 1e5) -> np.ndarray:
    """Optuna 탐색.

    n_jobs > 1: 각 trial을 별도 스레드에서 실행 (joblib prefer='threads').
    cycle_workers > 1: 각 trial 내부의 C-rate 사이클도 스레드 병렬 실행.
    CasADi/SUNDIALS 는 GIL을 해제하므로 실제 병렬 연산이 이뤄진다.
    각 계산 스레드는 _get_thread_model_gen() 을 통해 독립 fitting 모델을 보유한다.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    hist = out_dir / "fit_history.jsonl"

    phase_label = f"Optuna(×{n_jobs})" if n_jobs > 1 else "Optuna"
    monitor.set_phase(phase_label)
    if n_jobs > 1:
        print(f"  Optuna 스레드 병렬: {n_jobs} workers  (TPE batch mode)", flush=True)
    if cycle_workers > 1:
        print(f"  Optuna trial 내부 C-rate 병렬: {cycle_workers} workers", flush=True)
    if early_stop_loss > 0:
        print(f"  Optuna early stop loss: best loss <= {early_stop_loss:g}", flush=True)
    if early_stop_patience > 0:
        print(
            "  Optuna early stop patience: "
            f"{early_stop_patience} trials without improvement >= {early_stop_min_delta:g}",
            flush=True,
        )

    best_seen = {"value": float("inf"), "no_improve": 0}
    early_stop_lock = threading.Lock()

    def early_stop_callback(study, trial):
        with early_stop_lock:
            best_value = float(study.best_value)
            if early_stop_loss > 0 and best_value <= early_stop_loss:
                print(
                    f"  Optuna early stop: best loss {best_value:.6g} <= "
                    f"{early_stop_loss:.6g}",
                    flush=True,
                )
                study.stop()
                return

            if early_stop_patience <= 0:
                return

            improved = best_value < best_seen["value"] - max(0.0, early_stop_min_delta)
            if improved:
                best_seen["value"] = best_value
                best_seen["no_improve"] = 0
                return

            best_seen["no_improve"] += 1
            if best_seen["no_improve"] >= early_stop_patience:
                print(
                    "  Optuna early stop: "
                    f"{best_seen['no_improve']} completed trials without improvement >= "
                    f"{early_stop_min_delta:g} "
                    f"(best loss={best_seen['value']:.6g})",
                    flush=True,
                )
                study.stop()

    callbacks = [early_stop_callback] if early_stop_loss > 0 or early_stop_patience > 0 else None

    def make_objective(param_defs: list[tuple], stage: str):
        def objective(trial):
            v = np.array([trial.suggest_float(k, lo, hi) for k, lo, hi, _ in param_defs])
            # Optuna 병렬 경로에서는 전역 프로세스 풀 대신 스레드-로컬 모델을 사용한다.
            if n_jobs > 1 or cycle_workers > 1:
                loss = compute_loss_threaded_cycles(v, cycles_data, cycle_workers)
            else:
                loss = compute_loss(v, gen, fit_model, cycles_data)
            params = {k: float(v[i]) for i, k in enumerate(PARAM_KEYS)}
            monitor.update(loss, params)
            with _hist_lock, open(hist, "a") as f:
                f.write(json.dumps({"type": stage, "loss": loss, **params}) + "\n")
            return loss
        return objective

    def make_study(study_seed: int):
        return optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=study_seed),
        )

    studies = []
    dynamic_info = {
        "enabled": bool(dynamic_bounds),
        "base_bounds": _param_bounds_dict(PARAM_DEFS),
        "stages": [],
    }

    use_dynamic = bool(dynamic_bounds) and int(n_trials) > 4
    if use_dynamic:
        warmup = int(dynamic_warmup_trials)
        if warmup <= 0:
            warmup = max(int(n_jobs), min(30, max(4, int(n_trials) // 3)))
        warmup = max(3, min(warmup, int(n_trials) - 1))
        remaining = int(n_trials) - warmup
        print(
            f"  Optuna dynamic bounds: warmup={warmup}, refine={remaining}, "
            f"penalty_threshold={penalty_loss_threshold:g}",
            flush=True,
        )

        study1 = make_study(seed)
        study1.optimize(make_objective(PARAM_DEFS, "optuna_warmup"),
                        n_trials=warmup, n_jobs=n_jobs, callbacks=callbacks)
        studies.append(study1)

        narrowed_defs, stage_info = _derive_dynamic_param_defs(
            study1.trials,
            PARAM_DEFS,
            penalty_loss_threshold,
            dynamic_top_fraction,
            dynamic_margin_fraction,
        )
        dynamic_info["stages"].append(stage_info)
        dynamic_info["refined_bounds"] = _param_bounds_dict(narrowed_defs)
        (out_dir / "dynamic_bounds.json").write_text(json.dumps(dynamic_info, indent=2))

        if remaining > 0 and stage_info.get("enabled"):
            print(
                f"  Optuna dynamic bounds 적용: valid={stage_info['valid_trials']}, "
                f"top={stage_info['top_trials']}",
                flush=True,
            )
            study2 = make_study(seed + 1)
            best_warmup = [
                t.params for t in sorted(
                    [t for t in study1.trials if t.value is not None and t.value < penalty_loss_threshold],
                    key=lambda t: float(t.value),
                )[: min(3, len(study1.trials))]
            ]
            for params in _unique_complete_trials(best_warmup, narrowed_defs):
                study2.enqueue_trial(params)
            study2.optimize(make_objective(narrowed_defs, "optuna_refined"),
                            n_trials=remaining, n_jobs=n_jobs, callbacks=callbacks)
            studies.append(study2)
        elif remaining > 0:
            print("  Optuna dynamic bounds 미적용: 유효 warmup trial 부족", flush=True)
            study1.optimize(make_objective(PARAM_DEFS, "optuna"),
                            n_trials=remaining, n_jobs=n_jobs, callbacks=callbacks)
    else:
        study = make_study(seed)
        study.optimize(make_objective(PARAM_DEFS, "optuna"),
                       n_trials=n_trials, n_jobs=n_jobs, callbacks=callbacks)
        studies.append(study)
        (out_dir / "dynamic_bounds.json").write_text(json.dumps(dynamic_info, indent=2))

    best = min((s.best_trial for s in studies), key=lambda t: float(t.value))
    return np.array([best.params[k] for k in PARAM_KEYS])


# ─── scipy 정제 ───────────────────────────────────────────────────────────────

def run_scipy(init_v, gen, fit_model, cycles_data, monitor, max_fun, out_dir) -> np.ndarray:
    if max_fun <= 0:
        monitor.set_phase("scipy skipped")
        return np.array([monitor.best_params.get(k, init_v[i]) for i, k in enumerate(PARAM_KEYS)])

    hist = out_dir / "fit_history.jsonl"
    monitor.set_phase("scipy L-BFGS-B")
    n_eval = [0]

    def obj(x):
        xc = np.clip(x, LOWER, UPPER)
        loss = compute_loss(xc, gen, fit_model, cycles_data)
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
    best_seen = np.array([monitor.best_params.get(k, res.x[i]) for i, k in enumerate(PARAM_KEYS)])
    return np.clip(best_seen, LOWER, UPPER)


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
        "fit_model":   f"PyBaMM {FIT_MODEL_TYPE} (half-cell, 2-phase positive)",
        "note":        (
            "same-form fit: DFN truth -> DFN fit"
            if FIT_MODEL_TYPE == "DFN"
            else "model-form mismatch: DFN truth -> SPMe fit"
        ),
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
    global LOSS_W_VT, LOSS_W_VQ, LOSS_W_DQDV, FIT_MODEL_TYPE
    LOSS_W_VT = float(args.loss_w_vt)
    LOSS_W_VQ = float(args.loss_w_vq)
    LOSS_W_DQDV = float(args.loss_w_dqdv)
    FIT_MODEL_TYPE = str(args.fit_model)

    out_dir = _timestamped_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fit_history.jsonl").write_text("")
    optuna_cycle_workers = max(1, int(args.optuna_cycle_workers))
    parallel_config = {
        "optuna_trial_workers": int(args.n_jobs),
        "optuna_cycle_workers_per_trial": optuna_cycle_workers,
        "scipy_cycle_workers": int(args.n_cycle_workers),
        "cycles": str(args.cycles),
        "subsample": int(args.subsample),
        "branch_points_per_charge_discharge": int(args.branch_points),
        "n_optuna": int(args.n_optuna),
        "n_scipy": int(args.n_scipy),
        "optuna_early_stop": {
            "loss_threshold": float(args.early_stop_loss),
            "patience_trials": int(args.early_stop_patience),
            "min_delta": float(args.early_stop_min_delta),
        },
        "optuna_dynamic_bounds": {
            "enabled": not bool(args.disable_dynamic_bounds),
            "warmup_trials": int(args.dynamic_bounds_warmup),
            "top_fraction": float(args.dynamic_bounds_top_fraction),
            "margin_fraction": float(args.dynamic_bounds_margin_fraction),
            "penalty_loss_threshold": float(args.penalty_loss_threshold),
        },
        "fit_model": FIT_MODEL_TYPE,
        "loss_weights": {
            "Vt_MSE": LOSS_W_VT,
            "VQ_MSE": LOSS_W_VQ,
            "dQdV_MSE_normalized": LOSS_W_DQDV,
        },
        "note": "Optuna는 trial 병렬과 trial 내부 C-rate 병렬을 모두 사용한다.",
    }
    (out_dir / "parallel_config.json").write_text(json.dumps(parallel_config, indent=2))

    gen = _load_gen()

    # 사이클 파싱 & 데이터 로드
    cycle_ids = [int(c.strip()) for c in str(args.cycles).split(",") if c.strip()]
    from lmro2phase.io.toyo_ascii import parse_toyo
    profile = parse_toyo(args.data_csv)

    cycles_data, cycle_descs = [], []
    default_c_rates = [0.1, 0.33, 0.5, 1.0]
    dqdv_profile_points = []
    for cid in cycle_ids:
        cyc = profile.select_cycle(cid)
        if args.branch_points > 0:
            idx = _branch_balanced_indices(cyc.time_s, cyc.current_a, args.branch_points)
        else:
            idx = np.arange(0, len(cyc), args.subsample)
        t_s, i_a, v_v = cyc.time_s[idx], cyc.current_a[idx], cyc.voltage_v[idx]
        cycles_data.append((t_s, i_a, v_v))
        cycle_descs.append(
            f"cycle{cid}(n={len(t_s)}, chg={(i_a > 1e-5).sum()}, "
            f"dchg={(i_a < -1e-5).sum()}, span={t_s[-1]/3600:.1f}h)"
        )
        dqdv_profile_points.append(_dqdv_point_summary(
            t_s, v_v, i_a, cid,
            default_c_rates[cid - 1] if 1 <= cid <= len(default_c_rates) else None,
        ))
    parallel_config["dqdv_profile_points"] = dqdv_profile_points
    (out_dir / "parallel_config.json").write_text(json.dumps(parallel_config, indent=2))

    data_desc = (
        f"{Path(args.data_csv).parent.name}  "
        f"cycles={cycle_ids}  branch_points={args.branch_points}  sub=1/{args.subsample}  "
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
    parallel_config["parameter_bounds"] = _param_bounds_dict(PARAM_DEFS)
    (out_dir / "parallel_config.json").write_text(json.dumps(parallel_config, indent=2))

    model_desc = (
        f"PyBaMM {FIT_MODEL_TYPE}  |  particle phases: ('1','2')  |  "
        "Primary=R3m (고전압), Secondary=C2m (저전압)  |  "
        "Gaussian dQ/dV OCP"
    )
    monitor = Monitor(model_desc, data_desc, args.n_optuna, args.n_scipy, true_params)

    print(f"  PyBaMM {FIT_MODEL_TYPE} 2상 모델 초기화...", flush=True)
    fit_model = _build_fit_model()
    print("  모델 초기화 완료.")

    # scipy 단계용 사이클 병렬 풀 초기화 (첫 solve 전 fork)
    _init_pool(args.n_cycle_workers, fit_model, gen)
    print()

    try:
        best_optuna = run_optuna(gen, fit_model, cycles_data,
                                  monitor, args.n_optuna, args.seed, out_dir,
                                  n_jobs=args.n_jobs,
                                  cycle_workers=optuna_cycle_workers,
                                  early_stop_loss=args.early_stop_loss,
                                  early_stop_patience=args.early_stop_patience,
                                  early_stop_min_delta=args.early_stop_min_delta,
                                  dynamic_bounds=not args.disable_dynamic_bounds,
                                  dynamic_warmup_trials=args.dynamic_bounds_warmup,
                                  dynamic_top_fraction=args.dynamic_bounds_top_fraction,
                                  dynamic_margin_fraction=args.dynamic_bounds_margin_fraction,
                                  penalty_loss_threshold=args.penalty_loss_threshold)
        best_final  = run_scipy(best_optuna, gen, fit_model, cycles_data,
                                 monitor, args.n_scipy, out_dir)
        save_results(best_final, monitor, args.true_params, out_dir)
        parallel_config["elapsed_seconds"] = round(time.time() - monitor.t_start, 3)
        (out_dir / "parallel_config.json").write_text(json.dumps(parallel_config, indent=2))
    finally:
        if _POOL is not None:
            _POOL.shutdown(wait=False)

    print(f"\n  출력 경로: {out_dir}")


# ─── pybamm 임포트 (메인에서만) ───────────────────────────────────────────────

import pybamm  # noqa: E402  (generate 스크립트와 동일 환경 보장)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LMR 2상 PyBaMM 역추정 (DFN truth → SPMe/DFN fit)",
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
    parser.add_argument("--out-dir",   default="data/fit_results/lmr_2phase")
    parser.add_argument("--fit-model", default="SPMe", choices=["SPMe", "DFN"],
                        help="역추정에 사용할 PyBaMM 모델")
    parser.add_argument("--cycles",    default="1,2,3,4",
                        help="피팅 사이클 (쉼표 구분)")
    parser.add_argument("--subsample", type=int, default=600)
    parser.add_argument(
        "--branch-points", type=int, default=200,
        help="각 cycle의 충전/방전 branch별 목표 샘플 수. 0이면 기존 균일 subsample 사용",
    )
    parser.add_argument("--n-optuna",  type=int, default=80)
    parser.add_argument("--n-scipy",   type=int, default=500)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument(
        "--early-stop-loss", type=float, default=0.0,
        help="Optuna best loss가 이 값 이하가 되면 중지. 0이면 비활성화",
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=0,
        help="Optuna best loss 개선이 없는 trial 수. 0이면 patience 중지 비활성화",
    )
    parser.add_argument(
        "--early-stop-min-delta", type=float, default=0.0,
        help="patience 판단 시 개선으로 인정할 최소 loss 감소량",
    )
    parser.add_argument(
        "--disable-dynamic-bounds", action="store_true",
        help="warmup trial 기반 동적 bounds 축소를 끄고 단일 bounds로 Optuna 실행",
    )
    parser.add_argument(
        "--dynamic-bounds-warmup", type=int, default=30,
        help="동적 bounds 계산 전 warmup trial 수. 0이면 n_optuna 기반 자동 설정",
    )
    parser.add_argument(
        "--dynamic-bounds-top-fraction", type=float, default=0.35,
        help="warmup 유효 trial 중 bounds 축소에 사용할 상위 loss 비율",
    )
    parser.add_argument(
        "--dynamic-bounds-margin-fraction", type=float, default=0.35,
        help="상위 trial min/max 주변에 추가할 bounds margin 비율",
    )
    parser.add_argument(
        "--penalty-loss-threshold", type=float, default=1e5,
        help="동적 bounds 계산에서 solver 실패/penalty trial로 제외할 loss 기준",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Optuna 병렬 스레드 수 (trial 동시 실행). 1=순차, 20=최대 활용",
    )
    parser.add_argument(
        "--n-cycle-workers", type=int, default=4,
        help="scipy 단계 사이클 병렬 프로세스 수. 1=순차, 4=기본",
    )
    parser.add_argument(
        "--optuna-cycle-workers", type=int, default=4,
        help="Optuna 각 trial 내부의 C-rate 병렬 스레드 수. 1=trial 내부 순차, 4=4개 C-rate 동시",
    )
    parser.add_argument("--loss-w-vt", type=float, default=LOSS_W_VT,
                        help="V(t) MSE loss 가중치")
    parser.add_argument("--loss-w-vq", type=float, default=LOSS_W_VQ,
                        help="V(Q) MSE loss 가중치")
    parser.add_argument("--loss-w-dqdv", type=float, default=LOSS_W_DQDV,
                        help="normalized dQ/dV(V) MSE loss 가중치")
    args = parser.parse_args()
    main(args)
