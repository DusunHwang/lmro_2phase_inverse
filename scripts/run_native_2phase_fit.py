"""Native 2-phase 역추정 스크립트.

가상 측정 TOYO CSV를 입력받아 PyBaMM native 2-phase SPM 모델로
R, diffusivity, Gaussian OCP (center/sigma)를 추정합니다.

피팅 모델:
  - PyBaMM SPM (particle phases: ("1","2"))  — Primary=R3m, Secondary=C2m
  - 각 phase 독립 확산 경로 (D_R3m, R_R3m) / (D_C2m, R_C2m) 분리
  - OCP: Gaussian dQ/dV 기반 보간 OCP (center_v, sigma_v per phase)
  - 전압 범위 분리: R3m_center_v ∈ [3.40, 4.30] V (고전압), C2m_center_v ∈ [2.60, 3.30] V (저전압)
  - 멀티 C-rate 동시 피팅: 각 사이클별 loss 합산 → 확산계수 식별성 향상

사용법:
  python scripts/run_native_2phase_fit.py \\
      --data-csv  data/raw/toyo/<sample>/Toyo_LMR_*.csv \\
      --out-dir   data/fit_results/native2phase/<run_name> \\
      [--true-params  data/raw/toyo/<sample>/true_native_2phase_parameters.json] \\
      [--cycles 1,2,3,4]  [--n-optuna 60]  [--n-scipy 400]  [--subsample 600]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# ─── 프로젝트 패키지 경로 ─────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))


# ─── 의존 모듈 지연 임포트 ───────────────────────────────────────────────────

def _load_gen():
    """generate_toyo_native_2phase_sample 스크립트를 모듈로 임포트."""
    spec = importlib.util.spec_from_file_location(
        "native2phase_gen",
        _HERE / "generate_toyo_native_2phase_sample.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── 파라미터 공간 정의 ───────────────────────────────────────────────────────

PARAM_DEFS = [
    # (key,              lo,     hi,   description)
    ("frac_R3m",         0.10,   0.90, "R3m active fraction"),
    ("log10_D_R3m",     -18.0, -14.0, "R3m log10 diffusivity [m²/s]"),
    ("log10_R_R3m",      -7.5,  -5.5, "R3m log10 particle radius [m]"),
    ("log10_D_C2m",     -19.0, -15.0, "C2m log10 diffusivity [m²/s]"),
    ("log10_R_C2m",      -7.5,  -5.5, "C2m log10 particle radius [m]"),
    # 전압 범위를 고전압(R3m)/저전압(C2m)으로 분리해 swap 방지
    ("R3m_center_v",     3.40,   4.30, "R3m OCP Gaussian center [V]  (고전압 phase)"),
    ("R3m_sigma_v",      0.03,   0.45, "R3m OCP Gaussian sigma [V]"),
    ("C2m_center_v",     2.60,   3.30, "C2m OCP Gaussian center [V]  (저전압 phase)"),
    ("C2m_sigma_v",      0.05,   0.50, "C2m OCP Gaussian sigma [V]"),
    ("log10_contact_R", -5.00,  -1.00, "log10 contact resistance [Ω·m²]"),
]

PARAM_KEYS = [d[0] for d in PARAM_DEFS]
LOWER = np.array([d[1] for d in PARAM_DEFS])
UPPER = np.array([d[2] for d in PARAM_DEFS])

# 고정 물리 상수
C_S_MAX       = 47500.0     # mol/m³
ELECTRODE_THK = 75.0e-6    # m
ACTIVE_TOTAL  = 0.665
NOM_CAP_AH    = 0.005       # Ah
C_S_0_FRAC    = 0.99        # 초기 stoichiometry (충전 직전 완방전)

N_GRID = 512


# ─── Rich 모니터 ─────────────────────────────────────────────────────────────

class Monitor:
    """Rich 기반 실시간 모니터. rich 없으면 일반 print로 fallback."""

    def __init__(self, model_desc: str, data_desc: str,
                 n_optuna: int, n_scipy: int, true_params: dict | None):
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.live import Live
            from rich.panel import Panel
            from rich.layout import Layout
            self._rich = True
            self._console = Console()
        except ImportError:
            self._rich = False
            self._console = None

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

        self._print_header()

    def _print_header(self):
        elapsed = ""
        sep = "=" * 72
        print(sep)
        print(f"  피팅 모델  : {self.model_desc}")
        print(f"  데이터     : {self.data_desc}")
        print(f"  탐색 공간  :")
        for key, lo, hi, desc in PARAM_DEFS:
            print(f"    {key:<22} [{lo:>7.2f}, {hi:>7.2f}]   {desc}")
        print(sep)
        print(f"  {'단계':<10} {'trial':>6}  {'loss':>12}  "
              f"{'D_R3m':>10}  {'D_C2m':>10}  "
              f"{'R3m_ctr':>8}  {'C2m_ctr':>8}  {'frac_R3m':>9}")
        print("-" * 72)

    def update(self, loss: float, params: dict):
        self.n_trial += 1
        elapsed = time.time() - self.t_start
        improved = loss < self.best_loss
        if improved:
            self.best_loss   = loss
            self.best_params = params.copy()
            star = "★"
        else:
            star = " "

        D_R3m = 10.0 ** params.get("log10_D_R3m", -16)
        D_C2m = 10.0 ** params.get("log10_D_C2m", -18)
        ctr_r = params.get("R3m_center_v", 0.0)
        ctr_c = params.get("C2m_center_v", 0.0)
        frac  = params.get("frac_R3m",     0.0)

        line = (
            f"  {self.phase:<10} {self.n_trial:>6}  {loss:>12.6f}  "
            f"{D_R3m:>10.3e}  {D_C2m:>10.3e}  "
            f"{ctr_r:>8.3f}V  {ctr_c:>8.3f}V  {frac:>9.4f}  "
            f"{star}  [{elapsed:5.0f}s]"
        )
        print(line, flush=True)

    def set_phase(self, phase: str):
        self.phase = phase
        elapsed = time.time() - self.t_start
        print(f"\n{'─'*72}")
        print(f"  >> {phase} 시작 (경과 {elapsed:.0f}s)")
        print(f"{'─'*72}")

    def print_summary(self, true_params: dict | None = None):
        elapsed = time.time() - self.t_start
        print(f"\n{'=' * 72}")
        print(f"  피팅 완료  total elapsed = {elapsed:.0f}s")
        print(f"  최종 loss : {self.best_loss:.6f}")
        print(f"\n  {'파라미터':<22} {'추정값':>14}  {'진실값':>14}")
        print(f"  {'-'*52}")

        tp = true_params or self.true_params
        for key, lo, hi, desc in PARAM_DEFS:
            est_val = self.best_params.get(key, float("nan"))
            true_val = tp.get(key, "—")
            if isinstance(true_val, float):
                true_str = f"{true_val:.4f}"
            else:
                true_str = str(true_val)
            print(f"  {key:<22} {est_val:>14.4f}  {true_str:>14}")

        print(f"\n  파생 파라미터:")
        frac  = self.best_params.get("frac_R3m", 0.5)
        D_r   = 10.0 ** self.best_params.get("log10_D_R3m", -16)
        D_c   = 10.0 ** self.best_params.get("log10_D_C2m", -18)
        R_r   = 10.0 ** self.best_params.get("log10_R_R3m", -7)
        R_c   = 10.0 ** self.best_params.get("log10_R_C2m", -7)
        print(f"    D_R3m      = {D_r:.3e} m²/s")
        print(f"    D_C2m      = {D_c:.3e} m²/s")
        print(f"    R_R3m      = {R_r:.3e} m")
        print(f"    R_C2m      = {R_c:.3e} m")

        if tp:
            tD_r = 10.0 ** tp.get("log10_D_R3m", -16)
            tD_c = 10.0 ** tp.get("log10_D_C2m", -18)
            tR   = 10.0 ** tp.get("log10_R_R3m", -7)
            print(f"\n  진실값 비교:")
            print(f"    D_R3m  추정/진실 = {D_r:.3e} / {tD_r:.3e}  (비율 {D_r/tD_r:.2f}x)")
            print(f"    D_C2m  추정/진실 = {D_c:.3e} / {tD_c:.3e}  (비율 {D_c/tD_c:.2f}x)")
            print(f"    R_R3m  추정/진실 = {R_r:.3e} / {tR:.3e}  (비율 {R_r/tR:.2f}x)")
        print(f"{'=' * 72}")


# ─── 시뮬레이션 인프라 ────────────────────────────────────────────────────────

@dataclass
class PhysParams:
    nominal_capacity_ah: float = NOM_CAP_AH
    capacity_scale: float = 1.0
    frac_R3m: float = 0.333
    frac_C2m: float = 0.667
    D_R3m: float = 4.59e-17
    D_C2m: float = 1.0e-18
    R_R3m: float = 1.5e-7
    R_C2m: float = 1.5e-7


def _vec_to_phys(v: np.ndarray) -> tuple[PhysParams, dict]:
    d = {k: float(v[i]) for i, k in enumerate(PARAM_KEYS)}
    frac_R3m = float(np.clip(d["frac_R3m"], 0.05, 0.95))
    return PhysParams(
        frac_R3m=frac_R3m,
        frac_C2m=1.0 - frac_R3m,
        D_R3m=10.0 ** d["log10_D_R3m"],
        D_C2m=10.0 ** d["log10_D_C2m"],
        R_R3m=10.0 ** d["log10_R_R3m"],
        R_C2m=10.0 ** d["log10_R_C2m"],
    ), d


def _build_pybamm_params(gen, truth: PhysParams, d: dict):
    """Gaussian OCP + native 2-phase params."""
    import pybamm

    ocp_R3m, ocp_C2m = gen.gaussian_redox_native_ocps(
        r3m_center_v  = d["R3m_center_v"],
        c2m_center_v  = d["C2m_center_v"],
        r3m_sigma_v   = abs(d["R3m_sigma_v"]),
        c2m_sigma_v   = abs(d["C2m_sigma_v"]),
    )

    param = gen.build_native_params(truth, ocp_R3m, ocp_C2m,
                                     initial_fraction=C_S_0_FRAC)
    # contact resistance 덮어쓰기
    # Chen2020 base에는 없지만 lmr_parameter_set 방식으로 적용
    # (half-cell SPM에서 contact 저항 항목이 없으면 0으로 처리)
    contact = 10.0 ** d["log10_contact_R"]
    try:
        param.update(
            {"Contact resistance [Ohm]": contact},
            check_already_exists=False,
        )
    except Exception:
        pass  # 파라미터 셋에 해당 키 없으면 무시
    return param


# ─── Loss 계산 ────────────────────────────────────────────────────────────────

def _norm_vq_grid(t, v, i, n=N_GRID) -> np.ndarray:
    cap = np.cumsum(np.abs(i) * np.gradient(t) / 3600.0)
    if cap[-1] < 1e-10:
        return np.full(n, np.nan)
    fn = interp1d(cap / cap[-1], v, bounds_error=False, fill_value="extrapolate")
    return fn(np.linspace(0.0, 1.0, n))


def _single_cycle_loss(gen, native_model, pv,
                        time_s: np.ndarray,
                        current_a: np.ndarray,
                        voltage_ref: np.ndarray) -> float:
    """한 사이클에 대한 loss 계산 (파라미터 빌드는 호출 전에 완료)."""
    from lmro2phase.physics.simulator import run_current_drive

    t_rel = time_s - time_s[0]
    result = run_current_drive(native_model, pv, time_s, current_a, t_eval=t_rel)
    if not result.ok:
        return 1e8

    v_sim_t = interp1d(result.time_s, result.voltage_v,
                        bounds_error=False, fill_value="extrapolate")(t_rel)
    loss_vt = float(np.mean((v_sim_t - voltage_ref) ** 2))

    vg_sim = _norm_vq_grid(result.time_s, result.voltage_v, result.current_a)
    vg_ref = _norm_vq_grid(t_rel, voltage_ref, current_a)
    ok = ~(np.isnan(vg_sim) | np.isnan(vg_ref))
    loss_vq = float(np.mean((vg_sim[ok] - vg_ref[ok]) ** 2)) if ok.any() else 1e6

    return 0.3 * loss_vt + 1.0 * loss_vq


def compute_loss(v_vec: np.ndarray,
                  gen, native_model,
                  cycles_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float:
    """멀티 C-rate 동시 loss.

    cycles_data: [(time_s, current_a, voltage_ref), ...] — 각 사이클 튜플 리스트.
    모든 사이클의 loss 평균을 반환한다.
    """
    v_clip = np.clip(v_vec, LOWER, UPPER)
    try:
        truth, d = _vec_to_phys(v_clip)
        pv = _build_pybamm_params(gen, truth, d)
    except Exception:
        return 1e9

    total = 0.0
    n_ok  = 0
    for time_s, current_a, voltage_ref in cycles_data:
        lc = _single_cycle_loss(gen, native_model, pv, time_s, current_a, voltage_ref)
        if lc >= 1e7:
            total += lc
        else:
            total += lc
        n_ok += 1

    return total / max(n_ok, 1)


# ─── Optuna 탐색 ──────────────────────────────────────────────────────────────

def run_optuna(gen, native_model,
               cycles_data: list,
               monitor: Monitor,
               n_trials: int, seed: int,
               out_dir: Path) -> np.ndarray:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    hist_path = out_dir / "fit_history.jsonl"
    monitor.set_phase("Optuna")

    def objective(trial):
        v = np.array([
            trial.suggest_float(k, lo, hi)
            for k, lo, hi, _ in PARAM_DEFS
        ])
        loss = compute_loss(v, gen, native_model, cycles_data)
        params = {k: float(v[i]) for i, k in enumerate(PARAM_KEYS)}
        params["trial_type"] = "optuna"
        params["loss"] = loss
        monitor.update(loss, params)
        with open(hist_path, "a") as f:
            f.write(json.dumps({"type": "optuna", "loss": loss, **params}) + "\n")
        return loss

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    return np.array([best.params[k] for k in PARAM_KEYS])


# ─── scipy 정제 ───────────────────────────────────────────────────────────────

def run_scipy(init_v: np.ndarray,
              gen, native_model,
              cycles_data: list,
              monitor: Monitor,
              max_iter: int,
              out_dir: Path) -> np.ndarray:
    hist_path = out_dir / "fit_history.jsonl"
    monitor.set_phase("scipy")
    func_eval = [0]

    def obj(x):
        xc = np.clip(x, LOWER, UPPER)
        loss = compute_loss(xc, gen, native_model, cycles_data)
        func_eval[0] += 1
        params = {k: float(xc[i]) for i, k in enumerate(PARAM_KEYS)}
        params["func_eval"] = func_eval[0]
        # 콘솔 출력: 개선됐을 때 + 매 20번째 평가
        if loss < monitor.best_loss or func_eval[0] % 20 == 0:
            monitor.update(loss, params)
        else:
            if loss < monitor.best_loss:
                monitor.best_loss   = loss
                monitor.best_params = params.copy()
                monitor.n_trial    += 1
        with open(hist_path, "a") as f:
            f.write(json.dumps({"type": "scipy", "loss": loss, **params}) + "\n")
        return loss

    result = minimize(
        obj, init_v, method="L-BFGS-B",
        bounds=list(zip(LOWER, UPPER)),
        options={"maxfun": max_iter, "ftol": 1e-12, "gtol": 1e-8},
    )
    return np.clip(result.x, LOWER, UPPER)


# ─── 결과 저장 ────────────────────────────────────────────────────────────────

def save_results(best_v: np.ndarray, monitor: Monitor,
                 true_json: str | Path | None, out_dir: Path) -> None:
    best = {k: float(best_v[i]) for i, k in enumerate(PARAM_KEYS)}
    best["frac_R3m"] = float(np.clip(best["frac_R3m"], 0.05, 0.95))
    best["frac_C2m"] = 1.0 - best["frac_R3m"]

    # 물리 단위로 변환
    result = {
        "loss": monitor.best_loss,
        "frac_R3m":     round(best["frac_R3m"], 6),
        "frac_C2m":     round(best["frac_C2m"], 6),
        "log10_D_R3m":  round(best["log10_D_R3m"], 4),
        "D_R3m_m2_s":   10.0 ** best["log10_D_R3m"],
        "log10_R_R3m":  round(best["log10_R_R3m"], 4),
        "R_R3m_m":      10.0 ** best["log10_R_R3m"],
        "log10_D_C2m":  round(best["log10_D_C2m"], 4),
        "D_C2m_m2_s":   10.0 ** best["log10_D_C2m"],
        "log10_R_C2m":  round(best["log10_R_C2m"], 4),
        "R_C2m_m":      10.0 ** best["log10_R_C2m"],
        "R3m_center_v": round(best["R3m_center_v"], 4),
        "R3m_sigma_v":  round(abs(best["R3m_sigma_v"]), 4),
        "C2m_center_v": round(best["C2m_center_v"], 4),
        "C2m_sigma_v":  round(abs(best["C2m_sigma_v"]), 4),
        "log10_contact_R": round(best["log10_contact_R"], 4),
    }
    (out_dir / "best_params.json").write_text(json.dumps(result, indent=2))

    # 진실값 비교
    true_d = {}
    if true_json and Path(true_json).exists():
        raw = json.loads(Path(true_json).read_text())
        t = raw.get("truth", {})
        tp = raw.get("target_peaks", {})
        true_d = {
            "frac_R3m":        t.get("frac_R3m"),
            "log10_D_R3m":     round(t.get("log10_D_R3m", 0), 4),
            "log10_R_R3m":     round(t.get("log10_R_R3m", 0), 4),
            "log10_D_C2m":     round(t.get("log10_D_C2m", 0), 4),
            "log10_R_C2m":     round(t.get("log10_R_C2m", 0), 4),
            "R3m_center_v":    tp.get("R3m_primary_redox_feature_v"),
            "R3m_sigma_v":     tp.get("R3m_sigma_v"),
            "C2m_center_v":    tp.get("C2m_secondary_redox_feature_v"),
            "C2m_sigma_v":     tp.get("C2m_sigma_v"),
        }

    comp = {
        "model":      "native 2-phase SPM (Primary=R3m, Secondary=C2m)",
        "ocp_model":  "Gaussian dQ/dV (center_v, sigma_v per phase)",
        "final_loss": monitor.best_loss,
        "estimated":  result,
        "true":       true_d,
    }
    if true_d:
        comp["errors"] = {
            k: round(result.get(k, 0) - (true_d.get(k) or 0), 4)
            for k in ["frac_R3m","log10_D_R3m","log10_D_C2m","log10_R_R3m","log10_R_C2m",
                       "R3m_center_v","R3m_sigma_v","C2m_center_v","C2m_sigma_v"]
            if true_d.get(k) is not None
        }
    (out_dir / "comparison.json").write_text(json.dumps(comp, indent=2))

    monitor.print_summary(true_d)


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fit_history.jsonl").write_text("")  # 초기화

    # 생성 스크립트 모듈 로드
    gen = _load_gen()

    # 사이클 번호 파싱 ("1,2,3,4" → [1,2,3,4])
    cycle_ids = [int(c.strip()) for c in str(args.cycles).split(",") if c.strip()]

    # 데이터 로드 및 서브샘플링 (멀티 사이클)
    from lmro2phase.io.toyo_ascii import parse_toyo
    profile = parse_toyo(args.data_csv)

    cycles_data = []   # [(time_s, current_a, voltage_v), ...]
    cycle_descs = []
    for cyc_id in cycle_ids:
        cyc = profile.select_cycle(cyc_id)
        idx = np.arange(0, len(cyc), args.subsample)
        t_s = cyc.time_s[idx]
        i_a = cyc.current_a[idx]
        v_v = cyc.voltage_v[idx]
        cycles_data.append((t_s, i_a, v_v))
        c_rate = abs(i_a).mean() / (NOM_CAP_AH * 1000 / 1000)  # rough C-rate
        cycle_descs.append(f"cycle{cyc_id}(n={len(t_s)},span={t_s[-1]/3600:.1f}h)")

    data_desc = (
        f"{Path(args.data_csv).parent.name}  "
        f"cycles={cycle_ids}  subsample=1/{args.subsample}  "
        + "  ".join(cycle_descs)
    )

    # 진실값 로드 (옵션)
    true_params: dict = {}
    if args.true_params and Path(args.true_params).exists():
        raw = json.loads(Path(args.true_params).read_text())
        t  = raw.get("truth", {})
        tp = raw.get("target_peaks", {})
        true_params = {
            "frac_R3m":    t.get("frac_R3m"),
            "log10_D_R3m": round(t.get("log10_D_R3m", 0), 4),
            "log10_R_R3m": round(t.get("log10_R_R3m", 0), 4),
            "log10_D_C2m": round(t.get("log10_D_C2m", 0), 4),
            "log10_R_C2m": round(t.get("log10_R_C2m", 0), 4),
            "R3m_center_v": tp.get("R3m_primary_redox_feature_v"),
            "R3m_sigma_v":  tp.get("R3m_sigma_v"),
            "C2m_center_v": tp.get("C2m_secondary_redox_feature_v"),
            "C2m_sigma_v":  tp.get("C2m_sigma_v"),
        }

    model_desc = (
        "PyBaMM SPM  |  particle phases: ('1','2')  |  "
        "Primary=R3m, Secondary=C2m  |  "
        "Gaussian dQ/dV OCP (center_v, sigma_v)"
    )

    # 모니터 생성
    monitor = Monitor(model_desc, data_desc, args.n_optuna, args.n_scipy, true_params)

    # native 2-phase 모델 빌드 (한 번)
    print("  PyBaMM native 2-phase 모델 초기화...", flush=True)
    native_model = gen.build_native_model()
    print("  모델 초기화 완료.\n")

    # [1/2] Optuna
    best_optuna = run_optuna(
        gen, native_model, cycles_data,
        monitor, args.n_optuna, args.seed, out_dir,
    )

    # [2/2] scipy L-BFGS-B
    best_final = run_scipy(
        best_optuna, gen, native_model, cycles_data,
        monitor, args.n_scipy, out_dir,
    )

    # 결과 저장
    save_results(best_final, monitor, args.true_params, out_dir)
    print(f"\n  출력 경로: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Native 2-phase SPM Gaussian OCP 역추정",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-csv",
        default=(
            "data/raw/toyo/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/"
            "Toyo_LMR_native2phase_PyBaMM_0p1C_0p33C_0p5C_1C.csv"
        ),
        help="TOYO 형식 측정 CSV",
    )
    parser.add_argument(
        "--true-params",
        default=(
            "data/raw/toyo/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/"
            "true_native_2phase_parameters.json"
        ),
        help="진실값 JSON (옵션, 비교용)",
    )
    parser.add_argument("--out-dir",   default="data/fit_results/native2phase_gaussian")
    parser.add_argument("--cycles",    default="1,2,3,4",
                        help="피팅에 사용할 사이클 번호 (쉼표 구분, 예: '1,2,3,4')")
    parser.add_argument("--subsample", type=int, default=600, help="서브샘플링 스텝 (원본 0.5s 기준)")
    parser.add_argument("--n-optuna",  type=int, default=60,  help="Optuna 탐색 trial 수")
    parser.add_argument("--n-scipy",   type=int, default=400, help="scipy max 함수평가 수")
    parser.add_argument("--seed",      type=int, default=42,  help="Optuna random seed")
    args = parser.parse_args()
    main(args)
