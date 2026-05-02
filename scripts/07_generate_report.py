"""
Stage 7: 파이프라인 전체 결과 리포트 생성

실행:
    cd lmro_2phase_inverse
    python scripts/07_generate_report.py
"""

from __future__ import annotations

import glob
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

STAMP = datetime.now().strftime("%y%m%d_%H%M%S")
REPORT_DIR = ROOT / f"data/reports/result_report_{STAMP}"
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def savefig(name: str) -> str:
    path = FIG_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return f"figures/{name}"


def md_img(caption: str, rel_path: str) -> str:
    return f"![{caption}]({rel_path})\n*{caption}*\n"


def dqdv(voltage: np.ndarray, capacity_ah: np.ndarray,
          window: int = 5, polyorder: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """dQ/dV 계산 (Savitzky-Golay 스무딩, V 단조증가 보장).

    반환값은 전압 오름차순 정렬 기준의 dQ/dV (mAh/V).
    방전의 경우 호출측에서 부호를 반전(-) 하면 관습적 양수 피크를 얻는다.
    """
    from scipy.signal import savgol_filter
    # 전압 단조증가 순 정렬
    idx = np.argsort(voltage)
    v = voltage[idx]
    q = capacity_ah[idx] * 1000  # mAh

    # 중복 V 제거
    _, uniq = np.unique(v, return_index=True)
    v, q = v[uniq], q[uniq]
    if len(v) < window + 1:
        return v, np.gradient(q, v)

    w = min(window, len(q) if len(q) % 2 == 1 else len(q) - 1)
    q_smooth = savgol_filter(q, w, polyorder)
    dqdv_vals = np.gradient(q_smooth, v)
    return v, dqdv_vals


def half_cycle_q_ah(current_a: np.ndarray, time_s: np.ndarray,
                     mask: np.ndarray) -> np.ndarray:
    """반사이클(충전 또는 방전) 시작점을 0으로 리셋한 누적 용량 (Ah)."""
    curr = current_a[mask]
    t = time_s[mask]
    if len(curr) < 2:
        return np.zeros(len(curr))
    return np.cumsum(np.abs(curr) * np.gradient(t) / 3600)


def load_profile():
    from lmro2phase.io.toyo_ascii import ToyoAsciiParser
    parser = ToyoAsciiParser(ROOT / "configs/toyo_ascii.yaml")
    return parser.parse(ROOT / "data/raw/toyo/Toyo_LMR_HalfCell_Sample_50cycles.csv")


def classify_cycles(profile, tol_frac: float = 0.05) -> dict:
    """사이클을 C-rate 그룹으로 분류.

    Returns:
        {
          "varied": [(cyc_idx, i_mean_ma), ...],   # C-rate가 서로 다른 초반 사이클
          "repeated": [(cyc_idx, i_mean_ma), ...], # 동일 C-rate 반복 사이클
          "dominant_i_ma": float,                  # 반복 사이클의 대표 전류 (mA)
        }
    """
    cycles = profile.get_cycles()
    i_means = {}
    for cyc_idx in cycles:
        cyc = profile.select_cycle(cyc_idx)
        dchg = cyc.current_a > 0
        if dchg.any():
            i_means[cyc_idx] = float(np.mean(np.abs(cyc.current_a[dchg])) * 1000)
        else:
            i_means[cyc_idx] = 0.0

    all_vals = np.array(list(i_means.values()))
    # 최빈 전류 = 반복 사이클의 대표값
    counts, edges = np.histogram(all_vals, bins=30)
    dominant_i = float(edges[np.argmax(counts)] + (edges[1] - edges[0]) / 2)

    varied, repeated = [], []
    for cyc_idx in cycles:
        i = i_means[cyc_idx]
        if abs(i - dominant_i) / max(dominant_i, 1e-9) > tol_frac:
            varied.append((cyc_idx, i))
        else:
            repeated.append((cyc_idx, i))

    return {"varied": varied, "repeated": repeated, "dominant_i_ma": dominant_i}


def pick_repeated_display(repeated: list, max_n: int = 10) -> list[int]:
    """반복 사이클 중 균등 간격으로 최대 max_n개 선택."""
    idxs = [c for c, _ in repeated]
    if len(idxs) <= max_n:
        return idxs
    step = len(idxs) / max_n
    selected = [idxs[int(round(i * step))] for i in range(max_n)]
    # 마지막 사이클 포함
    if idxs[-1] not in selected:
        selected[-1] = idxs[-1]
    return sorted(set(selected))


# ──────────────────────────────────────────────
# § 2: 입력 데이터
# ──────────────────────────────────────────────

def section_input_data(profile) -> str:
    cycles = profile.get_cycles()
    n_cyc = len(cycles)

    # 통계 수집
    rows = []
    for cyc_idx in cycles:
        cyc = profile.select_cycle(cyc_idx)
        q_total = np.trapezoid(np.abs(cyc.current_a), cyc.time_s / 3600) * 1000  # mAh
        dchg = cyc.current_a > 0
        i_dchg_ma = float(np.mean(np.abs(cyc.current_a[dchg])) * 1000) if dchg.any() else 0.0
        rows.append({"cycle": cyc_idx, "n_pts": len(cyc),
                      "q_mah": q_total, "v_min": cyc.voltage_v.min(),
                      "v_max": cyc.voltage_v.max(),
                      "i_dchg_ma": i_dchg_ma,
                      "duration_h": (cyc.time_s[-1] - cyc.time_s[0]) / 3600})
    stats_df = pd.DataFrame(rows)

    # C-rate 그룹 분류
    cgroups = classify_cycles(profile)
    varied_cycs  = [c for c, _ in cgroups["varied"]]   # C-rate 상이 → 전부 표시
    repeated_cycs = [c for c, _ in cgroups["repeated"]] # 동일 반복 → 간격 샘플링
    rep_display  = pick_repeated_display(cgroups["repeated"], max_n=10)
    dom_i        = cgroups["dominant_i_ma"]
    log.info(f"C-rate 변동 사이클: {varied_cycs}, 반복 사이클(표시): {rep_display}")

    # ── Fig 1: Q-V 오버레이 (충전/방전 분리, 그룹별 다른 스타일) ──
    #   · C-rate 변동 사이클: 개별 색상 + 레이블 (tab10)
    #   · 반복 사이클: plasma 컬러맵, 선택된 간격만
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors_varied = plt.cm.tab10(np.linspace(0, 0.9, max(len(varied_cycs), 1)))
    cmap_rep = plt.colormaps["plasma"].resampled(max(len(rep_display), 2))

    for idx, cyc_idx in enumerate(varied_cycs):
        cyc = profile.select_cycle(cyc_idx)
        t = cyc.time_s
        chg = cyc.current_a < 0
        dchg = cyc.current_a > 0
        col = colors_varied[idx]
        i_ma = rows[cyc_idx - 1]["i_dchg_ma"]
        lbl = f"C{cyc_idx} ({i_ma:.2f} mA)"
        if chg.any():
            axes[0].plot(half_cycle_q_ah(cyc.current_a, t, chg) * 1000,
                         cyc.voltage_v[chg], color=col, linewidth=1.2, label=lbl, zorder=3)
        if dchg.any():
            axes[1].plot(half_cycle_q_ah(cyc.current_a, t, dchg) * 1000,
                         cyc.voltage_v[dchg], color=col, linewidth=1.2, label=lbl, zorder=3)

    for ri, cyc_idx in enumerate(rep_display):
        cyc = profile.select_cycle(cyc_idx)
        t = cyc.time_s
        chg = cyc.current_a < 0
        dchg = cyc.current_a > 0
        col = cmap_rep(ri / max(len(rep_display) - 1, 1))
        if chg.any():
            axes[0].plot(half_cycle_q_ah(cyc.current_a, t, chg) * 1000,
                         cyc.voltage_v[chg], color=col, linewidth=0.7, alpha=0.7)
        if dchg.any():
            axes[1].plot(half_cycle_q_ah(cyc.current_a, t, dchg) * 1000,
                         cyc.voltage_v[dchg], color=col, linewidth=0.7, alpha=0.7)

    for ax, (title, xlabel) in zip(axes, [
            ("Charge Q-V", "Charge Capacity (mAh)"),
            ("Discharge Q-V", "Discharge Capacity (mAh)")]):
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Voltage (V)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if varied_cycs:
            ax.legend(fontsize=7, title="C-rate variation", loc="best")

    # 반복 사이클 colorbar
    sm = plt.cm.ScalarMappable(cmap="plasma",
                                norm=plt.Normalize(vmin=min(rep_display), vmax=max(rep_display)))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label=f"Repeated cycles (every ~{max(1,len(repeated_cycs)//10)} cyc)", shrink=0.8)
    fig.suptitle(f"TOYO LMR Half-Cell — Q-V Overview  "
                 f"[C-rate varied: cyc {varied_cycs}  |  repeated @ {dom_i:.2f} mA: cyc {rep_display}]",
                 fontsize=11)
    fig1 = savefig("fig_01_qv_all_cycles.png")
    log.info("fig_01 저장")

    # ── Fig 2: dQ/dV — C-rate 변동 사이클 전부 + 반복 사이클 간격 샘플 ──
    dqdv_cycles = varied_cycs + rep_display
    n_dqdv = len(dqdv_cycles)
    colors_dqdv = (list(colors_varied) +
                   [cmap_rep(i / max(len(rep_display) - 1, 1)) for i in range(len(rep_display))])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for cyc_idx, col in zip(dqdv_cycles, colors_dqdv):
        cyc = profile.select_cycle(cyc_idx)
        t = cyc.time_s
        chg = cyc.current_a < 0
        dchg = cyc.current_a > 0
        i_ma = rows[cyc_idx - 1]["i_dchg_ma"]
        in_varied = cyc_idx in varied_cycs
        lbl = f"C{cyc_idx} ({i_ma:.2f} mA)" if in_varied else f"Cyc {cyc_idx}"
        lw = 1.4 if in_varied else 0.9
        alpha = 1.0 if in_varied else 0.7

        if chg.sum() > 5:
            win = max(3, min(7, (chg.sum() // 2) | 1))
            q_chg = half_cycle_q_ah(cyc.current_a, t, chg)
            v_c, dq_c = dqdv(cyc.voltage_v[chg], q_chg, window=win)
            axes[0].plot(v_c, dq_c * 1000, color=col, label=lbl,
                         linewidth=lw, alpha=alpha)
        if dchg.sum() > 5:
            win = max(3, min(7, (dchg.sum() // 2) | 1))
            q_dchg = half_cycle_q_ah(cyc.current_a, t, dchg)
            v_d, dq_d = dqdv(cyc.voltage_v[dchg], q_dchg, window=win)
            axes[1].plot(v_d, -dq_d * 1000, color=col, label=lbl,
                         linewidth=lw, alpha=alpha)

    for ax, title in zip(axes, ["Charge dQ/dV", "Discharge dQ/dV"]):
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("dQ/dV (mAh/V)")
        ax.set_title(title)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.suptitle("dQ/dV — C-rate Variation (bold) + Repeated Cycles (sampled)", fontsize=12)
    plt.tight_layout()
    fig2 = savefig("fig_02_dqdv_representative.png")
    log.info("fig_02 저장")

    # 통계 테이블 (마크다운)
    tbl = "| Cycle | I dchg (mA) | N pts | Q (mAh) | V min (V) | V max (V) | Duration (h) |\n"
    tbl += "|------:|------------:|------:|--------:|----------:|----------:|-------------:|\n"
    for _, r in stats_df.iterrows():
        tbl += (f"| {int(r.cycle)} | {r.i_dchg_ma:.2f} | {int(r.n_pts)} | {r.q_mah:.3f} |"
                f" {r.v_min:.3f} | {r.v_max:.3f} | {r.duration_h:.2f} |\n")

    s = stats_df
    varied_str = ", ".join(str(c) for c in varied_cycs)
    summary = (f"\n**요약**: 총 {n_cyc}사이클 — "
               f"C-rate 변동 사이클: [{varied_str}] / "
               f"반복 사이클(@ {dom_i:.2f} mA): {len(repeated_cycs)}개\n"
               f"Q 범위 {s.q_mah.min():.2f}–{s.q_mah.max():.2f} mAh "
               f"(평균 {s.q_mah.mean():.2f} mAh), "
               f"전압 범위 {s.v_min.min():.3f}–{s.v_max.max():.3f} V\n")

    return f"""
## § 2. 입력 데이터 개요

> **표시 방식**: C-rate 변동 사이클(cyc {varied_cycs})은 전부 표시, 동일 C-rate 반복 사이클은 적절한 간격으로 샘플링.

{md_img("Q-V 오버레이 (색 레이블=C-rate 변동 사이클, colorbar=반복 사이클)", fig1)}

{md_img("dQ/dV (굵은 선=C-rate 변동, 얇은 선=반복 사이클 샘플)", fig2)}

### 사이클별 통계

{tbl}{summary}
"""


# ──────────────────────────────────────────────
# § 3: Stage 1 — OCP 피팅
# ──────────────────────────────────────────────

def section_stage1(profile) -> str:
    stage1_dir = ROOT / "data/reports/stage1_fit"
    if not stage1_dir.exists():
        return "\n## § 3. Stage 1: OCP 피팅\n\n> ⚠ 결과 없음\n"

    with open(stage1_dir / "best_params.json") as f:
        bp = json.load(f)
    with open(stage1_dir / "best_ocp_tanh_R3m.json") as f:
        r3m = json.load(f)
    with open(stage1_dir / "best_ocp_tanh_C2m.json") as f:
        c2m = json.load(f)

    from lmro2phase.physics.ocp_tanh import TanhOCPParams, make_tanh_ocp_numpy
    from lmro2phase.physics.positive_2phase_factory import TwoPhaseOCPParams, make_effective_ocp

    sto = np.linspace(0.01, 0.99, 500)

    tp_r3m = TanhOCPParams(b0=r3m["b0"], b1=r3m["b1"],
                            amps=np.array(r3m["amps"]),
                            centers=np.array(r3m["centers"]),
                            widths=np.array(r3m["widths"]))
    tp_c2m = TanhOCPParams(b0=c2m["b0"], b1=c2m["b1"],
                            amps=np.array(c2m["amps"]),
                            centers=np.array(c2m["centers"]),
                            widths=np.array(c2m["widths"]))

    fn_r3m = make_tanh_ocp_numpy(tp_r3m)
    fn_c2m = make_tanh_ocp_numpy(tp_c2m)

    frac_r3m = float(bp["frac_R3m"])
    frac_c2m = float(bp["frac_C2m"])
    v_r3m = fn_r3m(sto)
    v_c2m = fn_c2m(sto)
    v_eff = frac_r3m * v_r3m + frac_c2m * v_c2m

    # ── Fig 3: OCP 3곡선 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sto, v_r3m, "b-",  label=f"R3m OCP  (b0={r3m['b0']:.3f}V, frac={frac_r3m:.3f})", linewidth=1.5)
    ax.plot(sto, v_c2m, "g-",  label=f"C2m OCP  (b0={c2m['b0']:.3f}V, frac={frac_c2m:.3f})", linewidth=1.5)
    ax.plot(sto, v_eff, "r--", label="Effective (weighted sum)", linewidth=2.0)
    ax.set_xlabel("Stoichiometry x")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Stage 1 — Fitted OCP Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(2.0, 5.0)
    fig3 = savefig("fig_03_ocp_phases.png")
    log.info("fig_03 저장")

    # ── Fig 4: Cycle 1 측정 V-Q vs Stage1 시뮬레이션 ──
    fig4 = _stage1_fit_overlay(profile, bp, tp_r3m, tp_c2m)

    # 파라미터 테이블
    def fmt(key, unit=""):
        val = bp.get(key, "N/A")
        if isinstance(val, float):
            return f"{val:.4f}{unit}"
        return str(val)

    param_tbl = (
        "| 파라미터 | 값 | 단위 |\n"
        "|---------|-----|------|\n"
        f"| log₁₀ D_R3m | {fmt('log10_D_R3m')} | m²/s |\n"
        f"| log₁₀ R_R3m | {fmt('log10_R_R3m')} | m |\n"
        f"| frac_R3m    | {fmt('frac_R3m')} | — |\n"
        f"| frac_C2m    | {fmt('frac_C2m')} | — |\n"
        f"| log₁₀ D_C2m | {fmt('log10_D_C2m')} | m²/s |\n"
        f"| log₁₀ R_C2m | {fmt('log10_R_C2m')} | m |\n"
        f"| log₁₀ R_contact | {fmt('log10_contact_resistance')} | Ω |\n"
        f"| capacity_scale | {fmt('capacity_scale')} | — |\n"
        f"| stoich_shift | {fmt('initial_stoichiometry_shift')} | — |\n"
    )

    return f"""
## § 3. Stage 1: OCP 피팅

{md_img("Fitted OCP curves — R3m, C2m, Effective", fig3)}

{md_img("Cycle 1: 측정 V-Q vs Stage 1 시뮬레이션", fig4)}

### 베스트 파라미터

{param_tbl}

- R3m OCP: b0={r3m['b0']:.4f} V, b1={r3m['b1']:.4f}, tanh terms={len(r3m['amps'])}
- C2m OCP: b0={c2m['b0']:.4f} V, b1={c2m['b1']:.4f}, tanh terms={len(c2m['amps'])}
"""


def _stage1_fit_overlay(profile, bp, tp_r3m, tp_c2m) -> str:
    """Stage 1 파라미터로 시뮬레이션 후 사이클 1 V-Q 비교 플롯."""
    try:
        from lmro2phase.physics.ocp_grid import OCPGrid
        from lmro2phase.physics.positive_2phase_factory import (
            TwoPhaseOCPParams, make_effective_ocp,
            make_effective_diffusivity, make_effective_radius,
        )
        from lmro2phase.physics.lmr_parameter_set import build_pybamm_halfcell_params
        from lmro2phase.physics.halfcell_model_factory import build_halfcell_model, ModelType
        from lmro2phase.physics.simulator import run_current_drive

        cyc1 = profile.select_cycle(1)

        ocp_r3m_grid = OCPGrid.from_tanh_params(tp_r3m)
        ocp_c2m_grid = OCPGrid.from_tanh_params(tp_c2m)

        two_phase = TwoPhaseOCPParams(
            frac_R3m=float(bp["frac_R3m"]),
            frac_C2m=float(bp["frac_C2m"]),
            U_R3m=ocp_r3m_grid.to_pybamm_interpolant(),
            U_C2m=ocp_c2m_grid.to_pybamm_interpolant(),
            D_R3m=10.0 ** float(bp["log10_D_R3m"]),
            D_C2m=10.0 ** float(bp["log10_D_C2m"]),
            R_R3m=10.0 ** float(bp["log10_R_R3m"]),
            R_C2m=10.0 ** float(bp["log10_R_C2m"]),
        )
        pv = build_pybamm_halfcell_params(
            ocp_fn=make_effective_ocp(two_phase),
            D_s=make_effective_diffusivity(two_phase),
            R_particle=make_effective_radius(two_phase),
            contact_resistance=10.0 ** float(bp["log10_contact_resistance"]),
            capacity_scale=float(bp["capacity_scale"]),
            c_s_0_fraction=0.5 + float(bp["initial_stoichiometry_shift"]),
        )
        model = build_halfcell_model(ModelType.SPMe)
        result = run_current_drive(model, pv, cyc1.time_s, cyc1.current_a)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        t_exp = cyc1.time_s
        chg_exp = cyc1.current_a < 0
        dchg_exp = cyc1.current_a > 0

        # ── 충전 Q-V ──
        if chg_exp.any():
            q_chg_exp = half_cycle_q_ah(cyc1.current_a, t_exp, chg_exp) * 1000
            axes[0].plot(q_chg_exp, cyc1.voltage_v[chg_exp], "k-", label="Measured", linewidth=1.5)
        if result.ok:
            t_sim = result.time_s
            chg_sim = result.current_a < 0
            if chg_sim.any():
                q_chg_sim = half_cycle_q_ah(result.current_a, t_sim, chg_sim) * 1000
                axes[0].plot(q_chg_sim, result.voltage_v[chg_sim], "r--", label="Stage1 Sim", linewidth=1.5)
        axes[0].set_xlabel("Charge Capacity (mAh)")
        axes[0].set_ylabel("Voltage (V)")
        axes[0].set_title("Cycle 1 — Charge Q-V")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # ── 방전 Q-V ──
        if dchg_exp.any():
            q_dchg_exp = half_cycle_q_ah(cyc1.current_a, t_exp, dchg_exp) * 1000
            axes[1].plot(q_dchg_exp, cyc1.voltage_v[dchg_exp], "k-", label="Measured", linewidth=1.5)
        if result.ok:
            dchg_sim = result.current_a > 0
            if dchg_sim.any():
                q_dchg_sim = half_cycle_q_ah(result.current_a, t_sim, dchg_sim) * 1000
                axes[1].plot(q_dchg_sim, result.voltage_v[dchg_sim], "r--", label="Stage1 Sim", linewidth=1.5)
        axes[1].set_xlabel("Discharge Capacity (mAh)")
        axes[1].set_ylabel("Voltage (V)")
        axes[1].set_title("Cycle 1 — Discharge Q-V")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # ── V-t ──
        axes[2].plot(t_exp / 3600, cyc1.voltage_v, "k-", label="Measured", linewidth=1.5)
        if result.ok:
            axes[2].plot(t_sim / 3600, result.voltage_v, "r--", label="Stage1 Sim", linewidth=1.5)
        axes[2].set_xlabel("Time (h)")
        axes[2].set_ylabel("Voltage (V)")
        axes[2].set_title("Cycle 1 — V-t")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return savefig("fig_04_stage1_fit_overlay.png")
    except Exception as e:
        log.warning(f"fig_04 실패: {e}")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"Simulation failed:\n{e}", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        return savefig("fig_04_stage1_fit_overlay.png")


# ──────────────────────────────────────────────
# § 4: Stage 2 — 합성 데이터셋
# ──────────────────────────────────────────────

def section_stage2(profile) -> str:
    params_path = ROOT / "data/synthetic/params.parquet"
    profiles_path = ROOT / "data/synthetic/profiles.zarr"
    failed_path = ROOT / "data/synthetic/failed_cases.parquet"

    if not params_path.exists():
        return "\n## § 4. Stage 2: 합성 데이터셋\n\n> ⚠ 결과 없음\n"

    df = pd.read_parquet(params_path)
    n_ok = len(df)
    n_fail = len(pd.read_parquet(failed_path)) if failed_path.exists() else "N/A"
    ocp_modes = df["ocp_mode"].value_counts().to_dict() if "ocp_mode" in df.columns else {}

    # ── Fig 5: 파라미터 분포 히스토그램 ──
    param_cols = ["log10_D_R3m", "log10_D_C2m", "frac_R3m",
                  "log10_R_R3m", "capacity_scale", "initial_stoichiometry_shift"]
    param_labels = ["log₁₀ D_R3m", "log₁₀ D_C2m", "frac_R3m",
                    "log₁₀ R_R3m", "capacity_scale", "stoich_shift"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col, label in zip(axes.flat, param_cols, param_labels):
        if col in df.columns:
            ax.hist(df[col], bins=40, color="steelblue", edgecolor="white", linewidth=0.3)
            ax.set_title(label, fontsize=11)
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
            mu, sigma = df[col].mean(), df[col].std()
            ax.axvline(mu, color="red", linewidth=1.5, linestyle="--", label=f"μ={mu:.3f}")
            ax.legend(fontsize=8)
    fig.suptitle("Synthetic Dataset — Parameter Distributions", fontsize=13)
    plt.tight_layout()
    fig5 = savefig("fig_05_param_distributions.png")
    log.info("fig_05 저장")

    # ── Fig 6: 합성 V-Q 샘플 오버레이 ──
    fig6 = _synthetic_profiles_overlay(profile, profiles_path)

    # OCP 모드 분포 문자열
    ocp_mode_str = ", ".join(f"{k}: {v}" for k, v in ocp_modes.items()) if ocp_modes else "N/A"

    return f"""
## § 4. Stage 2: 합성 데이터셋

{md_img("합성 데이터셋 파라미터 분포 히스토그램", fig5)}

{md_img("합성 V-Q 프로파일 샘플 (회색) + TOYO 실측 사이클 1 (빨강)", fig6)}

### 생성 통계

| 항목 | 값 |
|------|-----|
| 성공 샘플 수 | {n_ok} |
| 실패 샘플 수 | {n_fail} |
| 성공률 | {n_ok / (n_ok + (n_fail if isinstance(n_fail, int) else 0)) * 100:.1f}% |
| OCP 변동 모드 분포 | {ocp_mode_str} |
| D_R3m 범위 | {df['log10_D_R3m'].min():.2f} ~ {df['log10_D_R3m'].max():.2f} (log₁₀) |
| D_C2m 범위 | {df['log10_D_C2m'].min():.2f} ~ {df['log10_D_C2m'].max():.2f} (log₁₀) |
| frac_R3m 범위 | {df['frac_R3m'].min():.3f} ~ {df['frac_R3m'].max():.3f} |
"""


def _synthetic_profiles_overlay(profile, profiles_path: Path) -> str:
    """합성 프로파일 50개 오버레이 + 실측 사이클 1."""
    try:
        import zarr
        cyc1 = profile.select_cycle(1)
        t_exp = cyc1.time_s
        q_exp = np.cumsum(np.abs(cyc1.current_a) * np.gradient(t_exp) / 3600) * 1000

        zp = zarr.open(str(profiles_path))
        keys = list(zp.keys())
        rng = np.random.default_rng(0)
        sample_keys = rng.choice(keys, size=min(50, len(keys)), replace=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        for k in sample_keys:
            try:
                t_s = zp[k]["time_s"][:]
                v_s = zp[k]["voltage_v"][:]
                i_s = zp[k]["current_a"][:]
                q_s = np.cumsum(np.abs(i_s) * np.gradient(t_s) / 3600) * 1000
                ax.plot(q_s, v_s, color="gray", linewidth=0.5, alpha=0.4)
            except Exception:
                pass

        ax.plot(q_exp, cyc1.voltage_v, "r-", linewidth=2.0, label="TOYO Cycle 1 (measured)", zorder=5)
        ax.set_xlabel("Capacity (mAh)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title("Synthetic V-Q Profiles (gray) + TOYO Cycle 1 (red)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return savefig("fig_06_synthetic_profiles.png")
    except Exception as e:
        log.warning(f"fig_06 실패: {e}")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"Plot failed:\n{e}", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        return savefig("fig_06_synthetic_profiles.png")


# ──────────────────────────────────────────────
# § 5: Stage 3 — DL 모델 학습
# ──────────────────────────────────────────────

def section_stage3() -> str:
    model_path = ROOT / "data/models/inverse/best_model.pt"
    if not model_path.exists():
        return "\n## § 5. Stage 3: DL 모델 학습\n\n> ⚠ 모델 없음\n"

    # 모델 파라미터 수 계산
    try:
        import torch
        from lmro2phase.learning.model_inverse import InverseModel
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(ROOT / "configs/stage3_train_inverse.yaml")
        mc = cfg.model
        m = InverseModel(
            in_channels=int(mc.encoder.in_channels),
            base_channels=int(mc.encoder.base_channels),
            n_blocks=int(mc.encoder.n_blocks),
            kernel_size=int(mc.encoder.kernel_size),
            dropout=float(mc.encoder.dropout),
            use_transformer=bool(mc.encoder.use_transformer),
            transformer_heads=int(mc.encoder.transformer_heads),
            transformer_layers=int(mc.encoder.transformer_layers),
            n_scalar=int(mc.heads.scalar_dim),
            ocp_grid_len=int(mc.heads.ocp_grid_length),
        )
        total_params = sum(p.numel() for p in m.parameters())
        trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        model_size_mb = model_path.stat().st_size / (1024 ** 2)

        tc = cfg.training
        arch_tbl = (
            "| 항목 | 값 |\n"
            "|------|----|\n"
            f"| 인코더 타입 | {mc.encoder.type} |\n"
            f"| Base channels | {mc.encoder.base_channels} |\n"
            f"| ResBlock 수 | {mc.encoder.n_blocks} |\n"
            f"| Kernel size | {mc.encoder.kernel_size} |\n"
            f"| Transformer | {mc.encoder.use_transformer} "
            f"({mc.encoder.transformer_heads}h × {mc.encoder.transformer_layers}L) |\n"
            f"| Scalar head dim | {mc.heads.scalar_dim} |\n"
            f"| OCP grid length | {mc.heads.ocp_grid_length} |\n"
            f"| 총 파라미터 수 | {total_params:,} |\n"
            f"| Trainable 파라미터 | {trainable_params:,} |\n"
            f"| 모델 파일 크기 | {model_size_mb:.2f} MB |\n"
        )

        train_tbl = (
            "| 항목 | 값 |\n"
            "|------|----|\n"
            f"| Batch size | {tc.batch_size} |\n"
            f"| Max epochs | {tc.max_epochs} |\n"
            f"| Learning rate | {tc.learning_rate} |\n"
            f"| Weight decay | {tc.weight_decay} |\n"
            f"| Scheduler | {tc.scheduler} |\n"
            f"| DataLoader workers | {getattr(tc, 'dataloader_workers', 2)} |\n"
            f"| Train/Val split | {cfg.dataset.train_val_split:.0%} / "
            f"{1 - cfg.dataset.train_val_split:.0%} |\n"
        )
    except Exception as e:
        arch_tbl = f"> 모델 정보 로드 실패: {e}\n"
        train_tbl = ""

    return f"""
## § 5. Stage 3: DL 역모델 학습

### 모델 구조

{arch_tbl}

### 학습 설정

{train_tbl}

> 학습 손실 곡선: 현재 epoch별 로그가 파일로 저장되지 않습니다.
> 최종 best val_loss는 학습 실행 로그에서 확인하세요.
"""


# ──────────────────────────────────────────────
# § 6: Stage 4 — Forward Validation
# ──────────────────────────────────────────────

def section_stage4(profile) -> str:
    inf_dir = ROOT / "data/reports/inference"
    if not inf_dir.exists():
        return "\n## § 6. Stage 4: Forward Validation\n\n> ⚠ 결과 없음\n"

    # 전 사이클 RMSE 수집
    records = []
    for summary_f in sorted(inf_dir.glob("cycle_*/residual_summary.json")):
        with open(summary_f) as f:
            d = json.load(f)
        if d.get("ok"):
            records.append({
                "cycle": d.get("cycle", int(summary_f.parent.name.split("_")[1])),
                "rmse_mv": d["rmse_v"] * 1000,
                "mae_mv": d["mae_v"] * 1000,
                "max_err_mv": d["max_err_v"] * 1000,
            })
    if not records:
        return "\n## § 6. Stage 4: Forward Validation\n\n> ⚠ 성공한 validation 없음\n"

    df_rmse = pd.DataFrame(records).sort_values("cycle")

    # ── Fig 7: 사이클별 RMSE 막대 ──
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(df_rmse["cycle"], df_rmse["rmse_mv"], color="steelblue", alpha=0.8, width=0.7)
    ax.axhline(df_rmse["rmse_mv"].mean(), color="red", linewidth=1.5,
               linestyle="--", label=f"Mean {df_rmse['rmse_mv'].mean():.1f} mV")
    ax.set_xlabel("Cycle #")
    ax.set_ylabel("RMSE (mV)")
    ax.set_title("Forward Validation RMSE per Cycle")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig7 = savefig("fig_07_rmse_by_cycle.png")
    log.info("fig_07 저장")

    # ── Fig 8: 대표 사이클 V-t + 잔차 ──
    rep_cycles = _pick_representative_cycles(df_rmse)
    fig8 = _vt_comparison(profile, rep_cycles, inf_dir)

    # ── Fig 9: 대표 사이클 Q-V 비교 ──
    fig9 = _qv_comparison(profile, rep_cycles, inf_dir)

    # ── Fig 10: 대표 사이클 dQ/dV 비교 ──
    fig10 = _dqdv_comparison(profile, rep_cycles, inf_dir)

    # RMSE 테이블 (전체)
    rmse_tbl = (
        "| Cycle | RMSE (mV) | MAE (mV) | Max err (mV) |\n"
        "|------:|----------:|---------:|-------------:|\n"
    )
    for _, r in df_rmse.iterrows():
        rmse_tbl += (f"| {int(r.cycle)} | {r.rmse_mv:.1f} | "
                     f"{r.mae_mv:.1f} | {r.max_err_mv:.1f} |\n")

    summary_stats = (
        f"**전체 요약**: {len(df_rmse)}사이클 성공, "
        f"RMSE 평균 {df_rmse.rmse_mv.mean():.1f} mV / 최대 {df_rmse.rmse_mv.max():.1f} mV / "
        f"최소 {df_rmse.rmse_mv.min():.1f} mV"
    )

    return f"""
## § 6. Stage 4: Forward Validation

{md_img("사이클별 RMSE (빨간 점선=평균)", fig7)}

{md_img(f"대표 사이클 V-t 비교 (사이클 {rep_cycles})", fig8)}

{md_img(f"대표 사이클 Q-V 비교", fig9)}

{md_img(f"대표 사이클 dQ/dV 비교", fig10)}

### 사이클별 RMSE / MAE

{rmse_tbl}

{summary_stats}
"""


def _pick_representative_cycles(df_rmse: pd.DataFrame) -> list[int]:
    """best/worst/median 포함 5개 대표 사이클 선택."""
    cycles = df_rmse["cycle"].tolist()
    best = int(df_rmse.loc[df_rmse.rmse_mv.idxmin(), "cycle"])
    worst = int(df_rmse.loc[df_rmse.rmse_mv.idxmax(), "cycle"])
    median = int(df_rmse.iloc[len(df_rmse) // 2]["cycle"])
    picks = sorted(set([1, best, median, worst, cycles[-1]]))[:5]
    return picks


def _load_cycle_validation_data(profile, cyc_idx: int, inf_dir: Path):
    """
    실측 데이터와 predicted_params를 사용해 시뮬레이션을 재실행,
    V/Q/t 배열 반환. (forward_data.json이 없어 재계산)
    """
    from lmro2phase.physics.ocp_grid import OCPGrid
    from lmro2phase.physics.positive_2phase_factory import (
        TwoPhaseOCPParams, make_effective_ocp,
        make_effective_diffusivity, make_effective_radius,
    )
    from lmro2phase.physics.lmr_parameter_set import build_pybamm_halfcell_params
    from lmro2phase.physics.halfcell_model_factory import build_halfcell_model, ModelType
    from lmro2phase.physics.simulator import run_current_drive

    cyc = profile.select_cycle(cyc_idx)
    params_f = inf_dir / f"cycle_{cyc_idx:03d}" / "predicted_params.json"
    ocp_r3m_f = inf_dir / f"cycle_{cyc_idx:03d}" / "predicted_ocp_R3m.csv"
    ocp_c2m_f = inf_dir / f"cycle_{cyc_idx:03d}" / "predicted_ocp_C2m.csv"

    if not params_f.exists():
        return cyc, None

    with open(params_f) as f:
        pp = json.load(f)

    # phase fraction 정규화
    f_r3m = float(pp["frac_R3m"])
    f_c2m = float(pp["frac_C2m"])
    total = f_r3m + f_c2m
    if abs(total - 1.0) > 0.05:
        f_r3m /= total
        f_c2m /= total

    df_r3m = pd.read_csv(ocp_r3m_f)
    df_c2m = pd.read_csv(ocp_c2m_f)

    ocp_r3m_grid = OCPGrid(sto=df_r3m["stoichiometry"].values,
                            voltage=df_r3m["voltage_v"].values)
    ocp_c2m_grid = OCPGrid(sto=df_c2m["stoichiometry"].values,
                            voltage=df_c2m["voltage_v"].values)

    two_phase = TwoPhaseOCPParams(
        frac_R3m=f_r3m, frac_C2m=f_c2m,
        U_R3m=ocp_r3m_grid.to_pybamm_interpolant(),
        U_C2m=ocp_c2m_grid.to_pybamm_interpolant(),
        D_R3m=10.0 ** float(pp["log10_D_R3m"]),
        D_C2m=10.0 ** float(pp["log10_D_C2m"]),
        R_R3m=10.0 ** float(pp["log10_R_R3m"]),
        R_C2m=10.0 ** float(pp["log10_R_C2m"]),
    )
    pv = build_pybamm_halfcell_params(
        ocp_fn=make_effective_ocp(two_phase),
        D_s=make_effective_diffusivity(two_phase),
        R_particle=make_effective_radius(two_phase),
        contact_resistance=10.0 ** float(pp["log10_contact_resistance"]),
        capacity_scale=float(pp["capacity_scale"]),
        c_s_0_fraction=0.5 + float(pp["initial_stoichiometry_shift"]),
    )
    model = build_halfcell_model(ModelType.SPMe)
    result = run_current_drive(model, pv, cyc.time_s, cyc.current_a)
    return cyc, result


def _vt_comparison(profile, rep_cycles: list[int], inf_dir: Path) -> str:
    n = len(rep_cycles)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), squeeze=False)
    for col, cyc_idx in enumerate(rep_cycles):
        cyc, result = _load_cycle_validation_data(profile, cyc_idx, inf_dir)
        t_exp = (cyc.time_s - cyc.time_s[0]) / 3600
        axes[0, col].plot(t_exp, cyc.voltage_v, "k-", linewidth=1.5, label="Measured")
        if result and result.ok:
            t_sim = (result.time_s - result.time_s[0]) / 3600
            axes[0, col].plot(t_sim, result.voltage_v, "r--", linewidth=1.5, label="Sim")
            v_sim_interp = np.interp(t_exp, t_sim, result.voltage_v)
            residual_mv = (v_sim_interp - cyc.voltage_v) * 1000
            axes[1, col].plot(t_exp, residual_mv, "b-", linewidth=1.0)
            axes[1, col].axhline(0, color="k", linewidth=0.5, linestyle="--")
            rmse = np.sqrt(np.mean(residual_mv ** 2))
            axes[0, col].set_title(f"Cycle {cyc_idx}\nRMSE={rmse:.1f}mV", fontsize=10)
        else:
            axes[0, col].set_title(f"Cycle {cyc_idx}\n(sim fail)", fontsize=10)
        axes[0, col].set_ylabel("Voltage (V)")
        axes[0, col].legend(fontsize=7)
        axes[0, col].grid(True, alpha=0.3)
        axes[1, col].set_xlabel("Time (h)")
        axes[1, col].set_ylabel("Residual (mV)")
        axes[1, col].grid(True, alpha=0.3)
    plt.tight_layout()
    return savefig("fig_08_vt_comparison.png")


def _qv_comparison(profile, rep_cycles: list[int], inf_dir: Path) -> str:
    n = len(rep_cycles)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 9), squeeze=False)
    fig.text(0.02, 0.75, "Charge Q-V", va="center", rotation="vertical", fontsize=11, fontweight="bold")
    fig.text(0.02, 0.27, "Discharge Q-V", va="center", rotation="vertical", fontsize=11, fontweight="bold")

    for col, cyc_idx in enumerate(rep_cycles):
        cyc, result = _load_cycle_validation_data(profile, cyc_idx, inf_dir)
        t_exp = cyc.time_s
        chg_exp = cyc.current_a < 0
        dchg_exp = cyc.current_a > 0

        # ── 충전 Q-V (row 0) ──
        if chg_exp.any():
            q_chg = half_cycle_q_ah(cyc.current_a, t_exp, chg_exp) * 1000
            axes[0, col].plot(q_chg, cyc.voltage_v[chg_exp], "k-", linewidth=1.5, label="Measured")
        if result and result.ok:
            t_sim = result.time_s
            chg_sim = result.current_a < 0
            if chg_sim.any():
                q_chg_sim = half_cycle_q_ah(result.current_a, t_sim, chg_sim) * 1000
                axes[0, col].plot(q_chg_sim, result.voltage_v[chg_sim], "r--", linewidth=1.5, label="Sim")
        axes[0, col].set_title(f"Cycle {cyc_idx}", fontsize=10)
        axes[0, col].set_xlabel("Charge Capacity (mAh)")
        axes[0, col].set_ylabel("Voltage (V)")
        axes[0, col].legend(fontsize=7)
        axes[0, col].grid(True, alpha=0.3)

        # ── 방전 Q-V (row 1) ──
        if dchg_exp.any():
            q_dchg = half_cycle_q_ah(cyc.current_a, t_exp, dchg_exp) * 1000
            axes[1, col].plot(q_dchg, cyc.voltage_v[dchg_exp], "k-", linewidth=1.5, label="Measured")
        if result and result.ok:
            dchg_sim = result.current_a > 0
            if dchg_sim.any():
                q_dchg_sim = half_cycle_q_ah(result.current_a, t_sim, dchg_sim) * 1000
                axes[1, col].plot(q_dchg_sim, result.voltage_v[dchg_sim], "r--", linewidth=1.5, label="Sim")
        axes[1, col].set_xlabel("Discharge Capacity (mAh)")
        axes[1, col].set_ylabel("Voltage (V)")
        axes[1, col].legend(fontsize=7)
        axes[1, col].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0.04, 0, 1, 1])
    return savefig("fig_09_qv_comparison.png")


def _dqdv_comparison(profile, rep_cycles: list[int], inf_dir: Path) -> str:
    fig, axes = plt.subplots(1, len(rep_cycles), figsize=(4 * len(rep_cycles), 5), squeeze=False)
    for col, cyc_idx in enumerate(rep_cycles):
        cyc, result = _load_cycle_validation_data(profile, cyc_idx, inf_dir)
        t_exp = cyc.time_s
        dchg = cyc.current_a > 0
        win = max(3, min(7, (dchg.sum() // 2) | 1))
        if dchg.sum() > 5:
            q_dchg = half_cycle_q_ah(cyc.current_a, t_exp, dchg)  # 0-리셋, Ah
            v_d, dq_d = dqdv(cyc.voltage_v[dchg], q_dchg, window=win)
            axes[0, col].plot(v_d, -dq_d * 1000, "k-", linewidth=1.5, label="Measured")
        if result and result.ok:
            i_sim = result.current_a
            t_sim = result.time_s
            dchg_sim = i_sim > 0
            win_s = max(3, min(11, (dchg_sim.sum() // 2) | 1))
            if dchg_sim.sum() > 5:
                q_dchg_sim = half_cycle_q_ah(i_sim, t_sim, dchg_sim)
                v_ds, dq_ds = dqdv(result.voltage_v[dchg_sim], q_dchg_sim, window=win_s)
                axes[0, col].plot(v_ds, -dq_ds * 1000, "r--", linewidth=1.5, label="Sim")
        axes[0, col].axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
        axes[0, col].set_title(f"Cycle {cyc_idx} — Discharge dQ/dV", fontsize=10)
        axes[0, col].set_xlabel("Voltage (V)")
        axes[0, col].set_ylabel("dQ/dV (mAh/V)")
        axes[0, col].legend(fontsize=7)
        axes[0, col].grid(True, alpha=0.3)
    fig.suptitle("Discharge dQ/dV Comparison (Measured vs Simulated)", fontsize=12)
    plt.tight_layout()
    return savefig("fig_10_dqdv_comparison.png")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    log.info(f"=== 리포트 생성 시작: {STAMP} ===")

    profile = load_profile()
    cycles = profile.get_cycles()
    n_cyc = len(cycles)

    log.info("입력 데이터 섹션 생성 중...")
    sec2 = section_input_data(profile)

    log.info("Stage 1 섹션 생성 중...")
    sec3 = section_stage1(profile)

    log.info("Stage 2 섹션 생성 중...")
    sec4 = section_stage2(profile)

    log.info("Stage 3 섹션 생성 중...")
    sec5 = section_stage3()

    log.info("Stage 4 섹션 생성 중 (시뮬레이션 재실행 포함)...")
    sec6 = section_stage4(profile)

    # ── 파이프라인 체크리스트 ──
    def check(path_str):
        return "✅" if Path(ROOT / path_str).exists() else "❌"

    checklist = (
        "| 단계 | 스크립트 | 상태 | 주요 출력 |\n"
        "|------|---------|------|----------|\n"
        f"| 0 smoke test | 00_smoke_test_pybamm_halfcell.py | {check('data/smoke_test_report.json')} | smoke_test_report.json |\n"
        f"| 1 데이터 파싱 | 01_parse_toyo_ascii.py | {'✅' if n_cyc > 0 else '❌'} | {n_cyc}사이클 파싱 |\n"
        f"| 1b OCP 피팅 | 02_fit_tanh_ocp.py | {check('data/reports/stage1_fit/best_params.json')} | best_params.json |\n"
        f"| 2 합성 데이터 | 03_generate_synthetic_dataset.py | {check('data/synthetic/params.parquet')} | params.parquet, profiles.zarr |\n"
        f"| 3 DL 학습 | 04_train_inverse_model.py | {check('data/models/inverse/best_model.pt')} | best_model.pt |\n"
        f"| 4a 추론 | 05_infer_lmr_profile.py | {check('data/reports/inference/cycle_001/predicted_params.json')} | cycle_*/predicted_params.json |\n"
        f"| 4b 검증 | 06_forward_validate.py | {check('data/reports/inference/cycle_001/residual_summary.json')} | cycle_*/residual_summary.json |\n"
    )

    # ── 헤더 ──
    header = f"""# LMR 2-Phase Inverse Pipeline — Result Report

**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**데이터**: `data/raw/toyo/Toyo_LMR_HalfCell_Sample_50cycles.csv`
**총 사이클**: {n_cyc}

## § 1. 파이프라인 체크리스트

{checklist}
"""

    # ── 조립 & 저장 ──
    report = header + sec2 + sec3 + sec4 + sec5 + sec6

    # .md 파일을 figures/ 와 같은 폴더에 저장 → 상대경로 figures/fig_XX.png 정상 작동
    report_md = REPORT_DIR / f"result_report_{STAMP}.md"
    report_md.write_text(report, encoding="utf-8")

    log.info(f"리포트 저장: {report_md}")
    log.info(f"이미지:      {FIG_DIR}")
    log.info(f"총 이미지 수: {len(list(FIG_DIR.glob('*.png')))}")


if __name__ == "__main__":
    main()
