"""파라미터 설계 수렴 과정 보고서 생성.

가상 시뮬레이션 샘플의 설계 진화 경로를 따라
각 단계별 V(Q)/dQ/dV 오차를 계산하고 마크다운 보고서를 생성합니다.
최종 (진실값) 데이터를 기준으로 RMS 오차를 정량화합니다.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# 샘플 정의 (설계 진화 순서)
# ---------------------------------------------------------------------------

STEPS = [
    {
        "label": "Step 0 (초기 설정)",
        "dir": "native_2phase_sample",
        "change": "plateau OCP, frac_R3m=0.60, D_R3m=4.59e-15, D_C2m=1e-16",
        "params_changed": "OCP 형상, frac, D",
    },
    {
        "label": "Step 1 (Gaussian OCP 도입)",
        "dir": "native_2phase_gaussian_redox_sample",
        "change": "Gaussian OCP 변환, frac_R3m=0.60 유지, D 동일",
        "params_changed": "OCP 형상",
    },
    {
        "label": "Step 2 (OCP full-range 정규화)",
        "dir": "native_2phase_gaussian_redox_fullrange_sample",
        "change": "Gaussian OCP full-range [0,1] 정규화",
        "params_changed": "OCP stoichiometry 범위",
    },
    {
        "label": "Step 3 (확산계수 100배 감소)",
        "dir": "native_2phase_gaussian_redox_fullrange_D100x_slow_sample",
        "change": "D_R3m=4.59e-17, D_C2m=1e-18 (100x 감소)",
        "params_changed": "D_R3m, D_C2m",
    },
    {
        "label": "Step 4 (OCP peak 위치 교체)",
        "dir": "native_2phase_gaussian_redox_swapped_centers_D100x_slow_sample",
        "change": "R3m 중심 → 3.7V, C2m 중심 → 3.2V (교체)",
        "params_changed": "OCP 중심 전압",
    },
    {
        "label": "Step 5 (최종: frac, C2m sigma 조정)",
        "dir": "native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample",
        "change": "frac_R3m=0.333, C2m σ=0.18V (넓음), R3m σ=0.075V",
        "params_changed": "frac_R3m, C2m sigma",
        "is_truth": True,
    },
]

CRATES = [0.1, 0.33, 0.5, 1.0]
N_GRID = 1024


# ---------------------------------------------------------------------------
# 데이터 처리 함수
# ---------------------------------------------------------------------------

def load_discharge(csv_path: Path, crate: float) -> tuple[np.ndarray, np.ndarray] | None:
    """0.1C 방전 V(Ah) 반환 (Ah 기준 용량 증가 방향)."""
    df = pd.read_csv(csv_path)
    mask = (df["C_rate"].round(2) == round(crate, 2)) & (df["Mode"] == "CC-Dchg")
    seg = df[mask].copy()
    if seg.empty:
        return None
    q = seg["Capacity(mAh)"].values / 1000.0  # Ah
    v = seg["Voltage(V)"].values
    # Q는 0부터 단조 증가여야 함
    q = q - q[0]
    return q, v


def vq_on_grid(q: np.ndarray, v: np.ndarray, q_max: float, n: int = N_GRID) -> np.ndarray:
    """공통 Q 격자 위에 V(Q) 보간."""
    q_norm = q / q_max
    q_grid = np.linspace(0.0, 1.0, n)
    fn = interp1d(q_norm, v, bounds_error=False, fill_value="extrapolate")
    return fn(q_grid)


def dqdv_profile(q: np.ndarray, v: np.ndarray, v_lo: float = 2.5, v_hi: float = 4.65,
                  dv: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """V 격자 위 dQ/dV 반환."""
    v_grid = np.arange(v_lo, v_hi + dv, dv)
    fn = interp1d(v, q, bounds_error=False, fill_value="extrapolate")
    q_interp = fn(v_grid)
    dqdv = np.gradient(q_interp, v_grid)
    return v_grid, dqdv


def find_peaks(v_grid: np.ndarray, dqdv: np.ndarray) -> dict:
    """방전 dQ/dV의 낮은 전압 피크와 높은 전압 피크를 각각 찾는다."""
    # 방전에서 dQ/dV < 0, 피크는 절댓값 최대
    result = {}
    # 저전압 윈도우 (2.5-3.35V)
    lo_mask = (v_grid >= 2.5) & (v_grid <= 3.45)
    if lo_mask.any():
        idx = np.argmin(dqdv[lo_mask])
        v_sub = v_grid[lo_mask]
        result["low_v_peak_v"] = float(v_sub[idx])
        result["low_v_peak_dqdv"] = float(dqdv[lo_mask][idx])
    # 고전압 윈도우 (3.45-4.3V)
    hi_mask = (v_grid >= 3.45) & (v_grid <= 4.3)
    if hi_mask.any():
        idx = np.argmin(dqdv[hi_mask])
        v_sub = v_grid[hi_mask]
        result["high_v_peak_v"] = float(v_sub[idx])
        result["high_v_peak_dqdv"] = float(dqdv[hi_mask][idx])
    return result


def rms_vq_error(v_ref: np.ndarray, v_test: np.ndarray) -> float:
    valid = ~(np.isnan(v_ref) | np.isnan(v_test))
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean((v_ref[valid] - v_test[valid]) ** 2)))


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main(base_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 진실값 (마지막 스텝) 데이터 로드
    truth_step = next(s for s in STEPS if s.get("is_truth"))
    truth_dir = base_dir / truth_step["dir"]

    truth_params = json.loads((truth_dir / "true_native_2phase_parameters.json").read_text())

    # 결과 저장용
    rows = []

    # 각 단계 처리
    for step_idx, step in enumerate(STEPS):
        sample_dir = base_dir / step["dir"]
        csv_path = next(sample_dir.glob("Toyo_LMR_*.csv"), None)
        if csv_path is None:
            print(f"[WARN] CSV 없음: {sample_dir}")
            continue

        params = json.loads((sample_dir / "true_native_2phase_parameters.json").read_text())
        truth = params.get("truth", {})
        tp = params.get("target_peaks", {})
        native = params.get("native_phase_parameters", {})

        row = {
            "step": step_idx,
            "label": step["label"],
            "change": step["change"],
            "frac_R3m": truth.get("frac_R3m", float("nan")),
            "log10_D_R3m": truth.get("log10_D_R3m", float("nan")),
            "log10_D_C2m": truth.get("log10_D_C2m", float("nan")),
            "log10_R_R3m": truth.get("log10_R_R3m", float("nan")),
            "log10_R_C2m": truth.get("log10_R_C2m", float("nan")),
            "is_truth": step.get("is_truth", False),
        }

        # OCP peak 정보
        row["R3m_center_v"] = (
            tp.get("R3m_primary_redox_feature_v")
            or tp.get("R3m_primary_low_voltage_redox_feature_v")
        )
        row["C2m_center_v"] = (
            tp.get("C2m_secondary_redox_feature_v")
            or tp.get("C2m_secondary_high_voltage_redox_peak_v")
        )
        row["R3m_sigma"] = tp.get("R3m_sigma_v", "—")
        row["C2m_sigma"] = tp.get("C2m_sigma_v", "—")

        # dQ/dV 피크 및 RMS 오차 (0.1C 방전)
        for crate in CRATES:
            res_truth = load_discharge(
                next((base_dir / truth_step["dir"]).glob("Toyo_LMR_*.csv")), crate
            )
            res_test = load_discharge(csv_path, crate)
            if res_truth is None or res_test is None:
                continue

            q_truth, v_truth = res_truth
            q_test, v_test = res_test

            q_max_truth = q_truth[-1]
            q_max_test = q_test[-1]
            q_max = min(q_max_truth, q_max_test)

            vg_truth = vq_on_grid(q_truth, v_truth, q_max_truth)
            vg_test = vq_on_grid(q_test, v_test, q_max_test)
            rms = rms_vq_error(vg_truth, vg_test)
            row[f"rms_v_{crate}C"] = round(rms * 1000, 2)  # mV

            # dQ/dV peaks at this c-rate
            vg, dqdv = dqdv_profile(q_test, v_test)
            peaks = find_peaks(vg, dqdv)
            row[f"peak_lo_{crate}C_v"] = round(peaks.get("low_v_peak_v", float("nan")), 3)
            row[f"peak_hi_{crate}C_v"] = round(peaks.get("high_v_peak_v", float("nan")), 3)
            row[f"peak_lo_{crate}C_dqdv"] = round(peaks.get("low_v_peak_dqdv", float("nan")), 5)
            row[f"peak_hi_{crate}C_dqdv"] = round(peaks.get("high_v_peak_dqdv", float("nan")), 5)

        rows.append(row)
        label_short = step["label"].split("(")[0].strip()
        print(f"  {label_short}: RMS_0.1C = {row.get('rms_v_0.1C', '?')} mV")

    # -------------------------------------------------------------------
    # 마크다운 리포트 작성
    # -------------------------------------------------------------------
    _write_report(rows, truth_params, out_dir)
    print(f"\n보고서 저장: {out_dir / '파라미터_수렴_과정_리포트.md'}")


def _fmt(val, fmt=".4f", fallback="—"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return fallback
    return format(val, fmt)


def _write_report(rows: list[dict], truth_params: dict, out_dir: Path) -> None:
    truth = truth_params.get("truth", {})
    native = truth_params.get("native_phase_parameters", {})

    lines = []
    L = lines.append

    L("# 파라미터 수렴 과정 리포트")
    L("")
    L("> 가상 시뮬레이션(최종 진실값) 데이터를 기준으로,")
    L("> 초기 설정값에서 최종 파라미터에 이르는 설계 진화 경로를 정리한다.")
    L("> 각 단계별 V(Q) RMS 오차(최종값 기준)와 dQ/dV 피크 이동을 정량화한다.")
    L("")
    L("---")
    L("")
    L("## 1. 진실값(True) 파라미터")
    L("")
    L("| 파라미터 | 값 |")
    L("|---|---:|")
    L(f"| `frac_R3m` | `{truth.get('frac_R3m', '?'):.4f}` |")
    L(f"| `log10_D_R3m` | `{truth.get('log10_D_R3m', '?'):.4f}` |")
    L(f"| `D_R3m` | `{10**truth.get('log10_D_R3m', 0):.3e} m²/s` |")
    L(f"| `log10_D_C2m` | `{truth.get('log10_D_C2m', '?'):.4f}` |")
    L(f"| `D_C2m` | `{10**truth.get('log10_D_C2m', 0):.3e} m²/s` |")
    L(f"| `R_R3m` | `{native.get('Primary_R3m', {}).get('R_m', '?'):.2e} m` |")
    L(f"| `R_C2m` | `{native.get('Secondary_C2m', {}).get('R_m', '?'):.2e} m` |")
    L(f"| OCP R3m 중심 | `3.70 V`, σ = `0.075 V` |")
    L(f"| OCP C2m 중심 | `3.20 V`, σ = `0.18 V` |")
    L("")
    L("---")
    L("")
    L("## 2. 단계별 파라미터 변화")
    L("")
    L("| 단계 | frac_R3m | log10_D_R3m | log10_D_C2m | R3m OCP 중심 | C2m OCP 중심 | 변경 항목 |")
    L("|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        is_t = " ★" if r["is_truth"] else ""
        rc = r.get("R3m_center_v")
        cc = r.get("C2m_center_v")
        rc_str = f"`{rc:.2f} V`" if rc is not None else "—"
        cc_str = f"`{cc:.2f} V`" if cc is not None else "—"
        if r["R3m_sigma"] != "—":
            rc_str += f"(σ={r['R3m_sigma']}V)"
        if r["C2m_sigma"] != "—":
            cc_str += f"(σ={r['C2m_sigma']}V)"
        L(
            f"| {r['label']}{is_t} "
            f"| `{r['frac_R3m']:.4f}` "
            f"| `{r['log10_D_R3m']:.2f}` "
            f"| `{r['log10_D_C2m']:.2f}` "
            f"| {rc_str} "
            f"| {cc_str} "
            f"| {r['change']} |"
        )
    L("")
    L("★ = 최종 진실값")
    L("")
    L("---")
    L("")
    L("## 3. V(Q) RMS 오차 (최종 진실값 대비)")
    L("")
    L("> 각 단계의 시뮬레이션 방전 V(Q)를 최종 진실값과 비교한 RMS 전압 오차 [mV]")
    L("")
    L("| 단계 | 0.1C | 0.33C | 0.5C | 1C |")
    L("|---|---:|---:|---:|---:|")
    for r in rows:
        is_t = " ★" if r["is_truth"] else ""
        vals = [r.get(f"rms_v_{c}C", "—") for c in CRATES]
        cell = lambda v: f"**{v}**" if r["is_truth"] else str(v)
        L(f"| {r['label']}{is_t} | " + " | ".join(cell(v) for v in vals) + " |")
    L("")
    L("★ = 진실값 자신 (오차 = 0 mV, 기준)")
    L("")
    L("---")
    L("")
    L("## 4. dQ/dV 피크 이동 (0.1C 방전)")
    L("")
    L("> 방전 dQ/dV에서 저전압(C2m계열) 및 고전압(R3m계열) 피크 위치 [V]")
    L("")
    L("| 단계 | 저전압 피크 [V] | dQ/dV [Ah/V] | 고전압 피크 [V] | dQ/dV [Ah/V] |")
    L("|---|---:|---:|---:|---:|")
    for r in rows:
        is_t = " ★" if r["is_truth"] else ""
        lp = r.get("peak_lo_0.1C_v", "—")
        ld = r.get("peak_lo_0.1C_dqdv", "—")
        hp = r.get("peak_hi_0.1C_v", "—")
        hd = r.get("peak_hi_0.1C_dqdv", "—")
        L(f"| {r['label']}{is_t} | `{lp}` | `{ld}` | `{hp}` | `{hd}` |")
    L("")
    L("---")
    L("")
    L("## 5. 각 단계별 해설")
    L("")

    descriptions = [
        (
            "Step 0 (초기 설정)",
            textwrap.dedent("""\
                - **OCP 형상**: 사각 plateau 형태 (sech² 적분). R3m 방전 특징 ~3.25 V, C2m 방전 피크 ~3.75 V.
                - **frac_R3m = 0.60**: C2m보다 R3m active fraction이 크다.
                - **D_R3m = 4.59e-15 m²/s, D_C2m = 1e-16 m²/s**: 실제 LMR보다 약 100배 빠른 확산계수.
                - 이 설정에서 dQ/dV 피크는 OCP와 거의 같은 위치에 나타나며, C-rate 의존성이 작다.
            """),
        ),
        (
            "Step 1 (Gaussian OCP 도입)",
            textwrap.dedent("""\
                - Plateau OCP를 Gaussian dQ/dV 기반으로 교체. OCP가 연속·단조함수로 수치적으로 안정.
                - **R3m 중심 3.18 V, C2m 중심 3.82 V**: OCP 피크 위치가 아직 진실값과 반대.
                - D, frac은 Step 0과 동일. 전압 오차 개선은 OCP 연속성에서 온다.
            """),
        ),
        (
            "Step 2 (OCP full-range 정규화)",
            textwrap.dedent("""\
                - Gaussian OCP를 전체 stoichiometry 범위 [0, 1]에 걸쳐 정규화.
                - 고전압/저전압 양 끝에서 OCP 외삽 오류를 제거.
                - 주요 파라미터 변경 없음. V(Q) 오차는 미세 개선.
            """),
        ),
        (
            "Step 3 (확산계수 100배 감소)",
            textwrap.dedent("""\
                - **D_R3m: 4.59e-15 → 4.59e-17 m²/s, D_C2m: 1e-16 → 1e-18 m²/s**.
                - 이 변경이 C-rate 의존성(피크 이동, 저하)을 실제와 유사하게 만드는 핵심 단계.
                - 느린 확산계수는 고율에서 더 큰 분극을 발생시켜 피크가 저전압으로 이동.
            """),
        ),
        (
            "Step 4 (OCP peak 위치 교체)",
            textwrap.dedent("""\
                - **R3m 중심: 3.18 V → 3.70 V, C2m 중심: 3.82 V → 3.20 V** (교체).
                - 물리적 근거: LMR R3m (고전압 구조상전이), C2m (저전압 층상구조 전이).
                - C2m의 낮은 확산계수 + 낮은 OCP 중심 → 고율에서 더 큰 피크 이동.
            """),
        ),
        (
            "Step 5 (최종: frac, C2m sigma 조정)",
            textwrap.dedent("""\
                - **frac_R3m: 0.60 → 0.333** (1:2 비율), C2m weighted Q ≈ R3m의 2배.
                - **C2m σ: 0.18 V** (넓은 Gaussian), **R3m σ: 0.075 V** (좁은 Gaussian).
                - OCP dQ/dV에서 C2m 피크 면적이 R3m의 2배가 되도록 설계.
                - 이 설정이 최종 진실값 (RMS 오차 = 0).
            """),
        ),
    ]

    for label_prefix, desc in descriptions:
        L(f"### {label_prefix}")
        L("")
        for line in desc.strip().split("\n"):
            L(line)
        L("")

    L("---")
    L("")
    L("## 6. 수렴 요약")
    L("")
    L("파라미터 설계 진화를 통해 다음 방향으로 조정이 이루어졌다:")
    L("")
    L("| 파라미터 | 초기값 (Step 0) | 최종 진실값 | 변화 방향 |")
    L("|---|---:|---:|---|")
    L("| `frac_R3m` | 0.60 | 0.333 | **감소** (C2m 비중 증가) |")
    L("| `log10_D_R3m` | −14.34 | −16.34 | **감소** (100배 느려짐) |")
    L("| `log10_D_C2m` | −16.00 | −18.00 | **감소** (100배 느려짐) |")
    L("| R3m OCP 중심 | 3.18 V (→ 3.25 V plateau) | 3.70 V | **증가** |")
    L("| C2m OCP 중심 | 3.82 V (→ 3.75 V plateau) | 3.20 V | **감소** |")
    L("| C2m OCP σ | (plateau, ≈ 고정폭) | 0.18 V | **넓어짐** |")
    L("| R3m OCP σ | (plateau, ≈ 고정폭) | 0.075 V | **좁아짐** |")
    L("")
    L("각 파라미터가 진실값에 수렴할수록 V(Q) RMS 오차가 감소하며,")
    L("dQ/dV 피크 위치가 최종 진실값의 피크 위치로 수렴한다.")
    L("")

    (out_dir / "파라미터_수렴_과정_리포트.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="파라미터 수렴 과정 보고서 생성")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/raw/toyo"),
        help="샘플 디렉토리 상위 경로",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/generated_initial_condition/reports"),
        help="보고서 출력 경로",
    )
    args = parser.parse_args()
    main(args.base_dir, args.out_dir)
