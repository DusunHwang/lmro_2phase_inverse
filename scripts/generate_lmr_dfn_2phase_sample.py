"""LMR 2상 DFN 가상 데이터 생성 스크립트.

문헌 기반 LMR 파라미터를 사용하여 PyBaMM DFN 모델로 TOYO-형식 방전 데이터를 생성한다.
생성된 데이터는 LMR 2상 역추정 스크립트(run_lmr_2phase_fit.py)의 입력으로 사용된다.

물리적 근거 (문헌 기반 LMR 파라미터):
  R3m  (고전압 층상구조, R̄3m):
    D ≈ 10⁻¹² ~ 10⁻¹³ cm²/s = 10⁻¹⁶ ~ 10⁻¹⁷ m²/s
    방전 dQ/dV 피크: ~3.7 V
  C2m  (저전압 Li₂MnO₃ 단사정계, C2/m):
    D ≈ 10⁻¹⁴ ~ 10⁻¹⁵ cm²/s = 10⁻¹⁸ ~ 10⁻¹⁹ m²/s   (R3m 대비 100배 느림)
    방전 dQ/dV 피크: ~3.2 V

모델 vs 피팅:
  생성 (truth): PyBaMM DFN  — 전해액 농도 분포, 전극 내 전위 분포 포함
  역추정 (fit): PyBaMM SPMe 또는 DFN

사용법:
  .venv/bin/python scripts/generate_lmr_dfn_2phase_sample.py \\
      --out-dir data/raw/toyo/lmr_dfn_2phase_sample \\
      [--c-rates 0.1 0.33 0.5 1.0] \\
      [--frac-r3m 0.40] [--d-r3m 5e-17] [--d-c2m 5e-19]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pybamm

from lmro2phase.physics.lmr_parameter_set import FARADAY


# ─── OCP 생성 ────────────────────────────────────────────────────────────────

class InterpOCP:
    """sto(화학양론비) → voltage [V] 보간 OCP."""

    def __init__(self, sto: np.ndarray, voltage: np.ndarray, name: str):
        self.sto     = np.asarray(sto, dtype=float)
        self.voltage = np.asarray(voltage, dtype=float)
        self.name    = name

    def numpy(self, sto):
        return np.interp(sto, self.sto, self.voltage)

    def pybamm(self, sto):
        return pybamm.Interpolant(
            self.sto, self.voltage, sto,
            name=f"{self.name}_ocp", interpolator="cubic",
        )


def _gaussian_redox_ocp(
    name: str,
    voltage_low: float,
    voltage_high: float,
    center_voltage: float,
    sigma_voltage: float,
    peak_weight: float = 1.0,
    baseline_weight: float = 0.05,
    n: int = 1200,
) -> InterpOCP:
    """dQ/dV가 Gaussian 분포를 갖는 단조 OCP 생성.

    전압 축 위에서 Gaussian 밀도를 적분하여 sto(x) 를 구성한다.
    """
    u = np.linspace(voltage_low, voltage_high, n)
    density = baseline_weight + peak_weight * np.exp(
        -0.5 * ((u - center_voltage) / sigma_voltage) ** 2
    )
    s = np.concatenate([
        [0.0],
        np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(u)),
    ])
    s /= s[-1]
    sto = 1.0 - s
    order = np.argsort(sto)
    return InterpOCP(sto[order], u[order], name)


def build_gaussian_ocps(
    r3m_center_v: float = 3.70,
    c2m_center_v: float = 3.20,
    r3m_sigma_v:  float = 0.10,
    c2m_sigma_v:  float = 0.15,
) -> tuple[InterpOCP, InterpOCP]:
    """문헌 기반 LMR OCP 생성.

    R3m: 고전압 층상구조 (Ni/Co 산화환원, ~3.7V)
    C2m: 저전압 단사정계 (Mn 산화환원, ~3.2V)
    """
    ocp_R3m = _gaussian_redox_ocp(
        name="R3m",
        voltage_low=2.50, voltage_high=4.65,
        center_voltage=r3m_center_v, sigma_voltage=r3m_sigma_v,
        peak_weight=1.0, baseline_weight=0.045,
    )
    ocp_C2m = _gaussian_redox_ocp(
        name="C2m",
        voltage_low=2.50, voltage_high=4.65,
        center_voltage=c2m_center_v, sigma_voltage=c2m_sigma_v,
        peak_weight=1.0, baseline_weight=0.040,
    )
    return ocp_R3m, ocp_C2m


# ─── DFN 모델 빌더 ───────────────────────────────────────────────────────────

def build_dfn_model() -> pybamm.lithium_ion.DFN:
    """반전지(positive half-cell) DFN 모델, 2상 양극 입자."""
    options = {
        "working electrode":  "positive",
        "particle phases":    ("1", "2"),
        "particle":           "Fickian diffusion",
        "particle size":      "single",
        "surface form":       "differential",
    }
    return pybamm.lithium_ion.DFN(options)


def build_params(
    frac_R3m:  float,
    D_R3m:     float,
    D_C2m:     float,
    R_R3m:     float,
    R_C2m:     float,
    ocp_R3m:   InterpOCP,
    ocp_C2m:   InterpOCP,
    nominal_capacity_ah: float,
    initial_fraction: float = 0.99,
) -> pybamm.ParameterValues:
    """Chen2020 기반 2상 DFN/SPMe 공용 파라미터 세트."""
    frac_C2m = 1.0 - frac_R3m

    c_s_max       = 47500.0    # mol/m³
    electrode_thk = 75.0e-6   # m
    active_total  = 0.665
    nominal_cap_c = nominal_capacity_ah * 3600.0
    electrode_area = nominal_cap_c / (c_s_max * electrode_thk * active_total * FARADAY)
    electrode_side = float(np.sqrt(electrode_area))

    param = pybamm.ParameterValues("Chen2020")
    base_i0 = param["Positive electrode exchange-current density [A.m-2]"]

    updates = {
        # 전지 구조
        "Exchange-current density for lithium metal electrode [A.m-2]": 1.0,
        "Positive electrode thickness [m]":  electrode_thk,
        "Positive electrode porosity":       0.3,
        "Electrode height [m]":              electrode_side,
        "Electrode width [m]":               electrode_side,
        "Nominal cell capacity [A.h]":       nominal_capacity_ah,
        "Upper voltage cut-off [V]":         4.65,
        "Lower voltage cut-off [V]":         2.50,
        "Open-circuit voltage at 0% SOC [V]":   2.50,
        "Open-circuit voltage at 100% SOC [V]": 4.65,
        # Primary (R3m) 파라미터
        "Primary: Positive electrode OCP [V]":                    ocp_R3m.pybamm,
        "Primary: Positive particle diffusivity [m2.s-1]":        D_R3m,
        "Primary: Positive particle radius [m]":                   R_R3m,
        "Primary: Maximum concentration in positive electrode [mol.m-3]":        c_s_max,
        "Primary: Initial concentration in positive electrode [mol.m-3]":        c_s_max * initial_fraction,
        "Primary: Positive electrode active material volume fraction":            active_total * frac_R3m,
        "Primary: Positive electrode exchange-current density [A.m-2]":          base_i0,
        "Primary: Positive electrode OCP entropic change [V.K-1]":              0.0,
        # Secondary (C2m) 파라미터
        "Secondary: Positive electrode OCP [V]":                   ocp_C2m.pybamm,
        "Secondary: Positive particle diffusivity [m2.s-1]":       D_C2m,
        "Secondary: Positive particle radius [m]":                  R_C2m,
        "Secondary: Maximum concentration in positive electrode [mol.m-3]":      c_s_max,
        "Secondary: Initial concentration in positive electrode [mol.m-3]":      c_s_max * initial_fraction,
        "Secondary: Positive electrode active material volume fraction":          active_total * frac_C2m,
        "Secondary: Positive electrode exchange-current density [A.m-2]":        base_i0,
        "Secondary: Positive electrode OCP entropic change [V.K-1]":            0.0,
    }
    param.update(updates, check_already_exists=False)
    return param


# ─── 시뮬레이션 ──────────────────────────────────────────────────────────────

def _classify_modes(current_a: np.ndarray) -> np.ndarray:
    return np.where(current_a < -1e-9, "CC-Chg",
           np.where(current_a >  1e-9, "CC-Dchg", "Rest"))


def _step_indices(modes: np.ndarray) -> np.ndarray:
    out = np.ones(len(modes), dtype=int)
    for i in range(1, len(modes)):
        out[i] = out[i - 1] + (1 if modes[i] != modes[i - 1] else 0)
    return out


def _step_cap_mah(time_s, current_a, steps):
    cap = np.zeros_like(time_s)
    for s in sorted(set(steps.tolist())):
        idx = np.where(steps == s)[0]
        if abs(float(np.mean(current_a[idx]))) < 1e-12:
            cap[idx] = cap[idx[0] - 1] if idx[0] > 0 else 0.0
        else:
            t_step = time_s[idx] - time_s[idx[0]]
            q = np.cumsum(np.abs(current_a[idx]) * np.gradient(t_step) / 3600.0) * 1000.0
            q[0] = 0.0
            cap[idx] = q
    return cap


def simulate(args) -> tuple[pd.DataFrame, list[dict], dict]:
    ocp_R3m, ocp_C2m = build_gaussian_ocps(
        r3m_center_v=args.r3m_center_v,
        c2m_center_v=args.c2m_center_v,
        r3m_sigma_v=args.r3m_sigma_v,
        c2m_sigma_v=args.c2m_sigma_v,
    )

    model  = build_dfn_model()
    params = build_params(
        frac_R3m=args.frac_r3m,
        D_R3m=args.d_r3m, D_C2m=args.d_c2m,
        R_R3m=args.r_r3m, R_C2m=args.r_c2m,
        ocp_R3m=ocp_R3m, ocp_C2m=ocp_C2m,
        nominal_capacity_ah=args.nom_cap_ah,
        initial_fraction=args.initial_fraction,
    )

    solver = pybamm.CasadiSolver(
        mode="safe", rtol=1e-5, atol=1e-7,
        extra_options_setup={"max_num_steps": 20000},
    )

    rows, summaries = [], []
    start_s, data_point = 0.0, 1

    for cycle, c_rate in enumerate(args.c_rates, start=1):
        print(f"  [DFN] cycle {cycle} @ {c_rate}C ...", flush=True)
        experiment = pybamm.Experiment(
            [
                f"Charge at {c_rate:g}C until {args.voltage_upper:g} V",
                f"Rest for {args.rest_min:g} minutes",
                f"Discharge at {c_rate:g}C until {args.voltage_lower:g} V",
                f"Rest for {args.rest_min:g} minutes",
            ],
            period=f"{args.period_s:g} seconds",
        )
        sim = pybamm.Simulation(
            model, parameter_values=params.copy(),
            experiment=experiment, solver=solver,
        )
        sol = sim.solve()

        t_rel  = np.asarray(sol["Time [s]"].entries,    dtype=float)
        v_arr  = np.asarray(sol["Voltage [V]"].entries, dtype=float)
        i_arr  = np.asarray(sol["Current [A]"].entries, dtype=float)
        time_s = t_rel + start_s
        modes  = _classify_modes(i_arr)
        steps  = _step_indices(modes)
        cap    = _step_cap_mah(time_s, i_arr, steps)
        n      = len(time_s)

        rows.append(pd.DataFrame({
            "Data_Point":   np.arange(data_point, data_point + n),
            "Cycle_No":     cycle,
            "Step_No":      steps.astype(int),
            "Mode":         modes,
            "Time(s)":      np.round(time_s, 3),
            "Voltage(V)":   np.round(v_arr, 6),
            "Current(mA)":  np.round(-i_arr * 1000.0, 6),
            "Capacity(mAh)": np.round(cap, 6),
            "C_rate":       c_rate,
        }))
        summaries.append({"cycle": cycle, "c_rate": c_rate,
                          "v_min": float(v_arr.min()), "v_max": float(v_arr.max()),
                          "n_pts": n})
        data_point += n
        start_s = float(time_s[-1]) + args.inter_cycle_rest_s
        print(f"    → {n} pts  V=[{v_arr.min():.3f}, {v_arr.max():.3f}] V", flush=True)

    truth_dict = {
        "frac_R3m":    args.frac_r3m,
        "frac_C2m":    1.0 - args.frac_r3m,
        "D_R3m_m2_s":  args.d_r3m,
        "D_C2m_m2_s":  args.d_c2m,
        "log10_D_R3m": float(np.log10(args.d_r3m)),
        "log10_D_C2m": float(np.log10(args.d_c2m)),
        "R_R3m_m":     args.r_r3m,
        "R_C2m_m":     args.r_c2m,
        "log10_R_R3m": float(np.log10(args.r_r3m)),
        "log10_R_C2m": float(np.log10(args.r_c2m)),
    }

    return pd.concat(rows, ignore_index=True), summaries, truth_dict, ocp_R3m, ocp_C2m


def write_outputs(args, df, summaries, truth_dict, ocp_R3m, ocp_C2m):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rate_suffix = "_".join(f"{r:g}C".replace(".", "p") for r in args.c_rates)
    csv_path = out_dir / f"Toyo_LMR_DFN_2phase_{rate_suffix}.csv"
    df.to_csv(csv_path, index=False)

    sto = np.linspace(0.01, 0.99, 600)
    pd.DataFrame({
        "stoichiometry":     sto,
        "U_R3m_V":          ocp_R3m.numpy(sto),
        "U_C2m_V":          ocp_C2m.numpy(sto),
    }).to_csv(out_dir / "phase_ocp_curves.csv", index=False)

    manifest = {
        "simulation_model":   "PyBaMM DFN (half-cell, working_electrode=positive)",
        "fit_target_model":   "PyBaMM SPMe/DFN (half-cell)",
        "phase_mapping":      {"Primary": "R3m (고전압 층상)", "Secondary": "C2m (저전압 단사정계)"},
        "truth": truth_dict,
        "ocp": {
            "type":         "Gaussian dQ/dV",
            "R3m_center_v": args.r3m_center_v,
            "R3m_sigma_v":  args.r3m_sigma_v,
            "C2m_center_v": args.c2m_center_v,
            "C2m_sigma_v":  args.c2m_sigma_v,
        },
        "protocol": {
            "c_rates":           args.c_rates,
            "voltage_lower_v":   args.voltage_lower,
            "voltage_upper_v":   args.voltage_upper,
            "period_s":          args.period_s,
            "rest_min":          args.rest_min,
            "initial_fraction":  args.initial_fraction,
        },
        "literature_note": (
            "R3m D~10⁻¹²~10⁻¹³ cm²/s=10⁻¹⁶~10⁻¹⁷ m²/s; "
            "C2m D~10⁻¹⁴~10⁻¹⁵ cm²/s=10⁻¹⁸~10⁻¹⁹ m²/s; "
            "R3m OCP peak~3.7V, C2m OCP peak~3.2V (방전 기준)"
        ),
        # run_lmr_2phase_fit.py가 읽는 키
        "target_peaks": {
            "R3m_primary_redox_feature_v":   args.r3m_center_v,
            "R3m_sigma_v":                   args.r3m_sigma_v,
            "C2m_secondary_redox_feature_v": args.c2m_center_v,
            "C2m_sigma_v":                   args.c2m_sigma_v,
        },
        "cycle_summaries": summaries,
    }
    (out_dir / "true_lmr_dfn_parameters.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return csv_path


def parse_args():
    p = argparse.ArgumentParser(
        description="LMR DFN 2상 가상 데이터 생성",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 출력
    p.add_argument("--out-dir", default="data/raw/toyo/lmr_dfn_2phase_sample")
    # 전기화학 프로토콜
    p.add_argument("--c-rates",    type=float, nargs="+", default=[0.1, 0.33, 0.5, 1.0])
    p.add_argument("--voltage-lower",  type=float, default=2.50)
    p.add_argument("--voltage-upper",  type=float, default=4.65)
    p.add_argument("--rest-min",       type=float, default=10.0,  help="사이클 간 휴지 [분]")
    p.add_argument("--period-s",       type=float, default=0.5,   help="데이터 기록 주기 [s]")
    p.add_argument("--inter-cycle-rest-s", type=float, default=600.0)
    p.add_argument("--nom-cap-ah",     type=float, default=0.005, help="공칭 용량 [Ah]")
    p.add_argument("--initial-fraction", type=float, default=0.99,
                   help="초기 stoichiometry (방전 시작 기준 완방전 ≈ 0.99)")
    # 물리 파라미터 (문헌 기반 LMR)
    p.add_argument("--frac-r3m",  type=float, default=0.40,
                   help="R3m 활물질 분율 (나머지는 C2m)")
    p.add_argument("--d-r3m",     type=float, default=5e-17,
                   help="R3m 확산계수 [m²/s]  문헌: ~10⁻¹⁶~10⁻¹⁷")
    p.add_argument("--d-c2m",     type=float, default=5e-19,
                   help="C2m 확산계수 [m²/s]  문헌: ~10⁻¹⁸~10⁻¹⁹")
    p.add_argument("--r-r3m",     type=float, default=1.5e-7,  help="R3m 입자 반경 [m]")
    p.add_argument("--r-c2m",     type=float, default=1.5e-7,  help="C2m 입자 반경 [m]")
    # OCP 파라미터
    p.add_argument("--r3m-center-v", type=float, default=3.70,
                   help="R3m OCP Gaussian 중심 [V]  문헌: 3.6~3.8V")
    p.add_argument("--r3m-sigma-v",  type=float, default=0.10,
                   help="R3m OCP Gaussian σ [V]")
    p.add_argument("--c2m-center-v", type=float, default=3.20,
                   help="C2m OCP Gaussian 중심 [V]  문헌: 3.2~3.3V")
    p.add_argument("--c2m-sigma-v",  type=float, default=0.15,
                   help="C2m OCP Gaussian σ [V]")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("  LMR DFN 2상 가상 데이터 생성")
    print(f"  출력 디렉토리: {args.out_dir}")
    print(f"  모델: PyBaMM DFN (half-cell, 2-phase positive)")
    print(f"  C-rates: {args.c_rates}")
    print()
    print(f"  [진실값] frac_R3m = {args.frac_r3m:.3f}")
    print(f"  [진실값] D_R3m    = {args.d_r3m:.2e} m²/s  (log10={np.log10(args.d_r3m):.2f})")
    print(f"  [진실값] D_C2m    = {args.d_c2m:.2e} m²/s  (log10={np.log10(args.d_c2m):.2f})")
    print(f"  [진실값] R_R3m    = {args.r_r3m:.2e} m")
    print(f"  [진실값] R3m OCP  = {args.r3m_center_v:.2f} V  σ={args.r3m_sigma_v:.3f} V")
    print(f"  [진실값] C2m OCP  = {args.c2m_center_v:.2f} V  σ={args.c2m_sigma_v:.3f} V")
    print("=" * 60)

    df, summaries, truth_dict, ocp_R3m, ocp_C2m = simulate(args)
    csv_path = write_outputs(args, df, summaries, truth_dict, ocp_R3m, ocp_C2m)

    print(f"\n  완료: {csv_path}")
    print(f"  총 행수: {len(df)}")
    print(f"  전압 범위: {df['Voltage(V)'].min():.4f} ~ {df['Voltage(V)'].max():.4f} V")
    print(f"  사이클 요약:")
    for s in summaries:
        print(f"    cycle {s['cycle']} ({s['c_rate']}C): {s['n_pts']} pts  "
              f"V=[{s['v_min']:.3f}, {s['v_max']:.3f}] V")


if __name__ == "__main__":
    main()
