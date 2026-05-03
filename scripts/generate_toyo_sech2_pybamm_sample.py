"""
Generate TOYO-format LMR half-cell charge/discharge sample data with PyBaMM.

The phase OCPs are represented as integrated sech^2 peaks:

    dU/dx = b1 + sum_i A_i / w_i * sech^2((x - c_i) / w_i)
    U(x)  = b0 + b1*x + sum_i A_i * tanh((x - c_i) / w_i)

This script writes:
  - TOYO-style CSV for 0.1C, 0.33C, 0.5C, 1C, 2C cycles
  - true parameter JSON, including literature/assumption notes
  - dQ/dV/OCP basis grids
  - round-trip parse and recovery sanity-check JSON
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from lmro2phase.io.toyo_ascii import ToyoAsciiParser
from lmro2phase.physics.halfcell_model_factory import build_halfcell_model
from lmro2phase.physics.lmr_parameter_set import build_pybamm_halfcell_params
from lmro2phase.physics.ocp_tanh import TanhOCPParams, make_tanh_ocp_numpy, make_tanh_ocp_pybamm
from lmro2phase.physics.positive_2phase_factory import (
    TwoPhaseOCPParams,
    make_effective_diffusivity,
    make_effective_ocp,
    make_effective_radius,
)
from lmro2phase.physics.simulator import run_experiment


@dataclass(frozen=True)
class LiteratureBackedTruth:
    frac_R3m: float
    log10_D_R3m: float
    log10_R_R3m: float
    log10_D_C2m: float
    log10_R_C2m: float
    log10_contact_resistance: float
    capacity_scale: float
    c_s_0_fraction: float
    nominal_capacity_ah: float

    @property
    def frac_C2m(self) -> float:
        return 1.0 - self.frac_R3m

    @property
    def D_R3m(self) -> float:
        return 10.0 ** self.log10_D_R3m

    @property
    def D_C2m(self) -> float:
        return 10.0 ** self.log10_D_C2m

    @property
    def R_R3m(self) -> float:
        return 10.0 ** self.log10_R_R3m

    @property
    def R_C2m(self) -> float:
        return 10.0 ** self.log10_R_C2m

    @property
    def contact_resistance(self) -> float:
        return 10.0 ** self.log10_contact_resistance


def default_truth() -> tuple[LiteratureBackedTruth, TanhOCPParams, TanhOCPParams]:
    """Return one deterministic LMR surrogate truth set.

    Notes:
    - R3m values are aligned with nanoscale Li-rich LMR reports
      (100-300 nm particles and DLi around 4.59e-11 cm2/s = 4.59e-15 m2/s).
    - C2m values are a slower/larger effective surrogate phase within reported
      micron-grain kinetic ranges. These are model-identification targets, not
      direct crystallographic measurements.
    """
    truth = LiteratureBackedTruth(
        frac_R3m=0.62,
        log10_D_R3m=math.log10(4.59e-15),
        log10_R_R3m=math.log10(1.50e-7),
        log10_D_C2m=math.log10(1.00e-16),
        log10_R_C2m=math.log10(1.00e-6),
        log10_contact_resistance=-3.0,
        capacity_scale=1.0,
        c_s_0_fraction=0.99,
        nominal_capacity_ah=0.005,
    )

    # Positive electrode convention: larger stoichiometry is lower voltage.
    ocp_R3m = TanhOCPParams(
        b0=4.34,
        b1=-1.08,
        amps=np.array([0.105, 0.065, -0.035, 0.040]),
        centers=np.array([0.18, 0.38, 0.64, 0.82]),
        widths=np.array([0.040, 0.055, 0.075, 0.060]),
        voltage_lower=2.2,
        voltage_upper=4.8,
    )
    ocp_C2m = TanhOCPParams(
        b0=3.90,
        b1=-0.72,
        amps=np.array([0.145, -0.040, 0.060, 0.030]),
        centers=np.array([0.22, 0.48, 0.70, 0.88]),
        widths=np.array([0.055, 0.085, 0.070, 0.065]),
        voltage_lower=2.2,
        voltage_upper=4.8,
    )
    return truth, ocp_R3m, ocp_C2m


def scale_ocps_to_voltage_window(
    truth: LiteratureBackedTruth,
    ocp_R3m: TanhOCPParams,
    ocp_C2m: TanhOCPParams,
    voltage_lower: float,
    voltage_upper: float,
) -> tuple[TanhOCPParams, TanhOCPParams]:
    """Linearly scale both phase OCPs so U_eff spans the requested voltage window."""
    sto = np.linspace(0.01, 0.99, 1000)
    u_r = make_tanh_ocp_numpy(ocp_R3m)(sto)
    u_c = make_tanh_ocp_numpy(ocp_C2m)(sto)
    u_eff = truth.frac_R3m * u_r + truth.frac_C2m * u_c
    source_min = float(np.min(u_eff))
    source_max = float(np.max(u_eff))
    scale = (voltage_upper - voltage_lower) / (source_max - source_min)
    offset = voltage_lower - scale * source_min

    def transform(p: TanhOCPParams) -> TanhOCPParams:
        return TanhOCPParams(
            b0=offset + scale * p.b0,
            b1=scale * p.b1,
            amps=scale * p.amps,
            centers=p.centers.copy(),
            widths=p.widths.copy(),
            voltage_lower=voltage_lower,
            voltage_upper=voltage_upper,
        )

    return transform(ocp_R3m), transform(ocp_C2m)


def sech2(z: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(z) ** 2


def ocp_slope_from_sech2(params: TanhOCPParams, sto: np.ndarray) -> np.ndarray:
    slope = np.full_like(sto, params.b1, dtype=float)
    for amp, center, width in zip(params.amps, params.centers, params.widths):
        slope += amp / width * sech2((sto - center) / width)
    return slope


def build_two_phase(truth: LiteratureBackedTruth, ocp_R3m: TanhOCPParams, ocp_C2m: TanhOCPParams):
    return TwoPhaseOCPParams(
        frac_R3m=truth.frac_R3m,
        frac_C2m=truth.frac_C2m,
        U_R3m=make_tanh_ocp_pybamm(ocp_R3m),
        U_C2m=make_tanh_ocp_pybamm(ocp_C2m),
        D_R3m=truth.D_R3m,
        D_C2m=truth.D_C2m,
        R_R3m=truth.R_R3m,
        R_C2m=truth.R_C2m,
    )


def simulate_rates(args, truth: LiteratureBackedTruth, ocp_R3m: TanhOCPParams, ocp_C2m: TanhOCPParams):
    two_phase = build_two_phase(truth, ocp_R3m, ocp_C2m)
    params = build_pybamm_halfcell_params(
        ocp_fn=make_effective_ocp(two_phase),
        D_s=make_effective_diffusivity(two_phase),
        R_particle=make_effective_radius(two_phase),
        contact_resistance=truth.contact_resistance,
        capacity_scale=truth.capacity_scale,
        c_s_0_fraction=truth.c_s_0_fraction,
    )
    model = build_halfcell_model(args.model)

    all_rows: list[pd.DataFrame] = []
    start_s = 0.0
    data_point = 1
    for cycle, c_rate in enumerate(args.c_rates, start=1):
        import pybamm

        experiment = pybamm.Experiment(
            [
                f"Charge at {c_rate:g}C until {args.voltage_upper:g} V",
                f"Rest for {args.rest_minutes:g} minutes",
                f"Discharge at {c_rate:g}C until {args.voltage_lower:g} V",
                f"Rest for {args.rest_minutes:g} minutes",
            ],
            period=f"{args.period_s:g} seconds",
        )
        result = run_experiment(model, params.copy(), experiment)
        if not result.ok:
            raise RuntimeError(f"{c_rate}C PyBaMM simulation failed: {result.error}")

        out_t = np.asarray(result.time_s) + start_s
        out_v = np.asarray(result.voltage_v)
        out_i = np.asarray(result.current_a)
        mode_interp = np.where(
            out_i < -1e-9,
            "CC-Chg",
            np.where(out_i > 1e-9, "CC-Dchg", "Rest"),
        )
        step_interp = np.ones_like(out_t, dtype=int)
        step = 1
        for j in range(1, len(mode_interp)):
            if mode_interp[j] != mode_interp[j - 1]:
                step += 1
            step_interp[j] = step

        cap_mah = np.zeros_like(out_t)
        for step in sorted(set(step_interp.tolist())):
            mask = step_interp == step
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            if abs(float(np.mean(out_i[idx]))) < 1e-12:
                cap_mah[idx] = cap_mah[idx[0] - 1] if idx[0] > 0 else 0.0
            else:
                t_step = out_t[idx] - out_t[idx][0]
                q = np.cumsum(np.abs(out_i[idx]) * np.gradient(t_step) / 3600.0) * 1000.0
                q[0] = 0.0
                cap_mah[idx] = q

        n = len(out_t)
        df = pd.DataFrame(
            {
                "Data_Point": np.arange(data_point, data_point + n),
                "Cycle_No": cycle,
                "Step_No": step_interp.astype(int),
                "Mode": mode_interp,
                "Time(s)": np.round(out_t, 3),
                "Voltage(V)": np.round(out_v, 6),
                # TOYO convention: charge positive, discharge negative.
                "Current(mA)": np.round(-out_i * 1000.0, 6),
                "Capacity(mAh)": np.round(cap_mah, 6),
                "C_rate": c_rate,
            }
        )
        all_rows.append(df)
        data_point += n
        start_s = float(out_t[-1] + args.inter_cycle_rest_s)

    return pd.concat(all_rows, ignore_index=True), two_phase


def write_basis(out_dir: Path, truth, ocp_R3m, ocp_C2m, two_phase):
    sto = np.linspace(0.01, 0.99, 600)
    r3m_u = make_tanh_ocp_numpy(ocp_R3m)(sto)
    c2m_u = make_tanh_ocp_numpy(ocp_C2m)(sto)
    r3m_slope = ocp_slope_from_sech2(ocp_R3m, sto)
    c2m_slope = ocp_slope_from_sech2(ocp_C2m, sto)
    eff_u = truth.frac_R3m * r3m_u + truth.frac_C2m * c2m_u
    eff_slope = truth.frac_R3m * r3m_slope + truth.frac_C2m * c2m_slope
    basis = pd.DataFrame(
        {
            "stoichiometry": sto,
            "U_R3m_V": r3m_u,
            "U_C2m_V": c2m_u,
            "U_effective_V": eff_u,
            "dUdx_R3m_from_sech2": r3m_slope,
            "dUdx_C2m_from_sech2": c2m_slope,
            "dUdx_effective_from_sech2": eff_slope,
            "relative_dQdV_effective": np.where(np.abs(eff_slope) > 1e-9, 1.0 / eff_slope, np.nan),
        }
    )
    basis.to_csv(out_dir / "ocp_dqdv_sech2_basis.csv", index=False)

    def tanh_dict(p: TanhOCPParams):
        return {
            "b0": p.b0,
            "b1": p.b1,
            "amps": p.amps.tolist(),
            "centers": p.centers.tolist(),
            "widths": p.widths.tolist(),
            "voltage_lower": p.voltage_lower,
            "voltage_upper": p.voltage_upper,
        }

    manifest = {
        "truth": asdict(truth) | {"frac_C2m": truth.frac_C2m},
        "effective": {
            "D_eff_m2_s": make_effective_diffusivity(two_phase),
            "R_eff_m": make_effective_radius(two_phase),
        },
        "ocp_definition": {
            "form": "U(x)=b0+b1*x+sum(A_i*tanh((x-c_i)/w_i)); dU/dx=b1+sum((A_i/w_i)*sech^2((x-c_i)/w_i))",
            "R3m": tanh_dict(ocp_R3m),
            "C2m": tanh_dict(ocp_C2m),
        },
        "literature_notes": [
            "Argonne summary for Li1.2Mn0.54Ni0.13Co0.13O2 CP-FD reports 100-300 nm particles and DLi=4.59e-11 cm2/s.",
            "Energy Storage Materials 2022 reports micron-sized Li-rich layered oxides with diffusion on the order of 1e-12 cm2/s and nanosized grains around 1e-14 cm2/s.",
            "Phase-specific R3m/C2m values here are effective surrogate targets for inverse-model testing, not direct measured crystallographic phase constants.",
        ],
    }
    (out_dir / "true_parameters.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def roundtrip_check(out_dir: Path, csv_path: Path):
    profile = ToyoAsciiParser(ROOT / "configs/toyo_ascii.yaml").parse(csv_path)
    summary = {
        "parsed_points": int(len(profile)),
        "cycles": profile.get_cycles(),
        "segments": len(profile.segments),
        "voltage_min_v": float(np.min(profile.voltage_v)),
        "voltage_max_v": float(np.max(profile.voltage_v)),
        "pybamm_current_min_a_after_parse": float(np.min(profile.current_a)),
        "pybamm_current_max_a_after_parse": float(np.max(profile.current_a)),
        "recovery_interpretation": (
            "Round-trip parse succeeded. With the current FallbackA model, the generated data "
            "directly identifies effective OCP, D_eff, and R_eff. Individual phase D/R values "
            "are only recoverable if the inverse model keeps the same phase-fraction/OCP prior; "
            "otherwise D_eff and R_eff are the identifiable quantities."
        ),
    }
    (out_dir / "roundtrip_recovery_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw/toyo/sech2_pybamm_sample")
    parser.add_argument("--model", default="SPM", choices=["SPM", "SPMe", "DFN"])
    parser.add_argument("--c-rates", type=float, nargs="+", default=[0.1, 0.33, 0.5, 1.0, 2.0])
    parser.add_argument("--voltage-lower", type=float, default=2.5)
    parser.add_argument("--voltage-upper", type=float, default=4.65)
    parser.add_argument("--rest-minutes", type=float, default=10.0)
    parser.add_argument("--period-s", type=float, default=5.0)
    parser.add_argument("--inter-cycle-rest-s", type=float, default=600.0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    truth, ocp_R3m, ocp_C2m = default_truth()
    ocp_R3m, ocp_C2m = scale_ocps_to_voltage_window(
        truth, ocp_R3m, ocp_C2m, args.voltage_lower, args.voltage_upper
    )
    df, two_phase = simulate_rates(args, truth, ocp_R3m, ocp_C2m)

    csv_path = out_dir / "Toyo_LMR_sech2_PyBaMM_0p1C_0p33C_0p5C_1C_2C.csv"
    df.to_csv(csv_path, index=False)
    write_basis(out_dir, truth, ocp_R3m, ocp_C2m, two_phase)
    roundtrip_check(out_dir, csv_path)

    print(f"wrote {csv_path}")
    print(f"rows {len(df)}")
    print(f"voltage range {df['Voltage(V)'].min():.4f}..{df['Voltage(V)'].max():.4f} V")
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()
