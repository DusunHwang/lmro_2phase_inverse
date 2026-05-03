"""
Generate TOYO-format data with PyBaMM native positive-electrode 2-phase particles.

Unlike generate_toyo_sech2_pybamm_sample.py, this script does not collapse
transport to D_eff/R_eff. It injects separate phase parameters:

  Primary   phase: R3m OCP, D_R3m, R_R3m, active fraction
  Secondary phase: C2m OCP, D_C2m, R_C2m, active fraction
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pybamm

from lmro2phase.io.toyo_ascii import ToyoAsciiParser
from lmro2phase.physics.lmr_parameter_set import FARADAY
from lmro2phase.physics.ocp_tanh import TanhOCPParams, make_tanh_ocp_numpy, make_tanh_ocp_pybamm


def _load_effective_generator():
    path = ROOT / "scripts" / "generate_toyo_sech2_pybamm_sample.py"
    spec = importlib.util.spec_from_file_location("sech2_generator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sech2_generator"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class InterpOCP:
    def __init__(self, sto: np.ndarray, voltage: np.ndarray, name: str):
        self.sto = np.asarray(sto, dtype=float)
        self.voltage = np.asarray(voltage, dtype=float)
        self.name = name

    def numpy(self, sto):
        return np.interp(sto, self.sto, self.voltage)

    def pybamm(self, sto):
        return pybamm.Interpolant(
            self.sto,
            self.voltage,
            sto,
            name=f"{self.name}_ocp",
            interpolator="cubic",
        )


def _plateau_ocp(
    name: str,
    voltage_low: float,
    voltage_high: float,
    peak_voltage: float,
    plateau_width: float,
    plateau_depth: float = 0.72,
    n: int = 800,
) -> InterpOCP:
    """Monotonic OCP with a dQ/dV peak near peak_voltage.

    s = 1 - sto is delithiation progress. A local minimum in dU/ds creates
    a voltage plateau and therefore a dQ/dV peak.
    """
    s = np.linspace(0.0, 1.0, n)
    target = np.clip((peak_voltage - voltage_low) / (voltage_high - voltage_low), 0.08, 0.92)
    duds = 1.0 - plateau_depth * np.exp(-0.5 * ((s - target) / plateau_width) ** 2)
    duds += 0.10 * np.exp(-0.5 * ((s - 0.06) / 0.045) ** 2)
    duds += 0.10 * np.exp(-0.5 * ((s - 0.94) / 0.045) ** 2)
    u = np.concatenate([[0.0], np.cumsum(0.5 * (duds[1:] + duds[:-1]) * np.diff(s))])
    u = voltage_low + (voltage_high - voltage_low) * u / u[-1]
    sto = 1.0 - s
    order = np.argsort(sto)
    return InterpOCP(sto[order], u[order], name)


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
    """Monotonic OCP whose ideal phase dQ/dV is Gaussian-like on voltage.

    u is voltage and s = 1 - sto is delithiation progress. Defining ds/du as
    baseline + Gaussian and then inverting s(u) gives an OCP whose ideal
    capacity distribution is Gaussian on a dQ/dV plot.
    """
    u = np.linspace(voltage_low, voltage_high, n)
    redox_density = baseline_weight + peak_weight * np.exp(
        -0.5 * ((u - center_voltage) / sigma_voltage) ** 2
    )
    s = np.concatenate([[0.0], np.cumsum(0.5 * (redox_density[1:] + redox_density[:-1]) * np.diff(u))])
    s = s / s[-1]
    sto = 1.0 - s
    order = np.argsort(sto)
    return InterpOCP(sto[order], u[order], name)


def target_peak_native_ocps():
    """Phase OCPs designed for R3m shoulder near 3.25 V and C2m peak near 3.75 V on discharge."""
    ocp_R3m = _plateau_ocp(
        name="R3m",
        voltage_low=2.50,
        voltage_high=3.10,
        peak_voltage=2.80,
        plateau_width=0.095,
        plateau_depth=0.88,
    )
    ocp_C2m = _plateau_ocp(
        name="C2m",
        voltage_low=3.22,
        voltage_high=4.62,
        peak_voltage=3.95,
        plateau_width=0.080,
        plateau_depth=0.90,
    )
    return ocp_R3m, ocp_C2m


def gaussian_redox_native_ocps(
    r3m_center_v: float = 3.18,
    c2m_center_v: float = 3.82,
    r3m_sigma_v: float = 0.075,
    c2m_sigma_v: float = 0.080,
    r3m_peak_weight: float = 1.0,
    c2m_peak_weight: float = 1.0,
):
    """Phase OCPs with Gaussian redox distributions in dQ/dV space."""
    ocp_R3m = _gaussian_redox_ocp(
        name="R3m",
        voltage_low=2.50,
        voltage_high=4.65,
        center_voltage=r3m_center_v,
        sigma_voltage=r3m_sigma_v,
        peak_weight=r3m_peak_weight,
        baseline_weight=0.045,
    )
    ocp_C2m = _gaussian_redox_ocp(
        name="C2m",
        voltage_low=2.50,
        voltage_high=4.65,
        center_voltage=c2m_center_v,
        sigma_voltage=c2m_sigma_v,
        peak_weight=c2m_peak_weight,
        baseline_weight=0.040,
    )
    return ocp_R3m, ocp_C2m


def monotonic_native_ocps(truth, voltage_lower: float, voltage_upper: float):
    """Build phase OCPs with dU/dx <= 0 for positive-electrode stoichiometry.

    The previous native sample reused a C2m tanh basis containing broad
    positive dU/dx intervals. In a positive electrode this can make terminal
    voltage reverse during CC charge/discharge. Here all tanh amplitudes are
    non-positive and b1 is negative, so each phase OCP is monotonic decreasing
    with lithiation stoichiometry before and after linear voltage-window scaling.
    """
    ocp_R3m = TanhOCPParams(
        b0=4.35,
        b1=-0.70,
        amps=np.array([-0.090, -0.060, -0.045, -0.035]),
        centers=np.array([0.18, 0.38, 0.64, 0.82]),
        widths=np.array([0.040, 0.055, 0.075, 0.060]),
        voltage_lower=voltage_lower,
        voltage_upper=voltage_upper,
    )
    ocp_C2m = TanhOCPParams(
        b0=4.05,
        b1=-0.55,
        amps=np.array([-0.120, -0.055, -0.050, -0.030]),
        centers=np.array([0.22, 0.48, 0.70, 0.88]),
        widths=np.array([0.055, 0.085, 0.070, 0.065]),
        voltage_lower=voltage_lower,
        voltage_upper=voltage_upper,
    )

    sto = np.linspace(0.01, 0.99, 1000)
    u_r = make_tanh_ocp_numpy(ocp_R3m)(sto)
    u_c = make_tanh_ocp_numpy(ocp_C2m)(sto)
    u_eff = truth.frac_R3m * u_r + truth.frac_C2m * u_c
    scale = (voltage_upper - voltage_lower) / (float(np.max(u_eff)) - float(np.min(u_eff)))
    offset = voltage_lower - scale * float(np.min(u_eff))

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


def build_native_model():
    options = {
        "working electrode": "positive",
        "particle phases": ("1", "2"),
        "particle": "Fickian diffusion",
        "particle size": "single",
        "surface form": "differential",
    }
    return pybamm.lithium_ion.SPM(options)


def build_native_params(truth, ocp_R3m, ocp_C2m, initial_fraction: float = 0.98):
    param = pybamm.ParameterValues("Chen2020")

    c_s_max = 47500.0
    electrode_thickness = 75.0e-6
    active_total = 0.665
    nominal_cap = truth.nominal_capacity_ah * truth.capacity_scale
    nominal_cap_c = nominal_cap * 3600.0
    electrode_area = nominal_cap_c / (c_s_max * electrode_thickness * active_total * FARADAY)
    electrode_side = float(np.sqrt(electrode_area))
    base_exchange = param["Positive electrode exchange-current density [A.m-2]"]

    updates = {
        "Exchange-current density for lithium metal electrode [A.m-2]": 1.0,
        "Positive electrode thickness [m]": electrode_thickness,
        "Positive electrode porosity": 0.3,
        "Electrode height [m]": electrode_side,
        "Electrode width [m]": electrode_side,
        "Nominal cell capacity [A.h]": nominal_cap,
        "Upper voltage cut-off [V]": 4.65,
        "Lower voltage cut-off [V]": 2.5,
        "Open-circuit voltage at 0% SOC [V]": 2.5,
        "Open-circuit voltage at 100% SOC [V]": 4.65,
        "Primary: Positive electrode OCP [V]": ocp_R3m.pybamm,
        "Secondary: Positive electrode OCP [V]": ocp_C2m.pybamm,
        "Primary: Positive particle diffusivity [m2.s-1]": truth.D_R3m,
        "Secondary: Positive particle diffusivity [m2.s-1]": truth.D_C2m,
        "Primary: Positive particle radius [m]": truth.R_R3m,
        "Secondary: Positive particle radius [m]": truth.R_C2m,
        "Primary: Maximum concentration in positive electrode [mol.m-3]": c_s_max,
        "Secondary: Maximum concentration in positive electrode [mol.m-3]": c_s_max,
        "Primary: Initial concentration in positive electrode [mol.m-3]": c_s_max * initial_fraction,
        "Secondary: Initial concentration in positive electrode [mol.m-3]": c_s_max * initial_fraction,
        "Primary: Positive electrode active material volume fraction": active_total * truth.frac_R3m,
        "Secondary: Positive electrode active material volume fraction": active_total * truth.frac_C2m,
        "Primary: Positive electrode exchange-current density [A.m-2]": base_exchange,
        "Secondary: Positive electrode exchange-current density [A.m-2]": base_exchange,
        "Primary: Positive electrode OCP entropic change [V.K-1]": 0.0,
        "Secondary: Positive electrode OCP entropic change [V.K-1]": 0.0,
    }
    param.update(updates, check_already_exists=False)
    return param


def classify_modes(current_a: np.ndarray) -> np.ndarray:
    return np.where(current_a < -1e-9, "CC-Chg", np.where(current_a > 1e-9, "CC-Dchg", "Rest"))


def step_indices(modes: np.ndarray) -> np.ndarray:
    out = np.ones_like(modes, dtype=int)
    step = 1
    for i in range(1, len(modes)):
        if modes[i] != modes[i - 1]:
            step += 1
        out[i] = step
    return out


def step_capacity_mah(time_s: np.ndarray, current_a: np.ndarray, steps: np.ndarray) -> np.ndarray:
    cap = np.zeros_like(time_s, dtype=float)
    for step in sorted(set(steps.tolist())):
        idx = np.where(steps == step)[0]
        if not len(idx):
            continue
        if abs(float(np.mean(current_a[idx]))) < 1e-12:
            cap[idx] = cap[idx[0] - 1] if idx[0] > 0 else 0.0
        else:
            t_step = time_s[idx] - time_s[idx][0]
            q = np.cumsum(np.abs(current_a[idx]) * np.gradient(t_step) / 3600.0) * 1000.0
            q[0] = 0.0
            cap[idx] = q
    return cap


def concentration_summary(sol) -> dict:
    out: dict[str, float] = {}
    variables = {
        "primary_xavg": "X-averaged positive primary particle concentration [mol.m-3]",
        "secondary_xavg": "X-averaged positive secondary particle concentration [mol.m-3]",
        "primary_avg": "Average positive primary particle concentration [mol.m-3]",
        "secondary_avg": "Average positive secondary particle concentration [mol.m-3]",
    }
    for key, var in variables.items():
        try:
            arr = np.asarray(sol[var].entries, dtype=float)
            out[f"{key}_min"] = float(np.nanmin(arr))
            out[f"{key}_max"] = float(np.nanmax(arr))
            out[f"{key}_final"] = float(np.ravel(arr)[-1])
        except Exception as exc:
            out[f"{key}_error"] = str(exc)
    return out


def simulate(args):
    gen = _load_effective_generator()
    truth, _, _ = gen.default_truth()
    truth = replace(truth, frac_R3m=args.frac_r3m)
    if args.phase_radius_m is not None:
        truth = replace(
            truth,
            log10_R_R3m=float(np.log10(args.phase_radius_m)),
            log10_R_C2m=float(np.log10(args.phase_radius_m)),
        )
    if args.d_r3m_m2_s is not None:
        truth = replace(truth, log10_D_R3m=float(np.log10(args.d_r3m_m2_s)))
    if args.d_c2m_m2_s is not None:
        truth = replace(truth, log10_D_C2m=float(np.log10(args.d_c2m_m2_s)))
    if args.ocp_shape == "gaussian":
        ocp_R3m, ocp_C2m = gaussian_redox_native_ocps(
            args.r3m_center_v,
            args.c2m_center_v,
            args.r3m_sigma_v,
            args.c2m_sigma_v,
            args.r3m_peak_weight,
            args.c2m_peak_weight,
        )
    else:
        ocp_R3m, ocp_C2m = target_peak_native_ocps()

    model = build_native_model()
    params = build_native_params(truth, ocp_R3m, ocp_C2m, args.initial_fraction)
    solver = pybamm.CasadiSolver(
        mode="safe",
        rtol=1e-5,
        atol=1e-7,
        extra_options_setup={"max_num_steps": 10000},
    )

    rows = []
    phase_summaries = []
    start_s = 0.0
    data_point = 1
    for cycle, c_rate in enumerate(args.c_rates, start=1):
        experiment = pybamm.Experiment(
            [
                f"Charge at {c_rate:g}C until {args.voltage_upper:g} V",
                f"Rest for {args.rest_minutes:g} minutes",
                f"Discharge at {c_rate:g}C until {args.voltage_lower:g} V",
                f"Rest for {args.rest_minutes:g} minutes",
            ],
            period=f"{args.period_s:g} seconds",
        )
        sim = pybamm.Simulation(model, parameter_values=params.copy(), experiment=experiment, solver=solver)
        sol = sim.solve()

        t_rel = np.asarray(sol["Time [s]"].entries, dtype=float)
        time_s = t_rel + start_s
        voltage_v = np.asarray(sol["Voltage [V]"].entries, dtype=float)
        current_a = np.asarray(sol["Current [A]"].entries, dtype=float)
        modes = classify_modes(current_a)
        steps = step_indices(modes)
        cap_mah = step_capacity_mah(time_s, current_a, steps)
        n = len(time_s)
        rows.append(
            pd.DataFrame(
                {
                    "Data_Point": np.arange(data_point, data_point + n),
                    "Cycle_No": cycle,
                    "Step_No": steps.astype(int),
                    "Mode": modes,
                    "Time(s)": np.round(time_s, 3),
                    "Voltage(V)": np.round(voltage_v, 6),
                    "Current(mA)": np.round(-current_a * 1000.0, 6),
                    "Capacity(mAh)": np.round(cap_mah, 6),
                    "C_rate": c_rate,
                }
            )
        )
        phase_summaries.append({"cycle": cycle, "c_rate": c_rate} | concentration_summary(sol))
        data_point += n
        start_s = float(time_s[-1] + args.inter_cycle_rest_s)

    return pd.concat(rows, ignore_index=True), phase_summaries, truth, ocp_R3m, ocp_C2m


def write_outputs(args, df, phase_summaries, truth, ocp_R3m, ocp_C2m):
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rate_suffix = "_".join(f"{rate:g}C".replace(".", "p") for rate in args.c_rates)
    csv_path = out_dir / f"Toyo_LMR_native2phase_PyBaMM_{rate_suffix}.csv"
    df.to_csv(csv_path, index=False)

    sto = np.linspace(0.01, 0.99, 600)
    u_R3m = ocp_R3m.numpy(sto)
    u_C2m = ocp_C2m.numpy(sto)
    pd.DataFrame(
        {
            "stoichiometry": sto,
            "U_R3m_V": u_R3m,
            "U_C2m_V": u_C2m,
            "U_fraction_weighted_V": truth.frac_R3m * u_R3m + truth.frac_C2m * u_C2m,
        }
    ).to_csv(out_dir / "native_phase_ocp_basis.csv", index=False)

    manifest = {
        "simulation_mode": "pybamm_native_positive_2phase",
        "phase_mapping": {"Primary": "R3m", "Secondary": "C2m"},
        "truth": asdict(truth) | {"frac_C2m": truth.frac_C2m},
        "native_phase_parameters": {
            "Primary_R3m": {
                "D_m2_s": truth.D_R3m,
                "R_m": truth.R_R3m,
                "active_material_fraction_in_electrode": 0.665 * truth.frac_R3m,
            },
            "Secondary_C2m": {
                "D_m2_s": truth.D_C2m,
                "R_m": truth.R_C2m,
                "active_material_fraction_in_electrode": 0.665 * truth.frac_C2m,
            },
        },
        "protocol": {
            "voltage_lower_v": args.voltage_lower,
            "voltage_upper_v": args.voltage_upper,
            "rest_minutes": args.rest_minutes,
            "period_s": args.period_s,
            "initial_fraction": args.initial_fraction,
            "c_rates": args.c_rates,
        },
        "ocp_shape": args.ocp_shape,
        "ocp_note": (
            "Native phase OCPs are monotonic interpolation curves. In gaussian mode, "
            "R3m and C2m are generated by integrating Gaussian redox distributions "
            "on the dQ/dV voltage axis."
        ),
        "target_peaks": {
            "R3m_primary_redox_feature_v": args.r3m_center_v,
            "C2m_secondary_redox_feature_v": args.c2m_center_v,
            "R3m_sigma_v": args.r3m_sigma_v,
            "C2m_sigma_v": args.c2m_sigma_v,
            "R3m_peak_weight": args.r3m_peak_weight,
            "C2m_peak_weight": args.c2m_peak_weight,
        },
    }
    (out_dir / "true_native_2phase_parameters.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (out_dir / "phase_concentration_summary.json").write_text(
        json.dumps(phase_summaries, indent=2), encoding="utf-8"
    )

    profile = ToyoAsciiParser(ROOT / "configs/toyo_ascii.yaml").parse(csv_path)
    roundtrip = {
        "parsed_points": int(len(profile)),
        "cycles": profile.get_cycles(),
        "segments": len(profile.segments),
        "voltage_min_v": float(np.min(profile.voltage_v)),
        "voltage_max_v": float(np.max(profile.voltage_v)),
        "pybamm_current_min_a_after_parse": float(np.min(profile.current_a)),
        "pybamm_current_max_a_after_parse": float(np.max(profile.current_a)),
    }
    (out_dir / "roundtrip_check.json").write_text(json.dumps(roundtrip, indent=2), encoding="utf-8")
    return csv_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw/toyo/native_2phase_sample")
    parser.add_argument("--c-rates", type=float, nargs="+", default=[0.1, 0.33, 0.5, 1.0])
    parser.add_argument("--voltage-lower", type=float, default=2.5)
    parser.add_argument("--voltage-upper", type=float, default=4.65)
    parser.add_argument("--rest-minutes", type=float, default=10.0)
    parser.add_argument("--period-s", type=float, default=0.5)
    parser.add_argument("--initial-fraction", type=float, default=0.98)
    parser.add_argument("--frac-r3m", type=float, default=0.60)
    parser.add_argument("--phase-radius-m", type=float, default=None)
    parser.add_argument("--d-r3m-m2-s", type=float, default=None)
    parser.add_argument("--d-c2m-m2-s", type=float, default=None)
    parser.add_argument("--ocp-shape", choices=["plateau", "gaussian"], default="plateau")
    parser.add_argument("--r3m-center-v", type=float, default=3.18)
    parser.add_argument("--c2m-center-v", type=float, default=3.82)
    parser.add_argument("--r3m-sigma-v", type=float, default=0.075)
    parser.add_argument("--c2m-sigma-v", type=float, default=0.080)
    parser.add_argument("--r3m-peak-weight", type=float, default=1.0)
    parser.add_argument("--c2m-peak-weight", type=float, default=1.0)
    parser.add_argument("--inter-cycle-rest-s", type=float, default=600.0)
    return parser.parse_args()


def main():
    args = parse_args()
    df, phase_summaries, truth, ocp_R3m, ocp_C2m = simulate(args)
    csv_path = write_outputs(args, df, phase_summaries, truth, ocp_R3m, ocp_C2m)
    print(f"wrote {csv_path}")
    print(f"rows {len(df)}")
    print(f"voltage range {df['Voltage(V)'].min():.5f}..{df['Voltage(V)'].max():.5f} V")
    print(f"output dir {ROOT / args.out_dir}")


if __name__ == "__main__":
    main()
