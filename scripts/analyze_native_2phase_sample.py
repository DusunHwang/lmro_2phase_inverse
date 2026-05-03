#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/raw/toyo/native_2phase_sample"
CSV_PATH = OUT_DIR / "Toyo_LMR_native2phase_PyBaMM_0p1C_0p33C_0p5C_1C.csv"


def dqdv_for_step(step: pd.DataFrame, sign: float, grid_v: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    step = step.sort_values("Time(s)")
    voltage = step["Voltage(V)"].to_numpy(dtype=float)
    capacity = step["Capacity(mAh)"].to_numpy(dtype=float) / 1000.0
    ok = np.isfinite(voltage) & np.isfinite(capacity)
    voltage = voltage[ok]
    capacity = capacity[ok]
    if len(voltage) < 3:
        return np.array([]), np.array([])

    order = np.argsort(voltage)
    v_sorted = voltage[order]
    q_sorted = capacity[order]

    # Collapse repeated voltage samples before interpolating Q(V). Median is
    # robust to dense low-rate sampling and small solver jitter near plateaus.
    q_by_v = (
        pd.DataFrame({"v": v_sorted, "q": q_sorted})
        .groupby("v", sort=True, as_index=False)["q"]
        .median()
    )
    v_unique = q_by_v["v"].to_numpy(dtype=float)
    q_unique = q_by_v["q"].to_numpy(dtype=float)
    if len(v_unique) < 3 or (v_unique[-1] - v_unique[0]) < grid_v:
        return np.array([]), np.array([])

    v_grid = np.arange(v_unique[0], v_unique[-1] + 0.5 * grid_v, grid_v)
    if len(v_grid) < 3:
        return np.array([]), np.array([])
    q_grid = np.interp(v_grid, v_unique, q_unique)
    dqdv = np.gradient(q_grid, v_grid)
    return v_grid, sign * np.abs(dqdv)


def peak_in_window(v: np.ndarray, y: np.ndarray, lo: float, hi: float) -> dict[str, float | None]:
    mask = (v >= lo) & (v <= hi) & np.isfinite(y)
    if not np.any(mask):
        return {"voltage_v": None, "dqdv_ah_per_v": None}
    idx = np.argmax(np.abs(y[mask]))
    return {
        "voltage_v": float(v[mask][idx]),
        "dqdv_ah_per_v": float(y[mask][idx]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR.relative_to(ROOT)))
    parser.add_argument("--csv", default=CSV_PATH.name)
    parser.add_argument("--grid-v", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    csv_path = out_dir / args.csv
    df = pd.read_csv(csv_path)

    summary: dict[str, object] = {}
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, df["C_rate"].nunique()))

    for color, (c_rate, group) in zip(colors, df.groupby("C_rate", sort=True)):
        rate_key = f"{c_rate:g}C"
        summary[rate_key] = {}
        for mode, sign, linestyle in [("CC-Chg", 1.0, "-"), ("CC-Dchg", -1.0, "--")]:
            step = group[group["Mode"] == mode]
            if len(step) < 3:
                summary[rate_key][mode] = {
                    "available": False,
                    "points": int(len(step)),
                }
                continue
            v, y = dqdv_for_step(step, sign, args.grid_v)
            if len(y) == 0:
                summary[rate_key][mode] = {
                    "available": False,
                    "points": int(len(step)),
                }
                continue
            clip = np.nanpercentile(np.abs(y), 99.5)
            y_plot = np.clip(y, -clip, clip)
            ax.plot(v, y_plot, color=color, linestyle=linestyle, linewidth=1.2, label=f"{rate_key} {mode}")
            summary[rate_key][mode] = {
                "available": True,
                "points": int(len(step)),
                "dqdv_method": "Q(V) interpolation followed by finite difference",
                "grid_v": args.grid_v,
                "voltage_min_v": float(step["Voltage(V)"].min()),
                "voltage_max_v": float(step["Voltage(V)"].max()),
                "r3m_window_2p5_3p35": peak_in_window(v, y, 2.5, 3.35),
                "c2m_window_3p45_4p15": peak_in_window(v, y, 3.45, 4.15),
            }

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvspan(3.15, 3.35, color="#4477aa", alpha=0.08, linewidth=0)
    ax.axvspan(3.70, 3.90, color="#cc6677", alpha=0.08, linewidth=0)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("dQ/dV (Ah/V)")
    ax.set_title("Native 2-phase PyBaMM dQ/dV overlay")
    ax.set_xlim(2.5, 4.65)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "native_2phase_dqdv_overlay_by_crate.png")
    plt.close(fig)

    protocol = (
        df.groupby(["C_rate", "Mode"])
        .agg(
            rows=("Data_Point", "count"),
            voltage_min_v=("Voltage(V)", "min"),
            voltage_max_v=("Voltage(V)", "max"),
            time_min_s=("Time(s)", "min"),
            time_max_s=("Time(s)", "max"),
        )
        .reset_index()
    )
    protocol.to_json(
        out_dir / "native_2phase_protocol_step_summary.json",
        orient="records",
        indent=2,
    )
    (out_dir / "native_2phase_dqdv_overlay_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    monotonicity = {}
    for c_rate, group in df.groupby("C_rate", sort=True):
        monotonicity[f"{c_rate:g}C"] = {}
        for mode, expected in [("CC-Chg", 1), ("CC-Dchg", -1)]:
            step = group[group["Mode"] == mode].sort_values("Time(s)")
            dv = np.diff(step["Voltage(V)"].to_numpy(dtype=float))
            if len(dv) == 0:
                monotonicity[f"{c_rate:g}C"][mode] = {"available": False}
            else:
                monotonicity[f"{c_rate:g}C"][mode] = {
                    "available": True,
                    "wrong_sign_fraction": float(np.mean(np.sign(dv[np.abs(dv) > 1e-8]) != expected))
                    if np.any(np.abs(dv) > 1e-8)
                    else 0.0,
                }
    (out_dir / "native_2phase_monotonicity_check.json").write_text(
        json.dumps(monotonicity, indent=2), encoding="utf-8"
    )

    print(f"wrote {out_dir / 'native_2phase_dqdv_overlay_by_crate.png'}")
    print(f"wrote {out_dir / 'native_2phase_dqdv_overlay_summary.json'}")


if __name__ == "__main__":
    main()
