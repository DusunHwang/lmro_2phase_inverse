"""Validation 플롯 생성 유틸리티."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_forward_validation(val_result: dict, out_path: str | Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        t = np.array(val_result.get("time_s", []))
        v_sim = np.array(val_result.get("voltage_sim", []))
        v_exp = np.array(val_result.get("voltage_exp", []))
        res   = np.array(val_result.get("residual", []))

        ax = axes[0]
        ax.plot(t / 3600, v_exp, "k-", label="Measured", linewidth=1.0)
        ax.plot(t / 3600, v_sim, "r--", label="Simulated", linewidth=1.0)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title("Forward Validation")
        ax.legend()
        ax.grid(True, alpha=0.3)

        rmse = val_result.get("rmse_v", 0)
        ax.annotate(f"RMSE={rmse*1000:.1f} mV", xy=(0.02, 0.05),
                    xycoords="axes fraction", fontsize=10)

        ax2 = axes[1]
        ax2.plot(t / 3600, res * 1000, "b-", linewidth=0.8)
        ax2.axhline(0, color="k", linewidth=0.5)
        ax2.set_xlabel("Time (h)")
        ax2.set_ylabel("Residual (mV)")
        ax2.set_title("Voltage Residual")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"플롯 실패: {e}")
