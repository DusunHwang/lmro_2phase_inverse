"""Stage 4: 학습된 모델로 추론."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from ..physics.ocp_grid import OCPGrid, STO_GRID

log = logging.getLogger(__name__)

SCALAR_KEYS = [
    "log10_D_R3m", "log10_R_R3m", "frac_R3m",
    "log10_D_C2m", "log10_R_C2m",
    "log10_contact_resistance", "capacity_scale",
    "initial_stoichiometry_shift",
    "frac_C2m",
]


def load_model(checkpoint_path: str | Path, model_cfg: dict, device: str = "auto"):
    from .model_inverse import InverseModel

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model = InverseModel(**model_cfg).to(dev)
    state = torch.load(str(checkpoint_path), map_location=dev, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    log.info(f"모델 로드: {checkpoint_path}, device={dev}")
    return model, dev


def infer_profile(model, feature_tensor: np.ndarray, device) -> dict:
    """
    feature_tensor: [10, 512] numpy array
    Returns: dict with scalar params, OCP grids
    """
    x = torch.from_numpy(feature_tensor[None]).float().to(device)  # [1, 10, 512]

    with torch.no_grad():
        scalar, ocp_R3m, ocp_C2m = model(x)

    scalar_np = scalar.cpu().numpy()[0]
    ocp_R3m_np = ocp_R3m.cpu().numpy()[0]
    ocp_C2m_np = ocp_C2m.cpu().numpy()[0]

    result = {k: float(v) for k, v in zip(SCALAR_KEYS, scalar_np)}
    result["ocp_R3m_grid"] = ocp_R3m_np.tolist()
    result["ocp_C2m_grid"] = ocp_C2m_np.tolist()
    return result


def save_inference_results(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # scalar params JSON
    scalar_dict = {k: v for k, v in result.items() if not k.startswith("ocp_")}
    with open(out_dir / "predicted_params.json", "w") as f:
        json.dump(scalar_dict, f, indent=2)

    # OCP CSVs
    import pandas as pd
    for phase in ["R3m", "C2m"]:
        key = f"ocp_{phase}_grid"
        if key in result:
            ocp_v = np.array(result[key])
            sto = np.linspace(0.005, 0.995, len(ocp_v))
            pd.DataFrame({"stoichiometry": sto, "voltage_v": ocp_v}).to_csv(
                out_dir / f"predicted_ocp_{phase}.csv", index=False
            )

    log.info(f"추론 결과 저장: {out_dir}")
