"""PyTorch Dataset for synthetic simulation data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..features.capacity_axis import normalize_to_capacity_grid, FEATURE_GRID
from ..features.differential_features import build_feature_tensor
from ..physics.ocp_grid import GRID_LENGTH


class SyntheticDataset(Dataset):
    """
    Synthetic 시뮬레이션 데이터셋.

    X: [channels=10, grid_length=512] feature tensor
    y_scalar: [9] scalar parameter vector
    y_ocp: [2, 256] OCP grid (R3m, C2m)
    """

    SCALAR_KEYS = [
        "log10_D_R3m", "log10_R_R3m", "frac_R3m",
        "log10_D_C2m", "log10_R_C2m",
        "log10_contact_resistance", "capacity_scale",
        "initial_stoichiometry_shift",
        "frac_C2m",
    ]

    def __init__(self,
                  params_path: str | Path,
                  profiles_path: str | Path,
                  ocp_path: str | Path,
                  n_grid: int = FEATURE_GRID,
                  ocp_grid_len: int = GRID_LENGTH,
                  feature_stats=None):
        self.params_df = pd.read_parquet(str(params_path))
        self.sample_ids = self.params_df["sample_id"].tolist()
        self.n_grid = n_grid
        self.ocp_grid_len = ocp_grid_len
        self.feature_stats = feature_stats

        import zarr
        self.profiles_store = zarr.open(str(profiles_path), mode="r")
        self.ocp_store = zarr.open(str(ocp_path), mode="r")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sid = str(self.sample_ids[idx])
        row = self.params_df.iloc[idx]

        # --- 프로파일 로드 ---
        grp = self.profiles_store[sid]
        t = grp["time_s"][:]
        v = grp["voltage_v"][:]
        i = grp["current_a"][:]

        # charge / discharge 분리 (전류 부호로)
        chg_mask  = i > 0
        dchg_mask = i < 0

        # 각각 정규화 격자로 변환
        if chg_mask.sum() > 5:
            _, v_chg, i_chg = normalize_to_capacity_grid(
                t[chg_mask], v[chg_mask], i[chg_mask], self.n_grid)
        else:
            v_chg = np.full(self.n_grid, np.nan, dtype=np.float32)
            i_chg = np.zeros(self.n_grid, dtype=np.float32)

        if dchg_mask.sum() > 5:
            _, v_dchg, i_dchg = normalize_to_capacity_grid(
                t[dchg_mask], v[dchg_mask], np.abs(i[dchg_mask]), self.n_grid)
        else:
            v_dchg = np.full(self.n_grid, np.nan, dtype=np.float32)
            i_dchg = np.zeros(self.n_grid, dtype=np.float32)

        q_grid = np.linspace(0, 1, self.n_grid)
        X = build_feature_tensor(q_grid, v_chg, i_chg, v_dchg, i_dchg)  # [10, L]

        if self.feature_stats is not None:
            X = self.feature_stats.normalize(X)

        # --- OCP 로드 ---
        ocp_grp = self.ocp_store[sid]
        ocp_R3m = ocp_grp["ocp_R3m"][:self.ocp_grid_len].astype(np.float32)
        ocp_C2m = ocp_grp["ocp_C2m"][:self.ocp_grid_len].astype(np.float32)
        y_ocp = np.stack([ocp_R3m, ocp_C2m], axis=0)  # [2, 256]

        # --- scalar labels ---
        y_scalar = np.array([float(row.get(k, 0.0)) for k in self.SCALAR_KEYS],
                             dtype=np.float32)

        return (
            torch.from_numpy(X),
            torch.from_numpy(y_scalar),
            torch.from_numpy(y_ocp),
        )
