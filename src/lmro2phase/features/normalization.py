"""Feature 정규화 통계 계산 및 적용."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FeatureStats:
    mean: np.ndarray   # [channels]
    std: np.ndarray    # [channels]

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """x: [channels, L] or [B, channels, L]"""
        m = self.mean[:, None] if x.ndim == 2 else self.mean[None, :, None]
        s = self.std[:, None]  if x.ndim == 2 else self.std[None, :, None]
        return (x - m) / (s + 1e-8)

    def save(self, path: str | Path) -> None:
        np.savez(str(path), mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureStats":
        d = np.load(str(path))
        return cls(mean=d["mean"], std=d["std"])

    @classmethod
    def from_dataset(cls, tensors: np.ndarray) -> "FeatureStats":
        """tensors: [N, channels, L]"""
        mean = tensors.mean(axis=(0, 2))
        std  = tensors.std(axis=(0, 2))
        return cls(mean=mean, std=std)
