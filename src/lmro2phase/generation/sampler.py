"""Stage 2: 파라미터 샘플러.

Stage 1 best fit을 중심으로 R3m/C2m 파라미터를 샘플링합니다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@dataclass
class SampleRecord:
    """단일 샘플 파라미터."""
    sample_id: int
    log10_D_R3m: float
    log10_R_R3m: float
    frac_R3m: float
    log10_D_C2m: float
    log10_R_C2m: float
    log10_contact_resistance: float
    capacity_scale: float
    initial_stoichiometry_shift: float
    ocp_mode: str = "perturbed"  # base | perturbed

    @property
    def frac_C2m(self) -> float:
        return 1.0 - self.frac_R3m


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(np.clip(v, lo, hi))


def sample_parameters(n_samples: int,
                       center: dict,
                       cfg: DictConfig,
                       rng: np.random.Generator,
                       strategy: str = "local") -> list[SampleRecord]:
    """
    파라미터 샘플 목록 생성.

    strategy:
        local   - center 주변 Gaussian 샘플
        broad   - prior uniform 샘플
        edge    - 경계 근방 샘플
    """
    pb = OmegaConf.to_container(cfg.param_bounds, resolve=True)
    records = []

    for i in range(n_samples):
        if strategy == "local":
            rec = _sample_local(i, center, pb, rng)
        elif strategy == "broad":
            rec = _sample_broad(i, pb, rng)
        else:  # edge
            rec = _sample_edge(i, center, pb, rng)
        records.append(rec)

    return records


def _sample_local(sid: int, center: dict, pb: dict, rng) -> SampleRecord:
    def g(key, sigma, lo, hi):
        return _clamp(rng.normal(center.get(key, (lo + hi) / 2), sigma), lo, hi)

    r3m_b = pb["rhombohedral_R3m"]
    c2m_b = pb["monoclinic_C2m"]
    sh_b  = pb["shared"]

    frac_R3m = _clamp(rng.normal(center.get("frac_R3m", 0.6),
                                   r3m_b["active_fraction"]["sigma"]),
                       *r3m_b["active_fraction"]["bounds"])

    return SampleRecord(
        sample_id=sid,
        log10_D_R3m=g("log10_D_R3m", r3m_b["log10_D_s"]["sigma"], *r3m_b["log10_D_s"]["bounds"]),
        log10_R_R3m=g("log10_R_R3m", r3m_b["log10_R_particle"]["sigma"], *r3m_b["log10_R_particle"]["bounds"]),
        frac_R3m=frac_R3m,
        log10_D_C2m=g("log10_D_C2m", c2m_b["log10_D_s"]["sigma"], *c2m_b["log10_D_s"]["bounds"]),
        log10_R_C2m=g("log10_R_C2m", c2m_b["log10_R_particle"]["sigma"], *c2m_b["log10_R_particle"]["bounds"]),
        log10_contact_resistance=_clamp(
            rng.uniform(*sh_b["log10_contact_resistance"]["bounds"]),
            *sh_b["log10_contact_resistance"]["bounds"]),
        capacity_scale=_clamp(rng.normal(1.0, 0.1), *sh_b["capacity_scale"]["bounds"]),
        initial_stoichiometry_shift=_clamp(
            rng.normal(0.0, 0.02), *sh_b["initial_stoichiometry_shift"]["bounds"]),
    )


def _sample_broad(sid: int, pb: dict, rng) -> SampleRecord:
    def u(lo, hi):
        return float(rng.uniform(lo, hi))

    r3m_b = pb["rhombohedral_R3m"]
    c2m_b = pb["monoclinic_C2m"]
    sh_b  = pb["shared"]
    frac = u(0.05, 0.95)

    return SampleRecord(
        sample_id=sid,
        log10_D_R3m=u(*r3m_b["log10_D_s"]["bounds"]),
        log10_R_R3m=u(*r3m_b["log10_R_particle"]["bounds"]),
        frac_R3m=frac,
        log10_D_C2m=u(*c2m_b["log10_D_s"]["bounds"]),
        log10_R_C2m=u(*c2m_b["log10_R_particle"]["bounds"]),
        log10_contact_resistance=u(*sh_b["log10_contact_resistance"]["bounds"]),
        capacity_scale=u(*sh_b["capacity_scale"]["bounds"]),
        initial_stoichiometry_shift=u(*sh_b["initial_stoichiometry_shift"]["bounds"]),
    )


def _sample_edge(sid: int, center: dict, pb: dict, rng) -> SampleRecord:
    """경계 근방 (edge case) 샘플."""
    rec = _sample_broad(sid, pb, rng)
    # 일부 파라미터를 경계로 강제
    if rng.random() < 0.5:
        rec.frac_R3m = float(rng.choice([0.08, 0.92]))
    return rec


def records_to_dataframe(records: list[SampleRecord]):
    import pandas as pd
    return pd.DataFrame([vars(r) for r in records])
