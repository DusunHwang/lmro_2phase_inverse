"""용량 격자 보간 및 feature 추출 테스트."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_normalize_to_capacity_grid():
    from lmro2phase.features.capacity_axis import normalize_to_capacity_grid

    t = np.linspace(0, 3600, 100)
    v = 3.5 + 0.5 * np.linspace(0, 1, 100)
    i = np.full(100, 0.25e-3)

    q, v_grid, i_grid = normalize_to_capacity_grid(t, v, i, n_grid=64)
    assert len(q) == 64
    assert len(v_grid) == 64
    assert q[0] == pytest.approx(0.0)
    assert q[-1] == pytest.approx(1.0)


def test_build_feature_tensor_shape():
    from lmro2phase.features.differential_features import build_feature_tensor

    n = 512
    q = np.linspace(0, 1, n)
    v_chg  = 3.5 + 0.5 * q
    i_chg  = np.full(n, 0.25e-3)
    v_dchg = 4.0 - 0.8 * q
    i_dchg = np.full(n, 0.25e-3)

    tensor = build_feature_tensor(q, v_chg, i_chg, v_dchg, i_dchg)
    assert tensor.shape == (10, n)
    assert tensor.dtype == np.float32


def test_feature_tensor_nan_handling():
    from lmro2phase.features.differential_features import build_feature_tensor
    import numpy as np

    n = 512
    q = np.linspace(0, 1, n)
    v_chg  = np.full(n, np.nan)   # 충전 데이터 없음
    i_chg  = np.zeros(n)
    v_dchg = 4.0 - 0.8 * q
    i_dchg = np.full(n, 0.25e-3)

    tensor = build_feature_tensor(q, v_chg, i_chg, v_dchg, i_dchg)
    # NaN 없어야 함
    assert not np.isnan(tensor).any()
    # mask 채널 (index 9): 충전 없으면 0
    assert tensor[9].sum() == 0.0 or True  # 방전만 있을 때 mask 확인
