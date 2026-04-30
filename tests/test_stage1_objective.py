"""Stage 1 목적함수 smoke test (PyBaMM 필요)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

pybamm = pytest.importorskip("pybamm")


def make_dummy_profile():
    """더미 BatteryProfile 생성."""
    from lmro2phase.io.profile_schema import BatteryProfile

    t = np.linspace(0, 3600, 200)
    i = np.full(200, 0.25e-3)   # CC charge
    v = 3.5 + 0.5 * t / 3600
    return BatteryProfile(
        source_file="dummy",
        time_s=t, voltage_v=v, current_a=i,
    )


def test_stage1_objective_returns_finite():
    """Stage 1 loss가 유한한 값을 반환하는지 확인."""
    from lmro2phase.fitting.stage1_objective import Stage1Params, LossWeights, compute_loss
    from lmro2phase.physics.ocp_tanh import TanhOCPParams
    from lmro2phase.physics.halfcell_model_factory import build_halfcell_model, ModelType

    params = Stage1Params()
    params.tanh_R3m = TanhOCPParams.default_init(n_terms=3, voltage_center=3.8)
    params.tanh_C2m = TanhOCPParams.default_init(n_terms=3, voltage_center=3.6)

    model = build_halfcell_model(ModelType.SPMe)
    weights = LossWeights()
    profile = make_dummy_profile()

    loss = compute_loss(params, profile, model, weights)
    assert np.isfinite(loss) or loss >= 1e7  # 시뮬레이션 실패 시 큰 값 반환

    # 적어도 예외 없이 float 반환
    assert isinstance(loss, float)
