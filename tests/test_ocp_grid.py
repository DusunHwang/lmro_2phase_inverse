"""OCP grid 단위 테스트."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_ocp_grid_from_tanh():
    from lmro2phase.physics.ocp_tanh import TanhOCPParams
    from lmro2phase.physics.ocp_grid import OCPGrid, GRID_LENGTH

    params = TanhOCPParams.default_init(n_terms=5)
    grid = OCPGrid.from_tanh_params(params)

    assert len(grid.sto) == GRID_LENGTH
    assert len(grid.voltage) == GRID_LENGTH
    assert grid.voltage.min() >= grid.voltage_lower - 1e-6
    assert grid.voltage.max() <= grid.voltage_upper + 1e-6


def test_ocp_grid_smooth():
    from lmro2phase.physics.ocp_tanh import TanhOCPParams
    from lmro2phase.physics.ocp_grid import OCPGrid

    params = TanhOCPParams.default_init(n_terms=5)
    grid = OCPGrid.from_tanh_params(params)
    smoothed = grid.smooth(sigma=3.0)

    # 스무딩 후 2차 미분 페널티 감소
    assert smoothed.second_derivative_penalty() <= grid.second_derivative_penalty() + 1e-8


def test_ocp_grid_csv_roundtrip():
    import tempfile, os
    from lmro2phase.physics.ocp_tanh import TanhOCPParams
    from lmro2phase.physics.ocp_grid import OCPGrid

    params = TanhOCPParams.default_init(n_terms=3)
    grid = OCPGrid.from_tanh_params(params)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp = f.name

    try:
        grid.to_csv(tmp)
        grid2 = OCPGrid.from_csv(tmp)
        np.testing.assert_allclose(grid.voltage, grid2.voltage, rtol=1e-5)
    finally:
        os.unlink(tmp)


def test_ocp_perturbation():
    from lmro2phase.physics.ocp_tanh import TanhOCPParams
    from lmro2phase.physics.ocp_grid import OCPGrid
    from lmro2phase.physics.ocp_perturbation import perturb_ocp_grid

    params = TanhOCPParams.default_init(n_terms=5)
    base = OCPGrid.from_tanh_params(params)
    rng = np.random.default_rng(42)

    for mode in ["gp", "spline", "plateau", "transition", "shoulder", "tanh"]:
        perturbed = perturb_ocp_grid(base, rng, mode=mode)
        assert perturbed.voltage.min() >= base.voltage_lower - 1e-6
        assert perturbed.voltage.max() <= base.voltage_upper + 1e-6
        # 변동이 있어야 함
        assert not np.allclose(base.voltage, perturbed.voltage)
