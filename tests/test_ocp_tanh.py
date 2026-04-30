"""tanh OCP 함수 단위 테스트."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_tanh_ocp_numpy_evaluation():
    from lmro2phase.physics.ocp_tanh import TanhOCPParams, make_tanh_ocp_numpy

    params = TanhOCPParams.default_init(n_terms=3)
    fn = make_tanh_ocp_numpy(params)

    sto = np.linspace(0.01, 0.99, 100)
    v = fn(sto)

    assert v.shape == (100,)
    assert v.min() >= params.voltage_lower - 1e-6
    assert v.max() <= params.voltage_upper + 1e-6


def test_tanh_ocp_param_roundtrip():
    from lmro2phase.physics.ocp_tanh import TanhOCPParams

    params = TanhOCPParams.default_init(n_terms=5)
    vec = params.to_vector()
    params2 = TanhOCPParams.from_vector(vec, n_terms=5)

    np.testing.assert_allclose(params.b0, params2.b0, rtol=1e-6)
    np.testing.assert_allclose(params.amps, params2.amps, rtol=1e-6)


def test_tanh_ocp_pybamm_expression():
    pytest.importorskip("pybamm")
    from lmro2phase.physics.ocp_tanh import TanhOCPParams, make_tanh_ocp_pybamm
    import pybamm

    params = TanhOCPParams.default_init(n_terms=3)
    fn = make_tanh_ocp_pybamm(params)

    sto = pybamm.Scalar(0.5)
    result = fn(sto)
    assert hasattr(result, "evaluate") or hasattr(result, "__add__")


def test_smoothness_penalty():
    from lmro2phase.physics.ocp_tanh import TanhOCPParams, tanh_ocp_second_derivative_penalty

    # 매우 좁은 tanh → 높은 페널티
    p_sharp = TanhOCPParams(b0=3.5, b1=0.0,
                             amps=np.array([0.5]),
                             centers=np.array([0.5]),
                             widths=np.array([0.005]))
    # 넓은 tanh → 낮은 페널티
    p_smooth = TanhOCPParams(b0=3.5, b1=0.0,
                              amps=np.array([0.5]),
                              centers=np.array([0.5]),
                              widths=np.array([0.3]))

    pen_sharp  = tanh_ocp_second_derivative_penalty(p_sharp)
    pen_smooth = tanh_ocp_second_derivative_penalty(p_smooth)
    assert pen_sharp > pen_smooth
