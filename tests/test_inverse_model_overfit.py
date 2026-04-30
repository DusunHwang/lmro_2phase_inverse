"""DL inverse model 100-sample overfit 테스트 (PyTorch 필요)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")


def make_dummy_dataset(n: int = 100, n_grid: int = 512, ocp_len: int = 256):
    """더미 데이터셋 생성 (실제 zarr 없이)."""
    from torch.utils.data import TensorDataset

    X       = torch.randn(n, 10, n_grid)
    y_scalar= torch.randn(n, 9)
    y_ocp   = torch.randn(n, 2, ocp_len)

    return TensorDataset(X, y_scalar, y_ocp)


def test_inverse_model_forward_pass():
    """모델 forward pass 형태 확인."""
    from lmro2phase.learning.model_inverse import InverseModel

    model = InverseModel(
        in_channels=10, base_channels=32, n_blocks=2, kernel_size=7,
        dropout=0.0, use_transformer=False, n_scalar=9, ocp_grid_len=256,
    )
    model.eval()
    x = torch.randn(4, 10, 512)
    with torch.no_grad():
        scalar, ocp_R3m, ocp_C2m = model(x)

    assert scalar.shape == (4, 9)
    assert ocp_R3m.shape == (4, 256)
    assert ocp_C2m.shape == (4, 256)


def test_overfit_100_samples():
    """100개 샘플에 과적합되는지 확인 (loss 충분히 감소)."""
    from lmro2phase.learning.model_inverse import InverseModel
    from lmro2phase.learning.losses import total_loss
    import torch.optim as optim

    model = InverseModel(
        in_channels=10, base_channels=32, n_blocks=2, kernel_size=7,
        dropout=0.0, use_transformer=False, n_scalar=9, ocp_grid_len=256,
    )

    dataset = make_dummy_dataset(n=100, n_grid=512, ocp_len=256)
    X, y_scalar, y_ocp = dataset.tensors

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    weights = {"scalar_mse": 1.0, "ocp_grid_mse": 1.0,
               "ocp_deriv_mse": 0.1, "smoothness_penalty": 0.0,
               "voltage_range_penalty": 0.0, "phase_fraction_sum": 0.0}

    initial_loss = None
    for step in range(200):
        model.train()
        optimizer.zero_grad()
        scalar_pred, ocp_R3m_pred, ocp_C2m_pred = model(X)
        loss = total_loss(
            scalar_pred, ocp_R3m_pred, ocp_C2m_pred,
            y_scalar, y_ocp[:, 0], y_ocp[:, 1],
            weights, use_permutation=False,
        )
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()

    final_loss = loss.item()
    # 200 스텝 후 loss가 초기 대비 50% 이상 감소해야 함
    assert final_loss < initial_loss * 0.5, \
        f"Overfit 실패: initial={initial_loss:.4f}, final={final_loss:.4f}"
