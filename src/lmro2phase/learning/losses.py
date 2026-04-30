"""Stage 3 학습 손실함수."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ocp_grid_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """OCP grid MSE."""
    return F.mse_loss(pred, target)


def ocp_derivative_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """OCP 1차 미분 MSE (단조성 보존 보조 손실)."""
    dp = torch.diff(pred, dim=-1)
    dt = torch.diff(target, dim=-1)
    return F.mse_loss(dp, dt)


def smoothness_penalty(ocp: torch.Tensor) -> torch.Tensor:
    """OCP 2차 미분 L2 페널티."""
    d2 = torch.diff(ocp, n=2, dim=-1)
    return (d2 ** 2).mean()


def voltage_range_penalty(ocp: torch.Tensor,
                            v_min: float = 1.5, v_max: float = 5.2) -> torch.Tensor:
    """OCP가 물리적 전압 범위를 벗어날 때 페널티."""
    below = F.relu(v_min - ocp)
    above = F.relu(ocp - v_max)
    return (below ** 2 + above ** 2).mean()


def phase_fraction_sum_penalty(scalar_pred: torch.Tensor,
                                frac_R3m_idx: int = 2,
                                frac_C2m_idx: int = 8) -> torch.Tensor:
    """phase fraction 합이 1이 되도록 페널티."""
    f_sum = scalar_pred[:, frac_R3m_idx] + scalar_pred[:, frac_C2m_idx]
    return ((f_sum - 1.0) ** 2).mean()


def permutation_invariant_ocp_loss(pred_R3m: torch.Tensor, pred_C2m: torch.Tensor,
                                    true_R3m: torch.Tensor, true_C2m: torch.Tensor
                                    ) -> torch.Tensor:
    """
    phase 교환 대칭 loss:
    min(direct_loss, swapped_loss) per sample
    """
    direct = F.mse_loss(pred_R3m, true_R3m, reduction="none").mean(-1) + \
             F.mse_loss(pred_C2m, true_C2m, reduction="none").mean(-1)
    swapped = F.mse_loss(pred_R3m, true_C2m, reduction="none").mean(-1) + \
              F.mse_loss(pred_C2m, true_R3m, reduction="none").mean(-1)
    return torch.minimum(direct, swapped).mean()


def total_loss(scalar_pred, ocp_R3m_pred, ocp_C2m_pred,
               scalar_true, ocp_R3m_true, ocp_C2m_true,
               weights: dict,
               use_permutation: bool = True) -> torch.Tensor:
    """전체 loss 계산."""
    l_scalar = F.mse_loss(scalar_pred, scalar_true)

    if use_permutation:
        l_ocp = permutation_invariant_ocp_loss(
            ocp_R3m_pred, ocp_C2m_pred, ocp_R3m_true, ocp_C2m_true)
    else:
        l_ocp = ocp_grid_loss(ocp_R3m_pred, ocp_R3m_true) + \
                ocp_grid_loss(ocp_C2m_pred, ocp_C2m_true)

    l_deriv = (ocp_derivative_loss(ocp_R3m_pred, ocp_R3m_true) +
               ocp_derivative_loss(ocp_C2m_pred, ocp_C2m_true))

    l_smooth = (smoothness_penalty(ocp_R3m_pred) + smoothness_penalty(ocp_C2m_pred))

    l_range = (voltage_range_penalty(ocp_R3m_pred) + voltage_range_penalty(ocp_C2m_pred))

    l_frac = phase_fraction_sum_penalty(scalar_pred)

    return (
        weights.get("scalar_mse", 1.0) * l_scalar
        + weights.get("ocp_grid_mse", 2.0) * l_ocp
        + weights.get("ocp_deriv_mse", 1.0) * l_deriv
        + weights.get("smoothness_penalty", 0.5) * l_smooth
        + weights.get("voltage_range_penalty", 5.0) * l_range
        + weights.get("phase_fraction_sum", 10.0) * l_frac
    )
