"""Stage 3: DL inverse model.

구조:
  1D CNN residual encoder
  → optional Transformer encoder
  → global average pooling
  → scalar regression head
  → OCP grid decoder (R3m, C2m)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..physics.ocp_grid import GRID_LENGTH


class ResBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int,
                 n_blocks: int, kernel_size: int, dropout: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 1),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            ResBlock1d(base_channels, kernel_size, dropout) for _ in range(n_blocks)
        ])

    def forward(self, x):  # x: [B, C, L]
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return x  # [B, base_channels, L]


class InverseModel(nn.Module):
    """
    입력: [B, 10, 512] feature tensor
    출력:
        scalar: [B, n_scalar]
        ocp_R3m: [B, 256]
        ocp_C2m: [B, 256]
    """

    def __init__(self,
                 in_channels: int = 10,
                 base_channels: int = 64,
                 n_blocks: int = 6,
                 kernel_size: int = 7,
                 dropout: float = 0.1,
                 use_transformer: bool = True,
                 transformer_heads: int = 8,
                 transformer_layers: int = 2,
                 n_scalar: int = 9,
                 ocp_grid_len: int = GRID_LENGTH):
        super().__init__()

        self.encoder = CNNEncoder(in_channels, base_channels, n_blocks, kernel_size, dropout)

        # Transformer encoder (선택적)
        self.use_transformer = use_transformer
        if use_transformer:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=base_channels, nhead=transformer_heads,
                dim_feedforward=base_channels * 4, dropout=dropout,
                batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        # Global pooling → embedding
        self.pool = nn.AdaptiveAvgPool1d(1)  # [B, C, 1] → [B, C]

        # Scalar head
        self.scalar_head = nn.Sequential(
            nn.Linear(base_channels, base_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 2, n_scalar),
        )

        # OCP decoder (공유 backbone + 두 phase 별도 head)
        self.ocp_shared = nn.Sequential(
            nn.Linear(base_channels, base_channels * 2),
            nn.GELU(),
        )
        self.ocp_R3m_head = nn.Linear(base_channels * 2, ocp_grid_len)
        self.ocp_C2m_head = nn.Linear(base_channels * 2, ocp_grid_len)

    def forward(self, x: torch.Tensor):
        # CNN
        feat = self.encoder(x)  # [B, C, L]

        # Transformer
        if self.use_transformer:
            feat = feat.permute(0, 2, 1)   # [B, L, C]
            feat = self.transformer(feat)
            feat = feat.permute(0, 2, 1)   # [B, C, L]

        # Pooling
        emb = self.pool(feat).squeeze(-1)  # [B, C]

        # Heads
        scalar = self.scalar_head(emb)          # [B, n_scalar]
        ocp_shared = self.ocp_shared(emb)        # [B, C*2]
        ocp_R3m = self.ocp_R3m_head(ocp_shared)  # [B, 256]
        ocp_C2m = self.ocp_C2m_head(ocp_shared)  # [B, 256]

        return scalar, ocp_R3m, ocp_C2m
