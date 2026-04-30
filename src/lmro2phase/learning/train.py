"""Stage 3: PyTorch Lightning training loop."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

log = logging.getLogger(__name__)


class InverseLightningModule(nn.Module):
    """간단한 수동 학습 루프 (Lightning 없이도 동작)."""

    def __init__(self, model, optimizer, scheduler, weights: dict,
                 use_permutation: bool = True):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.weights = weights
        self.use_permutation = use_permutation

    def training_step(self, batch):
        from .losses import total_loss
        X, y_scalar, y_ocp = batch
        scalar_pred, ocp_R3m_pred, ocp_C2m_pred = self.model(X)
        loss = total_loss(
            scalar_pred, ocp_R3m_pred, ocp_C2m_pred,
            y_scalar, y_ocp[:, 0], y_ocp[:, 1],
            self.weights, self.use_permutation,
        )
        return loss

    def validation_step(self, batch):
        with torch.no_grad():
            return self.training_step(batch)


def train(model, dataset,
           weights: dict,
           batch_size: int = 64,
           max_epochs: int = 200,
           lr: float = 1e-3,
           weight_decay: float = 1e-4,
           train_frac: float = 0.9,
           device: str = "auto",
           checkpoint_dir: Optional[Path] = None,
           use_permutation: bool = True) -> nn.Module:
    """학습 메인 루프."""
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    log.info(f"학습 디바이스: {dev}")

    # train/val split
    n_train = int(len(dataset) * train_frac)
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=(device == "cuda"))

    model = model.to(dev)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    module = InverseLightningModule(model, optimizer, scheduler, weights, use_permutation)

    best_val_loss = float("inf")

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = [b.to(dev) for b in batch]
            optimizer.zero_grad()
            loss = module.training_step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        for batch in val_loader:
            batch = [b.to(dev) for b in batch]
            val_losses.append(module.validation_step(batch).item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss   = sum(val_losses) / len(val_losses)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            log.info(f"  Epoch {epoch:3d}/{max_epochs}: "
                     f"train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")

    log.info(f"학습 완료. best val_loss={best_val_loss:.4f}")
    return model
