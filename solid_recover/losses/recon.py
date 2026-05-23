"""Reconstruction loss (MSE summed over features, averaged over batch)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconLoss(nn.Module):
    """Per-sample sum of squared errors, averaged across the batch.

    Mirrors legacy ``sr_loss.recon_Loss`` exactly. Using ``reduction='none'``
    keeps the per-feature granularity before summing over the feature axis.
    """

    def forward(self, recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(recon_x, x, reduction="none")
        loss = loss.sum(dim=1).mean()
        return loss
