"""VAE reconstruction + KL loss.

Solid Recover dropped the deterministic-AE branch; ``AELoss`` and the
corresponding ``SRAE`` backbone are no longer provided.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from solid_recover.losses.recon import ReconLoss


class VAELoss(nn.Module):
    """Reconstruction + KL divergence, keyed by ``loss``, ``recon_loss``, ``kl_loss``.

    ``beta`` scales the KL term (beta-VAE). It is kept positional-compatible
    with the legacy API so that both ``VAELoss(1.0)`` and ``VAELoss(beta=1.0)``
    work; downstream callers (``SinglePretrain.set_loss``, the CLI, smoke
    tests, notebooks) all use the ``beta=`` keyword.
    """

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.recon_loss = ReconLoss()

    @staticmethod
    def _kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Per-sample KL for diagonal Gaussians, then mean over batch.
        per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return per_sample.mean()

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        recon = self.recon_loss(recon_x, x)
        kl = self._kl_loss(mu, logvar)
        loss = recon + self.beta * kl
        return {"loss": loss, "kl_loss": kl, "recon_loss": recon}
