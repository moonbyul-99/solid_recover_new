"""Harmony-inspired cluster-level batch alignment loss.

Strategy 3: Penalises batch divergence within latent clusters defined by
learnable cluster centres (EMA-updated).  The idea is that within each
"cluster" the distribution should be similar across batches.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchAlignmentLoss(nn.Module):
    """Cluster-level batch alignment loss.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the latent space (typically 64).
    n_clusters : int
        Number of learnable clustering centres.
    ema_momentum : float
        Exponential moving average coefficient for centre updates.
    temperature : float
        Soft-assignment temperature (lower = harder assignment).
    """

    def __init__(
        self,
        embed_dim: int,
        n_clusters: int = 20,
        ema_momentum: float = 0.9,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.ema_momentum = ema_momentum
        self.temperature = temperature

        # Learnable L2-normalised cluster centres
        centres = torch.randn(n_clusters, embed_dim)
        centres = F.normalize(centres, dim=1)
        self.register_buffer("centres", centres)

    def forward(
        self, z: torch.Tensor, batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute batch alignment loss for a batch of latent codes.

        Parameters
        ----------
        z : torch.Tensor, shape (N, embed_dim)
            Latent representations (z_mu).
        batch_indices : torch.Tensor, shape (N,)
            Integer batch labels.

        Returns
        -------
        torch.Tensor — scalar loss.
        """
        N, D = z.shape
        device = z.device

        # L2-normalise
        z_norm = F.normalize(z, dim=1)
        centres_norm = F.normalize(self.centres, dim=1)

        # Cosine similarity -> soft assignment
        sim = z_norm @ centres_norm.T  # (N, K)
        soft_assign = F.softmax(sim / self.temperature, dim=1)  # (N, K)

        unique_batches = torch.unique(batch_indices)
        if len(unique_batches) <= 1:
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(0.0, device=device)

        for k in range(self.n_clusters):
            weights_k = soft_assign[:, k]  # (N,)
            w_sum = weights_k.sum() + 1e-8

            # Per-batch weighted mean for cluster k
            batch_means = {}
            for b in unique_batches:
                mask = (batch_indices == b)
                w_b = weights_k[mask]
                if w_b.sum() < 1e-8:
                    continue
                batch_means[int(b.item())] = (z_norm[mask] * w_b.unsqueeze(-1)).sum(0) / (w_b.sum() + 1e-8)

            if len(batch_means) < 2:
                continue

            # Global cluster mean
            global_mean_k = (z_norm * weights_k.unsqueeze(-1)).sum(0) / w_sum

            # Penalise per-batch deviation from global mean
            for bm in batch_means.values():
                diff = bm - global_mean_k
                loss = loss + (diff * diff).sum()

        loss = loss / (self.n_clusters * max(len(unique_batches) - 1, 1))

        return loss
