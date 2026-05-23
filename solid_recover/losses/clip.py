"""CLIP-style contrastive losses for paired multi-omics.

Numerics and public behaviour are identical to legacy
``sr_loss.CLIPLoss`` / ``sr_loss.WeightedCLIPLoss`` with one intentional
change: ``logit_scale`` is a fixed buffer, **not** a learnable parameter.
The (short-lived) trainable-temperature variant was retired after the
corresponding ablation; keeping a non-parameter ``logit_scale`` avoids the
stateful edge case where a checkpoint saved via ``net.state_dict()`` silently
drops the loss temperature.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _log_inv_temperature(temperature: float) -> torch.Tensor:
    """Return ``log(1 / temperature)`` as a 1-element float32 tensor."""
    return torch.tensor([math.log(1.0 / temperature)], dtype=torch.float32)


class CLIPLoss(nn.Module):
    """Symmetric InfoNCE with a fixed log-temperature (``logit_scale``)."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        # Buffer (not Parameter): value is fixed and moves with .to(device)
        # but never receives gradients.
        self.register_buffer("logit_scale", _log_inv_temperature(temperature))

    def forward(self, cell_1: torch.Tensor, cell_2: torch.Tensor) -> torch.Tensor:
        cell_1 = F.normalize(cell_1, dim=-1)
        cell_2 = F.normalize(cell_2, dim=-1)

        logits = cell_1 @ cell_2.t()
        scale = torch.exp(self.logit_scale)
        logits_12 = scale * logits
        logits_21 = logits_12.t()

        target = torch.arange(len(cell_1), dtype=torch.long, device=cell_1.device)
        loss_12 = F.cross_entropy(logits_12, target)
        loss_21 = F.cross_entropy(logits_21, target)
        return (loss_12 + loss_21) / 2


class WeightedCLIPLoss(nn.Module):
    """Numerically stable weighted CLIP using log-sum-exp.

    Re-weights top-k and bottom-k off-diagonal entries per row to emphasise or
    down-weight hard negatives. Diagonal weights remain 1 (log 0).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        top_k_ratio: float = 0.1,
        bottom_k_ratio: float = 0.1,
        weight_top: float = 0.1,
        weight_bottom: float = 2.0,
    ) -> None:
        super().__init__()
        self.register_buffer("logit_scale", _log_inv_temperature(temperature))
        self.top_k_ratio = top_k_ratio
        self.bottom_k_ratio = bottom_k_ratio
        self.weight_top = weight_top
        self.weight_bottom = weight_bottom

        self.log_weight_top = float(np.log(weight_top + 1e-8))
        self.log_weight_bottom = float(np.log(weight_bottom + 1e-8))

    def _compute_weighted_logsumexp(
        self, logits: torch.Tensor, w_log: torch.Tensor
    ) -> torch.Tensor:
        """Return ``logsumexp_j(logits_ij + log W_ij)`` per row."""
        return torch.logsumexp(logits + w_log, dim=1)

    def _compute_weight_matrix(self, logits: torch.Tensor) -> torch.Tensor:
        """Vectorised log-weight matrix for all rows."""
        device = logits.device
        n = logits.size(0)

        logits_masked = logits.clone()
        logits_masked.fill_diagonal_(-float("inf"))  # keep diagonal out of top/bottom
        _, sorted_idx = torch.sort(logits_masked, descending=True, dim=1)

        num_off_diag = n - 1
        k_top = max(1, int(self.top_k_ratio * num_off_diag))
        k_bottom = max(1, int(self.bottom_k_ratio * num_off_diag))

        w_log = torch.zeros((n, n), device=device)

        top_indices = sorted_idx[:, :k_top]
        bottom_indices = sorted_idx[:, -k_bottom:]

        row_indices = torch.arange(n, device=device).unsqueeze(1)
        w_log[row_indices, top_indices] = self.log_weight_top
        w_log[row_indices, bottom_indices] = self.log_weight_bottom
        w_log.fill_diagonal_(0.0)

        return w_log

    def forward(self, rna_emb: torch.Tensor, atac_emb: torch.Tensor) -> torch.Tensor:
        rna_emb = F.normalize(rna_emb, dim=-1)
        atac_emb = F.normalize(atac_emb, dim=-1)

        logits = rna_emb @ atac_emb.t()
        logits = torch.exp(self.logit_scale) * logits

        # rna -> atac
        w_log_r2a = self._compute_weight_matrix(logits)
        log_numer = logits.diag()
        log_denom = self._compute_weighted_logsumexp(logits, w_log_r2a)
        loss_r2a = -(log_numer - log_denom).mean()

        # atac -> rna (re-compute weights on transposed logits)
        logits_t = logits.t()
        w_log_a2r = self._compute_weight_matrix(logits_t)
        log_numer_t = logits_t.diag()
        log_denom_t = self._compute_weighted_logsumexp(logits_t, w_log_a2r)
        loss_a2r = -(log_numer_t - log_denom_t).mean()

        return (loss_r2a + loss_a2r) / 2.0
