"""Gradient Reversal Layer (GRL) and Batch Discriminator for adversarial training.

Used by Strategy 2 to penalise the encoder for producing batch-discriminable
latent representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal: identity in forward, ``-lambda * grad`` in backward."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Thin wrapper around :class:`GradientReversalFunction`."""

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("lambda_", torch.tensor(lambda_))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class BatchDiscriminator(nn.Module):
    """Two-hidden-layer MLP predicting batch identity from latent ``z_mu``."""

    def __init__(
        self,
        input_dim: int,
        num_batches: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_batches),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


__all__ = ["GradientReversalLayer", "GradientReversalFunction", "BatchDiscriminator"]
