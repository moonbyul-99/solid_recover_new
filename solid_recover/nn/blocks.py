"""Basic fully-connected block used by encoders and decoders."""

from __future__ import annotations

import torch
import torch.nn as nn


class FCBlock(nn.Module):
    """Linear + GELU (+ optional RMSNorm) + Dropout, with optional residual.

    Parameters
    ----------
    input_dim:
        Input feature dimension.
    output_dim:
        Output feature dimension.
    use_rmsnorm:
        Whether to apply ``nn.RMSNorm`` after activation.
    use_residual:
        Whether to add a residual connection. Only enabled when
        ``input_dim == output_dim``.
    dropout_p:
        Dropout probability applied after (optional) normalisation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_rmsnorm: bool = True,
        use_residual: bool = False,
        dropout_p: float = 0.05,
    ) -> None:
        super().__init__()

        self.use_residual = use_residual and (input_dim == output_dim)
        self.use_rmsnorm = use_rmsnorm

        layers: list[nn.Module] = [nn.Linear(input_dim, output_dim), nn.GELU()]
        if use_rmsnorm:
            layers.append(nn.RMSNorm(output_dim))
        layers.append(nn.Dropout(dropout_p))
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        if self.use_residual:
            return x + out
        return out
