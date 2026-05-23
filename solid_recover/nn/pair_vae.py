"""Paired VAE backbone for two-modality integration.

Compared to legacy ``sr_net.sr_pair_vae``:
- ``forward`` no longer computes any loss; it only returns the forward-pass
  dict. Loss composition lives in :class:`solid_recover.losses.VAEClipLoss`.
- Sub-module names ``model_1`` and ``model_2`` are preserved so that legacy
  checkpoints remain loadable via ``load_state_dict``.
- Optional batch-conditional injection (CVAE-style): when ``num_batches > 0``
  and ``batch_embed_dim > 0``, a learned ``nn.Embedding`` is created and
  ``forward`` accepts a ``batch_indices`` tensor.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from solid_recover.nn.encoder import HiddenParams
from solid_recover.nn.vae import SRVAE


class SRPairVAE(nn.Module):
    """Two stacked :class:`SRVAE` modules sharing an embedding dimension."""

    def __init__(
        self,
        feature_num_1: int,
        feature_num_2: int,
        hidden_params_1: HiddenParams,
        hidden_params_2: HiddenParams,
        embed_dim: int,
        use_rmsnorm: bool = True,
        use_residual: bool = False,
        dropout_p: float = 0.05,
        num_batches: int = 0,
        batch_embed_dim: int = 0,
    ) -> None:
        super().__init__()

        self.num_batches = num_batches
        self.batch_embed_dim = batch_embed_dim

        if num_batches > 0 and batch_embed_dim > 0:
            self.batch_embedding = nn.Embedding(num_batches, batch_embed_dim)
        else:
            self.batch_embedding = None

        self.model_1 = SRVAE(
            feature_num=feature_num_1,
            hidden_params=hidden_params_1,
            embed_dim=embed_dim,
            use_rmsnorm=use_rmsnorm,
            use_residual=use_residual,
            dropout_p=dropout_p,
            batch_embed_dim=batch_embed_dim,
        )
        self.model_2 = SRVAE(
            feature_num=feature_num_2,
            hidden_params=hidden_params_2,
            embed_dim=embed_dim,
            use_rmsnorm=use_rmsnorm,
            use_residual=use_residual,
            dropout_p=dropout_p,
            batch_embed_dim=batch_embed_dim,
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Run both modalities plus cross reconstruction.

        Parameters
        ----------
        x1, x2 : torch.Tensor
            Modality input batches.
        batch_indices : Optional[torch.Tensor], shape (N,)
            Integer batch labels used for CVAE-style conditional injection.
            When ``None``, the model behaves identically to the original SRPairVAE.

        Returns
        -------
        dict with two keys ``x1`` and ``x2`` (per-modality output dicts):
            - ``x_recon``: self-reconstruction from own embedding
            - ``cross_recon``: reconstruction from the *other* modality's embedding
            - ``z``, ``z_mu``, ``z_logvar``, ``z_embed``: encoder outputs
        """
        # Compute the batch embedding once if enabled.
        batch_embed = None
        if self.batch_embedding is not None:
            if batch_indices is not None:
                batch_embed = self.batch_embedding(batch_indices)
            else:
                # Backward compatibility: no batch info → zero embedding
                N = x1.shape[0]
                batch_embed = torch.zeros(N, self.batch_embed_dim, device=x1.device)

        z1, z1_mu, z1_logvar, z1_embed = self.model_1.get_embedding(x1)
        z2, z2_mu, z2_logvar, z2_embed = self.model_2.get_embedding(x2)

        # Self-recon with batch context
        x1_self = self.model_1.decoder(z1_embed, batch_embed=batch_embed)
        x2_self = self.model_2.decoder(z2_embed, batch_embed=batch_embed)

        # Cross-recon: decoder receives batch_embed for the *target* modality
        # (which is the same batch_embed since both modalities share batch context)
        x1_cross = self.model_1.decoder(z2_embed, batch_embed=batch_embed)
        x2_cross = self.model_2.decoder(z1_embed, batch_embed=batch_embed)

        x1_dic = {
            "x_recon": x1_self,
            "cross_recon": x1_cross,
            "z": z1,
            "z_mu": z1_mu,
            "z_logvar": z1_logvar,
            "z_embed": z1_embed,
        }
        x2_dic = {
            "x_recon": x2_self,
            "cross_recon": x2_cross,
            "z": z2,
            "z_mu": z2_mu,
            "z_logvar": z2_logvar,
            "z_embed": z2_embed,
        }
        return {"x1": x1_dic, "x2": x2_dic}
