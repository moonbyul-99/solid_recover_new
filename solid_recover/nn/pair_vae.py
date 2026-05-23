"""Paired VAE backbone for two-modality integration.

Compared to legacy ``sr_net.sr_pair_vae``:
- ``forward`` no longer computes any loss; it only returns the forward-pass
  dict. Loss composition lives in :class:`solid_recover.losses.VAEClipLoss`.
- Sub-module names ``model_1`` and ``model_2`` are preserved so that legacy
  checkpoints remain loadable via ``load_state_dict``.
"""

from __future__ import annotations

from typing import Dict

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
    ) -> None:
        super().__init__()

        self.model_1 = SRVAE(
            feature_num=feature_num_1,
            hidden_params=hidden_params_1,
            embed_dim=embed_dim,
            use_rmsnorm=use_rmsnorm,
            use_residual=use_residual,
            dropout_p=dropout_p,
        )
        # v2: forward ``dropout_p`` to the second SRVAE as well (legacy
        # ``sr_pair_vae`` silently dropped it, which was a latent bug rather
        # than intentional asymmetry). ``dropout`` has no trainable params, so
        # legacy checkpoints still load cleanly.
        self.model_2 = SRVAE(
            feature_num=feature_num_2,
            hidden_params=hidden_params_2,
            embed_dim=embed_dim,
            use_rmsnorm=use_rmsnorm,
            use_residual=use_residual,
            dropout_p=dropout_p,
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """Run both modalities plus cross reconstruction.

        Returns
        -------
        dict with two keys ``x1`` and ``x2`` (per-modality output dicts):
            - ``x_recon``: self-reconstruction from own embedding
            - ``cross_recon``: reconstruction from the *other* modality's
              embedding (v2 renamed from the ambiguous legacy
              ``x1_c_recon`` / ``x2_c_recon`` which were pulled out into
              the top-level dict)
            - ``z``, ``z_mu``, ``z_logvar``, ``z_embed``: encoder outputs
        """
        z1, z1_mu, z1_logvar, z1_embed = self.model_1.get_embedding(x1)
        z2, z2_mu, z2_logvar, z2_embed = self.model_2.get_embedding(x2)

        x1_self = self.model_1.decoder(z1_embed)
        x2_self = self.model_2.decoder(z2_embed)

        # ``x1_cross``: modality-1 space reconstructed from modality-2 embed.
        # ``x2_cross``: modality-2 space reconstructed from modality-1 embed.
        x1_cross = self.model_1.decoder(z2_embed)
        x2_cross = self.model_2.decoder(z1_embed)

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
