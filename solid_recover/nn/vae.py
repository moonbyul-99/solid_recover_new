"""Single-omic VAE backbone for Solid Recover.

The ``forward`` method only returns tensors / dicts of tensors; **losses are no
longer computed inside ``forward``** (previously they were in
``sr_net.sr_vae.forward``). Downstream consumers compose losses explicitly,
which keeps inference and training decoupled.

The deterministic AE variant that used to live here (``SRAE``) has been
removed in v2: the project has converged on VAE-only training, so keeping a
parallel AE code path (plus its matching ``AELoss`` / ``vae_model`` switch)
was just dead weight.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from solid_recover.nn.encoder import FeatureDecoder, FeatureEncoder, HiddenParams


class SRVAE(nn.Module):
    """Variational autoencoder used for single-omic representation learning.

    Sub-module names (``encoder``, ``mu_proj``, ``logvar_proj``, ``decoder``) are
    preserved so that checkpoints produced by the legacy ``sr_net.sr_vae`` can
    be loaded directly.
    """

    def __init__(
        self,
        feature_num: int,
        hidden_params: HiddenParams,
        embed_dim: int,
        use_rmsnorm: bool = True,
        use_residual: bool = False,
        dropout_p: float = 0.05,
    ) -> None:
        super().__init__()

        self.encoder = FeatureEncoder(
            feature_num=feature_num,
            hidden_params=hidden_params,
            use_rmsnorm=use_rmsnorm,
            use_residual=use_residual,
            dropout_p=dropout_p,
        )

        # Keep ``hidden_dims`` for decoder construction and introspection.
        self.hidden_dims = self.encoder.hidden_dims.copy()

        last_hidden = self.hidden_dims[-1]
        self.mu_proj = nn.Sequential(nn.Linear(last_hidden, embed_dim), nn.RMSNorm(embed_dim))
        self.logvar_proj = nn.Sequential(nn.Linear(last_hidden, embed_dim), nn.RMSNorm(embed_dim))

        decoder_hidden = self.hidden_dims + [embed_dim]
        decoder_hidden.reverse()
        self.decoder = FeatureDecoder(
            feature_num=feature_num,
            hidden_params=decoder_hidden,
            use_rmsnorm=use_rmsnorm,
            use_residual=use_residual,
            dropout_p=dropout_p,
        )
        self.embed_dim = embed_dim

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Re-parameterisation trick."""
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        return mu + std * eps

    def get_embedding(self, x: torch.Tensor):
        """Return ``(z, z_mu, z_logvar, z_embed)`` without running the decoder."""
        z = self.encoder(x)
        z_mu = self.mu_proj(z)
        z_logvar = self.logvar_proj(z)
        z_embed = self.reparam(z_mu, z_logvar)
        return z, z_mu, z_logvar, z_embed

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z, z_mu, z_logvar, z_embed = self.get_embedding(x)
        x_recon = self.decoder(z_embed)
        return {
            "z_encoder": z,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "z_embed": z_embed,
            "x_recon": x_recon,
        }
