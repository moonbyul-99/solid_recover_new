"""Composite loss for paired VAE training (VAE + cross-recon + CLIP)."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from solid_recover.losses.clip import CLIPLoss, WeightedCLIPLoss
from solid_recover.losses.recon import ReconLoss
from solid_recover.losses.vae import VAELoss


class VAEClipLoss(nn.Module):
    """Self-recon + cross-recon + KL + CLIP.

    Total loss (kept identical to legacy ``sr_loss.VAE_clip_loss``):

    ::

        L = cross_recon_1 * recon_cross_1 + (1 - cross_recon_1) * recon_self_1
          + cross_recon_2 * recon_cross_2 + (1 - cross_recon_2) * recon_self_2
          + vae_beta_1 * KL_1
          + vae_beta_2 * KL_2
          + clip_weight * CLIP(z_embed_1, z_embed_2)
    """

    def __init__(
        self,
        vae_beta_1: float = 1.0,
        vae_beta_2: float = 1.0,
        clip_weight: float = 1.0,
        cross_recon_1: float = 0.2,
        cross_recon_2: float = 0.2,
        temperature: float = 0.07,
        use_weight: bool = False,
        top_k_ratio: float = 0.1,
        bottom_k_ratio: float = 0.1,
        weight_top: float = 0.1,
        weight_bottom: float = 2.0,
    ) -> None:
        super().__init__()

        if not (0.0 <= cross_recon_1 <= 1.0):
            raise ValueError("cross_recon_1 must be in [0, 1]")
        if not (0.0 <= cross_recon_2 <= 1.0):
            raise ValueError("cross_recon_2 must be in [0, 1]")

        self.vae_beta_1 = vae_beta_1
        self.vae_beta_2 = vae_beta_2
        self.clip_weight = clip_weight
        self.cross_recon_1 = cross_recon_1
        self.cross_recon_2 = cross_recon_2

        if use_weight:
            self.clip_loss = WeightedCLIPLoss(
                temperature=temperature,
                top_k_ratio=top_k_ratio,
                bottom_k_ratio=bottom_k_ratio,
                weight_top=weight_top,
                weight_bottom=weight_bottom,
            )
        else:
            self.clip_loss = CLIPLoss(temperature=temperature)

        # ``logit_scale`` is a fixed buffer inside :class:`CLIPLoss`; v2 dropped
        # the trainable-temperature ablation switch entirely.

        self.vae_loss_1 = VAELoss(self.vae_beta_1)
        self.vae_loss_2 = VAELoss(self.vae_beta_2)
        self.recon_loss = ReconLoss()


    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        sr_pair_out: Dict[str, object],
    ) -> Dict[str, torch.Tensor]:
        x1_dic = sr_pair_out["x1"]
        x2_dic = sr_pair_out["x2"]

        vae_1 = self.vae_loss_1(
            recon_x=x1_dic["x_recon"], x=x1, mu=x1_dic["z_mu"], logvar=x1_dic["z_logvar"]
        )
        vae_2 = self.vae_loss_2(
            recon_x=x2_dic["x_recon"], x=x2, mu=x2_dic["z_mu"], logvar=x2_dic["z_logvar"]
        )

        # CLIP on reparam-sampled embeddings (matches legacy behaviour).
        clip = self.clip_loss(x1_dic["z_embed"], x2_dic["z_embed"])

        cross_1 = self.recon_loss(x1_dic["cross_recon"], x1)
        cross_2 = self.recon_loss(x2_dic["cross_recon"], x2)

        loss = (
            self.cross_recon_1 * cross_1
            + (1 - self.cross_recon_1) * vae_1["recon_loss"]
            + self.cross_recon_2 * cross_2
            + (1 - self.cross_recon_2) * vae_2["recon_loss"]
            + self.vae_beta_1 * vae_1["kl_loss"]
            + self.vae_beta_2 * vae_2["kl_loss"]
            + self.clip_weight * clip
        )

        result: Dict[str, torch.Tensor] = {
            "loss": loss,
            "recon_loss_1": vae_1["recon_loss"],
            "recon_loss_2": vae_2["recon_loss"],
            "cross_loss_1": cross_1,
            "cross_loss_2": cross_2,
            "kl_loss_1": vae_1["kl_loss"],
            "kl_loss_2": vae_2["kl_loss"],
            "clip_loss": clip,
        }

        return result
