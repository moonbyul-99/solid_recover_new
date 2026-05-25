"""Composite loss for paired VAE training (VAE + cross-recon + CLIP + batch-aware losses)."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from solid_recover.losses.clip import CLIPLoss, WeightedCLIPLoss
from solid_recover.losses.recon import ReconLoss
from solid_recover.losses.vae import VAELoss


class VAEClipLoss(nn.Module):
    """Self-recon + cross-recon + KL + CLIP + optional batch-aware losses.

    Total loss ::

        L = cross_recon_1 * recon_cross_1 + (1 - cross_recon_1) * recon_self_1
          + cross_recon_2 * recon_cross_2 + (1 - cross_recon_2) * recon_self_2
          + vae_beta_1 * KL_1
          + vae_beta_2 * KL_2
          + clip_weight * CLIP(z_embed_1, z_embed_2)
          + adversarial_batch_weight * adversarial_loss
          + batch_alignment_weight * alignment_loss
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
        # --- GRL / Harmony (暂不启用) ---
        # adversarial_batch_weight: float = 0.0,
        # num_batches: int = 0,
        # discriminator_hidden_dim: int = 128,
        # embed_dim: int = 256,
        # batch_alignment_weight: float = 0.0,
        # alignment_n_clusters: int = 20,
        # alignment_ema_momentum: float = 0.9,
        # alignment_temperature: float = 1.0,
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

        self.vae_loss_1 = VAELoss(self.vae_beta_1)
        self.vae_loss_2 = VAELoss(self.vae_beta_2)
        self.recon_loss = ReconLoss()

        # --- Adversarial batch training (Strategy 2) — 暂不启用 ---
        # self.adversarial_batch_weight = adversarial_batch_weight
        # self.grl = None
        # self.batch_discriminator = None
        # if adversarial_batch_weight > 0 and num_batches > 0:
        #     from solid_recover.nn.batch_discriminator import (
        #         BatchDiscriminator,
        #         GradientReversalLayer,
        #     )
        #     self.grl = GradientReversalLayer(lambda_=1.0)
        #     self.batch_discriminator = BatchDiscriminator(
        #         input_dim=embed_dim,
        #         num_batches=num_batches,
        #         hidden_dim=discriminator_hidden_dim,
        #     )

        # --- Batch alignment loss (Strategy 3) — 暂不启用 ---
        # self.batch_alignment_weight = batch_alignment_weight
        # self.alignment_loss = None
        # if batch_alignment_weight > 0:
        #     from solid_recover.losses.batch_alignment import BatchAlignmentLoss
        #     self.alignment_loss = BatchAlignmentLoss(
        #         embed_dim=embed_dim,
        #         n_clusters=alignment_n_clusters,
        #         ema_momentum=alignment_ema_momentum,
        #         temperature=alignment_temperature,
        #     )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        sr_pair_out: Dict[str, object],
        batch_indices: Optional[torch.Tensor] = None,
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

        # --- Adversarial batch loss (Strategy 2) — 暂不启用 ---
        # if self.batch_discriminator is not None and batch_indices is not None:
        #     adv_1 = F.cross_entropy(
        #         self.batch_discriminator(self.grl(x1_dic["z_mu"])), batch_indices
        #     )
        #     adv_2 = F.cross_entropy(
        #         self.batch_discriminator(self.grl(x2_dic["z_mu"])), batch_indices
        #     )
        #     adv_loss = (adv_1 + adv_2) / 2
        #     loss = loss + self.adversarial_batch_weight * adv_loss
        #     result["loss"] = loss
        #     result["adversarial_loss"] = adv_loss

        # --- Batch alignment loss (Strategy 3) — 暂不启用 ---
        # if self.alignment_loss is not None and batch_indices is not None:
        #     align_1 = self.alignment_loss(x1_dic["z_mu"], batch_indices)
        #     align_2 = self.alignment_loss(x2_dic["z_mu"], batch_indices)
        #     align_total = (align_1 + align_2) / 2
        #     loss = loss + self.batch_alignment_weight * align_total
        #     result["loss"] = loss
        #     result["alignment_loss"] = align_total

        return result
