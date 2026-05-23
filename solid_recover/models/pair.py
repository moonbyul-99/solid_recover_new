"""Paired-omic model facades (scratch training and pretrain fine-tuning).

- :class:`PairScratch`: train :class:`SRPairVAE` + :class:`VAEClipLoss` from
  scratch.
- :class:`PairPretrain`: subclass that additionally loads per-modality VAE
  checkpoints before training (mirrors legacy ``pair_sr_pretrain``).

Checkpoint format on disk remains ``{'model_state_dict': ...}``; the state
dict keys match the legacy network (``model_1.*``, ``model_2.*``).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from muon import MuData
from sklearn.model_selection import train_test_split

from solid_recover._logging import get_logger
from solid_recover.data.adata_utils import adata_to_tensor
from solid_recover.data.datasets import PairDataset
from solid_recover.losses.composite import VAEClipLoss
from solid_recover.models.base import BaseModel
from solid_recover.nn.encoder import HiddenParams
from solid_recover.nn.pair_vae import SRPairVAE

_logger = get_logger(__name__)


class PairScratch(BaseModel):
    """Paired multi-omic model trained end-to-end from random init."""

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
        self.feature_num_1 = feature_num_1
        self.feature_num_2 = feature_num_2
        self.embed_dim = embed_dim
        self.net: torch.nn.Module = SRPairVAE(
            feature_num_1=feature_num_1,
            feature_num_2=feature_num_2,
            hidden_params_1=hidden_params_1,
            hidden_params_2=hidden_params_2,
            embed_dim=embed_dim,
            use_rmsnorm=use_rmsnorm,
            use_residual=use_residual,
            dropout_p=dropout_p,
        )
        # Placeholder loss; final config goes through :meth:`set_loss`.
        self.loss_fn: VAEClipLoss = VAEClipLoss()

    # ------------------------------------------------------------------
    def set_loss(
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
        self.loss_fn = VAEClipLoss(
            vae_beta_1=vae_beta_1,
            vae_beta_2=vae_beta_2,
            clip_weight=clip_weight,
            cross_recon_1=cross_recon_1,
            cross_recon_2=cross_recon_2,
            temperature=temperature,
            use_weight=use_weight,
            top_k_ratio=top_k_ratio,
            bottom_k_ratio=bottom_k_ratio,
            weight_top=weight_top,
            weight_bottom=weight_bottom,
        )

    # ------------------------------------------------------------------
    def create_dataset(
        self,
        mdata: MuData,
        key_1: str,
        key_2: str,
        train_idx: Optional[np.ndarray] = None,
        test_idx: Optional[np.ndarray] = None,
        test_size: Optional[float] = None,
        random_state: int = 42,
    ) -> Tuple[PairDataset, PairDataset]:
        """Build paired train/test datasets from a :class:`MuData`.

        Fixes the legacy bug where ``adata.shape[0]`` (undefined) was used to
        compute ``test_size``; now uses ``mdata.shape[0]`` as intended.
        """
        if train_idx is None and test_idx is None:
            if test_size is None:
                test_size = min(0.1, 50000 / mdata.shape[0])
            train_idx, test_idx = train_test_split(
                np.arange(mdata.shape[0]), test_size=test_size, random_state=random_state
            )
        assert train_idx is not None and test_idx is not None

        train_data_1 = mdata[key_1][train_idx, :]
        test_data_1 = mdata[key_1][test_idx, :]
        train_data_2 = mdata[key_2][train_idx, :]
        test_data_2 = mdata[key_2][test_idx, :]


        train_dataset = PairDataset(
            adata_to_tensor(train_data_1),
            adata_to_tensor(train_data_2),
        )
        test_dataset = PairDataset(
            adata_to_tensor(test_data_1),
            adata_to_tensor(test_data_2),
        )
        self.setup_data(train_dataset, test_dataset)
        return train_dataset, test_dataset

    # ------------------------------------------------------------------
    def _process_batch(
        self, batch: Dict[str, torch.Tensor], device: str
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        x1 = batch["omic_1"].to(device)
        x2 = batch["omic_2"].to(device)
        # Make sure the loss module lives on the same device.
        self.loss_fn.to(device)
        outputs = self.net(x1, x2)

        loss_dic = self.loss_fn(x1, x2, outputs)
        # Surface the clip logit_scale as a scalar for tensorboard parity.
        loss_dic["logit_scale"] = self.loss_fn.clip_loss.logit_scale.detach()
        # Flatten outputs so the Trainer's recon-detection heuristic still works.
        flat_outputs = {
            "x1_recon": outputs["x1"]["x_recon"],
            "x2_recon": outputs["x2"]["x_recon"],
            "x1_cross_recon": outputs["x1"]["cross_recon"],
            "x2_cross_recon": outputs["x2"]["cross_recon"],
        }
        return flat_outputs, loss_dic

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_cross_modality(
        self,
        omic_1: torch.Tensor,
        omic_2: torch.Tensor,
        device: str = "cuda",
    ) -> Dict[str, np.ndarray]:
        """Return cross-modality reconstructions and latent means.

        Returns
        -------
        dict with keys:
            - ``omic_1_cross_recon``: modality-1 reconstructed *from* the
              modality-2 embedding (shape ``[N, feature_num_1]``)
            - ``omic_2_cross_recon``: modality-2 reconstructed *from* the
              modality-1 embedding (shape ``[N, feature_num_2]``)
            - ``z_mu_1`` / ``z_mu_2``: per-modality latent means

        v2: renamed from the legacy ``rna2atac`` / ``atac2rna`` pair which
        was only accurate for RNA+ATAC experiments.
        """
        self.net.to(device).eval()
        outputs = self.net(omic_1.to(device), omic_2.to(device))
        return {
            "omic_1_cross_recon": outputs["x1"]["cross_recon"].detach().cpu().numpy(),
            "omic_2_cross_recon": outputs["x2"]["cross_recon"].detach().cpu().numpy(),
            "z_mu_1": outputs["x1"]["z_mu"].detach().cpu().numpy(),
            "z_mu_2": outputs["x2"]["z_mu"].detach().cpu().numpy(),
        }


class PairPretrain(PairScratch):
    """Paired model initialized from two single-omic VAE checkpoints."""

    def load_pretrained(self, omic_1_ckpt: str, omic_2_ckpt: str) -> None:
        """Load per-modality :class:`SRVAE` weights into the sub-modules.

        The state dicts were produced by :class:`SinglePretrain.save` (or the
        legacy ``single_sr`` class) and have identical key names as the
        ``model_1`` / ``model_2`` sub-modules of :class:`SRPairVAE`.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        _logger.info("Loading omic-1 pretrained weights: %s", omic_1_ckpt)
        ckpt1 = torch.load(omic_1_ckpt, map_location=device)
        self.net.model_1.load_state_dict(ckpt1["model_state_dict"])

        _logger.info("Loading omic-2 pretrained weights: %s", omic_2_ckpt)
        ckpt2 = torch.load(omic_2_ckpt, map_location=device)
        self.net.model_2.load_state_dict(ckpt2["model_state_dict"])

    # Convenient aliases ------------------------------------------------
    load_pretrained_model = load_pretrained  # legacy name
