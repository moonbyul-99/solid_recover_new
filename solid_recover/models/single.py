"""Single-omic VAE pretrain facade.

Wraps :class:`solid_recover.nn.SRVAE` with :class:`solid_recover.losses.VAELoss`.
Mirrors the API of the legacy ``sr_model.single_sr`` but with clearer
separation between the network, loss and training loop. The AE variant was
removed in v2 (see :mod:`solid_recover.nn.vae`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from anndata import AnnData
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from solid_recover._logging import get_logger
from solid_recover.data.adata_utils import adata_to_tensor
from solid_recover.data.datasets import SingleDataset
from solid_recover.losses.vae import VAELoss
from solid_recover.models.base import BaseModel
from solid_recover.nn.encoder import HiddenParams
from solid_recover.nn.vae import SRVAE

_logger = get_logger(__name__)


class SinglePretrain(BaseModel):
    """Single-omic VAE model facade."""

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
        self.net: torch.nn.Module = SRVAE(
            feature_num=feature_num,
            hidden_params=hidden_params,
            embed_dim=embed_dim,
            use_rmsnorm=use_rmsnorm,
            use_residual=use_residual,
            dropout_p=dropout_p,
        )
        self.embed_dim = embed_dim
        self.feature_num = feature_num
        # Loss is defaulted to VAE beta=1; override via ``set_loss``.
        self.loss_fn: torch.nn.Module = VAELoss(beta=1.0)

    # ------------------------------------------------------------------
    def set_loss(self, beta: float = 1.0) -> None:
        """(Re)configure the VAE reconstruction + KL loss."""
        self.loss_fn = VAELoss(beta=beta)

    # ------------------------------------------------------------------
    def create_dataset(
        self,
        adata: AnnData,
        train_idx: Optional[np.ndarray] = None,
        test_idx: Optional[np.ndarray] = None,
        test_size: Optional[float] = None,
        random_state: int = 42,
    ) -> Tuple[SingleDataset, SingleDataset]:
        """Build train/test :class:`SingleDataset` from an ``AnnData``."""
        if train_idx is None and test_idx is None:
            if test_size is None:
                test_size = min(0.1, 50000 / adata.shape[0])
            train_idx, test_idx = train_test_split(
                np.arange(adata.shape[0]), test_size=test_size, random_state=random_state
            )
        assert train_idx is not None and test_idx is not None

        train_dataset = SingleDataset(adata_to_tensor(adata[train_idx, :]))
        test_dataset = SingleDataset(adata_to_tensor(adata[test_idx, :]))
        self.setup_data(train_dataset, test_dataset)
        return train_dataset, test_dataset

    # ------------------------------------------------------------------
    def _process_batch(
        self, batch: Dict[str, torch.Tensor], device: str
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        x = batch["feature"].to(device)
        outputs = self.net(x)
        loss_dic = self._compute_loss(outputs, x)
        return outputs, loss_dic

    def _compute_loss(
        self, outputs: Dict[str, torch.Tensor], x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.loss_fn(
            recon_x=outputs["x_recon"],
            x=x,
            mu=outputs["z_mu"],
            logvar=outputs["z_logvar"],
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: AnnData,
        embedding_keys: Union[str, List[str]] = ("z_embed",),
        device: str = "cuda",
        batch_size: int = 128,
        inplace: bool = True,
    ) -> AnnData:
        """Populate ``adata.obsm['sr_<key>']`` for each requested embedding.

        Replaces legacy ``single_sr.get_embedding`` and fixes the dead
        ``self.calculate_loss(...)`` call that referenced a non-existent method.
        """
        if isinstance(embedding_keys, str):
            embedding_keys = [embedding_keys]
        embedding_keys = list(embedding_keys)

        loader = DataLoader(
            SingleDataset(adata_to_tensor(adata)), batch_size=batch_size, shuffle=False
        )

        self.net.to(device).eval()
        embed_dic: Dict[str, List[np.ndarray]] = {key: [] for key in embedding_keys}

        for batch in loader:
            x = batch["feature"].to(device)
            outputs = self.net(x)
            for key in embedding_keys:
                if key not in outputs:
                    _logger.warning("embedding key %r not found in outputs", key)
                    continue
                embed_dic[key].append(outputs[key].detach().cpu().numpy())

        target = adata if inplace else adata.copy()
        for key in embedding_keys:
            if embed_dic[key]:
                target.obsm[f"sr_{key}"] = np.concatenate(embed_dic[key], axis=0)
        return target
