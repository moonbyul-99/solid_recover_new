"""Feature encoder and decoder used by SRVAE / SRAE."""

from __future__ import annotations

from typing import Dict, List, Union

import torch
import torch.nn as nn

from solid_recover.nn.blocks import FCBlock

HiddenParams = Union[Dict[str, int], List[int]]


def _parse_hidden_params(hidden_params: HiddenParams) -> List[int]:
    """Normalise ``hidden_params`` into an explicit list of layer widths.

    Accepts either:
    - ``list[int]``: interpreted as explicit per-layer widths
    - ``dict``: must contain ``hidden_dim`` and ``block_num``; expanded to a
      flat list of repeated widths.
    """
    if isinstance(hidden_params, dict):
        required = {"hidden_dim", "block_num"}
        if not required.issubset(hidden_params.keys()):
            raise ValueError(
                "hidden_params is dict, must contain 'hidden_dim' and 'block_num'"
            )
        hidden_dim = int(hidden_params["hidden_dim"])
        block_num = int(hidden_params["block_num"])
        return [hidden_dim] * block_num
    if isinstance(hidden_params, list):
        return [int(v) for v in hidden_params]
    raise TypeError(
        "hidden_params must be either a dict (with 'hidden_dim', 'block_num') "
        "or a list of ints"
    )


class FeatureEncoder(nn.Module):
    """Stack of :class:`FCBlock` mapping raw features to a hidden representation.

    The architecture preserves the original ``sr_net.feature_encoder`` layout so
    that legacy checkpoints remain loadable:

    - ``encoder_header``: first :class:`FCBlock` (``feature_num -> hidden_dims[0]``)
    - ``fc_blocks``: remaining :class:`FCBlock`\\ s chained via ``nn.Sequential``
    """

    def __init__(
        self,
        feature_num: int,
        hidden_params: HiddenParams,
        use_rmsnorm: bool = True,
        use_residual: bool = False,
        dropout_p: float = 0.05,
    ) -> None:
        super().__init__()

        # Keep a copy so callers / state_dict introspection can read it back.
        self.hidden_params = (
            hidden_params.copy() if isinstance(hidden_params, (list, dict)) else hidden_params
        )
        self.hidden_dims = _parse_hidden_params(hidden_params)

        self.encoder_header = FCBlock(
            input_dim=feature_num,
            output_dim=self.hidden_dims[0],
            use_rmsnorm=use_rmsnorm,
            use_residual=False,
            dropout_p=dropout_p,
        )

        blocks = [
            FCBlock(
                self.hidden_dims[i - 1],
                self.hidden_dims[i],
                use_rmsnorm=use_rmsnorm,
                use_residual=use_residual,
                dropout_p=dropout_p,
            )
            for i in range(1, len(self.hidden_dims))
        ]
        self.fc_blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder_header(x)
        return self.fc_blocks(z)


class FeatureDecoder(nn.Module):
    """Mirror of :class:`FeatureEncoder`.

    The ``decoder_header`` is a ``Linear + LeakyReLU`` that projects the last
    hidden dim back to ``feature_num``. ``fc_blocks`` chains the intermediate
    hidden widths. Layout matches the original ``sr_net.feature_decoder`` to
    preserve checkpoint compatibility.

    When ``batch_embed_dim > 0``, the first :class:`FCBlock` accepts a wider
    input (``hidden_dims[0] + batch_embed_dim``) and :meth:`forward` expects an
    additional ``batch_embed`` argument that is concatenated with ``z``.
    """

    def __init__(
        self,
        feature_num: int,
        hidden_params: HiddenParams,
        use_rmsnorm: bool = True,
        use_residual: bool = False,
        dropout_p: float = 0.05,
        batch_embed_dim: int = 0,
    ) -> None:
        super().__init__()

        self.hidden_params = (
            hidden_params.copy() if isinstance(hidden_params, (list, dict)) else hidden_params
        )
        self.hidden_dims = _parse_hidden_params(hidden_params)
        self.batch_embed_dim = batch_embed_dim

        self.decoder_header = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], feature_num),
            nn.LeakyReLU(),
        )

        if batch_embed_dim > 0 and len(self.hidden_dims) >= 2:
            # First FCBlock accepts z || batch_embed
            first_input = self.hidden_dims[0] + batch_embed_dim
        else:
            first_input = self.hidden_dims[0] if self.hidden_dims else 0

        blocks = [
            FCBlock(
                first_input if i == 0 else self.hidden_dims[i - 1],
                self.hidden_dims[i],
                use_rmsnorm=use_rmsnorm,
                use_residual=use_residual,
                dropout_p=dropout_p,
            )
            for i in range(0, len(self.hidden_dims))
        ]
        # If batch_embed_dim > 0 and only one hidden_dim, we still need the widened first block
        if batch_embed_dim > 0 and len(self.hidden_dims) == 1 and not blocks:
            blocks = [
                FCBlock(
                    self.hidden_dims[0] + batch_embed_dim,
                    self.hidden_dims[0],
                    use_rmsnorm=use_rmsnorm,
                    use_residual=use_residual,
                    dropout_p=dropout_p,
                )
            ]
        self.fc_blocks = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor, batch_embed: torch.Tensor = None) -> torch.Tensor:
        if self.batch_embed_dim > 0:
            if batch_embed is None:
                raise ValueError(
                    f"FeatureDecoder expects batch_embed (batch_embed_dim={self.batch_embed_dim}), "
                    "but received None"
                )
            z = torch.cat([z, batch_embed], dim=-1)
        z = self.fc_blocks(z)
        return self.decoder_header(z)
