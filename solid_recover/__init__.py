"""Solid Recover: single-cell multi-omics integration and cross-modality prediction."""

from solid_recover._logging import get_logger
from solid_recover.models.single import SinglePretrain
from solid_recover.models.pair import PairPretrain, PairScratch
from solid_recover.analysis import (
    GRNBuilder,
    decompose_latent_to_features,
)

__version__ = "0.1.0"

__all__ = [
    "SinglePretrain",
    "PairScratch",
    "PairPretrain",
    "get_logger",
    "__version__",
    # Analysis utilities
    "decompose_latent_to_features",
    "GRNBuilder",
]
