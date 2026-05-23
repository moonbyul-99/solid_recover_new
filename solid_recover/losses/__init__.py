"""Loss functions for Solid Recover."""

from solid_recover.losses.clip import CLIPLoss, WeightedCLIPLoss
from solid_recover.losses.composite import VAEClipLoss
from solid_recover.losses.recon import ReconLoss
from solid_recover.losses.vae import VAELoss

__all__ = [
    "ReconLoss",
    "VAELoss",
    "CLIPLoss",
    "WeightedCLIPLoss",
    "VAEClipLoss",
]
