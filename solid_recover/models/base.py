"""Shared facade base class for high-level Solid Recover models.

Subclasses wrap a ``nn.Module`` (``self.net``) plus a loss module and expose a
scvi-style API (``setup_data``, ``train``, ``save``, ``load_state_dict``). The
heavy lifting of the training loop itself lives in
:class:`solid_recover.training.Trainer`.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from solid_recover._logging import get_logger
from solid_recover.training.scheduler import SRScheduler
from solid_recover.training.trainer import Trainer, TrainerConfig

_logger = get_logger(__name__)


class BaseModel:
    """Base facade; subclasses must set ``self.net`` and ``self.loss_fn``."""

    net: nn.Module
    loss_fn: nn.Module

    def __init__(self) -> None:
        self.train_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.batch_size: Optional[int] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[SRScheduler] = None
        self.project_dir: Optional[str] = None
        self._trainer: Optional[Trainer] = None

    # ------------------------------------------------------------------
    # data wiring
    # ------------------------------------------------------------------
    def setup_data(
        self,
        train_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 128,
        num_workers: int = 0,
    ) -> None:
        """Attach train / (optional) test datasets and build loaders.

        ``test_dataset`` is optional: ablation / quick-smoke runs that have no
        held-out split can pass ``None`` and :class:`Trainer.evaluate` will
        simply no-op. When omitted, ``self.test_loader`` stays ``None`` and
        the standard ``train(...)`` entry point will still run end-to-end.
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        else:
            self.test_loader = None

    # ------------------------------------------------------------------
    # optimizer & project
    # ------------------------------------------------------------------
    def configure_optimizer(
        self,
        lr: float,
        warmup_steps: int,
        steady_1_steps: int,
        cosine_anneal_steps: int,
        min_lr: float = 1e-6,
    ) -> None:
        """Build AdamW + :class:`SRScheduler` for ``self.net``."""
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)
        self.scheduler = SRScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            steady_1_steps=steady_1_steps,
            cosine_anneal_steps=cosine_anneal_steps,
            min_lr=min_lr,
        )

    def set_project(self, project_dir: str) -> str:
        """Record the desired project directory; actual creation happens in Trainer."""
        self.project_dir = project_dir
        return project_dir

    # ------------------------------------------------------------------
    # training / evaluation
    # ------------------------------------------------------------------
    def _process_batch(
        self, batch: Dict[str, torch.Tensor], device: str
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def train(
        self,
        train_steps: int,
        eval_points: int = 500,
        save_points: int = 1000,
        device: str = "cuda",
        config_copy_path: Optional[str] = None,
        training_config: Optional[Any] = None,
    ) -> Trainer:
        """Run the training loop.

        Parameters
        ----------
        train_steps, eval_points, save_points, device:
            Forwarded to :class:`TrainerConfig`.
        config_copy_path:
            Optional path to an existing YAML config file; copied verbatim into
            ``<project_dir>/config.yaml``. Use this when you want a byte-exact
            snapshot of the file the user launched the run with (mirrors the
            legacy behaviour).
        training_config:
            Optional :class:`solid_recover.config.schema.TrainConfig`. When
            provided (and ``config_copy_path`` is not), the *effective* config
            is serialised via :func:`dump_train_config` so that any in-memory
            tweaks the caller applied after loading the YAML are faithfully
            persisted alongside the checkpoints. This is the preferred hook
            for notebook / library users who build the config programmatically.
        """
        if self.project_dir is None:
            raise RuntimeError("call set_project(project_dir) first")
        if self.train_loader is None:
            raise RuntimeError("call setup_data(...) first")
        if self.optimizer is None or self.scheduler is None:
            raise RuntimeError("call configure_optimizer(...) first")

        self.net.to(device)
        self.net.train()

        trainer = Trainer(
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            process_batch=self._process_batch,
            save_state_dict=lambda: self.net.state_dict(),
            project_dir=self.project_dir,
            config=TrainerConfig(
                train_steps=train_steps,
                eval_points=eval_points,
                save_points=save_points,
                device=device,
            ),
        )
        self._trainer = trainer
        # Persist final project_dir (may have been timestamped).
        self.project_dir = trainer.project_dir

        self._persist_config(
            trainer.project_dir,
            config_copy_path=config_copy_path,
            training_config=training_config,
        )

        trainer.fit()
        return trainer

    # ------------------------------------------------------------------
    @staticmethod
    def _persist_config(
        project_dir: str,
        config_copy_path: Optional[str],
        training_config: Optional[Any],
    ) -> None:
        """Write ``<project_dir>/config.yaml`` from whichever source is given.

        Priority order:
            1. ``training_config`` (preferred; reflects in-memory edits)
            2. ``config_copy_path`` (byte-exact YAML snapshot)
            3. Neither -> no-op, warn once.
        """
        target = os.path.join(project_dir, "config.yaml")

        if training_config is not None:
            # Lazy import to avoid pulling the config layer into the nn path.
            from solid_recover.config.loader import dump_train_config

            dump_train_config(training_config, target)
            _logger.info("Wrote effective TrainConfig to %s", target)
            return

        if config_copy_path is not None and os.path.exists(config_copy_path):
            shutil.copyfile(config_copy_path, target)
            _logger.info("Copied config snapshot %s -> %s", config_copy_path, target)
            return

        _logger.warning(
            "No config_copy_path / training_config provided; %s will not be written. "
            "Consider passing training_config=cfg in model.train(...).",
            target,
        )

    # ------------------------------------------------------------------
    # checkpoint I/O
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save only the network state_dict (legacy format)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model_state_dict": self.net.state_dict()}, path)

    def load_state_dict(self, checkpoint_path: str, strict: bool = False) -> None:
        """Load a ``{'model_state_dict': ...}`` checkpoint.

        ``strict=False`` by default because the legacy paired checkpoint also
        contains a ``clip_loss.logit_scale`` buffer that now lives inside the
        loss module, not the network itself.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        missing, unexpected = self.net.load_state_dict(state, strict=strict)
        if missing:
            _logger.warning("load_state_dict missing keys: %s", missing)
        if unexpected:
            _logger.warning("load_state_dict unexpected keys: %s", unexpected)

    # convenience ------------------------------------------------------
    def to(self, device: str) -> "BaseModel":
        self.net.to(device)
        return self

    def eval(self) -> "BaseModel":
        self.net.eval()
        return self


# # ----------------------------------------------------------------------
# # Classifier-adapter helper (reused by SinglePretrain)
# # ----------------------------------------------------------------------
# def _assert_dict_or_float(lr_dic: Any, keys) -> Dict[str, float]:
#     if isinstance(lr_dic, float):
#         return {key: lr_dic for key in keys}
#     return {key: float(value) for key, value in lr_dic.items()}
