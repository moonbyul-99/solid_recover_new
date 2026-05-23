"""Generic training loop for Solid Recover models.

The :class:`Trainer` owns *only* the loop itself (steps, logging, checkpointing
and evaluation). It does not know how to build the network or compose losses;
callers are responsible for wiring up the optimizer, scheduler, data loaders
and two small callbacks:

- ``process_batch``: ``(batch, device) -> (outputs, loss_dict)``
- ``save_state_dict``: ``() -> dict`` that will be stored as ``model_state_dict``

This mirrors (and replaces) the monolithic training loop that previously lived
inside ``sr_model.Base_sr`` while preserving the exact on-disk checkpoint
format (``{'model_state_dict': ...}``) so legacy checkpoints stay loadable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from solid_recover._logging import get_logger

_logger = get_logger(__name__)

ProcessBatchFn = Callable[[dict, str], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
StateDictFn = Callable[[], Dict[str, torch.Tensor]]


@dataclass
class TrainerConfig:
    """Knobs controlling the training loop."""

    train_steps: int
    eval_points: int = 500
    save_points: int = 1000
    device: str = "cuda"
    drop_small_last_batch: bool = True
    small_batch_ratio: float = 0.8


@dataclass
class Trainer:
    """Encapsulates the training/eval loop.

    Parameters
    ----------
    optimizer, scheduler:
        Already constructed and bound to the module parameters.
    train_loader, test_loader:
        Torch data loaders.
    process_batch:
        Callback returning ``(outputs_dict, loss_dict)``. ``loss_dict`` must
        contain a ``"loss"`` entry with the scalar to ``.backward()``; other
        entries that contain ``"loss"`` or ``"logit_scale"`` in their name are
        logged verbatim.
    save_state_dict:
        Callback returning the ``nn.Module.state_dict()`` to persist. Decoupled
        so the legacy ``{'model_state_dict': net.state_dict()}`` format stays
        byte-compatible.
    project_dir:
        Project directory; ``logs/`` and ``models/`` are created underneath.
    config:
        :class:`TrainerConfig` instance.
    """

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    train_loader: DataLoader
    test_loader: Optional[DataLoader]
    process_batch: ProcessBatchFn
    save_state_dict: StateDictFn
    project_dir: str
    config: TrainerConfig

    _writer: SummaryWriter = field(init=False)
    model_dir: str = field(init=False)
    log_dir: str = field(init=False)

    def __post_init__(self) -> None:
        self.project_dir = self._ensure_project_dir(self.project_dir)
        self.model_dir = os.path.join(self.project_dir, "models")
        self.log_dir = os.path.join(self.project_dir, "logs")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=self.log_dir)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_project_dir(project_dir: str) -> str:
        if not os.path.exists(project_dir):
            os.makedirs(project_dir, exist_ok=True)
            return project_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        suffixed = f"{project_dir}_{timestamp}"
        os.makedirs(suffixed, exist_ok=True)
        return suffixed

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    def save_checkpoint(self, steps: int) -> str:
        """Persist ``{'model_state_dict': ...}`` to ``models/ckpt_<steps>.pth``."""
        ckpt_path = os.path.join(self.model_dir, f"ckpt_{steps}.pth")
        torch.save({"model_state_dict": self.save_state_dict()}, ckpt_path)
        return ckpt_path

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, step: int) -> Dict[str, float]:
        """Run one pass over ``test_loader`` and log scalar means to tensorboard."""
        if self.test_loader is None:
            return {}

        device = self.config.device
        total = 0
        accum: Dict[str, float] = {}

        for batch in self.test_loader:
            outputs, loss_dic = self.process_batch(batch, device)

            # Batch size is pulled from any recon output (same heuristic as legacy).
            b = 0
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and "recon" in key:
                    b = value.shape[0]
                    break
            if b == 0:
                # fall back to the first tensor in the batch dict
                for value in batch.values():
                    if isinstance(value, torch.Tensor):
                        b = value.shape[0]
                        break
            total += b
            for key, value in loss_dic.items():
                if "loss" in key:
                    accum[f"{key}/val"] = accum.get(f"{key}/val", 0.0) + value.item() * b

        means = {key: value / max(total, 1) for key, value in accum.items()}
        for key, value in means.items():
            self._writer.add_scalar(key, value, step)
        return means

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def fit(self) -> None:
        """Run ``train_steps`` optimisation steps."""
        cfg = self.config
        device = cfg.device
        steps = 0
        steps_per_epoch = max(len(self.train_loader), 1)
        epoch_num = cfg.train_steps // steps_per_epoch + 1

        for _ in range(epoch_num):
            for batch in tqdm(self.train_loader):
                if cfg.drop_small_last_batch and isinstance(batch, dict):
                    cur = next(
                        (v.size(0) for v in batch.values() if isinstance(v, torch.Tensor)),
                        None,
                    )
                    if (
                        cur is not None
                        and self.train_loader.batch_size is not None
                        and cur <= cfg.small_batch_ratio * self.train_loader.batch_size
                    ):
                        continue

                _, loss_dic = self.process_batch(batch, device)
                loss = loss_dic["loss"]
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                steps += 1
                for key, value in loss_dic.items():
                    if "loss" in key or "logit_scale" in key:
                        scalar = value.item() if isinstance(value, torch.Tensor) else value
                        self._writer.add_scalar(f"{key}/train", scalar, steps)
                self._writer.add_scalar(
                    "learning_rate", self.scheduler.get_last_lr()[0], steps
                )

                if steps >= cfg.train_steps:
                    if steps % cfg.eval_points == 0 or steps == cfg.train_steps:
                        self.evaluate(steps)
                    if steps % cfg.save_points == 0 or steps == cfg.train_steps:
                        self.save_checkpoint(steps)
                    break
                if steps % cfg.eval_points == 0:
                    self.evaluate(steps)
                if steps % cfg.save_points == 0:
                    self.save_checkpoint(steps)

            if steps >= cfg.train_steps:
                break

        # Save final checkpoint if training ended naturally (before reaching train_steps)
        if steps > 0 and steps < cfg.train_steps:
            _logger.info(
                "Training ended at step %d (< %d). Saving final checkpoint.",
                steps,
                cfg.train_steps,
            )
            self.evaluate(steps)
            self.save_checkpoint(steps)

        _logger.info("Training finished (%d steps)", steps)
