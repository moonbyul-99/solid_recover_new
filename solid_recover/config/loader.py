"""YAML <-> :class:`TrainConfig` conversion."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml

from solid_recover._logging import get_logger
from solid_recover.config.schema import (
    CkptConfig,
    DataConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    TrainConfig,
    TrainingConfig,
)

_logger = get_logger(__name__)

T = TypeVar("T")


def _filter_kwargs(cls: Type[T], values: Dict[str, Any]) -> Dict[str, Any]:
    """Drop any keys not declared on the dataclass, warning the user."""
    valid = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    extra = set(values.keys()) - valid
    if extra:
        _logger.warning("Ignoring unknown keys in %s: %s", cls.__name__, sorted(extra))
    return {key: values[key] for key in values if key in valid}


def load_raw_config(config_path: str) -> Dict[str, Any]:
    """Return the raw YAML dict (without schema validation)."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_train_config(config_path: str, validate: bool = True) -> TrainConfig:
    """Load a YAML file into a :class:`TrainConfig` dataclass.

    Missing sections fall back to defaults; unknown keys are logged and
    dropped instead of raising, so legacy configs continue to load.
    """
    raw = load_raw_config(config_path)

    task = raw.get("task") or _infer_task_from_raw(raw)

    cfg = TrainConfig(
        task=task,  # type: ignore[arg-type]
        data=DataConfig(**_filter_kwargs(DataConfig, raw.get("data", {}) or {})),
        model=ModelConfig(**_filter_kwargs(ModelConfig, raw.get("model", {}) or {})),
        optimizer=OptimizerConfig(
            **_filter_kwargs(OptimizerConfig, raw.get("optimizer", {}) or {})
        ),
        training=TrainingConfig(
            **_filter_kwargs(TrainingConfig, raw.get("training", {}) or {})
        ),
        loss=LossConfig(**_filter_kwargs(LossConfig, raw.get("loss", {}) or {})),
        ckpt=CkptConfig(**_filter_kwargs(CkptConfig, raw.get("ckpt", {}) or {})),
    )

    # Float coercion for a few fields commonly quoted as strings in YAML.
    cfg.optimizer.lr = float(cfg.optimizer.lr)
    cfg.optimizer.min_lr = float(cfg.optimizer.min_lr)
    cfg.loss.clip_weight = float(cfg.loss.clip_weight)
    cfg.loss.cross_recon_1 = float(cfg.loss.cross_recon_1)
    cfg.loss.cross_recon_2 = float(cfg.loss.cross_recon_2)
    cfg.loss.temperature = float(cfg.loss.temperature)

    if validate:
        cfg.validate()
    return cfg


def _infer_task_from_raw(raw: Dict[str, Any]) -> str:
    """Best-effort inference for legacy configs that omit ``task``."""
    model = raw.get("model", {}) or {}
    if "feature_num" in model and "hidden_params" in model:
        return "single_pretrain"
    ckpt = raw.get("ckpt") or {}
    if ckpt.get("omic_1") and ckpt.get("omic_2"):
        return "pair_pretrain"
    return "pair_scratch"


def dump_train_config(cfg: TrainConfig, path: str) -> None:
    """Persist a :class:`TrainConfig` back to YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
