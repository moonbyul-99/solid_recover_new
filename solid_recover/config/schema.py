"""Dataclass schema for Solid Recover training/eval configs.

One config file drives one training run. The discriminator is ``task``
(``"single_pretrain"``, ``"pair_scratch"``, or ``"pair_pretrain"``) which
controls which subset of fields is required.

The schema is intentionally permissive: extra keys in YAML are dropped with a
warning, and optional fields have sensible defaults. This mirrors what the
legacy ad-hoc scripts tolerated while still giving a single source of truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional, Union

from solid_recover._logging import get_logger

_logger = get_logger(__name__)

HiddenParams = Union[Dict[str, int], List[int]]
Task = Literal["single_pretrain", "pair_scratch", "pair_pretrain"]


@dataclass
class DataConfig:
    """Data wiring.

    Paired tasks accept either:

    - Pre-split: ``train_data_path`` + ``test_data_path`` (both ``.h5mu``)
    - Auto-split: ``data_path`` + ``test_size`` (random train/test split)

    single-omic pretraining uses ``dataset_path`` (HuggingFace ``datasets``
    folder) and splits it on the fly via ``test_size``.
    """

    batch_size: int = 128
    # paired (pre-split)
    train_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    # paired (auto-split from single file)
    data_path: Optional[str] = None
    test_size: float = 0.1
    seed: int = 42
    key_1: Optional[str] = None
    key_2: Optional[str] = None
    to_gpu: bool = False
    # single
    dataset_path: Optional[str] = None


@dataclass
class ModelConfig:
    """Network topology. Fields irrelevant to the current task are ignored."""

    embed_dim: int = 256
    use_rmsnorm: bool = True
    use_residual: bool = True
    dropout_p: float = 0.0
    # single-omic
    feature_num: Optional[int] = None
    hidden_params: Optional[HiddenParams] = None
    # paired
    feature_num_1: Optional[int] = None
    feature_num_2: Optional[int] = None
    hidden_params_1: Optional[HiddenParams] = None
    hidden_params_2: Optional[HiddenParams] = None


@dataclass
class OptimizerConfig:
    """AdamW + :class:`SRScheduler`."""

    lr: float = 5e-4
    warmup_steps: int = 1000
    steady_1_steps: int = 2000
    cosine_anneal_steps: int = 8000
    min_lr: float = 1e-6


@dataclass
class TrainingConfig:
    """Training loop knobs."""

    project_dir: str = "outputs/default_run"
    train_steps: int = 10000
    eval_points: int = 500
    save_points: int = 1000
    device: str = "cuda"


@dataclass
class LossConfig:
    """Loss weights; paired task uses the full set, single uses only ``beta``."""

    # single
    beta: float = 1.0
    # paired
    vae_beta_1: float = 1.0
    vae_beta_2: float = 1.0
    clip_weight: float = 1.0
    cross_recon_1: float = 0.2
    cross_recon_2: float = 0.2
    temperature: float = 0.07
    use_weight: bool = False
    top_k_ratio: float = 0.1
    bottom_k_ratio: float = 0.1
    weight_top: float = 0.1
    weight_bottom: float = 2.0


@dataclass
class CkptConfig:
    """Used by ``pair_pretrain`` to point at two single-omic checkpoints."""

    omic_1: Optional[str] = None
    omic_2: Optional[str] = None


@dataclass
class TrainConfig:
    """Top-level training config.

    Mirrors the flat YAML layout used by the legacy scripts but now
    dataclass-backed.
    """

    task: Task = "pair_scratch"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    ckpt: CkptConfig = field(default_factory=CkptConfig)

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Cross-field validation; raises :class:`ValueError` on failure."""
        task = self.task
        m = self.model
        d = self.data

        if task == "single_pretrain":
            if m.feature_num is None or m.hidden_params is None:
                raise ValueError(
                    "single_pretrain requires model.feature_num and "
                    "model.hidden_params"
                )
            if d.dataset_path is None:
                raise ValueError("single_pretrain requires data.dataset_path")
        elif task in ("pair_scratch", "pair_pretrain"):
            required_model = {
                "model.feature_num_1": m.feature_num_1,
                "model.feature_num_2": m.feature_num_2,
                "model.hidden_params_1": m.hidden_params_1,
                "model.hidden_params_2": m.hidden_params_2,
                "data.key_1": d.key_1,
                "data.key_2": d.key_2,
            }
            missing = [key for key, value in required_model.items() if value is None]
            if missing:
                raise ValueError(f"{task} is missing required fields: {missing}")

            # data_path: either pre-split OR auto-split from single file
            has_pre_split = d.train_data_path is not None and d.test_data_path is not None
            has_auto_split = d.data_path is not None
            if not has_pre_split and not has_auto_split:
                raise ValueError(
                    f"{task} requires either (data.train_data_path + data.test_data_path) "
                    f"or data.data_path (for auto-split)"
                )
            if task == "pair_pretrain":
                if self.ckpt.omic_1 is None or self.ckpt.omic_2 is None:
                    raise ValueError(
                        "pair_pretrain requires ckpt.omic_1 and ckpt.omic_2"
                    )
        else:
            raise ValueError(f"Unknown task: {task!r}")

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)
