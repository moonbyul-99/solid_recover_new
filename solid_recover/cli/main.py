"""Unified ``solid-recover`` command line entry point.

Usage
-----
::

    solid-recover train --config configs/pair_scratch_case8.yaml
    solid-recover train --config configs/rna_pretrain.yaml
    solid-recover eval  --output-dir outputs/pair_scratch_case8_YYYYMMDD

The ``train`` sub-command dispatches on ``config.task`` (``single_pretrain`` /
``pair_scratch`` / ``pair_pretrain``) and reproduces the behaviour of the old
top-level scripts ``single_omic_pretrain.py`` / ``pair_scratch.py`` /
``pair_pretrain.py`` in a single entry point.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from solid_recover._logging import get_logger
from solid_recover.config.loader import load_train_config
from solid_recover.config.schema import TrainConfig

_logger = get_logger(__name__)


# ----------------------------------------------------------------------
# dispatch table
# ----------------------------------------------------------------------
def _train_single(cfg: TrainConfig, config_path: str) -> None:
    from datasets import load_from_disk

    from solid_recover.models.single import SinglePretrain

    dataset = load_from_disk(cfg.data.dataset_path)
    split = dataset.train_test_split(test_size=cfg.data.test_size, seed=cfg.data.seed)
    train_ds, test_ds = split["train"], split["test"]

    # ``feature_num`` is authoritative from config; fall back to data shape.
    feature_num = cfg.model.feature_num or int(dataset[0]["feature"].shape[0])

    model = SinglePretrain(
        feature_num=feature_num,
        hidden_params=cfg.model.hidden_params,  # type: ignore[arg-type]
        embed_dim=cfg.model.embed_dim,
        use_rmsnorm=cfg.model.use_rmsnorm,
        use_residual=cfg.model.use_residual,
        dropout_p=cfg.model.dropout_p,
    )
    model.setup_data(train_ds, test_ds, batch_size=cfg.data.batch_size)
    model.set_loss(beta=cfg.loss.beta)
    model.configure_optimizer(
        lr=cfg.optimizer.lr,
        warmup_steps=cfg.optimizer.warmup_steps,
        steady_1_steps=cfg.optimizer.steady_1_steps,
        cosine_anneal_steps=cfg.optimizer.cosine_anneal_steps,
        min_lr=cfg.optimizer.min_lr,
    )
    model.set_project(cfg.training.project_dir)
    model.train(
        train_steps=cfg.training.train_steps,
        eval_points=cfg.training.eval_points,
        save_points=cfg.training.save_points,
        device=cfg.training.device,
        config_copy_path=config_path,
    )


def _train_pair(cfg: TrainConfig, config_path: str, pretrain: bool) -> None:
    import torch

    from solid_recover.data.prepare import (
        prepare_pair_data,
        prepare_pair_data_from_single,
    )
    from solid_recover.models.pair import PairPretrain, PairScratch

    # --- Determine batch_indices if batch-aware strategies are enabled ---
    batch_indices = None
    if cfg.data.num_batches > 0:
        import muon as mu

        if cfg.data.data_path is not None:
            data_file = cfg.data.data_path
        else:
            data_file = cfg.data.train_data_path  # type: ignore[arg-type]
        mdata = mu.read_h5mu(data_file)
        sample_ids = mdata[cfg.data.key_1].obs["Sample_ID"]  # type: ignore[index,arg-type]
        unique_ids = {v: i for i, v in enumerate(sample_ids.unique())}
        batch_indices = torch.tensor(
            [unique_ids[v] for v in sample_ids], dtype=torch.long
        )

    # --- Load data (auto-split or pre-split) ---
    if cfg.data.data_path is not None:
        train_ds, test_ds = prepare_pair_data_from_single(
            data_path=cfg.data.data_path,
            key_1=cfg.data.key_1,  # type: ignore[arg-type]
            key_2=cfg.data.key_2,  # type: ignore[arg-type]
            test_size=cfg.data.test_size,
            seed=cfg.data.seed,
            to_gpu=cfg.data.to_gpu,
            batch_indices=batch_indices,
        )
    else:
        train_ds, test_ds = prepare_pair_data(
            train_data_path=cfg.data.train_data_path,  # type: ignore[arg-type]
            test_data_path=cfg.data.test_data_path,  # type: ignore[arg-type]
            key_1=cfg.data.key_1,  # type: ignore[arg-type]
            key_2=cfg.data.key_2,  # type: ignore[arg-type]
            to_gpu=cfg.data.to_gpu,
            batch_indices=batch_indices,
        )

    feat_1 = cfg.model.feature_num_1 or train_ds[0]["omic_1"].shape[0]
    feat_2 = cfg.model.feature_num_2 or train_ds[0]["omic_2"].shape[0]
    if cfg.model.feature_num_1 is not None:
        assert feat_1 == train_ds[0]["omic_1"].shape[0], (
            f"feature_num_1 ({feat_1}) != omic_1 width ({train_ds[0]['omic_1'].shape[0]})"
        )
    if cfg.model.feature_num_2 is not None:
        assert feat_2 == train_ds[0]["omic_2"].shape[0], (
            f"feature_num_2 ({feat_2}) != omic_2 width ({train_ds[0]['omic_2'].shape[0]})"
        )

    cls = PairPretrain if pretrain else PairScratch
    model = cls(
        feature_num_1=feat_1,
        feature_num_2=feat_2,
        hidden_params_1=cfg.model.hidden_params_1,  # type: ignore[arg-type]
        hidden_params_2=cfg.model.hidden_params_2,  # type: ignore[arg-type]
        embed_dim=cfg.model.embed_dim,
        use_rmsnorm=cfg.model.use_rmsnorm,
        use_residual=cfg.model.use_residual,
        dropout_p=cfg.model.dropout_p,
        num_batches=cfg.data.num_batches,
        batch_embed_dim=cfg.model.batch_embed_dim,
    )

    if pretrain:
        assert isinstance(model, PairPretrain)
        model.load_pretrained(
            omic_1_ckpt=cfg.ckpt.omic_1,  # type: ignore[arg-type]
            omic_2_ckpt=cfg.ckpt.omic_2,  # type: ignore[arg-type]
        )

    model.setup_data(train_ds, test_ds, batch_size=cfg.data.batch_size)
    model.set_loss(
        vae_beta_1=cfg.loss.vae_beta_1,
        vae_beta_2=cfg.loss.vae_beta_2,
        clip_weight=cfg.loss.clip_weight,
        cross_recon_1=cfg.loss.cross_recon_1,
        cross_recon_2=cfg.loss.cross_recon_2,
        temperature=cfg.loss.temperature,
        use_weight=cfg.loss.use_weight,
        top_k_ratio=cfg.loss.top_k_ratio,
        bottom_k_ratio=cfg.loss.bottom_k_ratio,
        weight_top=cfg.loss.weight_top,
        weight_bottom=cfg.loss.weight_bottom,
        # --- GRL / Harmony (暂不启用) ---
        # adversarial_batch_weight=cfg.loss.adversarial_batch_weight,
        # num_batches=cfg.data.num_batches,
        # discriminator_hidden_dim=cfg.loss.discriminator_hidden_dim,
        # batch_alignment_weight=cfg.loss.batch_alignment_weight,
        # alignment_n_clusters=cfg.loss.alignment_n_clusters,
        # alignment_ema_momentum=cfg.loss.alignment_ema_momentum,
        # alignment_temperature=cfg.loss.alignment_temperature,
    )
    model.configure_optimizer(
        lr=cfg.optimizer.lr,
        warmup_steps=cfg.optimizer.warmup_steps,
        steady_1_steps=cfg.optimizer.steady_1_steps,
        cosine_anneal_steps=cfg.optimizer.cosine_anneal_steps,
        min_lr=cfg.optimizer.min_lr,
    )
    model.set_project(cfg.training.project_dir)
    model.train(
        train_steps=cfg.training.train_steps,
        eval_points=cfg.training.eval_points,
        save_points=cfg.training.save_points,
        device=cfg.training.device,
        config_copy_path=config_path,
    )


# ----------------------------------------------------------------------
# sub-commands
# ----------------------------------------------------------------------
def _cmd_train(args: argparse.Namespace) -> int:
    cfg = load_train_config(args.config)
    _logger.info("Loaded config task=%s project=%s", cfg.task, cfg.training.project_dir)

    if cfg.task == "single_pretrain":
        _train_single(cfg, args.config)
    elif cfg.task == "pair_scratch":
        _train_pair(cfg, args.config, pretrain=False)
    elif cfg.task == "pair_pretrain":
        _train_pair(cfg, args.config, pretrain=True)
    else:  # pragma: no cover - validated already
        raise ValueError(f"Unknown task {cfg.task!r}")

    _logger.info("Training finished.")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from solid_recover.evaluation.runner import evaluate_output_dir

    evaluate_output_dir(args.output_dir, device=args.device)
    return 0


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solid-recover",
        description="Solid Recover: single-cell multi-omics training / evaluation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a model from a YAML config")
    train_p.add_argument("--config", required=True, help="Path to YAML config file")
    train_p.set_defaults(func=_cmd_train)

    eval_p = sub.add_parser(
        "eval",
        help="Evaluate all checkpoints under an existing output directory",
    )
    eval_p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory produced by 'solid-recover train'",
    )
    eval_p.add_argument("--device", default="cpu", help="cpu or cuda (default cpu)")
    eval_p.set_defaults(func=_cmd_eval)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
