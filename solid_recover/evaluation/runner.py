"""Top-level evaluation entry point driven by :mod:`solid_recover.cli.main`.

Given an output directory produced by ``solid-recover train``, reconstruct the
:class:`PairScratch` model from its saved config, iterate over every
checkpoint in ``models/`` and run :func:`evaluate_checkpoint` for each.
"""

from __future__ import annotations

import json
import os
import re
from typing import List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

from solid_recover._logging import get_logger
from solid_recover.config.loader import load_train_config
from solid_recover.data.prepare import prepare_pair_test_only
from solid_recover.evaluation.reporting import evaluate_checkpoint
from solid_recover.models.pair import PairScratch

_logger = get_logger(__name__)

_CKPT_RE = re.compile(r"ckpt_(\d+)\.pth$")


def _discover_ckpt_steps(model_dir: str) -> List[int]:
    steps = []
    for name in os.listdir(model_dir):
        m = _CKPT_RE.match(name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def _plot_metrics_summary(output_dir: str, steps_list: List[int]) -> None:
    """Generate a summary line plot of all metrics across checkpoints."""
    eval_result_dir = os.path.join(output_dir, "eval_result")
    
    # Collect metrics from all checkpoints
    all_metrics = {}
    for steps in steps_list:
        metric_file = os.path.join(eval_result_dir, f"{steps}_result", "match_metric.json")
        if os.path.exists(metric_file):
            with open(metric_file, "r", encoding="utf-8") as f:
                all_metrics[steps] = json.load(f)
    
    if not all_metrics:
        _logger.warning("No metric files found for plotting.")
        return
    
    # Extract metrics
    steps_sorted = sorted(all_metrics.keys())
    
    # Collect all metric names (except ckpt_steps)
    metric_names = set()
    for metrics in all_metrics.values():
        for key in metrics.keys():
            if key != "ckpt_steps":
                metric_names.add(key)
    
    # Sort metrics: top_k_hit first (by k ascending), then other metrics alphabetically
    top_k_metrics = []
    other_metrics = []
    
    for metric_name in metric_names:
        if metric_name.startswith("top_") and metric_name.endswith("_hit"):
            # Extract k value
            try:
                k_value = int(metric_name.split("_")[1])
                top_k_metrics.append((k_value, metric_name))
            except (ValueError, IndexError):
                other_metrics.append(metric_name)
        else:
            other_metrics.append(metric_name)
    
    # Sort top_k by k value ascending
    top_k_metrics.sort(key=lambda x: x[0])
    other_metrics.sort()
    
    # Create ordered list of metrics to plot
    ordered_metrics = [name for _, name in top_k_metrics] + other_metrics
    
    # Build metrics data dict in ordered fashion
    metrics_to_plot = {}
    for metric_name in ordered_metrics:
        values = []
        for steps in steps_sorted:
            values.append(all_metrics[steps].get(metric_name, 0.0))
        metrics_to_plot[metric_name] = values
    
    # Create plot
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 4 * len(metrics_to_plot)), squeeze=False)
    fig.suptitle("Training Metrics Summary", fontsize=16, fontweight="bold")
    
    colors = plt.cm.tab10.colors
    for idx, (metric_name, values) in enumerate(metrics_to_plot.items()):
        ax = axes[idx, 0]
        color = colors[idx % len(colors)]
        
        ax.plot(steps_sorted, values, marker="o", linewidth=2, markersize=6, color=color)
        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
        ax.set_title(f"{metric_name.replace('_', ' ').title()} vs Training Steps", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        
        # Add value labels on points
        for i, (x, y) in enumerate(zip(steps_sorted, values)):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(eval_result_dir, "metrics_summary.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    _logger.info("Metrics summary plot saved to %s", plot_path)
    
    # Save summary data as JSON
    summary_data = {
        "steps": steps_sorted,
        "metrics": {}
    }
    
    for metric_name, values in metrics_to_plot.items():
        summary_data["metrics"][metric_name] = {
            "steps": steps_sorted,
            "values": values
        }
    
    summary_json_path = os.path.join(eval_result_dir, "metrics_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    
    _logger.info("Metrics summary data saved to %s", summary_json_path)


def evaluate_output_dir(output_dir: str, device: str = "cpu") -> None:
    """Evaluate every ``ckpt_<steps>.pth`` under ``output_dir/models``.

    Results are written to ``output_dir/eval_result/<steps>_result/``.
    """
    config_path = os.path.join(output_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path} not found")

    cfg = load_train_config(config_path)
    if cfg.task not in ("pair_scratch", "pair_pretrain"):
        raise ValueError(
            f"Only paired tasks can be evaluated via this runner; got task={cfg.task!r}"
        )

    # Test data only; there is no need to reload the training split.
    test_dataset = prepare_pair_test_only(
        test_data_path=cfg.data.test_data_path,  # type: ignore[arg-type]
        key_1=cfg.data.key_1,  # type: ignore[arg-type]
        key_2=cfg.data.key_2,  # type: ignore[arg-type]
        to_gpu=False,
    )

    feat_1 = cfg.model.feature_num_1 or test_dataset[0]["omic_1"].shape[0]
    feat_2 = cfg.model.feature_num_2 or test_dataset[0]["omic_2"].shape[0]

    model = PairScratch(
        feature_num_1=feat_1,
        feature_num_2=feat_2,
        hidden_params_1=cfg.model.hidden_params_1,  # type: ignore[arg-type]
        hidden_params_2=cfg.model.hidden_params_2,  # type: ignore[arg-type]
        embed_dim=cfg.model.embed_dim,
        use_rmsnorm=cfg.model.use_rmsnorm,
        use_residual=cfg.model.use_residual,
        dropout_p=cfg.model.dropout_p,
    )
    model.set_loss(
        vae_beta_1=cfg.loss.vae_beta_1,
        vae_beta_2=cfg.loss.vae_beta_2,
        clip_weight=cfg.loss.clip_weight,
        cross_recon_1=cfg.loss.cross_recon_1,
        cross_recon_2=cfg.loss.cross_recon_2,
        temperature=cfg.loss.temperature,
        use_weight=cfg.loss.use_weight,
    )

    model_dir = os.path.join(output_dir, "models")
    steps_list = _discover_ckpt_steps(model_dir)
    if not steps_list:
        _logger.warning("No ckpt_*.pth found under %s", model_dir)
        return

    for steps in tqdm(steps_list, desc="eval checkpoints"):
        eval_res_dir = os.path.join(output_dir, "eval_result", f"{steps}_result")
        if os.path.exists(os.path.join(eval_res_dir, "match_metric.json")):
            _logger.info("Skip ckpt %d (already evaluated)", steps)
            continue
        ckpt_path = os.path.join(model_dir, f"ckpt_{steps}.pth")
        model.load_state_dict(ckpt_path)
        evaluate_checkpoint(
            model=model,
            dataset=test_dataset,
            eval_res_dir=eval_res_dir,
            ckpt_steps=steps,
            device=device,
        )

    _logger.info("Finished evaluating %d checkpoints", len(steps_list))
    
    # Generate metrics summary plot
    _plot_metrics_summary(output_dir, steps_list)


__all__ = ["evaluate_output_dir"]
