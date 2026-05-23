"""Evaluation utilities for Solid Recover (metrics / embeddings / prediction / reporting)."""

from solid_recover.evaluation.embeddings import get_paired_embedding
from solid_recover.evaluation.metrics import (
    calculate_hit_rate,
    calculate_hit_rates_batch,
    matching_metrics,
    pearson_corr_columns,
)
from solid_recover.evaluation.prediction import (
    predict_cross_modality,
    predictions_to_anndata,
)
from solid_recover.evaluation.reporting import DEFAULT_HIT_KS, evaluate_checkpoint
from solid_recover.evaluation.runner import evaluate_output_dir

__all__ = [
    "matching_metrics",
    "calculate_hit_rate",
    "calculate_hit_rates_batch",
    "pearson_corr_columns",
    "get_paired_embedding",
    "predict_cross_modality",
    "predictions_to_anndata",
    "evaluate_checkpoint",
    "evaluate_output_dir",
    "DEFAULT_HIT_KS",
]
