"""Tests for metrics computation including single-class drugs."""

import numpy as np
import pandas as pd

from aacdr_pipeline.evaluation import compute_metrics_per_drug, compute_metrics_summary


def _pred_df():
    return pd.DataFrame(
        {
            "eval_dataset": ["primary"] * 4,
            "drug_id": ["a", "a", "b", "b"],
            "drug_index": [0, 0, 1, 1],
            "ground_truth": [0, 1, 1, 1],
            "probability": [0.2, 0.8, 0.9, 0.7],
            "pred_label": [0, 1, 1, 1],
            "has_supervised_source_label": [True, True, False, False],
            "is_target_eval_only": [False, False, True, True],
        }
    )


def test_per_drug_metrics_single_class_nan():
    df = pd.DataFrame(
        {
            "eval_dataset": ["primary"] * 3,
            "drug_id": ["only_pos"] * 3,
            "drug_index": [0] * 3,
            "ground_truth": [1, 1, 1],
            "probability": [0.9, 0.8, 0.7],
            "pred_label": [1, 1, 1],
            "has_supervised_source_label": [False] * 3,
            "is_target_eval_only": [True] * 3,
        }
    )
    per = compute_metrics_per_drug(df, fold=0, seed=0)
    assert np.isnan(per["auroc"].iloc[0])
    assert np.isnan(per["auprc"].iloc[0])


def test_summary_schema():
    pred = _pred_df()
    per = compute_metrics_per_drug(pred, fold=0, seed=0)
    summary = compute_metrics_summary(per, pred, fold=0, seed=0)
    assert "macro_auroc" in summary.columns
    assert summary["eval_dataset"].iloc[0] == "primary"
