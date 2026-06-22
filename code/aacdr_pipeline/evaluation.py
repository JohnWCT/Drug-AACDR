"""Prediction tables and metrics for source test + 3 TCGA eval sets."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from aacdr_pipeline.config import AACDRPipelineConfig
from aacdr_pipeline.datasets import FoldDataBundle
from aacdr_pipeline.drug_graph_adapter import DrugRepresentationBundle
from aacdr_pipeline.drug_index import DrugMetadata
from aacdr_pipeline.model_adapter import PipelineModels, predict_labeled_dataset

EVAL_SPECS = {
    "source_test": {
        "domain": "source",
        "split": "source_test",
        "eval_dataset": "source_test",
        "file_prefix": "source_test",
    },
    "primary": {
        "domain": "target",
        "split": "target_eval",
        "eval_dataset": "primary",
        "file_prefix": "target_primary",
    },
    "target_only": {
        "domain": "target",
        "split": "target_eval",
        "eval_dataset": "target_only",
        "file_prefix": "target_only",
    },
    "auxiliary": {
        "domain": "target",
        "split": "target_eval",
        "eval_dataset": "auxiliary",
        "file_prefix": "target_auxiliary",
    },
}


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan")


def _safe_auprc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y, scores))
    except ValueError:
        return float("nan")


def _drug_flags(drug_metadata: DrugMetadata) -> pd.DataFrame:
    return drug_metadata.drug_list[
        ["drug_id", "drug_index", "has_supervised_source_label", "is_target_eval_only"]
    ]


def build_prediction_table_from_dataset(
    models: PipelineModels,
    dataset,
    drug_graph: dict,
    sample_ids: list[str],
    drug_metadata: DrugMetadata,
    spec: dict,
    fold: int,
    seed: int,
    cancer_type_map: dict[str, str] | None = None,
    task_type: str = "classification",
) -> pd.DataFrame:
    gids, labels, probs = predict_labeled_dataset(models, dataset, drug_graph)
    flags = _drug_flags(drug_metadata).set_index("drug_index")
    rows = []
    for i, (gid, label, prob) in enumerate(zip(gids, labels, probs)):
        drug_index = int(gid)
        drug_id = drug_metadata.drug_index.index_to_drug[drug_index]
        sid = sample_ids[i] if i < len(sample_ids) else f"row_{i}"
        pred_label = int(prob >= 0.5)
        confidence = float(abs(prob - 0.5) * 2)
        row = {
            "sample_id": sid,
            "drug_id": drug_id,
            "drug_index": drug_index,
            "domain": spec["domain"],
            "split": spec["split"],
            "eval_dataset": spec["eval_dataset"],
            "ground_truth": int(label),
            "pred_score": float(prob),
            "probability": float(prob),
            "pred_label": pred_label,
            "confidence": confidence,
            "fold": fold,
            "seed": seed,
            "task_type": task_type,
            "cancer_type": cancer_type_map.get(sid, "") if cancer_type_map else "",
            "has_supervised_source_label": bool(flags.loc[drug_index, "has_supervised_source_label"]),
            "is_target_eval_only": bool(flags.loc[drug_index, "is_target_eval_only"]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _dataset_sample_drug_order(dataset) -> tuple[list[str], list[str]]:
    """Recover sample/drug order from LabeledDataset rows (stored externally)."""
    return [], []


def build_prediction_table_from_response(
    models: PipelineModels,
    response_df: pd.DataFrame,
    omics_df: pd.DataFrame,
    omics_sample_col: str,
    feature_names: list[str],
    drug_metadata: DrugMetadata,
    drug_graph: dict,
    spec: dict,
    fold: int,
    seed: int,
    cancer_type_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    from aacdr_pipeline.datasets import _build_labeled_rows, _omics_vector
    from dataset import LabeledDataset

    rows_data = []
    sample_ids = []
    for _, row in response_df.iterrows():
        sid = str(row["sample_id"])
        did = str(row["drug_id"])
        if did not in drug_metadata.drug_index.drug_to_index:
            continue
        if sid not in set(omics_df[omics_sample_col].astype(str)):
            continue
        graph_id = drug_metadata.drug_index.drug_to_index[did]
        expr = _omics_vector(omics_df, sid, omics_sample_col, feature_names)
        label = float(row["label"])
        rows_data.append([graph_id, expr, __import__("torch").tensor([label])])
        sample_ids.append(sid)

    if not rows_data:
        return pd.DataFrame()

    ds = LabeledDataset(rows_data)
    gids, labels, probs = predict_labeled_dataset(models, ds, drug_graph)
    flags = _drug_flags(drug_metadata).set_index("drug_index")
    out = []
    for i, (gid, label, prob) in enumerate(zip(gids, labels, probs)):
        drug_index = int(gid)
        drug_id = drug_metadata.drug_index.index_to_drug[drug_index]
        sid = sample_ids[i]
        out.append(
            {
                "sample_id": sid,
                "drug_id": drug_id,
                "drug_index": drug_index,
                "domain": spec["domain"],
                "split": spec["split"],
                "eval_dataset": spec["eval_dataset"],
                "ground_truth": int(label),
                "pred_score": float(prob),
                "probability": float(prob),
                "pred_label": int(prob >= 0.5),
                "confidence": float(abs(prob - 0.5) * 2),
                "fold": fold,
                "seed": seed,
                "task_type": "classification",
                "cancer_type": cancer_type_map.get(sid, "") if cancer_type_map else "",
                "has_supervised_source_label": bool(
                    flags.loc[drug_index, "has_supervised_source_label"]
                ),
                "is_target_eval_only": bool(flags.loc[drug_index, "is_target_eval_only"]),
            }
        )
    return pd.DataFrame(out)


def compute_metrics_per_drug(pred_df: pd.DataFrame, fold: int, seed: int) -> pd.DataFrame:
    rows = []
    for drug_id, g in pred_df.groupby("drug_id"):
        y = g["ground_truth"].astype(int).to_numpy()
        scores = g["probability"].astype(float).to_numpy()
        pred = g["pred_label"].astype(int).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = {
                "eval_dataset": g["eval_dataset"].iloc[0],
                "drug_id": drug_id,
                "drug_index": int(g["drug_index"].iloc[0]),
                "n_samples": len(g),
                "n_positive": int((y == 1).sum()),
                "n_negative": int((y == 0).sum()),
                "auroc": _safe_auc(y, scores),
                "auprc": _safe_auprc(y, scores),
                "accuracy": float(accuracy_score(y, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "fold": fold,
                "seed": seed,
                "has_supervised_source_label": bool(g["has_supervised_source_label"].iloc[0]),
                "is_target_eval_only": bool(g["is_target_eval_only"].iloc[0]),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def compute_metrics_summary(
    per_drug_df: pd.DataFrame, pred_df: pd.DataFrame, fold: int, seed: int
) -> pd.DataFrame:
    eval_dataset = pred_df["eval_dataset"].iloc[0] if len(pred_df) else ""
    n_drugs = per_drug_df["drug_id"].nunique() if len(per_drug_df) else 0
    n_obs = len(pred_df)

    def macro(col: str) -> float:
        vals = per_drug_df[col].dropna()
        return float(vals.mean()) if len(vals) else float("nan")

    def weighted(col: str) -> float:
        if per_drug_df.empty:
            return float("nan")
        w = per_drug_df["n_samples"].values
        v = per_drug_df[col].values.astype(float)
        mask = ~np.isnan(v)
        return float(np.average(v[mask], weights=w[mask])) if mask.any() else float("nan")

    y_all = pred_df["ground_truth"].astype(int).to_numpy() if len(pred_df) else np.array([])
    p_all = pred_df["pred_label"].astype(int).to_numpy() if len(pred_df) else np.array([])
    overall_acc = float(accuracy_score(y_all, p_all)) if len(y_all) else float("nan")

    return pd.DataFrame(
        [
            {
                "eval_dataset": eval_dataset,
                "n_drugs": n_drugs,
                "n_observed_pairs": n_obs,
                "macro_auroc": macro("auroc"),
                "macro_auprc": macro("auprc"),
                "macro_accuracy": macro("accuracy"),
                "macro_balanced_accuracy": macro("balanced_accuracy"),
                "macro_f1": macro("f1"),
                "weighted_auroc": weighted("auroc"),
                "weighted_auprc": weighted("auprc"),
                "weighted_accuracy": weighted("accuracy"),
                "weighted_balanced_accuracy": weighted("balanced_accuracy"),
                "weighted_f1": weighted("f1"),
                "overall_accuracy": overall_acc,
                "fold": fold,
                "seed": seed,
            }
        ]
    )


def evaluate_fold(
    models: PipelineModels,
    fold_data: FoldDataBundle,
    drug_bundle: DrugRepresentationBundle,
    drug_metadata: DrugMetadata,
    source_omics: pd.DataFrame,
    target_omics: pd.DataFrame,
    source_sample_col: str,
    target_omics_sample_col: str,
    feature_names: list[str],
    config: AACDRPipelineConfig,
    cancer_type_maps: dict[str, dict[str, str]],
) -> dict[str, dict[str, pd.DataFrame]]:
    results: dict[str, dict[str, pd.DataFrame]] = {}
    seed = config.seed
    fold = fold_data.fold

    src_resp = fold_data.source_test_rows.copy()
    if "sample_id" not in src_resp.columns:
        src_resp["sample_id"] = src_resp[source_sample_col].astype(str)
    src_pred = build_prediction_table_from_response(
        models,
        src_resp,
        source_omics,
        source_sample_col,
        feature_names,
        drug_metadata,
        drug_bundle.gdsc_drug_graph,
        EVAL_SPECS["source_test"],
        fold,
        seed,
        cancer_type_maps.get("source"),
    )

    eval_jobs = [
        ("source_test", src_pred),
        (
            "primary",
            build_prediction_table_from_response(
                models,
                fold_data.target_eval_primary_rows,
                target_omics,
                target_omics_sample_col,
                feature_names,
                drug_metadata,
                drug_bundle.tcga_drug_graph,
                EVAL_SPECS["primary"],
                fold,
                seed,
                cancer_type_maps.get("target"),
            ),
        ),
        (
            "target_only",
            build_prediction_table_from_response(
                models,
                fold_data.target_eval_target_only_rows,
                target_omics,
                target_omics_sample_col,
                feature_names,
                drug_metadata,
                drug_bundle.tcga_drug_graph,
                EVAL_SPECS["target_only"],
                fold,
                seed,
                cancer_type_maps.get("target"),
            ),
        ),
        (
            "auxiliary",
            build_prediction_table_from_response(
                models,
                fold_data.target_eval_auxiliary_rows,
                target_omics,
                target_omics_sample_col,
                feature_names,
                drug_metadata,
                drug_bundle.tcga_drug_graph,
                EVAL_SPECS["auxiliary"],
                fold,
                seed,
                cancer_type_maps.get("target"),
            ),
        ),
    ]

    for name, pred_df in eval_jobs:
        prefix = EVAL_SPECS[name]["file_prefix"]
        per_drug = compute_metrics_per_drug(pred_df, fold, seed) if len(pred_df) else pd.DataFrame()
        summary = compute_metrics_summary(per_drug, pred_df, fold, seed) if len(pred_df) else pd.DataFrame()
        results[prefix] = {
            "predictions": pred_df,
            "per_drug": per_drug,
            "summary": summary,
        }
    return results


def aggregate_metrics_across_folds(
    fold_frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not fold_frames:
        return pd.DataFrame(), pd.DataFrame()
    across = pd.concat(fold_frames, ignore_index=True)
    metric_cols = [
        c
        for c in across.columns
        if c
        not in {
            "drug_id",
            "drug_index",
            "eval_dataset",
            "fold",
            "seed",
            "has_supervised_source_label",
            "is_target_eval_only",
        }
        and across[c].dtype != object
    ]
    group_cols = ["eval_dataset"] if "eval_dataset" in across.columns else []
    if "drug_id" in across.columns:
        group_cols.append("drug_id")
    mean_std_rows = []
    for keys, g in across.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in metric_cols:
            vals = g[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
        mean_std_rows.append(row)
    return across, pd.DataFrame(mean_std_rows)
