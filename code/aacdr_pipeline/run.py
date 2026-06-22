"""Pipeline orchestrator – coordinates data prep, training, eval, and reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from aacdr_pipeline.config import AACDRPipelineConfig, save_config, validate_config
from aacdr_pipeline.data_io import load_raw_inputs
from aacdr_pipeline.datasets import build_fold_data_bundle
from aacdr_pipeline.drug_graph_adapter import build_drug_representations
from aacdr_pipeline.drug_index import build_drug_metadata, build_final_drug_index
from aacdr_pipeline.evaluation import EVAL_SPECS, evaluate_fold
from aacdr_pipeline.features import align_source_target_features
from aacdr_pipeline.fid import compute_latent_distribution_metrics
from aacdr_pipeline.kmeans import compute_kmeans_cancer_type_metrics
from aacdr_pipeline.latent import export_fold_latents, save_latent_pickle
from aacdr_pipeline.reports import (
    ensure_dir,
    fold_dir,
    write_cross_fold_reports,
    write_csv,
    write_json,
)
from aacdr_pipeline.schema import normalize_tcga_patient_id
from aacdr_pipeline.splits import build_grouped_source_splits, validate_no_sample_id_leakage
from aacdr_pipeline.target_eval import prepare_all_target_eval_datasets
from aacdr_pipeline.trainer_wrapper import train_one_fold
from aacdr_pipeline.tsne import run_tsne_plots


def _build_cancer_type_maps(
    ccle_info: pd.DataFrame | None,
    tcga_info: pd.DataFrame | None,
    source_sample_col: str,
    target_omics_sample_col: str,
) -> tuple[dict[str, str], dict[str, str]]:
    source_map: dict[str, str] = {}
    target_map: dict[str, str] = {}

    if ccle_info is not None and "cancer_type" in ccle_info.columns:
        id_col = ccle_info.columns[0]
        for _, row in ccle_info.iterrows():
            sid = str(row[id_col]).strip()
            source_map[sid] = str(row["cancer_type"])

    if tcga_info is not None and "cancer_type" in tcga_info.columns:
        id_col = tcga_info.columns[0]
        for _, row in tcga_info.iterrows():
            raw = str(row[id_col]).strip()
            key = normalize_tcga_patient_id(raw)
            target_map[key] = str(row["cancer_type"])

    return source_map, target_map


def _cancer_type_summary(
    source_map: dict[str, str],
    target_map: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for domain, cmap in [("source", source_map), ("target", target_map)]:
        for sid, ct in cmap.items():
            rows.append({"domain": domain, "sample_id": sid, "cancer_type": ct})
    return pd.DataFrame(rows)


def run_pipeline(config: AACDRPipelineConfig) -> None:
    validate_config(config)
    out = ensure_dir(config.output_dir)
    save_config(config, out / "config.json")

    raw = load_raw_inputs(config)
    aligned = align_source_target_features(
        raw.source_omics,
        raw.target_omics,
        config.source_sample_col,
        config.target_omics_sample_col,
        max_target_omics_samples=config.max_target_omics_samples,
    )
    write_csv(aligned.report, out / "feature_alignment_report.csv")
    write_csv(aligned.sample_filtering_report, out / "sample_filtering_report.csv")

    drug_index = build_final_drug_index(
        raw.source_response,
        raw.target_primary_response,
        raw.target_only_response,
        raw.target_auxiliary_response,
    )
    drug_metadata = build_drug_metadata(
        drug_index,
        raw.source_response,
        raw.target_primary_response,
        raw.target_only_response,
        raw.target_auxiliary_response,
        raw.drug_feature_table,
    )
    write_csv(drug_metadata.drug_list, out / "drug_list.csv")
    write_csv(drug_metadata.drug_availability_report, out / "drug_availability_report.csv")
    write_csv(drug_metadata.zero_shot_drug_report, out / "target_eval_zero_shot_drug_report.csv")

    drug_bundle = build_drug_representations(
        drug_index,
        raw.drug_feature_table,
        aacdr_data_root=config.aacdr_data_root,
    )
    write_csv(drug_bundle.availability_report, out / "drug_graph_availability_report.csv")
    write_csv(drug_bundle.edge_report, out / "drug_graph_edge_report.csv")

    target_eval = prepare_all_target_eval_datasets(
        raw.target_primary_response,
        raw.target_only_response,
        raw.target_auxiliary_response,
        aligned.target_omics,
        config.target_omics_sample_col,
        drug_index,
    )
    write_csv(target_eval.report, out / "target_eval_dataset_report.csv")

    splits = build_grouped_source_splits(
        raw.source_response,
        sample_col=config.source_sample_col,
        n_splits=config.n_splits,
        source_test_size=config.source_test_size,
        seed=config.seed,
    )
    validate_no_sample_id_leakage(splits)
    write_csv(splits.source_split_report, out / "source_split.csv")
    write_csv(splits.fold_summary, out / "fold_summary.csv")

    source_cancer_map, target_cancer_map = _build_cancer_type_maps(
        raw.ccle_cancer_info,
        raw.tcga_cancer_info,
        config.source_sample_col,
        config.target_omics_sample_col,
    )
    write_csv(_cancer_type_summary(source_cancer_map, target_cancer_map), out / "cancer_type_summary.csv")

    alignment_rows = [
        {
            "stage": "source_response",
            "n_rows": len(raw.source_response),
            "n_samples": raw.source_response[config.source_sample_col].nunique(),
            "n_drugs": raw.source_response["drug_id"].nunique(),
        },
        {
            "stage": "aligned_omics",
            "n_source_samples": len(aligned.source_omics),
            "n_target_samples": len(aligned.target_omics),
            "n_features": len(aligned.feature_names),
        },
    ]
    write_csv(pd.DataFrame(alignment_rows), out / "data_alignment_report.csv")

    fold_summaries = []
    latent_metric_frames = []
    kmeans_frames = []

    for fold_split in splits.folds:
        fdir = fold_dir(out, fold_split.fold)
        ensure_dir(fdir)

        fold_data = build_fold_data_bundle(
            fold_split,
            raw.source_response,
            aligned.source_omics,
            aligned.target_omics,
            target_eval.primary.response_long,
            target_eval.target_only.response_long,
            target_eval.auxiliary.response_long,
            drug_index,
            config.source_sample_col,
            config.target_omics_sample_col,
            aligned.feature_names,
            max_train_rows=config.max_train_rows,
        )

        ckpt_prefix = str(fdir / "best_model")
        train_result = train_one_fold(fold_data, drug_bundle, config, ckpt_prefix)
        fold_summaries.append(
            {
                "fold": fold_split.fold,
                "best_epoch": train_result.best_epoch,
                "best_val_auroc": train_result.best_val_auroc,
                "n_train_pairs": len(fold_data.source_train_dataset),
                "n_val_pairs": len(fold_data.source_valid_dataset),
            }
        )

        eval_results = evaluate_fold(
            train_result.models,
            fold_data,
            drug_bundle,
            drug_metadata,
            aligned.source_omics,
            aligned.target_omics,
            config.source_sample_col,
            config.target_omics_sample_col,
            aligned.feature_names,
            config,
            {"source": source_cancer_map, "target": target_cancer_map},
        )

        for prefix, bundle in eval_results.items():
            write_csv(bundle["predictions"], fdir / f"{prefix}_prediction_results.csv")
            write_csv(bundle["per_drug"], fdir / f"{prefix}_metrics_per_drug.csv")
            write_csv(bundle["summary"], fdir / f"{prefix}_metrics_summary.csv")

        src_latent, tgt_latent = export_fold_latents(
            train_result.models,
            aligned.source_omics,
            aligned.target_omics,
            config.source_sample_col,
            config.target_omics_sample_col,
            aligned.feature_names,
            fold_split.fold,
            config.seed,
            source_cancer_map,
            target_cancer_map,
        )
        save_latent_pickle(src_latent, fdir / "source_latent_representation.pkl")
        save_latent_pickle(tgt_latent, fdir / "target_latent_representation.pkl")

        combined_latent = {**src_latent, **tgt_latent}
        if config.run_fid:
            latent_df = compute_latent_distribution_metrics(
                src_latent, tgt_latent, fold_split.fold, config.seed
            )
            write_csv(latent_df, fdir / "latent_distribution_metrics.csv")
            latent_metric_frames.append(latent_df)

        if config.run_kmeans:
            km_df = compute_kmeans_cancer_type_metrics(
                combined_latent, fold_split.fold, config.seed
            )
            write_csv(km_df, fdir / "kmeans_cancer_type_metrics.csv")
            kmeans_frames.append(km_df)

        if config.run_tsne:
            run_tsne_plots(src_latent, tgt_latent, fdir, fold_split.fold, config.seed)

    write_csv(pd.DataFrame(fold_summaries), out / "fold_summary.csv")

    prefixes = [EVAL_SPECS[k]["file_prefix"] for k in EVAL_SPECS]
    write_cross_fold_reports(out, prefixes)

    if latent_metric_frames:
        latent_all = pd.concat(latent_metric_frames, ignore_index=True)
        write_csv(latent_all, out / "latent_metrics_summary.csv")

    if kmeans_frames:
        km_all = pd.concat(kmeans_frames, ignore_index=True)
        write_csv(km_all, out / "kmeans_cancer_type_summary.csv")
        metric_cols = ["ari", "nmi", "silhouette", "calinski_harabasz", "davies_bouldin"]
        rows = []
        for col in metric_cols:
            vals = km_all[col].dropna()
            rows.append(
                {
                    "metric": col,
                    "mean": float(vals.mean()) if len(vals) else float("nan"),
                    "std": float(vals.std(ddof=0)) if len(vals) > 1 else 0.0,
                    "n_folds": len(vals),
                }
            )
        write_csv(pd.DataFrame(rows), out / "kmeans_cancer_type_fold_mean_std.csv")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out),
        "n_folds": config.n_splits,
        "seed": config.seed,
        "n_drugs": len(drug_index.drug_ids),
        "n_features": len(aligned.feature_names),
        "tcga_labels_in_training": False,
    }
    write_json(manifest, out / "run_manifest.json")
