"""Inference-only evaluation on new TCGA target tables using pretrained fold checkpoints."""

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
from aacdr_pipeline.model_adapter import build_pipeline_models, load_checkpoint
from aacdr_pipeline.reports import ensure_dir, fold_dir, write_cross_fold_reports, write_csv
from aacdr_pipeline.run import _build_cancer_type_maps
from aacdr_pipeline.splits import FoldSplit, SourceSplits, build_grouped_source_splits
from aacdr_pipeline.target_eval import prepare_all_target_eval_datasets


def load_config_from_checkpoint_dir(checkpoint_dir: str | Path) -> AACDRPipelineConfig:
    path = Path(checkpoint_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return AACDRPipelineConfig(**{k: v for k, v in data.items() if k in AACDRPipelineConfig.__dataclass_fields__})


def load_source_splits_from_report(report_path: str | Path) -> SourceSplits:
    df = pd.read_csv(report_path)
    folds: list[FoldSplit] = []
    for fold_id in sorted(df["fold_id"].unique()):
        sub = df[df["fold_id"] == fold_id]
        folds.append(
            FoldSplit(
                fold=int(fold_id),
                train_sample_ids=sub.loc[sub["split"] == "train", "sample_id"].astype(str).tolist(),
                valid_sample_ids=sub.loc[sub["split"] == "val", "sample_id"].astype(str).tolist(),
                source_test_sample_ids=sub.loc[sub["split"] == "source_test", "sample_id"].astype(str).tolist(),
            )
        )
    fold_summary = pd.DataFrame(
        [
            {
                "fold": f.fold,
                "n_train_samples": len(f.train_sample_ids),
                "n_val_samples": len(f.valid_sample_ids),
                "n_source_test_samples": len(f.source_test_sample_ids),
            }
            for f in folds
        ]
    )
    return SourceSplits(folds=folds, source_split_report=df, fold_summary=fold_summary)


def run_target_inference(
    config: AACDRPipelineConfig,
    checkpoint_dir: str | Path,
    output_dir: str | Path | None = None,
    eval_prefixes: list[str] | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    out = ensure_dir(output_dir or config.output_dir)
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

    split_report = checkpoint_dir / "source_split.csv"
    if split_report.is_file():
        splits = load_source_splits_from_report(split_report)
    else:
        splits = build_grouped_source_splits(
            raw.source_response,
            sample_col=config.source_sample_col,
            n_splits=config.n_splits,
            source_test_size=config.source_test_size,
            seed=config.seed,
        )
    write_csv(splits.source_split_report, out / "source_split.csv")
    write_csv(splits.fold_summary, out / "fold_summary.csv")

    source_cancer_map, target_cancer_map = _build_cancer_type_maps(
        raw.ccle_cancer_info,
        raw.tcga_cancer_info,
        config.source_sample_col,
        config.target_omics_sample_col,
    )

    if eval_prefixes is None:
        eval_prefixes = ["target_primary", "target_only"]

    for fold_split in splits.folds:
        fdir = fold_dir(out, fold_split.fold)
        ckpt = checkpoint_dir / f"fold_{fold_split.fold}" / "best_model"
        for suffix in ("_fe.pt", "_dnn.pt", "_ae.pt"):
            if not Path(f"{ckpt}{suffix}").is_file():
                raise FileNotFoundError(f"Missing checkpoint file: {ckpt}{suffix}")

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

        models = build_pipeline_models(len(aligned.feature_names), device=config.device)
        load_checkpoint(models, str(ckpt))

        eval_results = evaluate_fold(
            models,
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
            if prefix not in eval_prefixes:
                continue
            write_csv(bundle["predictions"], fdir / f"{prefix}_prediction_results.csv")
            write_csv(bundle["per_drug"], fdir / f"{prefix}_metrics_per_drug.csv")
            write_csv(bundle["summary"], fdir / f"{prefix}_metrics_summary.csv")

    write_cross_fold_reports(out, eval_prefixes)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "inference_only",
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(out),
        "eval_prefixes": eval_prefixes,
        "n_folds": len(splits.folds),
        "n_features": len(aligned.feature_names),
        "n_drugs": len(drug_index.drug_ids),
    }
    with open(out / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out
