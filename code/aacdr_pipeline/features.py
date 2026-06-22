"""Align source/target omics feature columns."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aacdr_pipeline.schema import normalize_tcga_patient_id


@dataclass
class AlignedOmics:
    source_omics: pd.DataFrame
    target_omics: pd.DataFrame
    feature_names: list[str]
    report: pd.DataFrame
    sample_filtering_report: pd.DataFrame


def _feature_columns(df: pd.DataFrame, sample_col: str) -> list[str]:
    return [c for c in df.columns if c != sample_col]


def align_source_target_features(
    source_omics: pd.DataFrame,
    target_omics: pd.DataFrame,
    source_sample_col: str,
    target_sample_col: str,
    max_target_omics_samples: int | None = None,
) -> AlignedOmics:
    src_feats = set(_feature_columns(source_omics, source_sample_col))
    tgt_feats = set(_feature_columns(target_omics, target_sample_col))
    common = sorted(src_feats & tgt_feats)
    src_only = sorted(src_feats - tgt_feats)
    tgt_only = sorted(tgt_feats - src_feats)

    report = pd.DataFrame(
        [
            {"metric": "n_source_features", "value": len(src_feats)},
            {"metric": "n_target_features", "value": len(tgt_feats)},
            {"metric": "n_common_features", "value": len(common)},
            {"metric": "n_source_only_features", "value": len(src_only)},
            {"metric": "n_target_only_features", "value": len(tgt_only)},
        ]
    )
    if not common:
        raise ValueError("No common features between source and target omics")

    src = source_omics[[source_sample_col] + common].copy()
    src[source_sample_col] = src[source_sample_col].astype(str).str.strip()
    src = src.drop_duplicates(subset=[source_sample_col], keep="first")

    tgt = target_omics[[target_sample_col] + common].copy()
    tgt[target_sample_col] = tgt[target_sample_col].map(normalize_tcga_patient_id)
    tgt = tgt.drop_duplicates(subset=[target_sample_col], keep="first")
    if max_target_omics_samples is not None:
        tgt = tgt.head(max_target_omics_samples)

    filter_rows = [
        {
            "domain": "source",
            "n_input_samples": len(source_omics),
            "n_kept_samples": len(src),
            "n_dropped_duplicates": len(source_omics) - len(src),
        },
        {
            "domain": "target",
            "n_input_samples": len(target_omics),
            "n_kept_samples": len(tgt),
            "n_dropped_duplicates": len(target_omics) - len(tgt),
        },
    ]
    return AlignedOmics(
        source_omics=src,
        target_omics=tgt,
        feature_names=common,
        report=report,
        sample_filtering_report=pd.DataFrame(filter_rows),
    )
