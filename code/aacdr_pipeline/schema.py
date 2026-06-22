"""Schema validation and ID normalization."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from aacdr_pipeline.config import AACDRPipelineConfig

_TCGA_PREFIX = re.compile(r"^TCGA-", re.IGNORECASE)


def normalize_drug_name(value: Any) -> str:
    return str(value).strip().lower()


def normalize_tcga_patient_id(value: Any) -> str:
    sid = str(value).strip()
    parts = sid.split("-")
    if _TCGA_PREFIX.match(sid) and len(parts) >= 3:
        return "-".join(parts[:3])
    return sid


def normalize_source_sample_id(value: Any) -> str:
    return str(value).strip()


def validate_omics_schema(df: pd.DataFrame, sample_col: str, name: str) -> None:
    if sample_col not in df.columns:
        raise ValueError(f"{name}: missing sample column {sample_col!r}")
    if len(df) == 0:
        raise ValueError(f"{name}: empty omics table")


def validate_source_response_schema(
    df: pd.DataFrame, config: AACDRPipelineConfig
) -> None:
    required = [config.source_sample_col, config.source_drug_col, config.source_label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"source response missing columns: {missing}")


def validate_target_response_schema(
    df: pd.DataFrame, dataset_name: str, config: AACDRPipelineConfig
) -> None:
    required = [config.target_sample_col, config.target_drug_col, config.target_label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} response missing columns: {missing}")


def validate_binary_labels(df: pd.DataFrame, label_col: str, name: str) -> None:
    if label_col not in df.columns:
        raise ValueError(f"{name}: missing label column {label_col!r}")
    vals = pd.to_numeric(df[label_col], errors="coerce")
    bad = vals.dropna()
    if len(bad) and not set(bad.unique()).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"{name}: labels must be binary 0/1")


def normalize_source_response(df: pd.DataFrame, config: AACDRPipelineConfig) -> pd.DataFrame:
    validate_source_response_schema(df, config)
    validate_binary_labels(df, config.source_label_col, "source response")
    out = df.copy()
    out[config.source_sample_col] = out[config.source_sample_col].map(
        normalize_source_sample_id
    )
    out["drug_id"] = out[config.source_drug_col].map(normalize_drug_name)
    out["label"] = pd.to_numeric(out[config.source_label_col], errors="coerce").astype(int)
    return out


def normalize_target_response(
    df: pd.DataFrame, dataset_name: str, config: AACDRPipelineConfig
) -> pd.DataFrame:
    validate_target_response_schema(df, dataset_name, config)
    validate_binary_labels(df, config.target_label_col, dataset_name)
    out = df.copy()
    out["sample_id"] = out[config.target_sample_col].map(normalize_tcga_patient_id)
    out["drug_id"] = out[config.target_drug_col].map(normalize_drug_name)
    out["label"] = pd.to_numeric(out[config.target_label_col], errors="coerce").astype(int)
    if "cancers" in out.columns:
        out["cancer_type"] = out["cancers"].astype(str)
    elif "cancer_type" not in out.columns:
        out["cancer_type"] = np.nan
    return out
