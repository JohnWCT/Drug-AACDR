"""Load raw CSV inputs for the AACDR pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aacdr_pipeline.config import AACDRPipelineConfig
from aacdr_pipeline.schema import (
    normalize_source_response,
    normalize_target_response,
    validate_omics_schema,
)


@dataclass
class RawAACDRInputs:
    source_omics: pd.DataFrame
    source_response: pd.DataFrame
    target_omics: pd.DataFrame
    target_primary_response: pd.DataFrame
    target_only_response: pd.DataFrame
    target_auxiliary_response: pd.DataFrame
    drug_feature_table: pd.DataFrame
    ccle_cancer_info: pd.DataFrame | None
    tcga_cancer_info: pd.DataFrame | None


def load_csv_required(path: str, name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Cannot load {name} from {path}") from exc


def load_csv_optional(path: str | None, name: str) -> pd.DataFrame | None:
    if not path:
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        return None


def load_raw_inputs(config: AACDRPipelineConfig) -> RawAACDRInputs:
    source_omics = load_csv_required(config.source_omics_path, "source omics")
    validate_omics_schema(source_omics, config.source_sample_col, "source omics")

    target_omics = load_csv_required(config.target_omics_path, "target omics")
    validate_omics_schema(target_omics, config.target_omics_sample_col, "target omics")

    source_response_raw = load_csv_required(config.source_response_path, "source response")
    source_response = normalize_source_response(source_response_raw, config)

    primary_raw = load_csv_required(
        config.target_eval_primary_response_path, "target primary response"
    )
    target_only_raw = load_csv_required(
        config.target_eval_target_only_response_path, "target only response"
    )
    aux_raw = load_csv_required(
        config.target_eval_aux_response_path, "target auxiliary response"
    )

    drug_feature_table = load_csv_required(config.drug_smiles_path, "drug SMILES")

    return RawAACDRInputs(
        source_omics=source_omics,
        source_response=source_response,
        target_omics=target_omics,
        target_primary_response=normalize_target_response(primary_raw, "primary", config),
        target_only_response=normalize_target_response(target_only_raw, "target_only", config),
        target_auxiliary_response=normalize_target_response(aux_raw, "auxiliary", config),
        drug_feature_table=drug_feature_table,
        ccle_cancer_info=load_csv_optional(config.ccle_cancer_info_path, "ccle cancer info"),
        tcga_cancer_info=load_csv_optional(config.tcga_cancer_info_path, "tcga cancer info"),
    )
