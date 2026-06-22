"""Tests for schema normalization and validation."""

import pandas as pd
import pytest

from aacdr_pipeline.config import AACDRPipelineConfig
from aacdr_pipeline.schema import (
    normalize_drug_name,
    normalize_source_response,
    normalize_target_response,
    normalize_tcga_patient_id,
    validate_binary_labels,
)


def test_normalize_drug_name():
    assert normalize_drug_name(" Paclitaxel ") == "paclitaxel"


def test_normalize_tcga_patient_id():
    assert normalize_tcga_patient_id("TCGA-G2-A2EJ-01") == "TCGA-G2-A2EJ"
    assert normalize_tcga_patient_id("TCGA-G2-A2EJ") == "TCGA-G2-A2EJ"


def test_validate_binary_labels_ok():
    df = pd.DataFrame({"Label": [0, 1, 0]})
    validate_binary_labels(df, "Label", "test")


def test_normalize_source_response():
    cfg = AACDRPipelineConfig()
    df = pd.DataFrame(
        {"Sample_ID": ["A1"], "drug_name": ["Paclitaxel"], "Label": [1]}
    )
    out = normalize_source_response(df, cfg)
    assert out["drug_id"].iloc[0] == "paclitaxel"
    assert out["label"].iloc[0] == 1


def test_normalize_target_response():
    cfg = AACDRPipelineConfig()
    df = pd.DataFrame(
        {"Patient_id": ["TCGA-XX-0001-01"], "drug_name": ["Cisplatin"], "Label": [0]}
    )
    out = normalize_target_response(df, "primary", cfg)
    assert out["sample_id"].iloc[0] == "TCGA-XX-0001"
    assert out["drug_id"].iloc[0] == "cisplatin"
