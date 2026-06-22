"""Build AACDR LabeledDataset / UnlabeledDataset from standardized tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from aacdr_pipeline.drug_index import DrugIndex
from aacdr_pipeline.splits import FoldSplit
from dataset import LabeledDataset, UnlabeledDataset


@dataclass
class FoldDataBundle:
    fold: int
    source_train_dataset: LabeledDataset
    source_valid_dataset: LabeledDataset
    source_test_dataset: LabeledDataset
    target_unlabeled_dataset: UnlabeledDataset
    target_eval_primary_rows: pd.DataFrame
    target_eval_target_only_rows: pd.DataFrame
    target_eval_auxiliary_rows: pd.DataFrame
    source_test_rows: pd.DataFrame
    n_features: int


def _omics_vector(
    omics_df: pd.DataFrame, sample_id: str, sample_col: str, feature_names: list[str]
) -> torch.Tensor:
    row = omics_df.loc[omics_df[sample_col] == sample_id, feature_names]
    if row.empty:
        raise KeyError(f"Sample {sample_id} not in omics")
    return torch.tensor(row.iloc[0].astype(float).values, dtype=torch.float32)


def _build_labeled_rows(
    response: pd.DataFrame,
    omics_df: pd.DataFrame,
    sample_col: str,
    feature_names: list[str],
    drug_index: DrugIndex,
    allowed_samples: set[str] | None = None,
    max_rows: int | None = None,
) -> list[list[Any]]:
    data = []
    for _, row in response.iterrows():
        sid = str(row.get(sample_col, row.get("sample_id", "")))
        if allowed_samples is not None and sid not in allowed_samples:
            continue
        did = str(row["drug_id"])
        if did not in drug_index.drug_to_index:
            continue
        if sid not in set(omics_df[sample_col].astype(str)):
            continue
        graph_id = drug_index.drug_to_index[did]
        expr = _omics_vector(omics_df, sid, sample_col, feature_names)
        label = torch.tensor(float(row["label"]), dtype=torch.float32).view(1)
        data.append([graph_id, expr, label])
        if max_rows is not None and len(data) >= max_rows:
            break
    return data


def build_fold_data_bundle(
    fold_split: FoldSplit,
    source_response: pd.DataFrame,
    source_omics: pd.DataFrame,
    target_omics: pd.DataFrame,
    target_eval_primary: pd.DataFrame,
    target_eval_target_only: pd.DataFrame,
    target_eval_auxiliary: pd.DataFrame,
    drug_index: DrugIndex,
    source_sample_col: str,
    target_omics_sample_col: str,
    feature_names: list[str],
    max_train_rows: int | None = None,
) -> FoldDataBundle:
    train_set = set(fold_split.train_sample_ids)
    val_set = set(fold_split.valid_sample_ids)
    test_set = set(fold_split.source_test_sample_ids)

    train_rows = _build_labeled_rows(
        source_response,
        source_omics,
        source_sample_col,
        feature_names,
        drug_index,
        allowed_samples=train_set,
        max_rows=max_train_rows,
    )
    val_rows = _build_labeled_rows(
        source_response,
        source_omics,
        source_sample_col,
        feature_names,
        drug_index,
        allowed_samples=val_set,
    )
    test_rows_data = _build_labeled_rows(
        source_response,
        source_omics,
        source_sample_col,
        feature_names,
        drug_index,
        allowed_samples=test_set,
    )

    unlabeled = []
    for sid in target_omics[target_omics_sample_col].astype(str).tolist():
        expr = _omics_vector(target_omics, sid, target_omics_sample_col, feature_names)
        unlabeled.append(expr)

    source_test_df = source_response[
        source_response[source_sample_col].isin(test_set)
    ].copy()

    return FoldDataBundle(
        fold=fold_split.fold,
        source_train_dataset=LabeledDataset(train_rows),
        source_valid_dataset=LabeledDataset(val_rows),
        source_test_dataset=LabeledDataset(test_rows_data),
        target_unlabeled_dataset=UnlabeledDataset(unlabeled),
        target_eval_primary_rows=target_eval_primary,
        target_eval_target_only_rows=target_eval_target_only,
        target_eval_auxiliary_rows=target_eval_auxiliary,
        source_test_rows=source_test_df,
        n_features=len(feature_names),
    )
