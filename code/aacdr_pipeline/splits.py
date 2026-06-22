"""Grouped source split by Sample_ID."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


@dataclass
class FoldSplit:
    fold: int
    train_sample_ids: list[str]
    valid_sample_ids: list[str]
    source_test_sample_ids: list[str]


@dataclass
class SourceSplits:
    folds: list[FoldSplit]
    source_split_report: pd.DataFrame
    fold_summary: pd.DataFrame


def _pseudo_binary_labels(
    response: pd.DataFrame, sample_ids: list[str], sample_col: str
) -> np.ndarray:
    pseudo = np.zeros(len(sample_ids), dtype=int)
    sid_to_labels = response.groupby(sample_col)["label"].mean()
    for i, sid in enumerate(sample_ids):
        if sid in sid_to_labels.index:
            pseudo[i] = int(sid_to_labels.loc[sid] >= 0.5)
    return pseudo


def build_grouped_source_splits(
    source_response: pd.DataFrame,
    sample_col: str = "Sample_ID",
    n_splits: int = 5,
    source_test_size: float = 0.10,
    seed: int = 0,
) -> SourceSplits:
    unique_samples = sorted(source_response[sample_col].astype(str).unique().tolist())
    n = len(unique_samples)
    if n < n_splits + 1:
        raise ValueError(f"Need more unique samples than folds: n={n}, folds={n_splits}")

    indices = np.arange(n)
    pseudo = _pseudo_binary_labels(source_response, unique_samples, sample_col)

    eff_test_size = source_test_size
    min_test = max(1, int(round(n * source_test_size)))
    if min_test >= n:
        min_test = max(1, n // 10)
    eff_test_size = min_test / n

    try:
        train_val_idx, test_idx = train_test_split(
            indices, test_size=eff_test_size, random_state=seed, stratify=pseudo
        )
    except ValueError:
        train_val_idx, test_idx = train_test_split(
            indices, test_size=eff_test_size, random_state=seed
        )

    test_ids = [unique_samples[int(i)] for i in sorted(test_idx)]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[FoldSplit] = []

    for fold_id, (tr, va) in enumerate(kf.split(train_val_idx)):
        train_ids = [unique_samples[int(train_val_idx[i])] for i in tr]
        val_ids = [unique_samples[int(train_val_idx[i])] for i in va]
        folds.append(
            FoldSplit(
                fold=fold_id,
                train_sample_ids=train_ids,
                valid_sample_ids=val_ids,
                source_test_sample_ids=test_ids,
            )
        )

    rows = []
    for fold in folds:
        for sid in fold.train_sample_ids:
            rows.append({"fold_id": fold.fold, "sample_id": sid, "split": "train"})
        for sid in fold.valid_sample_ids:
            rows.append({"fold_id": fold.fold, "sample_id": sid, "split": "val"})
        for sid in fold.source_test_sample_ids:
            rows.append({"fold_id": fold.fold, "sample_id": sid, "split": "source_test"})

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
    return SourceSplits(
        folds=folds,
        source_split_report=pd.DataFrame(rows),
        fold_summary=fold_summary,
    )


def validate_no_sample_id_leakage(splits: SourceSplits) -> None:
    for fold in splits.folds:
        train = set(fold.train_sample_ids)
        val = set(fold.valid_sample_ids)
        test = set(fold.source_test_sample_ids)
        if train & val or train & test or val & test:
            raise ValueError(f"Sample_ID leakage detected in fold {fold.fold}")
