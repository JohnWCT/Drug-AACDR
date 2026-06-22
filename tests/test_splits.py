"""Tests for grouped source split."""

import pandas as pd

from aacdr_pipeline.splits import build_grouped_source_splits, validate_no_sample_id_leakage


def test_grouped_split_no_leakage():
    rows = []
    samples = [f"S{i}" for i in range(30)]
    for sid in samples:
        for d in ["a", "b"]:
            rows.append({"Sample_ID": sid, "drug_id": d, "label": 0})
    resp = pd.DataFrame(rows)
    splits = build_grouped_source_splits(resp, sample_col="Sample_ID", n_splits=5, seed=0)
    validate_no_sample_id_leakage(splits)
    assert len(splits.folds) == 5
    test_ids = set(splits.folds[0].source_test_sample_ids)
    for fold in splits.folds:
        assert test_ids == set(fold.source_test_sample_ids)
        assert not (set(fold.train_sample_ids) & set(fold.valid_sample_ids))
        assert not (set(fold.train_sample_ids) & test_ids)
