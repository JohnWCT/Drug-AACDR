"""CSV/JSON export helpers and cross-fold aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from aacdr_pipeline.evaluation import aggregate_metrics_across_folds


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)


def write_json(data: dict, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def fold_dir(output_dir: str | Path, fold: int) -> Path:
    return Path(output_dir) / f"fold_{fold}"


def aggregate_fold_metric_files(
    output_dir: str | Path,
    prefix: str,
    summary: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    frames = []
    for fold_path in sorted(output_dir.glob("fold_*")):
        if not fold_path.is_dir():
            continue
        fname = f"{prefix}_metrics_summary.csv" if summary else f"{prefix}_metrics_per_drug.csv"
        fpath = fold_path / fname
        if fpath.is_file():
            frames.append(pd.read_csv(fpath))
    across, mean_std = aggregate_metrics_across_folds(frames)
    return across, mean_std


def write_cross_fold_reports(output_dir: str | Path, prefixes: list[str]) -> None:
    output_dir = Path(output_dir)
    for prefix in prefixes:
        across, mean_std = aggregate_fold_metric_files(output_dir, prefix, summary=True)
        if len(across):
            write_csv(across, output_dir / f"{prefix}_metrics_summary_across_folds.csv")
            write_csv(mean_std, output_dir / f"{prefix}_metrics_summary_fold_mean_std.csv")
        across_d, mean_std_d = aggregate_fold_metric_files(output_dir, prefix, summary=False)
        if len(across_d):
            write_csv(across_d, output_dir / f"{prefix}_metrics_per_drug_across_folds.csv")
            write_csv(mean_std_d, output_dir / f"{prefix}_metrics_per_drug_fold_mean_std.csv")
