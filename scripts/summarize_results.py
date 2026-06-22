#!/usr/bin/env python3
"""
彙整 AACDR 訓練日誌中的 per-model 指標，輸出 mean ± std，
並可與文獻補充材料（100_random_initializations.txt）比對。
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

METRICS = ["auc", "acc", "precision", "recall", "f1"]
SPLITS = ["whole", "seen", "unseen"]

LINE_RE = re.compile(
    r"^(whole|seen|unseen):\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)


def parse_log(path: Path) -> List[Dict[str, Dict[str, float]]]:
    """從 log 檔解析每次 model 的 whole/seen/unseen 五項指標。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = [b.strip() for b in text.split("------------------------------") if b.strip()]
    records: List[Dict[str, Dict[str, float]]] = []

    for block in blocks:
        entry: Dict[str, Dict[str, float]] = {}
        for line in block.splitlines():
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            split = m.group(1)
            values = [float(m.group(i)) for i in range(2, 7)]
            entry[split] = dict(zip(METRICS, values))
        if entry:
            records.append(entry)
    return records


def aggregate(records: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """對 n 次初始化結果計算 mean 與 sample std。"""
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for split in SPLITS:
        out[split] = {}
        for metric in METRICS:
            vals = [r[split][metric] for r in records if split in r and metric in r[split]]
            if not vals:
                out[split][metric] = (float("nan"), float("nan"))
                continue
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            out[split][metric] = (mean, std)
    return out


def format_table(
    title: str,
    stats: Dict[str, Dict[str, Tuple[float, float]]],
    n_models: int,
) -> str:
    lines = [f"## {title}", f"n_models = {n_models}", ""]
    header = ["split", "metric", "mean", "std", "mean±std"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for split in SPLITS:
        for metric in METRICS:
            mean, std = stats[split][metric]
            lines.append(
                f"| {split} | {metric} | {mean:.6f} | {std:.6f} | {mean:.4f} ± {std:.4f} |"
            )
    return "\n".join(lines)


def write_csv(path: Path, stats: Dict[str, Dict[str, Tuple[float, float]]], n_models: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "metric", "mean", "std", "n_models"])
        for split in SPLITS:
            for metric in METRICS:
                mean, std = stats[split][metric]
                w.writerow([split, metric, f"{mean:.8f}", f"{std:.8f}", n_models])


def compare_tables(
    ours: Dict[str, Dict[str, Tuple[float, float]]],
    ref: Dict[str, Dict[str, Tuple[float, float]]],
) -> str:
    lines = ["## 與文獻參考值差異 (ours - reference)", ""]
    header = ["split", "metric", "ours_mean", "ref_mean", "delta_mean", "ours_std", "ref_std"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for split in SPLITS:
        for metric in METRICS:
            o_mean, o_std = ours[split][metric]
            r_mean, r_std = ref[split][metric]
            lines.append(
                f"| {split} | {metric} | {o_mean:.4f} | {r_mean:.4f} | {o_mean - r_mean:+.4f} | {o_std:.4f} | {r_std:.4f} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AACDR training metrics")
    parser.add_argument("--log", type=Path, required=True, help="Training log txt from ckpt/")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Literature reference file (100_random_initializations.txt)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    log_path = args.log.resolve()
    out_dir = (args.out_dir or log_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = parse_log(log_path)
    if not records:
        raise SystemExit(f"No metrics parsed from {log_path}")

    stats = aggregate(records)
    stem = log_path.stem

    md_path = out_dir / f"{stem}_summary.md"
    csv_path = out_dir / f"{stem}_summary.csv"

    sections = [format_table(f"Results: {log_path.name}", stats, len(records))]

    if args.reference and args.reference.exists():
        ref_records = parse_log(args.reference)
        ref_stats = aggregate(ref_records)
        sections.append(format_table(f"Reference: {args.reference.name}", ref_stats, len(ref_records)))
        sections.append(compare_tables(stats, ref_stats))

    md_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    write_csv(csv_path, stats, len(records))

    print(md_path.read_text(encoding="utf-8"))
    print(f"Saved: {md_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
