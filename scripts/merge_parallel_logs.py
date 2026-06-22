#!/usr/bin/env python3
"""合併平行訓練產生的 per-model partial logs。"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge parallel AACDR run logs")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--total-models", type=int, default=100)
    parser.add_argument("--out-log", type=Path, required=True)
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve()
    parts = []
    missing = []
    for i in range(args.total_models):
        part = runs_dir / f"model_{i:03d}.txt"
        if not part.exists():
            missing.append(str(part))
            continue
        parts.append(part.read_text(encoding="utf-8"))

    if missing:
        raise SystemExit(
            f"Missing {len(missing)} partial logs, e.g. {missing[:3]}"
        )

    merged = "".join(parts)
    args.out_log.parent.mkdir(parents=True, exist_ok=True)
    args.out_log.write_text(merged, encoding="utf-8")
    print(f"Merged {len(parts)} runs -> {args.out_log}")


if __name__ == "__main__":
    main()
