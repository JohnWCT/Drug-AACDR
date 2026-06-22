#!/usr/bin/env python3
"""Thin CLI entry point for the AACDR multilabel evaluation pipeline."""

from __future__ import annotations

import os
import sys

# Ensure legacy AACDR modules (model, trainer, dataset) are importable
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from aacdr_pipeline.config import build_arg_parser, config_from_args, validate_config
from aacdr_pipeline.run import run_pipeline


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    validate_config(config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
