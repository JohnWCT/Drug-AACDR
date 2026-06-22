"""Config defaults and CLI override tests."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from aacdr_pipeline.config import (
    AACDRPipelineConfig,
    build_arg_parser,
    config_from_args,
)


def test_config_defaults():
    cfg = AACDRPipelineConfig()
    assert cfg.n_splits == 5
    assert cfg.source_test_size == 0.10
    assert cfg.source_drug_col == "drug_name"
    assert cfg.source_label_col == "Label"


def test_cli_override():
    parser = build_arg_parser()
    args = parser.parse_args(["--seed", "42", "--n_splits", "3", "--max_epoch", "1"])
    cfg = config_from_args(args)
    assert cfg.seed == 42
    assert cfg.n_splits == 3
    assert cfg.max_epoch == 1
