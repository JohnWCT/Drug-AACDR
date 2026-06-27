#!/usr/bin/env python3
"""Run inference-only TCGA target evaluation from pretrained AACDR fold checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from aacdr_pipeline.config import AACDRPipelineConfig
from aacdr_pipeline.infer import load_config_from_checkpoint_dir, run_target_inference

DEFAULT_CHECKPOINT_DIR = "/workspace/AACDR/outputs_aacdr_eval3"
DEFAULT_OUTPUT_DIR = "/workspace/AACDR/outputs_aacdr_eval_aacdr_target_infer"
DEFAULT_PRIMARY = (
    "/workspace/DAPL-master/data/TCGA/"
    "TCGA_AACDR_response_final_with_smiles_intersect_pretrain_gdsc_intersect.csv"
)
DEFAULT_TARGET_ONLY = (
    "/workspace/DAPL-master/data/TCGA/"
    "TCGA_AACDR_response_final_with_smiles_intersect_pretrain_tcga_only.csv"
)
DEFAULT_DRUG_SMILES = (
    "/workspace/DAPL-master/data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint_dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory with fold_*/best_model_*.pt from prior training run.",
    )
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--target_eval_primary_response_path",
        default=DEFAULT_PRIMARY,
        help="TCGA primary eval (gdsc intersect).",
    )
    p.add_argument(
        "--target_eval_target_only_response_path",
        default=DEFAULT_TARGET_ONLY,
        help="TCGA target-only eval.",
    )
    p.add_argument(
        "--target_eval_aux_response_path",
        default=None,
        help="Optional auxiliary eval CSV; defaults to checkpoint config value.",
    )
    p.add_argument("--drug_smiles_path", default=DEFAULT_DRUG_SMILES)
    p.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    p.add_argument(
        "--eval_prefixes",
        default="target_primary,target_only",
        help="Comma-separated metric prefixes to export.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = load_config_from_checkpoint_dir(args.checkpoint_dir)

    config = AACDRPipelineConfig(
        **{
            **base.__dict__,
            "output_dir": args.output_dir,
            "target_eval_primary_response_path": args.target_eval_primary_response_path,
            "target_eval_target_only_response_path": args.target_eval_target_only_response_path,
            "target_eval_aux_response_path": args.target_eval_aux_response_path
            or base.target_eval_aux_response_path,
            "drug_smiles_path": args.drug_smiles_path,
            "device": args.device,
            "run_tsne": False,
            "run_fid": False,
            "run_kmeans": False,
        }
    )

    eval_prefixes = [x.strip() for x in args.eval_prefixes.split(",") if x.strip()]
    out = run_target_inference(
        config,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        eval_prefixes=eval_prefixes,
    )
    print(json.dumps({"output_dir": str(out), "eval_prefixes": eval_prefixes}, indent=2))


if __name__ == "__main__":
    main()
