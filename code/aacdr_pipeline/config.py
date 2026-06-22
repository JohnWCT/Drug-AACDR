"""CLI and configuration for the AACDR evaluation pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_OMICS = "/workspace/DAPL-master/data/pretrain_ccle.csv"
DEFAULT_SOURCE_RESPONSE = (
    "/workspace/DAPL-master/data/GDSC2_fitted_dose_response_MaxScreen_raw.csv"
)
DEFAULT_TARGET_OMICS = "/workspace/DAPL-master/data/TCGA/pretrain_tcga.csv"
DEFAULT_DRUG_SMILES = (
    "/workspace/DAPL-master/data/GDSC_drug_merge_pubchem_dropNA_MACCS.csv"
)
DEFAULT_TARGET_PRIMARY = (
    "/workspace/DAPL-master/data/TCGA/"
    "PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv"
)
DEFAULT_TARGET_ONLY = (
    "/workspace/DAPL-master/data/TCGA/"
    "PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv"
)
DEFAULT_TARGET_AUX = (
    "/workspace/DAPL-master/data/TCGA/TCGA_drug_response_from_DAPL.csv"
)
DEFAULT_CCLE_CANCER_INFO = "/workspace/DAPL-master/data/ccle_sample_info_df.csv"
DEFAULT_TCGA_CANCER_INFO = "/workspace/DAPL-master/data/TCGA/xena_sample_info_df.csv"


@dataclass
class AACDRPipelineConfig:
    source_omics_path: str = DEFAULT_SOURCE_OMICS
    source_response_path: str = DEFAULT_SOURCE_RESPONSE
    target_omics_path: str = DEFAULT_TARGET_OMICS
    target_eval_primary_response_path: str = DEFAULT_TARGET_PRIMARY
    target_eval_target_only_response_path: str = DEFAULT_TARGET_ONLY
    target_eval_aux_response_path: str = DEFAULT_TARGET_AUX
    drug_smiles_path: str = DEFAULT_DRUG_SMILES
    ccle_cancer_info_path: str = DEFAULT_CCLE_CANCER_INFO
    tcga_cancer_info_path: str = DEFAULT_TCGA_CANCER_INFO
    output_dir: str = "outputs_aacdr_eval3"
    source_sample_col: str = "Sample_ID"
    source_drug_col: str = "drug_name"
    source_label_col: str = "Label"
    target_sample_col: str = "Patient_id"
    target_drug_col: str = "drug_name"
    target_label_col: str = "Label"
    target_omics_sample_col: str = "tissue_id"
    n_splits: int = 5
    source_test_size: float = 0.10
    seed: int = 0
    max_epoch: int = 10
    batch_size: int = 768
    learning_rate: float = 0.0005
    device: str | None = None
    run_tsne: bool = True
    run_fid: bool = True
    run_kmeans: bool = True
    early_stop_patience: int = 3
  # Optional: limit samples/drugs for smoke / debug runs
    max_train_rows: int | None = None
    max_target_omics_samples: int | None = None
    aacdr_data_root: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AACDR multilabel evaluation pipeline (5-fold + 3 TCGA eval sets)"
    )
    p.add_argument("--source_omics_path", default=DEFAULT_SOURCE_OMICS)
    p.add_argument("--source_response_path", default=DEFAULT_SOURCE_RESPONSE)
    p.add_argument("--target_omics_path", default=DEFAULT_TARGET_OMICS)
    p.add_argument(
        "--target_eval_primary_response_path", default=DEFAULT_TARGET_PRIMARY
    )
    p.add_argument(
        "--target_eval_target_only_response_path", default=DEFAULT_TARGET_ONLY
    )
    p.add_argument("--target_eval_aux_response_path", default=DEFAULT_TARGET_AUX)
    p.add_argument("--drug_smiles_path", default=DEFAULT_DRUG_SMILES)
    p.add_argument("--ccle_cancer_info_path", default=DEFAULT_CCLE_CANCER_INFO)
    p.add_argument("--tcga_cancer_info_path", default=DEFAULT_TCGA_CANCER_INFO)
    p.add_argument("--output_dir", default="outputs_aacdr_eval3")
    p.add_argument("--source_sample_col", default="Sample_ID")
    p.add_argument("--source_drug_col", default="drug_name")
    p.add_argument("--source_label_col", default="Label")
    p.add_argument("--target_sample_col", default="Patient_id")
    p.add_argument("--target_drug_col", default="drug_name")
    p.add_argument("--target_label_col", default="Label")
    p.add_argument("--target_omics_sample_col", default="tissue_id")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--source_test_size", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_epoch", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=768)
    p.add_argument("--learning_rate", type=float, default=0.0005)
    p.add_argument("--device", default=None)
    p.add_argument("--no_tsne", action="store_true")
    p.add_argument("--no_fid", action="store_true")
    p.add_argument("--no_kmeans", action="store_true")
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--max_train_rows", type=int, default=None)
    p.add_argument("--max_target_omics_samples", type=int, default=None)
    p.add_argument("--aacdr_data_root", default=None)
    return p


def config_from_args(args: argparse.Namespace) -> AACDRPipelineConfig:
    return AACDRPipelineConfig(
        source_omics_path=args.source_omics_path,
        source_response_path=args.source_response_path,
        target_omics_path=args.target_omics_path,
        target_eval_primary_response_path=args.target_eval_primary_response_path,
        target_eval_target_only_response_path=args.target_eval_target_only_response_path,
        target_eval_aux_response_path=args.target_eval_aux_response_path,
        drug_smiles_path=args.drug_smiles_path,
        ccle_cancer_info_path=args.ccle_cancer_info_path,
        tcga_cancer_info_path=args.tcga_cancer_info_path,
        output_dir=args.output_dir,
        source_sample_col=args.source_sample_col,
        source_drug_col=args.source_drug_col,
        source_label_col=args.source_label_col,
        target_sample_col=args.target_sample_col,
        target_drug_col=args.target_drug_col,
        target_label_col=args.target_label_col,
        target_omics_sample_col=args.target_omics_sample_col,
        n_splits=args.n_splits,
        source_test_size=args.source_test_size,
        seed=args.seed,
        max_epoch=args.max_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        run_tsne=not args.no_tsne,
        run_fid=not args.no_fid,
        run_kmeans=not args.no_kmeans,
        early_stop_patience=args.early_stop_patience,
        max_train_rows=args.max_train_rows,
        max_target_omics_samples=args.max_target_omics_samples,
        aacdr_data_root=args.aacdr_data_root,
    )


def validate_config(config: AACDRPipelineConfig) -> None:
    if not (0.0 < config.source_test_size < 1.0):
        raise ValueError("source_test_size must be in (0, 1)")
    if config.n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    required = [
        ("source_omics_path", config.source_omics_path),
        ("source_response_path", config.source_response_path),
        ("target_omics_path", config.target_omics_path),
        ("target_eval_primary_response_path", config.target_eval_primary_response_path),
        (
            "target_eval_target_only_response_path",
            config.target_eval_target_only_response_path,
        ),
        ("target_eval_aux_response_path", config.target_eval_aux_response_path),
        ("drug_smiles_path", config.drug_smiles_path),
    ]
    for name, path in required:
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"Missing required file for {name}: {path}")


def config_to_dict(config: AACDRPipelineConfig) -> dict[str, Any]:
    return asdict(config)


def save_config(config: AACDRPipelineConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_to_dict(config), f, indent=2)
