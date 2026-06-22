"""Export omics-level latent representations."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from aacdr_pipeline.model_adapter import PipelineModels, extract_omics_latent


def build_latent_dict(
    sample_ids: list[str],
    latent_matrix: np.ndarray,
    domain: str,
    split: str,
    fold: int,
    seed: int,
    cancer_type_map: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i, sid in enumerate(sample_ids):
        out[sid] = {
            "sample_id": sid,
            "domain": domain,
            "split": split,
            "fold": fold,
            "seed": seed,
            "cancer_type": cancer_type_map.get(sid, "") if cancer_type_map else "",
            "latent": latent_matrix[i].tolist(),
        }
    return out


def export_fold_latents(
    models: PipelineModels,
    source_omics: pd.DataFrame,
    target_omics: pd.DataFrame,
    source_sample_col: str,
    target_sample_col: str,
    feature_names: list[str],
    fold: int,
    seed: int,
    source_cancer_map: dict[str, str] | None,
    target_cancer_map: dict[str, str] | None,
) -> tuple[dict, dict]:
    src_ids = source_omics[source_sample_col].astype(str).tolist()
    src_x = torch.tensor(source_omics[feature_names].astype(float).values, dtype=torch.float32)
    src_latent = extract_omics_latent(models, src_x)
    src_dict = build_latent_dict(
        src_ids, src_latent, "source", "all", fold, seed, source_cancer_map
    )

    tgt_ids = target_omics[target_sample_col].astype(str).tolist()
    tgt_x = torch.tensor(target_omics[feature_names].astype(float).values, dtype=torch.float32)
    tgt_latent = extract_omics_latent(models, tgt_x)
    tgt_dict = build_latent_dict(
        tgt_ids, tgt_latent, "target", "all", fold, seed, target_cancer_map
    )
    return src_dict, tgt_dict


def save_latent_pickle(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def latent_dict_to_matrix(latent_dict: dict) -> tuple[list[str], np.ndarray]:
    ids = sorted(latent_dict.keys())
    mat = np.asarray([latent_dict[sid]["latent"] for sid in ids], dtype=np.float64)
    return ids, mat
