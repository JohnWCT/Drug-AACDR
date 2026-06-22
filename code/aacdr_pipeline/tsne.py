"""t-SNE visualization on omics latent (visualization only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def _combine_latent_frames(
    source_dict: dict,
    target_dict: dict,
) -> pd.DataFrame:
    rows = []
    for sid, rec in source_dict.items():
        rows.append(
            {
                "sample_id": sid,
                "domain": rec.get("domain", "source"),
                "cancer_type": rec.get("cancer_type", "Unknown"),
                "latent": rec["latent"],
            }
        )
    for sid, rec in target_dict.items():
        rows.append(
            {
                "sample_id": sid,
                "domain": rec.get("domain", "target"),
                "cancer_type": rec.get("cancer_type", "Unknown"),
                "latent": rec["latent"],
            }
        )
    return pd.DataFrame(rows)


def run_tsne_plots(
    source_dict: dict,
    target_dict: dict,
    output_dir: str | Path,
    fold: int,
    seed: int = 0,
    max_samples: int = 2000,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _combine_latent_frames(source_dict, target_dict)
    if df.empty:
        return

    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)

    X = np.asarray(df["latent"].tolist(), dtype=np.float64)
    perplexity = min(30, max(5, len(df) // 10))
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")
    emb = tsne.fit_transform(X)
    df = df.copy()
    df["tsne_x"] = emb[:, 0]
    df["tsne_y"] = emb[:, 1]

    # Domain mixing
    fig, ax = plt.subplots(figsize=(8, 6))
    for domain, g in df.groupby("domain"):
        ax.scatter(g["tsne_x"], g["tsne_y"], label=domain, alpha=0.5, s=10)
    ax.set_title(f"t-SNE domain mixing (fold {fold})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_domain_mixing.png", dpi=120)
    plt.close(fig)

    # Cancer type
    fig, ax = plt.subplots(figsize=(10, 7))
    for ct, g in df.groupby("cancer_type"):
        ax.scatter(g["tsne_x"], g["tsne_y"], label=str(ct)[:30], alpha=0.5, s=8)
    ax.set_title(f"t-SNE cancer type (fold {fold})")
    if df["cancer_type"].nunique() <= 15:
        ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_cancer_type.png", dpi=120)
    plt.close(fig)
