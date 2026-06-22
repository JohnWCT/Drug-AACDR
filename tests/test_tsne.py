"""Tests for DAPL-style dual-panel t-SNE plotting."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from aacdr_pipeline.tsne import (
    TSNE_RANDOM_STATE,
    plot_latent_tsne_dual,
    plot_tsne_dual_from_latent_dicts,
)


def _fake_latent(n: int, dim: int = 8, prefix: str = "S") -> dict:
    rng = np.random.default_rng(0)
    return {
        f"{prefix}{i}": {
            "latent": rng.normal(size=dim).tolist(),
            "cancer_type": "Lung Cancer" if i % 2 == 0 else "Bladder Cancer",
        }
        for i in range(n)
    }


def test_tsne_random_state_constant():
    assert TSNE_RANDOM_STATE == 42


def test_plot_latent_tsne_dual_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    source_z = rng.normal(size=(40, 8))
    target_z = rng.normal(size=(40, 8))
    source_labels = np.array([0] * 20 + [1] * 20)
    target_labels = np.array([0] * 20 + [1] * 20)
    mapping = {0: "Lung Cancer", 1: "Bladder Cancer"}
    out = tmp_path / "tsne_latent_dual.png"
    ok = plot_latent_tsne_dual(
        source_z, target_z, source_labels, target_labels, mapping, out
    )
    assert ok and out.is_file() and out.stat().st_size > 0


def test_plot_tsne_dual_from_latent_dicts(tmp_path):
    src = _fake_latent(30, prefix="src")
    tgt = _fake_latent(30, prefix="tgt")
    out = tmp_path / "tsne_latent_dual.png"
    ok = plot_tsne_dual_from_latent_dicts(src, tgt, out)
    assert ok and out.is_file()
