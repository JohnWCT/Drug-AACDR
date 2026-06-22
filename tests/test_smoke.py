"""End-to-end smoke test on synthetic tiny data."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from aacdr_pipeline.config import AACDRPipelineConfig
from aacdr_pipeline.run import run_pipeline
from tests.make_fixtures import write_fixtures


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    fd = write_fixtures()
    return fd


def test_pipeline_smoke(fixture_dir, tmp_path):
    out = tmp_path / "smoke_out"
    cfg = AACDRPipelineConfig(
        source_omics_path=str(fixture_dir / "source_omics.csv"),
        source_response_path=str(fixture_dir / "source_response.csv"),
        target_omics_path=str(fixture_dir / "target_omics.csv"),
        target_eval_primary_response_path=str(fixture_dir / "target_primary.csv"),
        target_eval_target_only_response_path=str(fixture_dir / "target_only.csv"),
        target_eval_aux_response_path=str(fixture_dir / "target_aux.csv"),
        drug_smiles_path=str(fixture_dir / "drug_smiles.csv"),
        ccle_cancer_info_path=str(fixture_dir / "ccle_info.csv"),
        tcga_cancer_info_path=str(fixture_dir / "tcga_info.csv"),
        output_dir=str(out),
        n_splits=5,
        source_test_size=0.10,
        seed=0,
        max_epoch=1,
        batch_size=32,
        max_train_rows=200,
        run_tsne=True,
        run_fid=True,
        run_kmeans=True,
    )
    run_pipeline(cfg)

    assert (out / "config.json").is_file()
    assert (out / "source_split.csv").is_file()
    for i in range(5):
        fdir = out / f"fold_{i}"
        assert fdir.is_dir()
        for prefix in (
            "source_test",
            "target_primary",
            "target_only",
            "target_auxiliary",
        ):
            assert (fdir / f"{prefix}_prediction_results.csv").is_file()
            assert (fdir / f"{prefix}_metrics_per_drug.csv").is_file()
            assert (fdir / f"{prefix}_metrics_summary.csv").is_file()
        assert (fdir / "source_latent_representation.pkl").is_file()
        assert (fdir / "target_latent_representation.pkl").is_file()
        assert (fdir / "latent_distribution_metrics.csv").is_file()
        assert (fdir / "kmeans_cancer_type_metrics.csv").is_file()
        assert (fdir / "tsne_latent_dual.png").is_file()

    manifest = (out / "run_manifest.json").read_text(encoding="utf-8")
    assert "tcga_labels_in_training" in manifest
    assert "false" in manifest.lower()
