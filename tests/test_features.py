"""Feature alignment tests."""

import pandas as pd

from aacdr_pipeline.features import align_source_target_features


def test_feature_alignment_common_genes():
    src = pd.DataFrame(
        {
            "Sample_ID": ["A1", "A2"],
            "G1": [1.0, 2.0],
            "G2": [3.0, 4.0],
            "G3": [5.0, 6.0],
        }
    )
    tgt = pd.DataFrame(
        {
            "tissue_id": ["TCGA-XX-0001-01"],
            "G2": [1.0],
            "G3": [2.0],
            "G4": [3.0],
        }
    )
    aligned = align_source_target_features(src, tgt, "Sample_ID", "tissue_id")
    assert aligned.feature_names == ["G2", "G3"]
    assert len(aligned.source_omics) == 2
    assert aligned.target_omics["tissue_id"].iloc[0] == "TCGA-XX-0001"
