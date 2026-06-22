"""Tests for drug index union and zero-shot flags."""

import pandas as pd

from aacdr_pipeline.drug_index import build_drug_metadata, build_final_drug_index


def test_drug_index_union_and_zero_shot():
    src = pd.DataFrame({"drug_id": ["a", "b"]})
    primary = pd.DataFrame({"drug_id": ["a", "c"]})
    tonly = pd.DataFrame({"drug_id": ["d"]})
    aux = pd.DataFrame({"drug_id": ["e"]})
    idx = build_final_drug_index(src, primary, tonly, aux)
    assert set(idx.drug_ids) == {"a", "b", "c", "d", "e"}
    meta = build_drug_metadata(
        idx, src, primary, tonly, aux, pd.DataFrame({"drug_name": [], "SMILES": []})
    )
    z = meta.drug_list.set_index("drug_id")
    assert z.loc["d", "is_target_eval_only"]
    assert z.loc["a", "has_supervised_source_label"]
    assert not z.loc["a", "is_target_eval_only"]
