"""Synthetic fixtures for pipeline unit/smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def write_fixtures() -> Path:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    genes = ["G1", "G2", "G3", "G4", "G5"]

    src_samples = [f"ACH-{i:06d}" for i in range(1, 21)]
    src_omics = pd.DataFrame(
        np.random.randn(len(src_samples), len(genes)),
        columns=genes,
    )
    src_omics.insert(0, "Sample_ID", src_samples)
    src_omics.to_csv(FIXTURE_DIR / "source_omics.csv", index=False)

    drugs = ["paclitaxel", "cisplatin", "gemcitabine", "docetaxel", "methotrexate"]
    rows = []
    rng = np.random.default_rng(0)
    for sid in src_samples:
        for drug in drugs[:3]:
            rows.append(
                {
                    "Sample_ID": sid,
                    "drug_name": drug,
                    "Label": int(rng.random() > 0.5),
                }
            )
    pd.DataFrame(rows).to_csv(FIXTURE_DIR / "source_response.csv", index=False)

    tgt_samples = [f"TCGA-XX-{i:04d}-01" for i in range(1, 16)]
    tgt_omics = pd.DataFrame(
        np.random.randn(len(tgt_samples), len(genes)),
        columns=genes,
    )
    tgt_omics.insert(0, "tissue_id", tgt_samples)
    tgt_omics.to_csv(FIXTURE_DIR / "target_omics.csv", index=False)

    def _tgt_resp(name: str, drug_subset: list[str]) -> None:
        trows = []
        for sid in tgt_samples[:10]:
            for drug in drug_subset:
                trows.append(
                    {
                        "Patient_id": sid.replace("-01", ""),
                        "drug_name": drug,
                        "Label": int(rng.random() > 0.5),
                        "cancers": "TEST",
                    }
                )
        pd.DataFrame(trows).to_csv(FIXTURE_DIR / name, index=False)

    _tgt_resp("target_primary.csv", ["paclitaxel", "cisplatin"])
    _tgt_resp("target_only.csv", ["docetaxel"])
    _tgt_resp("target_aux.csv", ["methotrexate"])

    smiles_rows = []
    smi = {
        "paclitaxel": "CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C",
        "cisplatin": "N.N.[Cl-].[Cl-].[Pt+2]",
        "gemcitabine": "C1=CN(C(=O)N=C1N)[C@H]2C([C@@H]([C@H](O2)CO)O)(F)F",
        "docetaxel": "CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)OC(C)(C)C)O)O)OC(=O)C6=CC=CC=C6)(CO4)OC(=O)C)O)C)O",
        "methotrexate": "CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)N[C@@H](CCC(=O)O)C(=O)O",
    }
    for drug, s in smi.items():
        smiles_rows.append({"drug_name": drug, "SMILES": s, "pubchem": "0"})
    pd.DataFrame(smiles_rows).to_csv(FIXTURE_DIR / "drug_smiles.csv", index=False)

    ccle = pd.DataFrame(
        {"Sanger_Model_ID": src_samples, "cancer_type": ["Lung Cancer"] * len(src_samples)}
    )
    ccle.to_csv(FIXTURE_DIR / "ccle_info.csv", index=False)

    xena = pd.DataFrame(
        {
            "tissue_id": tgt_samples,
            "cancer_type": ["Bladder Cancer"] * len(tgt_samples),
        }
    )
    xena.to_csv(FIXTURE_DIR / "tcga_info.csv", index=False)
    return FIXTURE_DIR


if __name__ == "__main__":
    p = write_fixtures()
    print(f"Wrote fixtures to {p}")
