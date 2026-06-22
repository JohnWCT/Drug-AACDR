"""AACDR-compatible drug graph construction from SMILES / legacy npz."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aacdr_pipeline.drug_index import DrugIndex
from aacdr_pipeline.schema import normalize_drug_name

try:
    import deepchem as dc
    from rdkit import Chem
except ImportError:  # pragma: no cover
    dc = None
    Chem = None


GRAPH_SIZE = 100
FEAT_DIM = 75


@dataclass
class DrugRepresentationBundle:
    gdsc_drug_graph: dict[str, np.lib.npyio.NpzFile | dict]
    tcga_drug_graph: dict[str, np.lib.npyio.NpzFile | dict]
    availability_report: pd.DataFrame
    edge_report: pd.DataFrame


def _smiles_to_graph(smiles: str) -> tuple[np.ndarray, np.ndarray]:
    if dc is None or Chem is None:
        raise ImportError("deepchem and rdkit are required for drug graph construction")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fe = dc.feat.ConvMolFeaturizer()
    convmol = fe.featurize([mol])[0]
    adj_list = convmol.get_adjacency_list()
    n = len(adj_list)
    adj = np.zeros((GRAPH_SIZE, GRAPH_SIZE), dtype=np.float32)
    k = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in adj_list[i]:
            k[i, j] = 1.0
    np.fill_diagonal(k, 1.0)
    adj[:n, :n] = k
    features = np.zeros((GRAPH_SIZE, FEAT_DIM), dtype=np.float32)
    features[:n, :] = np.array(convmol.get_atom_features(), dtype=np.float32)
    return adj, features


def _try_load_legacy_npz(
    drug_id: str, pubchem_id: str | None, data_roots: list[Path]
) -> tuple[np.ndarray, np.ndarray] | None:
    keys = [drug_id]
    if pubchem_id:
        keys.append(str(int(float(pubchem_id))) if pubchem_id.replace(".", "", 1).isdigit() else pubchem_id)
    for root in data_roots:
        for sub in ("GDSC", "TCGA"):
            for folder in ("drug_graph", "drug_graph3"):
                base = root / "data" / sub / folder
                if not base.is_dir():
                    continue
                for key in keys:
                    path = base / f"{key}.npz"
                    if path.is_file():
                        data = np.load(path)
                        return data["adj"], data["feature"]
    return None


def build_drug_representations(
    drug_index: DrugIndex,
    drug_feature_table: pd.DataFrame,
    aacdr_data_root: str | None = None,
    smiles_drug_col: str = "drug_name",
    smiles_col: str = "SMILES",
    pubchem_col: str = "pubchem",
) -> DrugRepresentationBundle:
    smiles_map: dict[str, str] = {}
    pubchem_map: dict[str, str] = {}
    for _, row in drug_feature_table.iterrows():
        did = normalize_drug_name(row.get(smiles_drug_col, row.get("drug_name", "")))
        if smiles_col in row and pd.notna(row[smiles_col]):
            smiles_map[did] = str(row[smiles_col]).strip()
        if pubchem_col in row and pd.notna(row[pubchem_col]):
            pubchem_map[did] = str(row[pubchem_col])

    roots: list[Path] = []
    if aacdr_data_root:
        roots.append(Path(aacdr_data_root))
    roots.append(Path(__file__).resolve().parents[2])

    gdsc_graph: dict[str, dict] = {}
    tcga_graph: dict[str, dict] = {}
    avail_rows = []
    edge_rows = []

    for drug_id in drug_index.drug_ids:
        idx = drug_index.drug_to_index[drug_id]
        key = str(idx)
        legacy = _try_load_legacy_npz(drug_id, pubchem_map.get(drug_id), roots)
        source = "legacy_npz"
        n_edges = np.nan
        try:
            if legacy is not None:
                adj, feat = legacy
            elif drug_id in smiles_map:
                adj, feat = _smiles_to_graph(smiles_map[drug_id])
                source = "smiles_convmol"
            else:
                raise KeyError(f"No SMILES or legacy graph for drug {drug_id}")
            n_edges = int((adj[: adj.shape[0], : adj.shape[0]] > 0).sum())
            entry = {"adj": adj, "feature": feat}
            gdsc_graph[key] = entry
            tcga_graph[key] = entry
            ok = True
            err = ""
        except Exception as exc:  # noqa: BLE001
            ok = False
            err = str(exc)
            raise RuntimeError(
                f"Cannot build AACDR-compatible graph for drug {drug_id!r}: {err}"
            ) from exc

        avail_rows.append(
            {
                "drug_id": drug_id,
                "drug_index": idx,
                "graph_key": key,
                "available": ok,
                "source": source,
                "error": err,
            }
        )
        edge_rows.append(
            {
                "drug_id": drug_id,
                "drug_index": idx,
                "n_nodes": int(min(GRAPH_SIZE, np.count_nonzero(feat.sum(axis=1)))),
                "n_edges": n_edges,
                "source": source,
            }
        )

    return DrugRepresentationBundle(
        gdsc_drug_graph=gdsc_graph,
        tcga_drug_graph=tcga_graph,
        availability_report=pd.DataFrame(avail_rows),
        edge_report=pd.DataFrame(edge_rows),
    )
