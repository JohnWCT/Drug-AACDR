"""Final drug index and metadata reports."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aacdr_pipeline.schema import normalize_drug_name


@dataclass
class DrugIndex:
    drug_ids: list[str]
    drug_to_index: dict[str, int]
    index_to_drug: dict[int, str]


@dataclass
class DrugMetadata:
    drug_index: DrugIndex
    drug_list: pd.DataFrame
    drug_availability_report: pd.DataFrame
    zero_shot_drug_report: pd.DataFrame


def _unique_drugs(series_list: list[pd.Series]) -> set[str]:
    out: set[str] = set()
    for s in series_list:
        out.update(s.dropna().astype(str).tolist())
    return out


def build_final_drug_index(
    source_response: pd.DataFrame,
    target_primary: pd.DataFrame,
    target_only: pd.DataFrame,
    target_auxiliary: pd.DataFrame,
    source_drug_col: str = "drug_id",
) -> DrugIndex:
    source_drugs = set(source_response[source_drug_col].astype(str))
    all_drugs = sorted(
        source_drugs
        | set(target_primary["drug_id"])
        | set(target_only["drug_id"])
        | set(target_auxiliary["drug_id"])
    )
    drug_to_index = {d: i for i, d in enumerate(all_drugs)}
    index_to_drug = {i: d for d, i in drug_to_index.items()}
    return DrugIndex(drug_ids=all_drugs, drug_to_index=drug_to_index, index_to_drug=index_to_drug)


def build_drug_metadata(
    drug_index: DrugIndex,
    source_response: pd.DataFrame,
    target_primary: pd.DataFrame,
    target_only: pd.DataFrame,
    target_auxiliary: pd.DataFrame,
    drug_feature_table: pd.DataFrame,
    smiles_drug_col: str = "drug_name",
    smiles_col: str = "SMILES",
) -> DrugMetadata:
    source_drugs = set(source_response["drug_id"].astype(str))
    primary_drugs = set(target_primary["drug_id"].astype(str))
    target_only_drugs = set(target_only["drug_id"].astype(str))
    aux_drugs = set(target_auxiliary["drug_id"].astype(str))

    smiles_map: dict[str, str] = {}
    if smiles_drug_col in drug_feature_table.columns and smiles_col in drug_feature_table.columns:
        for _, row in drug_feature_table.iterrows():
            did = normalize_drug_name(row[smiles_drug_col])
            smi = str(row[smiles_col]).strip()
            if smi and smi.lower() != "nan":
                smiles_map.setdefault(did, smi)

    rows = []
    for drug_id in drug_index.drug_ids:
        idx = drug_index.drug_to_index[drug_id]
        in_source = drug_id in source_drugs
        in_primary = drug_id in primary_drugs
        in_target_only = drug_id in target_only_drugs
        in_aux = drug_id in aux_drugs
        in_any_target = in_primary or in_target_only or in_aux
        has_supervised = in_source
        is_target_eval_only = in_any_target and not has_supervised
        rows.append(
            {
                "drug_id": drug_id,
                "drug_index": idx,
                "in_source": in_source,
                "in_target_primary": in_primary,
                "in_target_only": in_target_only,
                "in_target_auxiliary": in_aux,
                "in_any_target_eval": in_any_target,
                "has_supervised_source_label": has_supervised,
                "is_target_eval_only": is_target_eval_only,
                "has_smiles": drug_id in smiles_map,
            }
        )

    drug_list = pd.DataFrame(rows)
    zero_shot = drug_list[drug_list["is_target_eval_only"]].copy()
    availability = drug_list[
        ["drug_id", "drug_index", "has_smiles", "has_supervised_source_label", "is_target_eval_only"]
    ].copy()
    return DrugMetadata(
        drug_index=drug_index,
        drug_list=drug_list,
        drug_availability_report=availability,
        zero_shot_drug_report=zero_shot,
    )
