"""Prepare TCGA evaluation datasets (labels never enter training)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aacdr_pipeline.drug_index import DrugIndex
from aacdr_pipeline.schema import normalize_tcga_patient_id


@dataclass
class TargetEvalDataset:
    name: str
    response_long: pd.DataFrame
    sample_ids: list[str]
    drug_ids: list[str]
    report: pd.DataFrame


@dataclass
class TargetEvalBundle:
    primary: TargetEvalDataset
    target_only: TargetEvalDataset
    auxiliary: TargetEvalDataset
    report: pd.DataFrame


def prepare_target_eval_dataset(
    name: str,
    response: pd.DataFrame,
    target_omics: pd.DataFrame,
    target_omics_sample_col: str,
    drug_index: DrugIndex,
) -> TargetEvalDataset:
    omics_ids = set(target_omics[target_omics_sample_col].map(normalize_tcga_patient_id))
    drug_set = set(drug_index.drug_ids)
    input_rows = len(response)

    usable = []
    skipped_sample = 0
    skipped_drug = 0
    for _, row in response.iterrows():
        sid = row["sample_id"]
        did = row["drug_id"]
        if sid not in omics_ids:
            skipped_sample += 1
            continue
        if did not in drug_set:
            skipped_drug += 1
            continue
        usable.append(row)

    usable_df = pd.DataFrame(usable) if usable else pd.DataFrame(columns=response.columns)
    label_0 = int((usable_df["label"] == 0).sum()) if len(usable_df) else 0
    label_1 = int((usable_df["label"] == 1).sum()) if len(usable_df) else 0

    report = pd.DataFrame(
        [
            {
                "eval_dataset": name,
                "input_rows": input_rows,
                "usable_rows": len(usable_df),
                "skipped_rows_sample_not_in_target_omics": skipped_sample,
                "skipped_rows_drug_not_in_final_index": skipped_drug,
                "n_observed_patients": usable_df["sample_id"].nunique() if len(usable_df) else 0,
                "n_observed_drugs": usable_df["drug_id"].nunique() if len(usable_df) else 0,
                "label_0_count": label_0,
                "label_1_count": label_1,
            }
        ]
    )
    return TargetEvalDataset(
        name=name,
        response_long=usable_df,
        sample_ids=sorted(usable_df["sample_id"].unique().tolist()) if len(usable_df) else [],
        drug_ids=sorted(usable_df["drug_id"].unique().tolist()) if len(usable_df) else [],
        report=report,
    )


def prepare_all_target_eval_datasets(
    target_primary: pd.DataFrame,
    target_only: pd.DataFrame,
    target_auxiliary: pd.DataFrame,
    target_omics: pd.DataFrame,
    target_omics_sample_col: str,
    drug_index: DrugIndex,
) -> TargetEvalBundle:
    primary = prepare_target_eval_dataset(
        "primary", target_primary, target_omics, target_omics_sample_col, drug_index
    )
    tonly = prepare_target_eval_dataset(
        "target_only", target_only, target_omics, target_omics_sample_col, drug_index
    )
    aux = prepare_target_eval_dataset(
        "auxiliary", target_auxiliary, target_omics, target_omics_sample_col, drug_index
    )
    report = pd.concat([primary.report, tonly.report, aux.report], ignore_index=True)
    return TargetEvalBundle(primary=primary, target_only=tonly, auxiliary=aux, report=report)
