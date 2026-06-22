# Drug-AACDR New Pipeline Architecture Design

## 1. Background

The current `Drug-AACDR` project should be extended to support a modernized data input/output workflow while preserving the original AACDR model and training design as much as possible.

The new workflow aligns its data and report conventions with newer drug response prediction pipelines such as Drug-TransDRP, drug-DCF, and SSDA-SingleModel. However, AACDR should not be converted into those models. The implementation should introduce modular pipeline layers around the existing AACDR code.

The system should support:

- source CCLE/GDSC-style omics and response inputs;
- target TCGA omics;
- three independent TCGA response evaluation datasets;
- source 10% test plus 5-fold train/validation;
- grouped split by `Sample_ID`;
- source and target latent export;
- t-SNE visualization;
- FID / latent distribution metrics;
- k-means cancer type clustering;
- detailed prediction and metrics outputs.

---

## 2. Goals and Non-goals

### 2.1 Goals

1. Preserve existing AACDR model and trainer logic as much as possible.
2. Add a new data pipeline that can load DAPL-style source and TCGA files.
3. Add grouped source split by `Sample_ID`.
4. Use 10% source cell lines as fixed source test.
5. Use remaining 90% source cell lines for 5-fold train/validation.
6. Support three independent TCGA evaluation datasets: `primary`, `target_only`, and `auxiliary`.
7. Build final drug list from source and all three TCGA eval datasets.
8. Export independent prediction and metrics tables for each TCGA eval dataset.
9. Export source/target latent representations after training.
10. Run t-SNE, FID, and k-means cancer type analyses on omics latent features.
11. Keep modules independently testable.

### 2.2 Non-goals

1. Do not rewrite AACDR as Drug-TransDRP.
2. Do not rewrite AACDR as drug-DCF.
3. Do not force a pretrain/fine-tune split if AACDR does not naturally use it.
4. Do not use TCGA response labels for training.
5. Do not use TCGA metrics for early stopping.
6. Do not use source test metrics for model selection.
7. Do not preserve AACDR multi-model initialization as the default behavior.
8. Do not implement sample-drug integrated latent analysis in this revision.
9. Do not use t-SNE coordinates as primary k-means input.
10. Do not output merged target by-dataset summary tables unless requested later.

---

## 3. Current AACDR Architecture Summary

The local AACDR project should be inspected before implementation. The design assumes the current project contains script-style modules such as:

```text
code/main.py
code/parallel_train.py
code/model.py
code/trainer.py
code/dataset.py
code/utils.py
code/result.py
```

The current code likely couples argument parsing, data loading, dataset construction, model initialization, training, and result export.

The new architecture should decouple these responsibilities without forcing a full rewrite of `model.py` or `trainer.py`. Existing AACDR components should be treated as core legacy modules and wrapped with adapters.

---

## 4. Reference Pipeline Summary

### 4.1 Drug-TransDRP-style references

Drug-TransDRP is a reference for:

- thin CLI entry points;
- config object construction;
- data preparation modules;
- source test and fold split;
- fold-level output directories;
- per-drug metrics;
- target eval output naming;
- latent and k-means style reports.

AACDR should adopt the output organization and config discipline where useful, not the model implementation.

### 4.2 drug-DCF-style references

drug-DCF is a reference for:

- DAPL data directory layout;
- TCGA evaluation response files;
- TCGA patient ID normalization;
- cancer type k-means reporting;
- source/target latent export ideas;
- SMILES-to-graph logic for drug representation comparison.

AACDR should prioritize its own drug graph / feature logic. drug-DCF’s graph creation should be used only as a compatibility reference, not blindly copied.

### 4.3 SSDA-SingleModel-style references

SSDA-SingleModel is a reference for:

- structured output reports;
- latent export;
- t-SNE;
- FID-like latent distribution metrics;
- k-means summaries;
- fold mean/std aggregation.

---

## 5. Proposed Architecture Overview

The new AACDR pipeline should be implemented as a modular layer around the existing code.

High-level flow:

```text
parse CLI
load config
load source omics
load source response
load target omics
load TCGA response datasets
validate schemas
normalize sample IDs and drug names
align source/target features
build final drug index
prepare AACDR-compatible drug representations
create grouped source split
for each fold:
    build train/validation/source-test datasets
    build target unlabeled dataset if required by AACDR
    train AACDR model
    select best model by source validation
    predict source test
    predict TCGA primary
    predict TCGA target_only
    predict TCGA auxiliary
    export source/target latent
    run t-SNE
    run FID / latent distribution metrics
    run k-means cancer type clustering
aggregate source metrics across folds
aggregate target_primary metrics across folds
aggregate target_only metrics across folds
aggregate target_auxiliary metrics across folds
aggregate latent/kmeans summaries
write reports
```

---

## 6. Module Breakdown

Each module should have one primary responsibility and avoid tight coupling with other modules.

### 6.1 Config / CLI Module

Suggested path:

```text
code/aacdr_pipeline/config.py
```

Responsibilities:

- Define CLI contract.
- Load default paths.
- Support CLI override.
- Validate argument combinations.
- Convert args to a config dataclass.

Main objects/functions:

```python
@dataclass
class AACDRPipelineConfig:
    source_omics_path: str
    source_response_path: str
    target_omics_path: str
    target_eval_primary_response_path: str
    target_eval_target_only_response_path: str
    target_eval_aux_response_path: str
    drug_smiles_path: str
    output_dir: str
    source_sample_col: str
    source_drug_col: str
    source_label_col: str
    target_sample_col: str
    target_drug_col: str
    target_label_col: str
    n_splits: int
    source_test_size: float
    seed: int
    max_epoch: int
    batch_size: int
    learning_rate: float
    run_tsne: bool
    run_fid: bool
    run_kmeans: bool

def build_arg_parser() -> argparse.ArgumentParser: ...
def config_from_args(args: argparse.Namespace) -> AACDRPipelineConfig: ...
def validate_config(config: AACDRPipelineConfig) -> None: ...
```

Required defaults:

```text
source_omics_path = /workspace/DAPL-master/data/pretrain_ccle.csv
source_response_path = /workspace/DAPL-master/data/GDSC2_fitted_dose_response_MaxScreen_raw.csv
target_omics_path = /workspace/DAPL-master/data/TCGA/pretrain_tcga.csv
drug_smiles_path = /workspace/DAPL-master/data/GDSC_drug_merge_pubchem_dropNA_MACCS.csv
target_eval_primary_response_path = /workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv
target_eval_target_only_response_path = /workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv
target_eval_aux_response_path = /workspace/DAPL-master/data/TCGA/TCGA_drug_response_from_DAPL.csv
source_sample_col = Sample_ID
target_sample_col = Patient_id
target_drug_col = drug_name
target_label_col = Label
n_splits = 5
source_test_size = 0.10
seed = 0
```

Testing:

- default config;
- CLI override;
- missing files;
- invalid split values;
- incompatible options.

### 6.2 Schema / Validation Module

Suggested path:

```text
code/aacdr_pipeline/schema.py
```

Responsibilities:

- Validate CSV columns.
- Normalize drug names.
- Normalize TCGA patient keys.
- Validate labels.
- Validate source split group column.

Main functions:

```python
def normalize_drug_name(value: Any) -> str: ...
def normalize_tcga_patient_id(value: Any) -> str: ...
def validate_source_response_schema(df: pd.DataFrame, config: AACDRPipelineConfig) -> None: ...
def validate_target_response_schema(df: pd.DataFrame, dataset_name: str, config: AACDRPipelineConfig) -> None: ...
def validate_omics_schema(df: pd.DataFrame, sample_col: str, name: str) -> None: ...
def validate_binary_labels(df: pd.DataFrame, label_col: str, name: str) -> None: ...
```

Testing:

- expected schemas;
- missing columns;
- drug normalization;
- TCGA barcode first-three-segment normalization;
- invalid labels;
- empty dataset handling.

### 6.3 Data IO Module

Suggested path:

```text
code/aacdr_pipeline/data_io.py
```

Responsibilities:

- Load source omics.
- Load source response.
- Load target omics.
- Load three TCGA response datasets.
- Load drug SMILES / feature data.

Main dataclass:

```python
@dataclass
class RawAACDRInputs:
    source_omics: pd.DataFrame
    source_response: pd.DataFrame
    target_omics: pd.DataFrame
    target_primary_response: pd.DataFrame
    target_only_response: pd.DataFrame
    target_auxiliary_response: pd.DataFrame
    drug_feature_table: pd.DataFrame
```

Main functions:

```python
def load_raw_inputs(config: AACDRPipelineConfig) -> RawAACDRInputs: ...
def load_csv_required(path: str, name: str) -> pd.DataFrame: ...
def load_csv_optional(path: str | None, name: str) -> pd.DataFrame | None: ...
```

### 6.4 Feature Alignment Module

Suggested path:

```text
code/aacdr_pipeline/features.py
```

Responsibilities:

- Align source and target omics features.
- Keep only common gene / feature columns.
- Preserve sample ID columns.
- Export feature alignment report.

Main dataclass/function:

```python
@dataclass
class AlignedOmics:
    source_omics: pd.DataFrame
    target_omics: pd.DataFrame
    feature_names: list[str]
    report: pd.DataFrame

def align_source_target_features(
    source_omics: pd.DataFrame,
    target_omics: pd.DataFrame,
    source_sample_col: str,
    target_sample_col: str,
) -> AlignedOmics: ...
```

### 6.5 Drug Index / Drug Metadata Module

Suggested path:

```text
code/aacdr_pipeline/drug_index.py
```

Responsibilities:

- Build final drug list.
- Assign deterministic drug index.
- Track drug origin across source and target eval datasets.
- Mark target-eval-only drugs.
- Export drug reports.

Final drug list:

```text
source drugs ∪ primary TCGA drugs ∪ target_only TCGA drugs ∪ auxiliary TCGA drugs
```

Main objects/functions:

```python
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

def build_final_drug_index(... ) -> DrugIndex: ...
def build_drug_metadata(... ) -> DrugMetadata: ...
```

Required report fields:

```text
drug_id
drug_index
in_source
in_target_primary
in_target_only
in_target_auxiliary
in_any_target_eval
has_supervised_source_label
is_target_eval_only
```

### 6.6 Source Split / Fold Module

Suggested path:

```text
code/aacdr_pipeline/splits.py
```

Responsibilities:

- Create fixed source test split.
- Create 5-fold train/validation splits.
- Enforce grouped split by `Sample_ID`.
- Write source split manifest and fold summary.
- Validate no group leakage.

Main objects/functions:

```python
@dataclass
class FoldSplit:
    fold: int
    train_sample_ids: list[str]
    valid_sample_ids: list[str]
    source_test_sample_ids: list[str]

@dataclass
class SourceSplits:
    folds: list[FoldSplit]
    source_split_report: pd.DataFrame
    fold_summary: pd.DataFrame

def build_grouped_source_splits(
    source_response: pd.DataFrame,
    sample_col: str = "Sample_ID",
    n_splits: int = 5,
    source_test_size: float = 0.10,
    seed: int = 0,
) -> SourceSplits: ...

def validate_no_sample_id_leakage(splits: SourceSplits) -> None: ...
```

### 6.7 Target Evaluation Dataset Module

Suggested path:

```text
code/aacdr_pipeline/target_eval.py
```

Responsibilities:

- Prepare TCGA primary, target-only, and auxiliary evaluation datasets.
- Align TCGA responses to target omics.
- Build evaluation matrices or long-form evaluation tables.
- Produce dataset reports.

Main objects/functions:

```python
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

def prepare_target_eval_dataset(... ) -> TargetEvalDataset: ...
def prepare_all_target_eval_datasets(... ) -> TargetEvalBundle: ...
```

Matching rule:

- Response sample column: `Patient_id`.
- Target omics sample key: normalized by TCGA first-three-segment rule.
- Drug column: `drug_name`.
- Label column: `Label`.

Dataset report fields:

```text
eval_dataset
input_rows
usable_rows
skipped_rows_sample_not_in_target_omics
skipped_rows_drug_not_in_final_index
n_observed_patients
n_observed_drugs
label_0_count
label_1_count
```

### 6.8 Drug Graph / Feature Adapter Module

Suggested path:

```text
code/aacdr_pipeline/drug_graph_adapter.py
```

Responsibilities:

- Convert final drug index into AACDR-compatible drug representations.
- Use AACDR’s original drug graph / feature logic first.
- Compare compatibility with drug-DCF SMILES-to-graph logic only as a reference.
- Handle target-eval-only drugs.
- Export graph availability reports.

Required behavior:

- Prefer AACDR original representation format.
- Use fallback only if compatible with AACDR model input.
- If target-only drug cannot be represented, stop and ask user.
- Do not silently drop target-only drugs.

Reports:

```text
drug_graph_availability_report.csv
drug_graph_edge_report.csv
```

### 6.9 Dataset Adapter Module

Suggested path:

```text
code/aacdr_pipeline/datasets.py
```

Responsibilities:

- Build AACDR-compatible datasets from standardized DataFrames.
- Hide pandas details from the trainer.
- Support source train/validation/test.
- Support target unlabeled omics input.
- Support target eval prediction inputs.

Main object/function:

```python
@dataclass
class FoldDataBundle:
    fold: int
    source_train_dataset: Any
    source_valid_dataset: Any
    source_test_dataset: Any
    target_unlabeled_dataset: Any | None
    target_eval_primary_dataset: Any
    target_eval_target_only_dataset: Any
    target_eval_auxiliary_dataset: Any

def build_fold_data_bundle(... ) -> FoldDataBundle: ...
```

### 6.10 AACDR Model Adapter Module

Suggested path:

```text
code/aacdr_pipeline/model_adapter.py
```

Responsibilities:

- Wrap existing AACDR model classes.
- Provide stable prediction API.
- Provide latent extraction API.
- Avoid modifying original model classes unless required.

Main functions:

```python
def build_aacdr_model(config: AACDRPipelineConfig, drug_bundle: DrugRepresentationBundle) -> Any: ...
def predict_batch(model: Any, batch: Any, config: AACDRPipelineConfig) -> np.ndarray: ...
def extract_omics_latent(model: Any, dataset: Any, config: AACDRPipelineConfig) -> pd.DataFrame: ...
def save_checkpoint(model: Any, path: str) -> None: ...
def load_checkpoint(path: str, config: AACDRPipelineConfig) -> Any: ...
```

### 6.11 AACDR Trainer Wrapper Module

Suggested path:

```text
code/aacdr_pipeline/trainer_wrapper.py
```

Responsibilities:

- Run one fold of AACDR training.
- Preserve original AACDR training concept.
- Ensure source validation is used for model selection.
- Prevent TCGA labels from entering training.
- Export train log and model selection report.

Main object/function:

```python
@dataclass
class FoldTrainingResult:
    fold: int
    best_model: Any
    best_checkpoint_path: str
    best_epoch: int
    train_log: pd.DataFrame
    selection_report: dict

def train_one_fold(fold_data: FoldDataBundle, model: Any, config: AACDRPipelineConfig) -> FoldTrainingResult: ...
```

Required behavior:

- Source train labels are used for supervised training.
- Source validation is used for model selection.
- Target omics may be used unlabeled if AACDR requires it.
- TCGA response labels are not used for training.

### 6.12 Prediction / Evaluation Module

Suggested path:

```text
code/aacdr_pipeline/evaluation.py
```

Responsibilities:

- Predict source test.
- Predict TCGA primary.
- Predict TCGA target-only.
- Predict TCGA auxiliary.
- Create long-form prediction tables.
- Compute per-drug metrics.
- Compute summary metrics.
- Aggregate each dataset independently across folds.

Fold-level outputs:

```text
source_test_prediction_results.csv
source_test_metrics_per_drug.csv
source_test_metrics_summary.csv
target_primary_prediction_results.csv
target_primary_metrics_per_drug.csv
target_primary_metrics_summary.csv
target_only_prediction_results.csv
target_only_metrics_per_drug.csv
target_only_metrics_summary.csv
target_auxiliary_prediction_results.csv
target_auxiliary_metrics_per_drug.csv
target_auxiliary_metrics_summary.csv
```

Main functions:

```python
def predict_source_test(...) -> pd.DataFrame: ...
def predict_target_eval_dataset(name: str, ...) -> pd.DataFrame: ...
def compute_metrics_per_drug(prediction_df: pd.DataFrame) -> pd.DataFrame: ...
def compute_metrics_summary(per_drug_df: pd.DataFrame, prediction_df: pd.DataFrame) -> pd.DataFrame: ...
def aggregate_metrics_across_folds(dataset_name: str, fold_metric_frames: list[pd.DataFrame]) -> dict[str, pd.DataFrame]: ...
```

### 6.13 Latent Export Module

Suggested path:

```text
code/aacdr_pipeline/latent.py
```

Responsibilities:

- Extract source omics latent representations.
- Extract target omics latent representations.
- Save latent representations for later t-SNE, FID, and k-means.

Outputs:

```text
fold_i/source_latent_representation.pkl
fold_i/target_latent_representation.pkl
```

Latent files should include:

```text
sample_id
domain
split
fold
seed
cancer_type
latent vector
```

### 6.14 t-SNE Module

Suggested path:

```text
code/aacdr_pipeline/tsne.py
```

Responsibilities:

- Run t-SNE on source and target omics latent.
- Create domain mixing visualization.
- Create cancer type visualization.

Outputs:

```text
fold_i/tsne_domain_mixing.png
fold_i/tsne_cancer_type.png
```

Optional CSV outputs:

```text
fold_i/tsne_domain_mixing.csv
fold_i/tsne_cancer_type.csv
```

### 6.15 FID / Latent Distribution Module

Suggested path:

```text
code/aacdr_pipeline/fid.py
```

Responsibilities:

- Compute source-target latent distribution distance.
- Use source and target omics latent representations.
- Export fold-level and run-level summaries.

Outputs:

```text
fold_i/latent_distribution_metrics.csv
latent_metrics_summary.csv
```

### 6.16 KMeans Cancer Type Module

Suggested path:

```text
code/aacdr_pipeline/kmeans.py
```

Responsibilities:

- Run k-means on high-dimensional latent features.
- Evaluate cancer type clustering.
- Export fold-level and across-fold reports.

Outputs:

```text
fold_i/kmeans_cancer_type_metrics.csv
kmeans_cancer_type_summary.csv
kmeans_cancer_type_fold_mean_std.csv
```

Required behavior:

- Use high-dimensional latent features.
- Do not use t-SNE coordinates as the primary clustering input.
- Main analysis target is cancer type clustering.

### 6.17 Report / Export Module

Suggested path:

```text
code/aacdr_pipeline/reports.py
```

Responsibilities:

- Write all CSV, JSON, PNG, and pickle outputs.
- Manage output directory layout.
- Provide standardized fold path utilities.
- Aggregate per-fold outputs.

### 6.18 Pipeline Orchestrator Module

Suggested path:

```text
code/aacdr_pipeline/run.py
```

Responsibilities:

- Coordinate the full workflow.
- Keep the entry point thin.
- Call modules in correct order.
- Stop on validation errors.

Main function:

```python
def run_pipeline(config: AACDRPipelineConfig) -> None: ...
```

---

## 7. Data Contracts

### 7.1 Source response contract

Default path:

```text
/workspace/DAPL-master/data/GDSC2_fitted_dose_response_MaxScreen_raw.csv
```

Required logical fields:

```text
Sample_ID
drug name column from CLI
label column from CLI
```

Default source sample column:

```text
Sample_ID
```

The source drug and label columns are CLI-configurable. If the implementation cannot determine valid defaults from the local data, it must stop and ask.

### 7.2 Source omics contract

Default path:

```text
/workspace/DAPL-master/data/pretrain_ccle.csv
```

Required logical fields:

```text
sample ID
gene / feature columns
```

The source omics sample column should be configurable if the local file does not use `Sample_ID`.

### 7.3 Target omics contract

Default path:

```text
/workspace/DAPL-master/data/TCGA/pretrain_tcga.csv
```

Target omics sample ID should be normalized to patient-level key using the first three TCGA barcode segments when matching response data.

### 7.4 TCGA response contract

Default paths:

```text
primary = /workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv
target_only = /workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv
auxiliary = /workspace/DAPL-master/data/TCGA/TCGA_drug_response_from_DAPL.csv
```

Required columns:

```text
Patient_id
drug_name
Label
```

### 7.5 Drug representation contract

Default path:

```text
/workspace/DAPL-master/data/GDSC_drug_merge_pubchem_dropNA_MACCS.csv
```

The final drug list includes drugs from source and all three TCGA eval response datasets. AACDR-compatible drug representation must be available for all final drugs.

If a target-only drug representation is missing, use an AACDR-compatible fallback method. If no compatible fallback can be determined, stop and ask the user.

---

## 8. CLI Contract

A minimal CLI should include:

```bash
python aacdr_multilabel_hyper_main.py \
  --source_omics_path /workspace/DAPL-master/data/pretrain_ccle.csv \
  --source_response_path /workspace/DAPL-master/data/GDSC2_fitted_dose_response_MaxScreen_raw.csv \
  --target_omics_path /workspace/DAPL-master/data/TCGA/pretrain_tcga.csv \
  --target_eval_primary_response_path /workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv \
  --target_eval_target_only_response_path /workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv \
  --target_eval_aux_response_path /workspace/DAPL-master/data/TCGA/TCGA_drug_response_from_DAPL.csv \
  --drug_smiles_path /workspace/DAPL-master/data/GDSC_drug_merge_pubchem_dropNA_MACCS.csv \
  --source_sample_col Sample_ID \
  --source_drug_col <SOURCE_DRUG_COLUMN> \
  --source_label_col <SOURCE_LABEL_COLUMN> \
  --n_splits 5 \
  --source_test_size 0.10 \
  --seed 0 \
  --output_dir outputs_aacdr_eval3
```

`<SOURCE_DRUG_COLUMN>` and `<SOURCE_LABEL_COLUMN>` must be confirmed from the local source response file before implementation. If these columns cannot be inferred safely, stop and ask the user.

---

## 9. Source Split Strategy: 5-fold + 10% Test

The source split strategy must be implemented as grouped split by `Sample_ID`.

Algorithm:

1. Read source response.
2. Extract unique `Sample_ID`.
3. Shuffle unique IDs using `seed`.
4. Split 10% of IDs into source test.
5. Use remaining 90% IDs for 5-fold cross-validation.
6. For each fold:
   - validation IDs = one fold;
   - training IDs = other four folds;
   - source test IDs = fixed 10% test IDs.

Validation:

```text
train_sample_ids ∩ valid_sample_ids = empty
train_sample_ids ∩ source_test_sample_ids = empty
valid_sample_ids ∩ source_test_sample_ids = empty
```

Reports:

```text
source_split.csv
fold_summary.csv
```

`source_split.csv` should contain:

```text
Sample_ID
split
fold
```

`fold_summary.csv` should contain:

```text
fold
n_train_sample_ids
n_valid_sample_ids
n_source_test_sample_ids
n_train_rows
n_valid_rows
n_source_test_rows
```

---

## 10. TCGA Evaluation Flow

For each fold, after training and model selection:

1. Predict TCGA primary response rows.
2. Predict TCGA target-only response rows.
3. Predict TCGA auxiliary response rows.
4. Write independent prediction tables.
5. Compute independent per-drug metrics.
6. Compute independent summary metrics.

TCGA response labels are accessed only inside evaluation functions after training is complete. No TCGA label is passed into trainer.

---

## 11. Drug List and Drug Graph Flow

Final drug list:

```text
source ∪ primary ∪ target_only ∪ auxiliary
```

Each drug receives:

```text
in_source
in_target_primary
in_target_only
in_target_auxiliary
in_any_target_eval
has_supervised_source_label
is_target_eval_only
```

Required reports:

```text
drug_list.csv
drug_availability_report.csv
drug_graph_availability_report.csv
drug_graph_edge_report.csv
target_eval_zero_shot_drug_report.csv
```

AACDR-compatible drug representation must be built for every final drug. If a drug appears only in TCGA evaluation, it still needs a valid representation.

---

## 12. Training Flow

Training uses:

- source train response labels;
- source train omics;
- possibly target omics as unlabeled target data if AACDR original training uses it.

Validation uses:

- source validation response labels;
- source validation omics.

Source test data is held out and used only after training.

TCGA labels are excluded from:

- training;
- validation;
- early stopping;
- model selection.

Select best model by source validation only.

Model run count:

```text
5 folds × 1 seed
```

---

## 13. Prediction and Metrics Flow

Required source test outputs per fold:

```text
source_test_prediction_results.csv
source_test_metrics_per_drug.csv
source_test_metrics_summary.csv
```

Required TCGA primary outputs per fold:

```text
target_primary_prediction_results.csv
target_primary_metrics_per_drug.csv
target_primary_metrics_summary.csv
```

Required TCGA target-only outputs per fold:

```text
target_only_prediction_results.csv
target_only_metrics_per_drug.csv
target_only_metrics_summary.csv
```

Required TCGA auxiliary outputs per fold:

```text
target_auxiliary_prediction_results.csv
target_auxiliary_metrics_per_drug.csv
target_auxiliary_metrics_summary.csv
```

Prediction table schema:

```text
sample_id
drug_id
drug_index
domain
split
eval_dataset
ground_truth
pred_score
probability
pred_label
confidence
fold
seed
task_type
cancer_type
has_supervised_source_label
is_target_eval_only
```

Required per-drug metrics:

```text
AUROC
AUPRC
accuracy
balanced_accuracy
F1
precision
recall
n_samples
n_positive
n_negative
```

If only one class is present for a drug, keep the drug row and set:

```text
AUROC = NaN
AUPRC = NaN
```

---

## 14. Latent / t-SNE / FID / KMeans Flow

After training each fold, export:

```text
source_latent_representation.pkl
target_latent_representation.pkl
```

Latent scope is omics-level only. No sample-drug integrated latent in this revision.

t-SNE uses source and target omics latent and outputs:

```text
tsne_domain_mixing.png
tsne_cancer_type.png
```

FID / latent distribution metrics use source vs target latent and output:

```text
latent_distribution_metrics.csv
latent_metrics_summary.csv
```

k-means uses high-dimensional latent. The target is cancer type clustering.

Outputs:

```text
kmeans_cancer_type_metrics.csv
kmeans_cancer_type_summary.csv
kmeans_cancer_type_fold_mean_std.csv
```

---

## 15. Output Directory Layout

```text
output_dir/
  config.json
  run_manifest.json

  data_alignment_report.csv
  sample_filtering_report.csv
  feature_alignment_report.csv
  cancer_type_summary.csv

  drug_list.csv
  drug_availability_report.csv
  drug_graph_availability_report.csv
  drug_graph_edge_report.csv
  target_eval_dataset_report.csv
  target_eval_zero_shot_drug_report.csv

  source_split.csv
  fold_summary.csv

  fold_0/
    best_model.pt
    checkpoint_load_report.json
    selection_report.json
    train_log.csv

    source_test_prediction_results.csv
    source_test_metrics_per_drug.csv
    source_test_metrics_summary.csv

    target_primary_prediction_results.csv
    target_primary_metrics_per_drug.csv
    target_primary_metrics_summary.csv

    target_only_prediction_results.csv
    target_only_metrics_per_drug.csv
    target_only_metrics_summary.csv

    target_auxiliary_prediction_results.csv
    target_auxiliary_metrics_per_drug.csv
    target_auxiliary_metrics_summary.csv

    source_latent_representation.pkl
    target_latent_representation.pkl
    latent_distribution_metrics.csv
    kmeans_cancer_type_metrics.csv

    tsne_domain_mixing.png
    tsne_cancer_type.png

  fold_1/
  fold_2/
  fold_3/
  fold_4/

  source_test_metrics_summary_across_folds.csv
  source_test_metrics_summary_fold_mean_std.csv
  source_test_metrics_per_drug_fold_mean_std.csv

  target_primary_metrics_summary_across_folds.csv
  target_primary_metrics_summary_fold_mean_std.csv
  target_primary_metrics_per_drug_across_folds.csv
  target_primary_metrics_per_drug_fold_mean_std.csv

  target_only_metrics_summary_across_folds.csv
  target_only_metrics_summary_fold_mean_std.csv
  target_only_metrics_per_drug_across_folds.csv
  target_only_metrics_per_drug_fold_mean_std.csv

  target_auxiliary_metrics_summary_across_folds.csv
  target_auxiliary_metrics_summary_fold_mean_std.csv
  target_auxiliary_metrics_per_drug_across_folds.csv
  target_auxiliary_metrics_per_drug_fold_mean_std.csv

  latent_metrics_summary.csv
  kmeans_cancer_type_summary.csv
  kmeans_cancer_type_fold_mean_std.csv
```

Do not output combined target by-dataset summary tables in this revision unless explicitly requested later.

---

## 16. Leakage Prevention Rules

The pipeline must enforce:

1. TCGA eval labels do not enter training loss.
2. TCGA eval labels do not enter domain adaptation loss.
3. TCGA eval labels do not enter validation.
4. TCGA eval labels do not enter early stopping.
5. TCGA eval metrics do not enter model selection.
6. Source test labels do not enter training or model selection.
7. A `Sample_ID` cannot appear in more than one source split.
8. t-SNE, FID, and k-means cannot affect model selection.
9. Target-eval-only drugs must be explicitly flagged.

---

## 17. Error Handling and Validation

The pipeline should fail early when:

- required input files are missing;
- required columns are missing;
- source split cannot produce 5 folds;
- `Sample_ID` leakage is detected;
- source and target omics have no overlapping features;
- TCGA response samples cannot be matched to target omics;
- a final drug lacks AACDR-compatible representation and no compatible fallback exists;
- source drug or label columns cannot be safely inferred from local data.

The pipeline must not silently drop target-only drugs.

---

## 18. Testing Strategy

### 18.1 Unit tests

Required test areas:

```text
config defaults and overrides
schema validation
drug name normalization
TCGA patient ID normalization
feature alignment
drug index union
target-eval-only drug flags
grouped source split by Sample_ID
no source split leakage
TCGA eval dataset preparation
AACDR drug representation coverage
prediction table schema
per-drug metrics
single-class drug metrics
latent export
k-means cancer type output
```

### 18.2 Integration tests

Use a small synthetic fixture to verify:

1. Source split creates 5 folds.
2. Fixed 10% source test is held out.
3. Same `Sample_ID` does not cross split.
4. Three TCGA eval datasets produce three independent output groups.
5. TCGA labels are not passed to trainer.
6. Latent outputs are created after training.
7. t-SNE, FID, and k-means outputs are created.
8. Required output files exist.

### 18.3 Smoke test

A minimal run should support:

```bash
python aacdr_multilabel_hyper_main.py \
  --source_omics_path /workspace/DAPL-master/data/pretrain_ccle.csv \
  --source_response_path /workspace/DAPL-master/data/GDSC2_fitted_dose_response_MaxScreen_raw.csv \
  --target_omics_path /workspace/DAPL-master/data/TCGA/pretrain_tcga.csv \
  --target_eval_primary_response_path /workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv \
  --target_eval_target_only_response_path /workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv \
  --target_eval_aux_response_path /workspace/DAPL-master/data/TCGA/TCGA_drug_response_from_DAPL.csv \
  --drug_smiles_path /workspace/DAPL-master/data/GDSC_drug_merge_pubchem_dropNA_MACCS.csv \
  --source_sample_col Sample_ID \
  --source_drug_col <SOURCE_DRUG_COLUMN> \
  --source_label_col <SOURCE_LABEL_COLUMN> \
  --n_splits 5 \
  --source_test_size 0.10 \
  --seed 0 \
  --output_dir outputs_aacdr_eval3
```

Before implementing this command, `<SOURCE_DRUG_COLUMN>` and `<SOURCE_LABEL_COLUMN>` must be confirmed from the actual source response CSV.

---

## 19. Migration Plan

### Phase 1: Documentation

- Create `docs/proposal.md`.
- Create `docs/design.md`.

### Phase 2: Non-invasive pipeline foundation

- Add config module.
- Add data loading module.
- Add schema validation module.
- Add split module.
- Add report directory utilities.

### Phase 3: AACDR adapters

- Add dataset adapter.
- Add drug representation adapter.
- Add model adapter.
- Add trainer wrapper.

### Phase 4: Evaluation and reporting

- Add source test evaluation.
- Add TCGA primary evaluation.
- Add TCGA target-only evaluation.
- Add TCGA auxiliary evaluation.
- Add fold aggregation.

### Phase 5: Latent analysis

- Add latent export.
- Add t-SNE.
- Add FID / latent distribution metrics.
- Add k-means cancer type analysis.

### Phase 6: Tests and smoke run

- Add unit tests.
- Add integration tests.
- Add smoke command.

---

## 20. Acceptance Criteria

The design is accepted if it satisfies the following:

1. It preserves AACDR model and training code as much as possible.
2. It defines a clear single-entry pipeline.
3. It defines default DAPL paths with CLI override.
4. It uses source 10% test and remaining 90% 5-fold.
5. It groups source split by `Sample_ID`.
6. It prevents `Sample_ID` leakage across splits.
7. It uses TCGA omics as target input where AACDR requires it.
8. It prohibits TCGA labels from training and model selection.
9. It builds final drug list from source, primary, target_only, and auxiliary drugs.
10. It handles target-only drugs with AACDR-compatible drug representation logic.
11. It outputs independent primary, target_only, and auxiliary prediction results.
12. It outputs independent primary, target_only, and auxiliary per-drug metrics.
13. It outputs independent primary, target_only, and auxiliary summary metrics.
14. It does not require combined target by-dataset summary outputs.
15. It exports source and target omics latent representations.
16. It runs t-SNE on omics latent.
17. It computes source-target latent distribution metrics.
18. It runs k-means cancer type clustering on high-dimensional latent.
19. It defines testable modules with low coupling.
20. It requires stopping and asking the user if any implementation detail is unclear.
