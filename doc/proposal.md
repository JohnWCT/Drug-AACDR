# Drug-AACDR New Data Input/Output Proposal

## 1. Background

This proposal defines the required changes for the local `Drug-AACDR` project so that it can use the newer DAPL / Drug-TransDRP / drug-DCF style data input and output workflow while preserving the original AACDR model and training concept as much as possible.

The intended revision should support:

- source CCLE/GDSC-style omics and response input;
- TCGA target omics input;
- three independent TCGA evaluation response files;
- source 10% fixed test split;
- 5-fold train/validation split on the remaining 90% source data;
- grouped split by cell line ID `Sample_ID`;
- final drug list built from source and all three TCGA evaluation datasets;
- independent prediction and metrics outputs for each TCGA evaluation dataset;
- source/target omics latent export;
- t-SNE, FID / latent distribution metrics, and k-means cancer type clustering after training.

This document is the proposal-level contract for the later architecture design and implementation.

---

## 2. Goals

### 2.1 Preserve AACDR model and training behavior

The implementation should preserve the original AACDR model and training logic as much as possible. New code should preferably be added as wrappers, adapters, orchestration modules, or reporting modules rather than replacing the core model.

Expected implementation style:

- keep existing AACDR model classes;
- keep existing AACDR training objective where possible;
- add a new pipeline layer for data loading, split creation, target evaluation, and report export;
- avoid unnecessary replacement with Drug-TransDRP or drug-DCF model logic.

### 2.2 Adopt standardized data input and output

The new pipeline should accept standardized input files, using default paths under `/workspace/DAPL-master/data`, while allowing CLI override.

The workflow should produce explicit fold-level and run-level outputs for:

- source test results;
- TCGA primary results;
- TCGA target-only results;
- TCGA auxiliary results;
- split and alignment reports;
- drug availability reports;
- latent, t-SNE, FID, and k-means reports.

### 2.3 Use source 10% test + 5-fold train/validation

The source training split strategy should be:

1. Group source response rows by `Sample_ID`.
2. Split out 10% of source cell lines as a fixed source test set.
3. Use the remaining 90% source cell lines for 5-fold cross-validation.
4. In each fold, one fold is validation and the other four folds are training.
5. The fixed source test set is never used for training, validation, early stopping, or model selection.

The split must be grouped by `Sample_ID` so all response rows for a cell line stay in the same split.

### 2.4 Support three independent TCGA evaluation datasets

The target TCGA evaluation datasets are:

| Generic eval key | Default response path |
|---|---|
| `primary` | `/workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv` |
| `target_only` | `/workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv` |
| `auxiliary` | `/workspace/DAPL-master/data/TCGA/TCGA_drug_response_from_DAPL.csv` |

All three use the same target omics file by default:

```text
/workspace/DAPL-master/data/TCGA/pretrain_tcga.csv
```

The three TCGA response files are evaluation-only. Their labels must not be used in training, domain adaptation loss, validation, early stopping, or model selection.

### 2.5 Produce independent TCGA score and metric tables

Each TCGA evaluation dataset must produce its own independent prediction and metric files.

Required fold-level outputs:

```text
fold_i/target_primary_prediction_results.csv
fold_i/target_primary_metrics_per_drug.csv
fold_i/target_primary_metrics_summary.csv

fold_i/target_only_prediction_results.csv
fold_i/target_only_metrics_per_drug.csv
fold_i/target_only_metrics_summary.csv

fold_i/target_auxiliary_prediction_results.csv
fold_i/target_auxiliary_metrics_per_drug.csv
fold_i/target_auxiliary_metrics_summary.csv
```

The target-only dataset should use the clean prefix `target_only_*`; do not use `target_target_only_*`.

Combined by-dataset target summary tables are not required in this revision. The three independent outputs are the primary reporting contract.

### 2.6 Export omics latent analysis outputs

After the final model for each fold is trained, export latent representations for source and target omics.

Latent analysis should be based on omics-level latent representations only:

- source omics latent;
- target TCGA omics latent.

Do not create sample-drug integrated latent representations in this revision.

Required downstream latent analyses:

- t-SNE visualization;
- source-target latent distribution metrics / FID;
- k-means cancer type clustering.

k-means should use high-dimensional latent features as input. t-SNE should be used only for visualization.

---

## 3. Non-goals

The following are explicitly out of scope:

1. Replacing the AACDR model with Drug-TransDRP or drug-DCF models.
2. Rewriting the AACDR trainer from scratch.
3. Introducing Drug-TransDRP-style pretrain/fine-tune separation.
4. Using TCGA response labels for training.
5. Using TCGA evaluation metrics for model selection.
6. Using source test rows for model selection.
7. Producing merged target evaluation tables instead of independent target dataset outputs.
8. Performing raw TCGA preprocessing.
9. Reintroducing AACDR multi-initialization or `model_nums` ensemble as the default behavior.
10. Running k-means on t-SNE coordinates as the primary clustering input.

---

## 4. Confirmed Decisions

### 4.1 Project-level decisions

| Topic | Decision |
|---|---|
| Documentation flow | Create `docs/proposal.md`, then `docs/design.md` |
| Implementation style | Modular architecture, but preserve AACDR model/training code |
| Entry point | Single new entry point with mode/stage support |
| Pretrain/fine-tune split | Do not force Drug-TransDRP two-stage split |
| Model repetitions | Default is `5 folds × 1 seed` |
| Unclear behavior | Stop and ask the user; do not guess |

### 4.2 Data path decisions

| Data type | Default path | CLI override |
|---|---|---|
| Source omics | `/workspace/DAPL-master/data/pretrain_ccle.csv` | yes |
| Source response | `/workspace/DAPL-master/data/GDSC2_fitted_dose_response_MaxScreen_raw.csv` | yes |
| Drug SMILES / drug features | `/workspace/DAPL-master/data/GDSC_drug_merge_pubchem_dropNA_MACCS.csv` | yes |
| Target omics | `/workspace/DAPL-master/data/TCGA/pretrain_tcga.csv` | yes |
| TCGA primary response | `/workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv` | yes |
| TCGA target-only response | `/workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv` | yes |
| TCGA auxiliary response | `/workspace/DAPL-master/data/TCGA/TCGA_drug_response_from_DAPL.csv` | yes |

### 4.3 Input schema decisions

Source response columns are configured by CLI.

Default source sample column:

```text
Sample_ID
```

TCGA response columns are fixed:

```text
Patient_id
drug_name
Label
```

Drug name normalization:

```python
str(drug_name).strip().lower()
```

TCGA patient matching uses the drug-DCF TCGA barcode rule:

```text
TCGA-XX-YYYY-ZZZ... -> TCGA-XX-YYYY
```

### 4.4 Split decisions

| Topic | Decision |
|---|---|
| Source test fraction | 10% |
| Cross-validation | 5 folds |
| Split group | `Sample_ID` |
| Same cell line across splits | Forbidden |
| Validation fold | One fold from remaining 90% |
| Source test usage | Evaluation only |
| Stratification | Not required; grouped random split by `Sample_ID` |

### 4.5 Target usage decisions

TCGA omics may be used as unlabeled target input if this is part of AACDR’s original domain adaptation or reconstruction flow.

TCGA response labels are evaluation-only.

Forbidden:

- TCGA labels in supervised loss;
- TCGA labels in domain adaptation loss;
- TCGA labels in early stopping;
- TCGA evaluation metrics in model selection.

---

## 5. Input Data Contract

### 5.1 Source omics

Default path:

```text
/workspace/DAPL-master/data/pretrain_ccle.csv
```

Expected role:

- Source omics matrix for cell line expression / feature input.
- Used for source training, source validation, source test prediction, and source latent export.

Required behavior:

- Allow CLI override via `--source_omics_path`.
- Allow CLI configuration of sample ID column if needed.
- Align feature columns with target omics.
- Emit feature alignment report.

### 5.2 Source response

Default path:

```text
/workspace/DAPL-master/data/GDSC2_fitted_dose_response_MaxScreen_raw.csv
```

Expected role:

- Source supervised drug response labels.
- Used to build source train/validation/test datasets.
- Used to define source drugs.

Required behavior:

- Allow CLI override via `--source_response_path`.
- Source sample column default is `Sample_ID`.
- Source drug and label columns should be CLI-configurable.
- All rows sharing the same `Sample_ID` must stay in the same split.

### 5.3 Drug SMILES / drug feature table

Default path:

```text
/workspace/DAPL-master/data/GDSC_drug_merge_pubchem_dropNA_MACCS.csv
```

Expected role:

- Provide drug-level features or graph construction inputs.
- Used to prepare AACDR-compatible drug representations.

Required behavior:

- Allow CLI override via `--drug_smiles_path` or equivalent drug feature argument.
- Prefer AACDR’s original drug graph / drug feature construction method.
- If target-only drugs lack AACDR-compatible representation, construct fallback representation using AACDR-consistent logic.
- If fallback is impossible, stop and ask the user rather than guessing.

### 5.4 Target omics

Default path:

```text
/workspace/DAPL-master/data/TCGA/pretrain_tcga.csv
```

Expected role:

- TCGA target omics.
- May be used as unlabeled target input for AACDR’s original domain adaptation flow.
- Used for TCGA prediction and target latent export.

Required behavior:

- Allow CLI override via `--target_omics_path`.
- Use already prepared TCGA expression data.
- Do not perform raw TCGA preprocessing.
- Match target response patients using TCGA patient key normalization.

### 5.5 TCGA response datasets

Three target response datasets are required.

Primary:

```text
/workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv
```

Target-only:

```text
/workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv
```

Auxiliary:

```text
/workspace/DAPL-master/data/TCGA/TCGA_drug_response_from_DAPL.csv
```

Each must contain:

```text
Patient_id
drug_name
Label
```

These labels are used only for prediction evaluation.

---

## 6. Source Split and Fold Strategy

The source split strategy is group-based by `Sample_ID`.

### 6.1 Step 1: Build source sample groups

From source response, collect all unique `Sample_ID` values. Every response row belongs to a group defined by `Sample_ID`.

### 6.2 Step 2: Fixed 10% source test

Randomly split 10% of unique `Sample_ID` groups into source test.

The source test set:

- is fixed across all folds;
- is never used for training;
- is never used for validation;
- is never used for model selection;
- is used only for final source test evaluation.

### 6.3 Step 3: 5-fold train/validation split

Use the remaining 90% `Sample_ID` groups for 5-fold cross-validation.

For each fold:

- validation set = one fold of source cell lines;
- training set = the other four folds;
- test set = fixed 10% source test set.

### 6.4 Leakage prevention

The same `Sample_ID` must never appear in more than one of:

- training;
- validation;
- source test.

This must be validated and reported in `source_split.csv` and `fold_summary.csv`.

---

## 7. TCGA Evaluation Strategy

### 7.1 Evaluation datasets

The three TCGA evaluation datasets are:

```text
primary
target_only
auxiliary
```

Each dataset is evaluated independently.

### 7.2 Target omics usage

TCGA omics may be used as unlabeled target input during AACDR training if AACDR’s original training flow requires target omics. TCGA response labels are not allowed to influence training.

### 7.3 Independent outputs

Each TCGA evaluation dataset must output:

- prediction results;
- per-drug metrics;
- summary metrics.

No combined by-dataset target evaluation summary is required for this revision.

---

## 8. Drug List and Drug Graph Strategy

### 8.1 Final drug list

The final drug list is:

```text
source drugs
∪ primary TCGA drugs
∪ target_only TCGA drugs
∪ auxiliary TCGA drugs
```

All drug names must be normalized with:

```python
str(drug_name).strip().lower()
```

### 8.2 Drug index

The pipeline must build a stable deterministic `drug_id -> drug_index` mapping, preferably sorted by normalized `drug_id`.

### 8.3 Drug reports

Required reports:

```text
drug_list.csv
drug_availability_report.csv
target_eval_zero_shot_drug_report.csv
```

Required fields include:

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

### 8.4 Target-only drugs

For drugs present in TCGA evaluation but absent from source response:

```text
has_supervised_source_label = False
is_target_eval_only = True
```

These drugs are zero-shot / target-eval-only from the supervised source perspective.

### 8.5 Drug graph / drug feature handling

AACDR must prioritize its own original drug graph or drug feature construction method.

If target-only drugs lack a usable AACDR-compatible representation, the implementation should build fallback graph / feature using AACDR-consistent logic.

The implementation must not blindly replace AACDR drug representation with drug-DCF’s GIN / SMILES flow unless this is confirmed to be compatible.

If compatibility cannot be determined, implementation must stop and ask the user.

---

## 9. Training Usage Rules

### 9.1 Source data

Source response labels are used for supervised AACDR training. Source training and validation are determined by the 5-fold group split. Source test is never used for training or model selection.

### 9.2 Target data

Target TCGA omics may be used as unlabeled target data if required by AACDR’s original training logic.

TCGA response labels must not be used in:

- training loss;
- domain adaptation loss;
- reconstruction target;
- validation;
- early stopping;
- model selection.

### 9.3 Model selection

Model selection must be based only on source validation. TCGA evaluation metrics and source test metrics must not affect model selection.

### 9.4 Model repetitions

The revised default training plan is:

```text
5 folds × 1 seed
```

Do not preserve AACDR multi-initialization / `model_nums` as the default behavior. If the original code requires `model_nums`, it should be set to one or wrapped with a single-run mode.

---

## 10. Output Contract

### 10.1 Global outputs

Required global outputs:

```text
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
```

### 10.2 Fold-level source test outputs

```text
fold_i/source_test_prediction_results.csv
fold_i/source_test_metrics_per_drug.csv
fold_i/source_test_metrics_summary.csv
```

### 10.3 Fold-level TCGA primary outputs

```text
fold_i/target_primary_prediction_results.csv
fold_i/target_primary_metrics_per_drug.csv
fold_i/target_primary_metrics_summary.csv
```

### 10.4 Fold-level TCGA target-only outputs

```text
fold_i/target_only_prediction_results.csv
fold_i/target_only_metrics_per_drug.csv
fold_i/target_only_metrics_summary.csv
```

### 10.5 Fold-level TCGA auxiliary outputs

```text
fold_i/target_auxiliary_prediction_results.csv
fold_i/target_auxiliary_metrics_per_drug.csv
fold_i/target_auxiliary_metrics_summary.csv
```

### 10.6 Across-fold source outputs

```text
source_test_metrics_summary_across_folds.csv
source_test_metrics_summary_fold_mean_std.csv
source_test_metrics_per_drug_fold_mean_std.csv
```

### 10.7 Across-fold TCGA primary outputs

```text
target_primary_metrics_summary_across_folds.csv
target_primary_metrics_summary_fold_mean_std.csv
target_primary_metrics_per_drug_across_folds.csv
target_primary_metrics_per_drug_fold_mean_std.csv
```

### 10.8 Across-fold TCGA target-only outputs

```text
target_only_metrics_summary_across_folds.csv
target_only_metrics_summary_fold_mean_std.csv
target_only_metrics_per_drug_across_folds.csv
target_only_metrics_per_drug_fold_mean_std.csv
```

### 10.9 Across-fold TCGA auxiliary outputs

```text
target_auxiliary_metrics_summary_across_folds.csv
target_auxiliary_metrics_summary_fold_mean_std.csv
target_auxiliary_metrics_per_drug_across_folds.csv
target_auxiliary_metrics_per_drug_fold_mean_std.csv
```

---

## 11. Prediction Table Contract

Each `*_prediction_results.csv` must include at least:

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

Expected values:

Source test:

```text
domain = source
split = source_test
eval_dataset = source_test
```

TCGA primary:

```text
domain = target
split = target_eval
eval_dataset = primary
```

TCGA target-only:

```text
domain = target
split = target_eval
eval_dataset = target_only
```

TCGA auxiliary:

```text
domain = target
split = target_eval
eval_dataset = auxiliary
```

---

## 12. Metrics Contract

Each `*_metrics_per_drug.csv` must include at least:

```text
eval_dataset
drug_id
drug_index
n_samples
n_positive
n_negative
auroc
auprc
accuracy
balanced_accuracy
f1
precision
recall
fold
seed
has_supervised_source_label
is_target_eval_only
```

Each `*_metrics_summary.csv` must include at least:

```text
eval_dataset
n_drugs
n_observed_pairs
macro_auroc
macro_auprc
macro_accuracy
macro_balanced_accuracy
macro_f1
weighted_auroc
weighted_auprc
weighted_accuracy
weighted_balanced_accuracy
weighted_f1
overall_accuracy
fold
seed
```

If a drug has only a single ground-truth class within an evaluation dataset:

```text
AUROC = NaN
AUPRC = NaN
```

The drug row should still be retained.

---

## 13. Latent / t-SNE / FID / KMeans Outputs

Latent analysis is performed after fold training is complete.

### 13.1 Latent scope

Only omics-level latent representations are required:

- source omics latent;
- target TCGA omics latent.

Do not generate sample-drug integrated latent in this revision.

### 13.2 t-SNE

t-SNE is used for visualization.

Required outputs:

```text
fold_i/source_latent_representation.pkl
fold_i/target_latent_representation.pkl
fold_i/tsne_domain_mixing.png
fold_i/tsne_cancer_type.png
```

### 13.3 FID / latent distribution metrics

FID or equivalent latent distribution distance is computed between source and target omics latent representations.

Required outputs:

```text
fold_i/latent_distribution_metrics.csv
latent_metrics_summary.csv
```

### 13.4 KMeans

k-means uses high-dimensional latent features as input.

The main target of k-means analysis is cancer type clustering.

Required outputs:

```text
fold_i/kmeans_cancer_type_metrics.csv
kmeans_cancer_type_summary.csv
kmeans_cancer_type_fold_mean_std.csv
```

---

## 14. Leakage Prevention Rules

The implementation must enforce the following:

1. TCGA evaluation labels must not enter training loss.
2. TCGA evaluation labels must not enter domain adaptation loss.
3. TCGA evaluation labels must not enter early stopping.
4. TCGA metrics must not be used for model selection.
5. Source test metrics must not be used for model selection.
6. A `Sample_ID` must not appear in multiple source splits.
7. t-SNE, FID, and k-means outputs must not affect model selection.
8. Target-eval-only drugs must be clearly labeled in reports.
9. Source test set must be fixed and excluded from all folds’ train/validation sets.

---

## 15. Implementation Scope

The later implementation should add modular wrappers around AACDR rather than replacing core model code.

Likely implementation components:

- config / CLI parser;
- schema validation;
- data loading;
- source group split;
- TCGA evaluation dataset preparation;
- drug index construction;
- drug graph / feature adapter;
- AACDR trainer wrapper;
- prediction and metrics exporter;
- latent exporter;
- t-SNE / FID / k-means analysis modules;
- report writer.

---

## 16. Acceptance Criteria

The implementation should be considered successful only if all of the following are true:

1. `docs/proposal.md` and `docs/design.md` exist.
2. Source training uses 10% fixed source test and 5-fold train/validation split.
3. Split is grouped by `Sample_ID`.
4. The same `Sample_ID` never appears in train, validation, and source test simultaneously.
5. AACDR original model and training concept are preserved as much as possible.
6. TCGA omics can be used as unlabeled target input if AACDR requires it.
7. TCGA response labels are evaluation-only.
8. Final drug list equals source ∪ primary ∪ target_only ∪ auxiliary drugs.
9. `target_primary_prediction_results.csv` is produced per fold.
10. `target_only_prediction_results.csv` is produced per fold.
11. `target_auxiliary_prediction_results.csv` is produced per fold.
12. Each TCGA eval dataset has independent per-drug metrics.
13. Each TCGA eval dataset has independent summary metrics.
14. Target-eval-only drugs are marked as zero-shot / no source supervised label.
15. t-SNE, FID, and k-means are based only on source/target omics latent.
16. k-means uses high-dimensional latent features.
17. k-means target is cancer type clustering.
18. Any unclear implementation detail must trigger a user question rather than guessed behavior.
