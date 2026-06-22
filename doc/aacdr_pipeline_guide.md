# AACDR Evaluation Pipeline 使用指南

本文件說明 `code/aacdr_pipeline/` 新 pipeline 的用途、架構、執行方式與輸出格式。設計依據見同目錄下的 `proposal.md` 與 `design.md`。

---

## 1. 概述

新 pipeline 在**不修改** AACDR 核心模型（`model.py`）與原始 trainer（`trainer.py`）的前提下，新增一層模組化 wrapper，支援：

- DAPL 風格的 source / TCGA 資料輸入
- Source：**10% 固定 test** + **90% 五折 train/validation**（依 `Sample_ID` 分組）
- TCGA response **僅用於 evaluation**，不進入 training loss
- 三個獨立 TCGA eval set：`primary`、`target_only`、`auxiliary`
- 完整預測表、per-drug / summary metrics、latent、t-SNE、FID、k-means 輸出

預設策略：**5 folds × 1 seed**（不再預設 100 次 initialization）。

---

## 2. 專案結構

```text
code/
  aacdr_multilabel_hyper_main.py    # 薄入口（CLI → config → run_pipeline）
  aacdr_pipeline/
    config.py          # CLI 與 AACDRPipelineConfig
    schema.py          # schema 驗證、drug/TCGA ID 正規化
    data_io.py         # 載入 source/target/TCGA CSV
    features.py        # source/target omics 特徵對齊
    splits.py          # 依 Sample_ID 的 grouped split
    drug_index.py      # 最終 drug list 與 zero-shot 標記
    target_eval.py     # 三個 TCGA eval dataset 準備
    drug_graph_adapter.py  # AACDR 相容 drug graph（legacy npz 或 SMILES）
    datasets.py        # 包裝 LabeledDataset / UnlabeledDataset
    model_adapter.py   # FE / DNN / DynamicAutoEncoder wrapper
    trainer_wrapper.py # PipelineAADATrainer（無 TCGA label leakage）
    evaluation.py      # 預測與 metrics
    latent.py          # omics latent 匯出
    tsne.py            # t-SNE 視覺化
    fid.py             # FID / MMD / Wasserstein
    kmeans.py          # 高維 latent k-means 癌種聚類
    reports.py         # CSV/JSON 寫出與跨 fold 彙總
    run.py             # 全流程 orchestrator

tests/                 # 單元測試與 synthetic smoke test
requirements-pipeline.txt  # pipeline 額外依賴（deepchem、pytest）
```

---

## 3. 環境需求（Docker AACDR）

Pipeline 在 Docker 容器 `AACDR` 內執行，工作目錄為 `/workspace/AACDR`。

```bash
# 進入容器
docker exec -it AACDR bash

# 安裝 pipeline 額外依賴（藥物圖需 deepchem + rdkit）
pip install -r /workspace/AACDR/requirements-pipeline.txt
```

| 依賴 | 用途 |
|------|------|
| `torch` | 模型訓練（容器內已具備） |
| `rdkit` | SMILES 解析（容器內已具備） |
| `deepchem` | ConvMolFeaturizer，與 AACDR 原始 drug graph 格式一致 |
| `pytest` | 執行測試 |

---

## 4. 快速開始

### 4.1 完整 DAPL 資料跑一輪

```bash
docker exec -it AACDR bash -c '
cd /workspace/AACDR/code && \
python aacdr_multilabel_hyper_main.py \
  --source_omics_path /workspace/DAPL-master/data/pretrain_ccle.csv \
  --source_response_path /workspace/DAPL-master/data/GDSC2_fitted_dose_response_MaxScreen_raw.csv \
  --target_omics_path /workspace/DAPL-master/data/TCGA/pretrain_tcga.csv \
  --target_eval_primary_response_path /workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv \
  --target_eval_target_only_response_path /workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv \
  --target_eval_aux_response_path /workspace/DAPL-master/data/TCGA/TCGA_drug_response_from_DAPL.csv \
  --drug_smiles_path /workspace/DAPL-master/data/GDSC_drug_merge_pubchem_dropNA_MACCS.csv \
  --source_sample_col Sample_ID \
  --source_drug_col drug_name \
  --source_label_col Label \
  --n_splits 5 \
  --source_test_size 0.10 \
  --seed 0 \
  --output_dir /workspace/AACDR/outputs_aacdr_eval3
'
```

### 4.2 執行測試（含 synthetic smoke test）

```bash
docker exec AACDR bash -c '
pip install -q -r /workspace/AACDR/requirements-pipeline.txt && \
cd /workspace/AACDR/code && \
python -m pytest /workspace/AACDR/tests -v
'
```

---

## 5. CLI 參數

### 5.1 資料路徑

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--source_omics_path` | `.../pretrain_ccle.csv` | CCLE 基因表達（sample col: `Sample_ID`） |
| `--source_response_path` | `.../GDSC2_fitted_dose_response_MaxScreen_raw.csv` | GDSC2 response |
| `--target_omics_path` | `.../TCGA/pretrain_tcga.csv` | TCGA omics（sample col: `tissue_id`） |
| `--target_eval_primary_response_path` | `.../PMID27354694_..._gdsc_intersect13.csv` | TCGA primary eval |
| `--target_eval_target_only_response_path` | `.../PMID27354694_..._tcga_only3.csv` | TCGA target-only eval |
| `--target_eval_aux_response_path` | `.../TCGA_drug_response_from_DAPL.csv` | TCGA auxiliary eval |
| `--drug_smiles_path` | `.../GDSC_drug_merge_pubchem_dropNA_MACCS.csv` | 藥物 SMILES / MACCS |
| `--ccle_cancer_info_path` | `.../ccle_sample_info_df.csv` | CCLE 癌種對照 |
| `--tcga_cancer_info_path` | `.../TCGA/xena_sample_info_df.csv` | TCGA 癌種對照 |

### 5.2 欄位名稱

| 參數 | 預設值 |
|------|--------|
| `--source_sample_col` | `Sample_ID` |
| `--source_drug_col` | `drug_name` |
| `--source_label_col` | `Label` |
| `--target_sample_col` | `Patient_id` |
| `--target_drug_col` | `drug_name` |
| `--target_label_col` | `Label` |
| `--target_omics_sample_col` | `tissue_id` |

### 5.3 實驗設定

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--n_splits` | `5` | 五折 CV |
| `--source_test_size` | `0.10` | 固定 source test 比例 |
| `--seed` | `0` | 隨機種子 |
| `--max_epoch` | `10` | 每 fold 最大 epoch |
| `--batch_size` | `768` | source batch size |
| `--learning_rate` | `0.0005` | 學習率 |
| `--output_dir` | `outputs_aacdr_eval3` | 輸出目錄 |
| `--device` | 自動偵測 | `cuda` 或 `cpu` |
| `--no_tsne` | — | 關閉 t-SNE |
| `--no_fid` | — | 關閉 FID 指標 |
| `--no_kmeans` | — | 關閉 k-means |
| `--max_train_rows` | — | 限制訓練列數（debug） |
| `--aacdr_data_root` | — | AACDR legacy drug graph 根目錄 |

---

## 6. 資料契約

### 6.1 正規化規則

- **Drug name**：`str(drug_name).strip().lower()`
- **TCGA patient ID**：`TCGA-XX-YYYY-ZZZ...` → `TCGA-XX-YYYY`（前三段）
- **Source sample**：`str(sample_id).strip()`

### 6.2 Source split

1. 從 response 取 unique `Sample_ID`
2. 固定切出 10% 作為 **source test**（所有 fold 共用）
3. 剩餘 90% 做 5-fold train/validation
4. 同一 `Sample_ID` 的所有 drug-response rows **不可跨** train / val / source test

### 6.3 Drug list

最終 drug list = source drugs ∪ primary ∪ target_only ∪ auxiliary

每個 drug 標記：`has_supervised_source_label`、`is_target_eval_only` 等（見 `drug_list.csv`）。

---

## 7. 訓練與 model selection

| 允許 | 禁止 |
|------|------|
| Source supervised training | TCGA response labels 進 training loss |
| TCGA omics 作 unlabeled target（domain adaptation） | TCGA labels 進 validation / early stopping |
| Source validation AUROC 選模型 | Source test 用於 model selection |
| | TCGA metrics 用於 model selection |

`PipelineAADATrainer` 訓練時傳入**空的 TCGA labeled dataset**；TCGA 僅以 unlabeled omics 參與重建與 margin loss。

---

## 8. 輸出目錄結構

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
  latent_metrics_summary.csv
  kmeans_cancer_type_summary.csv
  kmeans_cancer_type_fold_mean_std.csv
  *_metrics_summary_across_folds.csv
  *_metrics_summary_fold_mean_std.csv
  fold_0/ ... fold_4/
```

### 8.1 每個 fold 內容

| 類別 | 檔案 |
|------|------|
| Source test | `source_test_prediction_results.csv`、`source_test_metrics_per_drug.csv`、`source_test_metrics_summary.csv` |
| TCGA primary | `target_primary_prediction_results.csv`、`target_primary_metrics_per_drug.csv`、`target_primary_metrics_summary.csv` |
| TCGA target_only | `target_only_prediction_results.csv`、`target_only_metrics_per_drug.csv`、`target_only_metrics_summary.csv` |
| TCGA auxiliary | `target_auxiliary_prediction_results.csv`、`target_auxiliary_metrics_per_drug.csv`、`target_auxiliary_metrics_summary.csv` |
| Latent | `source_latent_representation.pkl`、`target_latent_representation.pkl` |
| 視覺化 | `tsne_latent_dual.png`（DAPL 風格 1×2 雙面板：Domain + Cancer Type，dpi=250，t-SNE random_state=42） |
| 分佈 / 聚類 | `latent_distribution_metrics.csv`、`kmeans_cancer_type_metrics.csv` |
| Checkpoint | `best_model_fe.pt`、`best_model_dnn.pt`、`best_model_ae.pt` |

### 8.2 Prediction 表必要欄位

`sample_id`, `drug_id`, `drug_index`, `domain`, `split`, `eval_dataset`, `ground_truth`, `pred_score`, `probability`, `pred_label`, `confidence`, `fold`, `seed`, `task_type`, `cancer_type`, `has_supervised_source_label`, `is_target_eval_only`

### 8.3 Metrics

- **Per-drug**：`auroc`, `auprc`, `accuracy`, `balanced_accuracy`, `f1`, `precision`, `recall`（單一 class 時 AUROC/AUPRC = NaN，保留該列）
- **Summary**：`macro_*`、`weighted_*`、`overall_*`（含 AUROC/AUPRC/accuracy/F1/precision/recall，overall 為所有 sample-drug pairs 合併計算）

---

## 9. 與原始 AACDR 的關係

```text
aacdr_multilabel_hyper_main.py
        │
        ▼
   run_pipeline()
        │
        ├── data_io / features / splits / drug_index
        ├── drug_graph_adapter  ──► legacy .npz 或 SMILES→ConvMol
        ├── datasets            ──► LabeledDataset / UnlabeledDataset
        ├── trainer_wrapper     ──► PipelineAADATrainer（複製 AACDR 訓練邏輯）
        │       └── model_adapter ──► FE, DNN, DynamicAutoEncoder
        └── evaluation / latent / tsne / fid / kmeans
```

- **未修改**：`code/model.py`、`code/trainer.py`、`code/dataset.py`
- **動態特徵維度**：`FE(n_input=對齊後基因數)`，不再硬編碼 702
- **藥物圖 key**：使用 `drug_index`（整數）作為 graph dict key

---

## 10. 常見問題

### Q: `ImportError: No module named 'deepchem'`

```bash
pip install deepchem>=2.7.0
```

### Q: 訓練很慢

可先縮小規模驗證流程：

```bash
python aacdr_multilabel_hyper_main.py --max_epoch 1 --max_train_rows 5000
```

### Q: 如何確認 TCGA labels 沒有進 training？

檢查 `run_manifest.json` 中 `"tcga_labels_in_training": false`，並確認 `trainer_wrapper.py` 使用空 `LabeledDataset` 作為 TCGA labeled set。

### Q: 與 `code/main.py` / `parallel_train.py` 的差異

| 項目 | 原始 AACDR | 新 pipeline |
|------|-----------|-------------|
| 資料 | pickle + 固定 702 基因 | DAPL CSV + 動態特徵對齊 |
| Split | 單次 5% val | 10% test + 5-fold |
| 模型數 | 預設 100 init | 預設 5 folds × 1 seed |
| TCGA eval | 單一 TCGA set | primary / target_only / auxiliary |
| 輸出 | ckpt + log | 完整 CSV / PNG / PKL 報表樹 |

---

## 11. 相關文件

- [proposal.md](./proposal.md) — 需求提案
- [design.md](./design.md) — 架構設計
- [../README.md](../README.md) — 專案總覽與原始 AACDR 重現流程
