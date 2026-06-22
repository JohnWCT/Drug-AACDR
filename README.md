# Drug-AACDR

本專案以 [tfwuSUDA/AACDR](https://github.com/tfwuSUDA/AACDR) 為基礎進行改寫與本地驗證，用於癌症藥物反應預測（Cancer Drug Response Prediction）相關實驗。

原始 AACDR 方法結合圖同構網路（GIN）與非對稱對抗域適應（Asymmetric Adversarial Domain Adaptation），將細胞株（source）知識遷移至病患（target）資料，以提升藥物反應預測表現。詳細方法說明請參考上游專案 README。

## 與上游專案的關係

| 項目 | 說明 |
|------|------|
| 上游來源 | [https://github.com/tfwuSUDA/AACDR](https://github.com/tfwuSUDA/AACDR) |
| 本倉庫用途 | 程式改寫、Docker 環境、平行訓練、**本地資料測試** |
| 資料與補充材料 | 不納入版本控制，請依下方說明自行準備 |

## 專案結構

```
AACDR/
├── code/
│   ├── main.py              # 循序訓練入口（100 次初始化）
│   ├── parallel_train.py    # 平行訓練 worker
│   ├── model.py
│   ├── trainer.py
│   ├── dataset.py
│   ├── utils.py
│   └── result.py
├── scripts/
│   ├── run_parallel_train.sh    # 啟動多 worker 平行訓練
│   ├── merge_parallel_logs.py   # 合併 per-model 結果
│   ├── summarize_results.py     # 計算 mean ± std，比對文獻
│   ├── docker_workflow_test.sh  # Docker 煙霧測試
│   └── aacdr_smoke_test.py
├── data/                    # 資料集（.gitignore）
├── supplymentmaterials/     # 文獻補充材料（.gitignore）
├── ckpt/                    # 訓練產物（.gitignore）
├── Dockerfile.aacdr
└── README.md
```

## 環境需求

建議環境（與上游一致）：

- Python 3.8
- CUDA 11.8
- PyTorch 2.3.x（cu118）

可使用上游專案的 `environment.yaml` 建立 conda 環境，或參考 `Dockerfile.aacdr` 以 Docker 建置。

## 資料準備

`data/` 與 `supplymentmaterials/` 已設定為 git ignore，不會上傳至本倉庫。請從 [上游 AACDR 專案](https://github.com/tfwuSUDA/AACDR) 取得原始資料後放置於對應目錄。

若需從原始 Expr 資料產生 pickle 資料集，可解壓 `data/TCGA/TCGA_unlabel_expr_702_01A.csv.gz` 後執行：

```bash
cd code
python -c "from utils import makeDataset; makeDataset()"
```

（上游 README 中指令為 `uitils.py`，實際檔名為 `utils.py`。）

## Docker

### 建置與啟動

```bash
# 於 Drug/ 目錄下建置（掛載整個 Drug 目錄）
cd AACDR
docker build -f Dockerfile.aacdr -t aacdr:cuda118 .

cd ..
docker run --gpus all -itd \
  --name AACDR \
  -v "$PWD":/workspace/ \
  -w /workspace/AACDR \
  aacdr:cuda118
```

### 煙霧測試（驗證環境與流程）

```bash
docker exec AACDR bash /workspace/AACDR/scripts/docker_workflow_test.sh
```

## 訓練

### 循序訓練（與上游相同）

```bash
cd code
python main.py --description Reproduce --id 0
```

預設：`model_nums=100`、`max_epoch=10`。

### 平行訓練（建議，可大幅縮短時間）

在 Docker 容器內執行，預設 8 個 worker 各訓練一部分模型：

```bash
docker exec AACDR bash /workspace/AACDR/scripts/run_parallel_train.sh 8 100
```

參數：`[workers] [total_models] [description] [run_id] [seed]`

訓練結束後會自動合併 log 並產生摘要報告。

## 輸出結果位置

訓練產物寫入 `ckpt/`（不納入版本控制）：

| 檔案 | 說明 |
|------|------|
| `ckpt/runs/model_XXX.txt` | 各次初始化的 per-model 指標 |
| `ckpt/log_0_0_Reproduce.txt` | 合併後 100 次完整 log |
| `ckpt/log_0_0_Reproduce_summary.md` | **彙整摘要**（mean ± std，含文獻比對） |
| `ckpt/log_0_0_Reproduce_summary.csv` | 同上，CSV 格式 |
| `ckpt/model_*.pt` | 模型 checkpoint |
| `ckpt/parallel_logs/worker_*.log` | 各 worker 訓練日誌 |

手動彙整（若需重新計算）：

```bash
python scripts/summarize_results.py \
  --log ckpt/log_0_0_Reproduce.txt \
  --reference supplymentmaterials/100_random_initializations.txt \
  --out-dir ckpt
```

## 本地驗證結果（Expr，100 次初始化）

與文獻 `supplymentmaterials/100_random_initializations.txt` 比對，趨勢一致：

| split | AUC | ACC | precision | recall | F1 |
|-------|-----|-----|-----------|--------|-----|
| **whole** | 0.7405 ± 0.0203 | 0.6979 ± 0.0144 | 0.6627 ± 0.0151 | 0.7708 ± 0.0592 | 0.7113 ± 0.0231 |
| **seen** | 0.7519 ± 0.0209 | 0.7092 ± 0.0153 | 0.6636 ± 0.0183 | 0.7567 ± 0.0725 | 0.7049 ± 0.0284 |
| **unseen** | 0.6872 ± 0.0326 | 0.6818 ± 0.0244 | 0.6922 ± 0.0200 | 0.7879 ± 0.0911 | 0.7337 ± 0.0361 |

文獻參考（whole AUC）：**0.7338 ± 0.0230**

## 本倉庫相對上游的主要改動

- 修正 TCGA pickle 路徑（移除原作者本機路徑）
- GPU device 自動偵測（不再硬編碼 `cuda:2`）
- 支援 `drug_graph` / `drug_graph3` 目錄名稱
- 新增 Docker 平行訓練與結果彙整腳本

## 授權與致謝

本專案改寫自 [tfwuSUDA/AACDR](https://github.com/tfwuSUDA/AACDR)。若使用或引用，請一併參考並致謝原始作者與論文。
