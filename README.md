# Drug-AACDR

本專案以 [tfwuSUDA/AACDR](https://github.com/tfwuSUDA/AACDR) 為基礎進行改寫與本地驗證，用於癌症藥物反應預測（Cancer Drug Response Prediction）相關實驗。

原始 AACDR 方法結合圖同構網路（GIN）與非對稱對抗域適應（Asymmetric Adversarial Domain Adaptation），將細胞株（source）知識遷移至病患（target）資料，以提升藥物反應預測表現。詳細方法說明請參考上游專案 README。

## 與上游專案的關係

| 項目 | 說明 |
|------|------|
| 上游來源 | [https://github.com/tfwuSUDA/AACDR](https://github.com/tfwuSUDA/AACDR) |
| 本倉庫用途 | 程式改寫、路徑與執行環境調整、**本地資料測試** |
| 資料與補充材料 | 不納入版本控制，請依下方說明自行準備 |

## 專案結構

```
AACDR/
├── code/                 # 主要程式碼
│   ├── main.py           # 訓練入口
│   ├── model.py
│   ├── trainer.py
│   ├── dataset.py
│   ├── utils.py
│   └── result.py
├── data/                 # 資料集（已加入 .gitignore，需本地準備）
├── supplymentmaterials/  # 補充材料（已加入 .gitignore）
├── Dockerfile.aacdr      # Docker 建置檔
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
python utils.py
```

（上游 README 中指令為 `uitils.py`，實際檔名為 `utils.py`。）

## 訓練與測試

在專案根目錄下，於 `code/` 目錄執行：

```bash
cd code
python main.py --description Reproduce --id 0
```

訓練紀錄與模型 checkpoint 預設寫入 `ckpt/`（同樣不納入版本控制）。

## Docker（選用）

於本專案根目錄建置映像：

```bash
docker build -f Dockerfile.aacdr -t aacdr:cu118 .
```

## 授權與致謝

本專案改寫自 [tfwuSUDA/AACDR](https://github.com/tfwuSUDA/AACDR)。若使用或引用，請一併參考並致謝原始作者與論文。
