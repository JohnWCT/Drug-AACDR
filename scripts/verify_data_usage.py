#!/usr/bin/env python3
"""
驗證 AACDR 主流成（Expr）資料使用是否符合預期。

設計思路：
1. 從 pickle 載入與 main.py 相同的資料集（與訓練流程一致）
2. 對 GDSC labeled 資料統計 entries / cell lines / drugs
3. 模擬 main.py 的 train_test_split(test_size=0.05, random_state=seed) 切分
4. 統計 TCGA unlabeled patients 與 TCGA test (labeled) 資料
5. 可選：從原始 CSV 交叉驗證 pickle 建構邏輯
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 讓腳本可從 scripts/ 或 code/ 執行
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CODE_DIR = os.path.join(PROJECT_ROOT, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from dataset import LabeledDataset  # noqa: E402


def project_root() -> str:
    return PROJECT_ROOT


def summarize_labeled_dataset(ds: LabeledDataset, name: str) -> dict:
    """從 LabeledDataset 統計 (drug, cell/patient) pair 數量。"""
    drugs = set()
    samples = set()  # cell line 或 patient
    labels = []
    for item in ds.data_list:
        drug_id, _expr, label = item
        drugs.add(int(drug_id))
        # GDSC/TCGA labeled entry 沒有單獨存 sample id，需從建構邏輯推回；
        # 這裡用 index 無法還原 patient，改由 CSV 補充（見 summarize_from_csv）
        labels.append(float(label))

    pos = sum(1 for x in labels if x > 0)
    neg = len(labels) - pos
    return {
        "name": name,
        "total_entries": len(ds),
        "unique_drugs": len(drugs),
        "positive": pos,
        "negative": neg,
        "pos_neg_ratio": f"1:{neg / pos:.2f}" if pos else "N/A",
    }


def summarize_labeled_from_response(
    expr_df: pd.DataFrame,
    response_df: pd.DataFrame,
    drug_list: list[str],
    name: str,
) -> dict:
    """重現 utils.makeDataset() 的 labeled pair 建構邏輯並統計。"""
    response = response_df.copy()
    response.columns = response.columns.astype(str)
    drugs = [d for d in drug_list if d in response.columns]
    response = response[drugs]

    samples = set()
    used_drugs = set()
    entries = 0
    pos = neg = 0

    for drug_id in drugs:
        for sample in response.index:
            x = response.loc[sample, drug_id]
            if sample in expr_df.index and not np.isnan(x):
                entries += 1
                samples.add(sample)
                used_drugs.add(drug_id)
                if x > 0:
                    pos += 1
                else:
                    neg += 1

    n_samples = len(samples)
    n_drugs = len(used_drugs)
    sample_drug_ratio = f"1:{n_drugs / n_samples:.1f}" if n_samples else "N/A"

    return {
        "name": name,
        "total_entries": entries,
        "unique_samples": n_samples,
        "unique_drugs": n_drugs,
        "sample_drug_ratio": sample_drug_ratio,
        "positive": pos,
        "negative": neg,
        "pos_neg_ratio": f"1:{neg / pos:.2f}" if pos else "N/A",
        "max_possible_pairs": n_samples * n_drugs,
        "coverage": entries / (n_samples * n_drugs) if n_samples * n_drugs else 0,
    }


def read_drug_list_csv(path: str) -> list[str]:
    drugs = []
    with open(path, "r") as f:
        for line in f.readlines()[1:]:
            drugs.append(line.strip().split(",")[-1])
    return drugs


def simulate_gdsc_split(gdsc_dataset: LabeledDataset, seed: int) -> dict:
    """模擬 main.py 的 GDSC train/val 切分（非 5-fold）。"""
    train_ds, val_ds = train_test_split(
        gdsc_dataset, test_size=0.05, random_state=seed
    )
    return {
        "split_method": "sklearn.train_test_split",
        "test_size": 0.05,
        "random_state": seed,
        "n_folds": 1,
        "same_val_for_all_models": True,
        "train_entries": len(train_ds),
        "val_entries": len(val_ds),
        "train_ratio": len(train_ds) / len(gdsc_dataset),
        "val_ratio": len(val_ds) / len(gdsc_dataset),
    }


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def compare_row(label: str, actual, expected, ok: bool) -> None:
    mark = "✓" if ok else "✗"
    print(f"  {mark} {label:28s} 實際={actual!s:>10}  預期={expected!s:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify AACDR Expr data usage")
    parser.add_argument("--seed", type=int, default=0, help="與 main.py 相同的 random_state")
    args = parser.parse_args()

    root = project_root()
    data = root + "/data"

    print_section("AACDR 主流成（Expr）資料使用驗證")
    print(f"專案根目錄: {root}")
    print(f"驗證 seed: {args.seed}")

    # --- 載入 pickle（與 main.py 一致）---
    with open(f"{data}/GDSC/GDSC_only_dataset.pkl", "rb") as f:
        gdsc_dataset = pickle.load(f)
    with open(f"{data}/TCGA/TCGA_unlabel_dataset.pkl", "rb") as f:
        tcga_unlabel = pickle.load(f)
    with open(f"{data}/TCGA/TCGA_dataset.pkl", "rb") as f:
        tcga_test = pickle.load(f)

    # --- 從 CSV 重建統計（可還原 cell line / patient 數）---
    gdsc_expr = pd.read_csv(f"{data}/GDSC/GDSC_expr_z_702.csv", index_col=0)
    gdsc_drugs = read_drug_list_csv(f"{data}/GDSC/GDSC_drug_binary.csv")
    gdsc_response = pd.read_csv(f"{data}/GDSC/GDSC_binary_response_151.csv", index_col=0)

    tcga_expr = pd.read_csv(f"{data}/TCGA/TCGA_expr_z_702.csv", index_col=0)
    tcga_unlabel_expr = pd.read_csv(f"{data}/TCGA/TCGA_unlabel_expr_702_01A.csv", index_col=0)
    tcga_drugs = read_drug_list_csv(f"{data}/TCGA/TCGA_drug_new.csv")
    tcga_response = pd.read_csv(f"{data}/TCGA/TCGA_response_new.csv", index_col=0)

    gdsc_stats = summarize_labeled_from_response(
        gdsc_expr, gdsc_response, gdsc_drugs, "GDSC (train+val pool)"
    )
    tcga_test_stats = summarize_labeled_from_response(
        tcga_expr, tcga_response, tcga_drugs, "TCGA (test)"
    )

    n_unlabel_patients_pkl = len(tcga_unlabel)
    # CSV: patients × genes → shape[0] = patient 數
    n_unlabel_patients_csv = tcga_unlabel_expr.shape[0]

    # --- Training & Validation pool (GDSC labeled) ---
    print_section("Training & Validation Set（GDSC labeled, Expr）")
    print("  說明: GDSC 全部 labeled pairs 先作為 train+val pool，")
    print("        再由 train_test_split(5%) 切成 train / val。")
    print("        無 5-fold；100 個 model 共用同一組 train/val。")

    expected_gdsc = {
        "total_entries": 112575,
        "unique_samples": 950,
        "unique_drugs": 151,
        "sample_drug_ratio": "1:7.5",  # 論文表格中的 ratio = 負:正樣本比 (neg:pos)
        "unlabeled_patients": 9424,
    }

    compare_row(
        "total labeled entries",
        gdsc_stats["total_entries"],
        expected_gdsc["total_entries"],
        gdsc_stats["total_entries"] == expected_gdsc["total_entries"],
    )
    compare_row(
        "cell lines",
        gdsc_stats["unique_samples"],
        expected_gdsc["unique_samples"],
        gdsc_stats["unique_samples"] == expected_gdsc["unique_samples"],
    )
    compare_row(
        "drugs",
        gdsc_stats["unique_drugs"],
        expected_gdsc["unique_drugs"],
        gdsc_stats["unique_drugs"] == expected_gdsc["unique_drugs"],
    )
    compare_row(
        "neg:pos ratio (labeled)",
        gdsc_stats["pos_neg_ratio"],
        expected_gdsc["sample_drug_ratio"],
        gdsc_stats["pos_neg_ratio"].startswith("1:7.5") or abs(float(gdsc_stats["pos_neg_ratio"].split(":")[1]) - 7.5) < 0.1,
    )
    compare_row(
        "unlabeled patients (TCGA)",
        n_unlabel_patients_pkl,
        expected_gdsc["unlabeled_patients"],
        n_unlabel_patients_pkl == expected_gdsc["unlabeled_patients"],
    )
    print(f"  pickle 與 CSV unlabeled 一致: {n_unlabel_patients_pkl == n_unlabel_patients_csv}")
    print(f"  GDSC 標籤覆蓋率: {gdsc_stats['coverage']:.1%} "
          f"({gdsc_stats['total_entries']}/{gdsc_stats['max_possible_pairs']} pairs)")
    print(f"  正負樣本比 (neg:pos): {gdsc_stats['pos_neg_ratio']}")

    split_info = simulate_gdsc_split(gdsc_dataset, args.seed)
    print_section("GDSC Train / Val 切分（main.py 邏輯）")
    for k, v in split_info.items():
        print(f"  {k}: {v}")

    # --- Test set (TCGA labeled) ---
    print_section("Test Set（TCGA labeled, Expr）")
    expected_tcga = {
        "total_entries": 666,
        "unique_samples": 569,
        "unique_drugs": 69,
        "sample_drug_ratio": "1:1",  # 論文表格中的 ratio = 負:正樣本比 (neg:pos)
    }

    compare_row(
        "total entries",
        tcga_test_stats["total_entries"],
        expected_tcga["total_entries"],
        tcga_test_stats["total_entries"] == expected_tcga["total_entries"],
    )
    compare_row(
        "patients",
        tcga_test_stats["unique_samples"],
        expected_tcga["unique_samples"],
        tcga_test_stats["unique_samples"] == expected_tcga["unique_samples"],
    )
    compare_row(
        "drugs",
        tcga_test_stats["unique_drugs"],
        expected_tcga["unique_drugs"],
        tcga_test_stats["unique_drugs"] == expected_tcga["unique_drugs"],
    )
    compare_row(
        "neg:pos ratio (labeled)",
        tcga_test_stats["pos_neg_ratio"],
        expected_tcga["sample_drug_ratio"],
        abs(float(tcga_test_stats["pos_neg_ratio"].split(":")[1]) - 1.0) < 0.1,
    )
    print(f"  TCGA test pickle entries: {len(tcga_test)}")
    print(f"  正負樣本比 (neg:pos): {tcga_test_stats['pos_neg_ratio']}")

    # --- Pickle 與 CSV 一致性 ---
    print_section("Pickle 與 CSV 建構一致性")
    compare_row(
        "GDSC pickle entries",
        len(gdsc_dataset),
        gdsc_stats["total_entries"],
        len(gdsc_dataset) == gdsc_stats["total_entries"],
    )
    compare_row(
        "TCGA test pickle entries",
        len(tcga_test),
        tcga_test_stats["total_entries"],
        len(tcga_test) == tcga_test_stats["total_entries"],
    )

    print_section("訓練流程摘要")
    print("""
  資料角色:
    - GDSC_only_dataset.pkl     → Source domain，有標籤 (drug, cell line, response)
    - TCGA_unlabel_dataset.pkl  → Target domain，無標籤 patient expression（域適應用）
    - TCGA_dataset.pkl          → Target domain test，有標籤 (drug, patient, response)

  Train/Val 切分:
    - 僅對 GDSC labeled 做切分: train_test_split(test_size=0.05, random_state=seed)
    - 約 95% train / 5% val（以 entry 為單位，非 cell line 分層）
    - 無 StratifiedKFold / 5-fold
    - main.py 訓練 100 個 model，每個 model 重新初始化權重，但 train/val 切分固定

  測試:
    - TCGA_dataset 全程作為 test，不參與 train/val
    - TCGA_unlabel 全程作為 unlabeled target，用於對抗域適應
""")


if __name__ == "__main__":
    main()
