#!/usr/bin/env python3
"""
AACDR Docker 煙霧測試（smoke test）

設計目標：
1. 在容器內快速驗證「從原始資料 → pickle 資料集 → 單次訓練」整條流程可跑通。
2. 每一步獨立回報 PASS/FAIL，方便定位問題（環境、資料、路徑、GPU、訓練邏輯）。
3. 不修改主程式訓練邏輯，僅透過子程序呼叫既有 main.py / utils.makeDataset。
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = PROJECT_ROOT / "data"
CKPT_DIR = PROJECT_ROOT / "ckpt"


def _ok(step: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"[PASS] {step}{suffix}")


def _fail(step: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"[FAIL] {step}{suffix}")
    sys.exit(1)


def step_imports() -> None:
    """驗證 Python 依賴與專案模組可在容器內正常 import。"""
    import torch
    import numpy
    import pandas
    import scipy
    import sklearn
    import rdkit

    os.chdir(CODE_DIR)
    sys.path.insert(0, str(CODE_DIR))
    import dataset  # noqa: F401
    import model  # noqa: F401
    import result  # noqa: F401
    import trainer  # noqa: F401
    import utils  # noqa: F401

    cuda = torch.cuda.is_available()
    _ok("imports", f"torch={torch.__version__}, cuda={cuda}")


def step_data_layout() -> None:
    """驗證原始 CSV 與 drug graph 是否存在（不依賴 pickle）。"""
    required = [
        DATA_DIR / "GDSC" / "GDSC_expr_z_702.csv",
        DATA_DIR / "GDSC" / "GDSC_binary_response_151.csv",
        DATA_DIR / "GDSC" / "GDSC_drug_binary.csv",
        DATA_DIR / "TCGA" / "TCGA_unlabel_expr_702_01A.csv",
        DATA_DIR / "TCGA" / "TCGA_expr_z_702.csv",
        DATA_DIR / "TCGA" / "TCGA_response_new.csv",
        DATA_DIR / "TCGA" / "TCGA_drug_new.csv",
    ]
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in required if not p.exists()]
    if missing:
        _fail("data layout", f"missing: {missing}")

    from utils import _drug_graph_dir, read_drug_list

    os.chdir(CODE_DIR)
    gdsc_graph = _drug_graph_dir("GDSC")
    tcga_graph = _drug_graph_dir("TCGA")
    gdsc_drugs = read_drug_list("GDSC")
    tcga_drugs = read_drug_list("TCGA")

    gdsc_missing = [d for d in gdsc_drugs if not (gdsc_graph / f"{d}.npz").exists()]
    tcga_missing = [d for d in tcga_drugs if not (tcga_graph / f"{d}.npz").exists()]
    if gdsc_missing or tcga_missing:
        _fail(
            "drug graphs",
            f"GDSC missing={len(gdsc_missing)}, TCGA missing={len(tcga_missing)}",
        )

    _ok("data layout", f"GDSC graphs={len(gdsc_drugs)}, TCGA graphs={len(tcga_drugs)}")


def step_make_dataset() -> None:
    """從 CSV 產生 pickle 資料集（與上游 utils.makeDataset 相同邏輯）。"""
    os.chdir(CODE_DIR)
    sys.path.insert(0, str(CODE_DIR))
    from utils import makeDataset

    makeDataset()

    pkls = [
        DATA_DIR / "GDSC" / "GDSC_only_dataset.pkl",
        DATA_DIR / "TCGA" / "TCGA_unlabel_dataset.pkl",
        DATA_DIR / "TCGA" / "TCGA_dataset.pkl",
    ]
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in pkls if not p.exists()]
    if missing:
        _fail("makeDataset", f"missing outputs: {missing}")

    with open(pkls[0], "rb") as f:
        gdsc = pickle.load(f)
    with open(pkls[1], "rb") as f:
        tcga_unlabel = pickle.load(f)
    with open(pkls[2], "rb") as f:
        tcga = pickle.load(f)

    _ok(
        "makeDataset",
        f"GDSC={len(gdsc)}, TCGA_unlabel={len(tcga_unlabel)}, TCGA={len(tcga)}",
    )


def step_train_smoke() -> None:
    """
    以最小參數執行 main.py：
    - model_nums=1：只訓練 1 個模型，縮短測試時間
    - max_epoch=1：每模型 1 epoch，驗證訓練迴圈可進入
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "main.py",
        "--description",
        "DockerSmoke",
        "--id",
        "smoke",
        "--model_nums",
        "1",
        "--max_epoch",
        "1",
        "--batch_size",
        "256",
    ]
    proc = subprocess.run(
        cmd,
        cwd=CODE_DIR,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        _fail("train smoke", f"exit code {proc.returncode}")

    ckpts = list(CKPT_DIR.glob("model_*DockerSmoke*.pt"))
    logs = list(CKPT_DIR.glob("log_*DockerSmoke*.txt"))
    if not ckpts:
        _fail("train smoke", "no checkpoint files written to ckpt/")
    _ok("train smoke", f"checkpoints={len(ckpts)}, logs={len(logs)}")


def main() -> None:
    print(f"=== AACDR smoke test @ {PROJECT_ROOT} ===")
    step_imports()
    step_data_layout()
    step_make_dataset()
    step_train_smoke()
    print("=== ALL STEPS PASSED ===")


if __name__ == "__main__":
    main()
