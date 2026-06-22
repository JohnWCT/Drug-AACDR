#!/usr/bin/env python3
"""
平行訓練 worker：負責連續訓練 [model_start, model_end) 範圍內的模型。

設計思路：
- 100 次隨機初始化彼此獨立，可安全平行。
- 每個 worker 寫入獨立 partial log，避免多進程搶寫同一檔案。
- checkpoint 檔名加入 model_number，避免 AUC 相同時覆蓋。
"""

import argparse
import os
import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from trainer import AADATrainer
from utils import read_drug_graph, read_drug_list


def project_root() -> str:
    cwd = os.getcwd().split("/")[:-1]
    return "/".join(cwd)


def set_global_seed(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_datasets(root: str):
    with open(root + "/data/GDSC/GDSC_only_dataset.pkl", "rb") as f:
        gdsc_dataset = pickle.load(f)
    with open(root + "/data/TCGA/TCGA_unlabel_dataset.pkl", "rb") as f:
        tcga_unlabel_dataset = pickle.load(f)
    with open(root + "/data/TCGA/TCGA_dataset.pkl", "rb") as f:
        tcga_dataset = pickle.load(f)
    return gdsc_dataset, tcga_unlabel_dataset, tcga_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="AACDR parallel training worker")
    parser.add_argument("--description", type=str, required=True)
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_start", type=int, required=True)
    parser.add_argument("--model_end", type=int, required=True)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=768)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--worker_id", type=int, default=0)
    args = parser.parse_args()

    if args.model_start >= args.model_end:
        raise ValueError("model_end must be greater than model_start")

    set_global_seed(args.seed)
    root = project_root()
    runs_dir = os.path.join(root, "ckpt", "runs")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(os.path.join(root, "ckpt"), exist_ok=True)

    gdsc_drug_list = read_drug_list("GDSC")
    gdsc_drug_graph = read_drug_graph("GDSC", gdsc_drug_list)
    tcga_drug_list = read_drug_list("TCGA")
    tcga_drug_graph = read_drug_graph("TCGA", tcga_drug_list)

    gdsc_dataset, tcga_unlabel_dataset, tcga_dataset = load_datasets(root)
    train_dataset, val_dataset = train_test_split(
        gdsc_dataset, test_size=0.05, random_state=args.seed
    )

    print(
        f"[worker {args.worker_id}] device={args.device} "
        f"models=[{args.model_start}, {args.model_end})"
    )

    for model_number in range(args.model_start, args.model_end):
        print("<============================>")
        print(f"model No.{model_number}")

        trainer = AADATrainer(
            args.seed,
            args.lr,
            args.batch_size,
            args.max_epoch,
            gdsc_drug_graph,
            tcga_drug_graph,
            train_dataset,
            val_dataset,
            tcga_unlabel_dataset,
            tcga_dataset,
            device=args.device,
        )
        metric, models = trainer.fit()

        auc = metric.whole["auc"]
        tag = (
            f"model_{model_number:03d}_{auc:.6f}_"
            f"{args.seed}_{args.id}_{args.description}"
        )
        torch.save(models[0], os.path.join(root, "ckpt", f"{tag}fe.pt"))
        torch.save(models[1], os.path.join(root, "ckpt", f"{tag}dnn.pt"))
        torch.save(models[2], os.path.join(root, "ckpt", f"{tag}ecd.pt"))

        partial_log = os.path.join(runs_dir, f"model_{model_number:03d}.txt")
        with open(partial_log, "w", encoding="utf-8") as f:
            f.write(metric.saved_metric())
            f.write("------------------------------\n")

        print(metric)
        print(f"[worker {args.worker_id}] saved {partial_log}")


if __name__ == "__main__":
    main()
