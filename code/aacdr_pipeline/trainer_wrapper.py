"""Train one fold with AACDR logic; TCGA labels never enter training."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader

from aacdr_pipeline.config import AACDRPipelineConfig
from aacdr_pipeline.datasets import FoldDataBundle
from aacdr_pipeline.drug_graph_adapter import DrugRepresentationBundle
from aacdr_pipeline.model_adapter import (
    PipelineModels,
    build_pipeline_models,
    save_checkpoint,
)
from dataset import LabeledDataset


@dataclass
class FoldTrainingResult:
    fold: int
    models: PipelineModels
    best_checkpoint_prefix: str
    best_epoch: int
    best_val_auroc: float
    train_log: list[dict]


class PipelineAADATrainer:
    """AACDR trainer wrapper: source-only model selection, no TCGA label leakage."""

    def __init__(
        self,
        config: AACDRPipelineConfig,
        fold_data: FoldDataBundle,
        drug_bundle: DrugRepresentationBundle,
    ):
        self.config = config
        self.fold_data = fold_data
        self.drug_bundle = drug_bundle
        self.models = build_pipeline_models(fold_data.n_features, config.device)
        self.tcga_size = min(512, max(1, len(fold_data.target_unlabeled_dataset)))
        self._set_seed()

    def _set_seed(self) -> None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _get_data(self, graph_id_list, graph_type: str = "GDSC"):
        adjs, features = [], []
        graph = (
            self.drug_bundle.gdsc_drug_graph
            if graph_type == "GDSC"
            else self.drug_bundle.tcga_drug_graph
        )
        for gid in graph_id_list:
            key = str(int(gid.item()))
            entry = graph[key]
            adj = entry["adj"] if isinstance(entry, dict) else entry["adj"]
            feat = entry["feature"] if isinstance(entry, dict) else entry["feature"]
            adjs.append(torch.tensor(adj).unsqueeze(0))
            features.append(torch.tensor(feat).unsqueeze(0))
        features_t = torch.cat(features, dim=0).float().to(self.models.device)
        adjs_t = torch.cat(adjs, dim=0).float().to(self.models.device)
        return adjs_t, features_t

    def fit(self, checkpoint_prefix: str) -> FoldTrainingResult:
        cfg = self.config
        n_feat = self.fold_data.n_features
        train_ds = self.fold_data.source_train_dataset
        val_ds = self.fold_data.source_valid_dataset
        tcga_unlabel = self.fold_data.target_unlabeled_dataset

        # Empty labeled TCGA dataset – labels must not enter training loop
        empty_tcga = LabeledDataset([])

        da_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=len(train_ds) > cfg.batch_size
        )
        tcga_loader = DataLoader(
            tcga_unlabel,
            batch_size=min(self.tcga_size, len(tcga_unlabel)),
            shuffle=True,
            drop_last=len(tcga_unlabel) > self.tcga_size,
        )
        val_loader = DataLoader(
            val_ds, batch_size=max(len(val_ds), 1), shuffle=False
        )

        cancer_fe = self.models.cancer_fe
        classifier = self.models.classifier
        auto_encoder = self.models.auto_encoder

        opt_cancer = optim.Adam(cancer_fe.parameters(), lr=cfg.learning_rate)
        opt_cls = optim.Adam(classifier.parameters(), lr=cfg.learning_rate)
        opt_ae = optim.Adam(auto_encoder.parameters(), lr=cfg.learning_rate)
        sched_cancer = optim.lr_scheduler.StepLR(opt_cancer, step_size=1, gamma=0.9)
        sched_cls = optim.lr_scheduler.StepLR(opt_cls, step_size=1, gamma=0.9)
        sched_ae = optim.lr_scheduler.StepLR(opt_ae, step_size=1, gamma=0.9)

        pos_weight = torch.tensor([1.0]).to(self.models.device)
        gdsc_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        recon_loss_fn = nn.SmoothL1Loss()

        best_auc = -1.0
        best_epoch = -1
        wait = 0
        save_fe = save_cls = save_ae = None
        log_rows: list[dict] = []

        for epoch in range(cfg.max_epoch):
            if epoch >= 1:
                sched_cancer.step()
                sched_cls.step()
                sched_ae.step()

            cancer_fe.train()
            classifier.train()
            auto_encoder.train()

            for g, tcga_expr in zip(da_loader, cycle(tcga_loader)):
                graph_id, gdsc_expr, label = g
                label = label.view(graph_id.shape[0], 1).float().to(self.models.device)
                gdsc_adjs, gdsc_features = self._get_data(graph_id, "GDSC")
                gdsc_expr = gdsc_expr.to(self.models.device).float().view(-1, n_feat)
                tcga_expr = tcga_expr.view(-1, n_feat).to(self.models.device).float()

                for opt in (opt_cancer, opt_cls, opt_ae):
                    opt.zero_grad()

                gdsc_latent = cancer_fe(gdsc_expr)
                tcga_latent = cancer_fe(tcga_expr)
                source_pred = classifier(gdsc_features, gdsc_adjs, gdsc_latent)
                reconstruct = auto_encoder(tcga_latent)
                loss = gdsc_loss_fn(source_pred, label) + 0.1 * recon_loss_fn(
                    reconstruct, tcga_expr
                )
                loss.backward()
                opt_cancer.step()
                opt_cls.step()

                opt_ae.zero_grad()
                gdsc_latent = cancer_fe(gdsc_expr)
                tcga_latent = cancer_fe(tcga_expr)
                gdsc_recon = auto_encoder(gdsc_latent)
                tcga_recon = auto_encoder(tcga_latent)
                tcga_loss = recon_loss_fn(tcga_recon, tcga_expr)
                gdsc_loss = recon_loss_fn(gdsc_recon, gdsc_expr)
                margin = 0.3
                loss2 = gdsc_loss + max(0.0, margin - tcga_loss) * 0.02
                loss2.backward()
                opt_ae.step()

            # Source validation only (no TCGA labels)
            cancer_fe.eval()
            classifier.eval()
            auto_encoder.eval()
            labels, preds = [], []
            with torch.no_grad():
                for graph_id, expr_data, label in val_loader:
                    bs = graph_id.shape[0]
                    adjs, features = self._get_data(graph_id, "GDSC")
                    expr_data = expr_data.to(self.models.device).float().view(bs, n_feat)
                    expr = cancer_fe(expr_data)
                    pred = classifier(features, adjs, expr)
                    labels.extend(label.numpy().reshape(-1).tolist())
                    preds.extend(pred.sigmoid().cpu().numpy().reshape(-1).tolist())

            y = np.array(labels)
            p = np.array(preds)
            try:
                val_auc = float(metrics.roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
            except ValueError:
                val_auc = float("nan")

            log_rows.append({"epoch": epoch, "val_auroc": val_auc})
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                wait = 0
                save_fe = copy.deepcopy(cancer_fe)
                save_cls = copy.deepcopy(classifier)
                save_ae = copy.deepcopy(auto_encoder)
            else:
                wait += 1
                if wait > cfg.early_stop_patience:
                    break

        if save_fe is not None:
            self.models.cancer_fe = save_fe
            self.models.classifier = save_cls
            self.models.auto_encoder = save_ae

        save_checkpoint(self.models, checkpoint_prefix)
        return FoldTrainingResult(
            fold=self.fold_data.fold,
            models=self.models,
            best_checkpoint_prefix=checkpoint_prefix,
            best_epoch=best_epoch,
            best_val_auroc=best_auc,
            train_log=log_rows,
        )


def train_one_fold(
    fold_data: FoldDataBundle,
    drug_bundle: DrugRepresentationBundle,
    config: AACDRPipelineConfig,
    checkpoint_prefix: str,
) -> FoldTrainingResult:
    trainer = PipelineAADATrainer(config, fold_data, drug_bundle)
    return trainer.fit(checkpoint_prefix)
