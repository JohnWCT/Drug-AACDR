"""Wrap AACDR model for configurable input dim and latent extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import DNN, FE


@dataclass
class PipelineModels:
    cancer_fe: FE
    classifier: DNN
    auto_encoder: nn.Module
    n_features: int
    device: torch.device


class DynamicAutoEncoder(nn.Module):
    """AutoEncoder with configurable reconstruction output size."""

    def __init__(self, n_output: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, n_output),
        )

    def forward(self, expr: torch.Tensor) -> torch.Tensor:
        return self.encoder(expr)


def resolve_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_pipeline_models(n_features: int, device: str | None = None) -> PipelineModels:
    dev = resolve_device(device)
    cancer_fe = FE(n_input=n_features).to(dev)
    classifier = DNN().to(dev)
    auto_encoder = DynamicAutoEncoder(n_output=n_features).to(dev)
    return PipelineModels(
        cancer_fe=cancer_fe,
        classifier=classifier,
        auto_encoder=auto_encoder,
        n_features=n_features,
        device=dev,
    )


def _get_graph_batch(
    graph_ids: torch.Tensor,
    drug_graph: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    adjs, features = [], []
    for gid in graph_ids:
        key = str(int(gid.item()))
        entry = drug_graph[key]
        adj = entry["adj"] if isinstance(entry, dict) else entry["adj"]
        feat = entry["feature"] if isinstance(entry, dict) else entry["feature"]
        adjs.append(torch.tensor(adj).unsqueeze(0))
        features.append(torch.tensor(feat).unsqueeze(0))
    features_t = torch.cat(features, dim=0).float().to(device)
    adjs_t = torch.cat(adjs, dim=0).float().to(device)
    return adjs_t, features_t


@torch.no_grad()
def predict_labeled_dataset(
    models: PipelineModels,
    dataset,
    drug_graph: dict,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return graph_ids, labels, probabilities."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    models.cancer_fe.eval()
    models.classifier.eval()
    all_gids, all_labels, all_probs = [], [], []
    for graph_id, expr_data, label in loader:
        bs = graph_id.shape[0]
        adjs, features = _get_graph_batch(graph_id, drug_graph, models.device)
        expr_data = expr_data.to(models.device).float().view(bs, models.n_features)
        expr = models.cancer_fe(expr_data)
        pred = models.classifier(features, adjs, expr)
        probs = pred.sigmoid().cpu().numpy().reshape(-1)
        all_gids.append(graph_id.numpy())
        all_labels.append(label.numpy().reshape(-1))
        all_probs.append(probs)
    return (
        np.concatenate(all_gids),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


@torch.no_grad()
def extract_omics_latent(
    models: PipelineModels,
    expr_tensors: list[torch.Tensor] | torch.Tensor,
    batch_size: int = 512,
) -> np.ndarray:
    models.cancer_fe.eval()
    if isinstance(expr_tensors, list):
        x = torch.stack(expr_tensors)
    else:
        x = expr_tensors
    out = []
    for i in range(0, len(x), batch_size):
        batch = x[i : i + batch_size].to(models.device).float()
        latent = models.cancer_fe(batch).cpu().numpy()
        out.append(latent)
    return np.concatenate(out, axis=0)


def save_checkpoint(models: PipelineModels, path_prefix: str) -> None:
    torch.save(models.cancer_fe.state_dict(), f"{path_prefix}_fe.pt")
    torch.save(models.classifier.state_dict(), f"{path_prefix}_dnn.pt")
    torch.save(models.auto_encoder.state_dict(), f"{path_prefix}_ae.pt")


def load_checkpoint(
    models: PipelineModels, path_prefix: str
) -> PipelineModels:
    models.cancer_fe.load_state_dict(torch.load(f"{path_prefix}_fe.pt", map_location=models.device))
    models.classifier.load_state_dict(torch.load(f"{path_prefix}_dnn.pt", map_location=models.device))
    models.auto_encoder.load_state_dict(torch.load(f"{path_prefix}_ae.pt", map_location=models.device))
    return models
