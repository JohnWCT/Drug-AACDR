"""FID and latent distribution metrics between source and target."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance


def calculate_fid(source: np.ndarray, target: np.ndarray) -> float:
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    diff = mu_s - mu_t
    cov_s = np.cov(source, rowvar=False) + np.eye(source.shape[1]) * 1e-6
    cov_t = np.cov(target, rowvar=False) + np.eye(target.shape[1]) * 1e-6
    if cov_s.ndim == 0:
        cov_s = np.array([[float(cov_s)]])
    if cov_t.ndim == 0:
        cov_t = np.array([[float(cov_t)]])
    covmean = sqrtm(cov_s @ cov_t)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(cov_s + cov_t - 2 * covmean))


def calculate_mmd(source: np.ndarray, target: np.ndarray, gamma: float | None = None) -> float:
    if gamma is None:
        gamma = 1.0 / max(source.shape[1], 1)
    n = min(500, source.shape[0], target.shape[0])
    rng = np.random.default_rng(0)
    xs = source[rng.choice(source.shape[0], n, replace=False)]
    xt = target[rng.choice(target.shape[0], n, replace=False)]

    def kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xx = np.sum(x * x, axis=1, keepdims=True)
        yy = np.sum(y * y, axis=1, keepdims=True)
        xy = x @ y.T
        return np.exp(-gamma * (xx - 2 * xy + yy.T))

    return float(kernel(xs, xs).mean() + kernel(xt, xt).mean() - 2 * kernel(xs, xt).mean())


def calculate_wasserstein(source: np.ndarray, target: np.ndarray) -> float:
    dists = [wasserstein_distance(source[:, j], target[:, j]) for j in range(source.shape[1])]
    return float(np.mean(dists))


def compute_latent_distribution_metrics(
    source_dict: dict,
    target_dict: dict,
    fold: int,
    seed: int,
) -> pd.DataFrame:
    src = np.asarray([v["latent"] for v in source_dict.values()], dtype=np.float64)
    tgt = np.asarray([v["latent"] for v in target_dict.values()], dtype=np.float64)
    if len(src) == 0 or len(tgt) == 0:
        return pd.DataFrame(
            [
                {
                    "fold": fold,
                    "seed": seed,
                    "source_n": len(src),
                    "target_n": len(tgt),
                    "fid": float("nan"),
                    "mmd": float("nan"),
                    "wasserstein_mean": float("nan"),
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "fold": fold,
                "seed": seed,
                "source_n": len(src),
                "target_n": len(tgt),
                "fid": calculate_fid(src, tgt),
                "mmd": calculate_mmd(src, tgt),
                "wasserstein_mean": calculate_wasserstein(src, tgt),
            }
        ]
    )
