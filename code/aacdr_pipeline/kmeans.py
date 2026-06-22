"""K-means clustering on high-dimensional latent for cancer type evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def compute_kmeans_cancer_type_metrics(
    latent_dict: dict,
    fold: int,
    seed: int,
) -> pd.DataFrame:
    sample_ids = sorted(latent_dict.keys())
    X = np.asarray([latent_dict[s]["latent"] for s in sample_ids], dtype=np.float64)
    labels = [str(latent_dict[s].get("cancer_type", "Unknown")) for s in sample_ids]
    known_mask = np.array([l not in ("", "Unknown", "nan") for l in labels])
    if known_mask.sum() < 3:
        return pd.DataFrame(
            [
                {
                    "fold": fold,
                    "seed": seed,
                    "n_samples": len(sample_ids),
                    "n_cancer_types": 0,
                    "ari": float("nan"),
                    "nmi": float("nan"),
                    "silhouette": float("nan"),
                    "calinski_harabasz": float("nan"),
                    "davies_bouldin": float("nan"),
                }
            ]
        )

    Xk = X[known_mask]
    y_true = np.array([labels[i] for i in range(len(labels)) if known_mask[i]])
    unique_types = sorted(set(y_true.tolist()))
    n_types = len(unique_types)
    k = int(np.clip(n_types, 2, Xk.shape[0] - 1))

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    y_pred = km.fit_predict(Xk)

    type_to_id = {t: i for i, t in enumerate(unique_types)}
    y_enc = np.array([type_to_id[t] for t in y_true])

    sil = float("nan")
    if k > 1 and Xk.shape[0] > k:
        try:
            sil = float(silhouette_score(Xk, y_pred))
        except ValueError:
            pass

    return pd.DataFrame(
        [
            {
                "fold": fold,
                "seed": seed,
                "n_samples": int(Xk.shape[0]),
                "n_cancer_types": n_types,
                "n_clusters": k,
                "ari": float(adjusted_rand_score(y_enc, y_pred)),
                "nmi": float(normalized_mutual_info_score(y_enc, y_pred)),
                "silhouette": sil,
                "calinski_harabasz": float(calinski_harabasz_score(Xk, y_pred)),
                "davies_bouldin": float(davies_bouldin_score(Xk, y_pred)),
            }
        ]
    )
