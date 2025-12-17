"""Moran's I utilities: KNN weights, global Moran, LISA, and time series."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from esda.moran import Moran, Moran_Local
    from libpysal.weights import W


def build_knn_weights(
    coords: np.ndarray, k: int = 8, transform: str = "r"
) -> "W":
    """KNN spatial weights (row-standardized by default)."""
    from libpysal.weights import KNN

    W = KNN(coords, k=k)
    W.transform = transform
    return W


def global_moran(
    x: np.ndarray, W: "W", permutations: int = 999
) -> "Moran":
    """Global Moran's I; set permutations=0 for a quick analytic run."""
    from esda.moran import Moran

    return Moran(x, W, permutations=permutations)


def local_moran(
    x: np.ndarray, W: "W", permutations: int = 199
) -> "Moran_Local":
    """Local Moran (LISA) with a small permutation budget by default."""
    from esda.moran import Moran_Local

    return Moran_Local(x, W, permutations=permutations)


def local_moran_clusters(
    x: np.ndarray, W: "W", alpha: float = 0.05, permutations: int = 199
) -> np.ndarray:
    """Return LISA cluster codes (0=NS, 1=HH, 2=LL, 3=LH, 4=HL)."""
    from esda.moran import Moran_Local

    lisa = Moran_Local(x, W, permutations=permutations)
    sig = lisa.p_sim < alpha

    cluster = np.zeros(len(x), dtype=int)
    q = lisa.q

    # Map quadrants to cluster codes
    cluster[sig & (q == 1)] = 1  # High-High
    cluster[sig & (q == 3)] = 2  # Low-Low
    cluster[sig & (q == 2)] = 3  # Low-High
    cluster[sig & (q == 4)] = 4  # High-Low

    return cluster


def classify_lisa_clusters(
    lisa: "Moran_Local", p_threshold: float = 0.05
) -> np.ndarray:
    """Map a Moran_Local object to cluster codes with a p-value cutoff."""
    sig = lisa.p_sim < p_threshold
    cluster = np.zeros(len(lisa.q), dtype=int)
    q = lisa.q

    cluster[sig & (q == 1)] = 1  # High-High
    cluster[sig & (q == 3)] = 2  # Low-Low
    cluster[sig & (q == 2)] = 3  # Low-High
    cluster[sig & (q == 4)] = 4  # High-Low

    return cluster


def moran_time_series(
    load_df: pd.DataFrame, W: "W", permutations: int = 0
) -> np.ndarray:
    """Global Moran's I for every snapshot in a time series."""
    from esda.moran import Moran

    T = load_df.shape[0]
    moran_values = np.zeros(T)

    for t in range(T):
        y = load_df.iloc[t].values
        mi = Moran(y, W, permutations=permutations)
        moran_values[t] = mi.I

    return moran_values


def get_cluster_labels() -> dict[int, str]:
    """Readable labels for LISA cluster codes."""
    return {
        0: "Not Significant",
        1: "High-High",
        2: "Low-Low",
        3: "Low-High",
        4: "High-Low",
    }


def get_cluster_colors() -> dict[int, str]:
    """Default colors for plotting LISA clusters."""
    return {
        0: "#cccccc",  # Gray for non-significant
        1: "#d7191c",  # Red for High-High
        2: "#2c7bb6",  # Blue for Low-Low
        3: "#fdae61",  # Orange for Low-High
        4: "#abd9e9",  # Light blue for High-Low
    }


def moran_analysis_summary(
    mean_load: np.ndarray,
    instability: np.ndarray,
    W: "W",
    permutations: int = 999,
) -> dict[str, Any]:
    """Convenience wrapper: global Moran for mean/instability plus LISA clusters."""
    from esda.moran import Moran, Moran_Local

    mi_mean = Moran(mean_load, W, permutations=permutations)
    mi_inst = Moran(instability, W, permutations=permutations)

    lisa_mean = Moran_Local(mean_load, W, permutations=min(199, permutations))
    lisa_inst = Moran_Local(instability, W, permutations=min(199, permutations))

    clusters_mean = classify_lisa_clusters(lisa_mean)
    clusters_inst = classify_lisa_clusters(lisa_inst)

    return {
        "moran_mean_load": mi_mean,
        "moran_instability": mi_inst,
        "lisa_mean_load": lisa_mean,
        "lisa_instability": lisa_inst,
        "clusters_mean_load": clusters_mean,
        "clusters_instability": clusters_inst,
    }

