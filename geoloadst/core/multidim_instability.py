"""Feature extraction, PCA, and clustering for multi-dimensional instability."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def build_instability_features(
    bus_load_df: pd.DataFrame,
    values_detrended: np.ndarray | None = None,
    include_voltage: bool = False,
    voltage_std: np.ndarray | None = None,
    voltage_sag: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Assemble per-node instability features (RMS, RoCoL, oscillation, optional voltage)."""
    values = bus_load_df.T.values  # (N, T)
    N, T = values.shape

    # Feature 1: Load RMS (detrended)
    if values_detrended is None:
        # Simple centering per node
        node_means = values.mean(axis=1, keepdims=True)
        values_centered = values - node_means
        load_rms = np.sqrt((values_centered ** 2).mean(axis=1))
    else:
        load_rms = np.sqrt((values_detrended ** 2).mean(axis=1))

    # Feature 2: Rate of Change (RoCoL)
    if values_detrended is not None:
        diff = np.diff(values_detrended, axis=1)
    else:
        diff = np.diff(values, axis=1)
    roc_mean = np.mean(np.abs(diff), axis=1)

    # Feature 3: Oscillation rate (sign changes)
    if values_detrended is not None:
        source = values_detrended
    else:
        source = values - values.mean(axis=1, keepdims=True)

    osc_rate = np.zeros(N)
    for i in range(N):
        signs = np.sign(source[i, :])
        # Propagate non-zero signs
        for t in range(1, T):
            if signs[t] == 0:
                signs[t] = signs[t - 1]
        osc_rate[i] = np.sum(signs[1:] * signs[:-1] < 0) / max(T - 1, 1)

    # Collect features
    features_list = [load_rms, roc_mean, osc_rate]
    feature_names = ["load_rms", "roc_mean", "osc_rate"]

    # Optional voltage features
    if include_voltage:
        if voltage_std is not None:
            features_list.append(voltage_std)
            feature_names.append("voltage_std")
        if voltage_sag is not None:
            features_list.append(voltage_sag)
            feature_names.append("voltage_sag")

    features = np.column_stack(features_list)

    return features, feature_names


def impute_features(
    features: np.ndarray, strategy: str = "median"
) -> np.ndarray:
    """Fill NaNs in the feature matrix (median by default)."""
    imputer = SimpleImputer(strategy=strategy)
    return imputer.fit_transform(features)


def pca_and_cluster(
    features: np.ndarray,
    n_clusters: int = 3,
    n_components: int = 2,
    impute: bool = True,
    impute_strategy: str = "median",
    random_state: int = 42,
) -> dict[str, Any]:
    """Standardize, run PCA, then KMeans; return the fitted objects and labels."""
    # Impute missing values
    if impute:
        features_clean = impute_features(features, strategy=impute_strategy)
    else:
        features_clean = features.copy()

    # Standardize features
    scaler = StandardScaler()
    features_std = scaler.fit_transform(features_clean)

    # PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_components = pca.fit_transform(features_std)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(features_std)

    return {
        "features_std": features_std,
        "pca_components": pca_components,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": pca.explained_variance_ratio_.sum(),
        "cluster_labels": cluster_labels,
        "cluster_centers": kmeans.cluster_centers_,
        "pca_model": pca,
        "scaler": scaler,
        "kmeans_model": kmeans,
    }


def cluster_feature_summary(
    features: np.ndarray,
    feature_names: list[str],
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """Per-cluster summary stats (mean/std/min/max) for each feature."""
    n_clusters = len(np.unique(cluster_labels))
    summary_data = []

    for k in range(n_clusters):
        mask = cluster_labels == k
        for i, fname in enumerate(feature_names):
            values = features[mask, i]
            summary_data.append(
                {
                    "cluster": k,
                    "feature": fname,
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                    "count": mask.sum(),
                }
            )

    return pd.DataFrame(summary_data)

