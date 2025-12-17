"""Industrial day/night scenario: detect industrial cluster and rescale loads."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def detect_industrial_cluster(
    coords: np.ndarray,
    mean_load: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Cluster buses spatially and mark the cluster with highest mean load as industrial."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(coords)

    # Find cluster with highest mean load
    cluster_means = []
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_means.append(mean_load[mask].mean())

    industrial_cluster_id = int(np.argmax(cluster_means))
    industrial_mask = cluster_labels == industrial_cluster_id

    return cluster_labels, industrial_cluster_id, industrial_mask


def apply_daynight_pattern(
    bus_load_df: pd.DataFrame,
    industrial_mask: np.ndarray,
    dt_minutes: float = 15.0,
    day_start_h: float = 8.0,
    day_end_h: float = 20.0,
    day_factor: float = 3.0,
    night_factor: float = 0.3,
) -> pd.DataFrame:
    """Scale industrial buses up by day_factor and down by night_factor; return the new series."""
    T = bus_load_df.shape[0]
    time_hours = np.arange(T) * (dt_minutes / 60.0)

    # Identify day/night periods
    day_mask = (time_hours >= day_start_h) & (time_hours < day_end_h)

    # Get industrial column names
    bus_ids = bus_load_df.columns.values
    industrial_cols = bus_ids[industrial_mask]

    # Create scenario DataFrame
    bus_load_scen = bus_load_df.copy()

    # Apply factors
    bus_load_scen.loc[day_mask, industrial_cols] *= day_factor
    bus_load_scen.loc[~day_mask, industrial_cols] *= night_factor

    return bus_load_scen


def apply_industrial_daynight_pattern(
    bus_load_df: pd.DataFrame,
    coords: np.ndarray,
    dt_minutes: float = 15.0,
    n_clusters: int = 3,
    day_start_h: float = 8.0,
    day_end_h: float = 20.0,
    day_factor: float = 3.0,
    night_factor: float = 0.3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Cluster nodes, pick the industrial cluster, apply day/night scaling, and return base+scenario data."""
    mean_load = bus_load_df.mean(axis=0).values
    bus_ids = bus_load_df.columns.values

    # Detect industrial cluster
    cluster_labels, industrial_cluster_id, industrial_mask = detect_industrial_cluster(
        coords, mean_load, n_clusters=n_clusters, random_state=random_state
    )

    # Apply day/night pattern
    bus_load_scen = apply_daynight_pattern(
        bus_load_df,
        industrial_mask,
        dt_minutes=dt_minutes,
        day_start_h=day_start_h,
        day_end_h=day_end_h,
        day_factor=day_factor,
        night_factor=night_factor,
    )

    # Compute time axis
    T = bus_load_df.shape[0]
    time_hours = np.arange(T) * (dt_minutes / 60.0)
    day_mask = (time_hours >= day_start_h) & (time_hours < day_end_h)

    # Get column subsets
    industrial_cols = bus_ids[industrial_mask]
    non_industrial_cols = bus_ids[~industrial_mask]

    return {
        "bus_load_base": bus_load_df,
        "bus_load_scenario": bus_load_scen,
        "industrial_mask": industrial_mask,
        "cluster_labels": cluster_labels,
        "industrial_cluster_id": industrial_cluster_id,
        "industrial_cols": industrial_cols,
        "non_industrial_cols": non_industrial_cols,
        "time_hours": time_hours,
        "day_mask": day_mask,
    }


def compare_scenario_moran(
    bus_load_base: pd.DataFrame,
    bus_load_scenario: pd.DataFrame,
    W: Any,
    permutations: int = 0,
) -> dict[str, np.ndarray]:
    """Moran's I over time for base vs scenario, plus their difference."""
    from geoloadst.core.moran import moran_time_series

    I_base = moran_time_series(bus_load_base, W, permutations=permutations)
    I_scen = moran_time_series(bus_load_scenario, W, permutations=permutations)

    return {
        "moran_base": I_base,
        "moran_scenario": I_scen,
        "delta": I_scen - I_base,
    }


def compute_scenario_lisa_comparison(
    bus_load_base: pd.DataFrame,
    bus_load_scenario: pd.DataFrame,
    W: Any,
    time_idx: int,
    p_threshold: float = 0.05,
    permutations: int = 199,
) -> dict[str, Any]:
    """Compare LISA clusters at one snapshot for base vs scenario."""
    from esda.moran import Moran_Local

    from geoloadst.core.moran import classify_lisa_clusters

    y_base = bus_load_base.iloc[time_idx].values
    y_scen = bus_load_scenario.iloc[time_idx].values

    lisa_base = Moran_Local(y_base, W, permutations=permutations)
    lisa_scen = Moran_Local(y_scen, W, permutations=permutations)

    clusters_base = classify_lisa_clusters(lisa_base, p_threshold=p_threshold)
    clusters_scen = classify_lisa_clusters(lisa_scen, p_threshold=p_threshold)

    return {
        "clusters_base": clusters_base,
        "clusters_scenario": clusters_scen,
        "lisa_base": lisa_base,
        "lisa_scenario": lisa_scen,
    }

