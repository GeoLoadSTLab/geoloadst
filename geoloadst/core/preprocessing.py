"""Preprocess load time series by peeling off trends and scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def detrend_and_standardize(
    bus_load_df: pd.DataFrame,
    coords: np.ndarray,
    dt_minutes: float = 15.0,
    remove_temporal_mean: bool = True,
    remove_spatial_trend: bool = True,
    scale: bool = True,
) -> np.ndarray:
    """
    Detrend and standardize bus load time series.

    Parameters
    ----------
    bus_load_df : pd.DataFrame
        DataFrame of shape (time × buses) containing load values.
    coords : np.ndarray
        N×2 array of bus coordinates corresponding to columns of bus_load_df.
    dt_minutes : float, default 15.0
        Time step resolution in minutes (for documentation purposes).
    remove_temporal_mean : bool, default True
        Whether to remove the global temporal mean.
    remove_spatial_trend : bool, default True
        Whether to remove the linear spatial trend.
    scale : bool, default True
        Whether to scale to unit variance.

    Returns
    -------
    np.ndarray
        Detrended and standardized values with shape (N, T),
        where N is the number of buses and T is the number of time steps.

    Notes
    -----
    Input is (T, N); output is (N, T) to match variogram routines.
    """
    # Transpose to (N, T) for per-node operations
    values_matrix = bus_load_df.T.values.astype(np.float64)
    N, T = values_matrix.shape

    # Ensure coords shape matches
    if coords.shape[0] != N:
        raise ValueError(
            f"Coordinate count ({coords.shape[0]}) does not match "
            f"number of buses ({N}) in the DataFrame."
        )

    # Step 1: Remove global temporal mean
    if remove_temporal_mean:
        global_time_mean = values_matrix.mean(axis=0, keepdims=True)  # (1, T)
        values_matrix = values_matrix - global_time_mean

    # Step 2: Remove linear spatial trend
    if remove_spatial_trend:
        node_mean_after_t = values_matrix.mean(axis=1)  # (N,)
        lin = LinearRegression()
        lin.fit(coords, node_mean_after_t)
        spatial_trend = lin.predict(coords)  # (N,)
        values_matrix = values_matrix - spatial_trend[:, np.newaxis]

    # Step 3: Scale to unit variance
    if scale:
        scaler = StandardScaler(with_mean=False)
        values_matrix = scaler.fit_transform(values_matrix)

    return values_matrix


def center_per_node(bus_load_df: pd.DataFrame) -> np.ndarray:
    """Subtract each node's temporal mean (returns N × T)."""
    values_matrix = bus_load_df.T.values.astype(np.float64)  # (N, T)
    node_means = values_matrix.mean(axis=1, keepdims=True)  # (N, 1)
    return values_matrix - node_means


def compute_temporal_diff(values: np.ndarray) -> np.ndarray:
    """First differences along time, shape (N, T-1)."""
    return np.diff(values, axis=1)

