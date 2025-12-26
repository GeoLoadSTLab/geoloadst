"""Space-time variogram helpers for STV fitting and directional ranges."""

from __future__ import annotations

from typing import Any

import numpy as np


def _trim_trailing_zero_bins(
    bins: np.ndarray,
    experimental: np.ndarray,
    eps_ratio: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Trim trailing bins where experimental semivariance is effectively zero or non-finite.

    Parameters
    ----------
    bins : np.ndarray
        Lag bin centers.
    experimental : np.ndarray
        Experimental semivariance values corresponding to bins.
    eps_ratio : float
        Ratio of max(experimental) below which a value is considered "zero".

    Returns
    -------
    trimmed_bins : np.ndarray
        Lag bins with trailing zeros removed.
    trimmed_exp : np.ndarray
        Experimental values with trailing zeros removed.
    n_valid : int
        Number of valid bins retained.
    """
    bins = np.asarray(bins)
    experimental = np.asarray(experimental)

    if len(bins) == 0:
        return bins, experimental, 0

    max_exp = np.nanmax(experimental)
    if not np.isfinite(max_exp) or max_exp == 0:
        # All zeros or non-finite; keep all bins as-is
        return bins, experimental, len(bins)

    eps = max_exp * eps_ratio

    # Find last index with a valid (finite and > eps) value
    last_valid = len(experimental) - 1
    while last_valid >= 0:
        val = experimental[last_valid]
        if np.isfinite(val) and val > eps:
            break
        last_valid -= 1

    n_valid = last_valid + 1
    if n_valid == 0:
        # Edge case: no valid bins; return at least first bin
        return bins[:1], experimental[:1], 1

    return bins[:n_valid], experimental[:n_valid], n_valid


def compute_stv(
    coords: np.ndarray,
    values_std: np.ndarray,
    dt_minutes: float = 15.0,
    x_lags: int = 12,
    t_lags: int = 8,
    maxlag: str | float = "median",
    model: str = "product-sum",
    max_pairs: int = 200_000,
    random_state: int | None = 42,
    trim_trailing_zeros: bool = True,
    eps_ratio: float = 1e-6,
) -> dict[str, Any]:
    """Fit a SpaceTimeVariogram while keeping pairwise computations bounded.

    Parameters
    ----------
    trim_trailing_zeros : bool
        If True, trim trailing lag bins where experimental semivariance is zero.
    eps_ratio : float
        Ratio of max experimental value below which a bin is considered zero.
    """
    from skgstat import SpaceTimeVariogram

    # Bound the pair count by subsampling coordinates if necessary
    n_points = coords.shape[0]
    pair_budget = max_pairs
    max_keep = n_points
    if pair_budget is not None and pair_budget > 0:
        # Solve k*(k-1)/2 <= pair_budget for k
        k = int((1 + np.sqrt(1 + 8 * pair_budget)) // 2)
        max_keep = min(n_points, max(2, k))

    if max_keep < n_points:
        rng = np.random.default_rng(random_state)
        keep_idx = rng.choice(n_points, size=max_keep, replace=False)
        coords_use = coords[keep_idx]
        values_use = values_std[keep_idx]
        print(
            f"[geoloadst] Subsampled buses for STV: {n_points} -> {max_keep} "
            f"(pair budget {max_pairs})"
        )
    else:
        coords_use = coords
        values_use = values_std
        keep_idx = None

    stv = SpaceTimeVariogram(
        coordinates=coords_use,
        values=values_use,
        x_lags=x_lags,
        t_lags=t_lags,
        maxlag=maxlag,
        model=model,
        verbose=False,
    )

    # Extract marginal variograms
    if hasattr(stv, "XMarginal"):
        Vx = stv.XMarginal
    else:
        Vx = stv.create_XMarginal()

    if hasattr(stv, "TMarginal"):
        Vt = stv.TMarginal
    else:
        Vt = stv.create_TMarginal()

    # Trim trailing zeros if requested
    if trim_trailing_zeros:
        x_bins_trimmed, x_exp_trimmed, n_valid_x = _trim_trailing_zero_bins(
            Vx.bins, Vx.experimental, eps_ratio
        )
        t_bins_trimmed, t_exp_trimmed, n_valid_t = _trim_trailing_zero_bins(
            Vt.bins, Vt.experimental, eps_ratio
        )
        max_valid_space_lag = x_bins_trimmed[-1] if n_valid_x > 0 else np.nan
        max_valid_time_lag = t_bins_trimmed[-1] if n_valid_t > 0 else np.nan
    else:
        x_bins_trimmed = Vx.bins
        x_exp_trimmed = Vx.experimental
        t_bins_trimmed = Vt.bins
        t_exp_trimmed = Vt.experimental
        max_valid_space_lag = Vx.bins[-1] if len(Vx.bins) > 0 else np.nan
        max_valid_time_lag = Vt.bins[-1] if len(Vt.bins) > 0 else np.nan

    # Extract spatial range and clamp to max valid lag
    try:
        space_range = Vx.effective_range
    except AttributeError:
        space_range = Vx.parameters[0] if len(Vx.parameters) > 0 else np.nan

    if np.isfinite(space_range) and np.isfinite(max_valid_space_lag):
        space_range = min(space_range, max_valid_space_lag)

    # Extract temporal range (in time steps) and clamp
    try:
        time_range_steps = Vt.effective_range
    except AttributeError:
        time_range_steps = Vt.parameters[0] if len(Vt.parameters) > 0 else np.nan

    if np.isfinite(time_range_steps) and np.isfinite(max_valid_time_lag):
        time_range_steps = min(time_range_steps, max_valid_time_lag)

    time_range_hours = time_range_steps * dt_minutes / 60.0

    return {
        "stv": stv,
        "space_range": space_range,
        "time_range_steps": time_range_steps,
        "time_range_hours": time_range_hours,
        # Primary keys
        "x_marginal": Vx,
        "t_marginal": Vt,
        # Aliases for backward compatibility
        "Vx": Vx,
        "Vt": Vt,
        # Trimmed data for plotting
        "x_bins_trimmed": x_bins_trimmed,
        "x_exp_trimmed": x_exp_trimmed,
        "t_bins_trimmed": t_bins_trimmed,
        "t_exp_trimmed": t_exp_trimmed,
        "max_valid_space_lag": max_valid_space_lag,
        "max_valid_time_lag": max_valid_time_lag,
        "indices_used": keep_idx,
    }


def compute_directional_variograms(
    coords: np.ndarray,
    values: np.ndarray,
    azimuths: list[int] | None = None,
    tolerance: float = 22.5,
    n_lags: int = 8,
    maxlag: float | None = None,
    model: str = "spherical",
) -> dict[str, Any]:
    """Directional variograms for a few azimuths; returns ranges and ellipse info."""
    from skgstat import DirectionalVariogram

    if azimuths is None:
        azimuths = [0, 45, 90, 135]

    # If values is 2D (N, T), compute RMS per node
    if values.ndim == 2:
        values = np.sqrt((values ** 2).mean(axis=1))

    # Estimate maxlag if not provided
    if maxlag is None:
        from scipy.spatial.distance import pdist

        dists = pdist(coords)
        maxlag = np.median(dists) * 1.2

    dir_variograms = {}
    dir_ranges = {}

    for az in azimuths:
        try:
            DV = DirectionalVariogram(
                coordinates=coords,
                values=values,
                azimuth=az,
                tolerance=tolerance,
                n_lags=n_lags,
                maxlag=maxlag,
                model=model,
            )

            try:
                r_eff = DV.effective_range
            except AttributeError:
                r_eff = DV.parameters[0] if len(DV.parameters) > 0 else np.nan

            dir_variograms[az] = DV
            dir_ranges[az] = r_eff

        except Exception:
            # Skip if fitting fails for this direction
            continue

    if not dir_ranges:
        return {
            "variograms": {},
            "ranges": {},
            "major_azimuth": None,
            "minor_azimuth": None,
            "semi_major": np.nan,
            "semi_minor": np.nan,
            "angle": 0.0,
        }

    # Find major and minor axes
    az_major = max(dir_ranges, key=dir_ranges.get)
    az_minor = min(dir_ranges, key=dir_ranges.get)

    return {
        "variograms": dir_variograms,
        "ranges": dir_ranges,
        "major_azimuth": az_major,
        "minor_azimuth": az_minor,
        "semi_major": dir_ranges[az_major],
        "semi_minor": dir_ranges[az_minor],
        "angle": az_major,
    }


def compute_local_variograms(
    coords: np.ndarray,
    values: np.ndarray,
    center_idx: int,
    k_neighbors: int = 40,
    n_lags: int = 6,
    model: str = "spherical",
) -> dict[str, Any]:
    """Local isotropic and two-direction variograms around one node."""
    from scipy.spatial.distance import cdist
    from skgstat import Variogram, DirectionalVariogram

    # Find k nearest neighbors
    dists = cdist(coords[center_idx : center_idx + 1], coords)[0]
    order = np.argsort(dists)
    k_eff = min(k_neighbors, len(coords))
    local_idx = order[:k_eff]

    coords_local = coords[local_idx]
    values_local = values[local_idx]

    # Check for sufficient variance
    if len(np.unique(values_local)) < 3:
        return {
            "iso_range": np.nan,
            "local_major": np.nan,
            "local_minor": np.nan,
            "local_angle": np.nan,
            "neighbor_indices": local_idx,
        }

    # Isotropic variogram
    try:
        V_iso = Variogram(
            coordinates=coords_local,
            values=values_local,
            n_lags=n_lags,
            model=model,
        )
        try:
            iso_range = V_iso.effective_range
        except AttributeError:
            iso_range = V_iso.parameters[0] if len(V_iso.parameters) > 0 else np.nan
    except Exception:
        iso_range = np.nan

    # Directional variograms (0° and 90°)
    ranges_loc = {}
    for az in [0, 90]:
        try:
            DV_local = DirectionalVariogram(
                coordinates=coords_local,
                values=values_local,
                azimuth=az,
                tolerance=30,
                n_lags=max(5, n_lags - 1),
                model=model,
            )
            try:
                r_eff = DV_local.effective_range
            except AttributeError:
                r_eff = DV_local.parameters[0] if len(DV_local.parameters) > 0 else np.nan
            ranges_loc[az] = r_eff
        except Exception:
            continue

    if ranges_loc:
        az_major = max(ranges_loc, key=ranges_loc.get)
        az_minor = min(ranges_loc, key=ranges_loc.get)
        local_major = ranges_loc[az_major]
        local_minor = ranges_loc[az_minor]
        local_angle = az_major
    else:
        local_major = np.nan
        local_minor = np.nan
        local_angle = np.nan

    return {
        "iso_range": iso_range,
        "local_major": local_major,
        "local_minor": local_minor,
        "local_angle": local_angle,
        "neighbor_indices": local_idx,
    }

