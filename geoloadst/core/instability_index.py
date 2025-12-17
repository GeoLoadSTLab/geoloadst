"""Instability indices derived from detrended load time series."""

from __future__ import annotations

import numpy as np


def rms_instability(values_std: np.ndarray) -> np.ndarray:
    """RMS of standardized loads per node; a quick instability score."""
    return np.sqrt((values_std ** 2).mean(axis=1))


def classify_critical_nodes(
    instability_index: np.ndarray,
    quantile: float = 0.9,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Mark critical nodes by thresholding the instability distribution."""
    if threshold is None:
        threshold_value = np.quantile(instability_index, quantile)
    else:
        threshold_value = threshold

    critical_mask = instability_index >= threshold_value
    critical_indices = np.where(critical_mask)[0]

    return critical_mask, critical_indices, threshold_value


def rate_of_change(values: np.ndarray) -> np.ndarray:
    """Mean absolute first difference per node (RoCoL-style)."""
    diff = np.diff(values, axis=1)  # (N, T-1)
    return np.mean(np.abs(diff), axis=1)


def oscillation_rate(values: np.ndarray) -> np.ndarray:
    """Fraction of sign changes in a node's series (zeros borrow prior sign)."""
    N, T = values.shape
    osc_rate = np.zeros(N)

    for i in range(N):
        s = values[i, :]
        signs = np.sign(s)

        # Handle zeros by propagating previous sign
        for t in range(1, T):
            if signs[t] == 0:
                signs[t] = signs[t - 1]

        # Count sign changes
        sign_changes = np.sum(signs[1:] * signs[:-1] < 0)
        osc_rate[i] = sign_changes / max(T - 1, 1)

    return osc_rate


def oscillation_rate_from_diff(values: np.ndarray) -> np.ndarray:
    """Oscillation rate based on sign flips in first differences."""
    N, T = values.shape
    diff = np.diff(values, axis=1)  # (N, T-1)
    osc_rates = np.zeros(N)

    for i in range(N):
        signs = np.sign(diff[i, :])
        nz_mask = signs != 0

        if nz_mask.sum() <= 1:
            osc_rates[i] = 0.0
            continue

        changes = 0
        prev = None
        for k in range(len(signs)):
            if not nz_mask[k]:
                continue
            if prev is None:
                prev = signs[k]
            elif signs[k] != prev:
                changes += 1
                prev = signs[k]

        denom = max(nz_mask.sum() - 1, 1)
        osc_rates[i] = changes / denom

    return osc_rates

