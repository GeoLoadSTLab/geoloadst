"""Lightweight helpers for comparing scenarios and sketching vulnerability indices."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compare_scenarios(
    baseline_metrics: dict[str, Any],
    scenario_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Pairwise deltas/ratios between baseline and scenario metric dicts."""
    results = {"delta": {}, "ratio": {}, "pct_change": {}}

    common_keys = set(baseline_metrics.keys()) & set(scenario_metrics.keys())

    for key in common_keys:
        base_val = baseline_metrics[key]
        scen_val = scenario_metrics[key]

        if isinstance(base_val, (int, float)) and isinstance(scen_val, (int, float)):
            results["delta"][key] = scen_val - base_val
            if base_val != 0:
                results["ratio"][key] = scen_val / base_val
                results["pct_change"][key] = ((scen_val - base_val) / base_val) * 100
            else:
                results["ratio"][key] = np.nan
                results["pct_change"][key] = np.nan

        elif isinstance(base_val, np.ndarray) and isinstance(scen_val, np.ndarray):
            results["delta"][key] = scen_val - base_val
            with np.errstate(divide="ignore", invalid="ignore"):
                results["ratio"][key] = np.where(base_val != 0, scen_val / base_val, np.nan)
                results["pct_change"][key] = np.where(
                    base_val != 0, ((scen_val - base_val) / base_val) * 100, np.nan
                )

    return results


def vulnerability_index(
    instability: np.ndarray,
    topology_degree: np.ndarray,
    weights: tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    """Blend instability and degree into a simple vulnerability score."""
    # Normalize both to [0, 1]
    inst_norm = (instability - instability.min()) / (instability.max() - instability.min() + 1e-10)
    deg_norm = (topology_degree - topology_degree.min()) / (topology_degree.max() - topology_degree.min() + 1e-10)

    return weights[0] * inst_norm + weights[1] * deg_norm


def critical_node_summary(
    bus_ids: np.ndarray,
    instability: np.ndarray,
    topology_metrics: dict[str, np.ndarray],
    n_top: int = 10,
) -> pd.DataFrame:
    """Summarize the top-n nodes by instability, adding available topo metrics."""
    data = {
        "bus_id": bus_ids,
        "instability": instability,
    }

    for metric_name, values in topology_metrics.items():
        if metric_name != "bus_ids" and isinstance(values, np.ndarray):
            data[metric_name] = values

    df = pd.DataFrame(data)
    df = df.sort_values("instability", ascending=False).head(n_top)

    return df.reset_index(drop=True)

