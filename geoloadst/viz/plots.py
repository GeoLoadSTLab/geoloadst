"""Plotting utilities for histograms, variograms, PCA views, and time series."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_instability_histogram(
    instability_index: np.ndarray,
    threshold: float | None = None,
    quantile: float = 0.9,
    ax: "Axes | None" = None,
    title: str = "Instability index distribution",
    xlabel: str = "Instability index (RMS)",
    **kwargs: Any,
) -> "Figure":
    """Histogram of instability with an optional percentile marker."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    hist_kwargs = {"bins": 30, "alpha": 0.7}
    hist_kwargs.update(kwargs)

    ax.hist(instability_index, **hist_kwargs)

    if threshold is None:
        threshold = np.quantile(instability_index, quantile)

    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        label=f"{quantile*100:.0f}th percentile ({threshold:.3f})",
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_variogram_marginals(
    stv_results: dict[str, Any],
    figsize: tuple[float, float] = (12, 5),
) -> "Figure":
    """Spatial and temporal marginals from an STV fit."""
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    Vx = stv_results["x_marginal"]
    Vt = stv_results["t_marginal"]
    space_range = stv_results["space_range"]
    time_range_steps = stv_results["time_range_steps"]
    time_range_hours = stv_results["time_range_hours"]

    # Spatial marginal
    axs[0].plot(Vx.bins, Vx.experimental, "o-", label="Experimental")
    axs[0].plot(Vx.bins, Vx.fitted_model(Vx.bins), "-", label="Model")
    axs[0].axvline(
        space_range,
        color="red",
        linestyle="--",
        label=f"Range ≈ {space_range:.2f}",
    )
    axs[0].set_title("Spatial marginal variogram")
    axs[0].set_xlabel("Spatial lag")
    axs[0].set_ylabel("Semivariance")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # Temporal marginal
    axs[1].plot(Vt.bins, Vt.experimental, "o-", label="Experimental")
    axs[1].plot(Vt.bins, Vt.fitted_model(Vt.bins), "-", label="Model")
    axs[1].axvline(
        time_range_steps,
        color="red",
        linestyle="--",
        label=f"Range ≈ {time_range_steps:.1f} steps\n≈ {time_range_hours:.1f} h",
    )
    axs[1].set_title("Temporal marginal variogram")
    axs[1].set_xlabel("Time lag (steps)")
    axs[1].set_ylabel("Semivariance")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    return fig


def plot_directional_variograms(
    dir_results: dict[str, Any],
    ax: "Axes | None" = None,
    title: str = "Directional variograms",
    fontsize: int | None = None,
    labelsize: int | None = None,
    ticksize: int | None = None,
    legend_fontsize: int | None = None,
) -> "Figure":
    """Directional variograms for each azimuth provided."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    for az, DV in dir_results["variograms"].items():
        ax.plot(DV.bins, DV.experimental, marker="o", linestyle="-", label=f"{az}°")

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("Spatial lag", fontsize=labelsize)
    ax.set_ylabel("Semivariance", fontsize=labelsize)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Azimuth", fontsize=legend_fontsize, title_fontsize=legend_fontsize)
    if ticksize is not None:
        ax.tick_params(axis="both", labelsize=ticksize)
    plt.tight_layout()

    return fig


def plot_polar_ranges(
    dir_results: dict[str, Any],
    ax: "Axes | None" = None,
    title: str = "Directional instability radius",
) -> "Figure":
    """Polar plot of directional ranges."""
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, polar=True)
    else:
        fig = ax.get_figure()

    ranges = dir_results["ranges"]
    angles_rad = np.deg2rad(list(ranges.keys()))
    values = np.array(list(ranges.values()))

    # Close the polygon
    angles_rad_closed = np.append(angles_rad, angles_rad[0])
    values_closed = np.append(values, values[0])

    ax.plot(angles_rad_closed, values_closed, marker="o")
    ax.fill(angles_rad_closed, values_closed, alpha=0.3)
    ax.set_title(title)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    plt.tight_layout()

    return fig


def plot_directional_ranges_polar(
    dir_results: dict[str, Any] | dict[int | float, float],
    title: str = "Directional ranges (polar)",
    fontsize: int | None = None,
    labelsize: int | None = None,
    ticksize: int | None = None,
    ax: "Axes | None" = None,
) -> "Figure":
    """Alias of plot_polar_ranges with font sizing controls."""
    # Accept either dir_results dict with "ranges" key or direct ranges dict
    if isinstance(dir_results, dict) and "ranges" in dir_results:
        ranges = dir_results["ranges"]
    else:
        ranges = dir_results

    fig = plot_polar_ranges({"ranges": ranges}, ax=ax, title=title)

    # Apply font sizing if provided
    polar_ax = fig.axes[0] if fig.axes else ax
    if polar_ax is not None:
        if fontsize is not None:
            polar_ax.set_title(title, fontsize=fontsize)
        if labelsize is not None:
            polar_ax.set_xlabel(polar_ax.get_xlabel(), fontsize=labelsize)
            polar_ax.set_ylabel(polar_ax.get_ylabel(), fontsize=labelsize)
        if ticksize is not None:
            polar_ax.tick_params(axis="both", labelsize=ticksize)

    return fig


def plot_pca_clusters(
    pca_components: np.ndarray,
    cluster_labels: np.ndarray,
    ax: "Axes | None" = None,
    title: str = "PCA space with clusters",
    cmap: str = "tab10",
) -> "Figure":
    """2D PCA scatter with cluster coloring."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    n_clusters = len(np.unique(cluster_labels))

    for k in range(n_clusters):
        mask = cluster_labels == k
        ax.scatter(
            pca_components[mask, 0],
            pca_components[mask, 1],
            s=50,
            label=f"Cluster {k}",
            alpha=0.8,
        )

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    return fig


def plot_moran_timeseries(
    time_hours: np.ndarray,
    moran_values: np.ndarray,
    moran_scenario: np.ndarray | None = None,
    day_start_h: float | None = None,
    day_end_h: float | None = None,
    ax: "Axes | None" = None,
    title: str = "Temporal evolution of Moran's I",
) -> "Figure":
    """Time series of Moran's I, optionally comparing a scenario and shading daytime."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    if moran_scenario is not None:
        ax.plot(time_hours, moran_values, label="Base", linestyle="--", marker="o", ms=3)
        ax.plot(time_hours, moran_scenario, label="Scenario", linewidth=2)
    else:
        ax.plot(time_hours, moran_values, marker="o", ms=3)

    ax.axhline(0.0, color="k", linewidth=0.8)

    if day_start_h is not None and day_end_h is not None:
        ax.axvspan(day_start_h, day_end_h, color="yellow", alpha=0.1, label="Day period")

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Moran's I")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    return fig


def plot_correlation_scatter(
    x: np.ndarray,
    y: np.ndarray,
    color_by: np.ndarray | None = None,
    ax: "Axes | None" = None,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Scatter plot",
    cmap: str = "viridis",
    colorbar_label: str | None = None,
) -> "Figure":
    """Scatter plot with optional color encoding and colorbar."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    if color_by is not None:
        sc = ax.scatter(x, y, c=color_by, cmap=cmap, s=40, edgecolors="k", alpha=0.9)
        cbar = plt.colorbar(sc, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    else:
        ax.scatter(x, y, s=40, edgecolors="k", alpha=0.9)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_mean_load_evolution(
    time_hours: np.ndarray,
    mean_base_industrial: np.ndarray,
    mean_base_other: np.ndarray,
    mean_scen_industrial: np.ndarray | None = None,
    mean_scen_other: np.ndarray | None = None,
    day_start_h: float | None = None,
    day_end_h: float | None = None,
    ax: "Axes | None" = None,
    title: str = "Mean load evolution",
) -> "Figure":
    """Mean load over time for industrial vs other nodes, base vs scenario."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    ax.plot(time_hours, mean_base_industrial, label="Industrial (base)", linestyle="--")
    ax.plot(time_hours, mean_base_other, label="Others (base)", linestyle=":")

    if mean_scen_industrial is not None:
        ax.plot(time_hours, mean_scen_industrial, label="Industrial (scenario)", linewidth=2)
    if mean_scen_other is not None:
        ax.plot(time_hours, mean_scen_other, label="Others (scenario)", linewidth=2)

    if day_start_h is not None and day_end_h is not None:
        ax.axvspan(day_start_h, day_end_h, color="yellow", alpha=0.1, label="Day period")

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Mean load [MW]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig

