"""Geospatial plotting helpers for networks, clusters, and anisotropy overlays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pandapower import pandapowerNet


def _create_bus_gdf(
    bus_ids: np.ndarray,
    coords: np.ndarray,
    data: dict[str, np.ndarray] | None = None,
    crs: str = "EPSG:4326",
) -> "Any":
    """Create a GeoDataFrame for bus points."""
    import geopandas as gpd
    from shapely.geometry import Point

    geometry = [Point(x, y) for x, y in coords]

    gdf_data = {"bus": bus_ids}
    if data is not None:
        gdf_data.update(data)

    return gpd.GeoDataFrame(gdf_data, geometry=geometry, crs=crs)


def _create_lines_gdf(
    net: "pandapowerNet",
    bus_to_coord: dict[int, np.ndarray],
    crs: str = "EPSG:4326",
) -> "Any":
    """Create a GeoDataFrame for network lines."""
    import geopandas as gpd
    from shapely.geometry import LineString

    line_geoms = []
    line_ids = []

    for idx, line in net.line.iterrows():
        fb = int(line["from_bus"])
        tb = int(line["to_bus"])
        if fb in bus_to_coord and tb in bus_to_coord:
            p1 = bus_to_coord[fb]
            p2 = bus_to_coord[tb]
            line_geoms.append(LineString([p1, p2]))
            line_ids.append(idx)

    if not line_geoms:
        return None

    return gpd.GeoDataFrame({"line": line_ids}, geometry=line_geoms, crs=crs)


def plot_network_topology(
    net: "pandapowerNet",
    bus_ids: np.ndarray,
    coords: np.ndarray,
    critical_mask: np.ndarray | None = None,
    ax: "Axes | None" = None,
    title: str = "Network topology",
    figsize: tuple[float, float] = (8, 6),
) -> "Figure":
    """
    Plot network topology with buses and lines.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network.
    bus_ids : np.ndarray
        Array of bus IDs.
    coords : np.ndarray
        N×2 array of coordinates.
    critical_mask : np.ndarray, optional
        Boolean mask for critical nodes.
    ax : Axes, optional
        Matplotlib axes to plot on.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    bus_to_coord = {int(bus_ids[i]): coords[i] for i in range(len(bus_ids))}

    # Plot lines
    for _, line in net.line.iterrows():
        fb = int(line["from_bus"])
        tb = int(line["to_bus"])
        if fb in bus_to_coord and tb in bus_to_coord:
            x1, y1 = bus_to_coord[fb]
            x2, y2 = bus_to_coord[tb]
            ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=0.7, zorder=1)

    # Plot all buses
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=15,
        c="gray",
        alpha=0.7,
        label="Buses",
        zorder=2,
    )

    # Plot critical buses
    if critical_mask is not None and critical_mask.any():
        critical_coords = coords[critical_mask]
        ax.scatter(
            critical_coords[:, 0],
            critical_coords[:, 1],
            s=60,
            c="red",
            edgecolors="black",
            label="Critical buses",
            zorder=3,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_critical_nodes_map(
    bus_ids: np.ndarray,
    coords: np.ndarray,
    critical_bus_ids: np.ndarray,
    net: "pandapowerNet | None" = None,
    ax: "Axes | None" = None,
    title: str = "Critical nodes map",
    figsize: tuple[float, float] = (8, 6),
) -> "Figure":
    """
    Plot map highlighting critical nodes.

    Parameters
    ----------
    bus_ids : np.ndarray
        Array of all bus IDs.
    coords : np.ndarray
        N×2 array of coordinates.
    critical_bus_ids : np.ndarray
        Bus IDs of critical nodes.
    net : pandapowerNet, optional
        Network for drawing lines.
    ax : Axes, optional
        Matplotlib axes to plot on.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    bus_to_coord = {int(bus_ids[i]): coords[i] for i in range(len(bus_ids))}

    # Create GeoDataFrames
    bus_gdf = _create_bus_gdf(bus_ids, coords)

    critical_coords = np.array([bus_to_coord[int(b)] for b in critical_bus_ids])
    critical_gdf = _create_bus_gdf(critical_bus_ids, critical_coords)

    # Plot lines if network provided
    if net is not None:
        lines_gdf = _create_lines_gdf(net, bus_to_coord)
        if lines_gdf is not None:
            lines_gdf.plot(ax=ax, linewidth=0.5, color="lightgray")

    # Plot buses
    bus_gdf.plot(ax=ax, markersize=5, color="gray", alpha=0.7, label="Buses")
    critical_gdf.plot(
        ax=ax, markersize=30, color="red", edgecolor="black", label="Critical buses"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend()
    ax.axis("equal")
    plt.tight_layout()

    return fig


def plot_lisa_clusters_map(
    bus_ids: np.ndarray,
    coords: np.ndarray,
    cluster_codes: np.ndarray,
    net: "pandapowerNet | None" = None,
    ax: "Axes | None" = None,
    title: str = "LISA clusters",
    figsize: tuple[float, float] = (8, 6),
) -> "Figure":
    """
    Plot LISA cluster map with standard color coding.

    Parameters
    ----------
    bus_ids : np.ndarray
        Array of bus IDs.
    coords : np.ndarray
        N×2 array of coordinates.
    cluster_codes : np.ndarray
        LISA cluster codes (0=NS, 1=HH, 2=LL, 3=LH, 4=HL).
    net : pandapowerNet, optional
        Network for drawing lines.
    ax : Axes, optional
        Matplotlib axes to plot on.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    bus_to_coord = {int(bus_ids[i]): coords[i] for i in range(len(bus_ids))}

    cluster_labels = {
        0: "Not Significant",
        1: "High-High",
        2: "Low-Low",
        3: "Low-High",
        4: "High-Low",
    }

    cluster_colors = {
        0: "#cccccc",
        1: "#d7191c",
        2: "#2c7bb6",
        3: "#fdae61",
        4: "#abd9e9",
    }

    # Plot lines if network provided
    if net is not None:
        for _, line in net.line.iterrows():
            fb = int(line["from_bus"])
            tb = int(line["to_bus"])
            if fb in bus_to_coord and tb in bus_to_coord:
                x1, y1 = bus_to_coord[fb]
                x2, y2 = bus_to_coord[tb]
                ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=0.3, zorder=1)

    # Plot each cluster
    for code, label in cluster_labels.items():
        mask = cluster_codes == code
        if mask.any():
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=20,
                c=cluster_colors[code],
                label=label,
                edgecolors="k",
                linewidths=0.3,
                zorder=2,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend(title="LISA cluster", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_cluster_map(
    bus_ids: np.ndarray,
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    net: "pandapowerNet | None" = None,
    ax: "Axes | None" = None,
    title: str = "Cluster map",
    cmap: str = "tab10",
    figsize: tuple[float, float] = (8, 6),
) -> "Figure":
    """
    Plot map with cluster coloring (e.g., from KMeans).

    Parameters
    ----------
    bus_ids : np.ndarray
        Array of bus IDs.
    coords : np.ndarray
        N×2 array of coordinates.
    cluster_labels : np.ndarray
        Cluster assignment per node.
    net : pandapowerNet, optional
        Network for drawing lines.
    ax : Axes, optional
        Matplotlib axes to plot on.
    title : str
        Plot title.
    cmap : str
        Colormap name.
    figsize : tuple
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    gdf = _create_bus_gdf(bus_ids, coords, {"cluster": cluster_labels})
    bus_to_coord = {int(bus_ids[i]): coords[i] for i in range(len(bus_ids))}

    # Plot lines if network provided
    if net is not None:
        lines_gdf = _create_lines_gdf(net, bus_to_coord)
        if lines_gdf is not None:
            lines_gdf.plot(ax=ax, linewidth=0.5, color="lightgray")

    # Plot buses with cluster colors
    gdf.plot(
        ax=ax,
        column="cluster",
        cmap=cmap,
        categorical=True,
        markersize=30,
        edgecolor="black",
        legend=True,
        legend_kwds={"title": "Cluster"},
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_feature_map(
    bus_ids: np.ndarray,
    coords: np.ndarray,
    feature_values: np.ndarray,
    feature_name: str = "Feature",
    net: "pandapowerNet | None" = None,
    ax: "Axes | None" = None,
    title: str | None = None,
    cmap: str = "Reds",
    figsize: tuple[float, float] = (8, 6),
) -> "Figure":
    """
    Plot map with continuous feature coloring.

    Parameters
    ----------
    bus_ids : np.ndarray
        Array of bus IDs.
    coords : np.ndarray
        N×2 array of coordinates.
    feature_values : np.ndarray
        Feature values per node.
    feature_name : str
        Name of the feature (for legend).
    net : pandapowerNet, optional
        Network for drawing lines.
    ax : Axes, optional
        Matplotlib axes to plot on.
    title : str, optional
        Plot title (defaults to feature_name).
    cmap : str
        Colormap name.
    figsize : tuple
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    import geopandas as gpd

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    gdf = _create_bus_gdf(bus_ids, coords, {feature_name: feature_values})
    bus_to_coord = {int(bus_ids[i]): coords[i] for i in range(len(bus_ids))}

    # Plot lines if network provided
    if net is not None:
        lines_gdf = _create_lines_gdf(net, bus_to_coord)
        if lines_gdf is not None:
            lines_gdf.plot(ax=ax, linewidth=0.5, color="lightgray")

    # Plot buses with feature colors
    gdf.plot(
        ax=ax,
        column=feature_name,
        cmap=cmap,
        markersize=30,
        edgecolor="black",
        legend=True,
        legend_kwds={"label": feature_name},
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title or feature_name)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_industrial_cluster_map(
    bus_ids: np.ndarray,
    coords: np.ndarray,
    industrial_mask: np.ndarray,
    ax: "Axes | None" = None,
    title: str = "Industrial cluster",
    figsize: tuple[float, float] = (8, 6),
) -> "Figure":
    """
    Plot map highlighting industrial vs non-industrial nodes.

    Parameters
    ----------
    bus_ids : np.ndarray
        Array of bus IDs.
    coords : np.ndarray
        N×2 array of coordinates.
    industrial_mask : np.ndarray
        Boolean mask for industrial nodes.
    ax : Axes, optional
        Matplotlib axes to plot on.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    import geopandas as gpd

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    gdf = _create_bus_gdf(bus_ids, coords, {"industrial": industrial_mask})

    # Plot non-industrial
    gdf[~gdf["industrial"]].plot(
        ax=ax, markersize=5, color="lightgrey", label="Non-industrial"
    )

    # Plot industrial
    gdf[gdf["industrial"]].plot(
        ax=ax, markersize=15, color="red", label="Industrial cluster"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_anisotropic_ellipses(
    coords: np.ndarray,
    center_indices: np.ndarray,
    semi_major: np.ndarray,
    semi_minor: np.ndarray,
    angles: np.ndarray,
    net: "pandapowerNet | None" = None,
    bus_ids: np.ndarray | None = None,
    critical_mask: np.ndarray | None = None,
    ax: "Axes | None" = None,
    title: str = "Anisotropic instability footprints",
    figsize: tuple[float, float] = (8, 6),
) -> "Figure":
    """
    Plot anisotropic ellipses around critical nodes.

    Parameters
    ----------
    coords : np.ndarray
        N×2 array of coordinates.
    center_indices : np.ndarray
        Indices of nodes to draw ellipses around.
    semi_major : np.ndarray
        Semi-major axis values per node.
    semi_minor : np.ndarray
        Semi-minor axis values per node.
    angles : np.ndarray
        Orientation angles in degrees per node.
    net : pandapowerNet, optional
        Network for drawing lines.
    bus_ids : np.ndarray, optional
        Bus IDs (for line drawing).
    critical_mask : np.ndarray, optional
        Boolean mask for highlighting critical nodes.
    ax : Axes, optional
        Matplotlib axes to plot on.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if bus_ids is not None:
        bus_to_coord = {int(bus_ids[i]): coords[i] for i in range(len(bus_ids))}
    else:
        bus_to_coord = {i: coords[i] for i in range(len(coords))}

    # Plot lines if network provided
    if net is not None and bus_ids is not None:
        for _, line in net.line.iterrows():
            fb = int(line["from_bus"])
            tb = int(line["to_bus"])
            if fb in bus_to_coord and tb in bus_to_coord:
                x1, y1 = bus_to_coord[fb]
                x2, y2 = bus_to_coord[tb]
                ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=0.5, zorder=1)

    # Plot all buses
    ax.scatter(
        coords[:, 0], coords[:, 1], s=5, c="gray", alpha=0.4, label="Buses", zorder=2
    )

    # Plot critical buses
    if critical_mask is not None:
        ax.scatter(
            coords[critical_mask, 0],
            coords[critical_mask, 1],
            s=20,
            c="red",
            alpha=0.7,
            edgecolors="black",
            label="Critical",
            zorder=3,
        )

    # Plot ellipses
    for i, idx in enumerate(center_indices):
        if np.isnan(semi_major[i]) or np.isnan(semi_minor[i]):
            continue

        x0, y0 = coords[idx]
        ell = Ellipse(
            (x0, y0),
            width=2 * semi_major[i],
            height=2 * semi_minor[i],
            angle=angles[i],
            fill=False,
            linestyle="-",
            edgecolor="blue",
            linewidth=1.2,
            alpha=0.8,
        )
        ax.add_patch(ell)

    # Legend
    legend_elements = [
        Line2D([0], [0], color="lightgray", lw=1, label="Lines"),
        Line2D(
            [0], [0], marker="o", color="gray", label="Buses",
            markerfacecolor="gray", markersize=5, linestyle="None"
        ),
        Line2D(
            [0], [0], marker="o", color="red", label="Critical buses",
            markerfacecolor="red", markersize=6, linestyle="None"
        ),
        Line2D([0], [0], color="blue", lw=1.5, label="Anisotropic range"),
    ]
    ax.legend(handles=legend_elements, loc="best")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_lisa_comparison(
    coords: np.ndarray,
    clusters_base: np.ndarray,
    clusters_scenario: np.ndarray,
    time_label: str = "",
    figsize: tuple[float, float] = (14, 6),
) -> "Figure":
    """
    Plot side-by-side LISA cluster comparison.

    Parameters
    ----------
    coords : np.ndarray
        N×2 array of coordinates.
    clusters_base : np.ndarray
        LISA cluster codes for baseline.
    clusters_scenario : np.ndarray
        LISA cluster codes for scenario.
    time_label : str
        Time label for titles.
    figsize : tuple
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    cluster_names = {
        0: "Not Significant",
        1: "High-High",
        2: "Low-High",
        3: "Low-Low",
        4: "High-Low",
    }
    cluster_colors = {
        0: "lightgrey",
        1: "red",
        2: "blue",
        3: "navy",
        4: "orange",
    }

    # Base plot
    for code, name in cluster_names.items():
        mask = clusters_base == code
        if mask.any():
            axes[0].scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=5,
                color=cluster_colors[code],
                label=name if code != 0 else "Not significant",
            )
    axes[0].set_title(f"LISA clusters (BASE)\n{time_label}")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    # Scenario plot
    for code, name in cluster_names.items():
        mask = clusters_scenario == code
        if mask.any():
            axes[1].scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=5,
                color=cluster_colors[code],
                label=name if code != 0 else "Not significant",
            )
    axes[1].set_title(f"LISA clusters (SCENARIO)\n{time_label}")
    axes[1].set_xlabel("X")

    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="upper right", title="LISA cluster")

    plt.tight_layout()
    return fig

