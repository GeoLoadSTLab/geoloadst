"""Network topology helpers: graphs, centralities, and simple correlations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import networkx as nx
    from pandapower import pandapowerNet


def build_network_graph(
    net: "pandapowerNet",
    bus_ids: np.ndarray | None = None,
    include_trafos: bool = True,
    use_length_weight: bool = True,
) -> "nx.Graph":
    """Build a NetworkX graph from a pandapower net; optional bus subset and weights."""
    import networkx as nx

    G = nx.Graph()

    # Add nodes
    if bus_ids is not None:
        G.add_nodes_from(bus_ids)
        bus_set = set(int(b) for b in bus_ids)
    else:
        G.add_nodes_from(net.bus.index)
        bus_set = None

    # Add line edges
    for _, line in net.line.iterrows():
        fb = int(line["from_bus"])
        tb = int(line["to_bus"])

        if bus_set is not None and (fb not in bus_set or tb not in bus_set):
            continue

        if use_length_weight:
            length = line.get("length_km", np.nan)
            if not np.isfinite(length) or length <= 0:
                length = 1.0
        else:
            length = 1.0

        G.add_edge(fb, tb, weight=length)

    # Add transformer edges
    if include_trafos and hasattr(net, "trafo") and len(net.trafo) > 0:
        for _, trafo in net.trafo.iterrows():
            hv_bus = int(trafo["hv_bus"])
            lv_bus = int(trafo["lv_bus"])

            if bus_set is not None and (hv_bus not in bus_set or lv_bus not in bus_set):
                continue

            G.add_edge(hv_bus, lv_bus, weight=1.0)

    return G


def compute_topological_metrics(
    G: "nx.Graph", bus_ids: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Degree, betweenness, and closeness for the chosen buses."""
    import networkx as nx

    if bus_ids is None:
        bus_ids = np.array(list(G.nodes()))

    # Compute centralities for entire graph
    degree_dict = dict(G.degree(bus_ids))
    betweenness_dict = nx.betweenness_centrality(G, weight="weight", normalized=True)
    closeness_dict = nx.closeness_centrality(G, distance="weight")

    # Extract values for requested bus_ids
    degree = np.array([degree_dict.get(int(b), 0) for b in bus_ids])
    betweenness = np.array([betweenness_dict.get(int(b), 0.0) for b in bus_ids])
    closeness = np.array([closeness_dict.get(int(b), 0.0) for b in bus_ids])

    return {
        "degree": degree,
        "betweenness": betweenness,
        "closeness": closeness,
        "bus_ids": bus_ids,
    }


def correlate_instability_topology(
    instability: np.ndarray,
    topology_metrics: dict[str, np.ndarray],
) -> dict[str, float]:
    """Pearson correlations between instability and each topological metric."""

    def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        """Compute correlation with safety checks."""
        if len(a) != len(b):
            return np.nan
        if np.all(a == a[0]) or np.all(b == b[0]):
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    correlations = {}

    for metric_name in ["degree", "betweenness", "closeness"]:
        if metric_name in topology_metrics:
            corr = safe_corr(instability, topology_metrics[metric_name])
            correlations[metric_name] = corr

    return correlations


def get_neighbors_isotropic(
    coords: np.ndarray, node_idx: int, radius: float
) -> np.ndarray:
    """Neighbors inside a given radius (excludes the center)."""
    from scipy.spatial.distance import cdist

    dists = cdist(coords[node_idx : node_idx + 1], coords)[0]
    mask = dists <= radius
    mask[node_idx] = False
    return np.where(mask)[0]


def get_neighbors_directional(
    coords: np.ndarray,
    node_idx: int,
    azimuth: float,
    radius: float,
    tolerance: float = 22.5,
) -> np.ndarray:
    """Neighbors within an azimuthal cone of width ±tolerance degrees."""
    dx = coords[:, 0] - coords[node_idx, 0]
    dy = coords[:, 1] - coords[node_idx, 1]
    dist = np.sqrt(dx ** 2 + dy ** 2)

    angles_rad = np.arctan2(dy, dx)
    angles_deg = np.degrees(angles_rad)

    # Normalize angle difference to [-180, 180]
    diff = (angles_deg - azimuth + 180) % 360 - 180

    mask = (dist <= radius) & (np.abs(diff) <= tolerance)
    mask[node_idx] = False

    return np.where(mask)[0]

