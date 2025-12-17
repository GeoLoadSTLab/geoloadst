"""Helpers for SimBench/pandapower grids: load a case, pull bus coords,
and build absolute load time series per bus."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandapower import pandapowerNet


def load_simbench_network(code: str) -> "pandapowerNet":
    """Load a SimBench grid by code, erroring if load profiles are missing."""
    import simbench

    net = simbench.get_simbench_net(code)

    if not hasattr(net, "profiles") or "load" not in net.profiles:
        raise RuntimeError(
            f"Network '{code}' does not contain load profiles. "
            "Make sure you are using a SimBench grid with time-series profiles."
        )

    return net


def _extract_xy_from_geo(geo_str: str | None) -> list[float]:
    """Parse a net.bus['geo'] entry into [x, y]; return NaNs if it can't be read."""
    if pd.isna(geo_str):
        return [np.nan, np.nan]

    try:
        gj = json.loads(geo_str)
    except (json.JSONDecodeError, TypeError):
        return [np.nan, np.nan]

    if isinstance(gj, dict) and "coordinates" in gj:
        coords = gj["coordinates"]
        if len(coords) >= 2:
            return [float(coords[0]), float(coords[1])]
    elif isinstance(gj, (list, tuple)) and len(gj) >= 2:
        return [float(gj[0]), float(gj[1])]

    return [np.nan, np.nan]


def extract_bus_coordinates(
    net: "pandapowerNet",
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Return bus IDs, coordinates, and a convenience mapping of bus→coord."""
    bus_coords_series = net.bus["geo"].apply(_extract_xy_from_geo)
    coords_all = np.vstack(bus_coords_series.values)

    # Keep only buses with valid coordinates
    valid_mask = ~np.isnan(coords_all).any(axis=1)
    coords = coords_all[valid_mask, :]
    bus_ids = net.bus.index[valid_mask].to_numpy()

    bus_to_coord = {int(b): coords[i] for i, b in enumerate(bus_ids)}

    return bus_ids, coords, bus_to_coord


def select_roi_buses(
    bus_ids: np.ndarray,
    coords: np.ndarray,
    roi: tuple[float, float, float, float] | None = None,
    roi_fraction: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick buses inside a bounding box or a centered fraction of the extent."""
    if roi is not None:
        x_min, x_max, y_min, y_max = roi
    elif roi_fraction is not None:
        minx, miny = coords.min(axis=0)
        maxx, maxy = coords.max(axis=0)
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0

        width = (maxx - minx) * roi_fraction
        height = (maxy - miny) * roi_fraction

        x_min = cx - width / 2.0
        x_max = cx + width / 2.0
        y_min = cy - height / 2.0
        y_max = cy + height / 2.0
    else:
        # No ROI filtering - return all
        return bus_ids.copy(), coords.copy()

    roi_mask = (
        (coords[:, 0] >= x_min)
        & (coords[:, 0] <= x_max)
        & (coords[:, 1] >= y_min)
        & (coords[:, 1] <= y_max)
    )

    return bus_ids[roi_mask], coords[roi_mask]


def build_bus_load_timeseries(
    net: "pandapowerNet",
    bus_ids: np.ndarray | list[int] | None = None,
    max_times: int = 96,
) -> pd.DataFrame:
    """Build absolute load profiles per bus from relative SimBench profiles and p_mw.
    Loads on the same bus are summed; optionally limit buses and time span."""
    # Relative load profiles from SimBench
    rel_profiles = net.profiles["load"]
    rel_profiles_small = rel_profiles.iloc[:max_times, :].copy()

    # Optionally limit to a bus subset
    if bus_ids is not None:
        bus_ids_set = set(int(b) for b in bus_ids)
        loads_mask = net.load["bus"].isin(bus_ids_set)
        load_indices = net.load.index[loads_mask]
    else:
        load_indices = net.load.index

    # Build absolute profiles for each load
    abs_profiles_dict = {}
    load_bus_list = []

    for lid in load_indices:
        row = net.load.loc[lid]
        profile_name = row.get("profile", None)
        p_nom = row["p_mw"]
        bus = int(row["bus"])

        if pd.isna(profile_name):
            # No profile assigned - use constant 1.0
            rel_series = np.ones(len(rel_profiles_small))
        else:
            # Try different column naming conventions
            # SimBench uses "{profile_name}_pload" or just "{profile_name}"
            col_name_pload = f"{profile_name}_pload"
            if col_name_pload in rel_profiles_small.columns:
                rel_series = rel_profiles_small[col_name_pload].values
            elif profile_name in rel_profiles_small.columns:
                rel_series = rel_profiles_small[profile_name].values
            else:
                # Profile not found - use constant 1.0
                rel_series = np.ones(len(rel_profiles_small))

        abs_profiles_dict[lid] = rel_series * p_nom
        load_bus_list.append(bus)

    if not abs_profiles_dict:
        raise ValueError("No loads found for the specified buses.")

    # Create DataFrame: time × loads
    abs_load_df = pd.DataFrame(
        data=abs_profiles_dict, index=rel_profiles_small.index
    )

    # Convert to bus-level by grouping loads on the same bus
    abs_load_df.columns = load_bus_list

    # Group by bus (sum loads on the same bus)
    bus_load_df = abs_load_df.T.groupby(level=0).sum().T

    # Filter to requested bus_ids and ensure they have data
    if bus_ids is not None:
        available_buses = bus_load_df.columns.intersection(bus_ids)
        bus_load_df = bus_load_df[available_buses]

    return bus_load_df


def get_bus_to_coord_mapping(
    bus_ids: np.ndarray, coords: np.ndarray
) -> dict[int, np.ndarray]:
    """Small helper to map bus IDs to their coordinate rows."""
    return {int(bus_ids[i]): coords[i] for i in range(len(bus_ids))}

