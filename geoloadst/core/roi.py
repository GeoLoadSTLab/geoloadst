"""ROI utilities for centered bounding boxes and bus filtering."""

from __future__ import annotations

import json
from typing import Iterable, Tuple

import numpy as np


def extract_xy_from_geo(geo_val: object) -> Tuple[float, float] | None:
    """Robustly extract (x, y) from pandapower bus.geo entry.

    Supports JSON strings (with {"coordinates": [x, y]}), lists/tuples, or dicts.
    Returns None if parsing fails or coordinates are incomplete.
    """
    try:
        if isinstance(geo_val, str):
            geo_val = json.loads(geo_val)
    except Exception:
        return None

    if isinstance(geo_val, dict):
        coords = geo_val.get("coordinates") or geo_val.get("coord") or geo_val.get("xy")
    else:
        coords = geo_val

    try:
        coords_list = list(coords)
    except Exception:
        return None

    if len(coords_list) < 2:
        return None

    try:
        x, y = float(coords_list[0]), float(coords_list[1])
    except Exception:
        return None

    return x, y


def compute_center_roi(coords: np.ndarray, roi_fraction: float) -> Tuple[float, float, float, float]:
    """Compute a centered bounding box covering roi_fraction of the full extent."""
    if coords is None or len(coords) == 0:
        raise ValueError("coords must be non-empty to compute center ROI.")
    if roi_fraction <= 0 or roi_fraction > 1:
        raise ValueError("roi_fraction must be in (0, 1].")

    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    width = (xmax - xmin) * roi_fraction
    height = (ymax - ymin) * roi_fraction
    half_w = 0.5 * width
    half_h = 0.5 * height
    return (cx - half_w, cx + half_w, cy - half_h, cy + half_h)


def filter_buses_in_roi(net: object, roi_bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Return indices of buses with coordinates inside roi_bbox (xmin,xmax,ymin,ymax)."""
    xmin, xmax, ymin, ymax = roi_bbox
    coords = []
    idxs: list[int] = []

    if not hasattr(net, "bus") or "geo" not in net.bus.columns:
        return np.array([], dtype=int)

    for idx, geo_val in net.bus["geo"].items():
        xy = extract_xy_from_geo(geo_val)
        if xy is None:
            continue
        x, y = xy
        if xmin <= x <= xmax and ymin <= y <= ymax:
            idxs.append(int(idx))
            coords.append([x, y])

    return np.array(idxs, dtype=int)

