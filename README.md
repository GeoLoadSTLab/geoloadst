# geoloadst
**Spatial and spatio-temporal load instability analysis for distribution networks**

`geoloadst` is a Python package for analyzing load instability patterns in power distribution networks using SimBench/pandapower. It provides tools for:

- **Spatio-temporal variogram analysis** - Quantify spatial and temporal correlation structures in load anomalies
- **Multi-dimensional instability features** - Extract RMS, rate of change, oscillation metrics with PCA & clustering
- **Moran's I spatial autocorrelation** - Global and Local Moran (LISA) for hotspot detection
- **Topological analysis** - Correlate instability with network topology (degree, betweenness centrality)
- **Scenario modeling** - Industrial day/night load pattern analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/GeoLoadSTLab/geoloadst.git geoloadst-repo
cd geoloadst-repo

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Dependencies

- `numpy`, `pandas`, `matplotlib`, `scipy`
- `scikit-learn` - PCA, KMeans, preprocessing
- `pandapower`, `simbench` - Power network modeling
- `scikit-gstat` - Geostatistical variogram analysis
- `geopandas`, `shapely` - Geospatial data handling
- `libpysal`, `esda` - Spatial statistics (Moran's I)
- `networkx` - Graph/topology analysis

## Quick Start

The examples below default to small ROI/time windows and bounded pairs to stay memory-friendly.

### Example 1 — Small ROI Moran + LISA map (plots)

```python
import matplotlib.pyplot as plt
import simbench
from geoloadst import InstabilityAnalyzer
from geoloadst.viz.maps import plot_lisa_clusters_map

net = simbench.get_simbench_net("1-complete_data-mixed-all-1-sw")
analyzer = InstabilityAnalyzer(net, roi=(10.8, 11.7, 53.1, 53.6), time_window=(0, 48), dt_minutes=15.0)
analyzer.prepare_data()
analyzer.compute_spatiotemporal_instability(max_buses=200, max_times=48, max_pairs=50_000)
moran = analyzer.compute_moran_analysis(k_neighbors=8, permutations=99)

fig, ax = plot_lisa_clusters_map(
    analyzer.bus_ids,
    analyzer.coords,
    moran["clusters_mean_load"],
    net=net,
    title="LISA clusters (mean load)",
)
plt.show()
```

### Example 2 — Instability overlay with critical buses (plots)

```python
import matplotlib.pyplot as plt
import simbench
from geoloadst import InstabilityAnalyzer
from geoloadst.viz.maps import plot_instability_overlay

net = simbench.get_simbench_net("1-complete_data-mixed-all-1-sw")
analyzer = InstabilityAnalyzer(net, roi=(10.8, 11.7, 53.1, 53.6), time_window=(0, 48), dt_minutes=15.0)
analyzer.prepare_data()
stv = analyzer.compute_spatiotemporal_instability(max_buses=200, max_times=48, max_pairs=50_000)

fig, ax = plot_instability_overlay(
    stv["bus_ids_used"],
    stv["coords_used"],
    stv["instability_index"],
    stv["critical_mask"],
    net=net,
    title="Critical buses (small ROI)",
)
plt.show()
```

### Example 3 — Quick summary (minimal/no plotting)

```python
import simbench
import matplotlib.pyplot as plt

from geoloadst import InstabilityAnalyzer
from geoloadst.viz.maps import plot_lisa_clusters_map, plot_network_topology
from geoloadst.viz.plots import plot_instability_histogram

# 1) Load network
net = simbench.get_simbench_net("1-complete_data-mixed-all-1-sw")

# 2) Small ROI + short time window (low RAM)
analyzer = InstabilityAnalyzer(
    net,
    roi=(10.8, 11.7, 53.1, 53.6),
    time_window=(0, 48),
    dt_minutes=15.0,
)
analyzer.prepare_data()

# 3) Resource-safe instability (avoid huge pairwise matrices)
analyzer.compute_spatiotemporal_instability(
    max_buses=150,
    max_times=48,
    max_pairs=50_000,
)

# --- Plot A: Network (ROI) + Instability heat (RMS anomaly)
fig = plot_network_topology(net=net, bus_ids=analyzer.bus_ids, coords=analyzer.coords, title="Network (ROI)")
plt.show()

# --- Plot B: Instability distribution
plot_instability_histogram(analyzer.instability_index, quantile=0.9, title="Instability distribution (ROI buses)")
plt.show()

# 4) Moran / LISA on the SAME subset (must match dimensions)
moran = analyzer.compute_moran_analysis()

# --- Plot C: LISA clusters (mean load) on top of network
plot_lisa_clusters_map(
    analyzer.bus_ids,
    analyzer.coords,
    moran["clusters_mean_load"],
    net=net,
    title="Network (ROI) + LISA clusters (Mean Load)",
)
plt.show()
```

## Memory Notes / RAM Warning

- Spatio-temporal variograms and pairwise distance computations can be heavy.
- On laptops or limited RAM, shrink the spatial ROI (roi=(xmin, xmax, ymin, ymax)) and shorten the time window (time_window=(start, end)).
- Use the STV knobs: max_buses, max_times, and max_pairs to bound work. Lower them if you still hit memory pressure.
- When you set max_buses, the analyzer replaces bus_ids/coords/bus_load_df with the subsampled active subset (full ROI copies stay in _bus_ids_full/_coords_full/_bus_load_df_full). Run Moran/LISA/plots with analyzer.bus_ids and analyzer.coords to avoid dimension mismatches.

## Troubleshooting

- Symptom: `geoloadst.__file__` is `None` and `geoloadst.__path__` shows `_NamespacePath(...)`. This usually means Python is picking up a folder that shadows the installed package (e.g., running Python from a parent directory named `geoloadst`).
- Fix:
  1) From the repo root (the one with `pyproject.toml`), run:
     ```bash
     python -m pip uninstall -y geoloadst
     python -m pip install -e .
     ```
  2) Confirm install:
     ```bash
     python -c "import geoloadst; print(geoloadst.__file__)"`
     ```
  3) Restart the Jupyter kernel so it picks up the edited install.
- Always import via the public API:
  ```python
  from geoloadst import InstabilityAnalyzer
  ```

## Package Structure

```
geoloadst/
├── __init__.py
├── api.py                  # High-level InstabilityAnalyzer class
├── io/
│   ├── __init__.py
│   └── simbench_adapter.py # Network loading, coordinate extraction, load profiles
├── core/
│   ├── __init__.py
│   ├── preprocessing.py    # Detrending and standardization
│   ├── instability_index.py # RMS instability, critical node classification
│   ├── spatiotemporal.py   # Space-time variogram analysis
│   ├── multidim_instability.py # Feature extraction, PCA, clustering
│   ├── moran.py            # Moran's I (global and local)
│   ├── topology.py         # NetworkX graph metrics
│   └── resilience.py       # Scenario comparison utilities
├── viz/
│   ├── __init__.py
│   ├── plots.py            # Histograms, variograms, time series
│   └── maps.py             # Geospatial maps with GeoPandas
├── scenarios/
│   ├── __init__.py
│   └── industrial_daynight.py # Industrial load pattern scenario
└── examples/
    └── example_simbench_basic.py
```

## Core Modules

### `geoloadst.io.simbench_adapter`

Functions for loading SimBench networks and extracting data:

```python
from geoloadst.io import load_simbench_network, extract_bus_coordinates, build_bus_load_timeseries

net = load_simbench_network("1-complete_data-mixed-all-1-sw")
bus_ids, coords, bus_to_coord = extract_bus_coordinates(net)
bus_load_df = build_bus_load_timeseries(net, bus_ids, max_times=96)
```

### `geoloadst.core.preprocessing`

Detrending and standardization:

```python
from geoloadst.core import detrend_and_standardize

values_std = detrend_and_standardize(bus_load_df, coords, dt_minutes=15.0)
# Returns (N, T) array with temporal and spatial trends removed
```

### `geoloadst.core.spatiotemporal`

Space-time variogram analysis:

```python
from geoloadst.core import compute_stv, compute_directional_variograms

stv_results = compute_stv(coords, values_std, dt_minutes=15.0)
dir_results = compute_directional_variograms(coords, instability_index)
```

### `geoloadst.core.moran`

Moran's I spatial autocorrelation:

```python
from geoloadst.core import build_knn_weights, global_moran, local_moran_clusters

W = build_knn_weights(coords, k=8)
moran = global_moran(mean_load, W)
clusters = local_moran_clusters(mean_load, W, alpha=0.05)
# clusters: 0=NS, 1=HH, 2=LL, 3=LH, 4=HL
```

### `geoloadst.scenarios.industrial_daynight`

Industrial day/night scenario:

```python
from geoloadst.scenarios import apply_industrial_daynight_pattern

results = apply_industrial_daynight_pattern(
    bus_load_df, coords,
    dt_minutes=15.0,
    day_factor=3.0,
    night_factor=0.3,
)
```

## Visualization

```python
from geoloadst.viz.plots import (
    plot_instability_histogram,
    plot_variogram_marginals,
    plot_pca_clusters,
    plot_moran_timeseries,
)
from geoloadst.viz.maps import (
    plot_network_topology,
    plot_lisa_clusters_map,
    plot_cluster_map,
)

# Plot instability distribution
plot_instability_histogram(instability_index, quantile=0.9)

# Plot variogram results
plot_variogram_marginals(stv_results['stv'])

# Plot LISA clusters on map
plot_lisa_clusters_map(bus_ids, coords, cluster_codes, net=net)
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{geoloadst,
  title = {geoloadst: Spatial and spatio-temporal load instability analysis},
  year = {2024},
  url = {https://github.com/yourusername/geoloadst}
}
```

