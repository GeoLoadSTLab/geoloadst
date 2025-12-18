# geoloadst
A geospatial Python toolbox for analysing load instability and spatial autocorrelation in power distribution networks.
=======

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
git clone https://github.com/GeoLoadSTLab/geoloadst.git
cd geoloadst

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

```python
import simbench
from geoloadst import InstabilityAnalyzer

# Load a SimBench network
net = simbench.get_simbench_net("1-complete_data-mixed-all-1-sw")

# Create analyzer with a spatial ROI
analyzer = InstabilityAnalyzer(
    net,
    roi=(10.8, 11.7, 53.1, 53.6),  # x_min, x_max, y_min, y_max
    time_window=(0, 96),            # 96 time steps = 24h at 15-min resolution
    dt_minutes=15.0,
)

# Prepare data (extract coords, build load time series)
analyzer.prepare_data()

# Run spatio-temporal instability analysis
stv_results = analyzer.compute_spatiotemporal_instability()
print(f"Spatial correlation range: {stv_results['stv']['space_range']:.2f}")
print(f"Temporal correlation range: {stv_results['stv']['time_range_hours']:.1f} hours")
print(f"Critical nodes: {len(stv_results['critical_bus_ids'])}")

# Multi-dimensional feature analysis with clustering
multi_results = analyzer.compute_multidim_instability(n_clusters=3)
print(f"PCA explained variance: {multi_results['pca_results']['cumulative_variance']:.1%}")

# Moran's I spatial autocorrelation
moran_results = analyzer.compute_moran_analysis()
print(f"Global Moran's I (mean load): {moran_results['moran_mean_load'].I:.4f}")
print(f"Global Moran's I (instability): {moran_results['moran_instability'].I:.4f}")

# Industrial day/night scenario
scenario_results = analyzer.run_industrial_daynight_scenario()
print(f"Industrial nodes: {scenario_results['scenario_data']['industrial_mask'].sum()}")

# Get summary DataFrame
summary_df = analyzer.get_summary()
print(summary_df.head())
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

