"""
geoloadst - Spatial and spatio-temporal load instability analysis for distribution networks.

This package provides tools for analyzing load instability patterns in power distribution
networks using SimBench/pandapower, including:
- Spatio-temporal variogram analysis
- Multi-dimensional instability feature extraction
- Moran's I (global and local) spatial autocorrelation
- Topological correlation analysis
- Industrial day/night scenario modeling
"""

__version__ = "0.1.0"

from geoloadst.api import InstabilityAnalyzer

__all__ = ["InstabilityAnalyzer", "__version__"]

