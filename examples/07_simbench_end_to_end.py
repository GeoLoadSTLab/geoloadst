"""End-to-end example: SimBench ROI, STV, directional, and maps (no basemap)."""

import matplotlib.pyplot as plt
import simbench

from geoloadst import InstabilityAnalyzer
from geoloadst.viz.plots import (
    plot_instability_histogram,
    plot_variogram_marginals,
    plot_directional_variograms,
    plot_polar_ranges,
)
from geoloadst.viz.maps import (
    plot_topology_with_critical_buses,
    plot_sample_critical_with_global_ellipse,
    plot_geopandas_map,
    plot_local_anisotropic_ellipses,
)


def main() -> None:
    net = simbench.get_simbench_net("1-complete_data-mixed-all-1-sw")
    analyzer = InstabilityAnalyzer(net, roi_fraction=0.3, time_window=(0, 96), dt_minutes=15.0)
    analyzer.prepare_data()

    stv = analyzer.compute_spatiotemporal_instability(
        max_buses=200, max_times=96, max_pairs=50_000, unstable_quantile=0.9
    )

    # A) Instability histogram
    plot_instability_histogram(analyzer.instability_index, q_thr=0.9, title="Instability distribution")
    plt.show()

    # B) Spatial & temporal marginal variograms
    plot_variogram_marginals(stv["stv"])
    plt.show()

    # C) Directional variograms
    dir_res = analyzer.compute_directional_variograms(
        values=analyzer.instability_index,
        angles_deg=(0, 45, 90, 135),
        tolerance_deg=22.5,
        n_lags=8,
    )
    plot_directional_variograms(dir_res, title="Directional variograms", fontsize=14, labelsize=12, ticksize=10)
    plt.show()

    # D) Polar directional ranges
    plot_polar_ranges(dir_res, title="Directional ranges", fontsize=14, labelsize=12, ticksize=10)
    plt.show()

    # E) Topology + critical buses
    plot_topology_with_critical_buses(analyzer, title="Critical buses", node_size=20, critical_size=80)
    plt.show()

    # F) Sample critical + global ellipse
    plot_sample_critical_with_global_ellipse(analyzer, title="Global ellipse on critical node")
    plt.show()

    # G) GeoPandas map (no basemap)
    plot_geopandas_map(analyzer, title="GeoPandas map (no basemap)")
    plt.show()

    # H) Local anisotropic ellipses
    plot_local_anisotropic_ellipses(analyzer, title="Local anisotropic ellipses")
    plt.show()


if __name__ == "__main__":
    main()

