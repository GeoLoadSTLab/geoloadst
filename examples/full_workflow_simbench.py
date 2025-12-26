"""Full SimBench workflow using the GeoLoadST high-level API."""

import matplotlib.pyplot as plt
import simbench

from geoloadst import InstabilityAnalyzer
from geoloadst.viz.plots import (
    plot_instability_histogram,
    plot_st_marginals,
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
    analyzer = InstabilityAnalyzer(net, dt_minutes=15.0, time_window=(0, 96))

    results = analyzer.run_full_workflow(
        roi_fraction=0.3,
        max_buses=200,
        max_times=96,
        max_pairs=50_000,
        unstable_quantile=0.9,
        make_geopandas_maps=True,
    )

    # 1) Instability histogram
    plot_instability_histogram(results["instability_index"], q_thr=0.9, title="Instability distribution")
    plt.show()

    # 2) ST marginals
    stv = results["space_time_variogram_results"]
    plot_st_marginals(
        stv["Vx"],
        stv["Vt"],
        stv["space_range"],
        stv["time_range_steps"],
        stv["time_range_hours"],
    )
    plt.show()

    # 3) Directional variograms
    plot_directional_variograms(results["directional_results"], title="Directional variograms", fontsize=14)
    plt.show()

    # 4) Polar ranges
    plot_polar_ranges(results["directional_results"], title="Directional ranges", fontsize=14)
    plt.show()

    # 5) Topology with critical buses
    plot_topology_with_critical_buses(analyzer, title="Critical buses", node_size=20, critical_size=80)
    plt.show()

    # 6) Sample critical with global ellipse
    plot_sample_critical_with_global_ellipse(analyzer, title="Global ellipse on critical node")
    plt.show()

    # 7) GeoPandas map (if available)
    plot_geopandas_map(analyzer, title="GeoPandas map (no basemap)")
    plt.show()

    # 8) Local anisotropic ellipses
    plot_local_anisotropic_ellipses(analyzer, title="Local anisotropic ellipses")
    plt.show()


if __name__ == "__main__":
    main()

