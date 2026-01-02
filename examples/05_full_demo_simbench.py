"""Full demo: instability, ST marginals, directional, polar, topology, ellipses (no basemap)."""

import matplotlib.pyplot as plt
import matplotlib as mpl
import simbench

from geoloadst import InstabilityAnalyzer
from geoloadst.viz.plots import (
    plot_instability_histogram,
    plot_st_marginals,
    plot_directional_variograms,
    plot_directional_ranges_polar,
    plot_polar_ranges,
)
from geoloadst.viz.maps import (
    plot_topology_with_critical_buses,
    plot_sample_critical_with_global_ellipse,
    plot_geopandas_map,
    plot_local_anisotropic_ellipses,
)

mpl.rcParams.update({"font.size": 12})


def main() -> None:
    net = simbench.get_simbench_net("1-complete_data-mixed-all-1-sw")
    analyzer = InstabilityAnalyzer(net, roi_fraction=0.3, time_window=(0, 96), dt_minutes=15.0)
    analyzer.prepare_data()

    stv = analyzer.compute_spatiotemporal_instability(
        max_buses=200, max_times=96, max_pairs=50_000, unstable_quantile=0.9
    )

    # A) Instability histogram
    plot_instability_histogram(
        analyzer.instability_index,
        q_thr=0.9,
        title="Instability distribution",
        fontsize=16,
        labelsize=14,
        ticksize=12,
        legend_fontsize=12,
    )
    plt.show()

    # B) Spatial & temporal marginal variograms
    plot_st_marginals(
        stv["stv"]["x_marginal"],
        stv["stv"]["t_marginal"],
        stv["stv"]["space_range"],
        stv["stv"]["time_range_steps"],
        stv["stv"]["time_range_hours"],
        title="ST marginals",
    )
    plt.show()

    # C) Directional variograms
    dir_res = analyzer.compute_directional_variograms(
        values=analyzer.instability_index,
        azimuths=(0, 45, 90, 135),
        tolerance_deg=22.5,
        n_lags=8,
    )
    plot_directional_variograms(
        dir_res,
        title="Directional variograms",
        fontsize=16,
        labelsize=14,
        ticksize=12,
        legend_fontsize=12,
    )
    plt.show()

    # D) Polar directional ranges (alias)
    plot_polar_ranges(dir_res, title="Directional ranges", fontsize=16, labelsize=14, ticksize=12)
    plt.show()

    # E) Topology + critical buses
    plot_topology_with_critical_buses(analyzer, title="Critical buses", node_size=20, critical_size=80)
    plt.show()

    # F) Sample critical + global ellipse
    plot_sample_critical_with_global_ellipse(analyzer, title="Global ellipse on critical node")
    plt.show()

    # G) GeoPandas map (no basemap)
    plot_geopandas_map({"lines_gdf": getattr(analyzer, "lines_gdf", None),
                        "bus_gdf": getattr(analyzer, "bus_gdf", None),
                        "critical_gdf": getattr(analyzer, "critical_gdf", None)},
                       title="GeoPandas map (no basemap)")
    plt.show()

    # H) Local anisotropic ellipses
    analyzer.compute_local_anisotropy()
    plot_local_anisotropic_ellipses(analyzer, title="Local anisotropic ellipses")
    plt.show()


if __name__ == "__main__":
    main()

