import matplotlib.pyplot as plt
import simbench

from geoloadst import InstabilityAnalyzer
from geoloadst.viz.maps import plot_lisa_clusters_map, plot_network_topology


def main():
    net = simbench.get_simbench_net("1-complete_data-mixed-all-1-sw")

    analyzer = InstabilityAnalyzer(
        net,
        roi=(10.8, 11.7, 53.1, 53.6),
        time_window=(0, 48),  # half day at 15-min resolution
        dt_minutes=15.0,
    )

    analyzer.prepare_data()
    analyzer.compute_spatiotemporal_instability(
        max_buses=200,
        max_times=48,
        max_pairs=50_000,
    )
    moran_results = analyzer.compute_moran_analysis(k_neighbors=8, permutations=99)

    bus_ids = getattr(analyzer, "_active_bus_ids", analyzer.bus_ids)
    coords = getattr(analyzer, "_active_coords", analyzer.coords)

    # Mean-load LISA
    fig1, ax1 = plot_lisa_clusters_map(
        bus_ids,
        coords,
        moran_results["clusters_mean_load"],
        net=net,
        title="LISA clusters (mean load, small ROI)",
        show_lines=True,
    )
    plt.show()

    # Instability LISA (if available)
    if "clusters_instability" in moran_results:
        fig2, ax2 = plot_lisa_clusters_map(
            bus_ids,
            coords,
            moran_results["clusters_instability"],
            net=net,
            title="LISA clusters (instability, small ROI)",
            show_lines=True,
        )
        plt.show()
    else:
        # Fallback: just plot topology
        fig2, ax2 = plot_network_topology(
            net,
            bus_ids,
            coords,
            title="Topology (instability clusters unavailable)",
            show_lines=True,
        )
        plt.show()


if __name__ == "__main__":
    main()

