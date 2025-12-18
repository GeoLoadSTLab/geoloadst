import matplotlib.pyplot as plt
import simbench

from geoloadst import InstabilityAnalyzer
from geoloadst.viz.maps import plot_instability_overlay


def main():
    net = simbench.get_simbench_net("1-complete_data-mixed-all-1-sw")

    analyzer = InstabilityAnalyzer(
        net,
        roi=(10.8, 11.7, 53.1, 53.6),
        time_window=(0, 48),
        dt_minutes=15.0,
    )

    analyzer.prepare_data()
    stv_results = analyzer.compute_spatiotemporal_instability(
        max_buses=200,
        max_times=48,
        max_pairs=50_000,
    )

    bus_ids = stv_results["bus_ids_used"]
    coords = stv_results["coords_used"]
    instability = stv_results["instability_index"]
    critical_mask = stv_results["critical_mask"]

    fig, ax = plot_instability_overlay(
        bus_ids,
        coords,
        instability,
        critical_mask,
        net=net,
        title="Critical buses (small ROI)",
    )
    plt.show()


if __name__ == "__main__":
    main()

