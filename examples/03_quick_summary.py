import simbench

from geoloadst import InstabilityAnalyzer


def main():
    net = simbench.get_simbench_net("1-complete_data-mixed-all-1-sw")

    analyzer = InstabilityAnalyzer(
        net,
        roi=(10.8, 11.7, 53.1, 53.6),
        time_window=(0, 24),  # first 6 hours
        dt_minutes=15.0,
    )

    analyzer.prepare_data()
    stv_results = analyzer.compute_spatiotemporal_instability(
        max_buses=120,
        max_times=24,
        max_pairs=30_000,
    )

    instability = stv_results["instability_index"]
    bus_ids = stv_results["bus_ids_used"]

    top_idx = instability.argsort()[::-1][:10]
    print("Top 10 instability (bus_id, value):")
    for i in top_idx:
        print(f"  {bus_ids[i]} -> {instability[i]:.4f}")

    moran_results = analyzer.compute_moran_analysis(k_neighbors=6, permutations=0)
    print("Global Moran's I (mean load):", moran_results["moran_mean_load"].I)


if __name__ == "__main__":
    main()

