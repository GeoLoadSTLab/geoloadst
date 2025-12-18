def test_imports():
    from geoloadst import InstabilityAnalyzer
    from geoloadst.io import (
        load_simbench_network,
        extract_bus_coordinates,
        build_bus_load_timeseries,
    )
    from geoloadst.core import (
        detrend_and_standardize,
        compute_stv,
        compute_directional_variograms,
        build_knn_weights,
        global_moran,
        local_moran_clusters,
    )
    from geoloadst.scenarios import apply_industrial_daynight_pattern
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

    # Smoke: ensure names are bound
    assert InstabilityAnalyzer
    assert load_simbench_network and extract_bus_coordinates and build_bus_load_timeseries
    assert detrend_and_standardize and compute_stv and compute_directional_variograms
    assert build_knn_weights and global_moran and local_moran_clusters
    assert apply_industrial_daynight_pattern
    assert plot_instability_histogram and plot_variogram_marginals and plot_pca_clusters
    assert plot_moran_timeseries
    assert plot_network_topology and plot_lisa_clusters_map and plot_cluster_map

