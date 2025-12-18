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


def test_spatiotemporal_small_smoke():
    import numpy as np
    from geoloadst.core.spatiotemporal import compute_stv

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    values = np.random.random((3, 12))

    res = compute_stv(
        coords,
        values,
        x_lags=2,
        t_lags=2,
        max_pairs=10,
        random_state=0,
    )

    assert "stv" in res
    assert res["space_range"] is not None


def test_moran_after_subsampled_stv():
    import numpy as np
    import pandas as pd
    import json
    from types import SimpleNamespace
    from geoloadst import InstabilityAnalyzer

    # Build a minimal fake net with required attributes
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.0, 1.0]])
    bus_geo = pd.Series(
        [json.dumps({"coordinates": [x, y]}) for x, y in coords],
        index=np.arange(len(coords)),
        name="geo",
    )
    bus_df = pd.DataFrame({"geo": bus_geo})

    load_bus = np.array([0, 1, 2, 3, 4])
    profiles = pd.DataFrame(
        {
            "p1_pload": np.linspace(0.8, 1.2, 30),
        }
    )
    profiles = pd.concat([profiles]*5, axis=1)
    profiles.columns = [f"p{i}_pload" for i in range(5)]

    load_df = pd.DataFrame(
        {
            "bus": load_bus,
            "profile": [f"p{i}_pload" for i in range(5)],
            "p_mw": 1.0,
        },
        index=np.arange(5),
    )

    net = SimpleNamespace(bus=bus_df, load=load_df, profiles={"load": profiles})

    analyzer = InstabilityAnalyzer(net, time_window=(0, 24), dt_minutes=15.0)
    analyzer.prepare_data()
    analyzer.compute_spatiotemporal_instability(
        max_buses=3,
        max_times=24,
        max_pairs=5_000,
    )
    moran_results = analyzer.compute_moran_analysis(k_neighbors=2, permutations=0)

    W = moran_results["weights"]
    assert W.n == len(analyzer.instability_index)

