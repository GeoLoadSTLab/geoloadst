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
        plot_directional_variograms,
        plot_directional_ranges_polar,
    )
    from geoloadst.viz.maps import (
        plot_network_topology,
        plot_topology_with_critical_buses,
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
    assert plot_directional_variograms and plot_directional_ranges_polar
    assert plot_network_topology and plot_topology_with_critical_buses
    assert plot_lisa_clusters_map and plot_cluster_map


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
    n_act = len(analyzer.instability_index)
    assert W.n == n_act
    assert n_act == len(getattr(analyzer, "_active_bus_ids"))
    assert n_act == len(getattr(analyzer, "_active_coords"))


def test_public_imports_viz_maps():
    from geoloadst.viz.maps import plot_lisa_clusters_map, plot_instability_overlay
    assert plot_lisa_clusters_map
    assert plot_instability_overlay


def test_public_imports_io():
    from geoloadst.io.simbench_adapter import load_simbench_network
    assert load_simbench_network


def test_directional_variograms_with_values():
    """Test compute_directional_variograms with explicit values argument."""
    import numpy as np
    import pandas as pd
    import json
    from types import SimpleNamespace
    from geoloadst import InstabilityAnalyzer

    # Build a minimal fake net
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.0, 1.0]])
    bus_geo = pd.Series(
        [json.dumps({"coordinates": [x, y]}) for x, y in coords],
        index=np.arange(len(coords)),
        name="geo",
    )
    bus_df = pd.DataFrame({"geo": bus_geo})

    load_bus = np.array([0, 1, 2, 3, 4])
    profiles = pd.DataFrame(
        {f"p{i}_pload": np.linspace(0.8, 1.2, 30) for i in range(5)}
    )
    load_df = pd.DataFrame(
        {"bus": load_bus, "profile": [f"p{i}_pload" for i in range(5)], "p_mw": 1.0},
        index=np.arange(5),
    )
    net = SimpleNamespace(bus=bus_df, load=load_df, profiles={"load": profiles})

    analyzer = InstabilityAnalyzer(net, time_window=(0, 24), dt_minutes=15.0)
    analyzer.prepare_data()
    analyzer.compute_spatiotemporal_instability(max_buses=5, max_times=24, max_pairs=100)

    # Test with instability_index (default)
    dir_results = analyzer.compute_directional_variograms()
    assert "variograms" in dir_results
    assert "ranges" in dir_results

    # Test with explicit values
    custom_values = np.random.rand(len(analyzer.bus_ids))
    dir_results2 = analyzer.compute_directional_variograms(values=custom_values)
    assert "variograms" in dir_results2
    assert "ranges" in dir_results2


def test_plot_directional_ranges_polar_smoke():
    """Test plot_directional_ranges_polar with mock data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from geoloadst.viz.plots import plot_directional_ranges_polar

    ranges = {0: 1.5, 45: 2.0, 90: 1.2, 135: 1.8}
    fig = plot_directional_ranges_polar(ranges, title="Test Polar")
    assert fig is not None
    plt.close(fig)


def test_plot_topology_with_critical_buses_smoke():
    """Test plot_topology_with_critical_buses wrapper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from geoloadst.viz.maps import plot_topology_with_critical_buses

    bus_ids = np.array([1, 2, 3, 4, 5])
    coords = np.array([[0, 0], [1, 0], [0.5, 0.5], [1, 1], [0, 1]], dtype=float)
    critical_mask = np.array([False, True, False, True, False])

    fig, ax = plot_topology_with_critical_buses(
        net=None, bus_ids=bus_ids, coords=coords, critical_mask=critical_mask
    )
    assert fig is not None
    plt.close(fig)


def test_compute_directional_variograms_no_instability_raises():
    """Test that compute_directional_variograms raises when no values and no instability."""
    import numpy as np
    import pandas as pd
    import json
    import pytest
    from types import SimpleNamespace
    from geoloadst import InstabilityAnalyzer

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    bus_geo = pd.Series(
        [json.dumps({"coordinates": [x, y]}) for x, y in coords],
        index=np.arange(len(coords)),
        name="geo",
    )
    bus_df = pd.DataFrame({"geo": bus_geo})
    profiles = pd.DataFrame({f"p{i}_pload": np.ones(10) for i in range(3)})
    load_df = pd.DataFrame(
        {"bus": np.arange(3), "profile": [f"p{i}_pload" for i in range(3)], "p_mw": 1.0},
        index=np.arange(3),
    )
    net = SimpleNamespace(bus=bus_df, load=load_df, profiles={"load": profiles})

    analyzer = InstabilityAnalyzer(net, time_window=(0, 10))
    analyzer.prepare_data()
    # Do NOT call compute_spatiotemporal_instability

    with pytest.raises(ValueError, match="No values provided"):
        analyzer.compute_directional_variograms()

