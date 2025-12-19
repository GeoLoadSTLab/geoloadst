import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from geoloadst import InstabilityAnalyzer
from geoloadst.viz.maps import plot_lisa_clusters_map

plt.switch_backend("Agg")


def _make_fake_net(n_buses: int = 200, n_times: int = 60) -> SimpleNamespace:
    coords = np.stack(
        [np.linspace(0, 10, n_buses), np.linspace(0, 5, n_buses)], axis=1
    )
    bus_geo = pd.Series(
        [json.dumps({"coordinates": [x, y]}) for x, y in coords],
        index=np.arange(n_buses),
        name="geo",
    )
    bus_df = pd.DataFrame({"geo": bus_geo})

    profile_vals = np.linspace(0.8, 1.2, n_times)
    profiles = pd.DataFrame({"base_pload": profile_vals})

    load_df = pd.DataFrame(
        {
            "bus": np.arange(n_buses),
            "profile": "base",
            "p_mw": 1.0,
        },
        index=np.arange(n_buses),
    )

    line_df = pd.DataFrame(
        {
            "from_bus": np.arange(0, n_buses - 1),
            "to_bus": np.arange(1, n_buses),
        },
        index=np.arange(n_buses - 1),
    )

    return SimpleNamespace(bus=bus_df, load=load_df, profiles={"load": profiles}, line=line_df)


def test_active_subset_smoke():
    net = _make_fake_net()
    analyzer = InstabilityAnalyzer(net, time_window=(0, 60), dt_minutes=15.0)

    analyzer.prepare_data()
    analyzer.compute_spatiotemporal_instability(
        max_buses=50, max_times=48, max_pairs=50_000, random_state=0
    )
    moran = analyzer.compute_moran_analysis(permutations=0)

    fig, _ = plot_lisa_clusters_map(
        analyzer.bus_ids, analyzer.coords, moran["clusters_mean_load"], net=net
    )
    plt.close(fig)

    assert len(analyzer.bus_ids) == len(analyzer.coords) == 50
    assert len(moran["clusters_mean_load"]) == len(analyzer.coords)
    assert moran["weights"].n == 50


def test_prepare_data_empty_roi_raises():
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    bus_geo = pd.Series(
        [json.dumps({"coordinates": [x, y]}) for x, y in coords],
        index=np.arange(len(coords)),
        name="geo",
    )
    bus_df = pd.DataFrame({"geo": bus_geo})
    load_df = pd.DataFrame(
        {"bus": np.arange(len(coords)), "profile": "p0_pload", "p_mw": 1.0},
        index=np.arange(len(coords)),
    )
    profiles = pd.DataFrame({"p0_pload": np.ones(10)})
    net = SimpleNamespace(bus=bus_df, load=load_df, profiles={"load": profiles})

    analyzer = InstabilityAnalyzer(net, roi=(10, 11, 10, 11))
    with pytest.raises(ValueError) as excinfo:
        analyzer.prepare_data()
    assert "No buses found after ROI selection" in str(excinfo.value)


