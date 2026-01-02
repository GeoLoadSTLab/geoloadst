#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic walkthrough of spatial and spatio-temporal instability analysis on SimBench.
Shows ROI selection, variograms, PCA/clustering, Moran's I, and a day/night scenario.
"""

import warnings
warnings.filterwarnings("ignore")

import simbench
import matplotlib.pyplot as plt

from geoloadst import InstabilityAnalyzer
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
    plot_industrial_cluster_map,
)


def main():
    # 1) Load SimBench network
    print("Loading SimBench network...")
    
    net = simbench.get_simbench_net("1-complete_data-mixed-all-1-sw")
    print(f"Network loaded: {len(net.bus)} buses, {len(net.load)} loads")
    
    # 2) Create InstabilityAnalyzer
    print("\nCreating InstabilityAnalyzer with a spatial ROI...")
    
    analyzer = InstabilityAnalyzer(
        net,
        roi=(10.8, 11.7, 53.1, 53.6),  # x_min, x_max, y_min, y_max
        time_window=(0, 96),            # 24 hours at 15-min resolution
        dt_minutes=15.0,
    )
    
    # Prepare data
    analyzer.prepare_data()
    print(f"Selected {len(analyzer.bus_ids)} buses in ROI")
    print(f"Load time series shape: {analyzer.bus_load_df.shape}")
    
    # 3) Spatio-temporal instability analysis
    print("\nComputing spatio-temporal instability...")
    
    stv_results = analyzer.compute_spatiotemporal_instability(
        x_lags=12,
        t_lags=8,
        unstable_quantile=0.9,
    )
    
    print(f"Spatial correlation range: {stv_results['stv']['space_range']:.2f}")
    print(f"Temporal correlation range: {stv_results['stv']['time_range_hours']:.1f} hours")
    print(f"Critical nodes (top 10%): {len(stv_results['critical_bus_ids'])}")
    print(f"Instability threshold: {stv_results['threshold']:.4f}")
    
    # Plot instability histogram
    fig1 = plot_instability_histogram(
        stv_results["instability_index"],
        threshold=stv_results["threshold"],
        title="Instability Index Distribution",
    )
    plt.show()
    
    # Plot variogram marginals
    fig2 = plot_variogram_marginals(stv_results["stv"])
    plt.show()
    
    # Plot network with critical nodes
    fig3 = plot_network_topology(
        net,
        analyzer.bus_ids,
        analyzer.coords,
        critical_mask=stv_results["critical_mask"],
        title="Network Topology with Critical Nodes",
    )
    plt.show()
    
    # 4) Multi-dimensional instability analysis
    print("\nComputing multi-dimensional instability features...")
    
    multi_results = analyzer.compute_multidim_instability(n_clusters=3)
    
    print(f"Features: {multi_results['feature_names']}")
    print(f"PCA explained variance: {multi_results['pca_results']['cumulative_variance']:.1%}")
    print(f"Cluster sizes: {[sum(multi_results['pca_results']['cluster_labels'] == k) for k in range(3)]}")
    
    # Plot PCA with clusters
    fig4 = plot_pca_clusters(
        multi_results["pca_results"]["pca_components"],
        multi_results["pca_results"]["cluster_labels"],
        title="Multi-dimensional Instability - PCA Space",
    )
    plt.show()
    
    # Plot cluster map
    fig5 = plot_cluster_map(
        analyzer.bus_ids,
        analyzer.coords,
        multi_results["pca_results"]["cluster_labels"],
        net=net,
        title="Instability Clusters on Network",
    )
    plt.show()
    
    # 5) Moran's I spatial autocorrelation
    print("\nComputing Moran's I spatial autocorrelation...")
    
    moran_results = analyzer.compute_moran_analysis(k_neighbors=8)
    
    mi_mean = moran_results["moran_mean_load"]
    mi_inst = moran_results["moran_instability"]
    
    print(f"Global Moran's I (mean load): {mi_mean.I:.4f} (p={mi_mean.p_norm:.4f})")
    print(f"Global Moran's I (instability): {mi_inst.I:.4f} (p={mi_inst.p_norm:.4f})")
    
    # LISA cluster counts
    clusters = moran_results["clusters_mean_load"]
    cluster_labels = moran_results["cluster_labels_map"]
    print("LISA clusters (mean load):")
    for code, label in cluster_labels.items():
        count = (clusters == code).sum()
        print(f"  {label}: {count}")
    
    # Plot LISA clusters
    fig6 = plot_lisa_clusters_map(
        analyzer.bus_ids,
        analyzer.coords,
        moran_results["clusters_mean_load"],
        net=net,
        title="LISA Clusters (Mean Load)",
    )
    plt.show()
    
    # 6) Industrial day/night scenario
    print("\nRunning industrial day/night scenario...")
    
    scenario_results = analyzer.run_industrial_daynight_scenario(
        n_clusters=3,
        day_start_h=8.0,
        day_end_h=20.0,
        day_factor=3.0,
        night_factor=0.3,
    )
    
    scen_data = scenario_results["scenario_data"]
    moran_comp = scenario_results["moran_comparison"]
    
    print(f"Industrial nodes: {scen_data['industrial_mask'].sum()}")
    print(f"Base Moran I range: [{moran_comp['moran_base'].min():.4f}, {moran_comp['moran_base'].max():.4f}]")
    print(f"Scenario Moran I range: [{moran_comp['moran_scenario'].min():.4f}, {moran_comp['moran_scenario'].max():.4f}]")
    
    # Plot industrial cluster
    fig7 = plot_industrial_cluster_map(
        analyzer.bus_ids,
        analyzer.coords,
        scen_data["industrial_mask"],
        title="Industrial Cluster Identification",
    )
    plt.show()
    
    # Plot Moran time series comparison
    fig8 = plot_moran_timeseries(
        scen_data["time_hours"],
        moran_comp["moran_base"],
        moran_scenario=moran_comp["moran_scenario"],
        day_start_h=8.0,
        day_end_h=20.0,
        title="Moran's I: Base vs Industrial Scenario",
    )
    plt.show()
    
    # 7) Summary
    print("\nSummary DataFrame")
    
    summary_df = analyzer.get_summary()
    print(summary_df.head(10).to_string())
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

