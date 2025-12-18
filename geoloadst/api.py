"""High-level wrapper around the instability pipeline: load data, analyze, and summarize."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandapower import pandapowerNet


class InstabilityAnalyzer:
    """End-to-end analyzer for spatial and spatio-temporal load instability."""

    def __init__(
        self,
        net: "pandapowerNet",
        roi: tuple[float, float, float, float] | None = None,
        roi_fraction: float | None = None,
        time_window: tuple[int, int] | None = None,
        dt_minutes: float = 15.0,
        config: dict | None = None,
    ) -> None:
        self.net = net
        self.roi = roi
        self.roi_fraction = roi_fraction
        self.time_window = time_window or (0, 96)
        self.dt_minutes = dt_minutes
        self.config = config or {}

        # Data attributes (populated by prepare_data)
        self.bus_ids: np.ndarray | None = None
        self.coords: np.ndarray | None = None
        self.bus_to_coord: dict[int, np.ndarray] | None = None
        self.bus_load_df: pd.DataFrame | None = None

        # Analysis results (populated by compute_* methods)
        self.values_std: np.ndarray | None = None
        self.instability_index: np.ndarray | None = None
        self._stv_results: dict | None = None
        self._multidim_results: dict | None = None
        self._moran_results: dict | None = None
        self._scenario_results: dict | None = None
        self._topology_results: dict | None = None
        self._spatial_weights: Any = None

    def prepare_data(self) -> "InstabilityAnalyzer":
        """Pull bus coords, apply ROI, and assemble per-bus load time series."""
        from geoloadst.io.simbench_adapter import (
            extract_bus_coordinates,
            select_roi_buses,
            build_bus_load_timeseries,
        )

        # Extract all bus coordinates
        all_bus_ids, all_coords, all_bus_to_coord = extract_bus_coordinates(self.net)

        # Select ROI
        roi_bus_ids, roi_coords = select_roi_buses(
            all_bus_ids,
            all_coords,
            roi=self.roi,
            roi_fraction=self.roi_fraction,
        )

        # Build load time series
        max_times = self.time_window[1] - self.time_window[0]
        bus_load_df = build_bus_load_timeseries(
            self.net,
            bus_ids=roi_bus_ids,
            max_times=max_times,
        )

        # Filter to buses that have both coordinates and load data
        available_buses = bus_load_df.columns.values
        mask = np.isin(roi_bus_ids, available_buses)

        self.bus_ids = roi_bus_ids[mask]
        self.coords = roi_coords[mask]
        self.bus_to_coord = {int(b): self.coords[i] for i, b in enumerate(self.bus_ids)}

        # Reorder bus_load_df to match bus_ids order
        self.bus_load_df = bus_load_df[self.bus_ids]

        return self

    def compute_spatiotemporal_instability(
        self,
        x_lags: int = 12,
        t_lags: int = 8,
        unstable_quantile: float = 0.9,
        max_buses: int = 500,
        max_times: int = 96,
        max_pairs: int = 200_000,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Detrend/standardize loads, score instability, flag critical nodes, fit STV.

        Parameters
        ----------
        max_buses : int
            If more buses are present, keep the highest-mean-load subset.
        max_times : int
            Trim time steps to this count.
        max_pairs : int
            Upper bound on pairwise distances (subsamples buses to respect this).
        random_state : int
            RNG seed for any subsampling.
        """
        self._check_data_prepared()

        from geoloadst.core.preprocessing import detrend_and_standardize
        from geoloadst.core.instability_index import rms_instability, classify_critical_nodes
        from geoloadst.core.spatiotemporal import compute_stv

        # Work on a bounded subset to avoid O(N^2) blowups
        bus_load_df = self.bus_load_df
        if max_times and len(bus_load_df) > max_times:
            bus_load_df = bus_load_df.iloc[:max_times]

        if max_buses and bus_load_df.shape[1] > max_buses:
            mean_load = bus_load_df.mean(axis=0)
            keep_buses = mean_load.sort_values(ascending=False).index[:max_buses]
            bus_load_df = bus_load_df[keep_buses]
            print(f"[geoloadst] Subsampled buses for STV: {self.bus_load_df.shape[1]} -> {len(keep_buses)}")
        else:
            keep_buses = bus_load_df.columns

        # Align coords to kept buses
        bus_ids_arr = np.array(keep_buses, dtype=int)
        coords_map = {int(b): c for b, c in zip(self.bus_ids, self.coords)}
        coords_subset = np.vstack([coords_map[int(b)] for b in bus_ids_arr])

        # Detrend and standardize
        self.values_std = detrend_and_standardize(
            bus_load_df,
            coords_subset,
            dt_minutes=self.dt_minutes,
        )

        # Compute instability index
        self.instability_index = rms_instability(self.values_std)

        # Classify critical nodes
        critical_mask, critical_indices, threshold = classify_critical_nodes(
            self.instability_index, quantile=unstable_quantile
        )

        # Compute space-time variogram with pair budget
        stv_results = compute_stv(
            coords_subset,
            self.values_std,
            dt_minutes=self.dt_minutes,
            x_lags=x_lags,
            t_lags=t_lags,
            max_pairs=max_pairs,
            random_state=random_state,
        )

        self._stv_results = {
            "values_std": self.values_std,
            "instability_index": self.instability_index,
            "critical_mask": critical_mask,
            "critical_indices": critical_indices,
            "critical_bus_ids": bus_ids_arr[critical_indices],
            "threshold": threshold,
            "stv": stv_results,
            "bus_ids_used": bus_ids_arr,
            "coords_used": coords_subset,
        }

        return self._stv_results

    def compute_directional_variograms(
        self,
        azimuths: list[int] | None = None,
    ) -> dict[str, Any]:
        """Directional variograms for anisotropy; returns ranges and ellipse params."""
        self._check_instability_computed()

        from geoloadst.core.spatiotemporal import compute_directional_variograms

        space_range = self._stv_results["stv"]["space_range"]

        dir_results = compute_directional_variograms(
            self.coords,
            self.instability_index,
            azimuths=azimuths,
            maxlag=space_range * 1.2 if not np.isnan(space_range) else None,
        )

        return dir_results

    def compute_multidim_instability(
        self,
        n_clusters: int = 3,
        n_pca_components: int = 2,
    ) -> dict[str, Any]:
        """Build instability features, run PCA + clustering, and summarize clusters."""
        self._check_data_prepared()

        from geoloadst.core.multidim_instability import (
            build_instability_features,
            pca_and_cluster,
            cluster_feature_summary,
        )

        # Use detrended values if available, otherwise compute
        if self.values_std is None:
            from geoloadst.core.preprocessing import detrend_and_standardize
            self.values_std = detrend_and_standardize(
                self.bus_load_df,
                self.coords,
                dt_minutes=self.dt_minutes,
            )

        # Build features
        features, feature_names = build_instability_features(
            self.bus_load_df,
            values_detrended=self.values_std,
        )

        # PCA and clustering
        pca_results = pca_and_cluster(
            features,
            n_clusters=n_clusters,
            n_components=n_pca_components,
        )

        # Cluster summary
        cluster_summary = cluster_feature_summary(
            pca_results["features_std"],
            feature_names,
            pca_results["cluster_labels"],
        )

        self._multidim_results = {
            "features": features,
            "feature_names": feature_names,
            "pca_results": pca_results,
            "cluster_summary": cluster_summary,
        }

        return self._multidim_results

    def compute_moran_analysis(
        self,
        k_neighbors: int = 8,
        permutations: int = 999,
    ) -> dict[str, Any]:
        """Build KNN weights, run global Moran on mean/instability, and compute LISA."""
        self._check_data_prepared()

        from geoloadst.core.moran import (
            build_knn_weights,
            moran_analysis_summary,
            get_cluster_labels,
        )

        # Ensure instability is computed
        if self.instability_index is None:
            from geoloadst.core.instability_index import rms_instability
            from geoloadst.core.preprocessing import detrend_and_standardize

            self.values_std = detrend_and_standardize(
                self.bus_load_df,
                self.coords,
                dt_minutes=self.dt_minutes,
            )
            self.instability_index = rms_instability(self.values_std)

        # Build spatial weights
        W = build_knn_weights(self.coords, k=k_neighbors)
        self._spatial_weights = W

        # Mean load
        mean_load = self.bus_load_df.mean(axis=0).values

        # Run Moran analysis
        moran_summary = moran_analysis_summary(
            mean_load, self.instability_index, W, permutations=permutations
        )

        self._moran_results = {
            "weights": W,
            "mean_load": mean_load,
            **moran_summary,
            "cluster_labels_map": get_cluster_labels(),
        }

        return self._moran_results

    def run_industrial_daynight_scenario(
        self,
        n_clusters: int = 3,
        day_start_h: float = 8.0,
        day_end_h: float = 20.0,
        day_factor: float = 3.0,
        night_factor: float = 0.3,
    ) -> dict[str, Any]:
        """Apply the industrial day/night scenario and compare Moran/LISA vs base."""
        self._check_data_prepared()

        from geoloadst.scenarios.industrial_daynight import (
            apply_industrial_daynight_pattern,
            compare_scenario_moran,
            compute_scenario_lisa_comparison,
        )

        # Ensure spatial weights are built
        if self._spatial_weights is None:
            from geoloadst.core.moran import build_knn_weights
            self._spatial_weights = build_knn_weights(self.coords, k=8)

        # Apply scenario
        scenario_data = apply_industrial_daynight_pattern(
            self.bus_load_df,
            self.coords,
            dt_minutes=self.dt_minutes,
            n_clusters=n_clusters,
            day_start_h=day_start_h,
            day_end_h=day_end_h,
            day_factor=day_factor,
            night_factor=night_factor,
        )

        # Compare Moran time series
        moran_comparison = compare_scenario_moran(
            scenario_data["bus_load_base"],
            scenario_data["bus_load_scenario"],
            self._spatial_weights,
        )

        # LISA comparison at midday
        T = self.bus_load_df.shape[0]
        t_mid = int(((day_start_h + day_end_h) / 2) * 60.0 / self.dt_minutes)
        t_mid = min(max(t_mid, 0), T - 1)

        lisa_comparison = compute_scenario_lisa_comparison(
            scenario_data["bus_load_base"],
            scenario_data["bus_load_scenario"],
            self._spatial_weights,
            time_idx=t_mid,
        )

        self._scenario_results = {
            "scenario_data": scenario_data,
            "moran_comparison": moran_comparison,
            "lisa_comparison": lisa_comparison,
            "midday_time_idx": t_mid,
        }

        return self._scenario_results

    def compute_topology_analysis(self) -> dict[str, Any]:
        """Compute centralities for the subnet graph and correlate with instability."""
        self._check_data_prepared()

        from geoloadst.core.topology import (
            build_network_graph,
            compute_topological_metrics,
            correlate_instability_topology,
        )

        # Ensure instability is computed
        if self.instability_index is None:
            from geoloadst.core.instability_index import rms_instability
            from geoloadst.core.preprocessing import detrend_and_standardize

            self.values_std = detrend_and_standardize(
                self.bus_load_df,
                self.coords,
                dt_minutes=self.dt_minutes,
            )
            self.instability_index = rms_instability(self.values_std)

        # Build graph
        G = build_network_graph(self.net, bus_ids=self.bus_ids)

        # Compute metrics
        metrics = compute_topological_metrics(G, bus_ids=self.bus_ids)

        # Correlate with instability
        correlations = correlate_instability_topology(
            self.instability_index, metrics
        )

        self._topology_results = {
            "graph": G,
            "metrics": metrics,
            "correlations": correlations,
        }

        return self._topology_results

    def get_summary(self) -> pd.DataFrame:
        """Return a bus-level summary DataFrame with whichever results are available."""
        self._check_data_prepared()

        data = {
            "bus_id": self.bus_ids,
            "x": self.coords[:, 0],
            "y": self.coords[:, 1],
            "mean_load": self.bus_load_df.mean(axis=0).values,
        }

        if self.instability_index is not None:
            data["instability"] = self.instability_index

        if self._stv_results is not None:
            data["is_critical"] = self._stv_results["critical_mask"]

        if self._multidim_results is not None:
            data["cluster"] = self._multidim_results["pca_results"]["cluster_labels"]

        if self._moran_results is not None:
            data["lisa_cluster"] = self._moran_results["clusters_mean_load"]

        if self._scenario_results is not None:
            data["is_industrial"] = self._scenario_results["scenario_data"]["industrial_mask"]

        if self._topology_results is not None:
            data["degree"] = self._topology_results["metrics"]["degree"]
            data["betweenness"] = self._topology_results["metrics"]["betweenness"]

        return pd.DataFrame(data)

    def _check_data_prepared(self) -> None:
        """Check that prepare_data has been called."""
        if self.bus_load_df is None:
            raise RuntimeError(
                "Data not prepared. Call prepare_data() first."
            )

    def _check_instability_computed(self) -> None:
        """Check that instability has been computed."""
        if self._stv_results is None:
            raise RuntimeError(
                "Instability not computed. Call compute_spatiotemporal_instability() first."
            )

