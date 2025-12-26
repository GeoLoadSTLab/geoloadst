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
        self._active_bus_idx: np.ndarray | None = None
        self._active_bus_ids: np.ndarray | None = None
        self._active_coords: np.ndarray | None = None
        self._active_bus_load_df: pd.DataFrame | None = None
        self._active_values_std: np.ndarray | None = None
        self._bus_ids_full: np.ndarray | None = None
        self._coords_full: np.ndarray | None = None
        self._bus_load_df_full: pd.DataFrame | None = None
        self._subsample_idx: np.ndarray | None = None

    def prepare_data(self) -> "InstabilityAnalyzer":
        """Pull bus coords, apply ROI, and assemble per-bus load time series."""
        from geoloadst.io.simbench_adapter import (
            extract_bus_coordinates,
            select_roi_buses,
            build_bus_load_timeseries,
        )

        # Extract all bus coordinates
        all_bus_ids, all_coords, all_bus_to_coord = extract_bus_coordinates(self.net)
        total_with_coords = len(all_bus_ids)

        # Select ROI
        roi_bus_ids, roi_coords = select_roi_buses(
            all_bus_ids,
            all_coords,
            roi=self.roi,
            roi_fraction=self.roi_fraction,
        )

        if roi_bus_ids.size == 0:
            roi_desc = (
                f"roi={self.roi}" if self.roi is not None else f"roi_fraction={self.roi_fraction}"
            )
            raise ValueError(
                "No buses found after ROI selection. "
                f"{roi_desc}. "
                f"Buses with valid coordinates: {total_with_coords}. "
                "Enlarge the ROI or verify bus coordinate fields (bus_geodata or bus.geo)."
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

        # Persist the full (pre-subsampling) data for reference
        self._bus_ids_full = self.bus_ids.copy()
        self._coords_full = self.coords.copy()
        self._bus_load_df_full = self.bus_load_df.copy()

        self._validate_prepared_state()

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

        # Align coords to kept buses and record indices within the full set
        bus_ids_arr = np.array(keep_buses, dtype=int)
        coords_source = self._coords_full if self._coords_full is not None else self.coords
        bus_ids_source = self._bus_ids_full if self._bus_ids_full is not None else self.bus_ids
        bus_id_to_idx = {int(b): i for i, b in enumerate(bus_ids_source)}

        try:
            active_idx = np.array([bus_id_to_idx[int(b)] for b in bus_ids_arr], dtype=int)
        except KeyError as exc:
            missing = int(exc.args[0])
            raise ValueError(
                f"Kept bus id {missing} not found in the prepared bus list. "
                "Ensure prepare_data() was called before compute_spatiotemporal_instability()."
            ) from exc

        coords_subset = coords_source[active_idx]

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

        # Persist the active subset for downstream steps
        self._set_active_subset(
            bus_ids_subset=bus_ids_arr,
            coords_subset=coords_subset,
            bus_load_df_subset=bus_load_df,
            active_idx=active_idx,
            values_std=self.values_std,
        )

        return self._stv_results

    def compute_directional_variograms(
        self,
        values: np.ndarray | None = None,
        angles_deg: tuple[int, ...] | list[int] | None = (0, 45, 90, 135),
        tolerance_deg: float = 22.5,
        n_lags: int = 8,
        maxlag: float | None = None,
        model: str = "spherical",
        azimuths: list[int] | None = None,
    ) -> dict[str, Any]:
        """Directional variograms for anisotropy; returns ranges and ellipse params.

        Parameters
        ----------
        values : np.ndarray, optional
            Values to compute variograms for. If None, uses self.instability_index.
            Must have same length as active coords if provided.
        angles_deg : tuple or list, optional
            Azimuth angles in degrees. Default is (0, 45, 90, 135).
        tolerance_deg : float, optional
            Angular tolerance in degrees for each direction. Default is 22.5.
        n_lags : int, optional
            Number of lag bins for the variogram. Default is 8.
        maxlag : float, optional
            Maximum lag distance. If None, auto-computed from STV or median distance.
        model : str, optional
            Variogram model type. Default is "spherical".
        azimuths : list[int], optional
            Deprecated alias for angles_deg. If provided, overrides angles_deg.

        Returns
        -------
        dict
            {"variograms": dict[azimuth -> DirectionalVariogram],
             "ranges": dict[azimuth -> float],
             "major_axis_azimuth": int, "minor_axis_azimuth": int,
             "a": float (semi-major), "b": float (semi-minor)}

        Raises
        ------
        ValueError
            If values is None and instability_index has not been computed.
        """
        from geoloadst.core.spatiotemporal import compute_directional_variograms

        coords_active = getattr(self, "_active_coords", self.coords)

        # Handle values
        if values is None:
            if self.instability_index is None:
                raise ValueError(
                    "No values provided and instability_index is None. "
                    "Either pass values explicitly or call compute_spatiotemporal_instability() first."
                )
            values = self.instability_index

        # Handle angles: azimuths takes precedence for backward compatibility
        if azimuths is not None:
            angles = list(azimuths)
        elif angles_deg is not None:
            angles = list(angles_deg)
        else:
            angles = [0, 45, 90, 135]

        # Determine maxlag from STV results if not provided
        if maxlag is None and self._stv_results is not None:
            space_range = self._stv_results["stv"]["space_range"]
            if not np.isnan(space_range):
                maxlag = space_range * 1.2

        dir_results = compute_directional_variograms(
            coords_active,
            values,
            azimuths=angles,
            tolerance=tolerance_deg,
            n_lags=n_lags,
            maxlag=maxlag,
            model=model,
        )

        return {
            "variograms": dir_results.get("variograms", {}),
            "ranges": dir_results.get("ranges", {}),
            "major_axis_azimuth": dir_results.get("major_azimuth"),
            "minor_axis_azimuth": dir_results.get("minor_azimuth"),
            "a": dir_results.get("semi_major", np.nan),
            "b": dir_results.get("semi_minor", np.nan),
            # Keep old keys for backward compatibility
            "major_azimuth": dir_results.get("major_azimuth"),
            "minor_azimuth": dir_results.get("minor_azimuth"),
            "semi_major": dir_results.get("semi_major", np.nan),
            "semi_minor": dir_results.get("semi_minor", np.nan),
            "angle": dir_results.get("angle", 0.0),
        }

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

        bus_load_df_active = getattr(self, "_active_bus_load_df", self.bus_load_df)
        coords_active = getattr(self, "_active_coords", self.coords)

        # Use detrended values if available, otherwise compute on active subset
        if self.values_std is None or getattr(self, "_active_values_std", None) is None:
            from geoloadst.core.preprocessing import detrend_and_standardize
            self.values_std = detrend_and_standardize(
                bus_load_df_active,
                coords_active,
                dt_minutes=self.dt_minutes,
            )
            self._active_values_std = self.values_std

        # Build features
        features, feature_names = build_instability_features(
            bus_load_df_active,
            values_detrended=self._active_values_std,
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

        # Use active subset if STV subsampled
        bus_ids_active = self.bus_ids_active
        coords_active = self.coords_active
        bus_load_df_active = self.bus_load_df_active

        n_active = len(bus_ids_active)
        if coords_active is None or bus_load_df_active is None:
            raise RuntimeError("Active subset not available. Call prepare_data() first.")
        if coords_active.shape[0] != n_active:
            raise ValueError(
                f"coords length ({coords_active.shape[0]}) does not match bus_ids ({n_active}). "
                "Downstream analyses require the same active subset."
            )
        if bus_load_df_active.shape[1] != n_active:
            raise ValueError(
                f"bus_load_df columns ({bus_load_df_active.shape[1]}) do not match bus_ids ({n_active}). "
                "Rerun compute_spatiotemporal_instability so the active subset is synchronized."
            )

        missing_cols = set(bus_ids_active) - set(bus_load_df_active.columns)
        if missing_cols:
            raise ValueError(
                f"bus_load_df is missing {len(missing_cols)} active buses; "
                "rerun compute_spatiotemporal_instability to refresh the subset."
            )

        # Ensure instability is computed
        if self.instability_index is None:
            from geoloadst.core.instability_index import rms_instability
            from geoloadst.core.preprocessing import detrend_and_standardize

            self.values_std = detrend_and_standardize(
                bus_load_df_active,
                coords_active,
                dt_minutes=self.dt_minutes,
            )
            self.instability_index = rms_instability(self.values_std)
            self._active_values_std = self.values_std

        if len(self.instability_index) != n_active:
            raise ValueError(
                "instability_index length does not match the active bus subset. "
                "Rerun compute_spatiotemporal_instability with the desired limits so "
                "Moran/LISA use the same buses."
            )

        # Build spatial weights
        W = build_knn_weights(coords_active, k=k_neighbors)
        self._spatial_weights = W

        # Mean load
        mean_load = bus_load_df_active.loc[:, bus_ids_active].mean(axis=0).values

        # Consistency checks
        n_instab = len(self.instability_index)
        if n_instab != len(coords_active) or n_instab != len(mean_load):
            raise ValueError(
                "Length mismatch between instability_index, coords, and mean_load. "
                "Rerun compute_spatiotemporal_instability with desired max_buses/max_times "
                "so downstream steps use the same subset."
            )

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
            "bus_id": self.bus_ids_active,
            "x": self.coords_active[:, 0],
            "y": self.coords_active[:, 1],
            "mean_load": self.bus_load_df_active.mean(axis=0).values,
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

    @property
    def bus_ids_active(self) -> np.ndarray:
        """Active bus IDs after any subsampling."""
        if self._active_bus_ids is not None:
            return self._active_bus_ids
        return self.bus_ids

    @property
    def coords_active(self) -> np.ndarray:
        """Active bus coordinates after any subsampling."""
        if self._active_coords is not None:
            return self._active_coords
        return self.coords

    @property
    def bus_load_df_active(self) -> pd.DataFrame:
        """Active bus load dataframe after any subsampling/time trimming."""
        if self._active_bus_load_df is not None:
            return self._active_bus_load_df
        return self.bus_load_df

    @property
    def active_bus_indices(self) -> np.ndarray | None:
        """Indices of active buses within the full prepared list."""
        return self._active_bus_idx

    def _set_active_subset(
        self,
        bus_ids_subset: np.ndarray,
        coords_subset: np.ndarray,
        bus_load_df_subset: pd.DataFrame,
        active_idx: np.ndarray | None = None,
        values_std: np.ndarray | None = None,
    ) -> None:
        """Persist and expose the currently active bus subset."""
        if bus_ids_subset is None or coords_subset is None:
            raise ValueError("bus_ids_subset and coords_subset are required to set the active subset.")

        bus_ids_arr = np.asarray(bus_ids_subset, dtype=int)
        coords_arr = np.asarray(coords_subset)

        if coords_arr.shape[0] != len(bus_ids_arr):
            raise ValueError(
                f"Active coords length ({coords_arr.shape[0]}) does not match bus_ids ({len(bus_ids_arr)})."
            )
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
            raise ValueError("coords_subset must be shaped (N, 2).")

        try:
            bus_load_df_aligned = bus_load_df_subset.loc[:, bus_ids_arr]
        except KeyError as exc:
            raise ValueError(
                "bus_load_df_subset is missing buses from the active subset. "
                "Ensure the dataframe columns align with bus_ids_subset."
            ) from exc

        self._active_bus_ids = bus_ids_arr
        self._active_coords = coords_arr
        self._active_bus_load_df = bus_load_df_aligned
        self._active_bus_idx = (
            np.asarray(active_idx, dtype=int) if active_idx is not None else np.arange(len(bus_ids_arr))
        )
        self._subsample_idx = self._active_bus_idx
        if values_std is not None:
            self._active_values_std = values_std

        # Keep mapping aligned to the active subset
        self.bus_to_coord = {int(bus_ids_arr[i]): coords_arr[i] for i in range(len(bus_ids_arr))}

        # Update public-facing attributes to reflect the active subset
        self.bus_ids = bus_ids_arr
        self.coords = coords_arr
        self.bus_load_df = bus_load_df_aligned

    def _check_data_prepared(self) -> None:
        """Check that prepare_data has been called and state is valid."""
        if self.bus_load_df is None or self.bus_ids is None or self.coords is None:
            raise RuntimeError("Data not prepared. Call prepare_data() first.")
        self._validate_prepared_state()

    def _check_instability_computed(self) -> None:
        """Check that instability has been computed."""
        if self._stv_results is None:
            raise RuntimeError(
                "Instability not computed. Call compute_spatiotemporal_instability() first."
            )

    def _validate_prepared_state(self) -> None:
        """Ensure prepared data are non-empty and consistent."""
        if self.bus_ids is None or self.coords is None or self.bus_load_df is None:
            raise RuntimeError("Data not prepared. Call prepare_data() first.")

        if len(self.bus_ids) == 0 or self.coords.shape[0] == 0 or self.bus_load_df.shape[1] == 0:
            roi_desc = (
                f"roi={self.roi}" if self.roi is not None else f"roi_fraction={self.roi_fraction}"
            )
            raise ValueError(
                "No buses available after preparation. "
                f"{roi_desc}. "
                "Enlarge the ROI or verify bus coordinate fields (bus_geodata or bus.geo)."
            )

        if self.coords.shape[0] != len(self.bus_ids):
            raise ValueError(
                f"coords length ({self.coords.shape[0]}) does not match bus_ids ({len(self.bus_ids)})."
            )

        if self.bus_load_df.shape[1] != len(self.bus_ids):
            raise ValueError(
                f"bus_load_df columns ({self.bus_load_df.shape[1]}) do not match bus_ids ({len(self.bus_ids)})."
            )

