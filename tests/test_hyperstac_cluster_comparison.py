from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from SpatialBiologyToolkit.hyperstac.cluster_comparison import (
    discover_scan_settings,
    run_cluster_comparison,
)
from SpatialBiologyToolkit.hyperstac.preprocessing import (
    build_preflight_summary,
    write_preflight_report,
)


def _within_cluster_graph(labels: np.ndarray) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        for offset, index in enumerate(members):
            for neighbour in (
                members[(offset - 1) % len(members)],
                members[(offset + 1) % len(members)],
            ):
                rows.append(int(index))
                columns.append(int(neighbour))
    values = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(labels), len(labels))
    )


def _scan_adata() -> ad.AnnData:
    rng = np.random.default_rng(7)
    n_obs = 120
    base = np.repeat(np.arange(3), n_obs // 3)
    coordinates = np.column_stack(
        [base * 4 + rng.normal(0, 0.3, n_obs), rng.normal(0, 0.3, n_obs)]
    )
    matrix = np.column_stack([coordinates, rng.normal(size=(n_obs, 4))]).astype(
        np.float32
    )
    adata = ad.AnnData(matrix)
    adata.obs["roi"] = np.tile(["ROI_1", "ROI_2", "ROI_3", "ROI_4"], n_obs // 4)
    columns: list[str] = []
    umap_mapping: dict[str, str] = {}
    neighbours_mapping: dict[str, str] = {}
    for n_pcs in (0, 2):
        if n_pcs:
            adata.obsm[f"X_pca_{n_pcs}"] = matrix[:, :n_pcs]
        for n_neighbors in (5, 10):
            graph_label = f"N{n_neighbors}_P{n_pcs}"
            neighbours_key = f"neighbors_{graph_label}"
            connectivity_key = f"{neighbours_key}_connectivities"
            adata.obsp[connectivity_key] = _within_cluster_graph(base)
            adata.uns[neighbours_key] = {"connectivities_key": connectivity_key}
            adata.obsm[f"X_umap_{graph_label}"] = coordinates + rng.normal(
                0, 0.02, coordinates.shape
            )
            for resolution in (0.2, 0.4):
                column = f"leiden_{resolution}_N{n_neighbors}_P{n_pcs}"
                labels = base.astype(str)
                if resolution == 0.4:
                    labels = labels.copy()
                    labels[(base == 2) & (np.arange(n_obs) % 2 == 0)] = "3"
                adata.obs[column] = pd.Categorical(labels)
                columns.append(column)
                umap_mapping[column] = f"X_umap_{graph_label}"
                neighbours_mapping[column] = neighbours_key
    adata.uns["cluster_scan"] = {"cluster_columns": columns}
    adata.uns["cluster_scan_umap_keys"] = umap_mapping
    adata.uns["cluster_scan_neighbors_keys"] = neighbours_mapping
    return adata


class ClusterComparisonTests(unittest.TestCase):
    def test_discovers_scan_columns_after_h5ad_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clustered.h5ad"
            adata = _scan_adata()
            expected = [setting.column for setting in discover_scan_settings(adata)]
            adata.write_h5ad(path)

            restored = ad.read_h5ad(path)
            settings = discover_scan_settings(restored)

            self.assertIsInstance(
                restored.uns["cluster_scan"]["cluster_columns"], np.ndarray
            )
            self.assertEqual([setting.column for setting in settings], expected)

    def test_compares_complete_scan_without_survival(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = run_cluster_comparison(
                _scan_adata(),
                output_dir=output,
                silhouette_max_patches=100,
                random_seed=3,
            )

            self.assertEqual(len(result.setting_summary), 8)
            self.assertEqual(len(result.pairwise_agreement), 28)
            self.assertEqual(
                set(result.parameter_transition_summary["parameter"]),
                {"resolution", "n_neighbors", "n_pcs"},
            )
            self.assertTrue(result.setting_summary["pareto_candidate"].any())
            self.assertTrue(result.setting_summary["mean_local_ari"].notna().all())
            self.assertTrue((output / "clustering_parameter_scan_report.md").is_file())
            self.assertTrue(
                (output / "pairwise_adjusted_rand_index_heatmap.png").is_file()
            )

    def test_marker_environment_paths_rebase_copied_visualisation_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adata = _scan_adata()
            rows = []
            for column in adata.uns["cluster_scan"]["cluster_columns"]:
                directory = root / column
                tables = directory / "tables"
                tables.mkdir(parents=True)
                clusters = adata.obs[column].cat.categories.astype(str)
                pd.DataFrame(
                    np.arange(len(clusters) * 3).reshape(len(clusters), 3),
                    index=clusters,
                    columns=["A", "B", "C"],
                ).to_csv(tables / "cluster_mean_channel_intensity_zscore.csv")
                rows.append(
                    {"cluster_col": column, "output_dir": f"/remote/run/{column}"}
                )
            pd.DataFrame(rows).to_csv(
                root / "all_cluster_visualisation_summary.csv", index=False
            )

            result = run_cluster_comparison(
                adata,
                visualisation_dir=root,
                silhouette_max_patches=100,
            )

            self.assertFalse(result.environment_summary.empty)
            self.assertEqual(result.environment_membership["setting"].nunique(), 8)


class NormalisationPreflightTests(unittest.TestCase):
    def test_warns_transparently_and_writes_markdown_without_optional_tabulate(self):
        report = pd.DataFrame(
            {
                "channel": ["weak", "healthy"],
                "roi": ["R1", "R1"],
                "raw_p99": [0.6, 10.0],
                "background_value": [0.5, 0.5],
                "output_p99": [0.02, 0.8],
                "output_saturated_fraction": [0.0, 0.01],
                "corrected_positive_fraction": [0.05, 0.9],
                "present": [False, True],
            }
        )
        summary = build_preflight_summary(report)
        self.assertGreater(
            int(summary.loc[summary["channel"] == "weak", "warning_count"].iloc[0]), 0
        )
        self.assertEqual(
            int(summary.loc[summary["channel"] == "healthy", "warning_count"].iloc[0]),
            0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = write_preflight_report(summary, Path(temporary))
            text = path.read_text(encoding="utf-8")
        self.assertIn("HyPERSTAC normalization preflight", text)
        self.assertIn("| channel |", text)


if __name__ == "__main__":
    unittest.main()
