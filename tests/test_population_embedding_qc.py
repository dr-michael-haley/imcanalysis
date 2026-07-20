import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from SpatialBiologyToolkit.config.models import PopulationEmbeddingQCConfig
from SpatialBiologyToolkit.population_embedding_qc import run_population_embedding_qc
from SpatialBiologyToolkit.population_embedding_qc.embedding_metrics import (
    calculate_embedding_metrics,
    stratified_sample_indices,
)
from SpatialBiologyToolkit.population_embedding_qc.graph_metrics import (
    calculate_graph_metrics,
)
from SpatialBiologyToolkit.population_embedding_qc.inspection import (
    detect_sweep_columns,
    inspect_anndata,
)
from SpatialBiologyToolkit.population_embedding_qc.models import DEFAULT_METRICS
from SpatialBiologyToolkit.population_embedding_qc.outputs import annotated_copy
from SpatialBiologyToolkit.population_embedding_qc.plotting import plot_concern_heatmap
from SpatialBiologyToolkit.population_embedding_qc.scoring import (
    normalize_concern,
    score_cluster_metrics,
    threshold_flags,
)
from SpatialBiologyToolkit.population_embedding_qc.sweep_metrics import (
    calculate_sweep_metrics,
)
from SpatialBiologyToolkit.pipeline.registry import get_stage


def make_adata(*, graph=True, pca=True, population=True, sweep=True):
    rng = np.random.default_rng(12)
    labels = np.array(["A"] * 12 + ["B"] * 12)
    obs = pd.DataFrame(index=[f"cell-{index}" for index in range(len(labels))])
    if population:
        obs["population"] = pd.Categorical(labels, categories=["B", "A"], ordered=True)
    obs["leiden"] = np.where(labels == "A", "0", "1")
    if sweep:
        obs["leiden_0.2"] = np.where(labels == "A", "0", "1")
        obs["leiden_0.5"] = np.where(labels == "A", "0", "1")
        obs["leiden_1.0"] = np.where(np.arange(len(labels)) < 6, "2", np.where(labels == "A", "0", "1"))
    adata = ad.AnnData(np.zeros((len(labels), 2)), obs=obs)
    adata.obsm["X_umap"] = np.r_[rng.normal(0, 0.2, (12, 2)), rng.normal(4, 0.2, (12, 2))]
    if pca:
        adata.obsm["X_pca"] = np.c_[adata.obsm["X_umap"], rng.normal(0, 0.1, (24, 2))]
    if graph:
        blocks = [np.ones((12, 12)) - np.eye(12), np.ones((12, 12)) - np.eye(12)]
        adata.obsp["connectivities"] = sparse.block_diag(blocks, format="csr")
    return adata


class PopulationInspectionTests(unittest.TestCase):
    def test_auto_reference_and_numeric_sweep_detection(self):
        adata = make_adata()
        inspection = inspect_anndata(
            adata,
            population_obs=None,
            mode="auto",
            sweep_columns=None,
            sweep_regex=r"^leiden_(?P<resolution>\d+(?:\.\d+)?)$",
            reference_resolution=None,
            umap_key="X_umap",
            pca_key="X_pca",
            pca_dimensions=3,
            connectivities_key=None,
        )
        self.assertEqual(inspection.reference_column, "population")
        self.assertEqual([resolution for _, resolution in inspection.sweep], [0.2, 0.5, 1.0])
        self.assertEqual(inspection.cluster_order, ["B", "A"])
        self.assertEqual(inspection.pca.shape[1], 3)
        self.assertTrue(sparse.issparse(inspection.connectivities))

    def test_auto_falls_back_to_leiden_then_median_sweep(self):
        adata = make_adata(population=False)
        inspection = inspect_anndata(
            adata,
            population_obs=None,
            mode="auto",
            sweep_columns=None,
            sweep_regex=r"^leiden_(?P<resolution>\d+(?:\.\d+)?)$",
            reference_resolution=None,
            umap_key="X_umap",
            pca_key="missing",
            pca_dimensions=30,
            connectivities_key=None,
        )
        self.assertEqual(inspection.reference_column, "leiden")
        del adata.obs["leiden"]
        inspection = inspect_anndata(
            adata,
            population_obs=None,
            mode="auto",
            sweep_columns=None,
            sweep_regex=r"^leiden_(?P<resolution>\d+(?:\.\d+)?)$",
            reference_resolution=None,
            umap_key="X_umap",
            pca_key="missing",
            pca_dimensions=30,
            connectivities_key=None,
        )
        self.assertEqual(inspection.reference_column, "leiden_0.5")

    def test_custom_sweep_regex_explicit_columns_and_duplicates(self):
        obs = pd.DataFrame({"cluster_r2": ["a", "a", "b"], "cluster_r10": ["a", "a", "b"]})
        detected, warnings = detect_sweep_columns(
            obs,
            sweep_regex=r"^cluster_r(?P<resolution>\d+)$",
            explicit_columns=["cluster_r10", "cluster_r2"],
        )
        self.assertEqual(detected, [("cluster_r2", 2.0), ("cluster_r10", 10.0)])
        self.assertTrue(any("Duplicate clustering" in warning for warning in warnings))

    def test_missing_inputs_and_labels_are_explicit(self):
        adata = make_adata(graph=False, pca=False)
        del adata.obsm["X_umap"]
        with self.assertRaisesRegex(ValueError, "Required UMAP"):
            inspect_anndata(
                adata,
                population_obs="population",
                mode="single",
                sweep_columns=None,
                sweep_regex=r"^leiden_(?P<resolution>\d+(?:\.\d+)?)$",
                reference_resolution=None,
                umap_key="X_umap",
                pca_key="X_pca",
                pca_dimensions=30,
                connectivities_key=None,
            )
        adata = make_adata(graph=False, pca=False)
        adata.obs.loc[adata.obs.index[:2], "population"] = np.nan
        result = run_population_embedding_qc(
            adata,
            config=PopulationEmbeddingQCConfig(min_cluster_size=3, umap_k=3, density_grid_size=16),
        )
        self.assertEqual(result.run_summary["excluded_cells"], 2)
        self.assertFalse(result.run_summary["graph_available"])
        self.assertFalse(result.run_summary["pca_available"])
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = plot_concern_heatmap(
                result,
                Path(temp_dir) / "detailed_missing_graph",
                detailed=True,
            )
            self.assertEqual(len(paths), 2)


class PopulationGraphMetricTests(unittest.TestCase):
    def test_isolated_graph_has_better_support_than_mixed_graph(self):
        labels = pd.Series(["A"] * 4 + ["B"] * 4, dtype="string")
        isolated = sparse.block_diag([np.ones((4, 4)) - np.eye(4)] * 2, format="csr")
        mixed = sparse.csr_matrix(np.ones((8, 8)) - np.eye(8))
        kwargs = dict(cluster_order=["A", "B"], boundary_threshold=0.7, high_entropy_threshold=0.6, min_component_size=2)
        good = calculate_graph_metrics(isolated, labels, **kwargs)
        bad = calculate_graph_metrics(mixed, labels, **kwargs)
        self.assertTrue((good.cluster_metrics["graph_neighbour_impurity"] < bad.cluster_metrics["graph_neighbour_impurity"]).all())
        self.assertTrue((good.cluster_metrics["graph_conductance"] < bad.cluster_metrics["graph_conductance"]).all())
        self.assertTrue(sparse.issparse(good.graph))

    def test_components_competitor_and_zero_degree_cells(self):
        labels = pd.Series(["A"] * 4 + ["B"] * 4, dtype="string")
        graph = sparse.lil_matrix((8, 8))
        graph[0, 1] = graph[1, 0] = 1
        graph[2, 3] = graph[3, 2] = 1
        graph[1, 4] = graph[4, 1] = 4
        result = calculate_graph_metrics(
            graph.tocsr(), labels, cluster_order=["A", "B"], boundary_threshold=0.7, high_entropy_threshold=0.6, min_component_size=2
        )
        self.assertEqual(result.cluster_metrics.loc["A", "graph_component_count"], 2)
        self.assertEqual(result.competitors.loc[result.competitors["cluster"] == "A", "competitor"].iloc[0], "B")
        self.assertGreater(result.cluster_metrics.loc["B", "graph_zero_degree_fraction"], 0)


class PopulationEmbeddingMetricTests(unittest.TestCase):
    def test_isolated_umap_has_lower_impurity_than_overlap(self):
        rng = np.random.default_rng(3)
        labels = pd.Series(["A"] * 20 + ["B"] * 20, dtype="string")
        isolated = np.r_[rng.normal(0, 0.1, (20, 2)), rng.normal(4, 0.1, (20, 2))]
        overlap = rng.normal(0, 1, (40, 2))
        kwargs = dict(cluster_order=["A", "B"], graph=None, pca=None, umap_k=5, silhouette_max_cells=100, density_max_cells_per_cluster=100, density_grid_size=16, min_cluster_size=3, include_optional_metrics=False, random_seed=42)
        good = calculate_embedding_metrics(isolated, labels, **kwargs)
        bad = calculate_embedding_metrics(overlap, labels, **kwargs)
        self.assertLess(good.cluster_metrics["umap_neighbour_impurity"].mean(), bad.cluster_metrics["umap_neighbour_impurity"].mean())
        self.assertGreater(good.cluster_metrics["umap_silhouette_median"].mean(), bad.cluster_metrics["umap_silhouette_median"].mean())

    def test_sampling_is_deterministic_and_preserves_small_clusters(self):
        labels = np.array(["large"] * 100 + ["small"] * 3)
        left = stratified_sample_indices(labels, 20, 7)
        right = stratified_sample_indices(labels, 20, 7)
        np.testing.assert_array_equal(left, right)
        self.assertEqual(int(np.sum(labels[left] == "small")), 3)


class PopulationSweepAndScoringTests(unittest.TestCase):
    def test_sweep_persistence_split_and_consensus(self):
        obs = pd.DataFrame(
            {
                "population": ["A"] * 6 + ["B"] * 6,
                "leiden_0.2": ["0"] * 6 + ["1"] * 6,
                "leiden_0.5": ["0"] * 3 + ["2"] * 3 + ["1"] * 6,
                "leiden_1.0": ["0"] * 2 + ["2"] * 2 + ["3"] * 2 + ["1"] * 6,
            }
        )
        result = calculate_sweep_metrics(
            obs,
            reference_column="population",
            cluster_order=["A", "B"],
            sweep=[("leiden_0.2", 0.2), ("leiden_0.5", 0.5), ("leiden_1.0", 1.0)],
            persistence_threshold=0.75,
        )
        self.assertGreater(result.reference_metrics.loc["B", "sweep_persistence_fraction"], result.reference_metrics.loc["A", "sweep_persistence_fraction"])
        self.assertGreater(result.reference_metrics.loc["A", "sweep_split_entropy"], result.reference_metrics.loc["B", "sweep_split_entropy"])
        self.assertEqual(len(result.jaccard_matrices), 2)
        self.assertLess(result.reference_metrics.size, len(obs) ** 2)

    def test_normalization_thresholds_missing_and_group_coverage(self):
        high_worse = next(item for item in DEFAULT_METRICS if item.key == "graph_conductance")
        high_better = next(item for item in DEFAULT_METRICS if item.key == "umap_silhouette_median")
        values = pd.Series([high_worse.good_anchor, high_worse.bad_anchor, 99.0, np.nan])
        np.testing.assert_allclose(normalize_concern(values, high_worse).iloc[:3], [0, 1, 1])
        better_values = pd.Series([high_better.good_anchor, high_better.bad_anchor])
        np.testing.assert_allclose(normalize_concern(better_values, high_better), [0, 1])
        flags = threshold_flags(pd.Series([0.24, 0.26, np.nan]), high_worse)
        self.assertEqual(flags.iloc[0], False)
        self.assertEqual(flags.iloc[1], True)
        self.assertTrue(pd.isna(flags.iloc[2]))
        raw = pd.DataFrame({"graph_conductance": [0.1, np.nan]}, index=["A", "B"])
        _scores, _flags, summary = score_cluster_metrics(raw, DEFAULT_METRICS)
        self.assertTrue(summary.loc["B", "graph_separation_concern_insufficient_coverage"])
        self.assertTrue(pd.isna(summary.loc["B", "overall_concern"]))


class PopulationIntegrationTests(unittest.TestCase):
    def test_api_does_not_mutate_input_and_annotation_is_namespaced(self):
        adata = make_adata()
        obs_columns = list(adata.obs.columns)
        uns_keys = list(adata.uns.keys())
        result = run_population_embedding_qc(
            adata,
            config=PopulationEmbeddingQCConfig(min_cluster_size=3, umap_k=3, density_grid_size=16),
        )
        self.assertEqual(list(adata.obs.columns), obs_columns)
        self.assertEqual(list(adata.uns.keys()), uns_keys)
        annotated = annotated_copy(adata, result)
        self.assertIn("embedding_qc_umap_purity", annotated.obs)
        self.assertIn("population_embedding_qc", annotated.uns)
        self.assertNotIn("embedding_qc_umap_purity", adata.obs)

    def test_config_registry_and_required_tables(self):
        stage = get_stage("popqc")
        self.assertEqual(stage.environment_keys, ["cellcharter"])
        self.assertEqual(stage.config_sections, ["general", "population_embedding_qc"])
        self.assertEqual(stage.depends_on, [])
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_population_embedding_qc(
                make_adata(),
                output_dir=temp_dir,
                config=PopulationEmbeddingQCConfig(min_cluster_size=3, umap_k=3, density_grid_size=16),
            )
            for filename in (
                "cluster_metrics_raw.tsv",
                "cluster_metrics_concern_scores.tsv",
                "cluster_metric_threshold_flags.tsv",
                "cluster_summary.tsv",
                "cluster_competitors.tsv",
                "pairwise_graph_connectivity.tsv",
                "pairwise_umap_density_overlap.tsv",
                "metric_definitions.tsv",
            ):
                self.assertTrue((Path(temp_dir) / "Tables" / filename).is_file(), filename)
            self.assertTrue((Path(temp_dir) / "Report" / "analysis_report.md").is_file())
            self.assertTrue((Path(temp_dir) / "Run" / "run_summary.json").is_file())
            self.assertTrue(result.output_files)


if __name__ == "__main__":
    unittest.main()
