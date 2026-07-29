from __future__ import annotations

import json
import unittest
from unittest.mock import patch
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse

from SpatialBiologyToolkit.config.models import PopulationEmbeddingQCConfig
from SpatialBiologyToolkit.population_embedding_qc.outputs import annotated_copy
from SpatialBiologyToolkit.population_qc import (
    MarkerExpectations,
    PopulationQCArtifactWriter,
    apply_population_mapping,
    assess_candidate_clustering,
    assess_clustering,
    compare_populations,
    compare_resolutions,
    create_leiden_sweep,
    discard_population_qc_columns,
    inspect_population_data,
    plot_clustering_qc,
    plot_clustering_qc_panels,
    plot_marker_distributions,
    plot_population_breakdown,
    plot_population_cell_gallery,
    plot_population_heatmap,
    plot_population_matrixplot,
    plot_population_representation,
    plot_population_umap,
    plot_resolution_membership,
    select_population_cell_panel,
    select_population_cells,
    subcluster_population,
    summarize_population_representation,
)


matplotlib.use("Agg")


def make_adata() -> ad.AnnData:
    rng = np.random.default_rng(14)
    populations = np.repeat(["T", "B", "Myeloid"], 24)
    centers = {
        "T": np.array([2.8, 0.2, 0.4, 0.7]),
        "B": np.array([0.2, 2.7, 0.3, 0.4]),
        "Myeloid": np.array([0.2, 0.3, 2.6, 1.5]),
    }
    matrix = np.vstack(
        [rng.normal(centers[value], 0.35, size=(1, 4)) for value in populations]
    ).clip(0)
    obs = pd.DataFrame(
        {
            "population": pd.Categorical(populations),
            "ROI": np.resize(["R1", "R2", "R3", "R4"], len(populations)),
            "animal": np.resize(["A1", "A1", "A2", "A2"], len(populations)),
            "ObjectNumber": np.arange(1, len(populations) + 1),
            "leiden_0.3": pd.Categorical(
                np.where(
                    populations == "T", "0", np.where(populations == "B", "1", "2")
                )
            ),
            "leiden_1.0": pd.Categorical(populations),
            "leiden_1.5": pd.Categorical(
                np.where(
                    populations == "T",
                    np.where(np.arange(len(populations)) % 2 == 0, "T0", "T1"),
                    populations,
                )
            ),
        },
        index=[f"cell-{index}" for index in range(len(populations))],
    )
    adata = ad.AnnData(
        matrix.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=["CD3", "CD19", "CD68", "PD1"]),
    )
    embedding_centers = {"T": (0, 0), "B": (4, 0), "Myeloid": (2, 4)}
    adata.obsm["X_umap"] = np.asarray(
        [rng.normal(embedding_centers[value], 0.3) for value in populations]
    )
    adata.obsm["X_pca"] = np.c_[
        adata.X,
        rng.normal(0, 0.1, size=(adata.n_obs, 2)),
    ]
    blocks = [np.ones((24, 24)) - np.eye(24) for _ in range(3)]
    adata.obsp["connectivities"] = sparse.block_diag(blocks, format="csr")
    adata.obsp["distances"] = sparse.block_diag(blocks, format="csr")
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"n_neighbors": 15, "method": "umap", "metric": "euclidean"},
    }
    return adata


class PopulationQCTests(unittest.TestCase):
    def test_stored_structural_qc_is_reused_without_recalculation(self):
        adata = make_adata()
        settings = PopulationEmbeddingQCConfig(
            roi_obs="ROI",
            sample_obs="animal",
            min_cluster_size=3,
            umap_k=3,
            density_grid_size=16,
            silhouette_max_cells=100,
        )
        calculated = assess_clustering(
            adata,
            "population",
            config=settings,
            reuse="never",
        )
        annotated = annotated_copy(adata, calculated)
        context = inspect_population_data(annotated, "population", case_key="animal")
        self.assertEqual(context.stored_population_qc["compatible"].tolist(), [True])
        with patch(
            "SpatialBiologyToolkit.population_qc.clustering.run_population_embedding_qc",
            side_effect=AssertionError("structural QC was unexpectedly recalculated"),
        ):
            restored = assess_clustering(
                annotated,
                "population",
                config=settings,
                reuse="require",
            )
        self.assertTrue(restored.run_summary["loaded_from_anndata_uns"])
        resolutions = compare_resolutions(
            annotated,
            "population",
            persistence_threshold=0.60,
            reuse="require",
        )
        self.assertFalse(resolutions.membership.empty)
        self.assertIn("sweep_persistence_fraction", resolutions.cluster_stability)

    def test_evidence_tables_and_focused_plots(self):
        import matplotlib.pyplot as plt

        adata = make_adata()
        expectations = MarkerExpectations(
            positive_markers=("CD3",),
            supportive_markers=("PD1",),
            negative_markers=("CD19", "CD68"),
            thresholds={"CD3": 1.0},
        )
        context = inspect_population_data(
            adata,
            "population",
            case_key="animal",
            expectations=expectations,
        )
        self.assertEqual(context.n_cells, 72)
        self.assertEqual(context.case_key, "animal")
        self.assertEqual(
            context.population_counts["population"].tolist(), ["T", "B", "Myeloid"]
        )

        expression = compare_populations(
            adata,
            "population",
            "T",
            reference_population="B",
            expectations=expectations,
            max_cells_per_group=12,
        )
        cd3 = expression.marker_statistics.set_index("marker").loc["CD3"]
        self.assertGreater(cd3["auc_effect"], 0.8)
        self.assertEqual(cd3["rule_status"], "supports expectation")
        distributions = plot_marker_distributions(expression, markers=["CD3", "CD19"])
        heatmap = plot_population_heatmap(
            adata,
            "population",
            markers=["CD3", "CD19", "CD68"],
            max_cells_per_population=12,
        )
        self.assertEqual(heatmap.display_data.shape, (3, 3))

        representation = summarize_population_representation(
            adata,
            "population",
            case_key="animal",
        )
        self.assertEqual(
            set(representation.group_summary["group_key"]), {"animal", "ROI"}
        )
        representation_plot = plot_population_representation(
            representation,
            "T",
            group_key="animal",
        )

        resolutions = compare_resolutions(adata, "population")
        resolution_plot = plot_resolution_membership(resolutions, "T")
        self.assertEqual(set(resolutions.sweep_columns["resolution"]), {0.3, 1.0, 1.5})

        qc = assess_clustering(
            adata,
            "population",
            config=PopulationEmbeddingQCConfig(
                min_cluster_size=3,
                umap_k=3,
                density_grid_size=16,
                silhouette_max_cells=100,
            ),
        )
        qc_plot = plot_clustering_qc(qc, populations=["T", "B", "Myeloid"])
        qc_panels = plot_clustering_qc_panels(
            qc,
            populations=["T", "B", "Myeloid"],
            max_metrics_per_panel=4,
        )
        self.assertGreaterEqual(len(qc_panels), 2)
        selection = select_population_cells(
            adata,
            "population",
            "T",
            strategies=["typical", "core", "boundary", "contradictory"],
            n_per_strategy=2,
            expectations=expectations,
            clustering_qc=qc,
        )
        self.assertIn("typical", set(selection.cells["strategy"]))
        self.assertEqual(selection.cells["obs_name"].nunique(), len(selection.cells))

        for figure in (
            distributions.figure,
            heatmap.figure,
            representation_plot.figure,
            resolution_plot.figure,
            qc_plot.figure,
            *(panel.figure for panel in qc_panels.values()),
        ):
            plt.close(figure)

    def test_candidate_columns_are_in_memory_and_reversible(self):
        adata = make_adata()

        def fake_pca(table, n_comps, **_kwargs):
            values = np.asarray(table.X)
            table.obsm["X_pca"] = values[:, :n_comps]

        def fake_neighbors(table, **_kwargs):
            graph = sparse.eye(table.n_obs, format="csr")
            table.obsp["connectivities"] = graph
            table.obsp["distances"] = graph
            table.uns["neighbors"] = {
                "connectivities_key": "connectivities",
                "distances_key": "distances",
            }

        def fake_umap(table, **_kwargs):
            values = np.asarray(table.obsm["X_population_qc"])
            table.obsm["X_umap"] = np.c_[
                values[:, 0], values[:, min(1, values.shape[1] - 1)]
            ]

        def fake_leiden(table, *, resolution, key_added, **_kwargs):
            groups = 2 if resolution < 0.4 else 3
            table.obs[key_added] = pd.Categorical(
                np.arange(table.n_obs).astype(int) % groups
            )

        fake_scanpy = SimpleNamespace(
            pp=SimpleNamespace(pca=fake_pca, neighbors=fake_neighbors),
            tl=SimpleNamespace(umap=fake_umap, leiden=fake_leiden),
        )
        with patch.dict(sys.modules, {"scanpy": fake_scanpy}):
            sweep = create_leiden_sweep(
                adata,
                [0.2, 0.5],
                output_prefix="qc_leiden",
            )
            candidate = subcluster_population(
                adata,
                "population",
                "T",
                resolutions=[0.2, 0.5],
                markers=["CD3", "PD1", "CD19"],
                output_prefix="qc_tcell",
                n_neighbors=5,
                n_pcs=2,
                attach=True,
            )

        self.assertEqual(sweep.columns, ("qc_leiden_0.2", "qc_leiden_0.5"))
        self.assertTrue(set(sweep.columns).issubset(adata.obs.columns))
        self.assertEqual(candidate.adata.n_obs, 24)
        self.assertTrue(candidate.attached_to_source)
        self.assertTrue(set(candidate.columns).issubset(adata.obs.columns))
        self.assertIn("X_population_qc", candidate.adata.obsm)
        self.assertIn("X_umap", candidate.adata.obsm)
        self.assertIn("sweep_regex", candidate.parameters)

        with patch(
            "SpatialBiologyToolkit.population_qc.clustering.run_population_embedding_qc",
            return_value="candidate-qc",
        ) as qc_runner:
            self.assertEqual(assess_candidate_clustering(candidate), "candidate-qc")
        qc_kwargs = qc_runner.call_args.kwargs
        self.assertEqual(qc_kwargs["mode"], "sweep")
        self.assertEqual(qc_kwargs["sweep_columns"], list(candidate.columns))

        mapped = apply_population_mapping(
            adata,
            "population",
            {"T": "Lymphocyte", "B": "Lymphocyte"},
            "population_candidate",
        )
        self.assertEqual(
            set(mapped.adata.obs["population_candidate"].astype(str)),
            {"Lymphocyte", "Myeloid"},
        )
        removed = discard_population_qc_columns(
            adata,
            [*sweep.columns, *candidate.columns, "population_candidate"],
        )
        self.assertEqual(len(removed), 5)
        self.assertFalse(any(column in adata.obs for column in removed))

    def test_staged_plots_cell_panel_and_artifact_exports(self):
        import matplotlib.pyplot as plt

        adata = make_adata()
        expectations = MarkerExpectations(
            positive_markers=("CD3",),
            supportive_markers=("PD1",),
            negative_markers=("CD19", "CD68"),
        )
        matrix = plot_population_matrixplot(
            adata,
            "population",
            markers=["CD3", "CD19", "CD68"],
            standardization="marker_robust_zscore",
            standardization_clip=2.5,
            max_cells_per_population=12,
        )
        self.assertIsInstance(matrix.data, pd.DataFrame)
        self.assertEqual(
            list(matrix.data.columns),
            ["cells", "sampled_cells", "CD3", "CD19", "CD68"],
        )
        self.assertLessEqual(float(matrix.display_data.max().max()), 2.5)
        self.assertGreaterEqual(float(matrix.display_data.min().min()), -2.5)

        global_umap = plot_population_umap(
            adata,
            "population",
            max_cells=30,
            random_state=9,
        )
        focused_umap = plot_population_umap(
            adata,
            "population",
            population="T",
            competitors=["B"],
            max_cells=30,
            random_state=9,
        )
        self.assertLessEqual(len(global_umap.data), 30)
        self.assertEqual(
            set(focused_umap.data["role"]),
            {"background", "competitor", "target"},
        )
        self.assertTrue(
            {"obs_name", "population", "umap_1", "umap_2"}.issubset(
                global_umap.data.columns
            )
        )

        representation = summarize_population_representation(
            adata,
            "population",
            group_keys=["animal", "ROI"],
        )
        breakdown = plot_population_breakdown(
            representation,
            group_key="animal",
        )
        self.assertEqual(breakdown.display_data.shape, (2, 3))

        t_positions = np.flatnonzero(
            adata.obs["population"].astype(str).to_numpy() == "T"
        )
        fake_cell_metrics = pd.DataFrame(
            {
                "cell_index": adata.obs_names[t_positions].astype(str),
                "reference_population": "T",
                "boundary_class": "boundary",
                "graph_neighbour_purity": np.linspace(0.0, 1.0, len(t_positions)),
            }
        )
        panel = select_population_cell_panel(
            adata,
            "population",
            "T",
            marker="CD3",
            expectations=expectations,
            clustering_qc=SimpleNamespace(cell_metrics=fake_cell_metrics),
            diversity_keys=["animal", "ROI"],
            max_per_diversity_group=5,
            random_state=7,
        )
        self.assertEqual(len(panel.cells), 20)
        self.assertEqual(panel.cells["obs_name"].nunique(), 20)
        self.assertEqual(
            set(panel.cells["strategy"]),
            {"typical", "boundary", "marker_high", "contradictory", "random"},
        )
        self.assertTrue({"animal", "ROI"}.issubset(panel.cells.columns))

        gallery_figure, gallery_axes = plt.subplots(4, 5)
        with patch(
            "SpatialBiologyToolkit.spatialdata.plot_spatialdata_cells",
            return_value=(gallery_figure, gallery_axes),
        ) as gallery_plotter:
            gallery = plot_population_cell_gallery(
                SimpleNamespace(images={}, labels={}),
                panel,
                channel=("CD3", "CD19", "CD68"),
                ncols=5,
                separate_strategies=False,
                compact_titles=True,
            )
        self.assertEqual(set(gallery), {"all"})
        self.assertEqual(len(gallery["all"].data), 20)
        self.assertIn("typical #1", gallery_axes.reshape(-1)[0].get_title())
        self.assertEqual(gallery_plotter.call_args.kwargs["ncols"], 5)

        expression = compare_populations(
            adata,
            "population",
            "T",
            reference_population="B",
            expectations=expectations,
            max_cells_per_group=12,
        )
        adata.obs["population_candidate"] = pd.Categorical(
            adata.obs["population"].astype(str).replace({"T": "T cell"})
        )
        with TemporaryDirectory() as directory:
            writer = PopulationQCArtifactWriter(
                directory,
                stage="global",
            )
            paths = writer.save_plot_result(
                matrix,
                "marker_matrix",
                source="plot_population_matrixplot",
            )
            result_paths = writer.save_result_tables(
                expression,
                "population_T_expression",
            )
            panel_path = writer.child(
                stage="population",
                population="T",
            ).save_table(panel.cells, "selected_cells")
            candidate_path = writer.child(
                stage="candidate",
                population="candidate_merge",
            ).save_observation_columns(
                adata,
                "population_candidate",
                source="apply_population_mapping",
            )
            spatialdata_candidate_path = writer.child(
                stage="candidate",
                population="candidate_merge",
            ).save_observation_columns(
                SimpleNamespace(tables={"table": adata}, attrs={}),
                ["population_candidate"],
                name="candidate_observation_columns_from_spatialdata",
                table_name="table",
                source="apply_population_mapping",
            )
            posterior = pd.DataFrame(
                [
                    {
                        "source_population": "T",
                        "proposed_label": "T cell",
                        "decision": "retain",
                        "candidate_key": "",
                        "structural_confidence": "high",
                        "identity_confidence": "high",
                        "panel_resolution": "high",
                        "replication_image_confidence": "moderate",
                        "alternatives": "",
                        "supporting_evidence": "global:all:figure:marker_matrix",
                        "contradictory_evidence": "",
                        "next_test": "",
                    }
                ]
            )
            posterior_path = writer.save_posterior_mapping(posterior)
            conclusion_path = writer.record_stage_conclusion(
                stage_id="global-1",
                hypothesis="The clustering is structurally supported.",
                conclusion="Synthetic evidence supports the test clustering.",
                decision="continue",
                evidence_artifact_ids=["global:all:figure:marker_matrix"],
                notebook_path="notebooks/10_global_clustering.ipynb",
                include_in_summary=True,
                priority=1,
            )
            audit_path = writer.save_execution_audit(
                {
                    "zarr_write_performed": False,
                    "source_population_column_preserved": True,
                    "candidate_columns_created": ["population_candidate"],
                    "candidate_columns_removed": ["population_candidate"],
                    "random_seed": 9,
                }
            )
            self.assertTrue(all(path.exists() for path in paths.values()))
            self.assertTrue(all(path.exists() for path in result_paths.values()))
            self.assertTrue(panel_path.exists())
            self.assertTrue(candidate_path.exists())
            self.assertTrue(spatialdata_candidate_path.exists())
            self.assertTrue(posterior_path.exists())
            self.assertTrue(conclusion_path.exists())
            self.assertTrue(audit_path.exists())
            candidate_columns = pd.read_csv(
                candidate_path,
                dtype=str,
                keep_default_na=False,
            )
            self.assertEqual(
                list(candidate_columns.columns),
                ["obs_name", "population_candidate"],
            )
            self.assertEqual(len(candidate_columns), adata.n_obs)
            self.assertTrue(candidate_columns["obs_name"].is_unique)
            self.assertEqual(
                candidate_columns["obs_name"].tolist(),
                adata.obs_names.astype(str).tolist(),
            )
            self.assertEqual(
                candidate_columns["population_candidate"].tolist(),
                adata.obs["population_candidate"].astype(str).tolist(),
            )
            manifest = writer.manifest()
            self.assertGreaterEqual(len(manifest), 7)
            self.assertEqual(
                set(manifest.columns),
                {
                    "artifact_id",
                    "stage",
                    "population",
                    "kind",
                    "name",
                    "path",
                    "source",
                    "created_at",
                    "sha256",
                    "metadata_json",
                },
            )
            self.assertTrue(
                all((Path(directory) / path).exists() for path in manifest["path"])
            )
            candidate_record = manifest.loc[
                manifest["path"] == candidate_path.relative_to(directory).as_posix()
            ].iloc[0]
            candidate_metadata = json.loads(candidate_record["metadata_json"])
            self.assertEqual(
                candidate_metadata["observation_columns"],
                ["population_candidate"],
            )
            self.assertEqual(candidate_metadata["merge_key"], "obs_name")
            self.assertTrue(candidate_metadata["merge_key_unique"])
            self.assertEqual(candidate_metadata["n_observations"], adata.n_obs)
            spatialdata_record = manifest.loc[
                manifest["path"]
                == spatialdata_candidate_path.relative_to(directory).as_posix()
            ].iloc[0]
            spatialdata_metadata = json.loads(spatialdata_record["metadata_json"])
            self.assertEqual(spatialdata_metadata["table_name"], "table")

        for figure in (
            matrix.figure,
            global_umap.figure,
            focused_umap.figure,
            breakdown.figure,
            gallery_figure,
        ):
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
