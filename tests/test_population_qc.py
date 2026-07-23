from __future__ import annotations

import unittest
from unittest.mock import patch
import sys
from types import SimpleNamespace

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse

from SpatialBiologyToolkit.config.models import PopulationEmbeddingQCConfig
from SpatialBiologyToolkit.population_qc import (
    MarkerExpectations,
    apply_population_mapping,
    assess_candidate_clustering,
    assess_clustering,
    compare_populations,
    compare_resolutions,
    create_leiden_sweep,
    discard_population_qc_columns,
    inspect_population_data,
    plot_clustering_qc,
    plot_marker_distributions,
    plot_population_heatmap,
    plot_population_representation,
    plot_resolution_membership,
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
            "leiden_0.3": pd.Categorical(np.where(populations == "T", "0", np.where(populations == "B", "1", "2"))),
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
        self.assertEqual(context.population_counts["population"].tolist(), ["T", "B", "Myeloid"])

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
        self.assertEqual(set(representation.group_summary["group_key"]), {"animal", "ROI"})
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
            table.obsm["X_umap"] = np.c_[values[:, 0], values[:, min(1, values.shape[1] - 1)]]

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


if __name__ == "__main__":
    unittest.main()
