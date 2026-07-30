from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SpatialBiologyToolkit.population_qc import (
    MaxFuseSourceSpec,
    PopulationQCArtifactWriter,
    inspect_maxfuse_inputs,
    plot_maxfuse_label_heatmap,
    plot_maxfuse_threshold_sensitivity,
    summarize_maxfuse_evidence,
)


matplotlib.use("Agg")


def make_target() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "population": pd.Categorical(["0", "0", "0", "1", "1", "1"]),
            "ROI": ["R1", "R1", "R2", "R1", "R2", "R2"],
            "ObjectNumber": [1, 2, 3, 4, 5, 6],
        },
        index=[f"cell-{index}" for index in range(6)],
    )
    return ad.AnnData(
        np.zeros((6, 1), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["marker"]),
    )


class MaxFusePopulationQCTests(unittest.TestCase):
    def test_no_maxfuse_evidence_is_optional_and_non_mutating(self):
        adata = make_target()
        before = adata.obs.copy()
        result = summarize_maxfuse_evidence(adata, "population")
        self.assertFalse(result.available)
        self.assertTrue(result.sources.empty)
        self.assertIn("No MaxFuse", result.warnings[0])
        pd.testing.assert_frame_equal(adata.obs, before)
        with self.assertRaisesRegex(ValueError, "No MaxFuse sources"):
            summarize_maxfuse_evidence(adata, "population", require=True)

    def test_embedded_multiple_references_keep_coverage_and_consensus_separate(self):
        adata = make_target()
        adata.obs["AtlasA_maxfuse_score"] = [0.95, 0.92, 0.20, 0.91, 0.40, np.nan]
        adata.obs["AtlasA_annotation_coarse"] = pd.Categorical(
            ["T", "T", "B", "Myeloid", "Myeloid", None]
        )
        adata.obs["AtlasA_cell_state"] = pd.Categorical(
            ["active", "resting", "active", "resting", "resting", None]
        )
        adata.obs["AtlasB_maxfuse_score"] = [0.88, 0.86, 0.84, 0.93, 0.91, 0.10]
        adata.obs["AtlasB_annotation_level_4"] = pd.Categorical(
            ["CD8 T", "CD8 T", "CD4 T", "Macrophage", "Macrophage", "Monocyte"]
        )
        columns_before = adata.obs.columns.tolist()

        result = summarize_maxfuse_evidence(
            adata,
            "population",
            score_threshold=0.85,
        )
        audit_only = inspect_maxfuse_inputs(adata, score_threshold=0.85)

        self.assertEqual(set(result.sources["source"]), {"AtlasA", "AtlasB"})
        self.assertTrue(audit_only.available)
        self.assertFalse(
            audit_only.parameters["population_label_associations_summarized"]
        )
        roles = result.sources.set_index("label_column")["label_role"].to_dict()
        self.assertEqual(roles["AtlasA_annotation_coarse"], "lineage")
        self.assertEqual(roles["AtlasA_cell_state"], "state")
        self.assertEqual(roles["AtlasB_annotation_level_4"], "subtype")

        atlas_a_zero = result.population_summary.loc[
            result.population_summary["source"].eq("AtlasA")
            & result.population_summary["population"].eq("0")
            & result.population_summary["label_column"].eq(
                "AtlasA_annotation_coarse"
            )
        ].iloc[0]
        self.assertEqual(atlas_a_zero["population_cells"], 3)
        self.assertEqual(atlas_a_zero["matched_cells"], 3)
        self.assertEqual(atlas_a_zero["evaluated_cells"], 2)
        self.assertAlmostEqual(atlas_a_zero["evaluated_coverage"], 2 / 3)
        self.assertEqual(atlas_a_zero["top_label"], "T")
        self.assertEqual(atlas_a_zero["top_label_fraction"], 1.0)

        selected = result.for_cells(["cell-0", "cell-3"])
        self.assertEqual(len(selected), 6)
        self.assertEqual(
            set(selected["label_role"]),
            {"lineage", "subtype", "state"},
        )
        self.assertEqual(adata.obs.columns.tolist(), columns_before)

        heatmap = plot_maxfuse_label_heatmap(
            result,
            source="AtlasA",
            label_column="AtlasA_annotation_coarse",
        )
        sensitivity = plot_maxfuse_threshold_sensitivity(result)
        self.assertEqual(heatmap.display_data.shape, (2, 2))
        self.assertEqual(set(sensitivity.data["source"]), {"AtlasA", "AtlasB"})
        plt.close(heatmap.figure)
        plt.close(sensitivity.figure)

    def test_external_h5ad_aligns_reordered_partial_obs_names_and_reads_metadata(self):
        adata = make_target()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "transfers.h5ad"
            sidecar = ad.AnnData(
                obs=pd.DataFrame(
                    {
                        "Reference_maxfuse_score": [0.91, 0.82, 0.96],
                        "Reference_annotation_coarse": pd.Categorical(
                            ["Myeloid", "T", "T"]
                        ),
                    },
                    index=["cell-4", "cell-1", "cell-0"],
                )
            )
            sidecar.uns["score_threshold_used_for_confusion_matrices"] = 0.8
            sidecar.uns["rna_reference_paths"] = {"Reference": "atlas.h5ad"}
            sidecar.uns["shared_proteins"] = ["CD3", "CD68"]
            sidecar.uns["shared_genes"] = ["CD3D", "CD68"]
            sidecar.write_h5ad(path)

            result = summarize_maxfuse_evidence(
                adata,
                "population",
                paths=path,
            )

        audit = result.alignment_audit.iloc[0]
        self.assertEqual(audit["overlap_cells"], 3)
        self.assertAlmostEqual(audit["target_coverage"], 0.5)
        self.assertFalse(audit["same_order"])
        source = result.sources.iloc[0]
        self.assertEqual(source["score_threshold"], 0.8)
        self.assertEqual(source["reference_path"], "atlas.h5ad")
        self.assertEqual(source["shared_proteins"], 2)
        missing = result.for_cells(["cell-5"])
        self.assertTrue(missing["score"].isna().all())
        self.assertTrue(missing["label"].isna().all())

    def test_external_csv_supports_explicit_composite_identity(self):
        adata = make_target()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nonstandard.csv"
            pd.DataFrame(
                {
                    "ROI": ["R2", "R1", "R2"],
                    "ObjectNumber": [5, 1, 3],
                    "similarity": [0.93, 0.91, 0.20],
                    "transferred_type": ["Myeloid", "T", "B"],
                }
            ).to_csv(path, index=False)
            spec = MaxFuseSourceSpec(
                name="CustomAtlas",
                path=str(path),
                score_column="similarity",
                score_threshold=0.85,
                label_columns=("transferred_type",),
                label_roles={"transferred_type": "lineage"},
                join_keys=("ROI", "ObjectNumber"),
            )
            result = summarize_maxfuse_evidence(
                adata,
                "population",
                source_specs=[spec],
            )

        audit = result.alignment_audit.iloc[0]
        self.assertEqual(audit["join_method"], "composite")
        self.assertEqual(audit["overlap_cells"], 3)
        cell_rows = result.for_cells(["cell-0", "cell-4"])
        self.assertEqual(
            cell_rows["label"].astype(str).tolist(),
            ["T", "Myeloid"],
        )

    def test_duplicate_external_observation_ids_are_rejected(self):
        adata = make_target()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "duplicates.csv"
            pd.DataFrame(
                {
                    "obs_name": ["cell-0", "cell-0"],
                    "Atlas_maxfuse_score": [0.9, 0.8],
                    "Atlas_label": ["T", "B"],
                }
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate observation"):
                summarize_maxfuse_evidence(
                    adata,
                    "population",
                    paths=path,
                )

    def test_artifact_writer_excludes_private_cell_cache(self):
        adata = make_target()
        adata.obs["Atlas_maxfuse_score"] = [0.9] * adata.n_obs
        adata.obs["Atlas_label"] = ["T", "T", "T", "B", "B", "B"]
        result = summarize_maxfuse_evidence(
            adata,
            "population",
            score_threshold=0.85,
        )
        with TemporaryDirectory() as directory:
            writer = PopulationQCArtifactWriter(directory, stage="global")
            paths = writer.save_result_tables(result, "maxfuse")
            self.assertIn("population_summary", paths)
            self.assertNotIn("_cell_evidence", paths)
            self.assertFalse(
                any("_cell_evidence" in path.name for path in paths.values())
            )


if __name__ == "__main__":
    unittest.main()
