from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from SpatialBiologyToolkit.scripts import hyperstac_full
from SpatialBiologyToolkit.config.models import (
    CoxFeatureSourceConfig,
    HyperstacConfig,
    PipelineConfig,
)
from SpatialBiologyToolkit.cox_survival import (
    CoxFeatureSource,
    build_multi_source_case_table,
)
from SpatialBiologyToolkit.environments.registry import load_environment_registry
from SpatialBiologyToolkit.pipeline.assets import resolve_assets
from SpatialBiologyToolkit.pipeline.executions import (
    execution_output_path,
    preview_executions,
    write_execution_index,
)
from SpatialBiologyToolkit.pipeline.models import ExecutionIndex
from SpatialBiologyToolkit.pipeline.planner import build_run_plan
from SpatialBiologyToolkit.pipeline.manifests import utc_now
from SpatialBiologyToolkit.pipeline.project import initialize_project
from SpatialBiologyToolkit.pipeline.registry import MODE_REGISTRY, STAGE_REGISTRY


class HyperstacConfigTests(unittest.TestCase):
    def test_defaults_preserve_imc_adaptation_without_dataset_markers(self):
        config = HyperstacConfig()
        self.assertEqual(config.patch_size, 100)
        self.assertEqual(config.epochs, 50)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.channels, [])
        self.assertEqual(config.leiden_resolutions, [0.2, 0.25, 0.3, 0.35])
        self.assertFalse(config.write_spatial_cluster_maps)
        self.assertTrue(config.full_include_survival)

    def test_pipeline_registers_hyperstac_and_cox_sections(self):
        config = PipelineConfig()
        self.assertEqual(config.hyperstac.asset_folder, "hyperstac")
        self.assertEqual(config.cox.models, ["coxph", "ridge", "coxnet"])
        self.assertEqual(config.cox.duration_col, "duration")
        self.assertEqual(config.cox.covariate_cols, [])

    def test_feature_source_needs_a_feature_definition(self):
        with self.assertRaisesRegex(ValueError, "needs population_obs"):
            CoxFeatureSourceConfig(name="empty", adata_path="empty.h5ad")


class HyperstacFullTests(unittest.TestCase):
    def test_image_only_full_run_skips_survival_components(self):
        config = SimpleNamespace(
            hyperstac=SimpleNamespace(full_include_survival=False)
        )
        with patch.object(hyperstac_full, "load_runtime", return_value=config), patch.object(
            hyperstac_full, "run_preprocess"
        ) as preprocess, patch.object(
            hyperstac_full, "run_model"
        ) as model, patch.object(
            hyperstac_full, "run_permutation"
        ) as permutation, patch.object(
            hyperstac_full, "run_visualisation"
        ) as visualisation, patch.object(
            hyperstac_full.cox_survival, "main"
        ) as cox, patch.object(
            hyperstac_full, "run_stability"
        ) as stability:
            hyperstac_full.main()

        preprocess.assert_called_once_with()
        model.assert_called_once_with()
        permutation.assert_called_once_with()
        visualisation.assert_called_once_with()
        cox.assert_not_called()
        stability.assert_not_called()

    def test_full_run_includes_survival_components_when_enabled(self):
        config = SimpleNamespace(
            hyperstac=SimpleNamespace(full_include_survival=True)
        )
        with patch.object(hyperstac_full, "load_runtime", return_value=config), patch.object(
            hyperstac_full, "run_preprocess"
        ), patch.object(hyperstac_full, "run_model"), patch.object(
            hyperstac_full, "run_permutation"
        ), patch.object(hyperstac_full, "run_visualisation"), patch.object(
            hyperstac_full.cox_survival, "main"
        ) as cox, patch.object(hyperstac_full, "run_stability") as stability:
            hyperstac_full.main()

        cox.assert_called_once_with()
        stability.assert_called_once_with()


class CoxMultiSourceTests(unittest.TestCase):
    def setUp(self):
        self.clinical = pd.DataFrame(
            {
                "Case": ["A", "A", "B", "B"],
                "ROI": ["A1", "A2", "B1", "B2"],
                "time": [10, 10, 20, 20],
                "event": [1, 1, 0, 0],
                "age": [50, 50, 60, 60],
                "sex": ["F", "F", "M", "M"],
            }
        )

    def test_combines_case_features_and_maps_patch_rois(self):
        cell_obs = pd.DataFrame(
            {
                "Case": ["A", "A", "A", "B", "B", "B"],
                "ROI": ["A1", "A1", "A2", "B1", "B2", "B2"],
                "population": ["x", "x", "y", "x", "y", "y"],
            }
        )
        patch_obs = pd.DataFrame(
            {
                "roi": ["A1", "A2", "B1", "B2"],
                "leiden_0.3": ["0", "1", "0", "2"],
            }
        )
        table = build_multi_source_case_table(
            [
                CoxFeatureSource(
                    name="nimbus",
                    obs=cell_obs,
                    population_obs=("population",),
                    case_obs="Case",
                    roi_obs="ROI",
                ),
                CoxFeatureSource(
                    name="hyperstac",
                    obs=patch_obs,
                    population_obs=("leiden_0.3",),
                    case_obs=None,
                    roi_obs="roi",
                ),
            ],
            self.clinical,
            clinical_case_col="Case",
            clinical_duration_col="time",
            clinical_event_col="event",
            clinical_roi_col="ROI",
            covariate_cols=["age", "sex"],
        )
        self.assertEqual(list(table.index), ["A", "B"])
        self.assertIn("nimbus", table.attrs["source_features"])
        self.assertIn("hyperstac", table.attrs["source_features"])
        self.assertTrue(
            all(
                feature.startswith(("nimbus__", "hyperstac__"))
                for feature in table.attrs["image_features"]
            )
        )
        self.assertTrue(
            all(
                feature.startswith("clinical__")
                for feature in table.attrs["clinical_features"]
            )
        )
        self.assertEqual(table["event"].tolist(), [True, False])

    def test_rejects_outcome_disagreement_between_sources(self):
        first = pd.DataFrame(
            {
                "Case": ["A", "B"],
                "ROI": ["A1", "B1"],
                "population": ["x", "y"],
            }
        )
        bad_clinical = self.clinical.copy()
        bad_clinical.loc[bad_clinical["Case"] == "A", "time"] = [10, 11]
        with self.assertRaisesRegex(ValueError, "multiple values"):
            build_multi_source_case_table(
                [
                    CoxFeatureSource(
                        name="source",
                        obs=first,
                        population_obs=("population",),
                        case_obs="Case",
                        roi_obs="ROI",
                    )
                ],
                bad_clinical,
                clinical_case_col="Case",
                clinical_duration_col="time",
                clinical_event_col="event",
                clinical_roi_col="ROI",
                metadata_conflict="error",
            )

    def test_direct_case_sources_accept_case_level_clinical_metadata(self):
        source_obs = pd.DataFrame(
            {
                "Case": ["A", "A", "B", "B"],
                "population": ["x", "y", "x", "x"],
            }
        )
        clinical = pd.DataFrame(
            {
                "Case": ["A", "B"],
                "duration": [10, 20],
                "event": [1, 0],
            }
        )
        table = build_multi_source_case_table(
            [
                CoxFeatureSource(
                    name="source",
                    obs=source_obs,
                    population_obs=("population",),
                    case_obs="Case",
                    roi_obs=None,
                )
            ],
            clinical,
            clinical_case_col="Case",
            clinical_duration_col="duration",
            clinical_event_col="event",
            clinical_roi_col="ROI",
        )
        self.assertEqual(list(table.index), ["A", "B"])


class HyperstacRegistryTests(unittest.TestCase):
    def test_atomic_and_composite_stages_are_registered(self):
        expected = {
            "hyperstac-preprocess",
            "hyperstac-model",
            "hyperstac-permutation",
            "hyperstac-visualise",
            "cox",
            "hyperstac-stability",
            "hyperstac-full",
        }
        self.assertTrue(expected.issubset(STAGE_REGISTRY))
        self.assertEqual(
            STAGE_REGISTRY["hyperstac-stability"].depends_on,
            ["hyperstac-visualise", "cox"],
        )
        self.assertEqual(
            set(STAGE_REGISTRY["hyperstac-stability"].required_executions),
            {"hyperstac-visualise", "cox"},
        )
        self.assertEqual(
            MODE_REGISTRY["hyperstac"].stages[-2:],
            ["cox", "hyperstac-stability"],
        )

    def test_all_stages_use_registered_hyperstac_environment(self):
        registry = load_environment_registry()
        for stage in (
            "hyperstac-preprocess",
            "hyperstac-model",
            "hyperstac-permutation",
            "hyperstac-visualise",
            "cox",
            "hyperstac-stability",
            "hyperstac-full",
        ):
            self.assertEqual(registry.stage_environments[stage], ["hyperstac"])

    def test_wrappers_and_docs_exist(self):
        root = Path(__file__).resolve().parents[1]
        for stage in (
            "hyperstac-preprocess",
            "hyperstac-model",
            "hyperstac-permutation",
            "hyperstac-visualise",
            "cox",
            "hyperstac-stability",
            "hyperstac-full",
        ):
            spec = STAGE_REGISTRY[stage]
            self.assertTrue((root / spec.slurm_script).is_file())
            self.assertTrue((root / spec.documentation_path).is_file())

    def test_stability_requires_direct_managed_reports_not_complete_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = initialize_project(Path(temporary) / "project")
            assets = {
                asset.role: asset
                for asset in resolve_assets(context.config, context.root)
            }
            assets["hyperstac_assets"].path.mkdir(parents=True, exist_ok=True)
            for name in (
                "imc_hyperstac_representations.h5ad",
                "imc_hyperstac_patch_metrics.h5ad",
            ):
                (assets["hyperstac_assets"].path / name).write_bytes(b"x")

            explicit = build_run_plan(
                context,
                ["hyperstac-stability"],
                dependency_policy="none",
            )
            asset_aware = build_run_plan(context, ["hyperstac-stability"])

            self.assertFalse(explicit.ready)
            self.assertEqual(
                explicit.resolved_stages[0].missing_executions,
                ["hyperstac-visualise", "cox"],
            )
            self.assertTrue(asset_aware.ready, asset_aware.errors)
            self.assertEqual(
                [stage.name for stage in asset_aware.resolved_stages],
                ["hyperstac-visualise", "cox", "hyperstac-stability"],
            )

            records = preview_executions(
                context,
                ["hyperstac-visualise", "cox"],
                workflow_run_id="prior-managed-reports",
            )
            write_execution_index(
                context,
                ExecutionIndex(
                    project_id=context.project_metadata.project_id,
                    updated_at=utc_now(),
                    executions=records,
                ),
            )
            required_folders = (
                (records[0], "files/hyperstac_visualisation"),
                (records[1], "files/cox"),
            )
            for record, relative in required_folders:
                (execution_output_path(context, record) / relative).mkdir(
                    parents=True,
                    exist_ok=True,
                )

            reuse = build_run_plan(context, ["hyperstac-stability"])

            self.assertTrue(reuse.ready, reuse.errors)
            self.assertEqual(
                [stage.name for stage in reuse.resolved_stages],
                ["hyperstac-stability"],
            )

    def test_hyperstac_docs_reference_only_the_preprint_manuscript(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "docs/source/stages/hyperstac.md").read_text(encoding="utf-8")
        self.assertIn("2025.10.16.682563v1", text)
        self.assertNotIn("resubmission", text.lower())
        self.assertNotIn("hyperstac-resubmission.pdf", text)


class LightweightImportTests(unittest.TestCase):
    def test_stage_adapters_do_not_import_tensorflow(self):
        import sys

        import SpatialBiologyToolkit.scripts._hyperstac_common  # noqa: F401
        import SpatialBiologyToolkit.scripts.cox_survival  # noqa: F401
        import SpatialBiologyToolkit.scripts.hyperstac_full  # noqa: F401

        self.assertNotIn("tensorflow", sys.modules)


if __name__ == "__main__":
    unittest.main()
