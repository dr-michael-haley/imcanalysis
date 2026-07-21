import json
import tempfile
import unittest
from pathlib import Path

import yaml

from SpatialBiologyToolkit.pipeline.assets import resolve_assets
from SpatialBiologyToolkit.pipeline.manifests import dump_json, dump_yaml
from SpatialBiologyToolkit.pipeline.planner import (
    build_run_plan,
    expand_requested,
    resolve_dependencies,
)
from SpatialBiologyToolkit.pipeline.project import (
    INITIAL_ASSET_INVENTORY,
    PROJECT_MARKER,
    adopt_project,
    discover_project_root,
    initialize_project,
    load_project,
    validate_project,
)
from SpatialBiologyToolkit.pipeline.registry import (
    MODE_REGISTRY,
    STAGE_REGISTRY,
    registry_aliases,
)


class ProjectAndPlanningTests(unittest.TestCase):
    def test_project_init_creates_minimal_project_and_discovers_from_nested_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            context = initialize_project(root)
            nested = root / "metadata" / "review"
            nested.mkdir()

            self.assertTrue((root / "config.yaml").is_file())
            self.assertTrue((root / "IMC_files").is_dir())
            self.assertTrue((root / "metadata").is_dir())
            self.assertTrue((root / PROJECT_MARKER).is_file())
            self.assertTrue((root / ".sbt" / "runs").is_dir())
            self.assertEqual(discover_project_root(nested), root.resolve())
            self.assertEqual(load_project(start=nested).root, context.root)

    def test_adopt_is_non_destructive_and_records_initial_inventory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "existing"
            root.mkdir()
            source = root / "config.yaml"
            source_text = "general:\n  imc_files_folder: raw_inputs\n"
            source.write_text(source_text, encoding="utf-8")
            raw = root / "raw_inputs"
            raw.mkdir()
            payload = raw / "case.mcd"
            payload.write_bytes(b"raw-data")
            extra = root / "notes.txt"
            extra.write_text("preserve me", encoding="utf-8")

            result = adopt_project(root)

            self.assertEqual(source.read_text(encoding="utf-8"), source_text)
            self.assertEqual(payload.read_bytes(), b"raw-data")
            self.assertEqual(extra.read_text(encoding="utf-8"), "preserve me")
            self.assertTrue((root / INITIAL_ASSET_INVENTORY).is_file())
            self.assertIn(extra.resolve(), result.unexpected_paths)

    def test_configured_asset_paths_resolve_relative_to_project_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            root.mkdir()
            (root / "config.yaml").write_text(
                "general:\n"
                "  imc_files_folder: inputs/imc\n"
                "  masks_folder: outputs/masks\n"
                "  anndata_path: data/cohort.h5ad\n",
                encoding="utf-8",
            )
            (root / "inputs" / "imc").mkdir(parents=True)
            result = adopt_project(root)
            assets = {
                asset.role: asset
                for asset in resolve_assets(result.context.config, root)
            }

            self.assertEqual(
                assets["raw_imc_files"].path, (root / "inputs" / "imc").resolve()
            )
            self.assertEqual(
                assets["masks"].path, (root / "outputs" / "masks").resolve()
            )
            self.assertEqual(
                assets["anndata"].path, (root / "data" / "cohort.h5ad").resolve()
            )

    def test_validation_distinguishes_generated_assets_and_stage_readiness(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            context = initialize_project(root)
            (root / "IMC_files" / "case.mcd").write_bytes(b"x")

            prep = validate_project(context, stages=[STAGE_REGISTRY["prep"]])
            cellpose = validate_project(context, stages=[STAGE_REGISTRY["cellpose"]])

            self.assertTrue(prep.valid)
            self.assertTrue(prep.stage_readiness["prep"])
            self.assertFalse(cellpose.stage_readiness["cellpose"])
            self.assertTrue(
                all(
                    item.status in {"ok", "not_created"}
                    for item in prep.generated_assets
                )
            )

    def test_asset_inventory_uses_shallow_bounded_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            context = initialize_project(root)
            nested = root / "metadata" / "large_nested_tree"
            nested.mkdir()
            for index in range(25):
                (nested / f"{index}.txt").write_text("x", encoding="utf-8")

            assets = {
                asset.role: asset for asset in resolve_assets(context.config, root)
            }

            self.assertEqual(assets["metadata"].file_count, 1)

    def test_registry_preserves_every_legacy_pipeline_alias(self):
        config_path = (
            Path(__file__).resolve().parents[1] / "SLURM_scripts" / "pipeline.conf"
        )
        legacy_aliases = [
            line.split("=", 1)[0].strip()
            for line in config_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

        self.assertEqual(registry_aliases(), legacy_aliases)
        self.assertTrue(all(stage.slurm_script for stage in STAGE_REGISTRY.values()))

    def test_mode_expansion_and_dependency_resolution_are_ordered(self):
        self.assertEqual(
            expand_requested(["segmentation"]),
            MODE_REGISTRY["segmentation"].stages,
        )
        self.assertEqual(
            [stage.name for stage in resolve_dependencies(["cellpose"])],
            ["prep", "denoise", "cellpose"],
        )
        with self.assertRaisesRegex(KeyError, "Unknown stage"):
            expand_requested(["cellpsoe"])

    def test_plan_contains_assets_and_serializes_to_yaml_and_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            context = initialize_project(root)
            (root / "IMC_files" / "case.mcd").write_bytes(b"x")
            plan = build_run_plan(context, ["segmentation"])

            self.assertTrue(plan.ready, plan.errors)
            self.assertEqual(
                [stage.name for stage in plan.resolved_stages],
                ["prep", "denoise", "dnqc", "cellpose", "nimbus"],
            )
            self.assertIn("raw_imc_files", plan.resolved_stages[0].requires_assets)
            self.assertIn("anndata", plan.resolved_stages[-1].produces_assets)
            yaml_data = yaml.safe_load(dump_yaml(plan))
            json_data = json.loads(dump_json(plan))
            self.assertEqual(yaml_data["execution_backend"], "slurm_scripts")
            self.assertEqual(json_data["resolved_stages"][0]["name"], "prep")

    def test_plan_can_skip_dependencies_but_requires_existing_stage_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            context = initialize_project(root)
            assets = {asset.role: asset for asset in resolve_assets(context.config, root)}

            missing = build_run_plan(
                context,
                ["cellvision-full"],
                include_dependencies=False,
            )
            self.assertFalse(missing.ready)
            self.assertEqual(
                [stage.name for stage in missing.resolved_stages],
                ["cellvision-full"],
            )
            self.assertEqual(missing.resolved_stages[0].depends_on, [])
            self.assertIn("anndata", missing.resolved_stages[0].missing_assets)
            self.assertIn("denoised_images", missing.resolved_stages[0].missing_assets)
            self.assertIn("masks", missing.resolved_stages[0].missing_assets)

            assets["anndata"].path.parent.mkdir(parents=True, exist_ok=True)
            assets["anndata"].path.write_bytes(b"placeholder")
            for role in ("denoised_images", "masks"):
                assets[role].path.mkdir(parents=True, exist_ok=True)
                (assets[role].path / "placeholder.tif").write_bytes(b"placeholder")

            ready = build_run_plan(
                context,
                ["cellvision-full"],
                include_dependencies=False,
            )
            self.assertTrue(ready.ready, ready.errors)
            self.assertEqual(
                [stage.name for stage in ready.resolved_stages],
                ["cellvision-full"],
            )
            self.assertTrue(
                any("Dependency expansion is disabled" in item for item in ready.warnings)
            )


if __name__ == "__main__":
    unittest.main()
