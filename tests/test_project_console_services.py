from __future__ import annotations

import ast
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from SpatialBiologyToolkit.config.editing import (
    ConfigChangedExternallyError,
    ConfigEditorSession,
    ConfigRecoverySession,
    InvalidConfigEditError,
)
from SpatialBiologyToolkit.config.models import DEFAULT_CONFIG_CLASSES
from SpatialBiologyToolkit.pipeline.inspection import (
    MAX_TEXT_BYTES,
    context_with_config,
    inspect_execution,
    inspect_project,
    inspect_readiness,
    open_project_console,
    read_text_bounded,
    stage_documentation,
)
from SpatialBiologyToolkit.pipeline.executions import (
    execution_output_path,
    load_execution_index,
    preview_executions,
    write_execution_index,
)
from SpatialBiologyToolkit.pipeline.manifests import utc_now, write_text, write_yaml
from SpatialBiologyToolkit.pipeline.models import RunManifest, RunStatus
from SpatialBiologyToolkit.pipeline.assets import asset_spec_map, resolve_assets
from SpatialBiologyToolkit.pipeline.notes import (
    ProjectNotesChangedError,
    ProjectNotesSession,
)
from SpatialBiologyToolkit.pipeline.project import initialize_project
from SpatialBiologyToolkit.pipeline.registry import get_stage
from SpatialBiologyToolkit.pipeline.runs import (
    RESOLVED_CONFIG,
    RUN_MANIFEST,
    STATUS_FILE,
)


class ProjectConsoleServiceTests(unittest.TestCase):
    def _project(self, directory: str):
        return initialize_project(Path(directory) / "project")

    def test_config_edit_preserves_unknowns_and_writes_backup_and_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = self._project(temporary)
            source = yaml.safe_load(context.config_path.read_text(encoding="utf-8"))
            source["legacy_plugin"] = {"keep": 42}
            context.config_path.write_text(
                yaml.safe_dump(source, sort_keys=False), encoding="utf-8"
            )
            session = ConfigEditorSession.open(context.config_path)
            for spec in session.field_specs():
                if spec.kind == "yaml":
                    yaml.safe_dump(spec.value)

            session.set_value("general.outputs_folder", "review_outputs")
            result = session.save(context.root)

            written = yaml.safe_load(context.config_path.read_text(encoding="utf-8"))
            self.assertEqual(written["general"]["outputs_folder"], "review_outputs")
            self.assertEqual(written["legacy_plugin"], {"keep": 42})
            self.assertTrue(result.backup_path.is_file())
            self.assertTrue(result.audit_path.is_file())
            audit = yaml.safe_load(result.audit_path.read_text(encoding="utf-8"))
            self.assertTrue(audit["edited_by"])
            self.assertEqual(audit["changed_paths"], ["general.outputs_folder"])
            self.assertFalse(session.dirty)

    def test_config_field_specs_distinguish_disk_defaults_and_unsaved_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = self._project(temporary)
            context.config_path.write_text(
                "general:\n  outputs_folder: review_outputs\n",
                encoding="utf-8",
            )
            session = ConfigEditorSession.open(context.config_path)

            stored = next(spec for spec in session.field_specs() if spec.stored)
            inherited = next(spec for spec in session.field_specs() if not spec.stored)
            self.assertTrue(stored.explicit)
            self.assertFalse(stored.staged)
            self.assertFalse(inherited.explicit)
            self.assertFalse(inherited.staged)

            session.set_value(inherited.path, inherited.value)
            proposed = next(
                spec for spec in session.field_specs() if spec.path == inherited.path
            )
            self.assertFalse(proposed.stored)
            self.assertTrue(proposed.explicit)
            self.assertTrue(proposed.staged)

            session.reset_to_default(inherited.path)
            session.reset_to_default(stored.path)
            pending_reset = next(
                spec for spec in session.field_specs() if spec.path == stored.path
            )
            self.assertTrue(pending_reset.stored)
            self.assertFalse(pending_reset.explicit)
            self.assertTrue(pending_reset.staged)
            self.assertTrue(pending_reset.pending_removal)

    def test_config_edit_detects_external_change_and_reset_removes_explicit_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = self._project(temporary)
            context.config_path.write_text(
                "general:\n  outputs_folder: outputs\n",
                encoding="utf-8",
            )
            session = ConfigEditorSession.open(context.config_path)
            session.set_value("general.outputs_folder", "elsewhere")
            context.config_path.write_text(
                context.config_path.read_text(encoding="utf-8") + "\n# external\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigChangedExternallyError):
                session.save(context.root)

            session = ConfigEditorSession.open(context.config_path)
            self.assertTrue(
                any(
                    item.path == "general.outputs_folder"
                    for item in session.field_specs()
                )
            )
            session.reset_to_default("general.outputs_folder")
            session.save(context.root)
            written = yaml.safe_load(context.config_path.read_text(encoding="utf-8"))
            self.assertNotIn("outputs_folder", written.get("general", {}))

    def test_invalid_config_can_be_repaired_in_recovery_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = self._project(temporary)
            context.config_path.write_text("general: [broken\n", encoding="utf-8")

            opened = open_project_console(context.root)

            self.assertTrue(opened.recovery_mode)
            self.assertIsInstance(opened.recovery, ConfigRecoverySession)
            with self.assertRaises(InvalidConfigEditError):
                opened.recovery.validate_text("general: [still broken")
            opened.recovery.save_text(
                "general:\n  outputs_folder: restored_outputs\n", context.root
            )
            reopened = open_project_console(context.root)
            self.assertFalse(reopened.recovery_mode)
            self.assertEqual(
                reopened.context.config.general.outputs_folder, "restored_outputs"
            )

    def test_project_inspection_and_readiness_never_query_scheduler(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = self._project(temporary)
            (context.root / "IMC_files" / "sample.mcd").write_bytes(b"x")
            with patch(
                "subprocess.run",
                side_effect=AssertionError("scheduler/subprocess access is forbidden"),
            ):
                snapshot = inspect_project(context)
                plan = inspect_readiness(context, ["prep"])

            self.assertEqual(snapshot.context.root, context.root)
            self.assertTrue(
                any(item.asset.role == "raw_imc_files" for item in snapshot.assets)
            )
            self.assertTrue(plan.ready, plan.errors)
            proposed = context_with_config(context, snapshot.context.config)
            self.assertEqual(proposed.root, context.root)

    def test_execution_inspection_reads_durable_records_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = self._project(temporary)
            workflow_run_id = "run-gui-inspection"
            record = preview_executions(
                context,
                ["prep"],
                workflow_run_id=workflow_run_id,
            )[0]
            index = load_execution_index(context)
            write_execution_index(
                context,
                index.model_copy(update={"executions": [record]}),
            )
            run_dir = context.runs_dir / workflow_run_id
            resolved_config = run_dir / RESOLVED_CONFIG
            write_yaml(
                run_dir / RUN_MANIFEST,
                RunManifest(
                    run_id=workflow_run_id,
                    workflow_run_id=workflow_run_id,
                    project_id=context.project_metadata.project_id,
                    project_root=context.root,
                    created_at=utc_now(),
                    requested_stages=["prep"],
                    resolved_stages=["prep"],
                    config_source=context.config_path,
                    resolved_config=resolved_config,
                    execution_backend="slurm_scripts",
                    working_directory=context.root,
                    command="sbt run prep",
                ),
            )
            write_yaml(
                run_dir / STATUS_FILE,
                RunStatus(
                    run_id=workflow_run_id,
                    workflow_run_id=workflow_run_id,
                    project_id=context.project_metadata.project_id,
                    checked_at=utc_now(),
                    overall_status="recorded-only",
                    stages=[],
                ),
            )
            write_text(resolved_config, "general:\n  outputs_folder: outputs\n")
            output = execution_output_path(context, record)
            write_text(output / "README.md", "# Recorded execution\n")

            with patch(
                "subprocess.run",
                side_effect=AssertionError("scheduler/subprocess access is forbidden"),
            ):
                detail = inspect_execution(context, record.technical_run_id)

            self.assertEqual(detail.execution.technical_run_id, record.technical_run_id)
            self.assertEqual(detail.recorded_status.overall_status, "recorded-only")
            self.assertIn("Recorded execution", detail.report_text)
            self.assertIn("outputs_folder", detail.resolved_config_text)

    def test_stage_documentation_and_notes_are_bounded_project_services(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = self._project(temporary)
            self.assertIn("Preprocess", stage_documentation(get_stage("prep")))
            notes = ProjectNotesSession.open(context)
            backup = notes.save("# Project notes\n\nReviewed.\n", context.root)
            self.assertTrue(backup.is_file())
            self.assertIn("Reviewed", notes.path.read_text(encoding="utf-8"))

            stale = ProjectNotesSession.open(context)
            notes.save("# Project notes\n\nSecond edit.\n", context.root)
            with self.assertRaises(ProjectNotesChangedError):
                stale.save("stale", context.root)

    def test_new_notes_file_can_be_saved_and_then_detects_concurrent_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = self._project(temporary)
            notes_path = context.root / context.project_metadata.notes_file
            notes_path.unlink(missing_ok=True)
            first = ProjectNotesSession.open(context)
            stale = ProjectNotesSession.open(context)

            backup = first.save("# First notes\n", context.root)

            self.assertEqual(backup.read_text(encoding="utf-8"), "")
            with self.assertRaises(ProjectNotesChangedError):
                stale.save("# Stale notes\n", context.root)

    def test_asset_catalogue_covers_every_resolved_role_and_text_reads_are_bounded(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            context = self._project(temporary)
            roles = {
                asset.role for asset in resolve_assets(context.config, context.root)
            }
            self.assertEqual(roles, set(asset_spec_map()))

            text_path = context.root / "large.txt"
            text_path.write_text("x" * 200, encoding="utf-8")
            rendered = read_text_bounded(text_path, max_bytes=32)
            self.assertEqual(rendered.count("x"), 32)
            self.assertIn("Display truncated", rendered)
            self.assertGreater(MAX_TEXT_BYTES, 32)

            raw = context.root / context.config.general.imc_files_folder
            (raw / "first.mcd").write_bytes(b"1")
            (raw / "second.mcd").write_bytes(b"2")
            bounded_assets = resolve_assets(context.config, context.root, count_limit=1)
            self.assertTrue(
                all((asset.file_count or 0) <= 1 for asset in bounded_assets)
            )
            raw_asset = next(
                item for item in bounded_assets if item.role == "raw_imc_files"
            )
            self.assertTrue(raw_asset.count_limited)

    def test_field_catalogue_matches_every_typed_config_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = self._project(temporary)
            fields = ConfigEditorSession.open(context.config_path).field_specs()
            expected = sum(
                len(model.model_fields) for model in DEFAULT_CONFIG_CLASSES.values()
            )

            self.assertEqual(len(fields), expected)
            self.assertEqual(len({field.path for field in fields}), expected)

    def test_gui_package_has_no_scheduler_scientific_or_shell_imports(self):
        package = Path(__file__).parents[1] / "SpatialBiologyToolkit" / "project_gui"
        forbidden = {
            "anndata",
            "napari",
            "numpy",
            "pandas",
            "scanpy",
            "subprocess",
            "SpatialBiologyToolkit.pipeline.slurm",
            "SpatialBiologyToolkit.pipeline.status",
        }
        imported: set[str] = set()
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
        violations = sorted(
            module
            for module in imported
            if any(
                module == item or module.startswith(f"{item}.") for item in forbidden
            )
        )
        self.assertEqual(violations, [])

    def test_gui_environment_is_isolated_from_the_scientific_stack(self):
        repository = Path(__file__).parents[1]
        environment = yaml.safe_load(
            (repository / "Local_envs" / "sbt_gui_env.yml").read_text(encoding="utf-8")
        )
        dependencies = {
            re.split(r"[<>=!~ ]", str(item), maxsplit=1)[0].casefold()
            for item in environment["dependencies"]
            if isinstance(item, str)
        }

        self.assertIn("pyside6", dependencies)
        self.assertIn("ruamel.yaml", dependencies)
        self.assertTrue(
            dependencies.isdisjoint(
                {
                    "anndata",
                    "cellpose",
                    "napari",
                    "numpy",
                    "pandas",
                    "scanpy",
                    "scipy",
                    "spatialdata",
                    "tensorflow",
                    "torch",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
