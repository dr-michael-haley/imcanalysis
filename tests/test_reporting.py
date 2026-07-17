import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from SpatialBiologyToolkit.cli.main import app
from SpatialBiologyToolkit.pipeline.executions import load_execution_index
from SpatialBiologyToolkit.pipeline.manifests import read_model, read_yaml
from SpatialBiologyToolkit.pipeline.models import ProjectMetadata
from SpatialBiologyToolkit.pipeline.planner import build_run_plan
from SpatialBiologyToolkit.pipeline.project import initialize_project, validate_project
from SpatialBiologyToolkit.pipeline.registry import STAGES
from SpatialBiologyToolkit.pipeline.runs import (
    RUN_MANIFEST,
    STAGE_EVENTS_DIRECTORY,
    create_run_record,
)
from SpatialBiologyToolkit.pipeline.slurm import (
    expected_log_paths,
    sbt_environment,
)
from SpatialBiologyToolkit.reporting import StageReporter
from SpatialBiologyToolkit.reporting.events import (
    finalize_shell_stage,
    start_shell_stage,
)
from SpatialBiologyToolkit.reporting.models import StageManifest
from SpatialBiologyToolkit.reporting.paths import (
    project_asset_path,
    resolve_reporting_context,
)
from SpatialBiologyToolkit.scripts.config_and_utils import (
    apply_reporting_output_routing,
)


class ReportingTests(unittest.TestCase):
    def test_reusable_asset_paths_resolve_from_project_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            with patch.dict(os.environ, {"SBT_PROJECT_ROOT": str(root)}):
                self.assertEqual(
                    project_asset_path("models/example.pt"),
                    root / "models" / "example.pt",
                )

    def _managed_stage(self, temp_dir: str, stage: str = "prep"):
        context = initialize_project(Path(temp_dir) / "project")
        (context.root / "IMC_files" / "case.mcd").write_bytes(b"x")
        plan = build_run_plan(context, [stage])
        self.assertTrue(plan.ready, plan.errors)
        run = create_run_record(
            context,
            plan,
            command=f"sbt run {stage}",
            reason="Focused reporting test.",
            notes=["Review the generated summary."],
        )
        return context, run, sbt_environment(context, run, stage)

    def test_registry_has_unique_unnumbered_reporting_metadata_and_docs(self):
        root = Path(__file__).resolve().parents[1]
        output_slugs = [stage.output_slug for stage in STAGES]
        orders = [stage.catalogue_order for stage in STAGES]

        self.assertEqual(len(output_slugs), len(set(output_slugs)))
        self.assertEqual(len(orders), len(set(orders)))
        self.assertEqual(sorted(orders), list(range(1, len(STAGES) + 1)))
        for stage in STAGES:
            self.assertTrue(stage.display_name)
            self.assertNotRegex(stage.output_slug, r"^\d{3}_")
            documentation = root / stage.documentation_path
            self.assertTrue(documentation.is_file(), documentation)
            content = documentation.read_text(encoding="utf-8")
            for heading in (
                "## What this stage does",
                "## Why it is performed",
                "## Main inputs",
                "## Reusable assets produced",
                "## Human-facing outputs produced",
                "## Important configuration options",
                "## How to interpret the results",
                "## Common problems and limitations",
            ):
                self.assertIn(heading, content)

    def test_every_python_stage_module_is_registered_for_shared_bootstrap(self):
        modules = [
            module
            for stage in STAGES
            for module in stage.python_modules
        ]
        self.assertEqual(len(modules), len(set(modules)))
        self.assertIn("SpatialBiologyToolkit.scripts.preprocess", modules)
        self.assertIn("SpatialBiologyToolkit.scripts.check_panel_consistency", modules)
        for stage in STAGES:
            if stage.name not in {"zipqc", "scport", "debug"}:
                self.assertTrue(stage.python_modules, stage.name)

    def test_every_registered_wrapper_sources_shared_reporting_job_hygiene(self):
        root = Path(__file__).resolve().parents[1]
        for stage in STAGES:
            wrapper = root / stage.slurm_script
            content = wrapper.read_text(encoding="utf-8")
            self.assertIn("SLURM_scripts/job_env.sh", content, stage.name)

    def test_every_stage_resolves_its_registered_run_output_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            for stage in STAGES:
                context = resolve_reporting_context(
                    environment={
                        "SBT_PROJECT_ROOT": str(root),
                        "SBT_PROJECT_ID": "project-id",
                        "SBT_EXECUTION_ID": "7",
                        "SBT_EXECUTION_LABEL": "007",
                        "SBT_TECHNICAL_RUN_ID": "stage-id",
                        "SBT_WORKFLOW_RUN_ID": "workflow-id",
                        "SBT_RUN_ID": "run-id",
                        "SBT_RUN_DIR": str(root / ".sbt" / "runs" / "workflow-id"),
                        "SBT_STAGE": stage.name,
                        "SBT_OUTPUTS_ROOT": str(root / "outputs"),
                        "SBT_STAGE_OUTPUT_DIR": str(
                            root / "outputs" / f"007_{stage.output_slug}"
                        ),
                    }
                )
                self.assertEqual(
                    context.stage_run_dir,
                    (root / "outputs" / f"007_{stage.output_slug}").resolve(),
                    stage.name,
                )

    def test_managed_reporter_writes_manifest_readmes_indexes_and_stage_event(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, run, environment = self._managed_stage(temp_dir)
            output_dir = Path(environment["SBT_STAGE_OUTPUT_DIR"])
            with patch.dict(os.environ, environment, clear=False):
                with StageReporter.from_environment() as report:
                    table = report.context.tables_dir / "summary.csv"
                    table.write_text("metric,value\ncells,10\n", encoding="utf-8")
                    report.add_metric("rois_processed", 2)

            manifest = read_model(output_dir / "stage_manifest.yaml", StageManifest)
            self.assertEqual(manifest.status, "completed")
            self.assertEqual(manifest.reason, "Focused reporting test.")
            self.assertEqual(manifest.metrics["rois_processed"], 2)
            self.assertEqual(manifest.metrics["tables"], 1)
            self.assertTrue((output_dir / "README.md").is_file())
            self.assertTrue((context.root / "outputs" / "README.md").is_file())
            technical_id = run.execution_for_stage("prep").technical_run_id
            self.assertTrue(
                (
                    run.run_dir
                    / STAGE_EVENTS_DIRECTORY
                    / f"{technical_id}.yaml"
                ).is_file()
            )
            readme = (output_dir / "README.md").read_text(encoding="utf-8")
            self.assertIn("Focused reporting test.", readme)
            self.assertIn("Technical workflow directory", readme)
            self.assertIn("## How to interpret these outputs", readme)

    def test_reporter_records_failure_and_reraises_scientific_exception(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _context, _run, environment = self._managed_stage(temp_dir)
            output_dir = Path(environment["SBT_STAGE_OUTPUT_DIR"])
            with patch.dict(os.environ, environment, clear=False):
                with self.assertRaisesRegex(RuntimeError, "scientific failure"):
                    with StageReporter.from_environment():
                        raise RuntimeError("scientific failure")

            manifest = read_model(output_dir / "stage_manifest.yaml", StageManifest)
            self.assertEqual(manifest.status, "failed")
            self.assertEqual(manifest.errors[-1].type, "RuntimeError")
            self.assertIn("scientific failure", manifest.errors[-1].message)

    def test_shell_stage_hooks_cover_external_registered_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, run, environment = self._managed_stage(temp_dir, "debug")
            output_dir = Path(environment["SBT_STAGE_OUTPUT_DIR"])
            with patch.dict(os.environ, environment, clear=False):
                start_shell_stage()
                running = read_model(
                    output_dir / "stage_manifest.yaml",
                    StageManifest,
                )
                self.assertEqual(running.status, "running")
                finalize_shell_stage(0)

            completed = read_model(
                output_dir / "stage_manifest.yaml",
                StageManifest,
            )
            self.assertEqual(completed.status, "completed")
            technical_id = run.execution_for_stage("debug").technical_run_id
            self.assertTrue(
                (run.run_dir / STAGE_EVENTS_DIRECTORY / f"{technical_id}.yaml").is_file()
            )
            self.assertTrue((context.root / "outputs" / "README.md").is_file())

    def test_direct_execution_fallback_creates_report_without_technical_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.yaml").write_text("{}\n", encoding="utf-8")
            clean_environment = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("SBT_")
            }
            with patch.dict(os.environ, clean_environment, clear=True):
                with patch("pathlib.Path.cwd", return_value=root):
                    with StageReporter.from_environment("prep") as report:
                        direct_dir = report.context.stage_run_dir

            manifest = read_model(
                direct_dir / "stage_manifest.yaml",
                StageManifest,
            )
            self.assertFalse(manifest.managed_run)
            self.assertIsNone(manifest.technical_run_record)
            self.assertTrue(any("Direct execution" in item for item in manifest.warnings))

    def test_bootstrapped_direct_nonzero_exit_is_recorded_as_failed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.yaml").write_text("{}\n", encoding="utf-8")
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("SBT_")
            }
            repository = str(Path(__file__).resolve().parents[1])
            environment["PYTHONPATH"] = os.pathsep.join(
                item
                for item in (repository, environment.get("PYTHONPATH", ""))
                if item
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from SpatialBiologyToolkit.reporting import "
                        "bootstrap_stage_reporting; "
                        "bootstrap_stage_reporting('prep'); exit(3)"
                    ),
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 3)
            manifests = list(
                (root / "outputs" / "direct").glob(
                    "direct-*_Preprocessing/stage_manifest.yaml"
                )
            )
            self.assertEqual(len(manifests), 1)
            manifest = read_model(manifests[0], StageManifest)
            self.assertEqual(manifest.status, "failed")
            self.assertIn("status 3", manifest.errors[-1].message)

    def test_reporting_route_is_runtime_only_and_preserves_legacy_without_context(self):
        original = {"general": {"qc_folder": "QC", "outputs_folder": "outputs"}}
        untouched = apply_reporting_output_routing(
            {"general": dict(original["general"])}
        )
        self.assertEqual(untouched["general"]["qc_folder"], "QC")

        with tempfile.TemporaryDirectory() as temp_dir:
            stage_dir = Path(temp_dir) / "outputs" / "001_Preprocessing" / "run"
            with patch.dict(
                os.environ,
                {
                    "SBT_STAGE": "prep",
                    "SBT_STAGE_OUTPUT_DIR": str(stage_dir),
                },
                clear=False,
            ):
                routed = apply_reporting_output_routing(
                    {"general": dict(original["general"])}
                )
            self.assertEqual(
                routed["general"]["qc_folder"],
                str(stage_dir.resolve()),
            )
        self.assertEqual(original["general"]["qc_folder"], "QC")

    def test_project_initialization_creates_empty_execution_navigation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = initialize_project(Path(temp_dir) / "project")
            outputs = context.root / "outputs"

            self.assertTrue((outputs / "README.md").is_file())
            self.assertEqual(
                [item.name for item in outputs.iterdir() if item.is_dir()],
                [],
            )
            self.assertEqual(load_execution_index(context).executions, [])
            self.assertTrue((context.root / ".sbt" / "project_notes.md").is_file())

    def test_run_reason_notes_environment_and_log_paths_are_recorded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _context, run, environment = self._managed_stage(temp_dir)
            manifest = read_yaml(run.run_dir / RUN_MANIFEST)

            self.assertEqual(manifest["reason"], "Focused reporting test.")
            self.assertEqual(manifest["notes"], ["Review the generated summary."])
            self.assertEqual(environment["SBT_RUN_REASON"], manifest["reason"])
            self.assertIn("SBT_STAGE_OUTPUT_DIR", environment)
            stdout, stderr = expected_log_paths(run.run_dir, "prep", "123")
            self.assertEqual(stdout.name, "prep_123.out")
            self.assertEqual(stderr.name, "prep_123.err")

    def test_validation_identifies_preserved_legacy_qc(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = initialize_project(Path(temp_dir) / "project")
            (context.root / "QC").mkdir()

            report = validate_project(context)

            legacy = [
                item
                for item in report.reporting_outputs
                if item.name == "legacy QC folder"
            ]
            self.assertEqual(len(legacy), 1)
            self.assertEqual(legacy[0].status, "warning")

    def test_cli_explain_uses_shared_document_and_project_notes_can_be_appended(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = initialize_project(Path(temp_dir) / "project")
            runner = CliRunner()

            explain = runner.invoke(app, ["stages", "explain", "prep"])
            self.assertEqual(explain.exit_code, 0, explain.stdout)
            self.assertIn("## What this stage does", explain.stdout)

            notes = runner.invoke(
                app,
                [
                    "project",
                    "notes",
                    "--project",
                    str(context.root),
                    "--add",
                    "Check panel mapping before publication.",
                ],
            )
            self.assertEqual(notes.exit_code, 0, notes.stdout)
            self.assertIn("Check panel mapping before publication.", notes.stdout)

    def test_historical_project_metadata_without_new_optional_fields_still_loads(self):
        raw = {
            "schema_version": 1,
            "project_id": "legacy",
            "created_at": "2025-01-01T00:00:00Z",
            "config_file": "config.yaml",
            "toolkit": "Spatial Biology Toolkit",
        }
        metadata = ProjectMetadata.model_validate(raw)
        self.assertIsNone(metadata.title)
        self.assertEqual(metadata.notes_file, ".sbt/project_notes.md")

    def test_stage_manifest_accepts_unknown_future_optional_metadata(self):
        data = {
            "schema_version": 1,
            "project_id": "p",
            "run_id": "r",
            "stage": "prep",
            "display_name": "Preprocessing",
            "status": "completed",
            "managed_run": True,
            "started_at": "2025-01-01T00:00:00Z",
            "future_optional_field": {"value": 1},
        }
        manifest = StageManifest.model_validate(data)
        dumped = yaml.safe_load(
            yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False)
        )
        self.assertIn("future_optional_field", dumped)


if __name__ == "__main__":
    unittest.main()
