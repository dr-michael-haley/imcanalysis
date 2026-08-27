import json
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from SpatialBiologyToolkit.cli.main import app
from SpatialBiologyToolkit.pipeline.asset_cleanup import plan_asset_cleanup
from SpatialBiologyToolkit.pipeline.assets import (
    asset_map,
    inventory_assets,
    resolve_assets,
)
from SpatialBiologyToolkit.pipeline.executions import (
    EXECUTION_LOCK,
    allocate_executions,
    execution_lock,
    execution_output_path,
    load_execution_index,
    resolve_execution,
    update_execution,
    remove_executions,
    write_execution_index,
)
from SpatialBiologyToolkit.pipeline.manifests import read_model, read_yaml, write_yaml
from SpatialBiologyToolkit.pipeline.models import (
    ProjectStatusRefresh,
    RunStatus,
    StageStatus,
)
from SpatialBiologyToolkit.pipeline.migration import (
    LEGACY_STAGE_FOLDERS,
    apply_execution_layout_migration,
    plan_execution_layout_migration,
)
from SpatialBiologyToolkit.pipeline.planner import build_run_plan
from SpatialBiologyToolkit.pipeline.project import initialize_project, validate_project
from SpatialBiologyToolkit.pipeline.runs import create_run_record
from SpatialBiologyToolkit.pipeline.slurm import SubmissionError, submit_run
from SpatialBiologyToolkit.reporting.models import StageManifest


class FakeSbatchRunner:
    def __init__(self, job_ids, fail_at=None):
        self.job_ids = list(job_ids)
        self.fail_at = fail_at
        self.calls = []

    def __call__(self, arguments, **kwargs):
        self.calls.append((arguments, kwargs))
        position = len(self.calls) - 1
        if position == self.fail_at:
            return subprocess.CompletedProcess(
                arguments, 1, stdout="", stderr="submission rejected"
            )
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=f"{self.job_ids[position]};cluster\n",
            stderr="",
        )


class ExecutionLayoutTests(unittest.TestCase):
    def _project(self, temp_dir):
        return initialize_project(Path(temp_dir) / "project")

    def _submit(self, context, targets, job_ids):
        plan = build_run_plan(context, targets)
        self.assertTrue(plan.ready, plan.errors)
        run = create_run_record(context, plan, command=f"sbt run {' '.join(targets)}")
        submitted = submit_run(context, plan, run, runner=FakeSbatchRunner(job_ids))
        return run, submitted

    def test_numbering_follows_execution_order_and_stage_reruns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            first, _ = self._submit(context, ["debug"], ["101"])
            second, _ = self._submit(context, ["debug"], ["102"])

            records = load_execution_index(context).executions
            self.assertEqual([item.execution_label for item in records], ["001", "002"])
            self.assertEqual([item.stage for item in records], ["debug", "debug"])
            self.assertEqual(
                [execution_output_path(context, item).name for item in records],
                ["001_Environment_Diagnostics", "002_Environment_Diagnostics"],
            )
            self.assertNotEqual(
                first.executions[0].technical_run_id,
                second.executions[0].technical_run_id,
            )

    def test_concurrent_allocation_is_unique_and_lock_is_released_after_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)

            def allocate(position):
                return allocate_executions(
                    context, ["debug"], workflow_run_id=f"workflow-{position}"
                )[0]

            with ThreadPoolExecutor(max_workers=4) as pool:
                records = list(pool.map(allocate, range(8)))

            self.assertEqual(
                sorted(item.execution_id for item in records), list(range(1, 9))
            )
            self.assertEqual(
                len({item.technical_run_id for item in records}), len(records)
            )
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                with execution_lock(context):
                    raise RuntimeError("deliberate")
            self.assertFalse((context.root / EXECUTION_LOCK).exists())
            with execution_lock(context):
                pass

    def test_failed_atomic_allocation_leaves_index_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            before = load_execution_index(context)
            with patch(
                "SpatialBiologyToolkit.pipeline.executions.write_yaml",
                side_effect=OSError("interrupted write"),
            ):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    allocate_executions(
                        context, ["debug"], workflow_run_id="interrupted"
                    )
            after = load_execution_index(context)
            self.assertEqual(after.executions, before.executions)
            self.assertFalse((context.root / EXECUTION_LOCK).exists())

    def test_multistage_submission_records_consecutive_separate_identities(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            run, submitted = self._submit(context, ["debug", "config"], ["401", "402"])

            self.assertEqual(
                [item.execution_id for item in run.executions],
                [1, 2],
            )
            self.assertEqual(
                [job.dependency_job_id for job in submitted.jobs],
                [None, None],
            )
            self.assertEqual(
                {item.workflow_run_id for item in run.executions},
                {run.workflow_run_id},
            )
            self.assertEqual(len({item.technical_run_id for item in run.executions}), 2)
            self.assertNotIn("401", {item.technical_run_id for item in run.executions})

    def test_partial_submission_retains_failed_and_blocked_execution_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            plan = build_run_plan(context, ["debug", "config"])
            run = create_run_record(context, plan, command="sbt run debug config")
            with self.assertRaises(SubmissionError):
                submit_run(
                    context,
                    plan,
                    run,
                    runner=FakeSbatchRunner(["501", "unused"], fail_at=1),
                )

            records = load_execution_index(context).executions
            self.assertEqual([item.status for item in records], ["pending", "failed"])
            self.assertTrue(execution_output_path(context, records[0]).is_dir())
            self.assertFalse(execution_output_path(context, records[1]).exists())

    def test_summary_formats_filters_and_latest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            self._submit(context, ["debug", "config"], ["601", "602"])
            config_record = resolve_execution(context, "002")
            update_execution(context, config_record.technical_run_id, status="failed")
            runner = CliRunner()

            table = runner.invoke(app, ["summary", "--project", str(context.root)])
            self.assertEqual(table.exit_code, 0, table.stdout)
            self.assertLess(table.stdout.index("001"), table.stdout.index("002"))
            filtered = runner.invoke(
                app,
                ["summary", "--project", str(context.root), "--stage", "config"],
            )
            self.assertNotIn("Environment Diagnostics", filtered.stdout)
            failed = runner.invoke(
                app,
                ["summary", "--project", str(context.root), "--status", "failed"],
            )
            self.assertIn("Configuration Maintenance", failed.stdout)
            self.assertNotIn("Environment Diagnostics", failed.stdout)
            latest = runner.invoke(
                app, ["summary", "--project", str(context.root), "--latest"]
            )
            self.assertIn("002", latest.stdout)
            self.assertNotIn("001", latest.stdout)
            json_result = runner.invoke(
                app,
                ["summary", "--project", str(context.root), "--format", "json"],
            )
            payload = json.loads(json_result.stdout)
            self.assertEqual(len(payload["executions"]), 2)
            yaml_result = runner.invoke(
                app,
                ["summary", "--project", str(context.root), "--format", "yaml"],
            )
            self.assertEqual(len(yaml.safe_load(yaml_result.stdout)["executions"]), 2)

            status_result = runner.invoke(
                app, ["status", "001", "--project", str(context.root)]
            )
            self.assertEqual(status_result.exit_code, 0, status_result.stdout)
            self.assertIn("Execution 001", status_result.stdout)
            logs_result = runner.invoke(
                app,
                ["logs", "001", "--project", str(context.root), "--path-only"],
            )
            self.assertEqual(logs_result.exit_code, 0, logs_result.stdout)
            self.assertIn("debug_601.out", logs_result.stdout)
            report_result = runner.invoke(
                app,
                ["report", "001", "--project", str(context.root), "--path-only"],
            )
            self.assertEqual(report_result.exit_code, 0, report_result.stdout)
            self.assertTrue(report_result.stdout.strip().endswith("README.md"))

    def test_remove_compacts_ids_updates_manifest_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            first_run, _ = self._submit(context, ["debug"], ["701"])
            second_run, _ = self._submit(context, ["config"], ["702"])
            first = resolve_execution(context, "001")
            second = resolve_execution(context, "002")
            update_execution(context, first.technical_run_id, status="completed")
            update_execution(context, second.technical_run_id, status="completed")

            runner = CliRunner()
            declined = runner.invoke(
                app,
                ["remove", "001", "--project", str(context.root)],
                input="n\n",
            )
            self.assertNotEqual(declined.exit_code, 0)
            self.assertEqual(len(load_execution_index(context).executions), 2)
            result = runner.invoke(
                app,
                ["remove", "001", "--project", str(context.root), "--yes"],
            )
            self.assertEqual(result.exit_code, 0, result.stdout)
            remaining = resolve_execution(context, "001")
            self.assertEqual(remaining.technical_run_id, second.technical_run_id)
            self.assertEqual(
                execution_output_path(context, remaining).name,
                "001_Configuration_Maintenance",
            )
            manifest = read_model(
                execution_output_path(context, remaining) / "stage_manifest.yaml",
                StageManifest,
            )
            self.assertEqual(manifest.execution_id, 1)
            self.assertEqual(manifest.technical_run_id, second.technical_run_id)
            self.assertTrue(first_run.run_dir.is_dir())
            self.assertTrue(second_run.run_dir.is_dir())
            self.assertFalse(
                (context.root / "outputs" / "001_Environment_Diagnostics").exists()
            )
            audits = list((context.root / ".sbt" / "audit" / "removals").glob("*.yaml"))
            self.assertEqual(len(audits), 1)
            self.assertEqual(
                read_yaml(audits[0])["renumbered"][0]["new_execution_id"], 1
            )
            summary = CliRunner().invoke(
                app, ["summary", "--project", str(context.root)]
            )
            self.assertNotIn("Environment Diagnostics", summary.stdout)
            removed = CliRunner().invoke(
                app,
                [
                    "summary",
                    "--project",
                    str(context.root),
                    "--include-removed",
                ],
            )
            self.assertIn("Environment Diagnostics", removed.stdout)

    def test_remove_command_accepts_multiple_ids_and_compacts_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            self._submit(context, ["debug"], ["703"])
            self._submit(context, ["config"], ["704"])
            self._submit(context, ["debug"], ["705"])
            self._submit(context, ["config"], ["706"])
            original = load_execution_index(context).executions
            for record in original:
                update_execution(context, record.technical_run_id, status="completed")

            result = CliRunner().invoke(
                app,
                [
                    "remove",
                    "001",
                    "003",
                    "--project",
                    str(context.root),
                    "--yes",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.stdout)
            self.assertIn("Remove 2 executions?", result.stdout)
            self.assertIn("Removed 2 executions (001, 003)", result.stdout)
            remaining = load_execution_index(context).executions
            self.assertEqual(
                [record.technical_run_id for record in remaining],
                [original[1].technical_run_id, original[3].technical_run_id],
            )
            self.assertEqual(
                [record.execution_label for record in remaining],
                ["001", "002"],
            )
            self.assertEqual(
                [execution_output_path(context, record).name for record in remaining],
                ["001_Configuration_Maintenance", "002_Configuration_Maintenance"],
            )
            audits = [
                read_yaml(path)
                for path in sorted(
                    (context.root / ".sbt" / "audit" / "removals").glob("*.yaml")
                )
            ]
            self.assertEqual(len(audits), 2)
            self.assertEqual(
                {audit["previous_execution_id"] for audit in audits},
                {1, 3},
            )
            self.assertTrue(all(len(audit["renumbered"]) == 2 for audit in audits))

    def test_remove_command_resolves_every_id_before_mutating(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            self._submit(context, ["debug"], ["707"])
            self._submit(context, ["config"], ["708"])
            records = load_execution_index(context).executions
            for record in records:
                update_execution(context, record.technical_run_id, status="completed")

            result = CliRunner().invoke(
                app,
                [
                    "remove",
                    "001",
                    "999",
                    "--project",
                    str(context.root),
                    "--yes",
                ],
            )

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Execution 999 is not active", result.output)
            self.assertEqual(
                load_execution_index(context).executions,
                [
                    resolve_execution(context, "001"),
                    resolve_execution(context, "002"),
                ],
            )
            self.assertFalse(
                (context.root / ".sbt" / "audit" / "removals").exists()
            )

    def test_bulk_remove_compacts_once_and_preserves_each_audit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            first_run, _ = self._submit(context, ["debug"], ["711"])
            second_run, _ = self._submit(context, ["config"], ["712"])
            third_run, _ = self._submit(context, ["debug"], ["713"])
            first, second, third = load_execution_index(context).executions
            update_execution(context, first.technical_run_id, status="failed")
            update_execution(context, second.technical_run_id, status="blocked")
            update_execution(context, third.technical_run_id, status="completed")

            audits = remove_executions(
                context,
                [first.technical_run_id, second.technical_run_id],
                reason="test cleanup",
                confirmation_mode="system",
            )

            self.assertEqual(len(audits), 2)
            remaining = load_execution_index(context).executions
            self.assertEqual(len(remaining), 1)
            self.assertEqual(remaining[0].technical_run_id, third.technical_run_id)
            self.assertEqual(remaining[0].execution_label, "001")
            self.assertEqual(len(audits[0].renumbered), 1)
            self.assertEqual(len(audits[1].renumbered), 1)
            self.assertTrue(first_run.run_dir.is_dir())
            self.assertTrue(second_run.run_dir.is_dir())
            self.assertTrue(third_run.run_dir.is_dir())
            audit_paths = list(
                (context.root / ".sbt" / "audit" / "removals").glob("*.yaml")
            )
            self.assertEqual(len(audit_paths), 2)

    def test_cleanup_refreshes_then_removes_failed_and_blocked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            self._submit(context, ["debug"], ["721"])
            self._submit(context, ["config"], ["722"])
            self._submit(context, ["debug"], ["723"])
            records = load_execution_index(context).executions
            target_statuses = ["failed", "blocked", "completed"]

            def fake_refresh(selected_context, *, persist=True):
                reports = []
                for record, status in zip(records, target_statuses):
                    if persist:
                        update_execution(
                            selected_context,
                            record.technical_run_id,
                            status=status,
                        )
                    reports.append(
                        RunStatus(
                            run_id=record.workflow_run_id,
                            workflow_run_id=record.workflow_run_id,
                            project_id=selected_context.project_metadata.project_id,
                            checked_at=datetime.now(timezone.utc),
                            overall_status=status,
                            stages=[
                                StageStatus(
                                    stage=record.stage,
                                    execution_id=record.execution_id,
                                    technical_run_id=record.technical_run_id,
                                    job_id=record.slurm_job_id,
                                    status=status,
                                    source="test",
                                )
                            ],
                        )
                    )
                return ProjectStatusRefresh(
                    project_id=selected_context.project_metadata.project_id,
                    checked_at=datetime.now(timezone.utc),
                    workflow_count=3,
                    execution_count=3,
                    reports=reports,
                )

            with patch(
                "SpatialBiologyToolkit.cli.main.refresh_project_status",
                side_effect=fake_refresh,
            ) as refresh:
                result = CliRunner().invoke(
                    app,
                    ["cleanup", "--project", str(context.root), "--yes"],
                )

            self.assertEqual(result.exit_code, 0, result.stdout)
            self.assertEqual(refresh.call_count, 1)
            self.assertTrue(refresh.call_args.kwargs["persist"])
            remaining = load_execution_index(context).executions
            self.assertEqual(len(remaining), 1)
            self.assertEqual(remaining[0].technical_run_id, records[2].technical_run_id)
            self.assertIn("Removed 2 execution(s)", result.stdout)

    def test_cleanup_dry_run_refreshes_without_mutating_project_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            self._submit(context, ["debug"], ["731"])
            record = load_execution_index(context).executions[0]

            def fake_refresh(selected_context, *, persist=True):
                self.assertFalse(persist)
                return ProjectStatusRefresh(
                    project_id=selected_context.project_metadata.project_id,
                    checked_at=datetime.now(timezone.utc),
                    workflow_count=1,
                    execution_count=1,
                    reports=[
                        RunStatus(
                            run_id=record.workflow_run_id,
                            workflow_run_id=record.workflow_run_id,
                            project_id=selected_context.project_metadata.project_id,
                            checked_at=datetime.now(timezone.utc),
                            overall_status="failed",
                            stages=[
                                StageStatus(
                                    stage=record.stage,
                                    execution_id=record.execution_id,
                                    technical_run_id=record.technical_run_id,
                                    job_id=record.slurm_job_id,
                                    status="failed",
                                    source="test",
                                )
                            ],
                        )
                    ],
                )

            with patch(
                "SpatialBiologyToolkit.cli.main.refresh_project_status",
                side_effect=fake_refresh,
            ) as refresh:
                result = CliRunner().invoke(
                    app,
                    [
                        "cleanup",
                        "--project",
                        str(context.root),
                        "--dry-run",
                    ],
                )

            self.assertEqual(result.exit_code, 0, result.stdout)
            self.assertEqual(refresh.call_count, 1)
            current = load_execution_index(context).executions
            self.assertEqual(current, [record])
            self.assertFalse(
                (context.root / ".sbt" / "audit" / "removals").exists()
            )
            self.assertIn("Dry run only", result.stdout)

    def test_bulk_remove_rolls_back_index_and_outputs_on_rewrite_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            self._submit(context, ["debug"], ["741"])
            self._submit(context, ["config"], ["742"])
            before = load_execution_index(context)
            first, second = before.executions
            update_execution(context, first.technical_run_id, status="failed")
            update_execution(context, second.technical_run_id, status="completed")
            before = load_execution_index(context)
            old_paths = [
                execution_output_path(context, record)
                for record in before.executions
            ]
            from SpatialBiologyToolkit.pipeline import executions as execution_module

            original_rewrite = execution_module.rewrite_execution_records
            call_count = 0

            def flaky_rewrite(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise OSError("interrupted rewrite")
                return original_rewrite(*args, **kwargs)

            with patch(
                "SpatialBiologyToolkit.pipeline.executions.rewrite_execution_records",
                side_effect=flaky_rewrite,
            ):
                with self.assertRaisesRegex(OSError, "interrupted rewrite"):
                    remove_executions(
                        context,
                        [first.technical_run_id],
                        reason="rollback test",
                        confirmation_mode="system",
                    )

            self.assertEqual(load_execution_index(context).executions, before.executions)
            self.assertTrue(all(path.exists() for path in old_paths))
            audit_root = context.root / ".sbt" / "audit" / "removals"
            self.assertEqual(list(audit_root.glob("*.yaml")), [])

    def test_summary_refreshes_by_default_with_offline_opt_out(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            refreshed = ProjectStatusRefresh(
                project_id=context.project_metadata.project_id,
                checked_at=datetime.now(timezone.utc),
            )
            with patch(
                "SpatialBiologyToolkit.cli.main.refresh_project_status",
                return_value=refreshed,
            ) as refresh:
                default = CliRunner().invoke(
                    app,
                    ["summary", "--project", str(context.root)],
                )
                offline = CliRunner().invoke(
                    app,
                    [
                        "summary",
                        "--project",
                        str(context.root),
                        "--no-refresh",
                    ],
                )

            self.assertEqual(default.exit_code, 0, default.stdout)
            self.assertEqual(offline.exit_code, 0, offline.stdout)
            self.assertEqual(refresh.call_count, 1)

    def test_unknown_asset_effect_requires_explicit_noninteractive_risk_flag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            record = allocate_executions(
                context, ["cellpose"], workflow_run_id="asset-workflow"
            )[0]
            update_execution(context, record.technical_run_id, status="failed")
            runner = CliRunner()
            declined = runner.invoke(
                app,
                ["remove", "001", "--project", str(context.root)],
                input="y\nn\n",
            )
            self.assertNotEqual(declined.exit_code, 0)
            self.assertIn("created or modified reusable assets", declined.stdout)
            self.assertEqual(len(load_execution_index(context).executions), 1)
            refused = runner.invoke(
                app,
                ["remove", "001", "--project", str(context.root), "--yes"],
            )
            self.assertNotEqual(refused.exit_code, 0)
            self.assertIn("--accept-asset-risk", refused.output)
            self.assertEqual(len(load_execution_index(context).executions), 1)
            accepted = runner.invoke(
                app,
                [
                    "remove",
                    "001",
                    "--project",
                    str(context.root),
                    "--yes",
                    "--accept-asset-risk",
                ],
            )
            self.assertEqual(accepted.exit_code, 0, accepted.stdout)
            self.assertIn("were not deleted or restored", accepted.stdout)

    def test_remove_offers_only_created_unused_assets_and_preserves_h5ad(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            workflow_id = "asset-cleanup-workflow"
            run_dir = context.runs_dir / workflow_id
            run_dir.mkdir(parents=True)
            write_yaml(
                run_dir / "project_assets.before.yaml",
                inventory_assets(
                    project_id=context.project_metadata.project_id,
                    project_root=context.root,
                    config=context.config,
                ),
            )
            prep, denoise = allocate_executions(
                context,
                ["prep", "denoise"],
                workflow_run_id=workflow_id,
            )
            update_execution(
                context,
                prep.technical_run_id,
                status="completed",
                asset_effect="created",
            )
            update_execution(context, denoise.technical_run_id, status="completed")

            assets = asset_map(resolve_assets(context.config, context.root))
            tiff_root = assets["tiff_stacks"].path
            raw_root = assets["raw_images"].path
            metadata_root = assets["metadata"].path
            tiff_root.mkdir(parents=True)
            (tiff_root / "unused.tiff").write_bytes(b"unused")
            protected = tiff_root / "analysis.h5ad"
            protected.write_bytes(b"protected")
            raw_root.mkdir(parents=True)
            (raw_root / "channel.tiff").write_bytes(b"shared")
            metadata_root.mkdir(parents=True, exist_ok=True)
            (metadata_root / "panel.csv").write_text("name\n", encoding="utf-8")

            plan = plan_asset_cleanup(context, resolve_execution(context, "001"))
            self.assertEqual(
                [item.role for item in plan.removable],
                ["tiff_stacks"],
            )
            dependent = {
                item.role: item
                for item in plan.retained
                if item.reason == "used by remaining stages"
            }
            self.assertEqual(set(dependent), {"raw_images"})
            self.assertTrue(
                all(
                    "002 denoise" in item.dependent_stages[0]
                    for item in dependent.values()
                )
            )
            combined_plan = plan_asset_cleanup(
                context,
                prep,
                excluded_technical_run_ids={
                    prep.technical_run_id,
                    denoise.technical_run_id,
                },
            )
            self.assertIn(
                "raw_images",
                [item.role for item in combined_plan.removable],
            )
            self.assertNotIn(
                "raw_images",
                [
                    item.role
                    for item in combined_plan.retained
                    if item.reason == "used by remaining stages"
                ],
            )
            self.assertIn(
                protected,
                [
                    item.path
                    for item in plan.retained
                    if item.reason == ".h5ad files are always protected"
                ],
            )

            result = CliRunner().invoke(
                app,
                ["remove", "001", "--project", str(context.root)],
                input="y\ny\nyes\n",
            )
            self.assertEqual(result.exit_code, 0, result.stdout)
            self.assertIn("Remaining unused assets eligible for removal", result.stdout)
            self.assertIn(
                "Assets retained because remaining stages depend on them",
                result.stdout,
            )
            self.assertFalse((tiff_root / "unused.tiff").exists())
            self.assertTrue(protected.is_file())
            self.assertTrue((raw_root / "channel.tiff").is_file())
            self.assertTrue((metadata_root / "panel.csv").is_file())
            audit_path = next(
                (context.root / ".sbt" / "audit" / "removals").glob("*.yaml")
            )
            cleanup = read_yaml(audit_path)["asset_cleanup"]
            self.assertTrue(cleanup["confirmed"])
            self.assertEqual(cleanup["removable"][0]["role"], "tiff_stacks")
            self.assertGreater(cleanup["removed_entries"], 0)

    def test_remove_keeps_eligible_assets_unless_literal_yes_is_entered(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            workflow_id = "asset-decline-workflow"
            run_dir = context.runs_dir / workflow_id
            run_dir.mkdir(parents=True)
            write_yaml(
                run_dir / "project_assets.before.yaml",
                inventory_assets(
                    project_id=context.project_metadata.project_id,
                    project_root=context.root,
                    config=context.config,
                ),
            )
            record = allocate_executions(
                context, ["cellpose"], workflow_run_id=workflow_id
            )[0]
            update_execution(
                context,
                record.technical_run_id,
                status="completed",
                asset_effect="created",
            )
            masks = asset_map(resolve_assets(context.config, context.root))[
                "masks"
            ].path
            masks.mkdir(parents=True)
            created = masks / "mask.tiff"
            created.write_bytes(b"mask")

            result = CliRunner().invoke(
                app,
                ["remove", "001", "--project", str(context.root)],
                input="y\ny\ny\n",
            )
            self.assertEqual(result.exit_code, 0, result.stdout)
            self.assertTrue(created.is_file())
            self.assertIn("were not deleted or restored", result.stdout)
            audit_path = next(
                (context.root / ".sbt" / "audit" / "removals").glob("*.yaml")
            )
            self.assertFalse(read_yaml(audit_path)["asset_cleanup"]["confirmed"])

    def _legacy_manifest(
        self,
        context,
        *,
        stage,
        run_name,
        started_at,
        technical_id,
    ):
        source = context.root / "outputs" / LEGACY_STAGE_FOLDERS[stage] / run_name
        source.mkdir(parents=True)
        write_yaml(
            source / "stage_manifest.yaml",
            StageManifest(
                schema_version=1,
                project_id=context.project_metadata.project_id,
                run_id=run_name,
                technical_run_id=technical_id,
                stage=stage,
                display_name=stage,
                status="completed",
                managed_run=True,
                started_at=started_at,
                completed_at=started_at,
            ),
        )
        (source / "README.md").write_text("legacy\n", encoding="utf-8")
        return source

    def test_explicit_migration_uses_manifest_chronology_and_preserves_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            later = datetime(2026, 1, 2, tzinfo=timezone.utc)
            earlier = datetime(2026, 1, 1, tzinfo=timezone.utc)
            prep_source = self._legacy_manifest(
                context,
                stage="prep",
                run_name="prep-run",
                started_at=later,
                technical_id="stage-prep-stable",
            )
            debug_source = self._legacy_manifest(
                context,
                stage="debug",
                run_name="debug-run",
                started_at=earlier,
                technical_id="stage-debug-stable",
            )

            dry = CliRunner().invoke(
                app,
                [
                    "project",
                    "migrate-execution-layout",
                    "--project",
                    str(context.root),
                    "--dry-run",
                ],
            )
            self.assertEqual(dry.exit_code, 0, dry.stdout)
            self.assertTrue(prep_source.is_dir())
            self.assertTrue(debug_source.is_dir())
            plan = plan_execution_layout_migration(context)
            self.assertTrue(plan.safe_to_apply, plan.ambiguities)
            self.assertEqual(
                [item.execution.stage for item in plan.records], ["debug", "prep"]
            )
            audit = apply_execution_layout_migration(context, plan)
            self.assertEqual(len(audit.records), 2)
            records = load_execution_index(context).executions
            self.assertEqual(
                [item.technical_run_id for item in records],
                ["stage-debug-stable", "stage-prep-stable"],
            )
            self.assertTrue(
                (context.root / "outputs" / "001_Environment_Diagnostics").is_dir()
            )
            self.assertTrue((context.root / "outputs" / "002_Preprocessing").is_dir())

    def test_repeated_legacy_stage_migrates_separately_and_ambiguity_aborts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            self._legacy_manifest(
                context,
                stage="prep",
                run_name="first",
                started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                technical_id="stage-first",
            )
            self._legacy_manifest(
                context,
                stage="prep",
                run_name="second",
                started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                technical_id="stage-second",
            )
            plan = plan_execution_layout_migration(context)
            self.assertEqual(len(plan.records), 2)
            apply_execution_layout_migration(context, plan)
            self.assertTrue((context.root / "outputs" / "001_Preprocessing").is_dir())
            self.assertTrue((context.root / "outputs" / "002_Preprocessing").is_dir())

        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            legacy = context.root / "outputs" / LEGACY_STAGE_FOLDERS["prep"]
            legacy.mkdir(parents=True)
            (legacy / "untracked.txt").write_text("ambiguous", encoding="utf-8")
            plan = plan_execution_layout_migration(context)
            self.assertFalse(plan.safe_to_apply)
            self.assertTrue(plan.ambiguities)
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                apply_execution_layout_migration(context, plan)
            self.assertTrue((legacy / "untracked.txt").is_file())

    def test_validation_detects_duplicate_ids_folder_mismatch_and_missing_link(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self._project(temp_dir)
            records = allocate_executions(
                context, ["debug", "config"], workflow_run_id="missing-workflow"
            )
            duplicate = records[1].model_copy(
                update={
                    "execution_id": 1,
                    "execution_label": "001",
                    "output_folder": Path("outputs/wrong-folder"),
                }
            )
            index = load_execution_index(context).model_copy(
                update={"executions": [records[0], duplicate]}
            )
            write_execution_index(context, index)

            report = validate_project(context)
            messages = "\n".join(item.message for item in report.reporting_outputs)
            self.assertIn("unique, ordered, and sequential", messages)
            self.assertIn("output folder does not match", messages)
            self.assertIn("Permanent workflow run record does not resolve", messages)


if __name__ == "__main__":
    unittest.main()
