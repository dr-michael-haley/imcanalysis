import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from SpatialBiologyToolkit.cli.main import app
from SpatialBiologyToolkit.environments.models import EnvironmentSummary
from SpatialBiologyToolkit.pipeline.assets import resolve_assets
from SpatialBiologyToolkit.pipeline.dependencies import (
    active_project_dependencies,
    recorded_external_dependency,
    refresh_external_dependency,
)
from SpatialBiologyToolkit.pipeline.environment_selection import (
    apply_environment_override,
)
from SpatialBiologyToolkit.pipeline.executions import (
    execution_summaries,
    load_execution_index,
    update_execution,
)
from SpatialBiologyToolkit.pipeline.logs import resolve_run_logs, tail_text
from SpatialBiologyToolkit.pipeline.manifests import read_yaml
from SpatialBiologyToolkit.pipeline.planner import build_run_plan
from SpatialBiologyToolkit.pipeline.project import initialize_project
from SpatialBiologyToolkit.pipeline.runs import (
    ASSETS_BEFORE,
    RESOLVED_CONFIG,
    RUN_MANIFEST,
    RUN_PLAN,
    STATUS_FILE,
    SUBMITTED_JOBS,
    USER_CONFIG,
    create_run_record,
    resolve_run_directory,
)
from SpatialBiologyToolkit.pipeline.slurm import (
    SubmissionError,
    sbt_environment,
    submit_run,
)
from SpatialBiologyToolkit.pipeline.status import (
    inspect_run_status,
    refresh_project_status,
)
from SpatialBiologyToolkit.scripts.config_and_utils import parse_arguments


class FakeSbatchRunner:
    def __init__(self, outputs, fail_at=None):
        self.outputs = list(outputs)
        self.fail_at = fail_at
        self.calls = []

    def __call__(self, arguments, **kwargs):
        self.calls.append((arguments, kwargs))
        index = len(self.calls) - 1
        if self.fail_at is not None and index == self.fail_at:
            return subprocess.CompletedProcess(
                arguments,
                1,
                stdout="",
                stderr="submission rejected",
            )
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=f"{self.outputs[index]};cluster\n",
            stderr="",
        )


class RunControlTests(unittest.TestCase):
    def test_run_help_has_no_environment_install_bypass_flag(self):
        result = CliRunner().invoke(app, ["run", "--help"])

        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertIn("--ignore-missing-assets", result.stdout)
        self.assertNotIn("--install-missing-envs", result.stdout)
        self.assertNotIn("--install-missing-environments", result.stdout)

    def _project_and_plan(self, temp_dir: str, targets=None):
        root = Path(temp_dir) / "project"
        context = initialize_project(root)
        (root / "IMC_files" / "case.mcd").write_bytes(b"x")
        plan = build_run_plan(context, targets or ["segmentation"])
        self.assertTrue(plan.ready, plan.errors)
        return context, plan

    def test_run_directory_contains_versioned_provenance_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, plan = self._project_and_plan(temp_dir)
            run = create_run_record(context, plan, command="sbt run segmentation")

            for filename in (
                RUN_MANIFEST,
                RUN_PLAN,
                USER_CONFIG,
                RESOLVED_CONFIG,
                SUBMITTED_JOBS,
                STATUS_FILE,
                ASSETS_BEFORE,
            ):
                self.assertTrue((run.run_dir / filename).is_file(), filename)
            self.assertTrue((run.run_dir / "logs").is_dir())
            manifest = read_yaml(run.run_dir / RUN_MANIFEST)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["workflow_run_id"], run.workflow_run_id)
            self.assertEqual(len(manifest["executions"]), len(plan.resolved_stages))
            self.assertIn("logging", read_yaml(run.run_dir / RESOLVED_CONFIG))

    def test_sbatch_commands_dependencies_and_exported_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, plan = self._project_and_plan(temp_dir)
            run = create_run_record(context, plan, command="sbt run segmentation")
            runner = FakeSbatchRunner(["101", "102", "103", "104", "105"])

            submitted = submit_run(context, plan, run, runner=runner)

            self.assertTrue(submitted.submission_complete)
            self.assertEqual(
                [job.job_id for job in submitted.jobs],
                ["101", "102", "103", "104", "105"],
            )
            first_args, first_kwargs = runner.calls[0]
            second_args, _ = runner.calls[1]
            third_args, _ = runner.calls[2]
            fourth_args, _ = runner.calls[3]
            self.assertIn("--parsable", first_args)
            self.assertIn("--export=ALL", first_args)
            self.assertNotIn("--dependency=afterok:101", first_args)
            self.assertIn("--dependency=afterok:101", second_args)
            self.assertIn("--dependency=afterok:102:101", third_args)
            self.assertIn("--dependency=afterok:102", fourth_args)
            self.assertNotIn("--dependency=afterok:103", fourth_args)
            exported = first_kwargs["env"]
            self.assertEqual(exported["SBT_PROJECT_ROOT"], str(context.root))
            self.assertEqual(
                exported["SBT_PROJECT_ID"], context.project_metadata.project_id
            )
            self.assertEqual(exported["SBT_CONFIG"], str(run.resolved_config_path))
            self.assertEqual(exported["SBT_RUN_ID"], run.run_id)
            self.assertEqual(exported["SBT_WORKFLOW_RUN_ID"], run.workflow_run_id)
            self.assertEqual(exported["SBT_EXECUTION_ID"], "1")
            self.assertEqual(exported["SBT_EXECUTION_LABEL"], "001")
            self.assertTrue(exported["SBT_TECHNICAL_RUN_ID"].startswith("stage-"))
            self.assertNotEqual(submitted.jobs[0].job_id, exported["SBT_EXECUTION_ID"])
            self.assertNotEqual(
                submitted.jobs[0].job_id,
                exported["SBT_TECHNICAL_RUN_ID"],
            )
            self.assertEqual(exported["SBT_STAGE"], "prep")
            self.assertEqual(exported["SBT_ENVIRONMENT_KEY"], "analysis")
            self.assertEqual(exported["SBT_CONDA_ENV"], "sbt-analysis")
            self.assertEqual(exported["SBT_CONDA_ENV_ANALYSIS"], "sbt-analysis")

    def test_single_stage_environment_override_is_exported_and_persisted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, plan = self._project_and_plan(temp_dir, ["prep"])
            plan = apply_environment_override(plan, "denoise")
            run = create_run_record(context, plan, command="sbt run prep")
            submit_run(
                context,
                plan,
                run,
                runner=FakeSbatchRunner(["751"]),
            )

            exported = sbt_environment(context, run, "prep")
            manifest = read_yaml(run.run_dir / RUN_MANIFEST)
            stage_manifest = read_yaml(
                context.root / run.executions[0].output_folder / "stage_manifest.yaml"
            )

            self.assertEqual(plan.environment_overrides, {"prep": "denoise"})
            self.assertEqual(exported["SBT_ENVIRONMENT_KEY"], "denoise")
            self.assertEqual(exported["SBT_CONDA_ENV"], "sbt-denoise")
            self.assertEqual(exported["SBT_CONDA_ENV_DENOISE"], "sbt-denoise")
            self.assertEqual(exported["SBT_ENVIRONMENT_OVERRIDE"], "1")
            self.assertEqual(
                exported["SBT_DEFAULT_ENVIRONMENT_KEYS"], "analysis"
            )
            self.assertEqual(
                manifest["environment_overrides"], {"prep": "denoise"}
            )
            self.assertEqual(stage_manifest["environment"]["key"], "denoise")
            self.assertTrue(stage_manifest["environment"]["overridden"])
            self.assertEqual(
                stage_manifest["environment"]["default_keys"], ["analysis"]
            )
            readme = (
                context.root / run.executions[0].output_folder / "README.md"
            ).read_text(encoding="utf-8")
            self.assertIn("Per-run override: `yes`", readme)

    def test_environment_override_rejects_multi_environment_stage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = initialize_project(Path(temp_dir) / "project")
            plan = build_run_plan(
                context,
                ["cellpose"],
                dependency_policy="none",
            )

            with self.assertRaisesRegex(ValueError, "multiple Conda environments"):
                apply_environment_override(plan, "analysis")

    def test_external_dependency_is_added_to_a_separate_run_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, first_plan = self._project_and_plan(temp_dir, ["debug"])
            first_run = create_run_record(
                context, first_plan, command="sbt run debug"
            )
            submit_run(
                context,
                first_plan,
                first_run,
                runner=FakeSbatchRunner(["801"]),
            )
            dependency = recorded_external_dependency(context, "001")
            second_plan = build_run_plan(context, ["debug"])
            second_run = create_run_record(
                context,
                second_plan,
                command="sbt run debug --after 001",
                external_dependency=dependency,
            )

            runner = FakeSbatchRunner(["802"])
            submitted = submit_run(
                context,
                second_plan,
                second_run,
                external_dependency=dependency,
                runner=runner,
            )

            self.assertEqual(submitted.jobs[0].dependency_job_id, "801")
            self.assertIn("--dependency=afterok:801", runner.calls[0][0])
            manifest = read_yaml(second_run.run_dir / RUN_MANIFEST)
            self.assertEqual(manifest["external_dependency"]["execution_label"], "001")
            self.assertEqual(manifest["external_dependency"]["job_id"], "801")

    def test_completed_external_dependency_does_not_add_scheduler_wait(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, first_plan = self._project_and_plan(temp_dir, ["debug"])
            first_run = create_run_record(
                context, first_plan, command="sbt run debug"
            )
            submit_run(
                context,
                first_plan,
                first_run,
                runner=FakeSbatchRunner(["811"]),
            )
            dependency = recorded_external_dependency(context, "001").model_copy(
                update={"observed_status": "completed"}
            )
            second_plan = build_run_plan(context, ["debug"])
            second_run = create_run_record(
                context,
                second_plan,
                command="sbt run debug --after 001",
                external_dependency=dependency,
            )

            runner = FakeSbatchRunner(["812"])
            submitted = submit_run(
                context,
                second_plan,
                second_run,
                external_dependency=dependency,
                runner=runner,
            )

            self.assertIsNone(submitted.jobs[0].dependency_job_id)
            self.assertNotIn("--dependency=", " ".join(runner.calls[0][0]))

    def test_active_dependency_discovery_is_scoped_to_project_job_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, plan = self._project_and_plan(temp_dir, ["debug"])
            run = create_run_record(context, plan, command="sbt run debug")
            submit_run(
                context,
                plan,
                run,
                runner=FakeSbatchRunner(["821"]),
            )

            def queue_runner(arguments, **_kwargs):
                output = (
                    "821|RUNNING|sbt_001_debug|gpu|00:01|01:00|1|2|None\n"
                    "999|RUNNING|unrelated|gpu|00:01|01:00|1|2|None\n"
                )
                return subprocess.CompletedProcess(arguments, 0, output, "")

            with patch(
                "SpatialBiologyToolkit.pipeline.scheduler._managed_jobs",
                return_value={},
            ):
                candidates = active_project_dependencies(
                    context, runner=queue_runner
                )

            self.assertEqual([item.job_id for item in candidates], ["821"])
            self.assertEqual(candidates[0].execution_label, "001")
            self.assertEqual(candidates[0].observed_status, "running")

    def test_dependency_refresh_uses_sacct_after_job_leaves_squeue(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, plan = self._project_and_plan(temp_dir, ["debug"])
            run = create_run_record(context, plan, command="sbt run debug")
            submit_run(
                context,
                plan,
                run,
                runner=FakeSbatchRunner(["841"]),
            )
            dependency = recorded_external_dependency(context, "001")
            calls = []

            def status_runner(arguments, **_kwargs):
                calls.append(arguments)
                if arguments[0] == "sacct":
                    return subprocess.CompletedProcess(
                        arguments,
                        0,
                        "841|COMPLETED|sbt_001_debug|0:0\n",
                        "",
                    )
                return subprocess.CompletedProcess(arguments, 0, "", "")

            with patch(
                "SpatialBiologyToolkit.pipeline.scheduler._managed_jobs",
                return_value={},
            ):
                refreshed = refresh_external_dependency(
                    context,
                    dependency,
                    runner=status_runner,
                )

            self.assertEqual(refreshed.observed_status, "completed")
            self.assertEqual(refreshed.source, "sacct")
            self.assertEqual([call[0] for call in calls].count("squeue"), 2)
            self.assertEqual([call[0] for call in calls].count("sacct"), 1)

    def test_submission_uses_all_actual_dependencies_without_chaining_independent_stages(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, plan = self._project_and_plan(
                temp_dir,
                ["debug", "config", "cox"],
            )
            stages = list(plan.resolved_stages)
            stages[2] = stages[2].model_copy(update={"depends_on": ["debug", "config"]})
            plan = plan.model_copy(update={"resolved_stages": stages})
            run = create_run_record(context, plan, command="sbt run debug config cox")
            runner = FakeSbatchRunner(["701", "702", "703"])

            submitted = submit_run(context, plan, run, runner=runner)

            self.assertEqual(
                [job.dependency_job_id for job in submitted.jobs],
                [None, None, "701:702"],
            )
            self.assertNotIn("--dependency=", " ".join(runner.calls[1][0]))
            self.assertIn(
                "--dependency=afterok:701:702",
                runner.calls[2][0],
            )

    def test_shared_scientific_config_parser_honors_sbt_config(self):
        config_path = str(Path("run") / "config.resolved.yaml")
        with patch.dict("os.environ", {"SBT_CONFIG": config_path}, clear=False):
            with patch("sys.argv", ["stage-module"]):
                arguments = parse_arguments()

        self.assertEqual(arguments.config, config_path)

    def test_invalid_config_stops_before_submission(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            initialize_project(root)
            (root / "config.yaml").write_text(
                "preprocess:\n  minimum_roi_dimensions: 0\n",
                encoding="utf-8",
            )
            with patch("SpatialBiologyToolkit.cli.main.submit_run") as submit_mock:
                result = CliRunner().invoke(
                    app,
                    ["run", "debug", "--project", str(root)],
                )

            self.assertNotEqual(result.exit_code, 0)
            submit_mock.assert_not_called()

    def test_partial_submission_failure_is_recorded_and_stops(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, plan = self._project_and_plan(temp_dir)
            run = create_run_record(context, plan, command="sbt run segmentation")
            runner = FakeSbatchRunner(["201", "unused"], fail_at=1)

            with self.assertRaises(SubmissionError):
                submit_run(context, plan, run, runner=runner)

            self.assertEqual(len(runner.calls), 2)
            submitted = read_yaml(run.run_dir / SUBMITTED_JOBS)
            self.assertFalse(submitted["submission_complete"])
            self.assertEqual(submitted["jobs"][0]["state"], "submitted")
            self.assertEqual(submitted["jobs"][1]["state"], "submission_failed")
            self.assertEqual(
                read_yaml(run.run_dir / STATUS_FILE)["overall_status"],
                "partial_submission_failed",
            )

    def test_status_combines_squeue_and_sacct(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, plan = self._project_and_plan(temp_dir, ["debug", "config"])
            run = create_run_record(context, plan, command="sbt run debug config")
            submit_run(
                context,
                plan,
                run,
                runner=FakeSbatchRunner(["301", "302"]),
            )

            def status_runner(arguments, **_kwargs):
                if arguments[0] == "squeue":
                    return subprocess.CompletedProcess(
                        arguments,
                        0,
                        stdout="301|RUNNING|debug\n",
                        stderr="",
                    )
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    stdout="302|COMPLETED|config|0:0\n",
                    stderr="",
                )

            report = inspect_run_status(context, run.run_dir, runner=status_runner)

            self.assertEqual(report.overall_status, "running")
            self.assertEqual(
                {stage.stage: stage.status for stage in report.stages},
                {"debug": "running", "config": "completed"},
            )
            completed_at = next(
                item.completed_at
                for item in execution_summaries(context)
                if item.stage == "config"
            )
            inspect_run_status(context, run.run_dir, runner=status_runner)
            self.assertEqual(
                next(
                    item.completed_at
                    for item in execution_summaries(context)
                    if item.stage == "config"
                ),
                completed_at,
            )

    def test_project_refresh_batches_all_active_workflows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, first_plan = self._project_and_plan(temp_dir, ["debug"])
            first = create_run_record(context, first_plan, command="sbt run debug")
            submit_run(context, first_plan, first, runner=FakeSbatchRunner(["301"]))
            second_plan = build_run_plan(context, ["config"])
            second = create_run_record(
                context,
                second_plan,
                command="sbt run config",
            )
            submit_run(
                context,
                second_plan,
                second,
                runner=FakeSbatchRunner(["302"]),
            )
            calls = []

            def status_runner(arguments, **_kwargs):
                calls.append(arguments)
                if arguments[0] == "squeue":
                    return subprocess.CompletedProcess(
                        arguments,
                        0,
                        stdout="301|RUNNING|debug|None\n",
                        stderr="",
                    )
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    stdout="302|COMPLETED|config|0:0\n",
                    stderr="",
                )

            refreshed = refresh_project_status(context, runner=status_runner)

            self.assertEqual(refreshed.workflow_count, 2)
            self.assertEqual(refreshed.execution_count, 2)
            self.assertEqual([call[0] for call in calls], ["squeue", "sacct"])
            self.assertIn("301,302", calls[0])
            self.assertEqual(
                [record.status for record in load_execution_index(context).executions],
                ["running", "completed"],
            )

    def test_project_refresh_retains_verified_terminal_state_missing_from_slurm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, plan = self._project_and_plan(temp_dir, ["debug"])
            run = create_run_record(context, plan, command="sbt run debug")
            submit_run(context, plan, run, runner=FakeSbatchRunner(["401"]))
            record = load_execution_index(context).executions[0]
            update_execution(context, record.technical_run_id, status="failed")

            def empty_runner(arguments, **_kwargs):
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    stdout="",
                    stderr="",
                )

            refreshed = refresh_project_status(context, runner=empty_runner)

            self.assertEqual(
                load_execution_index(context).executions[0].status,
                "failed",
            )
            self.assertEqual(
                refreshed.reports[0].stages[0].source,
                "recorded terminal state",
            )

    def test_project_refresh_reports_unavailable_scheduler_as_unknown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, plan = self._project_and_plan(temp_dir, ["debug"])
            run = create_run_record(context, plan, command="sbt run debug")
            submit_run(context, plan, run, runner=FakeSbatchRunner(["402"]))

            def unavailable_runner(arguments, **_kwargs):
                raise FileNotFoundError(arguments[0])

            refreshed = refresh_project_status(
                context,
                runner=unavailable_runner,
            )

            self.assertEqual(refreshed.unknown_count, 1)
            self.assertEqual(
                load_execution_index(context).executions[0].status,
                "unknown",
            )
            self.assertEqual(len(refreshed.warnings), 2)
            self.assertTrue(
                any("squeue unavailable" in warning for warning in refreshed.warnings)
            )
            self.assertTrue(
                any("sacct unavailable" in warning for warning in refreshed.warnings)
            )

    def test_failed_afterok_dependency_marks_pending_stage_blocked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, plan = self._project_and_plan(temp_dir, ["debug", "config"])
            stages = list(plan.resolved_stages)
            stages[1] = stages[1].model_copy(update={"depends_on": ["debug"]})
            plan = plan.model_copy(update={"resolved_stages": stages})
            run = create_run_record(context, plan, command="sbt run debug config")
            submitted = submit_run(
                context,
                plan,
                run,
                runner=FakeSbatchRunner(["311", "312"]),
            )
            self.assertEqual(submitted.jobs[1].dependency_job_id, "311")

            def status_runner(arguments, **_kwargs):
                if arguments[0] == "squeue":
                    return subprocess.CompletedProcess(
                        arguments,
                        0,
                        stdout="312|PENDING|config|Dependency\n",
                        stderr="",
                    )
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    stdout="311|FAILED|debug|1:0\n312|PENDING|config|0:0\n",
                    stderr="",
                )

            report = inspect_run_status(context, run.run_dir, runner=status_runner)

            self.assertEqual(report.overall_status, "failed")
            self.assertEqual(
                {stage.stage: stage.status for stage in report.stages},
                {"debug": "failed", "config": "blocked"},
            )
            blocked = report.stages[1]
            self.assertEqual(blocked.source, "recorded dependency")
            self.assertIn("afterok dependency job 311 ended failed", blocked.detail)

            summaries = execution_summaries(context)
            self.assertEqual(
                {summary.stage: summary.status for summary in summaries},
                {"debug": "failed", "config": "blocked"},
            )
            self.assertIsNotNone(summaries[1].completed_at)

            def cancelled_runner(arguments, **_kwargs):
                if arguments[0] == "squeue":
                    return subprocess.CompletedProcess(
                        arguments,
                        0,
                        stdout="",
                        stderr="",
                    )
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    stdout="311|FAILED|debug|1:0\n312|CANCELLED|config|0:15\n",
                    stderr="",
                )

            cancelled_report = inspect_run_status(
                context,
                run.run_dir,
                runner=cancelled_runner,
            )
            self.assertEqual(cancelled_report.stages[1].status, "blocked")
            self.assertEqual(execution_summaries(context)[1].status, "blocked")

    def test_failed_member_of_multi_job_dependency_marks_stage_blocked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, plan = self._project_and_plan(
                temp_dir,
                ["debug", "config", "cox"],
            )
            stages = list(plan.resolved_stages)
            stages[2] = stages[2].model_copy(update={"depends_on": ["debug", "config"]})
            plan = plan.model_copy(update={"resolved_stages": stages})
            run = create_run_record(context, plan, command="sbt run debug config cox")
            submitted = submit_run(
                context,
                plan,
                run,
                runner=FakeSbatchRunner(["711", "712", "713"]),
            )
            self.assertEqual(submitted.jobs[2].dependency_job_id, "711:712")

            def status_runner(arguments, **_kwargs):
                if arguments[0] == "squeue":
                    return subprocess.CompletedProcess(
                        arguments,
                        0,
                        stdout="713|PENDING|cox|Dependency\n",
                        stderr="",
                    )
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    stdout=(
                        "711|COMPLETED|debug|0:0\n"
                        "712|FAILED|config|1:0\n"
                        "713|PENDING|cox|0:0\n"
                    ),
                    stderr="",
                )

            report = inspect_run_status(context, run.run_dir, runner=status_runner)

            self.assertEqual(report.stages[2].status, "blocked")
            self.assertIn("dependency job 712 ended failed", report.stages[2].detail)

    def test_logs_resolve_recorded_paths_and_tail_without_tree_scan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, plan = self._project_and_plan(temp_dir, ["debug"])
            run = create_run_record(context, plan, command="sbt run debug")
            submitted = submit_run(
                context,
                plan,
                run,
                runner=FakeSbatchRunner(["401"]),
            )
            stdout_path = submitted.jobs[0].stdout_log
            stdout_path.write_text("one\ntwo\nthree\n", encoding="utf-8")

            logs = resolve_run_logs(
                run.run_dir,
                stage="debug",
                include_stdout=True,
                include_stderr=False,
            )

            self.assertEqual(len(logs), 1)
            self.assertTrue(logs[0].exists)
            self.assertEqual(tail_text(stdout_path, 2), "two\nthree")

    def test_latest_run_is_scoped_to_each_project(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first_context, first_plan = self._project_and_plan(
                str(Path(temp_dir) / "first")
            )
            second_context, second_plan = self._project_and_plan(
                str(Path(temp_dir) / "second")
            )
            first_run = create_run_record(
                first_context,
                first_plan,
                command="first",
                run_id="20260101T000000Z-first",
            )
            second_run = create_run_record(
                second_context,
                second_plan,
                command="second",
                run_id="20260101T000000Z-second",
            )

            self.assertEqual(
                resolve_run_directory(first_context, "latest"), first_run.run_dir
            )
            self.assertEqual(
                resolve_run_directory(second_context, "latest"), second_run.run_dir
            )

    def test_end_to_end_dry_run_creates_no_run_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            initialize_project(root)
            (root / "IMC_files" / "case.mcd").write_bytes(b"x")
            runs_dir = root / ".sbt" / "runs"

            result = CliRunner().invoke(
                app,
                ["run", "segmentation", "--project", str(root), "--dry-run"],
            )

            self.assertEqual(result.exit_code, 0, result.stdout)
            self.assertIn("no run directory was created", result.stdout)
            self.assertEqual(list(runs_dir.iterdir()), [])

    def test_environment_override_is_visible_in_dry_run_without_scheduler_access(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            initialize_project(root)
            (root / "IMC_files" / "case.mcd").write_bytes(b"x")

            with patch(
                "SpatialBiologyToolkit.cli.main.active_project_dependencies",
                side_effect=AssertionError("dry runs must not query SLURM"),
            ):
                result = CliRunner().invoke(
                    app,
                    [
                        "run",
                        "prep",
                        "--project",
                        str(root),
                        "--environment",
                        "denoise",
                        "--dry-run",
                    ],
                )

            self.assertEqual(result.exit_code, 0, result.stdout)
            self.assertIn("Environment override: prep=denoise", result.stdout)
            self.assertIn("SBT_CONDA_ENV=sbt-denoise", result.stdout)

    def test_unknown_environment_override_stops_before_run_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            initialize_project(root)
            (root / "IMC_files" / "case.mcd").write_bytes(b"x")

            result = CliRunner().invoke(
                app,
                [
                    "run",
                    "prep",
                    "--project",
                    str(root),
                    "--environment",
                    "not-registered",
                    "--dry-run",
                ],
            )

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Unknown environment 'not-registered'", result.output)
            self.assertEqual(list((root / ".sbt" / "runs").iterdir()), [])

    def test_interactive_run_can_chain_to_active_project_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, first_plan = self._project_and_plan(temp_dir, ["debug"])
            first_run = create_run_record(
                context, first_plan, command="sbt run debug"
            )
            submit_run(
                context,
                first_plan,
                first_run,
                runner=FakeSbatchRunner(["831"]),
            )
            dependency = recorded_external_dependency(context, "001")
            submitted = SimpleNamespace(jobs=[])

            with (
                patch(
                    "SpatialBiologyToolkit.cli.main.active_project_dependencies",
                    return_value=[dependency],
                ),
                patch(
                    "SpatialBiologyToolkit.cli.main.refresh_external_dependency",
                    return_value=dependency,
                ),
                patch(
                    "SpatialBiologyToolkit.cli.main.submit_run",
                    return_value=submitted,
                ) as submit_mock,
            ):
                result = CliRunner().invoke(
                    app,
                    ["run", "debug", "--project", str(context.root)],
                    input="y\n",
                )

            self.assertEqual(result.exit_code, 0, result.stdout)
            self.assertIn("Active SBT executions for this project", result.stdout)
            self.assertIn("afterok dependency", result.stdout)
            self.assertEqual(
                submit_mock.call_args.kwargs["external_dependency"].job_id,
                "831",
            )

    def test_no_after_suppresses_active_job_discovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context, _plan = self._project_and_plan(temp_dir, ["debug"])
            submitted = SimpleNamespace(jobs=[])

            with (
                patch(
                    "SpatialBiologyToolkit.cli.main.active_project_dependencies",
                    side_effect=AssertionError("--no-after must suppress discovery"),
                ),
                patch(
                    "SpatialBiologyToolkit.cli.main.submit_run",
                    return_value=submitted,
                ),
            ):
                result = CliRunner().invoke(
                    app,
                    [
                        "run",
                        "debug",
                        "--project",
                        str(context.root),
                        "--no-after",
                    ],
                )

            self.assertEqual(result.exit_code, 0, result.stdout)

    def test_run_prompts_to_install_only_missing_managed_environments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            initialize_project(root)
            (root / "IMC_files" / "case.mcd").write_bytes(b"x")

            class FakeEnvironmentManager:
                installed = False
                synced: list[str] = []

                def required_for_stages(self, stages):
                    return [
                        EnvironmentSummary(
                            key="analysis",
                            conda_name="sbt-analysis",
                            managed=True,
                            exists=self.installed,
                            stages=list(stages),
                        )
                    ]

                def sync(self, key, **_kwargs):
                    self.synced.append(key)
                    self.installed = True

                def validate(self, _key):
                    return SimpleNamespace(valid=True, issues=[])

            manager = FakeEnvironmentManager()
            submitted = SimpleNamespace(jobs=[])
            with (
                patch(
                    "SpatialBiologyToolkit.cli.main._env_manager", return_value=manager
                ),
                patch(
                    "SpatialBiologyToolkit.cli.main.submit_run",
                    return_value=submitted,
                ) as submit_mock,
            ):
                result = CliRunner().invoke(
                    app,
                    ["run", "prep", "--project", str(root)],
                    input="y\n",
                )

            self.assertEqual(result.exit_code, 0, result.stdout)
            self.assertIn("sbt-analysis", result.stdout)
            self.assertIn("Install the missing environment(s) now", result.stdout)
            self.assertIn("[y/N]", result.stdout)
            self.assertEqual(manager.synced, ["analysis"])
            submit_mock.assert_called_once()

    def test_run_declining_environment_install_stops_before_run_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            initialize_project(root)
            (root / "IMC_files" / "case.mcd").write_bytes(b"x")

            manager = SimpleNamespace(
                required_for_stages=lambda _stages: [
                    EnvironmentSummary(
                        key="analysis",
                        conda_name="sbt-analysis",
                        managed=True,
                        exists=False,
                        stages=["prep"],
                    )
                ],
                validate=lambda _key: SimpleNamespace(valid=True, issues=[]),
            )
            with (
                patch(
                    "SpatialBiologyToolkit.cli.main._env_manager", return_value=manager
                ),
                patch("SpatialBiologyToolkit.cli.main.submit_run") as submit_mock,
            ):
                result = CliRunner().invoke(
                    app,
                    ["run", "prep", "--project", str(root)],
                    input="n\n",
                )

            self.assertNotEqual(result.exit_code, 0)
            output = result.stdout + result.stderr
            self.assertIn("sbt env sync analysis", output)
            self.assertIn("No run record was created", output)
            submit_mock.assert_not_called()
            self.assertEqual(list((root / ".sbt" / "runs").iterdir()), [])

    def test_run_missing_external_environment_stops_with_guidance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            context = initialize_project(root)
            assets = {
                asset.role: asset for asset in resolve_assets(context.config, root)
            }
            assets["anndata"].path.write_bytes(b"placeholder")

            manager = SimpleNamespace(
                required_for_stages=lambda _stages: [
                    EnvironmentSummary(
                        key="starling",
                        conda_name="sbt-starling",
                        managed=False,
                        exists=False,
                        stages=["starling"],
                    )
                ]
            )
            with (
                patch(
                    "SpatialBiologyToolkit.cli.main._env_manager", return_value=manager
                ),
                patch("SpatialBiologyToolkit.cli.main.submit_run") as submit_mock,
            ):
                result = CliRunner().invoke(
                    app,
                    ["run", "starling", "--project", str(root), "--no-deps"],
                )

            self.assertNotEqual(result.exit_code, 0)
            output = result.stdout + result.stderr
            self.assertIn("externally managed Conda environments are missing", output)
            self.assertIn("sbt env show <key>", output)
            submit_mock.assert_not_called()
            self.assertEqual(list((root / ".sbt" / "runs").iterdir()), [])

    def test_run_invalid_environment_spec_stops_before_prompt_or_run_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            initialize_project(root)
            (root / "IMC_files" / "case.mcd").write_bytes(b"x")

            manager = SimpleNamespace(
                required_for_stages=lambda _stages: [
                    EnvironmentSummary(
                        key="analysis",
                        conda_name="sbt-analysis",
                        managed=True,
                        exists=False,
                        stages=["prep"],
                    )
                ],
                validate=lambda _key: SimpleNamespace(
                    valid=False,
                    issues=[
                        SimpleNamespace(
                            severity="error",
                            message="The lockfile needs maintainer attention.",
                        )
                    ],
                ),
            )
            with (
                patch(
                    "SpatialBiologyToolkit.cli.main._env_manager", return_value=manager
                ),
                patch("SpatialBiologyToolkit.cli.main.submit_run") as submit_mock,
            ):
                result = CliRunner().invoke(
                    app,
                    ["run", "prep", "--project", str(root)],
                )

            self.assertNotEqual(result.exit_code, 0)
            output = result.stdout + result.stderr
            self.assertIn("specifications are invalid", output)
            self.assertIn("The lockfile needs maintainer attention", output)
            self.assertNotIn("Install the missing environment(s) now", output)
            submit_mock.assert_not_called()
            self.assertEqual(list((root / ".sbt" / "runs").iterdir()), [])

    def test_no_deps_dry_run_submits_only_requested_stage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            context = initialize_project(root)
            assets = {
                asset.role: asset for asset in resolve_assets(context.config, root)
            }
            assets["anndata"].path.parent.mkdir(parents=True, exist_ok=True)
            assets["anndata"].path.write_bytes(b"placeholder")
            for role in ("denoised_images", "masks"):
                assets[role].path.mkdir(parents=True, exist_ok=True)
                (assets[role].path / "placeholder.tif").write_bytes(b"placeholder")
            runs_dir = root / ".sbt" / "runs"

            result = CliRunner().invoke(
                app,
                [
                    "run",
                    "cellvision-full",
                    "--project",
                    str(root),
                    "--no-deps",
                    "--dry-run",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.stdout)
            self.assertIn("1. cellvision-full", result.stdout)
            self.assertNotIn("2. ", result.stdout)
            self.assertIn("job_cellvision_full.sh", result.stdout)
            self.assertNotIn("--dependency=afterok", result.stdout)
            self.assertEqual(list(runs_dir.iterdir()), [])

    def test_no_deps_stops_before_submission_when_stage_assets_are_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            initialize_project(root)

            with patch("SpatialBiologyToolkit.cli.main.submit_run") as submit_mock:
                result = CliRunner().invoke(
                    app,
                    [
                        "run",
                        "cellvision-full",
                        "--project",
                        str(root),
                        "--no-deps",
                    ],
                )

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("missing blocking project assets", result.stdout)
            submit_mock.assert_not_called()
            self.assertEqual(list((root / ".sbt" / "runs").iterdir()), [])

    def test_ignore_missing_assets_allows_chained_rapids_dry_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            context = initialize_project(root)
            runs_dir = root / ".sbt" / "runs"
            predecessor_plan = build_run_plan(context, ["debug"])
            predecessor_run = create_run_record(
                context,
                predecessor_plan,
                command="sbt run debug",
            )
            submit_run(
                context,
                predecessor_plan,
                predecessor_run,
                runner=FakeSbatchRunner(["851"]),
            )
            existing_run_dirs = list(runs_dir.iterdir())

            plan = build_run_plan(
                context,
                ["rapids"],
                dependency_policy="none",
                ignore_missing_assets=True,
            )

            self.assertTrue(plan.ready, plan.errors)
            self.assertTrue(plan.ignore_missing_assets)
            self.assertEqual(plan.resolved_stages[0].missing_assets, ["anndata"])
            self.assertTrue(
                any("--ignore-missing-assets" in warning for warning in plan.warnings)
            )

            result = CliRunner().invoke(
                app,
                [
                    "run",
                    "rapids",
                    "--project",
                    str(root),
                    "--environment",
                    "analysis",
                    "--no-deps",
                    "--after",
                    "001",
                    "--ignore-missing-assets",
                    "--dry-run",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.stdout)
            self.assertIn("Missing assets ignored: yes", result.stdout)
            self.assertIn("missing blocking project assets: anndata", result.stdout)
            self.assertIn("Plan is ready.", result.stdout)
            self.assertIn("--dependency=afterok:851", result.stdout)
            self.assertEqual(list(runs_dir.iterdir()), existing_run_dirs)

    def test_ignore_missing_assets_does_not_ignore_required_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = initialize_project(Path(temp_dir) / "project")

            plan = build_run_plan(
                context,
                ["nimbus"],
                dependency_policy="none",
                ignore_missing_assets=True,
            )

            self.assertFalse(plan.ready)
            self.assertTrue(plan.resolved_stages[0].missing_assets)
            self.assertTrue(plan.resolved_stages[0].missing_files)
            self.assertFalse(
                any("missing blocking project assets" in error for error in plan.errors)
            )
            self.assertTrue(
                any("missing required files" in error for error in plan.errors)
            )


if __name__ == "__main__":
    unittest.main()
