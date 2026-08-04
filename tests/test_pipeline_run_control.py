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
from SpatialBiologyToolkit.pipeline.executions import execution_summaries
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
from SpatialBiologyToolkit.pipeline.slurm import SubmissionError, submit_run
from SpatialBiologyToolkit.pipeline.status import inspect_run_status
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
            self.assertEqual(exported["SBT_ENVIRONMENT_KEY"], "segmentation")
            self.assertEqual(exported["SBT_CONDA_ENV"], "imc_segmentation")
            self.assertEqual(exported["SBT_CONDA_ENV_SEGMENTATION"], "imc_segmentation")

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
            self.assertEqual(cancelled_report.stages[1].status, "cancelled")
            self.assertEqual(execution_summaries(context)[1].status, "cancelled")

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
                            key="segmentation",
                            conda_name="imc_segmentation",
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
            self.assertIn("imc_segmentation", result.stdout)
            self.assertIn("Install the missing environment(s) now", result.stdout)
            self.assertIn("[y/N]", result.stdout)
            self.assertEqual(manager.synced, ["segmentation"])
            submit_mock.assert_called_once()

    def test_run_declining_environment_install_stops_before_run_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            initialize_project(root)
            (root / "IMC_files" / "case.mcd").write_bytes(b"x")

            manager = SimpleNamespace(
                required_for_stages=lambda _stages: [
                    EnvironmentSummary(
                        key="segmentation",
                        conda_name="imc_segmentation",
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
            self.assertIn("sbt env sync segmentation", output)
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
                        key="rapids",
                        conda_name="rapids_singlecell",
                        managed=False,
                        exists=False,
                        stages=["rapids"],
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
                    ["run", "rapids", "--project", str(root), "--no-deps"],
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
                        key="segmentation",
                        conda_name="imc_segmentation",
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


if __name__ == "__main__":
    unittest.main()
