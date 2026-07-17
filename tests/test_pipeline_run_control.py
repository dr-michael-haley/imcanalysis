import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from SpatialBiologyToolkit.cli.main import app
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
            self.assertIn("--parsable", first_args)
            self.assertIn("--export=ALL", first_args)
            self.assertNotIn("--dependency=afterok:101", first_args)
            self.assertIn("--dependency=afterok:101", second_args)
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
            self.assertEqual(
                exported["SBT_CONDA_ENV_SEGMENTATION"], "imc_segmentation"
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


if __name__ == "__main__":
    unittest.main()
