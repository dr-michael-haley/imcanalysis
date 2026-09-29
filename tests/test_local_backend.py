"""Local control-plane checks plus real Linux subprocess lifecycle tests."""

import json
import os
from contextlib import contextmanager
import subprocess
import sys
import time
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from SpatialBiologyToolkit.cli.main import app
from SpatialBiologyToolkit.pipeline.commands import stage_commands
from SpatialBiologyToolkit.pipeline.control import run_preview_snapshot
from SpatialBiologyToolkit.pipeline.executions import load_execution_index
from SpatialBiologyToolkit.pipeline.local import (
    LOCAL_STATE,
    assert_no_orphan,
    inspect_local,
    launch_local,
    load_local_run,
    local_commands,
    prepare_local_run,
    request_cancel,
    wait_local,
)
from SpatialBiologyToolkit.pipeline.local_models import LocalCommand, ProcessIdentity
from SpatialBiologyToolkit.pipeline.local_process import alive, project_lock
from SpatialBiologyToolkit.pipeline.local_worker import run_worker
from SpatialBiologyToolkit.pipeline.logs import follow_logs, resolve_run_logs
from SpatialBiologyToolkit.pipeline.manifests import utc_now, write_yaml
from SpatialBiologyToolkit.pipeline.planner import build_run_plan
from SpatialBiologyToolkit.pipeline.project import initialize_project
from SpatialBiologyToolkit.pipeline.registry import STAGES
from SpatialBiologyToolkit.pipeline.runs import create_run_record
from SpatialBiologyToolkit.pipeline.runtime import sbt_environment
from SpatialBiologyToolkit.pipeline.status import (
    inspect_run_status,
    refresh_project_status,
)


@pytest.fixture
def project(tmp_path):
    context = initialize_project(tmp_path / "project with spaces")
    (context.root / "IMC_files" / "case.mcd").write_bytes(b"x")
    return context


def plan_for(context, targets=None):
    plan = build_run_plan(
        context,
        targets or ["prep", "vis"],
        backend="local",
        dependency_policy="none",
        ignore_missing_assets=True,
    )
    assert plan.ready, plan.errors
    return plan


def prepare(context, commands=None):
    plan = plan_for(context)
    run = create_run_record(context, plan, command="sbt run prep vis --backend local")
    if commands is None:
        commands = {
            stage.name: [
                LocalCommand(
                    argv=[sys.executable, "-c", f"print('{stage.name}', flush=True)"],
                    environment="test",
                )
            ]
            for stage in plan.resolved_stages
        }
    state = prepare_local_run(context, run, commands)
    return run, state


def test_all_python_stages_have_explicit_environment_mapping():
    for stage in STAGES:
        if stage.python_modules and stage.name != "slogs":
            assert [
                item.module for item in stage_commands(stage.name)
            ] == stage.python_modules
    assert [item.environment_key for item in stage_commands("cellpose")] == [
        "analysis",
        "cellposesam",
    ]
    assert [item.environment_key for item in stage_commands("cellvision-full")] == [
        "scportrait",
        "scportrait",
        "analysis",
        "scportrait",
    ]


def test_local_plan_does_not_require_slurm_wrappers(project, tmp_path):
    plan = build_run_plan(
        project, ["prep"], backend="local", toolkit_directory=tmp_path
    )
    assert plan.ready, plan.errors
    unsupported = build_run_plan(
        project, ["debug"], backend="local", dependency_policy="none"
    )
    assert not unsupported.ready
    assert any("no local scientific command" in item for item in unsupported.errors)


@pytest.mark.parametrize("backend", ["local", "slurm"])
def test_rapids_defaults_to_external_environment_in_mixed_workflow(project, backend):
    plan = build_run_plan(
        project,
        ["bbn", "rapids", "vis"],
        backend=backend,
        dependency_policy="none",
        ignore_missing_assets=True,
    )
    assert plan.ready, plan.errors
    run = create_run_record(project, plan, command="sbt run bbn rapids vis")
    expected = {"bbn": "sbt-analysis", "rapids": "rapids_singlecell", "vis": "sbt-analysis"}
    commands = local_commands(plan, validate=False)
    for stage, name in expected.items():
        exported = sbt_environment(project, run, stage)
        assert exported["SBT_CONDA_ENV"] == name
        assert "SBT_ENVIRONMENT_OVERRIDE" not in exported
        command = commands[stage][0]
        assert command.argv[command.argv.index("-n") + 1] == name
        assert command.variables["SBT_CONDA_ENV"] == name
    assert sbt_environment(project, run, "rapids")["SBT_ENVIRONMENT_KEY"] == "rapids"


def test_missing_external_rapids_does_not_fall_back_to_analysis(project, tmp_path):
    prefix = tmp_path / "sbt-analysis"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").touch()
    with (
        patch("SpatialBiologyToolkit.pipeline.local.find_conda_executable", return_value="conda"),
        patch(
            "SpatialBiologyToolkit.pipeline.local.conda_environment_names",
            return_value={"sbt-analysis": prefix},
        ),
        pytest.raises(ValueError, match="Python environment is missing: rapids_singlecell"),
    ):
        local_commands(plan_for(project, ["rapids"]))


def test_dry_run_uses_local_commands_and_creates_no_records(project):
    with patch(
        "SpatialBiologyToolkit.cli.main.submit_run",
        side_effect=AssertionError("SLURM called"),
    ):
        result = CliRunner().invoke(
            app,
            [
                "run",
                "prep",
                "--project",
                str(project.root),
                "--backend",
                "local",
                "--detach",
                "--dry-run",
                "--format",
                "json",
            ],
        )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["plan"]["execution_backend"] == "local"
    assert (
        data["commands"]["prep"][0]["argv"][-1]
        == "SpatialBiologyToolkit.scripts.preprocess"
    )
    assert not list(project.runs_dir.iterdir())
    assert not load_execution_index(project).executions


def test_backend_environment_default_and_incompatible_options(project):
    args = [
        "run",
        "prep",
        "--project",
        str(project.root),
        "--dry-run",
        "--format",
        "json",
    ]
    result = CliRunner().invoke(app, args, env={"SBT_BACKEND": "local"})
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["plan"]["execution_backend"] == "local"
    result = CliRunner().invoke(app, args + ["--backend", "slurm", "--detach"])
    assert result.exit_code != 0
    assert "require --backend local" in result.output


def test_backend_and_active_environment_bound_to_preview(project):
    plan = plan_for(project)
    first = run_preview_snapshot(project, plan)
    plan.execution_backend = "slurm_scripts"
    assert first != run_preview_snapshot(project, plan)
    plan.execution_backend = "local"
    plan.use_active_environment = True
    assert first != run_preview_snapshot(project, plan)


def test_active_environment_uses_activated_python_not_launcher(
    project, tmp_path, monkeypatch
):
    prefix = tmp_path / "scientific env"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").touch()
    monkeypatch.setenv("CONDA_PREFIX", str(prefix))
    plan = plan_for(project)
    plan.use_active_environment = True
    with patch(
        "SpatialBiologyToolkit.pipeline.local.conda_environment_names",
        side_effect=AssertionError("unexpected Conda query"),
    ):
        commands = local_commands(plan)
    assert commands["prep"][0].argv[0] == str(prefix / "bin" / "python")
    assert commands["prep"][0].variables["PYTHONUNBUFFERED"] == "1"


def test_missing_environment_fails_before_allocation(project):
    with (
        patch(
            "SpatialBiologyToolkit.pipeline.local.find_conda_executable",
            return_value="conda",
        ),
        patch(
            "SpatialBiologyToolkit.pipeline.local.conda_environment_names",
            return_value={},
        ),
    ):
        result = CliRunner().invoke(
            app, ["run", "prep", "--project", str(project.root), "--backend", "local"]
        )
    assert result.exit_code != 0
    assert "Python environment is missing" in result.output
    assert not load_execution_index(project).executions


def test_local_status_refresh_does_not_call_slurm(project):
    run, state = prepare(project)
    state.status = "completed"
    for stage in state.stages:
        stage.status = "completed"
        stage.started_at = stage.completed_at = utc_now()
    write_yaml(run.run_dir / LOCAL_STATE, state)

    def forbidden(*args, **kwargs):
        raise AssertionError("SLURM called for local run")

    assert (
        inspect_run_status(project, run.run_dir, runner=forbidden).overall_status
        == "completed"
    )
    refreshed = refresh_project_status(project, runner=forbidden)
    assert refreshed.unknown_count == 0
    assert [item.status for item in load_execution_index(project).executions] == [
        "completed",
        "completed",
    ]
    result = CliRunner().invoke(
        app,
        [
            "status",
            "latest",
            "--project",
            str(project.root),
            "--details",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"]["source"] == "local"


def test_stale_worker_is_unknown_and_foreign_pid_is_not_signalled(project):
    run, state = prepare(project)
    state.status = "running"
    state.worker = ProcessIdentity(
        pid=os.getpid(),
        start_ticks=0,
        boot_id="different-boot",
        hostname="different-host",
    )
    state.stages[0].status = "running"
    write_yaml(run.run_dir / LOCAL_STATE, state)
    assert not alive(state.worker)
    assert inspect_local(project, run.run_dir).overall_status == "unknown"
    with pytest.raises(RuntimeError, match="unavailable"):
        request_cancel(run.run_dir, reason="stop")
    assert not (run.run_dir / "local_cancel.yaml").exists()


def test_logs_merged_and_follow_drains_terminal_content(project):
    run, state = prepare(project)
    stage = state.stages[0]
    stage.log.write_text("first\nsecond\nthird\n", encoding="utf-8")
    state.status = "completed"
    stage.status = "completed"
    write_yaml(run.run_dir / LOCAL_STATE, state)
    records = resolve_run_logs(run.run_dir, stage="prep")
    assert len(records) == 1
    assert (
        resolve_run_logs(run.run_dir, stage="prep", include_stdout=False)[0].path
        == stage.log
    )
    output = "".join(
        follow_logs(project, run.run_dir, stage.technical_run_id, records, 2)
    )
    assert "second\nthird\n" in output
    assert "first\n" not in output


def test_follow_reads_new_file_from_beginning_even_with_zero_tail(project):
    run, state = prepare(project)
    state.status = "completed"
    state.stages[0].status = "completed"
    write_yaml(run.run_dir / LOCAL_STATE, state)
    records = resolve_run_logs(run.run_dir, stage="prep")
    stream = follow_logs(
        project, run.run_dir, state.stages[0].technical_run_id, records, 0
    )
    assert "prep" in next(stream)
    state.stages[0].log.write_text(
        "first output after following started\n", encoding="utf-8"
    )
    assert "first output" in "".join(stream)


def test_reused_pid_cannot_be_cancelled():
    from SpatialBiologyToolkit.pipeline.local_process import signal_worker

    old = ProcessIdentity(pid=1234, start_ticks=1, boot_id="test", hostname="test")
    reused = old.model_copy(update={"start_ticks": 2})
    with (
        patch(
            "SpatialBiologyToolkit.pipeline.local_process.same_host", return_value=True
        ),
        patch(
            "SpatialBiologyToolkit.pipeline.local_process.identity", return_value=reused
        ),
        patch(
            "SpatialBiologyToolkit.pipeline.local_process.signal.pidfd_send_signal",
            create=True,
        ) as signal_pid,
    ):
        with pytest.raises(RuntimeError, match="no process was signalled"):
            signal_worker(old)
    signal_pid.assert_not_called()


def test_resource_inspection_handles_missing_gpu_tools():
    from SpatialBiologyToolkit.pipeline.local_process import resource_snapshot

    process = ProcessIdentity(pid=1234, start_ticks=1, boot_id="test", hostname="test")
    with (
        patch("SpatialBiologyToolkit.pipeline.local_process.alive", return_value=True),
        patch(
            "SpatialBiologyToolkit.pipeline.local_process.group_stats",
            side_effect=[{1234: (10, 100)}, {1234: (20, 120)}],
        ),
        patch(
            "SpatialBiologyToolkit.pipeline.local_process.os.sysconf",
            side_effect=lambda key: 100 if key == "SC_CLK_TCK" else 4096,
            create=True,
        ),
        patch(
            "SpatialBiologyToolkit.pipeline.local_process.subprocess.run",
            side_effect=FileNotFoundError("nvidia-smi"),
        ),
    ):
        resources = resource_snapshot(process)
    assert resources["cpu_percent"] > 0
    assert resources["rss_mib"] == round(120 * 4096 / 2**20, 1)
    assert "unavailable" in resources["gpu_note"]


def test_cancel_preview_targets_workflow_even_for_pending_stage(project):
    run, state = prepare(project)
    args = [
        "cancel",
        "2",
        "--project",
        str(project.root),
        "--reason",
        "stop",
        "--format",
        "json",
    ]
    preview = CliRunner().invoke(app, args + ["--dry-run"])
    assert preview.exit_code == 0, preview.output
    data = json.loads(preview.output)
    assert data["affected_executions"] == [1, 2]
    with patch(
        "SpatialBiologyToolkit.cli.local.request_cancel", return_value=state
    ) as cancel:
        result = CliRunner().invoke(app, args + ["--plan-token", data["preview_token"]])
    assert result.exit_code == 0, result.output
    cancel.assert_called_once_with(run.run_dir, reason="stop")
    assert (run.run_dir / "local_cancellation_audit.json").is_file()


def test_launch_failure_blocks_allocated_stages_and_updates_reports(project):
    run, _ = prepare(project)
    with patch(
        "SpatialBiologyToolkit.pipeline.local.subprocess.Popen",
        side_effect=OSError("cannot spawn"),
    ):
        with pytest.raises(RuntimeError, match="could not be started"):
            launch_local(project, run, 123)
    state = load_local_run(run.run_dir)
    assert state.status == "failed"
    assert [item.status for item in state.stages] == ["blocked", "blocked"]
    assert [item.status for item in load_execution_index(project).executions] == [
        "blocked",
        "blocked",
    ]
    report = project.root / run.executions[1].output_folder / "README.md"
    assert "cannot spawn" in report.read_text(encoding="utf-8")


def test_ctrl_c_during_startup_requests_cleanup(project):
    from unittest.mock import Mock

    run, _ = prepare(project)
    child = Mock()
    child.poll.side_effect = KeyboardInterrupt
    child.wait.return_value = 1
    with patch(
        "SpatialBiologyToolkit.pipeline.local.subprocess.Popen", return_value=child
    ) as spawn:
        with pytest.raises(KeyboardInterrupt):
            launch_local(project, run, 123)
    assert (run.run_dir / "local_cancel.yaml").is_file()
    child.wait.assert_called_once_with(timeout=10)
    assert spawn.call_args.kwargs["start_new_session"] is True
    assert spawn.call_args.kwargs["pass_fds"] == (123,)
    assert spawn.call_args.kwargs["stdin"] == subprocess.DEVNULL


def test_used_local_preview_is_idempotent(project):
    from SpatialBiologyToolkit.pipeline.control import (
        canonical_digest,
        preview_run_identities,
    )

    args = [
        "run",
        "prep",
        "--project",
        str(project.root),
        "--backend",
        "local",
        "--format",
        "json",
    ]
    preview = CliRunner().invoke(app, args + ["--dry-run"])
    assert preview.exit_code == 0, preview.output
    token = json.loads(preview.stdout)["preview_token"]
    workflow_id, technical_ids = preview_run_identities(token, 1)
    plan = build_run_plan(project, ["prep"], backend="local")
    create_run_record(
        project,
        plan,
        command="test",
        run_id=workflow_id,
        technical_run_ids=technical_ids,
        plan_token_digest=canonical_digest(token),
    )
    with patch(
        "SpatialBiologyToolkit.cli.local.launch_local",
        side_effect=AssertionError("duplicate launch"),
    ):
        replay = CliRunner().invoke(app, args + ["--plan-token", token])
    assert replay.exit_code == 0, replay.output
    assert json.loads(replay.stdout)["idempotent_replay"] is True
    assert len(load_execution_index(project).executions) == 1


def test_modified_local_plan_rejects_preview_before_allocation(project):
    args = [
        "run",
        "prep",
        "--project",
        str(project.root),
        "--backend",
        "local",
        "--format",
        "json",
    ]
    preview = CliRunner().invoke(app, args + ["--dry-run"])
    token = json.loads(preview.stdout)["preview_token"]
    result = CliRunner().invoke(
        app, args + ["--plan-token", token, "--note", "changed"]
    )
    assert result.exit_code != 0
    assert "changed after preview" in result.output
    assert not load_execution_index(project).executions


def test_foreground_json_is_clean_and_propagates_failure(project):
    @contextmanager
    def lock(_directory):
        with (project.state_dir / "fake-launch-lock").open("w") as handle:
            yield handle

    def launch(_context, run, _fd):
        state = load_local_run(run.run_dir)
        state.worker = ProcessIdentity(
            pid=123, start_ticks=1, boot_id="test", hostname="test"
        )
        state.status = "failed"
        write_yaml(run.run_dir / LOCAL_STATE, state)
        return state

    def wait(_directory, *, emit):
        emit("scientific output belongs on stderr\n")
        return 1

    commands = {
        "prep": [LocalCommand(argv=[sys.executable, "-c", "pass"], environment="test")]
    }
    with (
        patch("SpatialBiologyToolkit.cli.local.require_linux"),
        patch("SpatialBiologyToolkit.cli.local.project_lock", lock),
        patch("SpatialBiologyToolkit.cli.local.local_commands", return_value=commands),
        patch("SpatialBiologyToolkit.cli.local.launch_local", launch),
        patch("SpatialBiologyToolkit.cli.local.wait_local", wait),
    ):
        result = CliRunner().invoke(
            app,
            [
                "run",
                "prep",
                "--project",
                str(project.root),
                "--backend",
                "local",
                "--format",
                "json",
            ],
        )
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["status"] == "failed"
    assert "scientific output" not in result.stdout
    assert "scientific output" in result.stderr


def test_removal_cannot_renumber_outputs_during_local_run(project):
    from SpatialBiologyToolkit.pipeline.executions import (
        remove_executions,
        update_execution,
        ExecutionLayoutError,
    )

    run, state = prepare(project)
    update_execution(project, state.stages[0].technical_run_id, status="completed")
    with patch("SpatialBiologyToolkit.pipeline.local.alive", return_value=True):
        with pytest.raises(ExecutionLayoutError, match="active local workflow"):
            remove_executions(
                project,
                [state.stages[0].technical_run_id],
                reason="test",
                confirmation_mode="non_interactive",
            )


def test_status_keeps_correct_display_ids_after_removal(project):
    from SpatialBiologyToolkit.pipeline.executions import remove_executions

    run, state = prepare(project)
    state.status = "completed"
    for stage in state.stages:
        stage.status = "completed"
    write_yaml(run.run_dir / LOCAL_STATE, state)
    inspect_local(project, run.run_dir)
    remove_executions(
        project,
        [state.stages[0].technical_run_id],
        reason="test",
        confirmation_mode="non_interactive",
    )
    report = inspect_local(project, run.run_dir)
    assert [(item.execution_id, item.stage) for item in report.stages] == [(1, "vis")]


@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
def test_worker_sequences_commands_and_finalizes_reports(project, outcome):
    run, state = prepare(project)
    calls = []
    original_popen = subprocess.Popen

    class Child:
        pid = 1234
        returncode = 0

        def __init__(self, argv, **kwargs):
            calls.append(argv)
            kwargs["stdout"].write("synthetic scientific output\n")
            self.returncode = 7 if outcome == "failed" else 0

    def launch(argv, **kwargs):
        if argv[0] == sys.executable:
            return Child(argv, **kwargs)
        return original_popen(argv, **kwargs)

    def wait_for_child(*args):
        if outcome == "cancelled":
            write_yaml(
                run.run_dir / "local_cancel.yaml", {"reason": "test cancellation"}
            )
        return object()

    fake_identity = ProcessIdentity(
        pid=1234, start_ticks=1, boot_id="test", hostname="test"
    )
    with (run.run_dir / "fake-lock").open("w") as lock:
        fd = os.dup(lock.fileno())
        with (
            patch(
                "SpatialBiologyToolkit.pipeline.local_worker.subprocess.Popen", launch
            ),
            patch(
                "SpatialBiologyToolkit.pipeline.local_worker.identity",
                return_value=fake_identity,
            ),
            patch("SpatialBiologyToolkit.pipeline.local_worker.signal.signal"),
            patch(
                "SpatialBiologyToolkit.pipeline.local_worker.signal.SIGHUP",
                1,
                create=True,
            ),
            patch(
                "SpatialBiologyToolkit.pipeline.local_worker.os.waitid",
                side_effect=wait_for_child,
                create=True,
            ),
            patch(
                "SpatialBiologyToolkit.pipeline.local_worker.os.P_PID", 1, create=True
            ),
            patch(
                "SpatialBiologyToolkit.pipeline.local_worker.os.WEXITED", 4, create=True
            ),
            patch(
                "SpatialBiologyToolkit.pipeline.local_worker.os.WNOHANG", 1, create=True
            ),
            patch(
                "SpatialBiologyToolkit.pipeline.local_worker.os.WNOWAIT",
                0x1000000,
                create=True,
            ),
            patch("SpatialBiologyToolkit.pipeline.local_worker.stop_child"),
        ):
            code = run_worker(run.run_dir, fd)
    assert code == (0 if outcome == "completed" else 1)
    assert len(calls) == (2 if outcome == "completed" else 1)
    expected = [outcome, "blocked" if outcome == "failed" else outcome]
    assert [stage.status for stage in load_local_run(run.run_dir).stages] == expected
    assert [
        execution.status for execution in load_execution_index(project).executions
    ] == expected
    assert "synthetic scientific output" in state.stages[0].log.read_text()
    assert (project.root / run.executions[0].output_folder / "README.md").exists()


linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="requires Linux /proc, flock and process groups",
)


@linux_only
def test_real_worker_detaches_and_finishes(project):
    run, _ = prepare(project)
    with project_lock(project.state_dir) as lock:
        state = launch_local(project, run, lock.fileno())
        assert state.worker.pid != os.getpid()
        assert os.getsid(state.worker.pid) == state.worker.pid
    chunks = []
    assert wait_local(run.run_dir, emit=chunks.append) == 0
    assert "prep" in "".join(chunks)
    assert load_local_run(run.run_dir).status == "completed"


@linux_only
def test_real_cancel_kills_descendants_and_blocks_overlap(project):
    script = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); "
        "print(p.pid,flush=True); time.sleep(120)"
    )
    commands = {
        name: [
            LocalCommand(argv=[sys.executable, "-u", "-c", script], environment="test")
        ]
        for name in ["prep", "vis"]
    }
    run, _ = prepare(project, commands)
    with project_lock(project.state_dir) as lock:
        launch_local(project, run, lock.fileno())
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            state = load_local_run(run.run_dir)
            lines = (
                state.stages[0].log.read_text().splitlines()
                if state.stages[0].log.exists()
                else []
            )
            pids = [int(line) for line in lines if line.isdigit()]
            if pids:
                break
            time.sleep(0.1)
        assert pids
        from SpatialBiologyToolkit.pipeline.local_process import identity

        descendant = identity(pids[0])
        with pytest.raises(RuntimeError, match="already running"):
            with project_lock(project.state_dir):
                pass
        with pytest.raises(RuntimeError, match="live processes"):
            assert_no_orphan(project)
        request_cancel(run.run_dir, reason="test cancellation")
        assert wait_local(run.run_dir, emit=lambda _: None) == 130
        assert not alive(descendant)
        assert [item.status for item in load_local_run(run.run_dir).stages] == [
            "cancelled",
            "cancelled",
        ]
    finally:
        request_cancel(run.run_dir, reason="test cleanup")
