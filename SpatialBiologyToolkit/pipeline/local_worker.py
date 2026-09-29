"""Internal detached worker. Start through ``sbt run --backend local``."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from .executions import update_execution
from .local import CANCEL_REQUEST, LOCAL_STATE, TERMINAL, load_local_run
from .local_process import identity, stop_child
from .manifests import utc_now, write_yaml
from .models import RunStatus, StageStatus
from .project import load_project
from .runs import load_run_manifest


@contextmanager
def reporting_environment(variables):
    previous = dict(os.environ)
    os.environ.update(variables)
    for key in ("SLURM_JOB_ID", "SBT_SLURM_JOB_ID", "IMC_SLURM_JOB_ID"):
        os.environ.pop(key, None)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def report_stage(stage, *, final=False):
    from SpatialBiologyToolkit.reporting.reporter import StageReporter

    with reporting_environment(
        {
            **stage.commands[0].variables,
            "SBT_STAGE_STARTED_AT": stage.started_at.isoformat(),
        }
    ):
        reporter = StageReporter.from_environment(stage.stage)
        reporter.__enter__()
        if final:
            if stage.detail:
                reporter.add_note(stage.detail)
            reporter.finalize(
                status=stage.status,
                error=RuntimeError(stage.detail) if stage.status == "failed" else None,
            )


def run_worker(directory: Path, lock_fd: int) -> int:
    # Verify and retain the inherited project lock for this worker's lifetime.
    os.fstat(lock_fd)
    state = load_local_run(directory)
    manifest = load_run_manifest(directory)
    context = load_project(
        manifest.project_root, config_override=manifest.resolved_config
    )
    cancelled = False

    def on_signal(_signum, _frame):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def cancellation_requested():
        return cancelled or (directory / CANCEL_REQUEST).exists()

    def save():
        write_yaml(directory / LOCAL_STATE, state)
        write_yaml(
            directory / "status.yaml",
            RunStatus(
                run_id=state.workflow_run_id,
                workflow_run_id=state.workflow_run_id,
                project_id=manifest.project_id,
                checked_at=utc_now(),
                overall_status=state.status,
                stages=[
                    StageStatus(
                        stage=item.stage,
                        execution_id=item.execution_id,
                        technical_run_id=item.technical_run_id,
                        job_id=None,
                        status=item.status,
                        source="local",
                        detail=item.detail,
                        process_id=item.process.pid if item.process else None,
                        environment=item.commands[item.command_index].environment,
                        started_at=item.started_at,
                    )
                    for item in state.stages
                ],
                warnings=[state.error] if state.error else [],
            ),
        )

    state.worker = identity(os.getpid())
    state.status = "running"
    save()  # Startup acknowledgement: handlers, config and lock are ready.
    child = None
    current = None
    try:
        for stage in state.stages:
            if cancellation_requested() or state.status in {"failed", "cancelled"}:
                stage.status = "cancelled" if cancellation_requested() else "blocked"
                stage.detail = (
                    "Not started: workflow cancelled."
                    if cancellation_requested()
                    else "Not started: an earlier stage failed."
                )
                stage.completed_at = utc_now()
                update_execution(
                    context,
                    stage.technical_run_id,
                    status=stage.status,
                    completed_at=stage.completed_at,
                    asset_effect="none",
                )
                from SpatialBiologyToolkit.reporting.render import (
                    finalize_unstarted_execution,
                )

                finalize_unstarted_execution(
                    context,
                    Path(stage.commands[0].variables["SBT_OUTPUT_DIR"]),
                    stage.technical_run_id,
                    directory,
                    status=stage.status,
                    detail=stage.detail,
                )
                save()
                continue
            current = stage
            stage.started_at = utc_now()
            stage.status = "running"
            update_execution(
                context,
                stage.technical_run_id,
                status="running",
                started_at=stage.started_at,
            )
            save()
            report_stage(stage)
            with stage.log.open("a", encoding="utf-8", buffering=1) as log:
                log.write(
                    f"[{stage.started_at.isoformat()}] Starting {stage.execution_id:03d} {stage.stage}\n"
                )
                for index, command in enumerate(stage.commands):
                    if cancellation_requested():
                        break
                    stage.command_index = index
                    environment = {
                        **os.environ,
                        **command.variables,
                        "SBT_STAGE_STARTED_AT": stage.started_at.isoformat(),
                    }
                    for key in ("SLURM_JOB_ID", "SBT_SLURM_JOB_ID", "IMC_SLURM_JOB_ID"):
                        environment.pop(key, None)
                    log.write(
                        f"Environment: {command.environment}\nCommand: {command.argv!r}\n"
                    )
                    child = subprocess.Popen(
                        command.argv,
                        cwd=context.root,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    stage.process = identity(child.pid)
                    log.write(f"Process group: {child.pid}; output: {stage.log}\n")
                    save()
                    # WNOWAIT leaves the group leader unreaped, preventing PID
                    # reuse while descendants are terminated (including after failure).
                    while not cancellation_requested():
                        exited = os.waitid(
                            os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                        )
                        if exited is not None:
                            break
                        time.sleep(0.2)
                    stop_child(child)
                    stage.exit_code = child.returncode
                    child = None
                    if stage.exit_code or cancellation_requested():
                        break
                stage.status = (
                    "cancelled"
                    if cancellation_requested()
                    else ("failed" if stage.exit_code else "completed")
                )
                stage.completed_at = utc_now()
                stage.detail = f"{stage.status}; exit={stage.exit_code}"
                log.write(f"[{stage.completed_at.isoformat()}] {stage.detail}\n")
            report_stage(stage, final=True)
            update_execution(
                context,
                stage.technical_run_id,
                status=stage.status,
                started_at=stage.started_at,
                completed_at=stage.completed_at,
            )
            if stage.status in {"failed", "cancelled"}:
                state.status = stage.status
            save()
        if cancellation_requested():
            state.status = "cancelled"
        elif state.status == "running":
            state.status = "completed"
    except BaseException as exc:
        if child is not None:
            stop_child(child)
        state.status = "cancelled" if cancellation_requested() else "failed"
        state.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        for stage in state.stages:
            if stage.status not in TERMINAL or stage is current:
                stage.status = (
                    state.status
                    if stage is current
                    else ("cancelled" if state.status == "cancelled" else "blocked")
                )
                stage.detail = state.error
                stage.completed_at = utc_now()
                update_execution(
                    context,
                    stage.technical_run_id,
                    status=stage.status,
                    completed_at=stage.completed_at,
                )
                if stage is current and stage.started_at:
                    try:
                        report_stage(stage, final=True)
                    except Exception:
                        traceback.print_exc()
                else:
                    try:
                        from SpatialBiologyToolkit.reporting.render import (
                            finalize_unstarted_execution,
                        )

                        finalize_unstarted_execution(
                            context,
                            Path(stage.commands[0].variables["SBT_OUTPUT_DIR"]),
                            stage.technical_run_id,
                            directory,
                            status=stage.status,
                            detail=f"Not started: {state.error}",
                        )
                    except Exception:
                        traceback.print_exc()
    finally:
        save()
        os.close(lock_fd)
    return 0 if state.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(run_worker(Path(sys.argv[1]), int(sys.argv[2])))
