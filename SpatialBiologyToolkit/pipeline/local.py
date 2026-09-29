"""Sequential Linux backend using the shared SBT plan and execution records.

The worker is the only writer of local state after launch. Controllers communicate
through a cancellation request; monitoring never overwrites a live worker's state.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

from SpatialBiologyToolkit.environments.registry import load_environment_registry
from SpatialBiologyToolkit.environments.runtime import (
    conda_environment_names,
    find_conda_executable,
)

from .commands import stage_commands
from .executions import load_execution_index, update_execution
from .local_models import LocalCommand, LocalRun, LocalStage
from .local_process import alive, group_stats, same_host, signal_worker
from .manifests import read_model, utc_now, write_yaml
from .models import RunPlan, RunStatus, StageStatus
from .project import ProjectContext
from .runs import RunRecord, load_run_manifest
from .runtime import sbt_environment

LOCAL_STATE = "local_run.yaml"
CANCEL_REQUEST = "local_cancel.yaml"
TERMINAL = {"completed", "failed", "cancelled", "blocked"}


def load_local_run(directory: Path) -> LocalRun:
    return read_model(directory / LOCAL_STATE, LocalRun)


def local_commands(
    plan: RunPlan, *, validate: bool = True
) -> dict[str, list[LocalCommand]]:
    """Resolve each interpreter before allocating a run; never install environments."""
    registry = load_environment_registry()
    conda = find_conda_executable()
    prefixes = {}
    active = os.environ.get("CONDA_PREFIX")
    if plan.use_active_environment:
        if not active:
            raise ValueError(
                "--use-active-env requires an activated Conda environment (CONDA_PREFIX)."
            )
    elif validate:
        if not conda:
            raise ValueError(
                "Conda was not found. Activate Conda or set CONDA_EXE to its executable."
            )
        prefixes = conda_environment_names(conda)
    commands = {}
    for stage in plan.resolved_stages:
        commands[stage.name] = []
        for spec in stage_commands(stage.name):
            key = plan.environment_overrides.get(stage.name, spec.environment_key)
            name = registry.environments[key].conda_name
            prefix = Path(active) if plan.use_active_environment else prefixes.get(name)
            python = prefix / "bin" / "python" if prefix else None
            if validate and (python is None or not python.is_file()):
                raise ValueError(
                    f"Python environment is missing: {active or name}. Use sbt env to prepare it first."
                )
            variables = {
                "PYTHONUNBUFFERED": "1",
                "MPLBACKEND": "Agg",
                "QT_QPA_PLATFORM": "offscreen",
                "SBT_ENVIRONMENT_KEY": "active" if plan.use_active_environment else key,
                "SBT_CONDA_ENV": str(prefix) if plan.use_active_environment else name,
                "SBT_EXECUTION_BACKEND": "local",
            }
            if stage.name in {
                "cell2location",
                "maxfuse",
                "popqc",
                "spatialdata",
                "neighsig",
                "cellfeat",
            }:
                for variable in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    variables[variable] = os.environ.get(variable, "1")
            if stage.name == "cell2location":
                variables["PYTHONNOUSERSITE"] = "1"
            # Match the native library selection used by the corresponding wrappers.
            if (
                key in {"analysis", "cellposesam", "starling", "cell2location"}
                and prefix
            ):
                variables["LD_LIBRARY_PATH"] = str(prefix / "lib") + (
                    ":" + os.environ["LD_LIBRARY_PATH"]
                    if os.environ.get("LD_LIBRARY_PATH")
                    else ""
                )
            if plan.use_active_environment:
                argv = [str(python), "-u", "-m", spec.module]
            else:
                selection = ["-p", str(prefix)] if prefix else ["-n", name]
                # Apply the target's C++ libraries after Conda activation, so
                # Conda's own interpreter does not load a different env's libs.
                library = variables.pop("LD_LIBRARY_PATH", None)
                native_env = ["env", f"LD_LIBRARY_PATH={library}"] if library else []
                argv = [
                    conda or "conda",
                    "run",
                    "--no-capture-output",
                    *selection,
                    *native_env,
                    "python",
                    "-u",
                    "-m",
                    spec.module,
                ]
            commands[stage.name].append(
                LocalCommand(
                    argv=argv, environment=str(prefix or name), variables=variables
                )
            )
    return commands


def prepare_local_run(
    context: ProjectContext, run: RunRecord, commands: dict[str, list[LocalCommand]]
) -> LocalRun:
    stages = []
    for stage in run.plan.resolved_stages:
        execution = run.execution_for_stage(stage.name)
        exported = sbt_environment(context, run, stage.name)
        stages.append(
            LocalStage(
                stage=stage.name,
                execution_id=execution.execution_id,
                technical_run_id=execution.technical_run_id,
                commands=[
                    command.model_copy(
                        update={"variables": {**exported, **command.variables}}
                    )
                    for command in commands[stage.name]
                ],
                log=run.run_dir / "logs" / f"{stage.name}.log",
            )
        )
    state = LocalRun(workflow_run_id=run.workflow_run_id, stages=stages)
    write_yaml(run.run_dir / LOCAL_STATE, state)
    from SpatialBiologyToolkit.reporting.render import prepare_execution_output

    for stage in state.stages:
        execution = update_execution(context, stage.technical_run_id, status="pending")
        prepare_execution_output(context, run, execution)
    return state


def assert_no_orphan(context: ProjectContext) -> None:
    """The lock covers live workers; this also catches children of a killed worker."""
    for path in context.runs_dir.glob(f"*/{LOCAL_STATE}"):
        state = load_local_run(path.parent)
        if alive(state.worker) or any(
            stage.process
            and same_host(stage.process)
            and (
                alive(stage.process)
                or (stage.status not in TERMINAL and group_stats(stage.process.pid))
            )
            for stage in state.stages
        ):
            raise RuntimeError(
                f"Local workflow {state.workflow_run_id} still has live processes. Inspect/cancel it first."
            )


def fail_local_start(context: ProjectContext, run: RunRecord, detail: str) -> None:
    """Persist a launch failure only when no worker is alive to own the record."""
    from SpatialBiologyToolkit.reporting.render import finalize_unstarted_execution

    state = load_local_run(run.run_dir)
    state.status = "failed"
    state.error = detail
    for stage in state.stages:
        stage.status = "blocked"
        stage.detail = detail
        stage.completed_at = utc_now()
        update_execution(
            context,
            stage.technical_run_id,
            status="blocked",
            completed_at=stage.completed_at,
            asset_effect="none",
        )
        finalize_unstarted_execution(
            context,
            Path(stage.commands[0].variables["SBT_OUTPUT_DIR"]),
            stage.technical_run_id,
            run.run_dir,
            status="blocked",
            detail=detail,
        )
    write_yaml(run.run_dir / LOCAL_STATE, state)
    write_yaml(
        run.run_dir / "status.yaml", inspect_local(context, run.run_dir, persist=False)
    )


def launch_local(context: ProjectContext, run: RunRecord, lock_fd: int) -> LocalRun:
    environment = dict(os.environ)
    # The control worker runs in the launcher interpreter, independent of the stage env.
    root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
    try:
        with (run.run_dir / "logs" / "worker.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "SpatialBiologyToolkit.pipeline.local_worker",
                    str(run.run_dir),
                    str(lock_fd),
                ],
                cwd=context.root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock_fd,),
            )
    except OSError as exc:
        fail_local_start(context, run, f"Worker could not be started: {exc}")
        raise RuntimeError(
            f"Worker could not be started; see {run.run_dir}: {exc}"
        ) from exc
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            state = load_local_run(run.run_dir)
            if state.worker is not None:
                return state
            if child.poll() is not None:
                fail_local_start(
                    context,
                    run,
                    f"Local worker exited during startup (exit {child.returncode}); inspect logs/worker.log.",
                )
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        write_yaml(
            run.run_dir / CANCEL_REQUEST,
            {"reason": "Interrupted during startup", "at": utc_now()},
        )
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(
                f"Cancellation remains requested; inspect {run.run_dir} for cleanup status.",
                file=sys.stderr,
            )
        raise
    # A delayed worker observes this before starting any scientific command.
    write_yaml(
        run.run_dir / CANCEL_REQUEST,
        {"reason": "Worker startup was not acknowledged", "at": utc_now()},
    )
    raise RuntimeError(
        f"Worker startup was not acknowledged. Inspect {run.run_dir / 'logs' / 'worker.log'}; "
        "a cancellation request has been saved."
    )


def inspect_local(
    context: ProjectContext, directory: Path, *, persist: bool = True
) -> RunStatus:
    manifest = load_run_manifest(directory)
    if not (directory / LOCAL_STATE).exists():
        return RunStatus(
            run_id=manifest.run_id,
            project_id=manifest.project_id,
            checked_at=utc_now(),
            overall_status="unknown",
            stages=[],
            warnings=["Local worker record is missing."],
        )
    state = load_local_run(directory)
    worker_live = alive(state.worker)
    stale = state.status not in TERMINAL and not worker_live
    statuses = []
    executions = {
        item.technical_run_id: item for item in load_execution_index(context).executions
    }
    for stage in state.stages:
        execution = executions.get(stage.technical_run_id)
        if execution is None:
            continue  # Removed executions remain only in the immutable run evidence.
        status = "unknown" if stale and stage.status not in TERMINAL else stage.status
        elapsed = (
            ((stage.completed_at or utc_now()) - stage.started_at).total_seconds()
            if stage.started_at
            else None
        )
        statuses.append(
            StageStatus(
                stage=stage.stage,
                execution_id=execution.execution_id,
                technical_run_id=stage.technical_run_id,
                job_id=None,
                status=status,
                source="local",
                process_id=stage.process.pid if stage.process else None,
                environment=stage.commands[stage.command_index].environment,
                started_at=stage.started_at,
                elapsed_seconds=elapsed,
                detail=stage.detail
                or (
                    "Worker unavailable on this host/boot; completion is unverified."
                    if stale
                    else None
                ),
            )
        )
        # A live worker owns the index. Refreshing it here would race the next stage.
        if persist and not worker_live:
            try:
                update_execution(
                    context,
                    stage.technical_run_id,
                    status=status,
                    started_at=stage.started_at,
                    completed_at=stage.completed_at,
                )
            except FileNotFoundError:
                pass
    return RunStatus(
        run_id=manifest.run_id,
        workflow_run_id=manifest.run_id,
        project_id=manifest.project_id,
        checked_at=utc_now(),
        overall_status="unknown" if stale else state.status,
        stages=statuses,
        warnings=[state.error] if state.error else [],
    )


def request_cancel(directory: Path, *, reason: str) -> LocalRun:
    state = load_local_run(directory)
    if state.status in TERMINAL:
        return state
    if not state.worker or not alive(state.worker):
        raise RuntimeError(
            "The worker is unavailable on this host/boot. Cancellation cannot be verified; no PID was signalled."
        )
    write_yaml(
        directory / CANCEL_REQUEST,
        {"reason": reason, "at": utc_now(), "worker": state.worker.model_dump()},
    )
    try:
        signal_worker(state.worker)
    except (ProcessLookupError, RuntimeError):
        # It may have finished between the request and signal. The request remains durable.
        pass
    return state


def wait_local(directory: Path, *, emit) -> int:
    """Follow the whole workflow; Ctrl+C requests cancellation and waits for cleanup."""
    offsets: dict[Path, int] = {}
    interrupted = False
    while True:
        try:
            state = load_local_run(directory)
            for stage in state.stages:
                if not stage.log.exists():
                    continue
                with stage.log.open("rb") as handle:
                    handle.seek(offsets.get(stage.log, 0))
                    data = handle.read()
                    offsets[stage.log] = handle.tell()
                if data:
                    emit(data.decode("utf-8", errors="replace"))
            if state.status in TERMINAL:
                return (
                    0
                    if state.status == "completed"
                    else (130 if state.status == "cancelled" else 1)
                )
            if state.worker and not alive(state.worker):
                emit(
                    "Local worker exited without a terminal record. Inspect worker.log and sbt status.\n"
                )
                return 1
            time.sleep(0.2)
        except KeyboardInterrupt:
            if not interrupted:
                request_cancel(
                    directory, reason="Foreground run interrupted with Ctrl+C"
                )
                emit("Cancellation requested; waiting for process cleanup...\n")
                interrupted = True
