"""Resolve and refresh cross-run SLURM dependencies for one managed project."""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from .executions import load_execution_index, resolve_execution
from .manifests import utc_now
from .models import ExecutionRecord, ExternalDependency
from .project import ProjectContext
from .runs import resolve_run_directory
from .scheduler import list_user_jobs
from .status import inspect_run_status


Runner = Callable[..., subprocess.CompletedProcess[str]]
RECORDED_ACTIVE_STATUSES = {"pending", "running", "unknown"}


def _queue_status(raw_state: str, reason: str | None = None) -> str:
    state_parts = raw_state.strip().upper().split("+", 1)[0].split(maxsplit=1)
    if not state_parts:
        return "unknown"
    state = state_parts[0]
    normalized_reason = "".join(
        character for character in (reason or "").upper() if character.isalnum()
    )
    if "DEPENDENCYNEVERSATISFIED" in normalized_reason:
        return "blocked"
    if state in {"RUNNING", "COMPLETING", "STAGE_OUT"}:
        return "running"
    if state in {
        "PENDING",
        "CONFIGURING",
        "SUSPENDED",
        "REQUEUED",
        "RESIZING",
    }:
        return "pending"
    if state == "COMPLETED":
        return "completed"
    if state in {
        "FAILED",
        "BOOT_FAIL",
        "DEADLINE",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "TIMEOUT",
        "SPECIAL_EXIT",
    }:
        return "failed"
    if state in {"CANCELLED", "PREEMPTED", "REVOKED"}:
        return "cancelled"
    return "unknown"


def _dependency(
    context: ProjectContext,
    execution: ExecutionRecord,
    *,
    status: str,
    source: str,
) -> ExternalDependency:
    if not execution.slurm_job_id:
        raise ValueError(
            f"Execution {execution.execution_label} has no recorded SLURM job."
        )
    return ExternalDependency(
        project_id=context.project_metadata.project_id,
        execution_id=execution.execution_id,
        execution_label=execution.execution_label,
        technical_run_id=execution.technical_run_id,
        workflow_run_id=execution.workflow_run_id,
        stage=execution.stage,
        job_id=execution.slurm_job_id,
        observed_status=status,
        checked_at=utc_now(),
        source=source,
    )


def recorded_external_dependency(
    context: ProjectContext,
    reference: str,
) -> ExternalDependency:
    """Resolve an execution without querying SLURM, for dry-run-safe previews."""

    if reference.casefold() == "latest-active":
        candidates = [
            item
            for item in load_execution_index(context, require_exists=True).executions
            if item.slurm_job_id and item.status in RECORDED_ACTIVE_STATUSES
        ]
        if not candidates:
            raise FileNotFoundError(
                "No execution with a recorded non-terminal status is available."
            )
        execution = candidates[-1]
    else:
        execution = resolve_execution(context, reference)
    if execution.status in {"failed", "cancelled", "blocked"}:
        raise ValueError(
            f"Execution {execution.execution_label} is recorded as "
            f"{execution.status}; an afterok dependency cannot be satisfied."
        )
    return _dependency(
        context,
        execution,
        status=execution.status,
        source="recorded execution state; SLURM not queried",
    )


def active_project_dependencies(
    context: ProjectContext,
    *,
    runner: Runner = subprocess.run,
) -> list[ExternalDependency]:
    """Return live queued/running jobs recorded by the current project."""

    index = load_execution_index(context, require_exists=True)
    by_job_id = {
        item.slurm_job_id: item for item in index.executions if item.slurm_job_id
    }
    if not by_job_id:
        return []
    snapshot = list_user_jobs(runner=runner)
    dependencies: list[ExternalDependency] = []
    for job in snapshot.jobs:
        if job.job_id not in by_job_id:
            continue
        status = _queue_status(job.state, job.reason)
        if status not in {"pending", "running"}:
            continue
        dependencies.append(
            _dependency(
                context,
                by_job_id[job.job_id],
                status=status,
                source="squeue",
            )
        )
    return sorted(dependencies, key=lambda item: item.execution_id)


def refresh_external_dependency(
    context: ProjectContext,
    dependency: ExternalDependency,
    *,
    runner: Runner = subprocess.run,
) -> ExternalDependency:
    """Revalidate one predecessor with squeue and sacct immediately before use."""

    if dependency.project_id != context.project_metadata.project_id:
        raise ValueError("The predecessor belongs to a different SBT project.")
    execution = resolve_execution(context, dependency.execution_id)
    if execution.technical_run_id != dependency.technical_run_id:
        raise ValueError(
            f"Execution {dependency.execution_label} no longer identifies the "
            "selected technical execution."
        )
    try:
        queued = list_user_jobs(job_id=dependency.job_id, runner=runner)
    except RuntimeError:
        queued = None
    if queued is not None and queued.jobs:
        job = queued.jobs[0]
        status = _queue_status(job.state, job.reason)
        if status in {"pending", "running", "completed"}:
            return _dependency(
                context,
                execution,
                status=status,
                source="squeue",
            )
        if status != "unknown":
            raise ValueError(
                f"Predecessor execution {execution.execution_label} is {status}; "
                "an afterok dependency cannot be satisfied."
            )

    run_dir = resolve_run_directory(context, execution.workflow_run_id)
    report = inspect_run_status(context, run_dir, runner=runner)
    status = next(
        (
            item
            for item in report.stages
            if item.technical_run_id == execution.technical_run_id
            or item.job_id == dependency.job_id
        ),
        None,
    )
    if status is None or status.status == "unknown":
        raise RuntimeError(
            f"SLURM could not confirm the current state of predecessor job "
            f"{dependency.job_id}; refusing to create an unverified dependency."
        )
    if status.status == "completed":
        return _dependency(
            context,
            execution,
            status="completed",
            source=status.source,
        )
    if status.status in {"pending", "running"}:
        return _dependency(
            context,
            execution,
            status=status.status,
            source=status.source,
        )
    raise ValueError(
        f"Predecessor execution {execution.execution_label} is {status.status}; "
        "an afterok dependency cannot be satisfied."
    )


__all__ = [
    "active_project_dependencies",
    "recorded_external_dependency",
    "refresh_external_dependency",
]
