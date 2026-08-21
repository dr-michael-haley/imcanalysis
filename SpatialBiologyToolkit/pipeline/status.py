"""Lightweight SLURM status inspection for recorded runs."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .manifests import utc_now, write_yaml
from .executions import (
    load_execution_index,
    resolve_technical_execution,
    update_execution,
)
from .models import (
    ExecutionRecord,
    ProjectStatusChange,
    ProjectStatusRefresh,
    RunStatus,
    StageStatus,
    SubmittedJobs,
)
from .project import ProjectContext
from .runs import (
    STATUS_FILE,
    load_run_manifest,
    load_submitted_jobs,
    resolve_run_directory,
)


Runner = Callable[..., subprocess.CompletedProcess[str]]
SCHEDULER_QUERY_CHUNK_SIZE = 200
TERMINAL_EXECUTION_STATUSES = {"completed", "failed", "cancelled", "blocked"}


@dataclass(frozen=True)
class _SchedulerSnapshot:
    queue_states: dict[str, tuple[str, str, str]]
    accounting_states: dict[str, tuple[str, str, str]]
    warnings: list[str]


def _run_command(
    arguments: list[str],
    *,
    runner: Runner,
) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
    try:
        completed = runner(
            arguments,
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError) as exc:
        return None, str(exc)
    if completed.returncode != 0:
        return (
            completed,
            completed.stderr.strip() or f"exit status {completed.returncode}",
        )
    return completed, None


def _parse_squeue(output: str) -> dict[str, tuple[str, str, str]]:
    states: dict[str, tuple[str, str, str]] = {}
    for raw_line in output.splitlines():
        parts = raw_line.strip().split("|", 3)
        if len(parts) < 2:
            continue
        job_id, state = parts[:2]
        name = parts[2] if len(parts) > 2 else ""
        reason = parts[3] if len(parts) > 3 else ""
        states[job_id] = (state, name, reason)
    return states


def _parse_sacct(output: str, job_ids: set[str]) -> dict[str, tuple[str, str, str]]:
    states: dict[str, tuple[str, str, str]] = {}
    for raw_line in output.splitlines():
        parts = raw_line.strip().split("|")
        if len(parts) < 2:
            continue
        job_id = parts[0].strip()
        if job_id not in job_ids:
            continue
        state = parts[1].strip()
        name = parts[2].strip() if len(parts) > 2 else ""
        exit_code = parts[3].strip() if len(parts) > 3 else ""
        states[job_id] = (state, name, exit_code)
    return states


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [
        values[position : position + size] for position in range(0, len(values), size)
    ]


def _query_scheduler(
    job_ids: set[str],
    *,
    runner: Runner,
) -> _SchedulerSnapshot:
    queue_states: dict[str, tuple[str, str, str]] = {}
    accounting_states: dict[str, tuple[str, str, str]] = {}
    warnings: list[str] = []
    ordered_ids = sorted(job_ids)

    for chunk in _chunks(ordered_ids, SCHEDULER_QUERY_CHUNK_SIZE):
        selected = set(chunk)
        queue, queue_error = _run_command(
            [
                "squeue",
                "--noheader",
                "--jobs",
                ",".join(chunk),
                "--format=%i|%T|%j|%r",
            ],
            runner=runner,
        )
        if queue and not queue_error:
            queue_states.update(_parse_squeue(queue.stdout))
        elif queue_error:
            warnings.append(f"squeue unavailable or failed: {queue_error}")

        accounting, accounting_error = _run_command(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--jobs",
                ",".join(chunk),
                "--format=JobIDRaw,State,JobName,ExitCode",
            ],
            runner=runner,
        )
        if accounting and not accounting_error:
            accounting_states.update(_parse_sacct(accounting.stdout, selected))
        elif accounting_error:
            warnings.append(f"sacct unavailable or failed: {accounting_error}")

    return _SchedulerSnapshot(
        queue_states=queue_states,
        accounting_states=accounting_states,
        warnings=list(dict.fromkeys(warnings)),
    )


def _normalize_state(
    raw_state: str,
) -> Literal["pending", "running", "completed", "failed", "cancelled", "unknown"]:
    normalized = raw_state.strip().upper().split("+", 1)[0].split(maxsplit=1)
    if not normalized:
        return "unknown"
    state = normalized[0]
    if state in {"PENDING", "CONFIGURING", "SUSPENDED", "REQUEUED", "RESIZING"}:
        return "pending"
    if state in {"RUNNING", "COMPLETING", "STAGE_OUT"}:
        return "running"
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


def _dependency_cannot_be_satisfied(reason: str) -> bool:
    normalized = "".join(
        character for character in reason.upper() if character.isalnum()
    )
    return "DEPENDENCYNEVERSATISFIED" in normalized


def _resolve_blocked_dependencies(
    stages: list[StageStatus],
    submitted: SubmittedJobs,
) -> list[StageStatus]:
    """Mark pending afterok jobs blocked when their recorded dependency failed."""
    resolved: list[StageStatus] = []
    by_job_id: dict[str, StageStatus] = {}

    for job, stage in zip(submitted.jobs, stages):
        dependency_ids = (job.dependency_job_id or "").split(":")
        failed_dependencies = [
            dependency
            for dependency_id in dependency_ids
            if (dependency := by_job_id.get(dependency_id)) is not None
            and dependency.status in {"failed", "cancelled", "not_submitted", "blocked"}
        ]
        if stage.status in {"pending", "cancelled"} and failed_dependencies:
            dependency = failed_dependencies[0]
            dependency_detail = (
                f"afterok dependency job {dependency.job_id} ended "
                f"{dependency.status}; this job cannot start"
            )
            detail = ", ".join(
                part for part in (stage.detail, dependency_detail) if part
            )
            stage = stage.model_copy(
                update={
                    "status": "blocked",
                    "source": "recorded dependency",
                    "detail": detail,
                }
            )
        resolved.append(stage)
        if job.job_id:
            by_job_id[job.job_id] = stage

    return resolved


def _overall_status(stages: list[StageStatus], submitted: SubmittedJobs) -> str:
    states = [stage.status for stage in stages]
    if any(state == "failed" for state in states):
        return "failed"
    if any(state == "cancelled" for state in states):
        return "cancelled"
    if not submitted.submission_complete or any(
        state == "not_submitted" for state in states
    ):
        return "partial"
    if any(state == "running" for state in states):
        return "running"
    if any(state == "blocked" for state in states):
        return "blocked"
    if any(state == "pending" for state in states):
        return "pending"
    if states and all(state == "completed" for state in states):
        return "completed"
    return "unknown"


def inspect_run_status(
    context: ProjectContext,
    run_dir: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> RunStatus:
    """Refresh one workflow from a bounded scheduler snapshot."""
    directory = Path(run_dir)
    submitted = load_submitted_jobs(directory)
    job_ids = {job.job_id for job in submitted.jobs if job.job_id}
    snapshot = _query_scheduler(job_ids, runner=runner)
    existing = {
        record.technical_run_id: record
        for record in load_execution_index(context).executions
    }
    return _inspect_run_with_snapshot(context, directory, snapshot, existing)


def _inspect_run_with_snapshot(
    context: ProjectContext,
    directory: Path,
    snapshot: _SchedulerSnapshot,
    existing: dict[str, ExecutionRecord],
    *,
    persist: bool = True,
) -> RunStatus:
    manifest = load_run_manifest(directory)
    submitted = load_submitted_jobs(directory)

    stage_statuses: list[StageStatus] = []
    for job in submitted.jobs:
        if job.state == "submission_failed" or not job.job_id:
            stage_statuses.append(
                StageStatus(
                    stage=job.stage,
                    execution_id=job.execution_id,
                    technical_run_id=job.technical_run_id,
                    job_id=job.job_id,
                    status="not_submitted",
                    source="run record",
                    detail=job.error,
                )
            )
            continue
        if job.job_id in snapshot.queue_states:
            raw_state, job_name, reason = snapshot.queue_states[job.job_id]
            detail = ", ".join(part for part in (job_name, reason) if part)
            status = _normalize_state(raw_state)
            if status == "pending" and _dependency_cannot_be_satisfied(reason):
                status = "blocked"
            stage_statuses.append(
                StageStatus(
                    stage=job.stage,
                    execution_id=job.execution_id,
                    technical_run_id=job.technical_run_id,
                    job_id=job.job_id,
                    status=status,
                    source="squeue",
                    detail=detail or raw_state,
                )
            )
            continue
        if job.job_id in snapshot.accounting_states:
            raw_state, job_name, exit_code = snapshot.accounting_states[job.job_id]
            detail = ", ".join(
                part
                for part in (
                    raw_state,
                    job_name,
                    f"exit={exit_code}" if exit_code else "",
                )
                if part
            )
            stage_statuses.append(
                StageStatus(
                    stage=job.stage,
                    execution_id=job.execution_id,
                    technical_run_id=job.technical_run_id,
                    job_id=job.job_id,
                    status=_normalize_state(raw_state),
                    source="sacct",
                    detail=detail or None,
                )
            )
            continue
        current = existing.get(job.technical_run_id or "")
        if current is not None and current.status in TERMINAL_EXECUTION_STATUSES:
            stage_statuses.append(
                StageStatus(
                    stage=job.stage,
                    execution_id=job.execution_id,
                    technical_run_id=job.technical_run_id,
                    job_id=job.job_id,
                    status=current.status,
                    source="recorded terminal state",
                    detail=(
                        "Job was not found in current squeue or sacct output; "
                        "the previously verified terminal state was retained."
                    ),
                )
            )
            continue
        stage_statuses.append(
            StageStatus(
                stage=job.stage,
                execution_id=job.execution_id,
                technical_run_id=job.technical_run_id,
                job_id=job.job_id,
                status="unknown",
                source="SLURM",
                detail="Job was not found in squeue or sacct.",
            )
        )

    stage_statuses = _resolve_blocked_dependencies(stage_statuses, submitted)
    report = RunStatus(
        run_id=manifest.run_id,
        workflow_run_id=manifest.workflow_run_id or manifest.run_id,
        project_id=manifest.project_id,
        checked_at=utc_now(),
        overall_status=_overall_status(stage_statuses, submitted),
        stages=stage_statuses,
        warnings=snapshot.warnings,
    )
    if not persist:
        return report
    write_yaml(directory / STATUS_FILE, report)
    for stage in report.stages:
        if not stage.technical_run_id:
            continue
        status = "failed" if stage.status == "not_submitted" else stage.status
        changes: dict[str, object] = {"status": status}
        try:
            current = resolve_technical_execution(context, stage.technical_run_id)
        except FileNotFoundError:
            continue
        if status == "running" and current.started_at is None:
            changes["started_at"] = utc_now()
        if (
            status in {"completed", "failed", "cancelled", "blocked"}
            and current.completed_at is None
        ):
            changes["completed_at"] = utc_now()
        try:
            update_execution(context, stage.technical_run_id, **changes)
        except FileNotFoundError:
            # A removed execution remains in immutable technical records.
            continue
    return report


def refresh_project_status(
    context: ProjectContext,
    *,
    runner: Runner = subprocess.run,
    persist: bool = True,
) -> ProjectStatusRefresh:
    """Reconcile every active project workflow with one scheduler snapshot."""
    before = load_execution_index(context, require_exists=True).executions
    existing = {record.technical_run_id: record for record in before}
    active_workflows = list(dict.fromkeys(record.workflow_run_id for record in before))
    selected_runs: list[Path] = []
    job_ids: set[str] = set()
    warnings: list[str] = []

    for workflow_run_id in active_workflows:
        try:
            run_dir = resolve_run_directory(context, workflow_run_id)
        except (OSError, ValueError) as exc:
            warnings.append(
                f"Technical run record is unavailable for workflow "
                f"{workflow_run_id}: {exc}"
            )
            continue
        try:
            submitted = load_submitted_jobs(run_dir)
        except (OSError, ValueError) as exc:
            warnings.append(f"Could not read workflow {workflow_run_id}: {exc}")
            continue
        selected_runs.append(run_dir)
        job_ids.update(job.job_id for job in submitted.jobs if job.job_id)

    snapshot = _query_scheduler(job_ids, runner=runner)
    warnings.extend(snapshot.warnings)
    reports: list[RunStatus] = []
    for run_dir in selected_runs:
        try:
            reports.append(
                _inspect_run_with_snapshot(
                    context,
                    run_dir,
                    snapshot,
                    existing,
                    persist=persist,
                )
            )
        except (OSError, ValueError) as exc:
            warnings.append(f"Could not refresh workflow {run_dir.name}: {exc}")

    after = (
        load_execution_index(context, require_exists=True).executions
        if persist
        else before
    )
    projected_statuses = {
        stage.technical_run_id: (
            "failed" if stage.status == "not_submitted" else stage.status
        )
        for report in reports
        for stage in report.stages
        if stage.technical_run_id in existing
    }
    previous_by_id = {record.technical_run_id: record for record in before}
    changes = [
        ProjectStatusChange(
            execution_id=record.execution_id,
            execution_label=record.execution_label,
            technical_run_id=record.technical_run_id,
            workflow_run_id=record.workflow_run_id,
            stage=record.stage,
            previous_status=previous_by_id[record.technical_run_id].status,
            current_status=(
                record.status
                if persist
                else projected_statuses.get(record.technical_run_id, record.status)
            ),
        )
        for record in after
        if record.technical_run_id in previous_by_id
        and (
            record.status
            if persist
            else projected_statuses.get(record.technical_run_id, record.status)
        )
        != previous_by_id[record.technical_run_id].status
    ]
    current_statuses = [
        (
            record.status
            if persist
            else projected_statuses.get(record.technical_run_id, record.status)
        )
        for record in after
    ]
    return ProjectStatusRefresh(
        project_id=context.project_metadata.project_id,
        checked_at=utc_now(),
        workflow_count=len(selected_runs),
        execution_count=len(after),
        unknown_count=sum(status == "unknown" for status in current_statuses),
        changes=changes,
        reports=reports,
        warnings=list(dict.fromkeys(warnings)),
    )


__all__ = ["inspect_run_status", "refresh_project_status"]
