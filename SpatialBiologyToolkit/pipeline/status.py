"""Lightweight SLURM status inspection for recorded runs."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from .manifests import utc_now, write_yaml
from .executions import resolve_technical_execution, update_execution
from .models import RunStatus, StageStatus, SubmittedJobs
from .project import ProjectContext
from .runs import STATUS_FILE, load_run_manifest, load_submitted_jobs


Runner = Callable[..., subprocess.CompletedProcess[str]]


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
    directory = Path(run_dir)
    manifest = load_run_manifest(directory)
    submitted = load_submitted_jobs(directory)
    job_ids = {job.job_id for job in submitted.jobs if job.job_id}
    warnings: list[str] = []
    queue_states: dict[str, tuple[str, str, str]] = {}
    accounting_states: dict[str, tuple[str, str, str]] = {}

    if job_ids:
        queue, queue_error = _run_command(
            [
                "squeue",
                "--noheader",
                "--jobs",
                ",".join(sorted(job_ids)),
                "--format=%i|%T|%j|%r",
            ],
            runner=runner,
        )
        if queue and not queue_error:
            queue_states = _parse_squeue(queue.stdout)
        elif queue_error:
            warnings.append(f"squeue unavailable or failed: {queue_error}")

        accounting, accounting_error = _run_command(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--jobs",
                ",".join(sorted(job_ids)),
                "--format=JobIDRaw,State,JobName,ExitCode",
            ],
            runner=runner,
        )
        if accounting and not accounting_error:
            accounting_states = _parse_sacct(accounting.stdout, job_ids)
        elif accounting_error:
            warnings.append(f"sacct unavailable or failed: {accounting_error}")

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
        if job.job_id in queue_states:
            raw_state, job_name, reason = queue_states[job.job_id]
            detail = ", ".join(part for part in (job_name, reason) if part)
            stage_statuses.append(
                StageStatus(
                    stage=job.stage,
                    execution_id=job.execution_id,
                    technical_run_id=job.technical_run_id,
                    job_id=job.job_id,
                    status=_normalize_state(raw_state),
                    source="squeue",
                    detail=detail or raw_state,
                )
            )
            continue
        if job.job_id in accounting_states:
            raw_state, job_name, exit_code = accounting_states[job.job_id]
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

    report = RunStatus(
        run_id=manifest.run_id,
        workflow_run_id=manifest.workflow_run_id or manifest.run_id,
        project_id=manifest.project_id,
        checked_at=utc_now(),
        overall_status=_overall_status(stage_statuses, submitted),
        stages=stage_statuses,
        warnings=warnings,
    )
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
        if status in {"completed", "failed", "cancelled"}:
            changes["completed_at"] = utc_now()
        try:
            update_execution(context, stage.technical_run_id, **changes)
        except FileNotFoundError:
            # A removed execution remains in immutable technical records.
            continue
    return report


__all__ = ["inspect_run_status"]
