"""Bounded own-user SLURM queue inspection and guarded cancellation."""

from __future__ import annotations

import getpass
import os
import re
import subprocess
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .control import (
    ActionRecord,
    action_receipt_payload,
    canonical_digest,
    make_preview_token,
    validate_preview_token,
)
from .executions import load_execution_index, resolve_execution
from .manifests import utc_now, write_json
from .models import PipelineModel
from .project import ProjectContext, load_project
from .project_registry import load_project_registry

Runner = Callable[..., subprocess.CompletedProcess[str]]
JOB_REFERENCE = re.compile(r"^\d+(?:_\d+)?$")


def _current_username() -> str:
    if os.name == "posix":
        import pwd

        return pwd.getpwuid(os.getuid()).pw_name
    return getpass.getuser()


class SchedulerJob(PipelineModel):
    job_id: str
    state: str
    name: str
    partition: str | None = None
    elapsed: str | None = None
    time_limit: str | None = None
    nodes: int | None = None
    cpus: int | None = None
    reason: str | None = None
    sbt_managed: bool = False
    project_id: str | None = None
    execution_label: str | None = None


class QueueSnapshot(PipelineModel):
    schema_version: Literal[1] = 1
    captured_at: datetime = Field(default_factory=utc_now)
    jobs: list[SchedulerJob] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _run(arguments: list[str], *, runner: Runner) -> subprocess.CompletedProcess[str]:
    try:
        completed = runner(
            arguments,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"SLURM command unavailable: {arguments[0]}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"{arguments[0]} failed with exit status {completed.returncode}"
            + (f": {detail}" if detail else "")
        )
    return completed


def _managed_jobs() -> dict[str, tuple[str, str]]:
    managed: dict[str, tuple[str, str]] = {}
    try:
        registry = load_project_registry()
    except Exception:  # noqa: BLE001 - one unreadable registry must not break own-job queue
        return managed
    for registered in registry.projects:
        try:
            context = load_project(registered.path)
            index = load_execution_index(context)
        except Exception:  # noqa: BLE001, S112 - stale projects are skipped and untrusted
            continue
        for execution in index.executions:
            if execution.slurm_job_id:
                managed[execution.slurm_job_id] = (
                    context.project_metadata.project_id,
                    execution.execution_label,
                )
    return managed


def _parse_queue(output: str, managed: dict[str, tuple[str, str]]) -> list[SchedulerJob]:
    jobs: list[SchedulerJob] = []
    for line in output.splitlines():
        parts = line.rstrip().split("|", 8)
        if len(parts) < 3 or not JOB_REFERENCE.fullmatch(parts[0].strip()):
            continue
        job_id = parts[0].strip()
        project_id, execution_label = managed.get(job_id, (None, None))
        jobs.append(
            SchedulerJob(
                job_id=job_id,
                state=parts[1].strip(),
                name=parts[2].strip(),
                partition=parts[3].strip() or None if len(parts) > 3 else None,
                elapsed=parts[4].strip() or None if len(parts) > 4 else None,
                time_limit=parts[5].strip() or None if len(parts) > 5 else None,
                nodes=int(parts[6]) if len(parts) > 6 and parts[6].isdigit() else None,
                cpus=int(parts[7]) if len(parts) > 7 and parts[7].isdigit() else None,
                reason=parts[8].strip() or None if len(parts) > 8 else None,
                sbt_managed=project_id is not None,
                project_id=project_id,
                execution_label=execution_label,
            )
        )
    return jobs


def list_user_jobs(
    *,
    job_id: str | None = None,
    runner: Runner = subprocess.run,
) -> QueueSnapshot:
    if job_id is not None and not JOB_REFERENCE.fullmatch(job_id):
        raise ValueError("SLURM job IDs must be numeric, with an optional array task suffix.")
    command = [
        "squeue",
        "--noheader",
        "--user",
        _current_username(),
        "--format=%i|%T|%j|%P|%M|%l|%D|%C|%r",
    ]
    if job_id is not None:
        command.extend(["--jobs", job_id])
    completed = _run(command, runner=runner)
    managed = _managed_jobs()
    return QueueSnapshot(jobs=_parse_queue(completed.stdout, managed))


def resolve_job_reference(
    reference: str,
    *,
    context: ProjectContext | None = None,
) -> tuple[str, str | None, str | None]:
    if context is not None:
        execution = resolve_execution(context, reference)
        if not execution.slurm_job_id:
            raise ValueError(
                f"Execution {execution.execution_label} has no recorded SLURM job."
            )
        return execution.slurm_job_id, context.project_metadata.project_id, execution.execution_label
    if not JOB_REFERENCE.fullmatch(reference):
        raise ValueError(
            "Without --project, cancel requires an exact job ID returned by sbt squeue."
        )
    return reference, None, None


def cancellation_snapshot(job: SchedulerJob, *, reason: str) -> dict[str, Any]:
    return {
        "kind": "cancel",
        "job_id": job.job_id,
        "state": job.state,
        "name": job.name,
        "partition": job.partition,
        "sbt_managed": job.sbt_managed,
        "project_id": job.project_id,
        "execution_label": job.execution_label,
        "reason": reason,
    }


def preview_cancellation(
    reference: str,
    *,
    reason: str,
    context: ProjectContext | None = None,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    job_id, project_id, execution_label = resolve_job_reference(reference, context=context)
    snapshot = list_user_jobs(job_id=job_id, runner=runner)
    if not snapshot.jobs:
        raise ValueError(f"Job {job_id} is not currently queued or running for this user.")
    job = snapshot.jobs[0].model_copy(
        update={
            "project_id": project_id or snapshot.jobs[0].project_id,
            "execution_label": execution_label or snapshot.jobs[0].execution_label,
            "sbt_managed": bool(project_id) or snapshot.jobs[0].sbt_managed,
        }
    )
    state = job.state.upper()
    confirmation_required = state in {"RUNNING", "COMPLETING", "STAGE_OUT"}
    token = make_preview_token(cancellation_snapshot(job, reason=reason), ttl_seconds=300)
    return {
        "schema_version": 1,
        "job": job.model_dump(mode="json"),
        "reason": reason,
        "preview_token": token,
        "preview_expires_in_seconds": 300,
        "confirmation_required": confirmation_required,
        "action_receipt": action_receipt_payload(
            operation="preview_cancel",
            target=job_id,
            actions=[
                ActionRecord(
                    action="Resolved and inspected the current user's SLURM job",
                    justification=reason,
                    outcome="succeeded",
                    evidence=[f"state={job.state}", f"sbt_managed={job.sbt_managed}"],
                ),
                ActionRecord(
                    action="Created a short-lived cancellation token",
                    justification="Cancellation must be rejected if job identity or state changes.",
                    outcome="succeeded",
                ),
            ],
        ),
    }


def cancel_job(
    reference: str,
    *,
    reason: str,
    preview_token: str,
    context: ProjectContext | None = None,
    provenance: dict[str, Any] | None = None,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    job_id, project_id, execution_label = resolve_job_reference(reference, context=context)
    queued = list_user_jobs(job_id=job_id, runner=runner)
    if not queued.jobs:
        return {
            "schema_version": 1,
            "job_id": job_id,
            "outcome": "already_terminal_or_absent",
            "idempotent": True,
            "action_receipt": action_receipt_payload(
                operation="cancel",
                target=job_id,
                actions=[
                    ActionRecord(
                        action="Skipped cancellation because the job was no longer active",
                        justification=reason,
                        outcome="skipped",
                    )
                ],
            ),
        }
    job = queued.jobs[0].model_copy(
        update={
            "project_id": project_id or queued.jobs[0].project_id,
            "execution_label": execution_label or queued.jobs[0].execution_label,
            "sbt_managed": bool(project_id) or queued.jobs[0].sbt_managed,
        }
    )
    validate_preview_token(
        preview_token,
        cancellation_snapshot(job, reason=reason),
    )
    if job.state.upper() in {"RUNNING", "COMPLETING", "STAGE_OUT"} and not (
        isinstance(provenance, dict) and provenance.get("user_confirmation")
    ):
        raise ValueError(
            "This active job requires explicit confirmation from the current preview."
        )
    _run(["scancel", "--user", _current_username(), job_id], runner=runner)
    audit_root = (
        context.root / ".sbt" / "audit" / "cancellations"
        if context is not None
        else Path.home() / ".sbt" / "audit" / "cancellations"
    )
    audit_id = f"cancel-{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    audit = {
        "schema_version": 1,
        "audit_id": audit_id,
        "requested_at": utc_now(),
        "job": job.model_dump(mode="json"),
        "reason": reason,
        "preview_token_digest": canonical_digest(preview_token),
        "provenance": provenance,
        "outcome": "cancellation_requested",
    }
    audit_path = write_json(audit_root / f"{audit_id}.json", audit)
    return {
        "schema_version": 1,
        "job_id": job_id,
        "outcome": "cancellation_requested",
        "idempotent": False,
        "audit_path": str(audit_path),
        "action_receipt": action_receipt_payload(
            operation="cancel",
            target=job_id,
            actions=[
                ActionRecord(
                    action="Revalidated ownership, identity, and current queue state",
                    justification="Only an exact active job owned by the current user may be cancelled.",
                    outcome="succeeded",
                ),
                ActionRecord(
                    action="Requested SLURM cancellation",
                    justification=reason,
                    outcome="succeeded",
                    state_changed=True,
                    evidence=[f"audit_id={audit_id}"],
                ),
            ],
        ),
    }


__all__ = [
    "JOB_REFERENCE",
    "QueueSnapshot",
    "SchedulerJob",
    "cancel_job",
    "list_user_jobs",
    "preview_cancellation",
    "resolve_job_reference",
]
