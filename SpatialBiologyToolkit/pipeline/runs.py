"""Creation and discovery of durable run records."""

from __future__ import annotations

import getpass
import importlib.metadata
import shlex
import shutil
import socket
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from SpatialBiologyToolkit.config.export import write_resolved_config

from .assets import inventory_assets
from .control import persist_provenance
from .manifests import read_model, utc_now, write_yaml
from .executions import (
    allocate_executions,
    execution_reference,
    preview_executions,
)
from .models import (
    ExecutionRecord,
    ExternalDependency,
    RunManifest,
    RunPlan,
    RunStatus,
    SubmittedJobs,
)
from .project import ProjectContext, copy_user_config
from .registry import toolkit_root


RUN_MANIFEST = "run_manifest.yaml"
RUN_PLAN = "run_plan.yaml"
USER_CONFIG = "config.user.yaml"
RESOLVED_CONFIG = "config.resolved.yaml"
SUBMITTED_JOBS = "submitted_jobs.yaml"
COMMAND_FILE = "command.txt"
STATUS_FILE = "status.yaml"
ASSETS_BEFORE = "project_assets.before.yaml"
LOGS_DIRECTORY = "logs"
STAGE_EVENTS_DIRECTORY = "stage_events"


@dataclass(frozen=True)
class RunRecord:
    workflow_run_id: str
    run_dir: Path
    manifest: RunManifest
    plan: RunPlan
    user_config_path: Path
    resolved_config_path: Path
    executions: list[ExecutionRecord]

    @property
    def run_id(self) -> str:
        """Deprecated compatibility alias for the workflow run ID."""
        return self.workflow_run_id

    def execution_for_stage(self, stage: str) -> ExecutionRecord:
        matches = [item for item in self.executions if item.stage == stage]
        if len(matches) != 1:
            raise KeyError(
                f"Expected one execution for stage '{stage}', found {len(matches)}."
            )
        return matches[0]


def new_run_id(now: datetime | None = None) -> str:
    timestamp = (now or utc_now()).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def command_text(arguments: Iterable[str]) -> str:
    return shlex.join(list(arguments))


def _pipeline_version() -> str | None:
    try:
        return importlib.metadata.version("SpatialBiologyToolkit")
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(toolkit_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else None


def create_run_record(
    context: ProjectContext,
    plan: RunPlan,
    *,
    command: str,
    run_id: str | None = None,
    reason: str | None = None,
    notes: Iterable[str] = (),
    plan_token_digest: str | None = None,
    provenance_digest: str | None = None,
    provenance_file: Path | None = None,
    provenance_payload: dict[str, Any] | None = None,
    technical_run_ids: list[str] | None = None,
    external_dependency: ExternalDependency | None = None,
) -> RunRecord:
    if not plan.ready:
        raise ValueError("Cannot create a submitted run record for an invalid plan.")
    workflow_run_id = run_id or new_run_id()
    run_dir = (context.runs_dir / workflow_run_id).resolve(strict=False)
    if context.runs_dir not in run_dir.parents:
        raise ValueError(f"Invalid workflow run ID: {workflow_run_id}")
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / LOGS_DIRECTORY).mkdir()
    (run_dir / STAGE_EVENTS_DIRECTORY).mkdir()

    user_config_path = run_dir / USER_CONFIG
    resolved_config_path = run_dir / RESOLVED_CONFIG
    copy_user_config(context, user_config_path)
    write_resolved_config(context.config, resolved_config_path)
    if provenance_payload is not None:
        provenance_path, provenance_digest = persist_provenance(
            run_dir,
            provenance_payload,
        )
        provenance_file = provenance_path.relative_to(run_dir)

    try:
        executions = allocate_executions(
            context,
            [stage.name for stage in plan.resolved_stages],
            workflow_run_id=workflow_run_id,
            technical_run_ids=technical_run_ids,
        )
    except Exception:
        # Allocation is the first operation that links this technical workflow to
        # the project-wide execution index. If it fails, this directory contains
        # no durable run evidence and can safely be discarded.
        shutil.rmtree(run_dir)
        raise
    manifest = RunManifest(
        run_id=workflow_run_id,
        workflow_run_id=workflow_run_id,
        project_id=context.project_metadata.project_id,
        project_root=context.root,
        created_at=utc_now(),
        requested_stages=plan.requested,
        resolved_stages=[stage.name for stage in plan.resolved_stages],
        config_source=context.config_path,
        resolved_config=resolved_config_path,
        execution_backend=plan.execution_backend,
        working_directory=Path.cwd().resolve(strict=False),
        command=command,
        reason=reason,
        notes=list(notes),
        pipeline_version=_pipeline_version(),
        git_commit=_git_commit(),
        hostname=socket.gethostname(),
        username=getpass.getuser(),
        executions=[execution_reference(record) for record in executions],
        plan_token_digest=plan_token_digest,
        provenance_digest=provenance_digest,
        provenance_file=provenance_file,
        environment_overrides=dict(plan.environment_overrides),
        external_dependency=external_dependency,
    )
    write_yaml(run_dir / RUN_MANIFEST, manifest)
    write_yaml(run_dir / RUN_PLAN, plan)
    (run_dir / COMMAND_FILE).write_text(command.rstrip() + "\n", encoding="utf-8")
    write_yaml(
        run_dir / ASSETS_BEFORE,
        inventory_assets(
            project_id=context.project_metadata.project_id,
            project_root=context.root,
            config=context.config,
        ),
    )
    write_yaml(
        run_dir / SUBMITTED_JOBS,
        SubmittedJobs(run_id=workflow_run_id, workflow_run_id=workflow_run_id),
    )
    write_yaml(
        run_dir / STATUS_FILE,
        RunStatus(
            run_id=workflow_run_id,
            workflow_run_id=workflow_run_id,
            project_id=context.project_metadata.project_id,
            checked_at=utc_now(),
            overall_status="created",
            stages=[],
        ),
    )
    return RunRecord(
        workflow_run_id=workflow_run_id,
        run_dir=run_dir,
        manifest=manifest,
        plan=plan,
        user_config_path=user_config_path,
        resolved_config_path=resolved_config_path,
        executions=executions,
    )


def prospective_run_record(
    context: ProjectContext,
    plan: RunPlan,
    *,
    run_id: str | None = None,
    command: str = "",
    reason: str | None = None,
    notes: Iterable[str] = (),
    plan_token_digest: str | None = None,
    technical_run_ids: list[str] | None = None,
    external_dependency: ExternalDependency | None = None,
) -> RunRecord:
    """Build run paths for a dry run without creating files or directories."""
    workflow_run_id = run_id or new_run_id()
    run_dir = (context.runs_dir / workflow_run_id).resolve(strict=False)
    resolved_config_path = run_dir / RESOLVED_CONFIG
    executions = preview_executions(
        context,
        [stage.name for stage in plan.resolved_stages],
        workflow_run_id=workflow_run_id,
        technical_run_ids=technical_run_ids,
    )
    manifest = RunManifest(
        run_id=workflow_run_id,
        workflow_run_id=workflow_run_id,
        project_id=context.project_metadata.project_id,
        project_root=context.root,
        created_at=utc_now(),
        requested_stages=plan.requested,
        resolved_stages=[stage.name for stage in plan.resolved_stages],
        config_source=context.config_path,
        resolved_config=resolved_config_path,
        execution_backend=plan.execution_backend,
        working_directory=Path.cwd().resolve(strict=False),
        command=command,
        reason=reason,
        notes=list(notes),
        pipeline_version=_pipeline_version(),
        hostname=socket.gethostname(),
        username=getpass.getuser(),
        executions=[execution_reference(record) for record in executions],
        plan_token_digest=plan_token_digest,
        environment_overrides=dict(plan.environment_overrides),
        external_dependency=external_dependency,
    )
    return RunRecord(
        workflow_run_id=workflow_run_id,
        run_dir=run_dir,
        manifest=manifest,
        plan=plan,
        user_config_path=run_dir / USER_CONFIG,
        resolved_config_path=resolved_config_path,
        executions=executions,
    )


def list_run_directories(context: ProjectContext) -> list[Path]:
    if not context.runs_dir.is_dir():
        return []
    runs: list[tuple[datetime, str, Path]] = []
    for path in context.runs_dir.iterdir():
        if not path.is_dir() or not (path / RUN_MANIFEST).is_file():
            continue
        try:
            created_at = load_run_manifest(path).created_at
        except (OSError, ValueError):
            continue
        runs.append((created_at, path.name, path))
    return [path for _created_at, _name, path in sorted(runs)]


def resolve_run_directory(context: ProjectContext, run_id: str) -> Path:
    if run_id == "latest":
        runs = list_run_directories(context)
        if not runs:
            raise FileNotFoundError(f"No recorded runs found in {context.runs_dir}.")
        return runs[-1]
    candidate = (context.runs_dir / run_id).resolve(strict=False)
    if context.runs_dir not in candidate.parents:
        raise ValueError(f"Invalid run ID: {run_id}")
    if not (candidate / RUN_MANIFEST).is_file():
        raise FileNotFoundError(f"Run record not found: {candidate}")
    return candidate


def load_run_manifest(run_dir: str | Path) -> RunManifest:
    return read_model(Path(run_dir) / RUN_MANIFEST, RunManifest)


def load_run_plan(run_dir: str | Path) -> RunPlan:
    return read_model(Path(run_dir) / RUN_PLAN, RunPlan)


def load_submitted_jobs(run_dir: str | Path) -> SubmittedJobs:
    return read_model(Path(run_dir) / SUBMITTED_JOBS, SubmittedJobs)


def find_run_by_plan_token_digest(
    context: ProjectContext,
    digest: str,
) -> tuple[Path, RunManifest, SubmittedJobs] | None:
    """Resolve an already-created workflow for idempotent gateway retries."""

    for run_dir in reversed(list_run_directories(context)):
        manifest = load_run_manifest(run_dir)
        if manifest.plan_token_digest == digest:
            return run_dir, manifest, load_submitted_jobs(run_dir)
    return None


__all__ = [
    "ASSETS_BEFORE",
    "COMMAND_FILE",
    "LOGS_DIRECTORY",
    "RESOLVED_CONFIG",
    "RUN_MANIFEST",
    "RUN_PLAN",
    "RunRecord",
    "STATUS_FILE",
    "STAGE_EVENTS_DIRECTORY",
    "SUBMITTED_JOBS",
    "USER_CONFIG",
    "command_text",
    "create_run_record",
    "find_run_by_plan_token_digest",
    "list_run_directories",
    "load_run_manifest",
    "load_run_plan",
    "load_submitted_jobs",
    "new_run_id",
    "prospective_run_record",
    "resolve_run_directory",
]
