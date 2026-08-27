"""Project-local execution numbering, locking, summaries, and removal."""

from __future__ import annotations

import getpass
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Literal

from .manifests import read_model, utc_now, write_yaml
from .models import (
    AssetCleanupAudit,
    AssetEffect,
    ExecutionIndex,
    ExecutionRecord,
    ExecutionReference,
    ExecutionSummary,
    RemovalAudit,
    RenumberRecord,
)
from .registry import get_stage

if TYPE_CHECKING:
    from .project import ProjectContext


EXECUTION_INDEX = Path(".sbt/executions.yaml")
EXECUTION_LOCK = Path(".sbt/locks/executions.lock")
REMOVAL_AUDIT_DIRECTORY = Path(".sbt/audit/removals")


class ExecutionLayoutError(RuntimeError):
    """Raised when the active execution index and project layout disagree."""


def execution_label(execution_id: int) -> str:
    if execution_id < 1:
        raise ValueError("Execution IDs must be positive integers.")
    return f"{execution_id:03d}"


def execution_folder_name(execution_id: int, stage: str) -> str:
    spec = get_stage(stage)
    return f"{execution_label(execution_id)}_{spec.output_slug}"


def outputs_root(context: ProjectContext) -> Path:
    configured = Path(context.config.general.outputs_folder).expanduser()
    if not configured.is_absolute():
        configured = context.root / configured
    return configured.resolve(strict=False)


def _stored_path(context: ProjectContext, path: Path) -> Path:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(context.root)
    except ValueError:
        return resolved


def execution_output_path(
    context: ProjectContext,
    execution: ExecutionRecord,
) -> Path:
    path = execution.output_folder
    if not path.is_absolute():
        path = context.root / path
    return path.resolve(strict=False)


def _new_index(context: ProjectContext) -> ExecutionIndex:
    return ExecutionIndex(
        project_id=context.project_metadata.project_id,
        updated_at=utc_now(),
    )


def load_execution_index(
    context: ProjectContext,
    *,
    require_exists: bool = False,
) -> ExecutionIndex:
    path = context.root / EXECUTION_INDEX
    if not path.is_file():
        if require_exists:
            raise FileNotFoundError(
                f"Execution index not found: {path}. Existing projects with the old "
                "layout must run 'sbt project migrate-execution-layout'."
            )
        return _new_index(context)
    index = read_model(path, ExecutionIndex)
    if index.project_id != context.project_metadata.project_id:
        raise ExecutionLayoutError(
            "Execution index belongs to a different project identity."
        )
    return index


def write_execution_index(
    context: ProjectContext,
    index: ExecutionIndex,
) -> Path:
    updated = index.model_copy(update={"updated_at": utc_now()})
    return write_yaml(context.root / EXECUTION_INDEX, updated)


def initialize_execution_index(context: ProjectContext) -> Path:
    path = context.root / EXECUTION_INDEX
    if not path.exists():
        write_execution_index(context, _new_index(context))
    return path


def has_legacy_execution_layout(context: ProjectContext) -> bool:
    root = outputs_root(context)
    if not root.is_dir():
        return False
    index = load_execution_index(context)
    active_paths = {
        execution_output_path(context, record) for record in index.executions
    }
    for path in root.glob("[0-9][0-9][0-9]_*"):
        resolved = path.resolve(strict=False)
        if path.is_dir() and resolved not in active_paths:
            return True
        if path.is_dir() and any(path.glob("*/stage_manifest.yaml")):
            return True
    return False


def validate_index_sequence(
    context: ProjectContext,
    index: ExecutionIndex,
) -> list[str]:
    errors: list[str] = []
    ids = [record.execution_id for record in index.executions]
    if ids != list(range(1, len(ids) + 1)):
        errors.append("Active execution IDs must be unique, ordered, and sequential.")
    technical_ids = [record.technical_run_id for record in index.executions]
    if len(technical_ids) != len(set(technical_ids)):
        errors.append("Technical execution IDs must be unique.")
    for record in index.executions:
        expected_label = execution_label(record.execution_id)
        expected_name = execution_folder_name(record.execution_id, record.stage)
        if record.execution_label != expected_label:
            errors.append(
                f"Execution {record.execution_id} has label "
                f"'{record.execution_label}', expected '{expected_label}'."
            )
        if execution_output_path(context, record).name != expected_name:
            errors.append(
                f"Execution {expected_label} output folder does not match "
                f"stage slug '{get_stage(record.stage).output_slug}'."
            )
    return errors


@contextmanager
def execution_lock(
    context: ProjectContext,
    *,
    timeout_seconds: float = 10.0,
    stale_seconds: float = 300.0,
) -> Iterator[None]:
    """Acquire the project execution lock using atomic exclusive creation."""
    path = context.root / EXECUTION_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(f"{token}\n{utc_now().isoformat()}\n")
            break
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
                if age > stale_seconds:
                    path.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for project execution lock: {path}"
                )
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            owner = path.read_text(encoding="utf-8").splitlines()[0]
            if owner == token:
                path.unlink()
        except FileNotFoundError:
            pass


def _default_asset_effect(stage: str) -> AssetEffect:
    reusable = {
        role
        for role in get_stage(stage).produces_assets
        if role not in {"human_outputs", "legacy_qc", "legacy_slurm_logs"}
    }
    return "unknown" if reusable else "none"


def _record_for_stage(
    context: ProjectContext,
    *,
    execution_id: int,
    stage: str,
    workflow_run_id: str,
    created_at: datetime,
    technical_run_id: str | None = None,
) -> ExecutionRecord:
    spec = get_stage(stage)
    output = outputs_root(context) / execution_folder_name(execution_id, stage)
    return ExecutionRecord(
        execution_id=execution_id,
        execution_label=execution_label(execution_id),
        original_execution_id=execution_id,
        technical_run_id=technical_run_id or f"stage-{uuid.uuid4().hex}",
        workflow_run_id=workflow_run_id,
        stage=stage,
        stage_display_name=spec.display_name,
        output_slug=spec.output_slug,
        output_folder=_stored_path(context, output),
        status="allocated",
        asset_effect=_default_asset_effect(stage),
        created_at=created_at,
    )


def preview_executions(
    context: ProjectContext,
    stages: list[str],
    *,
    workflow_run_id: str,
    technical_run_ids: list[str] | None = None,
) -> list[ExecutionRecord]:
    if technical_run_ids is not None and len(technical_run_ids) != len(stages):
        raise ValueError("Technical execution identity count does not match stages.")
    index = load_execution_index(context)
    start = len(index.executions) + 1
    now = utc_now()
    return [
        _record_for_stage(
            context,
            execution_id=start + offset,
            stage=stage,
            workflow_run_id=workflow_run_id,
            created_at=now,
            technical_run_id=(technical_run_ids[offset] if technical_run_ids else None),
        )
        for offset, stage in enumerate(stages)
    ]


def allocate_executions(
    context: ProjectContext,
    stages: list[str],
    *,
    workflow_run_id: str,
    technical_run_ids: list[str] | None = None,
) -> list[ExecutionRecord]:
    if not stages:
        return []
    if technical_run_ids is not None and len(technical_run_ids) != len(stages):
        raise ValueError("Technical execution identity count does not match stages.")
    with execution_lock(context):
        if has_legacy_execution_layout(context):
            raise ExecutionLayoutError(
                "Legacy or unindexed numbered output folders were detected. Run "
                "'sbt project validate' and "
                "'sbt project migrate-execution-layout --dry-run' before submitting "
                "new managed executions."
            )
        index = load_execution_index(context)
        errors = validate_index_sequence(context, index)
        if errors:
            raise ExecutionLayoutError(" ".join(errors))
        start = len(index.executions) + 1
        now = utc_now()
        records = [
            _record_for_stage(
                context,
                execution_id=start + offset,
                stage=stage,
                workflow_run_id=workflow_run_id,
                created_at=now,
                technical_run_id=(technical_run_ids[offset] if technical_run_ids else None),
            )
            for offset, stage in enumerate(stages)
        ]
        write_execution_index(
            context,
            index.model_copy(update={"executions": [*index.executions, *records]}),
        )
        return records


def execution_reference(record: ExecutionRecord) -> ExecutionReference:
    return ExecutionReference(
        execution_id=record.execution_id,
        execution_label=record.execution_label,
        original_execution_id=record.original_execution_id,
        technical_run_id=record.technical_run_id,
        stage=record.stage,
        output_folder=record.output_folder,
    )


def resolve_execution(
    context: ProjectContext,
    reference: str | int,
) -> ExecutionRecord:
    index = load_execution_index(context, require_exists=True)
    if str(reference).lower() == "latest":
        if not index.executions:
            raise FileNotFoundError("No active project executions are recorded.")
        return index.executions[-1]
    try:
        execution_id = int(str(reference))
    except ValueError as exc:
        raise ValueError(
            f"Execution reference must be a number or 'latest': {reference}"
        ) from exc
    for record in index.executions:
        if record.execution_id == execution_id:
            return record
    raise FileNotFoundError(
        f"Execution {execution_label(execution_id)} is not active in this project."
    )


def resolve_technical_execution(
    context: ProjectContext,
    technical_run_id: str,
) -> ExecutionRecord:
    index = load_execution_index(context, require_exists=True)
    for record in index.executions:
        if record.technical_run_id == technical_run_id:
            return record
    raise FileNotFoundError(
        f"Technical execution ID is not active: {technical_run_id}"
    )


def update_execution(
    context: ProjectContext,
    technical_run_id: str,
    **changes: object,
) -> ExecutionRecord:
    with execution_lock(context):
        index = load_execution_index(context, require_exists=True)
        records = list(index.executions)
        for position, record in enumerate(records):
            if record.technical_run_id != technical_run_id:
                continue
            updated = record.model_copy(update=changes)
            records[position] = updated
            write_execution_index(
                context,
                index.model_copy(update={"executions": records}),
            )
            return updated
    raise FileNotFoundError(
        f"Technical execution ID is not active: {technical_run_id}"
    )


def _duration(record: ExecutionRecord) -> float | None:
    if record.started_at is None or record.completed_at is None:
        return None
    return max(0.0, (record.completed_at - record.started_at).total_seconds())


def execution_summaries(
    context: ProjectContext,
    *,
    include_removed: bool = False,
) -> list[ExecutionSummary]:
    index = load_execution_index(context)
    summaries = [
        ExecutionSummary(
            execution_id=record.execution_id,
            execution_label=record.execution_label,
            stage=record.stage,
            stage_display_name=record.stage_display_name,
            status=record.status,
            started_at=record.started_at or record.created_at,
            completed_at=record.completed_at,
            duration_seconds=_duration(record),
            asset_effect=record.asset_effect,
            output_folder=record.output_folder,
            technical_run_id=record.technical_run_id,
            workflow_run_id=record.workflow_run_id,
            slurm_job_id=record.slurm_job_id,
        )
        for record in index.executions
    ]
    if include_removed:
        audit_root = context.root / REMOVAL_AUDIT_DIRECTORY
        if audit_root.is_dir():
            for path in sorted(audit_root.glob("*.yaml")):
                try:
                    audit = read_model(path, RemovalAudit)
                except (OSError, ValueError):
                    continue
                record = audit.previous_execution
                summaries.append(
                    ExecutionSummary(
                        execution_id=record.execution_id,
                        execution_label=record.execution_label,
                        stage=record.stage,
                        stage_display_name=record.stage_display_name,
                        status="removed",
                        started_at=record.started_at or record.created_at,
                        completed_at=record.completed_at,
                        duration_seconds=_duration(record),
                        asset_effect=audit.asset_effect,
                        output_folder=record.output_folder,
                        technical_run_id=record.technical_run_id,
                        workflow_run_id=record.workflow_run_id,
                        slurm_job_id=record.slurm_job_id,
                        removed=True,
                        removed_at=audit.removed_at,
                    )
                )
    return summaries


def _replace_prefix(path: Path, old: Path, new: Path) -> Path:
    try:
        suffix = path.resolve(strict=False).relative_to(old.resolve(strict=False))
    except ValueError:
        return path
    return (new / suffix).resolve(strict=False)


def rewrite_execution_records(
    context: ProjectContext,
    records: list[ExecutionRecord],
    old_paths: dict[str, Path],
) -> None:
    """Rewrite mutable display references while preserving technical evidence."""
    from SpatialBiologyToolkit.reporting.models import StageManifest
    from SpatialBiologyToolkit.reporting.render import (
        refresh_project_index,
        render_run_readme,
    )

    from .manifests import read_yaml
    from .runs import RUN_MANIFEST, STATUS_FILE, SUBMITTED_JOBS

    for record in records:
        output = execution_output_path(context, record)
        manifest_path = output / "stage_manifest.yaml"
        old_output = old_paths.get(record.technical_run_id, output)
        if manifest_path.is_file():
            manifest = read_model(manifest_path, StageManifest)
            generated = [
                item.model_copy(
                    update={"path": _replace_prefix(item.path, old_output, output)}
                )
                for item in manifest.generated_files
            ]
            produced = [
                item.model_copy(
                    update={"path": _replace_prefix(item.path, old_output, output)}
                )
                for item in manifest.produced_assets
            ]
            manifest = manifest.model_copy(
                update={
                    "schema_version": 2,
                    "execution_id": record.execution_id,
                    "execution_label": record.execution_label,
                    "technical_run_id": record.technical_run_id,
                    "workflow_run_id": record.workflow_run_id,
                    "output_folder": record.output_folder,
                    "run_id": record.workflow_run_id,
                    "stage_display_name": record.stage_display_name,
                    "asset_effect": record.asset_effect,
                    "generated_files": generated,
                    "produced_assets": produced,
                }
            )
            write_yaml(manifest_path, manifest)
            from .manifests import write_text

            write_text(output / "README.md", render_run_readme(manifest, output / "README.md"))
            event = (
                context.runs_dir
                / record.workflow_run_id
                / "stage_events"
                / f"{record.technical_run_id}.yaml"
            )
            if event.parent.is_dir():
                write_yaml(event, manifest)

        run_dir = context.runs_dir / record.workflow_run_id
        run_manifest_path = run_dir / RUN_MANIFEST
        if run_manifest_path.is_file():
            raw = read_yaml(run_manifest_path)
            links = raw.get("executions", [])
            matched = False
            for link in links:
                if link.get("technical_run_id") == record.technical_run_id:
                    link.update(execution_reference(record).model_dump(mode="json"))
                    matched = True
            if not matched:
                links.append(execution_reference(record).model_dump(mode="json"))
            raw["schema_version"] = 2
            raw.setdefault("workflow_run_id", record.workflow_run_id)
            raw["executions"] = links
            write_yaml(run_manifest_path, raw)
        submitted_path = run_dir / SUBMITTED_JOBS
        if submitted_path.is_file():
            raw = read_yaml(submitted_path)
            for job in raw.get("jobs", []):
                if job.get("technical_run_id") == record.technical_run_id or (
                    not job.get("technical_run_id") and job.get("stage") == record.stage
                ):
                    job["execution_id"] = record.execution_id
                    job["technical_run_id"] = record.technical_run_id
            raw["schema_version"] = 2
            raw.setdefault("workflow_run_id", record.workflow_run_id)
            write_yaml(submitted_path, raw)
        status_path = run_dir / STATUS_FILE
        if status_path.is_file():
            raw = read_yaml(status_path)
            for stage_status in raw.get("stages", []):
                if stage_status.get("technical_run_id") == record.technical_run_id or (
                    not stage_status.get("technical_run_id")
                    and stage_status.get("stage") == record.stage
                ):
                    stage_status["execution_id"] = record.execution_id
                    stage_status["technical_run_id"] = record.technical_run_id
            raw["schema_version"] = 2
            raw.setdefault("workflow_run_id", record.workflow_run_id)
            write_yaml(status_path, raw)
    refresh_project_index(context)


def remove_execution(
    context: ProjectContext,
    execution_id: int,
    *,
    reason: str | None,
    confirmation_mode: Literal["interactive", "non_interactive"],
    asset_cleanup: AssetCleanupAudit | None = None,
) -> RemovalAudit:
    """Remove one visible execution, compact IDs, and preserve an audit record."""
    with execution_lock(context):
        index = load_execution_index(context, require_exists=True)
        errors = validate_index_sequence(context, index)
        if errors:
            raise ExecutionLayoutError(" ".join(errors))
        try:
            removed = next(
                record
                for record in index.executions
                if record.execution_id == execution_id
            )
        except StopIteration as exc:
            raise FileNotFoundError(
                f"Execution {execution_label(execution_id)} is not active."
            ) from exc
        return _remove_selected_executions(
            context,
            index,
            [removed],
            reason=reason,
            confirmation_mode=confirmation_mode,
            asset_cleanup_by_technical_id={removed.technical_run_id: asset_cleanup},
        )[0]


def remove_executions(
    context: ProjectContext,
    technical_run_ids: list[str],
    *,
    reason: str | None,
    confirmation_mode: Literal["interactive", "non_interactive", "system"],
    asset_cleanup_by_technical_id: dict[
        str, AssetCleanupAudit | None
    ] | None = None,
) -> list[RemovalAudit]:
    """Remove several terminal executions under one lock and one renumber pass."""
    requested = list(dict.fromkeys(technical_run_ids))
    if not requested:
        return []
    with execution_lock(context):
        index = load_execution_index(context, require_exists=True)
        errors = validate_index_sequence(context, index)
        if errors:
            raise ExecutionLayoutError(" ".join(errors))
        requested_set = set(requested)
        removed = [
            record
            for record in index.executions
            if record.technical_run_id in requested_set
        ]
        missing = requested_set - {record.technical_run_id for record in removed}
        if missing:
            raise FileNotFoundError(
                "Technical execution IDs are not active: " + ", ".join(sorted(missing))
            )
        return _remove_selected_executions(
            context,
            index,
            removed,
            reason=reason,
            confirmation_mode=confirmation_mode,
            asset_cleanup_by_technical_id=asset_cleanup_by_technical_id,
        )


def _remove_selected_executions(
    context: ProjectContext,
    index: ExecutionIndex,
    removed: list[ExecutionRecord],
    *,
    reason: str | None,
    confirmation_mode: Literal["interactive", "non_interactive", "system"],
    asset_cleanup_by_technical_id: dict[str, AssetCleanupAudit | None] | None = None,
) -> list[RemovalAudit]:
    active = [
        record
        for record in removed
        if record.status in {"allocated", "pending", "running"}
    ]
    if active:
        labels = ", ".join(record.execution_label for record in active)
        raise ExecutionLayoutError(
            "Cannot remove allocated, pending, or running executions: "
            f"{labels}. Cancel or finish them first, then refresh status."
        )

    removed_ids = {record.technical_run_id for record in removed}
    operation_id = f"{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    old_paths = {
        record.technical_run_id: execution_output_path(context, record)
        for record in index.executions
    }
    remaining: list[ExecutionRecord] = []
    mappings: list[RenumberRecord] = []
    for new_id, record in enumerate(
        (item for item in index.executions if item.technical_run_id not in removed_ids),
        start=1,
    ):
        new_output = outputs_root(context) / execution_folder_name(new_id, record.stage)
        updated = record.model_copy(
            update={
                "execution_id": new_id,
                "execution_label": execution_label(new_id),
                "output_folder": _stored_path(context, new_output),
            }
        )
        remaining.append(updated)
        if record.execution_id != new_id:
            mappings.append(
                RenumberRecord(
                    technical_run_id=record.technical_run_id,
                    stage=record.stage,
                    previous_execution_id=record.execution_id,
                    new_execution_id=new_id,
                    previous_output_folder=record.output_folder,
                    new_output_folder=updated.output_folder,
                )
            )

    output = outputs_root(context)
    audit_root = context.root / REMOVAL_AUDIT_DIRECTORY
    staged_removed = {
        record.technical_run_id: audit_root / f".{operation_id}-{position:03d}.output"
        for position, record in enumerate(removed, start=1)
    }
    temporary: dict[str, Path] = {}
    moved_to_new: dict[str, Path] = {}
    audit_paths: list[Path] = []
    audits: list[RemovalAudit] = []
    old_index = index
    try:
        audit_root.mkdir(parents=True, exist_ok=True)
        for record in removed:
            old = old_paths[record.technical_run_id]
            staged = staged_removed[record.technical_run_id]
            if old.exists():
                old.replace(staged)
        for mapping in mappings:
            old = old_paths[mapping.technical_run_id]
            if not old.exists():
                continue
            temp = output / (f".sbt-renumber-{operation_id}-{mapping.technical_run_id}")
            old.replace(temp)
            temporary[mapping.technical_run_id] = temp
        for mapping in mappings:
            staged_path = temporary.get(mapping.technical_run_id)
            if staged_path is None:
                continue
            new = context.root / mapping.new_output_folder
            new.parent.mkdir(parents=True, exist_ok=True)
            staged_path.replace(new)
            moved_to_new[mapping.technical_run_id] = new

        write_execution_index(
            context,
            index.model_copy(update={"executions": remaining}),
        )
        rewrite_execution_records(context, remaining, old_paths)
        removed_at = utc_now()
        asset_cleanup_by_technical_id = asset_cleanup_by_technical_id or {}
        for position, record in enumerate(removed, start=1):
            audit_id = f"{operation_id}-{position:03d}"
            audit = RemovalAudit(
                audit_id=audit_id,
                removed_at=removed_at,
                removed_by=getpass.getuser(),
                previous_execution=record,
                previous_execution_id=record.execution_id,
                technical_run_id=record.technical_run_id,
                stage=record.stage,
                previous_output_folder=record.output_folder,
                asset_effect=record.asset_effect,
                confirmation_mode=confirmation_mode,
                reason=reason,
                renumbered=mappings,
                asset_cleanup=asset_cleanup_by_technical_id.get(
                    record.technical_run_id
                ),
            )
            audit_path = audit_root / f"{audit_id}.yaml"
            write_yaml(audit_path, audit)
            audits.append(audit)
            audit_paths.append(audit_path)
    except Exception:
        write_execution_index(context, old_index)
        rollback_sources = dict(old_paths)
        for mapping in mappings:
            rollback_sources[mapping.technical_run_id] = (
                context.root / mapping.new_output_folder
            )
        for mapping in reversed(mappings):
            old = old_paths[mapping.technical_run_id]
            rollback_new = moved_to_new.get(mapping.technical_run_id)
            rollback_temp = temporary.get(mapping.technical_run_id)
            current = (
                rollback_new
                if rollback_new is not None and rollback_new.exists()
                else rollback_temp
            )
            if current is not None and current.exists() and not old.exists():
                current.replace(old)
        for record in removed:
            staged = staged_removed[record.technical_run_id]
            old = old_paths[record.technical_run_id]
            if staged.exists() and not old.exists():
                staged.replace(old)
        for audit_path in audit_paths:
            if audit_path.exists():
                audit_path.unlink()
        rewrite_execution_records(context, index.executions, rollback_sources)
        raise

    for staged in staged_removed.values():
        if staged.is_dir():
            shutil.rmtree(staged)
        elif staged.exists():
            staged.unlink()
    return audits


__all__ = [
    "EXECUTION_INDEX",
    "EXECUTION_LOCK",
    "REMOVAL_AUDIT_DIRECTORY",
    "ExecutionLayoutError",
    "allocate_executions",
    "execution_folder_name",
    "execution_label",
    "execution_lock",
    "execution_output_path",
    "execution_reference",
    "execution_summaries",
    "has_legacy_execution_layout",
    "initialize_execution_index",
    "load_execution_index",
    "outputs_root",
    "preview_executions",
    "remove_execution",
    "remove_executions",
    "rewrite_execution_records",
    "resolve_execution",
    "resolve_technical_execution",
    "update_execution",
    "validate_index_sequence",
    "write_execution_index",
]
