"""Read-mostly project inspection services for lightweight user interfaces.

This module intentionally does not import the scheduler status or submission
modules. Recorded scheduler state is read from durable project files only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from SpatialBiologyToolkit.config.editing import (
    ConfigEditorSession,
    ConfigRecoverySession,
    InvalidConfigEditError,
)
from SpatialBiologyToolkit.config.models import PipelineConfig
from SpatialBiologyToolkit.reporting.models import StageManifest

from .assets import asset_spec_map, resolve_assets
from .executions import (
    execution_output_path,
    execution_summaries,
    resolve_technical_execution,
)
from .logs import resolve_run_logs, tail_text
from .manifests import read_model
from .models import (
    ExecutionRecord,
    ExecutionSummary,
    ProjectAsset,
    ProjectMetadata,
    ProjectValidationReport,
    RunManifest,
    RunPlan,
    RunStatus,
    StageSpec,
)
from .planner import build_run_plan
from .project import (
    PROJECT_MARKER,
    ProjectContext,
    discover_project_root,
    load_project,
    validate_project,
)
from .registry import STAGES, toolkit_root
from .runs import RESOLVED_CONFIG, STATUS_FILE, load_run_manifest, resolve_run_directory


MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_LOG_LINES = 1_000


@dataclass(frozen=True)
class ProjectOpenResult:
    """Normal or recovery-mode result from opening an existing project."""

    root: Path
    metadata: ProjectMetadata
    config_path: Path
    context: ProjectContext | None
    editor: ConfigEditorSession | None
    recovery: ConfigRecoverySession | None
    error: str | None = None

    @property
    def recovery_mode(self) -> bool:
        return self.context is None


@dataclass(frozen=True)
class AssetView:
    """Resolved asset plus catalogue relationships."""

    asset: ProjectAsset
    config_path: str | None
    producers: tuple[str, ...]
    consumers: tuple[str, ...]


@dataclass(frozen=True)
class ProjectSnapshot:
    """Bounded, scheduler-free snapshot used by the dashboard."""

    context: ProjectContext
    validation: ProjectValidationReport
    assets: tuple[AssetView, ...]
    executions: tuple[ExecutionSummary, ...]
    latest_recorded_status: RunStatus | None


@dataclass(frozen=True)
class ExecutionInspection:
    """Bounded detail for one immutable project execution."""

    execution: ExecutionRecord
    run_manifest: RunManifest | None
    recorded_status: RunStatus | None
    stage_manifest: StageManifest | None
    report_text: str
    resolved_config_text: str
    stdout_tail: str
    stderr_tail: str


def _configured_path(root: Path, metadata: ProjectMetadata) -> Path:
    path = Path(metadata.config_file)
    if not path.is_absolute():
        path = root / path
    return path.expanduser().resolve(strict=False)


def open_project_console(project: str | Path | None = None) -> ProjectOpenResult:
    """Open an existing project, falling back to config recovery mode."""

    root = (
        Path(project).expanduser().resolve(strict=False)
        if project is not None
        else discover_project_root()
    )
    marker = root / PROJECT_MARKER
    if not marker.is_file():
        raise FileNotFoundError(
            f"Existing SBT project marker not found: {marker}. "
            "Initialize or adopt the project from the CLI first."
        )
    metadata = read_model(marker, ProjectMetadata)
    config_path = _configured_path(root, metadata)
    try:
        editor = ConfigEditorSession.open(config_path)
        context = load_project(root)
    except (InvalidConfigEditError, ValueError) as exc:
        return ProjectOpenResult(
            root=root,
            metadata=metadata,
            config_path=config_path,
            context=None,
            editor=None,
            recovery=ConfigRecoverySession.open(config_path),
            error=str(exc),
        )
    return ProjectOpenResult(
        root=root,
        metadata=metadata,
        config_path=config_path,
        context=context,
        editor=editor,
        recovery=None,
    )


def context_with_config(
    context: ProjectContext,
    config: PipelineConfig,
) -> ProjectContext:
    """Return an in-memory project context for unsaved readiness previews."""

    return replace(context, config=config)


def asset_views(context: ProjectContext) -> list[AssetView]:
    producers: dict[str, list[str]] = {}
    consumers: dict[str, list[str]] = {}
    for stage in STAGES:
        for role in stage.produces_assets:
            producers.setdefault(role, []).append(stage.name)
        for role in stage.requires_assets:
            consumers.setdefault(role, []).append(stage.name)
    specifications = asset_spec_map()
    config_sources = {role: spec.config_path for role, spec in specifications.items()}
    if not context.config.hyperstac.input_images_folder:
        config_sources["hyperstac_input_images"] = (
            "general.denoised_images_folder (hyperstac fallback)"
        )
    if not context.config.maxfuse.target_adata_path:
        config_sources["maxfuse_target"] = "general.anndata_path (maxfuse fallback)"
    return [
        AssetView(
            asset=asset,
            config_path=config_sources.get(asset.role),
            producers=tuple(producers.get(asset.role, ())),
            consumers=tuple(consumers.get(asset.role, ())),
        )
        for asset in resolve_assets(context.config, context.root)
    ]


def read_recorded_status(
    context: ProjectContext,
    workflow_run_id: str,
) -> RunStatus | None:
    """Read the last status snapshot without querying or mutating a scheduler."""

    try:
        run_dir = resolve_run_directory(context, workflow_run_id)
    except FileNotFoundError:
        return None
    path = run_dir / STATUS_FILE
    if not path.is_file():
        return None
    try:
        return read_model(path, RunStatus)
    except (OSError, ValueError):
        return None


def inspect_project(context: ProjectContext) -> ProjectSnapshot:
    executions = execution_summaries(context)
    latest = executions[-1] if executions else None
    recorded = read_recorded_status(context, latest.workflow_run_id) if latest else None
    return ProjectSnapshot(
        context=context,
        validation=validate_project(context),
        assets=tuple(asset_views(context)),
        executions=tuple(executions),
        latest_recorded_status=recorded,
    )


def inspect_readiness(
    context: ProjectContext,
    targets: Iterable[str],
) -> RunPlan:
    """Return backend-neutral readiness without creating run records."""

    return build_run_plan(context, targets)


def stage_documentation(stage: StageSpec) -> str:
    path = toolkit_root() / stage.documentation_path
    if not path.is_file():
        return stage.description
    return read_text_bounded(path)


def read_text_bounded(path: str | Path, *, max_bytes: int = MAX_TEXT_BYTES) -> str:
    source = Path(path)
    with source.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    truncated = len(data) > max_bytes
    text = data[:max_bytes].decode("utf-8", errors="replace")
    if truncated:
        text += f"\n\n[Display truncated at {max_bytes} bytes.]\n"
    return text


def _safe_stage_manifest(path: Path) -> StageManifest | None:
    if not path.is_file():
        return None
    try:
        return read_model(path, StageManifest)
    except (OSError, ValueError):
        return None


def inspect_execution(
    context: ProjectContext,
    technical_run_id: str,
    *,
    log_lines: int = 100,
) -> ExecutionInspection:
    """Read bounded detail for one execution without scheduler access."""

    execution = resolve_technical_execution(context, technical_run_id)
    output = execution_output_path(context, execution)
    report_path = output / "README.md"
    run_dir = resolve_run_directory(context, execution.workflow_run_id)
    run_manifest: RunManifest | None
    try:
        run_manifest = load_run_manifest(run_dir)
    except (OSError, ValueError):
        run_manifest = None
    recorded = read_recorded_status(context, execution.workflow_run_id)
    report_text = read_text_bounded(report_path) if report_path.is_file() else ""
    resolved_path = run_dir / RESOLVED_CONFIG
    resolved_text = read_text_bounded(resolved_path) if resolved_path.is_file() else ""
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    bounded_lines = max(0, min(int(log_lines), MAX_LOG_LINES))
    try:
        log_records = resolve_run_logs(run_dir, stage=execution.stage)
    except (OSError, ValueError, KeyError, FileNotFoundError):
        log_records = []
    for record in log_records:
        if not record.exists or bounded_lines == 0:
            continue
        content = tail_text(record.path, bounded_lines)
        header = f"[{record.stage} job={record.job_id or '-'} {record.stream}]"
        target = stdout_parts if record.stream == "stdout" else stderr_parts
        target.extend((header, content))
    return ExecutionInspection(
        execution=execution,
        run_manifest=run_manifest,
        recorded_status=recorded,
        stage_manifest=_safe_stage_manifest(output / "stage_manifest.yaml"),
        report_text=report_text,
        resolved_config_text=resolved_text,
        stdout_tail="\n\n".join(stdout_parts),
        stderr_tail="\n\n".join(stderr_parts),
    )


__all__ = [
    "AssetView",
    "ExecutionInspection",
    "MAX_LOG_LINES",
    "MAX_TEXT_BYTES",
    "ProjectOpenResult",
    "ProjectSnapshot",
    "asset_views",
    "context_with_config",
    "inspect_execution",
    "inspect_project",
    "inspect_readiness",
    "open_project_console",
    "read_recorded_status",
    "read_text_bounded",
    "stage_documentation",
]
