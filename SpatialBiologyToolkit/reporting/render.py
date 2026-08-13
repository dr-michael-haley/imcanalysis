"""Render execution and project Markdown from structured records."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import yaml  # type: ignore[import-untyped]

from SpatialBiologyToolkit.pipeline.assets import resolve_assets
from SpatialBiologyToolkit.pipeline.environment_selection import (
    effective_environment_keys,
)
from SpatialBiologyToolkit.pipeline.executions import (
    execution_output_path,
    load_execution_index,
)
from SpatialBiologyToolkit.pipeline.manifests import write_text, write_yaml
from SpatialBiologyToolkit.pipeline.models import ExecutionRecord, ProjectMetadata
from SpatialBiologyToolkit.pipeline.project import PROJECT_MARKER
from SpatialBiologyToolkit.pipeline.registry import get_stage, toolkit_root

from .models import GeneratedFile, PathRecord, StageManifest
from .paths import ReportingContext

if TYPE_CHECKING:
    from SpatialBiologyToolkit.pipeline.project import ProjectContext
    from SpatialBiologyToolkit.pipeline.runs import RunRecord


def _link(target: Path, source_dir: Path) -> str:
    try:
        return Path(os.path.relpath(target, source_dir)).as_posix()
    except ValueError:
        return target.as_posix()


def _markdown_text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _path_rows(records: Iterable[PathRecord], source_dir: Path) -> list[str]:
    rows: list[str] = []
    for record in records:
        status = "present" if record.exists else "not found"
        rows.append(
            f"- **{record.role}**: [{record.path.name or record.path}]"
            f"({_link(record.path, source_dir)}) — {record.description or status}"
        )
    return rows or ["- None recorded."]


def _file_rows(records: Iterable[GeneratedFile], source_dir: Path) -> list[str]:
    rows = [
        f"- **{record.category}**: [{record.path.name}]"
        f"({_link(record.path, source_dir)})"
        for record in records
    ]
    return rows or ["- None recorded."]


def _environment_rows(manifest: StageManifest, source_dir: Path) -> list[str]:
    reference = manifest.environment
    if reference is None:
        return ["No managed environment record was available for this execution."]
    manifest_path = source_dir / reference.manifest
    runtime: dict = {}
    try:
        loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        runtime = loaded.get("runtime", {}) if isinstance(loaded, dict) else {}
    except (OSError, yaml.YAMLError):
        pass
    rows = [
        f"This stage used the fixed Conda environment `{reference.conda_name}`.",
        "",
        f"[Environment specification and installed-package snapshot]"
        f"({_link(source_dir / reference.specification_snapshot, source_dir)})",
        "",
        f"- Environment key: `{reference.key}`",
        f"- Python: `{runtime.get('python_version') or '-'}`",
        f"- Toolkit commit: `{runtime.get('toolkit_git_commit') or '-'}`",
        f"- Editable toolkit installation: `{'yes' if runtime.get('toolkit_editable') else 'no'}`",
        f"- Environment drift at execution: `{runtime.get('drift') or 'unknown'}`",
    ]
    if reference.overridden:
        defaults = "`, `".join(reference.default_keys) or "none"
        rows.append(
            f"- Per-run override: `yes` (registered stage default: `{defaults}`)"
        )
    if reference.additional_keys:
        rows.append(
            "- Additional stage environments: `"
            + "`, `".join(reference.additional_keys)
            + "`"
        )
    return rows


def render_run_readme(manifest: StageManifest, destination: Path) -> str:
    source_dir = destination.parent
    rationale = manifest.reason or (
        "This stage was run as part of the recorded pipeline request."
    )
    human_id = manifest.execution_label or "direct"
    lines = [
        f"# Execution {human_id} — {manifest.stage_display_name or manifest.display_name}",
        "",
        "## Stage overview",
        "",
        manifest.explainer_snapshot.strip()
        or "No shared stage explainer was available when this report was generated.",
        "",
        "## This execution",
        "",
        f"- Project ID: `{manifest.project_id}`",
        f"- Execution ID: `{human_id}`",
        f"- Stage alias: `{manifest.stage}`",
        f"- Started: `{manifest.started_at.isoformat()}`",
        f"- Completed: `{manifest.completed_at.isoformat() if manifest.completed_at else '-'}`",
        f"- Status: **{manifest.status}**",
        f"- Reusable asset effect: `{manifest.asset_effect}`",
        f"- SLURM job ID: `{manifest.slurm_job_id or '-'}`",
        f"- Pipeline version: `{manifest.pipeline_version or '-'}`",
        f"- Git commit: `{manifest.git_commit or '-'}`",
        "",
        "## Why this stage was run",
        "",
        rationale,
        "",
    ]
    if manifest.notes:
        lines.extend(["Run notes:", "", *(f"- {note}" for note in manifest.notes), ""])
    lines.extend(
        ["## Software environment", "", *_environment_rows(manifest, source_dir), ""]
    )
    lines.extend(["## Inputs", "", *_path_rows(manifest.inputs, source_dir), ""])
    lines.extend(
        [
            "## Reusable assets produced",
            "",
            *_path_rows(manifest.produced_assets, source_dir),
            "",
            "## Figures and tables",
            "",
            *_file_rows(manifest.generated_files, source_dir),
            "",
            "## Important configuration",
            "",
        ]
    )
    if manifest.parameters:
        lines.extend(
            [
                "| Setting | Value | Description |",
                "|---|---|---|",
                *(
                    f"| `{name}` | `{_markdown_text(record.value)}` | "
                    f"{_markdown_text(record.description)} |"
                    for name, record in manifest.parameters.items()
                ),
            ]
        )
    else:
        lines.append(
            "No concise stage parameters were available. See the full resolved "
            "configuration in the technical workflow record."
        )
    lines.extend(["", "## Metrics and findings", ""])
    lines.extend(
        (f"- **{name}**: `{value}`" for name, value in manifest.metrics.items())
        if manifest.metrics
        else ["- No objective metrics were registered."]
    )
    lines.extend(["", "## Warnings and limitations", ""])
    warnings = [f"- {warning}" for warning in manifest.warnings]
    warnings.extend(f"- **{error.type}**: {error.message}" for error in manifest.errors)
    lines.extend(warnings or ["- No warnings were recorded."])
    lines.extend(["", "## Technical record", ""])
    if manifest.technical_run_record:
        lines.extend(
            [
                f"- Technical execution ID: `{manifest.technical_run_id}`",
                f"- Workflow run ID: `{manifest.workflow_run_id}`",
                f"- [Technical workflow directory]"
                f"({_link(manifest.technical_run_record, source_dir)})",
                f"- [SLURM logs]({_link(manifest.technical_run_record / 'logs', source_dir)})",
                f"- [Full resolved configuration]"
                f"({_link(manifest.technical_run_record / 'config.resolved.yaml', source_dir)})",
            ]
        )
    else:
        lines.append(
            "- This was a direct execution; full SBT/SLURM provenance is unavailable."
        )
    lines.extend(
        [
            "",
            "## How to interpret these outputs",
            "",
            "Use the interpretation and limitations guidance in the stage overview "
            "above. Metrics are objective execution summaries; no automatic "
            "biological conclusions have been added.",
            "",
        ]
    )
    return "\n".join(lines)


def _project_identity(project_root: Path) -> tuple[str, str]:
    title = project_root.name or "Project"
    description = ""
    try:
        metadata = ProjectMetadata.model_validate(
            yaml.safe_load(
                (project_root / PROJECT_MARKER).read_text(encoding="utf-8")
            )
        )
        title = metadata.title or title
        description = metadata.description or ""
    except (OSError, ValueError, TypeError):
        pass
    return title, description


def render_project_index(context: ProjectContext) -> str:
    title, description = _project_identity(context.root)
    index = load_execution_index(context)
    rows = [
        "# Project Analysis Summary",
        "",
        f"Project: **{title}**",
        "",
        f"Project ID: `{context.project_metadata.project_id}`",
        "",
    ]
    if description:
        rows.extend([description, ""])
    rows.extend(
        [
            "Execution numbers show the active project workflow order. Permanent "
            "technical provenance and SLURM logs remain under `.sbt/`.",
            "",
            "| ID | Stage | Status | Started | Assets | Report |",
            "|---:|---|---|---|---|---|",
        ]
    )
    for execution in index.executions:
        output = execution_output_path(context, execution)
        started = execution.started_at or execution.created_at
        rows.append(
            f"| {execution.execution_label} | {execution.stage_display_name} | "
            f"{execution.status} | {started.strftime('%Y-%m-%d %H:%M')} | "
            f"{execution.asset_effect} | "
            f"[Open]({_link(output, outputs_root(context))}/) |"
        )
    if not index.executions:
        rows.append("| - | No executions recorded | - | - | - | - |")
    rows.extend(["", "## Reusable project assets", ""])
    try:
        for asset in resolve_assets(context.config, context.root):
            if asset.lifecycle not in {
                "required_input",
                "optional_input",
                "generated_output",
            }:
                continue
            rows.append(
                f"- **{asset.role}**: [{asset.path.name or asset.path}]"
                f"({_link(asset.path, outputs_root(context))})"
            )
    except (OSError, ValueError):
        rows.append("- Asset inventory could not be loaded from the project config.")
    rows.extend(
        [
            "",
            "## Folder roles",
            "",
            "- Project root: reusable inputs and computational assets.",
            "- `outputs/`: active, sequential human-facing execution records.",
            "- `.sbt/runs/`: permanent workflow commands, configs, statuses, and logs.",
            "- `.sbt/audit/`: technical migration and removal evidence.",
            "- `QC/`: deprecated legacy output location; existing files are not moved.",
            "- [Project notes](../.sbt/project_notes.md): durable project context.",
            "",
        ]
    )
    return "\n".join(rows)


def outputs_root(context: ProjectContext) -> Path:
    configured = Path(context.config.general.outputs_folder).expanduser()
    if not configured.is_absolute():
        configured = context.root / configured
    return configured.resolve(strict=False)


def refresh_project_index(context: ProjectContext) -> Path:
    destination = outputs_root(context) / "README.md"
    return write_text(destination, render_project_index(context))


def write_indexes(context: ReportingContext, manifest: StageManifest) -> None:
    readme = context.output_dir / "README.md"
    write_text(readme, render_run_readme(manifest, readme))
    if context.managed_run:
        from SpatialBiologyToolkit.pipeline.project import load_project

        refresh_project_index(load_project(context.project_root))


def prepare_execution_output(
    context: ProjectContext,
    run: RunRecord,
    execution: ExecutionRecord,
) -> StageManifest:
    """Create a truthful pending report only after SLURM accepts the stage."""
    output = execution_output_path(context, execution)
    output.mkdir(parents=True, exist_ok=True)
    spec = get_stage(execution.stage)
    doc_path = (toolkit_root() / spec.documentation_path).resolve(strict=False)
    try:
        explainer = doc_path.read_text(encoding="utf-8")
    except OSError:
        explainer = ""
    from SpatialBiologyToolkit.environments.provenance import (
        snapshot_stage_environment_specifications,
    )

    environment_reference = snapshot_stage_environment_specifications(
        stage=execution.stage,
        output_directory=output,
        repository_root=toolkit_root(),
        environment_keys=effective_environment_keys(run.plan, execution.stage),
        default_environment_keys=get_stage(execution.stage).environment_keys,
    )
    manifest = StageManifest(
        project_id=context.project_metadata.project_id,
        execution_id=execution.execution_id,
        execution_label=execution.execution_label,
        technical_run_id=execution.technical_run_id,
        workflow_run_id=execution.workflow_run_id,
        output_folder=execution.output_folder,
        run_id=execution.workflow_run_id,
        stage=execution.stage,
        display_name=spec.display_name,
        stage_display_name=spec.display_name,
        status="pending",
        managed_run=True,
        asset_effect=execution.asset_effect,
        started_at=execution.created_at,
        slurm_job_id=execution.slurm_job_id,
        technical_run_record=run.run_dir,
        reason=run.manifest.reason,
        notes=run.manifest.notes,
        documentation_source=doc_path if doc_path.is_file() else None,
        explainer_snapshot=explainer,
        environment=environment_reference,
    )
    write_yaml(output / "stage_manifest.yaml", manifest)
    write_text(output / "README.md", render_run_readme(manifest, output / "README.md"))
    write_yaml(
        run.run_dir / "stage_events" / f"{execution.technical_run_id}.yaml",
        manifest,
    )
    refresh_project_index(context)
    return manifest


def initialize_output_layout(
    *,
    project_root: Path,
    project_id: str,
    config_path: Path,
    outputs_root: Path,
) -> Path:
    """Create only the output root; execution folders are allocated per run."""
    outputs_root.mkdir(parents=True, exist_ok=True)
    destination = outputs_root / "README.md"
    if not destination.exists():
        write_text(
            destination,
            "# Project Analysis Summary\n\nNo executions recorded.\n",
        )
    return destination


__all__ = [
    "initialize_output_layout",
    "prepare_execution_output",
    "refresh_project_index",
    "render_project_index",
    "render_run_readme",
    "write_indexes",
]
