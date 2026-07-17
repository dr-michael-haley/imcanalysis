"""Render run, stage, and project Markdown indexes from structured manifests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from SpatialBiologyToolkit.config import load_config
from SpatialBiologyToolkit.pipeline.assets import resolve_assets
from SpatialBiologyToolkit.pipeline.manifests import read_model, write_text
from SpatialBiologyToolkit.pipeline.models import ProjectMetadata
from SpatialBiologyToolkit.pipeline.project import PROJECT_MARKER
from SpatialBiologyToolkit.pipeline.registry import STAGES, get_stage, toolkit_root

from .models import GeneratedFile, PathRecord, StageManifest
from .paths import ReportingContext


def _link(target: Path, source_dir: Path) -> str:
    try:
        return Path(os.path.relpath(target, source_dir)).as_posix()
    except ValueError:
        return target.as_posix()


def _markdown_text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _path_rows(
    records: Iterable[PathRecord],
    source_dir: Path,
) -> list[str]:
    rows: list[str] = []
    for record in records:
        status = "present" if record.exists else "not found"
        rows.append(
            f"- **{record.role}**: [{record.path.name or record.path}]"
            f"({_link(record.path, source_dir)}) — {record.description or status}"
        )
    return rows or ["- None recorded."]


def _file_rows(
    records: Iterable[GeneratedFile],
    source_dir: Path,
) -> list[str]:
    rows: list[str] = []
    for record in records:
        rows.append(
            f"- **{record.category}**: [{record.path.name}]"
            f"({_link(record.path, source_dir)})"
        )
    return rows or ["- None recorded."]


def render_run_readme(manifest: StageManifest, destination: Path) -> str:
    source_dir = destination.parent
    rationale = (
        manifest.reason
        or "This stage was run as part of the recorded pipeline request."
    )
    lines = [
        f"# {manifest.display_name} Report",
        "",
        "## Stage overview",
        "",
        manifest.explainer_snapshot.strip()
        or "No shared stage explainer was available when this report was generated.",
        "",
        "## This execution",
        "",
        f"- Project ID: `{manifest.project_id}`",
        f"- Run ID: `{manifest.run_id}`",
        f"- Stage alias: `{manifest.stage}`",
        f"- Started: `{manifest.started_at.isoformat()}`",
        f"- Completed: `{manifest.completed_at.isoformat() if manifest.completed_at else '-'}`",
        f"- Status: **{manifest.status}**",
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
            "configuration in the technical run record."
        )
    lines.extend(["", "## Metrics and findings", ""])
    lines.extend(
        (f"- **{name}**: `{value}`" for name, value in manifest.metrics.items())
        if manifest.metrics
        else ["- No objective metrics were registered."]
    )
    lines.extend(["", "## Warnings and limitations", ""])
    warning_rows = [f"- {warning}" for warning in manifest.warnings]
    warning_rows.extend(
        f"- **{error.type}**: {error.message}" for error in manifest.errors
    )
    lines.extend(warning_rows or ["- No warnings were recorded."])
    lines.extend(["", "## Technical record", ""])
    if manifest.technical_run_record:
        lines.append(
            f"- [Technical run directory]"
            f"({_link(manifest.technical_run_record, source_dir)})"
        )
        logs = manifest.technical_run_record / "logs"
        lines.append(f"- [SLURM logs]({_link(logs, source_dir)})")
        resolved = manifest.technical_run_record / "config.resolved.yaml"
        lines.append(f"- [Full resolved configuration]({_link(resolved, source_dir)})")
    else:
        lines.append(
            "- This was a direct execution; full SBT run and SLURM provenance is unavailable."
        )
    lines.extend(
        [
            "",
            "## How to interpret these outputs",
            "",
            "Use the interpretation and limitations guidance in the stage overview "
            "above. Metrics in this report are objective execution summaries; no "
            "automatic biological conclusions have been added.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_manifests(stage_root: Path) -> list[StageManifest]:
    records: list[StageManifest] = []
    if not stage_root.is_dir():
        return records
    for path in stage_root.glob("*/stage_manifest.yaml"):
        try:
            records.append(read_model(path, StageManifest))
        except (OSError, ValueError):
            continue
    return sorted(records, key=lambda item: (item.started_at, item.run_id))


def render_stage_index(context: ReportingContext, explainer: str) -> str:
    spec = get_stage(context.stage)
    manifests = _load_manifests(context.stage_root)
    latest = manifests[-1] if manifests else None
    lines = [
        f"# {spec.display_name}",
        "",
        explainer.strip(),
        "",
        "## Recorded executions",
        "",
    ]
    if latest:
        lines.extend(
            [
                f"Latest run: [{latest.run_id}]({latest.run_id}/README.md) "
                f"— **{latest.status}**",
                "",
                "| Run ID | Started | Status | SLURM job | Technical record |",
                "|---|---|---|---|---|",
            ]
        )
        for manifest in reversed(manifests):
            technical = (
                f"[run]({_link(manifest.technical_run_record, context.stage_root)})"
                if manifest.technical_run_record
                else "-"
            )
            lines.append(
                f"| [{manifest.run_id}]({manifest.run_id}/README.md) | "
                f"{manifest.started_at.isoformat()} | {manifest.status} | "
                f"{manifest.slurm_job_id or '-'} | {technical} |"
            )
        lines.extend(
            [
                "",
                "## Reusable assets from the latest run",
                "",
                *_path_rows(latest.produced_assets, context.stage_root),
            ]
        )
    else:
        lines.append("No executions have been recorded.")
    lines.extend(
        [
            "",
            "## Navigation",
            "",
            f"- Stage alias: `{spec.name}`",
            f"- CLI: `sbt stages explain {spec.name}`",
            f"- Documentation source: `{spec.documentation_path}`",
            f"- ReadTheDocs source: `stages/{Path(spec.documentation_path).name}`",
            "- Technical records are stored under `.sbt/runs/<run_id>/`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_project_index(context: ReportingContext) -> str:
    title = context.project_root.name or "Project"
    description = ""
    try:
        metadata = read_model(
            context.project_root / PROJECT_MARKER,
            ProjectMetadata,
        )
        title = metadata.title or title
        description = metadata.description or ""
    except (OSError, ValueError):
        pass
    rows = [
        f"# {title} analysis outputs",
        "",
        f"Project ID: `{context.project_id}`",
        "",
    ]
    if description:
        rows.extend([description, ""])
    rows.extend(
        [
        "This folder contains human-facing reports, figures, and result tables. "
        "Reusable computational assets remain in the project root, while operational "
        "records and SLURM logs live under `.sbt/runs/`.",
        "",
        "## Pipeline stages",
        "",
        "| Stage | Latest run | Status |",
        "|---|---|---|",
        ]
    )
    for spec in sorted(STAGES, key=lambda stage: (stage.display_order, stage.name)):
        stage_root = context.outputs_root / spec.output_folder
        manifests = _load_manifests(stage_root)
        latest = manifests[-1] if manifests else None
        rows.append(
            f"| [{spec.output_folder}]({spec.output_folder}/README.md) | "
            f"{latest.run_id if latest else '-'} | "
            f"{latest.status if latest else 'not run'} |"
        )

    rows.extend(["", "## Reusable project assets", ""])
    if context.config_path and context.config_path.is_file():
        try:
            config = load_config(context.config_path)
            for asset in resolve_assets(config, context.project_root):
                if asset.lifecycle not in {
                    "required_input",
                    "optional_input",
                    "generated_output",
                }:
                    continue
                rows.append(
                    f"- **{asset.role}**: [{asset.path.name or asset.path}]"
                    f"({_link(asset.path, context.outputs_root)})"
                )
        except (OSError, ValueError):
            rows.append("- Asset inventory could not be loaded from the project config.")
    else:
        rows.append("- No project configuration was available.")
    rows.extend(
        [
            "",
            "## Folder roles",
            "",
            "- Project root: reusable inputs and computational assets.",
            "- `outputs/`: numbered, human-facing stage records.",
            "- `.sbt/runs/`: technical commands, configs, statuses, events, and logs.",
            "- `QC/`: deprecated legacy output location; existing files are not moved.",
            "- [Project notes](../.sbt/project_notes.md): durable human or agent context.",
            "",
        ]
    )
    return "\n".join(rows)


def write_indexes(
    context: ReportingContext,
    manifest: StageManifest,
) -> None:
    run_readme = context.stage_run_dir / "README.md"
    write_text(run_readme, render_run_readme(manifest, run_readme))
    write_text(
        context.stage_root / "README.md",
        render_stage_index(context, manifest.explainer_snapshot),
    )
    write_text(context.outputs_root / "README.md", render_project_index(context))


def initialize_output_layout(
    *,
    project_root: Path,
    project_id: str,
    config_path: Path,
    outputs_root: Path,
) -> Path:
    """Create non-destructive stage indexes for a new or adopted project."""
    outputs_root.mkdir(parents=True, exist_ok=True)
    representative: ReportingContext | None = None
    for spec in sorted(STAGES, key=lambda stage: (stage.display_order, stage.name)):
        stage_root = (outputs_root / spec.output_folder).resolve(strict=False)
        context = ReportingContext(
            stage=spec.name,
            project_root=project_root,
            project_id=project_id,
            run_id="",
            outputs_root=outputs_root,
            stage_root=stage_root,
            stage_run_dir=stage_root,
            technical_run_record=None,
            config_path=config_path,
            managed_run=True,
        )
        representative = representative or context
        doc_path = (toolkit_root() / spec.documentation_path).resolve(strict=False)
        try:
            explainer = doc_path.read_text(encoding="utf-8")
        except OSError:
            explainer = ""
        stage_root.mkdir(parents=True, exist_ok=True)
        write_text(stage_root / "README.md", render_stage_index(context, explainer))
    if representative is None:
        raise RuntimeError("The stage registry is empty.")
    destination = outputs_root / "README.md"
    write_text(destination, render_project_index(representative))
    return destination


__all__ = [
    "initialize_output_layout",
    "render_project_index",
    "render_run_readme",
    "render_stage_index",
    "write_indexes",
]
