"""Validate sequential execution reports and permanent technical links."""

from __future__ import annotations

from pathlib import Path

from SpatialBiologyToolkit.pipeline.executions import (
    EXECUTION_INDEX,
    REMOVAL_AUDIT_DIRECTORY,
    execution_output_path,
    has_legacy_execution_layout,
    load_execution_index,
    validate_index_sequence,
)
from SpatialBiologyToolkit.pipeline.manifests import read_model
from SpatialBiologyToolkit.pipeline.models import RemovalAudit, ValidationItem
from SpatialBiologyToolkit.pipeline.project import ProjectContext

from .models import StageManifest


def validate_reporting_layout(
    *,
    context: ProjectContext,
    legacy_qc: Path,
) -> list[ValidationItem]:
    items: list[ValidationItem] = []
    outputs_root = Path(context.config.general.outputs_folder).expanduser()
    if not outputs_root.is_absolute():
        outputs_root = context.root / outputs_root
    outputs_root = outputs_root.resolve(strict=False)
    index_path = context.root / EXECUTION_INDEX
    index = load_execution_index(context)
    items.append(
        ValidationItem(
            name="execution index",
            path=index_path,
            status="ok" if index_path.is_file() else "warning",
            message=(
                "Typed active execution index is present."
                if index_path.is_file()
                else "Execution index is missing; initialize a new project or explicitly migrate an old one."
            ),
        )
    )
    for error in validate_index_sequence(context, index):
        items.append(
            ValidationItem(
                name="execution index consistency",
                path=index_path,
                status="warning",
                message=error,
            )
        )

    active_paths: set[Path] = set()
    active_technical = {record.technical_run_id for record in index.executions}
    for record in index.executions:
        output = execution_output_path(context, record)
        active_paths.add(output)
        technical = context.runs_dir / record.workflow_run_id
        if not technical.is_dir():
            items.append(
                ValidationItem(
                    name=f"execution {record.execution_label} technical link",
                    path=technical,
                    status="warning",
                    message="Permanent workflow run record does not resolve.",
                )
            )
        expected = record.slurm_job_id is not None or record.status in {
            "pending",
            "running",
            "completed",
            "cancelled",
        }
        if not output.is_dir():
            items.append(
                ValidationItem(
                    name=f"execution {record.execution_label} output",
                    path=output,
                    status="warning" if expected else "not_created",
                    message=(
                        "Accepted execution is missing its output folder."
                        if expected
                        else "No scheduler job was accepted, so no human output folder was created."
                    ),
                )
            )
            continue
        manifest_path = output / "stage_manifest.yaml"
        readme_path = output / "README.md"
        if not manifest_path.is_file() or not readme_path.is_file():
            items.append(
                ValidationItem(
                    name=f"execution {record.execution_label} report",
                    path=output,
                    status="warning",
                    message="Execution folder requires stage_manifest.yaml and README.md.",
                )
            )
            continue
        try:
            manifest = read_model(manifest_path, StageManifest)
        except (OSError, ValueError) as exc:
            items.append(
                ValidationItem(
                    name=f"execution {record.execution_label} manifest",
                    path=manifest_path,
                    status="warning",
                    message=f"Stage manifest is invalid: {exc}",
                )
            )
            continue
        mismatches: list[str] = []
        if manifest.execution_id != record.execution_id:
            mismatches.append("execution ID")
        if manifest.technical_run_id != record.technical_run_id:
            mismatches.append("technical execution ID")
        if manifest.workflow_run_id != record.workflow_run_id:
            mismatches.append("workflow run ID")
        if manifest.stage != record.stage:
            mismatches.append("stage")
        if mismatches:
            items.append(
                ValidationItem(
                    name=f"execution {record.execution_label} identity",
                    path=manifest_path,
                    status="warning",
                    message="Manifest/index mismatch: " + ", ".join(mismatches),
                )
            )
    if outputs_root.is_dir():
        seen_prefixes: dict[str, Path] = {}
        for path in outputs_root.iterdir():
            if not path.is_dir() or len(path.name) < 4 or not path.name[:3].isdigit():
                continue
            prefix = path.name[:3]
            if prefix in seen_prefixes:
                items.append(
                    ValidationItem(
                        name="duplicate execution number",
                        path=path,
                        status="warning",
                        message=f"Prefix {prefix} is also used by {seen_prefixes[prefix]}.",
                    )
                )
            seen_prefixes[prefix] = path
            if path.resolve(strict=False) not in active_paths and not has_legacy_execution_layout(context):
                items.append(
                    ValidationItem(
                        name="unindexed execution folder",
                        path=path,
                        status="warning",
                        message="Numbered folder is absent from the active execution index.",
                    )
                )

    if has_legacy_execution_layout(context):
        items.append(
            ValidationItem(
                name="legacy fixed-number execution layout",
                path=outputs_root,
                status="warning",
                message=(
                    "Run 'sbt project migrate-execution-layout --dry-run'; ordinary "
                    "validation never migrates or renumbers automatically."
                ),
            )
        )

    audit_root = context.root / REMOVAL_AUDIT_DIRECTORY
    if audit_root.is_dir():
        for path in audit_root.glob("*.yaml"):
            try:
                audit = read_model(path, RemovalAudit)
            except (OSError, ValueError):
                continue
            if audit.previous_execution.technical_run_id in active_technical:
                items.append(
                    ValidationItem(
                        name="removal audit contamination",
                        path=path,
                        status="warning",
                        message="A removed technical execution is still active in the index.",
                    )
                )

    if not (outputs_root / "README.md").is_file():
        items.append(
            ValidationItem(
                name="project output index",
                path=outputs_root / "README.md",
                status="warning",
                message="Project Analysis Summary README is missing.",
            )
        )
    if legacy_qc.is_dir():
        items.append(
            ValidationItem(
                name="legacy QC folder",
                path=legacy_qc,
                status="warning",
                message="Legacy QC outputs are preserved and are not migrated automatically.",
            )
        )
    if not items:
        items.append(
            ValidationItem(
                name="execution reporting layout",
                path=outputs_root,
                status="ok",
                message="No execution layout issues were identified.",
            )
        )
    return items


__all__ = ["validate_reporting_layout"]
