"""Explicit migration from fixed stage folders to sequential executions."""

from __future__ import annotations

import shutil
import uuid
from collections import Counter
from pathlib import Path

from SpatialBiologyToolkit.reporting.models import StageManifest

from .executions import (
    execution_folder_name,
    execution_label,
    execution_lock,
    load_execution_index,
    outputs_root,
    rewrite_execution_records,
    write_execution_index,
)
from .manifests import read_model, utc_now, write_yaml
from .models import (
    ExecutionIndex,
    ExecutionMigrationPlan,
    ExecutionRecord,
    MigrationAudit,
    MigrationRecord,
)
from .project import ProjectContext
from .registry import get_stage


LEGACY_STAGE_FOLDERS: dict[str, str] = {
    "prep": "001_Preprocessing",
    "denoise": "002_Denoising",
    "dnqc": "003_Denoising_QC",
    "cellpose": "004_Segmentation",
    "nimbus": "005_Quantification",
    "bint": "006_Batch_Integration",
    "rapids": "007_RAPIDS_Processing",
    "bbn": "008_BioBatchNet_Integration",
    "subcl": "009_Subclustering",
    "cchar": "010_CellCharter_Neighbourhoods",
    "starling": "011_STARLING_Phenotyping",
    "aiinter": "012_AI_Interpretation",
    "vis": "013_Visualisation",
    "pairsp": "014_Pairwise_Spatial_Analysis",
    "nxsp": "015_NetworkX_Spatial_Analysis",
    "reint": "016_Marker_Reintegration",
    "remap": "017_Observation_Remapping",
    "rebuildmeta": "018_Metadata_Rebuild",
    "scport": "019_scPortrait_Export",
    "config": "020_Configuration_Maintenance",
    "zipqc": "021_Output_Archive",
    "slogs": "022_Legacy_SLURM_Log_Migration",
    "debug": "023_Environment_Diagnostics",
}


MIGRATION_AUDIT_DIRECTORY = Path(".sbt/audit/migrations")


def _stored_path(context: ProjectContext, path: Path) -> Path:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(context.root)
    except ValueError:
        return resolved


def _technical_id(
    context: ProjectContext,
    *,
    workflow_run_id: str,
    stage: str,
    source: Path,
) -> str:
    identity = "|".join(
        [
            context.project_metadata.project_id,
            workflow_run_id,
            stage,
            str(source.resolve(strict=False)),
        ]
    )
    return f"stage-{uuid.uuid5(uuid.NAMESPACE_URL, identity).hex}"


def plan_execution_layout_migration(
    context: ProjectContext,
) -> ExecutionMigrationPlan:
    ambiguities: list[str] = []
    candidates: list[tuple[StageManifest, str, Path]] = []
    root = outputs_root(context)
    index = load_execution_index(context)
    legacy_roots = [
        root / legacy_name
        for legacy_name in LEGACY_STAGE_FOLDERS.values()
        if (root / legacy_name).is_dir()
    ]
    if index.executions and legacy_roots:
        ambiguities.append(
            "The active execution index is not empty; migration cannot merge layouts."
        )

    for stage, legacy_name in LEGACY_STAGE_FOLDERS.items():
        legacy_root = root / legacy_name
        if not legacy_root.is_dir():
            continue
        allowed = {"README.md"}
        for child in sorted(legacy_root.iterdir()):
            if child.name in allowed:
                continue
            manifest_path = child / "stage_manifest.yaml" if child.is_dir() else None
            if manifest_path is None or not manifest_path.is_file():
                ambiguities.append(
                    f"Unrecognized legacy content cannot be migrated safely: {child}"
                )
                continue
            try:
                manifest = read_model(manifest_path, StageManifest)
            except (OSError, ValueError) as exc:
                ambiguities.append(f"Invalid legacy manifest {manifest_path}: {exc}")
                continue
            if manifest.stage != stage:
                ambiguities.append(
                    f"Legacy folder {child} is registered as '{stage}' but its "
                    f"manifest says '{manifest.stage}'."
                )
                continue
            candidates.append((manifest, stage, child))

    duplicate_times = {
        started_at
        for started_at, count in Counter(
            manifest.started_at for manifest, _stage, _source in candidates
        ).items()
        if count > 1
    }
    for started_at in sorted(duplicate_times):
        ambiguities.append(
            "Multiple legacy executions have the same structured start time "
            f"{started_at.isoformat()}; their project order cannot be established "
            "safely."
        )

    candidates.sort(
        key=lambda item: (
            item[0].started_at,
            item[0].run_id or item[0].workflow_run_id or item[2].name,
            item[1],
        )
    )
    records: list[MigrationRecord] = []
    seen_technical: set[str] = set()
    for execution_id, (manifest, stage, source) in enumerate(candidates, start=1):
        workflow_run_id = manifest.workflow_run_id or manifest.run_id or source.name
        technical_run_id = manifest.technical_run_id or _technical_id(
            context,
            workflow_run_id=workflow_run_id,
            stage=stage,
            source=source,
        )
        if technical_run_id in seen_technical:
            ambiguities.append(
                f"Duplicate technical execution identity: {technical_run_id}"
            )
            continue
        seen_technical.add(technical_run_id)
        spec = get_stage(stage)
        target = root / execution_folder_name(execution_id, stage)
        status = manifest.status
        if status not in {
            "allocated",
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
            "blocked",
            "unknown",
        }:
            status = "unknown"
        record = ExecutionRecord(
            execution_id=execution_id,
            execution_label=execution_label(execution_id),
            original_execution_id=execution_id,
            technical_run_id=technical_run_id,
            workflow_run_id=workflow_run_id,
            stage=stage,
            stage_display_name=spec.display_name,
            output_slug=spec.output_slug,
            output_folder=_stored_path(context, target),
            status=status,
            asset_effect=manifest.asset_effect,
            created_at=manifest.started_at,
            started_at=manifest.started_at,
            completed_at=manifest.completed_at,
            slurm_job_id=manifest.slurm_job_id,
        )
        records.append(
            MigrationRecord(
                source_folder=source,
                target_folder=target,
                execution=record,
                manifest_path=source / "stage_manifest.yaml",
            )
        )

    return ExecutionMigrationPlan(
        project_id=context.project_metadata.project_id,
        created_at=utc_now(),
        legacy_layout_detected=bool(legacy_roots),
        safe_to_apply=not ambiguities,
        records=records,
        ambiguities=ambiguities,
    )


def apply_execution_layout_migration(
    context: ProjectContext,
    plan: ExecutionMigrationPlan,
) -> MigrationAudit:
    if not plan.safe_to_apply:
        raise ValueError("Migration plan is ambiguous: " + " ".join(plan.ambiguities))
    with execution_lock(context):
        current = load_execution_index(context)
        if current.executions:
            raise ValueError("Execution index changed after migration planning.")
        migration_id = f"{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        staging_root = context.state_dir / "migration-staging" / migration_id
        staged: dict[str, Path] = {}
        placed: dict[str, Path] = {}
        old_paths = {
            item.execution.technical_run_id: item.source_folder
            for item in plan.records
        }
        try:
            staging_root.mkdir(parents=True, exist_ok=False)
            for position, item in enumerate(plan.records, start=1):
                destination = staging_root / f"{position:03d}"
                item.source_folder.replace(destination)
                staged[item.execution.technical_run_id] = destination
                if not (destination / "stage_manifest.yaml").is_file():
                    raise ValueError(
                        f"Staged execution lost its manifest: {destination}"
                    )

            for legacy_name in LEGACY_STAGE_FOLDERS.values():
                legacy_root = outputs_root(context) / legacy_name
                if legacy_root.is_dir():
                    remaining = [
                        path for path in legacy_root.iterdir() if path.name != "README.md"
                    ]
                    if remaining:
                        raise ValueError(
                            f"Legacy folder still contains unmigrated data: {legacy_root}"
                        )
                    shutil.rmtree(legacy_root)

            for item in plan.records:
                source = staged[item.execution.technical_run_id]
                target = item.target_folder
                if target.exists():
                    raise FileExistsError(f"Migration target already exists: {target}")
                source.replace(target)
                placed[item.execution.technical_run_id] = target

            index = ExecutionIndex(
                project_id=context.project_metadata.project_id,
                updated_at=utc_now(),
                executions=[item.execution for item in plan.records],
            )
            write_execution_index(context, index)
            rewrite_execution_records(context, index.executions, old_paths)
            audit = MigrationAudit(migrated_at=utc_now(), records=plan.records)
            audit_path = (
                context.root
                / MIGRATION_AUDIT_DIRECTORY
                / f"{migration_id}.yaml"
            )
            write_yaml(audit_path, audit)
            shutil.rmtree(staging_root, ignore_errors=True)
            return audit
        except Exception:
            write_execution_index(context, current)
            for item in reversed(plan.records):
                source = item.source_folder
                current_path = placed.get(item.execution.technical_run_id)
                if current_path is None or not current_path.exists():
                    current_path = staged.get(item.execution.technical_run_id)
                if current_path is not None and current_path.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    current_path.replace(source)
            shutil.rmtree(staging_root, ignore_errors=True)
            raise


__all__ = [
    "LEGACY_STAGE_FOLDERS",
    "MIGRATION_AUDIT_DIRECTORY",
    "apply_execution_layout_migration",
    "plan_execution_layout_migration",
]
