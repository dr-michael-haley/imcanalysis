"""Plan and apply conservative cleanup of assets created by one execution."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import TYPE_CHECKING

from SpatialBiologyToolkit.reporting.models import StageManifest

from .assets import asset_map, resolve_assets
from .executions import (
    REMOVAL_AUDIT_DIRECTORY,
    execution_output_path,
    load_execution_index,
)
from .manifests import read_model, write_yaml
from .models import (
    AssetCleanupAudit,
    AssetCleanupItem,
    AssetCleanupPlan,
    AssetInventory,
    ExecutionRecord,
    RemovalAudit,
)
from .registry import get_stage

if TYPE_CHECKING:
    from .project import ProjectContext


ASSETS_BEFORE = "project_assets.before.yaml"
NON_REUSABLE_ROLES = {"human_outputs", "legacy_qc", "legacy_slurm_logs"}


def _within(path: Path, root: Path) -> bool:
    resolved = path.resolve(strict=False)
    project_root = root.resolve(strict=False)
    return resolved != project_root and project_root in resolved.parents


def _stage_manifest(
    context: ProjectContext,
    execution: ExecutionRecord,
) -> StageManifest | None:
    candidates = (
        execution_output_path(context, execution) / "stage_manifest.yaml",
        context.runs_dir
        / execution.workflow_run_id
        / "stage_events"
        / f"{execution.technical_run_id}.yaml",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            return read_model(path, StageManifest)
        except (OSError, ValueError):
            continue
    return None


def _before_inventory(
    context: ProjectContext,
    execution: ExecutionRecord,
) -> AssetInventory | None:
    path = context.runs_dir / execution.workflow_run_id / ASSETS_BEFORE
    try:
        inventory = read_model(path, AssetInventory)
    except (OSError, ValueError):
        return None
    if (
        inventory.project_id != context.project_metadata.project_id
        or inventory.project_root.resolve(strict=False)
        != context.root.resolve(strict=False)
    ):
        return None
    return inventory


def _remaining_claims(
    context: ProjectContext,
    excluded_technical_run_ids: Collection[str],
) -> dict[str, list[str]]:
    claims: dict[str, list[str]] = {}
    for execution in load_execution_index(context, require_exists=True).executions:
        if execution.technical_run_id in excluded_technical_run_ids:
            continue
        spec = get_stage(execution.stage)
        for role in sorted(set(spec.requires_assets) | set(spec.produces_assets)):
            kind = "requires" if role in spec.requires_assets else "produces"
            claims.setdefault(role, []).append(
                f"{execution.execution_label} {execution.stage} ({kind})"
            )
    return claims


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return bool(checker and checker())


def _protected_paths(
    path: Path,
    project_root: Path,
) -> tuple[list[Path], list[Path], bool]:
    """Return protected AnnData/linked paths and whether anything is removable."""
    if path.is_symlink():
        if path.suffix.lower() == ".h5ad":
            return [path], [], False
        return [], [], True
    if _is_junction(path):
        return [], [path.absolute()], False
    if path.is_file():
        if path.suffix.lower() == ".h5ad":
            return [path], [], False
        return [], [], True
    if not path.is_dir():
        return [], [], False

    protected_h5ad: list[Path] = []
    protected_links: list[Path] = []
    removable = False
    for child in path.iterdir():
        if child.is_symlink():
            if child.suffix.lower() == ".h5ad":
                protected_h5ad.append(child.absolute())
            else:
                removable = True
            continue
        if _is_junction(child) or not _within(child, project_root):
            protected_links.append(child.absolute())
            continue
        child_h5ad, child_links, child_removable = _protected_paths(
            child,
            project_root,
        )
        protected_h5ad.extend(child_h5ad)
        protected_links.extend(child_links)
        removable = removable or child_removable
    if not protected_h5ad and not protected_links and not removable:
        removable = True  # The empty directory itself can be removed.
    return (
        sorted(set(protected_h5ad), key=lambda item: str(item).lower()),
        sorted(set(protected_links), key=lambda item: str(item).lower()),
        removable,
    )


def plan_asset_cleanup(
    context: ProjectContext,
    selected: ExecutionRecord,
    *,
    excluded_technical_run_ids: Collection[str] = (),
) -> AssetCleanupPlan:
    """Identify created, unused asset roots while retaining uncertain/shared data."""
    spec = get_stage(selected.stage)
    roles = [
        role for role in spec.produces_assets if role not in NON_REUSABLE_ROLES
    ]
    plan = AssetCleanupPlan(
        execution_id=selected.execution_id,
        technical_run_id=selected.technical_run_id,
        stage=selected.stage,
    )
    if not roles:
        return plan

    before = _before_inventory(context, selected)
    before_by_role = asset_map(before.assets) if before is not None else {}
    current_by_role = asset_map(resolve_assets(context.config, context.root))
    manifest = _stage_manifest(context, selected)
    manifest_by_role = (
        {item.role: item.path.resolve(strict=False) for item in manifest.produced_assets}
        if manifest is not None
        else {}
    )
    excluded = set(excluded_technical_run_ids)
    excluded.add(selected.technical_run_id)
    claims = _remaining_claims(context, excluded)

    for role in roles:
        previous = before_by_role.get(role)
        path = manifest_by_role.get(role)
        if path is None:
            current = current_by_role.get(role)
            path = current.path if current is not None else (
                previous.path if previous is not None else None
            )
        if path is None:
            plan.retained.append(
                AssetCleanupItem(
                    role=role,
                    path=context.root,
                    reason="ownership could not be resolved",
                )
            )
            continue
        path = path.resolve(strict=False)

        if previous is None:
            plan.retained.append(
                AssetCleanupItem(
                    role=role,
                    path=path,
                    reason="pre-run ownership evidence is unavailable",
                )
            )
            continue
        if previous.path.resolve(strict=False) != path:
            plan.retained.append(
                AssetCleanupItem(
                    role=role,
                    path=path,
                    reason="asset path differs from the pre-run inventory",
                )
            )
            continue
        if previous.exists:
            plan.retained.append(
                AssetCleanupItem(
                    role=role,
                    path=path,
                    reason="asset existed before this workflow",
                )
            )
            continue
        if not (path.exists() or path.is_symlink()):
            continue
        if not _within(path, context.root):
            plan.retained.append(
                AssetCleanupItem(
                    role=role,
                    path=path,
                    reason="asset is outside the project root",
                )
            )
            continue
        if claims.get(role):
            plan.retained.append(
                AssetCleanupItem(
                    role=role,
                    path=path,
                    reason="used by remaining stages",
                    dependent_stages=claims[role],
                )
            )
            continue
        try:
            protected_h5ad, protected_links, has_removable = _protected_paths(
                path,
                context.root,
            )
        except OSError as exc:
            plan.retained.append(
                AssetCleanupItem(
                    role=role,
                    path=path,
                    reason=f"asset could not be inspected: {exc}",
                )
            )
            continue
        for protected in protected_h5ad:
            plan.retained.append(
                AssetCleanupItem(
                    role=role,
                    path=protected,
                    reason=".h5ad files are always protected",
                )
            )
        for protected in protected_links:
            plan.retained.append(
                AssetCleanupItem(
                    role=role,
                    path=protected,
                    reason="linked directories and paths outside the project are protected",
                )
            )
        if has_removable:
            plan.removable.append(
                AssetCleanupItem(
                    role=role,
                    path=path,
                    reason="created by this workflow and unused by remaining stages",
                )
            )
    return plan


def cleanup_audit(
    plan: AssetCleanupPlan,
    *,
    offered: bool,
    confirmed: bool,
) -> AssetCleanupAudit:
    return AssetCleanupAudit(
        offered=offered,
        confirmed=confirmed,
        removable=plan.removable,
        retained=plan.retained,
    )


def _remove_except_h5ad(path: Path, project_root: Path) -> int:
    if path.suffix.lower() == ".h5ad" and (path.is_file() or path.is_symlink()):
        return 0
    if path.is_symlink():
        path.unlink()
        return 1
    if _is_junction(path) or not _within(path, project_root):
        return 0
    if path.is_file():
        if path.suffix.lower() == ".h5ad":
            return 0
        path.unlink()
        return 1
    if not path.is_dir():
        return 0

    removed = 0
    for child in list(path.iterdir()):
        removed += _remove_except_h5ad(child, project_root)
    try:
        path.rmdir()
        removed += 1
    except OSError:
        pass
    return removed


def apply_asset_cleanup(
    context: ProjectContext,
    audit: RemovalAudit,
    plan: AssetCleanupPlan,
) -> RemovalAudit:
    """Apply a confirmed plan and update the durable removal audit."""
    if audit.technical_run_id != plan.technical_run_id:
        raise ValueError("Asset cleanup plan does not match the removal audit.")

    cleaned: list[Path] = []
    errors: list[str] = []
    removed_entries = 0
    for item in plan.removable:
        path = item.path.resolve(strict=False)
        if not _within(path, context.root):
            errors.append(f"Refused path outside project root: {path}")
            continue
        try:
            removed = _remove_except_h5ad(path, context.root)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        if removed:
            cleaned.append(path)
            removed_entries += removed

    updated_cleanup = cleanup_audit(plan, offered=True, confirmed=True).model_copy(
        update={
            "cleaned_paths": cleaned,
            "removed_entries": removed_entries,
            "errors": errors,
        }
    )
    updated = audit.model_copy(update={"asset_cleanup": updated_cleanup})
    audit_path = (
        context.root / REMOVAL_AUDIT_DIRECTORY / f"{audit.audit_id}.yaml"
    )
    write_yaml(audit_path, updated)
    return updated


__all__ = [
    "apply_asset_cleanup",
    "cleanup_audit",
    "plan_asset_cleanup",
]
