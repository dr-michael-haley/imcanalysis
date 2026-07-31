"""Asset-aware stage selection, lineage advice, and readiness checks."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable

from .assets import asset_is_ready, asset_map, resolve_assets
from .models import DependencyPolicy, PlannedStage, ProjectAsset, RunPlan, StageSpec
from .project import (
    ProjectContext,
    execution_requirement_is_ready,
    validate_project,
)
from .registry import MODE_REGISTRY, STAGE_REGISTRY, get_stage, stage_script_path


class PlanningError(RuntimeError):
    pass


def expand_requested(requested: Iterable[str]) -> list[str]:
    """Expand modes while preserving the user's declared stage order."""

    expanded: list[str] = []
    for name in requested:
        if name in MODE_REGISTRY:
            expanded.extend(MODE_REGISTRY[name].stages)
        elif name in STAGE_REGISTRY:
            expanded.append(name)
        else:
            get_stage(name)
    if not expanded:
        raise PlanningError("At least one stage or mode is required.")
    return expanded


def resolve_dependencies(stage_names: Iterable[str]) -> list[StageSpec]:
    """Return the complete conventional upstream lineage for selected stages."""

    ordered: list[StageSpec] = []
    visited: set[str] = set()
    visiting: list[str] = []

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            cycle = " -> ".join([*visiting, name])
            raise PlanningError(f"Stage dependency cycle detected: {cycle}")
        stage = get_stage(name)
        visiting.append(name)
        for dependency in [*stage.depends_on, *stage.required_executions]:
            visit(dependency)
        visiting.pop()
        visited.add(name)
        ordered.append(stage)

    for stage_name in stage_names:
        visit(stage_name)
    return ordered


def _unique_stage_names(stage_names: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for name in stage_names:
        stage = get_stage(name)
        if stage.name not in seen:
            ordered.append(stage.name)
            seen.add(stage.name)
    return ordered


def _order_selected(stage_names: Iterable[str]) -> list[StageSpec]:
    """Topologically order only selected stages without adding upstream stages."""

    requested = list(stage_names)
    selected = set(requested)
    ordered: list[StageSpec] = []
    visited: set[str] = set()
    visiting: list[str] = []

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            cycle = " -> ".join([*visiting, name])
            raise PlanningError(f"Selected stage dependency cycle detected: {cycle}")
        visiting.append(name)
        stage = get_stage(name)
        dependencies = [*stage.depends_on, *stage.required_executions]
        for dependency in dependencies:
            if dependency in selected:
                visit(dependency)
        visiting.pop()
        visited.add(name)
        ordered.append(stage)

    for stage_name in requested:
        visit(stage_name)
    return ordered


def _required_file_paths(
    stage: StageSpec,
    role: str,
    assets: dict[str, ProjectAsset],
) -> list[Path]:
    asset = assets.get(role)
    if asset is None:
        return []
    return [asset.path / relative for relative in stage.required_files.get(role, [])]


def _role_is_ready(
    stage: StageSpec,
    role: str,
    assets: dict[str, ProjectAsset],
    produced_in_plan: set[str],
) -> bool:
    if role in produced_in_plan:
        return True
    asset = assets.get(role)
    if asset is None or not asset_is_ready(asset):
        return False
    return all(path.is_file() for path in _required_file_paths(stage, role, assets))


def _nearest_upstream_producer(stage: StageSpec, role: str) -> StageSpec | None:
    """Find the nearest conventional ancestor that produces one asset role."""

    queue: deque[str] = deque(stage.depends_on)
    seen: set[str] = set()
    while queue:
        name = queue.popleft()
        if name in seen:
            continue
        seen.add(name)
        candidate = get_stage(name)
        if role in candidate.produces_assets:
            return candidate
        queue.extend(candidate.depends_on)
    return None


def _select_asset_aware(
    expanded: Iterable[str],
    assets: dict[str, ProjectAsset],
    context: ProjectContext,
) -> list[StageSpec]:
    """Add upstream producers only when a blocking direct input is not ready."""

    ordered: list[StageSpec] = []
    selected: set[str] = set()
    visiting: list[str] = []
    produced_in_plan: set[str] = set()

    def ensure(name: str) -> None:
        if name in selected:
            return
        if name in visiting:
            cycle = " -> ".join([*visiting, name])
            raise PlanningError(f"Stage dependency cycle detected: {cycle}")
        stage = get_stage(name)
        visiting.append(name)
        for required_stage, relative_paths in stage.required_executions.items():
            if not execution_requirement_is_ready(
                context,
                required_stage,
                relative_paths,
            ):
                ensure(required_stage)
        for role in stage.requires_assets:
            if _role_is_ready(stage, role, assets, produced_in_plan):
                continue
            producer = _nearest_upstream_producer(stage, role)
            if producer is not None:
                ensure(producer.name)
        visiting.pop()
        selected.add(stage.name)
        ordered.append(stage)
        produced_in_plan.update(stage.produces_assets)

    for stage_name in _unique_stage_names(expanded):
        ensure(stage_name)
    return _order_selected(stage.name for stage in ordered)


def _selected_stages(
    expanded: list[str],
    dependency_policy: DependencyPolicy,
    assets: dict[str, ProjectAsset],
    context: ProjectContext,
) -> list[StageSpec]:
    if dependency_policy == "all":
        return resolve_dependencies(expanded)
    if dependency_policy == "none":
        return _order_selected(_unique_stage_names(expanded))
    return _select_asset_aware(expanded, assets, context)


def _missing_requirements(
    stage: StageSpec,
    assets: dict[str, ProjectAsset],
    produced_in_plan: set[str],
) -> tuple[list[str], list[Path]]:
    missing_assets = [
        role
        for role in stage.requires_assets
        if not _role_is_ready(stage, role, assets, produced_in_plan)
    ]
    missing_files: list[Path] = []
    for role in stage.required_files:
        if role in produced_in_plan:
            continue
        missing_files.extend(
            path
            for path in _required_file_paths(stage, role, assets)
            if not path.is_file()
        )
    return missing_assets, missing_files


def _execution_dependencies(
    stage: StageSpec,
    selected_names: set[str],
    producer_by_role: dict[str, str],
) -> list[str]:
    dependencies = [
        dependency for dependency in stage.depends_on if dependency in selected_names
    ]
    for required_stage in stage.required_executions:
        if required_stage in selected_names and required_stage not in dependencies:
            dependencies.append(required_stage)
    for role in stage.requires_assets:
        producer = producer_by_role.get(role)
        if producer and producer != stage.name and producer not in dependencies:
            dependencies.append(producer)
    return dependencies


def build_run_plan(
    context: ProjectContext,
    requested: Iterable[str],
    *,
    toolkit_directory: str | Path | None = None,
    dependency_policy: DependencyPolicy = "assets",
    include_dependencies: bool | None = None,
) -> RunPlan:
    """Build a plan whose readiness is governed by direct blocking inputs.

    ``assets`` is the default policy: conventional upstream stages are advisory
    and are added only when they can produce a missing blocking asset. The legacy
    ``include_dependencies`` argument remains as a compatibility bridge, mapping
    ``True`` to ``all`` and ``False`` to ``none``.
    """

    if include_dependencies is not None:
        if dependency_policy != "assets":
            raise ValueError("Use dependency_policy or include_dependencies, not both.")
        dependency_policy = "all" if include_dependencies else "none"
    if dependency_policy not in {"assets", "none", "all"}:
        raise ValueError("dependency_policy must be 'assets', 'none', or 'all'.")

    requested_names = list(requested)
    expanded = expand_requested(requested_names)
    mapped_assets = asset_map(resolve_assets(context.config, context.root))
    stages = _selected_stages(
        expanded,
        dependency_policy,
        mapped_assets,
        context,
    )
    selected_names = {stage.name for stage in stages}
    conventional = resolve_dependencies(expanded)
    skipped_upstream = [
        stage.name for stage in conventional if stage.name not in selected_names
    ]

    errors: list[str] = []
    warnings: list[str] = []
    if dependency_policy == "none":
        warnings.append(
            "Automatic upstream producers are disabled. Only explicitly selected "
            "stages are planned; every blocking asset or managed execution must "
            "already exist or be produced by another selected stage."
        )
    elif dependency_policy == "all":
        warnings.append(
            "All conventional upstream stages are included even when their outputs "
            "already exist."
        )
    if skipped_upstream:
        warnings.append(
            "Conventional upstream stage(s) not scheduled: "
            f"{', '.join(skipped_upstream)}. This is advisory lineage only; direct "
            "blocking requirements determine readiness."
        )

    project_validation = validate_project(context)
    if not project_validation.valid:
        errors.extend(
            f"Project validation failed: {item.message}"
            for item in project_validation.required_inputs
            if item.status == "missing"
        )

    planned: list[PlannedStage] = []
    produced_in_plan: set[str] = set()
    producer_by_role: dict[str, str] = {}
    completed_in_plan: set[str] = set()
    for stage in stages:
        script = stage_script_path(stage, root=toolkit_directory)
        missing_assets, missing_files = _missing_requirements(
            stage, mapped_assets, produced_in_plan
        )
        missing_advisory = [
            role
            for role in stage.advisory_assets
            if not _role_is_ready(stage, role, mapped_assets, produced_in_plan)
        ]
        missing_executions = [
            required_stage
            for required_stage, relative_paths in stage.required_executions.items()
            if required_stage not in completed_in_plan
            and not execution_requirement_is_ready(
                context,
                required_stage,
                relative_paths,
            )
        ]
        execution_dependencies = _execution_dependencies(
            stage, selected_names, producer_by_role
        )
        conventional_for_stage = resolve_dependencies([stage.name])
        stage_skipped = [
            item.name
            for item in conventional_for_stage
            if item.name != stage.name and item.name not in selected_names
        ]

        if not script.is_file():
            errors.append(f"SLURM script for stage '{stage.name}' is missing: {script}")
        if missing_assets:
            errors.append(
                f"Stage '{stage.name}' is missing blocking project assets: "
                f"{', '.join(missing_assets)}"
            )
        if missing_files:
            errors.append(
                f"Stage '{stage.name}' is missing required files: "
                + ", ".join(str(path) for path in missing_files)
            )
        if missing_executions:
            errors.append(
                f"Stage '{stage.name}' is missing required managed executions: "
                + ", ".join(missing_executions)
            )
        if missing_advisory:
            warnings.append(
                f"{stage.name}: advisory asset(s) absent: "
                f"{', '.join(missing_advisory)}. Execution remains allowed."
            )
        if not stage.runnable:
            errors.append(f"Stage '{stage.name}' is not runnable through this backend.")
        warnings.extend(f"{stage.name}: {note}" for note in stage.notes)

        planned.append(
            PlannedStage(
                name=stage.name,
                description=stage.description,
                slurm_script=script,
                depends_on=execution_dependencies,
                requires_assets=stage.requires_assets,
                advisory_assets=stage.advisory_assets,
                produces_assets=stage.produces_assets,
                expected_outputs=stage.expected_outputs,
                script_exists=script.is_file(),
                missing_assets=missing_assets,
                missing_advisory_assets=missing_advisory,
                required_executions=stage.required_executions,
                missing_executions=missing_executions,
                missing_files=missing_files,
                skipped_upstream_stages=stage_skipped,
            )
        )
        for role in stage.produces_assets:
            producer_by_role[role] = stage.name
        produced_in_plan.update(stage.produces_assets)
        completed_in_plan.add(stage.name)

    return RunPlan(
        project_id=context.project_metadata.project_id,
        project_root=context.root,
        requested=requested_names,
        resolved_stages=planned,
        config_source=context.config_path,
        dependency_policy=dependency_policy,
        ready=not errors,
        errors=errors,
        warnings=warnings,
    )


__all__ = [
    "PlanningError",
    "build_run_plan",
    "expand_requested",
    "resolve_dependencies",
]
