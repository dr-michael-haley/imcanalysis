"""Backend-neutral stage expansion, dependency resolution, and readiness checks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .assets import asset_is_ready, asset_map, resolve_assets
from .models import PlannedStage, RunPlan, StageSpec
from .project import ProjectContext, validate_project
from .registry import MODE_REGISTRY, STAGE_REGISTRY, get_stage, stage_script_path


class PlanningError(RuntimeError):
    pass


def expand_requested(requested: Iterable[str]) -> list[str]:
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
        for dependency in stage.depends_on:
            visit(dependency)
        visiting.pop()
        visited.add(name)
        ordered.append(stage)

    for stage_name in stage_names:
        visit(stage_name)
    return ordered


def build_run_plan(
    context: ProjectContext,
    requested: Iterable[str],
    *,
    toolkit_directory: str | Path | None = None,
) -> RunPlan:
    requested_names = list(requested)
    expanded = expand_requested(requested_names)
    stages = resolve_dependencies(expanded)
    assets = resolve_assets(context.config, context.root)
    mapped_assets = asset_map(assets)
    available = {role for role, asset in mapped_assets.items() if asset_is_ready(asset)}
    produced_in_plan: set[str] = set()
    errors: list[str] = []
    warnings: list[str] = []
    planned: list[PlannedStage] = []
    project_validation = validate_project(context)
    if not project_validation.valid:
        errors.extend(
            f"Project validation failed: {item.message}"
            for item in project_validation.required_inputs
            if item.status == "missing"
        )

    for stage in stages:
        script = stage_script_path(stage, root=toolkit_directory)
        missing_assets = [
            role
            for role in stage.requires_assets
            if role not in available and role not in produced_in_plan
        ]
        missing_files: list[Path] = []
        for role, relative_paths in stage.required_files.items():
            if role in produced_in_plan:
                continue
            asset = mapped_assets.get(role)
            if asset is None:
                continue
            missing_files.extend(
                candidate
                for candidate in (asset.path / relative for relative in relative_paths)
                if not candidate.is_file()
            )

        if not script.is_file():
            errors.append(f"SLURM script for stage '{stage.name}' is missing: {script}")
        if missing_assets:
            errors.append(
                f"Stage '{stage.name}' is missing required project assets: "
                f"{', '.join(missing_assets)}"
            )
        if missing_files:
            errors.append(
                f"Stage '{stage.name}' is missing required files: "
                + ", ".join(str(path) for path in missing_files)
            )
        if not stage.runnable:
            errors.append(f"Stage '{stage.name}' is not runnable through this backend.")
        warnings.extend(f"{stage.name}: {note}" for note in stage.notes)

        planned.append(
            PlannedStage(
                name=stage.name,
                description=stage.description,
                slurm_script=script,
                depends_on=stage.depends_on,
                requires_assets=stage.requires_assets,
                produces_assets=stage.produces_assets,
                expected_outputs=stage.expected_outputs,
                script_exists=script.is_file(),
                missing_assets=missing_assets,
                missing_files=missing_files,
            )
        )
        produced_in_plan.update(stage.produces_assets)
        available.update(stage.produces_assets)

    return RunPlan(
        project_id=context.project_metadata.project_id,
        project_root=context.root,
        requested=requested_names,
        resolved_stages=planned,
        config_source=context.config_path,
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
