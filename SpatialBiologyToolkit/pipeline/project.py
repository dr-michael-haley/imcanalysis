"""SBT project identity, discovery, initialization, and validation."""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from SpatialBiologyToolkit.config import PipelineConfig, load_config

from .assets import (
    asset_map,
    count_raw_imc_files,
    inventory_assets,
    resolve_assets,
    resolve_project_path,
    unexpected_top_level_paths,
)
from .manifests import read_model, utc_now, write_yaml
from .models import (
    ProjectAsset,
    ProjectMetadata,
    ProjectValidationReport,
    StageSpec,
    ValidationItem,
)


PROJECT_MARKER = Path(".sbt/project.yaml")
RUNS_DIRECTORY = Path(".sbt/runs")
INITIAL_ASSET_INVENTORY = Path(".sbt/project_assets.initial.yaml")


class ProjectError(RuntimeError):
    """Base error for project discovery and ownership failures."""


class ProjectNotFoundError(ProjectError):
    pass


class ProjectNotInitializedError(ProjectError):
    pass


@dataclass(frozen=True)
class ProjectContext:
    root: Path
    config_path: Path
    config: PipelineConfig
    state_dir: Path
    runs_dir: Path
    project_metadata: ProjectMetadata


@dataclass(frozen=True)
class AdoptionResult:
    context: ProjectContext
    assets: list[ProjectAsset]
    unexpected_paths: list[Path]


def _ancestors(path: Path) -> list[Path]:
    resolved = path.expanduser().resolve(strict=False)
    if resolved.is_file():
        resolved = resolved.parent
    return [resolved, *resolved.parents]


def discover_project_root(start: str | Path | None = None) -> Path:
    search_start = Path(start) if start is not None else Path.cwd()
    ancestors = _ancestors(search_start)
    marker_matches = [path for path in ancestors if (path / PROJECT_MARKER).is_file()]
    if marker_matches:
        return marker_matches[0]

    config_matches = [path for path in ancestors if (path / "config.yaml").is_file()]
    if len(config_matches) == 1:
        return config_matches[0]
    if len(config_matches) > 1:
        rendered = ", ".join(str(path) for path in config_matches)
        raise ProjectNotFoundError(
            "Multiple ancestor directories contain config.yaml and no .sbt marker "
            f"selects project ownership: {rendered}. Use --project explicitly."
        )
    raise ProjectNotFoundError(
        f"No SBT project marker or unambiguous config.yaml found from {search_start}."
    )


def _resolve_root(
    project: str | Path | None,
    *,
    start: str | Path | None = None,
) -> Path:
    if project is not None:
        return Path(project).expanduser().resolve(strict=False)
    return discover_project_root(start)


def _resolve_config_path(
    root: Path,
    metadata: ProjectMetadata,
    config_override: str | Path | None,
) -> Path:
    configured = (
        Path(config_override)
        if config_override is not None
        else Path(metadata.config_file)
    )
    if not configured.is_absolute():
        configured = root / configured
    return configured.expanduser().resolve(strict=False)


def load_project(
    project: str | Path | None = None,
    *,
    start: str | Path | None = None,
    config_override: str | Path | None = None,
) -> ProjectContext:
    root = _resolve_root(project, start=start)
    marker_path = root / PROJECT_MARKER
    if not marker_path.is_file():
        raise ProjectNotInitializedError(
            f"{root} has no {PROJECT_MARKER}. Run 'sbt project adopt' first."
        )
    metadata = read_model(marker_path, ProjectMetadata)
    config_path = _resolve_config_path(root, metadata, config_override)
    config = load_config(config_path)
    return ProjectContext(
        root=root,
        config_path=config_path,
        config=config,
        state_dir=(root / ".sbt").resolve(strict=False),
        runs_dir=(root / RUNS_DIRECTORY).resolve(strict=False),
        project_metadata=metadata,
    )


def _config_template(config_level: str) -> dict:
    config = PipelineConfig()
    if config_level == "complete":
        return config.model_dump(mode="python")
    if config_level != "basic":
        raise ValueError("config level must be 'basic' or 'complete'")

    template: dict[str, dict] = {}
    for section_name in PipelineConfig.model_fields:
        section = getattr(config, section_name)
        selected: dict = {}
        for field_name, model_field in section.__class__.model_fields.items():
            metadata = model_field.json_schema_extra or {}
            if metadata.get("level") == "basic":
                selected[field_name] = getattr(section, field_name)
        if selected:
            template[section_name] = selected
    return template


def write_config_template(
    output_path: str | Path,
    *,
    config_level: str = "basic",
    force: bool = False,
) -> Path:
    destination = Path(output_path).expanduser().resolve(strict=False)
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing config: {destination}")
    write_yaml(destination, _config_template(config_level))
    return destination


def _new_metadata(config_file: str) -> ProjectMetadata:
    return ProjectMetadata(
        project_id=str(uuid.uuid4()),
        created_at=utc_now(),
        config_file=config_file,
    )


def _relative_config_name(root: Path, config_path: Path) -> str:
    try:
        return str(config_path.relative_to(root))
    except ValueError:
        return str(config_path)


def initialize_project(
    root: str | Path,
    *,
    config_name: str = "config.yaml",
    config_level: str = "basic",
    force: bool = False,
) -> ProjectContext:
    project_root = Path(root).expanduser().resolve(strict=False)
    project_root.mkdir(parents=True, exist_ok=True)
    config_path = (project_root / config_name).resolve(strict=False)
    if project_root not in config_path.parents:
        raise ValueError(
            "The initialized project config must be inside the project root."
        )
    marker_path = project_root / PROJECT_MARKER
    if marker_path.exists() and not force:
        raise FileExistsError(f"Project is already initialized: {marker_path}")
    write_config_template(config_path, config_level=config_level, force=force)
    config = load_config(config_path)
    resolve_project_path(project_root, config.general.imc_files_folder).mkdir(
        parents=True, exist_ok=True
    )
    resolve_project_path(project_root, config.general.metadata_folder).mkdir(
        parents=True, exist_ok=True
    )
    (project_root / RUNS_DIRECTORY).mkdir(parents=True, exist_ok=True)
    metadata = _new_metadata(_relative_config_name(project_root, config_path))
    write_yaml(marker_path, metadata)
    return load_project(project_root)


def adopt_project(
    root: str | Path,
    *,
    config_path: str | Path = "config.yaml",
    force: bool = False,
) -> AdoptionResult:
    project_root = Path(root).expanduser().resolve(strict=False)
    if not project_root.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project_root}")
    source_config = Path(config_path)
    if not source_config.is_absolute():
        source_config = project_root / source_config
    source_config = source_config.expanduser().resolve(strict=False)
    config = load_config(source_config)

    marker_path = project_root / PROJECT_MARKER
    if marker_path.is_file():
        existing = read_model(marker_path, ProjectMetadata)
        if (
            not force
            and _resolve_config_path(project_root, existing, None) != source_config
        ):
            raise FileExistsError(
                f"Project marker already points to {existing.config_file}; use --force "
                "to update the config reference while preserving project identity."
            )
        metadata = existing.model_copy(
            update={"config_file": _relative_config_name(project_root, source_config)}
        )
    else:
        metadata = _new_metadata(_relative_config_name(project_root, source_config))

    (project_root / RUNS_DIRECTORY).mkdir(parents=True, exist_ok=True)
    write_yaml(marker_path, metadata)
    context = load_project(project_root)
    assets = resolve_assets(config, project_root)
    initial_inventory = project_root / INITIAL_ASSET_INVENTORY
    if force or not initial_inventory.exists():
        write_yaml(
            initial_inventory,
            inventory_assets(
                project_id=metadata.project_id,
                project_root=project_root,
                config=config,
            ),
        )
    return AdoptionResult(
        context=context,
        assets=assets,
        unexpected_paths=unexpected_top_level_paths(
            project_root,
            assets,
            config_path=source_config,
        ),
    )


def copy_user_config(context: ProjectContext, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(context.config_path, destination)


def _required_file_paths(
    stage: StageSpec, assets: dict[str, ProjectAsset]
) -> list[Path]:
    paths: list[Path] = []
    for role, names in stage.required_files.items():
        asset = assets.get(role)
        if asset is None:
            continue
        paths.extend(asset.path / name for name in names)
    return paths


def stage_readiness(
    stage: StageSpec,
    assets: Iterable[ProjectAsset],
) -> tuple[bool, list[str]]:
    mapped = asset_map(assets)
    messages: list[str] = []
    for role in stage.requires_assets:
        asset = mapped.get(role)
        if asset is None or not asset.exists:
            messages.append(f"Required asset '{role}' is missing.")
            continue
        if asset.kind == "directory" and not asset.file_count:
            messages.append(f"Required asset '{role}' exists but is empty.")
        if role == "raw_imc_files" and count_raw_imc_files(asset.path) == 0:
            messages.append(
                "Raw IMC folder contains no recognized .mcd or .txt input files."
            )
    for path in _required_file_paths(stage, mapped):
        if not path.is_file():
            messages.append(f"Required file is missing: {path}")
    return not messages, messages


def validate_project(
    context: ProjectContext,
    *,
    stages: Iterable[StageSpec] = (),
) -> ProjectValidationReport:
    assets = resolve_assets(context.config, context.root)
    mapped = asset_map(assets)
    raw_asset = mapped["raw_imc_files"]
    metadata_asset = mapped["metadata"]
    raw_count = count_raw_imc_files(raw_asset.path)

    required = [
        ValidationItem(
            name="project marker",
            path=context.root / PROJECT_MARKER,
            status="ok",
            message="Versioned SBT project identity is present.",
        ),
        ValidationItem(
            name="configuration",
            path=context.config_path,
            status="ok",
            message="Configuration loaded successfully through the Pydantic schema.",
        ),
        ValidationItem(
            name="run records directory",
            path=context.runs_dir,
            status="ok" if context.runs_dir.is_dir() else "missing",
            message=(
                "Project-scoped run storage is present."
                if context.runs_dir.is_dir()
                else "Run storage is missing; re-run project adoption or initialization."
            ),
        ),
        ValidationItem(
            name="raw IMC folder",
            path=raw_asset.path,
            status="ok" if raw_asset.exists else "missing",
            message=(
                f"{raw_count} recognized MCD/TXT input file(s)."
                if raw_asset.exists
                else "Configured raw IMC folder does not exist."
            ),
        ),
    ]

    optional = [
        ValidationItem(
            name="metadata folder",
            path=metadata_asset.path,
            status=(
                "ok"
                if metadata_asset.exists and metadata_asset.file_count
                else "warning"
            ),
            message=(
                f"{metadata_asset.file_count} top-level item(s)."
                if metadata_asset.exists and metadata_asset.file_count
                else "No metadata files identified yet."
            ),
        ),
        ValidationItem(
            name="panel metadata",
            path=metadata_asset.path / "panel.csv",
            status="ok" if (metadata_asset.path / "panel.csv").is_file() else "warning",
            message=(
                "panel.csv is present."
                if (metadata_asset.path / "panel.csv").is_file()
                else "No panel.csv identified yet."
            ),
        ),
        ValidationItem(
            name="sample metadata",
            path=metadata_asset.path / "metadata.csv",
            status=(
                "ok" if (metadata_asset.path / "metadata.csv").is_file() else "warning"
            ),
            message=(
                "metadata.csv is present."
                if (metadata_asset.path / "metadata.csv").is_file()
                else "No sample metadata.csv identified yet."
            ),
        ),
    ]

    generated = [
        ValidationItem(
            name=asset.role,
            path=asset.path,
            status="ok" if asset.exists else "not_created",
            message=(
                "Present."
                if asset.exists
                else "Pipeline-generated asset has not been created."
            ),
        )
        for asset in assets
        if asset.lifecycle == "generated_output"
    ]

    stage_results: dict[str, bool] = {}
    readiness_messages: dict[str, list[str]] = {}
    for stage in stages:
        ready, messages = stage_readiness(stage, assets)
        stage_results[stage.name] = ready
        readiness_messages[stage.name] = messages

    return ProjectValidationReport(
        project_id=context.project_metadata.project_id,
        project_root=context.root,
        valid=all(item.status != "missing" for item in required),
        required_inputs=required,
        optional_inputs=optional,
        generated_assets=generated,
        stage_readiness=stage_results,
        readiness_messages=readiness_messages,
    )


__all__ = [
    "AdoptionResult",
    "INITIAL_ASSET_INVENTORY",
    "PROJECT_MARKER",
    "ProjectContext",
    "ProjectError",
    "ProjectNotFoundError",
    "ProjectNotInitializedError",
    "RUNS_DIRECTORY",
    "adopt_project",
    "copy_user_config",
    "discover_project_root",
    "initialize_project",
    "load_project",
    "stage_readiness",
    "validate_project",
    "write_config_template",
]
