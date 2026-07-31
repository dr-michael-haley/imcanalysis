"""Per-user SBT project registry stored safely inside ``~/.imc_config``."""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .manifests import read_model, write_text
from .models import ProjectMetadata, ProjectRegistry, RegisteredProject
from .project import PROJECT_MARKER


IMC_CONFIG_ENV = "SBT_IMC_CONFIG"
IMC_CONFIG_FILENAME = ".imc_config"
REGISTRY_VARIABLE = "SBT_PROJECTS_JSON"
REGISTRY_BLOCK_BEGIN = "# >>> SBT PROJECT REGISTRY >>>"
REGISTRY_BLOCK_END = "# <<< SBT PROJECT REGISTRY <<<"


class ProjectRegistryError(RuntimeError):
    """Raised when the central project registry cannot be parsed or updated."""


@dataclass(frozen=True)
class RegisteredProjectStatus:
    project: RegisteredProject
    available: bool
    issue: str | None = None


def imc_config_path(explicit: str | Path | None = None) -> Path:
    configured = explicit or os.environ.get(IMC_CONFIG_ENV)
    return (
        Path(configured or Path.home() / IMC_CONFIG_FILENAME)
        .expanduser()
        .resolve(strict=False)
    )


def _registry_assignment(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    try:
        tokens = shlex.split(stripped, comments=False, posix=True)
    except ValueError as exc:
        if REGISTRY_VARIABLE in stripped:
            raise ProjectRegistryError(
                f"Invalid {REGISTRY_VARIABLE} shell quoting: {exc}"
            ) from exc
        return None
    if tokens and tokens[0] == "export":
        tokens = tokens[1:]
    if len(tokens) != 1 or "=" not in tokens[0]:
        return None
    key, value = tokens[0].split("=", 1)
    return value if key == REGISTRY_VARIABLE else None


def load_project_registry(
    path: str | Path | None = None,
) -> ProjectRegistry:
    source = imc_config_path(path)
    if not source.is_file():
        return ProjectRegistry()
    payload: str | None = None
    for line in source.read_text(encoding="utf-8").splitlines():
        candidate = _registry_assignment(line)
        if candidate is not None:
            payload = candidate
    if payload is None:
        return ProjectRegistry()
    try:
        data = json.loads(payload)
        return ProjectRegistry.model_validate(data)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ProjectRegistryError(
            f"Invalid project registry in {source}: {exc}"
        ) from exc


def _without_registry_block(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    inside = False
    found_end = False
    for line in lines:
        if line.strip() == REGISTRY_BLOCK_BEGIN:
            if inside:
                raise ProjectRegistryError("Nested SBT project registry blocks found.")
            inside = True
            continue
        if line.strip() == REGISTRY_BLOCK_END:
            if not inside:
                raise ProjectRegistryError(
                    "SBT project registry end marker has no matching start marker."
                )
            inside = False
            found_end = True
            continue
        if inside:
            continue
        if _registry_assignment(line) is not None:
            continue
        output.append(line)
    if inside:
        raise ProjectRegistryError(
            "SBT project registry start marker has no matching end marker."
        )
    if found_end or output:
        return "\n".join(output).rstrip() + "\n"
    return ""


def write_project_registry(
    registry: ProjectRegistry,
    path: str | Path | None = None,
) -> Path:
    destination = imc_config_path(path)
    original = destination.read_text(encoding="utf-8") if destination.is_file() else ""
    original_mode = (
        destination.stat().st_mode & 0o777 if destination.exists() else 0o600
    )
    preserved = _without_registry_block(original)
    payload = json.dumps(
        registry.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    block = "\n".join(
        (
            REGISTRY_BLOCK_BEGIN,
            "# Managed by `sbt project register` and the Project Console.",
            f"export {REGISTRY_VARIABLE}={shlex.quote(payload)}",
            REGISTRY_BLOCK_END,
            "",
        )
    )
    if not preserved:
        preserved = (
            "# Per-user IMC settings. This file may also contain IMC_EMAIL "
            "and OPENAI_API_KEY exports.\n"
        )
    write_text(destination, preserved + block)
    try:
        destination.chmod(original_mode)
    except OSError:
        pass
    return destination


def _project_metadata(root: Path) -> ProjectMetadata:
    marker = root / PROJECT_MARKER
    if not marker.is_file():
        raise ProjectRegistryError(
            f"Existing SBT project marker not found: {marker}. Initialize or adopt it first."
        )
    try:
        return read_model(marker, ProjectMetadata)
    except (OSError, ValueError) as exc:
        raise ProjectRegistryError(f"Invalid project marker {marker}: {exc}") from exc


def register_project(
    project: str | Path,
    *,
    name: str | None = None,
    make_default: bool = False,
    registry_path: str | Path | None = None,
) -> tuple[ProjectRegistry, RegisteredProject]:
    root = Path(project).expanduser().resolve(strict=False)
    metadata = _project_metadata(root)
    display_name = (name or metadata.title or root.name).strip()
    if not display_name:
        raise ProjectRegistryError("Registered project name cannot be empty.")
    registry = load_project_registry(registry_path)
    replaced_default = any(
        item.project_id == registry.default_project_id
        and (item.project_id == metadata.project_id or item.path == root)
        for item in registry.projects
    )
    retained = [
        item
        for item in registry.projects
        if item.project_id != metadata.project_id and item.path != root
    ]
    conflict = next(
        (item for item in retained if item.name.casefold() == display_name.casefold()),
        None,
    )
    if conflict is not None:
        raise ProjectRegistryError(
            f"Project name '{display_name}' is already used by {conflict.path}."
        )
    registered = RegisteredProject(
        name=display_name,
        path=root,
        project_id=metadata.project_id,
    )
    projects = [*retained, registered]
    projects.sort(key=lambda item: item.name.casefold())
    default_id = registry.default_project_id
    if make_default or default_id is None or replaced_default:
        default_id = registered.project_id
    updated = ProjectRegistry(
        projects=projects,
        default_project_id=default_id,
    )
    write_project_registry(updated, registry_path)
    return updated, registered


def resolve_registered_project(
    registry: ProjectRegistry,
    reference: str | Path,
) -> RegisteredProject:
    text = str(reference)
    reference_path = Path(text).expanduser().resolve(strict=False)
    matches = [
        item
        for item in registry.projects
        if item.project_id == text
        or item.name.casefold() == text.casefold()
        or item.path == reference_path
    ]
    if len(matches) != 1:
        raise ProjectRegistryError(f"Registered project not found: {reference}")
    return matches[0]


def unregister_project(
    reference: str | Path,
    *,
    registry_path: str | Path | None = None,
) -> tuple[ProjectRegistry, RegisteredProject]:
    registry = load_project_registry(registry_path)
    selected = resolve_registered_project(registry, reference)
    projects = [
        item for item in registry.projects if item.project_id != selected.project_id
    ]
    default_id = registry.default_project_id
    if default_id == selected.project_id:
        default_id = projects[0].project_id if projects else None
    updated = ProjectRegistry(projects=projects, default_project_id=default_id)
    write_project_registry(updated, registry_path)
    return updated, selected


def set_default_project(
    reference: str | Path,
    *,
    registry_path: str | Path | None = None,
) -> tuple[ProjectRegistry, RegisteredProject]:
    registry = load_project_registry(registry_path)
    selected = resolve_registered_project(registry, reference)
    updated = registry.model_copy(update={"default_project_id": selected.project_id})
    write_project_registry(updated, registry_path)
    return updated, selected


def default_registered_project(registry: ProjectRegistry) -> RegisteredProject | None:
    if registry.default_project_id is None:
        return None
    return next(
        (
            item
            for item in registry.projects
            if item.project_id == registry.default_project_id
        ),
        None,
    )


def registered_project_statuses(
    registry: ProjectRegistry,
) -> list[RegisteredProjectStatus]:
    statuses: list[RegisteredProjectStatus] = []
    for project in registry.projects:
        try:
            metadata = _project_metadata(project.path)
            if metadata.project_id != project.project_id:
                raise ProjectRegistryError(
                    "Project marker identity differs from the registered identity."
                )
        except ProjectRegistryError as exc:
            statuses.append(
                RegisteredProjectStatus(project, available=False, issue=str(exc))
            )
        else:
            statuses.append(RegisteredProjectStatus(project, available=True))
    return statuses


__all__ = [
    "IMC_CONFIG_ENV",
    "IMC_CONFIG_FILENAME",
    "ProjectRegistryError",
    "REGISTRY_BLOCK_BEGIN",
    "REGISTRY_BLOCK_END",
    "REGISTRY_VARIABLE",
    "RegisteredProjectStatus",
    "default_registered_project",
    "imc_config_path",
    "load_project_registry",
    "register_project",
    "registered_project_statuses",
    "resolve_registered_project",
    "set_default_project",
    "unregister_project",
    "write_project_registry",
]
