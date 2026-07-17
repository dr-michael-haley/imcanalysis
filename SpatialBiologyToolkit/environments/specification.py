"""Repository environment specification validation and package parsing."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .models import (
    CondaPackageRecord,
    EnvironmentDefinition,
    EnvironmentPaths,
    PipPackageRecord,
    SpecificationIssue,
    SpecificationValidation,
)
from .registry import REGISTRY_RELATIVE_PATH
from .runtime import TOOLKIT_NAMES, normalize_package_name


EDITABLE_PATTERN = re.compile(
    r"(?:^|\s)(?:-e\s+|--editable\s+)|spatialbiologytoolkit\s*@\s*file:|file:.*imcanalysis",
    re.IGNORECASE,
)
REQUIREMENT_PATTERN = re.compile(
    r"^\s*([A-Za-z0-9_.-]+)\s*((?:==|~=|>=|<=|!=|>|<|=).*)?$"
)


def environment_paths(
    repository_root: Path,
    registry_path: Path,
    definition: EnvironmentDefinition,
) -> EnvironmentPaths:
    if definition.specification_directory is None:
        return EnvironmentPaths(registry=registry_path)
    directory = definition.specification_directory.expanduser()
    if not directory.is_absolute():
        directory = repository_root / directory
    directory = directory.resolve(strict=False)
    try:
        directory.relative_to(repository_root.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(
            f"Environment specification directory escapes repository root: {directory}"
        ) from exc
    return EnvironmentPaths(
        registry=registry_path,
        specification_directory=directory,
        environment_yml=directory / "environment.yml",
        lockfile=directory / f"conda-{definition.platform}.lock",
        pip_extras=directory / "pip-extras.txt",
        observed_snapshot=directory / "environment.snapshot.json",
    )


def load_environment_yml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read environment YAML {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Environment YAML must contain a mapping: {path}")
    return loaded


def _requirement_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def validate_specification(
    *,
    key: str,
    definition: EnvironmentDefinition,
    repository_root: Path,
    registry_path: Path | None = None,
) -> SpecificationValidation:
    registry_file = (registry_path or repository_root / REGISTRY_RELATIVE_PATH).resolve(
        strict=False
    )
    paths = environment_paths(repository_root, registry_file, definition)
    issues: list[SpecificationIssue] = []
    if not definition.managed:
        issues.append(
            SpecificationIssue(
                severity="warning",
                code="external_environment",
                message="Environment is externally managed and has no repository lock contract.",
            )
        )
        return SpecificationValidation(
            environment_key=key,
            conda_name=definition.conda_name,
            valid=True,
            paths=paths,
            issues=issues,
        )

    assert paths.specification_directory is not None
    assert paths.environment_yml is not None
    assert paths.lockfile is not None
    assert paths.pip_extras is not None
    if not paths.specification_directory.is_dir():
        issues.append(
            SpecificationIssue(
                severity="error",
                code="missing_specification_directory",
                message=f"Specification directory does not exist: {paths.specification_directory}",
                path=paths.specification_directory,
            )
        )
    data: dict[str, Any] = {}
    if not paths.environment_yml.is_file():
        issues.append(
            SpecificationIssue(
                severity="error",
                code="missing_environment_yml",
                message="Missing environment.yml.",
                path=paths.environment_yml,
            )
        )
    else:
        try:
            data = load_environment_yml(paths.environment_yml)
        except ValueError as exc:
            issues.append(
                SpecificationIssue(
                    severity="error",
                    code="invalid_environment_yml",
                    message=str(exc),
                    path=paths.environment_yml,
                )
            )
        if data:
            if data.get("name") != definition.conda_name:
                issues.append(
                    SpecificationIssue(
                        severity="error",
                        code="name_mismatch",
                        message=(
                            f"environment.yml name is {data.get('name')!r}; expected "
                            f"{definition.conda_name!r}."
                        ),
                        path=paths.environment_yml,
                    )
                )
            if "prefix" in data:
                issues.append(
                    SpecificationIssue(
                        severity="error",
                        code="prefix_not_allowed",
                        message="environment.yml must not contain a machine-specific prefix.",
                        path=paths.environment_yml,
                    )
                )
            dependencies = data.get("dependencies")
            if not isinstance(dependencies, list):
                issues.append(
                    SpecificationIssue(
                        severity="error",
                        code="invalid_dependencies",
                        message="environment.yml dependencies must be a list.",
                        path=paths.environment_yml,
                    )
                )
            else:
                for dependency in dependencies:
                    if isinstance(dependency, dict) and "pip" in dependency:
                        issues.append(
                            SpecificationIssue(
                                severity="error",
                                code="embedded_pip_dependencies",
                                message="Move pip requirements from environment.yml to pip-extras.txt.",
                                path=paths.environment_yml,
                            )
                        )
                    if EDITABLE_PATTERN.search(str(dependency)):
                        issues.append(
                            SpecificationIssue(
                                severity="error",
                                code="editable_toolkit_in_environment",
                                message="Editable/local toolkit installs are not allowed in environment.yml.",
                                path=paths.environment_yml,
                            )
                        )

    if not paths.lockfile.is_file():
        issues.append(
            SpecificationIssue(
                severity="error",
                code="missing_lockfile",
                message=f"Missing {paths.lockfile.name}; run `sbt env lock {key}`.",
                path=paths.lockfile,
            )
        )
    else:
        try:
            lock = yaml.safe_load(paths.lockfile.read_text(encoding="utf-8"))
            platforms = ((lock or {}).get("metadata") or {}).get("platforms", [])
            if definition.platform not in platforms:
                issues.append(
                    SpecificationIssue(
                        severity="error",
                        code="lock_platform_mismatch",
                        message=(
                            f"Lockfile does not declare target platform {definition.platform!r}."
                        ),
                        path=paths.lockfile,
                    )
                )
            if any(
                str(item.get("manager", "conda")).casefold() == "pip"
                for item in (lock or {}).get("package", [])
                if isinstance(item, dict)
            ):
                issues.append(
                    SpecificationIssue(
                        severity="error",
                        code="legacy_pip_in_lock",
                        message=(
                            "Lockfile contains legacy pip records; regenerate it after pip extras "
                            "have been separated before synchronization."
                        ),
                        path=paths.lockfile,
                    )
                )
        except (OSError, yaml.YAMLError, AttributeError) as exc:
            issues.append(
                SpecificationIssue(
                    severity="error",
                    code="invalid_lockfile",
                    message=f"Could not parse lockfile: {exc}",
                    path=paths.lockfile,
                )
            )

    if not paths.pip_extras.is_file():
        issues.append(
            SpecificationIssue(
                severity="warning",
                code="missing_pip_extras",
                message="pip-extras.txt is absent; an empty file is preferred for consistency.",
                path=paths.pip_extras,
            )
        )
    else:
        for line in _requirement_lines(paths.pip_extras):
            normalized = normalize_package_name(line.split(" @ ", 1)[0].split("==", 1)[0])
            if normalized in TOOLKIT_NAMES or EDITABLE_PATTERN.search(line):
                issues.append(
                    SpecificationIssue(
                        severity="error",
                        code="toolkit_in_pip_extras",
                        message=f"Toolkit editable/local requirement is not allowed: {line}",
                        path=paths.pip_extras,
                    )
                )
            elif line.startswith(("-e ", "--editable ", "file:", "/", ".")) or " @ file:" in line:
                issues.append(
                    SpecificationIssue(
                        severity="error",
                        code="local_pip_requirement",
                        message=f"Local pip requirement requires manual review: {line}",
                        path=paths.pip_extras,
                    )
                )
            elif "git+" in line:
                issues.append(
                    SpecificationIssue(
                        severity="warning",
                        code="vcs_pip_requirement",
                        message=f"VCS pip requirement should be reviewed: {line}",
                        path=paths.pip_extras,
                    )
                )
    return SpecificationValidation(
        environment_key=key,
        conda_name=definition.conda_name,
        valid=not any(issue.severity == "error" for issue in issues),
        paths=paths,
        issues=issues,
    )


def parse_requirement(value: str) -> tuple[str, str]:
    match = REQUIREMENT_PATTERN.match(value.strip())
    if not match:
        return normalize_package_name(value), ""
    return normalize_package_name(match.group(1)), (match.group(2) or "").strip()


def declared_conda_requirements(path: Path) -> dict[str, str]:
    data = load_environment_yml(path)
    result: dict[str, str] = {}
    for dependency in data.get("dependencies", []):
        if not isinstance(dependency, str):
            continue
        name, constraint = parse_requirement(dependency)
        result[name] = constraint
    return result


def locked_conda_packages(path: Path, platform: str) -> dict[str, CondaPackageRecord]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    result: dict[str, CondaPackageRecord] = {}
    for item in loaded.get("package", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("manager", "conda")).casefold() != "conda":
            continue
        if item.get("platform") not in {None, platform}:
            continue
        record = CondaPackageRecord(
            name=str(item.get("name", "")),
            version=str(item.get("version", "")),
            build=str(item.get("build", "")),
            channel=str(item.get("url") or item.get("channel") or ""),
        )
        result[normalize_package_name(record.name)] = record
    return result


def declared_pip_requirements(path: Path) -> dict[str, PipPackageRecord]:
    result: dict[str, PipPackageRecord] = {}
    for line in _requirement_lines(path):
        source_type = "index"
        if "git+" in line:
            source_type = "vcs"
        elif line.startswith(("-e ", "--editable ")):
            source_type = "editable"
        elif "file:" in line or line.startswith(("/", ".")):
            source_type = "local"
        candidate = line.removeprefix("-e ").removeprefix("--editable ")
        name, constraint = parse_requirement(candidate.split(" @ ", 1)[0])
        version = constraint.removeprefix("==") if constraint.startswith("==") else constraint
        result[name] = PipPackageRecord(
            name=name,
            version=version,
            editable=source_type == "editable",
            requirement=line,
            source_type=source_type,
        )
    return result


def _version_parts(value: str) -> tuple[tuple[int, object], ...]:
    parts: list[tuple[int, object]] = []
    for part in re.split(r"[.+_-]", value):
        parts.append((0, int(part)) if part.isdigit() else (1, part.casefold()))
    return tuple(parts)


def satisfies_constraint(version: str, constraint: str) -> bool:
    if not constraint:
        return True
    for raw in constraint.split(","):
        candidate = raw.strip()
        match = re.match(r"^(==|~=|>=|<=|!=|>|<|=)\s*(.+)$", candidate)
        if not match:
            continue
        operator, expected = match.groups()
        actual_parts = _version_parts(version)
        expected_parts = _version_parts(expected.rstrip(".*"))
        if operator in {"=", "=="}:
            passed = version == expected or (
                expected.endswith(".*") and version.startswith(expected[:-1])
            )
        elif operator == "!=":
            passed = version != expected
        elif operator == ">=":
            passed = actual_parts >= expected_parts
        elif operator == "<=":
            passed = actual_parts <= expected_parts
        elif operator == ">":
            passed = actual_parts > expected_parts
        elif operator == "<":
            passed = actual_parts < expected_parts
        else:  # ~=
            passed = actual_parts >= expected_parts and version.split(".", 1)[0] == expected.split(".", 1)[0]
        if not passed:
            return False
    return True


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = [
    "atomic_write_text",
    "atomic_write_bytes",
    "declared_conda_requirements",
    "declared_pip_requirements",
    "environment_paths",
    "load_environment_yml",
    "locked_conda_packages",
    "parse_requirement",
    "satisfies_constraint",
    "validate_specification",
]
