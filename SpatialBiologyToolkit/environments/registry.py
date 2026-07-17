"""Load and resolve the repository's central environment registry."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from .models import EnvironmentDefinition, EnvironmentRegistry


REGISTRY_RELATIVE_PATH = Path("HPC_env_files/environments.yaml")


def toolkit_root(explicit: str | Path | None = None) -> Path:
    configured = explicit or os.environ.get("SBT_TOOLKIT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve(strict=False)
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=8)
def _load_cached(path_text: str) -> EnvironmentRegistry:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"Environment registry not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Environment registry must contain a YAML mapping: {path}")
    return EnvironmentRegistry.model_validate(loaded)


def load_environment_registry(
    root: str | Path | None = None,
    *,
    registry_path: str | Path | None = None,
) -> EnvironmentRegistry:
    base = toolkit_root(root)
    path = Path(registry_path) if registry_path else base / REGISTRY_RELATIVE_PATH
    return _load_cached(str(path.expanduser().resolve(strict=False)))


def resolve_environment(
    registry: EnvironmentRegistry,
    selector: str,
) -> tuple[str, EnvironmentDefinition]:
    if selector in registry.environments:
        return selector, registry.environments[selector]
    matches = [
        (key, item)
        for key, item in registry.environments.items()
        if item.conda_name.casefold() == selector.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    choices = ", ".join(
        f"{key} ({item.conda_name})" for key, item in registry.environments.items()
    )
    raise KeyError(f"Unknown environment {selector!r}. Available: {choices}")


def environment_keys_for_stage(
    stage: str,
    *,
    root: str | Path | None = None,
) -> list[str]:
    return list(load_environment_registry(root).stage_environments.get(stage, []))


def associated_stages(registry: EnvironmentRegistry, key: str) -> list[str]:
    return [
        stage
        for stage, environment_keys in registry.stage_environments.items()
        if key in environment_keys
    ]


__all__ = [
    "REGISTRY_RELATIVE_PATH",
    "associated_stages",
    "environment_keys_for_stage",
    "load_environment_registry",
    "resolve_environment",
    "toolkit_root",
]
