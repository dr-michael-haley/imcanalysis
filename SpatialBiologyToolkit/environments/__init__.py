"""Lightweight fixed-name Conda environment management for ``sbt``."""

from .manager import EnvironmentManager
from .registry import (
    environment_keys_for_stage,
    load_environment_registry,
    resolve_environment,
)

__all__ = [
    "EnvironmentManager",
    "environment_keys_for_stage",
    "load_environment_registry",
    "resolve_environment",
]
