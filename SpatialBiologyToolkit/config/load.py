"""Load user YAML into the typed pipeline configuration."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .models import PipelineConfig


def _warn_unknown_keys(data: Mapping[str, Any]) -> None:
    """Match the legacy loader's permissive handling of unknown keys."""
    known_sections = PipelineConfig.model_fields
    for section, section_data in data.items():
        if section not in known_sections:
            logging.warning("Ignoring unrecognized config section '%s'.", section)
            continue
        annotation = known_sections[section].annotation
        section_fields = getattr(annotation, "model_fields", None)
        if section_fields is None or not isinstance(section_data, Mapping):
            continue
        for key in section_data:
            if key not in section_fields:
                logging.warning(
                    "Ignoring unrecognized config key '%s' in the '%s' section.",
                    key,
                    section,
                )


def load_config_data(data: Mapping[str, Any] | None = None) -> PipelineConfig:
    """Validate a mapping and fill every omitted value from model defaults."""
    raw_data = dict(data or {})
    _warn_unknown_keys(raw_data)
    return PipelineConfig.model_validate(raw_data)


def read_config_mapping(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file as a mapping without validating or modifying it."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in configuration file {config_path}: {exc}") from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ValueError(
            f"Configuration file {config_path} must contain a YAML mapping at its root."
        )
    return dict(loaded)


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a YAML config file without modifying the input file."""
    return load_config_data(read_config_mapping(path))


def config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    """Return a plain, fully resolved dictionary for legacy consumers."""
    return config.model_dump(mode="python")


__all__ = [
    "config_to_dict",
    "load_config",
    "load_config_data",
    "read_config_mapping",
]
