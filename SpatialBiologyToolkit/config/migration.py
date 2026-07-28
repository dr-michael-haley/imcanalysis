"""Helpers for migrating verbose legacy YAML to compact typed configuration."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .load import load_config_data, read_config_mapping
from .models import PipelineConfig


# These fields are accepted only to migrate old YAML into their canonical fields.
# They must not be emitted into a newly compacted configuration.
_DEPRECATED_ALIAS_FIELDS: dict[str, frozenset[str]] = {
    "batch_integration": frozenset({"batch_correction_method"}),
    "biobatchnet": frozenset(
        {
            "biobatchnet_data_type",
            "biobatchnet_latent_dim",
            "biobatchnet_epochs",
            "biobatchnet_device",
            "biobatchnet_kwargs",
            "biobatchnet_use_raw",
        }
    ),
    "process": frozenset(
        {
            "biobatchnet_data_type",
            "biobatchnet_latent_dim",
            "biobatchnet_epochs",
            "biobatchnet_device",
            "biobatchnet_kwargs",
            "biobatchnet_use_raw",
        }
    ),
}


def _canonical_config(config: PipelineConfig) -> PipelineConfig:
    """Remove deprecated input aliases after their values have been migrated."""
    resolved = config.model_dump(mode="python")
    for section, aliases in _DEPRECATED_ALIAS_FIELDS.items():
        section_data = resolved.get(section)
        if isinstance(section_data, dict):
            for alias in aliases:
                section_data.pop(alias, None)
    return PipelineConfig.model_validate(resolved)


def _preserve_unknown_values(
    source_data: Mapping[str, Any],
    compact: dict[str, Any],
) -> tuple[str, ...]:
    """Copy unrecognized legacy values into the compact output without interpreting them."""
    unknown_keys: list[str] = []
    known_sections = PipelineConfig.model_fields

    for section, section_data in source_data.items():
        if section not in known_sections:
            compact[section] = deepcopy(section_data)
            unknown_keys.append(str(section))
            continue

        annotation = known_sections[section].annotation
        section_fields = getattr(annotation, "model_fields", None)
        if section_fields is None or not isinstance(section_data, Mapping):
            continue

        for key, value in section_data.items():
            if key in section_fields:
                continue
            target_section = compact.setdefault(section, {})
            target_section[key] = deepcopy(value)
            unknown_keys.append(f"{section}.{key}")

    return tuple(unknown_keys)


def compact_config_data(
    source_data: Mapping[str, Any],
    *,
    preserve_unknown: bool = True,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Validate and reduce config data to canonical values that differ from defaults.

    Dictionary-valued fields are retained as complete values when they differ from
    their defaults because several stage APIs intentionally treat those mappings as
    replacements rather than deep-merged overrides.
    """
    validated = load_config_data(source_data)
    canonical = _canonical_config(validated)
    compact = canonical.model_dump(mode="python", exclude_defaults=True)
    unknown_keys: tuple[str, ...] = ()
    if preserve_unknown:
        unknown_keys = _preserve_unknown_values(source_data, compact)
    return compact, unknown_keys


def default_compact_config_path(source_path: str | Path) -> Path:
    """Return the non-destructive default output path beside a source config."""
    source = Path(source_path).expanduser().resolve(strict=False)
    suffix = source.suffix or ".yaml"
    stem = source.stem if source.suffix else source.name
    return source.with_name(f"{stem}.compact{suffix}")


def write_compact_config(
    source_path: str | Path,
    output_path: str | Path | None = None,
    *,
    force: bool = False,
    preserve_unknown: bool = True,
) -> tuple[Path, tuple[str, ...]]:
    """Write a separate compact config while leaving the verbose source untouched."""
    source = Path(source_path).expanduser().resolve(strict=False)
    destination = (
        default_compact_config_path(source)
        if output_path is None
        else Path(output_path).expanduser().resolve(strict=False)
    )
    if source == destination:
        raise ValueError("Compact config output must be different from the source file.")
    if destination.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing compact config: {destination}"
        )

    source_data = read_config_mapping(source)
    compact, unknown_keys = compact_config_data(
        source_data,
        preserve_unknown=preserve_unknown,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            compact,
            handle,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return destination, unknown_keys


__all__ = [
    "compact_config_data",
    "default_compact_config_path",
    "write_compact_config",
]
