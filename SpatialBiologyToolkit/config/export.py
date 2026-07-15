"""Resolved configuration serialization helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from .load import config_to_dict
from .models import PipelineConfig


def write_resolved_config(
    config: PipelineConfig,
    output_path: str | Path,
) -> None:
    """Write all user values and filled defaults to a YAML provenance file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config_to_dict(config),
            handle,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


__all__ = ["write_resolved_config"]
