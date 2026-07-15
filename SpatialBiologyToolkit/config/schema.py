"""JSON Schema generation for pipeline configuration tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import PipelineConfig


def generate_json_schema() -> dict[str, Any]:
    """Return the Pydantic validation schema for the complete config."""
    return PipelineConfig.model_json_schema(mode="validation")


def write_json_schema(output_path: str | Path) -> None:
    """Write the complete config schema as formatted JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(generate_json_schema(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


__all__ = ["generate_json_schema", "write_json_schema"]
