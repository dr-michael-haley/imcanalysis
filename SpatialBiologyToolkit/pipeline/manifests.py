"""Centralized, atomic serialization for project and run records."""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from .models import model_data


ModelT = TypeVar("ModelT", bound=BaseModel)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
        for attempt in range(5):
            try:
                os.replace(temporary_name, path)
                break
            except PermissionError:
                if os.name != "nt" or attempt == 4:
                    raise
                time.sleep(0.02 * (attempt + 1))
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def dump_yaml(value: Any) -> str:
    return yaml.safe_dump(
        _json_safe(value),
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


def dump_json(value: Any) -> str:
    return json.dumps(_json_safe(value), indent=2, ensure_ascii=False) + "\n"


def write_yaml(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    _atomic_write(destination, dump_yaml(value))
    return destination


def write_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    _atomic_write(destination, dump_json(value))
    return destination


def write_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    _atomic_write(destination, text)
    return destination


def read_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {source}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {source}.")
    return loaded


def read_model(path: str | Path, model: type[ModelT]) -> ModelT:
    return model.model_validate(read_yaml(path))


def format_machine_output(value: BaseModel | dict[str, Any], output_format: str) -> str:
    data = model_data(value)
    if output_format == "json":
        return dump_json(data)
    if output_format == "yaml":
        return dump_yaml(data)
    raise ValueError(f"Unsupported output format: {output_format}")


__all__ = [
    "dump_json",
    "dump_yaml",
    "format_machine_output",
    "read_model",
    "read_yaml",
    "utc_now",
    "write_json",
    "write_text",
    "write_yaml",
]
