"""Extract and render documentation from Pydantic config field metadata."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Type

from .models import ConfigModel, DEFAULT_CONFIG_CLASSES


@dataclass(frozen=True)
class ConfigFieldDoc:
    """Documentation-ready representation of one config model field."""

    name: str
    annotation: str
    default: Any
    description: str
    level: str
    stage: str
    ui_group: str
    advice: str


def _annotation_to_string(annotation: Any) -> str:
    """Return a concise, stable display string for a field annotation."""
    name = getattr(annotation, "__name__", None)
    if name:
        return str(name)
    return str(annotation).replace("typing.", "")


def iter_config_docs(
    model_class: Type[ConfigModel],
) -> Iterator[ConfigFieldDoc]:
    """Yield documentation records in the model's declared field order."""
    for field_name, model_field in model_class.model_fields.items():
        metadata = dict(model_field.json_schema_extra or {})
        default = (
            "<required>"
            if model_field.is_required()
            else model_field.get_default(call_default_factory=True)
        )
        yield ConfigFieldDoc(
            name=field_name,
            annotation=_annotation_to_string(model_field.annotation),
            default=default,
            description=model_field.description or "",
            level=str(metadata.get("level", "advanced")),
            stage=str(metadata.get("stage", "")),
            ui_group=str(metadata.get("ui_group", "Configuration")),
            advice=str(metadata.get("advice", "")),
        )


def _format_default(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return repr(value)


def generate_markdown_for_model(
    model_class: Type[ConfigModel],
    *,
    title: Optional[str] = None,
) -> str:
    """Generate simple Markdown documentation grouped by ``ui_group``."""
    records = list(iter_config_docs(model_class))
    inferred_stage = records[0].stage if records else model_class.__name__
    document_title = title or inferred_stage.replace("_", " ").title()

    grouped: dict[str, list[ConfigFieldDoc]] = {}
    for record in records:
        grouped.setdefault(record.ui_group, []).append(record)

    lines = [f"# {document_title}", ""]
    for ui_group, group_records in grouped.items():
        lines.extend([f"## {ui_group}", ""])
        for record in group_records:
            lines.extend(
                [
                    f"### `{record.name}`",
                    "",
                    f"- Type: `{record.annotation}`",
                    f"- Default: `{_format_default(record.default)}`",
                    f"- Level: `{record.level}`",
                    "",
                    record.description or "No description available.",
                    "",
                    "Advice:",
                    record.advice or "No additional advice.",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def write_config_docs(
    output_dir: str | Path,
    models: Optional[Mapping[str, Type[ConfigModel]]] = None,
) -> list[Path]:
    """Write one generated Markdown file per config section."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    section_models = DEFAULT_CONFIG_CLASSES if models is None else models

    written: list[Path] = []
    for section, model_class in section_models.items():
        output_path = destination / f"{section}.md"
        output_path.write_text(
            generate_markdown_for_model(
                model_class,
                title=section.replace("_", " ").title(),
            ),
            encoding="utf-8",
        )
        written.append(output_path)
    return written


__all__ = [
    "ConfigFieldDoc",
    "generate_markdown_for_model",
    "iter_config_docs",
    "write_config_docs",
]
