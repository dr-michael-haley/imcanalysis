"""Safe, schema-backed editing sessions for project configuration YAML."""

from __future__ import annotations

import copy
import difflib
import getpass
import hashlib
import io
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from pydantic import BaseModel, ValidationError

from SpatialBiologyToolkit.pipeline.manifests import utc_now, write_text, write_yaml

from .load import load_config_data
from .models import DEFAULT_CONFIG_CLASSES, PipelineConfig
from .schema import generate_json_schema


CONFIG_BACKUP_DIRECTORY = Path(".sbt/config-backups")
CONFIG_AUDIT_DIRECTORY = Path(".sbt/audit/config-edits")


class ConfigEditError(RuntimeError):
    """Base error for a safe configuration edit."""


class ConfigChangedExternallyError(ConfigEditError):
    """Raised when the source changed after an editing session was opened."""


class InvalidConfigEditError(ConfigEditError):
    """Raised when proposed configuration cannot be validated."""


@dataclass(frozen=True)
class ConfigFieldSpec:
    """UI-neutral description and current state for one config field."""

    section: str
    name: str
    path: str
    annotation: str
    kind: str
    default: Any
    value: Any
    explicit: bool
    required: bool
    nullable: bool
    description: str
    advice: str
    level: str
    ui_group: str
    enum_values: tuple[Any, ...] = ()
    minimum: float | int | None = None
    maximum: float | int | None = None


@dataclass(frozen=True)
class ConfigSaveResult:
    """Paths and hashes recorded by one successful save."""

    config_path: Path
    backup_path: Path
    audit_path: Path
    before_hash: str
    after_hash: str
    changed_paths: tuple[str, ...]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mapping_from_text(text: str, path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise InvalidConfigEditError(f"Invalid YAML in {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise InvalidConfigEditError(
            f"Configuration file {path} must contain a YAML mapping at its root."
        )
    return dict(loaded)


def _path_parts(path: str) -> tuple[str, ...]:
    parts = tuple(part for part in path.split(".") if part)
    if len(parts) != 2:
        raise ValueError("Config field paths must be '<section>.<field>'.")
    return parts


def _set_path(mapping: dict[str, Any], path: str, value: Any) -> None:
    section, field = _path_parts(path)
    section_data = mapping.get(section)
    if not isinstance(section_data, dict):
        section_data = {}
        mapping[section] = section_data
    section_data[field] = copy.deepcopy(value)


def _remove_path(mapping: dict[str, Any], path: str) -> None:
    section, field = _path_parts(path)
    section_data = mapping.get(section)
    if not isinstance(section_data, dict):
        return
    section_data.pop(field, None)
    if not section_data:
        mapping.pop(section, None)


def _schema_kind(field_schema: Mapping[str, Any]) -> str:
    if "enum" in field_schema:
        return "enum"
    if "anyOf" in field_schema:
        non_null = [
            item for item in field_schema["anyOf"] if item.get("type") != "null"
        ]
        if len(non_null) == 1:
            return _schema_kind(non_null[0])
        return "yaml"
    field_type = field_schema.get("type")
    if field_type in {"boolean", "integer", "number", "string"}:
        return str(field_type)
    if field_type in {"array", "object"} or "$ref" in field_schema:
        return "yaml"
    return "yaml"


def _schema_value(field_schema: Mapping[str, Any], key: str) -> Any:
    if key in field_schema:
        return field_schema[key]
    for item in field_schema.get("anyOf", []):
        if key in item:
            return item[key]
    return None


def _plain_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, dict):
        return {key: _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return copy.deepcopy(value)


def _render_yaml_with_round_trip(
    source_text: str,
    working: Mapping[str, Any],
    changed_paths: Iterable[str],
    removed_paths: Iterable[str],
) -> str:
    """Render changes with ruamel when available, falling back to PyYAML."""

    try:
        from ruamel.yaml import YAML  # type: ignore[import-not-found]
    except ImportError:
        return yaml.safe_dump(
            dict(working),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    round_trip = YAML()
    round_trip.preserve_quotes = True
    document = round_trip.load(source_text)
    if document is None:
        document = {}
    for path in removed_paths:
        section, field = _path_parts(path)
        section_data = document.get(section)
        if isinstance(section_data, Mapping):
            section_data.pop(field, None)
            if not section_data:
                document.pop(section, None)
    for path in changed_paths:
        section, field = _path_parts(path)
        if section not in document or not isinstance(document[section], Mapping):
            document[section] = {}
        document[section][field] = copy.deepcopy(working[section][field])
    stream = io.StringIO()
    round_trip.dump(document, stream)
    return stream.getvalue()


class ConfigEditorSession:
    """Mutable proposal over an immutable source config snapshot."""

    def __init__(self, path: Path, source_text: str, source_data: dict[str, Any]):
        self.path = path.expanduser().resolve(strict=False)
        self.source_text = source_text
        self.source_hash = _content_hash(source_text)
        self.source_data = copy.deepcopy(source_data)
        self.working_data = copy.deepcopy(source_data)
        self._changed_paths: set[str] = set()
        self._removed_paths: set[str] = set()
        self._validated = load_config_data(self.working_data)

    @classmethod
    def open(cls, path: str | Path) -> "ConfigEditorSession":
        source = Path(path).expanduser().resolve(strict=False)
        if not source.is_file():
            raise FileNotFoundError(f"Configuration file not found: {source}")
        text = source.read_text(encoding="utf-8")
        data = _mapping_from_text(text, source)
        try:
            load_config_data(data)
        except ValidationError as exc:
            raise InvalidConfigEditError(str(exc)) from exc
        return cls(source, text, data)

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self._changed_paths | self._removed_paths))

    @property
    def dirty(self) -> bool:
        return bool(self._changed_paths or self._removed_paths)

    @property
    def validated_config(self) -> PipelineConfig:
        return self._validated

    def set_value(self, path: str, value: Any) -> PipelineConfig:
        candidate = copy.deepcopy(self.working_data)
        _set_path(candidate, path, value)
        try:
            validated = load_config_data(candidate)
        except ValidationError as exc:
            raise InvalidConfigEditError(str(exc)) from exc
        self.working_data = candidate
        self._validated = validated
        self._removed_paths.discard(path)
        self._changed_paths.add(path)
        if self._value_matches_source(path):
            self._changed_paths.discard(path)
        return validated

    def reset_to_default(self, path: str) -> PipelineConfig:
        candidate = copy.deepcopy(self.working_data)
        _remove_path(candidate, path)
        try:
            validated = load_config_data(candidate)
        except ValidationError as exc:
            raise InvalidConfigEditError(str(exc)) from exc
        self.working_data = candidate
        self._validated = validated
        self._changed_paths.discard(path)
        if self._source_has_path(path):
            self._removed_paths.add(path)
        else:
            self._removed_paths.discard(path)
        return validated

    def discard(self) -> None:
        self.working_data = copy.deepcopy(self.source_data)
        self._validated = load_config_data(self.working_data)
        self._changed_paths.clear()
        self._removed_paths.clear()

    def field_specs(self) -> list[ConfigFieldSpec]:
        schema = generate_json_schema()
        specs: list[ConfigFieldSpec] = []
        for section, model_class in DEFAULT_CONFIG_CLASSES.items():
            section_schema = schema["$defs"][model_class.__name__]
            raw_section = self.working_data.get(section)
            raw_section = raw_section if isinstance(raw_section, Mapping) else {}
            resolved_section = getattr(self._validated, section)
            resolved_values = resolved_section.model_dump(mode="python")
            for name, model_field in model_class.model_fields.items():
                metadata = dict(model_field.json_schema_extra or {})
                field_schema = section_schema["properties"][name]
                enum_values = _schema_value(field_schema, "enum") or []
                annotation = str(model_field.annotation).replace("typing.", "")
                default = _plain_value(
                    None
                    if model_field.is_required()
                    else model_field.get_default(call_default_factory=True)
                )
                specs.append(
                    ConfigFieldSpec(
                        section=section,
                        name=name,
                        path=f"{section}.{name}",
                        annotation=annotation,
                        kind=_schema_kind(field_schema),
                        default=default,
                        value=_plain_value(resolved_values[name]),
                        explicit=name in raw_section,
                        required=model_field.is_required(),
                        nullable=any(
                            item.get("type") == "null"
                            for item in field_schema.get("anyOf", [])
                        ),
                        description=model_field.description or "",
                        advice=str(metadata.get("advice", "")),
                        level=str(metadata.get("level", "advanced")),
                        ui_group=str(metadata.get("ui_group", section.title())),
                        enum_values=tuple(enum_values),
                        minimum=_schema_value(field_schema, "minimum"),
                        maximum=_schema_value(field_schema, "maximum"),
                    )
                )
        return specs

    def diff(self) -> str:
        before = yaml.safe_dump(
            self.source_data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        ).splitlines()
        after = yaml.safe_dump(
            self.working_data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        ).splitlines()
        return "\n".join(
            difflib.unified_diff(
                before,
                after,
                fromfile=str(self.path),
                tofile=f"{self.path} (proposed)",
                lineterm="",
            )
        )

    def render(self) -> str:
        rendered = _render_yaml_with_round_trip(
            self.source_text,
            self.working_data,
            self._changed_paths,
            self._removed_paths,
        )
        proposed = _mapping_from_text(rendered, self.path)
        try:
            load_config_data(proposed)
        except ValidationError as exc:
            raise InvalidConfigEditError(str(exc)) from exc
        return rendered

    def save(self, project_root: str | Path) -> ConfigSaveResult:
        if not self.dirty:
            raise ConfigEditError("No configuration changes are staged.")
        current_text = self.path.read_text(encoding="utf-8")
        if _content_hash(current_text) != self.source_hash:
            raise ConfigChangedExternallyError(
                "The configuration changed after it was opened. Reload before saving."
            )
        rendered = self.render()
        before_hash = self.source_hash
        after_hash = _content_hash(rendered)
        root = Path(project_root).expanduser().resolve(strict=False)
        timestamp = utc_now()
        token = f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        backup_path = root / CONFIG_BACKUP_DIRECTORY / f"{token}.yaml"
        audit_path = root / CONFIG_AUDIT_DIRECTORY / f"{token}.yaml"
        write_text(backup_path, current_text)
        write_text(self.path, rendered)
        write_yaml(
            audit_path,
            {
                "schema_version": 1,
                "edited_at": timestamp,
                "edited_by": getpass.getuser(),
                "config_path": self.path,
                "backup_path": backup_path,
                "before_hash": before_hash,
                "after_hash": after_hash,
                "changed_paths": list(self.changed_paths),
            },
        )
        result = ConfigSaveResult(
            config_path=self.path,
            backup_path=backup_path,
            audit_path=audit_path,
            before_hash=before_hash,
            after_hash=after_hash,
            changed_paths=self.changed_paths,
        )
        refreshed = ConfigEditorSession.open(self.path)
        self.source_text = refreshed.source_text
        self.source_hash = refreshed.source_hash
        self.source_data = refreshed.source_data
        self.working_data = refreshed.working_data
        self._validated = refreshed.validated_config
        self._changed_paths.clear()
        self._removed_paths.clear()
        return result

    def _source_has_path(self, path: str) -> bool:
        section, field = _path_parts(path)
        data = self.source_data.get(section)
        return isinstance(data, Mapping) and field in data

    def _value_matches_source(self, path: str) -> bool:
        section, field = _path_parts(path)
        source_section = self.source_data.get(section)
        working_section = self.working_data.get(section)
        if not isinstance(source_section, Mapping) or field not in source_section:
            return False
        return isinstance(working_section, Mapping) and working_section.get(
            field
        ) == source_section.get(field)


class ConfigRecoverySession:
    """Raw-text repair session for syntactically or semantically invalid YAML."""

    def __init__(self, path: Path, text: str):
        self.path = path.expanduser().resolve(strict=False)
        self.source_text = text
        self.source_hash = _content_hash(text)

    @classmethod
    def open(cls, path: str | Path) -> "ConfigRecoverySession":
        source = Path(path).expanduser().resolve(strict=False)
        return cls(source, source.read_text(encoding="utf-8"))

    def validate_text(self, text: str) -> PipelineConfig:
        data = _mapping_from_text(text, self.path)
        try:
            return load_config_data(data)
        except ValidationError as exc:
            raise InvalidConfigEditError(str(exc)) from exc

    def save_text(self, text: str, project_root: str | Path) -> ConfigSaveResult:
        self.validate_text(text)
        current_text = self.path.read_text(encoding="utf-8")
        if _content_hash(current_text) != self.source_hash:
            raise ConfigChangedExternallyError(
                "The configuration changed after recovery mode opened it. Reload first."
            )
        root = Path(project_root).expanduser().resolve(strict=False)
        timestamp = utc_now()
        token = f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        backup_path = root / CONFIG_BACKUP_DIRECTORY / f"{token}.yaml"
        audit_path = root / CONFIG_AUDIT_DIRECTORY / f"{token}.yaml"
        write_text(backup_path, current_text)
        write_text(self.path, text)
        before_hash = self.source_hash
        after_hash = _content_hash(text)
        write_yaml(
            audit_path,
            {
                "schema_version": 1,
                "edited_at": timestamp,
                "edited_by": getpass.getuser(),
                "config_path": self.path,
                "backup_path": backup_path,
                "before_hash": before_hash,
                "after_hash": after_hash,
                "changed_paths": ["<raw-recovery>"],
            },
        )
        self.source_text = text
        self.source_hash = after_hash
        return ConfigSaveResult(
            config_path=self.path,
            backup_path=backup_path,
            audit_path=audit_path,
            before_hash=before_hash,
            after_hash=after_hash,
            changed_paths=("<raw-recovery>",),
        )


__all__ = [
    "CONFIG_AUDIT_DIRECTORY",
    "CONFIG_BACKUP_DIRECTORY",
    "ConfigChangedExternallyError",
    "ConfigEditError",
    "ConfigEditorSession",
    "ConfigFieldSpec",
    "ConfigRecoverySession",
    "ConfigSaveResult",
    "InvalidConfigEditError",
]
