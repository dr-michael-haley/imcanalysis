"""Inventory generated files, configured assets, and reportable parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]

from SpatialBiologyToolkit.config import load_config
from SpatialBiologyToolkit.pipeline.assets import asset_map, resolve_assets
from SpatialBiologyToolkit.pipeline.registry import get_stage

from .models import GeneratedFile, ParameterRecord, PathRecord
from .paths import ReportingContext


FIGURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".tif", ".tiff"}
TABLE_SUFFIXES = {".csv", ".tsv", ".xlsx", ".parquet"}
SUMMARY_SUFFIXES = {".md", ".txt", ".html"}
REPORT_FILES = {"README.md", "stage_manifest.yaml"}


def classify_file(
    path: Path,
) -> Literal["figure", "table", "summary", "file"]:
    suffix = path.suffix.lower()
    if suffix in FIGURE_SUFFIXES:
        return "figure"
    if suffix in TABLE_SUFFIXES:
        return "table"
    if suffix in SUMMARY_SUFFIXES:
        return "summary"
    return "file"


def discover_generated_files(run_dir: Path) -> list[GeneratedFile]:
    records: list[GeneratedFile] = []
    if not run_dir.is_dir():
        return records
    for path in sorted(run_dir.rglob("*"), key=lambda item: str(item).lower()):
        relative = path.relative_to(run_dir)
        if (
            not path.is_file()
            or path.name in REPORT_FILES
            or path.name.startswith(".")
            or (relative.parts and relative.parts[0] == "environment")
        ):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        records.append(
            GeneratedFile(
                category=classify_file(path),
                path=path.resolve(strict=False),
                size_bytes=size,
            )
        )
    return records


def _raw_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def extract_stage_parameters(context: ReportingContext) -> dict[str, ParameterRecord]:
    if context.config_path is None or not context.config_path.is_file():
        return {}
    try:
        config = load_config(context.config_path)
    except (OSError, ValueError):
        return {}
    user_config = (
        context.technical_run_record / "config.user.yaml"
        if context.technical_run_record
        else context.config_path
    )
    raw = _raw_config(
        user_config if user_config and user_config.is_file() else context.config_path
    )
    parameters: dict[str, ParameterRecord] = {}
    spec = get_stage(context.stage)

    for section_name in spec.config_sections:
        section = getattr(config, section_name, None)
        if section is None:
            continue
        explicit = raw.get(section_name, {})
        explicit_keys = set(explicit) if isinstance(explicit, dict) else set()
        for field_name, model_field in section.__class__.model_fields.items():
            metadata = model_field.json_schema_extra or {}
            level = str(metadata.get("level", "advanced"))
            if field_name not in explicit_keys and level != "basic":
                continue
            parameters[f"{section_name}.{field_name}"] = ParameterRecord(
                value=getattr(section, field_name),
                description=model_field.description or "",
                level=level,
                section=section_name,
            )
    return parameters


ASSET_DESCRIPTIONS = {
    "raw_imc_files": "Raw IMC source files.",
    "metadata": "Project metadata and panel tables.",
    "tiff_stacks": "Reusable multiplex TIFF stacks.",
    "raw_images": "Reusable unstacked channel images.",
    "denoised_images": "Reusable denoised channel images.",
    "masks": "Reusable segmentation masks.",
    "cell_tables": "Reusable quantified cell tables.",
    "anndata": "Canonical reusable AnnData object.",
    "human_outputs": "Human-facing sequential execution outputs.",
    "legacy_qc": "Deprecated legacy QC output folder.",
    "legacy_slurm_logs": "Deprecated legacy SLURM log folder.",
}


def configured_stage_paths(
    context: ReportingContext,
) -> tuple[list[PathRecord], list[PathRecord]]:
    if context.config_path is None or not context.config_path.is_file():
        return [], []
    try:
        config = load_config(context.config_path)
    except (OSError, ValueError):
        return [], []
    assets = asset_map(resolve_assets(config, context.project_root))
    spec = get_stage(context.stage)

    def records(roles: list[str]) -> list[PathRecord]:
        output: list[PathRecord] = []
        for role in roles:
            if role == "human_outputs":
                path = context.stage_run_dir
                exists = path.is_dir()
            else:
                asset = assets.get(role)
                if asset is None:
                    continue
                path = asset.path
                exists = asset.exists
            output.append(
                PathRecord(
                    role=role,
                    path=path,
                    description=ASSET_DESCRIPTIONS.get(role, ""),
                    exists=exists,
                )
            )
        return output

    return records(spec.requires_assets), records(spec.produces_assets)


__all__ = [
    "classify_file",
    "configured_stage_paths",
    "discover_generated_files",
    "extract_stage_parameters",
]
