"""Lightweight, stage-aware scientific reporting and provenance."""

from .paths import (
    ReportingContext,
    category_output_path,
    infer_stage_from_main_module,
    optional_category_output_path,
    project_asset_path,
    resolve_reporting_context,
)
from .reporter import (
    StageReporter,
    bootstrap_stage_reporting,
    ensure_stage_reporter,
    get_active_reporter,
)

__all__ = [
    "ReportingContext",
    "StageReporter",
    "bootstrap_stage_reporting",
    "category_output_path",
    "ensure_stage_reporter",
    "get_active_reporter",
    "infer_stage_from_main_module",
    "optional_category_output_path",
    "project_asset_path",
    "resolve_reporting_context",
]
