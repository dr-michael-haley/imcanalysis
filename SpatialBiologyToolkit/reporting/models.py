"""Typed records written for each human-facing stage execution."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ReportingModel(BaseModel):
    """Forward-compatible base model for historical reporting records."""

    model_config = ConfigDict(extra="allow")


class PathRecord(ReportingModel):
    role: str
    path: Path
    description: str = ""
    exists: bool | None = None


class GeneratedFile(ReportingModel):
    category: Literal["figure", "table", "summary", "file"]
    path: Path
    description: str = ""
    size_bytes: int | None = None


class ParameterRecord(ReportingModel):
    value: Any
    description: str = ""
    level: str = "advanced"
    section: str


class ErrorRecord(ReportingModel):
    type: str
    message: str
    traceback: str | None = None


class StageManifest(ReportingModel):
    schema_version: Literal[1, 2] = 2
    project_id: str
    execution_id: int | None = None
    execution_label: str = ""
    technical_run_id: str = ""
    workflow_run_id: str = ""
    output_folder: Path = Path(".")
    run_id: str | None = None
    stage: str
    display_name: str
    stage_display_name: str = ""
    status: Literal[
        "allocated",
        "pending",
        "running",
        "completed",
        "failed",
        "cancelled",
        "blocked",
        "unknown",
    ]
    managed_run: bool
    asset_effect: Literal["none", "created", "modified", "unknown"] = "unknown"

    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None

    pipeline_version: str | None = None
    git_commit: str | None = None
    slurm_job_id: str | None = None
    technical_run_record: Path | None = None

    reason: str | None = None
    notes: list[str] = Field(default_factory=list)
    documentation_source: Path | None = None
    explainer_snapshot: str = ""

    inputs: list[PathRecord] = Field(default_factory=list)
    produced_assets: list[PathRecord] = Field(default_factory=list)
    generated_files: list[GeneratedFile] = Field(default_factory=list)
    parameters: dict[str, ParameterRecord] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[ErrorRecord] = Field(default_factory=list)
    rendering_errors: list[str] = Field(default_factory=list)


__all__ = [
    "ErrorRecord",
    "GeneratedFile",
    "ParameterRecord",
    "PathRecord",
    "ReportingModel",
    "StageManifest",
]
