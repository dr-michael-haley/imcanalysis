"""Typed, backend-neutral models used by the lightweight control layer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PipelineModel(BaseModel):
    """Strict base model for versioned operational records."""

    model_config = ConfigDict(extra="forbid")


class ProjectMetadata(PipelineModel):
    schema_version: Literal[1] = 1
    project_id: str
    created_at: datetime
    config_file: str
    title: str | None = None
    description: str | None = None
    notes_file: str = ".sbt/project_notes.md"
    toolkit: str = "Spatial Biology Toolkit"


class ProjectAsset(PipelineModel):
    role: str
    path: Path
    kind: Literal["file", "directory"]
    lifecycle: Literal[
        "required_input",
        "optional_input",
        "generated_output",
        "human_output",
        "legacy_output",
        "operational_state",
    ]
    exists: bool
    size_bytes: int | None = None
    modified_at: datetime | None = None
    file_count: int | None = None
    count_limited: bool = False


class AssetInventory(PipelineModel):
    schema_version: Literal[1] = 1
    captured_at: datetime
    project_id: str
    project_root: Path
    assets: list[ProjectAsset]


class ValidationItem(PipelineModel):
    name: str
    path: Path | None = None
    status: Literal["ok", "warning", "missing", "not_created"]
    message: str


class ProjectValidationReport(PipelineModel):
    schema_version: Literal[1] = 1
    project_id: str
    project_root: Path
    valid: bool
    required_inputs: list[ValidationItem] = Field(default_factory=list)
    optional_inputs: list[ValidationItem] = Field(default_factory=list)
    generated_assets: list[ValidationItem] = Field(default_factory=list)
    reporting_outputs: list[ValidationItem] = Field(default_factory=list)
    stage_readiness: dict[str, bool] = Field(default_factory=dict)
    readiness_messages: dict[str, list[str]] = Field(default_factory=dict)


class StageSpec(PipelineModel):
    name: str
    display_name: str
    display_order: int
    output_folder: str
    documentation_path: str
    description: str
    slurm_script: str
    config_sections: list[str] = Field(default_factory=list)
    python_modules: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    requires_assets: list[str] = Field(default_factory=list)
    produces_assets: list[str] = Field(default_factory=list)
    required_files: dict[str, list[str]] = Field(default_factory=dict)
    expected_outputs: list[str] = Field(default_factory=list)
    log_patterns: list[str] = Field(default_factory=list)
    runnable: bool = True
    notes: list[str] = Field(default_factory=list)


class ModeSpec(PipelineModel):
    name: str
    description: str
    stages: list[str]


class PlannedStage(PipelineModel):
    name: str
    description: str
    slurm_script: Path
    depends_on: list[str]
    requires_assets: list[str]
    produces_assets: list[str]
    expected_outputs: list[str]
    script_exists: bool
    missing_assets: list[str] = Field(default_factory=list)
    missing_files: list[Path] = Field(default_factory=list)


class RunPlan(PipelineModel):
    schema_version: Literal[1] = 1
    project_id: str
    project_root: Path
    requested: list[str]
    resolved_stages: list[PlannedStage]
    config_source: Path
    execution_backend: str = "slurm_scripts"
    ready: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RunManifest(PipelineModel):
    schema_version: Literal[1] = 1
    run_id: str
    project_id: str
    project_root: Path
    created_at: datetime
    requested_stages: list[str]
    resolved_stages: list[str]
    config_source: Path
    resolved_config: Path
    execution_backend: str
    working_directory: Path
    command: str
    reason: str | None = None
    notes: list[str] = Field(default_factory=list)
    pipeline_version: str | None = None
    git_commit: str | None = None
    hostname: str | None = None
    username: str | None = None


class SubmissionRecord(PipelineModel):
    stage: str
    state: Literal["submitted", "submission_failed"]
    job_id: str | None = None
    dependency_job_id: str | None = None
    submitted_at: datetime
    command: list[str]
    exported_environment: dict[str, str]
    stdout_log: Path
    stderr_log: Path
    error: str | None = None


class SubmittedJobs(PipelineModel):
    schema_version: Literal[1] = 1
    run_id: str
    execution_backend: str = "slurm_scripts"
    jobs: list[SubmissionRecord] = Field(default_factory=list)
    submission_complete: bool = False


class StageStatus(PipelineModel):
    stage: str
    job_id: str | None
    status: Literal[
        "not_submitted",
        "pending",
        "running",
        "completed",
        "failed",
        "cancelled",
        "unknown",
    ]
    source: str
    detail: str | None = None


class RunStatus(PipelineModel):
    schema_version: Literal[1] = 1
    run_id: str
    project_id: str
    checked_at: datetime
    overall_status: str
    stages: list[StageStatus]
    warnings: list[str] = Field(default_factory=list)


class LogRecord(PipelineModel):
    stage: str
    job_id: str | None
    stream: Literal["stdout", "stderr"]
    path: Path
    exists: bool


def model_data(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    """Return JSON-safe data for CLI serialization."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


__all__ = [
    "AssetInventory",
    "LogRecord",
    "ModeSpec",
    "PipelineModel",
    "PlannedStage",
    "ProjectAsset",
    "ProjectMetadata",
    "ProjectValidationReport",
    "RunManifest",
    "RunPlan",
    "RunStatus",
    "StageSpec",
    "StageStatus",
    "SubmissionRecord",
    "SubmittedJobs",
    "ValidationItem",
    "model_data",
]
