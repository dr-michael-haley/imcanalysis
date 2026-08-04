"""Typed, backend-neutral models used by the lightweight control layer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DependencyPolicy = Literal["assets", "none", "all"]


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


class RegisteredProject(PipelineModel):
    name: str = Field(min_length=1)
    path: Path
    project_id: str


class ProjectRegistry(PipelineModel):
    schema_version: Literal[1] = 1
    projects: list[RegisteredProject] = Field(default_factory=list)
    default_project_id: str | None = None

    @model_validator(mode="after")
    def validate_projects(self) -> "ProjectRegistry":
        paths = [str(item.path).casefold() for item in self.projects]
        names = [item.name.casefold() for item in self.projects]
        project_ids = [item.project_id for item in self.projects]
        if len(paths) != len(set(paths)):
            raise ValueError("Registered project paths must be unique.")
        if len(names) != len(set(names)):
            raise ValueError("Registered project names must be unique.")
        if len(project_ids) != len(set(project_ids)):
            raise ValueError("Registered project IDs must be unique.")
        if self.default_project_id and self.default_project_id not in project_ids:
            raise ValueError("The default project must be present in the registry.")
        return self


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
    catalogue_order: int
    output_slug: str
    documentation_path: str
    description: str
    slurm_script: str
    environment_keys: list[str] = Field(default_factory=list)
    config_sections: list[str] = Field(default_factory=list)
    python_modules: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    requires_assets: list[str] = Field(default_factory=list)
    advisory_assets: list[str] = Field(default_factory=list)
    required_executions: dict[str, list[str]] = Field(default_factory=dict)
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
    advisory_assets: list[str] = Field(default_factory=list)
    produces_assets: list[str]
    expected_outputs: list[str]
    script_exists: bool
    missing_assets: list[str] = Field(default_factory=list)
    missing_advisory_assets: list[str] = Field(default_factory=list)
    required_executions: dict[str, list[str]] = Field(default_factory=dict)
    missing_executions: list[str] = Field(default_factory=list)
    missing_files: list[Path] = Field(default_factory=list)
    skipped_upstream_stages: list[str] = Field(default_factory=list)


class RunPlan(PipelineModel):
    schema_version: Literal[1] = 1
    project_id: str
    project_root: Path
    requested: list[str]
    resolved_stages: list[PlannedStage]
    config_source: Path
    execution_backend: str = "slurm_scripts"
    dependency_policy: DependencyPolicy = "assets"
    ready: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


ExecutionStatus = Literal[
    "allocated",
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
    "blocked",
    "unknown",
]
AssetEffect = Literal["none", "created", "modified", "unknown"]


class ExecutionRecord(PipelineModel):
    execution_id: int = Field(ge=1)
    execution_label: str
    original_execution_id: int = Field(ge=1)
    technical_run_id: str
    workflow_run_id: str
    stage: str
    stage_display_name: str
    output_slug: str
    output_folder: Path
    status: ExecutionStatus = "allocated"
    asset_effect: AssetEffect = "unknown"
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    slurm_job_id: str | None = None


class ExecutionIndex(PipelineModel):
    schema_version: Literal[1] = 1
    project_id: str
    updated_at: datetime
    executions: list[ExecutionRecord] = Field(default_factory=list)


class ExecutionReference(PipelineModel):
    execution_id: int = Field(ge=1)
    execution_label: str
    original_execution_id: int = Field(ge=1)
    technical_run_id: str
    stage: str
    output_folder: Path


class RenumberRecord(PipelineModel):
    technical_run_id: str
    stage: str
    previous_execution_id: int
    new_execution_id: int
    previous_output_folder: Path
    new_output_folder: Path


class AssetCleanupItem(PipelineModel):
    role: str
    path: Path
    reason: str
    dependent_stages: list[str] = Field(default_factory=list)


class AssetCleanupPlan(PipelineModel):
    execution_id: int
    technical_run_id: str
    stage: str
    removable: list[AssetCleanupItem] = Field(default_factory=list)
    retained: list[AssetCleanupItem] = Field(default_factory=list)


class AssetCleanupAudit(PipelineModel):
    offered: bool = False
    confirmed: bool = False
    removable: list[AssetCleanupItem] = Field(default_factory=list)
    retained: list[AssetCleanupItem] = Field(default_factory=list)
    cleaned_paths: list[Path] = Field(default_factory=list)
    removed_entries: int = 0
    errors: list[str] = Field(default_factory=list)


class RemovalAudit(PipelineModel):
    schema_version: Literal[1] = 1
    audit_id: str
    removed_at: datetime
    removed_by: str
    action: Literal["user_removal", "submission_exclusion"] = "user_removal"
    previous_execution: ExecutionRecord
    previous_execution_id: int
    technical_run_id: str
    stage: str
    previous_output_folder: Path
    asset_effect: AssetEffect
    confirmation_mode: Literal["interactive", "non_interactive", "system"]
    reason: str | None = None
    renumbered: list[RenumberRecord] = Field(default_factory=list)
    asset_cleanup: AssetCleanupAudit | None = None


class MigrationRecord(PipelineModel):
    source_folder: Path
    target_folder: Path
    execution: ExecutionRecord
    manifest_path: Path


class ExecutionMigrationPlan(PipelineModel):
    schema_version: Literal[1] = 1
    project_id: str
    created_at: datetime
    legacy_layout_detected: bool
    safe_to_apply: bool
    records: list[MigrationRecord] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)


class MigrationAudit(PipelineModel):
    schema_version: Literal[1] = 1
    migrated_at: datetime
    records: list[MigrationRecord] = Field(default_factory=list)


class ExecutionSummary(PipelineModel):
    schema_version: Literal[1] = 1
    execution_id: int
    execution_label: str
    stage: str
    stage_display_name: str
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    asset_effect: AssetEffect
    output_folder: Path
    technical_run_id: str
    workflow_run_id: str
    slurm_job_id: str | None = None
    removed: bool = False
    removed_at: datetime | None = None


class RunManifest(PipelineModel):
    schema_version: Literal[1, 2] = 2
    run_id: str
    workflow_run_id: str | None = None
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
    executions: list[ExecutionReference] = Field(default_factory=list)
    plan_token_digest: str | None = None
    provenance_digest: str | None = None
    provenance_file: Path | None = None


class SubmissionRecord(PipelineModel):
    stage: str
    execution_id: int | None = None
    technical_run_id: str | None = None
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
    schema_version: Literal[1, 2] = 2
    run_id: str
    workflow_run_id: str | None = None
    execution_backend: str = "slurm_scripts"
    jobs: list[SubmissionRecord] = Field(default_factory=list)
    submission_complete: bool = False


class StageStatus(PipelineModel):
    stage: str
    execution_id: int | None = None
    technical_run_id: str | None = None
    job_id: str | None
    status: Literal[
        "not_submitted",
        "pending",
        "running",
        "completed",
        "failed",
        "cancelled",
        "blocked",
        "unknown",
    ]
    source: str
    detail: str | None = None


class RunStatus(PipelineModel):
    schema_version: Literal[1, 2] = 2
    run_id: str
    workflow_run_id: str | None = None
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
    "DependencyPolicy",
    "AssetEffect",
    "AssetInventory",
    "ExecutionIndex",
    "ExecutionMigrationPlan",
    "ExecutionRecord",
    "ExecutionReference",
    "ExecutionStatus",
    "ExecutionSummary",
    "LogRecord",
    "MigrationAudit",
    "MigrationRecord",
    "ModeSpec",
    "PipelineModel",
    "PlannedStage",
    "ProjectAsset",
    "ProjectMetadata",
    "ProjectRegistry",
    "ProjectValidationReport",
    "RemovalAudit",
    "RegisteredProject",
    "RenumberRecord",
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
