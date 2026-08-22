"""Typed environment registry, observation, drift, and provenance records."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class EnvironmentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnvironmentDefinition(EnvironmentModel):
    conda_name: str
    specification_directory: Path | None = None
    platform: str = "linux-64"
    conda_channel_priority: Literal["strict", "flexible", "disabled"] | None = None
    toolkit_overlay: Literal["editable-no-deps", "none"] = "editable-no-deps"
    managed: bool = True
    smoke_tests: list[list[str]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _managed_has_specification(self) -> "EnvironmentDefinition":
        if self.managed and self.specification_directory is None:
            raise ValueError("managed environments require specification_directory")
        return self


class EnvironmentRegistry(EnvironmentModel):
    schema_version: Literal[1] = 1
    environments: dict[str, EnvironmentDefinition]
    stage_environments: dict[str, list[str]]

    @model_validator(mode="after")
    def _validate_identity(self) -> "EnvironmentRegistry":
        names: dict[str, str] = {}
        for key, definition in self.environments.items():
            normalized = definition.conda_name.casefold()
            if normalized in names:
                raise ValueError(
                    f"duplicate Conda name {definition.conda_name!r} for "
                    f"{names[normalized]!r} and {key!r}"
                )
            names[normalized] = key
        for stage, keys in self.stage_environments.items():
            if not keys:
                raise ValueError(f"stage {stage!r} has an empty environment mapping")
            unknown = [key for key in keys if key not in self.environments]
            if unknown:
                raise ValueError(
                    f"stage {stage!r} references unknown environment(s): {unknown}"
                )
        return self


class EnvironmentPaths(EnvironmentModel):
    registry: Path
    specification_directory: Path | None = None
    environment_yml: Path | None = None
    lockfile: Path | None = None
    pip_extras: Path | None = None
    observed_snapshot: Path | None = None


class CondaPackageRecord(EnvironmentModel):
    name: str
    version: str = ""
    build: str = ""
    channel: str = ""
    manager: str = "conda"


class PipPackageRecord(EnvironmentModel):
    name: str
    version: str = ""
    editable: bool = False
    location: str | None = None
    requirement: str | None = None
    source_type: Literal["index", "editable", "local", "vcs", "unknown"] = "index"


class ToolkitOverlayRecord(EnvironmentModel):
    installed: bool = False
    editable: bool = False
    repository_path: Path | None = None
    repository_matches: bool | None = None
    installed_git_commit: str | None = None
    checkout_git_commit: str | None = None
    checkout_dirty: bool | None = None


class ObservedEnvironmentSnapshot(EnvironmentModel):
    schema_version: Literal[1] = 1
    environment_key: str
    environment_name: str
    captured_at: datetime
    platform: str
    conda_prefix: Path | None = None
    python_version: str | None = None
    conda_version: str | None = None
    conda_packages: list[CondaPackageRecord] = Field(default_factory=list)
    pip_packages: list[PipPackageRecord] = Field(default_factory=list)
    editable_packages: list[PipPackageRecord] = Field(default_factory=list)
    review_requirements: list[str] = Field(default_factory=list)
    toolkit: ToolkitOverlayRecord = Field(default_factory=ToolkitOverlayRecord)
    repository_git_commit: str | None = None
    slurm_job_id: str | None = None
    execution_id: str | None = None
    technical_run_id: str | None = None


class SpecificationIssue(EnvironmentModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    path: Path | None = None


class SpecificationValidation(EnvironmentModel):
    environment_key: str
    conda_name: str
    valid: bool
    paths: EnvironmentPaths
    issues: list[SpecificationIssue] = Field(default_factory=list)


class DriftItem(EnvironmentModel):
    layer: Literal["specification", "conda-direct", "conda-lock", "pip", "toolkit"]
    kind: str
    package: str | None = None
    expected: str | None = None
    actual: str | None = None
    material: bool = True
    message: str


class EnvironmentComparison(EnvironmentModel):
    schema_version: Literal[1] = 1
    environment_key: str
    conda_name: str
    compared_at: datetime
    exists: bool
    completed: bool
    result: Literal["clean", "drift", "error", "missing"]
    specification: SpecificationValidation
    drift: list[DriftItem] = Field(default_factory=list)
    toolkit: ToolkitOverlayRecord = Field(default_factory=ToolkitOverlayRecord)
    observed_snapshot: ObservedEnvironmentSnapshot | None = None
    error: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def exit_code(self) -> int:
        if not self.completed or self.result in {"error", "missing"}:
            return 2
        return 1 if self.result == "drift" else 0


class EnvironmentSummary(EnvironmentModel):
    key: str
    conda_name: str
    managed: bool
    exists: bool | None = None
    drift: Literal["clean", "drift", "unknown", "error"] = "unknown"
    stages: list[str] = Field(default_factory=list)


class CondaEnvironmentRecord(EnvironmentModel):
    name: str
    prefix: Path
    platform: str
    is_base: bool = False


class EnvironmentCaptureTarget(EnvironmentModel):
    environment_key: str
    conda_name: str
    conda_prefix: Path
    platform: str
    registered: bool = False
    capture_directory_name: str


class DoctorCheck(EnvironmentModel):
    name: str
    status: Literal["ok", "warning", "error"]
    detail: str


class DoctorReport(EnvironmentModel):
    schema_version: Literal[1] = 1
    healthy: bool
    checks: list[DoctorCheck] = Field(default_factory=list)


class SmokeTestResult(EnvironmentModel):
    command: list[str]
    return_code: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_seconds: float
    passed: bool


class EnvironmentTestReport(EnvironmentModel):
    schema_version: Literal[1] = 1
    environment_key: str
    conda_name: str
    passed: bool
    tests: list[SmokeTestResult] = Field(default_factory=list)


class SyncPlan(EnvironmentModel):
    schema_version: Literal[1] = 1
    environment_key: str
    conda_name: str
    exists: bool
    drift: Literal["clean", "drift", "unknown"]
    recreation_required: bool
    actions: list[str]
    paths: EnvironmentPaths
    smoke_tests: list[list[str]] = Field(default_factory=list)


class OverlayRefreshResult(EnvironmentModel):
    environment_key: str
    conda_name: str
    status: Literal["updated", "planned", "skipped", "failed"]
    duration_seconds: float = 0.0
    message: str


class OverlayRefreshReport(EnvironmentModel):
    schema_version: Literal[1] = 1
    repository_root: Path
    dry_run: bool = False
    existing_only: bool = True
    results: list[OverlayRefreshResult] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return not any(item.status == "failed" for item in self.results)


class CapturePlan(EnvironmentModel):
    schema_version: Literal[1] = 1
    environment_key: str
    conda_name: str
    managed: bool = True
    registered: bool = True
    conda_prefix: Path | None = None
    candidate_directory: Path
    environment_yml: str
    pip_extras: str
    lockfile: Path | None = None
    lock_generation_error: str | None = None
    review_requirements: list[str] = Field(default_factory=list)
    excluded_toolkit: str | None = None
    differences: dict[str, str] = Field(default_factory=dict)


class SpecificationFileRecord(EnvironmentModel):
    path: Path
    sha256: str


class EnvironmentRuntimeRecord(EnvironmentModel):
    python_version: str | None = None
    conda_prefix: Path | None = None
    toolkit_editable: bool = False
    toolkit_git_commit: str | None = None
    toolkit_dirty: bool | None = None
    drift: Literal["clean", "drift", "unknown", "error"] = "unknown"


class EnvironmentProvenanceManifest(EnvironmentModel):
    schema_version: Literal[1] = 1
    environment_key: str
    conda_name: str
    platform: str
    specification: dict[str, SpecificationFileRecord] = Field(default_factory=dict)
    runtime: EnvironmentRuntimeRecord = Field(default_factory=EnvironmentRuntimeRecord)
    installed_snapshot: Path | None = None
    captured_at: datetime
    execution_id: str | None = None
    technical_run_id: str | None = None
    slurm_job_id: str | None = None
