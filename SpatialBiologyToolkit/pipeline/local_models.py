"""Durable local execution handles; PIDs are never scheduler job IDs."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from .models import PipelineModel


class ProcessIdentity(PipelineModel):
    pid: int = Field(gt=0)
    start_ticks: int = Field(ge=0)
    boot_id: str
    hostname: str


class LocalCommand(PipelineModel):
    argv: list[str] = Field(min_length=1)
    environment: str
    variables: dict[str, str] = Field(default_factory=dict)


class LocalStage(PipelineModel):
    stage: str
    execution_id: int
    technical_run_id: str
    commands: list[LocalCommand] = Field(min_length=1)
    log: Path
    status: Literal[
        "pending", "running", "completed", "failed", "cancelled", "blocked", "unknown"
    ] = "pending"
    process: ProcessIdentity | None = None
    command_index: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    exit_code: int | None = None
    detail: str | None = None


class LocalRun(PipelineModel):
    schema_version: Literal[1] = 1
    backend: Literal["local"] = "local"
    workflow_run_id: str
    worker: ProcessIdentity | None = None
    status: Literal[
        "starting", "running", "completed", "failed", "cancelled", "unknown"
    ] = "starting"
    stages: list[LocalStage]
    error: str | None = None
