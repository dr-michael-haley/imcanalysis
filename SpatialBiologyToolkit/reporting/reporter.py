"""StageReporter lifecycle, bootstrap integration, and durable report writing."""

from __future__ import annotations

import atexit
import builtins
import importlib.metadata
import logging
import os
import subprocess
import sys
import traceback as traceback_module
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, NoReturn

from SpatialBiologyToolkit.pipeline.manifests import read_model, write_yaml
from SpatialBiologyToolkit.pipeline.registry import get_stage, toolkit_root

from .inventory import (
    configured_stage_paths,
    discover_generated_files,
    extract_stage_parameters,
)
from .models import ErrorRecord, GeneratedFile, PathRecord, StageManifest
from .paths import (
    ReportingContext,
    documentation_path,
    infer_stage_from_main_module,
    resolve_reporting_context,
)
from .render import write_indexes


STAGE_MANIFEST = "stage_manifest.yaml"
_ACTIVE_REPORTER: "StageReporter | None" = None
_BOOTSTRAP_FINALIZED = False
_ORIGINAL_SYS_EXIT = sys.exit


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _pipeline_version() -> str | None:
    try:
        return importlib.metadata.version("SpatialBiologyToolkit")
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(toolkit_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else None


def _split_notes(value: str | None) -> list[str]:
    if not value:
        return []
    return [line.strip() for line in value.splitlines() if line.strip()]


def _started_at_from_environment() -> datetime:
    raw = os.environ.get("SBT_STAGE_STARTED_AT")
    if raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    return _utc_now()


class StageReporter:
    """Collect and render one stage's human-facing scientific record."""

    def __init__(self, context: ReportingContext):
        self.context = context
        self.spec = get_stage(context.stage)
        self.manifest_path = context.stage_run_dir / STAGE_MANIFEST
        self._finalized = False
        self.manifest = self._initial_manifest()

    @classmethod
    def from_environment(cls, stage: str | None = None) -> "StageReporter":
        return cls(resolve_reporting_context(stage))

    def _initial_manifest(self) -> StageManifest:
        if self.manifest_path.is_file():
            try:
                existing = read_model(self.manifest_path, StageManifest)
                return existing.model_copy(update={"status": "running"})
            except (OSError, ValueError):
                pass
        doc_path = documentation_path(self.context.stage)
        try:
            explainer = doc_path.read_text(encoding="utf-8")
        except OSError:
            explainer = ""
        slurm_job_id = (
            os.environ.get("SBT_SLURM_JOB_ID")
            or os.environ.get("IMC_SLURM_JOB_ID")
            or os.environ.get("SLURM_JOB_ID")
        )
        return StageManifest(
            project_id=self.context.project_id,
            run_id=self.context.run_id,
            stage=self.context.stage,
            display_name=self.spec.display_name,
            status="running",
            managed_run=self.context.managed_run,
            started_at=_started_at_from_environment(),
            pipeline_version=_pipeline_version(),
            git_commit=_git_commit(),
            slurm_job_id=slurm_job_id,
            technical_run_record=self.context.technical_run_record,
            reason=os.environ.get("SBT_RUN_REASON") or None,
            notes=_split_notes(os.environ.get("SBT_RUN_NOTES")),
            documentation_source=doc_path if doc_path.is_file() else None,
            explainer_snapshot=explainer,
        )

    def __enter__(self) -> "StageReporter":
        self.context.stage_run_dir.mkdir(parents=True, exist_ok=True)
        for directory in (
            self.context.figures_dir,
            self.context.tables_dir,
            self.context.summaries_dir,
            self.context.files_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("SBT_STAGE", self.context.stage)
        os.environ.setdefault("SBT_PROJECT_ROOT", str(self.context.project_root))
        os.environ.setdefault("SBT_PROJECT_ID", self.context.project_id)
        os.environ.setdefault("SBT_RUN_ID", self.context.run_id)
        os.environ.setdefault("SBT_OUTPUTS_ROOT", str(self.context.outputs_root))
        os.environ.setdefault(
            "SBT_STAGE_OUTPUT_DIR", str(self.context.stage_run_dir)
        )
        if not self.context.managed_run:
            warning = (
                "Direct execution: no managed SBT technical run record is available."
            )
            if warning not in self.manifest.warnings:
                self.manifest.warnings.append(warning)
        write_yaml(self.manifest_path, self.manifest)
        return self

    def add_input(self, role: str, path: str | Path, description: str = "") -> None:
        resolved = Path(path).expanduser().resolve(strict=False)
        self.manifest.inputs.append(
            PathRecord(
                role=role,
                path=resolved,
                description=description,
                exists=resolved.exists(),
            )
        )

    def add_asset(self, role: str, path: str | Path, description: str = "") -> None:
        resolved = Path(path).expanduser().resolve(strict=False)
        self.manifest.produced_assets.append(
            PathRecord(
                role=role,
                path=resolved,
                description=description,
                exists=resolved.exists(),
            )
        )

    def add_file(
        self,
        category: Literal["figure", "table", "summary", "file"],
        path: str | Path,
        description: str = "",
    ) -> None:
        resolved = Path(path).expanduser().resolve(strict=False)
        self.manifest.generated_files.append(
            GeneratedFile(
                category=category,
                path=resolved,
                description=description,
                size_bytes=resolved.stat().st_size if resolved.is_file() else None,
            )
        )

    def add_metric(self, name: str, value: Any) -> None:
        self.manifest.metrics[str(name)] = value

    def add_warning(self, message: str) -> None:
        self.manifest.warnings.append(str(message))

    def add_note(self, note: str) -> None:
        self.manifest.notes.append(str(note))

    @staticmethod
    def _merge_paths(
        existing: list[PathRecord],
        discovered: list[PathRecord],
    ) -> list[PathRecord]:
        merged = {(item.role, str(item.path)): item for item in discovered}
        for item in existing:
            merged[(item.role, str(item.path))] = item
        return list(merged.values())

    @staticmethod
    def _merge_files(
        existing: list[GeneratedFile],
        discovered: list[GeneratedFile],
    ) -> list[GeneratedFile]:
        merged = {str(item.path): item for item in discovered}
        for item in existing:
            merged[str(item.path)] = item
        return list(merged.values())

    def finalize(
        self,
        *,
        status: Literal["running", "completed", "failed"] = "completed",
        error: BaseException | None = None,
        traceback_text: str | None = None,
    ) -> StageManifest:
        if self._finalized and status == self.manifest.status:
            return self.manifest
        completed_at = _utc_now()
        self.manifest.status = "failed" if error is not None else status
        self.manifest.completed_at = completed_at
        self.manifest.duration_seconds = max(
            0.0, (completed_at - self.manifest.started_at).total_seconds()
        )
        if error is not None:
            self.manifest.errors.append(
                ErrorRecord(
                    type=type(error).__name__,
                    message=str(error),
                    traceback=traceback_text,
                )
            )

        inputs, assets = configured_stage_paths(self.context)
        self.manifest.inputs = self._merge_paths(self.manifest.inputs, inputs)
        self.manifest.produced_assets = self._merge_paths(
            self.manifest.produced_assets, assets
        )
        self.manifest.generated_files = self._merge_files(
            self.manifest.generated_files,
            discover_generated_files(self.context.stage_run_dir),
        )
        self.manifest.parameters.update(extract_stage_parameters(self.context))
        self.manifest.metrics.setdefault(
            "generated_files", len(self.manifest.generated_files)
        )
        self.manifest.metrics.setdefault(
            "figures",
            sum(
                item.category == "figure"
                for item in self.manifest.generated_files
            ),
        )
        self.manifest.metrics.setdefault(
            "tables",
            sum(
                item.category == "table"
                for item in self.manifest.generated_files
            ),
        )

        write_yaml(self.manifest_path, self.manifest)
        if self.context.technical_run_record:
            event_path = (
                self.context.technical_run_record
                / "stage_events"
                / f"{self.context.stage}.yaml"
            )
            write_yaml(event_path, self.manifest)

        try:
            write_indexes(self.context, self.manifest)
        except Exception as render_error:
            message = f"{type(render_error).__name__}: {render_error}"
            self.manifest.rendering_errors.append(message)
            logging.exception("Stage report Markdown rendering failed: %s", render_error)
            write_yaml(self.manifest_path, self.manifest)
            if self.context.technical_run_record:
                write_yaml(event_path, self.manifest)
        self._finalized = True
        return self.manifest

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        traceback_text = (
            "".join(traceback_module.format_exception(exc_type, exc, tb))
            if exc_type and exc
            else None
        )
        self.finalize(
            status="completed",
            error=exc,
            traceback_text=traceback_text,
        )
        return False


def _finalize_bootstrap() -> None:
    global _BOOTSTRAP_FINALIZED
    if _BOOTSTRAP_FINALIZED or _ACTIVE_REPORTER is None:
        return
    _ACTIVE_REPORTER.finalize(status="completed")
    _BOOTSTRAP_FINALIZED = True


def _bootstrap_excepthook(
    exc_type: type[BaseException],
    exc: BaseException,
    tb: TracebackType | None,
) -> None:
    global _BOOTSTRAP_FINALIZED
    if _ACTIVE_REPORTER is not None and not _BOOTSTRAP_FINALIZED:
        _ACTIVE_REPORTER.finalize(
            status="failed",
            error=exc,
            traceback_text="".join(
                traceback_module.format_exception(exc_type, exc, tb)
            ),
        )
        _BOOTSTRAP_FINALIZED = True
    sys.__excepthook__(exc_type, exc, tb)


def _bootstrap_sys_exit(code: str | int | None = 0) -> NoReturn:
    global _BOOTSTRAP_FINALIZED
    failed = code not in (None, 0, "0")
    if _ACTIVE_REPORTER is not None and not _BOOTSTRAP_FINALIZED:
        if failed:
            error = RuntimeError(f"Stage process exited with status {code}.")
            _ACTIVE_REPORTER.finalize(status="failed", error=error)
        else:
            _ACTIVE_REPORTER.finalize(status="completed")
        _BOOTSTRAP_FINALIZED = True
    _ORIGINAL_SYS_EXIT(code)


def bootstrap_stage_reporting(stage: str | None = None) -> StageReporter | None:
    """Start process-wide reporting for a stage module and finalize at exit."""
    global _ACTIVE_REPORTER
    if _ACTIVE_REPORTER is not None:
        return _ACTIVE_REPORTER
    stage_name = stage or os.environ.get("SBT_STAGE") or infer_stage_from_main_module()
    if not stage_name:
        return None
    reporter = StageReporter.from_environment(stage_name)
    reporter.__enter__()
    _ACTIVE_REPORTER = reporter
    atexit.register(_finalize_bootstrap)
    sys.excepthook = _bootstrap_excepthook
    sys.exit = _bootstrap_sys_exit
    builtins.exit = _bootstrap_sys_exit  # type: ignore[assignment]
    builtins.quit = _bootstrap_sys_exit  # type: ignore[assignment]
    return reporter


def ensure_stage_reporter() -> StageReporter | None:
    """Idempotently bootstrap reporting when a registered stage is executing."""
    return bootstrap_stage_reporting()


def get_active_reporter() -> StageReporter | None:
    return _ACTIVE_REPORTER


__all__ = [
    "STAGE_MANIFEST",
    "StageReporter",
    "bootstrap_stage_reporting",
    "ensure_stage_reporter",
    "get_active_reporter",
]
