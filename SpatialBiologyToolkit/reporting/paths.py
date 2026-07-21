"""Resolve managed and direct-execution reporting paths."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import yaml  # type: ignore[import-untyped]

from SpatialBiologyToolkit.pipeline.registry import get_stage, toolkit_root


@dataclass(frozen=True)
class ReportingContext:
    stage: str
    project_root: Path
    project_id: str
    execution_id: int | None
    execution_label: str
    technical_run_id: str
    workflow_run_id: str
    outputs_root: Path
    output_dir: Path
    technical_run_record: Path | None
    config_path: Path | None
    managed_run: bool

    @property
    def stage_run_dir(self) -> Path:
        """Compatibility alias for the resolved execution output directory."""
        return self.output_dir

    @property
    def stage_root(self) -> Path:
        """Compatibility alias; stage types no longer own fixed output roots."""
        return self.output_dir.parent

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def summaries_dir(self) -> Path:
        return self.output_dir / "summaries"

    @property
    def files_dir(self) -> Path:
        return self.output_dir / "files"


def infer_stage_from_main_module() -> str | None:
    """Infer a stage alias for ``python -m`` direct execution."""
    main_module = sys.modules.get("__main__")
    spec = getattr(main_module, "__spec__", None)
    module_name = getattr(spec, "name", None)
    if not module_name:
        return None

    from SpatialBiologyToolkit.pipeline.registry import STAGES

    matches = [stage for stage in STAGES if module_name in stage.python_modules]
    if matches:
        # A module may belong both to a checkpoint stage and a composite stage.
        # Direct ``python -m`` execution should report against the atomic stage;
        # managed composite jobs always provide SBT_STAGE explicitly.
        return min(matches, key=lambda stage: len(stage.python_modules)).name
    return None


def _read_yaml_mapping(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _project_metadata(root: Path) -> dict:
    return _read_yaml_mapping(root / ".sbt" / "project.yaml")


def _config_path(environment: Mapping[str, str], root: Path) -> Path | None:
    raw = environment.get("SBT_CONFIG")
    if raw:
        return Path(raw).expanduser().resolve(strict=False)
    candidate = root / "config.yaml"
    return candidate.resolve(strict=False) if candidate.is_file() else None


def _outputs_root(
    environment: Mapping[str, str],
    root: Path,
    config_path: Path | None,
) -> Path:
    explicit = environment.get("SBT_OUTPUTS_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve(strict=False)
    config = _read_yaml_mapping(config_path)
    general = config.get("general", {})
    configured = (
        general.get("outputs_folder", "outputs")
        if isinstance(general, dict)
        else "outputs"
    )
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def _direct_run_id(now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"direct-{timestamp}-{os.getpid()}"


def resolve_reporting_context(
    stage: str | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> ReportingContext:
    env = os.environ if environment is None else environment
    stage_name = stage or env.get("SBT_STAGE") or infer_stage_from_main_module()
    if not stage_name:
        raise ValueError(
            "Cannot determine the reporting stage. Set SBT_STAGE or run a registered "
            "pipeline module with 'python -m'."
        )
    spec = get_stage(stage_name)

    root = Path(env.get("SBT_PROJECT_ROOT", Path.cwd())).expanduser().resolve(
        strict=False
    )
    metadata = _project_metadata(root)
    project_id = str(
        env.get("SBT_PROJECT_ID")
        or metadata.get("project_id")
        or f"direct:{root.name or 'project'}"
    )
    workflow_run_id = (
        env.get("SBT_WORKFLOW_RUN_ID")
        or env.get("SBT_RUN_ID")
        or _direct_run_id(now)
    )
    technical_run_id = (
        env.get("SBT_TECHNICAL_RUN_ID")
        or env.get("SBT_RUN_ID")
        or workflow_run_id
    )
    raw_execution_id = env.get("SBT_EXECUTION_ID")
    execution_id = int(raw_execution_id) if raw_execution_id else None
    execution_label = env.get("SBT_EXECUTION_LABEL") or (
        f"{execution_id:03d}" if execution_id is not None else technical_run_id
    )
    managed = bool(env.get("SBT_RUN_DIR") and env.get("SBT_STAGE_OUTPUT_DIR"))
    config_path = _config_path(env, root)
    outputs_root = _outputs_root(env, root, config_path)
    explicit_output_dir = env.get("SBT_OUTPUT_DIR") or env.get("SBT_STAGE_OUTPUT_DIR")
    if explicit_output_dir:
        output_dir = Path(explicit_output_dir).expanduser().resolve(strict=False)
    else:
        direct_id = _direct_run_id(now)
        execution_label = direct_id
        technical_run_id = direct_id
        workflow_run_id = direct_id
        output_dir = (
            outputs_root / "direct" / f"{direct_id}_{spec.output_slug}"
        ).resolve(strict=False)
    technical = env.get("SBT_RUN_DIR")
    technical_path = (
        Path(technical).expanduser().resolve(strict=False) if technical else None
    )
    return ReportingContext(
        stage=stage_name,
        project_root=root,
        project_id=project_id,
        execution_id=execution_id,
        execution_label=execution_label,
        technical_run_id=technical_run_id,
        workflow_run_id=workflow_run_id,
        outputs_root=outputs_root,
        output_dir=output_dir,
        technical_run_record=technical_path,
        config_path=config_path,
        managed_run=managed,
    )


def category_output_path(
    category: str,
    filename: str | Path | None = None,
    *,
    stage: str | None = None,
) -> Path:
    """Return a standard stage-run category path, with a direct-run fallback."""
    context = resolve_reporting_context(stage)
    directories = {
        "figure": context.figures_dir,
        "figures": context.figures_dir,
        "table": context.tables_dir,
        "tables": context.tables_dir,
        "summary": context.summaries_dir,
        "summaries": context.summaries_dir,
        "file": context.files_dir,
        "files": context.files_dir,
    }
    try:
        directory = directories[category.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown reporting category: {category}") from exc
    directory.mkdir(parents=True, exist_ok=True)
    return directory if filename is None else directory / filename


def optional_category_output_path(
    category: str,
    fallback: str | Path,
) -> Path:
    """Use a report category when active, otherwise preserve a legacy path."""
    if not (
        os.environ.get("SBT_STAGE_OUTPUT_DIR")
        or os.environ.get("SBT_STAGE")
        or infer_stage_from_main_module()
    ):
        return Path(fallback)
    try:
        return category_output_path(category)
    except (KeyError, OSError, ValueError):
        return Path(fallback)


def project_asset_path(configured_path: str | Path) -> Path:
    """Resolve a reusable asset against the canonical project root."""
    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    root = Path(os.environ.get("SBT_PROJECT_ROOT", Path.cwd())).expanduser()
    return (root / path).resolve(strict=False)


def documentation_path(stage: str) -> Path:
    return (toolkit_root() / get_stage(stage).documentation_path).resolve(strict=False)


__all__ = [
    "ReportingContext",
    "category_output_path",
    "documentation_path",
    "infer_stage_from_main_module",
    "optional_category_output_path",
    "project_asset_path",
    "resolve_reporting_context",
]
