"""Shared project, execution and reporting environment for execution backends."""

from __future__ import annotations
import re
import sys
from SpatialBiologyToolkit.environments.registry import load_environment_registry
from pathlib import Path
from .environment_selection import effective_environment_keys
from .executions import execution_output_path
from .project import ProjectContext
from .registry import get_stage
from .runs import RunRecord


def sbt_environment(
    context: ProjectContext,
    run: RunRecord,
    stage_name: str,
) -> dict[str, str]:
    stage = get_stage(stage_name)
    planned_stage = next(
        item for item in run.plan.resolved_stages if item.name == stage_name
    )
    toolkit_root = planned_stage.slurm_script.parent.parent
    execution = run.execution_for_stage(stage_name)
    outputs_root = Path(context.config.general.outputs_folder).expanduser()
    if not outputs_root.is_absolute():
        outputs_root = context.root / outputs_root
    outputs_root = outputs_root.resolve(strict=False)
    environment = {
        "SBT_TOOLKIT_ROOT": str(toolkit_root),
        "SBT_PROJECT_ROOT": str(context.root),
        "SBT_PROJECT_ID": context.project_metadata.project_id,
        "SBT_CONFIG": str(run.resolved_config_path),
        "SBT_EXECUTION_ID": str(execution.execution_id),
        "SBT_EXECUTION_LABEL": execution.execution_label,
        "SBT_OUTPUT_DIR": str(execution_output_path(context, execution)),
        "SBT_TECHNICAL_RUN_ID": execution.technical_run_id,
        "SBT_WORKFLOW_RUN_ID": run.workflow_run_id,
        # Transitional aliases retained for wrappers and older stage adapters.
        "SBT_RUN_ID": run.workflow_run_id,
        "SBT_RUN_DIR": str(run.run_dir),
        "SBT_STAGE": stage_name,
        "SBT_OUTPUTS_ROOT": str(outputs_root),
        "SBT_STAGE_OUTPUT_DIR": str(execution_output_path(context, execution)),
        "SBT_STAGE_DISPLAY_NAME": stage.display_name,
        "SBT_STAGE_DOCUMENTATION": stage.documentation_path,
        "SBT_REPORTING_PYTHON": sys.executable,
    }
    environment_registry = load_environment_registry()
    environment_keys = effective_environment_keys(run.plan, stage_name)
    if environment_keys:
        environment["SBT_ENVIRONMENT_KEY"] = environment_keys[0]
        environment["SBT_ENVIRONMENT_KEYS"] = ",".join(environment_keys)
        environment["SBT_CONDA_ENV"] = environment_registry.environments[
            environment_keys[0]
        ].conda_name
        for environment_key in environment_keys:
            variable = (
                "SBT_CONDA_ENV_" + re.sub(r"[^A-Za-z0-9]", "_", environment_key).upper()
            )
            environment[variable] = environment_registry.environments[
                environment_key
            ].conda_name
    if stage_name in run.plan.environment_overrides:
        environment["SBT_ENVIRONMENT_OVERRIDE"] = "1"
        environment["SBT_DEFAULT_ENVIRONMENT_KEYS"] = ",".join(stage.environment_keys)
    if run.manifest.reason:
        environment["SBT_RUN_REASON"] = run.manifest.reason
    if run.manifest.notes:
        environment["SBT_RUN_NOTES"] = "\n".join(run.manifest.notes)
    return environment
