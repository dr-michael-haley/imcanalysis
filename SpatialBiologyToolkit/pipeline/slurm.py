"""Direct SLURM submission backend for existing per-stage wrappers."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

from SpatialBiologyToolkit.environments.registry import load_environment_registry

from .manifests import utc_now, write_yaml
from .executions import execution_output_path, update_execution
from .models import (
    RunPlan,
    RunStatus,
    StageStatus,
    SubmissionRecord,
    SubmittedJobs,
)
from .project import ProjectContext
from .registry import get_stage
from .runs import STATUS_FILE, SUBMITTED_JOBS, RunRecord


Runner = Callable[..., subprocess.CompletedProcess[str]]
JOB_ID_PATTERN = re.compile(r"^\d+$")


class SubmissionError(RuntimeError):
    def __init__(self, message: str, submitted_jobs: SubmittedJobs):
        super().__init__(message)
        self.submitted_jobs = submitted_jobs


def sbt_environment(
    context: ProjectContext,
    run: RunRecord,
    stage_name: str,
) -> dict[str, str]:
    stage = get_stage(stage_name)
    execution = run.execution_for_stage(stage_name)
    outputs_root = Path(context.config.general.outputs_folder).expanduser()
    if not outputs_root.is_absolute():
        outputs_root = context.root / outputs_root
    outputs_root = outputs_root.resolve(strict=False)
    environment = {
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
    environment_keys = stage.environment_keys
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
    if run.manifest.reason:
        environment["SBT_RUN_REASON"] = run.manifest.reason
    if run.manifest.notes:
        environment["SBT_RUN_NOTES"] = "\n".join(run.manifest.notes)
    return environment


def expected_log_paths(
    run_dir: Path, stage_name: str, job_id: str
) -> tuple[Path, Path]:
    logs_dir = run_dir / "logs"
    return (
        (logs_dir / f"{stage_name}_{job_id}.out").resolve(strict=False),
        (logs_dir / f"{stage_name}_{job_id}.err").resolve(strict=False),
    )


def build_sbatch_command(
    *,
    context: ProjectContext,
    run: RunRecord,
    stage_name: str,
    script: Path,
    dependency_job_id: str | None = None,
) -> list[str]:
    output_pattern = run.run_dir / "logs" / f"{stage_name}_%j.out"
    error_pattern = run.run_dir / "logs" / f"{stage_name}_%j.err"
    command = [
        "sbatch",
        "--parsable",
        f"--chdir={context.root}",
        f"--job-name=sbt_{run.execution_for_stage(stage_name).execution_label}_{stage_name}",
        f"--output={output_pattern}",
        f"--error={error_pattern}",
        "--export=ALL",
    ]
    if dependency_job_id:
        command.append(f"--dependency=afterok:{dependency_job_id}")
    command.append(str(script))
    return command


def parse_job_id(stdout: str) -> str:
    candidate = stdout.strip().split(";", 1)[0].strip()
    if not JOB_ID_PATTERN.fullmatch(candidate):
        raise ValueError(f"Could not parse SLURM job ID from sbatch output: {stdout!r}")
    return candidate


def _write_submission_state(
    context: ProjectContext,
    run: RunRecord,
    submitted: SubmittedJobs,
    *,
    overall_status: str,
) -> None:
    write_yaml(run.run_dir / SUBMITTED_JOBS, submitted)
    write_yaml(
        run.run_dir / STATUS_FILE,
        RunStatus(
            run_id=run.run_id,
            workflow_run_id=run.workflow_run_id,
            project_id=context.project_metadata.project_id,
            checked_at=utc_now(),
            overall_status=overall_status,
            stages=[
                StageStatus(
                    stage=job.stage,
                    execution_id=job.execution_id,
                    technical_run_id=job.technical_run_id,
                    job_id=job.job_id,
                    status=("pending" if job.state == "submitted" else "not_submitted"),
                    source="submission",
                    detail=job.error,
                )
                for job in submitted.jobs
            ],
        ),
    )


def preview_submission_commands(
    context: ProjectContext,
    plan: RunPlan,
    run: RunRecord,
) -> list[tuple[list[str], dict[str, str]]]:
    previews: list[tuple[list[str], dict[str, str]]] = []
    preview_job_ids: dict[str, str] = {}
    for index, stage in enumerate(plan.resolved_stages, start=1):
        fake_job_id = f"DRYRUN{index}"
        dependency_ids = [
            preview_job_ids[name]
            for name in stage.depends_on
            if name in preview_job_ids
        ]
        previews.append(
            (
                build_sbatch_command(
                    context=context,
                    run=run,
                    stage_name=stage.name,
                    script=stage.slurm_script,
                    dependency_job_id=":".join(dependency_ids) or None,
                ),
                sbt_environment(context, run, stage.name),
            )
        )
        preview_job_ids[stage.name] = fake_job_id
    return previews


def submit_run(
    context: ProjectContext,
    plan: RunPlan,
    run: RunRecord,
    *,
    runner: Runner = subprocess.run,
) -> SubmittedJobs:
    if not plan.ready:
        raise ValueError("Refusing to submit an invalid run plan.")

    submitted = SubmittedJobs(
        run_id=run.workflow_run_id,
        workflow_run_id=run.workflow_run_id,
    )
    submitted_job_ids: dict[str, str] = {}
    for stage in plan.resolved_stages:
        execution = run.execution_for_stage(stage.name)
        exported = sbt_environment(context, run, stage.name)
        dependency_ids = [
            submitted_job_ids[name]
            for name in stage.depends_on
            if name in submitted_job_ids
        ]
        dependency_job_id = ":".join(dependency_ids) or None
        command = build_sbatch_command(
            context=context,
            run=run,
            stage_name=stage.name,
            script=stage.slurm_script,
            dependency_job_id=dependency_job_id,
        )
        process_environment = os.environ.copy()
        process_environment.update(exported)
        try:
            completed = runner(
                command,
                cwd=context.root,
                env=process_environment,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or f"sbatch exited with status {completed.returncode}"
                )
            job_id = parse_job_id(completed.stdout)
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            output_pattern = run.run_dir / "logs" / f"{stage.name}_%j.out"
            error_pattern = run.run_dir / "logs" / f"{stage.name}_%j.err"
            submitted.jobs.append(
                SubmissionRecord(
                    stage=stage.name,
                    execution_id=execution.execution_id,
                    technical_run_id=execution.technical_run_id,
                    state="submission_failed",
                    dependency_job_id=dependency_job_id,
                    submitted_at=utc_now(),
                    command=command,
                    exported_environment=exported,
                    stdout_log=output_pattern,
                    stderr_log=error_pattern,
                    error=str(exc),
                )
            )
            _write_submission_state(
                context,
                run,
                submitted,
                overall_status="partial_submission_failed",
            )
            update_execution(
                context,
                execution.technical_run_id,
                status="failed",
                completed_at=utc_now(),
            )
            failed_position = [item.name for item in plan.resolved_stages].index(
                stage.name
            )
            for blocked_stage in plan.resolved_stages[failed_position + 1 :]:
                blocked = run.execution_for_stage(blocked_stage.name)
                update_execution(
                    context,
                    blocked.technical_run_id,
                    status="blocked",
                    completed_at=utc_now(),
                )
            raise SubmissionError(
                f"Failed to submit stage '{stage.name}': {exc}",
                submitted,
            ) from exc

        stdout_log, stderr_log = expected_log_paths(run.run_dir, stage.name, job_id)
        submitted.jobs.append(
            SubmissionRecord(
                stage=stage.name,
                execution_id=execution.execution_id,
                technical_run_id=execution.technical_run_id,
                state="submitted",
                job_id=job_id,
                dependency_job_id=dependency_job_id,
                submitted_at=utc_now(),
                command=command,
                exported_environment=exported,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
            )
        )
        updated_execution = update_execution(
            context,
            execution.technical_run_id,
            status="pending",
            slurm_job_id=job_id,
        )
        from SpatialBiologyToolkit.reporting.render import prepare_execution_output

        prepare_execution_output(context, run, updated_execution)
        _write_submission_state(context, run, submitted, overall_status="submitted")
        submitted_job_ids[stage.name] = job_id

    submitted.submission_complete = True
    _write_submission_state(context, run, submitted, overall_status="submitted")
    return submitted


__all__ = [
    "SubmissionError",
    "build_sbatch_command",
    "expected_log_paths",
    "parse_job_id",
    "preview_submission_commands",
    "sbt_environment",
    "submit_run",
]
