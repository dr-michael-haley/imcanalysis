"""Direct SLURM submission backend for existing per-stage wrappers."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Callable

from .manifests import utc_now, write_yaml
from .models import (
    RunPlan,
    RunStatus,
    StageStatus,
    SubmissionRecord,
    SubmittedJobs,
)
from .project import ProjectContext
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
    return {
        "SBT_PROJECT_ROOT": str(context.root),
        "SBT_PROJECT_ID": context.project_metadata.project_id,
        "SBT_CONFIG": str(run.resolved_config_path),
        "SBT_RUN_ID": run.run_id,
        "SBT_RUN_DIR": str(run.run_dir),
        "SBT_STAGE": stage_name,
    }


def expected_log_paths(
    run_dir: Path, stage_name: str, job_id: str
) -> tuple[Path, Path]:
    logs_dir = run_dir / "logs"
    return (
        (logs_dir / f"{stage_name}-{job_id}.out").resolve(strict=False),
        (logs_dir / f"{stage_name}-{job_id}.err").resolve(strict=False),
    )


def build_sbatch_command(
    *,
    context: ProjectContext,
    run: RunRecord,
    stage_name: str,
    script: Path,
    dependency_job_id: str | None = None,
) -> list[str]:
    output_pattern = run.run_dir / "logs" / f"{stage_name}-%j.out"
    error_pattern = run.run_dir / "logs" / f"{stage_name}-%j.err"
    command = [
        "sbatch",
        "--parsable",
        f"--chdir={context.root}",
        f"--job-name=sbt_{run.run_id[-8:]}_{stage_name}",
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
            project_id=context.project_metadata.project_id,
            checked_at=utc_now(),
            overall_status=overall_status,
            stages=[
                StageStatus(
                    stage=job.stage,
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
    previous = None
    for index, stage in enumerate(plan.resolved_stages, start=1):
        fake_job_id = f"DRYRUN{index}"
        previews.append(
            (
                build_sbatch_command(
                    context=context,
                    run=run,
                    stage_name=stage.name,
                    script=stage.slurm_script,
                    dependency_job_id=previous,
                ),
                sbt_environment(context, run, stage.name),
            )
        )
        previous = fake_job_id
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

    submitted = SubmittedJobs(run_id=run.run_id)
    previous_job_id: str | None = None
    for stage in plan.resolved_stages:
        exported = sbt_environment(context, run, stage.name)
        command = build_sbatch_command(
            context=context,
            run=run,
            stage_name=stage.name,
            script=stage.slurm_script,
            dependency_job_id=previous_job_id,
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
            output_pattern = run.run_dir / "logs" / f"{stage.name}-%j.out"
            error_pattern = run.run_dir / "logs" / f"{stage.name}-%j.err"
            submitted.jobs.append(
                SubmissionRecord(
                    stage=stage.name,
                    state="submission_failed",
                    dependency_job_id=previous_job_id,
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
            raise SubmissionError(
                f"Failed to submit stage '{stage.name}': {exc}",
                submitted,
            ) from exc

        stdout_log, stderr_log = expected_log_paths(run.run_dir, stage.name, job_id)
        submitted.jobs.append(
            SubmissionRecord(
                stage=stage.name,
                state="submitted",
                job_id=job_id,
                dependency_job_id=previous_job_id,
                submitted_at=utc_now(),
                command=command,
                exported_environment=exported,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
            )
        )
        _write_submission_state(context, run, submitted, overall_status="submitted")
        previous_job_id = job_id

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
