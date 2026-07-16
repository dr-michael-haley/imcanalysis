"""Resolve and safely tail logs recorded for an SBT run."""

from __future__ import annotations

from pathlib import Path

from .models import LogRecord
from .runs import load_submitted_jobs


def resolve_run_logs(
    run_dir: str | Path,
    *,
    stage: str | None = None,
    include_stdout: bool = True,
    include_stderr: bool = True,
) -> list[LogRecord]:
    directory = Path(run_dir)
    submitted = load_submitted_jobs(directory)
    if stage is not None and stage not in {job.stage for job in submitted.jobs}:
        valid = ", ".join(job.stage for job in submitted.jobs)
        raise KeyError(
            f"Stage '{stage}' is not recorded in this run. Valid stages: {valid}"
        )

    records: list[LogRecord] = []
    for job in submitted.jobs:
        if stage is not None and job.stage != stage:
            continue
        if include_stdout:
            records.append(
                LogRecord(
                    stage=job.stage,
                    job_id=job.job_id,
                    stream="stdout",
                    path=job.stdout_log,
                    exists=job.stdout_log.is_file(),
                )
            )
        if include_stderr:
            records.append(
                LogRecord(
                    stage=job.stage,
                    job_id=job.job_id,
                    stream="stderr",
                    path=job.stderr_log,
                    exists=job.stderr_log.is_file(),
                )
            )
    return records


def tail_text(path: str | Path, line_count: int = 40) -> str:
    if line_count < 0:
        raise ValueError("tail line count must be zero or greater")
    source = Path(path)
    if line_count == 0:
        return ""
    with source.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        blocks: list[bytes] = []
        newline_count = 0
        block_size = 8192
        while position > 0 and newline_count <= line_count:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            block = handle.read(read_size)
            blocks.append(block)
            newline_count += block.count(b"\n")
    data = b"".join(reversed(blocks)).decode("utf-8", errors="replace")
    return "\n".join(data.splitlines()[-line_count:])


__all__ = ["resolve_run_logs", "tail_text"]
