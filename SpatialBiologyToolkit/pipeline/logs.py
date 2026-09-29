"""Resolve and safely tail logs recorded for an SBT run."""

from __future__ import annotations

from pathlib import Path

from .models import LogRecord
from .runs import load_submitted_jobs, load_run_manifest


def resolve_run_logs(
    run_dir: str | Path,
    *,
    stage: str | None = None,
    include_stdout: bool = True,
    include_stderr: bool = True,
) -> list[LogRecord]:
    directory = Path(run_dir)
    if load_run_manifest(directory).execution_backend == "local":
        from .local import load_local_run
        state = load_local_run(directory)
        if stage is not None and stage not in {item.stage for item in state.stages}:
            raise KeyError(f"Stage '{stage}' is not recorded in this run.")
        # Local stdout/stderr share a log to preserve their order; return it once.
        return [LogRecord(stage=item.stage, job_id=None, stream="stdout", path=item.log,
                          exists=item.log.is_file()) for item in state.stages
                if (stage is None or item.stage == stage) and (include_stdout or include_stderr)]
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


def follow_logs(context, run_dir: Path, technical_run_id: str, records: list[LogRecord], line_count: int):
    """Tail existing content then follow each file, including files created later."""
    import time
    from .status import inspect_run_status

    positions: dict[Path, int] = {}
    for record in records:
        if not record.path.exists():
            positions[record.path] = 0
        yield f"[{record.stage} {record.stream}] {record.path}\n"
    while True:
        # Check completion before reading, so the final read drains terminal output.
        report = inspect_run_status(context, run_dir)
        selected = next((item for item in report.stages if item.technical_run_id == technical_run_id), None)
        for record in records:
            try:
                with record.path.open("rb") as handle:
                    end = handle.seek(0, 2)
                    if record.path not in positions:
                        position = end
                        data = b""
                        while position > 0 and line_count and data.count(b"\n") <= line_count:
                            size = min(position, 8192)
                            position -= size
                            handle.seek(position)
                            data = handle.read(size) + data
                        text = b"\n".join(data.splitlines()[-line_count:]) if line_count else b""
                        if text:
                            yield text.decode("utf-8", errors="replace") + "\n"
                    else:
                        start = positions[record.path]
                        handle.seek(start if start <= end else 0)
                        data = handle.read(end - handle.tell())
                        if data:
                            yield data.decode("utf-8", errors="replace")
                    positions[record.path] = end
            except FileNotFoundError:
                continue
        if selected is None or selected.status not in {"pending", "running"}:
            return
        time.sleep(1)
