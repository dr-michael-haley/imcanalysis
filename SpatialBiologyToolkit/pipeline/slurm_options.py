"""Per-submission overrides for the directives in SBT's SLURM wrappers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Annotated, NamedTuple

from pydantic import AfterValidator


_COUNT = r"[1-9][0-9]*"
_NAME = r"[A-Za-z0-9_][A-Za-z0-9_.-]*"
_GPU = rf"(?:{_NAME}:)?{_COUNT}"
_DURATION = re.compile(
    r"(?:([0-9]+)d)?(?:([0-9]+)h)?(?:([0-9]+)m)?(?:([0-9]+)s)?",
    re.IGNORECASE | re.ASCII,
)


class SlurmOption(NamedTuple):
    pattern: str
    example: str
    description: str


SLURM_OPTIONS = {
    "time": SlurmOption(r"(?:[0-9]+-)?[0-9]+(?::[0-9]+){0,2}", "2h", "Wall-clock time limit (-t)"),
    "partition": SlurmOption(rf"{_NAME}(?:,{_NAME})*", "gpuA", "Queue to submit to (-p)"),
    "gpus": SlurmOption(rf"{_GPU}(?:,{_GPU})*", "1", "GPUs for the job (-G)"),
    "ntasks": SlurmOption(_COUNT, "8", "Total task slots (-n)"),
    "cpus-per-task": SlurmOption(_COUNT, "4", "CPU cores for each task (-c)"),
    "ntasks-per-node": SlurmOption(_COUNT, "2", "Task slots on each node"),
    "nodes": SlurmOption(rf"{_COUNT}(?:-{_COUNT})?", "1", "Compute nodes for the job (-N)"),
    "mem": SlurmOption(r"[0-9]+(?:\.[0-9]+)?[KkMmGgTt]?", "32G", "Memory per node"),
    "mail-type": SlurmOption(r"[A-Za-z0-9_]+(?:,[A-Za-z0-9_]+)*", "FAIL", "Events that trigger email"),
    "mail-user": SlurmOption(r"[^\s=,]+(?:,[^\s=,]+)*", "user@example.org", "Notification recipient"),
}
SLURM_OPTION_NAMES = ", ".join(SLURM_OPTIONS)


def _slurm_value(key: str, value: str) -> str:
    option = SLURM_OPTIONS.get(key)
    if option is None:
        raise ValueError(
            f"Unsupported --slurm key {key!r}. Choose from {SLURM_OPTION_NAMES}. "
            "See 'sbt slurm-help' for examples."
        )
    if value.isprintable() and re.fullmatch(option.pattern, value):
        return value
    if key == "time" and value:
        duration = _DURATION.fullmatch(value)
        if duration:
            days, hours, minutes, seconds = (int(part or 0) for part in duration.groups())
            total_minutes = days * 1440 + hours * 60 + minutes + (seconds + 59) // 60
            if total_minutes:
                return str(total_minutes)
    raise ValueError(
        f"Invalid --slurm value for {key!r}: {value!r}. "
        f"Try {key}={option.example} or see 'sbt slurm-help'."
    )


def _validate_options(options: dict[str, str]) -> dict[str, str]:
    return {key: _slurm_value(key, value) for key, value in options.items()}


SlurmOptions = Annotated[dict[str, str], AfterValidator(_validate_options)]


def resolve_slurm_overrides(
    assignments: Iterable[str], stage_names: Iterable[str]
) -> dict[str, dict[str, str]]:
    """Resolve [stage.]key=value assignments, with stage settings taking precedence."""
    stages = list(stage_names)
    defaults: dict[str, str] = {}
    per_stage: dict[str, dict[str, str]] = {}
    for assignment in assignments:
        selector, equals, value = assignment.partition("=")
        if not equals or not selector or not value:
            raise ValueError(f"Expected --slurm [STAGE.]KEY=VALUE, got {assignment!r}.")
        stage, dot, key = selector.rpartition(".")
        options = defaults
        if dot:
            if stage not in stages:
                raise ValueError(
                    f"SLURM override stage {stage!r} is outside the plan "
                    f"({', '.join(stages)})."
                )
            options = per_stage.setdefault(stage, {})
        if key in options:
            raise ValueError(f"Duplicate --slurm override for {selector!r}.")
        options[key] = _slurm_value(key, value)

    return {
        stage: {**defaults, **per_stage.get(stage, {})}
        for stage in stages
        if defaults or stage in per_stage
    }
