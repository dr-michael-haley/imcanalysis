"""Resource discovery shared by interactive and managed NapariSBT workers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def process_cpu_limit(
    *,
    environ: Mapping[str, str] | None = None,
    affinity_count: int | None = None,
    cpu_count: int | None = None,
) -> tuple[int, str]:
    """Return the CPUs the current process may safely use and their source."""

    environment = os.environ if environ is None else environ
    in_slurm = bool(environment.get("SLURM_JOB_ID"))
    if affinity_count is None and hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            affinity_count = None
    affinity_count = _positive_int(affinity_count)
    cpus_per_task = _positive_int(environment.get("SLURM_CPUS_PER_TASK"))
    tasks = _positive_int(environment.get("SLURM_NTASKS"))
    declared_slurm_cpus = (
        cpus_per_task * tasks
        if cpus_per_task is not None and tasks is not None
        else cpus_per_task or tasks
    )
    if in_slurm and affinity_count is not None and declared_slurm_cpus is not None:
        return (
            min(affinity_count, declared_slurm_cpus),
            "Slurm CPU request and process affinity",
        )
    if in_slurm and affinity_count is not None:
        return affinity_count, "Slurm process CPU affinity"
    if in_slurm and declared_slurm_cpus is not None:
        return declared_slurm_cpus, "Slurm CPU request"
    if cpus_per_task is not None:
        return cpus_per_task, "SLURM_CPUS_PER_TASK"
    if tasks is not None:
        return tasks, "SLURM_NTASKS"
    if affinity_count is not None:
        return affinity_count, "process CPU affinity"
    available = _positive_int(cpu_count if cpu_count is not None else os.cpu_count()) or 1
    return available, "os.cpu_count"


@dataclass(frozen=True)
class WorkerResolution:
    requested: int
    effective: int
    cpu_limit: int
    limit_source: str

    @property
    def adjusted(self) -> bool:
        return self.effective != self.requested

    @property
    def message(self) -> str:
        if not self.adjusted:
            return (
                f"Using {self.effective} feature worker(s); the current process limit "
                f"is {self.cpu_limit} from {self.limit_source}."
            )
        return (
            f"Requested {self.requested} feature workers, but the current allocation "
            f"permits {self.cpu_limit} from {self.limit_source}; using "
            f"{self.effective}."
        )


def resolve_worker_count(
    requested: int | None,
    *,
    environ: Mapping[str, str] | None = None,
    affinity_count: int | None = None,
    cpu_count: int | None = None,
    default_maximum: int = 8,
) -> WorkerResolution:
    """Clamp feature workers to the CPUs available to this process."""

    limit, source = process_cpu_limit(
        environ=environ,
        affinity_count=affinity_count,
        cpu_count=cpu_count,
    )
    selected = _positive_int(requested)
    if selected is None:
        selected = min(limit, max(1, int(default_maximum)))
    return WorkerResolution(
        requested=selected,
        effective=min(selected, limit),
        cpu_limit=limit,
        limit_source=source,
    )


__all__ = ["WorkerResolution", "process_cpu_limit", "resolve_worker_count"]
