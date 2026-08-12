"""Safe subprocess-based inspection of fixed Conda environments."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import (
    CondaEnvironmentRecord,
    CondaPackageRecord,
    EnvironmentDefinition,
    ObservedEnvironmentSnapshot,
    PipPackageRecord,
    ToolkitOverlayRecord,
)


if TYPE_CHECKING:
    Runner = Callable[..., subprocess.CompletedProcess[str]]
else:
    # ``collections.abc.Callable`` and ``CompletedProcess`` became subscriptable
    # in Python 3.9. The legacy denoising runtime is Python 3.8, so keep the
    # runtime alias unsubscripted while retaining the precise static type.
    Runner = Callable
TOOLKIT_NAMES = {"spatialbiologytoolkit", "spatial-biology-toolkit"}


def normalize_package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def command_text(command: Sequence[str]) -> str:
    return " ".join(_quote_argument(part) for part in command)


def _quote_argument(value: str) -> str:
    if not value or any(character.isspace() or character in "'\"" for character in value):
        return repr(value)
    return value


def find_executable(name: str) -> str | None:
    return shutil.which(name)


def find_conda_executable() -> str | None:
    """Find Conda even before a SLURM wrapper sources ``conda.sh``."""
    discovered = find_executable("conda")
    if discovered:
        return discovered
    configured = os.environ.get("CONDA_EXE")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path.home() / "miniconda3" / "bin" / "conda",
        Path.home() / "anaconda3" / "bin" / "conda",
        Path("/opt/conda/bin/conda"),
    ]
    return next(
        (str(path) for path in candidates if path is not None and path.is_file()),
        None,
    )


def run_checked(
    command: Sequence[str],
    *,
    runner: Runner = subprocess.run,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = runner(
            list(command),
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Could not run {command_text(command)}: {exc}") from exc
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {command_text(command)}\n{detail}"
        )
    return completed


def conda_environment_names(
    conda: str,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Path]:
    return {
        item.name: item.prefix
        for item in conda_environment_records(conda, runner=runner)
    }


def conda_environment_records(
    conda: str,
    *,
    runner: Runner = subprocess.run,
) -> list[CondaEnvironmentRecord]:
    completed = run_checked([conda, "env", "list", "--json"], runner=runner)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Conda returned invalid JSON for `env list --json`.") from exc
    raw_environments = payload.get("envs", [])
    if not isinstance(raw_environments, list):
        raise RuntimeError("Conda returned an invalid environment inventory.")
    raw_root = payload.get("root_prefix")
    root_prefix = (
        Path(str(raw_root)).expanduser().resolve(strict=False) if raw_root else None
    )
    platform = str(payload.get("platform") or "linux-64")
    records: list[CondaEnvironmentRecord] = []
    seen_prefixes: set[str] = set()
    for raw_prefix in raw_environments:
        prefix = Path(raw_prefix).expanduser().resolve(strict=False)
        normalized_prefix = os.path.normcase(str(prefix))
        if normalized_prefix in seen_prefixes:
            continue
        seen_prefixes.add(normalized_prefix)
        is_base = bool(
            root_prefix
            and normalized_prefix == os.path.normcase(str(root_prefix))
        )
        records.append(
            CondaEnvironmentRecord(
                name="base" if is_base else prefix.name,
                prefix=prefix,
                platform=platform,
                is_base=is_base,
            )
        )
    return records


def environment_exists(
    conda: str,
    conda_name: str,
    *,
    runner: Runner = subprocess.run,
) -> bool:
    return conda_name in conda_environment_names(conda, runner=runner)


def _json_command(
    command: Sequence[str],
    *,
    runner: Runner,
) -> Any:
    completed = run_checked(command, runner=runner)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Command returned invalid JSON: {command_text(command)}") from exc


def _git(root: Path, *arguments: str, runner: Runner) -> str | None:
    try:
        completed = runner(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_state(root: Path, runner: Runner) -> tuple[str | None, bool | None]:
    commit = _git(root, "rev-parse", "HEAD", runner=runner)
    status = _git(root, "status", "--porcelain", runner=runner)
    return commit, (bool(status) if status is not None else None)


def _pip_records(
    all_packages: list[dict[str, Any]],
    editable_packages: list[dict[str, Any]],
    freeze_lines: list[str],
) -> tuple[list[PipPackageRecord], list[PipPackageRecord], list[str]]:
    editable_locations = {
        normalize_package_name(str(item.get("name", ""))): item.get("editable_project_location")
        or item.get("location")
        for item in editable_packages
    }
    requirements: dict[str, str] = {}
    review: list[str] = []
    for line in freeze_lines:
        candidate = line.strip()
        if not candidate or candidate.startswith("#"):
            continue
        source_type = "index"
        name_part = candidate.split("==", 1)[0]
        if candidate.startswith("-e "):
            source_type = "editable"
            name_part = candidate[3:].split("#egg=", 1)[-1]
        elif " @ git+" in candidate or candidate.startswith("git+"):
            source_type = "vcs"
            name_part = candidate.split(" @ ", 1)[0]
        elif " @ file:" in candidate or candidate.startswith(("file:", "/", ".")):
            source_type = "local"
            name_part = candidate.split(" @ ", 1)[0]
        normalized = normalize_package_name(name_part)
        requirements[normalized] = candidate
        if source_type != "index" and normalized not in TOOLKIT_NAMES:
            review.append(candidate)

    records: list[PipPackageRecord] = []
    editable_records: list[PipPackageRecord] = []
    for item in all_packages:
        name = str(item.get("name", ""))
        normalized = normalize_package_name(name)
        requirement = requirements.get(normalized)
        location = editable_locations.get(normalized)
        source_type = "editable" if normalized in editable_locations else "index"
        if requirement:
            if "git+" in requirement:
                source_type = "vcs"
            elif "file:" in requirement or requirement.startswith(("/", ".")):
                source_type = "local"
        record = PipPackageRecord(
            name=name,
            version=str(item.get("version", "")),
            editable=normalized in editable_locations,
            location=str(location) if location else None,
            requirement=requirement,
            source_type=source_type,
        )
        records.append(record)
        if record.editable:
            editable_records.append(record)
    records.sort(key=lambda item: normalize_package_name(item.name))
    editable_records.sort(key=lambda item: normalize_package_name(item.name))
    return records, editable_records, sorted(set(review), key=str.casefold)


def inspect_environment(
    *,
    key: str,
    definition: EnvironmentDefinition,
    repository_root: Path,
    conda: str,
    conda_prefix: Path | None = None,
    runner: Runner = subprocess.run,
    now: datetime | None = None,
    execution_environment: dict[str, str] | None = None,
) -> ObservedEnvironmentSnapshot:
    name = definition.conda_name
    selector = (
        ["--prefix", str(conda_prefix)]
        if conda_prefix is not None
        else ["--name", name]
    )
    conda_payload = _json_command(
        [conda, "list", *selector, "--json"], runner=runner
    )
    conda_records = [
        CondaPackageRecord(
            name=str(item.get("name", "")),
            version=str(item.get("version", "")),
            build=str(item.get("build_string") or item.get("build") or ""),
            channel=str(item.get("channel") or item.get("base_url") or ""),
            manager=("pip" if str(item.get("channel", "")).casefold() == "pypi" else "conda"),
        )
        for item in conda_payload
    ]
    conda_records.sort(key=lambda item: normalize_package_name(item.name))
    observed_platform = next(
        (
            str(item.get("platform"))
            for item in conda_payload
            if item.get("platform")
        ),
        definition.platform,
    )

    prefix_script = (
        "import importlib.util,json,platform,sys;"
        "s=importlib.util.find_spec('SpatialBiologyToolkit');"
        "print(json.dumps({'prefix':sys.prefix,'python':platform.python_version(),"
        "'toolkit_origin':getattr(s,'origin',None)}))"
    )
    review: list[str] = []
    runtime: dict[str, Any] = {}
    pip_all: list[dict[str, Any]] = []
    try:
        runtime_payload = _json_command(
            [conda, "run", *selector, "python", "-c", prefix_script],
            runner=runner,
        )
        if not isinstance(runtime_payload, dict):
            raise RuntimeError("Python runtime inspection returned invalid JSON data.")
        runtime = runtime_payload
        pip_payload = _json_command(
            [
                conda,
                "run",
                *selector,
                "python",
                "-m",
                "pip",
                "list",
                "--format=json",
            ],
            runner=runner,
        )
        if isinstance(pip_payload, list):
            pip_all = pip_payload
    except RuntimeError as exc:
        review.append(f"Python/pip inspection unavailable: {exc}")
    try:
        pip_editable = _json_command(
            [
                conda,
                "run",
                *selector,
                "python",
                "-m",
                "pip",
                "list",
                "--editable",
                "--format=json",
            ],
            runner=runner,
        )
    except RuntimeError:
        pip_editable = []
    try:
        freeze = run_checked(
            [conda, "run", *selector, "python", "-m", "pip", "freeze"],
            runner=runner,
        ).stdout.splitlines()
    except RuntimeError as exc:
        freeze = []
        message = f"pip freeze unavailable: {exc}"
        if message not in review:
            review.append(message)
    pip_records, editable_records, pip_review = _pip_records(
        pip_all, pip_editable if isinstance(pip_editable, list) else [], freeze
    )
    review.extend(pip_review)

    toolkit_package = next(
        (item for item in pip_records if normalize_package_name(item.name) in TOOLKIT_NAMES),
        None,
    )
    toolkit_origin = runtime.get("toolkit_origin")
    installed_root: Path | None = None
    if toolkit_origin:
        origin = Path(toolkit_origin).expanduser().resolve(strict=False)
        installed_root = origin.parent.parent if origin.name == "__init__.py" else origin.parent
    elif toolkit_package and toolkit_package.location:
        installed_root = Path(toolkit_package.location).expanduser().resolve(strict=False)
    checkout_commit, checkout_dirty = _git_state(repository_root, runner)
    installed_commit = (
        _git(installed_root, "rev-parse", "HEAD", runner=runner)
        if installed_root
        else None
    )
    toolkit = ToolkitOverlayRecord(
        installed=toolkit_package is not None or bool(toolkit_origin),
        editable=bool(toolkit_package and toolkit_package.editable),
        repository_path=installed_root,
        repository_matches=(
            installed_root.resolve(strict=False) == repository_root.resolve(strict=False)
            if installed_root
            else None
        ),
        installed_git_commit=installed_commit,
        checkout_git_commit=checkout_commit,
        checkout_dirty=checkout_dirty,
    )
    try:
        conda_version = run_checked([conda, "--version"], runner=runner).stdout.strip()
    except RuntimeError:
        conda_version = None
    environment = execution_environment or os.environ
    return ObservedEnvironmentSnapshot(
        environment_key=key,
        environment_name=name,
        captured_at=now or datetime.now(timezone.utc),
        platform=observed_platform,
        conda_prefix=(
            Path(runtime["prefix"])
            if runtime.get("prefix")
            else conda_prefix
        ),
        python_version=runtime.get("python"),
        conda_version=conda_version,
        conda_packages=conda_records,
        pip_packages=pip_records,
        editable_packages=editable_records,
        review_requirements=review,
        toolkit=toolkit,
        repository_git_commit=checkout_commit,
        slurm_job_id=environment.get("SLURM_JOB_ID") or environment.get("SBT_SLURM_JOB_ID"),
        execution_id=environment.get("SBT_EXECUTION_ID"),
        technical_run_id=environment.get("SBT_TECHNICAL_RUN_ID"),
    )


def tail(value: str, limit: int = 4000) -> str:
    return value if len(value) <= limit else value[-limit:]


__all__ = [
    "Runner",
    "TOOLKIT_NAMES",
    "command_text",
    "conda_environment_names",
    "conda_environment_records",
    "environment_exists",
    "find_conda_executable",
    "find_executable",
    "inspect_environment",
    "normalize_package_name",
    "run_checked",
    "tail",
]
