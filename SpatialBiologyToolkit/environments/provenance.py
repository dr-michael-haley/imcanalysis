"""Snapshot repository and live environment provenance into stage outputs."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from SpatialBiologyToolkit.pipeline.manifests import read_model, utc_now, write_yaml
from SpatialBiologyToolkit.reporting.models import EnvironmentReportReference, StageManifest

from .models import (
    EnvironmentProvenanceManifest,
    EnvironmentRuntimeRecord,
    SpecificationFileRecord,
)
from .registry import load_environment_registry, toolkit_root
from .runtime import find_conda_executable, inspect_environment
from .specification import atomic_write_text, environment_paths


ENVIRONMENT_DIRECTORY = "environment"
ENVIRONMENT_MANIFEST = "environment_manifest.yaml"
INSTALLED_SNAPSHOT = "installed.snapshot.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_directory(output: Path, key: str, primary: bool) -> Path:
    return output / ENVIRONMENT_DIRECTORY if primary else output / ENVIRONMENT_DIRECTORY / "additional" / key


def snapshot_stage_environment_specifications(
    *,
    stage: str,
    output_directory: Path,
    repository_root: Path | None = None,
    environment_keys: list[str] | None = None,
    default_environment_keys: list[str] | None = None,
) -> EnvironmentReportReference | None:
    root = toolkit_root(repository_root)
    registry = load_environment_registry(root)
    if environment_keys is None:
        exported_keys = os.environ.get("SBT_ENVIRONMENT_KEYS", "")
        environment_keys = [
            item.strip() for item in exported_keys.split(",") if item.strip()
        ] or None
    keys = list(environment_keys or registry.stage_environments.get(stage, []))
    if not keys:
        return None
    defaults = list(
        default_environment_keys or registry.stage_environments.get(stage, [])
    )
    reference: EnvironmentReportReference | None = None
    for index, key in enumerate(keys):
        definition = registry.environments[key]
        destination = _snapshot_directory(output_directory, key, index == 0)
        destination.mkdir(parents=True, exist_ok=True)
        specification: dict[str, SpecificationFileRecord] = {}
        if definition.managed:
            paths = environment_paths(root, root / "HPC_env_files/environments.yaml", definition)
            for label, source in (
                ("environment_yml", paths.environment_yml),
                ("lockfile", paths.lockfile),
                ("pip_extras", paths.pip_extras),
            ):
                if source is None or not source.is_file():
                    continue
                copied = destination / source.name
                shutil.copy2(source, copied, follow_symlinks=True)
                specification[label] = SpecificationFileRecord(
                    path=copied.relative_to(output_directory), sha256=_sha256(copied)
                )
        manifest_path = destination / ENVIRONMENT_MANIFEST
        manifest = EnvironmentProvenanceManifest(
            environment_key=key,
            conda_name=definition.conda_name,
            platform=definition.platform,
            specification=specification,
            captured_at=utc_now(),
            execution_id=os.environ.get("SBT_EXECUTION_ID"),
            technical_run_id=os.environ.get("SBT_TECHNICAL_RUN_ID"),
            slurm_job_id=os.environ.get("SBT_SLURM_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
        )
        write_yaml(manifest_path, manifest)
        if index == 0:
            reference = EnvironmentReportReference(
                key=key,
                conda_name=definition.conda_name,
                manifest=manifest_path.relative_to(output_directory),
                specification_snapshot=destination.relative_to(output_directory),
                additional_keys=keys[1:],
                overridden=keys != defaults,
                default_keys=defaults,
            )
    return reference


def capture_stage_environment_runtime(
    *,
    stage: str,
    output_directory: Path,
    repository_root: Path | None = None,
    environment_keys: list[str] | None = None,
    default_environment_keys: list[str] | None = None,
) -> EnvironmentReportReference | None:
    root = toolkit_root(repository_root)
    registry = load_environment_registry(root)
    if environment_keys is None:
        exported_keys = os.environ.get("SBT_ENVIRONMENT_KEYS", "")
        environment_keys = [
            item.strip() for item in exported_keys.split(",") if item.strip()
        ] or None
    keys = list(environment_keys or registry.stage_environments.get(stage, []))
    if not keys:
        return None
    reference = snapshot_stage_environment_specifications(
        stage=stage,
        output_directory=output_directory,
        repository_root=root,
        environment_keys=keys,
        default_environment_keys=default_environment_keys,
    )
    conda = find_conda_executable()
    if not conda:
        return reference
    for index, key in enumerate(keys):
        definition = registry.environments[key]
        destination = _snapshot_directory(output_directory, key, index == 0)
        try:
            snapshot = inspect_environment(
                key=key,
                definition=definition,
                repository_root=root,
                conda=conda,
            )
        except RuntimeError:
            continue
        drift_state = "unknown"
        try:
            from .manager import EnvironmentManager

            drift_state, _ = EnvironmentManager(
                root, conda_executable=conda
            ).compare_snapshot(key, snapshot)
        except (OSError, RuntimeError, ValueError):
            drift_state = "error"
        installed = destination / INSTALLED_SNAPSHOT
        atomic_write_text(installed, snapshot.model_dump_json(indent=2) + "\n")
        manifest_path = destination / ENVIRONMENT_MANIFEST
        try:
            manifest = read_model(manifest_path, EnvironmentProvenanceManifest)
        except (OSError, ValueError):
            manifest = EnvironmentProvenanceManifest(
                environment_key=key,
                conda_name=definition.conda_name,
                platform=definition.platform,
                captured_at=utc_now(),
            )
        manifest.runtime = EnvironmentRuntimeRecord(
            python_version=snapshot.python_version,
            conda_prefix=snapshot.conda_prefix,
            toolkit_editable=snapshot.toolkit.editable,
            toolkit_git_commit=snapshot.toolkit.installed_git_commit
            or snapshot.toolkit.checkout_git_commit,
            toolkit_dirty=snapshot.toolkit.checkout_dirty,
            drift=drift_state,
        )
        manifest.installed_snapshot = installed.relative_to(output_directory)
        manifest.captured_at = snapshot.captured_at
        manifest.execution_id = snapshot.execution_id
        manifest.technical_run_id = snapshot.technical_run_id
        manifest.slurm_job_id = snapshot.slurm_job_id
        write_yaml(manifest_path, manifest)
    return reference


def attach_environment_reference(manifest_path: Path, reference: EnvironmentReportReference | None) -> None:
    if reference is None or not manifest_path.is_file():
        return
    manifest = read_model(manifest_path, StageManifest)
    manifest.environment = reference
    write_yaml(manifest_path, manifest)


__all__ = [
    "attach_environment_reference",
    "capture_stage_environment_runtime",
    "snapshot_stage_environment_specifications",
]
