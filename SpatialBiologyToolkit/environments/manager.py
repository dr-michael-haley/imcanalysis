"""High-level environment validation, comparison, capture, sync, and testing."""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml  # type: ignore[import-untyped]

from .models import (
    CapturePlan,
    CondaEnvironmentRecord,
    DoctorCheck,
    DoctorReport,
    DriftItem,
    EnvironmentComparison,
    EnvironmentDefinition,
    EnvironmentCaptureTarget,
    EnvironmentPaths,
    EnvironmentRegistry,
    EnvironmentSummary,
    EnvironmentTestReport,
    OverlayRefreshReport,
    OverlayRefreshResult,
    SmokeTestResult,
    SpecificationValidation,
    SyncPlan,
)
from .registry import (
    REGISTRY_RELATIVE_PATH,
    associated_stages,
    load_environment_registry,
    resolve_environment,
    toolkit_root,
)
from .runtime import (
    Runner,
    command_text,
    conda_environment_names,
    conda_environment_records,
    find_conda_executable,
    find_executable,
    inspect_environment,
    normalize_package_name,
    run_checked,
    tail,
)
from .specification import (
    atomic_write_bytes,
    atomic_write_text,
    declared_conda_requirements,
    declared_pip_requirements,
    environment_paths,
    load_environment_yml,
    locked_conda_packages,
    satisfies_constraint,
    validate_specification,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _state_root() -> Path:
    explicit = os.environ.get("SBT_STATE_HOME")
    if explicit:
        return Path(explicit).expanduser().resolve(strict=False)
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        return (Path(xdg).expanduser() / "sbt").resolve(strict=False)
    return (Path.home() / ".local" / "state" / "sbt").resolve(strict=False)


def _normalized_map(records: list[Any]) -> dict[str, Any]:
    return {normalize_package_name(item.name): item for item in records}


def _safe_capture_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "environment"


def _capture_directory_name(name: str, prefix: Path, *, disambiguate: bool) -> str:
    safe_name = _safe_capture_name(name)
    if not disambiguate:
        return safe_name
    digest = hashlib.sha256(os.path.normcase(str(prefix)).encode("utf-8")).hexdigest()[:8]
    return f"{safe_name}-{digest}"


class EnvironmentManager:
    """Operate on registered and discovered Conda environments without activation."""

    def __init__(
        self,
        repository_root: str | Path | None = None,
        *,
        registry_path: str | Path | None = None,
        runner: Runner = subprocess.run,
        conda_executable: str | None = None,
        state_root: str | Path | None = None,
    ) -> None:
        self.repository_root = toolkit_root(repository_root)
        self.registry_path = (
            Path(registry_path).expanduser().resolve(strict=False)
            if registry_path
            else (self.repository_root / REGISTRY_RELATIVE_PATH).resolve(strict=False)
        )
        self.registry: EnvironmentRegistry = load_environment_registry(
            self.repository_root, registry_path=self.registry_path
        )
        self.runner = runner
        self.conda = conda_executable or find_conda_executable()
        self.conda_lock_command = (
            [self.conda, "run", "-n", "base", "conda-lock"]
            if self.conda
            else None
        )
        self.state_root = (
            Path(state_root).expanduser().resolve(strict=False)
            if state_root
            else _state_root()
        )

    def resolve(self, selector: str) -> tuple[str, EnvironmentDefinition]:
        return resolve_environment(self.registry, selector)

    def select(
        self, selector: str | None = None, *, all_environments: bool = False
    ) -> list[tuple[str, EnvironmentDefinition]]:
        if all_environments:
            if selector:
                raise ValueError("Choose an environment or --all, not both.")
            return list(self.registry.environments.items())
        if not selector:
            raise ValueError("Provide an environment or use --all.")
        return [self.resolve(selector)]

    def paths(self, key: str, definition: EnvironmentDefinition) -> EnvironmentPaths:
        return environment_paths(
            self.repository_root, self.registry_path, definition
        )

    def validate(self, selector: str) -> SpecificationValidation:
        key, definition = self.resolve(selector)
        return validate_specification(
            key=key,
            definition=definition,
            repository_root=self.repository_root,
            registry_path=self.registry_path,
        )

    def validate_all(self) -> list[SpecificationValidation]:
        return [self.validate(key) for key in self.registry.environments]

    def _environment_inventory(self) -> dict[str, Path]:
        if not self.conda:
            raise RuntimeError("Conda executable was not found on PATH.")
        return conda_environment_names(self.conda, runner=self.runner)

    def _environment_records(self) -> list[CondaEnvironmentRecord]:
        if not self.conda:
            raise RuntimeError("Conda executable was not found on PATH.")
        return conda_environment_records(self.conda, runner=self.runner)

    def discover_capture_targets(self) -> list[EnvironmentCaptureTarget]:
        """Return every distinct Conda prefix, enriched with SBT registry metadata."""
        records = self._environment_records()
        records_by_name: dict[str, list[CondaEnvironmentRecord]] = {}
        records_by_directory: dict[str, list[CondaEnvironmentRecord]] = {}
        for record in records:
            records_by_name.setdefault(record.name.casefold(), []).append(record)
            records_by_directory.setdefault(
                _safe_capture_name(record.name).casefold(), []
            ).append(record)
        registered_by_name = {
            definition.conda_name.casefold(): (key, definition)
            for key, definition in self.registry.environments.items()
        }
        targets: list[EnvironmentCaptureTarget] = []
        for record in sorted(records, key=lambda item: str(item.prefix).casefold()):
            same_name = records_by_name[record.name.casefold()]
            registered = registered_by_name.get(record.name.casefold())
            is_registered = registered is not None and len(same_name) == 1
            disambiguate = (
                len(same_name) > 1
                or len(records_by_directory[_safe_capture_name(record.name).casefold()]) > 1
            )
            directory_name = _capture_directory_name(
                record.name, record.prefix, disambiguate=disambiguate
            )
            if is_registered:
                assert registered is not None
                environment_key, definition = registered
                conda_name = definition.conda_name
            else:
                suffix = f"@{directory_name.rsplit('-', 1)[-1]}" if disambiguate else ""
                environment_key = f"conda:{record.name}{suffix}"
                conda_name = record.name
            targets.append(
                EnvironmentCaptureTarget(
                    environment_key=environment_key,
                    conda_name=conda_name,
                    conda_prefix=record.prefix,
                    platform=record.platform,
                    registered=is_registered,
                    capture_directory_name=directory_name,
                )
            )
        return targets

    def list_environments(self, *, compare: bool = False) -> list[EnvironmentSummary]:
        inventory: dict[str, Path] = {}
        if self.conda:
            try:
                inventory = self._environment_inventory()
            except RuntimeError:
                inventory = {}
        rows: list[EnvironmentSummary] = []
        for key, definition in self.registry.environments.items():
            exists = definition.conda_name in inventory if self.conda else None
            drift = "unknown"
            if compare and exists:
                comparison = self.compare(key)
                drift = (
                    "clean"
                    if comparison.result == "clean"
                    else "drift"
                    if comparison.result == "drift"
                    else "error"
                )
            rows.append(
                EnvironmentSummary(
                    key=key,
                    conda_name=definition.conda_name,
                    managed=definition.managed,
                    exists=exists,
                    drift=drift,
                    stages=associated_stages(self.registry, key),
                )
            )
        return rows

    def required_for_stages(
        self,
        stages: Iterable[str],
        *,
        environment_overrides: dict[str, str] | None = None,
    ) -> list[EnvironmentSummary]:
        """Return the live availability of environments used by selected stages.

        Environment keys are de-duplicated in first-use order so a multi-stage
        workflow checks Conda only once and presents each required environment
        once. Stages without a registered Conda environment (for example the
        shell-only debug stage) do not require Conda to be available.
        """

        uses: dict[str, list[str]] = {}
        overrides = environment_overrides or {}
        for stage in stages:
            keys = (
                [overrides[stage]]
                if stage in overrides
                else self.registry.stage_environments.get(stage, [])
            )
            for key in keys:
                stage_uses = uses.setdefault(key, [])
                if stage not in stage_uses:
                    stage_uses.append(stage)
        if not uses:
            return []

        inventory = self._environment_inventory()
        return [
            EnvironmentSummary(
                key=key,
                conda_name=self.registry.environments[key].conda_name,
                managed=self.registry.environments[key].managed,
                exists=self.registry.environments[key].conda_name in inventory,
                stages=stage_names,
            )
            for key, stage_names in uses.items()
        ]

    def show(self, selector: str) -> dict[str, Any]:
        key, definition = self.resolve(selector)
        inventory: dict[str, Path] = {}
        if self.conda:
            try:
                inventory = self._environment_inventory()
            except RuntimeError:
                pass
        paths = self.paths(key, definition)
        snapshots: list[Path] = []
        for category in ("environments", "environment_history"):
            directory = self.state_root / category / definition.conda_name
            if directory.is_dir():
                snapshots.extend(directory.glob("*.json"))
        capture_root = self.state_root / "captures" / definition.conda_name
        if capture_root.is_dir():
            snapshots.extend(capture_root.glob("*/environment.snapshot.json"))
        snapshots.sort(key=lambda path: path.stat().st_mtime)
        return {
            "key": key,
            "conda_name": definition.conda_name,
            "managed": definition.managed,
            "platform": definition.platform,
            "toolkit_overlay": definition.toolkit_overlay,
            "stages": associated_stages(self.registry, key),
            "smoke_tests": definition.smoke_tests,
            "notes": definition.notes,
            "paths": paths.model_dump(mode="json"),
            "exists": definition.conda_name in inventory if self.conda else None,
            "prefix": str(inventory.get(definition.conda_name, "")) or None,
            "last_observation": str(snapshots[-1]) if snapshots else None,
        }

    def doctor(self) -> DoctorReport:
        checks: list[DoctorCheck] = []
        checks.append(
            DoctorCheck(
                name="repository_root",
                status="ok" if self.repository_root.is_dir() else "error",
                detail=str(self.repository_root),
            )
        )
        checks.append(
            DoctorCheck(
                name="environment_registry",
                status="ok",
                detail=f"{len(self.registry.environments)} environments: {self.registry_path}",
            )
        )
        if self.conda:
            try:
                version = run_checked([self.conda, "--version"], runner=self.runner)
                checks.append(
                    DoctorCheck(name="conda", status="ok", detail=version.stdout.strip())
                )
                inventory = self._environment_inventory()
                probe = next(
                    (
                        definition.conda_name
                        for definition in self.registry.environments.values()
                        if definition.conda_name in inventory
                    ),
                    None,
                )
                if probe:
                    pip_version = run_checked(
                        [
                            self.conda,
                            "run",
                            "-n",
                            probe,
                            "python",
                            "-m",
                            "pip",
                            "--version",
                        ],
                        runner=self.runner,
                    )
                    checks.append(
                        DoctorCheck(
                            name="pip_through_conda_run",
                            status="ok",
                            detail=f"{probe}: {pip_version.stdout.strip()}",
                        )
                    )
                else:
                    checks.append(
                        DoctorCheck(
                            name="pip_through_conda_run",
                            status="warning",
                            detail="No registry environment exists to probe.",
                        )
                    )
            except RuntimeError as exc:
                checks.append(DoctorCheck(name="conda", status="error", detail=str(exc)))
        else:
            checks.append(
                DoctorCheck(name="conda", status="error", detail="Conda was not found on PATH.")
            )
        if self.conda_lock_command:
            try:
                version = run_checked(
                    [*self.conda_lock_command, "--version"], runner=self.runner
                )
                checks.append(
                    DoctorCheck(
                        name="conda_lock",
                        status="ok",
                        detail=f"Conda base: {version.stdout.strip()}",
                    )
                )
            except RuntimeError as exc:
                checks.append(
                    DoctorCheck(name="conda_lock", status="error", detail=str(exc))
                )
        else:
            checks.append(
                DoctorCheck(
                    name="conda_lock",
                    status="error",
                    detail="Conda is unavailable, so conda-lock cannot be run from base.",
                )
            )

        stage_names = set(self.registry.stage_environments)
        try:
            from SpatialBiologyToolkit.pipeline.registry import STAGES

            registered = {stage.name for stage in STAGES}
            unknown = sorted(stage_names - registered)
            unmapped = sorted(
                stage.name
                for stage in STAGES
                if stage.runnable and stage.name not in stage_names and stage.name not in {"zipqc", "debug"}
            )
            status = "ok" if not unknown and not unmapped else "error"
            checks.append(
                DoctorCheck(
                    name="stage_environment_mapping",
                    status=status,
                    detail=f"unknown={unknown or 'none'}, unmapped={unmapped or 'none'}",
                )
            )
        except Exception as exc:
            checks.append(
                DoctorCheck(
                    name="stage_environment_mapping", status="error", detail=str(exc)
                )
            )

        validations = self.validate_all()
        invalid = [item.environment_key for item in validations if not item.valid]
        checks.append(
            DoctorCheck(
                name="specifications",
                status="error" if invalid else "ok",
                detail=f"invalid={invalid or 'none'}",
            )
        )
        platforms = sorted(
            {definition.platform for definition in self.registry.environments.values()}
        )
        checks.append(
            DoctorCheck(
                name="target_platforms",
                status="ok" if platforms == ["linux-64"] else "warning",
                detail=", ".join(platforms),
            )
        )
        checks.append(
            DoctorCheck(
                name="git",
                status="ok" if find_executable("git") else "warning",
                detail=find_executable("git") or "Git was not found on PATH.",
            )
        )
        stale_files = []
        candidates = [Path("Makefile")]
        candidates.extend(
            path.relative_to(self.repository_root)
            for directory in ("install", "Bash_scripts")
            for path in (self.repository_root / directory).glob("*")
            if path.is_file()
        )
        for relative in candidates:
            path = self.repository_root / relative
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(definition.conda_name in text for definition in self.registry.environments.values()):
                stale_files.append(relative.as_posix())
        checks.append(
            DoctorCheck(
                name="legacy_environment_mappings",
                status="warning" if stale_files else "ok",
                detail=", ".join(stale_files) if stale_files else "No duplicate Make/install mapping.",
            )
        )
        state_parent = self.state_root if self.state_root.exists() else self.state_root.parent
        checks.append(
            DoctorCheck(
                name="state_directory",
                status="ok" if os.access(state_parent, os.W_OK) else "warning",
                detail=str(self.state_root),
            )
        )
        return DoctorReport(
            healthy=not any(check.status == "error" for check in checks), checks=checks
        )

    def compare(self, selector: str) -> EnvironmentComparison:
        key, definition = self.resolve(selector)
        specification = self.validate(key)
        if not definition.managed:
            return EnvironmentComparison(
                environment_key=key,
                conda_name=definition.conda_name,
                compared_at=_utc_now(),
                exists=False,
                completed=False,
                result="error",
                specification=specification,
                error="Externally managed environment has no authoritative repository lock.",
            )
        if not specification.valid:
            return EnvironmentComparison(
                environment_key=key,
                conda_name=definition.conda_name,
                compared_at=_utc_now(),
                exists=False,
                completed=False,
                result="error",
                specification=specification,
                error="Environment specification is invalid.",
            )
        if not self.conda:
            return EnvironmentComparison(
                environment_key=key,
                conda_name=definition.conda_name,
                compared_at=_utc_now(),
                exists=False,
                completed=False,
                result="error",
                specification=specification,
                error="Conda executable was not found on PATH.",
            )
        try:
            inventory = self._environment_inventory()
            if definition.conda_name not in inventory:
                return EnvironmentComparison(
                    environment_key=key,
                    conda_name=definition.conda_name,
                    compared_at=_utc_now(),
                    exists=False,
                    completed=False,
                    result="missing",
                    specification=specification,
                    error=f"Conda environment {definition.conda_name!r} does not exist.",
                )
            snapshot = inspect_environment(
                key=key,
                definition=definition,
                repository_root=self.repository_root,
                conda=self.conda,
                runner=self.runner,
            )
            drift = self._compare_snapshot(definition, specification, snapshot)
        except RuntimeError as exc:
            return EnvironmentComparison(
                environment_key=key,
                conda_name=definition.conda_name,
                compared_at=_utc_now(),
                exists=True,
                completed=False,
                result="error",
                specification=specification,
                error=str(exc),
            )
        return EnvironmentComparison(
            environment_key=key,
            conda_name=definition.conda_name,
            compared_at=_utc_now(),
            exists=True,
            completed=True,
            result="drift" if any(item.material for item in drift) else "clean",
            specification=specification,
            drift=drift,
            toolkit=snapshot.toolkit,
            observed_snapshot=snapshot,
        )

    def compare_snapshot(self, selector: str, snapshot) -> tuple[str, list[DriftItem]]:
        """Compare an already captured observation without running more commands."""
        key, definition = self.resolve(selector)
        specification = self.validate(key)
        if not definition.managed or not specification.valid:
            return "unknown", []
        drift = self._compare_snapshot(definition, specification, snapshot)
        return ("drift" if any(item.material for item in drift) else "clean"), drift

    def _compare_snapshot(self, definition, specification, snapshot) -> list[DriftItem]:
        paths = specification.paths
        assert paths.environment_yml and paths.lockfile and paths.pip_extras
        drift: list[DriftItem] = []
        if snapshot.platform != definition.platform:
            drift.append(
                DriftItem(
                    layer="specification",
                    kind="platform",
                    expected=definition.platform,
                    actual=snapshot.platform,
                    message=(
                        f"Live environment platform {snapshot.platform!r} does not match "
                        f"registry target {definition.platform!r}."
                    ),
                )
            )
        live_conda = {
            key: value
            for key, value in _normalized_map(snapshot.conda_packages).items()
            if value.manager == "conda"
        }
        for name, constraint in declared_conda_requirements(paths.environment_yml).items():
            live = live_conda.get(name)
            if not live:
                drift.append(
                    DriftItem(
                        layer="conda-direct",
                        kind="missing",
                        package=name,
                        expected=constraint or "declared",
                        message=f"Declared Conda package {name} is missing.",
                    )
                )
            elif not satisfies_constraint(live.version, constraint):
                drift.append(
                    DriftItem(
                        layer="conda-direct",
                        kind="version",
                        package=name,
                        expected=constraint,
                        actual=live.version,
                        message=f"{name}: expected {constraint}, found {live.version}.",
                    )
                )
        locked = locked_conda_packages(paths.lockfile, definition.platform)
        for name, expected in locked.items():
            live = live_conda.get(name)
            if not live:
                drift.append(
                    DriftItem(
                        layer="conda-lock",
                        kind="missing",
                        package=expected.name,
                        expected=expected.version,
                        message=f"Locked Conda package {expected.name} is missing.",
                    )
                )
                continue
            if live.version != expected.version:
                drift.append(
                    DriftItem(
                        layer="conda-lock",
                        kind="version",
                        package=expected.name,
                        expected=expected.version,
                        actual=live.version,
                        message=f"{expected.name}: locked {expected.version}, found {live.version}.",
                    )
                )
            if expected.build and live.build and live.build != expected.build:
                drift.append(
                    DriftItem(
                        layer="conda-lock",
                        kind="build",
                        package=expected.name,
                        expected=expected.build,
                        actual=live.build,
                        message=f"{expected.name}: build drift.",
                    )
                )
            if expected.channel and live.channel:
                expected_channel = self._channel_name(expected.channel)
                live_channel = self._channel_name(live.channel)
                if expected_channel and live_channel and expected_channel != live_channel:
                    drift.append(
                        DriftItem(
                            layer="conda-lock",
                            kind="channel",
                            package=expected.name,
                            expected=expected_channel,
                            actual=live_channel,
                            message=f"{expected.name}: channel/source drift.",
                        )
                    )
        for name, live in live_conda.items():
            if name not in locked:
                drift.append(
                    DriftItem(
                        layer="conda-lock",
                        kind="unexpected",
                        package=live.name,
                        actual=live.version,
                        message=f"Unexpected Conda package {live.name} {live.version}.",
                    )
                )

        expected_pip = declared_pip_requirements(paths.pip_extras)
        all_pip = _normalized_map(snapshot.pip_packages)
        pip_managed = {
            normalize_package_name(item.name): item
            for item in snapshot.conda_packages
            if item.manager == "pip"
        }
        for name, record in pip_managed.items():
            if name in locked:
                drift.append(
                    DriftItem(
                        layer="pip",
                        kind="shadows_conda",
                        package=record.name,
                        expected=locked[name].version,
                        actual=record.version,
                        message=f"pip-installed {record.name} shadows a locked Conda package.",
                    )
                )
        live_pip = {
            name: record for name, record in all_pip.items() if name not in live_conda
        }
        live_pip = {
            name: record
            for name, record in live_pip.items()
            if name not in {"spatialbiologytoolkit", "spatial-biology-toolkit"}
        }
        for name, expected in expected_pip.items():
            live = live_pip.get(name)
            if not live:
                drift.append(
                    DriftItem(
                        layer="pip",
                        kind="missing",
                        package=expected.name,
                        expected=expected.version or expected.requirement,
                        message=f"Declared pip extra {expected.name} is missing.",
                    )
                )
            elif expected.version and expected.version != live.version:
                drift.append(
                    DriftItem(
                        layer="pip",
                        kind="version",
                        package=expected.name,
                        expected=expected.version,
                        actual=live.version,
                        message=f"{expected.name}: expected {expected.version}, found {live.version}.",
                    )
                )
        for name, live in live_pip.items():
            if name not in expected_pip:
                drift.append(
                    DriftItem(
                        layer="pip",
                        kind="unexpected",
                        package=live.name,
                        actual=live.version,
                        message=f"Unexpected pip package {live.name} {live.version}.",
                    )
                )
        for record in snapshot.editable_packages:
            if normalize_package_name(record.name) not in {
                "spatialbiologytoolkit",
                "spatial-biology-toolkit",
            }:
                drift.append(
                    DriftItem(
                        layer="pip",
                        kind="editable",
                        package=record.name,
                        actual=record.location,
                        message=f"Unexpected editable pip package {record.name}.",
                    )
                )
        toolkit = snapshot.toolkit
        if definition.toolkit_overlay == "editable-no-deps":
            if not toolkit.installed:
                drift.append(
                    DriftItem(
                        layer="toolkit",
                        kind="missing",
                        message="SpatialBiologyToolkit overlay is not installed.",
                    )
                )
            elif not toolkit.editable:
                drift.append(
                    DriftItem(
                        layer="toolkit",
                        kind="not_editable",
                        message="SpatialBiologyToolkit is installed but not editable.",
                    )
                )
            if toolkit.repository_matches is False:
                drift.append(
                    DriftItem(
                        layer="toolkit",
                        kind="path_mismatch",
                        expected=str(self.repository_root),
                        actual=str(toolkit.repository_path),
                        message="Toolkit editable installation points to another checkout.",
                    )
                )
            if toolkit.checkout_dirty:
                drift.append(
                    DriftItem(
                        layer="toolkit",
                        kind="dirty_checkout",
                        actual=str(toolkit.checkout_git_commit or "unknown"),
                        material=False,
                        message="Toolkit checkout has uncommitted changes.",
                    )
                )
        return drift

    @staticmethod
    def _channel_name(value: str) -> str:
        candidate = value.rstrip("/")
        if "/" not in candidate:
            return candidate.casefold()
        parts = candidate.split("/")
        if "conda.anaconda.org" in parts:
            index = parts.index("conda.anaconda.org")
            return parts[index + 1].casefold() if len(parts) > index + 1 else ""
        return parts[-3].casefold() if len(parts) >= 3 else parts[-1].casefold()

    def _generate_lock(
        self,
        environment_yml: Path,
        destination: Path,
        platform: str,
        *,
        channel_priority: str | None = None,
        verbose: bool = False,
    ) -> list[str]:
        if not self.conda_lock_command:
            raise RuntimeError("Conda is unavailable; cannot run conda-lock from base.")
        file_option = self._conda_lock_option("lock", "--file", "-f")
        platform_option = self._conda_lock_option("lock", "--platform", "-p")
        command = [
            *self.conda_lock_command,
            "lock",
            file_option,
            str(environment_yml),
            platform_option,
            platform,
            "--lockfile",
            str(destination),
        ]
        if verbose:
            print(f"Running: {command_text(command)}")
        lock_environment = None
        if channel_priority:
            lock_environment = os.environ.copy()
            lock_environment["CONDA_CHANNEL_PRIORITY"] = channel_priority
        run_checked(
            command,
            runner=self.runner,
            cwd=self.repository_root,
            env=lock_environment,
        )
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError("conda-lock completed without creating a non-empty lockfile.")
        return command

    def _conda_lock_option(self, subcommand: str, long: str, short: str) -> str:
        """Inspect installed help while retaining a documented modern default."""
        if not self.conda_lock_command:
            return long
        try:
            completed = self.runner(
                [*self.conda_lock_command, subcommand, "--help"],
                capture_output=True,
                text=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return long
        help_text = f"{completed.stdout}\n{completed.stderr}"
        if long in help_text:
            return long
        if short in help_text:
            return short
        return long

    def lock(
        self,
        selector: str,
        *,
        check: bool = False,
        verbose: bool = False,
    ) -> tuple[bool, list[str]]:
        key, definition = self.resolve(selector)
        if not definition.managed:
            raise ValueError(f"Environment {key!r} is externally managed.")
        validation = self.validate(key)
        blocking = [
            issue
            for issue in validation.issues
            if issue.severity == "error"
            and issue.code not in {"missing_lockfile", "legacy_pip_in_lock"}
        ]
        if blocking:
            raise ValueError("Cannot lock invalid specification: " + "; ".join(item.message for item in blocking))
        paths = validation.paths
        assert paths.environment_yml and paths.lockfile
        paths.lockfile.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".sbt-env-lock-", dir=paths.lockfile.parent
        ) as temporary:
            candidate = Path(temporary) / paths.lockfile.name
            command = self._generate_lock(
                paths.environment_yml,
                candidate,
                definition.platform,
                channel_priority=definition.conda_channel_priority,
                verbose=verbose,
            )
            current = paths.lockfile.read_bytes() if paths.lockfile.is_file() else None
            generated = candidate.read_bytes()
            current_matches = current == generated
            if not check and not current_matches:
                paths.lockfile.parent.mkdir(parents=True, exist_ok=True)
                os.replace(candidate, paths.lockfile)
            return current_matches, command

    def sync_plan(self, selector: str) -> SyncPlan:
        key, definition = self.resolve(selector)
        if not definition.managed:
            raise ValueError(f"Environment {key!r} is externally managed.")
        validation = self.validate(key)
        if not validation.valid:
            raise ValueError("Invalid environment specification: " + "; ".join(
                issue.message for issue in validation.issues if issue.severity == "error"
            ))
        exists = False
        drift = "unknown"
        if self.conda:
            exists = definition.conda_name in self._environment_inventory()
            if exists:
                comparison = self.compare(key)
                if comparison.result == "error":
                    raise RuntimeError(comparison.error or "Environment comparison failed.")
                drift = "clean" if comparison.result == "clean" else "drift"
        actions: list[str] = []
        recreation = exists and drift == "drift"
        if not exists:
            actions.append(f"Create fixed environment {definition.conda_name} from lockfile")
        elif recreation:
            actions.extend(
                [
                    "Capture current installed state to SBT user history",
                    f"Remove fixed environment {definition.conda_name}",
                    f"Recreate fixed environment {definition.conda_name} from lockfile",
                ]
            )
        if not exists or recreation:
            if validation.paths.pip_extras and validation.paths.pip_extras.stat().st_size:
                actions.append("Install pinned pip extras through conda run")
            if definition.toolkit_overlay == "editable-no-deps":
                actions.append("Install SpatialBiologyToolkit editable overlay with --no-deps")
            actions.append("Run registered lightweight smoke tests")
            actions.append("Write observed environment snapshot to SBT user state")
        return SyncPlan(
            environment_key=key,
            conda_name=definition.conda_name,
            exists=exists,
            drift=drift,
            recreation_required=recreation,
            actions=actions,
            paths=validation.paths,
            smoke_tests=definition.smoke_tests,
        )

    def _write_state_snapshot(self, snapshot, category: str = "environments") -> Path:
        destination = (
            self.state_root
            / category
            / snapshot.environment_name
            / f"{snapshot.captured_at.strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        atomic_write_text(destination, snapshot.model_dump_json(indent=2) + "\n")
        return destination

    def sync(
        self,
        selector: str,
        *,
        dry_run: bool = False,
        recreate: bool = False,
        confirmed: bool = False,
        verbose: bool = False,
    ) -> SyncPlan:
        plan = self.sync_plan(selector)
        if dry_run or not plan.actions:
            return plan
        key, definition = self.resolve(selector)
        if not self.conda:
            raise RuntimeError("Conda executable was not found on PATH.")
        if not self.conda_lock_command:
            raise RuntimeError("Conda is unavailable; cannot run conda-lock from base.")
        if plan.recreation_required and not recreate:
            raise RuntimeError("Drift detected; pass --recreate to request destructive recreation.")
        if plan.recreation_required and not confirmed:
            raise RuntimeError("Destructive recreation was not confirmed.")
        paths = plan.paths
        assert paths.lockfile and paths.pip_extras
        if plan.recreation_required:
            snapshot = inspect_environment(
                key=key,
                definition=definition,
                repository_root=self.repository_root,
                conda=self.conda,
                runner=self.runner,
            )
            self._write_state_snapshot(snapshot, "environment_history")
            remove_command = [
                self.conda,
                "env",
                "remove",
                "--name",
                definition.conda_name,
                "--yes",
            ]
            if verbose:
                print(f"Running: {command_text(remove_command)}")
            run_checked(remove_command, runner=self.runner)
            if definition.conda_name in self._environment_inventory():
                raise RuntimeError(
                    f"Refusing to continue: environment {definition.conda_name!r} still exists."
                )
        name_option = self._conda_lock_option("install", "--name", "-n")
        install_command = [
            *self.conda_lock_command,
            "install",
            name_option,
            definition.conda_name,
            str(paths.lockfile),
        ]
        if verbose:
            print(f"Running: {command_text(install_command)}")
        run_checked(install_command, runner=self.runner, cwd=self.repository_root)
        if paths.pip_extras.is_file() and paths.pip_extras.read_text(encoding="utf-8").strip():
            pip_command = [
                self.conda,
                "run",
                "-n",
                definition.conda_name,
                "python",
                "-m",
                "pip",
                "install",
                "-r",
                str(paths.pip_extras),
            ]
            if verbose:
                print(f"Running: {command_text(pip_command)}")
            run_checked(pip_command, runner=self.runner)
        if definition.toolkit_overlay == "editable-no-deps":
            overlay_command = [
                self.conda,
                "run",
                "-n",
                definition.conda_name,
                "python",
                "-m",
                "pip",
                "install",
                "-e",
                str(self.repository_root),
                "--no-deps",
            ]
            if verbose:
                print(f"Running: {command_text(overlay_command)}")
            run_checked(overlay_command, runner=self.runner)
        report = self.test(key, verbose=verbose)
        if not report.passed:
            raise RuntimeError(f"Smoke tests failed for {definition.conda_name}.")
        snapshot = inspect_environment(
            key=key,
            definition=definition,
            repository_root=self.repository_root,
            conda=self.conda,
            runner=self.runner,
        )
        self._write_state_snapshot(snapshot)
        return plan

    def refresh_overlays(
        self,
        *,
        existing_only: bool = True,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> OverlayRefreshReport:
        """Refresh editable toolkit installs without changing environment dependencies."""

        if not self.conda:
            raise RuntimeError("Conda executable was not found on PATH.")
        inventory = self._environment_inventory()
        results: list[OverlayRefreshResult] = []
        for key, definition in self.registry.environments.items():
            if definition.toolkit_overlay != "editable-no-deps":
                results.append(
                    OverlayRefreshResult(
                        environment_key=key,
                        conda_name=definition.conda_name,
                        status="skipped",
                        message="The environment does not use an editable toolkit overlay.",
                    )
                )
                continue
            if definition.conda_name not in inventory:
                results.append(
                    OverlayRefreshResult(
                        environment_key=key,
                        conda_name=definition.conda_name,
                        status="skipped" if existing_only else "failed",
                        message=(
                            "The registered Conda environment is not installed."
                            if existing_only
                            else "The required registered Conda environment is not installed."
                        ),
                    )
                )
                continue

            command = [
                self.conda,
                "run",
                "-n",
                definition.conda_name,
                "python",
                "-m",
                "pip",
                "install",
                "-e",
                str(self.repository_root),
                "--no-deps",
                "--no-build-isolation",
            ]
            if dry_run:
                results.append(
                    OverlayRefreshResult(
                        environment_key=key,
                        conda_name=definition.conda_name,
                        status="planned",
                        message="Would refresh the editable SpatialBiologyToolkit overlay.",
                    )
                )
                continue
            if verbose:
                print(f"Running: {command_text(command)}")
            started = time.monotonic()
            try:
                run_checked(command, runner=self.runner, cwd=self.repository_root)
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                results.append(
                    OverlayRefreshResult(
                        environment_key=key,
                        conda_name=definition.conda_name,
                        status="failed",
                        duration_seconds=time.monotonic() - started,
                        message=str(exc),
                    )
                )
            else:
                results.append(
                    OverlayRefreshResult(
                        environment_key=key,
                        conda_name=definition.conda_name,
                        status="updated",
                        duration_seconds=time.monotonic() - started,
                        message="Refreshed the editable SpatialBiologyToolkit overlay.",
                    )
                )
        return OverlayRefreshReport(
            repository_root=self.repository_root,
            dry_run=dry_run,
            existing_only=existing_only,
            results=results,
        )

    def test(self, selector: str, *, verbose: bool = False) -> EnvironmentTestReport:
        key, definition = self.resolve(selector)
        if not self.conda:
            raise RuntimeError("Conda executable was not found on PATH.")
        if definition.conda_name not in self._environment_inventory():
            raise RuntimeError(f"Conda environment {definition.conda_name!r} does not exist.")
        results: list[SmokeTestResult] = []
        for smoke_test in definition.smoke_tests:
            command = [self.conda, "run", "-n", definition.conda_name, *smoke_test]
            if verbose:
                print(f"Running: {command_text(command)}")
            started = time.monotonic()
            try:
                completed = self.runner(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return_code = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
            except (OSError, subprocess.SubprocessError) as exc:
                return_code = 127
                stdout = ""
                stderr = str(exc)
            results.append(
                SmokeTestResult(
                    command=command,
                    return_code=return_code,
                    stdout_tail=tail(stdout),
                    stderr_tail=tail(stderr),
                    duration_seconds=time.monotonic() - started,
                    passed=return_code == 0,
                )
            )
        return EnvironmentTestReport(
            environment_key=key,
            conda_name=definition.conda_name,
            passed=all(item.passed for item in results),
            tests=results,
        )

    def _capture_export(
        self,
        definition: EnvironmentDefinition,
        *,
        conda_prefix: Path | None = None,
    ) -> dict[str, Any]:
        if not self.conda:
            raise RuntimeError("Conda executable was not found on PATH.")
        selector = (
            ["--prefix", str(conda_prefix)]
            if conda_prefix is not None
            else ["--name", definition.conda_name]
        )
        commands = [
            [
                self.conda,
                "env",
                "export",
                *selector,
                "--from-history",
            ],
            [
                self.conda,
                "export",
                *selector,
                "--from-history",
                "--format",
                "environment-yaml",
            ],
        ]
        errors: list[str] = []
        for command in commands:
            try:
                completed = run_checked(command, runner=self.runner)
                loaded = yaml.safe_load(completed.stdout)
                if isinstance(loaded, dict):
                    return loaded
            except (RuntimeError, yaml.YAMLError) as exc:
                errors.append(str(exc))
        raise RuntimeError("Conda from-history export failed:\n" + "\n".join(errors))

    def capture(
        self,
        selector: str,
        *,
        write: bool = False,
        accept_vcs: bool = False,
        verbose: bool = False,
    ) -> CapturePlan:
        key, definition = self.resolve(selector)
        return self._capture_definition(
            key,
            definition,
            write=write,
            accept_vcs=accept_vcs,
            verbose=verbose,
        )

    def capture_target(
        self,
        target: EnvironmentCaptureTarget,
        *,
        accept_vcs: bool = False,
        verbose: bool = False,
    ) -> CapturePlan:
        """Capture one exact Conda prefix discovered outside or inside the registry."""
        if target.registered:
            key, definition = self.resolve(target.environment_key)
            if definition.conda_name.casefold() != target.conda_name.casefold():
                raise RuntimeError(
                    f"Discovered target {target.environment_key!r} no longer matches "
                    "the environment registry."
                )
        else:
            key = target.environment_key
            definition = EnvironmentDefinition(
                conda_name=target.conda_name,
                platform=target.platform,
                toolkit_overlay="none",
                managed=False,
                notes=[
                    "Discovered from Conda and not registered with SpatialBiologyToolkit."
                ],
            )
        return self._capture_definition(
            key,
            definition,
            write=False,
            accept_vcs=accept_vcs,
            verbose=verbose,
            conda_prefix=target.conda_prefix,
            registered=target.registered,
            capture_directory_name=target.capture_directory_name,
            retain_lock_failure=True,
        )

    def _capture_definition(
        self,
        key: str,
        definition: EnvironmentDefinition,
        *,
        write: bool,
        accept_vcs: bool,
        verbose: bool,
        conda_prefix: Path | None = None,
        registered: bool = True,
        capture_directory_name: str | None = None,
        retain_lock_failure: bool = False,
    ) -> CapturePlan:
        if write and not definition.managed:
            raise ValueError(
                f"Environment {key!r} is externally managed; capture it without --write "
                "to create an observational compatibility bundle. Promote it to a "
                "repository-managed specification before writing repository files."
            )
        if not self.conda:
            raise RuntimeError("Conda executable was not found on PATH.")
        if (
            conda_prefix is None
            and definition.conda_name not in self._environment_inventory()
        ):
            raise RuntimeError(f"Conda environment {definition.conda_name!r} does not exist.")
        snapshot = inspect_environment(
            key=key,
            definition=definition,
            repository_root=self.repository_root,
            conda=self.conda,
            conda_prefix=conda_prefix,
            runner=self.runner,
        )
        exported = self._capture_export(definition, conda_prefix=conda_prefix)
        dependencies = sorted(
            [item for item in exported.get("dependencies", []) if isinstance(item, str)],
            key=lambda item: normalize_package_name(item.split("=", 1)[0]),
        )
        channels = list(dict.fromkeys(str(item) for item in exported.get("channels", [])))
        if not any(
            normalize_package_name(item.split("=", 1)[0]) == "pip"
            for item in dependencies
        ):
            dependencies.append("pip")
        candidate_data: dict[str, Any] = {
            "name": definition.conda_name,
            "channels": channels,
            "dependencies": dependencies,
        }
        current_paths = self.paths(key, definition)
        if current_paths.environment_yml and current_paths.environment_yml.is_file():
            current_data = load_environment_yml(current_paths.environment_yml)
            if not channels:
                candidate_data["channels"] = current_data.get("channels", [])
            if isinstance(current_data.get("variables"), dict):
                candidate_data["variables"] = current_data["variables"]
        environment_text = yaml.safe_dump(candidate_data, sort_keys=False)
        conda_names = {
            normalize_package_name(item.name)
            for item in snapshot.conda_packages
            if item.manager == "conda"
        }
        pip_lines: list[str] = []
        review: list[str] = [
            item
            for item in snapshot.review_requirements
            if item.startswith(
                ("Python/pip inspection unavailable:", "pip freeze unavailable:")
            )
        ]
        excluded_toolkit: str | None = None
        for package in snapshot.pip_packages:
            normalized = normalize_package_name(package.name)
            if normalized in {"spatialbiologytoolkit", "spatial-biology-toolkit"}:
                excluded_toolkit = package.requirement or package.location or package.name
                continue
            if normalized in conda_names:
                continue
            if package.source_type == "vcs" and accept_vcs and package.requirement:
                pip_lines.append(package.requirement)
                continue
            if package.source_type in {"editable", "local", "vcs"}:
                candidate = package.requirement or f"{package.name} ({package.source_type})"
                if candidate not in review:
                    review.append(candidate)
                continue
            if package.version:
                pip_lines.append(f"{package.name}=={package.version}")
        pip_text = "\n".join(sorted(set(pip_lines), key=str.casefold))
        if pip_text:
            pip_text += "\n"
        capture_directory = self.state_root / "captures" / (
            capture_directory_name or definition.conda_name
        ) / _timestamp()
        capture_directory.mkdir(parents=True, exist_ok=False)
        candidate_yml = capture_directory / "environment.yml"
        candidate_extras = capture_directory / "pip-extras.txt"
        candidate_lock = capture_directory / f"conda-{definition.platform}.lock"
        atomic_write_text(candidate_yml, environment_text)
        atomic_write_text(candidate_extras, pip_text)
        atomic_write_text(
            capture_directory / "environment.snapshot.json",
            snapshot.model_dump_json(indent=2) + "\n",
        )
        lock_generation_error: str | None = None
        try:
            self._generate_lock(
                candidate_yml,
                candidate_lock,
                definition.platform,
                channel_priority=definition.conda_channel_priority,
                verbose=verbose,
            )
        except RuntimeError as exc:
            if definition.managed and not retain_lock_failure:
                raise RuntimeError(
                    f"Lock generation failed; candidates retained in {capture_directory}: {exc}"
                ) from exc
            lock_generation_error = str(exc)
        paths = self.paths(key, definition)
        differences: dict[str, str] = {}
        if paths.environment_yml:
            differences["environment.yml"] = self._diff_file(
                paths.environment_yml, environment_text
            )
        else:
            differences["environment.yml"] = "no repository specification to compare"
        if paths.pip_extras:
            differences["pip-extras.txt"] = self._diff_file(
                paths.pip_extras, pip_text
            )
        else:
            differences["pip-extras.txt"] = "no repository specification to compare"
        lock_name = candidate_lock.name
        if lock_generation_error:
            differences[lock_name] = (
                "candidate lock generation failed; exact installed packages remain in "
                "environment.snapshot.json"
            )
        elif paths.lockfile and paths.lockfile.is_file():
            differences[lock_name] = (
                "unchanged"
                if paths.lockfile.read_bytes() == candidate_lock.read_bytes()
                else "generated lockfile differs"
            )
        else:
            differences[lock_name] = "no repository lock to compare; candidate generated"
        plan = CapturePlan(
            environment_key=key,
            conda_name=definition.conda_name,
            managed=definition.managed,
            registered=registered,
            conda_prefix=snapshot.conda_prefix or conda_prefix,
            candidate_directory=capture_directory,
            environment_yml=environment_text,
            pip_extras=pip_text,
            lockfile=candidate_lock if candidate_lock.is_file() else None,
            lock_generation_error=lock_generation_error,
            review_requirements=sorted(set(review), key=str.casefold),
            excluded_toolkit=excluded_toolkit,
            differences=differences,
        )
        atomic_write_text(
            capture_directory / "capture-plan.json",
            plan.model_dump_json(indent=2) + "\n",
        )
        if write:
            assert paths.environment_yml
            assert paths.pip_extras
            assert paths.lockfile
            assert paths.observed_snapshot
            if plan.review_requirements:
                raise RuntimeError(
                    "Capture contains local/editable/VCS requirements requiring manual review; "
                    f"candidates retained in {capture_directory}."
                )
            atomic_write_text(paths.environment_yml, environment_text)
            atomic_write_text(paths.pip_extras, pip_text)
            atomic_write_bytes(paths.lockfile, candidate_lock.read_bytes())
            atomic_write_text(
                paths.observed_snapshot, snapshot.model_dump_json(indent=2) + "\n"
            )
            validation = self.validate(key)
            if not validation.valid:
                raise RuntimeError(
                    "Written capture did not validate: "
                    + "; ".join(
                        issue.message
                        for issue in validation.issues
                        if issue.severity == "error"
                    )
                )
        return plan

    @staticmethod
    def _diff_file(path: Path, proposed: str) -> str:
        current = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        if current == proposed:
            return "unchanged"
        return "\n".join(
            difflib.unified_diff(
                current.splitlines(),
                proposed.splitlines(),
                fromfile=str(path),
                tofile=f"{path} (proposed)",
                lineterm="",
            )
        )


__all__ = ["EnvironmentManager"]
