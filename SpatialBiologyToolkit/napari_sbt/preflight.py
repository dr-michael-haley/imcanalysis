"""Side-effect-free launch checks for local and CSF3 NapariSBT sessions."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .resources import resolve_worker_count


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: Literal["ok", "warning", "error"]
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    checks: tuple[PreflightCheck, ...]

    @property
    def ready(self) -> bool:
        return not any(check.status == "error" for check in self.checks)

    @property
    def exit_code(self) -> int:
        return 0 if self.ready else 2


REQUIRED_MODULES = {
    "napari": "Napari viewer",
    "qtpy": "Qt abstraction",
    "anndata": "AnnData input",
    "numpy": "array processing",
    "pandas": "cell tables",
    "pyarrow": "Parquet feature assets",
    "skimage": "image and mask features",
    "sklearn": "classification",
    "tifffile": "TIFF images",
    "joblib": "model storage",
    "psutil": "worker health monitoring",
}
QT_BINDINGS = ("PyQt5", "PyQt6", "PySide2", "PySide6")


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _path_check(name: str, path: Path | None, *, kind: str) -> PreflightCheck | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve(strict=False)
    exists = resolved.is_file() if kind == "file" else resolved.is_dir()
    return PreflightCheck(
        name,
        "ok" if exists else "error",
        f"{resolved} ({'found' if exists else 'not found'})",
    )


def _writable_check(path: Path) -> PreflightCheck:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    writable = candidate.exists() and os.access(candidate, os.W_OK)
    return PreflightCheck(
        "Experiment output",
        "ok" if writable else "error",
        f"Nearest existing parent {candidate} is "
        f"{'writable' if writable else 'not writable'}.",
    )


def run_preflight(
    *,
    project_root: Path | None = None,
    experiment: Path | None = None,
    anndata_path: Path | None = None,
    masks_folder: Path | None = None,
    images_folders: tuple[Path, ...] = (),
    worker_count: int | None = None,
) -> PreflightReport:
    """Inspect launch prerequisites without importing Qt or opening datasets."""

    checks: list[PreflightCheck] = []
    missing = [
        description
        for module, description in REQUIRED_MODULES.items()
        if not _module_available(module)
    ]
    checks.append(
        PreflightCheck(
            "Python environment",
            "ok" if not missing else "error",
            (
                f"Required runtime modules are available in {sys.executable}."
                if not missing
                else "Missing: " + ", ".join(missing)
            ),
        )
    )
    binding = next((name for name in QT_BINDINGS if _module_available(name)), None)
    checks.append(
        PreflightCheck(
            "Qt binding",
            "ok" if binding else "error",
            f"Using {binding}." if binding else "No PyQt or PySide binding was found.",
        )
    )

    display = os.environ.get("DISPLAY")
    if sys.platform.startswith("linux"):
        checks.append(
            PreflightCheck(
                "X11 display",
                "ok" if display else "error",
                (
                    f"DISPLAY={display}"
                    if display
                    else "DISPLAY is unset; reconnect with X11 and use srun-x11."
                ),
            )
        )
    else:
        checks.append(PreflightCheck("Display", "ok", f"Native {sys.platform} display session."))

    job_id = os.environ.get("SLURM_JOB_ID")
    checks.append(
        PreflightCheck(
            "Slurm allocation",
            "ok" if job_id else "warning",
            (
                (
                    f"Job {job_id} on "
                    f"{os.environ.get('SLURMD_NODENAME') or os.environ.get('HOSTNAME', 'compute node')}."
                )
                if job_id
                else (
                    "No SLURM_JOB_ID is present. On CSF3, do not continue on a "
                    "login node; request srun-x11 first."
                )
            ),
        )
    )

    resolution = resolve_worker_count(worker_count)
    checks.append(
        PreflightCheck(
            "Feature workers",
            "warning" if resolution.adjusted else "ok",
            resolution.message,
        )
    )

    for check in (
        _path_check("Project", project_root, kind="directory"),
        _path_check("AnnData", anndata_path, kind="file"),
        _path_check("Masks", masks_folder, kind="directory"),
    ):
        if check is not None:
            checks.append(check)
    for index, folder in enumerate(images_folders, start=1):
        check = _path_check(f"Image folder {index}", folder, kind="directory")
        if check is not None:
            checks.append(check)

    if experiment is not None:
        manifest = (
            experiment
            if experiment.name == "experiment.yaml"
            else experiment / "experiment.yaml"
        )
        check = _path_check("Experiment manifest", manifest, kind="file")
        if check is not None:
            checks.append(check)
        checks.append(_writable_check(manifest.parent))
    elif project_root is not None:
        checks.append(_writable_check(project_root / "napari_sbt"))
    else:
        checks.append(
            PreflightCheck(
                "Dataset inputs",
                "warning",
                "No project or experiment was supplied; configure inputs in Setup.",
            )
        )
    return PreflightReport(tuple(checks))


def format_preflight(report: PreflightReport, output_format: str = "text") -> str:
    """Render a preflight report for people or simple automation."""

    if output_format == "json":
        return json.dumps(
            {
                "ready": report.ready,
                "checks": [asdict(check) for check in report.checks],
            },
            indent=2,
        )
    heading = "READY" if report.ready else "BLOCKED"
    lines = [f"NapariSBT preflight: {heading}"]
    labels = {"ok": "OK", "warning": "WARN", "error": "ERROR"}
    lines.extend(
        f"[{labels[check.status]}] {check.name}: {check.detail}"
        for check in report.checks
    )
    return "\n".join(lines)


__all__ = [
    "PreflightCheck",
    "PreflightReport",
    "format_preflight",
    "run_preflight",
]
