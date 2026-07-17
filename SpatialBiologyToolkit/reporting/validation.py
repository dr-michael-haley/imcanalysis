"""Conservative validation of numbered reports and legacy output folders."""

from __future__ import annotations

from pathlib import Path

from SpatialBiologyToolkit.pipeline.manifests import read_model
from SpatialBiologyToolkit.pipeline.models import ValidationItem
from SpatialBiologyToolkit.pipeline.registry import STAGES

from .models import StageManifest


def validate_reporting_layout(
    *,
    project_root: Path,
    outputs_root: Path,
    legacy_qc: Path,
) -> list[ValidationItem]:
    items: list[ValidationItem] = []
    expected_names = {stage.output_folder for stage in STAGES}
    items.append(
        ValidationItem(
            name="project output index",
            path=outputs_root / "README.md",
            status="ok" if (outputs_root / "README.md").is_file() else "warning",
            message=(
                "Human-facing project index is present."
                if (outputs_root / "README.md").is_file()
                else "outputs/README.md is missing; re-adopt the project or run a stage."
            ),
        )
    )

    for stage in STAGES:
        stage_root = outputs_root / stage.output_folder
        if not stage_root.is_dir():
            items.append(
                ValidationItem(
                    name=f"{stage.name} output folder",
                    path=stage_root,
                    status="not_created",
                    message="The numbered stage folder has not been created.",
                )
            )
            continue
        if not (stage_root / "README.md").is_file():
            items.append(
                ValidationItem(
                    name=f"{stage.name} stage index",
                    path=stage_root / "README.md",
                    status="warning",
                    message="Numbered stage folder is missing its README index.",
                )
            )
        for run_dir in sorted(path for path in stage_root.iterdir() if path.is_dir()):
            manifest_path = run_dir / "stage_manifest.yaml"
            readme_path = run_dir / "README.md"
            if not manifest_path.is_file() or not readme_path.is_file():
                items.append(
                    ValidationItem(
                        name=f"{stage.name} run {run_dir.name}",
                        path=run_dir,
                        status="warning",
                        message="Run folder must contain stage_manifest.yaml and README.md.",
                    )
                )
                continue
            try:
                manifest = read_model(manifest_path, StageManifest)
            except (OSError, ValueError) as exc:
                items.append(
                    ValidationItem(
                        name=f"{stage.name} run {run_dir.name}",
                        path=manifest_path,
                        status="warning",
                        message=f"Stage manifest is invalid: {exc}",
                    )
                )
                continue
            missing_paths = [
                str(record.path)
                for record in [*manifest.inputs, *manifest.produced_assets]
                if record.exists is True and not record.path.exists()
            ]
            technical_missing = (
                manifest.technical_run_record is not None
                and not manifest.technical_run_record.is_dir()
            )
            if missing_paths or technical_missing:
                details = missing_paths[:3]
                if technical_missing:
                    details.append(str(manifest.technical_run_record))
                items.append(
                    ValidationItem(
                        name=f"{stage.name} run {run_dir.name} links",
                        path=run_dir,
                        status="warning",
                        message="Recorded paths no longer resolve: " + ", ".join(details),
                    )
                )

    if outputs_root.is_dir():
        for path in outputs_root.iterdir():
            if (
                path.is_dir()
                and path.name[:3].isdigit()
                and path.name not in expected_names
            ):
                items.append(
                    ValidationItem(
                        name="unregistered numbered output folder",
                        path=path,
                        status="warning",
                        message="Folder name does not match the central stage registry.",
                    )
                )
    if legacy_qc.is_dir():
        items.append(
            ValidationItem(
                name="legacy QC folder",
                path=legacy_qc,
                status="warning",
                message=(
                    "Legacy outputs are preserved. New reported runs use numbered "
                    "folders under outputs/."
                ),
            )
        )
    if not items:
        items.append(
            ValidationItem(
                name="reporting layout",
                path=outputs_root,
                status="ok",
                message="No reporting layout issues were identified.",
            )
        )
    return items


__all__ = ["validate_reporting_layout"]
