"""Resolve configured project assets without loading scientific data."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

from SpatialBiologyToolkit.config.models import PipelineConfig

from .manifests import utc_now
from .models import AssetInventory, ProjectAsset


AssetKind = Literal["file", "directory"]
AssetLifecycle = Literal[
    "required_input",
    "optional_input",
    "generated_output",
    "human_output",
    "legacy_output",
    "operational_state",
]


ASSET_FIELDS: tuple[tuple[str, str, AssetKind, AssetLifecycle], ...] = (
    ("raw_imc_files", "imc_files_folder", "directory", "required_input"),
    ("metadata", "metadata_folder", "directory", "optional_input"),
    ("tiff_stacks", "tiff_stacks_folder", "directory", "generated_output"),
    ("raw_images", "raw_images_folder", "directory", "generated_output"),
    (
        "denoised_images",
        "denoised_images_folder",
        "directory",
        "generated_output",
    ),
    ("masks", "masks_folder", "directory", "generated_output"),
    ("cell_tables", "celltable_folder", "directory", "generated_output"),
    ("anndata", "anndata_path", "file", "generated_output"),
    ("human_outputs", "outputs_folder", "directory", "human_output"),
    ("legacy_qc", "qc_folder", "directory", "legacy_output"),
    ("legacy_slurm_logs", "slurm_logs_folder", "directory", "legacy_output"),
)

RAW_IMC_SUFFIXES = {".mcd", ".txt"}
DEFAULT_COUNT_LIMIT = 10_000


def resolve_project_path(root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def _direct_entry_count(path: Path, limit: int) -> tuple[int | None, bool]:
    if not path.is_dir():
        return None, False
    count = 0
    try:
        with os.scandir(path) as entries:
            for _entry in entries:
                count += 1
                if count >= limit:
                    return count, True
    except OSError:
        return None, False
    return count, False


def _modified_at(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def inspect_asset(
    *,
    role: str,
    path: Path,
    kind: AssetKind,
    lifecycle: AssetLifecycle,
    count_limit: int = DEFAULT_COUNT_LIMIT,
) -> ProjectAsset:
    exists = path.is_file() if kind == "file" else path.is_dir()
    size_bytes: int | None = None
    file_count: int | None = None
    count_limited = False
    if exists and kind == "file":
        try:
            size_bytes = path.stat().st_size
        except OSError:
            pass
    elif exists:
        file_count, count_limited = _direct_entry_count(path, count_limit)
    return ProjectAsset(
        role=role,
        path=path,
        kind=kind,
        lifecycle=lifecycle,
        exists=exists,
        size_bytes=size_bytes,
        modified_at=_modified_at(path) if exists else None,
        file_count=file_count,
        count_limited=count_limited,
    )


def resolve_assets(
    config: PipelineConfig,
    root: Path,
    *,
    count_limit: int = DEFAULT_COUNT_LIMIT,
) -> list[ProjectAsset]:
    general = config.general
    assets = [
        inspect_asset(
            role=role,
            path=resolve_project_path(root, getattr(general, field_name)),
            kind=kind,
            lifecycle=lifecycle,
            count_limit=count_limit,
        )
        for role, field_name, kind, lifecycle in ASSET_FIELDS
    ]
    assets.append(
        inspect_asset(
            role="cellvision_assets",
            path=resolve_project_path(root, config.cellvision.asset_folder),
            kind="directory",
            lifecycle="generated_output",
            count_limit=count_limit,
        )
    )
    assets.append(
        inspect_asset(
            role="hyperstac_input_images",
            path=resolve_project_path(
                root,
                config.hyperstac.input_images_folder
                or config.general.denoised_images_folder,
            ),
            kind="directory",
            lifecycle="required_input",
            count_limit=count_limit,
        )
    )
    assets.append(
        inspect_asset(
            role="hyperstac_assets",
            path=resolve_project_path(root, config.hyperstac.asset_folder),
            kind="directory",
            lifecycle="generated_output",
            count_limit=count_limit,
        )
    )
    assets.append(
        inspect_asset(
            role="population_qc_anndata",
            path=resolve_project_path(
                root, config.population_embedding_qc.annotated_adata_path
            ),
            kind="file",
            lifecycle="generated_output",
            count_limit=count_limit,
        )
    )
    assets.append(
        inspect_asset(
            role="napari_sbt_experiments",
            path=resolve_project_path(root, config.napari_sbt.experiment_folder),
            kind="directory",
            lifecycle="human_output",
            count_limit=count_limit,
        )
    )
    return assets


def asset_map(assets: Iterable[ProjectAsset]) -> dict[str, ProjectAsset]:
    return {asset.role: asset for asset in assets}


def count_raw_imc_files(path: Path, *, limit: int = DEFAULT_COUNT_LIMIT) -> int:
    if not path.is_dir():
        return 0
    count = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if (
                    entry.is_file()
                    and Path(entry.name).suffix.lower() in RAW_IMC_SUFFIXES
                ):
                    count += 1
                    if count >= limit:
                        break
    except OSError:
        return 0
    return count


def asset_is_ready(asset: ProjectAsset) -> bool:
    if not asset.exists:
        return False
    if asset.kind == "file":
        return True
    if asset.role == "raw_imc_files":
        return count_raw_imc_files(asset.path) > 0
    return bool(asset.file_count)


def inventory_assets(
    *,
    project_id: str,
    project_root: Path,
    config: PipelineConfig,
) -> AssetInventory:
    return AssetInventory(
        captured_at=utc_now(),
        project_id=project_id,
        project_root=project_root,
        assets=resolve_assets(config, project_root),
    )


def unexpected_top_level_paths(
    project_root: Path,
    assets: Iterable[ProjectAsset],
    *,
    config_path: Path,
) -> list[Path]:
    expected = {asset.path for asset in assets}
    expected.update({config_path, project_root / ".sbt"})
    unexpected: list[Path] = []
    try:
        entries = sorted(project_root.iterdir(), key=lambda path: path.name.lower())
    except OSError:
        return unexpected
    for path in entries:
        resolved = path.resolve(strict=False)
        if (
            resolved in expected
            or any(resolved in expected_path.parents for expected_path in expected)
            or path.name.startswith(".")
        ):
            continue
        unexpected.append(resolved)
    return unexpected


__all__ = [
    "ASSET_FIELDS",
    "RAW_IMC_SUFFIXES",
    "asset_is_ready",
    "asset_map",
    "count_raw_imc_files",
    "inspect_asset",
    "inventory_assets",
    "resolve_assets",
    "resolve_project_path",
    "unexpected_top_level_paths",
]
