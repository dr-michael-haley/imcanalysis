"""Shared lightweight orchestration helpers for CellVision stage modules."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Sequence

from SpatialBiologyToolkit.cellvision import CellVisionPaths, resolve_cellvision_paths
from SpatialBiologyToolkit.config import PipelineConfig, load_config
from SpatialBiologyToolkit.reporting import (
    bootstrap_stage_reporting,
    get_active_reporter,
    project_asset_path,
)
from SpatialBiologyToolkit.scripts.config_and_utils import setup_logging


def load_runtime(
    component: str,
    argv: Sequence[str] | None = None,
) -> tuple[PipelineConfig, CellVisionPaths]:
    """Parse a config path, bootstrap reporting, and resolve CellVision assets."""
    stage_by_component = {
        "extract": "cellvision-extract",
        "embed": "cellvision-embed",
        "cluster": "cellvision-cluster",
        "plot": "cellvision-plot",
    }
    try:
        component_stage = stage_by_component[component]
    except KeyError as exc:
        raise ValueError(f"Unknown CellVision component: {component!r}") from exc
    parser = argparse.ArgumentParser(description=f"Run CellVision {component}.")
    parser.add_argument(
        "--config",
        default=os.environ.get("SBT_CONFIG", "config.yaml"),
        help="Pipeline config path (default: SBT_CONFIG or ./config.yaml).",
    )
    arguments = parser.parse_args(argv)
    config_path = Path(arguments.config).expanduser().resolve(strict=False)
    os.environ["SBT_CONFIG"] = str(config_path)
    os.environ.setdefault("SBT_PROJECT_ROOT", str(config_path.parent))
    bootstrap_stage_reporting(os.environ.get("SBT_STAGE") or component_stage)
    config = load_config(config_path)
    setup_logging(config.logging.model_dump(mode="python"), f"CellVision:{component}")
    paths = resolve_cellvision_paths(project_asset_path(config.cellvision.asset_folder))
    logging.info("CellVision reusable asset folder: %s", paths.root)
    return config, paths


def input_paths(config: PipelineConfig) -> tuple[Path, Path, Path]:
    """Resolve the configured AnnData, image, and mask inputs."""
    cellvision = config.cellvision
    general = config.general
    adata_path = project_asset_path(cellvision.input_adata_path or general.anndata_path)
    images_folder = project_asset_path(cellvision.images_folder or general.denoised_images_folder)
    masks_folder = project_asset_path(cellvision.masks_folder or general.masks_folder)
    return adata_path, images_folder, masks_folder


def reporter():
    """Return the active CellVision reporter when reporting is available."""
    return get_active_reporter()


__all__ = ["input_paths", "load_runtime", "reporter"]
