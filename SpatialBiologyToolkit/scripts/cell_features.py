"""Managed cohort-first synthetic feature extraction for a napari_sbt experiment."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _active_experiment_path(settings, project_asset_path) -> Path:
    if not settings.active_experiment:
        raise ValueError(
            "napari_sbt.active_experiment is not configured. Confirm an experiment "
            "in the Napari Setup tab before running cellfeat."
        )
    configured = Path(settings.active_experiment).expanduser()
    if configured.is_absolute():
        candidate = configured
    else:
        root = project_asset_path(settings.experiment_folder)
        candidate = root / configured
    if candidate.name == "experiment.yaml":
        candidate = candidate.parent
    candidate = candidate.resolve(strict=False)
    if not (candidate / "experiment.yaml").is_file():
        raise FileNotFoundError(
            f"Active napari_sbt experiment manifest not found: "
            f"{candidate / 'experiment.yaml'}"
        )
    return candidate


def run_pipeline() -> int:
    from SpatialBiologyToolkit.config import load_config
    from SpatialBiologyToolkit.napari_sbt.storage import load_experiment
    from SpatialBiologyToolkit.napari_sbt.worker import run_feature_build
    from SpatialBiologyToolkit.reporting import (
        bootstrap_stage_reporting,
        get_active_reporter,
        project_asset_path,
    )

    reporter = bootstrap_stage_reporting("cellfeat")
    config_path = Path(
        os.environ.get("SBT_CONFIG", "config.yaml")
    ).expanduser().resolve(strict=False)
    config = load_config(config_path)
    settings = config.napari_sbt
    experiment_path = _active_experiment_path(settings, project_asset_path)
    manifest, _paths = load_experiment(experiment_path)
    result = run_feature_build(experiment_path, workers=settings.worker_count)
    active = get_active_reporter() or reporter
    if active:
        active.add_input(
            "napari_sbt_experiment",
            experiment_path / "experiment.yaml",
            "Versioned experiment manifest with a frozen cohort",
        )
        active.add_input(
            "masks",
            Path(manifest.masks_folder),
            "Original full segmentation used read-only for scientific context",
        )
        if manifest.anndata_path:
            active.add_input(
                "anndata",
                Path(manifest.anndata_path),
                "Identity source read without in-place modification",
            )
        active.add_asset(
            "napari_sbt_experiments",
            experiment_path,
            "Reusable cohort-only feature assets and provenance",
        )
        active.add_file("table", result.feature_table)
        active.add_file("table", result.feature_dictionary)
        active.add_file("table", result.coverage_report)
        active.add_file("table", result.failed_rois)
        active.add_file("summary", result.manifest)
        active.add_metric("selected_cells", result.eligible_cells)
        active.add_metric("target_eligible_cells", result.target_eligible_cells)
        active.add_metric("total_cells", manifest.cell_scope.total_cell_count)
        active.add_metric("represented_rois", result.represented_rois)
        active.add_metric(
            "target_represented_rois", result.target_represented_rois
        )
        active.add_metric("experiment_mode", manifest.experiment_mode)
        active.add_metric("resumed_rois", result.skipped_rois)
        active.add_metric("feature_count", result.feature_count)
        active.add_metric("erosion_losses", result.erosion_losses)
        active.add_metric("failed_rois", result.failures)
        active.add_metric("elapsed_seconds", round(result.elapsed_seconds, 3))
        for warning in result.warnings:
            active.add_warning(warning)
    LOGGER.info(
        "cellfeat completed: %d/%d eligible cells across %d/%d ROIs, %d "
        "features, %d failures",
        result.eligible_cells,
        result.target_eligible_cells,
        result.represented_rois,
        result.target_represented_rois,
        result.feature_count,
        result.failures,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        from SpatialBiologyToolkit.napari_sbt.worker import main as worker_main

        return worker_main(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [CellFeatures] %(message)s",
    )
    return run_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())
