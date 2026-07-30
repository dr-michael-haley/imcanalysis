"""Managed ``sbt run spatialdata`` entry point."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _runtime(argv: list[str] | None):
    from SpatialBiologyToolkit.config import load_config
    from SpatialBiologyToolkit.reporting import bootstrap_stage_reporting
    from SpatialBiologyToolkit.scripts.config_and_utils import setup_logging

    parser = argparse.ArgumentParser(
        description="Discover, plan, and optionally build a SpatialData Zarr."
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("SBT_CONFIG", "config.yaml"),
        help="Pipeline configuration path (default: SBT_CONFIG or ./config.yaml).",
    )
    arguments = parser.parse_args(argv)
    config_path = Path(arguments.config).expanduser().resolve(strict=False)
    os.environ["SBT_CONFIG"] = str(config_path)
    os.environ.setdefault("SBT_PROJECT_ROOT", str(config_path.parent))
    reporter = bootstrap_stage_reporting(os.environ.get("SBT_STAGE") or "spatialdata")
    config = load_config(config_path)
    setup_logging(config.logging.model_dump(mode="python"), "SpatialData")
    settings = config.spatialdata.model_copy(
        update={
            "roi_key": config.spatialdata.roi_key or config.general.roi_obs,
            "x_key": config.spatialdata.x_key or config.general.x_coord_obs,
            "y_key": config.spatialdata.y_key or config.general.y_coord_obs,
        }
    )
    return config, settings, reporter


def _asset_hints(settings):
    from SpatialBiologyToolkit.spatialdata import (
        HistologyAssetHint,
        IMCImageAssetHint,
        MaxFuseAssetHint,
        RegionLabelsAssetHint,
        SpatialDataAssetHints,
    )

    return SpatialDataAssetHints(
        anndata=settings.anndata_path,
        cell_masks=settings.cell_masks_folder,
        primary_images=settings.primary_images_folder,
        primary_table_name=settings.primary_table_name,
        primary_images_name=settings.primary_images_name,
        cell_masks_name=settings.cell_masks_name,
        primary_panel_name=settings.primary_panel_name,
        roi_key=settings.roi_key,
        instance_key=settings.instance_key,
        x_key=settings.x_key,
        y_key=settings.y_key,
        copy_adata=settings.copy_adata,
        additional_images=tuple(
            IMCImageAssetHint(
                name=item.name,
                folder=item.folder,
                panel_name=item.panel_name,
                channels=item.channels or None,
                reference=item.reference,
                allow_partial=item.allow_partial,
            )
            for item in settings.additional_image_panels
        ),
        histology=tuple(
            HistologyAssetHint(
                name=item.name,
                folder=item.folder,
                suffix=item.suffix,
                reference=item.reference,
                allow_partial=item.allow_partial,
                drop_alpha=item.drop_alpha,
            )
            for item in settings.histology
        ),
        region_labels=tuple(
            RegionLabelsAssetHint(
                name=item.name,
                folder=item.folder,
                value_names=item.mapping_path,
                suffix=item.suffix,
                reference=item.reference,
                allow_partial=item.allow_partial,
                value_key=item.value_key,
                name_key=item.name_key,
                mapping_roi_key=item.mapping_roi_key,
            )
            for item in settings.region_labels
        ),
        maxfuse=tuple(
            MaxFuseAssetHint(
                name=item.name,
                adata=item.adata_path,
                imc_table=item.imc_table,
                table_name=item.table_name,
                copy_adata=item.copy_adata,
            )
            for item in settings.maxfuse_tables
        ),
        attrs=settings.attrs,
        raster_chunks=settings.raster_chunks,
        scale_factors=settings.scale_factors,
        discover_unlisted_assets=settings.discover_unlisted_assets,
        include_discovered_image_panels=settings.include_discovered_image_panels,
        include_discovered_histology=settings.include_discovered_histology,
        include_discovered_maxfuse=settings.include_discovered_maxfuse,
    )


def _write_plan_report(asset_plan, *, direct_root: Path):
    from SpatialBiologyToolkit.reporting import optional_category_output_path

    tables = optional_category_output_path("tables", direct_root / "tables")
    summaries = optional_category_output_path("summaries", direct_root / "summaries")
    tables.mkdir(parents=True, exist_ok=True)
    summaries.mkdir(parents=True, exist_ok=True)

    candidates_path = tables / "spatialdata_asset_candidates.csv"
    selections_path = tables / "spatialdata_asset_selections.csv"
    discovery_issues_path = tables / "spatialdata_discovery_issues.csv"
    planner_issues_path = tables / "spatialdata_planner_issues.csv"
    summary_path = summaries / "spatialdata_plan_summary.json"

    asset_plan.inventory.candidates_frame().to_csv(candidates_path, index=False)
    asset_plan.proposal.selections_frame().to_csv(selections_path, index=False)
    asset_plan.proposal.issues_frame().to_csv(discovery_issues_path, index=False)
    if asset_plan.spatialdata_plan is None:
        import pandas as pd

        pd.DataFrame(
            columns=["severity", "code", "modality", "roi", "path", "message"]
        ).to_csv(planner_issues_path, index=False)
    else:
        asset_plan.spatialdata_plan.report.to_frame().to_csv(
            planner_issues_path, index=False
        )
    summary_path.write_text(
        json.dumps(asset_plan.summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "candidates": candidates_path,
        "selections": selections_path,
        "discovery_issues": discovery_issues_path,
        "planner_issues": planner_issues_path,
        "summary": summary_path,
    }


def run_pipeline(argv: list[str] | None = None) -> int:
    """Discover inputs and optionally create a new SpatialData Zarr."""

    from SpatialBiologyToolkit.reporting import (
        get_active_reporter,
        project_asset_path,
    )
    from SpatialBiologyToolkit.spatialdata import (
        SpatialDataDiscoveryOptions,
        build_spatialdata_from_assets,
        plan_spatialdata_from_assets,
    )

    _config, settings, reporter = _runtime(argv)
    active_reporter = get_active_reporter() or reporter
    root = project_asset_path(settings.root)
    output_path = project_asset_path(settings.output_path)
    options = SpatialDataDiscoveryOptions(
        max_depth=settings.scan_depth,
        max_entries=settings.max_scan_entries,
        sample_files=settings.sample_files,
    )
    hints = _asset_hints(settings)

    if active_reporter:
        active_reporter.add_input(
            "spatialdata_discovery_root",
            root,
            "Folder searched for explicit and likely spatial assets.",
        )
        active_reporter.add_metric("action", settings.action)
        active_reporter.add_metric("scan_depth", settings.scan_depth)
        active_reporter.add_metric("max_scan_entries", settings.max_scan_entries)

    LOGGER.info("Planning SpatialData from assets under %s", root)
    asset_plan = plan_spatialdata_from_assets(root, hints=hints, options=options)
    report_paths = _write_plan_report(
        asset_plan,
        direct_root=Path("spatialdata_report"),
    )

    if active_reporter:
        for name, path in report_paths.items():
            active_reporter.add_file(
                "summary" if name == "summary" else "table",
                path,
            )
        active_reporter.add_metric(
            "asset_candidates", len(asset_plan.inventory.candidates)
        )
        active_reporter.add_metric(
            "selected_assets", len(asset_plan.proposal.selected)
        )
        for issue in asset_plan.proposal.issues:
            if issue.severity == "warning":
                active_reporter.add_warning(issue.message)
            elif issue.severity == "info":
                active_reporter.add_note(issue.message)
        if asset_plan.spatialdata_plan is not None:
            for name, value in asset_plan.spatialdata_plan.summary().items():
                if name != "by_modality":
                    active_reporter.add_metric(f"plan.{name}", value)
            for issue in asset_plan.spatialdata_plan.report.warnings:
                active_reporter.add_warning(issue.message)

    asset_plan.raise_for_errors()
    if settings.action == "plan":
        LOGGER.info("SpatialData plan is valid; build was not requested")
        if active_reporter:
            active_reporter.add_note(
                "Plan-only execution completed. Set spatialdata.action=build "
                "after reviewing the selected assets and planner diagnostics."
            )
        return 0

    LOGGER.info("Building new SpatialData Zarr at %s", output_path)
    result = build_spatialdata_from_assets(
        root,
        output_path,
        asset_plan=asset_plan,
    )
    if active_reporter:
        active_reporter.add_asset(
            "spatialdata_zarr",
            result.output_path,
            "Reusable multimodal SpatialData Zarr created from the validated asset plan.",
        )
        for name, value in result.element_counts.items():
            active_reporter.add_metric(f"written.{name}", value)
    LOGGER.info(
        "SpatialData build complete: %s (%s)",
        result.output_path,
        ", ".join(f"{key}={value}" for key, value in result.element_counts.items()),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    return run_pipeline(argv)


if __name__ == "__main__":
    raise SystemExit(main())
