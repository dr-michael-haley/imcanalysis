"""Pipeline stage for empirical neighbour-attributable marker signal analysis."""

from __future__ import annotations

import argparse
import logging
import os
import uuid
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def _runtime(argv: list[str] | None):
    from SpatialBiologyToolkit.config import load_config
    from SpatialBiologyToolkit.reporting import bootstrap_stage_reporting
    from SpatialBiologyToolkit.scripts.config_and_utils import setup_logging

    parser = argparse.ArgumentParser(
        description="Calculate empirical neighbour-attributable marker signal scores."
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
    reporter = bootstrap_stage_reporting(os.environ.get("SBT_STAGE") or "neighsig")
    config = load_config(config_path)
    setup_logging(config.logging.model_dump(mode="python"), "NeighbourSignal")
    settings = config.neighbour_signal.model_copy(
        update={
            "population_obs": (
                config.neighbour_signal.population_obs
                or config.general.population_obs_primary
            )
        }
    )
    return config, settings, reporter


def _atomic_h5ad(adata, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp.h5ad")
    try:
        adata.write_h5ad(temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_pipeline(argv: list[str] | None = None) -> int:
    """Run both halo passes, write a separate AnnData, and create QC outputs."""

    from SpatialBiologyToolkit.cellvision import discover_roi_inputs, select_source_cells
    from SpatialBiologyToolkit.neighbour_signal import (
        HaloParameters,
        build_output_anndata,
        run_neighbour_signal_analysis,
    )
    from SpatialBiologyToolkit.neighbour_signal_reports import (
        generate_neighbour_signal_report,
    )
    from SpatialBiologyToolkit.reporting import (
        get_active_reporter,
        optional_category_output_path,
        project_asset_path,
    )
    from SpatialBiologyToolkit.scripts.config_and_utils import read_h5ad_compat

    config, settings, bootstrap_reporter = _runtime(argv)
    reporter = get_active_reporter() or bootstrap_reporter
    if not settings.enabled:
        LOGGER.info("Neighbour signal analysis is disabled; nothing to do")
        if reporter:
            reporter.add_note(
                "Stage skipped because neighbour_signal.enabled is false."
            )
        return 0

    input_path = project_asset_path(config.general.anndata_path)
    images_folder = project_asset_path(config.general.raw_images_folder)
    masks_folder = project_asset_path(config.general.masks_folder)
    output_path = project_asset_path(settings.output_adata_path)
    for label, path, expected in (
        ("input AnnData", input_path, "file"),
        ("raw ROI/channel images", images_folder, "directory"),
        ("segmentation masks", masks_folder, "directory"),
    ):
        exists = path.is_file() if expected == "file" else path.is_dir()
        if not exists:
            raise FileNotFoundError(f"Neighbour signal {label} not found: {path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "neighbour_signal.output_adata_path must differ from general.anndata_path"
        )

    if reporter:
        reporter.add_input(
            "anndata",
            input_path,
            "Source AnnData providing exact cell/marker order, exemplar labels, and independent original X values.",
        )
        reporter.add_input(
            "raw_images",
            images_folder,
            "Raw per-marker ROI TIFFs used for all source, background, profile, and score calculations.",
        )
        reporter.add_input(
            "masks",
            masks_folder,
            "Original labelled segmentation masks used for source geometry and target pixels.",
        )

    LOGGER.info("Reading source AnnData: %s", input_path)
    adata = read_h5ad_compat(input_path)
    roi_obs = config.general.roi_obs
    identity = select_source_cells(
        adata,
        roi_obs=roi_obs,
        object_id_obs=settings.object_id_obs,
    )
    roi_inputs, resolved_markers = discover_roi_inputs(
        images_folder,
        masks_folder,
        identity,
        roi_obs=roi_obs,
        markers=[str(value) for value in adata.var_names],
    )
    if resolved_markers != [str(value) for value in adata.var_names]:
        raise RuntimeError("Resolved raw marker order differs from AnnData.var_names")

    parameters = HaloParameters(
        max_halo_px=settings.max_halo_px,
        source_anchor_dilation_px=settings.source_anchor_dilation_px,
        source_anchor_quantile=settings.source_anchor_quantile,
        min_exemplars=settings.min_exemplars,
        source_threshold_quantile=settings.source_threshold_quantile,
        halo_aggregation=settings.halo_aggregation,
    )
    result = run_neighbour_signal_analysis(
        adata,
        roi_inputs,
        identity,
        roi_obs=roi_obs,
        object_id_obs=settings.object_id_obs,
        exemplar_obs=settings.exemplar_obs,
        parameters=parameters,
        n_jobs=settings.n_jobs,
    )
    provenance_parameters = settings.model_dump(mode="python")
    provenance_parameters.update(
        {
            "input_adata_path": str(input_path),
            "raw_images_folder": str(images_folder),
            "masks_folder": str(masks_folder),
            "roi_obs": roi_obs,
            "marker_axis": "adata.var_names",
        }
    )
    output = build_output_anndata(
        adata,
        result,
        parameters=provenance_parameters,
        calculate_classic_intensities=settings.calculate_classic_intensities,
        high_risk_threshold=settings.high_risk_threshold,
    )

    direct_root = Path("neighbour_signal_report")
    figures_dir = optional_category_output_path(
        "figures", direct_root / "figures"
    ) / "neighbour_signal"
    tables_dir = optional_category_output_path(
        "tables", direct_root / "tables"
    ) / "neighbour_signal"
    summaries_dir = optional_category_output_path(
        "summaries", direct_root / "summaries"
    ) / "neighbour_signal"
    report = generate_neighbour_signal_report(
        output,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        summaries_dir=summaries_dir,
        output_adata_path=output_path,
        qc_markers=settings.qc_markers,
        max_qc_markers=settings.max_qc_markers,
        population_obs=settings.population_obs,
    )
    _atomic_h5ad(output, output_path)
    LOGGER.info(
        "Neighbour signal analysis complete: %d cells, %d markers, %d learned profiles -> %s",
        output.n_obs,
        output.n_vars,
        int(output.var["halo_profile_available"].sum()),
        output_path,
    )

    if reporter:
        reporter.add_asset(
            "neighbour_signal_anndata",
            output_path,
            "Cell- and marker-aligned AnnData whose X contains Neighbour-Attributable Fractions.",
        )
        for path in report.figures:
            reporter.add_file("figure", path)
        for path in report.tables:
            reporter.add_file("table", path)
        for path in report.summaries:
            reporter.add_file("summary", path)
        for warning in [*result.warnings, *report.warnings]:
            reporter.add_warning(warning)
        for name, value in report.metrics.items():
            reporter.add_metric(name, value)
        reporter.add_metric("roi_workers", result.worker_usage.effective)
        reporter.add_metric("cpu_limit", result.worker_usage.cpu_limit)
        reporter.add_metric("rois", len(roi_inputs))
        reporter.add_metric(
            "unknown_exemplar_marker_values",
            len(result.unknown_exemplar_values),
        )
        reporter.add_note(
            "Neighbour-Attributable Fraction is a spatial explainability/QC score, not a calibrated probability or proof of artefact."
        )
        reporter.add_note(
            "Input AnnData.X was not used in the halo calculation and is preserved in layers['original_X']."
        )
    return 0


def main() -> int:
    return run_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())
