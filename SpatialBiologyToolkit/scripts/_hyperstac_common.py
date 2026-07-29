"""Config-driven orchestration shared by managed HyPERSTAC stages."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Sequence

from SpatialBiologyToolkit.config import PipelineConfig, load_config
from SpatialBiologyToolkit.reporting import (
    bootstrap_stage_reporting,
    category_output_path,
    get_active_reporter,
    project_asset_path,
)
from SpatialBiologyToolkit.scripts.config_and_utils import setup_logging


@contextmanager
def _argv(module_name: str, arguments: Sequence[str]) -> Iterator[None]:
    previous = sys.argv
    sys.argv = [module_name, *arguments]
    try:
        yield
    finally:
        sys.argv = previous


def _flag(arguments: list[str], name: str, enabled: bool) -> None:
    arguments.append(f"--{name}" if enabled else f"--no-{name}")


def _csv(values: Sequence[object]) -> str:
    return ",".join(str(value) for value in values)


def load_runtime(component: str) -> PipelineConfig:
    parser = argparse.ArgumentParser(description=f"Run HyPERSTAC {component}.")
    parser.add_argument(
        "--config",
        default=os.environ.get("SBT_CONFIG", "config.yaml"),
        help="Pipeline config path (default: SBT_CONFIG or ./config.yaml).",
    )
    arguments = parser.parse_args()
    config_path = Path(arguments.config).expanduser().resolve(strict=False)
    os.environ["SBT_CONFIG"] = str(config_path)
    os.environ.setdefault("SBT_PROJECT_ROOT", str(config_path.parent))
    atomic_stage = {
        "preprocess": "hyperstac-preprocess",
        "model": "hyperstac-model",
        "permutation": "hyperstac-permutation",
        "visualise": "hyperstac-visualise",
        "stability": "hyperstac-stability",
    }[component]
    bootstrap_stage_reporting(os.environ.get("SBT_STAGE") or atomic_stage)
    config = load_config(config_path)
    setup_logging(config.logging.model_dump(mode="python"), f"HyPERSTAC:{component}")
    return config


def asset_root(config: PipelineConfig) -> Path:
    return project_asset_path(config.hyperstac.asset_folder)


def source_images(config: PipelineConfig) -> Path:
    configured = (
        config.hyperstac.input_images_folder
        or config.general.denoised_images_folder
    )
    return project_asset_path(configured)


def normalised_images(config: PipelineConfig) -> Path:
    return asset_root(config) / "normalised_images"


def permutation_root(config: PipelineConfig) -> Path:
    return asset_root(config) / "permutation_sensitivity"


def _copy_report_snapshot(source: Path, category: str) -> Path:
    target = category_output_path(category) / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    reporter = get_active_reporter()
    if reporter is not None:
        reporter.add_file(
            {"figures": "figure", "tables": "table", "summaries": "summary", "files": "file"}[
                category
            ],
            target,
            f"HyPERSTAC {category} snapshot.",
        )
    return target


def _run(main: Callable[[], None], module_name: str, arguments: list[str]) -> None:
    logging.info("Running %s with %d config-derived arguments.", module_name, len(arguments))
    with _argv(module_name, arguments):
        main()


def run_preprocess() -> None:
    from SpatialBiologyToolkit.hyperstac.preprocessing import main

    config = load_runtime("preprocess")
    hs = config.hyperstac
    input_folder = source_images(config)
    output_folder = normalised_images(config)
    if not input_folder.is_dir():
        raise FileNotFoundError(f"HyPERSTAC input image folder does not exist: {input_folder}")

    arguments = [
        "--input-folder",
        str(input_folder),
        "--output-folder",
        str(output_folder),
        "--background-method",
        hs.background_method,
        "--background-fixed-value",
        str(hs.background_fixed_value),
        "--background-percentile",
        str(hs.background_percentile),
        "--presence-percentile",
        str(hs.presence_percentile),
        "--presence-threshold",
        str(hs.presence_threshold),
        "--scale-percentile",
        str(hs.scale_percentile),
        "--scale-sample-pixels",
        str(hs.scale_sample_pixels),
        "--seed",
        str(hs.seed),
    ]
    if hs.channels:
        arguments += ["--channels", _csv(hs.channels)]
    _flag(arguments, "background-subtraction", hs.background_subtraction)
    _flag(arguments, "presence-mask", hs.presence_mask)
    if hs.scale_present_only:
        arguments.append("--scale-present-only")
    if hs.tiff_compression:
        arguments += ["--compression", hs.tiff_compression]
    if hs.preprocess_overwrite:
        arguments.append("--overwrite")

    _run(main, "SpatialBiologyToolkit.hyperstac.preprocessing", arguments)
    reporter = get_active_reporter()
    if reporter is not None:
        reporter.add_input("hyperstac_source_images", input_folder, "ROI/channel TIFF images.")
        reporter.add_asset(
            "hyperstac_normalised_images",
            output_folder,
            "Reusable background-corrected and scaled ROI/channel TIFF images.",
        )
    for name in (
        "normalisation_channel_report.csv",
        "normalisation_roi_channel_report.csv",
    ):
        path = output_folder / name
        if path.is_file():
            _copy_report_snapshot(path, "tables")
    config_path = output_folder / "normalisation_config.json"
    if config_path.is_file():
        _copy_report_snapshot(config_path, "files")


def run_model() -> None:
    from SpatialBiologyToolkit.hyperstac.model import main

    config = load_runtime("model")
    hs = config.hyperstac
    images = (
        normalised_images(config)
        if hs.preprocess_images
        else source_images(config)
    )
    root = asset_root(config)
    if not images.is_dir():
        raise FileNotFoundError(
            f"HyPERSTAC modelling image folder does not exist: {images}. "
            "Run hyperstac-preprocess or set hyperstac.preprocess_images=false."
        )
    arguments = [
        "--image-folder",
        str(images),
        "--output-dir",
        str(root),
        "--patch-size",
        str(hs.patch_size),
        "--stride",
        str(hs.stride or hs.patch_size),
        "--pixel-size-um",
        str(hs.pixel_size_um),
        "--mask-threshold",
        str(hs.mask_threshold),
        "--min-mask-fraction",
        str(hs.min_mask_fraction),
        "--min-patch-signal",
        str(hs.min_patch_signal),
        "--subpatch-size",
        str(hs.subpatch_size),
        "--subpatch-signal-threshold",
        str(hs.subpatch_signal_threshold),
        "--min-tissue-subpatch-fraction",
        str(hs.min_tissue_subpatch_fraction),
        "--encoder",
        hs.encoder,
        "--epochs",
        str(hs.epochs),
        "--batch-size",
        str(hs.batch_size),
        "--learning-rate",
        str(hs.learning_rate),
        "--projector-output-size",
        str(hs.projector_output_size),
        "--seed",
        str(hs.seed),
    ]
    if hs.channels:
        arguments += ["--channels", _csv(hs.channels)]
    if hs.mask_channel:
        arguments += ["--mask-channel", hs.mask_channel]
    if hs.encoder_weights:
        arguments += ["--encoder-weights", str(project_asset_path(hs.encoder_weights))]
    if hs.representation_batch_size:
        arguments += ["--representation-batch-size", str(hs.representation_batch_size)]
    if hs.max_patches:
        arguments += ["--max-patches", str(hs.max_patches)]
    if hs.reuse_patches:
        arguments.append("--reuse-patches")
    if hs.overwrite:
        arguments.append("--overwrite")

    _run(main, "SpatialBiologyToolkit.hyperstac.model", arguments)
    reporter = get_active_reporter()
    if reporter is not None:
        reporter.add_input("hyperstac_images", images, "Normalized ROI/channel TIFF images.")
        reporter.add_asset("hyperstac_assets", root, "Reusable HyPERSTAC patches, models, and AnnData.")
        for role, relative, description in (
            ("hyperstac_representations", "imc_hyperstac_representations.h5ad", "Patch representation AnnData."),
            ("hyperstac_patch_metrics", "imc_hyperstac_patch_metrics.h5ad", "Patch metric AnnData."),
            ("hyperstac_encoder", "model/encoder.weights.h5", "Trained encoder weights."),
            ("hyperstac_patch_metadata", "patch_metadata.csv", "Patch identity and spatial metadata."),
        ):
            path = root / relative
            if path.exists():
                reporter.add_asset(role, path, description)
    for name in ("patch_metadata.csv",):
        path = root / name
        if path.is_file():
            _copy_report_snapshot(path, "tables")
    for name in ("run_config.json", "channels.json"):
        path = root / name
        if path.is_file():
            _copy_report_snapshot(path, "files")


def run_permutation() -> None:
    from SpatialBiologyToolkit.hyperstac.permutation import main

    config = load_runtime("permutation")
    hs = config.hyperstac
    root = asset_root(config)
    output = permutation_root(config)
    arguments = [
        "--hyperstac-output-dir",
        str(root),
        "--output-dir",
        str(output),
        "--batch-size",
        str(hs.permutation_batch_size),
        "--n-shuffle-repeats",
        str(hs.permutation_shuffle_repeats),
        "--shuffle-pixels",
        hs.permutation_shuffle_pixels,
        "--seed",
        str(hs.seed),
    ]
    if hs.permutation_channels:
        arguments += ["--channels", _csv(hs.permutation_channels)]
    _flag(
        arguments,
        "include-all-channel-perturbations",
        hs.permutation_include_all_channels,
    )
    if hs.permutation_recompute_original:
        arguments.append("--recompute-original")
    if hs.permutation_write_wide_csv:
        arguments.append("--write-wide-csv")
    _run(main, "SpatialBiologyToolkit.hyperstac.permutation", arguments)

    reporter = get_active_reporter()
    if reporter is not None:
        reporter.add_input("hyperstac_assets", root, "HyPERSTAC patches, representation, and encoder.")
        reporter.add_asset(
            "hyperstac_permutation",
            output / "imc_permutation_sensitivity.h5ad",
            "Reusable patch-by-perturbation sensitivity AnnData.",
        )
    for path in output.glob("*.csv*"):
        if path.is_file():
            _copy_report_snapshot(path, "tables")
    config_path = output / "permutation_run_config.json"
    if config_path.is_file():
        _copy_report_snapshot(config_path, "files")


def visualisation_output() -> Path:
    return category_output_path("files") / "hyperstac_visualisation"


def run_visualisation() -> None:
    from SpatialBiologyToolkit.hyperstac.visualisation import main

    config = load_runtime("visualise")
    hs = config.hyperstac
    root = asset_root(config)
    output = visualisation_output()
    arguments = [
        "--hyperstac-output-dir",
        str(root),
        "--output-dir",
        str(output),
        "--normalisation-report-dir",
        str(normalised_images(config)),
        "--cluster-col",
        hs.cluster_col,
        "--cluster-col-search",
        hs.cluster_col_search,
        "--leiden-resolution",
        _csv(hs.leiden_resolutions),
        "--scanpy-n-neighbors",
        _csv(hs.n_neighbors),
        "--scanpy-n-pcs",
        _csv(hs.n_pcs),
        "--scanpy-min-dist",
        str(hs.umap_min_dist),
        "--max-umap-channels",
        str(hs.max_umap_channels),
        "--max-permutation-umaps",
        str(hs.max_permutation_umaps),
        "--max-roi-maps",
        str(hs.max_roi_maps),
        "--patches-per-cluster",
        str(hs.patches_per_cluster),
        "--gallery-top-markers",
        str(hs.gallery_top_markers),
        "--gallery-cols",
        str(hs.gallery_columns),
        "--gallery-vmax",
        str(hs.gallery_vmax),
        "--gallery-contrast-percentile",
        str(hs.gallery_contrast_percentile),
        "--gallery-marker-source",
        hs.gallery_marker_source,
        "--gallery-permutation-type",
        hs.gallery_permutation_type,
        "--de-method",
        hs.differential_expression_method,
        "--seed",
        str(hs.seed),
    ]
    _flag(arguments, "run-cluster-scan", hs.run_cluster_scan)
    _flag(arguments, "write-clustered-adata", hs.write_clustered_adata)
    _flag(arguments, "replace-existing-cluster-scan", hs.replace_existing_cluster_scan)
    _flag(arguments, "spatial-cluster-maps", hs.write_spatial_cluster_maps)
    _flag(arguments, "split-channel-gallery", hs.split_channel_gallery)
    _flag(arguments, "gallery-auto-contrast", hs.gallery_auto_contrast)
    _run(main, "SpatialBiologyToolkit.hyperstac.visualisation", arguments)

    reporter = get_active_reporter()
    if reporter is not None:
        reporter.add_input("hyperstac_assets", root, "Representation, patch metrics, and optional perturbation assets.")
        reporter.add_file(
            "file",
            output,
            "Interlinked HyPERSTAC visualisation tree retained as one report attachment.",
        )
        summary = output / "all_cluster_visualisation_summary.csv"
        if summary.is_file():
            reporter.add_metric("cluster_reports", max(0, sum(1 for _ in summary.open(encoding="utf-8")) - 1))


def latest_stage_output(stage: str) -> Path:
    current_stage = os.environ.get("SBT_STAGE")
    current_output = Path(
        os.environ.get("SBT_STAGE_OUTPUT_DIR")
        or os.environ.get("SBT_OUTPUT_DIR")
        or "."
    ).resolve(strict=False)
    if current_stage == "hyperstac-full":
        return current_output

    from SpatialBiologyToolkit.pipeline.executions import (
        execution_output_path,
        load_execution_index,
    )
    from SpatialBiologyToolkit.pipeline.project import load_project

    context = load_project(os.environ.get("SBT_PROJECT_ROOT"))
    matches = [
        record
        for record in load_execution_index(context, require_exists=True).executions
        if record.stage == stage and record.status not in {"failed", "cancelled", "blocked"}
    ]
    if not matches:
        raise FileNotFoundError(
            f"No usable managed {stage!r} execution was found for this project."
        )
    return execution_output_path(context, max(matches, key=lambda item: item.execution_id))


def run_stability() -> None:
    from SpatialBiologyToolkit.hyperstac.stability import main

    config = load_runtime("stability")
    hs = config.hyperstac
    visual_root = latest_stage_output("hyperstac-visualise") / "files" / "hyperstac_visualisation"
    cox_root = latest_stage_output("cox") / "files" / "cox"
    output = category_output_path("files") / "hyperstac_stability"
    arguments = [
        "--visualisation-dir",
        str(visual_root),
        "--survival-dir",
        str(cox_root),
        "--output-dir",
        str(output),
        "--top-markers",
        str(hs.stability_top_markers),
        "--max-heatmap-markers",
        str(hs.stability_max_heatmap_markers),
        "--max-signature-markers",
        str(hs.stability_max_signature_markers),
        "--environment-distance-threshold",
        str(hs.stability_environment_distance_threshold),
        "--marker-enrichment-min-z",
        str(hs.stability_marker_enrichment_min_z),
        "--effect-threshold",
        str(hs.stability_effect_threshold),
        "--permutation-type",
        hs.stability_permutation_type,
        "--max-report-items",
        str(hs.stability_max_report_items),
        "--figure-format",
        hs.stability_figure_format,
        "--figure-dpi",
        str(hs.stability_figure_dpi),
        "--cluster-bubble-metric",
        hs.stability_cluster_bubble_metric,
        "--cluster-bubble-size-min",
        str(hs.stability_cluster_bubble_size_min),
        "--cluster-bubble-size-scale",
        str(hs.stability_cluster_bubble_size_scale),
        "--environment-bubble-metric",
        hs.stability_environment_bubble_metric,
        "--environment-bubble-size-min",
        str(hs.stability_environment_bubble_size_min),
        "--environment-bubble-size-scale",
        str(hs.stability_environment_bubble_size_scale),
        "--environment-color-metric",
        hs.stability_environment_color_metric,
        "--environment-color-quantile",
        str(hs.stability_environment_color_quantile),
    ]
    _flag(
        arguments,
        "cluster-bubble-log-scale",
        hs.stability_cluster_bubble_log_scale,
    )
    _flag(
        arguments,
        "environment-bubble-log-scale",
        hs.stability_environment_bubble_log_scale,
    )
    _flag(
        arguments,
        "per-clustering-html-reports",
        hs.stability_per_clustering_html,
    )
    _run(main, "SpatialBiologyToolkit.hyperstac.stability", arguments)
    reporter = get_active_reporter()
    if reporter is not None:
        reporter.add_input("hyperstac_visualisation", visual_root, "Matched Leiden visualisation reports.")
        reporter.add_input("cox_reports", cox_root, "Matched case-level Cox reports.")
        reporter.add_file(
            "file",
            output,
            "Cross-Leiden environment, marker, perturbation, and survival stability report tree.",
        )


__all__ = [
    "asset_root",
    "latest_stage_output",
    "normalised_images",
    "permutation_root",
    "run_model",
    "run_permutation",
    "run_preprocess",
    "run_stability",
    "run_visualisation",
    "source_images",
    "visualisation_output",
]
