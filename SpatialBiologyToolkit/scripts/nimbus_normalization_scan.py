"""Run a pre-AnnData Nimbus normalization-value sensitivity scan."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _runtime(argv: list[str] | None):
    from SpatialBiologyToolkit.config import load_config
    from SpatialBiologyToolkit.reporting import bootstrap_stage_reporting
    from SpatialBiologyToolkit.scripts.config_and_utils import setup_logging

    parser = argparse.ArgumentParser(
        description=(
            "Scan Nimbus normalization Vmax values before cell-table, AnnData, or "
            "clustering generation."
        )
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
    reporter = bootstrap_stage_reporting(os.environ.get("SBT_STAGE") or "nimbus-scan")
    config = load_config(config_path)
    setup_logging(config.logging.model_dump(mode="python"), "NimbusNormalizationScan")
    return config, config.nimbus_normalization_scan, reporter


def _absolute_general_config(config):
    """Resolve the image/mask metadata paths used by legacy Nimbus discovery."""

    from SpatialBiologyToolkit.reporting import project_asset_path

    updates = {
        name: str(project_asset_path(getattr(config.general, name)))
        for name in (
            "metadata_folder",
            "masks_folder",
            "raw_images_folder",
            "denoised_images_folder",
        )
    }
    return config.general.model_copy(update=updates)


def _load_baseline_normalization(
    path: Path,
    markers: Sequence[str],
    *,
    require_all: bool = True,
) -> tuple[dict[str, float], dict[str, float]]:
    from SpatialBiologyToolkit.nimbus_normalization import (
        load_normalization_file,
        resolve_normalization_parameters,
    )

    loaded = load_normalization_file(path)
    resolved = resolve_normalization_parameters(
        loaded,
        markers,
        require_all=require_all,
    )
    return (
        {marker: entry.vmax for marker, entry in resolved.items()},
        {marker: entry.lower_threshold for marker, entry in resolved.items()},
    )


def _pixel_summary_rows(
    dataset,
    marker: str,
    vmax_values: Sequence[float],
    rois: Sequence[str],
    *,
    lower_threshold: float,
):
    import numpy as np

    rows: list[dict[str, object]] = []
    for roi in rois:
        mask = dataset.get_segmentation(roi) > 0
        raw = dataset.get_channel(roi, marker).astype(float)
        values = raw[mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(
                f"Marker {marker!r} has no finite in-mask pixels in ROI {roi!r}."
            )
        for vmax in vmax_values:
            rows.append(
                {
                    "marker": marker,
                    "vmax": float(vmax),
                    "roi": str(roi),
                    "masked_pixel_count": int(values.size),
                    "saturated_pixel_fraction": float(np.mean(values >= float(vmax))),
                    "lower_threshold": float(lower_threshold),
                    "below_lower_threshold_fraction": float(
                        np.mean(values <= float(lower_threshold))
                    ),
                }
            )
    return rows


def _build_marker_roi_selections(
    dataset,
    markers: Sequence[str],
    available_rois: Sequence[str],
    lower_thresholds: Mapping[str, float],
    *,
    requested_rois: Sequence[str] | None,
    strategy: str,
    max_rois: int,
    random_seed: int,
):
    """Select scan ROIs and return an auditable marker/ROI selection table."""

    import pandas as pd

    from SpatialBiologyToolkit.nimbus_normalization_scan import (
        rank_rois_by_expression,
        select_rois_across_expression_range,
        select_scan_rois,
        summarize_intracellular_expression,
    )

    rois = list(dict.fromkeys(str(roi) for roi in available_rois))
    if requested_rois or strategy == "random":
        common_selection = select_scan_rois(
            rois,
            requested_rois=requested_rois,
            max_rois=max_rois,
            random_seed=random_seed,
        )
        method = "explicit" if requested_rois else "random"
        selected_set = set(common_selection)
        rows = [
            {
                "marker": marker,
                "roi": roi,
                "selection_strategy": method,
                "lower_threshold": float(lower_thresholds[marker]),
                "masked_pixel_count": None,
                "above_background_pixel_count": None,
                "above_background_fraction": None,
                "mean_above_background": None,
                "expression_rank": None,
                "expression_quantile": None,
                "selected": roi in selected_set,
            }
            for marker in markers
            for roi in rois
        ]
        return (
            {marker: list(common_selection) for marker in markers},
            pd.DataFrame(rows),
        )

    if strategy != "marker_expression_range":
        raise ValueError(f"Unsupported Nimbus scan ROI selection strategy: {strategy!r}.")

    marker_selections: dict[str, list[str]] = {}
    rows: list[dict[str, object]] = []
    for marker in markers:
        marker_rows: list[dict[str, object]] = []
        scores: dict[str, float] = {}
        lower_threshold = float(lower_thresholds[marker])
        for roi in rois:
            summary = summarize_intracellular_expression(
                dataset.get_channel(roi, marker),
                dataset.get_segmentation(roi),
                background_value=lower_threshold,
            )
            scores[roi] = summary.mean_above_background
            marker_rows.append(
                {
                    "marker": marker,
                    "roi": roi,
                    "selection_strategy": strategy,
                    "lower_threshold": lower_threshold,
                    "masked_pixel_count": summary.masked_pixel_count,
                    "above_background_pixel_count": summary.above_background_pixel_count,
                    "above_background_fraction": summary.above_background_fraction,
                    "mean_above_background": summary.mean_above_background,
                }
            )

        selected = select_rois_across_expression_range(scores, max_rois=max_rois)
        marker_selections[marker] = selected
        selected_set = set(selected)
        ranked = rank_rois_by_expression(scores)
        rank_lookup = {roi: rank for rank, roi in enumerate(ranked)}
        denominator = max(len(ranked) - 1, 1)
        for row in marker_rows:
            roi = str(row["roi"])
            rank = rank_lookup[roi]
            row["expression_rank"] = rank
            row["expression_quantile"] = float(rank / denominator)
            row["selected"] = roi in selected_set
        rows.extend(marker_rows)

    return marker_selections, pd.DataFrame(rows)


def _write_summary_markdown(
    path: Path,
    *,
    recommendations,
    selected_rois_by_marker: Mapping[str, Sequence[str]],
    roi_selection_strategy: str,
    primary_threshold: float,
    baseline_source: str,
) -> Path:
    review_count = int(recommendations["manual_review_required"].sum())
    cliff_count = int(recommendations["cliff_detected"].sum())
    unique_rois = {
        roi for selected_rois in selected_rois_by_marker.values() for roi in selected_rois
    }
    roi_marker_selections = sum(
        len(selected_rois) for selected_rois in selected_rois_by_marker.values()
    )
    lines = [
        "# Nimbus normalization scan summary",
        "",
        f"- Markers scanned: {len(recommendations)}",
        f"- ROI selection strategy: {roi_selection_strategy}",
        f"- Unique ROIs used for repeated inference: {len(unique_rois)}",
        f"- Marker/ROI selections: {roi_marker_selections}",
        f"- Primary Nimbus positive-score threshold: {primary_threshold:g}",
        f"- Baseline source: {baseline_source}",
        f"- Markers flagged for manual review: {review_count}",
        f"- Markers with an adjacent Vmax cliff: {cliff_count}",
        "",
        "## Interpretation",
        "",
        (
            "The suggested value is the scanned candidate nearest the baseline within a "
            "locally stable, low-saturation range with few individual call flips. If no "
            "candidate meets all tolerances, the least-sensitive candidate is reported "
            "and flagged. This is a computational sensitivity recommendation, not a "
            "biological positive/negative calibration."
        ),
        "",
        (
            "A cliff means that the primary positive-cell fraction changed sharply "
            "between two adjacent Vmax values. Review those markers carefully, ideally "
            "using known positive and negative tissue or cell controls. The suggested "
            "CSV is review-only; this stage does not overwrite the Nimbus normalization "
            "dictionary."
        ),
        "",
        "## Marker recommendations",
        "",
        "| Marker | Baseline | Lower | Suggested | Stable range | Status | Review? |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in recommendations.itertuples(index=False):
        if row.stable_vmax_min == row.stable_vmax_min:
            stable_range = f"{row.stable_vmax_min:.4g}–{row.stable_vmax_max:.4g}"
        else:
            stable_range = "none"
        lines.append(
            f"| {row.marker} | {row.baseline_vmax:.4g} | {row.lower_threshold:.4g} | "
            f"{row.suggested_vmax:.4g} | {stable_range} | {row.recommendation_status} | "
            f"{'yes' if row.manual_review_required else 'no'} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_pipeline(argv: list[str] | None = None) -> int:
    """Run marker-wise Vmax scans and write a managed diagnostic report."""

    config, settings, bootstrap_reporter = _runtime(argv)

    import numpy as np
    import pandas as pd
    from nimbus_inference.nimbus import Nimbus

    from SpatialBiologyToolkit.nimbus_normalization_scan import (
        analyze_normalization_scan,
        build_vmax_grid,
        plot_marker_scan,
        resolve_marker_baseline_vmax,
        resolve_marker_lower_thresholds,
        resolve_scan_marker_inputs,
        resolve_scan_markers,
        safe_marker_filename,
        write_suggested_normalization_dict,
    )
    from SpatialBiologyToolkit.reporting import (
        get_active_reporter,
        optional_category_output_path,
        project_asset_path,
    )
    from SpatialBiologyToolkit.scripts.segmentation_nimbus import (
        ToolkitNimbusDataset,
        _coerce_optional_area_bound,
        _discover_masks,
        _filter_rois_by_metadata,
        _load_panel,
        _predict_fovs_with_padding,
        _resolve_channel_paths,
    )

    reporter = get_active_reporter() or bootstrap_reporter
    general = _absolute_general_config(config)
    nimbus = config.nimbus

    metadata_folder = Path(general.metadata_folder)
    panel_path = metadata_folder / "panel.csv"
    metadata_path = metadata_folder / "metadata.csv"
    if not panel_path.is_file():
        raise FileNotFoundError(f"Nimbus panel table not found: {panel_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Nimbus ROI metadata table not found: {metadata_path}")
    panel = _load_panel(metadata_folder, nimbus)
    mask_lookup = _discover_masks(Path(general.masks_folder), nimbus.mask_extensions)
    if not mask_lookup:
        raise FileNotFoundError(f"No Nimbus mask files found in {general.masks_folder}")
    rois = _filter_rois_by_metadata(mask_lookup, metadata_path)
    (
        valid_rois,
        channel_paths,
        roi_image_roots,
        missing_summary,
        _expected_channels,
        channels_for_model,
    ) = _resolve_channel_paths(rois, panel, general, nimbus)
    if not valid_rois or not channels_for_model:
        raise ValueError(
            "No common ROI/marker inputs are available for the Nimbus scan."
        )
    if missing_summary:
        LOGGER.warning("Channels missing for some ROIs: %s", missing_summary)

    baseline_path = (
        project_asset_path(settings.baseline_normalization_dict_path)
        if settings.baseline_normalization_dict_path
        else None
    )
    file_baselines: dict[str, float] = {}
    file_lower_thresholds: dict[str, float] = {}
    markers, file_baselines, file_lower_thresholds = resolve_scan_marker_inputs(
        channels_for_model,
        settings.markers,
        marker_parameters_path=baseline_path,
    )
    if file_baselines:
        LOGGER.info(
            "Scanning the %d marker(s) defined by %s.",
            len(markers),
            baseline_path,
        )
    for config_key, configured_mapping in (
        ("marker_baseline_vmax", settings.marker_baseline_vmax),
        ("marker_lower_thresholds", settings.marker_lower_thresholds),
        ("marker_vmax_values", settings.marker_vmax_values),
    ):
        configured_markers = (
            resolve_scan_markers(channels_for_model, list(configured_mapping))
            if configured_mapping
            else []
        )
        outside_scope = sorted(set(configured_markers) - set(markers))
        if outside_scope:
            raise ValueError(
                f"{config_key} values were supplied for marker(s) outside the scan "
                f"scope: {outside_scope}."
            )
    min_cell_area = _coerce_optional_area_bound(
        "nimbus.min_cell_area", nimbus.min_cell_area
    )
    max_cell_area = _coerce_optional_area_bound(
        "nimbus.max_cell_area", nimbus.max_cell_area
    )
    if (
        min_cell_area is not None
        and max_cell_area is not None
        and min_cell_area > max_cell_area
    ):
        raise ValueError(
            "nimbus.min_cell_area cannot be greater than nimbus.max_cell_area"
        )

    direct_root = Path("nimbus_normalization_scan_report")
    figures_dir = (
        optional_category_output_path("figures", direct_root / "figures")
        / "nimbus_normalization_scan"
    )
    tables_dir = (
        optional_category_output_path("tables", direct_root / "tables")
        / "nimbus_normalization_scan"
    )
    summaries_dir = (
        optional_category_output_path("summaries", direct_root / "summaries")
        / "nimbus_normalization_scan"
    )
    files_dir = (
        optional_category_output_path("files", direct_root / "files")
        / "nimbus_normalization_scan"
    )
    for directory in (figures_dir, tables_dir, summaries_dir, files_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset = ToolkitNimbusDataset(
        fov_paths=[roi_image_roots[roi] for roi in valid_rois],
        channels=channels_for_model,
        channel_paths=channel_paths,
        mask_lookup={roi: mask_lookup[roi] for roi in valid_rois},
        suffix=".tiff",
        magnification=nimbus.dataset_magnification,
        output_dir=str(files_dir),
        qc_folder=str(figures_dir),
        normalization_jobs=1,
        clip_values=tuple(nimbus.normalization_clip),
        normalization_min_value=nimbus.normalization_min_value,
        mask_boundary_offset_pixels=nimbus.mask_boundary_offset_pixels,
        min_cell_area=min_cell_area,
        max_cell_area=max_cell_area,
    )

    supplied_baselines = resolve_marker_baseline_vmax(
        markers, settings.marker_baseline_vmax
    )
    fallback_markers = [marker for marker in markers if marker not in supplied_baselines]
    fallback_baselines: dict[str, float] = {}
    baseline_file_kind = (
        "scan parameter CSV"
        if baseline_path is not None and baseline_path.suffix.casefold() == ".csv"
        else "normalization dictionary"
    )
    baseline_sources = {
        marker: "nimbus_normalization_scan.marker_baseline_vmax"
        for marker in supplied_baselines
    }
    if baseline_path is not None:
        if not file_baselines:
            file_baselines, file_lower_thresholds = _load_baseline_normalization(
                baseline_path,
                markers,
                require_all=False,
            )
        missing_file_baselines = [
            marker for marker in fallback_markers if marker not in file_baselines
        ]
        if missing_file_baselines:
            raise ValueError(
                f"Baseline normalization file {baseline_path} is missing Vmax values "
                f"for scanned markers not covered by marker_baseline_vmax: "
                f"{missing_file_baselines}."
            )
        fallback_baselines = {
            marker: file_baselines[marker] for marker in fallback_markers
        }
        baseline_sources.update(
            {
                marker: f"{baseline_file_kind}: {baseline_path}"
                for marker in fallback_markers
            }
        )
        fallback_source = str(baseline_path)
    elif fallback_markers:
        fallback_baselines = dataset.compute_normalization_values(
            quantile=nimbus.normalization_quantile,
            channels=fallback_markers,
        )
        computed_source = (
            f"computed from all {len(valid_rois)} usable ROIs at in-mask quantile "
            f"{nimbus.normalization_quantile:g}"
        )
        baseline_sources.update(
            {marker: computed_source for marker in fallback_markers}
        )
        fallback_source = computed_source
    else:
        fallback_source = "not required"
    baselines = {
        marker: (
            supplied_baselines[marker]
            if marker in supplied_baselines
            else fallback_baselines[marker]
        )
        for marker in markers
    }
    if supplied_baselines and fallback_markers:
        baseline_source = (
            f"marker_baseline_vmax for {len(supplied_baselines)} marker(s); "
            f"remaining marker(s): {fallback_source}"
        )
    elif supplied_baselines:
        baseline_source = "nimbus_normalization_scan.marker_baseline_vmax"
    else:
        baseline_source = fallback_source

    configured_lower_thresholds = resolve_marker_lower_thresholds(
        markers, settings.marker_lower_thresholds
    )
    lower_thresholds = {
        marker: configured_lower_thresholds.get(
            marker, file_lower_thresholds.get(marker, 0.0)
        )
        for marker in markers
    }
    lower_threshold_sources = {
        marker: (
            "nimbus_normalization_scan.marker_lower_thresholds"
            if marker in configured_lower_thresholds
            else (
                f"{baseline_file_kind}: {baseline_path}"
                if marker in file_lower_thresholds
                else "default zero"
            )
        )
        for marker in markers
    }
    dataset.normalization_lower_thresholds = dict(lower_thresholds)
    vmax_grids = {
        marker: build_vmax_grid(
            marker,
            baselines[marker],
            factors=settings.vmax_factors,
            marker_vmax_values=settings.marker_vmax_values,
            lower_threshold=lower_thresholds[marker],
        )
        for marker in markers
    }
    selected_rois_by_marker, roi_selection = _build_marker_roi_selections(
        dataset,
        markers,
        valid_rois,
        lower_thresholds,
        requested_rois=settings.rois,
        strategy=settings.roi_selection_strategy,
        max_rois=settings.max_rois,
        random_seed=settings.random_seed,
    )
    roi_selection_method = (
        "explicit" if settings.rois else settings.roi_selection_strategy
    )
    unique_scan_rois = sorted(
        {
            roi
            for selected_rois in selected_rois_by_marker.values()
            for roi in selected_rois
        }
    )
    LOGGER.info(
        "Selected %d unique ROI(s) across %d marker/ROI selections using %s.",
        len(unique_scan_rois),
        sum(len(rois) for rois in selected_rois_by_marker.values()),
        roi_selection_method,
    )

    if reporter:
        reporter.add_input(
            "metadata", metadata_folder, "Nimbus panel and ROI metadata."
        )
        reporter.add_input(
            "masks",
            general.masks_folder,
            "Label masks used for scan inference and cell averaging.",
        )
        reporter.add_input(
            "raw_images",
            general.raw_images_folder,
            "Raw marker images used where selected by panel.csv or as fallback.",
        )
        reporter.add_input(
            "denoised_images",
            general.denoised_images_folder,
            "Denoised marker images used where selected by panel.csv.",
        )
        if baseline_path is not None:
            reporter.add_input(
                "normalization_dictionary",
                baseline_path,
                f"Read-only Nimbus {baseline_file_kind}.",
            )

    model = Nimbus(
        dataset=dataset,
        output_dir=str(files_dir),
        save_predictions=False,
        batch_size=nimbus.batch_size,
        test_time_aug=nimbus.test_time_augmentation,
        model_magnification=nimbus.model_magnification,
        device=nimbus.device,
        checkpoint=nimbus.checkpoint,
    )

    candidate_tables = []
    threshold_tables = []
    roi_tables = []
    recommendation_tables = []
    cell_score_dir = tables_dir / "cell_scores"
    used_filenames: set[str] = set()
    for marker_index, marker in enumerate(markers, start=1):
        vmax_values = vmax_grids[marker]
        marker_rois = selected_rois_by_marker[marker]
        LOGGER.info(
            "Scanning marker %s (%d/%d) across %d Vmax values and %d ROI(s).",
            marker,
            marker_index,
            len(markers),
            len(vmax_values),
            len(marker_rois),
        )
        pixel_summary = pd.DataFrame(
            _pixel_summary_rows(
                dataset,
                marker,
                vmax_values,
                marker_rois,
                lower_threshold=lower_thresholds[marker],
            )
        )
        marker_scores = []
        for vmax in vmax_values:
            dataset.normalization_dict = dict(baselines)
            dataset.normalization_dict[marker] = float(vmax)
            dataset.normalization_lower_thresholds = dict(lower_thresholds)
            predictions = _predict_fovs_with_padding(
                nimbus=model,
                dataset=dataset,
                output_dir=str(files_dir),
                save_predictions=False,
                batch_size=nimbus.batch_size,
                test_time_augmentation=nimbus.test_time_augmentation,
                allow_resize_on_mismatch=nimbus.allow_prediction_resize,
                channels=[marker],
                fovs=marker_rois,
            )
            scores = predictions[["fov", "label", marker]].rename(
                columns={"fov": "roi", marker: "nimbus_score"}
            )
            scores.insert(0, "vmax", float(vmax))
            scores.insert(0, "marker", marker)
            marker_scores.append(scores)
        cell_scores = pd.concat(marker_scores, ignore_index=True)
        if not np.isfinite(cell_scores["nimbus_score"].to_numpy(dtype=float)).all():
            raise ValueError(
                f"Nimbus returned non-finite cell scores for marker {marker!r}."
            )

        analysis = analyze_normalization_scan(
            cell_scores,
            pixel_summary,
            baseline_values={marker: baselines[marker]},
            positive_thresholds=settings.positive_score_thresholds,
            primary_threshold=settings.primary_positive_score_threshold,
            stability_tolerance=settings.stability_tolerance,
            call_flip_tolerance=settings.call_flip_tolerance,
            saturation_tolerance=settings.saturation_tolerance,
            cliff_tolerance=settings.cliff_tolerance,
        )
        candidate_tables.append(analysis.candidate_summary)
        threshold_tables.append(analysis.threshold_summary)
        roi_tables.append(analysis.roi_summary)
        recommendation_tables.append(analysis.recommendations)

        safe_name = safe_marker_filename(marker)
        if safe_name.casefold() in used_filenames:
            safe_name = f"{safe_name}_{marker_index:02d}"
        used_filenames.add(safe_name.casefold())
        plot_marker_scan(
            marker=marker,
            cell_scores=cell_scores,
            analysis=analysis,
            output_path=figures_dir / f"{safe_name}.png",
            primary_positive_score_threshold=settings.primary_positive_score_threshold,
        )
        if settings.save_cell_scores:
            cell_score_dir.mkdir(parents=True, exist_ok=True)
            cell_scores.to_csv(
                cell_score_dir / f"{safe_name}.csv.gz",
                index=False,
                compression="gzip",
            )

    candidate_summary = pd.concat(candidate_tables, ignore_index=True)
    threshold_summary = pd.concat(threshold_tables, ignore_index=True)
    roi_summary = pd.concat(roi_tables, ignore_index=True)
    recommendations = pd.concat(recommendation_tables, ignore_index=True)
    output_tables = {
        "candidate_summary": tables_dir / "normalization_scan_candidates.csv",
        "threshold_summary": tables_dir / "normalization_scan_thresholds.csv",
        "roi_summary": tables_dir / "normalization_scan_rois.csv",
        "recommendations": tables_dir / "normalization_scan_recommendations.csv",
        "baselines": tables_dir / "normalization_scan_baselines.csv",
        "roi_selection": tables_dir / "normalization_scan_roi_selection.csv",
    }
    candidate_summary.to_csv(output_tables["candidate_summary"], index=False)
    threshold_summary.to_csv(output_tables["threshold_summary"], index=False)
    roi_summary.to_csv(output_tables["roi_summary"], index=False)
    recommendations.to_csv(output_tables["recommendations"], index=False)
    pd.DataFrame(
        [
            {
                "marker": marker,
                "baseline_vmax": baselines[marker],
                "baseline_source": baseline_sources[marker],
                "lower_threshold": lower_thresholds[marker],
                "lower_threshold_source": lower_threshold_sources[marker],
            }
            for marker in markers
        ]
    ).to_csv(output_tables["baselines"], index=False)
    roi_selection.to_csv(output_tables["roi_selection"], index=False)
    suggested_path = write_suggested_normalization_dict(
        files_dir / "suggested_normalization_dict.csv",
        recommendations,
        lower_thresholds,
    )
    summary_path = _write_summary_markdown(
        summaries_dir / "normalization_scan_summary.md",
        recommendations=recommendations,
        selected_rois_by_marker=selected_rois_by_marker,
        roi_selection_strategy=roi_selection_method,
        primary_threshold=settings.primary_positive_score_threshold,
        baseline_source=baseline_source,
    )

    if reporter:
        for path in output_tables.values():
            reporter.add_file(
                "table", path, "Nimbus normalization scan diagnostic table."
            )
        reporter.add_file(
            "file",
            suggested_path,
            "Review-only Nimbus-compatible suggested Vmax dictionary.",
        )
        reporter.add_file(
            "summary", summary_path, "Interpretation and marker recommendation summary."
        )
        reporter.add_metric("markers_scanned", len(markers))
        reporter.add_metric("rois_scanned", len(unique_scan_rois))
        reporter.add_metric(
            "roi_marker_selections",
            sum(len(rois) for rois in selected_rois_by_marker.values()),
        )
        reporter.add_metric(
            "manual_review_markers",
            int(recommendations["manual_review_required"].sum()),
        )
        reporter.add_metric(
            "vmax_cliff_markers", int(recommendations["cliff_detected"].sum())
        )
        reporter.add_metric(
            "nonzero_lower_threshold_markers",
            sum(value > 0 for value in lower_thresholds.values()),
        )
        reporter.add_note(
            "The suggested normalization CSV is a stability diagnostic and was not applied to the canonical Nimbus normalization asset."
        )
        for row in recommendations[
            recommendations["manual_review_required"]
        ].itertuples(index=False):
            reporter.add_warning(f"{row.marker}: {row.review_reason}")

    LOGGER.info(
        "Nimbus normalization scan complete: %d marker(s), %d unique ROI(s); recommendations -> %s",
        len(markers),
        len(unique_scan_rois),
        output_tables["recommendations"],
    )
    return 0


def main() -> None:
    raise SystemExit(run_pipeline())


if __name__ == "__main__":
    main()
