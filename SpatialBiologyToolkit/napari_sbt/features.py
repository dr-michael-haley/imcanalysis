"""Cohort-first, full-mask-context feature extraction for IMC cells."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree
from skimage.measure import regionprops
from skimage.segmentation import expand_labels

from .feature_catalog import (
    CONTEXT_FEATURE_DESCRIPTIONS,
    DISTRIBUTION_FEATURE_DESCRIPTIONS,
    REGION_IMAGE_FEATURE_DESCRIPTIONS,
    SHAPE_FEATURE_DESCRIPTIONS,
)
from .models import SyntheticFeatureRecipe


@dataclass
class RoiFeatureResult:
    """Feature rows plus extraction warnings for one ROI."""

    roi: str
    table: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    vanished_object_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class _MeasurementRegion:
    """Minimal regionprops-compatible view of one overlapping measurement mask."""

    label: int
    slice: tuple[slice, slice]
    image: np.ndarray

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (
            int(self.slice[0].start),
            int(self.slice[1].start),
            int(self.slice[0].stop),
            int(self.slice[1].stop),
        )


def _finite(values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float64).ravel()
    return flattened[np.isfinite(flattened)]


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    try:
        numerator_value = float(numerator)
        denominator_value = float(denominator)
    except (TypeError, ValueError):
        return np.nan
    if (
        not np.isfinite(numerator_value)
        or not np.isfinite(denominator_value)
        or abs(denominator_value) <= np.finfo(float).eps
    ):
        return np.nan
    return numerator_value / denominator_value


def add_distribution_features(
    row: dict[str, Any],
    values: np.ndarray,
    prefix: str,
    *,
    separator: str = "::",
    selected_suffixes: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Add the CellPose-compatible distribution summaries."""

    selected_names = list(
        DISTRIBUTION_FEATURE_DESCRIPTIONS
        if selected_suffixes is None
        else selected_suffixes
    )
    selected = set(selected_names)
    values = _finite(values)
    if "pixel_count" in selected:
        row[f"{prefix}{separator}pixel_count"] = int(values.size)
    for suffix in selected_names:
        if suffix != "pixel_count":
            row[f"{prefix}{separator}{suffix}"] = np.nan
    if values.size == 0:
        return
    q05, q10, q25, q50, q75, q90, q95 = np.quantile(
        values, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    )
    mean = float(np.mean(values))
    std = float(np.std(values))
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    calculated = {
        f"{prefix}{separator}mean": mean,
        f"{prefix}{separator}median": float(q50),
        f"{prefix}{separator}std": std,
        f"{prefix}{separator}min": minimum,
        f"{prefix}{separator}max": maximum,
        f"{prefix}{separator}sum": float(np.sum(values)),
        f"{prefix}{separator}q05": float(q05),
        f"{prefix}{separator}q10": float(q10),
        f"{prefix}{separator}q25": float(q25),
        f"{prefix}{separator}q75": float(q75),
        f"{prefix}{separator}q90": float(q90),
        f"{prefix}{separator}q95": float(q95),
        f"{prefix}{separator}iqr": float(q75 - q25),
        f"{prefix}{separator}range": float(maximum - minimum),
        f"{prefix}{separator}cv": _safe_ratio(std, abs(mean)),
    }
    row.update(
        {
            key: value
            for key, value in calculated.items()
            if key.rsplit(separator, 1)[-1] in selected
        }
    )


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    gradient_y, gradient_x = np.gradient(image)
    return np.sqrt(gradient_y**2 + gradient_x**2).astype(np.float32)


def _shape_features(
    region,
    image_shape: tuple[int, int],
    selected_suffixes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    min_row, min_col, max_row, max_col = region.bbox
    bbox_height = max_row - min_row
    bbox_width = max_col - min_col
    bbox_area = bbox_height * bbox_width
    image_area = int(image_shape[0] * image_shape[1])
    area = float(region.area)
    perimeter = float(region.perimeter)
    major_axis = float(
        region.axis_major_length
        if hasattr(region, "axis_major_length")
        else region.major_axis_length
    )
    minor_axis = float(
        region.axis_minor_length
        if hasattr(region, "axis_minor_length")
        else region.minor_axis_length
    )
    convex_area = float(
        region.area_convex if hasattr(region, "area_convex") else region.convex_area
    )
    filled_area = float(
        region.area_filled if hasattr(region, "area_filled") else region.filled_area
    )
    equivalent_diameter = (
        region.equivalent_diameter_area
        if hasattr(region, "equivalent_diameter_area")
        else region.equivalent_diameter
    )
    feret = getattr(region, "feret_diameter_max", np.nan)
    crofton = getattr(region, "perimeter_crofton", np.nan)
    edge_distance = min(
        min_row,
        min_col,
        image_shape[0] - max_row,
        image_shape[1] - max_col,
    )
    calculated = {
        "X_loc": float(region.centroid[1]),
        "Y_loc": float(region.centroid[0]),
        "bbox_min_row": int(min_row),
        "bbox_min_col": int(min_col),
        "bbox_max_row": int(max_row),
        "bbox_max_col": int(max_col),
        "bbox_width": int(bbox_width),
        "bbox_height": int(bbox_height),
        "bbox_area": int(bbox_area),
        "mask_area": area,
        "mask_area_fraction_roi": _safe_ratio(area, image_area),
        "mask_area_fraction_bbox": _safe_ratio(area, bbox_area),
        "mask_perimeter": perimeter,
        "mask_perimeter_crofton": float(crofton)
        if np.isfinite(crofton)
        else np.nan,
        "mask_circularity": _safe_ratio(4 * np.pi * area, perimeter**2),
        "mask_compactness": _safe_ratio(perimeter**2, 4 * np.pi * area),
        "mask_major_axis_length": major_axis,
        "mask_minor_axis_length": minor_axis,
        "mask_axis_ratio": _safe_ratio(major_axis, minor_axis),
        "mask_eccentricity": float(region.eccentricity),
        "mask_solidity": float(region.solidity),
        "mask_extent": float(region.extent),
        "mask_orientation_degrees": float(np.degrees(region.orientation)),
        "mask_equivalent_diameter": float(equivalent_diameter),
        "mask_feret_diameter_max": float(feret) if np.isfinite(feret) else np.nan,
        "mask_convex_area": convex_area,
        "mask_filled_area": filled_area,
        "mask_hole_area": float(filled_area - area),
        "mask_hole_fraction": _safe_ratio(filled_area - area, filled_area),
        "mask_convexity": _safe_ratio(area, convex_area),
        "mask_euler_number": int(region.euler_number),
        "mask_edge_touching": bool(edge_distance == 0),
        "mask_min_distance_to_edge_px": int(edge_distance),
    }
    selected = set(
        SHAPE_FEATURE_DESCRIPTIONS
        if selected_suffixes is None
        else selected_suffixes
    )
    return {
        key: value for key, value in calculated.items() if key in selected
    }


def _context_features(
    mask: np.ndarray,
    region,
    selected_suffixes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    min_row, min_col, max_row, max_col = region.bbox
    pad = 5
    row_slice = slice(max(0, min_row - pad), min(mask.shape[0], max_row + pad))
    col_slice = slice(max(0, min_col - pad), min(mask.shape[1], max_col + pad))
    local_labels = mask[row_slice, col_slice]
    local_object = local_labels == int(region.label)
    nearby = binary_dilation(local_object, iterations=5) & (
        local_labels != int(region.label)
    )
    touching = binary_dilation(local_object, iterations=1) & (
        local_labels != int(region.label)
    )
    nearby_labels = np.unique(local_labels[nearby])
    touching_labels = np.unique(local_labels[touching])
    calculated = {
        "mask_neighbor_count_5px": int(np.count_nonzero(nearby_labels > 0)),
        "mask_touching_neighbor_count_1px": int(
            np.count_nonzero(touching_labels > 0)
        ),
    }
    selected = set(
        CONTEXT_FEATURE_DESCRIPTIONS
        if selected_suffixes is None
        else selected_suffixes
    )
    return {
        key: value for key, value in calculated.items() if key in selected
    }


def _measurement_labels(
    mask: np.ndarray,
    eligible_ids: set[int],
    offset_px: int,
) -> tuple[np.ndarray, list[int]]:
    if offset_px == 0:
        return mask.copy(), []
    if offset_px > 0:
        return expand_labels(mask, distance=offset_px), []

    labels = np.zeros_like(mask)
    vanished: list[int] = []
    for region in regionprops(mask):
        label = int(region.label)
        if label not in eligible_ids:
            continue
        eroded = binary_erosion(
            region.image.astype(bool),
            iterations=abs(offset_px),
            border_value=0,
        )
        if not np.any(eroded):
            vanished.append(label)
            continue
        local = labels[region.slice]
        local[eroded] = label
    return labels, vanished


def _overlapping_measurement_regions(
    mask: np.ndarray,
    eligible_ids: set[int],
    offset_px: int,
) -> dict[int, _MeasurementRegion]:
    """Dilate each eligible object independently, permitting shared pixels."""

    regions: dict[int, _MeasurementRegion] = {}
    for region in regionprops(mask):
        label = int(region.label)
        if label not in eligible_ids:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        row_slice = slice(
            max(0, min_row - offset_px),
            min(mask.shape[0], max_row + offset_px),
        )
        col_slice = slice(
            max(0, min_col - offset_px),
            min(mask.shape[1], max_col + offset_px),
        )
        seed = mask[row_slice, col_slice] == label
        expanded = distance_transform_edt(~seed) <= float(offset_px)
        regions[label] = _MeasurementRegion(
            label=label,
            slice=(row_slice, col_slice),
            image=expanded,
        )
    return regions


def _measurement_regions(
    mask: np.ndarray,
    eligible_ids: set[int],
    recipe: SyntheticFeatureRecipe,
) -> tuple[dict[int, Any], list[int]]:
    if recipe.mask_offset_px > 0 and recipe.allow_positive_offset_overlap:
        return (
            _overlapping_measurement_regions(
                mask,
                eligible_ids,
                recipe.mask_offset_px,
            ),
            [],
        )
    measurement_labels, vanished = _measurement_labels(
        mask,
        eligible_ids,
        recipe.mask_offset_px,
    )
    return (
        {
            int(region.label): region
            for region in regionprops(measurement_labels)
            if int(region.label) in eligible_ids
        },
        vanished,
    )


def _region_image_features(
    row: dict[str, Any],
    image: np.ndarray,
    full_mask: np.ndarray,
    measurement_region,
    original_region,
    prefix: str,
    ring_distance: int,
    distribution_suffixes: list[str] | tuple[str, ...] | None = None,
    region_suffixes: list[str] | tuple[str, ...] | None = None,
) -> None:
    local_mask = measurement_region.image.astype(bool)
    local_image = image[measurement_region.slice]
    values = local_image[local_mask]
    add_distribution_features(row, values, prefix)

    for suffix in REGION_IMAGE_FEATURE_DESCRIPTIONS:
        row[f"{prefix}::{suffix}"] = (
            0 if suffix == "local_bg_pixel_count" else np.nan
        )

    local_rows, local_cols = np.nonzero(local_mask)
    weights = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(weights)
    if np.any(finite):
        finite_values = weights[finite]
        shifted = finite_values - float(np.min(finite_values))
        if np.sum(shifted) <= np.finfo(float).eps:
            shifted = np.abs(finite_values)
        if np.sum(shifted) > np.finfo(float).eps:
            min_row, min_col, _, _ = measurement_region.bbox
            absolute_rows = local_rows[finite] + min_row
            absolute_cols = local_cols[finite] + min_col
            weighted_y = float(np.average(absolute_rows, weights=shifted))
            weighted_x = float(np.average(absolute_cols, weights=shifted))
            offset = float(
                np.hypot(
                    weighted_y - original_region.centroid[0],
                    weighted_x - original_region.centroid[1],
                )
            )
            radius = math.sqrt(float(original_region.area) / np.pi)
            row[f"{prefix}::weighted_x"] = weighted_x
            row[f"{prefix}::weighted_y"] = weighted_y
            row[f"{prefix}::weighted_centroid_offset_px"] = offset
            row[
                f"{prefix}::weighted_centroid_offset_fraction_radius"
            ] = _safe_ratio(offset, radius)

    core = binary_erosion(local_mask, iterations=1, border_value=0)
    border = local_mask & ~core
    core_values = local_image[core]
    border_values = local_image[border]
    core_mean = float(np.mean(core_values)) if core_values.size else np.nan
    border_mean = float(np.mean(border_values)) if border_values.size else np.nan
    row[f"{prefix}::core_mean"] = core_mean
    row[f"{prefix}::border_mean"] = border_mean
    row[f"{prefix}::core_to_border_ratio"] = _safe_ratio(core_mean, border_mean)

    min_row, min_col, max_row, max_col = measurement_region.bbox
    row_slice = slice(
        max(0, min_row - ring_distance),
        min(full_mask.shape[0], max_row + ring_distance),
    )
    col_slice = slice(
        max(0, min_col - ring_distance),
        min(full_mask.shape[1], max_col + ring_distance),
    )
    object_mask = np.zeros(
        (
            int(row_slice.stop) - int(row_slice.start),
            int(col_slice.stop) - int(col_slice.start),
        ),
        dtype=bool,
    )
    region_rows = slice(
        int(measurement_region.slice[0].start) - int(row_slice.start),
        int(measurement_region.slice[0].stop) - int(row_slice.start),
    )
    region_cols = slice(
        int(measurement_region.slice[1].start) - int(col_slice.start),
        int(measurement_region.slice[1].stop) - int(col_slice.start),
    )
    object_mask[region_rows, region_cols] = measurement_region.image.astype(bool)
    ring = binary_dilation(object_mask, iterations=ring_distance) & ~object_mask
    ring &= full_mask[row_slice, col_slice] == 0
    background = _finite(image[row_slice, col_slice][ring])
    row[f"{prefix}::local_bg_pixel_count"] = int(background.size)
    if background.size:
        background_mean = float(np.mean(background))
        background_std = float(np.std(background))
        foreground_mean = row[f"{prefix}::mean"]
        row[f"{prefix}::local_bg_mean"] = background_mean
        row[f"{prefix}::local_bg_std"] = background_std
        row[f"{prefix}::foreground_to_bg_ratio"] = _safe_ratio(
            foreground_mean, background_mean
        )
        row[f"{prefix}::foreground_bg_contrast"] = (
            float(foreground_mean) - background_mean
        )
        row[f"{prefix}::foreground_bg_contrast_z"] = _safe_ratio(
            float(foreground_mean) - background_mean,
            background_std,
        )
    selected_distribution = set(
        DISTRIBUTION_FEATURE_DESCRIPTIONS
        if distribution_suffixes is None
        else distribution_suffixes
    )
    selected_regions = set(
        REGION_IMAGE_FEATURE_DESCRIPTIONS
        if region_suffixes is None
        else region_suffixes
    )
    for suffix in DISTRIBUTION_FEATURE_DESCRIPTIONS:
        if suffix not in selected_distribution:
            row.pop(f"{prefix}::{suffix}", None)
    for suffix in REGION_IMAGE_FEATURE_DESCRIPTIONS:
        if suffix not in selected_regions:
            row.pop(f"{prefix}::{suffix}", None)


def _add_nearest_centroid_context(
    table: pd.DataFrame,
    all_regions: list,
    selected_suffixes: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if table.empty:
        return table
    selected_suffixes = list(
        CONTEXT_FEATURE_DESCRIPTIONS
        if selected_suffixes is None
        else selected_suffixes
    )
    selected_features = set(selected_suffixes)
    all_coordinates = np.asarray(
        [[region.centroid[0], region.centroid[1]] for region in all_regions],
        dtype=float,
    )
    if len(all_coordinates) <= 1:
        if "nearest_centroid_distance_px" in selected_features:
            table["nearest_centroid_distance_px"] = np.nan
        if "centroid_neighbor_count_25px" in selected_features:
            table["centroid_neighbor_count_25px"] = 0
        if "centroid_neighbor_count_50px" in selected_features:
            table["centroid_neighbor_count_50px"] = 0
        return table
    tree = cKDTree(all_coordinates)
    selected_coordinates = table[["Y_loc", "X_loc"]].to_numpy(dtype=float)
    nearest, _ = tree.query(selected_coordinates, k=2)
    if "nearest_centroid_distance_px" in selected_features:
        table["nearest_centroid_distance_px"] = nearest[:, 1]
    if "centroid_neighbor_count_25px" in selected_features:
        table["centroid_neighbor_count_25px"] = [
            len(tree.query_ball_point(point, r=25)) - 1
            for point in selected_coordinates
        ]
    if "centroid_neighbor_count_50px" in selected_features:
        table["centroid_neighbor_count_50px"] = [
            len(tree.query_ball_point(point, r=50)) - 1
            for point in selected_coordinates
        ]
    return table


def _add_cohort_roi_ranks(
    table: pd.DataFrame,
    statistics: list[str] | tuple[str, ...] = ("zscore", "percentile"),
) -> pd.DataFrame:
    candidates = [
        column
        for column in table.columns
        if column
        in {
            "mask_area",
            "mask_perimeter",
            "mask_circularity",
            "mask_axis_ratio",
        }
        or column.endswith(("::mean", "::sum", "::foreground_bg_contrast"))
    ]
    for column in candidates:
        values = pd.to_numeric(table[column], errors="coerce")
        mean = values.mean(skipna=True)
        std = values.std(skipna=True, ddof=0)
        if "zscore" in statistics:
            table[f"{column}::cohort_roi_zscore"] = (
                (values - mean) / std if pd.notna(std) and std > 0 else 0.0
            )
        if "percentile" in statistics:
            table[f"{column}::cohort_roi_percentile"] = values.rank(pct=True)
    return table


def build_roi_features(
    *,
    roi: str,
    full_mask: np.ndarray,
    eligible_ids: set[int],
    channel_images: Mapping[str, np.ndarray],
    recipe: SyntheticFeatureRecipe,
) -> RoiFeatureResult:
    """Calculate feature rows only for eligible cells in one ROI."""

    mask = np.asarray(full_mask)
    if mask.ndim != 2:
        raise ValueError(f"ROI {roi!r} mask must be 2D, got {mask.shape}.")
    if not np.issubdtype(mask.dtype, np.integer):
        raise TypeError(f"ROI {roi!r} mask must contain integer labels.")
    eligible_ids = {int(value) for value in eligible_ids if int(value) > 0}
    all_regions = list(regionprops(mask))
    original_regions = {
        int(region.label): region
        for region in all_regions
        if int(region.label) in eligible_ids
    }
    missing = sorted(eligible_ids - set(original_regions))
    if missing:
        raise ValueError(
            f"ROI {roi!r} has {len(missing)} eligible object IDs absent from its "
            f"mask; examples: {missing[:10]}"
        )
    for channel, image in channel_images.items():
        image = np.asarray(image).squeeze()
        if image.ndim != 2:
            raise ValueError(
                f"ROI {roi!r} channel {channel!r} must be 2D, got {image.shape}."
            )
        if image.shape != mask.shape:
            raise ValueError(
                f"ROI {roi!r} channel {channel!r} shape {image.shape} does not "
                f"match mask shape {mask.shape}; scientific images are not resized."
            )

    measurement_regions, vanished = _measurement_regions(
        mask,
        eligible_ids,
        recipe,
    )
    rows: list[dict[str, Any]] = []
    gradients = (
        {
            channel: gradient_magnitude(np.asarray(image))
            for channel, image in channel_images.items()
        }
        if recipe.gradient_features
        else {}
    )
    for object_id in sorted(eligible_ids):
        original = original_regions[object_id]
        measurement = measurement_regions.get(object_id)
        row: dict[str, Any] = {
            "ROI": str(roi),
            "ObjectNumber": object_id,
            "CellID": f"{roi}_{object_id}",
            "X_loc": float(original.centroid[1]),
            "Y_loc": float(original.centroid[0]),
            "measurement_mask_offset_px": int(recipe.mask_offset_px),
            "measurement_allows_cell_overlap": bool(
                recipe.allow_positive_offset_overlap
            ),
            "measurement_region_vanished": measurement is None,
        }
        if recipe.shape_features:
            row.update(
                _shape_features(
                    original,
                    mask.shape,
                    recipe.shape_feature_names,
                )
            )
        if recipe.context_features:
            row.update(
                _context_features(
                    mask,
                    original,
                    recipe.context_feature_names,
                )
            )
        if measurement is not None:
            for channel, image in channel_images.items():
                prefix = f"channel::{channel}"
                if recipe.region_features:
                    _region_image_features(
                        row,
                        np.asarray(image),
                        mask,
                        measurement,
                        original,
                        prefix,
                        recipe.background_ring_px,
                        (
                            recipe.distribution_feature_names
                            if recipe.distribution_features
                            else []
                        ),
                        recipe.region_feature_names,
                    )
                elif recipe.distribution_features:
                    values = np.asarray(image)[measurement.slice][
                        measurement.image.astype(bool)
                    ]
                    add_distribution_features(
                        row,
                        values,
                        prefix,
                        selected_suffixes=recipe.distribution_feature_names,
                    )
                if recipe.gradient_features:
                    values = gradients[channel][measurement.slice][
                        measurement.image.astype(bool)
                    ]
                    add_distribution_features(
                        row,
                        values,
                        f"{prefix}::gradient",
                        selected_suffixes=recipe.gradient_feature_names,
                    )
        else:
            for channel in channel_images:
                prefix = f"channel::{channel}"
                if recipe.distribution_features or recipe.region_features:
                    add_distribution_features(
                        row,
                        np.asarray([]),
                        prefix,
                        selected_suffixes=(
                            recipe.distribution_feature_names
                            if recipe.distribution_features
                            else []
                        ),
                    )
                if recipe.region_features:
                    for suffix in recipe.region_feature_names:
                        row[f"{prefix}::{suffix}"] = (
                            0 if suffix == "local_bg_pixel_count" else np.nan
                        )
                if recipe.gradient_features:
                    add_distribution_features(
                        row,
                        np.asarray([]),
                        f"{prefix}::gradient",
                        selected_suffixes=recipe.gradient_feature_names,
                    )
        rows.append(row)

    table = pd.DataFrame(rows)
    if recipe.context_features:
        table = _add_nearest_centroid_context(
            table,
            all_regions,
            recipe.context_feature_names,
        )
        if "roi_total_object_count" in recipe.context_feature_names:
            table["roi_total_object_count"] = len(all_regions)
        if "roi_eligible_object_count" in recipe.context_feature_names:
            table["roi_eligible_object_count"] = len(eligible_ids)
        if "roi_full_mask_area_fraction" in recipe.context_feature_names:
            table["roi_full_mask_area_fraction"] = (
                float(np.count_nonzero(mask)) / mask.size
            )
    if recipe.roi_rank_features:
        table = _add_cohort_roi_ranks(table, recipe.roi_rank_statistics)
    leading = ["ROI", "ObjectNumber", "CellID", "X_loc", "Y_loc"]
    table = table.loc[:, [column for column in leading if column in table] + [
        column for column in table.columns if column not in leading
    ]]
    warnings = []
    if vanished:
        warnings.append(
            f"ROI {roi!r}: {len(vanished)} eligible measurement regions vanished "
            f"after {recipe.mask_offset_px}px erosion."
        )
    return RoiFeatureResult(
        roi=str(roi),
        table=table,
        warnings=warnings,
        vanished_object_ids=vanished,
    )


def _human_channel_feature(column: str) -> tuple[str, str]:
    match = re.match(r"^channel::(.+?)::(.+)$", column)
    if not match:
        return "synthetic", "Synthetic cohort-first cell feature."
    channel, suffix = match.groups()
    if suffix.startswith("gradient::"):
        statistic = suffix.split("::", 1)[1]
        description = DISTRIBUTION_FEATURE_DESCRIPTIONS.get(
            statistic, "Gradient-derived channel statistic."
        )
        return "channel_gradient", f"{channel} gradient: {description}"
    if suffix in DISTRIBUTION_FEATURE_DESCRIPTIONS:
        return (
            "channel_intensity",
            f"{channel}: {DISTRIBUTION_FEATURE_DESCRIPTIONS[suffix]}",
        )
    if suffix in REGION_IMAGE_FEATURE_DESCRIPTIONS:
        return (
            "channel_region",
            f"{channel}: {REGION_IMAGE_FEATURE_DESCRIPTIONS[suffix]}",
        )
    if suffix.endswith("cohort_roi_zscore"):
        return "within_cohort_roi", f"Within-cohort ROI z-score of {column.rsplit('::', 1)[0]}."
    if suffix.endswith("cohort_roi_percentile"):
        return "within_cohort_roi", f"Within-cohort ROI percentile of {column.rsplit('::', 1)[0]}."
    return "channel_feature", f"Image-derived feature for channel {channel}."


def build_feature_dictionary(table: pd.DataFrame) -> pd.DataFrame:
    """Build selection metadata for a generated cohort feature table."""

    rows = []
    identifiers = {"ROI", "ObjectNumber", "CellID", "X_loc", "Y_loc"}
    for column in table.columns:
        if column in identifiers:
            category, description, usable = (
                "identifier",
                "Cell identity or centroid coordinate.",
                False,
            )
        elif column.startswith("mask_"):
            category, description, usable = (
                "mask_morphology",
                "Original full-segmentation mask morphology or context.",
                True,
            )
        elif column.startswith("channel::"):
            category, description = _human_channel_feature(column)
            usable = True
        elif column.startswith("roi_"):
            category, description, usable = (
                "roi_context",
                "Cohort or full-segmentation ROI context.",
                True,
            )
        elif column in {
            "measurement_allows_cell_overlap",
            "measurement_mask_offset_px",
            "measurement_region_vanished",
        }:
            category, description, usable = (
                "measurement_metadata",
                "Measurement-region construction metadata.",
                False,
            )
        else:
            category, description, usable = (
                "synthetic",
                "Synthetic cohort-first cell feature.",
                pd.api.types.is_numeric_dtype(table[column]),
            )
        rows.append(
            {
                "feature": column,
                "type/source": category,
                "description": description,
                "valid_model_input": bool(usable),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "DISTRIBUTION_FEATURE_DESCRIPTIONS",
    "REGION_IMAGE_FEATURE_DESCRIPTIONS",
    "RoiFeatureResult",
    "add_distribution_features",
    "build_feature_dictionary",
    "build_roi_features",
    "gradient_magnitude",
]
