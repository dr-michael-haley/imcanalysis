"""Empirical marker halos and neighbour-attributable signal scores.

Automatic exemplar selection may use the input AnnData expression matrix to
identify convincing marker-positive candidates.  Halo intensities, source
strengths, backgrounds, projected sources, and final scores are still learned
exclusively from raw marker pixels and segmentation geometry.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import ndimage

from SpatialBiologyToolkit.cellvision import ROIInput
from SpatialBiologyToolkit.napari_sbt.resources import process_cpu_limit


LOGGER = logging.getLogger(__name__)
FloatArray: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int64]
SOURCE_TARGET_COLUMNS = (
    "target_obs_index",
    "target_cell_id",
    "target_roi",
    "target_segmentation_label",
    "marker",
    "source_obs_index",
    "source_cell_id",
    "source_roi",
    "source_segmentation_label",
    "attributable_intensity",
    "fraction_of_observed_signal",
    "fraction_of_attributable_signal",
)


@dataclass(frozen=True)
class HaloParameters:
    """Scientific parameters shared by profile-learning and application workers."""

    max_halo_px: int = 8
    source_anchor_dilation_px: int = 2
    source_anchor_quantile: float = 0.95
    min_exemplars: int = 5
    source_threshold_quantile: float = 0.10
    halo_aggregation: str = "max"
    exemplar_mode: str = "automatic"
    automatic_positive_threshold: float = 0.5
    automatic_same_marker_clearance_px: float = 10.0
    automatic_target_exemplars_per_marker: int = 30
    automatic_max_exemplars_per_roi: int = 5
    automatic_min_pixels_per_bin: int = 8

    def validate(self) -> None:
        if self.max_halo_px < 1:
            raise ValueError("max_halo_px must be at least one pixel")
        if self.source_anchor_dilation_px < 0:
            raise ValueError("source_anchor_dilation_px cannot be negative")
        if self.source_anchor_dilation_px > self.max_halo_px:
            raise ValueError("source_anchor_dilation_px cannot exceed max_halo_px")
        if not 0 < self.source_anchor_quantile <= 1:
            raise ValueError("source_anchor_quantile must be in (0, 1]")
        if self.min_exemplars < 1:
            raise ValueError("min_exemplars must be positive")
        if not 0 <= self.source_threshold_quantile <= 1:
            raise ValueError("source_threshold_quantile must be in [0, 1]")
        if self.halo_aggregation not in {"max", "sum"}:
            raise ValueError("halo_aggregation must be 'max' or 'sum'")
        if self.exemplar_mode not in {"automatic", "manual", "augment"}:
            raise ValueError("exemplar_mode must be 'automatic', 'manual', or 'augment'")
        if not np.isfinite(self.automatic_positive_threshold):
            raise ValueError("automatic_positive_threshold must be finite")
        if self.automatic_same_marker_clearance_px < 0:
            raise ValueError("automatic_same_marker_clearance_px cannot be negative")
        if (
            self.exemplar_mode in {"automatic", "augment"}
            and self.automatic_target_exemplars_per_marker < self.min_exemplars
        ):
            raise ValueError(
                "automatic_target_exemplars_per_marker cannot be below min_exemplars"
            )
        if self.automatic_max_exemplars_per_roi < 1:
            raise ValueError("automatic_max_exemplars_per_roi must be positive")
        if self.automatic_min_pixels_per_bin < 1:
            raise ValueError("automatic_min_pixels_per_bin must be positive")


@dataclass(frozen=True)
class WorkerUsage:
    """Resolved ROI-worker allocation and the source of its CPU limit."""

    requested: int
    effective: int
    cpu_limit: int
    limit_source: str


@dataclass(frozen=True)
class ExemplarProfile:
    """One source-normalized radial profile learned from one exemplar cell."""

    marker: str
    roi: str
    object_id: int
    profile: FloatArray
    source_strength: float
    background: float
    source_excess_strength: float
    background_method: str
    valid: bool
    reason: str


@dataclass(frozen=True)
class ExemplarSelectionRecord:
    """Candidate- and selection-level provenance for one marker/cell pair."""

    marker: str
    roi: str
    object_id: int
    source_obs_index: int
    source_cell_id: str
    selection_origin: str
    input_x_value: float
    positive_threshold: float
    nearest_same_marker_positive_distance_px: float
    min_unassigned_pixels_per_bin: int
    min_unassigned_fraction: float
    eligible: bool
    selected: bool
    reason: str


@dataclass(frozen=True)
class MarkerHaloProfile:
    """Robust aggregate halo profile and source threshold for one marker."""

    marker: str
    available: bool
    raw_median: FloatArray
    final: FloatArray
    q25: FloatArray
    q75: FloatArray
    n_configured_exemplars: int
    n_valid_exemplars: int
    source_threshold: float
    effective_extent_px: float
    skip_reason: str


@dataclass(frozen=True)
class ProfileWorkerPayload:
    roi: str
    mask_path: str
    channel_paths: Mapping[str, str]
    exemplar_labels: Mapping[str, tuple[int, ...]]
    parameters: HaloParameters


@dataclass(frozen=True)
class CandidateWorkerPayload:
    roi: str
    mask_path: str
    positive_cells: Mapping[str, tuple[tuple[int, int, float, str], ...]]
    parameters: HaloParameters


@dataclass(frozen=True)
class ApplicationWorkerPayload:
    roi: str
    mask_path: str
    channel_paths: tuple[str, ...]
    marker_names: tuple[str, ...]
    target_rows: NDArray[np.int64]
    target_labels: NDArray[np.int64]
    total_cells: int
    profiles: Mapping[str, MarkerHaloProfile]
    parameters: HaloParameters


@dataclass(frozen=True)
class ProjectedHalo:
    """Pixelwise projected halo and the winning global source row."""

    predicted: FloatArray
    source_index: IntArray


@dataclass(frozen=True)
class MarkerHaloMaps:
    """Transient pixel maps for one ROI/marker application or targeted QC crop."""

    observed_excess: FloatArray
    projected: ProjectedHalo
    attributable: FloatArray
    residual: FloatArray
    source_strengths: Mapping[int, float]
    source_labels: tuple[int, ...]
    unmapped_source_labels: tuple[int, ...]
    background: float
    background_method: str
    background_pixels: int


@dataclass(frozen=True)
class SourceTargetAttribution:
    """One non-zero source-to-target relationship for one marker."""

    target_obs_index: int
    marker_index: int
    source_obs_index: int
    attributable_intensity: float
    fraction_of_observed_signal: float
    fraction_of_attributable_signal: float


@dataclass(frozen=True)
class ApplicationWorkerResult:
    roi: str
    target_rows: NDArray[np.int64]
    scores: FloatArray
    classic_intensities: FloatArray
    attributable_intensities: FloatArray
    residual_intensities: FloatArray
    dominant_source_indices: IntArray
    dominant_source_observed_fractions: FloatArray
    dominant_source_attributable_fractions: FloatArray
    source_target_attributions: tuple[SourceTargetAttribution, ...]
    background_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class NeighbourSignalResult:
    """Complete cell-by-marker result before it is attached to AnnData."""

    marker_names: tuple[str, ...]
    scores: FloatArray
    classic_intensities: FloatArray
    attributable_intensities: FloatArray
    residual_intensities: FloatArray
    dominant_source_indices: IntArray
    dominant_source_observed_fractions: FloatArray
    dominant_source_attributable_fractions: FloatArray
    source_target_attributions: tuple[SourceTargetAttribution, ...]
    source_provenance_available: bool
    profiles: Mapping[str, MarkerHaloProfile]
    exemplar_profiles: tuple[ExemplarProfile, ...]
    exemplar_selection: tuple[ExemplarSelectionRecord, ...]
    background_records: tuple[dict[str, Any], ...]
    unknown_exemplar_values: tuple[str, ...]
    warnings: tuple[str, ...]
    worker_usage: WorkerUsage


def resolve_analysis_workers(n_jobs: str | int, n_rois: int) -> WorkerUsage:
    """Resolve ROI workers from SLURM, affinity, and host CPU information."""

    if n_rois < 1:
        raise ValueError("At least one ROI is required")
    cpu_limit, source = process_cpu_limit()
    if n_jobs == "auto":
        requested = cpu_limit
    else:
        requested = int(n_jobs)
        if requested < 1:
            raise ValueError("n_jobs must be 'auto' or a positive integer")
    return WorkerUsage(
        requested=requested,
        effective=max(1, min(requested, cpu_limit, n_rois)),
        cpu_limit=cpu_limit,
        limit_source=source,
    )


def _validate_mask(mask: np.ndarray, *, path: str | Path) -> NDArray[np.int64]:
    labels = np.asarray(np.squeeze(mask))
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D segmentation mask at {path}, got {mask.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"Segmentation mask must use integer labels: {path}")
    if np.any(labels < 0):
        raise ValueError(f"Segmentation mask contains negative labels: {path}")
    labels = labels.astype(np.int64, copy=False)
    if not np.any(labels > 0):
        raise ValueError(f"Segmentation mask contains no positive cell labels: {path}")
    return labels


def _validate_image(
    image: np.ndarray,
    *,
    path: str | Path,
    expected_shape: tuple[int, int],
) -> FloatArray:
    values = np.asarray(np.squeeze(image))
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D raw marker image at {path}, got {image.shape}")
    if values.shape != expected_shape:
        raise ValueError(
            f"Raw marker image {path} has shape {values.shape}; expected {expected_shape}"
        )
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError(f"Raw marker image contains non-finite or non-numeric values: {path}")
    if np.any(values < 0):
        raise ValueError(f"Raw marker image contains negative intensities: {path}")
    return values.astype(np.float32, copy=False)


def source_anchor_labels(mask: np.ndarray, dilation_px: int) -> NDArray[np.int64]:
    """Assign nearby unsegmented pixels to their nearest cell for source anchors.

    Existing cell pixels never change owner. This efficiently represents a
    small non-overlapping dilation while ensuring that another segmented cell
    can never contribute pixels to a source-strength anchor.
    """

    labels = _validate_mask(mask, path="in-memory mask")
    if dilation_px < 0:
        raise ValueError("dilation_px cannot be negative")
    if dilation_px == 0:
        return labels.copy()
    background = labels == 0
    distance, indices = ndimage.distance_transform_edt(
        background,
        return_indices=True,
    )
    nearest = labels[tuple(indices)]
    anchors = labels.copy()
    assign = background & (distance <= float(dilation_px)) & (nearest > 0)
    anchors[assign] = nearest[assign]
    return anchors


def label_quantiles(
    image: np.ndarray,
    labels: np.ndarray,
    quantile: float,
    *,
    requested_labels: Sequence[int] | None = None,
) -> dict[int, float]:
    """Calculate an image quantile for each requested positive label."""

    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    values = np.asarray(image, dtype=np.float32)
    group_labels = np.asarray(labels, dtype=np.int64)
    if values.shape != group_labels.shape:
        raise ValueError("image and labels must have equal shapes")
    selected = group_labels > 0
    requested_set: set[int] | None = None
    if requested_labels is not None:
        requested_set = {int(label) for label in requested_labels}
        if not requested_set:
            return {}
        selected &= np.isin(group_labels, np.fromiter(requested_set, dtype=np.int64))
    flat_labels = group_labels[selected]
    flat_values = values[selected]
    if flat_labels.size == 0:
        return {}
    order = np.argsort(flat_labels, kind="stable")
    ordered_labels = flat_labels[order]
    ordered_values = flat_values[order]
    unique, starts, counts = np.unique(
        ordered_labels,
        return_index=True,
        return_counts=True,
    )
    result = {
        int(label): float(np.quantile(ordered_values[start : start + count], quantile))
        for label, start, count in zip(unique, starts, counts, strict=True)
    }
    if requested_set is not None:
        return {label: result[label] for label in requested_set if label in result}
    return result


def _object_slices(mask: NDArray[np.int64]) -> dict[int, tuple[slice, slice]]:
    slices = ndimage.find_objects(mask)
    return {
        label: object_slice
        for label, object_slice in enumerate(slices, start=1)
        if object_slice is not None
    }


def _expanded_slice(
    object_slice: tuple[slice, slice],
    shape: tuple[int, int],
    radius: int,
) -> tuple[slice, slice]:
    return (
        slice(
            max(0, int(object_slice[0].start) - radius),
            min(shape[0], int(object_slice[0].stop) + radius),
        ),
        slice(
            max(0, int(object_slice[1].start) - radius),
            min(shape[1], int(object_slice[1].stop) + radius),
        ),
    )


def _automatic_candidate_geometry(
    mask: NDArray[np.int64],
    object_id: int,
    object_slice: tuple[slice, slice],
    *,
    max_halo_px: int,
) -> tuple[int, float]:
    """Summarise radial pixels remaining after other cell masks are excluded."""

    patch_slice = _expanded_slice(
        object_slice,
        (int(mask.shape[0]), int(mask.shape[1])),
        max_halo_px,
    )
    patch_mask = mask[patch_slice]
    source = patch_mask == int(object_id)
    distance = ndimage.distance_transform_edt(~source)
    unassigned = patch_mask == 0
    counts: list[int] = []
    fractions: list[float] = []
    for bin_index in range(max_halo_px):
        ring = (distance > float(bin_index)) & (distance <= float(bin_index + 1))
        total = int(np.count_nonzero(ring))
        usable = int(np.count_nonzero(ring & unassigned))
        counts.append(usable)
        fractions.append(float(usable / total) if total else 0.0)
    return min(counts, default=0), min(fractions, default=0.0)


def _nearest_other_positive_distance(
    mask: NDArray[np.int64],
    object_id: int,
    object_slice: tuple[slice, slice],
    positive_lookup: NDArray[np.bool_],
    *,
    clearance_px: float,
) -> float:
    """Return local boundary distance to another positive cell, or infinity."""

    if clearance_px <= 0:
        return float("inf")
    radius = max(1, int(np.ceil(clearance_px)))
    patch_slice = _expanded_slice(
        object_slice,
        (int(mask.shape[0]), int(mask.shape[1])),
        radius,
    )
    patch_mask = mask[patch_slice]
    source = patch_mask == int(object_id)
    other_positive = (
        (patch_mask >= 0)
        & (patch_mask < len(positive_lookup))
        & positive_lookup[np.clip(patch_mask, 0, len(positive_lookup) - 1)]
        & (patch_mask != int(object_id))
    )
    if not np.any(other_positive):
        return float("inf")
    distance = ndimage.distance_transform_edt(~source)
    return float(np.min(distance[other_positive]))


def inspect_automatic_exemplar_candidates(
    payload: CandidateWorkerPayload,
) -> tuple[ExemplarSelectionRecord, ...]:
    """Inspect one ROI's X-positive cells without loading any marker image."""

    from tifffile import imread

    mask = _validate_mask(imread(payload.mask_path), path=payload.mask_path)
    slices = _object_slices(mask)
    maximum_label = int(mask.max(initial=0))
    geometry_cache: dict[int, tuple[int, float]] = {}
    records: list[ExemplarSelectionRecord] = []
    for marker, cells in payload.positive_cells.items():
        positive_lookup = np.zeros(maximum_label + 1, dtype=bool)
        for _obs_index, raw_label, _x_value, _cell_id in cells:
            label = int(raw_label)
            if 0 < label <= maximum_label:
                positive_lookup[label] = True
        for obs_index, raw_label, x_value, cell_id in cells:
            object_id = int(raw_label)
            object_slice = slices.get(object_id)
            if object_slice is None:
                records.append(
                    ExemplarSelectionRecord(
                        marker=str(marker),
                        roi=payload.roi,
                        object_id=object_id,
                        source_obs_index=int(obs_index),
                        source_cell_id=str(cell_id),
                        selection_origin="automatic",
                        input_x_value=float(x_value),
                        positive_threshold=float(
                            payload.parameters.automatic_positive_threshold
                        ),
                        nearest_same_marker_positive_distance_px=float("nan"),
                        min_unassigned_pixels_per_bin=0,
                        min_unassigned_fraction=0.0,
                        eligible=False,
                        selected=False,
                        reason="segmentation_label_missing_from_mask",
                    )
                )
                continue
            if object_id not in geometry_cache:
                geometry_cache[object_id] = _automatic_candidate_geometry(
                    mask,
                    object_id,
                    object_slice,
                    max_halo_px=payload.parameters.max_halo_px,
                )
            minimum_pixels, minimum_fraction = geometry_cache[object_id]
            nearest = _nearest_other_positive_distance(
                mask,
                object_id,
                object_slice,
                positive_lookup,
                clearance_px=(
                    payload.parameters.automatic_same_marker_clearance_px
                ),
            )
            reasons: list[str] = []
            if nearest < payload.parameters.automatic_same_marker_clearance_px:
                reasons.append("same_marker_positive_within_clearance")
            if minimum_pixels < payload.parameters.automatic_min_pixels_per_bin:
                reasons.append("insufficient_unassigned_radial_pixels")
            records.append(
                ExemplarSelectionRecord(
                    marker=str(marker),
                    roi=payload.roi,
                    object_id=object_id,
                    source_obs_index=int(obs_index),
                    source_cell_id=str(cell_id),
                    selection_origin="automatic",
                    input_x_value=float(x_value),
                    positive_threshold=float(
                        payload.parameters.automatic_positive_threshold
                    ),
                    nearest_same_marker_positive_distance_px=nearest,
                    min_unassigned_pixels_per_bin=minimum_pixels,
                    min_unassigned_fraction=minimum_fraction,
                    eligible=not reasons,
                    selected=False,
                    reason=";".join(reasons),
                )
            )
    return tuple(records)


def _radial_profile(
    image: FloatArray,
    mask: NDArray[np.int64],
    object_id: int,
    object_slice: tuple[slice, slice],
    source_strength: float,
    max_halo_px: int,
    global_background: float,
) -> tuple[FloatArray, float, float, str, str]:
    image_shape = (int(mask.shape[0]), int(mask.shape[1]))
    patch_slice = _expanded_slice(object_slice, image_shape, max_halo_px)
    patch_mask = mask[patch_slice]
    patch_image = image[patch_slice]
    source = patch_mask == object_id
    distance = ndimage.distance_transform_edt(~source)
    unassigned = patch_mask == 0

    background_method = "global_unassigned_median"
    background = float(global_background)
    if max_halo_px >= 3:
        outer = (
            unassigned
            & (distance > float(max_halo_px - 1))
            & (distance <= float(max_halo_px))
        )
        if np.count_nonzero(outer) >= 16:
            background = float(np.median(patch_image[outer]))
            background_method = "outermost_unassigned_bin_median"

    source_excess = float(source_strength - background)
    if not np.isfinite(source_excess) or source_excess <= np.finfo(np.float32).eps:
        return (
            np.full(max_halo_px, np.nan, dtype=np.float32),
            background,
            source_excess,
            background_method,
            "non_positive_source_excess",
        )

    profile: FloatArray = np.full(max_halo_px, np.nan, dtype=np.float32)
    for bin_index in range(max_halo_px):
        radial_pixels = (
            unassigned
            & (distance > float(bin_index))
            & (distance <= float(bin_index + 1))
        )
        if np.any(radial_pixels):
            median_intensity = float(np.median(patch_image[radial_pixels]))
            profile[bin_index] = max(median_intensity - background, 0.0) / source_excess
    if not np.any(np.isfinite(profile)):
        return profile, background, source_excess, background_method, "no_radial_pixels"
    return profile, background, source_excess, background_method, ""


def extract_exemplar_profiles(payload: ProfileWorkerPayload) -> tuple[ExemplarProfile, ...]:
    """Load one ROI and extract all requested marker/exemplar radial profiles."""

    from tifffile import imread

    mask = _validate_mask(imread(payload.mask_path), path=payload.mask_path)
    anchors = source_anchor_labels(mask, payload.parameters.source_anchor_dilation_px)
    slices = _object_slices(mask)
    unassigned = mask == 0
    profiles: list[ExemplarProfile] = []
    for marker, requested in payload.exemplar_labels.items():
        image_path = payload.channel_paths[marker]
        image = _validate_image(
            imread(image_path),
            path=image_path,
            expected_shape=(int(mask.shape[0]), int(mask.shape[1])),
        )
        global_background = float(np.median(image[unassigned])) if np.any(unassigned) else float(np.median(image))
        strengths = label_quantiles(
            image,
            anchors,
            payload.parameters.source_anchor_quantile,
            requested_labels=requested,
        )
        for object_id in requested:
            if object_id not in slices or object_id not in strengths:
                profiles.append(
                    ExemplarProfile(
                        marker=marker,
                        roi=payload.roi,
                        object_id=int(object_id),
                        profile=np.full(payload.parameters.max_halo_px, np.nan, dtype=np.float32),
                        source_strength=float("nan"),
                        background=global_background,
                        source_excess_strength=float("nan"),
                        background_method="unavailable",
                        valid=False,
                        reason="exemplar_label_missing_from_mask",
                    )
                )
                continue
            profile, background, source_excess, method, reason = _radial_profile(
                image,
                mask,
                int(object_id),
                slices[int(object_id)],
                strengths[int(object_id)],
                payload.parameters.max_halo_px,
                global_background,
            )
            profiles.append(
                ExemplarProfile(
                    marker=marker,
                    roi=payload.roi,
                    object_id=int(object_id),
                    profile=profile,
                    source_strength=float(strengths[int(object_id)]),
                    background=background,
                    source_excess_strength=source_excess,
                    background_method=method,
                    valid=not reason,
                    reason=reason,
                )
            )
    return tuple(profiles)


def aggregate_marker_profiles(
    marker_names: Sequence[str],
    exemplar_profiles: Sequence[ExemplarProfile],
    configured_counts: Mapping[str, int],
    parameters: HaloParameters,
) -> tuple[dict[str, MarkerHaloProfile], list[str]]:
    """Aggregate valid exemplar profiles with median and interquartile spread."""

    grouped: dict[str, list[ExemplarProfile]] = {str(marker): [] for marker in marker_names}
    for profile in exemplar_profiles:
        grouped[profile.marker].append(profile)
    result: dict[str, MarkerHaloProfile] = {}
    warnings: list[str] = []
    for marker in marker_names:
        configured = int(configured_counts.get(marker, 0))
        valid = [profile for profile in grouped[marker] if profile.valid]
        if configured < parameters.min_exemplars:
            reason = (
                f"insufficient configured exemplars ({configured} < {parameters.min_exemplars})"
            )
        elif len(valid) < parameters.min_exemplars:
            reason = (
                f"insufficient valid exemplar profiles ({len(valid)} < {parameters.min_exemplars})"
            )
        else:
            reason = ""
        if reason:
            nan_profile: FloatArray = np.full(
                parameters.max_halo_px, np.nan, dtype=np.float32
            )
            result[marker] = MarkerHaloProfile(
                marker=marker,
                available=False,
                raw_median=nan_profile.copy(),
                final=np.zeros(parameters.max_halo_px, dtype=np.float32),
                q25=nan_profile.copy(),
                q75=nan_profile.copy(),
                n_configured_exemplars=configured,
                n_valid_exemplars=len(valid),
                source_threshold=float("nan"),
                effective_extent_px=0.0,
                skip_reason=reason,
            )
            warnings.append(f"Skipped marker {marker!r}: {reason}.")
            continue

        matrix = np.vstack([profile.profile for profile in valid]).astype(np.float32)
        median: FloatArray = np.full(
            parameters.max_halo_px, np.nan, dtype=np.float32
        )
        q25 = median.copy()
        q75 = median.copy()
        for bin_index in range(parameters.max_halo_px):
            values = matrix[:, bin_index]
            values = values[np.isfinite(values)]
            if values.size:
                median[bin_index] = float(np.median(values))
                q25[bin_index] = float(np.quantile(values, 0.25))
                q75[bin_index] = float(np.quantile(values, 0.75))
        final = np.nan_to_num(median, nan=0.0, posinf=0.0, neginf=0.0)
        final = np.maximum(final, 0.0).astype(np.float32)
        strengths = np.asarray([profile.source_strength for profile in valid], dtype=float)
        threshold = float(np.quantile(strengths, parameters.source_threshold_quantile))
        positive_bins = np.flatnonzero(final > 0)
        extent = float(positive_bins[-1] + 1) if positive_bins.size else 0.0
        result[marker] = MarkerHaloProfile(
            marker=marker,
            available=True,
            raw_median=median,
            final=final,
            q25=q25,
            q75=q75,
            n_configured_exemplars=configured,
            n_valid_exemplars=len(valid),
            source_threshold=threshold,
            effective_extent_px=extent,
            skip_reason="",
        )
    return result, warnings


def _label_reductions(
    values: FloatArray,
    mask: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    maximum = int(mask.max(initial=0))
    sums = np.bincount(
        mask.ravel(),
        weights=values.ravel().astype(np.float64, copy=False),
        minlength=maximum + 1,
    )
    counts = np.bincount(mask.ravel(), minlength=maximum + 1)
    return (
        np.asarray(sums, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
    )


def estimate_roi_background(
    image: FloatArray,
    mask: NDArray[np.int64],
    source_labels: Sequence[int],
    max_halo_px: int,
) -> tuple[float, str, int]:
    """Estimate robust ROI/marker background outside segmented source halos."""

    unassigned = mask == 0
    candidates = unassigned
    method = "all_unassigned_median"
    if source_labels:
        source_pixels = np.isin(mask, np.asarray(source_labels, dtype=np.int64))
        outside_halo = ndimage.distance_transform_edt(~source_pixels) > float(max_halo_px)
        preferred = unassigned & outside_halo
        minimum = max(32, int(round(image.size * 0.001)))
        if np.count_nonzero(preferred) >= minimum:
            candidates = preferred
            method = "unassigned_outside_source_halos_median"
    if not np.any(candidates):
        candidates = np.ones(mask.shape, dtype=bool)
        method = "whole_image_median_fallback"
    return float(np.median(image[candidates])), method, int(np.count_nonzero(candidates))


def project_source_halos_with_sources(
    mask: np.ndarray,
    source_strengths: Mapping[int, float],
    source_labels: Sequence[int],
    source_obs_indices: Mapping[int, int],
    profile: Any,
    *,
    background: float,
    max_halo_px: int,
    aggregation: str = "max",
) -> ProjectedHalo:
    """Project halos and retain the winning global AnnData source row per pixel.

    Source provenance is unambiguous only for ``max`` aggregation. ``sum``
    retains the established predicted intensities but returns only ``-1``
    source sentinels.
    """

    labels = _validate_mask(mask, path="in-memory mask")
    if aggregation not in {"max", "sum"}:
        raise ValueError("aggregation must be 'max' or 'sum'")
    curve = np.asarray(profile, dtype=np.float32)
    if curve.shape != (max_halo_px,):
        raise ValueError(
            f"Halo profile has shape {curve.shape}; expected ({max_halo_px},)"
        )
    curve = np.nan_to_num(curve, nan=0.0, posinf=0.0, neginf=0.0)
    predicted = np.zeros(labels.shape, dtype=np.float32)
    source_index = np.full(labels.shape, -1, dtype=np.int64)
    slices = _object_slices(labels)
    for raw_object_id in source_labels:
        object_id = int(raw_object_id)
        object_slice = slices.get(object_id)
        if object_slice is None:
            continue
        amplitude = max(float(source_strengths.get(object_id, 0.0)) - background, 0.0)
        if amplitude <= 0:
            continue
        image_shape = (int(labels.shape[0]), int(labels.shape[1]))
        patch_slice = _expanded_slice(object_slice, image_shape, max_halo_px)
        patch_mask = labels[patch_slice]
        source = patch_mask == object_id
        distance = ndimage.distance_transform_edt(~source)
        influence = (distance > 0) & (distance <= float(max_halo_px)) & ~source
        if not np.any(influence):
            continue
        bin_index = np.clip(np.ceil(distance).astype(np.int16) - 1, 0, max_halo_px - 1)
        contribution = np.zeros(patch_mask.shape, dtype=np.float32)
        contribution[influence] = curve[bin_index[influence]] * amplitude
        target = predicted[patch_slice]
        if aggregation == "max":
            if object_id not in source_obs_indices:
                raise KeyError(
                    f"Source segmentation label {object_id} has no global AnnData row mapping"
                )
            wins = contribution > target
            target[wins] = contribution[wins]
            target_source = source_index[patch_slice]
            target_source[wins] = int(source_obs_indices[object_id])
        else:
            target += contribution
    return ProjectedHalo(predicted=predicted, source_index=source_index)


def project_source_halos(
    mask: np.ndarray,
    source_strengths: Mapping[int, float],
    source_labels: Sequence[int],
    profile: Any,
    *,
    background: float,
    max_halo_px: int,
    aggregation: str = "max",
) -> FloatArray:
    """Project source halos while preserving the original intensity-only API."""

    projected = project_source_halos_with_sources(
        mask,
        source_strengths,
        source_labels,
        {int(label): int(label) for label in source_labels},
        profile,
        background=background,
        max_halo_px=max_halo_px,
        aggregation=aggregation,
    )
    return projected.predicted


def _aggregate_source_target_attribution(
    *,
    mask: IntArray,
    attributable: FloatArray,
    source_index: IntArray,
    target_rows: IntArray,
    target_labels: IntArray,
    marker_index: int,
    observed_sums: NDArray[np.float64],
    attributable_sums: NDArray[np.float64],
) -> tuple[
    tuple[SourceTargetAttribution, ...],
    IntArray,
    FloatArray,
    FloatArray,
]:
    """Reduce attributable pixels by target label and winning source row."""

    n_cells = len(target_rows)
    dominant_source: IntArray = np.full(n_cells, -1, dtype=np.int64)
    dominant_observed_fraction: FloatArray = np.zeros(n_cells, dtype=np.float32)
    dominant_attributable_fraction: FloatArray = np.zeros(
        n_cells, dtype=np.float32
    )
    valid = (mask > 0) & (source_index >= 0) & (attributable > 0)
    if not np.any(valid):
        return (
            (),
            dominant_source,
            dominant_observed_fraction,
            dominant_attributable_fraction,
        )

    maximum_label = int(mask.max(initial=0))
    label_to_row: IntArray = np.full(maximum_label + 1, -1, dtype=np.int64)
    label_to_local: IntArray = np.full(maximum_label + 1, -1, dtype=np.int64)
    label_to_row[target_labels] = target_rows
    label_to_local[target_labels] = np.arange(n_cells, dtype=np.int64)
    pixel_labels = mask[valid]
    mapped = label_to_row[pixel_labels] >= 0
    if not np.any(mapped):
        return (
            (),
            dominant_source,
            dominant_observed_fraction,
            dominant_attributable_fraction,
        )
    pixel_labels = pixel_labels[mapped]
    pixel_sources = source_index[valid][mapped]
    pixel_values = attributable[valid][mapped].astype(np.float64, copy=False)
    pixel_targets = label_to_row[pixel_labels]
    if np.any(pixel_targets == pixel_sources):
        raise RuntimeError(
            "A source cell was assigned attributable signal inside its own segmentation mask"
        )

    order = np.lexsort((pixel_sources, pixel_labels))
    sorted_labels = pixel_labels[order]
    sorted_sources = pixel_sources[order]
    sorted_values = pixel_values[order]
    starts = np.flatnonzero(
        np.r_[
            True,
            (sorted_labels[1:] != sorted_labels[:-1])
            | (sorted_sources[1:] != sorted_sources[:-1]),
        ]
    )
    grouped_labels = sorted_labels[starts]
    grouped_sources = sorted_sources[starts]
    grouped_sums = np.add.reduceat(sorted_values, starts)
    observed_denominators = observed_sums[grouped_labels]
    attributable_denominators = attributable_sums[grouped_labels]
    observed_fractions = np.divide(
        grouped_sums,
        observed_denominators,
        out=np.zeros_like(grouped_sums),
        where=observed_denominators > 0,
    )
    attributable_fractions = np.divide(
        grouped_sums,
        attributable_denominators,
        out=np.zeros_like(grouped_sums),
        where=attributable_denominators > 0,
    )
    grouped_targets = label_to_row[grouped_labels]
    records = tuple(
        SourceTargetAttribution(
            target_obs_index=int(target),
            marker_index=int(marker_index),
            source_obs_index=int(source),
            attributable_intensity=float(intensity),
            fraction_of_observed_signal=float(observed_fraction),
            fraction_of_attributable_signal=float(attributable_fraction),
        )
        for target, source, intensity, observed_fraction, attributable_fraction in zip(
            grouped_targets,
            grouped_sources,
            grouped_sums,
            observed_fractions,
            attributable_fractions,
            strict=True,
        )
    )

    dominant_order = np.lexsort(
        (grouped_sources, -grouped_sums, grouped_labels)
    )
    ranked_labels = grouped_labels[dominant_order]
    first = np.r_[True, ranked_labels[1:] != ranked_labels[:-1]]
    winners = dominant_order[first]
    winner_local = label_to_local[grouped_labels[winners]]
    dominant_source[winner_local] = grouped_sources[winners]
    dominant_observed_fraction[winner_local] = observed_fractions[winners].astype(
        np.float32
    )
    dominant_attributable_fraction[winner_local] = attributable_fractions[
        winners
    ].astype(np.float32)
    return (
        records,
        dominant_source,
        dominant_observed_fraction,
        dominant_attributable_fraction,
    )


def calculate_marker_halo_maps(
    mask: np.ndarray,
    image: np.ndarray,
    anchors: np.ndarray,
    halo: MarkerHaloProfile,
    parameters: HaloParameters,
    source_obs_indices: Mapping[int, int],
    *,
    roi: str,
    marker: str,
) -> MarkerHaloMaps:
    """Apply one learned marker halo and return transient pixel-level maps.

    The application worker and targeted QC gallery renderer share this helper,
    ensuring that visualized predicted, attributable, and residual maps use the
    exact scientific calculation that produces the AnnData scores. Strong mask
    labels absent from the AnnData remain segmented geometry and inform
    background exclusion, but only mapped cells can project attributable halos
    and appear in source provenance.
    """

    labels = _validate_mask(mask, path=f"{roi} segmentation mask")
    values = _validate_image(
        image,
        path=f"{roi}/{marker} raw marker image",
        expected_shape=(int(labels.shape[0]), int(labels.shape[1])),
    )
    anchor_labels = np.asarray(anchors, dtype=np.int64)
    if anchor_labels.shape != labels.shape:
        raise ValueError(
            f"ROI {roi!r} source-anchor labels have shape {anchor_labels.shape}; "
            f"expected {labels.shape}"
        )
    source_strengths = label_quantiles(
        values,
        anchor_labels,
        parameters.source_anchor_quantile,
    )
    if halo.available:
        strong_source_labels = tuple(
            sorted(
                int(label)
                for label, strength in source_strengths.items()
                if strength >= halo.source_threshold
            )
        )
        source_labels = tuple(
            label for label in strong_source_labels if label in source_obs_indices
        )
        unmapped_source_labels = tuple(
            label for label in strong_source_labels if label not in source_obs_indices
        )
    else:
        strong_source_labels = ()
        source_labels = ()
        unmapped_source_labels = ()
    background, method, background_pixels = estimate_roi_background(
        values,
        labels,
        strong_source_labels,
        parameters.max_halo_px,
    )
    if halo.available and source_labels:
        projected = project_source_halos_with_sources(
            labels,
            source_strengths,
            source_labels,
            source_obs_indices,
            halo.final,
            background=background,
            max_halo_px=parameters.max_halo_px,
            aggregation=parameters.halo_aggregation,
        )
    else:
        projected = ProjectedHalo(
            predicted=np.zeros(labels.shape, dtype=np.float32),
            source_index=np.full(labels.shape, -1, dtype=np.int64),
        )
    observed_excess = np.maximum(values - background, 0.0).astype(np.float32)
    attributable = np.minimum(observed_excess, projected.predicted).astype(np.float32)
    residual = np.maximum(observed_excess - projected.predicted, 0.0).astype(
        np.float32
    )
    return MarkerHaloMaps(
        observed_excess=observed_excess,
        projected=projected,
        attributable=attributable,
        residual=residual,
        source_strengths=source_strengths,
        source_labels=source_labels,
        unmapped_source_labels=unmapped_source_labels,
        background=background,
        background_method=method,
        background_pixels=background_pixels,
    )


def _apply_profiles_to_roi(payload: ApplicationWorkerPayload) -> ApplicationWorkerResult:
    from tifffile import imread

    mask = _validate_mask(imread(payload.mask_path), path=payload.mask_path)
    anchors = source_anchor_labels(mask, payload.parameters.source_anchor_dilation_px)
    present = set(np.unique(mask).tolist())
    missing = [int(label) for label in payload.target_labels if int(label) not in present]
    if missing:
        raise ValueError(
            f"ROI {payload.roi!r} is missing {len(missing)} mapped cell label(s); "
            f"examples: {missing[:10]}"
        )
    n_cells = len(payload.target_labels)
    n_markers = len(payload.marker_names)
    if np.any((payload.target_rows < 0) | (payload.target_rows >= payload.total_cells)):
        raise ValueError(f"ROI {payload.roi!r} contains invalid global AnnData row indices")
    scores: FloatArray = np.zeros((n_cells, n_markers), dtype=np.float32)
    classic = np.zeros_like(scores)
    attributable_intensity = np.zeros_like(scores)
    residual_intensity = np.zeros_like(scores)
    dominant_source_indices: IntArray = np.full(
        (n_cells, n_markers), -1, dtype=np.int64
    )
    dominant_source_observed_fractions = np.zeros_like(scores)
    dominant_source_attributable_fractions = np.zeros_like(scores)
    source_target_attributions: list[SourceTargetAttribution] = []
    background_records: list[dict[str, Any]] = []
    source_obs_indices = {
        int(label): int(row)
        for label, row in zip(payload.target_labels, payload.target_rows, strict=True)
    }

    for marker_index, (marker, image_path) in enumerate(
        zip(payload.marker_names, payload.channel_paths, strict=True)
    ):
        image = _validate_image(
            imread(image_path),
            path=image_path,
            expected_shape=(int(mask.shape[0]), int(mask.shape[1])),
        )
        halo = payload.profiles[marker]
        maps = calculate_marker_halo_maps(
            mask,
            image,
            anchors,
            halo,
            payload.parameters,
            source_obs_indices,
            roi=payload.roi,
            marker=marker,
        )
        source_strengths = maps.source_strengths
        source_labels = maps.source_labels
        background = maps.background
        projected = maps.projected
        observed_excess = maps.observed_excess
        attributable = maps.attributable
        residual = maps.residual
        raw_sums, counts = _label_reductions(image, mask)
        observed_sums, _ = _label_reductions(observed_excess, mask)
        attributable_sums, _ = _label_reductions(attributable, mask)
        residual_sums, _ = _label_reductions(residual, mask)
        target_labels = payload.target_labels
        target_counts = counts[target_labels]
        if np.any(target_counts <= 0):
            raise ValueError(f"ROI {payload.roi!r} contains mapped labels with zero pixels")
        classic[:, marker_index] = (raw_sums[target_labels] / target_counts).astype(np.float32)
        attributable_intensity[:, marker_index] = (
            attributable_sums[target_labels] / target_counts
        ).astype(np.float32)
        residual_intensity[:, marker_index] = (
            residual_sums[target_labels] / target_counts
        ).astype(np.float32)
        denominators = observed_sums[target_labels]
        fractions = np.divide(
            attributable_sums[target_labels],
            denominators,
            out=np.zeros(n_cells, dtype=np.float64),
            where=denominators > 0,
        )
        scores[:, marker_index] = np.clip(fractions, 0.0, 1.0).astype(np.float32)
        if payload.parameters.halo_aggregation == "max":
            (
                marker_records,
                marker_dominant_sources,
                marker_dominant_observed,
                marker_dominant_attributable,
            ) = _aggregate_source_target_attribution(
                mask=mask,
                attributable=attributable,
                source_index=projected.source_index,
                target_rows=payload.target_rows,
                target_labels=payload.target_labels,
                marker_index=marker_index,
                observed_sums=observed_sums,
                attributable_sums=attributable_sums,
            )
            source_target_attributions.extend(marker_records)
            dominant_source_indices[:, marker_index] = marker_dominant_sources
            dominant_source_observed_fractions[:, marker_index] = (
                marker_dominant_observed
            )
            dominant_source_attributable_fractions[:, marker_index] = (
                marker_dominant_attributable
            )
        background_records.append(
            {
                "roi": payload.roi,
                "marker": marker,
                "background": background,
                "background_method": maps.background_method,
                "background_pixels": maps.background_pixels,
                "source_cells": len(source_labels),
                "projected_source_cells": len(source_labels),
                "strong_source_cells_total": (
                    len(source_labels) + len(maps.unmapped_source_labels)
                ),
                "unmapped_strong_source_cells": len(maps.unmapped_source_labels),
                "mapped_segmented_cells": len(source_obs_indices),
                "mask_only_segmented_cells": (
                    len(set(source_strengths).difference(source_obs_indices))
                ),
                "segmented_cells": len(source_strengths),
            }
        )

    return ApplicationWorkerResult(
        roi=payload.roi,
        target_rows=payload.target_rows,
        scores=scores,
        classic_intensities=classic,
        attributable_intensities=attributable_intensity,
        residual_intensities=residual_intensity,
        dominant_source_indices=dominant_source_indices,
        dominant_source_observed_fractions=dominant_source_observed_fractions,
        dominant_source_attributable_fractions=(
            dominant_source_attributable_fractions
        ),
        source_target_attributions=tuple(source_target_attributions),
        background_records=tuple(background_records),
    )


def _run_roi_workers(
    payloads: Sequence[Any],
    function: Callable[[Any], Any],
    *,
    workers: int,
    phase: str,
) -> list[Any]:
    if not payloads:
        return []
    effective = min(workers, len(payloads))
    if effective <= 1:
        results = []
        for index, payload in enumerate(payloads, start=1):
            results.append(function(payload))
            LOGGER.info("%s: completed ROI %d/%d", phase, index, len(payloads))
        return results
    ordered: list[Any] = [None] * len(payloads)
    with ProcessPoolExecutor(max_workers=effective) as executor:
        futures = {
            executor.submit(function, payload): index
            for index, payload in enumerate(payloads)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            ordered[index] = future.result()
            completed += 1
            LOGGER.info("%s: completed ROI %d/%d", phase, completed, len(payloads))
    return ordered


def _canonical_exemplars(
    adata: Any,
    marker_names: Sequence[str],
    exemplar_obs: str,
) -> tuple[dict[int, str], tuple[str, ...]]:
    if exemplar_obs not in adata.obs.columns:
        raise KeyError(f"Input AnnData is missing exemplar observation {exemplar_obs!r}")
    if len(set(marker_names)) != len(marker_names):
        raise ValueError("AnnData marker names must be unique")
    available = set(marker_names)
    resolved: dict[int, str] = {}
    unknown: set[str] = set()
    for position, value in enumerate(adata.obs[exemplar_obs].to_numpy()):
        if pd.isna(value):
            continue
        marker = str(value).strip()
        if not marker:
            continue
        if marker not in available:
            unknown.add(marker)
            continue
        resolved[position] = marker
    return resolved, tuple(sorted(unknown))


def _dense_input_x(adata: Any) -> np.ndarray:
    if adata.X is None:
        raise ValueError("Input AnnData.X is required for exemplar selection and preservation")
    values = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (adata.n_obs, adata.n_vars):
        raise ValueError(
            f"Input AnnData.X has shape {matrix.shape}; expected {(adata.n_obs, adata.n_vars)}"
        )
    return matrix


def _stable_candidate_order(record: ExemplarSelectionRecord) -> int:
    token = (
        f"{record.marker}\0{record.roi}\0{record.source_obs_index}\0"
        f"{record.source_cell_id}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little")


def _balanced_automatic_selection(
    records: Sequence[ExemplarSelectionRecord],
    marker_names: Sequence[str],
    parameters: HaloParameters,
    manual_pairs: set[tuple[int, str]],
    manual_roi_counts: Mapping[tuple[str, str], int],
) -> tuple[ExemplarSelectionRecord, ...]:
    """Select reproducible X-score/ROI-balanced automatic exemplars."""

    updated = list(records)
    for marker in marker_names:
        marker_indices = [
            index
            for index, record in enumerate(updated)
            if record.marker == marker
            and record.eligible
            and (record.source_obs_index, marker) not in manual_pairs
        ]
        existing = sum(1 for _row, selected_marker in manual_pairs if selected_marker == marker)
        target = max(0, parameters.automatic_target_exemplars_per_marker - existing)
        if target == 0 or not marker_indices:
            continue
        values = np.asarray(
            [updated[index].input_x_value for index in marker_indices], dtype=float
        )
        if len(np.unique(values)) > 1:
            cut_points = np.unique(np.quantile(values, [1 / 3, 2 / 3]))
            bands = np.searchsorted(cut_points, values, side="right").astype(int)
        else:
            bands = np.zeros(len(values), dtype=int)
        band_for_index = {
            index: int(band) for index, band in zip(marker_indices, bands, strict=True)
        }
        selected_by_roi: dict[str, int] = {
            roi: int(count)
            for (selected_marker, roi), count in manual_roi_counts.items()
            if selected_marker == marker
        }
        selected_by_band: dict[int, int] = {}
        remaining = set(marker_indices)
        chosen = 0
        while remaining and chosen < target:
            usable = [
                index
                for index in remaining
                if selected_by_roi.get(updated[index].roi, 0)
                < parameters.automatic_max_exemplars_per_roi
            ]
            if not usable:
                break
            winner = min(
                usable,
                key=lambda index: (
                    selected_by_roi.get(updated[index].roi, 0),
                    selected_by_band.get(band_for_index[index], 0),
                    _stable_candidate_order(updated[index]),
                ),
            )
            updated[winner] = replace(updated[winner], selected=True)
            selected_by_roi[updated[winner].roi] = (
                selected_by_roi.get(updated[winner].roi, 0) + 1
            )
            band = band_for_index[winner]
            selected_by_band[band] = selected_by_band.get(band, 0) + 1
            remaining.remove(winner)
            chosen += 1

    manual_duplicates = {
        (record.source_obs_index, record.marker)
        for record in updated
        if (record.source_obs_index, record.marker) in manual_pairs
    }
    return tuple(
        replace(
            record,
            reason=(
                ";".join(filter(None, [record.reason, "already_selected_manually"]))
                if (record.source_obs_index, record.marker) in manual_duplicates
                else record.reason
            ),
        )
        for record in updated
    )


def run_neighbour_signal_analysis(
    adata: Any,
    roi_inputs: Sequence[ROIInput],
    identity: pd.DataFrame,
    *,
    roi_obs: str,
    object_id_obs: str,
    exemplar_obs: str,
    parameters: HaloParameters,
    n_jobs: str | int = "auto",
) -> NeighbourSignalResult:
    """Learn marker halos and apply them to every mapped cell and marker."""

    parameters.validate()
    marker_names = tuple(str(name) for name in adata.var_names)
    if not marker_names:
        raise ValueError("Input AnnData contains no markers")
    if adata.n_obs != len(identity):
        raise ValueError("Identity mapping must contain exactly one row per AnnData observation")
    required_identity = {"source_obs_position", roi_obs, object_id_obs}
    missing_identity = required_identity.difference(identity.columns)
    if missing_identity:
        raise KeyError(f"Identity mapping is missing columns: {sorted(missing_identity)}")
    positions = identity["source_obs_position"].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(positions), np.arange(adata.n_obs, dtype=np.int64)):
        raise ValueError("Identity source positions must cover the input AnnData exactly once")
    duplicates = identity.duplicated([roi_obs, object_id_obs], keep=False)
    if duplicates.any():
        raise ValueError("Identity mapping contains duplicate (ROI, object ID) pairs")

    contexts = {context.name: context for context in roi_inputs}
    mapped_rois = set(identity[roi_obs].astype(str))
    if mapped_rois != set(contexts):
        missing = sorted(mapped_rois.difference(contexts))
        extra = sorted(set(contexts).difference(mapped_rois))
        raise ValueError(f"ROI inputs and AnnData mapping differ; missing={missing}, extra={extra}")
    for context in roi_inputs:
        if tuple(context.channel_names) != marker_names:
            raise ValueError(
                f"ROI {context.name!r} channel order {context.channel_names} does not match "
                f"AnnData marker order {marker_names}"
            )

    worker_usage = resolve_analysis_workers(n_jobs, len(roi_inputs))
    LOGGER.info(
        "Neighbour signal analysis uses %d ROI worker(s); CPU limit=%d from %s.",
        worker_usage.effective,
        worker_usage.cpu_limit,
        worker_usage.limit_source,
    )

    input_x = _dense_input_x(adata)
    manual_exemplars: dict[int, str] = {}
    unknown_values: tuple[str, ...] = ()
    if parameters.exemplar_mode in {"manual", "augment"}:
        manual_exemplars, unknown_values = _canonical_exemplars(
            adata, marker_names, exemplar_obs
        )
    identity_by_position = identity.set_index("source_obs_position", drop=False)
    marker_to_index = {marker: index for index, marker in enumerate(marker_names)}
    manual_records: list[ExemplarSelectionRecord] = []
    manual_pairs: set[tuple[int, str]] = set()
    manual_roi_counts: dict[tuple[str, str], int] = {}
    for position, marker in manual_exemplars.items():
        row = identity_by_position.loc[position]
        roi_name = str(row[roi_obs])
        pair = (int(position), marker)
        manual_pairs.add(pair)
        manual_roi_counts[(marker, roi_name)] = (
            manual_roi_counts.get((marker, roi_name), 0) + 1
        )
        manual_records.append(
            ExemplarSelectionRecord(
                marker=marker,
                roi=roi_name,
                object_id=int(row[object_id_obs]),
                source_obs_index=int(position),
                source_cell_id=str(adata.obs_names[position]),
                selection_origin="manual",
                input_x_value=float(input_x[position, marker_to_index[marker]]),
                positive_threshold=float("nan"),
                nearest_same_marker_positive_distance_px=float("nan"),
                min_unassigned_pixels_per_bin=-1,
                min_unassigned_fraction=float("nan"),
                eligible=True,
                selected=True,
                reason="",
            )
        )

    automatic_records: tuple[ExemplarSelectionRecord, ...] = ()
    nonfinite_automatic_values = 0
    if parameters.exemplar_mode in {"automatic", "augment"}:
        candidate_payloads: list[CandidateWorkerPayload] = []
        for roi, roi_frame in identity.groupby(roi_obs, sort=True, observed=True):
            roi_name = str(roi)
            positions_for_roi = roi_frame["source_obs_position"].to_numpy(
                dtype=np.int64
            )
            labels_for_roi = roi_frame[object_id_obs].to_numpy(dtype=np.int64)
            positive_cells: dict[
                str, tuple[tuple[int, int, float, str], ...]
            ] = {}
            for marker_index, marker in enumerate(marker_names):
                marker_values = input_x[positions_for_roi, marker_index]
                finite = np.isfinite(marker_values)
                nonfinite_automatic_values += int(np.count_nonzero(~finite))
                positive = finite & (
                    marker_values >= parameters.automatic_positive_threshold
                )
                if not np.any(positive):
                    continue
                positive_cells[marker] = tuple(
                    (
                        int(position),
                        int(label),
                        float(value),
                        str(adata.obs_names[int(position)]),
                    )
                    for position, label, value in zip(
                        positions_for_roi[positive],
                        labels_for_roi[positive],
                        marker_values[positive],
                        strict=True,
                    )
                )
            if positive_cells:
                candidate_payloads.append(
                    CandidateWorkerPayload(
                        roi=roi_name,
                        mask_path=str(contexts[roi_name].mask_path),
                        positive_cells=positive_cells,
                        parameters=parameters,
                    )
                )
        inspected_chunks = _run_roi_workers(
            candidate_payloads,
            inspect_automatic_exemplar_candidates,
            workers=worker_usage.effective,
            phase="automatic exemplar eligibility",
        )
        inspected = tuple(record for chunk in inspected_chunks for record in chunk)
        automatic_records = _balanced_automatic_selection(
            inspected,
            marker_names,
            parameters,
            manual_pairs,
            manual_roi_counts,
        )

    exemplar_selection = tuple(manual_records) + automatic_records
    selected_pairs = {
        (record.source_obs_index, record.marker)
        for record in exemplar_selection
        if record.selected
    }
    selected_by_position: dict[int, list[str]] = {}
    configured_counts = {marker: 0 for marker in marker_names}
    for position, marker in sorted(selected_pairs):
        selected_by_position.setdefault(position, []).append(marker)
        configured_counts[marker] += 1
    LOGGER.info(
        "Exemplar mode %s selected %d marker/cell pairs from %d recorded candidates.",
        parameters.exemplar_mode,
        len(selected_pairs),
        len(exemplar_selection),
    )

    eligible_markers = {
        marker
        for marker, count in configured_counts.items()
        if count >= parameters.min_exemplars
    }
    profile_payloads: list[ProfileWorkerPayload] = []
    for roi, roi_frame in identity.groupby(roi_obs, sort=True, observed=True):
        roi_name = str(roi)
        context = contexts[roi_name]
        exemplar_labels: dict[str, list[int]] = {}
        for _row_index, row in roi_frame.iterrows():
            position = int(row["source_obs_position"])
            for exemplar_marker in selected_by_position.get(position, []):
                if exemplar_marker in eligible_markers:
                    exemplar_labels.setdefault(exemplar_marker, []).append(
                        int(row[object_id_obs])
                    )
        if not exemplar_labels:
            continue
        channel_paths = {
            marker: str(path)
            for marker, path in zip(context.channel_names, context.channel_files, strict=True)
            if marker in exemplar_labels
        }
        profile_payloads.append(
            ProfileWorkerPayload(
                roi=roi_name,
                mask_path=str(context.mask_path),
                channel_paths=channel_paths,
                exemplar_labels={
                    marker: tuple(labels) for marker, labels in exemplar_labels.items()
                },
                parameters=parameters,
            )
        )
    learned_chunks = _run_roi_workers(
        profile_payloads,
        extract_exemplar_profiles,
        workers=worker_usage.effective,
        phase="halo profile extraction",
    )
    exemplar_profiles = tuple(profile for chunk in learned_chunks for profile in chunk)
    profiles, warnings = aggregate_marker_profiles(
        marker_names,
        exemplar_profiles,
        configured_counts,
        parameters,
    )
    if unknown_values:
        warnings.insert(
            0,
            "Ignored exemplar marker value(s) absent from AnnData.var_names: "
            + ", ".join(unknown_values),
        )
    if nonfinite_automatic_values:
        warnings.insert(
            0,
            f"Ignored {nonfinite_automatic_values} non-finite AnnData.X value(s) during "
            "automatic exemplar candidate discovery.",
        )

    application_payloads: list[ApplicationWorkerPayload] = []
    for roi, roi_frame in identity.groupby(roi_obs, sort=True, observed=True):
        roi_name = str(roi)
        context = contexts[roi_name]
        application_payloads.append(
            ApplicationWorkerPayload(
                roi=roi_name,
                mask_path=str(context.mask_path),
                channel_paths=tuple(str(path) for path in context.channel_files),
                marker_names=marker_names,
                target_rows=roi_frame["source_obs_position"].to_numpy(dtype=np.int64),
                target_labels=roi_frame[object_id_obs].to_numpy(dtype=np.int64),
                total_cells=int(adata.n_obs),
                profiles=profiles,
                parameters=parameters,
            )
        )
    application_results = _run_roi_workers(
        application_payloads,
        _apply_profiles_to_roi,
        workers=worker_usage.effective,
        phase="halo application",
    )
    shape = (adata.n_obs, adata.n_vars)
    scores = np.zeros(shape, dtype=np.float32)
    classic = np.zeros(shape, dtype=np.float32)
    attributable = np.zeros(shape, dtype=np.float32)
    residual = np.zeros(shape, dtype=np.float32)
    dominant_source_indices = np.full(shape, -1, dtype=np.int64)
    dominant_source_observed_fractions = np.zeros(shape, dtype=np.float32)
    dominant_source_attributable_fractions = np.zeros(shape, dtype=np.float32)
    covered = np.zeros(adata.n_obs, dtype=bool)
    background_records: list[dict[str, Any]] = []
    source_target_attributions: list[SourceTargetAttribution] = []
    for result in application_results:
        if np.any(covered[result.target_rows]):
            raise RuntimeError(f"ROI result rows overlap while assembling {result.roi!r}")
        scores[result.target_rows] = result.scores
        classic[result.target_rows] = result.classic_intensities
        attributable[result.target_rows] = result.attributable_intensities
        residual[result.target_rows] = result.residual_intensities
        dominant_source_indices[result.target_rows] = result.dominant_source_indices
        dominant_source_observed_fractions[result.target_rows] = (
            result.dominant_source_observed_fractions
        )
        dominant_source_attributable_fractions[result.target_rows] = (
            result.dominant_source_attributable_fractions
        )
        covered[result.target_rows] = True
        background_records.extend(result.background_records)
        source_target_attributions.extend(result.source_target_attributions)
    if not np.all(covered):
        raise RuntimeError("ROI application did not return every AnnData observation")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise RuntimeError("Neighbour-attributable fractions are not finite and bounded in [0, 1]")
    unmapped_source_occurrences = sum(
        int(record.get("unmapped_strong_source_cells", 0))
        for record in background_records
    )
    unmapped_source_pairs = sum(
        int(record.get("unmapped_strong_source_cells", 0)) > 0
        for record in background_records
    )
    if unmapped_source_occurrences:
        warnings.append(
            f"Ignored {unmapped_source_occurrences} strong mask-only source occurrence(s) "
            f"across {unmapped_source_pairs} ROI-marker combination(s) because those "
            "segmentation labels have no output AnnData row. Their pixels remained "
            "segmented geometry and their strong-source neighbourhoods were excluded "
            "from ROI background estimation; only AnnData-mapped cells projected halos "
            "or appeared in source provenance."
        )
    source_provenance_available = parameters.halo_aggregation == "max"
    if not source_provenance_available:
        warnings.append(
            "Detailed source-cell provenance is disabled for halo_aggregation='sum' because "
            "multiple sources contribute to the same pixel; dominant-source layers use sentinel "
            "or zero values and the source-target table is empty. Use the recommended 'max' "
            "aggregation for source-resolved provenance."
        )
    return NeighbourSignalResult(
        marker_names=marker_names,
        scores=scores,
        classic_intensities=classic,
        attributable_intensities=attributable,
        residual_intensities=residual,
        dominant_source_indices=dominant_source_indices,
        dominant_source_observed_fractions=dominant_source_observed_fractions,
        dominant_source_attributable_fractions=(
            dominant_source_attributable_fractions
        ),
        source_target_attributions=tuple(source_target_attributions),
        source_provenance_available=source_provenance_available,
        profiles=profiles,
        exemplar_profiles=exemplar_profiles,
        exemplar_selection=exemplar_selection,
        background_records=tuple(background_records),
        unknown_exemplar_values=unknown_values,
        warnings=tuple(warnings),
        worker_usage=worker_usage,
    )


def marker_profile_summary(result: NeighbourSignalResult) -> pd.DataFrame:
    """Return marker-indexed profile availability and source metadata."""

    rows = []
    for marker in result.marker_names:
        profile = result.profiles[marker]
        rows.append(
            {
                "marker": marker,
                "halo_profile_available": bool(profile.available),
                "halo_n_configured_exemplars": int(profile.n_configured_exemplars),
                "halo_n_exemplars": int(profile.n_valid_exemplars),
                "halo_source_threshold": float(profile.source_threshold),
                "halo_effective_extent_px": float(profile.effective_extent_px),
                "halo_skip_reason": profile.skip_reason,
            }
        )
    return pd.DataFrame(rows).set_index("marker")


def exemplar_statistics_table(result: NeighbourSignalResult) -> pd.DataFrame:
    """Return scalar exemplar provenance suitable for AnnData.uns and reports."""

    columns = [
        "marker",
        "roi",
        "object_id",
        "source_obs_index",
        "source_cell_id",
        "selection_origin",
        "input_x_value",
        "source_strength",
        "background",
        "source_excess_strength",
        "background_method",
        "valid",
        "reason",
    ]
    selection_lookup = {
        (record.marker, record.roi, record.object_id): record
        for record in result.exemplar_selection
        if record.selected
    }
    rows = []
    for profile in result.exemplar_profiles:
        selection = selection_lookup.get((profile.marker, profile.roi, profile.object_id))
        rows.append({
            "marker": profile.marker,
            "roi": profile.roi,
            "object_id": profile.object_id,
            "source_obs_index": (
                selection.source_obs_index if selection is not None else -1
            ),
            "source_cell_id": (
                selection.source_cell_id if selection is not None else ""
            ),
            "selection_origin": (
                selection.selection_origin if selection is not None else "unknown"
            ),
            "input_x_value": (
                selection.input_x_value if selection is not None else float("nan")
            ),
            "source_strength": profile.source_strength,
            "background": profile.background,
            "source_excess_strength": profile.source_excess_strength,
            "background_method": profile.background_method,
            "valid": profile.valid,
            "reason": profile.reason,
        })
    return pd.DataFrame(rows, columns=columns)


def exemplar_profile_values_table(result: NeighbourSignalResult) -> pd.DataFrame:
    """Return each selected exemplar's normalized radial values in long form."""

    rows = []
    for profile in result.exemplar_profiles:
        for bin_index, value in enumerate(profile.profile):
            rows.append(
                {
                    "marker": profile.marker,
                    "roi": profile.roi,
                    "object_id": profile.object_id,
                    "distance_start_px": int(bin_index),
                    "distance_end_px": int(bin_index + 1),
                    "normalized_excess": float(value),
                    "valid": bool(profile.valid),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "marker",
            "roi",
            "object_id",
            "distance_start_px",
            "distance_end_px",
            "normalized_excess",
            "valid",
        ],
    )


def exemplar_selection_table(result: NeighbourSignalResult) -> pd.DataFrame:
    """Return automatic/manual candidate decisions and spatial eligibility evidence."""

    columns = [
        "marker",
        "roi",
        "object_id",
        "source_obs_index",
        "source_cell_id",
        "selection_origin",
        "input_x_value",
        "positive_threshold",
        "nearest_same_marker_positive_distance_px",
        "min_unassigned_pixels_per_bin",
        "min_unassigned_fraction",
        "eligible",
        "selected",
        "reason",
    ]
    rows = [
        {
            "marker": record.marker,
            "roi": record.roi,
            "object_id": record.object_id,
            "source_obs_index": record.source_obs_index,
            "source_cell_id": record.source_cell_id,
            "selection_origin": record.selection_origin,
            "input_x_value": record.input_x_value,
            "positive_threshold": record.positive_threshold,
            "nearest_same_marker_positive_distance_px": (
                record.nearest_same_marker_positive_distance_px
            ),
            "min_unassigned_pixels_per_bin": record.min_unassigned_pixels_per_bin,
            "min_unassigned_fraction": record.min_unassigned_fraction,
            "eligible": record.eligible,
            "selected": record.selected,
            "reason": record.reason,
        }
        for record in result.exemplar_selection
    ]
    return pd.DataFrame(rows, columns=columns)


def exemplar_selection_summary_table(result: NeighbourSignalResult) -> pd.DataFrame:
    """Summarise candidate filtering and final selection for every marker."""

    decisions = exemplar_selection_table(result)
    rows = []
    for marker in result.marker_names:
        marker_rows = decisions.loc[decisions["marker"].eq(marker)]
        automatic = marker_rows.loc[marker_rows["selection_origin"].eq("automatic")]
        reasons = automatic["reason"].fillna("").astype(str)
        rows.append(
            {
                "marker": marker,
                "automatic_x_positive_candidates": int(len(automatic)),
                "automatic_spatially_eligible": int(automatic["eligible"].sum()),
                "automatic_selected": int(automatic["selected"].sum()),
                "manual_selected": int(
                    marker_rows.loc[
                        marker_rows["selection_origin"].eq("manual"), "selected"
                    ].sum()
                ),
                "rejected_same_marker_clearance": int(
                    reasons.str.contains("same_marker_positive_within_clearance").sum()
                ),
                "rejected_radial_pixel_coverage": int(
                    reasons.str.contains("insufficient_unassigned_radial_pixels").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_source_target_table(
    adata: Any,
    result: NeighbourSignalResult,
    *,
    roi_obs: str,
    object_id_obs: str,
    population_obs: str | None = None,
) -> pd.DataFrame:
    """Build the sparse source-target provenance table from worker reductions."""

    missing = [
        column
        for column in (roi_obs, object_id_obs)
        if column not in adata.obs.columns
    ]
    if missing:
        raise KeyError(f"AnnData is missing source-target identity columns: {missing}")
    include_population = bool(
        population_obs is not None and population_obs in adata.obs.columns
    )
    columns = list(SOURCE_TARGET_COLUMNS)
    if include_population:
        columns.extend(["target_population", "source_population"])
    records = result.source_target_attributions
    if not records:
        if result.source_provenance_available and np.any(result.scores > 2e-6):
            raise RuntimeError(
                "Neighbour-attributable scores are non-zero but max-aggregation source "
                "provenance contains no relationships"
            )
        dtypes: dict[str, str] = {
            "target_obs_index": "int64",
            "target_cell_id": "string",
            "target_roi": "string",
            "target_segmentation_label": "int64",
            "marker": "string",
            "source_obs_index": "int64",
            "source_cell_id": "string",
            "source_roi": "string",
            "source_segmentation_label": "int64",
            "attributable_intensity": "float64",
            "fraction_of_observed_signal": "float64",
            "fraction_of_attributable_signal": "float64",
            "target_population": "string",
            "source_population": "string",
        }
        return pd.DataFrame(
            {column: pd.Series(dtype=dtypes[column]) for column in columns}
        )

    target_indices = np.fromiter(
        (record.target_obs_index for record in records),
        dtype=np.int64,
        count=len(records),
    )
    marker_indices = np.fromiter(
        (record.marker_index for record in records),
        dtype=np.int64,
        count=len(records),
    )
    source_indices = np.fromiter(
        (record.source_obs_index for record in records),
        dtype=np.int64,
        count=len(records),
    )
    if np.any(
        (target_indices < 0)
        | (target_indices >= adata.n_obs)
        | (source_indices < 0)
        | (source_indices >= adata.n_obs)
    ):
        raise RuntimeError("Source-target provenance contains invalid global AnnData rows")
    if np.any((marker_indices < 0) | (marker_indices >= adata.n_vars)):
        raise RuntimeError("Source-target provenance contains invalid marker indices")
    attributable_intensities = np.fromiter(
        (record.attributable_intensity for record in records),
        dtype=np.float64,
        count=len(records),
    )
    observed_fractions = np.fromiter(
        (record.fraction_of_observed_signal for record in records),
        dtype=np.float64,
        count=len(records),
    )
    attributable_fractions = np.fromiter(
        (record.fraction_of_attributable_signal for record in records),
        dtype=np.float64,
        count=len(records),
    )
    if not np.all(np.isfinite(attributable_intensities)) or np.any(
        attributable_intensities <= 0
    ):
        raise RuntimeError("Source-target attributable intensities must be finite and positive")
    for name, values in (
        ("fraction_of_observed_signal", observed_fractions),
        ("fraction_of_attributable_signal", attributable_fractions),
    ):
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise RuntimeError(f"Source-target {name} values must be finite and bounded")

    observed_by_target = np.zeros(result.scores.shape, dtype=np.float64)
    attributable_by_target = np.zeros(result.scores.shape, dtype=np.float64)
    np.add.at(
        observed_by_target,
        (target_indices, marker_indices),
        observed_fractions,
    )
    np.add.at(
        attributable_by_target,
        (target_indices, marker_indices),
        attributable_fractions,
    )
    if result.source_provenance_available and not np.allclose(
        observed_by_target,
        result.scores,
        rtol=2e-5,
        atol=2e-6,
    ):
        raise RuntimeError(
            "Source-specific observed fractions do not reconstruct NeighbourAttributableFraction"
        )
    affected = observed_by_target > 0
    if result.source_provenance_available and np.any(affected) and not np.allclose(
        attributable_by_target[affected],
        1.0,
        rtol=2e-5,
        atol=2e-6,
    ):
        raise RuntimeError(
            "Source-specific attributable fractions do not sum to one for affected targets"
        )

    obs_names = adata.obs_names.astype(str).to_numpy()
    roi_values = adata.obs[roi_obs].astype(str).to_numpy()
    segmentation_labels = pd.to_numeric(
        adata.obs[object_id_obs], errors="raise"
    ).to_numpy(dtype=np.int64)
    marker_values = np.asarray(result.marker_names, dtype=object)[marker_indices]
    frame = pd.DataFrame(
        {
            "target_obs_index": target_indices,
            "target_cell_id": pd.array(obs_names[target_indices], dtype="string"),
            "target_roi": pd.array(roi_values[target_indices], dtype="string"),
            "target_segmentation_label": segmentation_labels[target_indices],
            "marker": pd.array(marker_values, dtype="string"),
            "source_obs_index": source_indices,
            "source_cell_id": pd.array(obs_names[source_indices], dtype="string"),
            "source_roi": pd.array(roi_values[source_indices], dtype="string"),
            "source_segmentation_label": segmentation_labels[source_indices],
            "attributable_intensity": attributable_intensities,
            "fraction_of_observed_signal": observed_fractions,
            "fraction_of_attributable_signal": attributable_fractions,
        }
    )
    if not frame["target_roi"].equals(frame["source_roi"]):
        raise RuntimeError("Source-target provenance crosses ROI boundaries")
    if include_population and population_obs is not None:
        populations = adata.obs[population_obs].astype("string").to_numpy()
        frame["target_population"] = pd.array(
            populations[target_indices], dtype="string"
        )
        frame["source_population"] = pd.array(
            populations[source_indices], dtype="string"
        )
    frame["_marker_index"] = marker_indices
    frame = frame.sort_values(
        ["target_obs_index", "_marker_index", "source_obs_index"],
        kind="stable",
    ).drop(columns="_marker_index")
    return frame.loc[:, columns].reset_index(drop=True)


def _safe_uns_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _safe_uns_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, tuple):
        return [_safe_uns_value(item) for item in value]
    if isinstance(value, list):
        return [_safe_uns_value(item) for item in value]
    return value


def build_output_anndata(
    input_adata: Any,
    result: NeighbourSignalResult,
    *,
    parameters: Mapping[str, Any],
    calculate_classic_intensities: bool,
    high_risk_threshold: float,
    source_target_table: pd.DataFrame | None = None,
    source_target_table_path: str | Path | None = None,
) -> Any:
    """Copy input AnnData and attach scores, layers, summaries, and provenance."""

    if input_adata.X is None:
        raise ValueError("Input AnnData.X is required so it can be preserved as original_X")
    output = input_adata.copy()
    original_x = input_adata.X.copy()
    backup_layer = ""
    if "original_X" in output.layers:
        backup_layer = "preexisting_original_X"
        suffix = 2
        while backup_layer in output.layers:
            backup_layer = f"preexisting_original_X_{suffix}"
            suffix += 1
        output.layers[backup_layer] = output.layers["original_X"].copy()
    output.layers["original_X"] = original_x
    if calculate_classic_intensities:
        output.layers["classic_intensities"] = result.classic_intensities.astype(
            np.float32,
            copy=False,
        )
    output.layers["neighbour_attributable_intensity"] = (
        result.attributable_intensities.astype(np.float32, copy=False)
    )
    output.layers["residual_excess_intensity"] = result.residual_intensities.astype(
        np.float32,
        copy=False,
    )
    output.layers["dominant_source_index"] = result.dominant_source_indices.astype(
        np.int64,
        copy=False,
    )
    output.layers["dominant_source_observed_fraction"] = (
        result.dominant_source_observed_fractions.astype(np.float32, copy=False)
    )
    output.layers["dominant_source_attributable_fraction"] = (
        result.dominant_source_attributable_fractions.astype(np.float32, copy=False)
    )
    output.X = result.scores.astype(np.float32, copy=False)

    output.obs["halo_max_score"] = result.scores.max(axis=1).astype(np.float32)
    output.obs["halo_mean_score"] = result.scores.mean(axis=1).astype(np.float32)
    output.obs["halo_n_high_risk"] = np.count_nonzero(
        result.scores >= float(high_risk_threshold),
        axis=1,
    ).astype(np.int32)
    marker_summary = marker_profile_summary(result).reindex(output.var_names)
    for column in marker_summary.columns:
        output.var[column] = marker_summary[column].to_numpy()

    profile_matrix = np.vstack(
        [result.profiles[marker].raw_median for marker in result.marker_names]
    ).astype(np.float32)
    final_matrix = np.vstack(
        [result.profiles[marker].final for marker in result.marker_names]
    ).astype(np.float32)
    q25_matrix = np.vstack(
        [result.profiles[marker].q25 for marker in result.marker_names]
    ).astype(np.float32)
    q75_matrix = np.vstack(
        [result.profiles[marker].q75 for marker in result.marker_names]
    ).astype(np.float32)
    background_table = pd.DataFrame(result.background_records)
    unknown_table = pd.DataFrame(
        {"unknown_exemplar_marker": list(result.unknown_exemplar_values)}
    )
    source_table = source_target_table
    if source_table is None:
        source_table = pd.DataFrame()
    source_table_metadata = {
        "schema_version": 1,
        "available": bool(result.source_provenance_available),
        "path": str(source_target_table_path) if source_target_table_path else "",
        "format": "parquet",
        "relationships": int(len(source_table)),
        "columns": [str(column) for column in source_table.columns],
        "dtypes": {
            str(column): str(dtype)
            for column, dtype in source_table.dtypes.items()
        },
        "authoritative_identity": (
            "target_obs_index/source_obs_index are zero-based global rows of this output AnnData; "
            "cell IDs are the corresponding obs_names."
        ),
        "interpretation": (
            "A source cell is a neighbouring cell whose projected marker halo spatially explains "
            "signal observed inside the target cell mask; this does not prove physical transfer."
        ),
        "disabled_reason": (
            "Source-resolved provenance is unambiguous only for halo_aggregation='max'."
            if not result.source_provenance_available
            else ""
        ),
    }
    output.uns["marker_halo"] = {
        "schema_version": 5,
        "score_name": "NeighbourAttributableFraction",
        "interpretation": (
            "Spatial explainability/QC score: the fraction of observed background-subtracted "
            "raw signal inside a cell that coincides with plausible halos from strong neighbouring "
            "cells. It is not a probability or proof of artefact."
        ),
        "marker_names": np.asarray(result.marker_names, dtype=str),
        "distance_bin_edges_px": np.arange(
            final_matrix.shape[1] + 1,
            dtype=np.float32,
        ),
        "raw_median_profile": profile_matrix,
        "final_profile": final_matrix,
        "profile_q25": q25_matrix,
        "profile_q75": q75_matrix,
        "marker_summary": marker_summary.copy(),
        "exemplar_statistics": exemplar_statistics_table(result),
        "exemplar_profile_values": exemplar_profile_values_table(result),
        "exemplar_selection": exemplar_selection_table(result),
        "exemplar_selection_summary": exemplar_selection_summary_table(result),
        "exemplar_selection_interpretation": (
            "Automatic candidates are positive in the preserved input AnnData.X, sufficiently "
            "far from another X-positive cell for the same marker, and retain enough unassigned "
            "radial pixels after other segmented cells are excluded. Selection is reproducibly "
            "balanced across ROIs and X-score ranges. X chooses candidates only; learned halo "
            "values and final scores come from raw pixels and masks."
        ),
        "mask_only_source_interpretation": (
            "Segmentation labels absent from the input AnnData may remain after cell "
            "filtering. They remain occupied segmented geometry and strong mask-only "
            "labels are excluded when selecting ROI background pixels, but they do not "
            "project halos or appear as named sources because source provenance must map "
            "to an authoritative output AnnData row. Counts are stored per ROI and marker "
            "in roi_marker_backgrounds."
        ),
        "roi_marker_backgrounds": background_table,
        "unknown_exemplar_markers": unknown_table,
        "source_target_table": source_table_metadata,
        "parameters": _safe_uns_value(dict(parameters)),
        "worker_usage": {
            "requested": result.worker_usage.requested,
            "effective": result.worker_usage.effective,
            "cpu_limit": result.worker_usage.cpu_limit,
            "limit_source": result.worker_usage.limit_source,
        },
        "layer_semantics": {
            "classic_intensities": "Mean raw marker intensity over each cell mask.",
            "neighbour_attributable_intensity": (
                "Mean observed excess intensity per cell pixel assigned to projected neighbour halos."
            ),
            "residual_excess_intensity": (
                "Mean max(observed excess - projected neighbour halo, 0) per cell pixel."
            ),
            "original_X": "Unmodified input AnnData.X expression/confidence matrix.",
            "dominant_source_index": (
                "Zero-based global AnnData row of the neighbouring source contributing the largest "
                "attributable intensity for each cell and marker; -1 means no attributable source."
            ),
            "dominant_source_observed_fraction": (
                "Fraction of target observed excess signal explained by its dominant spatial source."
            ),
            "dominant_source_attributable_fraction": (
                "Fraction of the target's total neighbour-attributable component assigned to its "
                "dominant spatial source."
            ),
        },
        "preexisting_original_X_backup_layer": backup_layer,
    }
    return output


__all__ = [
    "CandidateWorkerPayload",
    "ExemplarSelectionRecord",
    "HaloParameters",
    "MarkerHaloMaps",
    "MarkerHaloProfile",
    "NeighbourSignalResult",
    "ProjectedHalo",
    "SourceTargetAttribution",
    "WorkerUsage",
    "aggregate_marker_profiles",
    "build_output_anndata",
    "build_source_target_table",
    "calculate_marker_halo_maps",
    "estimate_roi_background",
    "exemplar_selection_summary_table",
    "exemplar_selection_table",
    "exemplar_profile_values_table",
    "exemplar_statistics_table",
    "extract_exemplar_profiles",
    "label_quantiles",
    "inspect_automatic_exemplar_candidates",
    "marker_profile_summary",
    "project_source_halos",
    "project_source_halos_with_sources",
    "resolve_analysis_workers",
    "run_neighbour_signal_analysis",
    "source_anchor_labels",
]
