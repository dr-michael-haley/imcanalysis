"""Analysis and reporting helpers for Nimbus normalization-value scans.

The functions in this module operate on compact tabular results.  Nimbus model
loading and image orchestration stay in the stage script so these summaries can
be tested without CUDA or the Nimbus dependency.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from SpatialBiologyToolkit.nimbus_normalization import (
    load_normalization_file,
    resolve_normalization_parameters,
    write_normalization_csv,
)

DEFAULT_VMAX_FACTORS: tuple[float, ...] = (
    0.25,
    2**-1.5,
    0.5,
    2**-0.5,
    1.0,
    2**0.5,
    2.0,
    2**1.5,
    4.0,
)
SCORE_QUANTILES: tuple[tuple[float, str], ...] = (
    (0.05, "score_q05"),
    (0.25, "score_q25"),
    (0.50, "score_median"),
    (0.75, "score_q75"),
    (0.95, "score_q95"),
)


@dataclass(frozen=True)
class NormalizationScanAnalysis:
    """Tabular summaries and provisional stable-range recommendations."""

    candidate_summary: pd.DataFrame
    threshold_summary: pd.DataFrame
    roi_summary: pd.DataFrame
    recommendations: pd.DataFrame


@dataclass(frozen=True)
class IntracellularExpressionSummary:
    """Raw intracellular-pixel summary used to select scan ROIs."""

    masked_pixel_count: int
    above_background_pixel_count: int
    above_background_fraction: float
    mean_above_background: float


def safe_marker_filename(marker: object) -> str:
    """Return a deterministic filesystem-safe marker label."""

    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(marker)).strip("._")
    return value or "marker"


def select_scan_rois(
    available_rois: Sequence[str],
    *,
    requested_rois: Sequence[str] | None = None,
    max_rois: int = 10,
    random_seed: int = 0,
) -> list[str]:
    """Select a reproducible ROI subset while preserving canonical ROI order."""

    available = list(dict.fromkeys(str(value) for value in available_rois))
    if not available:
        raise ValueError("Nimbus normalization scan requires at least one usable ROI.")
    if max_rois < 0:
        raise ValueError("max_rois must be zero (all ROIs) or a positive integer.")

    if requested_rois:
        requested = list(dict.fromkeys(str(value) for value in requested_rois))
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(
                "Requested Nimbus normalization scan ROI(s) are unavailable: "
                f"{missing}."
            )
        requested_set = set(requested)
        return [roi for roi in available if roi in requested_set]

    if max_rois == 0 or len(available) <= max_rois:
        return available
    generator = np.random.default_rng(int(random_seed))
    selected_indices = {
        int(value)
        for value in generator.choice(len(available), size=max_rois, replace=False)
    }
    return [roi for index, roi in enumerate(available) if index in selected_indices]


def summarize_intracellular_expression(
    raw_image: np.ndarray,
    segmentation_mask: np.ndarray,
    *,
    background_value: float = 0.0,
) -> IntracellularExpressionSummary:
    """Summarise raw marker signal inside cells and above a background value.

    This deliberately works on pixels rather than segmented-cell measurements:
    it is a cheap pre-AnnData statistic for ranking ROIs before repeated Nimbus
    inference. Pixels must be finite, lie within a non-zero segmentation label,
    and be strictly greater than ``background_value`` to contribute to the mean.
    """

    image = np.asarray(raw_image)
    mask = np.asarray(segmentation_mask)
    if image.shape != mask.shape:
        raise ValueError(
            "Raw marker image and segmentation mask must have identical shapes; "
            f"got {image.shape} and {mask.shape}."
        )
    threshold = float(background_value)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("background_value must be finite and non-negative.")

    intracellular = image[mask > 0].astype(float, copy=False)
    intracellular = intracellular[np.isfinite(intracellular)]
    if intracellular.size == 0:
        raise ValueError("Segmentation mask contains no finite intracellular pixels.")
    above_background = intracellular[intracellular > threshold]
    qualifying_count = int(above_background.size)
    return IntracellularExpressionSummary(
        masked_pixel_count=int(intracellular.size),
        above_background_pixel_count=qualifying_count,
        above_background_fraction=float(qualifying_count / intracellular.size),
        mean_above_background=(
            float(np.mean(above_background)) if qualifying_count else 0.0
        ),
    )


def rank_rois_by_expression(roi_scores: Mapping[str, float]) -> list[str]:
    """Return ROI names ordered from low to high expression, with stable ties."""

    scores = [(str(roi), float(score)) for roi, score in roi_scores.items()]
    if not scores:
        raise ValueError("At least one ROI expression score is required.")
    invalid = [roi for roi, score in scores if not math.isfinite(score)]
    if invalid:
        raise ValueError(f"ROI expression scores must be finite; invalid ROI(s): {invalid}.")
    canonical_order = {roi: index for index, (roi, _score) in enumerate(scores)}
    return [
        roi
        for roi, _score in sorted(
            scores, key=lambda item: (item[1], canonical_order[item[0]])
        )
    ]


def select_rois_across_expression_range(
    roi_scores: Mapping[str, float], *, max_rois: int = 10
) -> list[str]:
    """Select approximately quantile-spaced ROIs across an expression ranking.

    The lowest- and highest-expression ROIs are always included when at least
    two ROIs are requested. The returned list preserves the input/canonical ROI
    order so downstream image processing remains deterministic.
    """

    available = list(dict.fromkeys(str(roi) for roi in roi_scores))
    if max_rois < 0:
        raise ValueError("max_rois must be zero (all ROIs) or a positive integer.")
    ranked = rank_rois_by_expression(roi_scores)
    if max_rois == 0 or len(ranked) <= max_rois:
        return available
    if max_rois == 1:
        rank_indices = np.asarray([(len(ranked) - 1) // 2], dtype=int)
    else:
        rank_indices = np.rint(
            np.linspace(0, len(ranked) - 1, num=max_rois)
        ).astype(int)
    selected = {ranked[int(index)] for index in rank_indices}
    return [roi for roi in available if roi in selected]


def resolve_scan_markers(
    available_markers: Sequence[str], requested_markers: Sequence[str] | None
) -> list[str]:
    """Resolve exact or unique case-insensitive marker requests."""

    available = list(dict.fromkeys(str(value) for value in available_markers))
    if not available:
        raise ValueError("Nimbus normalization scan has no usable markers.")
    if not requested_markers:
        return available

    folded: dict[str, list[str]] = {}
    for marker in available:
        folded.setdefault(marker.casefold(), []).append(marker)
    resolved: list[str] = []
    for requested in requested_markers:
        value = str(requested)
        if value in available:
            match = value
        else:
            matches = folded.get(value.casefold(), [])
            if len(matches) != 1:
                raise ValueError(
                    f"Nimbus normalization scan marker {value!r} does not uniquely "
                    f"match an available marker. Available markers: {available}."
                )
            match = matches[0]
        if match not in resolved:
            resolved.append(match)
    return resolved


def resolve_marker_baseline_vmax(
    available_markers: Sequence[str],
    supplied_values: Mapping[str, float] | None,
) -> dict[str, float]:
    """Resolve a case-insensitive marker-to-baseline mapping to canonical names."""

    resolved: dict[str, float] = {}
    supplied = supplied_values or {}
    seen_requested: dict[str, str] = {}
    for requested_marker, raw_value in supplied.items():
        requested = str(requested_marker).strip()
        folded = requested.casefold()
        if not requested:
            raise ValueError("Marker baseline keys must be non-empty marker names.")
        if folded in seen_requested:
            raise ValueError(
                "Marker baseline keys must be unique ignoring case; found "
                f"{seen_requested[folded]!r} and {requested!r}."
            )
        seen_requested[folded] = requested
        marker = resolve_scan_markers(available_markers, [requested])[0]
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"Baseline Vmax for marker {requested!r} must be finite and "
                f"positive; got {raw_value!r}."
            )
        resolved[marker] = value
    return resolved


def resolve_marker_lower_thresholds(
    available_markers: Sequence[str],
    supplied_values: Mapping[str, float] | None,
) -> dict[str, float]:
    """Resolve non-negative per-marker lower thresholds to canonical names."""

    resolved: dict[str, float] = {}
    supplied = supplied_values or {}
    seen_requested: dict[str, str] = {}
    for requested_marker, raw_value in supplied.items():
        requested = str(requested_marker).strip()
        folded = requested.casefold()
        if not requested:
            raise ValueError("Marker lower-threshold keys must be non-empty names.")
        if folded in seen_requested:
            raise ValueError(
                "Marker lower-threshold keys must be unique ignoring case; found "
                f"{seen_requested[folded]!r} and {requested!r}."
            )
        seen_requested[folded] = requested
        marker = resolve_scan_markers(available_markers, [requested])[0]
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"Lower threshold for marker {requested!r} must be finite and "
                f"non-negative; got {raw_value!r}."
            )
        resolved[marker] = value
    return resolved


def load_scan_parameter_csv(
    path: str | Path,
    available_markers: Sequence[str],
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Load a CSV whose rows define scan markers, baseline Vmax, and lower bounds."""

    source = Path(path)
    if source.suffix.casefold() != ".csv":
        raise ValueError("Nimbus scan marker parameters must use a .csv file.")
    loaded = load_normalization_file(source)
    markers = resolve_scan_markers(available_markers, list(loaded))
    resolved = resolve_normalization_parameters(loaded, markers, require_all=True)
    return (
        markers,
        {marker: entry.vmax for marker, entry in resolved.items()},
        {marker: entry.lower_threshold for marker, entry in resolved.items()},
    )


def resolve_scan_marker_inputs(
    available_markers: Sequence[str],
    requested_markers: Sequence[str] | None,
    *,
    marker_parameters_path: str | Path | None = None,
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Resolve scan scope, allowing a CSV to define markers and their bounds."""

    if marker_parameters_path is not None and requested_markers is None:
        source = Path(marker_parameters_path)
        if source.suffix.casefold() == ".csv":
            return load_scan_parameter_csv(source, available_markers)
    return resolve_scan_markers(available_markers, requested_markers), {}, {}


def validate_positive_values(values: Sequence[float], *, label: str) -> list[float]:
    """Validate, sort, and de-duplicate positive finite values."""

    parsed = sorted({float(value) for value in values})
    if not parsed:
        raise ValueError(f"{label} must contain at least one value.")
    invalid = [value for value in parsed if not math.isfinite(value) or value <= 0]
    if invalid:
        raise ValueError(f"{label} values must be finite and positive; got {invalid}.")
    return parsed


def build_vmax_grid(
    marker: str,
    baseline_vmax: float,
    *,
    factors: Sequence[float] = DEFAULT_VMAX_FACTORS,
    marker_vmax_values: Mapping[str, Sequence[float]] | None = None,
    lower_threshold: float = 0.0,
) -> list[float]:
    """Build one marker's explicit or baseline-relative Vmax candidate grid."""

    baseline = float(baseline_vmax)
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError(
            f"Baseline Vmax for marker {marker!r} must be finite and positive; "
            f"got {baseline_vmax!r}."
        )
    overrides = marker_vmax_values or {}
    override_key = next(
        (key for key in overrides if str(key).casefold() == marker.casefold()), None
    )
    if override_key is not None:
        values = validate_positive_values(
            overrides[override_key], label=f"Explicit Vmax grid for {marker}"
        )
    else:
        valid_factors = validate_positive_values(factors, label="Vmax factors")
        values = sorted({baseline * factor for factor in valid_factors})
    if len(values) < 3:
        raise ValueError(
            f"Nimbus normalization scan for marker {marker!r} needs at least three "
            "distinct Vmax candidates to assess local stability."
        )
    lower = float(lower_threshold)
    if not math.isfinite(lower) or lower < 0:
        raise ValueError(
            f"Lower threshold for marker {marker!r} must be finite and non-negative."
        )
    if lower >= min(values):
        raise ValueError(
            f"Lower threshold {lower:g} for marker {marker!r} must be below every "
            f"scanned Vmax candidate; the smallest candidate is {min(values):g}."
        )
    return values


def _validate_scan_tables(
    cell_scores: pd.DataFrame, pixel_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_scores = {"marker", "vmax", "roi", "label", "nimbus_score"}
    required_pixels = {
        "marker",
        "vmax",
        "roi",
        "masked_pixel_count",
        "saturated_pixel_fraction",
    }
    missing_scores = sorted(required_scores - set(cell_scores.columns))
    missing_pixels = sorted(required_pixels - set(pixel_summary.columns))
    if missing_scores:
        raise ValueError(f"Cell-score table is missing columns: {missing_scores}.")
    if missing_pixels:
        raise ValueError(f"Pixel-summary table is missing columns: {missing_pixels}.")

    scores = cell_scores.copy()
    pixels = pixel_summary.copy()
    if "lower_threshold" not in pixels:
        pixels["lower_threshold"] = 0.0
    if "below_lower_threshold_fraction" not in pixels:
        pixels["below_lower_threshold_fraction"] = 0.0
    for column in ("vmax", "nimbus_score"):
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    for column in (
        "vmax",
        "masked_pixel_count",
        "saturated_pixel_fraction",
        "lower_threshold",
        "below_lower_threshold_fraction",
    ):
        pixels[column] = pd.to_numeric(pixels[column], errors="coerce")
    if scores.empty:
        raise ValueError("Nimbus normalization scan produced no cell scores.")
    if scores[["vmax", "nimbus_score"]].isna().any().any():
        raise ValueError(
            "Nimbus normalization scan cell scores contain non-numeric values."
        )
    if not np.isfinite(scores[["vmax", "nimbus_score"]].to_numpy()).all():
        raise ValueError("Nimbus normalization scan cell scores must be finite.")
    if ((scores["nimbus_score"] < 0) | (scores["nimbus_score"] > 1)).any():
        raise ValueError("Nimbus scores must lie in the closed interval [0, 1].")
    if (
        pixels[
            [
                "vmax",
                "masked_pixel_count",
                "saturated_pixel_fraction",
                "lower_threshold",
                "below_lower_threshold_fraction",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Nimbus normalization scan pixel summaries contain missing values."
        )
    if (
        (pixels["masked_pixel_count"] <= 0).any()
        or (pixels["saturated_pixel_fraction"] < 0).any()
        or (pixels["saturated_pixel_fraction"] > 1).any()
        or (pixels["lower_threshold"] < 0).any()
        or (pixels["below_lower_threshold_fraction"] < 0).any()
        or (pixels["below_lower_threshold_fraction"] > 1).any()
        or (pixels["lower_threshold"] >= pixels["vmax"]).any()
    ):
        raise ValueError(
            "Pixel counts must be positive, fractions must be in [0, 1], and each "
            "lower threshold must satisfy 0 <= lower_threshold < vmax."
        )
    for marker, marker_rows in pixels.groupby("marker", sort=False, observed=True):
        lower_values = marker_rows["lower_threshold"].to_numpy(dtype=float)
        if not np.allclose(lower_values, lower_values[0]):
            raise ValueError(
                f"Lower threshold must remain fixed throughout a Vmax scan; marker "
                f"{marker!r} contains multiple values."
            )

    duplicated = scores.duplicated(["marker", "vmax", "roi", "label"])
    if duplicated.any():
        raise ValueError(
            "Nimbus normalization scan cell identities must be unique within each "
            "marker/Vmax candidate."
        )
    return scores, pixels


def _threshold_column(threshold: float) -> str:
    return f"positive_fraction_{threshold:.12g}".replace(".", "_")


def _weighted_saturation(rows: pd.DataFrame) -> float:
    weights = rows["masked_pixel_count"].to_numpy(dtype=float)
    values = rows["saturated_pixel_fraction"].to_numpy(dtype=float)
    return float(np.average(values, weights=weights))


def _weighted_pixel_fraction(rows: pd.DataFrame, column: str) -> float:
    weights = rows["masked_pixel_count"].to_numpy(dtype=float)
    values = rows[column].to_numpy(dtype=float)
    return float(np.average(values, weights=weights))


def _candidate_summaries(
    scores: pd.DataFrame,
    pixels: pd.DataFrame,
    *,
    baseline_values: Mapping[str, float],
    positive_thresholds: Sequence[float],
    primary_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    roi_rows: list[dict[str, object]] = []

    grouped = scores.groupby(["marker", "vmax"], sort=True, observed=True)
    for (marker_value, vmax_value), rows in grouped:
        marker = str(marker_value)
        vmax = float(vmax_value)
        values = rows["nimbus_score"].to_numpy(dtype=float)
        baseline = float(baseline_values[marker])
        pixel_rows = pixels[
            (pixels["marker"].astype(str) == marker)
            & np.isclose(pixels["vmax"].to_numpy(dtype=float), vmax)
        ]
        if pixel_rows.empty:
            raise ValueError(
                f"Missing pixel summary for marker={marker!r}, Vmax={vmax:g}."
            )
        lower_values = pixel_rows["lower_threshold"].to_numpy(dtype=float)
        if not np.allclose(lower_values, lower_values[0]):
            raise ValueError(
                f"Lower threshold changed between ROIs for marker={marker!r}, "
                f"Vmax={vmax:g}."
            )
        row: dict[str, object] = {
            "marker": marker,
            "vmax": vmax,
            "baseline_vmax": baseline,
            "vmax_factor": vmax / baseline,
            "n_cells": len(rows),
            "n_rois": int(rows["roi"].astype(str).nunique()),
            "score_mean": float(np.mean(values)),
            "score_std": float(np.std(values)),
            "saturated_pixel_fraction": _weighted_saturation(pixel_rows),
            "lower_threshold": float(lower_values[0]),
            "below_lower_threshold_fraction": _weighted_pixel_fraction(
                pixel_rows, "below_lower_threshold_fraction"
            ),
        }
        for quantile, name in SCORE_QUANTILES:
            row[name] = float(np.quantile(values, quantile))
        for threshold in positive_thresholds:
            fraction = float(np.mean(values >= threshold))
            row[_threshold_column(threshold)] = fraction
            threshold_rows.append(
                {
                    "marker": marker,
                    "vmax": vmax,
                    "positive_score_threshold": float(threshold),
                    "positive_fraction": fraction,
                    "n_positive": int(np.sum(values >= threshold)),
                    "n_cells": len(values),
                }
            )
        row["primary_positive_fraction"] = float(np.mean(values >= primary_threshold))
        candidate_rows.append(row)

        for roi, roi_scores in rows.groupby("roi", sort=True, observed=True):
            roi_pixels = pixel_rows[pixel_rows["roi"].astype(str) == str(roi)]
            if len(roi_pixels) != 1:
                raise ValueError(
                    f"Expected one pixel summary for marker={marker!r}, Vmax={vmax:g}, "
                    f"ROI={roi!r}; got {len(roi_pixels)}."
                )
            roi_values = roi_scores["nimbus_score"].to_numpy(dtype=float)
            pixel_row = roi_pixels.iloc[0]
            roi_rows.append(
                {
                    "marker": marker,
                    "vmax": vmax,
                    "roi": str(roi),
                    "n_cells": len(roi_values),
                    "score_median": float(np.median(roi_values)),
                    "primary_positive_fraction": float(
                        np.mean(roi_values >= primary_threshold)
                    ),
                    "masked_pixel_count": int(pixel_row["masked_pixel_count"]),
                    "saturated_pixel_fraction": float(
                        pixel_row["saturated_pixel_fraction"]
                    ),
                    "lower_threshold": float(pixel_row["lower_threshold"]),
                    "below_lower_threshold_fraction": float(
                        pixel_row["below_lower_threshold_fraction"]
                    ),
                }
            )

    return (
        pd.DataFrame(candidate_rows)
        .sort_values(["marker", "vmax"])
        .reset_index(drop=True),
        pd.DataFrame(threshold_rows)
        .sort_values(["marker", "positive_score_threshold", "vmax"])
        .reset_index(drop=True),
        pd.DataFrame(roi_rows)
        .sort_values(["marker", "vmax", "roi"])
        .reset_index(drop=True),
    )


def _add_adjacency_metrics(
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    primary_threshold: float,
) -> pd.DataFrame:
    output = candidates.copy()
    metric_defaults = {
        "previous_vmax": np.nan,
        "log2_step_from_previous": np.nan,
        "positive_fraction_delta_from_previous": np.nan,
        "absolute_positive_fraction_delta_from_previous": np.nan,
        "positive_fraction_sensitivity_per_octave": np.nan,
        "call_flip_fraction_from_previous": np.nan,
        "mean_absolute_score_delta_from_previous": np.nan,
        "local_positive_fraction_sensitivity": np.nan,
        "local_call_flip_fraction": np.nan,
    }
    for column, value in metric_defaults.items():
        output[column] = value

    for marker, marker_candidates in output.groupby("marker", sort=False):
        indices = marker_candidates.sort_values("vmax").index.to_list()
        marker_scores = scores[scores["marker"].astype(str) == str(marker)]
        edge_sensitivities: list[float] = []
        edge_flips: list[float] = []
        for position in range(1, len(indices)):
            previous_index = indices[position - 1]
            current_index = indices[position]
            previous_vmax = float(output.loc[previous_index, "vmax"])
            current_vmax = float(output.loc[current_index, "vmax"])
            log_step = abs(math.log2(current_vmax / previous_vmax))
            previous = marker_scores[
                np.isclose(marker_scores["vmax"].to_numpy(dtype=float), previous_vmax)
            ].set_index(["roi", "label"])["nimbus_score"]
            current = marker_scores[
                np.isclose(marker_scores["vmax"].to_numpy(dtype=float), current_vmax)
            ].set_index(["roi", "label"])["nimbus_score"]
            if not previous.index.equals(current.index):
                previous, current = previous.align(current, join="outer")
                if previous.isna().any() or current.isna().any():
                    raise ValueError(
                        "Cell identities differ between adjacent Vmax candidates for "
                        f"marker {marker!r}."
                    )
            if previous.empty:
                raise ValueError(
                    f"No shared cells between adjacent Vmax candidates for marker {marker!r}."
                )
            previous_values = previous.to_numpy(dtype=float)
            current_values = current.to_numpy(dtype=float)
            fraction_delta = float(
                np.mean(current_values >= primary_threshold)
                - np.mean(previous_values >= primary_threshold)
            )
            absolute_delta = abs(fraction_delta)
            sensitivity = absolute_delta / log_step if log_step > 0 else np.inf
            call_flip = float(
                np.mean(
                    (current_values >= primary_threshold)
                    != (previous_values >= primary_threshold)
                )
            )
            output.loc[current_index, "previous_vmax"] = previous_vmax
            output.loc[current_index, "log2_step_from_previous"] = log_step
            output.loc[current_index, "positive_fraction_delta_from_previous"] = (
                fraction_delta
            )
            output.loc[
                current_index, "absolute_positive_fraction_delta_from_previous"
            ] = absolute_delta
            output.loc[current_index, "positive_fraction_sensitivity_per_octave"] = (
                sensitivity
            )
            output.loc[current_index, "call_flip_fraction_from_previous"] = call_flip
            output.loc[current_index, "mean_absolute_score_delta_from_previous"] = (
                float(np.mean(np.abs(current_values - previous_values)))
            )
            edge_sensitivities.append(float(sensitivity))
            edge_flips.append(call_flip)

        for position, index in enumerate(indices):
            adjacent_sensitivity = []
            adjacent_flips = []
            if position > 0:
                adjacent_sensitivity.append(edge_sensitivities[position - 1])
                adjacent_flips.append(edge_flips[position - 1])
            if position < len(indices) - 1:
                adjacent_sensitivity.append(edge_sensitivities[position])
                adjacent_flips.append(edge_flips[position])
            output.loc[index, "local_positive_fraction_sensitivity"] = max(
                adjacent_sensitivity
            )
            output.loc[index, "local_call_flip_fraction"] = max(adjacent_flips)
    return output


def _contiguous_true_ranges(mask: Sequence[bool]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            ranges.append((start, index - 1))
            start = None
    if start is not None:
        ranges.append((start, len(mask) - 1))
    return ranges


def _recommend_marker(
    rows: pd.DataFrame,
    *,
    stability_tolerance: float,
    call_flip_tolerance: float,
    saturation_tolerance: float,
    cliff_tolerance: float,
) -> dict[str, object]:
    ordered = rows.sort_values("vmax").reset_index(drop=True)
    marker = str(ordered.loc[0, "marker"])
    baseline = float(ordered.loc[0, "baseline_vmax"])
    stable = (
        (
            ordered["local_positive_fraction_sensitivity"].to_numpy(dtype=float)
            <= stability_tolerance
        )
        & (
            ordered["local_call_flip_fraction"].to_numpy(dtype=float)
            <= call_flip_tolerance
        )
        & (
            ordered["saturated_pixel_fraction"].to_numpy(dtype=float)
            <= saturation_tolerance
        )
    )
    stable_ranges = _contiguous_true_ranges(stable.tolist())
    reasons: list[str] = []

    if stable_ranges:

        def range_distance(bounds: tuple[int, int]) -> tuple[float, float]:
            start, end = bounds
            values = ordered.loc[start:end, "vmax"].to_numpy(dtype=float)
            distances = np.abs(np.log2(values / baseline))
            return float(np.min(distances)), -float(end - start + 1)

        chosen_start, chosen_end = min(stable_ranges, key=range_distance)
        chosen = ordered.loc[chosen_start:chosen_end]
        suggestion_index = int(
            np.argmin(np.abs(np.log2(chosen["vmax"].to_numpy(dtype=float) / baseline)))
        )
        suggestion = chosen.iloc[suggestion_index]
        status = "stable_plateau"
        stable_min = float(chosen["vmax"].min())
        stable_max = float(chosen["vmax"].max())
    else:
        sensitivity = ordered["local_positive_fraction_sensitivity"].to_numpy(
            dtype=float
        )
        saturation = ordered["saturated_pixel_fraction"].to_numpy(dtype=float)
        distance = np.abs(np.log2(ordered["vmax"].to_numpy(dtype=float) / baseline))
        penalty = (
            sensitivity / max(stability_tolerance, np.finfo(float).eps)
            + ordered["local_call_flip_fraction"].to_numpy(dtype=float)
            / max(call_flip_tolerance, np.finfo(float).eps)
            + np.maximum(0.0, saturation - saturation_tolerance)
            / max(saturation_tolerance, np.finfo(float).eps)
            + 0.05 * distance
        )
        suggestion = ordered.iloc[int(np.argmin(penalty))]
        status = "least_sensitive_candidate"
        stable_min = np.nan
        stable_max = np.nan
        reasons.append("No candidate met both stability and saturation tolerances.")

    adjacent = ordered.dropna(subset=["absolute_positive_fraction_delta_from_previous"])
    if adjacent.empty:
        largest_jump = 0.0
        jump_lower = np.nan
        jump_upper = np.nan
    else:
        jump = adjacent.loc[
            adjacent["absolute_positive_fraction_delta_from_previous"].idxmax()
        ]
        largest_jump = float(jump["absolute_positive_fraction_delta_from_previous"])
        jump_lower = float(jump["previous_vmax"])
        jump_upper = float(jump["vmax"])
    cliff_detected = largest_jump >= cliff_tolerance
    if cliff_detected:
        reasons.append(
            f"Largest adjacent positive-fraction shift is {largest_jump:.3f}, "
            f"at or above the {cliff_tolerance:.3f} review threshold."
        )

    suggestion_position = int(
        np.flatnonzero(
            np.isclose(ordered["vmax"].to_numpy(dtype=float), float(suggestion["vmax"]))
        )[0]
    )
    boundary_suggestion = suggestion_position in {0, len(ordered) - 1}
    if boundary_suggestion:
        reasons.append("Suggested value lies at the scanned range boundary.")

    fractions = ordered["primary_positive_fraction"].to_numpy(dtype=float)
    threshold_degenerate = bool(np.all(fractions <= 0.01) or np.all(fractions >= 0.99))
    if threshold_degenerate:
        reasons.append(
            "The primary threshold calls nearly every scanned cell the same class; "
            "this scan cannot calibrate biological positivity at that threshold."
        )

    manual_review = bool(
        status != "stable_plateau"
        or cliff_detected
        or boundary_suggestion
        or threshold_degenerate
    )
    if not reasons:
        reasons.append(
            "A locally stable, low-saturation range was found; image review remains required."
        )
    return {
        "marker": marker,
        "baseline_vmax": baseline,
        "lower_threshold": float(ordered.loc[0, "lower_threshold"]),
        "suggested_vmax": float(suggestion["vmax"]),
        "suggested_to_baseline_ratio": float(suggestion["vmax"]) / baseline,
        "stable_vmax_min": stable_min,
        "stable_vmax_max": stable_max,
        "recommendation_status": status,
        "positive_fraction_at_suggestion": float(
            suggestion["primary_positive_fraction"]
        ),
        "saturated_pixel_fraction_at_suggestion": float(
            suggestion["saturated_pixel_fraction"]
        ),
        "below_lower_threshold_fraction_at_suggestion": float(
            suggestion["below_lower_threshold_fraction"]
        ),
        "local_positive_fraction_sensitivity_at_suggestion": float(
            suggestion["local_positive_fraction_sensitivity"]
        ),
        "local_call_flip_fraction_at_suggestion": float(
            suggestion["local_call_flip_fraction"]
        ),
        "largest_adjacent_positive_fraction_jump": largest_jump,
        "largest_jump_lower_vmax": jump_lower,
        "largest_jump_upper_vmax": jump_upper,
        "cliff_detected": cliff_detected,
        "boundary_suggestion": boundary_suggestion,
        "threshold_degenerate": threshold_degenerate,
        "manual_review_required": manual_review,
        "review_reason": " ".join(reasons),
        "n_candidates": len(ordered),
        "n_cells": int(suggestion["n_cells"]),
        "n_rois": int(suggestion["n_rois"]),
    }


def analyze_normalization_scan(
    cell_scores: pd.DataFrame,
    pixel_summary: pd.DataFrame,
    *,
    baseline_values: Mapping[str, float],
    positive_thresholds: Sequence[float] = (0.25, 0.5, 0.75),
    primary_threshold: float = 0.5,
    stability_tolerance: float = 0.05,
    call_flip_tolerance: float = 0.05,
    saturation_tolerance: float = 0.05,
    cliff_tolerance: float = 0.15,
) -> NormalizationScanAnalysis:
    """Summarize scan outputs and identify provisional stable Vmax ranges.

    Stability is defined as a small change in primary positive-call proportion
    per log2 Vmax unit plus acceptably low input-pixel saturation.  It is a
    sensitivity diagnostic, not a biological ground truth estimator.
    """

    scores, pixels = _validate_scan_tables(cell_scores, pixel_summary)
    candidate_counts = (
        scores[["marker", "vmax"]]
        .drop_duplicates()
        .groupby("marker", observed=True)
        .size()
    )
    insufficient = sorted(
        str(marker) for marker, count in candidate_counts.items() if int(count) < 3
    )
    if insufficient:
        raise ValueError(
            "Nimbus normalization scan needs at least three distinct Vmax candidates "
            f"per marker; insufficient markers: {insufficient}."
        )
    thresholds = sorted({float(value) for value in positive_thresholds})
    if not thresholds or any(
        not math.isfinite(value) or value < 0 or value > 1 for value in thresholds
    ):
        raise ValueError("positive_thresholds must contain finite values in [0, 1].")
    primary = float(primary_threshold)
    if not math.isfinite(primary) or primary < 0 or primary > 1:
        raise ValueError("primary_threshold must lie in [0, 1].")
    if primary not in thresholds:
        thresholds.append(primary)
        thresholds.sort()
    for name, value in (
        ("stability_tolerance", stability_tolerance),
        ("call_flip_tolerance", call_flip_tolerance),
        ("saturation_tolerance", saturation_tolerance),
        ("cliff_tolerance", cliff_tolerance),
    ):
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError(f"{name} must be finite and lie in [0, 1].")

    markers = sorted(scores["marker"].astype(str).unique())
    missing_baselines = sorted(set(markers) - set(baseline_values))
    if missing_baselines:
        raise ValueError(
            f"Missing baseline Vmax values for markers: {missing_baselines}."
        )
    baselines = {marker: float(baseline_values[marker]) for marker in markers}
    validate_positive_values(list(baselines.values()), label="Baseline Vmax")

    candidate_summary, threshold_summary, roi_summary = _candidate_summaries(
        scores,
        pixels,
        baseline_values=baselines,
        positive_thresholds=thresholds,
        primary_threshold=primary,
    )
    candidate_summary = _add_adjacency_metrics(
        candidate_summary,
        scores,
        primary_threshold=primary,
    )
    recommendations = (
        pd.DataFrame(
            [
                _recommend_marker(
                    rows,
                    stability_tolerance=stability_tolerance,
                    call_flip_tolerance=call_flip_tolerance,
                    saturation_tolerance=saturation_tolerance,
                    cliff_tolerance=cliff_tolerance,
                )
                for _, rows in candidate_summary.groupby("marker", sort=True)
            ]
        )
        .sort_values("marker")
        .reset_index(drop=True)
    )
    return NormalizationScanAnalysis(
        candidate_summary=candidate_summary,
        threshold_summary=threshold_summary,
        roi_summary=roi_summary,
        recommendations=recommendations,
    )


def plot_marker_scan(
    *,
    marker: str,
    cell_scores: pd.DataFrame,
    analysis: NormalizationScanAnalysis,
    output_path: Path,
    primary_positive_score_threshold: float = 0.5,
) -> Path:
    """Plot Nimbus score distributions, positive fractions, and sensitivity."""

    import matplotlib.pyplot as plt

    candidates = analysis.candidate_summary[
        analysis.candidate_summary["marker"].astype(str) == str(marker)
    ].sort_values("vmax")
    thresholds = analysis.threshold_summary[
        analysis.threshold_summary["marker"].astype(str) == str(marker)
    ]
    scores = cell_scores[cell_scores["marker"].astype(str) == str(marker)]
    recommendation = analysis.recommendations[
        analysis.recommendations["marker"].astype(str) == str(marker)
    ].iloc[0]
    if candidates.empty or scores.empty:
        raise ValueError(f"No scan rows are available to plot marker {marker!r}.")
    if not math.isfinite(primary_positive_score_threshold) or not (
        0 <= primary_positive_score_threshold <= 1
    ):
        raise ValueError("primary_positive_score_threshold must lie in [0, 1].")

    vmax_values = candidates["vmax"].to_numpy(dtype=float)
    x_values = np.log2(candidates["vmax_factor"].to_numpy(dtype=float))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.fill_between(
        x_values,
        candidates["score_q05"].to_numpy(dtype=float),
        candidates["score_q95"].to_numpy(dtype=float),
        color="#9ecae1",
        alpha=0.35,
        label="5th–95th percentile",
    )
    ax.fill_between(
        x_values,
        candidates["score_q25"].to_numpy(dtype=float),
        candidates["score_q75"].to_numpy(dtype=float),
        color="#3182bd",
        alpha=0.35,
        label="25th–75th percentile",
    )
    ax.plot(
        x_values,
        candidates["score_median"],
        color="#08519c",
        marker="o",
        label="Median",
    )
    ax.set(
        title="Nimbus cell-score distribution",
        ylabel="Nimbus score",
        ylim=(-0.02, 1.02),
    )
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for threshold, rows in thresholds.groupby("positive_score_threshold", sort=True):
        rows = rows.sort_values("vmax")
        threshold_x = np.log2(
            rows["vmax"].to_numpy(dtype=float)
            / float(candidates["baseline_vmax"].iloc[0])
        )
        ax.plot(
            threshold_x,
            rows["positive_fraction"],
            marker="o",
            label=f"score ≥ {float(threshold):g}",
        )
    ax.set(
        title="Cells called positive",
        ylabel="Positive fraction",
        ylim=(-0.02, 1.02),
    )
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    score_bins = np.linspace(0, 1, 41)
    score_centres = (score_bins[:-1] + score_bins[1:]) / 2
    histogram_rows: list[np.ndarray] = []
    for vmax in vmax_values:
        values = scores.loc[
            np.isclose(scores["vmax"].to_numpy(dtype=float), vmax), "nimbus_score"
        ].to_numpy(dtype=float)
        histogram, _ = np.histogram(values, bins=score_bins)
        histogram_rows.append(histogram.astype(float) / max(len(values), 1))
    histogram_matrix = np.asarray(histogram_rows, dtype=float)
    global_height = float(histogram_matrix.max())
    if global_height <= 0:
        global_height = 1.0
    colour_map = plt.get_cmap("viridis")
    y_positions: np.ndarray = np.arange(len(vmax_values), dtype=float)
    suggested_vmax = float(recommendation["suggested_vmax"])
    baseline_vmax = float(recommendation["baseline_vmax"])
    tick_labels: list[str] = []
    for index, (vmax, histogram) in enumerate(zip(vmax_values, histogram_matrix)):
        height = histogram / global_height * 0.8
        colour = colour_map(index / max(len(vmax_values) - 1, 1))
        linewidth = 1.8 if np.isclose(vmax, suggested_vmax) else 0.8
        edge_colour = "#e6550d" if np.isclose(vmax, suggested_vmax) else colour
        ax.fill_between(
            score_centres,
            y_positions[index],
            y_positions[index] + height,
            step="mid",
            color=colour,
            alpha=0.65,
        )
        ax.step(
            score_centres,
            y_positions[index] + height,
            where="mid",
            color=edge_colour,
            linewidth=linewidth,
        )
        factor = vmax / baseline_vmax
        suffixes = []
        if np.isclose(vmax, baseline_vmax):
            suffixes.append("baseline")
        if np.isclose(vmax, suggested_vmax):
            suffixes.append("suggested")
        suffix = f"; {', '.join(suffixes)}" if suffixes else ""
        tick_labels.append(f"{vmax:.4g} ({factor:.3g}×{suffix})")
    ax.axvline(
        float(primary_positive_score_threshold),
        color="black",
        linestyle=":",
        linewidth=1.1,
        label="Primary positive threshold",
    )
    ax.set_yticks(y_positions, labels=tick_labels, fontsize=7)
    ax.set(
        title="Stacked Nimbus score histograms",
        xlabel="Nimbus cell score",
        ylabel="Vmax (factor)",
        xlim=(0, 1),
        ylim=(-0.1, len(vmax_values)),
    )
    ax.legend(fontsize=7, loc="upper center")
    ax.grid(axis="x", alpha=0.2)

    ax = axes[1, 1]
    ax.plot(
        x_values,
        candidates["saturated_pixel_fraction"],
        color="#d7301f",
        marker="o",
        label="Input pixels clipped at 1",
    )
    ax.plot(
        x_values,
        candidates["below_lower_threshold_fraction"],
        color="#636363",
        marker="o",
        label="Input pixels removed by lower threshold",
    )
    ax.plot(
        x_values,
        candidates["local_call_flip_fraction"],
        color="#31a354",
        marker="o",
        label="Local positive-call flips",
    )
    ax.plot(
        x_values,
        candidates["local_positive_fraction_sensitivity"],
        color="#756bb1",
        marker="o",
        label="Positive-fraction sensitivity / octave",
    )
    ax.set(
        title="Thresholding and adjacent-value sensitivity",
        ylabel="Fraction",
        ylim=(-0.02, 1.02),
    )
    ax.legend(fontsize=8)

    suggested_x = math.log2(
        float(recommendation["suggested_vmax"]) / float(recommendation["baseline_vmax"])
    )
    factor_axes = (axes[0, 0], axes[0, 1], axes[1, 1])
    for ax in factor_axes:
        ax.axvline(0, color="black", linestyle=":", linewidth=1, label="Baseline")
        ax.axvline(suggested_x, color="#e6550d", linestyle="--", linewidth=1.4)
        ax.set_xlabel("log2(Vmax / baseline)")
        ax.grid(alpha=0.2)
    if pd.notna(recommendation["stable_vmax_min"]):
        stable_min = math.log2(
            float(recommendation["stable_vmax_min"])
            / float(recommendation["baseline_vmax"])
        )
        stable_max = math.log2(
            float(recommendation["stable_vmax_max"])
            / float(recommendation["baseline_vmax"])
        )
        for ax in factor_axes:
            ax.axvspan(stable_min, stable_max, color="#74c476", alpha=0.1)
    fig.suptitle(
        f"{marker}: baseline {float(recommendation['baseline_vmax']):.4g}, "
        f"suggested {float(recommendation['suggested_vmax']):.4g} "
        f"lower {float(candidates['lower_threshold'].iloc[0]):.4g} "
        f"({recommendation['recommendation_status']})",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_suggested_normalization_dict(
    path: Path,
    recommendations: pd.DataFrame,
    lower_thresholds: Mapping[str, float] | None = None,
) -> Path:
    """Write a review-only preferred CSV or legacy JSON normalization file."""

    required = {"marker", "suggested_vmax"}
    missing = sorted(required - set(recommendations.columns))
    if missing:
        raise ValueError(f"Recommendation table is missing columns: {missing}.")
    vmax_values = {
        str(row.marker): float(row.suggested_vmax)
        for row in recommendations.sort_values("marker").itertuples(index=False)
    }
    if path.suffix.casefold() == ".csv":
        return write_normalization_csv(path, vmax_values, lower_thresholds)
    if path.suffix.casefold() != ".json":
        raise ValueError("Suggested normalization output must use .csv or .json.")
    payload = {marker: str(value) for marker, value in vmax_values.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


__all__ = [
    "DEFAULT_VMAX_FACTORS",
    "IntracellularExpressionSummary",
    "NormalizationScanAnalysis",
    "analyze_normalization_scan",
    "build_vmax_grid",
    "load_scan_parameter_csv",
    "plot_marker_scan",
    "rank_rois_by_expression",
    "resolve_marker_baseline_vmax",
    "resolve_marker_lower_thresholds",
    "resolve_scan_marker_inputs",
    "resolve_scan_markers",
    "safe_marker_filename",
    "select_scan_rois",
    "select_rois_across_expression_range",
    "summarize_intracellular_expression",
    "validate_positive_values",
    "write_suggested_normalization_dict",
]
