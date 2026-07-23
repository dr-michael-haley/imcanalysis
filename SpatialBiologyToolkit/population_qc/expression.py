"""Bounded, effect-size-first marker expression comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._utils import (
    matrix_for_positions,
    resolve_table,
    sample_positions,
    validate_markers,
    validate_population,
)
from .models import MarkerExpectations, PopulationExpressionResult


def _summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {f"{prefix}_{name}": np.nan for name in ("mean", "median", "q10", "q25", "q75", "q90")}
    q10, q25, median, q75, q90 = np.quantile(finite, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(median),
        f"{prefix}_q10": float(q10),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q75": float(q75),
        f"{prefix}_q90": float(q90),
    }


def _direction(marker: str, expectations: MarkerExpectations) -> str | None:
    if marker in expectations.positive_markers:
        return "positive"
    if marker in expectations.supportive_markers:
        return "supportive"
    if marker in expectations.negative_markers:
        return "negative"
    return None


def _rule_status(direction: str | None, auc_effect: float) -> str | None:
    if direction is None or not np.isfinite(auc_effect):
        return None
    aligned = (-1.0 if direction == "negative" else 1.0) * auc_effect
    return (
        "supports expectation"
        if aligned >= 0.20
        else "contradicts expectation"
        if aligned <= -0.20
        else "ambiguous"
    )


def compare_populations(
    data: Any,
    population_key: str,
    target_population: Any,
    *,
    reference_population: Any | None = None,
    markers: Sequence[str] | None = None,
    expectations: MarkerExpectations | Mapping[str, Any] | None = None,
    table_name: str | None = None,
    layer: str | None = None,
    max_cells_per_group: int | None = 50_000,
    histogram_bins: int = 36,
    random_state: int = 0,
) -> PopulationExpressionResult:
    """Compare marker expression in one population with another or the rest.

    Deterministic samples bound memory. The result reports quantiles, mean and
    median differences, pooled-within-group standardized mean difference, AUC,
    ``auc_effect`` (``2*AUC-1``), expected marker direction, and shared-bin
    histograms.

    Agent guidance
    --------------
    Compare the target first with its strongest structural competitor and then,
    if useful, with the rest. Prioritise effect size, distribution shape, and
    case-level reproducibility over cell-level p-values. Relative enrichment is
    not absolute positivity unless a justified threshold was supplied. Use the
    distribution plot when medians could conceal mixtures or long tails.
    """

    if histogram_bins < 5:
        raise ValueError("histogram_bins must be at least 5")
    _, adata = resolve_table(data, table_name)
    expectation_model = MarkerExpectations.from_value(expectations)
    selected_markers = validate_markers(adata, markers)
    target_mask = validate_population(adata, population_key, target_population)
    labels = adata.obs[population_key]
    if reference_population is None:
        reference_mask = labels.notna() & ~target_mask
        reference_label = "rest"
    else:
        if str(reference_population) == str(target_population):
            raise ValueError("target_population and reference_population must differ")
        reference_mask = validate_population(adata, population_key, reference_population)
        reference_label = str(reference_population)
    if not bool(reference_mask.any()):
        raise ValueError("The reference group contains no cells")

    target_all = np.flatnonzero(target_mask.to_numpy())
    reference_all = np.flatnonzero(reference_mask.to_numpy())
    rng = np.random.default_rng(random_state)
    target_positions = sample_positions(target_all, max_cells_per_group, rng)
    reference_positions = sample_positions(reference_all, max_cells_per_group, rng)
    target_matrix = matrix_for_positions(adata, target_positions, selected_markers, layer=layer)
    reference_matrix = matrix_for_positions(
        adata, reference_positions, selected_markers, layer=layer
    )
    warnings: list[str] = []
    if len(target_positions) < len(target_all):
        warnings.append(
            f"Target summaries use a deterministic sample of {len(target_positions):,} from {len(target_all):,} cells"
        )
    if len(reference_positions) < len(reference_all):
        warnings.append(
            f"Reference summaries use a deterministic sample of {len(reference_positions):,} from {len(reference_all):,} cells"
        )
    missing_expected = [
        marker for marker in expectation_model.markers if marker not in set(map(str, adata.var_names))
    ]
    if missing_expected:
        warnings.append("Expected markers absent from the panel: " + ", ".join(missing_expected))
    omitted_expected = [
        marker
        for marker in expectation_model.markers
        if marker in set(map(str, adata.var_names)) and marker not in selected_markers
    ]
    if omitted_expected:
        warnings.append(
            "Expected markers present in the panel but omitted from this comparison: "
            + ", ".join(omitted_expected)
        )

    from sklearn.metrics import roc_auc_score

    records: list[dict[str, Any]] = []
    histograms: list[dict[str, Any]] = []
    for index, marker in enumerate(selected_markers):
        target = target_matrix[:, index]
        reference = reference_matrix[:, index]
        target_finite = target[np.isfinite(target)]
        reference_finite = reference[np.isfinite(reference)]
        combined = np.r_[target_finite, reference_finite]
        if not len(target_finite) or not len(reference_finite):
            auc = np.nan
        elif np.nanmin(combined) == np.nanmax(combined):
            auc = 0.5
        else:
            binary = np.r_[np.ones(len(target_finite)), np.zeros(len(reference_finite))]
            auc = float(roc_auc_score(binary, combined))
        target_stats = _summary(target, "target")
        reference_stats = _summary(reference, "reference")
        mean_difference = target_stats["target_mean"] - reference_stats["reference_mean"]
        pooled_degrees = len(target_finite) + len(reference_finite) - 2
        if pooled_degrees > 0:
            target_variance = (
                float(np.var(target_finite, ddof=1)) if len(target_finite) > 1 else 0.0
            )
            reference_variance = (
                float(np.var(reference_finite, ddof=1))
                if len(reference_finite) > 1
                else 0.0
            )
            pooled_variance = (
                (len(target_finite) - 1) * target_variance
                + (len(reference_finite) - 1) * reference_variance
            ) / pooled_degrees
            pooled_sd = float(np.sqrt(max(0.0, pooled_variance)))
        else:
            pooled_sd = np.nan
        auc_effect = 2 * auc - 1 if np.isfinite(auc) else np.nan
        direction = _direction(marker, expectation_model)
        threshold = expectation_model.thresholds.get(marker)
        records.append(
            {
                "marker": marker,
                "expected_direction": direction,
                **target_stats,
                **reference_stats,
                "mean_difference": mean_difference,
                "median_difference": target_stats["target_median"] - reference_stats["reference_median"],
                "standardized_mean_difference": mean_difference / pooled_sd if np.isfinite(pooled_sd) and pooled_sd > 0 else np.nan,
                "auc": auc,
                "auc_effect": auc_effect,
                "rule_status": _rule_status(direction, auc_effect),
                "threshold": threshold,
                "target_fraction_above_threshold": float(np.mean(target_finite > threshold)) if threshold is not None and len(target_finite) else np.nan,
                "reference_fraction_above_threshold": float(np.mean(reference_finite > threshold)) if threshold is not None and len(reference_finite) else np.nan,
            }
        )
        if len(combined):
            low, high = float(np.nanmin(combined)), float(np.nanmax(combined))
            if high <= low:
                padding = max(0.5, abs(low) * 0.05)
                low, high = low - padding, high + padding
            edges = np.linspace(low, high, histogram_bins + 1)
            for group, values in (("target", target_finite), ("reference", reference_finite)):
                counts, _ = np.histogram(values, bins=edges)
                fractions = counts / max(1, int(counts.sum()))
                for bin_index, count in enumerate(counts):
                    histograms.append(
                        {
                            "marker": marker,
                            "group": group,
                            "bin_left": float(edges[bin_index]),
                            "bin_right": float(edges[bin_index + 1]),
                            "bin_midpoint": float((edges[bin_index] + edges[bin_index + 1]) / 2),
                            "count": int(count),
                            "fraction": float(fractions[bin_index]),
                        }
                    )
    statistics = pd.DataFrame(records)
    order = {"positive": 0, "supportive": 1, "negative": 2, None: 3}
    statistics["_expectation_order"] = statistics["expected_direction"].map(order)
    statistics = (
        statistics.sort_values(
            ["_expectation_order", "auc_effect"],
            ascending=[True, False],
            key=lambda values: values.abs() if values.name == "auc_effect" else values,
        )
        .drop(columns="_expectation_order")
        .reset_index(drop=True)
    )
    return PopulationExpressionResult(
        population_key=population_key,
        target_population=str(target_population),
        reference=reference_label,
        marker_statistics=statistics,
        histogram_data=pd.DataFrame(histograms),
        target_cells=int(len(target_all)),
        reference_cells=int(len(reference_all)),
        sampled_target_cells=int(len(target_positions)),
        sampled_reference_cells=int(len(reference_positions)),
        expectations=expectation_model,
        warnings=tuple(warnings),
        parameters={
            "layer": layer,
            "max_cells_per_group": max_cells_per_group,
            "histogram_bins": histogram_bins,
            "random_state": random_state,
        },
    )


def profile_population(
    data: Any,
    population_key: str,
    population: Any,
    **kwargs: Any,
) -> PopulationExpressionResult:
    """Profile one population against all other labelled cells.

    Agent guidance
    --------------
    Use this for an initial identity hypothesis. If structural QC identifies a
    strong competitor, follow with :func:`compare_populations` against that
    population explicitly; comparison with heterogeneous rest cells can make a
    weakly specific marker look more convincing.
    """

    return compare_populations(
        data,
        population_key,
        population,
        reference_population=None,
        **kwargs,
    )


__all__ = ["compare_populations", "profile_population"]
