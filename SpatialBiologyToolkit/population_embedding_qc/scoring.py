"""Concern normalization, raw thresholds, group scores, and ranking."""

from __future__ import annotations

import operator

import numpy as np
import pandas as pd

from .models import MetricDefinition


_OPERATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}

GROUP_OUTPUT_NAMES = {
    "graph_separation": "graph_separation_concern",
    "embedding_separation": "embedding_separation_concern",
    "embedding_reliability": "embedding_reliability_concern",
    "resolution_stability": "resolution_stability_concern",
}

OVERALL_GROUP_WEIGHTS = {
    "graph_separation": 0.35,
    "embedding_separation": 0.20,
    "embedding_reliability": 0.15,
    "resolution_stability": 0.30,
}


def normalize_concern(values: pd.Series, definition: MetricDefinition) -> pd.Series:
    """Apply clipped linear normalization without replacing missing values."""
    numeric = pd.to_numeric(values, errors="coerce")
    denominator = definition.bad_anchor - definition.good_anchor
    if denominator == 0:
        raise ValueError(f"Metric {definition.key} has identical good and bad anchors")
    score = (numeric - definition.good_anchor) / denominator
    return score.clip(lower=0, upper=1)


def threshold_flags(values: pd.Series, definition: MetricDefinition) -> pd.Series:
    """Calculate flags from raw values, independently of concern scores."""
    numeric = pd.to_numeric(values, errors="coerce")
    flags = pd.Series(pd.NA, index=values.index, dtype="boolean")
    valid = numeric.notna()
    flags.loc[valid] = _OPERATORS[definition.threshold_operator](
        numeric.loc[valid], definition.concern_threshold
    )
    return flags


def score_cluster_metrics(
    raw_metrics: pd.DataFrame,
    definitions: tuple[MetricDefinition, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return concern scores, raw threshold flags, and group summaries."""
    scores = pd.DataFrame(index=raw_metrics.index)
    flags = pd.DataFrame(index=raw_metrics.index)
    by_key = {definition.key: definition for definition in definitions}
    for key, definition in by_key.items():
        values = raw_metrics[key] if key in raw_metrics else pd.Series(np.nan, index=raw_metrics.index)
        scores[key] = normalize_concern(values, definition)
        flags[key] = threshold_flags(values, definition)

    summary = pd.DataFrame(index=raw_metrics.index)
    sufficient_groups: dict[str, pd.Series] = {}
    for group, output_name in GROUP_OUTPUT_NAMES.items():
        metrics = [definition for definition in definitions if definition.group == group]
        total_weight = float(sum(definition.weight for definition in metrics))
        weighted_sum = pd.Series(0.0, index=raw_metrics.index)
        available_weight = pd.Series(0.0, index=raw_metrics.index)
        available_count = pd.Series(0, index=raw_metrics.index, dtype=int)
        for definition in metrics:
            available = scores[definition.key].notna()
            weighted_sum = weighted_sum.add(scores[definition.key].fillna(0) * definition.weight)
            available_weight = available_weight.add(available.astype(float) * definition.weight)
            available_count = available_count.add(available.astype(int))
        summary[output_name] = weighted_sum.div(available_weight.replace(0, np.nan))
        summary[f"{output_name}_metric_count"] = available_count
        summary[f"{output_name}_available_weight"] = available_weight
        summary[f"{output_name}_insufficient_coverage"] = available_weight < (0.5 * total_weight)
        sufficient_groups[group] = (~summary[f"{output_name}_insufficient_coverage"]) & summary[output_name].notna()

    overall_numerator = pd.Series(0.0, index=raw_metrics.index)
    overall_weight = pd.Series(0.0, index=raw_metrics.index)
    for group, group_weight in OVERALL_GROUP_WEIGHTS.items():
        output_name = GROUP_OUTPUT_NAMES[group]
        usable = sufficient_groups[group]
        overall_numerator = overall_numerator.add(summary[output_name].fillna(0) * usable.astype(float) * group_weight)
        overall_weight = overall_weight.add(usable.astype(float) * group_weight)
    summary["overall_concern"] = overall_numerator.div(overall_weight.replace(0, np.nan))
    summary["overall_available_group_weight"] = overall_weight
    summary["overall_weights_redistributed"] = (overall_weight > 0) & (overall_weight < sum(OVERALL_GROUP_WEIGHTS.values()))
    summary["failed_thresholds"] = flags.fillna(False).sum(axis=1).astype(int)
    group_columns = list(GROUP_OUTPUT_NAMES.values())
    summary["maximum_group_concern"] = summary[group_columns].max(axis=1, skipna=True)
    summary["failed_metric_keys"] = [
        ", ".join(
            key
            for key in flags.columns
            if not pd.isna(flags.at[index, key]) and bool(flags.at[index, key])
        )
        for index in flags.index
    ]
    return scores, flags, summary


def ranked_cluster_order(summary: pd.DataFrame) -> list[str]:
    ranking = summary.copy()
    ranking["cluster_tiebreak"] = ranking.index.astype(str)
    ranking = ranking.sort_values(
        ["failed_thresholds", "maximum_group_concern", "overall_concern", "cluster_tiebreak"],
        ascending=[False, False, False, True],
        na_position="last",
        kind="mergesort",
    )
    return ranking.index.astype(str).tolist()


__all__ = [
    "GROUP_OUTPUT_NAMES",
    "normalize_concern",
    "ranked_cluster_order",
    "score_cluster_metrics",
    "threshold_flags",
]
