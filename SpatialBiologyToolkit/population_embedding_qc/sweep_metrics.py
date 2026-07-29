"""Contingency-table metrics for precomputed Leiden resolution sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass
class SweepMetricResult:
    reference_metrics: pd.DataFrame
    transition_edges: pd.DataFrame
    best_matches: pd.DataFrame
    reference_membership: pd.DataFrame
    global_metrics: pd.DataFrame
    jaccard_matrices: dict[str, pd.DataFrame]


def _entropy(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(counts.size))


def _contingency(left: pd.Series, right: pd.Series) -> pd.DataFrame:
    valid = left.notna() & right.notna()
    return pd.crosstab(left.loc[valid].astype(str), right.loc[valid].astype(str), dropna=False)


def _jaccard(contingency: pd.DataFrame) -> pd.DataFrame:
    shared = contingency.to_numpy(dtype=float)
    left = shared.sum(axis=1)[:, None]
    right = shared.sum(axis=0)[None, :]
    denominator = left + right - shared
    values = np.divide(shared, denominator, out=np.zeros_like(shared), where=denominator > 0)
    return pd.DataFrame(values, index=contingency.index, columns=contingency.columns)


def _transition_records(
    contingency: pd.DataFrame,
    *,
    source_column: str,
    source_resolution: float,
    target_column: str,
    target_resolution: float,
) -> list[dict[str, object]]:
    source_sizes = contingency.sum(axis=1)
    target_sizes = contingency.sum(axis=0)
    jaccard = _jaccard(contingency)
    records: list[dict[str, object]] = []
    for source in contingency.index:
        for target in contingency.columns:
            shared = int(contingency.loc[source, target])
            if shared == 0:
                continue
            records.append(
                {
                    "source_column": source_column,
                    "source_resolution": source_resolution,
                    "target_column": target_column,
                    "target_resolution": target_resolution,
                    "source_cluster": str(source),
                    "target_cluster": str(target),
                    "shared_cells": shared,
                    "source_fraction": shared / int(source_sizes.loc[source]),
                    "target_fraction": shared / int(target_sizes.loc[target]),
                    "jaccard": float(jaccard.loc[source, target]),
                    "source_size": int(source_sizes.loc[source]),
                    "target_size": int(target_sizes.loc[target]),
                }
            )
    return records


def calculate_sweep_metrics(
    obs: pd.DataFrame,
    *,
    reference_column: str,
    cluster_order: list[str],
    sweep: list[tuple[str, float]],
    persistence_threshold: float,
) -> SweepMetricResult:
    """Calculate cluster-level sweep stability without a cell-by-cell matrix."""
    if len(sweep) < 2:
        return SweepMetricResult(
            reference_metrics=pd.DataFrame(index=pd.Index(cluster_order, name="cluster")),
            transition_edges=pd.DataFrame(),
            best_matches=pd.DataFrame(),
            reference_membership=pd.DataFrame(),
            global_metrics=pd.DataFrame(),
            jaccard_matrices={},
        )
    transitions: list[dict[str, object]] = []
    best_records: list[dict[str, object]] = []
    membership_records: list[dict[str, object]] = []
    global_records: list[dict[str, object]] = []
    jaccard_matrices: dict[str, pd.DataFrame] = {}
    for (left_column, left_resolution), (right_column, right_resolution) in zip(sweep, sweep[1:]):
        contingency = _contingency(obs[left_column], obs[right_column])
        jaccard = _jaccard(contingency)
        key = f"{left_resolution:g}_to_{right_resolution:g}"
        jaccard_matrices[key] = jaccard
        transitions.extend(
            _transition_records(
                contingency,
                source_column=left_column,
                source_resolution=left_resolution,
                target_column=right_column,
                target_resolution=right_resolution,
            )
        )
        directional_best: list[float] = []
        for source in jaccard.index:
            target = jaccard.loc[source].idxmax()
            value = float(jaccard.loc[source, target])
            directional_best.append(value)
            best_records.append(
                {
                    "direction": "forward",
                    "source_column": left_column,
                    "source_resolution": left_resolution,
                    "source_cluster": str(source),
                    "target_column": right_column,
                    "target_resolution": right_resolution,
                    "target_cluster": str(target),
                    "jaccard": value,
                }
            )
        for target in jaccard.columns:
            source = jaccard[target].idxmax()
            best_records.append(
                {
                    "direction": "reverse",
                    "source_column": right_column,
                    "source_resolution": right_resolution,
                    "source_cluster": str(target),
                    "target_column": left_column,
                    "target_resolution": left_resolution,
                    "target_cluster": str(source),
                    "jaccard": float(jaccard.loc[source, target]),
                }
            )
        valid = obs[left_column].notna() & obs[right_column].notna()
        contingency_values = contingency.to_numpy(dtype=float)
        matched = 0.0
        if contingency_values.size:
            row_indices, column_indices = linear_sum_assignment(
                -contingency_values
            )
            matched = float(contingency_values[row_indices, column_indices].sum())
        valid_cells = int(valid.sum())
        global_records.append(
            {
                "source_column": left_column,
                "source_resolution": left_resolution,
                "target_column": right_column,
                "target_resolution": right_resolution,
                "source_clusters": int(contingency.shape[0]),
                "target_clusters": int(contingency.shape[1]),
                "adjusted_rand_index": float(adjusted_rand_score(obs.loc[valid, left_column].astype(str), obs.loc[valid, right_column].astype(str))),
                "normalized_mutual_information": float(normalized_mutual_info_score(obs.loc[valid, left_column].astype(str), obs.loc[valid, right_column].astype(str))),
                "mean_best_jaccard": float(np.mean(directional_best)) if directional_best else np.nan,
                "median_best_jaccard": float(np.median(directional_best)) if directional_best else np.nan,
                "fraction_changing_after_optimal_matching": (
                    1 - matched / valid_cells if valid_cells else np.nan
                ),
            }
        )

    reference = obs[reference_column]
    reference_valid = reference.notna()
    reference_values = reference.astype("string")
    per_cluster: dict[str, dict[str, list[Any]]] = {
        cluster: {
            "jaccard": [],
            "retention": [],
            "split_entropy": [],
            "merge_entropy": [],
            "within_consensus": [],
            "external_coassignment": [],
            "external_competitor": [],
        }
        for cluster in cluster_order
    }
    reference_in_sweep = reference_column in {column for column, _ in sweep}
    for sweep_column, resolution in sweep:
        contingency = _contingency(reference, obs[sweep_column])
        jaccard = _jaccard(contingency)
        reference_sizes = contingency.sum(axis=1)
        target_sizes = contingency.sum(axis=0)
        for cluster in contingency.index:
            for target_cluster, shared_value in contingency.loc[cluster].items():
                shared = int(shared_value)
                if not shared:
                    continue
                reference_size = int(reference_sizes.loc[cluster])
                target_size = int(target_sizes.loc[target_cluster])
                union = reference_size + target_size - shared
                membership_records.append(
                    {
                        "population": str(cluster),
                        "sweep_column": sweep_column,
                        "resolution": float(resolution),
                        "target_cluster": str(target_cluster),
                        "shared_cells": shared,
                        "reference_fraction": shared / max(1, reference_size),
                        "target_fraction": shared / max(1, target_size),
                        "jaccard": shared / max(1, union),
                    }
                )
        for cluster in cluster_order:
            if cluster not in contingency.index:
                continue
            counts = contingency.loc[cluster].to_numpy(dtype=float)
            reference_size = counts.sum()
            if reference_size <= 0:
                continue
            best_target = str(jaccard.loc[cluster].idxmax())
            best_jaccard = float(jaccard.loc[cluster, best_target])
            best_shared = float(contingency.loc[cluster, best_target])
            target_composition = contingency[best_target].to_numpy(dtype=float)
            within_pairs = sum(comb(int(value), 2) for value in counts if value >= 2)
            possible_pairs = comb(int(reference_size), 2) if reference_size >= 2 else 0
            per_cluster[cluster]["jaccard"].append(best_jaccard)
            per_cluster[cluster]["retention"].append(best_shared / reference_size)
            per_cluster[cluster]["split_entropy"].append(_entropy(counts))
            per_cluster[cluster]["merge_entropy"].append(_entropy(target_composition))
            per_cluster[cluster]["within_consensus"].append(within_pairs / possible_pairs if possible_pairs else np.nan)
            best_records.append(
                {
                    "direction": "reference_to_sweep",
                    "source_column": reference_column,
                    "source_resolution": np.nan,
                    "source_cluster": cluster,
                    "target_column": sweep_column,
                    "target_resolution": resolution,
                    "target_cluster": best_target,
                    "jaccard": best_jaccard,
                    "retention": best_shared / reference_size,
                    "split_entropy": _entropy(counts),
                    "merge_entropy": _entropy(target_composition),
                }
            )

        valid = reference_valid & obs[sweep_column].notna()
        labels = obs.loc[valid, sweep_column].astype(str)
        refs = reference_values.loc[valid].astype(str)
        reference_sizes = refs.value_counts()
        cross = pd.crosstab(refs, labels)
        for left_index, left in enumerate(cluster_order):
            if left not in cross.index or left not in reference_sizes:
                continue
            best_value = -1.0
            best_competitor: str | None = None
            for right in cluster_order[left_index + 1 :] + cluster_order[:left_index]:
                if right not in cross.index or right not in reference_sizes:
                    continue
                shared_probability = float(
                    (cross.loc[left].to_numpy(dtype=float) * cross.loc[right].to_numpy(dtype=float)).sum()
                    / (float(reference_sizes[left]) * float(reference_sizes[right]))
                )
                if shared_probability > best_value:
                    best_value, best_competitor = shared_probability, right
            if best_value >= 0:
                per_cluster[left]["external_coassignment"].append(best_value)
                per_cluster[left]["external_competitor"].append(best_competitor or "")

    records: list[dict[str, object]] = []
    for cluster in cluster_order:
        values = per_cluster[cluster]
        jaccards = np.asarray(values["jaccard"], dtype=float)
        retention = np.asarray(values["retention"], dtype=float)
        split = np.asarray(values["split_entropy"], dtype=float)
        merge = np.asarray(values["merge_entropy"], dtype=float)
        consensus = np.asarray(values["within_consensus"], dtype=float)
        external = np.asarray(values["external_coassignment"], dtype=float)
        competitors = list(values["external_competitor"])
        max_external_index = int(np.nanargmax(external)) if external.size and np.isfinite(external).any() else None
        records.append(
            {
                "cluster": cluster,
                "sweep_adjacent_jaccard": float(np.nanmean(jaccards)) if jaccards.size else np.nan,
                "sweep_minimum_jaccard": float(np.nanmin(jaccards)) if jaccards.size else np.nan,
                "reference_to_sweep_jaccard_mean": float(np.nanmean(jaccards)) if (jaccards.size and not reference_in_sweep) else np.nan,
                "reference_to_sweep_jaccard_minimum": float(np.nanmin(jaccards)) if (jaccards.size and not reference_in_sweep) else np.nan,
                "sweep_retention": float(np.nanmin(retention)) if retention.size else np.nan,
                "sweep_mean_retention": float(np.nanmean(retention)) if retention.size else np.nan,
                "sweep_persistence_fraction": float(np.mean(jaccards >= persistence_threshold)) if jaccards.size else np.nan,
                "sweep_supported_resolutions": int(np.sum(jaccards >= persistence_threshold)) if jaccards.size else 0,
                "sweep_split_entropy": float(np.nanmean(split)) if split.size else np.nan,
                "sweep_max_split_entropy": float(np.nanmax(split)) if split.size else np.nan,
                "sweep_merge_entropy": float(np.nanmean(merge)) if merge.size else np.nan,
                "sweep_max_merge_entropy": float(np.nanmax(merge)) if merge.size else np.nan,
                "sweep_within_cluster_consensus": float(np.nanmean(consensus)) if consensus.size else np.nan,
                "sweep_max_external_coassignment": float(np.nanmax(external)) if external.size else np.nan,
                "sweep_external_competitor": competitors[max_external_index] if max_external_index is not None else None,
            }
        )
    return SweepMetricResult(
        reference_metrics=pd.DataFrame(records).set_index("cluster"),
        transition_edges=pd.DataFrame(transitions),
        best_matches=pd.DataFrame(best_records),
        reference_membership=pd.DataFrame(membership_records),
        global_metrics=pd.DataFrame(global_records),
        jaccard_matrices=jaccard_matrices,
    )


__all__ = ["SweepMetricResult", "calculate_sweep_metrics"]
