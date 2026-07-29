"""Focused comparison of precomputed clustering resolutions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd

from SpatialBiologyToolkit.population_embedding_qc.inspection import detect_sweep_columns
from SpatialBiologyToolkit.population_embedding_qc.storage import (
    StoredPopulationQCError,
    load_stored_population_qc,
)
from SpatialBiologyToolkit.population_embedding_qc.sweep_metrics import calculate_sweep_metrics

from ._utils import ordered_labels, resolve_table
from .models import ResolutionComparisonResult


def compare_resolutions(
    data: Any,
    reference_column: str,
    *,
    table_name: str | None = None,
    sweep_columns: Sequence[str] | None = None,
    sweep_regex: str = r"^leiden_(?P<resolution>\d+(?:\.\d+)?)$",
    persistence_threshold: float = 0.60,
    reuse: Literal["auto", "require", "never"] = "auto",
) -> ResolutionComparisonResult:
    """Compare population membership across a precomputed resolution sweep.

    Returns per-population stability, full transition evidence, global metrics,
    and a compact reference-to-resolution membership table. No clustering is
    recalculated.

    Agent guidance
    --------------
    Use ``reuse="require"`` for an existing global assessment so missing cached
    evidence raises rather than scanning the full annotation table again.
    Stored best matches are sufficient to apply a different persistence
    threshold without recomputing graph, embedding, or contingency evidence.
    Use after structural QC suggests a merge or split. A likely split repeatedly
    maps into multiple sizeable higher-resolution children; confirm that those
    children have distinct marker profiles and appear across cases. A likely merge
    repeatedly coassigns at lower resolutions and overlaps in graph and expression
    evidence. Resolution stability alone never establishes biological validity.
    """

    if not 0 <= persistence_threshold <= 1:
        raise ValueError("persistence_threshold must be between 0 and 1")
    if reuse not in {"auto", "require", "never"}:
        raise ValueError("reuse must be 'auto', 'require', or 'never'")
    _, adata = resolve_table(data, table_name)
    if reference_column not in adata.obs:
        raise KeyError(f"Reference column {reference_column!r} is missing from the table")
    sweep, warnings = detect_sweep_columns(
        adata.obs,
        sweep_regex=sweep_regex,
        explicit_columns=list(sweep_columns) if sweep_columns is not None else None,
    )
    if len(sweep) < 2:
        raise ValueError("At least two numerically identified clustering columns are required")
    if reuse != "never":
        try:
            stored = load_stored_population_qc(
                adata,
                population_key=reference_column,
                strict=True,
            )
            stored_columns = list(map(str, stored.run_summary.get("sweep_columns", [])))
            requested_columns = [column for column, _ in sweep]
            mismatches: list[str] = []
            if stored_columns != requested_columns:
                mismatches.append(
                    f"stored sweep columns are {stored_columns}, not {requested_columns}"
                )
            if stored.sweep_reference_membership.empty:
                mismatches.append("stored result has no reference membership table")
            if mismatches:
                raise StoredPopulationQCError("; ".join(mismatches))
            stability = stored.sweep_reference_cluster_metrics.copy()
            reference_matches = (
                stored.sweep_best_matches.loc[
                    stored.sweep_best_matches["direction"]
                    .astype(str)
                    .eq("reference_to_sweep")
                ]
                if "direction" in stored.sweep_best_matches
                else pd.DataFrame()
            )
            if not reference_matches.empty:
                grouped = reference_matches.groupby("source_cluster", observed=True)[
                    "jaccard"
                ]
                persistence = grouped.apply(
                    lambda values: float(
                        (pd.to_numeric(values, errors="coerce") >= persistence_threshold).mean()
                    )
                )
                supported = grouped.apply(
                    lambda values: int(
                        (pd.to_numeric(values, errors="coerce") >= persistence_threshold).sum()
                    )
                )
                stability["sweep_persistence_fraction"] = persistence.reindex(
                    stability.index.astype(str)
                ).to_numpy()
                stability["sweep_supported_resolutions"] = supported.reindex(
                    stability.index.astype(str), fill_value=0
                ).to_numpy()
            return ResolutionComparisonResult(
                reference_column=reference_column,
                sweep_columns=stored.detected_sweep_columns.copy(),
                cluster_stability=(
                    stability.reset_index()
                    .rename(columns={"cluster": "population"})
                ),
                membership=stored.sweep_reference_membership.copy(),
                transition_edges=stored.sweep_transition_edges.copy(),
                best_matches=stored.sweep_best_matches.copy(),
                global_metrics=stored.sweep_global_metrics.copy(),
                warnings=tuple([*warnings, *stored.warnings]),
            )
        except StoredPopulationQCError:
            if reuse == "require":
                raise
    cluster_order = ordered_labels(adata.obs[reference_column])
    metrics = calculate_sweep_metrics(
        adata.obs,
        reference_column=reference_column,
        cluster_order=cluster_order,
        sweep=sweep,
        persistence_threshold=persistence_threshold,
    )
    reference_values = adata.obs[reference_column]
    membership_records: list[dict[str, Any]] = []
    for column, resolution in sweep:
        sweep_values = adata.obs[column]
        valid = reference_values.notna() & sweep_values.notna()
        contingency = pd.crosstab(
            reference_values.loc[valid].astype(str),
            sweep_values.loc[valid].astype(str),
            dropna=False,
        )
        reference_sizes = contingency.sum(axis=1)
        target_sizes = contingency.sum(axis=0)
        for population in contingency.index:
            for target_cluster, shared_value in contingency.loc[population].items():
                shared = int(shared_value)
                if not shared:
                    continue
                reference_size = int(reference_sizes.loc[population])
                target_size = int(target_sizes.loc[target_cluster])
                union = reference_size + target_size - shared
                membership_records.append(
                    {
                        "population": str(population),
                        "sweep_column": column,
                        "resolution": float(resolution),
                        "target_cluster": str(target_cluster),
                        "shared_cells": shared,
                        "reference_fraction": shared / max(1, reference_size),
                        "target_fraction": shared / max(1, target_size),
                        "jaccard": shared / max(1, union),
                    }
                )
    stability = metrics.reference_metrics.reset_index().rename(columns={"cluster": "population"})
    return ResolutionComparisonResult(
        reference_column=reference_column,
        sweep_columns=pd.DataFrame(sweep, columns=["column", "resolution"]),
        cluster_stability=stability,
        membership=pd.DataFrame(membership_records),
        transition_edges=metrics.transition_edges,
        best_matches=metrics.best_matches,
        global_metrics=metrics.global_metrics,
        warnings=tuple(warnings),
    )


__all__ = ["compare_resolutions"]
