"""Deterministic selection of informative cells for image inspection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ._utils import (
    matrix_for_positions,
    metadata_for,
    resolve_roi_key,
    resolve_table,
    sample_positions,
    validate_markers,
    validate_population,
)
from .models import CellSelectionResult, MarkerExpectations


VALID_STRATEGIES = (
    "typical",
    "core",
    "boundary",
    "random",
    "marker_high",
    "marker_low",
    "contradictory",
)


def select_population_cells(
    data: Any,
    population_key: str,
    population: Any,
    *,
    strategies: Sequence[str] = ("typical", "core", "boundary"),
    n_per_strategy: int = 4,
    markers: Sequence[str] | None = None,
    marker: str | None = None,
    expectations: MarkerExpectations | Mapping[str, Any] | None = None,
    clustering_qc: Any | None = None,
    table_name: str | None = None,
    roi_key: str | None = None,
    diversity_keys: Sequence[str] | None = None,
    max_per_diversity_group: int | None = None,
    layer: str | None = None,
    max_candidates: int | None = 50_000,
    random_state: int = 0,
) -> CellSelectionResult:
    """Select cells that answer specific population-QC questions.

    ``typical`` finds the robust multivariate centre; ``core`` and ``boundary``
    use an :func:`assess_clustering` result; ``marker_high``/``marker_low`` rank
    one marker; and ``contradictory`` finds cells least aligned with expected
    positive and negative markers. A cell is included only once across
    strategies. ``diversity_keys`` and ``max_per_diversity_group`` can prevent
    a gallery from being dominated by one case or ROI.

    Agent guidance
    --------------
    Use small galleries with a deliberate mix. Typical/core cells show the
    dominant phenotype, boundary cells test separation, and contradictory cells
    test the biological label. Marker extremes are useful when a distribution is
    bimodal. Do not generalise from a gallery alone: it is targeted qualitative
    evidence that must be combined with full distributions and case/ROI support.
    """

    requested = list(dict.fromkeys(map(str, strategies)))
    invalid = [value for value in requested if value not in VALID_STRATEGIES]
    if invalid:
        raise ValueError(
            f"Unknown strategies {invalid}; choose from {VALID_STRATEGIES}"
        )
    if not requested or n_per_strategy < 1:
        raise ValueError("At least one strategy and n_per_strategy >= 1 are required")
    if max_per_diversity_group is not None and max_per_diversity_group < 1:
        raise ValueError("max_per_diversity_group must be at least 1 or None")
    _, adata = resolve_table(data, table_name)
    selected_diversity_keys = list(dict.fromkeys(map(str, diversity_keys or ())))
    missing_diversity_keys = [
        key for key in selected_diversity_keys if key not in adata.obs
    ]
    if missing_diversity_keys:
        raise KeyError(
            f"Diversity columns are missing from the table: {missing_diversity_keys}"
        )
    target_mask = validate_population(adata, population_key, population)
    target_positions = np.flatnonzero(target_mask.to_numpy())
    rng = np.random.default_rng(random_state)
    candidate_positions = sample_positions(target_positions, max_candidates, rng)
    expectation_model = MarkerExpectations.from_value(expectations)
    available = set(map(str, adata.var_names))
    expected_available = [
        value for value in expectation_model.markers if value in available
    ]
    selected_markers = validate_markers(
        adata, markers if markers is not None else (expected_available or None)
    )
    selected_roi = resolve_roi_key(data, adata, roi_key)
    warnings: list[str] = []
    if len(candidate_positions) < len(target_positions):
        warnings.append(
            f"Expression ranking uses {len(candidate_positions):,} sampled candidates from {len(target_positions):,} population cells"
        )

    values: np.ndarray | None = None
    marker_to_index: dict[str, int] = {}

    def expression_values(
        required: Sequence[str] = (),
    ) -> tuple[np.ndarray, dict[str, int]]:
        nonlocal values, marker_to_index, selected_markers
        combined = validate_markers(
            adata, list(dict.fromkeys([*selected_markers, *map(str, required)]))
        )
        if values is None or combined != selected_markers:
            selected_markers = combined
            values = matrix_for_positions(
                adata, candidate_positions, selected_markers, layer=layer
            )
            marker_to_index = {
                name: index for index, name in enumerate(selected_markers)
            }
        return values, marker_to_index

    rankings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if "typical" in requested:
        matrix, _ = expression_values()
        median = np.nanmedian(matrix, axis=0)
        q25, q75 = np.nanquantile(matrix, [0.25, 0.75], axis=0)
        scale = q75 - q25
        fallback = np.nanstd(matrix, axis=0)
        scale = np.where(scale > 0, scale, np.where(fallback > 0, fallback, 1.0))
        distance = np.nanmean(np.abs((matrix - median) / scale), axis=1)
        order = np.argsort(distance, kind="stable")
        rankings["typical"] = (candidate_positions[order], distance[order])

    structural = getattr(clustering_qc, "cell_metrics", pd.DataFrame())
    if any(value in requested for value in ("core", "boundary")):
        required = {"cell_index", "reference_population", "boundary_class"}
        if (
            structural is None
            or structural.empty
            or not required.issubset(structural.columns)
        ):
            warnings.append(
                "Core/boundary selection was skipped because compatible clustering_qc cell metrics were not supplied"
            )
        else:
            focused = structural.loc[
                structural["reference_population"].astype(str) == str(population)
            ].copy()
            lookup = pd.Series(
                np.arange(adata.n_obs, dtype=np.int64),
                index=adata.obs_names.astype(str),
            )
            focused["_position"] = focused["cell_index"].astype(str).map(lookup)
            focused = focused.dropna(subset=["_position"])
            purity_key = next(
                (
                    key
                    for key in ("graph_neighbour_purity", "umap_neighbour_purity")
                    if key in focused
                ),
                None,
            )
            focused["_purity"] = focused[purity_key] if purity_key else np.nan
            for strategy, ascending in (("core", False), ("boundary", True)):
                if strategy not in requested:
                    continue
                frame = focused.loc[
                    focused["boundary_class"].astype(str) == strategy
                ].sort_values("_purity", ascending=ascending, na_position="last")
                rankings[strategy] = (
                    frame["_position"].to_numpy(dtype=np.int64),
                    frame["_purity"].to_numpy(dtype=float),
                )

    selected_marker = marker
    if selected_marker is None:
        preferred = (
            expectation_model.positive_markers + expectation_model.supportive_markers
        )
        selected_marker = next(
            (value for value in preferred if value in available), None
        )
    if any(value in requested for value in ("marker_high", "marker_low")):
        if selected_marker is None:
            raise ValueError(
                "marker_high/marker_low requires marker or an expected positive marker"
            )
        matrix, indices = expression_values([selected_marker])
        marker_values = matrix[:, indices[selected_marker]]
        for strategy, descending in (("marker_high", True), ("marker_low", False)):
            if strategy not in requested:
                continue
            order = np.argsort(marker_values, kind="stable")
            if descending:
                order = order[::-1]
            rankings[strategy] = (candidate_positions[order], marker_values[order])

    if "contradictory" in requested:
        rule_markers = [
            value for value in expectation_model.markers if value in available
        ]
        if not rule_markers:
            warnings.append(
                "Contradictory selection was skipped because no expected markers are present"
            )
        else:
            matrix, indices = expression_values(rule_markers)
            agreement_terms: list[np.ndarray] = []
            for name in rule_markers:
                percentile = rankdata(matrix[:, indices[name]], method="average") / len(
                    matrix
                )
                agreement_terms.append(
                    1 - percentile
                    if name in expectation_model.negative_markers
                    else percentile
                )
            agreement = np.nanmean(np.column_stack(agreement_terms), axis=1)
            order = np.argsort(agreement, kind="stable")
            rankings["contradictory"] = (candidate_positions[order], agreement[order])

    if "random" in requested:
        shuffled = rng.permutation(target_positions)
        rankings["random"] = (shuffled, np.full(len(shuffled), np.nan))

    metadata = metadata_for(data)
    instance_key = metadata.get("source_instance_key")
    if instance_key not in adata.obs:
        instance_key = "ObjectNumber" if "ObjectNumber" in adata.obs else None
    used: set[int] = set()
    diversity_counts: dict[tuple[str, ...], int] = {}
    records: list[dict[str, Any]] = []
    for strategy in requested:
        if strategy not in rankings:
            continue
        positions, scores = rankings[strategy]
        selected_count = 0
        for position, score in zip(positions, scores, strict=False):
            position = int(position)
            if position in used:
                continue
            row = adata.obs.iloc[position]
            diversity_group = tuple(str(row[key]) for key in selected_diversity_keys)
            if (
                selected_diversity_keys
                and max_per_diversity_group is not None
                and diversity_counts.get(diversity_group, 0) >= max_per_diversity_group
            ):
                continue
            used.add(position)
            selected_count += 1
            if selected_diversity_keys:
                diversity_counts[diversity_group] = (
                    diversity_counts.get(diversity_group, 0) + 1
                )
            record: dict[str, Any] = {
                "obs_name": str(adata.obs_names[position]),
                "population": str(population),
                "strategy": strategy,
                "rank": selected_count,
                "score": float(score) if np.isfinite(score) else np.nan,
            }
            if selected_roi is not None:
                record[selected_roi] = str(row[selected_roi])
            for key in selected_diversity_keys:
                record[key] = str(row[key])
            if instance_key is not None:
                value = row[instance_key]
                record[str(instance_key)] = (
                    value.item() if hasattr(value, "item") else value
                )
            records.append(record)
            if selected_count >= n_per_strategy:
                break
        if selected_count < n_per_strategy:
            warnings.append(
                f"Strategy {strategy!r} selected {selected_count} of {n_per_strategy} requested cells"
            )
    if not records:
        raise ValueError("No cells could be selected with the requested strategies")
    return CellSelectionResult(
        population_key=population_key,
        population=str(population),
        cells=pd.DataFrame(records),
        markers=tuple(selected_markers),
        warnings=tuple(warnings),
        parameters={
            "strategies": requested,
            "n_per_strategy": n_per_strategy,
            "marker": selected_marker,
            "max_candidates": max_candidates,
            "random_state": random_state,
            "layer": layer,
            "diversity_keys": selected_diversity_keys,
            "max_per_diversity_group": max_per_diversity_group,
        },
    )


def select_population_cell_panel(
    data: Any,
    population_key: str,
    population: Any,
    *,
    marker: str | None = None,
    expectations: MarkerExpectations | Mapping[str, Any] | None = None,
    clustering_qc: Any | None = None,
    table_name: str | None = None,
    roi_key: str | None = None,
    diversity_keys: Sequence[str] | None = None,
    max_per_diversity_group: int | None = 1,
    layer: str | None = None,
    max_candidates: int | None = 50_000,
    random_state: int = 0,
) -> CellSelectionResult:
    """Select a standardized 20-cell population-review panel.

    Four unique cells are requested from each of five complementary strategies:
    typical, boundary, marker-high, contradictory, and random. Supply the
    structural QC result and an expected positive marker so every strategy can
    contribute. When diversity keys are supplied, the default permits one cell
    per unique diversity group. Any unavailable strategy or resulting shortfall
    is recorded in the warnings rather than silently replaced with easier cells.
    """

    return select_population_cells(
        data,
        population_key,
        population,
        strategies=(
            "typical",
            "boundary",
            "marker_high",
            "contradictory",
            "random",
        ),
        n_per_strategy=4,
        marker=marker,
        expectations=expectations,
        clustering_qc=clustering_qc,
        table_name=table_name,
        roi_key=roi_key,
        diversity_keys=diversity_keys,
        max_per_diversity_group=max_per_diversity_group,
        layer=layer,
        max_candidates=max_candidates,
        random_state=random_state,
    )


__all__ = [
    "VALID_STRATEGIES",
    "select_population_cell_panel",
    "select_population_cells",
]
