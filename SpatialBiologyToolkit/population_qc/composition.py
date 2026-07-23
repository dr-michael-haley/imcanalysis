"""Case- and ROI-level representation evidence for cell populations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._utils import infer_case_key, ordered_labels, resolve_roi_key, resolve_table
from .models import PopulationRepresentationResult


def summarize_population_representation(
    data: Any,
    population_key: str,
    *,
    populations: Sequence[Any] | None = None,
    group_keys: Sequence[str] | None = None,
    table_name: str | None = None,
    roi_key: str | None = None,
    case_key: str | None = None,
) -> PopulationRepresentationResult:
    """Measure whether populations are distributed across cases and ROIs.

    The summary reports represented groups, largest-group share, entropy
    normalized by all available groups, and
    ``effective_groups = 1 / sum(p**2)``. The underlying table also contains
    each group's raw contribution and within-group population prevalence.

    Agent guidance
    --------------
    Use this before treating a small or unusual cluster as biological. A
    population restricted to one case may be genuine, but should be described as
    case-specific until replicated. Strong restriction to one ROI, batch, or
    damaged region should lower confidence and prompt image inspection. Examine
    both contribution and within-group prevalence because large ROIs can dominate
    raw counts without specific enrichment.
    """

    _, adata = resolve_table(data, table_name)
    if population_key not in adata.obs:
        raise KeyError(f"Population column {population_key!r} is missing from the table")
    warnings: list[str] = []
    if group_keys is None:
        selected_case = infer_case_key(adata, case_key)
        selected_roi = resolve_roi_key(data, adata, roi_key)
        selected_groups = list(
            dict.fromkeys(value for value in (selected_case, selected_roi) if value)
        )
        if selected_case is None:
            warnings.append(
                "No case/sample column was identified; representation is ROI-level only"
            )
    else:
        selected_groups = list(dict.fromkeys(map(str, group_keys)))
    if not selected_groups:
        raise ValueError("No case, ROI, or explicit group columns are available")
    missing_groups = [key for key in selected_groups if key not in adata.obs]
    if missing_groups:
        raise KeyError(f"Grouping columns are missing from the table: {missing_groups}")

    available = ordered_labels(adata.obs[population_key])
    selected_populations = available if populations is None else list(dict.fromkeys(map(str, populations)))
    missing_populations = [value for value in selected_populations if value not in set(available)]
    if missing_populations:
        raise KeyError(f"Populations are absent from {population_key!r}: {missing_populations}")

    summary_records: list[dict[str, Any]] = []
    count_records: list[dict[str, Any]] = []
    population_values = adata.obs[population_key].astype("string")
    for group_key in selected_groups:
        group_values = adata.obs[group_key].astype("string")
        valid = population_values.notna() & group_values.notna()
        contingency = pd.crosstab(
            population_values.loc[valid], group_values.loc[valid], dropna=False
        )
        group_totals = contingency.sum(axis=0)
        total_groups = int(contingency.shape[1])
        for population in selected_populations:
            if population not in contingency.index:
                continue
            counts = contingency.loc[population].astype(int)
            population_cells = int(counts.sum())
            nonzero = counts[counts > 0]
            proportions = nonzero.to_numpy(dtype=float) / max(1, population_cells)
            effective = float(1 / np.square(proportions).sum()) if len(proportions) else 0.0
            entropy = (
                float(-(proportions * np.log(proportions)).sum() / np.log(total_groups))
                if total_groups > 1
                else np.nan
            )
            summary_records.append(
                {
                    "population": population,
                    "group_key": group_key,
                    "cells": population_cells,
                    "available_groups": total_groups,
                    "represented_groups": int(len(nonzero)),
                    "fraction_groups_represented": len(nonzero) / total_groups if total_groups else np.nan,
                    "largest_group_fraction": float(proportions.max()) if len(proportions) else np.nan,
                    "effective_groups": effective,
                    "normalized_entropy": entropy,
                }
            )
            for group, count in nonzero.items():
                group_total = int(group_totals.loc[group])
                count_records.append(
                    {
                        "population": population,
                        "group_key": group_key,
                        "group": str(group),
                        "cell_count": int(count),
                        "population_fraction": int(count) / max(1, population_cells),
                        "group_total_cells": group_total,
                        "population_prevalence_in_group": int(count) / max(1, group_total),
                    }
                )
    return PopulationRepresentationResult(
        population_key=population_key,
        group_summary=pd.DataFrame(summary_records),
        group_counts=pd.DataFrame(count_records),
        warnings=tuple(warnings),
    )


__all__ = ["summarize_population_representation"]
