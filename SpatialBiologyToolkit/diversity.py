"""Non-spatial population diversity for AnnData observations and pandas tables.

All functions are read-only. No expression matrix, graph, or image is accessed.
Simpson dominance is D = sum(p**2); Gini-Simpson diversity is 1-D.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def population_counts(
    data: Any,
    population: str,
    groupby: str | Sequence[str] | None = None,
    *,
    dropna: bool = True,
    missing_label: str = "Unlabelled",
) -> pd.DataFrame:
    """Count observation labels within groups, accepting AnnData or an obs DataFrame.

    Missing labels are excluded by default; groups containing only missing labels
    are retained with zero counts. Missing grouping values raise ValueError rather
    than silently dropping cells. Unobserved categorical levels are not expanded.
    Multiple grouping keys produce a MultiIndex. With groupby=None, one 'all' row
    is returned. No labels, categories or AnnData slots are modified.
    """
    obs = data if isinstance(data, pd.DataFrame) else getattr(data, "obs", None)
    if not isinstance(obs, pd.DataFrame):
        raise TypeError("Expected AnnData or a pandas observation DataFrame")
    keys = (
        []
        if groupby is None
        else ([groupby] if isinstance(groupby, str) else list(groupby))
    )
    if len(keys) != len(set(keys)) or population in keys:
        raise ValueError("Grouping keys must be unique and different from population")
    for key in keys + [population]:
        if key not in obs:
            raise KeyError(key)
    frame = obs[keys + [population]].copy()
    if keys and frame[keys].isna().any().any():
        raise ValueError("Missing grouping values; filter or label them explicitly")
    if not dropna:
        if frame[population].eq(missing_label).any() and frame[population].isna().any():
            raise ValueError("missing_label collides with an existing population")
        frame[population] = frame[population].astype(object).fillna(missing_label)
    if not keys:
        counts = frame[population].value_counts(dropna=True, sort=False)
        counts = counts[counts > 0]
        return (
            pd.DataFrame([counts.to_dict()], index=pd.Index(["all"], name="group"))
            .fillna(0)
            .astype("int64")
        )
    groups = frame.groupby(keys, observed=True, sort=False).size().index
    valid = frame.dropna(subset=[population])
    if valid.empty:
        return pd.DataFrame(index=groups, dtype="int64")
    result = (
        valid.groupby(keys + [population], observed=True, sort=False)
        .size()
        .unstack(population, fill_value=0)
    )
    return result.reindex(groups, fill_value=0).astype("int64")


def diversity_from_counts(
    counts: Any, *, estimator: str = "plugin", base: float = 2
) -> pd.DataFrame:
    """Calculate diversity for rows of a nonnegative count/abundance matrix.

    DataFrame row indices are preserved; a 1-D array is treated as one group.
    Outputs: n_total, richness, simpson_dominance, simpson_diversity,
    inverse_simpson, and shannon. Plugin Simpson uses sum((n/N)**2), matching
    ATHENA; unbiased Simpson uses sum(n*(n-1))/(N*(N-1)) and requires integer
    counts. Shannon always uses empirical proportions and the selected log base.
    Empty groups have richness=0 and undefined (NaN) diversity. Unbiased Simpson
    is undefined for N<2; a zero dominance gives infinite inverse Simpson.
    Proportions or weights are accepted only with estimator='plugin'; n_total
    then denotes their sum, not the number of cells. NaN, inf and negatives raise.
    """
    if estimator not in ("plugin", "unbiased"):
        raise ValueError("estimator must be 'plugin' or 'unbiased'")
    if not np.isfinite(base) or base <= 0 or base == 1:
        raise ValueError("Log base must be positive, finite and different from 1")
    if isinstance(counts, pd.Series):
        frame = counts.to_frame().T
    elif isinstance(counts, pd.DataFrame):
        frame = counts.copy()
    else:
        values = np.asarray(counts)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2:
            raise ValueError("Counts must be one- or two-dimensional")
        frame = pd.DataFrame(values)
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    if estimator == "unbiased" and not np.equal(values, np.floor(values)).all():
        raise ValueError("Unbiased Simpson requires integer counts")
    total = values.sum(axis=1)
    if not np.isfinite(total).all():
        raise ValueError("Count totals overflowed")
    props = np.divide(
        values, total[:, None], out=np.zeros_like(values), where=total[:, None] > 0
    )
    dominance = (props**2).sum(axis=1)
    if estimator == "unbiased":
        dominance = np.divide(
            (values * (values - 1)).sum(axis=1),
            total * (total - 1),
            out=np.full(len(total), np.nan),
            where=total > 1,
        )
    dominance[total == 0] = np.nan
    logs = np.zeros_like(props)
    np.log(props, out=logs, where=props > 0)
    shannon = -(props * logs).sum(axis=1) / np.log(base)
    shannon[total == 0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1 / dominance
    result = pd.DataFrame(
        dict(
            n_total=total,
            richness=(values > 0).sum(axis=1),
            simpson_dominance=dominance,
            simpson_diversity=1 - dominance,
            inverse_simpson=inverse,
            shannon=shannon,
        ),
        index=frame.index,
    )
    result.attrs.update(
        estimator=estimator,
        shannon_base=base,
        simpson_definition="D=sum(p_i^2); diversity=1-D"
        if estimator == "plugin"
        else "D=sum(n_i*(n_i-1))/(N*(N-1)); diversity=1-D",
    )
    return result


def population_diversity(
    data: Any,
    population: str,
    groupby: str | Sequence[str] | None = None,
    *,
    dropna: bool = True,
    missing_label: str = "Unlabelled",
    estimator: str = "plugin",
    base: float = 2,
) -> pd.DataFrame:
    """Read-only AnnData/DataFrame convenience wrapper around counts and diversity.

    Example: population_diversity(adata, 'cell_type', ['Case', 'ROI']).
    For equal-ROI case averages, explicitly average the returned ROI metrics;
    this differs from pooling all cells and calling with groupby='Case'.
    """
    counts = population_counts(
        data, population, groupby, dropna=dropna, missing_label=missing_label
    )
    return diversity_from_counts(counts, estimator=estimator, base=base)
