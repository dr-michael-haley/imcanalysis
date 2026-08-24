"""Native Scanpy population-QC figures with auditable plotted values.

The helpers in this module deliberately keep Scanpy imports local.  Population
assessment notebooks can therefore import :mod:`SpatialBiologyToolkit.population_qc`
without making the lightweight CLI depend on Scanpy.  Every function works on a
copy of the selected AnnData rows and returns the plotted observations or
sample-level values alongside the native Scanpy figure.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from ._utils import (
    matrix_for_positions,
    ordered_labels,
    resolve_table,
    sample_positions,
    validate_markers,
)
from .models import PlotResult


@contextmanager
def temporary_numba_cache_dir(project_dir: str | Path):
    """Use a project-local Numba cache only for the enclosed Scanpy work.

    Enter this context before the first import of :mod:`scanpy` in a kernel.
    Numba otherwise attempts to cache compiled helpers beside Scanpy itself,
    which is not reliably writable in managed Conda installations.  The prior
    ``NUMBA_CACHE_DIR`` value is restored and the temporary directory is
    removed when the context exits.
    """

    root = Path(project_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project directory does not exist: {root}")
    previous = os.environ.get("NUMBA_CACHE_DIR")
    with tempfile.TemporaryDirectory(prefix=".sbt-numba-cache-", dir=root) as path:
        os.environ["NUMBA_CACHE_DIR"] = path
        try:
            yield Path(path)
        finally:
            if previous is None:
                os.environ.pop("NUMBA_CACHE_DIR", None)
            else:
                os.environ["NUMBA_CACHE_DIR"] = previous


def _scanpy() -> Any:
    """Import Scanpy only when a native Scanpy figure is requested."""

    try:
        import scanpy as sc
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Scanpy is required for native population-QC plotting. "
            "Use an environment that provides scanpy."
        ) from error
    return sc


def _balanced_positions(
    labels: pd.Series,
    *,
    max_cells: int | None,
    max_cells_per_population: int | None,
    random_state: int,
) -> tuple[np.ndarray, list[str]]:
    """Return a deterministic, population-balanced selection of row positions."""

    if max_cells is not None and max_cells < 1:
        raise ValueError("max_cells must be at least 1 or None")
    if max_cells_per_population is not None and max_cells_per_population < 1:
        raise ValueError("max_cells_per_population must be at least 1 or None")
    observed = ordered_labels(labels)
    if not observed:
        raise ValueError("The population column contains no non-missing labels")
    per_population = max_cells_per_population
    if per_population is None and max_cells is not None:
        per_population = max(1, int(np.ceil(max_cells / len(observed))))
    values = labels.astype("string").to_numpy()
    rng = np.random.default_rng(random_state)
    selected = [
        sample_positions(np.flatnonzero(values == population), per_population, rng)
        for population in observed
    ]
    positions = (
        np.concatenate(selected) if selected else np.asarray([], dtype=np.int64)
    )
    if max_cells is not None and len(positions) > max_cells:
        positions = sample_positions(positions, max_cells, rng)
    return np.sort(positions), observed


def _native_matrixplot_figure(plot: Any) -> tuple[Any, Any]:
    """Materialise a Scanpy BasePlot and return its Matplotlib figure/axes."""

    if hasattr(plot, "make_figure"):
        plot.make_figure()
    figure = getattr(plot, "fig", None)
    if figure is None:  # pragma: no cover - Scanpy version fallback
        import matplotlib.pyplot as plt

        figure = plt.gcf()
    return figure, getattr(plot, "ax_dict", getattr(figure, "axes", None))


def plot_population_scanpy_umap(
    data: Any,
    population_key: str,
    *,
    population: Any | None = None,
    competitors: Sequence[Any] | None = None,
    table_name: str | None = None,
    embedding_key: str = "X_umap",
    max_cells: int | None = 200_000,
    max_cells_per_population: int | None = None,
    random_state: int = 0,
    point_size: float | None = None,
    title: str | None = None,
) -> PlotResult:
    """Render a native Scanpy UMAP using deterministic balanced cell sampling.

    ``population`` and ``competitors`` switch the colour key to a temporary,
    copied focus annotation (background, target, and named competitors).  The
    source AnnData and SpatialData remain unchanged.
    """

    _, adata = resolve_table(data, table_name)
    if population_key not in adata.obs:
        raise KeyError(f"Population column {population_key!r} is missing")
    if embedding_key not in adata.obsm:
        raise KeyError(f"Embedding {embedding_key!r} is missing")
    positions, observed = _balanced_positions(
        adata.obs[population_key],
        max_cells=max_cells,
        max_cells_per_population=max_cells_per_population,
        random_state=random_state,
    )
    working = adata[positions].copy()
    labels = working.obs[population_key].astype("string")
    working.obs[population_key] = pd.Categorical(
        labels, categories=observed, ordered=True
    )
    target = None if population is None else str(population)
    competitor_labels = list(dict.fromkeys(map(str, competitors or ())))
    available = set(observed)
    if target is not None and target not in available:
        raise KeyError(f"Population {target!r} is absent from {population_key!r}")
    missing = [value for value in competitor_labels if value not in available]
    if missing:
        raise KeyError(
            f"Competitor populations are absent from {population_key!r}: {missing}"
        )
    roles = np.full(working.n_obs, "population", dtype=object)
    color_key = population_key
    if target is not None:
        roles[:] = "background"
        rendered = labels.astype(str).to_numpy()
        for competitor in competitor_labels:
            roles[rendered == competitor] = f"competitor: {competitor}"
        roles[rendered == target] = "target"
        color_key = "_sbt_population_qc_focus"
        categories = ["background", *[f"competitor: {value}" for value in competitor_labels], "target"]
        working.obs[color_key] = pd.Categorical(roles, categories=categories, ordered=True)
    embedding = np.asarray(working.obsm[embedding_key])
    frame = pd.DataFrame(
        {
            "obs_name": working.obs_names.astype(str),
            "population": labels.astype(str).to_numpy(),
            "role": roles,
            "umap_1": embedding[:, 0],
            "umap_2": embedding[:, 1],
        }
    )
    sc = _scanpy()
    sc.pl.umap(
        working,
        color=color_key,
        size=point_size,
        title=title
        or (
            f"{population_key}: Scanpy UMAP"
            if target is None
            else f"{target}: Scanpy UMAP context"
        ),
        frameon=False,
        legend_loc="right margin",
        show=False,
    )
    import matplotlib.pyplot as plt

    figure = plt.gcf()
    return PlotResult(figure=figure, axes=figure.axes, data=frame)


def plot_population_scanpy_matrixplot(
    data: Any,
    population_key: str,
    *,
    populations: Sequence[Any] | None = None,
    markers: Sequence[str] | None = None,
    table_name: str | None = None,
    layer: str | None = None,
    max_cells_per_population: int | None = 10_000,
    random_state: int = 0,
    vmax: float = 0.6,
    cmap: str = "viridis",
    dendrogram: bool = True,
    title: str | None = None,
) -> PlotResult:
    """Render a native Scanpy marker matrix plot with an optional dendrogram.

    The colour scale uses the unmodified expression values from ``adata.X`` or
    ``layer``; ``vmax`` is only a display cap.  Returned ``data`` holds the raw
    sampled population means and cell counts.
    """

    if vmax <= 0:
        raise ValueError("vmax must be positive")
    _, adata = resolve_table(data, table_name)
    if population_key not in adata.obs:
        raise KeyError(f"Population column {population_key!r} is missing")
    marker_names = validate_markers(adata, markers)
    observed = ordered_labels(adata.obs[population_key])
    selected = (
        observed if populations is None else list(dict.fromkeys(map(str, populations)))
    )
    missing = [value for value in selected if value not in set(observed)]
    if missing:
        raise KeyError(f"Populations are absent from {population_key!r}: {missing}")
    labels = adata.obs[population_key].astype("string")
    rng = np.random.default_rng(random_state)
    selected_positions: list[np.ndarray] = []
    values: list[np.ndarray] = []
    counts: list[int] = []
    sampled_counts: list[int] = []
    label_values = labels.to_numpy()
    for label in selected:
        positions = np.flatnonzero(label_values == label)
        sampled = sample_positions(positions, max_cells_per_population, rng)
        selected_positions.append(sampled)
        values.append(matrix_for_positions(adata, sampled, marker_names, layer=layer))
        counts.append(int(len(positions)))
        sampled_counts.append(int(len(sampled)))
    positions = np.sort(np.concatenate(selected_positions))
    working = adata[positions, marker_names].copy()
    working.obs[population_key] = pd.Categorical(
        working.obs[population_key].astype(str), categories=selected, ordered=True
    )
    raw_means = pd.DataFrame(
        [np.nanmean(value, axis=0) for value in values],
        index=pd.Index(selected, name="population"),
        columns=marker_names,
    )
    raw_means.insert(0, "sampled_cells", sampled_counts)
    raw_means.insert(0, "cells", counts)
    sc = _scanpy()
    use_dendrogram = bool(dendrogram and len(selected) > 1)
    if use_dendrogram:
        sc.tl.dendrogram(working, groupby=population_key, var_names=marker_names)
    plot = sc.pl.matrixplot(
        working,
        var_names=marker_names,
        groupby=population_key,
        layer=layer,
        use_raw=False,
        dendrogram=use_dendrogram,
        vmax=vmax,
        cmap=cmap,
        title=title or f"{population_key}: Scanpy marker matrix",
        return_fig=True,
        show=False,
    )
    figure, axes = _native_matrixplot_figure(plot)
    return PlotResult(
        figure=figure,
        axes=axes,
        data=raw_means,
        display_data=raw_means.loc[:, marker_names],
    )


def plot_population_scanpy_abundance(
    data: Any,
    population_key: str,
    *,
    case_key: str,
    group_key: str,
    roi_key: str | None = None,
    table_name: str | None = None,
    vmax: float = 1.0,
    cmap: str = "viridis",
    dendrogram: bool = True,
    title: str | None = None,
) -> PlotResult:
    """Render a Scanpy group-by-population sample-abundance matrix plot.

    Each row of the internal AnnData is one case/ROI sample and contains
    posterior-population fractions.  It preserves case, group, and ROI values
    in the returned table, avoiding cell-count dominance by large samples.
    """

    if not 0 < vmax <= 1:
        raise ValueError("vmax must be in the interval (0, 1]")
    _, adata = resolve_table(data, table_name)
    required = [population_key, case_key, group_key]
    if roi_key is not None:
        required.append(roi_key)
    missing = [key for key in required if key not in adata.obs]
    if missing:
        raise KeyError(f"Observation columns are missing: {missing}")
    labels = ordered_labels(adata.obs[population_key])
    if not labels:
        raise ValueError("The population column contains no non-missing labels")
    sample_keys = [case_key, *(() if roi_key is None else (roi_key,))]
    metadata = adata.obs.loc[:, [*sample_keys, group_key]].copy()
    metadata = metadata.astype("string")
    inconsistent = (
        metadata.groupby(sample_keys, dropna=False, observed=True)[group_key]
        .nunique(dropna=False)
        .gt(1)
    )
    if bool(inconsistent.any()):
        samples = inconsistent.index[inconsistent].tolist()
        raise ValueError(
            f"{group_key!r} is not unique within case/ROI samples: {samples[:10]}"
        )
    counts = pd.crosstab(
        [metadata[key].astype(str) for key in sample_keys],
        adata.obs[population_key].astype(str),
    ).reindex(columns=labels, fill_value=0)
    fractions = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
    sample_metadata = metadata.drop_duplicates(subset=sample_keys).copy()
    sample_metadata.index = pd.MultiIndex.from_frame(sample_metadata.loc[:, sample_keys])
    sample_metadata = sample_metadata.reindex(fractions.index)
    sample_ids = pd.Index(
        ["::".join(map(str, value if isinstance(value, tuple) else (value,))) for value in fractions.index],
        name="sample_id",
    )
    if not sample_ids.is_unique:
        raise ValueError("Case/ROI sample identifiers are not unique after serialisation")
    sample_metadata.index = sample_ids
    fractions.index = sample_ids
    from anndata import AnnData

    abundance = AnnData(
        X=fractions.to_numpy(dtype=float),
        obs=sample_metadata.copy(),
        var=pd.DataFrame(index=pd.Index(labels, name="population")),
    )
    abundance.obs[group_key] = pd.Categorical(abundance.obs[group_key].astype(str))
    group_means = (
        fractions.assign(**{group_key: abundance.obs[group_key].astype(str).to_numpy()})
        .groupby(group_key, observed=True)[labels]
        .mean()
    )
    sc = _scanpy()
    use_dendrogram = bool(dendrogram and abundance.obs[group_key].nunique() > 1)
    if use_dendrogram:
        sc.tl.dendrogram(abundance, groupby=group_key, var_names=labels)
    plot = sc.pl.matrixplot(
        abundance,
        var_names=labels,
        groupby=group_key,
        dendrogram=use_dendrogram,
        vmax=vmax,
        cmap=cmap,
        title=title or f"Sample-level {population_key} abundance by {group_key}",
        return_fig=True,
        show=False,
    )
    figure, axes = _native_matrixplot_figure(plot)
    exported = abundance.obs.loc[:, [*sample_keys, group_key]].copy()
    for label in labels:
        exported[label] = fractions[label].to_numpy()
    exported.insert(0, "sample_id", exported.index)
    exported = exported.reset_index(drop=True)
    return PlotResult(figure=figure, axes=axes, data=exported, display_data=group_means)


__all__ = [
    "plot_population_scanpy_abundance",
    "plot_population_scanpy_matrixplot",
    "plot_population_scanpy_umap",
]
