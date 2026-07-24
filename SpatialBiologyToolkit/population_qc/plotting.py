"""Focused, data-backed figures for population quality control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from .models import (
    CellSelectionResult,
    PlotResult,
    PopulationExpressionResult,
    PopulationRepresentationResult,
    ResolutionComparisonResult,
)


def plot_clustering_qc(
    qc_result: Any,
    *,
    populations: Sequence[Any] | None = None,
    metrics: Sequence[str] | None = None,
    annotate: bool = False,
    figsize: tuple[float, float] | None = None,
    title: str = "Structural clustering concern",
) -> PlotResult:
    """Plot structural concern scores and return the exact plotted table.

    Agent guidance
    --------------
    Use this as a triage map.  Look for agreement across graph, PCA, UMAP, and
    resolution evidence rather than reacting to one red cell.  High concern
    means investigate; it does not establish that a cluster is biologically
    invalid.
    """

    frame = qc_result.concern_scores.copy()
    if metrics is None:
        metrics = [
            definition.key
            for definition in qc_result.metric_definitions
            if definition.default_heatmap and definition.key in frame
        ]
    selected_metrics = list(dict.fromkeys(str(value) for value in metrics))
    missing = [value for value in selected_metrics if value not in frame]
    if missing:
        raise KeyError(f"Concern metrics are absent from the QC result: {missing}")
    frame.index = frame.index.astype(str)
    if populations is None:
        if "concern_rank" in qc_result.cluster_summary:
            order = (
                qc_result.cluster_summary["concern_rank"]
                .sort_values()
                .index.astype(str)
            )
        else:
            order = pd.Index(qc_result.cluster_order).astype(str)
        selected_populations = list(order)
    else:
        selected_populations = list(dict.fromkeys(str(value) for value in populations))
    missing = [value for value in selected_populations if value not in frame.index]
    if missing:
        raise KeyError(f"Populations are absent from the QC result: {missing}")
    frame = frame.loc[selected_populations, selected_metrics].dropna(axis=1, how="all")
    if frame.empty:
        raise ValueError("No available concern scores remain after selection")
    sizes = qc_result.cluster_summary.get("cluster_size", pd.Series(dtype=float))
    sizes.index = sizes.index.astype(str)
    display = frame.copy()
    display.index = [
        f"{population}  (n={int(sizes.get(population, 0)):,})"
        for population in frame.index
    ]

    import matplotlib.pyplot as plt
    import seaborn as sns

    if figsize is None:
        figsize = (
            max(8.0, min(22.0, 0.55 * frame.shape[1] + 5.0)),
            max(4.0, min(24.0, 0.38 * frame.shape[0] + 2.5)),
        )
    figure, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        display,
        cmap="rocket_r",
        vmin=0,
        vmax=1,
        annot=annotate,
        fmt=".2f",
        linewidths=0.25,
        cbar_kws={"label": "Concern score (higher = investigate)"},
        ax=ax,
    )
    ax.set_xlabel("Structural QC metric")
    ax.set_ylabel("Population")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=60)
    figure.tight_layout()
    return PlotResult(figure=figure, axes=ax, data=frame, display_data=display)


def plot_clustering_qc_panels(
    qc_result: Any,
    *,
    populations: Sequence[Any] | None = None,
    metric_groups: Mapping[str, Sequence[str]] | None = None,
    max_metrics_per_panel: int = 8,
    annotate: bool = False,
    title: str = "Structural clustering concern",
) -> dict[str, PlotResult]:
    """Split dense structural QC into small, legible heatmap panels."""

    if max_metrics_per_panel < 1:
        raise ValueError("max_metrics_per_panel must be at least 1")
    if metric_groups is None:
        metrics = [
            definition.key
            for definition in qc_result.metric_definitions
            if definition.default_heatmap and definition.key in qc_result.concern_scores
        ]
        groups = {
            f"panel_{index // max_metrics_per_panel + 1}": metrics[
                index : index + max_metrics_per_panel
            ]
            for index in range(0, len(metrics), max_metrics_per_panel)
        }
    else:
        groups = {
            str(name): list(dict.fromkeys(map(str, metrics)))
            for name, metrics in metric_groups.items()
        }
    if not groups:
        raise ValueError("No structural QC metrics are available for plotting")
    return {
        name: plot_clustering_qc(
            qc_result,
            populations=populations,
            metrics=metrics,
            annotate=annotate,
            title=f"{title} — {name.replace('_', ' ')}",
        )
        for name, metrics in groups.items()
    }


def plot_population_heatmap(
    data: Any,
    population_key: str,
    *,
    populations: Sequence[Any] | None = None,
    markers: Sequence[str] | None = None,
    table_name: str | None = None,
    layer: str | None = None,
    statistic: str = "median",
    standardization: str | None = "marker_zscore",
    standardization_clip: float | None = None,
    max_cells_per_population: int | None = 10_000,
    random_state: int = 0,
    annotate: bool = False,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> PlotResult:
    """Plot a population-marker heatmap with bounded deterministic sampling.

    ``standardization`` may be ``'marker_zscore'``,
    ``'marker_robust_zscore'``, or ``None`` for raw values.
    ``standardization_clip`` symmetrically clips standardized display values
    without changing the returned raw matrix. The result contains both raw and
    displayed matrices.

    Agent guidance
    --------------
    Start with the full landscape, then focus on a target, its strongest
    competitor, and relevant markers.  Relative z-scores show specificity, not
    absolute positivity.  Look for coherent programmes and inspect the raw
    values when colour scaling could exaggerate a small difference.
    """

    if statistic not in {"median", "mean"}:
        raise ValueError("statistic must be 'median' or 'mean'")
    if standardization not in {None, "marker_zscore", "marker_robust_zscore"}:
        raise ValueError("Invalid standardization")
    if standardization_clip is not None:
        if standardization is None:
            raise ValueError("standardization_clip requires standardization")
        if standardization_clip <= 0:
            raise ValueError("standardization_clip must be positive")
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
    if len(selected) > 60 or len(marker_names) > 80:
        raise ValueError("Select at most 60 populations and 80 markers per heatmap")
    labels = adata.obs[population_key].astype("string").to_numpy()
    summaries: list[np.ndarray] = []
    counts: list[int] = []
    sampled_counts: list[int] = []
    rng = np.random.default_rng(random_state)
    for population in selected:
        positions = np.flatnonzero(labels == population)
        sampled = sample_positions(positions, max_cells_per_population, rng)
        values = matrix_for_positions(adata, sampled, marker_names, layer=layer)
        summaries.append(
            np.nanmedian(values, axis=0)
            if statistic == "median"
            else np.nanmean(values, axis=0)
        )
        counts.append(int(len(positions)))
        sampled_counts.append(int(len(sampled)))
    raw = pd.DataFrame(
        summaries, index=pd.Index(selected, name="population"), columns=marker_names
    )
    if standardization == "marker_zscore":
        display = raw.sub(raw.mean()).div(raw.std(ddof=0).replace(0, 1.0))
    elif standardization == "marker_robust_zscore":
        center = raw.median()
        mad = raw.sub(center).abs().median().replace(0, 1.0)
        display = raw.sub(center).div(1.4826 * mad)
    else:
        display = raw.copy()
    if standardization_clip is not None:
        display = display.clip(
            lower=-float(standardization_clip),
            upper=float(standardization_clip),
        )

    import matplotlib.pyplot as plt
    import seaborn as sns

    if figsize is None:
        figsize = (
            max(7.0, min(22.0, 0.42 * len(marker_names) + 4.0)),
            max(4.0, min(24.0, 0.38 * len(selected) + 2.5)),
        )
    figure, ax = plt.subplots(figsize=figsize)
    plotted = display.copy()
    plotted.index = [
        f"{population}  (n={count:,})"
        + (f" [sample {sampled:,}]" if sampled < count else "")
        for population, count, sampled in zip(
            selected, counts, sampled_counts, strict=True
        )
    ]
    sns.heatmap(
        plotted,
        cmap="vlag" if standardization else "mako",
        center=0 if standardization else None,
        annot=annotate,
        fmt=".2g",
        linewidths=0.25,
        cbar_kws={
            "label": (
                f"{standardization} (clipped ±{standardization_clip:g})"
                if standardization_clip is not None
                else standardization or statistic
            )
        },
        ax=ax,
    )
    ax.set_xlabel("Marker")
    ax.set_ylabel(population_key)
    ax.set_title(title or f"{statistic.title()} marker expression by {population_key}")
    ax.tick_params(axis="x", labelrotation=60)
    figure.tight_layout()
    data = raw.copy()
    data.insert(0, "sampled_cells", sampled_counts)
    data.insert(0, "cells", counts)
    return PlotResult(figure=figure, axes=ax, data=data, display_data=display)


def plot_population_matrixplot(
    data: Any,
    population_key: str,
    *,
    populations: Sequence[Any] | None = None,
    markers: Sequence[str] | None = None,
    table_name: str | None = None,
    layer: str | None = None,
    statistic: str = "median",
    standardization: str | None = "marker_zscore",
    standardization_clip: float | None = 3.0,
    max_cells_per_population: int | None = 10_000,
    random_state: int = 0,
    annotate: bool = False,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> PlotResult:
    """Plot a legible population-by-marker matrix with bounded scaling.

    This notebook-oriented matrix plot uses the same exact summaries as
    :func:`plot_population_heatmap` but clips standardized display values by
    default so one marker with a near-zero MAD cannot flatten the rest of the
    landscape. Full raw and unclipped inputs remain available through the
    result's ``data`` table.
    """

    return plot_population_heatmap(
        data,
        population_key,
        populations=populations,
        markers=markers,
        table_name=table_name,
        layer=layer,
        statistic=statistic,
        standardization=standardization,
        standardization_clip=standardization_clip,
        max_cells_per_population=max_cells_per_population,
        random_state=random_state,
        annotate=annotate,
        figsize=figsize,
        title=title or f"{population_key}: population-by-marker matrix",
    )


def plot_population_umap(
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
    figsize: tuple[float, float] = (9.0, 7.0),
    title: str | None = None,
) -> PlotResult:
    """Plot global or focused UMAP context with deterministic stratification.

    The exact plotted observation names and coordinates are returned in
    ``PlotResult.data``. When ``population`` is supplied, all non-target and
    non-competitor populations are shown as a quiet background.
    """

    _, adata = resolve_table(data, table_name)
    if population_key not in adata.obs:
        raise KeyError(f"Population column {population_key!r} is missing")
    if embedding_key not in adata.obsm:
        raise KeyError(f"Embedding {embedding_key!r} is missing")
    embedding = np.asarray(adata.obsm[embedding_key])
    if embedding.ndim != 2 or embedding.shape[1] < 2:
        raise ValueError(f"Embedding {embedding_key!r} must have at least two columns")
    if max_cells is not None and max_cells < 1:
        raise ValueError("max_cells must be at least 1 or None")
    if max_cells_per_population is not None and max_cells_per_population < 1:
        raise ValueError("max_cells_per_population must be at least 1 or None")

    labels = adata.obs[population_key].astype("string")
    observed = ordered_labels(labels)
    target = None if population is None else str(population)
    if target is not None and target not in set(observed):
        raise KeyError(f"Population {target!r} is absent from {population_key!r}")
    competitor_labels = list(dict.fromkeys(map(str, competitors or ())))
    missing = [value for value in competitor_labels if value not in set(observed)]
    if missing:
        raise KeyError(
            f"Competitor populations are absent from {population_key!r}: {missing}"
        )

    rng = np.random.default_rng(random_state)
    if max_cells_per_population is None:
        per_population = (
            None
            if max_cells is None
            else max(1, int(np.ceil(max_cells / max(1, len(observed)))))
        )
    else:
        per_population = max_cells_per_population
    label_array = labels.to_numpy()
    selected_parts: list[np.ndarray] = []
    for label in observed:
        positions = np.flatnonzero(label_array == label)
        selected_parts.append(sample_positions(positions, per_population, rng))
    positions = (
        np.concatenate(selected_parts) if selected_parts else np.array([], dtype=int)
    )
    if max_cells is not None and len(positions) > max_cells:
        positions = sample_positions(positions, max_cells, rng)
    positions = np.sort(positions)

    plotted_labels = labels.iloc[positions].astype(str).to_numpy()
    roles = np.full(len(positions), "population", dtype=object)
    if target is not None:
        roles[:] = "background"
        roles[plotted_labels == target] = "target"
        roles[np.isin(plotted_labels, competitor_labels)] = "competitor"
    frame = pd.DataFrame(
        {
            "obs_name": adata.obs_names[positions].astype(str),
            "population": plotted_labels,
            "role": roles,
            "umap_1": embedding[positions, 0],
            "umap_2": embedding[positions, 1],
        }
    )

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, ax = plt.subplots(figsize=figsize)
    size = point_size or max(1.0, min(10.0, 100_000 / max(1, len(frame))))
    handles: list[Any] = []
    if target is None:
        cmap = plt.get_cmap("tab20")
        for index, label in enumerate(observed):
            mask = frame["population"].eq(label).to_numpy()
            if not mask.any():
                continue
            color = cmap(index % cmap.N)
            ax.scatter(
                frame.loc[mask, "umap_1"],
                frame.loc[mask, "umap_2"],
                s=size,
                alpha=0.65,
                linewidths=0,
                color=color,
                rasterized=True,
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=color,
                    label=label,
                    markersize=6,
                )
            )
    else:
        background = frame["role"].eq("background").to_numpy()
        ax.scatter(
            frame.loc[background, "umap_1"],
            frame.loc[background, "umap_2"],
            s=size,
            alpha=0.18,
            linewidths=0,
            color="#9A9A9A",
            rasterized=True,
        )
        focus = [*competitor_labels, target]
        colors = ["#0072B2", "#56B4E9", "#009E73", "#CC79A7", "#D55E00"]
        for index, label in enumerate(focus):
            mask = frame["population"].eq(label).to_numpy()
            color = "#D55E00" if label == target else colors[index % (len(colors) - 1)]
            ax.scatter(
                frame.loc[mask, "umap_1"],
                frame.loc[mask, "umap_2"],
                s=size * 1.5,
                alpha=0.8,
                linewidths=0,
                color=color,
                rasterized=True,
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=color,
                    label=label,
                    markersize=7,
                )
            )
    if handles:
        ax.legend(
            handles=handles,
            title=population_key,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            frameon=False,
            ncol=1 if len(handles) <= 20 else 2,
        )
    ax.set_xlabel(f"{embedding_key} 1")
    ax.set_ylabel(f"{embedding_key} 2")
    ax.set_title(
        title
        or (
            f"{population_key}: global UMAP"
            if target is None
            else f"{target}: UMAP context"
        )
    )
    ax.set_xticks([])
    ax.set_yticks([])
    figure.tight_layout()
    return PlotResult(figure=figure, axes=ax, data=frame)


def plot_marker_distributions(
    result: PopulationExpressionResult,
    *,
    markers: Sequence[str] | None = None,
    top_n: int = 8,
    ncols: int = 4,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> PlotResult:
    """Plot target/reference marker histograms from an expression result.

    Expected markers are prioritised when markers are omitted, followed by the
    largest absolute AUC effects.

    Agent guidance
    --------------
    Use when a heatmap or median suggests a difference.  Broad, bimodal, or
    overlapping distributions reveal mixtures that summary statistics hide.  A
    long positive tail is not uniform population-wide expression.
    """

    available = result.marker_statistics["marker"].astype(str).tolist()
    if markers is None:
        expected = [
            value for value in result.expectations.markers if value in set(available)
        ]
        ranked = result.strongest_markers(len(available))["marker"].astype(str).tolist()
        selected = list(dict.fromkeys([*expected, *ranked]))[:top_n]
    else:
        selected = list(dict.fromkeys(map(str, markers)))
    missing = [value for value in selected if value not in set(available)]
    if missing:
        raise KeyError(f"Markers are absent from the expression result: {missing}")
    if not selected or ncols < 1:
        raise ValueError("Select at least one marker and use ncols >= 1")

    import matplotlib.pyplot as plt

    ncols = min(ncols, len(selected))
    nrows = int(np.ceil(len(selected) / ncols))
    figure, grid = plt.subplots(
        nrows,
        ncols,
        figsize=figsize or (4.0 * ncols, 3.1 * nrows + 0.7),
        squeeze=False,
    )
    axes = grid.ravel()
    statistics = result.marker_statistics.set_index("marker")
    plotted_data = result.histogram_data.loc[
        result.histogram_data["marker"].isin(selected)
    ].copy()
    colors = {"target": "#D55E00", "reference": "#0072B2"}
    labels = {"target": result.target_population, "reference": result.reference}
    for ax, marker in zip(axes, selected, strict=False):
        marker_data = plotted_data.loc[plotted_data["marker"] == marker]
        for group in ("reference", "target"):
            frame = marker_data.loc[marker_data["group"] == group]
            ax.step(
                frame["bin_midpoint"],
                frame["fraction"],
                where="mid",
                linewidth=2,
                color=colors[group],
                label=labels[group],
            )
            ax.fill_between(
                frame["bin_midpoint"],
                frame["fraction"],
                step="mid",
                color=colors[group],
                alpha=0.12,
            )
        row = statistics.loc[marker]
        ax.axvline(row["target_median"], color=colors["target"], linestyle="--")
        ax.axvline(row["reference_median"], color=colors["reference"], linestyle="--")
        status = row["rule_status"] if pd.notna(row["rule_status"]) else "unconstrained"
        ax.set_title(
            f"{marker}\nAUC effect={row['auc_effect']:+.2f}; {status}", fontsize=10
        )
        ax.set_xlabel("Expression")
        ax.set_ylabel("Fraction of sampled cells")
    for ax in axes[len(selected) :]:
        ax.set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper right", frameon=False)
    figure.suptitle(title or f"{result.target_population} vs {result.reference}", y=1.0)
    figure.tight_layout()
    return PlotResult(figure=figure, axes=axes[: len(selected)], data=plotted_data)


def plot_population_representation(
    result: PopulationRepresentationResult,
    population: Any,
    *,
    group_key: str,
    top_n: int = 20,
    figsize: tuple[float, float] = (11.0, 5.0),
    title: str | None = None,
) -> PlotResult:
    """Plot population contribution and within-group prevalence together.

    Agent guidance
    --------------
    The left panel asks where population cells came from; the right asks how
    prevalent they are inside each case or ROI.  Inspect both before describing
    a population as sample-restricted.
    """

    frame = (
        result.group_counts.loc[
            (result.group_counts["population"].astype(str) == str(population))
            & (result.group_counts["group_key"].astype(str) == str(group_key))
        ]
        .sort_values("cell_count", ascending=False)
        .head(top_n)
        .copy()
    )
    if frame.empty:
        raise KeyError(f"No representation data for {population!r} and {group_key!r}")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    y = np.arange(len(frame))
    axes[0].barh(y, frame["population_fraction"], color="#4C78A8")
    axes[0].set_xlabel("Fraction of this population")
    axes[0].set_yticks(y, frame["group"].astype(str))
    axes[0].invert_yaxis()
    axes[1].barh(y, frame["population_prevalence_in_group"], color="#F58518")
    axes[1].set_xlabel("Population fraction within group")
    axes[1].tick_params(axis="y", labelleft=False)
    figure.suptitle(title or f"{population}: representation across {group_key}")
    figure.tight_layout()
    return PlotResult(figure=figure, axes=axes, data=frame)


def plot_population_breakdown(
    result: PopulationRepresentationResult,
    *,
    group_key: str,
    populations: Sequence[Any] | None = None,
    metric: str = "population_prevalence_in_group",
    max_groups: int | None = 40,
    annotate: bool = False,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> PlotResult:
    """Plot the population composition of cases, ROIs, or conditions.

    The complete long-form table is returned in ``data``. ``display_data``
    contains the bounded matrix used for the figure, allowing a notebook to
    show a legible subset while exporting every case or ROI to CSV.
    """

    allowed_metrics = {
        "cell_count",
        "population_fraction",
        "population_prevalence_in_group",
    }
    if metric not in allowed_metrics:
        raise ValueError(f"metric must be one of {sorted(allowed_metrics)}")
    if max_groups is not None and max_groups < 1:
        raise ValueError("max_groups must be at least 1 or None")
    frame = result.group_counts.loc[
        result.group_counts["group_key"].astype(str) == str(group_key)
    ].copy()
    if frame.empty:
        raise KeyError(
            f"No representation data are available for group_key {group_key!r}"
        )
    if populations is not None:
        selected = list(dict.fromkeys(map(str, populations)))
        observed = set(frame["population"].astype(str))
        missing = [value for value in selected if value not in observed]
        if missing:
            raise KeyError(
                f"Populations are absent from representation data: {missing}"
            )
        frame = frame.loc[frame["population"].astype(str).isin(selected)].copy()

    group_totals = (
        frame.groupby("group", observed=True)["group_total_cells"]
        .max()
        .sort_values(ascending=False)
    )
    shown_groups = group_totals.index.astype(str).tolist()
    if max_groups is not None:
        shown_groups = shown_groups[:max_groups]
    display_frame = frame.loc[frame["group"].astype(str).isin(shown_groups)].copy()
    matrix = display_frame.pivot_table(
        index="group",
        columns="population",
        values=metric,
        aggfunc="sum",
        fill_value=0.0,
    )
    matrix.index = matrix.index.astype(str)
    matrix = matrix.reindex(shown_groups)

    import matplotlib.pyplot as plt
    import seaborn as sns

    if figsize is None:
        figsize = (
            max(8.0, min(24.0, 0.42 * matrix.shape[1] + 5.0)),
            max(4.5, min(24.0, 0.28 * matrix.shape[0] + 2.5)),
        )
    figure, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        cmap="mako",
        annot=annotate,
        fmt=".2g",
        linewidths=0.2,
        cbar_kws={"label": metric},
        ax=ax,
    )
    ax.set_xlabel("Population")
    ax.set_ylabel(group_key)
    ax.set_title(title or f"{result.population_key}: breakdown across {group_key}")
    ax.tick_params(axis="x", labelrotation=60)
    figure.tight_layout()
    return PlotResult(figure=figure, axes=ax, data=frame, display_data=matrix)


def plot_resolution_membership(
    result: ResolutionComparisonResult,
    population: Any,
    *,
    max_target_clusters: int = 20,
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> PlotResult:
    """Plot how one reference population divides across each resolution.

    Agent guidance
    --------------
    One dominant column supports persistence.  Several persistent sizeable
    columns suggest a split candidate, but child clusters still need marker,
    case/ROI, and image validation.  Tiny fragments often reflect boundaries.
    """

    frame = result.membership.loc[
        result.membership["population"].astype(str) == str(population)
    ].copy()
    if frame.empty:
        raise KeyError(f"Population {population!r} is absent from resolution evidence")
    keep = (
        frame.groupby("target_cluster", observed=True)["reference_fraction"]
        .max()
        .nlargest(max_target_clusters)
        .index
    )
    frame = frame.loc[frame["target_cluster"].isin(keep)]
    matrix = frame.pivot_table(
        index="resolution",
        columns="target_cluster",
        values="reference_fraction",
        aggfunc="sum",
        fill_value=0,
    ).sort_index()
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure, ax = plt.subplots(
        figsize=figsize
        or (
            max(7.0, 0.65 * matrix.shape[1] + 3.0),
            max(3.2, 0.65 * matrix.shape[0] + 2.0),
        )
    )
    sns.heatmap(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=1,
        annot=annotate,
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"label": "Fraction of reference population"},
        ax=ax,
    )
    ax.set_xlabel("Cluster at sweep resolution")
    ax.set_ylabel("Resolution")
    ax.set_title(title or f"{population}: membership across resolutions")
    figure.tight_layout()
    return PlotResult(figure=figure, axes=ax, data=frame, display_data=matrix)


def plot_population_cell_gallery(
    sdata: Any,
    selection: CellSelectionResult,
    *,
    channel: str | Sequence[str] | None = None,
    color: str | None = None,
    table_name: str | None = None,
    crop_size: int | tuple[int, int] = 64,
    ncols: int = 4,
    outline_target_only: bool = True,
    mask_outside_target: bool = False,
    show_ax_titles: bool = True,
    separate_strategies: bool = True,
    compact_titles: bool = False,
) -> dict[str, PlotResult]:
    """Render selected cells as focused SpatialData galleries.

    Agent guidance
    --------------
    Choose channels that test the hypothesis, including a positive and a
    negative/competitor marker when possible.  Inspect unmasked context first;
    repeat with black masking to verify signal lies inside the target mask.
    """

    if not hasattr(sdata, "images") or not hasattr(sdata, "labels"):
        raise TypeError("A SpatialData object is required for image galleries")
    from SpatialBiologyToolkit.spatialdata import plot_spatialdata_cells

    groups = (
        list(selection.cells.groupby("strategy", sort=False))
        if separate_strategies
        else [("all", selection.cells)]
    )
    outputs: dict[str, PlotResult] = {}
    if channel is None:
        channel_note = ""
    elif isinstance(channel, str):
        channel_note = f" | channel: {channel}"
    else:
        channel_names = [str(value) for value in channel]
        channel_note = (
            ""
            if not channel_names
            else (
                f" | RGB: {' / '.join(channel_names)}"
                if len(channel_names) > 1
                else f" | channel: {channel_names[0]}"
            )
        )
    for strategy, frame in groups:
        figure, axes = plot_spatialdata_cells(
            sdata,
            frame["obs_name"].astype(str).tolist(),
            channel=channel,
            color=color,
            table_name=table_name,
            crop_size=crop_size,
            ncols=ncols,
            outline_target_only=outline_target_only,
            mask_outside_target=mask_outside_target,
            show_ax_titles=show_ax_titles,
            title=f"{selection.population}: {strategy} cells{channel_note}",
        )
        if show_ax_titles and compact_titles:
            flattened = np.asarray(axes, dtype=object).reshape(-1)
            for axis, (_, row) in zip(flattened, frame.iterrows(), strict=False):
                axis.set_title(
                    f"{row['strategy']} #{int(row['rank'])}\n{row['obs_name']}",
                    fontsize=8,
                )
        outputs[str(strategy)] = PlotResult(figure=figure, axes=axes, data=frame.copy())
    return outputs


# Descriptive alias retained for notebooks that prefer stability terminology.
plot_resolution_stability = plot_resolution_membership


__all__ = [
    "plot_clustering_qc",
    "plot_clustering_qc_panels",
    "plot_marker_distributions",
    "plot_population_breakdown",
    "plot_population_cell_gallery",
    "plot_population_heatmap",
    "plot_population_matrixplot",
    "plot_population_representation",
    "plot_population_umap",
    "plot_resolution_membership",
    "plot_resolution_stability",
]
