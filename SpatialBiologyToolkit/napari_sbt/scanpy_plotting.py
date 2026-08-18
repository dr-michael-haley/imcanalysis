"""Read-only AnnData preparation and figure builders for Scanpy-style QC plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from .explore import categorical_colour_map

PlotType = Literal[
    "embedding",
    "heatmap",
    "dotplot",
    "violin",
    "composition_bar",
    "composition_heatmap",
    "label_comparison",
]
MatrixSource = Literal["X", "raw"] | str
CellScope = Literal["all_cells", "cohort", "selected_groups"]
ExpressionScale = Literal["none", "zscore_marker", "minmax_marker"]
CompositionMeasure = Literal["count", "percent"]
ComparisonNormalisation = Literal["count", "row_percent", "column_percent"]


PLOT_TYPE_LABELS: dict[str, str] = {
    "embedding": "Embedding",
    "heatmap": "Marker heat map",
    "dotplot": "Marker dot plot",
    "violin": "Marker distributions",
    "composition_bar": "Population composition — stacked bars",
    "composition_heatmap": "Population composition — heat map",
    "label_comparison": "Compare two label columns",
}


class ScanpyPlotRequest(BaseModel):
    """Serializable description of one read-only plotting request."""

    plot_type: PlotType = "embedding"
    groupby: str
    cell_scope: CellScope = "all_cells"
    selected_groups: list[str] = Field(default_factory=list)
    roi_obs: str | None = None
    selected_rois: list[str] = Field(default_factory=list)
    matrix_source: MatrixSource = "X"
    markers: list[str] = Field(default_factory=list)
    expression_scale: ExpressionScale = "zscore_marker"
    positivity_threshold: float = 0.0
    embedding_key: str | None = None
    x_component: int = Field(default=1, ge=1)
    y_component: int = Field(default=2, ge=1)
    point_limit: int = Field(default=50_000, ge=100)
    point_size: float = Field(default=3.0, gt=0, le=100)
    point_alpha: float = Field(default=0.75, gt=0, le=1)
    label_centroids: bool = False
    composition_obs: str | None = None
    composition_measure: CompositionMeasure = "percent"
    comparison_obs: str | None = None
    comparison_normalisation: ComparisonNormalisation = "row_percent"

    @model_validator(mode="after")
    def validate_request(self) -> ScanpyPlotRequest:
        self.groupby = self.groupby.strip()
        self.selected_groups = list(
            dict.fromkeys(str(value) for value in self.selected_groups)
        )
        self.selected_rois = list(
            dict.fromkeys(str(value) for value in self.selected_rois)
        )
        self.markers = list(dict.fromkeys(str(value) for value in self.markers))
        if not self.groupby:
            raise ValueError("Choose a label column to group the plot.")
        if self.cell_scope == "selected_groups" and not self.selected_groups:
            raise ValueError(
                "Select at least one population for the chosen cell scope."
            )
        if self.x_component == self.y_component:
            raise ValueError("Embedding X and Y components must be different.")
        return self


@dataclass(slots=True)
class ScanpyPlotArtifact:
    """A generated figure plus the values and context needed to explain it."""

    figure: object
    title: str
    data: pd.DataFrame
    cell_count: int
    summary: str


def ordered_obs_values(
    series: pd.Series, *, include_missing: bool = False
) -> list[str]:
    """Return stable display values while respecting categorical order."""

    if isinstance(series.dtype, pd.CategoricalDtype):
        present = set(series.dropna().astype(str))
        values = [
            str(value) for value in series.cat.categories if str(value) in present
        ]
    else:
        values = list(dict.fromkeys(series.dropna().astype(str).tolist()))
    if include_missing and bool(series.isna().any()):
        values.append("Unassigned")
    return values


def groupable_obs_columns(adata, *, maximum_values: int = 200) -> list[str]:
    """Find observation columns suitable for population-style grouped plots."""

    columns: list[str] = []
    for column in adata.obs.columns:
        series = adata.obs[column]
        try:
            unique = int(series.nunique(dropna=True))
        except (TypeError, ValueError):
            continue
        if 1 <= unique <= int(maximum_values):
            columns.append(str(column))
    return columns


def matrix_source_choices(adata) -> list[str]:
    """Return expression matrices which share AnnData's observation identity."""

    choices = ["X"]
    if getattr(adata, "raw", None) is not None:
        choices.append("raw")
    choices.extend(f"layer::{key}" for key in adata.layers.keys())
    return choices


def matrix_source_var_names(adata, source: str) -> pd.Index:
    """Return marker names for one expression source."""

    source = str(source)
    if source == "raw":
        if getattr(adata, "raw", None) is None:
            raise ValueError("This AnnData object has no raw expression matrix.")
        return pd.Index(adata.raw.var_names.astype(str))
    if source == "X":
        return pd.Index(adata.var_names.astype(str))
    if source.startswith("layer::"):
        layer = source.split("::", 1)[1]
        if layer not in adata.layers:
            raise ValueError(f"AnnData layer does not exist: {layer!r}")
        return pd.Index(adata.var_names.astype(str))
    raise ValueError(f"Unknown expression source: {source!r}")


def _matrix_for_source(adata, source: str):
    if source == "X":
        return adata.X
    if source == "raw":
        if getattr(adata, "raw", None) is None:
            raise ValueError("This AnnData object has no raw expression matrix.")
        return adata.raw.X
    if source.startswith("layer::"):
        layer = source.split("::", 1)[1]
        if layer not in adata.layers:
            raise ValueError(f"AnnData layer does not exist: {layer!r}")
        return adata.layers[layer]
    raise ValueError(f"Unknown expression source: {source!r}")


def resolve_plot_cell_mask(
    adata,
    request: ScanpyPlotRequest,
    *,
    cohort_obs_names: set[str] | None = None,
) -> np.ndarray:
    """Resolve one plotting universe without mutating or reordering AnnData."""

    if request.groupby not in adata.obs:
        raise ValueError(f"Label column does not exist: {request.groupby!r}")
    mask = np.ones(int(adata.n_obs), dtype=bool)
    if request.cell_scope == "cohort":
        if cohort_obs_names is None:
            raise ValueError(
                "The current workspace does not provide a classification cohort."
            )
        mask &= adata.obs_names.astype(str).isin(cohort_obs_names)
    elif request.cell_scope == "selected_groups":
        mask &= (
            adata.obs[request.groupby]
            .astype("string")
            .isin(request.selected_groups)
            .fillna(False)
            .to_numpy(dtype=bool)
        )
    if request.selected_rois:
        if not request.roi_obs or request.roi_obs not in adata.obs:
            raise ValueError("Selected-ROI filtering requires a valid ROI observation.")
        mask &= (
            adata.obs[request.roi_obs]
            .astype("string")
            .isin(request.selected_rois)
            .fillna(False)
            .to_numpy(dtype=bool)
        )
    if not bool(mask.any()):
        raise ValueError("The selected plotting scope contains no cells.")
    return mask


def _expression_values(
    adata,
    *,
    source: str,
    markers: list[str],
    mask: np.ndarray,
) -> np.ndarray:
    if not markers:
        raise ValueError("Select at least one marker.")
    var_names = matrix_source_var_names(adata, source)
    positions = var_names.get_indexer(markers)
    missing = [marker for marker, position in zip(markers, positions) if position < 0]
    if missing:
        raise ValueError(
            "Markers are absent from the selected expression source: "
            + ", ".join(missing[:12])
        )
    rows = np.flatnonzero(mask)
    values = _matrix_for_source(adata, source)[rows, :][:, positions]
    if hasattr(values, "toarray"):
        values = values.toarray()
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape != (len(rows), len(markers)):
        raise ValueError("The selected expression matrix has an unexpected shape.")
    return values


def _display_labels(series: pd.Series) -> np.ndarray:
    return series.astype("string").fillna("Unassigned").to_numpy(dtype=str)


def _present_group_order(series: pd.Series, mask: np.ndarray) -> list[str]:
    selected = series.iloc[np.flatnonzero(mask)]
    order = ordered_obs_values(selected, include_missing=True)
    if not order:
        raise ValueError("The selected label column has no usable values.")
    if len(order) > 200:
        raise ValueError(
            "The selected label column has more than 200 visible groups. "
            "Filter the cells or choose a population-like observation."
        )
    return order


def deterministic_stratified_indices(
    labels: np.ndarray,
    *,
    maximum: int,
    seed: int = 0,
) -> np.ndarray:
    """Downsample display points while retaining every represented label."""

    labels = np.asarray(labels, dtype=str)
    count = len(labels)
    maximum = max(1, int(maximum))
    if count <= maximum:
        return np.arange(count, dtype=int)
    groups = list(dict.fromkeys(labels.tolist()))
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    remaining = maximum
    for index, group in enumerate(groups):
        positions = np.flatnonzero(labels == group)
        groups_left = len(groups) - index
        proportional = int(round(maximum * len(positions) / count))
        take = max(1, proportional)
        take = min(take, len(positions), remaining - max(0, groups_left - 1))
        if take > 0:
            selected.append(rng.choice(positions, size=take, replace=False))
            remaining -= take
    chosen = np.concatenate(selected) if selected else np.empty(0, dtype=int)
    if remaining > 0:
        available = np.setdiff1d(np.arange(count), chosen, assume_unique=False)
        if len(available):
            chosen = np.concatenate(
                [
                    chosen,
                    rng.choice(
                        available,
                        size=min(remaining, len(available)),
                        replace=False,
                    ),
                ]
            )
    return np.sort(chosen[:maximum])


def expression_group_summary(
    adata,
    request: ScanpyPlotRequest,
    mask: np.ndarray,
) -> pd.DataFrame:
    """Calculate group summaries from selected markers only."""

    if len(request.markers) > 100:
        raise ValueError("Select at most 100 markers for an expression summary.")
    values = _expression_values(
        adata,
        source=request.matrix_source,
        markers=request.markers,
        mask=mask,
    )
    labels = _display_labels(adata.obs[request.groupby].iloc[np.flatnonzero(mask)])
    groups = _present_group_order(adata.obs[request.groupby], mask)
    rows: list[dict[str, object]] = []
    for group in groups:
        group_values = values[labels == group]
        if len(group_values) == 0:
            continue
        means = np.nanmean(group_values, axis=0)
        medians = np.nanmedian(group_values, axis=0)
        fractions = np.nanmean(
            group_values > float(request.positivity_threshold), axis=0
        )
        for marker, mean, median, fraction in zip(
            request.markers,
            means,
            medians,
            fractions,
            strict=True,
        ):
            rows.append(
                {
                    "population": group,
                    "marker": marker,
                    "mean": float(mean),
                    "median": float(median),
                    "fraction_positive": float(fraction),
                    "cell_count": int(len(group_values)),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No expression summaries could be calculated.")
    result["display_value"] = result["mean"].astype(float)
    if request.expression_scale != "none":
        for _marker, positions in result.groupby("marker", sort=False).groups.items():
            position_list = list(positions)
            marker_values = result.loc[position_list, "mean"].to_numpy(dtype=float)
            if request.expression_scale == "zscore_marker":
                spread = float(np.nanstd(marker_values))
                scaled = (marker_values - float(np.nanmean(marker_values))) / (
                    spread if spread > 0 else 1.0
                )
            else:
                lower = float(np.nanmin(marker_values))
                spread = float(np.nanmax(marker_values) - lower)
                scaled = (marker_values - lower) / (spread if spread > 0 else 1.0)
            result.loc[position_list, "display_value"] = scaled
    return result


def _summary_matrix(
    summary: pd.DataFrame,
    *,
    value: str,
    groups: list[str],
    markers: list[str],
) -> np.ndarray:
    return (
        summary.pivot(index="population", columns="marker", values=value)
        .reindex(index=groups, columns=markers)
        .to_numpy(dtype=float)
    )


def _build_embedding_plot(
    adata,
    request: ScanpyPlotRequest,
    mask: np.ndarray,
) -> ScanpyPlotArtifact:
    if not request.embedding_key or request.embedding_key not in adata.obsm:
        raise ValueError("Choose an existing AnnData embedding.")
    coordinates = np.asarray(adata.obsm[request.embedding_key])
    if coordinates.ndim != 2:
        raise ValueError("The selected embedding is not a two-dimensional matrix.")
    x_position = request.x_component - 1
    y_position = request.y_component - 1
    if max(x_position, y_position) >= coordinates.shape[1]:
        raise ValueError(
            f"{request.embedding_key!r} has only {coordinates.shape[1]} components."
        )
    positions = np.flatnonzero(mask)
    labels = _display_labels(adata.obs[request.groupby].iloc[positions])
    keep = deterministic_stratified_indices(
        labels,
        maximum=request.point_limit,
        seed=0,
    )
    display_positions = positions[keep]
    display_labels = labels[keep]
    groups = _present_group_order(adata.obs[request.groupby], mask)
    colours = categorical_colour_map(adata, request.groupby)
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 7), constrained_layout=True)
    axis = figure.add_subplot(111)
    for group in groups:
        selected = display_labels == group
        if not bool(selected.any()):
            continue
        axis.scatter(
            coordinates[display_positions[selected], x_position],
            coordinates[display_positions[selected], y_position],
            s=float(request.point_size),
            c=colours.get(group, "#9ca3af"),
            alpha=float(request.point_alpha),
            linewidths=0,
            rasterized=True,
            label=group,
        )
        if request.label_centroids:
            x_values = coordinates[display_positions[selected], x_position]
            y_values = coordinates[display_positions[selected], y_position]
            axis.text(
                float(np.nanmedian(x_values)),
                float(np.nanmedian(y_values)),
                group,
                fontsize=8,
                weight="bold",
                ha="center",
                va="center",
            )
    axis.set_xlabel(f"{request.embedding_key} {request.x_component}")
    axis.set_ylabel(f"{request.embedding_key} {request.y_component}")
    title = f"{request.groupby} on {request.embedding_key}"
    axis.set_title(title)
    if len(groups) <= 30:
        axis.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            markerscale=3,
            frameon=False,
        )
    table = pd.DataFrame(
        {
            "obs_name": adata.obs_names[display_positions].astype(str),
            "population": display_labels,
            "x": coordinates[display_positions, x_position],
            "y": coordinates[display_positions, y_position],
        }
    )
    summary = (
        f"{len(display_positions):,} displayed of {len(positions):,} selected cells; "
        f"grouped by {request.groupby}; source {request.embedding_key}."
    )
    return ScanpyPlotArtifact(figure, title, table, len(positions), summary)


def _build_expression_plot(
    adata,
    request: ScanpyPlotRequest,
    mask: np.ndarray,
) -> ScanpyPlotArtifact:
    summary = expression_group_summary(adata, request, mask)
    groups = list(dict.fromkeys(summary["population"].astype(str)))
    markers = list(request.markers)
    displayed = _summary_matrix(
        summary,
        value="display_value",
        groups=groups,
        markers=markers,
    )
    scale_label = {
        "none": "mean expression",
        "zscore_marker": "marker-wise z-score",
        "minmax_marker": "marker-wise 0–1 range",
    }[request.expression_scale]
    from matplotlib.figure import Figure

    width = max(8, min(22, len(markers) * 0.38 + 4))
    height = max(5, min(22, len(groups) * 0.32 + 3))
    figure = Figure(figsize=(width, height), constrained_layout=True)
    axis = figure.add_subplot(111)
    if request.plot_type == "dotplot":
        fractions = _summary_matrix(
            summary,
            value="fraction_positive",
            groups=groups,
            markers=markers,
        )
        x, y = np.meshgrid(np.arange(len(markers)), np.arange(len(groups)))
        points = axis.scatter(
            x.ravel(),
            y.ravel(),
            s=20 + np.nan_to_num(fractions.ravel(), nan=0.0) * 300,
            c=displayed.ravel(),
            cmap="viridis",
            edgecolors="#374151",
            linewidths=0.25,
        )
        figure.colorbar(points, ax=axis, label=scale_label)
        title = f"{request.groupby}: marker dot plot"
    elif request.plot_type == "violin":
        if len(markers) > 12:
            raise ValueError("Select at most 12 markers for a distribution plot.")
        figure.clear()
        figure.set_size_inches(
            max(9, min(24, len(groups) * 0.45 + 5)),
            max(5, min(28, len(markers) * 2.0 + 2)),
            forward=True,
        )
        axes = figure.subplots(len(markers), 1, squeeze=False).ravel()
        values = _expression_values(
            adata,
            source=request.matrix_source,
            markers=markers,
            mask=mask,
        )
        labels = _display_labels(adata.obs[request.groupby].iloc[np.flatnonzero(mask)])
        rng = np.random.default_rng(0)
        for marker_index, (marker, marker_axis) in enumerate(zip(markers, axes)):
            distributions: list[np.ndarray] = []
            distribution_positions: list[int] = []
            constant_positions: list[int] = []
            constant_values: list[float] = []
            for group_index, group in enumerate(groups, start=1):
                population_values = values[labels == group, marker_index]
                population_values = population_values[np.isfinite(population_values)]
                if len(population_values) > 2_000:
                    population_values = rng.choice(
                        population_values, size=2_000, replace=False
                    )
                if len(population_values):
                    if (
                        len(population_values) >= 2
                        and float(
                            np.nanmax(population_values) - np.nanmin(population_values)
                        )
                        > 0
                    ):
                        distributions.append(population_values)
                        distribution_positions.append(group_index)
                    else:
                        constant_positions.append(group_index)
                        constant_values.append(float(population_values[0]))
            if distributions:
                marker_axis.violinplot(
                    distributions,
                    positions=distribution_positions,
                    showmedians=True,
                    showextrema=False,
                )
            if constant_positions:
                marker_axis.scatter(
                    constant_positions,
                    constant_values,
                    marker="_",
                    s=80,
                    color="#1f2937",
                    linewidths=1.5,
                )
            marker_axis.set_ylabel(marker)
            marker_axis.set_xticks(np.arange(1, len(groups) + 1))
            marker_axis.set_xticklabels(
                groups if marker_index == len(markers) - 1 else [],
                rotation=45,
                ha="right",
            )
        title = f"{request.groupby}: marker distributions"
        figure.suptitle(title)
        return ScanpyPlotArtifact(
            figure,
            title,
            summary,
            int(mask.sum()),
            f"{int(mask.sum()):,} selected cells; {len(groups)} populations; "
            f"{len(markers)} markers from {request.matrix_source}.",
        )
    else:
        image = axis.imshow(displayed, aspect="auto", cmap="coolwarm")
        figure.colorbar(image, ax=axis, label=scale_label)
        title = f"{request.groupby}: population mean marker expression"
    axis.set_xticks(np.arange(len(markers)), labels=markers, rotation=90)
    axis.set_yticks(np.arange(len(groups)), labels=groups)
    axis.set_ylim(len(groups) - 0.5, -0.5)
    axis.set_title(title)
    return ScanpyPlotArtifact(
        figure,
        title,
        summary,
        int(mask.sum()),
        f"{int(mask.sum()):,} selected cells; {len(groups)} populations; "
        f"{len(markers)} markers from {request.matrix_source}.",
    )


def _build_composition_plot(
    adata,
    request: ScanpyPlotRequest,
    mask: np.ndarray,
) -> ScanpyPlotArtifact:
    if not request.composition_obs or request.composition_obs not in adata.obs:
        raise ValueError("Choose an observation to define samples or ROIs.")
    positions = np.flatnonzero(mask)
    samples = _display_labels(adata.obs[request.composition_obs].iloc[positions])
    labels = _display_labels(adata.obs[request.groupby].iloc[positions])
    counts = pd.crosstab(
        pd.Series(samples, name=request.composition_obs),
        pd.Series(labels, name=request.groupby),
        dropna=False,
    )
    population_order = _present_group_order(adata.obs[request.groupby], mask)
    counts = counts.reindex(columns=population_order, fill_value=0)
    if request.composition_measure == "percent":
        plotted = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
        value_label = "Cells within sample (%)"
    else:
        plotted = counts.astype(float)
        value_label = "Cell count"
    colours = categorical_colour_map(adata, request.groupby)
    from matplotlib.figure import Figure

    width = max(9, min(24, len(plotted.index) * 0.35 + 5))
    height = max(6, min(20, len(population_order) * 0.28 + 4))
    figure = Figure(figsize=(width, height), constrained_layout=True)
    axis = figure.add_subplot(111)
    if request.plot_type == "composition_bar":
        bottom = np.zeros(len(plotted), dtype=float)
        for population in population_order:
            values = plotted[population].to_numpy(dtype=float)
            axis.bar(
                np.arange(len(plotted)),
                values,
                bottom=bottom,
                color=colours.get(population, "#9ca3af"),
                label=population,
                width=0.9,
            )
            bottom += np.nan_to_num(values, nan=0.0)
        axis.set_xticks(
            np.arange(len(plotted)), labels=plotted.index.astype(str), rotation=90
        )
        axis.set_ylabel(value_label)
        if len(population_order) <= 30:
            axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        title = f"{request.groupby} composition by {request.composition_obs}"
    else:
        image = axis.imshow(
            plotted.T.to_numpy(dtype=float), aspect="auto", cmap="viridis"
        )
        axis.set_xticks(
            np.arange(len(plotted.index)), labels=plotted.index.astype(str), rotation=90
        )
        axis.set_yticks(np.arange(len(population_order)), labels=population_order)
        figure.colorbar(image, ax=axis, label=value_label)
        title = f"{request.groupby} abundance across {request.composition_obs}"
    axis.set_title(title)
    exported = (
        plotted.rename_axis(index=request.composition_obs, columns=request.groupby)
        .stack(dropna=False)
        .rename("value")
        .reset_index()
    )
    exported["measure"] = request.composition_measure
    return ScanpyPlotArtifact(
        figure,
        title,
        exported,
        len(positions),
        f"{len(positions):,} selected cells across {len(plotted):,} "
        f"{request.composition_obs} values; {request.composition_measure}.",
    )


def _build_label_comparison_plot(
    adata,
    request: ScanpyPlotRequest,
    mask: np.ndarray,
) -> ScanpyPlotArtifact:
    if not request.comparison_obs or request.comparison_obs not in adata.obs:
        raise ValueError("Choose a second label column to compare.")
    if request.comparison_obs == request.groupby:
        raise ValueError("Choose two different label columns for comparison.")
    positions = np.flatnonzero(mask)
    original = _display_labels(adata.obs[request.comparison_obs].iloc[positions])
    current = _display_labels(adata.obs[request.groupby].iloc[positions])
    counts = pd.crosstab(
        pd.Series(original, name=request.comparison_obs),
        pd.Series(current, name=request.groupby),
        dropna=False,
    )
    if request.comparison_normalisation == "row_percent":
        plotted = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
        value_label = "Within original label (%)"
    elif request.comparison_normalisation == "column_percent":
        plotted = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1) * 100
        value_label = "Within new label (%)"
    else:
        plotted = counts.astype(float)
        value_label = "Cell count"
    from matplotlib.figure import Figure

    width = max(8, min(24, plotted.shape[1] * 0.42 + 5))
    height = max(6, min(24, plotted.shape[0] * 0.36 + 4))
    figure = Figure(figsize=(width, height), constrained_layout=True)
    axis = figure.add_subplot(111)
    image = axis.imshow(plotted.to_numpy(dtype=float), aspect="auto", cmap="magma")
    axis.set_xticks(
        np.arange(plotted.shape[1]), labels=plotted.columns.astype(str), rotation=90
    )
    axis.set_yticks(np.arange(plotted.shape[0]), labels=plotted.index.astype(str))
    axis.set_xlabel(request.groupby)
    axis.set_ylabel(request.comparison_obs)
    title = f"{request.comparison_obs} → {request.groupby} label mapping"
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=value_label)
    if plotted.size <= 225:
        values = plotted.to_numpy(dtype=float)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                label = f"{value:.0f}" if np.isfinite(value) else ""
                axis.text(column, row, label, ha="center", va="center", fontsize=7)
    exported = (
        plotted.rename_axis(index=request.comparison_obs, columns=request.groupby)
        .stack(dropna=False)
        .rename("value")
        .reset_index()
    )
    exported["normalisation"] = request.comparison_normalisation
    return ScanpyPlotArtifact(
        figure,
        title,
        exported,
        len(positions),
        f"{len(positions):,} selected cells; {plotted.shape[0]} original labels; "
        f"{plotted.shape[1]} current labels.",
    )


def build_scanpy_plot(
    adata,
    request: ScanpyPlotRequest,
    *,
    cohort_obs_names: set[str] | None = None,
) -> ScanpyPlotArtifact:
    """Build one read-only plot artifact from the current AnnData state."""

    mask = resolve_plot_cell_mask(
        adata,
        request,
        cohort_obs_names=cohort_obs_names,
    )
    if request.plot_type == "embedding":
        return _build_embedding_plot(adata, request, mask)
    if request.plot_type in {"heatmap", "dotplot", "violin"}:
        return _build_expression_plot(adata, request, mask)
    if request.plot_type in {"composition_bar", "composition_heatmap"}:
        return _build_composition_plot(adata, request, mask)
    if request.plot_type == "label_comparison":
        return _build_label_comparison_plot(adata, request, mask)
    raise ValueError(f"Unsupported plot type: {request.plot_type!r}")


__all__ = [
    "PLOT_TYPE_LABELS",
    "ScanpyPlotArtifact",
    "ScanpyPlotRequest",
    "build_scanpy_plot",
    "deterministic_stratified_indices",
    "expression_group_summary",
    "groupable_obs_columns",
    "matrix_source_choices",
    "matrix_source_var_names",
    "ordered_obs_values",
    "resolve_plot_cell_mask",
]
