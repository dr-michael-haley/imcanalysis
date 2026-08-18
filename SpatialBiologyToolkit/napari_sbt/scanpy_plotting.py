"""Read-only AnnData preparation and native Scanpy/Matplotlib QC plots."""

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
ExpressionColormap = Literal[
    "automatic", "viridis", "Blues", "Reds", "magma", "plasma", "coolwarm"
]
CompositionMeasure = Literal["count", "percent"]
ComparisonNormalisation = Literal["count", "row_percent", "column_percent"]
SideAnnotation = Literal["none", "dendrogram", "totals"]
TotalsSort = Literal["none", "ascending", "descending"]
DendrogramCorrelation = Literal["pearson", "spearman", "kendall"]
DendrogramLinkage = Literal["complete", "average", "single"]


PLOT_TYPE_LABELS: dict[str, str] = {
    "embedding": "Embedding (Scanpy)",
    "heatmap": "Marker matrix plot (Scanpy)",
    "dotplot": "Marker dot plot (Scanpy)",
    "violin": "Stacked violin (Scanpy)",
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
    expression_colormap: ExpressionColormap = "automatic"
    side_annotation: SideAnnotation = "none"
    totals_sort: TotalsSort = "none"
    dendrogram_correlation: DendrogramCorrelation = "pearson"
    dendrogram_linkage: DendrogramLinkage = "complete"
    dendrogram_optimal_ordering: bool = True
    swap_axes: bool = False
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
    groups = list(dict.fromkeys(labels.tolist()))
    # Keeping at least one point per visible category is more informative than
    # strictly enforcing a display limit smaller than the number of categories.
    maximum = max(1, int(maximum), len(groups))
    if count <= maximum:
        return np.arange(count, dtype=int)
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
    return _expression_summary_from_values(values, labels, groups, request)


def _expression_summary_from_values(
    values: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
    request: ScanpyPlotRequest,
) -> pd.DataFrame:
    """Aggregate one already-sliced marker matrix without loading it again."""

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


def _native_scanpy_modules():
    """Import the heavy plotting stack only when the user requests a plot."""

    try:
        import scanpy as sc
        from anndata import AnnData
    except ImportError as error:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "Native Scanpy plots require Scanpy and AnnData in the NapariSBT "
            "environment. Refresh the sbt-analysis environment and try again."
        ) from error
    return sc, AnnData


def _plot_obs(
    labels: np.ndarray,
    *,
    groups: list[str],
    groupby: str,
    obs_names,
) -> pd.DataFrame:
    """Create the small categorical observation table expected by Scanpy plots."""

    return pd.DataFrame(
        {
            groupby: pd.Categorical(
                np.asarray(labels, dtype=str),
                categories=list(groups),
                ordered=True,
            )
        },
        index=pd.Index(obs_names, dtype=str),
    )


def _apply_plot_palette(
    plot_adata, source_adata, groupby: str, groups: list[str]
) -> None:
    """Copy the live AnnData category palette onto a temporary plotting object."""

    colours = categorical_colour_map(source_adata, groupby)
    plot_adata.uns[f"{groupby}_colors"] = [
        colours.get(group, "#9ca3af") for group in groups
    ]


def _enable_tight_layout(figure) -> None:
    """Keep Scanpy labels and legends inside a dynamically resized canvas."""

    try:
        figure.set_layout_engine("tight", pad=1.2, w_pad=1.0, h_pad=1.0)
    except (AttributeError, RuntimeError, TypeError):
        # Compatibility with older Matplotlib releases supported by existing
        # local environments. The layout is evaluated again whenever Qt draws
        # the resized canvas.
        figure.set_tight_layout({"pad": 1.2, "w_pad": 1.0, "h_pad": 1.0})


def figure_subplot_margins(figure) -> tuple[float, float, float, float]:
    """Capture margins that can be restored before a responsive layout pass."""

    margins = figure.subplotpars
    return (
        float(margins.left),
        float(margins.right),
        float(margins.bottom),
        float(margins.top),
    )


def fit_scanpy_figure_to_canvas(
    figure,
    *,
    baseline_margins: tuple[float, float, float, float] | None = None,
    padding_pixels: float = 12.0,
    maximum_passes: int = 3,
) -> bool:
    """Fit Scanpy's nested axes to the current interactive canvas.

    Scanpy BasePlot figures use nested GridSpecs and separately positioned legend
    axes. Matplotlib's tight-layout engine can decline to adjust these figures,
    especially after Qt makes a wide matrix plot narrower than its requested
    ``figsize``. This routine measures the rendered artists and adjusts the outer
    subplot margins directly. Supplying the original margins makes repeated calls
    responsive in both directions instead of accumulating padding after each resize.
    """

    canvas = getattr(figure, "canvas", None)
    if canvas is None or not figure.axes:
        return False
    try:
        figure.set_layout_engine(None)
    except (AttributeError, RuntimeError, TypeError):
        figure.set_tight_layout(False)
    if baseline_margins is not None:
        left, right, bottom, top = baseline_margins
        figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    changed = False
    for _pass in range(max(1, int(maximum_passes))):
        canvas.draw()
        renderer = canvas.get_renderer()
        boxes = []
        for axis in figure.axes:
            if not axis.get_visible():
                continue
            box = axis.get_tightbbox(renderer)
            if box is not None and np.isfinite(box.extents).all():
                boxes.append(box)
        if not boxes:
            break

        from matplotlib.transforms import Bbox

        content = Bbox.union(boxes)
        figure_box = figure.bbox
        padding = max(0.0, float(padding_pixels))
        over_left = max(0.0, figure_box.x0 + padding - content.x0)
        over_right = max(0.0, content.x1 - (figure_box.x1 - padding))
        over_bottom = max(0.0, figure_box.y0 + padding - content.y0)
        over_top = max(0.0, content.y1 - (figure_box.y1 - padding))
        if max(over_left, over_right, over_bottom, over_top) < 0.5:
            break

        margins = figure.subplotpars
        width = max(1.0, float(figure_box.width))
        height = max(1.0, float(figure_box.height))
        new_left = float(margins.left) + over_left / width
        new_right = float(margins.right) - over_right / width
        new_bottom = float(margins.bottom) + over_bottom / height
        new_top = float(margins.top) - over_top / height
        # Keep a usable central plotting region even in a very small popup. The
        # remaining overflow will disappear naturally when the window is enlarged.
        if new_right - new_left < 0.12 or new_top - new_bottom < 0.12:
            break
        figure.subplots_adjust(
            left=new_left,
            right=new_right,
            bottom=new_bottom,
            top=new_top,
        )
        changed = True

    if changed:
        canvas.draw()
    return changed


def _expression_colormap(request: ScanpyPlotRequest) -> str:
    if request.expression_colormap != "automatic":
        return request.expression_colormap
    return "coolwarm" if request.expression_scale == "zscore_marker" else "viridis"


def _configure_scanpy_baseplot(
    sc,
    plot_adata,
    plot_object,
    request: ScanpyPlotRequest,
    groups: list[str],
) -> None:
    """Apply shared native options without consulting source-AnnData plot state."""

    if request.side_annotation == "dendrogram":
        if len(request.markers) < 2:
            raise ValueError(
                "A fresh expression dendrogram requires at least two markers."
            )
        if len(groups) < 3:
            raise ValueError(
                "A dendrogram requires at least three represented populations."
            )
        dendrogram_key = "_napari_sbt_fresh_dendrogram"
        try:
            sc.tl.dendrogram(
                plot_adata,
                groupby=request.groupby,
                var_names=request.markers,
                use_raw=False,
                cor_method=request.dendrogram_correlation,
                linkage_method=request.dendrogram_linkage,
                optimal_ordering=request.dendrogram_optimal_ordering,
                key_added=dendrogram_key,
                inplace=True,
            )
        except Exception as error:  # noqa: BLE001 - provide an actionable GUI error
            raise ValueError(
                "The fresh dendrogram could not be calculated from the currently "
                "selected cells and markers. Try adding informative markers or "
                "using more represented populations."
            ) from error
        plot_object.add_dendrogram(dendrogram_key=dendrogram_key)
    elif request.side_annotation == "totals":
        totals_sort = None if request.totals_sort == "none" else request.totals_sort
        plot_object.add_totals(sort=totals_sort)
    if request.swap_axes:
        plot_object.swap_axes()


def _expression_option_summary(request: ScanpyPlotRequest) -> str:
    options: list[str] = []
    if request.side_annotation == "dendrogram":
        options.append(
            "fresh "
            f"{request.dendrogram_correlation}/{request.dendrogram_linkage} "
            "dendrogram"
        )
    elif request.side_annotation == "totals":
        ordering = (
            "source order"
            if request.totals_sort == "none"
            else f"{request.totals_sort} order"
        )
        options.append(f"population totals ({ordering})")
    if request.swap_axes:
        options.append("axes swapped")
    options.append(f"{_expression_colormap(request)} colour map")
    return "; ".join(options)


def _scanpy_baseplot_figure(plot_object):
    """Materialize and detach a Scanpy BasePlot figure for the managed Qt popup."""

    plot_object.make_figure()
    figure = plot_object.fig
    if figure is None:
        raise RuntimeError("Scanpy did not create a Matplotlib figure.")
    # BasePlot uses nested GridSpecs plus separate legend axes. Its layout is fitted
    # after the Qt canvas has its real pixel dimensions, and again after a resize.
    try:
        figure.set_layout_engine(None)
    except (AttributeError, RuntimeError, TypeError):
        figure.set_tight_layout(False)
    # Scanpy builds through pyplot. Remove its hidden manager before NapariSBT
    # attaches the same Figure to the resizable popup canvas.
    from matplotlib import pyplot as plt

    plt.close(figure)
    return figure


def _expression_plot_adata(
    adata,
    request: ScanpyPlotRequest,
    groups: list[str],
    *,
    positions: np.ndarray,
    labels: np.ndarray,
    values: np.ndarray,
):
    """Create a marker-only AnnData so native plotting never copies full ``X``."""

    _sc, AnnData = _native_scanpy_modules()
    plot_adata = AnnData(
        X=values,
        obs=_plot_obs(
            labels,
            groups=groups,
            groupby=request.groupby,
            obs_names=adata.obs_names[positions].astype(str),
        ),
        var=pd.DataFrame(index=pd.Index(request.markers, dtype=str)),
    )
    _apply_plot_palette(plot_adata, adata, request.groupby, groups)
    return plot_adata


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
    groups = [
        group
        for group in _present_group_order(adata.obs[request.groupby], mask)
        if group in set(display_labels)
    ]
    title = f"{request.groupby} on {request.embedding_key}"
    sc, AnnData = _native_scanpy_modules()
    plot_adata = AnnData(
        X=np.empty((len(display_positions), 0), dtype=np.float32),
        obs=_plot_obs(
            display_labels,
            groups=groups,
            groupby=request.groupby,
            obs_names=adata.obs_names[display_positions].astype(str),
        ),
    )
    plot_adata.obsm[request.embedding_key] = np.asarray(
        coordinates[display_positions], dtype=float
    )
    _apply_plot_palette(plot_adata, adata, request.groupby, groups)
    figure = sc.pl.embedding(
        plot_adata,
        basis=request.embedding_key,
        color=request.groupby,
        dimensions=(x_position, y_position),
        size=float(request.point_size),
        alpha=float(request.point_alpha),
        legend_loc="on data" if request.label_centroids else "right margin",
        legend_fontsize=8,
        legend_fontweight="bold",
        frameon=True,
        title=title,
        return_fig=True,
        show=False,
    )
    if figure is None:
        raise RuntimeError("Scanpy did not create an embedding figure.")
    figure.set_size_inches(9, 7, forward=True)
    _enable_tight_layout(figure)
    from matplotlib import pyplot as plt

    plt.close(figure)
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
    positions = np.flatnonzero(mask)
    values = _expression_values(
        adata,
        source=request.matrix_source,
        markers=request.markers,
        mask=mask,
    )
    labels = _display_labels(adata.obs[request.groupby].iloc[positions])
    groups = _present_group_order(adata.obs[request.groupby], mask)
    summary = _expression_summary_from_values(values, labels, groups, request)
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
    width = max(8, min(22, len(markers) * 0.38 + 4))
    height = max(5, min(22, len(groups) * 0.32 + 3))
    colour_values = pd.DataFrame(displayed, index=groups, columns=markers)
    sc, _AnnData = _native_scanpy_modules()
    plot_adata = _expression_plot_adata(
        adata,
        request,
        groups,
        positions=positions,
        labels=labels,
        values=values,
    )
    cmap = _expression_colormap(request)
    option_summary = _expression_option_summary(request)
    if request.plot_type == "dotplot":
        title = f"{request.groupby}: marker dot plot"
        plot_object = sc.pl.dotplot(
            plot_adata,
            markers,
            groupby=request.groupby,
            use_raw=False,
            categories_order=groups,
            expression_cutoff=float(request.positivity_threshold),
            dot_color_df=colour_values,
            title=title,
            colorbar_title=scale_label,
            size_title=(f"Fraction > {float(request.positivity_threshold):g}"),
            figsize=(width, height),
            cmap=cmap,
            return_fig=True,
            show=False,
        )
        _configure_scanpy_baseplot(sc, plot_adata, plot_object, request, groups)
        figure = _scanpy_baseplot_figure(plot_object)
    elif request.plot_type == "violin":
        if len(markers) > 12:
            raise ValueError("Select at most 12 markers for a distribution plot.")
        title = f"{request.groupby}: marker distributions"
        plot_object = sc.pl.stacked_violin(
            plot_adata,
            markers,
            groupby=request.groupby,
            use_raw=False,
            categories_order=groups,
            title=title,
            colorbar_title="Median expression",
            figsize=(
                max(9, min(24, len(groups) * 0.45 + 5)),
                max(5, min(28, len(markers) * 0.75 + 3)),
            ),
            stripplot=False,
            cmap=cmap,
            return_fig=True,
            show=False,
        )
        _configure_scanpy_baseplot(sc, plot_adata, plot_object, request, groups)
        figure = _scanpy_baseplot_figure(plot_object)
        return ScanpyPlotArtifact(
            figure,
            title,
            summary,
            int(mask.sum()),
            f"{int(mask.sum()):,} selected cells; {len(groups)} populations; "
            f"{len(markers)} markers from {request.matrix_source}; {option_summary}.",
        )
    else:
        title = f"{request.groupby}: population mean marker expression"
        plot_object = sc.pl.matrixplot(
            plot_adata,
            markers,
            groupby=request.groupby,
            use_raw=False,
            categories_order=groups,
            values_df=colour_values,
            title=title,
            colorbar_title=scale_label,
            figsize=(width, height),
            cmap=cmap,
            return_fig=True,
            show=False,
        )
        _configure_scanpy_baseplot(sc, plot_adata, plot_object, request, groups)
        figure = _scanpy_baseplot_figure(plot_object)
    return ScanpyPlotArtifact(
        figure,
        title,
        summary,
        int(mask.sum()),
        f"{int(mask.sum()):,} selected cells; {len(groups)} populations; "
        f"{len(markers)} markers from {request.matrix_source}; {option_summary}.",
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
    "figure_subplot_margins",
    "fit_scanpy_figure_to_canvas",
    "groupable_obs_columns",
    "matrix_source_choices",
    "matrix_source_var_names",
    "ordered_obs_values",
    "resolve_plot_cell_mask",
]
