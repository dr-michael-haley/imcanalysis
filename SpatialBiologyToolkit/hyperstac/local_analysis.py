"""Local exploration helpers for pre-generated HyPERSTAC artifacts.

The managed HyPERSTAC stage writes patch coordinates, learned representations,
patch metrics, permutation-sensitivity scores, and normalized ROI images.  This
module turns those artifacts into reusable analysis products without rerunning
the encoder or requiring the usually very large saved patch arrays.

The functions deliberately keep the original Leiden labels unchanged.  Raster
label masks use positive integer values and reserve zero for pixels that were
not represented by an accepted patch.  The accompanying mapping table is the
authoritative translation between raster values and Leiden labels.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch, Rectangle
from scipy.stats import pearsonr, spearmanr

from SpatialBiologyToolkit.spatial_permutation import resolve_spatial_mask_paths

from .visualisation import (
    channel_mean_dataframe,
    cluster_col_umap_key,
    get_matrix,
    natural_key,
    permutation_channel_dataframe,
    safe_filename,
)

OverlapPolicy = Literal["error", "first", "last"]
GalleryMarkerSource = Literal["intensity", "zero_channel", "shuffle_channel"]
GalleryMarkerRanking = Literal[
    "relative_zscore", "cluster_mean", "difference_from_rest"
]
GallerySelectionStrategy = Literal["highest", "lowest", "median", "quantile", "random"]
GalleryContrastMode = Literal["per_patch", "per_roi", "fixed", "none"]
CorrelationMethod = Literal["spearman", "pearson"]


@dataclass(frozen=True)
class FigureSaveOptions:
    """Common output settings shared by every local HyPERSTAC plot.

    Parameters are passed to :meth:`matplotlib.figure.Figure.savefig`.  The
    explicit fields cover the common publication controls; ``savefig_kws`` is
    an escape hatch for any additional Matplotlib keyword such as
    ``orientation`` or ``pil_kwargs``.
    """

    formats: tuple[str, ...] = ("png", "svg")
    dpi: int = 220
    transparent: bool = False
    facecolor: str | None = None
    edgecolor: str | None = None
    bbox_inches: str | None = "tight"
    pad_inches: float = 0.1
    metadata: Mapping[str, Any] | None = None
    savefig_kws: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HeatmapOptions:
    """Rendering, ordering, and clustering controls for heatmap variants.

    ``cmap=None`` selects the table-aware SBT default. ``center="auto"`` uses
    zero for z-score/correlation tables and no forced center otherwise.  Set it
    explicitly to ``None`` to disable centering for a diverging table.

    ``heatmap_kws`` and ``clustermap_kws`` are applied last, so advanced
    Seaborn features such as custom color bars, masks, row/column colors, or
    dendrogram ratios remain available without expanding this API again.
    """

    cmap: Any | None = None
    center: float | None | Literal["auto"] = "auto"
    vmin: float | None = None
    vmax: float | None = None
    robust: bool = False
    annot: bool | np.ndarray | pd.DataFrame = False
    fmt: str = ".2g"
    annot_kws: Mapping[str, Any] = field(default_factory=dict)
    linewidths: float = 0.1
    linecolor: str = "white"
    cbar: bool = True
    cbar_kws: Mapping[str, Any] = field(default_factory=dict)
    square: bool = False
    mask: np.ndarray | pd.DataFrame | None = None
    xticklabels: Any = "auto"
    yticklabels: Any = "auto"
    x_tick_rotation: float = 90.0
    y_tick_rotation: float = 0.0
    tick_fontsize: float | None = None
    x_tick_fontsize: float | None = None
    y_tick_fontsize: float | None = None
    tick_params_kws: Mapping[str, Any] = field(default_factory=dict)
    xlabel: str | None = None
    ylabel: str | None = None
    xlabel_fontsize: float | None = None
    ylabel_fontsize: float | None = None
    xlabel_kws: Mapping[str, Any] = field(default_factory=dict)
    ylabel_kws: Mapping[str, Any] = field(default_factory=dict)
    title: str | None = None
    show_title: bool = True
    title_fontsize: float | None = None
    title_pad: float | None = None
    clustermap_title_y: float = 1.02
    title_kws: Mapping[str, Any] = field(default_factory=dict)
    cbar_tick_fontsize: float | None = None
    cbar_label: str | None = None
    cbar_label_fontsize: float | None = None
    figsize: tuple[float, float] | None = None
    min_figsize: tuple[float, float] = (6.0, 4.0)
    max_figsize: tuple[float, float] = (18.0, 22.0)
    cell_width: float = 0.42
    cell_height: float = 0.28
    transpose: bool = False
    row_order: Sequence[Any] | None = None
    column_order: Sequence[Any] | None = None
    row_labels: Mapping[Any, Any] | None = None
    column_labels: Mapping[Any, Any] | None = None
    show_fixed: bool = True
    show_clustermap: bool = True
    row_cluster: bool = True
    col_cluster: bool = True
    method: str = "average"
    metric: str = "euclidean"
    z_score: int | None = None
    standard_scale: int | None = None
    dendrogram_ratio: float | tuple[float, float] = 0.2
    colors_ratio: float | tuple[float, float] = 0.03
    cbar_pos: tuple[float, float, float, float] | None = None
    rasterized: bool = False
    tight_layout: bool = True
    tight_layout_kws: Mapping[str, Any] = field(default_factory=dict)
    subplot_adjust_kws: Mapping[str, Any] = field(default_factory=dict)
    heatmap_kws: Mapping[str, Any] = field(default_factory=dict)
    clustermap_kws: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HyperstacMaskResult:
    """Files and tables produced while reconstructing ROI label masks."""

    mask_dir: Path
    preview_dir: Path | None
    mapping_path: Path
    manifest_path: Path
    mapping: pd.DataFrame
    manifest: pd.DataFrame
    colorized_mask_dir: Path | None = None


@dataclass(frozen=True)
class HyperstacGalleryResult:
    """Files and selection metadata produced by the environment gallery."""

    image_paths: tuple[Path, ...]
    marker_table_path: Path
    selection_table_path: Path
    marker_table: pd.DataFrame
    selection_table: pd.DataFrame


@dataclass(frozen=True)
class EnvironmentAbundanceTables:
    """Plot-ready abundance tables, independent of any plotting library.

    These tables are the recommended interface for one-off publication figures:
    pass the desired matrix directly to :func:`seaborn.heatmap`,
    :func:`seaborn.clustermap`, Matplotlib, or another plotting library.  The
    batch :func:`plot_environment_abundance` wrapper remains available for
    standard exploratory output.
    """

    overall_fraction: pd.Series
    roi_fraction: pd.DataFrame
    sample_group_fraction: pd.DataFrame | None
    categorical_fraction: Mapping[str, pd.DataFrame]
    numeric_correlations: pd.DataFrame | None
    numeric_pvalues: pd.DataFrame | None
    environment_order: tuple[str, ...]


def _obs_frame(adata_or_obs: ad.AnnData | pd.DataFrame) -> pd.DataFrame:
    if isinstance(adata_or_obs, pd.DataFrame):
        return adata_or_obs
    if hasattr(adata_or_obs, "obs"):
        return adata_or_obs.obs
    raise TypeError("Expected an AnnData-like object or a pandas DataFrame")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _save_figure(
    fig: plt.Figure,
    output_stem: str | Path,
    *,
    options: FigureSaveOptions,
) -> tuple[Path, ...]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if options.dpi <= 0:
        raise ValueError("FigureSaveOptions.dpi must be positive")
    if options.pad_inches < 0:
        raise ValueError("FigureSaveOptions.pad_inches cannot be negative")
    for raw_format in options.formats:
        image_format = str(raw_format).lower().lstrip(".")
        if not image_format:
            raise ValueError("Figure formats cannot contain an empty value")
        path = stem.with_suffix(f".{image_format}")
        savefig_kws = {
            "dpi": options.dpi,
            "transparent": options.transparent,
            "bbox_inches": options.bbox_inches,
            "pad_inches": options.pad_inches,
        }
        if options.facecolor is not None:
            savefig_kws["facecolor"] = options.facecolor
        if options.edgecolor is not None:
            savefig_kws["edgecolor"] = options.edgecolor
        if options.metadata is not None:
            savefig_kws["metadata"] = dict(options.metadata)
        savefig_kws.update(options.savefig_kws)
        fig.savefig(path, **savefig_kws)
        paths.append(path)
    return tuple(paths)


def _resolve_save_options(
    save_options: FigureSaveOptions | None,
    *,
    formats: Sequence[str] | None,
    dpi: int | None,
) -> FigureSaveOptions:
    options = save_options or FigureSaveOptions()
    if formats is not None:
        options = replace(options, formats=tuple(str(value) for value in formats))
    if dpi is not None:
        options = replace(options, dpi=int(dpi))
    if not options.formats:
        raise ValueError("At least one output format is required")
    return options


def _apply_subplot_layout(
    fig: plt.Figure,
    *,
    use_tight_layout: bool,
    tight_layout_kws: Mapping[str, Any] | None,
    row_spacing: float | None,
    column_spacing: float | None,
    subplot_adjust_kws: Mapping[str, Any] | None,
) -> None:
    """Apply discoverable grid spacing after optional tight-layout sizing."""

    if row_spacing is not None and row_spacing < 0:
        raise ValueError("row_spacing cannot be negative")
    if column_spacing is not None and column_spacing < 0:
        raise ValueError("column_spacing cannot be negative")
    if use_tight_layout:
        fig.tight_layout(**dict(tight_layout_kws or {}))
    adjust = dict(subplot_adjust_kws or {})
    if row_spacing is not None:
        adjust.setdefault("hspace", row_spacing)
    if column_spacing is not None:
        adjust.setdefault("wspace", column_spacing)
    if adjust:
        fig.subplots_adjust(**adjust)


def _pack_gallery_columns(
    image_axes: np.ndarray,
    key_axes: np.ndarray,
    *,
    column_spacing: float | None,
    label_width: float,
    panel_width: float,
) -> None:
    """Pack equal-aspect gallery axes without hidden GridSpec cell padding.

    Matplotlib shrinks an ``imshow`` axis inside a wider GridSpec cell to retain
    square pixels. ``wspace=0`` only joins the cells and does not remove that
    internal padding. Repositioning the already-square axes makes the requested
    gap literal while preserving image aspect.
    """

    if column_spacing is None:
        return
    if column_spacing < 0:
        raise ValueError("column_spacing cannot be negative")
    for row_index in range(image_axes.shape[0]):
        row_axes = image_axes[row_index]
        first_position = row_axes[0].get_position()
        panel_gap = column_spacing * first_position.width
        next_x = first_position.x0
        for ax in row_axes:
            position = ax.get_position()
            ax.set_position(
                [next_x, position.y0, position.width, position.height],
                which="both",
            )
            next_x += position.width + panel_gap
        key_ax = key_axes[row_index]
        if key_ax is not None:
            key_width = first_position.width * label_width / panel_width
            key_ax.set_position(
                [
                    first_position.x0 - panel_gap - key_width,
                    first_position.y0,
                    key_width,
                    first_position.height,
                ],
                which="both",
            )


def _ordered_values(
    available: Sequence[Any],
    requested: Sequence[Any] | None,
    *,
    label: str,
) -> list[str]:
    available_strings = [str(value) for value in available]
    if requested is None:
        return sorted(available_strings, key=natural_key)
    requested_strings = [str(value) for value in requested]
    duplicates = sorted(
        {value for value in requested_strings if requested_strings.count(value) > 1}
    )
    if duplicates:
        raise ValueError(f"{label} contains duplicate values: {duplicates}")
    missing = sorted(
        set(available_strings).difference(requested_strings), key=natural_key
    )
    unknown = sorted(
        set(requested_strings).difference(available_strings), key=natural_key
    )
    if missing or unknown:
        raise ValueError(
            f"{label} must contain every available value exactly once; missing={missing}, unknown={unknown}"
        )
    return requested_strings


def _cluster_mapping(
    labels: pd.Series,
    cluster_order: Sequence[Any] | None = None,
) -> pd.DataFrame:
    if labels.isna().any():
        raise ValueError(
            "Cluster labels contain missing values; assign or remove them before rasterization"
        )
    clusters = _ordered_values(
        labels.astype(str).unique(),
        cluster_order,
        label="cluster_order",
    )
    if not clusters:
        raise ValueError("No cluster labels were found")
    return pd.DataFrame(
        {
            "cluster": clusters,
            "mask_value": np.arange(1, len(clusters) + 1, dtype=np.uint16),
        }
    )


def _cluster_palette(
    mapping: pd.DataFrame,
    palette: str | Sequence[Any] | Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    clusters = mapping["cluster"].astype(str).tolist()
    if isinstance(palette, Mapping):
        normalized = {str(key): value for key, value in palette.items()}
        missing = [cluster for cluster in clusters if cluster not in normalized]
        if missing:
            raise ValueError(f"palette is missing cluster colors for: {missing}")
        return {cluster: normalized[cluster] for cluster in clusters}
    if isinstance(palette, str) or palette is None:
        cmap_name = palette or ("tab20" if len(mapping) <= 20 else "gist_ncar")
        cmap = plt.colormaps[cmap_name].resampled(max(len(mapping), 1))
        return {cluster: cmap(position) for position, cluster in enumerate(clusters)}
    colors = list(palette)
    if len(colors) < len(clusters):
        raise ValueError(
            f"palette supplies {len(colors)} colors for {len(clusters)} clusters"
        )
    return dict(zip(clusters, colors))


def _roi_image_shape(normalised_image_dir: Path, roi: str) -> tuple[int, int]:
    roi_dir = normalised_image_dir / roi
    images = sorted(roi_dir.glob("*.tif")) + sorted(roi_dir.glob("*.tiff"))
    if not images:
        raise FileNotFoundError(
            f"No normalized TIFF images were found for ROI {roi!r} in {roi_dir}"
        )
    with tifffile.TiffFile(images[0]) as handle:
        shape = tuple(int(value) for value in handle.series[0].shape)
    if len(shape) != 2:
        raise ValueError(
            f"Expected a 2D normalized image for ROI {roi!r}, found shape {shape}"
        )
    return shape


def _roi_dimensions(
    roi_obs: pd.DataFrame,
    *,
    roi: str,
    normalised_image_dir: Path | None,
) -> tuple[int, int]:
    coordinate_shape = (
        int(pd.to_numeric(roi_obs["row_end"], errors="raise").max()),
        int(pd.to_numeric(roi_obs["col_end"], errors="raise").max()),
    )
    metadata_shape: tuple[int, int] | None = None
    if {"roi_height_px", "roi_width_px"}.issubset(roi_obs.columns):
        heights = (
            pd.to_numeric(roi_obs["roi_height_px"], errors="raise")
            .dropna()
            .astype(int)
            .unique()
        )
        widths = (
            pd.to_numeric(roi_obs["roi_width_px"], errors="raise")
            .dropna()
            .astype(int)
            .unique()
        )
        if len(heights) != 1 or len(widths) != 1:
            raise ValueError(
                f"ROI {roi!r} has inconsistent roi_height_px or roi_width_px values"
            )
        metadata_shape = (int(heights[0]), int(widths[0]))
        if (
            coordinate_shape[0] > metadata_shape[0]
            or coordinate_shape[1] > metadata_shape[1]
        ):
            raise ValueError(
                f"Patch coordinates for ROI {roi!r} exceed the declared image shape {metadata_shape}"
            )

    image_shape = None
    if normalised_image_dir is not None:
        image_shape = _roi_image_shape(normalised_image_dir, roi)
        if metadata_shape is not None and image_shape != metadata_shape:
            raise ValueError(
                f"ROI {roi!r} image shape {image_shape} does not match patch metadata {metadata_shape}"
            )
        if coordinate_shape[0] > image_shape[0] or coordinate_shape[1] > image_shape[1]:
            raise ValueError(
                f"Patch coordinates for ROI {roi!r} exceed normalized image shape {image_shape}"
            )

    return image_shape or metadata_shape or coordinate_shape


def _label_colormap(
    mapping: pd.DataFrame,
    *,
    palette: str | Sequence[Any] | Mapping[Any, Any] | None = None,
    uncovered_color: Any = "black",
    cluster_labels: Mapping[Any, str] | None = None,
    uncovered_label: str = "uncovered (0)",
) -> tuple[ListedColormap, list[Patch]]:
    color_lookup = _cluster_palette(mapping, palette)
    readable_labels = {
        str(key): str(value) for key, value in (cluster_labels or {}).items()
    }
    colors = [uncovered_color] + [
        color_lookup[str(cluster)] for cluster in mapping["cluster"]
    ]
    handles = [Patch(facecolor=uncovered_color, label=uncovered_label)]
    handles.extend(
        Patch(
            facecolor=color_lookup[str(row.cluster)],
            label=f"{readable_labels.get(str(row.cluster), str(row.cluster))} ({row.mask_value})",
        )
        for row in mapping.itertuples(index=False)
    )
    return ListedColormap(colors), handles


def reconstruct_cluster_label_masks(
    adata_or_obs: ad.AnnData | pd.DataFrame,
    cluster_col: str,
    output_dir: str | Path,
    *,
    roi_col: str = "roi",
    normalised_image_dir: str | Path | None = None,
    mask_pattern: str = "{roi}__hyperstac_labels.tiff",
    overlap_policy: OverlapPolicy = "error",
    write_colorized: bool = True,
    colorized_tiff_pattern: str = "{roi}__hyperstac_labels_colorized.tiff",
    colorized_png_pattern: str = "{roi}__hyperstac_labels_colorized.png",
    palette: str | Sequence[Any] | Mapping[Any, Any] | None = None,
    colorized_uncovered_color: Any = "black",
    write_previews: bool = False,
    preview_rois: Sequence[str] | None = None,
    cluster_order: Sequence[Any] | None = None,
    preview_palette: str | Sequence[Any] | Mapping[Any, Any] | None = None,
    preview_cluster_labels: Mapping[Any, str] | None = None,
    preview_uncovered_color: Any = "black",
    preview_uncovered_label: str = "uncovered (0)",
    preview_figsize: tuple[float, float] = (7.0, 7.0),
    preview_title_template: str = "{roi}: {cluster_col}",
    preview_show_legend: bool = True,
    preview_interpolation: str = "nearest",
    preview_figure_kws: Mapping[str, Any] | None = None,
    preview_imshow_kws: Mapping[str, Any] | None = None,
    preview_title_kws: Mapping[str, Any] | None = None,
    preview_legend_kws: Mapping[str, Any] | None = None,
    preview_save_options: FigureSaveOptions | None = None,
) -> HyperstacMaskResult:
    """Reconstruct Leiden patches as exact-size categorical ROI rasters.

    Positive mask values correspond to clusters through the written mapping
    CSV. Zero is retained for pixels with no accepted HyPERSTAC patch, which is
    important when these masks are subsequently used as spatial tissue
    regions. By default, matching exact-size RGB TIFF and PNG copies are also
    written for immediate viewing. ``palette`` is shared by those colorized
    rasters; ``preview_palette`` can override it only for optional figure
    previews. Image dimensions are validated against normalized TIFFs when
    ``normalised_image_dir`` is supplied.
    """

    if overlap_policy not in {"error", "first", "last"}:
        raise ValueError("overlap_policy must be one of: 'error', 'first', 'last'")
    if "{roi}" not in mask_pattern:
        raise ValueError("mask_pattern must contain the '{roi}' placeholder")
    for pattern_name, pattern in (
        ("colorized_tiff_pattern", colorized_tiff_pattern),
        ("colorized_png_pattern", colorized_png_pattern),
    ):
        if "{roi}" not in pattern:
            raise ValueError(f"{pattern_name} must contain the '{{roi}}' placeholder")

    obs = _obs_frame(adata_or_obs)
    required = [roi_col, "row_start", "row_end", "col_start", "col_end", cluster_col]
    _require_columns(obs, required, "HyPERSTAC patch observations")
    if obs[roi_col].isna().any():
        raise ValueError(f"{roi_col!r} contains missing ROI identifiers")

    output_root = Path(output_dir)
    mask_dir = output_root / "label_masks"
    colorized_mask_dir = (
        output_root / "colorized_label_masks" if write_colorized else None
    )
    table_dir = output_root / "tables"
    preview_dir = output_root / "cluster_maps" if write_previews else None
    mask_dir.mkdir(parents=True, exist_ok=True)
    if colorized_mask_dir is not None:
        colorized_mask_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    image_root = (
        Path(normalised_image_dir) if normalised_image_dir is not None else None
    )
    mapping = _cluster_mapping(obs[cluster_col], cluster_order)
    value_lookup = dict(
        zip(mapping["cluster"].astype(str), mapping["mask_value"].astype(int))
    )
    mapping_path = table_dir / "hyperstac_cluster_label_mapping.csv"
    mapping.to_csv(mapping_path, index=False)

    requested_previews = (
        None if preview_rois is None else {str(roi) for roi in preview_rois}
    )
    color_lookup = _cluster_palette(mapping, palette)
    cmap, legend_handles = _label_colormap(
        mapping,
        palette=preview_palette if preview_palette is not None else palette,
        uncovered_color=preview_uncovered_color,
        cluster_labels=preview_cluster_labels,
        uncovered_label=preview_uncovered_label,
    )
    resolved_preview_save = preview_save_options or FigureSaveOptions(formats=("png",))
    output_root_resolved = mask_dir.resolve()
    colorized_root_resolved = (
        colorized_mask_dir.resolve() if colorized_mask_dir is not None else None
    )
    manifest_rows: list[dict[str, Any]] = []

    roi_values = obs[roi_col].astype(str)
    for roi in sorted(roi_values.unique(), key=natural_key):
        roi_obs = obs.loc[roi_values == roi]
        height, width = _roi_dimensions(
            roi_obs, roi=roi, normalised_image_dir=image_root
        )
        mask = np.zeros((height, width), dtype=np.uint16)
        occupied = np.zeros((height, width), dtype=bool)

        for patch_id, row in roi_obs.iterrows():
            row_start = int(row["row_start"])
            row_end = int(row["row_end"])
            col_start = int(row["col_start"])
            col_end = int(row["col_end"])
            if (
                row_start < 0
                or col_start < 0
                or row_end <= row_start
                or col_end <= col_start
            ):
                raise ValueError(f"Patch {patch_id!r} has invalid bounds")
            if row_end > height or col_end > width:
                raise ValueError(
                    f"Patch {patch_id!r} exceeds ROI {roi!r} dimensions {(height, width)}"
                )
            region_occupied = occupied[row_start:row_end, col_start:col_end]
            if overlap_policy == "error" and region_occupied.any():
                raise ValueError(
                    f"Patch {patch_id!r} overlaps a previous patch in ROI {roi!r}; "
                    "choose overlap_policy='first' or 'last' only if this is intentional"
                )
            value = value_lookup[str(row[cluster_col])]
            region_mask = mask[row_start:row_end, col_start:col_end]
            if overlap_policy == "first":
                region_mask[~region_occupied] = value
            else:
                region_mask[...] = value
            region_occupied[...] = True

        relative = Path(mask_pattern.format(roi=roi))
        if relative.is_absolute():
            raise ValueError(
                "mask_pattern must create paths relative to output_dir/label_masks"
            )
        mask_path = (mask_dir / relative).resolve()
        if not mask_path.is_relative_to(output_root_resolved):
            raise ValueError(
                f"mask_pattern for ROI {roi!r} resolves outside the mask directory"
            )
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(mask_path, mask, compression="zlib")

        colorized_tiff_path: Path | None = None
        colorized_png_path: Path | None = None
        if colorized_mask_dir is not None and colorized_root_resolved is not None:
            rgb = np.empty((*mask.shape, 3), dtype=np.uint8)
            rgb[...] = np.rint(
                np.asarray(to_rgb(colorized_uncovered_color)) * 255
            ).astype(np.uint8)
            for row in mapping.itertuples(index=False):
                rgb[mask == int(row.mask_value)] = np.rint(
                    np.asarray(to_rgb(color_lookup[str(row.cluster)])) * 255
                ).astype(np.uint8)

            colorized_paths: list[Path] = []
            for pattern_name, pattern in (
                ("colorized_tiff_pattern", colorized_tiff_pattern),
                ("colorized_png_pattern", colorized_png_pattern),
            ):
                relative_colorized = Path(pattern.format(roi=roi))
                if relative_colorized.is_absolute():
                    raise ValueError(
                        f"{pattern_name} must create paths relative to "
                        "output_dir/colorized_label_masks"
                    )
                colorized_path = (colorized_mask_dir / relative_colorized).resolve()
                if not colorized_path.is_relative_to(colorized_root_resolved):
                    raise ValueError(
                        f"{pattern_name} for ROI {roi!r} resolves outside the "
                        "colorized mask directory"
                    )
                colorized_path.parent.mkdir(parents=True, exist_ok=True)
                colorized_paths.append(colorized_path)
            colorized_tiff_path, colorized_png_path = colorized_paths
            tifffile.imwrite(colorized_tiff_path, rgb, compression="zlib")
            plt.imsave(colorized_png_path, rgb)

        should_preview = write_previews and (
            requested_previews is None or roi in requested_previews
        )
        preview_path: Path | None = None
        if should_preview and preview_dir is not None:
            figure_kws = dict(preview_figure_kws or {})
            figure_kws.setdefault("figsize", preview_figsize)
            fig, ax = plt.subplots(**figure_kws)
            imshow_kws = {
                "cmap": cmap,
                "interpolation": preview_interpolation,
                "vmin": 0,
                "vmax": len(mapping),
            }
            imshow_kws.update(preview_imshow_kws or {})
            ax.imshow(mask, **imshow_kws)
            ax.set_title(
                preview_title_template.format(roi=roi, cluster_col=cluster_col),
                **dict(preview_title_kws or {}),
            )
            ax.set_axis_off()
            if preview_show_legend:
                legend_kws = {
                    "fontsize": 7,
                    "bbox_to_anchor": (1.02, 1),
                    "loc": "upper left",
                }
                legend_kws.update(preview_legend_kws or {})
                ax.legend(handles=legend_handles, **legend_kws)
            fig.tight_layout()
            preview_stem = preview_dir / f"{safe_filename(roi)}__cluster_map"
            preview_paths = _save_figure(
                fig, preview_stem, options=resolved_preview_save
            )
            preview_path = preview_paths[0]
            plt.close(fig)

        assigned_pixels = int(np.count_nonzero(mask))
        manifest_rows.append(
            {
                "roi": roi,
                "mask_path": str(mask_path),
                "colorized_tiff_path": (
                    str(colorized_tiff_path)
                    if colorized_tiff_path is not None
                    else None
                ),
                "colorized_png_path": (
                    str(colorized_png_path) if colorized_png_path is not None else None
                ),
                "preview_path": str(preview_path) if preview_path is not None else None,
                "height_px": height,
                "width_px": width,
                "n_patches": len(roi_obs),
                "assigned_pixels": assigned_pixels,
                "total_pixels": int(mask.size),
                "coverage_fraction": assigned_pixels / mask.size,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = table_dir / "hyperstac_label_mask_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return HyperstacMaskResult(
        mask_dir=mask_dir,
        colorized_mask_dir=colorized_mask_dir,
        preview_dir=preview_dir,
        mapping_path=mapping_path,
        manifest_path=manifest_path,
        mapping=mapping,
        manifest=manifest,
    )


def plot_cluster_map_gallery(
    mask_result: HyperstacMaskResult,
    output_stem: str | Path,
    *,
    rois: Sequence[str] | None = None,
    max_rois: int = 12,
    ncols: int = 4,
    roi_selection: Literal[
        "even", "first", "highest_coverage", "lowest_coverage"
    ] = "even",
    palette: str | Sequence[Any] | Mapping[Any, Any] | None = None,
    cluster_labels: Mapping[Any, str] | None = None,
    uncovered_color: Any = "black",
    uncovered_label: str = "uncovered (0)",
    show_uncovered: bool = True,
    background_image_dir: str | Path | None = None,
    background_channel: str | None = None,
    background_cmap: Any = "gray",
    background_percentiles: tuple[float, float] = (1.0, 99.5),
    background_imshow_kws: Mapping[str, Any] | None = None,
    mask_alpha: float = 1.0,
    interpolation: str = "nearest",
    origin: str = "upper",
    imshow_kws: Mapping[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
    panel_width: float = 3.2,
    panel_height: float = 3.1,
    legend_width: float = 2.2,
    title: str | None = "HyPERSTAC environments back-gated to original ROI coordinates",
    show_title: bool = True,
    title_fontsize: float | None = 12.0,
    roi_title_template: str = "{roi}",
    show_roi_titles: bool = True,
    roi_title_fontsize: float | None = 9.0,
    roi_title_kws: Mapping[str, Any] | None = None,
    title_kws: Mapping[str, Any] | None = None,
    show_legend: bool = True,
    include_uncovered_in_legend: bool = True,
    legend_title: str = "Leiden (mask value)",
    legend_fontsize: float | None = 7.0,
    legend_title_fontsize: float | None = 8.0,
    legend_kws: Mapping[str, Any] | None = None,
    panel_border_color: Any | None = "black",
    panel_border_width: float = 0.5,
    axis_off: bool = True,
    figure_kws: Mapping[str, Any] | None = None,
    row_spacing: float | None = 0.12,
    column_spacing: float | None = 0.08,
    use_tight_layout: bool = True,
    tight_layout_kws: Mapping[str, Any] | None = None,
    subplot_adjust_kws: Mapping[str, Any] | None = None,
    formats: Sequence[str] | None = None,
    dpi: int | None = None,
    save_options: FigureSaveOptions | None = None,
) -> tuple[Path, ...]:
    """Plot several reconstructed ROI label masks in one customizable figure.

    Supply ``background_image_dir`` and ``background_channel`` to overlay
    semi-transparent labels on a normalized marker image. ``imshow_kws``,
    ``legend_kws``, and the title keyword mappings are forwarded to Matplotlib.
    ROI/global title visibility, fonts, grid spacing, and panel outlines have
    explicit arguments because these are commonly adjusted for manuscripts.
    """

    manifest = mask_result.manifest.copy()
    if rois is not None:
        requested = [str(roi) for roi in rois]
        missing = sorted(set(requested).difference(manifest["roi"]))
        if missing:
            raise ValueError(
                f"Requested ROIs are absent from the mask manifest: {missing}"
            )
        manifest = manifest.set_index("roi").loc[requested].reset_index()
    elif len(manifest) > max_rois:
        if roi_selection == "even":
            positions = np.linspace(0, len(manifest) - 1, max_rois, dtype=int)
            manifest = manifest.iloc[positions]
        elif roi_selection == "first":
            manifest = manifest.iloc[:max_rois]
        elif roi_selection == "highest_coverage":
            manifest = manifest.nlargest(max_rois, "coverage_fraction")
        elif roi_selection == "lowest_coverage":
            manifest = manifest.nsmallest(max_rois, "coverage_fraction")
        else:
            raise ValueError(
                "roi_selection must be 'even', 'first', 'highest_coverage', or 'lowest_coverage'"
            )
    if manifest.empty:
        raise ValueError("No ROI masks were selected")
    if max_rois < 1 or ncols < 1:
        raise ValueError("max_rois and ncols must be positive")
    if not 0 <= mask_alpha <= 1:
        raise ValueError("mask_alpha must be between zero and one")
    if panel_border_width < 0:
        raise ValueError("panel_border_width cannot be negative")
    low_percentile, high_percentile = background_percentiles
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError(
            "background_percentiles must be increasing values within [0, 100]"
        )
    if (background_image_dir is None) != (background_channel is None):
        raise ValueError(
            "background_image_dir and background_channel must be supplied together"
        )

    ncols = max(1, min(int(ncols), len(manifest)))
    nrows = int(np.ceil(len(manifest) / ncols))
    cmap, handles = _label_colormap(
        mask_result.mapping,
        palette=palette,
        uncovered_color=uncovered_color,
        cluster_labels=cluster_labels,
        uncovered_label=uncovered_label,
    )
    figure_options = dict(figure_kws or {})
    figure_options.setdefault(
        "figsize",
        figsize
        or (
            panel_width * ncols + (legend_width if show_legend else 0),
            panel_height * nrows,
        ),
    )
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, **figure_options)
    background_root = (
        Path(background_image_dir) if background_image_dir is not None else None
    )
    for ax, row in zip(axes.flat, manifest.itertuples(index=False)):
        mask = tifffile.imread(row.mask_path)
        if background_root is not None and background_channel is not None:
            background = np.asarray(
                tifffile.memmap(
                    _normalised_channel_path(
                        background_root, str(row.roi), background_channel
                    )
                ),
                dtype=np.float32,
            )
            vmin, vmax = np.nanpercentile(background, [low_percentile, high_percentile])
            background_kws = {
                "cmap": background_cmap,
                "vmin": float(vmin),
                "vmax": float(vmax),
                "origin": origin,
                "interpolation": interpolation,
            }
            background_kws.update(background_imshow_kws or {})
            ax.imshow(background, **background_kws)
        display_mask: Any = mask if show_uncovered else np.ma.masked_equal(mask, 0)
        label_kws = {
            "cmap": cmap,
            "interpolation": interpolation,
            "origin": origin,
            "vmin": 0,
            "vmax": len(mask_result.mapping),
            "alpha": mask_alpha,
        }
        label_kws.update(imshow_kws or {})
        ax.imshow(display_mask, **label_kws)
        if show_roi_titles:
            ax.set_title(
                roi_title_template.format(
                    roi=row.roi,
                    coverage_fraction=row.coverage_fraction,
                    n_patches=row.n_patches,
                    height_px=row.height_px,
                    width_px=row.width_px,
                ),
                **({"fontsize": roi_title_fontsize} | dict(roi_title_kws or {})),
            )
        if panel_border_color is not None and panel_border_width > 0:
            ax.add_patch(
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=ax.transAxes,
                    fill=False,
                    clip_on=False,
                    edgecolor=panel_border_color,
                    linewidth=panel_border_width,
                )
            )
        if axis_off:
            ax.set_axis_off()
    for ax in axes.flat[len(manifest) :]:
        ax.set_visible(False)
    if show_legend:
        legend_options = {
            "fontsize": legend_fontsize,
            "title_fontsize": legend_title_fontsize,
            "bbox_to_anchor": (1.05, 1),
            "loc": "upper left",
            "title": legend_title,
        }
        legend_options.update(legend_kws or {})
        displayed_handles = handles if include_uncovered_in_legend else handles[1:]
        axes.flat[min(len(manifest), len(axes.flat)) - 1].legend(
            handles=displayed_handles,
            **legend_options,
        )
    if show_title and title:
        suptitle_options = {"y": 1.01, "fontsize": title_fontsize}
        suptitle_options.update(title_kws or {})
        fig.suptitle(title, **suptitle_options)
    _apply_subplot_layout(
        fig,
        use_tight_layout=use_tight_layout,
        tight_layout_kws=tight_layout_kws,
        row_spacing=row_spacing,
        column_spacing=column_spacing,
        subplot_adjust_kws=subplot_adjust_kws,
    )
    resolved_save = _resolve_save_options(save_options, formats=formats, dpi=dpi)
    paths = _save_figure(fig, output_stem, options=resolved_save)
    plt.close(fig)
    return paths


def _align_feature_frame(
    reference: ad.AnnData, frame: pd.DataFrame, label: str
) -> pd.DataFrame:
    missing = reference.obs_names.difference(frame.index)
    if len(missing):
        raise ValueError(
            f"{label} is missing {len(missing)} patches from the representation AnnData"
        )
    return frame.loc[reference.obs_names]


def hyperstac_cluster_feature_tables(
    representation_adata: ad.AnnData,
    metrics_adata: ad.AnnData,
    cluster_col: str,
    *,
    permutation_adata: ad.AnnData | None = None,
    cluster_order: Sequence[Any] | None = None,
    channel_order: Sequence[str] | None = None,
    summary_statistic: Literal["mean", "median"]
    | Callable[[pd.Series], float] = "mean",
    zscore_ddof: int = 1,
    permutation_types: Sequence[str] = ("zero_channel", "shuffle_channel"),
    include_all_channel: bool = True,
) -> dict[str, pd.DataFrame]:
    """Summarize marker intensity and perturbation sensitivity by cluster.

    The default returns the original mean-based tables. Marker order/subsetting,
    cluster order, mean versus median/custom aggregation, z-score degrees of
    freedom, perturbation types, and all-channel inclusion are configurable.
    """

    _require_columns(
        representation_adata.obs, [cluster_col], "representation AnnData obs"
    )
    clusters = representation_adata.obs[cluster_col].astype(str)
    order = _ordered_values(clusters.unique(), cluster_order, label="cluster_order")
    if zscore_ddof < 0:
        raise ValueError("zscore_ddof cannot be negative")
    if isinstance(summary_statistic, str) and summary_statistic not in {
        "mean",
        "median",
    }:
        raise ValueError("summary_statistic must be 'mean', 'median', or a callable")
    statistic_label = (
        summary_statistic if isinstance(summary_statistic, str) else "custom"
    )

    def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
        grouped = frame.groupby(clusters, observed=True)
        if summary_statistic == "mean":
            result = grouped.mean()
        elif summary_statistic == "median":
            result = grouped.median()
        else:
            result = grouped.aggregate(summary_statistic)
        return result.reindex(order)

    intensity = _align_feature_frame(
        representation_adata,
        channel_mean_dataframe(metrics_adata),
        "metrics AnnData",
    )
    if channel_order is not None:
        requested_channels = [str(channel) for channel in channel_order]
        unknown = [
            channel
            for channel in requested_channels
            if channel not in intensity.columns
        ]
        if unknown:
            raise ValueError(f"channel_order contains unknown channels: {unknown}")
        intensity = intensity.loc[:, requested_channels]
    intensity_mean = aggregate(intensity)
    intensity_z = (
        (intensity_mean - intensity_mean.mean(axis=0))
        / intensity_mean.std(axis=0, ddof=zscore_ddof).replace(0, np.nan)
    ).fillna(0.0)
    tables = {
        f"{statistic_label}_marker_intensity": intensity_mean,
        "relative_marker_intensity_zscore": intensity_z,
    }

    if permutation_adata is not None:
        for perturbation_type in permutation_types:
            scores = permutation_channel_dataframe(permutation_adata, perturbation_type)
            if scores.empty:
                continue
            scores = _align_feature_frame(
                representation_adata,
                scores,
                f"{perturbation_type} permutation AnnData",
            )
            if channel_order is not None:
                shared_order = [
                    channel
                    for channel in requested_channels
                    if channel in scores.columns
                ]
                scores = scores.loc[:, shared_order]
            tables[f"{statistic_label}_{perturbation_type}_cosine_distance"] = (
                aggregate(scores)
            )

        var = permutation_adata.var
        if include_all_channel and {"perturbation_type", "channel"}.issubset(
            var.columns
        ):
            matrix = get_matrix(permutation_adata)
            all_channel_rows: dict[str, np.ndarray] = {}
            channel_values = var["channel"].astype(str)
            type_values = var["perturbation_type"].astype(str)
            for perturbation_type in sorted(
                type_values[channel_values == "all_channels"].unique()
            ):
                positions = np.flatnonzero(
                    (
                        (channel_values == "all_channels")
                        & (type_values == perturbation_type)
                    ).to_numpy()
                )
                if len(positions):
                    all_channel_rows[perturbation_type] = np.nanmean(
                        matrix[:, positions], axis=1
                    )
            if all_channel_rows:
                all_scores = pd.DataFrame(
                    all_channel_rows, index=permutation_adata.obs_names
                )
                all_scores = _align_feature_frame(
                    representation_adata,
                    all_scores,
                    "all-channel permutation AnnData",
                )
                tables[
                    f"{statistic_label}_all_channel_perturbation_cosine_distance"
                ] = aggregate(all_scores)
    return tables


def _heatmap_style(name: str) -> tuple[str, float | None]:
    if "zscore" in name or "correlation" in name:
        return "vlag", 0.0
    if "permutation" in name or "cosine_distance" in name:
        return "rocket", None
    return "mako", None


def _reindex_heatmap_axis(
    data: pd.DataFrame,
    requested: Sequence[Any] | None,
    *,
    axis: Literal[0, 1],
    label: str,
) -> pd.DataFrame:
    if requested is None:
        return data
    available = data.index if axis == 0 else data.columns
    lookup = {str(value): value for value in available}
    requested_strings = [str(value) for value in requested]
    unknown = [value for value in requested_strings if value not in lookup]
    if unknown:
        raise ValueError(f"{label} contains values absent from the heatmap: {unknown}")
    resolved = [lookup[value] for value in requested_strings]
    return data.reindex(index=resolved) if axis == 0 else data.reindex(columns=resolved)


def _prepare_heatmap_data(data: pd.DataFrame, options: HeatmapOptions) -> pd.DataFrame:
    prepared = data.T.copy() if options.transpose else data.copy()
    prepared = _reindex_heatmap_axis(
        prepared, options.row_order, axis=0, label="row_order"
    )
    prepared = _reindex_heatmap_axis(
        prepared,
        options.column_order,
        axis=1,
        label="column_order",
    )
    if options.row_labels is not None:
        labels = {str(key): value for key, value in options.row_labels.items()}
        prepared.index = [labels.get(str(value), value) for value in prepared.index]
    if options.column_labels is not None:
        labels = {str(key): value for key, value in options.column_labels.items()}
        prepared.columns = [labels.get(str(value), value) for value in prepared.columns]
    return prepared


def _heatmap_figsize(
    data: pd.DataFrame, options: HeatmapOptions
) -> tuple[float, float]:
    if options.figsize is not None:
        return options.figsize
    width = max(
        options.min_figsize[0],
        min(options.max_figsize[0], options.cell_width * data.shape[1] + 2.5),
    )
    height = max(
        options.min_figsize[1],
        min(options.max_figsize[1], options.cell_height * data.shape[0] + 2.5),
    )
    return width, height


def _coerce_heatmap_options(
    value: HeatmapOptions | Mapping[str, Any] | None,
    *,
    base: HeatmapOptions | None = None,
) -> HeatmapOptions:
    starting = base or HeatmapOptions()
    if value is None:
        return starting
    if isinstance(value, HeatmapOptions):
        return value
    if isinstance(value, Mapping):
        updates = dict(value)
        for field_name in (
            "annot_kws",
            "cbar_kws",
            "tick_params_kws",
            "xlabel_kws",
            "ylabel_kws",
            "title_kws",
            "tight_layout_kws",
            "subplot_adjust_kws",
            "heatmap_kws",
            "clustermap_kws",
        ):
            if field_name in updates:
                updates[field_name] = dict(getattr(starting, field_name)) | dict(
                    updates[field_name]
                )
        return replace(starting, **updates)
    raise TypeError("Heatmap options must be HeatmapOptions, a field mapping, or None")


def _style_heatmap_axis(
    ax: plt.Axes,
    options: HeatmapOptions,
    *,
    colorbar_axis: plt.Axes | None = None,
) -> None:
    plt.setp(
        ax.get_xticklabels(),
        rotation=options.x_tick_rotation,
        ha="right" if options.x_tick_rotation else "center",
        fontsize=options.x_tick_fontsize or options.tick_fontsize,
    )
    plt.setp(
        ax.get_yticklabels(),
        rotation=options.y_tick_rotation,
        fontsize=options.y_tick_fontsize or options.tick_fontsize,
    )
    if options.tick_params_kws:
        ax.tick_params(**dict(options.tick_params_kws))
    if options.xlabel is not None:
        xlabel_kws = {"fontsize": options.xlabel_fontsize}
        xlabel_kws.update(options.xlabel_kws)
        ax.set_xlabel(options.xlabel, **xlabel_kws)
    if options.ylabel is not None:
        ylabel_kws = {"fontsize": options.ylabel_fontsize}
        ylabel_kws.update(options.ylabel_kws)
        ax.set_ylabel(options.ylabel, **ylabel_kws)
    if colorbar_axis is None and ax.collections:
        colorbar = getattr(ax.collections[0], "colorbar", None)
        colorbar_axis = colorbar.ax if colorbar is not None else None
    if colorbar_axis is not None:
        if options.cbar_tick_fontsize is not None:
            colorbar_axis.tick_params(labelsize=options.cbar_tick_fontsize)
        if options.cbar_label is not None:
            colorbar_axis.set_ylabel(
                options.cbar_label,
                fontsize=options.cbar_label_fontsize,
            )
    if options.rasterized:
        for collection in ax.collections:
            collection.set_rasterized(True)


def _plot_heatmap_variants(
    data: pd.DataFrame,
    output_stem: Path,
    *,
    default_title: str,
    default_cmap: Any,
    default_center: float | None,
    options: HeatmapOptions,
    save_options: FigureSaveOptions,
) -> dict[str, tuple[Path, ...]]:
    if data.empty:
        return {}
    prepared = _prepare_heatmap_data(data, options)
    title = options.title or default_title
    cmap = options.cmap if options.cmap is not None else default_cmap
    center = default_center if options.center == "auto" else options.center
    figsize = _heatmap_figsize(prepared, options)
    common_kws: dict[str, Any] = {
        "cmap": cmap,
        "center": center,
        "vmin": options.vmin,
        "vmax": options.vmax,
        "robust": options.robust,
        "annot": options.annot,
        "fmt": options.fmt,
        "annot_kws": dict(options.annot_kws),
        "linewidths": options.linewidths,
        "linecolor": options.linecolor,
        "cbar": options.cbar,
        "cbar_kws": dict(options.cbar_kws),
        "square": options.square,
        "xticklabels": options.xticklabels,
        "yticklabels": options.yticklabels,
    }
    if options.mask is not None:
        common_kws["mask"] = options.mask

    outputs: dict[str, tuple[Path, ...]] = {}
    if options.show_fixed:
        fig, ax = plt.subplots(figsize=figsize)
        fixed_kws = common_kws | dict(options.heatmap_kws)
        fixed_kws.pop("ax", None)
        sns.heatmap(prepared, ax=ax, **fixed_kws)
        _style_heatmap_axis(ax, options)
        if options.show_title:
            title_kws = {
                "fontsize": options.title_fontsize,
                "pad": options.title_pad,
            }
            title_kws.update(options.title_kws)
            ax.set_title(title, **title_kws)
        if options.tight_layout:
            fig.tight_layout(**dict(options.tight_layout_kws))
        if options.subplot_adjust_kws:
            fig.subplots_adjust(**dict(options.subplot_adjust_kws))
        outputs["fixed"] = _save_figure(fig, output_stem, options=save_options)
        plt.close(fig)

    can_cluster = prepared.shape[0] >= 2 and prepared.shape[1] >= 2
    if options.show_clustermap and can_cluster:
        clustered_kws = common_kws | {
            "figsize": figsize,
            "row_cluster": options.row_cluster,
            "col_cluster": options.col_cluster,
            "method": options.method,
            "metric": options.metric,
            "z_score": options.z_score,
            "standard_scale": options.standard_scale,
            "dendrogram_ratio": options.dendrogram_ratio,
            "colors_ratio": options.colors_ratio,
        }
        if options.cbar_pos is not None:
            clustered_kws["cbar_pos"] = options.cbar_pos
        clustered_kws.update(options.clustermap_kws)
        # Seaborn forwards heatmap kwargs to the auxiliary row/column-color
        # strips. Passing ``cbar`` there collides with the strip's forced
        # ``cbar=False`` on supported older releases. Clustermap colorbar
        # visibility is represented by ``cbar_pos`` instead.
        show_clustered_colorbar = bool(clustered_kws.pop("cbar", options.cbar))
        if not show_clustered_colorbar:
            clustered_kws["cbar_pos"] = None
        grid = sns.clustermap(prepared, **clustered_kws)
        _style_heatmap_axis(
            grid.ax_heatmap,
            options,
            colorbar_axis=getattr(grid, "cax", None),
        )
        cluster_modes = []
        if options.row_cluster:
            cluster_modes.append("rows")
        if options.col_cluster:
            cluster_modes.append("columns")
        suffix = " and ".join(cluster_modes) if cluster_modes else "fixed order"
        if options.show_title:
            title_kws = {
                "y": options.clustermap_title_y,
                "fontsize": options.title_fontsize,
            }
            title_kws.update(options.title_kws)
            grid.fig.suptitle(
                f"{title} — clustered {suffix}" if cluster_modes else title,
                **title_kws,
            )
        if options.tight_layout:
            grid.fig.tight_layout(**dict(options.tight_layout_kws))
        if options.subplot_adjust_kws:
            grid.fig.subplots_adjust(**dict(options.subplot_adjust_kws))
        outputs["clustermap"] = _save_figure(
            grid.fig,
            output_stem.with_name(f"{output_stem.name}_clustermap"),
            options=save_options,
        )
        plt.close(grid.fig)
    return outputs


def plot_hyperstac_cluster_features(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    heatmap_options: HeatmapOptions | Mapping[str, Any] | None = None,
    table_options: Mapping[str, HeatmapOptions | Mapping[str, Any]] | None = None,
    table_order: Sequence[str] | None = None,
    write_tables: bool = True,
    table_float_format: str | None = None,
    formats: Sequence[str] | None = None,
    dpi: int | None = None,
    save_options: FigureSaveOptions | None = None,
) -> dict[str, dict[str, tuple[Path, ...]]]:
    """Write configurable fixed and clustered variants for feature tables.

    ``heatmap_options`` defines the common style. ``table_options`` may contain
    a complete :class:`HeatmapOptions` or a small field-override mapping for a
    named table. The latter inherits the common style. Put raw
    :func:`seaborn.clustermap` arguments in ``clustermap_kws``; mapping-style
    per-table overrides merge those arguments with the common dictionary and
    are applied last. Existing calls retain the original automatic colors,
    sizes, fixed plot, and bi-clustermap.

    This function is a batch convenience. For exact control of one figure,
    select a DataFrame from ``hyperstac_cluster_feature_tables(...)`` and call
    :func:`seaborn.clustermap` directly.
    """

    output_root = Path(output_dir)
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    base_options = _coerce_heatmap_options(heatmap_options)
    per_table = dict(table_options or {})
    names = list(tables)
    unknown_options = sorted(set(per_table).difference(tables))
    if unknown_options:
        raise ValueError(f"table_options contains unknown tables: {unknown_options}")
    if table_order is not None:
        requested = [str(value) for value in table_order]
        unknown = [name for name in requested if name not in tables]
        if unknown:
            raise ValueError(f"table_order contains unknown tables: {unknown}")
        names = requested + [name for name in names if name not in requested]
    resolved_save = _resolve_save_options(save_options, formats=formats, dpi=dpi)
    outputs: dict[str, dict[str, tuple[Path, ...]]] = {}
    for name in names:
        table = tables[name]
        if write_tables:
            table.to_csv(table_dir / f"{name}.csv", float_format=table_float_format)
        cmap, center = _heatmap_style(name)
        title = name.replace("_", " ").capitalize()
        override = per_table.get(name)
        options = _coerce_heatmap_options(override, base=base_options)
        outputs[name] = _plot_heatmap_variants(
            table,
            figure_dir / name,
            default_title=title,
            default_cmap=cmap,
            default_center=center,
            options=options,
            save_options=resolved_save,
        )
    return outputs


def plot_hyperstac_umap(
    representation_adata: ad.AnnData,
    cluster_col: str,
    output_stem: str | Path,
    *,
    umap_key: str | None = None,
    cluster_order: Sequence[Any] | None = None,
    palette: str | Sequence[Any] | Mapping[Any, Any] | None = None,
    cluster_labels: Mapping[Any, str] | None = None,
    figsize: tuple[float, float] = (8.0, 7.0),
    point_size: float = 5.0,
    alpha: float = 0.8,
    marker: str = "o",
    linewidths: float = 0.0,
    edgecolors: Any = "none",
    rasterized: bool = True,
    shuffle_points: bool = False,
    random_state: int | None = 0,
    title: str | None = None,
    xlabel: str | None = "UMAP 1 (VICReg representation)",
    ylabel: str | None = "UMAP 2 (VICReg representation)",
    title_kws: Mapping[str, Any] | None = None,
    xlabel_kws: Mapping[str, Any] | None = None,
    ylabel_kws: Mapping[str, Any] | None = None,
    tick_params_kws: Mapping[str, Any] | None = None,
    show_legend: bool = True,
    legend_title: str | None = "Leiden environment",
    legend_kws: Mapping[str, Any] | None = None,
    background_color: Any | None = None,
    axis_off: bool = False,
    equal_aspect: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figure_kws: Mapping[str, Any] | None = None,
    scatter_kws: Mapping[str, Any] | None = None,
    scanpy_kws: Mapping[str, Any] | None = None,
    tight_layout_kws: Mapping[str, Any] | None = None,
    formats: Sequence[str] | None = None,
    dpi: int | None = None,
    save_options: FigureSaveOptions | None = None,
) -> tuple[Path, ...]:
    """Save a namespaced HyPERSTAC embedding using :func:`scanpy.pl.umap`.

    The wrapper only resolves the HyPERSTAC-specific UMAP key, constructs a
    lightweight plotting AnnData, applies stable category colors, and saves the
    requested formats. ``scanpy_kws`` is forwarded directly to
    :func:`scanpy.pl.umap` and applied after the compatibility defaults.
    ``scatter_kws`` is retained as an alias for older calls. For interactive or
    one-off figures, calling :func:`scanpy.pl.umap` directly is usually clearer.
    """

    import scanpy as sc

    _require_columns(
        representation_adata.obs, [cluster_col], "representation AnnData obs"
    )
    resolved_umap_key = umap_key or cluster_col_umap_key(
        representation_adata, cluster_col
    )
    if resolved_umap_key not in representation_adata.obsm:
        raise ValueError(f"No UMAP coordinates named {resolved_umap_key!r} were found")
    coords = np.asarray(representation_adata.obsm[resolved_umap_key])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"UMAP coordinates {resolved_umap_key!r} must have shape (n_obs, 2), "
            f"found {coords.shape}"
        )
    labels = representation_adata.obs[cluster_col].astype(str)
    mapping = _cluster_mapping(labels, cluster_order)
    color_lookup = _cluster_palette(mapping, palette)
    readable_labels = {
        str(key): str(value) for key, value in (cluster_labels or {}).items()
    }
    display_categories = [
        readable_labels.get(str(cluster), str(cluster))
        for cluster in mapping["cluster"]
    ]
    if len(set(display_categories)) != len(display_categories):
        raise ValueError(
            "cluster_labels must give every cluster a unique display label"
        )
    if point_size <= 0:
        raise ValueError("point_size must be positive")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")

    order = np.arange(len(labels))
    if shuffle_points:
        order = np.random.default_rng(random_state).permutation(order)
    display_lookup = {
        str(cluster): display
        for cluster, display in zip(mapping["cluster"], display_categories)
    }
    display_values = labels.map(display_lookup).iloc[order]
    plot_obs = pd.DataFrame(
        {
            "HyPERSTAC environment": pd.Categorical(
                display_values,
                categories=display_categories,
                ordered=True,
            )
        },
        index=representation_adata.obs_names[order].copy(),
    )
    plot_adata = ad.AnnData(obs=plot_obs)
    plot_adata.obsm["X_umap"] = coords[order].copy()

    figure_options = dict(figure_kws or {})
    figure_options.setdefault("figsize", figsize)
    fig, ax = plt.subplots(**figure_options)
    if background_color is not None:
        ax.set_facecolor(background_color)
    resolved_title = (
        title if title is not None else f"HyPERSTAC patch structure: {cluster_col}"
    )
    umap_options: dict[str, Any] = {
        "size": point_size,
        "alpha": alpha,
        "marker": marker,
        "linewidth": linewidths,
        "edgecolor": edgecolors,
        "legend_loc": "right margin" if show_legend else None,
        "palette": [color_lookup[str(cluster)] for cluster in mapping["cluster"]],
        "title": resolved_title,
        "frameon": not axis_off,
    }
    umap_options.update(scatter_kws or {})
    umap_options.update(scanpy_kws or {})
    # These define the adapter's data and lifecycle and cannot be overridden.
    umap_options.update(
        {
            "color": "HyPERSTAC environment",
            "ax": ax,
            "show": False,
        }
    )
    sc.pl.umap(plot_adata, **umap_options)
    if rasterized:
        for collection in ax.collections:
            collection.set_rasterized(True)
    if xlabel is not None:
        ax.set_xlabel(xlabel, **dict(xlabel_kws or {}))
    if ylabel is not None:
        ax.set_ylabel(ylabel, **dict(ylabel_kws or {}))
    if resolved_title and title_kws:
        ax.set_title(resolved_title, **dict(title_kws))
    if tick_params_kws:
        ax.tick_params(**dict(tick_params_kws))
    legend = ax.get_legend()
    if show_legend and legend is not None:
        if legend_title is not None:
            legend.set_title(legend_title)
        if legend_kws:
            handles, legend_labels = ax.get_legend_handles_labels()
            legend.remove()
            legend_options = {"title": legend_title}
            legend_options.update(legend_kws)
            ax.legend(handles, legend_labels, **legend_options)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if axis_off:
        ax.set_axis_off()
    fig.tight_layout(**dict(tight_layout_kws or {}))
    resolved_save = _resolve_save_options(save_options, formats=formats, dpi=dpi)
    paths = _save_figure(fig, output_stem, options=resolved_save)
    plt.close(fig)
    return paths


def _unique_roi_examples(
    obs: pd.DataFrame,
    patch_ids: pd.Index,
    scores: pd.Series,
    count: int,
    *,
    strategy: GallerySelectionStrategy,
    quantile: float,
    unique_rois: bool,
    rng: np.random.Generator,
) -> list[str]:
    patch_scores = scores.reindex(patch_ids)
    if strategy == "highest":
        ordered = patch_scores.sort_values(ascending=False, na_position="last").index
    elif strategy == "lowest":
        ordered = patch_scores.sort_values(ascending=True, na_position="last").index
    elif strategy == "median":
        ordered = (patch_scores - patch_scores.median()).abs().sort_values().index
    elif strategy == "quantile":
        target = patch_scores.quantile(quantile)
        ordered = (patch_scores - target).abs().sort_values().index
    elif strategy == "random":
        ordered = pd.Index(rng.permutation(np.asarray(patch_ids, dtype=object)))
    else:
        raise ValueError(
            "selection_strategy must be 'highest', 'lowest', 'median', 'quantile', or 'random'"
        )
    selected: list[str] = []
    seen_rois: set[str] = set()
    for patch_id in ordered:
        roi = str(obs.loc[patch_id, "roi"])
        if unique_rois and roi in seen_rois:
            continue
        selected.append(str(patch_id))
        seen_rois.add(roi)
        if len(selected) == count:
            return selected
    for patch_id in ordered:
        patch_id = str(patch_id)
        if patch_id not in selected:
            selected.append(patch_id)
        if len(selected) == count:
            break
    return selected


def _normalised_channel_path(root: Path, roi: str, channel: str) -> Path:
    roi_dir = root / roi
    candidates = {
        path.stem.casefold(): path
        for path in [*roi_dir.glob("*.tif"), *roi_dir.glob("*.tiff")]
    }
    path = candidates.get(str(channel).casefold())
    if path is None:
        raise FileNotFoundError(
            f"Normalized channel {channel!r} was not found for ROI {roi!r} in {roi_dir}"
        )
    return path


def _normalised_rgb_crop(
    root: Path,
    row: pd.Series,
    channels: Sequence[str],
    *,
    contrast_mode: GalleryContrastMode,
    contrast_percentiles: tuple[float, float],
    fixed_vmin: float | Mapping[str, float],
    fixed_vmax: float | Mapping[str, float],
    gamma: float,
    channel_weights: Sequence[float],
    invert_channels: set[str],
) -> np.ndarray:
    shape = (
        int(row["row_end"]) - int(row["row_start"]),
        int(row["col_end"]) - int(row["col_start"]),
    )
    rgb = np.zeros((*shape, 3), dtype=np.float32)
    for color_index, channel in enumerate(channels[:3]):
        path = _normalised_channel_path(root, str(row["roi"]), str(channel))
        image = tifffile.memmap(path)
        crop = np.asarray(
            image[
                int(row["row_start"]) : int(row["row_end"]),
                int(row["col_start"]) : int(row["col_end"]),
            ],
            dtype=np.float32,
        )
        if contrast_mode in {"per_patch", "per_roi"}:
            contrast_source = (
                crop if contrast_mode == "per_patch" else np.asarray(image)
            )
            low, high = np.nanpercentile(contrast_source, contrast_percentiles)
        elif contrast_mode == "fixed":
            low = (
                fixed_vmin.get(str(channel), 0.0)
                if isinstance(fixed_vmin, Mapping)
                else fixed_vmin
            )
            high = (
                fixed_vmax.get(str(channel), 1.0)
                if isinstance(fixed_vmax, Mapping)
                else fixed_vmax
            )
        elif contrast_mode == "none":
            low, high = 0.0, 1.0
        else:
            raise ValueError(
                "contrast_mode must be 'per_patch', 'per_roi', 'fixed', or 'none'"
            )
        if (
            contrast_mode != "none"
            and np.isfinite(low)
            and np.isfinite(high)
            and high > low
        ):
            crop = (crop - float(low)) / float(high - low)
        crop = np.clip(np.nan_to_num(crop, nan=0.0), 0.0, 1.0)
        if str(channel) in invert_channels:
            crop = 1.0 - crop
        if gamma != 1.0:
            crop = np.power(crop, gamma)
        rgb[:, :, color_index] = np.clip(crop * channel_weights[color_index], 0.0, 1.0)
    return rgb


def plot_hyperstac_environment_gallery(
    representation_adata: ad.AnnData,
    metrics_adata: ad.AnnData,
    cluster_col: str,
    normalised_image_dir: str | Path,
    output_stem: str | Path,
    *,
    permutation_adata: ad.AnnData | None = None,
    marker_source: GalleryMarkerSource = "zero_channel",
    marker_ranking: GalleryMarkerRanking = "relative_zscore",
    markers_by_cluster: Mapping[Any, Sequence[str]] | None = None,
    patch_ids_by_cluster: Mapping[Any, Sequence[str]] | None = None,
    examples_per_cluster: int = 4,
    channels_per_composite: int = 3,
    cluster_order: Sequence[Any] | None = None,
    selection_strategy: GallerySelectionStrategy = "highest",
    selection_quantile: float = 0.5,
    unique_rois: bool = True,
    random_state: int | None = 0,
    contrast_mode: GalleryContrastMode = "per_patch",
    contrast_lower_percentile: float = 0.0,
    contrast_percentile: float = 99.5,
    fixed_vmin: float | Mapping[str, float] = 0.0,
    fixed_vmax: float | Mapping[str, float] = 1.0,
    gamma: float = 1.0,
    channel_weights: Sequence[float] = (1.0, 1.0, 1.0),
    invert_channels: Sequence[str] = (),
    color_names: Sequence[str] = ("R", "G", "B"),
    figsize: tuple[float, float] | None = None,
    panel_width: float = 2.4,
    panel_height: float = 2.4,
    label_width: float = 1.55,
    title: str | None = None,
    show_title: bool = True,
    title_fontsize: float | None = 12.0,
    cluster_label_template: str = "Environment {cluster}",
    patch_title_template: str = "{roi}\n({col_start}, {row_start})",
    roi_title_template: str | None = None,
    show_cluster_labels: bool = True,
    show_marker_labels: bool = True,
    show_roi_titles: bool = True,
    show_patch_titles: bool | None = None,
    cluster_label_fontsize: float | None = 8.0,
    marker_label_fontsize: float | None = 8.0,
    roi_title_fontsize: float | None = 7.0,
    row_label_x: float = 0.95,
    row_label_kws: Mapping[str, Any] | None = None,
    cluster_label_kws: Mapping[str, Any] | None = None,
    marker_label_kws: Mapping[str, Any] | None = None,
    patch_title_kws: Mapping[str, Any] | None = None,
    roi_title_kws: Mapping[str, Any] | None = None,
    title_kws: Mapping[str, Any] | None = None,
    boxed_marker_key: bool = True,
    marker_key_x: float = 0.02,
    marker_key_y: float = 0.08,
    marker_key_width: float = 0.96,
    marker_key_height: float = 0.84,
    marker_key_facecolor: Any = "black",
    marker_key_edgecolor: Any = "black",
    marker_key_linewidth: float = 0.5,
    marker_key_kws: Mapping[str, Any] | None = None,
    cluster_label_color: Any = "white",
    marker_text_colors: Sequence[Any] = ("red", "lime", "dodgerblue"),
    marker_label_template: str = "{marker}",
    interpolation: str = "nearest",
    origin: str = "upper",
    image_kws: Mapping[str, Any] | None = None,
    panel_facecolor: Any | None = None,
    border_color: Any | None = "black",
    border_width: float = 0.5,
    axis_off: bool = True,
    figure_kws: Mapping[str, Any] | None = None,
    row_spacing: float | None = 0.12,
    column_spacing: float | None = 0.08,
    use_tight_layout: bool = True,
    tight_layout_kws: Mapping[str, Any] | None = None,
    subplot_adjust_kws: Mapping[str, Any] | None = None,
    write_tables: bool = True,
    formats: Sequence[str] | None = None,
    dpi: int | None = None,
    save_options: FigureSaveOptions | None = None,
) -> HyperstacGalleryResult:
    """Create one multi-row PNG/SVG gallery using normalized ROI TIFF crops.

    Each row represents one Leiden environment.  Up to three channels are
    rendered as red, green, and blue. Automatic marker and example selection
    can be replaced with explicit ``markers_by_cluster`` and
    ``patch_ids_by_cluster`` mappings. Contrast, gamma, ordering, annotations,
    panel geometry, and low-level Matplotlib image keywords are configurable.
    Patches have a black 0.5-point outline by default.  The left marker key is
    rendered in a dedicated grid column rather than outside the first patch;
    therefore ``column_spacing=0`` now removes the true inter-panel gap. The
    black box renders each marker in its RGB channel color. Set
    ``boxed_marker_key=False`` for a plain-text key.
    """

    if examples_per_cluster < 1:
        raise ValueError("examples_per_cluster must be at least one")
    if channels_per_composite < 1 or channels_per_composite > 3:
        raise ValueError("channels_per_composite must be between one and three")
    if not 0 <= contrast_lower_percentile < contrast_percentile <= 100:
        raise ValueError(
            "contrast percentiles must be increasing values within [0, 100]"
        )
    if not 0 <= selection_quantile <= 1:
        raise ValueError("selection_quantile must be between zero and one")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if len(channel_weights) < channels_per_composite:
        raise ValueError("channel_weights must cover every displayed channel")
    if len(color_names) < channels_per_composite:
        raise ValueError("color_names must cover every displayed channel")
    if len(marker_text_colors) < channels_per_composite:
        raise ValueError("marker_text_colors must cover every displayed channel")
    if panel_width <= 0 or panel_height <= 0 or label_width < 0:
        raise ValueError(
            "panel_width and panel_height must be positive; label_width cannot be negative"
        )
    if border_width < 0:
        raise ValueError("border_width cannot be negative")
    if marker_key_width <= 0 or marker_key_height <= 0:
        raise ValueError("marker_key_width and marker_key_height must be positive")
    if marker_key_linewidth < 0:
        raise ValueError("marker_key_linewidth cannot be negative")
    required = [cluster_col, "roi", "row_start", "row_end", "col_start", "col_end"]
    _require_columns(representation_adata.obs, required, "representation AnnData obs")

    intensity = _align_feature_frame(
        representation_adata,
        channel_mean_dataframe(metrics_adata),
        "metrics AnnData",
    )
    source_frame = intensity
    if marker_source != "intensity":
        if permutation_adata is None:
            raise ValueError(
                f"marker_source={marker_source!r} requires permutation_adata"
            )
        source_frame = permutation_channel_dataframe(permutation_adata, marker_source)
        source_frame = _align_feature_frame(
            representation_adata,
            source_frame,
            f"{marker_source} permutation AnnData",
        )
        shared = [
            channel for channel in intensity.columns if channel in source_frame.columns
        ]
        if not shared:
            raise ValueError(
                f"No normalized image channels overlap the {marker_source} scores"
            )
        source_frame = source_frame.loc[:, shared]

    labels = representation_adata.obs[cluster_col].astype(str)
    mapping = _cluster_mapping(labels, cluster_order)
    cluster_means = (
        source_frame.groupby(labels, observed=True).mean().reindex(mapping["cluster"])
    )
    relative = (
        (cluster_means - cluster_means.mean(axis=0))
        / cluster_means.std(axis=0).replace(0, np.nan)
    ).fillna(0.0)
    if marker_ranking == "relative_zscore":
        marker_scores = relative
    elif marker_ranking == "cluster_mean":
        marker_scores = cluster_means
    elif marker_ranking == "difference_from_rest":
        marker_scores = pd.DataFrame(
            index=cluster_means.index, columns=cluster_means.columns, dtype=float
        )
        for cluster in marker_scores.index.astype(str):
            marker_scores.loc[cluster] = source_frame.loc[labels == cluster].mean(
                axis=0
            ) - source_frame.loc[labels != cluster].mean(axis=0)
    else:
        raise ValueError(
            "marker_ranking must be 'relative_zscore', 'cluster_mean', or 'difference_from_rest'"
        )

    manual_markers = {
        str(cluster): [str(channel) for channel in channels]
        for cluster, channels in (markers_by_cluster or {}).items()
    }
    manual_patches = {
        str(cluster): [str(patch_id) for patch_id in patch_ids]
        for cluster, patch_ids in (patch_ids_by_cluster or {}).items()
    }
    unknown_marker_clusters = sorted(
        set(manual_markers).difference(mapping["cluster"].astype(str))
    )
    unknown_patch_clusters = sorted(
        set(manual_patches).difference(mapping["cluster"].astype(str))
    )
    if unknown_marker_clusters or unknown_patch_clusters:
        raise ValueError(
            "Manual gallery mappings contain unknown clusters: "
            f"markers={unknown_marker_clusters}, patches={unknown_patch_clusters}"
        )

    marker_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    selected_by_cluster: dict[str, list[str]] = {}
    resolved_markers: dict[str, list[str]] = {}
    rng = np.random.default_rng(random_state)
    for cluster in mapping["cluster"].astype(str):
        if cluster in manual_markers:
            markers = manual_markers[cluster][:channels_per_composite]
            if len(markers) != channels_per_composite:
                raise ValueError(
                    f"markers_by_cluster[{cluster!r}] must provide at least {channels_per_composite} channels"
                )
            unavailable = [
                channel for channel in markers if channel not in source_frame.columns
            ]
            if unavailable:
                raise ValueError(
                    f"Manual gallery markers are absent from normalized metrics: {unavailable}"
                )
        else:
            markers = (
                marker_scores.loc[cluster]
                .sort_values(ascending=False)
                .head(channels_per_composite)
                .index.astype(str)
                .tolist()
            )
        if len(markers) < channels_per_composite:
            for channel in intensity.columns:
                if channel not in markers:
                    markers.append(str(channel))
                if len(markers) == channels_per_composite:
                    break
        resolved_markers[cluster] = markers
        marker_rows.append(
            {
                "cluster": cluster,
                "marker_source": marker_source,
                "marker_ranking": marker_ranking,
                "red": markers[0] if len(markers) > 0 else None,
                "green": markers[1] if len(markers) > 1 else None,
                "blue": markers[2] if len(markers) > 2 else None,
                "manual_markers": cluster in manual_markers,
            }
        )
        patch_ids = representation_adata.obs_names[labels == cluster]
        scores = source_frame.loc[patch_ids, markers].mean(axis=1)
        if cluster in manual_patches:
            selected = manual_patches[cluster][:examples_per_cluster]
            unknown = [
                patch_id
                for patch_id in selected
                if patch_id not in representation_adata.obs_names
            ]
            wrong_cluster = [
                patch_id
                for patch_id in selected
                if patch_id in representation_adata.obs_names
                and str(labels.loc[patch_id]) != cluster
            ]
            if unknown or wrong_cluster:
                raise ValueError(
                    f"Invalid manual patches for cluster {cluster!r}: unknown={unknown}, wrong_cluster={wrong_cluster}"
                )
        else:
            selected = _unique_roi_examples(
                representation_adata.obs,
                patch_ids,
                scores,
                examples_per_cluster,
                strategy=selection_strategy,
                quantile=selection_quantile,
                unique_rois=unique_rois,
                rng=rng,
            )
        selected_by_cluster[cluster] = selected
        for rank, patch_id in enumerate(selected, start=1):
            row = representation_adata.obs.loc[patch_id]
            selections.append(
                {
                    "cluster": cluster,
                    "example_rank": rank,
                    "patch_id": patch_id,
                    "roi": str(row["roi"]),
                    "row_start": int(row["row_start"]),
                    "row_end": int(row["row_end"]),
                    "col_start": int(row["col_start"]),
                    "col_end": int(row["col_end"]),
                    "selection_score": float(scores.loc[patch_id]),
                    "selection_strategy": "manual"
                    if cluster in manual_patches
                    else selection_strategy,
                }
            )

    marker_table = pd.DataFrame(marker_rows)
    selection_table = pd.DataFrame(selections)
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    marker_table_path = output_stem.with_name(f"{output_stem.name}_markers.csv")
    selection_table_path = output_stem.with_name(f"{output_stem.name}_selections.csv")
    if write_tables:
        marker_table.to_csv(marker_table_path, index=False)
        selection_table.to_csv(selection_table_path, index=False)

    nrows = len(mapping)
    show_marker_key = show_cluster_labels or show_marker_labels
    if show_marker_key and label_width <= 0:
        raise ValueError(
            "label_width must be positive when cluster or marker labels are shown"
        )
    figure_options = dict(figure_kws or {})
    key_column_width = label_width if show_marker_key else 0.0
    figure_options.setdefault(
        "figsize",
        figsize
        or (
            panel_width * examples_per_cluster + key_column_width,
            panel_height * nrows,
        ),
    )
    n_grid_columns = examples_per_cluster + int(show_marker_key)
    gridspec_kws = dict(figure_options.pop("gridspec_kw", {}))
    gridspec_kws.setdefault(
        "width_ratios",
        ([label_width] if show_marker_key else [])
        + [panel_width] * examples_per_cluster,
    )
    fig, all_axes = plt.subplots(
        nrows,
        n_grid_columns,
        squeeze=False,
        gridspec_kw=gridspec_kws,
        **figure_options,
    )
    if show_marker_key:
        key_axes = all_axes[:, 0]
        axes = all_axes[:, 1:]
        for key_ax in key_axes:
            key_ax.set_axis_off()
    else:
        key_axes = np.full(nrows, None, dtype=object)
        axes = all_axes
    resolved_show_roi_titles = (
        show_roi_titles if show_patch_titles is None else show_patch_titles
    )
    resolved_roi_title_template = roi_title_template or patch_title_template
    root = Path(normalised_image_dir)
    for row_index, cluster in enumerate(mapping["cluster"].astype(str)):
        markers = resolved_markers[cluster]
        selected = selected_by_cluster[cluster]
        for col_index, patch_id in enumerate(selected):
            row = representation_adata.obs.loc[patch_id]
            rgb = _normalised_rgb_crop(
                root,
                row,
                markers,
                contrast_mode=contrast_mode,
                contrast_percentiles=(contrast_lower_percentile, contrast_percentile),
                fixed_vmin=fixed_vmin,
                fixed_vmax=fixed_vmax,
                gamma=gamma,
                channel_weights=channel_weights,
                invert_channels={str(channel) for channel in invert_channels},
            )
            ax = axes[row_index, col_index]
            if panel_facecolor is not None:
                ax.set_facecolor(panel_facecolor)
            resolved_image_kws = {"interpolation": interpolation, "origin": origin}
            resolved_image_kws.update(image_kws or {})
            ax.imshow(rgb, **resolved_image_kws)
            if resolved_show_roi_titles:
                patch_title = resolved_roi_title_template.format(
                    cluster=cluster,
                    patch_id=patch_id,
                    roi=row["roi"],
                    row_start=int(row["row_start"]),
                    row_end=int(row["row_end"]),
                    col_start=int(row["col_start"]),
                    col_end=int(row["col_end"]),
                    example_rank=col_index + 1,
                )
                resolved_roi_title_kws = {"fontsize": roi_title_fontsize}
                resolved_roi_title_kws.update(patch_title_kws or {})
                resolved_roi_title_kws.update(roi_title_kws or {})
                ax.set_title(patch_title, **resolved_roi_title_kws)
            if border_color is not None and border_width > 0:
                ax.add_patch(
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        transform=ax.transAxes,
                        fill=False,
                        clip_on=False,
                        edgecolor=border_color,
                        linewidth=border_width,
                    )
                )
            if axis_off:
                ax.set_axis_off()
        for col_index in range(len(selected), examples_per_cluster):
            axes[row_index, col_index].set_visible(False)
        row_text_parts = []
        cluster_text = cluster_label_template.format(cluster=cluster)
        if show_cluster_labels:
            row_text_parts.append(cluster_text)
        marker_texts = [
            marker_label_template.format(
                cluster=cluster,
                marker=marker,
                color_name=color_names[index],
            )
            for index, marker in enumerate(markers)
        ]
        if show_marker_labels:
            row_text_parts.extend(marker_texts)
        if row_text_parts and boxed_marker_key:
            key_kws = {
                "facecolor": marker_key_facecolor,
                "edgecolor": marker_key_edgecolor,
                "linewidth": marker_key_linewidth,
                "clip_on": False,
                "zorder": 5,
            }
            key_kws.update(marker_key_kws or {})
            key_ax = key_axes[row_index]
            key_ax.add_patch(
                Rectangle(
                    (marker_key_x, marker_key_y),
                    marker_key_width,
                    marker_key_height,
                    transform=key_ax.transAxes,
                    **key_kws,
                )
            )
            line_y = np.linspace(
                marker_key_y + marker_key_height - 0.12,
                marker_key_y + 0.12,
                len(row_text_parts),
            )
            text_index = 0
            if show_cluster_labels:
                cluster_kws = {
                    "ha": "left",
                    "va": "center",
                    "fontsize": cluster_label_fontsize,
                    "fontweight": "bold",
                    "color": cluster_label_color,
                    "clip_on": False,
                    "zorder": 6,
                }
                cluster_kws.update(cluster_label_kws or {})
                key_ax.text(
                    marker_key_x + 0.06,
                    line_y[text_index],
                    cluster_text,
                    transform=key_ax.transAxes,
                    **cluster_kws,
                )
                text_index += 1
            if show_marker_labels:
                for marker_index, marker_text in enumerate(marker_texts):
                    marker_kws = {
                        "ha": "left",
                        "va": "center",
                        "fontsize": marker_label_fontsize,
                        "color": marker_text_colors[marker_index],
                        "clip_on": False,
                        "zorder": 6,
                    }
                    marker_kws.update(marker_label_kws or {})
                    key_ax.text(
                        marker_key_x + 0.06,
                        line_y[text_index],
                        marker_text,
                        transform=key_ax.transAxes,
                        **marker_kws,
                    )
                    text_index += 1
        elif row_text_parts:
            text_kws = {
                "ha": "right",
                "va": "center",
                "fontsize": cluster_label_fontsize,
            }
            text_kws.update(row_label_kws or {})
            key_ax = key_axes[row_index]
            key_ax.text(
                row_label_x,
                0.5,
                "\n".join(row_text_parts),
                transform=key_ax.transAxes,
                **text_kws,
            )
    if show_title:
        resolved_title = title or (
            "Representative HyPERSTAC environments — markers selected by "
            f"{marker_source.replace('_', ' ')}"
        )
        suptitle_kws = {"y": 1.005, "fontsize": title_fontsize}
        suptitle_kws.update(title_kws or {})
        fig.suptitle(resolved_title, **suptitle_kws)
    _apply_subplot_layout(
        fig,
        use_tight_layout=use_tight_layout,
        tight_layout_kws=tight_layout_kws,
        row_spacing=row_spacing,
        column_spacing=None,
        subplot_adjust_kws=subplot_adjust_kws,
    )
    _pack_gallery_columns(
        axes,
        key_axes,
        column_spacing=column_spacing,
        label_width=label_width,
        panel_width=panel_width,
    )
    resolved_save = _resolve_save_options(save_options, formats=formats, dpi=dpi)
    image_paths = _save_figure(fig, output_stem, options=resolved_save)
    plt.close(fig)
    return HyperstacGalleryResult(
        image_paths=image_paths,
        marker_table_path=marker_table_path,
        selection_table_path=selection_table_path,
        marker_table=marker_table,
        selection_table=selection_table,
    )


def summarize_environment_abundance(
    adata_or_obs: ad.AnnData | pd.DataFrame,
    cluster_col: str,
    *,
    roi_col: str = "roi",
    metadata: pd.DataFrame | None = None,
    metadata_roi_col: str = "ROI_number",
) -> pd.DataFrame:
    """Return complete ROI-by-environment patch counts and fractions."""

    obs = _obs_frame(adata_or_obs)
    _require_columns(obs, [roi_col, cluster_col], "HyPERSTAC patch observations")
    rois = sorted(obs[roi_col].dropna().astype(str).unique(), key=natural_key)
    environments = sorted(
        obs[cluster_col].dropna().astype(str).unique(), key=natural_key
    )
    complete_index = pd.MultiIndex.from_product(
        [rois, environments], names=["roi", "environment"]
    )
    counts = (
        obs.assign(
            _roi=obs[roi_col].astype(str), _environment=obs[cluster_col].astype(str)
        )
        .groupby(["_roi", "_environment"], observed=True)
        .size()
        .reindex(complete_index, fill_value=0)
        .rename("patch_count")
        .reset_index()
    )
    counts["total_patches"] = counts.groupby("roi")["patch_count"].transform("sum")
    counts["fraction"] = counts["patch_count"] / counts["total_patches"]

    if metadata is not None:
        _require_columns(metadata, [metadata_roi_col], "sample metadata")
        sample_metadata = metadata.loc[metadata[metadata_roi_col].notna()].copy()
        sample_metadata[metadata_roi_col] = sample_metadata[metadata_roi_col].astype(
            str
        )
        duplicates = sample_metadata[metadata_roi_col][
            sample_metadata[metadata_roi_col].duplicated()
        ].unique()
        if len(duplicates):
            raise ValueError(
                f"Sample metadata contains duplicate ROI identifiers: {duplicates.tolist()}"
            )
        counts = counts.merge(
            sample_metadata,
            how="left",
            left_on="roi",
            right_on=metadata_roi_col,
            validate="many_to_one",
        )
    return counts


def aggregate_environment_abundance(
    abundance: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """Pool ROI patch counts within a sample-level group such as ``Case``."""

    _require_columns(
        abundance, [group_col, "environment", "patch_count"], "abundance table"
    )
    valid = abundance.loc[abundance[group_col].notna()].copy()
    grouped = (
        valid.groupby([group_col, "environment"], observed=True)["patch_count"]
        .sum()
        .rename("patch_count")
        .reset_index()
    )
    grouped["total_patches"] = grouped.groupby(group_col)["patch_count"].transform(
        "sum"
    )
    grouped["fraction"] = grouped["patch_count"] / grouped["total_patches"]
    return grouped


def prepare_environment_abundance_tables(
    abundance: pd.DataFrame,
    *,
    sample_group_col: str | None = "Case",
    categorical_metadata: Sequence[str] = (),
    numeric_metadata: Sequence[str] = (),
    environment_order: Sequence[Any] | None = None,
    categorical_statistic: Literal["mean", "median"] = "mean",
    correlation_method: CorrelationMethod = "spearman",
    min_numeric_observations: int = 3,
) -> EnvironmentAbundanceTables:
    """Prepare every abundance matrix without choosing a plotting library.

    This is the preferred interface when a figure needs exact Seaborn or
    Matplotlib control.  It retains the domain-specific work—complete ROI
    fractions, pooled sample fractions, categorical summaries, and numeric
    correlations—while leaving rendering entirely to the caller.
    """

    _require_columns(
        abundance, ["roi", "environment", "patch_count", "fraction"], "abundance table"
    )
    if min_numeric_observations < 3:
        raise ValueError("min_numeric_observations must be at least three")
    if categorical_statistic not in {"mean", "median"}:
        raise ValueError("categorical_statistic must be 'mean' or 'median'")
    if correlation_method not in {"spearman", "pearson"}:
        raise ValueError("correlation_method must be 'spearman' or 'pearson'")

    available_environments = abundance["environment"].dropna().astype(str).unique()
    resolved_order = _ordered_values(
        available_environments,
        environment_order,
        label="environment_order",
    )
    overall = abundance.groupby("environment", observed=True)["patch_count"].sum()
    overall_lookup = {str(index): index for index in overall.index}
    overall = overall.reindex([overall_lookup[value] for value in resolved_order])
    overall_fraction = overall / overall.sum()

    roi_matrix = abundance.pivot(
        index="roi", columns="environment", values="fraction"
    ).fillna(0.0)
    roi_matrix = _reindex_heatmap_axis(
        roi_matrix,
        resolved_order,
        axis=1,
        label="environment_order",
    )

    sample_matrix: pd.DataFrame | None = None
    if sample_group_col is not None and sample_group_col in abundance.columns:
        grouped = aggregate_environment_abundance(abundance, sample_group_col)
        sample_matrix = grouped.pivot(
            index=sample_group_col, columns="environment", values="fraction"
        ).fillna(0.0)
        sample_matrix = _reindex_heatmap_axis(
            sample_matrix,
            resolved_order,
            axis=1,
            label="environment_order",
        )

    categorical_tables: dict[str, pd.DataFrame] = {}
    for column in categorical_metadata:
        if column not in abundance.columns:
            raise ValueError(
                f"Categorical metadata column {column!r} is absent from the abundance table"
            )
        valid = abundance.loc[abundance[column].notna()]
        grouped_values = valid.groupby([column, "environment"], observed=True)[
            "fraction"
        ]
        table = (
            grouped_values.mean()
            if categorical_statistic == "mean"
            else grouped_values.median()
        ).unstack(fill_value=0.0)
        if len(table) < 2:
            continue
        categorical_tables[column] = _reindex_heatmap_axis(
            table,
            resolved_order,
            axis=1,
            label="environment_order",
        )

    correlations: pd.DataFrame | None = None
    pvalues: pd.DataFrame | None = None
    if numeric_metadata:
        roi_metadata = abundance.drop_duplicates("roi").set_index("roi")
        correlations = pd.DataFrame(
            index=roi_matrix.columns, columns=list(numeric_metadata), dtype=float
        )
        pvalues = correlations.copy()
        for column in numeric_metadata:
            if column not in roi_metadata.columns:
                raise ValueError(
                    f"Numeric metadata column {column!r} is absent from the abundance table"
                )
            values = pd.to_numeric(roi_metadata[column], errors="coerce").reindex(
                roi_matrix.index
            )
            for environment in roi_matrix.columns:
                valid = values.notna() & roi_matrix[environment].notna()
                if (
                    valid.sum() >= min_numeric_observations
                    and values[valid].nunique() >= 2
                ):
                    correlation_function = (
                        spearmanr if correlation_method == "spearman" else pearsonr
                    )
                    statistic, pvalue = correlation_function(
                        values[valid], roi_matrix.loc[valid, environment]
                    )
                    correlations.loc[environment, column] = statistic
                    pvalues.loc[environment, column] = pvalue

    return EnvironmentAbundanceTables(
        overall_fraction=overall_fraction,
        roi_fraction=roi_matrix,
        sample_group_fraction=sample_matrix,
        categorical_fraction=categorical_tables,
        numeric_correlations=correlations,
        numeric_pvalues=pvalues,
        environment_order=tuple(resolved_order),
    )


def plot_environment_abundance(
    abundance: pd.DataFrame,
    output_dir: str | Path,
    *,
    sample_group_col: str | None = "Case",
    categorical_metadata: Sequence[str] = (),
    numeric_metadata: Sequence[str] = (),
    environment_order: Sequence[Any] | None = None,
    environment_labels: Mapping[Any, str] | None = None,
    overall_sort: Literal[
        "natural", "abundance_ascending", "abundance_descending"
    ] = "natural",
    overall_color: Any = "#4c78a8",
    overall_palette: str | Sequence[Any] | Mapping[Any, Any] | None = None,
    overall_figsize: tuple[float, float] | None = None,
    overall_title: str = "Overall HyPERSTAC environment abundance",
    show_overall_title: bool = True,
    overall_title_fontsize: float | None = 12.0,
    overall_xlabel: str = "HyPERSTAC environment",
    overall_ylabel: str = "Fraction of accepted patches",
    overall_axis_label_fontsize: float | None = 10.0,
    overall_tick_fontsize: float | None = 9.0,
    overall_tick_rotation: float = 0.0,
    show_bar_values: bool = False,
    bar_value_format: str = ".1%",
    bar_value_fontsize: float | None = 8.0,
    bar_value_kws: Mapping[str, Any] | None = None,
    bar_kws: Mapping[str, Any] | None = None,
    overall_axis_border_color: Any | None = "black",
    overall_axis_border_width: float = 0.8,
    overall_grid_axis: Literal["x", "y", "both"] | None = None,
    overall_grid_kws: Mapping[str, Any] | None = None,
    overall_title_kws: Mapping[str, Any] | None = None,
    overall_xlabel_kws: Mapping[str, Any] | None = None,
    overall_ylabel_kws: Mapping[str, Any] | None = None,
    overall_tick_params_kws: Mapping[str, Any] | None = None,
    roi_heatmap_options: HeatmapOptions | Mapping[str, Any] | None = None,
    sample_heatmap_options: HeatmapOptions | Mapping[str, Any] | None = None,
    categorical_heatmap_options: HeatmapOptions | Mapping[str, Any] | None = None,
    categorical_table_options: Mapping[str, HeatmapOptions | Mapping[str, Any]]
    | None = None,
    numeric_heatmap_options: HeatmapOptions | Mapping[str, Any] | None = None,
    categorical_statistic: Literal["mean", "median"] = "mean",
    correlation_method: CorrelationMethod = "spearman",
    min_numeric_observations: int = 3,
    plot_overall: bool = True,
    plot_roi: bool = True,
    plot_sample_group: bool = True,
    plot_categorical: bool = True,
    plot_numeric: bool = True,
    write_tables: bool = True,
    figure_kws: Mapping[str, Any] | None = None,
    tight_layout_kws: Mapping[str, Any] | None = None,
    formats: Sequence[str] | None = None,
    dpi: int | None = None,
    save_options: FigureSaveOptions | None = None,
) -> dict[str, Any]:
    """Plot overall, ROI, sample-group, and metadata-associated abundance.

    Every panel can be enabled independently. Heatmap arguments accept either
    :class:`HeatmapOptions` or a compact field mapping. Use
    ``categorical_table_options`` for per-metadata-field overrides. For exact
    one-off figures, prefer :func:`prepare_environment_abundance_tables` and
    call Seaborn directly; this function is intentionally retained as a batch
    exploratory convenience wrapper.
    """

    output_root = Path(output_dir)
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if overall_axis_border_width < 0:
        raise ValueError("overall_axis_border_width cannot be negative")
    if overall_grid_axis not in {None, "x", "y", "both"}:
        raise ValueError("overall_grid_axis must be 'x', 'y', 'both', or None")
    prepared = prepare_environment_abundance_tables(
        abundance,
        sample_group_col=sample_group_col if plot_sample_group else None,
        categorical_metadata=categorical_metadata if plot_categorical else (),
        numeric_metadata=numeric_metadata if plot_numeric else (),
        environment_order=environment_order,
        categorical_statistic=categorical_statistic,
        correlation_method=correlation_method,
        min_numeric_observations=min_numeric_observations,
    )
    if write_tables:
        abundance.to_csv(table_dir / "roi_environment_abundance.csv", index=False)
    outputs: dict[str, Any] = {}
    resolved_save = _resolve_save_options(save_options, formats=formats, dpi=dpi)
    readable_labels = {
        str(key): str(value) for key, value in (environment_labels or {}).items()
    }

    overall_fraction = prepared.overall_fraction.copy()
    if overall_sort == "abundance_ascending":
        overall_fraction = overall_fraction.sort_values(ascending=True)
    elif overall_sort == "abundance_descending":
        overall_fraction = overall_fraction.sort_values(ascending=False)
    elif overall_sort != "natural":
        raise ValueError(
            "overall_sort must be 'natural', 'abundance_ascending', or 'abundance_descending'"
        )
    if write_tables:
        overall_fraction.rename("fraction").to_csv(
            table_dir / "overall_environment_abundance.csv"
        )
    if plot_overall:
        environment_keys = overall_fraction.index.astype(str).tolist()
        if isinstance(overall_palette, Mapping):
            palette_lookup = {str(key): value for key, value in overall_palette.items()}
            missing = [key for key in environment_keys if key not in palette_lookup]
            if missing:
                raise ValueError(
                    f"overall_palette is missing environment colors for: {missing}"
                )
            colors: Any = [palette_lookup[key] for key in environment_keys]
        elif isinstance(overall_palette, str):
            colors = sns.color_palette(overall_palette, n_colors=len(environment_keys))
        elif overall_palette is not None:
            colors = list(overall_palette)
            if len(colors) < len(environment_keys):
                raise ValueError("overall_palette does not contain enough colors")
        else:
            colors = overall_color
        figure_options = dict(figure_kws or {})
        figure_options.setdefault(
            "figsize",
            overall_figsize or (max(6, 0.7 * len(overall_fraction)), 4.5),
        )
        fig, ax = plt.subplots(**figure_options)
        resolved_bar_kws = {"color": colors}
        resolved_bar_kws.update(bar_kws or {})
        bars = ax.bar(
            [readable_labels.get(key, key) for key in environment_keys],
            overall_fraction.to_numpy(),
            **resolved_bar_kws,
        )
        xlabel_kws = {"fontsize": overall_axis_label_fontsize}
        xlabel_kws.update(overall_xlabel_kws or {})
        ylabel_kws = {"fontsize": overall_axis_label_fontsize}
        ylabel_kws.update(overall_ylabel_kws or {})
        ax.set_xlabel(overall_xlabel, **xlabel_kws)
        ax.set_ylabel(overall_ylabel, **ylabel_kws)
        if show_overall_title:
            title_options = {"fontsize": overall_title_fontsize}
            title_options.update(overall_title_kws or {})
            ax.set_title(overall_title, **title_options)
        tick_options = {
            "axis": "both",
            "labelsize": overall_tick_fontsize,
        }
        tick_options.update(overall_tick_params_kws or {})
        ax.tick_params(**tick_options)
        plt.setp(
            ax.get_xticklabels(),
            rotation=overall_tick_rotation,
            ha="right" if overall_tick_rotation else "center",
        )
        if overall_grid_axis is not None:
            grid_options = {"alpha": 0.25, "linewidth": 0.5}
            grid_options.update(overall_grid_kws or {})
            ax.grid(axis=overall_grid_axis, **grid_options)
            ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(overall_axis_border_color is not None)
            if overall_axis_border_color is not None:
                spine.set_color(overall_axis_border_color)
                spine.set_linewidth(overall_axis_border_width)
        if show_bar_values:
            annotation_kws = {"padding": 3, "fontsize": bar_value_fontsize}
            annotation_kws.update(bar_value_kws or {})
            ax.bar_label(
                bars,
                labels=[format(value, bar_value_format) for value in overall_fraction],
                **annotation_kws,
            )
        fig.tight_layout(**dict(tight_layout_kws or {}))
        outputs["overall"] = _save_figure(
            fig,
            figure_dir / "overall_environment_abundance",
            options=resolved_save,
        )
        plt.close(fig)

    roi_matrix = prepared.roi_fraction
    if write_tables:
        roi_matrix.to_csv(table_dir / "roi_environment_fraction_matrix.csv")
    if plot_roi:
        roi_options = _coerce_heatmap_options(roi_heatmap_options)
        if roi_options.column_labels is None and readable_labels:
            roi_options = replace(roi_options, column_labels=readable_labels)
        outputs["roi"] = _plot_heatmap_variants(
            roi_matrix,
            figure_dir / "roi_environment_fraction",
            default_title="Environment fractions by ROI",
            default_cmap="mako",
            default_center=None,
            options=roi_options,
            save_options=resolved_save,
        )

    if plot_sample_group and prepared.sample_group_fraction is not None:
        if sample_group_col is None:
            raise AssertionError("A prepared sample matrix requires sample_group_col")
        if write_tables:
            grouped = aggregate_environment_abundance(abundance, sample_group_col)
            grouped.to_csv(
                table_dir
                / f"{safe_filename(sample_group_col)}_environment_abundance.csv",
                index=False,
            )
        group_matrix = prepared.sample_group_fraction
        group_options = _coerce_heatmap_options(sample_heatmap_options)
        if group_options.column_labels is None and readable_labels:
            group_options = replace(group_options, column_labels=readable_labels)
        outputs["sample_group"] = _plot_heatmap_variants(
            group_matrix,
            figure_dir / f"{safe_filename(sample_group_col)}_environment_fraction",
            default_title=f"Environment fractions by {sample_group_col}",
            default_cmap="mako",
            default_center=None,
            options=group_options,
            save_options=resolved_save,
        )

    categorical_outputs: dict[str, Any] = {}
    per_categorical_table = dict(categorical_table_options or {})
    unknown_categorical_options = sorted(
        set(per_categorical_table).difference(categorical_metadata)
    )
    if unknown_categorical_options:
        raise ValueError(
            "categorical_table_options contains fields absent from "
            f"categorical_metadata: {unknown_categorical_options}"
        )
    for column, table in prepared.categorical_fraction.items():
        if write_tables:
            table.to_csv(
                table_dir
                / f"categorical_{safe_filename(column)}_{categorical_statistic}_fraction.csv"
            )
        category_base_options = _coerce_heatmap_options(categorical_heatmap_options)
        category_options = _coerce_heatmap_options(
            per_categorical_table.get(column),
            base=category_base_options,
        )
        if category_options.column_labels is None and readable_labels:
            category_options = replace(category_options, column_labels=readable_labels)
        categorical_outputs[column] = _plot_heatmap_variants(
            table,
            figure_dir
            / f"categorical_{safe_filename(column)}_{categorical_statistic}_fraction",
            default_title=f"{categorical_statistic.title()} ROI environment fraction by {column}",
            default_cmap="mako",
            default_center=None,
            options=category_options,
            save_options=resolved_save,
        )
    outputs["categorical_metadata"] = categorical_outputs

    if plot_numeric and prepared.numeric_correlations is not None:
        correlations = prepared.numeric_correlations
        pvalues = prepared.numeric_pvalues
        if pvalues is None:
            raise AssertionError(
                "Numeric correlations require a matching p-value table"
            )
        if write_tables:
            correlations.to_csv(
                table_dir / f"numeric_metadata_{correlation_method}_correlations.csv"
            )
            pvalues.to_csv(
                table_dir / f"numeric_metadata_{correlation_method}_pvalues.csv"
            )
        numeric_options = _coerce_heatmap_options(numeric_heatmap_options)
        if numeric_options.row_labels is None and readable_labels:
            numeric_options = replace(numeric_options, row_labels=readable_labels)
        outputs["numeric_metadata"] = _plot_heatmap_variants(
            correlations,
            figure_dir / f"numeric_metadata_{correlation_method}_correlations",
            default_title=(
                f"{correlation_method.title()} association: ROI environment fraction vs sample metadata"
            ),
            default_cmap="vlag",
            default_center=0.0,
            options=numeric_options,
            save_options=resolved_save,
        )
    return outputs


def _mapping_dictionary(mapping: pd.DataFrame | Mapping[Any, Any]) -> dict[int, str]:
    if isinstance(mapping, pd.DataFrame):
        _require_columns(mapping, ["mask_value", "cluster"], "cluster label mapping")
        return dict(
            zip(mapping["mask_value"].astype(int), mapping["cluster"].astype(str))
        )
    return {int(key): str(value) for key, value in mapping.items()}


def assign_cells_to_hyperstac_masks(
    adata_or_obs: ad.AnnData | pd.DataFrame,
    mask_folder: str | Path,
    mapping: pd.DataFrame | Mapping[Any, Any],
    *,
    rois: Sequence[str] | None = None,
    x_col: str = "X_loc",
    y_col: str = "Y_loc",
    pop_col: str = "population",
    roi_col: str = "ROI",
    mask_pattern: str = "{roi}__hyperstac_labels.tiff",
) -> pd.DataFrame:
    """Assign cell centres to reconstructed HyPERSTAC environments.

    The result retains all selected cells and gives each an auditable status:
    ``assigned``, ``uncovered``, ``out_of_bounds``, ``nonfinite_coordinate``,
    or ``unknown_mask_value``.
    """

    obs = _obs_frame(adata_or_obs)
    _require_columns(obs, [roi_col, x_col, y_col, pop_col], "cell observations")
    if rois is None:
        roi_keys = obs[roi_col].dropna().astype(str).drop_duplicates().tolist()
    else:
        roi_keys = [str(roi) for roi in rois]
    paths = resolve_spatial_mask_paths(mask_folder, roi_keys, mask_pattern=mask_pattern)
    value_to_cluster = _mapping_dictionary(mapping)

    selected = obs.loc[
        obs[roi_col].astype(str).isin(roi_keys), [roi_col, x_col, y_col, pop_col]
    ].copy()
    result = pd.DataFrame(index=selected.index)
    result["roi"] = selected[roi_col].astype(str)
    result["population"] = selected[pop_col]
    result["mask_value"] = pd.array([pd.NA] * len(selected), dtype="Int64")
    result["environment"] = pd.Series(pd.NA, index=selected.index, dtype="string")
    result["status"] = "nonfinite_coordinate"

    for roi in roi_keys:
        positions = np.flatnonzero((result["roi"] == roi).to_numpy())
        if not len(positions):
            continue
        frame = selected.iloc[positions]
        coordinates = frame[[x_col, y_col]].to_numpy(dtype=float, copy=True)
        finite = np.isfinite(coordinates).all(axis=1)
        result.iloc[positions[finite], result.columns.get_loc("status")] = (
            "out_of_bounds"
        )
        finite_positions = positions[finite]
        x = coordinates[finite, 0].astype(np.int64)
        y = coordinates[finite, 1].astype(np.int64)
        mask = tifffile.imread(paths[roi])
        in_bounds = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
        bounded_positions = finite_positions[in_bounds]
        values = mask[y[in_bounds], x[in_bounds]].astype(int)
        result.iloc[bounded_positions, result.columns.get_loc("mask_value")] = values
        mapped = pd.Series(values).map(value_to_cluster)
        statuses = np.where(
            values == 0,
            "uncovered",
            np.where(mapped.isna(), "unknown_mask_value", "assigned"),
        )
        result.iloc[bounded_positions, result.columns.get_loc("status")] = statuses
        result.iloc[bounded_positions, result.columns.get_loc("environment")] = (
            mapped.to_numpy()
        )
    return result


def summarize_cell_environment_composition(assignments: pd.DataFrame) -> pd.DataFrame:
    """Summarize assigned cell populations within each HyPERSTAC environment."""

    _require_columns(
        assignments, ["environment", "population", "status"], "cell assignments"
    )
    valid = assignments.loc[
        (assignments["status"] == "assigned")
        & assignments["environment"].notna()
        & assignments["population"].notna()
    ]
    if valid.empty:
        raise ValueError("No assigned cells with population labels were found")
    environments = sorted(valid["environment"].astype(str).unique(), key=natural_key)
    populations = sorted(valid["population"].astype(str).unique(), key=natural_key)
    complete_index = pd.MultiIndex.from_product(
        [environments, populations], names=["environment", "population"]
    )
    counts = (
        valid.assign(
            _environment=valid["environment"].astype(str),
            _population=valid["population"].astype(str),
        )
        .groupby(["_environment", "_population"], observed=True)
        .size()
        .reindex(complete_index, fill_value=0)
        .rename("cell_count")
        .reset_index()
    )
    counts["fraction_within_environment"] = counts["cell_count"] / counts.groupby(
        "environment"
    )["cell_count"].transform("sum")
    counts["fraction_within_population"] = counts["cell_count"] / counts.groupby(
        "population"
    )["cell_count"].transform("sum")
    return counts


def plot_cell_environment_composition(
    composition: pd.DataFrame,
    output_dir: str | Path,
    *,
    metrics: Sequence[str] = (
        "fraction_within_environment",
        "fraction_within_population",
    ),
    environment_order: Sequence[Any] | None = None,
    population_order: Sequence[Any] | None = None,
    environment_labels: Mapping[Any, str] | None = None,
    population_labels: Mapping[Any, str] | None = None,
    heatmap_options: HeatmapOptions | Mapping[str, Any] | None = None,
    metric_options: Mapping[str, HeatmapOptions | Mapping[str, Any]] | None = None,
    metric_titles: Mapping[str, str] | None = None,
    metric_cmaps: Mapping[str, Any] | None = None,
    write_tables: bool = True,
    table_float_format: str | None = None,
    formats: Sequence[str] | None = None,
    dpi: int | None = None,
    save_options: FigureSaveOptions | None = None,
) -> dict[str, dict[str, tuple[Path, ...]]]:
    """Plot customizable cell-population/environment composition heatmaps.

    ``metrics`` can include ``cell_count``, ``fraction_within_environment``,
    or ``fraction_within_population``. Ordering and display labels are
    independent of the underlying CSV values. Per-metric options inherit the
    common ``heatmap_options`` when supplied as field mappings.
    """

    _require_columns(
        composition,
        [
            "environment",
            "population",
            "cell_count",
            "fraction_within_environment",
            "fraction_within_population",
        ],
        "cell-environment composition",
    )
    output_root = Path(output_dir)
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if write_tables:
        composition.to_csv(
            table_dir / "cell_environment_composition.csv",
            index=False,
            float_format=table_float_format,
        )
    allowed_metrics = {
        "cell_count",
        "fraction_within_environment",
        "fraction_within_population",
    }
    unknown_metrics = [metric for metric in metrics if metric not in allowed_metrics]
    if unknown_metrics:
        raise ValueError(f"Unknown cell-composition metrics: {unknown_metrics}")
    environments = _ordered_values(
        composition["environment"].dropna().astype(str).unique(),
        environment_order,
        label="environment_order",
    )
    populations = _ordered_values(
        composition["population"].dropna().astype(str).unique(),
        population_order,
        label="population_order",
    )
    base_options = _coerce_heatmap_options(heatmap_options)
    per_metric = dict(metric_options or {})
    default_titles = {
        "cell_count": "Assigned cell counts by environment and population",
        "fraction_within_environment": "Cell-population composition within each environment",
        "fraction_within_population": "Distribution of each population across environments",
    }
    default_cmaps = {
        "cell_count": "viridis",
        "fraction_within_environment": "mako",
        "fraction_within_population": "rocket",
    }
    resolved_titles = default_titles | dict(metric_titles or {})
    resolved_cmaps = default_cmaps | dict(metric_cmaps or {})
    resolved_save = _resolve_save_options(save_options, formats=formats, dpi=dpi)
    outputs: dict[str, dict[str, tuple[Path, ...]]] = {}
    for value_col in metrics:
        matrix = composition.pivot(
            index="environment", columns="population", values=value_col
        ).fillna(0.0)
        matrix = _reindex_heatmap_axis(
            matrix, environments, axis=0, label="environment_order"
        )
        matrix = _reindex_heatmap_axis(
            matrix, populations, axis=1, label="population_order"
        )
        if write_tables:
            matrix.to_csv(
                table_dir / f"{value_col}_matrix.csv",
                float_format=table_float_format,
            )
        override = per_metric.get(value_col)
        options = _coerce_heatmap_options(override, base=base_options)
        if options.row_labels is None and environment_labels:
            options = replace(options, row_labels=environment_labels)
        if options.column_labels is None and population_labels:
            options = replace(options, column_labels=population_labels)
        outputs[value_col] = _plot_heatmap_variants(
            matrix,
            figure_dir / value_col,
            default_title=resolved_titles[value_col],
            default_cmap=resolved_cmaps[value_col],
            default_center=None,
            options=options,
            save_options=resolved_save,
        )
    return outputs


__all__ = [
    "EnvironmentAbundanceTables",
    "FigureSaveOptions",
    "HeatmapOptions",
    "HyperstacGalleryResult",
    "HyperstacMaskResult",
    "aggregate_environment_abundance",
    "assign_cells_to_hyperstac_masks",
    "hyperstac_cluster_feature_tables",
    "plot_cell_environment_composition",
    "plot_cluster_map_gallery",
    "plot_environment_abundance",
    "plot_hyperstac_cluster_features",
    "plot_hyperstac_environment_gallery",
    "plot_hyperstac_umap",
    "prepare_environment_abundance_tables",
    "reconstruct_cluster_label_masks",
    "summarize_cell_environment_composition",
    "summarize_environment_abundance",
]
