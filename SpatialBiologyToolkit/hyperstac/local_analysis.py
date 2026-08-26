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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import spearmanr

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


@dataclass(frozen=True)
class HyperstacMaskResult:
    """Files and tables produced while reconstructing ROI label masks."""

    mask_dir: Path
    preview_dir: Path | None
    mapping_path: Path
    manifest_path: Path
    mapping: pd.DataFrame
    manifest: pd.DataFrame


@dataclass(frozen=True)
class HyperstacGalleryResult:
    """Files and selection metadata produced by the environment gallery."""

    image_paths: tuple[Path, ...]
    marker_table_path: Path
    selection_table_path: Path
    marker_table: pd.DataFrame
    selection_table: pd.DataFrame


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
    formats: Sequence[str] = ("png", "svg"),
    dpi: int = 220,
) -> tuple[Path, ...]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for raw_format in formats:
        image_format = str(raw_format).lower().lstrip(".")
        if image_format not in {"png", "svg"}:
            raise ValueError("formats may contain only 'png' and 'svg'")
        path = stem.with_suffix(f".{image_format}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    return tuple(paths)


def _cluster_mapping(labels: pd.Series) -> pd.DataFrame:
    if labels.isna().any():
        raise ValueError("Cluster labels contain missing values; assign or remove them before rasterization")
    clusters = sorted(labels.astype(str).unique(), key=natural_key)
    if not clusters:
        raise ValueError("No cluster labels were found")
    return pd.DataFrame(
        {
            "cluster": clusters,
            "mask_value": np.arange(1, len(clusters) + 1, dtype=np.uint16),
        }
    )


def _cluster_palette(mapping: pd.DataFrame) -> dict[str, tuple[float, float, float, float]]:
    cmap_name = "tab20" if len(mapping) <= 20 else "gist_ncar"
    cmap = plt.colormaps[cmap_name].resampled(max(len(mapping), 1))
    return {
        str(row.cluster): cmap(position)
        for position, row in enumerate(mapping.itertuples(index=False))
    }


def _roi_image_shape(normalised_image_dir: Path, roi: str) -> tuple[int, int]:
    roi_dir = normalised_image_dir / roi
    images = sorted(roi_dir.glob("*.tif")) + sorted(roi_dir.glob("*.tiff"))
    if not images:
        raise FileNotFoundError(f"No normalized TIFF images were found for ROI {roi!r} in {roi_dir}")
    with tifffile.TiffFile(images[0]) as handle:
        shape = tuple(int(value) for value in handle.series[0].shape)
    if len(shape) != 2:
        raise ValueError(f"Expected a 2D normalized image for ROI {roi!r}, found shape {shape}")
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
        heights = pd.to_numeric(roi_obs["roi_height_px"], errors="raise").dropna().astype(int).unique()
        widths = pd.to_numeric(roi_obs["roi_width_px"], errors="raise").dropna().astype(int).unique()
        if len(heights) != 1 or len(widths) != 1:
            raise ValueError(f"ROI {roi!r} has inconsistent roi_height_px or roi_width_px values")
        metadata_shape = (int(heights[0]), int(widths[0]))
        if coordinate_shape[0] > metadata_shape[0] or coordinate_shape[1] > metadata_shape[1]:
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
            raise ValueError(f"Patch coordinates for ROI {roi!r} exceed normalized image shape {image_shape}")

    return image_shape or metadata_shape or coordinate_shape


def _label_colormap(mapping: pd.DataFrame) -> tuple[ListedColormap, list[Patch]]:
    palette = _cluster_palette(mapping)
    colors = [(0.0, 0.0, 0.0, 1.0)] + [palette[str(cluster)] for cluster in mapping["cluster"]]
    handles = [Patch(facecolor="black", label="uncovered (0)")]
    handles.extend(
        Patch(facecolor=palette[str(row.cluster)], label=f"{row.cluster} ({row.mask_value})")
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
    write_previews: bool = False,
    preview_rois: Sequence[str] | None = None,
) -> HyperstacMaskResult:
    """Reconstruct Leiden patches as exact-size categorical ROI rasters.

    Positive mask values correspond to clusters through the written mapping
    CSV.  Zero is retained for pixels with no accepted HyPERSTAC patch, which
    is important when these masks are subsequently used as spatial tissue
    regions.  Image dimensions are validated against normalized TIFFs when
    ``normalised_image_dir`` is supplied.
    """

    if overlap_policy not in {"error", "first", "last"}:
        raise ValueError("overlap_policy must be one of: 'error', 'first', 'last'")
    if "{roi}" not in mask_pattern:
        raise ValueError("mask_pattern must contain the '{roi}' placeholder")

    obs = _obs_frame(adata_or_obs)
    required = [roi_col, "row_start", "row_end", "col_start", "col_end", cluster_col]
    _require_columns(obs, required, "HyPERSTAC patch observations")
    if obs[roi_col].isna().any():
        raise ValueError(f"{roi_col!r} contains missing ROI identifiers")

    output_root = Path(output_dir)
    mask_dir = output_root / "label_masks"
    table_dir = output_root / "tables"
    preview_dir = output_root / "cluster_maps" if write_previews else None
    mask_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    image_root = Path(normalised_image_dir) if normalised_image_dir is not None else None
    mapping = _cluster_mapping(obs[cluster_col])
    value_lookup = dict(zip(mapping["cluster"].astype(str), mapping["mask_value"].astype(int)))
    mapping_path = table_dir / "hyperstac_cluster_label_mapping.csv"
    mapping.to_csv(mapping_path, index=False)

    requested_previews = None if preview_rois is None else {str(roi) for roi in preview_rois}
    cmap, legend_handles = _label_colormap(mapping)
    output_root_resolved = mask_dir.resolve()
    manifest_rows: list[dict[str, Any]] = []

    roi_values = obs[roi_col].astype(str)
    for roi in sorted(roi_values.unique(), key=natural_key):
        roi_obs = obs.loc[roi_values == roi]
        height, width = _roi_dimensions(roi_obs, roi=roi, normalised_image_dir=image_root)
        mask = np.zeros((height, width), dtype=np.uint16)
        occupied = np.zeros((height, width), dtype=bool)

        for patch_id, row in roi_obs.iterrows():
            row_start = int(row["row_start"])
            row_end = int(row["row_end"])
            col_start = int(row["col_start"])
            col_end = int(row["col_end"])
            if row_start < 0 or col_start < 0 or row_end <= row_start or col_end <= col_start:
                raise ValueError(f"Patch {patch_id!r} has invalid bounds")
            if row_end > height or col_end > width:
                raise ValueError(f"Patch {patch_id!r} exceeds ROI {roi!r} dimensions {(height, width)}")
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
            raise ValueError("mask_pattern must create paths relative to output_dir/label_masks")
        mask_path = (mask_dir / relative).resolve()
        if not mask_path.is_relative_to(output_root_resolved):
            raise ValueError(f"mask_pattern for ROI {roi!r} resolves outside the mask directory")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(mask_path, mask, compression="zlib")

        should_preview = write_previews and (requested_previews is None or roi in requested_previews)
        preview_path: Path | None = None
        if should_preview and preview_dir is not None:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(mask, cmap=cmap, interpolation="nearest", vmin=0, vmax=len(mapping))
            ax.set_title(f"{roi}: {cluster_col}")
            ax.set_axis_off()
            ax.legend(handles=legend_handles, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout()
            preview_path = preview_dir / f"{safe_filename(roi)}__cluster_map.png"
            fig.savefig(preview_path, dpi=220, bbox_inches="tight")
            plt.close(fig)

        assigned_pixels = int(np.count_nonzero(mask))
        manifest_rows.append(
            {
                "roi": roi,
                "mask_path": str(mask_path),
                "preview_path": str(preview_path) if preview_path is not None else None,
                "height_px": height,
                "width_px": width,
                "n_patches": int(len(roi_obs)),
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
    formats: Sequence[str] = ("png", "svg"),
) -> tuple[Path, ...]:
    """Plot several reconstructed ROI label masks in one compact figure."""

    manifest = mask_result.manifest.copy()
    if rois is not None:
        requested = [str(roi) for roi in rois]
        missing = sorted(set(requested).difference(manifest["roi"]))
        if missing:
            raise ValueError(f"Requested ROIs are absent from the mask manifest: {missing}")
        manifest = manifest.set_index("roi").loc[requested].reset_index()
    elif len(manifest) > max_rois:
        positions = np.linspace(0, len(manifest) - 1, max_rois, dtype=int)
        manifest = manifest.iloc[positions]
    if manifest.empty:
        raise ValueError("No ROI masks were selected")

    ncols = max(1, min(int(ncols), len(manifest)))
    nrows = int(np.ceil(len(manifest) / ncols))
    cmap, handles = _label_colormap(mask_result.mapping)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols + 2.2, 3.1 * nrows), squeeze=False)
    for ax, row in zip(axes.flat, manifest.itertuples(index=False)):
        mask = tifffile.imread(row.mask_path)
        ax.imshow(mask, cmap=cmap, interpolation="nearest", vmin=0, vmax=len(mask_result.mapping))
        ax.set_title(str(row.roi), fontsize=9)
        ax.set_axis_off()
    for ax in axes.flat[len(manifest) :]:
        ax.set_visible(False)
    axes.flat[min(len(manifest), len(axes.flat)) - 1].legend(
        handles=handles,
        fontsize=7,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title="Leiden (mask value)",
    )
    fig.suptitle("HyPERSTAC environments back-gated to original ROI coordinates", y=1.01)
    fig.tight_layout()
    paths = _save_figure(fig, output_stem, formats=formats)
    plt.close(fig)
    return paths


def _align_feature_frame(reference: ad.AnnData, frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = reference.obs_names.difference(frame.index)
    if len(missing):
        raise ValueError(f"{label} is missing {len(missing)} patches from the representation AnnData")
    return frame.loc[reference.obs_names]


def hyperstac_cluster_feature_tables(
    representation_adata: ad.AnnData,
    metrics_adata: ad.AnnData,
    cluster_col: str,
    *,
    permutation_adata: ad.AnnData | None = None,
) -> dict[str, pd.DataFrame]:
    """Summarize marker intensity and perturbation sensitivity by cluster."""

    _require_columns(representation_adata.obs, [cluster_col], "representation AnnData obs")
    clusters = representation_adata.obs[cluster_col].astype(str)
    order = sorted(clusters.unique(), key=natural_key)

    intensity = _align_feature_frame(
        representation_adata,
        channel_mean_dataframe(metrics_adata),
        "metrics AnnData",
    )
    intensity_mean = intensity.groupby(clusters, observed=True).mean().reindex(order)
    intensity_z = (
        (intensity_mean - intensity_mean.mean(axis=0))
        / intensity_mean.std(axis=0).replace(0, np.nan)
    ).fillna(0.0)
    tables = {
        "mean_marker_intensity": intensity_mean,
        "relative_marker_intensity_zscore": intensity_z,
    }

    if permutation_adata is not None:
        for perturbation_type in ("zero_channel", "shuffle_channel"):
            scores = permutation_channel_dataframe(permutation_adata, perturbation_type)
            if scores.empty:
                continue
            scores = _align_feature_frame(
                representation_adata,
                scores,
                f"{perturbation_type} permutation AnnData",
            )
            tables[f"mean_{perturbation_type}_cosine_distance"] = (
                scores.groupby(clusters, observed=True).mean().reindex(order)
            )

        var = permutation_adata.var
        if {"perturbation_type", "channel"}.issubset(var.columns):
            matrix = get_matrix(permutation_adata)
            all_channel_rows: dict[str, np.ndarray] = {}
            channel_values = var["channel"].astype(str)
            type_values = var["perturbation_type"].astype(str)
            for perturbation_type in sorted(type_values[channel_values == "all_channels"].unique()):
                positions = np.flatnonzero(
                    ((channel_values == "all_channels") & (type_values == perturbation_type)).to_numpy()
                )
                if len(positions):
                    all_channel_rows[perturbation_type] = np.nanmean(matrix[:, positions], axis=1)
            if all_channel_rows:
                all_scores = pd.DataFrame(all_channel_rows, index=permutation_adata.obs_names)
                all_scores = _align_feature_frame(
                    representation_adata,
                    all_scores,
                    "all-channel permutation AnnData",
                )
                tables["mean_all_channel_perturbation_cosine_distance"] = (
                    all_scores.groupby(clusters, observed=True).mean().reindex(order)
                )
    return tables


def _heatmap_style(name: str) -> tuple[str, float | None]:
    if "zscore" in name or "correlation" in name:
        return "vlag", 0.0
    if "permutation" in name or "cosine_distance" in name:
        return "rocket", None
    return "mako", None


def _plot_heatmap_variants(
    data: pd.DataFrame,
    output_stem: Path,
    title: str,
    *,
    cmap: str,
    center: float | None,
    formats: Sequence[str],
) -> dict[str, tuple[Path, ...]]:
    if data.empty:
        return {}
    width = max(6.0, min(18.0, 0.42 * data.shape[1] + 2.5))
    height = max(4.0, min(22.0, 0.28 * data.shape[0] + 2.5))
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(data, cmap=cmap, center=center, ax=ax, linewidths=0.1, linecolor="white")
    ax.set_title(title)
    fig.tight_layout()
    fixed_paths = _save_figure(fig, output_stem, formats=formats)
    plt.close(fig)
    outputs = {"fixed": fixed_paths}

    if data.shape[0] >= 2 and data.shape[1] >= 2:
        grid = sns.clustermap(
            data,
            cmap=cmap,
            center=center,
            figsize=(width, height),
            linewidths=0.1,
            linecolor="white",
            row_cluster=True,
            col_cluster=True,
        )
        grid.fig.suptitle(f"{title} — clustered rows and columns", y=1.02)
        clustered_paths = _save_figure(
            grid.fig,
            output_stem.with_name(f"{output_stem.name}_clustermap"),
            formats=formats,
        )
        plt.close(grid.fig)
        outputs["clustermap"] = clustered_paths
    return outputs


def plot_hyperstac_cluster_features(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    formats: Sequence[str] = ("png", "svg"),
) -> dict[str, dict[str, tuple[Path, ...]]]:
    """Write fixed-order heatmaps and bi-clustered variants for feature tables."""

    output_root = Path(output_dir)
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, tuple[Path, ...]]] = {}
    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv")
        cmap, center = _heatmap_style(name)
        title = name.replace("_", " ").capitalize()
        outputs[name] = _plot_heatmap_variants(
            table,
            figure_dir / name,
            title,
            cmap=cmap,
            center=center,
            formats=formats,
        )
    return outputs


def plot_hyperstac_umap(
    representation_adata: ad.AnnData,
    cluster_col: str,
    output_stem: str | Path,
    *,
    formats: Sequence[str] = ("png", "svg"),
    point_size: float = 5.0,
) -> tuple[Path, ...]:
    """Plot the UMAP graph paired with a namespaced HyPERSTAC clustering."""

    _require_columns(representation_adata.obs, [cluster_col], "representation AnnData obs")
    umap_key = cluster_col_umap_key(representation_adata, cluster_col)
    if umap_key not in representation_adata.obsm:
        raise ValueError(f"No UMAP coordinates named {umap_key!r} were found")
    coords = np.asarray(representation_adata.obsm[umap_key])
    labels = representation_adata.obs[cluster_col].astype(str)
    mapping = _cluster_mapping(labels)
    palette = _cluster_palette(mapping)

    fig, ax = plt.subplots(figsize=(8, 7))
    for cluster in mapping["cluster"]:
        selected = labels == str(cluster)
        ax.scatter(
            coords[selected, 0],
            coords[selected, 1],
            s=point_size,
            color=palette[str(cluster)],
            label=str(cluster),
            linewidths=0,
            alpha=0.8,
            rasterized=True,
        )
    ax.set_xlabel("UMAP 1 (VICReg representation)")
    ax.set_ylabel("UMAP 2 (VICReg representation)")
    ax.set_title(f"HyPERSTAC patch structure: {cluster_col}")
    ax.legend(title="Leiden environment", bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=3)
    fig.tight_layout()
    paths = _save_figure(fig, output_stem, formats=formats)
    plt.close(fig)
    return paths


def _unique_roi_examples(
    obs: pd.DataFrame,
    patch_ids: pd.Index,
    scores: pd.Series,
    count: int,
) -> list[str]:
    ordered = scores.reindex(patch_ids).sort_values(ascending=False, na_position="last").index
    selected: list[str] = []
    seen_rois: set[str] = set()
    for patch_id in ordered:
        roi = str(obs.loc[patch_id, "roi"])
        if roi in seen_rois:
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
        raise FileNotFoundError(f"Normalized channel {channel!r} was not found for ROI {roi!r} in {roi_dir}")
    return path


def _normalised_rgb_crop(
    root: Path,
    row: pd.Series,
    channels: Sequence[str],
    *,
    contrast_percentile: float,
) -> np.ndarray:
    shape = (int(row["row_end"]) - int(row["row_start"]), int(row["col_end"]) - int(row["col_start"]))
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
        high = float(np.nanpercentile(crop, contrast_percentile))
        if np.isfinite(high) and high > 0:
            crop = crop / high
        rgb[:, :, color_index] = np.clip(crop, 0.0, 1.0)
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
    examples_per_cluster: int = 4,
    channels_per_composite: int = 3,
    contrast_percentile: float = 99.5,
    formats: Sequence[str] = ("png", "svg"),
) -> HyperstacGalleryResult:
    """Create one multi-row PNG/SVG gallery using normalized ROI TIFF crops.

    Each row represents one Leiden environment.  Up to three channels are
    rendered as red, green, and blue.  Markers are selected by relative cluster
    intensity or cluster-specific zero/shuffle perturbation sensitivity, and
    examples preferentially come from distinct ROIs.
    """

    if examples_per_cluster < 1:
        raise ValueError("examples_per_cluster must be at least one")
    if channels_per_composite < 1 or channels_per_composite > 3:
        raise ValueError("channels_per_composite must be between one and three")
    if not 50 <= contrast_percentile <= 100:
        raise ValueError("contrast_percentile must be between 50 and 100")
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
            raise ValueError(f"marker_source={marker_source!r} requires permutation_adata")
        source_frame = permutation_channel_dataframe(permutation_adata, marker_source)
        source_frame = _align_feature_frame(
            representation_adata,
            source_frame,
            f"{marker_source} permutation AnnData",
        )
        shared = [channel for channel in intensity.columns if channel in source_frame.columns]
        if not shared:
            raise ValueError(f"No normalized image channels overlap the {marker_source} scores")
        source_frame = source_frame.loc[:, shared]

    labels = representation_adata.obs[cluster_col].astype(str)
    mapping = _cluster_mapping(labels)
    cluster_means = source_frame.groupby(labels, observed=True).mean().reindex(mapping["cluster"])
    relative = (
        (cluster_means - cluster_means.mean(axis=0))
        / cluster_means.std(axis=0).replace(0, np.nan)
    ).fillna(0.0)

    marker_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    selected_by_cluster: dict[str, list[str]] = {}
    markers_by_cluster: dict[str, list[str]] = {}
    for cluster in mapping["cluster"].astype(str):
        markers = relative.loc[cluster].sort_values(ascending=False).head(channels_per_composite).index.astype(str).tolist()
        if len(markers) < channels_per_composite:
            for channel in intensity.columns:
                if channel not in markers:
                    markers.append(str(channel))
                if len(markers) == channels_per_composite:
                    break
        markers_by_cluster[cluster] = markers
        marker_rows.append(
            {
                "cluster": cluster,
                "marker_source": marker_source,
                "red": markers[0] if len(markers) > 0 else None,
                "green": markers[1] if len(markers) > 1 else None,
                "blue": markers[2] if len(markers) > 2 else None,
            }
        )
        patch_ids = representation_adata.obs_names[labels == cluster]
        scores = source_frame.loc[patch_ids, markers].mean(axis=1)
        selected = _unique_roi_examples(
            representation_adata.obs,
            patch_ids,
            scores,
            examples_per_cluster,
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
                }
            )

    marker_table = pd.DataFrame(marker_rows)
    selection_table = pd.DataFrame(selections)
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    marker_table_path = output_stem.with_name(f"{output_stem.name}_markers.csv")
    selection_table_path = output_stem.with_name(f"{output_stem.name}_selections.csv")
    marker_table.to_csv(marker_table_path, index=False)
    selection_table.to_csv(selection_table_path, index=False)

    nrows = len(mapping)
    fig, axes = plt.subplots(
        nrows,
        examples_per_cluster,
        figsize=(2.4 * examples_per_cluster + 3.4, 2.15 * nrows),
        squeeze=False,
    )
    root = Path(normalised_image_dir)
    color_names = ["R", "G", "B"]
    for row_index, cluster in enumerate(mapping["cluster"].astype(str)):
        markers = markers_by_cluster[cluster]
        selected = selected_by_cluster[cluster]
        for col_index, patch_id in enumerate(selected):
            row = representation_adata.obs.loc[patch_id]
            rgb = _normalised_rgb_crop(
                root,
                row,
                markers,
                contrast_percentile=contrast_percentile,
            )
            ax = axes[row_index, col_index]
            ax.imshow(rgb, interpolation="nearest")
            ax.set_title(f"{row['roi']}\n({int(row['col_start'])}, {int(row['row_start'])})", fontsize=7)
            ax.set_axis_off()
        for col_index in range(len(selected), examples_per_cluster):
            axes[row_index, col_index].set_visible(False)
        marker_text = "\n".join(
            f"{color_names[index]}: {marker}" for index, marker in enumerate(markers)
        )
        axes[row_index, 0].text(
            -0.08,
            0.5,
            f"Environment {cluster}\n{marker_text}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8,
        )
    fig.suptitle(
        f"Representative HyPERSTAC environments — markers selected by {marker_source.replace('_', ' ')}",
        y=1.005,
    )
    fig.tight_layout()
    image_paths = _save_figure(fig, output_stem, formats=formats)
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
    environments = sorted(obs[cluster_col].dropna().astype(str).unique(), key=natural_key)
    complete_index = pd.MultiIndex.from_product(
        [rois, environments], names=["roi", "environment"]
    )
    counts = (
        obs.assign(_roi=obs[roi_col].astype(str), _environment=obs[cluster_col].astype(str))
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
        sample_metadata[metadata_roi_col] = sample_metadata[metadata_roi_col].astype(str)
        duplicates = sample_metadata[metadata_roi_col][sample_metadata[metadata_roi_col].duplicated()].unique()
        if len(duplicates):
            raise ValueError(f"Sample metadata contains duplicate ROI identifiers: {duplicates.tolist()}")
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

    _require_columns(abundance, [group_col, "environment", "patch_count"], "abundance table")
    valid = abundance.loc[abundance[group_col].notna()].copy()
    grouped = (
        valid.groupby([group_col, "environment"], observed=True)["patch_count"]
        .sum()
        .rename("patch_count")
        .reset_index()
    )
    grouped["total_patches"] = grouped.groupby(group_col)["patch_count"].transform("sum")
    grouped["fraction"] = grouped["patch_count"] / grouped["total_patches"]
    return grouped


def plot_environment_abundance(
    abundance: pd.DataFrame,
    output_dir: str | Path,
    *,
    sample_group_col: str | None = "Case",
    categorical_metadata: Sequence[str] = (),
    numeric_metadata: Sequence[str] = (),
    formats: Sequence[str] = ("png", "svg"),
) -> dict[str, Any]:
    """Plot overall, ROI, sample-group, and metadata-associated abundance."""

    _require_columns(abundance, ["roi", "environment", "patch_count", "fraction"], "abundance table")
    output_root = Path(output_dir)
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    abundance.to_csv(table_dir / "roi_environment_abundance.csv", index=False)
    outputs: dict[str, Any] = {}

    overall = abundance.groupby("environment", observed=True)["patch_count"].sum()
    overall = overall.reindex(sorted(overall.index.astype(str), key=natural_key))
    overall_fraction = overall / overall.sum()
    overall_fraction.rename("fraction").to_csv(table_dir / "overall_environment_abundance.csv")
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(overall_fraction)), 4.5))
    sns.barplot(x=overall_fraction.index.astype(str), y=overall_fraction.to_numpy(), ax=ax, color="#4c78a8")
    ax.set_xlabel("HyPERSTAC environment")
    ax.set_ylabel("Fraction of accepted patches")
    ax.set_title("Overall HyPERSTAC environment abundance")
    fig.tight_layout()
    outputs["overall"] = _save_figure(fig, figure_dir / "overall_environment_abundance", formats=formats)
    plt.close(fig)

    roi_matrix = abundance.pivot(index="roi", columns="environment", values="fraction").fillna(0.0)
    roi_matrix.to_csv(table_dir / "roi_environment_fraction_matrix.csv")
    outputs["roi"] = _plot_heatmap_variants(
        roi_matrix,
        figure_dir / "roi_environment_fraction",
        "Environment fractions by ROI",
        cmap="mako",
        center=None,
        formats=formats,
    )

    if sample_group_col is not None and sample_group_col in abundance.columns:
        grouped = aggregate_environment_abundance(abundance, sample_group_col)
        grouped.to_csv(table_dir / f"{safe_filename(sample_group_col)}_environment_abundance.csv", index=False)
        group_matrix = grouped.pivot(index=sample_group_col, columns="environment", values="fraction").fillna(0.0)
        outputs["sample_group"] = _plot_heatmap_variants(
            group_matrix,
            figure_dir / f"{safe_filename(sample_group_col)}_environment_fraction",
            f"Environment fractions by {sample_group_col}",
            cmap="mako",
            center=None,
            formats=formats,
        )

    categorical_outputs: dict[str, Any] = {}
    for column in categorical_metadata:
        if column not in abundance.columns:
            raise ValueError(f"Categorical metadata column {column!r} is absent from the abundance table")
        valid = abundance.loc[abundance[column].notna()]
        table = valid.groupby([column, "environment"], observed=True)["fraction"].mean().unstack(fill_value=0.0)
        if len(table) < 2:
            continue
        table.to_csv(table_dir / f"categorical_{safe_filename(column)}_mean_fraction.csv")
        categorical_outputs[column] = _plot_heatmap_variants(
            table,
            figure_dir / f"categorical_{safe_filename(column)}_mean_fraction",
            f"Mean ROI environment fraction by {column}",
            cmap="mako",
            center=None,
            formats=formats,
        )
    outputs["categorical_metadata"] = categorical_outputs

    if numeric_metadata:
        roi_metadata = abundance.drop_duplicates("roi").set_index("roi")
        correlations = pd.DataFrame(index=roi_matrix.columns, columns=list(numeric_metadata), dtype=float)
        pvalues = correlations.copy()
        for column in numeric_metadata:
            if column not in roi_metadata.columns:
                raise ValueError(f"Numeric metadata column {column!r} is absent from the abundance table")
            values = pd.to_numeric(roi_metadata[column], errors="coerce").reindex(roi_matrix.index)
            for environment in roi_matrix.columns:
                valid = values.notna() & roi_matrix[environment].notna()
                if valid.sum() >= 3 and values[valid].nunique() >= 2:
                    statistic, pvalue = spearmanr(values[valid], roi_matrix.loc[valid, environment])
                    correlations.loc[environment, column] = statistic
                    pvalues.loc[environment, column] = pvalue
        correlations.to_csv(table_dir / "numeric_metadata_spearman_correlations.csv")
        pvalues.to_csv(table_dir / "numeric_metadata_spearman_pvalues.csv")
        outputs["numeric_metadata"] = _plot_heatmap_variants(
            correlations,
            figure_dir / "numeric_metadata_spearman_correlations",
            "Spearman association: ROI environment fraction vs sample metadata",
            cmap="vlag",
            center=0.0,
            formats=formats,
        )
    return outputs


def _mapping_dictionary(mapping: pd.DataFrame | Mapping[Any, Any]) -> dict[int, str]:
    if isinstance(mapping, pd.DataFrame):
        _require_columns(mapping, ["mask_value", "cluster"], "cluster label mapping")
        return dict(zip(mapping["mask_value"].astype(int), mapping["cluster"].astype(str)))
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

    selected = obs.loc[obs[roi_col].astype(str).isin(roi_keys), [roi_col, x_col, y_col, pop_col]].copy()
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
        result.iloc[positions[finite], result.columns.get_loc("status")] = "out_of_bounds"
        finite_positions = positions[finite]
        x = coordinates[finite, 0].astype(np.int64)
        y = coordinates[finite, 1].astype(np.int64)
        mask = tifffile.imread(paths[roi])
        in_bounds = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
        bounded_positions = finite_positions[in_bounds]
        values = mask[y[in_bounds], x[in_bounds]].astype(int)
        result.iloc[bounded_positions, result.columns.get_loc("mask_value")] = values
        mapped = pd.Series(values).map(value_to_cluster)
        statuses = np.where(values == 0, "uncovered", np.where(mapped.isna(), "unknown_mask_value", "assigned"))
        result.iloc[bounded_positions, result.columns.get_loc("status")] = statuses
        result.iloc[bounded_positions, result.columns.get_loc("environment")] = mapped.to_numpy()
    return result


def summarize_cell_environment_composition(assignments: pd.DataFrame) -> pd.DataFrame:
    """Summarize assigned cell populations within each HyPERSTAC environment."""

    _require_columns(assignments, ["environment", "population", "status"], "cell assignments")
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
    counts["fraction_within_environment"] = counts["cell_count"] / counts.groupby("environment")["cell_count"].transform("sum")
    counts["fraction_within_population"] = counts["cell_count"] / counts.groupby("population")["cell_count"].transform("sum")
    return counts


def plot_cell_environment_composition(
    composition: pd.DataFrame,
    output_dir: str | Path,
    *,
    formats: Sequence[str] = ("png", "svg"),
) -> dict[str, dict[str, tuple[Path, ...]]]:
    """Plot descriptive cell-population composition of each environment."""

    _require_columns(
        composition,
        ["environment", "population", "cell_count", "fraction_within_environment", "fraction_within_population"],
        "cell-environment composition",
    )
    output_root = Path(output_dir)
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    composition.to_csv(table_dir / "cell_environment_composition.csv", index=False)
    outputs: dict[str, dict[str, tuple[Path, ...]]] = {}
    for value_col, title, cmap in (
        ("fraction_within_environment", "Cell-population composition within each environment", "mako"),
        ("fraction_within_population", "Distribution of each population across environments", "rocket"),
    ):
        matrix = composition.pivot(index="environment", columns="population", values=value_col).fillna(0.0)
        matrix.to_csv(table_dir / f"{value_col}_matrix.csv")
        outputs[value_col] = _plot_heatmap_variants(
            matrix,
            figure_dir / value_col,
            title,
            cmap=cmap,
            center=None,
            formats=formats,
        )
    return outputs


__all__ = [
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
    "reconstruct_cluster_label_masks",
    "summarize_cell_environment_composition",
    "summarize_environment_abundance",
]
