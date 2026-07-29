#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run clustering parameter scans and create HyPERSTAC IMC visualisation outputs.

Inputs are the representation AnnData, patch-metrics AnnData, optional
permutation-sensitivity AnnData, saved patch .npy files, and optional
normalisation reports. The script can add a grid of Scanpy neighbors, UMAP, and
Leiden clusterings to the representation AnnData before writing UMAPs, heatmaps,
ROI back-gated cluster maps, TIFF cluster label masks, and example patch
galleries.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import sparse
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HyPERSTAC IMC clustering parameter scans and visualisation reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hyperstac-output-dir", type=Path, required=True)
    parser.add_argument("--representation-adata", type=Path, default=None)
    parser.add_argument("--metrics-adata", type=Path, default=None)
    parser.add_argument("--permutation-adata", type=Path, default=None)
    parser.add_argument("--normalisation-report-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--cluster-col",
        type=str,
        default="leiden",
        help="Cluster column to plot. Can be a comma-separated list of column names or unique search terms.",
    )
    parser.add_argument(
        "--cluster-col-search",
        type=str,
        default="leiden",
        help=(
            "Generate one report for every adata.obs column containing this term. "
            "Use an empty string to disable search and use --cluster-col exactly."
        ),
    )
    parser.add_argument(
        "--run-cluster-scan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute the Scanpy neighbors/UMAP/Leiden parameter grid before plotting.",
    )
    parser.add_argument(
        "--leiden-resolution",
        "--leiden-resolutions",
        dest="leiden_resolution",
        nargs="+",
        default=["1.0"],
        help="Leiden resolution list for the clustering grid. Accepts comma- or space-separated values.",
    )
    parser.add_argument(
        "--scanpy-n-neighbors",
        nargs="+",
        default=["30"],
        help="Scanpy n_neighbors list for the clustering grid. Accepts comma- or space-separated values.",
    )
    parser.add_argument(
        "--scanpy-n-pcs",
        "--scanpy-pcs",
        dest="scanpy_n_pcs",
        nargs="+",
        default=["0"],
        help="PC-count list for the clustering grid. Use 0 for full adata.X without PCA.",
    )
    parser.add_argument("--scanpy-min-dist", type=float, default=0.1, help="UMAP min_dist for each clustering grid graph.")
    parser.add_argument(
        "--write-clustered-adata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the AnnData back to disk after adding clustering-grid columns.",
    )
    parser.add_argument(
        "--replace-existing-cluster-scan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Before running the grid, drop existing adata.obs columns matching --cluster-col-search.",
    )
    parser.add_argument(
        "--clustered-adata-output",
        type=Path,
        default=None,
        help="Optional path for clustered AnnData. Defaults to overwriting --representation-adata.",
    )
    parser.add_argument("--max-umap-channels", type=int, default=0, help="0 means all channel intensity UMAPs.")
    parser.add_argument("--max-permutation-umaps", type=int, default=30, help="0 disables permutation UMAPs.")
    parser.add_argument("--max-roi-maps", type=int, default=0, help="0 means all ROIs.")
    parser.add_argument(
        "--spatial-cluster-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write ROI back-gated cluster maps and TIFF label masks.",
    )
    parser.add_argument("--patches-per-cluster", type=int, default=12)
    parser.add_argument("--gallery-top-markers", type=int, default=3)
    parser.add_argument("--gallery-cols", type=int, default=4)
    parser.add_argument(
        "--split-channel-gallery",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show one selected patch per row, with an RGB merge of the top three markers plus one column per top marker channel.",
    )
    parser.add_argument("--gallery-auto-contrast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gallery-vmax", type=float, default=1.0, help="Fixed gallery channel maximum when auto contrast is disabled.")
    parser.add_argument("--gallery-contrast-percentile", type=float, default=100)
    parser.add_argument(
        "--gallery-marker-source",
        type=str,
        default="auto",
        choices=["auto", "permutation", "intensity"],
        help="Use cluster-specific permutation impact, intensity, or automatic fallback to choose RGB gallery markers.",
    )
    parser.add_argument(
        "--gallery-permutation-type",
        type=str,
        default="zero_channel",
        choices=["zero_channel", "shuffle_channel"],
        help="Permutation score used to choose gallery markers when permutation data are available.",
    )
    parser.add_argument("--de-method", type=str, default="wilcoxon", choices=["wilcoxon", "t-test", "logreg"])
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.hyperstac_output_dir = args.hyperstac_output_dir.resolve()
    args.representation_adata = (
        args.representation_adata
        or args.hyperstac_output_dir / "imc_hyperstac_representations.h5ad"
    ).resolve()
    args.metrics_adata = (
        args.metrics_adata
        or args.hyperstac_output_dir / "imc_hyperstac_patch_metrics.h5ad"
    ).resolve()
    args.permutation_adata = (
        args.permutation_adata
        or args.hyperstac_output_dir / "permutation_sensitivity" / "imc_permutation_sensitivity.h5ad"
    ).resolve()
    args.normalisation_report_dir = (
        args.normalisation_report_dir
        or args.hyperstac_output_dir / "hyperstac_normalised_images"
    ).resolve()
    args.output_dir = (args.output_dir or args.hyperstac_output_dir / "visualisations").resolve()
    args.clustered_adata_output = (args.clustered_adata_output or args.representation_adata).resolve()
    args.cluster_col_search = optional_string(args.cluster_col_search)
    return args


def ensure_dirs(base: Path) -> dict[str, Path]:
    paths = {
        "umaps": base / "umaps",
        "heatmaps": base / "heatmaps",
        "spatial": base / "spatial_cluster_maps",
        "masks": base / "spatial_cluster_maps" / "tiff_label_masks",
        "galleries": base / "patch_galleries",
        "qc": base / "qc",
        "tables": base / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def safe_filename(value: str) -> str:
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_")
    return name or "value"


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value))]


def optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def split_list_values(values: list[str] | tuple[str, ...] | str) -> list[str]:
    if isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = list(values)
    items: list[str] = []
    for raw_value in raw_values:
        items.extend(item for item in re.split(r"[,\s]+", str(raw_value).strip()) if item)
    return items


def parse_float_list(values: list[str] | tuple[str, ...] | str, label: str) -> list[float]:
    parsed = []
    for item in split_list_values(values):
        try:
            parsed.append(float(item))
        except ValueError as exc:
            raise ValueError(f"{label} contains a non-numeric value: {item!r}") from exc
    if not parsed:
        raise ValueError(f"{label} must contain at least one value.")
    return parsed


def parse_int_list(values: list[str] | tuple[str, ...] | str, label: str) -> list[int]:
    parsed = []
    for item in split_list_values(values):
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"{label} contains a non-integer value: {item!r}") from exc
        parsed.append(value)
    if not parsed:
        raise ValueError(f"{label} must contain at least one value.")
    return parsed


def format_resolution_label(resolution: float) -> str:
    return str(float(resolution))


def clustering_basis_label(n_pcs: int) -> str:
    return f"P{int(n_pcs)}"


def clustering_graph_label(n_neighbors: int, n_pcs: int) -> str:
    return f"N{int(n_neighbors)}_{clustering_basis_label(n_pcs)}"


def leiden_key_for_parameters(resolution: float, n_neighbors: int, n_pcs: int) -> str:
    return f"leiden_{format_resolution_label(resolution)}_N{int(n_neighbors)}_P{int(n_pcs)}"


def get_matrix(adata: ad.AnnData) -> np.ndarray:
    if sparse.issparse(adata.X):
        return adata.X.toarray()
    return np.asarray(adata.X)


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def get_umap_coords(adata: ad.AnnData, umap_key: str) -> np.ndarray | None:
    if umap_key in adata.obsm:
        return adata.obsm[umap_key]
    if umap_key != "X_umap" and "X_umap" in adata.obsm:
        return adata.obsm["X_umap"]
    return None


def resolve_cluster_col(adata: ad.AnnData, requested: str) -> str:
    if requested in adata.obs:
        return requested
    matches = [column for column in adata.obs.columns if requested.lower() in str(column).lower()]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"Cluster column '{requested}' was ambiguous; matches: {matches}")
    raise ValueError(f"Cluster column '{requested}' was not found in representation AnnData.")


def parse_cluster_col_list(value: str) -> list[str]:
    columns = [item.strip() for item in str(value).split(",") if item.strip()]
    if not columns:
        raise ValueError("--cluster-col must contain at least one column name or search term.")
    return columns


def resolve_cluster_cols(adata: ad.AnnData, args: argparse.Namespace) -> list[str]:
    if args.cluster_col_search is not None:
        search = args.cluster_col_search.lower()
        matches = [
            column
            for column in adata.obs.columns
            if search in str(column).lower()
        ]
        matches = sorted(matches, key=natural_key)
        if not matches:
            raise ValueError(
                f"No adata.obs columns matched --cluster-col-search '{args.cluster_col_search}'. "
                f"Available columns include: {list(adata.obs.columns[:20])}"
            )
        return matches

    resolved = []
    seen = set()
    for requested in parse_cluster_col_list(args.cluster_col):
        cluster_col = resolve_cluster_col(adata, requested)
        if cluster_col not in seen:
            resolved.append(cluster_col)
            seen.add(cluster_col)
    return resolved


def cluster_col_umap_key(adata: ad.AnnData, cluster_col: str) -> str:
    mapping = adata.uns.get("cluster_scan_umap_keys", {})
    if isinstance(mapping, dict):
        key = mapping.get(cluster_col)
        if isinstance(key, str):
            return key
    return "X_umap"


def parse_cluster_scan_parameters(args: argparse.Namespace) -> tuple[list[float], list[int], list[int]]:
    resolutions = parse_float_list(args.leiden_resolution, "--leiden-resolution")
    n_neighbors_values = parse_int_list(args.scanpy_n_neighbors, "--scanpy-n-neighbors")
    n_pcs_values = parse_int_list(args.scanpy_n_pcs, "--scanpy-n-pcs")

    if any(resolution <= 0 for resolution in resolutions):
        raise ValueError("--leiden-resolution values must be positive.")
    if any(n_neighbors <= 0 for n_neighbors in n_neighbors_values):
        raise ValueError("--scanpy-n-neighbors values must be positive.")
    if any(n_pcs < 0 for n_pcs in n_pcs_values):
        raise ValueError("--scanpy-n-pcs values must be non-negative. Use 0 for full adata.X.")
    return resolutions, n_neighbors_values, n_pcs_values


def prepare_pca_representations(adata: ad.AnnData, n_pcs_values: list[int]) -> dict[int, str]:
    rep_by_pcs = {0: "X"}
    requested_positive = sorted({int(n_pcs) for n_pcs in n_pcs_values if int(n_pcs) > 0})
    if not requested_positive:
        return rep_by_pcs

    import scanpy as sc

    max_possible = min(adata.n_obs, adata.n_vars) - 1
    if max_possible < 1:
        raise ValueError("PCA clustering requires at least two observations and two representation dimensions.")

    max_requested = max(requested_positive)
    if max_requested > max_possible:
        raise ValueError(
            f"Requested up to {max_requested} PCs, but only {max_possible} are possible "
            f"for AnnData with shape {adata.shape}."
        )
    max_n_pcs = max_requested

    print(f"Running Scanpy PCA with n_comps={max_n_pcs}")
    sc.pp.pca(adata, n_comps=max_n_pcs)
    for requested_n_pcs in requested_positive:
        rep_key = f"X_pca_{requested_n_pcs}"
        adata.obsm[rep_key] = adata.obsm["X_pca"][:, :requested_n_pcs].copy()
        rep_by_pcs[requested_n_pcs] = rep_key
    return rep_by_pcs


def run_cluster_parameter_scan(args: argparse.Namespace, adata: ad.AnnData) -> list[str]:
    import scanpy as sc

    if args.replace_existing_cluster_scan and args.cluster_col_search is not None:
        search = args.cluster_col_search.lower()
        existing = [column for column in adata.obs.columns if search in str(column).lower()]
        if existing:
            print(f"Dropping {len(existing)} existing cluster columns matching '{args.cluster_col_search}': {existing}")
            adata.obs = adata.obs.drop(columns=existing)

    resolutions, n_neighbors_values, n_pcs_values = parse_cluster_scan_parameters(args)
    rep_by_pcs = prepare_pca_representations(adata, n_pcs_values)
    scan_rows = []
    cluster_columns = []
    cluster_to_umap: dict[str, str] = {}
    cluster_to_neighbors: dict[str, str] = {}

    for n_pcs in n_pcs_values:
        rep_key = rep_by_pcs[n_pcs]
        for n_neighbors in n_neighbors_values:
            graph_label = clustering_graph_label(n_neighbors, n_pcs)
            neighbors_key = f"neighbors_{graph_label}"
            umap_key = f"X_umap_{graph_label}"
            print(f"Running Scanpy neighbors for {graph_label} using {rep_key}")
            sc.pp.neighbors(
                adata,
                use_rep=rep_key,
                n_neighbors=n_neighbors,
                key_added=neighbors_key,
            )
            print(f"Running UMAP for {graph_label} into adata.obsm['{umap_key}']")
            sc.tl.umap(
                adata,
                min_dist=args.scanpy_min_dist,
                neighbors_key=neighbors_key,
            )
            adata.obsm[umap_key] = adata.obsm["X_umap"].copy()

            for resolution in resolutions:
                cluster_col = leiden_key_for_parameters(resolution, n_neighbors, n_pcs)
                print(f"Running Leiden resolution={resolution}, {graph_label} into adata.obs['{cluster_col}']")
                sc.tl.leiden(
                    adata,
                    resolution=resolution,
                    neighbors_key=neighbors_key,
                    key_added=cluster_col,
                )
                cluster_columns.append(cluster_col)
                cluster_to_umap[cluster_col] = umap_key
                cluster_to_neighbors[cluster_col] = neighbors_key
                scan_rows.append(
                    {
                        "cluster_col": cluster_col,
                        "resolution": float(resolution),
                        "n_neighbors": int(n_neighbors),
                        "n_pcs": int(n_pcs),
                        "representation_key": rep_key,
                        "neighbors_key": neighbors_key,
                        "umap_key": umap_key,
                        "n_clusters": int(adata.obs[cluster_col].astype(str).nunique()),
                    }
                )

    adata.uns["cluster_scan"] = {
        "resolutions": [float(value) for value in resolutions],
        "n_neighbors": [int(value) for value in n_neighbors_values],
        "n_pcs": [int(value) for value in n_pcs_values],
        "min_dist": float(args.scanpy_min_dist),
        "cluster_columns": cluster_columns,
    }
    adata.uns["cluster_scan_umap_keys"] = cluster_to_umap
    adata.uns["cluster_scan_neighbors_keys"] = cluster_to_neighbors
    adata.uns["leiden_resolution_columns"] = {
        row["cluster_col"]: row["cluster_col"]
        for row in scan_rows
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scan_rows).to_csv(args.output_dir / "cluster_parameter_scan_summary.csv", index=False)
    return cluster_columns


def align_adata(reference: ad.AnnData, other: ad.AnnData, label: str) -> ad.AnnData:
    missing = reference.obs_names.difference(other.obs_names)
    if len(missing):
        raise ValueError(f"{label} is missing {len(missing)} patches present in representation AnnData.")
    return other[reference.obs_names].copy()


def channel_mean_dataframe(metrics_adata: ad.AnnData) -> pd.DataFrame:
    matrix = get_matrix(metrics_adata).astype(np.float32, copy=False)
    var = metrics_adata.var.copy()

    if {"metric", "channel"}.issubset(var.columns):
        if "is_channel_metric" in var.columns:
            is_channel_metric = var["is_channel_metric"].astype(bool)
        else:
            is_channel_metric = pd.Series(True, index=var.index)
        mask = (var["metric"].astype(str) == "mean_intensity_norm") & (
            is_channel_metric
        )
        columns = var.loc[mask, "channel"].astype(str).tolist()
        indices = np.flatnonzero(mask.to_numpy())
    else:
        var_names = pd.Index(metrics_adata.var_names.astype(str))
        mask = var_names.str.startswith("mean_intensity_norm_")
        mask_array = np.asarray(mask, dtype=bool)
        columns = [name.removeprefix("mean_intensity_norm_") for name in var_names[mask_array]]
        indices = np.flatnonzero(mask_array)

    if len(indices) == 0:
        raise ValueError("No mean_intensity_norm channel variables were found in the metrics AnnData.")

    df = pd.DataFrame(matrix[:, indices], index=metrics_adata.obs_names, columns=columns)
    df.columns = make_unique_columns(df.columns)
    return df


def make_unique_columns(columns: pd.Index | list[str]) -> list[str]:
    counts: dict[str, int] = {}
    output = []
    for column in columns:
        base = str(column)
        count = counts.get(base, 0)
        output.append(base if count == 0 else f"{base}_{count + 1}")
        counts[base] = count + 1
    return output


def categorical_palette(values: pd.Series) -> dict[str, tuple[float, float, float, float]]:
    categories = sorted(values.dropna().astype(str).unique(), key=natural_key)
    cmap_name = "tab20" if len(categories) <= 20 else "gist_ncar"
    cmap = plt.colormaps[cmap_name].resampled(max(len(categories), 1))
    return {category: cmap(idx) for idx, category in enumerate(categories)}


def plot_umap_categorical(
    adata: ad.AnnData,
    column: str,
    output_path: Path,
    title: str,
    umap_key: str = "X_umap",
    point_size: float = 3.0,
) -> None:
    coords = get_umap_coords(adata, umap_key)
    if coords is None:
        print(f"Skipping categorical UMAP because adata.obsm['{umap_key}'] is missing.")
        return
    values = adata.obs[column].astype(str)
    palette = categorical_palette(values)

    fig, ax = plt.subplots(figsize=(7, 6))
    for category, color in palette.items():
        mask = values == category
        ax.scatter(coords[mask, 0], coords[mask, 1], s=point_size, color=color, label=category, alpha=0.8, linewidths=0)
    ax.set_xlabel("UMAP1 (VICReg embedding)")
    ax.set_ylabel("UMAP2 (VICReg embedding)")
    ax.set_title(title)
    if len(palette) <= 30:
        ax.legend(
            title=column,
            markerscale=4,
            fontsize=7,
            title_fontsize=8,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_umap_continuous(
    adata: ad.AnnData,
    values: pd.Series,
    output_path: Path,
    title: str,
    umap_key: str = "X_umap",
    cmap: str = "viridis",
    point_size: float = 3.0,
) -> None:
    coords = get_umap_coords(adata, umap_key)
    if coords is None:
        print(f"Skipping continuous UMAP '{title}' because adata.obsm['{umap_key}'] is missing.")
        return
    values = values.reindex(adata.obs_names).astype(float)
    finite_values = values.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    vmax = float(np.nanquantile(finite_values, 0.95)) if finite_values.size else None

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values.to_numpy(),
        s=point_size,
        cmap=cmap,
        vmax=vmax,
        alpha=0.85,
        linewidths=0,
    )
    ax.set_xlabel("UMAP1 (VICReg embedding)")
    ax.set_ylabel("UMAP2 (VICReg embedding)")
    ax.set_title(title)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    if vmax is not None:
        colorbar.set_label("Value (colour capped at 95th percentile)")
    else:
        colorbar.set_label("Value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_heatmap(
    data: pd.DataFrame,
    output_path: Path,
    title: str,
    cmap: str = "viridis",
    center: float | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    if data.empty:
        return
    if figsize is None:
        figsize = (max(6, 0.35 * data.shape[1] + 2), max(4, 0.28 * data.shape[0] + 2))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data, cmap=cmap, center=center, ax=ax, linewidths=0.1, linecolor="white")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    clustermap_path = output_path.with_name(f"{output_path.stem}_clustermap{output_path.suffix}")
    try:
        cluster_grid = sns.clustermap(
            data,
            cmap=cmap,
            center=center,
            figsize=figsize,
            linewidths=0.1,
            linecolor="white",
        )
        cluster_grid.fig.suptitle(title)
        cluster_grid.fig.tight_layout()
        cluster_grid.fig.savefig(clustermap_path, dpi=220)
        plt.close(cluster_grid.fig)
    except Exception as exc:
        print(f"Failed to save clustermap for {output_path.name}: {exc}")


def plot_embedding_and_intensity_umaps(
    adata: ad.AnnData,
    cluster_col: str,
    intensity_df: pd.DataFrame,
    output_dir: Path,
    max_channels: int,
    umap_key: str,
) -> None:
    plot_umap_categorical(
        adata,
        cluster_col,
        output_dir / f"umap_{safe_filename(cluster_col)}.png",
        f"Patch UMAP coloured by {cluster_col}",
        umap_key=umap_key,
        point_size=10.0
    )

    channels = list(intensity_df.columns)
    if max_channels > 0:
        channels = channels[:max_channels]
    for channel in tqdm(channels, desc="Writing intensity UMAPs"):
        plot_umap_continuous(
            adata,
            intensity_df[channel],
            output_dir / f"umap_mean_intensity_norm_{safe_filename(channel)}.png",
            f"Patch UMAP coloured by mean normalised intensity: {channel}",
            umap_key=umap_key,
            point_size=10.0
        )


def cluster_intensity_tables(
    adata: ad.AnnData,
    cluster_col: str,
    intensity_df: pd.DataFrame,
    output_dirs: dict[str, Path],
) -> None:
    clusters = adata.obs[cluster_col].astype(str)
    cluster_mean = intensity_df.groupby(clusters).mean()
    cluster_mean = cluster_mean.loc[sorted(cluster_mean.index, key=natural_key)]
    cluster_mean.to_csv(output_dirs["tables"] / "cluster_mean_channel_intensity.csv")

    channel_std = cluster_mean.std(axis=0).replace(0, np.nan)
    cluster_z = (cluster_mean - cluster_mean.mean(axis=0)) / channel_std
    cluster_z = cluster_z.fillna(0.0)
    cluster_z.to_csv(output_dirs["tables"] / "cluster_mean_channel_intensity_zscore.csv")

    save_heatmap(
        cluster_mean,
        output_dirs["heatmaps"] / "cluster_mean_channel_intensity.png",
        "Mean normalised channel intensity by cluster (cluster rows, marker columns)",
        cmap="mako",
    )
    save_heatmap(
        cluster_z,
        output_dirs["heatmaps"] / "cluster_mean_channel_intensity_zscore.png",
        "Relative marker enrichment by cluster (z-scored cluster means)",
        cmap="vlag",
        center=0,
    )


def plot_qc(
    adata: ad.AnnData,
    cluster_col: str,
    output_dirs: dict[str, Path],
    umap_key: str,
) -> None:
    cluster_counts = adata.obs[cluster_col].astype(str).value_counts().sort_index(key=lambda idx: idx.map(str))
    fig, ax = plt.subplots(figsize=(max(6, 0.28 * len(cluster_counts) + 2), 4))
    cluster_counts.plot(kind="bar", ax=ax, color="#4c78a8")
    ax.set_xlabel(f"{cluster_col} cluster")
    ax.set_ylabel("Patch count")
    ax.set_title(f"Retained patch count by {cluster_col} cluster")
    fig.tight_layout()
    fig.savefig(output_dirs["qc"] / "patch_count_by_cluster.png", dpi=220)
    plt.close(fig)

    for column in ["patch_mean_signal", "tissue_subpatch_fraction", "mean_subpatch_signal", "original_embedding_norm"]:
        if column in adata.obs:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(adata.obs[column].astype(float), bins=60, ax=ax, color="#4c78a8")
            ax.set_xlabel(column)
            ax.set_ylabel("Patch count")
            ax.set_title(f"Patch QC distribution: {column}")
            fig.tight_layout()
            fig.savefig(output_dirs["qc"] / f"hist_{safe_filename(column)}.png", dpi=220)
            plt.close(fig)

            if get_umap_coords(adata, umap_key) is not None:
                plot_umap_continuous(
                    adata,
                    adata.obs[column].astype(float),
                    output_dirs["umaps"] / f"umap_qc_{safe_filename(column)}.png",
                    f"Patch UMAP coloured by QC metric: {column}",
                    umap_key=umap_key,
                    point_size=10.0
                )

    if "roi" in adata.obs:
        roi_counts = adata.obs["roi"].astype(str).value_counts().head(60)
        fig, ax = plt.subplots(figsize=(max(8, 0.18 * len(roi_counts) + 2), 4))
        roi_counts.plot(kind="bar", ax=ax, color="#72b7b2")
        ax.set_xlabel("ROI")
        ax.set_ylabel("Patch count")
        ax.set_title("Patch count by ROI")
        fig.tight_layout()
        fig.savefig(output_dirs["qc"] / "patch_count_by_roi_top60.png", dpi=220)
        plt.close(fig)


def plot_cluster_composition(adata: ad.AnnData, cluster_col: str, output_dirs: dict[str, Path]) -> None:
    if "roi" not in adata.obs:
        return
    composition = pd.crosstab(adata.obs["roi"].astype(str), adata.obs[cluster_col].astype(str), normalize="index")
    composition = composition.reindex(columns=sorted(composition.columns, key=natural_key))
    composition.to_csv(output_dirs["tables"] / "roi_cluster_composition.csv")

    if composition.shape[0] <= 80:
        save_heatmap(
            composition,
            output_dirs["heatmaps"] / "roi_cluster_composition_heatmap.png",
            "ROI cluster composition",
            cmap="mako",
            figsize=(max(6, 0.28 * composition.shape[1] + 2), max(5, 0.16 * composition.shape[0] + 2)),
        )


def cluster_label_mapping(labels: pd.Series) -> pd.DataFrame:
    categories = sorted(labels.dropna().astype(str).unique(), key=natural_key)
    return pd.DataFrame(
        {
            "cluster": categories,
            "mask_value": np.arange(1, len(categories) + 1, dtype=int),
        }
    )


def make_cluster_colors(mapping: pd.DataFrame) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.colormaps["tab20"].resampled(max(len(mapping), 1)) if len(mapping) <= 20 else plt.colormaps["gist_ncar"].resampled(len(mapping))
    return {row.cluster: cmap(idx) for idx, row in enumerate(mapping.itertuples(index=False))}


def write_spatial_cluster_maps(
    adata: ad.AnnData,
    cluster_col: str,
    output_dirs: dict[str, Path],
    max_roi_maps: int,
) -> None:
    required = ["roi", "row_start", "row_end", "col_start", "col_end", cluster_col]
    require_columns(adata.obs, required, "representation AnnData obs")

    mapping = cluster_label_mapping(adata.obs[cluster_col])
    mapping.to_csv(output_dirs["tables"] / "cluster_label_mask_mapping.csv", index=False)
    value_lookup = dict(zip(mapping["cluster"], mapping["mask_value"]))
    color_lookup = make_cluster_colors(mapping)

    rois = sorted(adata.obs["roi"].astype(str).unique())
    if max_roi_maps > 0:
        rois = rois[:max_roi_maps]

    legend_handles = [
        Patch(facecolor=color_lookup[row.cluster], label=f"{row.cluster} ({row.mask_value})")
        for row in mapping.itertuples(index=False)
    ]

    for roi in tqdm(rois, desc="Writing ROI cluster maps"):
        roi_obs = adata.obs[adata.obs["roi"].astype(str) == roi].copy()
        if {"roi_height_px", "roi_width_px"}.issubset(roi_obs.columns):
            height = int(roi_obs["roi_height_px"].astype(float).max())
            width = int(roi_obs["roi_width_px"].astype(float).max())
        else:
            height = int(roi_obs["row_end"].astype(int).max())
            width = int(roi_obs["col_end"].astype(int).max())
        mask = np.zeros((height, width), dtype=np.uint16)

        for _, row in roi_obs.iterrows():
            cluster = str(row[cluster_col])
            value = int(value_lookup[cluster])
            row_start = int(row["row_start"])
            row_end = int(row["row_end"])
            col_start = int(row["col_start"])
            col_end = int(row["col_end"])
            mask[row_start:row_end, col_start:col_end] = value

        tifffile.imwrite(output_dirs["masks"] / f"{safe_filename(roi)}__cluster_labels.tiff", mask)

        cmap_colors = [(0, 0, 0, 1)]
        for cluster in mapping["cluster"]:
            cmap_colors.append(color_lookup[cluster])
        cmap = ListedColormap(cmap_colors)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(mask, cmap=cmap, interpolation="nearest", vmin=0, vmax=len(cmap_colors) - 1)
        ax.set_title(f"{roi}: back-gated {cluster_col}")
        ax.set_axis_off()
        if len(legend_handles) <= 30:
            ax.legend(handles=legend_handles, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(output_dirs["spatial"] / f"{safe_filename(roi)}__cluster_map.png", dpi=220)
        plt.close(fig)


def fallback_de_table(intensity_df: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    global_mean = intensity_df.mean(axis=0)
    global_std = intensity_df.std(axis=0).replace(0, np.nan)
    rows = []
    for cluster in sorted(clusters.astype(str).unique(), key=natural_key):
        cluster_values = intensity_df.loc[clusters.astype(str) == cluster]
        scores = ((cluster_values.mean(axis=0) - global_mean) / global_std).fillna(0.0)
        for marker, score in scores.sort_values(ascending=False).items():
            rows.append(
                {
                    "group": cluster,
                    "names": marker,
                    "scores": float(score),
                    "logfoldchanges": np.nan,
                    "pvals": np.nan,
                    "pvals_adj": np.nan,
                }
            )
    return pd.DataFrame(rows)


def rank_cluster_markers(
    intensity_df: pd.DataFrame,
    clusters: pd.Series,
    output_dirs: dict[str, Path],
    method: str,
    output_name: str,
) -> pd.DataFrame:
    clusters = clusters.astype(str).reindex(intensity_df.index)
    try:
        import scanpy as sc

        tmp = ad.AnnData(
            X=intensity_df.to_numpy(dtype=np.float32),
            obs=pd.DataFrame({"cluster": pd.Categorical(clusters)}, index=intensity_df.index),
            var=pd.DataFrame(index=intensity_df.columns),
        )
        sc.tl.rank_genes_groups(tmp, groupby="cluster", method=method)
        de = sc.get.rank_genes_groups_df(tmp, group=None)
    except Exception as exc:
        print(f"Scanpy differential marker ranking failed; using z-score fallback. Reason: {exc}")
        de = fallback_de_table(intensity_df, clusters)

    de.to_csv(output_dirs["tables"] / output_name, index=False)
    return de


def permutation_channel_dataframe(perm_adata: ad.AnnData, perturbation_type: str) -> pd.DataFrame:
    var = perm_adata.var.copy()
    if not {"perturbation_type", "channel"}.issubset(var.columns):
        return pd.DataFrame(index=perm_adata.obs_names)

    matrix = get_matrix(perm_adata).astype(np.float32, copy=False)
    records = []
    type_mask = var["perturbation_type"].astype(str) == perturbation_type
    channels = sorted(var.loc[type_mask, "channel"].astype(str).unique(), key=natural_key)

    for channel in channels:
        if channel == "all_channels":
            continue
        mask = type_mask & (var["channel"].astype(str) == channel)
        indices = np.flatnonzero(mask.to_numpy())
        if len(indices) == 0:
            continue
        records.append(pd.Series(np.nanmean(matrix[:, indices], axis=1), index=perm_adata.obs_names, name=channel))

    if not records:
        return pd.DataFrame(index=perm_adata.obs_names)
    return pd.concat(records, axis=1)


def choose_gallery_marker_de_table(
    intensity_df: pd.DataFrame,
    clusters: pd.Series,
    output_dirs: dict[str, Path],
    args: argparse.Namespace,
    perm_adata: ad.AnnData | None,
) -> tuple[pd.DataFrame, str]:
    if args.gallery_marker_source == "permutation" and perm_adata is None:
        raise ValueError("Permutation marker source was requested, but the permutation AnnData was not found.")

    if args.gallery_marker_source in {"auto", "permutation"} and perm_adata is not None:
        candidate_types = [args.gallery_permutation_type]
        fallback_type = "shuffle_channel" if args.gallery_permutation_type == "zero_channel" else "zero_channel"
        candidate_types.append(fallback_type)

        for perturbation_type in candidate_types:
            impact_df = permutation_channel_dataframe(perm_adata, perturbation_type)
            shared_channels = [channel for channel in intensity_df.columns if channel in impact_df.columns]
            impact_df = impact_df.loc[:, shared_channels]
            if impact_df.empty:
                continue

            impact_df.to_csv(output_dirs["tables"] / f"gallery_marker_{perturbation_type}_patch_scores.csv.gz")
            de = rank_cluster_markers(
                intensity_df=impact_df,
                clusters=clusters,
                output_dirs=output_dirs,
                method=args.de_method,
                output_name=f"cluster_marker_differential_{perturbation_type}_impact.csv",
            )
            return de, f"{perturbation_type} permutation impact"

        if args.gallery_marker_source == "permutation":
            raise ValueError("Permutation marker source was requested, but no channel-level permutation scores were found.")

    de = rank_cluster_markers(
        intensity_df=intensity_df,
        clusters=clusters,
        output_dirs=output_dirs,
        method=args.de_method,
        output_name="cluster_marker_differential_intensity.csv",
    )
    return de, "mean intensity"


def top_markers_for_cluster(
    de_table: pd.DataFrame,
    cluster: str,
    fallback_intensities: pd.DataFrame,
    clusters: pd.Series,
    n_markers: int,
) -> list[str]:
    cluster_de = de_table[de_table["group"].astype(str) == str(cluster)].copy()
    if "scores" in cluster_de:
        cluster_de = cluster_de.sort_values("scores", ascending=False)
    markers = [str(marker) for marker in cluster_de["names"].head(n_markers).tolist() if str(marker) in fallback_intensities.columns]

    if len(markers) < n_markers:
        cluster_values = fallback_intensities.loc[clusters.astype(str) == str(cluster)]
        for marker in cluster_values.mean(axis=0).sort_values(ascending=False).index:
            marker = str(marker)
            if marker not in markers:
                markers.append(marker)
            if len(markers) >= n_markers:
                break
    return markers[:n_markers]


def render_patch_rgb(
    patch: np.ndarray,
    channel_indices: list[int],
    auto_contrast: bool,
    contrast_percentile: float,
    vmax: float,
) -> np.ndarray:
    rgb = np.zeros((*patch.shape[:2], 3), dtype=np.float32)
    for out_idx, channel_idx in enumerate(channel_indices[:3]):
        plane = patch[:, :, channel_idx].astype(np.float32, copy=False)
        if auto_contrast:
            high = np.percentile(plane, contrast_percentile)
            if high > 0:
                plane = plane / high
        elif vmax > 0:
            plane = plane / vmax
        rgb[:, :, out_idx] = plane
    return np.clip(rgb, 0.0, 1.0)


def render_patch_channel(
    patch: np.ndarray,
    channel_idx: int,
    auto_contrast: bool,
    contrast_percentile: float,
    vmax: float,
) -> np.ndarray:
    plane = patch[:, :, channel_idx].astype(np.float32, copy=False)
    if auto_contrast:
        high = np.percentile(plane, contrast_percentile)
        if high > 0:
            plane = plane / high
    elif vmax > 0:
        plane = plane / vmax
    return np.clip(plane, 0.0, 1.0)


def select_unique_roi_patches(
    adata: ad.AnnData,
    patch_ids: pd.Index,
    scores: pd.Series,
    max_patches: int,
    rng: np.random.Generator,
) -> list[str]:
    if "roi" not in adata.obs:
        if scores.notna().any():
            return scores.sort_values(ascending=False).head(max_patches).index.tolist()
        return rng.choice(patch_ids, size=min(max_patches, len(patch_ids)), replace=False).tolist()

    roi_by_patch = adata.obs.loc[patch_ids, "roi"].astype(str)
    scores = scores.reindex(patch_ids)
    if scores.notna().any():
        ordered_patch_ids = scores.sort_values(ascending=False, na_position="last").index.tolist()
    else:
        ordered_patch_ids = rng.permutation(np.asarray(patch_ids, dtype=object)).tolist()

    selected = []
    seen_rois = set()
    for patch_id in ordered_patch_ids:
        roi = roi_by_patch.loc[patch_id]
        if roi in seen_rois:
            continue
        selected.append(patch_id)
        seen_rois.add(roi)
        if len(selected) >= max_patches:
            break
    return selected


def write_rgb_patch_gallery(
    adata: ad.AnnData,
    selected: list[str],
    marker_indices: list[int],
    markers: list[str],
    cluster: str,
    scores: pd.Series,
    output_dirs: dict[str, Path],
    args: argparse.Namespace,
    gallery_records: list[dict[str, object]],
) -> None:
    n_cols = max(args.gallery_cols, 1)
    n_rows = int(math.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.35 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()

    rgb_markers = markers[:3]
    for idx, patch_id in enumerate(selected):
        row = adata.obs.loc[patch_id]
        patch = np.load(str(row["patch_path"])).astype(np.float32, copy=False)
        rgb = render_patch_rgb(
            patch,
            marker_indices,
            auto_contrast=args.gallery_auto_contrast,
            contrast_percentile=args.gallery_contrast_percentile,
            vmax=args.gallery_vmax,
        )
        ax = axes.ravel()[idx]
        ax.imshow(rgb)
        ax.set_title(f"{row.get('roi', '')}\n{patch_id}", fontsize=6)
        ax.set_axis_off()
        gallery_records.append(
            {
                "cluster": cluster,
                "gallery_mode": "rgb",
                "roi": row.get("roi", ""),
                "patch_id": patch_id,
                "patch_path": row["patch_path"],
                "markers": ",".join(rgb_markers),
                "markers_rgb": ",".join(rgb_markers),
                "selection_score": float(scores.get(patch_id, np.nan)),
            }
        )

    fig.suptitle(f"Cluster {cluster}: RGB markers {', '.join(rgb_markers)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dirs["galleries"] / f"cluster_{safe_filename(cluster)}__patch_gallery.png", dpi=220)
    plt.close(fig)


def write_split_channel_patch_gallery(
    adata: ad.AnnData,
    selected: list[str],
    marker_indices: list[int],
    markers: list[str],
    cluster: str,
    scores: pd.Series,
    output_dirs: dict[str, Path],
    args: argparse.Namespace,
    gallery_records: list[dict[str, object]],
) -> None:
    n_rows = len(selected)
    n_cols = len(marker_indices) + 1
    fig_width = max(4.0, 1.8 * n_cols + 1.2)
    fig_height = max(2.2, 1.8 * n_rows + 0.8)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

    rgb_markers = markers[:3]
    rgb_marker_indices = marker_indices[:3]
    for row_idx, patch_id in enumerate(selected):
        row = adata.obs.loc[patch_id]
        patch = np.load(str(row["patch_path"])).astype(np.float32, copy=False)
        row_label = f"{row.get('roi', '')}\n{patch_id}"

        ax = axes[row_idx, 0]
        rgb = render_patch_rgb(
            patch,
            rgb_marker_indices,
            auto_contrast=args.gallery_auto_contrast,
            contrast_percentile=args.gallery_contrast_percentile,
            vmax=args.gallery_vmax,
        )
        ax.imshow(rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if row_idx == 0:
            ax.set_title("Merged RGB", fontsize=7)
        ax.set_ylabel(row_label, rotation=0, ha="right", va="center", fontsize=6, labelpad=42)

        for col_idx, (marker, channel_idx) in enumerate(zip(markers, marker_indices)):
            ax = axes[row_idx, col_idx + 1]
            image = render_patch_channel(
                patch,
                channel_idx,
                auto_contrast=args.gallery_auto_contrast,
                contrast_percentile=args.gallery_contrast_percentile,
                vmax=args.gallery_vmax,
            )
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax.set_title(marker, fontsize=7)

        gallery_records.append(
            {
                "cluster": cluster,
                "gallery_mode": "split_channel",
                "roi": row.get("roi", ""),
                "patch_id": patch_id,
                "patch_path": row["patch_path"],
                "markers": ",".join(markers),
                "markers_rgb": ",".join(rgb_markers),
                "selection_score": float(scores.get(patch_id, np.nan)),
            }
        )

    fig.suptitle(f"Cluster {cluster}: split channels with merged RGB", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dirs["galleries"] / f"cluster_{safe_filename(cluster)}__patch_gallery.png", dpi=220)
    plt.close(fig)


def write_patch_galleries(
    adata: ad.AnnData,
    cluster_col: str,
    intensity_df: pd.DataFrame,
    de_table: pd.DataFrame,
    output_dirs: dict[str, Path],
    channels: list[str],
    args: argparse.Namespace,
) -> None:
    if args.patches_per_cluster <= 0:
        return
    if "patch_path" not in adata.obs:
        print("Skipping patch galleries because adata.obs['patch_path'] is missing.")
        return

    channel_to_index = {channel: idx for idx, channel in enumerate(channels)}
    clusters = adata.obs[cluster_col].astype(str)
    rng = np.random.default_rng(args.seed)
    gallery_records = []

    for cluster in tqdm(sorted(clusters.unique(), key=natural_key), desc="Writing patch galleries"):
        n_requested_markers = max(args.gallery_top_markers, 1)
        markers = top_markers_for_cluster(
            de_table,
            cluster,
            intensity_df,
            clusters,
            n_requested_markers,
        )
        marker_pairs = [
            (marker, channel_to_index[marker])
            for marker in markers
            if marker in channel_to_index
        ]
        if not marker_pairs:
            fallback_count = min(n_requested_markers, len(channels))
            marker_pairs = [(channels[idx], idx) for idx in range(fallback_count)]
        if not marker_pairs:
            continue
        markers = [marker for marker, _ in marker_pairs]
        marker_indices = [idx for _, idx in marker_pairs]

        cluster_patch_ids = adata.obs_names[clusters == cluster]
        score_markers = [marker for marker in markers if marker in intensity_df.columns]
        if score_markers:
            scores = intensity_df.loc[cluster_patch_ids, score_markers].mean(axis=1)
        else:
            scores = pd.Series(np.nan, index=cluster_patch_ids)
        selected = select_unique_roi_patches(
            adata,
            cluster_patch_ids,
            scores,
            args.patches_per_cluster,
            rng,
        )
        if not selected:
            continue

        if args.split_channel_gallery:
            write_split_channel_patch_gallery(
                adata=adata,
                selected=selected,
                marker_indices=marker_indices,
                markers=markers,
                cluster=cluster,
                scores=scores,
                output_dirs=output_dirs,
                args=args,
                gallery_records=gallery_records,
            )
        else:
            write_rgb_patch_gallery(
                adata=adata,
                selected=selected,
                marker_indices=marker_indices,
                markers=markers,
                cluster=cluster,
                scores=scores,
                output_dirs=output_dirs,
                args=args,
                gallery_records=gallery_records,
            )

    pd.DataFrame(gallery_records).to_csv(output_dirs["tables"] / "patch_gallery_index.csv", index=False)


def aggregate_permutation_scores(perm_adata: ad.AnnData) -> pd.DataFrame:
    distances = get_matrix(perm_adata).astype(np.float32, copy=False)
    var = perm_adata.var.copy()
    records = []

    for perturbation_type in sorted(var["perturbation_type"].astype(str).unique()):
        type_mask = var["perturbation_type"].astype(str) == perturbation_type
        for channel in sorted(var.loc[type_mask, "channel"].astype(str).unique(), key=natural_key):
            mask = type_mask & (var["channel"].astype(str) == channel)
            indices = np.flatnonzero(mask.to_numpy())
            if len(indices) == 0:
                continue
            column = f"{perturbation_type}__{channel}"
            values = np.nanmean(distances[:, indices], axis=1)
            records.append(pd.Series(values, index=perm_adata.obs_names, name=column))

    if not records:
        return pd.DataFrame(index=perm_adata.obs_names)
    return pd.concat(records, axis=1)


def plot_permutation_outputs(
    adata: ad.AnnData,
    perm_adata: ad.AnnData,
    cluster_col: str,
    output_dirs: dict[str, Path],
    max_umaps: int,
    umap_key: str,
) -> None:
    perm_adata = align_adata(adata, perm_adata, "permutation AnnData")
    perm_scores = aggregate_permutation_scores(perm_adata)
    if perm_scores.empty:
        return
    perm_scores.to_csv(output_dirs["tables"] / "permutation_patch_aggregated_scores.csv.gz")

    clusters = adata.obs[cluster_col].astype(str)
    cluster_means = perm_scores.groupby(clusters).mean()
    cluster_means = cluster_means.loc[sorted(cluster_means.index, key=natural_key)]
    cluster_means.to_csv(output_dirs["tables"] / "cluster_mean_permutation_cosine_distance.csv")

    for perturbation_type in ["zero_channel", "shuffle_channel"]:
        subset = cluster_means[[col for col in cluster_means.columns if col.startswith(f"{perturbation_type}__")]]
        if subset.empty:
            continue
        subset.columns = [col.split("__", 1)[1] for col in subset.columns]
        save_heatmap(
            subset,
            output_dirs["heatmaps"] / f"cluster_mean_permutation_{perturbation_type}.png",
            f"Mean cosine distance by cluster: {perturbation_type}",
            cmap="rocket",
        )

    all_channel_cols = [col for col in cluster_means.columns if col.endswith("__all_channels")]
    if all_channel_cols:
        all_channels = cluster_means[all_channel_cols].copy()
        all_channels.columns = [col.replace("__all_channels", "") for col in all_channels.columns]
        save_heatmap(
            all_channels,
            output_dirs["heatmaps"] / "cluster_mean_permutation_all_channels.png",
            "All-channel perturbation cosine distance by cluster",
            cmap="rocket",
        )

    if max_umaps > 0:
        ranked = perm_scores.mean(axis=0).sort_values(ascending=False).head(max_umaps)
        for column in tqdm(ranked.index, desc="Writing permutation UMAPs"):
            plot_umap_continuous(
                adata,
                perm_scores[column],
                output_dirs["umaps"] / f"umap_permutation_{safe_filename(column)}.png",
                f"Permutation cosine distance: {column}",
                umap_key=umap_key,
                cmap="magma",
                point_size=10.0
            )


def plot_normalisation_qc(report_dir: Path, output_dirs: dict[str, Path]) -> None:
    channel_report = report_dir / "normalisation_channel_report.csv"
    roi_report = report_dir / "normalisation_roi_channel_report.csv"
    if not channel_report.exists() and not roi_report.exists():
        print(f"No normalisation reports found in {report_dir}; skipping normalisation QC plots.")
        return

    if channel_report.exists():
        channel_df = pd.read_csv(channel_report)
        channel_df.to_csv(output_dirs["tables"] / "normalisation_channel_report.csv", index=False)
        if {"channel", "scale_value"}.issubset(channel_df.columns):
            fig, ax = plt.subplots(figsize=(max(7, 0.28 * len(channel_df) + 2), 4))
            sns.barplot(data=channel_df, x="channel", y="scale_value", ax=ax, color="#4c78a8")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_title("Normalisation scale value by channel")
            fig.tight_layout()
            fig.savefig(output_dirs["qc"] / "normalisation_scale_values.png", dpi=220)
            plt.close(fig)

        if {"channel", "n_present_rois", "n_rois"}.issubset(channel_df.columns):
            channel_df["present_roi_fraction"] = channel_df["n_present_rois"] / channel_df["n_rois"].replace(0, np.nan)
            fig, ax = plt.subplots(figsize=(max(7, 0.28 * len(channel_df) + 2), 4))
            sns.barplot(data=channel_df, x="channel", y="present_roi_fraction", ax=ax, color="#72b7b2")
            ax.set_ylim(0, 1)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_title("Fraction of ROIs with channel present")
            fig.tight_layout()
            fig.savefig(output_dirs["qc"] / "normalisation_present_roi_fraction.png", dpi=220)
            plt.close(fig)

    if roi_report.exists():
        roi_df = pd.read_csv(roi_report)
        roi_df.to_csv(output_dirs["tables"] / "normalisation_roi_channel_report.csv", index=False)
        if {"channel", "background_value"}.issubset(roi_df.columns):
            fig, ax = plt.subplots(figsize=(max(7, 0.28 * roi_df["channel"].nunique() + 2), 4))
            sns.boxplot(data=roi_df, x="channel", y="background_value", ax=ax, color="#bab0ac", fliersize=1)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_title("Background values by channel")
            fig.tight_layout()
            fig.savefig(output_dirs["qc"] / "normalisation_background_values.png", dpi=220)
            plt.close(fig)


def write_run_summary(
    args: argparse.Namespace,
    output_dir: Path,
    cluster_col: str,
    cluster_cols: list[str] | None = None,
) -> None:
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["resolved_cluster_col"] = cluster_col
    if cluster_cols is not None:
        config["resolved_cluster_cols"] = cluster_cols
    with open(output_dir / "visualisation_run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def run_visualisation_for_cluster(
    args: argparse.Namespace,
    adata: ad.AnnData,
    intensity_df: pd.DataFrame,
    channels: list[str],
    perm_adata: ad.AnnData | None,
    cluster_col: str,
    output_dir: Path,
    all_cluster_cols: list[str],
) -> dict[str, object]:
    output_dirs = ensure_dirs(output_dir)
    umap_key = cluster_col_umap_key(adata, cluster_col)

    print(f"Cluster column: {cluster_col}")
    print(f"UMAP key: {umap_key}")
    print(f"Report output dir: {output_dir}")
    plot_embedding_and_intensity_umaps(adata, cluster_col, intensity_df, output_dirs["umaps"], args.max_umap_channels, umap_key)
    cluster_intensity_tables(adata, cluster_col, intensity_df, output_dirs)
    plot_qc(adata, cluster_col, output_dirs, umap_key)
    plot_cluster_composition(adata, cluster_col, output_dirs)
    if args.spatial_cluster_maps:
        write_spatial_cluster_maps(adata, cluster_col, output_dirs, args.max_roi_maps)
    else:
        print("Skipping spatial back-gated cluster maps and TIFF label masks.")

    de_table, gallery_source = choose_gallery_marker_de_table(
        intensity_df=intensity_df,
        clusters=adata.obs[cluster_col].astype(str),
        output_dirs=output_dirs,
        args=args,
        perm_adata=perm_adata,
    )
    print(f"Patch gallery marker source: {gallery_source}")
    write_patch_galleries(
        adata=adata,
        cluster_col=cluster_col,
        intensity_df=intensity_df,
        de_table=de_table,
        output_dirs=output_dirs,
        channels=channels,
        args=args,
    )

    if perm_adata is not None:
        plot_permutation_outputs(adata, perm_adata, cluster_col, output_dirs, args.max_permutation_umaps, umap_key)
    else:
        print(f"Permutation AnnData not found at {args.permutation_adata}; skipping permutation plots.")

    plot_normalisation_qc(args.normalisation_report_dir, output_dirs)
    write_run_summary(args, output_dir, cluster_col, cluster_cols=all_cluster_cols)
    return {
        "cluster_col": cluster_col,
        "output_dir": str(output_dir),
        "n_clusters": int(adata.obs[cluster_col].astype(str).nunique()),
    }


def main() -> None:
    args = resolve_paths(parse_args())
    if not args.gallery_auto_contrast and args.gallery_vmax <= 0:
        raise ValueError("--gallery-vmax must be positive when gallery auto contrast is disabled.")
    np.random.seed(args.seed)

    adata = ad.read_h5ad(args.representation_adata)
    generated_cluster_cols: list[str] = []
    if args.run_cluster_scan:
        generated_cluster_cols = run_cluster_parameter_scan(args, adata)
        if args.write_clustered_adata:
            args.clustered_adata_output.parent.mkdir(parents=True, exist_ok=True)
            print(f"Writing clustered AnnData to {args.clustered_adata_output}")
            adata.write_h5ad(args.clustered_adata_output)
    else:
        print("Skipping clustering parameter scan; using existing adata.obs cluster columns.")

    metrics_adata = ad.read_h5ad(args.metrics_adata)
    metrics_adata = align_adata(adata, metrics_adata, "metrics AnnData")
    if args.run_cluster_scan and args.cluster_col_search == "leiden" and generated_cluster_cols:
        cluster_cols = generated_cluster_cols
    else:
        cluster_cols = resolve_cluster_cols(adata, args)
    intensity_df = channel_mean_dataframe(metrics_adata).loc[adata.obs_names]

    channels = [str(channel) for channel in adata.uns.get("channel_names", intensity_df.columns.tolist())]
    if len(channels) != len(intensity_df.columns):
        channels = intensity_df.columns.tolist()

    print(f"Representation AnnData: {args.representation_adata}")
    print(f"Metrics AnnData: {args.metrics_adata}")
    print(f"Output dir: {args.output_dir}")
    print(f"Cluster columns: {cluster_cols}")
    print(f"Patches: {adata.n_obs}")
    print(f"Channels: {len(intensity_df.columns)}")

    perm_adata = None
    if args.permutation_adata.exists():
        perm_adata = align_adata(adata, ad.read_h5ad(args.permutation_adata), "permutation AnnData")

    multi_cluster_mode = len(cluster_cols) > 1
    summaries = []
    for cluster_col in cluster_cols:
        report_output_dir = args.output_dir / safe_filename(cluster_col) if multi_cluster_mode else args.output_dir
        summaries.append(
            run_visualisation_for_cluster(
                args=args,
                adata=adata,
                intensity_df=intensity_df,
                channels=channels,
                perm_adata=perm_adata,
                cluster_col=cluster_col,
                output_dir=report_output_dir,
                all_cluster_cols=cluster_cols,
            )
        )

    if multi_cluster_mode:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(args.output_dir / "all_cluster_visualisation_summary.csv", index=False)
    print(f"Saved visualisations to {args.output_dir}")


if __name__ == "__main__":
    main()
