"""
Pairwise spatial analysis pipeline stage.

This stage orchestrates three pairwise population analyses from a single AnnData input:
1. Squidpy neighborhood enrichment via `spatial.squidpy_subregion_interactions`
2. Nearest-population distance bootstrap via `distance_analysis.bootstrap_nearest_population_distances_all_rois`
3. Pair-correlation function (PCF) via `pcf.run_paircorrelation_at_distance`

For each analysis it:
- saves raw outputs and tidy long-form tables
- merges ROI/subregion metadata from AnnData.obs
- creates population x population matrix plots (clustermap with heatmap fallback on NaN issues)
- optionally creates source-target pair bar plots
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

import SpatialBiologyToolkit.distance_analysis as sbt_distance
import SpatialBiologyToolkit.pcf as sbt_pcf
import SpatialBiologyToolkit.spatial as sbt_spatial
from .config_and_utils import (
    GeneralConfig,
    PairwiseSpatialConfig,
    cleanstring,
    coalesce_config_list,
    coalesce_config_text,
    filter_config_for_dataclass,
    load_pipeline_anndata,
    process_config_with_overrides,
    save_pipeline_anndata,
    setup_logging,
)


def _resolve_input_adata_path(
    general_config: GeneralConfig, pairwise_config: PairwiseSpatialConfig
) -> Path:
    override = pairwise_config.input_adata_path
    if override:
        return Path(override)
    return Path(general_config.anndata_path)


def _ensure_extension(ext: str) -> str:
    if not ext:
        return ".png"
    return ext if ext.startswith(".") else f".{ext}"


def _reversed_cmap_name(cmap: str) -> str:
    """Return a reversed matplotlib colormap name, preserving already-reversed names."""
    text = str(cmap).strip()
    if not text:
        return text
    return text if text.endswith("_r") else f"{text}_r"


def _figsize(values: Sequence[float], fallback: Tuple[float, float]) -> Tuple[float, float]:
    if not values or len(values) < 2:
        return fallback
    return float(values[0]), float(values[1])


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _resolve_metadata_columns(
    adata: ad.AnnData, pairwise_config: PairwiseSpatialConfig
) -> List[str]:
    if pairwise_config.include_all_obs_metadata:
        columns = list(adata.obs.columns)
    else:
        configured = pairwise_config.metadata_obs_columns or []
        columns = [c for c in configured if c in adata.obs.columns]

    must_have = [
        pairwise_config.roi_obs,
        pairwise_config.population_obs,
        pairwise_config.groupby_obs,
        pairwise_config.master_index_obs,
        pairwise_config.x_coord_obs,
        pairwise_config.y_coord_obs,
    ]
    must_have = [c for c in must_have if c and c in adata.obs.columns]
    return _dedupe_keep_order([*columns, *must_have])


def _build_key_metadata_table(
    adata: ad.AnnData, key_col: str, metadata_cols: Sequence[str]
) -> pd.DataFrame:
    if key_col not in adata.obs.columns:
        logging.warning("Metadata key column '%s' not found in AnnData.obs.", key_col)
        return pd.DataFrame()

    keep_cols = [key_col] + [c for c in metadata_cols if c in adata.obs.columns and c != key_col]
    obs = adata.obs[keep_cols].copy()

    for col in keep_cols:
        if col == key_col:
            continue
        variability = obs.groupby(key_col, observed=True)[col].nunique(dropna=False)
        n_conflicting = int((variability > 1).sum())
        if n_conflicting > 0:
            logging.warning(
                "Metadata column '%s' has multiple values for %d '%s' groups; using first observed value.",
                col,
                n_conflicting,
                key_col,
            )

    return obs.drop_duplicates(subset=[key_col]).set_index(key_col)


def _population_order(adata: ad.AnnData, population_obs: str) -> List[str]:
    if population_obs not in adata.obs.columns:
        raise KeyError(f"Population column '{population_obs}' not found in AnnData.obs.")

    series = adata.obs[population_obs]
    if pd.api.types.is_categorical_dtype(series):
        return [str(x) for x in series.cat.categories if pd.notna(x)]

    return [str(x) for x in pd.unique(series.dropna())]


def _population_color_map(
    adata: ad.AnnData, population_obs: str, ordered_pops: Sequence[str]
) -> Dict[str, str]:
    key = f"{population_obs}_colors"
    if key in adata.uns:
        colors = list(adata.uns[key])
        if len(colors) >= len(ordered_pops):
            return {str(pop): str(colors[i]) for i, pop in enumerate(ordered_pops)}
        logging.warning(
            "AnnData.uns['%s'] has fewer colors (%d) than populations (%d). Falling back to tab20.",
            key,
            len(colors),
            len(ordered_pops),
        )

    palette = sns.color_palette("tab20", n_colors=max(1, len(ordered_pops))).as_hex()
    return {str(pop): str(palette[i]) for i, pop in enumerate(ordered_pops)}


def _label_colors(labels: Sequence[str], color_map: Dict[str, str]) -> pd.Series:
    return pd.Series(
        [color_map.get(str(label), "lightgray") for label in labels],
        index=list(labels),
        dtype=object,
    )


def _normalise_population_pairs(
    population_pairs: Dict[str, Any],
    population_obs: str,
    available_pops: Sequence[str],
) -> Dict[str, List[str]]:
    if not population_pairs:
        return {}

    mapping: Any = population_pairs
    if (
        isinstance(population_pairs, dict)
        and population_obs in population_pairs
        and isinstance(population_pairs[population_obs], dict)
    ):
        mapping = population_pairs[population_obs]

    if not isinstance(mapping, dict):
        logging.warning(
            "pairwise_spatial.population_pairs should be a dictionary; received %s. Skipping pair bar plots.",
            type(mapping).__name__,
        )
        return {}

    available = [str(p) for p in available_pops]
    available_set = set(available)
    available_lower_to_exact: Dict[str, List[str]] = {}
    for pop in available:
        available_lower_to_exact.setdefault(pop.lower(), []).append(pop)

    def _resolve_target_spec(source_name: str, target_spec: Any) -> List[str]:
        if target_spec is None:
            return []
        if isinstance(target_spec, (list, tuple, set, pd.Index, np.ndarray)):
            raw_tokens = [t for t in target_spec if t is not None]
        else:
            raw_tokens = [target_spec]

        resolved: List[str] = []
        seen: set[str] = set()
        for token_raw in raw_tokens:
            token = str(token_raw).strip()
            if not token:
                continue

            token_upper = token.upper()
            matches: List[str] = []
            if token_upper == "ALL":
                matches = list(available)
            elif token_upper == "ALL_OTHERS":
                matches = [p for p in available if p != source_name]
            elif token_upper.startswith("MATCH_"):
                pattern = token[6:]
                if pattern:
                    pattern_lower = pattern.lower()
                    matches = [p for p in available if pattern_lower in p.lower()]
            elif token_upper.startswith("NOT_"):
                pattern = token[4:]
                if pattern:
                    pattern_lower = pattern.lower()
                    matches = [p for p in available if pattern_lower not in p.lower()]
            else:
                if token in available_set:
                    matches = [token]
                else:
                    ci = available_lower_to_exact.get(token.lower(), [])
                    if len(ci) == 1:
                        matches = [ci[0]]

            if not matches:
                logging.warning(
                    "population_pairs target spec '%s' for source '%s' matched no populations.",
                    token,
                    source_name,
                )
                continue

            for match in matches:
                if match not in seen:
                    seen.add(match)
                    resolved.append(match)
        return resolved

    normalised: Dict[str, List[str]] = {}
    for source, targets in mapping.items():
        source_name = str(source).strip()
        if not source_name or targets is None:
            continue

        if source_name not in available_set:
            ci = available_lower_to_exact.get(source_name.lower(), [])
            if len(ci) == 1:
                source_name = ci[0]
            else:
                logging.warning(
                    "population_pairs source '%s' not found in available populations for obs '%s'. "
                    "Pair may produce no data.",
                    source_name,
                    population_obs,
                )

        target_list = _resolve_target_spec(source_name, targets)
        if target_list:
            normalised[source_name] = target_list
    return normalised


def _compute_limits(
    matrix: pd.DataFrame, percentile: float, center: Optional[float]
) -> Tuple[Optional[float], Optional[float]]:
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None

    pct = float(np.clip(percentile, 50.0, 100.0))
    if center is None:
        lo = np.nanpercentile(finite, max(0.0, 100.0 - pct))
        hi = np.nanpercentile(finite, pct)
    else:
        delta = np.abs(finite - center)
        max_delta = np.nanpercentile(delta, pct)
        if np.isclose(max_delta, 0):
            max_delta = float(np.nanmax(delta))
        if np.isclose(max_delta, 0):
            max_delta = 1.0
        lo = center - max_delta
        hi = center + max_delta

    if np.isclose(lo, hi):
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if np.isclose(lo, hi):
            lo -= 1.0
            hi += 1.0

    return float(lo), float(hi)


def _shared_limits_from_all(
    matrix: pd.DataFrame,
    *,
    enabled: bool,
    percentile: float,
    center: Optional[float],
    analysis: str,
    metric: str,
) -> Optional[Tuple[float, float]]:
    if not enabled:
        return None

    vmin, vmax = _compute_limits(matrix, percentile=percentile, center=center)
    if vmin is None or vmax is None:
        logging.warning(
            "Could not compute shared vmin/vmax for %s metric '%s'. Falling back to per-matrix limits.",
            analysis,
            metric,
        )
        return None
    return float(vmin), float(vmax)


def _ordered_matrix(matrix: pd.DataFrame, ordered_pops: Sequence[str]) -> pd.DataFrame:
    if matrix.empty:
        return matrix
    row_order = [p for p in ordered_pops if p in matrix.index]
    col_order = [p for p in ordered_pops if p in matrix.columns]
    if row_order:
        matrix = matrix.reindex(index=row_order)
    if col_order:
        matrix = matrix.reindex(columns=col_order)
    return matrix


def _force_show_all_tick_labels(
    ax: Any,
    *,
    x_rotation: float = 90.0,
    y_rotation: float = 0.0,
) -> None:
    """Force all x/y tick labels to render on matrix plots."""
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
    ax.tick_params(axis="y", which="both", left=True, right=False, labelleft=True)

    for label in ax.get_xticklabels():
        label.set_visible(True)
        label.set_rotation(x_rotation)
        label.set_ha("right")
    for label in ax.get_yticklabels():
        label.set_visible(True)
        label.set_rotation(y_rotation)


def _normalise_cbar_corner(value: Optional[str]) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"lower_right", "upper_left"}:
        return text
    logging.warning(
        "Invalid pairwise_matrices_cbar_corner='%s'. Using default 'lower_right'.",
        value,
    )
    return "lower_right"


def _place_colorbar_in_corner(
    cbar_ax: Any,
    reference_ax: Any,
    *,
    corner: str,
) -> None:
    if cbar_ax is None or reference_ax is None:
        return

    ref_box = reference_ax.get_position()
    cbar_width = max(0.012, ref_box.width * 0.045)
    cbar_height = max(0.10, ref_box.height * 0.30)
    pad_x = max(0.005, ref_box.width * 0.02)
    pad_y = max(0.005, ref_box.height * 0.02)

    if corner == "upper_left":
        x0 = ref_box.x0 + pad_x
        y0 = ref_box.y1 - cbar_height - pad_y
    else:
        x0 = ref_box.x1 - cbar_width - pad_x
        y0 = ref_box.y0 + pad_y

    x0 = float(np.clip(x0, 0.0, max(0.0, 1.0 - cbar_width)))
    y0 = float(np.clip(y0, 0.0, max(0.0, 1.0 - cbar_height)))
    cbar_ax.set_position([x0, y0, cbar_width, cbar_height])


def _save_matrix_plot(
    matrix: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    cmap: str,
    center: Optional[float],
    use_clustermap: bool,
    row_cluster: bool,
    col_cluster: bool,
    figsize: Tuple[float, float],
    percentile: float,
    dpi: int,
    row_colors: Optional[pd.Series] = None,
    col_colors: Optional[pd.Series] = None,
    fixed_limits: Optional[Tuple[float, float]] = None,
    cbar_corner: str = "lower_right",
) -> None:
    if matrix.empty:
        logging.warning("Skipping matrix plot '%s': empty matrix.", title)
        return

    if fixed_limits is not None:
        vmin, vmax = fixed_limits
        if (
            vmin is None
            or vmax is None
            or not np.isfinite(vmin)
            or not np.isfinite(vmax)
            or np.isclose(vmin, vmax)
        ):
            logging.warning(
                "Invalid fixed limits supplied for '%s'. Falling back to per-matrix limits.",
                title,
            )
            vmin, vmax = _compute_limits(matrix, percentile=percentile, center=center)
    else:
        vmin, vmax = _compute_limits(matrix, percentile=percentile, center=center)
    if vmin is None or vmax is None:
        logging.warning("Skipping matrix plot '%s': matrix has no finite values.", title)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if use_clustermap:
        try:
            clustermap_kws: Dict[str, Any] = {
                "data": matrix,
                "cmap": cmap,
                "vmin": vmin,
                "vmax": vmax,
                "figsize": figsize,
                "row_cluster": bool(row_cluster),
                "col_cluster": bool(col_cluster),
                "xticklabels": 1,
                "yticklabels": 1,
                "linewidths": 0.4,
                "linecolor": "black",
                "dendrogram_ratio": 0.05,
                "cbar_kws": {"fraction": 0.046, "pad": 0.04},
            }
            if center is not None and vmin < center < vmax:
                clustermap_kws["center"] = center
                clustermap_kws["norm"] = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
            if row_colors is not None:
                clustermap_kws["row_colors"] = row_colors.reindex(matrix.index, fill_value="lightgray")
            if col_colors is not None:
                clustermap_kws["col_colors"] = col_colors.reindex(matrix.columns, fill_value="lightgray")

            grid = sns.clustermap(**clustermap_kws)
            grid.fig.canvas.draw()
            _place_colorbar_in_corner(grid.cax, grid.ax_heatmap, corner=cbar_corner)
            #_force_show_all_tick_labels(grid.ax_heatmap, x_rotation=90.0, y_rotation=0.0)
            grid.fig.suptitle(title, y=1.02)
            grid.fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
            plt.close(grid.fig)
            return
        except Exception as exc:
            logging.warning(
                "Clustermap failed for '%s' (%s). Falling back to heatmap.",
                title,
                exc,
            )

    fig, ax = plt.subplots(figsize=figsize)
    heatmap_kws: Dict[str, Any] = {
        "data": matrix,
        "ax": ax,
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "xticklabels": 1,
        "yticklabels": 1,
        "linewidths": 0.4,
        "linecolor": "black",
        "cbar_kws": {"fraction": 0.046, "pad": 0.04},
    }
    if center is not None and vmin < center < vmax:
        heatmap_kws["center"] = center
        heatmap_kws["norm"] = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    heatmap_artist = sns.heatmap(**heatmap_kws)
    #_force_show_all_tick_labels(ax, x_rotation=90.0, y_rotation=0.0)
    ax.set_title(title)
    fig.tight_layout()
    fig.canvas.draw()
    if heatmap_artist.collections:
        cbar = heatmap_artist.collections[0].colorbar
        if cbar is not None:
            _place_colorbar_in_corner(cbar.ax, ax, corner=cbar_corner)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _save_pair_barplots(
    data: pd.DataFrame,
    *,
    pairs: Dict[str, List[str]],
    group_col: Optional[str],
    analysis: str,
    metric: str,
    color_map: Dict[str, str],
    out_dir: Path,
    raw_dir: Path,
    figsize: Tuple[float, float],
    dpi: int,
    extension: str,
    add_points: bool,
    value_label: str,
) -> None:
    if data.empty or not pairs:
        return

    for source, targets in pairs.items():
        for target in targets:
            subset = data[
                (data["source_population"].astype(str) == str(source))
                & (data["target_population"].astype(str) == str(target))
            ].copy()
            if subset.empty:
                continue

            pair_stub = f"{cleanstring(source)}_to_{cleanstring(target)}"
            raw_path = raw_dir / f"{analysis}_{metric}_{pair_stub}.csv"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            subset.to_csv(raw_path, index=False)

            fig, ax = plt.subplots(figsize=figsize)
            if group_col and group_col in subset.columns and subset[group_col].notna().any():
                x_col = group_col
                order = (
                    subset[x_col].cat.categories.tolist()
                    if pd.api.types.is_categorical_dtype(subset[x_col])
                    else sorted(subset[x_col].dropna().unique().tolist())
                )
                errorbar = "se" if subset[x_col].nunique(dropna=True) > 1 else None
                sns.barplot(
                    data=subset,
                    x=x_col,
                    y="value",
                    order=order,
                    errorbar=errorbar,
                    palette="tab10",
                    ax=ax,
                )
                if add_points:
                    sns.stripplot(
                        data=subset,
                        x=x_col,
                        y="value",
                        order=order,
                        color="black",
                        size=3.0,
                        alpha=0.6,
                        jitter=0.2,
                        ax=ax,
                    )
            else:
                subset["pair"] = f"{source} -> {target}"
                sns.barplot(
                    data=subset,
                    x="pair",
                    y="value",
                    color=color_map.get(str(target), "#4c72b0"),
                    errorbar="se" if len(subset) > 1 else None,
                    ax=ax,
                )
                if add_points and len(subset) > 1:
                    sns.stripplot(
                        data=subset,
                        x="pair",
                        y="value",
                        color="black",
                        size=3.0,
                        alpha=0.6,
                        jitter=0.15,
                        ax=ax,
                    )

            ax.set_title(f"{source} -> {target}")
            ax.set_xlabel("")
            ax.set_ylabel(value_label)
            ax.tick_params(axis="x", labelrotation=90 if group_col else 0)
            ax.grid(False)
            fig.tight_layout()

            plot_path = out_dir / f"{analysis}_{metric}_{pair_stub}{extension}"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=int(dpi), bbox_inches="tight")
            plt.close(fig)


def _flatten_squidpy_results(
    results: Dict[str, Dict[str, pd.DataFrame]], subregion_key: str
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for metric, region_dict in results.items():
        for region_name, matrix in region_dict.items():
            if matrix is None or matrix.empty:
                continue
            df = matrix.copy()
            df.index = df.index.astype(str)
            df.columns = [str(c) for c in df.columns]
            long = (
                df.rename_axis("source_population")
                .reset_index()
                .melt(
                    id_vars=["source_population"],
                    var_name="target_population",
                    value_name="value",
                )
            )
            long[subregion_key] = region_name
            long["metric"] = metric
            frames.append(long)
    if not frames:
        return pd.DataFrame(
            columns=[subregion_key, "source_population", "target_population", "metric", "value"]
        )
    return pd.concat(frames, ignore_index=True)


def _distance_matrix_to_long(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["roi", "source_population", "target_population", "metric", "value"])

    reset = df.reset_index()
    id_cols = [c for c in ["roi", "source_population"] if c in reset.columns]
    if len(id_cols) < 2:
        rename_map = {}
        cols = list(reset.columns)
        if cols:
            rename_map[cols[0]] = "roi"
        if len(cols) > 1:
            rename_map[cols[1]] = "source_population"
        reset = reset.rename(columns=rename_map)
        id_cols = ["roi", "source_population"]

    long = reset.melt(
        id_vars=id_cols,
        var_name="target_population",
        value_name="value",
    )
    long["metric"] = metric
    return long


def _load_saved_csv(
    path: Path,
    *,
    required_cols: Optional[Sequence[str]] = None,
) -> Optional[pd.DataFrame]:
    """Load a saved CSV if present and structurally usable."""
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logging.warning("Could not load saved results from %s: %s", path, exc)
        return None

    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logging.warning(
                "Saved results file %s is missing required columns %s. Ignoring saved file.",
                path,
                missing,
            )
            return None
    return df


def run_pairwise_spatial_analyses(
    *,
    general_config: GeneralConfig,
    pairwise_config: PairwiseSpatialConfig,
) -> Path:
    stage_name = "PairwiseSpatial"
    pairwise_config.population_obs = coalesce_config_text(
        pairwise_config.population_obs,
        general_config.population_obs_primary,
        default="population",
    )
    pairwise_config.groupby_obs = coalesce_config_text(
        pairwise_config.groupby_obs,
        general_config.groupby_obs,
    )
    pairwise_config.roi_obs = coalesce_config_text(
        pairwise_config.roi_obs,
        general_config.roi_obs,
        default="ROI",
    )
    pairwise_config.x_coord_obs = coalesce_config_text(
        pairwise_config.x_coord_obs,
        general_config.x_coord_obs,
        default="X_loc",
    )
    pairwise_config.y_coord_obs = coalesce_config_text(
        pairwise_config.y_coord_obs,
        general_config.y_coord_obs,
        default="Y_loc",
    )
    pairwise_config.master_index_obs = coalesce_config_text(
        pairwise_config.master_index_obs,
        general_config.master_index_obs,
        default="Master_Index",
    )
    if not pairwise_config.metadata_obs_columns:
        pairwise_config.metadata_obs_columns = (
            coalesce_config_list(general_config.metadata_obs, default=[]) or []
        )

    input_path = _resolve_input_adata_path(general_config, pairwise_config)
    adata, _, skip_stage, _ = load_pipeline_anndata(
        general_config=general_config,
        stage_name=stage_name,
        stage_config=pairwise_config,
        override_path=str(input_path),
    )
    if skip_stage:
        logging.info("Skipping PairwiseSpatial stage based on AnnData stage policy.")
        return Path(general_config.qc_folder) / pairwise_config.output_subdir
    if adata is None:
        raise FileNotFoundError(f"AnnData not found for PairwiseSpatial stage: {input_path}")
    logging.info("Loaded AnnData: %d cells x %d markers", adata.n_obs, adata.n_vars)

    logging.info(
        "Resolved Pairwise obs keys: population_obs='%s', groupby_obs='%s', roi_obs='%s', x_coord_obs='%s', "
        "y_coord_obs='%s', master_index_obs='%s'.",
        pairwise_config.population_obs,
        pairwise_config.groupby_obs,
        pairwise_config.roi_obs,
        pairwise_config.x_coord_obs,
        pairwise_config.y_coord_obs,
        pairwise_config.master_index_obs,
    )

    if pairwise_config.population_obs not in adata.obs.columns:
        raise KeyError(
            f"Configured population_obs '{pairwise_config.population_obs}' not found in AnnData.obs."
        )
    adata.obs[pairwise_config.population_obs] = adata.obs[pairwise_config.population_obs].astype("category")

    metadata_cols = _resolve_metadata_columns(adata, pairwise_config)
    source_population_obs = pairwise_config.source_population_obs or pairwise_config.population_obs

    output_root = Path(general_config.qc_folder) / pairwise_config.output_subdir
    raw_dir = output_root / "raw_data"
    matrix_dir = output_root / "plots" / "pairwise_matrices"
    pair_bar_dir = output_root / "plots" / "selected_pairs"
    metadata_dir = output_root / "metadata"
    for folder in [raw_dir, matrix_dir, pair_bar_dir, metadata_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    extension = _ensure_extension(pairwise_config.figure_extension)
    matrix_figsize = _figsize(pairwise_config.heatmap_figsize, fallback=(8.0, 6.0))
    bar_figsize = _figsize(pairwise_config.barplot_figsize, fallback=(3.0, 3.0))
    reload_saved_results = bool(pairwise_config.reload_saved_results)
    cbar_corner = _normalise_cbar_corner(pairwise_config.pairwise_matrices_cbar_corner)
    analysis_sources: Dict[str, str] = {"squidpy": "skipped", "distance": "skipped", "pcf": "skipped"}
    logging.info("Pairwise reload_saved_results=%s", reload_saved_results)
    logging.info("Pairwise matrix colorbar corner=%s", cbar_corner)

    ordered_pops = _population_order(adata, pairwise_config.population_obs)
    color_map = _population_color_map(adata, pairwise_config.population_obs, ordered_pops)
    pair_map = _normalise_population_pairs(
        pairwise_config.population_pairs,
        pairwise_config.population_obs,
        ordered_pops,
    )

    obs_snapshot = adata.obs.copy()
    obs_snapshot.to_csv(metadata_dir / "anndata_obs_snapshot.csv.gz", index=True)
    logging.info("Saved AnnData.obs snapshot to %s", metadata_dir / "anndata_obs_snapshot.csv.gz")

    roi_metadata = _build_key_metadata_table(adata, pairwise_config.roi_obs, metadata_cols)
    if not roi_metadata.empty:
        roi_metadata.to_csv(metadata_dir / "roi_metadata.csv")

    if pairwise_config.run_squidpy_interactions:
        subregion_obs = pairwise_config.squidpy_subregion_obs or pairwise_config.roi_obs
        squidpy_raw_dir = raw_dir / "squidpy_interactions"
        squidpy_raw_dir.mkdir(parents=True, exist_ok=True)
        squidpy_long_path = squidpy_raw_dir / "squidpy_interactions_long.csv"
        squidpy_long = pd.DataFrame()
        loaded_squidpy = False

        if reload_saved_results:
            loaded = _load_saved_csv(
                squidpy_long_path,
                required_cols=["source_population", "target_population", "metric", "value"],
            )
            if loaded is not None:
                squidpy_long = loaded
                loaded_squidpy = True
                analysis_sources["squidpy"] = "loaded"
                logging.info("Reloaded saved Squidpy results from %s", squidpy_long_path)

        if not loaded_squidpy:
            logging.info(
                "Running Squidpy interactions (population_obs=%s, subregion=%s).",
                pairwise_config.population_obs,
                subregion_obs,
            )
            squidpy_results = sbt_spatial.squidpy_subregion_interactions(
                adata=adata,
                population_obs=pairwise_config.population_obs,
                subregion=subregion_obs,
                subregion_suffix=pairwise_config.squidpy_subregion_suffix,
                radius=(int(pairwise_config.squidpy_radius_min_um), int(pairwise_config.squidpy_radius_max_um)),
                n_permutations=int(pairwise_config.squidpy_n_permutations),
            )

            for metric, region_dict in squidpy_results.items():
                metric_dir = squidpy_raw_dir / metric
                metric_dir.mkdir(parents=True, exist_ok=True)
                for region_name, matrix in region_dict.items():
                    matrix.to_csv(metric_dir / f"{cleanstring(region_name)}.csv")

            squidpy_long = _flatten_squidpy_results(squidpy_results, subregion_obs)
            subregion_metadata = _build_key_metadata_table(adata, subregion_obs, metadata_cols)
            if not subregion_metadata.empty and not squidpy_long.empty:
                squidpy_long = squidpy_long.merge(
                    subregion_metadata.reset_index(),
                    on=subregion_obs,
                    how="left",
                )
            squidpy_long.to_csv(squidpy_long_path, index=False)
            analysis_sources["squidpy"] = "computed"

        if pairwise_config.make_matrix_plots and not squidpy_long.empty:
            row_colors = _label_colors(ordered_pops, color_map)
            col_colors = _label_colors(ordered_pops, color_map)

            for metric in sorted(squidpy_long["metric"].dropna().unique().tolist()):
                metric_df = squidpy_long[squidpy_long["metric"] == metric].copy()
                matrix = metric_df.pivot_table(
                    index="source_population",
                    columns="target_population",
                    values="value",
                    aggfunc="mean",
                )
                matrix = _ordered_matrix(matrix, ordered_pops)
                matrix.to_csv(raw_dir / "squidpy_interactions" / f"matrix_mean_{metric}.csv")

                if metric == "count":
                    cmap = pairwise_config.heatmap_cmap_counts
                    center = None
                else:
                    cmap = pairwise_config.heatmap_cmap_interactions
                    center = 0.0

                shared_limits = _shared_limits_from_all(
                    matrix,
                    enabled=bool(pairwise_config.pairwise_matrices_share_vmax_vmin),
                    percentile=pairwise_config.heatmap_percentile,
                    center=center,
                    analysis="squidpy",
                    metric=metric,
                )
                _save_matrix_plot(
                    matrix,
                    out_path=matrix_dir / f"squidpy_{metric}_all{extension}",
                    title=f"Squidpy {metric} (all)",
                    cmap=cmap,
                    center=center,
                    use_clustermap=pairwise_config.heatmap_use_clustermap,
                    row_cluster=pairwise_config.heatmap_row_cluster,
                    col_cluster=pairwise_config.heatmap_col_cluster,
                    figsize=matrix_figsize,
                    percentile=pairwise_config.heatmap_percentile,
                    dpi=pairwise_config.figure_dpi,
                    row_colors=row_colors,
                    col_colors=col_colors,
                    fixed_limits=shared_limits,
                    cbar_corner=cbar_corner,
                )

                if (
                    pairwise_config.groupby_obs
                    and pairwise_config.groupby_obs in metric_df.columns
                ):
                    for group_name, group_df in metric_df.groupby(pairwise_config.groupby_obs, observed=True):
                        group_matrix = group_df.pivot_table(
                            index="source_population",
                            columns="target_population",
                            values="value",
                            aggfunc="mean",
                        )
                        group_matrix = _ordered_matrix(group_matrix, ordered_pops)
                        _save_matrix_plot(
                            group_matrix,
                            out_path=matrix_dir
                            / f"squidpy_{metric}_{cleanstring(group_name)}{extension}",
                            title=f"Squidpy {metric} ({group_name})",
                            cmap=cmap,
                            center=center,
                            use_clustermap=pairwise_config.heatmap_use_clustermap,
                            row_cluster=pairwise_config.heatmap_row_cluster,
                            col_cluster=pairwise_config.heatmap_col_cluster,
                            figsize=matrix_figsize,
                            percentile=pairwise_config.heatmap_percentile,
                            dpi=pairwise_config.figure_dpi,
                            row_colors=row_colors,
                            col_colors=col_colors,
                            fixed_limits=shared_limits,
                            cbar_corner=cbar_corner,
                        )

                if pairwise_config.make_pair_barplots and pair_map:
                    group_col = (
                        pairwise_config.groupby_obs
                        if pairwise_config.groupby_obs in metric_df.columns
                        else None
                    )
                    _save_pair_barplots(
                        metric_df,
                        pairs=pair_map,
                        group_col=group_col,
                        analysis="squidpy",
                        metric=metric,
                        color_map=color_map,
                        out_dir=pair_bar_dir,
                        raw_dir=raw_dir / "squidpy_interactions" / "selected_pairs",
                        figsize=bar_figsize,
                        dpi=pairwise_config.figure_dpi,
                        extension=extension,
                        add_points=pairwise_config.barplot_add_points,
                        value_label=f"Squidpy {metric}",
                    )

    if pairwise_config.run_distance_bootstrap:
        distance_raw_dir = raw_dir / "distance_bootstrap"
        distance_raw_dir.mkdir(parents=True, exist_ok=True)
        distance_long_path = distance_raw_dir / "distance_long.csv"
        distance_long = pd.DataFrame()
        loaded_distance = False

        if reload_saved_results:
            loaded = _load_saved_csv(
                distance_long_path,
                required_cols=["source_population", "target_population", "metric", "value"],
            )
            if loaded is not None:
                distance_long = loaded
                loaded_distance = True
                analysis_sources["distance"] = "loaded"
                logging.info("Reloaded saved distance-bootstrap results from %s", distance_long_path)

        if not loaded_distance:
            if source_population_obs not in adata.obs.columns:
                raise KeyError(
                    f"Configured source_population_obs '{source_population_obs}' not found in AnnData.obs."
                )
            logging.info(
                "Running distance bootstrap (population_obs=%s, source_population_obs=%s).",
                pairwise_config.population_obs,
                source_population_obs,
            )
            observed_all, bootmean_all, delta_all, zscore_all = (
                sbt_distance.bootstrap_nearest_population_distances_all_rois(
                    adata,
                    roi_col=pairwise_config.roi_obs,
                    master_index_col=pairwise_config.master_index_obs,
                    cell_type_col=pairwise_config.population_obs,
                    x_col=pairwise_config.x_coord_obs,
                    y_col=pairwise_config.y_coord_obs,
                    populations=pairwise_config.distance_populations,
                    roi_ids=pairwise_config.distance_roi_ids,
                    n_bootstraps=int(pairwise_config.distance_n_bootstraps),
                    n_jobs=int(pairwise_config.distance_n_jobs),
                    source_population_col=source_population_obs,
                    ddof=int(pairwise_config.distance_ddof),
                )
            )

            metric_wide = {
                "observed": observed_all,
                "bootmean": bootmean_all,
                "delta": delta_all,
                "zscore": zscore_all,
            }
            for metric, df in metric_wide.items():
                df.to_csv(distance_raw_dir / f"distance_{metric}.csv")

            long_frames: List[pd.DataFrame] = []
            for metric, df in metric_wide.items():
                long_frames.append(_distance_matrix_to_long(df, metric=metric))
            distance_long = pd.concat(long_frames, ignore_index=True)
            if not roi_metadata.empty and not distance_long.empty:
                merge_roi_meta = roi_metadata.reset_index().rename(columns={pairwise_config.roi_obs: "roi"})
                distance_long = distance_long.merge(merge_roi_meta, on="roi", how="left")
            distance_long.to_csv(distance_long_path, index=False)
            analysis_sources["distance"] = "computed"

        if pairwise_config.make_matrix_plots and not distance_long.empty:
            row_colors = _label_colors(ordered_pops, color_map)
            col_colors = _label_colors(ordered_pops, color_map)

            for metric in sorted(distance_long["metric"].dropna().unique().tolist()):
                metric_df = distance_long[distance_long["metric"] == metric].copy()
                matrix = metric_df.pivot_table(
                    index="source_population",
                    columns="target_population",
                    values="value",
                    aggfunc="mean",
                )
                matrix = _ordered_matrix(matrix, ordered_pops)
                matrix.to_csv(distance_raw_dir / f"matrix_mean_{metric}.csv")

                center = 0.0 if metric in {"delta", "zscore"} else None
                base_distance_cmap = (
                    pairwise_config.heatmap_cmap_counts
                    if metric in {"observed", "bootmean"}
                    else pairwise_config.heatmap_cmap_distance
                )
                distance_cmap = _reversed_cmap_name(base_distance_cmap)
                shared_limits = _shared_limits_from_all(
                    matrix,
                    enabled=bool(pairwise_config.pairwise_matrices_share_vmax_vmin),
                    percentile=pairwise_config.heatmap_percentile,
                    center=center,
                    analysis="distance",
                    metric=metric,
                )
                _save_matrix_plot(
                    matrix,
                    out_path=matrix_dir / f"distance_{metric}_all{extension}",
                    title=f"Distance {metric} (all)",
                    cmap=distance_cmap,
                    center=center,
                    use_clustermap=pairwise_config.heatmap_use_clustermap,
                    row_cluster=pairwise_config.heatmap_row_cluster,
                    col_cluster=pairwise_config.heatmap_col_cluster,
                    figsize=matrix_figsize,
                    percentile=pairwise_config.heatmap_percentile,
                    dpi=pairwise_config.figure_dpi,
                    row_colors=row_colors,
                    col_colors=col_colors,
                    fixed_limits=shared_limits,
                    cbar_corner=cbar_corner,
                )

                if (
                    pairwise_config.groupby_obs
                    and pairwise_config.groupby_obs in metric_df.columns
                ):
                    for group_name, group_df in metric_df.groupby(pairwise_config.groupby_obs, observed=True):
                        group_matrix = group_df.pivot_table(
                            index="source_population",
                            columns="target_population",
                            values="value",
                            aggfunc="mean",
                        )
                        group_matrix = _ordered_matrix(group_matrix, ordered_pops)
                        _save_matrix_plot(
                            group_matrix,
                            out_path=matrix_dir
                            / f"distance_{metric}_{cleanstring(group_name)}{extension}",
                            title=f"Distance {metric} ({group_name})",
                            cmap=distance_cmap,
                            center=center,
                            use_clustermap=pairwise_config.heatmap_use_clustermap,
                            row_cluster=pairwise_config.heatmap_row_cluster,
                            col_cluster=pairwise_config.heatmap_col_cluster,
                            figsize=matrix_figsize,
                            percentile=pairwise_config.heatmap_percentile,
                            dpi=pairwise_config.figure_dpi,
                            row_colors=row_colors,
                            col_colors=col_colors,
                            fixed_limits=shared_limits,
                            cbar_corner=cbar_corner,
                        )

                if pairwise_config.make_pair_barplots and pair_map:
                    group_col = (
                        pairwise_config.groupby_obs
                        if pairwise_config.groupby_obs in metric_df.columns
                        else None
                    )
                    _save_pair_barplots(
                        metric_df,
                        pairs=pair_map,
                        group_col=group_col,
                        analysis="distance",
                        metric=metric,
                        color_map=color_map,
                        out_dir=pair_bar_dir,
                        raw_dir=distance_raw_dir / "selected_pairs",
                        figsize=bar_figsize,
                        dpi=pairwise_config.figure_dpi,
                        extension=extension,
                        add_points=pairwise_config.barplot_add_points,
                        value_label=f"Distance {metric}",
                    )

    if pairwise_config.run_pcf:
        pcf_raw_dir = raw_dir / "pcf"
        pcf_spoox_dir = pcf_raw_dir / "spoox_out"
        pcf_summary_dir = pcf_raw_dir / "summary"
        pcf_stats_path = pcf_raw_dir / "pcf_stats.txt"
        pcf_conditions_path = pcf_raw_dir / "pcf_conditions.json"
        pcf_summary_path = pcf_raw_dir / "pcf_summary.csv"
        pcf_long_path = pcf_raw_dir / "pcf_long.csv"
        for folder in [pcf_raw_dir, pcf_spoox_dir, pcf_summary_dir]:
            folder.mkdir(parents=True, exist_ok=True)

        pcf_summary = None
        pcf_long = None
        if reload_saved_results:
            loaded_summary = _load_saved_csv(
                pcf_summary_path,
                required_cols=["cell_type_1", "cell_type_2"],
            )
            loaded_long = _load_saved_csv(
                pcf_long_path,
                required_cols=["source_population", "target_population", "metric", "value"],
            )
            if loaded_summary is not None and loaded_long is not None:
                pcf_summary = loaded_summary
                pcf_long = loaded_long
                analysis_sources["pcf"] = "loaded"
                logging.info(
                    "Reloaded saved PCF results from %s and %s",
                    pcf_summary_path,
                    pcf_long_path,
                )
            elif loaded_summary is not None or loaded_long is not None:
                logging.warning(
                    "Found partial saved PCF outputs (summary or long only). Recomputing PCF analysis."
                )

        if pcf_summary is None or pcf_long is None:
            logging.info("Running PCF analysis at %.2f um.", pairwise_config.pcf_target_distance_um)
            pcf_summary = sbt_pcf.run_paircorrelation_at_distance(
                adata=adata,
                population_obs=pairwise_config.population_obs,
                groupby=pairwise_config.groupby_obs,
                target_distance=float(pairwise_config.pcf_target_distance_um),
                spoox_output_dir=pcf_spoox_dir,
                spoox_output_summary_dir=pcf_summary_dir,
                stats_file=pcf_stats_path,
                conditions_file=pcf_conditions_path,
                index_obs=pairwise_config.master_index_obs,
                roi_obs=pairwise_config.roi_obs,
                xloc_obs=pairwise_config.x_coord_obs,
                yloc_obs=pairwise_config.y_coord_obs,
                cluster_column=pairwise_config.pcf_cluster_column,
                samples=pairwise_config.pcf_samples,
                max_radius=float(pairwise_config.pcf_max_radius_um),
                radius_step=float(pairwise_config.pcf_radius_step_um),
                num_bootstrap=int(pairwise_config.pcf_num_bootstrap),
            )
            pcf_summary.to_csv(pcf_summary_path, index=False)

            if "condition" not in pcf_summary.columns:
                pcf_summary["condition"] = "All"

            pcf_long = pcf_summary.melt(
                id_vars=["condition", "cell_type_1", "cell_type_2"],
                value_vars=[c for c in ["g_mean", "g_min", "g_max"] if c in pcf_summary.columns],
                var_name="metric",
                value_name="value",
            ).rename(
                columns={
                    "cell_type_1": "source_population",
                    "cell_type_2": "target_population",
                }
            )
            pcf_long.to_csv(pcf_long_path, index=False)

            pcf_cell_input_cols = [
                c
                for c in [
                    pairwise_config.master_index_obs,
                    pairwise_config.roi_obs,
                    pairwise_config.population_obs,
                    pairwise_config.groupby_obs,
                    *metadata_cols,
                ]
                if c and c in adata.obs.columns
            ]
            if pcf_cell_input_cols:
                adata.obs[pcf_cell_input_cols].to_csv(pcf_raw_dir / "pcf_input_cell_metadata.csv")
            analysis_sources["pcf"] = "computed"

        if "condition" not in pcf_summary.columns:
            pcf_summary["condition"] = "All"
        if "condition" not in pcf_long.columns:
            pcf_long["condition"] = "All"

        if pairwise_config.make_matrix_plots and not pcf_summary.empty:
            if "g_mean" not in pcf_summary.columns:
                logging.warning(
                    "PCF summary is missing 'g_mean'; skipping PCF matrix plots."
                )
            else:
                pcf_all_matrix = pcf_summary.pivot_table(
                    index="cell_type_1",
                    columns="cell_type_2",
                    values="g_mean",
                    aggfunc="mean",
                )
                pcf_all_matrix = _ordered_matrix(pcf_all_matrix, ordered_pops)
                shared_pcf_limits = _shared_limits_from_all(
                    pcf_all_matrix,
                    enabled=bool(pairwise_config.pairwise_matrices_share_vmax_vmin),
                    percentile=pairwise_config.heatmap_percentile,
                    center=1.0,
                    analysis="pcf",
                    metric="g_mean",
                )
                _save_matrix_plot(
                    pcf_all_matrix,
                    out_path=matrix_dir / f"pcf_g_mean_all_conditions{extension}",
                    title="PCF g_mean (all conditions combined)",
                    cmap=pairwise_config.heatmap_cmap_pcf,
                    center=1.0,
                    use_clustermap=pairwise_config.heatmap_use_clustermap,
                    row_cluster=pairwise_config.heatmap_row_cluster,
                    col_cluster=pairwise_config.heatmap_col_cluster,
                    figsize=matrix_figsize,
                    percentile=pairwise_config.heatmap_percentile,
                    dpi=pairwise_config.figure_dpi,
                    row_colors=_label_colors(ordered_pops, color_map),
                    col_colors=_label_colors(ordered_pops, color_map),
                    fixed_limits=shared_pcf_limits,
                    cbar_corner=cbar_corner,
                )
                for condition_name, cond_df in pcf_summary.groupby("condition", observed=True):
                    cond_df = cond_df.copy()
                    matrix = cond_df.pivot_table(
                        index="cell_type_1",
                        columns="cell_type_2",
                        values="g_mean",
                        aggfunc="mean",
                    )
                    matrix = _ordered_matrix(matrix, ordered_pops)
                    labels = sorted(
                        pd.unique(
                            np.concatenate(
                                [
                                    cond_df["cell_type_1"].astype(str).to_numpy(),
                                    cond_df["cell_type_2"].astype(str).to_numpy(),
                                ]
                            )
                        ).tolist()
                    )
                    color_series = _label_colors(labels, color_map)
                    condition_stub = cleanstring(condition_name)
                    out_path = matrix_dir / f"pcf_g_mean_{condition_stub}{extension}"

                    try:
                        grid = sbt_pcf.plot_paircorrelation_clustermap(
                            summary=cond_df,
                            condition=condition_name,
                            percentile=pairwise_config.heatmap_percentile,
                            vmin=shared_pcf_limits[0] if shared_pcf_limits is not None else None,
                            vmax=shared_pcf_limits[1] if shared_pcf_limits is not None else None,
                            cmap=pairwise_config.heatmap_cmap_pcf,
                            cluster=bool(
                                pairwise_config.heatmap_row_cluster
                                and pairwise_config.heatmap_col_cluster
                            ),
                            figsize=matrix_figsize,
                            row_colors=color_series,
                            col_colors=color_series,
                        )
                        grid.fig.canvas.draw()
                        _place_colorbar_in_corner(grid.cax, grid.ax_heatmap, corner=cbar_corner)
                        #_force_show_all_tick_labels(grid.ax_heatmap, x_rotation=90.0, y_rotation=0.0)
                        grid.fig.savefig(out_path, dpi=int(pairwise_config.figure_dpi), bbox_inches="tight")
                        plt.close(grid.fig)
                    except Exception as exc:
                        logging.warning(
                            "PCF clustermap failed for condition '%s' (%s). Using heatmap fallback.",
                            condition_name,
                            exc,
                        )
                        _save_matrix_plot(
                            matrix,
                            out_path=out_path,
                            title=f"PCF g_mean ({condition_name})",
                            cmap=pairwise_config.heatmap_cmap_pcf,
                            center=1.0,
                            use_clustermap=pairwise_config.heatmap_use_clustermap,
                            row_cluster=pairwise_config.heatmap_row_cluster,
                            col_cluster=pairwise_config.heatmap_col_cluster,
                            figsize=matrix_figsize,
                            percentile=pairwise_config.heatmap_percentile,
                            dpi=pairwise_config.figure_dpi,
                            row_colors=_label_colors(ordered_pops, color_map),
                            col_colors=_label_colors(ordered_pops, color_map),
                            fixed_limits=shared_pcf_limits,
                            cbar_corner=cbar_corner,
                        )

        if pairwise_config.make_pair_barplots and pair_map and not pcf_long.empty:
            for metric in sorted(pcf_long["metric"].dropna().unique().tolist()):
                metric_df = pcf_long[pcf_long["metric"] == metric].copy()
                _save_pair_barplots(
                    metric_df,
                    pairs=pair_map,
                    group_col="condition",
                    analysis="pcf",
                    metric=metric,
                    color_map=color_map,
                    out_dir=pair_bar_dir,
                    raw_dir=pcf_raw_dir / "selected_pairs",
                    figsize=bar_figsize,
                    dpi=pairwise_config.figure_dpi,
                    extension=extension,
                    add_points=pairwise_config.barplot_add_points,
                    value_label=f"PCF {metric}",
                )

    run_metadata = {
        "input_adata_path": str(input_path),
        "output_root": str(output_root),
        "population_obs": pairwise_config.population_obs,
        "groupby_obs": pairwise_config.groupby_obs,
        "roi_obs": pairwise_config.roi_obs,
        "source_population_obs": source_population_obs,
        "ran_squidpy": bool(pairwise_config.run_squidpy_interactions),
        "ran_distance": bool(pairwise_config.run_distance_bootstrap),
        "ran_pcf": bool(pairwise_config.run_pcf),
        "reload_saved_results": reload_saved_results,
        "pairwise_matrices_cbar_corner": cbar_corner,
        "pairwise_matrices_share_vmax_vmin": bool(pairwise_config.pairwise_matrices_share_vmax_vmin),
        "analysis_sources": analysis_sources,
        "population_pairs": pair_map,
    }
    metadata_path = output_root / "pairwise_spatial_run_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    logging.info("Saved run metadata to: %s", metadata_path)
    save_pipeline_anndata(
        adata=adata,
        general_config=general_config,
        stage_name=stage_name,
        stage_config=pairwise_config,
        override_path=str(input_path),
        extra_details={
            "output_root": str(output_root),
            "run_metadata_path": str(metadata_path),
        },
    )
    logging.info("Pairwise spatial outputs saved to: %s", output_root)
    return output_root


if __name__ == "__main__":
    pipeline_stage = "PairwiseSpatial"
    config = process_config_with_overrides()
    setup_logging(config.get("logging", {}), pipeline_stage)

    general_config = GeneralConfig(
        **filter_config_for_dataclass(config.get("general", {}), GeneralConfig)
    )
    pairwise_config = PairwiseSpatialConfig(
        **filter_config_for_dataclass(config.get("pairwise_spatial", {}), PairwiseSpatialConfig)
    )

    run_pairwise_spatial_analyses(
        general_config=general_config,
        pairwise_config=pairwise_config,
    )
