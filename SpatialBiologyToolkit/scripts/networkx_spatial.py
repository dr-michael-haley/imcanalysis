"""
NetworkX spatial graph metrics stage.

This stage:
1. Loads an AnnData object from the configured pipeline output.
2. Builds a Squidpy spatial-neighbor graph independently for each ROI.
3. Computes NetworkX metrics per ROI:
   - attribute assortativity coefficient across the full ROI graph
   - average clustering on the induced subgraph for each population
4. Optionally bootstraps by shuffling population labels within each ROI,
   while allowing selected populations to remain fixed.
5. Aggregates ROI-level summaries to case-level means using the ROI bootstrap
   distributions so case z-scores reflect ROI averaging.
6. Saves tidy ROI/case summary tables, optional raw bootstrap tables, metadata
   snapshots, and run metadata.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "networkx is required for this script. Install it with 'pip install networkx'."
    ) from exc

try:
    import squidpy as sq
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "Squidpy is required for this script. Install it with 'pip install squidpy'."
    ) from exc

from .config_and_utils import (
    GeneralConfig,
    NetworkxSpatialConfig,
    cleanstring,
    coalesce_config_list,
    coalesce_config_text,
    filter_config_for_dataclass,
    load_pipeline_anndata,
    process_config_with_overrides,
    save_pipeline_anndata,
    setup_logging,
)


MetricKey = Tuple[str, str]


def _resolve_input_adata_path(
    general_config: GeneralConfig,
    networkx_config: NetworkxSpatialConfig,
) -> Path:
    if networkx_config.input_adata_path:
        return Path(networkx_config.input_adata_path)
    return Path(general_config.anndata_path)


def _resolve_metadata_columns(
    adata: ad.AnnData,
    networkx_config: NetworkxSpatialConfig,
) -> List[str]:
    excluded_default = {
        str(c)
        for c in [
            networkx_config.population_obs,
            networkx_config.x_coord_obs,
            networkx_config.y_coord_obs,
            networkx_config.master_index_obs,
        ]
        if c
    }
    if networkx_config.include_all_obs_metadata:
        columns = [c for c in adata.obs.columns if c not in excluded_default]
    else:
        configured = networkx_config.metadata_obs_columns or []
        columns = [c for c in configured if c in adata.obs.columns]

    required = [
        networkx_config.roi_obs,
        networkx_config.case_obs,
        networkx_config.groupby_obs,
    ]
    required = [c for c in required if c and c in adata.obs.columns]

    seen: set[str] = set()
    ordered: List[str] = []
    for col in [*columns, *required]:
        if col in seen:
            continue
        seen.add(col)
        ordered.append(col)
    return ordered


def _build_key_metadata_table(
    adata: ad.AnnData,
    key_col: Optional[str],
    metadata_cols: Sequence[str],
) -> pd.DataFrame:
    if not key_col or key_col not in adata.obs.columns:
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


def _ordered_unique(series: pd.Series) -> List[str]:
    if pd.api.types.is_categorical_dtype(series):
        return [str(x) for x in series.cat.categories if pd.notna(x)]
    return [str(x) for x in pd.unique(series.dropna())]


def _missing_label_mask(series: pd.Series) -> pd.Series:
    mask = series.isna()
    try:
        text = series.astype("string").str.strip()
        mask = mask | text.eq("").fillna(False)
    except Exception:
        pass
    return mask


def _parse_graph_radius(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "none", "null"}:
            return None
        text = text.strip("[]()")
        if "," in text:
            vals = [float(v.strip()) for v in text.split(",") if v.strip()]
        else:
            vals = [float(text)]
    elif isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        vals = [float(v) for v in value if str(v).strip().lower() not in {"", "none", "null"}]
    else:
        vals = [float(value)]

    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    if len(vals) > 2:
        logging.warning(
            "networkx_spatial.graph_radius expects 1 or 2 values; received %d. Using the first two.",
            len(vals),
        )
    return float(vals[0]), float(vals[1])


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "none", "null"}:
            return None
        return int(text)
    return int(value)


def _ensure_extension(ext: str) -> str:
    if not ext:
        return ".png"
    return ext if ext.startswith(".") else f".{ext}"


def _figsize(values: Sequence[float], fallback: Tuple[float, float]) -> Tuple[float, float]:
    if not values or len(values) < 2:
        return fallback
    return float(values[0]), float(values[1])


def _optional_figsize(values: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(values, (list, tuple, np.ndarray, pd.Index)) or len(values) < 2:
        return None
    return float(values[0]), float(values[1])


def _load_saved_csv(
    path: Path,
    *,
    required_cols: Optional[Sequence[str]] = None,
) -> Optional[pd.DataFrame]:
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


def _normalise_plot_summary_level(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"case_if_available", "case", "roi"}:
        return text
    logging.warning(
        "Invalid networkx_spatial.plot_summary_level='%s'. Using 'case_if_available'.",
        value,
    )
    return "case_if_available"


def _parse_plot_value_columns(value: Any) -> List[str]:
    allowed = {"observed", "bootstrap_mean", "delta", "zscore"}
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",") if v.strip()]
    elif isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        items = [str(v).strip() for v in value if str(v).strip()]
    else:
        items = []

    resolved = [item for item in items if item in allowed]
    if not resolved:
        return ["observed", "zscore"]
    return resolved


def _normalise_plot_kind(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"bar", "barplot"}:
        return "barplot"
    if text in {"box", "boxplot"}:
        return "boxplot"
    logging.warning(
        "Invalid networkx_spatial.plot_kind='%s'. Using 'barplot'.",
        value,
    )
    return "barplot"


def _obs_color_map(
    adata: ad.AnnData,
    obs_key: str,
    ordered_labels: Sequence[str],
) -> Dict[str, str]:
    color_key = f"{obs_key}_colors"
    if color_key in adata.uns:
        colors = list(adata.uns[color_key])
        if len(colors) >= len(ordered_labels):
            return {str(label): str(colors[i]) for i, label in enumerate(ordered_labels)}

    palette = sns.color_palette("tab10", n_colors=max(1, len(ordered_labels))).as_hex()
    return {str(label): str(palette[i]) for i, label in enumerate(ordered_labels)}


def _dedupe_legend(ax: Any, *, title: str) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    seen = set()
    unique_h = []
    unique_l = []
    for handle, label in zip(handles, labels):
        name = str(label)
        if not name or name.startswith("_") or name in seen:
            continue
        seen.add(name)
        unique_h.append(handle)
        unique_l.append(name)
    if unique_h:
        ax.legend(
            unique_h,
            unique_l,
            title=title,
            frameon=True,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()


def _ordered_plot_levels(series: pd.Series, configured: Optional[Sequence[str]] = None) -> List[str]:
    observed = series.dropna().astype(str).unique().tolist()
    if configured:
        configured_order = [str(x) for x in configured]
        ordered = [x for x in configured_order if x in observed]
        extras = [x for x in observed if x not in ordered]
        return [*ordered, *extras]
    if pd.api.types.is_categorical_dtype(series):
        return [str(x) for x in series.cat.categories if str(x) in observed]
    return sorted(observed)


def _resolve_population_subset(
    configured: Optional[Sequence[str]],
    available: Sequence[str],
    *,
    context: str,
) -> List[str]:
    available_order = [str(x) for x in available]
    if not configured:
        return available_order

    available_set = set(available_order)
    ordered: List[str] = []
    missing: List[str] = []
    seen: set[str] = set()
    for item in configured:
        label = str(item).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        if label in available_set:
            ordered.append(label)
        else:
            missing.append(label)

    if missing:
        logging.warning("%s: ignoring populations not present in the summaries: %s", context, missing)
    if configured and not ordered:
        logging.warning("%s: no configured populations were found in the summaries.", context)
    return ordered


def _plot_all_populations_by_group(
    data: pd.DataFrame,
    *,
    pop_order: Sequence[str],
    group_col: str,
    group_order: Sequence[str],
    group_palette: Dict[str, str],
    value_col: str,
    ylabel: str,
    title: str,
    save_path: Path,
    fixed_figsize: Optional[Tuple[float, float]],
    base_figsize: Tuple[float, float],
    width_scale: float,
    dpi: int,
    add_points: bool,
    plot_kind: str,
) -> None:
    plot_data = data[["population", group_col, value_col]].dropna().copy()
    if plot_data.empty:
        return
    plot_data["population"] = plot_data["population"].astype(str)
    plot_data[group_col] = plot_data[group_col].astype(str)
    pop_order = [str(x) for x in pop_order if str(x) in plot_data["population"].unique().tolist()]
    group_order = [str(x) for x in group_order if str(x) in plot_data[group_col].unique().tolist()]
    if not pop_order or not group_order:
        return

    if fixed_figsize is None:
        plot_width = max(float(base_figsize[0]), max(0.05, float(width_scale)) * max(1, len(pop_order)))
        figsize = (plot_width, float(base_figsize[1]))
    else:
        figsize = fixed_figsize
    fig, ax = plt.subplots(figsize=figsize)
    if plot_kind == "boxplot":
        sns.boxplot(
            data=plot_data,
            x="population",
            y=value_col,
            hue=group_col,
            order=pop_order,
            hue_order=group_order,
            palette=group_palette,
            showfliers=False,
            linewidth=0.8,
            ax=ax,
        )
    else:
        sns.barplot(
            data=plot_data,
            x="population",
            y=value_col,
            hue=group_col,
            order=pop_order,
            hue_order=group_order,
            errorbar="se" if len(plot_data) > 1 else None,
            palette=group_palette,
            edgecolor="black",
            linewidth=0.6,
            capsize=0.2,
            ax=ax,
        )
    if add_points:
        sns.stripplot(
            data=plot_data,
            x="population",
            y=value_col,
            hue=group_col,
            order=pop_order,
            hue_order=group_order,
            dodge=True,
            palette=group_palette,
            size=2.5,
            alpha=0.6,
            jitter=0.15,
            ax=ax,
        )
    ax.set_xlabel("population")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(False)
    _dedupe_legend(ax, title=group_col)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _plot_all_populations_no_group(
    data: pd.DataFrame,
    *,
    pop_order: Sequence[str],
    pop_palette: Dict[str, str],
    value_col: str,
    ylabel: str,
    title: str,
    save_path: Path,
    fixed_figsize: Optional[Tuple[float, float]],
    base_figsize: Tuple[float, float],
    width_scale: float,
    dpi: int,
    add_points: bool,
    plot_kind: str,
) -> None:
    plot_data = data[["population", value_col]].dropna().copy()
    if plot_data.empty:
        return
    plot_data["population"] = plot_data["population"].astype(str)
    pop_order = [str(x) for x in pop_order if str(x) in plot_data["population"].unique().tolist()]
    if not pop_order:
        return

    if fixed_figsize is None:
        plot_width = max(float(base_figsize[0]), max(0.05, float(width_scale)) * max(1, len(pop_order)))
        figsize = (plot_width, float(base_figsize[1]))
    else:
        figsize = fixed_figsize
    palette = {str(pop): str(pop_palette.get(str(pop), "#4c72b0")) for pop in pop_order}
    fig, ax = plt.subplots(figsize=figsize)
    if plot_kind == "boxplot":
        sns.boxplot(
            data=plot_data,
            x="population",
            y=value_col,
            order=pop_order,
            hue="population",
            hue_order=pop_order,
            dodge=False,
            palette=palette,
            showfliers=False,
            linewidth=0.8,
            ax=ax,
        )
    else:
        sns.barplot(
            data=plot_data,
            x="population",
            y=value_col,
            order=pop_order,
            hue="population",
            hue_order=pop_order,
            dodge=False,
            errorbar="se" if len(plot_data) > 1 else None,
            palette=palette,
            edgecolor="black",
            linewidth=0.6,
            capsize=0.2,
            ax=ax,
        )
    if add_points:
        sns.stripplot(
            data=plot_data,
            x="population",
            y=value_col,
            order=pop_order,
            color="black",
            size=2.5,
            alpha=0.6,
            jitter=0.15,
            ax=ax,
        )
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_xlabel("population")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(False)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _plot_single_population_by_group(
    data: pd.DataFrame,
    *,
    group_col: str,
    group_order: Sequence[str],
    group_palette: Dict[str, str],
    value_col: str,
    ylabel: str,
    title: str,
    save_path: Path,
    figsize: Tuple[float, float],
    dpi: int,
    add_points: bool,
    plot_kind: str,
) -> None:
    plot_data = data[[group_col, value_col]].dropna().copy()
    if plot_data.empty:
        return
    plot_data[group_col] = plot_data[group_col].astype(str)
    group_order = [str(x) for x in group_order if str(x) in plot_data[group_col].unique().tolist()]
    if not group_order:
        return
    palette = {str(group): str(group_palette.get(str(group), "#4c72b0")) for group in group_order}

    fig, ax = plt.subplots(figsize=figsize)
    if plot_kind == "boxplot":
        sns.boxplot(
            data=plot_data,
            x=group_col,
            y=value_col,
            order=group_order,
            hue=group_col,
            hue_order=group_order,
            dodge=False,
            palette=palette,
            showfliers=False,
            linewidth=0.8,
            ax=ax,
        )
    else:
        sns.barplot(
            data=plot_data,
            x=group_col,
            y=value_col,
            order=group_order,
            hue=group_col,
            hue_order=group_order,
            dodge=False,
            errorbar="se" if len(plot_data) > 1 else None,
            palette=palette,
            edgecolor="black",
            linewidth=0.6,
            capsize=0.2,
            ax=ax,
        )
    if add_points:
        sns.stripplot(
            data=plot_data,
            x=group_col,
            y=value_col,
            order=group_order,
            color="black",
            size=2.8,
            alpha=0.6,
            jitter=0.15,
            ax=ax,
        )
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_xlabel(group_col)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(False)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _save_summary_plots(
    *,
    adata: ad.AnnData,
    roi_summary_df: pd.DataFrame,
    case_summary_df: pd.DataFrame,
    output_root: Path,
    networkx_config: NetworkxSpatialConfig,
    populations: Sequence[str],
    groupby_obs: Optional[str],
    configured_groups: Optional[Sequence[str]],
) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "plot_source_level": None,
        "plot_value_columns": [],
        "groupby_obs": groupby_obs,
        "plot_kind": None,
        "all_populations_order": [],
        "all_populations_figsize": None,
        "all_populations_figsize_mode": "auto",
    }
    if not bool(networkx_config.make_plots):
        details["status"] = "disabled"
        return details

    plot_level = _normalise_plot_summary_level(networkx_config.plot_summary_level)
    if plot_level == "case":
        plot_df = case_summary_df.copy()
        plot_source = "case"
    elif plot_level == "roi":
        plot_df = roi_summary_df.copy()
        plot_source = "roi"
    else:
        if not case_summary_df.empty:
            plot_df = case_summary_df.copy()
            plot_source = "case"
        else:
            plot_df = roi_summary_df.copy()
            plot_source = "roi"

    if plot_df.empty:
        details["status"] = "no_summary_data"
        return details

    extension = _ensure_extension(networkx_config.figure_extension)
    base_figsize = _figsize(networkx_config.barplot_figsize, fallback=(4.0, 3.0))
    width_scale = max(0.05, float(networkx_config.all_populations_width_scale))
    plot_kind = _normalise_plot_kind(networkx_config.plot_kind)
    value_columns = [col for col in _parse_plot_value_columns(networkx_config.plot_value_columns) if col in plot_df.columns]
    if not value_columns:
        details["status"] = "no_value_columns"
        return details
    details["plot_kind"] = plot_kind

    all_populations_order = _resolve_population_subset(
        networkx_config.all_populations_plot_populations,
        populations,
        context="networkx_spatial.all_populations_plot_populations",
    )
    all_populations_fixed_figsize = _optional_figsize(networkx_config.all_populations_figsize)
    details["all_populations_order"] = list(all_populations_order)
    details["all_populations_figsize"] = (
        [float(all_populations_fixed_figsize[0]), float(all_populations_fixed_figsize[1])]
        if all_populations_fixed_figsize is not None
        else None
    )
    details["all_populations_figsize_mode"] = "fixed" if all_populations_fixed_figsize is not None else "auto"

    group_col = None
    group_order: List[str] = []
    group_palette: Dict[str, str] = {}
    if groupby_obs and groupby_obs in plot_df.columns:
        group_col = str(groupby_obs)
        group_order = _ordered_plot_levels(plot_df[group_col], configured_groups)
        if group_order:
            group_palette = _obs_color_map(adata, group_col, group_order)

    pop_palette = _obs_color_map(adata, str(networkx_config.population_obs), populations)
    plots_root = output_root / "plots" / plot_source
    plots_root.mkdir(parents=True, exist_ok=True)

    clustering_df = plot_df[plot_df["metric"] == "average_clustering"].copy()
    assort_df = plot_df[plot_df["metric"] == "assortativity"].copy()

    for value_col in value_columns:
        pretty_label = value_col.replace("_", " ").title()

        if bool(networkx_config.make_all_populations_plots) and all_populations_order and not clustering_df.empty:
            if group_col and group_order:
                _plot_all_populations_by_group(
                    clustering_df,
                    pop_order=all_populations_order,
                    group_col=group_col,
                    group_order=group_order,
                    group_palette=group_palette,
                    value_col=value_col,
                    ylabel=f"Average clustering ({pretty_label})",
                    title=f"Average clustering by population ({pretty_label}, {plot_source})",
                    save_path=plots_root / "average_clustering_all_populations" / f"{value_col}{extension}",
                    fixed_figsize=all_populations_fixed_figsize,
                    base_figsize=base_figsize,
                    width_scale=width_scale,
                    dpi=int(networkx_config.figure_dpi),
                    add_points=bool(networkx_config.barplot_add_points),
                    plot_kind=plot_kind,
                )
            else:
                _plot_all_populations_no_group(
                    clustering_df,
                    pop_order=all_populations_order,
                    pop_palette=pop_palette,
                    value_col=value_col,
                    ylabel=f"Average clustering ({pretty_label})",
                    title=f"Average clustering by population ({pretty_label}, {plot_source})",
                    save_path=plots_root / "average_clustering_all_populations" / f"{value_col}{extension}",
                    fixed_figsize=all_populations_fixed_figsize,
                    base_figsize=base_figsize,
                    width_scale=width_scale,
                    dpi=int(networkx_config.figure_dpi),
                    add_points=bool(networkx_config.barplot_add_points),
                    plot_kind=plot_kind,
                )

        if bool(networkx_config.make_population_group_plots) and group_col and group_order and not clustering_df.empty:
            for population in populations:
                pop_subset = clustering_df[clustering_df["population"].astype(str) == str(population)].copy()
                if pop_subset.empty:
                    continue
                _plot_single_population_by_group(
                    pop_subset,
                    group_col=group_col,
                    group_order=group_order,
                    group_palette=group_palette,
                    value_col=value_col,
                    ylabel=f"{population} clustering ({pretty_label})",
                    title=f"{population} clustering by {group_col} ({pretty_label}, {plot_source})",
                    save_path=plots_root
                    / "average_clustering_by_population"
                    / f"{cleanstring(population)}_{value_col}{extension}",
                    figsize=base_figsize,
                    dpi=int(networkx_config.figure_dpi),
                    add_points=bool(networkx_config.barplot_add_points),
                    plot_kind=plot_kind,
                )

        if bool(networkx_config.make_assortativity_group_plots) and not assort_df.empty:
            assort_title = f"Assortativity by {group_col} ({pretty_label}, {plot_source})" if group_col else f"Assortativity ({pretty_label}, {plot_source})"
            if group_col and group_order:
                _plot_single_population_by_group(
                    assort_df,
                    group_col=group_col,
                    group_order=group_order,
                    group_palette=group_palette,
                    value_col=value_col,
                    ylabel=f"Assortativity ({pretty_label})",
                    title=assort_title,
                    save_path=plots_root / "assortativity" / f"{value_col}{extension}",
                    figsize=base_figsize,
                    dpi=int(networkx_config.figure_dpi),
                    add_points=bool(networkx_config.barplot_add_points),
                    plot_kind=plot_kind,
                )

    details["status"] = "created"
    details["plot_source_level"] = plot_source
    details["plot_value_columns"] = value_columns
    details["plots_root"] = str(plots_root)
    return details


def _resolve_num_workers(requested: int, n_rois: int) -> int:
    slurm_hint = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_NTASKS")
    try:
        available = int(slurm_hint) if slurm_hint else int(os.cpu_count() or 1)
    except Exception:
        available = int(os.cpu_count() or 1)
    available = max(1, available)

    if requested is None:
        workers = 1
    else:
        requested = int(requested)
        workers = available if requested < 0 else requested
    workers = max(1, workers)
    return min(workers, max(1, int(n_rois)))


def _safe_numeric(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_attribute_assortativity(graph: nx.Graph, attr: str) -> float:
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = nx.attribute_assortativity_coefficient(graph, attr)
    except Exception as exc:
        logging.debug("Assortativity failed: %s", exc)
        return float("nan")
    return _safe_numeric(value)


def _safe_average_clustering(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = nx.average_clustering(graph)
    except Exception as exc:
        logging.debug("Average clustering failed: %s", exc)
        return float("nan")
    return _safe_numeric(value)


def _apply_population_labels(
    graph: nx.Graph,
    node_ids: Sequence[int],
    labels: Sequence[str],
    attr_name: str,
) -> None:
    for node_id, label in zip(node_ids, labels):
        graph.nodes[int(node_id)][attr_name] = str(label)


def _compute_metric_vector(
    graph: nx.Graph,
    *,
    node_ids: Sequence[int],
    labels: np.ndarray,
    populations: Sequence[str],
    minimum_cells_per_population: int,
    attr_name: str = "population",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.full(1 + len(populations), np.nan, dtype=float)
    metric_n_cells = np.zeros(1 + len(populations), dtype=float)
    metric_n_edges = np.zeros(1 + len(populations), dtype=float)

    _apply_population_labels(graph, node_ids=node_ids, labels=labels, attr_name=attr_name)

    values[0] = _safe_attribute_assortativity(graph, attr_name)
    metric_n_cells[0] = float(graph.number_of_nodes())
    metric_n_edges[0] = float(graph.number_of_edges())

    for idx, pop in enumerate(populations, start=1):
        pop_nodes = np.flatnonzero(labels == str(pop)).tolist()
        pop_subgraph = graph.subgraph(pop_nodes)
        metric_n_cells[idx] = float(len(pop_nodes))
        metric_n_edges[idx] = float(pop_subgraph.number_of_edges())
        if len(pop_nodes) >= int(minimum_cells_per_population):
            values[idx] = _safe_average_clustering(pop_subgraph)

    return values, metric_n_cells, metric_n_edges


def _shuffle_labels_with_static(
    labels: np.ndarray,
    *,
    static_populations: set[str],
    rng: np.random.Generator,
) -> np.ndarray:
    if not static_populations:
        return rng.permutation(labels)

    shuffled = labels.copy()
    mutable_mask = ~np.isin(labels, list(static_populations))
    mutable_idx = np.flatnonzero(mutable_mask)
    if mutable_idx.size <= 1:
        return shuffled
    shuffled[mutable_idx] = rng.permutation(shuffled[mutable_idx])
    return shuffled


def _bootstrap_summary(
    observed: np.ndarray,
    bootstraps: Optional[np.ndarray],
    *,
    ddof: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if bootstraps is None or bootstraps.size == 0:
        shape = observed.shape
        nan_arr = np.full(shape, np.nan, dtype=float)
        return nan_arr, nan_arr, nan_arr, nan_arr

    if bootstraps.ndim != 2:
        raise ValueError(f"bootstraps must be 2D, got shape {bootstraps.shape}.")

    eff_ddof = 0
    if bootstraps.shape[0] > 1:
        eff_ddof = int(max(0, min(int(ddof), bootstraps.shape[0] - 1)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        boot_mean = np.nanmean(bootstraps, axis=0)
        boot_std = np.nanstd(bootstraps, axis=0, ddof=eff_ddof)

    delta = observed - boot_mean
    zscore = np.divide(
        delta,
        boot_std,
        out=np.full_like(delta, np.nan, dtype=float),
        where=np.isfinite(boot_std) & (boot_std > 0),
    )
    return boot_mean, boot_std, delta, zscore


def _roi_result_to_summary_records(
    *,
    roi_id: str,
    metric_keys: Sequence[MetricKey],
    observed: np.ndarray,
    boot_mean: np.ndarray,
    boot_std: np.ndarray,
    delta: np.ndarray,
    zscore: np.ndarray,
    metric_n_cells: np.ndarray,
    metric_n_edges: np.ndarray,
    bootstrap_n_permutations: int,
    static_populations: set[str],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, (metric, population) in enumerate(metric_keys):
        records.append(
            {
                "summary_level": "roi",
                "roi": roi_id,
                "metric": metric,
                "population": population,
                "observed": _safe_numeric(observed[idx]),
                "bootstrap_mean": _safe_numeric(boot_mean[idx]),
                "bootstrap_std": _safe_numeric(boot_std[idx]),
                "delta": _safe_numeric(delta[idx]),
                "zscore": _safe_numeric(zscore[idx]),
                "metric_n_cells_observed": _safe_numeric(metric_n_cells[idx]),
                "metric_n_edges_observed": _safe_numeric(metric_n_edges[idx]),
                "bootstrap_n_permutations": int(bootstrap_n_permutations),
                "population_static": bool(metric == "average_clustering" and population in static_populations),
            }
        )
    return records


def _case_result_to_summary_records(
    *,
    case_id: str,
    metric_keys: Sequence[MetricKey],
    observed: np.ndarray,
    boot_mean: np.ndarray,
    boot_std: np.ndarray,
    delta: np.ndarray,
    zscore: np.ndarray,
    n_case_rois: int,
    n_case_rois_with_metric: np.ndarray,
    bootstrap_n_permutations: int,
    static_populations: set[str],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, (metric, population) in enumerate(metric_keys):
        records.append(
            {
                "summary_level": "case",
                "case": case_id,
                "metric": metric,
                "population": population,
                "observed": _safe_numeric(observed[idx]),
                "bootstrap_mean": _safe_numeric(boot_mean[idx]),
                "bootstrap_std": _safe_numeric(boot_std[idx]),
                "delta": _safe_numeric(delta[idx]),
                "zscore": _safe_numeric(zscore[idx]),
                "n_case_rois": int(n_case_rois),
                "n_case_rois_with_metric": int(n_case_rois_with_metric[idx]),
                "bootstrap_n_permutations": int(bootstrap_n_permutations),
                "population_static": bool(metric == "average_clustering" and population in static_populations),
            }
        )
    return records


def _bootstrap_long_records(
    *,
    summary_level: str,
    key_name: str,
    key_value: str,
    metric_keys: Sequence[MetricKey],
    bootstrap_values: np.ndarray,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if bootstrap_values.size == 0:
        return records
    for boot_idx in range(bootstrap_values.shape[0]):
        for metric_idx, (metric, population) in enumerate(metric_keys):
            records.append(
                {
                    "summary_level": summary_level,
                    key_name: key_value,
                    "bootstrap_id": int(boot_idx + 1),
                    "metric": metric,
                    "population": population,
                    "value": _safe_numeric(bootstrap_values[boot_idx, metric_idx]),
                }
            )
    return records


def _prepare_roi_obs(
    roi_df: pd.DataFrame,
    *,
    population_obs: str,
    x_coord_obs: str,
    y_coord_obs: str,
    ignore_cells_without_label: bool,
) -> pd.DataFrame:
    if population_obs not in roi_df.columns:
        raise KeyError(f"Population column '{population_obs}' not present in ROI data.")
    if x_coord_obs not in roi_df.columns or y_coord_obs not in roi_df.columns:
        raise KeyError(
            f"ROI data is missing coordinate columns '{x_coord_obs}' and/or '{y_coord_obs}'."
        )

    out = roi_df.copy()
    missing_label_mask = _missing_label_mask(out[population_obs])
    n_missing = int(missing_label_mask.sum())
    if n_missing > 0:
        if not ignore_cells_without_label:
            raise ValueError(
                f"Found {n_missing} cells with missing labels in column '{population_obs}'."
            )
        logging.warning(
            "Dropping %d cells with missing labels from ROI '%s' before NetworkX analysis.",
            n_missing,
            out.iloc[0].get("roi", "<unknown>") if not out.empty else "<unknown>",
        )
        out = out.loc[~missing_label_mask].copy()

    coords = out[[x_coord_obs, y_coord_obs]].to_numpy(dtype=float, copy=False)
    finite_coord_mask = np.isfinite(coords).all(axis=1)
    if not bool(finite_coord_mask.all()):
        n_bad = int((~finite_coord_mask).sum())
        raise ValueError(
            f"Found {n_bad} cells with non-finite coordinates in columns "
            f"'{x_coord_obs}'/'{y_coord_obs}'."
        )

    out[population_obs] = out[population_obs].astype(str)
    return out


def _build_squidpy_graph(
    roi_obs: pd.DataFrame,
    *,
    spatial_key: str,
    population_obs: str,
    x_coord_obs: str,
    y_coord_obs: str,
    graph_coord_type: str,
    graph_delaunay: bool,
    graph_n_neighs: Optional[int],
    graph_radius: Optional[Any],
    graph_percentile: Optional[float],
    graph_transform: Optional[str],
    graph_set_diag: bool,
) -> nx.Graph:
    coords = roi_obs[[x_coord_obs, y_coord_obs]].to_numpy(dtype=np.float32, copy=True)
    adata_roi = ad.AnnData(X=np.zeros((roi_obs.shape[0], 0), dtype=np.float32))
    adata_roi.obs = roi_obs[[population_obs]].copy()
    adata_roi.obsm[spatial_key] = coords

    kwargs: Dict[str, Any] = {
        "coord_type": str(graph_coord_type),
        "delaunay": bool(graph_delaunay),
        "spatial_key": spatial_key,
        "set_diag": bool(graph_set_diag),
    }
    if graph_n_neighs is not None:
        kwargs["n_neighs"] = int(graph_n_neighs)
    if graph_radius is not None:
        kwargs["radius"] = graph_radius
    if graph_percentile is not None:
        kwargs["percentile"] = float(graph_percentile)
    if graph_transform is not None and str(graph_transform).strip().lower() not in {"", "none", "null"}:
        kwargs["transform"] = str(graph_transform).strip()

    sq.gr.spatial_neighbors(adata_roi, **kwargs)

    if "spatial_connectivities" not in adata_roi.obsp:
        raise KeyError("Squidpy did not populate adata.obsp['spatial_connectivities'].")

    connectivities = sparse.csr_matrix(adata_roi.obsp["spatial_connectivities"].copy())
    connectivities.setdiag(0)
    connectivities.eliminate_zeros()
    connectivities = connectivities.maximum(connectivities.T)
    return nx.from_scipy_sparse_array(connectivities, create_using=nx.Graph())


def _analyze_single_roi(
    *,
    roi_id: str,
    metric_keys: Sequence[MetricKey],
    populations: Sequence[str],
    roi_obs: pd.DataFrame,
    population_obs: str,
    x_coord_obs: str,
    y_coord_obs: str,
    spatial_key: str,
    graph_coord_type: str,
    graph_delaunay: bool,
    graph_n_neighs: Optional[int],
    graph_radius: Optional[Any],
    graph_percentile: Optional[float],
    graph_transform: Optional[str],
    graph_set_diag: bool,
    minimum_cells_per_population: int,
    ignore_cells_without_label: bool,
    run_bootstrap: bool,
    bootstrap_n_permutations: int,
    bootstrap_ddof: int,
    static_populations: set[str],
    seed: Optional[int],
) -> Dict[str, Any]:
    prepared = _prepare_roi_obs(
        roi_obs,
        population_obs=population_obs,
        x_coord_obs=x_coord_obs,
        y_coord_obs=y_coord_obs,
        ignore_cells_without_label=ignore_cells_without_label,
    )
    if prepared.empty:
        raise ValueError(f"ROI '{roi_id}' has no cells left after filtering.")

    graph = _build_squidpy_graph(
        prepared,
        spatial_key=spatial_key,
        population_obs=population_obs,
        x_coord_obs=x_coord_obs,
        y_coord_obs=y_coord_obs,
        graph_coord_type=graph_coord_type,
        graph_delaunay=graph_delaunay,
        graph_n_neighs=graph_n_neighs,
        graph_radius=graph_radius,
        graph_percentile=graph_percentile,
        graph_transform=graph_transform,
        graph_set_diag=graph_set_diag,
    )
    node_ids = list(range(graph.number_of_nodes()))
    labels = prepared[population_obs].astype(str).to_numpy(copy=True)

    observed, metric_n_cells, metric_n_edges = _compute_metric_vector(
        graph,
        node_ids=node_ids,
        labels=labels,
        populations=populations,
        minimum_cells_per_population=minimum_cells_per_population,
    )

    bootstrap_values: Optional[np.ndarray] = None
    if run_bootstrap and bootstrap_n_permutations > 0:
        rng = np.random.default_rng(seed)
        bootstrap_values = np.full(
            (int(bootstrap_n_permutations), len(metric_keys)),
            np.nan,
            dtype=float,
        )
        for perm_idx in range(int(bootstrap_n_permutations)):
            shuffled_labels = _shuffle_labels_with_static(
                labels,
                static_populations=static_populations,
                rng=rng,
            )
            perm_values, _, _ = _compute_metric_vector(
                graph,
                node_ids=node_ids,
                labels=shuffled_labels,
                populations=populations,
                minimum_cells_per_population=minimum_cells_per_population,
            )
            bootstrap_values[perm_idx, :] = perm_values

    boot_mean, boot_std, delta, zscore = _bootstrap_summary(
        observed,
        bootstrap_values,
        ddof=bootstrap_ddof,
    )

    return {
        "roi": roi_id,
        "observed": observed,
        "metric_n_cells": metric_n_cells,
        "metric_n_edges": metric_n_edges,
        "bootstrap_values": bootstrap_values,
        "bootstrap_mean": boot_mean,
        "bootstrap_std": boot_std,
        "delta": delta,
        "zscore": zscore,
        "n_cells_total": int(graph.number_of_nodes()),
        "n_edges_total": int(graph.number_of_edges()),
    }


def _aggregate_case_result(
    *,
    roi_results: Sequence[Dict[str, Any]],
    bootstrap_ddof: int,
) -> Dict[str, Any]:
    observed_stack = np.vstack([res["observed"] for res in roi_results])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        observed_case = np.nanmean(observed_stack, axis=0)
    n_case_rois_with_metric = np.sum(np.isfinite(observed_stack), axis=0)

    bootstrap_case: Optional[np.ndarray] = None
    has_bootstrap = bool(roi_results) and all(res.get("bootstrap_values") is not None for res in roi_results)
    if has_bootstrap:
        bootstrap_stack = np.stack([res["bootstrap_values"] for res in roi_results], axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bootstrap_case = np.nanmean(bootstrap_stack, axis=0)

    boot_mean, boot_std, delta, zscore = _bootstrap_summary(
        observed_case,
        bootstrap_case,
        ddof=bootstrap_ddof,
    )

    return {
        "observed": observed_case,
        "bootstrap_values": bootstrap_case,
        "bootstrap_mean": boot_mean,
        "bootstrap_std": boot_std,
        "delta": delta,
        "zscore": zscore,
        "n_case_rois": int(len(roi_results)),
        "n_case_rois_with_metric": n_case_rois_with_metric.astype(int),
    }


def _merge_metadata(
    df: pd.DataFrame,
    *,
    metadata_df: pd.DataFrame,
    key_col: str,
) -> pd.DataFrame:
    if df.empty or metadata_df.empty or key_col not in df.columns:
        return df
    meta = metadata_df.reset_index()
    index_name = str(metadata_df.index.name) if metadata_df.index.name is not None else None
    if key_col not in meta.columns and index_name and index_name in meta.columns:
        meta = meta.rename(columns={index_name: key_col})
    cols_to_add = [c for c in meta.columns if c == key_col or c not in df.columns]
    return df.merge(meta[cols_to_add], on=key_col, how="left")


def run_networkx_spatial_analyses(
    *,
    general_config: GeneralConfig,
    networkx_config: NetworkxSpatialConfig,
) -> Path:
    stage_name = "NetworkXSpatial"

    networkx_config.population_obs = coalesce_config_text(
        networkx_config.population_obs,
        general_config.population_obs_primary,
        default="population",
    )
    networkx_config.roi_obs = coalesce_config_text(
        networkx_config.roi_obs,
        general_config.roi_obs,
        default="ROI",
    )
    networkx_config.case_obs = coalesce_config_text(
        networkx_config.case_obs,
        general_config.case_obs,
    )
    networkx_config.groupby_obs = coalesce_config_text(
        networkx_config.groupby_obs,
        general_config.groupby_obs,
    )
    networkx_config.x_coord_obs = coalesce_config_text(
        networkx_config.x_coord_obs,
        general_config.x_coord_obs,
        default="X_loc",
    )
    networkx_config.y_coord_obs = coalesce_config_text(
        networkx_config.y_coord_obs,
        general_config.y_coord_obs,
        default="Y_loc",
    )
    networkx_config.spatial_key = coalesce_config_text(
        networkx_config.spatial_key,
        general_config.spatial_key,
        default="spatial",
    )
    networkx_config.master_index_obs = coalesce_config_text(
        networkx_config.master_index_obs,
        general_config.master_index_obs,
        default="Master_Index",
    )
    if not networkx_config.metadata_obs_columns:
        networkx_config.metadata_obs_columns = (
            coalesce_config_list(general_config.metadata_obs, default=[]) or []
        )

    input_path = _resolve_input_adata_path(general_config, networkx_config)
    adata, _, skip_stage, _ = load_pipeline_anndata(
        general_config=general_config,
        stage_name=stage_name,
        stage_config=networkx_config,
        override_path=str(input_path),
    )
    if skip_stage:
        logging.info("Skipping NetworkXSpatial stage based on AnnData stage policy.")
        return Path(general_config.qc_folder) / networkx_config.output_subdir
    if adata is None:
        raise FileNotFoundError(f"AnnData not found for NetworkXSpatial stage: {input_path}")

    if networkx_config.population_obs not in adata.obs.columns:
        raise KeyError(
            f"Configured population_obs '{networkx_config.population_obs}' not found in AnnData.obs."
        )
    if networkx_config.roi_obs not in adata.obs.columns:
        raise KeyError(
            f"Configured roi_obs '{networkx_config.roi_obs}' not found in AnnData.obs."
        )
    for col in [networkx_config.x_coord_obs, networkx_config.y_coord_obs]:
        if col not in adata.obs.columns:
            raise KeyError(f"Configured coordinate column '{col}' not found in AnnData.obs.")

    metadata_cols = _resolve_metadata_columns(adata, networkx_config)
    output_root = Path(general_config.qc_folder) / networkx_config.output_subdir
    raw_dir = output_root / "raw_data"
    metadata_dir = output_root / "metadata"
    for folder in [raw_dir, metadata_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    adata.obs.copy().to_csv(metadata_dir / "anndata_obs_snapshot.csv.gz", index=True)

    roi_metadata = _build_key_metadata_table(adata, networkx_config.roi_obs, metadata_cols)
    if not roi_metadata.empty:
        roi_metadata.to_csv(metadata_dir / "roi_metadata.csv")

    case_metadata_cols = [c for c in metadata_cols if c != networkx_config.roi_obs]
    case_metadata = _build_key_metadata_table(adata, networkx_config.case_obs, case_metadata_cols)
    if networkx_config.case_obs and not case_metadata.empty:
        case_metadata.to_csv(metadata_dir / "case_metadata.csv")

    analysis_cols = [
        c
        for c in [
            networkx_config.population_obs,
            networkx_config.roi_obs,
            networkx_config.case_obs,
            networkx_config.groupby_obs,
            networkx_config.x_coord_obs,
            networkx_config.y_coord_obs,
            networkx_config.master_index_obs,
        ]
        if c and c in adata.obs.columns
    ]
    analysis_obs = adata.obs[analysis_cols].copy()
    analysis_obs = analysis_obs.rename(columns={networkx_config.roi_obs: "roi"})
    if networkx_config.case_obs and networkx_config.case_obs in analysis_obs.columns:
        analysis_obs = analysis_obs.rename(columns={networkx_config.case_obs: "case"})
    if networkx_config.groupby_obs and networkx_config.groupby_obs in analysis_obs.columns:
        analysis_obs = analysis_obs.rename(columns={networkx_config.groupby_obs: "groupby"})

    if bool(networkx_config.ignore_cells_without_label):
        missing_mask = _missing_label_mask(analysis_obs[networkx_config.population_obs])
        n_missing = int(missing_mask.sum())
        if n_missing > 0:
            logging.warning(
                "Dropping %d cells with missing labels from the NetworkXSpatial stage input.",
                n_missing,
            )
            analysis_obs = analysis_obs.loc[~missing_mask].copy()

    if analysis_obs.empty:
        raise ValueError("No cells remain for NetworkXSpatial analysis after filtering.")

    populations = _ordered_unique(analysis_obs[networkx_config.population_obs])
    if not populations:
        raise ValueError(
            f"No usable populations found in column '{networkx_config.population_obs}'."
        )
    metric_keys: List[MetricKey] = [("assortativity", "__all__")]
    metric_keys.extend([("average_clustering", pop) for pop in populations])

    static_populations = {str(pop) for pop in (networkx_config.bootstrap_static_populations or [])}
    unknown_static = sorted(static_populations.difference(set(populations)))
    if unknown_static:
        logging.warning(
            "Ignoring bootstrap_static_populations not present in '%s': %s",
            networkx_config.population_obs,
            unknown_static,
        )
        static_populations = static_populations.intersection(set(populations))

    graph_radius = _parse_graph_radius(networkx_config.graph_radius)
    graph_n_neighs = _parse_optional_int(networkx_config.graph_n_neighs)
    run_bootstrap = bool(networkx_config.run_bootstrap) and int(networkx_config.bootstrap_n_permutations) > 0
    roi_ids = [str(x) for x in _ordered_unique(analysis_obs["roi"])]

    roi_summary_path = raw_dir / "networkx_roi_summary.csv"
    case_summary_path = raw_dir / "networkx_case_summary.csv"
    combined_summary_path = raw_dir / "networkx_summary_combined.csv"
    roi_bootstrap_path = raw_dir / "networkx_roi_bootstrap.csv"
    case_bootstrap_path = raw_dir / "networkx_case_bootstrap.csv"

    has_case_level = bool(networkx_config.case_obs and networkx_config.case_obs in adata.obs.columns and "case" in analysis_obs.columns)
    configured_groups = (
        coalesce_config_list(
            getattr(general_config, "groupby_obs_primary_pairwise", None),
            getattr(general_config, "groupby_obs_groups", None),
            default=[],
        )
        or []
    )

    roi_summary_df = pd.DataFrame()
    case_summary_df = pd.DataFrame()
    combined_summary_df = pd.DataFrame()
    analysis_source = "computed"
    n_workers = _resolve_num_workers(int(networkx_config.n_threads), len(roi_ids))

    if bool(networkx_config.reload_saved_results):
        loaded_roi = _load_saved_csv(
            roi_summary_path,
            required_cols=["summary_level", "roi", "metric", "population", "observed", "bootstrap_mean", "delta", "zscore"],
        )
        loaded_case = _load_saved_csv(
            case_summary_path,
            required_cols=["summary_level", "case", "metric", "population", "observed", "bootstrap_mean", "delta", "zscore"],
        ) if has_case_level else pd.DataFrame()

        if loaded_roi is not None and (not has_case_level or loaded_case is not None):
            roi_summary_df = loaded_roi
            case_summary_df = loaded_case if isinstance(loaded_case, pd.DataFrame) else pd.DataFrame()
            loaded_combined = _load_saved_csv(
                combined_summary_path,
                required_cols=["summary_level", "metric", "population", "observed"],
            )
            if loaded_combined is not None:
                combined_summary_df = loaded_combined
            elif not roi_summary_df.empty or not case_summary_df.empty:
                combined_summary_df = pd.concat(
                    [df for df in [roi_summary_df, case_summary_df] if not df.empty],
                    ignore_index=True,
                )
            analysis_source = "loaded"
            logging.info("Reloaded saved NetworkXSpatial summaries from %s", raw_dir)

    if analysis_source != "loaded":
        n_workers = _resolve_num_workers(int(networkx_config.n_threads), len(roi_ids))
        logging.info(
            "Running NetworkXSpatial on %d ROI(s) with %d worker thread(s).",
            len(roi_ids),
            n_workers,
        )

        base_seed = networkx_config.bootstrap_seed
        seed_sequence = None if base_seed is None else np.random.SeedSequence(int(base_seed))
        roi_seed_map: Dict[str, Optional[int]] = {}
        if seed_sequence is None:
            roi_seed_map = {roi_id: None for roi_id in roi_ids}
        else:
            child_sequences = seed_sequence.spawn(len(roi_ids))
            for roi_id, child in zip(roi_ids, child_sequences):
                roi_seed_map[roi_id] = int(child.generate_state(1, dtype=np.uint64)[0])

        roi_results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_map = {}
            for roi_id in roi_ids:
                roi_df = analysis_obs.loc[analysis_obs["roi"].astype(str) == str(roi_id)].copy()
                future = executor.submit(
                    _analyze_single_roi,
                    roi_id=str(roi_id),
                    metric_keys=metric_keys,
                    populations=populations,
                    roi_obs=roi_df,
                    population_obs=networkx_config.population_obs,
                    x_coord_obs=networkx_config.x_coord_obs,
                    y_coord_obs=networkx_config.y_coord_obs,
                    spatial_key=networkx_config.spatial_key,
                    graph_coord_type=networkx_config.graph_coord_type,
                    graph_delaunay=bool(networkx_config.graph_delaunay),
                    graph_n_neighs=graph_n_neighs,
                    graph_radius=graph_radius,
                    graph_percentile=networkx_config.graph_percentile,
                    graph_transform=networkx_config.graph_transform,
                    graph_set_diag=bool(networkx_config.graph_set_diag),
                    minimum_cells_per_population=int(networkx_config.minimum_cells_per_population),
                    ignore_cells_without_label=bool(networkx_config.ignore_cells_without_label),
                    run_bootstrap=run_bootstrap,
                    bootstrap_n_permutations=int(networkx_config.bootstrap_n_permutations),
                    bootstrap_ddof=int(networkx_config.bootstrap_ddof),
                    static_populations=static_populations,
                    seed=roi_seed_map.get(str(roi_id)),
                )
                future_map[future] = str(roi_id)

            completed = 0
            for future in as_completed(future_map):
                roi_id = future_map[future]
                try:
                    roi_results.append(future.result())
                except Exception as exc:
                    raise RuntimeError(f"NetworkXSpatial failed for ROI '{roi_id}': {exc}") from exc
                completed += 1
                logging.info("Completed NetworkXSpatial ROI %d/%d: %s", completed, len(roi_ids), roi_id)

        roi_results.sort(key=lambda item: str(item["roi"]))

        roi_summary_records: List[Dict[str, Any]] = []
        roi_bootstrap_records: List[Dict[str, Any]] = []
        roi_to_case: Dict[str, Any] = {}

        for res in roi_results:
            roi_id = str(res["roi"])
            if "case" in analysis_obs.columns:
                case_series = analysis_obs.loc[analysis_obs["roi"].astype(str) == roi_id, "case"]
                roi_to_case[roi_id] = case_series.iloc[0] if not case_series.empty else None
            else:
                roi_to_case[roi_id] = None

            roi_summary_records.extend(
                _roi_result_to_summary_records(
                    roi_id=roi_id,
                    metric_keys=metric_keys,
                    observed=res["observed"],
                    boot_mean=res["bootstrap_mean"],
                    boot_std=res["bootstrap_std"],
                    delta=res["delta"],
                    zscore=res["zscore"],
                    metric_n_cells=res["metric_n_cells"],
                    metric_n_edges=res["metric_n_edges"],
                    bootstrap_n_permutations=int(networkx_config.bootstrap_n_permutations) if run_bootstrap else 0,
                    static_populations=static_populations,
                )
            )
            if bool(networkx_config.save_bootstrap_samples) and res["bootstrap_values"] is not None:
                roi_bootstrap_records.extend(
                    _bootstrap_long_records(
                        summary_level="roi",
                        key_name="roi",
                        key_value=roi_id,
                        metric_keys=metric_keys,
                        bootstrap_values=res["bootstrap_values"],
                    )
                )

        roi_summary_df = pd.DataFrame(roi_summary_records)
        roi_summary_df = _merge_metadata(
            roi_summary_df,
            metadata_df=roi_metadata,
            key_col="roi",
        )
        roi_summary_df.to_csv(roi_summary_path, index=False)

        if roi_bootstrap_records:
            roi_bootstrap_df = pd.DataFrame(roi_bootstrap_records)
            roi_bootstrap_df = _merge_metadata(
                roi_bootstrap_df,
                metadata_df=roi_metadata,
                key_col="roi",
            )
            roi_bootstrap_df.to_csv(roi_bootstrap_path, index=False)

        case_summary_df = pd.DataFrame()
        if has_case_level:
            case_groups: Dict[str, List[Dict[str, Any]]] = {}
            for res in roi_results:
                roi_id = str(res["roi"])
                case_value = roi_to_case.get(roi_id)
                if pd.isna(case_value):
                    continue
                case_groups.setdefault(str(case_value), []).append(res)

            case_summary_records: List[Dict[str, Any]] = []
            case_bootstrap_records: List[Dict[str, Any]] = []
            for case_id, case_roi_results in sorted(case_groups.items(), key=lambda item: item[0]):
                agg = _aggregate_case_result(
                    roi_results=case_roi_results,
                    bootstrap_ddof=int(networkx_config.bootstrap_ddof),
                )
                case_summary_records.extend(
                    _case_result_to_summary_records(
                        case_id=case_id,
                        metric_keys=metric_keys,
                        observed=agg["observed"],
                        boot_mean=agg["bootstrap_mean"],
                        boot_std=agg["bootstrap_std"],
                        delta=agg["delta"],
                        zscore=agg["zscore"],
                        n_case_rois=int(agg["n_case_rois"]),
                        n_case_rois_with_metric=agg["n_case_rois_with_metric"],
                        bootstrap_n_permutations=int(networkx_config.bootstrap_n_permutations) if run_bootstrap else 0,
                        static_populations=static_populations,
                    )
                )
                if bool(networkx_config.save_bootstrap_samples) and agg["bootstrap_values"] is not None:
                    case_bootstrap_records.extend(
                        _bootstrap_long_records(
                            summary_level="case",
                            key_name="case",
                            key_value=case_id,
                            metric_keys=metric_keys,
                            bootstrap_values=agg["bootstrap_values"],
                        )
                    )

            case_summary_df = pd.DataFrame(case_summary_records)
            case_summary_df = _merge_metadata(
                case_summary_df,
                metadata_df=case_metadata,
                key_col="case",
            )
            case_summary_df.to_csv(case_summary_path, index=False)

            if case_bootstrap_records:
                case_bootstrap_df = pd.DataFrame(case_bootstrap_records)
                case_bootstrap_df = _merge_metadata(
                    case_bootstrap_df,
                    metadata_df=case_metadata,
                    key_col="case",
                )
                case_bootstrap_df.to_csv(case_bootstrap_path, index=False)
        else:
            logging.warning(
                "Skipping case-level aggregation because case_obs is not configured or missing in adata.obs."
            )

        combined_summary_df = pd.concat(
            [df for df in [roi_summary_df, case_summary_df] if not df.empty],
            ignore_index=True,
        ) if (not roi_summary_df.empty or not case_summary_df.empty) else pd.DataFrame()
        if not combined_summary_df.empty:
            combined_summary_df.to_csv(combined_summary_path, index=False)

    plot_details = _save_summary_plots(
        adata=adata,
        roi_summary_df=roi_summary_df,
        case_summary_df=case_summary_df,
        output_root=output_root,
        networkx_config=networkx_config,
        populations=populations,
        groupby_obs=networkx_config.groupby_obs,
        configured_groups=configured_groups,
    )

    run_metadata = {
        "input_adata_path": str(input_path),
        "output_root": str(output_root),
        "analysis_source": analysis_source,
        "reload_saved_results": bool(networkx_config.reload_saved_results),
        "population_obs": networkx_config.population_obs,
        "roi_obs": networkx_config.roi_obs,
        "case_obs": networkx_config.case_obs,
        "groupby_obs": networkx_config.groupby_obs,
        "x_coord_obs": networkx_config.x_coord_obs,
        "y_coord_obs": networkx_config.y_coord_obs,
        "spatial_key": networkx_config.spatial_key,
        "graph_coord_type": networkx_config.graph_coord_type,
        "graph_delaunay": bool(networkx_config.graph_delaunay),
        "graph_n_neighs": graph_n_neighs,
        "graph_radius": graph_radius,
        "graph_percentile": networkx_config.graph_percentile,
        "graph_transform": networkx_config.graph_transform,
        "graph_set_diag": bool(networkx_config.graph_set_diag),
        "minimum_cells_per_population": int(networkx_config.minimum_cells_per_population),
        "run_bootstrap": run_bootstrap,
        "bootstrap_n_permutations": int(networkx_config.bootstrap_n_permutations) if run_bootstrap else 0,
        "bootstrap_static_populations": sorted(static_populations),
        "bootstrap_ddof": int(networkx_config.bootstrap_ddof),
        "bootstrap_seed": networkx_config.bootstrap_seed,
        "n_threads": int(networkx_config.n_threads),
        "resolved_worker_threads": int(n_workers),
        "save_bootstrap_samples": bool(networkx_config.save_bootstrap_samples),
        "ignore_cells_without_label": bool(networkx_config.ignore_cells_without_label),
        "make_plots": bool(networkx_config.make_plots),
        "plot_kind": str(networkx_config.plot_kind),
        "plot_summary_level": str(networkx_config.plot_summary_level),
        "plot_value_columns": list(networkx_config.plot_value_columns or []),
        "all_populations_plot_populations": list(networkx_config.all_populations_plot_populations or []),
        "all_populations_figsize": (
            list(networkx_config.all_populations_figsize)
            if isinstance(networkx_config.all_populations_figsize, (list, tuple))
            else None
        ),
        "plot_details": plot_details,
        "n_rois": int(len(roi_ids)),
        "n_populations": int(len(populations)),
        "populations": list(populations),
    }
    metadata_path = output_root / "networkx_spatial_run_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    adata.uns["networkx_spatial_pipeline"] = run_metadata
    save_pipeline_anndata(
        adata=adata,
        general_config=general_config,
        stage_name=stage_name,
        stage_config=networkx_config,
        override_path=str(input_path),
        extra_details={
            "output_root": str(output_root),
            "run_metadata_path": str(metadata_path),
        },
    )
    logging.info("NetworkXSpatial outputs saved to: %s", output_root)
    return output_root


if __name__ == "__main__":
    pipeline_stage = "NetworkXSpatial"
    config = process_config_with_overrides()
    setup_logging(config.get("logging", {}), pipeline_stage)

    general_config = GeneralConfig(
        **filter_config_for_dataclass(config.get("general", {}), GeneralConfig)
    )
    networkx_config = NetworkxSpatialConfig(
        **filter_config_for_dataclass(config.get("networkx_spatial", {}), NetworkxSpatialConfig)
    )

    run_networkx_spatial_analyses(
        general_config=general_config,
        networkx_config=networkx_config,
    )
