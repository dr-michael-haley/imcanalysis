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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox, blended_transform_factory

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


def _normalise_enrichment_color_mode(mode: Optional[str]) -> str:
    text = str(mode).strip().lower()
    if text in {"population", "population_color", "population_colors", "target_population"}:
        return "population"
    if text in {"direction", "enrichment", "enrichment_direction"}:
        return "direction"
    logging.warning(
        "Invalid enrichment_plot_color_mode='%s'. Using default 'direction'.",
        mode,
    )
    return "direction"


def _normalise_positive_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    label: str,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        logging.warning("Invalid %s='%s'. Using default %.4f.", label, value, default)
        return float(default)
    if numeric < float(minimum):
        logging.warning(
            "%s=%.4f is smaller than the minimum %.4f. Using %.4f.",
            label,
            numeric,
            minimum,
            minimum,
        )
        return float(minimum)
    return numeric


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
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [str(x) for x in series.cat.categories if pd.notna(x)]

    return [str(x) for x in pd.unique(series.dropna())]


def _obs_order(
    adata: ad.AnnData,
    obs_key: str,
    preferred_order: Optional[Sequence[str]] = None,
) -> List[str]:
    if obs_key not in adata.obs.columns:
        return []
    series = adata.obs[obs_key]
    if isinstance(series.dtype, pd.CategoricalDtype):
        observed = [str(x) for x in series.cat.categories if pd.notna(x)]
    else:
        observed = [str(x) for x in pd.unique(series.dropna())]

    if not preferred_order:
        return observed

    observed_set = set(observed)
    ordered = _dedupe_keep_order([str(x) for x in preferred_order if str(x) in observed_set])
    return ordered or observed


def _population_color_map(
    adata: ad.AnnData, population_obs: str, ordered_pops: Sequence[str]
) -> Dict[str, str]:
    key = f"{population_obs}_colors"
    if key in adata.uns:
        colors = list(adata.uns[key])
        series = adata.obs[population_obs] if population_obs in adata.obs.columns else None
        if series is not None and isinstance(series.dtype, pd.CategoricalDtype):
            categories = [str(x) for x in series.cat.categories if pd.notna(x)]
            if len(colors) >= len(categories):
                full_map = {str(pop): str(colors[i]) for i, pop in enumerate(categories)}
                return {
                    str(pop): full_map.get(str(pop), str(colors[min(i, len(colors) - 1)]))
                    for i, pop in enumerate(ordered_pops)
                }
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


def _safe_write_plot_table_csv(data: pd.DataFrame, path: Path, *, label: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
    except OSError as exc:
        logging.warning(
            "Could not write %s to %s (%s). Continuing without saving this derived CSV.",
            label,
            path,
            exc,
        )


def _filter_dataframe_to_allowed_groups(
    df: Optional[pd.DataFrame],
    *,
    group_col: Optional[str],
    allowed_groups: Optional[Sequence[str]],
    label: str,
) -> Optional[pd.DataFrame]:
    if df is None or df.empty or not group_col or not allowed_groups or group_col not in df.columns:
        return df

    allowed = [str(x) for x in allowed_groups]
    group_values = df[group_col].astype("string")
    mask = group_values.isin(allowed).fillna(False)
    removed = int((~mask).sum())
    filtered = df.loc[mask].copy()
    if removed > 0:
        logging.info(
            "Filtered %d row(s) from %s using %s groups %s.",
            removed,
            label,
            group_col,
            allowed,
        )
    if filtered.empty:
        logging.warning(
            "%s became empty after filtering %s to configured groups %s.",
            label,
            group_col,
            allowed,
        )
        return filtered

    filtered[group_col] = pd.Categorical(
        filtered[group_col].astype(str),
        categories=allowed,
        ordered=True,
    )
    return filtered


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


def _normalise_barplot_scale_mode(value: Any, *, context: str = "") -> Optional[str]:
    text = str(value).strip().lower()
    if text in {"linear", "log", "log1p", "intelligent"}:
        return text
    if context:
        logging.warning(
            "Invalid barplot y-scale '%s' for %s. Valid options are: linear, log, log1p, intelligent.",
            value,
            context,
        )
    return None


def _resolve_barplot_scale_mode(
    scale_cfg: Any,
    *,
    analysis: str,
    metric: str,
) -> str:
    analysis_key = str(analysis)
    metric_key = str(metric)

    if isinstance(scale_cfg, dict):
        # Highest-priority explicit key: "analysis.metric"
        direct_key = f"{analysis_key}.{metric_key}"
        if direct_key in scale_cfg:
            mode = _normalise_barplot_scale_mode(
                scale_cfg.get(direct_key), context=f"{analysis_key}.{metric_key}"
            )
            if mode:
                return mode

        # Analysis-level config: either direct string or nested dict
        analysis_cfg = scale_cfg.get(analysis_key)
        if isinstance(analysis_cfg, dict):
            for key in [metric_key, "default", "*"]:
                if key in analysis_cfg:
                    mode = _normalise_barplot_scale_mode(
                        analysis_cfg.get(key),
                        context=f"{analysis_key}.{key}",
                    )
                    if mode:
                        return mode
        elif analysis_cfg is not None:
            mode = _normalise_barplot_scale_mode(analysis_cfg, context=analysis_key)
            if mode:
                return mode

        # Optional metric-level fallback
        if metric_key in scale_cfg and not isinstance(scale_cfg.get(metric_key), dict):
            mode = _normalise_barplot_scale_mode(scale_cfg.get(metric_key), context=metric_key)
            if mode:
                return mode

        # Global fallback
        for key in ["default", "*"]:
            if key in scale_cfg:
                mode = _normalise_barplot_scale_mode(scale_cfg.get(key), context=key)
                if mode:
                    return mode
        return "linear"

    mode = _normalise_barplot_scale_mode(scale_cfg, context="barplot_y_scale")
    return mode or "linear"


def _resolve_intelligent_scale_params(params: Any) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "allow_log1p": True,
        "dynamic_range_thresh": 100.0,
        "skew_improve_ratio": 0.7,
        "crush_frac_thresh": 0.7,
    }
    if not isinstance(params, dict):
        return defaults

    resolved = defaults.copy()
    if "allow_log1p" in params:
        resolved["allow_log1p"] = bool(params.get("allow_log1p"))
    for key in ["dynamic_range_thresh", "skew_improve_ratio", "crush_frac_thresh"]:
        if key in params:
            try:
                resolved[key] = float(params.get(key))
            except Exception:
                logging.warning(
                    "Invalid barplot_y_scale_intelligent_params['%s']=%s. Using default %s.",
                    key,
                    params.get(key),
                    defaults[key],
                )
    return resolved


def _choose_scale_1d(
    x: Any,
    *,
    allow_log1p: bool = True,
    dynamic_range_thresh: float = 100.0,
    skew_improve_ratio: float = 0.7,
    crush_frac_thresh: float = 0.7,
) -> str:
    """
    Return "linear", "log", or "log1p" for plotting 1D values.
    Uses dynamic range + skewness reduction + linear-axis crush heuristic.

    For axis safety, log is only selected when all finite values are > 0.
    log1p is only selected when all finite values are >= 0 and at least one value is 0.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 10:
        return "linear"

    if np.any(arr < 0):
        return "linear"

    positive = arr[arr > 0]
    if positive.size == 0:
        return "linear"

    min_pos = float(np.min(positive))
    max_pos = float(np.max(arr))
    if not np.isfinite(min_pos) or not np.isfinite(max_pos) or max_pos <= 0:
        return "linear"

    dyn = max_pos / min_pos if min_pos > 0 else np.inf

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if np.isclose(hi, lo):
        return "linear"
    crush_frac = float(np.mean(arr <= lo + 0.05 * (hi - lo)))

    def _skew(values: np.ndarray) -> float:
        m = float(np.mean(values))
        s = float(np.std(values, ddof=0))
        if np.isclose(s, 0.0):
            return 0.0
        z = (values - m) / s
        return float(np.mean(z**3))

    skew_lin = abs(_skew(arr))
    has_zero = bool(np.any(arr == 0))
    if has_zero:
        if not allow_log1p:
            return "linear"
        transformed = np.log1p(arr)
        transformed_mode = "log1p"
    else:
        transformed = np.log(arr)
        transformed_mode = "log"
    skew_transformed = abs(_skew(transformed))

    transformed_helpful = (dyn >= float(dynamic_range_thresh)) and (
        (skew_transformed <= skew_lin * float(skew_improve_ratio))
        or (crush_frac >= float(crush_frac_thresh))
    )
    return transformed_mode if transformed_helpful else "linear"


def _apply_barplot_y_scale(
    ax: Any,
    values: pd.Series,
    *,
    requested_mode: str,
    analysis: str,
    metric: str,
    intelligent_params: Dict[str, Any],
) -> str:
    mode, limits = _resolve_axis_scale_and_limits(
        values,
        requested_mode=requested_mode,
        analysis=analysis,
        metric=metric,
        intelligent_params=intelligent_params,
        axis="y",
        plot_kind="barplot",
        include_linear_limits=False,
    )
    _apply_resolved_axis_scale(ax, axis="y", mode=mode, limits=limits)
    return mode


def _format_axis_tick_label(value: float) -> str:
    numeric = float(value)
    if not np.isfinite(numeric):
        return ""
    rounded = round(numeric)
    if np.isclose(numeric, rounded):
        return str(int(rounded))
    return f"{numeric:g}"


def _log1p_tick_values(
    lo: float,
    hi: float,
    *,
    max_ticks: int = 9,
) -> List[float]:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return []

    lower = max(0.0, float(lo))
    upper = float(hi)
    if upper <= lower:
        return [lower, upper] if upper > lower else [lower]

    candidates: List[float] = []
    for value in [0.0, 1.0, 2.0, 3.0, 5.0]:
        if lower <= value <= upper:
            candidates.append(float(value))

    max_exp = int(np.ceil(np.log10(max(upper, 1.0))))
    for exp in range(0, max_exp + 1):
        scale = 10.0**exp
        for mult in [1.0, 2.0, 5.0]:
            value = mult * scale
            if lower <= value <= upper:
                candidates.append(float(value))

    ordered = sorted(
        {
            float(value)
            for value in candidates
            if np.isfinite(value) and lower <= float(value) <= upper
        }
    )
    if not ordered:
        ordered = [lower, upper]
    if lower <= 0.0 <= upper and 0.0 not in ordered:
        ordered = [0.0] + ordered
    if len(ordered) == 1 and not np.isclose(ordered[0], upper):
        ordered.append(float(upper))
        ordered = sorted(set(ordered))

    if len(ordered) > max_ticks:
        indices = np.linspace(0, len(ordered) - 1, num=max_ticks, dtype=int)
        ordered = [ordered[idx] for idx in sorted(set(indices.tolist()))]

    return ordered


def _apply_log1p_axis_ticks(
    ax: Any,
    *,
    axis_name: str,
    limits: Optional[Tuple[float, float]] = None,
) -> None:
    if limits is None:
        lo, hi = ax.get_xlim() if axis_name == "x" else ax.get_ylim()
    else:
        lo, hi = limits

    ticks = _log1p_tick_values(float(lo), float(hi))
    if not ticks:
        return

    axis_obj = ax.xaxis if axis_name == "x" else ax.yaxis
    axis_obj.set_major_locator(mticker.FixedLocator(ticks))
    axis_obj.set_major_formatter(
        mticker.FuncFormatter(lambda value, pos: _format_axis_tick_label(value))
    )
    axis_obj.set_minor_locator(mticker.NullLocator())


def _compute_axis_limits_1d(
    values: Any,
    *,
    mode: str,
) -> Optional[Tuple[float, float]]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    mode_text = str(mode).strip().lower()
    if mode_text == "log":
        if np.any(arr <= 0):
            return None
        min_pos = float(np.min(arr))
        max_val = float(np.max(arr))
        if not np.isfinite(min_pos) or not np.isfinite(max_val) or max_val <= 0:
            return None
        lower = max(min_pos * 0.9, np.finfo(float).tiny)
        upper = max_val * 1.1
        if not np.isfinite(upper) or upper <= lower:
            upper = lower * 1.1
        return float(lower), float(upper)
    if mode_text == "log1p":
        if np.any(arr < 0):
            return None
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if np.isclose(lo, hi):
            upper = max(1.0, hi + max(0.1 * max(hi, 1.0), 1e-9))
            return 0.0, float(upper)
        pad = max(0.05 * (hi - lo), 1e-9)
        lower = max(0.0, lo - pad)
        upper = hi + pad
        if not np.isfinite(upper) or upper <= lower:
            upper = max(lower + 1.0, hi + 1.0)
        return float(lower), float(upper)

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if np.isclose(lo, hi):
        magnitude = max(abs(lo), abs(hi), 1.0)
        pad = max(0.1 * magnitude, 1e-9)
        return float(lo - pad), float(hi + pad)

    pad = max(0.05 * (hi - lo), 1e-9)
    return float(lo - pad), float(hi + pad)


def _resolve_axis_scale_and_limits(
    values: Any,
    *,
    requested_mode: str,
    analysis: str,
    metric: str,
    intelligent_params: Dict[str, Any],
    axis: str,
    plot_kind: str,
    include_linear_limits: bool,
) -> Tuple[str, Optional[Tuple[float, float]]]:
    mode = str(requested_mode).lower()
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if mode == "intelligent":
        mode = _choose_scale_1d(finite, **intelligent_params)

    axis_label = "x" if str(axis).lower() == "x" else "y"
    if mode == "log":
        if finite.size == 0 or np.any(finite <= 0):
            logging.warning(
                "Requested log %s-scale for %s/%s %s, but values include non-positive entries. Using linear scale.",
                axis_label,
                analysis,
                metric,
                plot_kind,
            )
            mode = "linear"
        else:
            return "log", _compute_axis_limits_1d(finite, mode="log")
    elif mode == "log1p":
        if finite.size == 0 or np.any(finite < 0):
            logging.warning(
                "Requested log1p %s-scale for %s/%s %s, but values include negative entries. Using linear scale.",
                axis_label,
                analysis,
                metric,
                plot_kind,
            )
            mode = "linear"
        else:
            return "log1p", _compute_axis_limits_1d(finite, mode="log1p")

    if include_linear_limits:
        return "linear", _compute_axis_limits_1d(finite, mode="linear")
    return "linear", None


def _apply_resolved_axis_scale(
    ax: Any,
    *,
    axis: str,
    mode: str,
    limits: Optional[Tuple[float, float]] = None,
) -> None:
    axis_name = "x" if str(axis).lower() == "x" else "y"
    mode_text = str(mode).strip().lower()
    if mode_text == "log":
        if axis_name == "x":
            ax.set_xscale("log")
        else:
            ax.set_yscale("log")
    elif mode_text == "log1p":
        if axis_name == "x":
            ax.set_xscale("function", functions=(np.log1p, np.expm1))
        else:
            ax.set_yscale("function", functions=(np.log1p, np.expm1))

    if limits is not None:
        lo, hi = limits
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            if axis_name == "x":
                ax.set_xlim(left=float(lo), right=float(hi))
            else:
                ax.set_ylim(bottom=float(lo), top=float(hi))

    if mode_text == "log1p":
        _apply_log1p_axis_ticks(ax, axis_name=axis_name, limits=limits)


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
    if text in {"lower_right", "upper_left", "off_plot_right"}:
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

    fig = reference_ax.figure
    try:
        renderer = fig.canvas.get_renderer()
    except Exception:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    def _bbox_to_tuple(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
        if bbox is None:
            return None
        vals = [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)]
        if not np.all(np.isfinite(vals)):
            return None
        return vals[0], vals[1], vals[2], vals[3]

    def _axis_tight_box(ax: Any) -> Optional[Tuple[float, float, float, float]]:
        try:
            tight = ax.get_tightbbox(renderer)
        except Exception:
            tight = None
        if tight is None:
            return _bbox_to_tuple(ax.get_position())
        return _bbox_to_tuple(tight.transformed(fig.transFigure.inverted()))

    def _ticklabel_box(axis: str) -> Optional[Tuple[float, float, float, float]]:
        labels = reference_ax.get_xticklabels() if axis == "x" else reference_ax.get_yticklabels()
        windows = []
        for label in labels:
            if not label.get_visible():
                continue
            if not str(label.get_text()).strip():
                continue
            try:
                bb = label.get_window_extent(renderer=renderer)
            except Exception:
                continue
            if bb is None or bb.width <= 0 or bb.height <= 0:
                continue
            windows.append(bb)
        if not windows:
            return None
        union = Bbox.union(windows).transformed(fig.transFigure.inverted())
        return _bbox_to_tuple(union)

    def _intersection_area(
        a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
    ) -> float:
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
        if dx <= 0 or dy <= 0:
            return 0.0
        return float(dx * dy)

    def _outside_penalty(rect: Tuple[float, float, float, float]) -> float:
        return float(
            max(0.0, -rect[0])
            + max(0.0, -rect[1])
            + max(0.0, rect[2] - 1.0)
            + max(0.0, rect[3] - 1.0)
        )

    def _to_rect(x0: float, y0: float, w: float, h: float) -> Tuple[float, float, float, float]:
        return float(x0), float(y0), float(x0 + w), float(y0 + h)

    def _rect_size(rect: Tuple[float, float, float, float]) -> Tuple[float, float]:
        return float(rect[2] - rect[0]), float(rect[3] - rect[1])

    def _score_rect(
        rect: Tuple[float, float, float, float], occupied_boxes: Sequence[Tuple[float, float, float, float]]
    ) -> float:
        overlap = float(sum(_intersection_area(rect, box) for box in occupied_boxes))
        return overlap + (1_000.0 * _outside_penalty(rect))

    ref_box = reference_ax.get_position()
    ref_tight = _axis_tight_box(reference_ax) or (
        float(ref_box.x0),
        float(ref_box.y0),
        float(ref_box.x1),
        float(ref_box.y1),
    )
    x_tick_box = _ticklabel_box("x")
    y_tick_box = _ticklabel_box("y")

    cbar_width = max(0.012, ref_box.width * 0.045)
    cbar_height = max(0.10, ref_box.height * 0.30)
    pad_x = max(0.005, ref_box.width * 0.02)
    pad_y = max(0.005, ref_box.height * 0.02)

    occupied: List[Tuple[float, float, float, float]] = []
    for ax in fig.axes:
        if ax is cbar_ax or not ax.get_visible():
            continue
        box = _axis_tight_box(ax)
        if box is not None:
            occupied.append(box)

    candidates: List[Tuple[float, float, float, float]] = []

    if corner == "off_plot_right":
        # Deterministic right-gutter placement:
        # 1) Find current right-most occupied artist bounds (labels included)
        # 2) If needed, shift visible non-cbar axes left to create space
        # 3) Place cbar in the reserved gutter at upper-right outside the plot area
        visible_axes = [ax for ax in fig.axes if ax is not cbar_ax and ax.get_visible()]
        occupied_boxes = [_axis_tight_box(ax) for ax in visible_axes]
        occupied_boxes = [b for b in occupied_boxes if b is not None]

        occupied_right = max((b[2] for b in occupied_boxes), default=float(ref_tight[2]))
        occupied_top = max((b[3] for b in occupied_boxes), default=float(ref_tight[3]))
        right_start = float(occupied_right + pad_x)
        desired_w = float(cbar_width)
        desired_h = float(min(cbar_height, max(0.08, ref_tight[3] - ref_box.y0)))
        right_limit = 0.995

        need = float((right_start + desired_w) - right_limit)
        if need > 0:
            min_x0 = min((ax.get_position().x0 for ax in visible_axes), default=0.0)
            max_shift = max(0.0, float(min_x0 - 0.01))
            shift = min(need + pad_x, max_shift)
            if shift > 0:
                for ax in visible_axes:
                    pos = ax.get_position()
                    ax.set_position([pos.x0 - shift, pos.y0, pos.width, pos.height])
                fig.canvas.draw()
                try:
                    renderer = fig.canvas.get_renderer()
                except Exception:
                    pass
                occupied_boxes = [_axis_tight_box(ax) for ax in visible_axes]
                occupied_boxes = [b for b in occupied_boxes if b is not None]
                occupied_right = max((b[2] for b in occupied_boxes), default=float(ref_tight[2]))
                occupied_top = max((b[3] for b in occupied_boxes), default=float(ref_tight[3]))
                right_start = float(occupied_right + pad_x)

        # If there is still not enough room, expand figure width to create a safe gutter.
        need_after_shift = float((right_start + desired_w) - right_limit)
        if need_after_shift > 0:
            try:
                w_in, h_in = fig.get_size_inches()
                scale = 1.0 + min(1.0, max(0.02, need_after_shift * 2.0))
                fig.set_size_inches(float(w_in * scale), float(h_in), forward=True)
                fig.canvas.draw()
                try:
                    renderer = fig.canvas.get_renderer()
                except Exception:
                    pass
                occupied_boxes = [_axis_tight_box(ax) for ax in visible_axes]
                occupied_boxes = [b for b in occupied_boxes if b is not None]
                occupied_right = max((b[2] for b in occupied_boxes), default=float(ref_tight[2]))
                occupied_top = max((b[3] for b in occupied_boxes), default=float(ref_tight[3]))
                right_start = float(occupied_right + pad_x)
            except Exception:
                pass

        available_w = float(right_limit - right_start)
        width = float(max(0.008, min(desired_w, available_w)))
        x0 = float(np.clip(right_start, 0.0, max(0.0, 1.0 - width)))
        # keep it strictly to the right of occupied content when possible
        if x0 < right_start and right_start <= 1.0:
            x0 = float(min(right_start, 1.0 - width))

        y0 = float(np.clip(occupied_top - desired_h, 0.0, max(0.0, 1.0 - desired_h)))
        cbar_ax.set_position([x0, y0, width, desired_h])
        return

    elif corner == "upper_left":
        candidates.extend(
            [
                _to_rect(ref_tight[0] - cbar_width - pad_x, ref_tight[3] - cbar_height, cbar_width, cbar_height),
                _to_rect(ref_box.x0 + pad_x, ref_box.y1 - cbar_height - pad_y, cbar_width, cbar_height),
                _to_rect(ref_box.x1 - cbar_width - pad_x, ref_box.y0 + pad_y, cbar_width, cbar_height),
            ]
        )
    else:
        candidates.extend(
            [
                _to_rect(ref_tight[2] + pad_x, ref_box.y0 + pad_y, cbar_width, cbar_height),
                _to_rect(ref_box.x1 - cbar_width - pad_x, ref_box.y0 + pad_y, cbar_width, cbar_height),
            ]
        )
        # Optional fallback for very dense layouts: lower-left whitespace
        # (left of x labels and below y labels), but only after right-side
        # candidates are attempted so lower_right remains lower-right by default.
        if x_tick_box is not None and y_tick_box is not None:
            available_w = float(x_tick_box[0] - 0.01)
            available_h = float(y_tick_box[1] - 0.01)
            if available_w > 0.02 and available_h > 0.06:
                adaptive_w = min(cbar_width, max(0.012, available_w * 0.9))
                adaptive_h = min(cbar_height, max(0.06, available_h * 0.9))
                x1 = x_tick_box[0] - pad_x
                y1 = y_tick_box[1] - pad_y
                candidates.append(_to_rect(x1 - adaptive_w, y1 - adaptive_h, adaptive_w, adaptive_h))
        candidates.extend(
            [
                _to_rect(ref_tight[0] - cbar_width - pad_x, ref_box.y0 + pad_y, cbar_width, cbar_height),
            ]
        )

    best_rect: Optional[Tuple[float, float, float, float]] = None
    best_score: Optional[float] = None
    for rect in candidates:
        score = _score_rect(rect, occupied)
        if best_score is None or score < best_score:
            best_rect = rect
            best_score = score
        if score <= 1e-10:
            break

    if best_rect is None:
        x0 = float(np.clip(ref_box.x1 - cbar_width - pad_x, 0.0, max(0.0, 1.0 - cbar_width)))
        y0 = float(np.clip(ref_box.y0 + pad_y, 0.0, max(0.0, 1.0 - cbar_height)))
        cbar_ax.set_position([x0, y0, cbar_width, cbar_height])
        return

    width, height = _rect_size(best_rect)
    width = float(max(0.008, width))
    height = float(max(0.05, height))
    x0 = float(np.clip(best_rect[0], 0.0, max(0.0, 1.0 - width)))
    y0 = float(np.clip(best_rect[1], 0.0, max(0.0, 1.0 - height)))
    cbar_ax.set_position([x0, y0, width, height])


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
    make_source_target_barplots: bool = True,
    source_target_width_scale: float = 0.35,
    group_color_map: Optional[Dict[str, str]] = None,
    y_scale_mode: str = "linear",
    y_scale_intelligent_params: Optional[Dict[str, Any]] = None,
) -> None:
    if data.empty or not pairs:
        return
    if y_scale_intelligent_params is None:
        y_scale_intelligent_params = _resolve_intelligent_scale_params(None)
    is_pcf_analysis = str(analysis).strip().lower() == "pcf"

    def _ordered_levels(series: pd.Series) -> List[str]:
        if isinstance(series.dtype, pd.CategoricalDtype):
            return [str(x) for x in series.cat.categories.tolist() if pd.notna(x)]
        return sorted(series.dropna().astype(str).unique().tolist())

    def _dedupe_legend(ax: Any, *, title: str) -> None:
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
        seen = set()
        unique_h: List[Any] = []
        unique_l: List[str] = []
        for h, l in zip(handles, labels):
            label = str(l)
            if not label or label.startswith("_") or label in seen:
                continue
            seen.add(label)
            unique_h.append(h)
            unique_l.append(label)
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
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    def _add_pcf_one_line(ax: Any) -> None:
        if not is_pcf_analysis:
            return
        if str(ax.get_yscale()).lower() == "log":
            logging.warning(
                "PCF barplot for metric '%s' is using log scale, so a y=1 reference line cannot be displayed.",
                metric,
            )
            return
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.8, zorder=0)
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(min(float(y_min), 1.0), max(float(y_max), 1.0))

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
            _safe_write_plot_table_csv(subset, raw_path, label="selected-pair raw table")

            fig, ax = plt.subplots(figsize=figsize)
            if group_col and group_col in subset.columns and subset[group_col].notna().any():
                x_col = group_col
                order = (
                    subset[x_col].cat.categories.tolist()
                    if isinstance(subset[x_col].dtype, pd.CategoricalDtype)
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
            _apply_barplot_y_scale(
                ax,
                subset["value"],
                requested_mode=y_scale_mode,
                analysis=analysis,
                metric=metric,
                intelligent_params=y_scale_intelligent_params,
            )
            _add_pcf_one_line(ax)
            ax.grid(False)
            fig.tight_layout()

            plot_path = out_dir / f"{analysis}_{metric}_{pair_stub}{extension}"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=int(dpi), bbox_inches="tight")
            plt.close(fig)

    if not make_source_target_barplots:
        return

    base_width = float(figsize[0])
    base_height = float(figsize[1])
    width_scale = max(0.05, float(source_target_width_scale))

    for source, targets in pairs.items():
        source_subset = data[data["source_population"].astype(str) == str(source)].copy()
        if source_subset.empty:
            continue
        source_subset = source_subset[source_subset["target_population"].astype(str).isin([str(t) for t in targets])]
        if source_subset.empty:
            continue
        source_subset["target_population"] = source_subset["target_population"].astype(str)

        target_order = [str(t) for t in targets if str(t) in source_subset["target_population"].astype(str).unique().tolist()]
        if not target_order:
            target_order = sorted(source_subset["target_population"].dropna().astype(str).unique().tolist())
        if not target_order:
            continue

        source_stub = cleanstring(source)
        raw_path = raw_dir / f"{analysis}_{metric}_{source_stub}_all_targets.csv"
        _safe_write_plot_table_csv(source_subset, raw_path, label="source-target raw table")

        palette = {t: color_map.get(str(t), "#4c72b0") for t in target_order}

        use_group = bool(group_col and group_col in source_subset.columns and source_subset[group_col].notna().any())
        if use_group:
            hue_col = str(group_col)
            source_subset[hue_col] = source_subset[hue_col].astype(str)
            x_col = "target_population"
            x_order = target_order
            hue_order = _ordered_levels(source_subset[hue_col])
            n_groups = max(1, len(hue_order))
            n_targets = max(1, len(target_order))
            # Scale primarily with x categories (targets); groups are dodge bars.
            plot_width = max(base_width, width_scale * n_targets)
            palette_vals = sns.color_palette("tab10", n_colors=n_groups).as_hex()
            if group_color_map:
                palette = {
                    g: str(group_color_map.get(g, palette_vals[i]))
                    for i, g in enumerate(hue_order)
                }
            else:
                palette = {g: str(palette_vals[i]) for i, g in enumerate(hue_order)}

            fig, ax = plt.subplots(figsize=(plot_width, base_height))
            sns.barplot(
                data=source_subset,
                x=x_col,
                y="value",
                hue=hue_col,
                order=x_order,
                hue_order=hue_order,
                errorbar="se" if len(source_subset) > 1 else None,
                palette=palette,
                ax=ax,
            )
            if add_points:
                sns.stripplot(
                    data=source_subset,
                    x=x_col,
                    y="value",
                    hue=hue_col,
                    order=x_order,
                    hue_order=hue_order,
                    dodge=True,
                    palette=palette,
                    size=2.5,
                    alpha=0.6,
                    jitter=0.15,
                    ax=ax,
                )
            ax.tick_params(axis="x", labelrotation=90, fontsize=8)
            _dedupe_legend(ax, title=hue_col)
        else:
            x_col = "target_population"
            order = target_order
            n_targets = max(1, len(target_order))
            plot_width = max(base_width, width_scale * n_targets)

            fig, ax = plt.subplots(figsize=(plot_width, base_height))
            sns.barplot(
                data=source_subset,
                x=x_col,
                y="value",
                hue="target_population",
                order=order,
                hue_order=target_order,
                dodge=False,
                errorbar="se" if len(source_subset) > 1 else None,
                palette=palette,
                ax=ax,
            )
            if add_points:
                sns.stripplot(
                    data=source_subset,
                    x=x_col,
                    y="value",
                    order=order,
                    color="black",
                    size=2.8,
                    alpha=0.6,
                    jitter=0.15,
                    ax=ax,
                )
            ax.tick_params(axis="x", labelrotation=90, fontsize=8)
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

        ax.set_title(f"{source} -> all selected targets", fontsize=10)
        ax.set_xlabel("target_population")
        ax.set_ylabel(value_label)
        _apply_barplot_y_scale(
            ax,
            source_subset["value"],
            requested_mode=y_scale_mode,
            analysis=analysis,
            metric=metric,
            intelligent_params=y_scale_intelligent_params,
        )
        _add_pcf_one_line(ax)
        ax.grid(False)
        fig.tight_layout()

        plot_path = out_dir / f"{analysis}_{metric}_{source_stub}_all_targets{extension}"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)


def _select_enrichment_targets(
    source_subset: pd.DataFrame,
    *,
    analysis: str,
    metric: str,
    top_n: int,
    bottom_n: int,
    restricted_targets: Optional[Sequence[str]] = None,
    exclude_homotypic: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    if source_subset.empty or (top_n <= 0 and bottom_n <= 0):
        return [], [], []

    target_summary = (
        source_subset.groupby("target_population", observed=True)["value"]
        .mean()
        .reset_index()
    )
    target_summary["target_population"] = target_summary["target_population"].astype(str)
    target_summary["value"] = pd.to_numeric(target_summary["value"], errors="coerce")
    target_summary = target_summary.dropna(subset=["value"])
    if target_summary.empty:
        return [], [], []

    if exclude_homotypic and "source_population" in source_subset.columns:
        source_names = {
            str(x)
            for x in source_subset["source_population"].dropna().astype(str).unique().tolist()
        }
        if source_names:
            target_summary = target_summary[
                ~target_summary["target_population"].isin(source_names)
            ].copy()
            if target_summary.empty:
                return [], [], []

    if restricted_targets:
        restricted = _dedupe_keep_order([str(x) for x in restricted_targets])
        available = set(target_summary["target_population"].tolist())
        missing = [target for target in restricted if target not in available]
        if missing:
            logging.warning(
                "Configured enrichment_plot_target_populations were not present for %s/%s: %s",
                analysis,
                metric,
                missing,
            )
        target_summary = target_summary[target_summary["target_population"].isin(restricted)].copy()
        if target_summary.empty:
            return [], [], []

    analysis_key = str(analysis).strip().lower()
    metric_key = str(metric).strip().lower()

    if analysis_key == "distance":
        enriched_pool = target_summary.sort_values("value", ascending=True)
        depleted_pool = target_summary.sort_values("value", ascending=False)
    elif analysis_key == "squidpy" and metric_key == "zscore":
        enriched_pool = target_summary[target_summary["value"] > 0].sort_values("value", ascending=False)
        depleted_pool = target_summary[target_summary["value"] < 0].sort_values("value", ascending=True)
        if enriched_pool.empty and depleted_pool.empty:
            enriched_pool = target_summary.sort_values("value", ascending=False)
            depleted_pool = target_summary.sort_values("value", ascending=True)
    else:
        enriched_pool = target_summary.sort_values("value", ascending=False)
        depleted_pool = target_summary.sort_values("value", ascending=True)

    enriched: List[str] = []
    if top_n > 0:
        for target in enriched_pool["target_population"].tolist():
            if target not in enriched:
                enriched.append(str(target))
            if len(enriched) >= int(top_n):
                break

    depleted: List[str] = []
    if bottom_n > 0:
        for target in depleted_pool["target_population"].tolist():
            target = str(target)
            if target in enriched or target in depleted:
                continue
            depleted.append(target)
            if len(depleted) >= int(bottom_n):
                break

    return enriched + depleted, enriched, depleted


def _save_enrichment_plot(
    data: pd.DataFrame,
    *,
    analysis: str,
    metric: str,
    color_map: Dict[str, str],
    out_dir: Path,
    raw_dir: Path,
    figsize: Tuple[float, float],
    dpi: int,
    extension: str,
    value_label: str,
    top_n: int,
    bottom_n: int,
    restricted_targets: Optional[Sequence[str]] = None,
    group_col: Optional[str] = None,
    color_mode: str = "direction",
    label_box_width: float = 0.03,
    height_per_target: float = 0.25,
    x_scale_mode: str = "linear",
    x_scale_intelligent_params: Optional[Dict[str, Any]] = None,
    share_x_axis_across_groups: bool = False,
    exclude_homotypic: bool = True,
) -> None:
    if data.empty or (top_n <= 0 and bottom_n <= 0):
        return
    if x_scale_intelligent_params is None:
        x_scale_intelligent_params = _resolve_intelligent_scale_params(None)

    analysis_key = str(analysis).strip().lower()
    metric_key = str(metric).strip().lower()
    color_mode = _normalise_enrichment_color_mode(color_mode)
    label_box_width = _normalise_positive_float(
        label_box_width,
        default=0.03,
        minimum=0.005,
        label="enrichment_plot_label_box_width",
    )
    height_per_target = _normalise_positive_float(
        height_per_target,
        default=0.25,
        minimum=0.05,
        label="enrichment_plot_height_per_target",
    )
    base_width = float(figsize[0])
    base_height = float(figsize[1])
    enriched_color = "#2ca02c"
    depleted_color = "#d62728"

    numeric = data.copy()
    numeric["source_population"] = numeric["source_population"].astype(str)
    numeric["target_population"] = numeric["target_population"].astype(str)
    numeric["value"] = pd.to_numeric(numeric["value"], errors="coerce")
    numeric = numeric.dropna(subset=["value", "source_population", "target_population"])
    if analysis_key == "squidpy" and metric_key == "count":
        n_neg = int((numeric["value"] < 0).sum())
        if n_neg > 0:
            logging.warning(
                "Squidpy/count contains %d negative values; clipping to 0 before enrichment plotting.",
                n_neg,
            )
            numeric.loc[numeric["value"] < 0, "value"] = 0.0
    if numeric.empty:
        return

    if group_col and group_col in numeric.columns:
        group_series = numeric[group_col]
        present_groups = set(group_series.dropna().astype(str).unique().tolist())
        if isinstance(group_series.dtype, pd.CategoricalDtype):
            ordered_groups = [
                str(x) for x in group_series.cat.categories if pd.notna(x) and str(x) in present_groups
            ]
        else:
            ordered_groups = sorted(present_groups)
    else:
        group_col = None
        ordered_groups = []

    def _group_outputs(group_name: Optional[str]) -> Tuple[str, str]:
        if group_name is None:
            if group_col:
                return f"{cleanstring(group_col)}_all", "all groups"
            return "all_data", "all data"
        group_label = str(group_name)
        if group_col:
            return f"{cleanstring(group_col)}_{cleanstring(group_label)}", f"{group_col}={group_label}"
        return cleanstring(group_label), group_label

    def _add_reference_line(ax: Any) -> None:
        ref_value: Optional[float] = None
        if metric_key == "zscore":
            ref_value = 0.0
        elif analysis_key == "pcf":
            ref_value = 1.0
        if ref_value is None:
            return
        if str(ax.get_xscale()).lower() == "log" and ref_value <= 0:
            logging.warning(
                "Enrichment plot for %s/%s is using log x-scale, so an x=%s reference line cannot be displayed.",
                analysis,
                metric,
                ref_value,
            )
            return
        ax.axvline(ref_value, color="black", linestyle=":", linewidth=1.0, alpha=0.8, zorder=0)
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(min(float(x_min), ref_value), max(float(x_max), ref_value))

    group_frames: List[Tuple[Optional[str], pd.DataFrame]] = [(None, numeric)]
    if group_col:
        group_values = numeric[group_col].astype("string")
        for group_name in ordered_groups:
            group_frame = numeric[group_values == str(group_name)].copy()
            if not group_frame.empty:
                group_frames.append((str(group_name), group_frame))

    box_height = 0.8
    strip_x = 1.01
    strip_width = label_box_width

    plot_specs: List[Dict[str, Any]] = []
    source_scale_values: Dict[str, List[np.ndarray]] = {}

    for group_name, group_frame in group_frames:
        group_stub, group_title = _group_outputs(group_name)
        for source in _dedupe_keep_order(group_frame["source_population"].tolist()):
            source_subset = group_frame[group_frame["source_population"] == str(source)].copy()
            if source_subset.empty:
                continue

            target_order, enriched_targets, depleted_targets = _select_enrichment_targets(
                source_subset,
                analysis=analysis,
                metric=metric,
                top_n=int(top_n),
                bottom_n=int(bottom_n),
                restricted_targets=restricted_targets,
                exclude_homotypic=exclude_homotypic,
            )
            if not target_order:
                continue

            plot_subset = source_subset[source_subset["target_population"].isin(target_order)].copy()
            if plot_subset.empty:
                continue
            plot_subset["target_population"] = pd.Categorical(
                plot_subset["target_population"].astype(str),
                categories=target_order,
                ordered=True,
            )
            direction_map = {
                target: ("enriched" if target in enriched_targets else "depleted")
                for target in target_order
            }
            plot_subset["enrichment_direction"] = plot_subset["target_population"].astype(str).map(
                direction_map
            )
            plot_subset["target_rank"] = plot_subset["target_population"].astype(str).map(
                {target: idx for idx, target in enumerate(target_order)}
            )

            source_stub = cleanstring(source)
            raw_path = raw_dir / f"{analysis}_{metric}_{source_stub}_{group_stub}_enrichment.csv"
            _safe_write_plot_table_csv(plot_subset, raw_path, label="enrichment raw table")

            plot_specs.append(
                {
                    "group_stub": group_stub,
                    "group_title": group_title,
                    "source": str(source),
                    "source_stub": source_stub,
                    "target_order": list(target_order),
                    "enriched_targets": list(enriched_targets),
                    "depleted_targets": list(depleted_targets),
                    "plot_subset": plot_subset,
                }
            )
            source_scale_values.setdefault(str(source), []).append(
                plot_subset["value"].to_numpy(dtype=float)
            )

    shared_scale_by_source: Dict[str, Tuple[str, Optional[Tuple[float, float]]]] = {}
    if group_col and share_x_axis_across_groups:
        for source, arrays in source_scale_values.items():
            valid_arrays: List[np.ndarray] = []
            for arr in arrays:
                numeric_arr = np.asarray(arr, dtype=float)
                numeric_arr = numeric_arr[np.isfinite(numeric_arr)]
                if numeric_arr.size > 0:
                    valid_arrays.append(numeric_arr)
            if not valid_arrays:
                continue
            combined = np.concatenate(valid_arrays)
            shared_scale_by_source[source] = _resolve_axis_scale_and_limits(
                combined,
                requested_mode=x_scale_mode,
                analysis=analysis,
                metric=metric,
                intelligent_params=x_scale_intelligent_params,
                axis="x",
                plot_kind="enrichment plot",
                include_linear_limits=True,
            )

    for plot_spec in plot_specs:
        source = str(plot_spec["source"])
        source_stub = str(plot_spec["source_stub"])
        group_stub = str(plot_spec["group_stub"])
        group_title = str(plot_spec["group_title"])
        target_order = list(plot_spec["target_order"])
        enriched_targets = list(plot_spec["enriched_targets"])
        depleted_targets = list(plot_spec["depleted_targets"])
        plot_subset = plot_spec["plot_subset"].copy()

        plot_height = max(base_height, height_per_target * max(2, len(target_order)))
        fig, ax = plt.subplots(figsize=(base_width, plot_height))
        if color_mode == "population":
            palette = {target: color_map.get(str(target), "#808080") for target in target_order}
        else:
            palette = {
                target: (enriched_color if target in enriched_targets else depleted_color)
                for target in target_order
            }
        sns.boxplot(
            data=plot_subset,
            x="value",
            y="target_population",
            order=target_order,
            orient="h",
            showfliers=False,
            palette=palette,
            linewidth=0.9,
            width=box_height,
            ax=ax,
        )

        if enriched_targets and depleted_targets:
            ax.axhline(len(enriched_targets) - 0.5, color="black", linewidth=1.0, alpha=0.8)

        if source in shared_scale_by_source:
            shared_mode, shared_limits = shared_scale_by_source[source]
            _apply_resolved_axis_scale(ax, axis="x", mode=shared_mode, limits=shared_limits)
        else:
            mode, limits = _resolve_axis_scale_and_limits(
                plot_subset["value"],
                requested_mode=x_scale_mode,
                analysis=analysis,
                metric=metric,
                intelligent_params=x_scale_intelligent_params,
                axis="x",
                plot_kind="enrichment plot",
                include_linear_limits=False,
            )
            _apply_resolved_axis_scale(ax, axis="x", mode=mode, limits=limits)

        _add_reference_line(ax)
        ax.set_title(f"{source}: enriched / depleted interactions ({group_title})", fontsize=10)
        ax.set_xlabel(value_label)
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=90, fontsize=8)
        ax.grid(False)

        y_positions = ax.get_yticks()
        y_labels = [tick.get_text() for tick in ax.get_yticklabels()]
        label_transform = blended_transform_factory(ax.transAxes, ax.transData)
        for y_pos, label in zip(y_positions, y_labels):
            rect = Rectangle(
                (strip_x, float(y_pos) - (box_height / 2.0)),
                strip_width,
                box_height,
                facecolor=color_map.get(str(label), "#808080"),
                edgecolor="black",
                linewidth=0.3,
                transform=label_transform,
                clip_on=False,
                zorder=4,
            )
            ax.add_patch(rect)

        reserved_right = min(0.20, strip_width + 0.05)
        fig.tight_layout(rect=(0.0, 0.0, 1.0 - reserved_right, 1.0))

        plot_path = out_dir / f"{analysis}_{metric}_{source_stub}_{group_stub}_enrichment{extension}"
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
    pairwise_config.groupby_obs_groups = coalesce_config_list(
        getattr(pairwise_config, "groupby_obs_groups", None),
        general_config.groupby_obs_groups,
        default=[],
    ) or []
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
        "y_coord_obs='%s', master_index_obs='%s', groupby_obs_groups=%s.",
        pairwise_config.population_obs,
        pairwise_config.groupby_obs,
        pairwise_config.roi_obs,
        pairwise_config.x_coord_obs,
        pairwise_config.y_coord_obs,
        pairwise_config.master_index_obs,
        pairwise_config.groupby_obs_groups,
    )

    if pairwise_config.population_obs not in adata.obs.columns:
        raise KeyError(
            f"Configured population_obs '{pairwise_config.population_obs}' not found in AnnData.obs."
        )
    adata.obs[pairwise_config.population_obs] = adata.obs[pairwise_config.population_obs].astype("category")

    allowed_groups: Optional[List[str]] = None
    original_group_color_map: Optional[Dict[str, str]] = None
    if pairwise_config.groupby_obs and pairwise_config.groupby_obs in adata.obs.columns:
        original_group_order = _obs_order(adata, pairwise_config.groupby_obs)
        if original_group_order:
            original_group_color_map = _population_color_map(
                adata,
                pairwise_config.groupby_obs,
                original_group_order,
            )
    if pairwise_config.groupby_obs and pairwise_config.groupby_obs_groups:
        if pairwise_config.groupby_obs not in adata.obs.columns:
            logging.warning(
                "Configured groupby_obs '%s' is missing from AnnData.obs, so groupby_obs_groups=%s cannot be applied.",
                pairwise_config.groupby_obs,
                pairwise_config.groupby_obs_groups,
            )
        else:
            available_groups = set(adata.obs[pairwise_config.groupby_obs].dropna().astype(str).unique().tolist())
            configured_groups = _dedupe_keep_order([str(x) for x in pairwise_config.groupby_obs_groups])
            missing_groups = [group for group in configured_groups if group not in available_groups]
            allowed_groups = [group for group in configured_groups if group in available_groups]
            if missing_groups:
                logging.warning(
                    "Configured groupby_obs_groups for '%s' were not found in AnnData.obs: %s",
                    pairwise_config.groupby_obs,
                    missing_groups,
                )
            if not allowed_groups:
                raise ValueError(
                    "None of the configured groupby_obs_groups were found in AnnData.obs['{}']: {}".format(
                        pairwise_config.groupby_obs,
                        configured_groups,
                    )
                )

            group_mask = adata.obs[pairwise_config.groupby_obs].astype("string").isin(allowed_groups).fillna(False)
            removed_cells = int((~group_mask).sum())
            adata = adata[group_mask].copy()
            if adata.n_obs == 0:
                raise ValueError(
                    "No cells remain after filtering '{}' to groupby_obs_groups={}".format(
                        pairwise_config.groupby_obs,
                        allowed_groups,
                    )
                )
            if original_group_color_map is not None:
                adata.uns[f"{pairwise_config.groupby_obs}_colors"] = [
                    str(original_group_color_map.get(group, "#808080"))
                    for group in allowed_groups
                ]
            adata.obs[pairwise_config.groupby_obs] = pd.Categorical(
                adata.obs[pairwise_config.groupby_obs].astype(str),
                categories=allowed_groups,
                ordered=True,
            )
            logging.info(
                "Filtered PairwiseSpatial input to %d/%d cells using %s groups %s (removed %d cells).",
                int(adata.n_obs),
                int(group_mask.shape[0]),
                pairwise_config.groupby_obs,
                allowed_groups,
                removed_cells,
            )
    elif pairwise_config.groupby_obs and pairwise_config.groupby_obs in adata.obs.columns:
        observed_groups = _obs_order(adata, pairwise_config.groupby_obs)
        if observed_groups:
            allowed_groups = observed_groups

    metadata_cols = _resolve_metadata_columns(adata, pairwise_config)
    source_population_obs = pairwise_config.source_population_obs or pairwise_config.population_obs
    ignore_cells_without_label = bool(pairwise_config.ignore_cells_without_label)

    output_root = Path(general_config.qc_folder) / pairwise_config.output_subdir
    raw_dir = output_root / "raw_data"
    matrix_dir = output_root / "plots" / "pairwise_matrices"
    pair_bar_dir = output_root / "plots" / "selected_pairs"
    enrichment_plot_dir = output_root / "plots" / "enrichment_plots"
    metadata_dir = output_root / "metadata"
    for folder in [raw_dir, matrix_dir, pair_bar_dir, enrichment_plot_dir, metadata_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    extension = _ensure_extension(pairwise_config.figure_extension)
    matrix_figsize = _figsize(pairwise_config.heatmap_figsize, fallback=(8.0, 6.0))
    bar_figsize = _figsize(pairwise_config.barplot_figsize, fallback=(3.0, 3.0))
    enrichment_figsize = _figsize(pairwise_config.enrichment_plot_figsize, fallback=(5.5, 4.0))
    reload_saved_results = bool(pairwise_config.reload_saved_results)
    make_enrichment_plots = bool(pairwise_config.make_enrichment_plots)
    enrichment_top_n = int(pairwise_config.enrichment_plot_top_n)
    enrichment_bottom_n = int(getattr(pairwise_config, "enrichment_plot_bottom_n", enrichment_top_n))
    enrichment_target_populations = coalesce_config_list(
        pairwise_config.enrichment_plot_target_populations,
        default=[],
    ) or []
    enrichment_color_mode = _normalise_enrichment_color_mode(
        getattr(pairwise_config, "enrichment_plot_color_mode", "direction")
    )
    enrichment_label_box_width = _normalise_positive_float(
        getattr(pairwise_config, "enrichment_plot_label_box_width", 0.03),
        default=0.03,
        minimum=0.005,
        label="enrichment_plot_label_box_width",
    )
    enrichment_height_per_target = _normalise_positive_float(
        getattr(pairwise_config, "enrichment_plot_height_per_target", 0.25),
        default=0.25,
        minimum=0.05,
        label="enrichment_plot_height_per_target",
    )
    enrichment_exclude_homotypic = bool(
        getattr(pairwise_config, "enrichment_plot_exclude_homotypic", True)
    )
    enrichment_share_x_axis_across_groups = bool(
        getattr(pairwise_config, "enrichment_plot_share_x_axis_across_groups", False)
    )
    cbar_corner = _normalise_cbar_corner(pairwise_config.pairwise_matrices_cbar_corner)
    barplot_intelligent_params = _resolve_intelligent_scale_params(
        pairwise_config.barplot_y_scale_intelligent_params
    )
    analysis_sources: Dict[str, str] = {"squidpy": "skipped", "distance": "skipped", "pcf": "skipped"}
    logging.info("Pairwise reload_saved_results=%s", reload_saved_results)
    logging.info("Pairwise matrix colorbar corner=%s", cbar_corner)
    logging.info(
        "Pairwise enrichment plots enabled=%s top_n=%d bottom_n=%d color_mode=%s label_box_width=%.4f height_per_target=%.4f exclude_homotypic=%s share_x_axis_across_groups=%s restricted_targets=%s",
        make_enrichment_plots,
        enrichment_top_n,
        enrichment_bottom_n,
        enrichment_color_mode,
        enrichment_label_box_width,
        enrichment_height_per_target,
        enrichment_exclude_homotypic,
        enrichment_share_x_axis_across_groups,
        enrichment_target_populations,
    )
    if make_enrichment_plots and enrichment_top_n <= 0 and enrichment_bottom_n <= 0:
        logging.warning(
            "make_enrichment_plots=True but enrichment_plot_top_n=%d and enrichment_plot_bottom_n=%d. Enrichment plots will be skipped.",
            enrichment_top_n,
            enrichment_bottom_n,
        )
    logging.info(
        "Pairwise distance ignore_cells_without_label=%s",
        ignore_cells_without_label,
    )

    ordered_pops = _population_order(adata, pairwise_config.population_obs)
    color_map = _population_color_map(adata, pairwise_config.population_obs, ordered_pops)
    group_color_map: Optional[Dict[str, str]] = None
    if pairwise_config.groupby_obs and pairwise_config.groupby_obs in adata.obs.columns:
        ordered_groups = _obs_order(adata, pairwise_config.groupby_obs, preferred_order=allowed_groups)
        if ordered_groups:
            if original_group_color_map is not None:
                fallback = sns.color_palette("tab10", n_colors=max(1, len(ordered_groups))).as_hex()
                group_color_map = {
                    str(group): str(original_group_color_map.get(str(group), fallback[i]))
                    for i, group in enumerate(ordered_groups)
                }
            else:
                group_color_map = _population_color_map(adata, pairwise_config.groupby_obs, ordered_groups)
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
                filtered_loaded = _filter_dataframe_to_allowed_groups(
                    loaded,
                    group_col=pairwise_config.groupby_obs,
                    allowed_groups=allowed_groups,
                    label="Squidpy long results",
                )
                if filtered_loaded is None or filtered_loaded.empty:
                    logging.warning(
                        "Reloaded Squidpy results from %s became empty after filtering to %s groups %s. Recomputing Squidpy interactions.",
                        squidpy_long_path,
                        pairwise_config.groupby_obs,
                        allowed_groups,
                    )
                else:
                    squidpy_long = filtered_loaded
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
            squidpy_long = _filter_dataframe_to_allowed_groups(
                squidpy_long,
                group_col=pairwise_config.groupby_obs,
                allowed_groups=allowed_groups,
                label="Squidpy long results",
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

                if make_enrichment_plots and (enrichment_top_n > 0 or enrichment_bottom_n > 0):
                    x_scale_mode = _resolve_barplot_scale_mode(
                        pairwise_config.barplot_y_scale,
                        analysis="squidpy",
                        metric=str(metric),
                    )
                    _save_enrichment_plot(
                        metric_df,
                        analysis="squidpy",
                        metric=metric,
                        color_map=color_map,
                        out_dir=enrichment_plot_dir,
                        raw_dir=squidpy_raw_dir / "enrichment_plots",
                        figsize=enrichment_figsize,
                        dpi=pairwise_config.figure_dpi,
                        extension=extension,
                        value_label=f"Squidpy {metric}",
                        top_n=enrichment_top_n,
                        bottom_n=enrichment_bottom_n,
                        restricted_targets=enrichment_target_populations,
                        group_col=(
                            pairwise_config.groupby_obs
                            if pairwise_config.groupby_obs in metric_df.columns
                            else None
                        ),
                        color_mode=enrichment_color_mode,
                        label_box_width=enrichment_label_box_width,
                        height_per_target=enrichment_height_per_target,
                        x_scale_mode=x_scale_mode,
                        x_scale_intelligent_params=barplot_intelligent_params,
                        share_x_axis_across_groups=enrichment_share_x_axis_across_groups,
                        exclude_homotypic=enrichment_exclude_homotypic,
                    )

                if pairwise_config.make_pair_barplots and pair_map:
                    group_col = (
                        pairwise_config.groupby_obs
                        if pairwise_config.groupby_obs in metric_df.columns
                        else None
                    )
                    barplot_metric_df = metric_df
                    if str(metric) == "count":
                        # Squidpy count values should be non-negative. Guard against
                        # unexpected negatives, but do not force epsilon values, which
                        # can collapse log-scaled plots around ~1e-308.
                        barplot_metric_df = metric_df.copy()
                        barplot_metric_df["value"] = pd.to_numeric(
                            barplot_metric_df["value"],
                            errors="coerce",
                        )
                        neg_mask = barplot_metric_df["value"] < 0
                        n_neg = int(neg_mask.sum())
                        if n_neg > 0:
                            logging.warning(
                                "Squidpy/count contains %d negative values; clipping to 0 before bar plotting.",
                                n_neg,
                            )
                            barplot_metric_df.loc[neg_mask, "value"] = 0.0

                        barplot_metric_df = barplot_metric_df.dropna(subset=["value"])

                    y_scale_mode = _resolve_barplot_scale_mode(
                        pairwise_config.barplot_y_scale,
                        analysis="squidpy",
                        metric=str(metric),
                    )
                    if str(metric) == "count" and str(y_scale_mode).lower() == "log":
                        n_nonpos = int((barplot_metric_df["value"] <= 0).sum())
                        if n_nonpos > 0:
                            logging.info(
                                "Overriding squidpy/count y-scale from log to linear because %d values are <= 0.",
                                n_nonpos,
                            )
                            y_scale_mode = "linear"
                    _save_pair_barplots(
                        barplot_metric_df,
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
                        make_source_target_barplots=pairwise_config.make_source_target_barplots,
                        source_target_width_scale=pairwise_config.source_target_barplot_width_scale,
                        group_color_map=group_color_map,
                        y_scale_mode=y_scale_mode,
                        y_scale_intelligent_params=barplot_intelligent_params,
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
                filtered_loaded = _filter_dataframe_to_allowed_groups(
                    loaded,
                    group_col=pairwise_config.groupby_obs,
                    allowed_groups=allowed_groups,
                    label="Distance long results",
                )
                if filtered_loaded is None or filtered_loaded.empty:
                    logging.warning(
                        "Reloaded distance-bootstrap results from %s became empty after filtering to %s groups %s. Recomputing distance analysis.",
                        distance_long_path,
                        pairwise_config.groupby_obs,
                        allowed_groups,
                    )
                else:
                    distance_long = filtered_loaded
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
                    ignore_cells_without_label=ignore_cells_without_label,
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
            distance_long = _filter_dataframe_to_allowed_groups(
                distance_long,
                group_col=pairwise_config.groupby_obs,
                allowed_groups=allowed_groups,
                label="Distance long results",
            )
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

                if make_enrichment_plots and (enrichment_top_n > 0 or enrichment_bottom_n > 0):
                    x_scale_mode = _resolve_barplot_scale_mode(
                        pairwise_config.barplot_y_scale,
                        analysis="distance",
                        metric=str(metric),
                    )
                    _save_enrichment_plot(
                        metric_df,
                        analysis="distance",
                        metric=metric,
                        color_map=color_map,
                        out_dir=enrichment_plot_dir,
                        raw_dir=distance_raw_dir / "enrichment_plots",
                        figsize=enrichment_figsize,
                        dpi=pairwise_config.figure_dpi,
                        extension=extension,
                        value_label=f"Distance {metric}",
                        top_n=enrichment_top_n,
                        bottom_n=enrichment_bottom_n,
                        restricted_targets=enrichment_target_populations,
                        group_col=(
                            pairwise_config.groupby_obs
                            if pairwise_config.groupby_obs in metric_df.columns
                            else None
                        ),
                        color_mode=enrichment_color_mode,
                        label_box_width=enrichment_label_box_width,
                        height_per_target=enrichment_height_per_target,
                        x_scale_mode=x_scale_mode,
                        x_scale_intelligent_params=barplot_intelligent_params,
                        share_x_axis_across_groups=enrichment_share_x_axis_across_groups,
                        exclude_homotypic=enrichment_exclude_homotypic,
                    )

                if pairwise_config.make_pair_barplots and pair_map:
                    group_col = (
                        pairwise_config.groupby_obs
                        if pairwise_config.groupby_obs in metric_df.columns
                        else None
                    )
                    y_scale_mode = _resolve_barplot_scale_mode(
                        pairwise_config.barplot_y_scale,
                        analysis="distance",
                        metric=str(metric),
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
                        make_source_target_barplots=pairwise_config.make_source_target_barplots,
                        source_target_width_scale=pairwise_config.source_target_barplot_width_scale,
                        group_color_map=group_color_map,
                        y_scale_mode=y_scale_mode,
                        y_scale_intelligent_params=barplot_intelligent_params,
                    )

    if pairwise_config.run_pcf:
        target_distance_um = float(pairwise_config.pcf_target_distance_um)
        pcf_raw_dir = raw_dir / "pcf"
        pcf_spoox_dir = pcf_raw_dir / "spoox_out"
        pcf_summary_dir = pcf_raw_dir / "summary"
        pcf_stats_path = pcf_raw_dir / "pcf_stats.txt"
        pcf_conditions_path = pcf_raw_dir / "pcf_conditions.json"
        pcf_summary_path = pcf_raw_dir / "pcf_summary.csv"
        pcf_long_path = pcf_raw_dir / "pcf_long.csv"
        pcf_roi_summary_path = pcf_raw_dir / "pcf_roi_summary.csv"
        pcf_roi_long_path = pcf_raw_dir / "pcf_roi_long.csv"
        pcf_roi_summary_tsv_path = (
            pcf_summary_dir
            / "paircorrelationfunction"
            / f"pcf_{target_distance_um:.1f}um_roi_summary.tsv"
        )
        for folder in [pcf_raw_dir, pcf_spoox_dir, pcf_summary_dir]:
            folder.mkdir(parents=True, exist_ok=True)

        pcf_summary = None
        pcf_long = None
        pcf_roi_summary = None
        pcf_roi_long = None
        if reload_saved_results:
            loaded_summary = _load_saved_csv(
                pcf_summary_path,
                required_cols=["cell_type_1", "cell_type_2"],
            )
            loaded_roi_summary = _load_saved_csv(
                pcf_roi_summary_path,
                required_cols=["roi", "cell_type_1", "cell_type_2", "g"],
            )
            loaded_roi_long = _load_saved_csv(
                pcf_roi_long_path,
                required_cols=["roi", "source_population", "target_population", "metric", "value"],
            )
            loaded_long = _load_saved_csv(
                pcf_long_path,
                required_cols=["source_population", "target_population", "metric", "value"],
            )
            if loaded_summary is not None and loaded_roi_long is not None:
                pcf_summary = loaded_summary
                pcf_roi_long = loaded_roi_long
                pcf_roi_summary = loaded_roi_summary
                pcf_long = loaded_long
                if pcf_long is None:
                    pcf_long = pcf_summary.melt(
                        id_vars=["condition", "cell_type_1", "cell_type_2"]
                        if "condition" in pcf_summary.columns
                        else ["cell_type_1", "cell_type_2"],
                        value_vars=[
                            c for c in ["g_mean", "g_min", "g_max"] if c in pcf_summary.columns
                        ],
                        var_name="metric",
                        value_name="value",
                    ).rename(
                        columns={
                            "cell_type_1": "source_population",
                            "cell_type_2": "target_population",
                        }
                    )
                    pcf_long.to_csv(pcf_long_path, index=False)
                pcf_summary = _filter_dataframe_to_allowed_groups(
                    pcf_summary,
                    group_col="condition",
                    allowed_groups=allowed_groups,
                    label="PCF summary results",
                )
                pcf_long = _filter_dataframe_to_allowed_groups(
                    pcf_long,
                    group_col="condition",
                    allowed_groups=allowed_groups,
                    label="PCF long results",
                )
                pcf_roi_summary = _filter_dataframe_to_allowed_groups(
                    pcf_roi_summary,
                    group_col="condition",
                    allowed_groups=allowed_groups,
                    label="PCF ROI summary results",
                )
                pcf_roi_long = _filter_dataframe_to_allowed_groups(
                    pcf_roi_long,
                    group_col="condition",
                    allowed_groups=allowed_groups,
                    label="PCF ROI long results",
                )

                if (
                    pcf_summary is None
                    or pcf_summary.empty
                    or pcf_long is None
                    or pcf_long.empty
                    or pcf_roi_long is None
                    or pcf_roi_long.empty
                ):
                    logging.warning(
                        "Reloaded PCF results from %s/%s are unusable after filtering to condition groups %s. Recomputing PCF analysis.",
                        pcf_summary_path,
                        pcf_roi_long_path,
                        allowed_groups,
                    )
                    pcf_summary = None
                    pcf_long = None
                    pcf_roi_summary = None
                    pcf_roi_long = None
                else:
                    analysis_sources["pcf"] = "loaded"
                    logging.info(
                        "Reloaded saved PCF results from %s and %s (ROI-level long).",
                        pcf_summary_path,
                        pcf_roi_long_path,
                    )
            elif (
                loaded_summary is not None
                or loaded_long is not None
                or loaded_roi_summary is not None
                or loaded_roi_long is not None
            ):
                logging.warning(
                    "Found partial saved PCF outputs (summary/long/ROI-level incomplete). "
                    "Recomputing PCF analysis."
                )

        if pcf_summary is None or pcf_roi_long is None:
            logging.info("Running PCF analysis at %.2f um.", target_distance_um)
            pcf_summary = sbt_pcf.run_paircorrelation_at_distance(
                adata=adata,
                population_obs=pairwise_config.population_obs,
                groupby=pairwise_config.groupby_obs,
                target_distance=target_distance_um,
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
                save_roi_level_summary=True,
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

            if not pcf_roi_summary_tsv_path.exists():
                raise FileNotFoundError(
                    "PCF ROI-level summary was not found after run: "
                    f"{pcf_roi_summary_tsv_path}"
                )
            pcf_roi_summary = pd.read_csv(pcf_roi_summary_tsv_path, sep="\t")
            required_roi_cols = {"roi", "cell_type_1", "cell_type_2", "g"}
            missing_roi_cols = sorted(required_roi_cols.difference(pcf_roi_summary.columns))
            if missing_roi_cols:
                raise ValueError(
                    "PCF ROI-level summary is missing required columns: "
                    f"{missing_roi_cols}"
                )
            pcf_roi_summary.to_csv(pcf_roi_summary_path, index=False)

            pcf_roi_long = pcf_roi_summary.rename(
                columns={
                    "cell_type_1": "source_population",
                    "cell_type_2": "target_population",
                    "g": "value",
                }
            ).copy()
            pcf_roi_long["metric"] = "g"
            pcf_roi_long.to_csv(pcf_roi_long_path, index=False)

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
        if pcf_long is not None and "condition" not in pcf_long.columns:
            pcf_long["condition"] = "All"
        if pcf_roi_long is not None and "condition" not in pcf_roi_long.columns:
            pcf_roi_long["condition"] = "All"
        pcf_summary = _filter_dataframe_to_allowed_groups(
            pcf_summary,
            group_col="condition",
            allowed_groups=allowed_groups,
            label="PCF summary results",
        )
        pcf_long = _filter_dataframe_to_allowed_groups(
            pcf_long,
            group_col="condition",
            allowed_groups=allowed_groups,
            label="PCF long results",
        )
        pcf_roi_summary = _filter_dataframe_to_allowed_groups(
            pcf_roi_summary,
            group_col="condition",
            allowed_groups=allowed_groups,
            label="PCF ROI summary results",
        )
        pcf_roi_long = _filter_dataframe_to_allowed_groups(
            pcf_roi_long,
            group_col="condition",
            allowed_groups=allowed_groups,
            label="PCF ROI long results",
        )
        if (
            pcf_roi_summary is not None
            and not roi_metadata.empty
            and "roi" in pcf_roi_summary.columns
        ):
            merge_roi_meta = roi_metadata.reset_index().rename(columns={pairwise_config.roi_obs: "roi"})
            cols_to_add = ["roi"] + [c for c in merge_roi_meta.columns if c != "roi" and c not in pcf_roi_summary.columns]
            pcf_roi_summary = pcf_roi_summary.merge(merge_roi_meta[cols_to_add], on="roi", how="left")
            pcf_roi_summary.to_csv(pcf_roi_summary_path, index=False)
        if (
            pcf_roi_long is not None
            and not roi_metadata.empty
            and "roi" in pcf_roi_long.columns
        ):
            merge_roi_meta = roi_metadata.reset_index().rename(columns={pairwise_config.roi_obs: "roi"})
            cols_to_add = ["roi"] + [c for c in merge_roi_meta.columns if c != "roi" and c not in pcf_roi_long.columns]
            pcf_roi_long = pcf_roi_long.merge(merge_roi_meta[cols_to_add], on="roi", how="left")
            pcf_roi_long.to_csv(pcf_roi_long_path, index=False)

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

        if make_enrichment_plots and (enrichment_top_n > 0 or enrichment_bottom_n > 0) and pcf_roi_long is not None and not pcf_roi_long.empty:
            for metric in sorted(pcf_roi_long["metric"].dropna().unique().tolist()):
                metric_df = pcf_roi_long[pcf_roi_long["metric"] == metric].copy()
                x_scale_mode = _resolve_barplot_scale_mode(
                    pairwise_config.barplot_y_scale,
                    analysis="pcf",
                    metric=str(metric),
                )
                _save_enrichment_plot(
                    metric_df,
                    analysis="pcf",
                    metric=metric,
                    color_map=color_map,
                    out_dir=enrichment_plot_dir,
                    raw_dir=pcf_raw_dir / "enrichment_plots",
                    figsize=enrichment_figsize,
                    dpi=pairwise_config.figure_dpi,
                    extension=extension,
                    value_label=f"PCF {metric}",
                    top_n=enrichment_top_n,
                    bottom_n=enrichment_bottom_n,
                    restricted_targets=enrichment_target_populations,
                    group_col=("condition" if "condition" in metric_df.columns else None),
                    color_mode=enrichment_color_mode,
                    label_box_width=enrichment_label_box_width,
                    height_per_target=enrichment_height_per_target,
                    x_scale_mode=x_scale_mode,
                    x_scale_intelligent_params=barplot_intelligent_params,
                    share_x_axis_across_groups=enrichment_share_x_axis_across_groups,
                    exclude_homotypic=enrichment_exclude_homotypic,
                )

        if pairwise_config.make_pair_barplots and pair_map and pcf_roi_long is not None and not pcf_roi_long.empty:
            for metric in sorted(pcf_roi_long["metric"].dropna().unique().tolist()):
                metric_df = pcf_roi_long[pcf_roi_long["metric"] == metric].copy()
                y_scale_mode = _resolve_barplot_scale_mode(
                    pairwise_config.barplot_y_scale,
                    analysis="pcf",
                    metric=str(metric),
                )
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
                    make_source_target_barplots=pairwise_config.make_source_target_barplots,
                    source_target_width_scale=pairwise_config.source_target_barplot_width_scale,
                    group_color_map=group_color_map,
                    y_scale_mode=y_scale_mode,
                    y_scale_intelligent_params=barplot_intelligent_params,
                )

    run_metadata = {
        "input_adata_path": str(input_path),
        "output_root": str(output_root),
        "population_obs": pairwise_config.population_obs,
        "groupby_obs": pairwise_config.groupby_obs,
        "groupby_obs_groups": allowed_groups,
        "roi_obs": pairwise_config.roi_obs,
        "source_population_obs": source_population_obs,
        "ran_squidpy": bool(pairwise_config.run_squidpy_interactions),
        "ran_distance": bool(pairwise_config.run_distance_bootstrap),
        "ran_pcf": bool(pairwise_config.run_pcf),
        "reload_saved_results": reload_saved_results,
        "pairwise_matrices_cbar_corner": cbar_corner,
        "pairwise_matrices_share_vmax_vmin": bool(pairwise_config.pairwise_matrices_share_vmax_vmin),
        "make_source_target_barplots": bool(pairwise_config.make_source_target_barplots),
        "source_target_barplot_width_scale": float(pairwise_config.source_target_barplot_width_scale),
        "make_enrichment_plots": make_enrichment_plots,
        "enrichment_plot_top_n": enrichment_top_n,
        "enrichment_plot_bottom_n": enrichment_bottom_n,
        "enrichment_plot_target_populations": enrichment_target_populations,
        "enrichment_plot_exclude_homotypic": enrichment_exclude_homotypic,
        "enrichment_plot_share_x_axis_across_groups": enrichment_share_x_axis_across_groups,
        "enrichment_plot_color_mode": enrichment_color_mode,
        "enrichment_plot_label_box_width": enrichment_label_box_width,
        "enrichment_plot_height_per_target": enrichment_height_per_target,
        "barplot_y_scale": pairwise_config.barplot_y_scale,
        "barplot_y_scale_intelligent_params": barplot_intelligent_params,
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
