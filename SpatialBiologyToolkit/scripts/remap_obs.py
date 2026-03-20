"""
Simple observation remapping stage.

Two modes are supported:
1. apply: read a CSV remap table and add new columns to adata.obs
2. generate_blank: scaffold a blank remap CSV from a chosen adata.obs column

The first column in the remap CSV is treated as the source obs key. Remaining
columns are added to adata.obs, except helper columns ignored by config
(`ignore_csv_columns_exact`, `ignore_csv_columns_contains`).
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config_and_utils import (
    GeneralConfig,
    RemapObsConfig,
    filter_config_for_dataclass,
    load_pipeline_anndata,
    process_config_with_overrides,
    save_pipeline_anndata,
    setup_logging,
)


def _resolve_csv_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _normalise_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"apply", "generate_blank"}:
        return text
    raise ValueError(
        f"Invalid remap_obs.mode='{value}'. Expected one of: apply, generate_blank."
    )


def _contains_any_token(name: str, tokens: Sequence[str]) -> bool:
    name_l = str(name).strip().lower()
    return any(str(token).strip().lower() in name_l for token in tokens if str(token).strip())


def _stringify_source_value(value: Any, *, integer_like_strings: bool) -> Optional[str]:
    if pd.isna(value):
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if integer_like_strings and re.fullmatch(r"[+-]?\d+(?:\.0+)?", text):
            return str(int(float(text)))
        return text

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, int):
        return str(int(value))

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if integer_like_strings and float(value).is_integer():
            return str(int(value))
        return str(value)

    text = str(value).strip()
    return text or None


def _should_use_integer_like_string_keys(
    source_obs: str,
    remap_config: RemapObsConfig,
) -> bool:
    if bool(remap_config.force_string_mapping):
        return True
    return "leiden" in str(source_obs).strip().lower()


def _ordered_unique_normalised_values(
    series: pd.Series,
    *,
    integer_like_strings: bool,
) -> List[str]:
    if pd.api.types.is_categorical_dtype(series):
        values = list(series.cat.categories)
    else:
        values = list(pd.unique(series.dropna()))

    ordered: List[str] = []
    seen: set[str] = set()
    for value in values:
        key = _stringify_source_value(value, integer_like_strings=integer_like_strings)
        if key is None or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _default_generate_columns(source_obs: str) -> List[str]:
    return [f"{str(source_obs).strip()}_label"]


def _ordered_non_null_values(series: pd.Series) -> List[Any]:
    ordered: List[Any] = []
    seen: set[str] = set()
    for value in series.tolist():
        if pd.isna(value):
            continue
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                continue
            value = cleaned
        marker = repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(value)
    return ordered


def _dedupe_ordered(items: Sequence[Any]) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _read_remap_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Remap CSV not found: {path}")
    df = pd.read_csv(path)
    if df.shape[1] < 1:
        raise ValueError(f"Remap CSV '{path}' does not contain any columns.")
    return df


def _resolve_source_obs(
    remap_df: pd.DataFrame,
    configured_source_obs: Optional[str],
) -> tuple[str, str]:
    file_source_col = str(remap_df.columns[0]).strip()
    configured = str(configured_source_obs).strip() if configured_source_obs else ""
    file_source_is_unnamed = file_source_col.lower().startswith("unnamed:")

    if configured:
        if file_source_col and not file_source_is_unnamed and file_source_col != configured:
            raise ValueError(
                f"remap_obs.source_obs='{configured}' does not match the first column "
                f"of the remap CSV ('{file_source_col}')."
            )
        return configured, file_source_col

    if not file_source_col or file_source_is_unnamed:
        raise ValueError(
            "Could not infer remap source_obs from the CSV first column. "
            "Set remap_obs.source_obs explicitly in config."
        )
    return file_source_col, file_source_col


def _ignored_apply_columns(remap_config: RemapObsConfig) -> set[str]:
    ignored = {str(col).strip() for col in (remap_config.ignore_csv_columns_exact or []) if str(col).strip()}
    ignored.update(_generated_helper_columns(remap_config))
    return ignored


def _generated_helper_columns(remap_config: RemapObsConfig) -> set[str]:
    helper_cols: set[str] = set()
    if bool(remap_config.generate_include_counts) and str(remap_config.generate_count_column_name).strip():
        helper_cols.add(str(remap_config.generate_count_column_name).strip())
    if bool(remap_config.generate_include_top_markers) and str(remap_config.generate_top_markers_column_name).strip():
        helper_cols.add(str(remap_config.generate_top_markers_column_name).strip())
    if (
        bool(remap_config.generate_include_roi_distribution_evenness)
        and str(remap_config.generate_roi_distribution_evenness_column_name).strip()
    ):
        helper_cols.add(str(remap_config.generate_roi_distribution_evenness_column_name).strip())
    return helper_cols


def _prepare_remap_targets(
    remap_df: pd.DataFrame,
    *,
    remap_config: RemapObsConfig,
) -> tuple[List[str], List[str]]:
    ignored_exact = _ignored_apply_columns(remap_config)
    ignored_contains = remap_config.ignore_csv_columns_contains or []

    target_columns: List[str] = []
    ignored_columns: List[str] = []
    for col in remap_df.columns[1:]:
        col_name = str(col).strip()
        if not col_name:
            ignored_columns.append(str(col))
            continue
        if col_name in ignored_exact or _contains_any_token(col_name, ignored_contains):
            ignored_columns.append(col_name)
            continue
        target_columns.append(col_name)
    return target_columns, ignored_columns


def _normalise_source_series(
    series: pd.Series,
    *,
    integer_like_strings: bool,
) -> pd.Series:
    return series.map(
        lambda value: _stringify_source_value(value, integer_like_strings=integer_like_strings)
    )


def _prepare_output_series(series: pd.Series) -> pd.Series:
    out = series.copy()

    def clean(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, str):
            text = value.strip()
            return text if text else pd.NA
        return value

    return out.map(clean)


def _build_column_lookup(
    remap_df: pd.DataFrame,
    *,
    source_key_col: str,
    target_col: str,
    integer_like_strings: bool,
) -> tuple[Dict[str, Any], List[Any]]:
    keys = _normalise_source_series(
        remap_df[source_key_col],
        integer_like_strings=integer_like_strings,
    )
    target_values = _prepare_output_series(remap_df[target_col])

    working = pd.DataFrame(
        {
            "__source_key__": keys,
            "__target_value__": target_values,
        }
    ).dropna(subset=["__source_key__"])

    duplicate_mask = working["__source_key__"].duplicated(keep=False)
    if bool(duplicate_mask.any()):
        duplicates = sorted(pd.unique(working.loc[duplicate_mask, "__source_key__"]).tolist())
        raise ValueError(
            f"Remap CSV contains duplicate source values for '{source_key_col}': {duplicates}"
        )

    lookup = working.set_index("__source_key__")["__target_value__"].to_dict()
    ordered_values = _ordered_non_null_values(working["__target_value__"])
    return lookup, ordered_values


def _existing_values_lookup(
    existing_df: pd.DataFrame,
    *,
    integer_like_strings: bool,
) -> pd.DataFrame:
    if existing_df.empty:
        return pd.DataFrame()

    source_col = str(existing_df.columns[0])
    working = existing_df.copy()
    working["__source_key__"] = _normalise_source_series(
        working[source_col],
        integer_like_strings=integer_like_strings,
    )
    working = working.dropna(subset=["__source_key__"]).drop_duplicates(
        subset=["__source_key__"],
        keep="last",
    )
    return working.set_index("__source_key__")


def _mean_over_rows(matrix: Any) -> np.ndarray:
    if getattr(matrix, "shape", (0, 0))[0] == 0:
        return np.array([], dtype=float)
    mean_value = matrix.mean(axis=0)
    return np.asarray(mean_value).ravel().astype(float, copy=False)


def _marker_names_from_source(
    *,
    var_names: Sequence[Any],
    var_df: Optional[pd.DataFrame],
    marker_label_col: Optional[str],
) -> List[str]:
    base_names = [str(v) for v in var_names]
    if not marker_label_col:
        return base_names
    if var_df is None or marker_label_col not in var_df.columns:
        logging.warning(
            "generate_top_markers_var_column='%s' was not found. Falling back to var_names.",
            marker_label_col,
        )
        return base_names

    labels = pd.Series(var_df[marker_label_col]).copy()
    resolved: List[str] = []
    for default_name, label in zip(base_names, labels.tolist()):
        if pd.isna(label):
            resolved.append(default_name)
            continue
        text = str(label).strip()
        resolved.append(text if text else default_name)
    return resolved


def _resolve_top_marker_matrix(
    *,
    adata: Any,
    remap_config: RemapObsConfig,
) -> Tuple[Any, List[str], str]:
    requested = str(remap_config.generate_top_markers_layer or "").strip()
    marker_label_col = (
        str(remap_config.generate_top_markers_var_column).strip()
        if remap_config.generate_top_markers_var_column
        else None
    )

    if requested:
        req_lower = requested.lower()
        if req_lower == "raw":
            if getattr(adata, "raw", None) is None:
                raise ValueError(
                    "generate_top_markers_layer='raw' was requested, but adata.raw is not available."
                )
            return (
                adata.raw.X,
                _marker_names_from_source(
                    var_names=adata.raw.var_names,
                    var_df=getattr(adata.raw, "var", None),
                    marker_label_col=marker_label_col,
                ),
                "adata.raw.X",
            )
        if req_lower in {"x", "adata.x"}:
            return (
                adata.X,
                _marker_names_from_source(
                    var_names=adata.var_names,
                    var_df=getattr(adata, "var", None),
                    marker_label_col=marker_label_col,
                ),
                "adata.X",
            )
        if requested not in adata.layers:
            raise KeyError(
                f"generate_top_markers_layer='{requested}' was requested but is not present in adata.layers."
            )
        return (
            adata.layers[requested],
            _marker_names_from_source(
                var_names=adata.var_names,
                var_df=getattr(adata, "var", None),
                marker_label_col=marker_label_col,
            ),
            f"adata.layers['{requested}']",
        )

    if bool(remap_config.generate_top_markers_use_raw) and getattr(adata, "raw", None) is not None:
        return (
            adata.raw.X,
            _marker_names_from_source(
                var_names=adata.raw.var_names,
                var_df=getattr(adata.raw, "var", None),
                marker_label_col=marker_label_col,
            ),
            "adata.raw.X",
        )
    return (
        adata.X,
        _marker_names_from_source(
            var_names=adata.var_names,
            var_df=getattr(adata, "var", None),
            marker_label_col=marker_label_col,
        ),
        "adata.X",
    )


def _compute_top_markers_lookup(
    *,
    adata: Any,
    source_keys: pd.Series,
    ordered_source_values: Sequence[str],
    remap_config: RemapObsConfig,
) -> Tuple[Dict[str, Any], str]:
    n_top = int(remap_config.generate_top_markers_n)
    if n_top <= 0:
        return {}, "disabled"

    matrix, marker_names, matrix_source = _resolve_top_marker_matrix(
        adata=adata,
        remap_config=remap_config,
    )
    if len(marker_names) == 0:
        logging.warning("No marker names were available for generate_top_markers. Leaving the column blank.")
        return {}, matrix_source

    separator = str(remap_config.generate_top_markers_separator or "; ")
    lookup: Dict[str, Any] = {}
    source_array = source_keys.to_numpy()
    for source_value in ordered_source_values:
        mask = source_array == source_value
        if not bool(mask.any()):
            lookup[source_value] = pd.NA
            continue

        group_mean = _mean_over_rows(matrix[mask])
        if group_mean.size == 0:
            lookup[source_value] = pd.NA
            continue

        if bool((~mask).any()):
            rest_mean = _mean_over_rows(matrix[~mask])
            score = group_mean - rest_mean
        else:
            score = group_mean

        if score.size == 0:
            lookup[source_value] = pd.NA
            continue

        order = np.argsort(np.where(np.isfinite(score), score, -np.inf))[::-1]
        chosen: List[str] = []
        seen: set[str] = set()
        for idx in order.tolist():
            if not np.isfinite(score[idx]):
                continue
            name = str(marker_names[idx]).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            chosen.append(name)
            if len(chosen) >= n_top:
                break
        lookup[source_value] = separator.join(chosen) if chosen else pd.NA

    return lookup, matrix_source


def _compute_roi_distribution_evenness_lookup(
    *,
    adata: Any,
    source_keys: pd.Series,
    ordered_source_values: Sequence[str],
    roi_obs: str,
) -> Dict[str, Any]:
    if roi_obs not in adata.obs.columns:
        raise KeyError(f"Configured roi_obs '{roi_obs}' not found in adata.obs.")

    roi_series = adata.obs[roi_obs]
    valid_roi_values = roi_series.dropna()
    n_total_rois = int(pd.unique(valid_roi_values).size)
    if n_total_rois == 0:
        logging.warning("No non-null ROI values were found in adata.obs['%s']; leaving ROI evenness blank.", roi_obs)
        return {}

    lookup: Dict[str, Any] = {}
    for source_value in ordered_source_values:
        mask = source_keys == source_value
        roi_counts = roi_series.loc[mask].dropna().value_counts()
        roi_counts = roi_counts.loc[roi_counts > 0]
        if roi_counts.empty:
            lookup[source_value] = pd.NA
            continue
        if n_total_rois <= 1:
            lookup[source_value] = 1.0
            continue
        proportions = roi_counts.to_numpy(dtype=float)
        proportions = proportions / proportions.sum()
        entropy = float(-(proportions * np.log(proportions)).sum())
        evenness = float(entropy / np.log(float(n_total_rois)))
        lookup[source_value] = evenness
    return lookup


def apply_remap_obs(
    *,
    adata: Any,
    remap_config: RemapObsConfig,
) -> Dict[str, Any]:
    remap_path = _resolve_csv_path(remap_config.remap_csv_path)
    remap_df = _read_remap_csv(remap_path)
    source_obs, source_key_col = _resolve_source_obs(remap_df, remap_config.source_obs)

    if source_obs not in adata.obs.columns:
        raise KeyError(f"Configured remap source_obs '{source_obs}' not found in adata.obs.")

    target_columns, ignored_columns = _prepare_remap_targets(remap_df, remap_config=remap_config)
    if not target_columns:
        raise ValueError(
            f"Remap CSV '{remap_path}' does not contain any target columns after ignoring helper columns."
        )

    if not bool(remap_config.overwrite_existing_obs_columns):
        existing = [col for col in target_columns if col in adata.obs.columns]
        if existing:
            raise ValueError(
                "Refusing to overwrite existing adata.obs columns with overwrite_existing_obs_columns=False: "
                + ", ".join(existing)
            )

    integer_like_strings = _should_use_integer_like_string_keys(source_obs, remap_config)
    source_keys = _normalise_source_series(
        adata.obs[source_obs],
        integer_like_strings=integer_like_strings,
    )

    missing_summary: Dict[str, int] = {}
    missing_examples: Dict[str, List[str]] = {}
    applied_columns: List[str] = []

    for target_col in target_columns:
        lookup, ordered_values = _build_column_lookup(
            remap_df,
            source_key_col=source_key_col,
            target_col=target_col,
            integer_like_strings=integer_like_strings,
        )
        mapped = source_keys.map(lookup)
        missing_mask = source_keys.notna() & mapped.isna()
        n_missing = int(missing_mask.sum())
        if n_missing > 0:
            examples = sorted(pd.unique(source_keys.loc[missing_mask]).tolist())[:10]
            missing_summary[target_col] = n_missing
            missing_examples[target_col] = examples
            if bool(remap_config.require_complete_mapping):
                raise ValueError(
                    f"Target column '{target_col}' is missing mappings for {n_missing} source values "
                    f"in adata.obs['{source_obs}'] (examples: {examples})."
                )
            logging.warning(
                "Target column '%s' is missing mappings for %d source values (examples: %s). "
                "Unmapped cells will be set to NA.",
                target_col,
                n_missing,
                examples,
            )

        if bool(remap_config.set_output_as_categorical):
            mapped = pd.Categorical(mapped, categories=ordered_values)
        adata.obs[target_col] = mapped
        applied_columns.append(target_col)
        logging.info(
            "Applied remap column '%s' from adata.obs['%s'] using %d remap rows.",
            target_col,
            source_obs,
            len(lookup),
        )

    return {
        "mode": "apply",
        "remap_csv_path": str(remap_path),
        "source_obs": source_obs,
        "source_key_column": source_key_col,
        "force_string_mapping": bool(integer_like_strings),
        "applied_columns": applied_columns,
        "ignored_columns": ignored_columns,
        "overwrite_existing_obs_columns": bool(remap_config.overwrite_existing_obs_columns),
        "require_complete_mapping": bool(remap_config.require_complete_mapping),
        "set_output_as_categorical": bool(remap_config.set_output_as_categorical),
        "missing_mapping_counts": missing_summary,
        "missing_mapping_examples": missing_examples,
    }


def generate_blank_remap(
    *,
    adata: Any,
    remap_config: RemapObsConfig,
    general_config: GeneralConfig,
) -> Dict[str, Any]:
    source_obs = str(remap_config.source_obs or "").strip()
    if not source_obs:
        raise ValueError("remap_obs.source_obs must be set when mode='generate_blank'.")
    if source_obs not in adata.obs.columns:
        raise KeyError(f"Configured remap source_obs '{source_obs}' not found in adata.obs.")

    remap_path = _resolve_csv_path(remap_config.remap_csv_path)
    remap_path.parent.mkdir(parents=True, exist_ok=True)

    integer_like_strings = _should_use_integer_like_string_keys(source_obs, remap_config)
    source_keys = _normalise_source_series(
        adata.obs[source_obs],
        integer_like_strings=integer_like_strings,
    )
    ordered_source_values = _ordered_unique_normalised_values(
        adata.obs[source_obs],
        integer_like_strings=integer_like_strings,
    )
    if not ordered_source_values:
        raise ValueError(f"No non-null values found in adata.obs['{source_obs}'].")

    count_col = str(remap_config.generate_count_column_name).strip()
    target_columns = _dedupe_ordered(
        remap_config.generate_columns or _default_generate_columns(source_obs)
    )
    note_columns = _dedupe_ordered(remap_config.generate_note_columns or [])
    helper_columns = _generated_helper_columns(remap_config)

    template_df = pd.DataFrame({source_obs: ordered_source_values})
    if bool(remap_config.generate_include_counts) and count_col:
        counts = source_keys.value_counts(dropna=True).to_dict()
        template_df[count_col] = template_df[source_obs].map(counts).astype("Int64")

    top_markers_source = None
    if (
        bool(remap_config.generate_include_top_markers)
        and str(remap_config.generate_top_markers_column_name).strip()
    ):
        top_markers_col = str(remap_config.generate_top_markers_column_name).strip()
        top_marker_lookup, top_markers_source = _compute_top_markers_lookup(
            adata=adata,
            source_keys=source_keys,
            ordered_source_values=ordered_source_values,
            remap_config=remap_config,
        )
        template_df[top_markers_col] = template_df[source_obs].map(top_marker_lookup)

    roi_evenness_obs = str(remap_config.roi_obs or general_config.roi_obs).strip()
    roi_evenness_generated = False
    if (
        bool(remap_config.generate_include_roi_distribution_evenness)
        and str(remap_config.generate_roi_distribution_evenness_column_name).strip()
    ):
        roi_evenness_col = str(remap_config.generate_roi_distribution_evenness_column_name).strip()
        if roi_evenness_obs and roi_evenness_obs in adata.obs.columns:
            roi_evenness_lookup = _compute_roi_distribution_evenness_lookup(
                adata=adata,
                source_keys=source_keys,
                ordered_source_values=ordered_source_values,
                roi_obs=roi_evenness_obs,
            )
            template_df[roi_evenness_col] = template_df[source_obs].map(roi_evenness_lookup)
            roi_evenness_generated = True
        else:
            logging.warning(
                "Could not generate ROI distribution evenness because roi_obs '%s' was not found in adata.obs.",
                roi_evenness_obs or remap_config.roi_obs,
            )

    for col in target_columns:
        template_df[col] = pd.NA
    for col in note_columns:
        template_df[col] = pd.NA

    preserved_columns: List[str] = []
    if bool(remap_config.generate_preserve_existing_values) and remap_path.exists():
        existing_df = pd.read_csv(remap_path)
        existing_lookup = _existing_values_lookup(
            existing_df,
            integer_like_strings=integer_like_strings,
        )
        if not existing_lookup.empty:
            existing_source_col = str(existing_df.columns[0]).strip()
            extra_existing = [
                str(col).strip()
                for col in existing_df.columns[1:]
                if str(col).strip()
                and str(col).strip() != count_col
                and str(col).strip() not in helper_columns
                and str(col).strip() not in target_columns
                and str(col).strip() not in note_columns
            ]
            for col in [*target_columns, *note_columns, *extra_existing]:
                if col not in template_df.columns:
                    template_df[col] = pd.NA
                if col in existing_lookup.columns:
                    template_df[col] = template_df[source_obs].map(existing_lookup[col].to_dict())
                    preserved_columns.append(col)
            if existing_source_col and existing_source_col != source_obs:
                logging.info(
                    "Preserved existing remap values from '%s' while renaming the source key column to '%s'.",
                    existing_source_col,
                    source_obs,
                )

    ordered_cols = [source_obs]
    if bool(remap_config.generate_include_counts) and count_col and count_col in template_df.columns:
        ordered_cols.append(count_col)
    if (
        bool(remap_config.generate_include_top_markers)
        and str(remap_config.generate_top_markers_column_name).strip() in template_df.columns
    ):
        ordered_cols.append(str(remap_config.generate_top_markers_column_name).strip())
    if (
        bool(remap_config.generate_include_roi_distribution_evenness)
        and str(remap_config.generate_roi_distribution_evenness_column_name).strip() in template_df.columns
    ):
        ordered_cols.append(str(remap_config.generate_roi_distribution_evenness_column_name).strip())
    for col in target_columns:
        if col in template_df.columns and col not in ordered_cols:
            ordered_cols.append(col)
    for col in note_columns:
        if col in template_df.columns and col not in ordered_cols:
            ordered_cols.append(col)
    for col in template_df.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)
    template_df = template_df[ordered_cols]

    template_df.to_csv(remap_path, index=False)
    logging.info("Generated blank remap CSV: %s", remap_path)

    return {
        "mode": "generate_blank",
        "remap_csv_path": str(remap_path),
        "source_obs": source_obs,
        "force_string_mapping": bool(integer_like_strings),
        "generated_columns": target_columns,
        "generated_note_columns": note_columns,
        "generated_helper_columns": sorted(helper_columns.intersection(set(template_df.columns))),
        "top_markers_matrix_source": top_markers_source,
        "roi_distribution_evenness_obs": roi_evenness_obs if roi_evenness_generated else None,
        "preserved_columns": sorted(set(preserved_columns)),
        "n_rows": int(template_df.shape[0]),
    }


def run_remap_obs(
    *,
    general_config: GeneralConfig,
    remap_config: RemapObsConfig,
) -> Optional[Path]:
    stage_name = "RemapObs"
    mode = _normalise_mode(remap_config.mode)

    adata, adata_path, skip_stage, _ = load_pipeline_anndata(
        general_config=general_config,
        stage_name=stage_name,
        stage_config=remap_config,
        override_path=remap_config.input_adata_path or general_config.anndata_path,
    )
    if adata is None:
        raise FileNotFoundError(f"AnnData not found for {stage_name}: {adata_path}")

    if skip_stage:
        logging.info(
            "Ignoring AnnData stage-run policy for %s because this stage depends on an external remap CSV/template, "
            "which can change without a config change.",
            stage_name,
        )

    if mode == "generate_blank":
        summary = generate_blank_remap(
            adata=adata,
            remap_config=remap_config,
            general_config=general_config,
        )
        logging.info("Generated remap template summary: %s", summary)
        return _resolve_csv_path(remap_config.remap_csv_path)

    summary = apply_remap_obs(adata=adata, remap_config=remap_config)
    save_pipeline_anndata(
        adata=adata,
        general_config=general_config,
        stage_name=stage_name,
        stage_config=remap_config,
        override_path=str(adata_path),
        extra_details=summary,
    )
    logging.info(
        "Applied remap CSV '%s' to source obs '%s' and added columns: %s",
        summary["remap_csv_path"],
        summary["source_obs"],
        summary["applied_columns"],
    )
    return adata_path


def main() -> None:
    stage_name = "RemapObs"
    config = process_config_with_overrides()
    setup_logging(config.get("logging", {}), stage_name)

    general_config = GeneralConfig(
        **filter_config_for_dataclass(config.get("general", {}), GeneralConfig)
    )
    remap_config = RemapObsConfig(
        **filter_config_for_dataclass(config.get("remap_obs", {}), RemapObsConfig)
    )
    run_remap_obs(general_config=general_config, remap_config=remap_config)


if __name__ == "__main__":
    main()
