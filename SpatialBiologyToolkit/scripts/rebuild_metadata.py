from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import pandas as pd
import tifffile as tiff

from .config_and_utils import (
    GeneralConfig,
    RebuildMetadataConfig,
    filter_config_for_dataclass,
    process_config_with_overrides,
    setup_logging,
)


def _normalise_scalar(value: Any) -> Any:
    """Convert unhashable values so nunique/groupby operations are stable."""
    if isinstance(value, (list, tuple, set, dict)):
        return str(value)
    return value


def _coerce_bool(value: Any) -> Optional[bool]:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    return None


def _make_unique(values: Sequence[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for raw in values:
        base = str(raw) if raw is not None else ""
        base = base if base else "channel"
        count = seen.get(base, 0)
        if count == 0:
            out.append(base)
        else:
            out.append(f"{base}_{count + 1}")
        seen[base] = count + 1
    return out


def _clean_channel_labels(labels: Iterable[Any]) -> List[str]:
    cleaned: List[str] = []
    for value in labels:
        value_str = "" if value is None else str(value)
        clean = re.sub(r"\W+", "", value_str)
        cleaned.append(clean if clean else value_str)
    return cleaned


def _to_string_no_na(series: pd.Series) -> pd.Series:
    as_object = series.astype(object)
    return as_object.where(~pd.isna(as_object), "").astype(str)


def _select_invariant_obs_columns(
    obs: pd.DataFrame,
    roi_obs: str,
    include_patterns: Optional[List[str]],
    exclude_obs: Sequence[str],
    exclude_obs_contains: Sequence[str],
) -> List[str]:
    excluded_exact = {str(x).lower() for x in exclude_obs}
    excluded_substrings = [str(x).lower() for x in exclude_obs_contains]
    include_regex = [re.compile(p) for p in (include_patterns or [])]

    roi_series = obs[roi_obs].astype(str)
    invariant_cols: List[str] = []

    for col in obs.columns:
        col_lower = str(col).lower()
        if col_lower == str(roi_obs).lower():
            continue
        if col_lower in excluded_exact:
            continue
        if any(x in col_lower for x in excluded_substrings):
            continue
        if include_regex and not any(r.search(str(col)) for r in include_regex):
            continue

        col_series = obs[col].map(_normalise_scalar)
        per_roi_nunique = col_series.groupby(roi_series, observed=False).nunique(dropna=True)
        if not per_roi_nunique.empty and int(per_roi_nunique.max()) <= 1:
            invariant_cols.append(str(col))

    return invariant_cols


def _build_roi_level_table(obs: pd.DataFrame, roi_obs: str, columns: Sequence[str]) -> pd.DataFrame:
    tmp = obs[[roi_obs] + list(columns)].copy()
    tmp[roi_obs] = tmp[roi_obs].astype(str)
    grouped = tmp.groupby(roi_obs, sort=True, observed=False)

    roi_index = sorted(tmp[roi_obs].dropna().unique().tolist())
    out = pd.DataFrame(index=roi_index)
    out.index.name = "ROI"

    for col in columns:
        values = grouped[col].agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) > 0 else (s.iloc[0] if len(s) > 0 else pd.NA))
        out[col] = values.reindex(out.index)

    return out


def _panel_from_anndata(adata, cfg: RebuildMetadataConfig) -> pd.DataFrame:
    var = adata.var.copy()
    var_names = [str(x) for x in adata.var_names]

    if cfg.panel_channel_name_var and cfg.panel_channel_name_var in var.columns:
        channel_name = [str(x) for x in var[cfg.panel_channel_name_var].tolist()]
    elif "channel_name" in var.columns:
        channel_name = [str(x) for x in var["channel_name"].tolist()]
    else:
        channel_name = var_names

    if cfg.panel_channel_label_var and cfg.panel_channel_label_var in var.columns:
        channel_label = [str(x) for x in var[cfg.panel_channel_label_var].tolist()]
    elif "channel_label" in var.columns:
        channel_label = [str(x) for x in var["channel_label"].tolist()]
    else:
        channel_label = var_names

    cleaned_labels = _clean_channel_labels(channel_label)
    unique_labels = _make_unique(cleaned_labels)

    panel = pd.DataFrame(
        {
            "channel_name": channel_name,
            "channel_label": unique_labels,
            "use_denoised": bool(cfg.panel_use_denoised_default),
            "to_denoise": bool(cfg.panel_to_denoise_default),
            "use_raw": bool(cfg.panel_use_raw_default),
            "remove_outliers": bool(cfg.panel_remove_outliers_default),
        }
    )
    return panel


def _preserve_existing_panel_flags(panel_df: pd.DataFrame, panel_path: Path) -> pd.DataFrame:
    if not panel_path.exists():
        return panel_df

    try:
        existing = pd.read_csv(panel_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.warning("Could not read existing panel.csv for flag preservation: %s", exc)
        return panel_df

    if "channel_label" not in existing.columns:
        return panel_df

    existing_labels = _make_unique(_clean_channel_labels(existing["channel_label"].tolist()))
    existing = existing.copy()
    existing["channel_label_clean"] = existing_labels

    flag_cols = ["use_denoised", "to_denoise", "use_raw", "remove_outliers"]
    available_flag_cols = [c for c in flag_cols if c in existing.columns]
    if not available_flag_cols:
        return panel_df

    existing_map = existing.set_index("channel_label_clean")[available_flag_cols].to_dict(orient="index")

    for idx, row in panel_df.iterrows():
        channel = str(row["channel_label"])
        if channel not in existing_map:
            continue
        for col in available_flag_cols:
            preserved = _coerce_bool(existing_map[channel].get(col))
            if preserved is not None:
                panel_df.at[idx, col] = preserved
    return panel_df


def _resolve_excluded_obs(general: GeneralConfig, cfg: RebuildMetadataConfig) -> List[str]:
    excluded = {str(x) for x in cfg.exclude_obs}
    excluded.add(str(general.roi_obs))
    excluded.add(str(general.x_coord_obs))
    excluded.add(str(general.y_coord_obs))
    excluded.add(str(general.master_index_obs))
    if general.population_obs_primary:
        excluded.add(str(general.population_obs_primary))
    if general.population_obs_all:
        excluded.update(str(x) for x in general.population_obs_all)
    return sorted(excluded)


def _discover_masks(masks_folder: Path, extensions: Sequence[str]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    if not masks_folder.exists():
        return lookup

    allowed = {ext.lower() for ext in extensions}
    for path in sorted(masks_folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        roi = path.stem
        if roi not in lookup:
            lookup[roi] = path
    return lookup


def _mask_width_height(mask_path: Path) -> Optional[Tuple[float, float]]:
    try:
        mask = tiff.imread(mask_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.warning("Could not read mask '%s' for dimensions: %s", mask_path, exc)
        return None

    if len(mask.shape) < 2:
        logging.warning("Mask '%s' has invalid shape %s; expected at least 2 dimensions.", mask_path, mask.shape)
        return None

    height = float(mask.shape[-2])
    width = float(mask.shape[-1])
    return width, height


def _build_roi_dimensions(rois: Sequence[str], masks_folder: Path) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    mask_lookup = _discover_masks(masks_folder, extensions=[".tiff", ".tif"])
    dimensions: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    missing_masks: List[str] = []
    invalid_shapes: List[str] = []

    for roi in rois:
        mask_path = mask_lookup.get(str(roi))
        if mask_path is None:
            missing_masks.append(str(roi))
            dimensions[str(roi)] = (None, None)
            continue

        width_height = _mask_width_height(mask_path)
        if width_height is None:
            invalid_shapes.append(str(roi))
            dimensions[str(roi)] = (None, None)
            continue
        dimensions[str(roi)] = width_height

    if missing_masks:
        logging.warning(
            "No matching mask found for %d ROI(s) in %s. width_um/height_um left blank for those ROIs.",
            len(missing_masks),
            masks_folder,
        )
    if invalid_shapes:
        logging.warning(
            "%d ROI mask(s) in %s had unreadable/invalid shapes. width_um/height_um left blank for those ROIs.",
            len(invalid_shapes),
            masks_folder,
        )
    return dimensions


def rebuild_metadata_from_anndata(general: GeneralConfig, cfg: RebuildMetadataConfig) -> None:
    adata_path = Path(cfg.input_adata_path or general.anndata_path)
    metadata_folder = Path(cfg.output_metadata_folder or general.metadata_folder)
    metadata_folder.mkdir(parents=True, exist_ok=True)

    if not adata_path.exists():
        raise FileNotFoundError(f"AnnData not found: {adata_path}")

    adata = ad.read_h5ad(adata_path)
    if general.roi_obs not in adata.obs.columns:
        raise KeyError(f"general.roi_obs='{general.roi_obs}' not found in adata.obs")
    if general.case_obs and general.case_obs not in adata.obs.columns:
        logging.warning("general.case_obs='%s' not found in adata.obs", general.case_obs)

    excluded_obs = _resolve_excluded_obs(general, cfg)
    invariant_cols = _select_invariant_obs_columns(
        obs=adata.obs,
        roi_obs=general.roi_obs,
        include_patterns=cfg.include_obs_patterns,
        exclude_obs=excluded_obs,
        exclude_obs_contains=cfg.exclude_obs_contains,
    )

    roi_table = _build_roi_level_table(adata.obs, general.roi_obs, invariant_cols)
    if general.case_obs and general.case_obs in adata.obs.columns and general.case_obs not in invariant_cols:
        logging.warning(
            "general.case_obs='%s' is not ROI-invariant and will not be written as ROI-level metadata.",
            general.case_obs,
        )

    metadata_df = pd.DataFrame({"unstacked_data_folder": roi_table.index.astype(str)})
    if cfg.metadata_description_obs and cfg.metadata_description_obs in roi_table.columns:
        description_vals = _to_string_no_na(roi_table[cfg.metadata_description_obs])
        metadata_df["description"] = description_vals.values
    elif "description" in roi_table.columns:
        metadata_df["description"] = _to_string_no_na(roi_table["description"]).values
    else:
        metadata_df["description"] = metadata_df["unstacked_data_folder"]

    roi_names = metadata_df["unstacked_data_folder"].astype(str).tolist()
    roi_dimensions = _build_roi_dimensions(roi_names, Path(general.masks_folder))
    metadata_df["width_um"] = [roi_dimensions[roi][0] for roi in roi_names]
    metadata_df["height_um"] = [roi_dimensions[roi][1] for roi in roi_names]

    # Rebuild policy: always import all ROIs by default.
    metadata_df["import_data"] = True

    reserved_cols = {"unstacked_data_folder", "description", "width_um", "height_um", "import_data"}
    invariant_write_cols = [c for c in invariant_cols if c not in reserved_cols]
    if cfg.include_invariant_obs_in_metadata_csv:
        for col in invariant_write_cols:
            metadata_df[col] = roi_table[col].reindex(metadata_df["unstacked_data_folder"]).values

    dictionary_df = pd.DataFrame(index=metadata_df["unstacked_data_folder"].astype(str))
    dictionary_df.index.name = "ROI"
    dictionary_df["description"] = metadata_df["description"].values
    if cfg.include_invariant_obs_in_dictionary_csv:
        for col in invariant_write_cols:
            dictionary_df[col] = roi_table[col].reindex(dictionary_df.index).values

    panel_df = _panel_from_anndata(adata, cfg)
    if cfg.preserve_existing_panel_flags:
        panel_df = _preserve_existing_panel_flags(panel_df, metadata_folder / "panel.csv")

    metadata_path = metadata_folder / "metadata.csv"
    dictionary_path = metadata_folder / "dictionary.csv"
    panel_path = metadata_folder / "panel.csv"

    metadata_df.to_csv(metadata_path, index=False)
    dictionary_df.to_csv(dictionary_path)
    panel_df.to_csv(panel_path, index=False)

    logging.info("Rebuilt metadata from %s", adata_path)
    logging.info("Saved %s (%d ROI rows)", metadata_path, len(metadata_df))
    logging.info("Saved %s (%d columns)", dictionary_path, len(dictionary_df.columns))
    logging.info("Saved %s (%d channels)", panel_path, len(panel_df))
    logging.info("Detected %d ROI-invariant obs columns", len(invariant_cols))


def main() -> None:
    pipeline_stage = "RebuildMetadata"
    config = process_config_with_overrides()
    setup_logging(config.get("logging", {}), pipeline_stage)

    general = GeneralConfig(**filter_config_for_dataclass(config.get("general", {}), GeneralConfig))
    rebuild_cfg = RebuildMetadataConfig(
        **filter_config_for_dataclass(config.get("rebuild_metadata", {}), RebuildMetadataConfig)
    )

    rebuild_metadata_from_anndata(general, rebuild_cfg)


if __name__ == "__main__":
    main()
