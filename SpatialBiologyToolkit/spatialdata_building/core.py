"""Planning and construction engine for multimodal SpatialData objects."""

from __future__ import annotations

import copy
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    CellMasks,
    HistologyImages,
    IMCAnnData,
    IMCImages,
    MaxFuseSCRNASeq,
    ModalitySpec,
    PlannedModality,
    RasterElementPlan,
    RegionLabels,
    SpatialDataPlan,
    SpatialDataSpec,
    ValidationIssue,
    ValidationReport,
)


SBT_METADATA_KEY = "spatial_biology_toolkit"
SBT_SCHEMA_VERSION = 3
TABLE_REGION_KEY = "_sbt_region"
TABLE_INSTANCE_KEY = "_sbt_instance_id"
CENTROID_OBS_KEY = "_sbt_obs_name"


def _safe_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value)).encode(
        "ascii", "ignore"
    ).decode("ascii")
    token = re.sub(r"[^0-9A-Za-z_]+", "_", normalized).strip("_")
    token = re.sub(r"_+", "_", token)
    if not token:
        raise ValueError(f"Name {value!r} cannot be converted to a safe key.")
    if token[0].isdigit():
        token = f"item_{token}"
    return token


def _normalise_extensions(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(
        dict.fromkeys(
            value.casefold() if str(value).startswith(".") else f".{value.casefold()}"
            for value in map(str, values)
        )
    )
    if not result:
        raise ValueError("At least one file extension must be provided.")
    return result


def _normalise_chunks(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        chunks = (value, value)
    else:
        if len(value) != 2:
            raise ValueError("raster_chunks must be an integer or a (y, x) pair.")
        chunks = (int(value[0]), int(value[1]))
    if any(item <= 0 for item in chunks):
        raise ValueError("raster_chunks values must be positive.")
    return chunks


def _unique_casefold(items: Iterable[Path], requested: str, *, kind: str) -> Path:
    matches = [item for item in items if item.name.casefold() == requested.casefold()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No {kind} named {requested!r} was found.")
    raise ValueError(
        f"Multiple case-insensitive {kind} matches were found for {requested!r}: "
        + ", ".join(str(item) for item in matches)
    )


def _match_roi_file(
    folder: str | Path,
    roi: str,
    *,
    suffix: str,
    extensions: Sequence[str],
) -> Path:
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root}")
    allowed = set(_normalise_extensions(extensions))
    requested_stem = f"{roi}{suffix}".casefold()
    matches = [
        path
        for path in root.iterdir()
        if path.is_file()
        and path.suffix.casefold() in allowed
        and path.stem.casefold() == requested_stem
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No file with stem {roi + suffix!r} and extension in "
            f"{sorted(allowed)} was found in {root}."
        )
    raise ValueError(
        f"Multiple files match ROI {roi!r} and suffix {suffix!r}: "
        + ", ".join(path.name for path in matches)
    )


def _bounded_marker_pattern(marker: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![0-9A-Za-z]){re.escape(marker)}(?![0-9A-Za-z])",
        flags=re.IGNORECASE,
    )


def _match_channel_file(
    folder: Path,
    channel: str,
    *,
    extensions: Sequence[str],
    mode: str,
) -> tuple[Path, str]:
    if not folder.is_dir():
        raise FileNotFoundError(f"ROI image folder not found: {folder}")
    allowed = set(_normalise_extensions(extensions))
    files = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.casefold() in allowed
    )
    exact = [path for path in files if path.stem.casefold() == channel.casefold()]
    if len(exact) == 1:
        return exact[0], "exact"
    if len(exact) > 1:
        raise ValueError(
            f"Multiple exact files match channel {channel!r} in {folder}: "
            + ", ".join(path.name for path in exact)
        )
    if mode == "exact":
        raise FileNotFoundError(
            f"No exact image file matches channel {channel!r} in {folder}."
        )
    pattern = _bounded_marker_pattern(channel)
    candidates = [path for path in files if pattern.search(path.stem)]
    if len(candidates) == 1:
        return candidates[0], "substring"
    if not candidates:
        raise FileNotFoundError(
            f"No image file contains bounded channel {channel!r} in {folder}."
        )
    raise ValueError(
        f"Multiple image files contain bounded channel {channel!r} in {folder}: "
        + ", ".join(path.name for path in candidates)
    )


def _tiff_metadata(path: Path) -> tuple[tuple[int, int], Any]:
    import tifffile

    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        shape = tuple(int(value) for value in series.shape)
        dtype = series.dtype
    if len(shape) != 2:
        raise ValueError(f"Expected a 2D TIFF at {path}, found shape {shape}.")
    return (shape[0], shape[1]), dtype


def _histology_metadata(path: Path) -> tuple[tuple[int, int], int, Any]:
    import imageio.v3 as iio
    import numpy as np

    properties = iio.improps(path)
    shape = tuple(int(value) for value in properties.shape)
    if len(shape) != 3 or (
        shape[-1] not in {3, 4} and shape[0] not in {3, 4}
    ):
        raise ValueError(
            f"Histology image {path} must have shape (y, x, 3|4) or "
            f"(3|4, y, x), found {shape}."
        )
    dtype = properties.dtype
    if not (
        np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)
    ):
        raise TypeError(f"Histology image {path} has unsupported dtype {dtype}.")
    if shape[-1] in {3, 4}:
        return (shape[0], shape[1]), shape[-1], dtype
    return (shape[1], shape[2]), shape[0], dtype


def _read_adata(source: Any, *, backed: bool) -> tuple[Any, bool]:
    import anndata as ad

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"AnnData file not found: {path}")
        return ad.read_h5ad(path, backed="r" if backed else None), True
    required = ("obs", "var_names", "obs_names", "n_obs", "n_vars")
    if not all(hasattr(source, name) for name in required):
        raise TypeError(
            "adata must be an AnnData-compatible object or a path to an .h5ad file."
        )
    return source, False


def _close_adata(adata: Any, owned: bool) -> None:
    if not owned:
        return
    file_manager = getattr(adata, "file", None)
    if file_manager is not None:
        file_manager.close()


def _copy_adata(source: Any) -> Any:
    adata, owned = _read_adata(source, backed=False)
    try:
        if getattr(adata, "isbacked", False):
            return adata.to_memory()
        return adata.copy()
    finally:
        _close_adata(adata, owned)


def _prepare_table_adata(source: Any, *, copy_adata: bool) -> Any:
    if copy_adata:
        return _copy_adata(source)
    if isinstance(source, (str, Path)):
        import anndata as ad

        return ad.read_h5ad(Path(source))
    return source


def _table_name_changes(adata: Any) -> tuple[dict[str, str], ...]:
    """Predict the exact key changes made by SpatialData's table sanitizer."""

    from spatialdata import sanitize_name

    changes: list[dict[str, str]] = []
    for attribute in ("obs", "var", "obsm", "obsp", "varm", "varp", "uns", "layers"):
        used: set[str] = set()
        for raw_name in getattr(adata, attribute).keys():
            original = str(raw_name)
            candidate = sanitize_name(
                original,
                is_dataframe_column=attribute in {"obs", "var"},
            )
            base = candidate
            counter = 1
            while candidate.casefold() in used:
                candidate = f"{base}_{counter}"
                counter += 1
            used.add(candidate.casefold())
            if candidate != original:
                changes.append(
                    {
                        "attribute": attribute,
                        "original": original,
                        "sanitized": candidate,
                    }
                )
    return tuple(changes)


def _sanitize_table_for_spatialdata(adata: Any, *, modality: str) -> None:
    """Sanitize AnnData attribute keys and retain an auditable rename record."""

    from spatialdata import sanitize_table

    changes = _table_name_changes(adata)
    if not changes:
        return
    sanitize_table(adata, inplace=True)
    existing_key = next(
        (
            str(key)
            for key in adata.uns
            if str(key).casefold() == SBT_METADATA_KEY.casefold()
        ),
        None,
    )
    if existing_key is not None and existing_key != SBT_METADATA_KEY:
        adata.uns[SBT_METADATA_KEY] = adata.uns.pop(existing_key)
    raw_metadata = adata.uns.get(SBT_METADATA_KEY, {})
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    metadata["table_name_sanitization"] = [dict(change) for change in changes]
    adata.uns[SBT_METADATA_KEY] = metadata
    logging.info(
        "Sanitized %d AnnData attribute name(s) for SpatialData modality %r.",
        len(changes),
        modality,
    )


def _report_table_name_changes(
    adata: Any,
    *,
    context: _PlanningContext,
    modality: str,
) -> tuple[dict[str, str], ...]:
    changes = _table_name_changes(adata)
    if changes:
        examples = [
            f"{change['attribute']}/{change['original']} -> {change['sanitized']}"
            for change in changes[:10]
        ]
        context.issue(
            "warning",
            "table_names_will_be_sanitized",
            f"SpatialData requires {len(changes)} AnnData attribute name(s) to "
            f"be sanitized during construction. First changes: {examples}",
            modality=modality,
        )
    return changes


def _ordered_strings(values: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _canonical_reference_rois(
    values: Sequence[str],
    *,
    reference_rois: Sequence[str],
    modality_kind: str,
) -> tuple[str, ...]:
    """Resolve requested ROI spelling against a reference modality."""

    lookup: dict[str, str] = {}
    for roi in reference_rois:
        folded = str(roi).casefold()
        if folded in lookup:
            raise ValueError(
                f"Reference ROI names are not unique case-insensitively: "
                f"{lookup[folded]!r}, {roi!r}."
            )
        lookup[folded] = str(roi)
    requested = tuple(str(value) for value in values)
    if not requested:
        raise ValueError(f"{modality_kind} ROIs must not be empty.")
    if len({value.casefold() for value in requested}) != len(requested):
        raise ValueError(
            f"{modality_kind} ROI names must be unique case-insensitively."
        )
    unknown = [value for value in requested if value.casefold() not in lookup]
    if unknown:
        raise ValueError(
            f"{modality_kind} contains ROI(s) absent from the reference: "
            + ", ".join(repr(value) for value in unknown[:10])
        )
    return tuple(lookup[value.casefold()] for value in requested)


def _report_partial_coverage(
    context: _PlanningContext,
    *,
    modality: str,
    included: Sequence[str],
    reference_rois: Sequence[str],
) -> None:
    included_keys = {str(value).casefold() for value in included}
    missing = [
        str(value)
        for value in reference_rois
        if str(value).casefold() not in included_keys
    ]
    if not missing:
        return
    context.issue(
        "warning",
        "partial_roi_coverage",
        f"Included {len(included)} of {len(reference_rois)} reference ROIs; "
        f"{len(missing)} are unavailable. First missing: {missing[:10]}",
        modality=modality,
    )


def _coerce_integer_series(values: Any, *, context: str) -> Any:
    import numpy as np
    import pandas as pd

    numeric = pd.to_numeric(values, errors="raise")
    array = numeric.to_numpy()
    if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
        raise ValueError(f"{context} must contain finite integers.")
    if np.any(array <= 0):
        raise ValueError(f"{context} must contain positive integers.")
    return array.astype(np.int64)


def _normalise_value_name_maps(
    value_names: Any,
    *,
    rois: Sequence[str],
    value_key: str,
    name_key: str,
    mapping_roi_key: str | None,
) -> dict[str, dict[int, str]]:
    import pandas as pd

    roi_lookup = {roi.casefold(): roi for roi in rois}

    def one_mapping(mapping: Mapping[Any, Any], context: str) -> dict[int, str]:
        import math

        result: dict[int, str] = {}
        for raw_value, raw_name in mapping.items():
            if isinstance(raw_value, bool):
                raise ValueError(
                    f"{context} label value {raw_value!r} is not an integer code."
                )
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{context} label value {raw_value!r} is not integer.") from exc
            if not math.isfinite(numeric) or numeric < 0 or numeric != math.floor(numeric):
                raise ValueError(f"{context} label value {raw_value!r} is not non-negative.")
            value = int(numeric)
            name = str(raw_name).strip()
            if not name:
                raise ValueError(f"{context} label value {value} has an empty name.")
            if value in result:
                raise ValueError(f"{context} duplicates label value {value}.")
            result[value] = name
        return result

    if isinstance(value_names, (str, Path)):
        frame = pd.read_csv(value_names)
    elif isinstance(value_names, pd.DataFrame):
        frame = value_names.copy()
    else:
        frame = None

    if frame is not None:
        missing = {value_key, name_key}.difference(frame.columns)
        if missing:
            raise KeyError(f"Label mapping is missing columns: {sorted(missing)}")
        roi_column = mapping_roi_key
        if roi_column is None and "ROI" in frame.columns:
            roi_column = "ROI"
        result = {roi: {} for roi in rois}
        if roi_column is None:
            if frame[value_key].duplicated().any():
                duplicates = frame.loc[
                    frame[value_key].duplicated(keep=False), value_key
                ].tolist()
                raise ValueError(
                    f"Global label mapping contains duplicate values: {duplicates[:10]}"
                )
            mapping = one_mapping(
                dict(zip(frame[value_key], frame[name_key], strict=False)),
                "Global label mapping",
            )
            return {roi: dict(mapping) for roi in rois}
        if roi_column not in frame.columns:
            raise KeyError(f"Label mapping ROI column {roi_column!r} is missing.")
        for raw_roi, group in frame.groupby(roi_column, observed=True, sort=False):
            key = str(raw_roi).casefold()
            if key not in roi_lookup:
                raise ValueError(
                    f"Label mapping references ROI {raw_roi!r}, which is not expected."
                )
            roi = roi_lookup[key]
            if group[value_key].duplicated().any():
                duplicates = group.loc[
                    group[value_key].duplicated(keep=False), value_key
                ].tolist()
                raise ValueError(
                    f"ROI {roi!r} label mapping contains duplicate values: "
                    f"{duplicates[:10]}"
                )
            result[roi] = one_mapping(
                dict(zip(group[value_key], group[name_key], strict=False)),
                f"ROI {roi!r} label mapping",
            )
        missing_rois = [roi for roi, mapping in result.items() if not mapping]
        if missing_rois:
            raise ValueError(
                "Label mapping has no entries for ROI(s): " + ", ".join(missing_rois)
            )
        return result

    if not isinstance(value_names, Mapping):
        raise TypeError(
            "value_names must be a mapping, pandas DataFrame, or CSV path."
        )
    if not value_names:
        raise ValueError("value_names must not be empty.")
    nested = all(isinstance(value, Mapping) for value in value_names.values())
    if nested:
        result: dict[str, dict[int, str]] = {}
        for raw_roi, mapping in value_names.items():
            key = str(raw_roi).casefold()
            if key not in roi_lookup:
                raise ValueError(
                    f"Label mapping references ROI {raw_roi!r}, which is not expected."
                )
            roi = roi_lookup[key]
            result[roi] = one_mapping(mapping, f"ROI {roi!r} label mapping")
        missing = [roi for roi in rois if roi not in result]
        if missing:
            raise ValueError(
                "Label mapping has no entries for ROI(s): " + ", ".join(missing)
            )
        return result
    if any(isinstance(value, Mapping) for value in value_names.values()):
        raise TypeError("value_names cannot mix global and ROI-specific mappings.")
    global_mapping = one_mapping(value_names, "Global label mapping")
    return {roi: dict(global_mapping) for roi in rois}


@dataclass
class _ContextModality:
    name: str
    kind: str
    rois: tuple[str, ...]
    elements_by_roi: dict[str, str] = field(default_factory=dict)
    coordinate_systems_by_roi: dict[str, str] = field(default_factory=dict)
    shapes_by_roi: dict[str, tuple[int, int]] = field(default_factory=dict)
    channels: tuple[str, ...] = ()
    table_name: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class _PlanningContext:
    issues: list[ValidationIssue] = field(default_factory=list)
    modalities: dict[str, _ContextModality] = field(default_factory=dict)
    planned: dict[str, PlannedModality] = field(default_factory=dict)
    occupied: dict[str, str] = field(default_factory=dict)

    def issue(
        self,
        severity: str,
        code: str,
        message: str,
        *,
        modality: str | None = None,
        roi: str | None = None,
        path: Path | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                severity=severity,  # type: ignore[arg-type]
                code=code,
                message=message,
                modality=modality,
                roi=roi,
                path=path,
            )
        )

    def reserve(self, name: str, owner: str) -> None:
        folded = name.casefold()
        if folded in self.occupied:
            raise ValueError(
                f"Generated SpatialData name {name!r} for {owner!r} collides "
                f"with {self.occupied[folded]!r}."
            )
        self.occupied[folded] = owner

    def get(self, name: str) -> _ContextModality:
        matches = [
            value
            for key, value in self.modalities.items()
            if key.casefold() == str(name).casefold()
        ]
        if len(matches) != 1:
            available = ", ".join(self.modalities)
            raise KeyError(f"Referenced modality {name!r} was not found. Available: {available}")
        return matches[0]


def _seed_existing_context(existing: Any, context: _PlanningContext) -> None:
    if existing is None:
        return
    metadata = getattr(existing, "attrs", {}).get(SBT_METADATA_KEY)
    if not isinstance(metadata, Mapping) or int(metadata.get("schema_version", 0)) != 3:
        context.issue(
            "error",
            "existing_metadata",
            "add_modality requires a SpatialData object created with metadata schema 3.",
        )
        return
    for element_type in ("images", "labels", "points", "shapes", "tables"):
        for name in getattr(existing, element_type):
            context.occupied[str(name).casefold()] = f"existing {element_type}"
    raw_modalities = metadata.get("modalities", {})
    if not isinstance(raw_modalities, Mapping):
        context.issue("error", "existing_metadata", "Existing modalities metadata is invalid.")
        return
    for raw_name, raw in raw_modalities.items():
        if not isinstance(raw, Mapping):
            continue
        name = str(raw_name)
        shapes = {
            str(roi): tuple(int(value) for value in shape)
            for roi, shape in dict(raw.get("shapes_by_roi", {})).items()
        }
        item = _ContextModality(
            name=name,
            kind=str(raw.get("kind", "")),
            rois=tuple(str(value) for value in raw.get("rois", ())),
            elements_by_roi={
                str(key): str(value)
                for key, value in dict(raw.get("elements_by_roi", {})).items()
            },
            coordinate_systems_by_roi={
                str(key): str(value)
                for key, value in dict(
                    raw.get("coordinate_systems_by_roi", {})
                ).items()
            },
            shapes_by_roi=shapes,
            channels=tuple(str(value) for value in raw.get("channels", ())),
            table_name=(
                None if raw.get("table_name") is None else str(raw.get("table_name"))
            ),
            details={**dict(raw), "_existing_sdata": existing},
        )
        context.modalities[name] = item


def _validate_spec_names(spec: SpatialDataSpec, context: _PlanningContext) -> None:
    if not spec.modalities:
        context.issue("error", "empty_spec", "SpatialDataSpec contains no modalities.")
        return
    names: dict[str, str] = {}
    for modality in spec.modalities:
        name = str(getattr(modality, "name", "")).strip()
        if not name:
            context.issue("error", "empty_name", "Every modality must have a non-empty name.")
            continue
        try:
            safe = _safe_name(name)
        except ValueError as exc:
            context.issue("error", "invalid_name", str(exc), modality=name)
            continue
        folded = safe.casefold()
        if folded in names:
            context.issue(
                "error",
                "duplicate_modality",
                f"Names {names[folded]!r} and {name!r} normalize to the same key {safe!r}.",
                modality=name,
            )
        elif any(existing.casefold() == name.casefold() for existing in context.modalities):
            context.issue(
                "error",
                "duplicate_modality",
                f"Modality {name!r} already exists in the target SpatialData.",
                modality=name,
            )
        else:
            names[folded] = name
    if SBT_METADATA_KEY in spec.attrs:
        context.issue(
            "error",
            "reserved_attr",
            f"SpatialDataSpec.attrs cannot define reserved key {SBT_METADATA_KEY!r}.",
        )
    try:
        _normalise_chunks(spec.raster_chunks)
    except ValueError as exc:
        context.issue("error", "invalid_chunks", str(exc))


def _inspect_imc_table(source: IMCAnnData, context: _PlanningContext) -> _ContextModality | None:
    adata = None
    owned = False
    try:
        adata, owned = _read_adata(source.adata, backed=True)
        required = {
            source.roi_key,
            source.instance_key,
            source.x_key,
            source.y_key,
        }
        missing = required.difference(adata.obs.columns)
        if missing:
            raise KeyError(f"AnnData obs is missing columns: {sorted(missing)}")
        if not adata.obs_names.is_unique:
            raise ValueError("AnnData observation names must be unique.")
        channels = tuple(str(value) for value in adata.var_names)
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("AnnData var_names must be non-empty and unique.")
        rois = _ordered_strings(adata.obs[source.roi_key])
        if not rois:
            raise ValueError("AnnData contains no ROI values.")
        instances = _coerce_integer_series(
            adata.obs[source.instance_key],
            context=f"obs[{source.instance_key!r}]",
        )
        import numpy as np
        import pandas as pd

        x = pd.to_numeric(adata.obs[source.x_key], errors="raise").to_numpy()
        y = pd.to_numeric(adata.obs[source.y_key], errors="raise").to_numpy()
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Centroid columns must contain finite coordinates.")
        table_name_changes = _report_table_name_changes(
            adata,
            context=context,
            modality=source.name,
        )
        roi_values = adata.obs[source.roi_key].astype(str).to_numpy()
        for roi in rois:
            selected = roi_values == roi
            if len(np.unique(instances[selected])) != int(selected.sum()):
                raise ValueError(
                    f"obs[{source.instance_key!r}] contains duplicate IDs in ROI {roi!r}."
                )
        table_name = source.table_name or f"table_{_safe_name(source.name)}"
        details = {
            "panel_name": source.panel_name,
            "images": source.images,
            "masks": source.masks,
            "roi_key": source.roi_key,
            "instance_key": source.instance_key,
            "x_key": source.x_key,
            "y_key": source.y_key,
            "include_centroids": bool(source.include_centroids),
            "check_centroids_in_mask": bool(source.check_centroids_in_mask),
            "copy_adata": bool(source.copy_adata),
            "n_obs": int(adata.n_obs),
            "table_name_changes": table_name_changes,
            "_obs_names": tuple(str(value) for value in adata.obs_names),
            "_instances": instances,
            "_roi_values": roi_values,
            "_x": x,
            "_y": y,
        }
        return _ContextModality(
            name=source.name,
            kind="imc_anndata",
            rois=rois,
            channels=channels,
            table_name=table_name,
            details=details,
        )
    except Exception as exc:
        context.issue(
            "error",
            "imc_table_invalid",
            str(exc),
            modality=source.name,
            path=Path(source.adata) if isinstance(source.adata, (str, Path)) else None,
        )
        return None
    finally:
        if adata is not None:
            _close_adata(adata, owned)


def _plan_cell_masks(source: CellMasks, context: _PlanningContext) -> PlannedModality:
    import numpy as np
    import tifffile
    from spatialdata.transformations import Identity

    linked_tables = [
        item
        for item in context.modalities.values()
        if item.kind == "imc_anndata"
        and str(item.details.get("masks", "")).casefold() == source.name.casefold()
    ]
    inferred_rois = _ordered_strings(
        roi for table in linked_tables for roi in table.rois
    )
    rois = tuple(str(value) for value in source.rois) if source.rois else inferred_rois
    if not rois:
        raise ValueError(
            "CellMasks.rois is required when no IMCAnnData references this modality."
        )
    if len(set(value.casefold() for value in rois)) != len(rois):
        raise ValueError("CellMasks ROI names must be unique case-insensitively.")
    elements: list[RasterElementPlan] = []
    unannotated: dict[str, int] = {}
    instances_by_roi: dict[str, tuple[int, ...]] = {}
    for roi in rois:
        path = _match_roi_file(
            source.folder,
            roi,
            suffix=source.suffix,
            extensions=source.extensions,
        )
        shape, dtype = _tiff_metadata(path)
        if not np.issubdtype(dtype, np.integer):
            raise TypeError(f"Cell mask {path} must have integer dtype, found {dtype}.")
        mask = tifffile.imread(path)
        if np.any(mask < 0):
            raise ValueError(f"Cell mask {path} contains negative values.")
        present = np.unique(mask)
        present = present[present != source.background].astype(np.int64)
        instances_by_roi[roi] = tuple(int(value) for value in present)
        expected_union: set[int] = set()
        for table in linked_tables:
            roi_values = table.details["_roi_values"]
            expected = table.details["_instances"][roi_values == roi]
            missing = np.setdiff1d(expected, present)
            if len(missing):
                raise ValueError(
                    f"Table {table.name!r} has {len(missing)} instance(s) absent "
                    f"from mask {path.name} for ROI {roi!r}; first: "
                    + ", ".join(str(value) for value in missing[:10])
                )
            expected_union.update(int(value) for value in expected)
            if bool(table.details.get("check_centroids_in_mask")):
                selected = roi_values == roi
                xs = np.rint(table.details["_x"][selected]).astype(int)
                ys = np.rint(table.details["_y"][selected]).astype(int)
                expected_ids = table.details["_instances"][selected]
                in_bounds = (
                    (xs >= 0)
                    & (ys >= 0)
                    & (xs < shape[1])
                    & (ys < shape[0])
                )
                wrong = ~in_bounds
                valid_index = np.flatnonzero(in_bounds)
                wrong[valid_index] |= mask[ys[in_bounds], xs[in_bounds]] != expected_ids[in_bounds]
                if np.any(wrong):
                    raise ValueError(
                        f"Table {table.name!r} has {int(wrong.sum())} centroid(s) "
                        f"outside their expected mask instance in ROI {roi!r}."
                    )
        unannotated[roi] = len(set(map(int, present)).difference(expected_union))
        element = f"labels_{_safe_name(source.name)}_{_safe_name(roi)}"
        coordinate = (
            f"{_safe_name(source.coordinate_system_prefix)}_"
            f"{_safe_name(source.name)}_{_safe_name(roi)}"
        )
        context.reserve(element, source.name)
        elements.append(
            RasterElementPlan(
                roi=roi,
                element_name=element,
                coordinate_system=coordinate,
                paths=(path,),
                shape=shape,
                dtype=str(dtype),
                transformations={coordinate: Identity()},
            )
        )
    details = {
        "background": int(source.background),
        "unannotated_instances_by_roi": unannotated,
        "instances_by_roi": instances_by_roi,
    }
    return PlannedModality(
        name=source.name,
        kind="cell_masks",
        source=source,
        elements=tuple(elements),
        rois=rois,
        details=details,
    )


def _reference_for_images(source: IMCImages, context: _PlanningContext) -> str | None:
    linked = [
        table
        for table in context.modalities.values()
        if table.kind == "imc_anndata"
        and str(table.details.get("images", "")).casefold() == source.name.casefold()
    ]
    masks = _ordered_strings(table.details["masks"] for table in linked)
    if source.reference is not None:
        if masks and any(
            value.casefold() != source.reference.casefold() for value in masks
        ):
            raise ValueError(
                f"IMCImages reference {source.reference!r} conflicts with linked "
                f"IMCAnnData mask reference(s) {masks}."
            )
        return source.reference
    if len(masks) == 1:
        return masks[0]
    if len(masks) > 1:
        raise ValueError(
            f"IMCImages {source.name!r} is linked to tables using multiple mask "
            f"modalities {masks}; set reference explicitly or separate the images."
        )
    return None


def _plan_imc_images(source: IMCImages, context: _PlanningContext) -> PlannedModality:
    from spatialdata.transformations import Identity

    linked = [
        table
        for table in context.modalities.values()
        if table.kind == "imc_anndata"
        and str(table.details.get("images", "")).casefold() == source.name.casefold()
    ]
    linked_channels = {table.channels for table in linked}
    if len(linked_channels) > 1:
        raise ValueError(
            f"Linked IMCAnnData tables disagree on channels for {source.name!r}."
        )
    table_channels = next(iter(linked_channels), ())
    if source.channels is None:
        channels = table_channels
    else:
        channels = tuple(str(value) for value in source.channels)
        if table_channels and set(channels) != set(table_channels):
            missing = sorted(set(table_channels).difference(channels))
            extra = sorted(set(channels).difference(table_channels))
            raise ValueError(
                f"IMCImages channels do not match linked AnnData var_names; "
                f"missing={missing}, extra={extra}."
            )
        if table_channels:
            channels = table_channels
    if not channels or len(set(value.casefold() for value in channels)) != len(channels):
        raise ValueError("IMCImages channels must be non-empty and unique.")

    reference_name = _reference_for_images(source, context)
    reference = context.get(reference_name) if reference_name else None
    linked_rois = _ordered_strings(roi for table in linked for roi in table.rois)
    root = Path(source.folder)
    if not root.is_dir():
        raise FileNotFoundError(f"IMCImages folder not found: {root}")
    if source.rois is not None:
        rois = tuple(str(value) for value in source.rois)
    elif linked_rois:
        rois = linked_rois
    elif reference is not None:
        rois = reference.rois
    else:
        raise ValueError(
            "Standalone IMCImages requires rois or a reference modality."
        )
    if reference is not None:
        rois = _canonical_reference_rois(
            rois,
            reference_rois=reference.rois,
            modality_kind="IMCImages",
        )
    if linked_rois and {
        value.casefold() for value in rois
    } != {value.casefold() for value in linked_rois}:
        raise ValueError(
            "IMCImages linked to IMCAnnData must cover every quantified table ROI."
        )
    if (
        source.allow_partial
        and source.rois is None
        and not linked_rois
        and reference is not None
    ):
        directories = [path for path in root.iterdir() if path.is_dir()]
        available = {path.name.casefold() for path in directories}
        rois = tuple(roi for roi in reference.rois if roi.casefold() in available)
        if not rois:
            raise FileNotFoundError(
                f"No ROI directories in {root} match reference {reference.name!r}."
            )
        _report_partial_coverage(
            context,
            modality=source.name,
            included=rois,
            reference_rois=reference.rois,
        )
    elements: list[RasterElementPlan] = []
    extra_files: dict[str, tuple[str, ...]] = {}
    for roi in rois:
        roi_folder = _unique_casefold(
            (path for path in root.iterdir() if path.is_dir()),
            roi,
            kind="ROI directory",
        )
        paths: list[Path] = []
        modes: list[str] = []
        shape: tuple[int, int] | None = None
        dtype: Any = None
        for channel in channels:
            path, match_mode = _match_channel_file(
                roi_folder,
                channel,
                extensions=source.extensions,
                mode=source.match_mode,
            )
            current_shape, current_dtype = _tiff_metadata(path)
            if shape is None:
                shape, dtype = current_shape, current_dtype
            elif current_shape != shape:
                raise ValueError(
                    f"Channel {channel!r} in ROI {roi!r} has shape {current_shape}; "
                    f"other channels have shape {shape}."
                )
            paths.append(path)
            modes.append(match_mode)
        matched = {path.resolve(strict=False) for path in paths}
        allowed = set(_normalise_extensions(source.extensions))
        extras = tuple(
            path.name
            for path in roi_folder.iterdir()
            if path.is_file()
            and path.suffix.casefold() in allowed
            and path.resolve(strict=False) not in matched
        )
        if extras and not source.allow_extra_files:
            raise ValueError(
                f"ROI {roi!r} contains unselected image files: {list(extras)}"
            )
        if extras:
            extra_files[roi] = extras
            context.issue(
                "warning",
                "extra_image_files",
                f"{len(extras)} image file(s) are not selected by the channel list.",
                modality=source.name,
                roi=roi,
                path=roi_folder,
            )
        assert shape is not None
        if reference is not None:
            if roi not in reference.shapes_by_roi:
                raise ValueError(
                    f"Reference {reference.name!r} has no raster shape for ROI {roi!r}."
                )
            explicit = None if source.transformations is None else source.transformations.get(roi)
            if explicit is None and shape != reference.shapes_by_roi[roi]:
                raise ValueError(
                    f"IMCImages ROI {roi!r} has shape {shape}, but identity-aligned "
                    f"reference {reference.name!r} has shape {reference.shapes_by_roi[roi]}; "
                    "provide an explicit transformation."
                )
            coordinate = reference.coordinate_systems_by_roi[roi]
            transform = explicit if explicit is not None else Identity()
        else:
            coordinate = f"roi_{_safe_name(source.name)}_{_safe_name(roi)}"
            transform = Identity()
        element = f"image_{_safe_name(source.name)}_{_safe_name(roi)}"
        context.reserve(element, source.name)
        elements.append(
            RasterElementPlan(
                roi=roi,
                element_name=element,
                coordinate_system=coordinate,
                paths=tuple(paths),
                shape=shape,
                dtype=str(dtype),
                channels=channels,
                channel_match_modes=tuple(modes),
                transformations={coordinate: transform},
            )
        )
    return PlannedModality(
        name=source.name,
        kind="imc_images",
        source=source,
        elements=tuple(elements),
        rois=rois,
        channels=channels,
        details={
            "panel_name": source.panel_name,
            "reference": reference_name,
            "extra_files_by_roi": extra_files,
        },
    )


def _plan_histology(source: HistologyImages, context: _PlanningContext) -> PlannedModality:
    from spatialdata.transformations import Identity

    reference = context.get(source.reference)
    requested = (
        tuple(str(value) for value in source.rois)
        if source.rois is not None
        else tuple(reference.rois)
    )
    requested = _canonical_reference_rois(
        requested,
        reference_rois=reference.rois,
        modality_kind="HistologyImages",
    )
    resolved: list[tuple[str, Path]] = []
    for roi in requested:
        try:
            path = _match_roi_file(
                source.folder,
                roi,
                suffix=source.suffix,
                extensions=source.extensions,
            )
        except FileNotFoundError:
            if source.allow_partial and source.rois is None:
                continue
            raise
        resolved.append((roi, path))
    if not resolved:
        raise FileNotFoundError(
            f"No histology files match reference {reference.name!r}."
        )
    rois = tuple(roi for roi, _path in resolved)
    if source.allow_partial and source.rois is None:
        _report_partial_coverage(
            context,
            modality=source.name,
            included=rois,
            reference_rois=reference.rois,
        )
    elements: list[RasterElementPlan] = []
    for roi, path in resolved:
        shape, n_channels, dtype = _histology_metadata(path)
        output_channels = 3 if source.drop_alpha and n_channels == 4 else n_channels
        explicit = None if source.transformations is None else source.transformations.get(roi)
        if explicit is None and shape != reference.shapes_by_roi[roi]:
            raise ValueError(
                f"Histology ROI {roi!r} has shape {shape}, but reference "
                f"{reference.name!r} has shape {reference.shapes_by_roi[roi]}; "
                "provide an explicit transformation."
            )
        coordinate = reference.coordinate_systems_by_roi[roi]
        element = f"image_{_safe_name(source.name)}_{_safe_name(roi)}"
        context.reserve(element, source.name)
        elements.append(
            RasterElementPlan(
                roi=roi,
                element_name=element,
                coordinate_system=coordinate,
                paths=(path,),
                shape=shape,
                dtype=str(dtype),
                channels=("r", "g", "b", "a")[:output_channels],
                transformations={
                    coordinate: explicit if explicit is not None else Identity()
                },
            )
        )
    return PlannedModality(
        name=source.name,
        kind="histology_images",
        source=source,
        elements=tuple(elements),
        rois=rois,
        channels=elements[0].channels if elements else (),
        details={
            "reference": source.reference,
            "drop_alpha": bool(source.drop_alpha),
        },
    )


def _plan_region_labels(source: RegionLabels, context: _PlanningContext) -> PlannedModality:
    import numpy as np
    import tifffile
    from spatialdata.transformations import Identity

    reference = context.get(source.reference)
    requested = (
        tuple(str(value) for value in source.rois)
        if source.rois is not None
        else tuple(reference.rois)
    )
    requested = _canonical_reference_rois(
        requested,
        reference_rois=reference.rois,
        modality_kind="RegionLabels",
    )
    resolved: list[tuple[str, Path]] = []
    for roi in requested:
        try:
            path = _match_roi_file(
                source.folder,
                roi,
                suffix=source.suffix,
                extensions=source.extensions,
            )
        except FileNotFoundError:
            if source.allow_partial and source.rois is None:
                continue
            raise
        resolved.append((roi, path))
    if not resolved:
        raise FileNotFoundError(
            f"No region-label files match reference {reference.name!r}."
        )
    rois = tuple(roi for roi, _path in resolved)
    if source.allow_partial and source.rois is None:
        _report_partial_coverage(
            context,
            modality=source.name,
            included=rois,
            reference_rois=reference.rois,
        )
    mappings = _normalise_value_name_maps(
        source.value_names,
        rois=rois,
        value_key=source.value_key,
        name_key=source.name_key,
        mapping_roi_key=source.mapping_roi_key,
    )
    elements: list[RasterElementPlan] = []
    present_by_roi: dict[str, tuple[int, ...]] = {}
    names_by_roi: dict[str, dict[int, str]] = {}
    for roi, path in resolved:
        shape, dtype = _tiff_metadata(path)
        if not np.issubdtype(dtype, np.integer):
            raise TypeError(f"Region label raster {path} must have integer dtype.")
        values = np.unique(tifffile.imread(path))
        if np.any(values < 0):
            raise ValueError(f"Region label raster {path} contains negative values.")
        present = tuple(int(value) for value in values if int(value) != 0)
        missing = [value for value in present if value not in mappings[roi]]
        if missing:
            raise ValueError(
                f"RegionLabels {source.name!r}, ROI {roi!r} has unnamed values: "
                + ", ".join(str(value) for value in missing[:10])
            )
        if not present:
            context.issue(
                "warning",
                "empty_region_labels",
                "The raster contains no positive region values.",
                modality=source.name,
                roi=roi,
                path=path,
            )
        unused = sorted(set(mappings[roi]).difference(present, {0}))
        if unused:
            context.issue(
                "warning",
                "unused_region_names",
                f"Mapped values are absent from this raster: {unused[:10]}",
                modality=source.name,
                roi=roi,
                path=path,
            )
        explicit = None if source.transformations is None else source.transformations.get(roi)
        if explicit is None and shape != reference.shapes_by_roi[roi]:
            raise ValueError(
                f"RegionLabels ROI {roi!r} has shape {shape}, but reference "
                f"{reference.name!r} has shape {reference.shapes_by_roi[roi]}; "
                "provide an explicit transformation."
            )
        coordinate = reference.coordinate_systems_by_roi[roi]
        element = f"labels_{_safe_name(source.name)}_{_safe_name(roi)}"
        context.reserve(element, source.name)
        elements.append(
            RasterElementPlan(
                roi=roi,
                element_name=element,
                coordinate_system=coordinate,
                paths=(path,),
                shape=shape,
                dtype=str(dtype),
                transformations={
                    coordinate: explicit if explicit is not None else Identity()
                },
            )
        )
        present_by_roi[roi] = present
        names_by_roi[roi] = {
            value: mappings[roi][value] for value in present
        }
    if not any(present_by_roi.values()):
        raise ValueError(
            f"RegionLabels {source.name!r} contains no positive values in any ROI."
        )
    table_name = source.table_name or f"table_{_safe_name(source.name)}"
    context.reserve(table_name, source.name)
    return PlannedModality(
        name=source.name,
        kind="region_labels",
        source=source,
        elements=tuple(elements),
        table_name=table_name,
        rois=rois,
        details={
            "reference": source.reference,
            "value_key": source.value_key,
            "name_key": source.name_key,
            "value_names_by_roi": names_by_roi,
            "values_by_roi": present_by_roi,
        },
    )


def _finalise_imc_table(
    source: IMCAnnData,
    table_context: _ContextModality,
    context: _PlanningContext,
) -> PlannedModality:
    images = context.get(source.images)
    masks = context.get(source.masks)
    if images.kind != "imc_images":
        raise TypeError(f"IMCAnnData.images must reference IMCImages, found {images.kind!r}.")
    if masks.kind != "cell_masks":
        raise TypeError(f"IMCAnnData.masks must reference CellMasks, found {masks.kind!r}.")
    if str(images.details.get("panel_name", "")) != str(source.panel_name):
        raise ValueError(
            f"IMCAnnData panel_name {source.panel_name!r} does not match "
            f"IMCImages panel_name {images.details.get('panel_name')!r}."
        )
    if set(table_context.rois) != set(images.rois) or set(table_context.rois) != set(masks.rois):
        raise ValueError(
            "IMCAnnData, IMCImages, and CellMasks must have the same ROI set."
        )
    if tuple(table_context.channels) != tuple(images.channels):
        raise ValueError(
            "IMCAnnData var_names order does not match the planned IMCImages channel order."
        )
    assert table_context.table_name is not None
    context.reserve(table_context.table_name, source.name)
    point_elements: dict[str, str] = {}
    if source.include_centroids:
        for roi in table_context.rois:
            name = f"points_{_safe_name(source.name)}_{_safe_name(roi)}"
            context.reserve(name, source.name)
            point_elements[roi] = name
    details = dict(table_context.details)
    details.update(
        {
            "point_elements_by_roi": point_elements,
            "coordinate_systems_by_roi": dict(masks.coordinate_systems_by_roi),
            "mask_elements_by_roi": dict(masks.elements_by_roi),
        }
    )
    return PlannedModality(
        name=source.name,
        kind="imc_anndata",
        source=source,
        table_name=table_context.table_name,
        rois=table_context.rois,
        channels=table_context.channels,
        details=details,
    )


def _plan_maxfuse(source: MaxFuseSCRNASeq, context: _PlanningContext) -> PlannedModality:
    target = context.get(source.imc_table)
    if target.kind != "imc_anndata":
        raise TypeError(
            f"MaxFuseSCRNASeq.imc_table must reference IMCAnnData, found {target.kind!r}."
        )
    target_obs: tuple[str, ...]
    if "_obs_names" in target.details:
        target_obs = tuple(target.details["_obs_names"])
    else:
        if target.table_name is None:
            raise ValueError(f"Linked IMC table {target.name!r} has no SpatialData table.")
        existing = context_existing_table = target.details.get("_existing_sdata")
        if existing is None:
            raise ValueError(
                f"Existing IMC modality {target.name!r} lacks observation-index metadata."
            )
        target_obs = tuple(str(value) for value in context_existing_table.tables[target.table_name].obs_names)

    adata = None
    owned = False
    try:
        adata, owned = _read_adata(source.adata, backed=True)
        if not adata.obs_names.is_unique:
            raise ValueError("MaxFuse AnnData observation names must be unique.")
        if not adata.var_names.is_unique:
            raise ValueError("MaxFuse AnnData var_names must be unique.")
        source_obs = tuple(str(value) for value in adata.obs_names)
        extra = sorted(set(source_obs).difference(target_obs))
        if extra:
            raise ValueError(
                f"{len(extra)} MaxFuse observation(s) are absent from linked IMC "
                f"table {source.imc_table!r}; first: {extra[:10]}"
            )
        if not source_obs:
            raise ValueError("MaxFuse AnnData contains no matched cells.")
        table_name_changes = _report_table_name_changes(
            adata,
            context=context,
            modality=source.name,
        )
        matched_fraction = len(source_obs) / len(target_obs) if target_obs else 0.0
        table_name = source.table_name or f"table_{_safe_name(source.name)}"
        context.reserve(table_name, source.name)
        return PlannedModality(
            name=source.name,
            kind="maxfuse_scrnaseq",
            source=source,
            table_name=table_name,
            rois=target.rois,
            channels=tuple(str(value) for value in adata.var_names),
            details={
                "imc_table": source.imc_table,
                "linked_table_name": target.table_name,
                "matched_cells": len(source_obs),
                "linked_imc_cells": len(target_obs),
                "matched_fraction": matched_fraction,
                "table_name_changes": table_name_changes,
                "_obs_names": source_obs,
            },
        )
    finally:
        if adata is not None:
            _close_adata(adata, owned)


def _register_planned(item: PlannedModality, context: _PlanningContext) -> None:
    context.planned[item.name] = item
    elements_by_roi = {element.roi: element.element_name for element in item.elements}
    coords_by_roi = {
        element.roi: element.coordinate_system for element in item.elements
    }
    shapes_by_roi = {element.roi: element.shape for element in item.elements}
    details = dict(item.details)
    context.modalities[item.name] = _ContextModality(
        name=item.name,
        kind=item.kind,
        rois=item.rois,
        elements_by_roi=elements_by_roi,
        coordinate_systems_by_roi=coords_by_roi,
        shapes_by_roi=shapes_by_roi,
        channels=item.channels,
        table_name=item.table_name,
        details=details,
    )


def _call_planner(
    source: ModalitySpec,
    context: _PlanningContext,
    function: Any,
    *args: Any,
) -> None:
    try:
        item = function(source, *args, context) if args else function(source, context)
        _register_planned(item, context)
    except Exception as exc:
        context.issue(
            "error",
            f"{type(source).__name__.casefold()}_invalid",
            str(exc),
            modality=source.name,
        )


def plan_spatialdata(
    spec: SpatialDataSpec,
    *,
    existing: Any | None = None,
) -> SpatialDataPlan:
    """Validate a multimodal source graph without constructing SpatialData.

    Planning resolves every file and relationship, reads raster metadata, and
    performs full cell-mask and categorical-label integrity checks.  It never
    changes source AnnData objects or the optional existing SpatialData.
    """

    if not isinstance(spec, SpatialDataSpec):
        raise TypeError("plan_spatialdata expects a SpatialDataSpec.")
    context = _PlanningContext()
    _seed_existing_context(existing, context)
    _validate_spec_names(spec, context)

    imc_sources = [
        source for source in spec.modalities if isinstance(source, IMCAnnData)
    ]
    inspected_imc: dict[str, _ContextModality] = {}
    for source in imc_sources:
        inspected = _inspect_imc_table(source, context)
        if inspected is not None:
            inspected_imc[source.name] = inspected
            context.modalities[source.name] = inspected

    for source in spec.modalities:
        if isinstance(source, CellMasks):
            _call_planner(source, context, _plan_cell_masks)

    for source in spec.modalities:
        if isinstance(source, IMCImages):
            _call_planner(source, context, _plan_imc_images)

    for source in imc_sources:
        inspected = inspected_imc.get(source.name)
        if inspected is None:
            continue
        try:
            item = _finalise_imc_table(source, inspected, context)
            _register_planned(item, context)
        except Exception as exc:
            context.issue(
                "error",
                "imc_anndata_relationship",
                str(exc),
                modality=source.name,
            )

    for source in spec.modalities:
        if isinstance(source, HistologyImages):
            _call_planner(source, context, _plan_histology)
        elif isinstance(source, RegionLabels):
            _call_planner(source, context, _plan_region_labels)

    for source in spec.modalities:
        if isinstance(source, MaxFuseSCRNASeq):
            _call_planner(source, context, _plan_maxfuse)

    ordered = tuple(
        context.planned[source.name]
        for source in spec.modalities
        if source.name in context.planned
    )
    report = ValidationReport(tuple(context.issues))
    logging.info(
        "Planned SpatialData with %d modality/modalities, %d warning(s), and %d error(s).",
        len(ordered),
        len(report.warnings),
        len(report.errors),
    )
    return SpatialDataPlan(
        spec=spec,
        modalities=ordered,
        report=report,
        existing=existing,
    )


def _read_tiff(path: str) -> Any:
    import tifffile

    return tifffile.imread(path)


def _read_histology(path: str, drop_alpha: bool) -> Any:
    import imageio.v3 as iio
    import numpy as np

    values = iio.imread(path)
    if values.ndim == 3 and values.shape[-1] not in {3, 4} and values.shape[0] in {
        3,
        4,
    }:
        values = np.moveaxis(values, 0, -1)
    if drop_alpha and values.shape[-1] == 4:
        values = values[..., :3]
    return values


def _lazy_tiff(path: Path, shape: tuple[int, int], dtype: Any, chunks: tuple[int, int]) -> Any:
    import dask.array as da
    from dask import delayed
    import numpy as np

    delayed_array = delayed(_read_tiff)(str(path))
    array = da.from_delayed(delayed_array, shape=shape, dtype=np.dtype(dtype))
    return array.rechunk(
        tuple(min(chunk, size) for chunk, size in zip(chunks, shape, strict=True))
    )


def _lazy_histology(
    element: RasterElementPlan,
    *,
    drop_alpha: bool,
    chunks: tuple[int, int],
) -> Any:
    import dask.array as da
    from dask import delayed
    import numpy as np

    shape = (*element.shape, len(element.channels))
    delayed_array = delayed(_read_histology)(str(element.paths[0]), drop_alpha)
    array = da.from_delayed(delayed_array, shape=shape, dtype=np.dtype(element.dtype))
    array = da.moveaxis(array, -1, 0)
    return array.rechunk(
        (
            len(element.channels),
            min(chunks[0], element.shape[0]),
            min(chunks[1], element.shape[1]),
        )
    )


@dataclass
class _ElementBundle:
    images: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    points: dict[str, Any] = field(default_factory=dict)
    shapes: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _BuildReference:
    name: str
    kind: str
    rois: tuple[str, ...]
    elements_by_roi: Mapping[str, str]
    coordinate_systems_by_roi: Mapping[str, str]
    shapes_by_roi: Mapping[str, tuple[int, int]]
    channels: tuple[str, ...]
    table_name: str | None
    planned: PlannedModality | None = None


def _build_reference(plan: SpatialDataPlan, name: str) -> _BuildReference:
    for item in plan.modalities:
        if item.name.casefold() != str(name).casefold():
            continue
        if item.kind == "imc_anndata":
            elements = dict(item.details.get("point_elements_by_roi", {}))
            coordinates = dict(item.details.get("coordinate_systems_by_roi", {}))
            shapes: dict[str, tuple[int, int]] = {}
        else:
            elements = {
                element.roi: element.element_name for element in item.elements
            }
            coordinates = {
                element.roi: element.coordinate_system for element in item.elements
            }
            shapes = {element.roi: element.shape for element in item.elements}
        return _BuildReference(
            name=item.name,
            kind=item.kind,
            rois=item.rois,
            elements_by_roi=elements,
            coordinate_systems_by_roi=coordinates,
            shapes_by_roi=shapes,
            channels=item.channels,
            table_name=item.table_name,
            planned=item,
        )
    if plan.existing is None:
        raise KeyError(f"Referenced modality {name!r} is not part of the plan.")
    metadata = plan.existing.attrs.get(SBT_METADATA_KEY, {})
    raw_modalities = metadata.get("modalities", {})
    matches = [
        (str(key), value)
        for key, value in raw_modalities.items()
        if str(key).casefold() == str(name).casefold()
    ]
    if len(matches) != 1 or not isinstance(matches[0][1], Mapping):
        raise KeyError(f"Referenced modality {name!r} is not present in SpatialData.")
    resolved_name, raw = matches[0]
    return _BuildReference(
        name=resolved_name,
        kind=str(raw.get("kind", "")),
        rois=tuple(str(value) for value in raw.get("rois", ())),
        elements_by_roi={
            str(key): str(value)
            for key, value in dict(raw.get("elements_by_roi", {})).items()
        },
        coordinate_systems_by_roi={
            str(key): str(value)
            for key, value in dict(raw.get("coordinate_systems_by_roi", {})).items()
        },
        shapes_by_roi={
            str(key): tuple(int(value) for value in shape)
            for key, shape in dict(raw.get("shapes_by_roi", {})).items()
        },
        channels=tuple(str(value) for value in raw.get("channels", ())),
        table_name=None if raw.get("table_name") is None else str(raw["table_name"]),
    )


def _element_by_roi(item: PlannedModality) -> dict[str, RasterElementPlan]:
    return {element.roi: element for element in item.elements}


def _build_imc_table(
    item: PlannedModality,
    plan: SpatialDataPlan,
    bundle: _ElementBundle,
) -> None:
    import numpy as np
    import pandas as pd
    from spatialdata.models import PointsModel, TableModel

    source = item.source
    assert isinstance(source, IMCAnnData)
    adata = _prepare_table_adata(source.adata, copy_adata=source.copy_adata)
    masks = _build_reference(plan, source.masks)
    mask_elements = dict(masks.elements_by_roi)
    roi_values = adata.obs[source.roi_key].astype(str)
    regions = roi_values.map(mask_elements)
    if regions.isna().any():
        raise RuntimeError(f"Could not map all rows in IMC table {source.name!r} to masks.")
    adata.obs[TABLE_REGION_KEY] = pd.Categorical(
        regions,
        categories=[mask_elements[roi] for roi in masks.rois],
    )
    adata.obs[TABLE_INSTANCE_KEY] = _coerce_integer_series(
        adata.obs[source.instance_key],
        context=f"obs[{source.instance_key!r}]",
    )
    adata.obsm["spatial"] = adata.obs[[source.x_key, source.y_key]].to_numpy(
        dtype=np.float64
    )
    for roi, point_name in item.details["point_elements_by_roi"].items():
        selected = roi_values == roi
        data = pd.DataFrame(
            {
                "x": adata.obs.loc[selected, source.x_key].to_numpy(dtype=float),
                "y": adata.obs.loc[selected, source.y_key].to_numpy(dtype=float),
                CENTROID_OBS_KEY: adata.obs_names[selected].astype(str),
                source.roi_key: roi,
                source.instance_key: adata.obs.loc[
                    selected, source.instance_key
                ].to_numpy(),
            },
            index=adata.obs_names[selected].astype(str),
        )
        if masks.planned is not None:
            transformations = dict(
                _element_by_roi(masks.planned)[roi].transformations
            )
        else:
            from spatialdata.transformations import get_transformation

            transformations = dict(
                get_transformation(
                    plan.existing.labels[mask_elements[roi]], get_all=True
                )
            )
        bundle.points[point_name] = PointsModel.parse(
            data,
            coordinates={"x": "x", "y": "y"},
            transformations=transformations,
        )
    _sanitize_table_for_spatialdata(adata, modality=source.name)
    assert item.table_name is not None
    bundle.tables[item.table_name] = TableModel.parse(
        adata,
        region=[mask_elements[roi] for roi in masks.rois],
        region_key=TABLE_REGION_KEY,
        instance_key=TABLE_INSTANCE_KEY,
        overwrite_metadata=True,
    )


def _build_region_table(item: PlannedModality, bundle: _ElementBundle) -> None:
    import anndata as ad
    import numpy as np
    import pandas as pd
    from spatialdata.models import TableModel

    source = item.source
    assert isinstance(source, RegionLabels)
    elements = _element_by_roi(item)
    records: list[dict[str, Any]] = []
    obs_names: list[str] = []
    for roi in item.rois:
        for value, name in item.details["value_names_by_roi"][roi].items():
            records.append(
                {
                    "ROI": roi,
                    TABLE_REGION_KEY: elements[roi].element_name,
                    TABLE_INSTANCE_KEY: int(value),
                    source.value_key: int(value),
                    source.name_key: str(name),
                }
            )
            obs_names.append(f"{_safe_name(source.name)}:{_safe_name(roi)}:{value}")
    obs = pd.DataFrame.from_records(records, index=obs_names)
    obs[TABLE_REGION_KEY] = pd.Categorical(
        obs[TABLE_REGION_KEY],
        categories=[element.element_name for element in item.elements],
    )
    obs[TABLE_INSTANCE_KEY] = obs[TABLE_INSTANCE_KEY].to_numpy(dtype=np.int64)
    obs[source.value_key] = obs[source.value_key].to_numpy(dtype=np.int64)
    obs[source.name_key] = pd.Categorical(obs[source.name_key].astype(str))
    annotation = ad.AnnData(
        X=np.empty((len(obs), 0), dtype=np.float32),
        obs=obs,
    )
    _sanitize_table_for_spatialdata(annotation, modality=source.name)
    assert item.table_name is not None
    bundle.tables[item.table_name] = TableModel.parse(
        annotation,
        region=[element.element_name for element in item.elements],
        region_key=TABLE_REGION_KEY,
        instance_key=TABLE_INSTANCE_KEY,
        overwrite_metadata=True,
    )


def _build_maxfuse(
    item: PlannedModality,
    plan: SpatialDataPlan,
    bundle: _ElementBundle,
) -> None:
    import pandas as pd
    from spatialdata.models import TableModel

    source = item.source
    assert isinstance(source, MaxFuseSCRNASeq)
    adata = _prepare_table_adata(source.adata, copy_adata=source.copy_adata)
    target_plan = _build_reference(plan, source.imc_table)
    assert target_plan.table_name is not None
    if target_plan.table_name in bundle.tables:
        target = bundle.tables[target_plan.table_name]
    elif plan.existing is not None and target_plan.table_name in plan.existing.tables:
        target = plan.existing.tables[target_plan.table_name]
    else:
        raise KeyError(f"Linked IMC table {target_plan.table_name!r} was not built.")
    linked = target.obs.loc[
        adata.obs_names, [TABLE_REGION_KEY, TABLE_INSTANCE_KEY]
    ]
    adata.obs[TABLE_REGION_KEY] = pd.Categorical(
        linked[TABLE_REGION_KEY].astype(str).to_numpy(),
        categories=list(target.obs[TABLE_REGION_KEY].cat.categories),
    )
    adata.obs[TABLE_INSTANCE_KEY] = linked[TABLE_INSTANCE_KEY].to_numpy()
    _sanitize_table_for_spatialdata(adata, modality=source.name)
    assert item.table_name is not None
    regions = list(adata.obs[TABLE_REGION_KEY].cat.categories)
    bundle.tables[item.table_name] = TableModel.parse(
        adata,
        region=regions,
        region_key=TABLE_REGION_KEY,
        instance_key=TABLE_INSTANCE_KEY,
        overwrite_metadata=True,
    )
    table_metadata = dict(bundle.tables[item.table_name].uns.get(SBT_METADATA_KEY, {}))
    table_metadata.update(
        {
            "schema_version": 1,
            "kind": "maxfuse_scrnaseq",
            "linked_imc_modality": source.imc_table,
            "linked_imc_table": target_plan.table_name,
            "matched_fraction": item.details["matched_fraction"],
        }
    )
    bundle.tables[item.table_name].uns[SBT_METADATA_KEY] = table_metadata


def _build_bundle(plan: SpatialDataPlan) -> _ElementBundle:
    import dask.array as da
    from spatialdata.models import Image2DModel, Labels2DModel

    plan.raise_for_errors()
    chunks = _normalise_chunks(plan.spec.raster_chunks)
    scale_factors = plan.spec.scale_factors
    bundle = _ElementBundle()
    for item in plan.modalities:
        if item.kind == "cell_masks":
            for element in item.elements:
                raster = _lazy_tiff(
                    element.paths[0], element.shape, element.dtype, chunks
                )
                bundle.labels[element.element_name] = Labels2DModel.parse(
                    raster,
                    dims=("y", "x"),
                    transformations=dict(element.transformations),
                    scale_factors=scale_factors,
                )
        elif item.kind == "region_labels":
            for element in item.elements:
                raster = _lazy_tiff(
                    element.paths[0], element.shape, element.dtype, chunks
                )
                bundle.labels[element.element_name] = Labels2DModel.parse(
                    raster,
                    dims=("y", "x"),
                    transformations=dict(element.transformations),
                    scale_factors=scale_factors,
                )
            _build_region_table(item, bundle)
        elif item.kind == "imc_images":
            for element in item.elements:
                channels = [
                    _lazy_tiff(path, element.shape, element.dtype, chunks)
                    for path in element.paths
                ]
                bundle.images[element.element_name] = Image2DModel.parse(
                    da.stack(channels, axis=0),
                    dims=("c", "y", "x"),
                    c_coords=list(element.channels),
                    transformations=dict(element.transformations),
                    scale_factors=scale_factors,
                )
        elif item.kind == "histology_images":
            source = item.source
            assert isinstance(source, HistologyImages)
            for element in item.elements:
                bundle.images[element.element_name] = Image2DModel.parse(
                    _lazy_histology(
                        element,
                        drop_alpha=source.drop_alpha,
                        chunks=chunks,
                    ),
                    dims=("c", "y", "x"),
                    c_coords=list(element.channels),
                    transformations=dict(element.transformations),
                    scale_factors=scale_factors,
                )
        elif item.kind == "imc_anndata":
            _build_imc_table(item, plan, bundle)
        elif item.kind == "maxfuse_scrnaseq":
            _build_maxfuse(item, plan, bundle)
    return bundle


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _public_details(item: PlannedModality) -> dict[str, Any]:
    construction_only = {
        "instances_by_roi",
        "value_names_by_roi",
        "values_by_roi",
        "point_elements_by_roi",
        "coordinate_systems_by_roi",
        "mask_elements_by_roi",
    }
    return {
        str(key): _json_ready(value)
        for key, value in item.details.items()
        if not str(key).startswith("_") and key not in construction_only
    }


def _metadata_from_plan(plan: SpatialDataPlan) -> dict[str, Any]:
    metadata: dict[str, Any]
    if plan.existing is None:
        metadata = {
            "schema_version": SBT_SCHEMA_VERSION,
            "modalities": {},
            "rois": {},
            "primary": {},
            "label_layers": {},
        }
    else:
        metadata = copy.deepcopy(
            plan.existing.attrs.get(
                SBT_METADATA_KEY,
                {
                    "schema_version": SBT_SCHEMA_VERSION,
                    "modalities": {},
                    "rois": {},
                    "primary": {},
                    "label_layers": {},
                },
            )
        )
    modalities = metadata.setdefault("modalities", {})
    for item in plan.modalities:
        elements = {element.roi: element.element_name for element in item.elements}
        coordinates = {
            element.roi: element.coordinate_system for element in item.elements
        }
        shapes = {element.roi: list(element.shape) for element in item.elements}
        raw = {
            "kind": item.kind,
            "rois": list(item.rois),
            "elements_by_roi": elements,
            "coordinate_systems_by_roi": coordinates,
            "shapes_by_roi": shapes,
            "channels": (
                []
                if item.kind == "maxfuse_scrnaseq"
                else list(item.channels)
            ),
            "table_name": item.table_name,
            **_public_details(item),
        }
        if item.kind == "maxfuse_scrnaseq":
            raw["features"] = len(item.channels)
        if item.kind == "imc_anndata":
            raw["elements_by_roi"] = dict(item.details["point_elements_by_roi"])
            raw["coordinate_systems_by_roi"] = dict(
                item.details["coordinate_systems_by_roi"]
            )
            masks = _build_reference(plan, str(item.details["masks"]))
            raw["shapes_by_roi"] = {
                roi: list(shape) for roi, shape in masks.shapes_by_roi.items()
            }
        modalities[item.name] = raw

    primary = metadata.setdefault("primary", {})
    all_modalities = list(modalities.items())
    if "imc_table" not in primary:
        first_imc = next(
            (name for name, value in all_modalities if value.get("kind") == "imc_anndata"),
            None,
        )
        if first_imc is not None:
            primary["imc_table"] = first_imc
            primary["imc_images"] = modalities[first_imc].get("images")
            primary["cell_masks"] = modalities[first_imc].get("masks")

    rois = metadata.setdefault("rois", {})
    for name, raw in modalities.items():
        for roi in raw.get("rois", ()):
            selected = rois.setdefault(
                roi,
                {
                    "modalities": {},
                    "images": {},
                    "label_elements": {},
                    "points": {},
                    "coordinate_systems": {},
                },
            )
            element = raw.get("elements_by_roi", {}).get(roi)
            kind = raw.get("kind")
            selected["modalities"][name] = element or raw.get("table_name")
            coordinate = raw.get("coordinate_systems_by_roi", {}).get(roi)
            if coordinate is not None:
                selected["coordinate_systems"][name] = coordinate
            if kind in {"imc_images", "histology_images"} and element:
                selected["images"][name] = element
            elif kind in {"cell_masks", "region_labels"} and element:
                selected["label_elements"][name] = element
            elif kind == "imc_anndata" and element:
                selected["points"][name] = element

    label_layers = metadata.setdefault("label_layers", {})
    primary_masks = primary.get("cell_masks")
    for name, raw in modalities.items():
        kind = raw.get("kind")
        if kind == "cell_masks":
            key = "cells" if name == primary_masks else name
            linked_tables = [
                candidate.get("table_name")
                for candidate in modalities.values()
                if candidate.get("kind") == "imc_anndata"
                and candidate.get("masks") == name
            ]
            label_layers[key] = {
                "display_name": name,
                "kind": "instances",
                "modality": name,
                "annotation_table": next(
                    (value for value in linked_tables if value is not None), None
                ),
                "region_key": TABLE_REGION_KEY,
                "instance_key": TABLE_INSTANCE_KEY,
                "elements_by_roi": dict(raw.get("elements_by_roi", {})),
            }
        elif kind == "region_labels":
            label_layers[name] = {
                "display_name": name,
                "kind": "categorical",
                "modality": name,
                "annotation_table": raw.get("table_name"),
                "roi_key": "ROI",
                "region_key": TABLE_REGION_KEY,
                "instance_key": TABLE_INSTANCE_KEY,
                "value_key": raw.get("value_key"),
                "name_key": raw.get("name_key"),
                "elements_by_roi": dict(raw.get("elements_by_roi", {})),
            }

    primary_imc = primary.get("imc_table")
    if primary_imc in modalities:
        table_meta = modalities[primary_imc]
        metadata["table_name"] = table_meta.get("table_name")
        metadata["table_region_key"] = TABLE_REGION_KEY
        metadata["table_instance_key"] = TABLE_INSTANCE_KEY
        metadata["roi_key"] = table_meta.get("roi_key")
        metadata["source_instance_key"] = table_meta.get("instance_key")
    for roi, selected in rois.items():
        primary_images = primary.get("imc_images")
        primary_labels = primary.get("cell_masks")
        selected["image"] = selected.get("images", {}).get(primary_images)
        selected["labels_primary"] = selected.get("label_elements", {}).get(
            primary_labels
        )
        # Retained as the primary helper-facing field, not as legacy metadata.
        selected["labels"] = selected["labels_primary"]
        selected["coordinate_system"] = selected.get("coordinate_systems", {}).get(
            primary_labels
        )
    metadata["schema_version"] = SBT_SCHEMA_VERSION
    return metadata


def _assemble(
    plan: SpatialDataPlan,
    bundle: _ElementBundle,
    *,
    existing: Any | None,
) -> Any:
    from spatialdata import SpatialData

    attrs = (
        dict(plan.spec.attrs)
        if existing is None
        else {**copy.deepcopy(existing.attrs), **dict(plan.spec.attrs)}
    )
    attrs[SBT_METADATA_KEY] = _metadata_from_plan(plan)
    if existing is None:
        images = bundle.images
        labels = bundle.labels
        points = bundle.points
        shapes = bundle.shapes
        tables = bundle.tables
    else:
        images = {**dict(existing.images), **bundle.images}
        labels = {**dict(existing.labels), **bundle.labels}
        points = {**dict(existing.points), **bundle.points}
        shapes = {**dict(existing.shapes), **bundle.shapes}
        tables = {**dict(existing.tables), **bundle.tables}
    candidate = SpatialData(
        images=images,
        labels=labels,
        points=points,
        shapes=shapes,
        tables=tables,
        attrs=attrs,
    )
    for table in candidate.tables.values():
        candidate.validate_table_in_spatialdata(table)
    return candidate


def create_spatialdata(
    value: SpatialDataSpec | SpatialDataPlan,
    *,
    plan: SpatialDataPlan | None = None,
) -> Any:
    """Create a lazy SpatialData object from a declarative specification.

    This function has one construction contract: pass a
    :class:`SpatialDataSpec`, optionally with its precomputed plan, or pass the
    plan itself.  Use :func:`plan_spatialdata` when a side-effect-free integrity
    report is needed before construction.
    """

    if isinstance(value, SpatialDataPlan):
        if plan is not None:
            raise TypeError("Do not pass plan= when value is already a SpatialDataPlan.")
        selected_plan = value
    elif isinstance(value, SpatialDataSpec):
        selected_plan = plan if plan is not None else plan_spatialdata(value)
        if selected_plan.spec is not value:
            raise ValueError(
                "The supplied plan was created from a different SpatialDataSpec object."
            )
        if selected_plan.existing is not None:
            raise ValueError("A plan against an existing SpatialData must use add_modality().")
    else:
        raise TypeError("create_spatialdata expects SpatialDataSpec or SpatialDataPlan.")
    selected_plan.raise_for_errors()
    bundle = _build_bundle(selected_plan)
    sdata = _assemble(selected_plan, bundle, existing=None)
    logging.info(
        "Created SpatialData from %d planned modality/modalities.",
        len(selected_plan.modalities),
    )
    return sdata


def add_modality(
    sdata: Any,
    value: ModalitySpec | Sequence[ModalitySpec] | SpatialDataSpec | SpatialDataPlan,
    *,
    inplace: bool = False,
) -> Any:
    """Validate and add one or several modalities to SpatialData in memory.

    The complete addition is built and validated in a candidate container
    before ``sdata`` is changed.  Disk persistence is intentionally separate:
    write the returned object to a new Zarr store after inspecting the plan.
    """

    if isinstance(value, SpatialDataPlan):
        plan = value
        if plan.existing is not sdata:
            raise ValueError("The SpatialDataPlan was not created against this object.")
    else:
        if isinstance(value, SpatialDataSpec):
            spec = value
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            spec = SpatialDataSpec(tuple(value))
        else:
            spec = SpatialDataSpec((value,))  # type: ignore[arg-type]
        plan = plan_spatialdata(spec, existing=sdata)
    plan.raise_for_errors()
    bundle = _build_bundle(plan)
    candidate = _assemble(plan, bundle, existing=sdata)
    if not inplace:
        return candidate
    for name, element in bundle.images.items():
        sdata.images[name] = element
    for name, element in bundle.labels.items():
        sdata.labels[name] = element
    for name, element in bundle.points.items():
        sdata.points[name] = element
    for name, element in bundle.shapes.items():
        sdata.shapes[name] = element
    for name, element in bundle.tables.items():
        sdata.tables[name] = element
    sdata.attrs.clear()
    sdata.attrs.update(copy.deepcopy(candidate.attrs))
    return sdata
