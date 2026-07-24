"""Create, inspect, and plot SpatialData objects for IMC projects.

The converter in this module bridges the common SpatialBiologyToolkit layout
(an AnnData cell table, one channel TIFF per marker and ROI, and one labelled
mask per ROI) to the scverse SpatialData data model.  Raster data are assembled
with Dask, so creating the in-memory object does not read every TIFF into RAM.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence


TIFF_SUFFIXES = frozenset({".tif", ".tiff"})
SBT_METADATA_KEY = "spatial_biology_toolkit"
DEFAULT_TABLE_NAME = "table"
DEFAULT_REGION_KEY = "_sbt_region"
DEFAULT_INSTANCE_KEY = "_sbt_instance_id"
DEFAULT_LABEL_REGION_KEY = "_sbt_label_region"
DEFAULT_LABEL_INSTANCE_KEY = "_sbt_label_instance_id"
DEFAULT_LABEL_VALUE_KEY = "label_value"
DEFAULT_LABEL_NAME_KEY = "label_name"


@dataclass(frozen=True)
class MarkerImageMatch:
    """The TIFF selected for one marker and how it was matched."""

    marker: str
    path: Path
    mode: Literal["exact", "substring"]


@dataclass(frozen=True)
class AdditionalLabelsSpec:
    """Describe one logical layer of ROI-aligned integer label TIFFs.

    Parameters
    ----------
    name
        Stable, human-readable layer name, for example ``"tissue_region"``.
        It is converted to a safe key for generated SpatialData element names
        while the original value is retained as the display name.
    folder
        Folder containing one TIFF per ROI.  Each filename must have the exact
        case-insensitive stem ``{ROI}{suffix}``.
    suffix
        Required filename suffix between the ROI name and ``.tif[f]``, for
        example ``"_regions"``.
    value_names
        Pixel-value-to-name annotations.  Accepted forms are a global
        ``{integer: name}`` mapping, a nested ``{ROI: {integer: name}}``
        mapping, a pandas DataFrame, or a CSV path.  DataFrame/CSV inputs use
        ``value_key`` and ``name_key`` columns and may optionally contain an
        ROI column.
    table_name
        Optional annotation-table name.  The default is
        ``"{safe_name}_annotations"``.
    value_key, name_key
        Column names used for the numeric pixel value and its semantic name.
    mapping_roi_key
        Optional ROI column in a DataFrame/CSV mapping.  When omitted, the
        converter uses the source ``roi_key`` column if it is present;
        otherwise the mapping is treated as global.

    Notes
    -----
    The TIFF remains an integer SpatialData Labels element.  Semantic names
    are stored in a linked annotation table so plotting and querying tools can
    resolve them without replacing every raster pixel with a string.
    """

    name: str
    folder: str | Path
    suffix: str
    value_names: Any
    table_name: str | None = None
    value_key: str = DEFAULT_LABEL_VALUE_KEY
    name_key: str = DEFAULT_LABEL_NAME_KEY
    mapping_roi_key: str | None = None


@dataclass(frozen=True)
class ROIConversionPlan:
    """Validated source files and SpatialData names for one ROI."""

    roi: str
    image_element: str | None
    labels_element: str
    coordinate_system: str
    mask_path: Path
    shape: tuple[int, int]
    mask_dtype: str
    marker_images: tuple[MarkerImageMatch, ...]
    table_instances: int
    unannotated_mask_instances: int


@dataclass(frozen=True)
class AdditionalLabelsROIPlan:
    """Validated source TIFF and annotations for one layer in one ROI."""

    roi: str
    labels_element: str
    path: Path
    shape: tuple[int, int]
    dtype: str
    value_names: tuple[tuple[int, str], ...]
    unused_mapping_values: tuple[int, ...] = ()


@dataclass(frozen=True)
class AdditionalLabelsPlan:
    """Validated files and annotation metadata for one logical label layer."""

    name: str
    key: str
    suffix: str
    table_name: str
    roi_key: str
    region_key: str
    instance_key: str
    value_key: str
    name_key: str
    rois: tuple[AdditionalLabelsROIPlan, ...]


@dataclass(frozen=True)
class SpatialDataConversionPlan:
    """A validated, side-effect-free plan for an IMC-to-SpatialData conversion."""

    roi_key: str
    instance_key: str
    markers: tuple[str, ...]
    rois: tuple[ROIConversionPlan, ...]
    additional_labels: tuple[AdditionalLabelsPlan, ...] = ()

    @property
    def n_rois(self) -> int:
        return len(self.rois)

    @property
    def n_image_files(self) -> int:
        return sum(len(roi.marker_images) for roi in self.rois)

    @property
    def n_additional_label_files(self) -> int:
        return sum(len(layer.rois) for layer in self.additional_labels)


def _tiff_files(folder: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.casefold() in TIFF_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )


def _format_paths(paths: Sequence[Path]) -> str:
    return ", ".join(path.name for path in paths)


def match_marker_image(roi_image_folder: str | Path, marker: str) -> MarkerImageMatch:
    """Find the unique TIFF corresponding to ``marker`` within an ROI folder.

    Matching is case-insensitive.  An exact filename stem is preferred.  If no
    exact match exists, the marker must occur as a bounded token in the stem
    (for example, ``CD3`` matches ``152Sm_CD3`` but not ``CD31``).  A final raw
    substring match supports unusual legacy names.  Every matching tier is
    strict: multiple candidates raise an error rather than choosing silently.
    """

    folder = Path(roi_image_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"ROI image folder not found: {folder}")
    marker = str(marker).strip()
    if not marker:
        raise ValueError("Marker names must not be empty.")

    files = _tiff_files(folder)
    exact = [path for path in files if path.stem.casefold() == marker.casefold()]
    if len(exact) == 1:
        return MarkerImageMatch(marker=marker, path=exact[0], mode="exact")
    if len(exact) > 1:
        raise ValueError(
            f"Multiple exact TIFF matches for marker {marker!r} in {folder}: "
            f"{_format_paths(exact)}"
        )

    bounded_pattern = re.compile(
        rf"(?<![0-9A-Za-z]){re.escape(marker)}(?![0-9A-Za-z])",
        flags=re.IGNORECASE,
    )
    bounded = [path for path in files if bounded_pattern.search(path.stem)]
    if len(bounded) == 1:
        return MarkerImageMatch(marker=marker, path=bounded[0], mode="substring")
    if len(bounded) > 1:
        raise ValueError(
            f"Multiple TIFFs contain the bounded marker {marker!r} in {folder}: "
            f"{_format_paths(bounded)}"
        )

    substring = [path for path in files if marker.casefold() in path.stem.casefold()]
    if len(substring) == 1:
        return MarkerImageMatch(marker=marker, path=substring[0], mode="substring")
    if len(substring) > 1:
        raise ValueError(
            f"Multiple TIFFs contain marker {marker!r} in {folder}: "
            f"{_format_paths(substring)}"
        )
    raise FileNotFoundError(
        f"No TIFF matches marker {marker!r} in ROI image folder {folder}."
    )


def _safe_name(value: str) -> str:
    normalized = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    token = re.sub(r"[^0-9A-Za-z_]+", "_", normalized).strip("_")
    token = re.sub(r"_+", "_", token)
    if not token:
        raise ValueError(f"Name {value!r} cannot be converted to a safe name.")
    if token[0].isdigit():
        token = f"roi_{token}"
    return token


def _normalise_additional_labels_specs(
    values: (
        AdditionalLabelsSpec
        | Mapping[str, Any]
        | Sequence[AdditionalLabelsSpec | Mapping[str, Any]]
        | None
    ),
) -> tuple[AdditionalLabelsSpec, ...]:
    if values is None:
        return ()
    if isinstance(values, AdditionalLabelsSpec) or isinstance(values, Mapping):
        candidates: Sequence[AdditionalLabelsSpec | Mapping[str, Any]] = (values,)
    else:
        candidates = values

    specs: list[AdditionalLabelsSpec] = []
    for index, candidate in enumerate(candidates):
        if isinstance(candidate, AdditionalLabelsSpec):
            spec = candidate
        elif isinstance(candidate, Mapping):
            try:
                spec = AdditionalLabelsSpec(**dict(candidate))
            except TypeError as exc:
                raise TypeError(
                    f"Invalid additional_labels specification at index {index}: {exc}"
                ) from exc
        else:
            raise TypeError(
                "additional_labels entries must be AdditionalLabelsSpec objects "
                f"or mappings, found {type(candidate).__name__} at index {index}."
            )

        name = str(spec.name).strip()
        if not name:
            raise ValueError(
                f"Additional label layer at index {index} has an empty name."
            )
        suffix = str(spec.suffix)
        if not suffix:
            raise ValueError(
                f"Additional label layer {name!r} must define a non-empty suffix."
            )
        specs.append(
            AdditionalLabelsSpec(
                name=name,
                folder=Path(spec.folder),
                suffix=suffix,
                value_names=spec.value_names,
                table_name=spec.table_name,
                value_key=str(spec.value_key).strip(),
                name_key=str(spec.name_key).strip(),
                mapping_roi_key=(
                    None
                    if spec.mapping_roi_key is None
                    else str(spec.mapping_roi_key).strip()
                ),
            )
        )
    return tuple(specs)


def _coerce_positive_label_value(value: Any, *, context: str) -> int:
    import numpy as np
    import pandas as pd

    if isinstance(value, (bool, np.bool_)) or pd.isna(value):
        raise ValueError(f"{context} must be a non-negative integer, found {value!r}.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} must be a non-negative integer, found {value!r}."
        ) from exc
    if not np.isfinite(numeric) or numeric != np.floor(numeric) or numeric < 0:
        raise ValueError(f"{context} must be a non-negative integer, found {value!r}.")
    return int(numeric)


def _coerce_label_name(value: Any, *, context: str) -> str:
    import pandas as pd

    if pd.isna(value):
        raise ValueError(f"{context} must not be missing.")
    name = str(value).strip()
    if not name:
        raise ValueError(f"{context} must not be empty.")
    return name


def _resolve_mapping_roi(value: Any, roi_names: Sequence[str], *, context: str) -> str:
    candidate = str(value)
    if candidate in roi_names:
        return candidate
    matches = [roi for roi in roi_names if roi.casefold() == candidate.casefold()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"{context} references ROI {candidate!r}, which is not present in adata."
    )


def _add_value_name(
    target: dict[int, str],
    value: Any,
    name: Any,
    *,
    context: str,
) -> None:
    label_value = _coerce_positive_label_value(value, context=f"{context} value")
    label_name = _coerce_label_name(name, context=f"{context} name")
    if label_value in target:
        raise ValueError(
            f"{context} contains duplicate mapping for pixel value {label_value}."
        )
    target[label_value] = label_name


def _load_value_name_maps(
    spec: AdditionalLabelsSpec,
    *,
    roi_names: Sequence[str],
    source_roi_key: str,
) -> dict[str, dict[int, str]]:
    """Normalize global or ROI-specific label-name inputs."""

    import pandas as pd

    source = spec.value_names
    if isinstance(source, (str, Path)):
        mapping_path = Path(source)
        if not mapping_path.is_file():
            raise FileNotFoundError(
                f"Mapping CSV for additional label layer {spec.name!r} was not found: "
                f"{mapping_path}"
            )
        source = pd.read_csv(mapping_path)

    per_roi: dict[str, dict[int, str]] = {roi: {} for roi in roi_names}
    if isinstance(source, pd.DataFrame):
        missing_columns = [
            key
            for key in (spec.value_key, spec.name_key)
            if not key or key not in source.columns
        ]
        if missing_columns:
            raise KeyError(
                f"Mapping table for additional label layer {spec.name!r} is missing "
                f"column(s): {missing_columns}"
            )
        mapping_roi_key = spec.mapping_roi_key
        if mapping_roi_key is None and source_roi_key in source.columns:
            mapping_roi_key = source_roi_key
        if mapping_roi_key is not None and mapping_roi_key not in source.columns:
            raise KeyError(
                f"Mapping ROI column {mapping_roi_key!r} is missing for additional "
                f"label layer {spec.name!r}."
            )

        if mapping_roi_key is None:
            global_map: dict[int, str] = {}
            for row_index, row in source.iterrows():
                _add_value_name(
                    global_map,
                    row[spec.value_key],
                    row[spec.name_key],
                    context=f"Layer {spec.name!r} mapping row {row_index!r}",
                )
            per_roi = {roi: dict(global_map) for roi in roi_names}
        else:
            for row_index, row in source.iterrows():
                roi = _resolve_mapping_roi(
                    row[mapping_roi_key],
                    roi_names,
                    context=f"Layer {spec.name!r} mapping row {row_index!r}",
                )
                _add_value_name(
                    per_roi[roi],
                    row[spec.value_key],
                    row[spec.name_key],
                    context=f"Layer {spec.name!r} mapping row {row_index!r}",
                )
    elif isinstance(source, Mapping):
        if not source:
            raise ValueError(
                f"Pixel-value mapping for additional label layer {spec.name!r} is empty."
            )
        nested_flags = [isinstance(value, Mapping) for value in source.values()]
        if any(nested_flags) and not all(nested_flags):
            raise TypeError(
                f"Mapping for additional label layer {spec.name!r} mixes global "
                "values with per-ROI mappings."
            )
        if all(nested_flags):
            seen_rois: set[str] = set()
            for roi_value, mapping in source.items():
                roi = _resolve_mapping_roi(
                    roi_value,
                    roi_names,
                    context=f"Layer {spec.name!r} mapping",
                )
                if roi in seen_rois:
                    raise ValueError(
                        f"Layer {spec.name!r} contains duplicate mappings for ROI {roi!r}."
                    )
                seen_rois.add(roi)
                for value, name in mapping.items():
                    _add_value_name(
                        per_roi[roi],
                        value,
                        name,
                        context=f"Layer {spec.name!r}, ROI {roi!r}",
                    )
            missing_rois = [roi for roi in roi_names if roi not in seen_rois]
            if missing_rois:
                raise ValueError(
                    f"Layer {spec.name!r} has no pixel-value mapping for ROI(s): "
                    + ", ".join(repr(roi) for roi in missing_rois)
                )
        else:
            global_map = {}
            for value, name in source.items():
                _add_value_name(
                    global_map,
                    value,
                    name,
                    context=f"Layer {spec.name!r}",
                )
            per_roi = {roi: dict(global_map) for roi in roi_names}
    else:
        raise TypeError(
            f"value_names for additional label layer {spec.name!r} must be a "
            "mapping, pandas DataFrame, or CSV path."
        )

    return per_roi


def _match_additional_labels_tiff(
    folder: Path,
    *,
    roi: str,
    suffix: str,
    layer_name: str,
) -> Path:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Folder for additional label layer {layer_name!r} was not found: {folder}"
        )
    expected_stem = f"{roi}{suffix}"
    matches = [
        path
        for path in _tiff_files(folder)
        if path.stem.casefold() == expected_stem.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple TIFFs match ROI {roi!r}, suffix {suffix!r}, and layer "
            f"{layer_name!r} in {folder}: {_format_paths(matches)}"
        )
    raise FileNotFoundError(
        f"No TIFF with stem {expected_stem!r} was found for additional label "
        f"layer {layer_name!r} in {folder}."
    )


def _unique_casefold_match(
    folder: Path,
    name: str,
    *,
    kind: Literal["directory", "tiff"],
) -> Path:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if kind == "directory":
        candidates = [path for path in folder.iterdir() if path.is_dir()]
        exact_path = folder / name
        if exact_path.is_dir():
            return exact_path
        matches = [
            path for path in candidates if path.name.casefold() == name.casefold()
        ]
    else:
        candidates = _tiff_files(folder)
        matches = [
            path for path in candidates if path.stem.casefold() == name.casefold()
        ]
    if len(matches) == 1:
        return matches[0]
    label = "ROI image directory" if kind == "directory" else "mask TIFF"
    if len(matches) > 1:
        raise ValueError(
            f"Multiple {label} matches for ROI {name!r} in {folder}: "
            f"{_format_paths(matches)}"
        )
    raise FileNotFoundError(f"No {label} found for ROI {name!r} in {folder}.")


def _tiff_metadata(path: Path) -> tuple[tuple[int, int], Any]:
    import numpy as np
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = tuple(int(value) for value in series.shape)
        dtype = np.dtype(series.dtype)
    if len(shape) != 2:
        raise ValueError(f"Expected a 2D TIFF at {path}, found shape {shape}.")
    return (shape[0], shape[1]), dtype


def _validate_obs(
    adata: Any,
    roi_key: str,
    instance_key: str,
) -> tuple[Any, Any, Any]:
    import numpy as np
    import pandas as pd

    missing_columns = [
        key for key in (roi_key, instance_key) if key not in adata.obs.columns
    ]
    if missing_columns:
        raise KeyError(
            "Required AnnData observation columns are missing: "
            + ", ".join(repr(key) for key in missing_columns)
        )
    if adata.obs[roi_key].isna().any():
        raise ValueError(f"adata.obs[{roi_key!r}] contains missing ROI identifiers.")
    if adata.obs[instance_key].isna().any():
        raise ValueError(f"adata.obs[{instance_key!r}] contains missing mask labels.")

    numeric_instances = pd.to_numeric(adata.obs[instance_key], errors="coerce")
    if numeric_instances.isna().any():
        raise ValueError(
            f"adata.obs[{instance_key!r}] must contain integer mask labels."
        )
    instance_values = numeric_instances.to_numpy()
    if not np.all(np.equal(instance_values, np.floor(instance_values))):
        raise ValueError(
            f"adata.obs[{instance_key!r}] must contain integer mask labels."
        )
    instance_values = instance_values.astype(np.int64, copy=False)
    if np.any(instance_values <= 0):
        raise ValueError(
            f"adata.obs[{instance_key!r}] must contain positive labels; 0 is background."
        )

    roi_codes, roi_categories = pd.factorize(adata.obs[roi_key], sort=False)
    if np.any(roi_codes < 0):
        raise ValueError(f"adata.obs[{roi_key!r}] contains missing ROI identifiers.")
    roi_names = np.asarray([str(value) for value in roi_categories], dtype=object)
    if len(set(roi_names)) != len(roi_names):
        raise ValueError(
            f"String conversion makes values in adata.obs[{roi_key!r}] non-unique."
        )
    return roi_codes, roi_names, instance_values


def plan_imc_spatialdata_conversion(
    adata: Any,
    images_folder: str | Path | None,
    masks_folder: str | Path,
    *,
    roi_key: str = "ROI",
    instance_key: str = "ObjectNumber",
    validate_instance_ids: bool = True,
    additional_labels: (
        AdditionalLabelsSpec
        | Mapping[str, Any]
        | Sequence[AdditionalLabelsSpec | Mapping[str, Any]]
        | None
    ) = None,
) -> SpatialDataConversionPlan:
    """Validate inputs and return a side-effect-free conversion plan.

    Extra positive labels in a mask are permitted because they commonly
    represent cells filtered out of the AnnData table.  A table instance that
    is absent from its mask is always an error when ``validate_instance_ids``
    is enabled.

    ``additional_labels`` accepts one or several independent logical layers.
    Every layer is validated across all ROIs, including exact suffix-based
    filename matching, integer dtype, shape agreement, non-negative values,
    and complete pixel-value-to-name annotations.
    """

    import numpy as np
    import tifffile

    masks_root = Path(masks_folder)
    if not masks_root.is_dir():
        raise FileNotFoundError(f"Masks folder not found: {masks_root}")
    images_root = Path(images_folder) if images_folder is not None else None
    if images_root is not None and not images_root.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_root}")

    markers = tuple(str(marker) for marker in adata.var_names)
    if not markers:
        raise ValueError("AnnData has no variables/markers to associate with images.")
    if len(set(markers)) != len(markers):
        raise ValueError(
            "adata.var_names must be unique before SpatialData conversion."
        )

    roi_codes, roi_names, instance_values = _validate_obs(
        adata, roi_key=roi_key, instance_key=instance_key
    )
    additional_specs = _normalise_additional_labels_specs(additional_labels)
    used_names: set[str] = set()
    roi_plans: list[ROIConversionPlan] = []

    for roi_index, roi in enumerate(roi_names):
        safe_roi = _safe_name(roi)
        image_element = f"image_{safe_roi}" if images_root is not None else None
        labels_element = f"labels_{safe_roi}"
        coordinate_system = f"roi_{safe_roi}"
        proposed_names = {labels_element, coordinate_system}
        if image_element is not None:
            proposed_names.add(image_element)
        collision = used_names.intersection(proposed_names)
        if collision:
            raise ValueError(
                f"ROI {roi!r} produces duplicate SpatialData names: {sorted(collision)}"
            )
        used_names.update(proposed_names)

        mask_path = _unique_casefold_match(masks_root, roi, kind="tiff")
        mask_shape, mask_dtype = _tiff_metadata(mask_path)
        if not np.issubdtype(mask_dtype, np.integer):
            raise TypeError(
                f"Mask {mask_path} has dtype {mask_dtype}; SpatialData labels require integers."
            )

        marker_matches: tuple[MarkerImageMatch, ...] = ()
        if images_root is not None:
            roi_image_folder = _unique_casefold_match(
                images_root, roi, kind="directory"
            )
            matches: list[MarkerImageMatch] = []
            for marker in markers:
                match = match_marker_image(roi_image_folder, marker)
                image_shape, _image_dtype = _tiff_metadata(match.path)
                if image_shape != mask_shape:
                    raise ValueError(
                        f"Image {match.path} has shape {image_shape}, but mask "
                        f"{mask_path} has shape {mask_shape}."
                    )
                matches.append(match)
            marker_matches = tuple(matches)

        expected_instances = instance_values[roi_codes == roi_index]
        if len(expected_instances) != len(np.unique(expected_instances)):
            raise ValueError(
                f"adata.obs[{instance_key!r}] contains duplicate labels within ROI {roi!r}."
            )
        unannotated = 0
        if validate_instance_ids:
            mask = tifffile.imread(mask_path)
            present = np.unique(mask)
            present = present[present != 0]
            missing = np.setdiff1d(expected_instances, present)
            if len(missing):
                preview = ", ".join(str(value) for value in missing[:10])
                raise ValueError(
                    f"ROI {roi!r} has {len(missing)} ObjectNumber value(s) absent "
                    f"from mask {mask_path.name}; first values: {preview}."
                )
            unannotated = int(len(np.setdiff1d(present, expected_instances)))

        roi_plans.append(
            ROIConversionPlan(
                roi=roi,
                image_element=image_element,
                labels_element=labels_element,
                coordinate_system=coordinate_system,
                mask_path=mask_path,
                shape=mask_shape,
                mask_dtype=str(mask_dtype),
                marker_images=marker_matches,
                table_instances=int(len(expected_instances)),
                unannotated_mask_instances=unannotated,
            )
        )

    additional_plans: list[AdditionalLabelsPlan] = []
    used_layer_keys: set[str] = set()
    used_table_names: set[str] = set()
    for spec in additional_specs:
        layer_key = _safe_name(spec.name)
        if layer_key.casefold() == "cells":
            raise ValueError(
                f"Additional label layer name {spec.name!r} is reserved for the "
                "primary cell mask."
            )
        folded_key = layer_key.casefold()
        if folded_key in used_layer_keys:
            raise ValueError(
                f"Additional label layer names must be unique after normalization; "
                f"{spec.name!r} produces duplicate key {layer_key!r}."
            )
        used_layer_keys.add(folded_key)

        requested_table_name = (
            spec.table_name
            if spec.table_name is not None
            else f"{layer_key}_annotations"
        )
        annotation_table_name = _safe_name(str(requested_table_name))
        folded_table_name = annotation_table_name.casefold()
        if folded_table_name in used_table_names:
            raise ValueError(
                f"Additional label annotation table names must be unique; "
                f"{annotation_table_name!r} is duplicated."
            )
        used_table_names.add(folded_table_name)

        annotation_columns = (
            roi_key,
            DEFAULT_LABEL_REGION_KEY,
            DEFAULT_LABEL_INSTANCE_KEY,
            spec.value_key,
            spec.name_key,
        )
        if any(not value for value in annotation_columns):
            raise ValueError(
                f"Annotation column names for additional label layer "
                f"{spec.name!r} must not be empty."
            )
        if len(set(annotation_columns)) != len(annotation_columns):
            raise ValueError(
                f"Annotation columns for additional label layer {spec.name!r} "
                f"must be distinct: {annotation_columns}"
            )

        value_names_by_roi = _load_value_name_maps(
            spec,
            roi_names=tuple(str(value) for value in roi_names),
            source_roi_key=roi_key,
        )
        folder = Path(spec.folder)
        layer_roi_plans: list[AdditionalLabelsROIPlan] = []
        for roi_plan in roi_plans:
            labels_element = f"labels_{layer_key}_{_safe_name(roi_plan.roi)}"
            if labels_element in used_names:
                raise ValueError(
                    f"Additional label layer {spec.name!r} and ROI "
                    f"{roi_plan.roi!r} produce duplicate SpatialData name "
                    f"{labels_element!r}."
                )
            used_names.add(labels_element)

            path = _match_additional_labels_tiff(
                folder,
                roi=roi_plan.roi,
                suffix=spec.suffix,
                layer_name=spec.name,
            )
            shape, dtype = _tiff_metadata(path)
            if not np.issubdtype(dtype, np.integer):
                raise TypeError(
                    f"Additional label TIFF {path} has dtype {dtype}; "
                    "SpatialData labels require integers."
                )
            if shape != roi_plan.shape:
                raise ValueError(
                    f"Additional label TIFF {path} has shape {shape}, but the "
                    f"primary mask for ROI {roi_plan.roi!r} has shape "
                    f"{roi_plan.shape}."
                )

            values = np.unique(tifffile.imread(path))
            if np.any(values < 0):
                raise ValueError(
                    f"Additional label TIFF {path} contains negative values; "
                    "0 must represent background and positive integers must "
                    "represent named labels."
                )
            present_values = tuple(
                int(value) for value in values.tolist() if int(value) != 0
            )
            roi_value_names = value_names_by_roi[roi_plan.roi]
            missing_values = [
                value for value in present_values if value not in roi_value_names
            ]
            if missing_values:
                preview = ", ".join(str(value) for value in missing_values[:10])
                raise ValueError(
                    f"Additional label layer {spec.name!r}, ROI {roi_plan.roi!r} "
                    f"has {len(missing_values)} positive pixel value(s) without "
                    f"a name; first values: {preview}."
                )
            unused_values = tuple(
                sorted(
                    value
                    for value in roi_value_names
                    if value != 0 and value not in present_values
                )
            )
            if unused_values:
                logging.warning(
                    "Additional label layer %r, ROI %r has %d mapped value(s) "
                    "not present in %s.",
                    spec.name,
                    roi_plan.roi,
                    len(unused_values),
                    path.name,
                )
            layer_roi_plans.append(
                AdditionalLabelsROIPlan(
                    roi=roi_plan.roi,
                    labels_element=labels_element,
                    path=path,
                    shape=shape,
                    dtype=str(dtype),
                    value_names=tuple(
                        (value, roi_value_names[value]) for value in present_values
                    ),
                    unused_mapping_values=unused_values,
                )
            )

        if not any(roi.value_names for roi in layer_roi_plans):
            raise ValueError(
                f"Additional label layer {spec.name!r} contains no positive "
                "pixel values across the selected ROIs."
            )
        additional_plans.append(
            AdditionalLabelsPlan(
                name=spec.name,
                key=layer_key,
                suffix=spec.suffix,
                table_name=annotation_table_name,
                roi_key=roi_key,
                region_key=DEFAULT_LABEL_REGION_KEY,
                instance_key=DEFAULT_LABEL_INSTANCE_KEY,
                value_key=spec.value_key,
                name_key=spec.name_key,
                rois=tuple(layer_roi_plans),
            )
        )

    return SpatialDataConversionPlan(
        roi_key=roi_key,
        instance_key=instance_key,
        markers=markers,
        rois=tuple(roi_plans),
        additional_labels=tuple(additional_plans),
    )


def _read_tiff(path: str) -> Any:
    import tifffile

    return tifffile.imread(path)


def _lazy_tiff(
    path: Path, shape: tuple[int, int], dtype: Any, chunks: tuple[int, int]
) -> Any:
    import dask.array as da
    import numpy as np
    from dask import delayed

    return da.from_delayed(
        delayed(_read_tiff)(str(path)),
        shape=shape,
        dtype=np.dtype(dtype),
    ).rechunk(chunks)


def _normalise_chunks(chunks: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(chunks, int):
        result = (chunks, chunks)
    else:
        result = (int(chunks[0]), int(chunks[1]))
    if any(value <= 0 for value in result):
        raise ValueError("raster_chunks must be a positive integer or a (y, x) pair.")
    return result


def _metadata_from_plan(
    plan: SpatialDataConversionPlan,
    *,
    table_name: str,
    table_region_key: str,
    table_instance_key: str,
) -> dict[str, Any]:
    rois: dict[str, Any] = {}
    for roi in plan.rois:
        rois[roi.roi] = {
            "image": roi.image_element,
            "labels": roi.labels_element,
            "coordinate_system": roi.coordinate_system,
            "mask_file": roi.mask_path.name,
            "image_files": {
                match.marker: match.path.name for match in roi.marker_images
            },
            "image_match_modes": {
                match.marker: match.mode for match in roi.marker_images
            },
            "table_instances": roi.table_instances,
            "unannotated_mask_instances": roi.unannotated_mask_instances,
            "additional_labels": {},
        }
    label_layers: dict[str, Any] = {
        "cells": {
            "display_name": "Cells",
            "kind": "instances",
            "annotation_table": table_name,
            "region_key": table_region_key,
            "instance_key": table_instance_key,
            "elements_by_roi": {roi.roi: roi.labels_element for roi in plan.rois},
        }
    }
    for layer in plan.additional_labels:
        elements_by_roi: dict[str, str] = {}
        files_by_roi: dict[str, str] = {}
        unused_by_roi: dict[str, list[int]] = {}
        for roi in layer.rois:
            elements_by_roi[roi.roi] = roi.labels_element
            files_by_roi[roi.roi] = roi.path.name
            if roi.unused_mapping_values:
                unused_by_roi[roi.roi] = list(roi.unused_mapping_values)
            rois[roi.roi]["additional_labels"][layer.key] = roi.labels_element
        label_layers[layer.key] = {
            "display_name": layer.name,
            "kind": "categorical",
            "suffix": layer.suffix,
            "annotation_table": layer.table_name,
            "roi_key": layer.roi_key,
            "region_key": layer.region_key,
            "instance_key": layer.instance_key,
            "value_key": layer.value_key,
            "name_key": layer.name_key,
            "elements_by_roi": elements_by_roi,
            "files_by_roi": files_by_roi,
            "unused_mapping_values_by_roi": unused_by_roi,
        }
    return {
        "schema_version": 2,
        "roi_key": plan.roi_key,
        "source_instance_key": plan.instance_key,
        "table_name": table_name,
        "table_region_key": table_region_key,
        "table_instance_key": table_instance_key,
        "rois": rois,
        "label_layers": label_layers,
    }


def _create_additional_labels_table(layer: AdditionalLabelsPlan) -> Any:
    import anndata as ad
    import numpy as np
    import pandas as pd
    from spatialdata.models import TableModel

    records: list[dict[str, Any]] = []
    obs_names: list[str] = []
    for roi in layer.rois:
        for value, name in roi.value_names:
            records.append(
                {
                    layer.roi_key: roi.roi,
                    layer.region_key: roi.labels_element,
                    layer.instance_key: value,
                    layer.value_key: value,
                    layer.name_key: name,
                }
            )
            obs_names.append(f"{layer.key}:{_safe_name(roi.roi)}:{value}")

    obs = pd.DataFrame.from_records(records, index=obs_names)
    region_categories = [roi.labels_element for roi in layer.rois]
    roi_categories = [roi.roi for roi in layer.rois]
    obs[layer.region_key] = pd.Categorical(
        obs[layer.region_key], categories=region_categories
    )
    obs[layer.roi_key] = pd.Categorical(obs[layer.roi_key], categories=roi_categories)
    obs[layer.instance_key] = obs[layer.instance_key].to_numpy(dtype=np.int64)
    obs[layer.value_key] = obs[layer.value_key].to_numpy(dtype=np.int64)
    name_categories = list(dict.fromkeys(obs[layer.name_key].astype(str)))
    obs[layer.name_key] = pd.Categorical(
        obs[layer.name_key].astype(str), categories=name_categories
    )

    annotation = ad.AnnData(
        X=np.empty((len(obs), 0), dtype=np.float32),
        obs=obs,
    )
    table = TableModel.parse(
        annotation,
        region=region_categories,
        region_key=layer.region_key,
        instance_key=layer.instance_key,
        overwrite_metadata=True,
    )
    table.uns[SBT_METADATA_KEY] = {
        "schema_version": 1,
        "kind": "label_annotations",
        "label_layer": layer.key,
        "display_name": layer.name,
        "roi_key": layer.roi_key,
        "value_key": layer.value_key,
        "name_key": layer.name_key,
    }
    return table


def create_spatialdata(
    adata: Any,
    images_folder: str | Path | None,
    masks_folder: str | Path,
    *,
    roi_key: str = "ROI",
    instance_key: str = "ObjectNumber",
    table_name: str = DEFAULT_TABLE_NAME,
    table_region_key: str = DEFAULT_REGION_KEY,
    table_instance_key: str = DEFAULT_INSTANCE_KEY,
    validate_instance_ids: bool = True,
    raster_chunks: int | tuple[int, int] = (512, 512),
    scale_factors: Sequence[int] | None = None,
    additional_labels: (
        AdditionalLabelsSpec
        | Mapping[str, Any]
        | Sequence[AdditionalLabelsSpec | Mapping[str, Any]]
        | None
    ) = None,
) -> Any:
    """Create a lazy SpatialData object from an IMC AnnData and TIFF folders.

    Parameters
    ----------
    adata:
        AnnData cell table. ``var_names`` define image marker order.
    images_folder:
        Folder containing ``{ROI}/{marker}.tiff``.  Use ``None`` to create a
        labels-and-table SpatialData object without intensity images.
    masks_folder:
        Folder containing one ``{ROI}.tiff`` integer label image per ROI.
    roi_key, instance_key:
        Observation columns linking cells to ROIs and mask labels.
    table_name:
        Name of the annotation table inside SpatialData.
    validate_instance_ids:
        Check that every table cell is present in its mask before construction.
    raster_chunks:
        Dask chunk size for the spatial dimensions.
    scale_factors:
        Optional relative downsampling factors used to create multiscale
        images and labels.
    additional_labels:
        One or several :class:`AdditionalLabelsSpec` objects (or equivalent
        mappings).  Each specification adds one named, ROI-aligned integer
        Labels layer and one linked annotation table containing the semantic
        name for every positive pixel value.

    Notes
    -----
    SpatialData's table parser stores annotation metadata on the supplied
    AnnData object.  This function also adds ``table_region_key`` and
    ``table_instance_key`` to ``adata.obs``; the source ROI and ObjectNumber
    columns are preserved unchanged.  This avoids copying a potentially very
    large AnnData object and also works for read-only backed ``.h5ad`` inputs.
    """

    import dask.array as da
    import numpy as np
    import pandas as pd
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel, Labels2DModel, TableModel
    from spatialdata.transformations import Identity

    chunks = _normalise_chunks(raster_chunks)
    plan = plan_imc_spatialdata_conversion(
        adata,
        images_folder,
        masks_folder,
        roi_key=roi_key,
        instance_key=instance_key,
        validate_instance_ids=validate_instance_ids,
        additional_labels=additional_labels,
    )

    table_names = [table_name, *(layer.table_name for layer in plan.additional_labels)]
    folded_table_names = [name.casefold() for name in table_names]
    if len(set(folded_table_names)) != len(folded_table_names):
        raise ValueError(
            "The primary table and additional label annotation tables must have "
            f"unique names: {table_names}"
        )
    spatial_element_names = {
        name.casefold()
        for roi in plan.rois
        for name in (roi.image_element, roi.labels_element)
        if name is not None
    }
    spatial_element_names.update(
        roi.labels_element.casefold()
        for layer in plan.additional_labels
        for roi in layer.rois
    )
    collisions = [
        name for name in table_names if name.casefold() in spatial_element_names
    ]
    if collisions:
        raise ValueError(
            "SpatialData table names collide with generated spatial elements: "
            + ", ".join(repr(name) for name in collisions)
        )

    region_by_roi = {roi.roi: roi.labels_element for roi in plan.rois}
    source_rois = adata.obs[roi_key].astype(str)
    region_values = source_rois.map(region_by_roi)
    if region_values.isna().any():
        raise RuntimeError("Could not map every AnnData ROI to a labels element.")
    adata.obs[table_region_key] = pd.Categorical(
        region_values,
        categories=[roi.labels_element for roi in plan.rois],
    )
    adata.obs[table_instance_key] = pd.to_numeric(
        adata.obs[instance_key], errors="raise"
    ).to_numpy(dtype=np.int64)

    table = TableModel.parse(
        adata,
        region=[roi.labels_element for roi in plan.rois],
        region_key=table_region_key,
        instance_key=table_instance_key,
        overwrite_metadata=True,
    )

    images: dict[str, Any] = {}
    labels: dict[str, Any] = {}
    coordinate_system_by_roi = {roi.roi: roi.coordinate_system for roi in plan.rois}
    for roi in plan.rois:
        transformations = {roi.coordinate_system: Identity()}
        mask = _lazy_tiff(
            roi.mask_path,
            shape=roi.shape,
            dtype=roi.mask_dtype,
            chunks=chunks,
        )
        labels[roi.labels_element] = Labels2DModel.parse(
            mask,
            dims=("y", "x"),
            transformations=transformations,
            scale_factors=scale_factors,
        )

        if roi.image_element is not None:
            channels = []
            for match in roi.marker_images:
                image_shape, image_dtype = _tiff_metadata(match.path)
                channels.append(
                    _lazy_tiff(
                        match.path,
                        shape=image_shape,
                        dtype=image_dtype,
                        chunks=chunks,
                    )
                )
            image = da.stack(channels, axis=0)
            images[roi.image_element] = Image2DModel.parse(
                image,
                dims=("c", "y", "x"),
                c_coords=list(plan.markers),
                transformations=transformations,
                scale_factors=scale_factors,
            )

    tables: dict[str, Any] = {table_name: table}
    for layer in plan.additional_labels:
        for roi in layer.rois:
            transformations = {coordinate_system_by_roi[roi.roi]: Identity()}
            raster = _lazy_tiff(
                roi.path,
                shape=roi.shape,
                dtype=roi.dtype,
                chunks=chunks,
            )
            labels[roi.labels_element] = Labels2DModel.parse(
                raster,
                dims=("y", "x"),
                transformations=transformations,
                scale_factors=scale_factors,
            )
        tables[layer.table_name] = _create_additional_labels_table(layer)

    metadata = _metadata_from_plan(
        plan,
        table_name=table_name,
        table_region_key=table_region_key,
        table_instance_key=table_instance_key,
    )
    logging.info(
        "Created lazy SpatialData with %d ROIs, %d marker images, %d additional "
        "label TIFFs, and %d cells.",
        plan.n_rois,
        plan.n_image_files,
        plan.n_additional_label_files,
        int(adata.n_obs),
    )
    return SpatialData(
        images=images,
        labels=labels,
        tables=tables,
        attrs={SBT_METADATA_KEY: metadata},
    )


def write_spatialdata(
    sdata: Any,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    zarr_format: Literal[2, 3] = 2,
) -> Path:
    """Write SpatialData to Zarr and return the resolved output path.

    Zarr v2 is the default because it remains broadly interoperable and avoids
    known mixed-version issues in environments where AnnData is paired with a
    newer Zarr v3 release.  Pass ``zarr_format=3`` to use SpatialData's current
    default format when the installed dependency set supports it.
    """

    path = Path(output_path).expanduser().resolve(strict=False)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"SpatialData output already exists: {path}. Pass overwrite=True to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if zarr_format not in {2, 3}:  # pragma: no cover - Literal for typed callers
        raise ValueError("zarr_format must be 2 or 3.")

    # Zarr 3's LocalStore performs metadata updates through temporary-file
    # replacements.  On Windows, concurrent metadata updates and short-lived
    # virus-scanner handles can otherwise produce intermittent WinError 5
    # failures during large, multi-element writes.  Serialising Zarr's async
    # store operations avoids the race without reducing Dask's raster compute
    # parallelism.
    import os
    import time
    import zarr
    from contextlib import contextmanager

    @contextmanager
    def _windows_metadata_replace_retries():
        if os.name != "nt":
            yield
            return

        original_replace = Path.replace
        output_root = path.resolve(strict=False)

        def replace_with_retry(source: Path, target: str | Path) -> Path:
            target_path = Path(target)
            try:
                return original_replace(source, target_path)
            except PermissionError:
                source_path = source.resolve(strict=False)
                if output_root not in source_path.parents or not source.name.endswith(
                    ".partial"
                ):
                    raise
                logging.warning(
                    "Windows temporarily locked Zarr metadata %s; retrying atomic replace.",
                    target_path,
                )
                for attempt in range(1, 101):
                    time.sleep(min(0.05 * attempt, 0.5))
                    try:
                        return original_replace(source, target_path)
                    except PermissionError:
                        if attempt == 100:
                            raise
                raise RuntimeError("Unreachable metadata-retry state.")

        Path.replace = replace_with_retry  # type: ignore[method-assign]
        try:
            yield
        finally:
            Path.replace = original_replace  # type: ignore[method-assign]

    with _windows_metadata_replace_retries(), zarr.config.set({"async.concurrency": 1}):
        if zarr_format == 2:
            from spatialdata._io.format import SpatialDataContainerFormatV01

            sdata.write(
                path,
                overwrite=overwrite,
                sdata_formats=[SpatialDataContainerFormatV01()],
            )
        else:
            sdata.write(path, overwrite=overwrite)
    logging.info("Wrote SpatialData Zarr store to %s", path)
    return path


def _toolkit_metadata(sdata: Any) -> Mapping[str, Any]:
    metadata = getattr(sdata, "attrs", {}).get(SBT_METADATA_KEY)
    if not isinstance(metadata, Mapping):
        raise KeyError(
            f"SpatialData attrs do not contain {SBT_METADATA_KEY!r} metadata."
        )
    return metadata


def _resolve_metadata_roi(
    sdata: Any,
    roi: str,
) -> tuple[str, Mapping[str, Any]]:
    rois = _toolkit_metadata(sdata).get("rois", {})
    if roi in rois:
        selected_roi = str(roi)
    else:
        matches = [key for key in rois if str(key).casefold() == str(roi).casefold()]
        if len(matches) != 1:
            raise KeyError(f"ROI {roi!r} is not present in this SpatialData object.")
        selected_roi = str(matches[0])
    selected = rois[selected_roi]
    if not isinstance(selected, Mapping):
        raise TypeError(f"Metadata for ROI {selected_roi!r} is not a mapping.")
    return selected_roi, selected


def _resolve_label_layer_metadata(
    sdata: Any,
    label_layer: str,
) -> tuple[str, Mapping[str, Any]]:
    metadata = _toolkit_metadata(sdata)
    layers = metadata.get("label_layers", {})
    if not isinstance(layers, Mapping) or not layers:
        if str(label_layer).casefold() == "cells":
            return (
                "cells",
                {
                    "display_name": "Cells",
                    "kind": "instances",
                    "annotation_table": metadata.get("table_name", DEFAULT_TABLE_NAME),
                    "region_key": metadata.get("table_region_key", DEFAULT_REGION_KEY),
                    "instance_key": metadata.get(
                        "table_instance_key", DEFAULT_INSTANCE_KEY
                    ),
                },
            )
        raise KeyError(
            f"Label layer {label_layer!r} is not present; this SpatialData object "
            "predates named additional label layers."
        )

    requested = str(label_layer).casefold()
    matches = [
        (str(key), value)
        for key, value in layers.items()
        if str(key).casefold() == requested
        or (
            isinstance(value, Mapping)
            and str(value.get("display_name", "")).casefold() == requested
        )
    ]
    if len(matches) != 1:
        available = ", ".join(str(key) for key in layers)
        raise KeyError(
            f"Label layer {label_layer!r} was not found. Available layers: {available}"
        )
    key, value = matches[0]
    if not isinstance(value, Mapping):
        raise TypeError(f"Metadata for label layer {key!r} is not a mapping.")
    return key, value


def get_roi_elements(sdata: Any, roi: str) -> dict[str, str | None]:
    """Return image, labels, and coordinate-system names for one source ROI."""

    _selected_roi, selected = _resolve_metadata_roi(sdata, roi)
    return {
        "image": selected.get("image"),
        "labels": selected["labels"],
        "coordinate_system": selected["coordinate_system"],
    }


def get_roi_label_elements(sdata: Any, roi: str) -> dict[str, str]:
    """Return all named Labels elements for one source ROI.

    The primary cell mask is always returned under ``"cells"``.  Additional
    entries use the stable normalized layer keys created from
    :class:`AdditionalLabelsSpec.name`.
    """

    _selected_roi, selected = _resolve_metadata_roi(sdata, roi)
    result = {"cells": str(selected["labels"])}
    additional = selected.get("additional_labels", {})
    if isinstance(additional, Mapping):
        result.update({str(key): str(value) for key, value in additional.items()})
    return result


def get_label_annotations(
    sdata: Any,
    label_layer: str,
    *,
    roi: str | None = None,
) -> Any:
    """Return the human-readable annotation rows for an additional label layer.

    Parameters
    ----------
    sdata
        SpatialData created by :func:`create_spatialdata`.
    label_layer
        Stable layer key or original display name.
    roi
        Optional ROI restriction, matched case-insensitively.

    Returns
    -------
    pandas.DataFrame
        A copy of the annotation table ``obs`` containing ROI, numeric value,
        semantic name, and the formal SpatialData region/instance columns.
    """

    key, layer = _resolve_label_layer_metadata(sdata, label_layer)
    if key == "cells":
        raise ValueError(
            "The primary cell layer is annotated by the cell table rather than "
            "a pixel-value-to-name table."
        )
    table_name = str(layer["annotation_table"])
    if table_name not in sdata.tables:
        raise KeyError(
            f"Annotation table {table_name!r} for label layer {key!r} was not found."
        )
    frame = sdata.tables[table_name].obs.copy()
    if roi is not None:
        selected_roi, _selected = _resolve_metadata_roi(sdata, roi)
        roi_key = str(layer.get("roi_key", _toolkit_metadata(sdata)["roi_key"]))
        if roi_key not in frame.columns:
            raise KeyError(
                f"ROI column {roi_key!r} is missing from annotation table "
                f"{table_name!r}."
            )
        frame = frame.loc[frame[roi_key].astype(str) == selected_roi].copy()
    return frame


def summarize_spatialdata(
    sdata: Any,
    *,
    table_name: str | None = None,
    population_key: str | None = None,
    case_key: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-friendly structural and annotation summary without raster IO."""

    metadata = _toolkit_metadata(sdata)
    selected_table = table_name or str(metadata.get("table_name", DEFAULT_TABLE_NAME))
    if selected_table not in sdata.tables:
        raise KeyError(f"SpatialData table {selected_table!r} was not found.")
    table = sdata.tables[selected_table]
    roi_key = str(metadata.get("roi_key", "ROI"))
    if roi_key not in table.obs.columns:
        raise KeyError(
            f"ROI column {roi_key!r} is missing from table {selected_table!r}."
        )

    def _counts(key: str) -> dict[str, int]:
        if key not in table.obs.columns:
            raise KeyError(f"Column {key!r} is missing from table {selected_table!r}.")
        counts = table.obs[key].value_counts(dropna=False)
        return {str(label): int(count) for label, count in counts.items()}

    summary: dict[str, Any] = {
        "table": selected_table,
        "cells": int(table.n_obs),
        "markers": int(table.n_vars),
        "marker_names": [str(value) for value in table.var_names],
        "rois": len(metadata.get("rois", {})),
        "images": len(sdata.images),
        "labels": len(sdata.labels),
        "shapes": len(sdata.shapes),
        "points": len(sdata.points),
        "cells_per_roi": _counts(roi_key),
        "unannotated_mask_instances": int(
            sum(
                int(values.get("unannotated_mask_instances", 0))
                for values in metadata.get("rois", {}).values()
            )
        ),
    }
    label_layers = metadata.get("label_layers", {})
    if isinstance(label_layers, Mapping):
        summary["label_layers"] = {
            str(key): {
                "display_name": str(value.get("display_name", key)),
                "kind": str(value.get("kind", "unknown")),
                "annotation_table": value.get("annotation_table"),
                "elements": len(value.get("elements_by_roi", {})),
            }
            for key, value in label_layers.items()
            if isinstance(value, Mapping)
        }
    if population_key is not None:
        summary["population_key"] = population_key
        summary["population_counts"] = _counts(population_key)
    if case_key is not None:
        summary["case_key"] = case_key
        summary["cells_per_case"] = _counts(case_key)
    return summary


def _compute_raster(raster: Any) -> Any:
    """Materialize one selected raster or raster slice as a NumPy array."""

    import numpy as np

    data = raster.data
    return np.asarray(data.compute() if hasattr(data, "compute") else data)


def _normalise_image(values: Any) -> Any:
    """Scale finite image intensities to [0, 1] using robust percentiles."""

    import numpy as np

    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.zeros(values.shape, dtype=float)
    low, high = np.percentile(finite, (1.0, 99.5))
    if high <= low:
        high = low + 1.0
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def plot_spatialdata_roi(
    sdata: Any,
    roi: str,
    *,
    channel: str | Sequence[str] | None = None,
    color: str | None = None,
    table_name: str | None = None,
    label_layer: str = "cells",
    image_cmap: str = "gray",
    fill_alpha: float = 0.35,
    contour_px: int | None = 1,
    ax: Any = None,
    figsize: tuple[float, float] = (8.0, 8.0),
    title: str | None = None,
) -> Any:
    """Plot marker intensity and/or named Labels annotations for one ROI.

    This focused Matplotlib renderer intentionally reads only the requested ROI
    and channels.  It is independent of the optional ``spatialdata-plot``
    accessor, which makes it useful when exploring stores across SpatialData
    and spatialdata-plot release combinations.  ``label_layer`` defaults to
    the primary cell mask.  For an additional categorical layer, ``color``
    defaults to that layer's semantic name column.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    elements = get_roi_elements(sdata, roi)
    layer_key, layer_metadata = _resolve_label_layer_metadata(sdata, label_layer)
    roi_label_elements = get_roi_label_elements(sdata, roi)
    if layer_key not in roi_label_elements:
        raise KeyError(
            f"Label layer {layer_key!r} has no Labels element for ROI {roi!r}."
        )
    labels_element = roi_label_elements[layer_key]
    selected_color = color
    if selected_color is None and layer_key != "cells":
        selected_color = str(layer_metadata["name_key"])
    if ax is None:
        _figure, ax = plt.subplots(figsize=figsize)

    image_element = elements["image"]
    if image_element is not None:
        image = sdata.images[image_element]
        selected_channels = (
            [str(image.coords["c"].values[0])]
            if channel is None
            else ([channel] if isinstance(channel, str) else list(channel))
        )
        if len(selected_channels) > 3:
            raise ValueError("At most three channels can be combined in one ROI plot.")
        missing_channels = [
            value
            for value in selected_channels
            if value not in set(str(item) for item in image.coords["c"].values)
        ]
        if missing_channels:
            raise KeyError(
                f"Channels not found in image {image_element!r}: {missing_channels}"
            )
        planes = [
            _normalise_image(_compute_raster(image.sel(c=value)).squeeze())
            for value in selected_channels
        ]
        if len(planes) == 1:
            ax.imshow(planes[0], cmap=image_cmap, interpolation="nearest")
        else:
            rgb = np.zeros((*planes[0].shape, 3), dtype=float)
            for index, plane in enumerate(planes):
                rgb[..., index] = plane
            ax.imshow(rgb, interpolation="nearest")

    labels = _compute_raster(sdata.labels[labels_element]).squeeze()
    if labels.ndim != 2:
        raise ValueError(
            f"Labels element {labels_element!r} is not 2D after loading: "
            f"{labels.shape}."
        )

    if selected_color is not None:
        metadata = _toolkit_metadata(sdata)
        if layer_key == "cells":
            selected_table = table_name or str(
                metadata.get("table_name", DEFAULT_TABLE_NAME)
            )
            region_key = str(metadata.get("table_region_key", DEFAULT_REGION_KEY))
            instance_key = str(metadata.get("table_instance_key", DEFAULT_INSTANCE_KEY))
        else:
            selected_table = table_name or str(layer_metadata["annotation_table"])
            region_key = str(layer_metadata["region_key"])
            instance_key = str(layer_metadata["instance_key"])
        if selected_table not in sdata.tables:
            raise KeyError(f"SpatialData table {selected_table!r} was not found.")
        table = sdata.tables[selected_table]
        if selected_color not in table.obs.columns:
            raise KeyError(
                f"Column {selected_color!r} is missing from table {selected_table!r}."
            )
        region_rows = table.obs[region_key].astype(str) == str(labels_element)
        annotation = table.obs.loc[region_rows, [instance_key, selected_color]].copy()
        annotation = annotation.dropna(subset=[instance_key, selected_color])
        instance_ids = annotation[instance_key].to_numpy(dtype=np.int64)
        max_label = int(labels.max(initial=0))
        valid = (instance_ids >= 0) & (instance_ids <= max_label)
        annotation = annotation.loc[valid]
        instance_ids = instance_ids[valid]

        rgba_lookup = np.zeros((max_label + 1, 4), dtype=float)
        values = annotation[selected_color]
        if isinstance(
            values.dtype, pd.CategoricalDtype
        ) or not pd.api.types.is_numeric_dtype(values):
            categories = [str(value) for value in pd.unique(values.astype(str))]
            cmap = plt.get_cmap("tab20", max(1, len(categories)))
            category_colors = {
                value: cmap(index) for index, value in enumerate(categories)
            }
            for instance_id, value in zip(
                instance_ids, values.astype(str), strict=False
            ):
                rgba_lookup[instance_id] = category_colors[value]
            handles = [
                Patch(facecolor=category_colors[value], label=value)
                for value in categories
            ]
            if handles:
                ax.legend(
                    handles=handles,
                    title=selected_color,
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                    borderaxespad=0,
                )
        else:
            numeric = values.to_numpy(dtype=float)
            norm = Normalize(
                vmin=float(np.nanmin(numeric)),
                vmax=float(np.nanmax(numeric))
                if float(np.nanmax(numeric)) > float(np.nanmin(numeric))
                else float(np.nanmin(numeric)) + 1.0,
            )
            cmap = plt.get_cmap("viridis")
            rgba_lookup[instance_ids] = cmap(norm(numeric))
            plt.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
                label=selected_color,
            )
        rgba_lookup[:, 3] *= fill_alpha
        ax.imshow(rgba_lookup[labels.astype(np.int64)], interpolation="nearest")
    elif image_element is None:
        ax.imshow(labels, cmap="nipy_spectral", interpolation="nearest")

    if contour_px is not None and contour_px > 0:
        from skimage.segmentation import find_boundaries

        boundaries = find_boundaries(labels, mode="inner")
        boundary_rgba = np.zeros((*labels.shape, 4), dtype=float)
        boundary_rgba[boundaries] = (1.0, 1.0, 1.0, 0.65)
        ax.imshow(boundary_rgba, interpolation="nearest")

    ax.set_title(title or str(roi))
    ax.set_axis_off()
    return ax


def plot_spatialdata_cells(
    sdata: Any,
    cells: Any | Sequence[Any],
    *,
    cell_key: str | None = None,
    roi: str | None = None,
    channel: str | Sequence[str] | None = None,
    color: str | None = None,
    table_name: str | None = None,
    crop_size: int | tuple[int, int] = 64,
    ncols: int = 4,
    image_cmap: str = "gray",
    fill_alpha: float = 0.35,
    contour_px: int | None = 1,
    outline_target_only: bool = False,
    mask_outside_target: bool = False,
    target_color: str = "#00FFFF",
    boundary_color: str = "white",
    show_ax_titles: bool = True,
    ax_title_size: float = 9.0,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> tuple[Any, Any]:
    """Plot one or more cells as fixed-size image crops in a gallery.

    By default, ``cells`` contains AnnData observation names.  To select by an
    ``obs`` column instead, pass ``cell_key``; each requested value must resolve
    to exactly one table row after the optional ``roi`` restriction.  This is
    particularly useful for ROI-local ``ObjectNumber`` values::

        plot_spatialdata_cells(
            sdata,
            cells=[12, 35],
            cell_key="ObjectNumber",
            roi="ROI_1",
            channel=["CD3", "CD4"],
            color="leiden_1.0",
        )

    The function reads each required ROI mask and marker plane once, then crops
    around the centre of each target mask instance.  It returns the Matplotlib
    figure and a one-dimensional array containing the populated gallery axes.

    Parameters
    ----------
    sdata
        SpatialData object created by :func:`create_spatialdata`.
    cells
        One cell identifier or a sequence of identifiers.  Identifiers are
        observation names unless ``cell_key`` is supplied.
    cell_key
        Optional ``table.obs`` column used to resolve ``cells``.
    roi
        Optional source ROI restriction.  Usually required when selecting by
        an ROI-local key such as ``ObjectNumber``.
    channel
        One marker or up to three markers rendered as grayscale or RGB.
    color
        Optional ``table.obs`` column used to color each highlighted target
        cell and add a gallery legend or colorbar.
    crop_size
        Crop height and width in pixels, either as one integer or ``(h, w)``.
    ncols
        Maximum number of gallery columns.
    outline_target_only
        If true, draw only the selected cell's boundary instead of outlining
        every labelled cell visible in its crop.
    mask_outside_target
        If true, cover every pixel outside the selected cell with opaque black.
    show_ax_titles
        If false, hide the observation, ROI, instance, and annotation title
        above each gallery panel.  This does not affect the figure-level
        ``title``.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import Normalize, to_rgba
    from matplotlib.patches import Patch
    from skimage.segmentation import find_boundaries

    selectors: list[Any]
    if isinstance(cells, (str, bytes)):
        selectors = [cells]
    else:
        try:
            selectors = list(cells)
        except TypeError:
            selectors = [cells]
    if not selectors:
        raise ValueError("At least one cell must be selected.")
    if ncols < 1:
        raise ValueError("ncols must be at least 1.")
    if not 0.0 <= fill_alpha <= 1.0:
        raise ValueError("fill_alpha must be between 0 and 1.")

    if isinstance(crop_size, int):
        crop_shape = (crop_size, crop_size)
    else:
        if len(crop_size) != 2:
            raise ValueError("crop_size must be an integer or a (height, width) pair.")
        crop_shape = (int(crop_size[0]), int(crop_size[1]))
    if any(value < 1 for value in crop_shape):
        raise ValueError("crop_size values must be positive integers.")

    if channel is None:
        requested_channels = None
    elif isinstance(channel, str):
        requested_channels = [channel]
    else:
        requested_channels = [str(value) for value in channel]
        if not requested_channels:
            raise ValueError("channel must contain at least one marker name.")
    if requested_channels is not None and len(requested_channels) > 3:
        raise ValueError("At most three channels can be combined in one cell plot.")

    metadata = _toolkit_metadata(sdata)
    selected_table = table_name or str(metadata.get("table_name", DEFAULT_TABLE_NAME))
    if selected_table not in sdata.tables:
        raise KeyError(f"SpatialData table {selected_table!r} was not found.")
    table = sdata.tables[selected_table]
    obs = table.obs
    region_key = str(metadata.get("table_region_key", DEFAULT_REGION_KEY))
    instance_key = str(metadata.get("table_instance_key", DEFAULT_INSTANCE_KEY))
    display_instance_key = str(metadata.get("source_instance_key", instance_key))
    required_columns = [region_key, instance_key]
    if color is not None:
        required_columns.append(color)
    missing_columns = [value for value in required_columns if value not in obs.columns]
    if missing_columns:
        raise KeyError(
            f"Columns missing from SpatialData table {selected_table!r}: "
            f"{missing_columns}"
        )
    if cell_key is not None and cell_key not in obs.columns:
        raise KeyError(
            f"Cell selection column {cell_key!r} is missing from table "
            f"{selected_table!r}."
        )

    roi_region: str | None = None
    if roi is not None:
        roi_region = str(get_roi_elements(sdata, roi)["labels"])

    positions: list[int] = []
    if cell_key is None:
        if not obs.index.is_unique:
            raise ValueError(
                "The SpatialData table has non-unique observation names; provide a "
                "unique obs column through cell_key."
            )
        for selector in selectors:
            obs_name = str(selector)
            try:
                position = int(obs.index.get_loc(obs_name))
            except KeyError as error:
                raise KeyError(
                    f"Observation {obs_name!r} is not present in table "
                    f"{selected_table!r}."
                ) from error
            if (
                roi_region is not None
                and str(obs.iloc[position][region_key]) != roi_region
            ):
                raise ValueError(
                    f"Observation {obs_name!r} does not belong to ROI {roi!r}."
                )
            positions.append(position)
    else:
        candidate_positions = np.arange(len(obs), dtype=np.int64)
        if roi_region is not None:
            in_roi = obs[region_key].astype(str).to_numpy() == roi_region
            candidate_positions = candidate_positions[in_roi]
        candidate_values = obs.iloc[candidate_positions][cell_key]
        for selector in selectors:
            matches = candidate_positions[
                np.asarray(candidate_values == selector, dtype=bool)
            ]
            if not len(matches):
                roi_text = f" in ROI {roi!r}" if roi is not None else ""
                raise KeyError(f"No table row has {cell_key!r}={selector!r}{roi_text}.")
            if len(matches) > 1:
                raise ValueError(
                    f"{len(matches)} table rows have {cell_key!r}={selector!r}; "
                    "specify roi=... or select by unique observation names."
                )
            positions.append(int(matches[0]))

    roi_by_labels: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for source_roi, values in metadata.get("rois", {}).items():
        labels_name = str(values["labels"])
        if labels_name in roi_by_labels:
            raise ValueError(
                f"SpatialData metadata maps more than one ROI to labels "
                f"element {labels_name!r}."
            )
        roi_by_labels[labels_name] = (str(source_roi), values)

    records: list[dict[str, Any]] = []
    for gallery_index, position in enumerate(positions):
        row = obs.iloc[position]
        labels_name = str(row[region_key])
        if labels_name not in roi_by_labels:
            raise KeyError(
                f"Table row {obs.index[position]!r} refers to labels element "
                f"{labels_name!r}, which is absent from toolkit ROI metadata."
            )
        source_roi, roi_values = roi_by_labels[labels_name]
        if labels_name not in sdata.labels:
            raise KeyError(f"Labels element {labels_name!r} was not found.")
        raw_instance = row[instance_key]
        if pd.isna(raw_instance):
            raise ValueError(
                f"Cell {obs.index[position]!r} has no value in {instance_key!r}."
            )
        try:
            numeric_instance = float(raw_instance)
            instance_id = int(numeric_instance)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"Cell {obs.index[position]!r} has invalid instance ID "
                f"{raw_instance!r}."
            ) from error
        if not np.isfinite(numeric_instance) or numeric_instance != instance_id:
            raise ValueError(
                f"Cell {obs.index[position]!r} has non-integer instance ID "
                f"{raw_instance!r}."
            )
        if instance_id < 1:
            raise ValueError(
                f"Cell {obs.index[position]!r} has non-positive instance ID "
                f"{instance_id}."
            )
        image_name = roi_values.get("image")
        if image_name is not None and image_name not in sdata.images:
            raise KeyError(f"Image element {image_name!r} was not found.")
        if image_name is None and requested_channels is not None:
            raise ValueError(
                f"ROI {source_roi!r} has no image element, so channel crops "
                "cannot be plotted."
            )
        records.append(
            {
                "gallery_index": gallery_index,
                "obs_name": str(obs.index[position]),
                "position": position,
                "roi": source_roi,
                "labels": labels_name,
                "image": image_name,
                "instance_id": instance_id,
                "display_instance": row.get(display_instance_key, instance_id),
            }
        )

    selected_obs = obs.iloc[positions]
    cell_colors = [to_rgba(target_color)] * len(records)
    legend_handles: list[Any] = []
    scalar_mappable: Any = None
    if color is not None:
        values = selected_obs[color]
        is_categorical = isinstance(values.dtype, pd.CategoricalDtype)
        if is_categorical or not pd.api.types.is_numeric_dtype(values):
            categories = [
                str(value) for value in pd.unique(values.dropna().astype(str))
            ]
            cmap = plt.get_cmap("tab20", max(1, len(categories)))
            category_colors = {
                value: cmap(index) for index, value in enumerate(categories)
            }
            cell_colors = [
                category_colors.get(str(value), to_rgba("#BDBDBD"))
                if not pd.isna(value)
                else to_rgba("#BDBDBD")
                for value in values
            ]
            legend_handles = [
                Patch(facecolor=category_colors[value], label=value)
                for value in categories
            ]
        else:
            numeric_values = values.to_numpy(dtype=float)
            finite_values = numeric_values[np.isfinite(numeric_values)]
            if len(finite_values):
                low = float(np.min(finite_values))
                high = float(np.max(finite_values))
                norm = Normalize(vmin=low, vmax=high if high > low else low + 1.0)
                cmap = plt.get_cmap("viridis")
                cell_colors = [
                    cmap(norm(value)) if np.isfinite(value) else to_rgba("#BDBDBD")
                    for value in numeric_values
                ]
                scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    gallery_columns = min(ncols, len(records))
    gallery_rows = int(np.ceil(len(records) / gallery_columns))
    gallery_figsize = figsize or (4.0 * gallery_columns, 4.0 * gallery_rows)
    figure, axes = plt.subplots(
        gallery_rows,
        gallery_columns,
        figsize=gallery_figsize,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for unused_axis in flat_axes[len(records) :]:
        unused_axis.set_axis_off()

    records_by_labels: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_labels.setdefault(record["labels"], []).append(record)

    def _centered_slice(center: int, size: int, limit: int) -> slice:
        start = center - size // 2
        stop = start + size
        if start < 0:
            start = 0
            stop = min(limit, size)
        elif stop > limit:
            stop = limit
            start = max(0, stop - size)
        return slice(start, stop)

    for labels_name, roi_records in records_by_labels.items():
        labels = _compute_raster(sdata.labels[labels_name]).squeeze()
        if labels.ndim != 2:
            raise ValueError(
                f"Labels element {labels_name!r} is not 2D after loading: "
                f"{labels.shape}."
            )

        image_name = roi_records[0]["image"]
        image_planes: list[Any] = []
        selected_channels: list[str] = []
        if image_name is not None:
            image = sdata.images[image_name]
            available_channels = [str(value) for value in image.coords["c"].values]
            selected_channels = (
                [available_channels[0]]
                if requested_channels is None
                else requested_channels
            )
            missing_channels = [
                value for value in selected_channels if value not in available_channels
            ]
            if missing_channels:
                raise KeyError(
                    f"Channels not found in image {image_name!r}: {missing_channels}"
                )
            image_planes = [
                _compute_raster(image.sel(c=value)).squeeze()
                for value in selected_channels
            ]
            bad_shapes = [
                plane.shape for plane in image_planes if plane.shape != labels.shape
            ]
            if bad_shapes:
                raise ValueError(
                    f"Image {image_name!r} and labels {labels_name!r} have "
                    f"different shapes: {bad_shapes[0]} and {labels.shape}."
                )

        for record in roi_records:
            instance_id = record["instance_id"]
            rows, columns = np.nonzero(labels == instance_id)
            if not len(rows):
                raise ValueError(
                    f"Instance {instance_id} for cell {record['obs_name']!r} is "
                    f"absent from labels element {labels_name!r}."
                )
            y_center = int(round((int(rows.min()) + int(rows.max())) / 2.0))
            x_center = int(round((int(columns.min()) + int(columns.max())) / 2.0))
            y_slice = _centered_slice(y_center, crop_shape[0], labels.shape[0])
            x_slice = _centered_slice(x_center, crop_shape[1], labels.shape[1])
            labels_crop = labels[y_slice, x_slice]
            target_mask = labels_crop == instance_id
            axis = flat_axes[record["gallery_index"]]

            if image_planes:
                crops = [
                    _normalise_image(plane[y_slice, x_slice]) for plane in image_planes
                ]
                if len(crops) == 1:
                    axis.imshow(crops[0], cmap=image_cmap, interpolation="nearest")
                else:
                    rgb = np.zeros((*crops[0].shape, 3), dtype=float)
                    for channel_index, crop in enumerate(crops):
                        rgb[..., channel_index] = crop
                    axis.imshow(rgb, interpolation="nearest")
            else:
                axis.imshow(
                    labels_crop > 0,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    alpha=0.25,
                    interpolation="nearest",
                )

            if mask_outside_target:
                outside_rgba = np.zeros((*labels_crop.shape, 4), dtype=float)
                outside_rgba[~target_mask] = (0.0, 0.0, 0.0, 1.0)
                axis.imshow(outside_rgba, interpolation="nearest")

            face_color = cell_colors[record["gallery_index"]]
            target_rgba = np.zeros((*labels_crop.shape, 4), dtype=float)
            target_rgba[target_mask] = (*face_color[:3], fill_alpha)
            axis.imshow(target_rgba, interpolation="nearest")

            if contour_px is not None and contour_px > 0:
                target_boundaries = find_boundaries(target_mask, mode="inner")
                if contour_px > 1:
                    from scipy.ndimage import binary_dilation

                    target_boundaries = binary_dilation(
                        target_boundaries, iterations=contour_px - 1
                    )
                boundary_rgba = np.zeros((*labels_crop.shape, 4), dtype=float)
                if not outline_target_only:
                    all_boundaries = find_boundaries(labels_crop, mode="inner")
                    if contour_px > 1:
                        all_boundaries = binary_dilation(
                            all_boundaries, iterations=contour_px - 1
                        )
                    boundary_rgba[all_boundaries] = (
                        *to_rgba(boundary_color)[:3],
                        0.6,
                    )
                boundary_rgba[target_boundaries] = (*face_color[:3], 1.0)
                axis.imshow(boundary_rgba, interpolation="nearest")

            if show_ax_titles:
                panel_title = (
                    f"{record['obs_name']}\n{record['roi']} | "
                    f"{display_instance_key}={record['display_instance']}"
                )
                if color is not None:
                    annotation = selected_obs.iloc[record["gallery_index"]][color]
                    panel_title += f"\n{color}={annotation}"
                axis.set_title(panel_title, fontsize=ax_title_size)
            axis.set_axis_off()

    used_axes = flat_axes[: len(records)]
    if title is not None:
        figure.suptitle(title)
    if legend_handles:
        figure.legend(
            handles=legend_handles,
            title=color,
            bbox_to_anchor=(0.99, 0.98),
            loc="upper right",
            borderaxespad=0,
        )
        figure.subplots_adjust(
            right=0.80,
            top=0.90 if title is not None else 0.95,
            hspace=0.35,
        )
    elif scalar_mappable is not None:
        figure.colorbar(
            scalar_mappable,
            ax=list(used_axes),
            label=color,
            fraction=0.025,
            pad=0.02,
        )
        figure.subplots_adjust(top=0.90 if title is not None else 0.95, hspace=0.35)
    else:
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95 if title is not None else 1.0))
    return figure, used_axes


def plot_population_counts(
    sdata: Any,
    population_key: str,
    *,
    table_name: str | None = None,
    ax: Any = None,
    max_populations: int | None = None,
) -> Any:
    """Plot descending cell counts for a population annotation column."""

    import matplotlib.pyplot as plt

    metadata = _toolkit_metadata(sdata)
    selected_table = table_name or str(metadata.get("table_name", DEFAULT_TABLE_NAME))
    if selected_table not in sdata.tables:
        raise KeyError(f"SpatialData table {selected_table!r} was not found.")
    table = sdata.tables[selected_table]
    if population_key not in table.obs.columns:
        raise KeyError(
            f"Column {population_key!r} is missing from table {selected_table!r}."
        )
    counts = table.obs[population_key].value_counts(dropna=False)
    if max_populations is not None:
        counts = counts.iloc[:max_populations]
    if ax is None:
        _figure, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(counts))))
    counts.sort_values().plot.barh(ax=ax, color="#4C78A8")
    ax.set_xlabel("Cells")
    ax.set_ylabel(population_key)
    ax.set_title(f"Population abundance: {population_key}")
    return ax


__all__ = [
    "DEFAULT_INSTANCE_KEY",
    "DEFAULT_LABEL_INSTANCE_KEY",
    "DEFAULT_LABEL_NAME_KEY",
    "DEFAULT_LABEL_REGION_KEY",
    "DEFAULT_LABEL_VALUE_KEY",
    "DEFAULT_REGION_KEY",
    "DEFAULT_TABLE_NAME",
    "AdditionalLabelsPlan",
    "AdditionalLabelsROIPlan",
    "AdditionalLabelsSpec",
    "MarkerImageMatch",
    "ROIConversionPlan",
    "SBT_METADATA_KEY",
    "SpatialDataConversionPlan",
    "create_spatialdata",
    "get_label_annotations",
    "get_roi_elements",
    "get_roi_label_elements",
    "match_marker_image",
    "plan_imc_spatialdata_conversion",
    "plot_population_counts",
    "plot_spatialdata_cells",
    "plot_spatialdata_roi",
    "summarize_spatialdata",
    "write_spatialdata",
]
