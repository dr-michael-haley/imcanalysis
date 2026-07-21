"""Reusable CellVision identity, scPortrait extraction, and plotting helpers.

CellVision learns image representations for segmented IMC cells.  This module
keeps the scientific operations importable while the modules in
``SpatialBiologyToolkit.scripts`` own configuration and execution reporting.
Optional heavy dependencies are imported inside the functions that need them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray


H5SC_FILENAME = "single_cells.h5sc"
IDENTITY_FILENAME = "cell_identity.csv"
EXTRACTION_METADATA_FILENAME = "extraction_metadata.json"
SCPORTRAIT_CONFIG_FILENAME = "scportrait_config.generated.yml"
NORMALIZATION_FILENAME = "normalization_dict.json"
MODEL_FILENAME = "vicreg_encoder.pt"
EMBEDDINGS_FILENAME = "cellvision_embeddings.h5ad"
CLUSTERED_FILENAME = "cellvision_clustered.h5ad"
TRAINING_HISTORY_FILENAME = "vicreg_training_history.csv"
SUPPORTED_TIFF_ENDINGS = (".tif", ".tiff", ".ome.tif", ".ome.tiff")


@dataclass(frozen=True)
class CellVisionPaths:
    """Canonical reusable paths below one configured CellVision asset folder."""

    root: Path
    h5sc: Path
    identity: Path
    extraction_metadata: Path
    scportrait_config: Path
    normalization_dict: Path
    model: Path
    embeddings: Path
    clustered: Path


@dataclass(frozen=True)
class ROIInput:
    """Validated image and mask inputs for one ROI."""

    name: str
    channel_files: tuple[Path, ...]
    channel_names: tuple[str, ...]
    mask_path: Path
    spatial_shape: tuple[int, int]


def resolve_cellvision_paths(asset_folder: str | Path) -> CellVisionPaths:
    """Return standard reusable asset paths below ``asset_folder``."""
    root = Path(asset_folder).expanduser().resolve(strict=False)
    return CellVisionPaths(
        root=root,
        h5sc=root / "extraction" / "data" / H5SC_FILENAME,
        identity=root / IDENTITY_FILENAME,
        extraction_metadata=root / EXTRACTION_METADATA_FILENAME,
        scportrait_config=root / SCPORTRAIT_CONFIG_FILENAME,
        normalization_dict=root / NORMALIZATION_FILENAME,
        model=root / MODEL_FILENAME,
        embeddings=root / EMBEDDINGS_FILENAME,
        clustered=root / CLUSTERED_FILENAME,
    )


def resolution_label(value: float) -> str:
    """Format one clustering resolution consistently for keys and filenames."""
    return format(float(value), ".12g")


def leiden_key(value: float) -> str:
    """Return the namespaced CellVision Leiden observation key."""
    return f"cellvision_leiden_{resolution_label(value)}"


def _clean_string_values(values: pd.Series, *, field_name: str) -> pd.Series:
    cleaned = values.astype("string").str.strip()
    invalid = cleaned.isna() | cleaned.eq("")
    if invalid.any():
        examples = values.index[invalid].astype(str).tolist()[:5]
        raise ValueError(
            f"{field_name} contains null or empty values for selected cells; "
            f"example source observations: {examples}"
        )
    return cleaned.astype(str)


def select_source_cells(
    adata: Any,
    *,
    roi_obs: str,
    object_id_obs: str,
    population_obs: str | None = None,
    populations: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Select source cells and build the canonical identity table.

    The returned frame is ordered like the source AnnData and includes both the
    immutable source row name and the exact ROI/mask-label pair.  Stringified
    source names and ROI/object pairs must each be unique.
    """
    obs = adata.obs
    obs_names = pd.Index(adata.obs_names)
    if not obs_names.is_unique:
        raise ValueError("Source AnnData obs_names must be unique for CellVision identity tracking.")
    source_ids = obs_names.astype(str)
    if not source_ids.is_unique:
        raise ValueError(
            "Source AnnData obs_names become non-unique after string conversion; "
            "use stable unique observation names before CellVision."
        )

    required = [roi_obs, object_id_obs]
    if population_obs is not None:
        required.append(population_obs)
    missing = [column for column in required if column not in obs.columns]
    if missing:
        raise KeyError(f"CellVision source AnnData is missing obs column(s): {missing}")

    frame = obs.loc[:, required].copy()
    frame.insert(0, "source_obs_position", np.arange(len(frame), dtype=np.int64))
    frame.insert(0, "source_obs_id", source_ids.to_numpy())
    frame[roi_obs] = _clean_string_values(frame[roi_obs], field_name=f"adata.obs[{roi_obs!r}]")

    numeric_ids = pd.to_numeric(frame[object_id_obs], errors="coerce")
    invalid_ids = numeric_ids.isna() | ~np.isfinite(numeric_ids) | (numeric_ids <= 0)
    integer_ids = numeric_ids.fillna(-1).astype(np.int64)
    invalid_ids |= numeric_ids.fillna(-1).ne(integer_ids)
    if invalid_ids.any():
        examples = frame.loc[invalid_ids, ["source_obs_id", object_id_obs]].head().to_dict("records")
        raise ValueError(
            f"adata.obs[{object_id_obs!r}] must contain positive integer mask labels; "
            f"examples: {examples}"
        )
    frame[object_id_obs] = integer_ids

    if population_obs is not None:
        frame[population_obs] = _clean_string_values(
            frame[population_obs], field_name=f"adata.obs[{population_obs!r}]"
        )
        if populations is not None:
            requested = [str(value) for value in populations]
            observed = set(frame[population_obs])
            absent = [value for value in requested if value not in observed]
            if absent:
                raise ValueError(
                    f"Configured cellvision.populations were not found in "
                    f"adata.obs[{population_obs!r}]: {absent}"
                )
            frame = frame[frame[population_obs].isin(requested)].copy()

    if frame.empty:
        raise ValueError("CellVision cell selection produced zero source cells.")
    duplicates = frame.duplicated([roi_obs, object_id_obs], keep=False)
    if duplicates.any():
        examples = frame.loc[
            duplicates, ["source_obs_id", roi_obs, object_id_obs]
        ].head().to_dict("records")
        raise ValueError(
            "Each selected CellVision cell must have a unique (ROI, object ID) pair; "
            f"examples: {examples}"
        )

    frame = frame.sort_values("source_obs_position", kind="stable").reset_index(drop=True)
    frame.insert(0, "cellvision_id", frame["source_obs_id"].astype(str))
    frame.insert(1, "scportrait_cell_id", np.arange(1, len(frame) + 1, dtype=np.uint64))
    frame["extraction_status"] = "requested"
    return frame


def identity_fingerprint(
    identity: pd.DataFrame,
    *,
    roi_obs: str,
    object_id_obs: str,
    markers: Sequence[str],
    image_size: int,
    extraction_parameters: Mapping[str, Any] | None = None,
    input_manifest: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Hash the ordered cell identities and extraction-defining choices."""
    records = identity[
        ["cellvision_id", "source_obs_position", roi_obs, object_id_obs, "scportrait_cell_id"]
    ].to_dict("records")
    payload = {
        "schema_version": 1,
        "cells": records,
        "markers": [str(value) for value in markers],
        "image_size": int(image_size),
        "extraction_parameters": dict(extraction_parameters or {}),
        "input_manifest": [dict(record) for record in (input_manifest or [])],
    }
    return configuration_fingerprint(payload)


def configuration_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 fingerprint for a JSON-compatible contract."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def input_file_manifest(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Capture cheap path, size, and modification metadata for source reuse checks."""
    manifest: list[dict[str, Any]] = []
    for path in sorted({value.resolve(strict=True) for value in paths}, key=str):
        stat = path.stat()
        manifest.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return manifest


def _tiff_stem(path: Path) -> str:
    name = path.name
    lowered = name.lower()
    for ending in sorted(SUPPORTED_TIFF_ENDINGS, key=len, reverse=True):
        if lowered.endswith(ending):
            return name[: -len(ending)]
    return path.stem


def _channel_components(path: Path) -> tuple[str, str, str]:
    stem = _tiff_stem(path)
    parts = stem.split("_", 3)
    if len(parts) == 4:
        return parts[2], parts[3], stem
    if len(parts) >= 3:
        return parts[2], parts[2], stem
    return stem, stem, stem


def _tiff_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"CellVision image ROI directory does not exist: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name.lower().endswith(SUPPORTED_TIFF_ENDINGS)
    )


def resolve_roi_channels(
    roi_directory: Path,
    markers: Sequence[str] | None,
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """Resolve ordered channels by an exact, case-insensitive TIFF-stem suffix."""
    files = _tiff_files(roi_directory)
    if not files:
        raise ValueError(f"No TIFF channel files were found in {roi_directory}")

    components = {path: _channel_components(path) for path in files}
    if markers is None:
        names = tuple(components[path][1] for path in files)
        if len(set(name.casefold() for name in names)) != len(names):
            raise ValueError(
                f"Derived channel labels are not unique in ROI {roi_directory.name}: {names}"
            )
        return tuple(files), names

    selected_files: list[Path] = []
    selected_names: list[str] = []
    for marker in markers:
        marker_name = str(marker).strip()
        if not marker_name:
            raise ValueError("CellVision marker names cannot be empty.")
        key = marker_name.casefold()
        matches = [path for path in files if _tiff_stem(path).casefold().endswith(key)]
        if not matches:
            available = [_tiff_stem(path) for path in files]
            raise ValueError(
                f"Marker {marker!r} was not found in ROI {roi_directory.name}. "
                "Markers must match the case-insensitive suffix immediately before the "
                f"TIFF extension. Available TIFF stems: {available}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Marker {marker!r} is ambiguous in ROI {roi_directory.name}: "
                f"{[path.name for path in matches]}"
            )
        if matches[0] in selected_files:
            previous = selected_names[selected_files.index(matches[0])]
            raise ValueError(
                f"CellVision markers {previous!r} and {marker_name!r} both resolve to "
                f"{matches[0].name!r} in ROI {roi_directory.name}. Use one canonical "
                "name for each selected TIFF channel."
            )
        selected_files.append(matches[0])
        selected_names.append(marker_name)
    return tuple(selected_files), tuple(selected_names)


def _spatial_shape(array: np.ndarray, *, path: Path) -> tuple[int, int]:
    squeezed = np.squeeze(array)
    if squeezed.ndim != 2:
        raise ValueError(f"Expected a 2D image at {path}, got shape {array.shape}")
    return int(squeezed.shape[0]), int(squeezed.shape[1])


def _resolve_mask_path(mask_folder: Path, roi: str) -> Path:
    matches = [
        path
        for path in mask_folder.iterdir()
        if path.is_file()
        and path.name.lower().endswith(SUPPORTED_TIFF_ENDINGS)
        and _tiff_stem(path) == roi
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one mask TIFF with stem {roi!r} in {mask_folder}; "
            f"found {[path.name for path in matches]}"
        )
    return matches[0]


def discover_roi_inputs(
    images_folder: Path,
    masks_folder: Path,
    identity: pd.DataFrame,
    *,
    roi_obs: str,
    markers: Sequence[str] | None,
) -> tuple[list[ROIInput], list[str]]:
    """Discover and validate all ROI image/mask inputs needed by selected cells."""
    from tifffile import imread

    if not images_folder.is_dir():
        raise FileNotFoundError(f"CellVision image folder does not exist: {images_folder}")
    if not masks_folder.is_dir():
        raise FileNotFoundError(f"CellVision mask folder does not exist: {masks_folder}")

    contexts: list[ROIInput] = []
    expected_names: tuple[str, ...] | None = None
    for roi in sorted(identity[roi_obs].astype(str).unique()):
        roi_directory = images_folder / roi
        channel_files, channel_names = resolve_roi_channels(roi_directory, markers)
        if expected_names is None:
            expected_names = channel_names
        elif channel_names != expected_names:
            raise ValueError(
                f"ROI {roi!r} resolved channels {channel_names}, expected {expected_names}. "
                "Every CellVision ROI must use the same ordered marker set."
            )
        example = np.asarray(imread(channel_files[0]))
        shape = _spatial_shape(example, path=channel_files[0])
        contexts.append(
            ROIInput(
                name=roi,
                channel_files=channel_files,
                channel_names=channel_names,
                mask_path=_resolve_mask_path(masks_folder, roi),
                spatial_shape=shape,
            )
        )
    if not contexts or expected_names is None:
        raise ValueError("No CellVision ROI inputs were discovered.")
    return contexts, list(expected_names)


def relabel_selected_mask(
    mask: np.ndarray,
    selected: pd.DataFrame,
    *,
    object_id_obs: str,
) -> np.ndarray:
    """Drop unselected mask labels and apply globally unique scPortrait IDs."""
    labels = np.asarray(np.squeeze(mask))
    if labels.ndim != 2:
        raise ValueError(f"CellVision masks must be 2D after squeeze; got {mask.shape}")
    if np.any(labels < 0):
        raise ValueError("CellVision masks cannot contain negative labels.")
    labels = labels.astype(np.uint64, copy=False)
    requested = selected[object_id_obs].astype(np.uint64).to_numpy()
    present = set(np.unique(labels).tolist())
    missing = [int(value) for value in requested if int(value) not in present]
    if missing:
        raise ValueError(
            f"{len(missing)} selected object label(s) were absent from the ROI mask; "
            f"examples: {missing[:10]}"
        )
    maximum = int(labels.max(initial=0))
    lookup: NDArray[np.uint64] = np.zeros(maximum + 1, dtype=np.uint64)
    lookup[requested] = selected["scportrait_cell_id"].astype(np.uint64).to_numpy()
    return lookup[labels]


def _validate_channel_image(array: np.ndarray, *, path: Path) -> np.ndarray:
    image = np.asarray(np.squeeze(array))
    if image.ndim != 2:
        raise ValueError(f"Expected 2D channel TIFF at {path}, got shape {array.shape}")
    if not np.issubdtype(image.dtype, np.number) or not np.all(np.isfinite(image)):
        raise ValueError(f"Channel TIFF contains non-finite or non-numeric values: {path}")
    if np.any(image < 0):
        raise ValueError(f"Channel TIFF contains negative intensities: {path}")
    return image.astype(np.float32, copy=False)


def validate_normalization_dict(
    values: Mapping[str, Any],
    *,
    channel_names: Sequence[str],
) -> dict[str, float]:
    """Resolve and validate a Nimbus-format marker-to-normalization mapping.

    CellVision channel names remain canonical downstream identities. Dictionary
    keys are resolved case-insensitively, preferring an exact match and otherwise
    requiring one unique suffix-compatible match. This permits a Nimbus key such
    as ``CD11c`` to serve ``165Ho_CD11c`` without allowing ``CD3`` to match
    ``CD31``.
    """
    expected = [str(name).strip() for name in channel_names]
    if any(not name for name in expected):
        raise ValueError("CellVision channel names cannot be empty.")
    if len({name.casefold() for name in expected}) != len(expected):
        raise ValueError(
            f"CellVision channel names must be unique ignoring case; got {expected}."
        )

    keyed_values = [(str(key).strip(), key, value) for key, value in values.items()]
    empty_keys = [original for key, original, _ in keyed_values if not key]
    if empty_keys:
        raise ValueError(
            f"CellVision normalization dictionary contains empty key(s): {empty_keys}."
        )

    resolved: dict[str, tuple[str, Any, Any]] = {}
    used_original_keys: set[Any] = set()
    for channel_name in expected:
        folded = channel_name.casefold()
        exact = [entry for entry in keyed_values if entry[0].casefold() == folded]
        candidates = exact or [
            entry
            for entry in keyed_values
            if folded.endswith(entry[0].casefold())
            or entry[0].casefold().endswith(folded)
        ]
        if not candidates:
            raise ValueError(
                "CellVision normalization dictionary is missing a selected marker; "
                f"channel={channel_name!r}. Keys may be exact names or unique "
                "case-insensitive suffix aliases."
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Normalization dictionary match for channel {channel_name!r} is "
                f"ambiguous: {[entry[0] for entry in candidates]}. Use an exact, "
                "unique key."
            )
        matched_key, original_key, raw_value = candidates[0]
        if original_key in used_original_keys:
            previous = next(
                name for name, (_, key, _) in resolved.items() if key == original_key
            )
            raise ValueError(
                f"Normalization key {matched_key!r} matches more than one selected "
                f"channel ({previous!r}, {channel_name!r}). Supply exact keys for "
                "those channels."
            )
        resolved[channel_name] = candidates[0]
        used_original_keys.add(original_key)

    extra = sorted(
        displayed_key
        for displayed_key, original_key, _ in keyed_values
        if original_key not in used_original_keys
    )
    if extra:
        logging.info(
            "Ignoring %d normalization dictionary channel(s) not selected for CellVision: %s",
            len(extra),
            extra,
        )
    normalized: dict[str, float] = {}
    for name in expected:
        matched_key, _, raw_value = resolved[name]
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Normalization value for channel {name!r} (dictionary key "
                f"{matched_key!r}) is not numeric: {raw_value!r}"
            ) from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"Normalization value for channel {name!r} (dictionary key "
                f"{matched_key!r}) must be finite and positive; got {value}."
            )
        normalized[name] = value
        if matched_key != name:
            logging.info(
                "Matched CellVision channel %r to normalization dictionary key %r.",
                name,
                matched_key,
            )
    return normalized


def load_normalization_dict(
    path: Path,
    *,
    channel_names: Sequence[str],
) -> dict[str, float]:
    """Load and validate the marker-to-value JSON format produced by Nimbus."""
    if not path.is_file():
        raise FileNotFoundError(f"CellVision normalization dictionary does not exist: {path}")
    return validate_normalization_dict(read_json(path), channel_names=channel_names)


def compute_normalization_dict(
    contexts: Sequence[ROIInput],
    *,
    quantile: float,
    minimum_value: float,
    mask_expand_px: int,
) -> dict[str, float]:
    """Compute Nimbus-compatible mean per-ROI, in-mask channel quantiles."""
    from skimage.segmentation import expand_labels
    from tifffile import imread

    if not contexts:
        raise ValueError("Cannot compute CellVision normalization without ROI inputs.")
    if not 0 < float(quantile) <= 1:
        raise ValueError("CellVision normalization quantile must lie in (0, 1].")
    if not np.isfinite(minimum_value) or float(minimum_value) <= 0:
        raise ValueError("CellVision normalization minimum must be finite and positive.")
    per_channel: dict[str, list[float]] = {
        str(name): [] for name in contexts[0].channel_names
    }
    for context in contexts:
        raw_mask = np.asarray(np.squeeze(imread(context.mask_path)))
        if raw_mask.shape != context.spatial_shape:
            raise ValueError(
                f"Mask for ROI {context.name!r} has shape {raw_mask.shape}; "
                f"expected {context.spatial_shape}."
            )
        if mask_expand_px:
            raw_mask = expand_labels(raw_mask.astype(np.int64), distance=int(mask_expand_px))
        mask_bool = raw_mask > 0
        if not np.any(mask_bool):
            continue
        for channel_index, channel_name in enumerate(context.channel_names):
            image = _validate_channel_image(
                imread(context.channel_files[channel_index]),
                path=context.channel_files[channel_index],
            )
            finite = image[mask_bool]
            finite = finite[np.isfinite(finite)]
            if finite.size:
                per_channel[str(channel_name)].append(
                    float(np.quantile(finite, float(quantile)))
                )

    result: dict[str, float] = {}
    for channel_name, roi_values in per_channel.items():
        if roi_values:
            value = float(np.mean(roi_values))
            result[channel_name] = max(value, float(minimum_value))
        else:
            logging.warning(
                "No finite in-mask pixels were available for channel %s; using %.3g.",
                channel_name,
                minimum_value,
            )
            result[channel_name] = float(minimum_value)
    return validate_normalization_dict(result, channel_names=list(per_channel))


def write_normalization_dict(path: Path, values: Mapping[str, float]) -> None:
    """Write Nimbus-compatible normalization JSON with string-valued scalars."""
    write_json(path, {str(key): str(float(value)) for key, value in values.items()})


def _normalized_uint16_image(
    array: np.ndarray,
    *,
    path: Path,
    normalization_value: float,
    clip_values: Sequence[float],
) -> np.ndarray:
    """Normalize raw intensity data to [0, 1] and encode it losslessly for scPortrait."""
    image = _validate_channel_image(array, path=path)
    if not np.isfinite(normalization_value) or float(normalization_value) <= 0:
        raise ValueError(f"Invalid normalization value for {path}: {normalization_value}")
    if len(clip_values) != 2:
        raise ValueError("CellVision normalization clip requires two values.")
    lower, upper = (float(value) for value in clip_values)
    if not 0 <= lower < upper <= 1:
        raise ValueError("CellVision normalization clip must satisfy 0 <= lower < upper <= 1.")
    normalized = np.clip(image / float(normalization_value), lower, upper)
    return np.rint(normalized * np.iinfo(np.uint16).max).astype(np.uint16)


def assemble_scportrait_inputs(
    contexts: Sequence[ROIInput],
    identity: pd.DataFrame,
    *,
    roi_obs: str,
    object_id_obs: str,
    assembled_folder: Path,
    image_size: int,
    mask_expand_px: int,
    normalization_values: Mapping[str, float],
    normalization_clip: Sequence[float],
) -> tuple[list[Path], np.ndarray]:
    """Assemble selected ROIs into the padded single-mask scPortrait workflow."""
    from skimage.segmentation import expand_labels
    from tifffile import imread, imwrite

    if assembled_folder.exists():
        shutil.rmtree(assembled_folder)
    assembled_folder.mkdir(parents=True, exist_ok=True)
    padding = int(image_size)
    canvas_width = max(context.spatial_shape[1] for context in contexts) + 2 * padding
    canvas_height = sum(context.spatial_shape[0] + 2 * padding for context in contexts)
    offsets: dict[str, tuple[int, int, int, int]] = {}
    cursor = 0
    for context in contexts:
        y0 = cursor + padding
        y1 = y0 + context.spatial_shape[0]
        x0 = padding
        x1 = x0 + context.spatial_shape[1]
        offsets[context.name] = (y0, y1, x0, x1)
        cursor += context.spatial_shape[0] + 2 * padding

    channel_paths: list[Path] = []
    for channel_index, channel_name in enumerate(contexts[0].channel_names):
        canvas: NDArray[np.uint16] = np.zeros(
            (canvas_height, canvas_width), dtype=np.uint16
        )
        for context in contexts:
            y0, y1, x0, x1 = offsets[context.name]
            image = _normalized_uint16_image(
                imread(context.channel_files[channel_index]),
                path=context.channel_files[channel_index],
                normalization_value=float(normalization_values[channel_name]),
                clip_values=normalization_clip,
            )
            if image.shape != context.spatial_shape:
                raise ValueError(
                    f"Channel {channel_name!r} in ROI {context.name!r} has shape {image.shape}; "
                    f"expected {context.spatial_shape}."
                )
            canvas[y0:y1, x0:x1] = image
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", channel_name).strip("._") or f"channel_{channel_index}"
        output = assembled_folder / f"{channel_index:03d}_{safe_name}.tif"
        imwrite(output, canvas)
        channel_paths.append(output)

    combined_mask: NDArray[np.uint64] = np.zeros(
        (canvas_height, canvas_width), dtype=np.uint64
    )
    for context in contexts:
        y0, y1, x0, x1 = offsets[context.name]
        raw_mask = np.asarray(np.squeeze(imread(context.mask_path)))
        if raw_mask.shape != context.spatial_shape:
            raise ValueError(
                f"Mask for ROI {context.name!r} has shape {raw_mask.shape}; "
                f"expected {context.spatial_shape}."
            )
        if mask_expand_px:
            raw_mask = expand_labels(raw_mask.astype(np.int64), distance=int(mask_expand_px))
        selected = identity[identity[roi_obs].astype(str).eq(context.name)]
        combined_mask[y0:y1, x0:x1] = relabel_selected_mask(
            raw_mask,
            selected,
            object_id_obs=object_id_obs,
        )
    return channel_paths, combined_mask


def write_scportrait_config(
    path: Path,
    *,
    image_size: int,
    threads: int,
) -> None:
    """Write the generated scPortrait adapter config from typed CellVision values."""
    import yaml

    cache_path = path.parent / "scportrait_cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "CellVision IMC extraction",
        "HDF5CellExtraction": {
            "threads": int(threads),
            "image_size": int(image_size),
            # CellVision has already normalized every assembled marker to [0, 1].
            # False makes scPortrait perform only its fixed uint16 -> float conversion.
            "normalize_output": False,
            "compression": "lzf",
            "cache": str(cache_path),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run_scportrait_extraction(
    *,
    project_folder: Path,
    config_path: Path,
    channel_paths: Sequence[Path],
    channel_names: Sequence[str],
    combined_mask: np.ndarray,
    gaussian_blur_mask: bool,
    overwrite: bool,
) -> Path:
    """Run scPortrait's HDF5CellExtraction on preassembled IMC inputs."""
    try:
        import scportrait.pipeline.extraction as extraction_module
        from scportrait.pipeline.extraction import HDF5CellExtraction
        from scportrait.pipeline.project import Project
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "CellVision extraction requires scPortrait. Activate the registered "
            "'scPortrait' environment."
        ) from exc

    cache_root = Path(tempfile.mkdtemp(prefix="cellvision_scp_"))
    project = Project(
        project_location=str(project_folder),
        config_path=str(config_path),
        extraction_f=HDF5CellExtraction,
        overwrite=bool(overwrite),
        debug=False,
    )
    try:
        project.load_input_from_tif_files(
            [str(path) for path in channel_paths],
            channel_names=[str(name) for name in channel_names],
            overwrite=True,
            cache=str(cache_root),
        )
        project.filehandler._write_segmentation_sdata(
            np.asarray(combined_mask),
            project.cyto_seg_name,
            chunks=project.DEFAULT_CHUNK_SIZE_2D,
            overwrite=True,
        )
        project.filehandler._add_centers(project.cyto_seg_name, overwrite=True)
        project.extraction_f.register_parameter("segmentation_mask", project.cyto_seg_name)
        original_gaussian = extraction_module.gaussian
        if not gaussian_blur_mask:
            # scPortrait 1.5.x hard-codes sigma=1. Linux fork workers inherit
            # this adapter; Windows is forced to single-process by scPortrait.
            extraction_module.gaussian = lambda image, **_kwargs: np.asarray(
                image, dtype=np.float32
            )
        try:
            project.extract(overwrite=bool(overwrite))
        finally:
            extraction_module.gaussian = original_gaussian
        output = Path(project.extraction_f.output_path)
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)
    if not output.is_file():
        raise RuntimeError(f"scPortrait completed without creating the expected H5SC file: {output}")
    return output.resolve(strict=True)


def _read_h5sc_metadata(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, tuple[int, ...]]:
    try:
        try:
            from scportrait.io import read_h5sc
        except ImportError:
            from scportrait.io.h5sc import read_h5sc
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError("Reading CellVision H5SC assets requires scPortrait.") from exc

    h5sc = read_h5sc(path)
    images = h5sc.obsm["single_cell_images"]
    shape = tuple(int(value) for value in images.shape)
    obs = h5sc.obs.copy()
    var = h5sc.var.copy()
    file_handle = getattr(images, "file", None)
    if file_handle is not None:
        file_handle.close()
    return obs, var, shape


def read_h5sc_metadata(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, tuple[int, ...]]:
    """Read H5SC row/channel metadata without retaining the image file handle."""
    if not path.is_file():
        raise FileNotFoundError(f"CellVision H5SC file does not exist: {path}")
    return _read_h5sc_metadata(path)


def _replace_h5sc_obs(path: Path, obs: pd.DataFrame) -> None:
    """Replace only the AnnData obs group while preserving the backed image tensor."""
    import anndata as ad
    import h5py

    descriptor, temporary_name = tempfile.mkstemp(prefix="cellvision_obs_", suffix=".h5ad")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        ad.AnnData(obs=obs).write_h5ad(temporary)
        with h5py.File(path, "r+") as destination, h5py.File(temporary, "r") as source:
            if "obs" in destination:
                del destination["obs"]
            source.copy("obs", destination)
    finally:
        temporary.unlink(missing_ok=True)


def annotate_h5sc_identity(
    path: Path,
    identity: pd.DataFrame,
    *,
    roi_obs: str,
    object_id_obs: str,
    population_obs: str | None,
) -> pd.DataFrame:
    """Attach source identities to H5SC rows and mark non-extracted requested cells."""
    obs, _var, _shape = _read_h5sc_metadata(path)
    id_candidates = [
        column
        for column in ("scportrait_cell_id", "cell_id", "CellID")
        if column in obs.columns
    ]
    if not id_candidates:
        raise KeyError(
            "scPortrait H5SC obs does not contain a recognized cell-ID column "
            "('scportrait_cell_id' or legacy 'cell_id')."
        )
    id_column = id_candidates[0]
    extracted_ids = pd.to_numeric(obs[id_column], errors="raise").astype(np.uint64)
    if extracted_ids.duplicated().any():
        raise ValueError("scPortrait H5SC contains duplicate extracted cell IDs.")

    mapping_columns = [
        "cellvision_id",
        "scportrait_cell_id",
        "source_obs_id",
        "source_obs_position",
        roi_obs,
        object_id_obs,
    ]
    if population_obs is not None:
        mapping_columns.append(population_obs)
    mapping = identity[mapping_columns].copy().set_index("scportrait_cell_id", drop=False)
    unknown = [int(value) for value in extracted_ids if int(value) not in mapping.index]
    if unknown:
        raise ValueError(
            "scPortrait returned cell IDs not present in the CellVision identity map; "
            f"examples: {unknown[:10]}"
        )
    extracted_mapping = mapping.loc[extracted_ids.to_numpy()].reset_index(drop=True)
    if len(extracted_mapping) != len(obs):
        raise RuntimeError("H5SC identity join changed the number of extracted rows.")

    annotated = obs.reset_index(drop=True).copy()
    for column in extracted_mapping.columns:
        annotated[column] = extracted_mapping[column].to_numpy()
    annotated.index = pd.Index(annotated["source_obs_id"].astype(str), name="source_obs_id")
    if not annotated.index.is_unique:
        raise ValueError("Annotated H5SC source observation IDs are not unique.")
    _replace_h5sc_obs(path, annotated)

    result = identity.copy()
    extracted_set = set(int(value) for value in extracted_ids)
    result["extraction_status"] = np.where(
        result["scportrait_cell_id"].astype(int).isin(extracted_set),
        "extracted",
        "not_extracted_by_scportrait",
    )
    return result


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a small reusable JSON metadata record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON mapping with a clear validation error."""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON mapping in {path}")
    return loaded


def validate_existing_extraction(
    paths: CellVisionPaths,
    *,
    expected_fingerprint: str,
) -> dict[str, Any]:
    """Validate a reusable extraction before a non-overwriting stage reuses it."""
    missing = [
        path
        for path in (
            paths.h5sc,
            paths.identity,
            paths.extraction_metadata,
            paths.normalization_dict,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileExistsError(
            f"CellVision asset folder already exists but extraction assets are incomplete: {missing}. "
            "Set cellvision.overwrite=true to rebuild it."
        )
    metadata = read_json(paths.extraction_metadata)
    observed = str(metadata.get("identity_fingerprint", ""))
    if observed != expected_fingerprint:
        raise ValueError(
            "Existing CellVision H5SC selection/marker fingerprint does not match the current "
            "configuration. Set cellvision.overwrite=true to rebuild reusable assets."
        )
    obs, _var, shape = read_h5sc_metadata(paths.h5sc)
    if len(obs) != int(metadata.get("n_extracted_cells", -1)) or len(shape) != 4:
        raise ValueError("Existing CellVision H5SC failed row-count or image-shape validation.")
    identity = pd.read_csv(paths.identity)
    required_columns = {
        "source_obs_id",
        "scportrait_cell_id",
        "extraction_status",
    }
    missing_columns = sorted(required_columns.difference(identity.columns))
    if missing_columns:
        raise ValueError(
            f"Existing CellVision identity table lacks required columns: {missing_columns}"
        )
    if len(identity) != int(metadata.get("n_requested_cells", -1)):
        raise ValueError("Existing CellVision identity table failed requested-row validation.")
    extracted_identity = identity.loc[
        identity["extraction_status"].eq("extracted"), "source_obs_id"
    ].astype(str)
    if extracted_identity.tolist() != obs.index.astype(str).tolist():
        raise ValueError(
            "Existing CellVision identity table and H5SC observation order do not match."
        )
    return metadata


def image_channel_metadata(var: pd.DataFrame, image_shape: Sequence[int]) -> tuple[list[int], list[str]]:
    """Return the H5SC indices/names of image channels, excluding segmentation masks."""
    n_channels = int(image_shape[1])
    if len(var) != n_channels:
        raise ValueError(
            f"H5SC var has {len(var)} rows but the image tensor has {n_channels} channels."
        )
    names = (
        var["channels"].astype(str).tolist()
        if "channels" in var.columns
        else var.index.astype(str).tolist()
    )
    if "channel_mapping" in var.columns:
        indices = [
            index
            for index, value in enumerate(var["channel_mapping"].astype(str))
            if value == "image_channel"
        ]
    else:
        # Older scPortrait files conventionally put a segmentation channel first.
        indices = list(range(1, n_channels))
        logging.warning(
            "H5SC var lacks channel_mapping; treating the first channel as a mask and the remainder as images."
        )
    if not indices:
        raise ValueError("H5SC contains no channels mapped as image_channel.")
    return indices, [names[index] for index in indices]


def mask_channel_index(var: pd.DataFrame, image_shape: Sequence[int]) -> int:
    """Return the unique H5SC segmentation-mask channel index."""
    n_channels = int(image_shape[1])
    if len(var) != n_channels:
        raise ValueError(
            f"H5SC var has {len(var)} rows but the image tensor has {n_channels} channels."
        )
    if "channel_mapping" not in var.columns:
        logging.warning("H5SC var lacks channel_mapping; treating the first channel as the mask.")
        return 0
    indices = [
        index
        for index, value in enumerate(var["channel_mapping"].astype(str))
        if value == "mask"
    ]
    if len(indices) != 1:
        raise ValueError(f"Expected exactly one H5SC mask channel, found indices {indices}.")
    return indices[0]


def categorical_palette(categories: Sequence[str]) -> dict[str, Any]:
    """Create a deterministic matplotlib palette for arbitrary category counts."""
    import matplotlib.pyplot as plt

    labels = [str(value) for value in categories]
    cmap = plt.get_cmap("tab20" if len(labels) <= 20 else "gist_ncar")
    denominator = max(1, len(labels))
    return {label: cmap(index / denominator) for index, label in enumerate(labels)}


def plot_categorical_embedding(
    coordinates: np.ndarray,
    labels: Sequence[Any],
    *,
    title: str,
    output_path: Path,
    dpi: int,
    background_coordinates: np.ndarray | None = None,
) -> Path:
    """Plot categorical labels on a 2D embedding with an optional grey background."""
    import matplotlib.pyplot as plt

    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(f"Expected 2D embedding coordinates, got {coordinates.shape}")
    label_series = pd.Series(labels, dtype="string")
    if len(label_series) != len(coordinates):
        raise ValueError("Embedding coordinates and categorical labels must have equal lengths.")
    categories = sorted(label_series.dropna().astype(str).unique().tolist())
    palette = categorical_palette(categories)
    fig, ax = plt.subplots(figsize=(8, 7))
    if background_coordinates is not None:
        background = np.asarray(background_coordinates)
        ax.scatter(background[:, 0], background[:, 1], s=2, c="#d6d6d6", alpha=0.45, linewidths=0)
    for category in categories:
        selected = label_series.astype(str).eq(category).to_numpy()
        ax.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=5,
            color=palette[category],
            alpha=0.8,
            linewidths=0,
            label=category,
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    if categories:
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
            markerscale=2,
            fontsize=max(6, 9 - len(categories) // 20),
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def confusion_tables(
    original_labels: Sequence[Any],
    learned_labels: Sequence[Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return count and row-normalized original-vs-CellVision contingency tables."""
    counts = pd.crosstab(
        pd.Series(original_labels, name="original_population", dtype="string"),
        pd.Series(learned_labels, name="cellvision_cluster", dtype="string"),
        dropna=False,
    )
    denominators = counts.sum(axis=1).replace(0, np.nan)
    normalized = counts.div(denominators, axis=0).fillna(0.0)
    return counts, normalized


def plot_confusion_matrix(
    values: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    dpi: int,
    colorbar_label: str,
) -> Path:
    """Plot a contingency matrix without requiring seaborn or scikit-learn."""
    import matplotlib.pyplot as plt

    width = max(7.0, 0.45 * max(1, values.shape[1]) + 4)
    height = max(5.0, 0.4 * max(1, values.shape[0]) + 2)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(values.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(values.shape[1]), labels=values.columns.astype(str), rotation=90)
    ax.set_yticks(np.arange(values.shape[0]), labels=values.index.astype(str))
    ax.set_xlabel("CellVision Leiden cluster")
    ax.set_ylabel("Original population")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def _gallery_image(image: np.ndarray, *, cell_id: str, channel_name: str) -> np.ndarray:
    """Return stored unit-range pixels unchanged for fixed-scale gallery display."""
    values = np.asarray(image, dtype=np.float32)
    if (
        not np.all(np.isfinite(values))
        or values.min(initial=0) < 0
        or values.max(initial=0) > 1
    ):
        raise ValueError(
            "CellVision galleries require stored H5SC images in [0, 1]; "
            f"cell={cell_id!r}, channel={channel_name!r}."
        )
    return values


def plot_cell_gallery(
    images: Any,
    *,
    row_indices: Sequence[int],
    cell_ids: Sequence[str],
    channel_indices: Sequence[int],
    channel_names: Sequence[str],
    title: str,
    output_path: Path,
    dpi: int,
) -> Path:
    """Plot cells as rows and trained image channels as columns from the H5SC tensor."""
    import matplotlib.pyplot as plt

    if len(row_indices) != len(cell_ids):
        raise ValueError("Gallery row indices and cell IDs must have equal lengths.")
    if not row_indices:
        raise ValueError("Cannot plot an empty CellVision gallery.")
    add_composite = len(channel_indices) <= 3
    n_columns = len(channel_indices) + int(add_composite)
    fig, axes = plt.subplots(
        len(row_indices),
        n_columns,
        figsize=(max(3.0, n_columns * 1.8), max(2.0, len(row_indices) * 1.55)),
        squeeze=False,
    )
    for row, (image_index, cell_id) in enumerate(zip(row_indices, cell_ids, strict=True)):
        channel_images: list[np.ndarray] = []
        for column, (channel_index, channel_name) in enumerate(
            zip(channel_indices, channel_names, strict=True)
        ):
            image = _gallery_image(
                images[int(image_index), int(channel_index)],
                cell_id=str(cell_id),
                channel_name=str(channel_name),
            )
            channel_images.append(image)
            axes[row, column].imshow(image, cmap="gray", vmin=0, vmax=1)
            if row == 0:
                axes[row, column].set_title(str(channel_name), fontsize=9)
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(str(cell_id), fontsize=7, rotation=0, ha="right", va="center")
        if add_composite:
            if len(channel_images) == 1:
                composite = np.repeat(channel_images[0][..., None], 3, axis=2)
            else:
                components = channel_images + [np.zeros_like(channel_images[0])] * (3 - len(channel_images))
                composite = np.stack(components[:3], axis=2)
            axes[row, -1].imshow(np.clip(composite, 0, 1))
            if row == 0:
                axes[row, -1].set_title("Composite", fontsize=9)
            axes[row, -1].axis("off")
    fig.suptitle(title, y=1.005)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def open_h5sc_images(path: Path) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    """Open an H5SC tensor for plotting; the caller must close ``images.file``."""
    try:
        try:
            from scportrait.io import read_h5sc
        except ImportError:
            from scportrait.io.h5sc import read_h5sc
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError("CellVision plotting requires scPortrait to read H5SC images.") from exc
    h5sc = read_h5sc(path)
    return h5sc.obsm["single_cell_images"], h5sc.obs.copy(), h5sc.var.copy()


def safe_slug(value: Any) -> str:
    """Return a compact filesystem-safe label for report artifacts."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return text or "value"


__all__ = [
    "CLUSTERED_FILENAME",
    "CellVisionPaths",
    "EMBEDDINGS_FILENAME",
    "EXTRACTION_METADATA_FILENAME",
    "H5SC_FILENAME",
    "IDENTITY_FILENAME",
    "MODEL_FILENAME",
    "ROIInput",
    "TRAINING_HISTORY_FILENAME",
    "annotate_h5sc_identity",
    "assemble_scportrait_inputs",
    "categorical_palette",
    "compute_normalization_dict",
    "configuration_fingerprint",
    "confusion_tables",
    "discover_roi_inputs",
    "identity_fingerprint",
    "image_channel_metadata",
    "input_file_manifest",
    "leiden_key",
    "load_normalization_dict",
    "mask_channel_index",
    "open_h5sc_images",
    "plot_categorical_embedding",
    "plot_cell_gallery",
    "plot_confusion_matrix",
    "read_h5sc_metadata",
    "read_json",
    "relabel_selected_mask",
    "resolution_label",
    "resolve_cellvision_paths",
    "resolve_roi_channels",
    "run_scportrait_extraction",
    "safe_slug",
    "select_source_cells",
    "validate_normalization_dict",
    "validate_existing_extraction",
    "write_json",
    "write_normalization_dict",
    "write_scportrait_config",
]
