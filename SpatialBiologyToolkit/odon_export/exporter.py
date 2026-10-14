"""Experimental Odon interchange assets, independent of the SBT stage registry.

Writes NGFF 0.4 / Zarr v2 uint16 images and WKB GeoParquet objects. Scientific
measurements are copied from AnnData, never recalculated from display imagery.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Sequence

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tifffile
import zarr
from numcodecs import Zlib
from anndata.io import read_elem
from scipy import sparse
from shapely.geometry import box
from shapely.ops import unary_union
from skimage.measure import regionprops

from SpatialBiologyToolkit.pipeline.manifests import utc_now
from SpatialBiologyToolkit.qc_classifier.io import (
    build_image_channel_aliases,
    discover_roi_image_index,
)

LOG = logging.getLogger(__name__)
ODON_REFERENCE = "b01faef010b14ea03c33e92a14e438061d257e39"


@dataclass(frozen=True)
class ImageSource:
    name: str
    path: Path
    plane: int | None
    shape: tuple[int, int]


@dataclass(frozen=True)
class RoiPlan:
    roi: str
    rows: np.ndarray
    object_ids: np.ndarray
    mask: Path
    images: tuple[ImageSource, ...]


def _names(values: Sequence[str] | None, label: str) -> list[str] | None:
    if values is None:
        return None
    names = [str(value) for value in values]
    if (
        not names
        or any(not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(f"{label} must contain unique, nonempty names.")
    return names


def _tiff_sources(
    path: Path, name: str, stack_channels: list[str] | None
) -> list[ImageSource]:
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Only TIFF images are supported in this trial: {path}")
    with tifffile.TiffFile(path) as tif:
        if len(tif.series) != 1:
            raise ValueError(f"Expected one image series: {path}")
        shape = tif.series[0].shape
        axes = tif.series[0].axes
        if tif.series[0].dtype.kind not in "uifb":
            raise ValueError(
                f"Unsupported image dtype in {path}: {tif.series[0].dtype}"
            )
    if len(shape) == 2:
        return [ImageSource(name, path, None, tuple(shape))]
    if len(shape) == 3 and stack_channels and axes.endswith("YX"):
        if shape[0] != len(stack_channels):
            raise ValueError(f"Stack channel count does not match {path}: {shape}")
        return [
            ImageSource(channel, path, index, tuple(shape[1:]))
            for index, channel in enumerate(stack_channels)
        ]
    raise ValueError(
        f"Expected a 2-D channel TIFF, or a CYX stack with explicit stack_channels; "
        f"found shape={shape}, axes={axes} in {path}. RGB/ZYX/TCZYX are not inferred."
    )


def _plan(
    adata: Any,
    masks: Path,
    images: list[Path],
    roi_key: str,
    object_id_key: str,
    rois: list[str] | None,
    channels: list[str] | None,
    stack_channels: list[str] | None,
) -> list[RoiPlan]:
    for column in (roi_key, object_id_key):
        if column not in adata.obs:
            raise ValueError(f"AnnData.obs is missing {column!r}.")
    if adata.obs[roi_key].isna().any():
        raise ValueError("AnnData contains missing ROI identities.")
    roi_values = adata.obs[roi_key].astype(str)
    available = list(dict.fromkeys(roi_values))
    selected = available if rois is None else rois
    if not selected:
        raise ValueError("No ROIs to export.")
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"ROIs absent from AnnData: {sorted(unknown)}")
    for roi in selected:
        if not roi.strip() or roi in {".", ".."} or any(c in roi for c in "/\\:"):
            raise ValueError(
                f"ROI must be a single safe input filename component: {roi!r}"
            )
    aliases = build_image_channel_aliases(adata.var_names, adata.var)
    indexed = discover_roi_image_index(images, selected, channel_aliases=aliases)
    mask_index: dict[str, list[Path]] = {}
    for path in masks.iterdir():
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
            mask_index.setdefault(path.stem, []).append(path)
    plans = []
    for roi in selected:
        rows = np.flatnonzero(roi_values.to_numpy() == roi)
        values = pd.to_numeric(adata.obs.iloc[rows][object_id_key], errors="raise")
        # int64 identities remain exact; floats above 2**53 cannot be trusted.
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{roi}: object IDs must be finite positive integers.")
        if (values <= 0).any() or (values >= 2**63).any() or (values % 1 != 0).any():
            raise ValueError(f"{roi}: object IDs must be positive int64 integers.")
        if pd.api.types.is_float_dtype(values) and (values > 2**53).any():
            raise ValueError(
                f"{roi}: large floating-point IDs are not exact; supply integer IDs."
            )
        object_ids = values.to_numpy(dtype=np.int64)
        if len(np.unique(object_ids)) != len(object_ids):
            raise ValueError(f"{roi}: duplicate (ROI, ObjectNumber) identities.")
        candidates = mask_index.get(roi, [])
        if len(candidates) != 1:
            raise ValueError(
                f"{roi}: expected exactly one mask TIFF; found {len(candidates)}."
            )
        mask_path = candidates[0]
        with tifffile.TiffFile(mask_path) as tif:
            series = tif.series[0]
            if (
                len(tif.series) != 1
                or len(series.shape) != 2
                or series.dtype.kind not in "ui"
            ):
                raise ValueError(f"{roi}: mask must be a single 2-D integer TIFF.")
            mask_shape = series.shape
        sources = []
        for name, path in indexed[roi].items():
            # Explicit channel selection may exclude unrelated RGB/overview files.
            if channels is not None and name not in channels and stack_channels is None:
                continue
            sources.extend(_tiff_sources(path, name, stack_channels))
        by_name = {source.name: source for source in sources}
        if len(by_name) != len(sources):
            raise ValueError(
                f"{roi}: duplicate image channel names after stack expansion."
            )
        names = channels if channels is not None else list(by_name)
        if not names:
            raise ValueError(f"{roi}: no image channels found.")
        if set(names) - set(by_name):
            raise ValueError(
                f"{roi}: missing image channels: {sorted(set(names) - set(by_name))}"
            )
        sources = [by_name[name] for name in names]
        for source in sources:
            if source.shape != mask_shape:
                raise ValueError(
                    f"{roi}/{source.name}: image {source.shape} and mask {mask_shape} differ. "
                    "Inputs must already be registered; no resizing is performed."
                )
        plans.append(RoiPlan(roi, rows, object_ids, mask_path, tuple(sources)))
    return plans


def _encode_image(image: np.ndarray, mode: str) -> tuple[np.ndarray, dict]:
    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinity; repair it before export.")
    low, high = float(image.min()), float(image.max())
    exact = (
        low >= 0
        and high <= 65535
        and (image.dtype.kind in "uib" or bool(np.equal(image, np.floor(image)).all()))
    )
    metadata = {"source_dtype": str(image.dtype), "source_min": low, "source_max": high}
    if exact:
        return image.astype(np.uint16), {
            **metadata,
            "encoding": "identity",
            "scale": 1,
            "offset": 0,
        }
    if mode == "preserve":
        raise ValueError(
            "Image cannot be represented losslessly as uint16. Use intensity_mode='auto' "
            "for a recorded display-only rescaling."
        )
    # Per-ROI/channel full-range affine encoding; never percentile-clips signal.
    scale = (high - low) / 65535 if high != low else 1.0
    encoded = np.rint((image.astype(np.float64) - low) / scale)
    return np.clip(encoded, 0, 65535).astype(np.uint16), {
        **metadata,
        "encoding": "affine_uint16",
        "scale": scale,
        "offset": low,
        "decode": "source_approx = stored * scale + offset",
    }


def _downsample(image: np.ndarray) -> np.ndarray:
    """2x2 area means, including odd edge pixels without zero padding."""
    total = np.zeros(((image.shape[0] + 1) // 2, (image.shape[1] + 1) // 2), np.float64)
    count = np.zeros(total.shape, np.uint8)
    for y in range(2):
        for x in range(2):
            part = image[y::2, x::2]
            total[: part.shape[0], : part.shape[1]] += part
            count[: part.shape[0], : part.shape[1]] += 1
    return np.rint(total / count).astype(np.uint16)


def _source_metadata(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _write_images(
    destination: Path,
    plan: RoiPlan,
    *,
    chunk_size: int,
    levels: int,
    intensity_mode: str,
    provenance: dict,
) -> None:
    # Explicitly write v2 even when the installed zarr Python library is v3.
    zarr3 = int(zarr.__version__.split(".")[0]) >= 3
    format_option = {"zarr_format": 2} if zarr3 else {"zarr_version": 2}
    group = zarr.open_group(str(destination), mode="w", **format_option)
    shape = plan.images[0].shape
    arrays = []
    datasets = []
    for level in range(levels):
        options: dict[str, Any] = {
            "shape": (len(plan.images), *shape),
            "dtype": "uint16",
            "chunks": (1, min(chunk_size, shape[0]), min(chunk_size, shape[1])),
            "compressor": Zlib(level=1),
        }
        if zarr3:
            options["chunk_key_encoding"] = {
                "name": "v2",
                "configuration": {"separator": "/"},
            }
            arrays.append(group.create_array(str(level), **options))
        else:
            arrays.append(
                group.create_dataset(str(level), dimension_separator="/", **options)
            )
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1, 2**level, 2**level]}
                ],
            }
        )
        if shape == (1, 1):
            break
        shape = ((shape[0] + 1) // 2, (shape[1] + 1) // 2)
    channel_metadata, omero_channels = [], []
    cached_stack_path, cached_stack = None, None
    for index, source in enumerate(plan.images):
        if source.plane is None:
            image = tifffile.imread(source.path)
        else:
            if source.path != cached_stack_path:
                cached_stack = tifffile.imread(source.path)
                cached_stack_path = source.path
            assert cached_stack is not None
            image = cached_stack[source.plane]
        encoded, encoding = _encode_image(image, intensity_mode)
        if encoding["encoding"] != "identity":
            LOG.warning(
                "%s/%s: rescaling display image to uint16; measurements unchanged",
                plan.roi,
                source.name,
            )
        window_low, window_high = int(encoded.min()), int(encoded.max())
        if window_low == window_high:
            window_low, window_high = (0, max(1, window_high))
        omero_channels.append(
            {
                "label": source.name,
                "color": "FFFFFF",
                "active": index == 0,
                "window": {
                    "start": window_low,
                    "end": window_high,
                    "min": 0,
                    "max": 65535,
                },
            }
        )
        channel_metadata.append(
            {
                "name": source.name,
                "source": _source_metadata(source.path),
                "source_plane": source.plane,
                **encoding,
            }
        )
        for array in arrays:
            array[index, :, :] = encoded
            if array is not arrays[-1]:
                encoded = _downsample(encoded)
    group.attrs.update(
        {
            "multiscales": [
                {
                    "version": "0.4",
                    "name": plan.roi,
                    "axes": [
                        {"name": "c", "type": "channel"},
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": datasets,
                }
            ],
            "omero": {"channels": omero_channels},
            "sbt_export": {
                **provenance,
                "roi": plan.roi,
                "mask": _source_metadata(plan.mask),
                "channels": channel_metadata,
                "coordinates": "level-0 pixels: x=column, y=row; pixel edges at integers",
            },
        }
    )


def _cell_geometries(mask: np.ndarray, object_ids: np.ndarray) -> dict[int, Any]:
    if mask.dtype.kind not in "ui" or mask.ndim != 2 or (mask < 0).any():
        raise ValueError("Mask must be a 2-D array of nonnegative integer labels.")
    if mask.size and int(mask.max()) >= 2**63:
        raise ValueError("Mask labels must fit int64.")
    labels, inverse = np.unique(mask, return_inverse=True)
    missing = set(object_ids) - set(labels)
    if missing:
        raise ValueError(f"AnnData cell IDs missing from mask: {sorted(missing)[:10]}")
    # Dense temporary labels avoid regionprops allocating by a sparse maximum ID.
    wanted = set(object_ids)
    lookup: np.ndarray = np.zeros(len(labels), np.int32)
    selected = [int(label) for label in labels if label in wanted]
    for dense_id, original_id in enumerate(selected, start=1):
        lookup[np.searchsorted(labels, original_id)] = dense_id
    dense = lookup[inverse].reshape(mask.shape)
    geometries = {}
    for region in regionprops(dense):
        y0, x0, _, _ = region.bbox
        rectangles: list[Any] = []
        for y, row in enumerate(region.image):
            edges = np.diff(np.pad(row.astype(np.int8), (1, 1)))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            rectangles.extend(
                box(x0 + int(start), y0 + y, x0 + int(end), y0 + y + 1)
                for start, end in zip(starts, ends)
            )
        # Exact pixel unions preserve holes, border cells, and disconnected parts.
        geometries[selected[region.label - 1]] = unary_union(rectangles)
    return geometries


def _property_array(series: pd.Series) -> pa.Array:
    # Odon expects primitive Arrow columns, not pandas dictionary categoricals.
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype(object)
    array = pa.array(series, from_pandas=True)
    if pa.types.is_dictionary(array.type):
        array = array.dictionary_decode()
    if pa.types.is_null(array.type):
        return array.cast(pa.string())
    if not (
        pa.types.is_boolean(array.type)
        or pa.types.is_integer(array.type)
        or pa.types.is_floating(array.type)
        or pa.types.is_string(array.type)
        or pa.types.is_large_string(array.type)
    ):
        raise ValueError(
            f"Unsupported obs dtype {series.dtype}; select scalar obs columns."
        )
    return array


def _matrix_info(node: Any) -> tuple[tuple[int, ...], np.dtype, str]:
    if isinstance(node, h5py.Dataset):
        return node.shape, node.dtype, "dense"
    encoding = node.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if encoding not in {"csr_matrix", "csc_matrix"}:
        raise ValueError(f"Unsupported H5AD matrix encoding: {encoding!r}")
    return tuple(node.attrs["shape"]), node["data"].dtype, encoding


def _publish(staging: Path, destination: Path) -> None:
    """Publish only complete data, tolerating transient Windows scanner locks."""
    for attempt in range(6):
        if destination.exists():
            raise FileExistsError(f"Output appeared during export: {destination}")
        try:
            staging.rename(destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 5:
                raise
            time.sleep(0.1 * 2**attempt)


def _measurement_rows(node: Any, rows: np.ndarray) -> np.ndarray:
    """Read standard H5AD dense/CSR/CSC arrays without SciPy private APIs.

    AnnData's backed sparse reader in older versions depends on SciPy internals
    that changed in SciPy 1.18. Construct public SciPy matrices from bounded
    HDF5 slices instead; never materialize the complete expression matrix.
    """
    shape, dtype, encoding = _matrix_info(node)
    if encoding == "dense":
        return np.asarray(node[rows, :])
    result: np.ndarray = np.zeros((len(rows), shape[1]), dtype=dtype)
    if encoding == "csr_matrix":
        # Coalesce consecutive ROI rows into bounded reads (also handles a
        # typical AnnData sorted by ROI without thousands of HDF5 reads).
        breaks = np.flatnonzero(np.diff(rows) != 1) + 1
        for run in np.split(np.arange(len(rows)), breaks):
            for start in range(0, len(run), 4096):
                positions = run[start : start + 4096]
                first, last = int(rows[positions[0]]), int(rows[positions[-1]])
                pointers = node["indptr"][first : last + 2]
                low, high = int(pointers[0]), int(pointers[-1])
                block = sparse.csr_matrix(
                    (node["data"][low:high], node["indices"][low:high], pointers - low),
                    shape=(len(positions), shape[1]),
                )
                result[positions] = block.toarray()
    else:
        # CSC is column-oriented: read one sparse column at a time, retaining
        # only selected rows. No all-cell dense or sparse matrix is allocated.
        pointers = node["indptr"][:]
        for column in range(shape[1]):
            low, high = int(pointers[column]), int(pointers[column + 1])
            indices = node["indices"][low:high]
            selected_positions: np.ndarray = np.asarray(np.searchsorted(rows, indices))
            keep: np.ndarray = selected_positions < len(rows)
            keep[keep] &= rows[selected_positions[keep]] == indices[keep]
            np.add.at(
                result[:, column],
                selected_positions[keep],
                node["data"][low:high][keep],
            )
    return result


def _write_objects(
    destination: Path,
    adata: Any,
    matrix: Any,
    marker_names: list[str],
    plan: RoiPlan,
    obs_columns: list[str],
    provenance: dict,
) -> int:
    mask = tifffile.imread(plan.mask)
    geometries = _cell_geometries(mask, plan.object_ids)
    ordered = [geometries[int(object_id)] for object_id in plan.object_ids]
    columns = {
        "id": pa.array(
            [
                json.dumps([plan.roi, int(i)], ensure_ascii=False)
                for i in plan.object_ids
            ]
        ),
        "ROI": pa.array([plan.roi] * len(plan.rows)),
        "ObjectNumber": pa.array(plan.object_ids),
        "anndata_index": pa.array(adata.obs_names[plan.rows].astype(str)),
        "x_centroid": pa.array([geometry.centroid.x for geometry in ordered]),
        "y_centroid": pa.array([geometry.centroid.y for geometry in ordered]),
        "geometry": pa.array([geometry.wkb for geometry in ordered], type=pa.binary()),
    }
    mapping: dict[str, dict[str, str]] = {"obs": {}, "measurements": {}}
    # Reserve marker names so marker->image binding remains intuitive in Odon.
    reserved = set(columns) | set(marker_names)
    for name in obs_columns:
        output_name = name
        while output_name in reserved:
            output_name = "obs:" + output_name
        reserved.add(output_name)
        try:
            columns[output_name] = _property_array(adata.obs.iloc[plan.rows][name])
        except (pa.ArrowException, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot export obs column {name!r}: {exc}") from exc
        mapping["obs"][name] = output_name
    values = _measurement_rows(matrix, plan.rows)
    for index, name in enumerate(marker_names):
        output_name = name
        while output_name in columns:
            output_name = "measurement:" + output_name
        columns[output_name] = pa.array(values[:, index])
        mapping["measurements"][name] = output_name
    table = pa.table(columns)
    metadata = {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "crs": None,
                "geometry_types": sorted({geometry.geom_type for geometry in ordered}),
            }
        },
    }
    table = table.replace_schema_metadata(
        {
            b"geo": json.dumps(metadata).encode(),
            b"sbt_export": json.dumps({**provenance, "columns": mapping}).encode(),
        }
    )
    pq.write_table(table, destination, compression="zstd", use_dictionary=False)
    return len(plan.rows)


def export_dataset(
    anndata_path: str | Path,
    masks_folder: str | Path,
    images_folders: str | Path | Sequence[str | Path],
    output: str | Path,
    *,
    roi_key: str = "ROI",
    object_id_key: str = "ObjectNumber",
    rois: Sequence[str] | None = None,
    channels: Sequence[str] | None = None,
    stack_channels: Sequence[str] | None = None,
    obs_columns: Sequence[str] | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    levels: int = 5,
    chunk_size: int = 512,
    intensity_mode: Literal["auto", "preserve"] = "auto",
    dry_run: bool = False,
) -> dict:
    """Create a new portable Odon dataset; inputs and existing outputs are untouched.

    ``obs_columns=None`` copies all scalar observations. Measurements come from
    X by default, or an explicit layer / raw.X. Only cells present in AnnData
    are exported. ``rois`` limits the trial. ``channels`` selects image channels;
    it does not subset measurements. ``stack_channels`` explicitly labels CYX
    TIFF planes. Geometry and imagery use level-0 pixel coordinates.

    One ROI's measurements/mask and one channel image are processed at a time
    (a TIFF stack is read as a whole). ``dry_run`` inspects identities, paths and
    TIFF headers, not image pixels or mask coverage. Output is published only
    after all requested ROIs succeed. No project registration is required.
    """
    if not 1 <= levels <= 16 or not 16 <= chunk_size <= 4096:
        raise ValueError("levels must be 1..16 and chunk_size must be 16..4096.")
    if intensity_mode not in {"auto", "preserve"}:
        raise ValueError("intensity_mode must be 'auto' or 'preserve'.")
    if layer is not None and use_raw:
        raise ValueError("Choose layer or use_raw, not both.")
    roi_names = _names(rois, "rois")
    channel_names = _names(channels, "channels")
    stack_names = _names(stack_channels, "stack_channels")
    source = Path(anndata_path).expanduser().resolve()
    masks = Path(masks_folder).expanduser().resolve()
    raw_folders = (
        [images_folders]
        if isinstance(images_folders, (str, Path))
        else list(images_folders)
    )
    folders = [Path(folder).expanduser().resolve() for folder in raw_folders]
    destination = Path(output).expanduser().resolve()
    if (
        not source.is_file()
        or not masks.is_dir()
        or not folders
        or any(not p.is_dir() for p in folders)
    ):
        raise ValueError(
            "Supply an existing .h5ad file, mask folder, and image folder(s)."
        )
    if destination.exists():
        raise FileExistsError(
            f"Output already exists; choose a new directory: {destination}"
        )
    if any(
        destination == root or root in destination.parents for root in [masks, *folders]
    ):
        raise ValueError("Output must be outside mask and image source folders.")
    # read_h5ad(backed='r') can still eagerly read layers, obsm and uns. Open
    # only the annotations and selected backed matrix needed by this adapter.
    handle = h5py.File(source, "r")
    try:
        if "obs" not in handle or "var" not in handle:
            raise ValueError("Expected an AnnData H5AD containing obs and var.")
        obs, var = read_elem(handle["obs"]), read_elem(handle["var"])
        adata = SimpleNamespace(
            obs=obs, var=var, obs_names=obs.index, var_names=var.index
        )
        if not adata.obs.columns.is_unique or not all(
            isinstance(c, str) for c in adata.obs.columns
        ):
            raise ValueError("AnnData.obs requires unique string column names.")
        observations = (
            list(adata.obs.columns) if obs_columns is None else list(obs_columns)
        )
        if len(set(observations)) != len(observations) or set(observations) - set(
            adata.obs.columns
        ):
            raise ValueError("obs_columns contains duplicates or missing columns.")
        matrix_key = (
            "raw/X" if use_raw else f"layers/{layer}" if layer is not None else "X"
        )
        if matrix_key not in handle:
            raise ValueError(f"AnnData measurement source is absent: {matrix_key}")
        marker_names = (
            list(read_elem(handle["raw/var"]).index) if use_raw else list(var.index)
        )
        matrix = handle[matrix_key]
        matrix_shape, matrix_dtype, _ = _matrix_info(matrix)
        if matrix_dtype.kind not in "biuf":
            raise ValueError(
                "Selected expression matrix must contain real numeric values."
            )
        if matrix_shape != (len(obs), len(marker_names)):
            raise ValueError("Expression matrix dimensions do not match obs/var.")
        _names(marker_names, "measurement variables")
        plans = _plan(
            adata,
            masks,
            folders,
            roi_key,
            object_id_key,
            roi_names,
            channel_names,
            stack_names,
        )
        result = {
            "output": str(destination),
            "samplesheet": str(destination / "samplesheet.csv"),
            "roi_count": len(plans),
            "cell_count": sum(len(plan.rows) for plan in plans),
            "measurement_source": "raw.X"
            if use_raw
            else f"layers/{layer}"
            if layer
            else "X",
            "rois": [
                {
                    "roi": plan.roi,
                    "cells": len(plan.rows),
                    "channels": [s.name for s in plan.images],
                }
                for plan in plans
            ],
            "dry_run": dry_run,
        }
        if dry_run:
            return result
        provenance = {
            "adapter": "SpatialBiologyToolkit.odon_export",
            "schema_version": 1,
            "odon_reference_commit": ODON_REFERENCE,
            "created_at": utc_now().isoformat(),
            "anndata": _source_metadata(source),
            "roi_key": roi_key,
            "object_id_key": object_id_key,
            "measurement_source": result["measurement_source"],
            "intensity_mode": intensity_mode,
            "levels_requested": levels,
            "chunk_size": chunk_size,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".sbt-odon-", dir=destination.parent
        ) as temporary:
            staging = Path(temporary) / "dataset"
            staging.mkdir()
            samplesheet = []
            for index, plan in enumerate(plans, start=1):
                LOG.info("Exporting ROI %s (%d/%d)", plan.roi, index, len(plans))
                relative = Path("rois") / f"roi_{index:06d}"
                roi_output = staging / relative
                roi_output.mkdir(parents=True)
                _write_objects(
                    roi_output / "cells.parquet",
                    adata,
                    matrix,
                    marker_names,
                    plan,
                    observations,
                    provenance,
                )
                _write_images(
                    roi_output / "image.ome.zarr",
                    plan,
                    chunk_size=chunk_size,
                    levels=levels,
                    intensity_mode=intensity_mode,
                    provenance=provenance,
                )
                samplesheet.append(
                    {
                        "id": plan.roi,
                        "path": (relative / "image.ome.zarr").as_posix(),
                        "segpath": (relative / "cells.parquet").as_posix(),
                        "cell_count": len(plan.rows),
                    }
                )
            pd.DataFrame(samplesheet).to_csv(staging / "samplesheet.csv", index=False)
            _publish(staging, destination)
        from SpatialBiologyToolkit.reporting import get_active_reporter

        reporter = get_active_reporter()
        if reporter is not None:
            reporter.add_input("odon_source_anndata", source)
            reporter.add_input("odon_source_masks", masks)
            for folder in folders:
                reporter.add_input("odon_source_images", folder)
            reporter.add_asset(
                "odon_dataset", destination, "Experimental Odon viewing derivative"
            )
            reporter.add_metric("odon_exported_cells", result["cell_count"])
            reporter.add_metric("odon_exported_rois", result["roi_count"])
        return result
    finally:
        handle.close()
