"""
Filesystem helpers for the CellPose active-learning QC tool.

These helpers mirror the ROI-centred conventions used by
``napari_imc_explorer``:

* masks live as one image per ROI directly inside a masks folder
* IMC channels usually live inside ``images/<ROI>/`` subfolders
* optional one-image-per-ROI image folders are also tolerated
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import skimage as sk

from SpatialBiologyToolkit._napari_imc_normalization import normalize_imc_image

MASK_EXTENSIONS = (".tif", ".tiff")
IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass
class RoiAssets:
    """Paths known for one ROI."""

    roi: str
    mask_path: Path | None = None
    image_paths: dict[str, Path] | None = None

    @property
    def has_mask(self) -> bool:
        return self.mask_path is not None and self.mask_path.exists()

    @property
    def has_images(self) -> bool:
        return bool(self.image_paths)


def timestamp_utc() -> str:
    """Return an ISO-8601 UTC timestamp safe for metadata."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def timestamp_slug() -> str:
    """Return a compact timestamp suitable for filenames."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalise_path(path: str | Path | None) -> Path | None:
    """Return a ``Path`` for non-empty path-like values."""

    if path is None:
        return None
    path_text = str(path).strip()
    return Path(path_text) if path_text else None


def ensure_output_structure(output_folder: str | Path) -> dict[str, Path]:
    """Create and return the standard QC output subfolders."""

    root = Path(output_folder)
    subfolders = {
        "root": root,
        "labels": root / "labels",
        "scores": root / "scores",
        "masks_cleaned": root / "masks_cleaned",
        "models": root / "models",
        "metadata": root / "metadata",
        "logs": root / "logs",
        "exports": root / "exports",
    }
    for folder in subfolders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return subfolders


def read_json(path: str | Path, default=None):
    """Read JSON, returning ``default`` when the file is absent."""

    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: dict) -> None:
    """Write pretty JSON with stable key ordering."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)


def append_log(output_folder: str | Path, message: str) -> None:
    """Append one timestamped line to the QC session log."""

    folders = ensure_output_structure(output_folder)
    with (folders["logs"] / "qc_session_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp_utc()}\t{message}\n")


def file_fingerprint(path: str | Path | None, max_bytes: int | None = None) -> dict:
    """
    Return path, size, modified time, and SHA256 hash metadata for a file.

    ``max_bytes`` can be used by callers that want a cheaper partial hash, but
    the default is a full-file hash for auditability.
    """

    if path is None:
        return {"path": None, "exists": False}

    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}

    sha256 = hashlib.sha256()
    bytes_read = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if max_bytes is not None and bytes_read >= max_bytes:
                break
            if max_bytes is not None and bytes_read + len(chunk) > max_bytes:
                chunk = chunk[: max_bytes - bytes_read]
            sha256.update(chunk)
            bytes_read += len(chunk)

    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_time": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "sha256": sha256.hexdigest(),
        "hash_bytes": bytes_read,
    }


def infer_mask_extension(
    masks_folder: str | Path, mask_extension: str | None = None
) -> str | None:
    """Infer the extension used by masks in ``masks_folder``."""

    if mask_extension:
        return (
            mask_extension
            if str(mask_extension).startswith(".")
            else f".{mask_extension}"
        )

    folder = Path(masks_folder)
    if not folder.exists():
        return None

    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in MASK_EXTENSIONS:
            return path.suffix
    return None


def discover_mask_files(
    masks_folder: str | Path, mask_extension: str | None = None
) -> dict[str, Path]:
    """Return ``{roi: mask_path}`` for TIFF masks in a folder."""

    folder = Path(masks_folder)
    if not folder.exists():
        return {}

    extension = infer_mask_extension(folder, mask_extension)
    mask_paths = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if extension is not None and path.suffix.lower() != extension.lower():
            continue
        if path.suffix.lower() not in MASK_EXTENSIONS:
            continue
        mask_paths.setdefault(path.stem, path)
    return mask_paths


def resolve_mask_file(masks_folder: str | Path, roi: str) -> Path | None:
    """Resolve one conventionally named ROI mask without scanning its folder."""

    folder = Path(masks_folder)
    roi = str(roi)
    for extension in MASK_EXTENSIONS:
        candidate = folder / f"{roi}{extension}"
        if candidate.is_file():
            return candidate
    return None


def _normalise_image_alias(value) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return None
    return re.sub(r"\W+", "", text).casefold()


def build_image_channel_aliases(
    var_names: Iterable,
    var_metadata: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Map normalized panel aliases to canonical AnnData variable names."""

    logical_names = [str(value) for value in var_names]
    metadata = var_metadata if var_metadata is not None else pd.DataFrame()
    channel_names = (
        metadata["channel_name"].tolist()
        if "channel_name" in metadata and len(metadata) == len(logical_names)
        else [None] * len(logical_names)
    )
    channel_labels = (
        metadata["channel_label"].tolist()
        if "channel_label" in metadata and len(metadata) == len(logical_names)
        else [None] * len(logical_names)
    )
    aliases: dict[str, str] = {}
    ambiguous: set[str] = set()
    for logical_name, channel_name, channel_label in zip(
        logical_names,
        channel_names,
        channel_labels,
        strict=True,
    ):
        candidates = [logical_name, channel_name, channel_label]
        if channel_name is not None and channel_label is not None:
            candidates.append(f"{channel_name}_{channel_label}")
        for candidate in candidates:
            key = _normalise_image_alias(candidate)
            if key is None:
                continue
            existing = aliases.get(key)
            if existing is None:
                aliases[key] = logical_name
            elif existing != logical_name:
                ambiguous.add(key)
    for key in ambiguous:
        aliases.pop(key, None)
    return aliases


def _image_channel_candidates(image_path: Path, roi: str) -> list[str]:
    stem = image_path.stem
    candidates: list[str] = []
    prefix = f"{roi}_"
    if stem.casefold().startswith(prefix.casefold()):
        candidates.append(stem[len(prefix) :])
    parts = stem.split("_", 3)
    if len(parts) == 4:
        candidates.extend((parts[3], parts[2], f"{parts[2]}_{parts[3]}"))
    elif len(parts) >= 3:
        candidates.append(parts[2])
    candidates.append(stem)
    return list(dict.fromkeys(value for value in candidates if value))


def _image_channel_name(
    image_path: Path,
    roi: str,
    channel_aliases: Mapping[str, str] | None = None,
) -> str:
    """
    Return a concise channel/display name for an ROI image path.

    This follows the same broad convention as ``napari_imc_explorer``:
    filenames like ``ROI_channel_marker.tiff`` use the later stem parts when
    possible, while simple filenames use the stem.
    """

    candidates = _image_channel_candidates(image_path, roi)
    if channel_aliases:
        for candidate in candidates:
            logical_name = channel_aliases.get(_normalise_image_alias(candidate) or "")
            if logical_name is not None:
                return logical_name
    return candidates[0]


def _add_discovered_image(
    image_paths: dict[str, Path],
    name: str,
    path: Path,
    source_name: str,
) -> None:
    if name not in image_paths:
        image_paths[name] = path
        return
    if image_paths[name] == path:
        return
    candidate = f"{name} [{source_name}]"
    suffix = 2
    while candidate in image_paths:
        candidate = f"{name} [{source_name} #{suffix}]"
        suffix += 1
    image_paths[candidate] = path


def discover_roi_images(
    image_folders: str | Path | Iterable[str | Path],
    roi: str,
    *,
    channel_aliases: Mapping[str, str] | None = None,
    scan_flat_folder: bool = True,
) -> dict[str, Path]:
    """
    Return display-name to image-path mappings for one ROI.

    Supports ``images/<ROI>/<channel>.tiff``, direct ``images/<ROI>.tiff``,
    and flat ``images/<ROI>_<channel>.tiff`` layouts. All folders are merged;
    duplicate logical names are retained with a source suffix.
    """

    if isinstance(image_folders, (str, Path)):
        image_folders = [image_folders]

    roi = str(roi)
    image_paths: dict[str, Path] = {}
    for folder in image_folders:
        folder = Path(folder)
        roi_folder = folder / roi
        if roi_folder.exists() and roi_folder.is_dir():
            for path in sorted(roi_folder.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    _add_discovered_image(
                        image_paths,
                        _image_channel_name(path, roi, channel_aliases),
                        path,
                        folder.name or str(folder),
                    )

        if not folder.exists():
            continue
        if not scan_flat_folder:
            for extension in IMAGE_EXTENSIONS:
                path = folder / f"{roi}{extension}"
                if path.is_file():
                    _add_discovered_image(
                        image_paths,
                        folder.name or path.stem,
                        path,
                        folder.name or str(folder),
                    )
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if path.stem == roi:
                _add_discovered_image(
                    image_paths,
                    folder.name or path.stem,
                    path,
                    folder.name or str(folder),
                )
            elif path.stem.casefold().startswith(f"{roi}_".casefold()):
                _add_discovered_image(
                    image_paths,
                    _image_channel_name(path, roi, channel_aliases),
                    path,
                    folder.name or str(folder),
                )

    return image_paths


def discover_roi_image_index(
    image_folders: str | Path | Iterable[str | Path],
    rois: Iterable[str],
    *,
    channel_aliases: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Path]]:
    """Scan configured image folders once and index paths for known ROIs."""

    if isinstance(image_folders, (str, Path)):
        image_folders = [image_folders]
    roi_names = list(dict.fromkeys(str(roi) for roi in rois))
    roi_set = set(roi_names)
    image_index: dict[str, dict[str, Path]] = {roi: {} for roi in roi_names}

    for raw_folder in image_folders:
        folder = Path(raw_folder)
        if not folder.is_dir():
            continue
        source_name = folder.name or str(folder)
        for entry in sorted(folder.iterdir()):
            if entry.is_dir() and entry.name in roi_set:
                roi = entry.name
                for path in sorted(entry.iterdir()):
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                        _add_discovered_image(
                            image_index[roi],
                            _image_channel_name(path, roi, channel_aliases),
                            path,
                            source_name,
                        )
                continue
            if not entry.is_file() or entry.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stem = entry.stem
            roi = stem if stem in roi_set else None
            if roi is None:
                candidates = [
                    stem[:index]
                    for index, character in enumerate(stem)
                    if character == "_" and stem[:index] in roi_set
                ]
                if candidates:
                    roi = max(candidates, key=len)
            if roi is None:
                continue
            name = (
                source_name
                if stem == roi
                else _image_channel_name(entry, roi, channel_aliases)
            )
            _add_discovered_image(
                image_index[roi],
                name,
                entry,
                source_name,
            )
    return image_index


def discover_image_rois(image_folders: str | Path | Iterable[str | Path]) -> set[str]:
    """Return ROI names discoverable from image folders."""

    if isinstance(image_folders, (str, Path)):
        image_folders = [image_folders]

    rois: set[str] = set()
    for folder in image_folders:
        folder = Path(folder)
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_dir():
                rois.add(path.name)
            elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rois.add(path.stem)
    return rois


def discover_roi_assets(
    masks_folder: str | Path,
    image_folders: str | Path | Iterable[str | Path],
    feature_rois: Iterable[str] | None = None,
    mask_extension: str | None = None,
) -> dict[str, RoiAssets]:
    """Discover masks and images and return assets keyed by ROI."""

    mask_paths = discover_mask_files(masks_folder, mask_extension=mask_extension)
    image_rois = discover_image_rois(image_folders)
    roi_names = set(mask_paths) | image_rois
    if feature_rois is not None:
        roi_names |= {str(roi) for roi in feature_rois}

    assets = {}
    for roi in sorted(roi_names):
        assets[roi] = RoiAssets(
            roi=roi,
            mask_path=mask_paths.get(roi),
            image_paths=discover_roi_images(image_folders, roi),
        )
    return assets


def load_mask(mask_path: str | Path) -> np.ndarray:
    """Load a 2D integer mask image."""

    mask = sk.io.imread(mask_path)
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(
            f"Mask must be 2D after squeezing, got shape {mask.shape}: {mask_path}"
        )
    return mask.astype(np.int32, copy=False)


def load_display_image(
    image_path: str | Path,
    *,
    quantile: float = 0.999,
    minimum_pixel_counts: float = 0.1,
    normalization_value: float | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Load an image for Napari display using explorer-style normalization.

    Returns ``(image, is_rgb)``.
    """

    image = sk.io.imread(image_path)
    image = np.asarray(image)
    is_rgb = image.ndim >= 3 and image.shape[-1] in (3, 4)
    if is_rgb:
        return image, True

    image = np.squeeze(image).astype(np.float32, copy=False)
    image = np.where(np.isfinite(image), image, 0)
    image = normalize_imc_image(
        image,
        quantile=quantile,
        minimum_pixel_counts=minimum_pixel_counts,
        normalization_value=normalization_value,
    )
    return image, False


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV or parquet table by extension."""

    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_table(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> None:
    """Save a CSV or parquet table by extension."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=index)
    else:
        df.to_csv(path, index=index)
