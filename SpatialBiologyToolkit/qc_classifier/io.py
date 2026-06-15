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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import skimage as sk

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
        "modified_time": datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat(),
        "sha256": sha256.hexdigest(),
        "hash_bytes": bytes_read,
    }


def infer_mask_extension(masks_folder: str | Path, mask_extension: str | None = None) -> str | None:
    """Infer the extension used by masks in ``masks_folder``."""

    if mask_extension:
        return mask_extension if str(mask_extension).startswith(".") else f".{mask_extension}"

    folder = Path(masks_folder)
    if not folder.exists():
        return None

    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in MASK_EXTENSIONS:
            return path.suffix
    return None


def discover_mask_files(masks_folder: str | Path, mask_extension: str | None = None) -> dict[str, Path]:
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


def _image_channel_name(image_path: Path) -> str:
    """
    Return a concise channel/display name for an ROI image path.

    This follows the same broad convention as ``napari_imc_explorer``:
    filenames like ``ROI_channel_marker.tiff`` use the later stem parts when
    possible, while simple filenames use the stem.
    """

    stem = image_path.stem
    parts = stem.split("_")
    if len(parts) >= 4:
        return "_".join(parts[3:]) or stem
    return stem


def discover_roi_images(image_folders: str | Path | Iterable[str | Path], roi: str) -> dict[str, Path]:
    """
    Return display-name to image-path mappings for one ROI.

    Supports both ``images/<ROI>/<channel>.tiff`` and direct
    ``images/<ROI>.tiff`` layouts.
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
                    image_paths.setdefault(_image_channel_name(path), path)
            continue

        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if path.stem == roi:
                image_paths.setdefault(folder.name or path.stem, path)

    return image_paths


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
        raise ValueError(f"Mask must be 2D after squeezing, got shape {mask.shape}: {mask_path}")
    return mask.astype(np.int32, copy=False)


def load_display_image(
    image_path: str | Path,
    *,
    quantile: float = 0.999,
    minimum_pixel_counts: float = 0.1,
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
    image = np.where(image > minimum_pixel_counts, image, 0)
    finite_values = image[np.isfinite(image)]
    if finite_values.size:
        max_quant = float(np.quantile(finite_values, quantile))
        if max_quant < 5:
            max_quant = 3.0
        if max_quant > 0:
            image = image / max_quant
    image = np.clip(image, 0, 1)
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
