"""
Nimbus-powered drop-in for segmentation.py.
Uses Nimbus-Inference on existing masks/images to build cell tables and an AnnData object.
"""
from __future__ import annotations

import logging
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import scanpy as sc
from alpineer import io_utils
from skimage import io
from skimage.measure import regionprops, regionprops_table
from tqdm import tqdm
import matplotlib.pyplot as plt

from SpatialBiologyToolkit.nimbus_normalization import (
    PREFERRED_NORMALIZATION_FILENAME,
    load_normalization_file,
    merge_computed_normalization_parameters,
    normalize_nimbus_image,
    resolve_normalization_input_path,
    resolve_normalization_parameters,
    write_normalization_csv,
)

try:
    from nimbus_inference.nimbus import Nimbus
    from nimbus_inference.utils import (
        MultiplexDataset,
        prepare_input_data,
        segment_mean,
        test_time_aug,
    )
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "Nimbus-Inference is required for this script. Install it with 'pip install nimbus-inference'."
    ) from exc

from .config_and_utils import (
    GeneralConfig,
    NimbusConfig,
    SegmentationConfig,
    filter_config_for_dataclass,
    get_filename,
    load_pipeline_anndata,
    process_config_with_overrides,
    save_pipeline_anndata,
    setup_logging,
)
from .segmentation import create_anndata, normalise_markers


_INSPECTED_CHANNEL_IMAGE_PATHS: Set[str] = set()
_NONFINITE_CHANNEL_IMAGE_PATHS: Set[str] = set()


def _warn_and_sanitize_channel_image(
    image: np.ndarray,
    image_path: Path | str,
    *,
    roi: Optional[str] = None,
    channel: Optional[str] = None,
) -> np.ndarray:
    """Warn once per image path when non-finite pixels are present, then replace them with zero."""
    image_key = str(Path(image_path))
    if image_key not in _INSPECTED_CHANNEL_IMAGE_PATHS:
        _INSPECTED_CHANNEL_IMAGE_PATHS.add(image_key)
        array = np.asarray(image)
        try:
            nan_count = int(np.isnan(array).sum())
            inf_count = int(np.isinf(array).sum())
        except TypeError:
            nan_count = 0
            inf_count = 0

        if nan_count > 0 or inf_count > 0:
            _NONFINITE_CHANNEL_IMAGE_PATHS.add(image_key)
            context_parts = []
            if roi is not None:
                context_parts.append(f"ROI='{roi}'")
            if channel is not None:
                context_parts.append(f"channel='{channel}'")
            context = f" ({', '.join(context_parts)})" if context_parts else ""
            logging.warning(
                "Channel image %s%s contains %d NaN pixel(s) and %d infinite pixel(s) out of %d total; "
                "replacing non-finite pixels with 0.",
                image_path,
                context,
                nan_count,
                inf_count,
                array.size,
            )

    if image_key not in _NONFINITE_CHANNEL_IMAGE_PATHS:
        return image

    sanitized = np.asarray(image, dtype=np.float32).copy()
    sanitized[~np.isfinite(sanitized)] = 0.0
    return sanitized


def _finite_values(values: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return array
    return array[np.isfinite(array)]


def _safe_quantile(values: np.ndarray | Sequence[float], quantile: float) -> Optional[float]:
    finite = _finite_values(values)
    if finite.size == 0:
        return None
    return float(np.quantile(finite, quantile))


def _load_mask_2d(mask_path: Path | str) -> np.ndarray:
    """Load a segmentation mask and coerce it to a 2D label image."""
    mask = io.imread(mask_path)
    mask = np.squeeze(mask)
    if mask.ndim == 3:
        mask = np.squeeze(mask[..., 0])
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask at {mask_path}, got shape {mask.shape}")
    return np.asarray(mask)


def _adjust_label_mask(mask: np.ndarray, offset_pixels: int) -> np.ndarray:
    """
    Expand or shrink a labeled mask while preserving label IDs.

    Positive offsets expand labels into background without overlap. Negative offsets
    erode each cell independently; cells that disappear completely are dropped.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Label mask must be 2D, got shape {mask.shape}")

    offset_pixels = int(offset_pixels)
    labels = mask.astype(np.uint32, copy=False)
    if offset_pixels == 0:
        return labels

    if offset_pixels > 0:
        from skimage.segmentation import expand_labels

        return np.asarray(expand_labels(labels, distance=offset_pixels), dtype=np.uint32)

    from scipy.ndimage import binary_erosion

    shrink_pixels = abs(offset_pixels)
    adjusted = np.zeros_like(labels, dtype=np.uint32)
    structure = np.ones((3, 3), dtype=bool)

    for region in regionprops(labels):
        rr_slice, cc_slice = region.slice
        cell_mask = labels[rr_slice, cc_slice] == region.label
        eroded = binary_erosion(
            cell_mask,
            structure=structure,
            iterations=shrink_pixels,
            border_value=0,
        )
        if np.any(eroded):
            adjusted_view = adjusted[rr_slice, cc_slice]
            adjusted_view[eroded] = np.uint32(region.label)

    return adjusted


def _coerce_optional_area_bound(bound_name: str, value: Optional[int | float]) -> Optional[float]:
    if value is None:
        return None
    try:
        bound = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{bound_name} must be a non-negative finite value or None, got {value!r}") from exc
    if not np.isfinite(bound) or bound < 0:
        raise ValueError(f"{bound_name} must be a non-negative finite value or None, got {value!r}")
    return bound


def _filter_label_mask_by_area(
    mask: np.ndarray,
    min_cell_area: Optional[int | float] = None,
    max_cell_area: Optional[int | float] = None,
) -> np.ndarray:
    """Remove labels outside configured post-adjustment area bounds while preserving label IDs."""
    labels = np.asarray(mask)
    if labels.ndim != 2:
        raise ValueError(f"Label mask must be 2D, got shape {labels.shape}")

    min_area = _coerce_optional_area_bound("min_cell_area", min_cell_area)
    max_area = _coerce_optional_area_bound("max_cell_area", max_cell_area)
    if min_area is not None and max_area is not None and min_area > max_area:
        raise ValueError(f"min_cell_area ({min_area:g}) cannot be greater than max_cell_area ({max_area:g})")
    labels = labels.astype(np.uint32, copy=False)
    if min_area is None and max_area is None:
        return labels

    filtered = labels.copy()
    for region in regionprops(labels):
        area = float(region.area)
        below_min = min_area is not None and area < min_area
        above_max = max_area is not None and area > max_area
        if below_min or above_max:
            filtered_view = filtered[region.slice]
            source_view = labels[region.slice]
            filtered_view[source_view == region.label] = 0

    return filtered


def _load_adjusted_mask(
    mask_path: Path | str,
    mask_boundary_offset_pixels: int = 0,
    min_cell_area: Optional[int | float] = None,
    max_cell_area: Optional[int | float] = None,
) -> np.ndarray:
    """Load a mask from disk, apply the boundary offset, then filter by post-offset cell area."""
    mask = _load_mask_2d(mask_path)
    adjusted = _adjust_label_mask(mask, mask_boundary_offset_pixels)
    return _filter_label_mask_by_area(adjusted, min_cell_area, max_cell_area)


class ToolkitNimbusDataset(MultiplexDataset):
    """Nimbus dataset wrapper aware of separate mask folder and mixed raw/denoised channels."""

    def __init__(
        self,
        fov_paths: List[Path] | List[str],
        channels: Iterable[str],
        channel_paths: Dict[str, Dict[str, Path]],
        mask_lookup: Dict[str, Path],
        *,
        suffix: str = ".tiff",
        magnification: int = 20,
        output_dir: str = "nimbus_output",
        qc_folder: str = "QC",
        normalization_jobs: int = 1,
        clip_values: Sequence[float] = (0.0, 2.0),
        normalization_min_value: float = 2.0,
        normalization_lower_threshold: float = 0.0,
        mask_boundary_offset_pixels: int = 0,
        min_cell_area: Optional[int | float] = None,
        max_cell_area: Optional[int | float] = None,
        suffix_match: Optional[str] = None,
    ) -> None:
        self._channels = sorted(channels)
        self._channel_paths = channel_paths
        self._mask_lookup = mask_lookup
        self.qc_folder = qc_folder
        self.normalization_n_jobs = max(1, int(normalization_jobs))
        self.clip_values = tuple(clip_values)
        self.normalization_min_value = float(normalization_min_value)
        self.normalization_lower_threshold = float(normalization_lower_threshold)
        self.mask_boundary_offset_pixels = int(mask_boundary_offset_pixels)
        self.min_cell_area = min_cell_area
        self.max_cell_area = max_cell_area
        self._suffix_match = suffix_match or suffix
        self._segmentation_cache: Dict[str, np.ndarray] = {}
        self.normalization_lower_thresholds: Dict[str, float] = {
            channel: self.normalization_lower_threshold for channel in self._channels
        }

        def _seg_lookup(fov_path: str) -> Path:
            return str(self._mask_lookup[Path(fov_path).name])

        str_fov_paths = [str(p) for p in fov_paths]

        super().__init__(
            str_fov_paths,
            segmentation_naming_convention=_seg_lookup,
            include_channels=self._channels,
            suffix=suffix,
            silent=True,
            magnification=magnification,
            output_dir=str(output_dir),
        )

        # Normalise FOV names to ROI folder names so downstream joins are stable
        self.fovs = [Path(p).name for p in str_fov_paths]
        self.channels = self._channels
        self.include_channels = self._channels

    def check_inputs(self):  # type: ignore[override]
        """
        Simplified check to avoid directory-derived channel validation; we supply channels explicitly.
        """
        paths = self.fov_paths if isinstance(self.fov_paths, (list, tuple)) else [self.fov_paths]
        io_utils.validate_paths(paths)
        self.channels = self._channels
        self.include_channels = self._channels
        if not getattr(self, "silent", True):
            print("All inputs are valid")

    def get_channels(self):  # type: ignore[override]
        return self._channels

    def get_channel_single(self, fov: str, channel: str):  # type: ignore[override]
        roi = Path(fov).name
        try:
            image_path = self._channel_paths[roi][channel]
        except KeyError as exc:  # pragma: no cover - defensive
            raise FileNotFoundError(f"Missing image for ROI '{roi}', channel '{channel}'") from exc
        img = io.imread(image_path)
        img = _warn_and_sanitize_channel_image(img, image_path, roi=roi, channel=channel)
        if img.ndim == 2:
            return img
        # If channel dimension leaked through (e.g., multichannel format), take first plane
        return np.squeeze(img)[0] if np.squeeze(img).ndim == 3 else np.squeeze(img)

    def get_channel_normalized(self, fov: str, channel: str):  # type: ignore[override]
        """Load and normalize one channel using its absolute lower and upper bounds."""

        if not hasattr(self, "normalization_dict"):
            logging.info("No Nimbus normalization table found; preparing one now.")
            self.prepare_normalization_dict()
        image = self.get_channel(fov, channel).astype(np.float32)
        vmax = self.normalization_dict.get(channel)
        if vmax is None:
            vmax = float(np.quantile(image, 0.999))
            logging.warning(
                "No saved Nimbus Vmax for channel %r; using this image's 0.999 "
                "quantile %.4g with lower_threshold=%g.",
                channel,
                vmax,
                self.normalization_lower_threshold,
            )
        lower_threshold = self.normalization_lower_thresholds.get(
            channel, self.normalization_lower_threshold
        )
        return normalize_nimbus_image(
            image,
            vmax=float(vmax),
            lower_threshold=float(lower_threshold),
            upper_clip=1.0,
        )

    def get_segmentation(self, fov: str):  # type: ignore[override]
        roi = Path(fov).name
        if roi not in self._segmentation_cache:
            self._segmentation_cache[roi] = _load_adjusted_mask(
                self._mask_lookup[roi],
                self.mask_boundary_offset_pixels,
                min_cell_area=self.min_cell_area,
                max_cell_area=self.max_cell_area,
            )
        mask = self._segmentation_cache[roi]

        # Align mask to the reference channel image shape if needed
        ref_path = next(iter(self._channel_paths[roi].values()))
        ref_img = io.imread(ref_path)
        ref_img = _warn_and_sanitize_channel_image(ref_img, ref_path, roi=roi)
        ref_shape = np.squeeze(ref_img).shape[-2:] if ref_img.ndim >= 2 else ref_img.shape

        if mask.shape != tuple(ref_shape):
            raise ValueError(f"Mask/Image shape mismatch for ROI {roi}: {mask.shape} vs {ref_shape}")
        return mask.astype(np.uint32, copy=False)

    def prepare_normalization_dict(  # type: ignore[override]
        self,
        quantile: float = 0.999,
        clip_values: Sequence[float] = (0, 2),
        n_subset: int = 10,
        multiprocessing: bool = False,  # kept for API compatibility
        reuse_saved: bool = False,
        normalization_file: Optional[Path | str] = None,
    ):
        """
        Compute per-channel normalization using ALL FOVs and only in-mask pixels.
        Also writes a QC gallery and QC histograms.
        
        An explicit normalization_file takes precedence and must use the preferred
        CSV format. Otherwise, if reuse_saved=True, normalization_dict.csv is
        preferred and a legacy normalization_dict.json remains readable. Legacy JSON
        values retain their saved lower thresholds and are migrated to CSV without
        deleting the source JSON. Newly computed rows use the configured default lower
        threshold. QC plots are regenerated with the resolved values.
        """
        self.clip_values = tuple(clip_values)
        preferred_path = Path(self.output_dir) / PREFERRED_NORMALIZATION_FILENAME
        self.normalization_dict_path = str(preferred_path)

        saved_path = resolve_normalization_input_path(
            self.output_dir,
            configured_path=normalization_file,
            reuse_saved=reuse_saved,
        )

        if saved_path is not None:
            logging.info("Found existing normalization table at %s", saved_path)
            if normalization_file is not None:
                logging.info(
                    "Using explicitly configured Nimbus normalization CSV "
                    "(nimbus.normalization_dict_path)."
                )
            else:
                logging.info(
                    "Reusing saved normalization values "
                    "(reuse_saved_normalization=True)."
                )
            loaded = load_normalization_file(saved_path)
            resolved = resolve_normalization_parameters(
                loaded,
                self._channels,
                require_all=False,
            )
            missing = [
                channel for channel in self._channels if channel not in resolved
            ]
            fallback_values: Dict[str, float] = {}
            if missing:
                logging.warning(
                    "Saved Nimbus normalization table is missing channels; computing "
                    "mask-aware cohort Vmax values with lower_threshold=%g for: %s",
                    self.normalization_lower_threshold,
                    missing,
                )
                fallback_values = self.compute_normalization_values(
                    quantile=quantile,
                    channels=missing,
                )
            merged = merge_computed_normalization_parameters(
                fallback_values,
                default_lower_threshold=self.normalization_lower_threshold,
                saved_parameters=resolved,
            )
            self.normalization_dict = {
                channel: entry.vmax for channel, entry in merged.items()
            }
            self.normalization_lower_thresholds = {
                channel: entry.lower_threshold for channel, entry in merged.items()
            }
            logging.info(
                "Loaded %d channel normalization rows (manual values preserved).",
                len(self.normalization_dict),
            )
            source_is_preferred = (
                saved_path.resolve(strict=False) == preferred_path.resolve(strict=False)
            )
            if saved_path.suffix.casefold() == ".json" or missing or not source_is_preferred:
                write_normalization_csv(
                    preferred_path,
                    self.normalization_dict,
                    self.normalization_lower_thresholds,
                )
                logging.info("Wrote resolved normalization values to: %s", preferred_path)
        else:
            computed_values = self.compute_normalization_values(
                quantile=quantile
            )
            computed = merge_computed_normalization_parameters(
                computed_values,
                default_lower_threshold=self.normalization_lower_threshold,
            )
            self.normalization_dict = {
                channel: entry.vmax for channel, entry in computed.items()
            }
            self.normalization_lower_thresholds = {
                channel: entry.lower_threshold for channel, entry in computed.items()
            }
            write_normalization_csv(
                preferred_path,
                self.normalization_dict,
                self.normalization_lower_thresholds,
            )

        # QC: histograms of raw norms and cell-level positivity, plus gallery of normalized images
        os.makedirs(os.path.join(self.qc_folder, "nimbus_normalization_qc"), exist_ok=True)
        norm_hist_dir = os.path.join(self.qc_folder, "nimbus_normalization_qc", "norm_hists")
        pos_hist_dir = os.path.join(self.qc_folder, "nimbus_normalization_qc", "cellpos_hists")
        os.makedirs(norm_hist_dir, exist_ok=True)
        os.makedirs(pos_hist_dir, exist_ok=True)

        upper_clip = self.clip_values[1] if len(self.clip_values) > 1 else 2.0

        def _save_hist(data: List[float], marker: Optional[float], out_path: str, xlabel: str, title: str):
            hist_data = _finite_values(data)
            if hist_data.size == 0:
                logging.warning(
                    "Skipping histogram for '%s' because there are no finite values to plot.",
                    title,
                )
                return
            marker_value: Optional[float] = None
            if marker is not None:
                marker_float = float(marker)
                if np.isfinite(marker_float):
                    marker_value = marker_float
                else:
                    logging.warning(
                        "Skipping non-finite histogram marker for '%s': %s",
                        title,
                        marker,
                    )
            plt.figure(figsize=(4, 3))
            plt.hist(hist_data, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
            if marker_value is not None:
                plt.axvline(marker_value, color="red", linestyle="--", label=f"marker={marker_value:.3g}")
            plt.xlabel(xlabel)
            plt.ylabel("Count")
            plt.title(title)
            if marker_value is not None:
                plt.legend()
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()

        # Collect per-ROI norms and positivity proportions
        norm_vals: Dict[str, List[float]] = {ch: [] for ch in self._channels}
        cell_pos_props: Dict[str, List[float]] = {ch: [] for ch in self._channels}

        for fov in self.fovs:
            mask = self.get_segmentation(fov)
            mask_bool = mask > 0
            if not np.any(mask_bool):
                continue
            labels = mask.astype(np.int32)
            for ch in self._channels:
                norm = self.normalization_dict.get(ch, 1.0) or 1.0
                lower_threshold = self.normalization_lower_thresholds.get(
                    ch, self.normalization_lower_threshold
                )
                img_raw = self.get_channel(fov, ch).astype(float)
                foreground_quantile = _safe_quantile(img_raw[mask_bool], quantile)
                if foreground_quantile is None:
                    logging.warning(
                        "Skipping normalization QC quantile for ROI '%s', channel '%s' because the masked pixels "
                        "contain no finite values.",
                        fov,
                        ch,
                    )
                else:
                    norm_vals[ch].append(foreground_quantile)

                img = normalize_nimbus_image(
                    img_raw,
                    vmax=norm,
                    lower_threshold=lower_threshold,
                    upper_clip=upper_clip,
                )
                props = regionprops_table(label_image=labels, intensity_image=img, properties=["intensity_mean"])
                means = _finite_values(props.get("intensity_mean", []))
                if means.size > 0:
                    cell_pos_props[ch].append(float(np.mean(means > 1.0)))

        for ch in self._channels:
            final_val = self.normalization_dict.get(ch, 1.0)
            final_label = f"{final_val:.3g}" if np.isfinite(final_val) else "non-finite"
            pos_marker = None
            finite_pos_props = _finite_values(cell_pos_props.get(ch, []))
            if finite_pos_props.size > 0:
                pos_marker = float(np.mean(finite_pos_props))
            _save_hist(
                norm_vals.get(ch, []),
                final_val,
                os.path.join(norm_hist_dir, f"{ch}.png"),
                xlabel=f"{ch} per-ROI quantiles",
                title=f"{ch} norm values (final {final_label})",
            )
            _save_hist(
                cell_pos_props.get(ch, []),
                pos_marker,
                os.path.join(pos_hist_dir, f"{ch}.png"),
                xlabel=f"{ch} proportion of cells > normalized 1.0",
                title=f"{ch} cell positivity per ROI",
            )

        # QC gallery: create side-by-side comparison images (unmasked left, masked right) per channel
        # Similar to qc_check_side_by_side from denoising.py
        if n_subset and n_subset > 0:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            
            qc_fovs = list(self.fovs)
            if len(qc_fovs) > n_subset:
                qc_fovs = random.sample(qc_fovs, n_subset)
            
            qc_gallery_dir = os.path.join(self.qc_folder, "nimbus_normalization_qc", "channel_galleries")
            os.makedirs(qc_gallery_dir, exist_ok=True)
            
            for ch in self._channels:
                norm = self.normalization_dict.get(ch, 1.0) or 1.0
                lower_threshold = self.normalization_lower_thresholds.get(
                    ch, self.normalization_lower_threshold
                )
                
                # Create figure with 4 columns (raw, normalized unmasked, normalized masked, clip diagnostic) and one row per FOV
                fig, axs = plt.subplots(len(qc_fovs), 4, figsize=(20, 5 * len(qc_fovs)), dpi=100)
                
                # Handle single ROI case (axs won't be 2D)
                if len(qc_fovs) == 1:
                    axs = np.array([axs])
                
                for row_idx, fov in enumerate(qc_fovs):
                    mask = self.get_segmentation(fov)
                    mask_bool = mask > 0
                    
                    img_raw = self.get_channel(fov, ch).astype(float)
                    img = normalize_nimbus_image(
                        img_raw,
                        vmax=norm,
                        lower_threshold=lower_threshold,
                        upper_clip=upper_clip,
                    )
                    img_masked = img * mask_bool
                    
                    # Column 0: Raw image (before normalization)
                    # Use 99th percentile for display scaling to handle outliers
                    raw_vmax = np.percentile(img_raw, 99) if np.any(img_raw > 0) else 1.0
                    im0 = axs[row_idx, 0].imshow(img_raw, vmin=0, vmax=raw_vmax, cmap='gray')
                    divider0 = make_axes_locatable(axs[row_idx, 0])
                    cax0 = divider0.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im0, cax=cax0, orientation='vertical')
                    axs[row_idx, 0].set_ylabel(fov, fontsize=8)
                    if row_idx == 0:
                        axs[row_idx, 0].set_title('Raw', fontsize=10)
                    
                    # Column 1: Normalized unmasked
                    im1 = axs[row_idx, 1].imshow(img, vmin=0, vmax=upper_clip, cmap='gray')
                    divider1 = make_axes_locatable(axs[row_idx, 1])
                    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im1, cax=cax1, orientation='vertical')
                    if row_idx == 0:
                        axs[row_idx, 1].set_title('Normalized', fontsize=10)
                    
                    # Column 2: Normalized masked
                    im2 = axs[row_idx, 2].imshow(img_masked, vmin=0, vmax=upper_clip, cmap='gray')
                    divider2 = make_axes_locatable(axs[row_idx, 2])
                    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im2, cax=cax2, orientation='vertical')
                    if row_idx == 0:
                        axs[row_idx, 2].set_title('Normalized + Masked', fontsize=10)
                    
                    # Column 3: Clipping diagnostic
                    # Create RGB image: grayscale base, red for clipped (max), blue for zeros
                    img_normalized = img / upper_clip  # Normalize to 0-1 for display
                    clip_diag = np.stack([img_normalized, img_normalized, img_normalized], axis=-1)
                    
                    # Identify clipped pixels (at or very close to upper_clip) and zero pixels
                    clipped_mask = img >= (upper_clip - 1e-6)
                    zero_mask = img <= 1e-6
                    
                    # Set clipped pixels to red (R=1, G=0, B=0)
                    clip_diag[clipped_mask] = [1.0, 0.0, 0.0]
                    # Set zero pixels to blue (R=0, G=0, B=1)
                    clip_diag[zero_mask] = [0.0, 0.0, 1.0]
                    
                    axs[row_idx, 3].imshow(clip_diag)
                    axs[row_idx, 3].set_xticks([])
                    axs[row_idx, 3].set_yticks([])
                    
                    # Add text overlay with normalization value, clip value, and counts
                    n_clipped = np.sum(clipped_mask)
                    n_zero = np.sum(zero_mask)
                    pct_clipped = 100.0 * n_clipped / img.size
                    pct_zero = 100.0 * n_zero / img.size
                    overlay_text = (
                        f'vmax={norm:.2f}\nlower={lower_threshold:.2f}\n'
                        f'clip={upper_clip:.2f}\nred(clip): {pct_clipped:.1f}%\n'
                        f'blue(zero): {pct_zero:.1f}%'
                    )
                    axs[row_idx, 3].text(
                        0.02, 0.98, overlay_text,
                        transform=axs[row_idx, 3].transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                    )
                    if row_idx == 0:
                        axs[row_idx, 3].set_title('Clip Diagnostic\n(red=clipped, blue=zero)', fontsize=10)
                
                fig.suptitle(
                    f'{ch} (vmax={norm:.3g}, lower={lower_threshold:.3g})',
                    fontsize=12,
                    fontweight='bold',
                )
                plt.tight_layout()
                fig.savefig(os.path.join(qc_gallery_dir, f'{ch}.png'), bbox_inches='tight')
                plt.close(fig)
                
            logging.info(f"Normalization QC galleries saved to: {qc_gallery_dir}")

    def compute_normalization_values(
        self,
        *,
        quantile: float = 0.999,
        channels: Optional[Sequence[str]] = None,
        fovs: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """Compute mask-aware Nimbus Vmax values without writing files or QC outputs."""

        selected_channels = list(channels) if channels is not None else list(self._channels)
        selected_fovs = list(fovs) if fovs is not None else list(self.fovs)
        unknown_channels = sorted(set(selected_channels) - set(self._channels))
        unknown_fovs = sorted(set(selected_fovs) - set(self.fovs))
        if unknown_channels:
            raise ValueError(f"Unknown Nimbus normalization channel(s): {unknown_channels}")
        if unknown_fovs:
            raise ValueError(f"Unknown Nimbus normalization ROI(s): {unknown_fovs}")
        if not 0 < float(quantile) <= 1:
            raise ValueError("Nimbus normalization quantile must lie in (0, 1].")

        norm_vals: Dict[str, List[float]] = {ch: [] for ch in selected_channels}
        for fov in selected_fovs:
            mask = self.get_segmentation(fov)
            mask_bool = mask > 0
            if not np.any(mask_bool):
                continue
            for channel in selected_channels:
                image = self.get_channel(fov, channel).astype(float)
                foreground_quantile = _safe_quantile(image[mask_bool], quantile)
                if foreground_quantile is None:
                    logging.warning(
                        "Skipping normalization quantile for ROI '%s', channel '%s' "
                        "because the masked pixels contain no finite values.",
                        fov,
                        channel,
                    )
                    continue
                norm_vals[channel].append(foreground_quantile)

        normalization_values: Dict[str, float] = {}
        for channel, values in norm_vals.items():
            if values:
                computed_value = float(np.mean(values))
                if np.isfinite(computed_value) and computed_value > 0:
                    normalization_values[channel] = max(
                        computed_value, self.normalization_min_value
                    )
                    continue
                logging.warning(
                    "Computed normalization value for channel '%s' is non-finite or "
                    "non-positive (%s); using minimum value %.3g instead.",
                    channel,
                    computed_value,
                    self.normalization_min_value,
                )
            else:
                logging.warning(
                    "No usable normalization pixels were found for channel '%s'; "
                    "using minimum value %.3g.",
                    channel,
                    self.normalization_min_value,
                )
            normalization_values[channel] = self.normalization_min_value
        return normalization_values


def _load_panel(metadata_folder: Path, nimbus_cfg: NimbusConfig) -> pd.DataFrame:
    panel = pd.read_csv(metadata_folder / "panel.csv")
    panel["channel_label"] = [re.sub(r"\\W+", "", str(x)) for x in panel["channel_label"]]
    if bool(getattr(nimbus_cfg, "simple_image_names", False)):
        panel["filename"] = panel["channel_label"] + ".tiff"
    else:
        panel["filename"] = panel["channel_name"] + "_" + panel["channel_label"]
    return panel


def _discover_masks(masks_folder: Path, extensions: Sequence[str]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for ext in extensions:
        for mask_path in masks_folder.glob(f"*{ext}"):
            roi = mask_path.stem
            if roi not in lookup:
                lookup[roi] = mask_path
    return lookup


def _filter_rois_by_metadata(mask_lookup: Dict[str, Path], metadata_path: Path) -> List[str]:
    rois = sorted(mask_lookup.keys())
    if not metadata_path.exists():
        return rois

    metadata = pd.read_csv(metadata_path, index_col="unstacked_data_folder")
    filtered: List[str] = []
    for roi in rois:
        if roi in metadata.index and not bool(metadata.loc[roi, "import_data"]):
            logging.info("Skipping ROI %s (import_data is False)", roi)
            continue
        filtered.append(roi)
    return filtered


def _resolve_channel_paths(
    rois: List[str],
    panel: pd.DataFrame,
    general: GeneralConfig,
    nimbus_cfg: NimbusConfig,
) -> Tuple[List[str], Dict[str, Dict[str, Path]], Dict[str, Path], Dict[str, List[str]], List[str], List[str]]:
    expected = panel.loc[panel["use_denoised"] | panel["use_raw"], "channel_label"].tolist()
    if not expected:
        raise ValueError("No channels flagged with use_denoised/use_raw in panel.csv")

    filename_lookup = dict(zip(panel["channel_label"], panel["filename"]))
    preferred_source = {
        row["channel_label"]: ("denoised" if row.get("use_denoised") else "raw")
        for _, row in panel.iterrows()
        if row["channel_label"] in expected
    }

    channel_paths: Dict[str, Dict[str, Path]] = {}
    roi_image_roots: Dict[str, Path] = {}
    missing_summary: Dict[str, List[str]] = {}
    available_sets: List[Set[str]] = []
    valid_rois: List[str] = []

    for roi in rois:
        paths: Dict[str, Path] = {}
        missing: List[str] = []
        representative: Optional[Path] = None

        for channel in expected:
            filename_hint = filename_lookup[channel]
            candidates: List[Path] = []
            if preferred_source.get(channel) == "denoised":
                candidates.append(Path(general.denoised_images_folder) / roi)
                if nimbus_cfg.allow_raw_fallback:
                    candidates.append(Path(general.raw_images_folder) / roi)
            else:
                candidates.append(Path(general.raw_images_folder) / roi)
                if nimbus_cfg.allow_raw_fallback:
                    candidates.append(Path(general.denoised_images_folder) / roi)

            found: Optional[Path] = None
            for base_dir in candidates:
                if not base_dir.exists():
                    continue
                # Match on any file containing the filename hint (handles prefixes like index_roi_)
                matches = sorted([p for p in base_dir.iterdir() if filename_hint in p.name])
                if matches:
                    found = matches[0]
                    representative = representative or base_dir
                    break

            if found:
                paths[channel] = found
            else:
                missing.append(channel)

        if paths:
            channel_paths[roi] = paths
            roi_image_roots[roi] = representative or next(iter(paths.values())).parent
            available_sets.append(set(paths))
            valid_rois.append(roi)
            if missing:
                missing_summary[roi] = missing
        else:
            missing_summary[roi] = missing or expected
            logging.warning("Skipping ROI %s because no channel images were found.", roi)

    common_channels = sorted(set.intersection(*available_sets)) if available_sets else []
    return valid_rois, channel_paths, roi_image_roots, missing_summary, expected, common_channels


def _build_mask_features(
    mask_lookup: Dict[str, Path],
    rois: List[str],
    mask_boundary_offset_pixels: int = 0,
    min_cell_area: Optional[int | float] = None,
    max_cell_area: Optional[int | float] = None,
) -> pd.DataFrame:
    circ = lambda r: (4 * np.pi * r.area) / (r.perimeter * r.perimeter) if r.perimeter > 0 else 0
    frames: List[pd.DataFrame] = []
    for roi in rois:
        mask = _load_adjusted_mask(
            mask_lookup[roi],
            mask_boundary_offset_pixels,
            min_cell_area=min_cell_area,
            max_cell_area=max_cell_area,
        )
        props = regionprops(mask)
        df = pd.DataFrame({
            "ObjectNumber": [p.label for p in props],
            "X_loc": [p.centroid[1] for p in props],
            "Y_loc": [p.centroid[0] for p in props],
            "mask_area": [p.area for p in props],
            "mask_perimeter": [p.perimeter for p in props],
            "mask_circularity": [circ(p) for p in props],
            "mask_largest_diameter": [p.major_axis_length for p in props],
            "mask_largest_diameter_angle": [np.degrees(p.orientation) for p in props],
        })
        df["ROI"] = roi
        cols = ["ROI"] + [c for c in df.columns if c != "ROI"]
        frames.append(df.loc[:, cols])
    return pd.concat(frames, ignore_index=True)


def _extract_classic_intensities(
    mask_lookup: Dict[str, Path],
    rois: List[str],
    channel_paths: Dict[str, Dict[str, Path]],
    expected_channels: List[str],
    mask_boundary_offset_pixels: int = 0,
    min_cell_area: Optional[int | float] = None,
    max_cell_area: Optional[int | float] = None,
) -> pd.DataFrame:
    """
    Extract classic mean intensities by measuring directly over masks (like original segmentation.py).
    Returns DataFrame with ROI, ObjectNumber, and channel intensities.
    """
    frames: List[pd.DataFrame] = []
    
    for roi in tqdm(rois, desc="Classic intensity extraction"):
        mask = _load_adjusted_mask(
            mask_lookup[roi],
            mask_boundary_offset_pixels,
            min_cell_area=min_cell_area,
            max_cell_area=max_cell_area,
        )
        props = regionprops(mask)
        
        # Initialize DataFrame with object numbers
        roi_df = pd.DataFrame({
            "ObjectNumber": [p.label for p in props],
        })
        
        # Extract mean intensities for each channel
        available_channels = channel_paths.get(roi, {})
        
        for channel in expected_channels:
            if channel not in available_channels:
                roi_df[channel] = np.nan
                continue
                
            try:
                image_path = available_channels[channel]
                image = io.imread(image_path)
                image = _warn_and_sanitize_channel_image(image, image_path, roi=roi, channel=channel)
                
                # Calculate mean intensity for each label
                mean_intensities = [region.mean_intensity for region in regionprops(mask, image)]
                roi_df[channel] = mean_intensities
                
            except Exception as e:
                logging.warning(f"Error extracting classic intensity for {roi}, channel {channel}: {e}")
                roi_df[channel] = np.nan
        
        roi_df["ROI"] = roi
        frames.append(roi_df)
    
    return pd.concat(frames, ignore_index=True)


def _process_roi_expansion(
    roi: str,
    mask_path: Path,
    roi_channel_paths: Dict[str, Path],
    expected_channels: List[str],
    expansion_pixels: int,
    mask_boundary_offset_pixels: int,
    min_cell_area: Optional[int | float],
    max_cell_area: Optional[int | float],
) -> pd.DataFrame:
    """
    Process a single ROI for expansion intensity extraction.
    This function is designed to be called in parallel.
    """
    from scipy.ndimage import binary_dilation
    
    mask = _load_adjusted_mask(
        mask_path,
        mask_boundary_offset_pixels,
        min_cell_area=min_cell_area,
        max_cell_area=max_cell_area,
    )
    
    # Get unique cell labels
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background
    
    # Pre-load all channel images for this ROI
    channel_images = {}
    for channel in expected_channels:
        if channel in roi_channel_paths:
            try:
                img_path = roi_channel_paths[channel]
                img = io.imread(img_path)
                img = _warn_and_sanitize_channel_image(img, img_path, roi=roi, channel=channel)
                img = np.squeeze(img)
                if img.ndim == 3:
                    img = img[..., 0]
                channel_images[channel] = img
            except Exception as e:
                logging.warning(f"Error loading {channel} for {roi}: {e}")
    
    # Process each cell: expand mask and measure intensities
    cell_data = []
    for label in unique_labels:
        cell_mask = (mask == label)
        # Expand the mask by dilation
        expanded_mask = binary_dilation(cell_mask, iterations=expansion_pixels)
        
        # Measure intensities for this cell across all channels
        cell_intensities = {"ObjectNumber": label}
        for channel in expected_channels:
            if channel in channel_images:
                cell_intensities[channel] = np.mean(channel_images[channel][expanded_mask])
            else:
                cell_intensities[channel] = np.nan
        
        cell_data.append(cell_intensities)
    
    # Create DataFrame for this ROI
    roi_df = pd.DataFrame(cell_data, columns=["ObjectNumber"] + list(expected_channels))
    roi_df["ROI"] = roi
    return roi_df


def _extract_expansion_intensities(
    mask_lookup: Dict[str, Path],
    rois: List[str],
    channel_paths: Dict[str, Dict[str, Path]],
    expected_channels: List[str],
    expansion_pixels: int,
    mask_boundary_offset_pixels: int = 0,
    min_cell_area: Optional[int | float] = None,
    max_cell_area: Optional[int | float] = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Extract mean intensities by expanding each cell mask by a specified number of pixels.
    Memory-efficient approach: processes one cell at a time without storing all expanded masks.
    Supports parallel processing at the ROI level.
    
    Parameters:
    -----------
    n_jobs : int
        Number of parallel jobs. 1 = sequential, -1 = use all CPUs, >1 = specific number
    
    Returns:
    --------
    DataFrame with ROI, ObjectNumber, and channel intensities.
    """
    from multiprocessing import Pool, cpu_count
    
    # Ensure n_jobs is an integer (handle config string values)
    n_jobs = int(n_jobs)
    
    # Determine number of processes
    if n_jobs == -1:
        n_processes = cpu_count()
    elif n_jobs > 1:
        n_processes = min(n_jobs, len(rois), cpu_count())
    else:
        n_processes = 1
    
    if n_processes > 1:
        logging.info(f"Processing {len(rois)} ROIs with {n_processes} parallel workers")
        
        # Prepare arguments for parallel processing
        args_list = [
            (
                roi,
                mask_lookup[roi],
                channel_paths.get(roi, {}),
                expected_channels,
                expansion_pixels,
                mask_boundary_offset_pixels,
                min_cell_area,
                max_cell_area,
            )
            for roi in rois
        ]
        
        # Process ROIs in parallel
        with Pool(processes=n_processes) as pool:
            frames = list(tqdm(
                pool.starmap(_process_roi_expansion, args_list),
                total=len(rois),
                desc="Expansion intensity extraction"
            ))
    else:
        # Sequential processing
        frames = []
        for roi in tqdm(rois, desc="Expansion intensity extraction"):
            roi_df = _process_roi_expansion(
                roi=roi,
                mask_path=mask_lookup[roi],
                roi_channel_paths=channel_paths.get(roi, {}),
                expected_channels=expected_channels,
                expansion_pixels=expansion_pixels,
                mask_boundary_offset_pixels=mask_boundary_offset_pixels,
                min_cell_area=min_cell_area,
                max_cell_area=max_cell_area,
            )
            frames.append(roi_df)
    
    return pd.concat(frames, ignore_index=True)


def _prepare_nimbus_output(cell_table: pd.DataFrame) -> pd.DataFrame:
    if cell_table is None or cell_table.empty:
        raise ValueError("Nimbus returned an empty cell table")
    renamed = cell_table.rename(columns={"label": "ObjectNumber", "fov": "ROI"})
    renamed["ROI"] = renamed["ROI"].astype(str)
    renamed["ObjectNumber"] = renamed["ObjectNumber"].astype(int)
    return renamed


def _resolve_master_celltable_path(celltable_value: Optional[str], output_dir: str) -> Path:
    value = celltable_value or ""
    if not value:
        raise ValueError("Master cell table path is empty")
    path = Path(value)
    if not path.is_absolute():
        path = Path(output_dir) / path
    return path


def _load_existing_master_celltable(path: Path, label: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logging.warning("Failed to read existing %s master cell table at %s: %s", label, path, exc)
        return None
    if df.empty:
        logging.warning("Existing %s master cell table at %s is empty; recomputing.", label, path)
        return None
    if "ROI" in df.columns:
        df["ROI"] = df["ROI"].astype(str)
    if "ObjectNumber" in df.columns:
        try:
            df["ObjectNumber"] = df["ObjectNumber"].astype(int)
        except ValueError:
            logging.warning("Could not coerce ObjectNumber to int for %s master cell table at %s.", label, path)
    logging.info("Loaded existing %s master cell table from %s", label, path)
    return df


def _merge_with_masks(
    mask_features: pd.DataFrame,
    nimbus_df: pd.DataFrame,
    expected_channels: List[str],
    predicted_channels: List[str],
    allow_missing: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    merged = mask_features.merge(nimbus_df, on=["ROI", "ObjectNumber"], how="left")

    if allow_missing:
        for ch in expected_channels:
            if ch not in merged.columns:
                merged[ch] = np.nan
    else:
        dropped = [ch for ch in expected_channels if ch not in predicted_channels]
        if dropped:
            logging.warning("Excluding channels missing across ROIs: %s", dropped)
        expected_channels = [ch for ch in expected_channels if ch in merged.columns]

    channel_cols = [ch for ch in expected_channels if ch in merged.columns]
    metadata_cols = [c for c in mask_features.columns if c not in {"ROI", "ObjectNumber"}]
    other_cols = [c for c in merged.columns if c not in {"ROI", "ObjectNumber"} | set(channel_cols) | set(metadata_cols)]

    ordered = ["ROI", "ObjectNumber"] + metadata_cols + other_cols + channel_cols
    merged = merged.loc[:, ordered]
    merged.reset_index(drop=True, inplace=True)
    merged["Master_Index"] = merged.index
    return merged, channel_cols


def _save_roi_tables(cell_df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for roi, roi_df in cell_df.groupby("ROI"):
        roi_df.to_csv(output_dir / f"{prefix}{roi}.csv", index=False)


def _create_anndata_with_layers(
    celltable: pd.DataFrame,
    classic_intensities: Optional[pd.DataFrame],
    expansion_intensities: Optional[pd.DataFrame],
    metadata_folder: Path,
    normalisation: Optional[List[str]],
    remove_channels: Optional[List[str]],
    expected_channels: List[str],
) -> sc.AnnData:
    """
    Create AnnData object with multiple layers:
    - .X: normalized Nimbus data (default)
    - .layers['nimbus_raw']: raw Nimbus predictions
    - .layers['mean_intensities_raw']: raw classic mean intensities (if available)
    - .layers['mean_intensities_normalized']: normalized classic mean intensities (if available)
    - .layers['expansion_intensities_raw']: raw expansion intensities (if available)
    - .layers['expansion_intensities_normalized']: normalized expansion intensities (if available)
    """
    
    # Get available channels in celltable
    available_channels = [ch for ch in expected_channels if ch in celltable.columns]
    if not available_channels:
        raise ValueError("No marker channels found in cell table")
    
    logging.info(f'Creating AnnData with {len(available_channels)} channels: {available_channels}')
    
    # Extract Nimbus raw data
    nimbus_raw = celltable.loc[:, available_channels].values
    
    # Create AnnData with raw Nimbus data first
    adata = sc.AnnData(nimbus_raw)
    adata.var_names = available_channels
    
    # Store raw Nimbus data in layer
    adata.layers['nimbus_raw'] = nimbus_raw.copy()
        
    # Add classic intensities if available
    if classic_intensities is not None:
        logging.info('Adding mean intensity over cell mask measurements to AnnData layers')
        
        # Merge classic data with celltable to ensure same cell order
        classic_merged = celltable[['ROI', 'ObjectNumber']].merge(
            classic_intensities, on=['ROI', 'ObjectNumber'], how='left'
        )
        
        # Extract classic raw data for available channels
        mean_intensities_raw_data = classic_merged.loc[:, available_channels].values
        adata.layers['mean_intensities_raw'] = mean_intensities_raw_data
        
        # Normalize classic data
        if normalisation:
            mean_intensities_normalized = normalise_markers(
                pd.DataFrame(mean_intensities_raw_data, columns=available_channels), normalisation
            ).values
            adata.layers['mean_intensities_normalized'] = mean_intensities_normalized
            logging.info(f'Mean intensities data normalized: {normalisation}')
        else:
            adata.layers['mean_intensities_normalized'] = mean_intensities_raw_data.copy()
    
    # Add expansion intensities if available
    if expansion_intensities is not None:
        logging.info('Adding expansion intensity measurements to AnnData layers')
        
        # Merge expansion data with celltable to ensure same cell order
        expansion_merged = celltable[['ROI', 'ObjectNumber']].merge(
            expansion_intensities, on=['ROI', 'ObjectNumber'], how='left'
        )
        
        # Extract expansion raw data for available channels
        expansion_intensities_raw_data = expansion_merged.loc[:, available_channels].values
        adata.layers['expansion_intensities_raw'] = expansion_intensities_raw_data
        
        # Normalize expansion data
        if normalisation:
            expansion_intensities_normalized = normalise_markers(
                pd.DataFrame(expansion_intensities_raw_data, columns=available_channels), normalisation
            ).values
            adata.layers['expansion_intensities_normalized'] = expansion_intensities_normalized
            logging.info(f'Expansion intensities data normalized: {normalisation}')
        else:
            adata.layers['expansion_intensities_normalized'] = expansion_intensities_raw_data.copy()
    
    # Add cellular obs from celltable
    non_channels = [x for x in celltable.columns if x not in expected_channels]
    for col in non_channels:
        adata.obs[col] = celltable[col].tolist()
    
    # Add metadata from metadata.csv (optional)
    metadata_path = metadata_folder / 'metadata.csv'
    if metadata_path.exists():
        expected_metadata_cols = {'description', 'width_um', 'height_um', 'mcd', 'source_file', 'file_type'}
        try:
            metadata = pd.read_csv(metadata_path, index_col='unstacked_data_folder')
        except Exception as exc:
            logging.warning(
                "Failed to parse metadata.csv at %s; skipping ROI metadata enrichment. Error: %s",
                metadata_path,
                exc,
            )
        else:
            present_expected_cols = expected_metadata_cols.intersection(set(metadata.columns))
            if not present_expected_cols:
                logging.warning(
                    "metadata.csv found at %s but contains none of the expected columns %s; skipping ROI metadata enrichment.",
                    metadata_path,
                    sorted(expected_metadata_cols),
                )
            else:
                adata.obs['ROI'] = adata.obs['ROI'].astype('category')

                if 'description' in metadata.columns:
                    adata.obs['ROI_name'] = adata.obs['ROI'].map(metadata['description'].to_dict())
                if 'width_um' in metadata.columns:
                    adata.obs['ROI_width'] = adata.obs['ROI'].map(metadata['width_um'].to_dict())
                if 'height_um' in metadata.columns:
                    adata.obs['ROI_height'] = adata.obs['ROI'].map(metadata['height_um'].to_dict())

                if 'mcd' in metadata.columns:
                    adata.obs['MCD_file'] = adata.obs['ROI'].map(metadata['mcd'].to_dict())
                elif 'source_file' in metadata.columns and 'file_type' in metadata.columns:
                    adata.obs['Source_file'] = adata.obs['ROI'].map(metadata['source_file'].to_dict())
                    adata.obs['File_type'] = adata.obs['ROI'].map(metadata['file_type'].to_dict())
                elif 'source_file' in metadata.columns or 'file_type' in metadata.columns:
                    logging.warning(
                        "metadata.csv at %s has only one of source_file/file_type; skipping source file metadata enrichment.",
                        metadata_path,
                    )
    else:
        logging.info("metadata.csv not found at %s; skipping ROI metadata enrichment.", metadata_path)
    
    # Add spatial coordinates
    adata.obsm['spatial'] = celltable[['X_loc', 'Y_loc']].to_numpy()
    
    # Process dictionary for additional metadata
    from .segmentation import convert_to_boolean
    dictionary_path = metadata_folder / 'dictionary.csv'
    if dictionary_path.exists():
        dictionary_file = pd.read_csv(dictionary_path, index_col='ROI')
        dictionary_file = convert_to_boolean(dictionary_file)
        
        cols = [x for x in dictionary_file.columns if 'Example' not in x and 'description' not in x]
        
        if len(cols) > 0:
            logging.info(f'Dictionary file found with columns: {cols}')
            adata.obs = adata.obs.copy()
            
            for c in cols:
                mapped_data = adata.obs['ROI'].map(dictionary_file[c].to_dict())
                adata.obs[c] = mapped_data.astype(dictionary_file[c].dtype)
            
            adata.obs = convert_to_boolean(adata.obs)
        else:
            logging.info('Dictionary file found but was empty')
    else:
        logging.info('No dictionary file found')
    
    # Remove specified channels
    if remove_channels:
        remove_channels_list = [
            channel for channel in adata.var_names
            if any(substring in channel for substring in remove_channels)
        ]
        if remove_channels_list:
            logging.info(f'Removing channels: {remove_channels_list}')
            keep_mask = [x not in remove_channels_list for x in adata.var_names]
            adata = adata[:, keep_mask]
    
    logging.info('AnnData created successfully with layers: %s', list(adata.layers.keys()))
    
    return adata


def _calc_padding_to_multiple(shape: Tuple[int, int], multiple: int = 16) -> Tuple[int, int]:
    """Return height/width padding needed to hit the requested multiple."""
    height, width = shape
    pad_y = (multiple - (height % multiple)) % multiple
    pad_x = (multiple - (width % multiple)) % multiple
    return pad_y, pad_x


def _pad_2d(array: np.ndarray, pad_y: int, pad_x: int, *, mode: str = "reflect") -> np.ndarray:
    """Pad a 2D array along Y/X axes."""
    if pad_y == 0 and pad_x == 0:
        return array
    pad_width = ((0, pad_y), (0, pad_x))
    if mode == "constant":
        return np.pad(array, pad_width, mode=mode, constant_values=0)
    return np.pad(array, pad_width, mode=mode)


def _crop_to_shape(array: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Crop array back to the requested shape."""
    target_y, target_x = shape
    return array[:target_y, :target_x]


def _predict_fovs_with_padding(
    nimbus,
    dataset,
    output_dir: str,
    suffix: str = ".tiff",
    save_predictions: bool = True,
    batch_size: int = 4,
    test_time_augmentation: bool = True,
    allow_resize_on_mismatch: bool = False,
    channels: Optional[Sequence[str]] = None,
    fovs: Optional[Sequence[str]] = None,
):
    """
    Run Nimbus predictions while padding/cropping arrays so the UNet's 16-pixel stride does not
    truncate ROI boundaries. Optionally fall back to resizing if a mismatch still occurs.
    """
    selected_channels = list(channels) if channels is not None else list(dataset.channels)
    selected_fovs = list(fovs) if fovs is not None else list(dataset.fovs)
    unknown_channels = sorted(set(selected_channels) - set(dataset.channels))
    unknown_fovs = sorted(set(selected_fovs) - set(dataset.fovs))
    if unknown_channels:
        raise ValueError(f"Unknown Nimbus prediction channel(s): {unknown_channels}")
    if unknown_fovs:
        raise ValueError(f"Unknown Nimbus prediction ROI(s): {unknown_fovs}")
    selected_fov_set = set(selected_fovs)
    fov_dict_list = []
    for fov_path, fov in zip(dataset.fov_paths, dataset.fovs):
        if fov not in selected_fov_set:
            continue
        logging.info("Predicting %s...", fov_path)
        out_fov_path = os.path.join(os.path.normpath(output_dir), os.path.basename(fov_path).replace(suffix, ""))
        df_fov = pd.DataFrame()
        instance_mask = dataset.get_segmentation(fov)
        mask_shape = instance_mask.shape
        # Nimbus UNet downsamples four times (stride 16); pre-pad so nothing gets dropped.
        pad_y, pad_x = _calc_padding_to_multiple(mask_shape, multiple=16)
        padded_mask = _pad_2d(instance_mask, pad_y, pad_x, mode="constant")  # zero-pad preserves ids
        for channel_name in tqdm(selected_channels, desc=f"{fov}", leave=False):
            mplex_img = dataset.get_channel_normalized(fov, channel_name)
            if pad_y or pad_x:
                # Reflect-pad intensity image so Nimbus sees mirrored context at the border
                mplex_img = _pad_2d(mplex_img, pad_y, pad_x, mode="reflect")
            input_data = prepare_input_data(mplex_img, padded_mask)
            if dataset.magnification != nimbus.model_magnification:
                # Nimbus expects its model_magnification; scale both channels of the input tensor
                scale = nimbus.model_magnification / dataset.magnification
                input_data = np.squeeze(input_data)
                _, h, w = input_data.shape
                img = cv2.resize(input_data[0], [int(w * scale), int(h * scale)])
                binary_mask = cv2.resize(input_data[1], [int(w * scale), int(h * scale)], interpolation=0)
                input_data = np.stack([img, binary_mask], axis=0)[np.newaxis, ...]
            if test_time_augmentation:
                # Average predictions over flips/rotations for robustness
                prediction = test_time_aug(
                    input_data,
                    channel_name,
                    nimbus,
                    dataset.normalization_dict,
                    batch_size=batch_size,
                    clip_values=dataset.clip_values,
                )
            else:
                prediction = nimbus.predict_segmentation(input_data)
            if not isinstance(prediction, np.ndarray):
                prediction = prediction.cpu().numpy()
            prediction = np.squeeze(prediction)
            if dataset.magnification != nimbus.model_magnification:
                # Return to the dataset magnification before removing padding
                prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)
            if pad_y or pad_x:
                prediction = _crop_to_shape(prediction, mask_shape)  # drop Nimbus-only padding
            if prediction.shape != instance_mask.shape:
                msg = (
                    f"Prediction/mask shape mismatch for {fov} channel {channel_name}: "
                    f"{prediction.shape} vs {instance_mask.shape}"
                )
                if allow_resize_on_mismatch:
                    logging.warning("%s -> resizing prediction (Nimbus allow_prediction_resize=True)", msg)
                    prediction = cv2.resize(
                        prediction,
                        (instance_mask.shape[1], instance_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    raise ValueError(msg)
            df = pd.DataFrame(segment_mean(instance_mask, prediction))
            if df_fov.empty:
                df_fov["label"] = df["label"]
                df_fov["fov"] = os.path.basename(fov_path)
            df_fov[channel_name] = df["intensity_mean"]
            if save_predictions:
                # Persist per-channel confidence map for downstream QC
                os.makedirs(out_fov_path, exist_ok=True)
                pred_int = (prediction * 255.0).astype(np.uint8)
                io.imsave(
                    os.path.join(out_fov_path, channel_name + suffix),
                    pred_int,
                    check_contrast=False,
                )
        fov_dict_list.append(df_fov)
    if not fov_dict_list:
        raise ValueError("No Nimbus ROI predictions were generated.")
    return pd.concat(fov_dict_list, ignore_index=True)


def main() -> None:
    from SpatialBiologyToolkit.reporting import project_asset_path

    pipeline_stage = "NimbusSegmentation"
    config = process_config_with_overrides()
    setup_logging(config.get("logging", {}), pipeline_stage)

    general_config = GeneralConfig(**filter_config_for_dataclass(config.get("general", {}), GeneralConfig))
    seg_config = SegmentationConfig(**filter_config_for_dataclass(config.get("segmentation", {}), SegmentationConfig))
    nimbus_config = NimbusConfig(**filter_config_for_dataclass(config.get("nimbus", {}), NimbusConfig))
    stage_config = {
        "segmentation": seg_config,
        "nimbus": nimbus_config,
    }
    _, canonical_anndata_path, skip_stage, _ = load_pipeline_anndata(
        general_config=general_config,
        stage_name=pipeline_stage,
        stage_config=stage_config,
        allow_missing=True,
    )
    if skip_stage:
        logging.info("Skipping NimbusSegmentation stage based on AnnData stage policy.")
        return

    configured_normalization_path: Optional[Path] = None
    if nimbus_config.normalization_dict_path:
        configured_normalization_path = project_asset_path(
            nimbus_config.normalization_dict_path
        )
        resolve_normalization_input_path(
            nimbus_config.output_dir,
            configured_path=configured_normalization_path,
            reuse_saved=nimbus_config.reuse_saved_normalization,
        )

    metadata_folder = Path(general_config.metadata_folder)
    panel = _load_panel(metadata_folder, nimbus_config)

    mask_lookup = _discover_masks(Path(general_config.masks_folder), nimbus_config.mask_extensions)
    if not mask_lookup:
        raise FileNotFoundError(f"No mask files found in {general_config.masks_folder}")

    rois = _filter_rois_by_metadata(mask_lookup, metadata_folder / "metadata.csv")
    if not rois:
        raise ValueError("No ROIs to process after applying metadata import filters")

    (
        valid_rois,
        channel_paths,
        roi_image_roots,
        missing_summary,
        expected_channels,
        channels_for_model,
        ) = _resolve_channel_paths(rois, panel, general_config, nimbus_config)

    if not valid_rois:
        raise ValueError("No ROIs with usable channel images were found for Nimbus")

    if missing_summary:
        logging.warning("Channels missing for some ROIs (files not found): %s", missing_summary)

    if not channels_for_model:
        raise ValueError("No channels were available across all ROIs for Nimbus inference")

    excluded_channels = sorted(set(expected_channels) - set(channels_for_model))
    if excluded_channels:
        logging.warning(
            "Dropping %d channel(s) absent in at least one ROI: %s", len(excluded_channels), excluded_channels
        )

    min_cell_area = _coerce_optional_area_bound("nimbus.min_cell_area", nimbus_config.min_cell_area)
    max_cell_area = _coerce_optional_area_bound("nimbus.max_cell_area", nimbus_config.max_cell_area)
    if min_cell_area is not None and max_cell_area is not None and min_cell_area > max_cell_area:
        raise ValueError(
            f"nimbus.min_cell_area ({min_cell_area:g}) cannot be greater than "
            f"nimbus.max_cell_area ({max_cell_area:g})"
        )

    mask_lookup = {roi: mask_lookup[roi] for roi in valid_rois}
    fov_paths = [roi_image_roots[roi] for roi in valid_rois]

    clip_values = tuple(nimbus_config.normalization_clip) if nimbus_config.normalization_clip else (0.0, 2.0)
    dataset = ToolkitNimbusDataset(
        fov_paths=fov_paths,
        channels=channels_for_model,
        channel_paths=channel_paths,
        mask_lookup=mask_lookup,
        suffix=".tiff",
        magnification=nimbus_config.dataset_magnification,
        output_dir=nimbus_config.output_dir,
        qc_folder=general_config.qc_folder,
        normalization_jobs=nimbus_config.normalization_jobs,
        clip_values=clip_values,
        normalization_min_value=nimbus_config.normalization_min_value,
        normalization_lower_threshold=nimbus_config.normalization_lower_threshold,
        mask_boundary_offset_pixels=nimbus_config.mask_boundary_offset_pixels,
        min_cell_area=min_cell_area,
        max_cell_area=max_cell_area,
    )

    if nimbus_config.mask_boundary_offset_pixels != 0:
        logging.info(
            "Applying Nimbus mask boundary offset of %d pixel(s): positive expands cells, negative shrinks cells.",
            int(nimbus_config.mask_boundary_offset_pixels),
        )

    if min_cell_area is not None or max_cell_area is not None:
        logging.info(
            "Filtering Nimbus masks by post-offset cell area: min_cell_area=%s, max_cell_area=%s.",
            "None" if min_cell_area is None else f"{min_cell_area:g}",
            "None" if max_cell_area is None else f"{max_cell_area:g}",
        )

    reuse_existing_master_celltables = bool(nimbus_config.use_existing_master_celltables)
    mask_geometry_changed = (
        nimbus_config.mask_boundary_offset_pixels != 0
        or min_cell_area is not None
        or max_cell_area is not None
    )
    if reuse_existing_master_celltables and mask_geometry_changed:
        logging.warning(
            "Disabling use_existing_master_celltables because mask_boundary_offset_pixels=%d, "
            "min_cell_area=%s, and max_cell_area=%s can change mask geometry or cell inclusion "
            "and existing master cell tables may be stale.",
            int(nimbus_config.mask_boundary_offset_pixels),
            "None" if min_cell_area is None else f"{min_cell_area:g}",
            "None" if max_cell_area is None else f"{max_cell_area:g}",
        )
        reuse_existing_master_celltables = False

    dataset.prepare_normalization_dict(
        quantile=nimbus_config.normalization_quantile,
        clip_values=clip_values,
        n_subset=nimbus_config.normalization_subset,
        multiprocessing=nimbus_config.normalization_jobs > 1,
        reuse_saved=nimbus_config.reuse_saved_normalization,
        normalization_file=configured_normalization_path,
    )

    # Early exit if only normalization dict and QC are requested
    if nimbus_config.norm_dict_qc_only:
        logging.info("norm_dict_qc_only=True: Stopping after normalization dictionary and QC generation.")
        logging.info(f"Normalization dictionary saved to: {dataset.normalization_dict_path}")
        logging.info(f"QC images saved to: {general_config.qc_folder}/nimbus_normalization_qc/")
        return

    master_path = _resolve_master_celltable_path(
        nimbus_config.master_celltable or seg_config.celltable_output,
        nimbus_config.output_dir,
    )

    merged_celltable: Optional[pd.DataFrame] = None
    predicted_channels: List[str] = []

    if reuse_existing_master_celltables:
        merged_celltable = _load_existing_master_celltable(master_path, "Nimbus")
        if merged_celltable is None:
            logging.info("Nimbus master cell table not found or invalid; running Nimbus inference.")
        else:
            predicted_channels = [c for c in expected_channels if c in merged_celltable.columns]
            roi_count = merged_celltable["ROI"].nunique() if "ROI" in merged_celltable.columns else 0
            logging.info(
                "Using existing Nimbus master cell table with %d cells across %d ROI(s) and %d predicted channel(s)",
                len(merged_celltable),
                roi_count,
                len(predicted_channels),
            )

    if merged_celltable is None:
        nimbus = Nimbus(
            dataset=dataset,
            output_dir=nimbus_config.output_dir,
            save_predictions=nimbus_config.save_prediction_maps,
            batch_size=nimbus_config.batch_size,
            test_time_aug=nimbus_config.test_time_augmentation,
            model_magnification=nimbus_config.model_magnification,
            device=nimbus_config.device,
            checkpoint=nimbus_config.checkpoint,
        )

        # Run Nimbus predictions
        nimbus_df = _prepare_nimbus_output(
            _predict_fovs_with_padding(
                nimbus=nimbus,
                dataset=dataset,
                output_dir=nimbus_config.output_dir,
                suffix=".tiff",
                save_predictions=nimbus_config.save_prediction_maps,
                batch_size=nimbus_config.batch_size,
                test_time_augmentation=nimbus_config.test_time_augmentation,
                allow_resize_on_mismatch=nimbus_config.allow_prediction_resize,
            )
        )
        
        # Build mask features
        mask_features = _build_mask_features(
            mask_lookup,
            valid_rois,
            mask_boundary_offset_pixels=nimbus_config.mask_boundary_offset_pixels,
            min_cell_area=min_cell_area,
            max_cell_area=max_cell_area,
        )
        
        # Merge Nimbus predictions with mask features
        merged_celltable, predicted_channels = _merge_with_masks(
            mask_features, nimbus_df, expected_channels, channels_for_model, seg_config.allow_missing_channels
        )

        logging.info(
            "Nimbus produced %d cells across %d ROI(s) with %d predicted channel(s)",
            len(merged_celltable),
            len(valid_rois),
            len(predicted_channels),
        )

        if seg_config.create_master_cell_table:
            master_path.parent.mkdir(parents=True, exist_ok=True)
            merged_celltable.to_csv(master_path, index=False)
            logging.info("Saved master Nimbus cell table to %s", master_path)
        else:
            logging.info("Skipping master cell table per config")
    elif seg_config.create_master_cell_table:
        logging.info("Using existing Nimbus master cell table; skipping save.")
    
    # Extract classic mean intensities if requested
    classic_intensities = None
    if nimbus_config.extract_classic_intensities:
        classic_master_path = _resolve_master_celltable_path(
            nimbus_config.master_classic_celltable,
            nimbus_config.output_dir,
        )
        if reuse_existing_master_celltables:
            classic_intensities = _load_existing_master_celltable(classic_master_path, "classic intensity")
            if classic_intensities is None:
                logging.info("Classic master cell table not found or invalid; running classic extraction.")
        if classic_intensities is None:
            logging.info("Extracting classic mean intensities over masks")
            classic_intensities = _extract_classic_intensities(
                mask_lookup=mask_lookup,
                rois=valid_rois,
                channel_paths=channel_paths,
                expected_channels=expected_channels,
                mask_boundary_offset_pixels=nimbus_config.mask_boundary_offset_pixels,
                min_cell_area=min_cell_area,
                max_cell_area=max_cell_area,
            )
            logging.info(f"Classic extraction complete for {len(classic_intensities)} cells")
            if seg_config.create_master_cell_table:
                classic_master_path.parent.mkdir(parents=True, exist_ok=True)
                classic_intensities.to_csv(classic_master_path, index=False)
                logging.info("Saved master classic intensity cell table to %s", classic_master_path)
            else:
                logging.info("Skipping master classic intensity cell table per config")
        elif seg_config.create_master_cell_table:
            logging.info("Using existing master classic intensity cell table; skipping save.")
    else:
        logging.info("Skipping classic intensity extraction per config")
    
    # Extract expansion intensities if requested
    expansion_intensities = None
    if nimbus_config.extract_expansion_intensities:
        expansion_master_path = _resolve_master_celltable_path(
            nimbus_config.master_expansion_celltable,
            nimbus_config.output_dir,
        )
        if reuse_existing_master_celltables:
            expansion_intensities = _load_existing_master_celltable(expansion_master_path, "expansion intensity")
            if expansion_intensities is None:
                logging.info("Expansion master cell table not found or invalid; running expansion extraction.")
        if expansion_intensities is None:
            logging.info(f"Extracting expansion intensities with {nimbus_config.expansion_pixels} pixel expansion")
            expansion_intensities = _extract_expansion_intensities(
                mask_lookup=mask_lookup,
                rois=valid_rois,
                channel_paths=channel_paths,
                expected_channels=expected_channels,
                expansion_pixels=nimbus_config.expansion_pixels,
                mask_boundary_offset_pixels=nimbus_config.mask_boundary_offset_pixels,
                min_cell_area=min_cell_area,
                max_cell_area=max_cell_area,
                n_jobs=nimbus_config.expansion_jobs,
            )
            logging.info(f"Expansion extraction complete for {len(expansion_intensities)} cells")
            if seg_config.create_master_cell_table:
                expansion_master_path.parent.mkdir(parents=True, exist_ok=True)
                expansion_intensities.to_csv(expansion_master_path, index=False)
                logging.info("Saved master expansion intensity cell table to %s", expansion_master_path)
            else:
                logging.info("Skipping master expansion intensity cell table per config")
        elif seg_config.create_master_cell_table:
            logging.info("Using existing master expansion intensity cell table; skipping save.")
    else:
        logging.info("Skipping expansion intensity extraction per config")

    roi_output_dir = Path(general_config.celltable_folder)
    if nimbus_config.roi_table_subfolder:
        roi_output_dir = roi_output_dir / nimbus_config.roi_table_subfolder

    if seg_config.create_roi_cell_tables:
        _save_roi_tables(merged_celltable, roi_output_dir, nimbus_config.roi_table_prefix or "")
        logging.info("Saved ROI-level Nimbus cell tables to %s", roi_output_dir)
    else:
        logging.info("Skipping ROI-level cell tables per config")

    if seg_config.create_anndata:
        if nimbus_config.anndata_output and str(nimbus_config.anndata_output) != str(general_config.anndata_path):
            logging.warning(
                "nimbus.anndata_output is deprecated for primary pipeline flow. "
                "Using general.anndata_path=%s instead.",
                general_config.anndata_path,
            )
        if seg_config.anndata_save_path and str(seg_config.anndata_save_path) != str(general_config.anndata_path):
            logging.warning(
                "segmentation.anndata_save_path is deprecated for primary pipeline flow. "
                "Using general.anndata_path=%s instead.",
                general_config.anndata_path,
            )

        # Create AnnData with layers for Nimbus, classic, and expansion data
        adata = _create_anndata_with_layers(
            celltable=merged_celltable,
            classic_intensities=classic_intensities,
            expansion_intensities=expansion_intensities,
            metadata_folder=metadata_folder,
            normalisation=seg_config.marker_normalisation,
            remove_channels=seg_config.remove_channels_list,
            expected_channels=expected_channels,
        )
        
        # Handle remove_and_store_markers: separate suboptimal markers into separate AnnData
        if seg_config.remove_and_store_markers:
            # Filter markers that actually exist in the dataset
            markers_to_remove = [m for m in seg_config.remove_and_store_markers if m in adata.var_names]
            
            if markers_to_remove:
                logging.info(f"Separating {len(markers_to_remove)} markers into separate AnnData: {markers_to_remove}")
                
                # Create a subset AnnData with only the markers to be removed
                adata_removed = adata[:, markers_to_remove].copy()
                
                # Save the removed markers AnnData
                removed_path = Path(seg_config.removed_markers_anndata_path)
                removed_path.parent.mkdir(parents=True, exist_ok=True)
                adata_removed.write_h5ad(removed_path)
                logging.info(f"Saved removed markers AnnData to {removed_path}")
                logging.info(f"Removed markers AnnData contains {adata_removed.n_obs} cells × {adata_removed.n_vars} markers")
                
                # Remove these markers from the main AnnData
                keep_markers = [m for m in adata.var_names if m not in markers_to_remove]
                adata = adata[:, keep_markers].copy()
                logging.info(f"Removed {len(markers_to_remove)} markers from main AnnData. Remaining: {adata.n_vars} markers")
            else:
                logging.info("No markers from remove_and_store_markers list found in dataset")
        
        anndata_path = save_pipeline_anndata(
            adata=adata,
            general_config=general_config,
            stage_name=pipeline_stage,
            stage_config=stage_config,
            override_path=str(canonical_anndata_path),
            extra_details={"n_cells": int(adata.n_obs), "n_markers": int(adata.n_vars)},
        )
        logging.info("Saved AnnData to %s", anndata_path)
        logging.info("AnnData structure: .X (normalized Nimbus), layers: %s", list(adata.layers.keys()))
    else:
        logging.info("Skipping AnnData creation per config")


if __name__ == "__main__":
    main()

