"""
Nimbus-powered drop-in for segmentation.py.
Uses Nimbus-Inference on existing masks/images to build cell tables and an AnnData object.
"""
from __future__ import annotations

import json
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

try:
    from nimbus_inference.nimbus import Nimbus
    from nimbus_inference.utils import (
        MultiplexDataset,
        prepare_input_data,
        prepare_normalization_dict,
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
    process_config_with_overrides,
    setup_logging,
)
from .segmentation import create_anndata, normalise_markers


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
        suffix_match: Optional[str] = None,
    ) -> None:
        self._channels = sorted(channels)
        self._channel_paths = channel_paths
        self._mask_lookup = mask_lookup
        self.qc_folder = qc_folder
        self.normalization_n_jobs = max(1, int(normalization_jobs))
        self.clip_values = tuple(clip_values)
        self.normalization_min_value = float(normalization_min_value)
        self._suffix_match = suffix_match or suffix

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
        if img.ndim == 2:
            return img
        # If channel dimension leaked through (e.g., multichannel format), take first plane
        return np.squeeze(img)[0] if np.squeeze(img).ndim == 3 else np.squeeze(img)

    def get_segmentation(self, fov: str):  # type: ignore[override]
        roi = Path(fov).name
        mask = io.imread(self._mask_lookup[roi])
        mask = np.squeeze(mask)
        if mask.ndim == 3:
            mask = np.squeeze(mask[..., 0])

        # Align mask to the reference channel image shape if needed
        ref_path = next(iter(self._channel_paths[roi].values()))
        ref_img = io.imread(ref_path)
        ref_shape = np.squeeze(ref_img).shape[-2:] if ref_img.ndim >= 2 else ref_img.shape

        if mask.shape != tuple(ref_shape):
            raise ValueError(f"Mask/Image shape mismatch for ROI {roi}: {mask.shape} vs {ref_shape}")
        return mask.astype(np.uint32)

    def prepare_normalization_dict(  # type: ignore[override]
        self,
        quantile: float = 0.999,
        clip_values: Sequence[float] = (0, 2),
        n_subset: int = 10,
        multiprocessing: bool = False,  # kept for API compatibility
        reuse_saved: bool = False,
    ):
        """
        Compute per-channel normalization using ALL FOVs and only in-mask pixels.
        Also writes a QC gallery and QC histograms.
        
        If reuse_saved=True and a normalization_dict.json exists, it will be loaded and reused
        (allowing manual tweaking). QC plots will still be generated with the loaded values.
        """
        self.clip_values = tuple(clip_values)
        self.normalization_dict_path = os.path.join(self.output_dir, "normalization_dict.json")

        if os.path.exists(self.normalization_dict_path) and reuse_saved:
            logging.info(f"Found existing normalization dictionary at {self.normalization_dict_path}")
            logging.info("Reusing saved normalization values (reuse_saved_normalization=True)")
            with open(self.normalization_dict_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            # Load values as-is without applying minimum constraint (preserves manual edits)
            self.normalization_dict = {k: float(v) for k, v in data.items()}
            logging.info(f"Loaded {len(self.normalization_dict)} channel normalization values (manual edits preserved)")
        else:
            norm_vals: Dict[str, List[float]] = {ch: [] for ch in self._channels}
            for fov in self.fovs:
                mask = self.get_segmentation(fov)
                mask_bool = mask > 0
                if not np.any(mask_bool):
                    continue
                for ch in self._channels:
                    img = self.get_channel(fov, ch).astype(float)
                    foreground = img[mask_bool]
                    if foreground.size:
                        norm_vals[ch].append(float(np.quantile(foreground, quantile)))

            self.normalization_dict = {}
            for ch, vals in norm_vals.items():
                if vals:
                    computed_val = float(np.mean(vals))
                    self.normalization_dict[ch] = max(computed_val, self.normalization_min_value)
                else:
                    self.normalization_dict[ch] = self.normalization_min_value

            norm_str = {k: str(v) for k, v in self.normalization_dict.items()}
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.normalization_dict_path, "w", encoding="utf-8") as handle:
                json.dump(norm_str, handle)

        # QC: histograms of raw norms and cell-level positivity, plus gallery of normalized images
        os.makedirs(os.path.join(self.qc_folder, "nimbus_normalization_qc"), exist_ok=True)
        norm_hist_dir = os.path.join(self.qc_folder, "nimbus_normalization_qc", "norm_hists")
        pos_hist_dir = os.path.join(self.qc_folder, "nimbus_normalization_qc", "cellpos_hists")
        os.makedirs(norm_hist_dir, exist_ok=True)
        os.makedirs(pos_hist_dir, exist_ok=True)

        # Histogram of per-ROI norms with marker at final value
        for ch, vals in self.normalization_dict.items():
            pass

        upper_clip = self.clip_values[1] if len(self.clip_values) > 1 else 2.0

        def _save_hist(data: List[float], marker: Optional[float], out_path: str, xlabel: str, title: str):
            if not data:
                return
            plt.figure(figsize=(4, 3))
            plt.hist(data, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
            if marker is not None:
                plt.axvline(marker, color="red", linestyle="--", label=f"marker={marker:.3g}")
            plt.xlabel(xlabel)
            plt.ylabel("Count")
            plt.title(title)
            if marker is not None:
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
                img_raw = self.get_channel(fov, ch).astype(float)
                norm_vals[ch].append(float(np.quantile(img_raw[mask_bool], quantile)))

                img = np.clip(img_raw / norm, 0, upper_clip)
                props = regionprops_table(label_image=labels, intensity_image=img, properties=["intensity_mean"])
                means = props.get("intensity_mean", [])
                if len(means) > 0:
                    cell_pos_props[ch].append(float(np.mean(np.array(means) > 1.0)))

        for ch in self._channels:
            final_val = self.normalization_dict.get(ch, 1.0)
            _save_hist(
                norm_vals.get(ch, []),
                final_val,
                os.path.join(norm_hist_dir, f"{ch}.png"),
                xlabel=f"{ch} per-ROI quantiles",
                title=f"{ch} norm values (final {final_val:.3g})",
            )
            _save_hist(
                cell_pos_props.get(ch, []),
                None if not cell_pos_props.get(ch, []) else np.mean(cell_pos_props[ch]),
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
                
                # Create figure with 3 columns (unmasked, masked, clip diagnostic) and one row per FOV
                fig, axs = plt.subplots(len(qc_fovs), 3, figsize=(15, 5 * len(qc_fovs)), dpi=100)
                
                # Handle single ROI case (axs won't be 2D)
                if len(qc_fovs) == 1:
                    axs = np.array([axs])
                
                for row_idx, fov in enumerate(qc_fovs):
                    mask = self.get_segmentation(fov)
                    mask_bool = mask > 0
                    
                    img_raw = self.get_channel(fov, ch).astype(float)
                    img = np.clip(img_raw / norm, 0, upper_clip)
                    img_masked = img * mask_bool
                    
                    # Left column: unmasked
                    im1 = axs[row_idx, 0].imshow(img, vmin=0, vmax=upper_clip, cmap='gray')
                    divider1 = make_axes_locatable(axs[row_idx, 0])
                    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im1, cax=cax1, orientation='vertical')
                    axs[row_idx, 0].set_ylabel(fov, fontsize=8)
                    if row_idx == 0:
                        axs[row_idx, 0].set_title('Unmasked', fontsize=10)
                    
                    # Middle column: masked
                    im2 = axs[row_idx, 1].imshow(img_masked, vmin=0, vmax=upper_clip, cmap='gray')
                    divider2 = make_axes_locatable(axs[row_idx, 1])
                    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im2, cax=cax2, orientation='vertical')
                    if row_idx == 0:
                        axs[row_idx, 1].set_title('Masked (cells only)', fontsize=10)
                    
                    # Right column: clipping diagnostic
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
                    
                    axs[row_idx, 2].imshow(clip_diag)
                    axs[row_idx, 2].set_xticks([])
                    axs[row_idx, 2].set_yticks([])
                    
                    # Add text overlay with clip value and counts
                    n_clipped = np.sum(clipped_mask)
                    n_zero = np.sum(zero_mask)
                    pct_clipped = 100.0 * n_clipped / img.size
                    pct_zero = 100.0 * n_zero / img.size
                    overlay_text = f'clip={upper_clip:.2f}\nred(clip): {pct_clipped:.1f}%\nblue(zero): {pct_zero:.1f}%'
                    axs[row_idx, 2].text(
                        0.02, 0.98, overlay_text,
                        transform=axs[row_idx, 2].transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                    )
                    if row_idx == 0:
                        axs[row_idx, 2].set_title('Clip Diagnostic\n(red=clipped, blue=zero)', fontsize=10)
                
                fig.suptitle(f'{ch} (norm={norm:.3g})', fontsize=12, fontweight='bold')
                plt.tight_layout()
                fig.savefig(os.path.join(qc_gallery_dir, f'{ch}.png'), bbox_inches='tight')
                plt.close(fig)
                
            logging.info(f"Normalization QC galleries saved to: {qc_gallery_dir}")


def _load_panel(metadata_folder: Path) -> pd.DataFrame:
    panel = pd.read_csv(metadata_folder / "panel.csv")
    panel["channel_label"] = [re.sub(r"\\W+", "", str(x)) for x in panel["channel_label"]]
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


def _build_mask_features(mask_lookup: Dict[str, Path], rois: List[str]) -> pd.DataFrame:
    circ = lambda r: (4 * np.pi * r.area) / (r.perimeter * r.perimeter) if r.perimeter > 0 else 0
    frames: List[pd.DataFrame] = []
    for roi in rois:
        mask = io.imread(mask_lookup[roi])
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
) -> pd.DataFrame:
    """
    Extract classic mean intensities by measuring directly over masks (like original segmentation.py).
    Returns DataFrame with ROI, ObjectNumber, and channel intensities.
    """
    frames: List[pd.DataFrame] = []
    
    for roi in tqdm(rois, desc="Classic intensity extraction"):
        mask = io.imread(mask_lookup[roi])
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
                
                # Calculate mean intensity for each label
                mean_intensities = [region.mean_intensity for region in regionprops(mask, image)]
                roi_df[channel] = mean_intensities
                
            except Exception as e:
                logging.warning(f"Error extracting classic intensity for {roi}, channel {channel}: {e}")
                roi_df[channel] = np.nan
        
        roi_df["ROI"] = roi
        frames.append(roi_df)
    
    return pd.concat(frames, ignore_index=True)


def _prepare_nimbus_output(cell_table: pd.DataFrame) -> pd.DataFrame:
    if cell_table is None or cell_table.empty:
        raise ValueError("Nimbus returned an empty cell table")
    renamed = cell_table.rename(columns={"label": "ObjectNumber", "fov": "ROI"})
    renamed["ROI"] = renamed["ROI"].astype(str)
    renamed["ObjectNumber"] = renamed["ObjectNumber"].astype(int)
    return renamed


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
    
    # Add cellular obs from celltable
    non_channels = [x for x in celltable.columns if x not in expected_channels]
    for col in non_channels:
        adata.obs[col] = celltable[col].tolist()
    
    # Add metadata from metadata.csv
    metadata = pd.read_csv(metadata_folder / 'metadata.csv', index_col='unstacked_data_folder')
    adata.obs['ROI'] = adata.obs['ROI'].astype('category')
    adata.obs['ROI_name'] = adata.obs['ROI'].map(metadata['description'].to_dict())
    adata.obs['ROI_width'] = adata.obs['ROI'].map(metadata['width_um'].to_dict())
    adata.obs['ROI_height'] = adata.obs['ROI'].map(metadata['height_um'].to_dict())
    
    if 'mcd' in metadata.columns:
        adata.obs['MCD_file'] = adata.obs['ROI'].map(metadata['mcd'].to_dict())
    elif 'source_file' in metadata.columns:
        adata.obs['Source_file'] = adata.obs['ROI'].map(metadata['source_file'].to_dict())
        adata.obs['File_type'] = adata.obs['ROI'].map(metadata['file_type'].to_dict())
    
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


def _predict_fovs_shape_guard(
    nimbus,
    dataset,
    output_dir: str,
    suffix: str = ".tiff",
    save_predictions: bool = True,
    batch_size: int = 4,
    test_time_augmentation: bool = True,
):
    """
    Nimbus predict_fovs with a shape guard: if prediction and mask differ in shape, resize prediction to mask.
    """
    fov_dict_list = []
    for fov_path, fov in zip(dataset.fov_paths, dataset.fovs):
        logging.info("Predicting %s...", fov_path)
        out_fov_path = os.path.join(os.path.normpath(output_dir), os.path.basename(fov_path).replace(suffix, ""))
        df_fov = pd.DataFrame()
        instance_mask = dataset.get_segmentation(fov)
        for channel_name in tqdm(dataset.channels, desc=f"{fov}", leave=False):
            mplex_img = dataset.get_channel_normalized(fov, channel_name)
            input_data = prepare_input_data(mplex_img, instance_mask)
            if dataset.magnification != nimbus.model_magnification:
                scale = nimbus.model_magnification / dataset.magnification
                input_data = np.squeeze(input_data)
                _, h, w = input_data.shape
                img = cv2.resize(input_data[0], [int(w * scale), int(h * scale)])
                binary_mask = cv2.resize(input_data[1], [int(w * scale), int(h * scale)], interpolation=0)
                input_data = np.stack([img, binary_mask], axis=0)[np.newaxis, ...]
            if test_time_augmentation:
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
                prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)
            if prediction.shape != instance_mask.shape:
                logging.info(
                    "Prediction/mask shape mismatch for %s channel %s: %s vs %s -> resizing prediction",
                    fov,
                    channel_name,
                    prediction.shape,
                    instance_mask.shape,
                )
                prediction = cv2.resize(
                    prediction,
                    (instance_mask.shape[1], instance_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            df = pd.DataFrame(segment_mean(instance_mask, prediction))
            if df_fov.empty:
                df_fov["label"] = df["label"]
                df_fov["fov"] = os.path.basename(fov_path)
            df_fov[channel_name] = df["intensity_mean"]
            if save_predictions:
                os.makedirs(out_fov_path, exist_ok=True)
                pred_int = (prediction * 255.0).astype(np.uint8)
                io.imsave(
                    os.path.join(out_fov_path, channel_name + suffix),
                    pred_int,
                    check_contrast=False,
                )
        fov_dict_list.append(df_fov)
    return pd.concat(fov_dict_list, ignore_index=True)


def main() -> None:
    pipeline_stage = "NimbusSegmentation"
    config = process_config_with_overrides()
    setup_logging(config.get("logging", {}), pipeline_stage)

    general_config = GeneralConfig(**filter_config_for_dataclass(config.get("general", {}), GeneralConfig))
    seg_config = SegmentationConfig(**filter_config_for_dataclass(config.get("segmentation", {}), SegmentationConfig))
    nimbus_config = NimbusConfig(**filter_config_for_dataclass(config.get("nimbus", {}), NimbusConfig))

    metadata_folder = Path(general_config.metadata_folder)
    panel = _load_panel(metadata_folder)

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
    )

    dataset.prepare_normalization_dict(
        quantile=nimbus_config.normalization_quantile,
        clip_values=clip_values,
        n_subset=nimbus_config.normalization_subset,
        multiprocessing=nimbus_config.normalization_jobs > 1,
        reuse_saved=nimbus_config.reuse_saved_normalization,
    )

    # Early exit if only normalization dict and QC are requested
    if nimbus_config.norm_dict_qc_only:
        logging.info("norm_dict_qc_only=True: Stopping after normalization dictionary and QC generation.")
        logging.info(f"Normalization dictionary saved to: {dataset.normalization_dict_path}")
        logging.info(f"QC images saved to: {general_config.qc_folder}/nimbus_normalization_qc/")
        return

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
        _predict_fovs_shape_guard(
            nimbus=nimbus,
            dataset=dataset,
            output_dir=nimbus_config.output_dir,
            suffix=".tiff",
            save_predictions=nimbus_config.save_prediction_maps,
            batch_size=nimbus_config.batch_size,
            test_time_augmentation=nimbus_config.test_time_augmentation,
        )
    )
    
    # Build mask features
    mask_features = _build_mask_features(mask_lookup, valid_rois)
    
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
    
    # Extract classic mean intensities if requested
    classic_intensities = None
    if nimbus_config.extract_classic_intensities:
        logging.info("Extracting classic mean intensities over masks")
        classic_intensities = _extract_classic_intensities(
            mask_lookup=mask_lookup,
            rois=valid_rois,
            channel_paths=channel_paths,
            expected_channels=expected_channels,
        )
        logging.info(f"Classic extraction complete for {len(classic_intensities)} cells")
    else:
        logging.info("Skipping classic intensity extraction per config")

    roi_output_dir = Path(general_config.celltable_folder)
    if nimbus_config.roi_table_subfolder:
        roi_output_dir = roi_output_dir / nimbus_config.roi_table_subfolder

    if seg_config.create_roi_cell_tables:
        _save_roi_tables(merged_celltable, roi_output_dir, nimbus_config.roi_table_prefix or "")
        logging.info("Saved ROI-level Nimbus cell tables to %s", roi_output_dir)
    else:
        logging.info("Skipping ROI-level cell tables per config")

    master_path = Path(nimbus_config.master_celltable or seg_config.celltable_output)
    if not master_path.is_absolute():
        master_path = Path(nimbus_config.output_dir) / master_path

    if seg_config.create_master_cell_table:
        master_path.parent.mkdir(parents=True, exist_ok=True)
        merged_celltable.to_csv(master_path, index=False)
        logging.info("Saved master Nimbus cell table to %s", master_path)
    else:
        logging.info("Skipping master cell table per config")

    if seg_config.create_anndata:
        anndata_path = Path(nimbus_config.anndata_output or seg_config.anndata_save_path)

        # Create AnnData with layers for both Nimbus and classic data
        adata = _create_anndata_with_layers(
            celltable=merged_celltable,
            classic_intensities=classic_intensities,
            metadata_folder=metadata_folder,
            normalisation=seg_config.marker_normalisation,
            remove_channels=seg_config.remove_channels_list,
            expected_channels=expected_channels,
        )
        
        anndata_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(anndata_path)
        logging.info("Saved AnnData to %s", anndata_path)
        logging.info("AnnData structure: .X (normalized Nimbus), layers: %s", list(adata.layers.keys()))
    else:
        logging.info("Skipping AnnData creation per config")


if __name__ == "__main__":
    main()








