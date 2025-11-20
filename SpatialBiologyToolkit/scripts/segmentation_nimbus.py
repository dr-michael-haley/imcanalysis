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

import numpy as np
import pandas as pd
from alpineer import io_utils
from skimage import io
from skimage.measure import regionprops

try:
    from nimbus_inference.nimbus import Nimbus
    from nimbus_inference.utils import MultiplexDataset, prepare_normalization_dict
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
from .segmentation import create_anndata


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
        normalization_jobs: int = 1,
        clip_values: Sequence[float] = (0.0, 2.0),
    ) -> None:
        self._channels = sorted(channels)
        self._channel_paths = channel_paths
        self._mask_lookup = mask_lookup
        self.normalization_n_jobs = max(1, int(normalization_jobs))
        self.clip_values = tuple(clip_values)

        def _seg_lookup(fov_path: str) -> Path:
            return self._mask_lookup[Path(fov_path).name]

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
        return np.squeeze(io.imread(image_path))

    def prepare_normalization_dict(  # type: ignore[override]
        self,
        quantile: float = 0.999,
        clip_values: Sequence[float] = (0, 2),
        n_subset: int = 10,
        multiprocessing: bool = False,  # kept for API compatibility
        overwrite: bool = False,
    ):
        """
        Compute per-channel normalization without re-instantiating MultiplexDataset (avoids filename-based discovery).
        """
        self.clip_values = tuple(clip_values)
        self.normalization_dict_path = os.path.join(self.output_dir, "normalization_dict.json")

        if os.path.exists(self.normalization_dict_path) and not overwrite:
            with open(self.normalization_dict_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.normalization_dict = {k: float(v) for k, v in data.items()}
            return

        fov_list = list(self.fovs)
        if n_subset is not None and len(fov_list) > n_subset:
            fov_list = random.sample(fov_list, n_subset)

        norm_vals: Dict[str, List[float]] = {ch: [] for ch in self._channels}
        for fov in fov_list:
            for ch in self._channels:
                img = self.get_channel(fov, ch).astype(float)
                if np.any(img):
                    foreground = img[img > 0]
                    if foreground.size:
                        norm_vals[ch].append(float(np.quantile(foreground, quantile)))

        self.normalization_dict = {}
        for ch, vals in norm_vals.items():
            if vals:
                self.normalization_dict[ch] = float(np.mean(vals))
            else:
                self.normalization_dict[ch] = 1e-8

        norm_str = {k: str(v) for k, v in self.normalization_dict.items()}
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.normalization_dict_path, "w", encoding="utf-8") as handle:
            json.dump(norm_str, handle)


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
                try:
                    filename = get_filename(base_dir, filename_hint)
                except Exception:
                    continue
                found = base_dir / filename
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
            "Label": [p.label for p in props],
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


def _prepare_nimbus_output(cell_table: pd.DataFrame) -> pd.DataFrame:
    if cell_table is None or cell_table.empty:
        raise ValueError("Nimbus returned an empty cell table")
    renamed = cell_table.rename(columns={"label": "Label", "fov": "ROI"})
    renamed["ROI"] = renamed["ROI"].astype(str)
    renamed["Label"] = renamed["Label"].astype(int)
    return renamed


def _merge_with_masks(
    mask_features: pd.DataFrame,
    nimbus_df: pd.DataFrame,
    expected_channels: List[str],
    predicted_channels: List[str],
    allow_missing: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    merged = mask_features.merge(nimbus_df, on=["ROI", "Label"], how="left")

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
    metadata_cols = [c for c in mask_features.columns if c not in {"ROI", "Label"}]
    other_cols = [c for c in merged.columns if c not in {"ROI", "Label"} | set(channel_cols) | set(metadata_cols)]

    ordered = ["ROI", "Label"] + metadata_cols + other_cols + channel_cols
    merged = merged.loc[:, ordered]
    merged.reset_index(drop=True, inplace=True)
    merged["Master_Index"] = merged.index
    return merged, channel_cols


def _save_roi_tables(cell_df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for roi, roi_df in cell_df.groupby("ROI"):
        roi_df.to_csv(output_dir / f"{prefix}{roi}.csv", index=False)


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
        normalization_jobs=nimbus_config.normalization_jobs,
        clip_values=clip_values,
    )

    dataset.prepare_normalization_dict(
        quantile=nimbus_config.normalization_quantile,
        clip_values=clip_values,
        n_subset=nimbus_config.normalization_subset,
        multiprocessing=nimbus_config.normalization_jobs > 1,
        overwrite=nimbus_config.overwrite_existing_outputs,
    )

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

    nimbus_df = _prepare_nimbus_output(nimbus.predict_fovs())
    mask_features = _build_mask_features(mask_lookup, valid_rois)
    merged_celltable, predicted_channels = _merge_with_masks(
        mask_features, nimbus_df, expected_channels, channels_for_model, seg_config.allow_missing_channels
    )

    logging.info(
        "Nimbus produced %d cells across %d ROI(s) with %d predicted channel(s)",
        len(merged_celltable),
        len(valid_rois),
        len(predicted_channels),
    )

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
        if not anndata_path.is_absolute():
            anndata_path = Path(nimbus_config.output_dir) / anndata_path

        adata = create_anndata(
            merged_celltable,
            metadata_folder=general_config.metadata_folder,
            normalisation=seg_config.marker_normalisation,
            store_raw=seg_config.store_raw_marker_data,
            remove_channels=seg_config.remove_channels_list,
        )
        anndata_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(anndata_path)
        logging.info("Saved Nimbus AnnData to %s", anndata_path)
    else:
        logging.info("Skipping AnnData creation per config")


if __name__ == "__main__":
    main()








