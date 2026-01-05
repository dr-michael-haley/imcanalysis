"""
Standalone extractor for expansion intensities.

Recomputes the expansion-based mean intensities created by segmentation_nimbus
without running Nimbus inference, so expansion size can be tweaked quickly.
Uses the same config/override mechanism and multiprocessing-friendly workflow.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm

from .config_and_utils import (
    GeneralConfig,
    NimbusConfig,
    filter_config_for_dataclass,
    process_config_with_overrides,
    setup_logging,
)


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


def _process_roi_expansion(
    roi: str,
    mask_path: Path,
    roi_channel_paths: Dict[str, Path],
    expected_channels: List[str],
    expansion_pixels: int,
) -> pd.DataFrame:
    from scipy.ndimage import binary_dilation

    mask = io.imread(mask_path)
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]

    channel_images = {}
    for channel in expected_channels:
        if channel in roi_channel_paths:
            try:
                img = io.imread(roi_channel_paths[channel])
                img = np.squeeze(img)
                if img.ndim == 3:
                    img = img[..., 0]
                channel_images[channel] = img
            except Exception as exc:
                logging.warning("Error loading %s for %s: %s", channel, roi, exc)

    cell_data = []
    for label in unique_labels:
        cell_mask = mask == label
        expanded_mask = binary_dilation(cell_mask, iterations=expansion_pixels)

        cell_intensities = {"ObjectNumber": label}
        for channel in expected_channels:
            if channel in channel_images:
                cell_intensities[channel] = np.mean(channel_images[channel][expanded_mask])
            else:
                cell_intensities[channel] = np.nan

        cell_data.append(cell_intensities)

    roi_df = pd.DataFrame(cell_data)
    roi_df["ROI"] = roi
    return roi_df


def _extract_expansion_intensities(
    mask_lookup: Dict[str, Path],
    rois: List[str],
    channel_paths: Dict[str, Dict[str, Path]],
    expected_channels: List[str],
    expansion_pixels: int,
    n_jobs: int = 1,
) -> pd.DataFrame:
    from multiprocessing import Pool, cpu_count

    n_jobs = int(n_jobs)
    if n_jobs == -1:
        n_processes = cpu_count()
    elif n_jobs > 1:
        n_processes = min(n_jobs, len(rois), cpu_count())
    else:
        n_processes = 1

    if n_processes > 1:
        logging.info("Processing %d ROIs with %d parallel workers", len(rois), n_processes)
        args_list = [
            (roi, mask_lookup[roi], channel_paths.get(roi, {}), expected_channels, expansion_pixels)
            for roi in rois
        ]
        with Pool(processes=n_processes) as pool:
            frames = list(tqdm(pool.starmap(_process_roi_expansion, args_list), total=len(rois), desc="Expansion intensity extraction"))
    else:
        frames = []
        for roi in tqdm(rois, desc="Expansion intensity extraction"):
            roi_df = _process_roi_expansion(
                roi=roi,
                mask_path=mask_lookup[roi],
                roi_channel_paths=channel_paths.get(roi, {}),
                expected_channels=expected_channels,
                expansion_pixels=expansion_pixels,
            )
            frames.append(roi_df)

    return pd.concat(frames, ignore_index=True)


def _save_expansion_table(expansion_df: pd.DataFrame, output_dir: Path, expansion_pixels: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"expansion_intensities_px{expansion_pixels}.csv"
    expansion_df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    pipeline_stage = "ExpansionIntensityExtraction"
    config = process_config_with_overrides()
    setup_logging(config.get("logging", {}), pipeline_stage)

    general_cfg = GeneralConfig(**filter_config_for_dataclass(config.get("general", {}), GeneralConfig))
    nimbus_cfg = NimbusConfig(**filter_config_for_dataclass(config.get("nimbus", {}), NimbusConfig))

    metadata_folder = Path(general_cfg.metadata_folder)
    panel = _load_panel(metadata_folder)

    mask_lookup = _discover_masks(Path(general_cfg.masks_folder), nimbus_cfg.mask_extensions)
    if not mask_lookup:
        raise FileNotFoundError(f"No mask files found in {general_cfg.masks_folder}")

    rois = _filter_rois_by_metadata(mask_lookup, metadata_folder / "metadata.csv")
    if not rois:
        raise ValueError("No ROIs to process after applying metadata import filters")

    (
        valid_rois,
        channel_paths,
        _,
        missing_summary,
        expected_channels,
        _,
    ) = _resolve_channel_paths(rois, panel, general_cfg, nimbus_cfg)

    if not valid_rois:
        raise ValueError("No ROIs with usable channel images were found for expansion extraction")

    if missing_summary:
        logging.warning("Channels missing for some ROIs (files not found): %s", missing_summary)

    expansion_pixels = int(nimbus_cfg.expansion_pixels)
    expansion_jobs = int(nimbus_cfg.expansion_jobs)

    logging.info(
        "Extracting expansion intensities with %d pixel expansion (%d ROIs, %d expected channels, %s jobs)",
        expansion_pixels,
        len(valid_rois),
        len(expected_channels),
        expansion_jobs,
    )

    expansion_df = _extract_expansion_intensities(
        mask_lookup=mask_lookup,
        rois=valid_rois,
        channel_paths=channel_paths,
        expected_channels=expected_channels,
        expansion_pixels=expansion_pixels,
        n_jobs=expansion_jobs,
    )

    expansion_df["ROI"] = expansion_df["ROI"].astype(str)
    expansion_df["ObjectNumber"] = expansion_df["ObjectNumber"].astype(int)

    output_path = _save_expansion_table(expansion_df, Path(nimbus_cfg.output_dir), expansion_pixels)
    logging.info("Expansion intensities saved to %s", output_path)
    logging.info("Completed expansion intensity extraction for %d ROIs", len(valid_rois))


if __name__ == "__main__":
    main()
