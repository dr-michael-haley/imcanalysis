#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocess IMC channel TIFF folders before running the HyPERSTAC pipeline.

Expected input layout:

    {input_folder}/{roi_name}/{channel_name}.tiff

The output folder keeps the same ROI/channel layout and contains float32 TIFFs
scaled to [0, 1]. Reports are written at the output-folder root:

    normalisation_config.json
    normalisation_channel_report.csv
    normalisation_roi_channel_report.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tqdm.auto import tqdm


TIFF_SUFFIXES = {".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Background-correct and normalise IMC channel TIFF folders to [0, 1].",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        required=True,
        help="Raw image folder organised as {input-folder}/{roi name}/{channel name}.tiff.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="Output image folder with the same ROI/channel structure.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="Optional comma-separated channel order. If omitted, inferred alphabetically.",
    )
    parser.add_argument(
        "--background-subtraction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Subtract background before scaling.",
    )
    parser.add_argument(
        "--background-method",
        type=str,
        default="fixed",
        choices=["fixed", "percentile"],
        help="Use a fixed background value for every pixel, or estimate one percentile per ROI/channel.",
    )
    parser.add_argument(
        "--background-fixed-value",
        type=float,
        default=0.25,
        help="Fixed value subtracted from every pixel when --background-method=fixed.",
    )
    parser.add_argument(
        "--background-percentile",
        type=float,
        default=20.0,
        help="Per-ROI/per-channel percentile used as background when --background-method=percentile.",
    )
    parser.add_argument(
        "--presence-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set ROI/channel images to zero when the corrected signal is below the presence threshold.",
    )
    parser.add_argument(
        "--presence-percentile",
        type=float,
        default=95.0,
        help="Corrected-signal percentile used to decide whether a channel is present in an ROI.",
    )
    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=0.05,
        help="Minimum corrected-signal presence percentile required to keep an ROI/channel image.",
    )
    parser.add_argument(
        "--scale-percentile",
        type=float,
        default=99.0,
        help="Per-channel global corrected-signal percentile used as the 1.0 scaling denominator.",
    )
    parser.add_argument(
        "--scale-present-only",
        action="store_true",
        help="Calculate per-channel scaling only from ROI/channel images marked present.",
    )
    parser.add_argument(
        "--scale-sample-pixels",
        type=int,
        default=0,
        help=(
            "Optional approximate scaling cap per channel. If >0, sample up to this "
            "many corrected pixels per channel instead of using every pixel."
        ),
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=None,
        help="Optional tifffile compression, e.g. zlib. Leave empty for uncompressed TIFFs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output ROI folders and report files.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed used for optional scaling samples.")
    return parser.parse_args()


def validate_percentile(value: float, name: str) -> None:
    if not 0 <= value <= 100:
        raise ValueError(f"{name} must be between 0 and 100; got {value}.")


def channel_file_map(roi_dir: Path) -> dict[str, Path]:
    files = [path for path in roi_dir.iterdir() if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES]
    mapping: dict[str, Path] = {}
    for path in files:
        channel = path.stem
        if channel in mapping:
            raise ValueError(f"Duplicate files for channel '{channel}' in {roi_dir}")
        mapping[channel] = path
    return mapping


def discover_rois(image_folder: Path) -> list[Path]:
    if not image_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {image_folder}")
    rois = sorted(path for path in image_folder.iterdir() if path.is_dir())
    if not rois:
        raise ValueError(f"No ROI folders found in {image_folder}")
    return rois


def resolve_channels(rois: list[Path], channels_arg: str | None) -> list[str]:
    if channels_arg:
        channels = [channel.strip() for channel in channels_arg.split(",") if channel.strip()]
        if not channels:
            raise ValueError("--channels was supplied but no channel names were parsed.")
        return channels
    channels = sorted(channel_file_map(rois[0]))
    if not channels:
        raise ValueError(f"No TIFF channel files found in first ROI folder: {rois[0]}")
    return channels


def validate_roi_channels(rois: list[Path], channels: list[str]) -> None:
    expected = set(channels)
    for roi in rois:
        available = set(channel_file_map(roi))
        missing = sorted(expected - available)
        if missing:
            raise ValueError(f"ROI '{roi.name}' is missing channels: {missing}")


def read_channel(path: Path) -> tuple[np.ndarray, int]:
    image = tifffile.imread(path)
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D channel TIFF after squeeze, got shape {image.shape}: {path}")
    image = image.astype(np.float32, copy=False)
    finite = np.isfinite(image)
    nonfinite_count = int((~finite).sum())
    if nonfinite_count:
        image = np.where(finite, image, 0.0).astype(np.float32, copy=False)
    return image, nonfinite_count


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.percentile(finite, percentile))


def finite_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_p99": np.nan,
        }
    return {
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_p95": float(np.percentile(finite, 95)),
        f"{prefix}_p99": float(np.percentile(finite, 99)),
    }


def corrected_image(raw: np.ndarray, background_value: float) -> np.ndarray:
    corrected = raw.astype(np.float32, copy=False) - np.float32(background_value)
    return np.maximum(corrected, 0.0).astype(np.float32, copy=False)


def calculate_background_value(raw: np.ndarray, args: argparse.Namespace) -> float:
    if not args.background_subtraction:
        return 0.0
    if args.background_method == "fixed":
        return float(args.background_fixed_value)
    if args.background_method == "percentile":
        return finite_percentile(raw, args.background_percentile)
    raise ValueError(f"Unknown background method: {args.background_method}")


def measure_roi_channels(
    rois: list[Path],
    channels: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for roi in tqdm(rois, desc="Measuring ROI/channel signal"):
        mapping = channel_file_map(roi)
        for channel in channels:
            input_path = mapping[channel]
            raw, nonfinite_count = read_channel(input_path)
            background_value = calculate_background_value(raw, args)
            corrected = corrected_image(raw, background_value)
            presence_value = finite_percentile(corrected, args.presence_percentile)
            present = (presence_value >= args.presence_threshold) if args.presence_mask else True

            row = {
                "roi": roi.name,
                "channel": channel,
                "input_path": str(input_path),
                "input_filename": input_path.name,
                "height": int(raw.shape[0]),
                "width": int(raw.shape[1]),
                "n_pixels": int(raw.size),
                "raw_nonfinite_pixels": nonfinite_count,
                "background_subtraction": bool(args.background_subtraction),
                "background_method": args.background_method if args.background_subtraction else "none",
                "background_fixed_value": float(args.background_fixed_value),
                "background_percentile": float(args.background_percentile),
                "background_value": float(background_value),
                "presence_mask": bool(args.presence_mask),
                "presence_percentile": float(args.presence_percentile),
                "presence_value": float(presence_value),
                "presence_threshold": float(args.presence_threshold),
                "present": bool(present),
            }
            row.update(finite_stats(raw, "raw"))
            row.update(finite_stats(corrected, "corrected"))
            rows.append(row)
    return pd.DataFrame(rows)


def sample_values(values: np.ndarray, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    values = values.reshape(-1)
    if sample_size <= 0 or sample_size >= values.size:
        return values
    indices = rng.choice(values.size, size=sample_size, replace=False)
    return values[indices]


def calculate_channel_scales(
    roi_report: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    rows = []
    for channel, channel_rows in tqdm(
        roi_report.groupby("channel", sort=False),
        total=roi_report["channel"].nunique(),
        desc="Calculating global channel scales",
    ):
        eligible = channel_rows
        if args.scale_present_only:
            eligible = eligible[eligible["present"]]

        total_eligible_pixels = int(eligible["n_pixels"].sum())
        chunks = []
        used_pixels = 0
        sampled = bool(args.scale_sample_pixels and args.scale_sample_pixels > 0)

        for row in eligible.itertuples(index=False):
            raw, _ = read_channel(Path(row.input_path))
            corrected = corrected_image(raw, float(row.background_value))

            if sampled and total_eligible_pixels > args.scale_sample_pixels:
                fraction = row.n_pixels / total_eligible_pixels
                target = max(1, int(np.ceil(args.scale_sample_pixels * fraction)))
                values = sample_values(corrected, target, rng)
            else:
                values = corrected.reshape(-1)

            chunks.append(values.astype(np.float32, copy=False))
            used_pixels += int(values.size)

        if chunks:
            all_values = np.concatenate(chunks)
            scale_value_raw = finite_percentile(all_values, args.scale_percentile)
        else:
            scale_value_raw = 0.0

        scale_value = scale_value_raw if scale_value_raw > 0 else 1.0
        rows.append(
            {
                "channel": channel,
                "scale_percentile": float(args.scale_percentile),
                "scale_value_raw": float(scale_value_raw),
                "scale_value": float(scale_value),
                "scale_value_was_zero": bool(scale_value_raw <= 0),
                "scale_present_only": bool(args.scale_present_only),
                "scale_sampled": sampled and total_eligible_pixels > args.scale_sample_pixels,
                "scale_sample_pixels_requested": int(args.scale_sample_pixels),
                "scale_pixels_available": total_eligible_pixels,
                "scale_pixels_used": used_pixels,
                "n_rois": int(channel_rows["roi"].nunique()),
                "n_present_rois": int(channel_rows["present"].sum()),
            }
        )
    return pd.DataFrame(rows)


def prepare_output_folder(output_folder: Path, rois: list[Path], args: argparse.Namespace) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    report_files = [
        output_folder / "normalisation_config.json",
        output_folder / "normalisation_channel_report.csv",
        output_folder / "normalisation_roi_channel_report.csv",
    ]
    existing_roi_dirs = [output_folder / roi.name for roi in rois if (output_folder / roi.name).exists()]
    existing_reports = [path for path in report_files if path.exists()]

    if (existing_roi_dirs or existing_reports) and not args.overwrite:
        existing = [str(path) for path in [*existing_roi_dirs, *existing_reports]]
        raise FileExistsError(
            "Output folder already contains preprocessing outputs. Use --overwrite to replace them:\n"
            + "\n".join(existing[:10])
        )

    if args.overwrite:
        for path in existing_roi_dirs:
            shutil.rmtree(path)
        for path in existing_reports:
            path.unlink()


def write_normalised_images(
    roi_report: pd.DataFrame,
    channel_report: pd.DataFrame,
    output_folder: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    scale_lookup = channel_report.set_index("channel")["scale_value"].to_dict()
    rows = []
    for row in tqdm(roi_report.itertuples(index=False), total=len(roi_report), desc="Writing normalised TIFFs"):
        input_path = Path(row.input_path)
        output_path = output_folder / row.roi / row.input_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        raw, _ = read_channel(input_path)
        corrected = corrected_image(raw, float(row.background_value))
        if bool(row.present):
            scale_value = float(scale_lookup[row.channel])
            normalised = np.clip(corrected / scale_value, 0.0, 1.0)
        else:
            normalised = np.zeros_like(corrected, dtype=np.float32)

        normalised = normalised.astype(np.float32, copy=False)
        tifffile.imwrite(output_path, normalised, compression=args.compression)

        updated = row._asdict()
        updated["output_path"] = str(output_path)
        updated["scale_value"] = float(scale_lookup[row.channel])
        updated["output_blank_due_to_absence"] = not bool(row.present)
        updated.update(finite_stats(normalised, "output"))
        rows.append(updated)
    return pd.DataFrame(rows)


def write_config(args: argparse.Namespace, channels: list[str], rois: list[Path]) -> None:
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["channels"] = channels
    config["n_rois"] = len(rois)
    with open(args.output_folder / "normalisation_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def main() -> None:
    args = parse_args()
    validate_percentile(args.background_percentile, "--background-percentile")
    validate_percentile(args.presence_percentile, "--presence-percentile")
    validate_percentile(args.scale_percentile, "--scale-percentile")
    if args.background_fixed_value < 0:
        raise ValueError("--background-fixed-value must be >= 0.")
    if args.scale_sample_pixels < 0:
        raise ValueError("--scale-sample-pixels must be >= 0.")
    if args.input_folder.resolve() == args.output_folder.resolve():
        raise ValueError("--input-folder and --output-folder must be different.")

    rois = discover_rois(args.input_folder)
    channels = resolve_channels(rois, args.channels)
    validate_roi_channels(rois, channels)
    prepare_output_folder(args.output_folder, rois, args)

    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"ROIs: {len(rois)}")
    print(f"Channels: {len(channels)}")
    if args.background_subtraction and args.background_method == "fixed":
        background_message = f"fixed value {args.background_fixed_value:g}"
    elif args.background_subtraction:
        background_message = f"per-ROI/channel p{args.background_percentile:g}"
    else:
        background_message = "disabled"
    print(f"Background subtraction: {args.background_subtraction} using {background_message}")
    print(
        "Presence mask: "
        f"{args.presence_mask} using corrected p{args.presence_percentile:g} >= {args.presence_threshold:g}"
    )
    print(f"Global channel scaling: p{args.scale_percentile:g}")

    roi_report = measure_roi_channels(rois, channels, args)
    channel_report = calculate_channel_scales(roi_report, args)
    roi_report = write_normalised_images(roi_report, channel_report, args.output_folder, args)

    channel_report.to_csv(args.output_folder / "normalisation_channel_report.csv", index=False)
    roi_report.to_csv(args.output_folder / "normalisation_roi_channel_report.csv", index=False)
    write_config(args, channels, rois)

    n_absent = int((~roi_report["present"]).sum())
    print(f"Wrote normalised TIFFs to {args.output_folder}")
    print(f"Absent ROI/channel images set to zero: {n_absent}")
    print(f"Saved reports to {args.output_folder}")


if __name__ == "__main__":
    main()
