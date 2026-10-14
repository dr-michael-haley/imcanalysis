"""Run the isolated exporter with ``python -m SpatialBiologyToolkit.odon_export``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="Experimental AnnData/TIFF -> Odon exporter (local CPU)."
    )
    cli.add_argument(
        "--project",
        type=Path,
        help="SBT project directory containing config.yaml; no registration needed",
    )
    cli.add_argument(
        "--anndata", type=Path, help="Source .h5ad; overrides project default"
    )
    cli.add_argument("--masks", type=Path, help="Folder containing ROI.tif[f] masks")
    cli.add_argument(
        "--images",
        type=Path,
        action="append",
        help="Image folder; repeat to merge folders",
    )
    cli.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory (never overwritten)",
    )
    cli.add_argument("--roi-key", default="ROI")
    cli.add_argument("--object-id-key", default="ObjectNumber")
    cli.add_argument(
        "--roi", action="append", help="Export only this ROI; repeat for a small trial"
    )
    cli.add_argument(
        "--channel",
        action="append",
        help="Image channel to include; repeat; default all discovered",
    )
    cli.add_argument(
        "--stack-channel",
        action="append",
        help="CYX TIFF plane name, in order; repeat for every plane",
    )
    cli.add_argument(
        "--obs-column",
        action="append",
        help="Observation to copy; repeat; default all scalar columns",
    )
    cli.add_argument(
        "--no-obs",
        action="store_true",
        help="Export only identity, geometry and measurements",
    )
    expression = cli.add_mutually_exclusive_group()
    expression.add_argument("--layer", help="Read measurements from this AnnData layer")
    expression.add_argument(
        "--use-raw",
        action="store_true",
        help="Read measurements from raw.X/raw.var_names",
    )
    cli.add_argument(
        "--levels",
        type=int,
        default=5,
        help="Maximum pyramid levels, including full resolution",
    )
    cli.add_argument("--chunk-size", type=int, default=512)
    cli.add_argument("--intensity-mode", choices=["auto", "preserve"], default="auto")
    cli.add_argument(
        "--dry-run",
        action="store_true",
        help="Check metadata, identities and TIFF headers without writing",
    )
    return cli


def _inputs(args: argparse.Namespace) -> tuple[Path, Path, list[Path]]:
    anndata, masks, images = args.anndata, args.masks, args.images
    if args.project is not None:
        from SpatialBiologyToolkit.config import load_config
        from SpatialBiologyToolkit.pipeline.assets import resolve_project_path

        root = args.project.expanduser().resolve()
        general = load_config(root / "config.yaml").general
        anndata = anndata or resolve_project_path(root, general.anndata_path)
        masks = masks or resolve_project_path(root, general.masks_folder)
        # Use the declared denoised image source, with no silent raw-image fallback.
        images = images or [resolve_project_path(root, general.denoised_images_folder)]
    if anndata is None or masks is None or not images:
        raise ValueError("Supply --anndata, --masks and --images, or use --project.")
    return anndata, masks, images


def main(argv: Sequence[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.no_obs and args.obs_column:
        cli.error("--no-obs and --obs-column cannot be combined")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        anndata, masks, images = _inputs(args)
        from .exporter import export_dataset

        result = export_dataset(
            anndata,
            masks,
            images,
            args.output,
            roi_key=args.roi_key,
            object_id_key=args.object_id_key,
            rois=args.roi,
            channels=args.channel,
            stack_channels=args.stack_channel,
            obs_columns=[] if args.no_obs else args.obs_column,
            layer=args.layer,
            use_raw=args.use_raw,
            levels=args.levels,
            chunk_size=args.chunk_size,
            intensity_mode=args.intensity_mode,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, ImportError) as exc:
        cli.exit(2, f"Odon export failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
