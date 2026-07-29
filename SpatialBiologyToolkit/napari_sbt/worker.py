"""Spawn-safe, resumable cohort-first feature extraction worker."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import skimage as sk

from SpatialBiologyToolkit._napari_imc_normalization import (
    find_normalization_value,
    prepare_normalization_dict,
)
from SpatialBiologyToolkit.pipeline.manifests import utc_now, write_json
from SpatialBiologyToolkit.qc_classifier.io import (
    discover_mask_files,
    discover_roi_images,
    file_fingerprint,
    load_mask,
)

from .cohort import eligible_ids_by_roi, validate_frozen_cohort
from .feature_sources import combine_feature_sources
from .features import build_feature_dictionary, build_roi_features
from .models import FeatureSource, SyntheticFeatureRecipe
from .storage import (
    feature_recipe_hash,
    load_experiment,
    read_dataframe,
    write_dataframe,
    write_feature_manifest,
)


@dataclass
class FeatureBuildResult:
    feature_table: Path
    feature_dictionary: Path
    coverage_report: Path
    failed_rois: Path
    manifest: Path
    eligible_cells: int
    represented_rois: int
    skipped_rois: int
    feature_count: int
    erosion_losses: int
    failures: int
    elapsed_seconds: float
    warnings: list[str]


class FeatureBuildCancelled(RuntimeError):
    """Raised after safely preserving completed ROI fragments."""


def _emit(event: dict) -> None:
    print(json.dumps(event, sort_keys=True, default=str), flush=True)


def _read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_normalization(recipe: SyntheticFeatureRecipe) -> dict[str, float]:
    if not recipe.normalization_dict_path:
        return {}
    payload = _read_json(Path(recipe.normalization_dict_path))
    if isinstance(payload, dict) and isinstance(payload.get("normalization_dict"), dict):
        payload = payload["normalization_dict"]
    return prepare_normalization_dict(payload)


def _resolve_input_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve(strict=False)


def _input_fingerprint(
    mask_path: Path,
    channel_paths: dict[str, Path],
    *,
    experiment_id: str,
    revision: int,
    cohort_hash: str,
    recipe: SyntheticFeatureRecipe,
) -> tuple[str, dict]:
    inputs = {
        "mask": file_fingerprint(mask_path),
        "channels": {
            channel: file_fingerprint(path)
            for channel, path in sorted(channel_paths.items())
        },
        "normalization_dict": file_fingerprint(recipe.normalization_dict_path),
    }
    payload = {
        "experiment_id": experiment_id,
        "revision": revision,
        "cohort_sha256": cohort_hash,
        "recipe": recipe.model_dump(mode="json"),
        "inputs": inputs,
    }
    return feature_recipe_hash(payload), payload


def _read_channel(path: str | Path) -> np.ndarray:
    image = np.asarray(sk.io.imread(path)).squeeze()
    if image.ndim != 2:
        raise ValueError(f"Scientific feature image must be 2D, got {image.shape}: {path}")
    return image.astype(np.float32, copy=False)


def _roi_task(payload: dict) -> dict:
    """Top-level process-pool task; all arguments and results are serializable."""

    started = time.monotonic()
    roi = str(payload["roi"])
    mask = load_mask(payload["mask_path"])
    normalization = payload["normalization"]
    recipe = SyntheticFeatureRecipe.model_validate(payload["recipe"])
    combined_table: pd.DataFrame | None = None
    result_warnings: list[str] = []
    vanished_object_ids: list[int] = []
    channel_items = list(payload["channel_paths"].items())
    for channel_index, (channel, path) in enumerate(channel_items):
        image = _read_channel(path)
        maximum = find_normalization_value(normalization, channel)
        if maximum is not None:
            image = image / float(maximum)
        channel_recipe = recipe.model_copy(
            update={
                "shape_features": recipe.shape_features and channel_index == 0,
                "context_features": recipe.context_features and channel_index == 0,
            }
        )
        channel_result = build_roi_features(
            roi=roi,
            full_mask=mask,
            eligible_ids=set(payload["eligible_ids"]),
            channel_images={channel: image},
            recipe=channel_recipe,
        )
        if combined_table is None:
            combined_table = channel_result.table
            vanished_object_ids = channel_result.vanished_object_ids
            result_warnings.extend(channel_result.warnings)
        else:
            channel_columns = [
                column
                for column in channel_result.table
                if column.startswith(f"channel::{channel}::")
            ]
            combined_table = combined_table.merge(
                channel_result.table.loc[
                    :, ["ROI", "ObjectNumber"] + channel_columns
                ],
                on=["ROI", "ObjectNumber"],
                how="left",
                validate="one_to_one",
            )
        del image
    if combined_table is None:
        shape_result = build_roi_features(
            roi=roi,
            full_mask=mask,
            eligible_ids=set(payload["eligible_ids"]),
            channel_images={},
            recipe=recipe,
        )
        combined_table = shape_result.table
        vanished_object_ids = shape_result.vanished_object_ids
        result_warnings.extend(shape_result.warnings)
    fragment = write_dataframe(payload["fragment_path"], combined_table)
    sidecar = write_json(
        payload["sidecar_path"],
        {
            "fingerprint": payload["fingerprint"],
            "roi": roi,
            "rows": len(combined_table),
            "features": len(combined_table.columns),
            "vanished_object_ids": vanished_object_ids,
            "warnings": result_warnings,
            "elapsed_seconds": time.monotonic() - started,
            "completed_at": utc_now().isoformat(),
            "inputs": payload["fingerprint_payload"]["inputs"],
        },
    )
    return {
        "roi": roi,
        "fragment": str(fragment),
        "sidecar": str(sidecar),
        "rows": len(combined_table),
        "vanished": len(vanished_object_ids),
        "warnings": result_warnings,
        "elapsed_seconds": time.monotonic() - started,
        "resumed": False,
    }


def _fragment_is_valid(fragment: Path, sidecar: Path, fingerprint: str) -> bool:
    if not fragment.exists() or not sidecar.exists():
        return False
    try:
        metadata = _read_json(sidecar)
        return metadata.get("fingerprint") == fingerprint
    except Exception:  # noqa: BLE001 - invalid cache sidecars are treated as stale
        return False


def run_feature_build(
    experiment: str | Path,
    *,
    workers: int | None = None,
    progress: Callable[[dict], None] | None = None,
) -> FeatureBuildResult:
    """Build resumable ROI fragments and a canonical cohort-only table."""

    started = time.monotonic()
    notify = progress or _emit
    manifest, paths = load_experiment(experiment)
    cohort = read_dataframe(paths.root / manifest.cell_scope.snapshot_path)
    validate_frozen_cohort(cohort, manifest.cell_scope)
    eligible = eligible_ids_by_roi(cohort)
    input_root = (
        Path(manifest.project_root).expanduser().resolve(strict=False)
        if manifest.project_root
        else paths.root
    )
    mask_folder = _resolve_input_path(manifest.masks_folder, input_root)
    image_folders = [
        _resolve_input_path(folder, input_root) for folder in manifest.images_folders
    ]
    recipe = manifest.synthetic_features.model_copy(deep=True)
    if recipe.normalization_dict_path:
        recipe.normalization_dict_path = str(
            _resolve_input_path(recipe.normalization_dict_path, input_root)
        )
    normalization = _load_normalization(recipe)
    mask_paths = discover_mask_files(mask_folder)
    worker_count = max(1, int(workers or min(os.cpu_count() or 1, 8)))
    cancel_path = paths.logs / "feature_build.cancel"
    if cancel_path.exists():
        cancel_path.unlink()
    tasks: list[dict] = []
    completed: list[dict] = []
    failures: list[dict] = []
    warnings: list[str] = []
    skipped_rois = 0
    for roi, object_ids in sorted(eligible.items()):
        mask_path = mask_paths.get(roi)
        if mask_path is None:
            failures.append({"ROI": roi, "error": "Eligible ROI has no mask file."})
            continue
        discovered = discover_roi_images(image_folders, roi)
        requested = recipe.channels
        if requested:
            missing_channels = sorted(set(requested) - set(discovered))
            if missing_channels:
                failures.append(
                    {
                        "ROI": roi,
                        "error": f"Missing requested channel(s): {missing_channels}",
                    }
                )
                continue
            channel_paths = {channel: discovered[channel] for channel in requested}
        else:
            channel_paths = discovered
        if (
            recipe.distribution_features
            or recipe.region_features
            or recipe.gradient_features
        ) and not channel_paths:
            failures.append({"ROI": roi, "error": "No channel images were discovered."})
            continue
        fingerprint, fingerprint_payload = _input_fingerprint(
            mask_path,
            channel_paths,
            experiment_id=manifest.experiment_id,
            revision=manifest.revision,
            cohort_hash=manifest.cell_scope.snapshot_sha256,
            recipe=recipe,
        )
        safe_roi = feature_recipe_hash({"roi": roi})[:12]
        fragment = paths.feature_fragments / f"{safe_roi}.parquet"
        sidecar = paths.feature_fragments / f"{safe_roi}.json"
        if _fragment_is_valid(fragment, sidecar, fingerprint):
            metadata = _read_json(sidecar)
            completed.append(
                {
                    "roi": roi,
                    "fragment": str(fragment),
                    "sidecar": str(sidecar),
                    "rows": metadata.get("rows", len(object_ids)),
                    "vanished": len(metadata.get("vanished_object_ids", [])),
                    "warnings": metadata.get("warnings", []),
                    "elapsed_seconds": metadata.get("elapsed_seconds", 0),
                    "resumed": True,
                }
            )
            skipped_rois += 1
            notify({"event": "roi_resumed", "roi": roi, "rows": len(object_ids)})
            continue
        tasks.append(
            {
                "roi": roi,
                "mask_path": str(mask_path),
                "channel_paths": {
                    channel: str(path) for channel, path in channel_paths.items()
                },
                "eligible_ids": sorted(object_ids),
                "normalization": normalization,
                "recipe": recipe.model_dump(mode="json"),
                "fingerprint": fingerprint,
                "fingerprint_payload": fingerprint_payload,
                "fragment_path": str(fragment),
                "sidecar_path": str(sidecar),
            }
        )

    notify(
        {
            "event": "build_started",
            "eligible_cells": len(cohort),
            "represented_rois": len(eligible),
            "pending_rois": len(tasks),
            "resumed_rois": skipped_rois,
            "workers": worker_count,
        }
    )
    if tasks:
        executor = ProcessPoolExecutor(max_workers=worker_count)
        future_to_roi = {executor.submit(_roi_task, task): task["roi"] for task in tasks}
        try:
            for future in as_completed(future_to_roi):
                roi = future_to_roi[future]
                try:
                    result = future.result()
                    completed.append(result)
                    warnings.extend(result["warnings"])
                    notify({"event": "roi_completed", **result})
                except Exception as exc:  # noqa: BLE001 - isolate failures per ROI
                    failures.append({"ROI": roi, "error": f"{type(exc).__name__}: {exc}"})
                    notify({"event": "roi_failed", "roi": roi, "error": str(exc)})
                if cancel_path.exists():
                    for pending in future_to_roi:
                        pending.cancel()
                    executor.shutdown(wait=True, cancel_futures=True)
                    cancel_path.unlink(missing_ok=True)
                    notify(
                        {
                            "event": "cancelled",
                            "completed_rois": len(completed),
                            "message": "Valid completed fragments were preserved.",
                        }
                    )
                    raise FeatureBuildCancelled(
                        "Feature build cancelled after preserving completed fragments."
                    )
        except KeyboardInterrupt:
            for future in future_to_roi:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            notify({"event": "cancelled", "completed_rois": len(completed)})
            raise
        else:
            executor.shutdown(wait=True)

    fragment_tables = [
        read_dataframe(result["fragment"])
        for result in sorted(completed, key=lambda item: item["roi"])
    ]
    if not fragment_tables:
        failed_path = write_dataframe(
            paths.features / "failed_rois.csv", pd.DataFrame(failures)
        )
        raise RuntimeError(
            f"No ROI feature fragments were available; see {failed_path}."
        )
    synthetic = pd.concat(fragment_tables, ignore_index=True)
    synthetic_path = write_dataframe(
        paths.features / "synthetic_features.parquet", synthetic
    )
    configured_sources = [
        source.model_copy(
            deep=True,
            update={
                "path": (
                    str(_resolve_input_path(source.path, input_root))
                    if source.path
                    else None
                )
            },
        )
        for source in manifest.feature_sources
        if source.enabled and source.kind != "synthetic"
    ]
    sources = [
        FeatureSource(
            source_id="imc",
            kind="synthetic",
            path=str(synthetic_path),
        ),
        *configured_sources,
    ]
    combined = combine_feature_sources(
        cohort,
        sources,
        roi_obs=manifest.roi_obs,
        object_id_obs=manifest.object_id_obs,
    )
    feature_table_path = write_dataframe(paths.feature_table, combined.table)
    dictionary = build_feature_dictionary(combined.table)
    valid_lookup = set(combined.feature_columns)
    dictionary["valid_model_input"] = dictionary["feature"].isin(valid_lookup)
    dictionary_path = write_dataframe(paths.feature_dictionary, dictionary)
    coverage_path = write_dataframe(
        paths.features / "coverage_report.csv", combined.coverage
    )
    missing_coverage_paths: dict[str, str] = {}
    for source_id, missing_cells in combined.missing_by_source.items():
        missing_path = write_dataframe(
            paths.features / "coverage_missing" / f"{source_id}.parquet",
            missing_cells,
        )
        missing_coverage_paths[source_id] = str(missing_path)
    failed_path = write_dataframe(
        paths.features / "failed_rois.csv",
        pd.DataFrame(failures, columns=["ROI", "error"]),
    )
    erosion_losses = sum(int(result["vanished"]) for result in completed)
    elapsed = time.monotonic() - started
    feature_set_id = feature_recipe_hash(
        {
            "columns": combined.feature_columns,
            "cohort": manifest.cell_scope.snapshot_sha256,
            "recipe": recipe.model_dump(mode="json"),
        }
    )
    provenance = {
        "schema_version": 1,
        "experiment_id": manifest.experiment_id,
        "experiment_revision": manifest.revision,
        "cohort_sha256": manifest.cell_scope.snapshot_sha256,
        "recipe": recipe.model_dump(mode="json"),
        "feature_sources": [source.model_dump(mode="json") for source in sources],
        "feature_source_fingerprints": {
            source.source_id: file_fingerprint(source.path)
            for source in sources
        },
        "missing_coverage_tables": missing_coverage_paths,
        "feature_set_id": feature_set_id,
        "eligible_cells": len(cohort),
        "represented_rois": len(eligible),
        "completed_rois": len(completed),
        "resumed_rois": skipped_rois,
        "failures": len(failures),
        "erosion_losses": erosion_losses,
        "feature_count": len(combined.feature_columns),
        "warnings": warnings,
        "elapsed_seconds": elapsed,
        "completed_at": utc_now().isoformat(),
    }
    manifest.active_feature_set_id = feature_set_id
    from .storage import save_experiment

    save_experiment(manifest, paths.root, audit_action="feature_build_completed")
    manifest_path = write_feature_manifest(paths, provenance)
    notify({"event": "build_completed", **provenance})
    return FeatureBuildResult(
        feature_table=feature_table_path,
        feature_dictionary=dictionary_path,
        coverage_report=coverage_path,
        failed_rois=failed_path,
        manifest=manifest_path,
        eligible_cells=len(cohort),
        represented_rois=len(eligible),
        skipped_rois=skipped_rois,
        feature_count=len(combined.feature_columns),
        erosion_losses=erosion_losses,
        failures=len(failures),
        elapsed_seconds=elapsed,
        warnings=warnings,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    features = subparsers.add_parser("features")
    features.add_argument("--experiment", required=True)
    features.add_argument("--workers", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "features":
        try:
            run_feature_build(args.experiment, workers=args.workers)
        except FeatureBuildCancelled:
            return 130
        except KeyboardInterrupt:
            return 130
        except Exception as exc:  # noqa: BLE001 - CLI reports a structured failure
            _emit({"event": "build_failed", "error": f"{type(exc).__name__}: {exc}"})
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "FeatureBuildCancelled",
    "FeatureBuildResult",
    "main",
    "run_feature_build",
]
