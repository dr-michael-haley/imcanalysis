"""Atomic storage helpers for reusable Napari SBT experiment assets."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from SpatialBiologyToolkit.pipeline.manifests import (
    read_model,
    utc_now,
    write_json,
    write_yaml,
)

from .models import ExperimentManifest


@dataclass(frozen=True)
class ExperimentPaths:
    """Filesystem paths owned by one experiment."""

    root: Path
    manifest: Path
    cohort: Path
    cohort_masks: Path
    features: Path
    feature_fragments: Path
    feature_table: Path
    feature_dictionary: Path
    feature_manifest: Path
    feature_refinement: Path
    feature_ranking: Path
    refinement_metrics: Path
    refinement_summary: Path
    labels: Path
    label_audit: Path
    models: Path
    scores: Path
    exports: Path
    annotations: Path
    logs: Path


def experiment_paths(root: str | Path) -> ExperimentPaths:
    root = Path(root).expanduser().resolve(strict=False)
    return ExperimentPaths(
        root=root,
        manifest=root / "experiment.yaml",
        cohort=root / "cohort" / "eligible_cells.parquet",
        cohort_masks=root / "cohort_masks",
        features=root / "features",
        feature_fragments=root / "features" / "fragments",
        feature_table=root / "features" / "feature_table.parquet",
        feature_dictionary=root / "features" / "feature_dictionary.csv",
        feature_manifest=root / "features" / "feature_manifest.json",
        feature_refinement=root / "feature_refinement",
        feature_ranking=root / "feature_refinement" / "feature_ranking.csv",
        refinement_metrics=root / "feature_refinement" / "fold_metrics.csv",
        refinement_summary=root / "feature_refinement" / "summary.json",
        labels=root / "labels" / "labels.parquet",
        label_audit=root / "labels" / "audit.jsonl",
        models=root / "models",
        scores=root / "scores" / "scores.parquet",
        exports=root / "exports",
        annotations=root / "annotations",
        logs=root / "logs",
    )


def ensure_experiment_directories(paths: ExperimentPaths) -> None:
    for path in (
        paths.root,
        paths.cohort.parent,
        paths.cohort_masks,
        paths.features,
        paths.feature_fragments,
        paths.feature_refinement,
        paths.labels.parent,
        paths.models,
        paths.scores.parent,
        paths.exports,
        paths.annotations,
        paths.logs,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _atomic_replace(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)


def write_dataframe(path: str | Path, frame: pd.DataFrame) -> Path:
    """Atomically write CSV or Parquet based on the destination suffix."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        if destination.suffix.lower() in {".parquet", ".pq"}:
            try:
                frame.to_parquet(temporary, index=False)
            except ImportError as exc:
                raise RuntimeError(
                    "Parquet support requires pyarrow. Install the napari_sbt "
                    "runtime dependencies before writing experiment features."
                ) from exc
        else:
            frame.to_csv(temporary, index=False)
        _atomic_replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def read_dataframe(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    return pd.read_csv(source)


def dataframe_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    """Hash stable string representations of selected dataframe columns."""

    ordered = frame.loc[:, columns].copy()
    ordered = ordered.sort_values(columns, kind="stable").reset_index(drop=True)
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_experiment(
    manifest: ExperimentManifest,
    root: str | Path,
    *,
    audit_action: str = "save_experiment",
) -> ExperimentPaths:
    paths = experiment_paths(root)
    ensure_experiment_directories(paths)
    cosmetic_changes: list[dict[str, Any]] = []
    if paths.manifest.exists():
        existing = read_model(paths.manifest, ExperimentManifest)
        if manifest.experiment_id != existing.experiment_id:
            raise ValueError("An existing experiment ID cannot be replaced in place.")
        if manifest.name != existing.name or manifest.task_type != existing.task_type:
            raise ValueError("Experiment name and task type are immutable.")
        if manifest.revision < existing.revision:
            raise ValueError("Experiment revision cannot move backwards.")
        if manifest.revision > existing.revision + 1:
            raise ValueError("Experiment revisions must advance one revision at a time.")
        if manifest.revision == existing.revision:
            if manifest.cell_scope.model_dump() != existing.cell_scope.model_dump():
                raise ValueError(
                    "The frozen cohort cannot change within an experiment revision. "
                    "Create an explicit revision."
                )
            existing_trial_scope = (
                existing.experiment_mode,
                (
                    existing.feature_trial.roi_selection,
                    existing.feature_trial.roi_count,
                    tuple(existing.feature_trial.selected_rois),
                )
                if existing.feature_trial is not None
                else None,
            )
            current_trial_scope = (
                manifest.experiment_mode,
                (
                    manifest.feature_trial.roi_selection,
                    manifest.feature_trial.roi_count,
                    tuple(manifest.feature_trial.selected_rois),
                )
                if manifest.feature_trial is not None
                else None,
            )
            stored_labels = (
                read_dataframe(paths.labels)
                if paths.labels.exists()
                else pd.DataFrame()
            )
            has_confirmed_labels = bool(
                "state" in stored_labels
                and stored_labels["state"].astype(str).eq("confirmed").any()
            )
            if (
                current_trial_scope != existing_trial_scope
                and (
                    existing.active_feature_set_id is not None
                    or has_confirmed_labels
                )
            ):
                raise ValueError(
                    "The trial/full execution scope is locked after features are "
                    "built or confirmed labels exist. Create an explicit experiment "
                    "revision."
                )
            existing_semantics = [
                (item.class_id, item.shortcut, item.mask_disposition)
                for item in existing.classes
            ]
            current_semantics = [
                (item.class_id, item.shortcut, item.mask_disposition)
                for item in manifest.classes
            ]
            semantics_locked = bool(existing.locked or has_confirmed_labels)
            if semantics_locked and current_semantics != existing_semantics:
                raise ValueError(
                    "Class IDs, shortcuts, ordering, and mask dispositions are locked "
                    "after confirmed labels exist."
                )
            manifest.locked = bool(semantics_locked)
            previous_cosmetics = {
                item.class_id: (item.name, item.color) for item in existing.classes
            }
            for item in manifest.classes:
                if (
                    item.class_id in previous_cosmetics
                    and previous_cosmetics[item.class_id] != (item.name, item.color)
                ):
                    cosmetic_changes.append(
                        {
                            "class_id": item.class_id,
                            "before": previous_cosmetics[item.class_id],
                            "after": (item.name, item.color),
                        }
                    )
        else:
            if (
                manifest.cell_scope.model_dump()
                != existing.cell_scope.model_dump()
            ):
                if (
                    manifest.cell_scope.snapshot_path
                    == existing.cell_scope.snapshot_path
                ):
                    raise ValueError(
                        "A cohort-changing revision must use a new snapshot path so "
                        "the previous frozen cohort remains auditable."
                    )
                revised_snapshot = paths.root / manifest.cell_scope.snapshot_path
                if not revised_snapshot.is_file():
                    raise FileNotFoundError(
                        f"Revised cohort snapshot does not exist: {revised_snapshot}"
                    )
                revised_frame = read_dataframe(revised_snapshot)
                revised_hash = dataframe_sha256(
                    revised_frame, ["obs_name", "ROI", "ObjectNumber"]
                )
                if revised_hash != manifest.cell_scope.snapshot_sha256:
                    raise ValueError(
                        "Revised cohort snapshot fingerprint does not match the "
                        "experiment revision."
                    )
            revisions = paths.root / "revisions"
            revisions.mkdir(parents=True, exist_ok=True)
            write_yaml(
                revisions / f"experiment_r{existing.revision}.yaml",
                existing,
            )
    manifest.updated_at = utc_now()
    write_yaml(paths.manifest, manifest)
    append_audit(
        paths,
        {
            "action": audit_action,
            "experiment_id": manifest.experiment_id,
            "revision": manifest.revision,
        },
    )
    for change in cosmetic_changes:
        append_audit(paths, {"action": "edit_class_cosmetics", **change})
    return paths


def load_experiment(root_or_manifest: str | Path) -> tuple[ExperimentManifest, ExperimentPaths]:
    source = Path(root_or_manifest).expanduser().resolve(strict=False)
    root = source.parent if source.name == "experiment.yaml" else source
    paths = experiment_paths(root)
    manifest = read_model(paths.manifest, ExperimentManifest)
    return manifest, paths


def append_audit(paths: ExperimentPaths, payload: dict[str, Any]) -> Path:
    """Append one JSONL event without rereading and rewriting the full audit."""

    event = {"timestamp": utc_now().isoformat(), **payload}
    encoded = (json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    paths.label_audit.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        paths.label_audit,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0),
        0o666,
    )
    try:
        remaining = memoryview(encoded)
        while remaining:
            remaining = remaining[os.write(descriptor, remaining) :]
    finally:
        os.close(descriptor)
    return paths.label_audit


def feature_recipe_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def write_feature_manifest(paths: ExperimentPaths, payload: dict[str, Any]) -> Path:
    return write_json(paths.feature_manifest, payload)


__all__ = [
    "ExperimentPaths",
    "append_audit",
    "dataframe_sha256",
    "ensure_experiment_directories",
    "experiment_paths",
    "feature_recipe_hash",
    "load_experiment",
    "read_dataframe",
    "save_experiment",
    "write_dataframe",
    "write_feature_manifest",
]
