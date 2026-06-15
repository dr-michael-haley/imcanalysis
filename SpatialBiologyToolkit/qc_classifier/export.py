"""Cleaned-mask and per-cell QC table export."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import skimage as sk

from .io import timestamp_utc, write_json
from .labels import CONFIRMED_ARTIFACT, FLAGGED_ARTIFACT, RoiLabels

EXPORT_CONFIRMED_ARTIFACTS = "confirmed artifacts only"
EXPORT_CONFIRMED_AND_FLAGGED_ARTIFACTS = "confirmed + flagged artifacts"
EXPORT_CLASSIFIER_THRESHOLD = "classifier threshold"
EXPORT_COMBINED = "confirmed/flagged artifacts + classifier threshold"
UINT16_MAX_LABEL = int(np.iinfo(np.uint16).max)


def _require_uint16_label_range(mask: np.ndarray, *, context: str) -> None:
    """Raise a clear error before label IDs can overflow a uint16 export."""

    if mask.size == 0:
        return
    min_label = int(np.nanmin(mask))
    max_label = int(np.nanmax(mask))
    if min_label < 0 or max_label > UINT16_MAX_LABEL:
        raise ValueError(
            f"{context} contains labels outside the uint16 range 0-{UINT16_MAX_LABEL}. "
            "Enable relabeling during export or use fewer preserved label IDs."
        )


def decide_excluded_object_ids(
    labels: RoiLabels,
    scores: pd.DataFrame | None,
    *,
    rule: str,
    threshold: float = 0.8,
) -> set[int]:
    """Return object IDs excluded by a named export rule."""

    rule = str(rule)
    excluded = set(labels.ids(CONFIRMED_ARTIFACT))
    if rule in {EXPORT_CONFIRMED_AND_FLAGGED_ARTIFACTS, EXPORT_COMBINED}:
        excluded |= set(labels.ids(FLAGGED_ARTIFACT))

    if rule in {EXPORT_CLASSIFIER_THRESHOLD, EXPORT_COMBINED} and scores is not None and not scores.empty:
        if "ObjectNumber" in scores.columns and "artifact_probability" in scores.columns:
            threshold_ids = (
                scores.loc[pd.to_numeric(scores["artifact_probability"], errors="coerce") >= float(threshold), "ObjectNumber"]
                .dropna()
                .astype(int)
                .tolist()
            )
            excluded |= set(threshold_ids)

    if rule == EXPORT_CLASSIFIER_THRESHOLD:
        excluded -= set(labels.ids(CONFIRMED_ARTIFACT))
        if scores is not None and not scores.empty and "artifact_probability" in scores.columns:
            return set(
                scores.loc[pd.to_numeric(scores["artifact_probability"], errors="coerce") >= float(threshold), "ObjectNumber"]
                .dropna()
                .astype(int)
                .tolist()
            )

    return excluded


def build_cleaned_mask(
    mask: np.ndarray,
    excluded_object_ids: Iterable[int],
    *,
    relabel: bool = False,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    """Return a cleaned mask, optionally relabelled, without modifying the source."""

    excluded = np.asarray(sorted(set(int(x) for x in excluded_object_ids)), dtype=np.int64)
    cleaned = np.asarray(mask).copy()
    if excluded.size:
        cleaned[np.isin(cleaned, excluded)] = 0

    if not relabel:
        _require_uint16_label_range(cleaned, context="Cleaned mask")
        return cleaned.astype(np.uint16, copy=False), None

    kept_labels = [int(label) for label in np.unique(cleaned) if int(label) != 0]
    if len(kept_labels) > UINT16_MAX_LABEL:
        raise ValueError(
            f"Relabelled export would require {len(kept_labels)} labels, which exceeds "
            f"the uint16 maximum of {UINT16_MAX_LABEL}."
        )
    mapping_rows = []
    relabelled = np.zeros_like(cleaned, dtype=np.uint16)
    for new_label, old_label in enumerate(kept_labels, start=1):
        relabelled[cleaned == old_label] = new_label
        mapping_rows.append({"old_ObjectNumber": old_label, "new_ObjectNumber": new_label})
    return relabelled, pd.DataFrame(mapping_rows)


def build_per_cell_qc_table(
    roi: str,
    feature_rows: pd.DataFrame,
    labels: RoiLabels,
    scores: pd.DataFrame | None,
    excluded_object_ids: Iterable[int],
    *,
    rule: str,
    threshold: float,
    feature_columns: Iterable[str] | None = None,
    model_id: str | None = None,
    feature_set_hash: str | None = None,
) -> pd.DataFrame:
    """Build the auditable per-cell QC table for an exported mask."""

    feature_rows = feature_rows.copy()
    object_column = "ObjectNumber" if "ObjectNumber" in feature_rows.columns else None
    if object_column is None:
        object_candidates = [col for col in feature_rows.columns if col.lower() in {"objectnumber", "object_number", "label_id"}]
        object_column = object_candidates[0] if object_candidates else None
    if object_column is None:
        raise ValueError("Feature rows must contain an ObjectNumber column for export.")

    qc = feature_rows.loc[:, [col for col in ["ROI", object_column] if col in feature_rows.columns]].copy()
    qc = qc.rename(columns={object_column: "ObjectNumber"})
    qc["ROI"] = str(roi)
    qc["ObjectNumber"] = pd.to_numeric(qc["ObjectNumber"], errors="coerce").astype("Int64")

    if scores is not None and not scores.empty:
        score_cols = scores.loc[:, ["ObjectNumber", "artifact_probability"]].copy()
        score_cols["ObjectNumber"] = pd.to_numeric(score_cols["ObjectNumber"], errors="coerce").astype("Int64")
        qc = qc.merge(score_cols, how="left", on="ObjectNumber")
    else:
        qc["artifact_probability"] = np.nan

    state_by_id = {
        object_id: record.label_state
        for object_id, record in labels.records.items()
    }
    excluded = set(int(x) for x in excluded_object_ids)
    qc["label_state"] = qc["ObjectNumber"].map(lambda value: state_by_id.get(int(value)) if not pd.isna(value) else None)
    qc["keep"] = qc["ObjectNumber"].map(lambda value: False if not pd.isna(value) and int(value) in excluded else True)
    qc["exclude_reason"] = qc["keep"].map(lambda keep: "" if keep else rule)
    qc["export_rule"] = rule
    qc["export_threshold"] = float(threshold)
    qc["model_id"] = model_id
    qc["selected_feature_set_hash"] = feature_set_hash
    qc["selected_feature_count"] = 0 if feature_columns is None else len(list(feature_columns))
    qc["exported_at"] = timestamp_utc()
    return qc


def export_cleaned_mask_and_table(
    *,
    roi: str,
    mask: np.ndarray,
    feature_rows: pd.DataFrame,
    labels: RoiLabels,
    scores: pd.DataFrame | None,
    output_folder: str | Path,
    rule: str,
    threshold: float = 0.8,
    feature_columns: Iterable[str] | None = None,
    model_metadata: dict | None = None,
    relabel: bool = False,
) -> dict[str, str | int | float | None]:
    """Export a cleaned mask, per-cell QC table, optional mapping, and metadata."""

    output_folder = Path(output_folder)
    feature_columns = [] if feature_columns is None else [str(feature) for feature in feature_columns]
    masks_folder = output_folder / "masks_cleaned"
    exports_folder = output_folder / "exports"
    metadata_folder = output_folder / "metadata"
    masks_folder.mkdir(parents=True, exist_ok=True)
    exports_folder.mkdir(parents=True, exist_ok=True)
    metadata_folder.mkdir(parents=True, exist_ok=True)

    excluded = decide_excluded_object_ids(labels, scores, rule=rule, threshold=threshold)
    cleaned, mapping = build_cleaned_mask(mask, excluded, relabel=relabel)
    mask_path = masks_folder / f"{roi}.tiff"
    sk.io.imsave(mask_path, cleaned.astype(np.uint16, copy=False), check_contrast=False)

    model_id = None if model_metadata is None else model_metadata.get("model_id")
    feature_set_hash = None if model_metadata is None else model_metadata.get("feature_set_hash")
    qc_table = build_per_cell_qc_table(
        roi,
        feature_rows,
        labels,
        scores,
        excluded,
        rule=rule,
        threshold=threshold,
        feature_columns=feature_columns,
        model_id=model_id,
        feature_set_hash=feature_set_hash,
    )
    qc_path = exports_folder / f"{roi}_cell_qc.csv"
    qc_table.to_csv(qc_path, index=False)

    mapping_path = None
    if mapping is not None:
        mapping_path = exports_folder / f"{roi}_label_mapping.csv"
        mapping.to_csv(mapping_path, index=False)

    metadata = {
        "roi": str(roi),
        "cleaned_mask": str(mask_path),
        "per_cell_qc_table": str(qc_path),
        "label_mapping": None if mapping_path is None else str(mapping_path),
        "excluded_object_count": int(len(excluded)),
        "export_rule": rule,
        "export_threshold": float(threshold),
        "mask_dtype": "uint16",
        "preserved_original_label_ids": not relabel,
        "model_id": model_id,
        "feature_set_hash": feature_set_hash,
        "selected_features": feature_columns,
        "exported_at": timestamp_utc(),
    }
    metadata_path = metadata_folder / f"{roi}_export_metadata.json"
    write_json(metadata_path, metadata)
    metadata["metadata"] = str(metadata_path)
    return metadata
