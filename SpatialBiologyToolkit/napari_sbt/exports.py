"""Cohort-only tables, atomic AnnData copies, and derived mask exports."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from .cohort import cohort_mask
from .models import ClassificationClass, ExperimentManifest
from .storage import write_dataframe

IDENTITY = ["ROI", "ObjectNumber"]


def build_assignment_table(
    cohort: pd.DataFrame,
    labels: pd.DataFrame,
    scores: pd.DataFrame | None,
    *,
    class_ids: Iterable[str],
    minimum_model_confidence: float = 0.0,
    maximum_model_uncertainty: float = 1.0,
    minimum_probability_margin: float = 0.0,
) -> pd.DataFrame:
    """Build thresholded final assignments with confirmations overriding models."""

    minimum_model_confidence = float(minimum_model_confidence)
    maximum_model_uncertainty = float(maximum_model_uncertainty)
    minimum_probability_margin = float(minimum_probability_margin)
    for name, value in (
        ("minimum_model_confidence", minimum_model_confidence),
        ("maximum_model_uncertainty", maximum_model_uncertainty),
        ("minimum_probability_margin", minimum_probability_margin),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1; received {value}.")

    table = cohort.copy()
    table["ROI"] = table["ROI"].astype(str)
    table["ObjectNumber"] = pd.to_numeric(
        table["ObjectNumber"], errors="raise"
    ).astype("int64")
    class_ids = [str(class_id) for class_id in class_ids]
    if scores is not None and not scores.empty:
        selected_score_columns = [
            column
            for column in scores.columns
            if column in IDENTITY
            or column
            in {
                "predicted_class",
                "maximum_probability",
                "probability_margin",
                "normalized_entropy",
                "model_id",
                "scorable",
            }
            or column.startswith("probability::")
        ]
        table = table.merge(
            scores.loc[:, selected_score_columns],
            on=IDENTITY,
            how="left",
            validate="one_to_one",
        )
        missing_defaults = {
            "predicted_class": pd.NA,
            "maximum_probability": np.nan,
            "probability_margin": np.nan,
            "normalized_entropy": np.nan,
            "model_id": pd.NA,
            "scorable": False,
        }
        for column, default in missing_defaults.items():
            if column not in table:
                table[column] = default
        for class_id in class_ids:
            probability_column = f"probability::{class_id}"
            if probability_column not in table:
                table[probability_column] = np.nan
    else:
        table["predicted_class"] = pd.NA
        table["maximum_probability"] = np.nan
        table["probability_margin"] = np.nan
        table["normalized_entropy"] = np.nan
        table["model_id"] = pd.NA
        table["scorable"] = False
        for class_id in class_ids:
            table[f"probability::{class_id}"] = np.nan

    confirmed = labels.loc[
        labels["state"].eq("confirmed"),
        IDENTITY + ["class_id", "source", "user", "timestamp"],
    ].rename(
        columns={
            "class_id": "confirmed_class",
            "source": "confirmation_source",
            "user": "confirmation_user",
            "timestamp": "confirmed_at",
        }
    )
    table = table.merge(confirmed, on=IDENTITY, how="left", validate="one_to_one")
    is_confirmed = table["confirmed_class"].notna()
    has_prediction = table["predicted_class"].notna()
    scorable = table["scorable"].fillna(False).astype(bool)
    confidence = pd.to_numeric(table["maximum_probability"], errors="coerce")
    uncertainty = pd.to_numeric(table["normalized_entropy"], errors="coerce")
    margin = pd.to_numeric(table["probability_margin"], errors="coerce")
    model_accepted = (
        ~is_confirmed
        & scorable
        & has_prediction
        & confidence.notna()
        & confidence.ge(minimum_model_confidence)
        & uncertainty.notna()
        & uncertainty.le(maximum_model_uncertainty)
        & margin.notna()
        & margin.ge(minimum_probability_margin)
    )
    table["model_prediction_accepted"] = model_accepted
    table["prediction_rejection_reason"] = np.select(
        [
            is_confirmed,
            ~scorable,
            ~has_prediction,
            confidence.isna(),
            confidence.lt(minimum_model_confidence),
            uncertainty.isna(),
            uncertainty.gt(maximum_model_uncertainty),
            margin.isna(),
            margin.lt(minimum_probability_margin),
        ],
        [
            "confirmed_override",
            "unscorable",
            "no_prediction",
            "missing_confidence",
            "below_minimum_confidence",
            "missing_uncertainty",
            "above_maximum_uncertainty",
            "missing_probability_margin",
            "below_minimum_probability_margin",
        ],
        default="accepted",
    )
    table["decision_minimum_confidence"] = minimum_model_confidence
    table["decision_maximum_uncertainty"] = maximum_model_uncertainty
    table["decision_minimum_probability_margin"] = minimum_probability_margin
    table["class_id"] = table["confirmed_class"].where(
        is_confirmed, table["predicted_class"].where(model_accepted)
    )
    table["assignment_source"] = np.select(
        [is_confirmed, model_accepted],
        ["confirmed", "model"],
        default="unassigned",
    )
    table["confidence"] = table["maximum_probability"]
    table.loc[is_confirmed, "confidence"] = 1.0
    table["uncertainty"] = table["normalized_entropy"]
    table.loc[is_confirmed, "uncertainty"] = 0.0
    leading = [
        "obs_name",
        "ROI",
        "ObjectNumber",
        "source_population",
        "class_id",
        "assignment_source",
        "confirmation_source",
        "confidence",
        "uncertainty",
    ]
    return table.loc[
        :,
        [column for column in leading if column in table.columns]
        + [column for column in table.columns if column not in leading],
    ]


def _atomic_h5ad_write(adata, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    try:
        adata.write_h5ad(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def export_annotated_anndata(
    source_path: str | Path,
    destination: str | Path,
    assignments: pd.DataFrame,
    manifest: ExperimentManifest,
    *,
    feature_provenance: Mapping | None = None,
    model_provenance: Mapping | None = None,
    metrics: Mapping | None = None,
) -> Path:
    """Write an annotated copy while leaving cells outside the cohort missing."""

    import anndata as ad

    source = Path(source_path).expanduser().resolve()
    output = Path(destination).expanduser().resolve(strict=False)
    if source == output:
        raise ValueError("Annotated AnnData export must not overwrite the source file.")
    adata = ad.read_h5ad(source)
    apply_assignments_to_anndata(
        adata,
        assignments,
        manifest,
        feature_provenance=feature_provenance,
        model_provenance=model_provenance,
        metrics=metrics,
    )
    return _atomic_h5ad_write(adata, output)


def apply_assignments_to_anndata(
    adata,
    assignments: pd.DataFrame,
    manifest: ExperimentManifest,
    *,
    feature_provenance: Mapping | None = None,
    model_provenance: Mapping | None = None,
    metrics: Mapping | None = None,
):
    """Apply final cohort assignments to an AnnData object in memory.

    This mutates only the supplied object.  It never writes or overwrites its source
    file, which makes the operation suitable for a live notebook session.
    """

    if "obs_name" not in assignments:
        raise ValueError("Assignment table must retain frozen AnnData observation names.")
    if assignments["obs_name"].duplicated().any():
        raise ValueError("Assignment table contains duplicate AnnData observation names.")
    missing_obs = sorted(set(assignments["obs_name"].astype(str)) - set(adata.obs_names))
    if missing_obs:
        raise ValueError(
            "Frozen cohort identities are missing from the source AnnData copy; create "
            f"an explicit experiment revision. Examples: {missing_obs[:10]}"
        )

    slug = str(manifest.output_obs_slug)
    aligned = assignments.set_index(assignments["obs_name"].astype(str))
    class_series = aligned["class_id"].reindex(adata.obs_names)
    source_series = aligned["assignment_source"].reindex(adata.obs_names)
    adata.obs[f"{slug}_subclass"] = pd.Categorical(
        class_series, categories=[item.class_id for item in manifest.classes]
    )
    adata.obs[f"{slug}_source"] = pd.Categorical(
        source_series, categories=["confirmed", "model", "unassigned"]
    )
    adata.obs[f"{slug}_confidence"] = pd.to_numeric(
        aligned["confidence"].reindex(adata.obs_names), errors="coerce"
    ).to_numpy()
    adata.obs[f"{slug}_uncertainty"] = pd.to_numeric(
        aligned["uncertainty"].reindex(adata.obs_names), errors="coerce"
    ).to_numpy()

    probability_columns = [
        f"probability::{item.class_id}" for item in manifest.classes
    ]
    probability_matrix = np.full(
        (adata.n_obs, len(probability_columns)), np.nan, dtype=np.float32
    )
    if all(column in aligned for column in probability_columns):
        probability_matrix[:, :] = (
            aligned.reindex(adata.obs_names)[probability_columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
    adata.obsm[f"{slug}_probabilities"] = probability_matrix

    if manifest.cell_scope.mode == "obs_values":
        source_column = str(manifest.cell_scope.obs_column)
        if source_column not in adata.obs:
            raise ValueError(
                f"Combined export requires source cohort observation {source_column!r}."
            )
        combined = adata.obs[source_column].astype("string").copy()
        eligible = class_series.notna()
        combined.loc[eligible] = class_series.loc[eligible].astype("string")
        adata.obs[f"{slug}_combined"] = pd.Categorical(combined)

    napari_uns = dict(adata.uns.get("napari_sbt", {}))
    napari_uns[slug] = {
        "experiment_id": manifest.experiment_id,
        "experiment_revision": manifest.revision,
        "experiment_mode": manifest.experiment_mode,
        "feature_trial": (
            manifest.feature_trial.model_dump(mode="json")
            if manifest.feature_trial is not None
            else None
        ),
        "active_model_features": list(manifest.active_model_features),
        "class_order": [item.class_id for item in manifest.classes],
        "class_names": {item.class_id: item.name for item in manifest.classes},
        "class_colours": {item.class_id: item.color for item in manifest.classes},
        "cohort_selector": {
            "mode": manifest.cell_scope.mode,
            "obs_column": manifest.cell_scope.obs_column,
            "obs_values": manifest.cell_scope.obs_values,
        },
        "frozen_identity_fingerprint": manifest.cell_scope.snapshot_sha256,
        "feature_provenance": dict(feature_provenance or {}),
        "model_provenance": dict(model_provenance or {}),
        "metrics": dict(metrics or {}),
    }
    adata.uns["napari_sbt"] = napari_uns
    return adata


def export_assignment_table(
    assignments: pd.DataFrame, destination: str | Path
) -> Path:
    return write_dataframe(destination, assignments)


def materialize_cohort_masks(
    masks: Mapping[str, str | Path],
    cohort: pd.DataFrame,
    destination: str | Path,
) -> list[Path]:
    """Write cohort-only masks with original labels preserved."""

    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    eligible = {
        str(roi): set(group["ObjectNumber"].astype(int))
        for roi, group in cohort.groupby("ROI", observed=True)
    }
    written: list[Path] = []
    for roi, identifiers in eligible.items():
        if roi not in masks:
            raise FileNotFoundError(f"No mask was supplied for eligible ROI {roi!r}.")
        mask = tifffile.imread(masks[roi])
        destination_path = output / f"{roi}.tiff"
        tifffile.imwrite(destination_path, cohort_mask(mask, identifiers))
        written.append(destination_path)
    return written


def export_cleaned_masks(
    masks: Mapping[str, str | Path],
    assignments: pd.DataFrame,
    classes: Iterable[ClassificationClass],
    destination: str | Path,
    *,
    prediction_confidence_threshold: float = 0.9,
) -> list[Path]:
    """Remove only confirmed exclusions or high-confidence predicted exclusions."""

    excluded = {
        item.class_id for item in classes if item.mask_disposition == "exclude"
    }
    if not excluded:
        raise ValueError("Cleaned-mask export requires at least one exclude class.")
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for roi, mask_path in masks.items():
        mask = tifffile.imread(mask_path)
        rows = assignments.loc[
            assignments["ROI"].astype(str).eq(str(roi))
            & assignments["class_id"].isin(excluded)
        ]
        remove = rows.loc[
            rows["assignment_source"].eq("confirmed")
            | (
                rows["assignment_source"].eq("model")
                & rows["predicted_class"].eq(rows["class_id"])
                & rows["confidence"].ge(float(prediction_confidence_threshold))
            ),
            "ObjectNumber",
        ].astype(int)
        cleaned = np.where(np.isin(mask, remove.to_numpy()), 0, mask).astype(
            mask.dtype, copy=False
        )
        destination_path = output / f"{roi}.tiff"
        tifffile.imwrite(destination_path, cleaned)
        written.append(destination_path)
    return written


__all__ = [
    "apply_assignments_to_anndata",
    "build_assignment_table",
    "export_annotated_anndata",
    "export_assignment_table",
    "export_cleaned_masks",
    "materialize_cohort_masks",
]
