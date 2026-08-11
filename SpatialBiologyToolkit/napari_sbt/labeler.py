"""Lightweight, cohort-aligned cell labelling for the Napari Labeler tab."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd
from pydantic import BaseModel, model_validator

LABELER_RECORD_COLUMNS = [
    "ROI",
    "ObjectNumber",
    "label_id",
    "user",
    "timestamp",
]

_LABEL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_HEX_COLOUR_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")


class LabelerClass(BaseModel):
    """One mutually exclusive class used by the lightweight cell labeler."""

    label_id: str
    name: str
    color: str

    @model_validator(mode="after")
    def validate_definition(self) -> LabelerClass:
        self.label_id = self.label_id.strip().lower()
        self.name = self.name.strip()
        self.color = self.color.strip().lower()
        if not _LABEL_ID_PATTERN.fullmatch(self.label_id):
            raise ValueError(
                "label_id must start with a letter and contain only lowercase "
                "letters, digits, and underscores."
            )
        if not self.name:
            raise ValueError("Label names must not be empty.")
        if not _HEX_COLOUR_PATTERN.fullmatch(self.color):
            raise ValueError("Label colours must use #RRGGBB notation.")
        return self


def default_labeler_classes() -> list[LabelerClass]:
    """Return a small editable starting palette."""

    return [
        LabelerClass(label_id="label_1", name="Label 1", color="#e11d48"),
        LabelerClass(label_id="label_2", name="Label 2", color="#2563eb"),
    ]


def validate_labeler_classes(
    definitions: Iterable[LabelerClass | dict[str, str]],
) -> list[LabelerClass]:
    """Validate unique label IDs and display names while preserving order."""

    result = [LabelerClass.model_validate(definition) for definition in definitions]
    if not result:
        raise ValueError("Define at least one Labeler class.")
    ids = [definition.label_id for definition in result]
    names = [definition.name.casefold() for definition in result]
    if len(set(ids)) != len(ids):
        raise ValueError("Labeler stable IDs must be unique.")
    if len(set(names)) != len(names):
        raise ValueError("Labeler names must be unique (ignoring case).")
    return result


def empty_labeler_records() -> pd.DataFrame:
    """Return an empty assignment table with a stable schema."""

    return pd.DataFrame(columns=LABELER_RECORD_COLUMNS)


def validate_labeler_records(
    records: pd.DataFrame,
    *,
    label_ids: Iterable[str],
    cohort: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate identities, label membership, and cohort membership."""

    missing = [
        column for column in LABELER_RECORD_COLUMNS if column not in records.columns
    ]
    if missing:
        raise ValueError(f"Labeler table is missing required column(s): {missing}")
    result = records.loc[:, LABELER_RECORD_COLUMNS].copy()
    result["ROI"] = result["ROI"].astype(str)
    object_ids = pd.to_numeric(result["ObjectNumber"], errors="coerce")
    invalid = object_ids.isna() | (object_ids <= 0) | (object_ids % 1 != 0)
    if invalid.any():
        raise ValueError("Labeler ObjectNumber values must be positive integers.")
    result["ObjectNumber"] = object_ids.astype("int64")
    allowed = {str(label_id) for label_id in label_ids}
    unknown = sorted(set(result["label_id"].astype(str)) - allowed)
    if unknown:
        raise ValueError(f"Labeler table contains unknown label IDs: {unknown}")
    for column in ("label_id", "user", "timestamp"):
        result[column] = result[column].astype(str)
    if result.duplicated(["ROI", "ObjectNumber"], keep=False).any():
        raise ValueError("Each cell may have only one active Labeler assignment.")
    if cohort is not None and not result.empty:
        identities = cohort.loc[:, ["ROI", "ObjectNumber"]].copy()
        identities["ROI"] = identities["ROI"].astype(str)
        identities["ObjectNumber"] = pd.to_numeric(
            identities["ObjectNumber"], errors="raise"
        ).astype("int64")
        membership = result.merge(
            identities.drop_duplicates(),
            on=["ROI", "ObjectNumber"],
            how="left",
            indicator=True,
        )
        if membership["_merge"].ne("both").any():
            examples = membership.loc[
                membership["_merge"].ne("both"), ["ROI", "ObjectNumber"]
            ].head()
            raise ValueError(
                "Labeler assignments must be inside the frozen experiment cohort; "
                f"examples: {examples.to_dict('records')}"
            )
    return result


def set_labeler_record(
    records: pd.DataFrame,
    *,
    roi: str,
    object_number: int,
    label_id: str,
    user: str = "",
    timestamp: str | None = None,
) -> pd.DataFrame:
    """Set or replace one cell's mutually exclusive Labeler assignment."""

    current = records.copy() if not records.empty else empty_labeler_records()
    keep = ~(
        current["ROI"].astype(str).eq(str(roi))
        & pd.to_numeric(current["ObjectNumber"], errors="coerce").eq(
            int(object_number)
        )
    )
    row = pd.DataFrame(
        [
            {
                "ROI": str(roi),
                "ObjectNumber": int(object_number),
                "label_id": str(label_id),
                "user": str(user),
                "timestamp": timestamp
                or pd.Timestamp.now(tz="UTC").isoformat(),
            }
        ]
    )
    return pd.concat([current.loc[keep], row], ignore_index=True)


def remove_labeler_record(
    records: pd.DataFrame, *, roi: str, object_number: int
) -> pd.DataFrame:
    """Remove one cell assignment."""

    if records.empty:
        return empty_labeler_records()
    keep = ~(
        records["ROI"].astype(str).eq(str(roi))
        & pd.to_numeric(records["ObjectNumber"], errors="coerce").eq(
            int(object_number)
        )
    )
    return records.loc[keep, LABELER_RECORD_COLUMNS].reset_index(drop=True)


def labeler_summary(
    records: pd.DataFrame,
    definitions: Iterable[LabelerClass],
    *,
    eligible_rois: Iterable[str],
) -> pd.DataFrame:
    """Summarise cell counts and ROI sampling coverage for every label."""

    definitions = validate_labeler_classes(definitions)
    roi_count = len({str(roi) for roi in eligible_rois})
    rows = []
    for definition in definitions:
        selected = records.loc[
            records["label_id"].astype(str).eq(definition.label_id)
        ]
        sampled = int(selected["ROI"].astype(str).nunique())
        rows.append(
            {
                "label_id": definition.label_id,
                "label": definition.name,
                "color": definition.color,
                "cells": int(len(selected)),
                "rois_sampled": sampled,
                "eligible_rois": roi_count,
            }
        )
    return pd.DataFrame(rows)


def build_labeler_export_table(
    records: pd.DataFrame,
    definitions: Iterable[LabelerClass],
    *,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Build an identity-rich, human-readable table for display and export."""

    definitions = validate_labeler_classes(definitions)
    validated = validate_labeler_records(
        records,
        label_ids=[definition.label_id for definition in definitions],
        cohort=cohort,
    )
    names = {definition.label_id: definition.name for definition in definitions}
    colors = {definition.label_id: definition.color for definition in definitions}
    result = validated.copy()
    result["label"] = result["label_id"].map(names)
    result["color"] = result["label_id"].map(colors)
    if "obs_name" in cohort.columns:
        identity_to_obs = cohort.loc[
            :, ["obs_name", "ROI", "ObjectNumber"]
        ].copy()
        identity_to_obs["ROI"] = identity_to_obs["ROI"].astype(str)
        identity_to_obs["ObjectNumber"] = pd.to_numeric(
            identity_to_obs["ObjectNumber"], errors="raise"
        ).astype("int64")
        result = result.merge(
            identity_to_obs,
            on=["ROI", "ObjectNumber"],
            how="left",
            validate="one_to_one",
        )
    columns = [
        column
        for column in (
            "obs_name",
            "ROI",
            "ObjectNumber",
            "label_id",
            "label",
            "color",
            "user",
            "timestamp",
        )
        if column in result.columns
    ]
    return result.loc[:, columns].sort_values(
        ["label", "ROI", "ObjectNumber"], kind="stable"
    ).reset_index(drop=True)


def apply_labeler_to_anndata(
    adata,
    records: pd.DataFrame,
    definitions: Iterable[LabelerClass],
    *,
    cohort: pd.DataFrame,
    obs_name: str,
    overwrite: bool = False,
) -> pd.Series:
    """Apply Labeler names to a new categorical AnnData observation in memory."""

    obs_name = str(obs_name).strip()
    if not obs_name:
        raise ValueError("Provide a non-empty AnnData observation name.")
    if obs_name in adata.obs and not overwrite:
        raise ValueError(
            f"adata.obs[{obs_name!r}] already exists. Enable overwrite explicitly."
        )
    definitions = validate_labeler_classes(definitions)
    table = build_labeler_export_table(records, definitions, cohort=cohort)
    if "obs_name" not in table.columns:
        raise ValueError(
            "The frozen cohort does not contain obs_name identities needed for "
            "safe AnnData assignment."
        )
    missing = sorted(
        set(table["obs_name"].astype(str)) - set(adata.obs_names.astype(str))
    )
    if missing:
        raise ValueError(
            "The live AnnData is missing labelled cohort cells; examples: "
            f"{missing[:10]}"
        )
    values = pd.Series(pd.NA, index=adata.obs_names.astype(str), dtype="string")
    if not table.empty:
        values.loc[table["obs_name"].astype(str)] = (
            table["label"].astype(str).to_numpy()
        )
    categories = [definition.name for definition in definitions]
    adata.obs[obs_name] = pd.Categorical(
        values.reindex(adata.obs_names.astype(str)), categories=categories
    )
    return values


__all__ = [
    "LABELER_RECORD_COLUMNS",
    "LabelerClass",
    "apply_labeler_to_anndata",
    "build_labeler_export_table",
    "default_labeler_classes",
    "empty_labeler_records",
    "labeler_summary",
    "remove_labeler_record",
    "set_labeler_record",
    "validate_labeler_classes",
    "validate_labeler_records",
]
