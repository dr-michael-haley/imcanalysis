"""Generic proposed and confirmed multiclass label records."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

LABEL_COLUMNS = [
    "ROI",
    "ObjectNumber",
    "class_id",
    "state",
    "source",
    "user",
    "timestamp",
]


def empty_labels() -> pd.DataFrame:
    return pd.DataFrame(columns=LABEL_COLUMNS)


def validate_labels(
    labels: pd.DataFrame,
    *,
    class_ids: Iterable[str],
    cohort: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate identities, class semantics, and at-most-one active record per cell."""

    missing = [column for column in LABEL_COLUMNS if column not in labels.columns]
    if missing:
        raise ValueError(f"Label table is missing required column(s): {missing}")
    result = labels.loc[:, LABEL_COLUMNS].copy()
    result["ROI"] = result["ROI"].astype(str)
    object_ids = pd.to_numeric(result["ObjectNumber"], errors="coerce")
    invalid = object_ids.isna() | (object_ids <= 0) | (object_ids % 1 != 0)
    if invalid.any():
        raise ValueError("Label ObjectNumber values must be positive integers.")
    result["ObjectNumber"] = object_ids.astype("int64")
    allowed_classes = {str(class_id) for class_id in class_ids}
    invalid_classes = sorted(set(result["class_id"].astype(str)) - allowed_classes)
    if invalid_classes:
        raise ValueError(f"Label table contains unknown class IDs: {invalid_classes}")
    invalid_states = sorted(set(result["state"].astype(str)) - {"proposed", "confirmed"})
    if invalid_states:
        raise ValueError(f"Label table contains invalid states: {invalid_states}")
    for column in ("class_id", "state", "source", "user", "timestamp"):
        result[column] = result[column].astype(str)
    duplicates = result.duplicated(["ROI", "ObjectNumber"], keep=False)
    if duplicates.any():
        raise ValueError("Each cohort cell may have only one active label record.")
    if cohort is not None:
        membership = result.merge(
            cohort.loc[:, ["ROI", "ObjectNumber"]].assign(
                ROI=lambda value: value["ROI"].astype(str),
                ObjectNumber=lambda value: pd.to_numeric(
                    value["ObjectNumber"], errors="raise"
                ).astype("int64"),
            ),
            on=["ROI", "ObjectNumber"],
            how="left",
            indicator=True,
        )
        if (membership["_merge"] != "both").any():
            examples = membership.loc[
                membership["_merge"] != "both", ["ROI", "ObjectNumber"]
            ].head()
            raise ValueError(
                "Label records must be inside the frozen experiment cohort; examples: "
                f"{examples.to_dict('records')}"
            )
    return result


def set_label(
    labels: pd.DataFrame,
    *,
    roi: str,
    object_number: int,
    class_id: str,
    state: str,
    source: str = "manual",
    user: str = "",
    timestamp: str | None = None,
) -> pd.DataFrame:
    """Set or replace one cell label."""

    timestamp = timestamp or pd.Timestamp.now(tz="UTC").isoformat()
    current = labels.copy() if not labels.empty else empty_labels()
    keep = ~(
        current["ROI"].astype(str).eq(str(roi))
        & pd.to_numeric(current["ObjectNumber"], errors="coerce").eq(int(object_number))
    )
    row = pd.DataFrame(
        [
            {
                "ROI": str(roi),
                "ObjectNumber": int(object_number),
                "class_id": str(class_id),
                "state": str(state),
                "source": str(source),
                "user": str(user),
                "timestamp": timestamp,
            }
        ]
    )
    return pd.concat([current.loc[keep], row], ignore_index=True)


def remove_label(
    labels: pd.DataFrame, *, roi: str, object_number: int
) -> pd.DataFrame:
    if labels.empty:
        return empty_labels()
    keep = ~(
        labels["ROI"].astype(str).eq(str(roi))
        & pd.to_numeric(labels["ObjectNumber"], errors="coerce").eq(int(object_number))
    )
    return labels.loc[keep, LABEL_COLUMNS].reset_index(drop=True)


def remove_proposed_label(
    labels: pd.DataFrame, *, roi: str, object_number: int
) -> pd.DataFrame:
    """Remove one proposed label while leaving confirmed labels untouched."""

    if labels.empty:
        return empty_labels()
    remove = (
        labels["ROI"].astype(str).eq(str(roi))
        & pd.to_numeric(labels["ObjectNumber"], errors="coerce").eq(
            int(object_number)
        )
        & labels["state"].astype(str).eq("proposed")
    )
    return labels.loc[~remove, LABEL_COLUMNS].reset_index(drop=True)


def remove_all_proposed_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Remove every reversible proposal while preserving confirmed labels."""

    if labels.empty:
        return empty_labels()
    keep = ~labels["state"].astype(str).eq("proposed")
    return labels.loc[keep, LABEL_COLUMNS].reset_index(drop=True)


def confirm_proposed(labels: pd.DataFrame) -> pd.DataFrame:
    result = labels.copy()
    result.loc[result["state"] == "proposed", "state"] = "confirmed"
    result.loc[result["state"] == "confirmed", "timestamp"] = pd.Timestamp.now(
        tz="UTC"
    ).isoformat()
    return result


__all__ = [
    "LABEL_COLUMNS",
    "confirm_proposed",
    "empty_labels",
    "remove_all_proposed_labels",
    "remove_label",
    "remove_proposed_label",
    "set_label",
    "validate_labels",
]
