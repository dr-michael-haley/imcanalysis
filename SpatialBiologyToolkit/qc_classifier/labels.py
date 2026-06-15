"""Confirmed and candidate object-label state management."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from .io import timestamp_utc

CONFIRMED_GOOD = "confirmed_good"
CONFIRMED_ARTIFACT = "confirmed_artifact"
FLAGGED_GOOD = "flagged_good"
FLAGGED_ARTIFACT = "flagged_artifact"

LABEL_STATES = {
    CONFIRMED_GOOD,
    CONFIRMED_ARTIFACT,
    FLAGGED_GOOD,
    FLAGGED_ARTIFACT,
}

STATE_TO_CLASS = {
    CONFIRMED_GOOD: 0,
    CONFIRMED_ARTIFACT: 1,
    FLAGGED_GOOD: 0,
    FLAGGED_ARTIFACT: 1,
}

STATE_TO_LABEL = {
    CONFIRMED_GOOD: "known good cell",
    CONFIRMED_ARTIFACT: "known artifact",
    FLAGGED_GOOD: "candidate good cell",
    FLAGGED_ARTIFACT: "candidate artifact",
}


@dataclass
class LabelRecord:
    """State for one object in one ROI."""

    object_id: int
    label_state: str
    source: str = "manual"
    updated_at: str = field(default_factory=timestamp_utc)

    @property
    def class_label(self) -> int:
        return STATE_TO_CLASS[self.label_state]


@dataclass
class RoiLabels:
    """Mutable label state for one ROI."""

    roi: str
    records: dict[int, LabelRecord] = field(default_factory=dict)

    def clone(self) -> "RoiLabels":
        return RoiLabels(
            roi=self.roi,
            records={
                object_id: LabelRecord(
                    object_id=record.object_id,
                    label_state=record.label_state,
                    source=record.source,
                    updated_at=record.updated_at,
                )
                for object_id, record in self.records.items()
            },
        )

    def ids(self, label_state: str) -> set[int]:
        return {
            object_id
            for object_id, record in self.records.items()
            if record.label_state == label_state
        }

    def state_for(self, object_id: int) -> str | None:
        record = self.records.get(int(object_id))
        return None if record is None else record.label_state

    def set_state(
        self,
        object_id: int,
        label_state: str,
        *,
        source: str = "manual",
        overwrite_confirmed: bool = False,
    ) -> bool:
        """Set the state for one object, returning True if it changed."""

        object_id = int(object_id)
        if label_state not in LABEL_STATES:
            raise ValueError(f"Unknown label state: {label_state}")

        current = self.records.get(object_id)
        if (
            current is not None
            and current.label_state in {CONFIRMED_GOOD, CONFIRMED_ARTIFACT}
            and label_state in {FLAGGED_GOOD, FLAGGED_ARTIFACT}
            and not overwrite_confirmed
        ):
            return False

        if current is not None and current.label_state == label_state:
            return False

        self.records[object_id] = LabelRecord(
            object_id=object_id,
            label_state=label_state,
            source=source,
        )
        return True

    def remove_flagged(self, object_id: int) -> bool:
        """Remove candidate state from one object without touching confirmed labels."""

        object_id = int(object_id)
        current = self.records.get(object_id)
        if current is None or current.label_state not in {FLAGGED_GOOD, FLAGGED_ARTIFACT}:
            return False
        del self.records[object_id]
        return True

    def remove_any(self, object_id: int) -> bool:
        """Remove any label state from one object."""

        object_id = int(object_id)
        if object_id not in self.records:
            return False
        del self.records[object_id]
        return True

    def clear_flagged(self) -> int:
        """Clear all candidate labels and return the number removed."""

        flagged_ids = [
            object_id
            for object_id, record in self.records.items()
            if record.label_state in {FLAGGED_GOOD, FLAGGED_ARTIFACT}
        ]
        for object_id in flagged_ids:
            del self.records[object_id]
        return len(flagged_ids)

    def confirm_flagged(self) -> int:
        """Promote all flagged labels to confirmed training labels."""

        changed = 0
        for object_id, record in list(self.records.items()):
            if record.label_state == FLAGGED_GOOD:
                self.records[object_id] = LabelRecord(
                    object_id=object_id,
                    label_state=CONFIRMED_GOOD,
                    source=record.source,
                )
                changed += 1
            elif record.label_state == FLAGGED_ARTIFACT:
                self.records[object_id] = LabelRecord(
                    object_id=object_id,
                    label_state=CONFIRMED_ARTIFACT,
                    source=record.source,
                )
                changed += 1
        return changed

    def counts(self) -> dict[str, int]:
        return {state: len(self.ids(state)) for state in sorted(LABEL_STATES)}

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for object_id, record in sorted(self.records.items()):
            rows.append(
                {
                    "ROI": self.roi,
                    "ObjectNumber": int(object_id),
                    "label_state": record.label_state,
                    "label_name": STATE_TO_LABEL[record.label_state],
                    "class_label": int(record.class_label),
                    "artifact_probability_target": int(record.class_label),
                    "source": record.source,
                    "updated_at": record.updated_at,
                }
            )
        columns = [
            "ROI",
            "ObjectNumber",
            "label_state",
            "label_name",
            "class_label",
            "artifact_probability_target",
            "source",
            "updated_at",
        ]
        return pd.DataFrame(rows, columns=columns)


def labels_path(labels_folder: str | Path, roi: str) -> Path:
    return Path(labels_folder) / f"{roi}_labels.csv"


def load_roi_labels(labels_folder: str | Path, roi: str) -> RoiLabels:
    """Load saved label states for one ROI, returning an empty set if absent."""

    path = labels_path(labels_folder, roi)
    labels = RoiLabels(str(roi))
    if not path.exists():
        return labels

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return labels
    if df.empty:
        return labels

    object_column = "ObjectNumber" if "ObjectNumber" in df.columns else "object_id"
    state_column = "label_state"
    if object_column not in df.columns or state_column not in df.columns:
        raise ValueError(f"Label file '{path}' must contain ObjectNumber/object_id and label_state columns.")

    for _, row in df.iterrows():
        label_state = str(row[state_column])
        if label_state not in LABEL_STATES:
            continue
        source = str(row["source"]) if "source" in df.columns and not pd.isna(row["source"]) else "imported"
        updated_at = str(row["updated_at"]) if "updated_at" in df.columns and not pd.isna(row["updated_at"]) else timestamp_utc()
        object_id = int(row[object_column])
        labels.records[object_id] = LabelRecord(
            object_id=object_id,
            label_state=label_state,
            source=source,
            updated_at=updated_at,
        )
    return labels


def save_roi_labels(labels_folder: str | Path, labels: RoiLabels) -> Path:
    """Save one ROI's label state table."""

    path = labels_path(labels_folder, labels.roi)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_dataframe().to_csv(path, index=False)
    return path


def load_all_confirmed_labels(labels_folder: str | Path) -> pd.DataFrame:
    """Load confirmed labels from all ROI label files in a folder."""

    labels_folder = Path(labels_folder)
    if not labels_folder.exists():
        return pd.DataFrame(columns=["ROI", "ObjectNumber", "label_state", "class_label"])

    frames = []
    for path in sorted(labels_folder.glob("*_labels.csv")):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty or "label_state" not in df.columns:
            continue
        confirmed = df[df["label_state"].isin([CONFIRMED_GOOD, CONFIRMED_ARTIFACT])].copy()
        if not confirmed.empty:
            confirmed["source_file"] = str(path)
            frames.append(confirmed)

    if not frames:
        return pd.DataFrame(columns=["ROI", "ObjectNumber", "label_state", "class_label"])
    return pd.concat(frames, ignore_index=True)


def label_state_for_ids(labels: RoiLabels, object_ids: Iterable[int]) -> dict[int, str | None]:
    """Return label states for a collection of object IDs."""

    return {int(object_id): labels.state_for(int(object_id)) for object_id in object_ids}
