"""Import-only adapter for features that require CellPose-specific maps/state."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .models import FeatureSource

CELLPOSE_ONLY_PREFIXES = (
    "cellpose_",
    "segmentation_",
    "diameter_",
    "cellprob_threshold",
    "flow_threshold",
)


def is_cellpose_specific_feature(column: str) -> bool:
    """Return whether a feature cannot be derived from ordinary IMC channels."""

    normalized = str(column).strip().lower()
    return normalized.startswith(CELLPOSE_ONLY_PREFIXES)


def partition_cellpose_features(
    columns: Iterable[str],
) -> tuple[list[str], list[str]]:
    cellpose_only: list[str] = []
    generic: list[str] = []
    for column in map(str, columns):
        (cellpose_only if is_cellpose_specific_feature(column) else generic).append(
            column
        )
    return generic, cellpose_only


def cellpose_feature_source(
    path: str | Path,
    *,
    source_id: str = "cellpose",
    selected_columns: Iterable[str] = (),
) -> FeatureSource:
    """Declare an existing CellPose metric table as an identity-aligned source."""

    return FeatureSource(
        source_id=source_id,
        kind="table",
        path=str(path),
        selected_columns=[str(column) for column in selected_columns],
    )


def derive_cellpose_specific_features(*_args, **_kwargs):
    raise RuntimeError(
        "CellPose probability, flow, flow-error, and segmentation-configuration "
        "features require CellPose maps/state and cannot be derived from ordinary "
        "IMC channel images. Import a previously calculated metric table instead."
    )


__all__ = [
    "CELLPOSE_ONLY_PREFIXES",
    "cellpose_feature_source",
    "derive_cellpose_specific_features",
    "is_cellpose_specific_feature",
    "partition_cellpose_features",
]
