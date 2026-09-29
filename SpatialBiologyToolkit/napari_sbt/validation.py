"""Asset checks independent of the GUI and workspace creation.

An index locates files; a report records checks actually performed. Quick opening
does neither a recursive discovery nor a dataset-wide mask read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from SpatialBiologyToolkit.qc_classifier.io import (
    discover_mask_files,
    discover_roi_image_index,
    load_mask,
)

from .cohort import validate_mask_coverage


QUICK_WORKFLOWS = frozenset(
    {
        "data_exploration",
        "population_qc",
        "cell_labeling",
        "population_curation",
        "dataset_maintenance",
    }
)


def allows_quick_open(workflow: str | None) -> bool:
    return workflow in QUICK_WORKFLOWS


@dataclass
class AssetCheckResult:
    masks: dict[str, Path] = field(default_factory=dict)
    images: dict[str, dict[str, Path]] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    checked_rois: int = 0
    cancelled: bool = False
    indexed_only: bool = False
    other_mask_labels: int = 0

    @property
    def ok(self) -> bool:
        return not self.cancelled and not self.indexed_only and not self.issues

    def summary(self) -> str:
        if self.cancelled:
            return "Check cancelled. The dataset has not been fully checked."
        if self.indexed_only:
            return "File index prepared. Mask contents and cell matching have not been checked."
        if self.issues:
            return f"Needs review: {len(self.issues)} issue(s). " + self.issues[0]
        return f"Checked masks, cell IDs and image coverage for {self.checked_rois:,} regions."


def check_assets(
    cohort: pd.DataFrame,
    masks_folder: str | Path,
    image_folders: list[str | Path],
    *,
    channel_aliases: dict[str, str] | None = None,
    require_masks: bool = True,
    require_images: bool = True,
    progress: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    index_only: bool = False,
) -> AssetCheckResult:
    """Explicit full coverage check; usable from a background thread or Python.

    Image contents are checked when consumed. This report deliberately describes
    coverage and mask identities, rather than claiming every image was decoded.
    """
    notify = progress or (lambda message: None)
    stopped = cancelled or (lambda: False)
    result = AssetCheckResult()
    notify("Finding asset paths…")
    if stopped():
        result.cancelled = True
        return result
    if masks_folder:
        result.masks = discover_mask_files(masks_folder)
    if stopped():
        result.cancelled = True
        return result
    rois = cohort["ROI"].astype(str).unique()
    result.images = (
        discover_roi_image_index(
            image_folders,
            rois,
            channel_aliases=channel_aliases,
        )
        if image_folders
        else {}
    )
    if stopped():
        result.cancelled = True
        return result
    if index_only:
        result.indexed_only = True
        return result
    for index, (roi, rows) in enumerate(cohort.groupby("ROI", observed=True), 1):
        if stopped():
            result.cancelled = True
            break
        roi = str(roi)
        notify(f"Checking region {index:,} of {len(rois):,}: {roi}")
        mask_path = result.masks.get(roi)
        if mask_path is None:
            if require_masks:
                result.issues.append(f"{roi}: cell outlines were not found.")
        else:
            try:
                mask = load_mask(mask_path)
                missing, other = validate_mask_coverage(
                    mask, rows["ObjectNumber"], roi=roi
                )
                result.other_mask_labels += len(other)
                if len(missing):
                    result.issues.append(
                        f"{roi}: {len(missing):,} cell IDs are missing from the mask."
                    )
            except Exception as exc:
                result.issues.append(f"{roi}: cannot read cell outlines: {exc}")
        if require_images and not result.images.get(roi):
            result.issues.append(f"{roi}: staining images were not found.")
        result.checked_rois += 1
    return result
