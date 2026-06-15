"""Loading, selecting, and validating precomputed object-level features."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .feature_dictionary import FeatureDictionary
from .io import load_table


ROI_COLUMN_CANDIDATES = ["ROI", "roi", "roi_name", "image", "image_id", "ImageID"]
OBJECT_COLUMN_CANDIDATES = ["ObjectNumber", "object_number", "label", "label_id", "mask_label", "CellID"]


@dataclass
class FeatureValidationReport:
    """Validation messages and selected model features."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    selected_features: list[str] = field(default_factory=list)
    dropped_features: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def messages(self) -> list[str]:
        return [f"ERROR: {message}" for message in self.errors] + [f"WARNING: {message}" for message in self.warnings]


@dataclass
class FeatureTable:
    """Canonical feature table with ROI and mask-label identifiers."""

    data: pd.DataFrame
    roi_column: str = "ROI"
    object_column: str = "ObjectNumber"
    path: Path | None = None

    def __post_init__(self) -> None:
        missing = [col for col in [self.roi_column, self.object_column] if col not in self.data.columns]
        if missing:
            raise ValueError(f"Feature table is missing required identifier columns: {missing}")
        self.data = self.data.copy()
        self.data[self.roi_column] = self.data[self.roi_column].astype(str)

    @property
    def rois(self) -> list[str]:
        return sorted(self.data[self.roi_column].dropna().astype(str).unique().tolist())

    def rows_for_roi(self, roi: str) -> pd.DataFrame:
        return self.data.loc[self.data[self.roi_column].astype(str) == str(roi), :].copy()

    def object_ids_for_roi(self, roi: str) -> set[int]:
        rows = self.rows_for_roi(roi)
        return set(_safe_int_series(rows[self.object_column]).dropna().astype(int).tolist())

    def selected_feature_frame(self, rows: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
        """Return selected features coerced to finite numeric values where possible."""

        return coerce_feature_frame(rows, feature_columns)


def _infer_column(columns: Iterable[str], candidates: Iterable[str], default: str | None = None) -> str | None:
    columns = list(columns)
    exact = {str(col): str(col) for col in columns}
    lower = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if str(candidate).lower() in lower:
            return lower[str(candidate).lower()]
    return default


def _safe_int_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_feature_table(
    path: str | Path,
    *,
    roi_column: str | None = None,
    object_column: str | None = None,
) -> FeatureTable:
    """Load a precomputed feature table and infer identifier columns if needed."""

    table_path = Path(path)
    df = load_table(table_path)
    roi_column = roi_column or _infer_column(df.columns, ROI_COLUMN_CANDIDATES, default="ROI")
    object_column = object_column or _infer_column(df.columns, OBJECT_COLUMN_CANDIDATES, default="ObjectNumber")
    return FeatureTable(df, roi_column=roi_column, object_column=object_column, path=table_path)


def coerce_feature_frame(rows: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    """Return a numeric feature frame with infinities replaced by NaN."""

    feature_columns = [str(col) for col in feature_columns]
    frame = pd.DataFrame(index=rows.index)
    for col in feature_columns:
        if col not in rows.columns:
            frame[col] = np.nan
            continue
        frame[col] = pd.to_numeric(rows[col], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def select_model_features(
    feature_table: FeatureTable,
    feature_dictionary: FeatureDictionary,
    categories: Iterable[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Select valid numeric feature columns.

    Returns ``(selected_features, dropped_features, warnings)``.
    """

    warnings = []
    dictionary_features = feature_dictionary.eligible_feature_names(categories=categories)
    missing_from_table = [feature for feature in dictionary_features if feature not in feature_table.data.columns]
    if missing_from_table:
        warnings.append(
            f"{len(missing_from_table)} dictionary feature(s) are absent from the feature table: "
            f"{', '.join(missing_from_table[:10])}"
        )

    selected_features = []
    dropped_features = []
    for feature in dictionary_features:
        if feature not in feature_table.data.columns:
            dropped_features.append(feature)
            continue
        values = pd.to_numeric(feature_table.data[feature], errors="coerce")
        non_missing_fraction = float(values.notna().mean()) if len(values) else 0.0
        if non_missing_fraction == 0.0:
            dropped_features.append(feature)
            warnings.append(f"Feature '{feature}' could not be converted to numeric values and was dropped.")
            continue
        if non_missing_fraction < 0.5:
            warnings.append(
                f"Feature '{feature}' has {1.0 - non_missing_fraction:.1%} missing/non-numeric values after conversion."
            )
        selected_features.append(feature)

    dictionary_set = set(feature_dictionary.feature_names)
    missing_from_dictionary = [
        col for col in feature_table.data.columns
        if col not in dictionary_set
    ]
    if missing_from_dictionary:
        warnings.append(
            f"{len(missing_from_dictionary)} feature table column(s) are not present in the dictionary: "
            f"{', '.join(missing_from_dictionary[:10])}"
        )

    return selected_features, dropped_features, warnings


def validate_feature_inputs(
    feature_table: FeatureTable,
    feature_dictionary: FeatureDictionary,
    *,
    categories: Iterable[str] | None = None,
    roi_assets: dict | None = None,
    mask_labels_by_roi: dict[str, set[int]] | None = None,
) -> FeatureValidationReport:
    """Validate feature/dictionary/ROI-mask alignment."""

    report = FeatureValidationReport()
    for column in [feature_table.roi_column, feature_table.object_column]:
        if column not in feature_table.data.columns:
            report.errors.append(f"Required identifier column '{column}' is missing from the feature table.")

    selected, dropped, warnings = select_model_features(feature_table, feature_dictionary, categories=categories)
    report.selected_features = selected
    report.dropped_features = dropped
    report.warnings.extend(warnings)

    if not selected:
        report.errors.append("No numeric model features were selected from the feature dictionary.")

    table_features = set(feature_table.data.columns)
    dictionary_features = set(feature_dictionary.feature_names)
    missing_dictionary_entries = sorted(table_features - dictionary_features)
    if missing_dictionary_entries:
        report.warnings.append(
            f"{len(missing_dictionary_entries)} feature table column(s) lack dictionary entries."
        )

    dictionary_not_found = sorted(dictionary_features - table_features)
    if dictionary_not_found:
        report.warnings.append(
            f"{len(dictionary_not_found)} dictionary row(s) are not present in the feature table."
        )

    if roi_assets is not None:
        feature_rois = set(feature_table.rois)
        asset_rois = set(roi_assets)
        masks_without_features = sorted(
            roi for roi, assets in roi_assets.items()
            if getattr(assets, "has_mask", False) and roi not in feature_rois
        )
        features_without_masks = sorted(roi for roi in feature_rois if roi not in asset_rois or not roi_assets[roi].has_mask)
        images_without_masks = sorted(
            roi for roi, assets in roi_assets.items()
            if getattr(assets, "has_images", False) and not getattr(assets, "has_mask", False)
        )
        masks_without_images = sorted(
            roi for roi, assets in roi_assets.items()
            if getattr(assets, "has_mask", False) and not getattr(assets, "has_images", False)
        )
        if masks_without_features:
            report.warnings.append(f"{len(masks_without_features)} ROI(s) have masks but no feature rows.")
        if features_without_masks:
            report.warnings.append(f"{len(features_without_masks)} ROI(s) have feature rows but no mask file.")
        if images_without_masks:
            report.warnings.append(f"{len(images_without_masks)} ROI(s) have images but no mask file.")
        if masks_without_images:
            report.warnings.append(f"{len(masks_without_images)} ROI(s) have masks but no images.")

    if mask_labels_by_roi is not None:
        for roi, mask_labels in mask_labels_by_roi.items():
            feature_labels = feature_table.object_ids_for_roi(roi)
            missing_in_features = sorted(mask_labels - feature_labels)
            missing_in_mask = sorted(feature_labels - mask_labels)
            if missing_in_features:
                report.warnings.append(
                    f"ROI '{roi}' has {len(missing_in_features)} mask label(s) without feature rows."
                )
            if missing_in_mask:
                report.warnings.append(
                    f"ROI '{roi}' has {len(missing_in_mask)} feature row(s) without corresponding mask labels."
                )

    return report
