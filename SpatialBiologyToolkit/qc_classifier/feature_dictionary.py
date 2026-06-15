"""
Feature dictionary handling for object-level CellPose QC models.

The upstream ``cellpose_sam.py`` writes a dictionary CSV with a ``feature``
column, a ``type/source`` category column, and a ``description`` column. This
module also supports richer dictionaries that add boolean include/exclude
columns later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_IDENTIFIER_COLUMNS = {"ROI", "ObjectNumber", "SourceObjectNumber", "CellID"}
DEFAULT_EXCLUDED_CATEGORY_TOKENS = {
    "identifier",
    "metadata",
    "file_metadata",
    "roi_metadata",
    "image_metadata",
    "segmentation_config",
    "excluded",
    "non-feature",
    "non_feature",
}


def _normalise_column_name(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _coerce_bool(value, default: bool | None = None) -> bool | None:
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "include", "included"}:
        return True
    if text in {"0", "false", "f", "no", "n", "exclude", "excluded"}:
        return False
    return default


@dataclass
class FeatureDictionary:
    """Parsed feature dictionary with helper methods for model input selection."""

    table: pd.DataFrame
    feature_column: str = "feature"
    category_column: str | None = None
    description_column: str | None = None

    def __post_init__(self) -> None:
        if self.feature_column not in self.table.columns:
            raise ValueError(f"Feature dictionary is missing feature column '{self.feature_column}'.")
        self.table = self.table.copy()
        self.table[self.feature_column] = self.table[self.feature_column].astype(str)
        if self.category_column is None:
            self.category_column = self._find_column(["type_source", "type/source", "category", "feature_category", "source"])
        if self.description_column is None:
            self.description_column = self._find_column(["description", "desc", "human_readable_description"])

    def _find_column(self, candidates: Iterable[str]) -> str | None:
        normalised_to_original = {_normalise_column_name(col): col for col in self.table.columns}
        for candidate in candidates:
            normalised = _normalise_column_name(candidate)
            if normalised in normalised_to_original:
                return normalised_to_original[normalised]
        return None

    @property
    def feature_names(self) -> list[str]:
        """Feature names in dictionary order."""

        return self.table[self.feature_column].astype(str).tolist()

    @property
    def categories(self) -> list[str]:
        """Sorted available feature categories."""

        if self.category_column is None:
            return ["unclassified"]
        categories = self.table[self.category_column].fillna("unclassified").astype(str).tolist()
        return sorted(set(categories))

    def category_for(self, feature: str) -> str:
        """Return the dictionary category for ``feature``."""

        if self.category_column is None:
            return "unclassified"
        rows = self.table[self.table[self.feature_column].astype(str) == str(feature)]
        if rows.empty:
            return "missing_from_dictionary"
        value = rows.iloc[0][self.category_column]
        if pd.isna(value) or str(value).strip() == "":
            return "unclassified"
        return str(value)

    def description_for(self, feature: str) -> str:
        """Return the plain-language description for ``feature`` when present."""

        if self.description_column is None:
            return ""
        rows = self.table[self.table[self.feature_column].astype(str) == str(feature)]
        if rows.empty:
            return ""
        value = rows.iloc[0][self.description_column]
        return "" if pd.isna(value) else str(value)

    def _row_is_marked_feature(self, row: pd.Series) -> bool:
        """Decide whether a dictionary row is eligible for model input."""

        normalised_columns = {_normalise_column_name(col): col for col in self.table.columns}

        for include_col in [
            "valid_model_input",
            "model_feature",
            "is_feature",
            "use_for_model",
            "include_in_model",
            "include_by_default",
        ]:
            column = normalised_columns.get(include_col)
            if column is not None:
                include_value = _coerce_bool(row[column], default=None)
                if include_value is not None:
                    return include_value

        for exclude_col in ["exclude", "excluded", "exclude_from_model", "non_feature"]:
            column = normalised_columns.get(exclude_col)
            if column is not None and _coerce_bool(row[column], default=False):
                return False

        feature_name = str(row[self.feature_column])
        if feature_name in DEFAULT_IDENTIFIER_COLUMNS:
            return False

        category = ""
        if self.category_column is not None and self.category_column in row.index:
            category = "" if pd.isna(row[self.category_column]) else str(row[self.category_column]).strip().lower()

        if category in DEFAULT_EXCLUDED_CATEGORY_TOKENS:
            return False
        if any(token in category for token in ["metadata", "identifier", "config", "excluded", "non_feature", "non-feature"]):
            return False

        return True

    def eligible_feature_names(self, categories: Iterable[str] | None = None) -> list[str]:
        """
        Return dictionary features eligible for model input.

        When ``categories`` is supplied, only eligible features in those
        categories are returned.
        """

        selected_categories = None
        if categories is not None:
            selected_categories = {str(category) for category in categories}

        features = []
        for _, row in self.table.iterrows():
            feature_name = str(row[self.feature_column])
            if not self._row_is_marked_feature(row):
                continue
            if selected_categories is not None and self.category_for(feature_name) not in selected_categories:
                continue
            features.append(feature_name)
        return features

    def to_metadata(self, selected_categories: Iterable[str] | None, selected_features: Iterable[str]) -> dict:
        """Return dictionary-derived model metadata."""

        selected_features = [str(feature) for feature in selected_features]
        return {
            "feature_dictionary_columns": list(self.table.columns),
            "feature_column": self.feature_column,
            "category_column": self.category_column,
            "description_column": self.description_column,
            "selected_feature_categories": None if selected_categories is None else [str(c) for c in selected_categories],
            "selected_feature_count": len(selected_features),
            "selected_features": selected_features,
            "selected_feature_categories_by_feature": {
                feature: self.category_for(feature) for feature in selected_features
            },
        }


def load_feature_dictionary(path: str | Path) -> FeatureDictionary:
    """Load a feature dictionary CSV."""

    dictionary_path = Path(path)
    df = pd.read_csv(dictionary_path)

    if "feature" not in df.columns:
        first_column = df.columns[0] if len(df.columns) else None
        if first_column is not None:
            df = df.rename(columns={first_column: "feature"})

    return FeatureDictionary(df)
