"""Versioned workspace models for Napari exploration and classification."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from .feature_catalog import (
    CONTEXT_FEATURE_DESCRIPTIONS,
    DISTRIBUTION_FEATURE_DESCRIPTIONS,
    REGION_IMAGE_FEATURE_DESCRIPTIONS,
    ROI_RANK_FEATURE_DESCRIPTIONS,
    SHAPE_FEATURE_DESCRIPTIONS,
)

SCHEMA_VERSION = 1
# Increment this whenever feature values can change without a manifest recipe or
# input-file change.  It is intentionally separate from the experiment schema:
# old experiments remain loadable, while their cached feature assets are rebuilt.
FEATURE_EXTRACTION_CONTRACT_VERSION = 2
CLASS_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
SHORTCUT_PATTERN = re.compile(r"^[1-8]$")
WorkflowMode = Literal[
    "data_exploration",
    "population_qc",
    "classification",
    "cell_labeling",
    "population_curation",
    "dataset_maintenance",
    "full_workspace",
]


def utc_timestamp() -> datetime:
    return datetime.now(timezone.utc)


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if not value:
        raise ValueError("A non-empty experiment name is required.")
    if value[0].isdigit():
        value = f"experiment_{value}"
    return value


class ClassificationClass(BaseModel):
    """One mutually exclusive class in an experiment."""

    class_id: str
    name: str
    color: str
    shortcut: str
    mask_disposition: Literal["keep", "exclude"] = "keep"

    @model_validator(mode="after")
    def validate_class(self) -> ClassificationClass:
        self.class_id = self.class_id.strip().lower()
        self.name = self.name.strip()
        self.color = self.color.strip()
        self.shortcut = self.shortcut.strip()
        if not CLASS_ID_PATTERN.fullmatch(self.class_id):
            raise ValueError(
                "class_id must start with a letter and contain only lowercase "
                "letters, digits, and underscores."
            )
        if not self.name:
            raise ValueError("Class names must not be empty.")
        if not self.color:
            raise ValueError("Class colours must not be empty.")
        if not SHORTCUT_PATTERN.fullmatch(self.shortcut):
            raise ValueError("Class shortcuts must be one digit from 1 to 8.")
        return self


class CellScope(BaseModel):
    """Definition and frozen identity snapshot for the classified cohort."""

    mode: Literal["all_cells", "obs_values"]
    obs_column: str | None = None
    obs_values: list[str] = Field(default_factory=list)
    snapshot_path: str = "cohort/eligible_cells.parquet"
    snapshot_sha256: str
    eligible_cell_count: int = Field(ge=1)
    total_cell_count: int = Field(ge=1)
    represented_roi_count: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_scope(self) -> CellScope:
        if self.mode == "obs_values":
            if not self.obs_column or not self.obs_column.strip():
                raise ValueError(
                    "obs_values scope requires an AnnData observation column."
                )
            self.obs_column = self.obs_column.strip()
            self.obs_values = [str(value) for value in self.obs_values]
            if not self.obs_values:
                raise ValueError(
                    "obs_values scope requires at least one selected value."
                )
        elif self.obs_column is not None or self.obs_values:
            raise ValueError(
                "all_cells scope must not define obs_column or obs_values."
            )
        if self.eligible_cell_count > self.total_cell_count:
            raise ValueError("Eligible cell count cannot exceed total cell count.")
        return self


class FeatureSource(BaseModel):
    """One identity-aligned source of model features."""

    source_id: str
    kind: Literal["table", "anndata", "synthetic"]
    path: str | None = None
    representation: str | None = None
    selected_columns: list[str] = Field(default_factory=list)
    enabled: bool = True

    @model_validator(mode="after")
    def validate_source(self) -> FeatureSource:
        self.source_id = slugify(self.source_id)
        if self.kind != "synthetic" and not self.path:
            raise ValueError(f"Feature source {self.source_id!r} requires a path.")
        if self.kind == "anndata" and not self.representation:
            self.representation = "X"
        return self


class SyntheticFeatureRecipe(BaseModel):
    """Configuration for cohort-first IMC feature extraction."""

    channels: list[str] = Field(default_factory=list)
    mask_offset_px: int = Field(default=0, ge=-1000, le=1000)
    allow_positive_offset_overlap: bool = False
    distribution_features: bool = True
    region_features: bool = True
    gradient_features: bool = False
    shape_features: bool = True
    context_features: bool = True
    roi_rank_features: bool = True
    distribution_feature_names: list[str] = Field(
        default_factory=lambda: list(DISTRIBUTION_FEATURE_DESCRIPTIONS)
    )
    region_feature_names: list[str] = Field(
        default_factory=lambda: list(REGION_IMAGE_FEATURE_DESCRIPTIONS)
    )
    gradient_feature_names: list[str] = Field(
        default_factory=lambda: list(DISTRIBUTION_FEATURE_DESCRIPTIONS)
    )
    shape_feature_names: list[str] = Field(
        default_factory=lambda: list(SHAPE_FEATURE_DESCRIPTIONS)
    )
    context_feature_names: list[str] = Field(
        default_factory=lambda: list(CONTEXT_FEATURE_DESCRIPTIONS)
    )
    roi_rank_statistics: list[str] = Field(
        default_factory=lambda: list(ROI_RANK_FEATURE_DESCRIPTIONS)
    )
    background_ring_px: int = Field(default=5, ge=1, le=100)
    normalization_dict_path: str | None = None

    @model_validator(mode="after")
    def validate_recipe(self) -> SyntheticFeatureRecipe:
        self.channels = list(dict.fromkeys(str(channel) for channel in self.channels))
        selections = (
            (
                "distribution",
                "distribution_features",
                "distribution_feature_names",
                DISTRIBUTION_FEATURE_DESCRIPTIONS,
            ),
            (
                "region",
                "region_features",
                "region_feature_names",
                REGION_IMAGE_FEATURE_DESCRIPTIONS,
            ),
            (
                "gradient",
                "gradient_features",
                "gradient_feature_names",
                DISTRIBUTION_FEATURE_DESCRIPTIONS,
            ),
            (
                "shape",
                "shape_features",
                "shape_feature_names",
                SHAPE_FEATURE_DESCRIPTIONS,
            ),
            (
                "context",
                "context_features",
                "context_feature_names",
                CONTEXT_FEATURE_DESCRIPTIONS,
            ),
            (
                "ROI-rank",
                "roi_rank_features",
                "roi_rank_statistics",
                ROI_RANK_FEATURE_DESCRIPTIONS,
            ),
        )
        for label, enabled_field, selection_field, catalog in selections:
            selected = list(
                dict.fromkeys(str(value) for value in getattr(self, selection_field))
            )
            unknown = sorted(set(selected) - set(catalog))
            if unknown:
                raise ValueError(f"Unknown {label} feature selection(s): {unknown}")
            setattr(self, selection_field, selected)
            if getattr(self, enabled_field) and not selected:
                raise ValueError(
                    f"Enabled {label} feature family requires at least one "
                    "selected feature."
                )
        if not any(
            (
                self.distribution_features,
                self.region_features,
                self.gradient_features,
                self.shape_features,
                self.context_features,
            )
        ):
            raise ValueError("At least one synthetic feature family must be enabled.")
        return self


class FeatureDiscoveryTrial(BaseModel):
    """Representative-ROI scope and outputs for feature discovery."""

    roi_selection: Literal["largest", "manual"] = "largest"
    roi_count: int = Field(default=3, ge=2, le=10000)
    selected_rois: list[str] = Field(default_factory=list)
    status: Literal["configured", "features_built", "refined", "promoted"] = (
        "configured"
    )
    refinement_report_path: str = "feature_refinement/summary.json"
    recommended_model_features: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_trial(self) -> FeatureDiscoveryTrial:
        self.selected_rois = list(
            dict.fromkeys(
                str(roi).strip() for roi in self.selected_rois if str(roi).strip()
            )
        )
        if self.selected_rois and len(self.selected_rois) != self.roi_count:
            raise ValueError(
                "The feature-discovery ROI count must match the number of selected "
                "ROIs."
            )
        self.recommended_model_features = list(
            dict.fromkeys(str(feature) for feature in self.recommended_model_features)
        )
        return self


class DisplaySettings(BaseModel):
    """Experiment-backed defaults for normalized image display."""

    normalization_dict_path: str | None = None
    fallback_quantile: float = Field(default=0.999, gt=0, le=1)
    minimum_pixel_counts: float = Field(default=0.1, ge=0)
    default_contrast_limits: tuple[float, float] = (0.0, 1.0)

    @model_validator(mode="after")
    def validate_display_settings(self) -> DisplaySettings:
        lower, upper = (float(value) for value in self.default_contrast_limits)
        if not 0 <= lower < upper <= 1:
            raise ValueError(
                "Default normalized image contrast limits must satisfy "
                f"0 <= lower < upper <= 1; got {(lower, upper)!r}."
            )
        self.default_contrast_limits = (lower, upper)
        if self.normalization_dict_path is not None:
            path = str(self.normalization_dict_path).strip()
            self.normalization_dict_path = path or None
        return self


class ExperimentManifest(BaseModel):
    """Canonical, versioned definition of one NapariSBT workflow workspace."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    experiment_id: str = Field(default_factory=lambda: str(uuid4()))
    revision: int = Field(default=1, ge=1)
    name: str
    slug: str | None = None
    workflow_mode: WorkflowMode = "classification"
    task_type: Literal["single_label_multiclass"] = "single_label_multiclass"
    created_at: datetime = Field(default_factory=utc_timestamp)
    updated_at: datetime = Field(default_factory=utc_timestamp)
    project_root: str | None = None
    anndata_path: str | None = None
    images_folders: list[str] = Field(default_factory=list)
    extra_images_folders: list[str] = Field(default_factory=list)
    masks_folder: str
    roi_obs: str = "ROI"
    object_id_obs: str = "ObjectNumber"
    cell_scope: CellScope
    classes: list[ClassificationClass]
    experiment_mode: Literal["full", "feature_discovery_trial"] = "full"
    feature_trial: FeatureDiscoveryTrial | None = None
    feature_sources: list[FeatureSource] = Field(default_factory=list)
    synthetic_features: SyntheticFeatureRecipe = Field(
        default_factory=SyntheticFeatureRecipe
    )
    display_settings: DisplaySettings = Field(default_factory=DisplaySettings)
    active_feature_set_id: str | None = None
    active_model_features: list[str] = Field(default_factory=list)
    output_obs_slug: str | None = None
    annotated_adata_path: str | None = None
    materialize_cohort_masks: bool = False
    locked: bool = False

    @model_validator(mode="after")
    def validate_manifest(self) -> ExperimentManifest:
        self.name = self.name.strip()
        self.slug = slugify(self.slug or self.name)
        self.output_obs_slug = slugify(self.output_obs_slug or self.slug)
        self.roi_obs = self.roi_obs.strip()
        self.object_id_obs = self.object_id_obs.strip()
        if not self.name:
            raise ValueError("Experiment name must not be empty.")
        if not self.roi_obs or not self.object_id_obs:
            raise ValueError("ROI and object-ID observation names are required.")
        if not 2 <= len(self.classes) <= 8:
            raise ValueError("Experiments require between two and eight classes.")
        for attribute, values in {
            "class IDs": [item.class_id for item in self.classes],
            "class names": [item.name.casefold() for item in self.classes],
            "class shortcuts": [item.shortcut for item in self.classes],
        }.items():
            if len(values) != len(set(values)):
                raise ValueError(f"Experiment {attribute} must be unique.")
        source_ids = [source.source_id for source in self.feature_sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Feature source IDs must be unique.")
        if self.cell_scope.mode == "obs_values" and not self.anndata_path:
            raise ValueError("AnnData is required for observation-defined cohorts.")
        if self.experiment_mode == "feature_discovery_trial":
            if self.feature_trial is None or not self.feature_trial.selected_rois:
                raise ValueError(
                    "Feature-discovery trials require an explicit representative-ROI "
                    "selection."
                )
            if self.feature_trial.status == "promoted":
                raise ValueError(
                    "A promoted feature trial must use full experiment mode."
                )
        elif self.feature_trial is not None and self.feature_trial.status != "promoted":
            raise ValueError(
                "A full experiment may retain feature-trial provenance only after "
                "promotion."
            )
        self.active_model_features = list(
            dict.fromkeys(str(feature) for feature in self.active_model_features)
        )
        self.images_folders = [str(Path(path)) for path in self.images_folders]
        self.extra_images_folders = [
            str(Path(path)) for path in self.extra_images_folders
        ]
        return self


def segmentation_qc_classes() -> list[ClassificationClass]:
    """Return the backwards-compatible good/artifact class definition."""

    return [
        ClassificationClass(
            class_id="good",
            name="Good cell",
            color="#1b7837",
            shortcut="1",
            mask_disposition="keep",
        ),
        ClassificationClass(
            class_id="artifact",
            name="Artifact",
            color="#b2182b",
            shortcut="2",
            mask_disposition="exclude",
        ),
    ]


__all__ = [
    "FEATURE_EXTRACTION_CONTRACT_VERSION",
    "SCHEMA_VERSION",
    "CellScope",
    "ClassificationClass",
    "DisplaySettings",
    "ExperimentManifest",
    "FeatureDiscoveryTrial",
    "FeatureSource",
    "SyntheticFeatureRecipe",
    "WorkflowMode",
    "segmentation_qc_classes",
    "slugify",
]
