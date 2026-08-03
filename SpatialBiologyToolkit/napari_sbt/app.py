"""Unified Napari dock for cohort-first IMC exploration and classification."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from SpatialBiologyToolkit.pipeline.manifests import write_json
from SpatialBiologyToolkit.qc_classifier.io import (
    build_image_channel_aliases,
    discover_mask_files,
    discover_roi_images,
    load_display_image,
    load_mask,
)

from .classifier import (
    HGB_MIN_SAMPLES_LEAF,
    confirmed_labels_fingerprint,
    feature_set_hash,
    high_confidence_queue,
    save_model_bundle,
    score_cohort,
    train_multiclass_classifier,
    uncertainty_queue,
)
from .cohort import (
    CohortPreview,
    cohort_mask,
    resolve_cohort,
    resolve_table_cohort,
    save_cohort_snapshot,
    validate_mask_coverage,
)
from .exports import (
    build_assignment_table,
    export_annotated_anndata,
    export_assignment_table,
    export_cleaned_masks,
    materialize_cohort_masks,
)
from .explore import (
    SIX_COLOUR_COLORMAPS,
    ExploreReviewState,
    ExploreViewRecipe,
    categorical_colour_map,
    marker_values,
    population_recipe_key,
)
from .feature_catalog import (
    FEATURE_FAMILY_CATALOG,
    FEATURE_FAMILY_DESCRIPTIONS,
)
from .feature_refinement import compact_synthetic_recipe
from .labels import confirm_proposed, empty_labels, set_label, validate_labels
from .models import (
    ClassificationClass,
    ExperimentManifest,
    FeatureDiscoveryTrial,
    FeatureSource,
    SyntheticFeatureRecipe,
    segmentation_qc_classes,
    slugify,
)
from .resources import resolve_worker_count
from .storage import (
    append_audit,
    dataframe_sha256,
    load_experiment,
    read_dataframe,
    save_experiment,
    write_dataframe,
)

CLASS_LAYER_NAMES = {
    "confirmed": "confirmed_classes",
    "proposed": "proposed_classes",
    "predicted": "predicted_classes",
    "uncertainty": "uncertainty_or_probability",
}

SELECTED_CELL_LAYER_NAME = "selected_cell_outline"

MANAGED_RECIPE_LAYERS = {
    "classification_cohort": "Eligible-cell classification mask",
    "excluded_segmentation_context": "Excluded-cell segmentation context",
    CLASS_LAYER_NAMES["confirmed"]: "Classifier: confirmed classes",
    CLASS_LAYER_NAMES["proposed"]: "Classifier: proposed classes",
    CLASS_LAYER_NAMES["predicted"]: "Classifier: predicted classes",
    CLASS_LAYER_NAMES["uncertainty"]: (
        "Classifier: uncertainty or selected-class probability"
    ),
    SELECTED_CELL_LAYER_NAME: "Classifier: currently selected cell",
}

MANAGED_LAYER_DEFAULT_VISIBILITY = {
    "classification_cohort": False,
    "excluded_segmentation_context": False,
    CLASS_LAYER_NAMES["confirmed"]: True,
    CLASS_LAYER_NAMES["proposed"]: True,
    CLASS_LAYER_NAMES["predicted"]: False,
    CLASS_LAYER_NAMES["uncertainty"]: True,
    SELECTED_CELL_LAYER_NAME: True,
}

MANAGED_LAYER_DEFAULT_OPACITY = {
    "classification_cohort": 1.0,
    "excluded_segmentation_context": 0.18,
    CLASS_LAYER_NAMES["confirmed"]: 1.0,
    CLASS_LAYER_NAMES["proposed"]: 1.0,
    CLASS_LAYER_NAMES["predicted"]: 1.0,
    CLASS_LAYER_NAMES["uncertainty"]: 1.0,
    SELECTED_CELL_LAYER_NAME: 1.0,
}

MANAGED_LAYER_DEFAULT_CONTOUR = {
    "classification_cohort": 1,
    "excluded_segmentation_context": 1,
    CLASS_LAYER_NAMES["confirmed"]: 0,
    CLASS_LAYER_NAMES["proposed"]: 2,
    CLASS_LAYER_NAMES["predicted"]: 1,
    SELECTED_CELL_LAYER_NAME: 2,
}


def _path_text(value: str | Path | None) -> str:
    return "" if value is None else str(Path(value))


def _split_paths(value: str) -> list[str]:
    return [item.strip() for item in value.replace(";", "\n").splitlines() if item.strip()]


def _identity_value_map(
    mask: np.ndarray,
    values: pd.Series,
    *,
    dtype=np.float32,
    background_value: float | int = 0,
) -> np.ndarray:
    output = np.full(mask.shape, background_value, dtype=dtype)
    for object_id, value in values.items():
        if pd.notna(value):
            output[mask == int(object_id)] = value
    return output


class NapariSBTController:
    """Qt-independent state plus Qt/Napari callbacks for one workflow dock."""

    def __init__(
        self,
        viewer,
        *,
        project_root: str | Path | None = None,
        experiment: str | Path | None = None,
        anndata_path: str | Path | None = None,
        masks_folder: str | Path | None = None,
        images_folders: Iterable[str | Path] = (),
        extra_images_folders: Iterable[str | Path] = (),
    ) -> None:
        from qtpy.QtCore import QTimer, Qt
        from qtpy.QtGui import QColor, QFont, QIcon, QPixmap
        from qtpy.QtWidgets import (
            QAbstractItemView,
            QButtonGroup,
            QCheckBox,
            QColorDialog,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QLineEdit,
            QListWidget,
            QMessageBox,
            QPushButton,
            QProgressBar,
            QRadioButton,
            QScrollArea,
            QSpinBox,
            QTableWidget,
            QTableWidgetItem,
            QTabWidget,
            QTextBrowser,
            QTextEdit,
            QTreeWidget,
            QTreeWidgetItem,
            QVBoxLayout,
            QWidget,
        )

        self.Qt = Qt
        self.QMessageBox = QMessageBox
        self.QFileDialog = QFileDialog
        self.QColorDialog = QColorDialog
        self.QDialog = QDialog
        self.QDialogButtonBox = QDialogButtonBox
        self.QColor = QColor
        self.QIcon = QIcon
        self.QPixmap = QPixmap
        self.QTableWidgetItem = QTableWidgetItem
        self.QTextBrowser = QTextBrowser
        self.QTreeWidgetItem = QTreeWidgetItem
        self.viewer = viewer
        self.project_root = (
            Path(project_root).expanduser().resolve(strict=False)
            if project_root
            else Path.cwd()
        )
        self.manifest: ExperimentManifest | None = None
        self.paths = None
        self.adata = None
        self.preview: CohortPreview | None = None
        self.cohort = pd.DataFrame()
        self.labels = empty_labels()
        self.scores = pd.DataFrame()
        self.model_bundle = None
        self.current_roi: str | None = None
        self.current_mask: np.ndarray | None = None
        self.current_mask_path: Path | None = None
        self.current_selected_object: int | None = None
        self.feature_process = None
        self.source_validation_process = None
        self.refinement_process = None
        self.refinement_cancel_requested = False
        self.feature_build_started_at: float | None = None
        self.feature_last_event_at: float | None = None
        self.feature_progress_state: dict[str, int | float | str] = {}
        self._feature_output_buffer = ""
        self._source_validation_output_buffer = ""
        self._refinement_output_buffer = ""
        self.reviewed_rois: set[str] = set()
        self._class_shortcuts: list[str] = []
        self.current_image_paths: dict[str, Path] = {}
        self.explore_recipe = ExploreViewRecipe()
        self.explore_review_state = ExploreReviewState()
        self._explore_layer_names: set[str] = set()
        self._applying_explore_recipe = False
        self._updating_recipe_layer_state = False
        self._updating_queue_controls = False
        self.cell_picking_enabled = True
        self.classifier_display_dialog = None
        self.classifier_visibility_controls: dict[str, object] = {}
        self.classifier_opacity_controls: dict[str, object] = {}
        self.classifier_contour_controls: dict[str, object] = {}
        self._retained_feature_source_columns: dict[
            tuple[str, str, str, str], list[str]
        ] = {}

        self.root = QWidget()
        self.feature_health_timer = QTimer(self.root)
        self.feature_health_timer.setInterval(1000)
        self.feature_health_timer.timeout.connect(self._update_feature_process_health)
        root_layout = QVBoxLayout(self.root)
        self.scope_label = QLabel("No experiment: classification is disabled.")
        self.scope_label.setWordWrap(True)
        root_layout.addWidget(self.scope_label)
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        def add_tab(widget, title: str, help_topic: str) -> None:
            help_row = QHBoxLayout()
            help_row.addStretch(1)
            help_button = QPushButton("❓ Help for this tab")
            help_button.clicked.connect(
                lambda _checked=False, topic=help_topic, tab_title=title: (
                    self.show_tab_help(topic, tab_title)
                )
            )
            help_row.addWidget(help_button)
            widget.layout().insertLayout(0, help_row)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            self.tabs.addTab(scroll, title)

        # Setup
        setup = QWidget()
        setup_layout = QVBoxLayout(setup)
        inputs = QGroupBox("Dataset inputs")
        inputs_form = QFormLayout(inputs)
        self.name_edit = QLineEdit("Cell classification")
        self.anndata_edit = QLineEdit(_path_text(anndata_path))
        self.masks_edit = QLineEdit(_path_text(masks_folder))
        self.images_edit = QTextEdit("\n".join(map(str, images_folders)))
        self.images_edit.setMaximumHeight(70)
        self.extra_images_edit = QTextEdit("\n".join(map(str, extra_images_folders)))
        self.extra_images_edit.setMaximumHeight(55)
        self.experiment_edit = QLineEdit(_path_text(experiment))
        self.roi_obs_edit = QLineEdit("ROI")
        self.object_obs_edit = QLineEdit("ObjectNumber")
        inputs_form.addRow("Experiment name", self.name_edit)
        inputs_form.addRow("AnnData", self.anndata_edit)
        inputs_form.addRow("Masks folder", self.masks_edit)
        inputs_form.addRow("IMC image folders", self.images_edit)
        inputs_form.addRow("Extra image folders", self.extra_images_edit)
        inputs_form.addRow("Experiment folder", self.experiment_edit)
        inputs_form.addRow("ROI identity observation", self.roi_obs_edit)
        inputs_form.addRow("Mask object-ID observation", self.object_obs_edit)
        setup_layout.addWidget(inputs)

        scope_group = QGroupBox("1. Required cell scope")
        scope_grid = QGridLayout(scope_group)
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("All cells", "all_cells")
        self.scope_combo.addItem("Selected adata.obs values", "obs_values")
        self.obs_combo = QComboBox()
        self.value_list = QListWidget()
        self.value_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.value_list.setMaximumHeight(105)
        self.load_adata_button = QPushButton("Load AnnData selectors")
        self.preview_button = QPushButton("Preview and validate cohort")
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(150)
        scope_grid.addWidget(QLabel("Scope"), 0, 0)
        scope_grid.addWidget(self.scope_combo, 0, 1)
        scope_grid.addWidget(self.load_adata_button, 0, 2)
        scope_grid.addWidget(QLabel("Observation"), 1, 0)
        scope_grid.addWidget(self.obs_combo, 1, 1, 1, 2)
        scope_grid.addWidget(QLabel("Selected values"), 2, 0)
        scope_grid.addWidget(self.value_list, 2, 1, 1, 2)
        scope_grid.addWidget(self.preview_button, 3, 0, 1, 3)
        scope_grid.addWidget(self.preview_text, 4, 0, 1, 3)
        setup_layout.addWidget(scope_group)

        trial_group = QGroupBox("2. Experiment mode and feature-discovery ROIs")
        trial_form = QFormLayout(trial_group)
        self.experiment_mode_combo = QComboBox()
        self.experiment_mode_combo.addItem("Full experiment", "full")
        self.experiment_mode_combo.addItem(
            "Feature Discovery Trial", "feature_discovery_trial"
        )
        self.trial_roi_count_spin = QSpinBox()
        self.trial_roi_count_spin.setRange(2, 10000)
        self.trial_roi_count_spin.setValue(3)
        self.trial_roi_strategy_combo = QComboBox()
        self.trial_roi_strategy_combo.addItem(
            "Largest eligible-cell ROIs", "largest"
        )
        self.trial_roi_strategy_combo.addItem("Choose manually", "manual")
        self.trial_roi_list = QListWidget()
        self.trial_roi_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.trial_roi_list.setMaximumHeight(145)
        self.trial_roi_summary = QLabel(
            "Preview the cohort to choose representative trial ROIs."
        )
        self.trial_roi_summary.setWordWrap(True)
        self.suggest_trial_rois_button = QPushButton(
            "Select suggested representative ROIs"
        )
        trial_form.addRow("Workflow", self.experiment_mode_combo)
        trial_form.addRow("Number of trial ROIs", self.trial_roi_count_spin)
        trial_form.addRow("ROI selection", self.trial_roi_strategy_combo)
        trial_form.addRow("Representative ROIs", self.trial_roi_list)
        trial_form.addRow("", self.suggest_trial_rois_button)
        trial_form.addRow("Trial scope", self.trial_roi_summary)
        setup_layout.addWidget(trial_group)

        class_group = QGroupBox("3. Mutually exclusive classes (2–8)")
        class_layout = QVBoxLayout(class_group)
        self.class_table = QTableWidget(0, 5)
        self.class_table.setHorizontalHeaderLabels(
            ["Stable ID", "Name", "Colour", "Shortcut", "Mask disposition"]
        )
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        class_buttons = QHBoxLayout()
        self.add_class_button = QPushButton("Add class")
        self.remove_class_button = QPushButton("Remove selected class")
        self.qc_template_button = QPushButton("Segmentation QC template")
        self.apply_classes_button = QPushButton("Apply class edits")
        class_buttons.addWidget(self.add_class_button)
        class_buttons.addWidget(self.remove_class_button)
        class_buttons.addWidget(self.qc_template_button)
        class_buttons.addWidget(self.apply_classes_button)
        class_layout.addWidget(self.class_table)
        class_layout.addLayout(class_buttons)
        setup_layout.addWidget(class_group)

        setup_actions = QHBoxLayout()
        self.create_button = QPushButton("Create confirmed experiment")
        self.load_experiment_button = QPushButton("Load experiment")
        setup_actions.addWidget(self.create_button)
        setup_actions.addWidget(self.load_experiment_button)
        setup_layout.addLayout(setup_actions)
        add_tab(setup, "⚙ Setup", "setup")

        # Feature Building
        feature_builder = QWidget()
        feature_builder_layout = QVBoxLayout(feature_builder)

        source_group = QGroupBox("1. Imported feature sources")
        source_form = QFormLayout(source_group)
        self.feature_tables_edit = QTextEdit()
        self.feature_tables_edit.setPlaceholderText(
            "Optional, one per line: source_id=features.parquet"
        )
        self.feature_tables_edit.setMaximumHeight(65)
        self.anndata_features_edit = QTextEdit()
        self.anndata_features_edit.setPlaceholderText(
            "Optional, one per line: cellvision=embeddings.h5ad::X_cellvision"
        )
        self.anndata_features_edit.setMaximumHeight(65)
        self.validate_sources_button = QPushButton(
            "Validate identities, cohort coverage, and numeric features"
        )
        self.source_validation_table = QTableWidget(0, 7)
        self.source_validation_table.setHorizontalHeaderLabels(
            [
                "Source",
                "Kind",
                "Status",
                "Covered",
                "Missing",
                "Features",
                "Details",
            ]
        )
        self.source_validation_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.source_validation_table.horizontalHeader().setSectionResizeMode(
            6, QHeaderView.Stretch
        )
        self.source_validation_table.setMaximumHeight(180)
        source_form.addRow("CSV / Parquet tables", self.feature_tables_edit)
        source_form.addRow(
            "AnnData / CellVision sources",
            self.anndata_features_edit,
        )
        source_form.addRow("", self.validate_sources_button)
        source_form.addRow("Validation results", self.source_validation_table)
        feature_builder_layout.addWidget(source_group)

        channel_group = QGroupBox("2. IMC channels")
        channel_layout = QVBoxLayout(channel_group)
        channel_explanation = QLabel(
            "Select channels from the AnnData panel and discovered ROI images. "
            "Blank selection means every channel discovered consistently by the worker."
        )
        channel_explanation.setWordWrap(True)
        self.feature_channel_list = QListWidget()
        self.feature_channel_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.feature_channel_list.setMaximumHeight(150)
        channel_actions = QHBoxLayout()
        self.refresh_feature_channels_button = QPushButton(
            "Refresh available channels"
        )
        self.select_all_feature_channels_button = QPushButton("Select all")
        self.clear_feature_channels_button = QPushButton("Clear selection")
        channel_actions.addWidget(self.refresh_feature_channels_button)
        channel_actions.addWidget(self.select_all_feature_channels_button)
        channel_actions.addWidget(self.clear_feature_channels_button)
        self.channels_edit = QLineEdit()
        self.channels_edit.setReadOnly(True)
        self.channels_edit.setPlaceholderText("Every discovered channel")
        channel_layout.addWidget(channel_explanation)
        channel_layout.addWidget(self.feature_channel_list)
        channel_layout.addLayout(channel_actions)
        channel_layout.addWidget(self.channels_edit)
        feature_builder_layout.addWidget(channel_group)

        feature_group = QGroupBox("3. Synthetic feature recipe")
        feature_form = QFormLayout(feature_group)
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(-1000, 1000)
        self.offset_overlap_check = QCheckBox(
            "Allow positive offsets to overlap other cells"
        )
        self.offset_overlap_check.setToolTip(
            "Independently expands each eligible cell through neighbouring masks. "
            "Pixels may contribute to more than one cell."
        )
        self.background_ring_spin = QSpinBox()
        self.background_ring_spin.setRange(1, 100)
        self.background_ring_spin.setValue(5)
        self.normalization_edit = QLineEdit()
        self.normalization_edit.setPlaceholderText(
            "Optional fixed Nimbus normalization JSON; display quantiles are separate"
        )
        self.distribution_check = QCheckBox("Distribution")
        self.distribution_check.setChecked(True)
        self.region_check = QCheckBox("Core/border/background/contrast")
        self.region_check.setChecked(True)
        self.gradient_check = QCheckBox("Channel gradients")
        self.shape_check = QCheckBox("Original-mask morphology")
        self.shape_check.setChecked(True)
        self.context_check = QCheckBox("Full-segmentation context")
        self.context_check.setChecked(True)
        self.roi_rank_check = QCheckBox("Cohort-relative ROI ranks")
        self.roi_rank_check.setChecked(True)
        feature_checks = QWidget()
        feature_checks_layout = QHBoxLayout(feature_checks)
        feature_checks_layout.setContentsMargins(0, 0, 0, 0)
        for widget in (
            self.distribution_check,
            self.region_check,
            self.gradient_check,
            self.shape_check,
            self.context_check,
            self.roi_rank_check,
        ):
            feature_checks_layout.addWidget(widget)
        self.feature_family_checks = {
            "distribution": self.distribution_check,
            "region": self.region_check,
            "gradient": self.gradient_check,
            "shape": self.shape_check,
            "context": self.context_check,
            "roi_rank": self.roi_rank_check,
        }
        self.feature_tree = QTreeWidget()
        self.feature_tree.setColumnCount(2)
        self.feature_tree.setHeaderLabels(
            ["Feature family / selected feature", "What it measures"]
        )
        self.feature_tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.feature_tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.feature_tree.setMaximumHeight(360)
        self.feature_tree_items: dict[str, dict[str, object]] = {}
        for family, catalog in FEATURE_FAMILY_CATALOG.items():
            family_item = self.QTreeWidgetItem(
                [
                    family.replace("_", " ").title(),
                    FEATURE_FAMILY_DESCRIPTIONS[family],
                ]
            )
            family_font = family_item.font(0)
            family_font.setBold(True)
            family_item.setFont(0, family_font)
            self.feature_tree.addTopLevelItem(family_item)
            self.feature_tree_items[family] = {}
            for key, description in catalog.items():
                child = self.QTreeWidgetItem([key, description])
                child.setData(0, self.Qt.UserRole, (family, key))
                child.setFlags(child.flags() | self.Qt.ItemIsUserCheckable)
                child.setCheckState(0, self.Qt.Checked)
                family_item.addChild(child)
                self.feature_tree_items[family][key] = child
        self.feature_tree.expandAll()
        self.feature_selection_summary = QLabel()
        self.feature_selection_summary.setWordWrap(True)
        worker_resolution = resolve_worker_count(None)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, worker_resolution.cpu_limit)
        self.workers_spin.setValue(worker_resolution.effective)
        self.workers_spin.setToolTip(worker_resolution.message)
        feature_form.addRow("Signed intensity-mask offset (px)", self.offset_spin)
        feature_form.addRow("Positive-offset collisions", self.offset_overlap_check)
        feature_form.addRow("Background ring (px)", self.background_ring_spin)
        feature_form.addRow("Nimbus normalization JSON", self.normalization_edit)
        feature_form.addRow("Enabled families", feature_checks)
        feature_form.addRow("Specific features", self.feature_tree)
        feature_form.addRow("Selection summary", self.feature_selection_summary)
        feature_form.addRow(
            f"Local workers (available: {worker_resolution.cpu_limit})",
            self.workers_spin,
        )
        feature_builder_layout.addWidget(feature_group)

        progress_group = QGroupBox("4. Build progress and process health")
        progress_layout = QVBoxLayout(progress_group)
        self.feature_progress_bar = QProgressBar()
        self.feature_progress_bar.setRange(0, 100)
        self.feature_progress_bar.setValue(0)
        self.feature_progress_bar.setFormat("Not started")
        self.feature_phase_label = QLabel("Phase: idle")
        self.feature_counts_label = QLabel(
            "ROIs: 0 completed, 0 resumed, 0 failed, 0 pending"
        )
        self.feature_current_roi_label = QLabel("Most recent ROI: none")
        self.feature_elapsed_label = QLabel("Elapsed: 00:00:00")
        self.feature_process_health_label = QLabel(
            "Python process: not running"
        )
        self.feature_process_health_label.setWordWrap(True)
        self.feature_progress_log = QTextEdit()
        self.feature_progress_log.setReadOnly(True)
        self.feature_progress_log.setMaximumHeight(180)
        progress_layout.addWidget(self.feature_progress_bar)
        progress_layout.addWidget(self.feature_phase_label)
        progress_layout.addWidget(self.feature_counts_label)
        progress_layout.addWidget(self.feature_current_roi_label)
        progress_layout.addWidget(self.feature_elapsed_label)
        progress_layout.addWidget(self.feature_process_health_label)
        progress_layout.addWidget(self.feature_progress_log)
        feature_builder_layout.addWidget(progress_group)

        feature_actions = QHBoxLayout()
        self.build_features_button = QPushButton("Build/resume features locally")
        self.cancel_features_button = QPushButton("Cancel build")
        self.cancel_features_button.setEnabled(False)
        self.hpc_button = QPushButton("HPC instructions")
        for widget in (
            self.build_features_button,
            self.cancel_features_button,
            self.hpc_button,
        ):
            feature_actions.addWidget(widget)
        feature_builder_layout.addLayout(feature_actions)
        add_tab(feature_builder, "🧬 Feature Building", "feature_building")

        # Feature Refinement
        refinement = QWidget()
        refinement_layout = QVBoxLayout(refinement)
        refinement_intro = QLabel(
            "Use confirmed labels from representative trial ROIs to identify a "
            "compact, stable feature set. Evaluation holds out complete ROIs; it "
            "never uses a random cell-level split."
        )
        refinement_intro.setWordWrap(True)
        refinement_layout.addWidget(refinement_intro)

        readiness_group = QGroupBox("1. Trial readiness")
        readiness_layout = QVBoxLayout(readiness_group)
        self.refinement_scope_label = QLabel(
            "Create or load a Feature Discovery Trial first."
        )
        self.refinement_scope_label.setWordWrap(True)
        self.refinement_class_table = QTableWidget(0, 4)
        self.refinement_class_table.setHorizontalHeaderLabels(
            ["Class", "Confirmed", "Represented ROIs", "Readiness"]
        )
        self.refinement_class_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self.refinement_class_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.refinement_class_table.setMaximumHeight(190)
        readiness_layout.addWidget(self.refinement_scope_label)
        readiness_layout.addWidget(self.refinement_class_table)
        refinement_layout.addWidget(readiness_group)

        refine_controls_group = QGroupBox("2. Grouped evaluation settings")
        refine_controls = QFormLayout(refine_controls_group)
        self.refine_candidate_spin = QSpinBox()
        self.refine_candidate_spin.setRange(10, 2000)
        self.refine_candidate_spin.setValue(150)
        self.refine_recommendation_spin = QSpinBox()
        self.refine_recommendation_spin.setRange(2, 500)
        self.refine_recommendation_spin.setValue(30)
        self.refine_repeats_spin = QSpinBox()
        self.refine_repeats_spin.setRange(1, 25)
        self.refine_repeats_spin.setValue(5)
        self.refine_missing_spin = QDoubleSpinBox()
        self.refine_missing_spin.setRange(0, 1)
        self.refine_missing_spin.setSingleStep(0.05)
        self.refine_missing_spin.setValue(0.30)
        self.refine_missing_spin.setDecimals(2)
        self.refine_correlation_spin = QDoubleSpinBox()
        self.refine_correlation_spin.setRange(0.50, 0.999)
        self.refine_correlation_spin.setSingleStep(0.01)
        self.refine_correlation_spin.setValue(0.95)
        self.refine_correlation_spin.setDecimals(3)
        refine_controls.addRow(
            "Maximum training-fold candidate features",
            self.refine_candidate_spin,
        )
        refine_controls.addRow(
            "Requested compact recommendation", self.refine_recommendation_spin
        )
        refine_controls.addRow(
            "Held-out permutation repeats", self.refine_repeats_spin
        )
        refine_controls.addRow(
            "Maximum allowed missing fraction", self.refine_missing_spin
        )
        refine_controls.addRow(
            "Redundancy correlation threshold", self.refine_correlation_spin
        )
        refinement_layout.addWidget(refine_controls_group)

        refinement_progress_group = QGroupBox("3. Analysis progress")
        refinement_progress_layout = QVBoxLayout(refinement_progress_group)
        self.refinement_progress_bar = QProgressBar()
        self.refinement_progress_bar.setRange(0, 100)
        self.refinement_progress_bar.setFormat("Not started")
        self.refinement_progress_label = QLabel("Refinement process: idle")
        self.refinement_progress_label.setWordWrap(True)
        self.refinement_log = QTextEdit()
        self.refinement_log.setReadOnly(True)
        self.refinement_log.setMaximumHeight(140)
        refinement_progress_layout.addWidget(self.refinement_progress_bar)
        refinement_progress_layout.addWidget(self.refinement_progress_label)
        refinement_progress_layout.addWidget(self.refinement_log)
        refinement_actions = QHBoxLayout()
        self.run_refinement_button = QPushButton(
            "Run leave-one-ROI-out feature refinement"
        )
        self.cancel_refinement_button = QPushButton("Cancel refinement")
        self.cancel_refinement_button.setEnabled(False)
        self.refresh_refinement_button = QPushButton("Reload saved results")
        refinement_actions.addWidget(self.run_refinement_button)
        refinement_actions.addWidget(self.cancel_refinement_button)
        refinement_actions.addWidget(self.refresh_refinement_button)
        refinement_progress_layout.addLayout(refinement_actions)
        refinement_layout.addWidget(refinement_progress_group)

        refinement_results_group = QGroupBox("4. Recommended feature set")
        refinement_results_layout = QVBoxLayout(refinement_results_group)
        self.refinement_metrics_label = QLabel("No refinement results yet.")
        self.refinement_metrics_label.setWordWrap(True)
        self.refinement_results_table = QTableWidget(0, 8)
        self.refinement_results_table.setHorizontalHeaderLabels(
            [
                "Rank",
                "Feature",
                "Source",
                "Family",
                "Importance",
                "Stability",
                "Missing",
                "Use",
            ]
        )
        self.refinement_results_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self.refinement_results_table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self.refinement_results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.refinement_results_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.refinement_results_table.setMinimumHeight(300)
        recommendation_actions = QHBoxLayout()
        self.select_recommended_button = QPushButton(
            "Restore recommended checks"
        )
        self.apply_model_features_button = QPushButton(
            "Use checked features for trial classifier"
        )
        self.promote_trial_button = QPushButton(
            "Promote checked features to full experiment revision"
        )
        recommendation_actions.addWidget(self.select_recommended_button)
        recommendation_actions.addWidget(self.apply_model_features_button)
        recommendation_actions.addWidget(self.promote_trial_button)
        refinement_results_layout.addWidget(self.refinement_metrics_label)
        refinement_results_layout.addWidget(self.refinement_results_table)
        refinement_results_layout.addLayout(recommendation_actions)
        refinement_layout.addWidget(refinement_results_group)
        add_tab(refinement, "🧪 Feature Refinement", "feature_refinement")

        # Explore
        explore = QWidget()
        explore_layout = QVBoxLayout(explore)
        roi_row = QHBoxLayout()
        self.roi_combo = QComboBox()
        self.previous_roi_button = QPushButton("Previous ROI")
        self.next_roi_button = QPushButton("Next ROI")
        self.reload_roi_button = QPushButton("Load ROI")
        roi_row.addWidget(self.previous_roi_button)
        roi_row.addWidget(QLabel("ROI"))
        roi_row.addWidget(self.roi_combo)
        roi_row.addWidget(self.next_roi_button)
        roi_row.addWidget(self.reload_roi_button)
        explore_layout.addLayout(roi_row)
        roi_options_row = QHBoxLayout()
        self.show_empty_rois = QCheckBox("Include ROIs with no eligible cells")
        self.context_check_display = QCheckBox("Show dimmed full-mask context")
        self.auto_reload_view_check = QCheckBox(
            "Reload the current Explore view when ROI changes"
        )
        self.auto_reload_view_check.setChecked(True)
        self.viewed_rois_label = QLabel("No Explore view is active.")
        self.viewed_rois_label.setWordWrap(True)
        roi_options_row.addWidget(self.show_empty_rois)
        roi_options_row.addWidget(self.context_check_display)
        roi_options_row.addWidget(self.auto_reload_view_check)
        explore_layout.addLayout(roi_options_row)
        explore_layout.addWidget(self.viewed_rois_label)

        layer_actions = QHBoxLayout()
        self.hide_all_layers_button = QPushButton("Hide all layers")
        self.show_all_layers_button = QPushButton("Show all layers")
        self.delete_all_layers_button = QPushButton("Delete all layers")
        layer_actions.addWidget(self.hide_all_layers_button)
        layer_actions.addWidget(self.show_all_layers_button)
        layer_actions.addWidget(self.delete_all_layers_button)
        explore_layout.addLayout(layer_actions)

        reload_recipe_group = QGroupBox("Layers re-added when the ROI changes")
        reload_recipe_layout = QVBoxLayout(reload_recipe_group)
        self.reload_recipe_help = QLabel(
            "This is the exact ROI reload recipe. Classifier layers are "
            "regenerated from labels and scores, while their visible/hidden "
            "state, opacity, contour style, and image contrast limits are "
            "replayed from this list."
        )
        self.reload_recipe_help.setWordWrap(True)
        self.reload_recipe_list = QListWidget()
        self.reload_recipe_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.reload_recipe_list.setMaximumHeight(240)
        reload_recipe_actions = QHBoxLayout()
        self.delete_recipe_items_button = QPushButton(
            "Delete/reset selected recipe items"
        )
        self.update_recipe_from_layers_button = QPushButton(
            "Update from current layers"
        )
        reload_recipe_actions.addWidget(self.delete_recipe_items_button)
        reload_recipe_actions.addWidget(self.update_recipe_from_layers_button)
        reload_recipe_layout.addWidget(self.reload_recipe_help)
        reload_recipe_layout.addWidget(self.reload_recipe_list)
        reload_recipe_layout.addLayout(reload_recipe_actions)
        explore_layout.addWidget(reload_recipe_group)

        overlay_group = QGroupBox("AnnData overlays and population-to-cohort transfer")
        overlay_form = QFormLayout(overlay_group)
        self.overlay_obs_combo = QComboBox()
        self.overlay_button = QPushButton("Render observation overlay")
        self.population_obs_combo = QComboBox()
        self.population_value_combo = QComboBox()
        self.population_layer_list = QListWidget()
        self.population_layer_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.population_layer_list.setMaximumHeight(110)
        self.load_population_layers_button = QPushButton(
            "Add selected populations as separate layers"
        )
        self.rank_rois_button = QPushButton("Rank ROIs by selected population")
        self.use_population_button = QPushButton(
            "Use this population as classification cohort"
        )
        self.save_population_view_button = QPushButton(
            "Save current view for selected population"
        )
        self.load_population_view_button = QPushButton(
            "Load saved view for selected population"
        )
        self.marker_overlay_list = QListWidget()
        self.marker_overlay_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.marker_overlay_list.setMaximumHeight(110)
        self.load_marker_overlays_button = QPushButton(
            "Add selected adata.X markers as cell overlays"
        )
        overlay_form.addRow("Categorical or numeric observation", self.overlay_obs_combo)
        overlay_form.addRow("", self.overlay_button)
        overlay_form.addRow("Population observation", self.population_obs_combo)
        overlay_form.addRow("Population", self.population_value_combo)
        overlay_form.addRow("Separate population layers", self.population_layer_list)
        overlay_form.addRow("", self.load_population_layers_button)
        overlay_form.addRow("Cell-level marker overlays", self.marker_overlay_list)
        overlay_form.addRow("", self.load_marker_overlays_button)
        population_actions = QWidget()
        population_actions_layout = QHBoxLayout(population_actions)
        population_actions_layout.setContentsMargins(0, 0, 0, 0)
        population_actions_layout.addWidget(self.rank_rois_button)
        population_actions_layout.addWidget(self.use_population_button)
        overlay_form.addRow("", population_actions)
        population_view_actions = QWidget()
        population_view_actions_layout = QHBoxLayout(population_view_actions)
        population_view_actions_layout.setContentsMargins(0, 0, 0, 0)
        population_view_actions_layout.addWidget(self.save_population_view_button)
        population_view_actions_layout.addWidget(self.load_population_view_button)
        overlay_form.addRow("Population verification view", population_view_actions)
        explore_layout.addWidget(overlay_group)
        image_group = QGroupBox("Raw, extra, greyscale, and multicolour images")
        image_layout = QVBoxLayout(image_group)
        self.image_coverage_label = QLabel("No ROI images discovered yet.")
        self.image_coverage_label.setWordWrap(True)
        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.channel_list.setMaximumHeight(120)
        image_actions = QHBoxLayout()
        self.load_channels_button = QPushButton("Load selected greyscale")
        self.load_six_colour_button = QPushButton(
            "Load selected as R/G/B/C/Y/M"
        )
        self.load_rgb_button = QPushButton("Load first three selected as RGB")
        image_actions.addWidget(self.load_channels_button)
        image_actions.addWidget(self.load_six_colour_button)
        image_actions.addWidget(self.load_rgb_button)
        image_layout.addWidget(self.image_coverage_label)
        image_layout.addWidget(self.channel_list)
        image_layout.addLayout(image_actions)
        explore_layout.addWidget(image_group)
        add_tab(explore, "🔬 Explore", "explore")

        # Classify
        classify = QWidget()
        classify_layout = QVBoxLayout(classify)
        selection_group = QGroupBox("Selected cell annotation")
        selection_form = QFormLayout(selection_group)
        self.selected_cell_label = QLabel("No cohort cell selected")
        self.cell_picking_help = QLabel(
            "Click any eligible cell in the viewer while this Classify tab is "
            "active. The selected click action is applied using the current "
            "class. The classification_cohort layer may remain hidden and does "
            "not need to be selected."
        )
        self.cell_picking_help.setWordWrap(True)
        self.class_combo = QComboBox()
        click_behavior_widget = QWidget()
        click_behavior_layout = QHBoxLayout(click_behavior_widget)
        click_behavior_layout.setContentsMargins(0, 0, 0, 0)
        self.click_behavior_group = QButtonGroup(click_behavior_widget)
        self.click_behavior_radios = {}
        for behavior, text in (
            ("select", "Select only"),
            ("proposed", "Set proposed on click"),
            ("confirmed", "Set confirmed on click"),
        ):
            radio = QRadioButton(text)
            radio.setProperty("napari_sbt_click_behavior", behavior)
            self.click_behavior_group.addButton(radio)
            self.click_behavior_radios[behavior] = radio
            click_behavior_layout.addWidget(radio)
        self.click_behavior_radios["proposed"].setChecked(True)
        click_behavior_layout.addStretch(1)
        self.classifier_display_button = QPushButton(
            "Classifier display & cell-picking options..."
        )
        self.class_tally_table = QTableWidget(0, 4)
        self.class_tally_table.setHorizontalHeaderLabels(
            ["Class", "Proposed", "Confirmed", "HGB target"]
        )
        self.class_tally_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.class_tally_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.class_tally_table.verticalHeader().setVisible(False)
        self.class_tally_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        for column in (1, 2, 3):
            self.class_tally_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeToContents
            )
        self.class_tally_table.setMaximumHeight(210)
        annotation_buttons = QWidget()
        annotation_layout = QHBoxLayout(annotation_buttons)
        annotation_layout.setContentsMargins(0, 0, 0, 0)
        self.propose_button = QPushButton("Set proposed")
        self.confirm_button = QPushButton("Set confirmed")
        self.confirm_proposed_button = QPushButton("Confirm all proposals")
        self.mark_reviewed_button = QPushButton("Mark current ROI reviewed")
        self.seed_obs_button = QPushButton(
            "Seed matching classes as proposals from overlay observation"
        )
        annotation_layout.addWidget(self.propose_button)
        annotation_layout.addWidget(self.confirm_button)
        annotation_layout.addWidget(self.confirm_proposed_button)
        annotation_layout.addWidget(self.mark_reviewed_button)
        selection_form.addRow("Cell", self.selected_cell_label)
        selection_form.addRow("Picking", self.cell_picking_help)
        selection_form.addRow("Class", self.class_combo)
        selection_form.addRow("Click action", click_behavior_widget)
        selection_form.addRow("", annotation_buttons)
        selection_form.addRow("Label tally", self.class_tally_table)
        selection_form.addRow("", self.classifier_display_button)
        selection_form.addRow("", self.seed_obs_button)
        classify_layout.addWidget(selection_group)
        model_group = QGroupBox("Model and active-learning queues")
        model_form = QFormLayout(model_group)
        self.model_combo = QComboBox()
        for name, key in (
            ("HistGradientBoosting (default)", "hist_gradient_boosting"),
            ("Random Forest", "random_forest"),
            ("XGBoost", "xgboost"),
            ("LightGBM", "lightgbm"),
        ):
            self.model_combo.addItem(name, key)
        self.model_storage_label = QLabel(
            "No active experiment. Models are stored inside the experiment folder."
        )
        self.model_storage_label.setWordWrap(True)
        self.model_storage_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        model_actions = QWidget()
        model_actions_layout = QHBoxLayout(model_actions)
        model_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.train_button = QPushButton("Train")
        self.score_button = QPushButton("Score cohort")
        self.refresh_queue_button = QPushButton("Apply queue filters / refresh")
        model_actions_layout.addWidget(self.train_button)
        model_actions_layout.addWidget(self.score_button)
        model_actions_layout.addWidget(self.refresh_queue_button)
        self.queue_list = QListWidget()
        self.queue_list.setMaximumHeight(145)
        self.queue_result_label = QLabel("Score the cohort to populate this queue.")
        self.queue_result_label.setWordWrap(True)
        self.queue_roi_combo = QComboBox()
        self.queue_class_combo = QComboBox()
        self.queue_review_combo = QComboBox()
        self.queue_review_combo.addItems(
            ["Unlabelled", "Proposed", "Confirmed", "All"]
        )
        self.queue_confidence_spin = QDoubleSpinBox()
        self.queue_confidence_spin.setRange(0, 1)
        self.queue_confidence_spin.setValue(0)
        self.queue_confidence_spin.setSingleStep(0.05)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0, 1)
        self.confidence_spin.setValue(0.9)
        self.confidence_spin.setSingleStep(0.05)
        self.bulk_propose_button = QPushButton(
            "Bulk-propose high-confidence selected class"
        )
        self.probability_class_combo = QComboBox()
        self.show_probability_button = QPushButton(
            "Show selected-class probability"
        )
        model_form.addRow("Model", self.model_combo)
        model_form.addRow("Model storage", self.model_storage_label)
        model_form.addRow("", model_actions)
        queue_filters = QWidget()
        queue_filters_layout = QHBoxLayout(queue_filters)
        queue_filters_layout.setContentsMargins(0, 0, 0, 0)
        queue_filters_layout.addWidget(self.queue_roi_combo)
        queue_filters_layout.addWidget(self.queue_class_combo)
        queue_filters_layout.addWidget(self.queue_review_combo)
        queue_filters_layout.addWidget(self.queue_confidence_spin)
        model_form.addRow("Queue filters (ROI/class/review/conf.)", queue_filters)
        model_form.addRow("Queue result", self.queue_result_label)
        model_form.addRow("Ambiguous unlabelled cells", self.queue_list)
        model_form.addRow("High-confidence threshold", self.confidence_spin)
        model_form.addRow("", self.bulk_propose_button)
        model_form.addRow("Probability class", self.probability_class_combo)
        model_form.addRow("", self.show_probability_button)
        classify_layout.addWidget(model_group)
        add_tab(classify, "🏷 Classify", "classify")
        self.classify_tab_index = self.tabs.count() - 1

        # Regions & Export
        regions = QWidget()
        regions_layout = QVBoxLayout(regions)
        region_group = QGroupBox("Manual tissue regions")
        region_form = QFormLayout(region_group)
        self.region_name_edit = QLineEdit("region")
        self.create_regions_button = QPushButton("Create/select regions layer")
        self.sync_regions_button = QPushButton("Synchronize regions to cell table")
        region_form.addRow("Region name", self.region_name_edit)
        region_form.addRow("", self.create_regions_button)
        region_form.addRow("", self.sync_regions_button)
        regions_layout.addWidget(region_group)
        export_group = QGroupBox("Cohort results")
        export_form = QFormLayout(export_group)
        self.annotated_path_edit = QLineEdit(
            str(self.project_root / "napari_sbt_annotated.h5ad")
        )
        self.export_assignments_button = QPushButton("Export assignment table")
        self.export_adata_button = QPushButton("Export annotated AnnData copy")
        self.export_cohort_masks_button = QPushButton("Export cohort masks")
        self.export_clean_masks_button = QPushButton("Export cleaned masks")
        export_form.addRow("Annotated AnnData destination", self.annotated_path_edit)
        export_form.addRow("", self.export_assignments_button)
        export_form.addRow("", self.export_adata_button)
        export_form.addRow("", self.export_cohort_masks_button)
        export_form.addRow("", self.export_clean_masks_button)
        regions_layout.addWidget(export_group)
        add_tab(regions, "🗺 Regions & Export", "regions_export")

        # Layers & Status
        layers = QWidget()
        layers_layout = QVBoxLayout(layers)
        utility_group = QGroupBox("Selected-layer utilities")
        utility_layout = QGridLayout(utility_group)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(
            ["gray", "green", "magenta", "cyan", "yellow", "red", "blue", "viridis"]
        )
        self.recolour_button = QPushButton("Apply colormap")
        self.flip_horizontal_button = QPushButton("Flip horizontal")
        self.flip_vertical_button = QPushButton("Flip vertical")
        self.transfer_colormap_button = QPushButton(
            "Transfer selected colormap to all images"
        )
        self.expand_spin = QSpinBox()
        self.expand_spin.setRange(1, 100)
        self.expand_spin.setValue(2)
        self.expand_button = QPushButton("Expand selected labels (derived layer)")
        self.resize_spin = QDoubleSpinBox()
        self.resize_spin.setRange(0.05, 10.0)
        self.resize_spin.setValue(1.0)
        self.resize_spin.setSingleStep(0.1)
        self.resize_button = QPushButton("Resize selected (derived layer)")
        self.mask_layer_button = QPushButton("Mask selected image to cohort")
        utility_layout.addWidget(self.colormap_combo, 0, 0)
        utility_layout.addWidget(self.recolour_button, 0, 1)
        utility_layout.addWidget(self.flip_horizontal_button, 1, 0)
        utility_layout.addWidget(self.flip_vertical_button, 1, 1)
        utility_layout.addWidget(self.transfer_colormap_button, 2, 0, 1, 2)
        utility_layout.addWidget(self.expand_spin, 3, 0)
        utility_layout.addWidget(self.expand_button, 3, 1)
        utility_layout.addWidget(self.resize_spin, 4, 0)
        utility_layout.addWidget(self.resize_button, 4, 1)
        utility_layout.addWidget(self.mask_layer_button, 5, 0, 1, 2)
        layers_layout.addWidget(utility_group)
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.refresh_status_button = QPushButton("Refresh experiment freshness status")
        layers_layout.addWidget(self.refresh_status_button)
        layers_layout.addWidget(self.status_text)
        add_tab(layers, "🎨 Layers & Status", "layers_status")
        self.tabs.setStyleSheet(
            """
            QTabBar::tab {
                color: #202124;
                border: 1px solid #9aa0a6;
                border-bottom: none;
                padding: 6px 10px;
                margin-right: 1px;
            }
            QTabBar::tab:nth-child(1) { background: #dbeafe; }
            QTabBar::tab:nth-child(2) { background: #dcfce7; }
            QTabBar::tab:nth-child(3) { background: #fef3c7; }
            QTabBar::tab:nth-child(4) { background: #fce7f3; }
            QTabBar::tab:nth-child(5) { background: #ede9fe; }
            QTabBar::tab:nth-child(6) { background: #e0f2fe; }
            QTabBar::tab:nth-child(7) { background: #f1f5f9; }
            QTabBar::tab:selected {
                font-weight: bold;
                border: 2px solid #5f6368;
                border-bottom: none;
            }
            """
        )
        tab_font = QFont(self.tabs.tabBar().font())
        tab_font.setBold(True)
        tab_font.setPointSize(max(tab_font.pointSize() + 2, 11))
        self.tabs.tabBar().setFont(tab_font)
        tab_text_colours = (
            "#1d4ed8",
            "#7c3aed",
            "#047857",
            "#b45309",
            "#be185d",
            "#0369a1",
            "#475569",
        )
        for index, colour in enumerate(tab_text_colours):
            self.tabs.tabBar().setTabTextColor(index, self.QColor(colour))

        self._set_class_rows(segmentation_qc_classes())
        self._connect_signals()
        self._bind_viewer_cell_picking()
        for family, checkbox in self.feature_family_checks.items():
            self._feature_family_toggled(family, checkbox.isChecked())
        self._update_feature_selection_summary()
        self._update_feature_channel_summary()
        self._refresh_reload_recipe_list()
        self._set_classification_enabled(False)
        if experiment:
            self.load_existing_experiment(Path(experiment))
        elif anndata_path:
            self.load_anndata_selectors()

    def _connect_signals(self) -> None:
        self.load_adata_button.clicked.connect(self._guard(self.load_anndata_selectors))
        self.obs_combo.currentTextChanged.connect(self._guard(self.refresh_scope_values))
        self.scope_combo.currentIndexChanged.connect(self._update_scope_widget_state)
        self.preview_button.clicked.connect(self._guard(self.preview_cohort))
        self.experiment_mode_combo.currentIndexChanged.connect(
            self._update_experiment_mode_state
        )
        self.trial_roi_strategy_combo.currentIndexChanged.connect(
            self._update_experiment_mode_state
        )
        self.trial_roi_count_spin.valueChanged.connect(
            self._guard(self._trial_roi_count_changed)
        )
        self.trial_roi_list.itemSelectionChanged.connect(
            self._update_trial_roi_summary
        )
        self.suggest_trial_rois_button.clicked.connect(
            self._guard(self.suggest_trial_rois)
        )
        self.qc_template_button.clicked.connect(
            lambda: self._set_class_rows(segmentation_qc_classes())
        )
        self.add_class_button.clicked.connect(self.add_class_row)
        self.remove_class_button.clicked.connect(self.remove_class_row)
        self.apply_classes_button.clicked.connect(self._guard(self.apply_class_edits))
        self.create_button.clicked.connect(self._guard(self.create_experiment))
        self.load_experiment_button.clicked.connect(
            self._guard(self.choose_and_load_experiment)
        )
        self.validate_sources_button.clicked.connect(
            self._guard(self.start_source_validation)
        )
        self.refresh_feature_channels_button.clicked.connect(
            self._guard(self.refresh_feature_channel_choices)
        )
        self.select_all_feature_channels_button.clicked.connect(
            lambda: self.feature_channel_list.selectAll()
        )
        self.clear_feature_channels_button.clicked.connect(
            lambda: self.feature_channel_list.clearSelection()
        )
        self.feature_channel_list.itemSelectionChanged.connect(
            self._update_feature_channel_summary
        )
        self.feature_tree.itemChanged.connect(self._feature_tree_item_changed)
        for family, checkbox in self.feature_family_checks.items():
            checkbox.toggled.connect(
                lambda checked, family=family: self._feature_family_toggled(
                    family,
                    checked,
                )
            )
        self.build_features_button.clicked.connect(self._guard(self.start_feature_build))
        self.cancel_features_button.clicked.connect(self.cancel_feature_build)
        self.hpc_button.clicked.connect(
            lambda: self.set_status(
                "Managed build: set napari_sbt.active_experiment in config.yaml, "
                "then run `sbt run cellfeat` (8 CPUs, 64 GB, 24 hours)."
            )
        )
        self.run_refinement_button.clicked.connect(
            self._guard(self.start_feature_refinement)
        )
        self.cancel_refinement_button.clicked.connect(
            self.cancel_feature_refinement
        )
        self.refresh_refinement_button.clicked.connect(
            self._guard(self.load_refinement_results)
        )
        self.select_recommended_button.clicked.connect(
            self._guard(self.restore_recommended_feature_checks)
        )
        self.apply_model_features_button.clicked.connect(
            self._guard(self.apply_checked_model_features)
        )
        self.promote_trial_button.clicked.connect(
            self._guard(self.promote_feature_trial)
        )
        self.roi_combo.currentTextChanged.connect(
            self._guard(self.load_roi, pass_signal_args=True)
        )
        self.reload_roi_button.clicked.connect(self._guard(self.load_roi))
        self.previous_roi_button.clicked.connect(
            lambda: self.move_roi(-1)
        )
        self.next_roi_button.clicked.connect(
            lambda: self.move_roi(1)
        )
        self.show_empty_rois.toggled.connect(self._guard(self.refresh_rois))
        self.context_check_display.toggled.connect(self.toggle_context)
        self.auto_reload_view_check.toggled.connect(
            self._guard(self.auto_reload_explore_view)
        )
        self.hide_all_layers_button.clicked.connect(self._guard(self.hide_all_layers))
        self.show_all_layers_button.clicked.connect(self._guard(self.show_all_layers))
        self.delete_all_layers_button.clicked.connect(
            self._guard(self.delete_all_layers)
        )
        self.delete_recipe_items_button.clicked.connect(
            self._guard(self.delete_selected_recipe_items)
        )
        self.update_recipe_from_layers_button.clicked.connect(
            self._guard(self.update_recipe_from_current_layers)
        )
        self.overlay_obs_combo.currentTextChanged.connect(
            lambda: self.set_status("Overlay selection changed.")
        )
        self.overlay_button.clicked.connect(self._guard(self.render_obs_overlay))
        self.population_obs_combo.currentTextChanged.connect(
            self._guard(self.refresh_population_values)
        )
        self.population_value_combo.currentTextChanged.connect(
            self._guard(self.restore_population_view)
        )
        self.load_population_layers_button.clicked.connect(
            self._guard(self.load_selected_population_layers)
        )
        self.load_marker_overlays_button.clicked.connect(
            self._guard(self.load_selected_marker_overlays)
        )
        self.rank_rois_button.clicked.connect(self._guard(self.rank_rois_by_population))
        self.use_population_button.clicked.connect(
            self._guard(self.use_population_as_cohort)
        )
        self.save_population_view_button.clicked.connect(
            self._guard(self.save_population_view)
        )
        self.load_population_view_button.clicked.connect(
            self._guard(self.restore_population_view)
        )
        self.load_channels_button.clicked.connect(self._guard(self.load_selected_channels))
        self.load_six_colour_button.clicked.connect(
            self._guard(self.load_six_colour_channels)
        )
        self.load_rgb_button.clicked.connect(self._guard(self.load_rgb))
        self.propose_button.clicked.connect(lambda: self.annotate_selected("proposed"))
        self.confirm_button.clicked.connect(lambda: self.annotate_selected("confirmed"))
        self.classifier_display_button.clicked.connect(
            self._guard(self.show_classifier_display_options)
        )
        self.confirm_proposed_button.clicked.connect(self._guard(self.confirm_all_proposed))
        self.mark_reviewed_button.clicked.connect(self._guard(self.mark_roi_reviewed))
        self.seed_obs_button.clicked.connect(self._guard(self.seed_proposals_from_obs))
        self.train_button.clicked.connect(self._guard(self.train_model))
        self.score_button.clicked.connect(self._guard(self.score_model))
        self.refresh_queue_button.clicked.connect(self._guard(self.refresh_uncertainty_queue))
        for widget_signal in (
            self.queue_roi_combo.currentIndexChanged,
            self.queue_class_combo.currentIndexChanged,
            self.queue_review_combo.currentIndexChanged,
            self.queue_confidence_spin.valueChanged,
        ):
            widget_signal.connect(self._guard(self._refresh_queue_if_scored))
        self.queue_list.itemDoubleClicked.connect(
            self._guard(self.navigate_queue_item, pass_signal_args=True)
        )
        self.bulk_propose_button.clicked.connect(self._guard(self.bulk_propose))
        self.show_probability_button.clicked.connect(
            self._guard(self.show_selected_probability)
        )
        self.create_regions_button.clicked.connect(self._guard(self.create_regions_layer))
        self.sync_regions_button.clicked.connect(self._guard(self.sync_regions))
        self.export_assignments_button.clicked.connect(
            self._guard(self.export_assignments)
        )
        self.export_adata_button.clicked.connect(self._guard(self.export_adata))
        self.export_cohort_masks_button.clicked.connect(
            self._guard(self.export_cohort_masks)
        )
        self.export_clean_masks_button.clicked.connect(
            self._guard(self.export_cleaned_masks)
        )
        self.recolour_button.clicked.connect(self._guard(self.apply_colormap))
        self.flip_horizontal_button.clicked.connect(
            lambda: self.flip_selected_layer(axis=1)
        )
        self.flip_vertical_button.clicked.connect(lambda: self.flip_selected_layer(axis=0))
        self.transfer_colormap_button.clicked.connect(
            self._guard(self.transfer_colormap)
        )
        self.expand_button.clicked.connect(self._guard(self.expand_selected_labels))
        self.resize_button.clicked.connect(self._guard(self.resize_selected_layer))
        self.mask_layer_button.clicked.connect(self._guard(self.mask_selected_image))
        self.refresh_status_button.clicked.connect(self._guard(self.refresh_status))
        self._update_scope_widget_state()
        self._update_experiment_mode_state()

    def _guard(self, callback, *, pass_signal_args: bool = False):
        def wrapped(*args, **kwargs):
            try:
                if pass_signal_args:
                    return callback(*args, **kwargs)
                return callback()
            except Exception as exc:  # noqa: BLE001 - Qt callback error boundary
                self.set_status(f"ERROR — {type(exc).__name__}: {exc}")
                self.QMessageBox.critical(
                    self.root, "napari_sbt", f"{type(exc).__name__}: {exc}"
                )
                return None

        return wrapped

    def set_status(self, message: str) -> None:
        self.status_text.append(str(message))
        self.scope_label.setToolTip(str(message))

    def show_tab_help(self, topic: str, title: str) -> None:
        """Show documentation-backed help for one workflow tab."""

        help_path = Path(__file__).with_name("help") / f"{topic}.md"
        if not help_path.is_file():
            raise FileNotFoundError(f"Tab help is missing: {help_path}")
        dialog = self.QDialog(self.root)
        dialog.setWindowTitle(f"napari_sbt help — {title}")
        dialog.resize(820, 680)
        from qtpy.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(dialog)
        browser = self.QTextBrowser(dialog)
        browser.setOpenExternalLinks(True)
        browser.setMarkdown(help_path.read_text(encoding="utf-8"))
        buttons = self.QDialogButtonBox(self.QDialogButtonBox.Close, parent=dialog)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(browser)
        layout.addWidget(buttons)
        dialog.exec()

    def _set_classification_enabled(self, enabled: bool) -> None:
        for widget in (
            self.class_combo,
            *self.click_behavior_radios.values(),
            self.propose_button,
            self.confirm_button,
            self.confirm_proposed_button,
            self.train_button,
            self.score_button,
            self.refresh_queue_button,
            self.bulk_propose_button,
            self.export_assignments_button,
            self.export_adata_button,
            self.export_cohort_masks_button,
            self.export_clean_masks_button,
        ):
            widget.setEnabled(enabled)

    def _update_scope_widget_state(self) -> None:
        selected = self.scope_combo.currentData() == "obs_values"
        self.obs_combo.setEnabled(selected)
        self.value_list.setEnabled(selected)

    def _update_experiment_mode_state(self, *_args) -> None:
        trial_mode = (
            self.experiment_mode_combo.currentData() == "feature_discovery_trial"
        )
        manual = self.trial_roi_strategy_combo.currentData() == "manual"
        self.trial_roi_count_spin.setEnabled(trial_mode)
        self.trial_roi_strategy_combo.setEnabled(trial_mode)
        self.trial_roi_list.setEnabled(trial_mode and manual)
        self.suggest_trial_rois_button.setEnabled(trial_mode)
        if trial_mode and not manual and self.trial_roi_list.count():
            self.suggest_trial_rois()
        self._update_trial_roi_summary()

    def _trial_roi_count_changed(self) -> None:
        if self.trial_roi_strategy_combo.currentData() == "largest":
            self.suggest_trial_rois()
        else:
            self._update_trial_roi_summary()

    def _populate_trial_roi_list(
        self,
        per_roi_counts: pd.DataFrame,
        *,
        selected_rois: Iterable[str] = (),
    ) -> None:
        from qtpy.QtWidgets import QListWidgetItem

        selected = {str(roi) for roi in selected_rois}
        available_roi_count = len(per_roi_counts)
        self.trial_roi_count_spin.blockSignals(True)
        self.trial_roi_count_spin.setMaximum(max(2, available_roi_count))
        if available_roi_count >= 2 and (
            self.trial_roi_count_spin.value() > available_roi_count
        ):
            self.trial_roi_count_spin.setValue(available_roi_count)
        self.trial_roi_count_spin.blockSignals(False)
        self.trial_roi_list.blockSignals(True)
        self.trial_roi_list.clear()
        for row in per_roi_counts.itertuples(index=False):
            item = QListWidgetItem(
                f"{row.ROI} — {int(row.eligible_cells):,} eligible cells"
            )
            item.setData(self.Qt.UserRole, str(row.ROI))
            self.trial_roi_list.addItem(item)
            item.setSelected(str(row.ROI) in selected)
        self.trial_roi_list.blockSignals(False)
        self._update_trial_roi_summary()

    def selected_trial_rois(self) -> list[str]:
        return [
            str(item.data(self.Qt.UserRole) or item.text())
            for item in self.trial_roi_list.selectedItems()
        ]

    def suggest_trial_rois(self) -> None:
        if self.trial_roi_list.count() == 0:
            self._update_trial_roi_summary()
            return
        requested = self.trial_roi_count_spin.value()
        self.trial_roi_list.blockSignals(True)
        for index in range(self.trial_roi_list.count()):
            self.trial_roi_list.item(index).setSelected(index < requested)
        self.trial_roi_list.blockSignals(False)
        self._update_trial_roi_summary()

    def _update_trial_roi_summary(self) -> None:
        if self.experiment_mode_combo.currentData() != "feature_discovery_trial":
            self.trial_roi_summary.setText(
                "Full mode: features, training and scoring use every eligible ROI."
            )
            return
        selected = self.selected_trial_rois()
        cells = 0
        if self.preview is not None and selected:
            cells = int(
                self.preview.eligible_cells["ROI"].astype(str).isin(selected).sum()
            )
        requested = self.trial_roi_count_spin.value()
        readiness = "ready" if len(selected) == requested else "selection incomplete"
        self.trial_roi_summary.setText(
            f"{len(selected)}/{requested} trial ROIs selected; {cells:,} eligible "
            f"trial cells ({readiness}). The full cohort remains frozen as the "
            "eventual classification target."
        )

    def load_anndata_selectors(self) -> None:
        import anndata as ad

        path = Path(self.anndata_edit.text()).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"AnnData not found: {path}")
        self.adata = ad.read_h5ad(path)
        columns = [str(column) for column in self.adata.obs.columns]
        for combo in (
            self.obs_combo,
            self.overlay_obs_combo,
            self.population_obs_combo,
        ):
            current = combo.currentText()
            combo.clear()
            combo.addItems(columns)
            if current in columns:
                combo.setCurrentText(current)
        preferred = [
            column
            for column in columns
            if isinstance(self.adata.obs[column].dtype, pd.CategoricalDtype)
        ]
        if preferred:
            self.obs_combo.setCurrentText(preferred[0])
            self.population_obs_combo.setCurrentText(preferred[0])
        selected_markers = {
            item.text() for item in self.marker_overlay_list.selectedItems()
        }
        self.marker_overlay_list.clear()
        self.marker_overlay_list.addItems(
            [str(marker) for marker in self.adata.var_names]
        )
        for index in range(self.marker_overlay_list.count()):
            item = self.marker_overlay_list.item(index)
            item.setSelected(item.text() in selected_markers)
        self.refresh_scope_values()
        self.refresh_population_values()
        self.refresh_feature_channel_choices()
        self.set_status(f"Loaded AnnData selectors for {self.adata.n_obs:,} cells.")

    def refresh_scope_values(self) -> None:
        self.value_list.clear()
        if self.adata is None or self.obs_combo.currentText() not in self.adata.obs:
            return
        values = sorted(
            self.adata.obs[self.obs_combo.currentText()].dropna().astype(str).unique()
        )
        for value in values:
            self.value_list.addItem(value)

    def selected_scope_values(self) -> list[str]:
        return [item.text() for item in self.value_list.selectedItems()]

    def preview_cohort(self) -> CohortPreview:
        mode = self.scope_combo.currentData()
        values = self.selected_scope_values() if mode == "obs_values" else []
        if self.adata is None and Path(self.anndata_edit.text()).is_file():
            self.load_anndata_selectors()
        if self.adata is None:
            if mode != "all_cells":
                raise ValueError("Observation-defined cohorts require AnnData.")
            table_sources = [
                source for source in self.feature_sources() if source.kind == "table"
            ]
            if not table_sources:
                raise ValueError(
                    "Standalone all-cells compatibility requires an identity-bearing "
                    "CSV/Parquet feature table."
                )
            self.preview = resolve_table_cohort(
                read_dataframe(table_sources[0].path),
                roi_column=self.roi_obs_edit.text().strip(),
                object_id_column=self.object_obs_edit.text().strip(),
            )
        else:
            self.preview = resolve_cohort(
                self.adata,
                roi_obs=self.roi_obs_edit.text().strip(),
                object_id_obs=self.object_obs_edit.text().strip(),
                mode=mode,
                obs_column=self.obs_combo.currentText() if mode == "obs_values" else None,
                obs_values=values,
            )
        masks = discover_mask_files(self.masks_edit.text())
        missing_masks: list[str] = []
        missing_ids = 0
        unmatched_ids = 0
        for roi, group in self.preview.eligible_cells.groupby("ROI", observed=True):
            path = masks.get(str(roi))
            if path is None:
                missing_masks.append(str(roi))
                continue
            missing, unmatched = validate_mask_coverage(
                load_mask(path),
                group["ObjectNumber"],
                roi=str(roi),
            )
            missing_ids += len(missing)
            unmatched_ids += len(unmatched)
        text = (
            f"{self.preview.eligible_cell_count:,} eligible cells "
            f"({self.preview.eligible_fraction:.1%}) / "
            f"{self.preview.total_cell_count:,} total\n"
            f"{self.preview.represented_roi_count:,} represented ROIs\n"
            f"Missing masks: {len(missing_masks)}; missing eligible object IDs: "
            f"{missing_ids}; other full-mask labels: {unmatched_ids}\n\n"
            + self.preview.per_roi_counts.to_string(index=False)
        )
        self.preview_text.setPlainText(text)
        previous_trial_rois = self.selected_trial_rois()
        self._populate_trial_roi_list(
            self.preview.per_roi_counts,
            selected_rois=previous_trial_rois,
        )
        if (
            self.experiment_mode_combo.currentData() == "feature_discovery_trial"
            and (
                self.trial_roi_strategy_combo.currentData() == "largest"
                or not previous_trial_rois
            )
        ):
            self.suggest_trial_rois()
        first_roi = str(self.preview.eligible_cells.iloc[0]["ROI"])
        if first_roi in masks:
            restricted = cohort_mask(
                load_mask(masks[first_roi]),
                self.preview.eligible_cells.loc[
                    self.preview.eligible_cells["ROI"].astype(str).eq(first_roi),
                    "ObjectNumber",
                ],
            )
            self._replace_layer(
                "cohort_preview",
                restricted,
                "labels",
                visible=True,
            )
        self.set_status("Cohort preview validated. Confirm to freeze these identities.")
        return self.preview

    def _set_class_rows(self, classes: Iterable[ClassificationClass]) -> None:
        self.class_table.setRowCount(0)
        for definition in classes:
            row = self.class_table.rowCount()
            self.class_table.insertRow(row)
            values = [
                definition.class_id,
                definition.name,
                definition.color,
                definition.shortcut,
                definition.mask_disposition,
            ]
            for column, value in enumerate(values):
                self.class_table.setItem(row, column, self.QTableWidgetItem(value))

    def add_class_row(self) -> None:
        if self.class_table.rowCount() >= 8:
            self.set_status("Experiments are limited to eight classes.")
            return
        row = self.class_table.rowCount()
        self.class_table.insertRow(row)
        values = [
            f"class_{row + 1}",
            f"Class {row + 1}",
            "#808080",
            str(row + 1),
            "keep",
        ]
        for column, value in enumerate(values):
            self.class_table.setItem(row, column, self.QTableWidgetItem(value))

    def remove_class_row(self) -> None:
        if self.class_table.rowCount() <= 2:
            self.set_status("Experiments require at least two classes.")
            return
        row = self.class_table.currentRow()
        if row >= 0:
            self.class_table.removeRow(row)

    def class_definitions(self) -> list[ClassificationClass]:
        result = []
        for row in range(self.class_table.rowCount()):
            values = [
                self.class_table.item(row, column).text()
                if self.class_table.item(row, column)
                else ""
                for column in range(5)
            ]
            result.append(
                ClassificationClass(
                    class_id=values[0],
                    name=values[1],
                    color=values[2],
                    shortcut=values[3],
                    mask_disposition=values[4],
                )
            )
        return result

    def apply_class_edits(self) -> None:
        if self.manifest is None:
            raise ValueError("Create or load an experiment first.")
        updated = self.manifest.model_copy(deep=True)
        updated.classes = self.class_definitions()
        save_experiment(
            updated,
            self.paths.root,
            audit_action="update_classes",
        )
        self.manifest = updated
        self.refresh_class_controls()
        self.refresh_classification_layers()
        self.set_status(
            "Applied class edits. Stable semantics remain locked when confirmed "
            "labels exist; cosmetic edits were audited."
        )

    def feature_sources(self) -> list[FeatureSource]:
        sources = []
        for line in _split_paths(self.feature_tables_edit.toPlainText()):
            if "=" in line:
                source_id, path = line.split("=", 1)
            else:
                path = line
                source_id = Path(path).stem
            source = FeatureSource(
                source_id=source_id.strip(), kind="table", path=path.strip()
            )
            source.selected_columns = self._retained_feature_source_columns.get(
                self._feature_source_signature(source), []
            )
            sources.append(source)
        for line in _split_paths(self.anndata_features_edit.toPlainText()):
            if "=" in line:
                source_id, specification = line.split("=", 1)
            else:
                specification = line
                source_id = Path(specification.split("::", 1)[0]).stem
            if "::" in specification:
                path, representation = specification.rsplit("::", 1)
            else:
                path, representation = specification, "X"
            source = FeatureSource(
                source_id=source_id.strip(),
                kind="anndata",
                path=path.strip(),
                representation=representation.strip(),
            )
            source.selected_columns = self._retained_feature_source_columns.get(
                self._feature_source_signature(source), []
            )
            sources.append(source)
        return sources

    @staticmethod
    def _feature_source_signature(source: FeatureSource) -> tuple[str, str, str, str]:
        return (
            source.source_id,
            source.kind,
            str(source.path or ""),
            str(source.representation or ""),
        )

    def refresh_feature_channel_choices(self) -> None:
        selected = set(self.selected_feature_channels())
        if not selected and self.manifest is not None:
            selected = set(self.manifest.synthetic_features.channels)
        panel_channels = (
            [str(value) for value in self.adata.var_names]
            if self.adata is not None
            else []
        )
        discovered_channels = list(self.current_image_paths)
        if self.manifest is not None and not self.current_image_paths:
            rois = (
                sorted(self.cohort["ROI"].astype(str).unique())
                if not self.cohort.empty
                else []
            )
            if rois:
                aliases = (
                    build_image_channel_aliases(
                        self.adata.var_names,
                        self.adata.var,
                    )
                    if self.adata is not None
                    else {}
                )
                discovered = discover_roi_images(
                    self.manifest.images_folders
                    + self.manifest.extra_images_folders,
                    rois[0],
                    channel_aliases=aliases,
                )
                discovered_channels.extend(discovered)
        channels = list(
            dict.fromkeys(
                discovered_channels
                or panel_channels
            )
        )
        for channel in (
            self.manifest.synthetic_features.channels
            if self.manifest is not None
            else []
        ):
            if channel not in channels:
                channels.append(channel)
        verified = set(discovered_channels)
        self.feature_channel_list.clear()
        for channel in channels:
            from qtpy.QtWidgets import QListWidgetItem

            item = QListWidgetItem(channel)
            item.setToolTip(
                (
                    "Discovered in the current/preview ROI. Selected channels "
                    "must also be available in every ROI used by the worker."
                    if channel in verified
                    else "Panel or saved-recipe channel not yet verified against "
                    "an experiment ROI."
                )
            )
            self.feature_channel_list.addItem(item)
            item.setSelected(channel in selected)
        self._update_feature_channel_summary()
        self.set_status(
            f"Found {len(channels)} feature channel choice(s); "
            f"{len(verified)} were discovered in an experiment ROI."
        )

    def selected_feature_channels(self) -> list[str]:
        if not hasattr(self, "feature_channel_list"):
            return []
        return [
            item.text() for item in self.feature_channel_list.selectedItems()
        ]

    def _update_feature_channel_summary(self) -> None:
        channels = self.selected_feature_channels()
        self.channels_edit.setText(
            ", ".join(channels) if channels else ""
        )
        self.channels_edit.setToolTip(
            (
                f"{len(channels)} explicitly selected channel(s)."
                if channels
                else "No explicit selection: use every channel discovered by "
                "the feature worker."
            )
        )
        self._update_feature_selection_summary()

    def _feature_family_toggled(self, family: str, checked: bool) -> None:
        items = self.feature_tree_items.get(family, {})
        if checked and not any(
            item.checkState(0) == self.Qt.Checked for item in items.values()
        ):
            self.feature_tree.blockSignals(True)
            for item in items.values():
                item.setCheckState(0, self.Qt.Checked)
            self.feature_tree.blockSignals(False)
        for item in items.values():
            item.setDisabled(not checked)
        self._update_feature_selection_summary()

    def _feature_tree_item_changed(self, _item, _column: int) -> None:
        self._update_feature_selection_summary()

    def selected_feature_names(self, family: str) -> list[str]:
        return [
            name
            for name, item in self.feature_tree_items.get(family, {}).items()
            if item.checkState(0) == self.Qt.Checked
        ]

    def _update_feature_selection_summary(self) -> None:
        if not hasattr(self, "feature_selection_summary"):
            return
        selected_counts = {
            family: len(self.selected_feature_names(family))
            if checkbox.isChecked()
            else 0
            for family, checkbox in self.feature_family_checks.items()
        }
        channel_count = len(self.selected_feature_channels())
        channel_text = (
            str(channel_count)
            if channel_count
            else "all consistently discovered"
        )
        self.feature_selection_summary.setText(
            "Channels: "
            f"{channel_text}; per-channel distribution "
            f"{selected_counts['distribution']}, region "
            f"{selected_counts['region']}, gradient "
            f"{selected_counts['gradient']}; per-cell shape "
            f"{selected_counts['shape']}, context "
            f"{selected_counts['context']}; ROI-rank statistics "
            f"{selected_counts['roi_rank']}."
        )

    def synthetic_recipe_from_controls(self) -> SyntheticFeatureRecipe:
        return SyntheticFeatureRecipe(
            channels=self.selected_feature_channels(),
            mask_offset_px=self.offset_spin.value(),
            allow_positive_offset_overlap=self.offset_overlap_check.isChecked(),
            distribution_features=self.distribution_check.isChecked(),
            region_features=self.region_check.isChecked(),
            gradient_features=self.gradient_check.isChecked(),
            shape_features=self.shape_check.isChecked(),
            context_features=self.context_check.isChecked(),
            roi_rank_features=self.roi_rank_check.isChecked(),
            distribution_feature_names=self.selected_feature_names(
                "distribution"
            ),
            region_feature_names=self.selected_feature_names("region"),
            gradient_feature_names=self.selected_feature_names("gradient"),
            shape_feature_names=self.selected_feature_names("shape"),
            context_feature_names=self.selected_feature_names("context"),
            roi_rank_statistics=self.selected_feature_names("roi_rank"),
            background_ring_px=self.background_ring_spin.value(),
            normalization_dict_path=(
                self.normalization_edit.text().strip() or None
            ),
        )

    def create_experiment(self) -> None:
        preview = self.preview_cohort()
        experiment_mode = str(self.experiment_mode_combo.currentData())
        feature_trial = None
        if experiment_mode == "feature_discovery_trial":
            selected_rois = self.selected_trial_rois()
            requested = self.trial_roi_count_spin.value()
            if len(selected_rois) != requested:
                raise ValueError(
                    f"Select exactly {requested} representative trial ROIs; "
                    f"currently selected: {len(selected_rois)}."
                )
            feature_trial = FeatureDiscoveryTrial(
                roi_selection=str(self.trial_roi_strategy_combo.currentData()),
                roi_count=requested,
                selected_rois=selected_rois,
            )
            trial_cells = int(
                preview.eligible_cells["ROI"].astype(str).isin(selected_rois).sum()
            )
            trial_text = (
                f" Feature extraction and initial classification will use "
                f"{trial_cells:,} cells in {requested} trial ROIs."
            )
        else:
            trial_text = ""
        reply = self.QMessageBox.question(
            self.root,
            "Freeze cohort",
            (
                f"Freeze {preview.eligible_cell_count:,} eligible identities across "
                f"{preview.represented_roi_count} ROIs? Later membership changes "
                f"require an explicit experiment revision.{trial_text}"
            ),
        )
        if reply != self.QMessageBox.Yes:
            return
        name = self.name_edit.text().strip()
        root_text = self.experiment_edit.text().strip()
        root = (
            Path(root_text)
            if root_text
            else self.project_root / "napari_sbt" / slugify(name)
        )
        root = root.expanduser().resolve(strict=False)
        if (root / "experiment.yaml").exists():
            raise FileExistsError(
                f"Experiment already exists at {root}. Load it or choose a new folder; "
                "the frozen cohort was not changed."
            )
        provisional_paths = save_cohort_snapshot(
            preview, root / "cohort" / "eligible_cells.parquet"
        )
        scope = preview.scope(
            mode=self.scope_combo.currentData(),
            obs_column=(
                self.obs_combo.currentText()
                if self.scope_combo.currentData() == "obs_values"
                else None
            ),
            obs_values=self.selected_scope_values(),
            snapshot_path=str(provisional_paths.relative_to(root)),
        )
        manifest = ExperimentManifest(
            name=name,
            project_root=str(self.project_root),
            anndata_path=self.anndata_edit.text().strip(),
            images_folders=_split_paths(self.images_edit.toPlainText()),
            extra_images_folders=_split_paths(self.extra_images_edit.toPlainText()),
            masks_folder=self.masks_edit.text().strip(),
            roi_obs=self.roi_obs_edit.text().strip(),
            object_id_obs=self.object_obs_edit.text().strip(),
            cell_scope=scope,
            classes=self.class_definitions(),
            experiment_mode=experiment_mode,
            feature_trial=feature_trial,
            feature_sources=self.feature_sources(),
            synthetic_features=self.synthetic_recipe_from_controls(),
            annotated_adata_path=self.annotated_path_edit.text().strip(),
        )
        self.paths = save_experiment(manifest, root, audit_action="create_experiment")
        self.experiment_edit.setText(str(root))
        self.load_existing_experiment(root)
        self.set_status(f"Created experiment {manifest.experiment_id} at {root}.")

    def choose_and_load_experiment(self) -> None:
        selected = self.QFileDialog.getExistingDirectory(
            self.root, "Choose napari_sbt experiment folder", str(self.project_root)
        )
        if selected:
            self.load_existing_experiment(Path(selected))

    def load_existing_experiment(self, path: Path) -> None:
        self.manifest, self.paths = load_experiment(path)
        self.model_bundle = None
        self.experiment_edit.setText(str(self.paths.root))
        self.name_edit.setText(self.manifest.name)
        self.anndata_edit.setText(_path_text(self.manifest.anndata_path))
        self.masks_edit.setText(self.manifest.masks_folder)
        self.roi_obs_edit.setText(self.manifest.roi_obs)
        self.object_obs_edit.setText(self.manifest.object_id_obs)
        self.experiment_mode_combo.setCurrentIndex(
            self.experiment_mode_combo.findData(self.manifest.experiment_mode)
        )
        if self.manifest.feature_trial is not None:
            self.trial_roi_count_spin.setMaximum(10000)
            self.trial_roi_count_spin.setValue(
                self.manifest.feature_trial.roi_count
            )
            self.trial_roi_strategy_combo.setCurrentIndex(
                self.trial_roi_strategy_combo.findData(
                    self.manifest.feature_trial.roi_selection
                )
            )
        self.images_edit.setPlainText("\n".join(self.manifest.images_folders))
        self.extra_images_edit.setPlainText("\n".join(self.manifest.extra_images_folders))
        self.offset_spin.setValue(self.manifest.synthetic_features.mask_offset_px)
        self.offset_overlap_check.setChecked(
            self.manifest.synthetic_features.allow_positive_offset_overlap
        )
        self.background_ring_spin.setValue(
            self.manifest.synthetic_features.background_ring_px
        )
        self.normalization_edit.setText(
            self.manifest.synthetic_features.normalization_dict_path or ""
        )
        self.distribution_check.setChecked(
            self.manifest.synthetic_features.distribution_features
        )
        self.region_check.setChecked(self.manifest.synthetic_features.region_features)
        self.gradient_check.setChecked(
            self.manifest.synthetic_features.gradient_features
        )
        self.shape_check.setChecked(self.manifest.synthetic_features.shape_features)
        self.context_check.setChecked(
            self.manifest.synthetic_features.context_features
        )
        self.roi_rank_check.setChecked(
            self.manifest.synthetic_features.roi_rank_features
        )
        selected_by_family = {
            "distribution": set(
                self.manifest.synthetic_features.distribution_feature_names
            ),
            "region": set(self.manifest.synthetic_features.region_feature_names),
            "gradient": set(
                self.manifest.synthetic_features.gradient_feature_names
            ),
            "shape": set(self.manifest.synthetic_features.shape_feature_names),
            "context": set(
                self.manifest.synthetic_features.context_feature_names
            ),
            "roi_rank": set(
                self.manifest.synthetic_features.roi_rank_statistics
            ),
        }
        self.feature_tree.blockSignals(True)
        for family, items in self.feature_tree_items.items():
            selected_names = selected_by_family[family]
            for name, item in items.items():
                item.setCheckState(
                    0,
                    self.Qt.Checked if name in selected_names else self.Qt.Unchecked,
                )
        self.feature_tree.blockSignals(False)
        self._retained_feature_source_columns = {
            self._feature_source_signature(source): list(source.selected_columns)
            for source in self.manifest.feature_sources
            if source.enabled
        }
        self.feature_tables_edit.setPlainText(
            "\n".join(
                f"{source.source_id}={source.path}"
                for source in self.manifest.feature_sources
                if source.kind == "table" and source.enabled
            )
        )
        self.anndata_features_edit.setPlainText(
            "\n".join(
                f"{source.source_id}={source.path}::{source.representation or 'X'}"
                for source in self.manifest.feature_sources
                if source.kind == "anndata" and source.enabled
            )
        )
        self._set_class_rows(self.manifest.classes)
        self.cohort = read_dataframe(
            self.paths.root / self.manifest.cell_scope.snapshot_path
        )
        per_roi_counts = (
            self.cohort.groupby("ROI", observed=True)
            .size()
            .rename("eligible_cells")
            .reset_index()
            .sort_values(["eligible_cells", "ROI"], ascending=[False, True])
            .reset_index(drop=True)
        )
        self.preview = CohortPreview(
            eligible_cells=self.cohort,
            total_cell_count=self.manifest.cell_scope.total_cell_count,
            per_roi_counts=per_roi_counts,
        )
        self._populate_trial_roi_list(
            per_roi_counts,
            selected_rois=(
                self.manifest.feature_trial.selected_rois
                if self.manifest.feature_trial is not None
                else ()
            ),
        )
        self._update_experiment_mode_state()
        if self.paths.labels.exists():
            self.labels = validate_labels(
                read_dataframe(self.paths.labels),
                class_ids=[item.class_id for item in self.manifest.classes],
                cohort=self.cohort,
            )
        else:
            self.labels = empty_labels()
        self.scores = pd.DataFrame()
        if self.paths.scores.exists() and self.manifest.active_feature_set_id:
            candidate_scores = read_dataframe(self.paths.scores)
            metadata_path = self.paths.models / "classifier_latest.json"
            try:
                model_metadata = (
                    json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata_path.is_file()
                    else {}
                )
            except (OSError, json.JSONDecodeError):
                model_metadata = {}
            scores_are_current = bool(
                not candidate_scores.empty
                and "feature_set_id" in candidate_scores
                and "model_id" in candidate_scores
                and candidate_scores["feature_set_id"]
                .fillna("")
                .eq(self.manifest.active_feature_set_id)
                .all()
                and candidate_scores["model_id"]
                .astype(str)
                .eq(str(model_metadata.get("model_id")))
                .all()
                and model_metadata.get("labels_fingerprint")
                == confirmed_labels_fingerprint(self.labels)
                and (
                    not self.manifest.active_model_features
                    or model_metadata.get("feature_set_hash")
                    == feature_set_hash(self.manifest.active_model_features)
                )
            )
            if scores_are_current:
                self.scores = candidate_scores
        reviewed_path = self.paths.labels.parent / "reviewed_rois.json"
        if reviewed_path.exists():
            self.reviewed_rois = set(
                json.loads(reviewed_path.read_text(encoding="utf-8")).get("rois", [])
            )
        else:
            self.reviewed_rois = set()
        self._load_explore_review_state()
        if self.manifest.anndata_path:
            self.load_anndata_selectors()
            self.scope_combo.setCurrentIndex(
                self.scope_combo.findData(self.manifest.cell_scope.mode)
            )
            if self.manifest.cell_scope.mode == "obs_values":
                self.obs_combo.setCurrentText(
                    self.manifest.cell_scope.obs_column or ""
                )
                self.refresh_scope_values()
                selected_values = set(self.manifest.cell_scope.obs_values)
                for index in range(self.value_list.count()):
                    item = self.value_list.item(index)
                    item.setSelected(item.text() in selected_values)
        self.refresh_feature_channel_choices()
        requested_channels = set(self.manifest.synthetic_features.channels)
        for index in range(self.feature_channel_list.count()):
            item = self.feature_channel_list.item(index)
            item.setSelected(item.text() in requested_channels)
        self._update_feature_channel_summary()
        self._update_feature_selection_summary()
        self.refresh_class_controls()
        self.refresh_rois()
        self._set_classification_enabled(True)
        self._update_scope_text()
        if self.roi_combo.count():
            self.load_roi(self.roi_combo.currentText())
        self.set_status(
            f"Loaded experiment {self.manifest.name!r}, revision "
            f"{self.manifest.revision}."
        )
        self.refresh_status()
        self.load_refinement_results(silent=True)

    def _update_scope_text(self) -> None:
        if self.manifest is None:
            self.scope_label.setText("No experiment: classification is disabled.")
            return
        mode_text = ""
        if (
            self.manifest.experiment_mode == "feature_discovery_trial"
            and self.manifest.feature_trial is not None
        ):
            trial_cells = int(
                self.cohort["ROI"]
                .astype(str)
                .isin(self.manifest.feature_trial.selected_rois)
                .sum()
            )
            mode_text = (
                f" — TRIAL: {trial_cells:,} cells in "
                f"{len(self.manifest.feature_trial.selected_rois)} selected ROIs"
            )
        elif self.manifest.feature_trial is not None:
            mode_text = " — full experiment promoted from a feature trial"
        self.scope_label.setText(
            f"{self.manifest.cell_scope.eligible_cell_count:,} eligible cells / "
            f"{self.manifest.cell_scope.total_cell_count:,} total cells across "
            f"{self.manifest.cell_scope.represented_roi_count} ROIs — "
            f"experiment {self.manifest.name!r} r{self.manifest.revision}"
            f"{mode_text}"
        )

    def refresh_class_controls(self) -> None:
        self._updating_queue_controls = True
        try:
            self.class_combo.clear()
            self.probability_class_combo.clear()
            self.queue_class_combo.clear()
            self.queue_class_combo.addItem("All predicted classes", None)
            for definition in self.manifest.classes:
                label = f"{definition.shortcut}: {definition.name}"
                icon = self._class_icon(definition.color)
                self.class_combo.addItem(icon, label, definition.class_id)
                self.probability_class_combo.addItem(icon, label, definition.class_id)
                self.queue_class_combo.addItem(
                    icon, definition.name, definition.class_id
                )
            self.queue_roi_combo.clear()
            self.queue_roi_combo.addItem("All current experiment ROIs", None)
            queue_rois = sorted(self.cohort["ROI"].astype(str).unique())
            if (
                self.manifest.experiment_mode == "feature_discovery_trial"
                and self.manifest.feature_trial is not None
            ):
                queue_rois = list(self.manifest.feature_trial.selected_rois)
            self.queue_roi_combo.addItems(queue_rois)
        finally:
            self._updating_queue_controls = False
        for shortcut in self._class_shortcuts:
            try:
                self.viewer.bind_key(shortcut, None, overwrite=True)
            except Exception as error:  # noqa: BLE001 - optional Napari key backend
                self.set_status(
                    f"Could not unbind previous class shortcut {shortcut!r}: {error}"
                )
        self._class_shortcuts = []
        for index, definition in enumerate(self.manifest.classes):
            shortcut = definition.shortcut

            def select_class(_viewer, selected_index=index):
                self.class_combo.setCurrentIndex(selected_index)

            self.viewer.bind_key(shortcut, overwrite=True)(select_class)
            self._class_shortcuts.append(shortcut)
        self._refresh_class_tally()
        self._refresh_model_storage_label()
        self._refresh_queue_if_scored()

    def _class_icon(self, color: str):
        pixmap = self.QPixmap(14, 14)
        pixmap.fill(self.QColor(color))
        return self.QIcon(pixmap)

    def _class_definition(self, class_id: str):
        if self.manifest is None:
            return None
        return next(
            (
                definition
                for definition in self.manifest.classes
                if definition.class_id == str(class_id)
            ),
            None,
        )

    def _refresh_model_storage_label(self) -> None:
        if self.paths is None:
            self.model_storage_label.setText(
                "No active experiment. Models are stored inside the experiment "
                "folder."
            )
            return
        model_path = self.paths.models / "classifier_latest.joblib"
        metadata_path = model_path.with_suffix(".json")
        state = "saved" if model_path.exists() else "not trained yet"
        current_model = ""
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                current_model = (
                    f"Current model: {metadata.get('model_id', 'unknown')} "
                    f"({metadata.get('model_type', 'unknown type')})\n"
                )
            except (OSError, ValueError):
                current_model = "Current model metadata could not be read.\n"
        self.model_storage_label.setText(
            f"{current_model}{state}: {model_path}\n"
            f"Provenance: {metadata_path}\n"
            f"Active model inputs: "
            f"{len(self.manifest.active_model_features) or 'all numeric features'}\n"
            "The Joblib file contains the fitted imputer and classifier. "
            "Retraining replaces the latest model; the JSON records "
            "its model ID, classes, features, label fingerprint, and versions."
        )

    def _refresh_class_tally(self) -> None:
        if not hasattr(self, "class_tally_table"):
            return
        definitions = [] if self.manifest is None else self.manifest.classes
        self.class_tally_table.setRowCount(len(definitions))
        for row, definition in enumerate(definitions):
            proposed = int(
                (
                    self.labels["class_id"].eq(definition.class_id)
                    & self.labels["state"].eq("proposed")
                ).sum()
            )
            confirmed = int(
                (
                    self.labels["class_id"].eq(definition.class_id)
                    & self.labels["state"].eq("confirmed")
                ).sum()
            )
            remaining = max(0, HGB_MIN_SAMPLES_LEAF - confirmed)
            values = (
                definition.name,
                str(proposed),
                str(confirmed),
                "target met" if remaining == 0 else f"{remaining} more",
            )
            for column, value in enumerate(values):
                item = self.QTableWidgetItem(value)
                if column == 0:
                    item.setIcon(self._class_icon(definition.color))
                self.class_tally_table.setItem(row, column, item)
        self.class_tally_table.setToolTip(
            "HistGradientBoosting uses 20 samples per leaf. Aim for at least "
            "20 confirmed cells in every class; proposed labels do not train "
            "the model."
        )

    def refresh_rois(self) -> None:
        if self.manifest is None:
            return
        eligible_rois = sorted(self.cohort["ROI"].astype(str).unique())
        if (
            self.manifest.experiment_mode == "feature_discovery_trial"
            and self.manifest.feature_trial is not None
        ):
            eligible_rois = list(self.manifest.feature_trial.selected_rois)
        rois = eligible_rois
        if (
            self.show_empty_rois.isChecked()
            and self.manifest.experiment_mode != "feature_discovery_trial"
        ):
            all_rois = set(discover_mask_files(self.manifest.masks_folder))
            rois = sorted(all_rois | set(eligible_rois))
        current = self.roi_combo.currentText()
        self.roi_combo.blockSignals(True)
        self.roi_combo.clear()
        self.roi_combo.addItems(rois)
        if current in rois:
            self.roi_combo.setCurrentText(current)
        self.roi_combo.blockSignals(False)
        self._refresh_roi_review_colours()

    def move_roi(self, step: int) -> None:
        """Move through the currently ordered ROI list."""

        if not self.roi_combo.count():
            return
        current = self.roi_combo.currentIndex()
        target = current + int(step)
        if target < 0 or target >= self.roi_combo.count():
            direction = "previous" if step < 0 else "next"
            self.set_status(f"There is no {direction} ROI in the current ordering.")
            return
        self.roi_combo.setCurrentIndex(target)

    def hide_all_layers(self) -> None:
        for layer in self.viewer.layers:
            layer.visible = False
        self.set_status("All Napari layers are hidden.")

    def show_all_layers(self) -> None:
        for layer in self.viewer.layers:
            layer.visible = True
        self.set_status("All Napari layers are visible.")

    def delete_all_layers(self) -> None:
        for layer in list(self.viewer.layers):
            self.viewer.layers.remove(layer)
        self._explore_layer_names.clear()
        self.set_status(
            "All Napari layers were deleted. Load the ROI again to restore the "
            "cohort and classification layers."
        )

    def _explore_state_path(self) -> Path | None:
        if self.paths is None:
            return None
        return self.paths.root / "explore" / "review_state.json"

    def _load_explore_review_state(self) -> None:
        self.explore_recipe = ExploreViewRecipe()
        self.explore_review_state = ExploreReviewState()
        path = self._explore_state_path()
        if path is None or not path.exists():
            self._refresh_reload_recipe_list()
            return
        try:
            self.explore_review_state = ExploreReviewState.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except Exception as exc:  # noqa: BLE001 - preserve usable experiment
            self.set_status(
                f"Could not read Explore review state from {path.name}: {exc}"
            )
        self._refresh_reload_recipe_list()

    def _save_explore_review_state(self) -> None:
        path = self._explore_state_path()
        if path is not None:
            write_json(
                path,
                self.explore_review_state.model_dump(mode="json"),
            )

    def _current_population_recipe_key(self) -> str | None:
        observation = self.population_obs_combo.currentText().strip()
        population = self.population_value_combo.currentText().strip()
        if not observation or not population:
            return None
        return population_recipe_key(observation, population)

    def save_population_view(self) -> None:
        if not self.explore_recipe.has_content:
            raise ValueError("Load at least one Explore layer before saving a view.")
        key = self._current_population_recipe_key()
        if key is None:
            raise ValueError("Select a population observation and population first.")
        self.explore_review_state.population_recipes[key] = (
            self.explore_recipe.model_copy(deep=True)
        )
        self._save_explore_review_state()
        if self.paths is not None:
            append_audit(
                self.paths,
                {
                    "action": "save_population_explore_view",
                    "population_observation": self.population_obs_combo.currentText(),
                    "population": self.population_value_combo.currentText(),
                    "view_fingerprint": self.explore_recipe.fingerprint,
                },
            )
        self.set_status(
            "Saved the current images, colours, populations, observation, "
            "marker overlays, layer visibility, opacity, contours, and contrast "
            "limits for "
            f"{self.population_value_combo.currentText()!r}."
        )

    def restore_population_view(self) -> None:
        if self._applying_explore_recipe:
            return
        key = self._current_population_recipe_key()
        recipe = (
            self.explore_review_state.population_recipes.get(key)
            if key is not None
            else None
        )
        if recipe is None:
            self._refresh_roi_review_colours()
            return
        self._apply_explore_recipe(recipe)
        self.set_status(
            f"Loaded the saved Explore view for "
            f"{self.population_value_combo.currentText()!r}."
        )

    def _set_list_selection(self, widget, values: Iterable[str]) -> None:
        selected = {str(value) for value in values}
        for index in range(widget.count()):
            item = widget.item(index)
            item.setSelected(item.text() in selected)

    def _recipe_layer_entries(self) -> list[dict]:
        recipe = self.explore_recipe
        entries: list[dict] = []
        if recipe.image_mode == "rgb" and recipe.image_channels:
            entries.append(
                {
                    "kind": "rgb",
                    "name": "population_qc_rgb",
                    "channels": list(recipe.image_channels),
                    "description": (
                        "RGB composite: " + " + ".join(recipe.image_channels)
                    ),
                }
            )
        elif recipe.image_mode != "none":
            for index, channel in enumerate(recipe.image_channels):
                name = f"image::{channel}"
                default_colormap = (
                    "gray"
                    if recipe.image_mode == "grayscale"
                    else SIX_COLOUR_COLORMAPS[
                        index % len(SIX_COLOUR_COLORMAPS)
                    ]
                )
                colormap = recipe.layer_colormaps.get(name, default_colormap)
                entries.append(
                    {
                        "kind": "image",
                        "name": name,
                        "channel": channel,
                        "description": f"Image [{colormap}]: {channel}",
                    }
                )
        if recipe.observation_overlay:
            name = f"obs::{recipe.observation_overlay}"
            colormap = recipe.layer_colormaps.get(name)
            if (
                not colormap
                and self.adata is not None
                and recipe.observation_overlay in self.adata.obs
                and not pd.api.types.is_numeric_dtype(
                    self.adata.obs[recipe.observation_overlay]
                )
            ):
                colormap = "adata.uns categorical palette"
            suffix = f" [{colormap}]" if colormap else ""
            entries.append(
                {
                    "kind": "observation",
                    "name": name,
                    "observation": recipe.observation_overlay,
                    "description": (
                        f"AnnData observation{suffix}: "
                        f"{recipe.observation_overlay}"
                    ),
                }
            )
        if recipe.population_observation:
            population_colours = (
                categorical_colour_map(
                    self.adata,
                    recipe.population_observation,
                )
                if (
                    self.adata is not None
                    and recipe.population_observation in self.adata.obs
                )
                else {}
            )
            for population in recipe.populations:
                name = (
                    f"population::{recipe.population_observation}::{population}"
                )
                colour = population_colours.get(population)
                suffix = f" [{colour}]" if colour else ""
                entries.append(
                    {
                        "kind": "population",
                        "name": name,
                        "observation": recipe.population_observation,
                        "population": population,
                        "description": (
                            f"Population{suffix}: "
                            f"{recipe.population_observation} = "
                            f"{population}"
                        ),
                    }
                )
        for marker in recipe.marker_overlays:
            name = f"adata.X::{marker}"
            colormap = recipe.layer_colormaps.get(name, "viridis")
            entries.append(
                {
                    "kind": "marker",
                    "name": name,
                    "marker": marker,
                    "description": f"adata.X marker [{colormap}]: {marker}",
                }
            )
        if self.manifest is not None:
            for name, description in MANAGED_RECIPE_LAYERS.items():
                entries.append(
                    {
                        "kind": "managed",
                        "name": name,
                        "description": description,
                    }
                )
        return entries

    def _refresh_reload_recipe_list(self) -> None:
        if not hasattr(self, "reload_recipe_list"):
            return
        self.reload_recipe_list.clear()
        entries = self._recipe_layer_entries()
        if not entries:
            from qtpy.QtWidgets import QListWidgetItem

            item = QListWidgetItem(
                "No Explore layers are currently configured for ROI reload."
            )
            item.setFlags(item.flags() & ~self.Qt.ItemIsSelectable)
            self.reload_recipe_list.addItem(item)
            return
        from qtpy.QtWidgets import QListWidgetItem

        for entry in entries:
            name = entry["name"]
            visible = self.explore_recipe.layer_visibility.get(name, True)
            if entry["kind"] == "managed":
                visible = self.explore_recipe.layer_visibility.get(
                    name,
                    MANAGED_LAYER_DEFAULT_VISIBILITY[name],
                )
                opacity = self.explore_recipe.layer_opacities.get(
                    name,
                    MANAGED_LAYER_DEFAULT_OPACITY[name],
                )
            else:
                opacity = self.explore_recipe.layer_opacities.get(name, 1.0)
            state = "👁 visible" if visible else "◌ hidden"
            contour = self.explore_recipe.layer_contours.get(
                name,
                MANAGED_LAYER_DEFAULT_CONTOUR.get(name),
            )
            contour_text = (
                f", contour {contour}px" if contour is not None else ""
            )
            contrast_limits = self.explore_recipe.layer_contrast_limits.get(name)
            contrast_text = (
                f", contrast {contrast_limits[0]:g}–{contrast_limits[1]:g}"
                if contrast_limits is not None
                else ""
            )
            item = QListWidgetItem(
                f"{entry['description']} — {state}, opacity {opacity:.2f}"
                f"{contour_text}"
                f"{contrast_text}"
            )
            item.setData(self.Qt.UserRole, entry)
            item.setToolTip(
                f"Napari layer: {name}\nThis layer will be reconstructed for "
                "the next ROI with this visibility, opacity, contour style, "
                "and contrast limits."
            )
            self.reload_recipe_list.addItem(item)

    def _prune_recipe_layer_settings(self) -> None:
        valid_names = {
            entry["name"] for entry in self._recipe_layer_entries()
        }
        payload = self.explore_recipe.model_dump(mode="json")
        changed = False
        for key in (
            "layer_colormaps",
            "layer_visibility",
            "layer_opacities",
            "layer_contours",
            "layer_contrast_limits",
        ):
            filtered = {
                name: value
                for name, value in payload[key].items()
                if name in valid_names
            }
            if filtered != payload[key]:
                payload[key] = filtered
                changed = True
        if changed:
            self.explore_recipe = ExploreViewRecipe.model_validate(payload)

    def _drop_recipe_layer_settings(self, payload: dict, name: str) -> None:
        for key in (
            "layer_colormaps",
            "layer_visibility",
            "layer_opacities",
            "layer_contours",
            "layer_contrast_limits",
        ):
            payload.get(key, {}).pop(name, None)

    def delete_selected_recipe_items(self) -> None:
        selected = [
            item.data(self.Qt.UserRole)
            for item in self.reload_recipe_list.selectedItems()
            if item.data(self.Qt.UserRole)
        ]
        if not selected:
            raise ValueError("Select one or more ROI reload recipe items.")
        payload = self.explore_recipe.model_dump(mode="json")
        removed = 0
        reset = 0
        for entry in selected:
            kind = entry["kind"]
            name = entry["name"]
            if kind in {"image", "rgb"}:
                if kind == "rgb":
                    payload["image_channels"] = []
                else:
                    payload["image_channels"] = [
                        channel
                        for channel in payload["image_channels"]
                        if channel != entry["channel"]
                    ]
                if not payload["image_channels"]:
                    payload["image_mode"] = "none"
            elif kind == "observation":
                payload["observation_overlay"] = None
            elif kind == "population":
                payload["populations"] = [
                    population
                    for population in payload["populations"]
                    if population != entry["population"]
                ]
                if not payload["populations"]:
                    payload["population_observation"] = None
            elif kind == "marker":
                payload["marker_overlays"] = [
                    marker
                    for marker in payload["marker_overlays"]
                    if marker != entry["marker"]
                ]
            elif kind == "managed":
                self._drop_recipe_layer_settings(payload, name)
                reset += 1
                continue
            self._drop_recipe_layer_settings(payload, name)
            removed += 1
        self._apply_explore_recipe(ExploreViewRecipe.model_validate(payload))
        actions = []
        if removed:
            actions.append(f"deleted {removed} Explore item(s)")
        if reset:
            actions.append(
                f"reset {reset} managed layer(s) to default display settings"
            )
        self.set_status("ROI reload recipe: " + " and ".join(actions) + ".")

    def _layer_reload_descriptor(self, layer) -> dict | None:
        metadata = getattr(layer, "metadata", None)
        if isinstance(metadata, dict):
            descriptor = metadata.get("napari_sbt_reload")
            if isinstance(descriptor, dict):
                return dict(descriptor)
        name = str(getattr(layer, "name", ""))
        if (
            name == "population_qc_rgb"
            and self.explore_recipe.image_mode == "rgb"
        ):
            return {
                "kind": "rgb",
                "name": name,
                "channels": list(self.explore_recipe.image_channels),
            }
        if name.startswith("image::"):
            return {
                "kind": "image",
                "name": name,
                "channel": name.removeprefix("image::"),
                "mode": "six_colour",
            }
        if name.startswith("obs::"):
            return {
                "kind": "observation",
                "name": name,
                "observation": name.removeprefix("obs::"),
            }
        if name.startswith("adata.X::"):
            return {
                "kind": "marker",
                "name": name,
                "marker": name.removeprefix("adata.X::"),
            }
        if name.startswith("population::"):
            parts = name.split("::", 2)
            if len(parts) == 3:
                return {
                    "kind": "population",
                    "name": name,
                    "observation": parts[1],
                    "population": parts[2],
                }
        return None

    def _layer_colormap_name(self, layer) -> str | None:
        colormap = getattr(layer, "colormap", None)
        if isinstance(colormap, str):
            return colormap
        name = getattr(colormap, "name", None)
        if name and not str(name).startswith("[unnamed"):
            return str(name)
        return None

    def update_recipe_from_current_layers(self) -> None:
        """Rebuild the ROI reload recipe from supported layers in the viewer."""

        descriptors: list[tuple[object, dict]] = []
        managed_layers: list[tuple[object, str]] = []
        non_recipe_layers: list[str] = []
        managed_names = set(MANAGED_RECIPE_LAYERS)
        separately_managed_names = {
            "cohort_preview",
            "manual_tissue_regions",
        }
        for layer in self.viewer.layers:
            name = str(getattr(layer, "name", ""))
            descriptor = self._layer_reload_descriptor(layer)
            if descriptor is not None:
                descriptors.append((layer, descriptor))
            elif name in managed_names:
                managed_layers.append((layer, name))
            elif name not in separately_managed_names:
                non_recipe_layers.append(name)
        if not descriptors and not managed_layers:
            self._apply_explore_recipe(ExploreViewRecipe())
            message = (
                "No replayable Explore or classifier layers were present; the ROI reload "
                "recipe is now empty."
            )
            if non_recipe_layers:
                message += (
                    " Unsupported layers were not added: "
                    + ", ".join(non_recipe_layers)
                    + "."
                )
            self.set_status(message)
            return

        rgb = [
            (layer, descriptor)
            for layer, descriptor in descriptors
            if descriptor["kind"] == "rgb"
        ]
        images = [
            (layer, descriptor)
            for layer, descriptor in descriptors
            if descriptor["kind"] == "image"
        ]
        ignored: list[str] = list(non_recipe_layers)
        if rgb:
            rgb_layer, rgb_descriptor = rgb[-1]
            image_mode = "rgb"
            image_channels = list(rgb_descriptor.get("channels", []))
            included = [(rgb_layer, rgb_descriptor)]
            ignored.extend(
                descriptor["name"] for _layer, descriptor in images
            )
        else:
            included = [
                (layer, descriptor)
                for layer, descriptor in images
                if descriptor.get("channel") in self.current_image_paths
            ]
            ignored.extend(
                descriptor["name"]
                for _layer, descriptor in images
                if descriptor.get("channel") not in self.current_image_paths
            )
            image_channels = [
                descriptor["channel"] for _layer, descriptor in included
            ]
            image_mode = "none"
            if image_channels:
                colormaps = {
                    self._layer_colormap_name(layer)
                    for layer, _descriptor in included
                }
                image_mode = (
                    "grayscale"
                    if colormaps <= {None, "gray", "grey"}
                    else "six_colour"
                )

        observation_layers = [
            (layer, descriptor)
            for layer, descriptor in descriptors
            if descriptor["kind"] == "observation"
        ]
        observation_overlay = None
        if observation_layers:
            observation_overlay = observation_layers[-1][1]["observation"]
            ignored.extend(
                descriptor["name"]
                for _layer, descriptor in observation_layers[:-1]
            )

        population_layers = [
            (layer, descriptor)
            for layer, descriptor in descriptors
            if descriptor["kind"] == "population"
        ]
        population_observation = None
        populations: list[str] = []
        if population_layers:
            population_observation = population_layers[0][1]["observation"]
            for _layer, descriptor in population_layers:
                if descriptor["observation"] == population_observation:
                    populations.append(descriptor["population"])
                else:
                    ignored.append(descriptor["name"])

        marker_layers = [
            (layer, descriptor)
            for layer, descriptor in descriptors
            if descriptor["kind"] == "marker"
        ]
        marker_overlays = [
            descriptor["marker"] for _layer, descriptor in marker_layers
        ]
        included_names = {
            descriptor["name"]
            for _layer, descriptor in (
                included
                + observation_layers[-1:]
                + [
                    item
                    for item in population_layers
                    if item[1]["observation"] == population_observation
                ]
                + marker_layers
            )
        }
        layer_colormaps: dict[str, str] = {}
        layer_visibility: dict[str, bool] = {
            name: value
            for name, value in self.explore_recipe.layer_visibility.items()
            if name in MANAGED_RECIPE_LAYERS
        }
        layer_opacities: dict[str, float] = {
            name: value
            for name, value in self.explore_recipe.layer_opacities.items()
            if name in MANAGED_RECIPE_LAYERS
        }
        layer_contours: dict[str, int] = {
            name: int(value)
            for name, value in self.explore_recipe.layer_contours.items()
            if name in MANAGED_RECIPE_LAYERS
        }
        layer_contrast_limits: dict[str, tuple[float, float]] = {
            name: (float(value[0]), float(value[1]))
            for name, value in self.explore_recipe.layer_contrast_limits.items()
            if name in MANAGED_RECIPE_LAYERS
        }
        for layer, descriptor in descriptors:
            name = descriptor["name"]
            if name not in included_names:
                continue
            colormap = self._layer_colormap_name(layer)
            if colormap and descriptor["kind"] in {
                "image",
                "observation",
                "marker",
            }:
                layer_colormaps[name] = colormap
            layer_visibility[name] = bool(getattr(layer, "visible", True))
            layer_opacities[name] = float(getattr(layer, "opacity", 1.0))
            if hasattr(layer, "contour"):
                layer_contours[name] = int(layer.contour)
            if hasattr(layer, "contrast_limits"):
                limits = layer.contrast_limits
                layer_contrast_limits[name] = (
                    float(limits[0]),
                    float(limits[1]),
                )
        for layer, name in managed_layers:
            layer_visibility[name] = bool(getattr(layer, "visible", True))
            layer_opacities[name] = float(getattr(layer, "opacity", 1.0))
            if hasattr(layer, "contour"):
                layer_contours[name] = int(layer.contour)
            if hasattr(layer, "contrast_limits"):
                limits = layer.contrast_limits
                layer_contrast_limits[name] = (
                    float(limits[0]),
                    float(limits[1]),
                )

        recipe = ExploreViewRecipe(
            image_mode=image_mode,
            image_channels=image_channels,
            observation_overlay=observation_overlay,
            population_observation=population_observation,
            populations=list(dict.fromkeys(populations)),
            marker_overlays=list(dict.fromkeys(marker_overlays)),
            layer_colormaps=layer_colormaps,
            layer_visibility=layer_visibility,
            layer_opacities=layer_opacities,
            layer_contours=layer_contours,
            layer_contrast_limits=layer_contrast_limits,
        )
        self._apply_explore_recipe(recipe)
        included_count = len(included_names) + len(managed_layers)
        message = (
            f"Updated the ROI reload recipe from {included_count} current "
            "Explore/classifier layer(s), including visibility, opacity, "
            "label contours, and contrast limits."
        )
        if ignored:
            message += (
                " Ignored unsupported or conflicting layers: "
                + ", ".join(ignored)
                + "."
            )
        self.set_status(message)

    def _apply_explore_recipe(self, recipe: ExploreViewRecipe) -> None:
        self._applying_explore_recipe = True
        try:
            self.explore_recipe = recipe.model_copy(deep=True)
            self._prune_recipe_layer_settings()
            if (
                recipe.observation_overlay
                and self.overlay_obs_combo.findText(recipe.observation_overlay) >= 0
            ):
                self.overlay_obs_combo.setCurrentText(recipe.observation_overlay)
            if (
                recipe.population_observation
                and self.population_obs_combo.findText(
                    recipe.population_observation
                )
                >= 0
            ):
                self.population_obs_combo.setCurrentText(
                    recipe.population_observation
                )
                self.refresh_population_values()
            self._set_list_selection(
                self.population_layer_list,
                recipe.populations,
            )
            self._set_list_selection(
                self.marker_overlay_list,
                recipe.marker_overlays,
            )
            self._set_list_selection(self.channel_list, recipe.image_channels)
        finally:
            self._applying_explore_recipe = False
        self._refresh_reload_recipe_list()
        if self.current_roi:
            self.replay_explore_view()

    def _mark_current_explore_viewed(self) -> None:
        if not self.current_roi or not self.explore_recipe.has_content:
            return
        fingerprint = self.explore_recipe.fingerprint
        viewed = set(self.explore_review_state.viewed_rois.get(fingerprint, []))
        viewed.add(str(self.current_roi))
        self.explore_review_state.viewed_rois[fingerprint] = sorted(viewed)
        self._save_explore_review_state()
        self._refresh_roi_review_colours()

    def _refresh_roi_review_colours(self) -> None:
        if not hasattr(self, "roi_combo"):
            return
        if not self.explore_recipe.has_content:
            for index in range(self.roi_combo.count()):
                self.roi_combo.setItemData(index, None, self.Qt.BackgroundRole)
                self.roi_combo.setItemData(index, None, self.Qt.ToolTipRole)
            self.viewed_rois_label.setText("No Explore view is active.")
            return
        viewed = set(
            self.explore_review_state.viewed_rois.get(
                self.explore_recipe.fingerprint,
                [],
            )
        )
        available = {
            self.roi_combo.itemText(index) for index in range(self.roi_combo.count())
        }
        for index in range(self.roi_combo.count()):
            roi = self.roi_combo.itemText(index)
            is_viewed = roi in viewed
            colour = self.QColor("#c6efce" if is_viewed else "#ffeb9c")
            self.roi_combo.setItemData(index, colour, self.Qt.BackgroundRole)
            self.roi_combo.setItemData(
                index,
                (
                    "Viewed with the current ROI reload recipe"
                    if is_viewed
                    else "Not yet viewed with the current ROI reload recipe"
                ),
                self.Qt.ToolTipRole,
            )
        reviewed_count = len(viewed & available)
        self.viewed_rois_label.setText(
            f"{reviewed_count}/{len(available)} ROIs viewed with the current "
            "images, overlays, and layer display settings. "
            "Green = viewed; amber = not viewed."
        )

    def _remove_layers(self, names: Iterable[str]) -> None:
        for name in names:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)

    def _is_recipe_tracked_layer(self, name: str) -> bool:
        if name in MANAGED_RECIPE_LAYERS:
            return True
        return name in {
            entry["name"]
            for entry in self._recipe_layer_entries()
            if entry["kind"] != "managed"
        }

    def _bind_recipe_display_tracking(self, layer) -> None:
        if getattr(layer, "_napari_sbt_recipe_display_bound", False):
            return
        events = getattr(layer, "events", None)
        if events is None:
            return

        def display_changed(_event=None, tracked_layer=layer):
            self._record_layer_display_state(tracked_layer)

        for event_name in ("visible", "opacity", "contour", "contrast_limits"):
            emitter = getattr(events, event_name, None)
            if emitter is not None:
                emitter.connect(display_changed)
        layer._napari_sbt_recipe_display_callback = display_changed
        layer._napari_sbt_recipe_display_bound = True

    def _record_layer_display_state(self, layer) -> None:
        if self._updating_recipe_layer_state:
            return
        name = str(getattr(layer, "name", ""))
        if not self._is_recipe_tracked_layer(name):
            return
        payload = self.explore_recipe.model_dump(mode="json")
        payload["layer_visibility"][name] = bool(
            getattr(layer, "visible", True)
        )
        payload["layer_opacities"][name] = float(
            getattr(layer, "opacity", 1.0)
        )
        if hasattr(layer, "contour"):
            payload["layer_contours"][name] = int(layer.contour)
        if hasattr(layer, "contrast_limits"):
            limits = layer.contrast_limits
            payload["layer_contrast_limits"][name] = [
                float(limits[0]),
                float(limits[1]),
            ]
        self.explore_recipe = ExploreViewRecipe.model_validate(payload)
        if name == "excluded_segmentation_context":
            self.context_check_display.blockSignals(True)
            self.context_check_display.setChecked(bool(layer.visible))
            self.context_check_display.blockSignals(False)
        self._refresh_reload_recipe_list()
        self._refresh_roi_review_colours()

    def _apply_managed_layer_display_settings(self) -> None:
        self._updating_recipe_layer_state = True
        try:
            for name in MANAGED_RECIPE_LAYERS:
                if name not in self.viewer.layers:
                    continue
                layer = self.viewer.layers[name]
                layer.visible = self.explore_recipe.layer_visibility.get(
                    name,
                    MANAGED_LAYER_DEFAULT_VISIBILITY[name],
                )
                layer.opacity = self.explore_recipe.layer_opacities.get(
                    name,
                    MANAGED_LAYER_DEFAULT_OPACITY[name],
                )
                if name in MANAGED_LAYER_DEFAULT_CONTOUR and hasattr(
                    layer, "contour"
                ):
                    layer.contour = self.explore_recipe.layer_contours.get(
                        name,
                        MANAGED_LAYER_DEFAULT_CONTOUR[name],
                    )
                if hasattr(layer, "contrast_limits"):
                    contrast_limits = (
                        self.explore_recipe.layer_contrast_limits.get(name)
                    )
                    if contrast_limits is not None:
                        layer.contrast_limits = contrast_limits
                self._bind_recipe_display_tracking(layer)
            context_visible = self.explore_recipe.layer_visibility.get(
                "excluded_segmentation_context",
                MANAGED_LAYER_DEFAULT_VISIBILITY[
                    "excluded_segmentation_context"
                ],
            )
            self.context_check_display.blockSignals(True)
            self.context_check_display.setChecked(context_visible)
            self.context_check_display.blockSignals(False)
        finally:
            self._updating_recipe_layer_state = False
        self._refresh_reload_recipe_list()

    def _replace_layer(self, name: str, data, layer_type: str, **kwargs):
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            previous_state = self._updating_recipe_layer_state
            self._updating_recipe_layer_state = True
            try:
                layer.data = data
                for key, value in kwargs.items():
                    if hasattr(layer, key):
                        setattr(layer, key, value)
            finally:
                self._updating_recipe_layer_state = previous_state
            self._bind_recipe_display_tracking(layer)
            return layer
        kwargs.setdefault("opacity", 1.0)
        method = getattr(self.viewer, f"add_{layer_type}")
        layer = method(data, name=name, **kwargs)
        self._bind_recipe_display_tracking(layer)
        return layer

    def _replace_explore_layer(
        self,
        name: str,
        data,
        layer_type: str,
        *,
        reload_descriptor: dict | None = None,
        **kwargs,
    ):
        layer = self._replace_layer(name, data, layer_type, **kwargs)
        if reload_descriptor is not None:
            metadata = dict(getattr(layer, "metadata", {}) or {})
            metadata["napari_sbt_reload"] = {
                "name": name,
                **reload_descriptor,
            }
            layer.metadata = metadata
        self._explore_layer_names.add(name)
        if (
            hasattr(layer, "contrast_limits")
            and name not in self.explore_recipe.layer_contrast_limits
        ):
            # Napari derives initial limits from the first ROI. Freeze those
            # limits immediately so subsequent ROIs use the identical view.
            self._record_layer_display_state(layer)
        return layer

    def _clear_explore_layers(self) -> None:
        self._remove_layers(list(self._explore_layer_names))
        self._explore_layer_names.clear()

    def load_roi(self, roi: str | None = None) -> None:
        if self.manifest is None:
            return
        roi = str(roi or self.roi_combo.currentText())
        if not roi:
            return
        mask_paths = discover_mask_files(self.manifest.masks_folder)
        if roi not in mask_paths:
            raise FileNotFoundError(f"No mask found for ROI {roi!r}.")
        full_mask = load_mask(mask_paths[roi])
        eligible = set(
            self.cohort.loc[
                self.cohort["ROI"].astype(str).eq(roi), "ObjectNumber"
            ].astype(int)
        )
        restricted = cohort_mask(full_mask, eligible)
        self.current_roi = roi
        self.current_mask = full_mask
        self.current_mask_path = mask_paths[roi]
        self.current_selected_object = None
        self.selected_cell_label.setText("No cohort cell selected")
        self._remove_layers([SELECTED_CELL_LAYER_NAME])
        cohort_visible = self.explore_recipe.layer_visibility.get(
            "classification_cohort",
            MANAGED_LAYER_DEFAULT_VISIBILITY["classification_cohort"],
        )
        self._replace_layer(
            "classification_cohort",
            restricted,
            "labels",
            visible=cohort_visible,
        )
        context = np.where(restricted == 0, full_mask, 0)
        context_visible = self.explore_recipe.layer_visibility.get(
            "excluded_segmentation_context",
            MANAGED_LAYER_DEFAULT_VISIBILITY[
                "excluded_segmentation_context"
            ],
        )
        context_layer = self._replace_layer(
            "excluded_segmentation_context",
            context,
            "labels",
            visible=context_visible,
            opacity=self.explore_recipe.layer_opacities.get(
                "excluded_segmentation_context",
                MANAGED_LAYER_DEFAULT_OPACITY[
                    "excluded_segmentation_context"
                ],
            ),
        )
        context_layer.visible = context_visible
        self._remove_layers(
            [
                CLASS_LAYER_NAMES["confirmed"],
                CLASS_LAYER_NAMES["proposed"],
                CLASS_LAYER_NAMES["predicted"],
                CLASS_LAYER_NAMES["uncertainty"],
            ]
        )
        self._clear_explore_layers()
        self.refresh_classification_layers()
        self.refresh_channel_list()
        if self.auto_reload_view_check.isChecked() and self.explore_recipe.has_content:
            self.replay_explore_view()
        else:
            self._refresh_roi_review_colours()
        self.set_status(
            f"ROI {roi}: {len(eligible)} eligible cells; clicks on other mask "
            "labels are ignored."
        )

    def auto_reload_explore_view(self) -> None:
        if (
            self.auto_reload_view_check.isChecked()
            and self.current_roi
            and self.explore_recipe.has_content
        ):
            self.replay_explore_view()
        else:
            self._refresh_roi_review_colours()

    def toggle_context(self, checked: bool) -> None:
        if "excluded_segmentation_context" in self.viewer.layers:
            self.viewer.layers["excluded_segmentation_context"].visible = checked

    def show_classifier_display_options(self) -> None:
        if self.classifier_display_dialog is None:
            self._build_classifier_display_dialog()
        self._sync_classifier_display_controls()
        self.classifier_display_dialog.show()
        self.classifier_display_dialog.raise_()
        self.classifier_display_dialog.activateWindow()

    def _build_classifier_display_dialog(self) -> None:
        from qtpy.QtWidgets import (
            QCheckBox,
            QDialog,
            QDoubleSpinBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )

        dialog = QDialog(self.root)
        dialog.setWindowTitle("Classifier display and cell picking")
        dialog.setMinimumWidth(620)
        layout = QVBoxLayout(dialog)
        explanation = QLabel(
            "These settings reproduce the useful display controls from the "
            "legacy CellPose QC viewer. A contour width of 0 shows filled "
            "cells; values above 0 show outlines that leave staining visible."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self.cell_picking_checkbox = QCheckBox(
            "Enable click-to-select while the Classify tab is active"
        )
        self.cell_picking_checkbox.setChecked(self.cell_picking_enabled)
        layout.addWidget(self.cell_picking_checkbox)

        form = QFormLayout()
        display_names = (
            "classification_cohort",
            "excluded_segmentation_context",
            CLASS_LAYER_NAMES["confirmed"],
            CLASS_LAYER_NAMES["proposed"],
            CLASS_LAYER_NAMES["predicted"],
            CLASS_LAYER_NAMES["uncertainty"],
            SELECTED_CELL_LAYER_NAME,
        )
        for name in display_names:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            visible = QCheckBox("Visible")
            opacity = QDoubleSpinBox()
            opacity.setRange(0.0, 1.0)
            opacity.setDecimals(2)
            opacity.setSingleStep(0.05)
            row_layout.addWidget(visible)
            row_layout.addWidget(QLabel("Opacity"))
            row_layout.addWidget(opacity)
            self.classifier_visibility_controls[name] = visible
            self.classifier_opacity_controls[name] = opacity
            if name in MANAGED_LAYER_DEFAULT_CONTOUR:
                contour = QSpinBox()
                contour.setRange(0, 20)
                row_layout.addWidget(QLabel("Contour px"))
                row_layout.addWidget(contour)
                self.classifier_contour_controls[name] = contour
            row_layout.addStretch(1)
            form.addRow(MANAGED_RECIPE_LAYERS[name], row)
        layout.addLayout(form)

        actions = QHBoxLayout()
        reset_button = QPushButton("Reset legacy-style defaults")
        apply_button = QPushButton("Apply")
        close_button = QPushButton("Close")
        actions.addWidget(reset_button)
        actions.addStretch(1)
        actions.addWidget(apply_button)
        actions.addWidget(close_button)
        layout.addLayout(actions)
        reset_button.clicked.connect(self._reset_classifier_display_controls)
        apply_button.clicked.connect(self.apply_classifier_display_options)
        close_button.clicked.connect(dialog.close)
        self.classifier_display_dialog = dialog

    def _sync_classifier_display_controls(self) -> None:
        for name, checkbox in self.classifier_visibility_controls.items():
            layer = self.viewer.layers[name] if name in self.viewer.layers else None
            visible = (
                bool(layer.visible)
                if layer is not None
                else self.explore_recipe.layer_visibility.get(
                    name,
                    MANAGED_LAYER_DEFAULT_VISIBILITY[name],
                )
            )
            opacity = (
                float(layer.opacity)
                if layer is not None
                else self.explore_recipe.layer_opacities.get(
                    name,
                    MANAGED_LAYER_DEFAULT_OPACITY[name],
                )
            )
            checkbox.setChecked(visible)
            self.classifier_opacity_controls[name].setValue(opacity)
            contour_control = self.classifier_contour_controls.get(name)
            if contour_control is not None:
                contour_control.setValue(
                    int(
                        getattr(layer, "contour")
                        if layer is not None and hasattr(layer, "contour")
                        else self.explore_recipe.layer_contours.get(
                            name,
                            MANAGED_LAYER_DEFAULT_CONTOUR[name],
                        )
                    )
                )
        self.cell_picking_checkbox.setChecked(self.cell_picking_enabled)

    def _reset_classifier_display_controls(self) -> None:
        for name, checkbox in self.classifier_visibility_controls.items():
            checkbox.setChecked(MANAGED_LAYER_DEFAULT_VISIBILITY[name])
            self.classifier_opacity_controls[name].setValue(
                MANAGED_LAYER_DEFAULT_OPACITY[name]
            )
            contour = self.classifier_contour_controls.get(name)
            if contour is not None:
                contour.setValue(MANAGED_LAYER_DEFAULT_CONTOUR[name])
        self.cell_picking_checkbox.setChecked(True)
        self.apply_classifier_display_options()

    def apply_classifier_display_options(self) -> None:
        payload = self.explore_recipe.model_dump(mode="json")
        for name, checkbox in self.classifier_visibility_controls.items():
            payload["layer_visibility"][name] = bool(checkbox.isChecked())
            payload["layer_opacities"][name] = float(
                self.classifier_opacity_controls[name].value()
            )
            contour = self.classifier_contour_controls.get(name)
            if contour is not None:
                payload["layer_contours"][name] = int(contour.value())
        self.cell_picking_enabled = bool(self.cell_picking_checkbox.isChecked())
        self.explore_recipe = ExploreViewRecipe.model_validate(payload)
        self._apply_managed_layer_display_settings()
        self._refresh_selected_cell_layer()
        self._refresh_roi_review_colours()
        self.set_status(
            "Applied classifier visibility, opacity, contour, and cell-picking "
            "settings to the active ROI reload recipe."
        )

    def _bind_viewer_cell_picking(self) -> None:
        callbacks = self.viewer.mouse_drag_callbacks
        if self._on_viewer_click not in callbacks:
            callbacks.append(self._on_viewer_click)

    def _on_viewer_click(self, _viewer, event) -> None:
        if (
            not self.cell_picking_enabled
            or self.current_mask is None
            or self.tabs.currentIndex() != self.classify_tab_index
            or event.type != "mouse_press"
            or getattr(event, "button", 1) != 1
        ):
            return
        position = event.position
        if len(position) < 2:
            return
        row = int(round(position[-2]))
        column = int(round(position[-1]))
        if (
            row < 0
            or column < 0
            or row >= self.current_mask.shape[0]
            or column >= self.current_mask.shape[1]
        ):
            return
        self._handle_clicked_object(int(self.current_mask[row, column]))

    def _on_cohort_click(self, layer, event) -> None:
        """Compatibility callback for integrations that bind the cohort layer."""

        if event.type == "mouse_press":
            self._handle_clicked_object(int(layer.get_value(event.position) or 0))

    def current_click_behavior(self) -> str:
        checked = self.click_behavior_group.checkedButton()
        if checked is None:
            return "select"
        return str(checked.property("napari_sbt_click_behavior") or "select")

    def _handle_clicked_object(self, object_id: int) -> None:
        if not self._select_cohort_object(object_id):
            return
        behavior = self.current_click_behavior()
        if behavior in {"proposed", "confirmed"}:
            self.annotate_selected(behavior)

    def _select_cohort_object(self, object_id: int) -> bool:
        eligible = set(
            pd.to_numeric(
                self.cohort.loc[
                    self.cohort["ROI"].astype(str).eq(str(self.current_roi)),
                    "ObjectNumber",
                ],
                errors="coerce",
            )
            .dropna()
            .astype(int)
        )
        if object_id <= 0 or object_id not in eligible:
            self.current_selected_object = None
            self._remove_layers([SELECTED_CELL_LAYER_NAME])
            self.selected_cell_label.setText("Cell is outside this experiment")
            self.set_status(
                "Cell is outside this experiment; annotation was ignored."
            )
            return False
        self.current_selected_object = int(object_id)
        self.selected_cell_label.setText(
            f"{self.current_roi} / object {self.current_selected_object}"
        )
        self._refresh_selected_cell_layer()
        return True

    def _refresh_selected_cell_layer(self) -> None:
        if self.current_mask is None or self.current_selected_object is None:
            self._remove_layers([SELECTED_CELL_LAYER_NAME])
            return
        from napari.utils.colormaps import DirectLabelColormap

        selected = (
            self.current_mask == int(self.current_selected_object)
        ).astype(np.uint8)
        layer = self._replace_layer(
            SELECTED_CELL_LAYER_NAME,
            selected,
            "labels",
            colormap=DirectLabelColormap(
                color_dict={None: "transparent", 0: "transparent", 1: "white"}
            ),
            visible=self.explore_recipe.layer_visibility.get(
                SELECTED_CELL_LAYER_NAME,
                MANAGED_LAYER_DEFAULT_VISIBILITY[SELECTED_CELL_LAYER_NAME],
            ),
            opacity=self.explore_recipe.layer_opacities.get(
                SELECTED_CELL_LAYER_NAME,
                MANAGED_LAYER_DEFAULT_OPACITY[SELECTED_CELL_LAYER_NAME],
            ),
        )
        layer.contour = self.explore_recipe.layer_contours.get(
            SELECTED_CELL_LAYER_NAME,
            MANAGED_LAYER_DEFAULT_CONTOUR[SELECTED_CELL_LAYER_NAME],
        )
        self._bind_recipe_display_tracking(layer)

    def refresh_channel_list(self) -> None:
        self.channel_list.clear()
        self.current_image_paths = {}
        if self.manifest is None or not self.current_roi:
            self.image_coverage_label.setText("No experiment ROI is loaded.")
            return
        channel_aliases = (
            build_image_channel_aliases(self.adata.var_names, self.adata.var)
            if self.adata is not None
            else {}
        )
        paths = discover_roi_images(
            self.manifest.images_folders + self.manifest.extra_images_folders,
            self.current_roi,
            channel_aliases=channel_aliases,
        )
        self.current_image_paths = dict(paths)
        logical_names = set(channel_aliases.values())
        matched = 0
        for channel, path in paths.items():
            from qtpy.QtWidgets import QListWidgetItem

            list_item = QListWidgetItem(channel)
            list_item.setData(self.Qt.UserRole, str(path))
            base_channel = channel.split(" [", 1)[0]
            if base_channel in logical_names:
                matched += 1
                list_item.setToolTip(
                    f"AnnData variable: {base_channel}\nImage: {path}"
                )
            else:
                list_item.setToolTip(f"Additional image (not in adata.var): {path}")
            self.channel_list.addItem(list_item)
            list_item.setSelected(channel in self.explore_recipe.image_channels)
        if paths:
            additional = len(paths) - matched
            self.image_coverage_label.setText(
                f"{len(paths)} images found for {self.current_roi}: "
                f"{matched} matched to adata.var, {additional} additional."
            )
        else:
            folders = (
                self.manifest.images_folders + self.manifest.extra_images_folders
            )
            self.image_coverage_label.setText(
                f"No images found for {self.current_roi} in {len(folders)} "
                "configured folder(s)."
            )

    def load_selected_channels(self) -> None:
        channels = [item.text() for item in self.channel_list.selectedItems()]
        if not channels:
            raise ValueError("Select at least one available image.")
        self._set_recipe_images("grayscale", channels)
        self.replay_explore_view()

    def load_six_colour_channels(self) -> None:
        channels = [item.text() for item in self.channel_list.selectedItems()]
        if not channels:
            raise ValueError("Select at least one available image.")
        self._set_recipe_images("six_colour", channels)
        self.replay_explore_view()

    def load_rgb(self) -> None:
        channels = [item.text() for item in self.channel_list.selectedItems()]
        if len(channels) < 3:
            raise ValueError("Select at least three channel images for an RGB view.")
        self._set_recipe_images("rgb", channels[:3])
        self.replay_explore_view()

    def _set_recipe_images(self, mode: str, channels: Iterable[str]) -> None:
        payload = self.explore_recipe.model_dump(mode="json")
        payload["image_mode"] = mode
        payload["image_channels"] = [str(channel) for channel in channels]
        for key in (
            "layer_colormaps",
            "layer_visibility",
            "layer_opacities",
            "layer_contours",
            "layer_contrast_limits",
        ):
            payload[key] = {
                name: value
                for name, value in payload[key].items()
                if not (
                    name.startswith("image::")
                    or name == "population_qc_rgb"
                )
            }
        self.explore_recipe = ExploreViewRecipe.model_validate(payload)

    def _recipe_display_settings(
        self,
        name: str,
        *,
        default_colormap: str | None = None,
    ) -> dict:
        settings = {
            "visible": self.explore_recipe.layer_visibility.get(name, True),
            "opacity": self.explore_recipe.layer_opacities.get(name, 1.0),
        }
        colormap = self.explore_recipe.layer_colormaps.get(
            name,
            default_colormap,
        )
        if colormap:
            settings["colormap"] = colormap
        contrast_limits = self.explore_recipe.layer_contrast_limits.get(name)
        if contrast_limits is not None:
            settings["contrast_limits"] = contrast_limits
        return settings

    def _render_recipe_images(self) -> int:
        recipe = self.explore_recipe
        if recipe.image_mode == "none" or not recipe.image_channels:
            return 0
        missing = [
            channel
            for channel in recipe.image_channels
            if channel not in self.current_image_paths
        ]
        available = [
            channel
            for channel in recipe.image_channels
            if channel in self.current_image_paths
        ]
        if missing:
            self.set_status(
                f"ROI {self.current_roi} is missing {len(missing)} requested "
                f"image(s): {', '.join(missing)}."
            )
        if recipe.image_mode == "rgb":
            if len(available) != 3:
                self.set_status(
                    "The RGB composite was not loaded because all three saved "
                    "channels are not available for this ROI."
                )
                return 0
            images = []
            for channel in available:
                image, is_rgb = load_display_image(
                    self.current_image_paths[channel]
                )
                if is_rgb:
                    raise ValueError(
                        f"RGB source {channel!r} cannot be one component of a "
                        "three-channel composite."
                    )
                images.append(image)
            if len({image.shape for image in images}) != 1:
                raise ValueError("RGB channels have mismatched shapes.")
            self._replace_explore_layer(
                "population_qc_rgb",
                np.stack(images, axis=-1),
                "image",
                reload_descriptor={
                    "kind": "rgb",
                    "channels": list(recipe.image_channels),
                },
                rgb=True,
                blending="translucent",
                **self._recipe_display_settings("population_qc_rgb"),
            )
            return 1
        loaded = 0
        for channel in available:
            image, is_rgb = load_display_image(self.current_image_paths[channel])
            name = f"image::{channel}"
            recipe_index = recipe.image_channels.index(channel)
            default_colormap = (
                "gray"
                if recipe.image_mode == "grayscale"
                else SIX_COLOUR_COLORMAPS[
                    recipe_index % len(SIX_COLOUR_COLORMAPS)
                ]
            )
            kwargs = {
                "rgb": is_rgb,
                "blending": "translucent" if is_rgb else "additive",
                **self._recipe_display_settings(
                    name,
                    default_colormap=None if is_rgb else default_colormap,
                ),
            }
            self._replace_explore_layer(
                name,
                image,
                "image",
                reload_descriptor={
                    "kind": "image",
                    "channel": channel,
                    "mode": recipe.image_mode,
                },
                **kwargs,
            )
            loaded += 1
        return loaded

    def refresh_population_values(self) -> None:
        current_value = self.population_value_combo.currentText()
        selected_layers = {
            item.text() for item in self.population_layer_list.selectedItems()
        }
        self.population_value_combo.blockSignals(True)
        self.population_value_combo.clear()
        self.population_layer_list.clear()
        if (
            self.adata is None
            or self.population_obs_combo.currentText() not in self.adata.obs
        ):
            self.population_value_combo.blockSignals(False)
            return
        observation = self.population_obs_combo.currentText()
        colour_map = categorical_colour_map(self.adata, observation)
        values = list(colour_map)
        self.population_value_combo.addItems(values)
        if current_value in values:
            self.population_value_combo.setCurrentText(current_value)
        for value in values:
            from qtpy.QtWidgets import QListWidgetItem

            item = QListWidgetItem(value)
            colour = self.QColor(colour_map[value])
            if colour.isValid():
                item.setBackground(colour)
                item.setForeground(
                    self.QColor("#111111" if colour.lightness() > 145 else "#ffffff")
                )
            item.setToolTip(
                f"{observation}={value}; colour retrieved from AnnData when available."
            )
            self.population_layer_list.addItem(item)
            item.setSelected(
                value in selected_layers
                or (
                    self.explore_recipe.population_observation == observation
                    and value in self.explore_recipe.populations
                )
            )
        self.population_value_combo.blockSignals(False)
        if not self._applying_explore_recipe:
            self.restore_population_view()

    def render_obs_overlay(self) -> None:
        observation = self.overlay_obs_combo.currentText()
        if not observation:
            raise ValueError("Select an AnnData observation.")
        self.explore_recipe = self.explore_recipe.model_copy(
            update={"observation_overlay": observation},
            deep=True,
        )
        self.replay_explore_view()

    def _roi_adata_rows(self):
        if self.adata is None or self.current_mask is None or self.manifest is None:
            raise RuntimeError("Load an experiment ROI and AnnData first.")
        roi_selector = (
            self.adata.obs[self.manifest.roi_obs].astype(str).eq(self.current_roi)
        )
        rows = self.adata.obs.loc[
            roi_selector
        ]
        object_ids = pd.to_numeric(
            rows[self.manifest.object_id_obs], errors="coerce"
        ).astype("Int64")
        eligible = set(
            self.cohort.loc[
                self.cohort["ROI"].astype(str).eq(self.current_roi), "ObjectNumber"
            ].astype(int)
        )
        selected = object_ids.notna() & object_ids.isin(eligible)
        return rows, object_ids, selected, roi_selector.to_numpy()

    def _direct_label_colormap(self, colours: dict[int, str]):
        from napari.utils.colormaps import DirectLabelColormap

        return DirectLabelColormap(color_dict={None: "#00000000", **colours})

    def _render_observation_overlay(self, observation: str) -> int:
        rows, object_ids, selected, _roi_selector = self._roi_adata_rows()
        if observation not in rows:
            self.set_status(
                f"Saved AnnData observation {observation!r} is no longer available."
            )
            return 0
        values = rows[observation]
        if pd.api.types.is_numeric_dtype(values):
            mapping = pd.Series(
                pd.to_numeric(values[selected], errors="coerce").to_numpy(),
                index=object_ids[selected].astype(int),
            )
            overlay = _identity_value_map(self.current_mask, mapping)
            name = f"obs::{observation}"
            self._replace_explore_layer(
                name,
                overlay,
                "image",
                reload_descriptor={
                    "kind": "observation",
                    "observation": observation,
                },
                blending="additive",
                **self._recipe_display_settings(
                    name,
                    default_colormap="viridis",
                ),
            )
        else:
            categories = sorted(values[selected].dropna().astype(str).unique())
            codes = {value: index + 1 for index, value in enumerate(categories)}
            mapping = pd.Series(
                values[selected].astype(str).map(codes).to_numpy(),
                index=object_ids[selected].astype(int),
            )
            overlay = _identity_value_map(self.current_mask, mapping, dtype=np.int32)
            population_colours = categorical_colour_map(self.adata, observation)
            name = f"obs::{observation}"
            layer = self._replace_explore_layer(
                name,
                overlay,
                "labels",
                reload_descriptor={
                    "kind": "observation",
                    "observation": observation,
                },
                colormap=self._direct_label_colormap(
                    {
                        code: population_colours[value]
                        for value, code in codes.items()
                    }
                ),
                visible=self.explore_recipe.layer_visibility.get(name, True),
                opacity=self.explore_recipe.layer_opacities.get(name, 1.0),
            )
            if hasattr(layer, "contour"):
                layer.contour = self.explore_recipe.layer_contours.get(name, 1)
        return 1

    def load_selected_population_layers(self) -> None:
        populations = [
            item.text() for item in self.population_layer_list.selectedItems()
        ]
        if not populations and self.population_value_combo.currentText():
            populations = [self.population_value_combo.currentText()]
            self._set_list_selection(self.population_layer_list, populations)
        if not populations:
            raise ValueError("Select at least one population.")
        observation = self.population_obs_combo.currentText()
        self.explore_recipe = self.explore_recipe.model_copy(
            update={
                "population_observation": observation,
                "populations": populations,
            },
            deep=True,
        )
        self.replay_explore_view()

    def _render_population_layers(
        self,
        observation: str,
        populations: Iterable[str],
    ) -> int:
        rows, object_ids, selected, _roi_selector = self._roi_adata_rows()
        if observation not in rows:
            self.set_status(
                f"Saved population observation {observation!r} is unavailable."
            )
            return 0
        values = rows[observation].astype(str)
        colour_map = categorical_colour_map(self.adata, observation)
        loaded = 0
        for population in populations:
            population_selected = selected & values.eq(str(population))
            mapping = pd.Series(
                np.ones(int(population_selected.sum()), dtype=np.int32),
                index=object_ids[population_selected].astype(int),
            )
            overlay = _identity_value_map(
                self.current_mask,
                mapping,
                dtype=np.int32,
            )
            colour = colour_map.get(str(population), "#ffffff")
            name = f"population::{observation}::{population}"
            layer = self._replace_explore_layer(
                name,
                overlay,
                "labels",
                reload_descriptor={
                    "kind": "population",
                    "observation": observation,
                    "population": population,
                },
                colormap=self._direct_label_colormap({1: colour}),
                visible=self.explore_recipe.layer_visibility.get(name, True),
                opacity=self.explore_recipe.layer_opacities.get(name, 1.0),
            )
            if hasattr(layer, "contour"):
                layer.contour = self.explore_recipe.layer_contours.get(name, 1)
            loaded += 1
        return loaded

    def load_selected_marker_overlays(self) -> None:
        markers = [item.text() for item in self.marker_overlay_list.selectedItems()]
        if not markers:
            raise ValueError("Select at least one AnnData marker.")
        self.explore_recipe = self.explore_recipe.model_copy(
            update={"marker_overlays": markers},
            deep=True,
        )
        self.replay_explore_view()

    def _render_marker_overlays(self, markers: Iterable[str]) -> int:
        _rows, object_ids, selected, roi_selector = self._roi_adata_rows()
        loaded = 0
        for marker in markers:
            try:
                values = marker_values(self.adata, marker)[roi_selector]
            except KeyError:
                self.set_status(
                    f"Saved AnnData marker {marker!r} is no longer available."
                )
                continue
            mapping = pd.Series(
                pd.to_numeric(values[selected.to_numpy()], errors="coerce"),
                index=object_ids[selected].astype(int),
            )
            overlay = _identity_value_map(self.current_mask, mapping)
            name = f"adata.X::{marker}"
            self._replace_explore_layer(
                name,
                overlay,
                "image",
                reload_descriptor={
                    "kind": "marker",
                    "marker": marker,
                },
                blending="additive",
                **self._recipe_display_settings(
                    name,
                    default_colormap="viridis",
                ),
            )
            loaded += 1
        return loaded

    def replay_explore_view(self) -> None:
        """Render the active ROI-independent recipe and record this review."""

        if not self.current_roi or self.current_mask is None:
            return
        self._prune_recipe_layer_settings()
        self._clear_explore_layers()
        loaded = self._render_recipe_images()
        if self.explore_recipe.observation_overlay:
            loaded += self._render_observation_overlay(
                self.explore_recipe.observation_overlay
            )
        if (
            self.explore_recipe.population_observation
            and self.explore_recipe.populations
        ):
            loaded += self._render_population_layers(
                self.explore_recipe.population_observation,
                self.explore_recipe.populations,
            )
        if self.explore_recipe.marker_overlays:
            loaded += self._render_marker_overlays(
                self.explore_recipe.marker_overlays
            )
        self._apply_managed_layer_display_settings()
        managed_present = sum(
            name in self.viewer.layers for name in MANAGED_RECIPE_LAYERS
        )
        self._refresh_reload_recipe_list()
        if loaded or managed_present:
            self._mark_current_explore_viewed()
            self.set_status(
                f"Replayed {loaded} Explore and {managed_present} managed "
                f"classification layer setting(s) for ROI {self.current_roi}."
            )
        else:
            self._refresh_roi_review_colours()

    def rank_rois_by_population(self) -> None:
        observation = self.population_obs_combo.currentText()
        value = self.population_value_combo.currentText()
        subset = self.adata.obs.loc[
            self.adata.obs[observation].astype(str).eq(value)
        ]
        counts = (
            subset.groupby(self.manifest.roi_obs, observed=True)
            .size()
            .sort_values(ascending=False)
        )
        eligible = set(self.cohort["ROI"].astype(str))
        if (
            self.manifest.experiment_mode == "feature_discovery_trial"
            and self.manifest.feature_trial is not None
        ):
            eligible &= set(self.manifest.feature_trial.selected_rois)
        ranked = [str(roi) for roi in counts.index if str(roi) in eligible]
        self.roi_combo.blockSignals(True)
        self.roi_combo.clear()
        self.roi_combo.addItems(ranked)
        self.roi_combo.blockSignals(False)
        self._refresh_roi_review_colours()
        if ranked:
            self.roi_combo.setCurrentIndex(0)
            self.load_roi(ranked[0])
        self.set_status(
            f"ROIs ranked by abundance of {observation}={value}; cohort-empty ROIs "
            "remain excluded."
        )

    def use_population_as_cohort(self) -> None:
        observation = self.population_obs_combo.currentText()
        value = self.population_value_combo.currentText()
        self.scope_combo.setCurrentIndex(self.scope_combo.findData("obs_values"))
        self.obs_combo.setCurrentText(observation)
        self.refresh_scope_values()
        matches = self.value_list.findItems(value, self.Qt.MatchExactly)
        for item in matches:
            item.setSelected(True)
        self.tabs.setCurrentIndex(0)
        self.preview_cohort()
        self.set_status(
            "Population transferred into Setup. Review the cohort-only mask preview "
            "and click Create confirmed experiment to freeze it."
        )

    def selected_class_id(self) -> str:
        value = self.class_combo.currentData()
        if value is None:
            raise ValueError("Select a classification class.")
        return str(value)

    def annotate_selected(self, state: str) -> None:
        try:
            if self.current_selected_object is None:
                raise ValueError("Select an eligible cohort cell first.")
            class_id = self.selected_class_id()
            self.labels = set_label(
                self.labels,
                roi=self.current_roi,
                object_number=self.current_selected_object,
                class_id=class_id,
                state=state,
                source="manual",
                user=os.environ.get("USERNAME") or os.environ.get("USER", ""),
            )
            # The viewer click has already been checked against the frozen
            # cohort and the class/state come from controlled widgets. Avoid
            # a full cohort merge on every click; bulk/import paths still use
            # the complete label validator.
            write_dataframe(self.paths.labels, self.labels)
            append_audit(
                self.paths,
                {
                    "action": "set_label",
                    "ROI": self.current_roi,
                    "ObjectNumber": self.current_selected_object,
                    "class_id": class_id,
                    "state": state,
                },
            )
            should_lock = bool((self.labels["state"] == "confirmed").any())
            if should_lock and not self.manifest.locked:
                self.manifest.locked = True
                save_experiment(
                    self.manifest,
                    self.paths.root,
                    audit_action="first_confirmed_label",
                )
            self._refresh_single_classification_object(
                self.current_selected_object,
                class_id=class_id,
                state=state,
            )
            self._refresh_class_tally()
            class_definition = self._class_definition(class_id)
            class_name = (
                class_definition.name if class_definition is not None else class_id
            )
            stale_note = (
                " Model now requires retraining." if state == "confirmed" else ""
            )
            self.set_status(
                f"Set {self.current_roi}/{self.current_selected_object} to "
                f"{class_name} ({state}).{stale_note}"
            )
        except Exception as error:  # noqa: BLE001 - Qt callback error boundary
            self.set_status(f"ERROR — {type(error).__name__}: {error}")
            self.QMessageBox.critical(
                self.root, "napari_sbt", f"{type(error).__name__}: {error}"
            )

    def _refresh_single_classification_object(
        self,
        object_id: int,
        *,
        class_id: str,
        state: str,
    ) -> None:
        """Update one annotated object without rebuilding whole-ROI rasters."""

        if self.current_mask is None or state not in {"proposed", "confirmed"}:
            self.refresh_classification_layers()
            return
        layer_names = {
            label_state: CLASS_LAYER_NAMES[label_state]
            for label_state in ("proposed", "confirmed")
        }
        if any(name not in self.viewer.layers for name in layer_names.values()):
            self.refresh_classification_layers()
            return
        pixels = self.current_mask == int(object_id)
        if not np.any(pixels):
            self.refresh_classification_layers()
            return
        class_code = self._class_code_map()[str(class_id)]
        for label_state, layer_name in layer_names.items():
            layer = self.viewer.layers[layer_name]
            data = np.asarray(layer.data)
            data[pixels] = class_code if label_state == state else 0
            layer.refresh()

    def confirm_all_proposed(self) -> None:
        self.labels = confirm_proposed(self.labels)
        write_dataframe(self.paths.labels, self.labels)
        self.manifest.locked = bool((self.labels["state"] == "confirmed").any())
        save_experiment(self.manifest, self.paths.root, audit_action="confirm_proposals")
        self.refresh_classification_layers()
        self.refresh_status()

    def mark_roi_reviewed(self) -> None:
        if not self.current_roi:
            raise ValueError("Load an eligible ROI first.")
        self.reviewed_rois.add(str(self.current_roi))
        write_json(
            self.paths.labels.parent / "reviewed_rois.json",
            {"rois": sorted(self.reviewed_rois)},
        )
        append_audit(
            self.paths, {"action": "mark_roi_reviewed", "ROI": self.current_roi}
        )
        self.refresh_status()

    def refresh_status(self) -> None:
        self._refresh_class_tally()
        self.refresh_refinement_readiness()
        if self.manifest is None:
            self.set_status("FRESHNESS — no active experiment.")
            return
        cohort_hash = dataframe_sha256(
            self.cohort, ["obs_name", "ROI", "ObjectNumber"]
        )
        cohort_state = (
            "current"
            if cohort_hash == self.manifest.cell_scope.snapshot_sha256
            else "STALE/MISMATCHED"
        )
        confirmed = self.labels.loc[self.labels["state"].eq("confirmed")]
        proposed = int(self.labels["state"].eq("proposed").sum())
        class_counts = {
            item.class_id: int(confirmed["class_id"].eq(item.class_id).sum())
            for item in self.manifest.classes
        }
        feature_state = (
            self.manifest.active_feature_set_id or "not built"
        )
        model_path = self.paths.models / "classifier_latest.json"
        model_metadata = (
            json.loads(model_path.read_text(encoding="utf-8"))
            if model_path.exists()
            else {}
        )
        model_state = "not trained"
        if model_metadata:
            labels_current = model_metadata.get(
                "labels_fingerprint"
            ) == confirmed_labels_fingerprint(self.labels)
            feature_selection_current = (
                not self.manifest.active_model_features
                or model_metadata.get("feature_set_hash")
                == feature_set_hash(self.manifest.active_model_features)
            )
            model_state = (
                "current cohort/features/labels"
                if model_metadata.get("cohort_fingerprint")
                == self.manifest.cell_scope.snapshot_sha256
                and model_metadata.get("feature_set_id")
                == self.manifest.active_feature_set_id
                and labels_current
                and feature_selection_current
                else "STALE cohort/features/labels"
            )
        scores_current = (
            not self.scores.empty
            and "scorable" in self.scores
            and "feature_set_id" in self.scores
            and self.scores["feature_set_id"]
            .fillna("")
            .eq(self.manifest.active_feature_set_id or "")
            .all()
            and (
                not model_metadata
                or self.scores["model_id"]
                .astype(str)
                .eq(str(model_metadata.get("model_id")))
                .all()
            )
        )
        scored = int(self.scores["scorable"].fillna(False).sum()) if scores_current else 0
        working_cohort = self.cohort
        if (
            self.manifest.experiment_mode == "feature_discovery_trial"
            and self.manifest.feature_trial is not None
        ):
            working_cohort = self.cohort.loc[
                self.cohort["ROI"]
                .astype(str)
                .isin(self.manifest.feature_trial.selected_rois)
            ]
        represented = set(working_cohort["ROI"].astype(str))
        reviewed = len(represented & self.reviewed_rois)
        self.set_status(
            "FRESHNESS — "
            f"cohort={cohort_state}; classes={class_counts}; "
            f"proposals={proposed}; feature_set={feature_state}; "
            f"model_features={len(self.manifest.active_model_features) or 'all'}; "
            f"model={model_state}; scored_current={scored}/{len(working_cohort)}; "
            f"reviewed_ROIs={reviewed}/{len(represented)}."
        )

    def seed_proposals_from_obs(self) -> None:
        if self.adata is None:
            raise ValueError("Seeding proposals requires AnnData.")
        observation = self.overlay_obs_combo.currentText()
        if observation not in self.adata.obs:
            raise ValueError("Select an AnnData observation in Explore first.")
        definitions = {
            item.class_id.casefold(): item.class_id for item in self.manifest.classes
        }
        definitions.update(
            {item.name.casefold(): item.class_id for item in self.manifest.classes}
        )
        values = self.adata.obs[observation].astype("string")
        working_rois = None
        if (
            self.manifest.experiment_mode == "feature_discovery_trial"
            and self.manifest.feature_trial is not None
        ):
            working_rois = set(self.manifest.feature_trial.selected_rois)
        confirmed_identities = set(
            self.labels.loc[
                self.labels["state"].eq("confirmed"), ["ROI", "ObjectNumber"]
            ]
            .astype({"ROI": str, "ObjectNumber": int})
            .itertuples(index=False, name=None)
        )
        seeded = 0
        for row in self.cohort.itertuples():
            if working_rois is not None and str(row.ROI) not in working_rois:
                continue
            if row.obs_name not in values.index:
                continue
            if (str(row.ROI), int(row.ObjectNumber)) in confirmed_identities:
                continue
            class_id = definitions.get(str(values.loc[row.obs_name]).casefold())
            if class_id is None:
                continue
            self.labels = set_label(
                self.labels,
                roi=row.ROI,
                object_number=row.ObjectNumber,
                class_id=class_id,
                state="proposed",
                source=f"adata.obs:{observation}",
            )
            seeded += 1
        write_dataframe(self.paths.labels, self.labels)
        append_audit(
            self.paths,
            {
                "action": "seed_proposals_from_obs",
                "observation": observation,
                "count": seeded,
            },
        )
        self.refresh_classification_layers()
        self.refresh_status()
        self.set_status(
            f"Seeded {seeded:,} matching assignments as proposals; none were confirmed."
        )

    def _class_code_map(self) -> dict[str, int]:
        return {
            definition.class_id: index + 1
            for index, definition in enumerate(self.manifest.classes)
        }

    def _class_colors(self) -> dict[int, str]:
        return {
            index + 1: definition.color
            for index, definition in enumerate(self.manifest.classes)
        }

    def _class_colormap(self):
        from napari.utils.colormaps import DirectLabelColormap

        return DirectLabelColormap(color_dict=self._class_colors())

    def refresh_classification_layers(self) -> None:
        if self.current_mask is None or self.manifest is None:
            return
        codes = self._class_code_map()
        roi_labels = self.labels.loc[
            self.labels["ROI"].astype(str).eq(self.current_roi)
        ]
        for state in ("confirmed", "proposed"):
            rows = roi_labels.loc[roi_labels["state"].eq(state)]
            mapping = pd.Series(
                rows["class_id"].map(codes).to_numpy(),
                index=rows["ObjectNumber"].astype(int),
            )
            data = _identity_value_map(self.current_mask, mapping, dtype=np.int32)
            self._replace_layer(
                CLASS_LAYER_NAMES[state],
                data,
                "labels",
                colormap=self._class_colormap(),
            )
        if not self.scores.empty:
            rows = self.scores.loc[
                self.scores["ROI"].astype(str).eq(self.current_roi)
            ]
            mapping = pd.Series(
                rows["predicted_class"].map(codes).to_numpy(),
                index=rows["ObjectNumber"].astype(int),
            )
            data = _identity_value_map(self.current_mask, mapping, dtype=np.int32)
            self._replace_layer(
                CLASS_LAYER_NAMES["predicted"],
                data,
                "labels",
                colormap=self._class_colormap(),
                visible=self.explore_recipe.layer_visibility.get(
                    CLASS_LAYER_NAMES["predicted"],
                    MANAGED_LAYER_DEFAULT_VISIBILITY[
                        CLASS_LAYER_NAMES["predicted"]
                    ],
                ),
            )
            uncertainty = pd.Series(
                pd.to_numeric(rows["normalized_entropy"], errors="coerce").to_numpy(),
                index=rows["ObjectNumber"].astype(int),
            )
            self._replace_layer(
                CLASS_LAYER_NAMES["uncertainty"],
                _identity_value_map(
                    self.current_mask,
                    uncertainty,
                    background_value=np.nan,
                ),
                "image",
                colormap="magma",
                contrast_limits=(0, 1),
            )
        self._apply_managed_layer_display_settings()

    def _load_feature_table(self) -> pd.DataFrame:
        if not self.paths.feature_table.exists():
            raise FileNotFoundError(
                "No canonical feature table exists. Build or resume features first."
            )
        return read_dataframe(self.paths.feature_table)

    def train_model(self) -> bool:
        if not self.manifest.active_feature_set_id:
            raise ValueError(
                "Build features for the current experiment revision before training."
            )
        if self.model_combo.currentData() == "hist_gradient_boosting":
            confirmed = self.labels.loc[self.labels["state"].eq("confirmed")]
            class_counts = {
                definition.name: int(
                    confirmed["class_id"].eq(definition.class_id).sum()
                )
                for definition in self.manifest.classes
            }
            below_target = {
                name: count
                for name, count in class_counts.items()
                if count < HGB_MIN_SAMPLES_LEAF
            }
            if below_target:
                count_text = ", ".join(
                    f"{name}: {count}/{HGB_MIN_SAMPLES_LEAF}"
                    for name, count in below_target.items()
                )
                reply = self.QMessageBox.question(
                    self.root,
                    "Not enough confirmed labels for HistGradientBoosting",
                    "HistGradientBoosting uses 20 samples per leaf. With too few "
                    "confirmed examples it may predict identical 0.5 probabilities "
                    "for every cell.\n\n"
                    f"Below target: {count_text}.\n\n"
                    "Confirm more cells, or use Random Forest during early active "
                    "learning. Train HistGradientBoosting anyway?",
                )
                if reply != self.QMessageBox.Yes:
                    self.set_status(
                        "HistGradientBoosting training cancelled: more confirmed "
                        "labels are needed."
                    )
                    return False
        result = train_multiclass_classifier(
            self._load_feature_table(),
            self.labels,
            class_ids=[item.class_id for item in self.manifest.classes],
            feature_columns=(self.manifest.active_model_features or None),
            cohort=self.cohort,
            model_type=self.model_combo.currentData(),
            cohort_fingerprint=self.manifest.cell_scope.snapshot_sha256,
            feature_set_id=self.manifest.active_feature_set_id,
        )
        if not result.ok:
            raise ValueError("; ".join(result.errors))
        self.model_bundle = result.bundle
        if (
            self.manifest.active_model_features
            and self.manifest.active_model_features
            != self.model_bundle.feature_columns
        ):
            self.manifest.active_model_features = list(
                self.model_bundle.feature_columns
            )
            save_experiment(
                self.manifest,
                self.paths.root,
                audit_action="drop_unusable_active_model_features",
            )
        save_model_bundle(
            self.model_bundle, self.paths.models / "classifier_latest.joblib"
        )
        self._refresh_model_storage_label()
        for warning in result.warnings:
            self.set_status(f"MODEL WARNING — {warning}")
        self.set_status(
            f"Trained {self.model_bundle.metadata['model_type']} on "
            f"{len(result.training_table)} confirmed cohort cells."
        )
        return True

    def score_model(self) -> None:
        if self.model_bundle is None:
            latest = self.paths.models / "classifier_latest.joblib"
            if not latest.exists():
                if not self.train_model():
                    return
            else:
                from .classifier import load_model_bundle

                self.model_bundle = load_model_bundle(latest)
        metadata = self.model_bundle.metadata
        feature_selection_stale = bool(
            self.manifest.active_model_features
            and metadata.get("feature_set_hash")
            != feature_set_hash(self.manifest.active_model_features)
        )
        if (
            metadata.get("cohort_fingerprint")
            != self.manifest.cell_scope.snapshot_sha256
            or metadata.get("feature_set_id")
            != self.manifest.active_feature_set_id
            or metadata.get("labels_fingerprint")
            != confirmed_labels_fingerprint(self.labels)
            or feature_selection_stale
        ):
            raise ValueError(
                "The loaded model is stale for the current cohort, feature revision, "
                "or confirmed labels. Retrain before scoring."
            )
        self.scores = score_cohort(self.model_bundle, self._load_feature_table())
        write_dataframe(self.paths.scores, self.scores)
        append_audit(
            self.paths,
            {
                "action": "score_cohort",
                "model_id": self.model_bundle.model_id,
                "scored_cells": int(self.scores["scorable"].sum()),
            },
        )
        self.refresh_classification_layers()
        self.refresh_uncertainty_queue()
        self.refresh_status()
        self.set_status(
            f"Scored {int(self.scores['scorable'].sum()):,}/"
            f"{len(self.scores):,} eligible feature rows."
        )

    def refresh_uncertainty_queue(self) -> None:
        if self.scores.empty:
            self.queue_result_label.setText(
                "No scored cells are available. Train and score the cohort first."
            )
            raise ValueError("Score the cohort before building an uncertainty queue.")
        selected_roi = self.queue_roi_combo.currentData()
        selected_class = self.queue_class_combo.currentData()
        review = self.queue_review_combo.currentText().lower()
        if review == "unlabelled":
            queue = uncertainty_queue(
                self.scores,
                self.labels,
                limit=max(len(self.scores), 1),
                roi=selected_roi,
                predicted_class=selected_class,
            )
        else:
            label_state = self.labels.loc[
                :, ["ROI", "ObjectNumber", "state"]
            ].copy()
            queue = self.scores.merge(
                label_state, on=["ROI", "ObjectNumber"], how="left"
            )
            if review in {"proposed", "confirmed"}:
                queue = queue.loc[queue["state"].eq(review)]
            if selected_roi is not None:
                queue = queue.loc[queue["ROI"].astype(str).eq(str(selected_roi))]
            if selected_class is not None:
                queue = queue.loc[queue["predicted_class"].eq(selected_class)]
            queue = queue.sort_values(
                ["normalized_entropy", "probability_margin"],
                ascending=[False, True],
            )
        queue = queue.loc[
            pd.to_numeric(queue["maximum_probability"], errors="coerce").ge(
                self.queue_confidence_spin.value()
            )
        ]
        matched_count = len(queue)
        queue = queue.head(250)
        self.queue_list.clear()
        for row in queue.itertuples():
            from qtpy.QtWidgets import QListWidgetItem

            item = QListWidgetItem(
                f"{row.ROI} / {row.ObjectNumber} — {row.predicted_class}, "
                f"entropy {row.normalized_entropy:.3f}"
            )
            class_definition = self._class_definition(str(row.predicted_class))
            if class_definition is not None:
                item.setIcon(self._class_icon(class_definition.color))
            item.setData(self.Qt.UserRole, (str(row.ROI), int(row.ObjectNumber)))
            self.queue_list.addItem(item)
        self.queue_result_label.setText(
            f"Showing {len(queue):,} of {matched_count:,} matching cells "
            f"({review.lower()}, confidence ≥ "
            f"{self.queue_confidence_spin.value():.2f}). Filter changes apply "
            "automatically; the button forces a refresh."
        )

    def _refresh_queue_if_scored(self) -> None:
        if self._updating_queue_controls or self.scores.empty:
            return
        self.refresh_uncertainty_queue()

    def navigate_queue_item(self, item) -> None:
        roi, object_number = item.data(self.Qt.UserRole)
        self.roi_combo.setCurrentText(roi)
        self.load_roi(roi)
        self._select_cohort_object(int(object_number))

    def bulk_propose(self) -> None:
        queue = high_confidence_queue(
            self.scores,
            self.labels,
            class_id=self.selected_class_id(),
            threshold=self.confidence_spin.value(),
        )
        if queue.empty:
            self.set_status("No unlabelled cells meet the bulk-proposal filter.")
            return
        reply = self.QMessageBox.question(
            self.root,
            "Bulk proposal",
            f"Create {len(queue):,} proposed labels? They will not be final exports.",
        )
        if reply != self.QMessageBox.Yes:
            return
        for row in queue.itertuples():
            self.labels = set_label(
                self.labels,
                roi=row.ROI,
                object_number=row.ObjectNumber,
                class_id=self.selected_class_id(),
                state="proposed",
                source=f"model:{row.model_id}",
            )
        write_dataframe(self.paths.labels, self.labels)
        append_audit(
            self.paths,
            {
                "action": "bulk_propose",
                "class_id": self.selected_class_id(),
                "count": len(queue),
                "threshold": self.confidence_spin.value(),
            },
        )
        self.refresh_classification_layers()
        self.refresh_status()

    def show_selected_probability(self) -> None:
        class_id = str(self.probability_class_combo.currentData())
        column = f"probability::{class_id}"
        rows = self.scores.loc[self.scores["ROI"].astype(str).eq(self.current_roi)]
        mapping = pd.Series(
            pd.to_numeric(rows[column], errors="coerce").to_numpy(),
            index=rows["ObjectNumber"].astype(int),
        )
        self._replace_layer(
            CLASS_LAYER_NAMES["uncertainty"],
            _identity_value_map(
                self.current_mask,
                mapping,
                background_value=np.nan,
            ),
            "image",
            colormap="viridis",
            contrast_limits=(0, 1),
        )
        self.set_status(f"Showing cohort-only probability for class {class_id!r}.")

    def start_source_validation(self) -> None:
        if self.manifest is None:
            raise RuntimeError("Create or load an experiment before validating sources.")
        if self.feature_process is not None:
            raise RuntimeError("Wait for the feature build to finish first.")
        if self.refinement_process is not None:
            raise RuntimeError("Wait for feature refinement to finish first.")
        if self.source_validation_process is not None:
            raise RuntimeError("Feature-source validation is already running.")
        self.manifest.feature_sources = self.feature_sources()
        save_experiment(
            self.manifest,
            self.paths.root,
            audit_action="update_feature_sources",
        )
        self.source_validation_table.setRowCount(0)
        self._source_validation_output_buffer = ""

        from qtpy.QtCore import QProcess

        process = QProcess(self.root)
        process.setProgram(sys.executable)
        process.setArguments(
            [
                "-m",
                "SpatialBiologyToolkit.napari_sbt.worker",
                "validate-sources",
                "--experiment",
                str(self.paths.root),
            ]
        )
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(
            self._read_source_validation_progress
        )
        process.finished.connect(self._source_validation_finished)
        self.source_validation_process = process
        self.validate_sources_button.setEnabled(False)
        process.start()
        self.set_status(
            "Validating imported feature sources against the frozen cohort in "
            "a separate Python process."
        )

    def _read_source_validation_progress(self, *, flush: bool = False) -> None:
        if self.source_validation_process is not None:
            self._source_validation_output_buffer += bytes(
                self.source_validation_process.readAllStandardOutput()
            ).decode(errors="replace")
        lines = self._source_validation_output_buffer.splitlines(keepends=True)
        self._source_validation_output_buffer = ""
        for raw_line in lines:
            if not flush and not raw_line.endswith(("\n", "\r")):
                self._source_validation_output_buffer = raw_line
                continue
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self.set_status(f"SOURCE VALIDATION — {line}")
                continue
            event_name = event.get("event", "source_validation")
            if event_name in {"source_valid", "source_invalid"}:
                self._add_source_validation_row(event)
                self.set_status(
                    f"Source {event.get('source_id')!r}: "
                    f"{event.get('status')} — "
                    f"{event.get('covered_cells', 0):,}/"
                    f"{event.get('eligible_cells', 0):,} cohort cells, "
                    f"{event.get('feature_count', 0):,} features."
                )
            elif event_name == "source_validation_running":
                self.set_status(
                    f"Checking source {event.get('source_index')}/"
                    f"{event.get('source_count')}: "
                    f"{event.get('source_id')!r}."
                )
            elif event_name == "source_validation_completed":
                self.set_status(
                    f"Source validation complete: "
                    f"{event.get('valid_sources', 0)} valid, "
                    f"{event.get('invalid_sources', 0)} invalid. "
                    f"Report: {event.get('report', '')}"
                )
            elif event_name == "source_validation_failed":
                self.set_status(
                    f"SOURCE VALIDATION FAILED — {event.get('error', '')}"
                )

    def _add_source_validation_row(self, event: dict) -> None:
        row = self.source_validation_table.rowCount()
        self.source_validation_table.insertRow(row)
        eligible = int(event.get("eligible_cells", 0) or 0)
        covered = int(event.get("covered_cells", 0) or 0)
        values = [
            str(event.get("source_id", "")),
            str(event.get("kind", "")),
            str(event.get("status", "")),
            f"{covered:,} / {eligible:,}",
            f"{int(event.get('missing_cells', 0) or 0):,}",
            f"{int(event.get('feature_count', 0) or 0):,}",
            str(
                event.get("error", "")
                or "Identity join and feature matrix are usable."
            ),
        ]
        for column, value in enumerate(values):
            item = self.QTableWidgetItem(value)
            if event.get("status") == "invalid":
                item.setBackground(self.QColor("#fee2e2"))
            elif event.get("status") == "valid":
                item.setBackground(self.QColor("#dcfce7"))
            self.source_validation_table.setItem(row, column, item)

    def _source_validation_finished(self, exit_code: int, _status) -> None:
        self._read_source_validation_progress(flush=True)
        self.source_validation_process = None
        self.validate_sources_button.setEnabled(True)
        if exit_code == 0:
            self.set_status(
                "Imported-source validation finished. Review coverage and missing "
                "cells in the Feature Building table."
            )
        else:
            self.set_status(f"Feature-source validation exited with code {exit_code}.")

    def refresh_refinement_readiness(self) -> None:
        """Summarize whether confirmed trial labels support grouped evaluation."""

        self.refinement_class_table.setRowCount(0)
        trial = self.manifest.feature_trial if self.manifest is not None else None
        is_trial = bool(
            self.manifest is not None
            and self.manifest.experiment_mode == "feature_discovery_trial"
            and trial is not None
        )
        self.run_refinement_button.setEnabled(False)
        self.apply_model_features_button.setEnabled(False)
        self.promote_trial_button.setEnabled(False)
        if not is_trial:
            if self.manifest is not None and trial is not None:
                self.refinement_scope_label.setText(
                    "This full experiment was promoted from a feature trial. Its "
                    "saved ranking remains available below as provenance."
                )
            else:
                self.refinement_scope_label.setText(
                    "Feature refinement requires a Feature Discovery Trial created "
                    "in Setup."
                )
            return
        trial_rois = set(trial.selected_rois)
        trial_cells = int(self.cohort["ROI"].astype(str).isin(trial_rois).sum())
        feature_rows = 0
        has_features = bool(
            self.paths is not None and self.paths.feature_table.is_file()
        )
        if has_features:
            feature_rows = len(read_dataframe(self.paths.feature_table))
        self.refinement_scope_label.setText(
            f"{trial_cells:,} eligible trial cells in {len(trial_rois)} ROIs; "
            f"{feature_rows:,} feature rows currently available. Confirm each class "
            "in at least two ROIs; 20–30 cells per class is a useful practical target."
        )
        confirmed = self.labels.loc[
            self.labels["state"].astype(str).eq("confirmed")
            & self.labels["ROI"].astype(str).isin(trial_rois)
        ]
        class_coverage_ready = True
        for definition in self.manifest.classes:
            rows = confirmed.loc[confirmed["class_id"].eq(definition.class_id)]
            count = len(rows)
            roi_count = int(rows["ROI"].astype(str).nunique())
            if count >= 20 and roi_count >= 2:
                readiness = "Good initial coverage"
                colour = "#dcfce7"
            elif count >= 2 and roi_count >= 2:
                readiness = "Runnable; add labels"
                colour = "#fef3c7"
            else:
                readiness = "Needs ≥2 cells in ≥2 ROIs"
                colour = "#fee2e2"
                class_coverage_ready = False
            row = self.refinement_class_table.rowCount()
            self.refinement_class_table.insertRow(row)
            for column, value in enumerate(
                (definition.name, f"{count:,}", str(roi_count), readiness)
            ):
                item = self.QTableWidgetItem(value)
                item.setBackground(self.QColor(colour))
                self.refinement_class_table.setItem(row, column, item)
        current_results = False
        if self.paths is not None and self.paths.refinement_summary.is_file():
            try:
                summary = json.loads(
                    self.paths.refinement_summary.read_text(encoding="utf-8")
                )
                current_results = bool(
                    self.manifest.active_feature_set_id
                    and summary.get("feature_set_id")
                    == self.manifest.active_feature_set_id
                )
            except (OSError, json.JSONDecodeError):
                current_results = False
        self.run_refinement_button.setEnabled(
            has_features
            and class_coverage_ready
            and self.refinement_process is None
        )
        self.apply_model_features_button.setEnabled(current_results)
        self.promote_trial_button.setEnabled(current_results)
        if not has_features:
            self.refinement_scope_label.setText(
                self.refinement_scope_label.text()
                + " Build trial features before refinement."
            )
        elif not class_coverage_ready:
            self.refinement_scope_label.setText(
                self.refinement_scope_label.text()
                + " Add confirmed labels where the table shows red rows."
            )

    def start_feature_refinement(self) -> None:
        if self.manifest is None or self.paths is None:
            raise RuntimeError("Create or load a feature-discovery trial first.")
        if self.manifest.experiment_mode != "feature_discovery_trial":
            raise ValueError("Switch to a Feature Discovery Trial in Setup first.")
        if self.refinement_process is not None:
            raise RuntimeError("Feature refinement is already running.")
        if self.feature_process is not None or self.source_validation_process is not None:
            raise RuntimeError("Wait for the current feature process to finish.")
        from qtpy.QtCore import QProcess

        process = QProcess(self.root)
        process.setProgram(sys.executable)
        process.setArguments(
            [
                "-m",
                "SpatialBiologyToolkit.napari_sbt.worker",
                "refine",
                "--experiment",
                str(self.paths.root),
                "--maximum-candidate-features",
                str(self.refine_candidate_spin.value()),
                "--recommendation-count",
                str(self.refine_recommendation_spin.value()),
                "--permutation-repeats",
                str(self.refine_repeats_spin.value()),
                "--maximum-missing-fraction",
                str(self.refine_missing_spin.value()),
                "--correlation-threshold",
                str(self.refine_correlation_spin.value()),
            ]
        )
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_refinement_progress)
        process.finished.connect(self._feature_refinement_finished)
        process.started.connect(
            lambda: self.refinement_progress_label.setText(
                f"Refinement process: live (PID {int(process.processId())}); "
                "waiting for the first ROI-fold update"
            )
        )
        self.refinement_process = process
        self.refinement_cancel_requested = False
        self._refinement_output_buffer = ""
        self.refinement_log.clear()
        self.refinement_progress_bar.setRange(0, 0)
        self.refinement_progress_bar.setFormat("Starting grouped evaluation…")
        self.refinement_progress_label.setText("Refinement process: starting")
        self.run_refinement_button.setEnabled(False)
        self.cancel_refinement_button.setEnabled(True)
        process.start()
        self.set_status(
            "Started leave-one-ROI-out feature refinement in a subprocess."
        )

    def _read_refinement_progress(self, *, flush: bool = False) -> None:
        if self.refinement_process is not None:
            self._refinement_output_buffer += bytes(
                self.refinement_process.readAllStandardOutput()
            ).decode(errors="replace")
        lines = self._refinement_output_buffer.splitlines(keepends=True)
        self._refinement_output_buffer = ""
        for raw_line in lines:
            if not flush and not raw_line.endswith(("\n", "\r")):
                self._refinement_output_buffer = raw_line
                continue
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self.refinement_log.append(line)
                continue
            name = str(event.get("event", "progress"))
            if name == "refinement_started":
                self.refinement_progress_label.setText(
                    "Refinement process: screening features and preparing ROI folds"
                )
            elif name in {
                "refinement_fold_started",
                "refinement_model_completed",
            }:
                completed = int(event.get("completed_fold_models", 0) or 0)
                total = int(event.get("total_fold_models", 0) or 0)
                self.refinement_progress_bar.setRange(0, max(total, 1))
                self.refinement_progress_bar.setValue(completed)
                self.refinement_progress_bar.setFormat(
                    f"{completed}/{total} held-out ROI/model evaluations"
                )
                self.refinement_progress_label.setText(
                    "Refinement process: "
                    f"held out {event.get('held_out_roi', '—')}; "
                    f"{event.get('model', 'screening')}"
                )
            elif name == "refinement_completed":
                self.refinement_progress_bar.setRange(0, 1)
                self.refinement_progress_bar.setValue(1)
                self.refinement_progress_bar.setFormat("Complete")
                self.refinement_progress_label.setText(
                    "Refinement complete: "
                    f"{int(event.get('recommended_feature_count', 0))} features "
                    "recommended"
                )
            elif name == "refinement_failed":
                self.refinement_progress_label.setText(
                    f"Refinement failed: {event.get('error', '')}"
                )
            self.refinement_log.append(
                "; ".join(
                    str(value)
                    for value in (
                        name.replace("_", " "),
                        event.get("held_out_roi"),
                        event.get("model"),
                        event.get("error"),
                    )
                    if value not in (None, "")
                )
            )

    def _feature_refinement_finished(self, exit_code: int, _status) -> None:
        self._read_refinement_progress(flush=True)
        self.refinement_process = None
        self.run_refinement_button.setEnabled(True)
        self.cancel_refinement_button.setEnabled(False)
        if self.refinement_cancel_requested:
            self.refinement_progress_bar.setRange(0, 100)
            self.refinement_progress_bar.setValue(0)
            self.refinement_progress_bar.setFormat("Cancelled")
            self.refinement_progress_label.setText("Refinement process: cancelled")
            self.set_status(
                "Feature refinement was cancelled; previously saved results were "
                "not replaced."
            )
        elif exit_code == 0:
            self.manifest, self.paths = load_experiment(self.paths.root)
            self.load_refinement_results()
            self.set_status("Feature refinement completed and results were loaded.")
        else:
            self.refinement_progress_bar.setRange(0, 100)
            self.refinement_progress_bar.setValue(0)
            self.refinement_progress_bar.setFormat("Failed")
            self.set_status(f"Feature refinement exited with code {exit_code}.")
        self.refresh_refinement_readiness()

    def cancel_feature_refinement(self) -> None:
        if self.refinement_process is None:
            return
        self.refinement_cancel_requested = True
        self.refinement_process.terminate()
        self.cancel_refinement_button.setEnabled(False)
        self.refinement_progress_label.setText(
            "Refinement process: cancellation requested"
        )
        self.set_status(
            "Feature refinement cancellation requested. Previously saved results "
            "remain unchanged."
        )

    def load_refinement_results(self, *, silent: bool = False) -> None:
        self.refinement_results_table.setRowCount(0)
        if self.paths is None or not self.paths.refinement_summary.is_file():
            self.refinement_metrics_label.setText("No refinement results yet.")
            if not silent:
                self.set_status("No saved feature-refinement report is available.")
            return
        summary = json.loads(
            self.paths.refinement_summary.read_text(encoding="utf-8")
        )
        ranking = read_dataframe(self.paths.feature_ranking)
        stale = summary.get("feature_set_id") != self.manifest.active_feature_set_id
        warning_text = " — STALE for current features" if stale else ""
        family_text = ", ".join(
            f"{row.get('source')}/{row.get('family')}"
            for row in summary.get("family_importance", [])[:3]
        )
        self.refinement_metrics_label.setText(
            f"Grouped validation: balanced accuracy "
            f"{float(summary.get('mean_balanced_accuracy', 0)):.3f}; macro-F1 "
            f"{float(summary.get('mean_macro_f1', 0)):.3f}; "
            f"{int(summary.get('recommended_feature_count', 0))} recommended "
            f"features{warning_text}. Leading source/families: "
            f"{family_text or 'not available'}. Rankings are exploratory estimates, not an "
            "independent final validation."
        )
        recommended = set(summary.get("recommended_features", []))
        checked = set(self.manifest.active_model_features) or recommended
        display = ranking.head(500).copy()
        missing_recommended = ranking.loc[
            ranking["feature"].isin(recommended - set(display["feature"]))
        ]
        if not missing_recommended.empty:
            display = pd.concat([display, missing_recommended], ignore_index=True)
        self.refinement_results_table.setRowCount(len(display))
        for row_index, row in enumerate(display.itertuples(index=False)):
            values = (
                str(int(row.rank)),
                str(row.feature),
                str(row.source),
                str(row.family),
                f"{float(row.mean_permutation_importance):.4f}",
                f"{float(row.positive_importance_frequency):.0%}",
                f"{float(row.missing_fraction):.1%}",
            )
            for column, value in enumerate(values):
                item = self.QTableWidgetItem(value)
                if column == 1:
                    item.setData(self.Qt.UserRole, str(row.feature))
                    item.setToolTip(
                        str(getattr(row, "redundant_with", "") or "")
                    )
                self.refinement_results_table.setItem(row_index, column, item)
            use_item = self.QTableWidgetItem("Include")
            use_item.setFlags(use_item.flags() | self.Qt.ItemIsUserCheckable)
            use_item.setCheckState(
                self.Qt.Checked
                if str(row.feature) in checked
                else self.Qt.Unchecked
            )
            self.refinement_results_table.setItem(row_index, 7, use_item)
        if not silent:
            self.set_status(
                f"Loaded {len(ranking):,} ranked features; showing "
                f"{len(display):,}."
            )

    def checked_refinement_features(self) -> list[str]:
        selected = []
        for row in range(self.refinement_results_table.rowCount()):
            use_item = self.refinement_results_table.item(row, 7)
            feature_item = self.refinement_results_table.item(row, 1)
            if (
                use_item is not None
                and feature_item is not None
                and use_item.checkState() == self.Qt.Checked
            ):
                selected.append(str(feature_item.data(self.Qt.UserRole)))
        return selected

    def _current_refinement_summary(self) -> dict:
        if self.paths is None or not self.paths.refinement_summary.is_file():
            raise FileNotFoundError("Run feature refinement first.")
        summary = json.loads(
            self.paths.refinement_summary.read_text(encoding="utf-8")
        )
        if summary.get("feature_set_id") != self.manifest.active_feature_set_id:
            raise ValueError(
                "The saved refinement is stale for the active feature build. Run "
                "feature refinement again before applying or promoting it."
            )
        return summary

    def restore_recommended_feature_checks(self) -> None:
        summary = self._current_refinement_summary()
        recommended = set(summary.get("recommended_features", []))
        for row in range(self.refinement_results_table.rowCount()):
            feature_item = self.refinement_results_table.item(row, 1)
            use_item = self.refinement_results_table.item(row, 7)
            if feature_item is not None and use_item is not None:
                use_item.setCheckState(
                    self.Qt.Checked
                    if str(feature_item.data(self.Qt.UserRole)) in recommended
                    else self.Qt.Unchecked
                )
        self.set_status("Restored the saved compact feature recommendation.")

    def apply_checked_model_features(self) -> None:
        if self.manifest is None:
            raise RuntimeError("Create or load an experiment first.")
        self._current_refinement_summary()
        selected = self.checked_refinement_features()
        if not selected:
            raise ValueError("Check at least one recommended model feature.")
        self.manifest.active_model_features = selected
        save_experiment(
            self.manifest,
            self.paths.root,
            audit_action="select_refined_model_features",
        )
        self.model_bundle = None
        self._refresh_model_storage_label()
        self.set_status(
            f"The classifier will use {len(selected)} checked features. Retrain "
            "before scoring."
        )
        self.refresh_status()

    def promote_feature_trial(self) -> None:
        if (
            self.manifest is None
            or self.manifest.experiment_mode != "feature_discovery_trial"
            or self.manifest.feature_trial is None
        ):
            raise ValueError("Only an active Feature Discovery Trial can be promoted.")
        self._current_refinement_summary()
        selected = self.checked_refinement_features()
        if not selected:
            raise ValueError("Check at least one feature before promotion.")
        reply = self.QMessageBox.question(
            self.root,
            "Promote feature trial",
            "Create the next experiment revision for the complete frozen cohort "
            f"using {len(selected)} model features? A new full feature build is "
            "required before training or scoring the promoted experiment.",
        )
        if reply != self.QMessageBox.Yes:
            return
        promoted = self.manifest.model_copy(deep=True)
        promoted.revision += 1
        promoted.experiment_mode = "full"
        promoted.feature_trial.status = "promoted"
        promoted.feature_trial.recommended_model_features = selected
        promoted.active_model_features = selected
        promoted.synthetic_features = compact_synthetic_recipe(
            promoted.synthetic_features,
            selected,
        )
        for source in promoted.feature_sources:
            prefix = f"source::{source.source_id}::"
            selected_columns = [
                feature.removeprefix(prefix)
                for feature in selected
                if feature.startswith(prefix)
            ]
            source.enabled = bool(selected_columns)
            source.selected_columns = selected_columns
        promoted.active_feature_set_id = None
        save_experiment(
            promoted,
            self.paths.root,
            audit_action="promote_feature_trial",
        )
        self.load_existing_experiment(self.paths.root)
        self.set_status(
            "Promoted the trial to a full-cohort experiment revision. Build full "
            "features before retraining."
        )

    def start_feature_build(self) -> None:
        if self.manifest is None:
            raise RuntimeError("Create or load an experiment before feature extraction.")
        if self.feature_process is not None:
            raise RuntimeError("A feature build is already running.")
        if self.source_validation_process is not None:
            raise RuntimeError("Wait for feature-source validation to finish first.")
        if self.refinement_process is not None:
            raise RuntimeError("Wait for feature refinement to finish first.")
        self.manifest.feature_sources = self.feature_sources()
        self.manifest.synthetic_features = self.synthetic_recipe_from_controls()
        save_experiment(
            self.manifest,
            self.paths.root,
            audit_action="update_feature_recipe",
        )
        from qtpy.QtCore import QProcess

        process = QProcess(self.root)
        process.setProgram(sys.executable)
        process.setArguments(
            [
                "-m",
                "SpatialBiologyToolkit.napari_sbt.worker",
                "features",
                "--experiment",
                str(self.paths.root),
                "--workers",
                str(self.workers_spin.value()),
            ]
        )
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_feature_progress)
        process.finished.connect(self._feature_build_finished)
        self.feature_process = process
        self.feature_build_started_at = time.monotonic()
        self.feature_last_event_at = self.feature_build_started_at
        self.feature_progress_state = {
            "phase": "Starting feature worker",
            "total_rois": 0,
            "completed_rois": 0,
            "failed_rois": 0,
            "pending_rois": 0,
            "workers": self.workers_spin.value(),
            "recent": "Waiting for worker startup",
        }
        self._feature_output_buffer = ""
        self.feature_progress_log.clear()
        self.feature_progress_bar.setRange(0, 0)
        self.feature_progress_bar.setFormat("Starting worker…")
        self._refresh_feature_progress_widgets()
        self.build_features_button.setEnabled(False)
        self.validate_sources_button.setEnabled(False)
        self.cancel_features_button.setEnabled(True)
        process.start()
        self.feature_health_timer.start()
        self.set_status("Started cohort-first feature build in a subprocess.")

    def _read_feature_progress(self, *, flush: bool = False) -> None:
        if self.feature_process is not None:
            self._feature_output_buffer += bytes(
                self.feature_process.readAllStandardOutput()
            ).decode(errors="replace")
        lines = self._feature_output_buffer.splitlines(keepends=True)
        self._feature_output_buffer = ""
        for raw_line in lines:
            if not flush and not raw_line.endswith(("\n", "\r")):
                self._feature_output_buffer = raw_line
                continue
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self.feature_progress_log.append(line)
                self.set_status(f"FEATURE WORKER — {line}")
                continue
            self._handle_feature_progress_event(event)

    def _handle_feature_progress_event(self, event: dict) -> None:
        self.feature_last_event_at = time.monotonic()
        state = self.feature_progress_state
        name = str(event.get("event", "progress"))
        if name == "build_started":
            total = int(event.get("represented_rois", 0) or 0)
            pending = int(event.get("pending_rois", 0) or 0)
            resumed = int(event.get("resumed_rois", 0) or 0)
            state.update(
                {
                    "phase": "Extracting ROI features",
                    "total_rois": total,
                    "completed_rois": resumed,
                    "failed_rois": max(0, total - pending - resumed),
                    "pending_rois": pending,
                    "workers": int(event.get("workers", 0) or 0),
                    "eligible_cells": int(event.get("eligible_cells", 0) or 0),
                    "target_eligible_cells": int(
                        event.get("target_eligible_cells", 0) or 0
                    ),
                    "target_represented_rois": int(
                        event.get("target_represented_rois", total) or total
                    ),
                    "recent": (
                        f"Started {pending} ROI task(s); reused "
                        f"{resumed} valid fragment(s)"
                    ),
                }
            )
        elif name in {"heartbeat", "roi_completed", "roi_failed"}:
            for key in (
                "completed_rois",
                "failed_rois",
                "pending_rois",
                "total_rois",
            ):
                if key in event:
                    state[key] = int(event[key] or 0)
            if name == "heartbeat":
                state["phase"] = (
                    "Combining fragments and imported sources"
                    if int(state.get("pending_rois", 0)) == 0
                    else "Extracting ROI features"
                )
                state["recent"] = (
                    f"Worker heartbeat; "
                    f"{event.get('running_workers', 0)} ROI worker(s) active"
                )
                state["orchestrator_pid"] = int(
                    event.get("orchestrator_pid", 0) or 0
                )
            elif name == "roi_completed":
                state["recent"] = (
                    f"Completed {event.get('roi')} in "
                    f"{float(event.get('elapsed_seconds', 0) or 0):.1f}s "
                    f"({int(event.get('rows', 0) or 0):,} cells; "
                    f"PID {event.get('worker_pid', '?')})"
                )
                if int(state.get("pending_rois", 0)) == 0:
                    state["phase"] = "Combining fragments and imported sources"
            else:
                state["recent"] = (
                    f"Failed {event.get('roi')}: {event.get('error', '')}"
                )
        elif name == "roi_resumed":
            state["recent"] = f"Reusing valid fragment for {event.get('roi')}"
        elif name == "build_completed":
            total = int(event.get("represented_rois", 0) or 0)
            state.update(
                {
                    "phase": "Feature build complete",
                    "total_rois": total,
                    "completed_rois": int(
                        event.get("completed_rois", total) or 0
                    ),
                    "failed_rois": int(event.get("failures", 0) or 0),
                    "pending_rois": 0,
                    "recent": (
                        f"Wrote {int(event.get('feature_count', 0) or 0):,} "
                        f"features for "
                        f"{int(event.get('eligible_cells', 0) or 0):,}/"
                        f"{int(event.get('target_eligible_cells', 0) or 0):,} "
                        "trial/target cells"
                    ),
                }
            )
        elif name == "cancelled":
            state["phase"] = "Cancellation acknowledged"
            state["recent"] = str(event.get("message", "Fragments were preserved."))
        elif name == "build_failed":
            state["phase"] = "Feature build failed"
            state["recent"] = str(event.get("error", "Unknown worker error"))
        else:
            state["recent"] = " ".join(
                str(value)
                for value in (name, event.get("roi"), event.get("error"))
                if value not in (None, "")
            )
        self.feature_progress_log.append(
            self._format_feature_progress_event(event)
        )
        self._refresh_feature_progress_widgets()

    @staticmethod
    def _format_feature_progress_event(event: dict) -> str:
        name = str(event.get("event", "progress")).replace("_", " ")
        details = []
        if event.get("roi"):
            details.append(f"ROI {event['roi']}")
        if event.get("rows") is not None:
            details.append(f"{int(event['rows']):,} cells")
        if event.get("elapsed_seconds") is not None:
            details.append(f"{float(event['elapsed_seconds']):.1f}s")
        if event.get("error"):
            details.append(str(event["error"]))
        if event.get("message"):
            details.append(str(event["message"]))
        return f"{name}: " + ("; ".join(details) if details else "received")

    def _refresh_feature_progress_widgets(self) -> None:
        state = self.feature_progress_state
        total = int(state.get("total_rois", 0) or 0)
        complete = int(state.get("completed_rois", 0) or 0)
        failed = int(state.get("failed_rois", 0) or 0)
        pending = int(state.get("pending_rois", 0) or 0)
        if total:
            processed = min(total, complete + failed)
            self.feature_progress_bar.setRange(0, total)
            self.feature_progress_bar.setValue(processed)
            self.feature_progress_bar.setFormat(f"{processed}/{total} ROIs (%p%)")
        elif self.feature_process is None:
            self.feature_progress_bar.setRange(0, 100)
            self.feature_progress_bar.setValue(0)
            phase = str(state.get("phase", "Not started"))
            self.feature_progress_bar.setFormat(
                "Failed" if "failed" in phase.lower() or "exited" in phase.lower()
                else "Not started"
            )
        self.feature_phase_label.setText(
            f"Phase: {state.get('phase', 'Not started')}"
        )
        self.feature_counts_label.setText(
            f"ROIs: {complete} complete, {failed} failed, "
            f"{pending} pending / {total} current-scope total"
            + (
                f" ({int(state.get('target_represented_rois', total))} target ROIs)"
                if int(state.get("target_represented_rois", total) or total) != total
                else ""
            )
        )
        self.feature_current_roi_label.setText(
            f"Latest: {state.get('recent', 'No worker events yet')}"
        )
        self._update_feature_process_health()

    def _update_feature_process_health(self) -> None:
        if self.feature_build_started_at is None:
            self.feature_elapsed_label.setText("Elapsed: —")
            self.feature_process_health_label.setText("Process: not running")
            return
        elapsed = max(0.0, time.monotonic() - self.feature_build_started_at)
        self.feature_elapsed_label.setText(f"Elapsed: {elapsed:.0f}s")
        process = self.feature_process
        if process is None:
            self.feature_process_health_label.setText("Process: finished")
            return
        try:
            process_state = process.state()
            state_value = getattr(process_state, "value", process_state)
            running = int(state_value) != 0
            pid = int(process.processId())
        except (AttributeError, TypeError, ValueError):
            running = True
            pid = 0
        heartbeat_age = (
            max(0.0, time.monotonic() - self.feature_last_event_at)
            if self.feature_last_event_at is not None
            else elapsed
        )
        if not running:
            health = "stopped"
        elif heartbeat_age <= 6:
            health = "live and reporting"
        elif int(self.feature_progress_state.get("pending_rois", 0) or 0) == 0:
            health = "live; finalizing outputs"
        else:
            health = (
                f"live; waiting {heartbeat_age:.0f}s for the next worker heartbeat"
            )
        pid_text = f" PID {pid};" if pid else ""
        self.feature_process_health_label.setText(
            f"Process:{pid_text} {health}"
        )

    def _feature_build_finished(self, exit_code: int, _status) -> None:
        self._read_feature_progress(flush=True)
        self.feature_health_timer.stop()
        self.feature_process = None
        self.build_features_button.setEnabled(True)
        self.validate_sources_button.setEnabled(True)
        self.cancel_features_button.setEnabled(False)
        if exit_code == 0 and self.paths is not None:
            self.manifest, self.paths = load_experiment(self.paths.root)
            self._update_scope_text()
            self.refresh_status()
            self.refresh_refinement_readiness()
            self.load_refinement_results(silent=True)
            self.feature_progress_state["phase"] = "Feature build complete"
            self.feature_progress_state["pending_rois"] = 0
            self.set_status("Feature build completed.")
        else:
            self.feature_progress_state["phase"] = (
                f"Feature build exited with code {exit_code}"
            )
            self.set_status(f"Feature build exited {exit_code}.")
        self._refresh_feature_progress_widgets()

    def cancel_feature_build(self) -> None:
        if self.feature_process is not None:
            (self.paths.logs / "feature_build.cancel").write_text(
                "cancel requested\n", encoding="utf-8"
            )
            self.cancel_features_button.setEnabled(False)
            self.set_status(
                "Cancellation requested. Running ROI workers will finish safely; "
                "valid fragments are preserved and pending work is cancelled."
            )

    def _assignments(self) -> pd.DataFrame:
        return build_assignment_table(
            self.cohort,
            self.labels,
            self.scores,
            class_ids=[item.class_id for item in self.manifest.classes],
        )

    def export_assignments(self) -> None:
        destination = self.paths.exports / "assignments.parquet"
        export_assignment_table(self._assignments(), destination)
        self.set_status(f"Exported cohort-only assignments: {destination}")

    def export_adata(self) -> None:
        if not self.manifest.anndata_path:
            raise ValueError("Annotated AnnData export requires an AnnData source.")
        destination = Path(self.annotated_path_edit.text())
        feature_provenance = (
            json.loads(self.paths.feature_manifest.read_text(encoding="utf-8"))
            if self.paths.feature_manifest.exists()
            else {}
        )
        model_provenance = (
            self.model_bundle.metadata if self.model_bundle is not None else {}
        )
        export_annotated_anndata(
            self.manifest.anndata_path,
            destination,
            self._assignments(),
            self.manifest,
            feature_provenance=feature_provenance,
            model_provenance=model_provenance,
        )
        self.set_status(f"Exported atomic annotated AnnData copy: {destination}")

    def export_cohort_masks(self) -> None:
        masks = discover_mask_files(self.manifest.masks_folder)
        written = materialize_cohort_masks(
            masks, self.cohort, self.paths.cohort_masks
        )
        self.set_status(f"Wrote {len(written)} cohort masks; originals were untouched.")

    def export_cleaned_masks(self) -> None:
        reply = self.QMessageBox.question(
            self.root,
            "Cleaned masks",
            "Write derived masks using confirmed exclusions and model exclusions "
            f"at confidence ≥ {self.confidence_spin.value():.2f}?",
        )
        if reply != self.QMessageBox.Yes:
            return
        written = export_cleaned_masks(
            discover_mask_files(self.manifest.masks_folder),
            self._assignments(),
            self.manifest.classes,
            self.paths.exports / "cleaned_masks",
            prediction_confidence_threshold=self.confidence_spin.value(),
        )
        self.set_status(f"Wrote {len(written)} cleaned derived masks.")

    def create_regions_layer(self) -> None:
        name = "manual_tissue_regions"
        if name not in self.viewer.layers:
            self.viewer.add_shapes(
                name=name,
                shape_type="polygon",
                edge_color="yellow",
                face_color=[1, 1, 0, 0.15],
                opacity=1.0,
            )
        self.viewer.layers.selection.active = self.viewer.layers[name]
        self.set_status("Draw polygons in the manual_tissue_regions shapes layer.")

    def sync_regions(self) -> None:
        from matplotlib.path import Path as PolygonPath
        from skimage.measure import regionprops

        if "manual_tissue_regions" not in self.viewer.layers:
            raise ValueError("Create and draw at least one tissue-region polygon.")
        shapes = self.viewer.layers["manual_tissue_regions"]
        rows = self.cohort.loc[
            self.cohort["ROI"].astype(str).eq(self.current_roi)
        ].copy()
        centroids = {
            int(region.label): (float(region.centroid[1]), float(region.centroid[0]))
            for region in regionprops(self.current_mask)
        }
        rows["X_loc"] = rows["ObjectNumber"].map(
            lambda value: centroids.get(int(value), (np.nan, np.nan))[0]
        )
        rows["Y_loc"] = rows["ObjectNumber"].map(
            lambda value: centroids.get(int(value), (np.nan, np.nan))[1]
        )
        region_rows = []
        for index, polygon in enumerate(shapes.data):
            polygon_yx = np.asarray(polygon)[:, -2:]
            polygon_xy = polygon_yx[:, ::-1]
            contains = PolygonPath(polygon_xy).contains_points(
                rows[["X_loc", "Y_loc"]].to_numpy()
            )
            for identity in rows.loc[contains, ["obs_name", "ROI", "ObjectNumber"]].itertuples(
                index=False
            ):
                region_rows.append(
                    {
                        "obs_name": identity.obs_name,
                        "ROI": identity.ROI,
                        "ObjectNumber": identity.ObjectNumber,
                        "region": f"{self.region_name_edit.text()}_{index + 1}",
                    }
                )
        table = pd.DataFrame(
            region_rows,
            columns=["obs_name", "ROI", "ObjectNumber", "region"],
        )
        destination = self.paths.annotations / f"{self.current_roi}_regions.csv"
        write_dataframe(destination, table)
        write_json(
            self.paths.annotations / f"{self.current_roi}_region_shapes.json",
            {"roi": self.current_roi, "polygons": [np.asarray(value).tolist() for value in shapes.data]},
        )
        self.set_status(f"Synchronised {len(table)} region-to-cell assignments.")

    def apply_colormap(self) -> None:
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, "colormap"):
            raise ValueError("Select an image layer.")
        layer.colormap = self.colormap_combo.currentText()

    def flip_selected_layer(self, axis: int) -> None:
        try:
            layer = self.viewer.layers.selection.active
            if layer is None:
                raise ValueError("Select a layer.")
            layer.data = np.flip(np.asarray(layer.data), axis=axis)
            self.set_status("Flipped selected display layer; source files were untouched.")
        except Exception as error:  # noqa: BLE001 - Qt callback error boundary
            self.set_status(f"ERROR — {type(error).__name__}: {error}")
            self.QMessageBox.critical(
                self.root, "napari_sbt", f"{type(error).__name__}: {error}"
            )

    def transfer_colormap(self) -> None:
        source = self.viewer.layers.selection.active
        if source is None or not hasattr(source, "colormap"):
            raise ValueError("Select a source image layer.")
        for layer in self.viewer.layers:
            if hasattr(layer, "colormap"):
                layer.colormap = source.colormap

    def expand_selected_labels(self) -> None:
        from skimage.segmentation import expand_labels

        layer = self.viewer.layers.selection.active
        if layer is None or layer.__class__.__name__.lower() != "labels":
            raise ValueError("Select a labels layer.")
        expanded = expand_labels(
            np.asarray(layer.data), distance=self.expand_spin.value()
        )
        self.viewer.add_labels(
            expanded,
            name=f"{layer.name}_expanded",
            opacity=1.0,
        )

    def resize_selected_layer(self) -> None:
        from skimage.transform import resize

        layer = self.viewer.layers.selection.active
        if layer is None:
            raise ValueError("Select a layer.")
        data = np.asarray(layer.data)
        factor = self.resize_spin.value()
        spatial_shape = (
            max(1, round(data.shape[0] * factor)),
            max(1, round(data.shape[1] * factor)),
        )
        output_shape = spatial_shape + data.shape[2:]
        is_labels = layer.__class__.__name__.lower() == "labels"
        resized = resize(
            data,
            output_shape,
            order=0 if is_labels else 1,
            preserve_range=True,
            anti_aliasing=not is_labels,
        ).astype(data.dtype)
        if is_labels:
            self.viewer.add_labels(
                resized,
                name=f"{layer.name}_resized",
                opacity=1.0,
            )
        else:
            self.viewer.add_image(
                resized,
                name=f"{layer.name}_resized",
                rgb=bool(data.ndim == 3 and data.shape[-1] in (3, 4)),
                opacity=1.0,
            )

    def mask_selected_image(self) -> None:
        layer = self.viewer.layers.selection.active
        if layer is None or self.current_mask is None:
            raise ValueError("Select an image layer after loading a cohort ROI.")
        eligible = set(
            self.cohort.loc[
                self.cohort["ROI"].astype(str).eq(self.current_roi), "ObjectNumber"
            ].astype(int)
        )
        keep = cohort_mask(self.current_mask, eligible) > 0
        data = np.asarray(layer.data)
        if data.shape[:2] != keep.shape:
            raise ValueError("Selected image and current mask shapes do not match.")
        masked = data * keep[..., None] if data.ndim == 3 else data * keep
        self.viewer.add_image(
            masked,
            name=f"{layer.name}_cohort_masked",
            opacity=1.0,
        )


def launch(
    *,
    viewer=None,
    project_root: str | Path | None = None,
    experiment: str | Path | None = None,
    anndata_path: str | Path | None = None,
    masks_folder: str | Path | None = None,
    images_folders: Iterable[str | Path] = (),
    extra_images_folders: Iterable[str | Path] = (),
):
    """Create the viewer and dock; the caller decides whether to enter napari.run."""

    import napari

    if viewer is None:
        viewer = napari.Viewer(title="napari_sbt — cohort-first cell classification")
    controller = NapariSBTController(
        viewer,
        project_root=project_root,
        experiment=experiment,
        anndata_path=anndata_path,
        masks_folder=masks_folder,
        images_folders=images_folders,
        extra_images_folders=extra_images_folders,
    )
    dock = viewer.window.add_dock_widget(
        controller.root,
        name="napari_sbt",
        area="right",
    )
    return viewer, controller, dock


__all__ = ["NapariSBTController", "launch"]
