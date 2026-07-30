"""Unified Napari dock for cohort-first IMC exploration and classification."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from SpatialBiologyToolkit.pipeline.manifests import write_json
from SpatialBiologyToolkit.qc_classifier.io import (
    discover_mask_files,
    discover_roi_images,
    load_display_image,
    load_mask,
)

from .classifier import (
    confirmed_labels_fingerprint,
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
from .labels import confirm_proposed, empty_labels, set_label, validate_labels
from .models import (
    ClassificationClass,
    ExperimentManifest,
    FeatureSource,
    SyntheticFeatureRecipe,
    segmentation_qc_classes,
    slugify,
)
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


def _path_text(value: str | Path | None) -> str:
    return "" if value is None else str(Path(value))


def _split_paths(value: str) -> list[str]:
    return [item.strip() for item in value.replace(";", "\n").splitlines() if item.strip()]


def _identity_value_map(
    mask: np.ndarray, values: pd.Series, *, dtype=np.float32
) -> np.ndarray:
    output = np.zeros(mask.shape, dtype=dtype)
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
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QColor
        from qtpy.QtWidgets import (
            QAbstractItemView,
            QCheckBox,
            QColorDialog,
            QComboBox,
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
            QScrollArea,
            QSpinBox,
            QTableWidget,
            QTableWidgetItem,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )

        self.Qt = Qt
        self.QMessageBox = QMessageBox
        self.QFileDialog = QFileDialog
        self.QColorDialog = QColorDialog
        self.QColor = QColor
        self.QTableWidgetItem = QTableWidgetItem
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
        self.reviewed_rois: set[str] = set()
        self._class_shortcuts: list[str] = []

        self.root = QWidget()
        root_layout = QVBoxLayout(self.root)
        self.scope_label = QLabel("No experiment: classification is disabled.")
        self.scope_label.setWordWrap(True)
        root_layout.addWidget(self.scope_label)
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        def add_tab(widget, title: str) -> None:
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

        class_group = QGroupBox("2. Mutually exclusive classes (2–8)")
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

        feature_group = QGroupBox("3. Feature sources and synthetic recipe")
        feature_form = QFormLayout(feature_group)
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
        self.channels_edit = QLineEdit()
        self.channels_edit.setPlaceholderText(
            "Comma-separated channel names; blank means every discovered channel"
        )
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
        feature_checks = QWidget()
        feature_checks_layout = QHBoxLayout(feature_checks)
        feature_checks_layout.setContentsMargins(0, 0, 0, 0)
        for widget in (
            self.distribution_check,
            self.region_check,
            self.gradient_check,
            self.shape_check,
            self.context_check,
        ):
            feature_checks_layout.addWidget(widget)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, max(os.cpu_count() or 1, 1))
        self.workers_spin.setValue(min(8, max(os.cpu_count() or 1, 1)))
        feature_form.addRow("Imported tables", self.feature_tables_edit)
        feature_form.addRow("AnnData / CellVision sources", self.anndata_features_edit)
        feature_form.addRow("Channels", self.channels_edit)
        feature_form.addRow("Signed intensity-mask offset (px)", self.offset_spin)
        feature_form.addRow("Positive-offset collisions", self.offset_overlap_check)
        feature_form.addRow("Background ring (px)", self.background_ring_spin)
        feature_form.addRow("Nimbus normalization JSON", self.normalization_edit)
        feature_form.addRow("Feature families", feature_checks)
        feature_form.addRow("Local workers", self.workers_spin)
        setup_layout.addWidget(feature_group)
        setup_actions = QHBoxLayout()
        self.create_button = QPushButton("Create confirmed experiment")
        self.load_experiment_button = QPushButton("Load experiment")
        self.build_features_button = QPushButton("Build/resume features locally")
        self.cancel_features_button = QPushButton("Cancel build")
        self.cancel_features_button.setEnabled(False)
        self.hpc_button = QPushButton("HPC instructions")
        for widget in (
            self.create_button,
            self.load_experiment_button,
            self.build_features_button,
            self.cancel_features_button,
            self.hpc_button,
        ):
            setup_actions.addWidget(widget)
        setup_layout.addLayout(setup_actions)
        add_tab(setup, "Setup")

        # Explore
        explore = QWidget()
        explore_layout = QVBoxLayout(explore)
        roi_row = QHBoxLayout()
        self.roi_combo = QComboBox()
        self.show_empty_rois = QCheckBox("Include ROIs with no eligible cells")
        self.context_check_display = QCheckBox("Show dimmed full-mask context")
        self.reload_roi_button = QPushButton("Load ROI")
        roi_row.addWidget(QLabel("ROI"))
        roi_row.addWidget(self.roi_combo)
        roi_row.addWidget(self.show_empty_rois)
        roi_row.addWidget(self.context_check_display)
        roi_row.addWidget(self.reload_roi_button)
        explore_layout.addLayout(roi_row)
        overlay_group = QGroupBox("AnnData overlays and population-to-cohort transfer")
        overlay_form = QFormLayout(overlay_group)
        self.overlay_obs_combo = QComboBox()
        self.overlay_button = QPushButton("Render observation overlay")
        self.population_obs_combo = QComboBox()
        self.population_value_combo = QComboBox()
        self.rank_rois_button = QPushButton("Rank ROIs by selected population")
        self.use_population_button = QPushButton(
            "Use this population as classification cohort"
        )
        overlay_form.addRow("Categorical or numeric observation", self.overlay_obs_combo)
        overlay_form.addRow("", self.overlay_button)
        overlay_form.addRow("Population observation", self.population_obs_combo)
        overlay_form.addRow("Population", self.population_value_combo)
        population_actions = QWidget()
        population_actions_layout = QHBoxLayout(population_actions)
        population_actions_layout.setContentsMargins(0, 0, 0, 0)
        population_actions_layout.addWidget(self.rank_rois_button)
        population_actions_layout.addWidget(self.use_population_button)
        overlay_form.addRow("", population_actions)
        explore_layout.addWidget(overlay_group)
        image_group = QGroupBox("Raw, extra, and RGB images")
        image_layout = QVBoxLayout(image_group)
        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.channel_list.setMaximumHeight(120)
        image_actions = QHBoxLayout()
        self.load_channels_button = QPushButton("Load selected images")
        self.load_rgb_button = QPushButton("Load first three selected as RGB")
        image_actions.addWidget(self.load_channels_button)
        image_actions.addWidget(self.load_rgb_button)
        image_layout.addWidget(self.channel_list)
        image_layout.addLayout(image_actions)
        explore_layout.addWidget(image_group)
        add_tab(explore, "Explore")

        # Classify
        classify = QWidget()
        classify_layout = QVBoxLayout(classify)
        selection_group = QGroupBox("Selected cell annotation")
        selection_form = QFormLayout(selection_group)
        self.selected_cell_label = QLabel("No cohort cell selected")
        self.class_combo = QComboBox()
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
        selection_form.addRow("Class", self.class_combo)
        selection_form.addRow("", annotation_buttons)
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
        model_actions = QWidget()
        model_actions_layout = QHBoxLayout(model_actions)
        model_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.train_button = QPushButton("Train")
        self.score_button = QPushButton("Score cohort")
        self.refresh_queue_button = QPushButton("Refresh uncertainty queue")
        model_actions_layout.addWidget(self.train_button)
        model_actions_layout.addWidget(self.score_button)
        model_actions_layout.addWidget(self.refresh_queue_button)
        self.queue_list = QListWidget()
        self.queue_list.setMaximumHeight(145)
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
        model_form.addRow("", model_actions)
        queue_filters = QWidget()
        queue_filters_layout = QHBoxLayout(queue_filters)
        queue_filters_layout.setContentsMargins(0, 0, 0, 0)
        queue_filters_layout.addWidget(self.queue_roi_combo)
        queue_filters_layout.addWidget(self.queue_class_combo)
        queue_filters_layout.addWidget(self.queue_review_combo)
        queue_filters_layout.addWidget(self.queue_confidence_spin)
        model_form.addRow("Queue filters (ROI/class/review/conf.)", queue_filters)
        model_form.addRow("Ambiguous unlabelled cells", self.queue_list)
        model_form.addRow("High-confidence threshold", self.confidence_spin)
        model_form.addRow("", self.bulk_propose_button)
        model_form.addRow("Probability class", self.probability_class_combo)
        model_form.addRow("", self.show_probability_button)
        classify_layout.addWidget(model_group)
        add_tab(classify, "Classify")

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
        add_tab(regions, "Regions & Export")

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
        add_tab(layers, "Layers & Status")

        self._set_class_rows(segmentation_qc_classes())
        self._connect_signals()
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
        self.build_features_button.clicked.connect(self._guard(self.start_feature_build))
        self.cancel_features_button.clicked.connect(self.cancel_feature_build)
        self.hpc_button.clicked.connect(
            lambda: self.set_status(
                "Managed build: set napari_sbt.active_experiment in config.yaml, "
                "then run `sbt run cellfeat` (8 CPUs, 64 GB, 24 hours)."
            )
        )
        self.roi_combo.currentTextChanged.connect(self._guard(self.load_roi))
        self.reload_roi_button.clicked.connect(self._guard(self.load_roi))
        self.show_empty_rois.toggled.connect(self.refresh_rois)
        self.context_check_display.toggled.connect(self.toggle_context)
        self.overlay_obs_combo.currentTextChanged.connect(
            lambda: self.set_status("Overlay selection changed.")
        )
        self.overlay_button.clicked.connect(self._guard(self.render_obs_overlay))
        self.population_obs_combo.currentTextChanged.connect(
            self._guard(self.refresh_population_values)
        )
        self.rank_rois_button.clicked.connect(self._guard(self.rank_rois_by_population))
        self.use_population_button.clicked.connect(
            self._guard(self.use_population_as_cohort)
        )
        self.load_channels_button.clicked.connect(self._guard(self.load_selected_channels))
        self.load_rgb_button.clicked.connect(self._guard(self.load_rgb))
        self.propose_button.clicked.connect(lambda: self.annotate_selected("proposed"))
        self.confirm_button.clicked.connect(lambda: self.annotate_selected("confirmed"))
        self.confirm_proposed_button.clicked.connect(self._guard(self.confirm_all_proposed))
        self.mark_reviewed_button.clicked.connect(self._guard(self.mark_roi_reviewed))
        self.seed_obs_button.clicked.connect(self._guard(self.seed_proposals_from_obs))
        self.train_button.clicked.connect(self._guard(self.train_model))
        self.score_button.clicked.connect(self._guard(self.score_model))
        self.refresh_queue_button.clicked.connect(self._guard(self.refresh_uncertainty_queue))
        self.queue_list.itemDoubleClicked.connect(self._guard(self.navigate_queue_item))
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

    def _guard(self, callback):
        def wrapped(*args, **kwargs):
            try:
                return callback(*args, **kwargs)
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

    def _set_classification_enabled(self, enabled: bool) -> None:
        for widget in (
            self.class_combo,
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
        self.refresh_scope_values()
        self.refresh_population_values()
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
            sources.append(
                FeatureSource(source_id=source_id.strip(), kind="table", path=path.strip())
            )
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
            sources.append(
                FeatureSource(
                    source_id=source_id.strip(),
                    kind="anndata",
                    path=path.strip(),
                    representation=representation.strip(),
                )
            )
        return sources

    def create_experiment(self) -> None:
        preview = self.preview_cohort()
        reply = self.QMessageBox.question(
            self.root,
            "Freeze cohort",
            (
                f"Freeze {preview.eligible_cell_count:,} eligible identities across "
                f"{preview.represented_roi_count} ROIs? Later membership changes "
                "require an explicit experiment revision."
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
        channels = [
            value.strip()
            for value in self.channels_edit.text().split(",")
            if value.strip()
        ]
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
            feature_sources=self.feature_sources(),
            synthetic_features=SyntheticFeatureRecipe(
                channels=channels,
                mask_offset_px=self.offset_spin.value(),
                allow_positive_offset_overlap=self.offset_overlap_check.isChecked(),
                distribution_features=self.distribution_check.isChecked(),
                region_features=self.region_check.isChecked(),
                gradient_features=self.gradient_check.isChecked(),
                shape_features=self.shape_check.isChecked(),
                context_features=self.context_check.isChecked(),
                background_ring_px=self.background_ring_spin.value(),
                normalization_dict_path=(
                    self.normalization_edit.text().strip() or None
                ),
            ),
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
        self.experiment_edit.setText(str(self.paths.root))
        self.name_edit.setText(self.manifest.name)
        self.anndata_edit.setText(_path_text(self.manifest.anndata_path))
        self.masks_edit.setText(self.manifest.masks_folder)
        self.roi_obs_edit.setText(self.manifest.roi_obs)
        self.object_obs_edit.setText(self.manifest.object_id_obs)
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
        self.channels_edit.setText(
            ", ".join(self.manifest.synthetic_features.channels)
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
        self.feature_tables_edit.setPlainText(
            "\n".join(
                f"{source.source_id}={source.path}"
                for source in self.manifest.feature_sources
                if source.kind == "table"
            )
        )
        self.anndata_features_edit.setPlainText(
            "\n".join(
                f"{source.source_id}={source.path}::{source.representation or 'X'}"
                for source in self.manifest.feature_sources
                if source.kind == "anndata"
            )
        )
        self._set_class_rows(self.manifest.classes)
        self.cohort = read_dataframe(
            self.paths.root / self.manifest.cell_scope.snapshot_path
        )
        if self.paths.labels.exists():
            self.labels = validate_labels(
                read_dataframe(self.paths.labels),
                class_ids=[item.class_id for item in self.manifest.classes],
                cohort=self.cohort,
            )
        else:
            self.labels = empty_labels()
        if self.paths.scores.exists():
            self.scores = read_dataframe(self.paths.scores)
        reviewed_path = self.paths.labels.parent / "reviewed_rois.json"
        if reviewed_path.exists():
            self.reviewed_rois = set(
                json.loads(reviewed_path.read_text(encoding="utf-8")).get("rois", [])
            )
        else:
            self.reviewed_rois = set()
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

    def _update_scope_text(self) -> None:
        if self.manifest is None:
            self.scope_label.setText("No experiment: classification is disabled.")
            return
        self.scope_label.setText(
            f"{self.manifest.cell_scope.eligible_cell_count:,} eligible cells / "
            f"{self.manifest.cell_scope.total_cell_count:,} total cells across "
            f"{self.manifest.cell_scope.represented_roi_count} ROIs — "
            f"experiment {self.manifest.name!r} r{self.manifest.revision}"
        )

    def refresh_class_controls(self) -> None:
        self.class_combo.clear()
        self.probability_class_combo.clear()
        self.queue_class_combo.clear()
        self.queue_class_combo.addItem("All predicted classes", None)
        for definition in self.manifest.classes:
            label = f"{definition.shortcut}: {definition.name}"
            self.class_combo.addItem(label, definition.class_id)
            self.probability_class_combo.addItem(label, definition.class_id)
            self.queue_class_combo.addItem(definition.name, definition.class_id)
        self.queue_roi_combo.clear()
        self.queue_roi_combo.addItem("All eligible ROIs", None)
        self.queue_roi_combo.addItems(sorted(self.cohort["ROI"].astype(str).unique()))
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

    def refresh_rois(self) -> None:
        if self.manifest is None:
            return
        eligible_rois = sorted(self.cohort["ROI"].astype(str).unique())
        rois = eligible_rois
        if self.show_empty_rois.isChecked():
            all_rois = set(discover_mask_files(self.manifest.masks_folder))
            rois = sorted(all_rois | set(eligible_rois))
        current = self.roi_combo.currentText()
        self.roi_combo.blockSignals(True)
        self.roi_combo.clear()
        self.roi_combo.addItems(rois)
        if current in rois:
            self.roi_combo.setCurrentText(current)
        self.roi_combo.blockSignals(False)

    def _remove_layers(self, names: Iterable[str]) -> None:
        for name in names:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)

    def _replace_layer(self, name: str, data, layer_type: str, **kwargs):
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            layer.data = data
            for key, value in kwargs.items():
                if hasattr(layer, key):
                    setattr(layer, key, value)
            return layer
        method = getattr(self.viewer, f"add_{layer_type}")
        return method(data, name=name, **kwargs)

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
        cohort_layer = self._replace_layer(
            "classification_cohort",
            restricted,
            "labels",
            visible=True,
        )
        if not getattr(cohort_layer, "_napari_sbt_click_bound", False):
            cohort_layer.mouse_drag_callbacks.append(self._on_cohort_click)
            cohort_layer._napari_sbt_click_bound = True
        context = np.where(restricted == 0, full_mask, 0)
        context_layer = self._replace_layer(
            "excluded_segmentation_context",
            context,
            "labels",
            visible=self.context_check_display.isChecked(),
            opacity=0.18,
        )
        context_layer.visible = self.context_check_display.isChecked()
        self._remove_layers(
            [
                CLASS_LAYER_NAMES["confirmed"],
                CLASS_LAYER_NAMES["proposed"],
                CLASS_LAYER_NAMES["predicted"],
                CLASS_LAYER_NAMES["uncertainty"],
            ]
        )
        self.refresh_classification_layers()
        self.refresh_channel_list()
        self.set_status(
            f"ROI {roi}: {len(eligible)} eligible cells; clicks on other mask "
            "labels are ignored."
        )

    def toggle_context(self, checked: bool) -> None:
        if "excluded_segmentation_context" in self.viewer.layers:
            self.viewer.layers["excluded_segmentation_context"].visible = checked

    def _on_cohort_click(self, layer, event) -> None:
        if event.type != "mouse_press":
            return
        value = layer.get_value(event.position)
        object_id = int(value or 0)
        if object_id <= 0:
            self.current_selected_object = None
            self.selected_cell_label.setText("Cell is outside this experiment")
            self.set_status("Cell is outside this experiment; annotation was ignored.")
            return
        self.current_selected_object = object_id
        self.selected_cell_label.setText(f"{self.current_roi} / object {object_id}")

    def refresh_channel_list(self) -> None:
        self.channel_list.clear()
        if self.manifest is None or not self.current_roi:
            return
        paths = discover_roi_images(
            self.manifest.images_folders + self.manifest.extra_images_folders,
            self.current_roi,
        )
        for channel, path in paths.items():
            from qtpy.QtWidgets import QListWidgetItem

            list_item = QListWidgetItem(channel)
            list_item.setData(self.Qt.UserRole, str(path))
            self.channel_list.addItem(list_item)

    def load_selected_channels(self) -> None:
        for item in self.channel_list.selectedItems():
            image, is_rgb = load_display_image(item.data(self.Qt.UserRole))
            name = f"image::{item.text()}"
            self._replace_layer(name, image, "image", rgb=is_rgb, blending="additive")

    def load_rgb(self) -> None:
        selected = self.channel_list.selectedItems()
        if len(selected) < 3:
            raise ValueError("Select at least three channel images for an RGB view.")
        images = [load_display_image(item.data(self.Qt.UserRole))[0] for item in selected[:3]]
        if len({image.shape for image in images}) != 1:
            raise ValueError("RGB channels have mismatched shapes.")
        rgb = np.stack(images, axis=-1)
        self._replace_layer(
            "population_qc_rgb",
            rgb,
            "image",
            rgb=True,
            blending="translucent",
        )

    def refresh_population_values(self) -> None:
        self.population_value_combo.clear()
        if self.adata is None or self.population_obs_combo.currentText() not in self.adata.obs:
            return
        values = sorted(
            self.adata.obs[self.population_obs_combo.currentText()]
            .dropna()
            .astype(str)
            .unique()
        )
        self.population_value_combo.addItems(values)

    def render_obs_overlay(self) -> None:
        if self.adata is None or self.current_mask is None:
            raise RuntimeError("Load an experiment ROI and AnnData first.")
        observation = self.overlay_obs_combo.currentText()
        rows = self.adata.obs.loc[
            self.adata.obs[self.manifest.roi_obs].astype(str).eq(self.current_roi)
        ]
        object_ids = pd.to_numeric(
            rows[self.manifest.object_id_obs], errors="coerce"
        ).astype("Int64")
        values = rows[observation]
        eligible = set(
            self.cohort.loc[
                self.cohort["ROI"].astype(str).eq(self.current_roi), "ObjectNumber"
            ].astype(int)
        )
        selected = object_ids.isin(eligible)
        if pd.api.types.is_numeric_dtype(values):
            mapping = pd.Series(
                pd.to_numeric(values[selected], errors="coerce").to_numpy(),
                index=object_ids[selected].astype(int),
            )
            overlay = _identity_value_map(self.current_mask, mapping)
            self._replace_layer(f"obs::{observation}", overlay, "image", colormap="viridis")
        else:
            categories = sorted(values[selected].dropna().astype(str).unique())
            codes = {value: index + 1 for index, value in enumerate(categories)}
            mapping = pd.Series(
                values[selected].astype(str).map(codes).to_numpy(),
                index=object_ids[selected].astype(int),
            )
            overlay = _identity_value_map(self.current_mask, mapping, dtype=np.int32)
            self._replace_layer(f"obs::{observation}", overlay, "labels")
        self.set_status(f"Rendered cohort-only AnnData overlay {observation!r}.")

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
        ranked = [str(roi) for roi in counts.index if str(roi) in eligible]
        self.roi_combo.blockSignals(True)
        self.roi_combo.clear()
        self.roi_combo.addItems(ranked)
        self.roi_combo.blockSignals(False)
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
            self.labels = set_label(
                self.labels,
                roi=self.current_roi,
                object_number=self.current_selected_object,
                class_id=self.selected_class_id(),
                state=state,
                source="manual",
                user=os.environ.get("USERNAME") or os.environ.get("USER", ""),
            )
            self.labels = validate_labels(
                self.labels,
                class_ids=[item.class_id for item in self.manifest.classes],
                cohort=self.cohort,
            )
            write_dataframe(self.paths.labels, self.labels)
            append_audit(
                self.paths,
                {
                    "action": "set_label",
                    "ROI": self.current_roi,
                    "ObjectNumber": self.current_selected_object,
                    "class_id": self.selected_class_id(),
                    "state": state,
                },
            )
            self.manifest.locked = bool(
                (self.labels["state"] == "confirmed").any()
            )
            save_experiment(self.manifest, self.paths.root, audit_action="label_update")
            self.refresh_classification_layers()
            self.refresh_status()
            self.set_status(
                f"Set {self.current_roi}/{self.current_selected_object} to "
                f"{self.selected_class_id()} ({state})."
            )
        except Exception as error:  # noqa: BLE001 - Qt callback error boundary
            self.set_status(f"ERROR — {type(error).__name__}: {error}")
            self.QMessageBox.critical(
                self.root, "napari_sbt", f"{type(error).__name__}: {error}"
            )

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
            model_state = (
                "current cohort/features/labels"
                if model_metadata.get("cohort_fingerprint")
                == self.manifest.cell_scope.snapshot_sha256
                and model_metadata.get("feature_set_id")
                == self.manifest.active_feature_set_id
                and labels_current
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
        represented = set(self.cohort["ROI"].astype(str))
        reviewed = len(represented & self.reviewed_rois)
        self.set_status(
            "FRESHNESS — "
            f"cohort={cohort_state}; classes={class_counts}; "
            f"proposals={proposed}; feature_set={feature_state}; "
            f"model={model_state}; scored_current={scored}/{len(self.cohort)}; "
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
        confirmed_identities = set(
            self.labels.loc[
                self.labels["state"].eq("confirmed"), ["ROI", "ObjectNumber"]
            ]
            .astype({"ROI": str, "ObjectNumber": int})
            .itertuples(index=False, name=None)
        )
        seeded = 0
        for row in self.cohort.itertuples():
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
                color=self._class_colors(),
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
                color=self._class_colors(),
                visible=False,
            )
            uncertainty = pd.Series(
                pd.to_numeric(rows["normalized_entropy"], errors="coerce").to_numpy(),
                index=rows["ObjectNumber"].astype(int),
            )
            self._replace_layer(
                CLASS_LAYER_NAMES["uncertainty"],
                _identity_value_map(self.current_mask, uncertainty),
                "image",
                colormap="magma",
                contrast_limits=(0, 1),
            )

    def _load_feature_table(self) -> pd.DataFrame:
        if not self.paths.feature_table.exists():
            raise FileNotFoundError(
                "No canonical feature table exists. Build or resume features first."
            )
        return read_dataframe(self.paths.feature_table)

    def train_model(self) -> None:
        result = train_multiclass_classifier(
            self._load_feature_table(),
            self.labels,
            class_ids=[item.class_id for item in self.manifest.classes],
            cohort=self.cohort,
            model_type=self.model_combo.currentData(),
            cohort_fingerprint=self.manifest.cell_scope.snapshot_sha256,
            feature_set_id=self.manifest.active_feature_set_id,
        )
        if not result.ok:
            raise ValueError("; ".join(result.errors))
        self.model_bundle = result.bundle
        save_model_bundle(
            self.model_bundle, self.paths.models / "classifier_latest.joblib"
        )
        for warning in result.warnings:
            self.set_status(f"MODEL WARNING — {warning}")
        self.set_status(
            f"Trained {self.model_bundle.metadata['model_type']} on "
            f"{len(result.training_table)} confirmed cohort cells."
        )

    def score_model(self) -> None:
        if self.model_bundle is None:
            latest = self.paths.models / "classifier_latest.joblib"
            if not latest.exists():
                self.train_model()
            else:
                from .classifier import load_model_bundle

                self.model_bundle = load_model_bundle(latest)
        metadata = self.model_bundle.metadata
        if (
            metadata.get("cohort_fingerprint")
            != self.manifest.cell_scope.snapshot_sha256
            or metadata.get("feature_set_id")
            != self.manifest.active_feature_set_id
            or metadata.get("labels_fingerprint")
            != confirmed_labels_fingerprint(self.labels)
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
            raise ValueError("Score the cohort before building an uncertainty queue.")
        selected_roi = self.queue_roi_combo.currentData()
        selected_class = self.queue_class_combo.currentData()
        review = self.queue_review_combo.currentText().lower()
        if review == "unlabelled":
            queue = uncertainty_queue(
                self.scores,
                self.labels,
                limit=10000,
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
        ].head(250)
        self.queue_list.clear()
        for row in queue.itertuples():
            from qtpy.QtWidgets import QListWidgetItem

            item = QListWidgetItem(
                f"{row.ROI} / {row.ObjectNumber} — {row.predicted_class}, "
                f"entropy {row.normalized_entropy:.3f}"
            )
            item.setData(self.Qt.UserRole, (str(row.ROI), int(row.ObjectNumber)))
            self.queue_list.addItem(item)

    def navigate_queue_item(self, item) -> None:
        roi, object_number = item.data(self.Qt.UserRole)
        self.roi_combo.setCurrentText(roi)
        self.load_roi(roi)
        self.current_selected_object = int(object_number)
        self.selected_cell_label.setText(f"{roi} / object {object_number}")

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
            _identity_value_map(self.current_mask, mapping),
            "image",
            colormap="viridis",
            contrast_limits=(0, 1),
        )
        self.set_status(f"Showing cohort-only probability for class {class_id!r}.")

    def start_feature_build(self) -> None:
        if self.manifest is None:
            raise RuntimeError("Create or load an experiment before feature extraction.")
        if self.feature_process is not None:
            raise RuntimeError("A feature build is already running.")
        self.manifest.feature_sources = self.feature_sources()
        self.manifest.synthetic_features = SyntheticFeatureRecipe(
            channels=[
                value.strip()
                for value in self.channels_edit.text().split(",")
                if value.strip()
            ],
            mask_offset_px=self.offset_spin.value(),
            allow_positive_offset_overlap=self.offset_overlap_check.isChecked(),
            distribution_features=self.distribution_check.isChecked(),
            region_features=self.region_check.isChecked(),
            gradient_features=self.gradient_check.isChecked(),
            shape_features=self.shape_check.isChecked(),
            context_features=self.context_check.isChecked(),
            background_ring_px=self.background_ring_spin.value(),
            normalization_dict_path=self.normalization_edit.text().strip() or None,
        )
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
        self.build_features_button.setEnabled(False)
        self.cancel_features_button.setEnabled(True)
        process.start()
        self.set_status("Started cohort-first feature build in a subprocess.")

    def _read_feature_progress(self) -> None:
        if self.feature_process is None:
            return
        text = bytes(self.feature_process.readAllStandardOutput()).decode(
            errors="replace"
        )
        for line in text.splitlines():
            try:
                event = json.loads(line)
                self.set_status(
                    f"FEATURE {event.get('event', 'progress')} — "
                    f"{event.get('roi', '')} {event.get('error', '')}".strip()
                )
            except json.JSONDecodeError:
                self.set_status(line)

    def _feature_build_finished(self, exit_code: int, _status) -> None:
        self._read_feature_progress()
        self.feature_process = None
        self.build_features_button.setEnabled(True)
        self.cancel_features_button.setEnabled(False)
        if exit_code == 0 and self.paths is not None:
            self.manifest, self.paths = load_experiment(self.paths.root)
            self._update_scope_text()
            self.refresh_status()
        self.set_status(
            "Feature build completed." if exit_code == 0 else f"Feature build exited {exit_code}."
        )

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
        self.viewer.add_labels(expanded, name=f"{layer.name}_expanded")

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
            self.viewer.add_labels(resized, name=f"{layer.name}_resized")
        else:
            self.viewer.add_image(
                resized,
                name=f"{layer.name}_resized",
                rgb=bool(data.ndim == 3 and data.shape[-1] in (3, 4)),
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
        self.viewer.add_image(masked, name=f"{layer.name}_cohort_masked")


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
