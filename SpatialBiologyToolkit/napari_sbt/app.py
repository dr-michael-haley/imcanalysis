"""Unified Napari dock for cohort-first IMC exploration and classification."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from SpatialBiologyToolkit._napari_imc_normalization import (
    find_normalization_value,
    load_normalization_mapping,
    prepare_normalization_dict,
)
from SpatialBiologyToolkit.pipeline.manifests import write_json
from SpatialBiologyToolkit.qc_classifier.io import (
    build_image_channel_aliases,
    discover_mask_files,
    discover_roi_image_index,
    discover_roi_images,
    load_display_image,
    load_mask,
    resolve_mask_file,
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
from .explore import (
    EXPLORE_RECIPE_FUNCTION_KEYS,
    EXPLORE_STATE_VERSION,
    SIX_COLOUR_COLORMAPS,
    ExploreRecipePreset,
    ExploreReviewState,
    ExploreViewRecipe,
    categorical_colour_map,
    format_roi_metadata_value,
    identity_value_map,
    marker_values,
    population_recipe_key,
    recipe_layer_data_is_current,
    roi_level_metadata,
)
from .exports import (
    apply_assignments_to_anndata,
    build_assignment_table,
    export_annotated_anndata,
    export_assignment_table,
    export_cleaned_masks,
    materialize_cohort_masks,
)
from .feature_catalog import (
    FEATURE_FAMILY_CATALOG,
    FEATURE_FAMILY_DESCRIPTIONS,
)
from .feature_refinement import compact_synthetic_recipe
from .features import classifier_seen_mask
from .help import load_help_markdown
from .labeler import (
    LabelerClass,
    apply_labeler_to_anndata,
    build_labeler_export_table,
    default_labeler_classes,
    empty_labeler_records,
    labeler_summary,
    remove_labeler_record,
    set_labeler_record,
    validate_labeler_classes,
)
from .labels import (
    confirm_proposed,
    empty_labels,
    remove_all_proposed_labels,
    remove_proposed_label,
    set_label,
    validate_labels,
)
from .models import (
    ClassificationClass,
    DisplaySettings,
    ExperimentManifest,
    FeatureDiscoveryTrial,
    FeatureSource,
    SyntheticFeatureRecipe,
    segmentation_qc_classes,
    slugify,
)
from .population_curation import (
    BASE_MAPPING_COLUMNS,
    COMPONENT_COLUMNS,
    GraphSubclusterRequest,
    PopulationDraft,
    PopulationWorkspace,
    PopulationWorkspacePaths,
    append_population_audit,
    apply_population_draft,
    atomic_write_curated_anndata,
    component_tables_from_assignments,
    empty_components,
    empty_membership,
    ensure_population_workspace,
    import_base_mapping_csv,
    integrate_component_tables,
    list_population_drafts,
    load_population_draft,
    ordered_source_labels,
    population_draft_paths,
    population_draft_sync_state,
    population_workspace_paths,
    read_population_audit,
    save_graph_subcluster_request,
    save_population_draft,
    source_obs_fingerprint,
    synthesize_population_labels,
)
from .population_curation import (
    create_population_draft as create_population_draft_asset,
)
from .population_qc import (
    POPULATION_QC_SETTINGS_COLUMNS,
    build_population_qc_recipe,
    inherit_setup_contrast_limits,
    parse_legacy_contrast,
    rank_population_rois,
    retarget_population_qc_recipe,
    top_population_markers,
)
from .resources import resolve_worker_count
from .scanpy_plotting import (
    build_scanpy_plot,
    figure_subplot_margins,
    fit_scanpy_figure_to_canvas,
)
from .setup import (
    WORKFLOW_PRESENTATIONS,
    WorkspaceSummary,
    discover_workspaces,
    setup_checks,
    setup_is_ready,
    suggest_identity_columns,
    workflow_presentation,
    workspace_destination,
    workspace_folder,
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

SELECTED_CELL_LAYER_NAME = "selected_cell_outline"
NONCONTEXT_MASK_LAYER_NAME = "noncontext_mask"
LABELER_LAYER_NAME = "labeler_assignments"
LABELER_SELECTED_CELL_LAYER_NAME = "labeler_selected_cell_outline"
EXPLORE_DATA_CACHE_MAX_BYTES = 512 * 1024 * 1024
EXPLORE_DATA_CACHE_MAX_ITEMS = 48

WORKFLOW_DESCRIPTIONS = {
    "data_exploration": (
        "Explore ROIs, images, AnnData overlays, reusable layer recipes, regions, "
        "population-focused views, and Scanpy plots. Classification controls are "
        "hidden."
    ),
    "population_qc": (
        "Review existing populations with saved RGB marker combinations, abundance-"
        "guided ROI sampling, tissue overlays, and Scanpy summary plots."
    ),
    "classification": (
        "Define a cohort and classes, build/refine features, annotate examples, "
        "train a model, review predictions, inspect Scanpy plots, and export final "
        "identities."
    ),
    "cell_labeling": (
        "Manually assign simple labels to selected cells and export identity lists "
        "without building or training a classifier."
    ),
    "population_curation": (
        "Name, merge, subcluster, inspect, and export AnnData population observations "
        "with image-based Population QC and dedicated Scanpy plotting."
    ),
    "full_workspace": (
        "Show every NapariSBT tab. Use this when combining exploration, population "
        "curation, manual labeling, and classification in one session."
    ),
}

WORKFLOW_VISIBLE_TABS = {
    "data_exploration": {
        "setup",
        "explore",
        "population_qc",
        "scanpy_plotting",
        "regions_export",
        "layers_status",
    },
    "population_qc": {
        "setup",
        "explore",
        "population_qc",
        "scanpy_plotting",
        "layers_status",
    },
    "classification": {
        "setup",
        "feature_building",
        "feature_refinement",
        "explore",
        "population_qc",
        "populations",
        "scanpy_plotting",
        "classify",
        "regions_export",
        "layers_status",
    },
    "cell_labeling": {
        "setup",
        "explore",
        "population_qc",
        "scanpy_plotting",
        "labeler",
        "layers_status",
    },
    "population_curation": {
        "setup",
        "explore",
        "population_qc",
        "populations",
        "scanpy_plotting",
        "regions_export",
        "layers_status",
    },
    "full_workspace": {
        "setup",
        "feature_building",
        "feature_refinement",
        "explore",
        "population_qc",
        "populations",
        "scanpy_plotting",
        "classify",
        "labeler",
        "regions_export",
        "layers_status",
    },
}

MANAGED_RECIPE_LAYERS = {
    "classification_cohort": "Eligible-cell classification mask",
    "excluded_segmentation_context": "Excluded-cell segmentation context",
    NONCONTEXT_MASK_LAYER_NAME: "Classifier: opaque mask outside feature context",
    CLASS_LAYER_NAMES["confirmed"]: "Classifier: confirmed classes",
    CLASS_LAYER_NAMES["proposed"]: "Classifier: proposed classes",
    CLASS_LAYER_NAMES["predicted"]: "Classifier: predicted classes",
    CLASS_LAYER_NAMES["uncertainty"]: (
        "Classifier: uncertainty or selected-class probability"
    ),
    SELECTED_CELL_LAYER_NAME: "Classifier: currently selected cell",
    LABELER_LAYER_NAME: "Labeler: assigned cells",
    LABELER_SELECTED_CELL_LAYER_NAME: "Labeler: currently selected cell",
}

MANAGED_LAYER_DEFAULT_VISIBILITY = {
    "classification_cohort": False,
    "excluded_segmentation_context": False,
    NONCONTEXT_MASK_LAYER_NAME: False,
    CLASS_LAYER_NAMES["confirmed"]: True,
    CLASS_LAYER_NAMES["proposed"]: True,
    CLASS_LAYER_NAMES["predicted"]: False,
    CLASS_LAYER_NAMES["uncertainty"]: True,
    SELECTED_CELL_LAYER_NAME: True,
    LABELER_LAYER_NAME: True,
    LABELER_SELECTED_CELL_LAYER_NAME: True,
}

MANAGED_LAYER_DEFAULT_OPACITY = {
    "classification_cohort": 1.0,
    "excluded_segmentation_context": 0.18,
    NONCONTEXT_MASK_LAYER_NAME: 1.0,
    CLASS_LAYER_NAMES["confirmed"]: 1.0,
    CLASS_LAYER_NAMES["proposed"]: 1.0,
    CLASS_LAYER_NAMES["predicted"]: 1.0,
    CLASS_LAYER_NAMES["uncertainty"]: 1.0,
    SELECTED_CELL_LAYER_NAME: 1.0,
    LABELER_LAYER_NAME: 1.0,
    LABELER_SELECTED_CELL_LAYER_NAME: 1.0,
}

MANAGED_LAYER_DEFAULT_CONTOUR = {
    "classification_cohort": 1,
    "excluded_segmentation_context": 1,
    NONCONTEXT_MASK_LAYER_NAME: 0,
    CLASS_LAYER_NAMES["confirmed"]: 0,
    CLASS_LAYER_NAMES["proposed"]: 2,
    CLASS_LAYER_NAMES["predicted"]: 1,
    SELECTED_CELL_LAYER_NAME: 2,
    LABELER_LAYER_NAME: 2,
    LABELER_SELECTED_CELL_LAYER_NAME: 3,
}


def _path_text(value: str | Path | None) -> str:
    return "" if value is None else str(Path(value))


def _normalise_anndata_input(
    anndata_path: str | Path | object | None,
    anndata: object | None,
) -> tuple[str | Path | None, object | None]:
    """Separate a filesystem source from a live AnnData object."""

    if anndata_path is not None and anndata is not None:
        raise ValueError("Supply either anndata_path or anndata, not both.")
    candidate = anndata if anndata is not None else anndata_path
    if candidate is None:
        return None, None
    if isinstance(candidate, (str, os.PathLike)):
        return candidate, None

    import anndata as ad

    if not isinstance(candidate, ad.AnnData):
        raise TypeError(
            "AnnData input must be a path or an anndata.AnnData object; "
            f"received {type(candidate).__name__}."
        )
    return None, candidate


def _write_anndata_snapshot(adata, destination: str | Path) -> Path:
    """Atomically persist a live AnnData as an experiment-owned input."""

    output = Path(destination).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite AnnData snapshot: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    try:
        adata.write_h5ad(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _split_paths(value: str) -> list[str]:
    return [
        item.strip() for item in value.replace(";", "\n").splitlines() if item.strip()
    ]


class NapariSBTController:
    """Qt-independent state plus Qt/Napari callbacks for one workflow dock."""

    def __init__(
        self,
        viewer,
        *,
        project_root: str | Path | None = None,
        experiment: str | Path | None = None,
        anndata_path: str | Path | object | None = None,
        anndata: object | None = None,
        masks_folder: str | Path | None = None,
        images_folders: Iterable[str | Path] = (),
        extra_images_folders: Iterable[str | Path] = (),
    ) -> None:
        from qtpy.QtCore import Qt, QTimer
        from qtpy.QtGui import QColor, QFont, QIcon, QPixmap
        from qtpy.QtWidgets import (
            QAbstractItemView,
            QApplication,
            QButtonGroup,
            QCheckBox,
            QColorDialog,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QFrame,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QHeaderView,
            QInputDialog,
            QLabel,
            QLineEdit,
            QListWidget,
            QMessageBox,
            QProgressBar,
            QPushButton,
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

        resolved_anndata_path, in_memory_anndata = _normalise_anndata_input(
            anndata_path,
            anndata,
        )

        self.Qt = Qt
        self.QMessageBox = QMessageBox
        self.QApplication = QApplication
        self.QFrame = QFrame
        self.QFileDialog = QFileDialog
        self.QColorDialog = QColorDialog
        self.QDialog = QDialog
        self.QDialogButtonBox = QDialogButtonBox
        self.QInputDialog = QInputDialog
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
        self._workspace_container = workspace_folder(self.project_root)
        self._workspace_summaries: list[WorkspaceSummary] = []
        self._launch_experiment = (
            Path(experiment).expanduser().resolve(strict=False) if experiment else None
        )
        self._setup_status_labels: dict[str, object] = {}
        self._workflow_radios: dict[str, object] = {}
        self._updating_setup_controls = False
        self._loaded_workspace_root: Path | None = None
        self.manifest: ExperimentManifest | None = None
        self.paths = None
        self._in_memory_adata = in_memory_anndata
        self.adata = in_memory_anndata
        self.preview: CohortPreview | None = None
        self.cohort = pd.DataFrame()
        self.labels = empty_labels()
        self.labeler_classes = default_labeler_classes()
        self.labeler_records = empty_labeler_records()
        self._labeler_experiment_id: str | None = None
        self.scores = pd.DataFrame()
        self.final_assignments = pd.DataFrame()
        self.final_identity_signature: str | None = None
        self.final_identity_decision: dict[str, object] = {}
        self.model_bundle = None
        self.current_roi: str | None = None
        self.current_mask: np.ndarray | None = None
        self.current_mask_path: Path | None = None
        self.current_selected_object: int | None = None
        self.current_labeler_object: int | None = None
        self.feature_process = None
        self.source_validation_process = None
        self.refinement_process = None
        self.population_process = None
        self.refinement_cancel_requested = False
        self.feature_build_started_at: float | None = None
        self.feature_last_event_at: float | None = None
        self.feature_progress_state: dict[str, int | float | str] = {}
        self._feature_output_buffer = ""
        self._source_validation_output_buffer = ""
        self._refinement_output_buffer = ""
        self._population_output_buffer = ""
        self._population_pending_run: dict[str, object] | None = None
        self.population_workspace: PopulationWorkspace | None = None
        self.population_workspace_paths: PopulationWorkspacePaths | None = None
        self.population_draft: PopulationDraft | None = None
        self._population_draft_dirty = False
        self.population_base_mapping = pd.DataFrame(columns=BASE_MAPPING_COLUMNS)
        self.population_components = pd.DataFrame(columns=COMPONENT_COLUMNS)
        self.population_membership = empty_membership()
        self.scanpy_plot_windows: dict[str, dict[str, object]] = {}
        self.reviewed_rois: set[str] = set()
        self._class_shortcuts: list[str] = []
        self._explore_recipe_shortcuts: list[str] = []
        self.current_image_paths: dict[str, Path] = {}
        self.explore_recipe = ExploreViewRecipe()
        self.explore_review_state = ExploreReviewState()
        self.display_normalization: dict[str, float] = {}
        self._workflow_tab_indices: dict[str, int] = {}
        self.population_qc_roi_buttons: dict[str, object] = {}
        self._mask_path_index: dict[str, Path] = {}
        self._roi_image_path_index: dict[str, dict[str, Path]] = {}
        self._asset_index_signature: str | None = None
        self._integrity_signature: str | None = None
        self._population_qc_cohort_selector: np.ndarray | None = None
        self._population_qc_marker_cache: dict[tuple, list[str]] = {}
        self._population_qc_ranking_cache: dict[tuple, list[tuple[str, int]]] = {}
        self._adata_roi_positions: dict[str, np.ndarray] = {}
        self._roi_level_metadata: dict[str, dict[str, object]] | None = None
        self._cohort_ids_by_roi: dict[str, set[int]] = {}
        self._recipe_tracking_workflow: str | None = None
        self._explore_layer_names: set[str] = set()
        self._explore_reused_layer_count = 0
        self._explore_cached_layer_count = 0
        self._explore_layer_data_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._explore_layer_data_cache_bytes = 0
        self._recipe_list_refresh_pending = False
        self._roi_review_refresh_pending = False
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
        self.activity_state = "idle"
        self.activity_action = "Ready"
        self.activity_detail = "No active operation."
        self.activity_started_at: float | None = None
        self.activity_finished_at: float | None = None
        self.activity_waiting_for_process = False

        self.root = QWidget()
        self.feature_health_timer = QTimer(self.root)
        self.feature_health_timer.setInterval(1000)
        self.feature_health_timer.timeout.connect(self._update_feature_process_health)
        root_layout = QVBoxLayout(self.root)
        self.scope_label = QLabel(
            "No workflow workspace: choose a task and dataset in Setup."
        )
        self.scope_label.setWordWrap(True)
        root_layout.addWidget(self.scope_label)
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        class WorkflowGroupBox(QGroupBox):
            """A coloured workflow box with an always-visible help affordance."""

            def __init__(
                group_self,
                title: str,
                *,
                accent: str,
                numbered: bool,
                help_callback,
            ) -> None:
                super().__init__(title)
                group_self.setProperty("sbtWorkflowBox", "true")
                group_self.setProperty("sbtAccent", accent)
                group_self.setProperty("sbtNumbered", "true" if numbered else "false")
                group_self.help_button = QPushButton("❓ Help", group_self)
                group_self.help_button.setObjectName("sbtBoxHelpButton")
                group_self.help_button.setCursor(Qt.PointingHandCursor)
                group_self.help_button.setToolTip(
                    f"Open focused instructions for: {title}"
                )
                group_self.help_button.setAccessibleName(f"Help for {title}")
                group_self.help_button.setFixedHeight(28)
                group_self.help_button.setMinimumWidth(88)
                group_self.help_button.clicked.connect(help_callback)
                group_self.help_button.raise_()
                group_self._position_help_button()

            def _position_help_button(group_self) -> None:
                width = max(
                    group_self.help_button.minimumWidth(),
                    group_self.help_button.sizeHint().width(),
                )
                group_self.help_button.setGeometry(
                    max(8, group_self.width() - width - 12),
                    1,
                    width,
                    group_self.help_button.height(),
                )
                group_self.help_button.raise_()

            def resizeEvent(group_self, event) -> None:  # noqa: N802 - Qt API
                super().resizeEvent(event)
                group_self._position_help_button()

            def showEvent(group_self, event) -> None:  # noqa: N802 - Qt API
                super().showEvent(event)
                group_self._position_help_button()

        group_help_counts: dict[str, int] = {}
        group_help_accents = ("blue", "violet", "teal", "amber", "rose", "cyan")

        def workflow_group(title: str, help_topic: str, help_section: str):
            accent_index = group_help_counts.get(help_topic, 0)
            group_help_counts[help_topic] = accent_index + 1
            stripped_title = title.lstrip()
            numbered = len(stripped_title) > 1 and stripped_title[0].isdigit()
            return WorkflowGroupBox(
                title,
                accent=group_help_accents[accent_index % len(group_help_accents)],
                numbered=numbered,
                help_callback=self._guard(
                    lambda: self.show_help(
                        help_topic,
                        title,
                        section=help_section,
                    )
                ),
            )

        def add_tab(widget, title: str, help_topic: str) -> None:
            help_row = QHBoxLayout()
            help_row.addStretch(1)
            help_button = QPushButton("❓ Help for this tab")
            help_button.setObjectName("sbtTabHelpButton")
            help_button.setCursor(Qt.PointingHandCursor)
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
            self._workflow_tab_indices[help_topic] = self.tabs.count() - 1

        # Setup
        setup = QWidget()
        setup_layout = QVBoxLayout(setup)

        workspace_group = workflow_group(
            "1. Start or resume",
            "setup",
            "Start or resume",
        )
        workspace_layout = QVBoxLayout(workspace_group)
        workspace_intro = QLabel(
            "A workspace is NapariSBT's saved analysis area. Open one to continue "
            "where you left off, or name a new one before connecting the dataset."
        )
        workspace_intro.setWordWrap(True)
        workspace_layout.addWidget(workspace_intro)

        project_row = QWidget()
        project_row_layout = QHBoxLayout(project_row)
        project_row_layout.setContentsMargins(0, 0, 0, 0)
        self.registered_project_combo = QComboBox()
        self.registered_project_combo.setMinimumWidth(180)
        self.project_edit = QLineEdit(str(self.project_root))
        self.project_edit.setReadOnly(True)
        self.choose_project_button = QPushButton("Choose project folder…")
        project_row_layout.addWidget(self.registered_project_combo)
        project_row_layout.addWidget(self.project_edit, 1)
        project_row_layout.addWidget(self.choose_project_button)
        project_form = QFormLayout()
        project_form.addRow("Project or dataset", project_row)
        workspace_layout.addLayout(project_form)

        existing_row = QWidget()
        existing_row_layout = QHBoxLayout(existing_row)
        existing_row_layout.setContentsMargins(0, 0, 0, 0)
        self.workspace_combo = QComboBox()
        self.workspace_combo.setMinimumWidth(280)
        self.refresh_workspaces_button = QPushButton("Refresh list")
        self.open_workspace_button = QPushButton("Open selected workspace")
        self.load_experiment_button = QPushButton("Browse elsewhere…")
        self.new_workspace_button = QPushButton("Set up a new workspace")
        existing_row_layout.addWidget(self.workspace_combo, 1)
        existing_row_layout.addWidget(self.refresh_workspaces_button)
        existing_row_layout.addWidget(self.open_workspace_button)
        existing_row_layout.addWidget(self.load_experiment_button)
        existing_row_layout.addWidget(self.new_workspace_button)
        workspace_layout.addWidget(QLabel("Continue an existing workspace"))
        workspace_layout.addWidget(existing_row)
        self.workspace_summary_label = QLabel(
            "No saved workspace has been selected yet."
        )
        self.workspace_summary_label.setWordWrap(True)
        workspace_layout.addWidget(self.workspace_summary_label)

        new_workspace_form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("For example: T-cell population review")
        location_row = QWidget()
        location_layout = QHBoxLayout(location_row)
        location_layout.setContentsMargins(0, 0, 0, 0)
        initial_experiment = _path_text(experiment) or str(
            self._workspace_container / "new_workspace"
        )
        self.experiment_edit = QLineEdit(initial_experiment)
        self._suggested_workspace_path = Path(initial_experiment).expanduser().resolve(
            strict=False
        )
        self.choose_experiment_folder_button = QPushButton("Change location…")
        location_layout.addWidget(self.experiment_edit, 1)
        location_layout.addWidget(self.choose_experiment_folder_button)
        new_workspace_form.addRow("New workspace name", self.name_edit)
        new_workspace_form.addRow("Saved at", location_row)
        workspace_layout.addLayout(new_workspace_form)

        self.setup_readiness_label = QLabel(
            "● Action required — choose a workflow and connect the dataset below."
        )
        self.setup_readiness_label.setObjectName("sbtSetupReadiness")
        self.setup_readiness_label.setWordWrap(True)
        workspace_layout.addWidget(self.setup_readiness_label)
        workspace_actions = QHBoxLayout()
        self.create_button = QPushButton("Create workspace and start")
        self.create_button.setObjectName("sbtPrimaryActionButton")
        self.create_button.setEnabled(False)
        self.next_setup_problem_button = QPushButton("Show next item to fix")
        workspace_actions.addWidget(self.create_button)
        workspace_actions.addWidget(self.next_setup_problem_button)
        workspace_actions.addStretch(1)
        workspace_layout.addLayout(workspace_actions)
        setup_layout.addWidget(workspace_group)

        workflow_choice_group = workflow_group(
            "2. What would you like to do?",
            "setup",
            "Workflow selection",
        )
        workflow_choice_layout = QVBoxLayout(workflow_choice_group)
        self.workflow_combo = QComboBox()
        self.workflow_combo.addItem("Choose a workflow...", None)
        for presentation in WORKFLOW_PRESENTATIONS:
            self.workflow_combo.addItem(presentation.title, presentation.mode)
        self.workflow_combo.hide()
        self.workflow_button_group = QButtonGroup(self.root)
        self.workflow_button_group.setExclusive(True)
        for presentation in WORKFLOW_PRESENTATIONS:
            card = QFrame()
            card.setObjectName("sbtWorkflowChoiceCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 7, 10, 7)
            radio = QRadioButton(presentation.title)
            radio.setProperty("workflowMode", presentation.mode)
            summary = QLabel(
                f"{presentation.summary}\n{presentation.requirements}"
            )
            summary.setWordWrap(True)
            summary.setStyleSheet("color: #475569; padding-left: 24px;")
            card_layout.addWidget(radio)
            card_layout.addWidget(summary)
            self.workflow_button_group.addButton(radio)
            self._workflow_radios[presentation.mode] = radio
            workflow_choice_layout.addWidget(card)
            if presentation.advanced:
                card.hide()
                self.advanced_workflow_card = card
        self.advanced_workflow_check = QCheckBox("Show advanced combined workflow")
        workflow_choice_layout.addWidget(self.advanced_workflow_check)
        self.live_recipe_tracking_check = QCheckBox(
            "Track manual Napari layer changes in the working recipe"
        )
        self.live_recipe_tracking_check.setChecked(True)
        self.live_recipe_tracking_check.setToolTip(
            "Disable this for the lightest ROI-switching path. Explicitly saved "
            "recipes still load, but manual layer display changes are not copied "
            "back into the working recipe automatically."
        )
        self.workflow_description_label = QLabel(
            "Choose the card that best matches your task. NapariSBT will show only "
            "the tabs you need; changing this choice does not delete saved data."
        )
        self.workflow_description_label.setWordWrap(True)
        workflow_choice_layout.insertWidget(0, self.workflow_description_label)
        workflow_choice_layout.addWidget(self.live_recipe_tracking_check)
        setup_layout.addWidget(workflow_choice_group)

        inputs = workflow_group(
            "3. Connect and check the dataset", "setup", "Dataset inputs"
        )
        inputs_form = QFormLayout(inputs)
        self.anndata_edit = QLineEdit(_path_text(resolved_anndata_path))
        if in_memory_anndata is not None:
            self.anndata_edit.setPlaceholderText(
                f"In-memory AnnData ({in_memory_anndata.n_obs:,} cells)"
            )
            self.anndata_edit.setToolTip(
                "This live AnnData is used directly. Creating an experiment writes "
                "an experiment-owned snapshot for restart and worker support."
            )
        self.masks_edit = QLineEdit(_path_text(masks_folder))
        self.images_edit = QTextEdit("\n".join(map(str, images_folders)))
        self.images_edit.setMaximumHeight(70)
        self.extra_images_edit = QTextEdit("\n".join(map(str, extra_images_folders)))
        self.extra_images_edit.setMaximumHeight(55)
        self.roi_obs_edit = QLineEdit("ROI")
        self.object_obs_edit = QLineEdit("ObjectNumber")

        def setup_picker(field, choose_button, reload_button=None):
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            status = QLabel("● Action required")
            status.setObjectName("sbtInputStatus")
            status.setMinimumWidth(128)
            layout.addWidget(status)
            layout.addWidget(field, 1)
            layout.addWidget(choose_button)
            if reload_button is not None:
                layout.addWidget(reload_button)
            return row, status

        self.choose_anndata_button = QPushButton("Choose…")
        self.reload_anndata_button = QPushButton("Load / reload")
        anndata_row, self._setup_status_labels["anndata"] = setup_picker(
            self.anndata_edit,
            self.choose_anndata_button,
            self.reload_anndata_button,
        )
        self.choose_masks_button = QPushButton("Choose…")
        masks_row, self._setup_status_labels["masks"] = setup_picker(
            self.masks_edit,
            self.choose_masks_button,
        )

        image_widget = QWidget()
        image_layout = QHBoxLayout(image_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self._setup_status_labels["images"] = QLabel("● Action required")
        self._setup_status_labels["images"].setObjectName("sbtInputStatus")
        self._setup_status_labels["images"].setMinimumWidth(128)
        image_actions = QVBoxLayout()
        self.add_images_folder_button = QPushButton("Add folder…")
        self.remove_images_folder_button = QPushButton("Remove folder…")
        self.clear_images_folders_button = QPushButton("Clear")
        image_actions.addWidget(self.add_images_folder_button)
        image_actions.addWidget(self.remove_images_folder_button)
        image_actions.addWidget(self.clear_images_folders_button)
        image_actions.addStretch(1)
        image_layout.addWidget(self._setup_status_labels["images"])
        image_layout.addWidget(self.images_edit, 1)
        image_layout.addLayout(image_actions)

        extra_widget = QWidget()
        extra_layout = QHBoxLayout(extra_widget)
        extra_layout.setContentsMargins(0, 0, 0, 0)
        self.extra_images_status_label = QLabel("○ Optional")
        self._setup_status_labels["extra_images"] = self.extra_images_status_label
        self.extra_images_status_label.setObjectName("sbtInputStatus")
        self.extra_images_status_label.setMinimumWidth(128)
        extra_actions = QVBoxLayout()
        self.add_extra_images_folder_button = QPushButton("Add folder…")
        self.remove_extra_images_folder_button = QPushButton("Remove folder…")
        self.clear_extra_images_folders_button = QPushButton("Clear")
        extra_actions.addWidget(self.add_extra_images_folder_button)
        extra_actions.addWidget(self.remove_extra_images_folder_button)
        extra_actions.addWidget(self.clear_extra_images_folders_button)
        extra_actions.addStretch(1)
        extra_layout.addWidget(self.extra_images_status_label)
        extra_layout.addWidget(self.extra_images_edit, 1)
        extra_layout.addLayout(extra_actions)

        inputs_form.addRow("Processed cell data (.h5ad)", anndata_row)
        inputs_form.addRow("Cell masks folder", masks_row)
        inputs_form.addRow("Staining image folders", image_widget)
        inputs_form.addRow("Additional image folders", extra_widget)

        identity_summary = QWidget()
        identity_summary_layout = QHBoxLayout(identity_summary)
        identity_summary_layout.setContentsMargins(0, 0, 0, 0)
        self._setup_status_labels["identity"] = QLabel("● Check needed")
        self._setup_status_labels["identity"].setObjectName("sbtInputStatus")
        self._setup_status_labels["identity"].setMinimumWidth(128)
        self.identity_summary_label = QLabel(
            "Defaulting to ROI for image names and ObjectNumber for mask IDs."
        )
        self.identity_summary_label.setWordWrap(True)
        identity_summary_layout.addWidget(self._setup_status_labels["identity"])
        identity_summary_layout.addWidget(self.identity_summary_label, 1)
        inputs_form.addRow("How cells match images", identity_summary)
        self.advanced_identity_check = QCheckBox(
            "Show advanced cell-identity settings"
        )
        self.identity_widget = QWidget()
        identity_form = QFormLayout(self.identity_widget)
        identity_form.setContentsMargins(0, 0, 0, 0)
        identity_form.addRow("Image / ROI name column", self.roi_obs_edit)
        identity_form.addRow("Cell mask ID column", self.object_obs_edit)
        self.identity_widget.hide()
        inputs_form.addRow(self.advanced_identity_check)
        inputs_form.addRow(self.identity_widget)
        self.validate_integrity_button = QPushButton(
            "Check dataset integrity and build the fast image index"
        )
        self.reload_all_inputs_button = QPushButton("Reload all selected components")
        self.integrity_status_label = QLabel(
            "Not validated in this session. Normal navigation uses direct, "
            "cached file lookups and does not scan complete folders."
        )
        self.integrity_status_label.setWordWrap(True)
        input_actions = QHBoxLayout()
        input_actions.addWidget(self.reload_all_inputs_button)
        input_actions.addWidget(self.validate_integrity_button)
        input_actions.addStretch(1)
        inputs_form.addRow(input_actions)
        inputs_form.addRow("Integrity status", self.integrity_status_label)
        setup_layout.addWidget(inputs)

        display_group = workflow_group(
            "4. Optional image brightness and display defaults",
            "setup",
            "Image normalization and default display",
        )
        display_layout = QVBoxLayout(display_group)
        display_explanation = QLabel(
            "Load a Nimbus channel-to-maximum JSON mapping or a CSV with Marker "
            "and Value columns, then review or edit the Marker/Value table below. "
            "Scalar images are "
            "normalized to 0-1; the default contrast handles below are used only "
            "when a saved recipe has no channel-specific range."
        )
        display_explanation.setWordWrap(True)
        normalization_source = QWidget()
        normalization_source_layout = QHBoxLayout(normalization_source)
        normalization_source_layout.setContentsMargins(0, 0, 0, 0)
        self.normalization_edit = QLineEdit()
        self.normalization_edit.setPlaceholderText(
            "Optional Nimbus normalization JSON or Marker/Value CSV"
        )
        self.choose_normalization_button = QPushButton("Choose...")
        self.load_normalization_button = QPushButton("Load into editor")
        self._setup_status_labels["normalization"] = QLabel("○ Optional")
        self._setup_status_labels["normalization"].setObjectName("sbtInputStatus")
        self._setup_status_labels["normalization"].setMinimumWidth(128)
        normalization_source_layout.addWidget(
            self._setup_status_labels["normalization"]
        )
        normalization_source_layout.addWidget(self.normalization_edit, 1)
        normalization_source_layout.addWidget(self.choose_normalization_button)
        normalization_source_layout.addWidget(self.load_normalization_button)
        self.normalization_table = QTableWidget(0, 2)
        self.normalization_table.setHorizontalHeaderLabels(["Marker", "Value"])
        self.normalization_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.normalization_table.setMaximumHeight(190)
        normalization_table_actions = QHBoxLayout()
        self.add_normalization_row_button = QPushButton("Add marker")
        self.remove_normalization_row_button = QPushButton("Remove selected marker")
        self.advanced_normalization_check = QCheckBox("Show technical JSON preview")
        normalization_table_actions.addWidget(self.add_normalization_row_button)
        normalization_table_actions.addWidget(self.remove_normalization_row_button)
        normalization_table_actions.addStretch(1)
        normalization_table_actions.addWidget(self.advanced_normalization_check)
        self.normalization_json_edit = QTextEdit("{}")
        self.normalization_json_edit.setReadOnly(True)
        self.normalization_json_edit.setMaximumHeight(190)
        self.normalization_json_edit.hide()
        normalization_actions = QHBoxLayout()
        self.validate_normalization_button = QPushButton("Validate edited values")
        self.save_normalization_button = QPushButton("Save edited copy into experiment")
        normalization_actions.addWidget(self.validate_normalization_button)
        normalization_actions.addWidget(self.save_normalization_button)
        normalization_actions.addStretch(1)
        display_defaults = QWidget()
        display_defaults_layout = QGridLayout(display_defaults)
        display_defaults_layout.setContentsMargins(0, 0, 0, 0)
        self.display_quantile_spin = QDoubleSpinBox()
        self.display_quantile_spin.setRange(0.001, 1.0)
        self.display_quantile_spin.setDecimals(4)
        self.display_quantile_spin.setSingleStep(0.001)
        self.display_quantile_spin.setValue(0.999)
        self.display_minimum_pixel_spin = QDoubleSpinBox()
        self.display_minimum_pixel_spin.setRange(0.0, 1_000_000.0)
        self.display_minimum_pixel_spin.setDecimals(4)
        self.display_minimum_pixel_spin.setValue(0.1)
        self.display_lower_contrast_spin = QDoubleSpinBox()
        self.display_lower_contrast_spin.setRange(0.0, 1.0)
        self.display_lower_contrast_spin.setDecimals(3)
        self.display_lower_contrast_spin.setSingleStep(0.01)
        self.display_lower_contrast_spin.setValue(0.0)
        self.display_upper_contrast_spin = QDoubleSpinBox()
        self.display_upper_contrast_spin.setRange(0.0, 1.0)
        self.display_upper_contrast_spin.setDecimals(3)
        self.display_upper_contrast_spin.setSingleStep(0.01)
        self.display_upper_contrast_spin.setValue(1.0)
        display_defaults_layout.addWidget(QLabel("Fallback quantile"), 0, 0)
        display_defaults_layout.addWidget(self.display_quantile_spin, 0, 1)
        display_defaults_layout.addWidget(QLabel("Minimum pixel count"), 0, 2)
        display_defaults_layout.addWidget(self.display_minimum_pixel_spin, 0, 3)
        display_defaults_layout.addWidget(QLabel("Default contrast lower"), 1, 0)
        display_defaults_layout.addWidget(self.display_lower_contrast_spin, 1, 1)
        display_defaults_layout.addWidget(QLabel("Default contrast upper"), 1, 2)
        display_defaults_layout.addWidget(self.display_upper_contrast_spin, 1, 3)
        self.normalization_status_label = QLabel(
            "No fixed normalization mapping is loaded; unmatched channels use "
            "the fallback quantile."
        )
        self.normalization_status_label.setWordWrap(True)
        display_layout.addWidget(display_explanation)
        display_layout.addWidget(normalization_source)
        display_layout.addWidget(self.normalization_table)
        display_layout.addLayout(normalization_table_actions)
        display_layout.addWidget(self.normalization_json_edit)
        display_layout.addLayout(normalization_actions)
        display_layout.addWidget(display_defaults)
        display_layout.addWidget(self.normalization_status_label)
        setup_layout.addWidget(display_group)

        self.classification_setup_widget = QWidget()
        classification_setup_layout = QVBoxLayout(self.classification_setup_widget)
        classification_setup_layout.setContentsMargins(0, 0, 0, 0)

        scope_group = workflow_group(
            "5. Classification cell scope", "setup", "Cell scope"
        )
        scope_grid = QGridLayout(scope_group)
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("All cells", "all_cells")
        self.scope_combo.addItem("Selected adata.obs values", "obs_values")
        self.obs_combo = QComboBox()
        self.value_list = QListWidget()
        self.value_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.value_list.setMaximumHeight(105)
        self.load_adata_button = QPushButton("Load AnnData selectors")
        self.preview_button = QPushButton("Validate integrity and preview cohort")
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
        classification_setup_layout.addWidget(scope_group)

        trial_group = workflow_group(
            "6. Classification mode and feature-discovery ROIs",
            "setup",
            "Full experiment or Feature Discovery Trial",
        )
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
        self.trial_roi_strategy_combo.addItem("Largest eligible-cell ROIs", "largest")
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
        classification_setup_layout.addWidget(trial_group)

        class_group = workflow_group(
            "7. Classification classes (2–8)", "setup", "Classes"
        )
        class_layout = QVBoxLayout(class_group)
        self.class_table = QTableWidget(0, 5)
        self.class_table.setHorizontalHeaderLabels(
            [
                "Stable ID",
                "Name",
                "Colour (double-click)",
                "Shortcut",
                "Mask disposition",
            ]
        )
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        class_colour_help = QLabel(
            "Each class colour is shown as a swatch. Double-click a swatch, or "
            "select its row and use the colour-picker button."
        )
        class_colour_help.setWordWrap(True)
        class_buttons = QHBoxLayout()
        self.add_class_button = QPushButton("Add class")
        self.remove_class_button = QPushButton("Remove selected class")
        self.pick_class_colour_button = QPushButton("🎨 Pick selected colour…")
        self.pick_class_colour_button.setToolTip(
            "Choose the display colour for the selected class."
        )
        self.qc_template_button = QPushButton("Segmentation QC template")
        self.apply_classes_button = QPushButton("Apply class edits")
        class_buttons.addWidget(self.add_class_button)
        class_buttons.addWidget(self.remove_class_button)
        class_buttons.addWidget(self.pick_class_colour_button)
        class_buttons.addWidget(self.qc_template_button)
        class_buttons.addWidget(self.apply_classes_button)
        class_layout.addWidget(class_colour_help)
        class_layout.addWidget(self.class_table)
        class_layout.addLayout(class_buttons)
        classification_setup_layout.addWidget(class_group)
        setup_layout.addWidget(self.classification_setup_widget)
        add_tab(setup, "⚙ Setup", "setup")

        # Feature Building
        feature_builder = QWidget()
        feature_builder_layout = QVBoxLayout(feature_builder)

        source_group = workflow_group(
            "1. Imported feature sources", "feature_building", "Imported sources"
        )
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

        channel_group = workflow_group(
            "2. IMC channels", "feature_building", "IMC channels"
        )
        channel_layout = QVBoxLayout(channel_group)
        channel_explanation = QLabel(
            "Select channels from the AnnData panel and discovered ROI images. "
            "Blank selection means every channel discovered consistently by the worker."
        )
        channel_explanation.setWordWrap(True)
        self.feature_channel_list = QListWidget()
        self.feature_channel_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.feature_channel_list.setMaximumHeight(150)
        channel_actions = QHBoxLayout()
        self.refresh_feature_channels_button = QPushButton("Refresh available channels")
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

        feature_group = workflow_group(
            "3. Synthetic feature recipe", "feature_building", "Synthetic features"
        )
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
        self.feature_normalization_summary = QLabel(
            "Configured in Setup. The experiment-backed copy is also used by "
            "synthetic feature extraction."
        )
        self.feature_normalization_summary.setWordWrap(True)
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
        self.feature_tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
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
        feature_form.addRow(
            "Nimbus normalization",
            self.feature_normalization_summary,
        )
        feature_form.addRow("Enabled families", feature_checks)
        feature_form.addRow("Specific features", self.feature_tree)
        feature_form.addRow("Selection summary", self.feature_selection_summary)
        feature_form.addRow(
            f"Local workers (available: {worker_resolution.cpu_limit})",
            self.workers_spin,
        )
        feature_builder_layout.addWidget(feature_group)

        progress_group = workflow_group(
            "4. Build progress and process health",
            "feature_building",
            "Execution and progress",
        )
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
        self.feature_process_health_label = QLabel("Python process: not running")
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

        readiness_group = workflow_group(
            "1. Trial readiness", "feature_refinement", "Readiness"
        )
        readiness_layout = QVBoxLayout(readiness_group)
        self.refinement_scope_label = QLabel(
            "Create or load a Feature Discovery Trial first."
        )
        self.refinement_scope_label.setWordWrap(True)
        self.refinement_class_table = QTableWidget(0, 4)
        self.refinement_class_table.setHorizontalHeaderLabels(
            ["Class", "Confirmed", "Represented ROIs", "Readiness"]
        )
        self.refinement_class_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.refinement_class_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.refinement_class_table.setMaximumHeight(190)
        readiness_layout.addWidget(self.refinement_scope_label)
        readiness_layout.addWidget(self.refinement_class_table)
        refinement_layout.addWidget(readiness_group)

        refine_controls_group = workflow_group(
            "2. Grouped evaluation settings", "feature_refinement", "Evaluation"
        )
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
        refine_controls.addRow("Held-out permutation repeats", self.refine_repeats_spin)
        refine_controls.addRow(
            "Maximum allowed missing fraction", self.refine_missing_spin
        )
        refine_controls.addRow(
            "Redundancy correlation threshold", self.refine_correlation_spin
        )
        refinement_layout.addWidget(refine_controls_group)

        refinement_progress_group = workflow_group(
            "3. Analysis progress", "feature_refinement", "Analysis progress"
        )
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

        refinement_results_group = workflow_group(
            "4. Recommended feature set",
            "feature_refinement",
            "Choosing and promoting features",
        )
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
        self.refinement_results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.refinement_results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.refinement_results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.refinement_results_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.refinement_results_table.setMinimumHeight(300)
        recommendation_actions = QHBoxLayout()
        self.select_recommended_button = QPushButton("Restore recommended checks")
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

        explore_metadata_group = workflow_group(
            "Current ROI sample metadata",
            "explore",
            "ROI sample metadata",
        )
        explore_metadata_layout = QVBoxLayout(explore_metadata_group)
        self.explore_roi_metadata_summary = QLabel(
            "Load an ROI to show automatically detected sample metadata."
        )
        self.explore_roi_metadata_summary.setWordWrap(True)
        self.explore_roi_metadata_table = QTableWidget(0, 2)
        self.explore_roi_metadata_table.setHorizontalHeaderLabels(
            ["Metadata field", "Value"]
        )
        self.explore_roi_metadata_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self.explore_roi_metadata_table.verticalHeader().setVisible(False)
        self.explore_roi_metadata_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.explore_roi_metadata_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.explore_roi_metadata_table.setMaximumHeight(180)
        explore_metadata_layout.addWidget(self.explore_roi_metadata_summary)
        explore_metadata_layout.addWidget(self.explore_roi_metadata_table)
        explore_layout.addWidget(explore_metadata_group)

        layer_actions = QHBoxLayout()
        self.hide_all_layers_button = QPushButton("Hide all layers")
        self.show_all_layers_button = QPushButton("Show all layers")
        self.delete_all_layers_button = QPushButton("Delete all layers")
        layer_actions.addWidget(self.hide_all_layers_button)
        layer_actions.addWidget(self.show_all_layers_button)
        layer_actions.addWidget(self.delete_all_layers_button)
        explore_layout.addLayout(layer_actions)

        reload_recipe_group = workflow_group(
            "Layers re-added when the ROI changes",
            "explore",
            "ROI reload recipe",
        )
        reload_recipe_layout = QVBoxLayout(reload_recipe_group)
        self.reload_recipe_help = QLabel(
            "Save multiple named recipes and optionally assign each one a unique "
            "F1–F12 shortcut. The list below is the exact active ROI reload "
            "recipe. Classifier layers are "
            "regenerated from labels and scores, while their visible/hidden "
            "state, opacity, contour style, and image contrast limits are "
            "replayed from this list."
        )
        self.reload_recipe_help.setWordWrap(True)
        preset_selector = QWidget()
        preset_selector_layout = QHBoxLayout(preset_selector)
        preset_selector_layout.setContentsMargins(0, 0, 0, 0)
        self.recipe_preset_combo = QComboBox()
        self.recipe_preset_combo.setMinimumWidth(230)
        self.load_recipe_preset_button = QPushButton("Load selected recipe")
        preset_selector_layout.addWidget(self.recipe_preset_combo, 1)
        preset_selector_layout.addWidget(self.load_recipe_preset_button)
        preset_editor = QWidget()
        preset_editor_layout = QHBoxLayout(preset_editor)
        preset_editor_layout.setContentsMargins(0, 0, 0, 0)
        self.recipe_preset_name_edit = QLineEdit()
        self.recipe_preset_name_edit.setPlaceholderText("e.g. T-cell verification")
        self.recipe_preset_shortcut_combo = QComboBox()
        self.recipe_preset_shortcut_combo.addItem("No F-key", None)
        for shortcut in EXPLORE_RECIPE_FUNCTION_KEYS:
            self.recipe_preset_shortcut_combo.addItem(shortcut, shortcut)
        preset_editor_layout.addWidget(QLabel("Name"))
        preset_editor_layout.addWidget(self.recipe_preset_name_edit, 1)
        preset_editor_layout.addWidget(QLabel("Shortcut"))
        preset_editor_layout.addWidget(self.recipe_preset_shortcut_combo)
        preset_actions = QHBoxLayout()
        self.save_new_recipe_preset_button = QPushButton(
            "Save current view as new recipe"
        )
        self.update_recipe_preset_button = QPushButton(
            "Update selected recipe from current view"
        )
        self.delete_recipe_preset_button = QPushButton("Delete selected recipe…")
        preset_actions.addWidget(self.save_new_recipe_preset_button)
        preset_actions.addWidget(self.update_recipe_preset_button)
        preset_actions.addWidget(self.delete_recipe_preset_button)
        self.import_recipe_preset_button = QPushButton("Import recipe JSON...")
        self.export_recipe_preset_button = QPushButton("Export selected JSON...")
        preset_actions.addWidget(self.import_recipe_preset_button)
        preset_actions.addWidget(self.export_recipe_preset_button)
        self.active_recipe_preset_label = QLabel(
            "Working view is not saved as a named recipe."
        )
        self.active_recipe_preset_label.setWordWrap(True)
        self.reload_recipe_list = QListWidget()
        self.reload_recipe_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
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
        reload_recipe_layout.addWidget(preset_selector)
        reload_recipe_layout.addWidget(preset_editor)
        reload_recipe_layout.addLayout(preset_actions)
        reload_recipe_layout.addWidget(self.active_recipe_preset_label)
        reload_recipe_layout.addWidget(self.reload_recipe_list)
        reload_recipe_layout.addLayout(reload_recipe_actions)
        explore_layout.addWidget(reload_recipe_group)

        overlay_group = workflow_group(
            "AnnData overlays and population-to-cohort transfer",
            "explore",
            "AnnData and population overlays",
        )
        overlay_form = QFormLayout(overlay_group)
        self.overlay_obs_combo = QComboBox()
        self.overlay_full_dataset_check = QCheckBox(
            "Include cells outside the classification cohort"
        )
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
        overlay_form.addRow(
            "Categorical or numeric observation", self.overlay_obs_combo
        )
        overlay_form.addRow("Overlay scope", self.overlay_full_dataset_check)
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
        image_group = workflow_group(
            "Raw, extra, greyscale, and multicolour images",
            "explore",
            "Image channels",
        )
        image_layout = QVBoxLayout(image_group)
        self.image_coverage_label = QLabel("No ROI images discovered yet.")
        self.image_coverage_label.setWordWrap(True)
        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.channel_list.setMaximumHeight(120)
        image_actions = QHBoxLayout()
        self.load_channels_button = QPushButton("Load selected greyscale")
        self.load_six_colour_button = QPushButton("Load selected as R/G/B/C/Y/M")
        self.load_rgb_button = QPushButton("Load first three selected as RGB")
        image_actions.addWidget(self.load_channels_button)
        image_actions.addWidget(self.load_six_colour_button)
        image_actions.addWidget(self.load_rgb_button)
        image_layout.addWidget(self.image_coverage_label)
        image_layout.addWidget(self.channel_list)
        image_layout.addLayout(image_actions)
        explore_layout.addWidget(image_group)
        add_tab(explore, "🔬 Explore", "explore")
        self.explore_tab_index = self.tabs.count() - 1

        # Population QC
        population_qc = QWidget()
        population_qc_layout = QVBoxLayout(population_qc)
        population_qc_intro = QLabel(
            "Review one AnnData population with a compact RGB image recipe and "
            "population outline. These views are ordinary Explore recipes, so "
            "their colours, contrast ranges, ROI replay, and viewed-ROI history "
            "are shared with Explore."
        )
        population_qc_intro.setWordWrap(True)
        population_qc_layout.addWidget(population_qc_intro)

        population_qc_selection_group = workflow_group(
            "1. Population to review",
            "population_qc",
            "Population selection",
        )
        population_qc_selection_form = QFormLayout(population_qc_selection_group)
        self.population_qc_obs_combo = QComboBox()
        self.population_qc_population_combo = QComboBox()
        self.population_qc_contour_spin = QSpinBox()
        self.population_qc_contour_spin.setRange(0, 20)
        self.population_qc_contour_spin.setValue(
            self.explore_review_state.population_qc_contour_width
        )
        self.population_qc_contour_spin.setSuffix(" px")
        self.population_qc_contour_spin.setToolTip(
            "One outline width shared by every Population QC population. "
            "New workspaces start at 1 px; set 0 to show filled labels."
        )
        population_qc_selection_form.addRow(
            "Population observation", self.population_qc_obs_combo
        )
        population_qc_selection_form.addRow(
            "Population", self.population_qc_population_combo
        )
        population_qc_selection_form.addRow(
            "Outline width for all populations", self.population_qc_contour_spin
        )
        population_qc_layout.addWidget(population_qc_selection_group)

        population_qc_metadata_group = workflow_group(
            "Current ROI sample metadata",
            "population_qc",
            "ROI sample metadata",
        )
        population_qc_metadata_layout = QVBoxLayout(population_qc_metadata_group)
        self.population_qc_roi_metadata_summary = QLabel(
            "Open an ROI to show automatically detected sample metadata."
        )
        self.population_qc_roi_metadata_summary.setWordWrap(True)
        self.population_qc_roi_metadata_table = QTableWidget(0, 2)
        self.population_qc_roi_metadata_table.setHorizontalHeaderLabels(
            ["Metadata field", "Value"]
        )
        self.population_qc_roi_metadata_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self.population_qc_roi_metadata_table.verticalHeader().setVisible(False)
        self.population_qc_roi_metadata_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.population_qc_roi_metadata_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.population_qc_roi_metadata_table.setMaximumHeight(180)
        population_qc_metadata_layout.addWidget(
            self.population_qc_roi_metadata_summary
        )
        population_qc_metadata_layout.addWidget(self.population_qc_roi_metadata_table)
        population_qc_layout.addWidget(population_qc_metadata_group)

        population_qc_rgb_group = workflow_group(
            "2. RGB verification recipe",
            "population_qc",
            "RGB verification recipe",
        )
        population_qc_rgb_layout = QGridLayout(population_qc_rgb_group)
        population_qc_rgb_layout.addWidget(QLabel("Colour"), 0, 0)
        population_qc_rgb_layout.addWidget(QLabel("Image channel"), 0, 1)
        population_qc_rgb_layout.addWidget(QLabel("Contrast lower"), 0, 2)
        population_qc_rgb_layout.addWidget(QLabel("Contrast upper"), 0, 3)
        self.population_qc_marker_combos = {}
        self.population_qc_lower_spins = {}
        self.population_qc_upper_spins = {}
        for row, (colour, display_name) in enumerate(
            (("red", "Red"), ("green", "Green"), ("blue", "Blue")),
            start=1,
        ):
            colour_label = QLabel(f"<b>{display_name}</b>")
            colour_label.setStyleSheet(f"color: {colour};")
            marker_combo = QComboBox()
            lower_spin = QDoubleSpinBox()
            lower_spin.setRange(0.0, 1.0)
            lower_spin.setDecimals(3)
            lower_spin.setSingleStep(0.01)
            lower_spin.setValue(0.0)
            upper_spin = QDoubleSpinBox()
            upper_spin.setRange(0.0, 1.0)
            upper_spin.setDecimals(3)
            upper_spin.setSingleStep(0.01)
            upper_spin.setValue(1.0)
            self.population_qc_marker_combos[colour] = marker_combo
            self.population_qc_lower_spins[colour] = lower_spin
            self.population_qc_upper_spins[colour] = upper_spin
            population_qc_rgb_layout.addWidget(colour_label, row, 0)
            population_qc_rgb_layout.addWidget(marker_combo, row, 1)
            population_qc_rgb_layout.addWidget(lower_spin, row, 2)
            population_qc_rgb_layout.addWidget(upper_spin, row, 3)
        self._population_qc_last_setup_defaults = (
            float(self.display_lower_contrast_spin.value()),
            float(self.display_upper_contrast_spin.value()),
        )
        self.population_qc_contrast_defaults_label = QLabel()
        self.population_qc_contrast_defaults_label.setWordWrap(True)
        population_qc_rgb_layout.addWidget(
            self.population_qc_contrast_defaults_label, 4, 0, 1, 4
        )
        population_qc_rgb_actions = QWidget()
        population_qc_rgb_actions_layout = QHBoxLayout(population_qc_rgb_actions)
        population_qc_rgb_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.suggest_population_qc_markers_button = QPushButton(
            "Suggest top three markers"
        )
        self.save_population_qc_recipe_button = QPushButton(
            "Save RGB recipe for population"
        )
        self.load_population_qc_view_button = QPushButton("Load population view")
        self.reset_population_qc_contrast_button = QPushButton(
            "Use Setup contrast defaults"
        )
        population_qc_rgb_actions_layout.addWidget(
            self.suggest_population_qc_markers_button
        )
        population_qc_rgb_actions_layout.addWidget(
            self.save_population_qc_recipe_button
        )
        population_qc_rgb_actions_layout.addWidget(
            self.reset_population_qc_contrast_button
        )
        population_qc_rgb_actions_layout.addWidget(self.load_population_qc_view_button)
        population_qc_rgb_layout.addWidget(population_qc_rgb_actions, 5, 0, 1, 4)
        population_qc_io_actions = QWidget()
        population_qc_io_actions_layout = QHBoxLayout(population_qc_io_actions)
        population_qc_io_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.import_population_qc_csv_button = QPushButton(
            "Import legacy settings CSV..."
        )
        self.export_population_qc_csv_button = QPushButton(
            "Export Population QC settings CSV..."
        )
        population_qc_io_actions_layout.addWidget(self.import_population_qc_csv_button)
        population_qc_io_actions_layout.addWidget(self.export_population_qc_csv_button)
        population_qc_rgb_layout.addWidget(population_qc_io_actions, 6, 0, 1, 4)
        self._update_population_qc_contrast_defaults_label()
        population_qc_layout.addWidget(population_qc_rgb_group)

        population_qc_roi_group = workflow_group(
            "3. ROI sampling",
            "population_qc",
            "ROI sampling",
        )
        population_qc_roi_layout = QVBoxLayout(population_qc_roi_group)
        population_qc_ranking_controls = QWidget()
        population_qc_ranking_controls_layout = QHBoxLayout(
            population_qc_ranking_controls
        )
        population_qc_ranking_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.population_qc_roi_order_combo = QComboBox()
        self.population_qc_roi_order_combo.addItem("Top abundance", "top")
        self.population_qc_roi_order_combo.addItem("Bottom abundance", "bottom")
        self.population_qc_roi_order_combo.addItem("Random", "random")
        self.population_qc_roi_limit_spin = QSpinBox()
        self.population_qc_roi_limit_spin.setRange(1, 10_000)
        self.population_qc_roi_limit_spin.setValue(10)
        self.population_qc_random_seed_spin = QSpinBox()
        self.population_qc_random_seed_spin.setRange(0, 2_147_483_647)
        self.population_qc_random_seed_spin.setValue(0)
        self.recalculate_population_qc_rois_button = QPushButton("Recalculate ROI list")
        population_qc_ranking_controls_layout.addWidget(QLabel("Order"))
        population_qc_ranking_controls_layout.addWidget(
            self.population_qc_roi_order_combo
        )
        population_qc_ranking_controls_layout.addWidget(QLabel("Number"))
        population_qc_ranking_controls_layout.addWidget(
            self.population_qc_roi_limit_spin
        )
        population_qc_ranking_controls_layout.addWidget(QLabel("Random seed"))
        population_qc_ranking_controls_layout.addWidget(
            self.population_qc_random_seed_spin
        )
        population_qc_ranking_controls_layout.addWidget(
            self.recalculate_population_qc_rois_button
        )
        self.population_qc_roi_buttons_widget = QWidget()
        self.population_qc_roi_buttons_layout = QGridLayout(
            self.population_qc_roi_buttons_widget
        )
        self.population_qc_roi_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.population_qc_status_label = QLabel(
            "Choose a population to build its RGB recipe and ROI list."
        )
        self.population_qc_status_label.setWordWrap(True)
        population_qc_roi_layout.addWidget(population_qc_ranking_controls)
        population_qc_roi_layout.addWidget(self.population_qc_roi_buttons_widget)
        population_qc_roi_layout.addWidget(self.population_qc_status_label)
        population_qc_layout.addWidget(population_qc_roi_group)
        population_qc_layout.addStretch(1)
        add_tab(population_qc, "🧿 Population QC", "population_qc")
        self.population_qc_tab_index = self.tabs.count() - 1

        # Population naming and curation
        populations = QWidget()
        populations_layout = QVBoxLayout(populations)
        populations_intro = QLabel(
            "Turn numbered clusters into a clearly named AnnData label column. "
            "Rename or merge populations here, save, then move between Population "
            "naming, Explore, and Population QC while refining the labels."
        )
        populations_intro.setWordWrap(True)
        populations_layout.addWidget(populations_intro)

        population_workspace_group = workflow_group(
            "1. Choose the source and new label column",
            "populations",
            "1. Source workspace and drafts",
        )
        population_workspace_form = QFormLayout(population_workspace_group)
        self.curation_source_combo = QComboBox()
        self.curation_draft_combo = QComboBox()
        self.curation_derived_obs_edit = QLineEdit("population_curated")
        self.curation_derived_obs_edit.setPlaceholderText(
            "For example: population_named"
        )
        self.create_population_draft_button = QPushButton(
            "Create new label draft"
        )
        self.save_population_draft_button = QPushButton(
            "Save and update Explore / Population QC"
        )
        self.save_population_draft_button.setObjectName("sbtPrimaryActionButton")
        self.view_population_history_button = QPushButton("View history…")
        population_draft_actions = QWidget()
        population_draft_actions_layout = QHBoxLayout(population_draft_actions)
        population_draft_actions_layout.setContentsMargins(0, 0, 0, 0)
        population_draft_actions_layout.addWidget(self.create_population_draft_button)
        population_draft_actions_layout.addWidget(self.save_population_draft_button)
        population_draft_actions_layout.addWidget(self.view_population_history_button)
        self.population_naming_readiness_label = QLabel(
            "● Start by choosing a source observation and a new label-column name."
        )
        self.population_naming_readiness_label.setWordWrap(True)
        self.population_workspace_label = QLabel(
            "Load AnnData, then choose the original clustering observation."
        )
        self.population_workspace_label.setWordWrap(True)
        population_workspace_form.addRow(
            "Original source obs", self.curation_source_combo
        )
        population_workspace_form.addRow(
            "Saved naming work", self.curation_draft_combo
        )
        population_workspace_form.addRow(
            "New label column (adata.obs)", self.curation_derived_obs_edit
        )
        population_workspace_form.addRow(
            "Readiness", self.population_naming_readiness_label
        )
        population_workspace_form.addRow("", population_draft_actions)
        population_workspace_form.addRow("Source", self.population_workspace_label)
        populations_layout.addWidget(population_workspace_group)

        self.population_editor_tabs = QTabWidget()

        base_mapping_page = QWidget()
        base_mapping_layout = QVBoxLayout(base_mapping_page)
        base_mapping_help = QLabel(
            "Edit Proposed name to rename a source population. Give two or more "
            "rows exactly the same name to propose a merge; those rows are "
            "highlighted and listed in the merge preview."
        )
        base_mapping_help.setWordWrap(True)
        self.population_base_table = QTableWidget(0, 5)
        self.population_base_table.setHorizontalHeaderLabels(
            ["Source population", "Cells", "Proposed name", "Colour", "Notes"]
        )
        self.population_base_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.population_base_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.population_base_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        self.population_base_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeToContents
        )
        self.population_base_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.Stretch
        )
        self.population_base_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.population_base_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # Napari's dark stylesheet can leave Qt's alternate-base colour white
        # while retaining a light foreground, making every second row unreadable.
        self.population_base_table.setAlternatingRowColors(False)
        self.population_base_table.setMinimumHeight(300)
        base_mapping_actions = QHBoxLayout()
        self.name_selected_populations_button = QPushButton(
            "Give selected rows one name / merge"
        )
        self.colour_selected_populations_button = QPushButton(
            "Set selected rows' colour"
        )
        self.import_population_mapping_button = QPushButton(
            "Import preliminary names from CSV"
        )
        self.export_population_mapping_button = QPushButton(
            "Export editable mapping CSV"
        )
        base_mapping_actions.addWidget(self.name_selected_populations_button)
        base_mapping_actions.addWidget(self.colour_selected_populations_button)
        base_mapping_actions.addWidget(self.import_population_mapping_button)
        base_mapping_actions.addWidget(self.export_population_mapping_button)
        base_mapping_layout.addWidget(base_mapping_help)
        base_mapping_layout.addWidget(self.population_base_table)
        base_mapping_layout.addLayout(base_mapping_actions)
        self.population_editor_tabs.addTab(base_mapping_page, "Rename & merge")

        split_page = QWidget()
        split_layout = QVBoxLayout(split_page)
        split_controls = workflow_group(
            "Batch-correction-preserving Scanpy subclustering",
            "populations",
            "Subclusters",
        )
        split_form = QFormLayout(split_controls)
        split_explanation = QLabel(
            "Default: isolate the selected cells, rebuild their neighbours from "
            "adata.obsm['X_biobatchnet'], then run Leiden. BioBatchNet itself, "
            "normalization, scaling, PCA, and UMAP are not rerun."
        )
        split_explanation.setWordWrap(True)
        self.population_split_values_list = QListWidget()
        self.population_split_values_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.population_split_values_list.setMaximumHeight(125)
        self.population_neighbor_source_combo = QComboBox()
        self.population_neighbor_source_combo.addItem(
            "Rebuild neighbours from corrected obsm (recommended)",
            "rebuild_from_rep",
        )
        self.population_neighbor_source_combo.addItem(
            "Reuse an existing obsp connectivity graph",
            "existing_graph",
        )
        self.population_representation_combo = QComboBox()
        self.population_n_neighbors_spin = QSpinBox()
        self.population_n_neighbors_spin.setRange(2, 1000)
        self.population_n_neighbors_spin.setValue(15)
        self.population_adjacency_combo = QComboBox()
        self.population_graph_provenance_label = QLabel(
            "Choose the corrected representation used to rebuild neighbours."
        )
        self.population_graph_provenance_label.setWordWrap(True)
        self.population_resolution_spin = QDoubleSpinBox()
        self.population_resolution_spin.setRange(0.01, 20.0)
        self.population_resolution_spin.setDecimals(2)
        self.population_resolution_spin.setSingleStep(0.1)
        self.population_resolution_spin.setValue(0.5)
        self.population_subcluster_mode_combo = QComboBox()
        self.population_subcluster_mode_combo.addItem(
            "Subcluster each selected population separately (recommended)",
            "within_each",
        )
        self.population_subcluster_mode_combo.addItem(
            "Subcluster selected populations together", "together"
        )
        self.run_population_subcluster_button = QPushButton(
            "Run monitored subclustering"
        )
        self.cancel_population_subcluster_button = QPushButton("Cancel")
        self.cancel_population_subcluster_button.setEnabled(False)
        subcluster_actions = QWidget()
        subcluster_actions_layout = QHBoxLayout(subcluster_actions)
        subcluster_actions_layout.setContentsMargins(0, 0, 0, 0)
        subcluster_actions_layout.addWidget(self.run_population_subcluster_button)
        subcluster_actions_layout.addWidget(self.cancel_population_subcluster_button)
        self.population_subcluster_status = QLabel("No subclustering run in progress.")
        self.population_subcluster_status.setWordWrap(True)
        split_form.addRow("Safety contract", split_explanation)
        split_form.addRow("Source populations", self.population_split_values_list)
        split_form.addRow("Neighbour strategy", self.population_neighbor_source_combo)
        split_form.addRow(
            "Corrected representation", self.population_representation_combo
        )
        split_form.addRow("n_neighbors", self.population_n_neighbors_spin)
        split_form.addRow(
            "Existing connectivity graph", self.population_adjacency_combo
        )
        split_form.addRow("Input provenance", self.population_graph_provenance_label)
        split_form.addRow("Leiden resolution", self.population_resolution_spin)
        split_form.addRow("Run populations", self.population_subcluster_mode_combo)
        split_form.addRow("", subcluster_actions)
        split_form.addRow("Progress", self.population_subcluster_status)
        split_layout.addWidget(split_controls)

        self.population_components_table = QTableWidget(0, 8)
        self.population_components_table.setHorizontalHeaderLabels(
            [
                "Parent",
                "Method",
                "Raw component",
                "Cells",
                "Proposed name",
                "Colour",
                "Run",
                "Notes",
            ]
        )
        for column in (0, 1, 2, 3, 5, 6):
            self.population_components_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeToContents
            )
        self.population_components_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.Stretch
        )
        self.population_components_table.horizontalHeader().setSectionResizeMode(
            7, QHeaderView.Stretch
        )
        self.population_components_table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self.population_components_table.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.population_components_table.setAlternatingRowColors(False)
        self.population_components_table.setMinimumHeight(250)
        split_table_actions = QHBoxLayout()
        self.import_population_components_button = QPushButton(
            "Import image/other cell-level assignments"
        )
        self.import_current_classifier_components_button = QPushButton(
            "Use current classifier assignments"
        )
        self.name_selected_components_button = QPushButton(
            "Give selected components one name / merge"
        )
        self.colour_selected_components_button = QPushButton(
            "Set selected components' colour"
        )
        self.remove_population_components_button = QPushButton(
            "Remove selected split components"
        )
        split_table_actions.addWidget(self.import_population_components_button)
        split_table_actions.addWidget(self.import_current_classifier_components_button)
        split_table_actions.addWidget(self.name_selected_components_button)
        split_table_actions.addWidget(self.colour_selected_components_button)
        split_table_actions.addWidget(self.remove_population_components_button)
        split_layout.addWidget(QLabel("Editable split components"))
        split_layout.addWidget(self.population_components_table)
        split_layout.addLayout(split_table_actions)
        self.population_editor_tabs.addTab(split_page, "Subclusters")

        preview_page = QWidget()
        preview_layout = QVBoxLayout(preview_page)
        self.population_merge_preview = QTextEdit()
        self.population_merge_preview.setReadOnly(True)
        self.population_merge_preview.setMinimumHeight(220)
        population_apply_group = workflow_group(
            "Use the current labels in the rest of NapariSBT",
            "populations",
            "Applying a draft",
        )
        population_apply_form = QFormLayout(population_apply_group)
        population_apply_help = QLabel(
            "Saving in box 1 updates the live AnnData label column and refreshes "
            "Explore and Population QC automatically. Use these buttons only to "
            "open the current overlay or export a separate AnnData copy."
        )
        population_apply_help.setWordWrap(True)
        self.population_overwrite_check = QCheckBox(
            "Advanced: allow replacing an unrelated existing obs with this name"
        )
        self.show_curated_population_overlay_button = QPushButton(
            "Open current labels in Explore"
        )
        self.open_population_scanpy_plotting_button = QPushButton(
            "Open these labels in Scanpy plotting"
        )
        self.export_curated_anndata_button = QPushButton(
            "Export all live curated observations to a new AnnData copy..."
        )
        apply_actions = QWidget()
        apply_actions_layout = QHBoxLayout(apply_actions)
        apply_actions_layout.setContentsMargins(0, 0, 0, 0)
        apply_actions_layout.addWidget(self.show_curated_population_overlay_button)
        apply_actions_layout.addWidget(self.open_population_scanpy_plotting_button)
        apply_actions_layout.addStretch(1)
        population_apply_form.addRow("", population_apply_help)
        population_apply_form.addRow("Overwrite guard", self.population_overwrite_check)
        population_apply_form.addRow("", apply_actions)
        population_apply_form.addRow("", self.export_curated_anndata_button)
        preview_layout.addWidget(QLabel("Effective labels and explicit merges"))
        preview_layout.addWidget(self.population_merge_preview)
        preview_layout.addWidget(population_apply_group)

        self.population_editor_tabs.addTab(preview_page, "Preview & QC")

        # Detailed audit data remains on disk, but is deliberately kept out of the
        # main naming loop. The compact history is shown only when requested.
        self.population_provenance_text = QTextEdit(populations)
        self.population_provenance_text.setReadOnly(True)
        self.population_provenance_text.hide()

        populations_layout.addWidget(self.population_editor_tabs)
        add_tab(populations, "🏷️ Population naming", "populations")

        # Scanpy plotting. Qt is imported lazily with the GUI and the reusable
        # data/figure logic remains in ``scanpy_plotting.py``.
        from .scanpy_plotting_ui import ScanpyPlottingPanel

        self.scanpy_plotting_panel = ScanpyPlottingPanel(
            group_factory=workflow_group,
            generate_callback=self._guard(self.generate_scanpy_plot),
            refresh_callback=self._guard(self.refresh_scanpy_plotting_choices),
            focus_callback=self._guard(
                self.focus_scanpy_plot_window,
                pass_signal_args=True,
            ),
            close_callback=self._guard(
                self.close_scanpy_plot_window,
                pass_signal_args=True,
            ),
            close_all_callback=self._guard(self.close_all_scanpy_plot_windows),
        )
        add_tab(
            self.scanpy_plotting_panel.widget,
            "📊 Scanpy plotting",
            "scanpy_plotting",
        )
        self.scanpy_plotting_tab_index = self.tabs.count() - 1

        # Classify
        classify = QWidget()
        classify_layout = QVBoxLayout(classify)
        classify_intro = QLabel(
            "Work left to right: annotate cells, train and review the raw model "
            "predictions, then apply explicit decision thresholds to create final "
            "cell identities and export them."
        )
        classify_intro.setWordWrap(True)
        classify_layout.addWidget(classify_intro)
        self.classify_workflow_tabs = QTabWidget()
        classify_layout.addWidget(self.classify_workflow_tabs)

        annotation_page = QWidget()
        annotation_page_layout = QVBoxLayout(annotation_page)
        selection_group = workflow_group(
            "1. Selected-cell annotation and proposals",
            "classify",
            "Cell annotation",
        )
        selection_form = QFormLayout(selection_group)
        self.selected_cell_label = QLabel("No cohort cell selected")
        self.cell_picking_help = QLabel(
            "Click any eligible cell in the viewer while this Classify tab is "
            "active. The selected click action is applied using the current "
            "class. The classification_cohort layer may remain hidden and does "
            "not need to be selected. Clear proposed removes only reversible "
            "proposals and will not erase confirmed labels."
        )
        self.cell_picking_help.setWordWrap(True)
        self.class_combo = QComboBox()
        click_behavior_widget = QWidget()
        click_behavior_layout = QGridLayout(click_behavior_widget)
        click_behavior_layout.setContentsMargins(0, 0, 0, 0)
        self.click_behavior_group = QButtonGroup(click_behavior_widget)
        self.click_behavior_radios = {}
        for index, (behavior, text) in enumerate(
            (
                ("select", "Select only"),
                ("proposed", "Set proposed on click"),
                ("confirmed", "Set confirmed on click"),
                ("clear_proposed", "Clear proposed on click"),
            )
        ):
            radio = QRadioButton(text)
            radio.setProperty("napari_sbt_click_behavior", behavior)
            self.click_behavior_group.addButton(radio)
            self.click_behavior_radios[behavior] = radio
            click_behavior_layout.addWidget(radio, index // 2, index % 2)
        self.click_behavior_radios["proposed"].setChecked(True)
        click_behavior_layout.setColumnStretch(2, 1)
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
        annotation_layout = QGridLayout(annotation_buttons)
        annotation_layout.setContentsMargins(0, 0, 0, 0)
        self.propose_button = QPushButton("Set proposed")
        self.confirm_button = QPushButton("Set confirmed")
        self.clear_proposed_button = QPushButton("Clear proposed")
        self.clear_proposed_button.setToolTip(
            "Remove the selected cell's proposed label; confirmed labels are protected."
        )
        self.clear_all_proposals_button = QPushButton("Clear all proposals (all ROIs)…")
        self.clear_all_proposals_button.setToolTip(
            "Remove every proposed label in this experiment after confirmation; "
            "confirmed labels are protected."
        )
        self.confirm_proposed_button = QPushButton("Confirm all proposals")
        self.mark_reviewed_button = QPushButton("Mark current ROI reviewed")
        self.seed_obs_button = QPushButton(
            "Seed matching classes as proposals from overlay observation"
        )
        annotation_layout.addWidget(self.propose_button, 0, 0)
        annotation_layout.addWidget(self.confirm_button, 0, 1)
        annotation_layout.addWidget(self.clear_proposed_button, 0, 2)
        annotation_layout.addWidget(self.confirm_proposed_button, 1, 0, 1, 2)
        annotation_layout.addWidget(self.mark_reviewed_button, 1, 2)
        annotation_layout.addWidget(self.clear_all_proposals_button, 2, 0, 1, 3)
        selection_form.addRow("Cell", self.selected_cell_label)
        selection_form.addRow("Picking", self.cell_picking_help)
        selection_form.addRow("Class", self.class_combo)
        selection_form.addRow("Click action", click_behavior_widget)
        selection_form.addRow("", annotation_buttons)
        selection_form.addRow("Label tally", self.class_tally_table)
        selection_form.addRow("", self.classifier_display_button)
        selection_form.addRow("", self.seed_obs_button)
        annotation_page_layout.addWidget(selection_group)
        annotation_page_layout.addStretch(1)
        self.classify_workflow_tabs.addTab(annotation_page, "1. Annotate")

        prediction_page = QWidget()
        prediction_page_layout = QVBoxLayout(prediction_page)
        model_group = workflow_group(
            "2. Model and active-learning queues",
            "classify",
            "Models and active-learning queues",
        )
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
        self.queue_review_combo.addItems(["Unlabelled", "Proposed", "Confirmed", "All"])
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
        self.show_probability_button = QPushButton("Show selected-class probability")
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
        model_form.addRow("Bulk-proposal minimum confidence", self.confidence_spin)
        model_form.addRow("", self.bulk_propose_button)
        model_form.addRow("Probability class", self.probability_class_combo)
        model_form.addRow("", self.show_probability_button)
        prediction_page_layout.addWidget(model_group)

        prediction_display_group = workflow_group(
            "3. Predicted-class display controls",
            "classify",
            "Prediction display controls",
        )
        prediction_display_form = QFormLayout(prediction_display_group)
        prediction_display_explanation = QLabel(
            "This range changes only which raw argmax predictions are visible in "
            "the predicted_classes layer. It does not alter scores or decide final "
            "cell identities."
        )
        prediction_display_explanation.setWordWrap(True)
        prediction_range = QWidget()
        prediction_range_layout = QHBoxLayout(prediction_range)
        prediction_range_layout.setContentsMargins(0, 0, 0, 0)
        self.prediction_review_min_confidence_spin = QDoubleSpinBox()
        self.prediction_review_min_confidence_spin.setRange(0, 1)
        self.prediction_review_min_confidence_spin.setSingleStep(0.05)
        self.prediction_review_min_confidence_spin.setValue(0)
        self.prediction_review_max_confidence_spin = QDoubleSpinBox()
        self.prediction_review_max_confidence_spin.setRange(0, 1)
        self.prediction_review_max_confidence_spin.setSingleStep(0.05)
        self.prediction_review_max_confidence_spin.setValue(1)
        prediction_range_layout.addWidget(QLabel("Minimum"))
        prediction_range_layout.addWidget(self.prediction_review_min_confidence_spin)
        prediction_range_layout.addWidget(QLabel("Maximum"))
        prediction_range_layout.addWidget(self.prediction_review_max_confidence_spin)
        self.apply_prediction_review_button = QPushButton(
            "Apply range to predicted-class layer"
        )
        self.reset_prediction_review_button = QPushButton("Show all predictions")
        prediction_display_actions = QWidget()
        prediction_display_actions_layout = QHBoxLayout(prediction_display_actions)
        prediction_display_actions_layout.setContentsMargins(0, 0, 0, 0)
        prediction_display_actions_layout.addWidget(self.apply_prediction_review_button)
        prediction_display_actions_layout.addWidget(self.reset_prediction_review_button)
        self.prediction_review_summary_label = QLabel(
            "Score the cohort to review predicted-class coverage."
        )
        self.prediction_review_summary_label.setWordWrap(True)
        prediction_display_form.addRow(prediction_display_explanation)
        prediction_display_form.addRow("Visible confidence range", prediction_range)
        prediction_display_form.addRow("", prediction_display_actions)
        prediction_display_form.addRow(
            "Layer coverage", self.prediction_review_summary_label
        )
        prediction_page_layout.addWidget(prediction_display_group)
        prediction_page_layout.addStretch(1)
        self.classify_workflow_tabs.addTab(
            prediction_page, "2. Train & review predictions"
        )

        finalize_page = QWidget()
        finalize_page_layout = QVBoxLayout(finalize_page)
        final_group = workflow_group(
            "4. Final identity decision rules",
            "classify",
            "Final identities and export",
        )
        final_form = QFormLayout(final_group)
        final_explanation = QLabel(
            "A model prediction becomes a final identity only when it passes all "
            "three rules below. Confirmed labels always override the model; proposed "
            "labels are never final. Raw scores remain available if you change these "
            "rules later."
        )
        final_explanation.setWordWrap(True)
        self.final_min_confidence_spin = QDoubleSpinBox()
        self.final_min_confidence_spin.setRange(0, 1)
        self.final_min_confidence_spin.setSingleStep(0.05)
        self.final_min_confidence_spin.setValue(0.9)
        self.final_max_uncertainty_spin = QDoubleSpinBox()
        self.final_max_uncertainty_spin.setRange(0, 1)
        self.final_max_uncertainty_spin.setSingleStep(0.05)
        self.final_max_uncertainty_spin.setValue(1.0)
        self.final_min_margin_spin = QDoubleSpinBox()
        self.final_min_margin_spin.setRange(0, 1)
        self.final_min_margin_spin.setSingleStep(0.05)
        self.final_min_margin_spin.setValue(0.0)
        self.create_final_identities_button = QPushButton(
            "Create / refresh final cell identities"
        )
        self.create_final_identities_button.setObjectName("sbtPrimaryActionButton")
        self.final_identity_summary_label = QLabel(
            "Not created. Set the rules, then create final identities before export."
        )
        self.final_identity_summary_label.setWordWrap(True)
        final_form.addRow(final_explanation)
        final_form.addRow("Minimum model confidence", self.final_min_confidence_spin)
        final_form.addRow("Maximum normalized entropy", self.final_max_uncertainty_spin)
        final_form.addRow(
            "Minimum top-two probability margin", self.final_min_margin_spin
        )
        final_form.addRow("", self.create_final_identities_button)
        final_form.addRow("Final identity status", self.final_identity_summary_label)
        finalize_page_layout.addWidget(final_group)

        final_export_group = workflow_group(
            "5. Export final identities",
            "classify",
            "Final identities and export",
        )
        final_export_form = QFormLayout(final_export_group)
        self.assignment_path_edit = QLineEdit(
            str(self.project_root / "napari_sbt_final_identities.csv")
        )
        self.annotated_path_edit = QLineEdit(
            str(self.project_root / "napari_sbt_annotated.h5ad")
        )
        self.export_assignments_button = QPushButton("Export final identities CSV")
        self.export_adata_button = QPushButton("Export annotated AnnData copy")
        self.apply_live_adata_button = QPushButton(
            "Apply to live AnnData object (memory only)…"
        )
        final_export_form.addRow("CSV/Parquet destination", self.assignment_path_edit)
        final_export_form.addRow(
            "Annotated AnnData destination", self.annotated_path_edit
        )
        final_export_form.addRow("", self.export_assignments_button)
        final_export_form.addRow("", self.export_adata_button)
        final_export_form.addRow("", self.apply_live_adata_button)
        finalize_page_layout.addWidget(final_export_group)
        finalize_page_layout.addStretch(1)
        self.classify_workflow_tabs.addTab(finalize_page, "3. Finalize & export")
        add_tab(classify, "🏷 Classify", "classify")
        self.classify_tab_index = self.tabs.count() - 1

        # Labeler
        labeler = QWidget()
        labeler_layout = QVBoxLayout(labeler)
        labeler_intro = QLabel(
            "Build simple, mutually exclusive cell lists without training a "
            "classifier. Labeler uses the active experiment's frozen cohort and "
            "the same direct cell-picking behaviour as Classify. Assignments stay "
            "in memory until you export them or apply them to the live AnnData."
        )
        labeler_intro.setWordWrap(True)
        labeler_layout.addWidget(labeler_intro)

        label_definition_group = workflow_group(
            "1. Define labels", "labeler", "Define labels"
        )
        label_definition_layout = QVBoxLayout(label_definition_group)
        self.labeler_class_table = QTableWidget(0, 3)
        self.labeler_class_table.setHorizontalHeaderLabels(
            ["Stable ID", "Name", "Colour (double-click)"]
        )
        self.labeler_class_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.labeler_class_table.setMaximumHeight(180)
        label_definition_buttons = QHBoxLayout()
        self.add_labeler_class_button = QPushButton("Add label")
        self.remove_labeler_class_button = QPushButton("Remove selected label")
        self.pick_labeler_colour_button = QPushButton("Pick selected colour...")
        self.apply_labeler_classes_button = QPushButton("Apply label edits")
        label_definition_buttons.addWidget(self.add_labeler_class_button)
        label_definition_buttons.addWidget(self.remove_labeler_class_button)
        label_definition_buttons.addWidget(self.pick_labeler_colour_button)
        label_definition_buttons.addWidget(self.apply_labeler_classes_button)
        label_definition_layout.addWidget(self.labeler_class_table)
        label_definition_layout.addLayout(label_definition_buttons)
        labeler_layout.addWidget(label_definition_group)

        label_cells_group = workflow_group("2. Label cells", "labeler", "Label cells")
        label_cells_form = QFormLayout(label_cells_group)
        self.labeler_selected_cell_label = QLabel("No cohort cell selected")
        labeler_picking_help = QLabel(
            "While Labeler is active, click any eligible cell in the viewer. "
            "The cohort layer can remain hidden and does not need to be selected."
        )
        labeler_picking_help.setWordWrap(True)
        self.labeler_class_combo = QComboBox()
        labeler_click_widget = QWidget()
        labeler_click_layout = QHBoxLayout(labeler_click_widget)
        labeler_click_layout.setContentsMargins(0, 0, 0, 0)
        self.labeler_click_behavior_group = QButtonGroup(labeler_click_widget)
        self.labeler_click_behavior_radios = {}
        for behavior, text in (
            ("assign", "Assign selected label"),
            ("select", "Select only"),
            ("clear", "Clear label"),
        ):
            radio = QRadioButton(text)
            radio.setProperty("napari_sbt_labeler_click_behavior", behavior)
            self.labeler_click_behavior_group.addButton(radio)
            self.labeler_click_behavior_radios[behavior] = radio
            labeler_click_layout.addWidget(radio)
        self.labeler_click_behavior_radios["assign"].setChecked(True)
        labeler_click_layout.addStretch(1)
        labeler_annotation_buttons = QWidget()
        labeler_annotation_layout = QHBoxLayout(labeler_annotation_buttons)
        labeler_annotation_layout.setContentsMargins(0, 0, 0, 0)
        self.assign_labeler_cell_button = QPushButton("Assign selected label")
        self.clear_labeler_cell_button = QPushButton("Clear selected cell label")
        self.clear_all_labeler_button = QPushButton("Clear all labels...")
        labeler_annotation_layout.addWidget(self.assign_labeler_cell_button)
        labeler_annotation_layout.addWidget(self.clear_labeler_cell_button)
        labeler_annotation_layout.addWidget(self.clear_all_labeler_button)
        label_cells_form.addRow("Cell", self.labeler_selected_cell_label)
        label_cells_form.addRow("Picking", labeler_picking_help)
        label_cells_form.addRow("Current label", self.labeler_class_combo)
        label_cells_form.addRow("Click action", labeler_click_widget)
        label_cells_form.addRow("", labeler_annotation_buttons)
        labeler_layout.addWidget(label_cells_group)

        sampling_group = workflow_group(
            "3. ROI sampling guidance", "labeler", "ROI sampling guidance"
        )
        sampling_form = QFormLayout(sampling_group)
        self.labeler_roi_combo = QComboBox()
        labeler_roi_buttons = QWidget()
        labeler_roi_buttons_layout = QHBoxLayout(labeler_roi_buttons)
        labeler_roi_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.previous_labeler_roi_button = QPushButton("Previous ROI")
        self.next_labeler_roi_button = QPushButton("Next ROI")
        self.next_unsampled_labeler_roi_button = QPushButton(
            "Next ROI without this label"
        )
        labeler_roi_buttons_layout.addWidget(self.previous_labeler_roi_button)
        labeler_roi_buttons_layout.addWidget(self.next_labeler_roi_button)
        labeler_roi_buttons_layout.addWidget(self.next_unsampled_labeler_roi_button)
        self.labeler_tally_table = QTableWidget(0, 4)
        self.labeler_tally_table.setHorizontalHeaderLabels(
            ["Label", "Cells", "ROIs sampled", "In current ROI"]
        )
        self.labeler_tally_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.labeler_tally_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.labeler_tally_table.verticalHeader().setVisible(False)
        self.labeler_tally_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        for column in (1, 2, 3):
            self.labeler_tally_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeToContents
            )
        self.labeler_tally_table.setMaximumHeight(210)
        self.labeler_sampling_summary_label = QLabel(
            "Create or load an experiment to begin ROI sampling."
        )
        self.labeler_sampling_summary_label.setWordWrap(True)
        sampling_form.addRow("ROI", self.labeler_roi_combo)
        sampling_form.addRow("", labeler_roi_buttons)
        sampling_form.addRow("Coverage", self.labeler_tally_table)
        sampling_form.addRow("Sampling status", self.labeler_sampling_summary_label)
        labeler_layout.addWidget(sampling_group)

        labeler_export_group = workflow_group(
            "4. Results and export", "labeler", "Results and export"
        )
        labeler_export_layout = QVBoxLayout(labeler_export_group)
        self.labeler_results_table = QTableWidget(0, 6)
        self.labeler_results_table.setHorizontalHeaderLabels(
            ["AnnData cell", "ROI", "ObjectNumber", "Label", "Stable ID", "Time"]
        )
        self.labeler_results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.labeler_results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.labeler_results_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch
        )
        self.labeler_results_table.setMaximumHeight(260)
        labeler_export_form = QFormLayout()
        self.labeler_csv_path_edit = QLineEdit(
            str(self.project_root / "napari_sbt_cell_labels.csv")
        )
        labeler_csv_destination = QWidget()
        labeler_csv_destination_layout = QHBoxLayout(labeler_csv_destination)
        labeler_csv_destination_layout.setContentsMargins(0, 0, 0, 0)
        self.choose_labeler_csv_button = QPushButton("Choose...")
        self.export_labeler_csv_button = QPushButton("Export CSV")
        labeler_csv_destination_layout.addWidget(self.labeler_csv_path_edit)
        labeler_csv_destination_layout.addWidget(self.choose_labeler_csv_button)
        labeler_csv_destination_layout.addWidget(self.export_labeler_csv_button)
        self.labeler_obs_name_edit = QLineEdit("napari_sbt_labels")
        self.labeler_overwrite_obs_check = QCheckBox(
            "Allow explicit overwrite if this obs already exists"
        )
        self.apply_labeler_to_adata_button = QPushButton(
            "Apply as categorical obs to live AnnData..."
        )
        labeler_export_form.addRow("CSV destination", labeler_csv_destination)
        labeler_export_form.addRow("AnnData obs name", self.labeler_obs_name_edit)
        labeler_export_form.addRow("", self.labeler_overwrite_obs_check)
        labeler_export_form.addRow("", self.apply_labeler_to_adata_button)
        labeler_export_layout.addWidget(self.labeler_results_table)
        labeler_export_layout.addLayout(labeler_export_form)
        labeler_layout.addWidget(labeler_export_group)
        labeler_layout.addStretch(1)
        add_tab(labeler, "📍 Labeler", "labeler")
        self.labeler_tab_index = self.tabs.count() - 1

        # Regions & Export
        regions = QWidget()
        regions_layout = QVBoxLayout(regions)
        region_group = workflow_group(
            "Manual tissue regions", "regions_export", "Manual tissue regions"
        )
        region_form = QFormLayout(region_group)
        self.region_name_edit = QLineEdit("region")
        self.create_regions_button = QPushButton("Create/select regions layer")
        self.sync_regions_button = QPushButton("Synchronize regions to cell table")
        region_form.addRow("Region name", self.region_name_edit)
        region_form.addRow("", self.create_regions_button)
        region_form.addRow("", self.sync_regions_button)
        regions_layout.addWidget(region_group)
        export_group = workflow_group(
            "Derived mask exports", "regions_export", "Cohort results and exports"
        )
        self.classification_export_group = export_group
        export_form = QFormLayout(export_group)
        classification_export_note = QLabel(
            "Create and export classification identities from Classify → "
            "Finalize & export. This panel contains only derived mask outputs."
        )
        classification_export_note.setWordWrap(True)
        self.export_cohort_masks_button = QPushButton("Export cohort masks")
        self.export_clean_masks_button = QPushButton("Export cleaned masks")
        export_form.addRow(classification_export_note)
        export_form.addRow("", self.export_cohort_masks_button)
        export_form.addRow("", self.export_clean_masks_button)
        regions_layout.addWidget(export_group)
        add_tab(regions, "🗺 Regions & Export", "regions_export")

        # Layers & Status
        layers = QWidget()
        layers_layout = QVBoxLayout(layers)
        utility_group = workflow_group(
            "Selected-layer utilities",
            "layers_status",
            "Selected-layer utilities",
        )
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
            QGroupBox[sbtWorkflowBox="true"] {
                border: 2px solid #94a3b8;
                border-radius: 9px;
                margin-top: 20px;
                padding: 15px 10px 10px 10px;
            }
            QGroupBox[sbtWorkflowBox="true"]::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 4px 9px;
                border: 1px solid #64748b;
                border-radius: 5px;
                background-color: #e2e8f0;
                color: #0f172a;
                font-size: 14px;
                font-weight: 700;
            }
            QGroupBox[sbtNumbered="true"]::title {
                padding: 5px 10px;
                font-size: 16px;
                font-weight: 800;
            }
            QGroupBox[sbtAccent="blue"] { border-color: #60a5fa; }
            QGroupBox[sbtAccent="blue"]::title {
                border-color: #3b82f6;
                background-color: #dbeafe;
                color: #1e3a8a;
            }
            QGroupBox[sbtAccent="violet"] { border-color: #a78bfa; }
            QGroupBox[sbtAccent="violet"]::title {
                border-color: #8b5cf6;
                background-color: #ede9fe;
                color: #4c1d95;
            }
            QGroupBox[sbtAccent="teal"] { border-color: #2dd4bf; }
            QGroupBox[sbtAccent="teal"]::title {
                border-color: #14b8a6;
                background-color: #ccfbf1;
                color: #134e4a;
            }
            QGroupBox[sbtAccent="amber"] { border-color: #fbbf24; }
            QGroupBox[sbtAccent="amber"]::title {
                border-color: #f59e0b;
                background-color: #fef3c7;
                color: #78350f;
            }
            QGroupBox[sbtAccent="rose"] { border-color: #fb7185; }
            QGroupBox[sbtAccent="rose"]::title {
                border-color: #f43f5e;
                background-color: #ffe4e6;
                color: #881337;
            }
            QGroupBox[sbtAccent="cyan"] { border-color: #22d3ee; }
            QGroupBox[sbtAccent="cyan"]::title {
                border-color: #06b6d4;
                background-color: #cffafe;
                color: #164e63;
            }
            QPushButton#sbtBoxHelpButton,
            QPushButton#sbtTabHelpButton {
                border: 1px solid #1d4ed8;
                border-radius: 6px;
                padding: 4px 10px;
                background-color: #2563eb;
                color: #ffffff;
                font-weight: 800;
            }
            QPushButton#sbtBoxHelpButton:hover,
            QPushButton#sbtTabHelpButton:hover {
                background-color: #1d4ed8;
                border-color: #bfdbfe;
            }
            QPushButton#sbtBoxHelpButton:pressed,
            QPushButton#sbtTabHelpButton:pressed {
                background-color: #1e3a8a;
            }
            QPushButton#sbtPrimaryActionButton {
                background-color: #047857;
                color: white;
                font-weight: 800;
                padding: 7px 12px;
                border: 2px solid #6ee7b7;
                border-radius: 5px;
            }
            QPushButton#sbtPrimaryActionButton:hover {
                background-color: #065f46;
            }
            QPushButton#sbtPrimaryActionButton:disabled {
                background-color: #cbd5e1;
                color: #64748b;
                border-color: #94a3b8;
            }
            QFrame#sbtWorkflowChoiceCard {
                background-color: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 7px;
            }
            QFrame#sbtWorkflowChoiceCard:hover {
                background-color: #eff6ff;
                border-color: #60a5fa;
            }
            QFrame#sbtWorkflowChoiceCard QRadioButton {
                color: #0f172a;
                font-size: 14px;
                font-weight: 800;
            }
            QLabel#sbtInputStatus {
                border-radius: 5px;
                padding: 4px 7px;
                font-weight: 700;
            }
            QLabel#sbtSetupReadiness {
                border: 2px solid #dc2626;
                border-radius: 7px;
                background-color: #fee2e2;
                color: #7f1d1d;
                padding: 8px;
                font-weight: 800;
            }
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
            QTabBar::tab:nth-child(7) { background: #ecfccb; }
            QTabBar::tab:nth-child(8) { background: #f1f5f9; }
            QTabBar::tab:nth-child(9) { background: #ffedd5; }
            QTabBar::tab:nth-child(10) { background: #f3e8ff; }
            QTabBar::tab:nth-child(11) { background: #cffafe; }
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
            "#4d7c0f",
            "#0369a1",
            "#475569",
            "#c2410c",
            "#7e22ce",
            "#0e7490",
        )
        for index, colour in enumerate(tab_text_colours):
            self.tabs.tabBar().setTabTextColor(index, self.QColor(colour))

        # This widget is installed in its own Napari dock by ``launch``.  Keeping
        # it parentless here lets QDockWidget take ownership without leaving a
        # floating overlay inside the main workflow panel.
        self.activity_widget = QFrame()
        self.activity_dock = None
        self.activity_widget.setObjectName("sbtActivityPanel")
        self.activity_widget.setMinimumWidth(240)
        self.activity_widget.setMinimumHeight(82)
        self.activity_widget.setStyleSheet(
            "QFrame#sbtActivityPanel { background: rgba(25, 31, 42, 235); "
            "border: 2px solid #60a5fa; border-radius: 8px; } "
            "QLabel { color: white; background: transparent; }"
        )
        activity_layout = QVBoxLayout(self.activity_widget)
        activity_layout.setContentsMargins(10, 7, 10, 7)
        activity_layout.setSpacing(2)
        self.activity_title_label = QLabel("● Ready")
        self.activity_title_label.setWordWrap(True)
        activity_title_font = QFont(self.activity_title_label.font())
        activity_title_font.setBold(True)
        self.activity_title_label.setFont(activity_title_font)
        self.activity_detail_label = QLabel("No active operation.")
        self.activity_detail_label.setWordWrap(True)
        activity_layout.addWidget(self.activity_title_label)
        activity_layout.addWidget(self.activity_detail_label)
        self.activity_widget.adjustSize()
        self.activity_timer = QTimer(self.root)
        self.activity_timer.setInterval(1000)
        self.activity_timer.timeout.connect(self._update_activity_monitor)
        self.activity_timer.start()
        self._update_activity_monitor()

        self._set_class_rows(segmentation_qc_classes())
        self._set_labeler_class_rows(self.labeler_classes)
        self._refresh_labeler_controls()
        self._connect_signals()
        self._initialise_setup_controls()
        self._update_workflow_mode()
        self._refresh_population_naming_readiness()
        self._bind_viewer_cell_picking()
        for family, checkbox in self.feature_family_checks.items():
            self._feature_family_toggled(family, checkbox.isChecked())
        self._update_feature_selection_summary()
        self._update_feature_channel_summary()
        self._refresh_reload_recipe_list()
        self._set_classification_enabled(False)
        if experiment:
            self.load_existing_experiment(Path(experiment))
        elif resolved_anndata_path is not None or in_memory_anndata is not None:
            self.load_anndata_selectors()
        self.refresh_setup_readiness()

    def _connect_signals(self) -> None:
        self.tabs.currentChanged.connect(self._workflow_tab_changed)
        self.workflow_button_group.buttonClicked.connect(
            self._guard(self._workflow_card_selected, pass_signal_args=True)
        )
        self.workflow_combo.currentIndexChanged.connect(
            self._guard(self._update_workflow_mode)
        )
        self.advanced_workflow_check.toggled.connect(
            self.advanced_workflow_card.setVisible
        )
        self.registered_project_combo.activated.connect(
            self._guard(self.use_selected_registered_project)
        )
        self.choose_project_button.clicked.connect(
            self._guard(self.choose_project_folder)
        )
        self.refresh_workspaces_button.clicked.connect(
            self._guard(self.refresh_workspace_choices)
        )
        self.workspace_combo.currentIndexChanged.connect(
            self._guard(self._workspace_selection_changed)
        )
        self.open_workspace_button.clicked.connect(
            self._guard(self.open_selected_workspace)
        )
        self.new_workspace_button.clicked.connect(
            self._guard(self.start_new_workspace)
        )
        self.name_edit.textChanged.connect(self._workspace_name_changed)
        self.experiment_edit.textChanged.connect(self.refresh_setup_readiness)
        self.choose_experiment_folder_button.clicked.connect(
            self._guard(self.choose_new_workspace_folder)
        )
        self.next_setup_problem_button.clicked.connect(
            self._guard(self.focus_next_setup_problem)
        )
        self.live_recipe_tracking_check.toggled.connect(
            self._guard(self._live_recipe_tracking_changed)
        )
        self.validate_integrity_button.clicked.connect(
            self._guard(self.preview_cohort)
        )
        for signal in (
            self.masks_edit.textChanged,
            self.images_edit.textChanged,
            self.extra_images_edit.textChanged,
            self.roi_obs_edit.textChanged,
            self.object_obs_edit.textChanged,
        ):
            signal.connect(self._invalidate_dataset_indexes)
        self.anndata_edit.textChanged.connect(self._invalidate_integrity_result)
        self.choose_anndata_button.clicked.connect(
            self._guard(self.choose_anndata_file)
        )
        self.reload_anndata_button.clicked.connect(
            self._guard(self.load_anndata_selectors)
        )
        self.choose_masks_button.clicked.connect(
            self._guard(self.choose_masks_folder)
        )
        self.add_images_folder_button.clicked.connect(
            self._guard(self.add_images_folder)
        )
        self.remove_images_folder_button.clicked.connect(
            self._guard(self.remove_images_folder)
        )
        self.clear_images_folders_button.clicked.connect(self.images_edit.clear)
        self.add_extra_images_folder_button.clicked.connect(
            self._guard(self.add_extra_images_folder)
        )
        self.remove_extra_images_folder_button.clicked.connect(
            self._guard(self.remove_extra_images_folder)
        )
        self.clear_extra_images_folders_button.clicked.connect(
            self.extra_images_edit.clear
        )
        self.advanced_identity_check.toggled.connect(self.identity_widget.setVisible)
        self.reload_all_inputs_button.clicked.connect(
            self._guard(self.reload_all_dataset_components)
        )
        self.normalization_edit.textChanged.connect(self.refresh_setup_readiness)
        self.choose_normalization_button.clicked.connect(
            self._guard(self.choose_normalization_json)
        )
        self.load_normalization_button.clicked.connect(
            self._guard(self.load_normalization_json)
        )
        self.add_normalization_row_button.clicked.connect(
            self.add_normalization_row
        )
        self.remove_normalization_row_button.clicked.connect(
            self.remove_selected_normalization_rows
        )
        self.advanced_normalization_check.toggled.connect(
            self.normalization_json_edit.setVisible
        )
        self.normalization_table.itemChanged.connect(
            self._sync_normalization_json_preview
        )
        self.validate_normalization_button.clicked.connect(
            self._guard(self.validate_normalization_editor)
        )
        self.save_normalization_button.clicked.connect(
            self._guard(self.save_normalization_to_experiment)
        )
        for display_signal in (
            self.display_quantile_spin.valueChanged,
            self.display_minimum_pixel_spin.valueChanged,
            self.display_lower_contrast_spin.valueChanged,
            self.display_upper_contrast_spin.valueChanged,
        ):
            display_signal.connect(self._guard(self._display_defaults_changed))
        self.load_adata_button.clicked.connect(self._guard(self.load_anndata_selectors))
        self.obs_combo.currentTextChanged.connect(
            self._guard(self.refresh_scope_values)
        )
        self.obs_combo.currentTextChanged.connect(self._invalidate_integrity_result)
        self.scope_combo.currentIndexChanged.connect(self._update_scope_widget_state)
        self.scope_combo.currentIndexChanged.connect(
            self._invalidate_integrity_result
        )
        self.value_list.itemSelectionChanged.connect(
            self._invalidate_integrity_result
        )
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
        self.trial_roi_list.itemSelectionChanged.connect(self._update_trial_roi_summary)
        self.suggest_trial_rois_button.clicked.connect(
            self._guard(self.suggest_trial_rois)
        )
        self.qc_template_button.clicked.connect(
            lambda: self._set_class_rows(segmentation_qc_classes())
        )
        self.add_class_button.clicked.connect(self.add_class_row)
        self.remove_class_button.clicked.connect(self.remove_class_row)
        self.pick_class_colour_button.clicked.connect(
            self._guard(self.pick_selected_class_colour)
        )
        self.class_table.cellDoubleClicked.connect(
            self._guard(self.pick_class_colour_from_cell, pass_signal_args=True)
        )
        self.apply_classes_button.clicked.connect(self._guard(self.apply_class_edits))
        self.add_labeler_class_button.clicked.connect(self.add_labeler_class_row)
        self.remove_labeler_class_button.clicked.connect(
            self._guard(self.remove_labeler_class_row)
        )
        self.pick_labeler_colour_button.clicked.connect(
            self._guard(self.pick_selected_labeler_colour)
        )
        self.labeler_class_table.cellDoubleClicked.connect(
            self._guard(
                self.pick_labeler_colour_from_cell,
                pass_signal_args=True,
            )
        )
        self.apply_labeler_classes_button.clicked.connect(
            self._guard(self.apply_labeler_class_edits)
        )
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
        self.build_features_button.clicked.connect(
            self._guard(self.start_feature_build)
        )
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
        self.cancel_refinement_button.clicked.connect(self.cancel_feature_refinement)
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
        self.previous_roi_button.clicked.connect(lambda: self.move_roi(-1))
        self.next_roi_button.clicked.connect(lambda: self.move_roi(1))
        self.labeler_roi_combo.currentTextChanged.connect(
            self._guard(self.load_labeler_roi, pass_signal_args=True)
        )
        self.previous_labeler_roi_button.clicked.connect(lambda: self.move_roi(-1))
        self.next_labeler_roi_button.clicked.connect(lambda: self.move_roi(1))
        self.next_unsampled_labeler_roi_button.clicked.connect(
            self._guard(self.move_to_next_unsampled_labeler_roi)
        )
        self.labeler_class_combo.currentIndexChanged.connect(
            self._refresh_labeler_tally
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
        self.recipe_preset_combo.currentIndexChanged.connect(
            self._recipe_preset_selection_changed
        )
        self.load_recipe_preset_button.clicked.connect(
            self._guard(self.load_selected_recipe_preset)
        )
        self.save_new_recipe_preset_button.clicked.connect(
            self._guard(self.save_new_recipe_preset)
        )
        self.update_recipe_preset_button.clicked.connect(
            self._guard(self.update_selected_recipe_preset)
        )
        self.delete_recipe_preset_button.clicked.connect(
            self._guard(self.delete_selected_recipe_preset)
        )
        self.import_recipe_preset_button.clicked.connect(
            self._guard(self.import_explore_recipe_preset)
        )
        self.export_recipe_preset_button.clicked.connect(
            self._guard(self.export_selected_explore_recipe_preset)
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
        self.load_channels_button.clicked.connect(
            self._guard(self.load_selected_channels)
        )
        self.load_six_colour_button.clicked.connect(
            self._guard(self.load_six_colour_channels)
        )
        self.load_rgb_button.clicked.connect(self._guard(self.load_rgb))
        self.population_qc_obs_combo.currentTextChanged.connect(
            self._guard(self.refresh_population_qc_populations)
        )
        self.population_qc_population_combo.currentTextChanged.connect(
            self._guard(self.load_population_qc_recipe_controls)
        )
        self.population_qc_contour_spin.valueChanged.connect(
            self._guard(
                self.set_population_qc_contour_width,
                pass_signal_args=True,
            )
        )
        self.suggest_population_qc_markers_button.clicked.connect(
            self._guard(self.suggest_population_qc_markers)
        )
        self.save_population_qc_recipe_button.clicked.connect(
            self._guard(self.save_population_qc_recipe)
        )
        self.reset_population_qc_contrast_button.clicked.connect(
            self._guard(self.reset_population_qc_contrasts_to_setup_defaults)
        )
        self.load_population_qc_view_button.clicked.connect(
            self._guard(self.load_population_qc_view)
        )
        self.recalculate_population_qc_rois_button.clicked.connect(
            self._guard(self.recalculate_population_qc_rois)
        )
        self.population_qc_roi_order_combo.currentIndexChanged.connect(
            self._guard(self.refresh_population_qc_rois)
        )
        self.population_qc_roi_limit_spin.valueChanged.connect(
            self._guard(self.refresh_population_qc_rois)
        )
        self.population_qc_random_seed_spin.valueChanged.connect(
            self._guard(self.refresh_population_qc_rois)
        )
        self.import_population_qc_csv_button.clicked.connect(
            self._guard(self.import_population_qc_settings_csv)
        )
        self.export_population_qc_csv_button.clicked.connect(
            self._guard(self.export_population_qc_settings_csv)
        )
        self.curation_source_combo.currentTextChanged.connect(
            self._guard(self.refresh_population_workspace)
        )
        self.create_population_draft_button.clicked.connect(
            self._guard(self.create_population_draft)
        )
        self.save_population_draft_button.clicked.connect(
            self._guard(self.save_current_population_draft)
        )
        self.view_population_history_button.clicked.connect(
            self._guard(self.show_population_history)
        )
        self.curation_draft_combo.currentIndexChanged.connect(
            self._guard(self.load_selected_population_draft)
        )
        self.curation_derived_obs_edit.textEdited.connect(
            self._guard(self._mark_population_draft_dirty)
        )
        self.population_base_table.itemChanged.connect(
            self._guard(self._population_tables_changed)
        )
        self.population_components_table.itemChanged.connect(
            self._guard(self._population_tables_changed)
        )
        self.name_selected_populations_button.clicked.connect(
            lambda: self._guard(lambda: self.name_selected_population_rows("base"))()
        )
        self.name_selected_components_button.clicked.connect(
            lambda: self._guard(
                lambda: self.name_selected_population_rows("components")
            )()
        )
        self.colour_selected_populations_button.clicked.connect(
            lambda: self._guard(lambda: self.colour_selected_population_rows("base"))()
        )
        self.colour_selected_components_button.clicked.connect(
            lambda: self._guard(
                lambda: self.colour_selected_population_rows("components")
            )()
        )
        self.import_population_mapping_button.clicked.connect(
            self._guard(self.import_population_mapping)
        )
        self.export_population_mapping_button.clicked.connect(
            self._guard(self.export_population_mapping)
        )
        self.run_population_subcluster_button.clicked.connect(
            self._guard(self.start_population_subclustering)
        )
        self.population_neighbor_source_combo.currentIndexChanged.connect(
            self._guard(self._update_population_neighbor_controls)
        )
        self.population_representation_combo.currentTextChanged.connect(
            self._guard(self._update_population_graph_provenance)
        )
        self.population_n_neighbors_spin.valueChanged.connect(
            self._guard(self._update_population_graph_provenance)
        )
        self.population_adjacency_combo.currentTextChanged.connect(
            self._guard(self._update_population_graph_provenance)
        )
        self.cancel_population_subcluster_button.clicked.connect(
            self.cancel_population_subclustering
        )
        self.import_population_components_button.clicked.connect(
            self._guard(self.import_population_components)
        )
        self.import_current_classifier_components_button.clicked.connect(
            self._guard(self.import_current_classifier_components)
        )
        self.remove_population_components_button.clicked.connect(
            self._guard(self.remove_selected_population_components)
        )
        self.show_curated_population_overlay_button.clicked.connect(
            self._guard(self.show_curated_population_overlay)
        )
        self.open_population_scanpy_plotting_button.clicked.connect(
            self._guard(self.open_population_scanpy_plotting)
        )
        self.export_curated_anndata_button.clicked.connect(
            self._guard(self.export_curated_anndata)
        )
        self.propose_button.clicked.connect(
            self._guard(lambda: self.annotate_selected("proposed"))
        )
        self.confirm_button.clicked.connect(
            self._guard(lambda: self.annotate_selected("confirmed"))
        )
        self.clear_proposed_button.clicked.connect(
            self._guard(self.clear_selected_proposed)
        )
        self.clear_all_proposals_button.clicked.connect(
            self._guard(self.clear_all_proposals)
        )
        self.classifier_display_button.clicked.connect(
            self._guard(self.show_classifier_display_options)
        )
        self.confirm_proposed_button.clicked.connect(
            self._guard(self.confirm_all_proposed)
        )
        self.mark_reviewed_button.clicked.connect(self._guard(self.mark_roi_reviewed))
        self.seed_obs_button.clicked.connect(self._guard(self.seed_proposals_from_obs))
        self.train_button.clicked.connect(self._guard(self.train_model))
        self.score_button.clicked.connect(self._guard(self.score_model))
        self.refresh_queue_button.clicked.connect(
            self._guard(self.refresh_uncertainty_queue)
        )
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
        self.apply_prediction_review_button.clicked.connect(
            self._guard(self.apply_prediction_review_filter)
        )
        self.reset_prediction_review_button.clicked.connect(
            self._guard(self.reset_prediction_review_filter)
        )
        for threshold_signal in (
            self.final_min_confidence_spin.valueChanged,
            self.final_max_uncertainty_spin.valueChanged,
            self.final_min_margin_spin.valueChanged,
        ):
            threshold_signal.connect(self._mark_final_identities_stale)
        self.create_final_identities_button.clicked.connect(
            self._guard(self.create_final_identities)
        )
        self.apply_live_adata_button.clicked.connect(
            self._guard(self.apply_final_identities_to_live_anndata)
        )
        self.assign_labeler_cell_button.clicked.connect(
            self._guard(self.assign_selected_labeler_cell)
        )
        self.clear_labeler_cell_button.clicked.connect(
            self._guard(self.clear_selected_labeler_cell)
        )
        self.clear_all_labeler_button.clicked.connect(
            self._guard(self.clear_all_labeler_records)
        )
        self.choose_labeler_csv_button.clicked.connect(
            self._guard(self.choose_labeler_csv_destination)
        )
        self.export_labeler_csv_button.clicked.connect(
            self._guard(self.export_labeler_csv)
        )
        self.apply_labeler_to_adata_button.clicked.connect(
            self._guard(self.apply_labeler_records_to_live_anndata)
        )
        self.create_regions_button.clicked.connect(
            self._guard(self.create_regions_layer)
        )
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
        self.flip_vertical_button.clicked.connect(
            lambda: self.flip_selected_layer(axis=0)
        )
        self.transfer_colormap_button.clicked.connect(
            self._guard(self.transfer_colormap)
        )
        self.expand_button.clicked.connect(self._guard(self.expand_selected_labels))
        self.resize_button.clicked.connect(self._guard(self.resize_selected_layer))
        self.mask_layer_button.clicked.connect(self._guard(self.mask_selected_image))
        self.refresh_status_button.clicked.connect(self._guard(self.refresh_status))
        self._update_scope_widget_state()
        self._update_experiment_mode_state()
        self._update_population_neighbor_controls()

    def _guard(self, callback, *, pass_signal_args: bool = False):
        def wrapped(*args, **kwargs):
            action = str(getattr(callback, "__name__", "Action"))
            if action == "<lambda>":
                action = "Interface action"
            action = action.replace("_", " ").strip().capitalize()
            monitor_ready = hasattr(self, "activity_widget")
            if monitor_ready:
                self._activity_begin(action)
            try:
                if pass_signal_args:
                    result = callback(*args, **kwargs)
                else:
                    result = callback()
            except Exception as exc:  # noqa: BLE001 - Qt callback error boundary
                if monitor_ready:
                    self._activity_finish(False, f"{type(exc).__name__}: {exc}")
                self.set_status(f"ERROR — {type(exc).__name__}: {exc}")
                self.QMessageBox.critical(
                    self.root, "napari_sbt", f"{type(exc).__name__}: {exc}"
                )
                return None
            if monitor_ready and self._active_background_processes():
                self.activity_waiting_for_process = True
                self._activity_update("Background Python process is running.")
            elif monitor_ready:
                self._activity_finish(True, "Completed.")
            return result

        return wrapped

    def set_status(self, message: str) -> None:
        self.status_text.append(str(message))
        self.scope_label.setToolTip(str(message))
        if getattr(self, "activity_state", "idle") == "running":
            self._activity_update(str(message))

    def _process_is_running(self, process) -> bool:
        if process is None:
            return False
        try:
            state = process.state()
            state_value = getattr(state, "value", state)
            return int(state_value) != 0
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False

    def _active_background_processes(self) -> list[tuple[str, object]]:
        processes = (
            ("feature build", self.feature_process),
            ("feature-source validation", self.source_validation_process),
            ("feature refinement", self.refinement_process),
            ("population subclustering", self.population_process),
        )
        return [
            (name, process)
            for name, process in processes
            if self._process_is_running(process)
        ]

    def _activity_begin(self, action: str, detail: str = "Action started…") -> None:
        self.activity_state = "running"
        self.activity_action = str(action)
        self.activity_detail = str(detail)
        self.activity_started_at = time.monotonic()
        self.activity_finished_at = None
        self.activity_waiting_for_process = False
        self._update_activity_monitor()
        self.QApplication.processEvents()

    def _activity_update(self, detail: str) -> None:
        self.activity_detail = str(detail)
        self._update_activity_monitor()

    def _activity_finish(self, success: bool, detail: str) -> None:
        self.activity_state = "complete" if success else "error"
        self.activity_detail = str(detail)
        self.activity_finished_at = time.monotonic()
        self.activity_waiting_for_process = False
        self._update_activity_monitor()

    def install_readiness_dock(self):
        """Install the activity tracker below Napari's layer list when possible."""

        if self.activity_dock is not None:
            return self.activity_dock
        add_dock_widget = getattr(
            getattr(self.viewer, "window", None), "add_dock_widget", None
        )
        if not callable(add_dock_widget):
            return None
        self.activity_dock = add_dock_widget(
            self.activity_widget,
            name="NapariSBT Readiness",
            area="left",
            add_vertical_stretch=False,
            tabify=False,
        )
        self._position_readiness_dock()
        return self.activity_dock

    def _position_readiness_dock(self) -> None:
        """Prefer a compact slot directly beneath Napari's built-in Layers dock."""

        if self.activity_dock is None:
            return
        window = getattr(self.viewer, "window", None)
        qt_window = getattr(window, "_qt_window", None)
        if qt_window is None:
            return

        qt_viewer = getattr(qt_window, "_qt_viewer", None)
        layer_dock = getattr(qt_viewer, "dockLayerList", None)
        if layer_dock is None:
            # Napari's public docking API is stable, but the attribute holding
            # its built-in layer list has changed across releases.  Fall back
            # to matching the dock title/object name before giving up on the
            # preferred placement; the readiness dock itself remains usable.
            from qtpy.QtWidgets import QDockWidget

            for candidate in qt_window.findChildren(QDockWidget):
                if candidate is self.activity_dock:
                    continue
                candidate_name = (
                    " ".join(
                        (
                            str(candidate.objectName() or ""),
                            str(candidate.windowTitle() or ""),
                        )
                    )
                    .strip()
                    .casefold()
                )
                if "layer list" in candidate_name or candidate_name == "layers":
                    layer_dock = candidate
                    break
        if layer_dock is None:
            return

        orientation = getattr(getattr(self.Qt, "Orientation", self.Qt), "Vertical")
        qt_window.splitDockWidget(layer_dock, self.activity_dock, orientation)
        qt_window.resizeDocks(
            [layer_dock, self.activity_dock],
            [10000, 130],
            orientation,
        )

    def _update_activity_monitor(self) -> None:
        if not hasattr(self, "activity_widget"):
            return
        active = self._active_background_processes()
        if (
            self.activity_state == "running"
            and self.activity_waiting_for_process
            and not active
        ):
            self._activity_finish(True, "Background computation finished.")
            return
        elapsed = (
            max(0.0, time.monotonic() - self.activity_started_at)
            if self.activity_started_at is not None
            else 0.0
        )
        if self.activity_state == "running":
            process_text = ""
            if active:
                names = ", ".join(name for name, _process in active)
                pids = []
                for _name, process in active:
                    try:
                        pid = int(process.processId())
                    except (AttributeError, TypeError, ValueError, RuntimeError):
                        pid = 0
                    if pid:
                        pids.append(str(pid))
                pid_text = f"; PID {', '.join(pids)}" if pids else ""
                process_text = f"\nLive: {names}{pid_text}"
            self.activity_title_label.setText(
                f"● Working — {self.activity_action} ({elapsed:.0f}s)"
            )
            self.activity_title_label.setStyleSheet("color: #93c5fd;")
            self.activity_detail_label.setText(
                f"{self.activity_detail}{process_text}\nHeartbeat: live"
            )
        elif self.activity_state == "error":
            self.activity_title_label.setText(f"● Failed — {self.activity_action}")
            self.activity_title_label.setStyleSheet("color: #fca5a5;")
            self.activity_detail_label.setText(self.activity_detail)
        elif self.activity_state == "complete":
            self.activity_title_label.setText(f"● Finished — {self.activity_action}")
            self.activity_title_label.setStyleSheet("color: #86efac;")
            self.activity_detail_label.setText(self.activity_detail)
            if (
                self.activity_finished_at is not None
                and time.monotonic() - self.activity_finished_at > 8
            ):
                self.activity_state = "idle"
        else:
            self.activity_title_label.setText("● Ready")
            self.activity_title_label.setStyleSheet("color: #cbd5e1;")
            self.activity_detail_label.setText("No active operation.")
        self.activity_widget.adjustSize()

    def show_help(
        self,
        topic: str,
        title: str,
        *,
        section: str | None = None,
    ) -> None:
        """Show a complete tab guide or one focused external Markdown section."""

        markdown = load_help_markdown(topic, section)
        dialog = self.QDialog(self.root)
        prefix = "Focused help" if section else "Tab help"
        dialog.setWindowTitle(f"napari_sbt {prefix} — {title}")
        dialog.resize(760 if section else 820, 560 if section else 680)
        from qtpy.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(dialog)
        browser = self.QTextBrowser(dialog)
        browser.setOpenExternalLinks(True)
        browser.setMarkdown(markdown)
        buttons = self.QDialogButtonBox(self.QDialogButtonBox.Close, parent=dialog)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(browser)
        layout.addWidget(buttons)
        dialog.exec()

    def show_tab_help(self, topic: str, title: str) -> None:
        """Show documentation-backed help for one workflow tab."""

        self.show_help(topic, title)

    def _initialise_setup_controls(self) -> None:
        """Populate project/workspace choices without scanning scientific assets."""

        self._refresh_registered_project_choices()
        self._apply_project_root(self.project_root, replace_inputs=False)
        self.refresh_workspace_choices()

    def _refresh_registered_project_choices(self) -> None:
        """List current and registered projects without modifying the registry."""

        self.registered_project_combo.blockSignals(True)
        self.registered_project_combo.clear()
        self.registered_project_combo.addItem(
            f"Current: {self.project_root.name or self.project_root}",
            str(self.project_root),
        )
        try:
            from SpatialBiologyToolkit.pipeline.project_registry import (
                load_project_registry,
            )

            registry = load_project_registry()
            for project in registry.projects:
                path = Path(project.path).expanduser().resolve(strict=False)
                if path == self.project_root:
                    continue
                self.registered_project_combo.addItem(project.name, str(path))
                index = self.registered_project_combo.count() - 1
                self.registered_project_combo.setItemData(
                    index,
                    f"Registered SBT project\n{path}",
                    self.Qt.ToolTipRole,
                )
        except Exception as exc:  # noqa: BLE001 - registry is optional in the GUI
            self.registered_project_combo.setToolTip(
                f"Registered projects could not be read: {exc}. Use Choose project folder."
            )
        self.registered_project_combo.blockSignals(False)

    @staticmethod
    def _project_relative_path(project_root: Path, configured: str | Path) -> Path:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = project_root / path
        return path.resolve(strict=False)

    def _apply_project_root(self, root: str | Path, *, replace_inputs: bool) -> None:
        """Use an SBT project's typed defaults, or retain a standalone folder."""

        root = Path(root).expanduser().resolve(strict=False)
        if not root.is_dir():
            raise FileNotFoundError(f"Project or dataset folder not found: {root}")
        self.project_root = root
        self.project_edit.setText(str(root))
        experiment_folder = "napari_sbt"
        context = None
        try:
            from SpatialBiologyToolkit.pipeline.project import load_project

            context = load_project(root)
            experiment_folder = context.config.napari_sbt.experiment_folder
        except Exception:  # Standalone datasets remain supported.
            context = None
        self._workspace_container = workspace_folder(root, experiment_folder)
        if replace_inputs:
            self._launch_experiment = None
        if context is not None and self._launch_experiment is None:
            active = context.config.napari_sbt.active_experiment
            if active:
                configured = Path(active).expanduser()
                self._launch_experiment = (
                    configured.resolve(strict=False)
                    if configured.is_absolute()
                    else (self._workspace_container / configured).resolve(
                        strict=False
                    )
                )
        normalization_to_load: Path | None = None
        if context is not None:
            self._updating_setup_controls = True
            try:
                if self._in_memory_adata is None and (
                    replace_inputs or not self.anndata_edit.text().strip()
                ):
                    self.anndata_edit.setText(
                        str(
                            self._project_relative_path(
                                root, context.config.general.anndata_path
                            )
                        )
                    )
                if replace_inputs or not self.masks_edit.text().strip():
                    self.masks_edit.setText(
                        str(
                            self._project_relative_path(
                                root, context.config.general.masks_folder
                            )
                        )
                    )
                if replace_inputs or not self.images_edit.toPlainText().strip():
                    self.images_edit.setPlainText(
                        str(
                            self._project_relative_path(
                                root, context.config.general.denoised_images_folder
                            )
                        )
                    )
                normalization = context.config.nimbus.normalization_dict_path
                if normalization and (
                    replace_inputs or not self.normalization_edit.text().strip()
                ):
                    normalization_to_load = self._project_relative_path(
                        root, normalization
                    )
                    self.normalization_edit.setText(str(normalization_to_load))
            finally:
                self._updating_setup_controls = False
        if normalization_to_load is not None and normalization_to_load.is_file():
            self.load_normalization_json()
        self._loaded_workspace_root = None
        if replace_inputs or self._launch_experiment is None:
            self._update_suggested_workspace_path(force=True)
        self._invalidate_dataset_indexes()
        self.refresh_workspace_choices()

    def use_selected_registered_project(self) -> None:
        path = self.registered_project_combo.currentData()
        if path:
            self._prepare_to_change_project(Path(path))

    def choose_project_folder(self) -> None:
        selected = self.QFileDialog.getExistingDirectory(
            self.root,
            "Choose an SBT project or standalone dataset folder",
            str(self.project_root),
        )
        if selected:
            self._prepare_to_change_project(Path(selected))

    def _prepare_to_change_project(self, root: Path) -> None:
        if self.paths is not None:
            reply = self.QMessageBox.question(
                self.root,
                "Change project",
                "Close the current NapariSBT workspace and choose data from a "
                "different project? Saved workspace files will not be deleted.",
            )
            if reply != self.QMessageBox.Yes:
                return
        self.start_new_workspace(confirm=False)
        self._apply_project_root(root, replace_inputs=True)
        self._refresh_registered_project_choices()
        self.set_status(f"Using project or dataset folder: {self.project_root}")

    def refresh_workspace_choices(self) -> None:
        """Refresh the bounded one-level workspace list."""

        current_root = self.workspace_combo.currentData() if self.workspace_combo.count() else None
        self._workspace_summaries = discover_workspaces(
            self._workspace_container,
            explicit=self._launch_experiment,
        )
        self.workspace_combo.blockSignals(True)
        self.workspace_combo.clear()
        if not self._workspace_summaries:
            self.workspace_combo.addItem("No saved workspaces found", None)
        for summary in self._workspace_summaries:
            presentation = workflow_presentation(summary.workflow_mode)
            workflow = presentation.title if presentation else "Unknown workflow"
            counts = (
                f"{summary.eligible_cells:,} cells / {summary.represented_rois:,} ROIs"
                if summary.eligible_cells is not None
                and summary.represented_rois is not None
                else "could not read details"
            )
            prefix = {
                "ready": "● Ready",
                "check": "● Check",
                "blocked": "⚠ Cannot open",
            }[summary.level]
            self.workspace_combo.addItem(
                f"{prefix} {summary.name} — {workflow} — {counts}",
                str(summary.root),
            )
            index = self.workspace_combo.count() - 1
            tooltip = str(summary.root)
            if summary.modified_at is not None:
                tooltip += f"\nLast changed: {summary.modified_at:%Y-%m-%d %H:%M}"
            if summary.issue:
                tooltip += f"\nCannot open: {summary.issue}"
                self.workspace_combo.setItemData(
                    index, self.QColor("#b91c1c"), self.Qt.ForegroundRole
                )
            elif summary.warnings:
                tooltip += "\nNeeds attention: " + "; ".join(summary.warnings)
                self.workspace_combo.setItemData(
                    index, self.QColor("#b45309"), self.Qt.ForegroundRole
                )
            else:
                self.workspace_combo.setItemData(
                    index, self.QColor("#15803d"), self.Qt.ForegroundRole
                )
            self.workspace_combo.setItemData(index, tooltip, self.Qt.ToolTipRole)
        selected_index = self.workspace_combo.findData(current_root)
        if selected_index < 0 and self._loaded_workspace_root is not None:
            selected_index = self.workspace_combo.findData(
                str(self._loaded_workspace_root)
            )
        self.workspace_combo.setCurrentIndex(max(0, selected_index))
        self.workspace_combo.blockSignals(False)
        self._workspace_selection_changed()

    def _selected_workspace_summary(self) -> WorkspaceSummary | None:
        root = self.workspace_combo.currentData()
        if not root:
            return None
        resolved = Path(root).expanduser().resolve(strict=False)
        return next(
            (summary for summary in self._workspace_summaries if summary.root == resolved),
            None,
        )

    def _workspace_selection_changed(self) -> None:
        summary = self._selected_workspace_summary()
        if summary is None:
            self.workspace_summary_label.setText(
                f"No saved workspace was found in {self._workspace_container}."
            )
            self.open_workspace_button.setEnabled(False)
            return
        self.open_workspace_button.setEnabled(summary.loadable)
        if summary.issue:
            self.workspace_summary_label.setText(
                f"⚠ {summary.root}\nThis workspace cannot currently be opened: "
                f"{summary.issue}"
            )
            return
        presentation = workflow_presentation(summary.workflow_mode)
        changed = (
            summary.modified_at.strftime("%d %b %Y, %H:%M")
            if summary.modified_at
            else "unknown"
        )
        self.workspace_summary_label.setText(
            f"{summary.name} — "
            f"{presentation.title if presentation else summary.workflow_mode}; "
            f"{summary.eligible_cells:,} eligible cells across "
            f"{summary.represented_rois:,} ROIs; last changed {changed}."
            + (
                " Needs attention: " + "; ".join(summary.warnings) + "."
                if summary.warnings
                else " All configured source locations were found."
            )
        )

    def open_selected_workspace(self) -> None:
        summary = self._selected_workspace_summary()
        if summary is None:
            raise ValueError("Choose a saved workspace first.")
        if not summary.loadable:
            raise ValueError(summary.issue or "The selected workspace cannot be opened.")
        self.load_existing_experiment(summary.root)

    def start_new_workspace(self, *, confirm: bool = True) -> None:
        """Leave a loaded workspace without changing any saved source files."""

        if confirm and self.paths is not None:
            reply = self.QMessageBox.question(
                self.root,
                "Set up a new workspace",
                "Leave the currently open workspace and start a new setup? Existing "
                "workspace files remain saved and can be reopened from this list.",
            )
            if reply != self.QMessageBox.Yes:
                return
        self.manifest = None
        self.paths = None
        self._loaded_workspace_root = None
        self.preview = None
        self.cohort = pd.DataFrame()
        self.labels = empty_labels()
        self.scores = pd.DataFrame()
        self.current_roi = None
        self.current_mask = None
        self.current_mask_path = None
        self._invalidate_population_qc_caches()
        self._refresh_roi_metadata_display()
        self._integrity_signature = None
        self._asset_index_signature = None
        self._mask_path_index.clear()
        self._roi_image_path_index.clear()
        self._clear_explore_layers()
        self._remove_layers(
            [
                "classification_cohort",
                "excluded_segmentation_context",
                NONCONTEXT_MASK_LAYER_NAME,
                *CLASS_LAYER_NAMES.values(),
                SELECTED_CELL_LAYER_NAME,
                LABELER_LAYER_NAME,
                LABELER_SELECTED_CELL_LAYER_NAME,
            ]
        )
        self.explore_recipe = ExploreViewRecipe()
        self.explore_review_state = ExploreReviewState()
        self._sync_population_qc_contour_control()
        self._sync_population_qc_contrast_defaults(force=True)
        self._set_classification_enabled(False)
        self.scope_label.setText(
            "No workflow workspace: complete Setup, then create it."
        )
        self.name_edit.setReadOnly(False)
        self.experiment_edit.setReadOnly(False)
        self.choose_experiment_folder_button.setEnabled(True)
        self._set_dataset_source_editable(True)
        self._updating_setup_controls = True
        try:
            self.name_edit.clear()
        finally:
            self._updating_setup_controls = False
        self._update_suggested_workspace_path(force=True)
        self.refresh_setup_readiness()

    def _set_dataset_source_editable(self, editable: bool) -> None:
        """Prevent a loaded manifest from being silently contradicted by the form."""

        self.anndata_edit.setReadOnly(not editable)
        self.masks_edit.setReadOnly(not editable)
        self.images_edit.setReadOnly(not editable)
        self.extra_images_edit.setReadOnly(not editable)
        for button in (
            self.choose_anndata_button,
            self.choose_masks_button,
            self.add_images_folder_button,
            self.remove_images_folder_button,
            self.clear_images_folders_button,
            self.add_extra_images_folder_button,
            self.remove_extra_images_folder_button,
            self.clear_extra_images_folders_button,
        ):
            button.setEnabled(editable)
        self.roi_obs_edit.setReadOnly(not editable)
        self.object_obs_edit.setReadOnly(not editable)

    def _workflow_card_selected(self, button) -> None:
        mode = str(button.property("workflowMode") or "")
        index = self.workflow_combo.findData(mode)
        if index >= 0:
            self.workflow_combo.setCurrentIndex(index)

    def _sync_workflow_cards(self) -> None:
        mode = self.current_workflow_mode()
        for value, radio in self._workflow_radios.items():
            was_blocked = radio.blockSignals(True)
            radio.setChecked(value == mode)
            radio.blockSignals(was_blocked)
        if mode == "full_workspace":
            self.advanced_workflow_check.setChecked(True)
            self.advanced_workflow_card.show()

    def _workspace_name_changed(self, *_args) -> None:
        if self._updating_setup_controls or self.manifest is not None:
            self.refresh_setup_readiness()
            return
        self._update_suggested_workspace_path()
        self.refresh_setup_readiness()

    def _update_suggested_workspace_path(self, *, force: bool = False) -> None:
        if not self.name_edit.text().strip():
            if force:
                suggested = (self._workspace_container / "new_workspace").resolve(
                    strict=False
                )
                self._suggested_workspace_path = suggested
                was_blocked = self.experiment_edit.blockSignals(True)
                self.experiment_edit.setText(str(suggested))
                self.experiment_edit.blockSignals(was_blocked)
            return
        current = Path(self.experiment_edit.text() or ".").expanduser().resolve(
            strict=False
        )
        if not force and current != getattr(self, "_suggested_workspace_path", current):
            return
        suggested = workspace_destination(self._workspace_container, self.name_edit.text())
        self._suggested_workspace_path = suggested
        was_blocked = self.experiment_edit.blockSignals(True)
        self.experiment_edit.setText(str(suggested))
        self.experiment_edit.blockSignals(was_blocked)

    def choose_new_workspace_folder(self) -> None:
        selected = self.QFileDialog.getExistingDirectory(
            self.root,
            "Choose an empty folder for the new workspace",
            str(Path(self.experiment_edit.text()).expanduser().parent),
        )
        if selected:
            self._suggested_workspace_path = Path(selected).resolve(strict=False)
            self.experiment_edit.setText(str(self._suggested_workspace_path))

    def choose_anndata_file(self) -> None:
        selected, _filter = self.QFileDialog.getOpenFileName(
            self.root,
            "Choose processed cell data",
            self.anndata_edit.text().strip() or str(self.project_root),
            "AnnData files (*.h5ad);;All files (*)",
        )
        if selected:
            self.anndata_edit.setText(selected)
            self.load_anndata_selectors()

    def choose_masks_folder(self) -> None:
        selected = self.QFileDialog.getExistingDirectory(
            self.root,
            "Choose cell masks folder",
            self.masks_edit.text().strip() or str(self.project_root),
        )
        if selected:
            self.masks_edit.setText(selected)

    def _append_folder_choice(self, editor, title: str) -> None:
        existing = _split_paths(editor.toPlainText())
        selected = self.QFileDialog.getExistingDirectory(
            self.root,
            title,
            existing[-1] if existing else str(self.project_root),
        )
        if selected and selected not in existing:
            editor.setPlainText("\n".join([*existing, selected]))

    def add_images_folder(self) -> None:
        self._append_folder_choice(self.images_edit, "Add staining image folder")

    def _remove_folder_choice(self, editor, title: str) -> None:
        existing = _split_paths(editor.toPlainText())
        if not existing:
            return
        selected, accepted = self.QInputDialog.getItem(
            self.root,
            title,
            "Folder to remove",
            existing,
            0,
            False,
        )
        if accepted and selected:
            editor.setPlainText(
                "\n".join(path for path in existing if path != selected)
            )

    def remove_images_folder(self) -> None:
        self._remove_folder_choice(
            self.images_edit, "Remove staining image folder"
        )

    def add_extra_images_folder(self) -> None:
        self._append_folder_choice(
            self.extra_images_edit, "Add optional additional image folder"
        )

    def remove_extra_images_folder(self) -> None:
        self._remove_folder_choice(
            self.extra_images_edit, "Remove optional additional image folder"
        )

    def reload_all_dataset_components(self) -> None:
        """Reload known sources without performing the expensive integrity scan."""

        if self.paths is not None:
            root = self.paths.root
            self.load_existing_experiment(root)
            self.set_status(
                "Reloaded the workspace manifest, AnnData, saved review state, and "
                "current ROI components. Integrity was not rescanned."
            )
            return
        if self.anndata_edit.text().strip() or self._in_memory_adata is not None:
            self.load_anndata_selectors()
        if self.normalization_edit.text().strip():
            self.load_normalization_json()
        self._clear_explore_layer_data_cache()
        self.refresh_setup_readiness()
        self.set_status(
            "Reloaded the selected setup components. Run the explicit integrity "
            "check when folders or identities have changed."
        )

    def focus_next_setup_problem(self) -> None:
        checks = getattr(self, "_current_setup_checks", ())
        problem = next(
            (check for check in checks if check.level == "blocked"),
            next(
                (
                    check
                    for check in checks
                    if check.level == "check" and check.key != "normalization"
                ),
                None,
            ),
        )
        if problem is None:
            self.create_button.setFocus()
            return
        if problem.key == "identity":
            self.advanced_identity_check.setChecked(True)
        targets = {
            "workspace": self.name_edit,
            "workflow": next(iter(self._workflow_radios.values())),
            "anndata": self.choose_anndata_button,
            "masks": self.choose_masks_button,
            "images": self.add_images_folder_button,
            "extra_images": self.add_extra_images_folder_button,
            "identity": self.roi_obs_edit,
            "normalization": self.choose_normalization_button,
            "integrity": self.validate_integrity_button,
        }
        target = targets.get(problem.key)
        if target is not None:
            target.setFocus()
        self.set_status(f"Setup needs attention: {problem.label} — {problem.detail}")

    def refresh_setup_readiness(self, *_args) -> None:
        """Update row badges and gate new-workspace creation."""

        if not hasattr(self, "setup_readiness_label"):
            return
        integrity_current = bool(
            self.preview is not None
            and self._integrity_signature == self._current_integrity_signature()
        )
        checks = setup_checks(
            workspace_name=self.name_edit.text(),
            workspace_path=self.experiment_edit.text(),
            workflow_mode=self.current_workflow_mode(),
            anndata_path=self.anndata_edit.text(),
            has_in_memory_anndata=self._in_memory_adata is not None,
            masks_folder=self.masks_edit.text(),
            image_folders=_split_paths(self.images_edit.toPlainText()),
            extra_image_folders=_split_paths(
                self.extra_images_edit.toPlainText()
            ),
            roi_obs=self.roi_obs_edit.text(),
            object_id_obs=self.object_obs_edit.text(),
            normalization_path=self.normalization_edit.text(),
            integrity_current=integrity_current,
        )
        self._current_setup_checks = checks
        styles = {
            "ready": ("● Ready", "#dcfce7", "#166534", "#22c55e"),
            "check": ("● Check needed", "#fef3c7", "#92400e", "#f59e0b"),
            "blocked": ("● Action required", "#fee2e2", "#991b1b", "#ef4444"),
            "optional": ("○ Optional", "#f1f5f9", "#475569", "#94a3b8"),
        }
        for check in checks:
            label = self._setup_status_labels.get(check.key)
            if label is None:
                continue
            text, background, foreground, border = styles[check.level]
            label.setText(text)
            label.setToolTip(f"{check.label}: {check.detail}")
            label.setStyleSheet(
                f"background: {background}; color: {foreground}; "
                f"border: 1px solid {border}; border-radius: 5px; "
                "padding: 4px 7px; font-weight: 700;"
            )
            if check.key == "identity":
                self.identity_summary_label.setText(check.detail)
        if self.manifest is not None and self.paths is not None:
            self.create_button.setEnabled(False)
            self.next_setup_problem_button.setEnabled(False)
            self.setup_readiness_label.setText(
                f"● Workspace open — {self.manifest.name}. Use Reload all selected "
                "components below, or choose Set up a new workspace."
            )
            self.setup_readiness_label.setStyleSheet(
                "background: #dcfce7; color: #166534; border: 2px solid #22c55e; "
                "border-radius: 7px; padding: 8px; font-weight: 800;"
            )
            return
        ready = setup_is_ready(checks)
        self.create_button.setEnabled(ready)
        self.next_setup_problem_button.setEnabled(not ready)
        problems = [
            check for check in checks if check.level in {"blocked", "check"}
        ]
        if ready:
            self.setup_readiness_label.setText(
                "● Ready — create the workspace and start the selected workflow."
            )
            style = (
                "background: #dcfce7; color: #166534; border: 2px solid #22c55e;"
            )
        else:
            first = problems[0].detail if problems else "Complete the setup below."
            self.setup_readiness_label.setText(
                f"● {len(problems)} item(s) need attention — {first}"
            )
            has_blocker = any(check.level == "blocked" for check in problems)
            style = (
                "background: #fee2e2; color: #991b1b; border: 2px solid #ef4444;"
                if has_blocker
                else "background: #fef3c7; color: #92400e; border: 2px solid #f59e0b;"
            )
        self.setup_readiness_label.setStyleSheet(
            style + " border-radius: 7px; padding: 8px; font-weight: 800;"
        )
        self.create_button.setToolTip(
            "\n".join(f"{check.label}: {check.detail}" for check in problems)
            if problems
            else "Create the new workspace."
        )

    def current_workflow_mode(self) -> str | None:
        value = self.workflow_combo.currentData()
        return str(value) if value is not None else None

    def _recipe_tracking_enabled(self) -> bool:
        return bool(
            hasattr(self, "live_recipe_tracking_check")
            and self.live_recipe_tracking_check.isChecked()
        )

    def _live_recipe_tracking_changed(self, enabled: bool) -> None:
        if enabled:
            for layer in self.viewer.layers:
                self._bind_recipe_display_tracking(layer)
            self._refresh_reload_recipe_list(force=True)
            self.set_status(
                "Live recipe tracking enabled. Manual visibility, opacity, colour, "
                "contour, and contrast changes will update the working recipe."
            )
        else:
            self.set_status(
                "Live recipe tracking disabled. Explicit recipe controls and saved "
                "Population QC views still work, but manual layer changes are not "
                "recorded automatically."
            )

    def _invalidate_population_qc_caches(self) -> None:
        self._population_qc_cohort_selector = None
        self._population_qc_marker_cache.clear()
        self._population_qc_ranking_cache.clear()
        self._adata_roi_positions.clear()
        self._roi_level_metadata = None
        self._cohort_ids_by_roi.clear()

    def _ensure_roi_level_metadata(self) -> dict[str, dict[str, object]]:
        """Detect sample-level obs fields once, then reuse them on ROI changes."""

        if self._roi_level_metadata is not None:
            return self._roi_level_metadata
        if self.adata is None:
            self._roi_level_metadata = {}
            return self._roi_level_metadata
        roi_obs = (
            self.manifest.roi_obs
            if self.manifest is not None
            else self.roi_obs_edit.text().strip()
        )
        object_obs = (
            self.manifest.object_id_obs
            if self.manifest is not None
            else self.object_obs_edit.text().strip()
        )
        if not roi_obs or roi_obs not in self.adata.obs:
            self._roi_level_metadata = {}
            return self._roi_level_metadata
        self._roi_level_metadata = roi_level_metadata(
            self.adata.obs,
            roi_obs=roi_obs,
            exclude_columns=(object_obs,),
        )
        return self._roi_level_metadata

    def _refresh_roi_metadata_display(self) -> None:
        """Show the current ROI's cached sample metadata in both review tabs."""

        if not hasattr(self, "explore_roi_metadata_table"):
            return
        roi = str(self.current_roi or "")
        values = self._ensure_roi_level_metadata().get(roi, {}) if roi else {}
        targets = (
            (self.explore_roi_metadata_summary, self.explore_roi_metadata_table),
            (
                self.population_qc_roi_metadata_summary,
                self.population_qc_roi_metadata_table,
            ),
        )
        if not roi:
            summary = "Load an ROI to show automatically detected sample metadata."
        elif values:
            summary = (
                f"ROI {roi!r}: {len(values):,} sample-level field(s) detected "
                "because they are constant within every ROI."
            )
        else:
            summary = (
                f"ROI {roi!r}: no additional obs fields are constant within "
                "every ROI."
            )
        for label, table in targets:
            label.setText(summary)
            table.setRowCount(0)
            for row, (field, value) in enumerate(values.items()):
                table.insertRow(row)
                field_item = self.QTableWidgetItem(str(field))
                value_item = self.QTableWidgetItem(format_roi_metadata_value(value))
                value_item.setToolTip(
                    f"Stored Python/pandas type: {type(value).__name__}"
                )
                table.setItem(row, 0, field_item)
                table.setItem(row, 1, value_item)

    def _invalidate_integrity_result(self, *_args) -> None:
        self._integrity_signature = None
        if hasattr(self, "integrity_status_label"):
            self.integrity_status_label.setText(
                "Dataset or cohort settings changed. Run Check dataset integrity "
                "before creating a new workspace."
            )
        self.refresh_setup_readiness()

    def _invalidate_dataset_indexes(self, *_args) -> None:
        self._mask_path_index.clear()
        self._roi_image_path_index.clear()
        self._asset_index_signature = None
        self._invalidate_integrity_result()
        self._invalidate_population_qc_caches()
        self._clear_explore_layer_data_cache()
        self.refresh_setup_readiness()

    def _channel_aliases(self) -> dict[str, str]:
        if self.adata is None:
            return {}
        return build_image_channel_aliases(self.adata.var_names, self.adata.var)

    def _current_asset_index_signature(self) -> str:
        def resolved(value: str) -> str:
            value = value.strip()
            return (
                str(Path(value).expanduser().resolve(strict=False)) if value else ""
            )

        payload = {
            "masks_folder": resolved(self.masks_edit.text().strip()),
            "images_folders": [
                resolved(path) for path in _split_paths(self.images_edit.toPlainText())
            ],
            "extra_images_folders": [
                resolved(path)
                for path in _split_paths(self.extra_images_edit.toPlainText())
            ],
            "channel_aliases": sorted(self._channel_aliases().items()),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _current_integrity_signature(self) -> str:
        payload = {
            "assets": self._current_asset_index_signature(),
            "adata_identity": id(self.adata),
            "adata_cells": int(self.adata.n_obs) if self.adata is not None else None,
            "roi_obs": self.roi_obs_edit.text().strip(),
            "object_obs": self.object_obs_edit.text().strip(),
            "scope": self.scope_combo.currentData(),
            "obs": self.obs_combo.currentText(),
            "values": sorted(self.selected_scope_values()),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _integrity_index_path(root: Path) -> Path:
        return Path(root) / "inputs" / "integrity_index.json"

    def _write_integrity_index(self, root: Path) -> None:
        if self._asset_index_signature is None:
            return
        write_json(
            self._integrity_index_path(root),
            {
                "schema_version": 1,
                "asset_signature": self._asset_index_signature,
                "masks": {
                    roi: str(path.expanduser().resolve(strict=False))
                    for roi, path in self._mask_path_index.items()
                },
                "images": {
                    roi: {
                        channel: str(path.expanduser().resolve(strict=False))
                        for channel, path in channels.items()
                    }
                    for roi, channels in self._roi_image_path_index.items()
                },
            },
        )

    def _load_integrity_index(self) -> bool:
        if self.paths is None:
            return False
        source = self._integrity_index_path(self.paths.root)
        if not source.is_file():
            return False
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        signature = self._current_asset_index_signature()
        if payload.get("asset_signature") != signature:
            return False
        self._mask_path_index = {
            str(roi): Path(path) for roi, path in payload.get("masks", {}).items()
        }
        self._roi_image_path_index = {
            str(roi): {
                str(channel): Path(path) for channel, path in channels.items()
            }
            for roi, channels in payload.get("images", {}).items()
        }
        self._asset_index_signature = signature
        self.integrity_status_label.setText(
            f"Loaded the saved fast asset index: {len(self._mask_path_index):,} "
            f"masks and {sum(map(len, self._roi_image_path_index.values())):,} "
            "images. Revalidate explicitly if source files changed."
        )
        return True

    def _mask_path_for_roi(self, roi: str) -> Path:
        roi = str(roi)
        cached = self._mask_path_index.get(roi)
        if cached is not None and cached.is_file():
            return cached
        direct = resolve_mask_file(self.manifest.masks_folder, roi)
        if direct is None:
            raise FileNotFoundError(
                f"No directly named mask was found for ROI {roi!r}. Run Setup → "
                "Validate integrity to rebuild the complete asset index."
            )
        self._mask_path_index[roi] = direct
        return direct

    def _image_paths_for_roi(self, roi: str) -> dict[str, Path]:
        roi = str(roi)
        cached = self._roi_image_path_index.get(roi)
        if cached is not None:
            return dict(cached)
        paths = discover_roi_images(
            self.manifest.images_folders + self.manifest.extra_images_folders,
            roi,
            channel_aliases=self._channel_aliases(),
            scan_flat_folder=False,
        )
        self._roi_image_path_index[roi] = dict(paths)
        return dict(paths)

    def _update_workflow_mode(self) -> None:
        """Show only the tabs relevant to the selected session workflow."""

        mode = self.current_workflow_mode()
        self._sync_workflow_cards()
        if mode != self._recipe_tracking_workflow:
            self.live_recipe_tracking_check.blockSignals(True)
            self.live_recipe_tracking_check.setChecked(mode != "population_qc")
            self.live_recipe_tracking_check.blockSignals(False)
            self._recipe_tracking_workflow = mode
        visible_topics = WORKFLOW_VISIBLE_TABS.get(mode, {"setup"})
        for topic, index in self._workflow_tab_indices.items():
            self.tabs.setTabVisible(index, topic in visible_topics)
        setup_index = self._workflow_tab_indices.get("setup", 0)
        if not self.tabs.isTabVisible(self.tabs.currentIndex()):
            self.tabs.setCurrentIndex(setup_index)
        classification_setup = mode in {"classification", "full_workspace"}
        self.classification_setup_widget.setVisible(classification_setup)
        self.classification_export_group.setVisible(classification_setup)
        self.create_button.setEnabled(mode is not None)
        self.workflow_description_label.setText(
            WORKFLOW_DESCRIPTIONS.get(
                mode,
                "Choose what you want to do. Only Setup remains visible until a "
                "workflow is selected.",
            )
        )
        if not classification_setup:
            self.scope_combo.setCurrentIndex(self.scope_combo.findData("all_cells"))
            self.experiment_mode_combo.setCurrentIndex(
                self.experiment_mode_combo.findData("full")
            )
        if mode not in {"classification", "full_workspace"}:
            self._remove_layers(
                [
                    NONCONTEXT_MASK_LAYER_NAME,
                    *CLASS_LAYER_NAMES.values(),
                    SELECTED_CELL_LAYER_NAME,
                ]
            )
            for shortcut in self._class_shortcuts:
                try:
                    self.viewer.bind_key(shortcut, None, overwrite=True)
                except Exception as error:  # noqa: BLE001 - optional Napari backend
                    self.set_status(
                        f"Could not disable hidden classifier shortcut "
                        f"{shortcut!r}: {error}"
                    )
            self._class_shortcuts = []
        if mode not in {"cell_labeling", "full_workspace"}:
            self._remove_layers([LABELER_LAYER_NAME, LABELER_SELECTED_CELL_LAYER_NAME])
        if (
            mode is not None
            and self.manifest is not None
            and self.paths is not None
            and self.manifest.workflow_mode != mode
        ):
            updated = self.manifest.model_copy(deep=True)
            updated.workflow_mode = mode
            save_experiment(
                updated,
                self.paths.root,
                audit_action="change_workflow_mode",
            )
            self.manifest = updated
        if classification_setup and self.manifest is not None:
            self.refresh_class_controls()
        self._refresh_reload_recipe_list()
        if mode is not None:
            self.set_status(
                f"Workflow view set to {self.workflow_combo.currentText()!r}. "
                "The workspace still uses the same experiment-backed data model."
            )
        self.refresh_setup_readiness()

    def _normalization_from_editor(self) -> dict[str, float]:
        payload: dict[str, float] = {}
        for row in range(self.normalization_table.rowCount()):
            marker_item = self.normalization_table.item(row, 0)
            value_item = self.normalization_table.item(row, 1)
            marker = marker_item.text().strip() if marker_item is not None else ""
            value_text = value_item.text().strip() if value_item is not None else ""
            if not marker and not value_text:
                continue
            if not marker or not value_text:
                raise ValueError(
                    f"Normalization row {row + 1} requires both Marker and Value."
                )
            if marker in payload:
                raise ValueError(f"Normalization marker {marker!r} is duplicated.")
            try:
                payload[marker] = float(value_text)
            except ValueError as exc:
                raise ValueError(
                    f"Normalization value for {marker!r} must be a number."
                ) from exc
        return prepare_normalization_dict(payload)

    def _set_normalization_table(self, mapping: dict[str, float]) -> None:
        self.normalization_table.blockSignals(True)
        try:
            self.normalization_table.setRowCount(len(mapping))
            for row, (marker, value) in enumerate(sorted(mapping.items())):
                self.normalization_table.setItem(
                    row, 0, self.QTableWidgetItem(str(marker))
                )
                self.normalization_table.setItem(
                    row, 1, self.QTableWidgetItem(f"{float(value):g}")
                )
        finally:
            self.normalization_table.blockSignals(False)
        self._sync_normalization_json_preview()

    def add_normalization_row(self) -> None:
        row = self.normalization_table.rowCount()
        self.normalization_table.insertRow(row)
        self.normalization_table.setItem(row, 0, self.QTableWidgetItem(""))
        self.normalization_table.setItem(row, 1, self.QTableWidgetItem(""))
        self.normalization_table.setCurrentCell(row, 0)
        self.normalization_table.editItem(self.normalization_table.item(row, 0))

    def remove_selected_normalization_rows(self) -> None:
        rows = sorted(
            {index.row() for index in self.normalization_table.selectedIndexes()},
            reverse=True,
        )
        for row in rows:
            self.normalization_table.removeRow(row)
        self._sync_normalization_json_preview()

    def _sync_normalization_json_preview(self, *_args) -> None:
        try:
            payload = self._normalization_from_editor()
            text = json.dumps(payload, indent=2, sort_keys=True)
        except ValueError as exc:
            text = json.dumps({"needs_attention": str(exc)}, indent=2)
        self.normalization_json_edit.setPlainText(text)

    def choose_normalization_json(self) -> None:
        selected, _filter = self.QFileDialog.getOpenFileName(
            self.root,
            "Choose Nimbus normalization JSON or CSV",
            self.normalization_edit.text().strip() or str(self.project_root),
            "Normalization files (*.json *.csv);;JSON files (*.json);;"
            "CSV files (*.csv)",
        )
        if selected:
            self.normalization_edit.setText(selected)
            self.load_normalization_json()

    def load_normalization_json(self) -> None:
        source = Path(self.normalization_edit.text().strip()).expanduser()
        self.display_normalization = load_normalization_mapping(source)
        self._set_normalization_table(self.display_normalization)
        self._clear_explore_layer_data_cache()
        self.normalization_status_label.setText(
            f"Loaded {len(self.display_normalization):,} channel maxima from "
            f"{source}. Save the workspace to create an experiment-backed copy."
        )
        self._refresh_feature_normalization_summary()

    def validate_normalization_editor(self) -> None:
        self.display_normalization = self._normalization_from_editor()
        self._clear_explore_layer_data_cache()
        self.normalization_status_label.setText(
            f"Valid normalization mapping: {len(self.display_normalization):,} "
            "channel maxima. Save it into the experiment to persist edits."
        )
        self._refresh_feature_normalization_summary()

    def _display_settings_from_controls(
        self, *, normalization_path: str | None = None
    ) -> DisplaySettings:
        return DisplaySettings(
            normalization_dict_path=normalization_path,
            fallback_quantile=float(self.display_quantile_spin.value()),
            minimum_pixel_counts=float(self.display_minimum_pixel_spin.value()),
            default_contrast_limits=(
                float(self.display_lower_contrast_spin.value()),
                float(self.display_upper_contrast_spin.value()),
            ),
        )

    def _write_experiment_normalization(self, root: Path) -> str | None:
        self.display_normalization = self._normalization_from_editor()
        destination = root / "display" / "normalization.json"
        write_json(
            destination,
            {"normalization_dict": self.display_normalization},
        )
        self.normalization_edit.setText(str(destination))
        return str(destination)

    def save_normalization_to_experiment(self) -> None:
        if self.manifest is None or self.paths is None:
            raise ValueError(
                "Create or load a workflow workspace before saving its "
                "normalization settings."
            )
        normalization_path = self._write_experiment_normalization(self.paths.root)
        updated = self.manifest.model_copy(deep=True)
        updated.display_settings = self._display_settings_from_controls(
            normalization_path=normalization_path
        )
        updated.synthetic_features.normalization_dict_path = normalization_path
        save_experiment(
            updated,
            self.paths.root,
            audit_action="update_display_normalization",
        )
        self.manifest = updated
        self._clear_explore_layer_data_cache()
        self.normalization_status_label.setText(
            f"Saved {len(self.display_normalization):,} channel maxima and display "
            f"defaults inside {self.paths.root / 'display'}."
        )
        self._refresh_feature_normalization_summary()

    def _refresh_feature_normalization_summary(self) -> None:
        if not hasattr(self, "feature_normalization_summary"):
            return
        source = self.normalization_edit.text().strip() or "none"
        self.feature_normalization_summary.setText(
            f"Configured in Setup: {len(self.display_normalization):,} fixed "
            f"channel maxima; source/copy: {source}. Unmatched channels use "
            f"quantile {self.display_quantile_spin.value():.4f}."
        )

    def _display_defaults_changed(self) -> None:
        lower = float(self.display_lower_contrast_spin.value())
        upper = float(self.display_upper_contrast_spin.value())
        if lower >= upper:
            self.normalization_status_label.setText(
                "Default contrast lower must remain below upper. Existing recipe "
                "ranges are unaffected."
            )
            return
        self._sync_population_qc_contrast_defaults()
        self._clear_explore_layer_data_cache()
        self._refresh_feature_normalization_summary()
        self.normalization_status_label.setText(
            f"New unspecific image layers will use contrast {lower:.3f}-{upper:.3f}. "
            "Population QC views without saved or manually edited ranges inherit "
            "these values; saved recipe-specific ranges still take precedence."
        )

    def _population_qc_setup_contrast_limits(self) -> tuple[float, float]:
        settings = self._display_settings_from_controls()
        return tuple(float(value) for value in settings.default_contrast_limits)

    def _update_population_qc_contrast_defaults_label(self) -> None:
        if not hasattr(self, "population_qc_contrast_defaults_label"):
            return
        lower, upper = self._population_qc_setup_contrast_limits()
        self.population_qc_contrast_defaults_label.setText(
            f"Setup default: {lower:.3f}–{upper:.3f}. It initializes populations "
            "without a saved RGB recipe. Edit any per-channel value below to "
            "override it for this population."
        )

    def _sync_population_qc_contrast_defaults(self, *, force: bool = False) -> None:
        """Apply Setup defaults only to Population QC controls still inheriting them."""

        if not hasattr(self, "population_qc_lower_spins"):
            return
        new_default = self._population_qc_setup_contrast_limits()
        previous_default = getattr(
            self,
            "_population_qc_last_setup_defaults",
            (0.0, 1.0),
        )
        observation = self.population_qc_obs_combo.currentText().strip()
        population = self.population_qc_population_combo.currentText().strip()
        has_saved_recipe = bool(
            observation
            and population
            and population_recipe_key(observation, population)
            in self.explore_review_state.population_recipes
        )
        for colour in ("red", "green", "blue"):
            lower_spin = self.population_qc_lower_spins[colour]
            upper_spin = self.population_qc_upper_spins[colour]
            current = (float(lower_spin.value()), float(upper_spin.value()))
            limits = (
                new_default
                if force
                else inherit_setup_contrast_limits(
                    current,
                    previous_default,
                    new_default,
                    has_saved_recipe=has_saved_recipe,
                )
            )
            lower_blocked = lower_spin.blockSignals(True)
            upper_blocked = upper_spin.blockSignals(True)
            lower_spin.setValue(float(limits[0]))
            upper_spin.setValue(float(limits[1]))
            lower_spin.blockSignals(lower_blocked)
            upper_spin.blockSignals(upper_blocked)
        self._population_qc_last_setup_defaults = new_default
        self._update_population_qc_contrast_defaults_label()

    def reset_population_qc_contrasts_to_setup_defaults(self) -> None:
        """Explicitly replace all three working RGB ranges with Setup defaults."""

        self._sync_population_qc_contrast_defaults(force=True)
        lower, upper = self._population_qc_setup_contrast_limits()
        self.population_qc_status_label.setText(
            f"Reset all working RGB contrasts to the Setup default "
            f"{lower:.3f}–{upper:.3f}. Edit individual channels if needed, then "
            "save or load the population view."
        )

    def _set_classification_enabled(self, enabled: bool) -> None:
        for widget in (
            self.class_combo,
            *self.click_behavior_radios.values(),
            self.propose_button,
            self.confirm_button,
            self.clear_proposed_button,
            self.clear_all_proposals_button,
            self.confirm_proposed_button,
            self.train_button,
            self.score_button,
            self.refresh_queue_button,
            self.bulk_propose_button,
            self.apply_prediction_review_button,
            self.reset_prediction_review_button,
            self.create_final_identities_button,
            self.export_assignments_button,
            self.export_adata_button,
            self.apply_live_adata_button,
            self.labeler_class_combo,
            *self.labeler_click_behavior_radios.values(),
            self.assign_labeler_cell_button,
            self.clear_labeler_cell_button,
            self.clear_all_labeler_button,
            self.previous_labeler_roi_button,
            self.next_labeler_roi_button,
            self.next_unsampled_labeler_roi_button,
            self.export_labeler_csv_button,
            self.apply_labeler_to_adata_button,
            self.export_cohort_masks_button,
            self.export_clean_masks_button,
            self.import_current_classifier_components_button,
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

    def _populate_anndata_selectors(self, *, source: str) -> None:
        if self.adata is None:
            raise RuntimeError("No AnnData object is available.")
        # Any live AnnData mutation can change observation or marker overlays.
        # Drop cached derived arrays before rebuilding their selectors.
        self._clear_explore_layer_data_cache()
        self._invalidate_population_qc_caches()
        columns = [str(column) for column in self.adata.obs.columns]
        if self.manifest is None:
            suggested_roi, suggested_object = suggest_identity_columns(columns)
            if self.roi_obs_edit.text().strip() not in columns and suggested_roi:
                self.roi_obs_edit.setText(suggested_roi)
            if (
                self.object_obs_edit.text().strip() not in columns
                and suggested_object
            ):
                self.object_obs_edit.setText(suggested_object)
        selector_combos = (
            self.obs_combo,
            self.overlay_obs_combo,
            self.population_obs_combo,
            self.population_qc_obs_combo,
            self.curation_source_combo,
        )
        previous_values = {id(combo): combo.currentText() for combo in selector_combos}
        for combo in selector_combos:
            current = combo.currentText()
            combo.blockSignals(True)
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
            for combo in (
                self.obs_combo,
                self.population_obs_combo,
                self.population_qc_obs_combo,
                self.curation_source_combo,
            ):
                if previous_values[id(combo)] not in columns:
                    preferred_value = preferred[0]
                    if combo is self.curation_source_combo:
                        preferred_value = next(
                            (
                                column
                                for column in preferred
                                if "leiden" in column.lower()
                            ),
                            preferred_value,
                        )
                    combo.setCurrentText(preferred_value)
        for combo in selector_combos:
            combo.blockSignals(False)
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
        self.refresh_population_qc_populations()
        self.refresh_feature_channel_choices()
        self._refresh_population_data_choices()
        self.refresh_population_workspace()
        self.mark_scanpy_plots_stale()
        self.refresh_scanpy_plotting_choices()
        self.refresh_setup_readiness()
        self.set_status(
            f"Loaded AnnData selectors for {self.adata.n_obs:,} cells from {source}."
        )

    def load_anndata_selectors(self) -> None:
        path_text = self.anndata_edit.text().strip()
        if path_text:
            import anndata as ad

            path = Path(path_text).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"AnnData not found: {path}")
            self.adata = ad.read_h5ad(path)
            self._in_memory_adata = None
            source = str(path)
        elif self._in_memory_adata is not None:
            self.adata = self._in_memory_adata
            source = "the live in-memory object"
        else:
            raise ValueError("Supply an AnnData path or launch with an AnnData object.")
        self._populate_anndata_selectors(source=source)

    def _population_curation_root(self) -> Path:
        if self.paths is not None:
            return self.paths.root / "population_curation"
        return self.project_root / "napari_sbt" / "population_curation"

    def _refresh_population_data_choices(self) -> None:
        """Refresh source-population and subclustering representation choices."""

        if self.adata is None:
            return
        source_obs = self.curation_source_combo.currentText().strip()
        selected_values = {
            item.text() for item in self.population_split_values_list.selectedItems()
        }
        self.population_split_values_list.clear()
        if source_obs in self.adata.obs:
            series = self.adata.obs[source_obs]
            values = ordered_source_labels(series)
            self.population_split_values_list.addItems(values)
            for index in range(self.population_split_values_list.count()):
                item = self.population_split_values_list.item(index)
                item.setSelected(item.text() in selected_values)

        current_representation = self.population_representation_combo.currentText()
        self.population_representation_combo.clear()
        representation_keys = [
            str(key)
            for key, value in self.adata.obsm.items()
            if getattr(value, "ndim", 0) == 2
            and value.shape[1] >= 1
            and not any(
                token in str(key).lower() for token in ("umap", "tsne", "spatial")
            )
        ]
        self.population_representation_combo.addItems(representation_keys)
        if current_representation in representation_keys:
            self.population_representation_combo.setCurrentText(current_representation)
        elif "X_biobatchnet" in representation_keys:
            self.population_representation_combo.setCurrentText("X_biobatchnet")
        else:
            # Do not silently substitute PCA or another embedding when the
            # requested batch-corrected representation is absent.
            self.population_representation_combo.setCurrentIndex(-1)

        current_graph = self.population_adjacency_combo.currentText()
        self.population_adjacency_combo.clear()
        graph_keys = [str(key) for key in self.adata.obsp.keys()]
        self.population_adjacency_combo.addItems(graph_keys)
        if current_graph in graph_keys:
            self.population_adjacency_combo.setCurrentText(current_graph)
        elif "connectivities" in graph_keys:
            self.population_adjacency_combo.setCurrentText("connectivities")
        self._update_population_neighbor_controls()

    def _update_population_graph_provenance(self) -> None:
        if self.adata is None:
            return
        if self.population_neighbor_source_combo.currentData() == "rebuild_from_rep":
            representation_key = (
                self.population_representation_combo.currentText().strip()
            )
            if not representation_key:
                self.population_graph_provenance_label.setText(
                    "No corrected representation is selected. X_biobatchnet is "
                    "the required default; if it is absent, choose another obsm "
                    "only after verifying that it is batch corrected."
                )
                return
            representation = self.adata.obsm[representation_key]
            correction_note = (
                "BioBatchNet-corrected representation selected."
                if representation_key == "X_biobatchnet"
                else "Verify that this representation is already batch corrected."
            )
            self.population_graph_provenance_label.setText(
                f"adata.obsm[{representation_key!r}], shape={representation.shape}. "
                f"Neighbours will be rebuilt only within the selected cells using "
                f"n_neighbors={self.population_n_neighbors_spin.value()}. "
                f"{correction_note}"
            )
            return
        graph_key = self.population_adjacency_combo.currentText().strip()
        if not graph_key:
            self.population_graph_provenance_label.setText(
                "No square connectivity graph is available in adata.obsp."
            )
            return
        matches = []
        for uns_key, payload in self.adata.uns.items():
            if not isinstance(payload, dict):
                continue
            if str(payload.get("connectivities_key", "")) != graph_key and not (
                graph_key == "connectivities" and uns_key == "neighbors"
            ):
                continue
            params = payload.get("params", {})
            use_rep = params.get("use_rep") if isinstance(params, dict) else None
            method = params.get("method") if isinstance(params, dict) else None
            details = [f"uns[{uns_key!r}]"]
            if use_rep:
                details.append(f"use_rep={use_rep}")
            if method:
                details.append(f"method={method}")
            matches.append(", ".join(details))
        graph = self.adata.obsp.get(graph_key)
        edge_count = getattr(graph, "nnz", "unknown")
        provenance = "; ".join(matches) if matches else "no linked uns metadata found"
        self.population_graph_provenance_label.setText(
            f"adata.obsp[{graph_key!r}], shape={getattr(graph, 'shape', None)}, "
            f"stored edges={edge_count}; {provenance}. The worker reuses this "
            "graph exactly and records the selected key."
        )

    def _update_population_neighbor_controls(self) -> None:
        rebuild = (
            self.population_neighbor_source_combo.currentData() == "rebuild_from_rep"
        )
        self.population_representation_combo.setEnabled(rebuild)
        self.population_n_neighbors_spin.setEnabled(rebuild)
        self.population_adjacency_combo.setEnabled(not rebuild)
        self._update_population_graph_provenance()

    def _mark_population_draft_dirty(self, *_args) -> None:
        if self.population_draft is None:
            return
        self._population_draft_dirty = True
        self._refresh_population_naming_readiness()

    def _refresh_population_naming_readiness(self) -> None:
        """Show whether edits are saved and visible in the rest of the app."""

        if not hasattr(self, "population_naming_readiness_label"):
            return
        if self.population_draft is None or self.adata is None:
            text = "● Not started — choose a source and create a new label draft."
            style = (
                "background: #fee2e2; color: #991b1b; border: 1px solid #ef4444;"
            )
            self.save_population_draft_button.setEnabled(False)
        elif self._population_draft_dirty:
            text = (
                "● Unsaved changes — Save and update the app before reviewing "
                "these names in Explore or Population QC."
            )
            style = (
                "background: #fef3c7; color: #92400e; border: 1px solid #f59e0b;"
            )
            self.save_population_draft_button.setEnabled(True)
        else:
            sync_state = population_draft_sync_state(
                self.adata, self.population_draft
            )
            if sync_state == "synced":
                text = (
                    f"● Ready — adata.obs[{self.population_draft.derived_obs!r}] "
                    "matches this saved draft and is available in Explore and "
                    "Population QC."
                )
                style = (
                    "background: #dcfce7; color: #166534; "
                    "border: 1px solid #22c55e;"
                )
            elif sync_state == "conflict":
                text = (
                    f"● Name conflict — adata.obs[{self.population_draft.derived_obs!r}] "
                    "exists but belongs to something else. Choose another name, or "
                    "use the advanced overwrite option deliberately."
                )
                style = (
                    "background: #fee2e2; color: #991b1b; "
                    "border: 1px solid #ef4444;"
                )
            else:
                text = (
                    "● Saved but not synchronized — use Save and update the app "
                    "to make this revision available in Explore and Population QC."
                )
                style = (
                    "background: #fef3c7; color: #92400e; "
                    "border: 1px solid #f59e0b;"
                )
            self.save_population_draft_button.setEnabled(True)
        self.population_naming_readiness_label.setText(text)
        self.population_naming_readiness_label.setStyleSheet(
            style + " border-radius: 6px; padding: 7px; font-weight: 700;"
        )

    def refresh_population_workspace(self) -> None:
        """Discover sibling drafts for the currently selected immutable source."""

        self._refresh_population_data_choices()
        if self.adata is None:
            return
        source_obs = self.curation_source_combo.currentText().strip()
        if not source_obs or source_obs not in self.adata.obs:
            return
        previous_draft_id = (
            self.population_draft.draft_id
            if self.population_draft is not None
            and self.population_draft.source_obs == source_obs
            else self.curation_draft_combo.currentData()
        )
        candidate_paths = population_workspace_paths(
            self._population_curation_root(), source_obs
        )
        self.curation_draft_combo.blockSignals(True)
        self.curation_draft_combo.clear()
        if not candidate_paths.manifest.is_file():
            self.population_workspace = None
            self.population_workspace_paths = None
            self.population_draft = None
            self.population_base_mapping = pd.DataFrame(columns=BASE_MAPPING_COLUMNS)
            self.population_components = empty_components()
            self.population_membership = empty_membership()
            self._population_draft_dirty = False
            self._set_population_tables(
                self.population_base_mapping,
                self.population_components,
            )
            self.population_workspace_label.setText(
                f"Original obs {source_obs!r} will remain unchanged. Create the "
                "first naming draft to begin."
            )
            default_obs = f"{slugify(source_obs)}_named"
            if not self.curation_derived_obs_edit.text().strip() or (
                self.curation_derived_obs_edit.text().strip() == "population_curated"
            ):
                self.curation_derived_obs_edit.setText(default_obs)
            self.curation_draft_combo.blockSignals(False)
            self.refresh_population_provenance()
            self._refresh_population_naming_readiness()
            return

        workspace, workspace_paths = ensure_population_workspace(
            self.adata,
            self._population_curation_root(),
            source_obs,
        )
        self.population_workspace = workspace
        self.population_workspace_paths = workspace_paths
        drafts = list_population_drafts(workspace_paths)
        for draft in drafts:
            sync_state = population_draft_sync_state(self.adata, draft)
            status = "✓ synced" if sync_state == "synced" else "○ saved"
            self.curation_draft_combo.addItem(
                f"{status} — {draft.derived_obs} (r{draft.revision})",
                draft.draft_id,
            )
        index = self.curation_draft_combo.findData(previous_draft_id)
        if index < 0 and drafts:
            index = 0
        if index >= 0:
            self.curation_draft_combo.setCurrentIndex(index)
        self.curation_draft_combo.blockSignals(False)
        self.population_workspace_label.setText(
            f"Original obs {workspace.source_obs!r} is protected; "
            f"{len(drafts)} saved label draft(s) use it as their source."
        )
        if index >= 0:
            self.load_selected_population_draft()
        else:
            self.refresh_population_provenance()
            self._refresh_population_naming_readiness()

    def create_population_draft(self) -> None:
        if self.adata is None:
            raise RuntimeError("Load AnnData before creating a population draft.")
        source_obs = self.curation_source_combo.currentText().strip()
        derived_obs = self.curation_derived_obs_edit.text().strip()
        if not source_obs:
            raise ValueError("Choose the original source observation first.")
        if not derived_obs:
            raise ValueError("Enter the new AnnData label-column name first.")
        (
            workspace,
            workspace_paths,
            draft,
            base,
            components,
            membership,
        ) = create_population_draft_asset(
            self.adata,
            self._population_curation_root(),
            source_obs=source_obs,
            name=derived_obs,
            derived_obs=derived_obs,
        )
        self.population_workspace = workspace
        self.population_workspace_paths = workspace_paths
        self.population_draft = draft
        self.population_base_mapping = base
        self.population_components = components
        self.population_membership = membership
        self.refresh_population_workspace()
        index = self.curation_draft_combo.findData(draft.draft_id)
        if index >= 0:
            self.curation_draft_combo.setCurrentIndex(index)
        self.population_editor_tabs.setCurrentIndex(0)
        self.set_status(
            f"Created naming draft for adata.obs[{derived_obs!r}]. Rename or "
            "merge populations, then save to update Explore and Population QC."
        )

    def load_selected_population_draft(self) -> None:
        if self.population_workspace_paths is None:
            return
        draft_id = self.curation_draft_combo.currentData()
        if not draft_id:
            return
        if (
            self._population_draft_dirty
            and self.population_draft is not None
            and str(draft_id) != self.population_draft.draft_id
        ):
            reply = self.QMessageBox.question(
                self.root,
                "Discard unsaved population names?",
                "This naming draft has unsaved edits. Discard them and open the "
                "selected saved work?",
            )
            if reply != self.QMessageBox.Yes:
                previous = self.curation_draft_combo.findData(
                    self.population_draft.draft_id
                )
                blocked = self.curation_draft_combo.blockSignals(True)
                self.curation_draft_combo.setCurrentIndex(previous)
                self.curation_draft_combo.blockSignals(blocked)
                return
        draft, base, components, membership = load_population_draft(
            self.population_workspace_paths, str(draft_id)
        )
        self.population_draft = draft
        self.population_base_mapping = base
        self.population_components = components
        self.population_membership = membership
        self.curation_derived_obs_edit.setText(draft.derived_obs)
        self._population_draft_dirty = False
        self._set_population_tables(base, components)
        self._refresh_population_merge_preview()
        self.refresh_population_provenance()
        self._refresh_population_naming_readiness()
        self.set_status(
            f"Loaded saved naming work for adata.obs[{draft.derived_obs!r}] "
            f"r{draft.revision}."
        )

    def _readonly_table_item(self, value: object):
        item = self.QTableWidgetItem(str(value))
        item.setFlags(item.flags() & ~self.Qt.ItemIsEditable)
        return item

    def _set_population_tables(
        self,
        base_mapping: pd.DataFrame,
        components: pd.DataFrame,
    ) -> None:
        self.population_base_table.blockSignals(True)
        self.population_base_table.setRowCount(0)
        for row_index, row in base_mapping.reset_index(drop=True).iterrows():
            self.population_base_table.insertRow(row_index)
            self.population_base_table.setItem(
                row_index, 0, self._readonly_table_item(row["source_value"])
            )
            self.population_base_table.setItem(
                row_index,
                1,
                self._readonly_table_item(f"{int(row['cell_count']):,}"),
            )
            for column, field in ((2, "proposed_label"), (3, "color"), (4, "notes")):
                item = self.QTableWidgetItem(str(row[field]))
                self.population_base_table.setItem(row_index, column, item)
            colour = self.QColor(str(row["color"]))
            if colour.isValid():
                self.population_base_table.item(row_index, 3).setBackground(colour)
        self.population_base_table.blockSignals(False)

        self.population_components_table.blockSignals(True)
        self.population_components_table.setRowCount(0)
        for row_index, row in components.reset_index(drop=True).iterrows():
            self.population_components_table.insertRow(row_index)
            parent_item = self._readonly_table_item(row["parent_source_value"])
            parent_item.setData(self.Qt.UserRole, str(row["component_id"]))
            self.population_components_table.setItem(row_index, 0, parent_item)
            for column, field in (
                (1, "method"),
                (2, "component_value"),
                (3, "cell_count"),
                (6, "run_id"),
            ):
                value = f"{int(row[field]):,}" if field == "cell_count" else row[field]
                self.population_components_table.setItem(
                    row_index, column, self._readonly_table_item(value)
                )
            for column, field in ((4, "proposed_label"), (5, "color"), (7, "notes")):
                self.population_components_table.setItem(
                    row_index, column, self.QTableWidgetItem(str(row[field]))
                )
            colour = self.QColor(str(row["color"]))
            if colour.isValid():
                self.population_components_table.item(row_index, 5).setBackground(
                    colour
                )
        self.population_components_table.blockSignals(False)

    def _population_tables_to_frames(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        base_rows = []
        for row in range(self.population_base_table.rowCount()):
            base_rows.append(
                {
                    "source_value": self.population_base_table.item(row, 0).text(),
                    "cell_count": self.population_base_table.item(row, 1)
                    .text()
                    .replace(",", ""),
                    "proposed_label": self.population_base_table.item(row, 2).text(),
                    "color": self.population_base_table.item(row, 3).text(),
                    "notes": self.population_base_table.item(row, 4).text(),
                }
            )
        component_rows = []
        for row in range(self.population_components_table.rowCount()):
            component_rows.append(
                {
                    "component_id": self.population_components_table.item(row, 0).data(
                        self.Qt.UserRole
                    ),
                    "parent_source_value": self.population_components_table.item(
                        row, 0
                    ).text(),
                    "method": self.population_components_table.item(row, 1).text(),
                    "component_value": self.population_components_table.item(
                        row, 2
                    ).text(),
                    "cell_count": self.population_components_table.item(row, 3)
                    .text()
                    .replace(",", ""),
                    "proposed_label": self.population_components_table.item(
                        row, 4
                    ).text(),
                    "color": self.population_components_table.item(row, 5).text(),
                    "run_id": self.population_components_table.item(row, 6).text(),
                    "notes": self.population_components_table.item(row, 7).text(),
                }
            )
        return (
            pd.DataFrame(base_rows, columns=BASE_MAPPING_COLUMNS),
            pd.DataFrame(component_rows, columns=COMPONENT_COLUMNS),
        )

    def _population_tables_changed(self) -> None:
        if self.population_draft is None:
            return
        self._mark_population_draft_dirty()
        try:
            for table, colour_column in (
                (self.population_base_table, 3),
                (self.population_components_table, 5),
            ):
                table.blockSignals(True)
                for row in range(table.rowCount()):
                    item = table.item(row, colour_column)
                    colour = self.QColor(item.text().strip())
                    item.setBackground(
                        colour if colour.isValid() else self.QColor("#fecaca")
                    )
                table.blockSignals(False)
            self._refresh_population_merge_preview()
        except Exception as exc:  # allow temporarily incomplete table edits
            self.population_merge_preview.setPlainText(
                f"Finish the current edit to refresh the preview: {exc}"
            )

    def _refresh_population_merge_preview(self) -> None:
        if self.population_draft is None or self.adata is None:
            self.population_merge_preview.setPlainText(
                "Create or load a population draft to preview its effective labels."
            )
            return
        base, components = self._population_tables_to_frames()
        labels, summary = synthesize_population_labels(
            self.adata,
            source_obs=self.population_draft.source_obs,
            base_mapping=base,
            components=components,
            membership=self.population_membership,
        )
        counts = labels.value_counts(dropna=False)
        lines = [
            f"New label column: {self.curation_derived_obs_edit.text().strip()}",
            f"{summary['label_count']:,} effective population(s) across "
            f"{summary['cell_count']:,} cells; {summary['split_cell_count']:,} "
            "cells currently use split-component assignments.",
            "",
            "Effective population counts:",
        ]
        lines.extend(
            f"  • {label}: {int(count):,} cells" for label, count in counts.items()
        )
        groups = summary["merge_groups"]
        lines.append("")
        if groups:
            lines.append("Explicit proposed merges (shared final name):")
            for label, contributors in groups.items():
                lines.append(f"  • {label} ← {', '.join(contributors)}")
        else:
            lines.append("No explicit merges are currently proposed.")
        if summary["missing_source_cells"]:
            lines.append(
                f"WARNING: {summary['missing_source_cells']:,} cells have missing "
                "source labels and remain unassigned."
            )
        colour_rows = pd.concat(
            [
                base[["proposed_label", "color"]],
                components[["proposed_label", "color"]]
                if not components.empty
                else pd.DataFrame(columns=["proposed_label", "color"]),
            ],
            ignore_index=True,
        )
        colour_conflicts = {
            str(label): list(dict.fromkeys(group["color"].astype(str)))
            for label, group in colour_rows.groupby("proposed_label", sort=False)
            if group["color"].astype(str).nunique() > 1 and str(label) in groups
        }
        if colour_conflicts:
            lines.append("")
            lines.append(
                "Colour conflicts (the first colour will be used when applied):"
            )
            for label, colours in colour_conflicts.items():
                lines.append(f"  • {label}: {', '.join(colours)}")
        self.population_merge_preview.setPlainText("\n".join(lines))

        merged_names = set(groups)
        self.population_base_table.blockSignals(True)
        for row in range(self.population_base_table.rowCount()):
            name_item = self.population_base_table.item(row, 2)
            if name_item.text().strip() in merged_names:
                name_item.setBackground(self.QColor("#fed7aa"))
                name_item.setForeground(self.QColor("#7c2d12"))
            else:
                name_item.setData(self.Qt.BackgroundRole, None)
                name_item.setData(self.Qt.ForegroundRole, None)
        self.population_base_table.blockSignals(False)
        self.population_components_table.blockSignals(True)
        for row in range(self.population_components_table.rowCount()):
            name_item = self.population_components_table.item(row, 4)
            if name_item.text().strip() in merged_names:
                name_item.setBackground(self.QColor("#fed7aa"))
                name_item.setForeground(self.QColor("#7c2d12"))
            else:
                name_item.setData(self.Qt.BackgroundRole, None)
                name_item.setData(self.Qt.ForegroundRole, None)
        self.population_components_table.blockSignals(False)

    def _save_current_population_draft(
        self,
        *,
        action: str,
        details: dict[str, object] | None = None,
    ) -> PopulationDraft:
        if (
            self.population_draft is None
            or self.population_workspace_paths is None
            or self.adata is None
        ):
            raise RuntimeError("Create or load a population draft first.")
        base, components = self._population_tables_to_frames()
        draft = self.population_draft.model_copy(
            update={
                "derived_obs": self.curation_derived_obs_edit.text().strip(),
            },
            deep=True,
        )
        updated = save_population_draft(
            self.population_workspace_paths,
            draft,
            base,
            components,
            self.population_membership,
            adata=self.adata,
            action=action,
            details=details,
        )
        self.population_draft = updated
        self.population_base_mapping = base
        self.population_components = components
        self._population_draft_dirty = False
        self._refresh_population_naming_readiness()
        return updated

    def save_current_population_draft(self) -> None:
        if self.population_draft is None:
            raise RuntimeError("Create or choose saved naming work first.")
        sync_context = self._capture_population_qc_naming_context()
        updated = self._save_current_population_draft(action="save_and_sync_labels")
        summary = self._sync_saved_population_draft(updated, sync_context)
        self.set_status(
            f"Saved and synchronized adata.obs[{updated.derived_obs!r}] r"
            f"{updated.revision}: {summary['label_count']} populations, "
            f"{len(summary['merge_groups'])} explicit merge group(s)."
        )

    def _capture_population_qc_naming_context(self) -> dict[str, object]:
        """Capture the current QC focus before AnnData selectors are rebuilt."""

        observation = self.population_qc_obs_combo.currentText().strip()
        population = self.population_qc_population_combo.currentText().strip()
        recipe = None
        if observation and population:
            try:
                recipe = self._population_qc_recipe_from_controls()
            except (TypeError, ValueError):
                recipe = self.explore_review_state.population_recipes.get(
                    population_recipe_key(observation, population)
                )
        source_values: list[str] = []
        selected_positions: list[int] = []
        if self.population_draft is not None and self.adata is not None and population:
            source_obs = self.population_draft.source_obs
            if observation in self.adata.obs:
                selected = (
                    self.adata.obs[observation]
                    .astype("string")
                    .eq(population)
                    .fillna(False)
                )
                selected_positions = np.flatnonzero(selected.to_numpy()).tolist()
            if observation == source_obs:
                source_values = [population]
            elif observation in self.adata.obs and source_obs in self.adata.obs:
                source_values = (
                    self.adata.obs.loc[selected, source_obs]
                    .astype("string")
                    .dropna()
                    .drop_duplicates()
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
        return {
            "observation": observation,
            "population": population,
            "recipe": recipe,
            "source_values": source_values,
            "selected_positions": selected_positions,
        }

    def _preferred_synced_population(
        self,
        draft: PopulationDraft,
        context: dict[str, object],
    ) -> str | None:
        categories = [
            str(value) for value in self.adata.obs[draft.derived_obs].cat.categories
        ]
        selected_positions = [
            int(value) for value in context.get("selected_positions", [])
        ]
        if selected_positions:
            overlap = (
                self.adata.obs[draft.derived_obs]
                .iloc[selected_positions]
                .astype("string")
                .dropna()
                .value_counts()
            )
            if not overlap.empty and str(overlap.index[0]) in categories:
                return str(overlap.index[0])
        source_values = {str(value) for value in context.get("source_values", [])}
        if source_values and not self.population_base_mapping.empty:
            mapped = self.population_base_mapping.loc[
                self.population_base_mapping["source_value"]
                .astype(str)
                .isin(source_values)
            ].copy()
            if not mapped.empty:
                mapped["_cell_count"] = pd.to_numeric(
                    mapped["cell_count"], errors="coerce"
                ).fillna(0)
                mapped = mapped.sort_values("_cell_count", ascending=False)
                proposed = str(mapped.iloc[0]["proposed_label"])
                if proposed in categories:
                    return proposed
        previous = str(context.get("population") or "")
        if previous in categories:
            return previous
        return categories[0] if categories else None

    def _sync_saved_population_draft(
        self,
        draft: PopulationDraft,
        context: dict[str, object],
    ) -> dict[str, object]:
        """Apply a saved draft and redirect Explore/Population QC to its labels."""

        if self.adata is None or self.population_workspace_paths is None:
            raise RuntimeError("A live AnnData and population workspace are required.")
        current_state = population_draft_sync_state(self.adata, draft)
        owned_existing_obs = current_state in {"stale", "synced"}
        if current_state == "conflict" and not self.population_overwrite_check.isChecked():
            raise ValueError(
                f"AnnData already contains unrelated obs[{draft.derived_obs!r}]. "
                "Choose another new label-column name, or enable the advanced "
                "overwrite option deliberately."
            )
        summary = apply_population_draft(
            self.adata,
            draft=draft,
            base_mapping=self.population_base_mapping,
            components=self.population_components,
            membership=self.population_membership,
            overwrite=(
                owned_existing_obs or self.population_overwrite_check.isChecked()
            ),
        )
        preferred_population = self._preferred_synced_population(draft, context)
        source_recipe = context.get("recipe")
        if source_recipe is not None and preferred_population:
            target_key = population_recipe_key(
                draft.derived_obs, preferred_population
            )
            if target_key not in self.explore_review_state.population_recipes:
                self.explore_review_state.population_recipes[target_key] = (
                    self._population_qc_recipe_for_storage(
                        retarget_population_qc_recipe(
                            source_recipe,
                            observation=draft.derived_obs,
                            population=preferred_population,
                        )
                    )
                )
                if self.paths is not None:
                    self._save_explore_review_state()
        append_population_audit(
            self.population_workspace_paths,
            action="sync_labels_into_live_app",
            draft_id=draft.draft_id,
            details={
                "derived_obs": draft.derived_obs,
                "draft_revision": draft.revision,
                "label_count": summary["label_count"],
                "merge_group_count": len(summary["merge_groups"]),
            },
        )
        self._populate_anndata_selectors(
            source=f"saved population labels {draft.derived_obs!r}"
        )
        self.overlay_obs_combo.setCurrentText(draft.derived_obs)
        population_obs_blocked = self.population_obs_combo.blockSignals(True)
        self.population_obs_combo.setCurrentText(draft.derived_obs)
        self.population_obs_combo.blockSignals(population_obs_blocked)
        self.refresh_population_values()
        if preferred_population:
            self.population_value_combo.setCurrentText(preferred_population)
        population_qc_obs_blocked = self.population_qc_obs_combo.blockSignals(True)
        self.population_qc_obs_combo.setCurrentText(draft.derived_obs)
        self.population_qc_obs_combo.blockSignals(population_qc_obs_blocked)
        self.refresh_population_qc_populations()
        if preferred_population:
            self.population_qc_population_combo.setCurrentText(preferred_population)
        self._population_draft_dirty = False
        self.refresh_population_provenance()
        self._refresh_population_naming_readiness()
        return summary

    def _selected_population_table_rows(self, table) -> list[int]:
        return sorted({index.row() for index in table.selectionModel().selectedRows()})

    def name_selected_population_rows(self, table_kind: str) -> None:
        table = (
            self.population_base_table
            if table_kind == "base"
            else self.population_components_table
        )
        column = 2 if table_kind == "base" else 4
        rows = self._selected_population_table_rows(table)
        if not rows:
            raise ValueError("Select one or more complete table rows first.")
        initial = table.item(rows[0], column).text()
        value, accepted = self.QInputDialog.getText(
            self.root,
            "Name population components",
            "Final population name (using one name for several rows proposes a merge):",
            text=initial,
        )
        if not accepted:
            return
        value = value.strip()
        if not value:
            raise ValueError("Population names must not be blank.")
        for row in rows:
            table.item(row, column).setText(value)
        self._refresh_population_merge_preview()

    def colour_selected_population_rows(self, table_kind: str) -> None:
        table = (
            self.population_base_table
            if table_kind == "base"
            else self.population_components_table
        )
        column = 3 if table_kind == "base" else 5
        rows = self._selected_population_table_rows(table)
        if not rows:
            raise ValueError("Select one or more complete table rows first.")
        initial = self.QColor(table.item(rows[0], column).text())
        colour = self.QColorDialog.getColor(initial, self.root)
        if not colour.isValid():
            return
        value = colour.name()
        for row in rows:
            item = table.item(row, column)
            item.setText(value)
            item.setBackground(colour)
        self._refresh_population_merge_preview()

    def import_population_mapping(self) -> None:
        if self.population_draft is None:
            raise RuntimeError("Create or load a population draft first.")
        selected, _ = self.QFileDialog.getOpenFileName(
            self.root,
            "Import preliminary population names",
            str(self.project_root),
            "CSV tables (*.csv);;All files (*)",
        )
        if not selected:
            return
        base, components = self._population_tables_to_frames()
        updated, summary = import_base_mapping_csv(
            selected,
            base,
            source_obs=self.population_draft.source_obs,
            derived_obs=self.population_draft.derived_obs,
        )
        self.population_base_mapping = updated
        self.population_components = components
        self._set_population_tables(updated, components)
        self._save_current_population_draft(
            action="import_preliminary_mapping_csv",
            details=summary,
        )
        self._refresh_population_merge_preview()
        self.refresh_population_provenance()
        self.set_status(
            f"Imported preliminary names for {summary['updated_population_count']} "
            "source populations."
        )

    def export_population_mapping(self) -> None:
        if self.population_draft is None:
            raise RuntimeError("Create or load a population draft first.")
        base, _components = self._population_tables_to_frames()
        selected, _ = self.QFileDialog.getSaveFileName(
            self.root,
            "Export editable population mapping",
            str(self.project_root / f"{self.population_draft.derived_obs}_mapping.csv"),
            "CSV tables (*.csv)",
        )
        if not selected:
            return
        destination = Path(selected).expanduser()
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing mapping export: {destination}"
            )
        write_dataframe(destination, base)
        append_population_audit(
            self.population_workspace_paths,
            action="export_mapping_csv",
            draft_id=self.population_draft.draft_id,
            details={"path": str(destination.resolve(strict=False))},
        )
        self.refresh_population_provenance()
        self.set_status(f"Exported editable mapping CSV to {destination}.")

    def import_population_components(self) -> None:
        if self.population_draft is None or self.adata is None:
            raise RuntimeError("Create or load a population draft first.")
        selected, _ = self.QFileDialog.getOpenFileName(
            self.root,
            "Import image/other cell-level split assignments",
            str(self.project_root),
            "Cell tables (*.csv *.parquet *.pq);;All files (*)",
        )
        if not selected:
            return
        current_base, current_components = self._population_tables_to_frames()
        self.population_base_mapping = current_base
        assignments = read_dataframe(selected)
        new_components, new_membership, import_summary = (
            component_tables_from_assignments(
                self.adata,
                source_obs=self.population_draft.source_obs,
                assignments=assignments,
                method="image_or_external",
            )
        )
        components, membership, merge_summary = integrate_component_tables(
            current_components,
            self.population_membership,
            new_components,
            new_membership,
        )
        self.population_components = components
        self.population_membership = membership
        self._set_population_tables(self.population_base_mapping, components)
        self._save_current_population_draft(
            action="import_cell_level_subclusters",
            details={
                "path": str(Path(selected).resolve(strict=False)),
                **import_summary,
                **merge_summary,
            },
        )
        self._refresh_population_merge_preview()
        self.refresh_population_provenance()
        self.set_status(
            f"Imported {len(new_membership):,} cell assignments as "
            f"{len(new_components)} editable split components."
        )

    def import_current_classifier_components(self) -> None:
        """Bridge finalized/current image-classifier assignments into a draft."""

        if (
            self.population_draft is None
            or self.adata is None
            or self.manifest is None
            or self.cohort.empty
        ):
            raise RuntimeError(
                "Load a classification experiment and a population draft first."
            )
        assignments = build_assignment_table(
            self.cohort,
            self.labels,
            self.scores if not self.scores.empty else None,
            class_ids=[item.class_id for item in self.manifest.classes],
        )
        assignments = assignments.loc[
            assignments["class_id"].notna()
            & assignments["assignment_source"].ne("unassigned")
        ].copy()
        if assignments.empty:
            raise ValueError(
                "The current classifier has no confirmed labels or scored model "
                "assignments to import."
            )
        class_names = {item.class_id: item.name for item in self.manifest.classes}
        assignments["curated_class_name"] = (
            assignments["class_id"].map(class_names).fillna(assignments["class_id"])
        )
        current_base, current_components = self._population_tables_to_frames()
        self.population_base_mapping = current_base
        run_id = (
            str(assignments["model_id"].dropna().iloc[0])
            if "model_id" in assignments and not assignments["model_id"].dropna().empty
            else f"labels-{int(time.time())}"
        )
        new_components, new_membership, import_summary = (
            component_tables_from_assignments(
                self.adata,
                source_obs=self.population_draft.source_obs,
                assignments=assignments,
                method="napari_sbt_image_classifier",
                run_id=run_id,
                label_column="curated_class_name",
            )
        )
        components, membership, integration_summary = integrate_component_tables(
            current_components,
            self.population_membership,
            new_components,
            new_membership,
        )
        self.population_components = components
        self.population_membership = membership
        self._set_population_tables(self.population_base_mapping, components)
        self._save_current_population_draft(
            action="import_current_image_classifier_assignments",
            details={
                **import_summary,
                **integration_summary,
                "confirmed_label_count": int(
                    assignments["assignment_source"].eq("confirmed").sum()
                ),
                "model_assignment_count": int(
                    assignments["assignment_source"].eq("model").sum()
                ),
            },
        )
        self._refresh_population_merge_preview()
        self.refresh_population_provenance()
        self.set_status(
            f"Imported {len(new_membership):,} current classifier assignments "
            f"as {len(new_components)} editable image-derived components."
        )

    def remove_selected_population_components(self) -> None:
        rows = self._selected_population_table_rows(self.population_components_table)
        if not rows:
            raise ValueError("Select one or more split-component rows first.")
        component_ids = {
            str(self.population_components_table.item(row, 0).data(self.Qt.UserRole))
            for row in rows
        }
        current_base, current_components = self._population_tables_to_frames()
        self.population_base_mapping = current_base
        self.population_components = current_components
        removed_cells = int(
            self.population_membership["component_id"].isin(component_ids).sum()
        )
        self.population_components = self.population_components.loc[
            ~self.population_components["component_id"].isin(component_ids)
        ].reset_index(drop=True)
        self.population_membership = self.population_membership.loc[
            ~self.population_membership["component_id"].isin(component_ids)
        ].reset_index(drop=True)
        self._set_population_tables(
            self.population_base_mapping, self.population_components
        )
        self._save_current_population_draft(
            action="remove_split_components",
            details={
                "component_ids": sorted(component_ids),
                "removed_cell_count": removed_cells,
            },
        )
        self._refresh_population_merge_preview()
        self.refresh_population_provenance()

    def _population_worker_anndata_path(self) -> Path:
        if self.population_workspace_paths is None or self.adata is None:
            raise RuntimeError("A population workspace and AnnData are required.")
        path_text = self.anndata_edit.text().strip()
        if path_text and Path(path_text).expanduser().is_file():
            return Path(path_text).expanduser().resolve(strict=False)
        fingerprint = source_obs_fingerprint(
            self.adata, self.population_draft.source_obs
        )
        snapshot = (
            self.population_workspace_paths.inputs / f"source_{fingerprint[:16]}.h5ad"
        )
        if not snapshot.is_file():
            self.set_status(
                "Writing a population-workspace AnnData snapshot so the monitored "
                "Scanpy subprocess can read the live notebook data."
            )
            _write_anndata_snapshot(self.adata, snapshot)
        return snapshot

    def start_population_subclustering(self) -> None:
        if self.population_process is not None:
            raise RuntimeError("Population subclustering is already running.")
        if (
            self.population_draft is None
            or self.population_workspace_paths is None
            or self.adata is None
        ):
            raise RuntimeError("Create or load a population draft first.")
        selected_values = [
            item.text() for item in self.population_split_values_list.selectedItems()
        ]
        if not selected_values:
            raise ValueError("Select at least one source population to subcluster.")
        neighbor_source = self.population_neighbor_source_combo.currentData()
        representation_key = None
        adjacency_key = None
        if neighbor_source == "rebuild_from_rep":
            representation_key = (
                self.population_representation_combo.currentText().strip()
            )
            if representation_key not in self.adata.obsm:
                raise ValueError(
                    "Choose an existing corrected adata.obsm representation."
                )
        else:
            adjacency_key = self.population_adjacency_combo.currentText().strip()
            if adjacency_key not in self.adata.obsp:
                raise ValueError("Choose an existing adata.obsp connectivity graph.")
            if "distance" in adjacency_key.lower():
                raise ValueError(
                    "The selected obsp key appears to be a distance matrix. Leiden "
                    "needs weighted connectivities/adjacency, not distances."
                )
        self._save_current_population_draft(action="save_before_population_subcluster")
        run_id = str(uuid4())
        draft_paths = population_draft_paths(
            self.population_workspace_paths, self.population_draft
        )
        output_folder = draft_paths.runs / run_id
        request = GraphSubclusterRequest(
            run_id=run_id,
            anndata_path=str(self._population_worker_anndata_path()),
            source_obs=self.population_draft.source_obs,
            source_fingerprint=self.population_draft.source_fingerprint,
            selected_values=selected_values,
            neighbor_source=neighbor_source,
            representation_key=representation_key,
            n_neighbors=self.population_n_neighbors_spin.value(),
            adjacency_key=adjacency_key,
            resolution=self.population_resolution_spin.value(),
            mode=self.population_subcluster_mode_combo.currentData(),
            output_folder=str(output_folder),
        )
        output_folder.mkdir(parents=True, exist_ok=False)
        request_path = save_graph_subcluster_request(
            request, output_folder / "request.json"
        )
        append_population_audit(
            self.population_workspace_paths,
            action="start_population_subclustering",
            draft_id=self.population_draft.draft_id,
            details=request.model_dump(mode="json"),
        )
        from qtpy.QtCore import QProcess

        process = QProcess(self.root)
        process.setProgram(sys.executable)
        process.setArguments(
            [
                "-m",
                "SpatialBiologyToolkit.napari_sbt.population_worker",
                "--request",
                str(request_path),
            ]
        )
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(
            self._read_population_subcluster_progress
        )
        process.finished.connect(
            self._guard(
                self._population_subcluster_finished,
                pass_signal_args=True,
            )
        )
        self.population_process = process
        self._population_output_buffer = ""
        self._population_pending_run = {
            "request": request,
            "output_folder": output_folder,
            "cancelled": False,
        }
        self.run_population_subcluster_button.setEnabled(False)
        self.cancel_population_subcluster_button.setEnabled(True)
        self.population_subcluster_status.setText(
            "Starting a separate Python process and validating the corrected "
            "representation/graph…"
        )
        process.start()

    def _read_population_subcluster_progress(self, *, flush: bool = False) -> None:
        if self.population_process is not None:
            self._population_output_buffer += bytes(
                self.population_process.readAllStandardOutput()
            ).decode(errors="replace")
        lines = self._population_output_buffer.splitlines(keepends=True)
        self._population_output_buffer = ""
        for raw_line in lines:
            if not flush and not raw_line.endswith(("\n", "\r")):
                self._population_output_buffer = raw_line
                continue
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self.population_subcluster_status.setText(line)
                continue
            event_name = event.get("event")
            if event_name == "population_subcluster_loading":
                message = (
                    "Loading AnnData and validating its frozen source fingerprint…"
                )
            elif event_name == "population_subcluster_running":
                if event.get("neighbor_source") == "rebuild_from_rep":
                    input_text = (
                        f"rebuilding n={event.get('n_neighbors')} neighbours from "
                        f"{event.get('representation_key')}"
                    )
                else:
                    input_text = f"reusing graph {event.get('adjacency_key')}"
                message = (
                    f"Running {event.get('task_index')}/{event.get('task_count')}: "
                    f"{event.get('population')} ({event.get('cell_count', 0):,} "
                    f"cells, {input_text}, resolution {event.get('resolution')}). "
                    "Process is live."
                )
            elif event_name == "population_subcluster_completed":
                message = (
                    f"Completed: {event.get('cluster_count')} components for "
                    f"{event.get('cell_count', 0):,} cells in "
                    f"{event.get('elapsed_seconds')} s. Importing results…"
                )
            elif event_name == "population_subcluster_failed":
                message = f"FAILED — {event.get('error_type')}: {event.get('error')}"
            else:
                message = line
            self.population_subcluster_status.setText(message)
            self.set_status(f"POPULATION SUBCLUSTER — {message}")

    def cancel_population_subclustering(self) -> None:
        if self.population_process is None:
            return
        if self._population_pending_run is not None:
            self._population_pending_run["cancelled"] = True
        self.population_process.kill()
        self.population_subcluster_status.setText(
            "Cancellation requested. No AnnData or draft mapping has been modified."
        )

    def _population_subcluster_finished(self, exit_code: int, _status) -> None:
        self._read_population_subcluster_progress(flush=True)
        pending = self._population_pending_run or {}
        process_cancelled = bool(pending.get("cancelled"))
        request = pending.get("request")
        output_folder = pending.get("output_folder")
        self.population_process = None
        self._population_pending_run = None
        self.run_population_subcluster_button.setEnabled(True)
        self.cancel_population_subcluster_button.setEnabled(False)
        if process_cancelled:
            self._activity_finish(False, "Population subclustering was cancelled.")
            if self.population_workspace_paths is not None and self.population_draft:
                append_population_audit(
                    self.population_workspace_paths,
                    action="cancel_population_subclustering",
                    draft_id=self.population_draft.draft_id,
                    details={"run_id": getattr(request, "run_id", None)},
                )
            return
        assignments_path = (
            Path(output_folder) / "assignments.csv" if output_folder else None
        )
        if exit_code != 0 or assignments_path is None or not assignments_path.is_file():
            self._activity_finish(
                False, f"Population subclustering exited with code {exit_code}."
            )
            self.population_subcluster_status.setText(
                f"Population subclustering exited with code {exit_code}; no draft "
                "membership was changed. Review the progress message above."
            )
            if self.population_workspace_paths is not None and self.population_draft:
                append_population_audit(
                    self.population_workspace_paths,
                    action="population_subclustering_failed",
                    draft_id=self.population_draft.draft_id,
                    details={
                        "run_id": getattr(request, "run_id", None),
                        "exit_code": exit_code,
                    },
                )
            return
        assignments = read_dataframe(assignments_path)
        component_method = (
            "scanpy_rebuilt_neighbors"
            if request.neighbor_source == "rebuild_from_rep"
            else "scanpy_existing_graph"
        )
        new_components, new_membership, import_summary = (
            component_tables_from_assignments(
                self.adata,
                source_obs=self.population_draft.source_obs,
                assignments=assignments,
                method=component_method,
                run_id=request.run_id,
                label_column="component_value",
            )
        )
        components, membership, integration_summary = integrate_component_tables(
            self.population_components,
            self.population_membership,
            new_components,
            new_membership,
        )
        self.population_components = components
        self.population_membership = membership
        self.population_draft = self.population_draft.model_copy(
            update={"latest_run_id": request.run_id}, deep=True
        )
        self._set_population_tables(self.population_base_mapping, components)
        self._save_current_population_draft(
            action="import_population_subclusters",
            details={
                **import_summary,
                **integration_summary,
                "provenance_path": str(Path(output_folder) / "provenance.json"),
            },
        )
        self._refresh_population_merge_preview()
        self.refresh_population_provenance()
        self.population_subcluster_status.setText(
            f"Imported {len(new_components)} editable components covering "
            f"{len(new_membership):,} cells. Rename them before applying the draft."
        )
        self._activity_finish(
            True,
            f"Imported {len(new_components)} population components covering "
            f"{len(new_membership):,} cells.",
        )

    def apply_current_population_draft(self) -> None:
        """Compatibility action: saving is now the single save-and-sync step."""

        self.save_current_population_draft()

    def show_curated_population_overlay(self) -> None:
        if self.population_draft is None or self.adata is None:
            raise RuntimeError("Create or load a population draft first.")
        observation = self.population_draft.derived_obs
        if observation not in self.adata.obs:
            raise ValueError(
                "Save and update the app before showing this label overlay."
            )
        self.overlay_obs_combo.setCurrentText(observation)
        self.overlay_full_dataset_check.setChecked(True)
        self.render_obs_overlay()
        self.tabs.setCurrentIndex(3)

    def export_curated_anndata(self) -> None:
        if self.adata is None:
            raise RuntimeError("No live AnnData is available to export.")
        default_name = (
            f"{self.population_draft.derived_obs}_curated.h5ad"
            if self.population_draft is not None
            else "napari_sbt_curated.h5ad"
        )
        selected, _ = self.QFileDialog.getSaveFileName(
            self.root,
            "Export curated AnnData copy",
            str(self.project_root / default_name),
            "AnnData (*.h5ad)",
        )
        if not selected:
            return
        destination = atomic_write_curated_anndata(self.adata, selected)
        if self.population_workspace_paths is not None:
            append_population_audit(
                self.population_workspace_paths,
                action="export_curated_anndata_copy",
                draft_id=(
                    self.population_draft.draft_id
                    if self.population_draft is not None
                    else None
                ),
                details={"path": str(destination)},
            )
        self.refresh_population_provenance()
        self.set_status(
            f"Exported a new curated AnnData copy to {destination}; the source "
            "AnnData file was not overwritten."
        )

    def refresh_population_provenance(self) -> None:
        if self.population_workspace_paths is None:
            self.population_provenance_text.setPlainText(
                "No naming history exists for this source observation yet."
            )
            return
        events = read_population_audit(self.population_workspace_paths)
        current_draft_id = (
            self.population_draft.draft_id
            if self.population_draft is not None
            else None
        )
        lines = [
            "Recent naming history",
            "The complete machine-readable audit remains in provenance.jsonl.",
            "",
        ]
        for event in events[-100:]:
            if event.get("draft_id") not in {None, current_draft_id}:
                continue
            details = event.get("details") or {}
            summary_parts = []
            for key, label in (
                ("draft_revision", "revision"),
                ("label_count", "populations"),
                ("merge_group_count", "merges"),
                ("updated_population_count", "renamed rows"),
                ("new_component_count", "new subclusters"),
                ("path", "file"),
            ):
                if key in details:
                    summary_parts.append(f"{label}: {details[key]}")
            action = str(event.get("action") or "updated").replace("_", " ")
            timestamp = str(event.get("timestamp") or "")[:19].replace("T", " ")
            suffix = f" — {', '.join(summary_parts)}" if summary_parts else ""
            lines.append(f"{timestamp}  {action}{suffix}")
        self.population_provenance_text.setPlainText("\n".join(lines))

    def show_population_history(self) -> None:
        """Show a compact human-readable view of the retained technical audit."""

        self.refresh_population_provenance()
        dialog = self.QDialog(self.root)
        dialog.setWindowTitle("NapariSBT population naming history")
        dialog.resize(760, 520)
        from qtpy.QtWidgets import QTextEdit, QVBoxLayout

        layout = QVBoxLayout(dialog)
        text = QTextEdit(dialog)
        text.setReadOnly(True)
        text.setPlainText(self.population_provenance_text.toPlainText())
        buttons = self.QDialogButtonBox(self.QDialogButtonBox.Close, parent=dialog)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(text)
        layout.addWidget(buttons)
        dialog.exec()

    def _population_derived_obs_for_qc(self) -> str:
        if self.population_draft is None or self.adata is None:
            raise RuntimeError("Create or load a population draft first.")
        observation = self.population_draft.derived_obs
        if observation not in self.adata.obs:
            raise ValueError(
                "Save and update the app before generating QC plots. This keeps "
                "the plotted labels identical to the Explore overlay."
            )
        return observation

    def _scanpy_plotting_cohort_names(self) -> set[str] | None:
        if self.cohort.empty or "obs_name" not in self.cohort:
            return None
        return set(self.cohort["obs_name"].astype(str))

    def refresh_scanpy_plotting_choices(
        self,
        preferred_groupby: str | None = None,
    ) -> None:
        """Synchronize the plotting panel with the live AnnData object."""

        if self.adata is None:
            raise ValueError("Load AnnData before configuring Scanpy plots.")
        roi_obs = (
            self.manifest.roi_obs
            if self.manifest is not None
            else self.roi_obs_edit.text().strip()
        )
        if roi_obs not in self.adata.obs:
            roi_obs = None
        self.scanpy_plotting_panel.refresh_from_anndata(
            self.adata,
            roi_obs=roi_obs,
            cohort_obs_names=self._scanpy_plotting_cohort_names(),
            preferred_groupby=preferred_groupby,
        )

    def open_population_scanpy_plotting(self) -> None:
        """Save pending population names and open them in the plotting workspace."""

        if self.population_draft is None or self.adata is None:
            raise RuntimeError("Create or load population naming work first.")
        observation = self.population_draft.derived_obs
        requires_sync = self._population_draft_dirty or observation not in self.adata.obs
        if requires_sync:
            reply = self.QMessageBox.question(
                self.root,
                "Save labels before plotting",
                "The current population names have not been synchronized with "
                "the live AnnData object. Save and update them now, then open "
                "Scanpy plotting?",
            )
            if reply != self.QMessageBox.Yes:
                self.set_status("Opening Scanpy plotting was cancelled.")
                return
            self.save_current_population_draft()
        observation = self._population_derived_obs_for_qc()
        self.refresh_scanpy_plotting_choices(preferred_groupby=observation)
        self.scanpy_plotting_panel.select_groupby(observation)
        self.tabs.setCurrentIndex(self.scanpy_plotting_tab_index)
        self.set_status(
            f"Scanpy plotting is ready with adata.obs[{observation!r}] selected."
        )

    def generate_scanpy_plot(self) -> None:
        """Generate one read-only artifact and show it in a managed popup."""

        if self.adata is None:
            raise ValueError("Load AnnData before generating a plot.")
        request = self.scanpy_plotting_panel.current_request()
        self._activity_update(
            f"Preparing {request.plot_type.replace('_', ' ')} from the selected "
            "AnnData cells…"
        )
        self.QApplication.processEvents()
        artifact = build_scanpy_plot(
            self.adata,
            request,
            cohort_obs_names=self._scanpy_plotting_cohort_names(),
        )
        self._show_scanpy_plot_artifact(artifact, request)
        self.set_status(f"Opened Scanpy plot: {artifact.summary}")

    def _show_scanpy_plot_artifact(self, artifact, request) -> None:
        """Open a modeless, resizable Matplotlib window and track its state."""

        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg,
                NavigationToolbar2QT,
            )
        except ImportError:  # Matplotlib < 3.5 compatibility
            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg,
                NavigationToolbar2QT,
            )
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import (
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSizePolicy,
            QVBoxLayout,
        )

        dialog = self.QDialog(self.root)
        window_id = uuid4().hex
        dialog.setWindowTitle(f"NapariSBT Scanpy plotting — {artifact.title}")
        requested_width = max(
            1040,
            int(artifact.figure.get_figwidth() * artifact.figure.dpi) + 40,
        )
        requested_height = max(
            800,
            int(artifact.figure.get_figheight() * artifact.figure.dpi) + 150,
        )
        try:
            available = self.root.screen().availableGeometry()
            requested_width = min(requested_width, int(available.width() * 0.94))
            requested_height = min(requested_height, int(available.height() * 0.94))
        except (AttributeError, RuntimeError):
            pass
        dialog.resize(requested_width, requested_height)
        layout = QVBoxLayout(dialog)
        canvas = FigureCanvasQTAgg(artifact.figure)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavigationToolbar2QT(canvas, dialog)
        stale_label = QLabel("● Current snapshot")
        stale_label.setStyleSheet("color: #166534; font-weight: 700;")
        summary_label = QLabel(artifact.summary)
        summary_label.setWordWrap(True)
        actions = QHBoxLayout()
        export_data_button = QPushButton("Export plotted data CSV…")
        close_button = QPushButton("Close")
        actions.addWidget(stale_label)
        actions.addStretch(1)
        actions.addWidget(export_data_button)
        actions.addWidget(close_button)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        layout.addWidget(summary_label)
        layout.addLayout(actions)
        canvas.draw()
        responsive_scanpy_layout = request.plot_type in {
            "heatmap",
            "dotplot",
            "violin",
        }
        baseline_margins = (
            figure_subplot_margins(artifact.figure)
            if responsive_scanpy_layout
            else None
        )
        layout_timer = QTimer(dialog)
        layout_timer.setSingleShot(True)
        layout_timer.setInterval(60)

        def refit_scanpy_layout() -> None:
            if not responsive_scanpy_layout or not canvas.isVisible():
                return
            fit_scanpy_figure_to_canvas(
                artifact.figure,
                baseline_margins=baseline_margins,
            )

        layout_timer.timeout.connect(refit_scanpy_layout)

        def schedule_scanpy_layout(*_args) -> None:
            if responsive_scanpy_layout:
                layout_timer.start()

        resize_connection = canvas.mpl_connect(
            "resize_event",
            schedule_scanpy_layout,
        )
        dialog.setAttribute(self.Qt.WA_DeleteOnClose, True)
        self.scanpy_plot_windows[window_id] = {
            "dialog": dialog,
            "artifact": artifact,
            "request": request,
            "stale_label": stale_label,
            "layout_timer": layout_timer,
            "resize_connection": resize_connection,
        }
        self.scanpy_plotting_panel.add_window(
            window_id,
            artifact.title,
            f"{artifact.cell_count:,} cells",
        )
        export_data_button.clicked.connect(
            self._guard(lambda: self.export_scanpy_plot_data(window_id))
        )
        close_button.clicked.connect(dialog.close)

        def forget_dialog(*_args):
            self.scanpy_plot_windows.pop(window_id, None)
            self.scanpy_plotting_panel.remove_window(window_id)

        dialog.destroyed.connect(forget_dialog)
        dialog.show()
        schedule_scanpy_layout()

    def export_scanpy_plot_data(self, window_id: str) -> None:
        record = self.scanpy_plot_windows.get(str(window_id))
        if record is None:
            raise ValueError("The selected plot window is no longer open.")
        artifact = record["artifact"]
        default_folder = self.paths.root / "plots" if self.paths else self.project_root
        selected, _filter = self.QFileDialog.getSaveFileName(
            self.root,
            "Export plotted values",
            str(default_folder / f"{slugify(artifact.title)}.csv"),
            "CSV tables (*.csv)",
        )
        if not selected:
            return
        destination = Path(selected)
        if destination.suffix.casefold() != ".csv":
            destination = destination.with_suffix(".csv")
        write_dataframe(destination, artifact.data)
        self.set_status(
            f"Exported {len(artifact.data):,} plotted-value rows to {destination}."
        )

    def focus_scanpy_plot_window(self, window_id: str) -> None:
        if not window_id:
            self.set_status("Select an open plot first.")
            return
        record = self.scanpy_plot_windows.get(str(window_id))
        if record is None:
            self.scanpy_plotting_panel.remove_window(window_id)
            self.set_status("That plot window is no longer open.")
            return
        dialog = record["dialog"]
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def close_scanpy_plot_window(self, window_id: str) -> None:
        if not window_id:
            self.set_status("Select an open plot first.")
            return
        record = self.scanpy_plot_windows.get(str(window_id))
        if record is None:
            self.scanpy_plotting_panel.remove_window(window_id)
            return
        record["dialog"].close()

    def close_all_scanpy_plot_windows(self) -> None:
        if not self.scanpy_plot_windows:
            self.set_status("There are no Scanpy plot windows to close.")
            return
        count = len(self.scanpy_plot_windows)
        for record in list(self.scanpy_plot_windows.values()):
            record["dialog"].close()
        self.set_status(f"Closed {count:,} Scanpy plot windows.")

    def mark_scanpy_plots_stale(self) -> None:
        """Mark snapshots visibly when the live AnnData labels may have changed."""

        if not hasattr(self, "scanpy_plotting_panel"):
            return
        self.scanpy_plotting_panel.mark_windows_stale()
        for record in self.scanpy_plot_windows.values():
            label = record.get("stale_label")
            if label is not None:
                label.setText("● Out of date — live AnnData may have changed")
                label.setStyleSheet("color: #b45309; font-weight: 700;")

    # Compatibility callbacks retained for callers of the former Population
    # naming Live QC actions. They now route through the dedicated workspace.
    def show_population_embedding_qc(self) -> None:
        observation = self._population_derived_obs_for_qc()
        self.refresh_scanpy_plotting_choices(preferred_groupby=observation)
        self.scanpy_plotting_panel.plot_type_combo.setCurrentIndex(
            self.scanpy_plotting_panel.plot_type_combo.findData("embedding")
        )
        self.generate_scanpy_plot()

    def show_population_heatmap_qc(self) -> None:
        observation = self._population_derived_obs_for_qc()
        self.refresh_scanpy_plotting_choices(preferred_groupby=observation)
        self.scanpy_plotting_panel.plot_type_combo.setCurrentIndex(
            self.scanpy_plotting_panel.plot_type_combo.findData("heatmap")
        )
        self.generate_scanpy_plot()

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
                obs_column=self.obs_combo.currentText()
                if mode == "obs_values"
                else None,
                obs_values=values,
            )
        masks = discover_mask_files(self.masks_edit.text())
        missing_masks: list[str] = []
        missing_ids = 0
        unmatched_ids = 0
        first_roi = str(self.preview.eligible_cells.iloc[0]["ROI"])
        first_mask: np.ndarray | None = None
        for roi, group in self.preview.eligible_cells.groupby("ROI", observed=True):
            path = masks.get(str(roi))
            if path is None:
                missing_masks.append(str(roi))
                continue
            full_mask = load_mask(path)
            if str(roi) == first_roi:
                first_mask = full_mask
            missing, unmatched = validate_mask_coverage(
                full_mask,
                group["ObjectNumber"],
                roi=str(roi),
            )
            missing_ids += len(missing)
            unmatched_ids += len(unmatched)
        eligible_rois = self.preview.eligible_cells["ROI"].astype(str).unique()
        image_index = discover_roi_image_index(
            _split_paths(self.images_edit.toPlainText())
            + _split_paths(self.extra_images_edit.toPlainText()),
            eligible_rois,
            channel_aliases=self._channel_aliases(),
        )
        missing_image_rois = [
            roi for roi in eligible_rois if not image_index.get(str(roi))
        ]
        indexed_images = sum(len(paths) for paths in image_index.values())
        self._mask_path_index = dict(masks)
        self._roi_image_path_index = image_index
        self._asset_index_signature = self._current_asset_index_signature()
        self._integrity_signature = self._current_integrity_signature()
        if self.paths is not None:
            self._write_integrity_index(self.paths.root)
        text = (
            f"{self.preview.eligible_cell_count:,} eligible cells "
            f"({self.preview.eligible_fraction:.1%}) / "
            f"{self.preview.total_cell_count:,} total\n"
            f"{self.preview.represented_roi_count:,} represented ROIs\n"
            f"Missing masks: {len(missing_masks)}; missing eligible object IDs: "
            f"{missing_ids}; other full-mask labels: {unmatched_ids}\n"
            f"Indexed images: {indexed_images}; ROIs without images: "
            f"{len(missing_image_rois)}\n\n"
            + self.preview.per_roi_counts.to_string(index=False)
        )
        self.preview_text.setPlainText(text)
        previous_trial_rois = self.selected_trial_rois()
        self._populate_trial_roi_list(
            self.preview.per_roi_counts,
            selected_rois=previous_trial_rois,
        )
        if self.experiment_mode_combo.currentData() == "feature_discovery_trial" and (
            self.trial_roi_strategy_combo.currentData() == "largest"
            or not previous_trial_rois
        ):
            self.suggest_trial_rois()
        if first_roi in masks and first_mask is not None:
            restricted = cohort_mask(
                first_mask,
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
        self.integrity_status_label.setText(
            f"Validated and indexed {len(masks):,} masks and {indexed_images:,} "
            f"images across {len(eligible_rois):,} eligible ROIs. Normal ROI "
            "navigation will reuse this index without rescanning folders."
        )
        self.set_status(
            "Dataset integrity validated and the fast ROI asset index was built."
        )
        self.refresh_setup_readiness()
        return self.preview

    def _class_colour_item(self, colour_text: str):
        colour = self.QColor(str(colour_text))
        if not colour.isValid():
            colour = self.QColor("#808080")
        item = self.QTableWidgetItem(colour.name())
        item.setBackground(colour)
        foreground = "#111827" if colour.lightness() > 150 else "#ffffff"
        item.setForeground(self.QColor(foreground))
        item.setToolTip(
            f"Class colour {colour.name()}. Double-click to open the colour picker."
        )
        item.setFlags(item.flags() & ~self.Qt.ItemIsEditable)
        return item

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
                item = (
                    self._class_colour_item(value)
                    if column == 2
                    else self.QTableWidgetItem(value)
                )
                self.class_table.setItem(row, column, item)

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
            item = (
                self._class_colour_item(value)
                if column == 2
                else self.QTableWidgetItem(value)
            )
            self.class_table.setItem(row, column, item)
        self.class_table.selectRow(row)

    def pick_class_colour_from_cell(self, row: int, column: int) -> None:
        if int(column) != 2:
            return
        self.class_table.selectRow(int(row))
        self.pick_selected_class_colour()

    def pick_selected_class_colour(self) -> None:
        row = self.class_table.currentRow()
        if row < 0:
            raise ValueError("Select a class row before choosing its colour.")
        current_item = self.class_table.item(row, 2)
        initial = self.QColor(current_item.text() if current_item else "#808080")
        colour = self.QColorDialog.getColor(
            initial,
            self.root,
            "Choose classification class colour",
        )
        if not colour.isValid():
            return
        self.class_table.setItem(row, 2, self._class_colour_item(colour.name()))
        self.set_status(
            f"Selected {colour.name()} for class row {row + 1}. Apply class edits "
            "or create the experiment to save it."
        )

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
        self._mark_final_identities_stale()
        self.refresh_class_controls()
        self.refresh_classification_layers()
        self.set_status(
            "Applied class edits. Stable semantics remain locked when confirmed "
            "labels exist; cosmetic edits were audited."
        )

    def _set_labeler_class_rows(self, definitions: Iterable[LabelerClass]) -> None:
        """Populate the editable Labeler definition table."""

        self.labeler_class_table.setRowCount(0)
        for definition in definitions:
            row = self.labeler_class_table.rowCount()
            self.labeler_class_table.insertRow(row)
            stable_id = self.QTableWidgetItem(definition.label_id)
            stable_id.setFlags(stable_id.flags() & ~self.Qt.ItemIsEditable)
            self.labeler_class_table.setItem(row, 0, stable_id)
            self.labeler_class_table.setItem(
                row, 1, self.QTableWidgetItem(definition.name)
            )
            self.labeler_class_table.setItem(
                row, 2, self._class_colour_item(definition.color)
            )

    def add_labeler_class_row(self) -> None:
        """Add a label definition with an immutable generated ID."""

        if self.labeler_class_table.rowCount() >= 24:
            self.set_status("Labeler is limited to 24 labels in one session.")
            return
        row = self.labeler_class_table.rowCount()
        palette = (
            "#e11d48",
            "#2563eb",
            "#16a34a",
            "#9333ea",
            "#ea580c",
            "#0891b2",
            "#ca8a04",
            "#db2777",
        )
        stable_id = self.QTableWidgetItem(f"label_{uuid4().hex[:8]}")
        stable_id.setFlags(stable_id.flags() & ~self.Qt.ItemIsEditable)
        self.labeler_class_table.insertRow(row)
        self.labeler_class_table.setItem(row, 0, stable_id)
        self.labeler_class_table.setItem(
            row, 1, self.QTableWidgetItem(f"Label {row + 1}")
        )
        self.labeler_class_table.setItem(
            row, 2, self._class_colour_item(palette[row % len(palette)])
        )
        self.labeler_class_table.selectRow(row)

    def remove_labeler_class_row(self) -> None:
        """Remove an unused Labeler class from the working definition table."""

        row = self.labeler_class_table.currentRow()
        if row < 0:
            raise ValueError("Select a Labeler row to remove.")
        item = self.labeler_class_table.item(row, 0)
        label_id = item.text() if item is not None else ""
        if (
            not self.labeler_records.empty
            and self.labeler_records["label_id"].astype(str).eq(label_id).any()
        ):
            raise ValueError(
                "This label is already assigned to cells. Clear or reassign those "
                "cells before removing its definition."
            )
        self.labeler_class_table.removeRow(row)
        self.apply_labeler_class_edits()

    def pick_labeler_colour_from_cell(self, row: int, column: int) -> None:
        if int(column) != 2:
            return
        self.labeler_class_table.selectRow(int(row))
        self.pick_selected_labeler_colour()

    def pick_selected_labeler_colour(self) -> None:
        row = self.labeler_class_table.currentRow()
        if row < 0:
            raise ValueError("Select a Labeler row before choosing its colour.")
        current_item = self.labeler_class_table.item(row, 2)
        initial = self.QColor(current_item.text() if current_item else "#808080")
        colour = self.QColorDialog.getColor(
            initial,
            self.root,
            "Choose Labeler class colour",
        )
        if not colour.isValid():
            return
        self.labeler_class_table.setItem(row, 2, self._class_colour_item(colour.name()))

    def labeler_class_definitions(self) -> list[LabelerClass]:
        definitions = []
        for row in range(self.labeler_class_table.rowCount()):
            values = [
                self.labeler_class_table.item(row, column).text()
                if self.labeler_class_table.item(row, column)
                else ""
                for column in range(3)
            ]
            definitions.append(
                LabelerClass(
                    label_id=values[0],
                    name=values[1],
                    color=values[2],
                )
            )
        return validate_labeler_classes(definitions)

    def apply_labeler_class_edits(self) -> None:
        """Apply working names and colours without changing cell identities."""

        definitions = self.labeler_class_definitions()
        assigned_ids = set(self.labeler_records["label_id"].astype(str))
        missing = sorted(
            assigned_ids - {definition.label_id for definition in definitions}
        )
        if missing:
            raise ValueError(
                "Label definitions cannot remove IDs that are assigned to cells: "
                f"{missing}"
            )
        self.labeler_classes = definitions
        self._refresh_labeler_controls()
        self.refresh_labeler_layers()
        self.set_status(
            f"Applied {len(definitions)} Labeler definitions. Existing assignments "
            "retain their stable IDs."
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
                discovered = self._image_paths_for_roi(rois[0])
                discovered_channels.extend(discovered)
        channels = list(dict.fromkeys(discovered_channels or panel_channels))
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
        return [item.text() for item in self.feature_channel_list.selectedItems()]

    def _update_feature_channel_summary(self) -> None:
        channels = self.selected_feature_channels()
        self.channels_edit.setText(", ".join(channels) if channels else "")
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
            str(channel_count) if channel_count else "all consistently discovered"
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
            distribution_feature_names=self.selected_feature_names("distribution"),
            region_feature_names=self.selected_feature_names("region"),
            gradient_feature_names=self.selected_feature_names("gradient"),
            shape_feature_names=self.selected_feature_names("shape"),
            context_feature_names=self.selected_feature_names("context"),
            roi_rank_statistics=self.selected_feature_names("roi_rank"),
            background_ring_px=self.background_ring_spin.value(),
            normalization_dict_path=(self.normalization_edit.text().strip() or None),
        )

    def create_experiment(self) -> None:
        workflow_mode = self.current_workflow_mode()
        if workflow_mode is None:
            raise ValueError("Choose a Setup workflow before creating its workspace.")
        if (
            self.preview is None
            or self._integrity_signature != self._current_integrity_signature()
        ):
            raise ValueError(
                "Run Setup → Check dataset integrity and build the fast image index after "
                "choosing the dataset and cohort, then create the workspace. This "
                "keeps costly folder and mask checks out of normal navigation."
            )
        preview = self.preview
        classification_workflow = workflow_mode in {
            "classification",
            "full_workspace",
        }
        experiment_mode = (
            str(self.experiment_mode_combo.currentData())
            if classification_workflow
            else "full"
        )
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
        snapshot_text = (
            " The live AnnData will be copied into the experiment inputs so the "
            "experiment can be reopened and used by separate feature workers."
            if self._in_memory_adata is not None
            else ""
        )
        reply = self.QMessageBox.question(
            self.root,
            "Create workspace and start",
            (
                f"Create a {self.workflow_combo.currentText()!r} workspace with "
                f"{preview.eligible_cell_count:,} eligible identities across "
                f"{preview.represented_roi_count} ROIs? The identity snapshot and "
                f"display recipes will be stored in its experiment folder. Later "
                f"cohort changes require an explicit revision.{trial_text}"
                f"{snapshot_text}"
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
        normalization_path = self._write_experiment_normalization(root)
        anndata_source = self.anndata_edit.text().strip()
        if self._in_memory_adata is not None:
            snapshot = _write_anndata_snapshot(
                self._in_memory_adata,
                root / "inputs" / "anndata.h5ad",
            )
            anndata_source = str(snapshot)
            self.anndata_edit.setText(anndata_source)
            self._in_memory_adata = None
            self.set_status(
                "Saved and activated an experiment-owned snapshot of the live "
                f"AnnData: {snapshot}"
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
        synthetic_recipe = self.synthetic_recipe_from_controls()
        synthetic_recipe.normalization_dict_path = normalization_path
        manifest = ExperimentManifest(
            name=name,
            workflow_mode=workflow_mode,
            project_root=str(self.project_root),
            anndata_path=anndata_source,
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
            synthetic_features=synthetic_recipe,
            display_settings=self._display_settings_from_controls(
                normalization_path=normalization_path
            ),
            annotated_adata_path=self.annotated_path_edit.text().strip(),
        )
        self.paths = save_experiment(manifest, root, audit_action="create_experiment")
        self._write_integrity_index(root)
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
        self._loaded_workspace_root = self.paths.root
        self._launch_experiment = self.paths.root
        self.name_edit.setReadOnly(True)
        self.experiment_edit.setReadOnly(True)
        self.choose_experiment_folder_button.setEnabled(False)
        self._set_dataset_source_editable(False)
        if self.manifest.project_root:
            manifest_project = Path(self.manifest.project_root).expanduser().resolve(
                strict=False
            )
            if manifest_project.is_dir():
                self.project_root = manifest_project
                self.project_edit.setText(str(manifest_project))
                configured_folder = "napari_sbt"
                try:
                    from SpatialBiologyToolkit.pipeline.project import load_project

                    configured_folder = load_project(
                        manifest_project
                    ).config.napari_sbt.experiment_folder
                except Exception:
                    pass
                self._workspace_container = workspace_folder(
                    manifest_project, configured_folder
                )
        workflow_index = self.workflow_combo.findData(self.manifest.workflow_mode)
        self.workflow_combo.blockSignals(True)
        self.workflow_combo.setCurrentIndex(max(0, workflow_index))
        self.workflow_combo.blockSignals(False)
        self._update_workflow_mode()
        if self._labeler_experiment_id != self.manifest.experiment_id:
            self.labeler_classes = default_labeler_classes()
            self.labeler_records = empty_labeler_records()
            self._labeler_experiment_id = self.manifest.experiment_id
            self._set_labeler_class_rows(self.labeler_classes)
        self.model_bundle = None
        self.final_assignments = pd.DataFrame()
        self.final_identity_signature = None
        self.final_identity_decision = {}
        self.final_identity_summary_label.setText(
            "Not created in this session. Set the rules, then create final "
            "identities before export."
        )
        self.experiment_edit.setText(str(self.paths.root))
        self.assignment_path_edit.setText(
            str(self.paths.exports / "final_identities.csv")
        )
        self.annotated_path_edit.setText(
            self.manifest.annotated_adata_path
            or str(self.paths.exports / "annotated.h5ad")
        )
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
            self.trial_roi_count_spin.setValue(self.manifest.feature_trial.roi_count)
            self.trial_roi_strategy_combo.setCurrentIndex(
                self.trial_roi_strategy_combo.findData(
                    self.manifest.feature_trial.roi_selection
                )
            )
        self.images_edit.setPlainText("\n".join(self.manifest.images_folders))
        self.extra_images_edit.setPlainText(
            "\n".join(self.manifest.extra_images_folders)
        )
        self.offset_spin.setValue(self.manifest.synthetic_features.mask_offset_px)
        self.offset_overlap_check.setChecked(
            self.manifest.synthetic_features.allow_positive_offset_overlap
        )
        self.background_ring_spin.setValue(
            self.manifest.synthetic_features.background_ring_px
        )
        display_settings = self.manifest.display_settings
        for widget, value in (
            (self.display_quantile_spin, display_settings.fallback_quantile),
            (
                self.display_minimum_pixel_spin,
                display_settings.minimum_pixel_counts,
            ),
            (
                self.display_lower_contrast_spin,
                display_settings.default_contrast_limits[0],
            ),
            (
                self.display_upper_contrast_spin,
                display_settings.default_contrast_limits[1],
            ),
        ):
            widget.blockSignals(True)
            widget.setValue(float(value))
            widget.blockSignals(False)
        self._sync_population_qc_contrast_defaults()
        normalization_path = (
            display_settings.normalization_dict_path
            or self.manifest.synthetic_features.normalization_dict_path
        )
        resolved_normalization_path = ""
        if normalization_path:
            candidate = Path(normalization_path).expanduser()
            if not candidate.is_absolute():
                experiment_candidate = self.paths.root / candidate
                manifest_project_root = (
                    Path(self.manifest.project_root).expanduser()
                    if self.manifest.project_root
                    else self.project_root
                )
                project_candidate = manifest_project_root / candidate
                candidate = (
                    experiment_candidate
                    if experiment_candidate.is_file()
                    else project_candidate
                )
            resolved_normalization_path = str(candidate.resolve(strict=False))
        self.normalization_edit.setText(resolved_normalization_path)
        if resolved_normalization_path and Path(resolved_normalization_path).is_file():
            self.load_normalization_json()
        else:
            self.display_normalization = {}
            self._set_normalization_table({})
            self.normalization_status_label.setText(
                "No usable fixed normalization mapping is stored; channels use "
                "the configured fallback quantile."
            )
            self._refresh_feature_normalization_summary()
        self.distribution_check.setChecked(
            self.manifest.synthetic_features.distribution_features
        )
        self.region_check.setChecked(self.manifest.synthetic_features.region_features)
        self.gradient_check.setChecked(
            self.manifest.synthetic_features.gradient_features
        )
        self.shape_check.setChecked(self.manifest.synthetic_features.shape_features)
        self.context_check.setChecked(self.manifest.synthetic_features.context_features)
        self.roi_rank_check.setChecked(
            self.manifest.synthetic_features.roi_rank_features
        )
        selected_by_family = {
            "distribution": set(
                self.manifest.synthetic_features.distribution_feature_names
            ),
            "region": set(self.manifest.synthetic_features.region_feature_names),
            "gradient": set(self.manifest.synthetic_features.gradient_feature_names),
            "shape": set(self.manifest.synthetic_features.shape_feature_names),
            "context": set(self.manifest.synthetic_features.context_feature_names),
            "roi_rank": set(self.manifest.synthetic_features.roi_rank_statistics),
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
        self._invalidate_population_qc_caches()
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
        if self._in_memory_adata is not None:
            if "obs_name" in self.cohort:
                frozen_names = set(self.cohort["obs_name"].astype(str))
                live_names = set(self._in_memory_adata.obs_names.astype(str))
                missing_names = sorted(frozen_names - live_names)
                if missing_names:
                    raise ValueError(
                        "The supplied in-memory AnnData is missing frozen experiment "
                        f"cells. Examples: {missing_names[:10]}"
                    )
            self.adata = self._in_memory_adata
            self._populate_anndata_selectors(source="the live in-memory object")
        elif self.manifest.anndata_path:
            self.load_anndata_selectors()
        if self.adata is not None:
            self.scope_combo.setCurrentIndex(
                self.scope_combo.findData(self.manifest.cell_scope.mode)
            )
            if self.manifest.cell_scope.mode == "obs_values":
                self.obs_combo.setCurrentText(self.manifest.cell_scope.obs_column or "")
                self.refresh_scope_values()
                selected_values = set(self.manifest.cell_scope.obs_values)
                for index in range(self.value_list.count()):
                    item = self.value_list.item(index)
                    item.setSelected(item.text() in selected_values)
        if not self._load_integrity_index():
            self.integrity_status_label.setText(
                "No matching saved asset index is available. Nested ROI folders "
                "and directly named masks will use fast lazy lookups; run Check "
                "dataset integrity to index flat image folders and check complete "
                "coverage."
            )
        self.refresh_feature_channel_choices()
        requested_channels = set(self.manifest.synthetic_features.channels)
        for index in range(self.feature_channel_list.count()):
            item = self.feature_channel_list.item(index)
            item.setSelected(item.text() in requested_channels)
        self._update_feature_channel_summary()
        self._update_feature_selection_summary()
        self.refresh_class_controls()
        self._refresh_labeler_controls()
        if self.explore_review_state.active_recipe_id is not None:
            self._apply_explore_recipe(self.explore_recipe, replay=False)
        self.refresh_rois()
        self._set_classification_enabled(True)
        self._update_scope_text()
        if self.roi_combo.count():
            self.load_roi(self.roi_combo.currentText())
        self.set_status(
            f"Loaded experiment {self.manifest.name!r}, revision "
            f"{self.manifest.revision}."
        )
        self._refresh_registered_project_choices()
        self.refresh_workspace_choices()
        self.refresh_setup_readiness()
        self.refresh_status()
        self.load_refinement_results(silent=True)

    def _update_scope_text(self) -> None:
        if self.manifest is None:
            self.scope_label.setText(
                "No workflow workspace: choose a task and dataset in Setup."
            )
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
                "No active experiment. Models are stored inside the experiment folder."
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

    def _refresh_labeler_controls(self) -> None:
        """Refresh Labeler class selectors, tallies, and visible assignments."""

        if not hasattr(self, "labeler_class_combo"):
            return
        current_id = self.labeler_class_combo.currentData()
        self.labeler_class_combo.blockSignals(True)
        self.labeler_class_combo.clear()
        for definition in self.labeler_classes:
            self.labeler_class_combo.addItem(
                self._class_icon(definition.color),
                definition.name,
                definition.label_id,
            )
        if current_id is not None:
            index = self.labeler_class_combo.findData(current_id)
            if index >= 0:
                self.labeler_class_combo.setCurrentIndex(index)
        self.labeler_class_combo.blockSignals(False)
        self._refresh_labeler_tally()
        self._refresh_labeler_results_table()

    def selected_labeler_class_id(self) -> str:
        value = self.labeler_class_combo.currentData()
        if value is None:
            raise ValueError("Define and select a Labeler class first.")
        return str(value)

    def _working_labeler_rois(self) -> list[str]:
        if self.manifest is None or self.cohort.empty:
            return []
        if (
            self.manifest.experiment_mode == "feature_discovery_trial"
            and self.manifest.feature_trial is not None
        ):
            return list(self.manifest.feature_trial.selected_rois)
        return sorted(self.cohort["ROI"].astype(str).unique())

    def _refresh_labeler_tally(self, *_args) -> None:
        if not hasattr(self, "labeler_tally_table"):
            return
        rois = self._working_labeler_rois()
        summary = labeler_summary(
            self.labeler_records,
            self.labeler_classes,
            eligible_rois=rois,
        )
        self.labeler_tally_table.setRowCount(len(summary))
        current_roi = str(self.current_roi or "")
        for row, result in summary.iterrows():
            current_count = int(
                (
                    self.labeler_records["label_id"]
                    .astype(str)
                    .eq(str(result["label_id"]))
                    & self.labeler_records["ROI"].astype(str).eq(current_roi)
                ).sum()
            )
            values = (
                str(result["label"]),
                f"{int(result['cells']):,}",
                f"{int(result['rois_sampled']):,}/{int(result['eligible_rois']):,}",
                f"{current_count:,}",
            )
            for column, value in enumerate(values):
                item = self.QTableWidgetItem(value)
                if column == 0:
                    item.setIcon(self._class_icon(str(result["color"])))
                self.labeler_tally_table.setItem(row, column, item)
        total = int(len(self.labeler_records))
        sampled_rois = int(self.labeler_records["ROI"].astype(str).nunique())
        if not rois:
            self.labeler_sampling_summary_label.setText(
                "Create or load an experiment to begin ROI sampling."
            )
            return
        label_id = self.labeler_class_combo.currentData()
        if label_id is None:
            selected_text = "No label is selected."
        else:
            selected = summary.loc[summary["label_id"].astype(str).eq(str(label_id))]
            if selected.empty:
                selected_text = "No label is selected."
            else:
                result = selected.iloc[0]
                remaining = max(
                    0,
                    int(result["eligible_rois"]) - int(result["rois_sampled"]),
                )
                selected_text = (
                    f"{result['label']}: {int(result['cells']):,} cells from "
                    f"{int(result['rois_sampled']):,} ROIs; {remaining:,} eligible "
                    "ROIs have not yet contributed this label."
                )
        self.labeler_sampling_summary_label.setText(
            f"{total:,} labelled cells across {sampled_rois:,}/{len(rois):,} "
            f"eligible ROIs. {selected_text}"
        )

    def _refresh_labeler_results_table(self) -> None:
        if not hasattr(self, "labeler_results_table"):
            return
        if self.cohort.empty:
            table = pd.DataFrame()
        else:
            table = build_labeler_export_table(
                self.labeler_records,
                self.labeler_classes,
                cohort=self.cohort,
            )
        self.labeler_results_table.setRowCount(len(table))
        for row, result in table.iterrows():
            values = (
                result.get("obs_name", ""),
                result.get("ROI", ""),
                result.get("ObjectNumber", ""),
                result.get("label", ""),
                result.get("label_id", ""),
                result.get("timestamp", ""),
            )
            for column, value in enumerate(values):
                item = self.QTableWidgetItem(str(value))
                if column == 3:
                    definition = next(
                        (
                            candidate
                            for candidate in self.labeler_classes
                            if candidate.label_id == str(result.get("label_id", ""))
                        ),
                        None,
                    )
                    if definition is not None:
                        item.setIcon(self._class_icon(definition.color))
                self.labeler_results_table.setItem(row, column, item)

    def _refresh_single_labeler_result_row(self, roi: str, object_number: int) -> None:
        """Incrementally update the visible list after one viewer click."""

        matching_row = -1
        for row in range(self.labeler_results_table.rowCount()):
            roi_item = self.labeler_results_table.item(row, 1)
            object_item = self.labeler_results_table.item(row, 2)
            if (
                roi_item is not None
                and object_item is not None
                and roi_item.text() == str(roi)
                and object_item.text() == str(int(object_number))
            ):
                matching_row = row
                break
        assignment = self.labeler_records.loc[
            self.labeler_records["ROI"].astype(str).eq(str(roi))
            & pd.to_numeric(self.labeler_records["ObjectNumber"], errors="coerce").eq(
                int(object_number)
            )
        ]
        if assignment.empty:
            if matching_row >= 0:
                self.labeler_results_table.removeRow(matching_row)
            return
        result = assignment.iloc[-1]
        definition = next(
            item
            for item in self.labeler_classes
            if item.label_id == str(result["label_id"])
        )
        cohort_row = self.cohort.loc[
            self.cohort["ROI"].astype(str).eq(str(roi))
            & pd.to_numeric(self.cohort["ObjectNumber"], errors="coerce").eq(
                int(object_number)
            )
        ]
        obs_name = (
            str(cohort_row.iloc[0]["obs_name"])
            if not cohort_row.empty and "obs_name" in cohort_row
            else ""
        )
        if matching_row < 0:
            self.labeler_results_table.insertRow(0)
            matching_row = 0
        values = (
            obs_name,
            str(roi),
            str(int(object_number)),
            definition.name,
            definition.label_id,
            str(result["timestamp"]),
        )
        for column, value in enumerate(values):
            item = self.QTableWidgetItem(value)
            if column == 3:
                item.setIcon(self._class_icon(definition.color))
            self.labeler_results_table.setItem(matching_row, column, item)

    def _refresh_labeler_roi_choices(self, rois: Iterable[str]) -> None:
        if not hasattr(self, "labeler_roi_combo"):
            return
        values = [str(roi) for roi in rois]
        current = str(self.current_roi or self.labeler_roi_combo.currentText())
        self.labeler_roi_combo.blockSignals(True)
        self.labeler_roi_combo.clear()
        self.labeler_roi_combo.addItems(values)
        if current in values:
            self.labeler_roi_combo.setCurrentText(current)
        self.labeler_roi_combo.blockSignals(False)
        self._refresh_labeler_tally()

    def load_labeler_roi(self, roi: str) -> None:
        """Navigate the shared viewer from Labeler's ROI selector."""

        roi = str(roi).strip()
        if not roi:
            return
        index = self.roi_combo.findText(roi)
        if index < 0:
            raise ValueError(f"ROI {roi!r} is outside the active experiment scope.")
        if self.roi_combo.currentIndex() != index:
            self.roi_combo.setCurrentIndex(index)
        elif self.current_roi != roi:
            self.load_roi(roi)

    def move_to_next_unsampled_labeler_roi(self) -> None:
        """Move to the next ROI with no assignments for the selected label."""

        label_id = self.selected_labeler_class_id()
        rois = self._working_labeler_rois()
        if not rois:
            raise ValueError("The active experiment has no eligible ROIs.")
        sampled = set(
            self.labeler_records.loc[
                self.labeler_records["label_id"].astype(str).eq(label_id), "ROI"
            ].astype(str)
        )
        start = (
            rois.index(str(self.current_roi)) if str(self.current_roi) in rois else -1
        )
        for offset in range(1, len(rois) + 1):
            candidate = rois[(start + offset) % len(rois)]
            if candidate not in sampled:
                self.roi_combo.setCurrentText(candidate)
                self.set_status(
                    f"Moved to unsampled ROI {candidate!r} for the selected "
                    "Labeler class."
                )
                return
        definition = next(
            item for item in self.labeler_classes if item.label_id == label_id
        )
        self.set_status(
            f"Every eligible ROI has at least one {definition.name!r} assignment."
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
            if self._mask_path_index:
                rois = sorted(set(self._mask_path_index) | set(eligible_rois))
            else:
                self.set_status(
                    "Run Setup → Validate integrity before including cohort-empty "
                    "ROIs; normal navigation will not scan the complete mask folder."
                )
        current = self.roi_combo.currentText()
        self.roi_combo.blockSignals(True)
        self.roi_combo.clear()
        self.roi_combo.addItems(rois)
        if current in rois:
            self.roi_combo.setCurrentText(current)
        self.roi_combo.blockSignals(False)
        self._refresh_labeler_roi_choices(rois)
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

    def _sync_population_qc_contour_control(self) -> None:
        if not hasattr(self, "population_qc_contour_spin"):
            return
        blocked = self.population_qc_contour_spin.blockSignals(True)
        self.population_qc_contour_spin.setValue(
            int(self.explore_review_state.population_qc_contour_width)
        )
        self.population_qc_contour_spin.blockSignals(blocked)

    def _load_explore_review_state(self) -> None:
        self.explore_recipe = ExploreViewRecipe()
        self.explore_review_state = ExploreReviewState()
        path = self._explore_state_path()
        if path is None or not path.exists():
            self._sync_population_qc_contour_control()
            self._bind_explore_recipe_shortcuts()
            self._refresh_recipe_preset_controls()
            self._refresh_reload_recipe_list()
            return
        try:
            self.explore_review_state = ExploreReviewState.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
            self.explore_review_state.schema_version = EXPLORE_STATE_VERSION
            self.explore_review_state.population_recipes = {
                key: self._population_qc_recipe_for_storage(recipe)
                for key, recipe in self.explore_review_state.population_recipes.items()
            }
            active_id = self.explore_review_state.active_recipe_id
            if active_id is not None:
                self.explore_recipe = self.explore_review_state.recipe_presets[
                    active_id
                ].recipe.model_copy(deep=True)
        except Exception as exc:  # noqa: BLE001 - preserve usable experiment
            self.set_status(
                f"Could not read Explore review state from {path.name}: {exc}"
            )
        self._sync_population_qc_contour_control()
        self._bind_explore_recipe_shortcuts()
        self._refresh_recipe_preset_controls()
        self._refresh_reload_recipe_list()

    def _save_explore_review_state(self) -> None:
        path = self._explore_state_path()
        if path is not None:
            write_json(
                path,
                self.explore_review_state.model_dump(mode="json"),
            )

    def _refresh_recipe_preset_controls(
        self, *, selected_id: str | None = None
    ) -> None:
        if not hasattr(self, "recipe_preset_combo"):
            return
        previous_id = selected_id or self.recipe_preset_combo.currentData()
        if previous_id not in self.explore_review_state.recipe_presets:
            previous_id = self.explore_review_state.active_recipe_id
        self.recipe_preset_combo.blockSignals(True)
        self.recipe_preset_combo.clear()
        self.recipe_preset_combo.addItem("Choose a saved recipe…", None)
        presets = sorted(
            self.explore_review_state.recipe_presets.values(),
            key=lambda preset: preset.name.casefold(),
        )
        for preset in presets:
            shortcut = f"{preset.shortcut} · " if preset.shortcut else ""
            self.recipe_preset_combo.addItem(
                f"{shortcut}{preset.name}", preset.preset_id
            )
        index = self.recipe_preset_combo.findData(previous_id)
        self.recipe_preset_combo.setCurrentIndex(max(0, index))
        self.recipe_preset_combo.blockSignals(False)
        self._recipe_preset_selection_changed()
        self._refresh_active_recipe_preset_label()

    def _select_recipe_preset_control(self, preset_id: str) -> None:
        """Select an existing preset without rebuilding the whole selector."""

        if not hasattr(self, "recipe_preset_combo"):
            return
        index = self.recipe_preset_combo.findData(str(preset_id))
        if index < 0:
            self._refresh_recipe_preset_controls(selected_id=str(preset_id))
            return
        self.recipe_preset_combo.blockSignals(True)
        if self.recipe_preset_combo.currentIndex() != index:
            self.recipe_preset_combo.setCurrentIndex(index)
        self.recipe_preset_combo.blockSignals(False)
        self._recipe_preset_selection_changed()

    def _refresh_active_recipe_preset_label(self) -> None:
        if not hasattr(self, "active_recipe_preset_label"):
            return
        active_id = self.explore_review_state.active_recipe_id
        preset = self.explore_review_state.recipe_presets.get(active_id)
        if preset is None:
            self.active_recipe_preset_label.setText(
                "Working view is not linked to a named recipe. Save it as a new "
                "recipe to make it reusable."
            )
            return
        modified = preset.recipe.fingerprint != self.explore_recipe.fingerprint
        shortcut = f" ({preset.shortcut})" if preset.shortcut else ""
        if modified:
            state = (
                "MODIFIED — use ‘Update selected recipe from current view’ to "
                "save these changes"
            )
        else:
            state = "saved and current"
        self.active_recipe_preset_label.setText(
            f"Active recipe: {preset.name}{shortcut} — {state}."
        )

    def _recipe_preset_selection_changed(self, *_args) -> None:
        if not hasattr(self, "recipe_preset_combo"):
            return
        preset_id = self.recipe_preset_combo.currentData()
        preset = self.explore_review_state.recipe_presets.get(preset_id)
        enabled = preset is not None
        self.load_recipe_preset_button.setEnabled(enabled)
        self.update_recipe_preset_button.setEnabled(enabled)
        self.delete_recipe_preset_button.setEnabled(enabled)
        self.export_recipe_preset_button.setEnabled(enabled)
        if preset is None:
            self.recipe_preset_name_edit.clear()
            self.recipe_preset_shortcut_combo.setCurrentIndex(0)
            return
        self.recipe_preset_name_edit.setText(preset.name)
        shortcut_index = self.recipe_preset_shortcut_combo.findData(preset.shortcut)
        self.recipe_preset_shortcut_combo.setCurrentIndex(max(0, shortcut_index))

    def _selected_recipe_preset(self) -> ExploreRecipePreset:
        preset_id = self.recipe_preset_combo.currentData()
        preset = self.explore_review_state.recipe_presets.get(preset_id)
        if preset is None:
            raise ValueError("Select a saved Explore recipe first.")
        return preset

    def _validate_recipe_preset_identity(
        self,
        *,
        name: str,
        shortcut: str | None,
        exclude_id: str | None = None,
    ) -> tuple[str, str | None]:
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("Enter a name for the Explore recipe.")
        for preset in self.explore_review_state.recipe_presets.values():
            if preset.preset_id == exclude_id:
                continue
            if preset.name.casefold() == clean_name.casefold():
                raise ValueError(
                    f"An Explore recipe named {clean_name!r} already exists."
                )
            if shortcut is not None and preset.shortcut == shortcut:
                raise ValueError(
                    f"{shortcut} is already assigned to Explore recipe {preset.name!r}."
                )
        return clean_name, shortcut

    def save_new_recipe_preset(self) -> None:
        if self.paths is None:
            raise ValueError(
                "Create or load a workflow workspace before saving recipes."
            )
        self._capture_current_recipe_display_state()
        if not self.explore_recipe.has_content:
            raise ValueError(
                "Build an Explore view with at least one layer before saving it."
            )
        name, shortcut = self._validate_recipe_preset_identity(
            name=self.recipe_preset_name_edit.text(),
            shortcut=self.recipe_preset_shortcut_combo.currentData(),
        )
        preset_id = str(uuid4())
        preset = ExploreRecipePreset(
            preset_id=preset_id,
            name=name,
            shortcut=shortcut,
            recipe=self.explore_recipe.model_copy(deep=True),
        )
        self.explore_review_state.recipe_presets[preset_id] = preset
        self.explore_review_state.active_recipe_id = preset_id
        self._save_explore_review_state()
        self._bind_explore_recipe_shortcuts()
        self._refresh_recipe_preset_controls(selected_id=preset_id)
        append_audit(
            self.paths,
            {
                "action": "create_explore_recipe_preset",
                "preset_id": preset_id,
                "name": name,
                "shortcut": shortcut,
                "view_fingerprint": preset.recipe.fingerprint,
            },
        )
        self.set_status(
            f"Saved Explore recipe {name!r}"
            + (f" and assigned {shortcut}." if shortcut else ".")
        )

    def update_selected_recipe_preset(self) -> None:
        self._capture_current_recipe_display_state()
        if not self.explore_recipe.has_content:
            raise ValueError(
                "The current Explore view is empty and cannot replace a saved recipe."
            )
        existing = self._selected_recipe_preset()
        name, shortcut = self._validate_recipe_preset_identity(
            name=self.recipe_preset_name_edit.text(),
            shortcut=self.recipe_preset_shortcut_combo.currentData(),
            exclude_id=existing.preset_id,
        )
        updated = ExploreRecipePreset(
            preset_id=existing.preset_id,
            name=name,
            shortcut=shortcut,
            recipe=self.explore_recipe.model_copy(deep=True),
        )
        self.explore_review_state.recipe_presets[existing.preset_id] = updated
        self.explore_review_state.active_recipe_id = existing.preset_id
        self._save_explore_review_state()
        self._bind_explore_recipe_shortcuts()
        self._refresh_recipe_preset_controls(selected_id=existing.preset_id)
        append_audit(
            self.paths,
            {
                "action": "update_explore_recipe_preset",
                "preset_id": existing.preset_id,
                "previous_name": existing.name,
                "name": name,
                "previous_shortcut": existing.shortcut,
                "shortcut": shortcut,
                "view_fingerprint": updated.recipe.fingerprint,
            },
        )
        self.set_status(f"Updated Explore recipe {name!r} from the current layer view.")

    def load_selected_recipe_preset(self) -> None:
        self._activate_recipe_preset(
            self._selected_recipe_preset().preset_id,
            source="Explore recipe selector",
        )

    def _activate_recipe_preset(self, preset_id: str, *, source: str) -> None:
        preset = self.explore_review_state.recipe_presets.get(str(preset_id))
        if preset is None:
            raise ValueError("The requested Explore recipe no longer exists.")
        active_changed = self.explore_review_state.active_recipe_id != preset.preset_id
        self.explore_review_state.active_recipe_id = preset.preset_id
        if active_changed:
            self._save_explore_review_state()
        self._select_recipe_preset_control(preset.preset_id)
        self._apply_explore_recipe(preset.recipe)
        if not self.current_roi:
            self.set_status(f"Loaded Explore recipe {preset.name!r} via {source}.")

    def delete_selected_recipe_preset(self) -> None:
        preset = self._selected_recipe_preset()
        reply = self.QMessageBox.question(
            self.root,
            "Delete Explore recipe",
            f"Delete saved Explore recipe {preset.name!r}?\n\nThe current Napari "
            "layers will remain visible, but the recipe and its F-key assignment "
            "will be removed.",
        )
        if reply != self.QMessageBox.Yes:
            self.set_status("Explore recipe deletion cancelled.")
            return
        del self.explore_review_state.recipe_presets[preset.preset_id]
        if self.explore_review_state.active_recipe_id == preset.preset_id:
            self.explore_review_state.active_recipe_id = None
        self._save_explore_review_state()
        self._bind_explore_recipe_shortcuts()
        self._refresh_recipe_preset_controls()
        append_audit(
            self.paths,
            {
                "action": "delete_explore_recipe_preset",
                "preset_id": preset.preset_id,
                "name": preset.name,
                "shortcut": preset.shortcut,
                "view_fingerprint": preset.recipe.fingerprint,
            },
        )
        self.set_status(f"Deleted Explore recipe {preset.name!r}.")

    def import_explore_recipe_preset(self) -> None:
        """Import a portable named recipe without discarding unavailable layers."""

        if self.paths is None:
            raise ValueError(
                "Create or load a workflow workspace before importing recipes."
            )
        selected, _filter = self.QFileDialog.getOpenFileName(
            self.root,
            "Import Explore recipe",
            str(self.paths.root),
            "Explore recipe JSON (*.json)",
        )
        if not selected:
            return
        payload = json.loads(Path(selected).read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "recipe" in payload:
            imported = ExploreRecipePreset.model_validate(payload)
        else:
            imported = ExploreRecipePreset(
                preset_id=str(uuid4()),
                name=Path(selected).stem,
                shortcut=None,
                recipe=ExploreViewRecipe.model_validate(payload),
            )

        existing_names = {
            preset.name.casefold()
            for preset in self.explore_review_state.recipe_presets.values()
        }
        name = imported.name
        suffix = 2
        while name.casefold() in existing_names:
            name = f"{imported.name} (imported {suffix})"
            suffix += 1
        shortcut = imported.shortcut
        used_shortcuts = {
            preset.shortcut
            for preset in self.explore_review_state.recipe_presets.values()
            if preset.shortcut is not None
        }
        if shortcut in used_shortcuts:
            shortcut = None
        preset_id = imported.preset_id
        if preset_id in self.explore_review_state.recipe_presets:
            preset_id = str(uuid4())
        imported = ExploreRecipePreset(
            preset_id=preset_id,
            name=name,
            shortcut=shortcut,
            recipe=imported.recipe.model_copy(deep=True),
        )
        self.explore_review_state.recipe_presets[preset_id] = imported
        self._save_explore_review_state()
        self._bind_explore_recipe_shortcuts()
        self._refresh_recipe_preset_controls(selected_id=preset_id)
        append_audit(
            self.paths,
            {
                "action": "import_explore_recipe_preset",
                "preset_id": preset_id,
                "name": name,
                "shortcut": shortcut,
                "source": str(Path(selected).resolve(strict=False)),
                "view_fingerprint": imported.recipe.fingerprint,
            },
        )
        self.set_status(
            f"Imported Explore recipe {name!r}. Layers unavailable in this "
            "workflow remain stored and are highlighted in the recipe contents."
        )

    def export_selected_explore_recipe_preset(self) -> None:
        """Export the selected named recipe as portable JSON."""

        preset = self._selected_recipe_preset()
        initial_folder = (
            self.paths.root if self.paths is not None else self.project_root
        )
        default_name = slugify(preset.name) or "explore_recipe"
        selected, _filter = self.QFileDialog.getSaveFileName(
            self.root,
            "Export Explore recipe",
            str(initial_folder / f"{default_name}.json"),
            "Explore recipe JSON (*.json)",
        )
        if not selected:
            return
        destination = Path(selected)
        if destination.suffix.lower() != ".json":
            destination = destination.with_suffix(".json")
        write_json(destination, preset.model_dump(mode="json"))
        self.set_status(f"Exported Explore recipe {preset.name!r} to {destination}.")

    def _bind_explore_recipe_shortcuts(self) -> None:
        for shortcut in self._explore_recipe_shortcuts:
            try:
                self.viewer.bind_key(shortcut, None, overwrite=True)
            except Exception as error:  # noqa: BLE001 - optional Napari key backend
                self.set_status(
                    f"Could not unbind Explore recipe shortcut {shortcut}: {error}"
                )
        self._explore_recipe_shortcuts = []
        for preset in self.explore_review_state.recipe_presets.values():
            if preset.shortcut is None:
                continue

            def activate_recipe(
                _viewer,
                selected_id=preset.preset_id,
                selected_shortcut=preset.shortcut,
            ):
                try:
                    self._activate_recipe_preset(
                        selected_id,
                        source=str(selected_shortcut),
                    )
                except Exception as error:  # noqa: BLE001 - Napari key boundary
                    self.set_status(
                        f"Could not load Explore recipe via "
                        f"{selected_shortcut}: {error}"
                    )

            self.viewer.bind_key(preset.shortcut, overwrite=True)(activate_recipe)
            self._explore_recipe_shortcuts.append(preset.shortcut)

    def _current_population_recipe_key(self) -> str | None:
        observation = self.population_obs_combo.currentText().strip()
        population = self.population_value_combo.currentText().strip()
        if not observation or not population:
            return None
        return population_recipe_key(observation, population)

    def save_population_view(self) -> None:
        self._capture_current_recipe_display_state()
        if not self.explore_recipe.has_content:
            raise ValueError("Load at least one Explore layer before saving a view.")
        key = self._current_population_recipe_key()
        if key is None:
            raise ValueError("Select a population observation and population first.")
        self.explore_review_state.population_recipes[key] = (
            self._population_qc_recipe_for_storage(self.explore_recipe)
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
            "marker overlays, layer visibility, opacity, and contrast limits for "
            f"{self.population_value_combo.currentText()!r}. The population "
            "outline remains a workspace-wide setting."
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
        self.explore_review_state.active_recipe_id = None
        self._save_explore_review_state()
        self._refresh_recipe_preset_controls()
        self._apply_explore_recipe(
            self._population_recipe_with_global_outline(recipe)
        )
        self.set_status(
            f"Loaded the saved Explore view for "
            f"{self.population_value_combo.currentText()!r}."
        )

    def _set_list_selection(self, widget, values: Iterable[str]) -> None:
        selected = {str(value) for value in values}
        for index in range(widget.count()):
            item = widget.item(index)
            should_select = item.text() in selected
            if item.isSelected() != should_select:
                item.setSelected(should_select)

    def _recipe_colour_summary(
        self, name: str, default: str | None = None
    ) -> str | None:
        spec = self.explore_recipe.layer_colormap_specs.get(name)
        if spec and spec.get("kind") == "direct_labels":
            colours = [
                value
                for key, value in spec.get("colours", {}).items()
                if key not in {"__default__", "0"}
            ]
            if len(colours) == 1:
                rgba = np.asarray(colours[0], dtype=float)
                rgb = np.clip(np.rint(rgba[:3] * 255), 0, 255).astype(np.uint8)
                return "#" + "".join(f"{channel:02x}" for channel in rgb)
            if colours:
                return f"{len(colours)} saved layer colours"
        if spec and spec.get("kind") == "continuous":
            return str(spec.get("name") or "saved custom colormap")
        return self.explore_recipe.layer_colormaps.get(name, default)

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
                    else SIX_COLOUR_COLORMAPS[index % len(SIX_COLOUR_COLORMAPS)]
                )
                colormap = self._recipe_colour_summary(name, default_colormap)
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
            colormap = self._recipe_colour_summary(name)
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
            scope_suffix = (
                " [full dataset]"
                if recipe.observation_overlay_full_dataset
                else " [classification cohort]"
            )
            entries.append(
                {
                    "kind": "observation",
                    "name": name,
                    "observation": recipe.observation_overlay,
                    "full_dataset": recipe.observation_overlay_full_dataset,
                    "description": (
                        f"AnnData observation{suffix}{scope_suffix}: "
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
                name = f"population::{recipe.population_observation}::{population}"
                colour = self._recipe_colour_summary(
                    name,
                    population_colours.get(population),
                )
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
            colormap = self._recipe_colour_summary(name, "viridis")
            entries.append(
                {
                    "kind": "marker",
                    "name": name,
                    "marker": marker,
                    "description": f"adata.X marker [{colormap}]: {marker}",
                }
            )
        if self.manifest is not None:
            mode = self.current_workflow_mode()
            classifier_names = {
                NONCONTEXT_MASK_LAYER_NAME,
                *CLASS_LAYER_NAMES.values(),
                SELECTED_CELL_LAYER_NAME,
            }
            labeler_names = {LABELER_LAYER_NAME, LABELER_SELECTED_CELL_LAYER_NAME}
            configured_names = set().union(
                recipe.layer_colormaps,
                recipe.layer_colormap_specs,
                recipe.layer_visibility,
                recipe.layer_opacities,
                recipe.layer_contours,
                recipe.layer_contrast_limits,
            )
            for name, description in MANAGED_RECIPE_LAYERS.items():
                relevant = name not in classifier_names | labeler_names
                relevant |= name in classifier_names and mode in {
                    "classification",
                    "full_workspace",
                }
                relevant |= name in labeler_names and mode in {
                    "cell_labeling",
                    "full_workspace",
                }
                relevant |= name in configured_names or name in self.viewer.layers
                if not relevant:
                    continue
                entries.append(
                    {
                        "kind": "managed",
                        "name": name,
                        "description": description,
                    }
                )
        represented = {entry["name"] for entry in entries}
        saved_layer_names = set().union(
            recipe.layer_colormaps,
            recipe.layer_colormap_specs,
            recipe.layer_visibility,
            recipe.layer_opacities,
            recipe.layer_contours,
            recipe.layer_contrast_limits,
        )
        for name in sorted(saved_layer_names - represented):
            entries.append(
                {
                    "kind": "saved_absent",
                    "name": name,
                    "description": (
                        f"Saved layer without a current reconstruction source: {name}"
                    ),
                }
            )
        return entries

    def _recipe_entry_available(self, entry: dict) -> bool | None:
        """Report whether a saved layer can be reconstructed in this session/ROI."""

        kind = entry.get("kind")
        if kind == "saved_absent":
            return False
        if kind == "image":
            if not self.current_image_paths:
                return None
            return str(entry.get("channel", "")) in self.current_image_paths
        if kind == "rgb":
            if not self.current_image_paths:
                return None
            return all(
                str(channel) in self.current_image_paths
                for channel in entry.get("channels", [])
            )
        if kind == "observation":
            return bool(
                self.adata is not None
                and str(entry.get("observation", "")) in self.adata.obs
            )
        if kind == "population":
            observation = str(entry.get("observation", ""))
            population = str(entry.get("population", ""))
            return bool(
                self.adata is not None
                and observation in self.adata.obs
                and self.adata.obs[observation].astype("string").eq(population).any()
            )
        if kind == "marker":
            return bool(
                self.adata is not None
                and str(entry.get("marker", "")) in self.adata.var_names.astype(str)
            )
        if kind == "managed":
            if not self.current_roi:
                return None
            return str(entry.get("name", "")) in self.viewer.layers
        return None

    def _workflow_tab_changed(self, index: int) -> None:
        if index == getattr(self, "population_qc_tab_index", -1):
            self.refresh_population_qc_rois()
        if index != self.explore_tab_index:
            return
        if self._recipe_list_refresh_pending:
            self._refresh_reload_recipe_list(force=True)
        if self._roi_review_refresh_pending:
            self._refresh_roi_review_colours(force=True)

    def _refresh_reload_recipe_list(self, *, force: bool = False) -> None:
        if not hasattr(self, "reload_recipe_list"):
            return
        if (
            not force
            and hasattr(self, "explore_tab_index")
            and self.tabs.currentIndex() != self.explore_tab_index
        ):
            self._recipe_list_refresh_pending = True
            return
        self._recipe_list_refresh_pending = False
        self.reload_recipe_list.clear()
        self._refresh_active_recipe_preset_label()
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
            contour_text = f", contour {contour}px" if contour is not None else ""
            contrast_limits = self.explore_recipe.layer_contrast_limits.get(name)
            contrast_text = (
                f", contrast {contrast_limits[0]:g}–{contrast_limits[1]:g}"
                if contrast_limits is not None
                else ""
            )
            availability = self._recipe_entry_available(entry)
            availability_text = ""
            if availability is False:
                availability_text = " — ⚠ absent in this workflow or ROI"
            elif availability is None:
                availability_text = " — availability checked when an ROI is loaded"
            item = QListWidgetItem(
                f"{entry['description']} — {state}, opacity {opacity:.2f}"
                f"{contour_text}"
                f"{contrast_text}"
                f"{availability_text}"
            )
            item.setData(self.Qt.UserRole, entry)
            if availability is False:
                item.setForeground(self.QColor("#9a3412"))
                item.setBackground(self.QColor("#ffedd5"))
            item.setToolTip(
                f"Napari layer: {name}\nThis layer will be reconstructed for "
                "the next ROI with this visibility, opacity, contour style, "
                "and contrast limits. Missing layers remain saved and are retried "
                "when their data or workflow becomes available."
            )
            self.reload_recipe_list.addItem(item)

    def _drop_recipe_layer_settings(self, payload: dict, name: str) -> None:
        for key in (
            "layer_colormaps",
            "layer_colormap_specs",
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
                payload["observation_overlay_full_dataset"] = False
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
        if name == "population_qc_rgb" and self.explore_recipe.image_mode == "rgb":
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
                "full_dataset": self.explore_recipe.observation_overlay_full_dataset,
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

    def _layer_colormap_spec(self, layer) -> dict | None:
        """Serialize the colours currently displayed by one Napari layer."""

        colormap = getattr(layer, "colormap", None)
        if colormap is None:
            return None
        color_dict = getattr(colormap, "color_dict", None)
        if isinstance(color_dict, dict):
            colours: dict[str, list[float]] = {}
            for label, colour in color_dict.items():
                key = "__default__" if label is None else str(int(label))
                rgba = np.asarray(colour, dtype=float).reshape(-1)
                if rgba.size in {3, 4} and np.isfinite(rgba).all():
                    if rgba.size == 3:
                        rgba = np.append(rgba, 1.0)
                    colours[key] = rgba.tolist()
            return {"kind": "direct_labels", "colours": colours} if colours else None

        colours = getattr(colormap, "colors", None)
        if colours is None:
            return None
        rgba = np.asarray(colours, dtype=float)
        if rgba.ndim != 2 or rgba.shape[1] not in {3, 4} or not np.isfinite(rgba).all():
            return None
        controls = np.asarray(getattr(colormap, "controls", []), dtype=float)
        interpolation = getattr(colormap, "interpolation", "linear")
        interpolation = getattr(interpolation, "value", interpolation)
        return {
            "kind": "continuous",
            "name": self._layer_colormap_name(layer) or "napari_sbt_saved",
            "colours": rgba.tolist(),
            "controls": controls.tolist(),
            "interpolation": str(interpolation),
        }

    def _recipe_colormap(self, name: str, default=None):
        """Rebuild the exact saved colormap, falling back to legacy names."""

        spec = self.explore_recipe.layer_colormap_specs.get(name)
        if spec:
            try:
                if spec.get("kind") == "direct_labels":
                    from napari.utils.colormaps import DirectLabelColormap

                    colour_dict = {}
                    for key, colour in spec.get("colours", {}).items():
                        label = None if key == "__default__" else int(key)
                        colour_dict[label] = colour
                    return DirectLabelColormap(color_dict=colour_dict)
                if spec.get("kind") == "continuous":
                    from napari.utils.colormaps import Colormap

                    return Colormap(
                        colors=spec["colours"],
                        controls=spec.get("controls") or None,
                        interpolation=spec.get("interpolation", "linear"),
                        name=spec.get("name", "napari_sbt_saved"),
                    )
            except (KeyError, TypeError, ValueError):
                # A legacy or externally edited recipe should remain usable.
                pass
        return self.explore_recipe.layer_colormaps.get(name, default)

    def _write_layer_display_state(self, payload: dict, layer) -> None:
        """Copy live display settings into a mutable recipe payload."""

        name = str(getattr(layer, "name", ""))
        payload["layer_visibility"][name] = bool(getattr(layer, "visible", True))
        payload["layer_opacities"][name] = float(getattr(layer, "opacity", 1.0))
        if hasattr(layer, "contour"):
            payload["layer_contours"][name] = int(layer.contour)
        if hasattr(layer, "contrast_limits"):
            limits = layer.contrast_limits
            payload["layer_contrast_limits"][name] = [
                float(limits[0]),
                float(limits[1]),
            ]
        descriptor = self._layer_reload_descriptor(layer)
        if descriptor is None or descriptor.get("kind") == "rgb":
            return
        spec = self._layer_colormap_spec(layer)
        if spec is not None:
            payload["layer_colormap_specs"][name] = spec
        colormap_name = self._layer_colormap_name(layer)
        if colormap_name and spec is not None and spec.get("kind") == "continuous":
            payload["layer_colormaps"][name] = colormap_name

    def _capture_current_recipe_display_state(self) -> None:
        """Synchronize the recipe with the actual colours and display state."""

        valid_names = {entry["name"] for entry in self._recipe_layer_entries()}
        payload = self.explore_recipe.model_dump(mode="json")
        for name in valid_names:
            if name in self.viewer.layers:
                self._write_layer_display_state(payload, self.viewer.layers[name])
        self.explore_recipe = ExploreViewRecipe.model_validate(payload)
        self._refresh_reload_recipe_list()
        self._refresh_roi_review_colours()

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
            message = (
                "No replayable Explore or classifier layers are currently present. "
                "The saved ROI reload recipe was left unchanged so temporarily absent "
                "layers are not lost. Use Delete selected recipe items to remove them."
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
            ignored.extend(descriptor["name"] for _layer, descriptor in images)
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
            image_channels = [descriptor["channel"] for _layer, descriptor in included]
            image_mode = "none"
            if image_channels:
                colormaps = {
                    self._layer_colormap_name(layer) for layer, _descriptor in included
                }
                image_mode = (
                    "grayscale" if colormaps <= {None, "gray", "grey"} else "six_colour"
                )
        unavailable_saved_channels = [
            channel
            for channel in self.explore_recipe.image_channels
            if channel not in self.current_image_paths
        ]
        if image_mode == "none" and unavailable_saved_channels:
            image_mode = self.explore_recipe.image_mode
            image_channels = (
                list(self.explore_recipe.image_channels)
                if image_mode == "rgb"
                else unavailable_saved_channels
            )
        elif image_mode == self.explore_recipe.image_mode and image_mode != "rgb":
            image_channels = list(
                dict.fromkeys([*image_channels, *unavailable_saved_channels])
            )

        observation_layers = [
            (layer, descriptor)
            for layer, descriptor in descriptors
            if descriptor["kind"] == "observation"
        ]
        observation_overlay = None
        observation_overlay_full_dataset = False
        if observation_layers:
            observation_overlay = observation_layers[-1][1]["observation"]
            observation_overlay_full_dataset = bool(
                observation_layers[-1][1].get("full_dataset", False)
            )
            ignored.extend(
                descriptor["name"] for _layer, descriptor in observation_layers[:-1]
            )
        elif self.explore_recipe.observation_overlay and (
            self.adata is None
            or self.explore_recipe.observation_overlay not in self.adata.obs
        ):
            observation_overlay = self.explore_recipe.observation_overlay
            observation_overlay_full_dataset = (
                self.explore_recipe.observation_overlay_full_dataset
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
        elif self.explore_recipe.population_observation and (
            self.adata is None
            or self.explore_recipe.population_observation not in self.adata.obs
        ):
            population_observation = self.explore_recipe.population_observation
            populations = list(self.explore_recipe.populations)

        marker_layers = [
            (layer, descriptor)
            for layer, descriptor in descriptors
            if descriptor["kind"] == "marker"
        ]
        marker_overlays = [descriptor["marker"] for _layer, descriptor in marker_layers]
        if self.adata is None:
            unavailable_saved_markers = list(self.explore_recipe.marker_overlays)
        else:
            available_markers = set(self.adata.var_names.astype(str))
            unavailable_saved_markers = [
                marker
                for marker in self.explore_recipe.marker_overlays
                if marker not in available_markers
            ]
        marker_overlays = list(
            dict.fromkeys([*marker_overlays, *unavailable_saved_markers])
        )
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
        current_layer_names = {
            str(getattr(layer, "name", "")) for layer in self.viewer.layers
        }
        preserved_names = (
            set().union(
                self.explore_recipe.layer_colormaps,
                self.explore_recipe.layer_colormap_specs,
                self.explore_recipe.layer_visibility,
                self.explore_recipe.layer_opacities,
                self.explore_recipe.layer_contours,
                self.explore_recipe.layer_contrast_limits,
            )
            - current_layer_names
        )
        layer_colormaps: dict[str, str] = {
            name: value
            for name, value in self.explore_recipe.layer_colormaps.items()
            if name in preserved_names
        }
        layer_colormap_specs: dict[str, dict] = {
            name: value
            for name, value in self.explore_recipe.layer_colormap_specs.items()
            if name in preserved_names
        }
        layer_visibility: dict[str, bool] = {
            name: value
            for name, value in self.explore_recipe.layer_visibility.items()
            if name in preserved_names or name in MANAGED_RECIPE_LAYERS
        }
        layer_opacities: dict[str, float] = {
            name: value
            for name, value in self.explore_recipe.layer_opacities.items()
            if name in preserved_names or name in MANAGED_RECIPE_LAYERS
        }
        layer_contours: dict[str, int] = {
            name: int(value)
            for name, value in self.explore_recipe.layer_contours.items()
            if name in preserved_names or name in MANAGED_RECIPE_LAYERS
        }
        layer_contrast_limits: dict[str, tuple[float, float]] = {
            name: (float(value[0]), float(value[1]))
            for name, value in self.explore_recipe.layer_contrast_limits.items()
            if name in preserved_names or name in MANAGED_RECIPE_LAYERS
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
            colormap_spec = self._layer_colormap_spec(layer)
            if colormap_spec is not None and descriptor["kind"] != "rgb":
                layer_colormap_specs[name] = colormap_spec
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
            observation_overlay_full_dataset=observation_overlay_full_dataset,
            population_observation=population_observation,
            populations=list(dict.fromkeys(populations)),
            marker_overlays=list(dict.fromkeys(marker_overlays)),
            layer_colormaps=layer_colormaps,
            layer_colormap_specs=layer_colormap_specs,
            layer_visibility=layer_visibility,
            layer_opacities=layer_opacities,
            layer_contours=layer_contours,
            layer_contrast_limits=layer_contrast_limits,
        )
        self._apply_explore_recipe(recipe)
        included_count = len(included_names) + len(managed_layers)
        message = (
            f"Updated the ROI reload recipe from {included_count} current "
            "Explore/classifier layer(s), including the actual layer colours, "
            "visibility, opacity, label contours, and contrast limits."
        )
        if ignored:
            message += (
                " Ignored unsupported or conflicting layers: "
                + ", ".join(ignored)
                + "."
            )
        self.set_status(message)

    def _apply_explore_recipe(
        self, recipe: ExploreViewRecipe, *, replay: bool = True
    ) -> None:
        self._applying_explore_recipe = True
        try:
            self.explore_recipe = recipe.model_copy(deep=True)
            if (
                recipe.observation_overlay
                and self.overlay_obs_combo.findText(recipe.observation_overlay) >= 0
            ):
                overlay_was_blocked = self.overlay_obs_combo.blockSignals(True)
                self.overlay_obs_combo.setCurrentText(recipe.observation_overlay)
                self.overlay_obs_combo.blockSignals(overlay_was_blocked)
            self.overlay_full_dataset_check.setChecked(
                recipe.observation_overlay_full_dataset
            )
            if (
                recipe.population_observation
                and self.population_obs_combo.findText(recipe.population_observation)
                >= 0
            ):
                population_changed = (
                    self.population_obs_combo.currentText()
                    != recipe.population_observation
                )
                population_was_blocked = self.population_obs_combo.blockSignals(True)
                self.population_obs_combo.setCurrentText(recipe.population_observation)
                self.population_obs_combo.blockSignals(population_was_blocked)
                if population_changed or self.population_layer_list.count() == 0:
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
        if replay and self.current_roi:
            self.replay_explore_view()
        else:
            self._refresh_reload_recipe_list()

    def _mark_current_explore_viewed(self) -> None:
        if not self.current_roi or not self.explore_recipe.has_content:
            return
        fingerprint = self.explore_recipe.fingerprint
        viewed = set(self.explore_review_state.viewed_rois.get(fingerprint, []))
        roi = str(self.current_roi)
        if roi not in viewed:
            viewed.add(roi)
            self.explore_review_state.viewed_rois[fingerprint] = sorted(viewed)
        # One coalesced write persists both a newly selected Population QC recipe
        # and its review state, including the revisit case where the ROI was
        # already present under this fingerprint.
        self._save_explore_review_state()
        self._refresh_roi_review_colours()

    def _refresh_roi_review_colours(self, *, force: bool = False) -> None:
        if not hasattr(self, "roi_combo"):
            return
        if (
            not force
            and hasattr(self, "explore_tab_index")
            and self.tabs.currentIndex() != self.explore_tab_index
        ):
            self._roi_review_refresh_pending = True
            return
        self._roi_review_refresh_pending = False
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
        if not self._recipe_tracking_enabled():
            return
        if getattr(layer, "_napari_sbt_recipe_display_bound", False):
            return
        events = getattr(layer, "events", None)
        if events is None:
            return

        def display_changed(_event=None, tracked_layer=layer):
            self._record_layer_display_state(tracked_layer)

        for event_name in (
            "visible",
            "opacity",
            "contour",
            "contrast_limits",
            "colormap",
        ):
            emitter = getattr(events, event_name, None)
            if emitter is not None:
                emitter.connect(display_changed)
        layer._napari_sbt_recipe_display_callback = display_changed
        layer._napari_sbt_recipe_display_bound = True

    def _record_layer_display_state(self, layer) -> None:
        if self._updating_recipe_layer_state or not self._recipe_tracking_enabled():
            return
        name = str(getattr(layer, "name", ""))
        if not self._is_recipe_tracked_layer(name):
            return
        payload = self.explore_recipe.model_dump(mode="json")
        self._write_layer_display_state(payload, layer)
        self.explore_recipe = ExploreViewRecipe.model_validate(payload)
        if name == "excluded_segmentation_context":
            self.context_check_display.blockSignals(True)
            self.context_check_display.setChecked(bool(layer.visible))
            self.context_check_display.blockSignals(False)
        self._refresh_reload_recipe_list()
        self._refresh_roi_review_colours()

    @staticmethod
    def _colormap_specs_match(current: dict | None, desired: dict | None) -> bool:
        if not current or not desired or current.get("kind") != desired.get("kind"):
            return False
        try:
            if current["kind"] == "direct_labels":
                current_colours = current.get("colours", {})
                desired_colours = desired.get("colours", {})
                return set(current_colours) == set(desired_colours) and all(
                    np.allclose(current_colours[key], desired_colours[key])
                    for key in current_colours
                )
            if current["kind"] == "continuous":
                return (
                    str(current.get("interpolation", "linear"))
                    == str(desired.get("interpolation", "linear"))
                    and np.allclose(
                        current.get("colours", []), desired.get("colours", [])
                    )
                    and np.allclose(
                        current.get("controls", []), desired.get("controls", [])
                    )
                )
        except (KeyError, TypeError, ValueError):
            return False
        return False

    def _layer_display_setting_matches(
        self,
        layer,
        name: str,
        key: str,
        desired,
    ) -> bool:
        if key == "colormap":
            desired_spec = self.explore_recipe.layer_colormap_specs.get(name)
            if desired_spec is not None:
                return self._colormap_specs_match(
                    self._layer_colormap_spec(layer),
                    desired_spec,
                )
            if isinstance(desired, str):
                return self._layer_colormap_name(layer) == desired
            return False
        current = getattr(layer, key)
        if key in {"opacity", "contrast_limits"}:
            try:
                return bool(np.allclose(current, desired))
            except (TypeError, ValueError):
                return False
        try:
            return bool(current == desired)
        except (TypeError, ValueError):
            return False

    def _set_layer_display_setting(
        self,
        layer,
        name: str,
        key: str,
        desired,
    ) -> bool:
        if not hasattr(layer, key) or self._layer_display_setting_matches(
            layer,
            name,
            key,
            desired,
        ):
            return False
        setattr(layer, key, desired)
        return True

    def _set_label_contour_from_recipe(
        self,
        layer,
        name: str,
        default: int = 1,
    ) -> None:
        if not hasattr(layer, "contour"):
            return
        previous_state = self._updating_recipe_layer_state
        self._updating_recipe_layer_state = True
        try:
            self._set_layer_display_setting(
                layer,
                name,
                "contour",
                self.explore_recipe.layer_contours.get(name, default),
            )
        finally:
            self._updating_recipe_layer_state = previous_state

    def _apply_managed_layer_display_settings(
        self, *, refresh_recipe_list: bool = True
    ) -> None:
        self._updating_recipe_layer_state = True
        try:
            for name in MANAGED_RECIPE_LAYERS:
                if name not in self.viewer.layers:
                    continue
                layer = self.viewer.layers[name]
                self._set_layer_display_setting(
                    layer,
                    name,
                    "visible",
                    self.explore_recipe.layer_visibility.get(
                        name,
                        MANAGED_LAYER_DEFAULT_VISIBILITY[name],
                    ),
                )
                self._set_layer_display_setting(
                    layer,
                    name,
                    "opacity",
                    self.explore_recipe.layer_opacities.get(
                        name,
                        MANAGED_LAYER_DEFAULT_OPACITY[name],
                    ),
                )
                if name in MANAGED_LAYER_DEFAULT_CONTOUR and hasattr(layer, "contour"):
                    self._set_layer_display_setting(
                        layer,
                        name,
                        "contour",
                        self.explore_recipe.layer_contours.get(
                            name,
                            MANAGED_LAYER_DEFAULT_CONTOUR[name],
                        ),
                    )
                if hasattr(layer, "contrast_limits"):
                    contrast_limits = self.explore_recipe.layer_contrast_limits.get(
                        name
                    )
                    if contrast_limits is not None:
                        self._set_layer_display_setting(
                            layer,
                            name,
                            "contrast_limits",
                            contrast_limits,
                        )
                self._bind_recipe_display_tracking(layer)
            context_visible = self.explore_recipe.layer_visibility.get(
                "excluded_segmentation_context",
                MANAGED_LAYER_DEFAULT_VISIBILITY["excluded_segmentation_context"],
            )
            self.context_check_display.blockSignals(True)
            self.context_check_display.setChecked(context_visible)
            self.context_check_display.blockSignals(False)
        finally:
            self._updating_recipe_layer_state = False
        if refresh_recipe_list:
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
            self._set_explore_reload_metadata(layer, name, reload_descriptor)
            self._set_normalized_image_contrast_range(layer, reload_descriptor)
            self._cache_explore_layer_data(name, reload_descriptor, data)
        self._explore_layer_names.add(name)
        if (
            hasattr(layer, "contrast_limits")
            and name not in self.explore_recipe.layer_contrast_limits
        ):
            # Napari derives initial limits from the first ROI. Freeze those
            # limits immediately so subsequent ROIs use the identical view.
            self._record_layer_display_state(layer)
        return layer

    def _set_explore_reload_metadata(
        self, layer, name: str, reload_descriptor: dict
    ) -> None:
        metadata = dict(getattr(layer, "metadata", {}) or {})
        metadata["napari_sbt_reload"] = {
            "name": name,
            "roi": str(self.current_roi or ""),
            **reload_descriptor,
        }
        layer.metadata = metadata

    def _reuse_explore_layer(
        self,
        name: str,
        reload_descriptor: dict,
        **display_settings,
    ):
        """Apply recipe display settings without replacing current layer data."""

        if name not in self.viewer.layers:
            return None
        layer = self.viewer.layers[name]
        if not recipe_layer_data_is_current(
            getattr(layer, "metadata", None),
            name=name,
            roi=str(self.current_roi or ""),
            reload_descriptor=reload_descriptor,
        ):
            return None
        previous_state = self._updating_recipe_layer_state
        self._updating_recipe_layer_state = True
        try:
            for key, value in display_settings.items():
                if key == "rgb":
                    continue
                self._set_layer_display_setting(layer, name, key, value)
            self._set_normalized_image_contrast_range(layer, reload_descriptor)
            self._set_explore_reload_metadata(layer, name, reload_descriptor)
        finally:
            self._updating_recipe_layer_state = previous_state
        self._explore_layer_names.add(name)
        self._bind_recipe_display_tracking(layer)
        self._explore_reused_layer_count += 1
        return layer

    @staticmethod
    def _set_normalized_image_contrast_range(layer, reload_descriptor: dict) -> None:
        """Keep normalized disk-image slider bounds at 0..1 after recipe replay."""

        if (
            reload_descriptor.get("kind") != "image"
            or bool(getattr(layer, "rgb", False))
            or not hasattr(layer, "contrast_limits_range")
        ):
            return
        limits = [float(value) for value in layer.contrast_limits]
        if not (np.isfinite(limits).all() and 0.0 <= limits[0] < limits[1] <= 1.0):
            # An old or deliberately unusual recipe may contain out-of-range
            # handles.  Do not clip and silently alter those scientific display
            # settings merely to normalize the slider extent.
            return
        # load_display_image clips scalar image channels to this scientific
        # display scale.  Set the range *after* restoring saved handles: Napari
        # otherwise adopts the handles themselves as the slider's full extent.
        layer.contrast_limits_range = (0.0, 1.0)

    @staticmethod
    def _image_source_identity(path: Path) -> dict[str, int | str]:
        """Return a cheap identity that invalidates reuse when an image changes."""

        resolved = Path(path).expanduser().resolve(strict=False)
        stat = resolved.stat()
        return {
            "path": str(resolved),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def _remove_explore_layer(self, name: str) -> None:
        self._remove_layers([name])
        self._explore_layer_names.discard(name)

    def _clear_explore_layers(self) -> None:
        self._remove_layers(list(self._explore_layer_names))
        self._explore_layer_names.clear()

    def _explore_data_cache_key(self, name: str, reload_descriptor: dict) -> str:
        descriptor = {
            key: value
            for key, value in reload_descriptor.items()
            if key not in {"mode"}
        }
        return json.dumps(
            {
                "roi": str(self.current_roi or ""),
                "name": str(name),
                "descriptor": descriptor,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _clear_explore_layer_data_cache(self) -> None:
        self._explore_layer_data_cache.clear()
        self._explore_layer_data_cache_bytes = 0

    def _cache_explore_layer_data(
        self,
        name: str,
        reload_descriptor: dict,
        data,
    ) -> None:
        array = np.asarray(data)
        size = int(getattr(array, "nbytes", 0))
        if size <= 0 or size > EXPLORE_DATA_CACHE_MAX_BYTES:
            return
        key = self._explore_data_cache_key(name, reload_descriptor)
        previous = self._explore_layer_data_cache.pop(key, None)
        if previous is not None:
            self._explore_layer_data_cache_bytes -= int(previous.nbytes)
        self._explore_layer_data_cache[key] = array
        self._explore_layer_data_cache_bytes += size
        while (
            len(self._explore_layer_data_cache) > EXPLORE_DATA_CACHE_MAX_ITEMS
            or self._explore_layer_data_cache_bytes > EXPLORE_DATA_CACHE_MAX_BYTES
        ):
            _discarded_key, discarded = self._explore_layer_data_cache.popitem(
                last=False
            )
            self._explore_layer_data_cache_bytes -= int(discarded.nbytes)

    def _restore_cached_explore_layer(
        self,
        name: str,
        reload_descriptor: dict,
        layer_type: str,
        **display_settings,
    ):
        key = self._explore_data_cache_key(name, reload_descriptor)
        cached = self._explore_layer_data_cache.pop(key, None)
        if cached is None:
            return None
        self._explore_layer_data_cache_bytes -= int(cached.nbytes)
        layer = self._replace_explore_layer(
            name,
            cached,
            layer_type,
            reload_descriptor=reload_descriptor,
            **display_settings,
        )
        self._explore_cached_layer_count += 1
        return layer

    def load_roi(self, roi: str | None = None) -> None:
        if self.manifest is None:
            return
        roi = str(roi or self.roi_combo.currentText())
        if not roi:
            return
        mask_path = self._mask_path_for_roi(roi)
        full_mask = load_mask(mask_path)
        eligible = self._eligible_ids_for_roi(roi)
        self.current_roi = roi
        self.current_mask = full_mask
        self.current_mask_path = mask_path
        self.current_selected_object = None
        self.current_labeler_object = None
        self.selected_cell_label.setText("No cohort cell selected")
        self.labeler_selected_cell_label.setText("No cohort cell selected")
        self._remove_layers(
            [SELECTED_CELL_LAYER_NAME, LABELER_SELECTED_CELL_LAYER_NAME]
        )
        if self.labeler_roi_combo.findText(roi) >= 0:
            self.labeler_roi_combo.blockSignals(True)
            self.labeler_roi_combo.setCurrentText(roi)
            self.labeler_roi_combo.blockSignals(False)
        workflow_mode = self.current_workflow_mode()
        needs_cohort_layers = workflow_mode in {
            "classification",
            "cell_labeling",
            "full_workspace",
        }
        if needs_cohort_layers:
            restricted = cohort_mask(full_mask, eligible)
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
                MANAGED_LAYER_DEFAULT_VISIBILITY["excluded_segmentation_context"],
            )
            context_layer = self._replace_layer(
                "excluded_segmentation_context",
                context,
                "labels",
                visible=context_visible,
                opacity=self.explore_recipe.layer_opacities.get(
                    "excluded_segmentation_context",
                    MANAGED_LAYER_DEFAULT_OPACITY["excluded_segmentation_context"],
                ),
            )
            context_layer.visible = context_visible
        else:
            self._remove_layers(
                ["classification_cohort", "excluded_segmentation_context"]
            )
        self._remove_layers(
            [
                CLASS_LAYER_NAMES["confirmed"],
                CLASS_LAYER_NAMES["proposed"],
                CLASS_LAYER_NAMES["predicted"],
                CLASS_LAYER_NAMES["uncertainty"],
            ]
        )
        replay_view = bool(
            self.auto_reload_view_check.isChecked()
            and self.explore_recipe.has_content
        )
        if not replay_view:
            self._clear_explore_layers()
        if workflow_mode in {"classification", "full_workspace"}:
            self.refresh_classification_layers()
            self._refresh_noncontext_mask()
        else:
            self._remove_layers(
                [NONCONTEXT_MASK_LAYER_NAME, *CLASS_LAYER_NAMES.values()]
            )
        if workflow_mode in {"cell_labeling", "full_workspace"}:
            self.refresh_labeler_layers()
        else:
            self._remove_layers([LABELER_LAYER_NAME])
        self._refresh_roi_metadata_display()
        self.refresh_channel_list()
        self.refresh_population_qc_marker_choices()
        if replay_view:
            self.replay_explore_view()
        else:
            self._refresh_roi_review_colours()
        self._refresh_labeler_tally()
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

    def _refresh_noncontext_mask(self) -> None:
        """Mask pixels outside the cohort's recipe-defined staining context."""

        if self.manifest is None or self.current_mask is None or not self.current_roi:
            return
        eligible = self._eligible_ids_for_roi(self.current_roi)
        seen = classifier_seen_mask(
            self.current_mask,
            eligible,
            self.manifest.synthetic_features,
        )
        outside = (~seen).astype(np.uint8)
        layer = self._replace_layer(
            NONCONTEXT_MASK_LAYER_NAME,
            outside,
            "labels",
            colormap=self._direct_label_colormap({1: "#000000"}),
            visible=self.explore_recipe.layer_visibility.get(
                NONCONTEXT_MASK_LAYER_NAME,
                MANAGED_LAYER_DEFAULT_VISIBILITY[NONCONTEXT_MASK_LAYER_NAME],
            ),
            opacity=self.explore_recipe.layer_opacities.get(
                NONCONTEXT_MASK_LAYER_NAME,
                MANAGED_LAYER_DEFAULT_OPACITY[NONCONTEXT_MASK_LAYER_NAME],
            ),
        )
        if hasattr(layer, "contour"):
            layer.contour = self.explore_recipe.layer_contours.get(
                NONCONTEXT_MASK_LAYER_NAME,
                MANAGED_LAYER_DEFAULT_CONTOUR[NONCONTEXT_MASK_LAYER_NAME],
            )
        if hasattr(layer, "editable"):
            layer.editable = False
        self._raise_noncontext_mask()

    def _raise_noncontext_mask(self) -> None:
        """Keep the opaque focus mask above image and classification layers."""

        if NONCONTEXT_MASK_LAYER_NAME not in self.viewer.layers:
            return
        layer = self.viewer.layers[NONCONTEXT_MASK_LAYER_NAME]
        source = self.viewer.layers.index(layer)
        if source != len(self.viewer.layers) - 1:
            self.viewer.layers.move(source, len(self.viewer.layers))

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
            "cells; values above 0 show outlines that leave staining visible. "
            "The noncontext mask is an opaque black focus aid outside the "
            "staining regions used by the feature recipe."
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
            NONCONTEXT_MASK_LAYER_NAME,
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
        self._raise_noncontext_mask()
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
        active_tab = self.tabs.currentIndex()
        if (
            self.current_mask is None
            or active_tab not in {self.classify_tab_index, self.labeler_tab_index}
            or event.type != "mouse_press"
            or getattr(event, "button", 1) != 1
        ):
            return
        if active_tab == self.classify_tab_index and not self.cell_picking_enabled:
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
        object_id = int(self.current_mask[row, column])
        if active_tab == self.labeler_tab_index:
            self._handle_clicked_labeler_object(object_id)
        else:
            self._handle_clicked_object(object_id)

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
        elif behavior == "clear_proposed":
            self.clear_selected_proposed()

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
            self.set_status("Cell is outside this experiment; annotation was ignored.")
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

        selected = (self.current_mask == int(self.current_selected_object)).astype(
            np.uint8
        )
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

    def current_labeler_click_behavior(self) -> str:
        checked = self.labeler_click_behavior_group.checkedButton()
        if checked is None:
            return "select"
        return str(checked.property("napari_sbt_labeler_click_behavior") or "select")

    def _handle_clicked_labeler_object(self, object_id: int) -> None:
        if not self._select_labeler_object(object_id):
            return
        behavior = self.current_labeler_click_behavior()
        if behavior == "assign":
            self.assign_selected_labeler_cell()
        elif behavior == "clear":
            self.clear_selected_labeler_cell()

    def _select_labeler_object(self, object_id: int) -> bool:
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
            self.current_labeler_object = None
            self._remove_layers([LABELER_SELECTED_CELL_LAYER_NAME])
            self.labeler_selected_cell_label.setText("Cell is outside this experiment")
            self.set_status(
                "Cell is outside this experiment; Labeler action was ignored."
            )
            return False
        self.current_labeler_object = int(object_id)
        self.labeler_selected_cell_label.setText(
            f"{self.current_roi} / object {self.current_labeler_object}"
        )
        self._refresh_labeler_selected_cell_layer()
        return True

    def _refresh_labeler_selected_cell_layer(self) -> None:
        if self.current_mask is None or self.current_labeler_object is None:
            self._remove_layers([LABELER_SELECTED_CELL_LAYER_NAME])
            return
        selected = (self.current_mask == int(self.current_labeler_object)).astype(
            np.uint8
        )
        layer = self._replace_layer(
            LABELER_SELECTED_CELL_LAYER_NAME,
            selected,
            "labels",
            colormap=self._direct_label_colormap({1: "#ffffff"}),
            visible=self.explore_recipe.layer_visibility.get(
                LABELER_SELECTED_CELL_LAYER_NAME,
                MANAGED_LAYER_DEFAULT_VISIBILITY[LABELER_SELECTED_CELL_LAYER_NAME],
            ),
            opacity=self.explore_recipe.layer_opacities.get(
                LABELER_SELECTED_CELL_LAYER_NAME,
                MANAGED_LAYER_DEFAULT_OPACITY[LABELER_SELECTED_CELL_LAYER_NAME],
            ),
        )
        if hasattr(layer, "contour"):
            layer.contour = self.explore_recipe.layer_contours.get(
                LABELER_SELECTED_CELL_LAYER_NAME,
                MANAGED_LAYER_DEFAULT_CONTOUR[LABELER_SELECTED_CELL_LAYER_NAME],
            )
        self._bind_recipe_display_tracking(layer)

    def refresh_channel_list(self) -> None:
        self.channel_list.clear()
        self.current_image_paths = {}
        if self.manifest is None or not self.current_roi:
            self.image_coverage_label.setText("No experiment ROI is loaded.")
            return
        channel_aliases = self._channel_aliases()
        paths = self._image_paths_for_roi(self.current_roi)
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
                list_item.setToolTip(f"AnnData variable: {base_channel}\nImage: {path}")
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
            folders = self.manifest.images_folders + self.manifest.extra_images_folders
            suffix = (
                " Run Setup → Validate integrity to index flat image folders."
                if self._asset_index_signature is None
                else ""
            )
            self.image_coverage_label.setText(
                f"No images found for {self.current_roi} in {len(folders)} "
                f"configured folder(s).{suffix}"
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
            "layer_colormap_specs",
            "layer_visibility",
            "layer_opacities",
            "layer_contours",
            "layer_contrast_limits",
        ):
            payload[key] = {
                name: value
                for name, value in payload[key].items()
                if not (name.startswith("image::") or name == "population_qc_rgb")
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
        colormap = self._recipe_colormap(name, default_colormap)
        if colormap:
            settings["colormap"] = colormap
        contrast_limits = self.explore_recipe.layer_contrast_limits.get(name)
        if contrast_limits is None and name.startswith("image::"):
            contrast_limits = (
                float(self.display_lower_contrast_spin.value()),
                float(self.display_upper_contrast_spin.value()),
            )
        if contrast_limits is not None:
            settings["contrast_limits"] = contrast_limits
        return settings

    def _display_image_settings(self) -> DisplaySettings:
        return self._display_settings_from_controls(
            normalization_path=self.normalization_edit.text().strip() or None
        )

    def _display_normalization_value(self, channel: str) -> float | None:
        candidates = [str(channel), str(channel).split(" [", 1)[0]]
        path = self.current_image_paths.get(str(channel))
        if path is not None:
            candidates.append(path.stem)
        for candidate in candidates:
            value = find_normalization_value(
                self.display_normalization,
                candidate,
            )
            if value is not None:
                return float(value)
        return None

    def _display_image_load_kwargs(self, channel: str) -> dict[str, float | None]:
        settings = self._display_image_settings()
        return {
            "quantile": float(settings.fallback_quantile),
            "minimum_pixel_counts": float(settings.minimum_pixel_counts),
            "normalization_value": self._display_normalization_value(channel),
        }

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
                self._remove_explore_layer("population_qc_rgb")
                self.set_status(
                    "The RGB composite was not loaded because all three saved "
                    "channels are not available for this ROI."
                )
                return 0
            reload_descriptor = {
                "kind": "rgb",
                "channels": list(recipe.image_channels),
                "normalization": [
                    self._display_image_load_kwargs(channel) for channel in available
                ],
                "sources": [
                    self._image_source_identity(self.current_image_paths[channel])
                    for channel in available
                ],
            }
            display_settings = {
                "rgb": True,
                "blending": "translucent",
                **self._recipe_display_settings("population_qc_rgb"),
            }
            if (
                self._reuse_explore_layer(
                    "population_qc_rgb",
                    reload_descriptor,
                    **display_settings,
                )
                is not None
            ):
                return 1
            if (
                self._restore_cached_explore_layer(
                    "population_qc_rgb",
                    reload_descriptor,
                    "image",
                    **display_settings,
                )
                is not None
            ):
                return 1
            images = []
            for channel in available:
                image, is_rgb = load_display_image(
                    self.current_image_paths[channel],
                    **self._display_image_load_kwargs(channel),
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
                reload_descriptor=reload_descriptor,
                **display_settings,
            )
            return 1
        loaded = 0
        for channel in available:
            name = f"image::{channel}"
            recipe_index = recipe.image_channels.index(channel)
            default_colormap = (
                "gray"
                if recipe.image_mode == "grayscale"
                else SIX_COLOUR_COLORMAPS[recipe_index % len(SIX_COLOUR_COLORMAPS)]
            )
            # The saved image mode controls display colour, not pixel data. A
            # greyscale/six-colour recipe switch can therefore reuse the same
            # source array after only checking the file identity.
            reload_descriptor = {
                "kind": "image",
                "channel": channel,
                "mode": recipe.image_mode,
                "normalization": self._display_image_load_kwargs(channel),
                "source": self._image_source_identity(
                    self.current_image_paths[channel]
                ),
            }
            existing_layer = (
                self.viewer.layers[name] if name in self.viewer.layers else None
            )
            existing_is_rgb = bool(
                existing_layer is not None and getattr(existing_layer, "rgb", False)
            )
            display_settings = self._recipe_display_settings(
                name,
                default_colormap=None if existing_is_rgb else default_colormap,
            )
            if (
                self._reuse_explore_layer(
                    name,
                    reload_descriptor,
                    **display_settings,
                )
                is not None
            ):
                loaded += 1
                continue
            if (
                self._restore_cached_explore_layer(
                    name,
                    reload_descriptor,
                    "image",
                    **display_settings,
                )
                is not None
            ):
                loaded += 1
                continue
            image, is_rgb = load_display_image(
                self.current_image_paths[channel],
                **self._display_image_load_kwargs(channel),
            )
            kwargs = {
                "rgb": is_rgb,
                "blending": "translucent" if is_rgb else "additive",
                **(self._recipe_display_settings(name) if is_rgb else display_settings),
            }
            self._replace_explore_layer(
                name,
                image,
                "image",
                reload_descriptor=reload_descriptor,
                **kwargs,
            )
            loaded += 1
        for channel in missing:
            self._remove_explore_layer(f"image::{channel}")
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
            update={
                "observation_overlay": observation,
                "observation_overlay_full_dataset": (
                    self.overlay_full_dataset_check.isChecked()
                ),
            },
            deep=True,
        )
        self.replay_explore_view()

    def _roi_adata_rows(self):
        if self.adata is None or self.current_mask is None or self.manifest is None:
            raise RuntimeError("Load an experiment ROI and AnnData first.")
        if not self._adata_roi_positions:
            groups = self.adata.obs.groupby(
                self.adata.obs[self.manifest.roi_obs].astype(str),
                sort=False,
                observed=True,
            ).indices
            self._adata_roi_positions = {
                str(roi): np.asarray(positions, dtype=np.int64)
                for roi, positions in groups.items()
            }
        roi_positions = self._adata_roi_positions.get(
            str(self.current_roi), np.empty(0, dtype=np.int64)
        )
        rows = self.adata.obs.iloc[roi_positions]
        object_ids = pd.to_numeric(
            rows[self.manifest.object_id_obs], errors="coerce"
        ).astype("Int64")
        eligible = self._eligible_ids_for_roi(str(self.current_roi))
        selected = object_ids.notna() & object_ids.isin(eligible)
        return rows, object_ids, selected, roi_positions

    def _eligible_ids_for_roi(self, roi: str) -> set[int]:
        if not self._cohort_ids_by_roi and not self.cohort.empty:
            self._cohort_ids_by_roi = {
                str(group_roi): set(group["ObjectNumber"].astype(int))
                for group_roi, group in self.cohort.groupby("ROI", observed=True)
            }
        return self._cohort_ids_by_roi.get(str(roi), set())

    def _direct_label_colormap(self, colours: dict[int, str]):
        from napari.utils.colormaps import DirectLabelColormap

        return DirectLabelColormap(color_dict={None: "#00000000", **colours})

    def _render_observation_overlay(self, observation: str) -> int:
        name = f"obs::{observation}"
        if self.adata is None or observation not in self.adata.obs:
            self._remove_explore_layer(name)
            self.set_status(
                f"Saved AnnData observation {observation!r} is no longer available."
            )
            return 0
        numeric = pd.api.types.is_numeric_dtype(self.adata.obs[observation])
        reload_descriptor = {
            "kind": "observation",
            "observation": observation,
            "full_dataset": self.explore_recipe.observation_overlay_full_dataset,
            "value_kind": "numeric" if numeric else "categorical",
        }
        if numeric:
            display_settings = {
                "blending": "additive",
                **self._recipe_display_settings(
                    name,
                    default_colormap="viridis",
                ),
            }
            if (
                self._reuse_explore_layer(
                    name,
                    reload_descriptor,
                    **display_settings,
                )
                is not None
            ):
                return 1
            if (
                self._restore_cached_explore_layer(
                    name,
                    reload_descriptor,
                    "image",
                    **display_settings,
                )
                is not None
            ):
                return 1
        else:
            population_colours = categorical_colour_map(self.adata, observation)
            # Use dataset-wide category order so saved direct-label colours keep
            # the same biological meaning even when an ROI lacks categories.
            categories = list(population_colours)
            codes = {value: index + 1 for index, value in enumerate(categories)}
            default_colormap = self._direct_label_colormap(
                {code: population_colours[value] for value, code in codes.items()}
            )
            display_settings = {
                "colormap": self._recipe_colormap(name, default_colormap),
                "visible": self.explore_recipe.layer_visibility.get(name, True),
                "opacity": self.explore_recipe.layer_opacities.get(name, 1.0),
            }
            layer = self._reuse_explore_layer(
                name,
                reload_descriptor,
                **display_settings,
            )
            if layer is None:
                layer = self._restore_cached_explore_layer(
                    name,
                    reload_descriptor,
                    "labels",
                    **display_settings,
                )
            if layer is not None:
                self._set_label_contour_from_recipe(layer, name)
                return 1

        rows, object_ids, selected, _roi_selector = self._roi_adata_rows()
        if self.explore_recipe.observation_overlay_full_dataset:
            selected = object_ids.notna()
        values = rows[observation]
        if numeric:
            mapping = pd.Series(
                pd.to_numeric(values[selected], errors="coerce").to_numpy(),
                index=object_ids[selected].astype(int),
            )
            overlay = identity_value_map(self.current_mask, mapping)
            self._replace_explore_layer(
                name,
                overlay,
                "image",
                reload_descriptor=reload_descriptor,
                **display_settings,
            )
        else:
            mapping = pd.Series(
                values[selected].astype(str).map(codes).to_numpy(),
                index=object_ids[selected].astype(int),
            )
            overlay = identity_value_map(self.current_mask, mapping, dtype=np.int32)
            layer = self._replace_explore_layer(
                name,
                overlay,
                "labels",
                reload_descriptor=reload_descriptor,
                **display_settings,
            )
            self._set_label_contour_from_recipe(layer, name)
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
        populations = list(populations)
        if self.adata is None or observation not in self.adata.obs:
            for population in populations:
                self._remove_explore_layer(f"population::{observation}::{population}")
            self.set_status(
                f"Saved population observation {observation!r} is unavailable."
            )
            return 0
        colour_map = categorical_colour_map(self.adata, observation)
        roi_data = None
        loaded = 0
        for population in populations:
            colour = colour_map.get(str(population), "#ffffff")
            name = f"population::{observation}::{population}"
            reload_descriptor = {
                "kind": "population",
                "observation": observation,
                "population": population,
            }
            display_settings = {
                "colormap": self._recipe_colormap(
                    name,
                    self._direct_label_colormap({1: colour}),
                ),
                "visible": self.explore_recipe.layer_visibility.get(name, True),
                "opacity": self.explore_recipe.layer_opacities.get(name, 1.0),
            }
            layer = self._reuse_explore_layer(
                name,
                reload_descriptor,
                **display_settings,
            )
            if layer is None:
                layer = self._restore_cached_explore_layer(
                    name,
                    reload_descriptor,
                    "labels",
                    **display_settings,
                )
            if layer is not None:
                self._set_label_contour_from_recipe(layer, name)
                loaded += 1
                continue
            if roi_data is None:
                rows, object_ids, selected, _roi_selector = self._roi_adata_rows()
                values = rows[observation].astype(str)
                roi_data = (object_ids, selected, values)
            object_ids, selected, values = roi_data
            population_selected = selected & values.eq(str(population))
            mapping = pd.Series(
                np.ones(int(population_selected.sum()), dtype=np.int32),
                index=object_ids[population_selected].astype(int),
            )
            overlay = identity_value_map(
                self.current_mask,
                mapping,
                dtype=np.int32,
            )
            layer = self._replace_explore_layer(
                name,
                overlay,
                "labels",
                reload_descriptor=reload_descriptor,
                **display_settings,
            )
            self._set_label_contour_from_recipe(layer, name)
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
        loaded = 0
        available_markers = set(self.adata.var_names.astype(str))
        roi_data = None
        for marker in markers:
            name = f"adata.X::{marker}"
            if str(marker) not in available_markers:
                self._remove_explore_layer(name)
                self.set_status(
                    f"Saved AnnData marker {marker!r} is no longer available."
                )
                continue
            reload_descriptor = {
                "kind": "marker",
                "marker": marker,
            }
            display_settings = {
                "blending": "additive",
                **self._recipe_display_settings(
                    name,
                    default_colormap="viridis",
                ),
            }
            if (
                self._reuse_explore_layer(
                    name,
                    reload_descriptor,
                    **display_settings,
                )
                is not None
            ):
                loaded += 1
                continue
            if (
                self._restore_cached_explore_layer(
                    name,
                    reload_descriptor,
                    "image",
                    **display_settings,
                )
                is not None
            ):
                loaded += 1
                continue
            if roi_data is None:
                _rows, object_ids, selected, roi_selector = self._roi_adata_rows()
                roi_data = (object_ids, selected, roi_selector)
            object_ids, selected, roi_selector = roi_data
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
            overlay = identity_value_map(self.current_mask, mapping)
            self._replace_explore_layer(
                name,
                overlay,
                "image",
                reload_descriptor=reload_descriptor,
                **display_settings,
            )
            loaded += 1
        return loaded

    def replay_explore_view(self) -> None:
        """Render the active ROI-independent recipe and record this review."""

        if not self.current_roi or self.current_mask is None:
            return
        recipe_entries = self._recipe_layer_entries()
        desired_names = {
            entry["name"] for entry in recipe_entries if entry["kind"] != "managed"
        }
        for name in self._explore_layer_names - desired_names:
            self._remove_explore_layer(name)
        self._explore_reused_layer_count = 0
        self._explore_cached_layer_count = 0
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
            loaded += self._render_marker_overlays(self.explore_recipe.marker_overlays)
        self._apply_managed_layer_display_settings(refresh_recipe_list=False)
        self._raise_noncontext_mask()
        managed_present = sum(
            name in self.viewer.layers for name in MANAGED_RECIPE_LAYERS
        )
        self._refresh_reload_recipe_list()
        if loaded or managed_present:
            self._mark_current_explore_viewed()
            active_preset = self.explore_review_state.recipe_presets.get(
                self.explore_review_state.active_recipe_id
            )
            recipe_label = "working Explore recipe"
            if (
                active_preset is not None
                and active_preset.recipe.fingerprint == self.explore_recipe.fingerprint
            ):
                recipe_label = f"Explore recipe {active_preset.name!r}"
            rendered = max(
                0,
                loaded
                - self._explore_reused_layer_count
                - self._explore_cached_layer_count,
            )
            self.set_status(
                f"Applied {recipe_label} for ROI {self.current_roi}: reused "
                f"{self._explore_reused_layer_count} existing Explore layer(s), "
                f"restored {self._explore_cached_layer_count} from the cross-ROI "
                f"memory cache, loaded or recalculated {rendered}, and updated "
                f"{managed_present} managed classification layer setting(s)."
            )
        else:
            self._refresh_roi_review_colours()

    def _population_qc_selection(self) -> tuple[str, str]:
        observation = self.population_qc_obs_combo.currentText().strip()
        population = self.population_qc_population_combo.currentText().strip()
        if not observation or not population:
            raise ValueError("Choose a Population QC observation and population first.")
        return observation, population

    def refresh_population_qc_populations(self) -> None:
        """Refresh the populations offered by the dedicated QC tab."""

        previous = self.population_qc_population_combo.currentText()
        observation = self.population_qc_obs_combo.currentText().strip()
        values: list[str] = []
        if self.adata is not None and observation in self.adata.obs:
            series = self.adata.obs[observation]
            if isinstance(series.dtype, pd.CategoricalDtype):
                values = [str(value) for value in series.cat.categories]
            else:
                values = sorted(series.dropna().astype(str).unique().tolist())
        self.population_qc_population_combo.blockSignals(True)
        self.population_qc_population_combo.clear()
        self.population_qc_population_combo.addItems(values)
        if previous in values:
            self.population_qc_population_combo.setCurrentText(previous)
        self.population_qc_population_combo.blockSignals(False)
        self.load_population_qc_recipe_controls()

    def refresh_population_qc_marker_choices(self) -> None:
        """Populate RGB marker selectors from the images available for this ROI."""

        channels = list(self.current_image_paths)
        for combo in self.population_qc_marker_combos.values():
            previous = combo.currentText().strip()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("", None)
            combo.addItems(channels)
            if previous:
                index = combo.findText(previous)
                if index < 0:
                    combo.addItem(previous)
                    index = combo.count() - 1
                    combo.setItemData(
                        index,
                        self.QColor("#9a3412"),
                        self.Qt.ForegroundRole,
                    )
                    combo.setItemData(
                        index,
                        "Saved marker is absent from the current ROI; it remains in "
                        "the recipe and will be retried on other ROIs.",
                        self.Qt.ToolTipRole,
                    )
                combo.setCurrentIndex(index)
            combo.blockSignals(False)

    def _population_qc_marker_candidates(self) -> list[tuple[str, str]]:
        if self.adata is None:
            return []
        aliases = build_image_channel_aliases(self.adata.var_names, self.adata.var)
        candidates: list[tuple[str, str]] = []
        available_paths = dict(self.current_image_paths)
        for roi_paths in self._roi_image_path_index.values():
            for display_name, path in roi_paths.items():
                available_paths.setdefault(display_name, path)
        for display_name, path in available_paths.items():
            names = (
                str(display_name),
                str(display_name).split(" [", 1)[0],
                Path(path).stem,
            )
            canonical = None
            for name in names:
                key = "".join(
                    character for character in name if character.isalnum()
                ).casefold()
                canonical = aliases.get(key)
                if canonical is not None:
                    break
            if canonical is not None:
                candidates.append((str(display_name), str(canonical)))
        return list(dict.fromkeys(candidates))

    def _population_qc_adata_view(self):
        """Return the AnnData rows that can actually appear in this workspace."""

        if self.adata is None or self.manifest is None or self.cohort.empty:
            return self.adata
        if (
            self._population_qc_cohort_selector is not None
            and len(self._population_qc_cohort_selector) == self.adata.n_obs
        ):
            return self.adata[self._population_qc_cohort_selector]
        if (
            self.manifest.cell_scope.mode == "all_cells"
            and len(self.cohort) == self.adata.n_obs
        ):
            self._population_qc_cohort_selector = np.ones(
                self.adata.n_obs, dtype=bool
            )
            return self.adata
        eligible = pd.MultiIndex.from_arrays(
            [
                self.cohort["ROI"].astype(str),
                pd.to_numeric(
                    self.cohort["ObjectNumber"], errors="coerce"
                ).fillna(-1).astype(np.int64),
            ],
            names=["ROI", "ObjectNumber"],
        )
        available = pd.MultiIndex.from_arrays(
            [
                self.adata.obs[self.manifest.roi_obs].astype(str),
                pd.to_numeric(
                    self.adata.obs[self.manifest.object_id_obs], errors="coerce"
                ).fillna(-1).astype(np.int64),
            ],
            names=["ROI", "ObjectNumber"],
        )
        self._population_qc_cohort_selector = np.asarray(
            available.isin(eligible), dtype=bool
        )
        return self.adata[self._population_qc_cohort_selector]

    def _cached_population_qc_marker_suggestions(
        self,
        observation: str,
        population: str,
    ) -> list[str]:
        candidates = self._population_qc_marker_candidates()
        key = (
            str(observation),
            str(population),
            tuple(candidates),
        )
        if key not in self._population_qc_marker_cache:
            self._population_qc_marker_cache[key] = top_population_markers(
                self._population_qc_adata_view(),
                observation=observation,
                population=population,
                candidates=candidates,
                top_n=3,
            )
        return list(self._population_qc_marker_cache[key])

    def _apply_population_qc_marker_suggestions(
        self, suggestions: Iterable[str]
    ) -> None:
        suggestions = list(suggestions)
        lower, upper = self._display_settings_from_controls().default_contrast_limits
        for index, colour in enumerate(("red", "green", "blue")):
            combo = self.population_qc_marker_combos[colour]
            value = suggestions[index] if index < len(suggestions) else ""
            match = combo.findText(value)
            combo.setCurrentIndex(max(0, match))
            self.population_qc_lower_spins[colour].setValue(lower)
            self.population_qc_upper_spins[colour].setValue(upper)

    def suggest_population_qc_markers(self) -> None:
        observation, population = self._population_qc_selection()
        suggestions = self._cached_population_qc_marker_suggestions(
            observation, population
        )
        if not suggestions:
            raise ValueError(
                "No current ROI image channels could be matched safely to adata.var. "
                "Load an ROI, then choose the RGB channels manually if necessary."
            )
        self._apply_population_qc_marker_suggestions(suggestions)
        self.population_qc_status_label.setText(
            f"Suggested the highest-mean matched image markers for {population!r}. "
            "Review the colours and contrast limits before loading the view."
        )

    def _population_qc_recipe_from_controls(self) -> ExploreViewRecipe:
        observation, population = self._population_qc_selection()
        channels: list[str] = []
        limits: list[tuple[float, float]] = []
        for colour in ("red", "green", "blue"):
            channel = self.population_qc_marker_combos[colour].currentText().strip()
            if not channel:
                continue
            channels.append(channel)
            limits.append(
                (
                    float(self.population_qc_lower_spins[colour].value()),
                    float(self.population_qc_upper_spins[colour].value()),
                )
            )
        return build_population_qc_recipe(
            observation=observation,
            population=population,
            channels=channels,
            contrast_limits=limits,
            contour_width=int(self.population_qc_contour_spin.value()),
        )

    @staticmethod
    def _population_qc_recipe_for_storage(
        recipe: ExploreViewRecipe,
    ) -> ExploreViewRecipe:
        """Remove the workspace-wide outline preference from a population recipe."""

        payload = recipe.model_dump(mode="python")
        payload["layer_contours"] = {
            name: value
            for name, value in payload.get("layer_contours", {}).items()
            if not str(name).startswith("population::")
        }
        return ExploreViewRecipe.model_validate(payload)

    def _population_recipe_with_global_outline(
        self,
        recipe: ExploreViewRecipe,
    ) -> ExploreViewRecipe:
        """Apply the workspace-wide outline to a stored population view."""

        if not recipe.population_observation or not recipe.populations:
            return recipe.model_copy(deep=True)
        payload = recipe.model_dump(mode="python")
        contours = dict(payload.get("layer_contours", {}))
        width = int(self.explore_review_state.population_qc_contour_width)
        for population in recipe.populations:
            contours[
                f"population::{recipe.population_observation}::{population}"
            ] = width
        payload["layer_contours"] = contours
        return ExploreViewRecipe.model_validate(payload)

    def set_population_qc_contour_width(self, value: int) -> None:
        """Persist and apply one outline width across all Population QC views."""

        width = int(value)
        if not 0 <= width <= 20:
            raise ValueError("Population QC outline width must be between 0 and 20 px.")
        self.explore_review_state.population_qc_contour_width = width
        self.explore_review_state.population_recipes = {
            key: self._population_qc_recipe_for_storage(recipe)
            for key, recipe in self.explore_review_state.population_recipes.items()
        }
        if self.paths is not None:
            self._save_explore_review_state()

        observation = self.population_qc_obs_combo.currentText().strip()
        population = self.population_qc_population_combo.currentText().strip()
        layer_name = (
            f"population::{observation}::{population}"
            if observation and population
            else ""
        )
        if (
            layer_name
            and self.explore_review_state.active_recipe_id is None
            and self.explore_recipe.population_observation == observation
            and self.explore_recipe.populations == [population]
        ):
            contours = dict(self.explore_recipe.layer_contours)
            contours[layer_name] = width
            self.explore_recipe = self.explore_recipe.model_copy(
                update={"layer_contours": contours},
                deep=True,
            )
            if layer_name in self.viewer.layers:
                layer = self.viewer.layers[layer_name]
                if hasattr(layer, "contour"):
                    previous_state = self._updating_recipe_layer_state
                    self._updating_recipe_layer_state = True
                    try:
                        layer.contour = width
                    finally:
                        self._updating_recipe_layer_state = previous_state
        self.refresh_population_qc_rois()
        self.set_status(
            f"Population QC outline width is now {width} px for every population."
        )

    def _store_population_qc_recipe(
        self,
        recipe: ExploreViewRecipe,
        *,
        action: str = "save_population_qc_recipe",
        persist: bool = True,
        audit: bool = True,
    ) -> None:
        if self.paths is None:
            raise ValueError(
                "Create or load a workflow workspace before saving recipes."
            )
        observation, population = self._population_qc_selection()
        key = population_recipe_key(observation, population)
        self.explore_review_state.population_recipes[key] = (
            self._population_qc_recipe_for_storage(recipe)
        )
        if persist:
            self._save_explore_review_state()
        if audit:
            append_audit(
                self.paths,
                {
                    "action": action,
                    "population_observation": observation,
                    "population": population,
                    "view_fingerprint": recipe.fingerprint,
                },
            )

    def _append_population_qc_audit(
        self, recipe: ExploreViewRecipe, action: str
    ) -> None:
        observation, population = self._population_qc_selection()
        append_audit(
            self.paths,
            {
                "action": action,
                "population_observation": observation,
                "population": population,
                "view_fingerprint": recipe.fingerprint,
            },
        )

    def save_population_qc_recipe(self) -> None:
        recipe = self._population_qc_recipe_from_controls()
        self._store_population_qc_recipe(recipe)
        self.refresh_population_qc_rois()
        observation, population = self._population_qc_selection()
        self.set_status(
            f"Saved the RGB channels, colours, and contrast ranges for "
            f"{observation}={population}. The {self.population_qc_contour_spin.value()} "
            "px outline is shared by every population."
        )

    def _set_population_qc_combo_value(self, combo, value: str) -> None:
        index = combo.findText(str(value))
        if index < 0:
            combo.addItem(str(value))
            index = combo.count() - 1
            combo.setItemData(index, self.QColor("#9a3412"), self.Qt.ForegroundRole)
            combo.setItemData(
                index,
                "Saved channel is not available in the current ROI.",
                self.Qt.ToolTipRole,
            )
        combo.setCurrentIndex(index)

    def load_population_qc_recipe_controls(self) -> None:
        """Restore a population's compact RGB controls from the shared recipe store."""

        self._update_population_qc_contrast_defaults_label()
        observation = self.population_qc_obs_combo.currentText().strip()
        population = self.population_qc_population_combo.currentText().strip()
        if not observation or not population:
            self.refresh_population_qc_rois()
            return
        self.refresh_population_qc_marker_choices()
        recipe = self.explore_review_state.population_recipes.get(
            population_recipe_key(observation, population)
        )
        lower_default, upper_default = (
            self._display_settings_from_controls().default_contrast_limits
        )
        for index, colour in enumerate(("red", "green", "blue")):
            channel = (
                recipe.image_channels[index]
                if recipe is not None and index < len(recipe.image_channels)
                else ""
            )
            self._set_population_qc_combo_value(
                self.population_qc_marker_combos[colour], channel
            )
            layer_name = f"image::{channel}" if channel else ""
            limits = (
                recipe.layer_contrast_limits.get(layer_name)
                if recipe is not None and layer_name
                else None
            ) or (lower_default, upper_default)
            self.population_qc_lower_spins[colour].setValue(float(limits[0]))
            self.population_qc_upper_spins[colour].setValue(float(limits[1]))
        if recipe is None:
            suggestions = self._cached_population_qc_marker_suggestions(
                observation, population
            )
            if suggestions:
                self._apply_population_qc_marker_suggestions(suggestions)
                self.population_qc_status_label.setText(
                    "No saved RGB recipe exists, so the cached top three matched "
                    "markers were selected automatically. Review them before "
                    "loading the population view."
                )
            else:
                self.population_qc_status_label.setText(
                    "No saved RGB recipe or safely matched image markers are "
                    "available. Choose channels manually, then save or load the view."
                )
        else:
            self.population_qc_status_label.setText(
                "Restored this population's saved RGB channels, colours, and "
                "contrast ranges. The outline width remains the workspace-wide "
                "setting shown above; missing current-ROI channels remain selectable."
            )
        self.refresh_population_qc_rois()

    def load_population_qc_view(self) -> None:
        recipe = self._population_qc_recipe_from_controls()
        self._store_population_qc_recipe(recipe, persist=False, audit=False)
        self.explore_review_state.active_recipe_id = None
        self._apply_explore_recipe(
            self._population_recipe_with_global_outline(recipe)
        )
        if not self.current_roi:
            self._save_explore_review_state()
        self._append_population_qc_audit(recipe, "load_population_qc_view")
        self.refresh_population_qc_rois()

    def _population_qc_eligible_rois(self) -> list[str]:
        if self.manifest is None or self.cohort.empty:
            return []
        rois = sorted(self.cohort["ROI"].astype(str).unique().tolist())
        if (
            self.manifest.experiment_mode == "feature_discovery_trial"
            and self.manifest.feature_trial is not None
        ):
            selected = set(self.manifest.feature_trial.selected_rois)
            rois = [roi for roi in rois if roi in selected]
        return rois

    def _clear_population_qc_roi_buttons(self) -> None:
        while self.population_qc_roi_buttons_layout.count():
            item = self.population_qc_roi_buttons_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.population_qc_roi_buttons = {}

    def refresh_population_qc_rois(self) -> None:
        """Recalculate live top, bottom, or random ROI sampling buttons."""

        if not hasattr(self, "population_qc_roi_buttons_layout"):
            return
        self._clear_population_qc_roi_buttons()
        observation = self.population_qc_obs_combo.currentText().strip()
        population = self.population_qc_population_combo.currentText().strip()
        if (
            self.adata is None
            or not observation
            or not population
            or self.manifest is None
        ):
            self.population_qc_status_label.setText(
                "Load a workflow workspace and choose a population to rank ROIs."
            )
            return
        eligible_rois = self._population_qc_eligible_rois()
        ranking_key = (
            observation,
            population,
            tuple(eligible_rois),
            str(self.population_qc_roi_order_combo.currentData()),
            int(self.population_qc_roi_limit_spin.value()),
            int(self.population_qc_random_seed_spin.value()),
        )
        ranking = self._population_qc_ranking_cache.get(ranking_key)
        if ranking is None:
            ranking = rank_population_rois(
                self._population_qc_adata_view(),
                observation=observation,
                population=population,
                roi_obs=self.manifest.roi_obs,
                eligible_rois=eligible_rois,
                ordering=str(self.population_qc_roi_order_combo.currentData()),
                limit=int(self.population_qc_roi_limit_spin.value()),
                random_seed=int(self.population_qc_random_seed_spin.value()),
            )
            self._population_qc_ranking_cache[ranking_key] = list(ranking)
        try:
            fingerprint = self._population_qc_recipe_from_controls().fingerprint
        except ValueError:
            fingerprint = ""
        viewed = set(self.explore_review_state.viewed_rois.get(fingerprint, []))
        from qtpy.QtWidgets import QPushButton

        for index, (roi, count) in enumerate(ranking):
            button = QPushButton(f"{roi} ({count:,})")
            is_viewed = roi in viewed
            button.setStyleSheet(
                "QPushButton { background-color: "
                + ("#d1d5db" if is_viewed else "#bbf7d0")
                + "; color: #111827; font-weight: 600; padding: 5px; }"
            )
            button.setToolTip(
                ("Viewed" if is_viewed else "Not yet viewed")
                + f" with this exact RGB/contrast/outline recipe; {count:,} "
                "matching cells."
            )
            button.clicked.connect(
                lambda _checked=False,
                selected_roi=roi: self.activate_population_qc_roi(selected_roi)
            )
            self.population_qc_roi_buttons_layout.addWidget(
                button, index // 3, index % 3
            )
            self.population_qc_roi_buttons[roi] = button
        ordering_text = self.population_qc_roi_order_combo.currentText().lower()
        self.population_qc_status_label.setText(
            f"Showing {len(ranking):,} {ordering_text} eligible ROIs for "
            f"{observation}={population}. Green is unvisited; grey is viewed with "
            "this exact RGB, contrast, and outline recipe."
        )

    def recalculate_population_qc_rois(self) -> None:
        """Explicitly invalidate cached abundance rankings and rebuild the list."""

        self._population_qc_ranking_cache.clear()
        self.refresh_population_qc_rois()

    def activate_population_qc_roi(self, roi: str) -> None:
        """Load one ranked ROI with the current Population QC recipe exactly once."""

        recipe = self._population_qc_recipe_from_controls()
        self._store_population_qc_recipe(recipe, persist=False, audit=False)
        self.explore_review_state.active_recipe_id = None
        self._apply_explore_recipe(recipe, replay=False)
        index = self.roi_combo.findText(str(roi))
        if index < 0:
            raise ValueError(f"ROI {roi!r} is outside the current workflow scope.")
        was_blocked = self.roi_combo.blockSignals(True)
        self.roi_combo.setCurrentIndex(index)
        self.roi_combo.blockSignals(was_blocked)
        self.load_roi(str(roi))
        if not self.auto_reload_view_check.isChecked():
            self.replay_explore_view()
        self._append_population_qc_audit(recipe, "open_population_qc_roi")
        button = self.population_qc_roi_buttons.get(str(roi))
        if button is not None:
            button.setStyleSheet(
                "QPushButton { background-color: #d1d5db; color: #111827; "
                "font-weight: 600; padding: 5px; }"
            )
            button.setToolTip(
                "Viewed with this exact RGB/contrast/outline recipe; click to revisit."
            )
        self.population_qc_status_label.setText(
            f"Loaded ROI {roi!r} with the cached Population QC recipe and marked "
            "it as viewed."
        )

    def import_population_qc_settings_csv(self) -> None:
        """Import the legacy one-row-per-population RGB settings format."""

        if self.paths is None:
            raise ValueError(
                "Create or load a workflow workspace before importing settings."
            )
        observation = self.population_qc_obs_combo.currentText().strip()
        if not observation:
            raise ValueError(
                "Choose the target population observation before importing."
            )
        selected, _filter = self.QFileDialog.getOpenFileName(
            self.root,
            "Import Population QC settings",
            str(self.paths.root),
            "CSV files (*.csv)",
        )
        if not selected:
            return
        frame = pd.read_csv(selected)
        if "Population" in frame.columns:
            populations = frame.pop("Population").astype(str)
        elif "population" in frame.columns:
            populations = frame.pop("population").astype(str)
        else:
            frame = pd.read_csv(selected, index_col=0)
            populations = pd.Series(frame.index.astype(str), index=frame.index)
        lower_default, upper_default = (
            self._display_settings_from_controls().default_contrast_limits
        )
        imported = 0
        for row_number, (_index, row) in enumerate(frame.iterrows()):
            population = str(populations.iloc[row_number]).strip()
            channels: list[str] = []
            limits: list[tuple[float, float]] = []
            for display_name in ("Red", "Green", "Blue"):
                channel = str(row.get(display_name, "")).strip()
                if not channel or channel.casefold() == "nan":
                    continue
                channels.append(channel)
                limits.append(
                    (
                        parse_legacy_contrast(
                            row.get(f"{display_name}_min"), fallback=lower_default
                        ),
                        parse_legacy_contrast(
                            row.get(f"{display_name}_max"), fallback=upper_default
                        ),
                    )
                )
            if not population or not channels:
                continue
            recipe = build_population_qc_recipe(
                observation=observation,
                population=population,
                channels=channels,
                contrast_limits=limits,
                contour_width=self.explore_review_state.population_qc_contour_width,
            )
            self.explore_review_state.population_recipes[
                population_recipe_key(observation, population)
            ] = self._population_qc_recipe_for_storage(recipe)
            imported += 1
        self._save_explore_review_state()
        append_audit(
            self.paths,
            {
                "action": "import_population_qc_settings",
                "observation": observation,
                "source": str(Path(selected).resolve(strict=False)),
                "population_count": imported,
            },
        )
        self.load_population_qc_recipe_controls()
        self.set_status(
            f"Imported Population QC RGB settings for {imported:,} populations."
        )

    def export_population_qc_settings_csv(self) -> None:
        """Export saved Population QC recipes in a legacy-readable CSV layout."""

        if self.paths is None:
            raise ValueError(
                "Create or load a workflow workspace before exporting settings."
            )
        observation = self.population_qc_obs_combo.currentText().strip()
        if not observation:
            raise ValueError("Choose the population observation to export.")
        rows: list[dict[str, object]] = []
        for key, recipe in self.explore_review_state.population_recipes.items():
            try:
                recipe_observation, population = json.loads(key)
            except (TypeError, ValueError):
                continue
            if str(recipe_observation) != observation:
                continue
            row: dict[str, object] = {"Population": str(population)}
            for index, display_name in enumerate(("Red", "Green", "Blue")):
                channel = (
                    recipe.image_channels[index]
                    if index < len(recipe.image_channels)
                    else ""
                )
                row[display_name] = channel
                limits = recipe.layer_contrast_limits.get(f"image::{channel}")
                row[f"{display_name}_min"] = limits[0] if limits else ""
                row[f"{display_name}_max"] = limits[1] if limits else ""
            rows.append(row)
        if not rows:
            raise ValueError(
                f"No saved Population QC recipes exist for {observation!r}."
            )
        selected, _filter = self.QFileDialog.getSaveFileName(
            self.root,
            "Export Population QC settings",
            str(
                self.paths.root
                / "explore"
                / f"{slugify(observation)}_population_qc.csv"
            ),
            "CSV files (*.csv)",
        )
        if not selected:
            return
        destination = Path(selected)
        if destination.suffix.lower() != ".csv":
            destination = destination.with_suffix(".csv")
        frame = pd.DataFrame(rows)
        frame = frame.reindex(columns=["Population", *POPULATION_QC_SETTINGS_COLUMNS])
        frame = frame.sort_values("Population", kind="stable").reset_index(drop=True)
        write_dataframe(destination, frame)
        self.set_status(
            f"Exported {len(frame):,} Population QC recipes to {destination}."
        )

    def rank_rois_by_population(self) -> None:
        observation = self.population_obs_combo.currentText()
        value = self.population_value_combo.currentText()
        subset = self.adata.obs.loc[self.adata.obs[observation].astype(str).eq(value)]
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
        workflow_index = self.workflow_combo.findData("classification")
        if workflow_index >= 0:
            self.workflow_combo.setCurrentIndex(workflow_index)
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
            "and click Create workspace and start to freeze it."
        )

    def _labeler_code_map(self) -> dict[str, int]:
        return {
            definition.label_id: index + 1
            for index, definition in enumerate(self.labeler_classes)
        }

    def _labeler_colormap(self):
        return self._direct_label_colormap(
            {
                index + 1: definition.color
                for index, definition in enumerate(self.labeler_classes)
            }
        )

    def refresh_labeler_layers(self) -> None:
        """Render current-ROI Labeler assignments as coloured cell outlines."""

        if self.current_mask is None or self.manifest is None:
            return
        codes = self._labeler_code_map()
        rows = self.labeler_records.loc[
            self.labeler_records["ROI"].astype(str).eq(str(self.current_roi))
        ]
        mapping = pd.Series(
            rows["label_id"].map(codes).to_numpy(),
            index=pd.to_numeric(rows["ObjectNumber"], errors="coerce")
            .dropna()
            .astype(int),
            dtype="float64",
        )
        data = identity_value_map(self.current_mask, mapping, dtype=np.int32)
        layer = self._replace_layer(
            LABELER_LAYER_NAME,
            data,
            "labels",
            colormap=self._labeler_colormap(),
            visible=self.explore_recipe.layer_visibility.get(
                LABELER_LAYER_NAME,
                MANAGED_LAYER_DEFAULT_VISIBILITY[LABELER_LAYER_NAME],
            ),
            opacity=self.explore_recipe.layer_opacities.get(
                LABELER_LAYER_NAME,
                MANAGED_LAYER_DEFAULT_OPACITY[LABELER_LAYER_NAME],
            ),
        )
        if hasattr(layer, "contour"):
            layer.contour = self.explore_recipe.layer_contours.get(
                LABELER_LAYER_NAME,
                MANAGED_LAYER_DEFAULT_CONTOUR[LABELER_LAYER_NAME],
            )
        self._bind_recipe_display_tracking(layer)
        self._refresh_labeler_selected_cell_layer()
        self._raise_noncontext_mask()

    def _refresh_single_labeler_object(
        self, object_id: int, *, label_id: str | None
    ) -> None:
        """Update one Labeler object without rebuilding its whole ROI raster."""

        if self.current_mask is None or LABELER_LAYER_NAME not in self.viewer.layers:
            self.refresh_labeler_layers()
            return
        pixels = self.current_mask == int(object_id)
        if not np.any(pixels):
            self.refresh_labeler_layers()
            return
        code = 0 if label_id is None else self._labeler_code_map()[str(label_id)]
        layer = self.viewer.layers[LABELER_LAYER_NAME]
        data = np.asarray(layer.data)
        data[pixels] = code
        layer.refresh()

    def assign_selected_labeler_cell(self) -> None:
        if self.manifest is None:
            raise ValueError("Create or load an experiment first.")
        if self.current_labeler_object is None:
            raise ValueError("Select an eligible cohort cell first.")
        label_id = self.selected_labeler_class_id()
        self.labeler_records = set_labeler_record(
            self.labeler_records,
            roi=str(self.current_roi),
            object_number=self.current_labeler_object,
            label_id=label_id,
            user=os.environ.get("USERNAME") or os.environ.get("USER", ""),
        )
        self._refresh_single_labeler_object(
            self.current_labeler_object,
            label_id=label_id,
        )
        self._refresh_labeler_tally()
        self._refresh_single_labeler_result_row(
            str(self.current_roi), self.current_labeler_object
        )
        definition = next(
            item for item in self.labeler_classes if item.label_id == label_id
        )
        self.set_status(
            f"Labeler assigned {self.current_roi}/{self.current_labeler_object} "
            f"to {definition.name!r}."
        )

    def clear_selected_labeler_cell(self) -> None:
        if self.current_labeler_object is None:
            raise ValueError("Select an eligible cohort cell first.")
        before = len(self.labeler_records)
        self.labeler_records = remove_labeler_record(
            self.labeler_records,
            roi=str(self.current_roi),
            object_number=self.current_labeler_object,
        )
        if len(self.labeler_records) == before:
            self.set_status(
                f"{self.current_roi}/{self.current_labeler_object} has no Labeler "
                "assignment to clear."
            )
            return
        self._refresh_single_labeler_object(
            self.current_labeler_object,
            label_id=None,
        )
        self._refresh_labeler_tally()
        self._refresh_single_labeler_result_row(
            str(self.current_roi), self.current_labeler_object
        )
        self.set_status(
            f"Cleared the Labeler assignment for "
            f"{self.current_roi}/{self.current_labeler_object}."
        )

    def clear_all_labeler_records(self) -> None:
        count = len(self.labeler_records)
        if count == 0:
            self.set_status("There are no Labeler assignments to clear.")
            return
        reply = self.QMessageBox.question(
            self.root,
            "Clear all Labeler assignments",
            f"Remove all {count:,} in-memory Labeler assignments across every ROI?",
        )
        if reply != self.QMessageBox.Yes:
            self.set_status("Clear all Labeler assignments cancelled.")
            return
        self.labeler_records = empty_labeler_records()
        self.refresh_labeler_layers()
        self._refresh_labeler_tally()
        self._refresh_labeler_results_table()
        self.set_status(f"Cleared all {count:,} Labeler assignments.")

    def choose_labeler_csv_destination(self) -> None:
        selected, _filter = self.QFileDialog.getSaveFileName(
            self.root,
            "Export Labeler cell list",
            self.labeler_csv_path_edit.text().strip()
            or str(self.project_root / "napari_sbt_cell_labels.csv"),
            "CSV tables (*.csv)",
        )
        if selected:
            self.labeler_csv_path_edit.setText(selected)

    def _labeler_export_table(self) -> pd.DataFrame:
        if self.manifest is None or self.cohort.empty:
            raise ValueError("Create or load an experiment before exporting labels.")
        return build_labeler_export_table(
            self.labeler_records,
            self.labeler_classes,
            cohort=self.cohort,
        )

    def export_labeler_csv(self) -> None:
        table = self._labeler_export_table()
        destination = Path(self.labeler_csv_path_edit.text().strip()).expanduser()
        if destination.suffix.casefold() != ".csv":
            destination = destination.with_suffix(".csv")
            self.labeler_csv_path_edit.setText(str(destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.stem}.{uuid4().hex}.tmp.csv")
        try:
            table.to_csv(temporary, index=False)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        if self.paths is not None:
            append_audit(
                self.paths,
                {
                    "action": "export_labeler_csv",
                    "path": str(destination),
                    "assignment_count": int(len(table)),
                    "labels": [
                        definition.model_dump(mode="json")
                        for definition in self.labeler_classes
                    ],
                },
            )
        self.set_status(
            f"Exported {len(table):,} Labeler assignments to {destination}."
        )

    def apply_labeler_records_to_live_anndata(self) -> None:
        if self.adata is None:
            raise ValueError(
                "Applying Labeler assignments requires a loaded AnnData object."
            )
        obs_name = self.labeler_obs_name_edit.text().strip()
        overwrite = bool(self.labeler_overwrite_obs_check.isChecked())
        if obs_name in self.adata.obs and overwrite:
            warning = f"This will replace the live in-memory adata.obs[{obs_name!r}]."
        else:
            warning = f"This will create live in-memory adata.obs[{obs_name!r}]."
        reply = self.QMessageBox.question(
            self.root,
            "Apply Labeler assignments to AnnData",
            f"{warning}\n\n{len(self.labeler_records):,} labelled cells will "
            "receive values; all other cells will be missing. No file is written.",
        )
        if reply != self.QMessageBox.Yes:
            self.set_status("Applying Labeler assignments to AnnData was cancelled.")
            return
        apply_labeler_to_anndata(
            self.adata,
            self.labeler_records,
            self.labeler_classes,
            cohort=self.cohort,
            obs_name=obs_name,
            overwrite=overwrite,
        )
        napari_metadata = self.adata.uns.setdefault("napari_sbt", {})
        labeler_metadata = napari_metadata.setdefault("labeler", {})
        labeler_metadata[obs_name] = {
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "experiment_id": self.manifest.experiment_id,
            "experiment_revision": int(self.manifest.revision),
            "cohort_fingerprint": self.manifest.cell_scope.snapshot_sha256,
            "assignment_count": int(len(self.labeler_records)),
            "classes": [
                definition.model_dump(mode="json")
                for definition in self.labeler_classes
            ],
        }
        if self.paths is not None:
            append_audit(
                self.paths,
                {
                    "action": "apply_labeler_to_live_anndata",
                    "obs_name": obs_name,
                    "overwrite": overwrite,
                    "assignment_count": int(len(self.labeler_records)),
                },
            )
        self._populate_anndata_selectors(source="Labeler live AnnData update")
        self.set_status(
            f"Applied {len(self.labeler_records):,} Labeler assignments to "
            f"live adata.obs[{obs_name!r}]. No AnnData file was written."
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
            if state == "confirmed":
                self._mark_final_identities_stale()
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

    def clear_selected_proposed(self) -> None:
        """Clear the selected cell's proposal without touching confirmations."""

        if self.current_selected_object is None:
            raise ValueError("Select an eligible cohort cell first.")
        selected = self.labels.loc[
            self.labels["ROI"].astype(str).eq(str(self.current_roi))
            & pd.to_numeric(self.labels["ObjectNumber"], errors="coerce").eq(
                int(self.current_selected_object)
            )
        ]
        if selected.empty:
            self.set_status(
                f"{self.current_roi}/{self.current_selected_object} has no proposed "
                "label to clear."
            )
            return
        label = selected.iloc[-1]
        if str(label["state"]) != "proposed":
            self.set_status(
                f"{self.current_roi}/{self.current_selected_object} is confirmed; "
                "Clear proposed left it unchanged."
            )
            return
        class_id = str(label["class_id"])
        self.labels = remove_proposed_label(
            self.labels,
            roi=self.current_roi,
            object_number=self.current_selected_object,
        )
        write_dataframe(self.paths.labels, self.labels)
        append_audit(
            self.paths,
            {
                "action": "clear_proposed_label",
                "ROI": self.current_roi,
                "ObjectNumber": self.current_selected_object,
                "previous_class_id": class_id,
            },
        )
        self._refresh_single_classification_object(
            self.current_selected_object,
            class_id=None,
            state="cleared",
        )
        self._refresh_class_tally()
        class_definition = self._class_definition(class_id)
        class_name = class_definition.name if class_definition is not None else class_id
        self.set_status(
            f"Cleared proposed {class_name} label from "
            f"{self.current_roi}/{self.current_selected_object}."
        )

    def clear_all_proposals(self) -> None:
        """Clear reversible proposals throughout the experiment after confirmation."""

        proposed_count = int(self.labels["state"].astype(str).eq("proposed").sum())
        if proposed_count == 0:
            self.set_status("There are no proposed labels to clear.")
            return
        reply = self.QMessageBox.question(
            self.root,
            "Clear all proposals",
            f"Remove all {proposed_count:,} proposed labels across every ROI in "
            "this experiment?\n\nConfirmed labels will not be changed.",
        )
        if reply != self.QMessageBox.Yes:
            self.set_status("Clear all proposals cancelled.")
            return
        self.labels = remove_all_proposed_labels(self.labels)
        write_dataframe(self.paths.labels, self.labels)
        append_audit(
            self.paths,
            {
                "action": "clear_all_proposals",
                "removed_count": proposed_count,
                "confirmed_labels_preserved": True,
            },
        )
        self.refresh_classification_layers()
        self._refresh_class_tally()
        self.set_status(
            f"Cleared {proposed_count:,} proposals across all ROIs; confirmed "
            "labels were preserved."
        )

    def _refresh_single_classification_object(
        self,
        object_id: int,
        *,
        class_id: str | None,
        state: str,
    ) -> None:
        """Update one annotated object without rebuilding whole-ROI rasters."""

        if self.current_mask is None or state not in {
            "proposed",
            "confirmed",
            "cleared",
        }:
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
        class_code = self._class_code_map()[str(class_id)] if state != "cleared" else 0
        for label_state, layer_name in layer_names.items():
            layer = self.viewer.layers[layer_name]
            data = np.asarray(layer.data)
            data[pixels] = (
                class_code if state != "cleared" and label_state == state else 0
            )
            layer.refresh()

    def confirm_all_proposed(self) -> None:
        proposed_count = int(self.labels["state"].astype(str).eq("proposed").sum())
        self.labels = confirm_proposed(self.labels)
        write_dataframe(self.paths.labels, self.labels)
        self.manifest.locked = bool((self.labels["state"] == "confirmed").any())
        save_experiment(
            self.manifest, self.paths.root, audit_action="confirm_proposals"
        )
        if proposed_count:
            self._mark_final_identities_stale()
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
        cohort_hash = dataframe_sha256(self.cohort, ["obs_name", "ROI", "ObjectNumber"])
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
        feature_state = self.manifest.active_feature_set_id or "not built"
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
        scored = (
            int(self.scores["scorable"].fillna(False).sum()) if scores_current else 0
        )
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
            data = identity_value_map(self.current_mask, mapping, dtype=np.int32)
            self._replace_layer(
                CLASS_LAYER_NAMES[state],
                data,
                "labels",
                colormap=self._class_colormap(),
            )
        if not self.scores.empty:
            rows = self.scores.loc[self.scores["ROI"].astype(str).eq(self.current_roi)]
            minimum, maximum = self._prediction_review_range()
            confidence = pd.to_numeric(rows["maximum_probability"], errors="coerce")
            predicted_rows = rows.loc[confidence.between(minimum, maximum)]
            mapping = pd.Series(
                predicted_rows["predicted_class"].map(codes).to_numpy(),
                index=predicted_rows["ObjectNumber"].astype(int),
            )
            data = identity_value_map(self.current_mask, mapping, dtype=np.int32)
            self._replace_layer(
                CLASS_LAYER_NAMES["predicted"],
                data,
                "labels",
                colormap=self._class_colormap(),
                visible=self.explore_recipe.layer_visibility.get(
                    CLASS_LAYER_NAMES["predicted"],
                    MANAGED_LAYER_DEFAULT_VISIBILITY[CLASS_LAYER_NAMES["predicted"]],
                ),
            )
            uncertainty = pd.Series(
                pd.to_numeric(rows["normalized_entropy"], errors="coerce").to_numpy(),
                index=rows["ObjectNumber"].astype(int),
            )
            self._replace_layer(
                CLASS_LAYER_NAMES["uncertainty"],
                identity_value_map(
                    self.current_mask,
                    uncertainty,
                    background_value=np.nan,
                ),
                "image",
                colormap="magma",
                contrast_limits=(0, 1),
            )
            self._update_prediction_review_summary()
        else:
            if CLASS_LAYER_NAMES["predicted"] in self.viewer.layers:
                predicted_layer = self.viewer.layers[CLASS_LAYER_NAMES["predicted"]]
                predicted_layer.data = np.zeros_like(self.current_mask, dtype=np.int32)
                predicted_layer.refresh()
            if CLASS_LAYER_NAMES["uncertainty"] in self.viewer.layers:
                uncertainty_layer = self.viewer.layers[CLASS_LAYER_NAMES["uncertainty"]]
                uncertainty_layer.data = np.full(
                    self.current_mask.shape, np.nan, dtype=np.float32
                )
                uncertainty_layer.refresh()
            self._update_prediction_review_summary()
        self._apply_managed_layer_display_settings()
        self._raise_noncontext_mask()

    def _prediction_review_range(self) -> tuple[float, float]:
        minimum = float(self.prediction_review_min_confidence_spin.value())
        maximum = float(self.prediction_review_max_confidence_spin.value())
        return minimum, maximum

    def _update_prediction_review_summary(self) -> None:
        if self.scores.empty:
            self.prediction_review_summary_label.setText(
                "Score the cohort to review predicted-class coverage."
            )
            return
        minimum, maximum = self._prediction_review_range()
        if minimum > maximum:
            self.prediction_review_summary_label.setText(
                "Invalid display range: the minimum exceeds the maximum. Raw scores "
                "are unchanged."
            )
            return
        confidence = pd.to_numeric(self.scores["maximum_probability"], errors="coerce")
        scorable = (
            self.scores.get("scorable", pd.Series(True, index=self.scores.index))
            .fillna(False)
            .astype(bool)
        )
        visible = scorable & confidence.between(minimum, maximum)
        self.prediction_review_summary_label.setText(
            f"{int(visible.sum()):,}/{int(scorable.sum()):,} scorable predictions "
            f"are visible at confidence {minimum:.2f}–{maximum:.2f}. Raw scores "
            "are unchanged."
        )

    def apply_prediction_review_filter(self) -> None:
        minimum, maximum = self._prediction_review_range()
        if minimum > maximum:
            raise ValueError(
                "The visible prediction minimum cannot exceed its maximum."
            )
        self.refresh_classification_layers()
        self.set_status(
            "Applied the confidence range to the predicted_classes display layer."
        )

    def reset_prediction_review_filter(self) -> None:
        self.prediction_review_min_confidence_spin.setValue(0)
        self.prediction_review_max_confidence_spin.setValue(1)
        self.refresh_classification_layers()
        self.set_status("The predicted_classes layer now shows all scored predictions.")

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
            and self.manifest.active_model_features != self.model_bundle.feature_columns
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
        self.scores = pd.DataFrame()
        self._mark_final_identities_stale()
        self.refresh_classification_layers()
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
            or metadata.get("feature_set_id") != self.manifest.active_feature_set_id
            or metadata.get("labels_fingerprint")
            != confirmed_labels_fingerprint(self.labels)
            or feature_selection_stale
        ):
            raise ValueError(
                "The loaded model is stale for the current cohort, feature revision, "
                "or confirmed labels. Retrain before scoring."
            )
        self.scores = score_cohort(self.model_bundle, self._load_feature_table())
        self._mark_final_identities_stale()
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
            label_state = self.labels.loc[:, ["ROI", "ObjectNumber", "state"]].copy()
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
            identity_value_map(
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
            raise RuntimeError(
                "Create or load an experiment before validating sources."
            )
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
        process.readyReadStandardOutput.connect(self._read_source_validation_progress)
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
                self.set_status(f"SOURCE VALIDATION FAILED — {event.get('error', '')}")

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
                event.get("error", "") or "Identity join and feature matrix are usable."
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
        self._activity_finish(
            exit_code == 0,
            "Feature-source validation completed."
            if exit_code == 0
            else f"Feature-source validation exited with code {exit_code}.",
        )

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
            has_features and class_coverage_ready and self.refinement_process is None
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
        if (
            self.feature_process is not None
            or self.source_validation_process is not None
        ):
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
        self.set_status("Started leave-one-ROI-out feature refinement in a subprocess.")

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
        self._activity_finish(
            exit_code == 0 and not self.refinement_cancel_requested,
            "Feature refinement completed."
            if exit_code == 0 and not self.refinement_cancel_requested
            else (
                "Feature refinement was cancelled."
                if self.refinement_cancel_requested
                else f"Feature refinement exited with code {exit_code}."
            ),
        )

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
        summary = json.loads(self.paths.refinement_summary.read_text(encoding="utf-8"))
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
                    item.setToolTip(str(getattr(row, "redundant_with", "") or ""))
                self.refinement_results_table.setItem(row_index, column, item)
            use_item = self.QTableWidgetItem("Include")
            use_item.setFlags(use_item.flags() | self.Qt.ItemIsUserCheckable)
            use_item.setCheckState(
                self.Qt.Checked if str(row.feature) in checked else self.Qt.Unchecked
            )
            self.refinement_results_table.setItem(row_index, 7, use_item)
        if not silent:
            self.set_status(
                f"Loaded {len(ranking):,} ranked features; showing {len(display):,}."
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
        summary = json.loads(self.paths.refinement_summary.read_text(encoding="utf-8"))
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
            raise RuntimeError(
                "Create or load an experiment before feature extraction."
            )
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
        self._refresh_noncontext_mask()
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
                state["orchestrator_pid"] = int(event.get("orchestrator_pid", 0) or 0)
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
                state["recent"] = f"Failed {event.get('roi')}: {event.get('error', '')}"
        elif name == "roi_resumed":
            state["recent"] = f"Reusing valid fragment for {event.get('roi')}"
        elif name == "build_completed":
            total = int(event.get("represented_rois", 0) or 0)
            state.update(
                {
                    "phase": "Feature build complete",
                    "total_rois": total,
                    "completed_rois": int(event.get("completed_rois", total) or 0),
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
        self.feature_progress_log.append(self._format_feature_progress_event(event))
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
                "Failed"
                if "failed" in phase.lower() or "exited" in phase.lower()
                else "Not started"
            )
        self.feature_phase_label.setText(f"Phase: {state.get('phase', 'Not started')}")
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
            health = f"live; waiting {heartbeat_age:.0f}s for the next worker heartbeat"
        pid_text = f" PID {pid};" if pid else ""
        self.feature_process_health_label.setText(f"Process:{pid_text} {health}")

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
        self._activity_finish(
            exit_code == 0,
            "Feature build completed."
            if exit_code == 0
            else f"Feature build exited with code {exit_code}.",
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
        minimum_confidence, maximum_uncertainty, minimum_margin = (
            self._final_identity_thresholds()
        )
        assignments = build_assignment_table(
            self.cohort,
            self.labels,
            self.scores,
            class_ids=[item.class_id for item in self.manifest.classes],
            minimum_model_confidence=minimum_confidence,
            maximum_model_uncertainty=maximum_uncertainty,
            minimum_probability_margin=minimum_margin,
        )
        names = {item.class_id: item.name for item in self.manifest.classes}
        colours = {item.class_id: item.color for item in self.manifest.classes}
        assignments.insert(
            assignments.columns.get_loc("class_id") + 1,
            "class_name",
            assignments["class_id"].map(names),
        )
        assignments.insert(
            assignments.columns.get_loc("class_name") + 1,
            "class_colour",
            assignments["class_id"].map(colours),
        )
        return assignments

    def _final_identity_thresholds(self) -> tuple[float, float, float]:
        return (
            float(self.final_min_confidence_spin.value()),
            float(self.final_max_uncertainty_spin.value()),
            float(self.final_min_margin_spin.value()),
        )

    def _final_identity_signature_value(self) -> str:
        if self.manifest is None:
            return ""
        confirmed = self.labels.loc[self.labels["state"].astype(str).eq("confirmed")]
        label_columns = [
            "ROI",
            "ObjectNumber",
            "class_id",
            "state",
            "source",
            "user",
            "timestamp",
        ]
        score_columns = [
            column
            for column in (
                "ROI",
                "ObjectNumber",
                "predicted_class",
                "maximum_probability",
                "probability_margin",
                "normalized_entropy",
                "model_id",
                "scorable",
            )
            if column in self.scores.columns
        ]
        payload = {
            "experiment_id": self.manifest.experiment_id,
            "revision": self.manifest.revision,
            "cohort_fingerprint": self.manifest.cell_scope.snapshot_sha256,
            "confirmed_labels": dataframe_sha256(confirmed, label_columns),
            "scores": (
                dataframe_sha256(self.scores, score_columns)
                if not self.scores.empty and score_columns
                else "none"
            ),
            "classes": [item.model_dump(mode="json") for item in self.manifest.classes],
            "thresholds": self._final_identity_thresholds(),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _mark_final_identities_stale(self, *_args) -> None:
        if self.final_identity_signature is None:
            return
        self.final_identity_signature = None
        self.final_identity_summary_label.setText(
            "Decision rules or inputs changed. Click Create / refresh final cell "
            "identities before exporting."
        )

    def create_final_identities(self) -> pd.DataFrame:
        assignments = self._assignments()
        minimum_confidence, maximum_uncertainty, minimum_margin = (
            self._final_identity_thresholds()
        )
        source_counts = assignments["assignment_source"].value_counts().to_dict()
        class_counts = (
            assignments.loc[assignments["class_id"].notna(), "class_id"]
            .astype(str)
            .value_counts()
            .to_dict()
        )
        named_class_counts = {}
        for class_id, count in class_counts.items():
            definition = self._class_definition(class_id)
            class_name = definition.name if definition is not None else class_id
            named_class_counts[class_name] = int(count)
        rejected = int(
            (
                assignments["predicted_class"].notna()
                & assignments["assignment_source"].eq("unassigned")
            ).sum()
        )
        decision = {
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "experiment_id": self.manifest.experiment_id,
            "experiment_revision": self.manifest.revision,
            "model_id": (
                str(self.scores["model_id"].dropna().iloc[0])
                if not self.scores.empty
                and "model_id" in self.scores
                and not self.scores["model_id"].dropna().empty
                else None
            ),
            "minimum_model_confidence": minimum_confidence,
            "maximum_model_uncertainty": maximum_uncertainty,
            "minimum_probability_margin": minimum_margin,
            "precedence": "confirmed > accepted model > unassigned",
            "proposals_are_final": False,
            "counts_by_source": {
                str(key): int(value) for key, value in source_counts.items()
            },
            "counts_by_class": named_class_counts,
            "rejected_model_predictions": rejected,
        }
        self.final_assignments = assignments
        self.final_identity_decision = decision
        self.final_identity_signature = self._final_identity_signature_value()
        canonical = self.paths.exports / "final_identities.parquet"
        export_assignment_table(assignments, canonical)
        write_json(self.paths.exports / "final_identity_decision.json", decision)
        append_audit(
            self.paths,
            {
                "action": "create_final_identities",
                **decision,
                "canonical_table": str(canonical),
            },
        )
        confirmed_count = int(source_counts.get("confirmed", 0))
        model_count = int(source_counts.get("model", 0))
        unassigned_count = int(source_counts.get("unassigned", 0))
        self.final_identity_summary_label.setText(
            f"Current: {confirmed_count:,} confirmed + {model_count:,} accepted "
            f"model assignments; {unassigned_count:,} unassigned. Rules: confidence "
            f"≥ {minimum_confidence:.2f}, entropy ≤ {maximum_uncertainty:.2f}, "
            f"margin ≥ {minimum_margin:.2f}. Canonical provenance assets were saved "
            f"in {self.paths.exports}."
        )
        self.set_status(
            f"Created final identities: {confirmed_count:,} confirmed, "
            f"{model_count:,} model, {unassigned_count:,} unassigned."
        )
        return assignments

    def _require_current_final_identities(self) -> pd.DataFrame:
        if self.final_identity_signature is None or self.final_assignments.empty:
            raise ValueError(
                "Final identities have not been created, or their inputs changed. "
                "Open Classify → Finalize & export and click Create / refresh final "
                "cell identities."
            )
        if self.final_identity_signature != self._final_identity_signature_value():
            self._mark_final_identities_stale()
            raise ValueError(
                "Labels, scores, classes, or decision rules changed after final "
                "identities were created. Refresh them before export."
            )
        return self.final_assignments.copy()

    def _feature_provenance(self) -> dict:
        return (
            json.loads(self.paths.feature_manifest.read_text(encoding="utf-8"))
            if self.paths.feature_manifest.exists()
            else {}
        )

    def _model_provenance(self) -> dict:
        if self.model_bundle is not None:
            return dict(self.model_bundle.metadata)
        metadata_path = self.paths.models / "classifier_latest.json"
        if metadata_path.exists():
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        return {}

    def export_assignments(self) -> None:
        assignments = self._require_current_final_identities()
        destination = Path(self.assignment_path_edit.text()).expanduser()
        if not destination.suffix:
            destination = destination.with_suffix(".csv")
            self.assignment_path_edit.setText(str(destination))
        export_assignment_table(assignments, destination)
        append_audit(
            self.paths,
            {
                "action": "export_final_identity_table",
                "destination": str(destination),
                "decision": self.final_identity_decision,
            },
        )
        self.set_status(f"Exported final cohort identities: {destination}")

    def export_adata(self) -> None:
        if not self.manifest.anndata_path:
            raise ValueError("Annotated AnnData export requires an AnnData source.")
        assignments = self._require_current_final_identities()
        destination = Path(self.annotated_path_edit.text())
        export_annotated_anndata(
            self.manifest.anndata_path,
            destination,
            assignments,
            self.manifest,
            feature_provenance=self._feature_provenance(),
            model_provenance=self._model_provenance(),
            metrics={"final_identity_decision": self.final_identity_decision},
        )
        append_audit(
            self.paths,
            {
                "action": "export_final_identities_to_anndata_copy",
                "destination": str(destination),
                "decision": self.final_identity_decision,
            },
        )
        self.set_status(f"Exported atomic annotated AnnData copy: {destination}")

    def apply_final_identities_to_live_anndata(self) -> None:
        if self.adata is None:
            raise ValueError("No live AnnData object is loaded in this session.")
        assignments = self._require_current_final_identities()
        reply = self.QMessageBox.question(
            self.root,
            "Apply final identities to live AnnData",
            "Add the final subclass, source, confidence, uncertainty, probability, "
            "combined-population, and provenance fields to the AnnData object held "
            "in memory?\n\nThis does not write to or overwrite its source file.",
        )
        if reply != self.QMessageBox.Yes:
            self.set_status("Applying final identities to live AnnData was cancelled.")
            return
        apply_assignments_to_anndata(
            self.adata,
            assignments,
            self.manifest,
            feature_provenance=self._feature_provenance(),
            model_provenance=self._model_provenance(),
            metrics={"final_identity_decision": self.final_identity_decision},
        )
        self._populate_anndata_selectors(
            source="the live AnnData after final identity application"
        )
        append_audit(
            self.paths,
            {
                "action": "apply_final_identities_to_live_anndata",
                "written_to_disk": False,
                "decision": self.final_identity_decision,
            },
        )
        self.set_status(
            f"Applied final identities to the live AnnData as "
            f"{self.manifest.output_obs_slug}_subclass and related fields; no "
            "source file was written."
        )

    def export_cohort_masks(self) -> None:
        masks = {
            roi: self._mask_path_for_roi(roi)
            for roi in sorted(self.cohort["ROI"].astype(str).unique())
        }
        written = materialize_cohort_masks(masks, self.cohort, self.paths.cohort_masks)
        self.set_status(f"Wrote {len(written)} cohort masks; originals were untouched.")

    def export_cleaned_masks(self) -> None:
        assignments = self._require_current_final_identities()
        minimum_confidence, maximum_uncertainty, minimum_margin = (
            self._final_identity_thresholds()
        )
        reply = self.QMessageBox.question(
            self.root,
            "Cleaned masks",
            "Write derived masks using confirmed exclusions and accepted model "
            "exclusions?\n\nCurrent model rules: confidence ≥ "
            f"{minimum_confidence:.2f}, entropy ≤ {maximum_uncertainty:.2f}, "
            f"margin ≥ {minimum_margin:.2f}.",
        )
        if reply != self.QMessageBox.Yes:
            return
        masks = {
            roi: self._mask_path_for_roi(roi)
            for roi in sorted(assignments["ROI"].astype(str).unique())
        }
        written = export_cleaned_masks(
            masks,
            assignments,
            self.manifest.classes,
            self.paths.exports / "cleaned_masks",
            prediction_confidence_threshold=minimum_confidence,
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
            for identity in rows.loc[
                contains, ["obs_name", "ROI", "ObjectNumber"]
            ].itertuples(index=False):
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
            {
                "roi": self.current_roi,
                "polygons": [np.asarray(value).tolist() for value in shapes.data],
            },
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
            self.set_status(
                "Flipped selected display layer; source files were untouched."
            )
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
    anndata_path: str | Path | object | None = None,
    anndata: object | None = None,
    masks_folder: str | Path | None = None,
    images_folders: Iterable[str | Path] = (),
    extra_images_folders: Iterable[str | Path] = (),
):
    """Create the viewer and dock; paths or a live AnnData object are accepted."""

    import napari

    if viewer is None:
        viewer = napari.Viewer(title="napari_sbt — cohort-first cell classification")
    controller = NapariSBTController(
        viewer,
        project_root=project_root,
        experiment=experiment,
        anndata_path=anndata_path,
        anndata=anndata,
        masks_folder=masks_folder,
        images_folders=images_folders,
        extra_images_folders=extra_images_folders,
    )
    dock = viewer.window.add_dock_widget(
        controller.root,
        name="napari_sbt",
        area="right",
    )
    controller.install_readiness_dock()
    # Reapply the preferred split once Qt has completed this event-loop turn;
    # this keeps the compact readiness dock below Layers after Napari finishes
    # sizing all newly added docks.
    from qtpy.QtCore import QTimer

    QTimer.singleShot(0, controller._position_readiness_dock)
    return viewer, controller, dock


def launch_notebook(
    adata,
    *,
    viewer=None,
    project_root: str | Path | None = None,
    experiment: str | Path | None = None,
    masks_folder: str | Path | None = None,
    images_folders: Iterable[str | Path] = (),
    extra_images_folders: Iterable[str | Path] = (),
):
    """Launch from Jupyter with a live AnnData and Qt event-loop integration."""

    try:
        from IPython import get_ipython
    except ImportError:
        shell = None
    else:
        shell = get_ipython()
    if shell is not None:
        shell.run_line_magic("gui", "qt")
    return launch(
        viewer=viewer,
        project_root=project_root,
        experiment=experiment,
        anndata=adata,
        masks_folder=masks_folder,
        images_folders=images_folders,
        extra_images_folders=extra_images_folders,
    )


__all__ = ["NapariSBTController", "launch", "launch_notebook"]
