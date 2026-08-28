"""Unified Napari dock for cohort-first IMC exploration and classification."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from datetime import datetime
from html import escape
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from SpatialBiologyToolkit._napari_imc_normalization import (
    find_normalization_parameters,
    load_normalization_parameters,
    normalization_parameters_payload,
    prepare_normalization_parameters,
)
from SpatialBiologyToolkit.nimbus_normalization import NimbusNormalizationParameters
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

from .anndata_io import write_h5ad_compat
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
from .colour_helper import (
    assign_categorical_colours,
    categorical_colour_collisions,
    categorical_palette_catalog,
    contrasting_text_colour,
)
from .dataset_maintenance import (
    CellFilterRequest,
    append_maintenance_audit,
    apply_cell_filter,
    apply_var_rename,
    atomic_write_anndata,
    copy_renamed_images,
    dataset_readiness,
    plan_image_renames,
    preview_cell_filter,
    preview_mask_rebuild,
    preview_var_rename,
    rebuild_masks_and_object_numbers,
    remap_categorical_observation,
    remove_anndata_vars,
)
from .explore import (
    EXPLORE_RECIPE_FUNCTION_KEYS,
    EXPLORE_STATE_VERSION,
    SIX_COLOUR_COLORMAPS,
    ExploreRecipePreset,
    ExploreReviewState,
    ExploreViewRecipe,
    categorical_colour_map,
    categorical_object_categories,
    cell_level_observations,
    format_roi_metadata_value,
    identity_value_map,
    marker_values,
    population_identity_map,
    population_recipe_key,
    rank_marker_rois,
    recipe_layer_data_is_current,
    roi_level_metadata,
)
from .exports import (
    apply_assignments_to_anndata,
    build_assignment_table,
    build_integrated_identity_table,
    export_annotated_anndata,
    export_assignment_table,
    export_cleaned_masks,
    integrated_identity_crosstab,
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
    FEATURE_EXTRACTION_CONTRACT_VERSION,
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
    harmonize_merge_colours,
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
from .publication_export import (
    DEFAULT_FILENAME_TEMPLATE,
    PUBLICATION_EXPORT_SCHEMA_VERSION,
    PixelCalibration,
    PublicationAnnotations,
    PublicationExportPreset,
    PublicationExportState,
    PublicationFrame,
    PublicationOutput,
    PublicationScaleBar,
    ResolvedPublicationFrame,
    build_publication_filename,
    camera_frame_from_canvas,
    compose_publication_image,
    detect_tiff_pixel_calibration,
    downsample_publication_image,
    publication_render_geometry,
    publication_resolution_scale,
    resolve_publication_dpi,
    resolve_publication_frame,
    resolve_publication_output_size,
    save_publication_image,
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
    discover_dataset_assets,
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
from .variable_ordering import VARIABLE_ORDER_OPTIONS, VariableOrderRegistry

CLASS_LAYER_NAMES = {
    "confirmed": "confirmed_classes",
    "proposed": "proposed_classes",
    "predicted": "predicted_classes",
    "uncertainty": "uncertainty_or_probability",
}

SELECTED_CELL_LAYER_NAME = "selected_cell_outline"
ALL_CELLS_LAYER_NAME = "all_cells"
NONCONTEXT_MASK_LAYER_NAME = "noncontext_mask"
LABELER_LAYER_NAME = "labeler_assignments"
LABELER_SELECTED_CELL_LAYER_NAME = "labeler_selected_cell_outline"
CELL_PROPERTIES_SELECTED_LAYER_NAME = "cell_properties_selected_cell_outline"
EXPLORE_DATA_CACHE_MAX_BYTES = 512 * 1024 * 1024
EXPLORE_DATA_CACHE_MAX_ITEMS = 48
ASSET_INDEX_SCHEMA_VERSION = 2

_POPULATION_OBSERVATION_HINTS = (
    "population",
    "cell_type",
    "celltype",
    "phenotype",
    "leiden",
    "cluster",
    "class",
    "label",
)


def _population_observation_columns(
    columns: Iterable[str],
    *,
    roi_obs: str,
    object_obs: str,
) -> list[str]:
    """Exclude cell-identity fields from population-oriented selectors."""

    identity_columns = {str(roi_obs), str(object_obs)}
    return [str(column) for column in columns if str(column) not in identity_columns]


def _preferred_population_observation(
    obs: pd.DataFrame,
    candidates: Iterable[str],
    *,
    prefer_leiden: bool = False,
) -> str | None:
    """Choose a sensible population label without selecting identity columns."""

    candidates = [str(column) for column in candidates]
    if not candidates:
        return None
    categorical = [
        column
        for column in candidates
        if isinstance(obs[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(obs[column].dtype)
    ]
    pool = categorical or candidates
    hints = (
        ("leiden", *_POPULATION_OBSERVATION_HINTS)
        if prefer_leiden
        else _POPULATION_OBSERVATION_HINTS
    )
    for hint in hints:
        match = next(
            (column for column in pool if hint in column.casefold()),
            None,
        )
        if match is not None:
            return match
    return pool[0]

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
    "dataset_maintenance": (
        "Save or derive synchronized AnnData, channel-image, cell, ROI, and mask "
        "assets with explicit previews and readiness checks."
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
    "dataset_maintenance": {
        "setup",
        "dataset_maintenance",
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
        "dataset_maintenance",
        "classify",
        "labeler",
        "regions_export",
        "layers_status",
    },
}

MANAGED_RECIPE_LAYERS = {
    ALL_CELLS_LAYER_NAME: "Complete original segmentation: all cells",
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
    ALL_CELLS_LAYER_NAME: True,
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
    ALL_CELLS_LAYER_NAME: 1.0,
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

MANAGED_LAYER_DEFAULT_BLENDING = {
    CLASS_LAYER_NAMES["uncertainty"]: "additive",
}

MANAGED_LAYER_DEFAULT_CONTOUR = {
    ALL_CELLS_LAYER_NAME: 1,
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
        write_h5ad_compat(adata, temporary)
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
            QListWidgetItem,
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
        self.QComboBox = QComboBox
        self.QColor = QColor
        self.QIcon = QIcon
        self.QPixmap = QPixmap
        self.QListWidgetItem = QListWidgetItem
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
        self._launch_experiment_was_explicit = experiment is not None
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
        self.integrated_identity_table = pd.DataFrame()
        self.identity_integration_signature: str | None = None
        self.identity_integration_plan: dict[str, object] = {}
        self._identity_integration_custom_names: dict[str, str] = {}
        self._updating_identity_integration_controls = False
        self._classification_enabled = False
        self.model_bundle = None
        self.current_roi: str | None = None
        self.current_mask: np.ndarray | None = None
        self.current_mask_path: Path | None = None
        self.current_selected_object: int | None = None
        self.current_labeler_object: int | None = None
        self.cell_properties_selected_object: int | None = None
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
        self._maintenance_obs_default_output = ""
        self.scanpy_plot_windows: dict[str, dict[str, object]] = {}
        self.reviewed_rois: set[str] = set()
        self._class_shortcuts: list[str] = []
        self._explore_recipe_shortcuts: list[str] = []
        self.current_image_paths: dict[str, Path] = {}
        self.explore_recipe = ExploreViewRecipe()
        self.explore_review_state = ExploreReviewState()
        self.publication_export_state = PublicationExportState()
        self.publication_export_dialog = None
        self.publication_batch: dict[str, object] | None = None
        self._publication_export_running = False
        self.display_normalization: dict[str, NimbusNormalizationParameters] = {}
        self.variable_order_registry = VariableOrderRegistry()
        self.variable_order_registry.set_adata(self.adata)
        self._variable_order_combos: list[object] = []
        self._syncing_variable_order = False
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
        self._cell_properties_position_index: dict[str, dict[int, int]] = {}
        self._cell_properties_colour_maps: dict[str, dict[str, str]] = {}
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
        self._active_recipe_label_refresh_pending = False
        self._applying_explore_recipe = False
        self._updating_recipe_layer_state = False
        self._updating_queue_controls = False
        self.maintenance_dirty = False
        self.maintenance_image_rename_plan = None
        self.maintenance_last_filter_request: CellFilterRequest | None = None
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
        self.activity_state_changed_at = datetime.now().astimezone()
        self.activity_waiting_for_process = False
        self._activity_styled_state: str | None = None
        self.cell_properties_settings_dialog = None

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
            "Track manual Napari layer changes in the working recipe (this session)"
        )
        self.live_recipe_tracking_check.setChecked(True)
        self.live_recipe_tracking_check.setToolTip(
            "This can be changed at any time in Setup, Explore, or Population QC. "
            "Disable it for the lightest display path. Explicitly saved recipes "
            "still load, but manual layer display changes are not copied back into "
            "the working recipe automatically."
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
        self.detect_dataset_inputs_button = QPushButton(
            "Automatically detect missing inputs"
        )
        self.detect_dataset_inputs_button.setToolTip(
            "Look only at the project root and conventional immediate folders. "
            "This does not scan image or mask contents and does not replace the "
            "dataset integrity check."
        )
        self.reload_all_inputs_button = QPushButton("Reload all selected components")
        self.integrity_status_label = QLabel(
            "Not validated in this session. Normal navigation uses direct, "
            "cached file lookups and does not scan complete folders."
        )
        self.integrity_status_label.setWordWrap(True)
        input_actions = QHBoxLayout()
        input_actions.addWidget(self.detect_dataset_inputs_button)
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
            "Load a Nimbus marker/Vmax/lower-threshold JSON or CSV, then review "
            "or edit all three values below. Legacy marker-to-value JSON and "
            "Marker/Value CSV files remain supported with a lower threshold of "
            "zero. Scalar images are "
            "normalized to 0-1; the default contrast handles below are used only "
            "when a saved recipe has no channel-specific range."
        )
        display_explanation.setWordWrap(True)
        normalization_source = QWidget()
        normalization_source_layout = QHBoxLayout(normalization_source)
        normalization_source_layout.setContentsMargins(0, 0, 0, 0)
        self.normalization_edit = QLineEdit()
        self.normalization_edit.setPlaceholderText(
            "Optional Nimbus normalization CSV or legacy JSON"
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
        self.normalization_table = QTableWidget(0, 3)
        self.normalization_table.setHorizontalHeaderLabels(
            ["Marker", "Vmax", "Lower threshold"]
        )
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

        feature_readiness_group = workflow_group(
            "Feature readiness",
            "feature_building",
            "Feature readiness",
        )
        feature_readiness_layout = QVBoxLayout(feature_readiness_group)
        self.feature_readiness_banner = QLabel(
            "⚪ No workspace — feature status is unavailable"
        )
        self.feature_readiness_banner.setWordWrap(True)
        self.feature_readiness_banner.setMinimumHeight(48)
        self.feature_readiness_detail = QLabel(
            "Create or load a classification workspace to configure and build features."
        )
        self.feature_readiness_detail.setWordWrap(True)
        self.feature_readiness_coverage = QProgressBar()
        self.feature_readiness_coverage.setRange(0, 1)
        self.feature_readiness_coverage.setValue(0)
        self.feature_readiness_coverage.setFormat("No feature table")
        self.feature_readiness_next_step = QLabel(
            "Next: create or load a workspace in Setup."
        )
        self.feature_readiness_next_step.setWordWrap(True)
        readiness_actions = QHBoxLayout()
        self.refresh_feature_readiness_button = QPushButton(
            "Refresh saved feature status"
        )
        readiness_actions.addWidget(self.refresh_feature_readiness_button)
        readiness_actions.addStretch(1)
        feature_readiness_layout.addWidget(self.feature_readiness_banner)
        feature_readiness_layout.addWidget(self.feature_readiness_detail)
        feature_readiness_layout.addWidget(self.feature_readiness_coverage)
        feature_readiness_layout.addWidget(self.feature_readiness_next_step)
        feature_readiness_layout.addLayout(readiness_actions)
        feature_builder_layout.addWidget(feature_readiness_group)

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
        self.feature_variable_order_combo = self._create_variable_order_combo()
        self.feature_channel_list = QListWidget()
        self.feature_channel_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.feature_channel_list.setMaximumHeight(150)
        channel_actions = QHBoxLayout()
        self.refresh_feature_channels_button = QPushButton("Refresh available channels")
        self.select_built_feature_channels_button = QPushButton(
            "Select feature markers"
        )
        self.select_built_feature_channels_button.setToolTip(
            "Select channel-derived markers from the active built feature table. "
            "Before a build, use the saved/current synthetic-feature recipe."
        )
        self.select_all_feature_channels_button = QPushButton("Select all")
        self.clear_feature_channels_button = QPushButton("Clear selection")
        channel_actions.addWidget(self.refresh_feature_channels_button)
        channel_actions.addWidget(self.select_built_feature_channels_button)
        channel_actions.addWidget(self.select_all_feature_channels_button)
        channel_actions.addWidget(self.clear_feature_channels_button)
        self.channels_edit = QLineEdit()
        self.channels_edit.setReadOnly(True)
        self.channels_edit.setPlaceholderText("Every discovered channel")
        channel_layout.addWidget(channel_explanation)
        feature_order_row = QHBoxLayout()
        feature_order_row.addWidget(QLabel("Variable list order"))
        feature_order_row.addWidget(self.feature_variable_order_combo, 1)
        channel_layout.addLayout(feature_order_row)
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
        self.add_all_cells_mask_button = QPushButton("Add all-cells mask")
        self.add_all_cells_mask_button.setToolTip(
            "Add the complete original segmentation for the current ROI as the "
            "editable-off 'all_cells' labels layer. This reuses the mask already "
            "in memory and does not reload the ROI."
        )
        self.publication_export_button = QPushButton("Publication export…")
        self.publication_export_button.setToolTip(
            "Open reproducible framing, scale-bar, single-image, and bulk ROI "
            "publication export controls for the current Explore view."
        )
        roi_row.addWidget(self.previous_roi_button)
        roi_row.addWidget(QLabel("ROI"))
        roi_row.addWidget(self.roi_combo)
        roi_row.addWidget(self.next_roi_button)
        roi_row.addWidget(self.reload_roi_button)
        roi_row.addWidget(self.add_all_cells_mask_button)
        roi_row.addWidget(self.publication_export_button)
        explore_layout.addLayout(roi_row)
        roi_options_row = QHBoxLayout()
        self.show_empty_rois = QCheckBox("Include ROIs with no eligible cells")
        self.context_check_display = QCheckBox("Show dimmed full-mask context")
        self.auto_reload_view_check = QCheckBox(
            "Reapply the current Explore recipe after changing ROI"
        )
        self.auto_reload_view_check.setToolTip(
            "When you choose a different ROI, recreate the same image channels, "
            "colours, contrasts, overlays, visibility, and opacity for that ROI. "
            "This setting does not react to layer edits or reload the current ROI."
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
        self.explore_live_recipe_tracking_check = QCheckBox(
            "Track manual layer display changes in the working recipe"
        )
        self.explore_live_recipe_tracking_check.setChecked(True)
        self.explore_live_recipe_tracking_check.setToolTip(
            "Session control: switch this off when you only want to inspect layers. "
            "Saved recipes and the explicit ‘Update from current layers’ button "
            "continue to work. Switching visibility never reloads ROI data."
        )
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
        reload_recipe_layout.addWidget(self.explore_live_recipe_tracking_check)
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
        self.explore_variable_order_combo = self._create_variable_order_combo()
        self.select_feature_marker_overlays_button = QPushButton(
            "Select feature markers"
        )
        self.select_feature_marker_overlays_button.setToolTip(
            "Select AnnData markers that contributed channel-derived features "
            "to the active feature table."
        )
        self.load_marker_overlays_button = QPushButton(
            "Add selected adata.X markers as cell overlays"
        )
        self.rank_marker_rois_button = QPushButton("Rank ROIs by selected marker")
        self.rank_marker_rois_button.setToolTip(
            "Select exactly one marker above. ROIs are ranked by its mean adata.X "
            "signal quantified inside segmented cells using the current Overlay "
            "scope; raw image background is not measured."
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
        overlay_form.addRow(
            "Variable list order", self.explore_variable_order_combo
        )
        overlay_form.addRow("Cell-level marker overlays", self.marker_overlay_list)
        marker_overlay_actions = QWidget()
        marker_overlay_actions_layout = QHBoxLayout(marker_overlay_actions)
        marker_overlay_actions_layout.setContentsMargins(0, 0, 0, 0)
        marker_overlay_actions_layout.addWidget(
            self.select_feature_marker_overlays_button
        )
        marker_overlay_actions_layout.addWidget(self.load_marker_overlays_button)
        marker_overlay_actions_layout.addWidget(self.rank_marker_rois_button)
        overlay_form.addRow("", marker_overlay_actions)
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
        self.select_feature_image_channels_button = QPushButton(
            "Select feature markers"
        )
        self.select_feature_image_channels_button.setToolTip(
            "Select current-ROI image channels that contributed channel-derived "
            "features to the active feature table."
        )
        self.load_channels_button = QPushButton("Load selected greyscale")
        self.load_six_colour_button = QPushButton("Load selected as R/G/B/C/Y/M")
        self.load_rgb_button = QPushButton("Load first three selected as RGB")
        image_actions.addWidget(self.select_feature_image_channels_button)
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
        self.population_qc_scope_banner = QLabel(
            "Open a workspace to show the Population QC cell scope."
        )
        self.population_qc_scope_banner.setWordWrap(True)
        self.population_qc_scope_banner.setObjectName("sbtPopulationQCScope")
        self.population_qc_scope_banner.setStyleSheet(
            "background: #e0f2fe; color: #075985; border: 2px solid #38bdf8; "
            "border-radius: 7px; padding: 9px; font-weight: 800;"
        )
        population_qc_layout.addWidget(self.population_qc_scope_banner)

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
        self.population_qc_live_recipe_tracking_check = QCheckBox(
            "Track manual layer display changes in the working recipe"
        )
        self.population_qc_live_recipe_tracking_check.setChecked(True)
        self.population_qc_live_recipe_tracking_check.setToolTip(
            "Session control shared with Setup and Explore. Population QC defaults "
            "to off for speed, but saved RGB recipes and explicit save/load actions "
            "continue to work."
        )
        population_qc_selection_form.addRow(
            "Live recipe tracking", self.population_qc_live_recipe_tracking_check
        )
        self.population_qc_variable_order_combo = (
            self._create_variable_order_combo()
        )
        population_qc_selection_form.addRow(
            "Variable list order", self.population_qc_variable_order_combo
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
        self._curation_auto_obs_source: str | None = None
        self._curation_auto_obs_value = "population_curated"
        self.curation_derived_obs_edit.setPlaceholderText(
            "For example: population_named"
        )
        self.create_population_draft_button = QPushButton(
            "Create new label draft"
        )
        self.save_population_draft_button = QPushButton("Save and Update")
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
        self.auto_colour_populations_button = QPushButton("Automatically colour…")
        self.import_population_mapping_button = QPushButton(
            "Import preliminary names from CSV"
        )
        self.export_population_mapping_button = QPushButton(
            "Export editable mapping CSV"
        )
        base_mapping_actions.addWidget(self.name_selected_populations_button)
        base_mapping_actions.addWidget(self.colour_selected_populations_button)
        base_mapping_actions.addWidget(self.auto_colour_populations_button)
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
            variable_order_combo_factory=self._create_variable_order_combo,
            order_variables=self._ordered_variable_values,
        )
        add_tab(
            self.scanpy_plotting_panel.widget,
            "📊 Scanpy plotting",
            "scanpy_plotting",
        )
        self.scanpy_plotting_tab_index = self.tabs.count() - 1

        # Dataset Maintenance. File-system scans are deliberately explicit; the
        # dashboard otherwise consumes the index produced by Setup validation.
        maintenance = QWidget()
        maintenance_layout = QVBoxLayout(maintenance)
        maintenance_warning = QLabel(
            "⚠ Dataset Maintenance can change the live AnnData and create aligned "
            "image or mask assets. Changes remain in memory until AnnData is saved. "
            "New output files/folders are the default; original disk assets are not "
            "modified unless replacement is explicitly enabled."
        )
        maintenance_warning.setWordWrap(True)
        maintenance_warning.setObjectName("sbtMaintenanceWarning")
        maintenance_layout.addWidget(maintenance_warning)

        maintenance_readiness_group = workflow_group(
            "1. Dataset and synchronization readiness",
            "dataset_maintenance",
            "Dataset and synchronization readiness",
        )
        maintenance_readiness_layout = QVBoxLayout(maintenance_readiness_group)
        maintenance_readiness_actions = QHBoxLayout()
        self.refresh_maintenance_readiness_button = QPushButton(
            "Refresh from current index"
        )
        self.reindex_maintenance_assets_button = QPushButton(
            "Rebuild mask/image index now…"
        )
        self.reindex_maintenance_assets_button.setToolTip(
            "Explicitly scan configured mask and image folders. This is never run "
            "automatically when changing tabs."
        )
        maintenance_readiness_actions.addWidget(
            self.refresh_maintenance_readiness_button
        )
        maintenance_readiness_actions.addWidget(self.reindex_maintenance_assets_button)
        maintenance_readiness_actions.addStretch(1)
        self.maintenance_unsaved_label = QLabel("No in-memory maintenance changes.")
        self.maintenance_unsaved_label.setWordWrap(True)
        self.maintenance_readiness_tree = QTreeWidget()
        self.maintenance_readiness_tree.setHeaderLabels(
            ["State", "Resource", "Details"]
        )
        self.maintenance_readiness_tree.setRootIsDecorated(False)
        self.maintenance_readiness_tree.setAlternatingRowColors(False)
        self.maintenance_readiness_tree.setMinimumHeight(145)
        self.maintenance_readiness_tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.maintenance_readiness_tree.header().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.maintenance_readiness_tree.header().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        maintenance_readiness_layout.addLayout(maintenance_readiness_actions)
        maintenance_readiness_layout.addWidget(self.maintenance_unsaved_label)
        maintenance_readiness_layout.addWidget(self.maintenance_readiness_tree)
        maintenance_layout.addWidget(maintenance_readiness_group)

        self.maintenance_tool_tabs = QTabWidget()

        maintenance_save_page = QWidget()
        maintenance_save_layout = QVBoxLayout(maintenance_save_page)
        maintenance_save_group = workflow_group(
            "2. Save the current in-memory AnnData",
            "dataset_maintenance",
            "Save current AnnData",
        )
        maintenance_save_form = QFormLayout(maintenance_save_group)
        maintenance_save_destination = QWidget()
        maintenance_save_destination_layout = QHBoxLayout(
            maintenance_save_destination
        )
        maintenance_save_destination_layout.setContentsMargins(0, 0, 0, 0)
        self.maintenance_anndata_path_edit = QLineEdit(
            str(self.project_root / "napari_sbt_maintained.h5ad")
        )
        self.choose_maintenance_anndata_button = QPushButton("Choose…")
        maintenance_save_destination_layout.addWidget(
            self.maintenance_anndata_path_edit
        )
        maintenance_save_destination_layout.addWidget(
            self.choose_maintenance_anndata_button
        )
        self.maintenance_overwrite_anndata_check = QCheckBox(
            "Allow replacement of this exact existing .h5ad file"
        )
        self.save_maintenance_anndata_button = QPushButton("Save current AnnData")
        self.save_maintenance_anndata_button.setObjectName("sbtPrimaryActionButton")
        self.use_saved_maintenance_anndata_check = QCheckBox(
            "Use the saved file as the Setup AnnData path"
        )
        self.use_saved_maintenance_anndata_check.setChecked(True)
        self.maintenance_save_status_label = QLabel(
            "Nothing is written until Save current AnnData is pressed."
        )
        self.maintenance_save_status_label.setWordWrap(True)
        maintenance_save_form.addRow("Destination", maintenance_save_destination)
        maintenance_save_form.addRow("", self.maintenance_overwrite_anndata_check)
        maintenance_save_form.addRow("", self.use_saved_maintenance_anndata_check)
        maintenance_save_form.addRow("", self.save_maintenance_anndata_button)
        maintenance_save_form.addRow("Status", self.maintenance_save_status_label)
        maintenance_save_layout.addWidget(maintenance_save_group)
        maintenance_save_layout.addStretch(1)
        self.maintenance_tool_tabs.addTab(maintenance_save_page, "Overview & Save")

        maintenance_channels_page = QWidget()
        maintenance_channels_layout = QVBoxLayout(maintenance_channels_page)
        maintenance_order_row = QHBoxLayout()
        maintenance_order_row.addWidget(QLabel("Variable list order"))
        self.maintenance_variable_order_combo = self._create_variable_order_combo()
        maintenance_order_row.addWidget(self.maintenance_variable_order_combo, 1)
        maintenance_channels_layout.addLayout(maintenance_order_row)
        maintenance_rename_group = workflow_group(
            "2. Rename variables and matching images",
            "dataset_maintenance",
            "Rename variables and images",
        )
        maintenance_rename_layout = QVBoxLayout(maintenance_rename_group)
        maintenance_rename_help = QLabel(
            "Enter only the new names you want to use. Preview uses the explicit "
            "Setup image index; matched images can be copied into a new synchronized "
            "folder without touching their sources."
        )
        maintenance_rename_help.setWordWrap(True)
        self.maintenance_var_rename_table = QTableWidget(0, 3)
        self.maintenance_var_rename_table.setHorizontalHeaderLabels(
            ["Current variable", "New variable", "Indexed images"]
        )
        self.maintenance_var_rename_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.maintenance_var_rename_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.maintenance_var_rename_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self.maintenance_var_rename_table.setMinimumHeight(245)
        self.maintenance_update_raw_names_check = QCheckBox(
            "Apply the same exact renames to matching adata.raw variables"
        )
        maintenance_image_output = QWidget()
        maintenance_image_output_layout = QHBoxLayout(maintenance_image_output)
        maintenance_image_output_layout.setContentsMargins(0, 0, 0, 0)
        self.maintenance_image_output_edit = QLineEdit(
            str(self.project_root / "napari_sbt_renamed_images")
        )
        self.choose_maintenance_image_output_button = QPushButton("Choose…")
        maintenance_image_output_layout.addWidget(self.maintenance_image_output_edit)
        maintenance_image_output_layout.addWidget(
            self.choose_maintenance_image_output_button
        )
        maintenance_rename_actions = QHBoxLayout()
        self.preview_maintenance_var_rename_button = QPushButton("Preview rename")
        self.apply_maintenance_var_rename_button = QPushButton(
            "Rename variables in memory"
        )
        self.copy_maintenance_renamed_images_button = QPushButton(
            "Copy matched images with new names"
        )
        maintenance_rename_actions.addWidget(
            self.preview_maintenance_var_rename_button
        )
        maintenance_rename_actions.addWidget(
            self.apply_maintenance_var_rename_button
        )
        maintenance_rename_actions.addWidget(
            self.copy_maintenance_renamed_images_button
        )
        self.maintenance_var_rename_status = QLabel(
            "Enter a new variable name, then preview the synchronized change."
        )
        self.maintenance_var_rename_status.setWordWrap(True)
        maintenance_rename_layout.addWidget(maintenance_rename_help)
        maintenance_rename_layout.addWidget(self.maintenance_var_rename_table)
        maintenance_rename_layout.addWidget(self.maintenance_update_raw_names_check)
        maintenance_rename_layout.addWidget(QLabel("Derived image output folder"))
        maintenance_rename_layout.addWidget(maintenance_image_output)
        maintenance_rename_layout.addLayout(maintenance_rename_actions)
        maintenance_rename_layout.addWidget(self.maintenance_var_rename_status)
        maintenance_channels_layout.addWidget(maintenance_rename_group)

        maintenance_remove_vars_group = workflow_group(
            "3. Remove variables from AnnData",
            "dataset_maintenance",
            "Remove variables",
        )
        maintenance_remove_vars_layout = QVBoxLayout(maintenance_remove_vars_group)
        maintenance_remove_vars_help = QLabel(
            "Selected variables are removed only from the live AnnData. Image files "
            "are deliberately retained and reported as orphan channels."
        )
        maintenance_remove_vars_help.setWordWrap(True)
        self.maintenance_remove_vars_list = QListWidget()
        self.maintenance_remove_vars_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.maintenance_remove_vars_list.setMaximumHeight(180)
        self.maintenance_subset_raw_check = QCheckBox(
            "Also subset adata.raw to variables retained in AnnData"
        )
        self.remove_maintenance_vars_button = QPushButton(
            "Remove selected variables in memory"
        )
        maintenance_remove_vars_layout.addWidget(maintenance_remove_vars_help)
        maintenance_remove_vars_layout.addWidget(self.maintenance_remove_vars_list)
        maintenance_remove_vars_layout.addWidget(self.maintenance_subset_raw_check)
        maintenance_remove_vars_layout.addWidget(self.remove_maintenance_vars_button)
        maintenance_channels_layout.addWidget(maintenance_remove_vars_group)
        maintenance_channels_layout.addStretch(1)
        self.maintenance_tool_tabs.addTab(maintenance_channels_page, "Channels")

        maintenance_cells_page = QWidget()
        maintenance_cells_layout = QVBoxLayout(maintenance_cells_page)
        maintenance_filter_group = workflow_group(
            "2. Filter cells using an AnnData observation",
            "dataset_maintenance",
            "Filter cells",
        )
        maintenance_filter_form = QFormLayout(maintenance_filter_group)
        self.maintenance_filter_obs_combo = QComboBox()
        self.maintenance_filter_mode_combo = QComboBox()
        for label, value in (
            ("Keep selected values", "keep_values"),
            ("Remove selected values", "remove_values"),
            ("Keep numeric range", "keep_range"),
            ("Remove numeric range", "remove_range"),
            ("Keep only missing values", "keep_missing"),
            ("Remove missing values", "remove_missing"),
        ):
            self.maintenance_filter_mode_combo.addItem(label, value)
        self.maintenance_filter_values_list = QListWidget()
        self.maintenance_filter_values_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.maintenance_filter_values_list.setMaximumHeight(160)
        maintenance_filter_range = QWidget()
        maintenance_filter_range_layout = QHBoxLayout(maintenance_filter_range)
        maintenance_filter_range_layout.setContentsMargins(0, 0, 0, 0)
        self.maintenance_filter_lower_spin = QDoubleSpinBox()
        self.maintenance_filter_lower_spin.setRange(-1e15, 1e15)
        self.maintenance_filter_lower_spin.setDecimals(6)
        self.maintenance_filter_upper_spin = QDoubleSpinBox()
        self.maintenance_filter_upper_spin.setRange(-1e15, 1e15)
        self.maintenance_filter_upper_spin.setDecimals(6)
        maintenance_filter_range_layout.addWidget(QLabel("Minimum"))
        maintenance_filter_range_layout.addWidget(
            self.maintenance_filter_lower_spin
        )
        maintenance_filter_range_layout.addWidget(QLabel("Maximum"))
        maintenance_filter_range_layout.addWidget(
            self.maintenance_filter_upper_spin
        )
        maintenance_filter_actions = QWidget()
        maintenance_filter_actions_layout = QHBoxLayout(maintenance_filter_actions)
        maintenance_filter_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_maintenance_filter_button = QPushButton("Preview filter")
        self.apply_maintenance_filter_button = QPushButton(
            "Apply cell filter in memory"
        )
        maintenance_filter_actions_layout.addWidget(
            self.preview_maintenance_filter_button
        )
        maintenance_filter_actions_layout.addWidget(
            self.apply_maintenance_filter_button
        )
        self.maintenance_filter_status = QLabel(
            "Choose an observation and filter mode. Preview before applying."
        )
        self.maintenance_filter_status.setWordWrap(True)
        maintenance_filter_form.addRow("Observation", self.maintenance_filter_obs_combo)
        maintenance_filter_form.addRow("Filter", self.maintenance_filter_mode_combo)
        maintenance_filter_form.addRow(
            "Values", self.maintenance_filter_values_list
        )
        maintenance_filter_form.addRow("Numeric range", maintenance_filter_range)
        maintenance_filter_form.addRow("", maintenance_filter_actions)
        maintenance_filter_form.addRow("Preview", self.maintenance_filter_status)
        maintenance_cells_layout.addWidget(maintenance_filter_group)

        maintenance_masks_group = workflow_group(
            "3. Rebuild masks and align ObjectNumbers",
            "dataset_maintenance",
            "Rebuild masks and ObjectNumbers",
        )
        maintenance_masks_form = QFormLayout(maintenance_masks_group)
        maintenance_masks_explanation = QLabel(
            "This explicitly reads every represented ROI mask. A new mask folder and "
            "identity crosswalk are written. Existing masks are never modified."
        )
        maintenance_masks_explanation.setWordWrap(True)
        self.maintenance_mask_mode_combo = QComboBox()
        self.maintenance_mask_mode_combo.addItem(
            "Preserve retained ObjectNumbers", "preserve"
        )
        self.maintenance_mask_mode_combo.addItem(
            "Compact to 1…N within each ROI", "compact"
        )
        maintenance_mask_output = QWidget()
        maintenance_mask_output_layout = QHBoxLayout(maintenance_mask_output)
        maintenance_mask_output_layout.setContentsMargins(0, 0, 0, 0)
        self.maintenance_mask_output_edit = QLineEdit(
            str(self.project_root / "napari_sbt_rebuilt_masks")
        )
        self.choose_maintenance_mask_output_button = QPushButton("Choose…")
        maintenance_mask_output_layout.addWidget(self.maintenance_mask_output_edit)
        maintenance_mask_output_layout.addWidget(
            self.choose_maintenance_mask_output_button
        )
        maintenance_mask_actions = QWidget()
        maintenance_mask_actions_layout = QHBoxLayout(maintenance_mask_actions)
        maintenance_mask_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_maintenance_masks_button = QPushButton("Validate mask alignment")
        self.apply_maintenance_masks_button = QPushButton(
            "Write derived masks and update memory"
        )
        maintenance_mask_actions_layout.addWidget(
            self.preview_maintenance_masks_button
        )
        maintenance_mask_actions_layout.addWidget(
            self.apply_maintenance_masks_button
        )
        self.maintenance_mask_status = QLabel(
            "Build or refresh the mask index, then validate alignment."
        )
        self.maintenance_mask_status.setWordWrap(True)
        maintenance_masks_form.addRow(maintenance_masks_explanation)
        maintenance_masks_form.addRow("ObjectNumber mode", self.maintenance_mask_mode_combo)
        maintenance_masks_form.addRow("New mask folder", maintenance_mask_output)
        maintenance_masks_form.addRow("", maintenance_mask_actions)
        maintenance_masks_form.addRow("Readiness", self.maintenance_mask_status)
        maintenance_cells_layout.addWidget(maintenance_masks_group)
        maintenance_cells_layout.addStretch(1)
        self.maintenance_tool_tabs.addTab(maintenance_cells_page, "Cells & Masks")

        maintenance_obs_page = QWidget()
        maintenance_obs_layout = QVBoxLayout(maintenance_obs_page)
        maintenance_obs_group = workflow_group(
            "2. Create or remap an observation",
            "dataset_maintenance",
            "Manage observations",
        )
        maintenance_obs_group_layout = QVBoxLayout(maintenance_obs_group)
        maintenance_obs_explanation = QLabel(
            "Create a new categorical obs by renaming or merging values from an "
            "existing obs. The source stays unchanged unless overwrite is enabled."
        )
        maintenance_obs_explanation.setWordWrap(True)
        maintenance_obs_group_layout.addWidget(maintenance_obs_explanation)
        maintenance_obs_form = QFormLayout()
        self.maintenance_obs_combo = QComboBox()
        self.maintenance_obs_output_edit = QLineEdit()
        self.maintenance_obs_output_edit.setPlaceholderText(
            "For example: population_reviewed"
        )
        self.maintenance_obs_overwrite_checkbox = QCheckBox(
            "Allow replacing an existing observation in the live AnnData"
        )
        maintenance_obs_form.addRow(
            "Source observation", self.maintenance_obs_combo
        )
        maintenance_obs_form.addRow(
            "Output observation", self.maintenance_obs_output_edit
        )
        maintenance_obs_form.addRow("Advanced", self.maintenance_obs_overwrite_checkbox)
        maintenance_obs_group_layout.addLayout(maintenance_obs_form)

        self.maintenance_obs_mapping_table = QTableWidget(0, 4)
        self.maintenance_obs_mapping_table.setHorizontalHeaderLabels(
            ["Source value", "Cells", "Proposed name", "Colour"]
        )
        self.maintenance_obs_mapping_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.maintenance_obs_mapping_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.maintenance_obs_mapping_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        self.maintenance_obs_mapping_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeToContents
        )
        self.maintenance_obs_mapping_table.setMinimumHeight(280)
        self.maintenance_obs_mapping_table.setAlternatingRowColors(False)
        self.maintenance_obs_mapping_table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self.maintenance_obs_mapping_table.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        maintenance_obs_group_layout.addWidget(self.maintenance_obs_mapping_table)

        maintenance_mapping_edit_actions = QWidget()
        maintenance_mapping_edit_actions_layout = QHBoxLayout(
            maintenance_mapping_edit_actions
        )
        maintenance_mapping_edit_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.name_selected_maintenance_obs_button = QPushButton(
            "Give selected rows one name / merge"
        )
        self.colour_selected_maintenance_obs_button = QPushButton(
            "Set selected rows' colour"
        )
        self.auto_colour_maintenance_obs_button = QPushButton("Automatically colour…")
        maintenance_mapping_edit_actions_layout.addWidget(
            self.name_selected_maintenance_obs_button
        )
        maintenance_mapping_edit_actions_layout.addWidget(
            self.colour_selected_maintenance_obs_button
        )
        maintenance_mapping_edit_actions_layout.addWidget(
            self.auto_colour_maintenance_obs_button
        )
        maintenance_obs_group_layout.addWidget(maintenance_mapping_edit_actions)

        maintenance_mapping_actions = QWidget()
        maintenance_mapping_actions_layout = QHBoxLayout(maintenance_mapping_actions)
        maintenance_mapping_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.reset_maintenance_obs_mapping_button = QPushButton(
            "Reset names and colours"
        )
        self.apply_maintenance_obs_mapping_button = QPushButton(
            "Create / update observation in memory"
        )
        self.apply_maintenance_obs_mapping_button.setObjectName(
            "sbtPrimaryActionButton"
        )
        maintenance_mapping_actions_layout.addWidget(
            self.reset_maintenance_obs_mapping_button
        )
        maintenance_mapping_actions_layout.addWidget(
            self.apply_maintenance_obs_mapping_button
        )
        maintenance_obs_group_layout.addWidget(maintenance_mapping_actions)

        self.maintenance_obs_status = QLabel(
            "Choose a source observation. Repeated proposed names are explicit merges."
        )
        self.maintenance_obs_status.setWordWrap(True)
        maintenance_obs_group_layout.addWidget(self.maintenance_obs_status)
        maintenance_obs_layout.addWidget(maintenance_obs_group)

        maintenance_obs_utilities_group = workflow_group(
            "3. Observation column utilities",
            "dataset_maintenance",
            "Observation column utilities",
        )
        maintenance_obs_utilities_form = QFormLayout(maintenance_obs_utilities_group)
        self.maintenance_obs_utility_source_label = QLabel("No observation selected")
        self.maintenance_obs_rename_edit = QLineEdit()
        self.maintenance_obs_rename_edit.setPlaceholderText("New column name")
        # Compatibility alias for callers which used the first implementation.
        self.maintenance_obs_new_name_edit = self.maintenance_obs_rename_edit
        maintenance_obs_actions = QWidget()
        maintenance_obs_actions_layout = QHBoxLayout(maintenance_obs_actions)
        maintenance_obs_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.rename_maintenance_obs_button = QPushButton("Rename observation")
        self.remove_maintenance_obs_button = QPushButton("Remove observation")
        self.repair_maintenance_palette_button = QPushButton(
            "Repair categorical colours"
        )
        maintenance_obs_actions_layout.addWidget(self.rename_maintenance_obs_button)
        maintenance_obs_actions_layout.addWidget(self.remove_maintenance_obs_button)
        maintenance_obs_actions_layout.addWidget(
            self.repair_maintenance_palette_button
        )
        maintenance_obs_utilities_form.addRow(
            "Selected source", self.maintenance_obs_utility_source_label
        )
        maintenance_obs_utilities_form.addRow(
            "Rename column to", self.maintenance_obs_rename_edit
        )
        maintenance_obs_utilities_form.addRow("", maintenance_obs_actions)
        maintenance_obs_layout.addWidget(maintenance_obs_utilities_group)
        maintenance_obs_layout.addStretch(1)
        self.maintenance_tool_tabs.addTab(maintenance_obs_page, "Observations")

        maintenance_layout.addWidget(self.maintenance_tool_tabs)
        maintenance_layout.addStretch(1)
        add_tab(maintenance, "🛠️ Dataset Maintenance", "dataset_maintenance")
        self.dataset_maintenance_tab_index = self.tabs.count() - 1

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
            "proposals and will not erase confirmed labels. With the viewer "
            "canvas focused, press a class number key to switch class without "
            "changing the selected click action."
        )
        self.cell_picking_help.setWordWrap(True)
        self.class_combo = QComboBox()
        self.class_hotkey_label = QLabel(
            "Create or load a classification workspace to show class hotkeys."
        )
        self.class_hotkey_label.setWordWrap(True)
        self.class_hotkey_label.setToolTip(
            "Number keys select the current class only. They never change whether "
            "a click selects, proposes, confirms, or clears a proposal."
        )
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
        selection_form.addRow("Hotkeys", self.class_hotkey_label)
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

        integration_group = workflow_group(
            "5. Optional: integrate with existing population labels",
            "classify",
            "Integrate with existing labels",
        )
        integration_form = QFormLayout(integration_group)
        integration_explanation = QLabel(
            "Use this when the classifier subdivides only part of the dataset. "
            "Cells with accepted final classes receive the names below; cells "
            "outside the cohort and unassigned cohort cells retain their existing "
            "source label. Nothing is written to AnnData until the export action."
        )
        integration_explanation.setWordWrap(True)
        self.final_integration_enable_check = QCheckBox(
            "Create a full-dataset integrated population label"
        )
        self.final_integration_source_combo = QComboBox()
        self.final_integration_output_edit = QLineEdit("classified_populations")
        self.final_integration_naming_combo = QComboBox()
        self.final_integration_naming_combo.addItem(
            "Use classification class names", "class_names"
        )
        self.final_integration_naming_combo.addItem(
            "Source population → classification name", "source_and_class"
        )
        self.final_integration_naming_combo.addItem(
            "Define custom final names", "custom"
        )
        self.final_integration_mapping_help = QLabel()
        self.final_integration_mapping_help.setWordWrap(True)
        self.final_integration_mapping_table = QTableWidget(0, 3)
        self.final_integration_mapping_table.setHorizontalHeaderLabels(
            ["Final class", "Colour", "Integrated population name"]
        )
        self.final_integration_mapping_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.final_integration_mapping_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.final_integration_mapping_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        self.final_integration_mapping_table.verticalHeader().setVisible(False)
        self.final_integration_mapping_table.setMaximumHeight(230)
        integration_actions = QWidget()
        integration_actions_layout = QHBoxLayout(integration_actions)
        integration_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_identity_integration_button = QPushButton(
            "Preview overlap / confusion matrix…"
        )
        self.build_identity_integration_button = QPushButton(
            "Build / refresh integrated labels"
        )
        self.build_identity_integration_button.setObjectName("sbtPrimaryActionButton")
        integration_actions_layout.addWidget(self.preview_identity_integration_button)
        integration_actions_layout.addWidget(self.build_identity_integration_button)
        self.identity_integration_summary_label = QLabel(
            "Optional integration is disabled; exports will contain cohort-only "
            "final identities."
        )
        self.identity_integration_summary_label.setWordWrap(True)
        integration_form.addRow(integration_explanation)
        integration_form.addRow("", self.final_integration_enable_check)
        integration_form.addRow(
            "Existing population observation", self.final_integration_source_combo
        )
        integration_form.addRow(
            "New integrated observation", self.final_integration_output_edit
        )
        integration_form.addRow(
            "Population naming", self.final_integration_naming_combo
        )
        integration_form.addRow("Naming preview", self.final_integration_mapping_help)
        integration_form.addRow("Class names", self.final_integration_mapping_table)
        integration_form.addRow("", integration_actions)
        integration_form.addRow(
            "Integration status", self.identity_integration_summary_label
        )
        finalize_page_layout.addWidget(integration_group)

        final_export_group = workflow_group(
            "6. Export final identities",
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
        self.export_assignments_button = QPushButton(
            "Export cohort final identities CSV/Parquet"
        )
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
            QLabel#sbtMaintenanceWarning {
                border: 2px solid #f59e0b;
                border-radius: 7px;
                background-color: #fef3c7;
                color: #78350f;
                padding: 9px;
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
            QTabBar::tab:nth-child(12) { background: #fee2e2; }
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
            "#b91c1c",
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
        self.activity_widget.setMinimumHeight(165)
        self.activity_widget.setStyleSheet(
            "QFrame#sbtActivityPanel { background: rgba(25, 31, 42, 235); "
            "border: 3px solid #22c55e; border-radius: 8px; } "
            "QLabel { color: white; background: transparent; }"
        )
        activity_layout = QVBoxLayout(self.activity_widget)
        activity_layout.setContentsMargins(10, 8, 10, 8)
        activity_layout.setSpacing(3)
        self.activity_title_label = QLabel("✅ Ready")
        self.activity_title_label.setObjectName("sbtActivityTitle")
        self.activity_title_label.setWordWrap(True)
        activity_title_font = QFont(self.activity_title_label.font())
        activity_title_font.setBold(True)
        point_size = activity_title_font.pointSizeF()
        if point_size > 0:
            scaled_point_size = point_size * 2.5
            activity_title_font.setPointSizeF(scaled_point_size)
            self._activity_title_css_size = f"{scaled_point_size:.1f}pt"
        else:
            pixel_size = activity_title_font.pixelSize()
            scaled_pixel_size = round(pixel_size * 2.5) if pixel_size > 0 else 24
            activity_title_font.setPixelSize(scaled_pixel_size)
            self._activity_title_css_size = f"{scaled_pixel_size}px"
        self.activity_title_label.setFont(activity_title_font)
        self.activity_title_label.setStyleSheet(
            f"color: #86efac; font-size: {self._activity_title_css_size}; "
            "font-weight: 900;"
        )
        self.activity_title_label.setMinimumHeight(
            max(40, self.activity_title_label.sizeHint().height() + 6)
        )
        self.activity_action_label = QLabel("Ready for the next action.")
        self.activity_action_label.setWordWrap(True)
        activity_action_font = QFont(self.activity_action_label.font())
        activity_action_font.setBold(True)
        self.activity_action_label.setFont(activity_action_font)
        self.activity_timestamp_label = QLabel()
        self.activity_timestamp_label.setWordWrap(True)
        self.activity_detail_label = QLabel("No active operation.")
        self.activity_detail_label.setWordWrap(True)
        activity_layout.addWidget(self.activity_title_label)
        activity_layout.addWidget(self.activity_action_label)
        activity_layout.addWidget(self.activity_timestamp_label)
        activity_layout.addWidget(self.activity_detail_label)
        self.activity_widget.adjustSize()

        # Passive cell inspection lives in a separate dock beside Readiness. It
        # observes viewer clicks without taking ownership of classifier/Labeler
        # selection state or their transient outline layers.
        self.cell_properties_widget = QFrame()
        self.cell_properties_dock = None
        self.cell_properties_widget.setObjectName("sbtCellPropertiesPanel")
        self.cell_properties_widget.setMinimumWidth(290)
        self.cell_properties_widget.setMinimumHeight(150)
        self.cell_properties_widget.setStyleSheet(
            "QFrame#sbtCellPropertiesPanel { background: rgba(25, 31, 42, 235); "
            "border: 2px solid #22c55e; border-radius: 8px; } "
            "QLabel, QCheckBox { color: white; background: transparent; }"
        )
        cell_properties_layout = QVBoxLayout(self.cell_properties_widget)
        cell_properties_layout.setContentsMargins(10, 7, 10, 7)
        cell_properties_layout.setSpacing(4)
        cell_properties_header = QHBoxLayout()
        self.cell_properties_title_label = QLabel("Cell properties")
        cell_properties_title_font = QFont(self.cell_properties_title_label.font())
        cell_properties_title_font.setBold(True)
        self.cell_properties_title_label.setFont(cell_properties_title_font)
        self.cell_properties_tracking_check = QCheckBox("Track clicks")
        self.cell_properties_tracking_check.setChecked(True)
        self.cell_properties_settings_button = QPushButton("⚙ Settings")
        self.cell_properties_settings_button.setToolTip(
            "Choose cell-level AnnData observations and optional cell outlining."
        )
        cell_properties_header.addWidget(self.cell_properties_title_label)
        cell_properties_header.addStretch(1)
        cell_properties_header.addWidget(self.cell_properties_tracking_check)
        cell_properties_header.addWidget(self.cell_properties_settings_button)
        self.cell_properties_summary_label = QLabel(
            "Load an ROI, then click a cell to inspect it."
        )
        self.cell_properties_summary_label.setWordWrap(True)
        self.cell_properties_tree = QTreeWidget()
        self.cell_properties_tree.setHeaderLabels(["Observation", "Value"])
        self.cell_properties_tree.setRootIsDecorated(False)
        self.cell_properties_tree.setAlternatingRowColors(False)
        self.cell_properties_tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.cell_properties_tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        cell_properties_layout.addLayout(cell_properties_header)
        cell_properties_layout.addWidget(self.cell_properties_summary_label)
        cell_properties_layout.addWidget(self.cell_properties_tree)
        self.cell_properties_widget.adjustSize()

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
        self.refresh_maintenance_readiness()
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
            self._live_recipe_tracking_changed
        )
        self.explore_live_recipe_tracking_check.toggled.connect(
            self._live_recipe_tracking_changed
        )
        self.population_qc_live_recipe_tracking_check.toggled.connect(
            self._live_recipe_tracking_changed
        )
        self.validate_integrity_button.clicked.connect(
            self._guard(self.preview_cohort)
        )
        self.detect_dataset_inputs_button.clicked.connect(
            self._guard(self.detect_dataset_inputs)
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
        self.refresh_maintenance_readiness_button.clicked.connect(
            self._guard(self.refresh_maintenance_readiness)
        )
        self.reindex_maintenance_assets_button.clicked.connect(
            self._guard(self.rebuild_maintenance_asset_index)
        )
        self.choose_maintenance_anndata_button.clicked.connect(
            self._guard(self.choose_maintenance_anndata_destination)
        )
        self.save_maintenance_anndata_button.clicked.connect(
            self._guard(self.save_current_maintenance_anndata)
        )
        self.choose_maintenance_image_output_button.clicked.connect(
            self._guard(self.choose_maintenance_image_output)
        )
        self.preview_maintenance_var_rename_button.clicked.connect(
            self._guard(self.preview_maintenance_var_rename)
        )
        self.apply_maintenance_var_rename_button.clicked.connect(
            self._guard(self.apply_maintenance_var_rename)
        )
        self.copy_maintenance_renamed_images_button.clicked.connect(
            self._guard(self.copy_maintenance_renamed_images)
        )
        self.maintenance_var_rename_table.itemChanged.connect(
            self._guard(
                self._maintenance_var_mapping_changed,
                pass_signal_args=True,
            )
        )
        self.remove_maintenance_vars_button.clicked.connect(
            self._guard(self.remove_selected_maintenance_vars)
        )
        self.maintenance_filter_obs_combo.currentTextChanged.connect(
            self._guard(self.refresh_maintenance_filter_values)
        )
        self.maintenance_filter_mode_combo.currentIndexChanged.connect(
            self._guard(self.update_maintenance_filter_controls)
        )
        self.maintenance_filter_values_list.itemSelectionChanged.connect(
            self.update_maintenance_filter_controls
        )
        self.maintenance_filter_lower_spin.valueChanged.connect(
            self.update_maintenance_filter_controls
        )
        self.maintenance_filter_upper_spin.valueChanged.connect(
            self.update_maintenance_filter_controls
        )
        self.preview_maintenance_filter_button.clicked.connect(
            self._guard(self.preview_maintenance_cell_filter)
        )
        self.apply_maintenance_filter_button.clicked.connect(
            self._guard(self.apply_maintenance_cell_filter)
        )
        self.choose_maintenance_mask_output_button.clicked.connect(
            self._guard(self.choose_maintenance_mask_output)
        )
        self.preview_maintenance_masks_button.clicked.connect(
            self._guard(self.preview_maintenance_mask_rebuild)
        )
        self.apply_maintenance_masks_button.clicked.connect(
            self._guard(self.apply_maintenance_mask_rebuild)
        )
        self.maintenance_obs_combo.currentTextChanged.connect(
            self._guard(self.refresh_maintenance_observation_mapping)
        )
        self.maintenance_obs_mapping_table.itemChanged.connect(
            self._guard(
                self._maintenance_observation_mapping_changed,
                pass_signal_args=True,
            )
        )
        self.reset_maintenance_obs_mapping_button.clicked.connect(
            self._guard(self.refresh_maintenance_observation_mapping)
        )
        self.auto_colour_maintenance_obs_button.clicked.connect(
            self._guard(self.auto_colour_maintenance_observation)
        )
        self.name_selected_maintenance_obs_button.clicked.connect(
            self._guard(self.name_selected_maintenance_observation_rows)
        )
        self.colour_selected_maintenance_obs_button.clicked.connect(
            self._guard(self.colour_selected_maintenance_observation_rows)
        )
        self.apply_maintenance_obs_mapping_button.clicked.connect(
            self._guard(self.apply_maintenance_observation_mapping)
        )
        self.maintenance_obs_output_edit.textChanged.connect(
            self._maintenance_observation_mapping_changed
        )
        self.maintenance_obs_overwrite_checkbox.stateChanged.connect(
            self._maintenance_observation_mapping_changed
        )
        self.rename_maintenance_obs_button.clicked.connect(
            self._guard(self.rename_maintenance_observation)
        )
        self.remove_maintenance_obs_button.clicked.connect(
            self._guard(self.remove_maintenance_observation)
        )
        self.repair_maintenance_palette_button.clicked.connect(
            self._guard(self.repair_maintenance_observation_palette)
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
        self.refresh_feature_readiness_button.clicked.connect(
            self._guard(self.refresh_saved_feature_status)
        )
        self.feature_tables_edit.textChanged.connect(self.refresh_feature_readiness)
        self.anndata_features_edit.textChanged.connect(
            self.refresh_feature_readiness
        )
        self.offset_spin.valueChanged.connect(self.refresh_feature_readiness)
        self.offset_overlap_check.toggled.connect(self.refresh_feature_readiness)
        self.background_ring_spin.valueChanged.connect(
            self.refresh_feature_readiness
        )
        self.refresh_feature_channels_button.clicked.connect(
            self._guard(self.refresh_feature_channel_choices)
        )
        self.select_built_feature_channels_button.clicked.connect(
            self._guard(self.select_feature_build_channels)
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
            self._guard(self._roi_selection_changed, pass_signal_args=True)
        )
        self.reload_roi_button.clicked.connect(self._guard(self.load_roi))
        self.add_all_cells_mask_button.clicked.connect(
            self._guard(self.add_all_cells_mask)
        )
        self.publication_export_button.clicked.connect(
            self._guard(self.show_publication_export)
        )
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
            self._guard(
                self.auto_reload_setting_changed,
                pass_signal_args=True,
            )
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
        self.select_feature_marker_overlays_button.clicked.connect(
            self._guard(self.select_feature_marker_overlays)
        )
        self.rank_marker_rois_button.clicked.connect(
            self._guard(self.rank_rois_by_marker)
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
        self.select_feature_image_channels_button.clicked.connect(
            self._guard(self.select_feature_image_channels)
        )
        self.load_six_colour_button.clicked.connect(
            self._guard(self.load_six_colour_channels)
        )
        self.load_rgb_button.clicked.connect(self._guard(self.load_rgb))
        self.scanpy_plotting_panel.select_feature_markers_button.clicked.connect(
            self._guard(self.select_scanpy_expression_feature_markers)
        )
        scanpy_embedding_feature_button = (
            self.scanpy_plotting_panel.select_feature_embedding_markers_button
        )
        scanpy_embedding_feature_button.clicked.connect(
            self._guard(self.select_scanpy_embedding_feature_markers)
        )
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
            self._guard(self._population_tables_changed, pass_signal_args=True)
        )
        self.population_components_table.itemChanged.connect(
            self._guard(self._population_tables_changed, pass_signal_args=True)
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
        self.auto_colour_populations_button.clicked.connect(
            self._guard(self.auto_colour_population_draft)
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
        self.final_integration_enable_check.toggled.connect(
            self._identity_integration_enabled_changed
        )
        self.final_integration_source_combo.currentTextChanged.connect(
            self._mark_identity_integration_stale
        )
        self.final_integration_output_edit.textChanged.connect(
            self._mark_identity_integration_stale
        )
        self.final_integration_naming_combo.currentIndexChanged.connect(
            self._identity_integration_naming_changed
        )
        self.final_integration_mapping_table.itemChanged.connect(
            self._identity_integration_mapping_changed
        )
        self.preview_identity_integration_button.clicked.connect(
            self._guard(self.preview_identity_integration)
        )
        self.build_identity_integration_button.clicked.connect(
            self._guard(self.build_identity_integration)
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
        self.cell_properties_settings_button.clicked.connect(
            self.show_cell_properties_settings
        )
        self.cell_properties_tracking_check.toggled.connect(
            self.set_cell_properties_tracking
        )
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
            if monitor_ready and self.publication_batch is not None:
                self.activity_waiting_for_process = True
                self._activity_update("Publication bulk export is running.")
            elif monitor_ready and self._active_background_processes():
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
        self.activity_state_changed_at = datetime.now().astimezone()
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
        self.activity_state_changed_at = datetime.now().astimezone()
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

    def install_cell_properties_dock(self):
        """Install passive cell inspection in its own movable Napari dock."""

        if self.cell_properties_dock is not None:
            return self.cell_properties_dock
        add_dock_widget = getattr(
            getattr(self.viewer, "window", None), "add_dock_widget", None
        )
        if not callable(add_dock_widget):
            return None
        self.cell_properties_dock = add_dock_widget(
            self.cell_properties_widget,
            name="NapariSBT Cell properties",
            area="left",
            add_vertical_stretch=False,
            tabify=False,
        )
        return self.cell_properties_dock

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
            [10000, 190],
            orientation,
        )

    def _position_auxiliary_docks(self) -> None:
        """Place Readiness and Cell properties side-by-side beneath Layers."""

        self._position_readiness_dock()
        if self.activity_dock is None or self.cell_properties_dock is None:
            return
        qt_window = getattr(getattr(self.viewer, "window", None), "_qt_window", None)
        if qt_window is None:
            return
        horizontal = getattr(getattr(self.Qt, "Orientation", self.Qt), "Horizontal")
        qt_window.splitDockWidget(
            self.activity_dock,
            self.cell_properties_dock,
            horizontal,
        )
        qt_window.resizeDocks(
            [self.activity_dock, self.cell_properties_dock],
            [260, 360],
            horizontal,
        )

    def _cell_property_candidates(self) -> list[str]:
        """Return cell-varying obs fields, excluding identity and ROI metadata."""

        if self.adata is None:
            return []
        roi_obs, object_obs = self._maintenance_identity_columns()
        if not roi_obs or not object_obs or roi_obs not in self.adata.obs:
            return []
        return cell_level_observations(
            self.adata.obs,
            roi_obs=roi_obs,
            object_obs=object_obs,
        )

    def _default_cell_property_observations(
        self, candidates: Iterable[str]
    ) -> list[str]:
        """Choose a compact, label-focused first view for non-technical users."""

        candidates = list(candidates)
        if self.adata is None:
            return candidates[:8]
        label_terms = (
            "population",
            "cell_type",
            "celltype",
            "phenotype",
            "leiden",
            "cluster",
            "class",
            "label",
        )
        ranked: list[tuple[int, int, str]] = []
        for index, observation in enumerate(candidates):
            series = self.adata.obs[observation]
            name = observation.casefold()
            term_rank = next(
                (
                    term_index
                    for term_index, term in enumerate(label_terms)
                    if term in name
                ),
                len(label_terms),
            )
            categorical = isinstance(series.dtype, pd.CategoricalDtype)
            categorical = categorical or pd.api.types.is_bool_dtype(series.dtype)
            priority = term_rank if term_rank < len(label_terms) else len(label_terms)
            if not categorical and term_rank == len(label_terms):
                priority += len(label_terms)
            ranked.append((priority, index, observation))
        return [observation for _priority, _index, observation in sorted(ranked)[:8]]

    def _refresh_cell_properties_available_observations(self) -> None:
        """Reconcile saved property choices with the live AnnData schema."""

        if not hasattr(self, "cell_properties_tracking_check"):
            return
        candidates = self._cell_property_candidates()
        if candidates and not self.explore_review_state.cell_properties_configured:
            self.explore_review_state.cell_properties_observations = (
                self._default_cell_property_observations(candidates)
            )
            self.explore_review_state.cell_properties_configured = True
            self._save_explore_review_state()
        self._cell_properties_colour_maps.clear()
        self._sync_cell_properties_controls()
        if self.cell_properties_settings_dialog is not None:
            self._populate_cell_properties_settings_list()

    def _sync_cell_properties_controls(self) -> None:
        """Copy persisted inspection settings into any currently built widgets."""

        if not hasattr(self, "cell_properties_tracking_check"):
            return
        blocked = self.cell_properties_tracking_check.blockSignals(True)
        self.cell_properties_tracking_check.setChecked(
            bool(self.explore_review_state.cell_properties_tracking_enabled)
        )
        self.cell_properties_tracking_check.blockSignals(blocked)
        if self.cell_properties_settings_dialog is None:
            return
        self.cell_properties_settings_tracking.setChecked(
            bool(self.explore_review_state.cell_properties_tracking_enabled)
        )
        self.cell_properties_settings_outline.setChecked(
            bool(self.explore_review_state.cell_properties_outline_enabled)
        )
        self.cell_properties_settings_outline_width.setValue(
            int(self.explore_review_state.cell_properties_outline_width)
        )
        self._set_cell_properties_colour_button(
            self.explore_review_state.cell_properties_outline_colour
        )

    def _populate_cell_properties_settings_list(self) -> None:
        candidates = self._cell_property_candidates()
        selected = set(self.explore_review_state.cell_properties_observations)
        self.cell_properties_settings_list.clear()
        for observation in candidates:
            item = self.QListWidgetItem(observation)
            item.setFlags(item.flags() | self.Qt.ItemIsUserCheckable)
            item.setCheckState(
                self.Qt.Checked if observation in selected else self.Qt.Unchecked
            )
            item.setToolTip(f"adata.obs[{observation!r}]")
            self.cell_properties_settings_list.addItem(item)
        self.cell_properties_settings_count.setText(
            f"{len(candidates):,} cell-level fields available. ROI-level fields and "
            "identity columns are excluded automatically."
        )

    def show_cell_properties_settings(self) -> None:
        """Show property selection, tracking, and optional outline controls."""

        if self.cell_properties_settings_dialog is None:
            from qtpy.QtWidgets import (
                QCheckBox,
                QDialogButtonBox,
                QFormLayout,
                QHBoxLayout,
                QLabel,
                QListWidget,
                QPushButton,
                QSpinBox,
                QVBoxLayout,
            )

            dialog = self.QDialog(self.root)
            dialog.setWindowTitle("Cell properties settings")
            dialog.resize(560, 620)
            layout = QVBoxLayout(dialog)
            explanation = QLabel(
                "Choose the cell-level AnnData observations shown after a tissue "
                "click. This passive inspector does not replace classifier or "
                "Labeler click actions."
            )
            explanation.setWordWrap(True)
            self.cell_properties_settings_tracking = QCheckBox(
                "Track left-clicked cells"
            )
            self.cell_properties_settings_outline = QCheckBox(
                "Outline the inspected cell"
            )
            outline_form = QFormLayout()
            self.cell_properties_settings_outline_colour = QPushButton()
            self.cell_properties_settings_outline_colour.clicked.connect(
                self._choose_cell_properties_outline_colour
            )
            self.cell_properties_settings_outline_width = QSpinBox()
            self.cell_properties_settings_outline_width.setRange(1, 20)
            self.cell_properties_settings_outline_width.setSuffix(" px")
            outline_form.addRow(
                "Outline colour", self.cell_properties_settings_outline_colour
            )
            outline_form.addRow(
                "Outline width", self.cell_properties_settings_outline_width
            )
            self.cell_properties_settings_count = QLabel()
            self.cell_properties_settings_count.setWordWrap(True)
            self.cell_properties_settings_list = QListWidget()
            list_actions = QHBoxLayout()
            recommended_button = QPushButton("Recommended labels")
            select_all_button = QPushButton("Select all")
            clear_button = QPushButton("Clear selection")
            recommended_button.clicked.connect(
                self._select_recommended_cell_properties
            )
            select_all_button.clicked.connect(
                lambda: self._set_all_cell_property_checks(True)
            )
            clear_button.clicked.connect(
                lambda: self._set_all_cell_property_checks(False)
            )
            list_actions.addWidget(recommended_button)
            list_actions.addWidget(select_all_button)
            list_actions.addWidget(clear_button)
            buttons = QDialogButtonBox(
                QDialogButtonBox.Apply | QDialogButtonBox.Close
            )
            buttons.button(QDialogButtonBox.Apply).clicked.connect(
                self.apply_cell_properties_settings
            )
            buttons.rejected.connect(dialog.close)
            layout.addWidget(explanation)
            layout.addWidget(self.cell_properties_settings_tracking)
            layout.addWidget(self.cell_properties_settings_outline)
            layout.addLayout(outline_form)
            layout.addWidget(self.cell_properties_settings_count)
            layout.addWidget(self.cell_properties_settings_list)
            layout.addLayout(list_actions)
            layout.addWidget(buttons)
            self.cell_properties_settings_dialog = dialog
        self._sync_cell_properties_controls()
        self._populate_cell_properties_settings_list()
        self.cell_properties_settings_dialog.show()
        self.cell_properties_settings_dialog.raise_()
        self.cell_properties_settings_dialog.activateWindow()

    def _set_all_cell_property_checks(self, checked: bool) -> None:
        state = self.Qt.Checked if checked else self.Qt.Unchecked
        for index in range(self.cell_properties_settings_list.count()):
            self.cell_properties_settings_list.item(index).setCheckState(state)

    def _select_recommended_cell_properties(self) -> None:
        recommended = set(
            self._default_cell_property_observations(
                self._cell_property_candidates()
            )
        )
        for index in range(self.cell_properties_settings_list.count()):
            item = self.cell_properties_settings_list.item(index)
            item.setCheckState(
                self.Qt.Checked if item.text() in recommended else self.Qt.Unchecked
            )

    def _set_cell_properties_colour_button(self, colour: str) -> None:
        colour = self.QColor(str(colour))
        if not colour.isValid():
            colour = self.QColor("#facc15")
        value = colour.name()
        self.cell_properties_settings_outline_colour.setProperty("colour", value)
        self.cell_properties_settings_outline_colour.setText(value)
        self.cell_properties_settings_outline_colour.setStyleSheet(
            f"background-color: {value}; color: {contrasting_text_colour(value)};"
        )

    def _choose_cell_properties_outline_colour(self) -> None:
        current = self.cell_properties_settings_outline_colour.property("colour")
        colour = self.QColorDialog.getColor(
            self.QColor(str(current or "#facc15")),
            self.root,
            "Choose inspected-cell outline colour",
        )
        if colour.isValid():
            self._set_cell_properties_colour_button(colour.name())

    def apply_cell_properties_settings(self) -> None:
        selected = [
            self.cell_properties_settings_list.item(index).text()
            for index in range(self.cell_properties_settings_list.count())
            if self.cell_properties_settings_list.item(index).checkState()
            == self.Qt.Checked
        ]
        payload = self.explore_review_state.model_dump(mode="json")
        payload.update(
            {
                "cell_properties_configured": True,
                "cell_properties_tracking_enabled": bool(
                    self.cell_properties_settings_tracking.isChecked()
                ),
                "cell_properties_observations": selected,
                "cell_properties_outline_enabled": bool(
                    self.cell_properties_settings_outline.isChecked()
                ),
                "cell_properties_outline_colour": str(
                    self.cell_properties_settings_outline_colour.property("colour")
                    or "#facc15"
                ),
                "cell_properties_outline_width": int(
                    self.cell_properties_settings_outline_width.value()
                ),
            }
        )
        self.explore_review_state = ExploreReviewState.model_validate(payload)
        self._cell_properties_colour_maps.clear()
        self._sync_cell_properties_controls()
        self._save_explore_review_state()
        if self.explore_review_state.cell_properties_tracking_enabled:
            if self.cell_properties_selected_object is not None:
                self._show_cell_properties_for_object(
                    self.cell_properties_selected_object
                )
            else:
                self.cell_properties_summary_label.setText(
                    "Tracking enabled. Click a segmented cell to inspect it."
                )
        else:
            self._remove_layers([CELL_PROPERTIES_SELECTED_LAYER_NAME])
            self.cell_properties_summary_label.setText(
                "Tracking paused. Enable Track clicks here or in Settings."
            )
        self.set_status(
            f"Cell properties now tracks {len(selected):,} cell-level obs field(s)."
        )

    def set_cell_properties_tracking(self, enabled: bool) -> None:
        """Enable or pause passive click inspection without changing other tools."""

        payload = self.explore_review_state.model_dump(mode="json")
        payload["cell_properties_configured"] = True
        payload["cell_properties_tracking_enabled"] = bool(enabled)
        self.explore_review_state = ExploreReviewState.model_validate(payload)
        if self.cell_properties_settings_dialog is not None:
            self.cell_properties_settings_tracking.setChecked(bool(enabled))
        if enabled:
            self.cell_properties_summary_label.setText(
                "Tracking enabled. Click a segmented cell to inspect it."
            )
            if self.cell_properties_selected_object is not None:
                self._show_cell_properties_for_object(
                    self.cell_properties_selected_object
                )
        else:
            self._remove_layers([CELL_PROPERTIES_SELECTED_LAYER_NAME])
            self.cell_properties_summary_label.setText(
                "Tracking paused. Enable Track clicks here or in Settings."
            )
        self._save_explore_review_state()

    def _cell_property_position(self, object_id: int) -> int | None:
        """Resolve one current-ROI mask ID to an AnnData integer position."""

        if self.adata is None or self.current_roi is None:
            return None
        roi = str(self.current_roi)
        cached = self._cell_properties_position_index.get(roi)
        if cached is None:
            roi_obs, object_obs = self._maintenance_identity_columns()
            if roi_obs not in self.adata.obs or object_obs not in self.adata.obs:
                return None
            if not self._adata_roi_positions:
                groups = self.adata.obs.groupby(
                    self.adata.obs[roi_obs].astype(str),
                    sort=False,
                    observed=True,
                ).indices
                self._adata_roi_positions = {
                    str(group_roi): np.asarray(positions, dtype=np.int64)
                    for group_roi, positions in groups.items()
                }
            positions = self._adata_roi_positions.get(
                roi, np.empty(0, dtype=np.int64)
            )
            object_ids = pd.to_numeric(
                self.adata.obs.iloc[positions][object_obs], errors="coerce"
            )
            cached = {
                int(value): int(position)
                for position, value in zip(positions, object_ids)
                if pd.notna(value) and int(value) > 0
            }
            self._cell_properties_position_index[roi] = cached
        return cached.get(int(object_id))

    def _cell_property_colour(self, observation: str, value) -> str | None:
        if self.adata is None or observation not in self.adata.obs:
            return None
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            missing = False
        if not isinstance(missing, (bool, np.bool_)) or bool(missing):
            return None
        series = self.adata.obs[observation]
        dtype = series.dtype
        categorical = isinstance(dtype, pd.CategoricalDtype)
        categorical = categorical or pd.api.types.is_bool_dtype(dtype)
        categorical = categorical or pd.api.types.is_string_dtype(dtype)
        categorical = categorical or pd.api.types.is_object_dtype(dtype)
        if not categorical:
            return None
        if observation not in self._cell_properties_colour_maps:
            self._cell_properties_colour_maps[observation] = categorical_colour_map(
                self.adata, observation
            )
        return self._cell_properties_colour_maps[observation].get(str(value))

    def _show_cell_properties_for_object(self, object_id: int) -> None:
        """Display selected obs values and an optional independent cell outline."""

        if not self.explore_review_state.cell_properties_tracking_enabled:
            return
        self.cell_properties_tree.clear()
        if object_id <= 0:
            self.cell_properties_selected_object = None
            self._remove_layers([CELL_PROPERTIES_SELECTED_LAYER_NAME])
            self.cell_properties_summary_label.setText(
                "Background selected. Click inside a segmented cell."
            )
            return
        self.cell_properties_selected_object = int(object_id)
        self._refresh_cell_properties_outline()
        position = self._cell_property_position(object_id)
        if position is None:
            self.cell_properties_summary_label.setText(
                f"ROI {self.current_roi} / object {object_id}: no matching AnnData row."
            )
            return
        row = self.adata.obs.iloc[position]
        observations = [
            observation
            for observation in self.explore_review_state.cell_properties_observations
            if observation in self.adata.obs
        ]
        self.cell_properties_summary_label.setText(
            f"{self.current_roi} / object {object_id} / AnnData cell "
            f"{self.adata.obs_names[position]}"
        )
        if not observations:
            self.cell_properties_tree.addTopLevelItem(
                self.QTreeWidgetItem(
                    ["No fields selected", "Open Settings to choose cell-level obs"]
                )
            )
            return
        for observation in observations:
            value = row[observation]
            item = self.QTreeWidgetItem(
                [observation, format_roi_metadata_value(value)]
            )
            colour = self._cell_property_colour(observation, value)
            if colour:
                item.setBackground(1, self.QColor(colour))
                item.setForeground(
                    1, self.QColor(contrasting_text_colour(colour))
                )
            self.cell_properties_tree.addTopLevelItem(item)

    def _refresh_cell_properties_outline(self) -> None:
        state = self.explore_review_state
        if (
            not state.cell_properties_tracking_enabled
            or not state.cell_properties_outline_enabled
            or self.current_mask is None
            or self.cell_properties_selected_object is None
        ):
            self._remove_layers([CELL_PROPERTIES_SELECTED_LAYER_NAME])
            return
        selected = (
            self.current_mask == int(self.cell_properties_selected_object)
        ).astype(np.uint8)
        active_layer = self.viewer.layers.selection.active
        layer = self._replace_layer(
            CELL_PROPERTIES_SELECTED_LAYER_NAME,
            selected,
            "labels",
            colormap=self._direct_label_colormap(
                {1: state.cell_properties_outline_colour}
            ),
            visible=True,
            opacity=1.0,
        )
        if hasattr(layer, "contour"):
            layer.contour = int(state.cell_properties_outline_width)
        if hasattr(layer, "editable"):
            layer.editable = False
        source = self.viewer.layers.index(layer)
        if source != len(self.viewer.layers) - 1:
            self.viewer.layers.move(source, len(self.viewer.layers))
        if (
            active_layer is not None
            and str(getattr(active_layer, "name", "")) in self.viewer.layers
        ):
            self.viewer.layers.selection.active = active_layer

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
        state_presentation = {
            "idle": ("✅ Ready", "#86efac", "#22c55e", "Ready since"),
            "running": ("⏳ Working", "#fde047", "#f59e0b", "Started"),
            "complete": ("🏁 Finished", "#93c5fd", "#3b82f6", "Finished"),
            "error": ("❌ Failed", "#fca5a5", "#ef4444", "Failed"),
        }
        title, title_colour, border_colour, timestamp_prefix = state_presentation.get(
            self.activity_state, state_presentation["idle"]
        )
        if self._activity_styled_state != self.activity_state:
            self.activity_widget.setStyleSheet(
                "QFrame#sbtActivityPanel { background: rgba(25, 31, 42, 235); "
                f"border: 3px solid {border_colour}; border-radius: 8px; }} "
                "QLabel { color: white; background: transparent; }"
            )
            self._activity_styled_state = self.activity_state
        self.activity_title_label.setText(title)
        self.activity_title_label.setStyleSheet(
            f"color: {title_colour}; font-size: {self._activity_title_css_size}; "
            "font-weight: 900;"
        )
        timestamp = self.activity_state_changed_at.strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        )
        self.activity_timestamp_label.setText(f"{timestamp_prefix}: {timestamp}")
        self.activity_timestamp_label.setStyleSheet("color: #cbd5e1;")
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
            self.activity_action_label.setText(
                f"{self.activity_action} — elapsed {elapsed:.0f}s"
            )
            self.activity_detail_label.setText(
                f"{self.activity_detail}{process_text}\nHeartbeat: live"
            )
        elif self.activity_state == "error":
            self.activity_action_label.setText(self.activity_action)
            self.activity_detail_label.setText(self.activity_detail)
        elif self.activity_state == "complete":
            self.activity_action_label.setText(self.activity_action)
            self.activity_detail_label.setText(self.activity_detail)
            if (
                self.activity_finished_at is not None
                and time.monotonic() - self.activity_finished_at > 8
            ):
                self.activity_state = "idle"
                self.activity_action = "Ready"
                self.activity_detail = "No active operation."
                self.activity_started_at = None
                self.activity_finished_at = None
                self.activity_state_changed_at = datetime.now().astimezone()
                self._update_activity_monitor()
                return
        else:
            self.activity_action_label.setText("Ready for the next action.")
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
        help_font = browser.font()
        current_size = help_font.pointSizeF()
        help_font.setPointSizeF(max(13.0, current_size + 3.0))
        dialog.setFont(help_font)
        browser.setFont(help_font)
        browser.document().setDefaultFont(help_font)
        browser.document().setDefaultStyleSheet(
            "body { line-height: 1.4; } "
            "h1 { font-size: 21pt; margin-top: 8px; margin-bottom: 12px; } "
            "h2 { font-size: 18pt; margin-top: 12px; margin-bottom: 8px; } "
            "h3 { font-size: 15pt; margin-top: 10px; margin-bottom: 6px; } "
            "li { margin-bottom: 5px; } "
            "code, pre { font-size: 12pt; }"
        )
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
        if replace_inputs:
            self._updating_setup_controls = True
            try:
                if self._in_memory_adata is None:
                    self.anndata_edit.clear()
                self.masks_edit.clear()
                self.images_edit.clear()
                self.extra_images_edit.clear()
                self.normalization_edit.clear()
            finally:
                self._updating_setup_controls = False
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
            self._launch_experiment_was_explicit = False
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
        self._invalidate_dataset_indexes()
        if not self._launch_experiment_was_explicit:
            self.detect_dataset_inputs(announce=False)
        self._loaded_workspace_root = None
        if replace_inputs or self._launch_experiment is None:
            self._update_suggested_workspace_path(force=True)
        self.refresh_workspace_choices()

    def detect_dataset_inputs(self, *, announce: bool = True) -> None:
        """Fill missing Setup inputs using a shallow, convention-based lookup."""

        suggestions = discover_dataset_assets(self.project_root)
        updates: list[str] = []
        notices: list[str] = []

        current_anndata = self.anndata_edit.text().strip()
        anndata_is_usable = bool(
            self._in_memory_adata is not None
            or (current_anndata and Path(current_anndata).expanduser().is_file())
        )
        if not anndata_is_usable and suggestions.anndata_candidates:
            selected_anndata: Path | None = None
            if len(suggestions.anndata_candidates) == 1:
                selected_anndata = suggestions.anndata_candidates[0]
            else:
                display_to_path: dict[str, Path] = {}
                for path in suggestions.anndata_candidates:
                    try:
                        display = str(path.relative_to(self.project_root))
                    except ValueError:
                        display = str(path)
                    display_to_path[display] = path
                selected, accepted = self.QInputDialog.getItem(
                    self.root,
                    "Choose processed cell data",
                    (
                        f"NapariSBT found {len(display_to_path)} AnnData files. "
                        "Select the one to use:"
                    ),
                    list(display_to_path),
                    0,
                    False,
                )
                if accepted and selected:
                    selected_anndata = display_to_path[str(selected)]
                else:
                    notices.append(
                        f"{len(display_to_path)} AnnData files found; selection is required"
                    )
            if selected_anndata is not None:
                self.anndata_edit.setText(str(selected_anndata))
                updates.append(f"AnnData: {selected_anndata.name}")

        current_masks = self.masks_edit.text().strip()
        masks_are_usable = bool(
            current_masks and Path(current_masks).expanduser().is_dir()
        )
        if not masks_are_usable:
            if len(suggestions.masks_candidates) == 1:
                selected_masks = suggestions.masks_candidates[0]
                self.masks_edit.setText(str(selected_masks))
                updates.append(f"masks: {selected_masks.name}")
            elif len(suggestions.masks_candidates) > 1:
                notices.append(
                    f"{len(suggestions.masks_candidates)} possible mask folders found; choose one"
                )

        current_images = _split_paths(self.images_edit.toPlainText())
        images_are_usable = bool(
            current_images
            and all(Path(path).expanduser().is_dir() for path in current_images)
        )
        if not images_are_usable and suggestions.image_candidates:
            self.images_edit.setPlainText(
                "\n".join(map(str, suggestions.image_candidates))
            )
            updates.append(
                f"{len(suggestions.image_candidates)} staining image folder(s)"
            )

        if updates or notices:
            message = "Automatic detection: " + "; ".join([*updates, *notices]) + "."
        else:
            message = (
                "Automatic detection found no missing conventional dataset inputs."
            )
        self.integrity_status_label.setText(
            message
            + " Review the choices, then run the explicit dataset integrity check."
        )
        self.refresh_setup_readiness()
        if announce:
            self.set_status(message)

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
        self.current_image_paths.clear()
        self._invalidate_population_qc_caches()
        self._refresh_roi_metadata_display()
        self._integrity_signature = None
        self._asset_index_signature = None
        self._mask_path_index.clear()
        self._roi_image_path_index.clear()
        self.channel_list.clear()
        self._clear_explore_layers()
        self._remove_layers(
            [
                ALL_CELLS_LAYER_NAME,
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
        scope_index = self.scope_combo.findData("all_cells")
        self.scope_combo.blockSignals(True)
        self.scope_combo.setCurrentIndex(max(0, scope_index))
        self.scope_combo.blockSignals(False)
        self.value_list.blockSignals(True)
        self.value_list.clearSelection()
        self.value_list.blockSignals(False)
        self._update_scope_widget_state()
        self.preview_text.clear()
        self.integrity_status_label.setText(
            "Not yet checked for this new workspace. Run the dataset integrity "
            "check after reviewing the inputs and cell scope."
        )
        self._sync_population_qc_contour_control()
        self._sync_population_qc_contrast_defaults(force=True)
        self.refresh_population_qc_populations()
        self._refresh_population_qc_scope_banner()
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
            self.detect_dataset_inputs_button,
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

    def _sync_recipe_tracking_controls(self, enabled: bool) -> None:
        """Keep the three session controls consistent without signal feedback."""

        for name in (
            "live_recipe_tracking_check",
            "explore_live_recipe_tracking_check",
            "population_qc_live_recipe_tracking_check",
        ):
            control = getattr(self, name, None)
            if control is None:
                continue
            control.blockSignals(True)
            control.setChecked(bool(enabled))
            control.blockSignals(False)

    def _live_recipe_tracking_changed(self, enabled: bool) -> None:
        self._sync_recipe_tracking_controls(bool(enabled))
        if enabled:
            for layer in self.viewer.layers:
                self._bind_recipe_display_tracking(layer)
            self._refresh_reload_recipe_list()
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
        self._cell_properties_position_index.clear()
        self._cell_properties_colour_maps.clear()
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
            "schema_version": ASSET_INDEX_SCHEMA_VERSION,
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
                "schema_version": ASSET_INDEX_SCHEMA_VERSION,
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
        if payload.get("schema_version") != ASSET_INDEX_SCHEMA_VERSION:
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
            self._sync_recipe_tracking_controls(mode != "population_qc")
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
        self._refresh_population_qc_scope_banner()
        self.refresh_setup_readiness()

    def _normalization_from_editor(
        self,
    ) -> dict[str, NimbusNormalizationParameters]:
        payload: dict[str, object] = {}
        for row in range(self.normalization_table.rowCount()):
            marker_item = self.normalization_table.item(row, 0)
            vmax_item = self.normalization_table.item(row, 1)
            lower_item = self.normalization_table.item(row, 2)
            marker = marker_item.text().strip() if marker_item is not None else ""
            vmax_text = vmax_item.text().strip() if vmax_item is not None else ""
            lower_text = lower_item.text().strip() if lower_item is not None else ""
            if not marker and not vmax_text and not lower_text:
                continue
            if not marker or not vmax_text:
                raise ValueError(
                    f"Normalization row {row + 1} requires Marker and Vmax."
                )
            if marker in payload:
                raise ValueError(f"Normalization marker {marker!r} is duplicated.")
            payload[marker] = {
                "vmax": vmax_text,
                "lower_threshold": lower_text or 0.0,
            }
        return prepare_normalization_parameters(payload)

    def _set_normalization_table(self, mapping: dict[str, object]) -> None:
        parameters = prepare_normalization_parameters(mapping)
        self.normalization_table.blockSignals(True)
        try:
            self.normalization_table.setRowCount(len(parameters))
            for row, (marker, entry) in enumerate(sorted(parameters.items())):
                self.normalization_table.setItem(
                    row, 0, self.QTableWidgetItem(str(marker))
                )
                self.normalization_table.setItem(
                    row, 1, self.QTableWidgetItem(f"{entry.vmax:g}")
                )
                self.normalization_table.setItem(
                    row,
                    2,
                    self.QTableWidgetItem(f"{entry.lower_threshold:g}"),
                )
        finally:
            self.normalization_table.blockSignals(False)
        self._sync_normalization_json_preview()

    def add_normalization_row(self) -> None:
        row = self.normalization_table.rowCount()
        self.normalization_table.insertRow(row)
        self.normalization_table.setItem(row, 0, self.QTableWidgetItem(""))
        self.normalization_table.setItem(row, 1, self.QTableWidgetItem(""))
        self.normalization_table.setItem(row, 2, self.QTableWidgetItem("0"))
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
            payload = normalization_parameters_payload(
                self._normalization_from_editor()
            )
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
        try:
            self.display_normalization = load_normalization_parameters(source)
        except ValueError:
            # Early Setup workspaces persisted an empty editor as
            # {"normalization_dict": {}} and then failed while reopening it.
            # Treat only that known, valid NapariSBT placeholder as "not set";
            # malformed or empty user-supplied normalization files still fail.
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise
            if payload != {"normalization_dict": {}}:
                raise
            self.display_normalization = {}
        self._set_normalization_table(self.display_normalization)
        self._clear_explore_layer_data_cache()
        if self.display_normalization:
            lower_count = sum(
                entry.lower_threshold > 0
                for entry in self.display_normalization.values()
            )
            self.normalization_status_label.setText(
                f"Loaded {len(self.display_normalization):,} channel bounds from "
                f"{source}; {lower_count:,} use a non-zero lower threshold. Save "
                "the workspace to create an experiment-backed copy."
            )
        else:
            self.normalization_status_label.setText(
                "No channel-specific normalization is stored; images use the "
                "configured fallback quantile and display defaults."
            )
        self._refresh_feature_normalization_summary()

    def validate_normalization_editor(self) -> None:
        self.display_normalization = self._normalization_from_editor()
        self._clear_explore_layer_data_cache()
        lower_count = sum(
            entry.lower_threshold > 0 for entry in self.display_normalization.values()
        )
        self.normalization_status_label.setText(
            f"Valid normalization mapping: {len(self.display_normalization):,} "
            f"channel bounds; {lower_count:,} use a non-zero lower threshold. "
            "Save it into the experiment to persist edits."
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
        if not self.display_normalization:
            self.normalization_edit.clear()
            return None
        destination = root / "display" / "normalization.json"
        write_json(
            destination,
            {
                "normalization_dict": normalization_parameters_payload(
                    self.display_normalization
                )
            },
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
        lower_count = sum(
            entry.lower_threshold > 0 for entry in self.display_normalization.values()
        )
        self.normalization_status_label.setText(
            f"Saved {len(self.display_normalization):,} channel bounds "
            f"({lower_count:,} non-zero lower thresholds) and display defaults "
            f"inside {self.paths.root / 'display'}."
        )
        self._refresh_feature_normalization_summary()

    def _refresh_feature_normalization_summary(self) -> None:
        if not hasattr(self, "feature_normalization_summary"):
            return
        source = self.normalization_edit.text().strip() or "none"
        lower_count = sum(
            entry.lower_threshold > 0 for entry in self.display_normalization.values()
        )
        self.feature_normalization_summary.setText(
            f"Configured in Setup: {len(self.display_normalization):,} fixed "
            f"channel bounds ({lower_count:,} non-zero lower thresholds); "
            f"source/copy: {source}. Unmatched channels use quantile "
            f"{self.display_quantile_spin.value():.4f}."
        )
        self.refresh_feature_readiness()

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
        self._classification_enabled = bool(enabled)
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
            self.final_integration_enable_check,
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
        self._update_identity_integration_controls()

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

    def _create_variable_order_combo(self):
        """Create one view onto the session-wide variable-order registry."""

        combo = self.QComboBox()
        for label, mode in VARIABLE_ORDER_OPTIONS:
            combo.addItem(label, mode)
        current = combo.findData(self.variable_order_registry.mode)
        combo.setCurrentIndex(max(0, current))
        combo.setToolTip(
            "This is shared by every AnnData variable and image-channel list in "
            "NapariSBT. Similarity uses the same expression clustering as the "
            "matrix-plot ordering option and is cached for the live AnnData."
        )
        combo.currentIndexChanged.connect(
            self._guard(
                lambda source=combo: self._variable_order_changed(
                    str(source.currentData())
                )
            )
        )
        self._variable_order_combos.append(combo)
        return combo

    def _ordered_variable_values(self, values: Iterable[object]) -> list[str]:
        """Order marker/image display names against their canonical AnnData vars."""

        display_values = list(dict.fromkeys(str(value) for value in values))
        canonical_names: dict[str, str] = {}
        aliases = self._channel_aliases() if self.adata is not None else {}
        available = (
            set(self.adata.var_names.astype(str)) if self.adata is not None else set()
        )
        for value in display_values:
            base = value.split(" [", 1)[0]
            if base in available:
                canonical_names[value] = base
                continue
            key = self._normalise_marker_selection_name(base)
            canonical = aliases.get(key)
            if canonical is not None:
                canonical_names[value] = str(canonical)
        return self.variable_order_registry.ordered(
            display_values,
            canonical_names=canonical_names,
        )

    def _refresh_marker_overlay_list(self) -> None:
        if not hasattr(self, "marker_overlay_list"):
            return
        selected = {
            item.text() for item in self.marker_overlay_list.selectedItems()
        }
        self.marker_overlay_list.blockSignals(True)
        self.marker_overlay_list.clear()
        if self.adata is not None:
            self.marker_overlay_list.addItems(
                self._ordered_variable_values(self.adata.var_names.astype(str))
            )
        for index in range(self.marker_overlay_list.count()):
            self.marker_overlay_list.item(index).setSelected(
                self.marker_overlay_list.item(index).text() in selected
            )
        self.marker_overlay_list.blockSignals(False)

    def _refresh_variable_ordered_controls(self) -> None:
        """Reorder all variable selectors without reloading viewer layers or ROIs."""

        self._refresh_marker_overlay_list()
        if hasattr(self, "channel_list") and self.current_roi:
            self.refresh_channel_list()
        if hasattr(self, "feature_channel_list"):
            self.refresh_feature_channel_choices()
        if hasattr(self, "population_qc_marker_combos"):
            self.refresh_population_qc_marker_choices()
        if hasattr(self, "maintenance_var_rename_table"):
            self._refresh_maintenance_controls()
        if hasattr(self, "scanpy_plotting_panel"):
            self.scanpy_plotting_panel.refresh_variable_order()

    def _variable_order_changed(self, mode: str) -> None:
        if self._syncing_variable_order:
            return
        self.variable_order_registry.set_mode(mode)
        self._syncing_variable_order = True
        try:
            for combo in self._variable_order_combos:
                index = combo.findData(mode)
                if index >= 0 and combo.currentIndex() != index:
                    blocked = combo.blockSignals(True)
                    combo.setCurrentIndex(index)
                    combo.blockSignals(blocked)
        finally:
            self._syncing_variable_order = False

        working = mode == "similarity" and self.adata is not None
        if working:
            self._activity_begin(
                "Ordering variables",
                "Clustering adata.X variables by expression similarity once; the "
                "result will be cached and shared by every variable list…",
            )
            self.QApplication.processEvents()
        self._refresh_variable_ordered_controls()
        label = next(
            label for label, value in VARIABLE_ORDER_OPTIONS if value == mode
        )
        detail = f"Variable lists now use {label.lower()}."
        if self.variable_order_registry.last_warning:
            detail = self.variable_order_registry.last_warning
        if working:
            self._activity_finish(True, detail)
        self.set_status(detail)

    def _populate_anndata_selectors(self, *, source: str) -> None:
        if self.adata is None:
            raise RuntimeError("No AnnData object is available.")
        self.variable_order_registry.set_adata(self.adata)
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
            self.final_integration_source_combo,
        )
        roi_obs, object_obs = self._maintenance_identity_columns()
        population_columns = _population_observation_columns(
            columns,
            roi_obs=roi_obs,
            object_obs=object_obs,
        )
        choices_by_combo = {
            id(self.obs_combo): columns,
            id(self.overlay_obs_combo): columns,
            id(self.population_obs_combo): population_columns,
            id(self.population_qc_obs_combo): population_columns,
            id(self.curation_source_combo): population_columns,
            id(self.final_integration_source_combo): population_columns,
        }
        previous_values = {id(combo): combo.currentText() for combo in selector_combos}
        for combo in selector_combos:
            choices = choices_by_combo[id(combo)]
            previous = previous_values[id(combo)]
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(choices)
            if previous in choices:
                combo.setCurrentText(previous)
                continue
            preferred_value = _preferred_population_observation(
                self.adata.obs,
                population_columns,
                prefer_leiden=combo
                in {self.curation_source_combo, self.final_integration_source_combo},
            )
            if (
                combo is self.final_integration_source_combo
                and self.manifest is not None
                and self.manifest.cell_scope.mode == "obs_values"
                and self.manifest.cell_scope.obs_column in choices
            ):
                preferred_value = str(self.manifest.cell_scope.obs_column)
            if combo is self.overlay_obs_combo:
                preferred_value = columns[0] if columns else None
            if preferred_value is not None and combo.findText(preferred_value) >= 0:
                combo.setCurrentText(preferred_value)
        for combo in selector_combos:
            combo.blockSignals(False)
        self._refresh_marker_overlay_list()
        self.refresh_scope_values()
        self.refresh_population_values()
        self.refresh_population_qc_populations()
        self.refresh_feature_channel_choices()
        self._refresh_population_data_choices()
        self.refresh_population_workspace()
        self.mark_scanpy_plots_stale()
        self.refresh_scanpy_plotting_choices()
        self._refresh_maintenance_controls()
        self._refresh_cell_properties_available_observations()
        self._refresh_identity_integration_mapping()
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

    def _maintenance_identity_columns(self) -> tuple[str, str]:
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
        return roi_obs, object_obs

    def _maintenance_input_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve(strict=False)

    def _maintenance_image_roots(self) -> list[Path]:
        if self.manifest is not None:
            values = [
                *self.manifest.images_folders,
                *self.manifest.extra_images_folders,
            ]
        else:
            values = [
                *_split_paths(self.images_edit.toPlainText()),
                *_split_paths(self.extra_images_edit.toPlainText()),
            ]
        return [self._maintenance_input_path(value) for value in values]

    def _maintenance_mask_folder(self) -> Path | None:
        value = (
            self.manifest.masks_folder
            if self.manifest is not None
            else self.masks_edit.text().strip()
        )
        return self._maintenance_input_path(value) if value else None

    def _maintenance_audit_path(self) -> Path:
        root = (
            self.paths.root
            if self.paths is not None
            else self.project_root / "napari_sbt"
        )
        return root / "dataset_maintenance" / "audit.jsonl"

    def _maintenance_audit(self, action: str, details: dict[str, object]) -> None:
        append_maintenance_audit(
            self._maintenance_audit_path(), action=action, details=details
        )

    def _set_maintenance_checks(self, checks, *, summary: str | None = None) -> None:
        tree = self.maintenance_readiness_tree
        tree.clear()
        presentations = {
            "ready": ("● Ready", "#22c55e"),
            "warning": ("▲ Check", "#f59e0b"),
            "blocked": ("✕ Blocked", "#ef4444"),
            "optional": ("○ Optional", "#60a5fa"),
        }
        if summary:
            summary_item = self.QTreeWidgetItem(["", "Current preview", summary])
            summary_item.setForeground(2, self.QColor("#bfdbfe"))
            tree.addTopLevelItem(summary_item)
        for check in checks:
            text, colour = presentations.get(
                str(check.level), (str(check.level), "#d1d5db")
            )
            item = self.QTreeWidgetItem([text, check.label, check.detail])
            item.setForeground(0, self.QColor(colour))
            tree.addTopLevelItem(item)

    def refresh_maintenance_readiness(self) -> None:
        if self.adata is None:
            self.maintenance_readiness_tree.clear()
            self.maintenance_readiness_tree.addTopLevelItem(
                self.QTreeWidgetItem(
                    ["✕ Blocked", "Live AnnData", "Load AnnData in Setup first."]
                )
            )
            return
        roi_obs, object_obs = self._maintenance_identity_columns()
        checks = dataset_readiness(
            self.adata,
            roi_obs=roi_obs,
            object_obs=object_obs,
            mask_paths=self._mask_path_index,
            image_index=self._roi_image_path_index,
            expect_masks=self._maintenance_mask_folder() is not None,
            expect_images=bool(self._maintenance_image_roots()),
        )
        self._set_maintenance_checks(checks)
        self.maintenance_unsaved_label.setText(
            "● The live AnnData contains unsaved maintenance changes. Save it "
            "from Overview & Save when ready."
            if self.maintenance_dirty
            else "No unsaved Dataset Maintenance changes are recorded."
        )

    def rebuild_maintenance_asset_index(self) -> None:
        if self.adata is None:
            raise ValueError("Load AnnData before indexing dataset assets.")
        roi_obs, _object_obs = self._maintenance_identity_columns()
        if roi_obs not in self.adata.obs:
            raise ValueError(f"ROI observation is missing: {roi_obs!r}.")
        rois = self.adata.obs[roi_obs].astype("string").dropna().astype(str).unique()
        self._activity_begin(
            "Indexing maintenance assets",
            "Scanning configured mask and image folders once…",
        )
        try:
            mask_folder = self._maintenance_mask_folder()
            self._mask_path_index = (
                discover_mask_files(mask_folder)
                if mask_folder is not None and mask_folder.is_dir()
                else {}
            )
            image_roots = self._maintenance_image_roots()
            self._roi_image_path_index = discover_roi_image_index(
                image_roots,
                rois,
                channel_aliases=self._channel_aliases(),
            )
            self._asset_index_signature = self._current_asset_index_signature()
            self._refresh_maintenance_controls()
            detail = (
                f"Indexed {len(self._mask_path_index):,} masks and images for "
                f"{sum(bool(value) for value in self._roi_image_path_index.values()):,} "
                "ROIs."
            )
            self._activity_finish(True, detail)
            self.set_status(detail)
        except Exception:
            self._activity_finish(False, "Dataset asset indexing failed.")
            raise

    def _refresh_maintenance_controls(self) -> None:
        if not hasattr(self, "maintenance_var_rename_table"):
            return
        if self.adata is None:
            for widget in (
                self.maintenance_var_rename_table,
                self.maintenance_remove_vars_list,
                self.maintenance_filter_values_list,
                self.maintenance_obs_mapping_table,
            ):
                widget.clear()
            self.maintenance_filter_obs_combo.clear()
            self.maintenance_obs_combo.clear()
            self.refresh_maintenance_readiness()
            return
        prior_mapping = self._maintenance_var_mapping(allow_empty=True)
        image_counts: dict[str, int] = {}
        for channels in self._roi_image_path_index.values():
            for channel in channels:
                logical = str(channel).split(" [", 1)[0]
                image_counts[logical] = image_counts.get(logical, 0) + 1
        self.maintenance_var_rename_table.blockSignals(True)
        variables = self._ordered_variable_values(self.adata.var_names.astype(str))
        self.maintenance_var_rename_table.setRowCount(len(variables))
        for row, variable in enumerate(variables):
            current_item = self.QTableWidgetItem(variable)
            current_item.setFlags(current_item.flags() & ~self.Qt.ItemIsEditable)
            renamed_item = self.QTableWidgetItem(prior_mapping.get(variable, ""))
            count_item = self.QTableWidgetItem(f"{image_counts.get(variable, 0):,}")
            count_item.setFlags(count_item.flags() & ~self.Qt.ItemIsEditable)
            self.maintenance_var_rename_table.setItem(row, 0, current_item)
            self.maintenance_var_rename_table.setItem(row, 1, renamed_item)
            self.maintenance_var_rename_table.setItem(row, 2, count_item)
        self.maintenance_var_rename_table.blockSignals(False)

        selected_vars = {
            item.text() for item in self.maintenance_remove_vars_list.selectedItems()
        }
        self.maintenance_remove_vars_list.clear()
        self.maintenance_remove_vars_list.addItems(variables)
        for index in range(self.maintenance_remove_vars_list.count()):
            item = self.maintenance_remove_vars_list.item(index)
            item.setSelected(item.text() in selected_vars)

        columns = [str(column) for column in self.adata.obs.columns]
        for combo in (self.maintenance_filter_obs_combo, self.maintenance_obs_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(columns)
            if current in columns:
                combo.setCurrentText(current)
            combo.blockSignals(False)
        self.refresh_maintenance_filter_values()
        self.refresh_maintenance_observation_mapping()
        self.refresh_maintenance_readiness()

    def _maintenance_var_mapping(self, *, allow_empty: bool = False) -> dict[str, str]:
        mapping: dict[str, str] = {}
        if not hasattr(self, "maintenance_var_rename_table"):
            return mapping
        for row in range(self.maintenance_var_rename_table.rowCount()):
            source_item = self.maintenance_var_rename_table.item(row, 0)
            destination_item = self.maintenance_var_rename_table.item(row, 1)
            source = source_item.text().strip() if source_item else ""
            destination = destination_item.text().strip() if destination_item else ""
            if source and destination and source != destination:
                mapping[source] = destination
        if not mapping and not allow_empty:
            raise ValueError("Enter at least one new variable name in the table.")
        return mapping

    def _maintenance_var_mapping_changed(self, *_args) -> None:
        self.maintenance_image_rename_plan = None
        self.maintenance_var_rename_status.setText(
            "Variable mapping changed. Preview it before applying or copying images."
        )

    def choose_maintenance_anndata_destination(self) -> None:
        selected, _filter = self.QFileDialog.getSaveFileName(
            self.root,
            "Save maintained AnnData",
            self.maintenance_anndata_path_edit.text(),
            "AnnData files (*.h5ad)",
        )
        if selected:
            self.maintenance_anndata_path_edit.setText(selected)

    def _choose_new_maintenance_folder(self, editor, title: str) -> None:
        current = Path(editor.text()).expanduser()
        parent = current.parent if current.parent.exists() else self.project_root
        selected_parent = self.QFileDialog.getExistingDirectory(
            self.root, title, str(parent)
        )
        if not selected_parent:
            return
        name, accepted = self.QInputDialog.getText(
            self.root,
            title,
            "Name for the new output folder:",
            text=current.name or "derived_assets",
        )
        if accepted and str(name).strip():
            editor.setText(str(Path(selected_parent) / str(name).strip()))

    def choose_maintenance_image_output(self) -> None:
        self._choose_new_maintenance_folder(
            self.maintenance_image_output_edit,
            "Choose derived image output",
        )

    def choose_maintenance_mask_output(self) -> None:
        self._choose_new_maintenance_folder(
            self.maintenance_mask_output_edit,
            "Choose derived mask output",
        )

    def _update_maintenance_manifest_sources(self, **changes) -> None:
        if self.manifest is None or self.paths is None:
            return
        updated = self.manifest.model_copy(deep=True)
        for field, value in changes.items():
            setattr(updated, field, value)
        save_experiment(
            updated,
            self.paths.root,
            audit_action="dataset_maintenance_update_sources",
        )
        self.manifest = updated

    def save_current_maintenance_anndata(self) -> None:
        if self.adata is None:
            raise ValueError("Load AnnData before saving it.")
        destination = self._maintenance_input_path(
            self.maintenance_anndata_path_edit.text()
        )
        if destination.exists():
            if not self.maintenance_overwrite_anndata_check.isChecked():
                raise FileExistsError(
                    "The selected AnnData file already exists. Choose a new filename "
                    "or explicitly allow replacement of this exact file."
                )
            reply = self.QMessageBox.question(
                self.root,
                "Replace the selected AnnData file?",
                f"Replace this exact file with the current in-memory AnnData?\n\n"
                f"{destination}\n\nThis cannot be undone from NapariSBT.",
            )
            if reply != self.QMessageBox.Yes:
                return
        self._activity_begin(
            "Saving maintained AnnData",
            f"Writing an atomic copy to {destination}…",
        )
        try:
            saved = atomic_write_anndata(
                self.adata,
                destination,
                overwrite=self.maintenance_overwrite_anndata_check.isChecked(),
            )
            if self.use_saved_maintenance_anndata_check.isChecked():
                self.anndata_edit.setText(str(saved))
                self._in_memory_adata = None
                self._update_maintenance_manifest_sources(anndata_path=str(saved))
            self.maintenance_dirty = False
            self.maintenance_save_status_label.setText(
                f"Saved {self.adata.n_obs:,} cells and {self.adata.n_vars:,} "
                f"variables to {saved}."
            )
            self._maintenance_audit(
                "save_anndata",
                {
                    "destination": str(saved),
                    "cells": int(self.adata.n_obs),
                    "variables": int(self.adata.n_vars),
                    "replaced_existing": bool(
                        self.maintenance_overwrite_anndata_check.isChecked()
                    ),
                },
            )
            self.refresh_maintenance_readiness()
            self._activity_finish(True, f"Saved maintained AnnData to {saved}.")
            self.set_status(f"Saved maintained AnnData to {saved}.")
        except Exception:
            self._activity_finish(False, "Maintained AnnData save failed.")
            raise

    def preview_maintenance_var_rename(self) -> None:
        if self.adata is None:
            raise ValueError("Load AnnData before renaming variables.")
        mapping = self._maintenance_var_mapping()
        preview = preview_var_rename(
            self.adata,
            mapping,
            image_index=self._roi_image_path_index,
        )
        image_roots = self._maintenance_image_roots()
        self.maintenance_image_rename_plan = (
            plan_image_renames(
                self._roi_image_path_index,
                mapping,
                image_roots=image_roots,
                output_root=self._maintenance_input_path(
                    self.maintenance_image_output_edit.text()
                ),
            )
            if image_roots and self._roi_image_path_index
            else None
        )
        if self.maintenance_image_rename_plan is not None:
            plan = self.maintenance_image_rename_plan
            preview.checks.append(
                type(preview.checks[0])(
                    key="image_copy_plan",
                    label="Derived image collection",
                    level=("ready" if plan.ready else "blocked"),
                    detail=(
                        f"{len(plan.items):,} images will be copied into a complete "
                        "derived collection."
                        if plan.ready
                        else (
                            f"{len(plan.unresolved):,} unmatched filename(s) and "
                            f"{len(plan.collisions):,} collision(s) must be resolved."
                        )
                    ),
                )
            )
        self._set_maintenance_checks(preview.checks, summary=preview.summary)
        self.maintenance_var_rename_status.setText(
            preview.summary
            + (
                " The image copy plan is ready."
                if self.maintenance_image_rename_plan is not None
                and self.maintenance_image_rename_plan.ready
                else " Variable renaming is ready; image copying needs a complete index."
            )
        )

    @staticmethod
    def _rename_recipe_channels(recipe, mapping: dict[str, str]) -> None:
        recipe.image_channels = [mapping.get(value, value) for value in recipe.image_channels]
        recipe.marker_overlays = [mapping.get(value, value) for value in recipe.marker_overlays]
        for attribute in (
            "layer_colormaps",
            "layer_colormap_specs",
            "layer_visibility",
            "layer_opacities",
            "layer_contours",
            "layer_contrast_limits",
        ):
            values = getattr(recipe, attribute)
            setattr(
                recipe,
                attribute,
                {mapping.get(str(key), str(key)): value for key, value in values.items()},
            )

    def _rename_live_channel_references(self, mapping: dict[str, str]) -> None:
        renamed_normalization: dict[str, NimbusNormalizationParameters] = {}
        for key, value in self.display_normalization.items():
            renamed_normalization[mapping.get(str(key), str(key))] = value
        self.display_normalization = renamed_normalization
        self._set_normalization_table(self.display_normalization)
        self._rename_recipe_channels(self.explore_recipe, mapping)
        for recipe in self.explore_review_state.population_recipes.values():
            self._rename_recipe_channels(recipe, mapping)
        for preset in self.explore_review_state.recipe_presets.values():
            self._rename_recipe_channels(preset.recipe, mapping)
        self._save_explore_review_state()
        if self.manifest is not None and self.paths is not None:
            updated = self.manifest.model_copy(deep=True)
            updated.synthetic_features.channels = [
                mapping.get(value, value)
                for value in updated.synthetic_features.channels
            ]
            save_experiment(
                updated,
                self.paths.root,
                audit_action="dataset_maintenance_rename_channels",
            )
            self.manifest = updated

    def _after_maintenance_anndata_change(
        self,
        *,
        action: str,
        detail: str,
        identity_changed: bool = False,
        model_inputs_changed: bool = False,
        audit_details: dict[str, object] | None = None,
    ) -> None:
        self.maintenance_dirty = True
        if not self.anndata_edit.text().strip():
            self._in_memory_adata = self.adata
        if identity_changed:
            if (
                self.manifest is not None
                and self.paths is not None
                and self.manifest.workflow_mode == "dataset_maintenance"
            ):
                self._revise_maintenance_cohort()
            else:
                self.preview = None
                self.cohort = pd.DataFrame()
            self.labels = empty_labels()
            self.scores = pd.DataFrame()
            self.final_assignments = pd.DataFrame()
            self._set_classification_enabled(False)
        elif model_inputs_changed:
            self.model_bundle = None
            self.scores = pd.DataFrame()
            self.final_assignments = pd.DataFrame()
        self._populate_anndata_selectors(source="Dataset Maintenance in-memory result")
        self._maintenance_audit(action, audit_details or {"detail": detail})
        self.maintenance_unsaved_label.setText(
            "● The live AnnData contains unsaved maintenance changes."
        )
        self.set_status(detail)

    def _revise_maintenance_cohort(self) -> None:
        """Keep an AnnData-maintenance workspace aligned after identity changes."""

        if self.adata is None or self.manifest is None or self.paths is None:
            return
        revised_preview = resolve_cohort(
            self.adata,
            roi_obs=self.manifest.roi_obs,
            object_id_obs=self.manifest.object_id_obs,
            mode="all_cells",
        )
        revision = int(self.manifest.revision) + 1
        relative_snapshot = Path("cohort") / f"eligible_cells_r{revision}.parquet"
        save_cohort_snapshot(revised_preview, self.paths.root / relative_snapshot)
        updated = self.manifest.model_copy(deep=True)
        updated.revision = revision
        updated.cell_scope = revised_preview.scope(
            mode="all_cells",
            obs_column=None,
            obs_values=(),
            snapshot_path=relative_snapshot.as_posix(),
        )
        updated.active_feature_set_id = None
        updated.active_model_features = []
        self.paths = save_experiment(
            updated,
            self.paths.root,
            audit_action="dataset_maintenance_revise_identity",
        )
        self.manifest = updated
        self.preview = revised_preview
        self.cohort = revised_preview.eligible_cells.copy()
        self._invalidate_population_qc_caches()
        self._update_scope_text()

    def apply_maintenance_var_rename(self) -> None:
        if self.adata is None:
            raise ValueError("Load AnnData before renaming variables.")
        mapping = self._maintenance_var_mapping()
        self.preview_maintenance_var_rename()
        self.adata = apply_var_rename(
            self.adata,
            mapping,
            update_raw=self.maintenance_update_raw_names_check.isChecked(),
        )
        self._rename_live_channel_references(mapping)
        self._after_maintenance_anndata_change(
            action="rename_variables",
            detail=f"Renamed {len(mapping):,} variables in the live AnnData.",
            model_inputs_changed=True,
            audit_details={"mapping": mapping},
        )

    def copy_maintenance_renamed_images(self) -> None:
        plan = self.maintenance_image_rename_plan
        if plan is None:
            self.preview_maintenance_var_rename()
            plan = self.maintenance_image_rename_plan
        if plan is None or not plan.ready:
            raise ValueError(
                "The derived image plan is not ready. Build the image index and "
                "resolve the filename warnings shown by Preview rename."
            )
        self._activity_begin(
            "Copying synchronized images",
            f"Copying {len(plan.items):,} indexed images into {plan.output_root}…",
        )
        try:
            output = copy_renamed_images(plan)
            derived_roots = [path for path in sorted(output.iterdir()) if path.is_dir()]
            self.images_edit.setPlainText("\n".join(map(str, derived_roots)))
            self.extra_images_edit.clear()
            self._roi_image_path_index.clear()
            self._update_maintenance_manifest_sources(
                images_folders=list(map(str, derived_roots)),
                extra_images_folders=[],
            )
            self._maintenance_audit(
                "copy_renamed_images",
                {
                    "output": str(output),
                    "image_count": len(plan.items),
                    "mapping": {
                        item.channel_before: item.channel_after for item in plan.items
                    },
                },
            )
            detail = (
                f"Copied {len(plan.items):,} synchronized images to {output}. "
                "The Setup image folders now point to the derived collection."
            )
            self.maintenance_var_rename_status.setText(detail)
            self._activity_finish(True, detail)
            self.set_status(detail)
            self.refresh_maintenance_readiness()
        except Exception:
            self._activity_finish(False, "Synchronized image copying failed.")
            raise

    def remove_selected_maintenance_vars(self) -> None:
        if self.adata is None:
            raise ValueError("Load AnnData before removing variables.")
        selected = [
            item.text() for item in self.maintenance_remove_vars_list.selectedItems()
        ]
        if not selected:
            raise ValueError("Select at least one AnnData variable to remove.")
        reply = self.QMessageBox.question(
            self.root,
            "Remove variables from the live AnnData?",
            f"Remove {len(selected):,} selected variable(s) in memory? Image files "
            "will remain untouched. Reload the source AnnData before saving if "
            "you need to discard this change.",
        )
        if reply != self.QMessageBox.Yes:
            return
        self.adata = remove_anndata_vars(
            self.adata,
            selected,
            subset_raw=self.maintenance_subset_raw_check.isChecked(),
        )
        self._after_maintenance_anndata_change(
            action="remove_variables",
            detail=(
                f"Removed {len(selected):,} variables from the live AnnData; "
                "image files were left intact."
            ),
            model_inputs_changed=True,
            audit_details={
                "removed_variables": selected,
                "images_modified": False,
                "subset_raw": self.maintenance_subset_raw_check.isChecked(),
            },
        )

    def refresh_maintenance_filter_values(self) -> None:
        if self.adata is None:
            self.maintenance_filter_values_list.clear()
            return
        observation = self.maintenance_filter_obs_combo.currentText()
        if observation not in self.adata.obs:
            return
        selected = {
            item.text()
            for item in self.maintenance_filter_values_list.selectedItems()
        }
        series = self.adata.obs[observation]
        if isinstance(series.dtype, pd.CategoricalDtype):
            values = series.cat.categories.astype(str).tolist()
        else:
            values = (
                series.astype("string")
                .dropna()
                .drop_duplicates()
                .astype(str)
                .tolist()
            )
            values.sort(key=str.casefold)
        self.maintenance_filter_values_list.clear()
        self.maintenance_filter_values_list.addItems(values[:5000])
        for index in range(self.maintenance_filter_values_list.count()):
            item = self.maintenance_filter_values_list.item(index)
            item.setSelected(item.text() in selected)
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if not numeric.empty:
            self.maintenance_filter_lower_spin.setValue(float(numeric.min()))
            self.maintenance_filter_upper_spin.setValue(float(numeric.max()))
        self.update_maintenance_filter_controls()

    def update_maintenance_filter_controls(self) -> None:
        mode = str(self.maintenance_filter_mode_combo.currentData())
        values_mode = mode in {"keep_values", "remove_values"}
        range_mode = mode in {"keep_range", "remove_range"}
        self.maintenance_filter_values_list.setEnabled(values_mode)
        self.maintenance_filter_lower_spin.setEnabled(range_mode)
        self.maintenance_filter_upper_spin.setEnabled(range_mode)
        self.maintenance_last_filter_request = None
        self.maintenance_filter_status.setText(
            "Filter settings changed. Preview the retained cell and ROI counts."
        )

    def _current_maintenance_filter_request(self) -> CellFilterRequest:
        return CellFilterRequest(
            observation=self.maintenance_filter_obs_combo.currentText(),
            mode=str(self.maintenance_filter_mode_combo.currentData()),
            values=[
                item.text()
                for item in self.maintenance_filter_values_list.selectedItems()
            ],
            lower=float(self.maintenance_filter_lower_spin.value()),
            upper=float(self.maintenance_filter_upper_spin.value()),
        )

    def preview_maintenance_cell_filter(self) -> None:
        if self.adata is None:
            raise ValueError("Load AnnData before filtering cells.")
        request = self._current_maintenance_filter_request()
        roi_obs, _object_obs = self._maintenance_identity_columns()
        preview = preview_cell_filter(self.adata, request, roi_obs=roi_obs)
        self.maintenance_last_filter_request = request
        self._set_maintenance_checks(preview.checks, summary=preview.summary)
        self.maintenance_filter_status.setText(preview.summary)

    def apply_maintenance_cell_filter(self) -> None:
        if self.adata is None:
            raise ValueError("Load AnnData before filtering cells.")
        request = self._current_maintenance_filter_request()
        if (
            self.maintenance_last_filter_request is None
            or self.maintenance_last_filter_request.model_dump()
            != request.model_dump()
        ):
            raise ValueError(
                "Preview these exact filter settings before applying them."
            )
        roi_obs, _object_obs = self._maintenance_identity_columns()
        preview = preview_cell_filter(self.adata, request, roi_obs=roi_obs)
        if not preview.ready:
            raise ValueError(preview.summary)
        reply = self.QMessageBox.question(
            self.root,
            "Filter cells in the live AnnData?",
            f"{preview.summary}\n\nThe source file and masks are unchanged until "
            "you explicitly save AnnData or write derived masks.",
        )
        if reply != self.QMessageBox.Yes:
            return
        before = int(self.adata.n_obs)
        self.adata = apply_cell_filter(self.adata, request)
        removed = before - int(self.adata.n_obs)
        self._after_maintenance_anndata_change(
            action="filter_cells",
            detail=(
                f"Filtered the live AnnData to {self.adata.n_obs:,} cells; "
                f"{removed:,} cells were removed. Rebuild masks before saving a "
                "fully synchronized dataset."
            ),
            identity_changed=True,
            model_inputs_changed=True,
            audit_details={
                "filter": request.model_dump(mode="json"),
                "cells_before": before,
                "cells_after": int(self.adata.n_obs),
            },
        )

    def preview_maintenance_mask_rebuild(self) -> None:
        if self.adata is None:
            raise ValueError("Load AnnData before rebuilding masks.")
        if not self._mask_path_index:
            raise ValueError(
                "No mask index is available. Press Rebuild mask/image index now first."
            )
        roi_obs, object_obs = self._maintenance_identity_columns()
        mode = str(self.maintenance_mask_mode_combo.currentData())
        self._activity_begin(
            "Validating mask alignment",
            "Reading represented masks and comparing ObjectNumbers…",
        )
        try:
            preview = preview_mask_rebuild(
                self.adata,
                self._mask_path_index,
                roi_obs=roi_obs,
                object_obs=object_obs,
                mode=mode,
            )
            self._set_maintenance_checks(preview.checks, summary=preview.summary)
            self.maintenance_mask_status.setText(preview.summary)
            self._activity_finish(preview.ready, preview.summary)
        except Exception:
            self._activity_finish(False, "Mask-alignment validation failed.")
            raise

    def apply_maintenance_mask_rebuild(self) -> None:
        if self.adata is None:
            raise ValueError("Load AnnData before rebuilding masks.")
        if not self._mask_path_index:
            raise ValueError("Build the mask index before rebuilding masks.")
        roi_obs, object_obs = self._maintenance_identity_columns()
        mode = str(self.maintenance_mask_mode_combo.currentData())
        output = self._maintenance_input_path(self.maintenance_mask_output_edit.text())
        self._activity_begin(
            "Rebuilding synchronized masks",
            f"Writing derived masks to {output}…",
        )
        try:
            updated, crosswalk, written = rebuild_masks_and_object_numbers(
                self.adata,
                self._mask_path_index,
                output,
                roi_obs=roi_obs,
                object_obs=object_obs,
                mode=mode,
            )
            changed_ids = int(
                crosswalk["ObjectNumber_before"]
                .ne(crosswalk["ObjectNumber_after"])
                .sum()
            )
            self.adata = updated
            self.masks_edit.setText(str(written))
            self._mask_path_index = discover_mask_files(written)
            self._update_maintenance_manifest_sources(masks_folder=str(written))
            self._after_maintenance_anndata_change(
                action="rebuild_masks",
                detail=(
                    f"Wrote {len(self._mask_path_index):,} derived masks to {written}; "
                    f"{changed_ids:,} ObjectNumbers changed. Setup now uses the "
                    "derived mask folder."
                ),
                identity_changed=changed_ids > 0,
                model_inputs_changed=changed_ids > 0,
                audit_details={
                    "output": str(written),
                    "mode": mode,
                    "cells": len(crosswalk),
                    "changed_object_numbers": changed_ids,
                },
            )
            self.maintenance_mask_status.setText(
                f"Derived masks are active from {written}."
            )
            self._activity_finish(True, f"Derived masks written to {written}.")
        except Exception:
            self._activity_finish(False, "Mask rebuilding failed.")
            raise

    def refresh_maintenance_observation_mapping(self) -> None:
        """Load the selected observation into the editable remapping table."""

        table = self.maintenance_obs_mapping_table
        table.blockSignals(True)
        table.setRowCount(0)
        if self.adata is None:
            table.blockSignals(False)
            self.maintenance_obs_utility_source_label.setText(
                "No observation selected"
            )
            return
        observation = self.maintenance_obs_combo.currentText().strip()
        if not observation or observation not in self.adata.obs:
            table.blockSignals(False)
            return

        series = self.adata.obs[observation]
        displayed = series.astype("string")
        if isinstance(series.dtype, pd.CategoricalDtype):
            values = [
                str(value)
                for value in series.cat.categories
                if displayed.eq(str(value)).any()
            ]
        else:
            values = sorted(displayed.dropna().astype(str).unique().tolist())
        counts = displayed.value_counts(dropna=True).to_dict()
        colours = categorical_colour_map(self.adata, observation)
        table.setRowCount(len(values))
        for row, value in enumerate(values):
            table.setItem(row, 0, self._readonly_table_item(value))
            table.setItem(
                row,
                1,
                self._readonly_table_item(f"{int(counts.get(value, 0)):,}"),
            )
            table.setItem(row, 2, self.QTableWidgetItem(value))
            table.setItem(
                row,
                3,
                self.QTableWidgetItem(str(colours.get(value, "#808080"))),
            )
        table.blockSignals(False)
        self._style_colour_mapping_table(table, name_column=2, colour_column=3)

        default_output = f"{observation}_remapped"
        current_output = self.maintenance_obs_output_edit.text().strip()
        if not current_output or current_output == self._maintenance_obs_default_output:
            blocked = self.maintenance_obs_output_edit.blockSignals(True)
            self.maintenance_obs_output_edit.setText(default_output)
            self.maintenance_obs_output_edit.blockSignals(blocked)
        self._maintenance_obs_default_output = default_output
        self.maintenance_obs_utility_source_label.setText(observation)
        self._maintenance_observation_mapping_changed()

    def _maintenance_observation_mapping_frames(
        self,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
        mapping: dict[str, str] = {}
        colours: dict[str, str] = {}
        counts: dict[str, int] = {}
        table = self.maintenance_obs_mapping_table
        for row in range(table.rowCount()):
            source = table.item(row, 0).text().strip()
            proposed = table.item(row, 2).text().strip()
            colour = table.item(row, 3).text().strip()
            count = int(table.item(row, 1).text().replace(",", ""))
            mapping[source] = proposed
            counts[proposed] = counts.get(proposed, 0) + count
            colours.setdefault(proposed, colour)
        return mapping, colours, counts

    def _maintenance_observation_mapping_changed(self, changed_item=None) -> None:
        """Keep explicit merges synchronized and report colour/readiness issues."""

        table = self.maintenance_obs_mapping_table
        if table.rowCount() == 0:
            return
        if hasattr(changed_item, "column") and changed_item.column() in {2, 3}:
            changed_row = changed_item.row()
            label = table.item(changed_row, 2).text().strip()
            if changed_item.column() == 2:
                shared_colour = next(
                    (
                        table.item(row, 3).text().strip()
                        for row in range(table.rowCount())
                        if row != changed_row
                        and table.item(row, 2).text().strip() == label
                    ),
                    table.item(changed_row, 3).text().strip(),
                )
            else:
                shared_colour = changed_item.text().strip()
            blocked = table.blockSignals(True)
            try:
                for row in range(table.rowCount()):
                    if table.item(row, 2).text().strip() == label:
                        table.item(row, 3).setText(shared_colour)
            finally:
                table.blockSignals(blocked)

        collisions = self._style_colour_mapping_table(
            table,
            name_column=2,
            colour_column=3,
        )
        mapping, _colours, counts = self._maintenance_observation_mapping_frames()
        blank = [source for source, proposed in mapping.items() if not proposed]
        merge_count = sum(
            1
            for label in set(mapping.values())
            if label and list(mapping.values()).count(label) > 1
        )
        destination = self.maintenance_obs_output_edit.text().strip()
        warnings = []
        if blank:
            warnings.append(f"{len(blank)} source value(s) have no proposed name")
        if collisions:
            warnings.append(f"{len(collisions)} colour collision(s)")
        if self.adata is not None and destination in self.adata.obs:
            if not self.maintenance_obs_overwrite_checkbox.isChecked():
                warnings.append("the output obs already exists and overwrite is off")
        if warnings:
            self.maintenance_obs_status.setText(
                "▲ Check before applying: " + "; ".join(warnings) + "."
            )
            return
        self.maintenance_obs_status.setText(
            f"● Ready: {len(counts):,} final population(s), {merge_count:,} explicit "
            f"merge group(s), output adata.obs[{destination!r}]."
        )

    def name_selected_maintenance_observation_rows(self) -> None:
        rows = self._selected_population_table_rows(
            self.maintenance_obs_mapping_table
        )
        if not rows:
            raise ValueError("Select one or more complete table rows first.")
        initial = self.maintenance_obs_mapping_table.item(rows[0], 2).text()
        value, accepted = self.QInputDialog.getText(
            self.root,
            "Name or merge observation values",
            "Final population name:",
            text=initial,
        )
        if not accepted:
            return
        value = value.strip()
        if not value:
            raise ValueError("Population names must not be blank.")
        for row in rows:
            self.maintenance_obs_mapping_table.item(row, 2).setText(value)
        self._maintenance_observation_mapping_changed()

    def colour_selected_maintenance_observation_rows(self) -> None:
        rows = self._selected_population_table_rows(
            self.maintenance_obs_mapping_table
        )
        if not rows:
            raise ValueError("Select one or more complete table rows first.")
        initial = self.QColor(
            self.maintenance_obs_mapping_table.item(rows[0], 3).text()
        )
        colour = self.QColorDialog.getColor(initial, self.root)
        if not colour.isValid():
            return
        for row in rows:
            self.maintenance_obs_mapping_table.item(row, 3).setText(colour.name())
        self._maintenance_observation_mapping_changed()

    def auto_colour_maintenance_observation(self) -> None:
        mapping, _colours, counts = self._maintenance_observation_mapping_frames()
        assignment = self._choose_automatic_colours(
            list(mapping.values()),
            counts,
            context="final observation categories",
        )
        if assignment is None:
            return
        table = self.maintenance_obs_mapping_table
        blocked = table.blockSignals(True)
        try:
            for row in range(table.rowCount()):
                proposed = table.item(row, 2).text().strip()
                table.item(row, 3).setText(assignment[proposed])
        finally:
            table.blockSignals(blocked)
        self._maintenance_observation_mapping_changed()

    def apply_maintenance_observation_mapping(self) -> None:
        if self.adata is None:
            raise ValueError("Load AnnData before creating an observation.")
        source = self.maintenance_obs_combo.currentText().strip()
        destination = self.maintenance_obs_output_edit.text().strip()
        overwrite = self.maintenance_obs_overwrite_checkbox.isChecked()
        if (
            source == destination
            and source in self._protected_maintenance_observations()
        ):
            raise ValueError(
                f"Observation {source!r} defines dataset identity or a frozen cohort "
                "and cannot be overwritten here. Create a new output obs instead."
            )
        mapping, colours, _counts = self._maintenance_observation_mapping_frames()
        result = remap_categorical_observation(
            self.adata,
            source,
            destination,
            mapping,
            colours,
            overwrite=overwrite,
        )
        category_count = len(result.obs[destination].cat.categories)
        self.adata = result
        self._after_maintenance_anndata_change(
            action="remap_observation",
            detail=(
                f"Created adata.obs[{destination!r}] from {source!r} with "
                f"{category_count:,} named categories and synchronized colours."
            ),
            audit_details={
                "source": source,
                "destination": destination,
                "overwrite": overwrite,
                "mapping": mapping,
                "colours": colours,
            },
        )

    def _protected_maintenance_observations(self) -> set[str]:
        roi_obs, object_obs = self._maintenance_identity_columns()
        protected = {roi_obs, object_obs}
        if self.manifest is not None and self.manifest.cell_scope.obs_column:
            protected.add(self.manifest.cell_scope.obs_column)
        if self.population_workspace is not None:
            protected.add(self.population_workspace.source_obs)
        if self.population_draft is not None:
            protected.add(self.population_draft.derived_obs)
        return {value for value in protected if value}

    @staticmethod
    def _rename_recipe_observation(recipe, source: str, destination: str) -> None:
        if recipe.observation_overlay == source:
            recipe.observation_overlay = destination
        if recipe.population_observation == source:
            recipe.population_observation = destination
        old_obs_layer = f"obs::{source}"
        new_obs_layer = f"obs::{destination}"
        old_population_prefix = f"population::{source}::"
        new_population_prefix = f"population::{destination}::"
        for attribute in (
            "layer_colormaps",
            "layer_colormap_specs",
            "layer_visibility",
            "layer_opacities",
            "layer_contours",
            "layer_contrast_limits",
        ):
            renamed = {}
            for key, value in getattr(recipe, attribute).items():
                key = str(key)
                if key == old_obs_layer:
                    key = new_obs_layer
                elif key.startswith(old_population_prefix):
                    key = new_population_prefix + key[len(old_population_prefix) :]
                renamed[key] = value
            setattr(recipe, attribute, renamed)

    def _rename_live_observation_references(
        self, source: str, destination: str
    ) -> None:
        self._rename_recipe_observation(self.explore_recipe, source, destination)
        for recipe in self.explore_review_state.population_recipes.values():
            self._rename_recipe_observation(recipe, source, destination)
        for preset in self.explore_review_state.recipe_presets.values():
            self._rename_recipe_observation(preset.recipe, source, destination)
        self._save_explore_review_state()

    def rename_maintenance_observation(self) -> None:
        if self.adata is None:
            raise ValueError("Load AnnData before renaming an observation.")
        source = self.maintenance_obs_combo.currentText().strip()
        destination = self.maintenance_obs_new_name_edit.text().strip()
        if source in self._protected_maintenance_observations():
            raise ValueError(
                f"Observation {source!r} defines dataset identity or a frozen cohort "
                "and cannot be renamed here."
            )
        if not destination:
            raise ValueError("Enter a new observation name.")
        if destination in self.adata.obs:
            raise ValueError(f"Observation already exists: {destination!r}.")
        self.adata.obs.rename(columns={source: destination}, inplace=True)
        colour_key = f"{source}_colors"
        if colour_key in self.adata.uns:
            self.adata.uns[f"{destination}_colors"] = self.adata.uns.pop(colour_key)
        self._rename_live_observation_references(source, destination)
        self._after_maintenance_anndata_change(
            action="rename_observation",
            detail=f"Renamed adata.obs[{source!r}] to {destination!r} in memory.",
            audit_details={"source": source, "destination": destination},
        )

    def remove_maintenance_observation(self) -> None:
        if self.adata is None:
            raise ValueError("Load AnnData before removing an observation.")
        observation = self.maintenance_obs_combo.currentText().strip()
        if observation in self._protected_maintenance_observations():
            raise ValueError(
                f"Observation {observation!r} defines dataset identity or a frozen "
                "cohort and cannot be removed."
            )
        reply = self.QMessageBox.question(
            self.root,
            "Remove observation from the live AnnData?",
            f"Remove adata.obs[{observation!r}] in memory? Reload the source "
            "AnnData before saving if you need to discard this change.",
        )
        if reply != self.QMessageBox.Yes:
            return
        self.adata.obs.drop(columns=[observation], inplace=True)
        self.adata.uns.pop(f"{observation}_colors", None)
        self._after_maintenance_anndata_change(
            action="remove_observation",
            detail=f"Removed adata.obs[{observation!r}] from the live AnnData.",
            audit_details={"observation": observation},
        )

    def repair_maintenance_observation_palette(self) -> None:
        if self.adata is None:
            raise ValueError("Load AnnData before repairing a colour palette.")
        observation = self.maintenance_obs_combo.currentText().strip()
        series = self.adata.obs[observation]
        if not isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype("category")
            self.adata.obs[observation] = series
        colours = categorical_colour_map(self.adata, observation)
        categories = self.adata.obs[observation].cat.categories.astype(str).tolist()
        self.adata.uns[f"{observation}_colors"] = [
            colours[category] for category in categories
        ]
        self._after_maintenance_anndata_change(
            action="repair_observation_palette",
            detail=(
                f"Stored {len(categories):,} category colours in "
                f"adata.uns[{observation + '_colors'!r}]."
            ),
            audit_details={"observation": observation, "categories": categories},
        )

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
        source_changed = self._curation_auto_obs_source != source_obs
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
            current_obs = self.curation_derived_obs_edit.text().strip()
            if (
                source_changed
                or not current_obs
                or current_obs == self._curation_auto_obs_value
            ):
                self.curation_derived_obs_edit.setText(default_obs)
            self._curation_auto_obs_source = source_obs
            self._curation_auto_obs_value = default_obs
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
        self._curation_auto_obs_source = source_obs
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

    def _style_colour_mapping_table(
        self,
        table,
        *,
        name_column: int,
        colour_column: int,
    ) -> dict[str, list[str]]:
        """Render name and colour cells as swatches and return label collisions."""

        labels = []
        colours = []
        for row in range(table.rowCount()):
            name_item = table.item(row, name_column)
            colour_item = table.item(row, colour_column)
            labels.append(name_item.text().strip() if name_item else "")
            colours.append(colour_item.text().strip() if colour_item else "")
        collisions = categorical_colour_collisions(labels, colours)

        blocked = table.blockSignals(True)
        try:
            for row, (label, colour_text) in enumerate(
                zip(labels, colours, strict=True)
            ):
                name_item = table.item(row, name_column)
                colour_item = table.item(row, colour_column)
                if name_item is None or colour_item is None:
                    continue
                colour = self.QColor(colour_text)
                if colour.isValid():
                    foreground = self.QColor(contrasting_text_colour(colour.name()))
                    for item in (name_item, colour_item):
                        item.setBackground(colour)
                        item.setForeground(foreground)
                else:
                    for item in (name_item, colour_item):
                        item.setBackground(self.QColor("#fecaca"))
                        item.setForeground(self.QColor("#7f1d1d"))
                canonical = colour.name().lower() if colour.isValid() else ""
                if canonical in collisions:
                    detail = (
                        f"Colour collision: {canonical} is assigned to different "
                        f"final populations: {', '.join(collisions[canonical])}."
                    )
                    name_item.setToolTip(detail)
                    colour_item.setToolTip(detail)
                else:
                    name_item.setToolTip(
                        "Rows with the same proposed name are an explicit merge and "
                        "intentionally share this colour."
                        if labels.count(label) > 1
                        else ""
                    )
                    colour_item.setToolTip(name_item.toolTip())
        finally:
            table.blockSignals(blocked)
        return collisions

    def _choose_automatic_colours(
        self,
        labels: list[str],
        counts: dict[str, int],
        *,
        context: str,
    ) -> dict[str, str] | None:
        """Show the reusable Colour Helper and return its chosen assignment."""

        from qtpy.QtWidgets import (
            QAbstractItemView,
            QComboBox,
            QDialogButtonBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QListWidget,
            QListWidgetItem,
            QPushButton,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
        )

        labels = list(dict.fromkeys(label.strip() for label in labels if label.strip()))
        if not labels:
            raise ValueError("There are no proposed population names to colour.")
        palettes = categorical_palette_catalog()
        dialog = self.QDialog(self.root)
        dialog.setWindowTitle("Colour Helper")
        dialog.resize(820, 680)
        layout = QVBoxLayout(dialog)
        introduction = QLabel(
            f"Assign distinct categorical colours to {len(labels):,} {context}. "
            "Choose a palette, untick colours you want to avoid, then choose how "
            "population names should be matched to the palette order."
        )
        introduction.setWordWrap(True)
        layout.addWidget(introduction)

        controls = QFormLayout()
        palette_combo = QComboBox()
        palette_combo.addItems(list(palettes))
        order_combo = QComboBox()
        for text, value in (
            ("Abundance — largest first", "abundance_desc"),
            ("Abundance — smallest first", "abundance_asc"),
            ("Alphabetical — A to Z", "alphabetical_asc"),
            ("Alphabetical — Z to A", "alphabetical_desc"),
        ):
            order_combo.addItem(text, value)
        controls.addRow("Categorical palette", palette_combo)
        controls.addRow("Assign colours by", order_combo)
        layout.addLayout(controls)

        colour_list = QListWidget()
        colour_list.setSelectionMode(QAbstractItemView.NoSelection)
        colour_list.setMinimumHeight(190)
        layout.addWidget(colour_list)
        colour_actions = QHBoxLayout()
        select_all_button = QPushButton("Use all colours")
        select_none_button = QPushButton("Use none")
        colour_actions.addWidget(select_all_button)
        colour_actions.addWidget(select_none_button)
        colour_actions.addStretch(1)
        layout.addLayout(colour_actions)

        preview = QTableWidget(0, 3)
        preview.setHorizontalHeaderLabels(["Assignment order", "Cells", "Colour"])
        preview.horizontalHeader().setStretchLastSection(True)
        preview.setAlternatingRowColors(False)
        layout.addWidget(preview)
        status = QLabel()
        status.setWordWrap(True)
        layout.addWidget(status)
        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        apply_button = buttons.button(QDialogButtonBox.Apply)
        layout.addWidget(buttons)
        result: dict[str, str] = {}

        def selected_colours() -> list[str]:
            return [
                colour_list.item(index).data(self.Qt.UserRole)
                for index in range(colour_list.count())
                if colour_list.item(index).checkState() == self.Qt.Checked
            ]

        def update_preview(*_args) -> None:
            nonlocal result
            try:
                result = assign_categorical_colours(
                    labels,
                    counts,
                    selected_colours(),
                    order=str(order_combo.currentData()),
                )
            except ValueError as exc:
                result = {}
                preview.setRowCount(0)
                status.setText(f"⚠ {exc}")
                apply_button.setEnabled(False)
                return
            preview.setRowCount(len(result))
            for row, (label, colour_text) in enumerate(result.items()):
                name_item = QTableWidgetItem(label)
                count_item = QTableWidgetItem(f"{counts.get(label, 0):,}")
                colour_item = QTableWidgetItem(colour_text)
                for item in (name_item, count_item, colour_item):
                    item.setFlags(item.flags() & ~self.Qt.ItemIsEditable)
                colour = self.QColor(colour_text)
                colour_item.setBackground(colour)
                colour_item.setForeground(
                    self.QColor(contrasting_text_colour(colour_text))
                )
                preview.setItem(row, 0, name_item)
                preview.setItem(row, 1, count_item)
                preview.setItem(row, 2, colour_item)
            status.setText(
                f"● Ready: {len(result):,} populations will receive distinct colours; "
                f"{len(selected_colours()):,} palette colours are enabled."
            )
            apply_button.setEnabled(True)

        def load_palette(*_args) -> None:
            colour_list.blockSignals(True)
            colour_list.clear()
            for colour_text in palettes[palette_combo.currentText()]:
                item = QListWidgetItem(colour_text)
                item.setFlags(item.flags() | self.Qt.ItemIsUserCheckable)
                item.setCheckState(self.Qt.Checked)
                item.setData(self.Qt.UserRole, colour_text)
                item.setBackground(self.QColor(colour_text))
                item.setForeground(
                    self.QColor(contrasting_text_colour(colour_text))
                )
                colour_list.addItem(item)
            colour_list.blockSignals(False)
            update_preview()

        def set_all(check_state) -> None:
            colour_list.blockSignals(True)
            for index in range(colour_list.count()):
                colour_list.item(index).setCheckState(check_state)
            colour_list.blockSignals(False)
            update_preview()

        palette_combo.currentTextChanged.connect(load_palette)
        order_combo.currentIndexChanged.connect(update_preview)
        colour_list.itemChanged.connect(update_preview)
        select_all_button.clicked.connect(lambda: set_all(self.Qt.Checked))
        select_none_button.clicked.connect(lambda: set_all(self.Qt.Unchecked))
        buttons.rejected.connect(dialog.reject)
        apply_button.clicked.connect(dialog.accept)
        load_palette()
        if dialog.exec() != self.QDialog.Accepted:
            return None
        return result

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
        self._style_colour_mapping_table(
            self.population_base_table,
            name_column=2,
            colour_column=3,
        )
        self._style_colour_mapping_table(
            self.population_components_table,
            name_column=4,
            colour_column=5,
        )

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

    def _population_tables_changed(self, changed_item=None) -> None:
        if self.population_draft is None:
            return
        self._mark_population_draft_dirty()
        try:
            if changed_item is not None:
                table = changed_item.tableWidget()
                colour_column = (
                    3 if table is self.population_base_table else 5
                )
                name_column = 2 if table is self.population_base_table else 4
                if changed_item.column() == colour_column:
                    colour = self.QColor(changed_item.text().strip())
                    if colour.isValid():
                        label = table.item(changed_item.row(), name_column).text()
                        self._propagate_population_colour(label, colour.name())
            self._refresh_population_merge_preview()
        except Exception as exc:  # allow temporarily incomplete table edits
            self.population_merge_preview.setPlainText(
                f"Finish the current edit to refresh the preview: {exc}"
            )

    def _propagate_population_colour(self, proposed_label: str, colour: str) -> None:
        """Apply an edited final-label colour across base and component rows."""

        label = str(proposed_label).strip()
        for table, name_column, colour_column in (
            (self.population_base_table, 2, 3),
            (self.population_components_table, 4, 5),
        ):
            blocked = table.blockSignals(True)
            try:
                for row in range(table.rowCount()):
                    if table.item(row, name_column).text().strip() != label:
                        continue
                    item = table.item(row, colour_column)
                    item.setText(colour)
                    item.setBackground(self.QColor(colour))
            finally:
                table.blockSignals(blocked)

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
        harmonized_base, harmonized_components, merge_colours = (
            harmonize_merge_colours(
                base,
                components,
                merge_labels=summary["merge_groups"],
            )
        )
        colours_changed = not base["color"].astype(str).equals(
            harmonized_base["color"].astype(str)
        ) or not components["color"].astype(str).equals(
            harmonized_components["color"].astype(str)
        )
        if colours_changed:
            base = harmonized_base
            components = harmonized_components
            self._set_population_colour_columns(base, components)
            self._mark_population_draft_dirty()
        counts = labels.value_counts(dropna=False)
        lines = [
            f"New label column: {self.curation_derived_obs_edit.text().strip()}",
            (
                f"{summary['label_count']:,} effective population(s) across "
                f"{summary['cell_count']:,} cells; "
                f"{summary['split_cell_count']:,} cells currently use "
                "split-component assignments."
            ),
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
                colour = merge_colours.get(str(label), "")
                colour_text = f" [{colour}]" if colour else ""
                lines.append(
                    f"  • {label}{colour_text} ← {', '.join(contributors)}"
                )
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
        colour_collisions = categorical_colour_collisions(
            colour_rows["proposed_label"].astype(str).tolist(),
            colour_rows["color"].astype(str).tolist(),
        )
        if colour_collisions:
            lines.append("")
            lines.append(
                "WARNING — different final populations share a colour. Change these "
                "before saving:"
            )
            for colour, names in colour_collisions.items():
                lines.append(f"  • {colour}: {', '.join(names)}")
        self.population_merge_preview.setPlainText("\n".join(lines))

        for table, name_column, colour_column in (
            (self.population_base_table, 2, 3),
            (self.population_components_table, 4, 5),
        ):
            self._style_colour_mapping_table(
                table,
                name_column=name_column,
                colour_column=colour_column,
            )
            for row in range(table.rowCount()):
                colour_item = table.item(row, colour_column)
                name_item = table.item(row, name_column)
                colour = self.QColor(colour_item.text().strip())
                canonical = colour.name().lower() if colour.isValid() else ""
                if canonical not in colour_collisions:
                    continue
                warning = (
                    f"Colour collision: {canonical} is assigned to different final "
                    f"populations: {', '.join(colour_collisions[canonical])}."
                )
                name_item.setToolTip(warning)
                colour_item.setToolTip(warning)
        self._refresh_population_naming_readiness()
        if colour_collisions:
            self.population_naming_readiness_label.setText(
                f"● Colour collision — {len(colour_collisions)} colour(s) are used "
                "by different final population names. Resolve them before saving."
            )
            self.population_naming_readiness_label.setStyleSheet(
                "background: #fee2e2; color: #991b1b; border: 1px solid #ef4444; "
                "border-radius: 6px; padding: 7px; font-weight: 700;"
            )
            self.save_population_draft_button.setEnabled(False)

    def _set_population_colour_columns(
        self,
        base_mapping: pd.DataFrame,
        components: pd.DataFrame,
    ) -> None:
        """Update editable colour cells without rebuilding either table."""

        for table, frame, column in (
            (self.population_base_table, base_mapping, 3),
            (self.population_components_table, components, 5),
        ):
            blocked = table.blockSignals(True)
            try:
                for row, colour_text in enumerate(frame["color"].astype(str)):
                    item = table.item(row, column)
                    if item is None:
                        continue
                    item.setText(colour_text)
                    colour = self.QColor(colour_text)
                    item.setBackground(
                        colour if colour.isValid() else self.QColor("#fecaca")
                    )
            finally:
                table.blockSignals(blocked)
        self._style_colour_mapping_table(
            self.population_base_table,
            name_column=2,
            colour_column=3,
        )
        self._style_colour_mapping_table(
            self.population_components_table,
            name_column=4,
            colour_column=5,
        )

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
        self._refresh_population_merge_preview()
        base, components = self._population_tables_to_frames()
        colour_rows = pd.concat(
            [
                base[["proposed_label", "color"]],
                components[["proposed_label", "color"]]
                if not components.empty
                else pd.DataFrame(columns=["proposed_label", "color"]),
            ],
            ignore_index=True,
        )
        colour_collisions = categorical_colour_collisions(
            colour_rows["proposed_label"].astype(str).tolist(),
            colour_rows["color"].astype(str).tolist(),
        )
        if colour_collisions:
            detail = "; ".join(
                f"{colour}: {', '.join(names)}"
                for colour, names in colour_collisions.items()
            )
            raise ValueError(
                "Different final populations cannot share one colour. " + detail
            )
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

    def auto_colour_population_draft(self) -> None:
        """Assign one distinct Colour Helper colour per effective final label."""

        if self.population_draft is None or self.adata is None:
            raise RuntimeError("Create or load population naming work first.")
        base, components = self._population_tables_to_frames()
        effective, _summary = synthesize_population_labels(
            self.adata,
            source_obs=self.population_draft.source_obs,
            base_mapping=base,
            components=components,
            membership=self.population_membership,
        )
        labels = list(
            dict.fromkeys(
                [
                    *base["proposed_label"].astype(str).str.strip().tolist(),
                    *components["proposed_label"].astype(str).str.strip().tolist(),
                ]
            )
        )
        counts = {
            str(label): int(count)
            for label, count in effective.value_counts(dropna=True).items()
        }
        assignment = self._choose_automatic_colours(
            labels,
            counts,
            context="effective populations",
        )
        if assignment is None:
            return
        for table, name_column, colour_column in (
            (self.population_base_table, 2, 3),
            (self.population_components_table, 4, 5),
        ):
            blocked = table.blockSignals(True)
            try:
                for row in range(table.rowCount()):
                    label = table.item(row, name_column).text().strip()
                    colour = assignment.get(label)
                    if colour:
                        table.item(row, colour_column).setText(colour)
            finally:
                table.blockSignals(blocked)
        self._mark_population_draft_dirty()
        self._refresh_population_merge_preview()
        self.set_status(
            f"Colour Helper assigned {len(assignment):,} distinct population colours."
        )

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
        # Earlier versions rendered the first eligible ROI as a temporary
        # ``cohort_preview`` labels layer.  That layer was not part of ROI
        # navigation or the frozen experiment state, so it became an orphan as
        # soon as the workflow started.  Keep the useful validation summary,
        # but remove any stale preview left by an older session/build.
        self._remove_layers(("cohort_preview",))
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
        masks_folder = self.masks_edit.text().strip()
        masks = (
            discover_mask_files(masks_folder)
            if masks_folder and Path(masks_folder).expanduser().is_dir()
            else {}
        )
        missing_masks: list[str] = []
        missing_ids = 0
        unmatched_ids = 0
        for roi, group in self.preview.eligible_cells.groupby("ROI", observed=True):
            path = masks.get(str(roi))
            if path is None:
                missing_masks.append(str(roi))
                continue
            full_mask = load_mask(path)
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

    @staticmethod
    def _normalise_marker_selection_name(value: object) -> str:
        text = str(value).split(" [", 1)[0].strip()
        return "".join(
            character for character in text if character.isalnum()
        ).casefold()

    def _feature_marker_names(self) -> tuple[list[str], str]:
        """Resolve image markers represented by the active synthetic features."""

        active_feature_set = (
            self.manifest.active_feature_set_id if self.manifest is not None else None
        )
        if (
            active_feature_set
            and self.paths is not None
            and self.paths.feature_dictionary.is_file()
            and self.paths.feature_manifest.is_file()
        ):
            try:
                provenance = json.loads(
                    self.paths.feature_manifest.read_text(encoding="utf-8")
                )
                dictionary = read_dataframe(self.paths.feature_dictionary)
            except (OSError, ValueError, json.JSONDecodeError):
                provenance = {}
                dictionary = pd.DataFrame()
            if (
                provenance.get("feature_set_id") == active_feature_set
                and "feature" in dictionary
            ):
                feature_names = dictionary["feature"].dropna().astype(str).tolist()
                if "valid_model_input" in dictionary:
                    usable = dictionary["valid_model_input"].map(
                        lambda value: str(value).strip().casefold()
                        not in {"", "0", "false", "nan", "no", "none"}
                    )
                    feature_names = (
                        dictionary.loc[usable, "feature"].dropna().astype(str).tolist()
                    )
                recipe = compact_synthetic_recipe(
                    SyntheticFeatureRecipe(),
                    feature_names,
                )
                channels = list(dict.fromkeys(recipe.channels))
                if channels:
                    return channels, "the active built feature table"
                raise ValueError(
                    "The active feature table contains no channel-derived IMC "
                    "features, so there are no staining markers to select."
                )

        recipe = self.synthetic_recipe_from_controls()
        if recipe.channels:
            return list(recipe.channels), "the current synthetic-feature recipe"
        if not (
            recipe.distribution_features
            or recipe.region_features
            or recipe.gradient_features
        ):
            raise ValueError(
                "The current feature recipe contains only mask/context features; "
                "it does not use staining markers."
            )

        available: list[str] = []
        if self.adata is not None:
            available.extend(str(value) for value in self.adata.var_names)
        available.extend(
            str(value).split(" [", 1)[0] for value in self.current_image_paths
        )
        if hasattr(self, "feature_channel_list"):
            available.extend(
                self.feature_channel_list.item(index).text()
                for index in range(self.feature_channel_list.count())
            )
        if hasattr(self, "channel_list"):
            available.extend(
                self.channel_list.item(index).text().split(" [", 1)[0]
                for index in range(self.channel_list.count())
            )
        channels = list(dict.fromkeys(value for value in available if value))
        if not channels:
            raise ValueError(
                "The feature recipe uses every discovered channel, but no marker "
                "choices are currently available. Load AnnData or an ROI first."
            )
        return channels, "the all-channel synthetic-feature recipe"

    def _select_feature_markers_in_list(self, marker_list, *, destination: str) -> None:
        markers, source = self._feature_marker_names()
        target_names = {
            self._normalise_marker_selection_name(marker)
            for marker in markers
            if self._normalise_marker_selection_name(marker)
        }
        aliases = self._channel_aliases()
        for marker in markers:
            canonical = aliases.get(self._normalise_marker_selection_name(marker))
            if canonical:
                target_names.add(self._normalise_marker_selection_name(canonical))

        matched_indices: list[int] = []
        for index in range(marker_list.count()):
            item = marker_list.item(index)
            item_name = self._normalise_marker_selection_name(item.text())
            canonical = aliases.get(item_name)
            candidate_names = {item_name}
            if canonical:
                candidate_names.add(self._normalise_marker_selection_name(canonical))
            if candidate_names & target_names:
                matched_indices.append(index)
        if not matched_indices:
            raise ValueError(
                f"None of the {len(markers)} marker(s) from {source} are available "
                f"in {destination}. Check the selected expression source or load "
                "an ROI containing those image channels."
            )
        marker_list.blockSignals(True)
        marker_list.clearSelection()
        for index in matched_indices:
            marker_list.item(index).setSelected(True)
        marker_list.blockSignals(False)
        missing_count = max(0, len(markers) - len(matched_indices))
        suffix = (
            f" {max(0, missing_count)} feature marker(s) were unavailable here."
            if missing_count > 0
            else ""
        )
        self.set_status(
            f"Selected {len(matched_indices)} marker(s) in {destination} using "
            f"{source}."
            f"{suffix}"
        )

    def select_feature_build_channels(self) -> None:
        self._select_feature_markers_in_list(
            self.feature_channel_list,
            destination="Feature Building",
        )
        self._update_feature_channel_summary()

    def select_feature_image_channels(self) -> None:
        self._select_feature_markers_in_list(
            self.channel_list,
            destination="Explore image channels",
        )

    def select_feature_marker_overlays(self) -> None:
        self._select_feature_markers_in_list(
            self.marker_overlay_list,
            destination="Explore marker overlays",
        )

    def select_scanpy_expression_feature_markers(self) -> None:
        self.scanpy_plotting_panel.marker_search_edit.clear()
        self._select_feature_markers_in_list(
            self.scanpy_plotting_panel.marker_list,
            destination="Scanpy expression markers",
        )
        self.scanpy_plotting_panel._controls_changed()

    def select_scanpy_embedding_feature_markers(self) -> None:
        self.scanpy_plotting_panel.embedding_marker_search_edit.clear()
        self._select_feature_markers_in_list(
            self.scanpy_plotting_panel.embedding_marker_list,
            destination="Scanpy embedding expression variables",
        )
        self.scanpy_plotting_panel._controls_changed()

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
        channels = self._ordered_variable_values(channels)
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
        self.refresh_feature_readiness()

    def _feature_scope_counts(self) -> tuple[int, int]:
        """Return the intended cell/ROI coverage for the active build scope."""

        if self.manifest is None:
            return 0, 0
        if self.cohort.empty:
            return (
                int(self.manifest.cell_scope.eligible_cell_count),
                int(self.manifest.cell_scope.represented_roi_count),
            )
        working = self.cohort
        if (
            self.manifest.experiment_mode == "feature_discovery_trial"
            and self.manifest.feature_trial is not None
        ):
            trial_rois = set(self.manifest.feature_trial.selected_rois)
            working = working.loc[working["ROI"].astype(str).isin(trial_rois)]
        return len(working), int(working["ROI"].astype(str).nunique())

    def _normalise_feature_recipe_payload(self, recipe) -> dict:
        payload = recipe.model_dump(mode="json")
        normalization_path = payload.get("normalization_dict_path")
        if not normalization_path:
            return payload
        candidate = Path(str(normalization_path)).expanduser()
        if not candidate.is_absolute() and self.paths is not None:
            experiment_candidate = self.paths.root / candidate
            project_candidate = self.project_root / candidate
            candidate = (
                experiment_candidate
                if experiment_candidate.is_file()
                else project_candidate
            )
        payload["normalization_dict_path"] = os.path.normcase(
            str(candidate.resolve(strict=False))
        )
        return payload

    def _feature_controls_match_manifest(self) -> tuple[bool, str | None]:
        """Check whether current builder controls describe the saved recipe."""

        if self.manifest is None:
            return True, None
        try:
            current_recipe = self._normalise_feature_recipe_payload(
                self.synthetic_recipe_from_controls()
            )
            saved_recipe = self._normalise_feature_recipe_payload(
                self.manifest.synthetic_features
            )
            current_sources = [
                source.model_dump(mode="json") for source in self.feature_sources()
            ]
            saved_sources = [
                source.model_dump(mode="json")
                for source in self.manifest.feature_sources
                if source.enabled
            ]
        except (TypeError, ValueError) as exc:
            return False, f"The current recipe is incomplete: {exc}"
        if current_recipe != saved_recipe:
            return False, "The synthetic-feature controls differ from the saved build."
        if current_sources != saved_sources:
            return (
                False,
                "The imported feature-source list differs from the saved build.",
            )
        return True, None

    def _set_feature_readiness_display(
        self,
        state: str,
        title: str,
        detail: str,
        next_step: str,
        *,
        coverage_value: int = 0,
        coverage_maximum: int = 1,
        coverage_format: str = "No feature table",
    ) -> None:
        if not hasattr(self, "feature_readiness_banner"):
            return
        styles = {
            "idle": (
                "#f1f5f9",
                "#334155",
                "#94a3b8",
            ),
            "working": (
                "#dbeafe",
                "#1e40af",
                "#3b82f6",
            ),
            "ready": (
                "#dcfce7",
                "#166534",
                "#22c55e",
            ),
            "warning": (
                "#fef3c7",
                "#92400e",
                "#f59e0b",
            ),
            "error": (
                "#fee2e2",
                "#991b1b",
                "#ef4444",
            ),
        }
        background, foreground, border = styles[state]
        self.feature_readiness_banner.setText(title)
        self.feature_readiness_banner.setStyleSheet(
            f"background: {background}; color: {foreground}; "
            f"border: 2px solid {border}; border-radius: 8px; padding: 10px; "
            "font-size: 16px; font-weight: 900;"
        )
        self.feature_readiness_detail.setText(detail)
        self.feature_readiness_next_step.setText(f"Next: {next_step}")
        maximum = max(1, int(coverage_maximum))
        self.feature_readiness_coverage.setRange(0, maximum)
        self.feature_readiness_coverage.setValue(
            min(maximum, max(0, int(coverage_value)))
        )
        self.feature_readiness_coverage.setFormat(coverage_format)

    def refresh_feature_readiness(self, *_args) -> None:
        """Summarize durable feature assets independently of live progress."""

        if not hasattr(self, "feature_readiness_banner"):
            return
        expected_cells, expected_rois = self._feature_scope_counts()
        if self.feature_process is not None:
            total = int(self.feature_progress_state.get("total_rois", 0) or 0)
            completed = int(
                self.feature_progress_state.get("completed_rois", 0) or 0
            )
            failed = int(self.feature_progress_state.get("failed_rois", 0) or 0)
            processed = completed + failed
            self._set_feature_readiness_display(
                "working",
                "⏳ Building features — the final table is not ready yet",
                f"{completed:,} ROI(s) complete, {failed:,} failed, and "
                f"{max(0, total - processed):,} pending. The process-health panel "
                "below confirms that the Python worker is still reporting.",
                "wait for this banner to become Ready, or cancel safely and "
                "resume later.",
                coverage_value=processed,
                coverage_maximum=max(1, total),
                coverage_format=(
                    f"{processed:,}/{total:,} ROIs processed"
                    if total
                    else "Starting feature worker…"
                ),
            )
            return
        if self.manifest is None or self.paths is None:
            self._set_feature_readiness_display(
                "idle",
                "⚪ No workspace — feature status is unavailable",
                "Create or load a classification workspace to configure and "
                "build features.",
                "create or load a workspace in Setup.",
            )
            return

        active_id = self.manifest.active_feature_set_id
        required = {
            "canonical feature table": self.paths.feature_table,
            "feature dictionary": self.paths.feature_dictionary,
            "feature provenance": self.paths.feature_manifest,
        }
        existing = [label for label, path in required.items() if path.is_file()]
        if not active_id:
            detail = (
                "Some feature files exist, but this experiment revision does not "
                "identify them as its active feature set. They are not treated as "
                "ready for training."
                if existing
                else f"No canonical feature table has been built for the current "
                f"scope of {expected_cells:,} cells across {expected_rois:,} ROIs."
            )
            self._set_feature_readiness_display(
                "warning" if existing else "idle",
                "⚠ Features are not built for this experiment revision"
                if existing
                else "⚪ Features have not been built",
                detail,
                "review sources and the synthetic recipe, then click Build/resume "
                "features locally.",
                coverage_maximum=max(1, expected_cells),
                coverage_format=(
                    f"0/{expected_cells:,} cells in the active feature table"
                ),
            )
            return

        missing = [label for label, path in required.items() if not path.is_file()]
        if missing:
            self._set_feature_readiness_display(
                "error",
                "❌ Feature build is incomplete",
                "The experiment names an active feature set, but the following "
                f"required asset(s) are missing: {', '.join(missing)}.",
                "run Build/resume to reconstruct the missing canonical outputs.",
                coverage_maximum=max(1, expected_cells),
                coverage_format="Required feature assets are missing",
            )
            return
        try:
            provenance = json.loads(
                self.paths.feature_manifest.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            self._set_feature_readiness_display(
                "error",
                "❌ Feature provenance cannot be read",
                "The canonical feature table exists, but its provenance is "
                f"invalid: {exc}",
                "run Build/resume before training so the outputs are recreated safely.",
                coverage_maximum=max(1, expected_cells),
                coverage_format="Feature provenance is invalid",
            )
            return

        built_cells = int(provenance.get("eligible_cells", 0) or 0)
        built_rois = int(provenance.get("represented_rois", 0) or 0)
        feature_count = int(provenance.get("feature_count", 0) or 0)
        failures = int(provenance.get("failures", 0) or 0)
        mismatches = []
        for actual, expected, label in (
            (
                provenance.get("feature_extraction_contract_version"),
                FEATURE_EXTRACTION_CONTRACT_VERSION,
                "feature-extraction contract",
            ),
            (
                provenance.get("experiment_id"),
                self.manifest.experiment_id,
                "experiment",
            ),
            (provenance.get("experiment_revision"), self.manifest.revision, "revision"),
            (
                provenance.get("cohort_sha256"),
                self.manifest.cell_scope.snapshot_sha256,
                "frozen cohort",
            ),
            (provenance.get("feature_set_id"), active_id, "active feature set"),
        ):
            if actual != expected:
                mismatches.append(label)
        saved_recipe = provenance.get("recipe")
        if isinstance(saved_recipe, dict):
            current_saved_recipe = self._normalise_feature_recipe_payload(
                self.manifest.synthetic_features
            )
            try:
                provenance_recipe = self._normalise_feature_recipe_payload(
                    SyntheticFeatureRecipe.model_validate(saved_recipe)
                )
            except ValueError:
                provenance_recipe = {}
            if provenance_recipe != current_saved_recipe:
                mismatches.append("saved feature recipe")
        if mismatches:
            self._set_feature_readiness_display(
                "error",
                "❌ Saved features are stale for the current experiment",
                f"The table contains {built_cells:,} cells and {feature_count:,} "
                f"model features, but its provenance does not match the current "
                f"{', '.join(mismatches)}.",
                "rebuild features before training or scoring.",
                coverage_value=built_cells,
                coverage_maximum=max(1, expected_cells),
                coverage_format=f"{built_cells:,}/{expected_cells:,} expected cells",
            )
            return

        controls_match, controls_detail = self._feature_controls_match_manifest()
        warnings = provenance.get("warnings", [])
        warning_count = len(warnings) if isinstance(warnings, list) else 0
        partial = (
            built_cells < expected_cells
            or built_rois < expected_rois
            or failures > 0
            or feature_count <= 0
        )
        completed_at = str(provenance.get("completed_at", "unknown time"))
        details = (
            f"Canonical table: {built_cells:,}/{expected_cells:,} cells across "
            f"{built_rois:,}/{expected_rois:,} ROIs, with {feature_count:,} usable "
            f"model features. Completed {completed_at}. Feature set "
            f"{str(active_id)[:12]}…."
        )
        if failures:
            details += f" {failures:,} ROI build failure(s) were recorded."
        if warning_count:
            details += (
                f" {warning_count:,} build warning(s) are recorded in provenance."
            )
        if not controls_match:
            details += f" {controls_detail}"

        if partial:
            state = "warning"
            title = "⚠ Features were built, but coverage is incomplete"
            next_step = (
                "inspect failed ROIs and warnings, then resume the build before "
                "training."
            )
        elif not controls_match:
            state = "warning"
            title = "✅ Built features are available — controls have unapplied changes"
            next_step = (
                "use the existing table as-is, or rebuild to apply the currently "
                "displayed controls."
            )
        elif warning_count:
            state = "warning"
            title = "✅ Features are ready, with recorded warnings"
            next_step = (
                "review the warnings below, then continue to labelling or training "
                "if acceptable."
            )
        else:
            state = "ready"
            title = (
                "✅ Features are ready for refinement"
                if self.manifest.experiment_mode == "feature_discovery_trial"
                else "✅ Features are ready for classification"
            )
            next_step = (
                "open Feature Refinement and confirm labels across trial ROIs."
                if self.manifest.experiment_mode == "feature_discovery_trial"
                else "continue to Classify; training will use this active feature set."
            )
        self._set_feature_readiness_display(
            state,
            title,
            details,
            next_step,
            coverage_value=built_cells,
            coverage_maximum=max(1, expected_cells),
            coverage_format=f"{built_cells:,}/{expected_cells:,} cells (%p%)",
        )

    def refresh_saved_feature_status(self) -> None:
        """Reload feature-related experiment state after an external/HPC build."""

        if self.paths is not None and self.paths.manifest.is_file():
            stored_manifest, stored_paths = load_experiment(self.paths.root)
            if (
                self.manifest is not None
                and stored_manifest.experiment_id != self.manifest.experiment_id
            ):
                raise ValueError(
                    "The saved feature manifest belongs to a different experiment."
                )
            if (
                self.manifest is not None
                and stored_manifest.revision != self.manifest.revision
            ):
                raise ValueError(
                    "The saved experiment revision changed externally. Reload the "
                    "workspace from Setup before using its new feature table."
                )
            previous_feature_set = (
                self.manifest.active_feature_set_id
                if self.manifest is not None
                else None
            )
            self.manifest = stored_manifest
            self.paths = stored_paths
            if previous_feature_set != self.manifest.active_feature_set_id:
                self.model_bundle = None
                self.scores = pd.DataFrame()
                self.final_assignments = pd.DataFrame()
                self.final_identity_signature = None
                self._refresh_model_storage_label()
        self.refresh_feature_readiness()
        self.refresh_refinement_readiness()
        self.set_status("Refreshed feature readiness from the saved experiment assets.")

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
        if self.publication_batch is not None:
            raise ValueError(
                "Cancel the active publication bulk export before changing workspace."
            )
        if self.publication_export_dialog is not None:
            self.publication_export_dialog.hide()
        self.manifest, self.paths = load_experiment(path)
        # A workspace can share ROI names with the previously open workspace.
        # Reset the loaded-ROI identity so its first ROI is always a real load,
        # while later model/view data changes can safely ignore same-text combo
        # notifications.
        self.current_roi = None
        self.current_mask = None
        self.current_mask_path = None
        self.current_image_paths.clear()
        self._clear_explore_layers()
        self._remove_layers([ALL_CELLS_LAYER_NAME])
        self._clear_explore_layer_data_cache()
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
        self.integrated_identity_table = pd.DataFrame()
        self.identity_integration_signature = None
        self.identity_integration_plan = {}
        self._identity_integration_custom_names = {}
        integration_default = self.manifest.cell_scope.mode == "obs_values"
        blocked = self.final_integration_enable_check.blockSignals(True)
        self.final_integration_enable_check.setChecked(integration_default)
        self.final_integration_enable_check.blockSignals(blocked)
        self.final_integration_output_edit.setText(
            f"{self.manifest.output_obs_slug}_combined"
        )
        self.identity_integration_summary_label.setText(
            "Subset classification detected. Build optional integrated labels "
            "after creating final identities."
            if integration_default
            else "Optional integration is disabled; exports will contain "
            "cohort-only final identities."
        )
        self._update_identity_integration_controls()
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
        self._load_publication_export_state()
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
                source_obs = str(self.manifest.cell_scope.obs_column or "")
                source_index = self.final_integration_source_combo.findText(source_obs)
                if source_index >= 0:
                    blocked = self.final_integration_source_combo.blockSignals(True)
                    self.final_integration_source_combo.setCurrentIndex(source_index)
                    self.final_integration_source_combo.blockSignals(blocked)
                    self._refresh_identity_integration_mapping()
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
        if (
            self.roi_combo.count()
            and self.manifest.workflow_mode != "dataset_maintenance"
        ):
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
            self._refresh_population_qc_scope_banner()
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
        self._refresh_population_qc_scope_banner()

    def refresh_class_controls(self) -> None:
        current_class_id = self.class_combo.currentData()
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
            selected_index = self.class_combo.findData(current_class_id)
            if selected_index >= 0:
                self.class_combo.setCurrentIndex(selected_index)
            self.queue_roi_combo.clear()
            self.queue_roi_combo.addItem("All current experiment ROIs", None)
            queue_rois = (
                sorted(self.cohort["ROI"].astype(str).unique())
                if "ROI" in self.cohort
                else []
            )
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
        hotkey_parts = []
        for definition in self.manifest.classes:
            shortcut = definition.shortcut

            def select_class(
                _viewer,
                selected_class_id=definition.class_id,
                selected_shortcut=shortcut,
            ):
                self._select_class_from_hotkey(
                    selected_class_id,
                    selected_shortcut,
                )

            self.viewer.bind_key(shortcut, overwrite=True)(select_class)
            self._class_shortcuts.append(shortcut)
            hotkey_parts.append(
                f'<span style="color: {definition.color};">■</span> '
                f"<b>{escape(shortcut)}</b> = {escape(definition.name)}"
            )
        self.class_hotkey_label.setText(
            " &nbsp;&nbsp; ".join(hotkey_parts)
            + "<br><span style=\"color: #9ca3af;\">Selects the class only; "
            "the current click action is unchanged.</span>"
        )
        self._refresh_class_tally()
        self._refresh_model_storage_label()
        self._refresh_queue_if_scored()
        self._refresh_identity_integration_mapping()

    def _select_class_from_hotkey(self, class_id: str, shortcut: str) -> None:
        """Select one class while deliberately preserving annotation behaviour."""

        index = self.class_combo.findData(str(class_id))
        if index < 0:
            return
        # Only the combo selection changes. In particular, do not touch the
        # click-behaviour button group: users can keep proposing or confirming
        # cells while rapidly moving between classes.
        self.class_combo.setCurrentIndex(index)
        definition = self._class_definition(str(class_id))
        class_name = definition.name if definition is not None else str(class_id)
        checked_action = self.click_behavior_group.checkedButton()
        action_text = (
            checked_action.text() if checked_action is not None else "Select only"
        )
        self.set_status(
            f"Hotkey {shortcut}: selected class {class_name!r}. "
            f"Click action remains {action_text!r}."
        )

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
        self._set_all_layer_visibility(False)
        self.set_status("All Napari layers are hidden.")

    def show_all_layers(self) -> None:
        self._set_all_layer_visibility(True)
        self.set_status("All Napari layers are visible.")

    def _set_all_layer_visibility(self, visible: bool) -> None:
        """Change every layer while coalescing live-recipe bookkeeping."""

        previous_state = self._updating_recipe_layer_state
        self._updating_recipe_layer_state = True
        tracked_names: list[str] = []
        try:
            for layer in self.viewer.layers:
                layer.visible = bool(visible)
                name = str(getattr(layer, "name", ""))
                if self._is_recipe_tracked_layer(name, layer):
                    tracked_names.append(name)
        finally:
            self._updating_recipe_layer_state = previous_state
        if "excluded_segmentation_context" in self.viewer.layers:
            self.context_check_display.blockSignals(True)
            self.context_check_display.setChecked(bool(visible))
            self.context_check_display.blockSignals(False)
        if not self._recipe_tracking_enabled():
            return
        for name in tracked_names:
            self.explore_recipe.layer_visibility[name] = bool(visible)
        self._refresh_active_recipe_preset_label()
        self._refresh_reload_recipe_list()
        self._refresh_roi_review_colours()

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

    def _publication_export_state_path(self) -> Path | None:
        if self.paths is None:
            return None
        return self.paths.root / "explore" / "publication_export_presets.json"

    def _load_publication_export_state(self) -> None:
        """Load the cold publication-preset catalogue for this workspace."""

        self.publication_export_state = PublicationExportState()
        state_path = self._publication_export_state_path()
        if state_path is None or not state_path.is_file():
            return
        try:
            self.publication_export_state = PublicationExportState.model_validate(
                json.loads(state_path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.set_status(
                "Publication export presets could not be loaded; the saved file "
                f"was left untouched: {error}"
            )

    def _save_publication_export_state(self) -> None:
        state_path = self._publication_export_state_path()
        if state_path is None:
            raise ValueError(
                "Create or load a workflow workspace before saving publication presets."
            )
        self.publication_export_state.schema_version = (
            PUBLICATION_EXPORT_SCHEMA_VERSION
        )
        write_json(
            state_path,
            self.publication_export_state.model_dump(mode="json"),
        )

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
            self._sync_cell_properties_controls()
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
        self._sync_cell_properties_controls()
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
        if self._publication_export_running:
            self.set_status(
                "Publication bulk export is using a frozen recipe; recipe switching "
                "is available again when the batch finishes or is cancelled."
            )
            return
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
        if spec and spec.get("kind") == "categorical_labels":
            colours = spec.get("colours", {})
            if isinstance(colours, dict) and colours:
                return f"{len(colours)} saved category colours"
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
                if name == ALL_CELLS_LAYER_NAME:
                    relevant = (
                        name in configured_names or name in self.viewer.layers
                    )
                else:
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
        elif self._active_recipe_label_refresh_pending:
            self._active_recipe_label_refresh_pending = False
            self._refresh_active_recipe_preset_label()
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
        self._active_recipe_label_refresh_pending = False
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
            name = str(entry["name"])
            availability = self._recipe_entry_available(entry)
            stored_entry = dict(entry)
            stored_entry["_availability"] = availability
            item = QListWidgetItem(
                self._recipe_entry_display_text(stored_entry, availability)
            )
            item.setData(self.Qt.UserRole, stored_entry)
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

    def _recipe_entry_display_text(
        self,
        entry: dict,
        availability: bool | None,
    ) -> str:
        """Format one recipe row without rechecking AnnData or image coverage."""

        name = str(entry["name"])
        visible_default = (
            MANAGED_LAYER_DEFAULT_VISIBILITY[name]
            if entry["kind"] == "managed"
            else True
        )
        opacity_default = (
            MANAGED_LAYER_DEFAULT_OPACITY[name]
            if entry["kind"] == "managed"
            else 1.0
        )
        visible = self.explore_recipe.layer_visibility.get(name, visible_default)
        opacity = self.explore_recipe.layer_opacities.get(name, opacity_default)
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
        availability_text = ""
        if availability is False:
            availability_text = " — ⚠ absent in this workflow or ROI"
        elif availability is None:
            availability_text = " — availability checked when an ROI is loaded"
        return (
            f"{entry['description']} — {state}, opacity {opacity:.2f}"
            f"{contour_text}{contrast_text}{availability_text}"
        )

    def _refresh_reload_recipe_item(self, name: str) -> None:
        """Refresh one visible row after a display-only event.

        Availability cannot change when a layer is merely hidden, recoloured, or
        given a new opacity. Reusing the result from the full refresh avoids an
        AnnData population scan for every Napari display event.
        """

        if not hasattr(self, "reload_recipe_list"):
            return
        if (
            hasattr(self, "explore_tab_index")
            and self.tabs.currentIndex() != self.explore_tab_index
        ):
            self._recipe_list_refresh_pending = True
            return
        for index in range(self.reload_recipe_list.count()):
            item = self.reload_recipe_list.item(index)
            entry = item.data(self.Qt.UserRole)
            if not isinstance(entry, dict) or str(entry.get("name", "")) != name:
                continue
            availability = entry.get("_availability")
            item.setText(self._recipe_entry_display_text(entry, availability))
            return
        # The layer was added after the last list build, so a full refresh is
        # necessary once. Existing rows stay on the incremental fast path.
        self._refresh_reload_recipe_list()

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
        categorical_spec = self._categorical_layer_colormap_spec(layer)
        if categorical_spec is not None:
            return categorical_spec
        try:
            from napari.utils.colormaps import CyclicLabelColormap
        except ImportError:  # pragma: no cover - guarded by the GUI dependency
            CyclicLabelColormap = ()
        if isinstance(colormap, CyclicLabelColormap):
            rgba = np.asarray(colormap.colors, dtype=float)
            if (
                rgba.ndim == 2
                and rgba.shape[1] in {3, 4}
                and np.isfinite(rgba).all()
            ):
                return {
                    "kind": "cyclic_labels",
                    "colours": rgba.tolist(),
                    "background_value": int(colormap.background_value),
                }
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
                if spec.get("kind") == "cyclic_labels":
                    from napari.utils.colormaps import CyclicLabelColormap

                    return CyclicLabelColormap(
                        colors=spec["colours"],
                        background_value=int(spec.get("background_value", 0)),
                    )
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

    def _single_colour_label_colormap(self, colour):
        """Map every non-background label ID to one display colour."""

        from napari.utils.colormaps import CyclicLabelColormap

        if isinstance(colour, str):
            colours = ["#00000000", colour]
        else:
            rgba = np.asarray(colour, dtype=float).reshape(-1)
            if rgba.size == 3:
                rgba = np.append(rgba, 1.0)
            colours = [[0.0, 0.0, 0.0, 0.0], rgba.tolist()]
        return CyclicLabelColormap(
            colors=colours,
            background_value=0,
        )

    def _population_layer_colormap(self, name: str, default_colour: str):
        """Restore one population colour, including from legacy binary recipes."""

        spec = self.explore_recipe.layer_colormap_specs.get(name)
        if spec:
            colours = spec.get("colours", {})
            if spec.get("kind") == "direct_labels" and isinstance(colours, dict):
                colour = colours.get("1")
                if colour is None:
                    colour = next(
                        (
                            value
                            for key, value in colours.items()
                            if key != "__default__"
                        ),
                        None,
                    )
                if colour is not None:
                    return self._single_colour_label_colormap(colour)
            if spec.get("kind") in {"cyclic_labels", "continuous"} and isinstance(
                colours, list
            ):
                candidates = (
                    colours[1:]
                    if spec.get("kind") == "cyclic_labels"
                    else colours
                )
                for colour in reversed(candidates):
                    rgba = np.asarray(colour, dtype=float).reshape(-1)
                    if rgba.size in {3, 4} and (
                        rgba.size == 3 or float(rgba[3]) > 0
                    ):
                        return self._single_colour_label_colormap(colour)
        return self._single_colour_label_colormap(default_colour)

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
                self.roi_combo.setItemData(index, None, self.Qt.ForegroundRole)
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
                self.QColor("#111827"),
                self.Qt.ForegroundRole,
            )
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

    def _is_recipe_tracked_layer(self, name: str, layer=None) -> bool:
        if name in MANAGED_RECIPE_LAYERS:
            return True
        if layer is not None:
            return self._layer_reload_descriptor(layer) is not None
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

        callbacks = {}
        for event_name in (
            "visible",
            "opacity",
            "contour",
            "contrast_limits",
            "colormap",
        ):
            emitter = getattr(events, event_name, None)
            if emitter is not None:
                def display_changed(
                    _event=None,
                    tracked_layer=layer,
                    property_name=event_name,
                ):
                    self._record_layer_display_state(
                        tracked_layer,
                        property_name=property_name,
                    )

                emitter.connect(display_changed)
                callbacks[event_name] = display_changed
        layer._napari_sbt_recipe_display_callbacks = callbacks
        layer._napari_sbt_recipe_display_bound = True

    def _record_layer_display_state(
        self,
        layer,
        *,
        property_name: str | None = None,
    ) -> None:
        """Record one display change without rebuilding or rescanning the view.

        Napari emits a specific event for visibility, opacity, contour, contrast,
        and colormap changes. The former implementation treated every event as a
        request to serialize all five properties and rebuild the complete recipe
        list. In particular, hiding a layer serialized its full colormap and
        rescanned categorical AnnData values. Keep ordinary toggles on a small,
        in-memory path; full capture remains available through the explicit UI.
        """

        if self._updating_recipe_layer_state or not self._recipe_tracking_enabled():
            return
        name = str(getattr(layer, "name", ""))
        if not self._is_recipe_tracked_layer(name, layer):
            return

        if property_name in {"visible", "opacity"}:
            if property_name == "visible":
                self.explore_recipe.layer_visibility[name] = bool(
                    getattr(layer, "visible", True)
                )
                if name == "excluded_segmentation_context":
                    self.context_check_display.blockSignals(True)
                    self.context_check_display.setChecked(bool(layer.visible))
                    self.context_check_display.blockSignals(False)
            else:
                self.explore_recipe.layer_opacities[name] = float(
                    getattr(layer, "opacity", 1.0)
                )
            # Visibility and opacity are the common hot paths, especially for
            # full-size classifier label layers. Opacity emits repeatedly while
            # its slider is dragged. Do not hash the complete recipe or recolour
            # every ROI selector entry while Napari is repainting the layer. The
            # scalar value is already authoritative for the next replay; heavier
            # derived UI state is refreshed on entering Explore or during the
            # next normal recipe/ROI refresh.
            self._active_recipe_label_refresh_pending = True
            self._roi_review_refresh_pending = True
            if property_name == "opacity":
                # A slider drag can emit many events. Do not even rewrite the
                # recipe-list row for each intermediate value; its source value
                # is already current and the text is refreshed by the next
                # normal Explore/recipe update.
                self._recipe_list_refresh_pending = True
            else:
                self._refresh_reload_recipe_item(name)
            return
        elif property_name == "contour" and hasattr(layer, "contour"):
            self.explore_recipe.layer_contours[name] = int(layer.contour)
        elif property_name == "contrast_limits" and hasattr(
            layer, "contrast_limits"
        ):
            limits = layer.contrast_limits
            self.explore_recipe.layer_contrast_limits[name] = (
                float(limits[0]),
                float(limits[1]),
            )
        elif property_name == "colormap":
            descriptor = self._layer_reload_descriptor(layer)
            if descriptor is not None and descriptor.get("kind") != "rgb":
                spec = self._layer_colormap_spec(layer)
                if spec is None:
                    self.explore_recipe.layer_colormap_specs.pop(name, None)
                    self.explore_recipe.layer_colormaps.pop(name, None)
                else:
                    self.explore_recipe.layer_colormap_specs[name] = spec
                    colormap_name = self._layer_colormap_name(layer)
                    if colormap_name and spec.get("kind") == "continuous":
                        self.explore_recipe.layer_colormaps[name] = colormap_name
                    else:
                        self.explore_recipe.layer_colormaps.pop(name, None)
        else:
            payload = self.explore_recipe.model_dump(mode="json")
            self._write_layer_display_state(payload, layer)
            self.explore_recipe = ExploreViewRecipe.model_validate(payload)

        if name == "excluded_segmentation_context" and property_name is None:
            self.context_check_display.blockSignals(True)
            self.context_check_display.setChecked(bool(layer.visible))
            self.context_check_display.blockSignals(False)
        self._refresh_active_recipe_preset_label()
        if property_name == "colormap":
            # The row description includes the colour name. This event is much
            # rarer than visibility changes and therefore merits a full rebuild.
            self._refresh_reload_recipe_list()
        else:
            self._refresh_reload_recipe_item(name)
        self._refresh_roi_review_colours()

    @staticmethod
    def _colormap_specs_match(current: dict | None, desired: dict | None) -> bool:
        if not current or not desired or current.get("kind") != desired.get("kind"):
            return False
        try:
            if current["kind"] == "categorical_labels":
                current_colours = current.get("colours", {})
                desired_colours = desired.get("colours", {})
                return set(current_colours) == set(desired_colours) and all(
                    np.allclose(current_colours[key], desired_colours[key])
                    for key in current_colours
                )
            if current["kind"] == "direct_labels":
                current_colours = current.get("colours", {})
                desired_colours = desired.get("colours", {})
                return set(current_colours) == set(desired_colours) and all(
                    np.allclose(current_colours[key], desired_colours[key])
                    for key in current_colours
                )
            if current["kind"] == "cyclic_labels":
                return (
                    int(current.get("background_value", 0))
                    == int(desired.get("background_value", 0))
                    and np.allclose(
                        current.get("colours", []), desired.get("colours", [])
                    )
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
        default_blending = MANAGED_LAYER_DEFAULT_BLENDING.get(name)
        if default_blending is not None:
            # Entropy and selected-class probability are intensity overlays.
            # Reapply their intended blending on both creation and reuse so an
            # older/translucent instance cannot obscure the staining beneath it.
            kwargs.setdefault("blending", default_blending)
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

    def _roi_selection_changed(self, roi: str) -> None:
        """Load only when the ROI selector's biological identity changed."""

        if self._publication_export_running:
            return
        roi = str(roi).strip()
        if not roi:
            return
        if self.current_mask is not None and str(self.current_roi) == roi:
            # Qt may re-emit currentTextChanged when item background, foreground,
            # or tooltip roles change. Those are display-only model updates.
            return
        self.load_roi(roi)

    def load_roi(self, roi: str | None = None) -> None:
        if self.manifest is None:
            return
        roi = str(roi or self.roi_combo.currentText())
        if not roi:
            return
        roi_changed = self.current_mask is None or str(self.current_roi) != roi
        mask_path = self._mask_path_for_roi(roi)
        full_mask = load_mask(mask_path)
        eligible = self._eligible_ids_for_roi(roi)
        self.current_roi = roi
        self.current_mask = full_mask
        self.current_mask_path = mask_path
        configured_layer_names = set().union(
            self.explore_recipe.layer_colormaps,
            self.explore_recipe.layer_colormap_specs,
            self.explore_recipe.layer_visibility,
            self.explore_recipe.layer_opacities,
            self.explore_recipe.layer_contours,
            self.explore_recipe.layer_contrast_limits,
        )
        all_cells_present = ALL_CELLS_LAYER_NAME in self.viewer.layers
        if all_cells_present or ALL_CELLS_LAYER_NAME in configured_layer_names:
            if all_cells_present:
                all_cells_layer = self._replace_layer(
                    ALL_CELLS_LAYER_NAME,
                    full_mask,
                    "labels",
                )
            else:
                all_cells_layer = self._replace_layer(
                    ALL_CELLS_LAYER_NAME,
                    full_mask,
                    "labels",
                    visible=self.explore_recipe.layer_visibility.get(
                        ALL_CELLS_LAYER_NAME,
                        MANAGED_LAYER_DEFAULT_VISIBILITY[ALL_CELLS_LAYER_NAME],
                    ),
                    opacity=self.explore_recipe.layer_opacities.get(
                        ALL_CELLS_LAYER_NAME,
                        MANAGED_LAYER_DEFAULT_OPACITY[ALL_CELLS_LAYER_NAME],
                    ),
                )
                self._set_label_contour_from_recipe(
                    all_cells_layer,
                    ALL_CELLS_LAYER_NAME,
                    MANAGED_LAYER_DEFAULT_CONTOUR[ALL_CELLS_LAYER_NAME],
                )
            if hasattr(all_cells_layer, "editable"):
                all_cells_layer.editable = False
        self.current_selected_object = None
        self.current_labeler_object = None
        self.cell_properties_selected_object = None
        self.selected_cell_label.setText("No cohort cell selected")
        self.labeler_selected_cell_label.setText("No cohort cell selected")
        self.cell_properties_tree.clear()
        self.cell_properties_summary_label.setText(
            "Tracking enabled. Click a segmented cell to inspect it."
            if self.explore_review_state.cell_properties_tracking_enabled
            else "Tracking paused. Enable Track clicks here or in Settings."
        )
        self._remove_layers(
            [
                SELECTED_CELL_LAYER_NAME,
                LABELER_SELECTED_CELL_LAYER_NAME,
                CELL_PROPERTIES_SELECTED_LAYER_NAME,
            ]
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
            roi_changed
            and self.auto_reload_view_check.isChecked()
            and self.explore_recipe.has_content
        )
        if roi_changed and not replay_view:
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

    def auto_reload_setting_changed(self, enabled: bool) -> None:
        """Describe the setting without replaying or loading the current ROI."""

        if enabled:
            message = (
                "Explore recipe replay enabled for the next genuine ROI change. "
                "The current ROI and layers were not reloaded."
            )
        else:
            message = (
                "Explore recipe replay disabled. Changing ROI will load its base "
                "mask/classification layers without recreating Explore images and "
                "overlays; the current ROI was not reloaded."
            )
        self.set_status(message)

    def toggle_context(self, checked: bool) -> None:
        if "excluded_segmentation_context" in self.viewer.layers:
            self.viewer.layers["excluded_segmentation_context"].visible = checked

    def add_all_cells_mask(self) -> None:
        """Show the complete original segmentation already loaded for this ROI."""

        if self.current_mask is None or not self.current_roi:
            raise ValueError("Load an ROI before adding its all-cells mask.")
        layer = self._replace_layer(
            ALL_CELLS_LAYER_NAME,
            self.current_mask,
            "labels",
            visible=True,
            opacity=self.explore_recipe.layer_opacities.get(
                ALL_CELLS_LAYER_NAME,
                MANAGED_LAYER_DEFAULT_OPACITY[ALL_CELLS_LAYER_NAME],
            ),
        )
        self._set_label_contour_from_recipe(
            layer,
            ALL_CELLS_LAYER_NAME,
            MANAGED_LAYER_DEFAULT_CONTOUR[ALL_CELLS_LAYER_NAME],
        )
        if hasattr(layer, "editable"):
            layer.editable = False
        self.viewer.layers.selection.active = layer
        if self._recipe_tracking_enabled():
            self._record_layer_display_state(layer)
        else:
            self._refresh_reload_recipe_list()
        self.set_status(
            f"Added the complete original segmentation for ROI {self.current_roi!r} "
            "as the 'all_cells' labels layer without reloading the ROI."
        )

    def show_publication_export(self) -> None:
        """Open the modeless, reproducible publication-image export window."""

        if self.paths is None or self.manifest is None:
            raise ValueError(
                "Create or load a workflow workspace before exporting images."
            )
        if self.current_mask is None or not self.current_roi:
            raise ValueError("Load an ROI before opening publication export.")
        if self.publication_export_dialog is not None:
            self._refresh_publication_export_controls()
            self.publication_export_dialog.show()
            self.publication_export_dialog.raise_()
            self.publication_export_dialog.activateWindow()
            return

        from qtpy.QtWidgets import (
            QAbstractItemView,
            QCheckBox,
            QComboBox,
            QDialog,
            QDoubleSpinBox,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QProgressBar,
            QPushButton,
            QScrollArea,
            QSpinBox,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )

        dialog = QDialog(self.root)
        dialog.setWindowTitle("NapariSBT publication image export")
        dialog.setModal(False)
        dialog.resize(980, 780)
        layout = QVBoxLayout(dialog)
        intro = QLabel(
            "Create a frozen, reproducible composition from an Explore recipe. "
            "Output size and field of view are independent of the current monitor "
            "or dock layout; original images and masks are never modified."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        top_actions = QHBoxLayout()
        top_actions.addStretch(1)
        help_button = QPushButton("❓ Publication export help")
        help_button.setObjectName("sbtTabHelpButton")
        help_button.clicked.connect(
            lambda: self.show_help(
                "explore",
                "Publication image export",
                section="Publication image export",
            )
        )
        top_actions.addWidget(help_button)
        layout.addLayout(top_actions)

        tabs = QTabWidget()
        layout.addWidget(tabs, 1)

        def publication_help_button():
            button = QPushButton("❓ Help")
            button.setObjectName("sbtBoxHelpButton")
            button.setToolTip("Open instructions for the publication export workflow.")
            button.clicked.connect(
                lambda: self.show_help(
                    "explore",
                    "Publication image export",
                    section="Publication image export",
                )
            )
            return button

        def publication_relative_scale_spin():
            spin = QDoubleSpinBox()
            spin.setRange(10.0, 500.0)
            spin.setDecimals(0)
            spin.setSingleStep(10.0)
            spin.setSuffix(" %")
            spin.setValue(100.0)
            spin.setToolTip(
                "100% keeps the automatic size; lower or higher values scale it."
            )
            return spin

        # View and recipe tab.
        view_tab = QWidget()
        view_layout = QVBoxLayout(view_tab)
        recipe_group = QGroupBox("1. Recipe and reusable publication preset")
        recipe_form = QFormLayout(recipe_group)
        recipe_form.addRow("", publication_help_button())
        self.publication_recipe_combo = QComboBox()
        self.publication_preset_combo = QComboBox()
        self.publication_preset_name_edit = QLineEdit()
        self.publication_preset_name_edit.setPlaceholderText(
            "e.g. Myeloid whole-ROI panel"
        )
        preset_actions = QWidget()
        preset_actions_layout = QHBoxLayout(preset_actions)
        preset_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.publication_save_preset_button = QPushButton("Save as new preset")
        self.publication_update_preset_button = QPushButton("Update selected")
        self.publication_delete_preset_button = QPushButton("Delete selected…")
        preset_actions_layout.addWidget(self.publication_save_preset_button)
        preset_actions_layout.addWidget(self.publication_update_preset_button)
        preset_actions_layout.addWidget(self.publication_delete_preset_button)
        recipe_form.addRow("Explore recipe", self.publication_recipe_combo)
        recipe_form.addRow("Saved publication preset", self.publication_preset_combo)
        recipe_form.addRow("Preset name", self.publication_preset_name_edit)
        recipe_form.addRow("", preset_actions)
        view_layout.addWidget(recipe_group)

        frame_group = QGroupBox("2. Exact frame and field of view")
        frame_form = QFormLayout(frame_group)
        frame_form.addRow("", publication_help_button())
        self.publication_frame_mode_combo = QComboBox()
        self.publication_frame_mode_combo.addItem("Current Napari viewport", "current_view")
        self.publication_frame_mode_combo.addItem("Entire ROI", "full_roi")
        self.publication_frame_mode_combo.addItem("Fixed centre and field of view", "fixed")
        self.publication_aspect_combo = QComboBox()
        self.publication_aspect_combo.addItem("Crop to output aspect", "crop")
        self.publication_aspect_combo.addItem("Pad field to output aspect", "pad")
        self.publication_center_y_spin = QDoubleSpinBox()
        self.publication_center_x_spin = QDoubleSpinBox()
        self.publication_field_height_spin = QDoubleSpinBox()
        self.publication_field_width_spin = QDoubleSpinBox()
        for spin in (
            self.publication_center_y_spin,
            self.publication_center_x_spin,
            self.publication_field_height_spin,
            self.publication_field_width_spin,
        ):
            spin.setRange(-1_000_000_000, 1_000_000_000)
            spin.setDecimals(3)
        self.publication_field_height_spin.setMinimum(0.001)
        self.publication_field_width_spin.setMinimum(0.001)
        frame_actions = QWidget()
        frame_actions_layout = QHBoxLayout(frame_actions)
        frame_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.publication_capture_view_button = QPushButton("Capture current viewport")
        self.publication_use_rectangle_button = QPushButton("Use selected Shapes rectangle")
        self.publication_preview_frame_button = QPushButton("Preview frame in viewer")
        frame_actions_layout.addWidget(self.publication_capture_view_button)
        frame_actions_layout.addWidget(self.publication_use_rectangle_button)
        frame_actions_layout.addWidget(self.publication_preview_frame_button)
        frame_form.addRow("Frame source", self.publication_frame_mode_combo)
        frame_form.addRow("Aspect handling", self.publication_aspect_combo)
        frame_form.addRow("Centre Y", self.publication_center_y_spin)
        frame_form.addRow("Centre X", self.publication_center_x_spin)
        frame_form.addRow("Field height (source pixels)", self.publication_field_height_spin)
        frame_form.addRow("Field width (source pixels)", self.publication_field_width_spin)
        frame_form.addRow("", frame_actions)
        self.publication_frame_summary = QLabel()
        self.publication_frame_summary.setWordWrap(True)
        frame_form.addRow("Resolved frame", self.publication_frame_summary)
        view_layout.addWidget(frame_group)
        view_layout.addStretch(1)
        tabs.addTab(view_tab, "View & recipe")

        # Scale bar and annotation tab.
        appearance_tab = QWidget()
        appearance_layout = QVBoxLayout(appearance_tab)
        calibration_group = QGroupBox("3. Physical calibration and scale bar")
        calibration_form = QFormLayout(calibration_group)
        calibration_form.addRow("", publication_help_button())
        self.publication_calibration_confirmed_check = QCheckBox(
            "I have verified this image-pixel calibration"
        )
        self.publication_pixel_x_spin = QDoubleSpinBox()
        self.publication_pixel_y_spin = QDoubleSpinBox()
        for spin in (self.publication_pixel_x_spin, self.publication_pixel_y_spin):
            spin.setRange(0.000001, 1_000_000)
            spin.setDecimals(6)
            spin.setValue(1.0)
        self.publication_unit_edit = QLineEdit("µm")
        self.publication_detect_calibration_button = QPushButton(
            "Detect from current TIFF metadata"
        )
        self.publication_scale_visible_check = QCheckBox("Draw scale bar")
        self.publication_scale_visible_check.setChecked(False)
        self.publication_scale_mode_combo = QComboBox()
        self.publication_scale_mode_combo.addItem("Automatic nice length", "auto")
        self.publication_scale_mode_combo.addItem("Fixed physical length", "fixed")
        self.publication_scale_length_spin = QDoubleSpinBox()
        self.publication_scale_length_spin.setRange(0.000001, 1_000_000_000)
        self.publication_scale_length_spin.setDecimals(4)
        self.publication_scale_length_spin.setValue(50.0)
        self.publication_scale_fraction_spin = QDoubleSpinBox()
        self.publication_scale_fraction_spin.setRange(5.0, 50.0)
        self.publication_scale_fraction_spin.setSuffix(" % of image width")
        self.publication_scale_fraction_spin.setValue(20.0)
        self.publication_scale_position_combo = QComboBox()
        for label, value in (
            ("Bottom right", "bottom_right"),
            ("Bottom left", "bottom_left"),
            ("Top right", "top_right"),
            ("Top left", "top_left"),
        ):
            self.publication_scale_position_combo.addItem(label, value)
        self.publication_scale_colour_edit = QLineEdit("#ffffff")
        self.publication_scale_box_colour_edit = QLineEdit("#000000a6")
        colour_row = QWidget()
        colour_layout = QHBoxLayout(colour_row)
        colour_layout.setContentsMargins(0, 0, 0, 0)
        self.publication_scale_colour_button = QPushButton("Bar/text colour…")
        self.publication_scale_box_colour_button = QPushButton("Box colour…")
        colour_layout.addWidget(self.publication_scale_colour_edit)
        colour_layout.addWidget(self.publication_scale_colour_button)
        colour_layout.addWidget(self.publication_scale_box_colour_edit)
        colour_layout.addWidget(self.publication_scale_box_colour_button)
        self.publication_scale_thickness_spin = QSpinBox()
        self.publication_scale_thickness_spin.setRange(1, 100)
        self.publication_scale_thickness_spin.setValue(5)
        self.publication_scale_font_spin = QSpinBox()
        self.publication_scale_font_spin.setRange(6, 300)
        self.publication_scale_font_spin.setValue(28)
        self.publication_scale_margin_spin = QSpinBox()
        self.publication_scale_margin_spin.setRange(0, 1000)
        self.publication_scale_margin_spin.setValue(30)
        self.publication_scale_box_padding_spin = QSpinBox()
        self.publication_scale_box_padding_spin.setRange(0, 500)
        self.publication_scale_box_padding_spin.setValue(12)
        self.publication_scale_show_label_check = QCheckBox("Show physical-length text")
        self.publication_scale_show_label_check.setChecked(True)
        self.publication_scale_label_scale_spin = publication_relative_scale_spin()
        self.publication_scale_thickness_scale_spin = publication_relative_scale_spin()
        self.publication_scale_margin_scale_spin = publication_relative_scale_spin()
        self.publication_scale_box_padding_scale_spin = (
            publication_relative_scale_spin()
        )
        self.publication_scale_ticks_check = QCheckBox("End ticks")
        self.publication_scale_ticks_check.setChecked(True)
        self.publication_scale_box_check = QCheckBox("Translucent background box")
        self.publication_scale_box_check.setChecked(True)
        calibration_form.addRow(
            "Calibration", self.publication_calibration_confirmed_check
        )
        calibration_form.addRow(
            "Physical size per pixel — X", self.publication_pixel_x_spin
        )
        calibration_form.addRow(
            "Physical size per pixel — Y", self.publication_pixel_y_spin
        )
        calibration_form.addRow("Unit", self.publication_unit_edit)
        calibration_form.addRow("", self.publication_detect_calibration_button)
        calibration_form.addRow("Scale bar", self.publication_scale_visible_check)
        calibration_form.addRow("Length", self.publication_scale_mode_combo)
        calibration_form.addRow("Fixed length", self.publication_scale_length_spin)
        calibration_form.addRow(
            "Automatic target", self.publication_scale_fraction_spin
        )
        calibration_form.addRow("Position", self.publication_scale_position_combo)
        calibration_form.addRow("Colours", colour_row)
        scale_sizing_help = QLabel(
            "These percentages multiply the automatic size chosen for the final "
            "resolution. They also apply to older custom-resolution presets."
        )
        scale_sizing_help.setWordWrap(True)
        calibration_form.addRow("Relative sizing", scale_sizing_help)
        calibration_form.addRow("Length text", self.publication_scale_show_label_check)
        calibration_form.addRow(
            "Length-text size", self.publication_scale_label_scale_spin
        )
        calibration_form.addRow(
            "Bar/tick thickness", self.publication_scale_thickness_scale_spin
        )
        calibration_form.addRow(
            "Outer margin", self.publication_scale_margin_scale_spin
        )
        calibration_form.addRow(
            "Box/label padding", self.publication_scale_box_padding_scale_spin
        )
        calibration_form.addRow("Style", self.publication_scale_ticks_check)
        calibration_form.addRow("", self.publication_scale_box_check)
        appearance_layout.addWidget(calibration_group)

        annotation_group = QGroupBox("4. Optional image annotations")
        annotation_form = QFormLayout(annotation_group)
        annotation_form.addRow("", publication_help_button())
        self.publication_show_roi_check = QCheckBox("Include ROI name")
        self.publication_show_channels_check = QCheckBox("Include channel names")
        self.publication_colour_channels_check = QCheckBox(
            "Match each channel name to its image colour"
        )
        self.publication_title_edit = QLineEdit()
        self.publication_annotation_position_combo = QComboBox()
        for label, value in (
            ("Top left", "top_left"),
            ("Top right", "top_right"),
            ("Bottom left", "bottom_left"),
            ("Bottom right", "bottom_right"),
        ):
            self.publication_annotation_position_combo.addItem(label, value)
        self.publication_annotation_font_spin = QSpinBox()
        self.publication_annotation_font_spin.setRange(6, 300)
        self.publication_annotation_font_spin.setValue(28)
        self.publication_annotation_margin_spin = QSpinBox()
        self.publication_annotation_margin_spin.setRange(0, 1000)
        self.publication_annotation_margin_spin.setValue(30)
        self.publication_annotation_box_padding_spin = QSpinBox()
        self.publication_annotation_box_padding_spin.setRange(0, 500)
        self.publication_annotation_box_padding_spin.setValue(12)
        self.publication_title_scale_spin = publication_relative_scale_spin()
        self.publication_roi_scale_spin = publication_relative_scale_spin()
        self.publication_channel_scale_spin = publication_relative_scale_spin()
        self.publication_annotation_margin_scale_spin = (
            publication_relative_scale_spin()
        )
        self.publication_annotation_box_padding_scale_spin = (
            publication_relative_scale_spin()
        )
        self.publication_annotation_colour_edit = QLineEdit("#ffffff")
        self.publication_annotation_box_colour_edit = QLineEdit("#000000a6")
        annotation_colour_row = QWidget()
        annotation_colour_layout = QHBoxLayout(annotation_colour_row)
        annotation_colour_layout.setContentsMargins(0, 0, 0, 0)
        self.publication_annotation_colour_button = QPushButton("Title/ROI colour…")
        self.publication_annotation_box_colour_button = QPushButton("Box colour…")
        annotation_colour_layout.addWidget(self.publication_annotation_colour_edit)
        annotation_colour_layout.addWidget(self.publication_annotation_colour_button)
        annotation_colour_layout.addWidget(self.publication_annotation_box_colour_edit)
        annotation_colour_layout.addWidget(
            self.publication_annotation_box_colour_button
        )
        self.publication_annotation_box_check = QCheckBox("Translucent background box")
        self.publication_annotation_box_check.setChecked(True)
        annotation_form.addRow("ROI", self.publication_show_roi_check)
        annotation_form.addRow("Channels", self.publication_show_channels_check)
        annotation_form.addRow(
            "Channel colours", self.publication_colour_channels_check
        )
        annotation_form.addRow("Custom title", self.publication_title_edit)
        annotation_form.addRow("Position", self.publication_annotation_position_combo)
        annotation_scale_help = QLabel(
            "Set each text type relative to the automatic size for the chosen "
            "resolution. Channel colours come from the frozen recipe, not the live ROI."
        )
        annotation_scale_help.setWordWrap(True)
        annotation_form.addRow("Relative sizing", annotation_scale_help)
        annotation_form.addRow("Title size", self.publication_title_scale_spin)
        annotation_form.addRow("ROI-name size", self.publication_roi_scale_spin)
        annotation_form.addRow("Channel-name size", self.publication_channel_scale_spin)
        annotation_form.addRow(
            "Outer margin", self.publication_annotation_margin_scale_spin
        )
        annotation_form.addRow(
            "Box padding", self.publication_annotation_box_padding_scale_spin
        )
        annotation_form.addRow("Colours", annotation_colour_row)
        annotation_form.addRow("Style", self.publication_annotation_box_check)
        appearance_layout.addWidget(annotation_group)
        appearance_layout.addStretch(1)
        appearance_scroll = QScrollArea()
        appearance_scroll.setWidgetResizable(True)
        appearance_scroll.setWidget(appearance_tab)
        tabs.addTab(appearance_scroll, "Scale bar & labels")

        # Output and batch tab.
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)
        output_group = QGroupBox("5. Resolution and files")
        output_form = QFormLayout(output_group)
        output_form.addRow("", publication_help_button())
        self.publication_width_spin = QSpinBox()
        self.publication_height_spin = QSpinBox()
        for spin in (self.publication_width_spin, self.publication_height_spin):
            spin.setRange(128, 30000)
        self.publication_width_spin.setValue(2400)
        self.publication_height_spin.setValue(1800)
        self.publication_size_mode_combo = QComboBox()
        self.publication_size_mode_combo.addItem(
            "Native source pixels — no resampling (recommended)", "native"
        )
        self.publication_size_mode_combo.addItem(
            "Custom output pixels — resample", "custom"
        )
        self.publication_size_mode_combo.setToolTip(
            "Native keeps one output pixel per source-image pixel. Custom changes "
            "sampling density while preserving the frozen field of view."
        )
        self.publication_supersampling_combo = QComboBox()
        self.publication_supersampling_combo.addItem("1× (fast)", 1)
        self.publication_supersampling_combo.addItem("2×", 2)
        self.publication_supersampling_combo.addItem("4×", 4)
        self.publication_resolution_combo = QComboBox()
        self.publication_resolution_combo.addItem(
            "Low — native pixels, 150 DPI (fast)", "low"
        )
        self.publication_resolution_combo.addItem(
            "Medium — 2× pixels, 300 DPI (recommended)", "medium"
        )
        self.publication_resolution_combo.addItem(
            "High — 4× pixels, 600 DPI (large figures)", "high"
        )
        self.publication_resolution_combo.setCurrentIndex(
            self.publication_resolution_combo.findData("medium")
        )
        self.publication_resolution_combo.setToolTip(
            "Resolution changes raster sampling and DPI together while preserving "
            "the field of view and automatically scaling all labels and scale-bar styling."
        )
        self.publication_format_combo = QComboBox()
        self.publication_format_combo.addItem("PNG — recommended", "png")
        self.publication_format_combo.addItem("TIFF — lossless", "tiff")
        self.publication_format_combo.addItem("JPEG — lossy", "jpeg")
        self.publication_dpi_spin = QSpinBox()
        self.publication_dpi_spin.setRange(30, 2400)
        self.publication_dpi_spin.setValue(300)
        self.publication_dpi_spin.setToolTip(
            "Print-resolution metadata only. DPI does not change the source-pixel "
            "field of view, image detail, or physical scale-bar length."
        )
        self.publication_filename_edit = QLineEdit(DEFAULT_FILENAME_TEMPLATE)
        self.publication_output_folder_edit = QLineEdit()
        output_folder_row = QWidget()
        output_folder_layout = QHBoxLayout(output_folder_row)
        output_folder_layout.setContentsMargins(0, 0, 0, 0)
        self.publication_choose_folder_button = QPushButton("Choose…")
        output_folder_layout.addWidget(self.publication_output_folder_edit, 1)
        output_folder_layout.addWidget(self.publication_choose_folder_button)
        self.publication_conflict_combo = QComboBox()
        self.publication_conflict_combo.addItem("Resume: skip exact matching outputs", "resume")
        self.publication_conflict_combo.addItem("Create a versioned filename", "version")
        self.publication_conflict_combo.addItem("Overwrite existing files", "overwrite")
        self.publication_filename_preview = QLabel()
        self.publication_filename_preview.setWordWrap(True)
        self.publication_print_size_label = QLabel()
        self.publication_print_size_label.setWordWrap(True)
        output_form.addRow("Resolution", self.publication_resolution_combo)
        output_form.addRow("Format", self.publication_format_combo)
        output_form.addRow("Filename template", self.publication_filename_edit)
        output_form.addRow("Output folder", output_folder_row)
        output_form.addRow("Existing files", self.publication_conflict_combo)
        output_form.addRow("Filename preview", self.publication_filename_preview)
        output_form.addRow("Estimated print size", self.publication_print_size_label)
        output_layout.addWidget(output_group)

        bulk_group = QGroupBox("6. Current ROI or reproducible bulk export")
        bulk_layout = QVBoxLayout(bulk_group)
        bulk_help_row = QHBoxLayout()
        bulk_help_row.addStretch(1)
        bulk_help_row.addWidget(publication_help_button())
        bulk_layout.addLayout(bulk_help_row)
        bulk_help = QLabel(
            "Select ROIs below. Bulk export uses one frozen recipe and processes "
            "ROIs sequentially so Napari/OpenGL rendering remains safe."
        )
        bulk_help.setWordWrap(True)
        self.publication_roi_list = QListWidget()
        self.publication_roi_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.publication_roi_list.setMaximumHeight(180)
        roi_actions = QHBoxLayout()
        self.publication_select_all_rois_button = QPushButton("Select all")
        self.publication_select_current_roi_button = QPushButton("Select current")
        self.publication_clear_rois_button = QPushButton("Clear selection")
        roi_actions.addWidget(self.publication_select_all_rois_button)
        roi_actions.addWidget(self.publication_select_current_roi_button)
        roi_actions.addWidget(self.publication_clear_rois_button)
        export_actions = QHBoxLayout()
        self.publication_render_preview_button = QPushButton("Render preview")
        self.publication_export_current_button = QPushButton("Export current ROI")
        self.publication_export_bulk_button = QPushButton("Preflight and bulk export…")
        self.publication_cancel_button = QPushButton("Cancel bulk export")
        self.publication_cancel_button.setEnabled(False)
        export_actions.addWidget(self.publication_render_preview_button)
        export_actions.addWidget(self.publication_export_current_button)
        export_actions.addWidget(self.publication_export_bulk_button)
        export_actions.addWidget(self.publication_cancel_button)
        self.publication_progress_bar = QProgressBar()
        self.publication_progress_bar.setRange(0, 1)
        self.publication_progress_label = QLabel("No export is running.")
        self.publication_progress_label.setWordWrap(True)
        bulk_layout.addWidget(bulk_help)
        bulk_layout.addWidget(self.publication_roi_list)
        bulk_layout.addLayout(roi_actions)
        bulk_layout.addLayout(export_actions)
        bulk_layout.addWidget(self.publication_progress_bar)
        bulk_layout.addWidget(self.publication_progress_label)
        output_layout.addWidget(bulk_group)
        output_layout.addStretch(1)
        tabs.addTab(output_tab, "Output & bulk")

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.hide)
        close_row.addWidget(close_button)
        layout.addLayout(close_row)

        self.publication_export_dialog = dialog

        # Dialog-local signals deliberately do not touch ROI loading unless an
        # explicit preview/export button is pressed.
        self.publication_preset_combo.currentIndexChanged.connect(
            self._load_selected_publication_preset
        )
        self.publication_save_preset_button.clicked.connect(
            self._guard(self.save_new_publication_preset)
        )
        self.publication_update_preset_button.clicked.connect(
            self._guard(self.update_selected_publication_preset)
        )
        self.publication_delete_preset_button.clicked.connect(
            self._guard(self.delete_selected_publication_preset)
        )
        self.publication_capture_view_button.clicked.connect(
            self._guard(self.capture_publication_view)
        )
        self.publication_use_rectangle_button.clicked.connect(
            self._guard(self.capture_publication_rectangle)
        )
        self.publication_preview_frame_button.clicked.connect(
            self._guard(self.preview_publication_frame)
        )
        self.publication_scale_colour_button.clicked.connect(
            lambda: self._pick_publication_colour(self.publication_scale_colour_edit)
        )
        self.publication_detect_calibration_button.clicked.connect(
            self._guard(self.detect_publication_calibration)
        )
        self.publication_scale_box_colour_button.clicked.connect(
            lambda: self._pick_publication_colour(
                self.publication_scale_box_colour_edit,
                allow_alpha=True,
            )
        )
        self.publication_annotation_colour_button.clicked.connect(
            lambda: self._pick_publication_colour(
                self.publication_annotation_colour_edit
            )
        )
        self.publication_annotation_box_colour_button.clicked.connect(
            lambda: self._pick_publication_colour(
                self.publication_annotation_box_colour_edit,
                allow_alpha=True,
            )
        )
        self.publication_choose_folder_button.clicked.connect(
            self._guard(self.choose_publication_output_folder)
        )
        self.publication_select_all_rois_button.clicked.connect(
            self.publication_roi_list.selectAll
        )
        self.publication_select_current_roi_button.clicked.connect(
            self._select_current_publication_roi
        )
        self.publication_clear_rois_button.clicked.connect(
            self.publication_roi_list.clearSelection
        )
        self.publication_export_current_button.clicked.connect(
            self._guard(self.export_current_publication_image)
        )
        self.publication_render_preview_button.clicked.connect(
            self._guard(self.render_publication_preview)
        )
        self.publication_export_bulk_button.clicked.connect(
            self.start_publication_bulk_export
        )
        self.publication_cancel_button.clicked.connect(
            self.cancel_publication_bulk_export
        )
        for control in (
            self.publication_recipe_combo,
            self.publication_frame_mode_combo,
            self.publication_aspect_combo,
            self.publication_scale_mode_combo,
            self.publication_scale_position_combo,
            self.publication_resolution_combo,
            self.publication_size_mode_combo,
            self.publication_supersampling_combo,
            self.publication_format_combo,
            self.publication_annotation_position_combo,
        ):
            control.currentIndexChanged.connect(
                self._publication_export_controls_changed
            )
        for control in (
            self.publication_center_y_spin,
            self.publication_center_x_spin,
            self.publication_field_height_spin,
            self.publication_field_width_spin,
            self.publication_pixel_x_spin,
            self.publication_pixel_y_spin,
            self.publication_scale_length_spin,
            self.publication_scale_fraction_spin,
            self.publication_scale_thickness_spin,
            self.publication_scale_font_spin,
            self.publication_scale_margin_spin,
            self.publication_scale_box_padding_spin,
            self.publication_annotation_font_spin,
            self.publication_annotation_margin_spin,
            self.publication_annotation_box_padding_spin,
            self.publication_scale_label_scale_spin,
            self.publication_scale_thickness_scale_spin,
            self.publication_scale_margin_scale_spin,
            self.publication_scale_box_padding_scale_spin,
            self.publication_title_scale_spin,
            self.publication_roi_scale_spin,
            self.publication_channel_scale_spin,
            self.publication_annotation_margin_scale_spin,
            self.publication_annotation_box_padding_scale_spin,
            self.publication_width_spin,
            self.publication_height_spin,
            self.publication_dpi_spin,
        ):
            control.valueChanged.connect(self._publication_export_controls_changed)
        for control in (
            self.publication_preset_name_edit,
            self.publication_unit_edit,
            self.publication_scale_colour_edit,
            self.publication_scale_box_colour_edit,
            self.publication_annotation_colour_edit,
            self.publication_annotation_box_colour_edit,
            self.publication_title_edit,
            self.publication_filename_edit,
            self.publication_output_folder_edit,
        ):
            control.textChanged.connect(self._publication_export_controls_changed)
        for control in (
            self.publication_calibration_confirmed_check,
            self.publication_scale_visible_check,
            self.publication_scale_show_label_check,
            self.publication_scale_ticks_check,
            self.publication_scale_box_check,
            self.publication_show_roi_check,
            self.publication_show_channels_check,
            self.publication_colour_channels_check,
            self.publication_annotation_box_check,
        ):
            control.toggled.connect(self._publication_export_controls_changed)

        self._capture_current_recipe_display_state()
        self._refresh_publication_export_controls()
        if (
            self.publication_preset_combo.currentData()
            not in self.publication_export_state.presets
        ):
            initial_frame = self._current_publication_camera_frame()
            self.publication_center_y_spin.setValue(initial_frame.center_y)
            self.publication_center_x_spin.setValue(initial_frame.center_x)
            self.publication_field_height_spin.setValue(initial_frame.field_height)
            self.publication_field_width_spin.setValue(initial_frame.field_width)
            self.publication_frame_mode_combo.setCurrentIndex(
                self.publication_frame_mode_combo.findData("current_view")
            )
        self._publication_export_controls_changed()
        dialog.show()

    def _publication_canvas_size(self) -> tuple[float, float]:
        canvas = getattr(
            getattr(getattr(self.viewer, "window", None), "_qt_viewer", None),
            "canvas",
            None,
        )
        size = getattr(canvas, "size", None)
        if size is None or len(size) != 2:
            raise ValueError("Napari canvas dimensions are not available.")
        return float(size[0]), float(size[1])

    def _current_publication_camera_frame(self) -> ResolvedPublicationFrame:
        width, height = self._publication_canvas_size()
        return camera_frame_from_canvas(
            center=tuple(self.viewer.camera.center),
            zoom=float(self.viewer.camera.zoom),
            canvas_width=width,
            canvas_height=height,
        )

    def capture_publication_view(self) -> None:
        frame = self._current_publication_camera_frame()
        self.publication_center_y_spin.setValue(frame.center_y)
        self.publication_center_x_spin.setValue(frame.center_x)
        self.publication_field_height_spin.setValue(frame.field_height)
        self.publication_field_width_spin.setValue(frame.field_width)
        self.publication_frame_mode_combo.setCurrentIndex(
            self.publication_frame_mode_combo.findData("fixed")
        )
        self._publication_export_controls_changed()
        self.set_status(
            "Captured the current Napari centre and field of view as a fixed "
            "publication frame."
        )

    def capture_publication_rectangle(self) -> None:
        layer = self.viewer.layers.selection.active
        if layer is None or layer.__class__.__name__.lower() != "shapes":
            raise ValueError(
                "Select a Shapes layer containing the desired rectangular crop."
            )
        selected = list(getattr(layer, "selected_data", []))
        if len(selected) != 1:
            raise ValueError("Select exactly one rectangle in the Shapes layer.")
        points = np.asarray(layer.data[selected[0]], dtype=float)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            raise ValueError("The selected shape has no two-dimensional extent.")
        y_values = points[:, -2]
        x_values = points[:, -1]
        height = float(np.max(y_values) - np.min(y_values))
        width = float(np.max(x_values) - np.min(x_values))
        if height <= 0 or width <= 0:
            raise ValueError("The selected shape has zero width or height.")
        self.publication_center_y_spin.setValue(float(np.mean([np.min(y_values), np.max(y_values)])))
        self.publication_center_x_spin.setValue(float(np.mean([np.min(x_values), np.max(x_values)])))
        self.publication_field_height_spin.setValue(height)
        self.publication_field_width_spin.setValue(width)
        self.publication_frame_mode_combo.setCurrentIndex(
            self.publication_frame_mode_combo.findData("fixed")
        )
        self._publication_export_controls_changed()
        self.set_status("Captured the selected Shapes rectangle as the export frame.")

    def _pick_publication_colour(self, target, *, allow_alpha: bool = False) -> None:
        current_text = target.text().strip() or "#ffffff"
        if allow_alpha and len(current_text) == 9 and current_text.startswith("#"):
            current = self.QColor(
                int(current_text[1:3], 16),
                int(current_text[3:5], 16),
                int(current_text[5:7], 16),
                int(current_text[7:9], 16),
            )
        else:
            current = self.QColor(current_text)
        options = (
            self.QColorDialog.ShowAlphaChannel
            if allow_alpha
            else self.QColorDialog.ColorDialogOptions()
        )
        colour = self.QColorDialog.getColor(current, self.root, "Choose export colour", options)
        if not colour.isValid():
            return
        if allow_alpha:
            target.setText(
                f"#{colour.red():02x}{colour.green():02x}{colour.blue():02x}"
                f"{colour.alpha():02x}"
            )
        else:
            target.setText(colour.name(self.QColor.HexRgb))

    def choose_publication_output_folder(self) -> None:
        initial = self.publication_output_folder_edit.text().strip()
        if not initial:
            initial = str(self.paths.exports / "publication_images")
        selected = self.QFileDialog.getExistingDirectory(
            self.root,
            "Choose publication image output folder",
            initial,
        )
        if selected:
            self.publication_output_folder_edit.setText(selected)

    def detect_publication_calibration(self) -> None:
        recipe, _recipe_id, _recipe_name = self._publication_recipe_snapshot(
            capture_live=False
        )
        candidates = [
            self.current_image_paths[channel]
            for channel in recipe.image_channels
            if channel in self.current_image_paths
            and self.current_image_paths[channel].suffix.lower() in {".tif", ".tiff"}
        ]
        if not candidates:
            candidates = [
                path
                for path in self.current_image_paths.values()
                if path.suffix.lower() in {".tif", ".tiff"}
            ]
        if not candidates:
            raise ValueError(
                "No current ROI TIFF is available. Load at least one image channel first."
            )
        detected = detect_tiff_pixel_calibration(candidates[0])
        if detected is None:
            raise ValueError(
                "The current TIFF has no supported OME PhysicalSize or calibrated "
                "TIFF resolution tags. Enter the pixel size manually."
            )
        calibration, source = detected
        self.publication_pixel_x_spin.setValue(calibration.x_size)
        self.publication_pixel_y_spin.setValue(calibration.y_size)
        self.publication_unit_edit.setText(calibration.unit)
        self.publication_calibration_confirmed_check.setChecked(False)
        self.set_status(
            f"Detected {calibration.x_size:g} × {calibration.y_size:g} "
            f"{calibration.unit}/pixel from {source} in {candidates[0].name}. "
            "Review the values, then tick the verification box."
        )

    def _select_current_publication_roi(self) -> None:
        self.publication_roi_list.clearSelection()
        matches = self.publication_roi_list.findItems(
            str(self.current_roi or ""), self.Qt.MatchExactly
        )
        if matches:
            matches[0].setSelected(True)
            self.publication_roi_list.scrollToItem(matches[0])

    def _refresh_publication_export_controls(self) -> None:
        if self.publication_export_dialog is None:
            return
        selected_recipe = self.publication_recipe_combo.currentData()
        self.publication_recipe_combo.blockSignals(True)
        self.publication_recipe_combo.clear()
        self.publication_recipe_combo.addItem("Current live Explore view", None)
        for recipe in sorted(
            self.explore_review_state.recipe_presets.values(),
            key=lambda item: item.name.casefold(),
        ):
            self.publication_recipe_combo.addItem(recipe.name, recipe.preset_id)
        recipe_index = self.publication_recipe_combo.findData(selected_recipe)
        self.publication_recipe_combo.setCurrentIndex(max(0, recipe_index))
        self.publication_recipe_combo.blockSignals(False)

        selected_preset = self.publication_preset_combo.currentData()
        self.publication_preset_combo.blockSignals(True)
        self.publication_preset_combo.clear()
        self.publication_preset_combo.addItem("Unsaved publication settings", None)
        for preset in sorted(
            self.publication_export_state.presets.values(),
            key=lambda item: item.name.casefold(),
        ):
            self.publication_preset_combo.addItem(preset.name, preset.preset_id)
        if selected_preset is None:
            selected_preset = self.publication_export_state.active_preset_id
        preset_index = self.publication_preset_combo.findData(selected_preset)
        self.publication_preset_combo.setCurrentIndex(max(0, preset_index))
        self.publication_preset_combo.blockSignals(False)

        selected_rois = {
            item.text() for item in self.publication_roi_list.selectedItems()
        }
        self.publication_roi_list.clear()
        self.publication_roi_list.addItems(
            [self.roi_combo.itemText(index) for index in range(self.roi_combo.count())]
        )
        for index in range(self.publication_roi_list.count()):
            item = self.publication_roi_list.item(index)
            item.setSelected(item.text() in selected_rois)
        if not selected_rois:
            self._select_current_publication_roi()
        if not self.publication_output_folder_edit.text().strip():
            self.publication_output_folder_edit.setText(
                str(self.paths.exports / "publication_images")
            )
        if selected_preset in self.publication_export_state.presets:
            self._load_selected_publication_preset()
        self._publication_export_controls_changed()

    def _publication_export_controls_changed(self, *_args) -> None:
        if self.publication_export_dialog is None:
            return
        fixed = self.publication_frame_mode_combo.currentData() == "fixed"
        custom_resolution = self.publication_resolution_combo.currentData() == "custom"
        custom_size = (
            custom_resolution
            and self.publication_size_mode_combo.currentData() == "custom"
        )
        for control in (
            self.publication_center_y_spin,
            self.publication_center_x_spin,
            self.publication_field_height_spin,
            self.publication_field_width_spin,
        ):
            control.setEnabled(fixed)
        self.publication_aspect_combo.setEnabled(custom_size)
        self.publication_width_spin.setEnabled(custom_size)
        self.publication_height_spin.setEnabled(custom_size)
        scale_visible = self.publication_scale_visible_check.isChecked()
        scale_fixed = self.publication_scale_mode_combo.currentData() == "fixed"
        self.publication_scale_length_spin.setEnabled(scale_visible and scale_fixed)
        self.publication_scale_fraction_spin.setEnabled(
            scale_visible and not scale_fixed
        )
        self.publication_scale_label_scale_spin.setEnabled(
            scale_visible and self.publication_scale_show_label_check.isChecked()
        )
        for control in (
            self.publication_scale_thickness_scale_spin,
            self.publication_scale_margin_scale_spin,
            self.publication_scale_box_padding_scale_spin,
            self.publication_scale_ticks_check,
            self.publication_scale_box_check,
        ):
            control.setEnabled(scale_visible)
        self.publication_colour_channels_check.setEnabled(
            self.publication_show_channels_check.isChecked()
        )
        self.publication_update_preset_button.setEnabled(
            self.publication_preset_combo.currentData()
            in self.publication_export_state.presets
        )
        self.publication_delete_preset_button.setEnabled(
            self.publication_preset_combo.currentData()
            in self.publication_export_state.presets
        )
        try:
            preset = self._publication_preset_from_controls(
                preset_id="preview",
                capture_live=False,
                freeze_current_frame=False,
            )
            current = self._current_publication_camera_frame()
            frame = resolve_publication_frame(
                preset.frame,
                output=preset.output,
                current_frame=current,
                roi_shape=tuple(self.current_mask.shape[:2]),
            )
            output_width, output_height = resolve_publication_output_size(
                preset.output, frame
            )
            filename = build_publication_filename(
                preset,
                roi=str(self.current_roi or "ROI"),
                output_size=(output_width, output_height),
            )
            self.publication_filename_preview.setText(filename)
            dpi = max(1, resolve_publication_dpi(preset.output))
            width_inches = output_width / dpi
            height_inches = output_height / dpi
            resolution_labels = {
                "low": "Low — 1× source pixels",
                "medium": "Medium — 2× source pixels",
                "high": "High — 4× source pixels",
                "custom": "Existing custom settings",
            }
            size_description = resolution_labels[preset.output.resolution]
            annotation_scale = publication_resolution_scale(preset.output)
            self.publication_print_size_label.setText(
                f"{output_width:,} × {output_height:,} pixels ({size_description}); "
                f"{width_inches:.2f} × {height_inches:.2f} inches at {dpi} DPI. "
                f"Automatic annotation baseline: {annotation_scale:g}×; "
                "the relative percentages are applied afterwards."
            )
            self.publication_frame_summary.setText(
                f"centre Y/X {frame.center_y:.2f}, {frame.center_x:.2f}; "
                f"field {frame.field_height:.2f} × {frame.field_width:.2f} source pixels; "
                f"output {output_width:,} × {output_height:,} pixels."
            )
            error = ""
            if preset.scale_bar.visible and not preset.calibration.confirmed:
                error = " Scale bar disabled until calibration is verified."
            self.publication_filename_preview.setToolTip(error)
        except Exception as error:  # noqa: BLE001 - live validation boundary
            self.publication_filename_preview.setText(f"⚠ {error}")
            self.publication_frame_summary.setText(f"⚠ {error}")
            self.publication_print_size_label.setText(f"⚠ {error}")

    def _publication_recipe_snapshot(
        self, *, capture_live: bool
    ) -> tuple[ExploreViewRecipe, str | None, str]:
        recipe_id = self.publication_recipe_combo.currentData()
        if isinstance(recipe_id, str) and recipe_id.startswith(
            "publication_frozen::"
        ):
            publication_id = recipe_id.split("::", 1)[1]
            publication = self.publication_export_state.presets.get(publication_id)
            if publication is not None:
                return (
                    publication.recipe.model_copy(deep=True),
                    publication.source_recipe_id,
                    publication.source_recipe_name,
                )
        saved = self.explore_review_state.recipe_presets.get(recipe_id)
        if saved is not None:
            return saved.recipe.model_copy(deep=True), saved.preset_id, saved.name
        if capture_live:
            self._capture_current_recipe_display_state()
        active = self.explore_review_state.recipe_presets.get(
            self.explore_review_state.active_recipe_id
        )
        live_name = "Current Explore view"
        live_id = None
        if active is not None and active.recipe.fingerprint == self.explore_recipe.fingerprint:
            live_name = active.name
            live_id = active.preset_id
        return self.explore_recipe.model_copy(deep=True), live_id, live_name

    def _publication_preset_from_controls(
        self,
        *,
        preset_id: str,
        capture_live: bool,
        freeze_current_frame: bool,
    ) -> PublicationExportPreset:
        recipe, recipe_id, recipe_name = self._publication_recipe_snapshot(
            capture_live=capture_live
        )
        if not recipe.has_content:
            raise ValueError(
                "Build or select an Explore recipe with at least one visible source first."
            )
        frame_mode = str(self.publication_frame_mode_combo.currentData())
        frame = PublicationFrame(
            mode=frame_mode,
            center_y=self.publication_center_y_spin.value() if frame_mode == "fixed" else None,
            center_x=self.publication_center_x_spin.value() if frame_mode == "fixed" else None,
            field_height=self.publication_field_height_spin.value() if frame_mode == "fixed" else None,
            field_width=self.publication_field_width_spin.value() if frame_mode == "fixed" else None,
            aspect_mode=str(self.publication_aspect_combo.currentData()),
        )
        if freeze_current_frame and frame.mode == "current_view":
            current = self._current_publication_camera_frame()
            frame = PublicationFrame(
                mode="fixed",
                center_y=current.center_y,
                center_x=current.center_x,
                field_height=current.field_height,
                field_width=current.field_width,
                aspect_mode=frame.aspect_mode,
            )
        resolution = str(self.publication_resolution_combo.currentData())
        simple_resolution = resolution in {"low", "medium", "high"}
        output = PublicationOutput(
            resolution=resolution,
            size_mode=(
                "native"
                if simple_resolution
                else str(self.publication_size_mode_combo.currentData())
            ),
            width=self.publication_width_spin.value(),
            height=self.publication_height_spin.value(),
            supersampling=(
                1
                if simple_resolution
                else int(self.publication_supersampling_combo.currentData())
            ),
            format=str(self.publication_format_combo.currentData()),
            dpi=self.publication_dpi_spin.value(),
            filename_template=self.publication_filename_edit.text(),
        )
        if simple_resolution:
            output = output.model_copy(
                update={"dpi": resolve_publication_dpi(output)}
            )
        scale_bar = PublicationScaleBar(
            visible=self.publication_scale_visible_check.isChecked(),
            length_mode=str(self.publication_scale_mode_combo.currentData()),
            length=self.publication_scale_length_spin.value(),
            target_fraction=self.publication_scale_fraction_spin.value() / 100.0,
            position=str(self.publication_scale_position_combo.currentData()),
            color=self.publication_scale_colour_edit.text(),
            thickness=self.publication_scale_thickness_spin.value(),
            font_size=self.publication_scale_font_spin.value(),
            margin=self.publication_scale_margin_spin.value(),
            show_label=self.publication_scale_show_label_check.isChecked(),
            label_scale=self.publication_scale_label_scale_spin.value() / 100.0,
            thickness_scale=(
                self.publication_scale_thickness_scale_spin.value() / 100.0
            ),
            margin_scale=self.publication_scale_margin_scale_spin.value() / 100.0,
            ticks=self.publication_scale_ticks_check.isChecked(),
            box=self.publication_scale_box_check.isChecked(),
            box_color=self.publication_scale_box_colour_edit.text(),
            box_padding=self.publication_scale_box_padding_spin.value(),
            box_padding_scale=(
                self.publication_scale_box_padding_scale_spin.value() / 100.0
            ),
        )
        annotations = PublicationAnnotations(
            show_roi=self.publication_show_roi_check.isChecked(),
            show_channels=self.publication_show_channels_check.isChecked(),
            custom_title=self.publication_title_edit.text(),
            position=str(self.publication_annotation_position_combo.currentData()),
            color=self.publication_annotation_colour_edit.text(),
            font_size=self.publication_annotation_font_spin.value(),
            margin=self.publication_annotation_margin_spin.value(),
            title_scale=self.publication_title_scale_spin.value() / 100.0,
            roi_scale=self.publication_roi_scale_spin.value() / 100.0,
            channel_scale=self.publication_channel_scale_spin.value() / 100.0,
            margin_scale=(
                self.publication_annotation_margin_scale_spin.value() / 100.0
            ),
            color_channels=self.publication_colour_channels_check.isChecked(),
            box=self.publication_annotation_box_check.isChecked(),
            box_color=self.publication_annotation_box_colour_edit.text(),
            box_padding=self.publication_annotation_box_padding_spin.value(),
            box_padding_scale=(
                self.publication_annotation_box_padding_scale_spin.value() / 100.0
            ),
        )
        return PublicationExportPreset(
            preset_id=preset_id,
            name=self.publication_preset_name_edit.text().strip()
            or "Publication export",
            source_recipe_id=recipe_id,
            source_recipe_name=recipe_name,
            recipe=recipe,
            frame=frame,
            calibration=PixelCalibration(
                confirmed=self.publication_calibration_confirmed_check.isChecked(),
                x_size=self.publication_pixel_x_spin.value(),
                y_size=self.publication_pixel_y_spin.value(),
                unit=self.publication_unit_edit.text(),
            ),
            scale_bar=scale_bar,
            annotations=annotations,
            output=output,
        )

    def _load_selected_publication_preset(self, *_args) -> None:
        preset_id = self.publication_preset_combo.currentData()
        preset = self.publication_export_state.presets.get(preset_id)
        if preset is None:
            self.publication_export_state.active_preset_id = None
            self._publication_export_controls_changed()
            return
        self.publication_export_state.active_preset_id = preset.preset_id
        self._save_publication_export_state()
        self.publication_preset_name_edit.setText(preset.name)
        frozen_recipe_id = f"publication_frozen::{preset.preset_id}"
        existing_frozen = self.publication_recipe_combo.findData(frozen_recipe_id)
        if existing_frozen < 0:
            self.publication_recipe_combo.insertItem(
                0,
                f"Frozen snapshot — {preset.source_recipe_name}",
                frozen_recipe_id,
            )
            existing_frozen = 0
        self.publication_recipe_combo.setCurrentIndex(existing_frozen)
        self.publication_frame_mode_combo.setCurrentIndex(
            self.publication_frame_mode_combo.findData(preset.frame.mode)
        )
        self.publication_aspect_combo.setCurrentIndex(
            self.publication_aspect_combo.findData(preset.frame.aspect_mode)
        )
        for control, value in (
            (self.publication_center_y_spin, preset.frame.center_y),
            (self.publication_center_x_spin, preset.frame.center_x),
            (self.publication_field_height_spin, preset.frame.field_height),
            (self.publication_field_width_spin, preset.frame.field_width),
        ):
            if value is not None:
                control.setValue(float(value))
        self.publication_calibration_confirmed_check.setChecked(preset.calibration.confirmed)
        self.publication_pixel_x_spin.setValue(preset.calibration.x_size)
        self.publication_pixel_y_spin.setValue(preset.calibration.y_size)
        self.publication_unit_edit.setText(preset.calibration.unit)
        self.publication_scale_visible_check.setChecked(preset.scale_bar.visible)
        self.publication_scale_mode_combo.setCurrentIndex(
            self.publication_scale_mode_combo.findData(preset.scale_bar.length_mode)
        )
        self.publication_scale_length_spin.setValue(preset.scale_bar.length)
        self.publication_scale_fraction_spin.setValue(preset.scale_bar.target_fraction * 100)
        self.publication_scale_position_combo.setCurrentIndex(
            self.publication_scale_position_combo.findData(preset.scale_bar.position)
        )
        self.publication_scale_colour_edit.setText(preset.scale_bar.color)
        self.publication_scale_box_colour_edit.setText(preset.scale_bar.box_color)
        self.publication_scale_thickness_spin.setValue(preset.scale_bar.thickness)
        self.publication_scale_font_spin.setValue(preset.scale_bar.font_size)
        self.publication_scale_margin_spin.setValue(preset.scale_bar.margin)
        self.publication_scale_box_padding_spin.setValue(preset.scale_bar.box_padding)
        self.publication_scale_show_label_check.setChecked(preset.scale_bar.show_label)
        self.publication_scale_label_scale_spin.setValue(
            preset.scale_bar.label_scale * 100
        )
        self.publication_scale_thickness_scale_spin.setValue(
            preset.scale_bar.thickness_scale * 100
        )
        self.publication_scale_margin_scale_spin.setValue(
            preset.scale_bar.margin_scale * 100
        )
        self.publication_scale_box_padding_scale_spin.setValue(
            preset.scale_bar.box_padding_scale * 100
        )
        self.publication_scale_ticks_check.setChecked(preset.scale_bar.ticks)
        self.publication_scale_box_check.setChecked(preset.scale_bar.box)
        self.publication_show_roi_check.setChecked(preset.annotations.show_roi)
        self.publication_show_channels_check.setChecked(
            preset.annotations.show_channels
        )
        self.publication_title_edit.setText(preset.annotations.custom_title)
        self.publication_annotation_position_combo.setCurrentIndex(
            self.publication_annotation_position_combo.findData(
                preset.annotations.position
            )
        )
        self.publication_annotation_font_spin.setValue(preset.annotations.font_size)
        self.publication_annotation_margin_spin.setValue(preset.annotations.margin)
        self.publication_annotation_box_padding_spin.setValue(
            preset.annotations.box_padding
        )
        self.publication_title_scale_spin.setValue(preset.annotations.title_scale * 100)
        self.publication_roi_scale_spin.setValue(preset.annotations.roi_scale * 100)
        self.publication_channel_scale_spin.setValue(
            preset.annotations.channel_scale * 100
        )
        self.publication_annotation_margin_scale_spin.setValue(
            preset.annotations.margin_scale * 100
        )
        self.publication_colour_channels_check.setChecked(
            preset.annotations.color_channels
        )
        self.publication_annotation_colour_edit.setText(preset.annotations.color)
        self.publication_annotation_box_check.setChecked(preset.annotations.box)
        self.publication_annotation_box_colour_edit.setText(
            preset.annotations.box_color
        )
        self.publication_annotation_box_padding_scale_spin.setValue(
            preset.annotations.box_padding_scale * 100
        )
        resolution_index = self.publication_resolution_combo.findData(
            preset.output.resolution
        )
        if resolution_index < 0 and preset.output.resolution == "custom":
            self.publication_resolution_combo.addItem(
                "Custom — existing saved preset", "custom"
            )
            resolution_index = self.publication_resolution_combo.findData("custom")
        self.publication_resolution_combo.setCurrentIndex(resolution_index)
        self.publication_size_mode_combo.setCurrentIndex(
            self.publication_size_mode_combo.findData(preset.output.size_mode)
        )
        self.publication_width_spin.setValue(preset.output.width)
        self.publication_height_spin.setValue(preset.output.height)
        self.publication_supersampling_combo.setCurrentIndex(
            self.publication_supersampling_combo.findData(preset.output.supersampling)
        )
        self.publication_format_combo.setCurrentIndex(
            self.publication_format_combo.findData(preset.output.format)
        )
        self.publication_dpi_spin.setValue(preset.output.dpi)
        self.publication_filename_edit.setText(preset.output.filename_template)
        self._publication_export_controls_changed()

    def save_new_publication_preset(self) -> None:
        name = self.publication_preset_name_edit.text().strip()
        if not name:
            raise ValueError("Enter a descriptive publication preset name.")
        if any(
            item.name.casefold() == name.casefold()
            for item in self.publication_export_state.presets.values()
        ):
            raise ValueError(f"A publication preset named {name!r} already exists.")
        preset = self._publication_preset_from_controls(
            preset_id=str(uuid4()),
            capture_live=True,
            freeze_current_frame=True,
        )
        self.publication_export_state.presets[preset.preset_id] = preset
        self.publication_export_state.active_preset_id = preset.preset_id
        self._save_publication_export_state()
        self._refresh_publication_export_controls()
        index = self.publication_preset_combo.findData(preset.preset_id)
        self.publication_preset_combo.setCurrentIndex(index)
        append_audit(
            self.paths,
            {
                "action": "save_publication_export_preset",
                "preset_id": preset.preset_id,
                "name": preset.name,
                "fingerprint": preset.fingerprint,
            },
        )
        self.set_status(f"Saved publication export preset {preset.name!r}.")

    def update_selected_publication_preset(self) -> None:
        preset_id = self.publication_preset_combo.currentData()
        existing = self.publication_export_state.presets.get(preset_id)
        if existing is None:
            raise ValueError("Select a saved publication preset to update.")
        name = self.publication_preset_name_edit.text().strip()
        if not name:
            raise ValueError("Enter a descriptive publication preset name.")
        if any(
            item.preset_id != existing.preset_id
            and item.name.casefold() == name.casefold()
            for item in self.publication_export_state.presets.values()
        ):
            raise ValueError(f"A publication preset named {name!r} already exists.")
        preset = self._publication_preset_from_controls(
            preset_id=existing.preset_id,
            capture_live=True,
            freeze_current_frame=True,
        )
        self.publication_export_state.presets[preset.preset_id] = preset
        self.publication_export_state.active_preset_id = preset.preset_id
        self._save_publication_export_state()
        self._refresh_publication_export_controls()
        self.set_status(f"Updated publication export preset {preset.name!r}.")

    def delete_selected_publication_preset(self) -> None:
        preset_id = self.publication_preset_combo.currentData()
        preset = self.publication_export_state.presets.get(preset_id)
        if preset is None:
            raise ValueError("Select a saved publication preset to delete.")
        reply = self.QMessageBox.question(
            self.root,
            "Delete publication preset",
            f"Delete publication preset {preset.name!r}? Existing exported images "
            "and their provenance files will not be removed.",
        )
        if reply != self.QMessageBox.Yes:
            return
        del self.publication_export_state.presets[preset.preset_id]
        if self.publication_export_state.active_preset_id == preset.preset_id:
            self.publication_export_state.active_preset_id = None
        self._save_publication_export_state()
        self._refresh_publication_export_controls()
        self.set_status(f"Deleted publication export preset {preset.name!r}.")

    def _resolved_publication_frame(
        self, preset: PublicationExportPreset
    ) -> ResolvedPublicationFrame:
        if self.current_mask is None:
            raise ValueError("Load an ROI before resolving its publication frame.")
        current = self._current_publication_camera_frame()
        return resolve_publication_frame(
            preset.frame,
            output=preset.output,
            current_frame=current,
            roi_shape=tuple(self.current_mask.shape[:2]),
        )

    def preview_publication_frame(self) -> None:
        preset = self._publication_preset_from_controls(
            preset_id="preview",
            capture_live=False,
            freeze_current_frame=False,
        )
        frame = self._resolved_publication_frame(preset)
        width, height = self._publication_canvas_size()
        centre = list(self.viewer.camera.center)
        centre[-2:] = [frame.center_y, frame.center_x]
        self.viewer.camera.center = tuple(centre)
        self.viewer.camera.zoom = min(
            width / frame.field_width,
            height / frame.field_height,
        )
        self.set_status(
            "Previewed the publication centre and field of view in Napari. "
            "Use Render preview to inspect the exact requested pixel dimensions, "
            "annotations, and deterministically composited scale bar."
        )

    @staticmethod
    def _publication_file_identity(path: Path | None) -> dict[str, object] | None:
        if path is None:
            return None
        resolved = Path(path).expanduser().resolve(strict=False)
        try:
            stat = resolved.stat()
        except OSError:
            return {"path": str(resolved), "missing": True}
        return {
            "path": str(resolved),
            "size": int(stat.st_size),
            "modified_ns": int(stat.st_mtime_ns),
        }

    def _publication_input_payload(
        self, preset: PublicationExportPreset
    ) -> dict[str, object]:
        sources = {
            channel: self._publication_file_identity(
                self.current_image_paths.get(channel)
            )
            for channel in preset.recipe.image_channels
        }
        return {
            "roi": str(self.current_roi or ""),
            "mask": self._publication_file_identity(self.current_mask_path),
            "images": sources,
        }

    def _publication_input_fingerprint(
        self, preset: PublicationExportPreset
    ) -> str:
        import hashlib

        payload = self._publication_input_payload(preset)
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _snapshot_napari_scale_bar(scale_bar) -> dict[str, object]:
        if scale_bar is None:
            return {}
        values = {}
        for name in (
            "visible",
            "unit",
            "length",
            "colored",
            "color",
            "ticks",
            "font_size",
            "box",
            "box_color",
            "position",
            "opacity",
        ):
            if hasattr(scale_bar, name):
                value = getattr(scale_bar, name)
                values[name] = (
                    value.tolist() if hasattr(value, "tolist") else value
                )
        return values

    @staticmethod
    def _restore_napari_scale_bar(scale_bar, values: dict[str, object]) -> None:
        if scale_bar is None:
            return
        for name, value in values.items():
            if hasattr(scale_bar, name):
                try:
                    setattr(scale_bar, name, value)
                except (TypeError, ValueError):
                    continue

    def _render_publication_screenshot(
        self,
        preset: PublicationExportPreset,
        frame: ResolvedPublicationFrame,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Render one exact-FOV canvas and restore all temporary viewer state."""

        camera = self.viewer.camera
        old_center = tuple(camera.center)
        old_zoom = float(camera.zoom)
        canvas = getattr(
            getattr(getattr(self.viewer, "window", None), "_qt_viewer", None),
            "canvas",
            None,
        )
        if canvas is None or not hasattr(canvas, "size"):
            raise ValueError("Napari canvas dimensions are not available.")
        old_canvas_size = tuple(canvas.size)
        qt_window = getattr(getattr(self.viewer, "window", None), "_qt_window", None)
        ratio_getter = getattr(qt_window, "devicePixelRatioF", None)
        if ratio_getter is None:
            ratio_getter = getattr(qt_window, "devicePixelRatio", None)
        device_pixel_ratio = float(ratio_getter()) if ratio_getter else 1.0
        scale_bar = getattr(self.viewer, "scale_bar", None)
        old_scale_bar = self._snapshot_napari_scale_bar(scale_bar)
        render_scale = int(preset.output.supersampling)
        output_width, output_height = resolve_publication_output_size(
            preset.output, frame
        )
        geometry = publication_render_geometry(
            frame,
            output_width=output_width,
            output_height=output_height,
            supersampling=render_scale,
            device_pixel_ratio=device_pixel_ratio,
        )
        try:
            # Napari zoom is expressed in current *logical canvas* pixels per
            # world/source pixel.  Resize first, then set zoom.  Calling
            # screenshot(size=...) after setting zoom preserves the old canvas
            # rectangle and is the source of output-size-dependent zooming.
            canvas.size = (
                geometry.logical_canvas_width,
                geometry.logical_canvas_height,
            )
            self.QApplication.processEvents()
            centre = list(old_center)
            centre[-2:] = [frame.center_y, frame.center_x]
            camera.center = tuple(centre)
            camera.zoom = geometry.zoom
            if scale_bar is not None:
                # The final scale bar is composited at final-output resolution;
                # hiding Napari's overlay prevents monitor/DPI-dependent text.
                scale_bar.visible = False
            self.QApplication.processEvents()
            screenshot = self.viewer.screenshot(
                canvas_only=True,
                flash=False,
            )
        finally:
            canvas.size = old_canvas_size
            self.QApplication.processEvents()
            camera.center = old_center
            camera.zoom = old_zoom
            self._restore_napari_scale_bar(scale_bar, old_scale_bar)
        final_image = downsample_publication_image(
            screenshot,
            width=output_width,
            height=output_height,
        )
        composed, annotation_metadata = compose_publication_image(
            final_image,
            preset=preset,
            frame=frame,
            roi=str(self.current_roi),
        )
        render_metadata = {
            "resolution": preset.output.resolution,
            "resolution_scale": publication_resolution_scale(preset.output),
            "size_mode": preset.output.size_mode,
            "requested_width": preset.output.width,
            "requested_height": preset.output.height,
            "output_width": output_width,
            "output_height": output_height,
            "output_dpi": resolve_publication_dpi(preset.output),
            "supersampling": render_scale,
            "render_width": geometry.render_width,
            "render_height": geometry.render_height,
            "logical_canvas_width": geometry.logical_canvas_width,
            "logical_canvas_height": geometry.logical_canvas_height,
            "device_pixel_ratio": device_pixel_ratio,
            "source_pixel_size_x": preset.calibration.x_size,
            "source_pixel_size_y": preset.calibration.y_size,
            "source_pixel_unit": preset.calibration.unit,
            "physical_field_width": frame.field_width * preset.calibration.x_size,
            "physical_field_height": frame.field_height * preset.calibration.y_size,
            "output_pixel_size_x": (
                frame.field_width * preset.calibration.x_size / output_width
            ),
            "output_pixel_size_y": (
                frame.field_height * preset.calibration.y_size / output_height
            ),
            **frame.as_dict(),
            **annotation_metadata,
        }
        return composed, render_metadata

    def _publication_provenance(
        self,
        *,
        preset: PublicationExportPreset,
        frame: ResolvedPublicationFrame,
        rendering: dict[str, object],
        destination: Path,
    ) -> dict[str, object]:
        import importlib.metadata as package_metadata

        versions = {}
        for package in ("SpatialBiologyToolkit", "napari", "numpy", "pillow"):
            try:
                versions[package] = package_metadata.version(package)
            except package_metadata.PackageNotFoundError:
                versions[package] = "unknown"
        return {
            "schema_version": 3,
            "exported_at": datetime.now().astimezone().isoformat(),
            "destination": str(destination.resolve(strict=False)),
            "experiment_id": self.manifest.experiment_id,
            "experiment_revision": self.manifest.revision,
            "roi": str(self.current_roi),
            "preset_id": preset.preset_id,
            "preset_name": preset.name,
            "preset_fingerprint": preset.fingerprint,
            "recipe_fingerprint": preset.recipe.fingerprint,
            "input_fingerprint": self._publication_input_fingerprint(preset),
            "inputs": self._publication_input_payload(preset),
            "frame": frame.as_dict(),
            "rendering": rendering,
            "preset": preset.model_dump(mode="json"),
            "versions": versions,
        }

    def _publication_destination(
        self,
        *,
        preset: PublicationExportPreset,
        frame: ResolvedPublicationFrame,
        output_folder: Path,
        conflict_policy: str,
    ) -> tuple[Path, bool]:
        output_size = resolve_publication_output_size(preset.output, frame)
        destination = output_folder / build_publication_filename(
            preset,
            roi=str(self.current_roi),
            output_size=output_size,
        )
        if not destination.exists():
            return destination, False
        sidecar = destination.with_suffix(destination.suffix + ".json")
        if conflict_policy == "resume" and sidecar.is_file():
            try:
                existing = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            if (
                existing.get("preset_fingerprint") == preset.fingerprint
                and existing.get("input_fingerprint")
                == self._publication_input_fingerprint(preset)
            ):
                return destination, True
        if conflict_policy == "overwrite":
            return destination, False
        stem = destination.stem
        suffix = destination.suffix
        version = 2
        while True:
            candidate = destination.with_name(f"{stem}_v{version}{suffix}")
            if not candidate.exists():
                return candidate, False
            version += 1

    def _save_publication_for_current_roi(
        self,
        *,
        preset: PublicationExportPreset,
        output_folder: Path,
        conflict_policy: str,
    ) -> dict[str, object]:
        frame = self._resolved_publication_frame(preset)
        destination, skipped = self._publication_destination(
            preset=preset,
            frame=frame,
            output_folder=output_folder,
            conflict_policy=conflict_policy,
        )
        if skipped:
            return {
                "roi": str(self.current_roi),
                "status": "skipped_matching",
                "path": str(destination),
                "error": "",
            }
        image, rendering = self._render_publication_screenshot(preset, frame)
        provenance = self._publication_provenance(
            preset=preset,
            frame=frame,
            rendering=rendering,
            destination=destination,
        )
        save_publication_image(
            image,
            destination,
            dpi=resolve_publication_dpi(preset.output),
            metadata={
                "preset_fingerprint": preset.fingerprint,
                "recipe_fingerprint": preset.recipe.fingerprint,
                "roi": str(self.current_roi),
                "output_dpi": resolve_publication_dpi(preset.output),
                "source_pixel_size_x": preset.calibration.x_size,
                "source_pixel_size_y": preset.calibration.y_size,
                "source_pixel_unit": preset.calibration.unit,
                "physical_field_width": rendering["physical_field_width"],
                "physical_field_height": rendering["physical_field_height"],
                "output_pixel_size_x": rendering["output_pixel_size_x"],
                "output_pixel_size_y": rendering["output_pixel_size_y"],
            },
        )
        write_json(destination.with_suffix(destination.suffix + ".json"), provenance)
        return {
            "roi": str(self.current_roi),
            "status": "exported",
            "path": str(destination),
            "error": "",
        }

    def render_publication_preview(self) -> None:
        """Render the exact final composition into a resizable preview window."""

        from qtpy.QtCore import Qt
        from qtpy.QtGui import QImage, QPixmap
        from qtpy.QtWidgets import QDialog, QLabel, QScrollArea, QVBoxLayout

        preset = self._publication_preset_from_controls(
            preset_id="preview",
            capture_live=True,
            freeze_current_frame=True,
        )
        frame = self._resolved_publication_frame(preset)
        image, _rendering = self._render_publication_screenshot(preset, frame)
        rgba = np.ascontiguousarray(image)
        height, width = rgba.shape[:2]
        qimage = QImage(
            rgba.data,
            width,
            height,
            int(rgba.strides[0]),
            QImage.Format_RGBA8888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        preview = QDialog(self.root)
        preview.setWindowTitle(
            f"Publication preview — {self.current_roi} — {preset.source_recipe_name}"
        )
        preview.resize(min(width + 40, 1200), min(height + 80, 900))
        layout = QVBoxLayout(preview)
        summary = QLabel(
            f"Exact {width} × {height} px output preview. Scroll at 100% or resize "
            "the window; exporting uses this same renderer."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        scroll = QScrollArea()
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setPixmap(pixmap)
        image_label.resize(pixmap.size())
        scroll.setWidget(image_label)
        scroll.setWidgetResizable(False)
        layout.addWidget(scroll, 1)
        preview.setAttribute(Qt.WA_DeleteOnClose, True)
        preview.show()
        # Retain the modeless window through Qt ownership and an explicit list.
        if not hasattr(self, "publication_preview_windows"):
            self.publication_preview_windows = []
        self.publication_preview_windows.append(preview)
        preview.destroyed.connect(
            lambda: self.publication_preview_windows.remove(preview)
            if preview in self.publication_preview_windows
            else None
        )
        self.set_status(
            f"Rendered publication preview for ROI {self.current_roi!r} at "
            f"{width} × {height} pixels."
        )

    def export_current_publication_image(self) -> None:
        if self.current_mask is None or not self.current_roi:
            raise ValueError("Load an ROI before exporting it.")
        preset = self._publication_preset_from_controls(
            preset_id=str(uuid4()),
            capture_live=True,
            freeze_current_frame=True,
        )
        output_text = self.publication_output_folder_edit.text().strip()
        if not output_text:
            raise ValueError("Choose a publication output folder first.")
        output_folder = Path(output_text).expanduser().resolve(strict=False)
        result = self._save_publication_for_current_roi(
            preset=preset,
            output_folder=output_folder,
            conflict_policy=str(self.publication_conflict_combo.currentData()),
        )
        append_audit(
            self.paths,
            {
                "action": "export_publication_image",
                "roi": self.current_roi,
                "preset_fingerprint": preset.fingerprint,
                "status": result["status"],
                "path": result["path"],
            },
        )
        self.publication_progress_label.setText(
            f"{result['status'].replace('_', ' ').title()}: {result['path']}"
        )
        self.set_status(
            f"Publication image {result['status'].replace('_', ' ')}: "
            f"{result['path']}"
        )

    def _publication_selected_rois(self) -> list[str]:
        return [item.text() for item in self.publication_roi_list.selectedItems()]

    def _publication_bulk_preflight(
        self, rois: list[str], preset: PublicationExportPreset, output_folder: Path
    ) -> bool:
        from qtpy.QtWidgets import (
            QAbstractItemView,
            QDialog,
            QDialogButtonBox,
            QLabel,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
        )

        dialog = QDialog(self.root)
        dialog.setWindowTitle("Publication bulk-export preflight")
        dialog.resize(1000, min(760, 180 + len(rois) * 28))
        layout = QVBoxLayout(dialog)
        label = QLabel(
            "This check uses the Setup asset index and exact known paths; it does "
            "not rescan image folders. Missing requested channels are explicit "
            "warnings because those exports would not be composition-equivalent."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        table = QTableWidget(len(rois), 4)
        table.setHorizontalHeaderLabels(["ROI", "Mask", "Requested channels", "Output"])
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        warnings = 0
        for row, roi in enumerate(rois):
            mask_path = None
            try:
                mask_path = self._mask_path_index.get(roi) or self._mask_path_for_roi(roi)
                mask_text = "Ready" if Path(mask_path).is_file() else "Missing"
            except (FileNotFoundError, ValueError):
                mask_text = "Missing"
            indexed_channels = set(self._roi_image_path_index.get(roi, {}))
            requested = list(preset.recipe.image_channels)
            missing = [channel for channel in requested if channel not in indexed_channels]
            if not requested:
                channel_text = "No image channels in recipe"
            elif missing:
                channel_text = "Missing: " + ", ".join(missing)
            else:
                channel_text = f"Ready ({len(requested)})"
            if mask_text != "Ready" or missing:
                warnings += 1
            try:
                if preset.frame.mode == "full_roi":
                    if mask_path is None or mask_text != "Ready":
                        raise ValueError("mask is required to resolve native ROI size")
                    roi_shape = tuple(load_mask(mask_path).shape[:2])
                    current_frame = None
                elif preset.frame.mode == "current_view":
                    roi_shape = (1, 1)
                    current_frame = self._current_publication_camera_frame()
                else:
                    roi_shape = (1, 1)
                    current_frame = None
                frame = resolve_publication_frame(
                    preset.frame,
                    output=preset.output,
                    current_frame=current_frame,
                    roi_shape=roi_shape,
                )
                output_size = resolve_publication_output_size(preset.output, frame)
                filename = build_publication_filename(
                    preset,
                    roi=roi,
                    output_size=output_size,
                )
            except (OSError, ValueError) as error:
                filename = f"Unable to resolve: {error}"
            for column, text in enumerate((roi, mask_text, channel_text, filename)):
                table.setItem(row, column, QTableWidgetItem(str(text)))
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)
        summary = QLabel(
            f"{len(rois)} ROI(s); {warnings} row(s) contain warnings; output root: "
            f"{output_folder}. Continue only if any missing-channel exports are intentional."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Start bulk export")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        return dialog.exec() == QDialog.Accepted

    def start_publication_bulk_export(self) -> None:
        if self.publication_batch is not None:
            self.set_status("A publication bulk export is already running.")
            return
        self._activity_begin("Bulk publication export", "Preparing frozen recipe…")
        try:
            rois = self._publication_selected_rois()
            if not rois:
                raise ValueError("Select at least one ROI for bulk export.")
            preset = self._publication_preset_from_controls(
                preset_id=str(uuid4()),
                capture_live=True,
                freeze_current_frame=True,
            )
            output_text = self.publication_output_folder_edit.text().strip()
            if not output_text:
                raise ValueError("Choose a publication output folder first.")
            output_root = Path(output_text).expanduser().resolve(strict=False)
            output_folder = output_root / (slugify(preset.name) or "publication_export")
            if not self._publication_bulk_preflight(rois, preset, output_folder):
                self._activity_finish(True, "Bulk publication export cancelled at preflight.")
                return
            output_folder.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
            write_json(
                output_folder / f"export_preset_{preset.fingerprint[:10]}.json",
                preset.model_dump(mode="json"),
            )
            scale_bar = getattr(self.viewer, "scale_bar", None)
            locked_controls = (
                self.roi_combo,
                self.previous_roi_button,
                self.next_roi_button,
                self.reload_roi_button,
                self.recipe_preset_combo,
                self.load_recipe_preset_button,
                self.save_new_recipe_preset_button,
                self.update_recipe_preset_button,
                self.delete_recipe_preset_button,
            )
            self.publication_batch = {
                "preset": preset,
                "rois": list(rois),
                "index": 0,
                "results": [],
                "cancelled": False,
                "output_folder": output_folder,
                "conflict_policy": str(self.publication_conflict_combo.currentData()),
                "timestamp": timestamp,
                "started_at": datetime.now().astimezone().isoformat(),
                "original_roi": self.current_roi,
                "original_recipe": self.explore_recipe.model_copy(deep=True),
                "original_active_recipe_id": self.explore_review_state.active_recipe_id,
                "original_auto_reload": self.auto_reload_view_check.isChecked(),
                "original_camera_center": tuple(self.viewer.camera.center),
                "original_camera_zoom": float(self.viewer.camera.zoom),
                "original_scale_bar": self._snapshot_napari_scale_bar(scale_bar),
                "locked_controls": [
                    (control, control.isEnabled()) for control in locked_controls
                ],
            }
            self._publication_export_running = True
            blocked = self.auto_reload_view_check.blockSignals(True)
            self.auto_reload_view_check.setChecked(False)
            self.auto_reload_view_check.blockSignals(blocked)
            self.publication_progress_bar.setRange(0, len(rois))
            self.publication_progress_bar.setValue(0)
            self.publication_cancel_button.setEnabled(True)
            self.publication_export_bulk_button.setEnabled(False)
            self.publication_export_current_button.setEnabled(False)
            for control, _enabled in self.publication_batch["locked_controls"]:
                control.setEnabled(False)
            self.publication_progress_label.setText(
                f"Starting 0/{len(rois)} — frozen recipe {preset.source_recipe_name!r}."
            )
            self._activity_update(f"Starting 0/{len(rois)} publication images.")
            from qtpy.QtCore import QTimer

            QTimer.singleShot(0, self._publication_export_next)
        except Exception as error:  # noqa: BLE001 - modeless callback boundary
            pending = self.publication_batch
            if pending is not None:
                blocked = self.auto_reload_view_check.blockSignals(True)
                self.auto_reload_view_check.setChecked(
                    bool(pending.get("original_auto_reload", True))
                )
                self.auto_reload_view_check.blockSignals(blocked)
                for control, enabled in pending.get("locked_controls", []):
                    control.setEnabled(bool(enabled))
                self._restore_napari_scale_bar(
                    getattr(self.viewer, "scale_bar", None),
                    dict(pending.get("original_scale_bar", {})),
                )
            self.publication_batch = None
            self._publication_export_running = False
            self._activity_finish(False, f"{type(error).__name__}: {error}")
            self.set_status(f"ERROR — {type(error).__name__}: {error}")
            self.QMessageBox.critical(
                self.root, "napari_sbt", f"{type(error).__name__}: {error}"
            )

    def cancel_publication_bulk_export(self) -> None:
        if self.publication_batch is None:
            return
        self.publication_batch["cancelled"] = True
        self.publication_cancel_button.setEnabled(False)
        self.publication_progress_label.setText(
            "Cancellation requested; the current atomic image write will finish safely."
        )
        self._activity_update("Publication export cancellation requested.")

    def _publication_export_next(self) -> None:
        batch = self.publication_batch
        if batch is None:
            return
        rois = list(batch["rois"])
        index = int(batch["index"])
        if bool(batch["cancelled"]) or index >= len(rois):
            self._finish_publication_bulk_export()
            return
        roi = str(rois[index])
        preset = batch["preset"]
        self.publication_progress_label.setText(
            f"Loading {index + 1}/{len(rois)}: {roi}"
        )
        self._activity_update(f"Loading ROI {roi} ({index + 1}/{len(rois)}).")
        try:
            combo_blocked = self.roi_combo.blockSignals(True)
            if self.roi_combo.findText(roi) >= 0:
                self.roi_combo.setCurrentText(roi)
            self.roi_combo.blockSignals(combo_blocked)
            self.load_roi(roi)
            self._apply_explore_recipe(preset.recipe, replay=True)
            self.QApplication.processEvents()
        except Exception as error:  # noqa: BLE001 - continue-on-error batch boundary
            batch["results"].append(
                {
                    "roi": roi,
                    "status": "failed",
                    "path": "",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            batch["index"] = index + 1
            self.publication_progress_bar.setValue(index + 1)
            from qtpy.QtCore import QTimer

            QTimer.singleShot(0, self._publication_export_next)
            return
        from qtpy.QtCore import QTimer

        # A short event-loop yield allows Vispy to upload changed textures before
        # the canvas-only screenshot, without blocking the interface in a loop.
        QTimer.singleShot(60, self._publication_capture_batch_current)

    def _publication_capture_batch_current(self) -> None:
        batch = self.publication_batch
        if batch is None:
            return
        rois = list(batch["rois"])
        index = int(batch["index"])
        roi = str(rois[index])
        preset = batch["preset"]
        try:
            result = self._save_publication_for_current_roi(
                preset=preset,
                output_folder=Path(batch["output_folder"]),
                conflict_policy=str(batch["conflict_policy"]),
            )
        except Exception as error:  # noqa: BLE001 - continue-on-error batch boundary
            result = {
                "roi": roi,
                "status": "failed",
                "path": "",
                "error": f"{type(error).__name__}: {error}",
            }
        batch["results"].append(result)
        batch["index"] = index + 1
        completed = index + 1
        self.publication_progress_bar.setValue(completed)
        exported = sum(item["status"] == "exported" for item in batch["results"])
        skipped = sum(
            item["status"] == "skipped_matching" for item in batch["results"]
        )
        failed = sum(item["status"] == "failed" for item in batch["results"])
        self.publication_progress_label.setText(
            f"Completed {completed}/{len(rois)} — exported {exported}, "
            f"resumed/skipped {skipped}, failed {failed}."
        )
        self._activity_update(
            f"Publication images {completed}/{len(rois)}; {failed} failed."
        )
        from qtpy.QtCore import QTimer

        QTimer.singleShot(0, self._publication_export_next)

    def _finish_publication_bulk_export(self) -> None:
        batch = self.publication_batch
        if batch is None:
            return
        results = list(batch["results"])
        output_folder = Path(batch["output_folder"])
        timestamp = str(batch["timestamp"])
        cancelled = bool(batch["cancelled"])
        restore_error = ""
        try:
            original_recipe = batch["original_recipe"]
            self.explore_review_state.active_recipe_id = batch[
                "original_active_recipe_id"
            ]
            self.explore_recipe = original_recipe.model_copy(deep=True)
            original_auto = bool(batch["original_auto_reload"])
            original_roi = batch["original_roi"]
            if original_roi:
                combo_blocked = self.roi_combo.blockSignals(True)
                if self.roi_combo.findText(str(original_roi)) >= 0:
                    self.roi_combo.setCurrentText(str(original_roi))
                self.roi_combo.blockSignals(combo_blocked)
                self.load_roi(str(original_roi))
                self._apply_explore_recipe(original_recipe, replay=True)
            blocked = self.auto_reload_view_check.blockSignals(True)
            self.auto_reload_view_check.setChecked(original_auto)
            self.auto_reload_view_check.blockSignals(blocked)
            self.viewer.camera.center = batch["original_camera_center"]
            self.viewer.camera.zoom = float(batch["original_camera_zoom"])
            self._restore_napari_scale_bar(
                getattr(self.viewer, "scale_bar", None),
                dict(batch["original_scale_bar"]),
            )
        except Exception as error:  # noqa: BLE001 - best-effort UI restoration
            restore_error = f"{type(error).__name__}: {error}"
        finally:
            self._publication_export_running = False
            for control, enabled in batch.get("locked_controls", []):
                control.setEnabled(bool(enabled))
            self.publication_batch = None
            self.publication_cancel_button.setEnabled(False)
            self.publication_export_bulk_button.setEnabled(True)
            self.publication_export_current_button.setEnabled(True)

        manifest_path = output_folder / f"export_manifest_{timestamp}.csv"
        write_dataframe(manifest_path, pd.DataFrame(results))
        run_payload = {
            "schema_version": 1,
            "started_at": batch["started_at"],
            "finished_at": datetime.now().astimezone().isoformat(),
            "cancelled": cancelled,
            "preset_id": batch["preset"].preset_id,
            "preset_fingerprint": batch["preset"].fingerprint,
            "requested_rois": list(batch["rois"]),
            "result_counts": {
                "exported": sum(item["status"] == "exported" for item in results),
                "skipped_matching": sum(
                    item["status"] == "skipped_matching" for item in results
                ),
                "failed": sum(item["status"] == "failed" for item in results),
            },
            "manifest": str(manifest_path),
            "restore_error": restore_error,
        }
        write_json(output_folder / f"run_provenance_{timestamp}.json", run_payload)
        failed = run_payload["result_counts"]["failed"]
        status = "cancelled" if cancelled else "finished"
        detail = (
            f"Bulk publication export {status}: {len(results)}/"
            f"{len(batch['rois'])} ROI(s) processed, {failed} failed. "
            f"Manifest: {manifest_path}."
        )
        if restore_error:
            detail += f" Viewer restoration warning: {restore_error}."
        self.publication_progress_label.setText(detail)
        self._activity_finish(not failed and not restore_error, detail)
        self.set_status(detail)
        append_audit(
            self.paths,
            {
                "action": "bulk_export_publication_images",
                "preset_fingerprint": batch["preset"].fingerprint,
                "requested_rois": len(batch["rois"]),
                "processed_rois": len(results),
                "failed": failed,
                "cancelled": cancelled,
                "manifest": str(manifest_path),
            },
        )

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
        object_id = int(self.current_mask[row, column])
        try:
            self._show_cell_properties_for_object(object_id)
        except Exception as error:  # noqa: BLE001 - inspector must stay non-blocking
            self.cell_properties_summary_label.setText(
                f"Could not inspect this cell: {type(error).__name__}: {error}"
            )
        if active_tab not in {self.classify_tab_index, self.labeler_tab_index}:
            return
        if active_tab == self.classify_tab_index and not self.cell_picking_enabled:
            return
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
        selected_channels = {
            item.text() for item in self.channel_list.selectedItems()
        }
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
        for channel in self._ordered_variable_values(list(paths)):
            path = paths[channel]
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
            list_item.setSelected(
                channel in selected_channels
                or channel in self.explore_recipe.image_channels
            )
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

    def _display_normalization_parameters(
        self,
        channel: str,
    ) -> NimbusNormalizationParameters | None:
        candidates = [str(channel), str(channel).split(" [", 1)[0]]
        path = self.current_image_paths.get(str(channel))
        if path is not None:
            candidates.append(path.stem)
        for candidate in candidates:
            parameters = find_normalization_parameters(
                self.display_normalization,
                candidate,
            )
            if parameters is not None:
                return parameters
        return None

    def _display_image_load_kwargs(self, channel: str) -> dict[str, float | None]:
        settings = self._display_image_settings()
        parameters = self._display_normalization_parameters(channel)
        return {
            "quantile": float(settings.fallback_quantile),
            "minimum_pixel_counts": float(settings.minimum_pixel_counts),
            "normalization_value": (
                None if parameters is None else float(parameters.vmax)
            ),
            "normalization_lower_threshold": (
                0.0 if parameters is None else float(parameters.lower_threshold)
            ),
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
            display_settings = {
                # Apply blending before every reuse/cache branch. Otherwise a
                # cached scalar channel recreated after an ROI/recipe switch
                # inherits Napari's translucent_no_depth default.
                "blending": "translucent" if existing_is_rgb else "additive",
                **self._recipe_display_settings(
                    name,
                    default_colormap=None if existing_is_rgb else default_colormap,
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

    def _direct_label_colormap(self, colours: dict[int, object]):
        from napari.utils.colormaps import DirectLabelColormap

        return DirectLabelColormap(color_dict={None: "#00000000", **colours})

    def _categorical_observation_colours(
        self,
        name: str,
        defaults: dict[str, str],
    ) -> dict[str, object]:
        """Restore category colours without tying a recipe to one ROI's IDs."""

        resolved: dict[str, object] = dict(defaults)
        spec = self.explore_recipe.layer_colormap_specs.get(name)
        if not isinstance(spec, dict):
            return resolved
        colours = spec.get("colours", {})
        if spec.get("kind") == "categorical_labels" and isinstance(colours, dict):
            for category in defaults:
                if category in colours:
                    resolved[category] = colours[category]
            return resolved
        if spec.get("kind") == "direct_labels" and isinstance(colours, dict):
            # Recipes written before identity-preserving categorical overlays
            # keyed colours by their stable dataset-wide category code.
            for code, category in enumerate(defaults, start=1):
                if str(code) in colours:
                    resolved[category] = colours[str(code)]
        return resolved

    def _set_categorical_overlay_metadata(
        self,
        layer,
        *,
        observation: str,
        object_categories: pd.Series,
        category_colours: dict[str, object],
    ) -> None:
        """Attach enough semantics to save colours independently of object IDs."""

        representative_ids: dict[str, int] = {}
        for object_id, category in object_categories.items():
            representative_ids.setdefault(str(category), int(object_id))
        metadata = dict(getattr(layer, "metadata", {}) or {})
        metadata["napari_sbt_categorical_overlay"] = {
            "observation": str(observation),
            "representative_ids": representative_ids,
            "category_colours": dict(category_colours),
        }
        layer.metadata = metadata

    def _categorical_layer_colormap_spec(self, layer) -> dict | None:
        """Collapse an identity-keyed live colormap into a category-keyed recipe."""

        metadata = getattr(layer, "metadata", None)
        state = (
            metadata.get("napari_sbt_categorical_overlay")
            if isinstance(metadata, dict)
            else None
        )
        if not isinstance(state, dict):
            return None
        category_colours = state.get("category_colours", {})
        if not isinstance(category_colours, dict) or not category_colours:
            return None
        categories = [str(category) for category in category_colours]
        try:
            default_colormap = self._direct_label_colormap(
                {
                    index: category_colours[category]
                    for index, category in enumerate(categories, start=1)
                }
            )
            rgba = np.asarray(
                default_colormap.map(
                    np.arange(1, len(categories) + 1, dtype=np.int64)
                ),
                dtype=float,
            )
            resolved = {
                category: rgba[index].tolist()
                for index, category in enumerate(categories)
            }
        except (TypeError, ValueError):
            return None

        representatives = state.get("representative_ids", {})
        present = [
            (str(category), int(object_id))
            for category, object_id in representatives.items()
            if str(category) in resolved
        ]
        if present:
            try:
                live_rgba = np.asarray(
                    layer.colormap.map(
                        np.asarray([object_id for _category, object_id in present])
                    ),
                    dtype=float,
                )
                for index, (category, _object_id) in enumerate(present):
                    resolved[category] = live_rgba[index].tolist()
            except (AttributeError, TypeError, ValueError):
                pass
        return {"kind": "categorical_labels", "colours": resolved}

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
            reload_descriptor["label_encoding"] = "object_id"
            object_categories = categorical_object_categories(
                object_ids[selected],
                values[selected],
            )
            default_category_colours = categorical_colour_map(
                self.adata,
                observation,
            )
            category_colours = self._categorical_observation_colours(
                name,
                default_category_colours,
            )
            object_colours = {
                int(object_id): category_colours[str(category)]
                for object_id, category in object_categories.items()
                if str(category) in category_colours
            }
            display_settings = {
                "colormap": self._direct_label_colormap(object_colours),
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
                self._set_categorical_overlay_metadata(
                    layer,
                    observation=observation,
                    object_categories=object_categories,
                    category_colours=category_colours,
                )
                self._set_label_contour_from_recipe(layer, name)
                return 1
            overlay = population_identity_map(
                self.current_mask,
                object_categories.index,
                dtype=np.int32,
            )
            layer = self._replace_explore_layer(
                name,
                overlay,
                "labels",
                reload_descriptor=reload_descriptor,
                **display_settings,
            )
            self._set_categorical_overlay_metadata(
                layer,
                observation=observation,
                object_categories=object_categories,
                category_colours=category_colours,
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
                "colormap": self._population_layer_colormap(name, colour),
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
            overlay = population_identity_map(
                self.current_mask,
                object_ids[population_selected],
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
            if not self._publication_export_running:
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

    def _population_qc_scope_is_limited(self) -> bool:
        if self.manifest is None:
            return False
        scope = self.manifest.cell_scope
        return bool(
            scope.mode != "all_cells"
            or scope.eligible_cell_count < scope.total_cell_count
        )

    def _refresh_population_qc_scope_banner(self) -> None:
        """Make the cell universe used by Population QC explicit in the tab."""

        if not hasattr(self, "population_qc_scope_banner"):
            return
        if self.manifest is None:
            cell_text = (
                f"{self.adata.n_obs:,} loaded cells"
                if self.adata is not None
                else "no AnnData loaded"
            )
            self.population_qc_scope_banner.setText(
                "ℹ SETUP MODE — No frozen workspace cell scope is active "
                f"({cell_text}). A new workspace starts at All cells; review the "
                "scope in Setup before running the integrity check."
            )
            self.population_qc_scope_banner.setStyleSheet(
                "background: #e0f2fe; color: #075985; "
                "border: 2px solid #38bdf8; border-radius: 7px; padding: 9px; "
                "font-weight: 800;"
            )
            return

        scope = self.manifest.cell_scope
        eligible = int(scope.eligible_cell_count)
        total = int(scope.total_cell_count)
        represented_rois = int(scope.represented_roi_count)
        if not self._population_qc_scope_is_limited():
            self.population_qc_scope_banner.setText(
                f"✅ WHOLE DATASET — Population QC is using all {total:,} cells "
                f"across {represented_rois:,} ROIs. Population lists, marker "
                "suggestions, ROI rankings, and overlays use the complete dataset."
            )
            self.population_qc_scope_banner.setStyleSheet(
                "background: #dcfce7; color: #166534; "
                "border: 2px solid #22c55e; border-radius: 7px; padding: 9px; "
                "font-weight: 800;"
            )
            return

        percentage = 100.0 * eligible / total if total else 0.0
        selector = "frozen identity list"
        if scope.mode == "obs_values":
            values = [str(value) for value in scope.obs_values]
            displayed_values = values[:4]
            values_text = ", ".join(repr(value) for value in displayed_values)
            if len(values) > len(displayed_values):
                values_text += f", +{len(values) - len(displayed_values)} more"
            selector = f"{scope.obs_column} ∈ [{values_text}]"
        trial_note = ""
        if (
            self.manifest.experiment_mode == "feature_discovery_trial"
            and self.manifest.feature_trial is not None
        ):
            trial_note = (
                f" ROI buttons are further restricted to its "
                f"{len(self.manifest.feature_trial.selected_rois):,} trial ROIs."
            )
        self.population_qc_scope_banner.setText(
            f"⚠ LIMITED CELL SCOPE — Population QC is using {eligible:,} of "
            f"{total:,} cells ({percentage:.1f}%) across {represented_rois:,} ROIs; "
            "this is not a whole-dataset review. "
            f"Frozen selector: {selector}. Populations, marker suggestions, ROI "
            f"rankings, and overlays outside this scope are excluded.{trial_note}"
        )
        self.population_qc_scope_banner.setStyleSheet(
            "background: #fef3c7; color: #92400e; "
            "border: 3px solid #f59e0b; border-radius: 7px; padding: 9px; "
            "font-weight: 900;"
        )

    def refresh_population_qc_populations(self) -> None:
        """Refresh the populations offered by the dedicated QC tab."""

        self._refresh_population_qc_scope_banner()
        previous = self.population_qc_population_combo.currentText()
        observation = self.population_qc_obs_combo.currentText().strip()
        values: list[str] = []
        adata_view = self._population_qc_adata_view()
        if adata_view is not None and observation in adata_view.obs:
            series = adata_view.obs[observation]
            observed_values = set(series.dropna().astype(str))
            if isinstance(series.dtype, pd.CategoricalDtype):
                values = [
                    str(value)
                    for value in series.cat.categories
                    if str(value) in observed_values
                ]
            else:
                values = sorted(observed_values)
        self.population_qc_population_combo.blockSignals(True)
        self.population_qc_population_combo.clear()
        self.population_qc_population_combo.addItems(values)
        if previous in values:
            self.population_qc_population_combo.setCurrentText(previous)
        self.population_qc_population_combo.blockSignals(False)
        self.load_population_qc_recipe_controls()
        if observation and not values and adata_view is not None:
            scope_text = (
                "the limited workspace cell scope"
                if self._population_qc_scope_is_limited()
                else "the loaded dataset"
            )
            self.population_qc_status_label.setText(
                f"No non-missing {observation!r} populations are represented in "
                f"{scope_text}. Choose another observation or workspace."
            )

    def refresh_population_qc_marker_choices(self) -> None:
        """Populate RGB marker selectors from the images available for this ROI."""

        channels = self._ordered_variable_values(list(self.current_image_paths))
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
        seen_variables: set[str] = set()

        def append_matched(paths: Mapping[str, Path]) -> None:
            for display_name, path in paths.items():
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
                    canonical = str(canonical)
                    if canonical in seen_variables:
                        continue
                    seen_variables.add(canonical)
                    candidates.append((str(display_name), canonical))

        # Prefer names available in the ROI being reviewed. If it provides
        # enough matched channels, every suggestion can be selected directly.
        append_matched(self.current_image_paths)
        if len(candidates) >= 3:
            return candidates
        # Retain the dataset-wide index as a fallback for sparse/incomplete ROIs.
        for roi_paths in self._roi_image_path_index.values():
            append_matched(roi_paths)
        return candidates

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
            if value:
                self._set_population_qc_combo_value(combo, value)
            else:
                combo.setCurrentIndex(0)
            self.population_qc_lower_spins[colour].setValue(lower)
            self.population_qc_upper_spins[colour].setValue(upper)

    def suggest_population_qc_markers(self) -> None:
        observation, population = self._population_qc_selection()
        suggestions = self._cached_population_qc_marker_suggestions(
            observation, population
        )
        if not suggestions:
            adata_view = self._population_qc_adata_view()
            population_cells = 0
            if adata_view is not None and observation in adata_view.obs:
                population_cells = int(
                    adata_view.obs[observation]
                    .astype("string")
                    .eq(str(population))
                    .fillna(False)
                    .sum()
                )
            if population_cells == 0:
                message = (
                    f"{observation}={population!r} has no cells inside the active "
                    "Population QC cell scope. Review the scope banner at the top "
                    "of this tab or open an all-cells workspace."
                )
            else:
                message = (
                    f"The active scope contains {population_cells:,} cells from this "
                    "population, but no indexed image channels could be matched "
                    "safely to adata.var. Load an ROI, check image/marker names in "
                    "Setup, or choose the RGB channels manually."
                )
            self.population_qc_status_label.setText(f"⚠ {message}")
            self.set_status(f"Population QC marker suggestion unavailable: {message}")
            return
        self._apply_population_qc_marker_suggestions(suggestions)
        if len(suggestions) < 3:
            detail = (
                f"Only {len(suggestions)} distinct image marker(s) could be matched; "
                "the remaining RGB selector is blank."
            )
        else:
            unavailable = [
                marker
                for marker in suggestions
                if marker not in self.current_image_paths
            ]
            detail = (
                "A dataset-wide fallback marker is unavailable in this ROI and is "
                "shown in orange."
                if unavailable
                else "All three suggestions are available in the current ROI."
            )
        self.population_qc_status_label.setText(
            f"Suggested the highest-mean distinct matched image markers for "
            f"{population!r}. {detail} Review them before loading the view."
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
        scope_text = (
            " within the LIMITED workspace cell scope"
            if self._population_qc_scope_is_limited()
            else " across the whole dataset"
        )
        self.population_qc_status_label.setText(
            f"Showing {len(ranking):,} {ordering_text} eligible ROIs for "
            f"{observation}={population}{scope_text}. Green is unvisited; grey is "
            "viewed with this exact RGB, contrast, and outline recipe."
        )

    def recalculate_population_qc_rois(self) -> None:
        """Explicitly invalidate cached abundance rankings and rebuild the list."""

        self._population_qc_ranking_cache.clear()
        self.refresh_population_qc_rois()

    def activate_population_qc_roi(self, roi: str) -> None:
        """Load one ranked ROI with the current Population QC recipe exactly once."""

        roi = str(roi)
        roi_changed = self.current_mask is None or str(self.current_roi) != roi
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
        self.load_roi(roi)
        if not roi_changed or not self.auto_reload_view_check.isChecked():
            # A Population QC ROI button explicitly requests this RGB recipe.
            # A genuine ROI change already replays it through load_roi when
            # automatic replay is enabled; revisit/off-mode cases need one
            # deliberate replay here.
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

    def rank_rois_by_marker(self) -> None:
        selected = [item.text() for item in self.marker_overlay_list.selectedItems()]
        if len(selected) != 1:
            raise ValueError(
                "Select exactly one cell-level marker before ranking ROIs."
            )
        marker = selected[0]
        eligible = set(self.cohort["ROI"].astype(str))
        if (
            self.manifest.experiment_mode == "feature_discovery_trial"
            and self.manifest.feature_trial is not None
        ):
            eligible &= set(self.manifest.feature_trial.selected_rois)
        whole_dataset = self.overlay_full_dataset_check.isChecked()
        adata_view = self.adata if whole_dataset else self._population_qc_adata_view()
        ranking = rank_marker_rois(
            adata_view,
            marker=marker,
            roi_obs=self.manifest.roi_obs,
            eligible_rois=eligible,
        )
        ranked = [roi for roi, _mean in ranking]
        if not ranked:
            raise ValueError(
                f"No eligible ROIs contain finite adata.X values for {marker!r}."
            )
        self.roi_combo.blockSignals(True)
        self.roi_combo.clear()
        self.roi_combo.addItems(ranked)
        self.roi_combo.blockSignals(False)
        self._refresh_roi_review_colours()
        self.roi_combo.setCurrentIndex(0)
        self.load_roi(ranked[0])
        scope = "the whole AnnData" if whole_dataset else "the active cell scope"
        self.set_status(
            f"Ranked {len(ranked):,} ROIs by mean intracellular {marker} signal "
            f"across {scope}; highest mean {ranking[0][1]:.4g}."
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
            "Population transferred into Setup. Review the eligible-cell counts and "
            "validation summary, then click Create workspace and start to freeze it."
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
        self.refresh_feature_readiness()
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
        try:
            provenance = json.loads(
                self.paths.feature_manifest.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "The feature table has no readable provenance. Rebuild features "
                "before training or scoring."
            ) from exc
        if (
            provenance.get("feature_extraction_contract_version")
            != FEATURE_EXTRACTION_CONTRACT_VERSION
        ):
            raise ValueError(
                "The active feature table was built with an older extraction "
                "contract. Rebuild/resume features so Nimbus lower thresholds "
                "are applied before training or scoring."
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
        roi = str(roi)
        if self.roi_combo.currentText() != roi:
            # currentTextChanged is synchronous and performs the real load.
            self.roi_combo.setCurrentText(roi)
        if self.current_mask is None or str(self.current_roi) != roi:
            # Fallback for a blocked signal or an ROI absent from the selector.
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
        self.refresh_feature_readiness()

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

    def _update_identity_integration_controls(self) -> None:
        enabled = bool(
            self._classification_enabled
            and self.final_integration_enable_check.isChecked()
        )
        for widget in (
            self.final_integration_source_combo,
            self.final_integration_output_edit,
            self.final_integration_naming_combo,
            self.final_integration_mapping_table,
            self.preview_identity_integration_button,
            self.build_identity_integration_button,
        ):
            widget.setEnabled(enabled)
        self.export_assignments_button.setText(
            "Export integrated labels CSV/Parquet"
            if self.final_integration_enable_check.isChecked()
            else "Export cohort final identities CSV/Parquet"
        )

    def _identity_integration_enabled_changed(self, checked: bool) -> None:
        self._update_identity_integration_controls()
        self._mark_identity_integration_stale()
        if not checked:
            self.identity_integration_summary_label.setText(
                "Optional integration is disabled; exports will contain cohort-only "
                "final identities."
            )

    def _identity_integration_naming_changed(self, *_args) -> None:
        self._refresh_identity_integration_mapping()
        self._mark_identity_integration_stale()

    def _identity_integration_mapping_changed(self, item) -> None:
        if self._updating_identity_integration_controls or item.column() != 2:
            return
        class_item = self.final_integration_mapping_table.item(item.row(), 0)
        if class_item is None:
            return
        class_id = class_item.data(self.Qt.UserRole)
        if class_id is not None:
            self._identity_integration_custom_names[str(class_id)] = item.text()
        self._mark_identity_integration_stale()

    def _refresh_identity_integration_mapping(self) -> None:
        if not hasattr(self, "final_integration_mapping_table"):
            return
        self._updating_identity_integration_controls = True
        try:
            self.final_integration_mapping_table.setRowCount(0)
            if self.manifest is None:
                self.final_integration_mapping_help.setText(
                    "Create or load a classification workspace to define names."
                )
                return
            strategy = str(
                self.final_integration_naming_combo.currentData() or "class_names"
            )
            custom = strategy == "custom"
            for definition in self.manifest.classes:
                row = self.final_integration_mapping_table.rowCount()
                self.final_integration_mapping_table.insertRow(row)
                class_item = self.QTableWidgetItem(
                    f"{definition.shortcut}: {definition.name}"
                )
                class_item.setData(self.Qt.UserRole, definition.class_id)
                class_item.setFlags(class_item.flags() & ~self.Qt.ItemIsEditable)
                name = (
                    self._identity_integration_custom_names.get(
                        definition.class_id, definition.name
                    )
                    if custom
                    else definition.name
                )
                name_item = self.QTableWidgetItem(name)
                if not custom:
                    name_item.setFlags(name_item.flags() & ~self.Qt.ItemIsEditable)
                self.final_integration_mapping_table.setItem(row, 0, class_item)
                self.final_integration_mapping_table.setItem(
                    row, 1, self._class_colour_item(definition.color)
                )
                self.final_integration_mapping_table.setItem(row, 2, name_item)
            if strategy == "source_and_class":
                self.final_integration_mapping_help.setText(
                    "Each assigned cell becomes ‘existing source label → class "
                    "name’. The table shows the class-name portion."
                )
            elif custom:
                self.final_integration_mapping_help.setText(
                    "Edit the final population name for each classification class."
                )
            else:
                self.final_integration_mapping_help.setText(
                    "Assigned cells use the classification class display names "
                    "exactly as shown."
                )
        finally:
            self._updating_identity_integration_controls = False

    def _identity_integration_class_labels(self) -> dict[str, str]:
        labels: dict[str, str] = {}
        for row in range(self.final_integration_mapping_table.rowCount()):
            class_item = self.final_integration_mapping_table.item(row, 0)
            name_item = self.final_integration_mapping_table.item(row, 2)
            if class_item is None or name_item is None:
                continue
            class_id = class_item.data(self.Qt.UserRole)
            if class_id is not None:
                labels[str(class_id)] = name_item.text().strip()
        return labels

    def _identity_integration_table_from_controls(
        self, assignments: pd.DataFrame
    ) -> pd.DataFrame:
        if self.adata is None:
            raise ValueError("Load AnnData before integrating final identities.")
        return build_integrated_identity_table(
            self.adata,
            assignments,
            source_obs=self.final_integration_source_combo.currentText(),
            output_obs=self.final_integration_output_edit.text(),
            class_labels=self._identity_integration_class_labels(),
            naming_strategy=str(
                self.final_integration_naming_combo.currentData() or "class_names"
            ),
            roi_obs=self.manifest.roi_obs,
            object_id_obs=self.manifest.object_id_obs,
        )

    def _identity_integration_plan_from_table(
        self, table: pd.DataFrame
    ) -> dict[str, object]:
        source_obs = self.final_integration_source_combo.currentText().strip()
        output_obs = self.final_integration_output_edit.text().strip()
        class_labels = self._identity_integration_class_labels()
        source_colours = categorical_colour_map(self.adata, source_obs)
        class_colours = {
            definition.class_id: definition.color
            for definition in self.manifest.classes
        }
        category_colours = dict(source_colours)
        assigned = table.loc[table["final_class_id"].notna()]
        for class_id, label in zip(
            assigned["final_class_id"].astype(str),
            assigned[output_obs].astype(str),
        ):
            category_colours.setdefault(
                str(label),
                source_colours.get(
                    str(label), class_colours.get(str(class_id), "#808080")
                ),
            )
        source_labels = set(table["source_label"].dropna().astype(str))
        integrated_class_labels = set(
            assigned[output_obs].dropna().astype(str)
        )
        collisions = sorted(source_labels & integrated_class_labels)
        duplicate_targets = {
            label: sorted(class_ids)
            for label, class_ids in pd.Series(class_labels).groupby(
                pd.Series(class_labels)
            ).groups.items()
            if len(class_ids) > 1
        }
        source_frame = pd.DataFrame(
            {
                "obs_name": self.adata.obs_names.astype(str),
                "source_label": self.adata.obs[source_obs].astype("string"),
            }
        )
        return {
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "source_obs": source_obs,
            "output_obs": output_obs,
            "naming_strategy": str(
                self.final_integration_naming_combo.currentData() or "class_names"
            ),
            "class_labels": class_labels,
            "category_colours": category_colours,
            "source_label_collisions": collisions,
            "duplicate_class_targets": duplicate_targets,
            "source_label_fingerprint": dataframe_sha256(
                source_frame, ["obs_name", "source_label"]
            ),
            "output_obs_already_exists": output_obs in self.adata.obs,
            "total_cells": int(len(table)),
            "cohort_cells": int(table["is_classification_cohort"].sum()),
            "replaced_cells": int(table["final_class_id"].notna().sum()),
            "retained_source_cells": int(table["final_class_id"].isna().sum()),
            "final_identity_signature": self.final_identity_signature,
        }

    def _identity_integration_signature_value(self) -> str:
        if self.adata is None or self.final_identity_signature is None:
            return ""
        source_obs = self.final_integration_source_combo.currentText().strip()
        if source_obs not in self.adata.obs:
            return ""
        source_frame = pd.DataFrame(
            {
                "obs_name": self.adata.obs_names.astype(str),
                "source_label": self.adata.obs[source_obs].astype("string"),
            }
        )
        payload = {
            "final_identity_signature": self.final_identity_signature,
            "source_obs": source_obs,
            "source_fingerprint": dataframe_sha256(
                source_frame, ["obs_name", "source_label"]
            ),
            "output_obs": self.final_integration_output_edit.text().strip(),
            "naming_strategy": self.final_integration_naming_combo.currentData(),
            "class_labels": self._identity_integration_class_labels(),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _mark_identity_integration_stale(self, *_args) -> None:
        self.identity_integration_signature = None
        self.integrated_identity_table = pd.DataFrame()
        self.identity_integration_plan = {}
        if self.final_integration_enable_check.isChecked():
            self.identity_integration_summary_label.setText(
                "Integration not built, or its inputs changed. Create final cell "
                "identities first, then build the integrated labels before export."
            )

    def preview_identity_integration(self) -> None:
        assignments = self._require_current_final_identities()
        table = self._identity_integration_table_from_controls(assignments)
        plan = self._identity_integration_plan_from_table(table)
        overlap = integrated_identity_crosstab(
            table,
            output_obs=str(plan["output_obs"]),
        )
        self._show_identity_integration_overlap(overlap, plan)

    def _show_identity_integration_overlap(
        self, overlap: pd.DataFrame, plan: dict[str, object]
    ) -> None:
        from qtpy.QtWidgets import (
            QAbstractItemView,
            QDialog,
            QDialogButtonBox,
            QHeaderView,
            QLabel,
            QTableWidget,
            QVBoxLayout,
        )

        dialog = QDialog(self.root)
        dialog.setWindowTitle("Final-label integration overlap / confusion matrix")
        dialog.resize(900, 600)
        layout = QVBoxLayout(dialog)
        explanation = QLabel(
            f"Rows are existing labels from {plan['source_obs']!r}; columns are "
            f"accepted integrated labels for observation {plan['output_obs']!r}. "
            "This is an overlap table, not a model-accuracy score."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        display = overlap.copy()
        if not display.empty:
            display["Total"] = display.sum(axis=1)
            display.loc["Total"] = display.sum(axis=0)
        widget = QTableWidget(display.shape[0], display.shape[1] + 1)
        widget.setHorizontalHeaderLabels(
            ["Existing source label", *[str(value) for value in display.columns]]
        )
        for row_index, (source_label, values) in enumerate(display.iterrows()):
            widget.setItem(
                row_index, 0, self.QTableWidgetItem(str(source_label))
            )
            for column_index, value in enumerate(values, start=1):
                widget.setItem(
                    row_index, column_index, self.QTableWidgetItem(f"{int(value):,}")
                )
        widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(widget)
        warning_parts = []
        if plan["source_label_collisions"]:
            warning_parts.append(
                "Matching source/final names will merge explicitly: "
                + ", ".join(plan["source_label_collisions"])
            )
        if plan["duplicate_class_targets"]:
            warning_parts.append(
                "Multiple classifier classes share a final name: "
                + ", ".join(plan["duplicate_class_targets"])
            )
        if plan["output_obs_already_exists"]:
            warning_parts.append(
                f"The target observation {plan['output_obs']!r} already exists. "
                "AnnData export will replace it in the output copy; applying to a "
                "live object requires confirmation."
            )
        summary = QLabel(
            "\n".join(warning_parts)
            if warning_parts
            else "No name collisions or classifier-class merges were detected."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def build_identity_integration(self) -> pd.DataFrame:
        assignments = self._require_current_final_identities()
        table = self._identity_integration_table_from_controls(assignments)
        plan = self._identity_integration_plan_from_table(table)
        signature = self._identity_integration_signature_value()
        plan["integration_signature"] = signature
        self.integrated_identity_table = table
        self.identity_integration_plan = plan
        self.identity_integration_signature = signature
        canonical = self.paths.exports / "integrated_identities.parquet"
        export_assignment_table(table, canonical)
        write_json(self.paths.exports / "identity_integration.json", plan)
        append_audit(
            self.paths,
            {
                "action": "build_integrated_final_identities",
                "canonical_table": str(canonical),
                "integration": plan,
            },
        )
        collision_text = (
            f" {len(plan['source_label_collisions'])} source/final name collision(s) "
            "will merge explicitly."
            if plan["source_label_collisions"]
            else " No source/final name collisions were detected."
        )
        self.identity_integration_summary_label.setText(
            f"Current: {plan['replaced_cells']:,} classified cells replace their "
            f"source label; {plan['retained_source_cells']:,} cells retain it. "
            f"Output observation: {plan['output_obs']!r}.{collision_text}"
        )
        self.set_status(
            f"Built full-dataset integrated labels in {canonical.name}; exports "
            "will now use this table."
        )
        return table

    def _require_current_identity_integration(
        self,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        if (
            self.identity_integration_signature is None
            or self.integrated_identity_table.empty
        ):
            raise ValueError(
                "Integrated labels have not been built. Use optional step 5 before "
                "exporting, or disable integration for a cohort-only export."
            )
        if (
            self.identity_integration_signature
            != self._identity_integration_signature_value()
        ):
            self._mark_identity_integration_stale()
            raise ValueError(
                "Final identities, source labels, or integration names changed. "
                "Build the integrated labels again before export."
            )
        return self.integrated_identity_table.copy(), dict(
            self.identity_integration_plan
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
        self._mark_identity_integration_stale()
        if self.final_identity_signature is None:
            return
        self.final_identity_signature = None
        self.final_identity_summary_label.setText(
            "Decision rules or inputs changed. Click Create / refresh final cell "
            "identities before exporting."
        )

    def create_final_identities(self) -> pd.DataFrame:
        self._mark_identity_integration_stale()
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
        integration_plan: dict[str, object] | None = None
        export_table = assignments
        export_kind = "cohort final identities"
        if self.final_integration_enable_check.isChecked():
            export_table, integration_plan = (
                self._require_current_identity_integration()
            )
            export_kind = (
                f"full-dataset integrated labels ({integration_plan['output_obs']})"
            )
        destination = Path(self.assignment_path_edit.text()).expanduser()
        if not destination.suffix:
            destination = destination.with_suffix(".csv")
            self.assignment_path_edit.setText(str(destination))
        export_assignment_table(export_table, destination)
        append_audit(
            self.paths,
            {
                "action": "export_final_identity_table",
                "destination": str(destination),
                "decision": self.final_identity_decision,
                "integration": integration_plan,
            },
        )
        self.set_status(f"Exported {export_kind}: {destination}")

    def export_adata(self) -> None:
        if not self.manifest.anndata_path:
            raise ValueError("Annotated AnnData export requires an AnnData source.")
        assignments = self._require_current_final_identities()
        integration_table = None
        integration_plan = None
        if self.final_integration_enable_check.isChecked():
            integration_table, integration_plan = (
                self._require_current_identity_integration()
            )
        destination = Path(self.annotated_path_edit.text())
        export_annotated_anndata(
            self.manifest.anndata_path,
            destination,
            assignments,
            self.manifest,
            feature_provenance=self._feature_provenance(),
            model_provenance=self._model_provenance(),
            metrics={"final_identity_decision": self.final_identity_decision},
            integration_table=integration_table,
            integration_provenance=integration_plan,
            create_legacy_combined=False,
        )
        append_audit(
            self.paths,
            {
                "action": "export_final_identities_to_anndata_copy",
                "destination": str(destination),
                "decision": self.final_identity_decision,
                "integration": integration_plan,
            },
        )
        integration_text = (
            f" with integrated observation {integration_plan['output_obs']!r}"
            if integration_plan is not None
            else " with cohort-only classification annotations"
        )
        self.set_status(
            f"Exported atomic annotated AnnData copy{integration_text}: "
            f"{destination}"
        )

    def apply_final_identities_to_live_anndata(self) -> None:
        if self.adata is None:
            raise ValueError("No live AnnData object is loaded in this session.")
        assignments = self._require_current_final_identities()
        integration_table = None
        integration_plan = None
        if self.final_integration_enable_check.isChecked():
            integration_table, integration_plan = (
                self._require_current_identity_integration()
            )
        integration_question = (
            f"The full-dataset integrated observation "
            f"{integration_plan['output_obs']!r} will also be added or replaced."
            if integration_plan is not None
            else "No full-dataset integrated observation will be created."
        )
        reply = self.QMessageBox.question(
            self.root,
            "Apply final identities to live AnnData",
            "Add the cohort subclass, source, confidence, uncertainty, probability, "
            "and provenance fields to the AnnData object held in memory?\n\n"
            f"{integration_question}\n\n"
            "This does not write to or overwrite its source file.",
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
            integration_table=integration_table,
            integration_provenance=integration_plan,
            create_legacy_combined=False,
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
                "integration": integration_plan,
            },
        )
        integration_text = (
            f" and full-dataset {integration_plan['output_obs']}"
            if integration_plan is not None
            else ""
        )
        self.set_status(
            f"Applied final identities to the live AnnData as "
            f"{self.manifest.output_obs_slug}_subclass{integration_text} and related "
            "fields; no source file was written."
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
    controller.install_cell_properties_dock()
    # Reapply the preferred split once Qt has completed this event-loop turn;
    # this keeps the compact readiness dock below Layers after Napari finishes
    # sizing all newly added docks.
    from qtpy.QtCore import QTimer

    QTimer.singleShot(0, controller._position_auxiliary_docks)
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
