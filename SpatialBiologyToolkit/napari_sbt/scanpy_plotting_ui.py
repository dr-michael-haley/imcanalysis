"""Qt controls for the NapariSBT Scanpy plotting workspace."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .scanpy_plotting import (
    PLOT_TYPE_LABELS,
    ScanpyPlotRequest,
    groupable_obs_columns,
    matrix_source_choices,
    matrix_source_var_names,
    ordered_obs_values,
    resolve_plot_cell_mask,
)


class ScanpyPlottingPanel:
    """Build and manage the controls without coupling plot logic to ``app.py``."""

    def __init__(
        self,
        *,
        group_factory: Callable[[str, str, str], object],
        generate_callback: Callable[[], None],
        refresh_callback: Callable[[], None],
        focus_callback: Callable[[str], None],
        close_callback: Callable[[str], None],
        close_all_callback: Callable[[], None],
    ) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QAbstractItemView,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QPushButton,
            QSpinBox,
            QStackedWidget,
            QTreeWidget,
            QTreeWidgetItem,
            QVBoxLayout,
            QWidget,
        )

        self.Qt = Qt
        self.QTreeWidgetItem = QTreeWidgetItem
        self._adata = None
        self._roi_obs: str | None = None
        self._cohort_obs_names: set[str] | None = None
        self._window_items: dict[str, object] = {}

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        intro = QLabel(
            "Create quick, regenerable QC plots from the live AnnData object. "
            "Plotting is read-only: it does not recompute embeddings, neighbours, "
            "clustering, normalization, or batch correction."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        data_group = group_factory(
            "1. Choose data and cells",
            "scanpy_plotting",
            "Choose data and cells",
        )
        data_form = QFormLayout(data_group)
        self.groupby_combo = QComboBox()
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("All cells", "all_cells")
        self.scope_combo.addItem("Current classification cohort", "cohort")
        self.scope_combo.addItem("Only selected populations", "selected_groups")
        self.group_values_list = QListWidget()
        self.group_values_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.group_values_list.setMaximumHeight(125)
        group_actions = QWidget()
        group_actions_layout = QHBoxLayout(group_actions)
        group_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.select_all_groups_button = QPushButton("Select all")
        self.clear_groups_button = QPushButton("Clear selection")
        group_actions_layout.addWidget(self.select_all_groups_button)
        group_actions_layout.addWidget(self.clear_groups_button)
        group_actions_layout.addStretch(1)
        self.roi_list = QListWidget()
        self.roi_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.roi_list.setMaximumHeight(105)
        roi_actions = QWidget()
        roi_actions_layout = QHBoxLayout(roi_actions)
        roi_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.clear_rois_button = QPushButton("Clear ROI selection")
        self.clear_rois_button.setToolTip(
            "Use every available ROI by clearing the optional ROI filter."
        )
        roi_actions_layout.addWidget(self.clear_rois_button)
        roi_actions_layout.addStretch(1)
        self.matrix_source_combo = QComboBox()
        self.data_summary_label = QLabel("Load AnnData to configure plotting.")
        self.data_summary_label.setWordWrap(True)
        data_form.addRow("Labels / populations", self.groupby_combo)
        data_form.addRow("Cell scope", self.scope_combo)
        data_form.addRow("Populations", self.group_values_list)
        data_form.addRow("", group_actions)
        data_form.addRow("ROIs (none selected = all)", self.roi_list)
        data_form.addRow("", roi_actions)
        data_form.addRow("Expression matrix", self.matrix_source_combo)
        data_form.addRow("Selection summary", self.data_summary_label)
        layout.addWidget(data_group)

        type_group = group_factory(
            "2. Choose what to plot",
            "scanpy_plotting",
            "Choose a plot",
        )
        type_form = QFormLayout(type_group)
        self.plot_type_combo = QComboBox()
        for key, label in PLOT_TYPE_LABELS.items():
            self.plot_type_combo.addItem(label, key)
        self.plot_description_label = QLabel()
        self.plot_description_label.setWordWrap(True)
        type_form.addRow("Plot", self.plot_type_combo)
        type_form.addRow("What it shows", self.plot_description_label)
        layout.addWidget(type_group)

        options_group = group_factory(
            "3. Adjust this plot",
            "scanpy_plotting",
            "Plot options",
        )
        options_layout = QVBoxLayout(options_group)
        self.options_stack = QStackedWidget()
        options_layout.addWidget(self.options_stack)

        embedding_page = QWidget()
        embedding_form = QFormLayout(embedding_page)
        self.embedding_combo = QComboBox()
        self.embedding_x_spin = QSpinBox()
        self.embedding_x_spin.setRange(1, 2)
        self.embedding_y_spin = QSpinBox()
        self.embedding_y_spin.setRange(1, 2)
        self.embedding_y_spin.setValue(2)
        self.point_limit_spin = QSpinBox()
        self.point_limit_spin.setRange(100, 1_000_000)
        self.point_limit_spin.setSingleStep(10_000)
        self.point_limit_spin.setValue(50_000)
        self.point_limit_spin.setSuffix(" cells")
        self.point_size_spin = QDoubleSpinBox()
        self.point_size_spin.setRange(0.1, 100.0)
        self.point_size_spin.setDecimals(1)
        self.point_size_spin.setValue(3.0)
        self.point_alpha_spin = QDoubleSpinBox()
        self.point_alpha_spin.setRange(0.05, 1.0)
        self.point_alpha_spin.setSingleStep(0.05)
        self.point_alpha_spin.setValue(0.75)
        self.centroid_labels_check = QCheckBox("Write population names at centroids")
        embedding_form.addRow("Embedding", self.embedding_combo)
        embedding_form.addRow("Horizontal component", self.embedding_x_spin)
        embedding_form.addRow("Vertical component", self.embedding_y_spin)
        embedding_form.addRow("Maximum displayed points", self.point_limit_spin)
        embedding_form.addRow("Point size", self.point_size_spin)
        embedding_form.addRow("Point opacity", self.point_alpha_spin)
        embedding_form.addRow("Labels", self.centroid_labels_check)
        self.options_stack.addWidget(embedding_page)

        expression_page = QWidget()
        expression_form = QFormLayout(expression_page)
        self.marker_search_edit = QLineEdit()
        self.marker_search_edit.setPlaceholderText("Type to filter markers…")
        self.marker_list = QListWidget()
        self.marker_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.marker_list.setMaximumHeight(190)
        marker_actions = QWidget()
        marker_actions_layout = QHBoxLayout(marker_actions)
        marker_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.select_visible_markers_button = QPushButton("Select visible")
        self.clear_markers_button = QPushButton("Clear")
        marker_actions_layout.addWidget(self.select_visible_markers_button)
        marker_actions_layout.addWidget(self.clear_markers_button)
        marker_actions_layout.addStretch(1)
        self.expression_scale_combo = QComboBox()
        self.expression_scale_combo.addItem("Marker-wise z-score", "zscore_marker")
        self.expression_scale_combo.addItem("Marker-wise 0–1 range", "minmax_marker")
        self.expression_scale_combo.addItem("Unscaled mean expression", "none")
        self.positivity_spin = QDoubleSpinBox()
        self.positivity_spin.setRange(-1_000_000, 1_000_000)
        self.positivity_spin.setDecimals(4)
        self.positivity_spin.setValue(0.0)
        self.positivity_spin.setToolTip(
            "A cell is counted as positive when its selected-matrix value is "
            "strictly greater than this threshold."
        )
        self.expression_colormap_combo = QComboBox()
        for label, value in (
            ("Automatic", "automatic"),
            ("Viridis", "viridis"),
            ("Blue", "Blues"),
            ("Red", "Reds"),
            ("Magma", "magma"),
            ("Plasma", "plasma"),
            ("Diverging blue–red", "coolwarm"),
        ):
            self.expression_colormap_combo.addItem(label, value)
        self.side_annotation_combo = QComboBox()
        self.side_annotation_combo.addItem("None", "none")
        self.side_annotation_combo.addItem(
            "Fresh dendrogram — recalculate now", "dendrogram"
        )
        self.side_annotation_combo.addItem("Population cell totals", "totals")
        self.totals_sort_combo = QComboBox()
        self.totals_sort_combo.addItem("Keep population order", "none")
        self.totals_sort_combo.addItem("Smallest population first", "ascending")
        self.totals_sort_combo.addItem("Largest population first", "descending")
        self.dendrogram_correlation_combo = QComboBox()
        self.dendrogram_correlation_combo.addItem("Pearson", "pearson")
        self.dendrogram_correlation_combo.addItem("Spearman", "spearman")
        self.dendrogram_correlation_combo.addItem("Kendall", "kendall")
        self.dendrogram_linkage_combo = QComboBox()
        self.dendrogram_linkage_combo.addItem("Complete", "complete")
        self.dendrogram_linkage_combo.addItem("Average", "average")
        self.dendrogram_linkage_combo.addItem("Single", "single")
        self.dendrogram_optimal_ordering_check = QCheckBox(
            "Minimise distances between adjacent leaves"
        )
        self.dendrogram_optimal_ordering_check.setChecked(True)
        self.swap_axes_check = QCheckBox("Put populations on the horizontal axis")
        self.fresh_dendrogram_label = QLabel(
            "Fresh dendrograms use only the currently selected cells and markers. "
            "Any dendrogram stored in the source AnnData is ignored."
        )
        self.fresh_dendrogram_label.setWordWrap(True)
        expression_form.addRow("Find marker", self.marker_search_edit)
        expression_form.addRow("Markers", self.marker_list)
        expression_form.addRow("", marker_actions)
        expression_form.addRow("Colour scaling", self.expression_scale_combo)
        expression_form.addRow("Dot-plot positivity threshold", self.positivity_spin)
        expression_form.addRow("Colour map", self.expression_colormap_combo)
        expression_form.addRow("Side annotation", self.side_annotation_combo)
        expression_form.addRow("Totals ordering", self.totals_sort_combo)
        expression_form.addRow(
            "Dendrogram correlation", self.dendrogram_correlation_combo
        )
        expression_form.addRow("Dendrogram linkage", self.dendrogram_linkage_combo)
        expression_form.addRow(
            "Dendrogram ordering", self.dendrogram_optimal_ordering_check
        )
        expression_form.addRow("Axis arrangement", self.swap_axes_check)
        expression_form.addRow("", self.fresh_dendrogram_label)
        self.options_stack.addWidget(expression_page)

        composition_page = QWidget()
        composition_form = QFormLayout(composition_page)
        self.composition_obs_combo = QComboBox()
        self.composition_measure_combo = QComboBox()
        self.composition_measure_combo.addItem(
            "Percentage within each sample", "percent"
        )
        self.composition_measure_combo.addItem("Cell count", "count")
        composition_help = QLabel(
            "Use ROI, patient, condition, or another observation to compare how "
            "the selected populations are represented across samples."
        )
        composition_help.setWordWrap(True)
        composition_form.addRow("Samples / grouping", self.composition_obs_combo)
        composition_form.addRow("Values", self.composition_measure_combo)
        composition_form.addRow("", composition_help)
        self.options_stack.addWidget(composition_page)

        comparison_page = QWidget()
        comparison_form = QFormLayout(comparison_page)
        self.comparison_obs_combo = QComboBox()
        self.comparison_normalisation_combo = QComboBox()
        self.comparison_normalisation_combo.addItem(
            "Percentage within each original label", "row_percent"
        )
        self.comparison_normalisation_combo.addItem(
            "Percentage within each new label", "column_percent"
        )
        self.comparison_normalisation_combo.addItem("Cell count", "count")
        comparison_help = QLabel(
            "Rows are the comparison/original labels and columns are the primary "
            "labels selected in box 1. This makes merges and splits visible."
        )
        comparison_help.setWordWrap(True)
        comparison_form.addRow(
            "Original / comparison labels", self.comparison_obs_combo
        )
        comparison_form.addRow("Display", self.comparison_normalisation_combo)
        comparison_form.addRow("", comparison_help)
        self.options_stack.addWidget(comparison_page)
        layout.addWidget(options_group)

        generate_group = group_factory(
            "4. Generate and manage plots",
            "scanpy_plotting",
            "Generate and manage plots",
        )
        generate_layout = QVBoxLayout(generate_group)
        self.readiness_label = QLabel("● Load AnnData to begin.")
        self.readiness_label.setWordWrap(True)
        generate_actions = QHBoxLayout()
        self.generate_button = QPushButton("Open in a new resizable window")
        self.generate_button.setObjectName("sbtPrimaryActionButton")
        self.refresh_button = QPushButton("Refresh choices from live AnnData")
        generate_actions.addWidget(self.generate_button)
        generate_actions.addWidget(self.refresh_button)
        generate_actions.addStretch(1)
        self.open_windows_tree = QTreeWidget()
        self.open_windows_tree.setHeaderLabels(["Open plot", "Created from", "State"])
        self.open_windows_tree.setRootIsDecorated(False)
        self.open_windows_tree.setMaximumHeight(160)
        window_actions = QHBoxLayout()
        self.focus_window_button = QPushButton("Bring selected plot to front")
        self.close_window_button = QPushButton("Close selected plot")
        self.close_all_windows_button = QPushButton("Close all plots")
        window_actions.addWidget(self.focus_window_button)
        window_actions.addWidget(self.close_window_button)
        window_actions.addWidget(self.close_all_windows_button)
        window_actions.addStretch(1)
        generate_layout.addWidget(self.readiness_label)
        generate_layout.addLayout(generate_actions)
        generate_layout.addWidget(self.open_windows_tree)
        generate_layout.addLayout(window_actions)
        layout.addWidget(generate_group)
        layout.addStretch(1)

        self.generate_button.clicked.connect(generate_callback)
        self.refresh_button.clicked.connect(refresh_callback)
        self.focus_window_button.clicked.connect(
            lambda: focus_callback(self.selected_window_id() or "")
        )
        self.close_window_button.clicked.connect(
            lambda: close_callback(self.selected_window_id() or "")
        )
        self.close_all_windows_button.clicked.connect(close_all_callback)
        self.groupby_combo.currentTextChanged.connect(self._groupby_changed)
        self.scope_combo.currentIndexChanged.connect(self._controls_changed)
        self.group_values_list.itemSelectionChanged.connect(self._controls_changed)
        self.roi_list.itemSelectionChanged.connect(self._controls_changed)
        self.matrix_source_combo.currentTextChanged.connect(self._matrix_source_changed)
        self.plot_type_combo.currentIndexChanged.connect(self._plot_type_changed)
        self.embedding_combo.currentTextChanged.connect(self._embedding_changed)
        self.marker_search_edit.textChanged.connect(self._filter_markers)
        self.marker_list.itemSelectionChanged.connect(self._controls_changed)
        self.select_visible_markers_button.clicked.connect(self._select_visible_markers)
        self.clear_markers_button.clicked.connect(self.marker_list.clearSelection)
        self.select_all_groups_button.clicked.connect(self.group_values_list.selectAll)
        self.clear_groups_button.clicked.connect(self.group_values_list.clearSelection)
        self.clear_rois_button.clicked.connect(self.roi_list.clearSelection)
        for control in (
            self.embedding_x_spin,
            self.embedding_y_spin,
            self.point_limit_spin,
            self.point_size_spin,
            self.point_alpha_spin,
            self.centroid_labels_check,
            self.expression_scale_combo,
            self.positivity_spin,
            self.expression_colormap_combo,
            self.side_annotation_combo,
            self.totals_sort_combo,
            self.dendrogram_correlation_combo,
            self.dendrogram_linkage_combo,
            self.dendrogram_optimal_ordering_check,
            self.swap_axes_check,
            self.composition_obs_combo,
            self.composition_measure_combo,
            self.comparison_obs_combo,
            self.comparison_normalisation_combo,
        ):
            signal = getattr(control, "valueChanged", None)
            if signal is None:
                signal = getattr(control, "currentIndexChanged", None)
            if signal is None:
                signal = getattr(control, "toggled")
            signal.connect(self._controls_changed)
        self.side_annotation_combo.currentIndexChanged.connect(
            self._expression_annotation_changed
        )
        self._plot_type_changed()

    @staticmethod
    def _selected_text(widget) -> list[str]:
        return [item.text() for item in widget.selectedItems()]

    def _preferred_groupby(self, columns: list[str], requested: str | None) -> str:
        if requested in columns:
            return str(requested)
        current = self.groupby_combo.currentText()
        if current in columns:
            return current
        for token in ("population", "celltype", "cell_type", "leiden"):
            match = next(
                (column for column in columns if token in column.casefold()), None
            )
            if match:
                return match
        return columns[0] if columns else ""

    def refresh_from_anndata(
        self,
        adata,
        *,
        roi_obs: str | None,
        cohort_obs_names: set[str] | None,
        preferred_groupby: str | None = None,
    ) -> None:
        """Refresh choices while retaining selections that still exist."""

        self._adata = adata
        self._roi_obs = str(roi_obs) if roi_obs else None
        self._cohort_obs_names = cohort_obs_names
        columns = groupable_obs_columns(adata)
        target_groupby = self._preferred_groupby(columns, preferred_groupby)
        self.groupby_combo.blockSignals(True)
        self.groupby_combo.clear()
        self.groupby_combo.addItems(columns)
        if target_groupby:
            self.groupby_combo.setCurrentText(target_groupby)
        self.groupby_combo.blockSignals(False)

        cohort_index = self.scope_combo.findData("cohort")
        cohort_item = self.scope_combo.model().item(cohort_index)
        if cohort_item is not None:
            cohort_item.setEnabled(cohort_obs_names is not None)
        if self.scope_combo.currentData() == "cohort" and cohort_obs_names is None:
            self.scope_combo.setCurrentIndex(self.scope_combo.findData("all_cells"))

        current_source = self.matrix_source_combo.currentText()
        sources = matrix_source_choices(adata)
        self.matrix_source_combo.blockSignals(True)
        self.matrix_source_combo.clear()
        self.matrix_source_combo.addItems(sources)
        if current_source in sources:
            self.matrix_source_combo.setCurrentText(current_source)
        self.matrix_source_combo.blockSignals(False)

        current_embedding = self.embedding_combo.currentText()
        embeddings = [
            str(key)
            for key, value in adata.obsm.items()
            if getattr(value, "ndim", 0) == 2 and int(value.shape[1]) >= 2
        ]
        self.embedding_combo.blockSignals(True)
        self.embedding_combo.clear()
        self.embedding_combo.addItems(embeddings)
        if current_embedding in embeddings:
            self.embedding_combo.setCurrentText(current_embedding)
        elif "X_umap" in embeddings:
            self.embedding_combo.setCurrentText("X_umap")
        self.embedding_combo.blockSignals(False)

        current_composition = self.composition_obs_combo.currentText()
        current_comparison = self.comparison_obs_combo.currentText()
        for combo, current in (
            (self.composition_obs_combo, current_composition),
            (self.comparison_obs_combo, current_comparison),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(columns)
            if current in columns:
                combo.setCurrentText(current)
            combo.blockSignals(False)
        if self._roi_obs in columns and current_composition not in columns:
            self.composition_obs_combo.setCurrentText(self._roi_obs)
        comparison_default = next(
            (
                column
                for column in columns
                if column != target_groupby and "leiden" in column.casefold()
            ),
            next((column for column in columns if column != target_groupby), ""),
        )
        if current_comparison not in columns and comparison_default:
            self.comparison_obs_combo.setCurrentText(comparison_default)

        self._refresh_group_values()
        self._refresh_rois()
        self._refresh_markers()
        self._embedding_changed()
        self._controls_changed()

    def _refresh_group_values(self) -> None:
        selected = set(self._selected_text(self.group_values_list))
        self.group_values_list.clear()
        groupby = self.groupby_combo.currentText()
        if self._adata is None or groupby not in self._adata.obs:
            return
        self.group_values_list.addItems(ordered_obs_values(self._adata.obs[groupby]))
        for index in range(self.group_values_list.count()):
            item = self.group_values_list.item(index)
            item.setSelected(item.text() in selected)

    def _refresh_rois(self) -> None:
        selected = set(self._selected_text(self.roi_list))
        self.roi_list.clear()
        if (
            self._adata is None
            or not self._roi_obs
            or self._roi_obs not in self._adata.obs
        ):
            self.roi_list.setEnabled(False)
            self.clear_rois_button.setEnabled(False)
            return
        self.roi_list.setEnabled(True)
        self.clear_rois_button.setEnabled(True)
        self.roi_list.addItems(ordered_obs_values(self._adata.obs[self._roi_obs]))
        for index in range(self.roi_list.count()):
            item = self.roi_list.item(index)
            item.setSelected(item.text() in selected)

    def _refresh_markers(self) -> None:
        selected = set(self._selected_text(self.marker_list))
        self.marker_list.clear()
        if self._adata is None or not self.matrix_source_combo.currentText():
            return
        markers = matrix_source_var_names(
            self._adata, self.matrix_source_combo.currentText()
        ).tolist()
        self.marker_list.addItems([str(marker) for marker in markers])
        for index in range(self.marker_list.count()):
            item = self.marker_list.item(index)
            item.setSelected(item.text() in selected)
        self._filter_markers(self.marker_search_edit.text())

    def _groupby_changed(self, *_args) -> None:
        self._refresh_group_values()
        if self.comparison_obs_combo.currentText() == self.groupby_combo.currentText():
            for index in range(self.comparison_obs_combo.count()):
                if (
                    self.comparison_obs_combo.itemText(index)
                    != self.groupby_combo.currentText()
                ):
                    self.comparison_obs_combo.setCurrentIndex(index)
                    break
        self._controls_changed()

    def _matrix_source_changed(self, *_args) -> None:
        self._refresh_markers()
        self._controls_changed()

    def _embedding_changed(self, *_args) -> None:
        components = 2
        key = self.embedding_combo.currentText()
        if self._adata is not None and key in self._adata.obsm:
            components = int(self._adata.obsm[key].shape[1])
        self.embedding_x_spin.setMaximum(max(1, components))
        self.embedding_y_spin.setMaximum(max(1, components))
        if self.embedding_x_spin.value() == self.embedding_y_spin.value():
            self.embedding_y_spin.setValue(2 if components >= 2 else 1)
        self._controls_changed()

    def _plot_type_changed(self, *_args) -> None:
        plot_type = self.plot_type_combo.currentData()
        if plot_type == "embedding":
            page = 0
            description = (
                "Uses scanpy.pl.embedding to show the selected labels on an "
                "existing AnnData embedding; "
                "large views are stratified and downsampled for responsiveness."
            )
        elif plot_type in {"heatmap", "dotplot", "violin"}:
            page = 1
            description = {
                "heatmap": (
                    "Uses scanpy.pl.matrixplot to compare mean marker expression "
                    "between populations."
                ),
                "dotplot": (
                    "Uses scanpy.pl.dotplot: mean expression is shown by colour "
                    "and the fraction above the positivity threshold by dot size."
                ),
                "violin": (
                    "Uses scanpy.pl.stacked_violin to show marker-value "
                    "distributions within each population."
                ),
            }[plot_type]
        elif plot_type in {"composition_bar", "composition_heatmap"}:
            page = 2
            description = (
                "Compares population counts or percentages across ROIs, patients, "
                "conditions, or another sample grouping."
            )
        else:
            page = 3
            description = (
                "Cross-tabulates original and new labels so renames, merges, and "
                "subclusters are explicit."
            )
        self.options_stack.setCurrentIndex(page)
        self.expression_scale_combo.setEnabled(plot_type in {"heatmap", "dotplot"})
        self.expression_scale_combo.setToolTip(
            "Controls matrix and dot colours. Stacked violins always show the "
            "stored values from the selected expression matrix."
        )
        self.positivity_spin.setEnabled(plot_type == "dotplot")
        self._expression_annotation_changed()
        self.plot_description_label.setText(description)
        self._controls_changed()

    def _expression_annotation_changed(self, *_args) -> None:
        plot_type = self.plot_type_combo.currentData()
        expression_plot = plot_type in {"heatmap", "dotplot", "violin"}
        annotation = self.side_annotation_combo.currentData()
        dendrogram = expression_plot and annotation == "dendrogram"
        totals = expression_plot and annotation == "totals"
        for control in (
            self.dendrogram_correlation_combo,
            self.dendrogram_linkage_combo,
            self.dendrogram_optimal_ordering_check,
            self.fresh_dendrogram_label,
        ):
            control.setEnabled(dendrogram)
        self.totals_sort_combo.setEnabled(totals)

    def _filter_markers(self, text: str) -> None:
        needle = str(text).strip().casefold()
        for index in range(self.marker_list.count()):
            item = self.marker_list.item(index)
            item.setHidden(bool(needle and needle not in item.text().casefold()))

    def _select_visible_markers(self) -> None:
        for index in range(self.marker_list.count()):
            item = self.marker_list.item(index)
            if not item.isHidden():
                item.setSelected(True)

    def current_request(self) -> ScanpyPlotRequest:
        """Return the validated controls as a portable request."""

        return ScanpyPlotRequest(
            plot_type=str(self.plot_type_combo.currentData()),
            groupby=self.groupby_combo.currentText(),
            cell_scope=str(self.scope_combo.currentData()),
            selected_groups=self._selected_text(self.group_values_list),
            roi_obs=self._roi_obs,
            selected_rois=self._selected_text(self.roi_list),
            matrix_source=self.matrix_source_combo.currentText() or "X",
            markers=self._selected_text(self.marker_list),
            expression_scale=str(self.expression_scale_combo.currentData()),
            positivity_threshold=float(self.positivity_spin.value()),
            expression_colormap=str(self.expression_colormap_combo.currentData()),
            side_annotation=str(self.side_annotation_combo.currentData()),
            totals_sort=str(self.totals_sort_combo.currentData()),
            dendrogram_correlation=str(self.dendrogram_correlation_combo.currentData()),
            dendrogram_linkage=str(self.dendrogram_linkage_combo.currentData()),
            dendrogram_optimal_ordering=bool(
                self.dendrogram_optimal_ordering_check.isChecked()
            ),
            swap_axes=bool(self.swap_axes_check.isChecked()),
            embedding_key=self.embedding_combo.currentText() or None,
            x_component=int(self.embedding_x_spin.value()),
            y_component=int(self.embedding_y_spin.value()),
            point_limit=int(self.point_limit_spin.value()),
            point_size=float(self.point_size_spin.value()),
            point_alpha=float(self.point_alpha_spin.value()),
            label_centroids=bool(self.centroid_labels_check.isChecked()),
            composition_obs=self.composition_obs_combo.currentText() or None,
            composition_measure=str(self.composition_measure_combo.currentData()),
            comparison_obs=self.comparison_obs_combo.currentText() or None,
            comparison_normalisation=str(
                self.comparison_normalisation_combo.currentData()
            ),
        )

    def _controls_changed(self, *_args) -> None:
        if self._adata is None:
            self._set_readiness(False, "Load AnnData to begin.")
            return
        try:
            request = self.current_request()
            mask = resolve_plot_cell_mask(
                self._adata,
                request,
                cohort_obs_names=self._cohort_obs_names,
            )
            if request.plot_type == "embedding" and not request.embedding_key:
                raise ValueError("Choose an existing embedding.")
            if request.plot_type in {"heatmap", "dotplot", "violin"}:
                if not request.markers:
                    raise ValueError("Select at least one marker.")
                maximum = 12 if request.plot_type == "violin" else 100
                if len(request.markers) > maximum:
                    raise ValueError(f"Select at most {maximum} markers for this plot.")
                if request.side_annotation == "dendrogram":
                    if len(request.markers) < 2:
                        raise ValueError(
                            "A fresh expression dendrogram requires at least two "
                            "markers."
                        )
                    represented_groups = int(
                        self._adata.obs[request.groupby]
                        .iloc[np.flatnonzero(mask)]
                        .astype("string")
                        .fillna("Unassigned")
                        .nunique()
                    )
                    if represented_groups < 3:
                        raise ValueError(
                            "A dendrogram requires at least three represented "
                            "populations in the selected cells."
                        )
            if request.plot_type.startswith("composition"):
                if not request.composition_obs:
                    raise ValueError("Choose a sample or ROI grouping.")
            if request.plot_type == "label_comparison":
                if not request.comparison_obs:
                    raise ValueError("Choose the original label column.")
                if request.comparison_obs == request.groupby:
                    raise ValueError("Choose two different label columns.")
        except Exception as error:  # noqa: BLE001 - readiness should remain usable
            self.data_summary_label.setText(str(error))
            self._set_readiness(False, str(error))
            return
        group_count = int(
            self._adata.obs[request.groupby].iloc[mask].nunique(dropna=True)
        )
        roi_text = (
            f" across {len(request.selected_rois)} selected ROIs"
            if request.selected_rois
            else " across all available ROIs"
        )
        annotation_text = ""
        if request.plot_type in {"heatmap", "dotplot", "violin"}:
            annotation_text = {
                "none": "",
                "dendrogram": "; a fresh dendrogram will be recalculated",
                "totals": "; population totals will be displayed",
            }[request.side_annotation]
        summary = (
            f"{int(mask.sum()):,} cells in {group_count:,} label groups{roi_text}; "
            f"expression source {request.matrix_source}{annotation_text}."
        )
        self.data_summary_label.setText(summary)
        self._set_readiness(True, f"Ready — {summary}")

    def _set_readiness(self, ready: bool, text: str) -> None:
        self.generate_button.setEnabled(bool(ready))
        colour = "#166534" if ready else "#991b1b"
        background = "#dcfce7" if ready else "#fee2e2"
        self.readiness_label.setText(("● " if ready else "● Not ready — ") + text)
        self.readiness_label.setStyleSheet(
            f"color: {colour}; background: {background}; border-radius: 5px; "
            "padding: 7px; font-weight: 700;"
        )

    def select_groupby(self, observation: str) -> bool:
        """Select a live observation and return whether it was available."""

        index = self.groupby_combo.findText(str(observation))
        if index < 0:
            return False
        self.groupby_combo.setCurrentIndex(index)
        return True

    def selected_window_id(self) -> str | None:
        items = self.open_windows_tree.selectedItems()
        if not items:
            return None
        return str(items[0].data(0, self.Qt.UserRole) or "") or None

    def add_window(self, window_id: str, title: str, summary: str) -> None:
        item = self.QTreeWidgetItem([str(title), str(summary), "Current"])
        item.setData(0, self.Qt.UserRole, str(window_id))
        self.open_windows_tree.addTopLevelItem(item)
        self.open_windows_tree.setCurrentItem(item)
        self._window_items[str(window_id)] = item

    def remove_window(self, window_id: str) -> None:
        item = self._window_items.pop(str(window_id), None)
        if item is None:
            return
        index = self.open_windows_tree.indexOfTopLevelItem(item)
        if index >= 0:
            self.open_windows_tree.takeTopLevelItem(index)

    def mark_windows_stale(self) -> None:
        for item in self._window_items.values():
            item.setText(2, "Labels/data may have changed")


__all__ = ["ScanpyPlottingPanel"]
