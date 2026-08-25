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
    sample_level_obs_columns,
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
        self.metadata_filter_combo = QComboBox()
        self.metadata_filter_values_list = QListWidget()
        self.metadata_filter_values_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.metadata_filter_values_list.setMaximumHeight(105)
        metadata_actions = QWidget()
        metadata_actions_layout = QHBoxLayout(metadata_actions)
        metadata_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.clear_metadata_filter_button = QPushButton("Clear metadata selection")
        metadata_actions_layout.addWidget(self.clear_metadata_filter_button)
        metadata_actions_layout.addStretch(1)
        self.matrix_source_combo = QComboBox()
        self.data_summary_label = QLabel("Load AnnData to configure plotting.")
        self.data_summary_label.setWordWrap(True)
        data_form.addRow("Labels / populations", self.groupby_combo)
        data_form.addRow("Cell scope", self.scope_combo)
        data_form.addRow("Populations", self.group_values_list)
        data_form.addRow("", group_actions)
        data_form.addRow("ROIs (none selected = all)", self.roi_list)
        data_form.addRow("", roi_actions)
        data_form.addRow(
            "Additional ROI/sample observation", self.metadata_filter_combo
        )
        data_form.addRow(
            "Values (none selected = all)", self.metadata_filter_values_list
        )
        data_form.addRow("", metadata_actions)
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
        self.embedding_colour_mode_combo = QComboBox()
        self.embedding_colour_mode_combo.addItem("Population labels", "labels")
        self.embedding_colour_mode_combo.addItem("Expression variables", "expression")
        self.embedding_marker_search_edit = QLineEdit()
        self.embedding_marker_search_edit.setPlaceholderText(
            "Type to filter expression variables…"
        )
        self.embedding_marker_list = QListWidget()
        self.embedding_marker_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.embedding_marker_list.setMaximumHeight(150)
        embedding_marker_actions = QWidget()
        embedding_marker_actions_layout = QHBoxLayout(embedding_marker_actions)
        embedding_marker_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.select_visible_embedding_markers_button = QPushButton("Select visible")
        self.select_feature_embedding_markers_button = QPushButton(
            "Select feature markers"
        )
        self.select_feature_embedding_markers_button.setToolTip(
            "Select expression variables that contributed channel-derived "
            "features to the active NapariSBT feature table."
        )
        self.clear_embedding_markers_button = QPushButton("Clear")
        embedding_marker_actions_layout.addWidget(
            self.select_visible_embedding_markers_button
        )
        embedding_marker_actions_layout.addWidget(
            self.select_feature_embedding_markers_button
        )
        embedding_marker_actions_layout.addWidget(self.clear_embedding_markers_button)
        embedding_marker_actions_layout.addStretch(1)
        self.embedding_ncols_spin = QSpinBox()
        self.embedding_ncols_spin.setRange(1, 8)
        self.embedding_ncols_spin.setValue(3)
        self.embedding_colormap_combo = QComboBox()
        for label, value in (
            ("Viridis", "viridis"),
            ("Blue", "Blues"),
            ("Red", "Reds"),
            ("Magma", "magma"),
            ("Plasma", "plasma"),
            ("Diverging blue–red", "coolwarm"),
        ):
            self.embedding_colormap_combo.addItem(label, value)
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
        embedding_form.addRow("Embedding", self.embedding_combo)
        embedding_form.addRow("Colour by", self.embedding_colour_mode_combo)
        embedding_form.addRow(
            "Find expression variable", self.embedding_marker_search_edit
        )
        embedding_form.addRow("Expression variables", self.embedding_marker_list)
        embedding_form.addRow("", embedding_marker_actions)
        embedding_form.addRow("Expression panel columns", self.embedding_ncols_spin)
        embedding_form.addRow("Expression colour map", self.embedding_colormap_combo)
        embedding_form.addRow("Horizontal component", self.embedding_x_spin)
        embedding_form.addRow("Vertical component", self.embedding_y_spin)
        embedding_form.addRow("Maximum displayed points", self.point_limit_spin)
        embedding_form.addRow("Point size", self.point_size_spin)
        embedding_form.addRow("Point opacity", self.point_alpha_spin)
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
        self.select_feature_markers_button = QPushButton("Select feature markers")
        self.select_feature_markers_button.setToolTip(
            "Select markers that contributed channel-derived features to the "
            "active NapariSBT feature table."
        )
        self.clear_markers_button = QPushButton("Clear")
        marker_actions_layout.addWidget(self.select_visible_markers_button)
        marker_actions_layout.addWidget(self.select_feature_markers_button)
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
        self.reorder_markers_check = QCheckBox(
            "Cluster selected markers by expression similarity"
        )
        self.reorder_markers_check.setToolTip(
            "Uses SpatialBiologyToolkit.utils.reorder_vars_by_expression on the "
            "currently selected cells, markers, and expression matrix."
        )
        self.population_colour_strip_check = QCheckBox(
            "Show colours from the population labels"
        )
        self.population_colour_strip_check.setToolTip(
            "Adds a narrow colour strip beside the population axis using the "
            "current AnnData palette."
        )
        self.population_colour_gap_spin = QSpinBox()
        self.population_colour_gap_spin.setRange(0, 100)
        self.population_colour_gap_spin.setValue(25)
        self.population_colour_gap_spin.setSuffix(" pt")
        self.population_colour_gap_spin.setToolTip(
            "Sets the space between population names and the colour strip."
        )
        self.population_colour_box_width_spin = QSpinBox()
        self.population_colour_box_width_spin.setRange(1, 50)
        self.population_colour_box_width_spin.setValue(10)
        self.population_colour_box_width_spin.setSuffix(" pt")
        self.population_colour_box_width_spin.setToolTip(
            "Sets the thickness of each population-colour box."
        )
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
        expression_form.addRow("Marker ordering", self.reorder_markers_check)
        expression_form.addRow("Population colours", self.population_colour_strip_check)
        expression_form.addRow(
            "Colour box width", self.population_colour_box_width_spin
        )
        expression_form.addRow("Colour/label gap", self.population_colour_gap_spin)
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
        self.bar_width_spin = QDoubleSpinBox()
        self.bar_width_spin.setRange(0.05, 1.0)
        self.bar_width_spin.setSingleStep(0.05)
        self.bar_width_spin.setValue(0.9)
        self.bar_start_padding_spin = QDoubleSpinBox()
        self.bar_start_padding_spin.setRange(0.0, 10.0)
        self.bar_start_padding_spin.setSingleStep(0.1)
        self.bar_start_padding_spin.setValue(0.25)
        self.bar_start_padding_spin.setSuffix(" bar units")
        self.bar_end_padding_spin = QDoubleSpinBox()
        self.bar_end_padding_spin.setRange(0.0, 10.0)
        self.bar_end_padding_spin.setSingleStep(0.1)
        self.bar_end_padding_spin.setValue(0.25)
        self.bar_end_padding_spin.setSuffix(" bar units")
        self.bar_sort_population_combo = QComboBox()
        self.bar_sort_population_combo.addItem("Do not sort", "")
        self.bar_sort_direction_combo = QComboBox()
        self.bar_sort_direction_combo.addItem("No sorting", "none")
        self.bar_sort_direction_combo.addItem("Smallest first", "ascending")
        self.bar_sort_direction_combo.addItem("Largest first", "descending")
        self.bar_manual_y_limits_check = QCheckBox("Set manually")
        self.bar_y_min_spin = QDoubleSpinBox()
        self.bar_y_min_spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.bar_y_min_spin.setDecimals(3)
        self.bar_y_min_spin.setValue(0.0)
        self.bar_y_max_spin = QDoubleSpinBox()
        self.bar_y_max_spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.bar_y_max_spin.setDecimals(3)
        self.bar_y_max_spin.setValue(100.0)
        composition_help = QLabel(
            "Use ROI, patient, condition, or another observation to compare how "
            "the selected populations are represented across samples."
        )
        composition_help.setWordWrap(True)
        composition_form.addRow("Samples / grouping", self.composition_obs_combo)
        composition_form.addRow("Values", self.composition_measure_combo)
        composition_form.addRow("Bar width", self.bar_width_spin)
        composition_form.addRow("Space before first bar", self.bar_start_padding_spin)
        composition_form.addRow("Space after last bar", self.bar_end_padding_spin)
        composition_form.addRow(
            "Sort bars by population", self.bar_sort_population_combo
        )
        composition_form.addRow("Bar sorting", self.bar_sort_direction_combo)
        composition_form.addRow("Y-axis limits", self.bar_manual_y_limits_check)
        composition_form.addRow("Y-axis minimum", self.bar_y_min_spin)
        composition_form.addRow("Y-axis maximum", self.bar_y_max_spin)
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

        common_options = QWidget()
        common_form = QFormLayout(common_options)
        common_form.setContentsMargins(0, 8, 0, 0)
        self.show_legend_check = QCheckBox("Show legend or colour scale")
        self.show_legend_check.setChecked(True)
        self.legend_location_combo = QComboBox()
        for label, value in (
            ("Right margin", "right margin"),
            ("On the data", "on data"),
            ("Automatic / best", "best"),
            ("Upper right", "upper right"),
            ("Upper left", "upper left"),
            ("Lower right", "lower right"),
            ("Lower left", "lower left"),
        ):
            self.legend_location_combo.addItem(label, value)
        self.legend_location_combo.setToolTip(
            "Position applies to categorical embedding and stacked-bar legends. "
            "Scanpy expression colour scales retain their native right-hand "
            "position, but can be hidden."
        )
        axis_labels = QWidget()
        axis_labels_layout = QHBoxLayout(axis_labels)
        axis_labels_layout.setContentsMargins(0, 0, 0, 0)
        self.show_x_axis_label_check = QCheckBox("X label")
        self.show_y_axis_label_check = QCheckBox("Y label")
        self.show_x_axis_label_check.setChecked(True)
        self.show_y_axis_label_check.setChecked(True)
        axis_labels_layout.addWidget(self.show_x_axis_label_check)
        axis_labels_layout.addWidget(self.show_y_axis_label_check)
        axis_labels_layout.addStretch(1)
        axis_ticks = QWidget()
        axis_ticks_layout = QHBoxLayout(axis_ticks)
        axis_ticks_layout.setContentsMargins(0, 0, 0, 0)
        self.show_x_ticks_check = QCheckBox("X ticks")
        self.show_y_ticks_check = QCheckBox("Y ticks")
        self.show_x_ticks_check.setChecked(True)
        self.show_y_ticks_check.setChecked(True)
        axis_ticks_layout.addWidget(self.show_x_ticks_check)
        axis_ticks_layout.addWidget(self.show_y_ticks_check)
        axis_ticks_layout.addStretch(1)
        self.title_mode_combo = QComboBox()
        self.title_mode_combo.addItem("Automatic", "automatic")
        self.title_mode_combo.addItem("Custom", "custom")
        self.title_mode_combo.addItem("Hidden", "hidden")
        self.custom_title_edit = QLineEdit()
        self.custom_title_edit.setPlaceholderText("Enter the plot title…")
        self.heatmap_colormap_combo = QComboBox()
        self.heatmap_colormap_combo.setEditable(True)
        for label, value in (
            ("Viridis", "viridis"),
            ("Magma", "magma"),
            ("Plasma", "plasma"),
            ("Blue", "Blues"),
            ("Red", "Reds"),
            ("Diverging blue–red", "coolwarm"),
        ):
            self.heatmap_colormap_combo.addItem(label, value)
        self.heatmap_population_colours_check = QCheckBox(
            "Show population colour strip"
        )
        self.edge_colour_combo = QComboBox()
        self.edge_colour_combo.setEditable(True)
        for label, value in (
            ("White", "#ffffff"),
            ("Black", "#000000"),
            ("Dark grey", "#374151"),
            ("Light grey", "#d1d5db"),
        ):
            self.edge_colour_combo.addItem(label, value)
        self.edge_width_spin = QDoubleSpinBox()
        self.edge_width_spin.setRange(0.0, 10.0)
        self.edge_width_spin.setSingleStep(0.1)
        self.edge_width_spin.setValue(0.0)
        self.edge_width_spin.setSuffix(" pt")
        common_form.addRow("Legend", self.show_legend_check)
        common_form.addRow("Legend position", self.legend_location_combo)
        common_form.addRow("Axis labels", axis_labels)
        common_form.addRow("Axis ticks", axis_ticks)
        common_form.addRow("Title", self.title_mode_combo)
        common_form.addRow("Custom title", self.custom_title_edit)
        common_form.addRow("Heatmap colour map", self.heatmap_colormap_combo)
        common_form.addRow(
            "Heatmap population colours", self.heatmap_population_colours_check
        )
        common_form.addRow("Cell/bar edge colour", self.edge_colour_combo)
        common_form.addRow("Cell/bar edge width", self.edge_width_spin)
        options_layout.addWidget(common_options)
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
        self.metadata_filter_combo.currentIndexChanged.connect(
            self._metadata_filter_obs_changed
        )
        self.metadata_filter_values_list.itemSelectionChanged.connect(
            self._controls_changed
        )
        self.clear_metadata_filter_button.clicked.connect(
            self.metadata_filter_values_list.clearSelection
        )
        self.matrix_source_combo.currentTextChanged.connect(self._matrix_source_changed)
        self.plot_type_combo.currentIndexChanged.connect(self._plot_type_changed)
        self.embedding_combo.currentTextChanged.connect(self._embedding_changed)
        self.embedding_colour_mode_combo.currentIndexChanged.connect(
            self._embedding_colour_mode_changed
        )
        self.embedding_marker_search_edit.textChanged.connect(
            self._filter_embedding_markers
        )
        self.embedding_marker_list.itemSelectionChanged.connect(self._controls_changed)
        self.select_visible_embedding_markers_button.clicked.connect(
            self._select_visible_embedding_markers
        )
        self.clear_embedding_markers_button.clicked.connect(
            self.embedding_marker_list.clearSelection
        )
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
            self.embedding_ncols_spin,
            self.embedding_colormap_combo,
            self.expression_scale_combo,
            self.positivity_spin,
            self.expression_colormap_combo,
            self.side_annotation_combo,
            self.totals_sort_combo,
            self.dendrogram_correlation_combo,
            self.dendrogram_linkage_combo,
            self.dendrogram_optimal_ordering_check,
            self.reorder_markers_check,
            self.population_colour_strip_check,
            self.population_colour_box_width_spin,
            self.population_colour_gap_spin,
            self.swap_axes_check,
            self.composition_obs_combo,
            self.composition_measure_combo,
            self.bar_width_spin,
            self.bar_start_padding_spin,
            self.bar_end_padding_spin,
            self.bar_sort_population_combo,
            self.bar_sort_direction_combo,
            self.bar_manual_y_limits_check,
            self.bar_y_min_spin,
            self.bar_y_max_spin,
            self.comparison_obs_combo,
            self.comparison_normalisation_combo,
            self.show_legend_check,
            self.legend_location_combo,
            self.show_x_axis_label_check,
            self.show_y_axis_label_check,
            self.show_x_ticks_check,
            self.show_y_ticks_check,
            self.title_mode_combo,
            self.heatmap_colormap_combo,
            self.heatmap_population_colours_check,
            self.edge_colour_combo,
            self.edge_width_spin,
        ):
            signal = getattr(control, "valueChanged", None)
            if signal is None:
                signal = getattr(control, "currentIndexChanged", None)
            if signal is None:
                signal = control.toggled
            signal.connect(self._controls_changed)
        self.custom_title_edit.textChanged.connect(self._controls_changed)
        self.heatmap_colormap_combo.editTextChanged.connect(self._controls_changed)
        self.edge_colour_combo.editTextChanged.connect(self._controls_changed)
        self.title_mode_combo.currentIndexChanged.connect(
            self._common_plot_options_changed
        )
        self.bar_manual_y_limits_check.toggled.connect(
            self._common_plot_options_changed
        )
        self.side_annotation_combo.currentIndexChanged.connect(
            self._expression_annotation_changed
        )
        self.population_colour_strip_check.toggled.connect(
            self._expression_annotation_changed
        )
        self.show_legend_check.toggled.connect(self._legend_options_changed)
        self._plot_type_changed()

    @staticmethod
    def _selected_text(widget) -> list[str]:
        return [item.text() for item in widget.selectedItems()]

    @staticmethod
    def _editable_combo_value(widget) -> str:
        """Return preset data or genuinely custom editable-combo text."""

        index = widget.currentIndex()
        text = widget.currentText().strip()
        if index >= 0 and text == widget.itemText(index):
            return str(widget.itemData(index) or text)
        return text

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

        current_filter = self.metadata_filter_combo.currentData()
        sample_columns = sample_level_obs_columns(adata, roi_obs=self._roi_obs)
        self.metadata_filter_combo.blockSignals(True)
        self.metadata_filter_combo.clear()
        self.metadata_filter_combo.addItem("No additional filter", "")
        for column in sample_columns:
            self.metadata_filter_combo.addItem(column, column)
        filter_index = self.metadata_filter_combo.findData(current_filter)
        self.metadata_filter_combo.setCurrentIndex(max(0, filter_index))
        self.metadata_filter_combo.blockSignals(False)

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
        self._refresh_bar_sort_populations()
        self._refresh_rois()
        self._metadata_filter_obs_changed()
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

    def _refresh_bar_sort_populations(self) -> None:
        current = self.bar_sort_population_combo.currentData()
        self.bar_sort_population_combo.blockSignals(True)
        self.bar_sort_population_combo.clear()
        self.bar_sort_population_combo.addItem("Do not sort", "")
        if (
            self._adata is not None
            and self.groupby_combo.currentText() in self._adata.obs
        ):
            for value in ordered_obs_values(
                self._adata.obs[self.groupby_combo.currentText()]
            ):
                self.bar_sort_population_combo.addItem(value, value)
        index = self.bar_sort_population_combo.findData(current)
        self.bar_sort_population_combo.setCurrentIndex(max(0, index))
        self.bar_sort_population_combo.blockSignals(False)

    def _metadata_filter_obs_changed(self, *_args) -> None:
        selected = set(self._selected_text(self.metadata_filter_values_list))
        self.metadata_filter_values_list.clear()
        observation = str(self.metadata_filter_combo.currentData() or "")
        enabled = bool(
            self._adata is not None and observation and observation in self._adata.obs
        )
        self.metadata_filter_values_list.setEnabled(enabled)
        self.clear_metadata_filter_button.setEnabled(enabled)
        if enabled:
            self.metadata_filter_values_list.addItems(
                ordered_obs_values(self._adata.obs[observation], include_missing=True)
            )
            for index in range(self.metadata_filter_values_list.count()):
                item = self.metadata_filter_values_list.item(index)
                item.setSelected(item.text() in selected)
        self._controls_changed()

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
        selected_embedding = set(self._selected_text(self.embedding_marker_list))
        self.marker_list.clear()
        self.embedding_marker_list.clear()
        if self._adata is None or not self.matrix_source_combo.currentText():
            return
        markers = matrix_source_var_names(
            self._adata, self.matrix_source_combo.currentText()
        ).tolist()
        marker_names = [str(marker) for marker in markers]
        self.marker_list.addItems(marker_names)
        self.embedding_marker_list.addItems(marker_names)
        for index in range(self.marker_list.count()):
            item = self.marker_list.item(index)
            item.setSelected(item.text() in selected)
        for index in range(self.embedding_marker_list.count()):
            item = self.embedding_marker_list.item(index)
            item.setSelected(item.text() in selected_embedding)
        self._filter_markers(self.marker_search_edit.text())
        self._filter_embedding_markers(self.embedding_marker_search_edit.text())

    def _groupby_changed(self, *_args) -> None:
        self._refresh_group_values()
        self._refresh_bar_sort_populations()
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
        self._embedding_colour_mode_changed()
        self._legend_options_changed()
        self._common_plot_options_changed()
        self.plot_description_label.setText(description)
        self._controls_changed()

    def _common_plot_options_changed(self, *_args) -> None:
        plot_type = self.plot_type_combo.currentData()
        heatmap = plot_type in {"heatmap", "composition_heatmap", "label_comparison"}
        bar_plot = plot_type == "composition_bar"
        self.custom_title_edit.setEnabled(
            self.title_mode_combo.currentData() == "custom"
        )
        self.heatmap_colormap_combo.setEnabled(
            plot_type in {"composition_heatmap", "label_comparison"}
        )
        self.heatmap_population_colours_check.setEnabled(
            plot_type in {"composition_heatmap", "label_comparison"}
        )
        self.edge_colour_combo.setEnabled(heatmap or bar_plot)
        self.edge_width_spin.setEnabled(heatmap or bar_plot)
        for control in (
            self.bar_width_spin,
            self.bar_start_padding_spin,
            self.bar_end_padding_spin,
            self.bar_sort_population_combo,
            self.bar_sort_direction_combo,
            self.bar_manual_y_limits_check,
        ):
            control.setEnabled(bar_plot)
        manual_y_limits = bar_plot and self.bar_manual_y_limits_check.isChecked()
        self.bar_y_min_spin.setEnabled(manual_y_limits)
        self.bar_y_max_spin.setEnabled(manual_y_limits)
        self._controls_changed()

    def _embedding_colour_mode_changed(self, *_args) -> None:
        expression_mode = (
            self.plot_type_combo.currentData() == "embedding"
            and self.embedding_colour_mode_combo.currentData() == "expression"
        )
        for control in (
            self.embedding_marker_search_edit,
            self.embedding_marker_list,
            self.select_visible_embedding_markers_button,
            self.select_feature_embedding_markers_button,
            self.clear_embedding_markers_button,
            self.embedding_ncols_spin,
            self.embedding_colormap_combo,
        ):
            control.setEnabled(expression_mode)
        self._legend_options_changed()
        self._controls_changed()

    def _legend_options_changed(self, *_args) -> None:
        plot_type = self.plot_type_combo.currentData()
        location_supported = plot_type == "composition_bar" or (
            plot_type == "embedding"
            and self.embedding_colour_mode_combo.currentData() == "labels"
        )
        self.legend_location_combo.setEnabled(
            self.show_legend_check.isChecked() and location_supported
        )

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
        self.reorder_markers_check.setEnabled(expression_plot)
        self.population_colour_strip_check.setEnabled(expression_plot)
        self.population_colour_gap_spin.setEnabled(
            expression_plot and self.population_colour_strip_check.isChecked()
        )
        self.population_colour_box_width_spin.setEnabled(
            expression_plot and self.population_colour_strip_check.isChecked()
        )

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

    def _filter_embedding_markers(self, text: str) -> None:
        needle = str(text).strip().casefold()
        for index in range(self.embedding_marker_list.count()):
            item = self.embedding_marker_list.item(index)
            item.setHidden(bool(needle and needle not in item.text().casefold()))

    def _select_visible_embedding_markers(self) -> None:
        for index in range(self.embedding_marker_list.count()):
            item = self.embedding_marker_list.item(index)
            if not item.isHidden():
                item.setSelected(True)

    def current_request(self) -> ScanpyPlotRequest:
        """Return the validated controls as a portable request."""

        plot_type = str(self.plot_type_combo.currentData())
        embedding_expression = (
            self.embedding_colour_mode_combo.currentData() == "expression"
        )
        show_population_colours = (
            self.population_colour_strip_check.isChecked()
            if plot_type in {"heatmap", "dotplot", "violin"}
            else self.heatmap_population_colours_check.isChecked()
            if plot_type in {"composition_heatmap", "label_comparison"}
            else False
        )
        return ScanpyPlotRequest(
            plot_type=plot_type,
            groupby=self.groupby_combo.currentText(),
            cell_scope=str(self.scope_combo.currentData()),
            selected_groups=self._selected_text(self.group_values_list),
            roi_obs=self._roi_obs,
            selected_rois=self._selected_text(self.roi_list),
            metadata_filter_obs=(
                str(self.metadata_filter_combo.currentData())
                if self.metadata_filter_combo.currentData()
                else None
            ),
            metadata_filter_values=self._selected_text(
                self.metadata_filter_values_list
            ),
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
            reorder_markers_by_expression=bool(self.reorder_markers_check.isChecked()),
            show_population_colours=bool(show_population_colours),
            population_colour_label_gap=float(self.population_colour_gap_spin.value()),
            population_colour_box_width=float(
                self.population_colour_box_width_spin.value()
            ),
            swap_axes=bool(self.swap_axes_check.isChecked()),
            embedding_key=self.embedding_combo.currentText() or None,
            embedding_markers=(
                self._selected_text(self.embedding_marker_list)
                if embedding_expression
                else []
            ),
            embedding_ncols=int(self.embedding_ncols_spin.value()),
            embedding_colormap=str(self.embedding_colormap_combo.currentData()),
            x_component=int(self.embedding_x_spin.value()),
            y_component=int(self.embedding_y_spin.value()),
            point_limit=int(self.point_limit_spin.value()),
            point_size=float(self.point_size_spin.value()),
            point_alpha=float(self.point_alpha_spin.value()),
            label_centroids=False,
            show_legend=bool(self.show_legend_check.isChecked()),
            legend_location=str(self.legend_location_combo.currentData()),
            show_x_axis_label=bool(self.show_x_axis_label_check.isChecked()),
            show_y_axis_label=bool(self.show_y_axis_label_check.isChecked()),
            show_x_ticks=bool(self.show_x_ticks_check.isChecked()),
            show_y_ticks=bool(self.show_y_ticks_check.isChecked()),
            title_mode=str(self.title_mode_combo.currentData()),
            custom_title=self.custom_title_edit.text(),
            heatmap_colormap=self._editable_combo_value(self.heatmap_colormap_combo),
            edge_color=self._editable_combo_value(self.edge_colour_combo),
            edge_width=float(self.edge_width_spin.value()),
            composition_obs=self.composition_obs_combo.currentText() or None,
            composition_measure=str(self.composition_measure_combo.currentData()),
            bar_width=float(self.bar_width_spin.value()),
            bar_start_padding=float(self.bar_start_padding_spin.value()),
            bar_end_padding=float(self.bar_end_padding_spin.value()),
            bar_sort_population=(
                str(self.bar_sort_population_combo.currentData())
                if self.bar_sort_population_combo.currentData()
                else None
            ),
            bar_sort_direction=str(self.bar_sort_direction_combo.currentData()),
            bar_manual_y_limits=bool(self.bar_manual_y_limits_check.isChecked()),
            bar_y_min=float(self.bar_y_min_spin.value()),
            bar_y_max=float(self.bar_y_max_spin.value()),
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
            if (
                request.plot_type == "embedding"
                and self.embedding_colour_mode_combo.currentData() == "expression"
                and not request.embedding_markers
            ):
                raise ValueError(
                    "Select at least one expression variable for the embedding."
                )
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
            if (
                request.plot_type.startswith("composition")
                and not request.composition_obs
            ):
                raise ValueError("Choose a sample or ROI grouping.")
            if (
                request.plot_type == "composition_bar"
                and request.bar_sort_direction != "none"
                and not request.bar_sort_population
            ):
                raise ValueError("Choose a population to sort the bars by.")
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
        metadata_text = ""
        if request.metadata_filter_obs and request.metadata_filter_values:
            values = ", ".join(request.metadata_filter_values[:4])
            if len(request.metadata_filter_values) > 4:
                values += f", +{len(request.metadata_filter_values) - 4} more"
            metadata_text = f"; {request.metadata_filter_obs}: {values}"
        annotation_text = ""
        if request.plot_type == "embedding" and request.embedding_markers:
            annotation_text = (
                f"; {len(request.embedding_markers)} expression panel(s) in "
                f"{request.embedding_ncols} columns"
            )
        if request.plot_type in {"heatmap", "dotplot", "violin"}:
            annotation_text = {
                "none": "",
                "dendrogram": "; a fresh dendrogram will be recalculated",
                "totals": "; population totals will be displayed",
            }[request.side_annotation]
            if request.reorder_markers_by_expression:
                annotation_text += "; markers will be ordered by expression"
            if request.show_population_colours:
                annotation_text += (
                    "; population colours will be shown with a "
                    f"{request.population_colour_box_width:g} pt box width and "
                    f"{request.population_colour_label_gap:g} pt label gap"
                )
        summary = (
            f"{int(mask.sum()):,} cells in {group_count:,} label groups{roi_text}; "
            f"expression source {request.matrix_source}{metadata_text}"
            f"{annotation_text}."
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
