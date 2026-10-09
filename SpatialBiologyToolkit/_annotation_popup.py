"""Qt presentation for :mod:`SpatialBiologyToolkit.annotation`."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import LassoSelector, PolygonSelector
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .annotation import CohortFilter

# QtPy chooses the same binding as Napari before Matplotlib imports a backend.
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

# Keep modeless windows and a standalone QApplication alive until callers close
# them, even when a notebook does not keep the function's return value.
_WINDOWS: list[QDialog] = []
_APPLICATION = None


class AnnotationDialog(QDialog):
    def __init__(
        self,
        session,
        *,
        key_added,
        color,
        point_limit,
        layer=None,
        use_raw=False,
        parent=None,
        on_apply=None,
        before_apply=None,
    ):
        super().__init__(parent)
        self.session = session
        self.on_apply = on_apply
        self.before_apply = before_apply
        self.layer = layer
        self.use_raw = use_raw
        self.selector = None
        self.colorbar = None
        self.applied_key = None
        self.setWindowTitle("SBT — Annotate embedding regions")
        self.resize(1050, 950)
        available = self.screen().availableGeometry()
        self.resize(
            min(self.width(), int(available.width() * 0.95)),
            min(self.height(), int(available.height() * 0.95)),
        )
        self.point_limit = point_limit
        self._sample_display()

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.key_edit = QLineEdit(key_added)
        self.source_combo = QComboBox()
        self.source_combo.addItem("Unassigned (no parent)", None)
        for name in session.adata.obs:
            values = session.adata.obs[name]
            if isinstance(
                values.dtype, pd.CategoricalDtype
            ) or not pd.api.types.is_numeric_dtype(values):
                self.source_combo.addItem(str(name), name)
        if session.source_obs is not None:
            if self.source_combo.findData(session.source_obs) < 0:
                self.source_combo.addItem(session.source_obs, session.source_obs)
            self.source_combo.setCurrentIndex(
                self.source_combo.findData(session.source_obs)
            )
        self.color_combo = QComboBox()
        self.color_combo.addItem("Draft labels", None)
        for name in session.adata.obs:
            self.color_combo.addItem(f"obs: {name}", ("obs", name))
        matrix = session.adata.raw if use_raw else session.adata
        if matrix is None:
            raise ValueError("AnnData has no raw expression matrix.")
        if layer is not None and (use_raw or layer not in session.adata.layers):
            raise ValueError("Choose an existing layer or raw expression, not both.")
        for name in matrix.var_names:
            self.color_combo.addItem(f"marker: {name}", ("marker", name))
        if color is not None:
            kind = "obs" if color in session.adata.obs else "marker"
            # Qt's findData does not compare Python tuple payloads by value.
            index = next(
                (
                    i
                    for i in range(self.color_combo.count())
                    if self.color_combo.itemData(i) == (kind, color)
                ),
                -1,
            )
            if index < 0:
                raise ValueError(f"No observation or marker named {color!r}.")
            self.color_combo.setCurrentIndex(index)
        form.addRow("Output obs column", self.key_edit)
        parent_controls = QWidget()
        parent_layout = QHBoxLayout(parent_controls)
        parent_layout.setContentsMargins(0, 0, 0, 0)
        parent_layout.addWidget(self.source_combo, 1)
        self.parent_prefix_check = QCheckBox("Prefix new labels with parent")
        self.parent_prefix_check.setToolTip(
            "For example, assigning Activated to T cells creates T cells / Activated. "
            "Unannotated cells retain their parent labels either way."
        )
        self.parent_prefix_check.setEnabled(session.source_obs is not None)
        parent_layout.addWidget(self.parent_prefix_check)
        form.addRow("Inherit labels from", parent_controls)
        form.addRow("Colour by", self.color_combo)
        layout.addLayout(form)
        self._build_cohort_controls(layout)
        self.figure = Figure(figsize=(8, 5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(200)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel(f"{session.basis} [{session.components[0] + 1}]")
        self.ax.set_ylabel(f"{session.basis} [{session.components[1] + 1}]")
        xy = session.coordinates[self.display_positions]
        background_xy = session.coordinates[self.background_positions]
        self.background = self.ax.scatter(
            background_xy[:, 0],
            background_xy[:, 1],
            s=6,
            linewidths=0,
            color="#cccccc",
            alpha=0.55,
            zorder=1,
        )
        self.points = self.ax.scatter(xy[:, 0], xy[:, 1], s=6, linewidths=0, zorder=2)
        self.highlight = self.ax.scatter(
            [],
            [],
            s=20,
            facecolors="none",
            edgecolors="#111111",
            linewidths=0.7,
            zorder=3,
        )
        # Keep the complete embedding as spatial context when the cohort changes.
        finite_xy = session.coordinates[session.finite]
        self.ax.update_datalim(
            np.asarray([finite_xy.min(axis=0), finite_xy.max(axis=0)])
        )
        self.ax.autoscale_view()
        self.ax.margins(0.04)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        controls = QHBoxLayout()
        self.tool_combo = QComboBox()
        self.tool_combo.addItems(["Lasso", "Polygon", "Navigate"])
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Category name")
        self.assign_button = QPushButton("Assign label")
        self.undo_button = QPushButton("Undo assignment")
        self.clear_button = QPushButton("Clear selection")
        for widget in (
            self.tool_combo,
            self.label_edit,
            self.assign_button,
            self.undo_button,
            self.clear_button,
        ):
            controls.addWidget(widget)
        layout.addLayout(controls)
        hint = QLabel(
            "Lasso: drag around cells. Polygon: click vertices, then click the first to close. "
            "Esc clears the shape. Later assignments replace earlier labels in overlaps. "
            "Turn off toolbar pan/zoom to draw."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.count_label = QLabel()
        self.count_label.setWordWrap(True)
        layout.addWidget(self.count_label)
        self.status_label = QLabel(
            "Draft only — Apply writes labels to live AnnData; Save writes a new H5AD copy."
        )
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        actions = QHBoxLayout()
        self.overwrite_check = QCheckBox("Overwrite existing obs column")
        self.apply_button = QPushButton("Apply to AnnData")
        self.save_button = QPushButton("Save H5AD copy…")
        self.save_recipe_button = QPushButton("Save recipe…")
        self.save_recipe_button.setToolTip(
            "Save region shapes, labels and cohort filters for reuse without a window. "
            "Only assigned regions are applied when replaying the recipe."
        )
        self.close_button = QPushButton("Close")
        for widget in (
            self.overwrite_check,
            self.apply_button,
            self.save_button,
            self.save_recipe_button,
            self.close_button,
        ):
            actions.addWidget(widget)
        layout.addLayout(actions)
        # Enter in a label field should assign, never trigger Apply or Save.
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
        self.assign_button.clicked.connect(lambda: self._run(self.assign))
        self.label_edit.returnPressed.connect(lambda: self._run(self.assign))
        self.undo_button.clicked.connect(lambda: self._run(self.undo))
        self.clear_button.clicked.connect(self.clear_selection)
        self.tool_combo.currentIndexChanged.connect(self._set_selector)
        self.source_combo.currentIndexChanged.connect(self._reset_source)
        self.color_combo.currentIndexChanged.connect(lambda: self._run(self.redraw))
        self.apply_button.clicked.connect(lambda: self._run(self.apply))
        self.save_button.clicked.connect(lambda: self._run(self.save_copy))
        self.save_recipe_button.clicked.connect(lambda: self._run(self.save_recipe))
        self.close_button.clicked.connect(self.close)
        self.canvas.mpl_connect("key_press_event", self._key_pressed)
        self._set_selector()
        self.redraw()

    def _sample_display(self):
        """Reserve drawing capacity for the cohort so small populations stay visible."""
        foreground = np.flatnonzero(self.session.eligible)
        background = np.flatnonzero(self.session.finite & ~self.session.eligible)
        count = min(len(foreground), max(1, int(self.point_limit * 0.8)))
        context_count = min(len(background), self.point_limit - count)
        count = min(len(foreground), self.point_limit - context_count)
        rng = np.random.default_rng(0)
        self.display_positions = np.sort(rng.choice(foreground, count, replace=False))
        self.background_positions = np.sort(
            rng.choice(background, context_count, replace=False)
        )

    def _build_cohort_controls(self, layout):
        group = QGroupBox(
            "Cohort to annotate — all filters must match; other cells stay grey"
        )
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        columns = QHBoxLayout(group)
        editor = QVBoxLayout()
        fields = QHBoxLayout()
        self.filter_kind_combo = QComboBox()
        self.filter_kind_combo.addItems(["obs values", "obs range", "X range"])
        self.filter_field_combo = QComboBox()
        self.filter_field_combo.setMinimumContentsLength(12)
        self.filter_field_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        fields.addWidget(self.filter_kind_combo)
        fields.addWidget(self.filter_field_combo, 1)
        editor.addLayout(fields)
        self.filter_editors = QStackedWidget()
        self.filter_editors.setFixedHeight(70)
        self.filter_values_list = QListWidget()
        self.filter_values_list.setMaximumHeight(70)
        self.filter_values_list.setToolTip(
            "Tick one or more values; a cell can match any ticked value."
        )
        self.filter_editors.addWidget(self.filter_values_list)
        range_widget = QWidget()
        ranges = QHBoxLayout(range_widget)
        ranges.setContentsMargins(0, 0, 0, 0)
        self.filter_min_edit = QLineEdit()
        self.filter_min_edit.setPlaceholderText("Minimum (inclusive)")
        self.filter_max_edit = QLineEdit()
        self.filter_max_edit.setPlaceholderText("Maximum (inclusive)")
        ranges.addWidget(self.filter_min_edit)
        ranges.addWidget(self.filter_max_edit)
        self.filter_editors.addWidget(range_widget)
        editor.addWidget(self.filter_editors)
        self.add_filter_button = QPushButton("Add filter")
        editor.addWidget(self.add_filter_button)
        columns.addLayout(editor, 1)
        active = QVBoxLayout()
        self.filter_rules_list = QListWidget()
        self.filter_rules_list.setFixedHeight(105)
        self.filter_rules_list.setToolTip(
            "Filters combine with AND. Removing filters keeps existing draft annotations."
        )
        active.addWidget(self.filter_rules_list)
        actions = QHBoxLayout()
        self.remove_filter_button = QPushButton("Remove selected filter")
        self.clear_filters_button = QPushButton("Clear filters")
        actions.addWidget(self.remove_filter_button)
        actions.addWidget(self.clear_filters_button)
        active.addLayout(actions)
        columns.addLayout(active, 1)
        layout.addWidget(group)
        self.filter_kind_combo.currentIndexChanged.connect(self._filter_fields_changed)
        self.filter_field_combo.currentIndexChanged.connect(self._filter_values_changed)
        self.add_filter_button.clicked.connect(lambda: self._run(self._add_filter))
        self.remove_filter_button.clicked.connect(
            lambda: self._run(self._remove_filter)
        )
        self.clear_filters_button.clicked.connect(
            lambda: self._run(lambda: self._set_filters([]))
        )
        self._filter_fields_changed()
        if self.session.source_obs is not None:
            self.filter_field_combo.setCurrentText(self.session.source_obs)
        self._refresh_filter_rules()

    def _filter_fields_changed(self, *_args):
        kind = self.filter_kind_combo.currentText()
        self.filter_field_combo.blockSignals(True)
        self.filter_field_combo.clear()
        if kind == "X range":
            fields = list(self.session.adata.var_names)
        else:
            fields = [
                name
                for name in self.session.adata.obs
                if kind == "obs values"
                or pd.api.types.is_numeric_dtype(self.session.adata.obs[name])
            ]
        self.filter_field_combo.addItems([str(name) for name in fields])
        self.filter_field_combo.blockSignals(False)
        self.filter_editors.setCurrentIndex(0 if kind == "obs values" else 1)
        self._filter_values_changed()

    def _filter_values_changed(self, *_args):
        self.filter_values_list.clear()
        self.filter_min_edit.clear()
        self.filter_max_edit.clear()
        key = self.filter_field_combo.currentText()
        if self.filter_kind_combo.currentText() != "obs values" or not key:
            return
        for value in self.session.adata.obs[key].drop_duplicates().tolist():
            missing = pd.isna(value)
            item = QListWidgetItem("<missing>" if missing else str(value))
            item.setData(Qt.UserRole, None if missing else value)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.filter_values_list.addItem(item)

    def _refresh_filter_rules(self):
        self.filter_rules_list.clear()
        for rule in self.session.filters:
            item = QListWidgetItem(rule.describe())
            item.setToolTip(rule.describe())
            self.filter_rules_list.addItem(item)

    def _add_filter(self):
        key = self.filter_field_combo.currentText()
        if not key:
            raise ValueError("Choose a field to filter.")
        if self.filter_kind_combo.currentText() == "obs values":
            values = tuple(
                self.filter_values_list.item(i).data(Qt.UserRole)
                for i in range(self.filter_values_list.count())
                if self.filter_values_list.item(i).checkState() == Qt.Checked
            )
            rule = CohortFilter("obs", key, values=values)
        else:
            lower, upper = (
                self.filter_min_edit.text().strip(),
                self.filter_max_edit.text().strip(),
            )
            try:
                minimum = float(lower) if lower else None
                maximum = float(upper) if upper else None
            except ValueError as exc:
                raise ValueError(
                    "Range bounds must be numbers; leave a bound blank for no limit."
                ) from exc
            rule = CohortFilter(
                "X" if self.filter_kind_combo.currentText() == "X range" else "obs",
                key,
                minimum=minimum,
                maximum=maximum,
            )
        self._set_filters([*self.session.filters, rule])

    def _remove_filter(self):
        row = self.filter_rules_list.currentRow()
        if row < 0:
            raise ValueError("Select a filter to remove.")
        self._set_filters(
            [rule for i, rule in enumerate(self.session.filters) if i != row]
        )

    def _set_filters(self, filters):
        self.session.set_filters(filters)
        self._refresh_filter_rules()
        self._sample_display()
        self.background.set_offsets(self.session.coordinates[self.background_positions])
        self.points.set_offsets(self.session.coordinates[self.display_positions])
        self.clear_selection()
        self.redraw()
        self.status_label.setText(
            "Cohort updated. Draft labels are preserved; only coloured cells can be annotated."
            if self.session.eligible.any()
            else "No cells match these filters. Remove or change a filter to annotate cells."
        )

    def _run(self, action):
        try:
            action()
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            self.status_label.setText(str(exc))

    def _key_pressed(self, event):
        if event.key == "escape":
            self.clear_selection()

    def _set_selector(self, *_args):
        if self.selector is not None:
            self.selector.set_visible(False)
            self.selector.disconnect_events()
            for artist in self.selector.artists:
                artist.remove()
        tool = self.tool_combo.currentText()
        self.selector = None
        if tool == "Lasso":
            self.selector = LassoSelector(self.ax, self._selected, useblit=True)
        elif tool == "Polygon":
            self.selector = PolygonSelector(self.ax, self._selected, useblit=True)
        self.canvas.draw_idle()

    def _selected(self, vertices):
        def select():
            self.session.select_polygon(vertices)
            self.status_label.setText(
                "Selection ready. Enter a category name and click Assign label."
            )

        self._run(select)
        self._update_selection()

    def _update_selection(self):
        selected = self.display_positions[self.session.selected[self.display_positions]]
        self.highlight.set_offsets(self.session.coordinates[selected])
        excluded = int((self.session.scope & ~self.session.finite).sum())
        self.count_label.setText(
            f"Cohort: {self.session.eligible.sum():,} cells ({len(self.display_positions):,} shown in colour). "
            f"Context: {len(self.background_positions):,} shown in grey. "
            f"{self.session.selected.sum():,} selected across the full cohort. "
            f"{excluded:,} cells omitted for missing coordinates."
        )
        self.assign_button.setEnabled(bool(self.session.selected.any()))
        self.undo_button.setEnabled(bool(self.session.history))
        self.canvas.draw_idle()

    def clear_selection(self):
        self.session.selected[:] = False
        self.session.selection_vertices = None
        self._set_selector()
        self._update_selection()

    def _reset_source(self, *_args):
        if (
            self.session.dirty
            and QMessageBox.question(
                self,
                "Reset draft labels?",
                "Changing the starting labels discards unapplied assignments. Continue?",
            )
            != QMessageBox.Yes
        ):
            self.source_combo.blockSignals(True)
            self.source_combo.setCurrentIndex(
                self.source_combo.findData(self.session.source_obs)
            )
            self.source_combo.blockSignals(False)
            return
        self.session.reset_labels(self.source_combo.currentData())
        self.session.dirty = True
        self.parent_prefix_check.setEnabled(self.session.source_obs is not None)
        if self.session.source_obs is None:
            self.parent_prefix_check.setChecked(False)
        self.clear_selection()
        self.redraw()
        self.status_label.setText(
            "Parent labels copied into the draft. Unannotated cells keep these labels."
            if self.session.source_obs is not None
            else "Draft labels reset to Unassigned."
        )

    def redraw(self):
        session = self.session
        if self.ax.get_legend() is not None:
            self.ax.get_legend().remove()
        choice = self.color_combo.currentData()
        if choice is None:
            values = session.labels.iloc[self.display_positions]
        elif choice[0] == "obs":
            values = (
                session.adata.obs[choice[1]]
                .reindex(session.obs_names)
                .iloc[self.display_positions]
            )
        else:
            matrix = session.adata.raw if self.use_raw else session.adata
            column = matrix.var_names.get_loc(choice[1])
            if not isinstance(column, (int, np.integer)):
                raise ValueError("Marker names must be unique to colour by expression.")
            row_positions = session.adata.obs_names.get_indexer(
                session.obs_names[self.display_positions]
            )
            if (row_positions < 0).any():
                raise ValueError(
                    "The AnnData cell set changed; reopen the annotation window."
                )
            expression = session.adata.layers[self.layer] if self.layer else matrix.X
            # Slice the marker first so sparse matrices are never fully densified.
            values = expression[:, column : column + 1][row_positions]
            values = np.asarray(
                values.toarray() if hasattr(values, "toarray") else values
            ).ravel()
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        if choice is not None and pd.api.types.is_numeric_dtype(values):
            values = np.asarray(values, dtype=float)
            finite = values[np.isfinite(values)]
            norm = Normalize(
                vmin=finite.min() if len(finite) else 0,
                vmax=finite.max() if len(finite) else 1,
            )
            self.points.set_array(np.ma.masked_invalid(values))
            self.points.set_norm(norm)
            self.points.set_cmap("viridis")
            self.colorbar = self.figure.colorbar(
                self.points, ax=self.ax, label=choice[1]
            )
        else:
            labels = pd.Series(values, dtype=object).fillna("Unassigned").astype(str)
            if choice is None:
                palette = session.palette()
            else:
                palette = session.observation_palette(choice[1])
            self.points.set_array(None)
            self.points.set_facecolors([palette[label] for label in labels])
            categories = list(pd.unique(labels))
            if len(categories) <= 15:
                self.ax.legend(
                    handles=[
                        Line2D(
                            [],
                            [],
                            marker="o",
                            linestyle="",
                            color=palette[name],
                            label=name,
                        )
                        for name in categories
                    ],
                    loc="best",
                    fontsize=8,
                )
        self.ax.set_title("Draft labels" if choice is None else str(choice[1]))
        self._update_selection()

    def assign(self):
        count = self.session.assign(
            self.label_edit.text(), inherit_parent=self.parent_prefix_check.isChecked()
        )
        self.color_combo.setCurrentIndex(0)
        self.clear_selection()
        self.redraw()
        self.status_label.setText(
            f"Assigned {self.label_edit.text().strip()!r} to {count:,} cells in the draft."
        )

    def undo(self):
        self.session.undo()
        self.redraw()
        self.status_label.setText(
            "Assignment undone in the draft. Apply to update AnnData."
        )

    def apply(self):
        if self.before_apply is not None:
            self.before_apply()
        key = self.session.apply(
            self.key_edit.text(), overwrite=self.overwrite_check.isChecked()
        )
        self.applied_key = key
        self.status_label.setText(
            f"Applied to adata.obs[{key!r}]. Save H5AD copy to persist on disk."
        )
        if self.on_apply is not None:
            self.on_apply(key)

    def save_recipe(self, path=None):
        """Save assigned shapes and settings; this does not apply draft labels."""
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save annotation recipe", "annotation_recipe.json", "JSON (*.json)"
            )
            if not path:
                return None
            from pathlib import Path

            path = Path(path)
            if path.suffix.lower() != ".json":
                path = path.with_suffix(".json")
        saved = self.session.save_recipe(
            path,
            self.key_edit.text(),
            color=self.color_combo.currentData(),
            point_limit=self.point_limit,
            layer=self.layer,
            use_raw=self.use_raw,
            tool=self.tool_combo.currentText(),
            inherit_parent=self.parent_prefix_check.isChecked(),
        )
        self.status_label.setText(
            f"Saved {len(self.session.regions)} assigned regions to {saved}. "
            "Replay applies assigned regions only."
        )
        return saved

    def save_copy(self):
        if (
            self.applied_key is None
            or self.session.dirty
            or self.key_edit.text().strip() != self.applied_key
        ):
            raise ValueError("Apply your draft to AnnData before saving a copy.")
        destination, _ = QFileDialog.getSaveFileName(
            self, "Save annotated AnnData copy", "annotated.h5ad", "AnnData (*.h5ad)"
        )
        if not destination:
            return
        from pathlib import Path
        from .napari_sbt.population_curation import atomic_write_curated_anndata

        path = Path(destination)
        if path.suffix.lower() != ".h5ad":
            path = path.with_suffix(".h5ad")
        atomic_write_curated_anndata(self.session.adata, path)
        self.status_label.setText(f"Saved annotated AnnData to {path}.")

    def closeEvent(self, event):
        if (
            self.session.dirty
            and QMessageBox.question(
                self,
                "Discard draft changes?",
                "Close without applying the remaining draft changes to AnnData?",
            )
            != QMessageBox.Yes
        ):
            event.ignore()
            return
        if self.selector is not None:
            self.selector.disconnect_events()
        if self in _WINDOWS:
            _WINDOWS.remove(self)
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.clear_selection()
            event.accept()
        else:
            super().keyPressEvent(event)


def open_annotation_popup(session, *, block=None, **kwargs):
    global _APPLICATION
    notebook = False
    if kwargs.get("parent") is None and block is not True:
        try:
            from IPython import get_ipython
        except ImportError:
            shell = None
        else:
            shell = get_ipython()
        if shell is not None:
            shell.run_line_magic("gui", "qt")
            notebook = True
    application = QApplication.instance()
    if application is None:
        _APPLICATION = QApplication([])
    dialog = AnnotationDialog(session, **kwargs)
    _WINDOWS.append(dialog)
    if block is None:
        block = not notebook and kwargs.get("parent") is None
    if block:
        dialog.exec()
    else:
        dialog.show()
    return dialog
