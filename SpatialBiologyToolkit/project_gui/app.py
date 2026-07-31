"""PySide6 desktop application for lightweight SBT project inspection."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import yaml
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from SpatialBiologyToolkit.config.editing import (
    ConfigChangedExternallyError,
    ConfigFieldSpec,
    InvalidConfigEditError,
)
from SpatialBiologyToolkit.pipeline.notes import (
    ProjectNotesChangedError,
    ProjectNotesSession,
)

from .controller import ProjectConsoleController


APP_TITLE = "SBT Project Console"


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child is not None:
            _clear_layout(child)


def _markdown_browser() -> QTextBrowser:
    browser = QTextBrowser()
    browser.setOpenExternalLinks(True)
    return browser


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        return yaml.safe_dump(value, sort_keys=False).strip()
    return str(value)


class DiffDialog(QDialog):
    def __init__(
        self,
        diff: str,
        parent: QWidget | None = None,
        *,
        confirm: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle("Review configuration changes")
        self.resize(900, 650)
        layout = QVBoxLayout(self)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(diff or "No changes are staged.")
        font = QFont("monospace")
        font.setStyleHint(QFont.StyleHint.Monospace)
        text.setFont(font)
        layout.addWidget(text)
        standards = (
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
            if confirm
            else QDialogButtonBox.StandardButton.Close
        )
        buttons = QDialogButtonBox(standards)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class DashboardPage(QWidget):
    def __init__(self, controller: ProjectConsoleController):
        super().__init__()
        self.controller = controller
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        title = QLabel("<h1>Project dashboard</h1>")
        header.addWidget(title)
        header.addStretch()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        header.addWidget(refresh)
        layout.addLayout(header)
        self.browser = _markdown_browser()
        layout.addWidget(self.browser)
        self.render_snapshot()

    def refresh(self) -> None:
        self.controller.refresh()
        self.render_snapshot()

    def render_snapshot(self) -> None:
        snapshot = self.controller.snapshot
        if snapshot is None:
            self.browser.setMarkdown("Project inspection is unavailable.")
            return
        context = snapshot.context
        status = snapshot.latest_recorded_status
        required_missing = sum(
            item.status == "missing" for item in snapshot.validation.required_inputs
        )
        present_assets = sum(view.asset.exists for view in snapshot.assets)
        lines = [
            f"# {context.project_metadata.title or context.root.name}",
            "",
            f"- **Project ID:** `{context.project_metadata.project_id}`",
            f"- **Root:** `{context.root}`",
            f"- **Configuration:** `{context.config_path}`",
            f"- **Config validation:** {'valid' if snapshot.validation.valid else 'invalid'}",
            f"- **Required items missing:** {required_missing}",
            f"- **Assets present:** {present_assets} / {len(snapshot.assets)}",
            f"- **Recorded executions:** {len(snapshot.executions)}",
        ]
        if status is None:
            lines.append("- **Latest recorded status:** no status snapshot")
        else:
            lines.extend(
                [
                    f"- **Latest recorded status:** {status.overall_status}",
                    f"- **Status snapshot time:** {status.checked_at.isoformat()}",
                    "",
                    "> Status is read from the project record. The GUI never queries SLURM.",
                ]
            )
        warnings = [
            item
            for item in (
                *snapshot.validation.optional_inputs,
                *snapshot.validation.reporting_outputs,
            )
            if item.status == "warning"
        ]
        if warnings:
            lines.extend(["", "## Attention", ""])
            lines.extend(
                f"- {item.message} (`{item.path or '-'}`)" for item in warnings
            )
        self.browser.setMarkdown("\n".join(lines))


class CataloguePage(QWidget):
    def __init__(self, controller: ProjectConsoleController):
        super().__init__()
        self.controller = controller
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h1>Stages and modes</h1>"))
        controls = QHBoxLayout()
        self.kind = QComboBox()
        self.kind.addItems(["Stages", "Modes"])
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search names and descriptions")
        controls.addWidget(self.kind)
        controls.addWidget(self.search)
        layout.addLayout(controls)
        splitter = QSplitter()
        self.items = QListWidget()
        self.detail = _markdown_browser()
        splitter.addWidget(self.items)
        splitter.addWidget(self.detail)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        self.kind.currentTextChanged.connect(self.populate)
        self.search.textChanged.connect(self.populate)
        self.items.currentItemChanged.connect(self.show_selected)
        self.populate()

    def populate(self) -> None:
        query = self.search.text().strip().casefold()
        selected = self.kind.currentText()
        self.items.clear()
        records = (
            self.controller.stages()
            if selected == "Stages"
            else self.controller.modes()
        )
        for record in records:
            haystack = f"{record.name} {record.description}".casefold()
            if query and query not in haystack:
                continue
            label = (
                f"{record.display_name}  [{record.name}]"
                if selected == "Stages"
                else f"{record.name}"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, record.name)
            self.items.addItem(item)
        if self.items.count():
            self.items.setCurrentRow(0)
        else:
            self.detail.setMarkdown("No catalogue entries match the search.")

    def show_selected(self, current: QListWidgetItem | None) -> None:
        if current is None:
            return
        name = str(current.data(Qt.ItemDataRole.UserRole))
        if self.kind.currentText() == "Modes":
            mode = self.controller.explain_mode(name)
            lines = [
                f"# Mode `{mode.name}`",
                "",
                mode.description,
                "",
                "## Ordered stages",
                "",
                *[f"{index}. `{stage}`" for index, stage in enumerate(mode.stages, 1)],
            ]
            self.detail.setMarkdown("\n".join(lines))
            return
        stage, documentation = self.controller.explain_stage(name)
        metadata = [
            f"# {stage.display_name}",
            "",
            stage.description,
            "",
            f"- **Alias:** `{stage.name}`",
            f"- **Output slug:** `{stage.output_slug}`",
            f"- **Dependencies:** {', '.join(f'`{item}`' for item in stage.depends_on) or '-'}",
            f"- **Requires assets:** {', '.join(f'`{item}`' for item in stage.requires_assets) or '-'}",
            f"- **Produces assets:** {', '.join(f'`{item}`' for item in stage.produces_assets) or '-'}",
            f"- **Config sections:** {', '.join(f'`{item}`' for item in stage.config_sections) or '-'}",
            f"- **Environment keys:** {', '.join(f'`{item}`' for item in stage.environment_keys) or '-'}",
            "",
            "---",
            "",
            documentation,
        ]
        self.detail.setMarkdown("\n".join(metadata))


class FieldEditor(QFrame):
    def __init__(
        self,
        spec: ConfigFieldSpec,
        on_change: Callable[[str, Any], None],
        on_reset: Callable[[str], None],
        on_validity: Callable[[str, bool], None],
        *,
        read_only: bool,
    ):
        super().__init__()
        self.spec = spec
        self.on_change = on_change
        self.on_reset = on_reset
        self.on_validity = on_validity
        self._yaml_timer: QTimer | None = None
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        badge = "explicit" if spec.explicit else "default"
        title = QLabel(
            f"<b>{spec.name}</b> &nbsp; <small>{badge} · {spec.level}</small>"
        )
        top.addWidget(title)
        top.addStretch()
        reset = QPushButton("Reset to default")
        reset.setEnabled(not read_only and spec.explicit)
        reset.clicked.connect(lambda: self.on_reset(spec.path))
        top.addWidget(reset)
        layout.addLayout(top)
        if spec.description:
            description = QLabel(spec.description)
            description.setWordWrap(True)
            layout.addWidget(description)
        if spec.advice:
            advice = QLabel(f"Advice: {spec.advice}")
            advice.setWordWrap(True)
            advice.setStyleSheet("color: #5b6573;")
            layout.addWidget(advice)
        self.error = QLabel("")
        self.error.setStyleSheet("color: #a01818;")
        self.editor = self._create_editor(read_only)
        layout.addWidget(self.editor)
        layout.addWidget(self.error)

    def _create_editor(self, read_only: bool) -> QWidget:
        spec = self.spec
        if spec.kind == "boolean":
            widget = QCheckBox()
            widget.setChecked(bool(spec.value))
            widget.setEnabled(not read_only)
            widget.toggled.connect(lambda value: self._submit(value))
            return widget
        if spec.kind == "integer" and not spec.nullable:
            widget = QSpinBox()
            widget.setRange(
                int(spec.minimum if spec.minimum is not None else -2_000_000_000),
                int(spec.maximum if spec.maximum is not None else 2_000_000_000),
            )
            widget.setValue(int(spec.value or 0))
            widget.setReadOnly(read_only)
            widget.editingFinished.connect(lambda: self._submit(widget.value()))
            return widget
        if spec.kind == "number" and not spec.nullable:
            widget = QDoubleSpinBox()
            widget.setDecimals(8)
            widget.setRange(
                float(spec.minimum if spec.minimum is not None else -1e15),
                float(spec.maximum if spec.maximum is not None else 1e15),
            )
            widget.setValue(float(spec.value or 0.0))
            widget.setReadOnly(read_only)
            widget.editingFinished.connect(lambda: self._submit(widget.value()))
            return widget
        if spec.kind == "enum" and spec.enum_values:
            widget = QComboBox()
            if spec.nullable:
                widget.addItem("null", None)
            for value in spec.enum_values:
                widget.addItem(_format_value(value), value)
            index = widget.findData(spec.value)
            if index >= 0:
                widget.setCurrentIndex(index)
            widget.setEnabled(not read_only)
            widget.currentIndexChanged.connect(
                lambda _index: self._submit(widget.currentData())
            )
            return widget
        if spec.kind == "yaml":
            widget = QPlainTextEdit()
            widget.setMaximumHeight(110)
            widget.setPlainText(_format_value(spec.value))
            widget.setReadOnly(read_only)
            if not read_only:
                self._yaml_timer = QTimer(self)
                self._yaml_timer.setSingleShot(True)
                self._yaml_timer.setInterval(350)
                self._yaml_timer.timeout.connect(lambda: self._yaml_changed(widget))
                widget.textChanged.connect(self._schedule_yaml_validation)
            return widget
        widget = QLineEdit(_format_value(spec.value))
        widget.setReadOnly(read_only)
        widget.editingFinished.connect(lambda: self._line_changed(widget.text()))
        return widget

    def _schedule_yaml_validation(self) -> None:
        self.on_validity(self.spec.path, False)
        if self._yaml_timer is not None:
            self._yaml_timer.start()

    def _yaml_changed(self, widget: QPlainTextEdit) -> None:
        try:
            value = yaml.safe_load(widget.toPlainText())
        except yaml.YAMLError as exc:
            self.error.setText(str(exc))
            self.on_validity(self.spec.path, False)
            return
        self._submit(value)

    def _line_changed(self, text: str) -> None:
        value: Any = text
        if self.spec.nullable and text.strip().casefold() in {"", "null", "none"}:
            value = None
        elif self.spec.kind == "integer":
            try:
                value = int(text)
            except ValueError:
                self.error.setText("Enter an integer or null.")
                self.on_validity(self.spec.path, False)
                return
        elif self.spec.kind == "number":
            try:
                value = float(text)
            except ValueError:
                self.error.setText("Enter a number or null.")
                self.on_validity(self.spec.path, False)
                return
        self._submit(value)

    def _submit(self, value: Any) -> None:
        try:
            self.on_change(self.spec.path, value)
        except InvalidConfigEditError as exc:
            self.error.setText(str(exc).splitlines()[0])
            self.on_validity(self.spec.path, False)
        else:
            self.error.clear()
            self.on_validity(self.spec.path, True)


class ConfigurationPage(QWidget):
    def __init__(
        self,
        controller: ProjectConsoleController,
        on_saved: Callable[[], None],
    ):
        super().__init__()
        self.controller = controller
        self.on_saved = on_saved
        self.specs: list[ConfigFieldSpec] = []
        self.specs_dirty = False
        self.invalid_paths: set[str] = set()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h1>Configuration</h1>"))
        controls = QHBoxLayout()
        self.scope = QComboBox()
        self.scope.addItem("All config sections", None)
        for mode in controller.modes():
            self.scope.addItem(f"Mode: {mode.name}", ("mode", mode.name))
        for stage in controller.stages():
            self.scope.addItem(f"Stage: {stage.display_name}", ("stage", stage.name))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search fields, descriptions and advice")
        self.level = QComboBox()
        self.level.addItems(["Basic", "Advanced", "Expert", "All"])
        self.reload_button = QPushButton("Reload from disk")
        self.diff_button = QPushButton("Review changes")
        self.save_button = QPushButton("Save configuration")
        self.diff_button.setEnabled(False)
        self.save_button.setEnabled(False)
        controls.addWidget(self.scope)
        controls.addWidget(self.search)
        controls.addWidget(self.level)
        controls.addWidget(self.reload_button)
        controls.addWidget(self.diff_button)
        controls.addWidget(self.save_button)
        layout.addLayout(controls)
        splitter = QSplitter()
        self.sections = QListWidget()
        container = QWidget()
        self.form = QVBoxLayout(container)
        self.form.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        splitter.addWidget(self.sections)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        self.status = QLabel("")
        layout.addWidget(self.status)
        self.search.textChanged.connect(self.populate_sections)
        self.scope.currentIndexChanged.connect(self.populate_sections)
        self.level.currentTextChanged.connect(self.populate_sections)
        self.sections.currentItemChanged.connect(self.render_section)
        self.reload_button.clicked.connect(self.reload_from_disk)
        self.diff_button.clicked.connect(self.show_diff)
        self.save_button.clicked.connect(self.save)
        self.reload_specs()

    def reload_specs(self) -> None:
        self.invalid_paths.clear()
        self.specs = self.controller.config_fields()
        self.specs_dirty = False
        self._update_actions()
        self.populate_sections()

    def _update_actions(self) -> None:
        dirty = bool(self.controller.editor and self.controller.editor.dirty)
        valid_widgets = not self.invalid_paths
        self.diff_button.setEnabled(dirty and valid_widgets)
        self.save_button.setEnabled(
            dirty and valid_widgets and not self.controller.read_only
        )

    def set_validity(self, path: str, valid: bool) -> None:
        if valid:
            self.invalid_paths.discard(path)
        else:
            self.invalid_paths.add(path)
        self._update_actions()

    def _matches(self, spec: ConfigFieldSpec) -> bool:
        scope = self.scope.currentData()
        if scope is not None:
            kind, name = scope
            if spec.section not in self.controller.config_sections_for_scope(
                kind, name
            ):
                return False
        level = self.level.currentText().casefold()
        if level != "all":
            allowed = {
                "basic": {"basic"},
                "advanced": {"basic", "advanced"},
                "expert": {"basic", "advanced", "expert"},
            }[level]
            if spec.level not in allowed:
                return False
        query = self.search.text().strip().casefold()
        if query:
            haystack = " ".join(
                (spec.section, spec.name, spec.description, spec.advice, spec.ui_group)
            ).casefold()
            if query not in haystack:
                return False
        return True

    def populate_sections(self) -> None:
        current = (
            self.sections.currentItem().text() if self.sections.currentItem() else None
        )
        visible = []
        for spec in self.specs:
            if self._matches(spec) and spec.section not in visible:
                visible.append(spec.section)
        if (
            not visible
            and self.scope.currentData() is not None
            and self.level.currentText() == "Basic"
            and not self.search.text().strip()
        ):
            self.level.setCurrentText("Advanced")
            return
        self.sections.clear()
        self.sections.addItems(visible)
        if not visible:
            _clear_layout(self.form)
            self.form.addWidget(QLabel("No fields match the current filters."))
            self.form.addStretch()
            return
        matches = self.sections.findItems(current or "", Qt.MatchFlag.MatchExactly)
        self.sections.setCurrentItem(matches[0] if matches else self.sections.item(0))

    def render_section(self, current: QListWidgetItem | None, *_args) -> None:
        # Invalid widget text is only a local draft. Rebuilding a section restores
        # the last fully validated in-memory proposal.
        self.invalid_paths.clear()
        if current is not None and self.specs_dirty:
            self.specs = self.controller.config_fields()
            self.specs_dirty = False
        self._update_actions()
        _clear_layout(self.form)
        if current is None:
            self.form.addStretch()
            return
        section = current.text()
        groups: dict[str, list[ConfigFieldSpec]] = {}
        for spec in self.specs:
            if spec.section == section and self._matches(spec):
                groups.setdefault(spec.ui_group, []).append(spec)
        for group_name, fields in groups.items():
            group = QGroupBox(group_name)
            group_layout = QVBoxLayout(group)
            for spec in fields:
                group_layout.addWidget(
                    FieldEditor(
                        spec,
                        self.update_value,
                        self.reset_value,
                        self.set_validity,
                        read_only=self.controller.read_only,
                    )
                )
            self.form.addWidget(group)
        self.form.addStretch()

    def update_value(self, path: str, value: Any) -> None:
        if self.controller.read_only:
            return
        self.controller.editor.set_value(path, value)
        self.specs_dirty = True
        self.status.setText(
            f"Staged {len(self.controller.editor.changed_paths)} change(s)."
        )
        self._update_actions()

    def reset_value(self, path: str) -> None:
        if self.controller.read_only:
            return
        self.controller.editor.reset_to_default(path)
        self.reload_specs()
        self.status.setText(
            f"Staged {len(self.controller.editor.changed_paths)} change(s)."
        )

    def show_diff(self) -> None:
        DiffDialog(self.controller.editor.diff(), self).exec()

    def reload_from_disk(self) -> None:
        editor = self.controller.editor
        if (editor.dirty or self.invalid_paths) and QMessageBox.question(
            self,
            APP_TITLE,
            "Discard staged and invalid field changes, then reload from disk?",
        ) != QMessageBox.StandardButton.Yes:
            return
        self.controller.reload()
        if not self.controller.recovery_mode:
            self.reload_specs()
            self.status.setText("Reloaded configuration from disk.")
        self.on_saved()

    def save(self) -> None:
        editor = self.controller.editor
        if self.invalid_paths:
            QMessageBox.warning(
                self,
                APP_TITLE,
                "Correct or discard invalid field text before saving.",
            )
            return
        if not editor.dirty:
            QMessageBox.information(
                self, APP_TITLE, "No configuration changes are staged."
            )
            return
        dialog = DiffDialog(editor.diff(), self, confirm=True)
        dialog.setWindowTitle("Review before saving")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            result = editor.save(self.controller.opened.root)
        except (ConfigChangedExternallyError, InvalidConfigEditError, OSError) as exc:
            QMessageBox.critical(self, APP_TITLE, str(exc))
            return
        self.status.setText(f"Saved. Backup: {result.backup_path}")
        self.controller.reload()
        self.reload_specs()
        self.on_saved()


class AssetsPage(QWidget):
    HEADERS = [
        "Role",
        "Lifecycle",
        "Status",
        "Details",
        "Config field",
        "Producers",
        "Consumers",
        "Path",
    ]

    def __init__(self, controller: ProjectConsoleController):
        super().__init__()
        self.controller = controller
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(QLabel("<h1>Asset register</h1>"))
        header.addStretch()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        header.addWidget(refresh)
        layout.addLayout(header)
        note = QLabel(
            "Computed from the typed config and stage registry. Direct directory counts "
            "are bounded; no scientific files are opened."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)
        self.render_snapshot()

    def refresh(self) -> None:
        self.controller.refresh()
        self.render_snapshot()

    def render_snapshot(self) -> None:
        snapshot = self.controller.snapshot
        if snapshot is None:
            self.table.setRowCount(0)
            return
        self.table.setRowCount(len(snapshot.assets))
        for row, view in enumerate(snapshot.assets):
            asset = view.asset
            status = "present" if asset.exists else "absent"
            if asset.exists and asset.kind == "directory" and not asset.file_count:
                status = "empty"
            details = (
                f"{asset.size_bytes} bytes"
                if asset.kind == "file" and asset.size_bytes is not None
                else f"{asset.file_count or 0}{'+' if asset.count_limited else ''} item(s)"
            )
            values = [
                asset.role,
                asset.lifecycle,
                status,
                details,
                view.config_path or "-",
                ", ".join(view.producers) or "-",
                ", ".join(view.consumers) or "-",
                str(asset.path),
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()


class ReadinessPage(QWidget):
    def __init__(self, controller: ProjectConsoleController):
        super().__init__()
        self.controller = controller
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h1>Workflow readiness</h1>"))
        notice = QLabel(
            "Inspection only: dependencies and assets are evaluated, but the Project "
            "Console has no submission capability."
        )
        notice.setWordWrap(True)
        layout.addWidget(notice)
        controls = QHBoxLayout()
        self.target = QComboBox()
        for mode in controller.modes():
            self.target.addItem(f"Mode: {mode.name}", mode.name)
        for stage in controller.stages():
            self.target.addItem(f"Stage: {stage.display_name}", stage.name)
        inspect = QPushButton("Inspect readiness")
        inspect.clicked.connect(self.refresh)
        controls.addWidget(self.target)
        controls.addWidget(inspect)
        layout.addLayout(controls)
        self.browser = _markdown_browser()
        layout.addWidget(self.browser)
        self.refresh()

    def refresh(self) -> None:
        plan = self.controller.readiness(str(self.target.currentData()))
        lines = [
            f"# {'Ready' if plan.ready else 'Not ready'}",
            "",
            f"Requested: {', '.join(f'`{item}`' for item in plan.requested)}",
            "",
            "## Dependency order",
            "",
        ]
        for index, stage in enumerate(plan.resolved_stages, 1):
            lines.extend(
                [
                    f"{index}. **{stage.name}** — {stage.description}",
                    f"   - Requires: {', '.join(stage.requires_assets) or '-'}",
                    f"   - Produces: {', '.join(stage.produces_assets) or '-'}",
                    f"   - Missing: {', '.join(stage.missing_assets) or '-'}",
                ]
            )
        if plan.errors:
            lines.extend(["", "## Blocking issues", ""])
            lines.extend(f"- {item}" for item in plan.errors)
        if plan.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {item}" for item in plan.warnings)
        lines.extend(
            [
                "",
                "> To execute work, leave this application and use the audited `sbt run` CLI.",
            ]
        )
        self.browser.setMarkdown("\n".join(lines))


class ExecutionsPage(QWidget):
    HEADERS = ["ID", "Stage", "Status", "Started", "Duration", "Assets"]

    def __init__(self, controller: ProjectConsoleController):
        super().__init__()
        self.controller = controller
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(QLabel("<h1>Runs and executions</h1>"))
        header.addStretch()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        header.addWidget(refresh)
        layout.addLayout(header)
        splitter = QSplitter()
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tabs = QTabWidget()
        self.overview = _markdown_browser()
        self.report = _markdown_browser()
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.config = QPlainTextEdit()
        self.config.setReadOnly(True)
        self.tabs.addTab(self.overview, "Overview")
        self.tabs.addTab(self.report, "Report")
        self.tabs.addTab(self.logs, "Log tails")
        self.tabs.addTab(self.config, "Resolved config")
        splitter.addWidget(self.table)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        self.table.currentCellChanged.connect(self.show_selected)
        self.render_snapshot()

    def refresh(self) -> None:
        self.controller.refresh()
        self.render_snapshot()

    def render_snapshot(self) -> None:
        records = self.controller.executions()
        self.table.setRowCount(len(records))
        for row, item in enumerate(records):
            values = [
                item.execution_label,
                item.stage_display_name,
                item.status,
                item.started_at.isoformat(timespec="minutes")
                if item.started_at
                else "-",
                f"{item.duration_seconds:.0f}s"
                if item.duration_seconds is not None
                else "-",
                item.asset_effect,
            ]
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if column == 0:
                    cell.setData(Qt.ItemDataRole.UserRole, item.technical_run_id)
                self.table.setItem(row, column, cell)
        self.table.resizeColumnsToContents()
        if records:
            self.table.selectRow(0)
        else:
            self.overview.setMarkdown("No executions are recorded in this project.")
            self.report.clear()
            self.logs.clear()
            self.config.clear()

    def show_selected(self, current_row: int, _current_column: int, *_args) -> None:
        if current_row < 0:
            return
        cell = self.table.item(current_row, 0)
        if cell is None:
            return
        technical_id = cell.data(Qt.ItemDataRole.UserRole)
        try:
            detail = self.controller.execution_detail(str(technical_id))
        except Exception as exc:  # noqa: BLE001 - inspection errors are displayed
            self.overview.setMarkdown(f"Unable to read execution detail: `{exc}`")
            return
        execution = detail.execution
        lines = [
            f"# {execution.execution_label} — {execution.stage_display_name}",
            "",
            f"- **Recorded status:** {execution.status}",
            f"- **Technical execution ID:** `{execution.technical_run_id}`",
            f"- **Workflow run ID:** `{execution.workflow_run_id}`",
            f"- **SLURM job ID (recorded):** `{execution.slurm_job_id or '-'}`",
            f"- **Output:** `{execution.output_folder}`",
        ]
        if detail.recorded_status:
            lines.extend(
                [
                    f"- **Status snapshot:** {detail.recorded_status.checked_at.isoformat()}",
                    "",
                    "> This is durable recorded state; no scheduler query was made.",
                ]
            )
        if detail.run_manifest:
            lines.extend(
                [
                    "",
                    "## Run provenance",
                    "",
                    f"- Reason: {detail.run_manifest.reason or '-'}",
                    f"- Command: `{detail.run_manifest.command}`",
                    f"- Pipeline version: `{detail.run_manifest.pipeline_version or '-'}`",
                    f"- Git commit: `{detail.run_manifest.git_commit or '-'}`",
                ]
            )
        if detail.stage_manifest:
            manifest = detail.stage_manifest
            lines.extend(
                [
                    "",
                    "## Stage metrics",
                    "",
                    "```json",
                    json.dumps(manifest.metrics, indent=2, default=str),
                    "```",
                ]
            )
        self.overview.setMarkdown("\n".join(lines))
        self.report.setMarkdown(detail.report_text or "No execution report is present.")
        self.logs.setPlainText(
            "STDOUT\n======\n"
            + (detail.stdout_tail or "No recorded stdout log.")
            + "\n\nSTDERR\n======\n"
            + (detail.stderr_tail or "No recorded stderr log.")
        )
        self.config.setPlainText(
            detail.resolved_config_text
            or "No resolved configuration snapshot is present."
        )


class NotesPage(QWidget):
    def __init__(self, controller: ProjectConsoleController):
        super().__init__()
        self.controller = controller
        self.dirty = False
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h1>Project notes</h1>"))
        self.editor = QPlainTextEdit()
        self.editor.setPlainText(
            controller.notes.source_text if controller.notes else ""
        )
        self.editor.setReadOnly(controller.read_only)
        self.editor.textChanged.connect(self._changed)
        layout.addWidget(self.editor)
        buttons = QHBoxLayout()
        self.status = QLabel("")
        buttons.addWidget(self.status)
        buttons.addStretch()
        reload_button = QPushButton("Reload")
        reload_button.clicked.connect(self.reload)
        self.save_button = QPushButton("Save notes")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save)
        buttons.addWidget(reload_button)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)

    def _changed(self) -> None:
        if not self.controller.read_only:
            self.dirty = True
            self.save_button.setEnabled(True)

    def reload(self) -> None:
        if (
            self.dirty
            and QMessageBox.question(self, APP_TITLE, "Discard unsaved note changes?")
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.controller.notes = ProjectNotesSession.open(self.controller.context)
        self.editor.blockSignals(True)
        self.editor.setPlainText(self.controller.notes.source_text)
        self.editor.blockSignals(False)
        self.dirty = False
        self.save_button.setEnabled(False)
        self.status.setText("Reloaded.")

    def save(self) -> None:
        try:
            backup = self.controller.notes.save(
                self.editor.toPlainText(), self.controller.opened.root
            )
        except (ProjectNotesChangedError, OSError) as exc:
            QMessageBox.critical(self, APP_TITLE, str(exc))
            return
        self.dirty = False
        self.save_button.setEnabled(False)
        self.status.setText(f"Saved. Backup: {backup}")


class HelpPage(QWidget):
    def __init__(self, controller: ProjectConsoleController):
        super().__init__()
        layout = QVBoxLayout(self)
        browser = _markdown_browser()
        mode = (
            "read-only" if controller.read_only else "config and notes editing enabled"
        )
        browser.setMarkdown(
            "\n".join(
                [
                    "# About the SBT Project Console",
                    "",
                    "A lightweight view of one existing Spatial Biology Toolkit project.",
                    "",
                    f"- **Mode:** {mode}",
                    f"- **Project:** `{controller.opened.root}`",
                    "- **Scientific data loading:** disabled",
                    "- **SLURM submission and scheduler queries:** disabled",
                    "- **Destructive project operations:** disabled",
                    "",
                    "## What this application can do",
                    "",
                    "- Explain stages, modes and configuration fields.",
                    "- Validate and safely edit configuration.",
                    "- Inspect configured assets and workflow readiness.",
                    "- Read durable execution records, reports and bounded log tails.",
                    "- Edit project notes explicitly.",
                    "",
                    "To submit work, close or leave this application and use the `sbt run` CLI.",
                ]
            )
        )
        layout.addWidget(browser)


class RecoveryPage(QWidget):
    def __init__(
        self,
        controller: ProjectConsoleController,
        on_recovered: Callable[[], None],
    ):
        super().__init__()
        self.controller = controller
        self.on_recovered = on_recovered
        self.dirty = False
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h1>Configuration recovery mode</h1>"))
        message = QLabel(
            "The project marker was found, but its configuration could not be validated. "
            "Other project pages are disabled until the YAML is repaired."
        )
        message.setWordWrap(True)
        layout.addWidget(message)
        self.error = QLabel(controller.opened.error or "")
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #a01818;")
        layout.addWidget(self.error)
        self.editor = QPlainTextEdit(controller.recovery.source_text)
        self.editor.setReadOnly(controller.read_only)
        if not controller.read_only:
            self.editor.textChanged.connect(self._changed)
        layout.addWidget(self.editor)
        buttons = QHBoxLayout()
        validate = QPushButton("Validate")
        validate.clicked.connect(self.validate)
        save = QPushButton("Save repaired config")
        save.setEnabled(not controller.read_only)
        save.clicked.connect(self.save)
        buttons.addStretch()
        buttons.addWidget(validate)
        buttons.addWidget(save)
        layout.addLayout(buttons)

    def _changed(self) -> None:
        self.dirty = True

    def validate(self) -> bool:
        try:
            self.controller.recovery.validate_text(self.editor.toPlainText())
        except InvalidConfigEditError as exc:
            self.error.setText(str(exc))
            return False
        self.error.setText("Configuration is valid and can be saved.")
        self.error.setStyleSheet("color: #176b35;")
        return True

    def save(self) -> None:
        if not self.validate():
            return
        if (
            QMessageBox.question(
                self,
                APP_TITLE,
                "Save the repaired configuration? An exact backup and audit will be created.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        try:
            self.controller.recovery.save_text(
                self.editor.toPlainText(), self.controller.opened.root
            )
            self.dirty = False
            self.controller.reload()
        except Exception as exc:  # noqa: BLE001 - recovery errors are user-facing
            QMessageBox.critical(self, APP_TITLE, str(exc))
            return
        self.on_recovered()


class ProjectConsoleWindow(QMainWindow):
    def __init__(self, controller: ProjectConsoleController):
        super().__init__()
        self.controller = controller
        self.notes_page: NotesPage | None = None
        self.recovery_page: RecoveryPage | None = None
        self.dashboard_page: DashboardPage | None = None
        self.assets_page: AssetsPage | None = None
        self.readiness_page: ReadinessPage | None = None
        self.executions_page: ExecutionsPage | None = None
        self.setWindowTitle(
            f"{APP_TITLE} — {controller.opened.metadata.title or controller.opened.root.name}"
        )
        self.resize(1450, 900)
        self.build()

    def build(self) -> None:
        if self.controller.recovery_mode:
            self.recovery_page = RecoveryPage(
                self.controller, self.rebuild_after_recovery
            )
            self.setCentralWidget(self.recovery_page)
            self.statusBar().showMessage("Configuration recovery mode")
            return
        self.recovery_page = None
        central = QWidget()
        layout = QHBoxLayout(central)
        navigation = QListWidget()
        stack = QStackedWidget()
        self.dashboard_page = DashboardPage(self.controller)
        self.assets_page = AssetsPage(self.controller)
        self.readiness_page = ReadinessPage(self.controller)
        self.executions_page = ExecutionsPage(self.controller)
        pages: list[tuple[str, QWidget]] = [
            ("Dashboard", self.dashboard_page),
            ("Stages & modes", CataloguePage(self.controller)),
            (
                "Configuration",
                ConfigurationPage(self.controller, self.refresh_pages),
            ),
            ("Assets", self.assets_page),
            ("Readiness", self.readiness_page),
            ("Runs", self.executions_page),
        ]
        self.notes_page = NotesPage(self.controller)
        pages.extend((("Notes", self.notes_page), ("Help", HelpPage(self.controller))))
        for name, page in pages:
            navigation.addItem(name)
            stack.addWidget(page)
        navigation.setFixedWidth(180)
        navigation.currentRowChanged.connect(stack.setCurrentIndex)
        navigation.setCurrentRow(0)
        layout.addWidget(navigation)
        layout.addWidget(stack, 1)
        self.setCentralWidget(central)
        mode = (
            "read-only" if self.controller.read_only else "config/notes writes enabled"
        )
        self.statusBar().showMessage(
            f"{mode} · scheduler and scientific execution disabled · {self.controller.opened.root}"
        )

    def rebuild_after_recovery(self) -> None:
        if self.controller.recovery_mode:
            QMessageBox.critical(
                self, APP_TITLE, self.controller.opened.error or "Recovery failed"
            )
            return
        self.build()

    def refresh_pages(self) -> None:
        if self.controller.recovery_mode:
            self.build()
            return
        self.controller.refresh()
        if self.dashboard_page is not None:
            self.dashboard_page.render_snapshot()
        if self.assets_page is not None:
            self.assets_page.render_snapshot()
        if self.readiness_page is not None:
            self.readiness_page.refresh()
        if self.executions_page is not None:
            self.executions_page.render_snapshot()
        self.statusBar().showMessage("Project views refreshed · scheduler disabled")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        dirty_config = bool(self.controller.editor and self.controller.editor.dirty)
        dirty_notes = bool(self.notes_page and self.notes_page.dirty)
        dirty_recovery = bool(self.recovery_page and self.recovery_page.dirty)
        if dirty_config or dirty_notes or dirty_recovery:
            answer = QMessageBox.question(
                self,
                APP_TITLE,
                "Discard unsaved configuration or note changes and close?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        event.accept()


def launch(project: Path | None = None, *, read_only: bool = False) -> int:
    """Launch the Project Console without scientific or scheduler imports."""

    application = QApplication.instance() or QApplication(sys.argv)
    try:
        controller = ProjectConsoleController.open(project, read_only=read_only)
    except Exception as first_error:  # noqa: BLE001 - offer a project chooser
        selected = QFileDialog.getExistingDirectory(
            None,
            "Choose an existing SBT project",
            str(project or Path.cwd()),
        )
        if not selected:
            QMessageBox.critical(None, APP_TITLE, str(first_error))
            return 2
        try:
            controller = ProjectConsoleController.open(selected, read_only=read_only)
        except Exception as exc:  # noqa: BLE001 - user-facing launch failure
            QMessageBox.critical(None, APP_TITLE, str(exc))
            return 2
    window = ProjectConsoleWindow(controller)
    window.show()
    return int(application.exec())


__all__ = ["APP_TITLE", "ProjectConsoleWindow", "launch"]
