"""Home and guided workspace pages for the existing NapariSBT dock."""

from __future__ import annotations

from pathlib import Path
from threading import Event

from qtpy.QtCore import QEvent, QObject, QSettings, QSize, Qt, QThread, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .validation import allows_quick_open, check_assets


STARTUP_STYLE = """
QLabel#sbtWelcomeTitle { font-size: 20pt; font-weight: 600; }
QLabel#sbtPageTitle { font-size: 17pt; font-weight: 600; }
QLabel#sbtStepCaption { font-weight: 600; }
QLabel#sbtSectionTitle { font-size: 12pt; font-weight: 600; padding-top: 8px; }
QToolButton#sbtDisclosure {
    border: none; padding: 7px 0; text-align: left; font-size: 11pt;
}
QToolButton#sbtDisclosure:hover { background-color: palette(alternate-base); }
QGroupBox {
    border: 1px solid palette(mid); border-radius: 6px;
    margin-top: 14px; padding: 14px 8px 8px 8px;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 10px; padding: 0 5px; font-weight: 600;
}
QPushButton { min-height: 24px; padding: 5px 9px; border-radius: 5px; }
QProgressBar#sbtStepProgress {
    border: none; background-color: palette(mid); border-radius: 2px;
}
QProgressBar#sbtStepProgress::chunk { background-color: #22b899; border-radius: 2px; }
"""


def action_icon(button, icon):
    button.setIcon(button.style().standardIcon(icon))
    button.setIconSize(QSize(18, 18))


class AssetCheckThread(QThread):
    progress = Signal(str)
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, request, parent):
        super().__init__(parent)
        self.request = request
        self.stop = Event()

    def run(self):
        try:
            result = check_assets(
                **self.request,
                progress=self.progress.emit,
                cancelled=self.stop.is_set,
            )
            self.result.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class ValidationCloseGuard(QObject):
    """Keep Qt's worker alive until an in-flight network read has finished."""

    def __init__(self, flow, parent):
        super().__init__(parent)
        self.flow = flow

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Close and self.flow.checking:
            self.flow.cancel_validation()
            self.flow.check_status.setText(
                "Stopping the dataset check. Close again once it finishes."
            )
            event.ignore()
            return True
        return False


def disclosure(title, widgets):
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    toggle = QToolButton()
    toggle.setObjectName("sbtDisclosure")
    toggle.setText(title)
    toggle.setCheckable(True)
    toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    toggle.setArrowType(Qt.RightArrow)
    toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    content = QWidget()
    inner = QVBoxLayout(content)
    inner.setContentsMargins(4, 4, 4, 8)
    inner.setSpacing(12)
    for widget in widgets:
        inner.addWidget(widget)
    content.hide()
    toggle.toggled.connect(content.setVisible)
    toggle.toggled.connect(
        lambda expanded: toggle.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )
    )
    layout.addWidget(toggle)
    layout.addWidget(content)
    return container, toggle


class WorkspaceFlow:
    """Reuse controller controls; only this class owns page navigation."""

    def __init__(self, controller, parts, *, supplied_inputs):
        self.c = c = controller
        self.worker = None
        self.step = 0
        self._last_mode = None
        self.settings = QSettings("SpatialBiologyToolkit", "NapariSBT")
        self.validation_badge = QLabel()
        self.validation_badge.setWordWrap(True)
        c.root.layout().insertWidget(1, self.validation_badge)
        self.root = QWidget()
        self.root.setStyleSheet(STARTUP_STYLE)
        layout = QVBoxLayout(self.root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        header = QHBoxLayout()
        self.step_caption = QLabel()
        self.step_caption.setObjectName("sbtStepCaption")
        header.addWidget(self.step_caption, 1)
        help_button = QPushButton("Help")
        action_icon(help_button, QStyle.SP_DialogHelpButton)
        help_button.clicked.connect(lambda: c.show_tab_help("setup", "Workspace"))
        header.addWidget(help_button)
        layout.addLayout(header)
        self.heading = QLabel()
        self.heading.setObjectName("sbtPageTitle")
        self.heading.setWordWrap(True)
        layout.addWidget(self.heading)
        self.step_progress = QProgressBar()
        self.step_progress.setObjectName("sbtStepProgress")
        self.step_progress.setRange(0, 4)
        self.step_progress.setTextVisible(False)
        self.step_progress.setFixedHeight(4)
        layout.addWidget(self.step_progress)
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)
        self.page_widgets = []
        for _ in range(5):
            page = QWidget()
            page.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(2, 12, 8, 8)
            page_layout.setSpacing(12)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.NoFrame)
            scroll.setWidget(page)
            self.pages.addWidget(scroll)
            self.page_widgets.append(page)
        dataset, task, check, save, summary = self.page_widgets

        # Reparent existing widgets rather than duplicating their state/callbacks.
        dataset.layout().addWidget(QLabel("Choose the folder containing your dataset."))
        dataset.layout().addWidget(c.project_edit)
        dataset.layout().addWidget(c.choose_project_button)
        registered, _ = disclosure(
            "Registered SBT projects", [c.registered_project_combo]
        )
        dataset.layout().addWidget(registered)
        self.files_summary = QLabel()
        self.files_summary.setWordWrap(True)
        dataset.layout().addWidget(self.files_summary)
        file_details = QWidget()
        file_layout = QVBoxLayout(file_details)
        file_layout.setContentsMargins(0, 0, 0, 0)
        # Compact vertical file rows fit a narrow dock; no off-screen chooser buttons.
        for title, key, field, buttons in (
            ("Cell data (.h5ad)", "anndata", c.anndata_edit, [c.choose_anndata_button]),
            ("Cell outlines", "masks", c.masks_edit, [c.choose_masks_button]),
            (
                "Staining images",
                "images",
                c.images_edit,
                [
                    c.add_images_folder_button,
                    c.remove_images_folder_button,
                    c.clear_images_folders_button,
                ],
            ),
        ):
            group = QGroupBox(title)
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(8)
            badge = c._setup_status_labels[key]
            badge.setMinimumWidth(0)
            group_layout.addWidget(badge)
            group_layout.addWidget(field)
            group_layout.addWidget(buttons[0])
            action_icon(buttons[0], QStyle.SP_DirOpenIcon)
            if len(buttons) > 1:
                manage, _ = disclosure("Manage selected folders", buttons[1:])
                group_layout.addWidget(manage)
            file_layout.addWidget(group)
        file_layout.addWidget(c.advanced_identity_check)
        file_layout.addWidget(c.identity_widget)
        c.identity_widget.hide()
        details, self.files_toggle = disclosure("Review dataset files", [file_details])
        dataset.layout().addWidget(details)
        c.registered_project_combo.setMinimumWidth(0)
        c.project_edit.setMinimumWidth(0)
        c.choose_project_button.setText("Choose dataset…")
        action_icon(c.choose_project_button, QStyle.SP_DirOpenIcon)
        parts["extra_widget"].layout().setDirection(QBoxLayout.TopToBottom)
        self.advanced_files, self.advanced_files_toggle = disclosure(
            "Additional image folders",
            [parts["extra_widget"]],
        )
        dataset.layout().addWidget(self.advanced_files)
        # Loading and rescanning are explicit troubleshooting actions.
        tools = QWidget()
        tools_layout = QVBoxLayout(tools)
        for button in (
            c.detect_dataset_inputs_button,
            c.reload_all_inputs_button,
            c.reload_anndata_button,
        ):
            tools_layout.addWidget(button)
        self.troubleshooting, _ = disclosure("Troubleshooting", [tools])
        dataset.layout().addWidget(self.troubleshooting)
        dataset.layout().addStretch()

        task.layout().addWidget(QLabel("What would you like to do?"))
        task.layout().addWidget(c.workflow_combo)
        c.workflow_combo.show()
        full_index = c.workflow_combo.findData("full_workspace")
        c.workflow_combo.setItemText(full_index, "Advanced: show every tool")
        task.layout().addWidget(c.workflow_description_label)
        task.layout().addWidget(c.classification_setup_widget)
        self.cohort_summary = QLabel()
        self.cohort_summary.setWordWrap(True)
        task.layout().addWidget(self.cohort_summary)
        c.load_adata_button.hide()
        c.preview_button.hide()
        # Trial setup stays available, with its own cohort preparation action.
        c.preview_button.setText("Prepare cell selection")
        c.preview_button.clicked.disconnect()
        c.preview_button.clicked.connect(c._guard(c.prepare_cohort))
        c.preview_button.show()
        for group in (parts["scope"], parts["trial"], parts["classes"]):
            group.setTitle(group.title().split(". ", 1)[-1])
        trial_options, _ = disclosure(
            "Feature Discovery Trial options", [parts["trial"]]
        )
        c.classification_setup_widget.layout().insertWidget(1, trial_options)
        task.layout().addStretch()

        self.quick = QCheckBox("Open quickly (check as I go)")
        self.quick.setChecked(True)
        self.warning = QLabel(
            "Dataset not fully checked. Missing files or mismatched cell outlines "
            "may only become apparent when you open a region. Full checking is "
            "available here at any time."
        )
        self.warning.setWordWrap(True)
        self.check_status = QLabel("No full asset check has been run.")
        self.check_status.setWordWrap(True)
        check.layout().addWidget(self.quick)
        check.layout().addWidget(self.warning)
        c.validate_integrity_button.setText("Check entire dataset")
        c.validate_integrity_button.clicked.disconnect()
        c.validate_integrity_button.clicked.connect(c._guard(self.start_check))
        check.layout().addWidget(c.validate_integrity_button)
        self.index_button = QPushButton("Find files only")
        self.index_button.setToolTip(
            "Build a file index without checking mask contents."
        )
        self.index_button.clicked.connect(
            c._guard(lambda: self.start_check(index_only=True))
        )
        check.layout().addWidget(self.index_button)
        self.cancel_check = QPushButton("Cancel check")
        self.cancel_check.hide()
        self.cancel_check.clicked.connect(self.cancel_validation)
        check.layout().addWidget(self.cancel_check)
        check.layout().addWidget(self.check_status)
        check.layout().addWidget(c.integrity_status_label)
        details, self.details_toggle = disclosure(
            "Show check details", [c.preview_text]
        )
        check.layout().addWidget(details)
        check.layout().addStretch()

        save.layout().addWidget(
            QLabel("Name this workspace and choose where your work is saved.")
        )
        save.layout().addWidget(c.name_edit)
        save.layout().addWidget(parts["location_row"])
        parts["location_row"].layout().setDirection(QBoxLayout.TopToBottom)
        parts["display"].setTitle("Channel intensity limits")
        parts["display"].setProperty("sbtWorkflowBox", False)
        parts["display"].setProperty("sbtNumbered", False)
        parts["display"].setProperty("sbtAccent", "")
        # The page Help button replaces the floating title button in this narrow pane.
        parts["display"].help_button.hide()
        for group in (parts["display"], parts["classes"]):
            for row in group.findChildren(QHBoxLayout):
                row.setDirection(QBoxLayout.TopToBottom)
        editor = QWidget()
        editor_layout = QVBoxLayout(editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        for widget in (
            c.normalization_table,
            c.add_normalization_row_button,
            c.remove_normalization_row_button,
            c.advanced_normalization_check,
            c.normalization_json_edit,
            c.validate_normalization_button,
        ):
            editor_layout.addWidget(widget)
        c.normalization_json_edit.setVisible(c.advanced_normalization_check.isChecked())
        editor_section, _ = disclosure("Edit channel limits", [editor])
        parts["display"].layout().insertWidget(2, editor_section)
        c._setup_status_labels["normalization"].hide()
        normalisation_intro = QLabel(
            "Use a saved normalisation dictionary to keep each channel’s minimum "
            "and maximum intensity consistent across regions. This is optional."
        )
        normalisation_intro.setWordWrap(True)
        save.layout().addWidget(normalisation_intro)
        display, self.display_toggle = disclosure(
            "Saved normalisation dictionary", [parts["display"]]
        )
        save.layout().addWidget(display)
        defaults, _ = disclosure(
            "Other display settings",
            [parts["display_defaults"], c.live_recipe_tracking_check],
        )
        save.layout().addWidget(defaults)
        action_icon(c.choose_normalization_button, QStyle.SP_DirOpenIcon)
        action_icon(c.save_normalization_button, QStyle.SP_DialogSaveButton)
        save.layout().addStretch()

        self.summary_text = QLabel()
        self.summary_text.setWordWrap(True)
        summary.layout().addWidget(self.summary_text)
        self.return_tools = QPushButton("Continue working")
        self.return_tools.clicked.connect(self.open_tools)
        summary.layout().addWidget(self.return_tools)
        results = QPushButton("Open results folder")
        action_icon(results, QStyle.SP_DirOpenIcon)
        results.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(c.paths.exports)))
            if c.paths
            else None
        )
        summary.layout().addWidget(results)
        for title, step in (
            ("Dataset settings / troubleshooting", 0),
            ("Change task", 1),
            ("Check dataset", 2),
            ("Display settings", 3),
        ):
            button = QPushButton(title)
            button.clicked.connect(
                lambda checked=False, index=step: self.show_step(index)
            )
            summary.layout().addWidget(button)
        summary.layout().addStretch()

        for page in self.page_widgets:
            for label in page.findChildren(QLabel):
                label.setWordWrap(True)
                label.setMinimumWidth(0)

        self.footer_status = QLabel()
        self.footer_status.setWordWrap(True)
        layout.addWidget(self.footer_status)
        footer = QHBoxLayout()
        self.back = QPushButton("Back")
        action_icon(self.back, QStyle.SP_ArrowBack)
        self.next = QPushButton("Continue")
        action_icon(self.next, QStyle.SP_ArrowForward)
        self.next.setObjectName("sbtPrimaryActionButton")
        self.fix = QPushButton("Show what needs attention")
        self.fix.clicked.connect(self.fix_problem)
        self.back.clicked.connect(self.go_back)
        self.next.clicked.connect(c._guard(self.advance))
        layout.addWidget(self.fix)
        footer.addWidget(self.back)
        footer.addWidget(self.next)
        layout.addLayout(footer)
        self.quick.toggled.connect(lambda: c.refresh_setup_readiness())

        # Replace the outer scroll area so navigation stays visible when scrolling.
        old_index = c._workflow_tab_indices["setup"]
        old_scroll = c.tabs.widget(old_index)
        old_setup = old_scroll.takeWidget()
        old_setup.setParent(self.root)
        old_setup.hide()
        c.tabs.removeTab(old_index)
        c.tabs.insertTab(old_index, self.root, "Workspace")
        old_scroll.deleteLater()

        home = QWidget()
        home.setStyleSheet(STARTUP_STYLE)
        home_layout = QVBoxLayout(home)
        home_layout.setContentsMargins(18, 22, 18, 18)
        home_layout.setSpacing(12)
        title = QLabel("Welcome to NapariSBT")
        title.setObjectName("sbtWelcomeTitle")
        title.setWordWrap(True)
        home_layout.addWidget(title)
        intro = QLabel("Open saved work, or choose a dataset to create a workspace.")
        intro.setWordWrap(True)
        home_layout.addWidget(intro)
        new = QPushButton("Create a workspace")
        new.setObjectName("sbtPrimaryActionButton")
        action_icon(new, QStyle.SP_FileIcon)
        new.clicked.connect(c._guard(self.new_workspace))
        home_layout.addWidget(new)
        c.load_experiment_button.setText("Open a workspace…")
        action_icon(c.load_experiment_button, QStyle.SP_DirOpenIcon)
        home_layout.addWidget(c.load_experiment_button)
        self.recent = QComboBox()
        recent_heading = QLabel("Recent workspaces")
        recent_heading.setObjectName("sbtSectionTitle")
        home_layout.addWidget(recent_heading)
        home_layout.addWidget(self.recent)
        recent_open = QPushButton("Open selected workspace")
        action_icon(recent_open, QStyle.SP_ArrowForward)
        recent_open.clicked.connect(c._guard(self.open_recent))
        home_layout.addWidget(recent_open)
        existing, _ = disclosure(
            "Workspaces in this dataset",
            [
                c.workspace_combo,
                c.refresh_workspaces_button,
                c.open_workspace_button,
                c.workspace_summary_label,
            ],
        )
        home_layout.addWidget(existing)
        c.new_workspace_button.hide()
        self.return_workspace = QPushButton("Return to current workspace")
        self.return_workspace.clicked.connect(self.show_workspace)
        home_layout.addWidget(self.return_workspace)
        home_layout.addStretch()
        c.tabs.insertTab(0, home, "Home")
        c.tabs.setTabIcon(0, home.style().standardIcon(QStyle.SP_DirHomeIcon))
        c._workflow_tab_indices = {
            key: index + 1 for key, index in c._workflow_tab_indices.items()
        }
        c._workflow_tab_indices["home"] = 0
        self.load_recents()
        if c.manifest:
            self.remember_workspace()
        self.show_step(4 if c.manifest else 0)
        c.tabs.setCurrentIndex(
            c._workflow_tab_indices["setup"] if supplied_inputs or c.manifest else 0
        )

    def protect_window(self):
        window = self.c.root.window()
        self.close_guard = ValidationCloseGuard(self, window)
        window.installEventFilter(self.close_guard)

    def load_recents(self):
        self.recent.clear()
        for path in self.settings.value("recentWorkspaces", [], type=list):
            self.recent.addItem(Path(path).name, str(path))
            self.recent.setItemData(
                self.recent.count() - 1, str(path), self.c.Qt.ToolTipRole
            )

    def remember_workspace(self):
        if not self.c.paths:
            return
        path = str(self.c.paths.root)
        recent = self.settings.value("recentWorkspaces", [], type=list)
        self.settings.setValue(
            "recentWorkspaces", [path, *[str(p) for p in recent if str(p) != path]][:12]
        )
        self.load_recents()

    def new_workspace(self):
        c = self.c
        c.start_new_workspace()
        if c.paths is None:
            self.show_step(0)
            c.tabs.setCurrentIndex(c._workflow_tab_indices["setup"])

    def open_recent(self):
        path = self.recent.currentData()
        if not path:
            return
        if not (Path(path) / "experiment.yaml").is_file():
            self.footer_status.setText(
                "Workspace unavailable. Reconnect its drive or use Open a workspace to locate it."
            )
            self.c.QMessageBox.information(
                self.root,
                "Workspace unavailable",
                "Reconnect the workspace drive, or use Open a workspace to locate it.",
            )
            return
        self.c.load_existing_experiment(Path(path))

    def show_workspace(self):
        self.show_step(4 if self.c.manifest else self.step)
        self.c.tabs.setCurrentIndex(self.c._workflow_tab_indices["setup"])

    def open_tools(self):
        c = self.c
        preferred = {
            "classification": "feature_building",
            "cell_labeling": "labeler",
            "population_qc": "population_qc",
            "population_curation": "populations",
            "dataset_maintenance": "dataset_maintenance",
        }.get(c.current_workflow_mode(), "explore")
        for topic in (
            preferred,
            "explore",
            "population_qc",
            "labeler",
            "populations",
            "dataset_maintenance",
            "feature_building",
        ):
            index = c._workflow_tab_indices.get(topic)
            if index is not None and c.tabs.isTabVisible(index):
                c.tabs.setCurrentIndex(index)
                return

    def show_step(self, step):
        self.step = step
        self.pages.setCurrentIndex(step)
        self.refresh()

    def go_back(self):
        if self.c.manifest:
            self.show_step(4)
        elif self.step:
            self.show_step(self.step - 1)
        else:
            self.c.tabs.setCurrentIndex(0)

    def fix_problem(self):
        c = self.c
        issue = next(
            (item for item in c._current_setup_checks if item.level == "blocked"), None
        )
        if issue is None:
            self.show_step(2)
            return
        self.show_step(
            {"workspace": 3, "workflow": 1, "normalization": 3}.get(issue.key, 0)
        )
        if issue.key == "identity":
            self.files_toggle.setChecked(True)
            c.advanced_identity_check.setChecked(True)
        if issue.key in {"anndata", "masks", "images"}:
            self.files_toggle.setChecked(True)
        if issue.key == "extra_images":
            self.advanced_files_toggle.setChecked(True)
        if issue.key == "normalization":
            self.display_toggle.setChecked(True)
        self.footer_status.setText(f"{issue.label}: {issue.detail}")

    def advance(self):
        c = self.c
        if c.manifest:
            self.show_step(4)
            return
        if self.step == 0:
            problems = [
                check
                for check in c._current_setup_checks
                if check.level == "blocked" and check.key in {"anndata", "extra_images"}
            ]
            if problems:
                self.advanced_files_toggle.setChecked(True)
                raise ValueError(problems[0].detail)
            if c.adata is None or (
                c.anndata_edit.text().strip()
                and c.anndata_edit.text().strip()
                != getattr(c, "_loaded_anndata_source", None)
            ):
                c.load_anndata_selectors()
        if self.step == 1:
            if not c.current_workflow_mode():
                raise ValueError("Choose what you would like to do.")
            c.prepare_cohort()
            problems = [
                check
                for check in c._current_setup_checks
                if check.level == "blocked"
                and check.key in {"masks", "images", "identity", "extra_images"}
            ]
            if problems:
                self.show_step(0)
                self.files_toggle.setChecked(True)
                raise ValueError(f"{problems[0].label}: {problems[0].detail}")
        if self.step == 2:
            if not self.quick_open and not c.integrity_is_current():
                self.start_check()
                return
        if self.step == 3:
            c.create_experiment()
            return
        self.show_step(self.step + 1)

    @property
    def quick_open(self):
        return (
            allows_quick_open(self.c.current_workflow_mode()) and self.quick.isChecked()
        )

    @property
    def checking(self):
        # Remain busy until the queued result and finished signals are handled.
        return self.worker is not None

    def refresh(self):
        c = self.c
        mode = c.current_workflow_mode()
        c.scope_label.setVisible(c.manifest is not None)
        c.save_normalization_button.setEnabled(c.manifest is not None)
        c.save_normalization_button.setToolTip(
            "Save these limits to the current workspace."
            if c.manifest
            else "The limits are saved automatically when you create the workspace."
        )
        cells = (
            "Notebook data supplied"
            if c._in_memory_adata is not None
            else Path(c.anndata_edit.text()).name
            if c.anndata_edit.text().strip()
            else "Choose a cell-data file"
        )
        masks = (
            Path(c.masks_edit.text()).name
            if c.masks_edit.text().strip()
            else "No cell outlines selected"
        )
        folder_count = len(
            [line for line in c.images_edit.toPlainText().splitlines() if line.strip()]
        )
        self.files_summary.setText(
            f"Cell data: {cells}\nCell outlines: {masks}\nStaining images: {folder_count} folder(s)"
        )
        if mode != self._last_mode:
            self._last_mode = mode
            self.quick.blockSignals(True)
            self.quick.setChecked(allows_quick_open(mode))
            self.quick.blockSignals(False)
        self.quick.setEnabled(allows_quick_open(mode) and not self.checking)
        self.warning.setText(
            "Dataset not fully checked. Missing files or mismatched cell outlines may only "
            "become apparent when you open a region. Checks are required before training or batch processing."
            if allows_quick_open(mode)
            else "This task requires a full asset check before creating the workspace."
        )
        self.warning.setVisible(not c.integrity_is_current())
        self.validation_badge.setVisible(c.manifest is not None)
        self.validation_badge.setText(
            "Assets: checked in this session"
            if c.integrity_is_current()
            else "Assets: "
            + getattr(
                c,
                "_recorded_validation_status",
                "Not fully checked — regions are checked as opened",
            )
        )
        if hasattr(self, "return_workspace"):
            self.return_workspace.setVisible(c.manifest is not None)
        names = ("Dataset", "Task", "Check data", "Save workspace", "Workspace")
        self.heading.setText(names[self.step])
        self.step_caption.setText(
            f"STEP {self.step + 1} OF 4"
            if not c.manifest and self.step < 4
            else "YOUR WORKSPACE"
        )
        self.step_progress.setVisible(not c.manifest and self.step < 4)
        self.step_progress.setValue(min(self.step + 1, 4))
        self.back.setEnabled(not self.checking)
        self.back.setText("Summary" if c.manifest and self.step < 4 else "Back")
        self.next.setText(
            "Done"
            if c.manifest
            else "Create workspace"
            if self.step == 3
            else "Check dataset"
            if self.step == 2 and not self.quick_open and not c.integrity_is_current()
            else "Continue"
        )
        self.next.setVisible(self.step != 4)
        self.next.setEnabled(
            not self.checking
            and (
                self.step != 3 or c.manifest is not None or c.create_button.isEnabled()
            )
        )
        self.fix.setVisible(
            not c.manifest and self.step == 3 and not c.create_button.isEnabled()
        )
        if not c.manifest and not self.checking:
            if self.step == 0:
                self.next.setEnabled(
                    bool(
                        c.anndata_edit.text().strip() or c._in_memory_adata is not None
                    )
                )
            elif self.step == 1:
                self.next.setEnabled(mode is not None)
        if c.manifest:
            self.summary_text.setText(
                f"{c.manifest.name}\n\nDataset: {c.project_root}\nSaved work: {c.paths.root}\n\n{c.integrity_status_label.text()}"
            )
        self.footer_status.setText(
            c.setup_readiness_label.text()
            if self.step == 3
            else ""
            if not self.checking
            else "Checking assets. You can cancel; network reads finish before cancellation takes effect."
        )

    def start_check(self, *, index_only=False):
        if self.checking:
            return
        c = self.c
        c.prepare_cohort()
        self.signature = c._current_integrity_signature()
        self.worker = AssetCheckThread(
            dict(c.asset_check_request(), index_only=index_only), c.root
        )
        self.worker.progress.connect(self.check_status.setText)
        self.worker.result.connect(self.checked)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.finished)
        self.worker.start()
        c.validate_integrity_button.setEnabled(False)
        self.index_button.setEnabled(False)
        self.cancel_check.show()
        self.refresh()

    def cancel_validation(self):
        if self.worker:
            self.worker.stop.set()
            self.check_status.setText("Cancelling after the current network read…")

    def checked(self, result):
        if self.signature != self.c._current_integrity_signature():
            self.check_status.setText(
                "Inputs changed while checking. Check again to validate the new selection."
            )
            return
        self.c.apply_asset_check(result)
        self.check_status.setText(result.summary())
        self.details_toggle.setChecked(bool(result.issues))

    def failed(self, message):
        self.check_status.setText(f"Check could not finish: {message}")

    def finished(self):
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        self.c.validate_integrity_button.setEnabled(True)
        self.index_button.setEnabled(True)
        self.cancel_check.hide()
        self.c.refresh_setup_readiness()
        self.refresh()
