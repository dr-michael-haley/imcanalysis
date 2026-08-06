from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from SpatialBiologyToolkit.pipeline.project import initialize_project
from SpatialBiologyToolkit.config.models import DEFAULT_CONFIG_CLASSES


PYSIDE_AVAILABLE = importlib.util.find_spec("PySide6") is not None


@unittest.skipUnless(
    PYSIDE_AVAILABLE, "PySide6 is installed only in the GUI environment"
)
class ProjectConsoleGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    def test_window_builds_every_primary_page_without_scientific_data(self):
        from PySide6.QtWidgets import QPushButton, QSplitter, QStackedWidget

        from SpatialBiologyToolkit.project_gui.app import (
            CataloguePage,
            ConfigurationPage,
            FieldEditor,
            ProjectConsoleWindow,
        )
        from SpatialBiologyToolkit.project_gui.controller import (
            ProjectConsoleController,
        )

        with tempfile.TemporaryDirectory() as temporary:
            context = initialize_project(Path(temporary) / "project")
            with patch.dict(
                os.environ,
                {"SBT_IMC_CONFIG": str(Path(temporary) / ".imc_config")},
            ):
                controller = ProjectConsoleController.open(context.root, read_only=True)
                window = ProjectConsoleWindow(controller)
                window.show()
                self.application.processEvents()
                stacks = window.findChildren(QStackedWidget)

                page_stacks = [stack for stack in stacks if stack.count() == 9]
                self.assertEqual(len(page_stacks), 1)
                expected = sum(
                    len(model.model_fields) for model in DEFAULT_CONFIG_CLASSES.values()
                )
                self.assertEqual(len(controller.config_fields()), expected)
                self.assertLess(len(window.findChildren(FieldEditor)), expected)
                self.assertIn("scheduler", window.statusBar().currentMessage())
                self.assertIn("QFrame#sidebar", window.styleSheet())
                self.assertIsNotNone(window.logo_label)
                self.assertFalse(window.logo_label.pixmap().isNull())
                self.assertEqual(window.navigation.objectName(), "navigation")
                button_labels = {
                    button.text().strip().casefold()
                    for button in window.findChildren(QPushButton)
                }
                self.assertTrue(
                    button_labels.isdisjoint(
                        {"run", "submit", "cancel job", "delete execution"}
                    )
                )

                page_stack = page_stacks[0]
                for page_type in (CataloguePage, ConfigurationPage):
                    page = window.findChild(page_type)
                    page_stack.setCurrentWidget(page)
                    self.application.processEvents()
                    splitter = page.findChild(QSplitter)
                    self.assertGreater(splitter.height(), page.height() * 0.65)
                window.close()

    def test_configuration_origin_filters_and_preparation_navigation(self):
        from PySide6.QtCore import Qt

        from SpatialBiologyToolkit.project_gui.app import (
            ConfigurationPage,
            FieldEditor,
        )
        from SpatialBiologyToolkit.project_gui.controller import (
            ProjectConsoleController,
        )

        with tempfile.TemporaryDirectory() as temporary:
            context = initialize_project(Path(temporary) / "project")
            context.config_path.write_text(
                "general:\n  outputs_folder: review_outputs\n",
                encoding="utf-8",
            )
            controller = ProjectConsoleController.open(context.root)
            page = ConfigurationPage(controller, lambda: None)
            page.level.setCurrentText("All")

            def rendered_editors():
                editors = []
                for index in range(page.form.count()):
                    widget = page.form.itemAt(index).widget()
                    if widget is not None:
                        editors.extend(widget.findChildren(FieldEditor))
                return editors

            page.origin.setCurrentIndex(page.origin.findData("inherited"))
            self.application.processEvents()
            inherited_editors = rendered_editors()
            self.assertTrue(inherited_editors)
            self.assertTrue(
                all(
                    editor.property("configState") == "inherited"
                    for editor in inherited_editors
                )
            )

            page.origin.setCurrentIndex(page.origin.findData("custom"))
            self.application.processEvents()
            custom_editors = rendered_editors()
            self.assertTrue(custom_editors)
            self.assertTrue(
                all(
                    editor.property("configState")
                    in {"stored", "staged", "pending-reset"}
                    for editor in custom_editors
                )
            )

            section = page.prepare_section.currentData()
            self.assertIsNotNone(section)
            page.prepare_selected_section()
            self.application.processEvents()
            self.assertEqual(
                page.sections.currentItem().data(Qt.ItemDataRole.UserRole),
                section,
            )
            self.assertIn("not yet present", page.section_notice.text())
            page.deleteLater()

    def test_invalid_yaml_opens_recovery_page(self):
        from SpatialBiologyToolkit.project_gui.app import (
            ProjectConsoleWindow,
            RecoveryPage,
        )
        from SpatialBiologyToolkit.project_gui.controller import (
            ProjectConsoleController,
        )

        with tempfile.TemporaryDirectory() as temporary:
            context = initialize_project(Path(temporary) / "project")
            context.config_path.write_text("general: [broken\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"SBT_IMC_CONFIG": str(Path(temporary) / ".imc_config")},
            ):
                controller = ProjectConsoleController.open(context.root, read_only=True)
                window = ProjectConsoleWindow(controller)

                self.assertTrue(controller.recovery_mode)
                self.assertIsNotNone(window.findChild(RecoveryPage))
                window.close()

    def test_invalid_widget_draft_disables_config_save(self):
        from SpatialBiologyToolkit.project_gui.app import ConfigurationPage
        from SpatialBiologyToolkit.project_gui.controller import (
            ProjectConsoleController,
        )

        with tempfile.TemporaryDirectory() as temporary:
            context = initialize_project(Path(temporary) / "project")
            controller = ProjectConsoleController.open(context.root)
            page = ConfigurationPage(controller, lambda: None)
            controller.editor.set_value("general.outputs_folder", "review")
            page.reload_specs()
            self.assertTrue(page.save_button.isEnabled())

            page.set_validity("general.outputs_folder", False)

            self.assertFalse(page.save_button.isEnabled())
            self.assertFalse(page.diff_button.isEnabled())
            page.deleteLater()

    def test_registered_projects_can_be_switched_without_scheduler_access(self):
        from SpatialBiologyToolkit.pipeline.project_registry import register_project
        from SpatialBiologyToolkit.project_gui.app import ProjectConsoleWindow
        from SpatialBiologyToolkit.project_gui.controller import (
            ProjectConsoleController,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = root / ".imc_config"
            first = initialize_project(root / "first")
            second = initialize_project(root / "second")
            register_project(first.root, registry_path=settings)
            register_project(second.root, registry_path=settings)
            with patch.dict(os.environ, {"SBT_IMC_CONFIG": str(settings)}):
                controller = ProjectConsoleController.open(first.root, read_only=True)
                window = ProjectConsoleWindow(controller)

                self.assertEqual(window.project_combo.count(), 2)
                window.switch_project(second.root)

                self.assertEqual(window.controller.opened.root, second.root)
                self.assertEqual(window.projects_page.table.rowCount(), 2)
                window.close()


if __name__ == "__main__":
    unittest.main()
