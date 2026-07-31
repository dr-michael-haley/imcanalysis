from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

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
        from PySide6.QtWidgets import QPushButton, QStackedWidget

        from SpatialBiologyToolkit.project_gui.app import (
            FieldEditor,
            ProjectConsoleWindow,
        )
        from SpatialBiologyToolkit.project_gui.controller import (
            ProjectConsoleController,
        )

        with tempfile.TemporaryDirectory() as temporary:
            context = initialize_project(Path(temporary) / "project")
            controller = ProjectConsoleController.open(context.root, read_only=True)
            window = ProjectConsoleWindow(controller)
            stacks = window.findChildren(QStackedWidget)

            page_stacks = [stack for stack in stacks if stack.count() == 8]
            self.assertEqual(len(page_stacks), 1)
            expected = sum(
                len(model.model_fields) for model in DEFAULT_CONFIG_CLASSES.values()
            )
            self.assertEqual(len(controller.config_fields()), expected)
            self.assertLess(len(window.findChildren(FieldEditor)), expected)
            self.assertIn("scheduler", window.statusBar().currentMessage())
            button_labels = {
                button.text().strip().casefold()
                for button in window.findChildren(QPushButton)
            }
            self.assertTrue(
                button_labels.isdisjoint(
                    {"run", "submit", "cancel job", "delete execution"}
                )
            )
            window.close()

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
            controller = ProjectConsoleController.open(context.root, read_only=True)
            window = ProjectConsoleWindow(controller)

            self.assertTrue(controller.recovery_mode)
            self.assertIsInstance(window.centralWidget(), RecoveryPage)
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


if __name__ == "__main__":
    unittest.main()
