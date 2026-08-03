from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from SpatialBiologyToolkit.cli.main import app
from SpatialBiologyToolkit.pipeline.inspection import open_project_console
from SpatialBiologyToolkit.pipeline.manifests import write_yaml
from SpatialBiologyToolkit.pipeline.project import (
    PROJECT_MARKER,
    ProjectNotFoundError,
    initialize_project,
)
from SpatialBiologyToolkit.pipeline.project_registry import (
    ProjectRegistryError,
    load_project_registry,
    load_project_reference,
    register_project,
    registered_project_statuses,
    set_default_project,
    unregister_project,
)


class ProjectRegistryTests(unittest.TestCase):
    def test_registered_alias_and_id_load_from_outside_the_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = root / ".imc_config"
            context = initialize_project(root / "cohort")
            _registry, registered = register_project(
                context.root,
                name="Cohort-A",
                registry_path=settings,
            )

            by_name = load_project_reference(
                "Cohort-A",
                registry_path=settings,
            )
            by_id = load_project_reference(
                registered.project_id,
                registry_path=settings,
            )

            self.assertEqual(by_name.root, context.root)
            self.assertEqual(by_id.root, context.root)

    def test_cli_project_commands_accept_registered_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = root / ".imc_config"
            context = initialize_project(root / "cohort")
            register_project(
                context.root,
                name="Cohort-A",
                registry_path=settings,
            )
            runner = CliRunner()

            with patch.dict("os.environ", {"SBT_IMC_CONFIG": str(settings)}):
                result = runner.invoke(
                    app,
                    [
                        "project",
                        "describe",
                        "--project",
                        "Cohort-A",
                        "--format",
                        "json",
                    ],
                )
                summary_result = runner.invoke(
                    app,
                    ["summary", "--project", "Cohort-A", "--format", "json"],
                )

            self.assertEqual(result.exit_code, 0, result.stdout)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["project_id"], context.project_metadata.project_id)
            self.assertEqual(Path(payload["project_root"]), context.root)
            self.assertEqual(summary_result.exit_code, 0, summary_result.stdout)
            summary = json.loads(summary_result.stdout)
            self.assertEqual(summary["project_id"], context.project_metadata.project_id)

    def test_stale_registered_location_has_an_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = root / ".imc_config"
            context = initialize_project(root / "original")
            register_project(
                context.root,
                name="Cohort-A",
                registry_path=settings,
            )
            moved = root / "moved"
            context.root.rename(moved)

            with self.assertRaisesRegex(
                ProjectNotFoundError,
                "Re-register the project from its current root",
            ):
                load_project_reference("Cohort-A", registry_path=settings)

    def test_cli_registers_lists_defaults_and_unregisters_projects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = root / ".imc_config"
            first = initialize_project(root / "first")
            second = initialize_project(root / "second")
            runner = CliRunner()
            environment = {"SBT_IMC_CONFIG": str(settings)}

            with patch.dict("os.environ", environment):
                first_result = runner.invoke(
                    app,
                    [
                        "project",
                        "register",
                        "--project",
                        str(first.root),
                        "--name",
                        "First cohort",
                    ],
                )
                second_result = runner.invoke(
                    app,
                    ["project", "register", "--project", str(second.root)],
                )
                default_result = runner.invoke(
                    app,
                    ["project", "set-default", "second"],
                )
                list_result = runner.invoke(app, ["project", "list"])
                remove_result = runner.invoke(
                    app,
                    ["project", "unregister", "First cohort"],
                )

            for result in (
                first_result,
                second_result,
                default_result,
                list_result,
                remove_result,
            ):
                self.assertEqual(result.exit_code, 0, result.stdout)
            self.assertIn("First cohort", list_result.stdout)
            self.assertIn("second", list_result.stdout)
            self.assertIn("No project files were changed", remove_result.stdout)

    def test_round_trip_preserves_other_imc_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = root / ".imc_config"
            settings.write_text(
                "# existing settings\n"
                "export IMC_EMAIL='analyst@example.org'\n"
                "export OPENAI_API_KEY='keep-this-value'\n",
                encoding="utf-8",
            )
            first = initialize_project(root / "first")
            second = initialize_project(root / "second")

            registry, registered_first = register_project(
                first.root,
                name="First cohort",
                registry_path=settings,
            )
            registry, registered_second = register_project(
                second.root,
                name="Second cohort",
                registry_path=settings,
            )
            registry, selected = set_default_project(
                registered_second.project_id,
                registry_path=settings,
            )

            content = settings.read_text(encoding="utf-8")
            self.assertIn("export IMC_EMAIL='analyst@example.org'", content)
            self.assertIn("export OPENAI_API_KEY='keep-this-value'", content)
            self.assertIn("SBT_PROJECTS_JSON", content)
            self.assertEqual(registry.default_project_id, selected.project_id)
            self.assertEqual(load_project_registry(settings), registry)
            self.assertTrue(
                all(
                    status.available for status in registered_project_statuses(registry)
                )
            )

            updated, removed = unregister_project(
                registered_first.name,
                registry_path=settings,
            )
            self.assertEqual(removed.project_id, registered_first.project_id)
            self.assertEqual(
                [item.name for item in updated.projects], ["Second cohort"]
            )

    def test_reregistering_replaced_default_tracks_new_project_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = root / ".imc_config"
            context = initialize_project(root / "project")
            registry, original = register_project(
                context.root,
                registry_path=settings,
            )
            replacement = context.project_metadata.model_copy(
                update={"project_id": "replacement-project-id"}
            )
            write_yaml(context.root / PROJECT_MARKER, replacement)

            registry, current = register_project(
                context.root,
                registry_path=settings,
            )

            self.assertNotEqual(original.project_id, registry.projects[0].project_id)
            self.assertEqual(current.project_id, "replacement-project-id")
            self.assertEqual(registry.default_project_id, current.project_id)

    def test_malformed_managed_block_is_rejected_without_rewrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = root / ".imc_config"
            original = "# >>> SBT PROJECT REGISTRY >>>\nexport SBT_PROJECTS_JSON='{}'\n"
            settings.write_text(original, encoding="utf-8")
            context = initialize_project(root / "project")

            with self.assertRaisesRegex(ProjectRegistryError, "start marker"):
                register_project(context.root, registry_path=settings)

            self.assertEqual(settings.read_text(encoding="utf-8"), original)

    def test_console_falls_back_from_stale_default_to_available_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = root / ".imc_config"
            stale = initialize_project(root / "stale")
            available = initialize_project(root / "available")
            register_project(stale.root, registry_path=settings)
            register_project(available.root, registry_path=settings)
            (stale.root / PROJECT_MARKER).unlink()

            with (
                patch.dict("os.environ", {"SBT_IMC_CONFIG": str(settings)}),
                patch(
                    "SpatialBiologyToolkit.pipeline.inspection.discover_project_root",
                    side_effect=ProjectNotFoundError("not in a project"),
                ),
            ):
                opened = open_project_console()

            self.assertEqual(opened.root, available.root)


if __name__ == "__main__":
    unittest.main()
