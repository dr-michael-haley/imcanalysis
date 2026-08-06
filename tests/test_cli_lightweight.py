import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from typer.testing import CliRunner

from SpatialBiologyToolkit.cli.main import DOCUMENTATION_URL, REPOSITORY_URL, app


class LightweightCliTests(unittest.TestCase):
    def test_project_init_writes_compact_config_by_default(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"

            result = runner.invoke(
                app,
                ["project", "init", "--project", str(project)],
            )

            self.assertEqual(result.exit_code, 0, result.stdout)
            self.assertEqual(
                yaml.safe_load(
                    (project / "config.yaml").read_text(encoding="utf-8")
                ),
                {},
            )

    def test_cli_startup_does_not_import_heavy_analysis_dependencies(self):
        heavy_modules = [
            "anndata",
            "cellpose",
            "scanpy",
            "skimage",
            "squidpy",
            "tifffile",
            "torch",
            "PySide6",
        ]
        script = (
            "import json, sys; "
            "import SpatialBiologyToolkit.cli.main; "
            f"print(json.dumps({{name: name in sys.modules for name in {heavy_modules!r}}}))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        imported = json.loads(completed.stdout)
        self.assertEqual(imported, {name: False for name in heavy_modules})

    def test_homepage_links_and_explained_commands_are_exposed(self):
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertIn(REPOSITORY_URL, result.stdout)
        self.assertIn(DOCUMENTATION_URL, result.stdout)
        expected = {
            "plan": "Validate and preview stages",
            "run": "Allocate execution IDs",
            "status": "Refresh and show scheduler status",
            "logs": "Show or locate recorded stdout and stderr",
            "report": "Display the human-facing report",
            "summary": "List project executions",
            "remove": "Remove a visible execution safely",
            "config": "Validate, compact, and export typed pipeline configuration",
            "project": "Initialize, adopt, validate, and inspect SBT projects",
            "stages": "List and explain registered pipeline stages",
            "modes": "List and explain named workflow modes",
            "gui": "Launch optional interactive desktop applications",
        }
        for command, explanation in expected.items():
            self.assertIn(command, result.stdout)
            self.assertIn(explanation, result.stdout)

        homepage = runner.invoke(app)
        self.assertEqual(homepage.exit_code, 0, homepage.stdout)
        self.assertIn(REPOSITORY_URL, homepage.stdout)
        self.assertIn(DOCUMENTATION_URL, homepage.stdout)
        self.assertIn("Commands", homepage.stdout)

    def test_stages_list_uses_color_for_terminal_table_only(self):
        runner = CliRunner()
        colored = runner.invoke(app, ["stages", "list"], color=True)

        self.assertEqual(colored.exit_code, 0, colored.stdout)
        self.assertIn("\x1b[", colored.stdout)
        self.assertIn("cellvision-cluster", colored.stdout)

        plain = runner.invoke(app, ["stages", "list"], color=False)
        self.assertEqual(plain.exit_code, 0, plain.stdout)
        self.assertNotIn("\x1b[", plain.stdout)
        header = plain.stdout.splitlines()[0]
        cluster_row = next(
            line
            for line in plain.stdout.splitlines()
            if line.startswith("cellvision-cluster")
        )
        self.assertEqual(header.index("ENVIRONMENT"), cluster_row.index("rapids"))

        machine = runner.invoke(
            app, ["stages", "list", "--format", "json"], color=True
        )
        self.assertEqual(machine.exit_code, 0, machine.stdout)
        self.assertNotIn("\x1b[", machine.stdout)
        self.assertEqual(json.loads(machine.stdout)[0]["name"], "prep")

    def test_project_gui_is_discoverable_without_importing_qt(self):
        runner = CliRunner()
        result = runner.invoke(app, ["gui", "--help"])

        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertIn("project", result.stdout)
        self.assertIn("no scheduler", result.stdout)
        self.assertIn("capability", result.stdout)

    def test_project_gui_launcher_forwards_project_and_read_only(self):
        runner = CliRunner()
        with patch(
            "SpatialBiologyToolkit.cli.main.importlib.util.find_spec",
            return_value=object(),
        ), patch(
            "SpatialBiologyToolkit.cli.main.subprocess.run",
            return_value=SimpleNamespace(returncode=0),
        ) as run:
            result = runner.invoke(
                app,
                ["gui", "project", "--project", "example", "--read-only"],
            )

        self.assertEqual(result.exit_code, 0, result.stdout)
        command = run.call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, "-m", "SpatialBiologyToolkit.project_gui"])
        self.assertEqual(command[-3:], ["--project", "example", "--read-only"])


if __name__ == "__main__":
    unittest.main()
