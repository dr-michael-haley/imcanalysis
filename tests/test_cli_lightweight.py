import json
import subprocess
import sys
import unittest

from typer.testing import CliRunner

from SpatialBiologyToolkit.cli.main import app


class LightweightCliTests(unittest.TestCase):
    def test_cli_startup_does_not_import_heavy_analysis_dependencies(self):
        heavy_modules = [
            "anndata",
            "cellpose",
            "scanpy",
            "skimage",
            "squidpy",
            "tifffile",
            "torch",
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

    def test_expected_command_groups_are_exposed(self):
        result = CliRunner().invoke(app, ["--help"])

        self.assertEqual(result.exit_code, 0, result.stdout)
        for command in (
            "config",
            "project",
            "stages",
            "modes",
            "plan",
            "run",
            "status",
            "logs",
        ):
            self.assertIn(command, result.stdout)


if __name__ == "__main__":
    unittest.main()
