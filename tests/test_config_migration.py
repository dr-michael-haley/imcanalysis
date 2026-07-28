import tempfile
import unittest
from pathlib import Path

import yaml
from typer.testing import CliRunner

from SpatialBiologyToolkit.cli.main import app
from SpatialBiologyToolkit.config import PipelineConfig, load_config
from SpatialBiologyToolkit.config.migration import (
    compact_config_data,
    write_compact_config,
)


class ConfigMigrationTests(unittest.TestCase):
    def test_verbose_config_is_compacted_and_aliases_are_canonicalized(self):
        source_data = PipelineConfig().model_dump(mode="python")
        source_data["general"]["anndata_path"] = "cohort.h5ad"
        source_data["general"]["legacy_custom_value"] = "preserved"
        source_data["batch_integration"]["batch_correction_method"] = "BBKNN"
        source_data["biobatchnet"]["biobatchnet_latent_dim"] = 12
        source_data["legacy_section"] = {"enabled": True}

        compact, unknown_keys = compact_config_data(source_data)

        self.assertEqual(
            compact["general"],
            {
                "anndata_path": "cohort.h5ad",
                "legacy_custom_value": "preserved",
            },
        )
        self.assertEqual(
            compact["batch_integration"],
            {"integration_method": "bbknn"},
        )
        self.assertEqual(
            compact["biobatchnet"]["biobatchnet_params"]["latent_dim"],
            12,
        )
        self.assertNotIn(
            "biobatchnet_latent_dim",
            compact["biobatchnet"],
        )
        self.assertEqual(compact["legacy_section"], {"enabled": True})
        self.assertEqual(
            unknown_keys,
            ("general.legacy_custom_value", "legacy_section"),
        )

    def test_writer_creates_new_default_path_and_preserves_source(self):
        source_text = (
            "general:\n"
            "  anndata_path: cohort.h5ad\n"
            "  outputs_folder: outputs\n"
            "logging:\n"
            "  level: INFO\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "config.yaml"
            source.write_text(source_text, encoding="utf-8")

            output, unknown_keys = write_compact_config(source)
            written = yaml.safe_load(output.read_text(encoding="utf-8"))
            reloaded = load_config(output)

            self.assertEqual(source.read_text(encoding="utf-8"), source_text)
            self.assertEqual(output, source.with_name("config.compact.yaml"))
            self.assertEqual(written, {"general": {"anndata_path": "cohort.h5ad"}})
            self.assertEqual(reloaded.general.anndata_path, "cohort.h5ad")
            self.assertEqual(unknown_keys, ())

    def test_writer_refuses_source_path_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "config.yaml"
            output = Path(temp_dir) / "new.yaml"
            source.write_text("{}\n", encoding="utf-8")
            output.write_text("existing: true\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "different from the source"):
                write_compact_config(source, source, force=True)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                write_compact_config(source, output)

    def test_cli_compact_writes_output_and_reports_invalid_input(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "verbose.yaml"
            output = Path(temp_dir) / "compact.yaml"
            source.write_text(
                "general:\n"
                "  anndata_path: cohort.h5ad\n"
                "  outputs_folder: outputs\n",
                encoding="utf-8",
            )

            result = runner.invoke(
                app,
                ["config", "compact", str(source), "--output", str(output)],
            )
            missing = runner.invoke(
                app,
                ["config", "compact", str(Path(temp_dir) / "missing.yaml")],
            )
            missing_project = runner.invoke(
                app,
                ["config", "compact", "--project", temp_dir],
            )

            self.assertEqual(result.exit_code, 0, result.stdout)
            self.assertIn("Wrote compact configuration", result.stdout)
            self.assertTrue(output.is_file())
            self.assertEqual(missing.exit_code, 2)
            self.assertIn("Configuration file not found", missing.stderr)
            self.assertEqual(missing_project.exit_code, 2)
            self.assertIn("project.yaml", missing_project.stderr)
            self.assertIn("sbt project adopt", missing_project.stderr)


if __name__ == "__main__":
    unittest.main()
