from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from SpatialBiologyToolkit.config import (
    ConfigModel,
    GeneralConfig,
    config_field,
    config_section,
    generate_markdown_for_model,
    iter_config_docs,
    write_config_docs,
)


@config_section("fallback_demo")
class DocumentationDemoConfig(ConfigModel):
    explicit: int = config_field(
        2,
        description="Explicitly documented value.",
        level="basic",
        stage="explicit_stage",
        ui_group="Explicit group",
        advice="Keep this non-negative.",
        ge=0,
    )
    fallback: str = "plain"
    values: list[str] = config_field(
        description="Values created independently for each model.",
        stage="explicit_stage",
        ui_group="Explicit group",
        default_factory=list,
    )


class ConfigDocumentationTests(unittest.TestCase):
    def test_config_field_stores_supported_metadata_and_constraints(self):
        field = DocumentationDemoConfig.model_fields["explicit"]

        self.assertEqual(field.description, "Explicitly documented value.")
        self.assertEqual(
            field.json_schema_extra,
            {
                "level": "basic",
                "stage": "explicit_stage",
                "ui_group": "Explicit group",
                "advice": "Keep this non-negative.",
            },
        )
        with self.assertRaises(ValidationError):
            DocumentationDemoConfig(explicit=-1)

        first = DocumentationDemoConfig()
        second = DocumentationDemoConfig()
        first.values.append("one")
        self.assertEqual(second.values, [])

    def test_config_section_adds_fallback_without_overwriting_explicit_metadata(self):
        explicit = DocumentationDemoConfig.model_fields["explicit"]
        fallback = DocumentationDemoConfig.model_fields["fallback"]

        self.assertEqual(explicit.json_schema_extra["stage"], "explicit_stage")
        self.assertEqual(explicit.json_schema_extra["ui_group"], "Explicit group")
        self.assertEqual(fallback.json_schema_extra["level"], "advanced")
        self.assertEqual(fallback.json_schema_extra["stage"], "fallback_demo")
        self.assertEqual(fallback.json_schema_extra["ui_group"], "Fallback Demo")
        self.assertEqual(fallback.json_schema_extra["advice"], "")
        self.assertIn("fallback", fallback.description)

    def test_general_config_documentation_extraction(self):
        records = {record.name: record for record in iter_config_docs(GeneralConfig)}

        self.assertEqual(records["imc_files_folder"].annotation, "str")
        self.assertEqual(records["imc_files_folder"].default, "IMC_files")
        self.assertEqual(records["imc_files_folder"].level, "basic")
        self.assertEqual(records["imc_files_folder"].stage, "general")
        self.assertEqual(records["imc_files_folder"].ui_group, "Input folders")
        self.assertIn("raw IMC", records["imc_files_folder"].description)
        self.assertIn("primary folder", records["imc_files_folder"].advice)

        self.assertEqual(records["anndata_uns_log_key"].level, "expert")
        self.assertEqual(records["masks_folder"].ui_group, "General")

    def test_markdown_generation_contains_documented_metadata(self):
        markdown = generate_markdown_for_model(GeneralConfig)

        self.assertIn("# General", markdown)
        self.assertIn("## Input folders", markdown)
        self.assertIn("### `imc_files_folder`", markdown)
        self.assertIn("- Type: `str`", markdown)
        self.assertIn("- Default: `IMC_files`", markdown)
        self.assertIn("- Level: `basic`", markdown)
        self.assertIn("Folder containing raw IMC input files", markdown)
        self.assertIn("Advice:", markdown)
        self.assertIn("Use this as the primary folder", markdown)

    def test_write_config_docs_writes_selected_sections(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            written = write_config_docs(
                temp_dir,
                models={"general": GeneralConfig},
            )
            output = Path(temp_dir) / "general.md"

            self.assertEqual(written, [output])
            self.assertTrue(output.is_file())
            self.assertIn(
                "### `population_obs_primary`",
                output.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
