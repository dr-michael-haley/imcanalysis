import json
import tempfile
import unittest
from pathlib import Path

from SpatialBiologyToolkit.config.schema import (
    generate_json_schema,
    write_json_schema,
)
from SpatialBiologyToolkit.config.models import DEFAULT_CONFIG_CLASSES


class ConfigSchemaTests(unittest.TestCase):
    def test_schema_contains_all_sections_and_field_metadata(self):
        schema = generate_json_schema()

        self.assertEqual(len(schema["properties"]), 27)
        self.assertIn("maxfuse", schema["properties"])
        self.assertIn("spatialdata", schema["properties"])
        self.assertIn("neighbour_signal", schema["properties"])
        self.assertIn("general", schema["properties"])
        self.assertIn("population_embedding_qc", schema["properties"])
        self.assertIn("napari_sbt", schema["properties"])
        diameter = schema["$defs"]["CreateMasksConfig"]["properties"][
            "cellpose_cell_diameter"
        ]
        self.assertEqual(diameter["level"], "basic")
        self.assertEqual(diameter["stage"], "createmasks")
        self.assertEqual(diameter["ui_group"], "Segmentation")
        self.assertIn("description", diameter)
        self.assertIn("advice", diameter)

        generic = schema["$defs"]["GeneralConfig"]["properties"]["anndata_path"]
        self.assertEqual(generic["stage"], "general")
        self.assertEqual(generic["level"], "basic")

        metadata_keys = {"description", "level", "stage", "ui_group", "advice"}
        allowed_schema_extra = {"level", "stage", "ui_group", "advice"}
        for model in DEFAULT_CONFIG_CLASSES.values():
            properties = schema["$defs"][model.__name__]["properties"]
            for field_name, field_schema in properties.items():
                with self.subTest(model=model.__name__, field=field_name):
                    self.assertTrue(metadata_keys.issubset(field_schema))
                    model_field = model.model_fields[field_name]
                    self.assertEqual(
                        set(model_field.json_schema_extra or {}),
                        allowed_schema_extra,
                    )

    def test_schema_can_be_written_as_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "schema" / "config.schema.json"
            write_json_schema(output)
            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(written["title"], "PipelineConfig")
        self.assertIn("visualization", written["properties"])


if __name__ == "__main__":
    unittest.main()
