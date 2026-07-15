import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from SpatialBiologyToolkit.config import load_config
from SpatialBiologyToolkit.config.export import write_resolved_config


class ConfigLoadTests(unittest.TestCase):
    def test_representative_existing_yaml_loads(self):
        source = """
general:
  anndata_path: cohort.h5ad
  population_obs_all: [population, leiden_1.0]
createmasks:
  cellpose_cell_diameter: 24
batch_integration:
  batch_correction_method: BBKNN
biobatchnet:
  biobatchnet_latent_dim: 12
visualization:
  create_umaps: false
pairwise_spatial:
  population_pairs:
    T_cells: [B_cells, Macrophages]
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            path.write_text(source, encoding="utf-8")
            config = load_config(path)

        self.assertEqual(config.general.anndata_path, "cohort.h5ad")
        self.assertEqual(config.createmasks.cellpose_cell_diameter, 24.0)
        self.assertEqual(config.batch_integration.integration_method, "bbknn")
        self.assertEqual(config.biobatchnet.biobatchnet_params["latent_dim"], 12)
        self.assertFalse(config.visualization.create_umaps)

    def test_invalid_value_has_field_location(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.yaml"
            path.write_text(
                "preprocess:\n  minimum_roi_dimensions: 0\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError) as context:
                load_config(path)

        message = str(context.exception)
        self.assertIn("preprocess.minimum_roi_dimensions", message)
        self.assertIn("greater than 0", message)

    def test_root_yaml_must_be_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.yaml"
            path.write_text("- general\n- preprocess\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mapping at its root"):
                load_config(path)

    def test_resolved_yaml_export_contains_all_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "user.yaml"
            output = Path(temp_dir) / "provenance" / "resolved.yaml"
            source.write_text(
                "createmasks:\n  cellpose_cell_diameter: 31\n",
                encoding="utf-8",
            )

            config = load_config(source)
            write_resolved_config(config, output)
            exported = yaml.safe_load(output.read_text(encoding="utf-8"))
            reloaded = load_config(output)

        self.assertEqual(exported["createmasks"]["cellpose_cell_diameter"], 31.0)
        self.assertEqual(exported["general"]["anndata_path"], "anndata.h5ad")
        self.assertIn("logging", exported)
        self.assertEqual(reloaded, config)

    def test_typed_loader_does_not_rewrite_sparse_user_yaml(self):
        source_text = "general:\n  anndata_path: sparse.h5ad\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "user.yaml"
            source.write_text(source_text, encoding="utf-8")

            load_config(source)

            self.assertEqual(source.read_text(encoding="utf-8"), source_text)


if __name__ == "__main__":
    unittest.main()
