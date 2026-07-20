import tempfile
import unittest
from pathlib import Path

from SpatialBiologyToolkit.config import PipelineConfig, load_config_data
from SpatialBiologyToolkit.scripts.config_and_utils import (
    generate_default_config_dict,
    load_config as load_legacy_config,
)


class ConfigDefaultsTests(unittest.TestCase):
    def test_pipeline_defaults_match_legacy_default_dictionary(self):
        expected = generate_default_config_dict()
        resolved = PipelineConfig().model_dump(mode="python")

        self.assertEqual(resolved, expected)
        self.assertEqual(len(resolved), 21)
        self.assertEqual(resolved["population_embedding_qc"]["mode"], "auto")

    def test_sparse_data_fills_missing_sections_and_fields(self):
        config = load_config_data(
            {"createmasks": {"cellpose_cell_diameter": 30}}
        )

        self.assertEqual(config.createmasks.cellpose_cell_diameter, 30.0)
        self.assertEqual(config.general.anndata_path, "anndata.h5ad")
        self.assertEqual(config.preprocess.minimum_roi_dimensions, 200)
        self.assertEqual(config.logging.level, "INFO")

    def test_mutable_defaults_are_isolated(self):
        first = PipelineConfig()
        second = PipelineConfig()

        first.denoising.channels.append("CD3")
        self.assertEqual(second.denoising.channels, [])

    def test_complete_default_config_round_trips_through_validation(self):
        defaults = generate_default_config_dict()
        resolved = load_config_data(defaults).model_dump(mode="python")

        self.assertEqual(resolved, defaults)

    def test_legacy_loader_still_creates_complete_dictionary_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            config = load_legacy_config(str(path))

            self.assertTrue(path.is_file())
            self.assertEqual(config, generate_default_config_dict())


if __name__ == "__main__":
    unittest.main()
