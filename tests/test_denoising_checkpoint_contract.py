import unittest
from pathlib import Path

from pydantic import ValidationError

from SpatialBiologyToolkit.config import DenoisingConfig
from SpatialBiologyToolkit.denoising_contract import (
    DEFAULT_WEIGHTS_NAME_TEMPLATE,
    resolve_weights_name,
)


class DenoisingCheckpointContractTests(unittest.TestCase):
    def test_active_stage_uses_the_shared_checkpoint_contract(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "SpatialBiologyToolkit"
            / "scripts"
            / "denoising.py"
        )
        source = source_path.read_text(encoding="utf-8")

        self.assertIn("resolve_weights_name(", source)
        self.assertIn(
            "validate_weights_name(weights_name, loading=is_load_weights)", source
        )

    def test_checkpoint_template_supports_supported_legacy_weight_files(self):
        self.assertEqual(
            resolve_weights_name(DEFAULT_WEIGHTS_NAME_TEMPLATE, "Y89_SMA"),
            "weights_Y89_SMA.weights.h5",
        )
        self.assertEqual(
            resolve_weights_name("weights_{channel}.hdf5", "Y89_SMA"),
            "weights_Y89_SMA.hdf5",
        )
        with self.assertRaisesRegex(ValueError, "\\{channel\\}"):
            resolve_weights_name("shared.weights.h5", "Y89_SMA")

    def test_config_rejects_values_the_package_does_not_support(self):
        with self.assertRaises(ValidationError):
            DenoisingConfig(loss_function="mae")
        with self.assertRaises(ValidationError):
            DenoisingConfig(network_size="large")
        with self.assertRaises(ValidationError):
            DenoisingConfig(weights_name_template="shared.weights.h5")

    def test_legacy_module_uses_the_current_package_interfaces(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "SpatialBiologyToolkit"
            / "denoising.py"
        )
        source = source_path.read_text(encoding="utf-8")

        self.assertIn(
            "from IMC_Denoise.DeepSNiF_utils.DeepSNiF_DataGenerator import ",
            source,
        )
        self.assertIn(
            "validate_weights_name(weights_name, loading=is_load_weights)", source
        )

    def test_tensorflow_environment_smoke_test_checks_the_checkpoint_contract(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "HPC_env_files"
            / "sbt-tensorflow"
            / "smoke_test.py"
        )
        source = source_path.read_text(encoding="utf-8")

        self.assertIn(
            "from IMC_Denoise.checkpoints import validate_weights_name", source
        )
        self.assertIn(
            "validate_weights_name(\"smoke.weights.h5\", loading=False)", source
        )


if __name__ == "__main__":
    unittest.main()
