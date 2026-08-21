import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from SpatialBiologyToolkit.scripts._visualization_paths import (
    prepare_visualization_report_paths,
)


class VisualizationPathTests(unittest.TestCase):
    def test_managed_visualizations_stay_inside_execution_figures_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            stage_output = root / "outputs" / "007_Visualisation"
            with patch.dict(
                os.environ,
                {
                    "SBT_STAGE": "vis",
                    "SBT_PROJECT_ROOT": str(root),
                    "SBT_STAGE_OUTPUT_DIR": str(stage_output),
                },
                clear=True,
            ):
                paths = prepare_visualization_report_paths(
                    root / "outputs" / "BasicProcess_QC"
                )

            self.assertEqual(paths.root, stage_output / "figures")
            self.assertEqual(paths.umaps, stage_output / "figures" / "UMAPs")
            self.assertEqual(
                paths.matrixplots,
                stage_output / "figures" / "Matrixplots",
            )
            self.assertEqual(
                paths.color_legends,
                stage_output / "figures" / "Color_legends",
            )
            self.assertEqual(
                paths.population_images,
                stage_output / "figures" / "Population_images",
            )
            self.assertTrue(all(path.is_dir() for path in paths.__dict__.values()))
            self.assertFalse((root / "outputs" / "UMAPs").exists())

    def test_no_reporting_context_preserves_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fallback = Path(temp_dir) / "QC" / "BasicProcess_QC"
            with patch.dict(os.environ, {}, clear=True):
                paths = prepare_visualization_report_paths(fallback)

            self.assertEqual(paths.root, fallback)
            self.assertEqual(paths.umaps, fallback / "UMAPs")
            self.assertTrue(paths.umaps.is_dir())


if __name__ == "__main__":
    unittest.main()
