from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import tifffile
import yaml

from SpatialBiologyToolkit.config import PipelineConfig, SpatialDataConfig
from SpatialBiologyToolkit.pipeline.assets import resolve_assets
from SpatialBiologyToolkit.pipeline.planner import build_run_plan
from SpatialBiologyToolkit.pipeline.project import initialize_project
from SpatialBiologyToolkit.pipeline.registry import MODES, STAGE_REGISTRY
from SpatialBiologyToolkit.scripts.spatialdata_builder import run_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_minimal_assets(root: Path) -> None:
    masks = root / "masks"
    images = root / "images" / "ROI 1"
    masks.mkdir(parents=True)
    images.mkdir(parents=True)
    mask = np.array([[0, 1], [2, 0]], dtype=np.uint16)
    tifffile.imwrite(masks / "ROI 1.tiff", mask)
    tifffile.imwrite(images / "CD3.tiff", np.ones((2, 2), dtype=np.uint16))
    adata = ad.AnnData(
        X=np.array([[1.0], [2.0]], dtype=np.float32),
        obs=pd.DataFrame(
            {
                "ROI": ["ROI 1", "ROI 1"],
                "ObjectNumber": [1, 2],
                "X_loc": [1, 0],
                "Y_loc": [0, 1],
            },
            index=["cell_1", "cell_2"],
        ),
        var=pd.DataFrame(index=["CD3"]),
    )
    adata.write_h5ad(root / "cells.h5ad")


class SpatialDataPipelineTests(unittest.TestCase):
    def test_config_defaults_and_validation(self):
        settings = SpatialDataConfig()
        self.assertEqual(settings.action, "plan")
        self.assertEqual(settings.output_path, "spatialdata.zarr")
        self.assertTrue(settings.discover_unlisted_assets)
        with self.assertRaisesRegex(ValueError, "must end with .zarr"):
            SpatialDataConfig(output_path="spatialdata")
        with self.assertRaisesRegex(ValueError, "unique case-insensitively"):
            SpatialDataConfig(primary_images_name="cells")

    def test_registry_environment_assets_wrapper_and_docs_align(self):
        stage = STAGE_REGISTRY["spatialdata"]
        self.assertEqual(stage.catalogue_order, 39)
        self.assertEqual(stage.environment_keys, ["segmentation"])
        self.assertEqual(stage.config_sections, ["general", "spatialdata"])
        self.assertEqual(stage.requires_assets, [])
        self.assertEqual(
            stage.produces_assets,
            ["spatialdata_zarr", "human_outputs"],
        )
        self.assertTrue(all("spatialdata" not in mode.stages for mode in MODES))
        self.assertTrue((REPO_ROOT / stage.documentation_path).is_file())
        wrapper = (REPO_ROOT / stage.slurm_script).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=8", wrapper)
        self.assertIn("#SBATCH --mem=128G", wrapper)
        self.assertIn("#@ENV:  imc_segmentation", wrapper)
        self.assertIn(
            "SpatialBiologyToolkit.scripts.spatialdata_builder",
            wrapper,
        )
        pip_extras = (
            REPO_ROOT
            / "HPC_env_files"
            / "imc_segmentation"
            / "pip-extras.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("spatialdata>=0.7,<0.8", pip_extras)

        config = PipelineConfig()
        assets = {asset.role: asset for asset in resolve_assets(config, Path("."))}
        self.assertEqual(
            assets["spatialdata_zarr"].path,
            (Path(".") / "spatialdata.zarr").resolve(),
        )

    def test_temporary_project_plan_needs_no_fixed_upstream_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = initialize_project(Path(temporary) / "project")
            plan = build_run_plan(context, ["spatialdata"])
            self.assertTrue(plan.ready, plan.errors)
            self.assertEqual(
                [stage.name for stage in plan.resolved_stages],
                ["spatialdata"],
            )

    def test_direct_plan_execution_writes_report_without_zarr(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_minimal_assets(root)
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "general": {
                            "roi_obs": "ROI",
                            "x_coord_obs": "X_loc",
                            "y_coord_obs": "Y_loc",
                        },
                        "spatialdata": {
                            "action": "plan",
                            "root": ".",
                            "anndata_path": "cells.h5ad",
                            "cell_masks_folder": "masks",
                            "primary_images_folder": "images",
                            "output_path": "assembled.zarr",
                            "discover_unlisted_assets": False,
                        },
                        "logging": {
                            "console_only": True,
                            "to_console": False,
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch.dict(
                    os.environ,
                    {
                        "SBT_CONFIG": str(config_path),
                        "SBT_PROJECT_ROOT": str(root),
                    },
                    clear=False,
                ), patch(
                    "SpatialBiologyToolkit.reporting.bootstrap_stage_reporting",
                    return_value=None,
                ):
                    self.assertEqual(run_pipeline([]), 0)
            finally:
                os.chdir(previous)
            self.assertFalse((root / "assembled.zarr").exists())
            self.assertTrue(
                (
                    root
                    / "spatialdata_report"
                    / "summaries"
                    / "spatialdata_plan_summary.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    root
                    / "spatialdata_report"
                    / "tables"
                    / "spatialdata_asset_selections.csv"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
