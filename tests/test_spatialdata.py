from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import tifffile

from SpatialBiologyToolkit.spatialdata import (
    create_spatialdata,
    get_roi_elements,
    match_marker_image,
    plan_imc_spatialdata_conversion,
    plot_population_counts,
    plot_spatialdata_cells,
    plot_spatialdata_roi,
    summarize_spatialdata,
    write_spatialdata,
)


matplotlib.use("Agg")


def _write_fixture(root: Path, *, include_missing_instance: bool = False) -> ad.AnnData:
    images = root / "images" / "ROI 1"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir()

    tifffile.imwrite(
        images / "141Pr_CD3.tiff",
        np.arange(30, dtype=np.float32).reshape(5, 6),
    )
    tifffile.imwrite(
        images / "CD31.tiff",
        np.full((5, 6), 2, dtype=np.uint16),
    )
    mask = np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 2, 2],
            [0, 0, 0, 0, 2, 2],
            [3, 3, 0, 0, 0, 0],
            [3, 3, 0, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    tifffile.imwrite(masks / "ROI 1.tiff", mask)

    second_instance = 4 if include_missing_instance else 2
    return ad.AnnData(
        X=np.array([[1, 2], [3, 4]], dtype=np.float32),
        obs=pd.DataFrame(
            {
                "ROI": ["ROI 1", "ROI 1"],
                "ObjectNumber": [1, second_instance],
                "animal": ["mouse_a", "mouse_a"],
                "leiden_1.0": pd.Categorical(["0", "1"]),
            },
            index=["cell_1", "cell_2"],
        ),
        var=pd.DataFrame(index=["CD3", "CD31"]),
    )


class SpatialDataConversionTests(unittest.TestCase):
    def test_marker_matching_prefers_exact_then_unique_bounded_substring(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir) / "images"
            folder.mkdir()
            tifffile.imwrite(
                folder / "141Pr_CD3.tiff", np.zeros((2, 2), dtype=np.uint8)
            )
            tifffile.imwrite(folder / "CD31.tiff", np.zeros((2, 2), dtype=np.uint8))

            fallback = match_marker_image(folder, "CD3")
            self.assertEqual(fallback.path.name, "141Pr_CD3.tiff")
            self.assertEqual(fallback.mode, "substring")

            tifffile.imwrite(folder / "CD3.tif", np.zeros((2, 2), dtype=np.uint8))
            exact = match_marker_image(folder, "CD3")
            self.assertEqual(exact.path.name, "CD3.tif")
            self.assertEqual(exact.mode, "exact")

    def test_marker_matching_rejects_ambiguous_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir) / "images"
            folder.mkdir()
            for name in ("141Pr_CD3.tiff", "panel-CD3-extra.tiff"):
                tifffile.imwrite(folder / name, np.zeros((2, 2), dtype=np.uint8))
            with self.assertRaisesRegex(
                ValueError, "Multiple TIFFs contain the bounded marker"
            ):
                match_marker_image(folder, "CD3")

    def test_plan_allows_mask_only_instances_but_rejects_missing_table_instances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adata = _write_fixture(root)
            plan = plan_imc_spatialdata_conversion(
                adata, root / "images", root / "masks"
            )
            self.assertEqual(plan.n_rois, 1)
            self.assertEqual(plan.n_image_files, 2)
            self.assertEqual(plan.rois[0].unannotated_mask_instances, 1)

            missing_root = root / "missing"
            missing_adata = _write_fixture(missing_root, include_missing_instance=True)
            with self.assertRaisesRegex(ValueError, "absent from mask"):
                plan_imc_spatialdata_conversion(
                    missing_adata,
                    missing_root / "images",
                    missing_root / "masks",
                )

    def test_create_summarize_and_zarr_roundtrip(self):
        from spatialdata import read_zarr

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adata = _write_fixture(root)
            sdata = create_spatialdata(
                adata,
                root / "images",
                root / "masks",
                raster_chunks=(2, 3),
            )

            self.assertEqual(list(sdata.images), ["image_ROI_1"])
            self.assertEqual(list(sdata.labels), ["labels_ROI_1"])
            self.assertEqual(
                list(sdata.images["image_ROI_1"].coords["c"].values),
                ["CD3", "CD31"],
            )
            self.assertEqual(adata.obs["ROI"].tolist(), ["ROI 1", "ROI 1"])
            self.assertEqual(
                adata.obs["_sbt_region"].tolist(),
                ["labels_ROI_1", "labels_ROI_1"],
            )

            summary = summarize_spatialdata(
                sdata, population_key="leiden_1.0", case_key="animal"
            )
            self.assertEqual(summary["cells"], 2)
            self.assertEqual(summary["markers"], 2)
            self.assertEqual(summary["unannotated_mask_instances"], 1)
            self.assertEqual(summary["population_counts"], {"0": 1, "1": 1})
            self.assertEqual(summary["cells_per_case"], {"mouse_a": 2})

            original_replace = Path.replace
            replace_attempts = {"failures": 0}

            def fail_first_zarr_metadata_replace(source, target):
                if (
                    source.name.endswith(".partial")
                    and replace_attempts["failures"] == 0
                ):
                    replace_attempts["failures"] += 1
                    raise PermissionError("simulated transient Windows lock")
                return original_replace(source, target)

            with patch.object(Path, "replace", fail_first_zarr_metadata_replace):
                output = write_spatialdata(sdata, root / "example.zarr")
            self.assertEqual(replace_attempts["failures"], 1)
            restored = read_zarr(output)
            self.assertEqual(
                get_roi_elements(restored, "roi 1"),
                {
                    "image": "image_ROI_1",
                    "labels": "labels_ROI_1",
                    "coordinate_system": "roi_ROI_1",
                },
            )
            value = float(
                restored.images["image_ROI_1"].sel(c="CD3").data.compute().sum()
            )
            self.assertAlmostEqual(value, 435.0)
            with self.assertRaises(FileExistsError):
                write_spatialdata(sdata, output)

    def test_create_without_images_and_population_plot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adata = _write_fixture(root)
            sdata = create_spatialdata(adata, None, root / "masks")
            self.assertFalse(sdata.images)
            self.assertEqual(list(sdata.labels), ["labels_ROI_1"])

            ax = plot_population_counts(sdata, "leiden_1.0")
            self.assertEqual(ax.get_xlabel(), "Cells")
            self.assertEqual(ax.get_ylabel(), "leiden_1.0")

            roi_ax = plot_spatialdata_roi(sdata, "ROI 1", color="leiden_1.0")
            self.assertEqual(roi_ax.get_title(), "ROI 1")

    def test_cell_gallery_by_obs_name_and_roi_local_instance(self):
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adata = _write_fixture(root)
            sdata = create_spatialdata(
                adata,
                root / "images",
                root / "masks",
                raster_chunks=(2, 3),
            )

            figure, axes = plot_spatialdata_cells(
                sdata,
                ["cell_1", "cell_2"],
                channel=["CD3", "CD31"],
                color="leiden_1.0",
                crop_size=(4, 4),
                ncols=2,
            )
            self.assertEqual(len(axes), 2)
            self.assertIn("cell_1", axes[0].get_title())
            self.assertIn("ObjectNumber=1", axes[0].get_title())
            self.assertEqual(len(figure.legends), 1)
            plt.close(figure)

            figure, axes = plot_spatialdata_cells(
                sdata,
                2,
                cell_key="ObjectNumber",
                roi="roi 1",
                channel="CD3",
                crop_size=3,
            )
            self.assertEqual(len(axes), 1)
            self.assertIn("cell_2", axes[0].get_title())
            plt.close(figure)

            figure, axes = plot_spatialdata_cells(
                sdata,
                ["cell_1", "cell_2"],
                color="ObjectNumber",
                crop_size=4,
            )
            self.assertEqual(len(axes), 2)
            self.assertEqual(len(figure.axes), 3)  # Two cells plus one colorbar.
            plt.close(figure)

            with self.assertRaisesRegex(KeyError, "not present"):
                plot_spatialdata_cells(sdata, "missing_cell")

            sdata.tables["table"].obs["duplicate_id"] = "duplicate"
            with self.assertRaisesRegex(ValueError, "2 table rows"):
                plot_spatialdata_cells(
                    sdata,
                    "duplicate",
                    cell_key="duplicate_id",
                )

    def test_cell_gallery_without_images_and_validation(self):
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adata = _write_fixture(root)
            sdata = create_spatialdata(adata, None, root / "masks")

            figure, axes = plot_spatialdata_cells(
                sdata,
                "cell_1",
                crop_size=4,
            )
            self.assertEqual(len(axes), 1)
            self.assertGreaterEqual(len(axes[0].images), 2)
            plt.close(figure)

            with self.assertRaisesRegex(ValueError, "no image element"):
                plot_spatialdata_cells(sdata, "cell_1", channel="CD3")
            with self.assertRaisesRegex(ValueError, "At least one cell"):
                plot_spatialdata_cells(sdata, [])


if __name__ == "__main__":
    unittest.main()
