from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import imageio.v3 as iio
import matplotlib
import numpy as np
import pandas as pd
import tifffile

from SpatialBiologyToolkit.spatialdata import (
    CellMasks,
    HistologyImages,
    IMCAnnData,
    IMCImages,
    MaxFuseSCRNASeq,
    RegionLabels,
    SpatialDataSpec,
    add_modality,
    create_spatialdata,
    get_label_annotations,
    get_roi_elements,
    get_roi_label_elements,
    get_roi_modalities,
    plan_spatialdata,
    plot_spatialdata_cells,
    plot_spatialdata_roi,
    summarize_spatialdata,
    write_spatialdata,
)


matplotlib.use("Agg")


def _write_project(root: Path) -> tuple[ad.AnnData, ad.AnnData]:
    masks = root / "masks"
    immune = root / "immune"
    ecm = root / "ecm"
    histology = root / "histology"
    regions = root / "regions"
    for folder in (masks, histology, regions):
        folder.mkdir(parents=True)
    for roi in ("ROI 1", "ROI 2"):
        (immune / roi).mkdir(parents=True)
        (ecm / roi).mkdir(parents=True)

    mask_one = np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 2, 2],
            [0, 0, 0, 0, 2, 2],
            [3, 3, 0, 0, 0, 0],
            [3, 3, 0, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    mask_two = np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 2, 2],
            [0, 0, 0, 0, 2, 2],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    for index, (roi, mask) in enumerate(
        (("ROI 1", mask_one), ("ROI 2", mask_two)), start=1
    ):
        tifffile.imwrite(masks / f"{roi}.tiff", mask)
        tifffile.imwrite(
            immune / roi / f"14{index}Pr_CD3.tiff",
            np.arange(30, dtype=np.float32).reshape(5, 6) + index,
        )
        tifffile.imwrite(
            immune / roi / "CD31.tiff",
            np.full((5, 6), index + 1, dtype=np.uint16),
        )
        tifffile.imwrite(
            ecm / roi / "panel_Collagen.tiff",
            np.full((5, 6), index + 3, dtype=np.uint16),
        )
        iio.imwrite(
            histology / f"{roi}.png",
            np.full((5, 6, 3), 20 * index, dtype=np.uint8),
        )
        region_values = np.where(mask > 0, 1, 0).astype(np.uint8)
        region_values[:, 4:] = np.where(mask[:, 4:] > 0, 2, 0)
        tifffile.imwrite(regions / f"{roi}_regions.tiff", region_values)

    obs = pd.DataFrame(
        {
            "ROI": ["ROI 1", "ROI 1", "ROI 2", "ROI 2"],
            "ObjectNumber": [1, 2, 1, 2],
            "X_loc": [1, 4, 1, 4],
            "Y_loc": [1, 1, 1, 1],
            "animal": ["a", "a", "b", "b"],
            "leiden_1.0": pd.Categorical(["0", "1", "0", "1"]),
        },
        index=["cell_1", "cell_2", "cell_3", "cell_4"],
    )
    imc = ad.AnnData(
        X=np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["CD3", "CD31"]),
    )
    maxfuse = ad.AnnData(
        X=np.array([[10, 0, 1], [0, 4, 2]], dtype=np.float32),
        obs=pd.DataFrame(
            {"atlas_cell": ["reference_a", "reference_b"]},
            index=["cell_1", "cell_4"],
        ),
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )
    return imc, maxfuse


def _complete_spec(root: Path, imc: ad.AnnData, maxfuse: ad.AnnData) -> SpatialDataSpec:
    return SpatialDataSpec(
        modalities=[
            CellMasks(name="cells", folder=root / "masks"),
            IMCImages(
                name="immune_images",
                panel_name="Immune panel",
                folder=root / "immune",
            ),
            IMCAnnData(
                name="immune_cells",
                panel_name="Immune panel",
                adata=imc,
                images="immune_images",
                masks="cells",
            ),
            IMCImages(
                name="ecm_images",
                panel_name="Extracellular panel",
                folder=root / "ecm",
                channels=["Collagen"],
                reference="cells",
            ),
            HistologyImages(
                name="he",
                folder=root / "histology",
                reference="cells",
            ),
            RegionLabels(
                name="tissue_regions",
                folder=root / "regions",
                suffix="_regions",
                value_names={1: "Tissue", 2: "Edge"},
                reference="cells",
            ),
            MaxFuseSCRNASeq(
                name="atlas",
                adata=maxfuse,
                imc_table="immune_cells",
            ),
        ],
        raster_chunks=(2, 3),
    )


class SpatialDataConstructionTests(unittest.TestCase):
    def test_plan_build_interrogate_plot_and_roundtrip_all_modalities(self):
        import matplotlib.pyplot as plt
        from spatialdata import read_zarr

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            imc, maxfuse = _write_project(root)
            original_columns = list(imc.obs.columns)
            spec = _complete_spec(root, imc, maxfuse)

            plan = plan_spatialdata(spec)
            self.assertTrue(plan.ok, plan.report.to_frame())
            self.assertEqual(plan.summary()["modalities"], 7)
            self.assertEqual(plan.summary()["images"], 6)
            self.assertEqual(plan.summary()["labels"], 4)
            self.assertEqual(plan.summary()["tables"], 3)

            sdata = create_spatialdata(spec, plan=plan)
            self.assertEqual(list(imc.obs.columns), original_columns)
            self.assertEqual(len(sdata.images), 6)
            self.assertEqual(len(sdata.labels), 4)
            self.assertEqual(len(sdata.points), 2)
            self.assertEqual(
                list(sdata.tables),
                ["table_immune_cells", "table_tissue_regions", "table_atlas"],
            )
            self.assertEqual(sdata.tables["table_atlas"].n_obs, 2)
            self.assertEqual(
                sdata.tables["table_atlas"].obs_names.tolist(),
                ["cell_1", "cell_4"],
            )

            modalities = get_roi_modalities(sdata, "roi 1")
            self.assertEqual(
                set(modalities["images"]),
                {"immune_images", "ecm_images", "he"},
            )
            self.assertEqual(
                get_roi_elements(
                    sdata, "ROI 1", image_modality="ecm_images"
                )["image"],
                "image_ecm_images_ROI_1",
            )
            self.assertEqual(
                get_roi_label_elements(sdata, "ROI 1"),
                {
                    "cells": "labels_cells_ROI_1",
                    "tissue_regions": "labels_tissue_regions_ROI_1",
                },
            )
            annotations = get_label_annotations(
                sdata, "tissue_regions", roi="ROI 1"
            )
            self.assertEqual(
                annotations["label_name"].astype(str).tolist(),
                ["Tissue", "Edge"],
            )

            summary = summarize_spatialdata(
                sdata,
                population_key="leiden_1.0",
                case_key="animal",
            )
            self.assertEqual(summary["cells"], 4)
            self.assertEqual(summary["rois"], 2)
            self.assertEqual(summary["unannotated_mask_instances"], 1)

            ax = plot_spatialdata_roi(
                sdata,
                "ROI 1",
                channel="Collagen",
                image_modality="ecm_images",
                label_layer="tissue_regions",
            )
            self.assertEqual(ax.get_legend().get_title().get_text(), "label_name")
            plt.close(ax.figure)

            figure, axes = plot_spatialdata_cells(
                sdata,
                ["cell_1", "cell_2"],
                channel="CD3",
                crop_size=(5, 6),
                outline_target_only=True,
                mask_outside_target=True,
                show_ax_titles=False,
            )
            self.assertEqual(len(axes), 2)
            self.assertEqual(axes[0].get_title(), "")
            plt.close(figure)

            output = write_spatialdata(sdata, root / "multimodal.zarr")
            restored = read_zarr(output)
            self.assertEqual(
                restored.attrs["spatial_biology_toolkit"]["schema_version"], 3
            )
            self.assertEqual(len(restored.images), 6)
            self.assertEqual(restored.tables["table_atlas"].n_vars, 3)

    def test_multiple_quantified_panels_require_explicit_relationships(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            imc, _maxfuse = _write_project(root)
            ecm_table = ad.AnnData(
                X=np.ones((4, 1), dtype=np.float32),
                obs=imc.obs.copy(),
                var=pd.DataFrame(index=["Collagen"]),
            )
            spec = SpatialDataSpec(
                [
                    CellMasks("cells", root / "masks"),
                    IMCImages(
                        "immune_images", "Immune", root / "immune"
                    ),
                    IMCImages(
                        "ecm_images",
                        "ECM",
                        root / "ecm",
                        reference="cells",
                    ),
                    IMCAnnData(
                        "immune_cells",
                        "Immune",
                        imc,
                        "immune_images",
                        "cells",
                    ),
                    IMCAnnData(
                        "ecm_cells",
                        "ECM",
                        ecm_table,
                        "ecm_images",
                        "cells",
                    ),
                ]
            )
            plan = plan_spatialdata(spec)
            plan.raise_for_errors()
            sdata = create_spatialdata(plan)
            self.assertEqual(
                set(sdata.tables), {"table_immune_cells", "table_ecm_cells"}
            )
            self.assertEqual(
                list(sdata.images["image_ecm_images_ROI_1"].coords["c"].values),
                ["Collagen"],
            )

    def test_validation_report_collects_cross_modality_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            imc, _maxfuse = _write_project(root)
            bad = imc.copy()
            bad.obs.loc["cell_1", "ObjectNumber"] = 99
            spec = SpatialDataSpec(
                [
                    CellMasks("cells", root / "masks"),
                    IMCImages("images", "Panel", root / "immune"),
                    IMCAnnData(
                        "table", "Panel", bad, "images", "cells"
                    ),
                ]
            )
            plan = plan_spatialdata(spec)
            self.assertFalse(plan.ok)
            self.assertIn("absent from mask", plan.report.errors[0].message)
            with self.assertRaisesRegex(ValueError, "SpatialData planning found"):
                create_spatialdata(plan)

    def test_maxfuse_must_be_a_subset_of_linked_imc_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            imc, maxfuse = _write_project(root)
            maxfuse.obs_names = ["cell_1", "not_an_imc_cell"]
            spec = _complete_spec(root, imc, maxfuse)
            plan = plan_spatialdata(spec)
            self.assertFalse(plan.ok)
            self.assertTrue(
                any("absent from linked IMC" in item.message for item in plan.report.errors)
            )

    def test_add_modality_is_transactional_and_optionally_inplace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            imc, maxfuse = _write_project(root)
            base_spec = SpatialDataSpec(
                [
                    CellMasks("cells", root / "masks"),
                    IMCImages("images", "Panel", root / "immune"),
                    IMCAnnData(
                        "cells_table", "Panel", imc, "images", "cells"
                    ),
                ]
            )
            sdata = create_spatialdata(base_spec)

            updated = add_modality(
                sdata,
                HistologyImages("he", root / "histology", "cells"),
            )
            self.assertNotIn("image_he_ROI_1", sdata.images)
            self.assertIn("image_he_ROI_1", updated.images)

            with_atlas = add_modality(
                sdata,
                MaxFuseSCRNASeq("atlas", maxfuse, "cells_table"),
            )
            self.assertNotIn("table_atlas", sdata.tables)
            self.assertEqual(with_atlas.tables["table_atlas"].n_obs, 2)

            result = add_modality(
                sdata,
                RegionLabels(
                    "regions",
                    root / "regions",
                    "_regions",
                    {1: "Tissue", 2: "Edge"},
                    "cells",
                ),
                inplace=True,
            )
            self.assertIs(result, sdata)
            self.assertIn("labels_regions_ROI_1", sdata.labels)
            self.assertIn("table_regions", sdata.tables)

    def test_histology_matching_rejects_ambiguous_extensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            imc, _maxfuse = _write_project(root)
            iio.imwrite(
                root / "histology" / "ROI 1.jpg",
                np.zeros((5, 6, 3), dtype=np.uint8),
            )
            spec = SpatialDataSpec(
                [
                    CellMasks("cells", root / "masks"),
                    IMCImages("images", "Panel", root / "immune"),
                    IMCAnnData(
                        "cells_table", "Panel", imc, "images", "cells"
                    ),
                    HistologyImages("he", root / "histology", "cells"),
                ]
            )
            plan = plan_spatialdata(spec)
            self.assertFalse(plan.ok)
            self.assertTrue(
                any("Multiple files match ROI" in item.message for item in plan.report.errors)
            )

    def test_partial_reference_modalities_are_discovered_and_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            imc, _maxfuse = _write_project(root)
            partial_histology = root / "partial_histology"
            partial_regions = root / "partial_regions"
            partial_ecm = root / "partial_ecm"
            partial_histology.mkdir()
            partial_regions.mkdir()
            (partial_ecm / "ROI 1").mkdir(parents=True)
            iio.imwrite(
                partial_histology / "ROI 1.png",
                np.ones((5, 6, 3), dtype=np.uint8),
            )
            tifffile.imwrite(
                partial_regions / "ROI 1_regions.tiff",
                np.where(
                    tifffile.imread(root / "masks" / "ROI 1.tiff") > 0,
                    1,
                    0,
                ).astype(np.uint8),
            )
            tifffile.imwrite(
                partial_ecm / "ROI 1" / "Collagen.tiff",
                np.ones((5, 6), dtype=np.uint16),
            )
            spec = SpatialDataSpec(
                [
                    CellMasks("cells", root / "masks"),
                    IMCImages("images", "Panel", root / "immune"),
                    IMCAnnData(
                        "cells_table", "Panel", imc, "images", "cells"
                    ),
                    IMCImages(
                        "partial_ecm",
                        "ECM",
                        partial_ecm,
                        channels=["Collagen"],
                        reference="cells",
                        allow_partial=True,
                    ),
                    HistologyImages(
                        "partial_he",
                        partial_histology,
                        "cells",
                        allow_partial=True,
                    ),
                    RegionLabels(
                        "partial_regions",
                        partial_regions,
                        "_regions",
                        {1: "Tissue"},
                        "cells",
                        allow_partial=True,
                    ),
                ]
            )
            plan = plan_spatialdata(spec)
            self.assertTrue(plan.ok, plan.report.to_frame())
            self.assertEqual(plan.modality("partial_ecm").rois, ("ROI 1",))
            self.assertEqual(plan.modality("partial_he").rois, ("ROI 1",))
            self.assertEqual(plan.modality("partial_regions").rois, ("ROI 1",))
            self.assertEqual(
                sum(
                    issue.code == "partial_roi_coverage"
                    for issue in plan.report.warnings
                ),
                3,
            )

    def test_invalid_anndata_attribute_names_are_planned_and_sanitized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            imc, _maxfuse = _write_project(root)
            imc.obs["Prediction Accuracy"] = [0.9, 0.8, 0.7, 0.6]
            imc.obs["*Background*"] = [1, 2, 3, 4]
            imc.uns["leiden,cluster_colors"] = ["#000000", "#ffffff"]
            original_obs_columns = list(imc.obs.columns)
            original_uns_keys = list(imc.uns)
            spec = SpatialDataSpec(
                [
                    CellMasks("cells", root / "masks"),
                    IMCImages("images", "Panel", root / "immune"),
                    IMCAnnData("cells_table", "Panel", imc, "images", "cells"),
                ]
            )

            plan = plan_spatialdata(spec)
            self.assertTrue(plan.ok, plan.report.to_frame())
            warnings = [
                issue
                for issue in plan.report.warnings
                if issue.code == "table_names_will_be_sanitized"
            ]
            self.assertEqual(len(warnings), 1)
            self.assertIn("3 AnnData attribute name(s)", warnings[0].message)

            sdata = create_spatialdata(plan)
            table = sdata.tables["table_cells_table"]
            self.assertIn("Prediction_Accuracy", table.obs.columns)
            self.assertIn("_Background_", table.obs.columns)
            self.assertIn("leiden_cluster_colors", table.uns)
            changes = table.uns["spatial_biology_toolkit"]["table_name_sanitization"]
            self.assertEqual(
                {
                    (item["attribute"], item["original"], item["sanitized"])
                    for item in changes
                },
                {
                    ("obs", "Prediction Accuracy", "Prediction_Accuracy"),
                    ("obs", "*Background*", "_Background_"),
                    (
                        "uns",
                        "leiden,cluster_colors",
                        "leiden_cluster_colors",
                    ),
                },
            )
            self.assertEqual(list(imc.obs.columns), original_obs_columns)
            self.assertEqual(list(imc.uns), original_uns_keys)

    def test_empty_region_label_raster_is_kept_without_table_region(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            imc, _maxfuse = _write_project(root)
            tifffile.imwrite(
                root / "regions" / "ROI 2_regions.tiff",
                np.zeros((5, 6), dtype=np.uint8),
            )
            spec = SpatialDataSpec(
                [
                    CellMasks("cells", root / "masks"),
                    IMCImages("images", "Panel", root / "immune"),
                    IMCAnnData("cells_table", "Panel", imc, "images", "cells"),
                    RegionLabels(
                        "regions",
                        root / "regions",
                        "_regions",
                        {1: "Tissue", 2: "Edge"},
                        "cells",
                    ),
                ]
            )

            plan = plan_spatialdata(spec)
            self.assertTrue(plan.ok, plan.report.to_frame())
            self.assertTrue(
                any(
                    issue.code == "empty_region_labels" and issue.roi == "ROI 2"
                    for issue in plan.report.warnings
                )
            )

            sdata = create_spatialdata(plan)
            self.assertIn("labels_regions_ROI_2", sdata.labels)
            table = sdata.tables["table_regions"]
            self.assertEqual(
                table.obs["_sbt_region"].cat.categories.tolist(),
                ["labels_regions_ROI_1"],
            )
            self.assertTrue(
                get_label_annotations(sdata, "regions", roi="ROI 2").empty
            )


if __name__ == "__main__":
    unittest.main()
