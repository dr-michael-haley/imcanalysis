from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
from pydantic import ValidationError
from tifffile import imwrite

from SpatialBiologyToolkit.cellvision import (
    ROIInput,
    _gallery_image,
    _normalized_uint16_image,
    aligned_marker_matrix,
    aligned_obsm_representation,
    categorical_entropy_explained,
    complete_normalization_dict,
    compute_normalization_dict,
    configuration_fingerprint,
    confusion_tables,
    continuous_cluster_explained_variance,
    fuse_connectivity_graphs,
    identity_fingerprint,
    leiden_key,
    load_normalization_dict,
    relabel_selected_mask,
    resolve_roi_channels,
    select_source_cells,
    validate_normalization_dict,
    write_scportrait_config,
)
from SpatialBiologyToolkit.cellvision_vicreg import (
    H5SCImageDataset,
    MaskSafeAugment,
    VICRegNetwork,
    validate_h5sc_unit_range,
    vicreg_loss,
)
from SpatialBiologyToolkit.config.models import CellVisionConfig, PipelineConfig
from SpatialBiologyToolkit.pipeline.assets import resolve_assets
from SpatialBiologyToolkit.pipeline.planner import build_run_plan
from SpatialBiologyToolkit.pipeline.project import initialize_project
from SpatialBiologyToolkit.pipeline.registry import get_mode, get_stage
from SpatialBiologyToolkit.pipeline.runs import create_run_record
from SpatialBiologyToolkit.pipeline.slurm import sbt_environment
from SpatialBiologyToolkit.scripts.cellvision_cluster import (
    _atomic_write_h5ad,
    _canonicalize_leiden_columns,
    _cellvision_umap_params,
    _register_fused_neighbors,
)
from SpatialBiologyToolkit.scripts.config_and_utils import read_h5ad_compat


class CellVisionIdentityTests(unittest.TestCase):
    def _adata(self) -> ad.AnnData:
        return ad.AnnData(
            X=np.zeros((4, 1), dtype=np.float32),
            obs=pd.DataFrame(
                {
                    "ROI": ["r1", "r1", "r2", "r2"],
                    "ObjectNumber": [1, 2, 1, 2],
                    "population": ["A", "B", "A", "C"],
                },
                index=["cell-a", "cell-b", "cell-c", "cell-d"],
            ),
        )

    def test_selection_preserves_source_and_mask_identity(self):
        selected = select_source_cells(
            self._adata(),
            roi_obs="ROI",
            object_id_obs="ObjectNumber",
            population_obs="population",
            populations=["A"],
        )

        self.assertEqual(selected["cellvision_id"].tolist(), ["cell-a", "cell-c"])
        self.assertEqual(selected["source_obs_position"].tolist(), [0, 2])
        self.assertEqual(selected["scportrait_cell_id"].tolist(), [1, 2])
        self.assertEqual(selected["extraction_status"].tolist(), ["requested", "requested"])

    def test_duplicate_roi_object_pair_is_rejected(self):
        data = self._adata()
        data.obs.loc["cell-b", "ObjectNumber"] = 1
        with self.assertRaisesRegex(ValueError, r"unique \(ROI, object ID\)"):
            select_source_cells(data, roi_obs="ROI", object_id_obs="ObjectNumber")

    def test_mask_relabel_drops_unselected_objects(self):
        selected = pd.DataFrame(
            {"ObjectNumber": [1, 3], "scportrait_cell_id": [10, 11]}
        )
        mask = np.array([[0, 1, 1], [2, 2, 3]], dtype=np.uint16)

        result = relabel_selected_mask(mask, selected, object_id_obs="ObjectNumber")

        np.testing.assert_array_equal(
            result,
            np.array([[0, 10, 10], [0, 0, 11]], dtype=np.uint64),
        )

    def test_identity_fingerprint_changes_with_marker_contract(self):
        selected = select_source_cells(
            self._adata(), roi_obs="ROI", object_id_obs="ObjectNumber"
        )
        first = identity_fingerprint(
            selected,
            roi_obs="ROI",
            object_id_obs="ObjectNumber",
            markers=["CD3"],
            image_size=36,
        )
        second = identity_fingerprint(
            selected,
            roi_obs="ROI",
            object_id_obs="ObjectNumber",
            markers=["CD3", "CD8"],
            image_size=36,
        )
        self.assertNotEqual(first, second)

    def test_configuration_fingerprint_is_order_stable_and_value_sensitive(self):
        first = configuration_fingerprint({"b": [1, 2], "a": {"x": 3}})
        reordered = configuration_fingerprint({"a": {"x": 3}, "b": [1, 2]})
        changed = configuration_fingerprint({"a": {"x": 4}, "b": [1, 2]})

        self.assertEqual(first, reordered)
        self.assertNotEqual(first, changed)

    def test_identity_fingerprint_includes_extraction_settings(self):
        selected = select_source_cells(
            self._adata(), roi_obs="ROI", object_id_obs="ObjectNumber"
        )
        first = identity_fingerprint(
            selected,
            roi_obs="ROI",
            object_id_obs="ObjectNumber",
            markers=["CD3"],
            image_size=36,
            extraction_parameters={"mask_expand_px": 0},
        )
        second = identity_fingerprint(
            selected,
            roi_obs="ROI",
            object_id_obs="ObjectNumber",
            markers=["CD3"],
            image_size=36,
            extraction_parameters={"mask_expand_px": 1},
        )

        self.assertNotEqual(first, second)


class CellVisionAnnDataCompatibilityTests(unittest.TestCase):
    def test_cluster_write_removes_null_rapids_metadata(self):
        data = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32))
        data.uns["X_cellvision_pca"] = {
            "params": {"mask_var": None, "n_comps": 2}
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "cellvision_clustered.h5ad"
            _atomic_write_h5ad(data, output)
            restored = ad.read_h5ad(output)

        self.assertNotIn(
            "mask_var", restored.uns["X_cellvision_pca"]["params"]
        )
        self.assertEqual(restored.uns["X_cellvision_pca"]["params"]["n_comps"], 2)

    def test_compat_reader_repairs_null_uns_encoding_and_retries(self):
        expected = object()
        error = RuntimeError(
            "No read method registered for IOSpec(encoding_type='null', "
            "encoding_version='0.1.0')"
        )
        path = Path("cellvision_clustered.h5ad")

        with (
            patch("anndata.read_h5ad", side_effect=[error, expected]) as read,
            patch(
                "SpatialBiologyToolkit.scripts.config_and_utils."
                "_remove_null_encoded_uns_entries_in_h5ad",
                return_value=["/uns/X_cellvision_pca/params/mask_var"],
            ) as repair,
        ):
            result = read_h5ad_compat(path)

        self.assertIs(result, expected)
        self.assertEqual(read.call_count, 2)
        repair.assert_called_once_with(path)


class CellVisionChannelAndConfigTests(unittest.TestCase):
    def test_scportrait_config_creates_its_cache_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "scportrait_config.yml"
            write_scportrait_config(
                config_path,
                image_size=36,
                threads=1,
            )

            self.assertTrue(config_path.is_file())
            self.assertTrue((Path(temp_dir) / "scportrait_cache").is_dir())

    def test_multichannel_resolution_preserves_requested_marker_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            roi = Path(temp_dir) / "ROI_1"
            roi.mkdir()
            imwrite(roi / "01_01_Nd143_CD3.tiff", np.ones((4, 4), dtype=np.uint16))
            imwrite(roi / "02_01_Sm149_CD45.tiff", np.ones((4, 4), dtype=np.uint16))

            files, names = resolve_roi_channels(roi, ["CD45", "CD3"])

        self.assertEqual(names, ("CD45", "CD3"))
        self.assertEqual([path.name for path in files], ["02_01_Sm149_CD45.tiff", "01_01_Nd143_CD3.tiff"])

    def test_channel_resolution_supports_full_suffixes_and_avoids_substrings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            roi = Path(temp_dir) / "ROI_1"
            roi.mkdir()
            imwrite(roi / "01_01_165Ho_CD11c.tiff", np.ones((4, 4), dtype=np.uint16))
            imwrite(roi / "02_01_174Yb_MHCII.ome.tif", np.ones((4, 4), dtype=np.uint16))
            imwrite(roi / "03_01_Nd143_CD3.tiff", np.ones((4, 4), dtype=np.uint16))
            imwrite(roi / "04_01_Gd160_CD31.tiff", np.ones((4, 4), dtype=np.uint16))

            files, names = resolve_roi_channels(
                roi,
                ["174yb_mhcii", "165Ho_CD11c", "CD3"],
            )

        self.assertEqual(names, ("174yb_mhcii", "165Ho_CD11c", "CD3"))
        self.assertEqual(
            [path.name for path in files],
            [
                "02_01_174Yb_MHCII.ome.tif",
                "01_01_165Ho_CD11c.tiff",
                "03_01_Nd143_CD3.tiff",
            ],
        )

    def test_channel_resolution_rejects_ambiguous_or_duplicate_aliases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            roi = Path(temp_dir) / "ROI_1"
            roi.mkdir()
            imwrite(roi / "01_01_Nd143_CD3.tiff", np.ones((4, 4), dtype=np.uint16))
            imwrite(roi / "02_01_Er167_CD3.tiff", np.ones((4, 4), dtype=np.uint16))

            with self.assertRaisesRegex(ValueError, "ambiguous"):
                resolve_roi_channels(roi, ["CD3"])
            with self.assertRaisesRegex(ValueError, "both resolve"):
                resolve_roi_channels(roi, ["Nd143_CD3", "01_01_Nd143_CD3"])

    def test_config_defaults_and_cross_field_validation(self):
        config = CellVisionConfig()
        self.assertEqual(config.image_size, 36)
        self.assertFalse(config.mask_gaussian_blur)
        self.assertEqual(config.normalization_quantile, 0.999)
        self.assertEqual(config.normalization_clip, [0.0, 1.0])
        self.assertEqual(config.leiden_resolutions, [0.2, 0.3, 0.5, 0.7, 1.0])
        self.assertEqual(config.augmentation_intensity_jitter, 0.2)
        self.assertEqual(config.augmentation_intensity_jitter_probability, 0.0)
        self.assertTrue(config.fusion_enabled)
        self.assertEqual(config.fusion_intensity_representation, "X_biobatchnet")
        self.assertEqual(config.fusion_intensity_weight, 0.5)
        with self.assertRaises(ValidationError):
            CellVisionConfig(populations=["T cell"])
        with self.assertRaises(ValidationError):
            CellVisionConfig(warmup_epochs=31, epochs=30)
        with self.assertRaises(ValidationError):
            CellVisionConfig(normalization_clip=[0, 2])
        with self.assertRaises(ValidationError):
            CellVisionConfig(fusion_intensity_weight=1.1)

    def test_nimbus_normalization_dict_accepts_string_values_and_exact_channels(self):
        values = validate_normalization_dict(
            {"CD3": "12.5", "CD20": 8, "unused": 99},
            channel_names=["CD3", "CD20"],
        )
        self.assertEqual(values, {"CD3": 12.5, "CD20": 8.0})
        with self.assertRaises(ValueError):
            validate_normalization_dict({"CD3": 1}, channel_names=["CD3", "CD20"])

        partial = validate_normalization_dict(
            {"CD3": 1},
            channel_names=["CD3", "CD20"],
            allow_missing=True,
        )
        self.assertEqual(partial, {"CD3": 1.0})

    def test_preferred_nimbus_normalization_csv_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "normalization_dict.csv"
            path.write_text(
                "marker,vmax,lower_threshold\nCD3,12.5,0.8\nCD20,8,0\n",
                encoding="utf-8",
            )
            values = load_normalization_dict(
                path,
                channel_names=["CD3", "CD20"],
            )
        self.assertEqual(values, {"CD3": 12.5, "CD20": 8.0})

    def test_nimbus_normalization_dict_resolves_unique_channel_suffixes(self):
        values = validate_normalization_dict(
            {"CD11c": "18", "MHCII": "210", "CD3": "8", "CD31": "10"},
            channel_names=["165Ho_CD11c", "174Yb_MHCII", "Nd143_CD3", "Gd160_CD31"],
        )
        self.assertEqual(
            values,
            {
                "165Ho_CD11c": 18.0,
                "174Yb_MHCII": 210.0,
                "Nd143_CD3": 8.0,
                "Gd160_CD31": 10.0,
            },
        )

    def test_normalization_exact_match_precedes_suffix_and_ambiguity_fails(self):
        values = validate_normalization_dict(
            {"165Ho_CD11c": "19", "CD11c": "18"},
            channel_names=["165ho_cd11c"],
        )
        self.assertEqual(values, {"165ho_cd11c": 19.0})
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            validate_normalization_dict(
                {"CD3": 8, "Nd143_CD3": 9},
                channel_names=["prefix_Nd143_CD3"],
            )

    def test_normalized_uint16_conversion_has_no_float_max_heuristic(self):
        first = _normalized_uint16_image(
            np.array([[0.0, 0.5, 0.99], [0.0, 0.5, 0.99]], dtype=np.float32),
            path=Path("first.tif"),
            normalization_value=2.0,
            clip_values=[0, 1],
        )
        second = _normalized_uint16_image(
            np.array([[0.0, 0.5, 1.01], [0.0, 0.5, 1.01]], dtype=np.float32),
            path=Path("second.tif"),
            normalization_value=2.0,
            clip_values=[0, 1],
        )
        self.assertEqual(int(first[0, 1]), int(second[0, 1]))
        self.assertGreater(int(second[0, 2]), int(first[0, 2]))

    def test_computed_normalization_uses_in_mask_quantile_and_nimbus_floor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "A.tif"
            mask_path = root / "ROI.tif"
            image = np.zeros((4, 4), dtype=np.float32)
            image[1, 1] = 1
            image[1, 2] = 3
            mask = np.zeros((4, 4), dtype=np.uint16)
            mask[1, 1:3] = 1
            imwrite(image_path, image)
            imwrite(mask_path, mask)
            context = ROIInput(
                name="ROI",
                channel_files=(image_path,),
                channel_names=("A",),
                mask_path=mask_path,
                spatial_shape=(4, 4),
            )
            values = compute_normalization_dict(
                [context], quantile=0.5, minimum_value=3.0, mask_expand_px=0
            )

        self.assertEqual(values, {"A": 3.0})

    def test_partial_normalization_dict_computes_only_missing_channel_fallbacks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mask_path = root / "ROI.tif"
            first_path = root / "A.tif"
            second_path = root / "DNA1.tif"
            mask = np.zeros((4, 4), dtype=np.uint16)
            mask[1, 1:3] = 1
            first = np.zeros((4, 4), dtype=np.float32)
            first[1, 1:3] = [100, 200]
            second = np.zeros((4, 4), dtype=np.float32)
            second[1, 1:3] = [2, 4]
            imwrite(mask_path, mask)
            imwrite(first_path, first)
            imwrite(second_path, second)
            context = ROIInput(
                name="ROI",
                channel_files=(first_path, second_path),
                channel_names=("A", "DNA1"),
                mask_path=mask_path,
                spatial_shape=(4, 4),
            )

            values, fallback_channels = complete_normalization_dict(
                {"A": "50"},
                [context],
                channel_names=["A", "DNA1"],
                quantile=0.5,
                minimum_value=1.0,
                mask_expand_px=0,
            )

        self.assertEqual(values, {"A": 50.0, "DNA1": 3.0})
        self.assertEqual(fallback_channels, ["DNA1"])

    def test_gallery_uses_stored_unit_range_without_autoscaling(self):
        image = np.array([[0.0, 0.001], [0.002, 0.0]], dtype=np.float32)
        displayed = _gallery_image(image, cell_id="cell", channel_name="CD3")
        np.testing.assert_array_equal(displayed, image)

    def test_registry_environment_mode_and_asset_role(self):
        full_stage = get_stage("cellvision-full")
        self.assertEqual(full_stage.environment_keys, ["scportrait", "rapids"])
        self.assertEqual(
            get_mode("cellvision").stages,
            [
                "cellvision-extract",
                "cellvision-embed",
                "cellvision-cluster",
                "cellvision-plot",
            ],
        )
        self.assertEqual(len(full_stage.python_modules), 4)
        self.assertEqual(
            get_stage("cellvision-extract").environment_keys, ["scportrait"]
        )
        self.assertEqual(
            get_stage("cellvision-cluster").environment_keys, ["rapids"]
        )
        assets = {
            asset.role: asset for asset in resolve_assets(PipelineConfig(), Path("."))
        }
        self.assertIn("cellvision_assets", assets)

    def test_planner_simulates_upstream_assets_and_exports_both_environments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = initialize_project(Path(temp_dir) / "project")
            (context.root / "IMC_files" / "case.mcd").write_bytes(b"x")
            plan = build_run_plan(context, ["cellvision"])

            self.assertTrue(plan.ready, plan.errors)
            self.assertEqual(
                [stage.name for stage in plan.resolved_stages],
                [
                    "prep",
                    "denoise",
                    "cellpose",
                    "nimbus",
                    "cellvision-extract",
                    "cellvision-embed",
                    "cellvision-cluster",
                    "cellvision-plot",
                ],
            )
            self.assertIn(
                "cellvision_assets", plan.resolved_stages[-2].produces_assets
            )

            full_plan = build_run_plan(context, ["cellvision-full"])
            run = create_run_record(
                context, full_plan, command="sbt run cellvision-full"
            )
            environment = sbt_environment(context, run, "cellvision-full")

        self.assertEqual(environment["SBT_CONDA_ENV"], "scPortrait")
        self.assertEqual(environment["SBT_CONDA_ENV_SCPORTRAIT"], "scPortrait")
        self.assertEqual(environment["SBT_CONDA_ENV_RAPIDS"], "rapids_singlecell")

    def test_cluster_component_no_deps_requires_existing_embeddings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = initialize_project(Path(temp_dir) / "project")
            (context.root / "IMC_files" / "case.mcd").write_bytes(b"x")
            assets = {
                asset.role: asset
                for asset in resolve_assets(context.config, context.root)
            }
            asset_root = assets["cellvision_assets"].path
            asset_root.mkdir(parents=True)
            (asset_root / "placeholder.txt").write_text("partial", encoding="utf-8")

            missing = build_run_plan(
                context, ["cellvision-cluster"], include_dependencies=False
            )
            self.assertFalse(missing.ready)
            self.assertIn(
                asset_root / "cellvision_embeddings.h5ad",
                missing.resolved_stages[0].missing_files,
            )

            assets["anndata"].path.parent.mkdir(parents=True, exist_ok=True)
            assets["anndata"].path.write_bytes(b"placeholder")
            (asset_root / "cellvision_embeddings.h5ad").write_bytes(b"placeholder")
            ready = build_run_plan(
                context, ["cellvision-cluster"], include_dependencies=False
            )
            self.assertTrue(ready.ready, ready.errors)
            self.assertEqual(
                [stage.name for stage in ready.resolved_stages],
                ["cellvision-cluster"],
            )

    def test_cellvision_mode_no_deps_retains_component_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = initialize_project(Path(temp_dir) / "project")
            assets = {
                asset.role: asset
                for asset in resolve_assets(context.config, context.root)
            }
            (context.root / "IMC_files" / "case.mcd").write_bytes(b"x")
            assets["anndata"].path.parent.mkdir(parents=True, exist_ok=True)
            assets["anndata"].path.write_bytes(b"placeholder")
            for role in ("denoised_images", "masks"):
                assets[role].path.mkdir(parents=True, exist_ok=True)
                (assets[role].path / "placeholder.tif").write_bytes(b"placeholder")

            plan = build_run_plan(
                context, ["cellvision"], include_dependencies=False
            )

            self.assertTrue(plan.ready, plan.errors)
            self.assertEqual(
                [stage.depends_on for stage in plan.resolved_stages],
                [
                    [],
                    ["cellvision-extract"],
                    ["cellvision-embed"],
                    ["cellvision-cluster"],
                ],
            )


class CellVisionVICRegTests(unittest.TestCase):
    def test_h5sc_dataset_reads_pre_normalized_images_and_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cells.h5sc"
            values = np.zeros((3, 3, 4, 4), dtype=np.float32)
            values[:, 0, 1:3, 1:3] = 1
            values[:, 1] = 0.25
            values[:, 2] = 0.75
            with h5py.File(path, "w") as handle:
                obsm = handle.create_group("obsm")
                obsm.create_dataset("single_cell_images", data=values)
            dataset = H5SCImageDataset(
                path,
                channel_indices=[1, 2],
                mask_index=0,
            )
            image, mask, row = dataset[1]
            dataset.close()

        self.assertEqual(row, 1)
        self.assertEqual(tuple(image.shape), (2, 4, 4))
        self.assertTrue(torch.allclose(image[0], torch.full((4, 4), 0.25)))
        self.assertTrue(torch.allclose(image[1], torch.full((4, 4), 0.75)))
        self.assertEqual(tuple(mask.shape), (1, 4, 4))

    def test_h5sc_range_validation_rejects_unnormalized_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cells.h5sc"
            with h5py.File(path, "w") as handle:
                obsm = handle.create_group("obsm")
                obsm.create_dataset(
                    "single_cell_images", data=np.full((2, 2, 3, 3), 1.01, dtype=np.float32)
                )
            with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                validate_h5sc_unit_range(path, channel_indices=[1])

    def test_mask_safe_noise_keeps_true_background_zero(self):
        image = torch.zeros((2, 2, 7, 7), dtype=torch.float32)
        image[:, 0, 3, 3] = 0.5
        masks = torch.zeros((2, 1, 7, 7), dtype=torch.float32)
        masks[:, :, 2:5, 2:5] = 1
        augment = MaskSafeAugment(
            translation_px=0,
            intensity_jitter=0,
            noise_std=0.1,
            horizontal_flip_probability=0,
            vertical_flip_probability=0,
            rotation_probability=0,
            noise_support="channel",
        )

        result = augment(image, masks)

        self.assertTrue(torch.all(result[:, :, 0, 0] == 0))
        self.assertTrue(torch.all(result[:, 1] == 0))

    def test_torch_model_and_vicreg_objective_are_finite(self):
        model = VICRegNetwork(
            2,
            width=8,
            embedding_dim=16,
            projector_dim=32,
        )
        first = torch.rand((4, 2, 36, 36))
        second = torch.rand((4, 2, 36, 36))
        embedding, first_projection = model(first)
        _embedding, second_projection = model(second)
        loss, components = vicreg_loss(
            first_projection,
            second_projection,
            invariance_weight=25,
            variance_weight=25,
            covariance_weight=1,
        )
        loss.backward()

        self.assertEqual(tuple(embedding.shape), (4, 16))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(components),
            {"invariance_loss", "variance_loss", "covariance_loss"},
        )

    def test_confusion_and_leiden_names_are_stable(self):
        counts, normalized = confusion_tables(["A", "A", "B"], ["0", "1", "1"])
        self.assertEqual(int(counts.loc["A"].sum()), 2)
        self.assertAlmostEqual(float(normalized.loc["A"].sum()), 1.0)
        self.assertEqual(leiden_key(0.3), "cellvision_leiden_0.3")

    def test_source_representations_and_markers_align_by_observation_identity(self):
        source = ad.AnnData(
            X=np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float32),
            obs=pd.DataFrame(index=["cell-c", "cell-a", "cell-b"]),
            var=pd.DataFrame(index=["165Ho_CD11c", "174Yb_MHCII"]),
        )
        source.obsm["X_biobatchnet"] = np.array(
            [[30, 31], [10, 11], [20, 21]], dtype=np.float32
        )

        representation = aligned_obsm_representation(
            source, ["cell-a", "cell-b"], "X_biobatchnet"
        )
        markers, names = aligned_marker_matrix(
            source, ["cell-a", "cell-b"], ["CD11c", "MHCII"]
        )

        np.testing.assert_array_equal(representation, [[10, 11], [20, 21]])
        np.testing.assert_array_equal(markers, [[2, 20], [3, 30]])
        self.assertEqual(names, ["165Ho_CD11c", "174Yb_MHCII"])

    def test_graph_fusion_is_symmetric_and_respects_weight_endpoints(self):
        from scipy import sparse

        morphology = sparse.csr_matrix(
            [[0, 2, 0], [2, 0, 1], [0, 1, 0]], dtype=np.float32
        )
        intensity = sparse.csr_matrix(
            [[0, 0, 3], [0, 0, 1], [3, 1, 0]], dtype=np.float32
        )

        morphology_only = fuse_connectivity_graphs(
            morphology, intensity, intensity_weight=0
        )
        intensity_only = fuse_connectivity_graphs(
            morphology, intensity, intensity_weight=1
        )
        joint = fuse_connectivity_graphs(morphology, intensity, intensity_weight=0.5)

        np.testing.assert_allclose(joint.toarray(), joint.T.toarray())
        np.testing.assert_allclose(
            joint.toarray(),
            0.5 * morphology_only.toarray() + 0.5 * intensity_only.toarray(),
        )

    def test_registered_fusion_graph_and_umap_contract_are_scanpy_compatible(self):
        from scipy import sparse

        data = ad.AnnData(X=np.ones((3, 1), dtype=np.float32))
        data.obsp["m_connectivities"] = sparse.csr_matrix(
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float32
        )
        data.obsp["i_connectivities"] = sparse.csr_matrix(
            [[0, 2, 1], [2, 0, 1], [1, 1, 0]], dtype=np.float32
        )
        data.uns["m"] = {"connectivities_key": "m_connectivities"}
        data.uns["i"] = {"connectivities_key": "i_connectivities"}

        key = _register_fused_neighbors(
            data,
            morphology_neighbors_key="m",
            intensity_neighbors_key="i",
            joint_neighbors_key="cellvision_neighbors",
            intensity_weight=0.25,
            n_neighbors=2,
            representation_key="X_cellvision_pca",
            n_pcs=2,
            random_state=17,
        )

        self.assertEqual(key, "cellvision_neighbors")
        self.assertEqual(
            data.uns[key]["connectivities_key"],
            "cellvision_neighbors_connectivities",
        )
        self.assertEqual(
            data.uns[key]["distances_key"],
            "cellvision_neighbors_distances",
        )
        self.assertIn("cellvision_neighbors_connectivities", data.obsp)
        self.assertTrue(
            sparse.isspmatrix_csr(data.obsp["cellvision_neighbors_connectivities"])
        )
        self.assertTrue(
            hasattr(data.obsp["cellvision_neighbors_connectivities"], "nonzero")
        )
        self.assertNotIn("cellvision_neighbors_distances", data.obsp)
        self.assertEqual(data.uns[key]["params"]["n_neighbors"], 2)
        self.assertEqual(data.uns[key]["params"]["use_rep"], "X_cellvision_pca")
        self.assertEqual(
            _cellvision_umap_params(17),
            {"init_pos": "random", "random_state": 17},
        )

    def test_cluster_explanation_qc_handles_continuous_and_categorical_targets(self):
        labels = ["0", "0", "1", "1"]
        continuous = continuous_cluster_explained_variance(
            np.array([[0, 1], [0, 2], [10, 3], [10, 4]], dtype=float),
            labels,
            feature_names=["separated", "gradient"],
            modality="test",
        )
        categorical = categorical_entropy_explained(
            labels,
            ["ROI-A", "ROI-A", "ROI-B", "ROI-B"],
        )

        self.assertAlmostEqual(
            float(
                continuous.loc[
                    continuous["feature"].eq("separated"),
                    "explained_variance_fraction",
                ].iloc[0]
            ),
            1.0,
        )
        self.assertAlmostEqual(
            float(categorical["entropy_explained_fraction"]), 1.0
        )

    def test_cellvision_leiden_rename_preserves_source_leiden_columns(self):
        data = ad.AnnData(
            X=np.ones((2, 1)),
            obs=pd.DataFrame(
                {
                    "leiden": pd.Categorical(["original-a", "original-b"]),
                    "leiden_1.0": pd.Categorical(["4", "5"]),
                    "cellvision_leiden_1.0": pd.Categorical(["0", "1"]),
                },
                index=["cell-1", "cell-2"],
            ),
        )

        keys = _canonicalize_leiden_columns(
            data,
            resolutions=[1.0],
            generated_keys=["cellvision_leiden_1.0"],
        )

        self.assertEqual(keys, ["cellvision_leiden_1"])
        self.assertNotIn("cellvision_leiden_1.0", data.obs)
        self.assertEqual(
            data.obs["cellvision_leiden_1"].astype(str).tolist(), ["0", "1"]
        )
        self.assertEqual(
            data.obs["leiden"].astype(str).tolist(),
            ["original-a", "original-b"],
        )
        self.assertEqual(data.obs["leiden_1.0"].astype(str).tolist(), ["4", "5"])

    def test_namespaced_rapids_leiden_preserves_source_column(self):
        class FakeTools:
            @staticmethod
            def leiden(adata, *, resolution, key_added, **_kwargs):
                adata.obs[key_added] = pd.Categorical(
                    [f"cellvision-{resolution}"] * adata.n_obs
                )

        fake_rapids = SimpleNamespace(tl=FakeTools())
        with patch.dict(sys.modules, {"rapids_singlecell": fake_rapids}):
            rapids_module = importlib.import_module(
                "SpatialBiologyToolkit.scripts.basic_process_rapids"
            )
        with patch.object(rapids_module, "rsc", fake_rapids):
            data = ad.AnnData(
                X=np.ones((2, 1)),
                obs=pd.DataFrame(
                    {"leiden_1.0": pd.Categorical(["4", "5"])},
                    index=["cell-1", "cell-2"],
                ),
            )
            keys = rapids_module._run_rapids_leiden(
                data,
                resolutions=[1.0],
                enabled=True,
                neighbors_key="cellvision_neighbors",
                leiden_params={},
                key_prefix="cellvision_leiden",
            )

        self.assertEqual(keys, ["cellvision_leiden_1.0"])
        self.assertEqual(data.obs["leiden_1.0"].astype(str).tolist(), ["4", "5"])
        self.assertEqual(
            data.obs["cellvision_leiden_1.0"].astype(str).tolist(),
            ["cellvision-1.0", "cellvision-1.0"],
        )


if __name__ == "__main__":
    unittest.main()
