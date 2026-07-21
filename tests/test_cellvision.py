from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
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
    compute_normalization_dict,
    configuration_fingerprint,
    confusion_tables,
    identity_fingerprint,
    leiden_key,
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
from SpatialBiologyToolkit.scripts.cellvision_cluster import _atomic_write_h5ad
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

    def test_config_defaults_and_cross_field_validation(self):
        config = CellVisionConfig()
        self.assertEqual(config.image_size, 36)
        self.assertFalse(config.mask_gaussian_blur)
        self.assertEqual(config.normalization_quantile, 0.999)
        self.assertEqual(config.normalization_clip, [0.0, 1.0])
        self.assertEqual(config.leiden_resolutions, [0.2, 0.3, 0.5, 0.7, 1.0])
        with self.assertRaises(ValidationError):
            CellVisionConfig(populations=["T cell"])
        with self.assertRaises(ValidationError):
            CellVisionConfig(warmup_epochs=31, epochs=30)
        with self.assertRaises(ValidationError):
            CellVisionConfig(normalization_clip=[0, 2])

    def test_nimbus_normalization_dict_accepts_string_values_and_exact_channels(self):
        values = validate_normalization_dict(
            {"CD3": "12.5", "CD20": 8, "unused": 99},
            channel_names=["CD3", "CD20"],
        )
        self.assertEqual(values, {"CD3": 12.5, "CD20": 8.0})
        with self.assertRaises(ValueError):
            validate_normalization_dict({"CD3": 1}, channel_names=["CD3", "CD20"])

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

    def test_gallery_uses_stored_unit_range_without_autoscaling(self):
        image = np.array([[0.0, 0.001], [0.002, 0.0]], dtype=np.float32)
        displayed = _gallery_image(image, cell_id="cell", channel_name="CD3")
        np.testing.assert_array_equal(displayed, image)

    def test_registry_environment_mode_and_asset_role(self):
        stage = get_stage("cellvision")
        self.assertEqual(stage.environment_keys, ["scportrait", "rapids"])
        self.assertEqual(get_mode("cellvision").stages, ["cellvision"])
        self.assertEqual(len(stage.python_modules), 4)
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
                ["prep", "denoise", "cellpose", "nimbus", "cellvision"],
            )
            self.assertIn("cellvision_assets", plan.resolved_stages[-1].produces_assets)
            run = create_run_record(context, plan, command="sbt run cellvision")
            environment = sbt_environment(context, run, "cellvision")

        self.assertEqual(environment["SBT_CONDA_ENV"], "scPortrait")
        self.assertEqual(environment["SBT_CONDA_ENV_SCPORTRAIT"], "scPortrait")
        self.assertEqual(environment["SBT_CONDA_ENV_RAPIDS"], "rapids_singlecell")


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
