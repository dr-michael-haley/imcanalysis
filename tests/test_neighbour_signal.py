from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import tifffile
import yaml
from scipy import ndimage

matplotlib.use("Agg")

from SpatialBiologyToolkit.cellvision import discover_roi_inputs, select_source_cells
from SpatialBiologyToolkit.config import NeighbourSignalConfig, PipelineConfig
from SpatialBiologyToolkit.neighbour_signal import (
    HaloParameters,
    build_output_anndata,
    project_source_halos,
    run_neighbour_signal_analysis,
)
from SpatialBiologyToolkit.pipeline.assets import resolve_assets
from SpatialBiologyToolkit.pipeline.planner import build_run_plan
from SpatialBiologyToolkit.pipeline.project import initialize_project
from SpatialBiologyToolkit.pipeline.registry import MODES, STAGE_REGISTRY
from SpatialBiologyToolkit.scripts.neighbour_signal import run_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWN_CURVE = np.array([0.8, 0.5, 0.25, 0.0], dtype=np.float32)


def _paint_halo(
    image: np.ndarray,
    mask: np.ndarray,
    object_id: int,
    *,
    background: float,
    strength_excess: float,
    curve: np.ndarray = KNOWN_CURVE,
) -> None:
    source = mask == object_id
    distance = ndimage.distance_transform_edt(~source)
    influence = (distance > 0) & (distance <= len(curve))
    bins = np.clip(np.ceil(distance).astype(int) - 1, 0, len(curve) - 1)
    contribution = np.full(mask.shape, background, dtype=np.float32)
    contribution[influence] = background + curve[bins[influence]] * strength_excess
    np.maximum(image, contribution, out=image)
    image[source] = background + strength_excess


def _write_roi(root: Path, roi: str) -> None:
    mask_folder = root / "masks"
    image_folder = root / "tiffs" / roi
    mask_folder.mkdir(parents=True, exist_ok=True)
    image_folder.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((72, 72), dtype=np.uint16)
    positions = {
        1: (8, 8),
        2: (8, 30),
        3: (30, 8),
        4: (30, 30),
        5: (50, 8),
        6: (13, 8),  # nearby target: 3-5 px from source 1
        7: (50, 45),  # far target and the sole CD3 exemplar
        8: (62, 30),  # background-only target
    }
    for object_id, (row, column) in positions.items():
        mask[row : row + 3, column : column + 3] = object_id

    cd31 = np.full(mask.shape, 2.0, dtype=np.float32)
    for object_id in range(1, 6):
        _paint_halo(
            cd31,
            mask,
            object_id,
            background=2.0,
            strength_excess=100.0,
        )
    cd31[mask == 7] = 52.0  # genuine but sub-threshold far-cell signal
    cd31[mask == 8] = 2.0

    cd3 = np.full(mask.shape, 1.0, dtype=np.float32)
    _paint_halo(
        cd3,
        mask,
        7,
        background=1.0,
        strength_excess=80.0,
    )
    tifffile.imwrite(mask_folder / f"{roi}.tiff", mask)
    tifffile.imwrite(image_folder / "CD31.tiff", cd31)
    tifffile.imwrite(image_folder / "CD3.tiff", cd3)


def _adata(rois: list[str], *, unknown_exemplar: bool = False) -> ad.AnnData:
    rows = []
    obs_names = []
    for roi in rois:
        for object_id in range(1, 9):
            exemplar = "CD31" if object_id <= 5 else ("CD3" if object_id == 7 else None)
            if unknown_exemplar and object_id == 8:
                exemplar = "MissingMarker"
            rows.append(
                {
                    "ROI": roi,
                    "ObjectNumber": object_id,
                    "Exemplar_stains": exemplar,
                    "Population": "target" if object_id >= 6 else "source",
                }
            )
            obs_names.append(f"{roi}_cell_{object_id}")
    n_cells = len(rows)
    source = ad.AnnData(
        X=np.linspace(0.05, 0.95, n_cells * 2, dtype=np.float32).reshape(n_cells, 2),
        obs=pd.DataFrame(rows, index=obs_names),
        var=pd.DataFrame({"panel_role": ["vascular", "lymphoid"]}, index=["CD31", "CD3"]),
    )
    source.layers["existing_layer"] = np.ones((n_cells, 2), dtype=np.float32)
    source.obsm["X_umap"] = np.column_stack(
        [np.arange(n_cells, dtype=np.float32), np.arange(n_cells, dtype=np.float32) / 2]
    )
    source.uns["existing_metadata"] = {"retained": True}
    return source


def _analysis_inputs(root: Path, rois: list[str], *, unknown_exemplar: bool = False):
    for roi in rois:
        _write_roi(root, roi)
    source = _adata(rois, unknown_exemplar=unknown_exemplar)
    identity = select_source_cells(
        source,
        roi_obs="ROI",
        object_id_obs="ObjectNumber",
    )
    contexts, marker_names = discover_roi_inputs(
        root / "tiffs",
        root / "masks",
        identity,
        roi_obs="ROI",
        markers=list(source.var_names),
    )
    assert marker_names == list(source.var_names)
    return source, identity, contexts


def _run(root: Path, rois: list[str], *, n_jobs: int, unknown_exemplar: bool = False):
    source, identity, contexts = _analysis_inputs(
        root,
        rois,
        unknown_exemplar=unknown_exemplar,
    )
    result = run_neighbour_signal_analysis(
        source,
        contexts,
        identity,
        roi_obs="ROI",
        object_id_obs="ObjectNumber",
        exemplar_obs="Exemplar_stains",
        parameters=HaloParameters(
            max_halo_px=4,
            source_anchor_dilation_px=2,
            source_anchor_quantile=0.95,
            min_exemplars=5,
            source_threshold_quantile=0.1,
            halo_aggregation="max",
        ),
        n_jobs=n_jobs,
    )
    return source, result


def _check_learns_known_halo_scores_near_target_and_preserves_output_contract(tmp_path: Path):
    source, result = _run(tmp_path, ["ROI_1"], n_jobs=1, unknown_exemplar=True)

    profile = result.profiles["CD31"]
    assert profile.available
    assert profile.n_valid_exemplars == 5
    np.testing.assert_allclose(profile.final[:3], KNOWN_CURVE[:3], atol=0.12)
    assert not result.profiles["CD3"].available
    assert "insufficient configured exemplars" in result.profiles["CD3"].skip_reason
    assert result.unknown_exemplar_values == ("MissingMarker",)

    near = source.obs_names.get_loc("ROI_1_cell_6")
    far = source.obs_names.get_loc("ROI_1_cell_7")
    no_signal = source.obs_names.get_loc("ROI_1_cell_8")
    source_cell = source.obs_names.get_loc("ROI_1_cell_1")
    assert result.scores[near, 0] > 0.85
    assert np.isclose(result.scores[far, 0], 0.0)
    assert np.isclose(result.scores[no_signal, 0], 0.0)
    assert np.isclose(result.scores[source_cell, 0], 0.0)
    assert np.all(result.scores[:, 1] == 0)

    output = build_output_anndata(
        source,
        result,
        parameters={"exemplar_obs": "Exemplar_stains", "max_halo_px": 4},
        calculate_classic_intensities=True,
        high_risk_threshold=0.5,
    )
    assert output.obs_names.equals(source.obs_names)
    assert output.var_names.equals(source.var_names)
    assert output.X.dtype == np.float32
    assert np.all(np.isfinite(output.X))
    assert np.all((output.X >= 0) & (output.X <= 1))
    assert {
        "original_X",
        "classic_intensities",
        "neighbour_attributable_intensity",
        "residual_excess_intensity",
        "existing_layer",
    }.issubset(output.layers)
    np.testing.assert_array_equal(output.layers["original_X"], source.X)
    np.testing.assert_array_equal(output.obsm["X_umap"], source.obsm["X_umap"])
    assert output.uns["existing_metadata"]["retained"]
    assert "marker_halo" in output.uns
    assert output.obs.loc["ROI_1_cell_6", "halo_max_score"] > 0.85

    output_path = tmp_path / "roundtrip.h5ad"
    output.write_h5ad(output_path)
    restored = ad.read_h5ad(output_path)
    assert restored.obs_names.equals(source.obs_names)
    assert restored.var_names.equals(source.var_names)
    assert restored.X.dtype == np.float32
    assert restored.uns["marker_halo"]["score_name"] == "NeighbourAttributableFraction"


def _check_projection_uses_max_by_default_and_never_projects_inside_own_mask():
    mask = np.zeros((18, 18), dtype=np.uint16)
    mask[5:7, 5:7] = 1
    single = project_source_halos(
        mask,
        {1: 10.0},
        [1],
        [1.0, 0.8, 0.6, 0.4],
        background=0.0,
        max_halo_px=4,
        aggregation="max",
    )
    assert np.all(single[mask == 1] == 0)

    mask[5:7, 11:13] = 2
    maximum = project_source_halos(
        mask,
        {1: 10.0, 2: 20.0},
        [1, 2],
        [1.0, 0.8, 0.6, 0.4],
        background=0.0,
        max_halo_px=4,
        aggregation="max",
    )
    summed = project_source_halos(
        mask,
        {1: 10.0, 2: 20.0},
        [1, 2],
        [1.0, 0.8, 0.6, 0.4],
        background=0.0,
        max_halo_px=4,
        aggregation="sum",
    )
    assert np.isclose(maximum[5, 8], 12.0)
    assert np.isclose(summed[5, 8], 20.0)
    assert np.all(maximum <= summed)


def _check_serial_and_roi_multiprocessing_are_numerically_equivalent(tmp_path: Path):
    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"
    _source_serial, serial = _run(serial_root, ["ROI_1", "ROI_2"], n_jobs=1)
    _source_parallel, parallel = _run(parallel_root, ["ROI_1", "ROI_2"], n_jobs=2)

    np.testing.assert_allclose(serial.scores, parallel.scores, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        serial.classic_intensities,
        parallel.classic_intensities,
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        serial.attributable_intensities,
        parallel.attributable_intensities,
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        serial.residual_intensities,
        parallel.residual_intensities,
        rtol=0,
        atol=1e-7,
    )


def _check_config_registry_assets_wrapper_environment_and_plan_align(tmp_path: Path):
    settings = NeighbourSignalConfig()
    assert settings.exemplar_obs == "Exemplar_stains"
    assert settings.max_halo_px == 8
    assert settings.n_jobs == "auto"
    try:
        NeighbourSignalConfig(n_jobs=0)
    except ValueError as exc:
        assert "positive integer" in str(exc)
    else:
        raise AssertionError("n_jobs=0 should fail validation")
    try:
        NeighbourSignalConfig(max_halo_px=2, source_anchor_dilation_px=3)
    except ValueError as exc:
        assert "cannot exceed" in str(exc)
    else:
        raise AssertionError("anchor dilation above max_halo_px should fail")
    try:
        NeighbourSignalConfig(output_adata_path="scores.csv")
    except ValueError as exc:
        assert "must end with .h5ad" in str(exc)
    else:
        raise AssertionError("non-h5ad output should fail validation")

    stage = STAGE_REGISTRY["neighsig"]
    assert stage.catalogue_order == 40
    assert stage.depends_on == []
    assert stage.groups == ["qc"]
    assert stage.environment_keys == ["segmentation"]
    assert stage.config_sections == ["general", "neighbour_signal"]
    assert stage.requires_assets == ["anndata", "raw_images", "masks"]
    assert stage.produces_assets == ["neighbour_signal_anndata", "human_outputs"]
    assert all("neighsig" not in mode.stages for mode in MODES)
    assert (REPO_ROOT / stage.documentation_path).is_file()
    wrapper = (REPO_ROOT / stage.slurm_script).read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=8" in wrapper
    assert "#SBATCH --mem=64G" in wrapper
    assert "#@ENV:  imc_segmentation" in wrapper
    assert "OMP_NUM_THREADS=1" in wrapper
    environments = yaml.safe_load(
        (REPO_ROOT / "HPC_env_files" / "environments.yaml").read_text(encoding="utf-8")
    )
    assert environments["stage_environments"]["neighsig"] == ["segmentation"]

    config = PipelineConfig()
    assets = {asset.role: asset for asset in resolve_assets(config, tmp_path)}
    assert assets["neighbour_signal_anndata"].path == (
        tmp_path / "neighbour_attributable_signal.h5ad"
    ).resolve()

    context = initialize_project(tmp_path / "project")
    anndata_path = context.root / context.config.general.anndata_path
    raw_images = context.root / context.config.general.raw_images_folder
    masks = context.root / context.config.general.masks_folder
    anndata_path.write_bytes(b"placeholder")
    (raw_images / "ROI_1").mkdir(parents=True)
    (raw_images / "ROI_1" / "CD31.tiff").write_bytes(b"placeholder")
    masks.mkdir(parents=True)
    (masks / "ROI_1.tiff").write_bytes(b"placeholder")
    plan = build_run_plan(context, ["neighsig"])
    assert plan.ready, plan.errors
    assert [candidate.name for candidate in plan.resolved_stages] == ["neighsig"]


def _check_direct_stage_smoke_writes_asset_and_concise_qc(tmp_path: Path):
    source, _identity, _contexts = _analysis_inputs(tmp_path, ["ROI_1"])
    del source.obsm["X_umap"]
    source_path = tmp_path / "input.h5ad"
    source.write_h5ad(source_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "general": {
                    "anndata_path": "input.h5ad",
                    "raw_images_folder": "tiffs",
                    "masks_folder": "masks",
                    "roi_obs": "ROI",
                },
                "neighbour_signal": {
                    "output_adata_path": "halo_scores.h5ad",
                    "max_halo_px": 4,
                    "source_anchor_dilation_px": 2,
                    "min_exemplars": 5,
                    "n_jobs": 1,
                    "max_qc_markers": 1,
                },
                "logging": {"console_only": True, "to_console": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    previous = Path.cwd()
    try:
        os.chdir(tmp_path)
        with patch.dict(
            os.environ,
            {
                "SBT_CONFIG": str(config_path),
                "SBT_PROJECT_ROOT": str(tmp_path),
            },
            clear=False,
        ), patch(
            "SpatialBiologyToolkit.reporting.bootstrap_stage_reporting",
            return_value=None,
        ):
            assert run_pipeline([]) == 0
    finally:
        os.chdir(previous)

    output_path = tmp_path / "halo_scores.h5ad"
    assert output_path.is_file()
    restored = ad.read_h5ad(output_path)
    assert restored.obs_names.equals(source.obs_names)
    assert restored.var_names.equals(source.var_names)
    report_root = tmp_path / "neighbour_signal_report"
    assert (
        report_root
        / "tables"
        / "neighbour_signal"
        / "neighbour_attributable_score_summary.csv"
    ).is_file()
    assert (
        report_root
        / "summaries"
        / "neighbour_signal"
        / "neighbour_signal_summary.md"
    ).is_file()
    assert (
        report_root
        / "figures"
        / "neighbour_signal"
        / "marker_halo_profiles_01.png"
    ).is_file()


class NeighbourSignalTests(unittest.TestCase):
    def test_halo_learning_scoring_and_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            _check_learns_known_halo_scores_near_target_and_preserves_output_contract(
                Path(temporary)
            )

    def test_projection_overlap_and_self_exclusion(self):
        _check_projection_uses_max_by_default_and_never_projects_inside_own_mask()

    def test_serial_parallel_equivalence(self):
        with tempfile.TemporaryDirectory() as temporary:
            _check_serial_and_roi_multiprocessing_are_numerically_equivalent(
                Path(temporary)
            )

    def test_control_plane(self):
        with tempfile.TemporaryDirectory() as temporary:
            _check_config_registry_assets_wrapper_environment_and_plan_align(
                Path(temporary)
            )

    def test_direct_stage_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            _check_direct_stage_smoke_writes_asset_and_concise_qc(Path(temporary))


if __name__ == "__main__":
    unittest.main()
