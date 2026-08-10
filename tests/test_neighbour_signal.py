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
    _aggregate_source_target_attribution,
    build_output_anndata,
    build_source_target_table,
    project_source_halos,
    project_source_halos_with_sources,
    run_neighbour_signal_analysis,
)
from SpatialBiologyToolkit.neighbour_signal_reports import (
    _plot_source_target_population_heatmaps,
    dominant_source_summary,
    population_source_target_summary,
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


def _run(
    root: Path,
    rois: list[str],
    *,
    n_jobs: int,
    unknown_exemplar: bool = False,
    aggregation: str = "max",
):
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
            halo_aggregation=aggregation,
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
        source_target_table=build_source_target_table(
            source,
            result,
            roi_obs="ROI",
            object_id_obs="ObjectNumber",
            population_obs="Population",
        ),
        source_target_table_path=tmp_path / "source_target.parquet",
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
        "dominant_source_index",
        "dominant_source_observed_fraction",
        "dominant_source_attributable_fraction",
        "existing_layer",
    }.issubset(output.layers)
    np.testing.assert_array_equal(output.layers["original_X"], source.X)
    np.testing.assert_array_equal(output.obsm["X_umap"], source.obsm["X_umap"])
    assert output.uns["existing_metadata"]["retained"]
    assert "marker_halo" in output.uns
    assert output.obs.loc["ROI_1_cell_6", "halo_max_score"] > 0.85
    source_target = build_source_target_table(
        source,
        result,
        roi_obs="ROI",
        object_id_obs="ObjectNumber",
        population_obs="Population",
    )
    near_relationships = source_target.loc[
        (source_target["target_obs_index"] == near)
        & source_target["marker"].eq("CD31")
    ]
    assert len(near_relationships) == 1
    relationship = near_relationships.iloc[0]
    assert int(relationship["source_obs_index"]) == source_cell
    assert relationship["source_cell_id"] == "ROI_1_cell_1"
    assert relationship["source_roi"] == relationship["target_roi"] == "ROI_1"
    assert relationship["source_population"] == "source"
    assert relationship["target_population"] == "target"
    assert np.isclose(
        relationship["fraction_of_observed_signal"],
        result.scores[near, 0],
    )
    assert np.isclose(relationship["fraction_of_attributable_signal"], 1.0)
    assert result.dominant_source_indices[near, 0] == source_cell
    assert result.dominant_source_indices[far, 0] == -1
    assert result.dominant_source_indices[no_signal, 0] == -1
    assert not source_target["target_obs_index"].isin([far, no_signal]).any()
    assert not (
        (source_target["target_obs_index"] == source_target["source_obs_index"])
    ).any()
    observed_fraction_sums = source_target.groupby(
        ["target_obs_index", "marker"], observed=True
    )["fraction_of_observed_signal"].sum()
    for (target_index, marker), value in observed_fraction_sums.items():
        assert np.isclose(
            value,
            result.scores[int(target_index), source.var_names.get_loc(marker)],
        )
    attributable_fraction_sums = source_target.groupby(
        ["target_obs_index", "marker"], observed=True
    )["fraction_of_attributable_signal"].sum()
    np.testing.assert_allclose(attributable_fraction_sums.to_numpy(), 1.0)

    population_summary = population_source_target_summary(source_target)
    route = population_summary.loc[
        population_summary["source_population"].eq("source")
        & population_summary["target_population"].eq("target")
        & population_summary["marker"].eq("CD31")
    ]
    assert len(route) == 1
    assert int(route.iloc[0]["unique_target_cells"]) >= 1
    heatmap_path = tmp_path / "source_target_population_heatmap.png"
    assert (
        _plot_source_target_population_heatmaps(
            population_summary,
            ["CD31"],
            heatmap_path,
            exclude_same_population=True,
        )
        == heatmap_path
    )
    assert heatmap_path.is_file()
    dominant_summary = dominant_source_summary(output, source_target).set_index("marker")
    assert dominant_summary.loc["CD31", "affected_target_cells"] >= 1
    assert (
        dominant_summary.loc[
            "CD31", "fraction_affected_targets_dominant_source_gt_0.5"
        ]
        > 0
    )

    output_path = tmp_path / "roundtrip.h5ad"
    output.write_h5ad(output_path)
    restored = ad.read_h5ad(output_path)
    assert restored.obs_names.equals(source.obs_names)
    assert restored.var_names.equals(source.var_names)
    assert restored.X.dtype == np.float32
    assert restored.uns["marker_halo"]["score_name"] == "NeighbourAttributableFraction"
    assert restored.layers["dominant_source_index"].dtype == np.int64
    assert restored.uns["marker_halo"]["source_target_table"]["relationships"] == len(
        source_target
    )


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


def _check_competing_sources_preserve_pixel_and_cell_provenance():
    mask = np.zeros((24, 24), dtype=np.int64)
    mask[10:13, 2:5] = 1
    mask[10:13, 18:21] = 2
    mask[9:14, 9:14] = 3
    curve = np.linspace(1.0, 0.1, 10, dtype=np.float32)
    projected = project_source_halos_with_sources(
        mask,
        {1: 10.0, 2: 12.0},
        [1, 2],
        {1: 0, 2: 1},
        curve,
        background=0.0,
        max_halo_px=10,
        aggregation="max",
    )
    target_sources = set(projected.source_index[mask == 3].tolist())
    assert target_sources == {0, 1}
    source_one = project_source_halos(
        mask,
        {1: 10.0},
        [1],
        curve,
        background=0.0,
        max_halo_px=10,
    )
    source_two = project_source_halos(
        mask,
        {2: 12.0},
        [2],
        curve,
        background=0.0,
        max_halo_px=10,
    )
    expected_target_sources = np.where(
        source_two[mask == 3] > source_one[mask == 3],
        1,
        0,
    )
    np.testing.assert_array_equal(
        projected.source_index[mask == 3],
        expected_target_sources,
    )
    assert np.all(projected.source_index[mask == 1] != 0)
    assert np.all(projected.source_index[mask == 2] != 1)

    observed = np.zeros(mask.shape, dtype=np.float32)
    observed[mask == 3] = 20.0
    attributable = np.minimum(observed, projected.predicted).astype(np.float32)
    observed_sums = np.bincount(
        mask.ravel(), weights=observed.ravel(), minlength=4
    ).astype(np.float64)
    attributable_sums = np.bincount(
        mask.ravel(), weights=attributable.ravel(), minlength=4
    ).astype(np.float64)
    records, dominant, dominant_observed, dominant_attributable = (
        _aggregate_source_target_attribution(
            mask=mask,
            attributable=attributable,
            source_index=projected.source_index,
            target_rows=np.array([0, 1, 2], dtype=np.int64),
            target_labels=np.array([1, 2, 3], dtype=np.int64),
            marker_index=0,
            observed_sums=observed_sums,
            attributable_sums=attributable_sums,
        )
    )
    target_records = [record for record in records if record.target_obs_index == 2]
    assert {record.source_obs_index for record in target_records} == {0, 1}
    expected_dominant = max(
        target_records,
        key=lambda record: (record.attributable_intensity, -record.source_obs_index),
    )
    assert dominant[2] == expected_dominant.source_obs_index == 1
    assert np.isclose(
        sum(record.fraction_of_observed_signal for record in target_records),
        attributable_sums[3] / observed_sums[3],
    )
    assert np.isclose(
        sum(record.fraction_of_attributable_signal for record in target_records),
        1.0,
    )
    assert np.isclose(
        dominant_observed[2], expected_dominant.fraction_of_observed_signal
    )
    assert np.isclose(
        dominant_attributable[2],
        expected_dominant.fraction_of_attributable_signal,
    )

    summed = project_source_halos_with_sources(
        mask,
        {1: 10.0, 2: 12.0},
        [1, 2],
        {1: 0, 2: 1},
        curve,
        background=0.0,
        max_halo_px=10,
        aggregation="sum",
    )
    assert np.all(summed.source_index == -1)


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
    np.testing.assert_array_equal(
        serial.dominant_source_indices,
        parallel.dominant_source_indices,
    )
    np.testing.assert_allclose(
        serial.dominant_source_observed_fractions,
        parallel.dominant_source_observed_fractions,
        rtol=0,
        atol=1e-7,
    )
    serial_table = build_source_target_table(
        _source_serial,
        serial,
        roi_obs="ROI",
        object_id_obs="ObjectNumber",
        population_obs="Population",
    )
    parallel_table = build_source_target_table(
        _source_parallel,
        parallel,
        roi_obs="ROI",
        object_id_obs="ObjectNumber",
        population_obs="Population",
    )
    pd.testing.assert_frame_equal(serial_table, parallel_table)
    for roi_index, roi in enumerate(("ROI_1", "ROI_2")):
        target_index = _source_serial.obs_names.get_loc(f"{roi}_cell_6")
        expected_source = _source_serial.obs_names.get_loc(f"{roi}_cell_1")
        assert serial.dominant_source_indices[target_index, 0] == expected_source
        assert expected_source == roi_index * 8


def _check_sum_aggregation_disables_source_resolved_provenance(tmp_path: Path):
    source, result = _run(
        tmp_path,
        ["ROI_1"],
        n_jobs=1,
        aggregation="sum",
    )
    near = source.obs_names.get_loc("ROI_1_cell_6")
    assert result.scores[near, 0] > 0
    assert not result.source_provenance_available
    assert not result.source_target_attributions
    assert np.all(result.dominant_source_indices == -1)
    assert np.all(result.dominant_source_observed_fractions == 0)
    assert any("halo_aggregation='sum'" in warning for warning in result.warnings)
    table = build_source_target_table(
        source,
        result,
        roi_obs="ROI",
        object_id_obs="ObjectNumber",
        population_obs="Population",
    )
    assert table.empty
    assert "source_population" in table.columns


def _check_config_registry_assets_wrapper_environment_and_plan_align(tmp_path: Path):
    settings = NeighbourSignalConfig()
    assert settings.exemplar_obs == "Exemplar_stains"
    assert settings.max_halo_px == 8
    assert settings.n_jobs == "auto"
    assert settings.source_target_table_path == "neighbour_signal_source_target.parquet"
    assert settings.source_target_qc_exclude_same_population
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
    try:
        NeighbourSignalConfig(source_target_table_path="source_target.csv")
    except ValueError as exc:
        assert "must end with .parquet or .pq" in str(exc)
    else:
        raise AssertionError("non-Parquet source-target output should fail validation")

    stage = STAGE_REGISTRY["neighsig"]
    assert stage.catalogue_order == 40
    assert stage.depends_on == []
    assert stage.groups == ["qc"]
    assert stage.environment_keys == ["segmentation"]
    assert stage.config_sections == ["general", "neighbour_signal"]
    assert stage.requires_assets == ["anndata", "raw_images", "masks"]
    assert stage.produces_assets == [
        "neighbour_signal_anndata",
        "neighbour_signal_source_target_table",
        "human_outputs",
    ]
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
    assert assets["neighbour_signal_source_target_table"].path == (
        tmp_path / "neighbour_signal_source_target.parquet"
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
    source_target_path = tmp_path / "neighbour_signal_source_target.parquet"
    assert source_target_path.is_file()
    source_target = pd.read_parquet(source_target_path)
    assert len(source_target) > 0
    assert {
        "target_obs_index",
        "source_obs_index",
        "marker",
        "fraction_of_observed_signal",
        "fraction_of_attributable_signal",
    }.issubset(source_target.columns)
    restored = ad.read_h5ad(output_path)
    assert restored.obs_names.equals(source.obs_names)
    assert restored.var_names.equals(source.var_names)
    assert restored.uns["marker_halo"]["source_target_table"]["path"] == str(
        source_target_path
    )
    assert restored.uns["marker_halo"]["source_target_table"]["relationships"] == len(
        source_target
    )
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

    def test_competing_source_provenance(self):
        _check_competing_sources_preserve_pixel_and_cell_provenance()

    def test_serial_parallel_equivalence(self):
        with tempfile.TemporaryDirectory() as temporary:
            _check_serial_and_roi_multiprocessing_are_numerically_equivalent(
                Path(temporary)
            )

    def test_sum_disables_source_resolved_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            _check_sum_aggregation_disables_source_resolved_provenance(
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
