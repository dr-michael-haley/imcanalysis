from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from SpatialBiologyToolkit.config.load import load_config
from SpatialBiologyToolkit.config.models import CellCharterConfig, GeneralConfig


@pytest.mark.parametrize("value", [11, [8, 11, 14], [11]])
def test_cluster_count_config_round_trip(value, tmp_path):
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"cellcharter": {"n_clusters": value}}))
    config = load_config(path).cellcharter
    assert config.n_clusters == value
    assert CellCharterConfig.model_validate(config.model_dump()).n_clusters == value
    schema = CellCharterConfig.model_json_schema()["properties"]["n_clusters"]
    assert {branch["type"] for branch in schema["anyOf"]} == {"integer", "array"}


@pytest.mark.parametrize("value", [[], 0, -1, [2, 0], [2, -3], [2, 2], True,
                                  [2, False], 2.5, [2, 3.5], "2,3", None])
def test_invalid_cluster_counts(value):
    with pytest.raises(ValidationError):
        CellCharterConfig(n_clusters=value)


@pytest.fixture
def stage(monkeypatch):
    """Exercise real orchestration/exports with external scientific engines mocked."""
    module_name = "SpatialBiologyToolkit.scripts.cellcharter_neighborhoods"
    previous = sys.modules.pop(module_name, None)
    monkeypatch.setitem(sys.modules, "cellcharter", ModuleType("cellcharter"))
    monkeypatch.setitem(sys.modules, "squidpy", ModuleType("squidpy"))
    monkeypatch.setitem(sys.modules, "scanpy", None)
    monkeypatch.setitem(sys.modules, "SpatialBiologyToolkit.plotting", ModuleType("plotting"))
    module = importlib.import_module(module_name)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous


@pytest.fixture
def run_context(stage, monkeypatch, tmp_path):
    adata = ad.AnnData(np.arange(36, dtype=np.float32).reshape(12, 3))
    adata.obs["ROI"] = ["r1"] * 6 + ["r2"] * 6
    adata.obs["population"] = pd.Categorical(["A", "B"] * 6)
    adata.obs["condition"] = pd.Categorical(["control"] * 6 + ["treated"] * 6)
    adata.obs["case"] = adata.obs["ROI"].astype(str)
    adata.obsm["spatial"] = np.arange(24).reshape(12, 2).astype(float)
    adata.obsm["X_test"] = adata.X.copy()
    general = GeneralConfig(
        anndata_path=str(tmp_path / "cells.h5ad"), qc_folder=str(tmp_path / "qc"),
        population_obs_primary="population", case_obs="case", groupby_obs="condition",
    )
    config = CellCharterConfig(
        n_clusters=[2, 3], use_rep="X_test", scale_by_sample=False,
        cluster_default_cmap="tab10", save_high_res=False,
        run_enrichment=True, run_nhood_enrichment=True,
        run_diff_nhood_enrichment=True, run_shape_characterisation=True,
        shape_plot_metrics=False, save_enrichment_heatmap=False,
    )
    loader = Mock(return_value=(adata, None, False, None))
    monkeypatch.setattr(stage, "load_pipeline_anndata", loader)

    def save(**kwargs):
        output = Path(kwargs["override_path"])
        kwargs["adata"].write_h5ad(output)
        return output

    saver = Mock(side_effect=save)
    monkeypatch.setattr(stage, "save_pipeline_anndata", saver)

    def make_model(**kwargs):
        count = kwargs["n_clusters"]
        return SimpleNamespace(
            fit=Mock(), predict=Mock(return_value=np.arange(adata.n_obs) % count),
        )

    def aggregate(data, *, out_key, **kwargs):
        data.obsm[out_key] = data.obsm["X_test"].copy()

    def matrix(data, key):
        labels = data.obs[key].cat.categories
        return pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)

    def enrich(data, *, group_key, label_key, **kwargs):
        data.uns[f"{group_key}_{label_key}_enrichment"] = {
            "enrichment": pd.crosstab(data.obs[group_key], data.obs[label_key]).astype(float),
        }

    def nhood(data, *, cluster_key, **kwargs):
        data.uns[f"{cluster_key}_nhood_enrichment"] = {"enrichment": matrix(data, cluster_key)}

    def diff(data, *, cluster_key, condition_key, **kwargs):
        data.uns[f"{cluster_key}_{condition_key}_diff_nhood_enrichment"] = {
            "control_treated": {"enrichment": matrix(data, cluster_key)},
        }

    def components(data, *, cluster_key, out_key, **kwargs):
        data.obs[out_key] = data.obs[cluster_key].astype(str)

    def boundaries(data, *, cluster_key, **kwargs):
        data.uns[f"shape_{cluster_key}"] = {"boundary": {}}

    def metric(data, *, cluster_key, out_key, **kwargs):
        data.uns[f"shape_{cluster_key}"][out_key] = {
            label: 0.5 for label in data.obs[cluster_key].unique()
        }

    cc = SimpleNamespace(
        gr=SimpleNamespace(
            aggregate_neighbors=Mock(side_effect=aggregate), remove_long_links=Mock(),
            enrichment=Mock(side_effect=enrich), nhood_enrichment=Mock(side_effect=nhood),
            diff_nhood_enrichment=Mock(side_effect=diff), connected_components=Mock(side_effect=components),
        ),
        tl=SimpleNamespace(
            Cluster=Mock(side_effect=make_model), boundaries=Mock(side_effect=boundaries),
            linearity_metric=Mock(side_effect=metric), curl_metric=Mock(side_effect=metric),
        ),
        pl=SimpleNamespace(enrichment=Mock(), nhood_enrichment=Mock(), diff_nhood_enrichment=Mock()),
    )
    monkeypatch.setattr(stage, "cc", cc)
    graph = Mock()
    monkeypatch.setattr(stage, "sq", SimpleNamespace(gr=SimpleNamespace(spatial_neighbors=graph)))
    plots = {}
    for name in ("_save_cluster_umap", "_save_cluster_composition_plots", "_save_roi_cluster_masks"):
        plots[name] = Mock()
        monkeypatch.setattr(stage, name, plots[name])
    # Spatial plotting is spied on here and exercised with real figures below.
    spatial_plot = stage._save_spatial_cluster_plots
    plots["_save_spatial_cluster_plots"] = Mock()
    monkeypatch.setattr(stage, "_save_spatial_cluster_plots", plots["_save_spatial_cluster_plots"])
    return SimpleNamespace(adata=adata, general=general, config=config, cc=cc, graph=graph,
                           plots=plots, loader=loader, saver=saver, stage=stage, spatial_plot=spatial_plot)


def test_all_solutions_export_and_persist_separately(run_context):
    ctx = run_context
    original = ctx.config.model_dump()
    output = ctx.stage.run_cellcharter_neighborhoods(ctx.general, ctx.config)
    restored = ad.read_h5ad(output)
    assert ctx.config.model_dump() == original
    assert ctx.loader.call_count == ctx.saver.call_count == ctx.graph.call_count == 1
    assert ctx.cc.gr.aggregate_neighbors.call_count == 1
    assert [call.kwargs["n_clusters"] for call in ctx.cc.tl.Cluster.call_args_list] == [2, 3]
    assert "spatial_cluster" not in restored.obs
    for count in (2, 3):
        key = f"spatial_cluster_k{count}"
        assert restored.obs[key].nunique() == count
        assert len(restored.uns[f"{key}_colors"]) == count
        assert f"component_k{count}" in restored.obs
        assert f"shape_component_k{count}" in restored.uns
        for suffix in ("population_enrichment", "nhood_enrichment", "condition_diff_nhood_enrichment"):
            assert f"{key}_{suffix}" in restored.uns
        details = restored.uns["cellcharter_pipeline"]["solutions"][key]
        assert details["n_clusters"] == count
        for analysis in ("enrichment", "nhood_enrichment", "diff_nhood_enrichment", "shape_characterisation"):
            assert details[analysis]["ran"]
        folder = Path(details["qc_dir"])
        assert folder.name == f"n_clusters_{count}"
        counts = pd.read_csv(folder / "cluster_counts_global.csv")
        assert len(counts) == count
        assert counts.n_cells.sum() == restored.n_obs
        assert (folder / "cluster_counts_by_sample.csv").is_file()
        assert (folder / "enrichment_matrix.csv").is_file()
        assert (folder / "nhood_enrichment/enrichment.csv").is_file()
        assert (folder / "diff_nhood_enrichment/diff_nhood_enrichment_long.csv").is_file()
        assert (folder / "shape_characterisation/shape_metrics_by_component.csv").is_file()
    for plot in ctx.plots.values():
        assert [call.kwargs["cluster_key"] for call in plot.call_args_list] == [
            "spatial_cluster_k2", "spatial_cluster_k3",
        ]
        assert [call.kwargs["qc_dir"].name for call in plot.call_args_list] == ["n_clusters_2", "n_clusters_3"]
    for plot in (ctx.cc.pl.enrichment, ctx.cc.pl.nhood_enrichment, ctx.cc.pl.diff_nhood_enrichment):
        assert plot.call_count == 2


@pytest.mark.parametrize("value,key,folder", [(2, "spatial_cluster", ""), ([2], "spatial_cluster_k2", "n_clusters_2")])
def test_scalar_and_single_item_list_layout(run_context, value, key, folder):
    ctx = run_context
    ctx.config.n_clusters = value
    ctx.stage.run_cellcharter_neighborhoods(ctx.general, ctx.config)
    assert key in ctx.adata.obs
    assert ctx.cc.tl.Cluster.call_count == 1
    expected = Path(ctx.general.qc_folder) / ctx.config.qc_output_subdir / folder
    assert (expected / "cluster_counts_global.csv").is_file()
    if isinstance(value, int):
        assert ctx.adata.uns["cellcharter_pipeline"]["cluster_key"] == key
        assert "solutions" not in ctx.adata.uns["cellcharter_pipeline"]
        assert "component" in ctx.adata.obs


def test_reuse_is_independent_and_regenerates_outputs(run_context):
    ctx = run_context
    ctx.stage.run_cellcharter_neighborhoods(ctx.general, ctx.config)
    ctx.config.repeat_analysis = False
    ctx.config.n_clusters = [2, 3, 4]
    before = ctx.adata.obs["spatial_cluster_k2"].copy()
    ctx.stage.run_cellcharter_neighborhoods(ctx.general, ctx.config)
    assert [call.kwargs["n_clusters"] for call in ctx.cc.tl.Cluster.call_args_list] == [2, 3, 4]
    assert ctx.graph.call_count == ctx.cc.gr.aggregate_neighbors.call_count == 2
    pd.testing.assert_series_equal(before, ctx.adata.obs["spatial_cluster_k2"])
    results = ctx.adata.uns["cellcharter_pipeline"]["solutions"]
    for count in (2, 3):
        assert results[f"spatial_cluster_k{count}"]["reused_existing_cluster_key"]
        for analysis in ("enrichment", "nhood_enrichment", "diff_nhood_enrichment", "shape_characterisation"):
            assert results[f"spatial_cluster_k{count}"][analysis]["reused_existing"]
    assert not results["spatial_cluster_k4"]["reused_existing_cluster_key"]
    # A complete rerun reuses all scientific results while exporting again.
    counts_path = Path(results["spatial_cluster_k2"]["qc_dir"]) / "cluster_counts_global.csv"
    counts_path.unlink()
    ctx.stage.run_cellcharter_neighborhoods(ctx.general, ctx.config)
    assert counts_path.exists()
    assert ctx.graph.call_count == ctx.cc.gr.aggregate_neighbors.call_count == 2
    for method in (ctx.cc.tl.Cluster, ctx.cc.gr.enrichment, ctx.cc.gr.nhood_enrichment,
                   ctx.cc.gr.diff_nhood_enrichment, ctx.cc.gr.connected_components):
        assert method.call_count == 3
    for plot in ctx.plots.values():
        assert plot.call_count == 8


def test_too_many_clusters_fails_before_preparation(run_context):
    ctx = run_context
    ctx.config.n_clusters = [2, 13]
    with pytest.raises(ValueError, match="number of cells"):
        ctx.stage.run_cellcharter_neighborhoods(ctx.general, ctx.config)
    ctx.graph.assert_not_called()
    ctx.cc.tl.Cluster.assert_not_called()
    ctx.saver.assert_not_called()


def test_trvae_prepared_once(run_context, monkeypatch):
    ctx = run_context
    ctx.config.use_trvae = True
    trvae = Mock(return_value=("X_test", {"enabled": True, "ran": True}))
    monkeypatch.setattr(ctx.stage, "_compute_trvae_representation", trvae)
    ctx.stage.run_cellcharter_neighborhoods(ctx.general, ctx.config)
    trvae.assert_called_once()
    assert ctx.cc.gr.aggregate_neighbors.call_count == 1
    assert ctx.cc.tl.Cluster.call_count == 2


def test_custom_shape_references_follow_current_solution(stage):
    config = CellCharterConfig(n_clusters=[2, 3], cluster_key="niche",
                              shape_component_key="regions", shape_component_cluster_key="niche",
                              shape_metrics_cluster_key="external_annotation")
    solutions = stage._cellcharter_solution_configs(config)
    assert [item.shape_component_key for item in solutions] == ["regions_k2", "regions_k3"]
    assert [item.shape_component_cluster_key for item in solutions] == ["niche_k2", "niche_k3"]
    assert all(item.shape_metrics_cluster_key == "external_annotation" for item in solutions)
    config.shape_metrics_cluster_key = "niche"
    assert stage._cellcharter_solution_configs(config)[0].shape_metrics_cluster_key == "niche_k2"


def test_spatial_plots_are_written_for_every_count(run_context):
    ctx = run_context
    # Supply the tiny fixture directly to the real plotting helper.
    for count in (2, 3):
        key = f"spatial_cluster_k{count}"
        ctx.adata.obs[key] = pd.Categorical((np.arange(ctx.adata.n_obs) % count).astype(str))
        folder = Path(ctx.general.qc_folder) / f"n_clusters_{count}"
        folder.mkdir(parents=True)
        ctx.spatial_plot(
            ctx.adata, "ROI", "spatial", key, 2, 1, folder,
            cluster_color_map={str(i): "#224466" for i in range(count)}, save_high_res=False,
        )
        assert len(list(folder.glob("spatial_clusters_*.png"))) == 1
