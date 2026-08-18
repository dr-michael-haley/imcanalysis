from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from SpatialBiologyToolkit.napari_sbt.scanpy_plotting import (
    ScanpyPlotRequest,
    build_scanpy_plot,
    deterministic_stratified_indices,
    expression_group_summary,
    matrix_source_choices,
    resolve_plot_cell_mask,
)


def _adata() -> SimpleNamespace:
    obs = pd.DataFrame(
        {
            "ROI": pd.Categorical(["A", "A", "B", "B", "C", "C"]),
            "leiden": pd.Categorical(
                ["0", "0", "1", "1", "2", "2"],
                categories=["0", "1", "2"],
            ),
            "population_named": pd.Categorical(
                ["T cell", "T cell", "Macrophage", "Myeloid", "Tumour", "Tumour"]
            ),
            "patient": pd.Categorical(["P1", "P1", "P1", "P1", "P2", "P2"]),
        },
        index=[f"cell_{index}" for index in range(6)],
    )
    return SimpleNamespace(
        obs=obs,
        obs_names=obs.index,
        n_obs=len(obs),
        var_names=pd.Index(["CD3", "CD68", "PanCK"]),
        X=np.asarray(
            [
                [8.0, 1.0, 0.0],
                [7.0, 1.0, 0.0],
                [1.0, 8.0, 0.0],
                [1.0, 7.0, 0.0],
                [0.0, 1.0, 9.0],
                [0.0, 1.0, 8.0],
            ]
        ),
        raw=None,
        layers={"scaled": np.arange(18, dtype=float).reshape(6, 3)},
        obsm={"X_umap": np.arange(12, dtype=float).reshape(6, 2)},
        uns={
            "population_named_colors": [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
            ]
        },
    )


def test_plot_scope_combines_selected_population_and_roi_filters():
    adata = _adata()
    request = ScanpyPlotRequest(
        plot_type="embedding",
        groupby="population_named",
        cell_scope="selected_groups",
        selected_groups=["T cell", "Tumour"],
        roi_obs="ROI",
        selected_rois=["A", "C"],
        embedding_key="X_umap",
    )

    mask = resolve_plot_cell_mask(adata, request)

    assert mask.tolist() == [True, True, False, False, True, True]


def test_plot_scope_can_use_frozen_cohort_obs_names():
    adata = _adata()
    request = ScanpyPlotRequest(
        plot_type="embedding",
        groupby="population_named",
        cell_scope="cohort",
        embedding_key="X_umap",
    )

    mask = resolve_plot_cell_mask(
        adata,
        request,
        cohort_obs_names={"cell_1", "cell_4"},
    )

    assert adata.obs_names[mask].tolist() == ["cell_1", "cell_4"]


def test_expression_summary_slices_selected_markers_and_scales_per_marker():
    adata = _adata()
    request = ScanpyPlotRequest(
        plot_type="dotplot",
        groupby="leiden",
        matrix_source="X",
        markers=["CD3", "CD68"],
        expression_scale="zscore_marker",
        positivity_threshold=2.0,
    )

    summary = expression_group_summary(
        adata,
        request,
        np.ones(adata.n_obs, dtype=bool),
    )

    assert set(summary["marker"]) == {"CD3", "CD68"}
    assert len(summary) == 6
    assert (
        summary.loc[
            (summary["population"] == "0") & (summary["marker"] == "CD3"),
            "fraction_positive",
        ].item()
        == 1.0
    )
    assert np.isclose(
        summary.groupby("marker")["display_value"].mean().to_numpy(), 0
    ).all()


def test_embedding_downsampling_is_deterministic_and_keeps_each_population():
    labels = np.asarray(["common"] * 100 + ["rare"] * 3 + ["other"] * 20)

    first = deterministic_stratified_indices(labels, maximum=20, seed=7)
    second = deterministic_stratified_indices(labels, maximum=20, seed=7)

    np.testing.assert_array_equal(first, second)
    assert set(labels[first]) == {"common", "rare", "other"}


def test_building_label_comparison_returns_exportable_values_without_mutation():
    adata = _adata()
    before = adata.obs.copy(deep=True)
    request = ScanpyPlotRequest(
        plot_type="label_comparison",
        groupby="population_named",
        comparison_obs="leiden",
        comparison_normalisation="row_percent",
    )

    artifact = build_scanpy_plot(adata, request)

    assert artifact.cell_count == adata.n_obs
    assert set(artifact.data.columns) == {
        "leiden",
        "population_named",
        "value",
        "normalisation",
    }
    pd.testing.assert_frame_equal(adata.obs, before)


def test_expression_sources_include_x_and_layers():
    assert matrix_source_choices(_adata()) == ["X", "layer::scaled"]
