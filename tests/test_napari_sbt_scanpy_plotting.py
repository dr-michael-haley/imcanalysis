from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd

import SpatialBiologyToolkit.napari_sbt.scanpy_plotting as plotting_module
from SpatialBiologyToolkit.napari_sbt.scanpy_plotting import (
    PLOT_TYPE_LABELS,
    ScanpyPlotRequest,
    build_scanpy_plot,
    deterministic_stratified_indices,
    expression_group_summary,
    matrix_source_choices,
    resolve_plot_cell_mask,
    sample_level_obs_columns,
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


def test_sample_metadata_filter_combines_with_the_plot_scope():
    adata = _adata()
    assert "patient" in sample_level_obs_columns(adata, roi_obs="ROI")
    assert "population_named" not in sample_level_obs_columns(adata, roi_obs="ROI")
    request = ScanpyPlotRequest(
        plot_type="composition_bar",
        groupby="population_named",
        composition_obs="ROI",
        metadata_filter_obs="patient",
        metadata_filter_values=["P2"],
    )

    mask = resolve_plot_cell_mask(adata, request)

    assert adata.obs_names[mask].tolist() == ["cell_4", "cell_5"]


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


def test_embedding_display_limit_never_drops_an_entire_population():
    labels = np.asarray([f"population_{index}" for index in range(150)])

    selected = deterministic_stratified_indices(labels, maximum=100)

    assert len(selected) == 150
    assert set(labels[selected]) == set(labels)


def test_native_scanpy_renderers_are_explicit_in_the_plot_choices():
    for plot_type in ("embedding", "heatmap", "dotplot", "violin"):
        assert "Scanpy" in PLOT_TYPE_LABELS[plot_type]


def test_embedding_leaves_rasterization_control_to_scanpy(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAnnData:
        def __init__(self, X, obs):
            self.X = X
            self.obs = obs
            self.obsm: dict[str, object] = {}
            self.uns: dict[str, object] = {}

    def embedding(_adata, **kwargs):
        from matplotlib.figure import Figure

        captured.update(kwargs)
        return Figure()

    fake_scanpy = SimpleNamespace(pl=SimpleNamespace(embedding=embedding))
    monkeypatch.setattr(
        plotting_module,
        "_native_scanpy_modules",
        lambda: (fake_scanpy, FakeAnnData),
    )
    adata = _adata()
    request = ScanpyPlotRequest(
        plot_type="embedding",
        groupby="population_named",
        embedding_key="X_umap",
    )

    artifact = build_scanpy_plot(adata, request)

    assert artifact.cell_count == adata.n_obs
    assert "rasterized" not in captured
    assert artifact.figure.get_layout_engine() is not None


def test_embedding_can_render_multiple_expression_variables(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAnnData:
        def __init__(self, X, obs, var=None):
            self.X = X
            self.obs = obs
            self.var = var
            self.obsm: dict[str, object] = {}
            self.uns: dict[str, object] = {}

    def embedding(plot_adata, **kwargs):
        from matplotlib.figure import Figure

        captured.update(kwargs)
        captured["X"] = np.asarray(plot_adata.X)
        captured["var_names"] = list(plot_adata.var.index)
        figure = Figure()
        axis = figure.add_subplot(111)
        axis.set_xlabel("UMAP1")
        axis.set_ylabel("UMAP2")
        return figure

    fake_scanpy = SimpleNamespace(pl=SimpleNamespace(embedding=embedding))
    monkeypatch.setattr(
        plotting_module,
        "_native_scanpy_modules",
        lambda: (fake_scanpy, FakeAnnData),
    )
    adata = _adata()
    request = ScanpyPlotRequest(
        plot_type="embedding",
        groupby="population_named",
        embedding_key="X_umap",
        embedding_markers=["CD3", "CD68"],
        embedding_ncols=2,
        embedding_colormap="magma",
        show_legend=False,
    )

    artifact = build_scanpy_plot(adata, request)

    assert captured["color"] == ["CD3", "CD68"]
    assert captured["ncols"] == 2
    assert captured["cmap"] == "magma"
    assert captured["colorbar_loc"] is None
    assert captured["var_names"] == ["CD3", "CD68"]
    np.testing.assert_array_equal(captured["X"], adata.X[:, :2])
    assert {"CD3", "CD68"}.issubset(artifact.data.columns)


def test_common_axis_display_options_hide_labels_and_ticks():
    from matplotlib.figure import Figure

    figure = Figure()
    axis = figure.add_subplot(111)
    axis.plot([0, 1], [0, 1])
    axis.set_xlabel("Embedding 1")
    axis.set_ylabel("Embedding 2")
    request = ScanpyPlotRequest(
        plot_type="embedding",
        groupby="population_named",
        embedding_key="X_umap",
        show_x_axis_label=False,
        show_y_axis_label=False,
        show_x_ticks=False,
        show_y_ticks=False,
    )

    plotting_module._apply_axis_display_options(axis, request)

    assert axis.get_xlabel() == ""
    assert axis.get_ylabel() == ""
    assert not any(label.get_visible() for label in axis.get_xticklabels())
    assert not any(label.get_visible() for label in axis.get_yticklabels())


def test_scanpy_layout_fitter_reserves_dynamic_space_for_long_labels():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(4, 3), dpi=100)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    axis.set_yticks([0.5], labels=["Smooth muscle / myofibroblast population"])
    baseline = plotting_module.figure_subplot_margins(figure)

    changed = plotting_module.fit_scanpy_figure_to_canvas(
        figure,
        baseline_margins=baseline,
    )

    assert changed is True
    assert figure.subplotpars.left > baseline[0]
    renderer = figure.canvas.get_renderer()
    assert axis.get_tightbbox(renderer).x0 >= 11.0


def test_expression_marker_order_uses_shared_utility_on_plotting_data(monkeypatch):
    calls: list[tuple[object, list[str]]] = []
    fake_utils = ModuleType("SpatialBiologyToolkit.utils")

    def reorder_vars_by_expression(adata, markers):
        calls.append((adata, list(markers)))
        return list(reversed(markers))

    fake_utils.reorder_vars_by_expression = reorder_vars_by_expression
    monkeypatch.setitem(sys.modules, fake_utils.__name__, fake_utils)
    plotting_adata = SimpleNamespace()
    request = ScanpyPlotRequest(
        plot_type="heatmap",
        groupby="population_named",
        markers=["CD3", "CD68", "PanCK"],
        reorder_markers_by_expression=True,
    )

    ordered = plotting_module._ordered_expression_markers(plotting_adata, request)

    assert ordered == ["PanCK", "CD68", "CD3"]
    assert calls == [(plotting_adata, ["CD3", "CD68", "PanCK"])]


def test_population_colour_strip_uses_plotting_anndata_palette():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(4, 3), dpi=100)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    axis.set_yticks([0.5, 1.5], labels=["T cell", "Tumour"])
    plot_adata = SimpleNamespace(
        obs=pd.DataFrame(
            {
                "population_named": pd.Categorical(
                    ["T cell", "Tumour"], categories=["T cell", "Tumour"]
                )
            }
        ),
        uns={"population_named_colors": ["#112233", "#abcdef"]},
    )
    plot_object = SimpleNamespace(ax_dict={"mainplot_ax": axis}, are_axes_swapped=False)
    request = ScanpyPlotRequest(
        plot_type="heatmap",
        groupby="population_named",
        markers=["CD3"],
        show_population_colours=True,
        population_colour_label_gap=37,
        population_colour_box_width=18,
    )

    plotting_module._add_population_colour_strip(plot_object, plot_adata, request)

    colour_boxes = [
        line for line in axis.lines if line.get_gid() == "napari_sbt_population_colour"
    ]
    assert len(colour_boxes) == 2
    assert all(line.get_linewidth() == 18 for line in colour_boxes)
    assert all(tick.get_pad() == 37 for tick in axis.yaxis.get_major_ticks())


def test_fresh_dendrogram_uses_current_temporary_plot_data_only():
    dendrogram_calls: list[dict[str, object]] = []

    def dendrogram(plot_adata, **kwargs):
        dendrogram_calls.append({"adata": plot_adata, **kwargs})
        plot_adata.uns[str(kwargs["key_added"])] = {"fresh": True}

    class FakePlot:
        def __init__(self):
            self.dendrogram_keys: list[str] = []
            self.axes_swapped = False

        def add_dendrogram(self, *, dendrogram_key):
            self.dendrogram_keys.append(dendrogram_key)

        def swap_axes(self):
            self.axes_swapped = True

    temporary_adata = SimpleNamespace(uns={})
    plot = FakePlot()
    request = ScanpyPlotRequest(
        plot_type="heatmap",
        groupby="population_named",
        markers=["CD3", "CD68"],
        side_annotation="dendrogram",
        dendrogram_correlation="spearman",
        dendrogram_linkage="average",
        dendrogram_optimal_ordering=False,
        swap_axes=True,
    )

    plotting_module._configure_scanpy_baseplot(
        SimpleNamespace(tl=SimpleNamespace(dendrogram=dendrogram)),
        temporary_adata,
        plot,
        request,
        ["T cell", "Myeloid", "Tumour"],
    )

    assert len(dendrogram_calls) == 1
    assert dendrogram_calls[0]["adata"] is temporary_adata
    assert dendrogram_calls[0]["var_names"] == ["CD3", "CD68"]
    assert dendrogram_calls[0]["cor_method"] == "spearman"
    assert dendrogram_calls[0]["linkage_method"] == "average"
    assert plot.dendrogram_keys == ["_napari_sbt_fresh_dendrogram"]
    assert plot.axes_swapped is True


def test_population_totals_are_mutually_exclusive_with_dendrograms():
    class FakePlot:
        def __init__(self):
            self.sort = "unset"

        def add_totals(self, *, sort):
            self.sort = sort

    plot = FakePlot()
    request = ScanpyPlotRequest(
        plot_type="dotplot",
        groupby="population_named",
        markers=["CD3"],
        side_annotation="totals",
        totals_sort="descending",
    )

    plotting_module._configure_scanpy_baseplot(
        SimpleNamespace(),
        SimpleNamespace(uns={}),
        plot,
        request,
        ["T cell", "Myeloid", "Tumour"],
    )

    assert plot.sort == "descending"


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


def test_composition_bar_sort_spacing_edges_and_filtered_autotitle():
    request = ScanpyPlotRequest(
        plot_type="composition_bar",
        groupby="population_named",
        composition_obs="ROI",
        composition_measure="count",
        metadata_filter_obs="patient",
        metadata_filter_values=["P1", "P2"],
        bar_sort_population="Tumour",
        bar_sort_direction="descending",
        bar_width=0.5,
        bar_start_padding=0.75,
        bar_end_padding=1.25,
        edge_color="#123456",
        edge_width=1.5,
        bar_manual_y_limits=True,
        bar_y_min=5.0,
        bar_y_max=25.0,
    )

    artifact = build_scanpy_plot(_adata(), request)
    axis = artifact.figure.axes[0]

    assert "patient: P1, P2" in artifact.title
    assert artifact.data["ROI"].iloc[0] == "C"
    assert all(np.isclose(patch.get_width(), 0.5) for patch in axis.patches)
    assert np.allclose(axis.get_xlim(), (-1.0, 3.5))
    assert np.allclose(axis.get_ylim(), (5.0, 25.0))
    assert all(np.isclose(patch.get_linewidth(), 1.5) for patch in axis.patches)


def test_composition_heatmap_supports_colormap_edges_population_colours_and_no_title():
    request = ScanpyPlotRequest(
        plot_type="composition_heatmap",
        groupby="population_named",
        composition_obs="ROI",
        heatmap_colormap="plasma",
        edge_color="#111111",
        edge_width=2.0,
        show_population_colours=True,
        title_mode="hidden",
    )

    artifact = build_scanpy_plot(_adata(), request)
    axis = artifact.figure.axes[0]
    mesh = axis.collections[0]

    assert artifact.title == "population_named abundance across ROI"
    assert axis.get_title() == ""
    assert mesh.get_cmap().name == "plasma"
    assert np.allclose(mesh.get_linewidths(), 2.0)
    assert (
        len(
            [
                line
                for line in axis.lines
                if line.get_gid() == "napari_sbt_population_colour"
            ]
        )
        == 4
    )


def test_expression_sources_include_x_and_layers():
    assert matrix_source_choices(_adata()) == ["X", "layer::scaled"]
