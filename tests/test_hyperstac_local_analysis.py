from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tifffile

from SpatialBiologyToolkit.hyperstac import local_analysis
from SpatialBiologyToolkit.hyperstac.local_analysis import (
    FigureSaveOptions,
    HeatmapOptions,
    assign_cells_to_hyperstac_masks,
    hyperstac_cluster_feature_tables,
    plot_cell_environment_composition,
    plot_cluster_map_gallery,
    plot_environment_abundance,
    plot_hyperstac_cluster_features,
    plot_hyperstac_environment_gallery,
    plot_hyperstac_umap,
    prepare_environment_abundance_tables,
    reconstruct_cluster_label_masks,
    summarize_cell_environment_composition,
    summarize_environment_abundance,
)


def _representation() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "roi": ["roi_1", "roi_1", "roi_2", "roi_2"],
            "row_start": [0, 0, 0, 0],
            "row_end": [2, 2, 2, 2],
            "col_start": [0, 2, 0, 2],
            "col_end": [2, 4, 2, 4],
            "roi_height_px": [4, 4, 4, 4],
            "roi_width_px": [4, 4, 4, 4],
            "leiden_test": pd.Categorical(["0", "1", "0", "1"]),
        },
        index=["p0", "p1", "p2", "p3"],
    )
    result = ad.AnnData(X=np.zeros((4, 2), dtype=np.float32), obs=obs)
    result.obsm["X_umap"] = np.arange(8, dtype=np.float32).reshape(4, 2)
    return result


def _metrics() -> ad.AnnData:
    var = pd.DataFrame(
        {
            "metric": ["mean_intensity_norm"] * 3,
            "channel": ["A", "B", "C"],
            "is_channel_metric": [True, True, True],
        },
        index=[
            "mean_intensity_norm_A",
            "mean_intensity_norm_B",
            "mean_intensity_norm_C",
        ],
    )
    return ad.AnnData(
        X=np.array(
            [
                [0.9, 0.2, 0.1],
                [0.1, 0.8, 0.2],
                [0.8, 0.1, 0.2],
                [0.2, 0.9, 0.1],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame(index=["p0", "p1", "p2", "p3"]),
        var=var,
    )


def _permutation() -> ad.AnnData:
    var = pd.DataFrame(
        {
            "perturbation_type": ["zero_channel"] * 3 + ["shuffle_channel"] * 3,
            "channel": ["A", "B", "C"] * 2,
        },
        index=[f"condition_{index}" for index in range(6)],
    )
    values = np.array(
        [
            [0.9, 0.2, 0.1, 0.7, 0.2, 0.1],
            [0.1, 0.8, 0.2, 0.2, 0.7, 0.2],
            [0.8, 0.1, 0.2, 0.6, 0.1, 0.2],
            [0.2, 0.9, 0.1, 0.1, 0.8, 0.1],
        ],
        dtype=np.float32,
    )
    return ad.AnnData(values, obs=pd.DataFrame(index=["p0", "p1", "p2", "p3"]), var=var)


def _write_normalised_images(folder) -> None:
    for roi, offset in (("roi_1", 0.0), ("roi_2", 0.1)):
        roi_dir = folder / roi
        roi_dir.mkdir(parents=True)
        base = np.arange(16, dtype=np.float32).reshape(4, 4) / 15
        for channel_index, channel in enumerate(("A", "B", "C")):
            tifffile.imwrite(
                roi_dir / f"{channel}.tiff", base + offset + channel_index * 0.05
            )


def test_reconstruct_cluster_label_masks_preserves_image_shape_and_zero_gaps(tmp_path):
    normalised = tmp_path / "normalised"
    _write_normalised_images(normalised)

    result = reconstruct_cluster_label_masks(
        _representation(),
        "leiden_test",
        tmp_path / "output",
        normalised_image_dir=normalised,
        palette={"0": "#ff0000", "1": "#00ff00"},
    )

    roi_1 = tifffile.imread(result.mask_dir / "roi_1__hyperstac_labels.tiff")
    assert roi_1.shape == (4, 4)
    np.testing.assert_array_equal(
        roi_1,
        np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint16,
        ),
    )
    assert result.mapping.to_dict("records") == [
        {"cluster": "0", "mask_value": 1},
        {"cluster": "1", "mask_value": 2},
    ]
    assert result.colorized_mask_dir is not None
    colorized_tiff = tifffile.imread(
        result.colorized_mask_dir / "roi_1__hyperstac_labels_colorized.tiff"
    )
    assert colorized_tiff.shape == (4, 4, 3)
    assert colorized_tiff.dtype == np.uint8
    np.testing.assert_array_equal(colorized_tiff[0, 0], [255, 0, 0])
    np.testing.assert_array_equal(colorized_tiff[0, 2], [0, 255, 0])
    np.testing.assert_array_equal(colorized_tiff[3, 0], [0, 0, 0])
    colorized_png = plt.imread(
        result.colorized_mask_dir / "roi_1__hyperstac_labels_colorized.png"
    )
    assert colorized_png.shape[:2] == roi_1.shape
    assert result.manifest.loc[0, "colorized_tiff_path"].endswith(
        "roi_1__hyperstac_labels_colorized.tiff"
    )
    assert result.manifest.loc[0, "colorized_png_path"].endswith(
        "roi_1__hyperstac_labels_colorized.png"
    )
    assert (
        result.manifest.loc[
            result.manifest["roi"] == "roi_1", "coverage_fraction"
        ].item()
        == 0.5
    )


def test_reconstruct_cluster_label_masks_rejects_unexpected_overlap(tmp_path):
    representation = _representation()
    representation.obs.loc["p1", ["col_start", "col_end"]] = [1, 3]

    with pytest.raises(ValueError, match="overlaps a previous patch"):
        reconstruct_cluster_label_masks(representation, "leiden_test", tmp_path)


def test_gallery_uses_normalised_tiffs_without_saved_patch_arrays(tmp_path):
    normalised = tmp_path / "normalised"
    _write_normalised_images(normalised)

    result = plot_hyperstac_environment_gallery(
        _representation(),
        _metrics(),
        "leiden_test",
        normalised,
        tmp_path / "gallery" / "environments",
        permutation_adata=_permutation(),
        examples_per_cluster=2,
    )

    assert {path.suffix for path in result.image_paths} == {".png", ".svg"}
    assert all(path.is_file() for path in result.image_paths)
    assert set(result.marker_table["cluster"]) == {"0", "1"}
    assert len(result.selection_table) == 4
    assert result.selection_table.groupby("cluster")["roi"].nunique().eq(2).all()


def test_gallery_exposes_boxed_marker_key_fonts_titles_spacing_and_outline(
    tmp_path, monkeypatch
):
    normalised = tmp_path / "normalised"
    _write_normalised_images(normalised)
    captured: list[plt.Figure] = []
    original_close = plt.close

    def capture_figure(fig, output_stem, *, options):
        captured.append(fig)
        return (Path(output_stem).with_suffix(".png"),)

    monkeypatch.setattr(local_analysis, "_save_figure", capture_figure)
    monkeypatch.setattr(local_analysis.plt, "close", lambda _fig: None)
    plot_hyperstac_environment_gallery(
        _representation(),
        _metrics(),
        "leiden_test",
        normalised,
        tmp_path / "gallery" / "styled",
        permutation_adata=_permutation(),
        examples_per_cluster=2,
        show_roi_titles=False,
        show_title=False,
        cluster_label_fontsize=12,
        marker_label_fontsize=11,
        row_spacing=0.31,
        column_spacing=0,
        write_tables=False,
    )

    fig = captured[0]
    fig.canvas.draw()
    assert fig._suptitle is None
    assert fig.subplotpars.hspace == pytest.approx(0.31)
    assert all(not ax.get_title() for ax in fig.axes)
    image_axes = [ax for ax in fig.axes if ax.images]
    first_row_axes = image_axes[:2]
    assert (
        first_row_axes[1].get_position().x0 - first_row_axes[0].get_position().x1 < 0.01
    )
    patch_outlines = [
        patch
        for ax in image_axes
        for patch in ax.patches
        if patch.get_x() == 0 and patch.get_y() == 0
    ]
    assert patch_outlines
    assert all(patch.get_linewidth() == pytest.approx(0.5) for patch in patch_outlines)
    key_boxes = [patch for ax in fig.axes for patch in ax.patches if patch.get_fill()]
    assert key_boxes
    assert all(
        patch.get_facecolor()[:3] == pytest.approx((0, 0, 0)) for patch in key_boxes
    )
    marker_text = {
        text.get_text(): text
        for ax in fig.axes
        for text in ax.texts
        if text.get_text() in {"A", "B", "C"}
    }
    assert marker_text
    assert {text.get_fontsize() for text in marker_text.values()} == {11}
    original_close(fig)


def test_feature_and_abundance_tables_are_aligned_and_complete():
    representation = _representation()
    tables = hyperstac_cluster_feature_tables(
        representation,
        _metrics(),
        "leiden_test",
        permutation_adata=_permutation(),
    )
    assert set(tables) >= {
        "mean_marker_intensity",
        "relative_marker_intensity_zscore",
        "mean_zero_channel_cosine_distance",
        "mean_shuffle_channel_cosine_distance",
    }
    assert tables["mean_marker_intensity"].loc["0", "A"] == pytest.approx(0.85)

    metadata = pd.DataFrame(
        {
            "ROI_number": ["roi_1", "roi_2"],
            "Case": ["case_a", "case_b"],
        }
    )
    abundance = summarize_environment_abundance(
        representation,
        "leiden_test",
        metadata=metadata,
    )
    assert len(abundance) == 4
    assert abundance.groupby("roi")["fraction"].sum().eq(1).all()
    assert set(abundance["Case"]) == {"case_a", "case_b"}


def test_cell_assignment_reports_uncovered_and_out_of_bounds_cells(tmp_path):
    mask_result = reconstruct_cluster_label_masks(
        _representation(), "leiden_test", tmp_path
    )
    cells = pd.DataFrame(
        {
            "ROI": ["roi_1"] * 4,
            "X_loc": [0.5, 2.5, 0.5, 9.0],
            "Y_loc": [0.5, 0.5, 3.0, 0.5],
            "population": ["A", "A", "B", "B"],
        },
        index=["c0", "c1", "c2", "c3"],
    )
    assignments = assign_cells_to_hyperstac_masks(
        cells,
        mask_result.mask_dir,
        mask_result.mapping,
        rois=["roi_1"],
    )

    assert assignments["status"].tolist() == [
        "assigned",
        "assigned",
        "uncovered",
        "out_of_bounds",
    ]
    assert assignments.loc["c0", "environment"] == "0"
    assert assignments.loc["c1", "environment"] == "1"
    composition = summarize_cell_environment_composition(assignments)
    assert composition["cell_count"].sum() == 2
    assert (
        composition.groupby("environment")["fraction_within_environment"]
        .sum()
        .eq(1)
        .all()
    )


def test_heatmap_and_umap_customization_preserve_data_and_output_contract(
    tmp_path, monkeypatch
):
    representation = _representation()
    tables = hyperstac_cluster_feature_tables(
        representation,
        _metrics(),
        "leiden_test",
        cluster_order=["1", "0"],
        channel_order=["B", "A"],
        summary_statistic="median",
        zscore_ddof=0,
    )
    assert list(tables["median_marker_intensity"].index) == ["1", "0"]
    assert list(tables["median_marker_intensity"].columns) == ["B", "A"]

    heatmap_outputs = plot_hyperstac_cluster_features(
        {"custom_zscore": tables["relative_marker_intensity_zscore"]},
        tmp_path / "features",
        heatmap_options=HeatmapOptions(
            transpose=True,
            row_order=["A", "B"],
            column_order=["0", "1"],
            cmap="coolwarm",
            center=0.0,
            annot=True,
            fmt=".1f",
            show_clustermap=False,
            x_tick_rotation=0,
            figsize=(4, 3),
        ),
        save_options=FigureSaveOptions(formats=("png",), dpi=72, transparent=True),
    )
    assert set(heatmap_outputs["custom_zscore"]) == {"fixed"}
    assert heatmap_outputs["custom_zscore"]["fixed"][0].suffix == ".png"

    clustered_outputs = plot_hyperstac_cluster_features(
        {"custom_zscore": tables["relative_marker_intensity_zscore"]},
        tmp_path / "clustered_features",
        heatmap_options=HeatmapOptions(
            show_fixed=False,
            figsize=(5, 4),
            dendrogram_ratio=(0.08, 0.25),
            clustermap_kws={
                "row_colors": pd.Series(
                    {"0": "#112233", "1": "#abcdef"}, name="environment"
                ),
                "tree_kws": {"linewidths": 0.4},
            },
        ),
        table_options={
            "custom_zscore": {
                "title_fontsize": 11,
                "x_tick_fontsize": 8,
                "clustermap_kws": {"cbar_pos": (0.02, 0.8, 0.03, 0.15)},
            }
        },
        formats=("png",),
        dpi=72,
    )
    assert set(clustered_outputs["custom_zscore"]) == {"clustermap"}

    scanpy_call = {}

    def fake_umap(plot_adata, **kwargs):
        scanpy_call["adata"] = plot_adata
        scanpy_call["kwargs"] = kwargs
        coordinates = np.asarray(plot_adata.obsm["X_umap"])
        kwargs["ax"].scatter(coordinates[:, 0], coordinates[:, 1], s=kwargs["size"])

    monkeypatch.setitem(
        sys.modules,
        "scanpy",
        SimpleNamespace(pl=SimpleNamespace(umap=fake_umap)),
    )
    umap_paths = plot_hyperstac_umap(
        representation,
        "leiden_test",
        tmp_path / "umap" / "custom",
        cluster_order=["1", "0"],
        palette={"0": "#112233", "1": "#abcdef"},
        cluster_labels={"0": "Matrix", "1": "Vascular"},
        point_size=12,
        alpha=0.5,
        marker="s",
        show_legend=False,
        axis_off=True,
        shuffle_points=True,
        formats=("png",),
        dpi=80,
    )
    assert len(umap_paths) == 1
    assert umap_paths[0].is_file()
    assert scanpy_call["kwargs"]["color"] == "HyPERSTAC environment"
    assert scanpy_call["kwargs"]["show"] is False
    assert scanpy_call["kwargs"]["palette"] == ["#abcdef", "#112233"]
    assert list(scanpy_call["adata"].obs["HyPERSTAC environment"].cat.categories) == [
        "Vascular",
        "Matrix",
    ]


def test_map_and_gallery_accept_manual_publication_customization(tmp_path):
    normalised = tmp_path / "normalised"
    _write_normalised_images(normalised)
    representation = _representation()
    mask_result = reconstruct_cluster_label_masks(
        representation,
        "leiden_test",
        tmp_path / "masks",
        normalised_image_dir=normalised,
        cluster_order=["1", "0"],
    )

    map_paths = plot_cluster_map_gallery(
        mask_result,
        tmp_path / "maps" / "overlay",
        rois=["roi_1"],
        palette={"0": "cyan", "1": "magenta"},
        cluster_labels={"0": "Matrix", "1": "Vascular"},
        show_uncovered=False,
        background_image_dir=normalised,
        background_channel="A",
        mask_alpha=0.45,
        roi_title_template="{roi}: {coverage_fraction:.0%}",
        show_legend=False,
        title=None,
        formats=("png",),
        dpi=72,
    )
    assert len(map_paths) == 1 and map_paths[0].is_file()

    gallery = plot_hyperstac_environment_gallery(
        representation,
        _metrics(),
        "leiden_test",
        normalised,
        tmp_path / "gallery" / "manual",
        permutation_adata=_permutation(),
        cluster_order=["1", "0"],
        markers_by_cluster={"0": ["A", "B", "C"], "1": ["B", "A", "C"]},
        patch_ids_by_cluster={"0": ["p2", "p0"], "1": ["p3", "p1"]},
        examples_per_cluster=2,
        contrast_mode="fixed",
        fixed_vmin=0.0,
        fixed_vmax={"A": 1.0, "B": 1.1, "C": 1.2},
        gamma=0.8,
        channel_weights=(1.0, 0.8, 0.6),
        show_patch_titles=False,
        show_marker_labels=False,
        show_title=False,
        border_color="white",
        border_width=1.0,
        formats=("png",),
        dpi=72,
    )
    assert gallery.marker_table["cluster"].tolist() == ["1", "0"]
    assert gallery.marker_table["manual_markers"].all()
    assert (
        gallery.selection_table.groupby("cluster")["selection_strategy"]
        .first()
        .eq("manual")
        .all()
    )
    assert gallery.image_paths[0].is_file()


def test_cluster_map_exposes_titles_fonts_spacing_and_panel_outline(
    tmp_path, monkeypatch
):
    normalised = tmp_path / "normalised"
    _write_normalised_images(normalised)
    mask_result = reconstruct_cluster_label_masks(
        _representation(),
        "leiden_test",
        tmp_path / "masks",
        normalised_image_dir=normalised,
    )
    captured: list[plt.Figure] = []
    original_close = plt.close

    def capture_figure(fig, output_stem, *, options):
        captured.append(fig)
        return (Path(output_stem).with_suffix(".png"),)

    monkeypatch.setattr(local_analysis, "_save_figure", capture_figure)
    monkeypatch.setattr(local_analysis.plt, "close", lambda _fig: None)
    plot_cluster_map_gallery(
        mask_result,
        tmp_path / "maps" / "styled",
        ncols=2,
        show_roi_titles=False,
        show_title=False,
        show_legend=False,
        panel_border_color="purple",
        panel_border_width=1.25,
        row_spacing=0.23,
        column_spacing=0.19,
    )

    fig = captured[0]
    assert fig._suptitle is None
    assert fig.subplotpars.hspace == pytest.approx(0.23)
    assert fig.subplotpars.wspace == pytest.approx(0.19)
    assert all(not ax.get_title() for ax in fig.axes)
    borders = [patch for ax in fig.axes for patch in ax.patches]
    assert len(borders) == 2
    assert all(patch.get_linewidth() == pytest.approx(1.25) for patch in borders)
    original_close(fig)


def test_abundance_and_cell_composition_customization(tmp_path):
    representation = _representation()
    metadata = pd.DataFrame(
        {
            "ROI_number": ["roi_1", "roi_2"],
            "Case": ["case_a", "case_b"],
            "Group": ["primary", "recurrent"],
            "Score": [1.0, 2.0],
        }
    )
    abundance = summarize_environment_abundance(
        representation,
        "leiden_test",
        metadata=metadata,
    )
    third_roi = abundance.loc[abundance["roi"] == "roi_2"].copy()
    third_roi["roi"] = "roi_3"
    third_roi["ROI_number"] = "roi_3"
    third_roi["Case"] = "case_c"
    third_roi["Group"] = "primary"
    third_roi["Score"] = 3.0
    abundance = pd.concat([abundance, third_roi], ignore_index=True)
    for roi, counts in {
        "roi_1": {"0": 3, "1": 1},
        "roi_2": {"0": 2, "1": 2},
        "roi_3": {"0": 1, "1": 3},
    }.items():
        for environment, count in counts.items():
            selected = (abundance["roi"] == roi) & (
                abundance["environment"] == environment
            )
            abundance.loc[selected, ["patch_count", "total_patches", "fraction"]] = [
                count,
                4,
                count / 4,
            ]
    prepared = prepare_environment_abundance_tables(
        abundance,
        sample_group_col="Case",
        categorical_metadata=["Group"],
        numeric_metadata=["Score"],
        environment_order=["1", "0"],
        categorical_statistic="median",
        correlation_method="pearson",
    )
    assert list(prepared.roi_fraction.columns.astype(str)) == ["1", "0"]
    assert prepared.sample_group_fraction is not None
    assert set(prepared.categorical_fraction) == {"Group"}
    assert prepared.numeric_correlations is not None
    abundance_outputs = plot_environment_abundance(
        abundance,
        tmp_path / "abundance",
        categorical_metadata=["Group"],
        numeric_metadata=["Score"],
        environment_order=["1", "0"],
        environment_labels={"0": "Matrix", "1": "Vascular"},
        overall_sort="abundance_descending",
        overall_palette={"0": "#123456", "1": "#abcdef"},
        show_bar_values=True,
        show_overall_title=False,
        overall_axis_label_fontsize=8,
        overall_tick_fontsize=7,
        bar_value_fontsize=6,
        overall_axis_border_color="purple",
        overall_axis_border_width=1.1,
        plot_roi=False,
        categorical_statistic="median",
        correlation_method="pearson",
        min_numeric_observations=3,
        sample_heatmap_options={
            "show_fixed": False,
            "annot": True,
            "clustermap_kws": {
                "col_colors": pd.Series(
                    {"0": "#123456", "1": "#abcdef"}, name="environment"
                )
            },
        },
        categorical_heatmap_options={"show_clustermap": False},
        categorical_table_options={"Group": {"figsize": (5, 3), "title_fontsize": 9}},
        numeric_heatmap_options={"show_clustermap": False},
        formats=("png",),
    )
    assert "roi" not in abundance_outputs
    assert abundance_outputs["overall"][0].suffix == ".png"
    assert set(abundance_outputs["sample_group"]) == {"clustermap"}
    assert (
        tmp_path / "abundance" / "tables" / "numeric_metadata_pearson_correlations.csv"
    ).is_file()

    assignments = pd.DataFrame(
        {
            "environment": ["0", "0", "1", "1"],
            "population": ["A", "B", "A", "B"],
            "status": ["assigned"] * 4,
        }
    )
    composition = summarize_cell_environment_composition(assignments)
    cell_outputs = plot_cell_environment_composition(
        composition,
        tmp_path / "cells",
        metrics=("cell_count", "fraction_within_environment"),
        environment_order=["1", "0"],
        population_order=["B", "A"],
        environment_labels={"0": "Matrix", "1": "Vascular"},
        population_labels={"A": "Type A", "B": "Type B"},
        heatmap_options={
            "show_clustermap": False,
            "annot": True,
            "x_tick_rotation": 30,
        },
        metric_options={"cell_count": {"cmap": "Blues", "fmt": ".0f"}},
        formats=("png",),
    )
    assert set(cell_outputs) == {"cell_count", "fraction_within_environment"}
    assert all(set(output) == {"fixed"} for output in cell_outputs.values())
