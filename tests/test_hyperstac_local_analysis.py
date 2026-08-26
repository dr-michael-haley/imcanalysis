from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import tifffile

from SpatialBiologyToolkit.hyperstac.local_analysis import (
    assign_cells_to_hyperstac_masks,
    hyperstac_cluster_feature_tables,
    plot_hyperstac_environment_gallery,
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
        index=["mean_intensity_norm_A", "mean_intensity_norm_B", "mean_intensity_norm_C"],
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
            tifffile.imwrite(roi_dir / f"{channel}.tiff", base + offset + channel_index * 0.05)


def test_reconstruct_cluster_label_masks_preserves_image_shape_and_zero_gaps(tmp_path):
    normalised = tmp_path / "normalised"
    _write_normalised_images(normalised)

    result = reconstruct_cluster_label_masks(
        _representation(),
        "leiden_test",
        tmp_path / "output",
        normalised_image_dir=normalised,
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
    assert result.manifest.loc[result.manifest["roi"] == "roi_1", "coverage_fraction"].item() == 0.5


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
    mask_result = reconstruct_cluster_label_masks(_representation(), "leiden_test", tmp_path)
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

    assert assignments["status"].tolist() == ["assigned", "assigned", "uncovered", "out_of_bounds"]
    assert assignments.loc["c0", "environment"] == "0"
    assert assignments.loc["c1", "environment"] == "1"
    composition = summarize_cell_environment_composition(assignments)
    assert composition["cell_count"].sum() == 2
    assert composition.groupby("environment")["fraction_within_environment"].sum().eq(1).all()
