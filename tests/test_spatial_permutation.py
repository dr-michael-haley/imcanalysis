from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

from SpatialBiologyToolkit.spatial_permutation import (
    resolve_spatial_mask_paths,
    spatial_mask_alignment_qc,
    spatial_permutation_zscores,
)


def _example_mask() -> np.ndarray:
    return np.array(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 2],
            [0, 2, 2, 2],
            [0, 2, 2, 2],
        ],
        dtype=np.uint8,
    )


def _example_obs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ROI": ["roi_1"] * 5,
            "X_loc": [1.2, 2.8, 1.1, 2.4, 3.8],
            "Y_loc": [0.1, 0.9, 2.2, 3.1, 3.9],
            "population": ["A", "A", "A", "B", "B"],
        }
    )


def test_resolve_spatial_mask_paths_uses_exact_names(tmp_path):
    tifffile.imwrite(tmp_path / "roi_1_pixel_mask.tiff", _example_mask())
    tifffile.imwrite(tmp_path / "roi_10_pixel_mask.tiff", _example_mask())

    paths = resolve_spatial_mask_paths(tmp_path, ["roi_1", "roi_10"])

    assert paths["roi_1"].name == "roi_1_pixel_mask.tiff"
    assert paths["roi_10"].name == "roi_10_pixel_mask.tiff"


def test_resolve_spatial_mask_paths_reports_all_missing_masks(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing 2 of 2 expected ROI masks"):
        resolve_spatial_mask_paths(tmp_path, ["missing_1", "missing_2"])


def test_spatial_mask_alignment_qc_reports_coordinate_and_tissue_losses(tmp_path):
    tifffile.imwrite(tmp_path / "roi_1_pixel_mask.tiff", _example_mask())
    obs = pd.DataFrame(
        {
            "ROI": ["roi_1"] * 4,
            "X_loc": [1.0, 0.0, 8.0, np.nan],
            "Y_loc": [0.0, 0.0, 1.0, 2.0],
            "population": ["A", "A", "B", None],
        }
    )

    qc = spatial_mask_alignment_qc(obs, tmp_path)

    row = qc.iloc[0]
    assert row["n_cells"] == 4
    assert row["n_missing_population"] == 1
    assert row["n_nonfinite_coordinates"] == 1
    assert row["n_out_of_bounds"] == 1
    assert row["n_excluded_by_tissue"] == 1
    assert row["n_in_tissue"] == 1
    assert row["environment_values"] == "1;2"


def test_spatial_permutation_zscores_match_hypergeometric_marginals(tmp_path):
    tifffile.imwrite(tmp_path / "roi_1_pixel_mask.tiff", _example_mask())

    result = spatial_permutation_zscores(
        _example_obs(),
        tmp_path,
        n_permutations=20_000,
        tissue_exclude_values=(0,),
        random_state=42,
        pixel_dictionary={1: "one", 2: "two"},
    )
    indexed = result.set_index(["pixel", "population"])

    assert indexed.loc[(1, "A"), "observed"] == 2
    assert indexed.loc[(2, "A"), "observed"] == 1
    assert indexed.loc[(1, "B"), "observed"] == 0
    assert indexed.loc[(2, "B"), "observed"] == 2
    assert indexed.loc[(1, "A"), "perm_mean"] == pytest.approx(3 * 5 / 12, abs=0.03)
    assert indexed.loc[(1, "B"), "perm_mean"] == pytest.approx(2 * 5 / 12, abs=0.03)
    expected_variance = 3 * (5 / 12) * (7 / 12) * ((12 - 3) / 11)
    assert indexed.loc[(1, "A"), "perm_std"] == pytest.approx(
        np.sqrt(expected_variance), abs=0.03
    )
    assert set(result["pixel_mapped"]) == {"one", "two"}
    assert set(result["n_analysed_cells"]) == {5}


def test_spatial_permutation_is_reproducible_across_worker_counts(tmp_path):
    for roi in ("roi_1", "roi_10"):
        tifffile.imwrite(tmp_path / f"{roi}_pixel_mask.tiff", _example_mask())
    obs = pd.concat(
        [
            _example_obs(),
            _example_obs().assign(ROI="roi_10"),
        ],
        ignore_index=True,
    )

    serial = spatial_permutation_zscores(
        obs, tmp_path, n_permutations=100, random_state=7, n_jobs=1
    )
    parallel = spatial_permutation_zscores(
        obs, tmp_path, n_permutations=100, random_state=7, n_jobs=2
    )

    pd.testing.assert_frame_equal(serial, parallel)
