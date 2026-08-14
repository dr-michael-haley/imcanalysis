from __future__ import annotations

from pathlib import Path

import pandas as pd

from SpatialBiologyToolkit.qc_classifier.io import (
    build_image_channel_aliases,
    discover_roi_image_index,
    discover_roi_images,
    resolve_mask_file,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_multiple_image_folders_merge_and_match_anndata_panel(tmp_path: Path):
    primary = tmp_path / "Images"
    aligned = tmp_path / "matrix_images_alligned"
    primary_cd3 = _touch(primary / "r1" / "sample_001_191Ir_CD3e.tiff")
    primary_cd20 = _touch(primary / "r1" / "CD20.tiff")
    aligned_cd3 = _touch(aligned / "r1" / "CD3e.tiff")
    composite = _touch(aligned / "r1" / "matrix_composite.png")
    overview = _touch(aligned / "r1_overview.png")
    direct_roi = _touch(aligned / "r1.png")
    var = pd.DataFrame(
        {
            "channel_name": ["191Ir", "209Bi"],
            "channel_label": ["CD3e", "CD20"],
        },
        index=["CD3", "CD20"],
    )

    aliases = build_image_channel_aliases(var.index, var)
    discovered = discover_roi_images(
        [primary, aligned],
        "r1",
        channel_aliases=aliases,
    )

    assert discovered == {
        "CD3": primary_cd3,
        "CD20": primary_cd20,
        "CD3 [matrix_images_alligned]": aligned_cd3,
        "matrix_composite": composite,
        "matrix_images_alligned": direct_roi,
        "overview": overview,
    }


def test_ambiguous_panel_alias_is_not_assigned_to_the_wrong_variable():
    var = pd.DataFrame(
        {"channel_label": ["shared", "shared"]},
        index=["marker_a", "marker_b"],
    )

    aliases = build_image_channel_aliases(var.index, var)

    assert "shared" not in aliases
    assert aliases["marker_a"] == "marker_a"
    assert aliases["marker_b"] == "marker_b"


def test_integrity_index_scans_nested_and_flat_images_once(tmp_path: Path):
    images = tmp_path / "images"
    nested = _touch(images / "ROI_A" / "CD3.tiff")
    flat = _touch(images / "ROI_B_CD20.tiff")

    index = discover_roi_image_index(images, ["ROI_A", "ROI_B"])

    assert index["ROI_A"] == {"CD3": nested}
    assert index["ROI_B"] == {"CD20": flat}


def test_fast_mask_resolution_does_not_require_directory_discovery(tmp_path: Path):
    expected = _touch(tmp_path / "ROI_A.tiff")

    assert resolve_mask_file(tmp_path, "ROI_A") == expected
    assert resolve_mask_file(tmp_path, "missing") is None
