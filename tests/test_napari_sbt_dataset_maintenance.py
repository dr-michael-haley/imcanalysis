from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from tifffile import imread, imwrite

from SpatialBiologyToolkit.napari_sbt.dataset_maintenance import (
    CellFilterRequest,
    apply_cell_filter,
    apply_var_rename,
    copy_renamed_images,
    dataset_readiness,
    normalise_var_rename_mapping,
    plan_image_renames,
    preview_cell_filter,
    preview_mask_rebuild,
    rebuild_masks_and_object_numbers,
    remove_anndata_vars,
)


def _adata() -> AnnData:
    obs = pd.DataFrame(
        {
            "ROI": pd.Categorical(["A", "A", "B", "B"]),
            "ObjectNumber": [2, 7, 3, 9],
            "population": pd.Categorical(["T", "B", "T", "Myeloid"]),
            "score": [0.1, 0.8, 0.4, np.nan],
        },
        index=["cell_1", "cell_2", "cell_3", "cell_4"],
    )
    var = pd.DataFrame(
        {"channel_label": ["CD3", "CD20", "CD68"]},
        index=["CD3", "CD20", "CD68"],
    )
    return AnnData(np.arange(12, dtype=float).reshape(4, 3), obs=obs, var=var)


def test_variable_rename_and_removal_preserve_anndata_alignment():
    adata = _adata()
    mapping = normalise_var_rename_mapping(adata, {"CD3": "CD3e"})

    renamed = apply_var_rename(adata, mapping)
    reduced = remove_anndata_vars(renamed, ["CD20"])

    assert adata.var_names.tolist() == ["CD3", "CD20", "CD68"]
    assert renamed.var_names.tolist() == ["CD3e", "CD20", "CD68"]
    assert renamed.var["channel_label"].tolist() == ["CD3e", "CD20", "CD68"]
    assert reduced.var_names.tolist() == ["CD3e", "CD68"]
    assert reduced.X.shape == (4, 2)


def test_variable_rename_rejects_collisions():
    with pytest.raises(ValueError, match="duplicate"):
        normalise_var_rename_mapping(_adata(), {"CD3": "CD20"})


def test_cell_filter_previews_and_slices_selected_values():
    adata = _adata()
    request = CellFilterRequest(
        observation="population",
        mode="keep_values",
        values=["T"],
    )

    preview = preview_cell_filter(adata, request, roi_obs="ROI")
    filtered = apply_cell_filter(adata, request)

    assert preview.ready
    assert preview.details["retained_cells"] == 2
    assert preview.details["represented_rois"] == 2
    assert filtered.obs_names.tolist() == ["cell_1", "cell_3"]


def test_readiness_treats_unconfigured_assets_as_optional():
    checks = dataset_readiness(
        _adata(),
        roi_obs="ROI",
        object_obs="ObjectNumber",
        mask_paths={},
        image_index={},
        expect_masks=False,
        expect_images=False,
    )

    levels = {check.key: check.level for check in checks}
    assert levels["mask_index"] == "optional"
    assert levels["image_index"] == "optional"


def test_image_rename_plan_copies_complete_collection(tmp_path):
    source_root = tmp_path / "images"
    roi_folder = source_root / "A"
    roi_folder.mkdir(parents=True)
    cd3 = roi_folder / "191Ir_CD3.tiff"
    cd20 = roi_folder / "193Ir_CD20.tiff"
    cd3.write_bytes(b"cd3")
    cd20.write_bytes(b"cd20")
    output = tmp_path / "derived_images"
    plan = plan_image_renames(
        {"A": {"CD3": cd3, "CD20": cd20}},
        {"CD3": "CD3e"},
        image_roots=[source_root],
        output_root=output,
    )

    written = copy_renamed_images(plan)

    assert plan.ready
    assert len(plan.items) == 2
    assert (written / "01_images" / "A" / "191Ir_CD3e.tiff").read_bytes() == b"cd3"
    assert (written / "01_images" / "A" / "193Ir_CD20.tiff").read_bytes() == b"cd20"
    assert (written / "image_rename_crosswalk.csv").is_file()


def test_mask_rebuild_compacts_ids_and_updates_anndata(tmp_path):
    adata = _adata()
    masks = tmp_path / "masks"
    masks.mkdir()
    mask_a = np.array([[0, 2, 2], [0, 7, 12]], dtype=np.uint16)
    mask_b = np.array([[3, 3, 0], [9, 0, 20]], dtype=np.uint16)
    imwrite(masks / "A.tiff", mask_a)
    imwrite(masks / "B.tiff", mask_b)
    mask_paths = {"A": masks / "A.tiff", "B": masks / "B.tiff"}

    preview = preview_mask_rebuild(
        adata,
        mask_paths,
        roi_obs="ROI",
        object_obs="ObjectNumber",
        mode="compact",
    )
    updated, crosswalk, output = rebuild_masks_and_object_numbers(
        adata,
        mask_paths,
        tmp_path / "derived_masks",
        roi_obs="ROI",
        object_obs="ObjectNumber",
        mode="compact",
    )

    assert preview.ready
    assert preview.details["extra_mask_labels"] == 2
    assert updated.obs["ObjectNumber"].tolist() == [1, 2, 1, 2]
    assert set(np.unique(imread(output / "A.tiff"))) == {0, 1, 2}
    assert set(np.unique(imread(output / "B.tiff"))) == {0, 1, 2}
    assert crosswalk.columns.tolist() == [
        "obs_name",
        "ROI",
        "ObjectNumber_before",
        "ObjectNumber_after",
    ]
    assert (output / "object_number_crosswalk.csv").is_file()
    assert (output / "maintenance_manifest.json").is_file()
