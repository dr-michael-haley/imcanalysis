"""Interchange and failure-safety checks using small synthetic IMC data."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tifffile
import zarr
from scipy import sparse
from shapely import wkb

from SpatialBiologyToolkit.odon_export.__main__ import main
from SpatialBiologyToolkit.odon_export.exporter import (
    _cell_geometries,
    _downsample,
    _encode_image,
    export_dataset,
)


@pytest.fixture
def dataset(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    masks.mkdir()
    # Intentionally interleave ROI rows; object IDs are only unique within an ROI.
    obs = pd.DataFrame(
        {
            "ROI": ["r1", "r2", "r1", "r2"],
            "ObjectNumber": [10, 10, 30, 30],
            "population": pd.Categorical(["T", "B", "B", "T"]),
            "CD3": [100, 101, 102, 103],
            "id": ["old1", "old2", "old3", "old4"],
        },
        index=["cell1", "cell2", "cell3", "cell4"],
    )
    values = np.array([[1.25, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    adata = ad.AnnData(
        sparse.csr_matrix(values), obs=obs, var=pd.DataFrame(index=["CD3", "CD20"])
    )
    adata.layers["counts"] = sparse.csr_matrix(values * 10)
    adata.raw = adata.copy()
    source = tmp_path / "cells.h5ad"
    adata.write_h5ad(source)
    mask = np.zeros((7, 9), np.uint32)
    mask[:3, :3] = 10  # border cell
    mask[4:7, 5:9] = 30
    mask[3, 4] = 99  # cell absent from AnnData: do not fabricate measurements
    for roi in ("r1", "r2"):
        folder = images / roi
        folder.mkdir(parents=True)
        tifffile.imwrite(masks / f"{roi}.tiff", mask)
        tifffile.imwrite(
            folder / "141Pr_CD3.tiff", np.arange(63, dtype=np.uint16).reshape(7, 9)
        )
        tifffile.imwrite(
            folder / "CD20.tiff", np.arange(63, dtype=np.uint16).reshape(7, 9) + 100
        )
    return source, masks, images, tmp_path / "odon"


def test_portable_export_preserves_measurements_identities_and_pixels(dataset):
    source, masks, images, output = dataset
    original = source.read_bytes()
    result = export_dataset(*dataset, levels=3, chunk_size=16)
    assert result["cell_count"] == 4
    assert source.read_bytes() == original
    sheet = pd.read_csv(output / "samplesheet.csv")
    assert sheet["id"].tolist() == ["r1", "r2"]
    for row in sheet.itertuples():
        assert not Path(row.path).is_absolute()
        group = zarr.open_group(str(output / row.path), mode="r")
        assert (output / row.path / ".zgroup").exists()
        assert group["0"].dtype == np.uint16
        assert group["0"].chunks == (1, 7, 9)
        assert group["2"].shape == (2, 2, 3)
        channels = [c["label"] for c in group.attrs["omero"]["channels"]]
        cd3 = channels.index("CD3")
        np.testing.assert_array_equal(group["0"][cd3], np.arange(63).reshape(7, 9))
        assert group.attrs["multiscales"][0]["datasets"][2][
            "coordinateTransformations"
        ][0]["scale"] == [1, 4, 4]
        table = pq.read_table(output / row.segpath)
        assert pa.types.is_string(table.schema.field("population").type)
        assert (
            json.loads(table.schema.metadata[b"geo"])["columns"]["geometry"]["crs"]
            is None
        )
        cells = table.to_pandas()
        assert cells["ObjectNumber"].tolist() == [10, 30]
        assert cells["id"].tolist() == [
            json.dumps([row.id, 10]),
            json.dumps([row.id, 30]),
        ]
        expected = [1.25, 5] if row.id == "r1" else [3, 7]
        np.testing.assert_array_equal(cells["CD3"], expected)
        assert "obs:CD3" in cells and "obs:id" in cells
        shapes = [wkb.loads(value) for value in cells["geometry"]]
        assert shapes[0].bounds == (0, 0, 3, 3)
        assert shapes[0].area == 9
        assert cells["x_centroid"].iloc[0] == 1.5
        assert cells["y_centroid"].iloc[0] == 1.5


def test_geometry_preserves_holes_disconnected_parts_and_sparse_ids():
    mask = np.zeros((10, 11), dtype=np.int64)
    mask[1:6, 1:6] = 2**40
    mask[2:5, 2:5] = 0
    mask[8, 9] = 2**40
    geometry = _cell_geometries(mask, np.array([2**40]))[2**40]
    assert geometry.geom_type == "MultiPolygon"
    assert geometry.area == 17
    assert sum(len(poly.interiors) for poly in geometry.geoms) == 1
    assert geometry.is_valid


def test_float_encoding_is_reversible_within_quantization_bound():
    image = np.array([[-2.5, 0.1], [30.0, 100.25]], dtype=np.float32)
    stored, metadata = _encode_image(image, "auto")
    recovered = stored * metadata["scale"] + metadata["offset"]
    np.testing.assert_allclose(recovered, image, atol=metadata["scale"] / 2 + 1e-6)
    with pytest.raises(ValueError, match="losslessly"):
        _encode_image(image, "preserve")
    with pytest.raises(ValueError, match="NaN"):
        _encode_image(np.array([[np.nan]]), "auto")
    stored, metadata = _encode_image(np.full((2, 2), 0.5), "auto")
    np.testing.assert_array_equal(stored * metadata["scale"] + metadata["offset"], 0.5)


def test_odd_pyramid_edges_do_not_include_zero_padding():
    np.testing.assert_array_equal(
        _downsample(np.full((3, 5), 200, np.uint16)), np.full((2, 3), 200)
    )


def test_dry_run_and_roi_channel_subset(dataset):
    result = export_dataset(*dataset, rois=["r2"], channels=["CD3"], dry_run=True)
    assert result["roi_count"] == 1
    assert result["rois"] == [{"roi": "r2", "cells": 2, "channels": ["CD3"]}]
    assert not dataset[3].exists()
    export_dataset(
        *dataset, rois=["r2"], channels=["CD3"], layer="counts", obs_columns=[]
    )
    cells = pq.read_table(dataset[3] / "rois/roi_000001/cells.parquet").to_pandas()
    assert "population" not in cells
    np.testing.assert_array_equal(cells["CD3"], [30, 70])


def test_raw_expression_and_stack_input(dataset):
    source, masks, images, output = dataset
    stack_folder = source.parent / "stacks"
    stack_folder.mkdir()
    stack = np.stack([np.full((7, 9), 7, np.uint16), np.full((7, 9), 19, np.uint16)])
    tifffile.imwrite(stack_folder / "r1.tiff", stack, metadata={"axes": "CYX"})
    with pytest.raises(ValueError, match="stack_channels"):
        export_dataset(source, masks, stack_folder, output, rois=["r1"])
    export_dataset(
        source,
        masks,
        stack_folder,
        output,
        rois=["r1"],
        stack_channels=["CD3", "CD20"],
        use_raw=True,
    )
    group = zarr.open_group(str(output / "rois/roi_000001/image.ome.zarr"))
    np.testing.assert_array_equal(group["0"][:], stack)


@pytest.mark.parametrize("encoding", ["dense", "csr", "csc"])
def test_backed_matrix_formats_with_zeroes_and_interleaved_rows(dataset, encoding):
    source, masks, images, output = dataset
    adata = ad.read_h5ad(source)
    values = np.array([[0, 9], [13, 0], [0, 0], [2**40, 2]], dtype=np.int64)
    adata.X = {
        "dense": lambda x: x,
        "csr": sparse.csr_matrix,
        "csc": sparse.csc_matrix,
    }[encoding](values)
    adata.write_h5ad(source)
    export_dataset(*dataset)
    for index, rows in [(1, [0, 2]), (2, [1, 3])]:
        table = pq.read_table(
            output / f"rois/roi_{index:06d}/cells.parquet"
        ).to_pandas()
        np.testing.assert_array_equal(table[["CD3", "CD20"]].to_numpy(), values[rows])


def test_raw_uses_its_own_variables(dataset):
    source, masks, images, output = dataset
    adata = ad.read_h5ad(source)
    adata.raw = ad.AnnData(
        np.arange(12).reshape(4, 3),
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=["raw_a", "raw_b", "raw_c"]),
    )
    adata.write_h5ad(source)
    export_dataset(*dataset, use_raw=True, rois=["r1"], obs_columns=[])
    table = pq.read_table(output / "rois/roi_000001/cells.parquet").to_pandas()
    np.testing.assert_array_equal(
        table[["raw_a", "raw_b", "raw_c"]], [[0, 1, 2], [6, 7, 8]]
    )


def test_float_image_export_records_encoding_and_preserve_failure(dataset):
    source, masks, images, output = dataset
    image = np.linspace(-3.25, 17.5, 63, dtype=np.float32).reshape(7, 9)
    tifffile.imwrite(images / "r1/CD20.tiff", image)
    with pytest.raises(ValueError, match="losslessly"):
        export_dataset(*dataset, rois=["r1"], intensity_mode="preserve")
    assert not output.exists()
    export_dataset(*dataset, rois=["r1"])
    group = zarr.open_group(str(output / "rois/roi_000001/image.ome.zarr"))
    names = [c["label"] for c in group.attrs["omero"]["channels"]]
    encoding = group.attrs["sbt_export"]["channels"][names.index("CD20")]
    restored = group["0"][names.index("CD20")] * encoding["scale"] + encoding["offset"]
    np.testing.assert_allclose(restored, image, atol=encoding["scale"] / 2 + 1e-6)


def test_unselected_layers_are_not_loaded(dataset, monkeypatch):
    import SpatialBiologyToolkit.odon_export.exporter as exporter

    original = exporter.read_elem
    visited = []

    def tracked(node):
        visited.append(node.name)
        return original(node)

    monkeypatch.setattr(exporter, "read_elem", tracked)
    export_dataset(*dataset, dry_run=True)
    assert visited == ["/obs", "/var"]


def test_failed_second_roi_does_not_publish_partial_dataset(dataset):
    source, masks, images, output = dataset
    tifffile.imwrite(masks / "r2.tiff", np.zeros((7, 9), np.uint32))
    with pytest.raises(ValueError, match="missing from mask"):
        export_dataset(*dataset)
    assert not output.exists()
    assert not list(output.parent.glob(".sbt-odon-*"))


@pytest.mark.parametrize(
    "case", ["duplicate", "fractional", "null", "negative", "traversal"]
)
def test_invalid_cell_identities_fail_before_output(dataset, case):
    source, _, _, output = dataset
    adata = ad.read_h5ad(source)
    if case == "duplicate":
        adata.obs.loc["cell3", "ObjectNumber"] = 10
    elif case == "traversal":
        adata.obs["ROI"] = ["../escape"] * 4
    else:
        adata.obs["ObjectNumber"] = adata.obs["ObjectNumber"].astype(float)
        adata.obs.loc["cell1", "ObjectNumber"] = {
            "fractional": 1.5,
            "null": np.nan,
            "negative": -1,
        }[case]
    adata.write_h5ad(source)
    with pytest.raises(ValueError):
        export_dataset(*dataset)
    assert not output.exists()


def test_missing_or_misaligned_inputs_and_existing_output(dataset):
    source, masks, images, output = dataset
    with pytest.raises(ValueError, match="absent from AnnData"):
        export_dataset(*dataset, rois=["absent"])
    with pytest.raises(ValueError, match="missing image channels"):
        export_dataset(*dataset, channels=["missing"])
    tifffile.imwrite(images / "r1/CD20.tiff", np.zeros((3, 3), np.uint16))
    with pytest.raises(ValueError, match="differ"):
        export_dataset(*dataset)
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("unchanged")
    with pytest.raises(FileExistsError):
        export_dataset(*dataset)
    assert sentinel.read_text() == "unchanged"


def test_cli_project_defaults_and_explicit_paths(dataset, capsys):
    source, masks, images, output = dataset
    import yaml

    config = {
        "general": {
            "anndata_path": source.name,
            "masks_folder": masks.name,
            "denoised_images_folder": images.name,
        }
    }
    (source.parent / "config.yaml").write_text(yaml.safe_dump(config))
    assert (
        main(["--project", str(source.parent), "--output", str(output), "--dry-run"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["roi_count"] == 2
    assert not output.exists()
    assert (
        main(
            [
                "--anndata",
                str(source),
                "--masks",
                str(masks),
                "--images",
                str(images),
                "--output",
                str(output),
                "--roi",
                "r1",
            ]
        )
        == 0
    )
    assert (output / "samplesheet.csv").exists()
    with pytest.raises(SystemExit) as error:
        main(["--output", str(output)])
    assert error.value.code == 2
