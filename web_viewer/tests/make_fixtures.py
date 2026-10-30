"""Create tiny synthetic AnnData/TIFF inputs; never reads a user dataset.

Run with the SBT scientific Python environment before the JavaScript tests.
"""

from pathlib import Path
import json

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
import tifffile
from SpatialBiologyToolkit._napari_imc_normalization import normalize_imc_image


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "fixtures"
    root.mkdir(exist_ok=True)
    obs = pd.DataFrame(
        {
            "ROI": pd.Categorical(["ROI_A", "ROI_A_2"] * 16),
            "ObjectNumber": np.repeat(np.arange(1, 17) * 300, 2),
            "population": pd.Categorical((["T cell", "B cell", "Stroma", None] * 8), categories=["T cell", "B cell", "Stroma"]),
            "quality": np.linspace(0, 1, 32),
            "nullable": pd.array([1, None] * 16, dtype="Int64"),
        },
        index=[f"cell_{i}" for i in range(32)],
    )
    var = pd.DataFrame({"channel_name": ["191Ir", "209Bi", "193Ir"], "channel_label": ["CD3e", "CD20", "DNA"]}, index=["CD3", "CD20", "DNA"])
    matrix = np.arange(96, dtype=np.float32).reshape(32, 3)
    matrix[matrix % 3 == 0] = 0
    data = ad.AnnData(X=matrix, obs=obs, var=var)
    data.uns["population_colors"] = ["#ff6666", "#70f0b6", "#779cff"]
    data.obsm["X_umap"] = np.column_stack([np.sin(np.arange(32)), np.cos(np.arange(32))]).astype(np.float32)
    data.layers["counts"] = sparse.csr_matrix(matrix * 2)
    for name, values, compression in [("dense", matrix, "gzip"), ("csr", sparse.csr_matrix(matrix), "gzip"), ("csc", sparse.csc_matrix(matrix), "gzip"), ("lzf", matrix, "lzf")]:
        data.X = values
        data.write_h5ad(root / f"{name}.h5ad", compression=compression)
    # Exercise modern nullable string encoding independently of pandas defaults.
    with h5py.File(root / "dense.h5ad", "a") as handle:
        group = handle["obs"].create_group("nullable_text")
        group.attrs["encoding-type"] = "nullable-string-array"
        group.attrs["encoding-version"] = "0.1.0"
        group.create_dataset("values", data=np.array(["ok", ""] * 16, dtype=h5py.string_dtype()))
        group.create_dataset("mask", data=np.array([False, True] * 16))
        plain = handle["obs"].create_dataset("plain_text", data=np.array(["first", "second", "second", "first"] * 8, dtype=h5py.string_dtype()))
        plain.attrs["encoding-type"] = "string-array"
        plain.attrs["encoding-version"] = "0.2.0"
    data.obs.loc["cell_2", "ObjectNumber"] = 300
    data.write_h5ad(root / "duplicate.h5ad")
    masks = root / "masks"
    images = root / "images"
    masks.mkdir(exist_ok=True)
    for roi in ["ROI_A", "ROI_A_2"]:
        folder = images / roi
        folder.mkdir(parents=True, exist_ok=True)
        mask = np.zeros((256, 256), dtype=np.uint32)
        channels = [np.zeros((256, 256), dtype=np.float32) for _ in range(3)]
        yy, xx = np.mgrid[:256, :256]
        for i in range(16):
            cy, cx = 32 + (i // 4) * 64, 32 + (i % 4) * 64
            distance = (yy - cy) ** 2 + (xx - cx) ** 2
            mask[distance < 21**2] = (i + 1) * 300
            for c in range(3):
                channels[c] += (np.exp(-distance / (2 * (8 + c * 4) ** 2)) * (70 + i * 5) * (1 if i % 3 == c else 0.25)).astype(np.float32)
        mask[0:3, 0:3] = 65537
        tifffile.imwrite(masks / f"{roi}.tiff", mask, compression="deflate")
        for name, channel in zip(["sample_001_191Ir_CD3e", "209Bi_CD20", "DNA"], channels, strict=True):
            tifffile.imwrite(folder / f"{name}.tiff", channel, compression="deflate")
        six_channels = root / "six_channels" / roi
        six_channels.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            stripe = np.zeros((256, 256), dtype=np.uint16)
            stripe[:, 32 + i * 32:64 + i * 32] = 100
            tifffile.imwrite(six_channels / f"Ch{i + 1}.tiff", stripe, compression="deflate")
    tifffile.imwrite(root / "stack.tiff", np.stack(channels), photometric="minisblack")
    tifffile.imwrite(root / "wrong_shape.tiff", np.zeros((16, 16), dtype=np.uint16))
    bad_images = root / "bad_images" / "ROI_A"
    bad_images.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(bad_images / "CD3.tiff", np.zeros((16, 16), dtype=np.uint16))
    stacks = root / "stacks" / "ROI_A"
    stacks.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(stacks / "stack.tiff", np.stack(channels), photometric="minisblack")
    (root / "normalization.csv").write_text("marker,vmax,lower_threshold\nCh1,180,20\nCh2,100,100\n", encoding="utf-8")
    (root / "normalization_valid.csv").write_text("marker,vmax,lower_threshold\nCh1,180,20\n", encoding="utf-8")
    (root / "normalization.json").write_text(json.dumps({"normalization_dict": {"Ch1": {"vmax": 180, "lower_threshold": 20}}}), encoding="utf-8")
    # Numerical reference comes from the existing NapariSBT helper, not a JS reimplementation.
    reference = []
    for vmax, lower, quantile, minimum in [(10, 0, 0.999, 0.1), (10, 2, 0.999, 0.1), (None, 0, 0.83, 0.1), (None, 0, 0.1, 0.1), (10, 0, 1, 4)]:
        values = np.array([0, 0.05, 0.2, 1, 2, 5, 10, 20], dtype=np.float32)
        expected = normalize_imc_image(values, quantile=quantile, minimum_pixel_counts=minimum, normalization_value=vmax, normalization_lower_threshold=lower)
        reference.append({"values": values.tolist(), "expected": expected.tolist(), "options": {"bounds": None if vmax is None else {"vmax": vmax, "lower_threshold": lower}, "quantile": quantile, "minimum": minimum}})
    (root / "normalization_reference.json").write_text(json.dumps(reference), encoding="utf-8")
    print(f"Synthetic fixtures: {root}")


if __name__ == "__main__":
    main()
