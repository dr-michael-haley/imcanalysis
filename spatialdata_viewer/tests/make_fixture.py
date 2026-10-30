"""Generate tiny, explicitly synthetic stores for real browser/loader checks."""
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
import json
from uuid import uuid4

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Identity
from spatialdata._io.format import SpatialDataContainerFormatV01
import dask.array as da
import inspect

# Older local Dask versions predate ome-zarr's explicit create-array kwargs.
# Adapt that call only in this fixture process; installed packages stay intact.
array_kwargs_parameter = inspect.signature(da.to_zarr).parameters.get("zarr_array_kwargs")
if array_kwargs_parameter is None or array_kwargs_parameter.kind == inspect.Parameter.VAR_KEYWORD:
    original_to_zarr = da.to_zarr
    def fixture_to_zarr(*args, **kwargs):
        kwargs.update(kwargs.pop("zarr_array_kwargs", {}))
        if "compressor" in kwargs:
            kwargs["compressors"] = kwargs.pop("compressor")
        kwargs["zarr_format"] = 2
        return original_to_zarr(*args, **kwargs)
    da.to_zarr = fixture_to_zarr

root = Path(__file__).resolve().parents[1] / "fixtures"
root.mkdir(exist_ok=True)
destination = root / f"tissue-{uuid4().hex[:8]}.zarr"

y, x = np.mgrid[:256, :256]
centres = [(48, 52), (125, 62), (205, 54), (68, 155), (158, 178), (218, 211)]
labels = np.zeros((256, 256), dtype=np.uint32)
channels = np.zeros((3, 256, 256), dtype=np.float32)
for identity, (cx, cy) in enumerate(centres, 1):
    distance = (x - cx) ** 2 + (y - cy) ** 2
    labels[distance < 19**2] = identity
    channels[0] += np.exp(-distance / 110) * 220
    channels[1] += np.exp(-((np.sqrt(distance) - 16) ** 2) / 15) * (180 if identity % 2 else 40)
    channels[2] += np.exp(-distance / 300) * (190 if identity % 2 == 0 else 20)
channels += ((x + y) % 13)[None] * 0.4
images, masks = {}, {}
for suffix, system in [("A", "roi_A"), ("B", "roi_B")]:
    images[f"Tissue_{suffix}"] = Image2DModel.parse(channels, dims=("c", "y", "x"), c_coords=["DNA", "CD3", "CD68"], chunks=(1, 128, 128), scale_factors=[2], transformations={system: Identity()})
    masks[f"Cells_{suffix}"] = Labels2DModel.parse(labels, dims=("y", "x"), chunks=(128, 128), scale_factors=[2], transformations={system: Identity()})
ids = [4, 1, 6, 2, 5, 3]  # Intentionally not the order of the label ids.
obs = pd.DataFrame({
    "region": pd.Categorical([f"Cells_{suffix}" for suffix in ["A", "B"] for _ in ids]),
    "instance_id": ids * 2,
    "cell_type": pd.Categorical(["Macrophage" if i % 2 == 0 else "T cell" for i in ids] * 2),
    "score": [float(i) / 6 for i in ids] * 2,
}, index=[f"cell_{i}" for i in range(12)])
table = ad.AnnData(X=np.ones((12, 3), dtype=np.float32), obs=obs, var=pd.DataFrame(index=["DNA", "CD3", "CD68"]))
table = TableModel.parse(table, region=["Cells_A", "Cells_B"], region_key="region", instance_key="instance_id")
shapes = ShapesModel.parse(gpd.GeoDataFrame({"radius": [8.0] * len(centres)}, geometry=[Point(cx, cy) for cx, cy in centres]), transformations={"roi_A": Identity()})
points = PointsModel.parse(pd.DataFrame({"x": [30.0, 80.0, 150.0, 210.0], "y": [220.0, 220.0, 220.0, 220.0], "gene": pd.Categorical(["Cd3", "Cd3", "Cd68", "Cd68"])}), coordinates={"x": "x", "y": "y"}, feature_key="gene", transformations={"roi_A": Identity()})
sdata = SpatialData(images=images, labels=masks, tables={"cells": table}, shapes={"Spots_A": shapes}, points={"Transcripts_A": points})
sdata.write(destination, sdata_formats=SpatialDataContainerFormatV01())
with ZipFile(destination.with_suffix(".zarr.zip"), "w", compression=ZIP_STORED) as archive:
    for path in destination.rglob("*"):
        if path.is_file():
            archive.write(path, path.relative_to(root).as_posix())
(root / "invalid").mkdir(exist_ok=True)
(root / "invalid" / "readme.txt").write_text("This is not SpatialData.")
(root / "fixture.json").write_text(json.dumps({"folder": destination.name, "zip": destination.name + ".zip"}))
print(f"Synthetic fixture: {destination}")
