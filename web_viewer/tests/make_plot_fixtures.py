"""Small plotting edge cases; contains no user data."""
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

root = Path(__file__).resolve().parents[1] / "fixtures"
matrix = np.array([[0, 0], [np.nan, 0], [6, 0], [9, np.nan], [3, 0], [2, 0]], dtype=np.float32)
obs = pd.DataFrame({"ROI": ["one", "one", "two", "one", "two", "two"], "population": pd.Categorical(["A", "A", "A", "B", "B", "C"], categories=["C", "B", "A", "unused"]), "condition": ["x", "x", "y", "x", "y", "y"]}, index=[str(i) for i in range(6)])
for name, values in [("dense", matrix), ("csr", sparse.csr_matrix(matrix)), ("csc", sparse.csc_matrix(matrix))]:
    data = ad.AnnData(X=values, obs=obs, var=pd.DataFrame(index=["signal", "zero"]))
    data.write_h5ad(root / f"plot_{name}.h5ad", compression="gzip")

# Compact, fill-valued HDF5 for an interactive cancellation check. No real cells
# and no physical 200 MB matrix allocation are needed to exercise a long scan.
import h5py
with h5py.File(root / "plot_cancel.h5ad", "w") as handle:
    obs_group = handle.create_group("obs")
    obs_group.attrs["_index"] = "_index"
    obs_group.create_dataset("_index", shape=(500_000,), dtype="S16", fillvalue=b"synthetic")
    population = obs_group.create_group("population")
    population.attrs["encoding-type"] = "categorical"
    population.create_dataset("categories", data=np.array(["A"], dtype=h5py.string_dtype()))
    population.create_dataset("codes", shape=(500_000,), dtype="i4", fillvalue=0)
    var_group = handle.create_group("var")
    var_group.attrs["_index"] = "_index"
    var_group.create_dataset("_index", data=np.array([f"M{i}" for i in range(100)], dtype=h5py.string_dtype()))
    handle.create_dataset("X", shape=(500_000, 100), dtype="f4", chunks=(1024, 100), fillvalue=1)
