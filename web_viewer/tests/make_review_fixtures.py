"""Small synthetic fixtures for flags, ROI metadata and marker ranking."""
from pathlib import Path
import shutil
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

root = Path(__file__).resolve().parents[1] / "fixtures"
target = root.parent / "test-results" / "review-inputs"
target.mkdir(parents=True, exist_ok=True)
source = ad.read_h5ad(root / "dense.h5ad")
obs = source.obs.copy()
obs["donor"] = pd.Categorical(["Donor A", "Donor B"] * 16)
obs["batch"] = 1
markers = ["CD3", "CD20", "DNA", "CD4", "CD8", "FoxP3", "Ki67", "PanCK"]
x = np.full((32, 8), 2, dtype=np.float32)
x[:, 2] = 5
x[np.arange(32) % 4 == 1, 1] = 20
for column in [0, 3, 4, 5, 6, 7]:
    x[np.arange(32) % 4 == 0, column] = 20
x[:, 6] -= 2  # Exercise implicit sparse zeros.
x[0, 3] = np.nan
data = ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=markers))
data.obsm["X_umap"] = source.obsm["X_umap"]
data.layers["counts"] = sparse.csr_matrix(x * 2)
for name, matrix in [("dense", x), ("csr", sparse.csr_matrix(x)), ("csc", sparse.csc_matrix(x))]:
    data.X = matrix
    data.write_h5ad(target / f"review_{name}.h5ad", compression="gzip")
shutil.copytree(root / "masks", target / "masks", dirs_exist_ok=True)
for roi, prefix in [("ROI_A", "44_90"), ("ROI_A_2", "55_20")]:
    folder = target / "images" / roi
    folder.mkdir(parents=True, exist_ok=True)
    for marker in ["CD3", "CD20", "CD4", "CD8", "FoxP3", "Ki67", "PanCK"]:
        shutil.copyfile(root / "images" / roi / "209Bi_CD20.tiff", folder / f"{prefix}_Ir193_{marker}.tiff")
    shutil.copyfile(root / "images" / roi / "DNA.tiff", folder / f"{prefix}_Ir193_DNA3.tiff")
print(target)
