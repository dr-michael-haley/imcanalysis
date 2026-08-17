"""GPU acceptance test for the consolidated ``sbt-analysis`` environment."""

from __future__ import annotations

from importlib.metadata import version

import anndata as ad
import cupy as cp
import cudf
import cugraph
import cuml
import dask
import dask_cudf
import distributed
import numpy as np
import rapids_singlecell as rsc
import rmm
import torch


def main() -> None:
    """Exercise the CUDA stack and the complete RAPIDS clustering path."""

    expected = {
        "rapids-singlecell": "0.12.0",
        "dask": "2024.11.2",
        "distributed": "2024.11.2",
        "dask-expr": "1.1.19",
    }
    for package, expected_version in expected.items():
        observed = version(package)
        assert observed == expected_version, (package, observed, expected_version)

    for package in ("cudf", "cugraph", "cuml", "dask-cudf", "rmm"):
        observed = version(package)
        assert observed.startswith("24.12"), (package, observed)

    assert torch.cuda.is_available(), "Torch cannot see a CUDA device"
    device_name = torch.cuda.get_device_name(0)
    torch_result = torch.arange(12, device="cuda", dtype=torch.float32).sum().item()
    assert torch_result == 66.0

    cupy_result = float(cp.arange(12, dtype=cp.float32).sum().get())
    assert cupy_result == 66.0
    cudf_result = int(cudf.Series([1, 2, 3]).sum())
    assert cudf_result == 6

    rng = np.random.default_rng(20260817)
    first = rng.normal(loc=-2.0, scale=0.35, size=(48, 8))
    second = rng.normal(loc=2.0, scale=0.35, size=(48, 8))
    adata = ad.AnnData(np.vstack([first, second]).astype(np.float32))
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.pca(adata, n_comps=4, random_state=0)
    rsc.pp.neighbors(
        adata,
        n_neighbors=10,
        n_pcs=4,
        use_rep="X_pca",
        random_state=0,
    )
    rsc.tl.umap(adata, min_dist=0.1, random_state=0)
    rsc.tl.leiden(
        adata,
        resolution=0.5,
        key_added="gpu_smoke_leiden",
        random_state=0,
    )

    assert adata.obsm["X_pca"].shape == (96, 4)
    assert adata.obsm["X_umap"].shape == (96, 2)
    assert "connectivities" in adata.obsp
    assert "gpu_smoke_leiden" in adata.obs
    cluster_count = int(adata.obs["gpu_smoke_leiden"].nunique())
    assert cluster_count >= 1

    print(f"GPU: {device_name}")
    print(f"Torch PASS: {torch.__version__} ({torch.version.cuda})")
    print(f"CuPy PASS: {cp.__version__}")
    print(f"cuDF/cuGraph/cuML PASS: {cudf.__version__} / {cugraph.__version__} / {cuml.__version__}")
    print(f"Dask/dask-cuDF PASS: {dask.__version__} / {dask_cudf.__version__}")
    print(f"Distributed/RMM PASS: {distributed.__version__} / {rmm.__version__}")
    print(f"RAPIDS-singlecell clustering PASS: {rsc.__version__} ({cluster_count} clusters)")
    print("GPU_SMOKE_PASS")


if __name__ == "__main__":
    main()
