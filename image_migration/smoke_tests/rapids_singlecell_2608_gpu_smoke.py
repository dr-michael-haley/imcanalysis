"""Subprocess-isolated GPU acceptance test for ``rapids_singlecell``.

The direct cuGraph and full RAPIDS-singlecell cases run in separate Python
processes so a native CUDA/UCXX failure is reported without losing the other
diagnostic output.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path


def _direct_cugraph_leiden() -> None:
    import cudf
    import cugraph
    import cupy as cp

    assert cp.cuda.runtime.getDeviceCount() >= 1
    edges = cudf.DataFrame(
        {
            "source": [0, 0, 1, 1, 2, 3, 3, 4],
            "destination": [1, 2, 2, 3, 3, 4, 5, 5],
        }
    )
    graph = cugraph.Graph(directed=False)
    graph.from_cudf_edgelist(
        edges,
        source="source",
        destination="destination",
        renumber=False,
    )
    partitions, modularity = cugraph.leiden(graph, resolution=0.5)
    assert len(partitions) == 6
    assert "partition" in partitions.columns
    print(
        "DIRECT_CUGRAPH_LEIDEN_PASS "
        f"vertices={len(partitions)} modularity={float(modularity):.6f}"
    )


def _rapids_singlecell_workflow() -> None:
    import anndata as ad
    import cupy as cp
    import numpy as np
    import rapids_singlecell as rsc

    assert cp.cuda.runtime.getDeviceCount() >= 1
    rng = np.random.default_rng(20260820)
    first = rng.normal(loc=-2.0, scale=0.35, size=(64, 10))
    second = rng.normal(loc=2.0, scale=0.35, size=(64, 10))
    adata = ad.AnnData(np.vstack([first, second]).astype(np.float32))

    rsc.get.anndata_to_GPU(adata)
    rsc.pp.pca(adata, n_comps=5, random_state=0)
    rsc.pp.neighbors(
        adata,
        n_neighbors=10,
        n_pcs=5,
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
    rsc.get.anndata_to_CPU(adata)

    assert adata.obsm["X_pca"].shape == (128, 5)
    assert adata.obsm["X_umap"].shape == (128, 2)
    assert "connectivities" in adata.obsp
    assert "gpu_smoke_leiden" in adata.obs
    cluster_count = int(adata.obs["gpu_smoke_leiden"].nunique())
    assert cluster_count >= 1
    print(
        "RAPIDS_SINGLECELL_WORKFLOW_PASS "
        f"cells={adata.n_obs} clusters={cluster_count}"
    )


CASES = {
    "direct-cugraph-leiden": _direct_cugraph_leiden,
    "rapids-singlecell-workflow": _rapids_singlecell_workflow,
}


def _run_all_cases() -> int:
    expected = {
        "rapids-singlecell-cu13": "0.16.1",
        "cudf": "26.08",
        "cuml": "26.08",
        "cugraph": "26.08",
        "cupy": "14.",
        "numpy": "2.",
        "pandas": "3.",
    }
    print("=== runtime ===")
    print(f"python={sys.version.split()[0]}")
    assert sys.version_info[:2] == (3, 14), sys.version
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    for package, prefix in expected.items():
        observed = version(package)
        assert observed == prefix or observed.startswith(prefix), (
            package,
            observed,
            prefix,
        )
        print(f"{package}={observed}")

    script = Path(__file__).resolve()
    failed: list[tuple[str, int]] = []
    for name in CASES:
        print(f"\n=== case: {name} ===", flush=True)
        result = subprocess.run(
            [sys.executable, str(script), "--case", name],
            check=False,
            text=True,
        )
        print(f"CASE_RESULT {name} {result.returncode}", flush=True)
        if result.returncode != 0:
            failed.append((name, result.returncode))

    if failed:
        print(f"GPU_SMOKE_FAILED {failed}", file=sys.stderr)
        return 1
    print("GPU_SMOKE_PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES))
    args = parser.parse_args()
    if args.case:
        CASES[args.case]()
        return 0
    return _run_all_cases()


if __name__ == "__main__":
    raise SystemExit(main())
