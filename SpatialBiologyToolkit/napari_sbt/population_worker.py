"""Monitored workers for population-curation operations.

The default Scanpy worker rebuilds neighbours only within the selected cells
from an existing corrected representation such as ``X_biobatchnet``. It never
reruns normalization, scaling, PCA, batch correction, or UMAP. Reusing an
already-computed connectivity graph remains available as an explicit option.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from SpatialBiologyToolkit.pipeline.manifests import utc_now, write_json

from .population_curation import (
    GraphSubclusterRequest,
    normalise_source_label,
    ordered_source_labels,
    source_obs_fingerprint,
)
from .storage import write_dataframe


def _emit(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"event": event, **payload},
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )


def _cluster_adjacency(
    *,
    obs_names: pd.Index,
    adjacency: Any,
    resolution: float,
    random_state: int,
) -> pd.Series:
    """Run Leiden against the supplied induced graph without preprocessing."""

    import anndata as ad
    import scanpy as sc

    if len(obs_names) < 2:
        raise ValueError("At least two cells are required for graph subclustering.")
    if adjacency.shape != (len(obs_names), len(obs_names)):
        raise ValueError(
            f"Induced adjacency shape {adjacency.shape} does not match "
            f"{len(obs_names)} selected cells."
        )
    edge_count = (
        int(adjacency.nnz)
        if hasattr(adjacency, "nnz")
        else int(np.count_nonzero(adjacency))
    )
    if edge_count == 0:
        raise ValueError(
            "The selected cells have no edges in this connectivity graph. "
            "Choose another graph or a broader source population."
        )
    miniature = ad.AnnData(
        X=np.zeros((len(obs_names), 1), dtype=np.float32),
        obs=pd.DataFrame(index=obs_names.astype(str)),
    )
    sc.tl.leiden(
        miniature,
        adjacency=adjacency,
        resolution=float(resolution),
        random_state=int(random_state),
        key_added="_napari_sbt_subcluster",
    )
    result = miniature.obs["_napari_sbt_subcluster"].astype(str).copy()
    result.index = obs_names.astype(str)
    return result


def _cluster_corrected_representation(
    *,
    obs_names: pd.Index,
    representation: Any,
    representation_key: str,
    n_neighbors: int,
    resolution: float,
    random_state: int,
) -> tuple[pd.Series, int, int]:
    """Rebuild a subset graph from a precomputed corrected representation."""

    import anndata as ad
    import scanpy as sc
    from scipy import sparse

    if len(obs_names) < 3:
        raise ValueError(
            "At least three cells are required to rebuild a useful neighbour graph."
        )
    if sparse.issparse(representation):
        representation = representation.toarray()
    representation = np.asarray(representation)
    if representation.ndim != 2 or representation.shape[0] != len(obs_names):
        raise ValueError(
            f"Subset representation {representation_key!r} has shape "
            f"{representation.shape}; expected ({len(obs_names)}, n_features)."
        )
    if representation.shape[1] < 1:
        raise ValueError(
            f"Representation {representation_key!r} contains no dimensions."
        )
    if not bool(np.isfinite(representation).all()):
        raise ValueError(
            f"Representation {representation_key!r} contains NaN or infinite values."
        )
    effective_n_neighbors = min(int(n_neighbors), len(obs_names) - 1)
    miniature = ad.AnnData(
        X=np.zeros((len(obs_names), 1), dtype=np.float32),
        obs=pd.DataFrame(index=obs_names.astype(str)),
    )
    worker_rep_key = "_napari_sbt_corrected_rep"
    miniature.obsm[worker_rep_key] = representation
    sc.pp.neighbors(
        miniature,
        n_neighbors=effective_n_neighbors,
        use_rep=worker_rep_key,
        random_state=int(random_state),
    )
    connectivities = miniature.obsp["connectivities"]
    sc.tl.leiden(
        miniature,
        adjacency=connectivities,
        resolution=float(resolution),
        random_state=int(random_state),
        key_added="_napari_sbt_subcluster",
    )
    result = miniature.obs["_napari_sbt_subcluster"].astype(str).copy()
    result.index = obs_names.astype(str)
    return result, effective_n_neighbors, int(connectivities.nnz)


def run_graph_subclustering(request: GraphSubclusterRequest) -> Path:
    """Subcluster labels from a corrected representation or existing graph."""

    import anndata as ad
    from scipy import sparse

    started = time.monotonic()
    source_path = Path(request.anndata_path).expanduser()
    output = Path(request.output_folder).expanduser().resolve(strict=False)
    if not source_path.is_file():
        raise FileNotFoundError(f"AnnData source not found: {source_path}")
    output.mkdir(parents=True, exist_ok=True)
    _emit(
        "population_subcluster_loading",
        run_id=request.run_id,
        anndata_path=str(source_path),
    )
    adata = ad.read_h5ad(source_path)
    if request.source_obs not in adata.obs:
        raise KeyError(
            f"Source observation {request.source_obs!r} is not in AnnData."
        )
    fingerprint = source_obs_fingerprint(adata, request.source_obs)
    if fingerprint != request.source_fingerprint:
        raise ValueError(
            "The worker AnnData no longer matches the population workspace "
            "source fingerprint."
        )
    graph = None
    representation = None
    if request.neighbor_source == "existing_graph":
        if request.adjacency_key not in adata.obsp:
            raise KeyError(
                f"Connectivity graph {request.adjacency_key!r} is absent from "
                "adata.obsp."
            )
        graph = adata.obsp[request.adjacency_key]
        if graph.shape != (adata.n_obs, adata.n_obs):
            raise ValueError(
                f"adata.obsp[{request.adjacency_key!r}] has shape {graph.shape}; "
                f"expected {(adata.n_obs, adata.n_obs)}."
            )
        if not sparse.issparse(graph):
            graph = sparse.csr_matrix(graph)
        else:
            graph = graph.tocsr()
    else:
        if request.representation_key not in adata.obsm:
            raise KeyError(
                f"Corrected representation {request.representation_key!r} is "
                "absent from adata.obsm."
            )
        representation = adata.obsm[request.representation_key]
        if representation.shape[0] != adata.n_obs:
            raise ValueError(
                f"adata.obsm[{request.representation_key!r}] has "
                f"{representation.shape[0]} rows; expected {adata.n_obs}."
            )

    ordered_source_labels(adata.obs[request.source_obs])
    source_values = (
        adata.obs[request.source_obs]
        .astype(object)
        .map(normalise_source_label)
        .astype("string")
    )
    available = set(source_values.dropna().astype(str))
    missing = sorted(set(request.selected_values) - available)
    if missing:
        raise ValueError(f"Selected source populations do not exist: {missing}")

    assignments: list[pd.DataFrame] = []
    if request.mode == "within_each":
        tasks = [
            (
                value,
                source_values.eq(value).fillna(False).to_numpy(dtype=bool),
            )
            for value in request.selected_values
        ]
    else:
        tasks = [
            (
                "selected_populations_together",
                source_values.isin(request.selected_values).to_numpy(dtype=bool),
            )
        ]
    for index, (task_name, selector) in enumerate(tasks, start=1):
        positions = np.flatnonzero(selector)
        _emit(
            "population_subcluster_running",
            run_id=request.run_id,
            task_index=index,
            task_count=len(tasks),
            population=task_name,
            cell_count=len(positions),
            resolution=request.resolution,
            neighbor_source=request.neighbor_source,
            representation_key=request.representation_key,
            n_neighbors=request.n_neighbors,
            adjacency_key=request.adjacency_key,
        )
        subset_names = pd.Index(adata.obs_names[positions].astype(str))
        if request.neighbor_source == "rebuild_from_rep":
            clusters, effective_n_neighbors, edge_count = (
                _cluster_corrected_representation(
                    obs_names=subset_names,
                    representation=representation[positions],
                    representation_key=str(request.representation_key),
                    n_neighbors=request.n_neighbors,
                    resolution=request.resolution,
                    random_state=request.random_state,
                )
            )
            method = "scanpy_rebuilt_neighbors"
        else:
            subset_graph = graph[positions][:, positions]
            clusters = _cluster_adjacency(
                obs_names=subset_names,
                adjacency=subset_graph,
                resolution=request.resolution,
                random_state=request.random_state,
            )
            effective_n_neighbors = None
            edge_count = int(subset_graph.nnz)
            method = "scanpy_existing_graph"
        parent = source_values.iloc[positions].astype(str).to_numpy()
        if request.mode == "within_each":
            labels = np.asarray(
                [f"{task_name} · {cluster}" for cluster in clusters],
                dtype=object,
            )
        else:
            labels = np.asarray(
                [f"Subcluster {cluster}" for cluster in clusters],
                dtype=object,
            )
        assignments.append(
            pd.DataFrame(
                {
                    "obs_name": subset_names,
                    "source_value": parent,
                    "component_value": labels,
                    "leiden_cluster": clusters.to_numpy(),
                    "method": method,
                    "run_id": request.run_id,
                    "resolution": float(request.resolution),
                    "neighbor_source": request.neighbor_source,
                    "representation_key": request.representation_key,
                    "n_neighbors_requested": request.n_neighbors,
                    "n_neighbors_effective": effective_n_neighbors,
                    "adjacency_key": request.adjacency_key,
                    "connectivity_edges": edge_count,
                }
            )
        )

    result = pd.concat(assignments, ignore_index=True)
    assignments_path = output / "assignments.csv"
    write_dataframe(assignments_path, result)
    provenance = {
        "schema_version": 1,
        "run_id": request.run_id,
        "created_at": utc_now().isoformat(),
        "anndata_path": str(source_path.resolve(strict=False)),
        "anndata_size_bytes": source_path.stat().st_size,
        "anndata_mtime_ns": source_path.stat().st_mtime_ns,
        "source_obs": request.source_obs,
        "source_fingerprint": request.source_fingerprint,
        "selected_values": request.selected_values,
        "resolution": request.resolution,
        "random_state": request.random_state,
        "mode": request.mode,
        "neighbor_source": request.neighbor_source,
        "representation_key": request.representation_key,
        "n_neighbors": request.n_neighbors,
        "effective_n_neighbors": sorted(
            {
                int(value)
                for value in result["n_neighbors_effective"].dropna().tolist()
            }
        ),
        "adjacency_key": request.adjacency_key,
        "cell_count": len(result),
        "cluster_count": int(result["component_value"].nunique()),
        "preprocessing": {
            "normalization": False,
            "scaling": False,
            "pca": False,
            "batch_correction": False,
            "neighbors": request.neighbor_source == "rebuild_from_rep",
            "umap": False,
            "graph_source": (
                f"rebuilt within selected cells from "
                f"adata.obsm[{request.representation_key!r}] with "
                f"n_neighbors={request.n_neighbors}"
                if request.neighbor_source == "rebuild_from_rep"
                else f"adata.obsp[{request.adjacency_key!r}]"
            ),
        },
        "assignments_path": str(assignments_path),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(output / "provenance.json", provenance)
    _emit(
        "population_subcluster_completed",
        **provenance,
    )
    return assignments_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NapariSBT population-curation worker"
    )
    parser.add_argument("--request", required=True, help="Graph request JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        request = GraphSubclusterRequest.model_validate_json(
            Path(args.request).read_text(encoding="utf-8")
        )
        run_graph_subclustering(request)
    except Exception as exc:  # noqa: BLE001 - process boundary
        _emit(
            "population_subcluster_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
