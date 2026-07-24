"""Build morphology/intensity CellVision graphs and cluster them with RAPIDS."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable


def _atomic_write_h5ad(adata, path: Path) -> None:
    from SpatialBiologyToolkit.scripts.config_and_utils import (
        _sanitize_anndata_uns_inplace,
    )

    removed = _sanitize_anndata_uns_inplace(adata)
    if removed:
        logging.info(
            "Removed %d null-like AnnData.uns entries before writing the "
            "cross-environment CellVision asset.",
            removed,
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    adata.write_h5ad(temporary)
    os.replace(temporary, path)


def _canonicalize_leiden_columns(
    adata,
    *,
    resolutions,
    generated_keys,
) -> list[str]:
    """Move RAPIDS Leiden outputs into stable, collision-safe CellVision keys."""
    if len(resolutions) != len(generated_keys):
        raise RuntimeError(
            "RAPIDS returned an unexpected number of CellVision Leiden keys: "
            f"{len(generated_keys)} for {len(resolutions)} resolutions."
        )

    from SpatialBiologyToolkit.cellvision import leiden_key

    final_keys: list[str] = []
    for resolution, generated_key in zip(resolutions, generated_keys, strict=True):
        if generated_key not in adata.obs.columns:
            raise RuntimeError(
                "RAPIDS did not create the reported CellVision Leiden key: "
                f"{generated_key!r}."
            )
        final_key = leiden_key(resolution)
        if generated_key != final_key:
            if final_key in adata.obs.columns:
                raise RuntimeError(
                    "Cannot rename a RAPIDS CellVision Leiden result because the "
                    f"destination obs key already exists: {final_key!r}."
                )
            adata.obs.rename(columns={generated_key: final_key}, inplace=True)
        adata.obs[final_key] = adata.obs[final_key].astype("category")
        final_keys.append(final_key)
    return final_keys


def _register_fused_neighbors(
    adata: Any,
    *,
    morphology_neighbors_key: str,
    intensity_neighbors_key: str,
    joint_neighbors_key: str,
    intensity_weight: float,
    n_neighbors: int,
    representation_key: str,
    n_pcs: int | None,
    random_state: int,
    to_backend: Callable[[Any], Any] | None = None,
) -> str:
    """Fuse named neighbour graphs and register a Scanpy-compatible graph key."""
    from SpatialBiologyToolkit.cellvision import fuse_connectivity_graphs

    def connectivities(graph_key: str):
        metadata = adata.uns.get(graph_key)
        if not isinstance(metadata, dict):
            raise KeyError(f"CellVision neighbor metadata {graph_key!r} is missing.")
        matrix_key = metadata.get("connectivities_key")
        if not matrix_key or matrix_key not in adata.obsp:
            raise KeyError(
                f"CellVision neighbor metadata {graph_key!r} does not reference "
                "an available connectivity matrix."
            )
        return adata.obsp[matrix_key]

    fused = fuse_connectivity_graphs(
        connectivities(morphology_neighbors_key),
        connectivities(intensity_neighbors_key),
        intensity_weight=intensity_weight,
    )
    connectivity_key = f"{joint_neighbors_key}_connectivities"
    distance_key = f"{joint_neighbors_key}_distances"
    adata.obsp[connectivity_key] = to_backend(fused) if to_backend is not None else fused
    adata.uns[joint_neighbors_key] = {
        "connectivities_key": connectivity_key,
        # Graph fusion produces an affinity graph, not a meaningful metric
        # distance graph.  Scanpy's NeighborsView nevertheless requires the
        # metadata key to exist for named neighbour graphs.  Leaving the
        # corresponding obsp entry absent accurately advertises that distances
        # are unavailable while allowing connectivity consumers such as UMAP
        # and Leiden to use the fused graph.
        "distances_key": distance_key,
        "params": {
            "method": "cellvision_degree_normalized_graph_fusion",
            "morphology_neighbors_key": morphology_neighbors_key,
            "intensity_neighbors_key": intensity_neighbors_key,
            "morphology_weight": float(1.0 - intensity_weight),
            "intensity_weight": float(intensity_weight),
            "n_neighbors": int(n_neighbors),
            "use_rep": representation_key,
            "n_pcs": n_pcs,
            "metric": "euclidean",
            "random_state": int(random_state),
        },
    }
    return joint_neighbors_key


def _cellvision_umap_params(seed: int) -> dict[str, Any]:
    """Return the explicit random UMAP initialization contract."""
    return {"init_pos": "random", "random_state": int(seed)}


def main() -> None:
    import anndata as ad
    import rapids_singlecell as rsc

    from SpatialBiologyToolkit.cellvision import (
        aligned_obsm_representation,
        configuration_fingerprint,
        input_file_manifest,
        leiden_key,
    )
    from SpatialBiologyToolkit.scripts._cellvision_common import (
        fusion_intensity_path,
        load_runtime,
        reporter,
    )
    from SpatialBiologyToolkit.scripts.basic_process_rapids import (
        _ensure_cpu_storage,
        _move_input_matrix_to_cpu,
        _move_input_matrix_to_gpu,
        _run_rapids_leiden,
        _run_rapids_neighbors,
        _run_rapids_pca,
        _run_rapids_umap,
    )

    config, paths = load_runtime("cluster")
    cellvision = config.cellvision
    if not paths.embeddings.is_file():
        raise FileNotFoundError(
            f"CellVision embeddings do not exist: {paths.embeddings}. "
            "Run the cellvision-embed stage first."
        )
    embeddings = ad.read_h5ad(paths.embeddings)
    metadata = embeddings.uns.get("cellvision", {})
    fingerprint = str(metadata.get("identity_fingerprint", ""))
    if not fingerprint:
        raise ValueError("CellVision embeddings lack an identity fingerprint.")
    if embeddings.n_obs < 3:
        raise ValueError("RAPIDS CellVision clustering requires at least three embedded cells.")
    if not embeddings.obs_names.is_unique:
        raise ValueError("CellVision embedding observation IDs must be unique.")

    intensity_source_path: Path | None = None
    intensity_representation: Any | None = None
    intensity_manifest: list[dict[str, Any]] = []
    if cellvision.fusion_enabled:
        intensity_source_path = fusion_intensity_path(config)
        if not intensity_source_path.is_file():
            raise FileNotFoundError(
                "CellVision fusion intensity AnnData does not exist: "
                f"{intensity_source_path}. Run BioBatchNet first, configure "
                "cellvision.fusion_intensity_adata_path, or set "
                "cellvision.fusion_enabled=false."
            )
        source = ad.read_h5ad(intensity_source_path, backed="r")
        try:
            intensity_representation = aligned_obsm_representation(
                source,
                embeddings.obs_names,
                cellvision.fusion_intensity_representation,
            )
        finally:
            source.file.close()
        embeddings.obsm["X_cellvision_intensity"] = intensity_representation
        intensity_manifest = input_file_manifest([intensity_source_path])

    n_pcs = min(int(cellvision.n_pcs), embeddings.n_vars, embeddings.n_obs - 1)
    n_neighbors = min(int(cellvision.n_neighbors), embeddings.n_obs - 1)
    clustering_fingerprint = configuration_fingerprint(
        {
            "schema_version": 2,
            "identity_fingerprint": fingerprint,
            "training_fingerprint": str(metadata.get("training_fingerprint", "")),
            "n_pcs": n_pcs,
            "n_neighbors": n_neighbors,
            "fusion_enabled": bool(cellvision.fusion_enabled),
            "fusion_intensity_source": intensity_manifest,
            "fusion_intensity_representation": (
                cellvision.fusion_intensity_representation
                if cellvision.fusion_enabled
                else None
            ),
            "fusion_intensity_weight": (
                float(cellvision.fusion_intensity_weight)
                if cellvision.fusion_enabled
                else None
            ),
            "umap_init_pos": "random",
            "umap_min_dist": float(cellvision.umap_min_dist),
            "leiden_resolutions": [
                float(value) for value in cellvision.leiden_resolutions
            ],
            "seed": int(cellvision.seed),
        }
    )

    stage_reporter = reporter()
    if stage_reporter is not None:
        stage_reporter.add_input("cellvision_embeddings", paths.embeddings, "Cell-level VICReg embeddings used for RAPIDS clustering.")
        if intensity_source_path is not None:
            stage_reporter.add_input(
                "cellvision_fusion_intensity_anndata",
                intensity_source_path,
                "Source AnnData providing the identity-aligned batch-corrected intensity embedding.",
            )

    if paths.clustered.exists() and not cellvision.overwrite:
        clustered = ad.read_h5ad(paths.clustered, backed="r")
        existing_metadata = clustered.uns.get("cellvision", {})
        observed = str(existing_metadata.get("identity_fingerprint", ""))
        observed_clustering = str(
            existing_metadata.get("clustering", {}).get("configuration_fingerprint", "")
        )
        expected_leiden = [leiden_key(value) for value in cellvision.leiden_resolutions]
        leiden_columns = [
            column for column in expected_leiden if column in clustered.obs.columns
        ]
        n_cells = int(clustered.n_obs)
        clustered.file.close()
        if (
            observed != fingerprint
            or observed_clustering != clustering_fingerprint
            or len(leiden_columns) != len(expected_leiden)
        ):
            raise ValueError(
                "Existing CellVision clustered AnnData is incompatible or incomplete. "
                "Set cellvision.overwrite=true to rebuild it."
            )
        logging.info("Reusing validated CellVision clustered AnnData at %s", paths.clustered)
        if stage_reporter is not None:
            stage_reporter.add_asset("cellvision_clustered", paths.clustered, "RAPIDS UMAP and Leiden annotations for CellVision cells.")
            stage_reporter.add_metric("clustered_cells", n_cells)
            stage_reporter.add_metric("leiden_resolutions", len(leiden_columns))
            stage_reporter.add_note("Reused an existing clustered AnnData with a matching identity fingerprint.")
        return

    if n_pcs < cellvision.n_pcs:
        logging.warning("Capping cellvision.n_pcs from %d to %d for data shape %s.", cellvision.n_pcs, n_pcs, embeddings.shape)
    if n_neighbors < cellvision.n_neighbors:
        logging.warning("Capping cellvision.n_neighbors from %d to %d for %d cells.", cellvision.n_neighbors, n_neighbors, embeddings.n_obs)

    gpu_layer = _move_input_matrix_to_gpu(embeddings, {})
    try:
        _run_rapids_pca(
            embeddings,
            n_pcs=n_pcs,
            pca_key="X_cellvision_pca",
            pca_params={},
        )
        if cellvision.fusion_enabled:
            morphology_graph_key = _run_rapids_neighbors(
                embeddings,
                representation_key="X_cellvision_pca",
                n_neighbors=n_neighbors,
                n_pcs=n_pcs,
                neighbors_key="cellvision_morphology_neighbors",
                neighbors_params={},
            )
            embeddings.obsm["X_cellvision_intensity"] = rsc.get.X_to_GPU(
                embeddings.obsm["X_cellvision_intensity"],
                warning="X_cellvision_intensity",
            )
            intensity_graph_key = _run_rapids_neighbors(
                embeddings,
                representation_key="X_cellvision_intensity",
                n_neighbors=n_neighbors,
                n_pcs=None,
                neighbors_key="cellvision_intensity_neighbors",
                neighbors_params={},
            )
            graph_key = _register_fused_neighbors(
                embeddings,
                morphology_neighbors_key=morphology_graph_key,
                intensity_neighbors_key=intensity_graph_key,
                joint_neighbors_key="cellvision_neighbors",
                intensity_weight=cellvision.fusion_intensity_weight,
                n_neighbors=n_neighbors,
                representation_key="X_cellvision_pca",
                n_pcs=n_pcs,
                random_state=cellvision.seed,
                to_backend=lambda value: rsc.get.X_to_GPU(
                    value, warning="cellvision_neighbors_connectivities"
                ),
            )
        else:
            morphology_graph_key = _run_rapids_neighbors(
                embeddings,
                representation_key="X_cellvision_pca",
                n_neighbors=n_neighbors,
                n_pcs=n_pcs,
                neighbors_key="cellvision_neighbors",
                neighbors_params={},
            )
            intensity_graph_key = None
            graph_key = morphology_graph_key
        umap_key = _run_rapids_umap(
            embeddings,
            umap_min_dist=cellvision.umap_min_dist,
            neighbors_key=graph_key,
            umap_key="X_cellvision_umap",
            umap_params=_cellvision_umap_params(cellvision.seed),
        )
        generated_leiden = _run_rapids_leiden(
            embeddings,
            resolutions=list(cellvision.leiden_resolutions),
            enabled=True,
            neighbors_key=graph_key,
            leiden_params={"random_state": cellvision.seed},
            key_prefix="cellvision_leiden",
        )
        final_leiden = _canonicalize_leiden_columns(
            embeddings,
            resolutions=cellvision.leiden_resolutions,
            generated_keys=generated_leiden,
        )
    finally:
        _move_input_matrix_to_cpu(embeddings, gpu_layer)
        _ensure_cpu_storage(embeddings)

    embeddings.uns["cellvision"] = {
        **metadata,
        "identity_fingerprint": fingerprint,
        "clustering": {
            "method": "rapids_singlecell",
            "configuration_fingerprint": clustering_fingerprint,
            "n_pcs": n_pcs,
            "n_neighbors": n_neighbors,
            "fusion_enabled": bool(cellvision.fusion_enabled),
            "fusion_intensity_adata_path": (
                str(intensity_source_path) if intensity_source_path is not None else None
            ),
            "fusion_intensity_representation": (
                cellvision.fusion_intensity_representation
                if cellvision.fusion_enabled
                else None
            ),
            "fusion_intensity_weight": (
                float(cellvision.fusion_intensity_weight)
                if cellvision.fusion_enabled
                else None
            ),
            "morphology_neighbors_key": morphology_graph_key,
            "intensity_neighbors_key": intensity_graph_key,
            "umap_min_dist": float(cellvision.umap_min_dist),
            "umap_init_pos": "random",
            "umap_key": umap_key,
            "neighbors_key": graph_key,
            "leiden_keys": final_leiden,
            "seed": int(cellvision.seed),
        },
    }
    _atomic_write_h5ad(embeddings, paths.clustered)
    logging.info(
        "Saved CellVision RAPIDS joint UMAP and %d Leiden resolutions for %d cells.",
        len(final_leiden),
        embeddings.n_obs,
    )
    if stage_reporter is not None:
        stage_reporter.add_asset("cellvision_clustered", paths.clustered, "RAPIDS PCA, modality graphs, joint UMAP, and Leiden annotations for CellVision cells.")
        stage_reporter.add_metric("clustered_cells", embeddings.n_obs)
        stage_reporter.add_metric("rapids_n_pcs", n_pcs)
        stage_reporter.add_metric("rapids_n_neighbors", n_neighbors)
        stage_reporter.add_metric("leiden_resolutions", len(final_leiden))
        stage_reporter.add_metric("fusion_enabled", bool(cellvision.fusion_enabled))
        if cellvision.fusion_enabled:
            stage_reporter.add_metric(
                "fusion_intensity_weight", float(cellvision.fusion_intensity_weight)
            )
            stage_reporter.add_note(
                "CellVision clusters use the degree-normalized fusion of morphology "
                "and BioBatchNet intensity neighbor graphs."
            )


if __name__ == "__main__":
    main()
