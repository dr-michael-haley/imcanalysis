"""Cluster CellVision VICReg embeddings with the existing RAPIDS sequence."""

from __future__ import annotations

import logging
import os
from pathlib import Path


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


def main() -> None:
    import anndata as ad

    from SpatialBiologyToolkit.cellvision import configuration_fingerprint, leiden_key
    from SpatialBiologyToolkit.scripts._cellvision_common import load_runtime, reporter
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

    n_pcs = min(int(cellvision.n_pcs), embeddings.n_vars, embeddings.n_obs - 1)
    n_neighbors = min(int(cellvision.n_neighbors), embeddings.n_obs - 1)
    clustering_fingerprint = configuration_fingerprint(
        {
            "schema_version": 1,
            "identity_fingerprint": fingerprint,
            "training_fingerprint": str(metadata.get("training_fingerprint", "")),
            "n_pcs": n_pcs,
            "n_neighbors": n_neighbors,
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
            stage_reporter.add_asset("cellvision_clustered", paths.clustered, "RAPIDS UMAP and Leiden annotations for VICReg cells.")
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
        graph_key = _run_rapids_neighbors(
            embeddings,
            representation_key="X_cellvision_pca",
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            neighbors_key="cellvision_neighbors",
            neighbors_params={},
        )
        umap_key = _run_rapids_umap(
            embeddings,
            umap_min_dist=cellvision.umap_min_dist,
            neighbors_key=graph_key,
            umap_key="X_cellvision_umap",
            umap_params={"random_state": cellvision.seed},
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
            "umap_min_dist": float(cellvision.umap_min_dist),
            "umap_key": umap_key,
            "neighbors_key": graph_key,
            "leiden_keys": final_leiden,
            "seed": int(cellvision.seed),
        },
    }
    _atomic_write_h5ad(embeddings, paths.clustered)
    logging.info(
        "Saved CellVision RAPIDS UMAP and %d Leiden resolutions for %d cells.",
        len(final_leiden),
        embeddings.n_obs,
    )
    if stage_reporter is not None:
        stage_reporter.add_asset("cellvision_clustered", paths.clustered, "RAPIDS PCA, graph, UMAP, and Leiden annotations for VICReg cells.")
        stage_reporter.add_metric("clustered_cells", embeddings.n_obs)
        stage_reporter.add_metric("rapids_n_pcs", n_pcs)
        stage_reporter.add_metric("rapids_n_neighbors", n_neighbors)
        stage_reporter.add_metric("leiden_resolutions", len(final_leiden))


if __name__ == "__main__":
    main()
