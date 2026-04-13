"""
GPU processing stage using rapids-singlecell.

Runs PCA, optional Harmony batch correction, neighbors, UMAP, and Leiden
clustering on the post-segmentation AnnData object.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import anndata as ad
import matplotlib
import numpy as np
import scanpy as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import rapids_singlecell as rsc
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "rapids-singlecell is required for this script. Activate the "
        "'rapids_singlecell' environment before running basic_process_rapids."
    ) from exc

try:
    from SpatialBiologyToolkit import plotting as sbt_plotting
except Exception:  # pragma: no cover - optional plotting helper
    try:
        from .. import plotting as sbt_plotting  # type: ignore
    except Exception:
        sbt_plotting = None  # type: ignore

from .config_and_utils import (
    GeneralConfig,
    RapidsProcessConfig,
    VisualizationConfig,
    cleanstring,
    filter_config_for_dataclass,
    load_pipeline_anndata,
    process_config_with_overrides,
    save_pipeline_anndata,
    setup_logging,
)


def _resolve_n_pcs(adata: ad.AnnData, requested: Optional[int]) -> int:
    """Choose a PCA dimensionality compatible with the current AnnData shape."""
    if adata.n_obs < 2:
        raise ValueError("RAPIDS processing requires at least 2 cells in the AnnData object.")
    if adata.n_vars < 1:
        raise ValueError("RAPIDS processing requires at least 1 marker in the AnnData object.")

    default_n_pcs = adata.n_vars - 1 if adata.n_vars > 1 else 1
    target = int(requested if requested is not None else default_n_pcs)
    max_allowed = max(1, min(int(adata.n_vars), int(adata.n_obs - 1)))
    clipped = max(1, min(target, max_allowed))
    if clipped != target:
        logging.warning(
            "Requested n_for_pca=%s is incompatible with data shape %s. Using %s PCs instead.",
            target,
            adata.shape,
            clipped,
        )
    return clipped


def _resolve_neighbor_n_pcs(
    adata: ad.AnnData,
    *,
    representation_key: str,
    requested: Optional[int],
    default: int,
) -> int:
    """Choose a neighbor-graph PC count compatible with the active representation."""
    if representation_key not in adata.obsm:
        raise KeyError(f"Representation '{representation_key}' was not found in adata.obsm")

    rep = adata.obsm[representation_key]
    if len(rep.shape) != 2:
        raise ValueError(
            f"Representation '{representation_key}' must be 2-dimensional; got shape {rep.shape}."
        )
    max_allowed = int(rep.shape[1])
    target = int(requested if requested is not None else default)
    clipped = max(1, min(target, max_allowed))
    if clipped != target:
        logging.warning(
            "Requested n_pcs_neighbors=%s exceeds representation '%s' with %s columns. "
            "Using %s PCs for neighbors.",
            target,
            representation_key,
            max_allowed,
            clipped,
        )
    return clipped


def _clean_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Drop null-like values from config dictionaries before dispatch."""
    if not params:
        return {}
    cleaned: Dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
            continue
        cleaned[key] = value
    return cleaned


def _drop_managed_params(
    params: Dict[str, Any],
    *,
    managed: Iterable[str],
    section_name: str,
) -> Dict[str, Any]:
    """Remove pass-through parameters that are controlled by first-class config fields."""
    cleaned = dict(params)
    for key in managed:
        if key in cleaned:
            logging.warning(
                "Ignoring rapids.%s_params.%s because it is controlled by a dedicated "
                "RapidsProcessConfig field.",
                section_name,
                key,
            )
            cleaned.pop(key, None)
    return cleaned


def _normalise_dtype_param(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert YAML-friendly dtype strings into dtype objects for RAPIDS Harmony."""
    if "dtype" not in params or not isinstance(params["dtype"], str):
        return params

    value = params["dtype"].strip().lower()
    dtype_map = {
        "float32": np.float32,
        "np.float32": np.float32,
        "numpy.float32": np.float32,
        "cp.float32": np.float32,
        "cupy.float32": np.float32,
        "float64": np.float64,
        "np.float64": np.float64,
        "numpy.float64": np.float64,
        "cp.float64": np.float64,
        "cupy.float64": np.float64,
    }
    if value not in dtype_map:
        raise ValueError(
            "rapids.harmony_params.dtype must be one of: float32, float64, "
            "np.float32, np.float64, cp.float32, cp.float64."
        )
    params = dict(params)
    params["dtype"] = dtype_map[value]
    return params


def _normalise_optional_key(value: Optional[str]) -> Optional[str]:
    """Normalize config keys where None/empty means use RAPIDS defaults."""
    if value is None:
        return None
    cleaned = str(value).strip()
    if cleaned.lower() in {"", "none", "null"}:
        return None
    return cleaned


def _as_resolution_list(values: Any) -> List[float]:
    """Normalize scalar/list-like resolution settings."""
    if values is None:
        return []
    if isinstance(values, list):
        return [float(x) for x in values]
    return [float(values)]


def _copy_array(value: Any) -> Any:
    """Copy NumPy/CuPy array-like values while preserving their backend."""
    if hasattr(value, "copy"):
        return value.copy()
    return value


def _to_cpu(value: Any) -> Any:
    """Convert CuPy/cupyx values to CPU equivalents when possible."""
    module = type(value).__module__
    if module.startswith(("cupy", "cupyx")) and hasattr(value, "get"):
        return value.get()
    return value


def _ensure_cpu_storage(adata: ad.AnnData) -> None:
    """Move GPU-backed AnnData side arrays to CPU before plotting and H5AD writing."""
    adata.X = _to_cpu(adata.X)

    for collection in (adata.layers, adata.obsm, adata.varm, adata.obsp):
        for key in list(collection.keys()):
            collection[key] = _to_cpu(collection[key])


def _move_input_matrix_to_gpu(adata: ad.AnnData, pca_params: Dict[str, Any]) -> Optional[str]:
    """Move the matrix used by RAPIDS PCA to the GPU."""
    layer = pca_params.get("layer")
    if layer:
        layer = str(layer)
        logging.info("Moving AnnData layer '%s' to GPU for RAPIDS PCA.", layer)
        rsc.get.anndata_to_GPU(adata, layer=layer)
        return layer

    logging.info("Moving AnnData.X to GPU for RAPIDS processing.")
    rsc.get.anndata_to_GPU(adata)
    return None


def _move_input_matrix_to_cpu(adata: ad.AnnData, layer: Optional[str]) -> None:
    """Move the matrix used by RAPIDS back to CPU memory."""
    if layer:
        logging.info("Moving AnnData layer '%s' back to CPU memory.", layer)
        rsc.get.anndata_to_CPU(adata, layer=layer)
    else:
        logging.info("Moving AnnData.X back to CPU memory.")
        rsc.get.anndata_to_CPU(adata)
    _ensure_cpu_storage(adata)


def _plot_embedding(adata: ad.AnnData, *, embedding_key: str, color: Any) -> None:
    """Plot the configured UMAP embedding with Scanpy."""
    if embedding_key == "X_umap":
        sc.pl.umap(adata, color=color, show=False)
        return
    sc.pl.embedding(adata, basis=embedding_key, color=color, show=False)


def _reorder_vars_by_expression(adata: ad.AnnData, var_names: List[str]) -> List[str]:
    """Order markers by hierarchical clustering of expression profiles."""
    if len(var_names) < 2:
        return list(var_names)

    from scipy import sparse
    import scipy.cluster.hierarchy as sch
    import scipy.spatial.distance as ssd

    subset = adata[:, var_names]
    expression_matrix = subset.X
    if sparse.issparse(expression_matrix):
        expression_matrix = expression_matrix.toarray()

    distances = ssd.pdist(np.asarray(expression_matrix).T, metric="euclidean")
    if distances.size == 0:
        return list(var_names)
    linkage_matrix = sch.linkage(distances, method="ward")
    ordered_indices = sch.dendrogram(linkage_matrix, no_plot=True)["leaves"]
    return [var_names[idx] for idx in ordered_indices]


def _save_qc_umaps(
    adata: ad.AnnData,
    *,
    batch_key: Optional[str],
    leiden_keys: List[str],
    qc_dir: Path,
    embedding_key: str,
    method_slug: str,
) -> None:
    """Save a small set of UMAP QC plots for the RAPIDS stage."""
    if embedding_key not in adata.obsm:
        logging.warning(
            "UMAP embedding '%s' missing; skipping RAPIDS QC plots.",
            embedding_key,
        )
        return

    if batch_key and batch_key in adata.obs.columns:
        _plot_embedding(adata, embedding_key=embedding_key, color=batch_key)
        fig = plt.gcf()
        fig.savefig(
            qc_dir / f"umap_{method_slug}_{cleanstring(batch_key) or 'batch'}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    for leiden_key in leiden_keys:
        if leiden_key not in adata.obs.columns:
            continue
        colors: Any = [leiden_key]
        suffix = cleanstring(leiden_key) or "leiden"
        if batch_key and batch_key in adata.obs.columns:
            colors = [batch_key, leiden_key]
            suffix = f"{cleanstring(batch_key) or 'batch'}_vs_{suffix}"
        _plot_embedding(adata, embedding_key=embedding_key, color=colors)
        fig = plt.gcf()
        fig.savefig(
            qc_dir / f"umap_{method_slug}_{suffix}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def _save_leiden_matrixplots(
    adata: ad.AnnData,
    *,
    leiden_keys: List[str],
    qc_dir: Path,
    viz_config: VisualizationConfig,
) -> List[str]:
    """Save non-scaled vmax-capped MatrixPlots for RAPIDS Leiden outputs."""
    if not leiden_keys:
        logging.info("No Leiden keys found; skipping RAPIDS MatrixPlots.")
        return []
    if adata.n_vars == 0:
        logging.warning("AnnData has no markers; skipping RAPIDS MatrixPlots.")
        return []

    matrix_dir = qc_dir / "Matrixplots"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    save_dpi = 300 if getattr(viz_config, "save_high_res", True) else 150
    figure_format = getattr(viz_config, "figure_format", "png")
    matrixplot_vmax = getattr(viz_config, "matrixplot_vmax", 0.5)
    markers_to_plot = adata.var_names.tolist()

    try:
        ordered_markers = _reorder_vars_by_expression(adata, markers_to_plot)
    except Exception as exc:  # pragma: no cover - defensive QC fallback
        logging.warning(
            "Could not reorder markers by expression for RAPIDS MatrixPlots (%s). "
            "Using AnnData.var_names order.",
            exc,
        )
        ordered_markers = markers_to_plot

    use_row_color_matrixplot = getattr(viz_config, "matrixplot_use_row_colors", True)
    if use_row_color_matrixplot:
        if sbt_plotting is None or not hasattr(sbt_plotting, "matrixplot_with_row_colors"):
            logging.warning(
                "matrixplot_use_row_colors=True but plotting.matrixplot_with_row_colors "
                "is unavailable. Falling back to scanpy.pl.matrixplot."
            )
            use_row_color_matrixplot = False

    saved_paths: List[str] = []
    for leiden_key in leiden_keys:
        if leiden_key not in adata.obs.columns:
            logging.warning(
                "Leiden key '%s' not found in adata.obs; skipping RAPIDS MatrixPlot.",
                leiden_key,
            )
            continue

        safe_key = cleanstring(leiden_key) or "leiden"
        out_path = matrix_dir / f"Matrixplot_{safe_key}_vmax.{figure_format}"
        logging.info(
            "Creating vmax-capped RAPIDS MatrixPlot for %s (vmax=%s).",
            leiden_key,
            matrixplot_vmax,
        )
        try:
            sc.tl.dendrogram(adata, groupby=leiden_key)
            if use_row_color_matrixplot:
                _, fig = sbt_plotting.matrixplot_with_row_colors(
                    adata,
                    marker_groups=ordered_markers,
                    groupby_key=leiden_key,
                    out_path=str(out_path),
                    reorder_var_by_expression=False,
                    standard_scale=None,
                    vmax=matrixplot_vmax,
                    dendrogram=True,
                    save_dpi=save_dpi,
                )
                plt.close(fig)
            else:
                matrixplot_obj = sc.pl.matrixplot(
                    adata,
                    var_names=ordered_markers,
                    groupby=leiden_key,
                    standard_scale=None,
                    dendrogram=True,
                    vmax=matrixplot_vmax,
                    show=False,
                    return_fig=True,
                )
                matrixplot_obj.savefig(out_path, bbox_inches="tight", dpi=save_dpi)
                plt.close()
            saved_paths.append(str(out_path))
            logging.info("RAPIDS MatrixPlot saved to %s", out_path)
        except Exception as exc:  # pragma: no cover - QC should not block saving AnnData
            logging.exception(
                "Failed to create RAPIDS MatrixPlot for Leiden key '%s': %s",
                leiden_key,
                exc,
            )

    return saved_paths


def _run_rapids_pca(
    adata: ad.AnnData,
    *,
    n_pcs: int,
    pca_key: str,
    pca_params: Dict[str, Any],
) -> None:
    """Run RAPIDS PCA and store the embedding under the configured key."""
    params = _drop_managed_params(
        pca_params,
        managed={"n_comps", "key_added", "copy"},
        section_name="pca",
    )
    if pca_key != "X_pca":
        params["key_added"] = pca_key

    logging.info("Running RAPIDS PCA with %d components.", n_pcs)
    rsc.pp.pca(adata, n_comps=n_pcs, **params)
    if pca_key not in adata.obsm:
        raise RuntimeError(f"RAPIDS PCA did not create adata.obsm['{pca_key}'].")


def _run_rapids_harmony(
    adata: ad.AnnData,
    *,
    batch_key: str,
    pca_key: str,
    harmony_key: str,
    harmony_params: Dict[str, Any],
) -> None:
    """Run RAPIDS Harmony integration against the configured PCA embedding."""
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Configured batch key '{batch_key}' was not found in adata.obs")

    params = _drop_managed_params(
        _normalise_dtype_param(harmony_params),
        managed={"key", "basis", "adjusted_basis"},
        section_name="harmony",
    )
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    logging.info(
        "Running RAPIDS Harmony on adata.obsm['%s'] using obs key '%s'.",
        pca_key,
        batch_key,
    )
    rsc.pp.harmony_integrate(
        adata,
        key=batch_key,
        basis=pca_key,
        adjusted_basis=harmony_key,
        **params,
    )
    if harmony_key not in adata.obsm:
        raise RuntimeError(f"RAPIDS Harmony did not create adata.obsm['{harmony_key}'].")


def _run_rapids_neighbors(
    adata: ad.AnnData,
    *,
    representation_key: str,
    n_neighbors: Optional[int],
    n_pcs: int,
    neighbors_key: Optional[str],
    neighbors_params: Dict[str, Any],
) -> str:
    """Build the RAPIDS neighborhood graph."""
    params = _drop_managed_params(
        neighbors_params,
        managed={"n_neighbors", "n_pcs", "use_rep", "key_added", "copy"},
        section_name="neighbors",
    )
    if n_neighbors is not None:
        params["n_neighbors"] = int(n_neighbors)
    params["n_pcs"] = int(n_pcs)
    params["use_rep"] = representation_key
    if neighbors_key:
        params["key_added"] = neighbors_key

    logging.info(
        "Computing RAPIDS neighbors from adata.obsm['%s'] with n_pcs=%s.",
        representation_key,
        n_pcs,
    )
    rsc.pp.neighbors(adata, **params)
    return neighbors_key or "neighbors"


def _run_rapids_umap(
    adata: ad.AnnData,
    *,
    umap_min_dist: float,
    neighbors_key: str,
    umap_key: Optional[str],
    umap_params: Dict[str, Any],
) -> str:
    """Run RAPIDS UMAP."""
    params = _drop_managed_params(
        umap_params,
        managed={"min_dist", "key_added", "neighbors_key", "copy"},
        section_name="umap",
    )
    params["min_dist"] = float(umap_min_dist)
    if neighbors_key != "neighbors":
        params["neighbors_key"] = neighbors_key
    if umap_key:
        params["key_added"] = umap_key

    logging.info("Running RAPIDS UMAP (min_dist=%s).", umap_min_dist)
    rsc.tl.umap(adata, **params)
    return umap_key or "X_umap"


def _run_rapids_leiden(
    adata: ad.AnnData,
    *,
    resolutions: List[float],
    enabled: bool,
    neighbors_key: str,
    leiden_params: Dict[str, Any],
) -> List[str]:
    """Run RAPIDS Leiden clustering for each configured resolution."""
    if not enabled:
        logging.info("Leiden clustering skipped (run_leiden=False).")
        return []

    params = _drop_managed_params(
        leiden_params,
        managed={"resolution", "key_added", "neighbors_key", "copy"},
        section_name="leiden",
    )
    if neighbors_key != "neighbors":
        params["neighbors_key"] = neighbors_key

    leiden_keys: List[str] = []
    for res in resolutions:
        leiden_key = f"leiden_{res}"
        logging.info("Running RAPIDS Leiden clustering at resolution %s.", res)
        rsc.tl.leiden(adata, resolution=res, key_added=leiden_key, **params)
        leiden_keys.append(leiden_key)
    return leiden_keys


def main() -> None:
    pipeline_stage = "RapidsProcess"
    config = process_config_with_overrides()
    setup_logging(config.get("logging", {}), pipeline_stage)

    general_config = GeneralConfig(
        **filter_config_for_dataclass(config.get("general", {}), GeneralConfig)
    )
    rapids_config = RapidsProcessConfig(
        **filter_config_for_dataclass(config.get("rapids", {}), RapidsProcessConfig)
    )
    # Reuse visualization settings for QC MatrixPlot formatting.
    viz_config = VisualizationConfig(
        **filter_config_for_dataclass(config.get("visualization", {}), VisualizationConfig)
    )

    input_path = rapids_config.input_adata_path or general_config.anndata_path
    output_path = rapids_config.output_adata_path or general_config.anndata_path
    adata, resolved_input_path, skip_stage, _ = load_pipeline_anndata(
        general_config=general_config,
        stage_name=pipeline_stage,
        stage_config=rapids_config,
        override_path=input_path,
    )
    if skip_stage:
        logging.info("Skipping RAPIDS processing stage based on AnnData stage policy.")
        return
    if adata is None:
        raise FileNotFoundError(
            f"AnnData could not be loaded for RAPIDS processing stage: {resolved_input_path}"
        )

    logging.info("AnnData loaded with shape %s and %d markers.", adata.shape, adata.n_vars)

    run_harmony = bool(rapids_config.run_harmony)
    batch_key = _normalise_optional_key(rapids_config.batch_correction_obs)
    if run_harmony and not batch_key:
        raise ValueError("rapids.batch_correction_obs must be set when rapids.run_harmony=True.")
    if batch_key and batch_key not in adata.obs.columns:
        raise KeyError(f"Configured batch key '{batch_key}' was not found in adata.obs")

    pca_params = _clean_params(rapids_config.pca_params)
    harmony_params = _clean_params(rapids_config.harmony_params)
    neighbors_params = _clean_params(rapids_config.neighbors_params)
    umap_params = _clean_params(rapids_config.umap_params)
    leiden_params = _clean_params(rapids_config.leiden_params)

    pca_key = rapids_config.pca_key
    harmony_key = rapids_config.harmony_key
    representation_key = rapids_config.representation_key
    neighbors_key = _normalise_optional_key(rapids_config.neighbors_key)
    umap_key = _normalise_optional_key(rapids_config.umap_key)
    qc_dir = Path(general_config.qc_folder) / rapids_config.qc_output_subdir
    qc_dir.mkdir(parents=True, exist_ok=True)

    n_pcs = _resolve_n_pcs(adata, rapids_config.n_for_pca)
    gpu_layer = _move_input_matrix_to_gpu(adata, pca_params)

    try:
        _run_rapids_pca(
            adata,
            n_pcs=n_pcs,
            pca_key=pca_key,
            pca_params=pca_params,
        )

        active_representation = pca_key
        if run_harmony:
            _run_rapids_harmony(
                adata,
                batch_key=str(batch_key),
                pca_key=pca_key,
                harmony_key=harmony_key,
                harmony_params=harmony_params,
            )
            active_representation = harmony_key

        adata.obsm[representation_key] = _copy_array(adata.obsm[active_representation])
        logging.info(
            "Stored active RAPIDS representation in adata.obsm['%s'] from adata.obsm['%s'].",
            representation_key,
            active_representation,
        )

        n_pcs_neighbors = _resolve_neighbor_n_pcs(
            adata,
            representation_key=representation_key,
            requested=rapids_config.n_pcs_neighbors,
            default=n_pcs,
        )
        graph_key = _run_rapids_neighbors(
            adata,
            representation_key=representation_key,
            n_neighbors=rapids_config.n_neighbors,
            n_pcs=n_pcs_neighbors,
            neighbors_key=neighbors_key,
            neighbors_params=neighbors_params,
        )
        embedding_key = _run_rapids_umap(
            adata,
            umap_min_dist=rapids_config.umap_min_dist,
            neighbors_key=graph_key,
            umap_key=umap_key,
            umap_params=umap_params,
        )
        leiden_keys = _run_rapids_leiden(
            adata,
            resolutions=_as_resolution_list(rapids_config.leiden_resolutions_list),
            enabled=bool(rapids_config.run_leiden),
            neighbors_key=graph_key,
            leiden_params=leiden_params,
        )
    finally:
        _move_input_matrix_to_cpu(adata, gpu_layer)

    method = "rapids_harmony" if run_harmony else "rapids"
    _save_qc_umaps(
        adata,
        batch_key=batch_key,
        leiden_keys=leiden_keys,
        qc_dir=qc_dir,
        embedding_key=embedding_key,
        method_slug=method,
    )
    matrixplot_paths = _save_leiden_matrixplots(
        adata,
        leiden_keys=leiden_keys,
        qc_dir=qc_dir,
        viz_config=viz_config,
    )

    adata.uns["rapids_process"] = {
        "method": method,
        "batch_key": batch_key,
        "n_pcs": int(n_pcs),
        "n_pcs_neighbors": int(n_pcs_neighbors),
        "representation_key": representation_key,
        "source_representation_key": active_representation,
        "pca_key": pca_key,
        "harmony_key": harmony_key if run_harmony else None,
        "neighbors_key": graph_key,
        "umap_key": embedding_key,
        "run_leiden": bool(rapids_config.run_leiden),
        "leiden_keys": leiden_keys,
        "pca_params": pca_params,
        "harmony_params": harmony_params if run_harmony else {},
        "neighbors_params": neighbors_params,
        "umap_params": umap_params,
        "leiden_params": leiden_params,
        "qc_dir": str(qc_dir),
        "matrixplot_paths": matrixplot_paths,
    }
    adata.uns["batch_integration"] = {
        "method": method,
        "batch_key": batch_key,
        "representation_key": representation_key,
        "source_representation_key": active_representation,
        "pca_key": pca_key,
        "harmony_key": harmony_key if run_harmony else None,
    }

    saved_path = save_pipeline_anndata(
        adata=adata,
        general_config=general_config,
        stage_name=pipeline_stage,
        stage_config=rapids_config,
        override_path=output_path,
        extra_details={
            "input_adata_path": str(resolved_input_path),
            "output_adata_path": str(output_path),
            "method": method,
            "representation_key": representation_key,
            "neighbors_key": graph_key,
            "umap_key": embedding_key,
            "qc_dir": str(qc_dir),
            "matrixplot_count": len(matrixplot_paths),
            "n_cells": int(adata.n_obs),
            "n_markers": int(adata.n_vars),
        },
    )
    logging.info("Saved RAPIDS-processed AnnData to %s", saved_path)


if __name__ == "__main__":
    main()
