"""
Batch integration stage for IMC AnnData outputs.

Supports Harmony (via harmonypy), BBKNN, or Harmony+BBKNN after Nimbus /
segmentation, and then computes neighbors, UMAP, and Leiden clustering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import anndata as ad
import matplotlib
import numpy as np
import scanpy as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import harmonypy as hm
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "harmonypy is required for this script. Please install it with "
        "'pip install harmonypy' (and ensure PyTorch is available)."
    ) from exc

from .config_and_utils import (
    BatchIntegrationConfig,
    GeneralConfig,
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
        raise ValueError("Batch integration requires at least 2 cells in the AnnData object.")
    if adata.n_vars < 1:
        raise ValueError("Batch integration requires at least 1 marker in the AnnData object.")

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


def _normalise_method(method: str) -> str:
    value = str(method).strip().lower()
    if value in {"", "none", "null"}:
        return "none"
    if value not in {"harmony", "bbknn", "both", "none"}:
        raise ValueError(
            "batch_integration.integration_method must be one of: 'harmony', 'bbknn', 'both', 'none'."
        )
    return value


def _run_direct_harmony(
    adata: ad.AnnData,
    *,
    batch_key: str,
    pca_key: str,
    harmony_key: str,
    representation_key: str,
    harmony_params: Dict[str, Any],
) -> None:
    """Run harmonypy directly on the stored PCA embedding."""
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Batch correction observation '{batch_key}' not found in adata.obs")

    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    pcs = np.asarray(adata.obsm[pca_key], dtype=np.float32)

    logging.info("Running Harmony directly on adata.obsm['%s'].", pca_key)
    harmony_out = hm.run_harmony(
        pcs,
        adata.obs,
        batch_key,
        **harmony_params,
    )
    corrected = np.asarray(harmony_out.Z_corr, dtype=np.float32)
    adata.obsm[harmony_key] = corrected
    adata.obsm[representation_key] = corrected.copy()
    logging.info(
        "Stored Harmony-corrected PCs in adata.obsm['%s'] and adata.obsm['%s'].",
        harmony_key,
        representation_key,
    )


def _run_neighbors_stage(
    adata: ad.AnnData,
    *,
    method: str,
    batch_key: Optional[str],
    n_pcs: int,
    representation_key: str,
    n_neighbors: Optional[int],
    bbknn_params: Dict[str, Any],
) -> str:
    """Build the neighborhood graph using the configured integration strategy."""
    if method in {"bbknn", "both"}:
        if not batch_key:
            raise ValueError("A batch key is required when integration_method is 'bbknn' or 'both'.")
        params = dict(bbknn_params)
        params.setdefault("n_pcs", n_pcs)
        params.setdefault("use_rep", representation_key)
        if n_neighbors is not None and "neighbors_within_batch" not in params:
            n_batches = max(1, int(adata.obs[batch_key].astype(str).nunique()))
            params["neighbors_within_batch"] = max(1, int(np.ceil(int(n_neighbors) / n_batches)))
            logging.info(
                "Derived BBKNN neighbors_within_batch=%s from n_neighbors=%s across %s batch(es).",
                params["neighbors_within_batch"],
                int(n_neighbors),
                n_batches,
            )
        logging.info(
            "Running BBKNN with batch_key='%s' on adata.obsm['%s'].",
            batch_key,
            representation_key,
        )
        sc.external.pp.bbknn(adata, batch_key=batch_key, **params)
        return "bbknn"

    neighbors_kwargs: Dict[str, Any] = {}
    if n_neighbors is not None:
        neighbors_kwargs["n_neighbors"] = int(n_neighbors)
    logging.info("Computing neighbors from adata.obsm['%s'].", representation_key)
    sc.pp.neighbors(adata, use_rep=representation_key, **neighbors_kwargs)
    return "scanpy_neighbors"


def _run_leiden(adata: ad.AnnData, *, resolutions: List[float], enabled: bool) -> List[str]:
    """Run Leiden clustering for each configured resolution."""
    if not enabled:
        logging.info("Leiden clustering skipped (run_leiden=False).")
        return []

    leiden_keys: List[str] = []
    for res in resolutions:
        leiden_key = f"leiden_{res}"
        logging.info("Running Leiden clustering at resolution %s.", res)
        sc.tl.leiden(adata, resolution=res, key_added=leiden_key)
        leiden_keys.append(leiden_key)
    return leiden_keys


def _save_qc_umaps(
    adata: ad.AnnData,
    *,
    batch_key: Optional[str],
    leiden_keys: List[str],
    qc_dir: Path,
    method: str,
) -> None:
    """Save a small set of UMAP QC plots for batch correction review."""
    if "X_umap" not in adata.obsm:
        logging.warning("UMAP coordinates missing; skipping batch integration QC plots.")
        return

    method_slug = cleanstring(method) or "integration"

    if batch_key and batch_key in adata.obs.columns:
        sc.pl.umap(adata, color=batch_key, show=False)
        fig = plt.gcf()
        fig.savefig(
            qc_dir / f"umap_{method_slug}_{cleanstring(batch_key) or 'batch'}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    for leiden_key in leiden_keys:
        colors = [leiden_key]
        suffix = cleanstring(leiden_key) or "leiden"
        if batch_key and batch_key in adata.obs.columns:
            colors = [batch_key, leiden_key]
            suffix = f"{cleanstring(batch_key) or 'batch'}_vs_{suffix}"
        sc.pl.umap(adata, color=colors, show=False)
        fig = plt.gcf()
        fig.savefig(
            qc_dir / f"umap_{method_slug}_{suffix}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    pipeline_stage = "BatchIntegration"
    config = process_config_with_overrides()
    setup_logging(config.get("logging", {}), pipeline_stage)

    general_config = GeneralConfig(
        **filter_config_for_dataclass(config.get("general", {}), GeneralConfig)
    )
    batch_section = config.get("batch_integration", config.get("process", {}))
    batch_config = BatchIntegrationConfig(
        **filter_config_for_dataclass(batch_section, BatchIntegrationConfig)
    )
    # Parsed for consistency with other stage entrypoints.
    _ = VisualizationConfig(
        **filter_config_for_dataclass(config.get("visualization", {}), VisualizationConfig)
    )

    input_path = batch_config.input_adata_path or general_config.anndata_path
    output_path = batch_config.output_adata_path or general_config.anndata_path
    
    adata, resolved_input_path, skip_stage, _ = load_pipeline_anndata(
        general_config=general_config,
        stage_name=pipeline_stage,
        stage_config=batch_config,
        override_path=input_path,
    )
    if skip_stage:
        logging.info("Skipping batch integration stage based on AnnData stage policy.")
        return
    if adata is None:
        raise FileNotFoundError(
            f"AnnData could not be loaded for batch integration stage: {resolved_input_path}"
        )

    method = _normalise_method(batch_config.integration_method)
    batch_key = batch_config.batch_correction_obs
    if method != "none" and not batch_key:
        raise ValueError(
            "batch_integration.batch_correction_obs must be set when using Harmony and/or BBKNN."
        )
    if batch_key and batch_key not in adata.obs.columns:
        raise KeyError(f"Configured batch key '{batch_key}' was not found in adata.obs")

    n_pcs = _resolve_n_pcs(adata, batch_config.n_for_pca)
    harmony_params = _clean_params(batch_config.harmony_params)
    bbknn_params = _clean_params(batch_config.bbknn_params)
    qc_dir = Path(general_config.qc_folder) / batch_config.qc_output_subdir
    qc_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Running PCA with %d components.", n_pcs)
    sc.pp.pca(adata, n_comps=n_pcs)
    adata.obsm[batch_config.representation_key] = np.asarray(
        adata.obsm[batch_config.pca_key],
        dtype=np.float32,
    ).copy()

    if method in {"harmony", "both"}:
        _run_direct_harmony(
            adata,
            batch_key=str(batch_key),
            pca_key=batch_config.pca_key,
            harmony_key=batch_config.harmony_key,
            representation_key=batch_config.representation_key,
            harmony_params=harmony_params,
        )

    graph_method = _run_neighbors_stage(
        adata,
        method=method,
        batch_key=batch_key,
        n_pcs=n_pcs,
        representation_key=batch_config.representation_key,
        n_neighbors=batch_config.n_neighbors,
        bbknn_params=bbknn_params,
    )

    logging.info("Running UMAP (min_dist=%s).", batch_config.umap_min_dist)
    sc.tl.umap(adata, min_dist=batch_config.umap_min_dist)

    resolutions = batch_config.leiden_resolutions_list
    if resolutions and not isinstance(resolutions, list):
        resolutions = [resolutions]
    else:
        resolutions = resolutions or []
    leiden_keys = _run_leiden(
        adata,
        resolutions=[float(x) for x in resolutions],
        enabled=bool(batch_config.run_leiden),
    )
    _save_qc_umaps(
        adata,
        batch_key=batch_key,
        leiden_keys=leiden_keys,
        qc_dir=qc_dir,
        method=method,
    )

    adata.uns["batch_integration"] = {
        "method": method,
        "batch_key": batch_key,
        "n_pcs": int(n_pcs),
        "graph_method": graph_method,
        "representation_key": batch_config.representation_key,
        "pca_key": batch_config.pca_key,
        "harmony_key": batch_config.harmony_key if method in {"harmony", "both"} else None,
        "run_leiden": bool(batch_config.run_leiden),
        "leiden_keys": leiden_keys,
        "harmony_params": harmony_params,
        "bbknn_params": bbknn_params,
        "qc_dir": str(qc_dir),
    }

    saved_path = save_pipeline_anndata(
        adata=adata,
        general_config=general_config,
        stage_name=pipeline_stage,
        stage_config=batch_config,
        override_path=output_path,
        extra_details={
            "input_adata_path": str(resolved_input_path),
            "output_adata_path": str(output_path),
            "method": method,
            "representation_key": batch_config.representation_key,
            "qc_dir": str(qc_dir),
            "n_cells": int(adata.n_obs),
            "n_markers": int(adata.n_vars),
        },
    )
    logging.info("Saved batch-integrated AnnData to %s", saved_path)


if __name__ == "__main__":
    main()
