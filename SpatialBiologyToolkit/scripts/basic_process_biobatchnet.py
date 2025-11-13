"""
Processing pipeline that runs BioBatchNet for batch correction and produces
a processed AnnData object ready for downstream visualization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

try:
    from biobatchnet import correct_batch_effects
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "BioBatchNet is required for this script. Please install it with "
        "'pip install biobatchnet' (and ensure PyTorch is available)."
    ) from exc

try:  # Torch is optional but helps us pick the right device
    import torch
except ImportError:  # pragma: no cover - torch optional
    torch = None  # type: ignore

from .config_and_utils import (  # pylint: disable=relative-beyond-top-level
    BasicProcessConfig,
    GeneralConfig,
    VisualizationConfig,
    filter_config_for_dataclass,
    process_config_with_overrides,
    setup_logging,
)


def _ensure_dense_matrix(X) -> np.ndarray:
    """Convert AnnData.X (possibly sparse) into a dense NumPy array for BioBatchNet."""
    from scipy import sparse  # Local import to avoid hard dependency at module import time

    if isinstance(X, np.ndarray):
        return X
    if sparse.issparse(X):
        logging.info("Converting sparse matrix to dense array for BioBatchNet.")
        return X.toarray()
    return np.asarray(X)


def _infer_device(preferred: Optional[str] = None) -> str:
    """Pick an execution device for BioBatchNet."""
    if preferred:
        pref = preferred.lower()
        if pref == "cuda" and torch is not None and torch.cuda.is_available():  # type: ignore[attr-defined]
            return "cuda"
        if pref in {"cpu", "cuda"}:
            if pref == "cuda":
                logging.warning("CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"
    if torch is not None and torch.cuda.is_available():  # type: ignore[attr-defined]
        return "cuda"
    return "cpu"


def run_biobatchnet_correction(
    adata: ad.AnnData,
    batch_key: str,
    *,
    data_type: str = "imc",
    latent_dim: int = 20,
    epochs: int = 100,
    device: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    use_raw: bool = True,
) -> None:
    """
    Run BioBatchNet on the provided AnnData object and attach the embeddings.
    
    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing expression data
    batch_key : str
        The column name in adata.obs that contains batch information
    data_type : str, default "imc"
        Type of data for BioBatchNet
    latent_dim : int, default 20
        Latent dimension for BioBatchNet
    epochs : int, default 100
        Number of training epochs
    device : str, optional
        Computing device ('cpu' or 'cuda')
    extra_params : dict, optional
        Additional parameters for BioBatchNet
    use_raw : bool, default True
        If True, use adata.raw.X (raw data) for batch correction.
        If False, use adata.X (potentially normalized data).
    """
    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch column '{batch_key}' not found in AnnData.obs")

    # Use raw data by default, fall back to .X if raw is not available
    if use_raw and adata.raw is not None:
        matrix = _ensure_dense_matrix(adata.raw.X)
        var_names = adata.raw.var_names
        logging.info("Using raw data (adata.raw.X) for BioBatchNet correction")
    else:
        if use_raw and adata.raw is None:
            logging.warning("Raw data requested but not available. Using normalized data (adata.X)")
        else:
            logging.info("Using normalized data (adata.X) for BioBatchNet correction")
        matrix = _ensure_dense_matrix(adata.X)
        var_names = adata.var_names
    
    expression_df = pd.DataFrame(matrix, index=adata.obs_names, columns=var_names)

    batch_series = adata.obs[batch_key].astype(str)
    unique_batches = pd.Index(sorted(batch_series.unique()))
    batch_map = {batch: idx for idx, batch in enumerate(unique_batches)}
    batch_ids = batch_series.map(batch_map).astype(int)
    batch_df = pd.DataFrame({batch_key: batch_ids}, index=adata.obs_names)

    device_to_use = _infer_device(device)
    params = {
        "data": expression_df,
        "batch_info": batch_df,
        "batch_key": batch_key,
        "data_type": data_type,
        "latent_dim": latent_dim,
        "epochs": epochs,
        "device": device_to_use,
    }
    if extra_params:
        params.update(extra_params)

    logging.info(
        "Running BioBatchNet (data_type=%s, latent_dim=%s, epochs=%s, device=%s, batches=%d)",
        data_type,
        latent_dim,
        epochs,
        device_to_use,
        len(unique_batches),
    )

    bio_embeddings, batch_embeddings = correct_batch_effects(**params)

    def _to_numpy(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    bio_embeddings = _to_numpy(bio_embeddings)
    if bio_embeddings is None:
        raise RuntimeError("BioBatchNet returned no biological embeddings.")

    adata.obsm["X_biobatchnet"] = bio_embeddings
    batch_embeddings = _to_numpy(batch_embeddings)
    if batch_embeddings is not None:
        adata.obsm["X_biobatchnet_batch"] = batch_embeddings

    adata.uns["biobatchnet"] = {
        "batch_key": batch_key,
        "batch_mapping": batch_map,
        "data_type": data_type,
        "latent_dim": latent_dim,
        "epochs": epochs,
        "device": device_to_use,
    }
    logging.info("BioBatchNet embeddings stored in adata.obsm['X_biobatchnet'].")


def main() -> None:
    pipeline_stage = "BioBatchNetProcess"
    config = process_config_with_overrides()
    setup_logging(config.get("logging", {}), pipeline_stage)

    general_config = GeneralConfig(
        **filter_config_for_dataclass(config.get("general", {}), GeneralConfig)
    )
    process_config = BasicProcessConfig(
        **filter_config_for_dataclass(config.get("process", {}), BasicProcessConfig)
    )
    # Visualization config is still parsed so downstream modules can reuse the file
    _ = VisualizationConfig(
        **filter_config_for_dataclass(config.get("visualization", {}), VisualizationConfig)
    )

    input_path = Path(process_config.input_adata_path)
    output_path = Path(process_config.output_adata_path)

    logging.info("Loading AnnData from %s", input_path)
    adata = ad.read_h5ad(input_path)
    logging.info("AnnData loaded with shape %s and %d markers.", adata.shape, adata.n_vars)

    batch_key = process_config.batch_correction_obs
    if not batch_key:
        raise ValueError(
            "process.batch_correction_obs must be set in the config to use BioBatchNet."
        )

    biobatchnet_params = {
        "data_type": process_config.biobatchnet_data_type,
        "latent_dim": process_config.biobatchnet_latent_dim,
        "epochs": process_config.biobatchnet_epochs,
        "device": process_config.biobatchnet_device,
        "extra_params": process_config.biobatchnet_kwargs,
        "use_raw": process_config.biobatchnet_use_raw,
    }

    run_biobatchnet_correction(
        adata,
        batch_key=batch_key,
        data_type=biobatchnet_params["data_type"],
        latent_dim=biobatchnet_params["latent_dim"],
        epochs=biobatchnet_params["epochs"],
        device=biobatchnet_params["device"],
        extra_params=biobatchnet_params["extra_params"],
        use_raw=biobatchnet_params["use_raw"],
    )

    neighbors_kwargs: Dict[str, Any] = {}
    if process_config.n_neighbors is not None:
        neighbors_kwargs["n_neighbors"] = process_config.n_neighbors
    logging.info("Computing neighbors using BioBatchNet embeddings.")
    sc.pp.neighbors(adata, use_rep="X_biobatchnet", **neighbors_kwargs)

    logging.info("Running UMAP (min_dist=%s).", process_config.umap_min_dist)
    sc.tl.umap(adata, min_dist=process_config.umap_min_dist)

    resolutions = process_config.leiden_resolutions_list
    if resolutions and not isinstance(resolutions, list):
        resolutions = [resolutions]

    for res in resolutions:
        logging.info("Running Leiden clustering at resolution %s.", res)
        sc.tl.leiden(adata, resolution=res, key_added=f"leiden_{res}")

    qc_dir = Path(general_config.qc_folder) / "BioBatchNet"
    qc_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Saving UMAP plots to %s", qc_dir)

    if batch_key in adata.obs:
        def _save_combined_umap(leiden_key: str, suffix: str) -> None:
            sc.pl.umap(
                adata,
                color=[batch_key, leiden_key],
                ncols=2,
                legend_loc="on data",
                frameon=False,
                show=False,
            )
            fig = plt.gcf()
            fig.savefig(qc_dir / f"umap_{suffix}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

        for res in resolutions:
            leiden_key = f"leiden_{res}"
            if leiden_key in adata.obs:
                suffix = f"{batch_key}_vs_{leiden_key}"
                _save_combined_umap(leiden_key, suffix)
    else:
        logging.warning("Batch key '%s' missing from AnnData.obs; skipping QC UMAPs.", batch_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    logging.info("Saved batch-corrected AnnData to %s", output_path)


if __name__ == "__main__":
    main()
