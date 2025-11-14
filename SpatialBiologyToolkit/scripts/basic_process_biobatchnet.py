"""
Processing pipeline that runs BioBatchNet for batch correction and produces
a processed AnnData object ready for downstream visualization.
"""

from __future__ import annotations

import copy
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    cleanstring,
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


def _merge_run_params(
    base_params: Dict[str, Any],
    overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a deep-copied parameter dictionary with overrides applied."""
    params = copy.deepcopy(base_params)
    if not overrides:
        return params
    for key, value in overrides.items():
        if key == "name":
            continue
        if key == "extra_params":
            merged_extra = copy.deepcopy(params.get("extra_params") or {})
            if value:
                merged_extra.update(value)
            params["extra_params"] = merged_extra
            continue
        params[key] = value
    return params


def _slugify_label(label: Optional[str], fallback: str) -> Optional[str]:
    """Convert a label into a filesystem-friendly suffix."""
    if not label:
        return None
    slug = cleanstring(label)
    if not slug:
        return fallback
    return slug


def _prepare_output_path(base_path: Path, suffix: Optional[str]) -> Path:
    """Attach a suffix to the output filename if provided."""
    if not suffix:
        return base_path
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def _prepare_qc_dir(base_dir: Path, suffix: Optional[str]) -> Path:
    """Return a QC directory for the given suffix."""
    if not suffix:
        return base_dir
    return base_dir / suffix


def _postprocess_biobatchnet_results(
    adata: ad.AnnData,
    process_config: BasicProcessConfig,
    batch_key: str,
    qc_dir: Path,
) -> None:
    """Run neighbors/UMAP/clustering and save QC plots."""
    neighbors_kwargs: Dict[str, Any] = {}
    if process_config.n_neighbors is not None:
        neighbors_kwargs["n_neighbors"] = process_config.n_neighbors

    logging.info("Computing neighbors using BioBatchNet embeddings.")
    sc.pp.neighbors(adata, use_rep="X_biobatchnet", **neighbors_kwargs)

    logging.info("Running UMAP (min_dist=%s).", process_config.umap_min_dist)
    sc.tl.umap(adata, min_dist=process_config.umap_min_dist)

    # Only run Leiden clustering if enabled
    if process_config.biobatchnet_run_leiden:
        resolutions = process_config.leiden_resolutions_list
        if resolutions and not isinstance(resolutions, list):
            resolutions = [resolutions]
        else:
            resolutions = resolutions or []

        for res in resolutions:
            logging.info("Running Leiden clustering at resolution %s.", res)
            sc.tl.leiden(adata, resolution=res, key_added=f"leiden_{res}")
    else:
        logging.info("Leiden clustering skipped (biobatchnet_run_leiden=False).")

    qc_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Saving UMAP plots to %s", qc_dir)

    if batch_key not in adata.obs:
        logging.warning("Batch key '%s' missing from AnnData.obs; skipping QC UMAPs.", batch_key)
        return

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

    # Only create Leiden clustering UMAPs if Leiden was run
    if process_config.biobatchnet_run_leiden:
        resolutions = process_config.leiden_resolutions_list
        if resolutions and not isinstance(resolutions, list):
            resolutions = [resolutions]
        else:
            resolutions = resolutions or []
            
        for res in resolutions:
            leiden_key = f"leiden_{res}"
            if leiden_key in adata.obs:
                suffix = f"{batch_key}_vs_{leiden_key}"
                _save_combined_umap(leiden_key, suffix)
    else:
        # Create a simple batch-only UMAP
        sc.pl.umap(
            adata,
            color=batch_key,
            legend_loc="on data",
            frameon=False,
            show=False,
        )
        fig = plt.gcf()
        fig.savefig(qc_dir / f"umap_{batch_key}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def _run_single_parameter_set(
    base_adata: ad.AnnData,
    *,
    batch_key: str,
    process_config: BasicProcessConfig,
    base_params: Dict[str, Any],
    overrides: Optional[Dict[str, Any]],
    label: Optional[str],
    base_output_path: Path,
    base_qc_dir: Path,
) -> Dict[str, Any]:
    """Execute BioBatchNet + downstream processing for a specific parameter set."""
    run_params = _merge_run_params(base_params, overrides)
    suffix = _slugify_label(label, "scan")
    output_path = _prepare_output_path(base_output_path, suffix)
    qc_dir = _prepare_qc_dir(base_qc_dir, suffix)

    adata_copy = base_adata.copy()
    logging.info(
        "Running BioBatchNet for '%s' with parameters: %s",
        label or "base",
        {k: v for k, v in run_params.items() if k != "extra_params"},
    )
    run_biobatchnet_correction(
        adata_copy,
        batch_key=batch_key,
        data_type=run_params["data_type"],
        latent_dim=run_params["latent_dim"],
        epochs=run_params["epochs"],
        device=run_params["device"],
        extra_params=run_params.get("extra_params"),
        use_raw=run_params["use_raw"],
    )

    _postprocess_biobatchnet_results(
        adata_copy,
        process_config=process_config,
        batch_key=batch_key,
        qc_dir=qc_dir,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_copy.write_h5ad(output_path)
    logging.info("Saved batch-corrected AnnData to %s", output_path)

    return {
        "label": label or "base",
        "output_path": str(output_path),
        "qc_dir": str(qc_dir),
        "data_type": run_params["data_type"],
        "latent_dim": run_params["latent_dim"],
        "epochs": run_params["epochs"],
        "device": run_params["device"],
        "use_raw": run_params["use_raw"],
        "extra_params": run_params.get("extra_params"),
    }


def _write_scan_summary(rows: List[Dict[str, Any]], qc_dir: Path) -> None:
    """Persist a CSV summary of all parameter runs."""
    if not rows:
        return
    summary_path = qc_dir / "biobatchnet_parameter_scan_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "output_path",
        "qc_dir",
        "data_type",
        "latent_dim",
        "epochs",
        "device",
        "use_raw",
        "extra_params",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = row.copy()
            extra = serialized.get("extra_params")
            serialized["extra_params"] = json.dumps(extra, sort_keys=True) if extra else ""
            writer.writerow(serialized)
    logging.info("Wrote parameter scan summary to %s", summary_path)


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

    scan_sets = process_config.biobatchnet_scan_parameter_sets or []
    include_base = process_config.biobatchnet_scan_include_base or not scan_sets
    base_qc_dir = Path(general_config.qc_folder) / "BioBatchNet"
    base_qc_dir.mkdir(parents=True, exist_ok=True)

    scan_results: List[Dict[str, Any]] = []
    if include_base:
        scan_results.append(
            _run_single_parameter_set(
                adata,
                batch_key=batch_key,
                process_config=process_config,
                base_params=biobatchnet_params,
                overrides=None,
                label=None,
                base_output_path=output_path,
                base_qc_dir=base_qc_dir,
            )
        )

    for idx, scan_override in enumerate(scan_sets, start=1):
        label = scan_override.get("name") if isinstance(scan_override, dict) else None
        fallback_label = label or f"scan_{idx}"
        overrides = scan_override if isinstance(scan_override, dict) else {}
        logging.info("Running parameter scan '%s'.", fallback_label)
        scan_results.append(
            _run_single_parameter_set(
                adata,
                batch_key=batch_key,
                process_config=process_config,
                base_params=biobatchnet_params,
                overrides=overrides,
                label=fallback_label,
                base_output_path=output_path,
                base_qc_dir=base_qc_dir,
            )
        )

    if len(scan_results) > 1:
        _write_scan_summary(scan_results, base_qc_dir)


if __name__ == "__main__":
    main()
