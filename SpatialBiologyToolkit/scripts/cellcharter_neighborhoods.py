"""
CellCharter neighborhood analysis stage for IMC AnnData outputs.

This stage:
1. Loads an AnnData object from the configured pipeline output.
2. Computes TRVAE latent embeddings (default, configurable).
3. Builds a spatial graph per ROI/sample.
4. Aggregates neighborhood features with CellCharter.
5. Clusters cells into spatial neighborhoods.
6. Optionally runs neighborhood enrichment analyses.
7. Optionally runs shape characterisation analyses.
8. Saves updated AnnData plus QC tables/plots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import squidpy as sq
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "Squidpy is required for this script. Install it with 'pip install squidpy'."
    ) from exc

try:
    import cellcharter as cc
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "CellCharter is required for this script. Install it with 'pip install cellcharter'."
    ) from exc

from .config_and_utils import (
    CellCharterConfig,
    GeneralConfig,
    cleanstring,
    coalesce_config_list,
    coalesce_config_text,
    filter_config_for_dataclass,
    load_pipeline_anndata,
    process_config_with_overrides,
    save_pipeline_anndata,
    setup_logging,
)

# Optional plotting import for ROI mask overlays
try:
    from SpatialBiologyToolkit import plotting as sbt_plotting
except Exception:
    try:
        from .. import plotting as sbt_plotting  # type: ignore
    except Exception:
        sbt_plotting = None


def _to_dense_matrix(matrix: Any) -> np.ndarray:
    """Return a dense NumPy array from sparse/dense matrix-like input."""
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _parse_n_layers(value: Union[int, str, Sequence[Any]]) -> Union[int, List[int]]:
    """Parse n_layers from config into int or list[int]."""
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if "," in text:
            return [int(v.strip()) for v in text.split(",") if v.strip()]
        return int(text)
    return int(value)


def _parse_aggregations(value: Union[str, Sequence[Any]]) -> Union[str, List[str]]:
    """Parse aggregations config into string or list[str]."""
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and "," in value:
        return [v.strip() for v in value.split(",") if v.strip()]
    return str(value)


def _resolve_input_adata_path(
    general_config: GeneralConfig,
    cellcharter_config: CellCharterConfig,
) -> Path:
    """Resolve input AnnData path."""
    if cellcharter_config.input_adata_path:
        return Path(cellcharter_config.input_adata_path)
    return Path(general_config.anndata_path)


def _resolve_output_adata_path(
    general_config: GeneralConfig,
    cellcharter_config: CellCharterConfig,
) -> Path:
    """Resolve output AnnData path."""
    if cellcharter_config.output_adata_path:
        logging.info("Using configured CellCharter output path: %s", cellcharter_config.output_adata_path)
        return Path(cellcharter_config.output_adata_path)
    return Path(general_config.anndata_path)


def _resolve_sample_key(adata: ad.AnnData, requested_key: str) -> str:
    """Resolve sample/ROI key in adata.obs, with a small fallback set."""
    if requested_key in adata.obs.columns:
        return requested_key

    for fallback in ["ROI", "sample", "Sample", "roi", "ROI_name"]:
        if fallback in adata.obs.columns:
            logging.warning(
                "Sample key '%s' not found. Falling back to '%s'.",
                requested_key,
                fallback,
            )
            return fallback

    raise KeyError(
        f"No sample key found in adata.obs. Requested '{requested_key}' and common fallbacks were missing."
    )


def _ensure_spatial_coordinates(
    adata: ad.AnnData,
    spatial_key: str,
    x_coord_col: str,
    y_coord_col: str,
) -> str:
    """Ensure adata.obsm[spatial_key] exists and contains XY coordinates."""
    if spatial_key in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_key])
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(
                f"adata.obsm['{spatial_key}'] must be shape (n_cells, >=2), got {coords.shape}."
            )
        adata.obsm[spatial_key] = coords[:, :2].astype(np.float32, copy=False)
        return spatial_key

    if x_coord_col in adata.obs.columns and y_coord_col in adata.obs.columns:
        coords = adata.obs[[x_coord_col, y_coord_col]].to_numpy(dtype=np.float32)
        if np.isnan(coords).any():
            raise ValueError(
                f"Found NaN values while creating adata.obsm['{spatial_key}'] from "
                f"obs columns '{x_coord_col}', '{y_coord_col}'."
            )
        adata.obsm[spatial_key] = coords
        logging.info(
            "Created adata.obsm['%s'] from obs columns '%s' and '%s'.",
            spatial_key,
            x_coord_col,
            y_coord_col,
        )
        return spatial_key

    raise KeyError(
        f"Could not find spatial coordinates. Missing adata.obsm['{spatial_key}'] and "
        f"obs columns '{x_coord_col}', '{y_coord_col}'."
    )


def _select_feature_matrix(
    adata: ad.AnnData,
    cellcharter_config: CellCharterConfig,
    *,
    allow_auto_reduced: bool = True,
) -> Tuple[np.ndarray, Optional[str], str]:
    """
    Select the feature matrix used for neighborhood aggregation.

    Returns
    -------
    tuple
        (matrix, obsm_key_if_directly_usable, source_label)
    """
    if cellcharter_config.use_rep is not None:
        rep_key = cellcharter_config.use_rep
        if rep_key not in adata.obsm:
            raise KeyError(f"Configured use_rep '{rep_key}' was not found in adata.obsm.")
        return _to_dense_matrix(adata.obsm[rep_key]), rep_key, f"obsm['{rep_key}']"

    if cellcharter_config.use_layer is not None:
        layer_key = cellcharter_config.use_layer
        if layer_key not in adata.layers:
            raise KeyError(f"Configured use_layer '{layer_key}' was not found in adata.layers.")
        return _to_dense_matrix(adata.layers[layer_key]), None, f"layers['{layer_key}']"

    if allow_auto_reduced:
        for auto_rep in ("X_biobatchnet", "X_pca"):
            if auto_rep in adata.obsm:
                logging.info(
                    "No explicit cellcharter.use_rep/use_layer set; using adata.obsm['%s'].",
                    auto_rep,
                )
                return _to_dense_matrix(adata.obsm[auto_rep]), auto_rep, f"obsm['{auto_rep}']"

    logging.info("No explicit representation found; using adata.X.")
    return _to_dense_matrix(adata.X), None, "X"


def _parse_hidden_layer_sizes(value: Union[str, Sequence[Any]]) -> List[int]:
    """Parse TRVAE hidden layer sizes from list-like or comma-separated text."""
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]

    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if "," in text:
        return [int(v.strip()) for v in text.split(",") if v.strip()]
    if text:
        return [int(text)]
    return [128, 128]


def _zscore_by_sample(features: np.ndarray, sample_ids: np.ndarray) -> np.ndarray:
    """Z-score each feature within each sample."""
    scaled = np.empty(features.shape, dtype=np.float32)

    unique_samples = pd.unique(sample_ids)
    for sample in unique_samples:
        idx = np.where(sample_ids == sample)[0]
        X_sample = features[idx].astype(np.float64, copy=False)
        mean = X_sample.mean(axis=0)
        std = X_sample.std(axis=0)
        std[std == 0] = 1.0
        scaled[idx] = ((X_sample - mean) / std).astype(np.float32)

    return scaled


def _resolve_trvae_condition_key(
    adata: ad.AnnData,
    cellcharter_config: CellCharterConfig,
    sample_key: str,
) -> str:
    """Resolve condition key for TRVAE; create constant label if no key is available."""
    candidates: List[str] = []
    if cellcharter_config.trvae_condition_key:
        candidates.append(cellcharter_config.trvae_condition_key)

    if cellcharter_config.trvae_use_sample_key_fallback and sample_key not in candidates:
        candidates.append(sample_key)

    for fallback in ("dataset", "sample", "ROI", "condition"):
        if fallback not in candidates:
            candidates.append(fallback)

    for key in candidates:
        if key in adata.obs.columns:
            labels = adata.obs[key].astype(str)
            if labels.notna().any():
                adata.obs[key] = pd.Categorical(labels)
                return key

    generated_key = "__cellcharter_condition"
    generated_label = str(cellcharter_config.trvae_constant_condition_label)
    adata.obs[generated_key] = pd.Categorical([generated_label] * adata.n_obs)
    logging.warning(
        "No TRVAE condition key found in adata.obs. Created '%s' with constant value '%s'.",
        generated_key,
        generated_label,
    )
    return generated_key


def _build_trvae_input_adata(
    adata: ad.AnnData,
    cellcharter_config: CellCharterConfig,
    sample_key: str,
) -> Tuple[ad.AnnData, str]:
    """Build an AnnData object for TRVAE training/inference from X or configured layer."""
    if cellcharter_config.use_layer is not None:
        layer_key = cellcharter_config.use_layer
        if layer_key not in adata.layers:
            raise KeyError(f"Configured use_layer '{layer_key}' was not found in adata.layers.")
        matrix = _to_dense_matrix(adata.layers[layer_key])
        source_label = f"layers['{layer_key}']"
    else:
        matrix = _to_dense_matrix(adata.X)
        source_label = "X"

    matrix = matrix.astype(np.float32, copy=False)
    if matrix.shape[0] != adata.n_obs:
        raise ValueError(
            f"TRVAE input rows ({matrix.shape[0]}) must match adata.n_obs ({adata.n_obs})."
        )

    if cellcharter_config.scale_by_sample:
        sample_ids = adata.obs[sample_key].astype(str).to_numpy()
        matrix = _zscore_by_sample(matrix, sample_ids)
        logging.info("Applied sample-wise scaling to TRVAE input using sample key '%s'.", sample_key)

    adata_trvae = ad.AnnData(X=matrix, obs=adata.obs.copy())
    if matrix.shape[1] == adata.n_vars:
        adata_trvae.var_names = adata.var_names.copy()
    else:
        adata_trvae.var_names = pd.Index([f"feature_{i}" for i in range(matrix.shape[1])], dtype=str)

    return adata_trvae, source_label


def _train_trvae_model(model: Any, cellcharter_config: CellCharterConfig) -> None:
    """Train a TRVAE model with configurable kwargs and a safe fallback."""
    kwargs: Dict[str, Any] = {
        "early_stopping": bool(cellcharter_config.trvae_train_early_stopping),
        "enable_progress_bar": bool(cellcharter_config.trvae_train_enable_progress_bar),
    }
    if cellcharter_config.trvae_train_max_epochs is not None:
        kwargs["max_epochs"] = int(cellcharter_config.trvae_train_max_epochs)

    logging.info("Training TRVAE model with args: %s", kwargs)
    try:
        model.train(**kwargs)
    except TypeError:
        logging.warning(
            "TRVAE.train did not accept configured kwargs. Retrying with model.train() default signature."
        )
        model.train()


def _save_trvae_model(model: Any, save_dir: Path) -> Optional[Path]:
    """Save TRVAE model to disk if supported by the installed TRVAE implementation."""
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(save_dir), overwrite=True, save_anndata=False)
        return save_dir
    except TypeError:
        try:
            model.save(str(save_dir))
            return save_dir
        except Exception as exc:  # pragma: no cover - defensive save fallback
            logging.warning("Could not save TRVAE model to %s: %s", save_dir, exc)
    except Exception as exc:  # pragma: no cover - defensive save fallback
        logging.warning("Could not save TRVAE model to %s: %s", save_dir, exc)
    return None


def _compute_trvae_representation(
    adata: ad.AnnData,
    cellcharter_config: CellCharterConfig,
    sample_key: str,
    qc_dir: Path,
) -> Tuple[str, Dict[str, Any]]:
    """Train or load TRVAE and store latent embeddings in adata.obsm."""
    adata_trvae, trvae_source_label = _build_trvae_input_adata(
        adata=adata,
        cellcharter_config=cellcharter_config,
        sample_key=sample_key,
    )
    condition_key = _resolve_trvae_condition_key(
        adata=adata_trvae,
        cellcharter_config=cellcharter_config,
        sample_key=sample_key,
    )

    load_path: Optional[Path] = None
    if cellcharter_config.trvae_load_path:
        load_path = Path(cellcharter_config.trvae_load_path)
        if not load_path.is_absolute():
            load_path = Path.cwd() / load_path
        if not load_path.exists():
            logging.warning(
                "Configured trvae_load_path does not exist: %s. Training a new TRVAE model.",
                load_path,
            )
            load_path = None

    if load_path is not None:
        logging.info("Loading pretrained TRVAE model from %s", load_path)
        try:
            model = cc.tl.TRVAE.load(
                str(load_path),
                adata_trvae,
                map_location=cellcharter_config.trvae_map_location,
            )
        except ImportError as exc:
            raise ImportError(
                "TRVAE requires scArches. Install it in the cellcharter environment "
                "(for example: pip install scarches)."
            ) from exc
        if cellcharter_config.trvae_train:
            _train_trvae_model(model, cellcharter_config)
    else:
        if not cellcharter_config.trvae_train:
            raise ValueError(
                "cellcharter.trvae_train=False but no valid trvae_load_path was provided."
            )
        hidden_layer_sizes = _parse_hidden_layer_sizes(cellcharter_config.trvae_hidden_layer_sizes)
        logging.info(
            "Initializing new TRVAE model (latent_dim=%d, hidden_layer_sizes=%s, condition_key=%s).",
            int(cellcharter_config.trvae_latent_dim),
            hidden_layer_sizes,
            condition_key,
        )
        try:
            model = cc.tl.TRVAE(
                adata_trvae,
                condition_key=condition_key,
                hidden_layer_sizes=hidden_layer_sizes,
                latent_dim=int(cellcharter_config.trvae_latent_dim),
                dr_rate=float(cellcharter_config.trvae_dr_rate),
                use_mmd=bool(cellcharter_config.trvae_use_mmd),
                mmd_on=cellcharter_config.trvae_mmd_on,
                mmd_boundary=cellcharter_config.trvae_mmd_boundary,
                recon_loss=cellcharter_config.trvae_recon_loss,
                beta=float(cellcharter_config.trvae_beta),
                use_bn=bool(cellcharter_config.trvae_use_bn),
                use_ln=bool(cellcharter_config.trvae_use_ln),
            )
        except ImportError as exc:
            raise ImportError(
                "TRVAE requires scArches. Install it in the cellcharter environment "
                "(for example: pip install scarches)."
            ) from exc
        _train_trvae_model(model, cellcharter_config)

    latent = model.get_latent(adata_trvae.X, adata_trvae.obs[condition_key])
    latent = np.asarray(latent, dtype=np.float32)
    if latent.shape[0] != adata.n_obs:
        raise ValueError(
            f"TRVAE latent rows ({latent.shape[0]}) must match adata.n_obs ({adata.n_obs})."
        )

    adata.obsm[cellcharter_config.trvae_latent_key] = latent
    logging.info(
        "Stored TRVAE latent embeddings in adata.obsm['%s'] (shape=%s).",
        cellcharter_config.trvae_latent_key,
        latent.shape,
    )

    saved_to: Optional[Path] = None
    if cellcharter_config.trvae_save_path:
        trvae_save_dir = Path(cellcharter_config.trvae_save_path)
        if not trvae_save_dir.is_absolute():
            trvae_save_dir = qc_dir / trvae_save_dir
        saved_to = _save_trvae_model(model, trvae_save_dir)

    details: Dict[str, Any] = {
        "enabled": True,
        "latent_key": cellcharter_config.trvae_latent_key,
        "condition_key": condition_key,
        "input_source": trvae_source_label,
        "loaded_from": str(load_path) if load_path is not None else None,
        "saved_to": str(saved_to) if saved_to is not None else None,
    }
    return cellcharter_config.trvae_latent_key, details


def _category_sort_key(value: Any) -> Tuple[int, Any]:
    """Sort numeric-like cluster labels numerically, others lexicographically."""
    text = str(value)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def _save_cluster_tables(
    adata: ad.AnnData,
    sample_key: str,
    cluster_key: str,
    qc_dir: Path,
) -> None:
    """Save global and per-sample cluster count tables."""
    global_counts = (
        adata.obs[cluster_key]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis(cluster_key)
        .reset_index(name="n_cells")
    )
    global_counts["fraction"] = global_counts["n_cells"] / max(1, adata.n_obs)
    global_counts.to_csv(qc_dir / "cluster_counts_global.csv", index=False)

    if sample_key in adata.obs.columns:
        per_sample = (
            adata.obs.groupby([sample_key, cluster_key], dropna=False)
            .size()
            .reset_index(name="n_cells")
        )
        totals = per_sample.groupby(sample_key)["n_cells"].transform("sum")
        per_sample["fraction_within_sample"] = per_sample["n_cells"] / totals.replace(0, np.nan)
        per_sample.to_csv(qc_dir / "cluster_counts_by_sample.csv", index=False)


def _save_spatial_cluster_plots(
    adata: ad.AnnData,
    sample_key: str,
    spatial_key: str,
    cluster_key: str,
    point_size: float,
    max_samples: int,
    qc_dir: Path,
) -> None:
    """Save per-sample spatial scatter plots colored by CellCharter clusters."""
    if sample_key not in adata.obs.columns:
        logging.warning("Cannot save spatial QC plots: sample key '%s' missing.", sample_key)
        return

    samples = (
        adata.obs[sample_key]
        .astype(str)
        .value_counts()
        .head(max(1, int(max_samples)))
        .index.tolist()
    )

    labels_all = adata.obs[cluster_key].astype(str)
    categories = sorted(pd.unique(labels_all), key=_category_sort_key)
    if len(categories) <= 20:
        cmap = plt.get_cmap("tab20", len(categories))
    else:
        cmap = plt.get_cmap("gist_ncar", len(categories))
    color_map = {cat: cmap(i) for i, cat in enumerate(categories)}

    coords_all = np.asarray(adata.obsm[spatial_key], dtype=np.float32)

    for sample in samples:
        mask = adata.obs[sample_key].astype(str).to_numpy() == sample
        if not np.any(mask):
            continue

        coords = coords_all[mask, :2]
        labels = adata.obs.loc[mask, cluster_key].astype(str).to_numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        for cat in categories:
            cat_mask = labels == cat
            if np.any(cat_mask):
                ax.scatter(
                    coords[cat_mask, 0],
                    coords[cat_mask, 1],
                    s=float(point_size),
                    c=[color_map[cat]],
                    label=cat,
                    alpha=0.9,
                    linewidths=0,
                )

        ax.set_title(f"{sample_key}: {sample} ({cluster_key})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        if len(categories) <= 30:
            ax.legend(
                title=cluster_key,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                fontsize=8,
                frameon=False,
            )

        fig.tight_layout()
        sample_slug = cleanstring(sample)
        fig.savefig(
            qc_dir / f"spatial_clusters_{sample_slug}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)


def _save_enrichment_outputs(
    adata: ad.AnnData,
    cluster_key: str,
    label_key: str,
    save_heatmap: bool,
    qc_dir: Path,
) -> None:
    """Save CellCharter enrichment tables and optional heatmap figure."""
    key = f"{cluster_key}_{label_key}_enrichment"
    result = adata.uns.get(key)
    if result is None:
        alt_key = f"{cluster_key}_{label_key}_nhood_enrichment"
        result = adata.uns.get(alt_key)
        if result is not None:
            key = alt_key

    if result is None:
        logging.warning(
            "Enrichment result not found in adata.uns (expected '%s' or fallback).", key
        )
        return

    enrichment = result.get("enrichment") if isinstance(result, dict) else None
    if enrichment is None:
        logging.warning("Enrichment result '%s' is missing the 'enrichment' matrix.", key)
        return

    enrich_df = enrichment if isinstance(enrichment, pd.DataFrame) else pd.DataFrame(enrichment)
    enrich_df.to_csv(qc_dir / "enrichment_matrix.csv")

    if isinstance(result, dict) and "pvalue" in result and result["pvalue"] is not None:
        pvalue_df = result["pvalue"]
        if not isinstance(pvalue_df, pd.DataFrame):
            pvalue_df = pd.DataFrame(pvalue_df)
        pvalue_df.to_csv(qc_dir / "enrichment_pvalues.csv")

    if not save_heatmap or enrich_df.empty:
        return

    matrix = enrich_df.to_numpy(dtype=float)
    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    if vmax == 0:
        vmax = 1.0

    width = max(6.0, 0.35 * enrich_df.shape[1] + 2.5)
    height = max(4.0, 0.35 * enrich_df.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(enrich_df.shape[1]))
    ax.set_xticklabels(enrich_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(enrich_df.shape[0]))
    ax.set_yticklabels(enrich_df.index, fontsize=8)
    ax.set_title(f"CellCharter enrichment: {cluster_key} vs {label_key}")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log2 enrichment")
    fig.tight_layout()
    fig.savefig(qc_dir / "enrichment_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_cellcharter_enrichment_plot(
    adata: ad.AnnData,
    cluster_key: str,
    label_key: str,
    qc_dir: Path,
) -> None:
    """Save CellCharter's native enrichment plot via cc.pl.enrichment."""
    out_path = qc_dir / "enrichment_cellcharter_plot.png"
    try:
        cc.pl.enrichment(
            adata,
            group_key=cluster_key,
            label_key=label_key,
            figsize=(8, 6),
            save=str(out_path),
        )
    except Exception as exc:
        logging.warning("Could not generate CellCharter enrichment plot: %s", exc)
    finally:
        plt.close("all")


def _ensure_categorical_obs(adata: ad.AnnData, key: str) -> bool:
    """Ensure adata.obs[key] exists and is categorical."""
    if key not in adata.obs.columns:
        return False

    series = adata.obs[key]
    if isinstance(series.dtype, pd.CategoricalDtype):
        adata.obs[key] = series.cat.remove_unused_categories()
    else:
        adata.obs[key] = pd.Categorical(series.astype(str))
    return True


def _to_dataframe(matrix: Any) -> pd.DataFrame:
    """Coerce matrix-like output to DataFrame."""
    if isinstance(matrix, pd.DataFrame):
        return matrix
    arr = np.asarray(matrix)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    return pd.DataFrame(arr)


def _sanitize_h5ad_token(value: Any) -> Any:
    """Replace HDF5 path separators in string labels used in AnnData uns payloads."""
    if isinstance(value, str):
        return value.replace("/", "_")
    return value


def _sanitize_index_for_h5ad(index: pd.Index) -> Tuple[pd.Index, int]:
    """Sanitize HDF5-unsafe '/' characters in pandas Index and name."""
    changes = 0

    if isinstance(index, pd.MultiIndex):
        sanitized_tuples = []
        for values in index.to_list():
            sanitized = tuple(_sanitize_h5ad_token(v) for v in values)
            changes += sum(1 for old, new in zip(values, sanitized) if old != new)
            sanitized_tuples.append(sanitized)
        names = [_sanitize_h5ad_token(name) for name in index.names]
        changes += sum(1 for old, new in zip(index.names, names) if old != new)
        return pd.MultiIndex.from_tuples(sanitized_tuples, names=names), changes

    values = index.to_list()
    sanitized_values = [_sanitize_h5ad_token(v) for v in values]
    changes += sum(1 for old, new in zip(values, sanitized_values) if old != new)
    name = _sanitize_h5ad_token(index.name)
    if name != index.name:
        changes += 1
    return pd.Index(sanitized_values, name=name), changes


def _sanitize_dataframe_for_h5ad(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Sanitize DataFrame row/column labels for safe H5AD serialization."""
    new_index, index_changes = _sanitize_index_for_h5ad(df.index)
    new_columns, column_changes = _sanitize_index_for_h5ad(df.columns)
    total = index_changes + column_changes
    if total == 0:
        return df, 0
    out = df.copy()
    out.index = new_index
    out.columns = new_columns
    return out, total


def _sanitize_series_for_h5ad(series: pd.Series) -> Tuple[pd.Series, int]:
    """Sanitize Series index/name for safe H5AD serialization."""
    new_index, index_changes = _sanitize_index_for_h5ad(series.index)
    new_name = _sanitize_h5ad_token(series.name)
    name_changes = 1 if new_name != series.name else 0
    total = index_changes + name_changes
    if total == 0:
        return series, 0
    out = series.copy()
    out.index = new_index
    out.name = new_name
    return out, total


def _sanitize_object_for_h5ad(obj: Any) -> Tuple[Any, int]:
    """Recursively sanitize nested objects in adata.uns for h5ad compatibility."""
    if isinstance(obj, pd.DataFrame):
        return _sanitize_dataframe_for_h5ad(obj)

    if isinstance(obj, pd.Series):
        return _sanitize_series_for_h5ad(obj)

    if isinstance(obj, dict):
        changes = 0
        sanitized: Dict[Any, Any] = {}
        for key, value in obj.items():
            new_key = _sanitize_h5ad_token(key)
            if new_key != key:
                changes += 1
            if isinstance(new_key, str):
                base_key = new_key
                suffix = 1
                while new_key in sanitized:
                    new_key = f"{base_key}_{suffix}"
                    suffix += 1
                    changes += 1
            new_value, new_changes = _sanitize_object_for_h5ad(value)
            sanitized[new_key] = new_value
            changes += new_changes
        return sanitized, changes

    if isinstance(obj, list):
        changes = 0
        out = []
        for item in obj:
            new_item, new_changes = _sanitize_object_for_h5ad(item)
            out.append(new_item)
            changes += new_changes
        return out, changes

    if isinstance(obj, tuple):
        changes = 0
        out = []
        for item in obj:
            new_item, new_changes = _sanitize_object_for_h5ad(item)
            out.append(new_item)
            changes += new_changes
        return tuple(out), changes

    return obj, 0


def _sanitize_uns_for_h5ad_write(adata: ad.AnnData) -> int:
    """Sanitize adata.uns labels/keys that are invalid in h5 path-like storage."""
    sanitized, changes = _sanitize_object_for_h5ad(dict(adata.uns))
    if changes > 0:
        adata.uns.clear()
        adata.uns.update(sanitized)
    return changes


def _save_matrix_heatmap(
    matrix_df: pd.DataFrame,
    out_path: Path,
    title: str,
    cbar_label: str,
    cmap: str = "coolwarm",
    symmetric: bool = True,
) -> None:
    """Save a compact heatmap from a DataFrame matrix."""
    if matrix_df.empty:
        return

    matrix = matrix_df.to_numpy(dtype=float)
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return

    if symmetric:
        vmax = float(np.max(np.abs(finite)))
        if vmax == 0:
            vmax = 1.0
        vmin = -vmax
    else:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmin == vmax:
            vmax = vmin + 1.0

    width = max(6.0, 0.35 * matrix_df.shape[1] + 2.5)
    height = max(4.0, 0.35 * matrix_df.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(matrix_df.shape[1]))
    ax.set_xticklabels(matrix_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(matrix_df.shape[0]))
    ax.set_yticklabels(matrix_df.index, fontsize=8)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _resolve_figsize(value: Any, fallback: Tuple[float, float]) -> Tuple[float, float]:
    """Parse a 2-number figsize configuration with safe fallback."""
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return float(value[0]), float(value[1])
    except Exception:
        pass
    return fallback


def _run_nhood_enrichment(
    adata: ad.AnnData,
    cellcharter_config: CellCharterConfig,
    qc_dir: Path,
) -> Dict[str, Any]:
    """Run CellCharter neighborhood enrichment and save outputs."""
    details: Dict[str, Any] = {"enabled": True, "ran": False}
    cluster_key = cellcharter_config.cluster_key

    if not _ensure_categorical_obs(adata, cluster_key):
        logging.warning(
            "Skipping nhood enrichment: cluster key '%s' is missing in adata.obs.",
            cluster_key,
        )
        return details

    out_dir = qc_dir / "nhood_enrichment"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Running CellCharter nhood enrichment (cluster_key=%s, pvalues=%s, n_perms=%d).",
        cluster_key,
        cellcharter_config.nhood_with_pvalues,
        int(cellcharter_config.nhood_n_perms),
    )
    try:
        cc.gr.nhood_enrichment(
            adata,
            cluster_key=cluster_key,
            connectivity_key=cellcharter_config.nhood_connectivity_key,
            log_fold_change=bool(cellcharter_config.nhood_log_fold_change),
            only_inter=bool(cellcharter_config.nhood_only_inter),
            symmetric=bool(cellcharter_config.nhood_symmetric),
            pvalues=bool(cellcharter_config.nhood_with_pvalues),
            n_perms=int(cellcharter_config.nhood_n_perms),
            n_jobs=int(cellcharter_config.nhood_n_jobs),
            batch_size=int(cellcharter_config.nhood_batch_size),
            observed_expected=bool(cellcharter_config.nhood_observed_expected),
        )
    except Exception as exc:
        logging.warning("CellCharter nhood enrichment failed: %s", exc)
        return details

    uns_key = f"{cluster_key}_nhood_enrichment"
    result = adata.uns.get(uns_key)
    if not isinstance(result, dict):
        logging.warning("Neighborhood enrichment output '%s' not found or invalid.", uns_key)
        return details

    for key in ("enrichment", "pvalue", "observed", "expected"):
        if key not in result or result[key] is None:
            continue
        table = _to_dataframe(result[key])
        table.to_csv(out_dir / f"{key}.csv")
        if key == "enrichment" and bool(cellcharter_config.save_enrichment_heatmap):
            _save_matrix_heatmap(
                table,
                out_path=out_dir / "enrichment_heatmap.png",
                title=f"Neighborhood enrichment ({cluster_key})",
                cbar_label="enrichment",
                cmap="coolwarm",
                symmetric=True,
            )

    if isinstance(result.get("params"), dict):
        pd.Series(result["params"], name="value").to_csv(out_dir / "params.csv")

    if cellcharter_config.save_nhood_enrichment_plot:
        nhood_figsize = _resolve_figsize(
            cellcharter_config.nhood_plot_figsize,
            fallback=(8.0, 6.0),
        )
        try:
            cc.pl.nhood_enrichment(
                adata,
                cluster_key=cluster_key,
                figsize=nhood_figsize,
                significance=cellcharter_config.nhood_enrichment_significance,
                save=str(out_dir / "nhood_enrichment_cellcharter_plot.png"),
            )
        except Exception as exc:
            logging.warning("Could not generate CellCharter nhood enrichment plot: %s", exc)
        finally:
            plt.close("all")

    details.update({"ran": True, "uns_key": uns_key, "output_dir": str(out_dir)})
    return details


def _run_diff_nhood_enrichment(
    adata: ad.AnnData,
    general_config: GeneralConfig,
    cellcharter_config: CellCharterConfig,
    sample_key: str,
    qc_dir: Path,
) -> Dict[str, Any]:
    """Run differential neighborhood enrichment and save outputs."""
    details: Dict[str, Any] = {"enabled": True, "ran": False}
    cluster_key = cellcharter_config.cluster_key
    condition_key = coalesce_config_text(
        cellcharter_config.diff_nhood_condition_key,
        general_config.groupby_obs,
    )

    if not condition_key:
        for fallback in ("condition", "dataset", "group", "sample_type"):
            if fallback in adata.obs.columns:
                condition_key = fallback
                logging.info(
                    "No diff_nhood_condition_key set; falling back to '%s' from adata.obs.",
                    fallback,
                )
                break
        if not condition_key:
            logging.warning(
                "Skipping differential nhood enrichment: configure cellcharter.diff_nhood_condition_key "
                "or general.groupby_obs."
            )
            return details

    if not _ensure_categorical_obs(adata, cluster_key):
        logging.warning(
            "Skipping differential nhood enrichment: cluster key '%s' is missing in adata.obs.",
            cluster_key,
        )
        return details

    if not _ensure_categorical_obs(adata, condition_key):
        logging.warning(
            "Skipping differential nhood enrichment: condition key '%s' is missing in adata.obs.",
            condition_key,
        )
        return details

    condition_groups = coalesce_config_list(
        cellcharter_config.diff_nhood_condition_groups,
        general_config.groupby_obs_primary_pairwise,
        general_config.groupby_obs_groups,
    )
    if condition_groups:
        available = set(adata.obs[condition_key].cat.categories.tolist())
        condition_groups = [str(g) for g in condition_groups if str(g) in available]
        if len(condition_groups) < 2:
            logging.warning(
                "Skipping differential nhood enrichment: fewer than 2 valid condition groups in '%s'.",
                condition_key,
            )
            return details

    library_key = cellcharter_config.diff_nhood_library_key or sample_key
    if bool(cellcharter_config.diff_nhood_with_pvalues) and library_key not in adata.obs.columns:
        logging.warning(
            "P-value estimation requested but library key '%s' is missing. Falling back to sample key '%s'.",
            library_key,
            sample_key,
        )
        library_key = sample_key

    out_dir = qc_dir / "diff_nhood_enrichment"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Running differential nhood enrichment (%s by %s, pvalues=%s, n_perms=%d).",
        cluster_key,
        condition_key,
        cellcharter_config.diff_nhood_with_pvalues,
        int(cellcharter_config.diff_nhood_n_perms),
    )
    kwargs: Dict[str, Any] = {
        "log_fold_change": bool(cellcharter_config.diff_nhood_log_fold_change),
        "only_inter": bool(cellcharter_config.diff_nhood_only_inter),
        "symmetric": bool(cellcharter_config.diff_nhood_symmetric),
    }
    n_jobs = cellcharter_config.diff_nhood_n_jobs
    try:
        cc.gr.diff_nhood_enrichment(
            adata,
            cluster_key=cluster_key,
            condition_key=condition_key,
            condition_groups=condition_groups,
            connectivity_key=cellcharter_config.diff_nhood_connectivity_key,
            pvalues=bool(cellcharter_config.diff_nhood_with_pvalues),
            library_key=library_key,
            n_perms=int(cellcharter_config.diff_nhood_n_perms),
            n_jobs=int(n_jobs) if n_jobs is not None else None,
            **kwargs,
        )
    except Exception as exc:
        logging.warning("CellCharter differential nhood enrichment failed: %s", exc)
        return details

    uns_key = f"{cluster_key}_{condition_key}_diff_nhood_enrichment"
    result = adata.uns.get(uns_key)
    if not isinstance(result, dict) or not result:
        logging.warning("Differential nhood enrichment output '%s' not found or empty.", uns_key)
        return details

    long_tables: List[pd.DataFrame] = []
    for pair_key, pair_result in result.items():
        if not isinstance(pair_result, dict):
            continue
        safe_pair = cleanstring(str(pair_key))
        for metric_key in ("enrichment", "pvalue"):
            matrix = pair_result.get(metric_key)
            if matrix is None:
                continue
            table = _to_dataframe(matrix)
            table.to_csv(out_dir / f"{safe_pair}_{metric_key}.csv")
            if metric_key == "enrichment":
                long_df = (
                    table.stack(dropna=False)
                    .rename("enrichment")
                    .reset_index()
                    .rename(columns={"level_0": "source_cluster", "level_1": "target_cluster"})
                )
                long_df.insert(0, "condition_pair", str(pair_key))
                long_tables.append(long_df)

    if long_tables:
        pd.concat(long_tables, ignore_index=True).to_csv(
            out_dir / "diff_nhood_enrichment_long.csv",
            index=False,
        )

    if cellcharter_config.save_diff_nhood_enrichment_plot:
        try:
            cc.pl.diff_nhood_enrichment(
                adata,
                cluster_key=cluster_key,
                condition_key=condition_key,
                condition_groups=condition_groups,
                ncols=int(cellcharter_config.diff_nhood_plot_ncols),
                cmap="PRGn_r",
                save=str(out_dir / "diff_nhood_enrichment_cellcharter_plot.png"),
            )
        except Exception as exc:
            logging.warning("Could not generate CellCharter differential nhood plot: %s", exc)
        finally:
            plt.close("all")

    details.update(
        {
            "ran": True,
            "uns_key": uns_key,
            "condition_key": condition_key,
            "condition_groups": condition_groups,
            "output_dir": str(out_dir),
        }
    )
    return details


def _run_shape_characterisation(
    adata: ad.AnnData,
    general_config: GeneralConfig,
    cellcharter_config: CellCharterConfig,
    sample_key: str,
    qc_dir: Path,
) -> Dict[str, Any]:
    """Run CellCharter shape characterisation pipeline and save outputs."""
    details: Dict[str, Any] = {"enabled": True, "ran": False}
    cluster_key = cellcharter_config.cluster_key
    component_key = cellcharter_config.shape_component_key
    component_cluster_key = (
        cellcharter_config.shape_component_cluster_key or cluster_key
    )

    if component_cluster_key not in adata.obs.columns:
        logging.warning(
            "Skipping shape characterisation: component cluster key '%s' missing in adata.obs.",
            component_cluster_key,
        )
        return details

    _ensure_categorical_obs(adata, component_cluster_key)

    out_dir = qc_dir / "shape_characterisation"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Running connected components (cluster_key=%s, min_cells=%d, out_key=%s).",
        component_cluster_key,
        int(cellcharter_config.shape_min_cells),
        component_key,
    )
    try:
        cc.gr.connected_components(
            adata,
            cluster_key=component_cluster_key,
            min_cells=int(cellcharter_config.shape_min_cells),
            connectivity_key=cellcharter_config.shape_connectivity_key,
            out_key=component_key,
        )
    except Exception as exc:
        logging.warning("Connected components failed: %s", exc)
        return details

    cols = [component_key]
    for opt_col in (sample_key, cluster_key, component_cluster_key):
        if opt_col in adata.obs.columns and opt_col not in cols:
            cols.append(opt_col)
    component_table = adata.obs[cols].copy()
    component_table.to_csv(out_dir / "components_per_cell.csv")

    if sample_key in component_table.columns:
        counts = (
            component_table.dropna(subset=[component_key])
            .groupby([sample_key, component_key], dropna=False)
            .size()
            .reset_index(name="n_cells")
        )
        counts.to_csv(out_dir / "components_by_sample.csv", index=False)

    logging.info(
        "Computing shape boundaries (component_key=%s, min_hole_area_ratio=%.3f, alpha_start=%d).",
        component_key,
        float(cellcharter_config.shape_min_hole_area_ratio),
        int(cellcharter_config.shape_alpha_start),
    )
    try:
        cc.tl.boundaries(
            adata,
            cluster_key=component_key,
            min_hole_area_ratio=float(cellcharter_config.shape_min_hole_area_ratio),
            alpha_start=int(cellcharter_config.shape_alpha_start),
        )
    except Exception as exc:
        logging.warning("Shape boundary computation failed: %s", exc)
        return details

    shape_uns_key = f"shape_{component_key}"
    shape_data = adata.uns.get(shape_uns_key, {})
    boundaries = shape_data.get("boundary") if isinstance(shape_data, dict) else None
    if isinstance(boundaries, dict):
        boundary_rows = []
        for comp_id, poly in boundaries.items():
            try:
                boundary_rows.append(
                    {
                        component_key: str(comp_id),
                        "boundary_area": float(poly.area),
                        "boundary_perimeter": float(poly.length),
                        "boundary_num_vertices": int(len(poly.exterior.coords)),
                    }
                )
            except Exception:
                continue
        if boundary_rows:
            pd.DataFrame(boundary_rows).to_csv(
                out_dir / "boundary_summary.csv",
                index=False,
            )

    computed_metrics: List[str] = []
    if cellcharter_config.shape_compute_linearity:
        try:
            if hasattr(cc.tl, "linearity_metric"):
                cc.tl.linearity_metric(
                    adata,
                    cluster_key=component_key,
                    out_key=cellcharter_config.shape_linearity_key,
                    height=int(cellcharter_config.shape_linearity_height),
                    min_ratio=float(cellcharter_config.shape_linearity_min_ratio),
                )
            else:
                cc.tl.linearity(
                    adata,
                    cluster_key=component_key,
                    out_key=cellcharter_config.shape_linearity_key,
                    height=int(cellcharter_config.shape_linearity_height),
                    min_ratio=float(cellcharter_config.shape_linearity_min_ratio),
                )
            computed_metrics.append(cellcharter_config.shape_linearity_key)
        except Exception as exc:
            logging.warning("Linearity metric failed: %s", exc)

    if cellcharter_config.shape_compute_curl:
        try:
            if hasattr(cc.tl, "curl_metric"):
                cc.tl.curl_metric(
                    adata,
                    cluster_key=component_key,
                    out_key=cellcharter_config.shape_curl_key,
                )
            else:
                cc.tl.curl(
                    adata,
                    cluster_key=component_key,
                    out_key=cellcharter_config.shape_curl_key,
                )
            computed_metrics.append(cellcharter_config.shape_curl_key)
        except Exception as exc:
            logging.warning("Curl metric failed: %s", exc)

    shape_data = adata.uns.get(shape_uns_key, {})
    metric_cols: Dict[str, pd.Series] = {}
    for metric_key in computed_metrics:
        metric_dict = shape_data.get(metric_key) if isinstance(shape_data, dict) else None
        if isinstance(metric_dict, dict) and metric_dict:
            metric_series = pd.Series(metric_dict, name=metric_key, dtype=float)
            metric_series.index = metric_series.index.astype(str)
            metric_cols[metric_key] = metric_series

    metrics_df = None
    if metric_cols:
        metrics_df = pd.concat(metric_cols.values(), axis=1)
        metrics_df.index.name = component_key
        metrics_df.to_csv(out_dir / "shape_metrics_by_component.csv")

    if metrics_df is not None:
        metadata_cols = [component_key]
        for opt_col in (sample_key, component_cluster_key, cluster_key):
            if opt_col in adata.obs.columns and opt_col not in metadata_cols:
                metadata_cols.append(opt_col)
        condition_key = (
            coalesce_config_text(
                cellcharter_config.shape_metrics_condition_key,
                cellcharter_config.diff_nhood_condition_key,
                general_config.groupby_obs,
            )
        )
        if condition_key and condition_key in adata.obs.columns:
            metadata_cols.append(condition_key)

        component_meta = (
            adata.obs[metadata_cols]
            .dropna(subset=[component_key])
            .copy()
        )
        component_meta[component_key] = component_meta[component_key].astype(str)
        component_meta = component_meta.drop_duplicates(subset=[component_key]).set_index(component_key)
        component_meta = component_meta[~component_meta.index.duplicated(keep="first")]
        component_meta.join(metrics_df, how="left").to_csv(
            out_dir / "shape_metrics_with_metadata.csv"
        )

    if cellcharter_config.shape_plot_metrics and computed_metrics:
        condition_key = (
            coalesce_config_text(
                cellcharter_config.shape_metrics_condition_key,
                cellcharter_config.diff_nhood_condition_key,
                general_config.groupby_obs,
            )
        )
        cluster_plot_key = cellcharter_config.shape_metrics_cluster_key
        if cluster_plot_key is None and component_cluster_key in adata.obs.columns:
            cluster_plot_key = component_cluster_key

        if condition_key and condition_key in adata.obs.columns:
            _ensure_categorical_obs(adata, condition_key)
        else:
            condition_key = None

        if cluster_plot_key and cluster_plot_key in adata.obs.columns:
            _ensure_categorical_obs(adata, cluster_plot_key)
        else:
            cluster_plot_key = None

        if condition_key is not None or cluster_plot_key is not None:
            try:
                cc.pl.shape_metrics(
                    adata,
                    condition_key=condition_key,
                    condition_groups=cellcharter_config.shape_metrics_condition_groups,
                    cluster_key=cluster_plot_key,
                    cluster_groups=cellcharter_config.shape_metrics_cluster_groups,
                    component_key=component_key,
                    metrics=computed_metrics,
                    ncols=int(cellcharter_config.shape_metrics_ncols),
                    save=out_dir / "shape_metrics_cellcharter_plot.png",
                )
            except Exception as exc:
                logging.warning("Could not generate shape metrics plot: %s", exc)
            finally:
                plt.close("all")

    details.update(
        {
            "ran": True,
            "component_key": component_key,
            "shape_uns_key": shape_uns_key,
            "metrics": computed_metrics,
            "output_dir": str(out_dir),
        }
    )
    return details


def _save_roi_cluster_masks(
    adata: ad.AnnData,
    sample_key: str,
    cluster_key: str,
    masks_folder: str,
    qc_dir: Path,
) -> None:
    """Save one mask-style cluster plot per ROI using plotting.obs_to_mask."""
    if sbt_plotting is None:
        logging.warning("plotting module unavailable; skipping ROI obs_to_mask plots.")
        return

    if sample_key not in adata.obs.columns:
        logging.warning(
            "Sample key '%s' missing in adata.obs; skipping ROI obs_to_mask plots.",
            sample_key,
        )
        return

    roi_dir = qc_dir / "ROI_cluster_masks"
    roi_dir.mkdir(parents=True, exist_ok=True)
    rois = sorted(pd.unique(adata.obs[sample_key].astype(str)))

    for roi in rois:
        save_path = roi_dir / f"{cleanstring(roi)}.png"
        try:
            sbt_plotting.obs_to_mask(
                adata=adata,
                roi=str(roi),
                roi_obs=sample_key,
                cat_obs=cluster_key,
                masks_folder=masks_folder,
                save_path=str(save_path),
                background_color="white",
                separator_color="black",
            )
        except Exception as exc:
            logging.warning(
                "Could not create ROI mask plot for ROI '%s' using cluster key '%s': %s",
                roi,
                cluster_key,
                exc,
            )


def run_cellcharter_neighborhoods(
    general_config: GeneralConfig,
    cellcharter_config: CellCharterConfig,
) -> Path:
    """Run CellCharter neighborhood analysis and return output AnnData path."""
    stage_name = "CellCharter"
    cellcharter_config.sample_key = coalesce_config_text(
        cellcharter_config.sample_key,
        general_config.roi_obs,
        default="ROI",
    )
    cellcharter_config.spatial_key = coalesce_config_text(
        cellcharter_config.spatial_key,
        general_config.spatial_key,
        default="spatial",
    )
    cellcharter_config.x_coord_col = coalesce_config_text(
        cellcharter_config.x_coord_col,
        general_config.x_coord_obs,
        default="X_loc",
    )
    cellcharter_config.y_coord_col = coalesce_config_text(
        cellcharter_config.y_coord_col,
        general_config.y_coord_obs,
        default="Y_loc",
    )
    if cellcharter_config.diff_nhood_condition_key is None:
        cellcharter_config.diff_nhood_condition_key = coalesce_config_text(
            general_config.groupby_obs,
        )
    if cellcharter_config.diff_nhood_condition_groups is None:
        cellcharter_config.diff_nhood_condition_groups = coalesce_config_list(
            general_config.groupby_obs_primary_pairwise,
            general_config.groupby_obs_groups,
        )

    input_path = _resolve_input_adata_path(general_config, cellcharter_config)
    output_path = _resolve_output_adata_path(general_config, cellcharter_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    qc_dir = Path(general_config.qc_folder) / cellcharter_config.qc_output_subdir
    qc_dir.mkdir(parents=True, exist_ok=True)

    adata, _, skip_stage, _ = load_pipeline_anndata(
        general_config=general_config,
        stage_name=stage_name,
        stage_config=cellcharter_config,
        override_path=str(input_path),
    )
    if skip_stage:
        logging.info("Skipping CellCharter stage based on AnnData stage policy.")
        return output_path
    if adata is None:
        raise FileNotFoundError(f"AnnData could not be loaded for CellCharter stage: {input_path}")
    logging.info("Loaded AnnData: %d cells x %d features", adata.n_obs, adata.n_vars)

    sample_key_requested = str(cellcharter_config.sample_key)
    spatial_key_requested = str(cellcharter_config.spatial_key)
    x_coord_col = str(cellcharter_config.x_coord_col)
    y_coord_col = str(cellcharter_config.y_coord_col)
    population_obs_primary = coalesce_config_text(general_config.population_obs_primary)

    sample_key = _resolve_sample_key(adata, sample_key_requested)
    spatial_key = _ensure_spatial_coordinates(
        adata,
        spatial_key=spatial_key_requested,
        x_coord_col=x_coord_col,
        y_coord_col=y_coord_col,
    )

    trvae_details: Dict[str, Any] = {"enabled": False}
    if cellcharter_config.use_trvae:
        logging.info("TRVAE mode enabled (cellcharter.use_trvae=True).")
        aggregation_rep, trvae_details = _compute_trvae_representation(
            adata=adata,
            cellcharter_config=cellcharter_config,
            sample_key=sample_key,
            qc_dir=qc_dir,
        )
        logging.info("Using TRVAE representation adata.obsm['%s'] for neighborhood aggregation.", aggregation_rep)
    else:
        feature_matrix, feature_rep_key, source_label = _select_feature_matrix(
            adata,
            cellcharter_config,
            allow_auto_reduced=True,
        )
        feature_matrix = feature_matrix.astype(np.float32, copy=False)
        if feature_matrix.shape[0] != adata.n_obs:
            raise ValueError(
                f"Feature matrix rows ({feature_matrix.shape[0]}) must match adata.n_obs ({adata.n_obs})."
            )

        if cellcharter_config.scale_by_sample:
            logging.info("Applying sample-wise feature scaling using '%s'.", sample_key)
            sample_ids = adata.obs[sample_key].astype(str).to_numpy()
            scaled = _zscore_by_sample(feature_matrix, sample_ids)
            adata.obsm[cellcharter_config.scaled_rep_key] = scaled
            aggregation_rep = cellcharter_config.scaled_rep_key
            logging.info(
                "Stored scaled features in adata.obsm['%s'] (source: %s).",
                cellcharter_config.scaled_rep_key,
                source_label,
            )
        else:
            if feature_rep_key is not None:
                aggregation_rep = feature_rep_key
                logging.info("Using feature representation %s without additional scaling.", source_label)
            elif source_label == "X":
                aggregation_rep = None
                logging.info("Using adata.X for neighborhood aggregation.")
            else:
                adata.obsm[cellcharter_config.scaled_rep_key] = feature_matrix
                aggregation_rep = cellcharter_config.scaled_rep_key
                logging.info(
                    "Stored features from %s in adata.obsm['%s'] for aggregation.",
                    source_label,
                    cellcharter_config.scaled_rep_key,
                )

    logging.info("Building spatial graph with Squidpy (delaunay=%s).", cellcharter_config.delaunay)
    sq.gr.spatial_neighbors(
        adata,
        library_key=sample_key,
        coord_type="generic",
        delaunay=bool(cellcharter_config.delaunay),
        spatial_key=spatial_key,
    )

    if cellcharter_config.remove_long_links:
        logging.info(
            "Removing long graph links above distance percentile %.2f.",
            float(cellcharter_config.distance_percentile),
        )
        cc.gr.remove_long_links(
            adata,
            distance_percentile=float(cellcharter_config.distance_percentile),
        )

    n_layers = _parse_n_layers(cellcharter_config.n_layers)
    aggregations = _parse_aggregations(cellcharter_config.aggregations)
    logging.info(
        "Aggregating neighborhoods (n_layers=%s, aggregations=%s, use_rep=%s).",
        n_layers,
        aggregations,
        aggregation_rep,
    )
    cc.gr.aggregate_neighbors(
        adata,
        n_layers=n_layers,
        aggregations=aggregations,
        use_rep=aggregation_rep,
        sample_key=sample_key,
        out_key=cellcharter_config.aggregated_rep_key,
    )

    trainer_params = {
        "accelerator": cellcharter_config.trainer_accelerator,
        "max_epochs": int(cellcharter_config.trainer_max_epochs),
    }
    if cellcharter_config.trainer_devices is not None:
        trainer_params["devices"] = int(cellcharter_config.trainer_devices)

    logging.info(
        "Clustering neighborhoods with CellCharter (n_clusters=%d, covariance_type=%s).",
        int(cellcharter_config.n_clusters),
        cellcharter_config.covariance_type,
    )
    model = cc.tl.Cluster(
        n_clusters=int(cellcharter_config.n_clusters),
        covariance_type=cellcharter_config.covariance_type,
        batch_size=cellcharter_config.batch_size,
        trainer_params=trainer_params,
        random_state=int(cellcharter_config.random_state),
    )
    model.fit(adata, use_rep=cellcharter_config.aggregated_rep_key)
    predicted = model.predict(adata, use_rep=cellcharter_config.aggregated_rep_key)

    labels = pd.Series(predicted.astype(str), index=adata.obs_names, dtype="object")
    categories = sorted(pd.unique(labels), key=_category_sort_key)
    adata.obs[cellcharter_config.cluster_key] = pd.Categorical(labels, categories=categories)

    nhood_details: Dict[str, Any] = {
        "enabled": bool(cellcharter_config.run_nhood_enrichment),
        "ran": False,
    }
    diff_nhood_details: Dict[str, Any] = {
        "enabled": bool(cellcharter_config.run_diff_nhood_enrichment),
        "ran": False,
    }
    shape_details: Dict[str, Any] = {
        "enabled": bool(cellcharter_config.run_shape_characterisation),
        "ran": False,
    }

    if cellcharter_config.run_enrichment:
        if not population_obs_primary:
            logging.warning(
                "Skipping enrichment because general.population_obs_primary is not set."
            )
        elif population_obs_primary not in adata.obs.columns:
            logging.warning(
                "Skipping enrichment because general.population_obs_primary '%s' is missing in adata.obs.",
                population_obs_primary,
            )
        else:
            logging.info(
                "Running enrichment for %s vs %s (pvalues=%s, n_perms=%d).",
                cellcharter_config.cluster_key,
                population_obs_primary,
                cellcharter_config.enrichment_with_pvalues,
                int(cellcharter_config.enrichment_n_perms),
            )
            cc.gr.enrichment(
                adata,
                group_key=cellcharter_config.cluster_key,
                label_key=population_obs_primary,
                pvalues=bool(cellcharter_config.enrichment_with_pvalues),
                n_perms=int(cellcharter_config.enrichment_n_perms),
            )
            _save_enrichment_outputs(
                adata,
                cluster_key=cellcharter_config.cluster_key,
                label_key=population_obs_primary,
                save_heatmap=cellcharter_config.save_enrichment_heatmap,
                qc_dir=qc_dir,
            )
            _save_cellcharter_enrichment_plot(
                adata,
                cluster_key=cellcharter_config.cluster_key,
                label_key=population_obs_primary,
                qc_dir=qc_dir,
            )

    if cellcharter_config.run_nhood_enrichment:
        nhood_details = _run_nhood_enrichment(
            adata=adata,
            cellcharter_config=cellcharter_config,
            qc_dir=qc_dir,
        )

    if cellcharter_config.run_diff_nhood_enrichment:
        diff_nhood_details = _run_diff_nhood_enrichment(
            adata=adata,
            general_config=general_config,
            cellcharter_config=cellcharter_config,
            sample_key=sample_key,
            qc_dir=qc_dir,
        )

    if cellcharter_config.run_shape_characterisation:
        shape_details = _run_shape_characterisation(
            adata=adata,
            general_config=general_config,
            cellcharter_config=cellcharter_config,
            sample_key=sample_key,
            qc_dir=qc_dir,
        )

    _save_cluster_tables(
        adata,
        sample_key=sample_key,
        cluster_key=cellcharter_config.cluster_key,
        qc_dir=qc_dir,
    )

    if cellcharter_config.save_spatial_plots:
        _save_spatial_cluster_plots(
            adata,
            sample_key=sample_key,
            spatial_key=spatial_key,
            cluster_key=cellcharter_config.cluster_key,
            point_size=float(cellcharter_config.point_size),
            max_samples=int(cellcharter_config.max_rois_for_plots),
            qc_dir=qc_dir,
        )

    _save_roi_cluster_masks(
        adata=adata,
        sample_key=sample_key,
        cluster_key=cellcharter_config.cluster_key,
        masks_folder=general_config.masks_folder,
        qc_dir=qc_dir,
    )

    adata.uns["cellcharter_pipeline"] = {
        "input_adata_path": str(input_path),
        "output_adata_path": str(output_path),
        "sample_key": sample_key,
        "spatial_key": spatial_key,
        "population_obs_primary": population_obs_primary,
        "cluster_key": cellcharter_config.cluster_key,
        "aggregated_rep_key": cellcharter_config.aggregated_rep_key,
        "aggregation_use_rep": aggregation_rep,
        "trvae": trvae_details,
        "nhood_enrichment": nhood_details,
        "diff_nhood_enrichment": diff_nhood_details,
        "shape_characterisation": shape_details,
    }

    sanitized_changes = _sanitize_uns_for_h5ad_write(adata)
    if sanitized_changes > 0:
        logging.warning(
            "Sanitized %d HDF5-unsafe labels in adata.uns before write_h5ad (replaced '/' with '_').",
            sanitized_changes,
        )

    output_path = save_pipeline_anndata(
        adata=adata,
        general_config=general_config,
        stage_name=stage_name,
        stage_config=cellcharter_config,
        override_path=str(output_path),
        extra_details={
            "input_adata_path": str(input_path),
            "qc_dir": str(qc_dir),
        },
    )
    logging.info("Saved CellCharter AnnData output to %s", output_path)
    logging.info("Saved CellCharter QC outputs to %s", qc_dir)
    return output_path


if __name__ == "__main__":
    pipeline_stage = "CellCharter"
    config = process_config_with_overrides()
    setup_logging(config.get("logging", {}), pipeline_stage)

    general_config = GeneralConfig(
        **filter_config_for_dataclass(config.get("general", {}), GeneralConfig)
    )
    cellcharter_config = CellCharterConfig(
        **filter_config_for_dataclass(config.get("cellcharter", {}), CellCharterConfig)
    )

    run_cellcharter_neighborhoods(
        general_config=general_config,
        cellcharter_config=cellcharter_config,
    )
