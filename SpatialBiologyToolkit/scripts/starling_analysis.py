"""
STARLING segmentation-aware phenotyping stage for IMC AnnData outputs.

The stage runs Starling on a non-negative marker-expression matrix and writes
prefixed results back into the full AnnData object. User-provided initial labels
from ``adata.obs`` are encoded to contiguous integer IDs because Starling's
trainer expects ``init_label`` values in the range ``0..K-1``.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

try:
    from scipy import sparse
except Exception:  # pragma: no cover
    sparse = None  # type: ignore

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from SpatialBiologyToolkit.reporting import get_active_reporter, project_asset_path

from .config_and_utils import (
    GeneralConfig,
    StarlingConfig,
    cleanstring,
    filter_config_for_dataclass,
    load_pipeline_anndata,
    process_config_with_overrides,
    save_pipeline_anndata,
    setup_logging,
)

_NULL_STRINGS = {"", "none", "null"}
_STARLING_METHODS = {"user": "User", "km": "KM", "gmm": "GMM", "fs": "FS", "pg": "PG"}


def _optional_text(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _NULL_STRINGS else text


def _text_list(value: Optional[Any]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if text.lower() in _NULL_STRINGS:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if "," in text:
        return [part.strip().strip("'\"") for part in text.split(",") if part.strip()]
    return [text.strip("'\"")]


def _natural_key(value: Any) -> List[Any]:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(value))]


def _prefix(value: str) -> str:
    cleaned = cleanstring(value)
    return cleaned if cleaned else "starling"


def _initial_method(method: str) -> str:
    key = str(method).strip().lower()
    if key not in _STARLING_METHODS:
        raise ValueError("starling.initial_clustering_method must be one of: User, KM, GMM, FS, PG.")
    return _STARLING_METHODS[key]


def _import_starling(repo_path: Optional[str]):
    path = _optional_text(repo_path)
    if path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"starling.starling_repo_path does not exist: {resolved}")
        sys.path.insert(0, str(resolved))
        logging.info("Prepended local Starling checkout to sys.path: %s", resolved)
    try:
        from starling import starling, utility
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Starling is required for this stage. Activate the 'imc_starling' "
            "environment, install biostarling, or set starling.starling_repo_path."
        ) from exc
    return starling, utility


def _import_lightning():
    try:
        from lightning_lite import seed_everything
    except ImportError:  # pragma: no cover
        from pytorch_lightning import seed_everything
    try:
        from pytorch_lightning.callbacks import EarlyStopping
        from pytorch_lightning.loggers import TensorBoardLogger
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pytorch_lightning is required in the 'imc_starling' environment.") from exc
    return EarlyStopping, TensorBoardLogger, seed_everything


def _resolve_markers(adata: ad.AnnData, cfg: StarlingConfig) -> List[str]:
    var_names = pd.Index([str(v) for v in adata.var_names])
    include = _text_list(cfg.marker_include)
    exclude = set(_text_list(cfg.marker_exclude))
    if include:
        missing = [m for m in include if m not in var_names]
        if missing:
            raise KeyError("starling.marker_include contains missing markers: " + ", ".join(missing[:20]))
        markers = include
    else:
        markers = list(var_names)
    markers = [m for m in markers if m not in exclude]
    if len(markers) < 10:
        raise ValueError(f"STARLING requires at least 10 markers/features; selected {len(markers)}.")
    return markers


def _dense_float64(matrix: Any) -> np.ndarray:
    if sparse is not None and sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


def _select_matrix(
    adata: ad.AnnData,
    markers: Sequence[str],
    cfg: StarlingConfig,
) -> Tuple[np.ndarray, str]:
    idx = adata.var_names.get_indexer(markers)
    if np.any(idx < 0):
        missing = [m for m, i in zip(markers, idx) if i < 0]
        raise KeyError("Selected markers missing from adata.var_names: " + ", ".join(missing[:20]))

    layer = _optional_text(cfg.use_layer)
    if layer is None:
        source = adata.X
        label = "X"
    else:
        if layer not in adata.layers:
            raise KeyError(f"starling.use_layer '{layer}' was not found in adata.layers.")
        source = adata.layers[layer]
        label = f"layers['{layer}']"

    selected = source.iloc[:, idx] if isinstance(source, pd.DataFrame) else source[:, idx]
    matrix = _dense_float64(selected)
    if matrix.shape != (adata.n_obs, len(markers)):
        raise ValueError(f"STARLING matrix shape {matrix.shape} does not match cells/markers.")
    if not np.isfinite(matrix).all():
        raise ValueError("STARLING matrix contains NaN or infinite values.")
    min_value = float(np.min(matrix))
    if min_value < 0:
        tol = abs(float(cfg.negative_value_tolerance))
        if bool(cfg.clip_small_negative_values) and min_value >= -tol:
            logging.warning("Clipping tiny negative expression values to zero (minimum=%g).", min_value)
            matrix = matrix.copy()
            matrix[matrix < 0] = 0.0
        else:
            raise ValueError(
                "STARLING expects non-negative expression values. "
                f"The selected matrix has minimum {min_value:g}."
            )
    return matrix, label


def _feature_adata(adata: ad.AnnData, matrix: np.ndarray, markers: Sequence[str]) -> ad.AnnData:
    var = pd.DataFrame(index=pd.Index(list(markers), name=adata.var_names.name))
    return ad.AnnData(X=matrix, obs=adata.obs.copy(), var=var)


def _cell_size_col(adata: ad.AnnData, cfg: StarlingConfig) -> Optional[str]:
    if not bool(cfg.model_cell_size):
        return None
    candidates = [str(cfg.cell_size_col_name)] + _text_list(cfg.cell_size_fallback_cols)
    for col in candidates:
        if col not in adata.obs.columns:
            continue
        values = pd.to_numeric(adata.obs[col], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Cell size column '{col}' contains missing or non-numeric values.")
        if (values <= 0).any():
            raise ValueError(f"Cell size column '{col}' contains non-positive values.")
        adata.obs[col] = values.astype(float)
        if col != str(cfg.cell_size_col_name):
            logging.warning("Using fallback cell size column '%s'.", col)
        return col
    raise KeyError("STARLING cell-size modeling is enabled, but none of these obs columns exist: " + ", ".join(candidates))


def _ordered_labels(series: pd.Series) -> List[str]:
    present = {str(v) for v in series.dropna().astype("object").unique()}
    if isinstance(series.dtype, pd.CategoricalDtype):
        ordered = [str(v) for v in series.cat.categories if str(v) in present]
        return ordered + sorted(present.difference(ordered), key=_natural_key)
    return sorted(present, key=_natural_key)


def _init_clusters(
    fadata: ad.AnnData,
    *,
    general: GeneralConfig,
    cfg: StarlingConfig,
    utility: Any,
) -> Tuple[ad.AnnData, Optional[str], Optional[pd.Series], pd.DataFrame]:
    method = _initial_method(cfg.initial_clustering_method)
    if method == "User":
        label_col = _optional_text(cfg.initial_label_obs) or _optional_text(general.population_obs_primary)
        if label_col is None:
            raise ValueError(
                "Set starling.initial_label_obs, or set general.population_obs_primary, "
                "when starling.initial_clustering_method is User."
            )
        if label_col not in fadata.obs.columns:
            raise KeyError(f"Initial label column '{label_col}' was not found in adata.obs.")
        raw = fadata.obs[label_col]
        if raw.isna().any():
            raise ValueError(f"Initial label column '{label_col}' contains {int(raw.isna().sum())} missing values.")
        labels = raw.astype(str)
        ordered = _ordered_labels(raw)
        encoded = labels.map({label: i for i, label in enumerate(ordered)}).to_numpy(dtype=int)
        logging.info("Using adata.obs['%s'] as STARLING initialization (%d clusters).", label_col, len(ordered))
        fadata = utility.init_clustering("User", fadata, labels=encoded)
        mapping = pd.DataFrame(
            {
                "source_label_obs": label_col,
                "source_label": ordered,
                "starling_init_label": list(range(len(ordered))),
            }
        )
        return fadata, label_col, labels, mapping

    if method in {"KM", "GMM", "FS"} and cfg.n_clusters is None:
        raise ValueError(f"starling.n_clusters must be set for {method} initialization.")
    logging.info(
        "Using STARLING built-in %s initialization%s.",
        method,
        f" with n_clusters={int(cfg.n_clusters)}" if cfg.n_clusters is not None else "",
    )
    fadata = utility.init_clustering(method, fadata, k=cfg.n_clusters)
    classes = sorted(pd.unique(np.asarray(fadata.obs["init_label"])), key=_natural_key)
    mapping = pd.DataFrame(
        {
            "source_label_obs": "",
            "source_label": [str(c) for c in classes],
            "starling_init_label": [str(c) for c in classes],
        }
    )
    return fadata, None, None, mapping


def _categorical(values: Sequence[Any]) -> pd.Categorical:
    text = pd.Series(values, dtype="object").astype(str)
    return pd.Categorical(text, categories=sorted(pd.unique(text), key=_natural_key))


def _copy_varm(
    adata: ad.AnnData,
    result: ad.AnnData,
    *,
    markers: Sequence[str],
    source_key: str,
    target_key: str,
) -> None:
    if source_key not in result.varm:
        return
    matrix = np.asarray(result.varm[source_key], dtype=float)
    full = np.full((adata.n_vars, matrix.shape[1]), np.nan, dtype=float)
    idx = adata.var_names.get_indexer(markers)
    valid = idx >= 0
    full[idx[valid], :] = matrix[valid, :]
    adata.varm[target_key] = full


def _copy_results(
    adata: ad.AnnData,
    result: ad.AnnData,
    *,
    markers: Sequence[str],
    prefix: str,
    cfg: StarlingConfig,
    source_label_obs: Optional[str],
    source_labels: Optional[pd.Series],
    mapping: pd.DataFrame,
) -> Dict[str, Any]:
    obs_keys = {
        "init_label": f"{prefix}_init_label",
        "label": f"{prefix}_label",
        "doublet_prob": f"{prefix}_doublet_prob",
        "doublet": f"{prefix}_doublet",
        "max_assign_prob": f"{prefix}_max_assign_prob",
    }
    if source_labels is not None:
        source_key = f"{prefix}_source_label"
        adata.obs[source_key] = pd.Categorical(
            source_labels.astype(str),
            categories=list(mapping["source_label"].astype(str)),
        )
        obs_keys["source_label"] = source_key

    adata.obs[obs_keys["init_label"]] = _categorical(result.obs["init_label"].to_numpy())
    adata.obs[obs_keys["label"]] = _categorical(result.obs["st_label"].to_numpy())
    adata.obs[obs_keys["doublet_prob"]] = pd.to_numeric(result.obs["doublet_prob"], errors="raise").to_numpy()
    adata.obs[obs_keys["doublet"]] = (
        adata.obs[obs_keys["doublet_prob"]].to_numpy(dtype=float) > float(cfg.doublet_threshold)
    ).astype(int)
    adata.obs[obs_keys["max_assign_prob"]] = pd.to_numeric(result.obs["max_assign_prob"], errors="raise").to_numpy()

    if bool(cfg.write_canonical_starling_keys):
        adata.obs["init_label"] = result.obs["init_label"].to_numpy()
        adata.obs["st_label"] = result.obs["st_label"].to_numpy()
        adata.obs["doublet_prob"] = adata.obs[obs_keys["doublet_prob"]].to_numpy()
        adata.obs["doublet"] = adata.obs[obs_keys["doublet"]].to_numpy()
        adata.obs["max_assign_prob"] = adata.obs[obs_keys["max_assign_prob"]].to_numpy()

    obsm_keys: Dict[str, str] = {}
    if bool(cfg.store_assignment_prob_matrix) and "assignment_prob_matrix" in result.obsm:
        key = f"{prefix}_assignment_prob_matrix"
        adata.obsm[key] = np.asarray(result.obsm["assignment_prob_matrix"])
        obsm_keys["assignment_prob_matrix"] = key
        if bool(cfg.write_canonical_starling_keys):
            adata.obsm["assignment_prob_matrix"] = adata.obsm[key]
    gamma = "gamma_assignment_prob_matrix"
    if bool(cfg.store_gamma_assignment_prob_matrix) and gamma in result.obsm:
        key = f"{prefix}_gamma_assignment_prob_matrix"
        adata.obsm[key] = np.asarray(result.obsm[gamma])
        obsm_keys[gamma] = key
        if bool(cfg.write_canonical_starling_keys):
            adata.obsm[gamma] = adata.obsm[key]

    varm_keys = {
        "init_exp_centroids": f"{prefix}_init_exp_centroids",
        "init_exp_variances": f"{prefix}_init_exp_variances",
        "st_exp_centroids": f"{prefix}_exp_centroids",
    }
    for source, target in varm_keys.items():
        _copy_varm(adata, result, markers=markers, source_key=source, target_key=target)
        if bool(cfg.write_canonical_starling_keys) and target in adata.varm:
            adata.varm[source] = adata.varm[target]

    cell_size = {}
    for key in ("init_cell_size_centroids", "init_cell_size_variances", "st_cell_size_centroids"):
        if key in result.uns and result.uns[key] is not None:
            cell_size[key] = np.asarray(result.uns[key]).tolist()

    return {
        "obs_keys": obs_keys,
        "obsm_keys": obsm_keys,
        "varm_keys": varm_keys,
        "source_label_obs": source_label_obs,
        "n_clusters": int(result.varm["st_exp_centroids"].shape[1]),
        "cell_size": cell_size,
    }


def _centroid_csv(result: ad.AnnData, *, key: str, markers: Sequence[str], path: Path) -> None:
    if key not in result.varm:
        return
    matrix = np.asarray(result.varm[key])
    columns = [f"cluster_{i}" for i in range(matrix.shape[1])]
    pd.DataFrame(matrix, index=list(markers), columns=columns).to_csv(path)


def _write_qc_tables(
    adata: ad.AnnData,
    result: ad.AnnData,
    *,
    general: GeneralConfig,
    cfg: StarlingConfig,
    prefix: str,
    markers: Sequence[str],
    mapping: pd.DataFrame,
    details: Dict[str, Any],
    qc_dir: Path,
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    mapping_path = qc_dir / f"{prefix}_init_label_mapping.csv"
    mapping.to_csv(mapping_path, index=False)
    paths["init_label_mapping"] = str(mapping_path)

    centroid_paths = {
        "init_exp_centroids": qc_dir / f"{prefix}_init_expression_centroids.csv",
        "init_exp_variances": qc_dir / f"{prefix}_init_expression_variances.csv",
        "st_exp_centroids": qc_dir / f"{prefix}_expression_centroids.csv",
    }
    for key, path in centroid_paths.items():
        _centroid_csv(result, key=key, markers=markers, path=path)
        if path.exists():
            paths[key] = str(path)

    obs_keys = details["obs_keys"]
    label_col = obs_keys["label"]
    doublet_col = obs_keys["doublet"]
    doublet_prob_col = obs_keys["doublet_prob"]
    max_prob_col = obs_keys["max_assign_prob"]

    counts = adata.obs[label_col].astype(str).value_counts().rename_axis(label_col).reset_index(name="n_cells")
    counts = counts.sort_values(label_col, key=lambda s: s.map(_natural_key))
    counts["fraction"] = counts["n_cells"] / float(adata.n_obs)
    counts_path = qc_dir / f"{prefix}_cluster_counts.csv"
    counts.to_csv(counts_path, index=False)
    paths["cluster_counts"] = str(counts_path)

    doublet_path = qc_dir / f"{prefix}_doublet_by_cluster.csv"
    pd.crosstab(
        adata.obs[label_col].astype(str),
        adata.obs[doublet_col].astype(int),
        rownames=[label_col],
        colnames=[doublet_col],
    ).to_csv(doublet_path)
    paths["doublet_by_cluster"] = str(doublet_path)

    summary_path = qc_dir / f"{prefix}_summary.csv"
    pd.DataFrame(
        [
            {
                "n_cells": int(adata.n_obs),
                "n_markers": int(len(markers)),
                "n_clusters": int(details["n_clusters"]),
                "doublet_threshold": float(cfg.doublet_threshold),
                "doublet_fraction": float(adata.obs[doublet_col].mean()),
                "mean_doublet_prob": float(adata.obs[doublet_prob_col].mean()),
                "median_doublet_prob": float(adata.obs[doublet_prob_col].median()),
                "mean_max_assign_prob": float(adata.obs[max_prob_col].mean()),
                "median_max_assign_prob": float(adata.obs[max_prob_col].median()),
            }
        ]
    ).to_csv(summary_path, index=False)
    paths["summary"] = str(summary_path)

    candidate_cols = [
        general.roi_obs,
        general.case_obs,
        general.groupby_obs,
        details.get("source_label_obs"),
        obs_keys.get("source_label"),
        obs_keys["init_label"],
        label_col,
        doublet_prob_col,
        doublet_col,
        max_prob_col,
    ]
    selected_cols: List[str] = []
    for col in candidate_cols:
        if col and col in adata.obs.columns and col not in selected_cols:
            selected_cols.append(col)
    cells_path = qc_dir / f"{prefix}_cell_results.csv"
    adata.obs[selected_cols].to_csv(cells_path)
    paths["cell_results"] = str(cells_path)
    return paths


def _write_qc_plots(
    adata: ad.AnnData,
    *,
    cfg: StarlingConfig,
    prefix: str,
    details: Dict[str, Any],
    qc_dir: Path,
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    fmt = cleanstring(cfg.figure_format).lower() or "png"
    obs_keys = details["obs_keys"]

    plot_specs = [
        (obs_keys["doublet_prob"], "Doublet probability", "Cells", "STARLING doublet probabilities", "doublet_probability_hist", "#4c78a8"),
        (obs_keys["max_assign_prob"], "Maximum assignment probability", "Cells", "STARLING assignment confidence", "max_assignment_probability_hist", "#72b7b2"),
    ]
    for col, xlabel, ylabel, title, suffix, color in plot_specs:
        values = pd.to_numeric(adata.obs[col], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(values, bins=50, color=color, edgecolor="white", linewidth=0.4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        path = qc_dir / f"{prefix}_{suffix}.{fmt}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths[suffix] = str(path)

    counts = adata.obs[obs_keys["label"]].astype(str).value_counts()
    counts = counts.sort_index(key=lambda idx: idx.map(_natural_key))
    fig, ax = plt.subplots(figsize=(max(5.0, min(14.0, 0.35 * len(counts))), 3.5))
    ax.bar(counts.index.astype(str), counts.to_numpy(), color="#54a24b")
    ax.set_xlabel("STARLING cluster")
    ax.set_ylabel("Cells")
    ax.set_title("STARLING cluster sizes")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    path = qc_dir / f"{prefix}_cluster_counts.{fmt}"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["cluster_counts_plot"] = str(path)
    return paths


def _train(
    fadata: ad.AnnData,
    *,
    cfg: StarlingConfig,
    qc_dir: Path,
    cell_size_col: Optional[str],
    starling_module: Any,
    early_stopping_cls: Any,
    tensorboard_logger_cls: Any,
) -> Any:
    st = starling_module.ST(
        fadata,
        dist_option=str(cfg.dist_option),
        singlet_prop=float(cfg.singlet_prop),
        model_cell_size=bool(cfg.model_cell_size),
        cell_size_col_name=str(cell_size_col or cfg.cell_size_col_name),
        model_zplane_overlap=bool(cfg.model_zplane_overlap),
        model_regularizer=float(cfg.model_regularizer),
        learning_rate=float(cfg.learning_rate),
    )
    callbacks = None
    if bool(cfg.early_stopping):
        callbacks = [early_stopping_cls(monitor=str(cfg.early_stopping_monitor), mode="min", verbose=False)]
    logger: Any = False
    if bool(cfg.tensorboard_logging):
        logger = tensorboard_logger_cls(save_dir=str(qc_dir / "lightning_logs"), name="starling")

    kwargs: Dict[str, Any] = {
        "callbacks": callbacks,
        "logger": logger,
        "max_epochs": cfg.max_epochs,
        "accelerator": str(cfg.trainer_accelerator),
        "enable_checkpointing": bool(cfg.enable_checkpointing),
        "enable_progress_bar": bool(cfg.enable_progress_bar),
        "default_root_dir": str(qc_dir),
    }
    if cfg.trainer_devices is not None:
        kwargs["devices"] = int(cfg.trainer_devices)
    if cfg.trainer_precision is not None:
        kwargs["precision"] = cfg.trainer_precision
    if cfg.log_every_n_steps is not None:
        kwargs["log_every_n_steps"] = int(cfg.log_every_n_steps)
    if cfg.limit_train_batches is not None:
        kwargs["limit_train_batches"] = cfg.limit_train_batches

    logging.info(
        "Training STARLING (max_epochs=%s, accelerator=%s, model_cell_size=%s).",
        cfg.max_epochs,
        cfg.trainer_accelerator,
        bool(cfg.model_cell_size),
    )
    st.train_and_fit(**kwargs)
    return st


def run_starling_analysis(*, general_config: GeneralConfig, starling_config: StarlingConfig) -> Path:
    stage_name = "Starling"
    input_path = starling_config.input_adata_path or general_config.anndata_path
    output_path = starling_config.output_adata_path or general_config.anndata_path
    qc_dir = Path(general_config.qc_folder) / starling_config.qc_output_subdir
    qc_dir.mkdir(parents=True, exist_ok=True)

    adata, resolved_input_path, skip_stage, _ = load_pipeline_anndata(
        general_config=general_config,
        stage_name=stage_name,
        stage_config=starling_config,
        override_path=input_path,
    )
    if skip_stage:
        logging.info("Skipping STARLING stage based on AnnData stage policy.")
        return Path(output_path)
    if adata is None:
        raise FileNotFoundError(f"AnnData could not be loaded for STARLING stage: {resolved_input_path}")
    if adata.n_obs < 10:
        raise ValueError(f"STARLING requires at least 10 cells; loaded {adata.n_obs}.")

    starling_module, utility = _import_starling(starling_config.starling_repo_path)
    early_stopping_cls, tensorboard_logger_cls, seed_everything = _import_lightning()
    seed_everything(int(starling_config.seed), workers=True)

    markers = _resolve_markers(adata, starling_config)
    matrix, feature_source = _select_matrix(adata, markers, starling_config)
    fadata = _feature_adata(adata, matrix, markers)
    cell_size_col = _cell_size_col(fadata, starling_config)
    fadata, source_label_obs, source_labels, mapping = _init_clusters(
        fadata,
        general=general_config,
        cfg=starling_config,
        utility=utility,
    )

    st = _train(
        fadata,
        cfg=starling_config,
        qc_dir=qc_dir,
        cell_size_col=cell_size_col,
        starling_module=starling_module,
        early_stopping_cls=early_stopping_cls,
        tensorboard_logger_cls=tensorboard_logger_cls,
    )
    result = st.result(threshold=float(starling_config.doublet_threshold))

    prefix = _prefix(starling_config.output_prefix)
    details = _copy_results(
        adata,
        result,
        markers=markers,
        prefix=prefix,
        cfg=starling_config,
        source_label_obs=source_label_obs,
        source_labels=source_labels,
        mapping=mapping,
    )
    qc_tables = {}
    qc_plots = {}
    if bool(starling_config.save_qc_tables):
        qc_tables = _write_qc_tables(
            adata,
            result,
            general=general_config,
            cfg=starling_config,
            prefix=prefix,
            markers=markers,
            mapping=mapping,
            details=details,
            qc_dir=qc_dir,
        )
    if bool(starling_config.save_qc_plots):
        qc_plots = _write_qc_plots(
            adata,
            cfg=starling_config,
            prefix=prefix,
            details=details,
            qc_dir=qc_dir,
        )

    model_path = ""
    if bool(starling_config.save_model):
        import torch

        resolved_model_path = project_asset_path(starling_config.model_output_name)
        resolved_model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path = str(resolved_model_path)
        torch.save(st, model_path)
        logging.info("Saved STARLING model to %s", model_path)
        reporter = get_active_reporter()
        if reporter is not None:
            reporter.add_asset(
                "starling_model",
                resolved_model_path,
                "Reusable STARLING model checkpoint.",
            )

    run_details = {
        "input_adata_path": str(resolved_input_path),
        "output_adata_path": str(output_path),
        "qc_dir": str(qc_dir),
        "feature_source": feature_source,
        "markers": list(map(str, markers)),
        "n_cells": int(adata.n_obs),
        "n_markers": int(len(markers)),
        "initial_clustering_method": _initial_method(starling_config.initial_clustering_method),
        "initial_label_obs": source_label_obs,
        "cell_size_col_name": cell_size_col,
        "doublet_threshold": float(starling_config.doublet_threshold),
        "copy_details": details,
        "qc_tables": qc_tables,
        "qc_plots": qc_plots,
        "model_path": model_path,
    }
    adata.uns[prefix] = run_details

    saved_path = save_pipeline_anndata(
        adata=adata,
        general_config=general_config,
        stage_name=stage_name,
        stage_config=starling_config,
        override_path=str(output_path),
        extra_details={
            "input_adata_path": str(resolved_input_path),
            "output_adata_path": str(output_path),
            "qc_dir": str(qc_dir),
            "n_cells": int(adata.n_obs),
            "n_markers": int(len(markers)),
            "n_clusters": int(details["n_clusters"]),
            "initial_label_obs": source_label_obs,
        },
    )
    logging.info("Saved STARLING AnnData output to %s", saved_path)
    logging.info("Saved STARLING QC outputs to %s", qc_dir)
    return saved_path


def main() -> None:
    pipeline_stage = "Starling"
    config = process_config_with_overrides()
    setup_logging(config.get("logging", {}), pipeline_stage)
    general_config = GeneralConfig(
        **filter_config_for_dataclass(config.get("general", {}), GeneralConfig)
    )
    starling_config = StarlingConfig(
        **filter_config_for_dataclass(config.get("starling", {}), StarlingConfig)
    )
    run_starling_analysis(general_config=general_config, starling_config=starling_config)


if __name__ == "__main__":
    main()
