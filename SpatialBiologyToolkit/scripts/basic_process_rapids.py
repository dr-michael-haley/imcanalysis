"""
GPU processing stage using rapids-singlecell.

Runs PCA, optional Harmony batch correction, neighbors, UMAP, and Leiden
clustering on the post-segmentation AnnData object.
"""

from __future__ import annotations

import logging
import copy
import csv
import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
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


_SCAN_PARAMETER_ALIASES = {
    "n_neighbors": "n_neighbors",
    "n_for_pca": "n_for_pca",
    "umap_min_dist": "umap_min_dist",
    "run_harmony": "run_harmony",
    "harmony_flavor": "harmony_flavor",
    "harmony_flavour": "harmony_flavor",
}


def _coerce_optional_float(value: Any, *, name: str) -> Optional[float]:
    """Convert config scalar values to optional floats."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rapids.{name} must be numeric or null; got {value!r}.") from exc


def _coerce_optional_int(value: Any, *, name: str) -> Optional[int]:
    """Convert config scalar values to optional ints."""
    coerced = _coerce_optional_float(value, name=name)
    if coerced is None:
        return None
    if not float(coerced).is_integer():
        raise ValueError(f"rapids.{name} must be an integer or null; got {value!r}.")
    return int(coerced)


def _coerce_bool(value: Any, *, name: str) -> bool:
    """Convert config scalar values to booleans."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return bool(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "t", "yes", "y", "1"}:
            return True
        if cleaned in {"false", "f", "no", "n", "0"}:
            return False
    raise ValueError(f"rapids.{name} must be a boolean; got {value!r}.")


def _normalise_harmony_flavor(value: Any) -> str:
    """Normalize RAPIDS Harmony flavor config."""
    flavor = str(value).strip().lower()
    if flavor not in {"harmony1", "harmony2"}:
        raise ValueError("rapids.harmony_flavor must be one of: 'harmony1', 'harmony2'.")
    return flavor


def _as_resolution_list(values: Any) -> List[float]:
    """Normalize scalar/list-like resolution settings."""
    if values is None:
        return []
    if isinstance(values, list):
        return [float(x) for x in values]
    return [float(values)]


def _filter_cells_by_obs(adata: ad.AnnData, rapids_config: RapidsProcessConfig) -> ad.AnnData:
    """Filter cells by a numeric obs column using optional lower/upper bounds."""
    min_value = _coerce_optional_float(
        rapids_config.filter_min_value,
        name="filter_min_value",
    )
    max_value = _coerce_optional_float(
        rapids_config.filter_max_value,
        name="filter_max_value",
    )
    if min_value is None and max_value is None:
        logging.info(
            "RAPIDS cell filter disabled: rapids.filter_min_value and "
            "rapids.filter_max_value are both null."
        )
        return adata
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError(
            "rapids.filter_min_value cannot be greater than rapids.filter_max_value "
            f"({min_value} > {max_value})."
        )

    obs_key = _normalise_optional_key(rapids_config.filter_obs_key)
    if not obs_key:
        raise ValueError("rapids.filter_obs_key must be set when RAPIDS cell filtering is active.")
    if obs_key not in adata.obs.columns:
        raise KeyError(
            f"Configured RAPIDS cell filter obs key '{obs_key}' was not found in adata.obs."
        )

    values = pd.to_numeric(adata.obs[obs_key], errors="coerce")
    valid = values.notna()
    keep = valid.copy()
    below = pd.Series(False, index=values.index)
    above = pd.Series(False, index=values.index)
    if min_value is not None:
        below = values < min_value
        keep &= ~below
    if max_value is not None:
        above = values > max_value
        keep &= ~above

    kept_count = int(keep.sum())
    total_count = int(adata.n_obs)
    removed_count = total_count - kept_count
    invalid_count = int((~valid).sum())
    logging.info(
        "RAPIDS cell filter on adata.obs['%s'] with min=%s max=%s: kept %d/%d cells "
        "(%.2f%%), removed %d. Removed breakdown: below_min=%d, above_max=%d, "
        "missing_or_non_numeric=%d.",
        obs_key,
        min_value,
        max_value,
        kept_count,
        total_count,
        (100.0 * kept_count / total_count) if total_count else 0.0,
        removed_count,
        int((below & valid).sum()),
        int((above & valid).sum()),
        invalid_count,
    )
    if kept_count == 0:
        raise ValueError(
            "RAPIDS cell filter removed all cells. Check rapids.filter_min_value and "
            "rapids.filter_max_value."
        )
    if removed_count == 0:
        return adata
    return adata[keep.to_numpy(), :].copy()


def _as_scan_values(value: Any) -> List[Any]:
    """Normalize a parameter scan value to a non-empty list when configured."""
    if value is None:
        return []
    if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _coerce_scan_override(key: str, value: Any) -> Any:
    """Coerce parameter scan overrides to the same types as RapidsProcessConfig."""
    if key == "n_neighbors":
        return _coerce_optional_int(value, name="parameter_scan_dict.n_neighbors")
    if key == "n_for_pca":
        return _coerce_optional_int(value, name="parameter_scan_dict.n_for_pca")
    if key == "umap_min_dist":
        coerced = _coerce_optional_float(value, name="parameter_scan_dict.umap_min_dist")
        if coerced is None:
            raise ValueError("rapids.parameter_scan_dict.umap_min_dist cannot contain null values.")
        return coerced
    if key == "run_harmony":
        return _coerce_bool(value, name="parameter_scan_dict.run_harmony")
    if key == "harmony_flavor":
        return _normalise_harmony_flavor(value)
    raise KeyError(key)


def _build_parameter_scan_overrides(parameter_scan_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a parameter scan dictionary into Cartesian-product run overrides."""
    if not parameter_scan_dict:
        return []

    scan_values: Dict[str, List[Any]] = {}
    for raw_key, raw_values in parameter_scan_dict.items():
        canonical_key = _SCAN_PARAMETER_ALIASES.get(str(raw_key).strip())
        if canonical_key is None:
            allowed = ", ".join(sorted(set(_SCAN_PARAMETER_ALIASES.values())))
            raise ValueError(
                f"Unsupported rapids.parameter_scan_dict key '{raw_key}'. "
                f"Supported keys are: {allowed}."
            )
        values = [
            _coerce_scan_override(canonical_key, value)
            for value in _as_scan_values(raw_values)
        ]
        if values:
            scan_values[canonical_key] = values

    if not scan_values:
        return []

    scan_keys = list(scan_values.keys())
    return [
        dict(zip(scan_keys, combination))
        for combination in product(*(scan_values[key] for key in scan_keys))
    ]


def _apply_parameter_scan_overrides(
    rapids_config: RapidsProcessConfig,
    overrides: Dict[str, Any],
) -> RapidsProcessConfig:
    """Return a copied RapidsProcessConfig with scan overrides applied."""
    run_config = copy.deepcopy(rapids_config)
    for key, value in overrides.items():
        setattr(run_config, key, value)
    return run_config


def _parameter_scan_label(index: int, overrides: Dict[str, Any]) -> str:
    """Create a stable filesystem-safe label for a parameter scan run."""
    parts = [f"scan_{index:03d}"]
    for key, value in overrides.items():
        value_label = "none" if value is None else str(value)
        parts.append(f"{key}_{value_label}")
    return cleanstring("_".join(parts)) or f"scan_{index:03d}"


def _prepare_scan_output_path(base_path: Path, label: str) -> Path:
    """Attach a parameter scan label to an output AnnData filename."""
    return base_path.with_name(f"{base_path.stem}_{label}{base_path.suffix}")


def _write_parameter_scan_summary(rows: List[Dict[str, Any]], qc_dir: Path) -> Optional[Path]:
    """Write a summary CSV for RAPIDS parameter scans."""
    if not rows:
        return None

    summary_path = qc_dir / "rapids_parameter_scan_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "overrides",
        "qc_dir",
        "saved_anndata_path",
        "method",
        "n_cells",
        "n_markers",
        "n_pcs",
        "n_pcs_neighbors",
        "n_neighbors",
        "umap_min_dist",
        "run_harmony",
        "harmony_flavor",
        "matrixplot_count",
        "leiden_keys",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = row.copy()
            serialized["overrides"] = json.dumps(serialized.get("overrides", {}), sort_keys=True)
            serialized["leiden_keys"] = json.dumps(serialized.get("leiden_keys", []))
            writer.writerow({key: serialized.get(key, "") for key in fieldnames})
    logging.info("Wrote RAPIDS parameter scan summary to %s", summary_path)
    return summary_path


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
    harmony_flavor: str,
    harmony_params: Dict[str, Any],
) -> None:
    """Run RAPIDS Harmony integration against the configured PCA embedding."""
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Configured batch key '{batch_key}' was not found in adata.obs")

    params = _drop_managed_params(
        _normalise_dtype_param(harmony_params),
        managed={"key", "basis", "adjusted_basis", "flavor"},
        section_name="harmony",
    )
    params["flavor"] = _normalise_harmony_flavor(harmony_flavor)
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    logging.info(
        "Running RAPIDS Harmony on adata.obsm['%s'] using obs key '%s' (flavor=%s).",
        pca_key,
        batch_key,
        params["flavor"],
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
    n_pcs: Optional[int],
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
    if n_pcs is not None:
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
    key_prefix: str = "leiden",
) -> List[str]:
    """Run RAPIDS Leiden clustering for each configured resolution.

    ``key_prefix`` lets callers isolate labels produced in a distinct feature
    space without temporarily overwriting an existing ``leiden_<resolution>``
    observation column.
    """
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
        leiden_key = f"{key_prefix}_{res}"
        logging.info("Running RAPIDS Leiden clustering at resolution %s.", res)
        rsc.tl.leiden(adata, resolution=res, key_added=leiden_key, **params)
        leiden_keys.append(leiden_key)
    return leiden_keys


def _run_rapids_processing(
    adata: ad.AnnData,
    *,
    rapids_config: RapidsProcessConfig,
    viz_config: VisualizationConfig,
    qc_dir: Path,
) -> Dict[str, Any]:
    """Run one RAPIDS processing configuration and write its QC outputs."""
    run_harmony = bool(rapids_config.run_harmony)
    batch_key = _normalise_optional_key(rapids_config.batch_correction_obs)
    if run_harmony and not batch_key:
        raise ValueError("rapids.batch_correction_obs must be set when rapids.run_harmony=True.")
    if batch_key and batch_key not in adata.obs.columns:
        raise KeyError(f"Configured batch key '{batch_key}' was not found in adata.obs")

    harmony_flavor = _normalise_harmony_flavor(rapids_config.harmony_flavor)
    input_representation_key = _normalise_optional_key(
        rapids_config.input_representation_key
    )
    if input_representation_key and run_harmony:
        raise ValueError(
            "rapids.input_representation_key cannot be combined with "
            "rapids.run_harmony=True."
        )
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
    qc_dir.mkdir(parents=True, exist_ok=True)

    n_pcs: Optional[int] = None
    gpu_layer: Optional[str] = None

    try:
        if input_representation_key:
            if input_representation_key not in adata.obsm:
                raise KeyError(
                    f"Configured RAPIDS input representation "
                    f"'{input_representation_key}' was not found in adata.obsm"
                )
            active_representation = input_representation_key
            logging.info(
                "Using existing RAPIDS input representation adata.obsm['%s']; "
                "skipping PCA and Harmony.",
                input_representation_key,
            )
        else:
            n_pcs = _resolve_n_pcs(adata, rapids_config.n_for_pca)
            gpu_layer = _move_input_matrix_to_gpu(adata, pca_params)
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
                    harmony_flavor=harmony_flavor,
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
            default=n_pcs or adata.obsm[representation_key].shape[1],
        )
        if input_representation_key and rapids_config.n_pcs_neighbors is None:
            n_pcs_neighbors = None
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
        if not input_representation_key:
            _move_input_matrix_to_cpu(adata, gpu_layer)

    method = (
        "rapids_existing_representation"
        if input_representation_key
        else "rapids_harmony"
        if run_harmony
        else "rapids"
    )
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

    run_details: Dict[str, Any] = {
        "method": method,
        "batch_key": batch_key,
        "n_pcs": int(n_pcs) if n_pcs is not None else None,
        "n_pcs_neighbors": (
            int(n_pcs_neighbors) if n_pcs_neighbors is not None else None
        ),
        "n_neighbors": rapids_config.n_neighbors,
        "umap_min_dist": float(rapids_config.umap_min_dist),
        "run_harmony": bool(run_harmony),
        "harmony_flavor": harmony_flavor if run_harmony else None,
        "representation_key": representation_key,
        "source_representation_key": active_representation,
        "input_representation_key": input_representation_key,
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
        "matrixplot_count": len(matrixplot_paths),
        "n_cells": int(adata.n_obs),
        "n_markers": int(adata.n_vars),
    }
    adata.uns["rapids_process"] = run_details
    adata.uns["batch_integration"] = {
        "method": method,
        "batch_key": batch_key,
        "representation_key": representation_key,
        "source_representation_key": active_representation,
        "pca_key": pca_key,
        "harmony_key": harmony_key if run_harmony else None,
    }
    return run_details


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
    adata = _filter_cells_by_obs(adata, rapids_config)
    logging.info("AnnData shape after RAPIDS cell filter: %s.", adata.shape)

    qc_dir = Path(general_config.qc_folder) / rapids_config.qc_output_subdir
    scan_overrides = _build_parameter_scan_overrides(rapids_config.parameter_scan_dict)
    output_path = Path(output_path)

    if scan_overrides:
        scan_qc_subdir = _normalise_optional_key(rapids_config.parameter_scan_qc_subdir)
        scan_qc_dir = qc_dir / (scan_qc_subdir or "ParameterScan")
        scan_qc_dir.mkdir(parents=True, exist_ok=True)
        logging.info(
            "RAPIDS parameter scan enabled with %d run(s). AnnData saving is %s "
            "(rapids.parameter_scan_save_anndata=%s). QC outputs will be written under %s.",
            len(scan_overrides),
            "enabled" if rapids_config.parameter_scan_save_anndata else "disabled",
            rapids_config.parameter_scan_save_anndata,
            scan_qc_dir,
        )

        scan_rows: List[Dict[str, Any]] = []
        for scan_index, overrides in enumerate(scan_overrides, start=1):
            scan_label = _parameter_scan_label(scan_index, overrides)
            scan_config = _apply_parameter_scan_overrides(rapids_config, overrides)
            run_qc_dir = scan_qc_dir / scan_label
            logging.info(
                "Running RAPIDS parameter scan %d/%d (%s): %s",
                scan_index,
                len(scan_overrides),
                scan_label,
                overrides,
            )
            run_adata = adata.copy()
            run_details = _run_rapids_processing(
                run_adata,
                rapids_config=scan_config,
                viz_config=viz_config,
                qc_dir=run_qc_dir,
            )

            saved_anndata_path = ""
            if rapids_config.parameter_scan_save_anndata:
                scan_output_path = _prepare_scan_output_path(output_path, scan_label)
                saved_path = save_pipeline_anndata(
                    adata=run_adata,
                    general_config=general_config,
                    stage_name=pipeline_stage,
                    stage_config=scan_config,
                    override_path=str(scan_output_path),
                    extra_details={
                        "input_adata_path": str(resolved_input_path),
                        "output_adata_path": str(scan_output_path),
                        "parameter_scan_label": scan_label,
                        "parameter_scan_overrides": overrides,
                        **{
                            key: run_details.get(key)
                            for key in (
                                "method",
                                "representation_key",
                                "neighbors_key",
                                "umap_key",
                                "qc_dir",
                                "matrixplot_count",
                                "n_cells",
                                "n_markers",
                            )
                        },
                    },
                )
                saved_anndata_path = str(saved_path)
                logging.info(
                    "Saved RAPIDS parameter scan AnnData for '%s' to %s",
                    scan_label,
                    saved_path,
                )

            scan_rows.append(
                {
                    "label": scan_label,
                    "overrides": overrides,
                    "saved_anndata_path": saved_anndata_path,
                    **run_details,
                }
            )

        summary_path = _write_parameter_scan_summary(scan_rows, scan_qc_dir)
        logging.info(
            "Completed RAPIDS parameter scan with %d run(s). Summary: %s",
            len(scan_rows),
            summary_path,
        )
        return

    run_details = _run_rapids_processing(
        adata,
        rapids_config=rapids_config,
        viz_config=viz_config,
        qc_dir=qc_dir,
    )

    saved_path = save_pipeline_anndata(
        adata=adata,
        general_config=general_config,
        stage_name=pipeline_stage,
        stage_config=rapids_config,
        override_path=str(output_path),
        extra_details={
            "input_adata_path": str(resolved_input_path),
            "output_adata_path": str(output_path),
            **{
                key: run_details.get(key)
                for key in (
                    "method",
                    "representation_key",
                    "neighbors_key",
                    "umap_key",
                    "qc_dir",
                    "matrixplot_count",
                    "n_cells",
                    "n_markers",
                )
            },
        },
    )
    logging.info("Saved RAPIDS-processed AnnData to %s", saved_path)


if __name__ == "__main__":
    main()
