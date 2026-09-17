"""Notebook-friendly cell2location reference fitting and multi-library Visium mapping.

Inputs are copied, never normalized or overwritten. Heavy modelling dependencies
are imported only by fitting functions. See ``docs/source/stages/cell2location.md``.
"""

from __future__ import annotations

import copy
import importlib.metadata
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from SpatialBiologyToolkit.config.models import Cell2locationConfig

LOGGER = logging.getLogger(__name__)
BARCODE = "_sbt_c2l_barcode"
SOURCE_OBS = "_sbt_c2l_source_obs_name"
PRIOR_MEAN = "_sbt_c2l_prior_mean"
PRIOR_RAW = "_sbt_c2l_prior_raw"


@dataclass
class Cell2locationResult:
    """A fitted upstream model plus its posterior AnnData and optional signatures."""

    model: Any
    adata: ad.AnnData
    signatures: pd.DataFrame | None = None


def _string_values(values: Any, label: str) -> np.ndarray:
    series = pd.Series(np.asarray(values))
    if series.isna().any() or series.astype(str).str.strip().eq("").any():
        raise ValueError(f"{label} contains missing/empty identifiers")
    return series.astype(str).to_numpy()


def validate_counts(matrix: Any, label: str) -> None:
    """Validate finite, nonnegative integer counts without densifying sparse data."""
    if matrix is None or len(matrix.shape) != 2 or 0 in matrix.shape:
        raise ValueError(f"{label}: counts must be a nonempty two-dimensional matrix")
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).reshape(-1)
    for start in range(0, len(values), 1_000_000):
        block = values[start : start + 1_000_000]
        if not np.isfinite(block).all() or (block < 0).any():
            raise ValueError(f"{label}: counts contain negative or non-finite values")
        if not np.allclose(block, np.rint(block), rtol=0, atol=1e-5):
            raise ValueError(
                f"{label}: use raw integer counts, not normalized/log-transformed expression"
            )


def _count_copy(
    source: ad.AnnData,
    layer: str | None,
    gene_id_key: str | None,
    label: str,
) -> ad.AnnData:
    if source.isbacked:
        raise ValueError(
            "Load the H5AD into memory before preparing cell2location inputs"
        )
    if layer == "raw":
        if source.raw is None:
            raise ValueError(f"{label}: adata.raw is absent")
        matrix, var = source.raw.X, source.raw.var.copy()
    elif layer:
        if layer not in source.layers:
            raise ValueError(f"{label}: missing counts layer {layer!r}")
        matrix, var = source.layers[layer], source.var.copy()
    else:
        matrix, var = source.X, source.var.copy()
    validate_counts(matrix, label)
    if gene_id_key and gene_id_key not in var:
        raise ValueError(f"{label}: missing gene ID column {gene_id_key!r}")
    ids = _string_values(
        var[gene_id_key] if gene_id_key else var.index, f"{label} gene IDs"
    )
    if not pd.Index(ids).is_unique:
        raise ValueError(
            f"{label}: duplicate gene IDs; resolve/aggregate them explicitly before fitting"
        )
    var.index = pd.Index(ids)
    obs = source.obs.copy()
    # Registration state from unrelated scvi fits must not leak into a new model.
    obs = obs.drop(
        columns=[c for c in obs if c.startswith("_scvi_") or c == "_indices"]
    )
    if any(c.startswith("_sbt_c2l_") for c in obs):
        raise ValueError(
            f"{label}: input already contains reserved _sbt_c2l_ columns; use the original counts input"
        )
    result = ad.AnnData(sparse.csr_matrix(matrix, dtype=np.float32), obs=obs, var=var)
    if "spatial" in source.obsm:
        result.obsm["spatial"] = np.asarray(source.obsm["spatial"]).copy()
    if "spatial" in source.uns:
        result.uns["spatial"] = copy.deepcopy(source.uns["spatial"])
    return result


def _exclude_mitochondrial(
    adata: ad.AnnData, settings: Cell2locationConfig
) -> ad.AnnData:
    if not settings.mitochondrial_prefixes:
        return adata
    if settings.gene_symbol_key:
        if settings.gene_symbol_key not in adata.var:
            raise ValueError(f"Missing gene_symbol_key {settings.gene_symbol_key!r}")
        symbols = _string_values(adata.var[settings.gene_symbol_key], "gene symbols")
    else:
        symbols = adata.var_names.to_numpy()
    mask = np.array(
        [str(g).startswith(tuple(settings.mitochondrial_prefixes)) for g in symbols]
    )
    adata.obs["_sbt_c2l_mito_counts"] = np.asarray(adata.X[:, mask].sum(axis=1)).ravel()
    return adata[:, ~mask].copy()


def _require_nonzero_rows(adata: ad.AnnData, label: str) -> None:
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    if adata.n_vars == 0 or np.any(totals <= 0):
        raise ValueError(
            f"{label}: zero-count cells/spots after filtering; curate these explicitly before fitting"
        )
    adata.obs["_sbt_c2l_total_counts"] = totals


def prepare_reference(source: ad.AnnData, settings: Cell2locationConfig) -> ad.AnnData:
    """Select raw counts, validate annotations and apply the upstream tutorial filter."""
    result = _count_copy(
        source,
        settings.reference_counts_layer,
        settings.reference_gene_id_key,
        "reference",
    )
    if not result.obs_names.is_unique:
        raise ValueError("Reference obs_names must be unique")
    keys = [
        settings.reference_labels_key,
        *settings.reference_categorical_covariate_keys,
    ]
    if settings.reference_batch_key:
        keys.append(settings.reference_batch_key)
    for key in keys:
        if key not in result.obs:
            raise ValueError(f"Reference obs is missing {key!r}")
        result.obs[key] = pd.Categorical(
            _string_values(result.obs[key], f"reference {key}")
        )
    result = _exclude_mitochondrial(result, settings)
    n_cells = np.asarray((result.X > 0).sum(axis=0)).ravel()
    means = np.divide(
        np.asarray(result.X.sum(axis=0)).ravel(),
        n_cells,
        out=np.zeros(result.n_vars),
        where=n_cells > 0,
    )
    # Exactly the upstream filter, evaluated in linear space to avoid log(0).
    keep = n_cells > 0
    if settings.filter_reference_genes:
        keep &= (n_cells > result.n_obs * settings.cell_percentage_cutoff) | (
            (n_cells > settings.cell_count_cutoff)
            & (means > settings.nonzero_mean_cutoff)
        )
    result.var["n_cells"] = n_cells
    result.var["nonz_mean"] = means
    result = result[:, keep].copy()
    _require_nonzero_rows(result, "Reference")
    result.uns["sbt_cell2location"] = {
        "counts_source": settings.reference_counts_layer or "X",
        "input_genes": source.n_vars,
        "retained_genes": result.n_vars,
    }
    return result


def prepare_visium(
    sources: Sequence[ad.AnnData],
    settings: Cell2locationConfig,
    *,
    library_ids: Sequence[str | None] | None = None,
) -> ad.AnnData:
    """Combine Visium libraries on shared genes, preserving slide metadata and identities.

    A library may occur in only one file. Composite output IDs encode both the
    library and original barcode; the original identity is retained in obs.
    """
    if not sources:
        raise ValueError("At least one Visium input is required")
    ids = list(library_ids) if library_ids is not None else [None] * len(sources)
    if len(ids) != len(sources):
        raise ValueError("library_ids must have one entry per input")
    parts: list[ad.AnnData] = []
    seen: set[str] = set()
    spatial_metadata: dict[str, Any] = {}
    for number, (source, explicit_id) in enumerate(zip(sources, ids)):
        part = _count_copy(
            source,
            settings.visium_counts_layer,
            settings.visium_gene_id_key,
            f"Visium input {number}",
        )
        key = settings.library_key
        if explicit_id is not None:
            if key in part.obs:
                existing = set(_string_values(part.obs[key], key))
                if existing != {str(explicit_id)}:
                    raise ValueError(
                        f"Explicit library_id {explicit_id!r} disagrees with obs[{key!r}]"
                    )
            part.obs[key] = str(explicit_id)
        if key not in part.obs:
            raise ValueError(
                f"Visium input {number}: provide library_id or obs[{key!r}]"
            )
        libraries = _string_values(part.obs[key], key)
        unique = set(libraries)
        if seen.intersection(unique):
            raise ValueError(
                f"Libraries appear in multiple files: {sorted(seen.intersection(unique))}"
            )
        seen.update(unique)
        part.obs[key] = libraries
        if settings.barcode_key and settings.barcode_key not in part.obs:
            raise ValueError(f"Missing barcode_key {settings.barcode_key!r}")
        barcodes = _string_values(
            part.obs[settings.barcode_key] if settings.barcode_key else part.obs_names,
            "spot barcodes",
        )
        part.obs[BARCODE] = barcodes
        part.obs[SOURCE_OBS] = part.obs_names.astype(str)
        composite = [
            f"{quote(lib, safe='')}::{quote(bc, safe='')}"
            for lib, bc in zip(libraries, barcodes)
        ]
        if not pd.Index(composite).is_unique:
            raise ValueError("Duplicate (library, barcode) spot identities")
        part.obs_names = composite
        coords = part.obsm.get("spatial")
        if (
            coords is None
            or coords.shape != (part.n_obs, 2)
            or not np.isfinite(coords).all()
        ):
            raise ValueError(
                "Visium requires finite obsm['spatial'] coordinates with shape (n_spots, 2)"
            )
        tissue = settings.in_tissue_key
        if tissue and tissue in part.obs:
            if not part.obs[tissue].isin([0, 1, False, True]).all():
                raise ValueError(f"{tissue} must contain only 0/1 values")
            part = part[part.obs[tissue].eq(1)].copy()
        if part.n_obs == 0 or set(part.obs[key]) != unique:
            raise ValueError("Tissue filtering removed every spot from a library")
        metadata = part.uns.get("spatial", {})
        if metadata:
            if len(unique) == 1 and len(metadata) == 1:
                metadata = {next(iter(unique)): next(iter(metadata.values()))}
            elif set(metadata) != unique:
                raise ValueError(
                    "uns['spatial'] keys must match the library IDs for combined inputs"
                )
            spatial_metadata.update(metadata)
        part = _exclude_mitochondrial(part, settings)
        parts.append(part)
    result = ad.concat(parts, join="inner", merge="first", index_unique=None)
    # Gene intersection must not discard library-specific observation metadata.
    # Missing prior columns become NaN and are rejected by prior validation.
    result.obs = (
        pd.concat([part.obs for part in parts], axis=0, sort=False)
        .loc[result.obs_names]
        .copy()
    )
    result.uns["spatial"] = spatial_metadata
    result.obs[settings.library_key] = pd.Categorical(result.obs[settings.library_key])
    _require_nonzero_rows(result, "Visium")
    result.uns["sbt_cell2location"] = {
        "counts_source": settings.visium_counts_layer or "X",
        "library_key": settings.library_key,
        "input_genes_per_file": np.asarray([p.n_vars for p in parts]),
        "shared_input_genes": result.n_vars,
    }
    return result


def validate_signatures(signatures: pd.DataFrame) -> pd.DataFrame:
    result = signatures.copy()
    result.index = _string_values(result.index, "signature genes")
    result.columns = _string_values(result.columns, "signature cell types")
    if not result.index.is_unique or not result.columns.is_unique or result.empty:
        raise ValueError(
            "Signatures require unique gene IDs and cell types and must not be empty"
        )
    values = result.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values.sum(axis=0) <= 0).any()
    ):
        raise ValueError(
            "Signatures must be finite and nonnegative with positive expression for every cell type"
        )
    return result.astype(np.float32)


def align_mapping_genes(
    visium: ad.AnnData,
    signatures: pd.DataFrame,
    *,
    min_shared_genes: int = 100,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Align both modalities in identical order, excluding globally unexpressed genes."""
    signatures = validate_signatures(signatures)
    expressed = np.asarray(visium.X.sum(axis=0)).ravel() > 0
    genes = visium.var_names[expressed & visium.var_names.isin(signatures.index)]
    genes = genes[signatures.loc[genes].sum(axis=1).to_numpy() > 0]
    if len(genes) < min_shared_genes:
        raise ValueError(
            f"Only {len(genes)} shared expressed genes; require {min_shared_genes}. Check gene ID namespaces."
        )
    result = visium[:, genes].copy()
    _require_nonzero_rows(result, "Mapping")
    aligned = validate_signatures(signatures.loc[genes])
    result.uns.setdefault("sbt_cell2location", {})["mapping_genes"] = len(genes)
    return result, aligned


def attach_cell_count_prior(
    visium: ad.AnnData,
    settings: Cell2locationConfig,
    *,
    counts: pd.DataFrame | None = None,
) -> ad.AnnData:
    """Align registered IMC spot counts and return a copy with audited Gamma means.

    Table columns are ``library_id``, ``barcode``, ``n_cells``. Extra table rows
    (e.g. excluded off-tissue spots) are recorded; missing or duplicate retained
    spot keys fail. Zero counts use an explicit positive floor, never a hard zero.
    """
    result = visium.copy()
    if settings.cell_count_prior_csv and counts is None:
        raise ValueError(
            "Pass the configured cell-count CSV as a counts DataFrame to the notebook API"
        )
    if counts is not None and settings.cell_count_prior_obs_key:
        raise ValueError("Provide either a count table or an obs prior column")
    extras = 0
    if counts is not None:
        required = {"library_id", "barcode", "n_cells"}
        if not required.issubset(counts):
            raise ValueError(f"Cell-count table requires columns {sorted(required)}")
        table = counts.copy()
        for key in ["library_id", "barcode"]:
            table[key] = _string_values(table[key], f"prior {key}")
        if table.duplicated(["library_id", "barcode"]).any():
            raise ValueError(
                "Cell-count table has duplicate (library_id, barcode) keys"
            )
        table = table.set_index(["library_id", "barcode"])
        index = pd.MultiIndex.from_arrays(
            [result.obs[settings.library_key].astype(str), result.obs[BARCODE]]
        )
        missing = index.difference(table.index)
        if len(missing):
            raise ValueError(
                f"Cell-count prior missing {len(missing)} spots, e.g. {missing[:3].tolist()}"
            )
        raw = table.reindex(index)["n_cells"].to_numpy(dtype=float)
        extras = len(table.index.difference(index))
    elif settings.cell_count_prior_obs_key:
        key = settings.cell_count_prior_obs_key
        if key not in result.obs:
            raise ValueError(f"Missing cell-count prior obs column {key!r}")
        raw = result.obs[key].to_numpy(dtype=float)
    else:
        result.obs.drop(columns=[PRIOR_MEAN, PRIOR_RAW], errors="ignore", inplace=True)
        audit = result.uns.get("sbt_cell2location", {})
        for key in list(audit):
            if key.startswith("prior_"):
                del audit[key]
        return result
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Cell-count priors must be finite and nonnegative")
    scaled = raw * settings.cell_count_prior_scale
    means = np.maximum(scaled, settings.cell_count_prior_floor)
    if not np.isfinite(means).all():
        raise ValueError("Cell-count prior scaling overflowed")
    result.obs[PRIOR_RAW] = raw
    result.obs[PRIOR_MEAN] = means.astype(np.float32)
    result.uns.setdefault("sbt_cell2location", {}).update(
        {
            "prior_source": "table"
            if counts is not None
            else str(settings.cell_count_prior_obs_key),
            "prior_scale": settings.cell_count_prior_scale,
            "prior_floor": settings.cell_count_prior_floor,
            "prior_floored_spots": int(
                (scaled < settings.cell_count_prior_floor).sum()
            ),
            "prior_extra_table_rows": extras,
            "prior_mean_var_ratio": settings.cell_count_prior_mean_var_ratio,
        }
    )
    return result


def _runtime(settings: Cell2locationConfig) -> None:
    import scvi
    import torch

    installed = importlib.metadata.version("cell2location")
    if installed != "0.1.5":
        raise RuntimeError(
            f"SBT cell2location integration targets 0.1.5; found {installed}. Use its dedicated environment."
        )
    if settings.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU was requested but is unavailable; select accelerator=cpu explicitly for CPU work"
        )
    scvi.settings.seed = settings.seed


def _provenance(
    adata: ad.AnnData, settings: Cell2locationConfig, operation: str
) -> None:
    versions = {
        p: importlib.metadata.version(p)
        for p in ["cell2location", "scvi-tools", "torch", "pyro-ppl", "anndata"]
    }
    adata.uns.setdefault("sbt_cell2location", {}).update(
        {
            "operation": operation,
            "settings_json": settings.model_dump_json(),
            "versions": versions,
        }
    )


def extract_reference_signatures(adata: ad.AnnData) -> pd.DataFrame:
    """Extract the posterior mean signature in the upstream factor ordering."""
    factors = list(adata.uns["mod"]["factor_names"])
    columns = [f"means_per_cluster_mu_fg_{name}" for name in factors]
    if "means_per_cluster_mu_fg" in adata.varm:
        matrix = adata.varm["means_per_cluster_mu_fg"]
        signatures = (
            matrix.loc[:, columns].copy()
            if isinstance(matrix, pd.DataFrame)
            else pd.DataFrame(matrix, index=adata.var_names, columns=columns)
        )
    else:
        signatures = adata.var.loc[:, columns].copy()
    signatures.columns = factors
    return validate_signatures(signatures)


def fit_reference(
    source: ad.AnnData, settings: Cell2locationConfig
) -> Cell2locationResult:
    """Fit the NB regression model and export reusable cell-type expression signatures."""
    from cell2location.models import RegressionModel

    prepared = prepare_reference(source, settings)
    _runtime(settings)
    RegressionModel.setup_anndata(
        prepared,
        labels_key=settings.reference_labels_key,
        batch_key=settings.reference_batch_key,
        categorical_covariate_keys=settings.reference_categorical_covariate_keys
        or None,
    )
    model = RegressionModel(prepared)
    model.train(
        max_epochs=settings.reference_max_epochs,
        batch_size=settings.reference_batch_size,
        train_size=1,
        accelerator=settings.accelerator,
    )
    posterior = model.export_posterior(
        prepared,
        sample_kwargs={
            "num_samples": settings.posterior_samples,
            "batch_size": settings.posterior_batch_size,
            "accelerator": settings.accelerator,
            "device": "auto",
        },
    )
    _provenance(posterior, settings, "reference")
    return Cell2locationResult(
        model, posterior, extract_reference_signatures(posterior)
    )


def fit_mapping(
    visium: ad.AnnData,
    signatures: pd.DataFrame,
    settings: Cell2locationConfig,
    *,
    counts: pd.DataFrame | None = None,
) -> Cell2locationResult:
    """Map prepared Visium counts jointly using per-library detection and optional spot priors."""
    from cell2location.models import Cell2location

    prepared, aligned = align_mapping_genes(
        visium, signatures, min_shared_genes=settings.min_shared_genes
    )
    prepared = attach_cell_count_prior(prepared, settings, counts=counts)
    _runtime(settings)
    Cell2location.setup_anndata(prepared, batch_key=settings.library_key)
    kwargs: dict[str, Any] = {
        "N_cells_per_location": settings.n_cells_per_location,
        "detection_alpha": settings.detection_alpha,
        "detection_mean_per_sample": settings.detection_mean_per_sample,
        "N_cells_mean_var_ratio": settings.cell_count_prior_mean_var_ratio,
    }
    if PRIOR_MEAN in prepared.obs:
        from SpatialBiologyToolkit.cell2location_prior import SpotCellCountModel

        prior = prepared.obs[PRIOR_MEAN].to_numpy(dtype=np.float32)
        kwargs.update(
            model_class=SpotCellCountModel,
            spot_cell_count_prior=prior,
            N_cells_per_location=float(prior.mean()),
        )
    model = Cell2location(prepared, cell_state_df=aligned, **kwargs)
    if PRIOR_MEAN in prepared.obs:
        # Upstream initialization divides every library's total RNA by one global
        # count. Match each library's expected density when initializing detection.
        batch = prepared.obs["_scvi_batch"].to_numpy(dtype=int)
        totals = np.asarray(prepared.X.sum(axis=1)).ravel()
        prior = prepared.obs[PRIOR_MEAN].to_numpy()
        sc_total = float(aligned.sum(axis=0).mean())
        if settings.detection_mean_per_sample:
            detection = np.array(
                [
                    totals[batch == b].mean() / prior[batch == b].mean() / sc_total
                    for b in range(model.summary_stats["n_batch"])
                ],
                dtype=np.float32,
            ).reshape(-1, 1)
        else:
            detection = np.asarray(
                totals.mean() / prior.mean() / sc_total, dtype=np.float32
            )
        import torch

        module = model.module.model
        module.detection_mean_hyp_prior_beta.copy_(
            module.detection_mean_hyp_prior_alpha / torch.as_tensor(detection)
        )
        model.detection_mean_ = detection
    model.train(
        max_epochs=settings.mapping_max_epochs,
        batch_size=settings.mapping_batch_size,
        train_size=1,
        accelerator=settings.accelerator,
    )
    posterior = model.export_posterior(
        prepared,
        sample_kwargs={
            "num_samples": settings.posterior_samples,
            "batch_size": settings.posterior_batch_size,
            "accelerator": settings.accelerator,
            "device": "auto",
        },
    )
    _provenance(posterior, settings, "map")
    return Cell2locationResult(model, posterior, aligned)


def save_result(result: Cell2locationResult, destination: str | Path) -> Path:
    """Save upstream reloadable model, posterior counts and signatures in a new directory.

    The caller supplies the final location. Existing directories are never replaced.
    A failed save retains a sibling temporary directory for inspection/recovery.
    """
    import tempfile

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(
            f"Result already exists: {destination}; select a new asset_folder"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        result.model.save(str(temporary / "model"), overwrite=False)
        result.adata.write_h5ad(temporary / "posterior.h5ad", compression="gzip")
        if result.signatures is not None:
            result.signatures.to_csv(
                temporary / "signatures.csv", index_label="gene_id"
            )
        temporary.rename(destination)
    except BaseException:
        LOGGER.error("Result save failed; partial files retained at %s", temporary)
        raise
    return destination


def read_signatures(path: str | Path) -> pd.DataFrame:
    """Read signatures with gene IDs preserved as strings."""
    frame = pd.read_csv(path, index_col=0, dtype={0: str})
    return validate_signatures(frame)
