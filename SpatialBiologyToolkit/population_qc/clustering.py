"""In-memory clustering experiments and structural QC wrappers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any
import warnings

import numpy as np
import pandas as pd

from SpatialBiologyToolkit.config.models import PopulationEmbeddingQCConfig
from SpatialBiologyToolkit.population_embedding_qc import (
    PopulationEmbeddingQCResult,
    run_population_embedding_qc,
)

from ._utils import (
    clean_key,
    infer_case_key,
    resolve_roi_key,
    resolve_table,
    validate_markers,
    validate_population,
)
from .models import InMemoryClusteringResult, SubclusteringResult


_RESOLUTION_PATTERN = r"(?P<resolution>[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"


def _prefixed_sweep_regex(prefix: str) -> str:
    return rf"^{re.escape(prefix)}_{_RESOLUTION_PATTERN}$"


def assess_clustering(
    data: Any,
    population_key: str,
    *,
    table_name: str | None = None,
    mode: str = "auto",
    sweep_columns: Sequence[str] | None = None,
    sweep_regex: str = r"^leiden_(?P<resolution>\d+(?:\.\d+)?)$",
    reference_resolution: float | None = None,
    roi_key: str | None = None,
    case_key: str | None = None,
    umap_key: str = "X_umap",
    pca_key: str | None = None,
    connectivities_key: str | None = None,
    config: PopulationEmbeddingQCConfig | None = None,
) -> PopulationEmbeddingQCResult:
    """Assess structural support for an existing or newly created clustering.

    This wraps the established population embedding QC implementation. It reads
    existing graph, PCA, UMAP, and sweep state without recomputing or mutating it.
    For a local subclustering result, pass ``result.adata``; the wrapper uses its
    ``X_population_qc`` representation when available.

    Agent guidance
    --------------
    Use once for the global clustering and after each serious candidate split.
    Start with graph and PCA/local-feature evidence; use UMAP as supporting visual
    evidence because its geometry is distorted. High concern identifies clusters
    needing investigation, not biologically invalid clusters. Cache this result
    on multi-million-cell data and follow ambiguity with expression, case/ROI,
    and image evidence.
    """

    _, adata = resolve_table(data, table_name)
    selected_roi = resolve_roi_key(data, adata, roi_key)
    selected_case = infer_case_key(adata, case_key)
    selected_pca = pca_key or (
        "X_population_qc" if "X_population_qc" in adata.obsm else "X_pca"
    )
    settings = config or PopulationEmbeddingQCConfig(
        population_obs=population_key,
        roi_obs=selected_roi,
        sample_obs=selected_case,
    )
    if config is not None:
        updates: dict[str, Any] = {}
        if selected_roi is not None:
            updates["roi_obs"] = selected_roi
        if selected_case is not None:
            updates["sample_obs"] = selected_case
        if updates:
            settings = config.model_copy(update=updates)
    return run_population_embedding_qc(
        adata,
        population_obs=population_key,
        mode=mode,
        sweep_columns=list(sweep_columns) if sweep_columns is not None else None,
        sweep_regex=sweep_regex,
        reference_resolution=reference_resolution,
        umap_key=umap_key,
        pca_key=selected_pca,
        connectivities_key=connectivities_key,
        config=settings,
    )


def assess_candidate_clustering(
    candidate: SubclusteringResult,
    *,
    reference_column: str | None = None,
    config: PopulationEmbeddingQCConfig | None = None,
) -> PopulationEmbeddingQCResult:
    """Run structural QC directly on an in-memory subclustering experiment.

    Parameters
    ----------
    candidate
        Result returned by :func:`subcluster_population`. Its copied local
        AnnData object, local graph, representation, and candidate columns are
        reused; the parent SpatialData table is not changed by this function.
    reference_column
        Candidate column to treat as the main partition. By default the middle
        requested resolution is used, which avoids silently favouring either
        extreme of a sweep.
    config
        Optional advanced settings for the established population embedding QC
        implementation.

    Agent guidance
    --------------
    Call this immediately after a plausible subclustering run. Use the result to
    identify structurally weak children and unstable splits, then test those
    candidates with marker profiles, sample/ROI representation, and backgated
    images. Passing structural QC does not by itself justify a biological split.

    Notes
    -----
    This is a read-only assessment of ``candidate.adata``. It neither attaches
    further columns to the source table nor writes to disk.
    """

    if not candidate.columns:
        raise ValueError("candidate does not contain any clustering columns")
    selected = reference_column or candidate.columns[(len(candidate.columns) - 1) // 2]
    if selected not in candidate.columns:
        raise ValueError(
            f"reference_column must be one of the candidate columns: {list(candidate.columns)}"
        )
    sweep_regex = str(
        candidate.parameters.get(
            "sweep_regex",
            r"^.+_(?P<resolution>[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$",
        )
    )
    has_sweep = len(candidate.columns) >= 2
    return assess_clustering(
        candidate.adata,
        selected,
        mode="sweep" if has_sweep else "single",
        sweep_columns=candidate.columns if has_sweep else None,
        sweep_regex=sweep_regex,
        pca_key="X_population_qc",
        config=config,
    )


def _leiden_defaults(
    leiden_kwargs: Mapping[str, Any] | None, random_state: int
) -> dict[str, Any]:
    kwargs = dict(leiden_kwargs or {})
    kwargs.setdefault("random_state", random_state)
    # Scanpy's igraph RNG adapter can request uint32 values from NumPy's
    # Windows int32 RandomState path and flood stderr during interpreter exit.
    # The sbt environment includes leidenalg, whose explicit seed path avoids
    # that adapter while remaining deterministic.
    kwargs.setdefault("flavor", "leidenalg")
    kwargs.setdefault("n_iterations", 2)
    kwargs.setdefault("directed", False)
    return kwargs


def _run_leiden(scanpy: Any, adata: Any, **kwargs: Any) -> None:
    """Run Leiden while silencing Scanpy's known backend-transition notice."""

    with warnings.catch_warnings():
        if kwargs.get("flavor") == "leidenalg":
            warnings.filterwarnings(
                "ignore",
                message=r"In the future, the default backend for leiden will be igraph.*",
                category=FutureWarning,
            )
        scanpy.tl.leiden(adata, **kwargs)


def create_leiden_sweep(
    data: Any,
    resolutions: Sequence[float],
    *,
    table_name: str | None = None,
    output_prefix: str = "agent_leiden",
    neighbors_key: str | None = None,
    random_state: int = 0,
    copy_table: bool = False,
    overwrite: bool = False,
    leiden_kwargs: Mapping[str, Any] | None = None,
) -> InMemoryClusteringResult:
    """Create new Leiden columns from an existing in-memory neighbour graph.

    With ``copy_table=False`` the new columns are attached to the live table in
    memory. ``copy_table=True`` duplicates the AnnData table first, which can use
    substantial RAM. No Zarr or H5AD file is written in either mode.

    Agent guidance
    --------------
    Use to compare alternative partitions of the same scientifically justified
    graph. Do not treat resolution as a biological scale or choose it merely to
    obtain the expected number of populations. Compare structure, expression,
    sample representation, and cell images before preferring a candidate.
    """

    values = sorted(set(float(value) for value in resolutions))
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("resolutions must contain positive finite values")
    _, source = resolve_table(data, table_name)
    working = source.copy() if copy_table else source
    columns = tuple(f"{output_prefix}_{value:g}" for value in values)
    existing = [column for column in columns if column in working.obs]
    if existing and not overwrite:
        raise ValueError(
            f"Output columns already exist: {existing}. Pass overwrite=True or use another prefix."
        )
    if neighbors_key is None:
        if "neighbors" not in working.uns and "connectivities" not in working.obsp:
            raise KeyError("No default Scanpy neighbour graph is available")
    elif neighbors_key not in working.uns:
        raise KeyError(f"Named neighbour graph {neighbors_key!r} is missing from adata.uns")

    import scanpy as sc

    kwargs = _leiden_defaults(leiden_kwargs, random_state)
    size_records: list[dict[str, Any]] = []
    for resolution, column in zip(values, columns, strict=True):
        _run_leiden(
            sc,
            working,
            resolution=resolution,
            key_added=column,
            neighbors_key=neighbors_key,
            **kwargs,
        )
        counts = working.obs[column].astype(str).value_counts()
        size_records.extend(
            {
                "column": column,
                "resolution": resolution,
                "population": str(population),
                "cells": int(count),
            }
            for population, count in counts.items()
        )
    provenance = {
        "kind": "leiden_sweep",
        "columns": list(columns),
        "resolutions": values,
        "sweep_regex": _prefixed_sweep_regex(output_prefix),
        "neighbors_key": neighbors_key,
        "random_state": random_state,
        "leiden_flavor": kwargs.get("flavor"),
        "leiden_n_iterations": kwargs.get("n_iterations"),
    }
    working.uns.setdefault("population_qc", {}).setdefault("clustering_runs", []).append(
        provenance
    )
    return InMemoryClusteringResult(
        adata=working,
        columns=columns,
        cluster_sizes=pd.DataFrame(size_records),
        copied=copy_table,
        parameters=provenance,
    )


def subcluster_population(
    data: Any,
    population_key: str,
    population: Any,
    *,
    resolutions: Sequence[float] = (0.3, 0.6, 1.0),
    markers: Sequence[str] | None = None,
    use_rep: str | None = None,
    table_name: str | None = None,
    output_prefix: str | None = None,
    n_neighbors: int = 15,
    n_pcs: int = 30,
    compute_umap: bool = True,
    attach: bool = True,
    overwrite: bool = False,
    random_state: int = 0,
    neighbors_kwargs: Mapping[str, Any] | None = None,
    leiden_kwargs: Mapping[str, Any] | None = None,
) -> SubclusteringResult:
    """Create candidate subclusters using a copied local population subset.

    ``markers`` builds a local PCA/graph from exact marker values. Alternatively,
    ``use_rep`` copies an existing integrated representation into the subset. When
    both are omitted, ``X_pca`` is preferred and ``X`` is the fallback. Candidate
    columns retain original labels outside the parent when attached to the source.

    Agent guidance
    --------------
    Use only after evidence suggests a parent may contain multiple phenotypes.
    Choose markers that distinguish plausible children and record that choice.
    QC ``result.adata`` locally, then compare child marker profiles, case/ROI
    support, and backgated images. A visually separated UMAP island or a larger
    number of clusters is not sufficient evidence for a biological split.

    Notes
    -----
    Graph, UMAP, and candidate Leiden columns are created on a copied subset.
    ``attach=True`` changes only source ``obs`` columns and provenance in memory;
    it never writes the SpatialData store. Set ``compute_umap=False`` only when
    no immediate structural QC or local visualization is required; inherited
    global UMAP coordinates are removed so they cannot be mistaken for a locally
    recomputed embedding.

    ``neighbors_kwargs`` can supply focused Scanpy neighbour options such as a
    distance metric or transformer.  The toolkit owns ``n_neighbors``, ``use_rep``,
    and ``random_state`` so those keys are rejected in this mapping.
    """

    if markers is not None and use_rep is not None:
        raise ValueError("markers and use_rep are alternative feature selections")
    values = sorted(set(float(value) for value in resolutions))
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("resolutions must contain positive finite values")
    if n_neighbors < 2 or n_pcs < 2:
        raise ValueError("n_neighbors and n_pcs must be at least 2")
    _, source = resolve_table(data, table_name)
    mask = validate_population(source, population_key, population)
    subset = source[mask.to_numpy()].copy()
    if subset.n_obs < 3:
        raise ValueError("At least three cells are required for subclustering")
    # A sliced global UMAP is not a local subclustering embedding. Remove it so
    # downstream QC cannot silently use stale coordinates when UMAP is disabled.
    if "X_umap" in subset.obsm:
        del subset.obsm["X_umap"]
    prefix = output_prefix or f"{clean_key(population_key)}_{clean_key(population)}_subcluster"
    columns = tuple(f"{prefix}_{value:g}" for value in values)
    collisions = [column for column in columns if column in source.obs or column in subset.obs]
    if collisions and not overwrite:
        raise ValueError(
            f"Output columns already exist: {collisions}. Pass overwrite=True or use another prefix."
        )

    import scanpy as sc

    selected_markers: list[str] | None = None
    warnings: list[str] = []
    source_representation = use_rep
    if markers is not None:
        selected_markers = validate_markers(subset, markers)
        work = subset[:, selected_markers].copy()
        components = min(int(n_pcs), work.n_vars - 1, work.n_obs - 1)
        if components >= 2:
            sc.pp.pca(work, n_comps=components, random_state=random_state)
            local_representation = np.asarray(work.obsm["X_pca"])
        else:
            local_representation = (
                work.X.toarray() if hasattr(work.X, "toarray") else np.asarray(work.X)
            )
        subset.obsm["X_population_qc"] = np.asarray(local_representation)
    elif use_rep is not None:
        if use_rep not in subset.obsm:
            raise KeyError(f"Representation {use_rep!r} is missing from adata.obsm")
        subset.obsm["X_population_qc"] = np.asarray(subset.obsm[use_rep]).copy()
    elif "X_pca" in subset.obsm:
        subset.obsm["X_population_qc"] = np.asarray(subset.obsm["X_pca"]).copy()
        source_representation = "X_pca"
    else:
        subset.obsm["X_population_qc"] = (
            subset.X.toarray() if hasattr(subset.X, "toarray") else np.asarray(subset.X)
        )
        warnings.append("No markers or representation supplied; local graph uses all values in X")

    neighbour_options = dict(neighbors_kwargs or {})
    reserved_neighbour_keys = {"n_neighbors", "use_rep", "random_state"}
    conflicts = sorted(reserved_neighbour_keys & set(neighbour_options))
    if conflicts:
        raise ValueError(
            "neighbors_kwargs cannot override toolkit-owned values: "
            + ", ".join(conflicts)
        )
    sc.pp.neighbors(
        subset,
        n_neighbors=min(int(n_neighbors), subset.n_obs - 1),
        use_rep="X_population_qc",
        random_state=random_state,
        **neighbour_options,
    )
    if compute_umap:
        sc.tl.umap(subset, random_state=random_state)
    kwargs = _leiden_defaults(leiden_kwargs, random_state)
    size_records: list[dict[str, Any]] = []
    parent_text = str(population)
    for resolution, column in zip(values, columns, strict=True):
        _run_leiden(sc, subset, resolution=resolution, key_added=column, **kwargs)
        prefixed = parent_text + "__" + subset.obs[column].astype(str)
        subset.obs[column] = pd.Categorical(prefixed)
        counts = subset.obs[column].astype(str).value_counts()
        size_records.extend(
            {
                "column": column,
                "resolution": resolution,
                "population": str(candidate),
                "cells": int(count),
            }
            for candidate, count in counts.items()
        )
        if attach:
            attached = source.obs[population_key].astype("string").copy()
            attached.loc[subset.obs_names] = subset.obs[column].astype(str).to_numpy()
            source.obs[column] = pd.Categorical(attached)
    provenance = {
        "kind": "population_subclustering",
        "parent_population_key": population_key,
        "parent_population": parent_text,
        "columns": list(columns),
        "resolutions": values,
        "sweep_regex": _prefixed_sweep_regex(prefix),
        "markers": selected_markers,
        "source_representation": source_representation,
        "local_representation": "X_population_qc",
        "n_neighbors": n_neighbors,
        "neighbors_kwargs": neighbour_options,
        "random_state": random_state,
        "leiden_flavor": kwargs.get("flavor"),
        "leiden_n_iterations": kwargs.get("n_iterations"),
        "attached_to_source": attach,
    }
    subset.uns.setdefault("population_qc", {}).setdefault("clustering_runs", []).append(
        provenance
    )
    if attach:
        source.uns.setdefault("population_qc", {}).setdefault("clustering_runs", []).append(
            provenance
        )
    return SubclusteringResult(
        adata=subset,
        columns=columns,
        cluster_sizes=pd.DataFrame(size_records),
        copied=True,
        parameters=provenance,
        warnings=tuple(warnings),
        parent_population_key=population_key,
        parent_population=parent_text,
        attached_to_source=attach,
    )


def apply_population_mapping(
    data: Any,
    source_key: str,
    mapping: Mapping[Any, Any],
    output_key: str,
    *,
    table_name: str | None = None,
    keep_unmapped: bool = True,
    copy_table: bool = False,
    overwrite: bool = False,
) -> InMemoryClusteringResult:
    """Create a reviewable in-memory relabel or merge annotation column.

    Agent guidance
    --------------
    Use only after recording the evidence for a proposed relabel or merge. Always
    write to a clearly named candidate column and compare it with the source
    clustering. The source column is never overwritten and no file is saved.
    """

    if output_key == source_key:
        raise ValueError("output_key must differ from source_key")
    _, source = resolve_table(data, table_name)
    if source_key not in source.obs:
        raise KeyError(f"Source population column {source_key!r} is missing")
    if output_key in source.obs and not overwrite:
        raise ValueError(f"Output column {output_key!r} already exists")
    working = source.copy() if copy_table else source
    existing = set(working.obs[source_key].dropna().astype(str))
    normalised = {str(key): str(value) for key, value in mapping.items()}
    missing = sorted(set(normalised) - existing)
    if missing:
        raise KeyError(f"Mapping contains labels absent from {source_key!r}: {missing}")
    values = working.obs[source_key].astype("string")
    mapped = values.map(normalised)
    if keep_unmapped:
        mapped = mapped.fillna(values)
    working.obs[output_key] = pd.Categorical(mapped)
    counts = (
        working.obs[output_key]
        .astype(str)
        .value_counts()
        .rename_axis("population")
        .rename("cells")
        .reset_index()
    )
    counts.insert(0, "column", output_key)
    provenance = {
        "kind": "population_mapping",
        "columns": [output_key],
        "source_key": source_key,
        "mapping": normalised,
        "keep_unmapped": keep_unmapped,
    }
    working.uns.setdefault("population_qc", {}).setdefault("clustering_runs", []).append(
        provenance
    )
    return InMemoryClusteringResult(
        adata=working,
        columns=(output_key,),
        cluster_sizes=counts,
        copied=copy_table,
        parameters=provenance,
    )


def discard_population_qc_columns(
    data: Any,
    columns: Sequence[str] | None = None,
    *,
    table_name: str | None = None,
) -> tuple[str, ...]:
    """Remove tracked candidate columns from a live table in memory.

    Only columns recorded under ``adata.uns['population_qc']['clustering_runs']``
    can be removed. Passing ``columns=None`` removes every currently attached
    tracked column. The source clustering and the on-disk SpatialData store are
    untouched.

    Agent guidance
    --------------
    Use after a candidate has been rejected or superseded, once its evidence and
    rationale have been captured in the notebook. This keeps the live table clear
    without erasing the audit narrative.
    """

    _, adata = resolve_table(data, table_name)
    qc_state = adata.uns.get("population_qc", {})
    runs = qc_state.get("clustering_runs", []) if isinstance(qc_state, Mapping) else []
    tracked = {
        str(column)
        for run in runs
        if isinstance(run, Mapping)
        for column in run.get("columns", [])
    }
    selected = sorted(tracked & set(adata.obs.columns)) if columns is None else list(map(str, columns))
    untracked = [column for column in selected if column not in tracked]
    if untracked:
        raise ValueError(f"Refusing to remove untracked columns: {untracked}")
    for column in selected:
        if column in adata.obs:
            del adata.obs[column]
    return tuple(selected)


__all__ = [
    "apply_population_mapping",
    "assess_candidate_clustering",
    "assess_clustering",
    "create_leiden_sweep",
    "discard_population_qc_columns",
    "subcluster_population",
]
