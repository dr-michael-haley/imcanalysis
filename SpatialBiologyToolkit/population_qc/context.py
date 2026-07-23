"""Dataset inspection for agent-guided population quality control."""

from __future__ import annotations

from typing import Any

import pandas as pd

from SpatialBiologyToolkit.population_embedding_qc.inspection import detect_sweep_columns

from ._utils import infer_case_key, resolve_roi_key, resolve_table, shape_tuple
from .models import MarkerExpectations, PopulationDataContext


def inspect_population_data(
    data: Any,
    population_key: str | None = None,
    *,
    table_name: str | None = None,
    roi_key: str | None = None,
    case_key: str | None = None,
    expectations: MarkerExpectations | None = None,
    sweep_regex: str = r"^leiden_(?P<resolution>\d+(?:\.\d+)?)$",
) -> PopulationDataContext:
    """Inspect population-QC capabilities without reading image pixels.

    Parameters
    ----------
    data
        SpatialData or AnnData. SpatialData is preferred because later tools can
        link population evidence to masks and marker images.
    population_key
        Population column to inspect. Leave unset to inventory likely columns.
    table_name, roi_key, case_key
        Optional table and annotation overrides. Common case spellings including
        ``animal`` and ``patient`` are inferred when ``case_key`` is omitted.
    expectations
        Optional marker expectations. Missing expected markers are reported as
        warnings here, allowing a tissue specification to include markers absent
        from a particular panel.
    sweep_regex
        Pattern used to detect numerical Leiden resolution columns.

    Returns
    -------
    PopulationDataContext
        Cell/marker counts, candidate annotations, resolution columns,
        representations, graphs, image coverage, and warnings.

    Agent guidance
    --------------
    Call this first. Verify exact marker names, the intended population column,
    graph/embedding availability, and whether conclusions can be checked across
    independent cases or only ROIs. This function performs no raster IO and is
    safe on a large lazy SpatialData store.
    """

    selected_table, adata = resolve_table(data, table_name)
    selected_roi = resolve_roi_key(data, adata, roi_key)
    selected_case = infer_case_key(adata, case_key)
    warnings: list[str] = []
    if population_key is not None and population_key not in adata.obs:
        raise KeyError(f"Population column {population_key!r} is missing from the table")

    candidate_keys: list[str] = []
    name_tokens = ("leiden", "cluster", "population", "label", "cell_type")
    for column in adata.obs.columns:
        series = adata.obs[column]
        name_match = any(token in str(column).casefold() for token in name_tokens)
        categorical = isinstance(series.dtype, pd.CategoricalDtype)
        if (name_match or categorical) and 1 < int(series.nunique(dropna=True)) <= 500:
            candidate_keys.append(str(column))

    detected_sweep, sweep_warnings = detect_sweep_columns(
        adata.obs, sweep_regex=sweep_regex, explicit_columns=None
    )
    warnings.extend(sweep_warnings)
    if selected_case is None:
        warnings.append(
            "No case/sample column was identified; biological replication can only be assessed at ROI level"
        )
    if selected_roi is None:
        warnings.append("No ROI column was identified")
    if "X_pca" not in adata.obsm:
        warnings.append("PCA coordinates are unavailable")
    if "X_umap" not in adata.obsm:
        warnings.append("UMAP coordinates are unavailable")
    if "connectivities" not in adata.obsp:
        warnings.append("A default connectivity graph is unavailable")
    expectation_model = MarkerExpectations.from_value(expectations)
    missing_expected = [
        marker for marker in expectation_model.markers if marker not in set(map(str, adata.var_names))
    ]
    if missing_expected:
        warnings.append("Expected markers absent from the panel: " + ", ".join(missing_expected))

    population_counts = pd.DataFrame(columns=["population", "cells", "fraction"])
    if population_key is not None:
        counts = adata.obs[population_key].dropna().astype(str).value_counts()
        population_counts = counts.rename_axis("population").rename("cells").reset_index()
        population_counts["fraction"] = population_counts["cells"] / max(
            1, int(population_counts["cells"].sum())
        )
    is_spatialdata = hasattr(data, "tables")
    return PopulationDataContext(
        table_name=selected_table,
        population_key=population_key,
        roi_key=selected_roi,
        case_key=selected_case,
        n_cells=int(adata.n_obs),
        n_markers=int(adata.n_vars),
        markers=tuple(map(str, adata.var_names)),
        obs_columns=tuple(map(str, adata.obs.columns)),
        population_counts=population_counts,
        candidate_population_keys=tuple(candidate_keys),
        leiden_sweep_columns=pd.DataFrame(detected_sweep, columns=["column", "resolution"]),
        representations={key: shape_tuple(value) for key, value in adata.obsm.items()},
        pairwise_matrices={key: shape_tuple(value) for key, value in adata.obsp.items()},
        image_elements=len(data.images) if is_spatialdata else 0,
        label_elements=len(data.labels) if is_spatialdata else 0,
        warnings=tuple(warnings),
    )


__all__ = ["inspect_population_data"]
