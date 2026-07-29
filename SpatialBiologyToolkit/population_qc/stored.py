"""Agent-facing access to cached population QC in AnnData or SpatialData."""

from __future__ import annotations

from typing import Any

import pandas as pd

from SpatialBiologyToolkit.config.models import PopulationEmbeddingQCConfig
from SpatialBiologyToolkit.population_embedding_qc.models import (
    PopulationEmbeddingQCResult,
)
from SpatialBiologyToolkit.population_embedding_qc.storage import (
    StoredPopulationQCError,
    focused_population_summary,
    list_stored_population_qc as _list_stored_population_qc,
    load_stored_population_qc as _load_stored_population_qc,
)

from ._utils import resolve_table


def list_stored_population_qc(
    data: Any,
    *,
    table_name: str | None = None,
) -> pd.DataFrame:
    """Inventory cached runs without performing any QC calculations."""
    _, adata = resolve_table(data, table_name)
    return _list_stored_population_qc(adata)


def has_stored_population_qc(
    data: Any,
    population_key: str,
    *,
    table_name: str | None = None,
) -> bool:
    """Return whether a compatible cached run exists for a population column."""
    inventory = list_stored_population_qc(data, table_name=table_name)
    if inventory.empty:
        return False
    return bool(
        (
            inventory["compatible"].astype(bool)
            & inventory["reference_column"].astype(str).eq(str(population_key))
        ).any()
    )


def load_stored_population_qc(
    data: Any,
    population_key: str,
    *,
    table_name: str | None = None,
    run_id: str | None = None,
    config: PopulationEmbeddingQCConfig | None = None,
    strict: bool = True,
) -> PopulationEmbeddingQCResult:
    """Load cached structural QC and never fall back to recalculation."""
    _, adata = resolve_table(data, table_name)
    return _load_stored_population_qc(
        adata,
        population_key=population_key,
        run_id=run_id,
        config=config,
        strict=strict,
    )


__all__ = [
    "StoredPopulationQCError",
    "focused_population_summary",
    "has_stored_population_qc",
    "list_stored_population_qc",
    "load_stored_population_qc",
]
