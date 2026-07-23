"""Typed, notebook-friendly results for population quality control.

The models keep numerical evidence, plotting data, and in-memory clustering
experiments explicit.  None of them persists AnnData or SpatialData.  Their
``to_agent_summary()`` methods intentionally omit large cell-level tables so a
future agent skill can request compact evidence without losing the full tables
needed for scientific review in Jupyter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


def _unique_strings(values: Sequence[Any] | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in (values or ())))


@dataclass(frozen=True)
class MarkerExpectations:
    """Expected marker directions for one proposed biological identity.

    Parameters
    ----------
    positive_markers
        Strong identity markers expected to be higher in the target population.
    supportive_markers
        Markers that support the identity but may describe a subtype, activation
        state, or only a fraction of the population.
    negative_markers
        Markers expected to be lower or absent.  These provide important
        counter-evidence for mixed clusters, doublets, or an incorrect label.
    thresholds
        Optional justified expression thresholds keyed by marker.  Fractions
        above threshold are only calculated when a threshold is supplied; the
        toolkit never invents a universal IMC positivity cutoff.
    notes
        Biological context retained for the notebook or a future agent prompt.

    Agent guidance
    --------------
    Build this before interpreting a population.  Marker names are matched
    exactly to ``var_names``.  Use an explicit alias mapping before QC when panel
    names differ; fuzzy matching can confuse biologically unrelated markers such
    as CD3 and CD31.
    """

    positive_markers: tuple[str, ...] = ()
    supportive_markers: tuple[str, ...] = ()
    negative_markers: tuple[str, ...] = ()
    thresholds: Mapping[str, float] = field(default_factory=dict)
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "positive_markers", _unique_strings(self.positive_markers))
        object.__setattr__(
            self, "supportive_markers", _unique_strings(self.supportive_markers)
        )
        object.__setattr__(self, "negative_markers", _unique_strings(self.negative_markers))
        object.__setattr__(
            self,
            "thresholds",
            {str(marker): float(value) for marker, value in self.thresholds.items()},
        )
        overlap = (
            set(self.positive_markers) & set(self.negative_markers)
        ) | (set(self.supportive_markers) & set(self.negative_markers))
        if overlap:
            raise ValueError(
                "Markers cannot be both positive/supportive and negative: "
                + ", ".join(sorted(overlap))
            )

    @property
    def markers(self) -> tuple[str, ...]:
        """Return all referenced markers once, preserving semantic order."""

        return tuple(
            dict.fromkeys(
                [
                    *self.positive_markers,
                    *self.supportive_markers,
                    *self.negative_markers,
                    *self.thresholds.keys(),
                ]
            )
        )

    @classmethod
    def from_value(
        cls,
        value: "MarkerExpectations | Mapping[str, Any] | None",
    ) -> "MarkerExpectations":
        """Normalize a model, mapping, or ``None`` into one expectation model."""

        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("expectations must be MarkerExpectations, a mapping, or None")
        allowed = {
            "positive_markers",
            "supportive_markers",
            "negative_markers",
            "thresholds",
            "notes",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "Unknown marker expectation fields: " + ", ".join(sorted(unknown))
            )
        return cls(**dict(value))


# The singular form is retained as an intuitive convenience for notebook users.
MarkerExpectation = MarkerExpectations


@dataclass
class PopulationDataContext:
    """Bounded inventory of the evidence available for population QC."""

    table_name: str | None
    population_key: str | None
    roi_key: str | None
    case_key: str | None
    n_cells: int
    n_markers: int
    markers: tuple[str, ...]
    obs_columns: tuple[str, ...]
    population_counts: pd.DataFrame
    candidate_population_keys: tuple[str, ...]
    leiden_sweep_columns: pd.DataFrame
    representations: dict[str, tuple[int, ...]]
    pairwise_matrices: dict[str, tuple[int, ...]]
    image_elements: int
    label_elements: int
    warnings: tuple[str, ...] = ()

    def to_agent_summary(self) -> dict[str, Any]:
        """Return compact JSON-compatible context without the full obs inventory."""

        return {
            "table_name": self.table_name,
            "population_key": self.population_key,
            "roi_key": self.roi_key,
            "case_key": self.case_key,
            "n_cells": self.n_cells,
            "n_markers": self.n_markers,
            "markers": list(self.markers),
            "population_counts": self.population_counts.to_dict(orient="records"),
            "candidate_population_keys": list(self.candidate_population_keys),
            "leiden_sweep_columns": self.leiden_sweep_columns.to_dict(orient="records"),
            "representations": self.representations,
            "pairwise_matrices": self.pairwise_matrices,
            "image_elements": self.image_elements,
            "label_elements": self.label_elements,
            "warnings": list(self.warnings),
        }


@dataclass
class PopulationExpressionResult:
    """Effect-size-first marker comparison for one target population."""

    population_key: str
    target_population: str
    reference: str
    marker_statistics: pd.DataFrame
    histogram_data: pd.DataFrame
    target_cells: int
    reference_cells: int
    sampled_target_cells: int
    sampled_reference_cells: int
    expectations: MarkerExpectations
    warnings: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)

    def strongest_markers(self, n: int = 8) -> pd.DataFrame:
        """Return markers with the largest absolute AUC effect size."""

        if n < 1:
            raise ValueError("n must be at least 1")
        return (
            self.marker_statistics.assign(
                _magnitude=self.marker_statistics["auc_effect"].abs()
            )
            .sort_values("_magnitude", ascending=False)
            .drop(columns="_magnitude")
            .head(n)
            .copy()
        )

    def to_agent_summary(self, n_markers: int = 8) -> dict[str, Any]:
        """Return compact marker evidence without the binned distribution table."""

        return {
            "population_key": self.population_key,
            "target_population": self.target_population,
            "reference": self.reference,
            "target_cells": self.target_cells,
            "reference_cells": self.reference_cells,
            "sampled_target_cells": self.sampled_target_cells,
            "sampled_reference_cells": self.sampled_reference_cells,
            "strongest_markers": self.strongest_markers(n_markers).to_dict(
                orient="records"
            ),
            "warnings": list(self.warnings),
            "parameters": self.parameters,
        }


@dataclass
class PopulationRepresentationResult:
    """Case-, sample-, and ROI-level representation evidence."""

    population_key: str
    group_summary: pd.DataFrame
    group_counts: pd.DataFrame
    warnings: tuple[str, ...] = ()

    def for_population(self, population: Any) -> pd.DataFrame:
        """Return representation summary rows for one population."""

        return self.group_summary.loc[
            self.group_summary["population"].astype(str) == str(population)
        ].copy()

    def to_agent_summary(self, population: Any | None = None) -> dict[str, Any]:
        """Return compact concentration metrics, optionally for one population."""

        frame = self.group_summary
        if population is not None:
            frame = self.for_population(population)
        return {
            "population_key": self.population_key,
            "group_summary": frame.to_dict(orient="records"),
            "warnings": list(self.warnings),
        }


@dataclass
class ResolutionComparisonResult:
    """Cluster stability and membership across precomputed resolutions."""

    reference_column: str
    sweep_columns: pd.DataFrame
    cluster_stability: pd.DataFrame
    membership: pd.DataFrame
    transition_edges: pd.DataFrame
    best_matches: pd.DataFrame
    global_metrics: pd.DataFrame
    warnings: tuple[str, ...] = ()

    def for_population(self, population: Any) -> dict[str, pd.DataFrame]:
        """Return stability, membership, and best-match evidence for one label."""

        target = str(population)
        stability = self.cluster_stability.loc[
            self.cluster_stability["population"].astype(str) == target
        ].copy()
        membership = self.membership.loc[
            self.membership["population"].astype(str) == target
        ].copy()
        best = self.best_matches
        if not best.empty and "source_cluster" in best:
            best = best.loc[best["source_cluster"].astype(str) == target].copy()
        return {"stability": stability, "membership": membership, "best_matches": best}

    def to_agent_summary(self, population: Any | None = None) -> dict[str, Any]:
        """Return bounded resolution evidence for an agent or notebook narrative."""

        if population is None:
            stability = self.cluster_stability
            membership = pd.DataFrame()
        else:
            focused = self.for_population(population)
            stability = focused["stability"]
            membership = focused["membership"]
        return {
            "reference_column": self.reference_column,
            "sweep_columns": self.sweep_columns.to_dict(orient="records"),
            "cluster_stability": stability.to_dict(orient="records"),
            "membership": membership.to_dict(orient="records"),
            "warnings": list(self.warnings),
        }


@dataclass
class CellSelectionResult:
    """Deterministic cell identities selected for image-based QC."""

    population_key: str
    population: str
    cells: pd.DataFrame
    markers: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def cell_ids(self) -> list[str]:
        """Return selected observation names in gallery order."""

        return self.cells["obs_name"].astype(str).tolist()

    def to_agent_summary(self) -> dict[str, Any]:
        """Return selected identities, strategies, parameters, and warnings."""

        return {
            "population_key": self.population_key,
            "population": self.population,
            "cells": self.cells.to_dict(orient="records"),
            "markers": list(self.markers),
            "warnings": list(self.warnings),
            "parameters": self.parameters,
        }


@dataclass
class PlotResult:
    """A focused figure plus the numerical values needed to audit it."""

    figure: Any
    axes: Any
    data: pd.DataFrame
    display_data: pd.DataFrame | None = None


@dataclass
class InMemoryClusteringResult:
    """New candidate label columns created without writing to disk."""

    adata: Any
    columns: tuple[str, ...]
    cluster_sizes: pd.DataFrame
    copied: bool
    parameters: dict[str, Any]
    warnings: tuple[str, ...] = ()

    def to_agent_summary(self) -> dict[str, Any]:
        """Return candidate columns, sizes, and provenance without AnnData values."""

        return {
            "columns": list(self.columns),
            "cluster_sizes": self.cluster_sizes.to_dict(orient="records"),
            "copied": self.copied,
            "parameters": self.parameters,
            "warnings": list(self.warnings),
        }


@dataclass
class SubclusteringResult(InMemoryClusteringResult):
    """Local population subset, graph, and candidate subcluster annotations."""

    parent_population_key: str = ""
    parent_population: str = ""
    attached_to_source: bool = False

    def to_agent_summary(self) -> dict[str, Any]:
        """Return local candidate sizes and provenance without expression values."""

        summary = super().to_agent_summary()
        summary.update(
            {
                "parent_population_key": self.parent_population_key,
                "parent_population": self.parent_population,
                "attached_to_source": self.attached_to_source,
            }
        )
        return summary


__all__ = [
    "CellSelectionResult",
    "InMemoryClusteringResult",
    "MarkerExpectation",
    "MarkerExpectations",
    "PlotResult",
    "PopulationDataContext",
    "PopulationExpressionResult",
    "PopulationRepresentationResult",
    "ResolutionComparisonResult",
    "SubclusteringResult",
]
