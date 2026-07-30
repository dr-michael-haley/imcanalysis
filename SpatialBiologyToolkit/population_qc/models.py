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
from math import isfinite
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
    stored_population_qc: pd.DataFrame = field(default_factory=pd.DataFrame)
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
            "stored_population_qc": self.stored_population_qc.to_dict(
                orient="records"
            ),
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


@dataclass(frozen=True)
class MaxFuseSourceSpec:
    """One MaxFuse reference and its transferred annotation fields.

    ``path=None`` denotes columns already present in the selected AnnData table.
    Label roles are explicit because state programmes must not be interpreted as
    lineage labels.
    """

    name: str
    score_column: str | None
    label_columns: tuple[str, ...]
    label_roles: Mapping[str, str] = field(default_factory=dict)
    path: str | None = None
    score_threshold: float | None = None
    obs_name_column: str | None = None
    join_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("MaxFuse source name cannot be empty")
        labels = _unique_strings(self.label_columns)
        if not labels:
            raise ValueError("A MaxFuse source must provide at least one label column")
        roles = {
            str(key): str(value).strip().lower()
            for key, value in self.label_roles.items()
        }
        allowed_roles = {"lineage", "subtype", "state", "other"}
        unknown_roles = sorted(set(roles.values()) - allowed_roles)
        if unknown_roles:
            raise ValueError(
                "MaxFuse label roles must be lineage, subtype, state, or other; "
                f"received {unknown_roles}"
            )
        unknown_columns = sorted(set(roles) - set(labels))
        if unknown_columns:
            raise ValueError(
                "MaxFuse label_roles contains columns absent from label_columns: "
                + ", ".join(unknown_columns)
            )
        threshold = self.score_threshold
        if threshold is not None and not isfinite(float(threshold)):
            raise ValueError("MaxFuse score_threshold must be finite or None")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "score_column",
            None if self.score_column is None else str(self.score_column),
        )
        object.__setattr__(self, "label_columns", labels)
        object.__setattr__(self, "label_roles", roles)
        object.__setattr__(self, "path", None if self.path is None else str(self.path))
        object.__setattr__(
            self,
            "score_threshold",
            None if threshold is None else float(threshold),
        )
        object.__setattr__(
            self,
            "obs_name_column",
            None if self.obs_name_column is None else str(self.obs_name_column),
        )
        object.__setattr__(self, "join_keys", _unique_strings(self.join_keys))

    @classmethod
    def from_value(
        cls,
        value: "MaxFuseSourceSpec | Mapping[str, Any]",
    ) -> "MaxFuseSourceSpec":
        """Normalize a source model or notebook-friendly mapping."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("MaxFuse source specifications must be mappings or models")
        return cls(**dict(value))


@dataclass
class MaxFuseInputAudit:
    """Metadata and identity-alignment audit without population-label summaries."""

    sources: pd.DataFrame
    alignment_audit: pd.DataFrame
    warnings: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        """Return whether at least one MaxFuse source was discovered."""

        return not self.sources.empty

    def to_agent_summary(self) -> dict[str, Any]:
        """Return a JSON-compatible audit suitable for the priors notebook."""

        return {
            "available": self.available,
            "sources": self.sources.to_dict(orient="records"),
            "alignment_audit": self.alignment_audit.to_dict(orient="records"),
            "warnings": list(self.warnings),
            "parameters": self.parameters,
        }


@dataclass
class MaxFuseEvidenceResult:
    """Read-only MaxFuse label-transfer evidence aligned to one population key."""

    population_key: str
    sources: pd.DataFrame
    alignment_audit: pd.DataFrame
    source_summary: pd.DataFrame
    population_summary: pd.DataFrame
    label_distribution: pd.DataFrame
    threshold_sensitivity: pd.DataFrame
    warnings: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    _cell_evidence: pd.DataFrame = field(
        default_factory=pd.DataFrame,
        repr=False,
    )

    @property
    def available(self) -> bool:
        """Return whether at least one MaxFuse source was discovered."""

        return not self.sources.empty

    def for_population(self, population: Any) -> dict[str, pd.DataFrame]:
        """Return all MaxFuse summaries for one source population."""

        target = str(population)

        def focused(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty or "population" not in frame:
                return frame.copy()
            return frame.loc[frame["population"].astype(str).eq(target)].copy()

        return {
            "population_summary": focused(self.population_summary),
            "label_distribution": focused(self.label_distribution),
            "threshold_sensitivity": focused(self.threshold_sensitivity),
        }

    def for_cells(self, obs_names: Sequence[Any]) -> pd.DataFrame:
        """Return source labels and scores for selected cells in long form."""

        requested = list(dict.fromkeys(map(str, obs_names)))
        if not requested:
            return pd.DataFrame(
                columns=[
                    "obs_name",
                    "source",
                    "score",
                    "score_threshold",
                    "thresholded",
                    "passes_threshold",
                    "label_column",
                    "label_role",
                    "label",
                ]
            )
        missing = [
            value for value in requested if value not in self._cell_evidence.index
        ]
        if missing:
            raise KeyError(
                "Selected observation names are absent from the aligned MaxFuse evidence: "
                + ", ".join(missing[:20])
            )
        rows: list[dict[str, Any]] = []
        selected = self._cell_evidence.loc[requested]
        for source in self.sources["source"].drop_duplicates().astype(str):
            score_key = f"{source}::score"
            scores = (
                selected[score_key]
                if score_key in selected
                else pd.Series(index=selected.index, dtype=float)
            )
            source_frame = self.sources.loc[
                self.sources["source"].astype(str).eq(source)
            ]
            raw_threshold = source_frame["score_threshold"].iloc[0]
            threshold = None if pd.isna(raw_threshold) else float(raw_threshold)
            for label_row in source_frame.itertuples(index=False):
                label_key = f"{source}::{label_row.label_column}"
                labels = selected[label_key]
                for obs_name in requested:
                    score = scores.get(obs_name, pd.NA)
                    label = labels.get(obs_name, pd.NA)
                    passes = (
                        (pd.notna(score) or pd.notna(label))
                        if threshold is None
                        else (pd.notna(score) and float(score) >= threshold)
                    )
                    rows.append(
                        {
                            "obs_name": obs_name,
                            "source": source,
                            "score": score,
                            "score_threshold": threshold,
                            "thresholded": threshold is not None,
                            "passes_threshold": bool(passes),
                            "label_column": str(label_row.label_column),
                            "label_role": str(label_row.label_role),
                            "label": label,
                        }
                    )
        return pd.DataFrame(rows)

    def to_agent_summary(self, population: Any | None = None) -> dict[str, Any]:
        """Return bounded, JSON-compatible MaxFuse evidence."""

        if population is None:
            population_summary = self.population_summary
            distribution = self.label_distribution.loc[
                self.label_distribution.get("rank", pd.Series(dtype=int)).eq(1)
            ]
        else:
            focused = self.for_population(population)
            population_summary = focused["population_summary"]
            distribution = focused["label_distribution"]
            if "rank" in distribution:
                distribution = distribution.loc[distribution["rank"].eq(1)]
        return {
            "population_key": self.population_key,
            "available": self.available,
            "sources": self.sources.to_dict(orient="records"),
            "alignment_audit": self.alignment_audit.to_dict(orient="records"),
            "population_summary": population_summary.to_dict(orient="records"),
            "top_labels": distribution.to_dict(orient="records"),
            "warnings": list(self.warnings),
            "parameters": self.parameters,
        }


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
    "MaxFuseEvidenceResult",
    "MaxFuseInputAudit",
    "MaxFuseSourceSpec",
    "MarkerExpectation",
    "MarkerExpectations",
    "PlotResult",
    "PopulationDataContext",
    "PopulationExpressionResult",
    "PopulationRepresentationResult",
    "ResolutionComparisonResult",
    "SubclusteringResult",
]
