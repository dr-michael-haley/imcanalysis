"""Agent-friendly population labelling and quality-control tools.

The package connects structural clustering evidence, marker expression,
case/ROI representation, resolution stability, and targeted cell images.
Candidate clustering and annotation functions may add reversible observation
columns in memory; no function writes SpatialData or AnnData to disk.
"""

from .clustering import (
    apply_population_mapping,
    assess_candidate_clustering,
    assess_clustering,
    create_leiden_sweep,
    discard_population_qc_columns,
    subcluster_population,
)
from .composition import summarize_population_representation
from .context import inspect_population_data
from .expression import compare_populations, profile_population
from .models import (
    CellSelectionResult,
    InMemoryClusteringResult,
    MarkerExpectation,
    MarkerExpectations,
    PlotResult,
    PopulationDataContext,
    PopulationExpressionResult,
    PopulationRepresentationResult,
    ResolutionComparisonResult,
    SubclusteringResult,
)
from .plotting import (
    plot_clustering_qc,
    plot_marker_distributions,
    plot_population_cell_gallery,
    plot_population_heatmap,
    plot_population_representation,
    plot_resolution_membership,
    plot_resolution_stability,
)
from .resolution import compare_resolutions
from .selection import VALID_STRATEGIES, select_population_cells


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
    "VALID_STRATEGIES",
    "apply_population_mapping",
    "assess_candidate_clustering",
    "assess_clustering",
    "compare_populations",
    "compare_resolutions",
    "create_leiden_sweep",
    "discard_population_qc_columns",
    "inspect_population_data",
    "plot_clustering_qc",
    "plot_marker_distributions",
    "plot_population_cell_gallery",
    "plot_population_heatmap",
    "plot_population_representation",
    "plot_resolution_membership",
    "plot_resolution_stability",
    "profile_population",
    "select_population_cells",
    "subcluster_population",
    "summarize_population_representation",
]
