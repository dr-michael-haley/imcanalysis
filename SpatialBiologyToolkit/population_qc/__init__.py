"""Agent-friendly population labelling and quality-control tools.

The package connects structural clustering evidence, MaxFuse atlas transfers,
marker expression, case/ROI representation, resolution stability, and targeted
cell images. Candidate clustering and annotation functions may add reversible
observation columns in memory. Assessment functions never persist those changes;
the separately named ``publish_posterior_mapping()`` is the one-write
end-of-workflow finalisation operation.
"""

from .artifacts import (
    MANIFEST_COLUMNS,
    POSTERIOR_MAPPING_COLUMNS,
    STAGE_CONCLUSION_COLUMNS,
    PopulationQCArtifactWriter,
)
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
from .maxfuse import (
    inspect_maxfuse_inputs,
    plot_maxfuse_label_heatmap,
    plot_maxfuse_threshold_sensitivity,
    summarize_maxfuse_evidence,
)
from .models import (
    CellSelectionResult,
    InMemoryClusteringResult,
    MaxFuseEvidenceResult,
    MaxFuseInputAudit,
    MaxFuseSourceSpec,
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
    plot_clustering_qc_panels,
    plot_marker_distributions,
    plot_population_breakdown,
    plot_population_cell_gallery,
    plot_population_heatmap,
    plot_population_matrixplot,
    plot_population_representation,
    plot_population_umap,
    plot_resolution_membership,
    plot_resolution_stability,
)
from .publishing import (
    PosteriorPublicationConfig,
    map_posterior_labels,
    publish_posterior_mapping,
)
from .resolution import compare_resolutions
from .scanpy import (
    plot_population_scanpy_abundance,
    plot_population_scanpy_matrixplot,
    plot_population_scanpy_umap,
    temporary_numba_cache_dir,
)
from .selection import (
    VALID_STRATEGIES,
    select_population_cell_panel,
    select_population_cells,
)
from .stored import (
    StoredPopulationQCError,
    focused_population_summary,
    has_stored_population_qc,
    list_stored_population_qc,
    load_stored_population_qc,
)


__all__ = [
    "CellSelectionResult",
    "InMemoryClusteringResult",
    "MaxFuseEvidenceResult",
    "MaxFuseInputAudit",
    "MaxFuseSourceSpec",
    "MarkerExpectation",
    "MarkerExpectations",
    "MANIFEST_COLUMNS",
    "POSTERIOR_MAPPING_COLUMNS",
    "PlotResult",
    "PopulationQCArtifactWriter",
    "PosteriorPublicationConfig",
    "PopulationDataContext",
    "PopulationExpressionResult",
    "PopulationRepresentationResult",
    "ResolutionComparisonResult",
    "SubclusteringResult",
    "STAGE_CONCLUSION_COLUMNS",
    "StoredPopulationQCError",
    "VALID_STRATEGIES",
    "apply_population_mapping",
    "assess_candidate_clustering",
    "assess_clustering",
    "compare_populations",
    "compare_resolutions",
    "create_leiden_sweep",
    "discard_population_qc_columns",
    "inspect_population_data",
    "inspect_maxfuse_inputs",
    "focused_population_summary",
    "has_stored_population_qc",
    "list_stored_population_qc",
    "load_stored_population_qc",
    "plot_clustering_qc",
    "plot_clustering_qc_panels",
    "plot_maxfuse_label_heatmap",
    "plot_maxfuse_threshold_sensitivity",
    "plot_marker_distributions",
    "plot_population_breakdown",
    "plot_population_cell_gallery",
    "plot_population_heatmap",
    "plot_population_matrixplot",
    "plot_population_representation",
    "plot_population_scanpy_abundance",
    "plot_population_scanpy_matrixplot",
    "plot_population_scanpy_umap",
    "temporary_numba_cache_dir",
    "plot_population_umap",
    "plot_resolution_membership",
    "plot_resolution_stability",
    "profile_population",
    "publish_posterior_mapping",
    "map_posterior_labels",
    "select_population_cell_panel",
    "select_population_cells",
    "subcluster_population",
    "summarize_maxfuse_evidence",
    "summarize_population_representation",
]
