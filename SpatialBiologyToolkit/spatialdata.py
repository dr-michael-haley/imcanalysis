"""Build, extend, inspect, and plot multimodal SpatialData objects.

Construction uses declarative source specifications such as
:class:`IMCAnnData`, :class:`IMCImages`, :class:`CellMasks`,
:class:`HistologyImages`, :class:`RegionLabels`, and
:class:`MaxFuseSCRNASeq`.  Each adapter validates its source and relationships
before producing standard SpatialData model elements.  Raster data remain
Dask-backed until accessed or written.
"""

from .spatialdata_building._legacy import (
    DEFAULT_INSTANCE_KEY,
    DEFAULT_LABEL_INSTANCE_KEY,
    DEFAULT_LABEL_NAME_KEY,
    DEFAULT_LABEL_REGION_KEY,
    DEFAULT_LABEL_VALUE_KEY,
    DEFAULT_REGION_KEY,
    DEFAULT_TABLE_NAME,
    SBT_METADATA_KEY,
    get_label_annotations,
    get_roi_elements,
    get_roi_label_elements,
    get_roi_modalities,
    match_marker_image,
    plot_population_counts,
    plot_spatialdata_cells,
    plot_spatialdata_roi,
    summarize_spatialdata,
    write_spatialdata,
)
from .spatialdata_building import (
    CellMasks,
    HistologyImages,
    IMCAnnData,
    IMCImages,
    MaxFuseSCRNASeq,
    RegionLabels,
    SpatialDataPlan,
    SpatialDataSpec,
    ValidationIssue,
    ValidationReport,
    add_modality,
    create_spatialdata,
    plan_spatialdata,
)

__all__ = [
    "DEFAULT_INSTANCE_KEY",
    "DEFAULT_LABEL_INSTANCE_KEY",
    "DEFAULT_LABEL_NAME_KEY",
    "DEFAULT_LABEL_REGION_KEY",
    "DEFAULT_LABEL_VALUE_KEY",
    "DEFAULT_REGION_KEY",
    "DEFAULT_TABLE_NAME",
    "CellMasks",
    "HistologyImages",
    "IMCAnnData",
    "IMCImages",
    "MaxFuseSCRNASeq",
    "RegionLabels",
    "SBT_METADATA_KEY",
    "SpatialDataPlan",
    "SpatialDataSpec",
    "ValidationIssue",
    "ValidationReport",
    "add_modality",
    "create_spatialdata",
    "get_label_annotations",
    "get_roi_elements",
    "get_roi_label_elements",
    "get_roi_modalities",
    "match_marker_image",
    "plan_spatialdata",
    "plot_population_counts",
    "plot_spatialdata_cells",
    "plot_spatialdata_roi",
    "summarize_spatialdata",
    "write_spatialdata",
]
