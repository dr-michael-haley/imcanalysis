"""Declarative construction of multimodal :class:`spatialdata.SpatialData`.

The public classes in this package describe source data and relationships.
They are deliberately not subclasses of SpatialData's element models.  The
planner validates the complete source graph and the builder then creates
model-compliant Images, Labels, Points, and Tables elements.
"""

from .core import add_modality, create_spatialdata, plan_spatialdata
from .models import (
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
)

__all__ = [
    "CellMasks",
    "HistologyImages",
    "IMCAnnData",
    "IMCImages",
    "MaxFuseSCRNASeq",
    "RegionLabels",
    "SpatialDataPlan",
    "SpatialDataSpec",
    "ValidationIssue",
    "ValidationReport",
    "add_modality",
    "create_spatialdata",
    "plan_spatialdata",
]
