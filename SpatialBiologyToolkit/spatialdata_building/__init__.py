"""Declarative construction of multimodal :class:`spatialdata.SpatialData`.

The public classes in this package describe source data and relationships.
They are deliberately not subclasses of SpatialData's element models.  The
planner validates the complete source graph and the builder then creates
model-compliant Images, Labels, Points, and Tables elements.
"""

from .core import add_modality, create_spatialdata, plan_spatialdata
from .discovery import (
    AssetCandidate,
    DiscoveryIssue,
    HistologyAssetHint,
    IMCImageAssetHint,
    MaxFuseAssetHint,
    RegionLabelsAssetHint,
    SpatialDataAssetHints,
    SpatialDataAssetInventory,
    SpatialDataAssetPlan,
    SpatialDataAssetProposal,
    SpatialDataBuildResult,
    SpatialDataDiscoveryOptions,
    build_spatialdata_from_assets,
    discover_spatialdata_assets,
    plan_spatialdata_from_assets,
    propose_spatialdata_spec,
)
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
    "AssetCandidate",
    "CellMasks",
    "DiscoveryIssue",
    "HistologyAssetHint",
    "HistologyImages",
    "IMCImageAssetHint",
    "IMCAnnData",
    "IMCImages",
    "MaxFuseAssetHint",
    "MaxFuseSCRNASeq",
    "RegionLabelsAssetHint",
    "RegionLabels",
    "SpatialDataAssetHints",
    "SpatialDataAssetInventory",
    "SpatialDataAssetPlan",
    "SpatialDataAssetProposal",
    "SpatialDataBuildResult",
    "SpatialDataDiscoveryOptions",
    "SpatialDataPlan",
    "SpatialDataSpec",
    "ValidationIssue",
    "ValidationReport",
    "add_modality",
    "build_spatialdata_from_assets",
    "create_spatialdata",
    "discover_spatialdata_assets",
    "plan_spatialdata_from_assets",
    "plan_spatialdata",
    "propose_spatialdata_spec",
]
