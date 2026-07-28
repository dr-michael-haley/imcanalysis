"""Public specifications and immutable plans for SpatialData construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence


PathLike = str | Path
Severity = Literal["info", "warning", "error"]
ModalityKind = Literal[
    "cell_masks",
    "imc_images",
    "imc_anndata",
    "histology_images",
    "region_labels",
    "maxfuse_scrnaseq",
]


@dataclass(frozen=True)
class CellMasks:
    """Describe ROI-specific integer cell-segmentation masks.

    Use this modality when positive raster values identify individual cells
    and zero is background.  One or more :class:`IMCAnnData` tables can
    annotate the same masks.

    Parameters
    ----------
    name
        Stable modality identifier used by other specifications.
    folder
        Folder containing one mask per ROI.
    rois
        Optional ROI names.  They are normally inferred from linked
        ``IMCAnnData`` modalities.  Supply them when masks are added without a
        linked table.
    suffix, extensions
        Files are matched exactly and case-insensitively as
        ``{ROI}{suffix}{extension}``.  Ambiguous matches are errors.
    background
        Background pixel value.  SpatialData labels conventionally use zero.
    coordinate_system_prefix
        Prefix used for the independent ROI-local coordinate systems.
    """

    name: str
    folder: PathLike
    rois: Sequence[str] | None = None
    suffix: str = ""
    extensions: Sequence[str] = (".tif", ".tiff")
    background: int = 0
    coordinate_system_prefix: str = "roi"


@dataclass(frozen=True)
class IMCImages:
    """Describe a panel of single-channel IMC images arranged by ROI.

    Images are expected under ``folder/{ROI}/`` with one 2D image per channel.
    Channels can be supplied explicitly or inferred from a linked
    :class:`IMCAnnData`.  A panel without a quantified table must define
    ``channels``.

    ``reference`` identifies a modality whose ROI coordinate systems are
    reused.  When a linked ``IMCAnnData`` selects one cell-mask modality, that
    mask is inferred as the reference.  Identity alignment requires matching
    raster shapes; otherwise provide one SpatialData transformation per ROI
    in ``transformations``.

    Set ``allow_partial=True`` for an unquantified panel that is available for
    only a subset of the reference ROIs.  When ``rois`` is omitted, planning
    discovers the matching ROI directories and records one coverage warning.
    Quantified panels linked to ``IMCAnnData`` must still cover every table ROI.
    """

    name: str
    panel_name: str
    folder: PathLike
    channels: Sequence[str] | None = None
    rois: Sequence[str] | None = None
    reference: str | None = None
    allow_partial: bool = False
    extensions: Sequence[str] = (".tif", ".tiff")
    match_mode: Literal["exact", "exact_or_unique_substring"] = (
        "exact_or_unique_substring"
    )
    allow_extra_files: bool = True
    transformations: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class IMCAnnData:
    """Describe a quantified IMC cell table.

    The table annotates the ``CellMasks`` selected by ``masks`` and is linked
    to the quantified ``IMCImages`` selected by ``images``.  Multiple IMC
    tables and panels can coexist in one SpatialData object.

    By default the source AnnData is copied before SpatialData annotation
    columns and metadata are added.  Set ``copy_adata=False`` for very large
    in-memory or backed objects when mutating their table metadata is
    acceptable. ``x_key`` and ``y_key`` are validated and used to create one
    centroid Points element per ROI when ``include_centroids`` is true.
    """

    name: str
    panel_name: str
    adata: Any
    images: str
    masks: str
    roi_key: str = "ROI"
    instance_key: str = "ObjectNumber"
    x_key: str = "X_loc"
    y_key: str = "Y_loc"
    table_name: str | None = None
    include_centroids: bool = True
    check_centroids_in_mask: bool = False
    copy_adata: bool = True


@dataclass(frozen=True)
class HistologyImages:
    """Describe ROI-aligned RGB or RGBA histology images.

    TIFF, PNG, and JPEG are supported by default.  Matching is exact and
    case-insensitive using ``{ROI}{suffix}{extension}``; extension
    autodetection never resolves ambiguity silently.  Set
    ``allow_partial=True`` to discover and include the subset of reference
    ROIs that have histology files; missing coverage is reported as one
    planner warning.
    """

    name: str
    folder: PathLike
    reference: str
    rois: Sequence[str] | None = None
    allow_partial: bool = False
    suffix: str = ""
    extensions: Sequence[str] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    drop_alpha: bool = False
    transformations: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RegionLabels:
    """Describe named ROI-aligned categorical label rasters.

    Integer pixels remain SpatialData Labels elements.  Their semantic names
    are stored in a linked annotation table because Labels elements cannot
    directly contain per-region annotations.

    ``value_names`` accepts a global ``{value: name}`` mapping, a nested
    ``{ROI: {value: name}}`` mapping, a DataFrame, or a CSV path.  DataFrames
    and CSV files use ``value_key`` and ``name_key`` and may optionally
    contain ``mapping_roi_key``.  Set ``allow_partial=True`` to discover and
    include the subset of reference ROIs that have label rasters.
    """

    name: str
    folder: PathLike
    suffix: str
    value_names: Any
    reference: str
    rois: Sequence[str] | None = None
    allow_partial: bool = False
    extensions: Sequence[str] = (".tif", ".tiff")
    table_name: str | None = None
    value_key: str = "label_value"
    name_key: str = "label_name"
    mapping_roi_key: str | None = None
    transformations: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MaxFuseSCRNASeq:
    """Describe transcriptomes matched by MaxFuse to one IMC table.

    The source observation index must be a unique subset of the linked IMC
    table's observation index.  Only matched cells are stored; missing
    transcriptomes remain absent rather than being represented as biological
    zeros.  Region and instance links are copied from the IMC table so this
    transcriptomic table formally annotates the same cell masks.
    """

    name: str
    adata: Any
    imc_table: str
    table_name: str | None = None
    copy_adata: bool = True


ModalitySpec = (
    CellMasks
    | IMCImages
    | IMCAnnData
    | HistologyImages
    | RegionLabels
    | MaxFuseSCRNASeq
)


@dataclass(frozen=True)
class SpatialDataSpec:
    """Complete declarative description of a SpatialData object."""

    modalities: Sequence[ModalitySpec]
    attrs: Mapping[str, Any] = field(default_factory=dict)
    raster_chunks: int | tuple[int, int] = (512, 512)
    scale_factors: Sequence[int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "modalities", tuple(self.modalities))
        object.__setattr__(self, "attrs", dict(self.attrs))
        if self.scale_factors is not None:
            object.__setattr__(self, "scale_factors", tuple(self.scale_factors))


@dataclass(frozen=True)
class ValidationIssue:
    """One structured planner diagnostic."""

    severity: Severity
    code: str
    message: str
    modality: str | None = None
    roi: str | None = None
    path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "modality": self.modality,
            "roi": self.roi,
            "path": None if self.path is None else str(self.path),
            "message": self.message,
        }


@dataclass(frozen=True)
class ValidationReport:
    """Structured result of source and cross-modality validation."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        """Raise one actionable exception containing every planner error."""

        if not self.errors:
            return
        details = "\n".join(
            f"- [{issue.code}] "
            f"{f'{issue.modality}: ' if issue.modality else ''}{issue.message}"
            for issue in self.errors
        )
        raise ValueError(
            f"SpatialData planning found {len(self.errors)} error(s):\n{details}"
        )

    def to_frame(self) -> Any:
        """Return diagnostics as a pandas DataFrame for notebooks and agents."""

        import pandas as pd

        columns = ["severity", "code", "modality", "roi", "path", "message"]
        return pd.DataFrame.from_records(
            (issue.as_dict() for issue in self.issues),
            columns=columns,
        )


@dataclass(frozen=True)
class RasterElementPlan:
    """Resolved raster input for one modality and ROI."""

    roi: str
    element_name: str
    coordinate_system: str
    paths: tuple[Path, ...]
    shape: tuple[int, int]
    dtype: str
    channels: tuple[str, ...] = ()
    channel_match_modes: tuple[str, ...] = ()
    transformations: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannedModality:
    """Validated modality and its resolved output contract."""

    name: str
    kind: ModalityKind
    source: ModalitySpec
    elements: tuple[RasterElementPlan, ...] = ()
    table_name: str | None = None
    rois: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpatialDataPlan:
    """Immutable, side-effect-free plan used by the builder."""

    spec: SpatialDataSpec
    modalities: tuple[PlannedModality, ...]
    report: ValidationReport
    existing: Any | None = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return self.report.ok

    def raise_for_errors(self) -> None:
        self.report.raise_for_errors()

    def modality(self, name: str) -> PlannedModality:
        matches = [
            modality
            for modality in self.modalities
            if modality.name.casefold() == str(name).casefold()
        ]
        if len(matches) != 1:
            available = ", ".join(modality.name for modality in self.modalities)
            raise KeyError(f"Modality {name!r} was not found. Available: {available}")
        return matches[0]

    def summary(self) -> dict[str, Any]:
        """Return a concise, JSON-friendly construction summary."""

        return {
            "ok": self.ok,
            "modalities": len(self.modalities),
            "images": sum(
                len(item.elements)
                for item in self.modalities
                if item.kind in {"imc_images", "histology_images"}
            ),
            "labels": sum(
                len(item.elements)
                for item in self.modalities
                if item.kind in {"cell_masks", "region_labels"}
            ),
            "points": sum(
                len(item.rois)
                for item in self.modalities
                if item.kind == "imc_anndata"
                and bool(item.details.get("include_centroids"))
            ),
            "tables": sum(
                item.kind
                in {"imc_anndata", "region_labels", "maxfuse_scrnaseq"}
                for item in self.modalities
            ),
            "errors": len(self.report.errors),
            "warnings": len(self.report.warnings),
            "by_modality": {
                item.name: {
                    "kind": item.kind,
                    "rois": len(item.rois),
                    "channels": len(item.channels),
                    "table": item.table_name,
                    "elements": len(item.elements),
                }
                for item in self.modalities
            },
        }
