"""Discover spatial assets and turn explicit or inferred relationships into plans.

The discovery layer is deliberately conservative.  It inventories likely
assets, honours user-supplied paths, and proposes only relationships supported
by file contents and cross-asset identities.  The existing declarative
SpatialData planner remains authoritative for scientific validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .models import (
    CellMasks,
    HistologyImages,
    IMCAnnData,
    IMCImages,
    MaxFuseSCRNASeq,
    RegionLabels,
    SpatialDataPlan,
    SpatialDataSpec,
)


PathLike = str | Path
CandidateKind = Literal[
    "anndata",
    "roi_image_collection",
    "raster_collection",
    "mapping_table",
]
Confidence = Literal["explicit", "high", "medium", "low"]
DiscoverySeverity = Literal["info", "warning", "error"]
RASTER_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _tuple(value: Sequence[Any] | None) -> tuple[Any, ...]:
    return () if value is None else tuple(value)


def _safe_name(value: str) -> str:
    import re

    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_.-")
    return cleaned or "asset"


def _resolve(root: Path, value: PathLike) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


@dataclass(frozen=True)
class IMCImageAssetHint:
    """Explicit standalone IMC image panel."""

    name: str
    folder: PathLike
    panel_name: str | None = None
    channels: Sequence[str] | None = None
    reference: str = "cells"
    allow_partial: bool = True
    transformations: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class HistologyAssetHint:
    """Explicit histology image collection."""

    name: str
    folder: PathLike
    suffix: str = ""
    reference: str = "cells"
    allow_partial: bool = True
    drop_alpha: bool = False
    transformations: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RegionLabelsAssetHint:
    """Explicit categorical label collection and its semantic mapping."""

    name: str
    folder: PathLike
    value_names: Any
    suffix: str = ""
    reference: str = "cells"
    allow_partial: bool = True
    value_key: str = "label_value"
    name_key: str = "label_name"
    mapping_roi_key: str | None = None
    transformations: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MaxFuseAssetHint:
    """Explicit MaxFuse-derived AnnData linked to the primary IMC table."""

    name: str
    adata: Any
    imc_table: str = "cell_quantification"
    table_name: str | None = None
    copy_adata: bool = True


@dataclass(frozen=True)
class SpatialDataAssetHints:
    """User-supplied paths and relationships that override inference."""

    anndata: PathLike | None = None
    cell_masks: PathLike | None = None
    primary_images: PathLike | None = None
    primary_table_name: str = "cell_quantification"
    primary_images_name: str = "cell_images"
    cell_masks_name: str = "cells"
    primary_panel_name: str = "IMC panel"
    roi_key: str = "ROI"
    instance_key: str = "ObjectNumber"
    x_key: str = "X_loc"
    y_key: str = "Y_loc"
    copy_adata: bool = False
    additional_images: Sequence[IMCImageAssetHint] = ()
    histology: Sequence[HistologyAssetHint] = ()
    region_labels: Sequence[RegionLabelsAssetHint] = ()
    maxfuse: Sequence[MaxFuseAssetHint] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)
    raster_chunks: int | tuple[int, int] = (512, 512)
    scale_factors: Sequence[int] | None = None
    discover_unlisted_assets: bool = True
    include_discovered_image_panels: bool = True
    include_discovered_histology: bool = True
    include_discovered_maxfuse: bool = True

    def __post_init__(self) -> None:
        if self.anndata is not None and not isinstance(self.anndata, (str, Path)):
            raise TypeError(
                "Folder discovery requires hints.anndata to be a filesystem path; "
                "use SpatialDataSpec directly for an in-memory AnnData object."
            )
        object.__setattr__(self, "additional_images", _tuple(self.additional_images))
        object.__setattr__(self, "histology", _tuple(self.histology))
        object.__setattr__(self, "region_labels", _tuple(self.region_labels))
        object.__setattr__(self, "maxfuse", _tuple(self.maxfuse))
        object.__setattr__(self, "attrs", dict(self.attrs))


@dataclass(frozen=True)
class SpatialDataDiscoveryOptions:
    """Bounds for recursive discovery and representative inspection."""

    max_depth: int = 3
    max_entries: int = 50_000
    sample_files: int = 5
    ignore_names: Sequence[str] = (
        ".git",
        ".sbt",
        "__pycache__",
        "outputs",
        "QC",
        "SLURM_logs",
    )

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        if self.max_entries < 1:
            raise ValueError("max_entries must be positive.")
        if self.sample_files < 1:
            raise ValueError("sample_files must be positive.")
        object.__setattr__(self, "ignore_names", tuple(self.ignore_names))


@dataclass(frozen=True)
class DiscoveryIssue:
    severity: DiscoverySeverity
    code: str
    message: str
    path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "path": None if self.path is None else str(self.path),
            "message": self.message,
        }


def _public_details(details: Mapping[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for key, value in details.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, tuple) and len(value) > 20:
            public[f"{key}_count"] = len(value)
            public[f"{key}_examples"] = list(value[:5])
        else:
            public[str(key)] = value
    return public


@dataclass(frozen=True)
class AssetCandidate:
    candidate_id: str
    kind: CandidateKind
    path: Path
    confidence: Confidence
    reason: str
    source: Literal["discovered", "explicit"] = "discovered"
    details: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "path": str(self.path),
            "confidence": self.confidence,
            "source": self.source,
            "reason": self.reason,
            **_public_details(self.details),
        }


@dataclass(frozen=True)
class SpatialDataAssetInventory:
    root: Path
    hints: SpatialDataAssetHints
    candidates: tuple[AssetCandidate, ...]
    issues: tuple[DiscoveryIssue, ...] = ()

    def candidates_frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame.from_records(candidate.as_dict() for candidate in self.candidates)

    def issues_frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame.from_records(issue.as_dict() for issue in self.issues)

    def by_kind(self, kind: CandidateKind) -> tuple[AssetCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.kind == kind)

    def candidate_for_path(
        self, kind: CandidateKind, value: PathLike
    ) -> AssetCandidate | None:
        requested = _resolve(self.root, value)
        return next(
            (
                candidate
                for candidate in self.candidates
                if candidate.kind == kind and candidate.path == requested
            ),
            None,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "candidates": len(self.candidates),
            "issues": len(self.issues),
            "by_kind": {
                kind: len(self.by_kind(kind))
                for kind in (
                    "anndata",
                    "roi_image_collection",
                    "raster_collection",
                    "mapping_table",
                )
            },
        }


@dataclass(frozen=True)
class SpatialDataAssetProposal:
    inventory: SpatialDataAssetInventory
    spec: SpatialDataSpec | None
    selected: Mapping[str, str] = field(default_factory=dict)
    issues: tuple[DiscoveryIssue, ...] = ()

    @property
    def errors(self) -> tuple[DiscoveryIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def ok(self) -> bool:
        return self.spec is not None and not self.errors

    def raise_for_errors(self) -> None:
        if not self.errors:
            if self.spec is None:
                raise ValueError("SpatialData discovery did not produce a specification.")
            return
        details = "\n".join(
            f"- [{issue.code}] {issue.message}" for issue in self.errors
        )
        raise ValueError(
            f"SpatialData asset proposal found {len(self.errors)} error(s):\n{details}"
        )

    def issues_frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame.from_records(issue.as_dict() for issue in self.issues)

    def selections_frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame.from_records(
            {"role": role, "candidate_id": candidate_id}
            for role, candidate_id in self.selected.items()
        )


@dataclass(frozen=True)
class SpatialDataAssetPlan:
    """Folder discovery plus the authoritative declarative SpatialData plan."""

    inventory: SpatialDataAssetInventory
    proposal: SpatialDataAssetProposal
    spatialdata_plan: SpatialDataPlan | None

    @property
    def ok(self) -> bool:
        return (
            self.proposal.ok
            and self.spatialdata_plan is not None
            and self.spatialdata_plan.ok
        )

    def raise_for_errors(self) -> None:
        self.proposal.raise_for_errors()
        if self.spatialdata_plan is None:
            raise ValueError("No SpatialData plan was produced.")
        self.spatialdata_plan.raise_for_errors()

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "inventory": self.inventory.summary(),
            "proposal_errors": len(self.proposal.errors),
            "plan": (
                None
                if self.spatialdata_plan is None
                else self.spatialdata_plan.summary()
            ),
        }


@dataclass(frozen=True)
class SpatialDataBuildResult:
    output_path: Path
    asset_plan: SpatialDataAssetPlan
    element_counts: Mapping[str, int]


def _bounded_tree(
    root: Path, options: SpatialDataDiscoveryOptions
) -> tuple[list[Path], list[Path], bool]:
    directories = [root]
    files: list[Path] = []
    stack = [(root, 0)]
    seen = 0
    ignored = {name.casefold() for name in options.ignore_names}
    while stack:
        directory, depth = stack.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda path: path.name.casefold())
        except OSError:
            continue
        for entry in entries:
            seen += 1
            if seen > options.max_entries:
                return directories, files, True
            if entry.is_dir():
                if (
                    entry.name.casefold() in ignored
                    or entry.name.startswith(".")
                    or entry.suffix.casefold() == ".zarr"
                ):
                    continue
                directories.append(entry)
                if depth < options.max_depth:
                    stack.append((entry, depth + 1))
            elif entry.is_file():
                files.append(entry)
    return directories, files, False


def _inspect_h5ad(path: Path, hints: SpatialDataAssetHints) -> AssetCandidate:
    import anndata as ad

    if not path.is_file():
        raise FileNotFoundError(f"AnnData file not found: {path}")
    handle = ad.read_h5ad(path, backed="r")
    try:
        obs_columns = tuple(str(value) for value in handle.obs.columns)
        var_names = tuple(str(value) for value in handle.var_names)
        required = (
            hints.roi_key,
            hints.instance_key,
            hints.x_key,
            hints.y_key,
        )
        present = tuple(key for key in required if key in handle.obs.columns)
        rois = (
            tuple(dict.fromkeys(map(str, handle.obs[hints.roi_key])))
            if hints.roi_key in handle.obs.columns
            else ()
        )
        confidence: Confidence = "high" if len(present) == len(required) else "low"
        reason = (
            "Contains the configured ROI, instance, X, and Y observation columns."
            if confidence == "high"
            else f"Contains {len(present)}/{len(required)} configured spatial columns."
        )
        return AssetCandidate(
            candidate_id=f"anndata:{path}",
            kind="anndata",
            path=path,
            confidence=confidence,
            reason=reason,
            details={
                "n_obs": int(handle.n_obs),
                "n_vars": int(handle.n_vars),
                "obs_columns": obs_columns,
                "var_names": var_names,
                "rois": rois,
                "uns_keys": tuple(map(str, handle.uns.keys())),
                "_spatial_columns": present,
            },
        )
    finally:
        if getattr(handle, "file", None) is not None:
            handle.file.close()


def _raster_properties(path: Path) -> tuple[tuple[int, ...], str]:
    import imageio.v3 as iio

    properties = iio.improps(path)
    return tuple(int(value) for value in properties.shape), str(properties.dtype)


def _inspect_raster_collection(
    path: Path, options: SpatialDataDiscoveryOptions
) -> AssetCandidate:
    import numpy as np

    if not path.is_dir():
        raise FileNotFoundError(f"Raster folder not found: {path}")
    files = tuple(
        candidate
        for candidate in sorted(path.iterdir(), key=lambda item: item.name.casefold())
        if candidate.is_file() and candidate.suffix.casefold() in RASTER_EXTENSIONS
    )
    if not files:
        raise ValueError(f"Raster folder contains no supported images: {path}")
    shapes: list[tuple[int, ...]] = []
    dtypes: list[str] = []
    raster_kind = "unknown"
    for candidate in files[: options.sample_files]:
        shape, dtype = _raster_properties(candidate)
        shapes.append(shape)
        dtypes.append(dtype)
    if all(len(shape) == 2 for shape in shapes) and all(
        np.issubdtype(np.dtype(dtype), np.integer) for dtype in dtypes
    ):
        raster_kind = "integer_labels"
    elif all(
        len(shape) == 3 and shape[-1] in {3, 4}
        for shape in shapes
    ):
        raster_kind = "rgb"
    return AssetCandidate(
        candidate_id=f"raster_collection:{path}",
        kind="raster_collection",
        path=path,
        confidence="medium" if raster_kind != "unknown" else "low",
        reason=(
            f"Contains {len(files)} directly stored raster(s); representative "
            f"content classified as {raster_kind}."
        ),
        details={
            "file_count": len(files),
            "stems": tuple(candidate.stem for candidate in files),
            "extensions": tuple(sorted({candidate.suffix.casefold() for candidate in files})),
            "sample_shapes": tuple(shapes),
            "sample_dtypes": tuple(dtypes),
            "raster_kind": raster_kind,
        },
    )


def _inspect_roi_image_collection(
    path: Path, options: SpatialDataDiscoveryOptions
) -> AssetCandidate:
    if not path.is_dir():
        raise FileNotFoundError(f"ROI image folder not found: {path}")
    roi_folders = tuple(
        child
        for child in sorted(path.iterdir(), key=lambda item: item.name.casefold())
        if child.is_dir()
        and any(
            file.is_file() and file.suffix.casefold() in RASTER_EXTENSIONS
            for file in child.iterdir()
        )
    )
    if not roi_folders:
        raise ValueError(f"Folder contains no ROI image subdirectories: {path}")
    signatures: list[tuple[str, ...]] = []
    for roi_folder in roi_folders[: options.sample_files]:
        signatures.append(
            tuple(
                file.stem
                for file in sorted(
                    roi_folder.iterdir(), key=lambda item: item.name.casefold()
                )
                if file.is_file() and file.suffix.casefold() in RASTER_EXTENSIONS
            )
        )
    consistent = bool(signatures) and all(
        signature == signatures[0] for signature in signatures[1:]
    )
    return AssetCandidate(
        candidate_id=f"roi_image_collection:{path}",
        kind="roi_image_collection",
        path=path,
        confidence="high" if consistent else "medium",
        reason=(
            f"Contains {len(roi_folders)} ROI image folder(s); representative "
            f"channel signatures are {'consistent' if consistent else 'variable'}."
        ),
        details={
            "rois": tuple(folder.name for folder in roi_folders),
            "channels": signatures[0] if consistent else (),
            "sample_signatures": tuple(signatures),
            "consistent_channels": consistent,
        },
    )


def _inspect_mapping_table(path: Path) -> AssetCandidate:
    import pandas as pd

    if not path.is_file():
        raise FileNotFoundError(f"Mapping table not found: {path}")
    frame = pd.read_csv(path, nrows=20)
    columns = tuple(str(value) for value in frame.columns)
    supported_pairs = (
        ("label_value", "label_name"),
        ("original_num", "name"),
        ("value", "name"),
        ("label", "name"),
    )
    pair = next(
        (
            candidate
            for candidate in supported_pairs
            if candidate[0] in columns and candidate[1] in columns
        ),
        None,
    )
    return AssetCandidate(
        candidate_id=f"mapping_table:{path}",
        kind="mapping_table",
        path=path,
        confidence="high" if pair is not None else "low",
        reason=(
            f"Contains supported value/name columns {pair}."
            if pair is not None
            else "CSV does not contain a recognized value/name column pair."
        ),
        details={
            "columns": columns,
            "value_key": None if pair is None else pair[0],
            "name_key": None if pair is None else pair[1],
        },
    )


def _candidate_key(kind: CandidateKind, path: Path) -> tuple[CandidateKind, Path]:
    return kind, path.resolve(strict=False)


def _explicit_paths(
    root: Path, hints: SpatialDataAssetHints
) -> tuple[tuple[CandidateKind, Path], ...]:
    values: list[tuple[CandidateKind, Path]] = []
    if isinstance(hints.anndata, (str, Path)):
        values.append(("anndata", _resolve(root, hints.anndata)))
    if hints.cell_masks is not None:
        values.append(("raster_collection", _resolve(root, hints.cell_masks)))
    if hints.primary_images is not None:
        values.append(("roi_image_collection", _resolve(root, hints.primary_images)))
    for item in hints.additional_images:
        values.append(("roi_image_collection", _resolve(root, item.folder)))
    for item in hints.histology:
        values.append(("raster_collection", _resolve(root, item.folder)))
    for item in hints.region_labels:
        values.append(("raster_collection", _resolve(root, item.folder)))
        if isinstance(item.value_names, (str, Path)):
            values.append(("mapping_table", _resolve(root, item.value_names)))
    for item in hints.maxfuse:
        if isinstance(item.adata, (str, Path)):
            values.append(("anndata", _resolve(root, item.adata)))
    return tuple(values)


def discover_spatialdata_assets(
    root: PathLike,
    *,
    hints: SpatialDataAssetHints | None = None,
    options: SpatialDataDiscoveryOptions | None = None,
) -> SpatialDataAssetInventory:
    """Inventory likely SpatialData inputs beneath ``root``.

    Explicit hints are always inspected even when they are outside the scan
    depth or project root.  Discovery is read-only and bounded by
    :class:`SpatialDataDiscoveryOptions`.
    """

    selected_hints = hints or SpatialDataAssetHints()
    selected_options = options or SpatialDataDiscoveryOptions()
    resolved_root = Path(root).expanduser().resolve(strict=False)
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"SpatialData discovery root not found: {resolved_root}")
    directories, files, truncated = _bounded_tree(resolved_root, selected_options)
    issues: list[DiscoveryIssue] = []
    if truncated:
        issues.append(
            DiscoveryIssue(
                "warning",
                "scan_entry_limit",
                f"Discovery stopped after {selected_options.max_entries} entries.",
                resolved_root,
            )
        )

    candidates: dict[tuple[CandidateKind, Path], AssetCandidate] = {}

    def add(candidate: AssetCandidate) -> None:
        candidates[_candidate_key(candidate.kind, candidate.path)] = candidate

    for path in files:
        try:
            if path.suffix.casefold() == ".h5ad":
                add(_inspect_h5ad(path, selected_hints))
            elif path.suffix.casefold() == ".csv":
                mapping = _inspect_mapping_table(path)
                if mapping.confidence != "low":
                    add(mapping)
        except Exception as exc:
            issues.append(
                DiscoveryIssue("warning", "asset_inspection_failed", str(exc), path)
            )

    for directory in directories:
        try:
            direct_rasters = any(
                child.is_file() and child.suffix.casefold() in RASTER_EXTENSIONS
                for child in directory.iterdir()
            )
            roi_children = any(
                child.is_dir()
                and any(
                    file.is_file() and file.suffix.casefold() in RASTER_EXTENSIONS
                    for file in child.iterdir()
                )
                for child in directory.iterdir()
            )
            if direct_rasters:
                add(_inspect_raster_collection(directory, selected_options))
            if roi_children:
                add(_inspect_roi_image_collection(directory, selected_options))
        except Exception as exc:
            issues.append(
                DiscoveryIssue(
                    "warning", "collection_inspection_failed", str(exc), directory
                )
            )

    for kind, path in _explicit_paths(resolved_root, selected_hints):
        key = _candidate_key(kind, path)
        try:
            if kind == "anndata":
                candidate = _inspect_h5ad(path, selected_hints)
            elif kind == "raster_collection":
                candidate = _inspect_raster_collection(path, selected_options)
            elif kind == "roi_image_collection":
                candidate = _inspect_roi_image_collection(path, selected_options)
            else:
                candidate = _inspect_mapping_table(path)
            candidates[key] = replace(
                candidate,
                confidence="explicit",
                source="explicit",
                reason=f"Explicitly supplied by the user. {candidate.reason}",
            )
        except Exception as exc:
            issues.append(
                DiscoveryIssue("error", "explicit_asset_invalid", str(exc), path)
            )

    return SpatialDataAssetInventory(
        root=resolved_root,
        hints=selected_hints,
        candidates=tuple(
            sorted(
                candidates.values(),
                key=lambda candidate: (candidate.kind, str(candidate.path).casefold()),
            )
        ),
        issues=tuple(issues),
    )


def _infer_suffix(
    stems: Sequence[str], rois: Sequence[str]
) -> tuple[str | None, tuple[str, ...]]:
    by_casefold = {stem.casefold(): stem for stem in stems}
    suffix_counts: dict[str, list[str]] = {}
    for roi in rois:
        roi_fold = roi.casefold()
        for stem_fold, original in by_casefold.items():
            if stem_fold.startswith(roi_fold):
                suffix = original[len(roi) :]
                suffix_counts.setdefault(suffix, []).append(roi)
    if not suffix_counts:
        return None, ()
    suffix, matched = max(
        suffix_counts.items(), key=lambda item: (len(set(item[1])), -len(item[0]))
    )
    ordered = tuple(roi for roi in rois if roi in set(matched))
    return suffix, ordered


def _bounded_marker_match(channel_files: Sequence[str], marker: str) -> bool:
    import re

    exact = [value for value in channel_files if value.casefold() == marker.casefold()]
    if len(exact) == 1:
        return True
    pattern = re.compile(
        rf"(?<![0-9A-Za-z]){re.escape(marker)}(?![0-9A-Za-z])", re.IGNORECASE
    )
    return sum(bool(pattern.search(value)) for value in channel_files) == 1


def _panel_match_score(
    candidate: AssetCandidate, rois: Sequence[str], markers: Sequence[str]
) -> tuple[float, bool]:
    candidate_rois = {
        str(value).casefold() for value in candidate.details.get("rois", ())
    }
    coverage = (
        sum(roi.casefold() in candidate_rois for roi in rois) / len(rois)
        if rois
        else 0.0
    )
    signatures = tuple(
        tuple(str(value) for value in signature)
        for signature in candidate.details.get("sample_signatures", ())
    )
    if not signatures:
        channels = tuple(str(value) for value in candidate.details.get("channels", ()))
        signatures = (channels,) if channels else ()
    marker_match = bool(markers) and bool(signatures) and all(
        all(_bounded_marker_match(signature, marker) for marker in markers)
        for signature in signatures
    )
    return coverage, marker_match


def _mask_compatibility(
    adata_path: Path,
    candidate: AssetCandidate,
    *,
    roi_key: str,
    instance_key: str,
    rois: Sequence[str],
    sample_rois: int = 3,
) -> tuple[float, float]:
    import anndata as ad
    import numpy as np
    import tifffile

    suffix, matched = _infer_suffix(candidate.details.get("stems", ()), rois)
    if suffix is None or not matched:
        return 0.0, 0.0
    handle = ad.read_h5ad(adata_path, backed="r")
    try:
        obs = handle.obs[[roi_key, instance_key]]
        tested = 0
        compatible = 0
        expected_instances = 0
        present_instances = 0
        allowed = {
            str(value).casefold()
            for value in candidate.details.get("extensions", (".tif", ".tiff"))
        }
        for roi in matched[:sample_rois]:
            requested = f"{roi}{suffix}".casefold()
            paths = [
                path
                for path in candidate.path.iterdir()
                if path.is_file()
                and path.suffix.casefold() in allowed
                and path.stem.casefold() == requested
            ]
            if len(paths) != 1:
                continue
            expected = set(
                map(
                    int,
                    obs.loc[obs[roi_key].astype(str) == roi, instance_key].tolist(),
                )
            )
            present = set(map(int, np.unique(tifffile.imread(paths[0]))))
            present.discard(0)
            tested += 1
            compatible += int(expected.issubset(present))
            expected_instances += len(expected)
            present_instances += len(present)
        return (
            compatible / tested if tested else 0.0,
            (
                present_instances / expected_instances
                if expected_instances
                else 0.0
            ),
        )
    finally:
        if getattr(handle, "file", None) is not None:
            handle.file.close()


def _select_explicit(
    inventory: SpatialDataAssetInventory,
    kind: CandidateKind,
    value: PathLike | None,
) -> AssetCandidate | None:
    if value is None:
        return None
    return inventory.candidate_for_path(kind, value)


def _choose_primary_anndata(
    inventory: SpatialDataAssetInventory,
    issues: list[DiscoveryIssue],
) -> AssetCandidate | None:
    explicit = _select_explicit(inventory, "anndata", inventory.hints.anndata)
    if explicit is not None:
        return explicit
    candidates = [
        candidate
        for candidate in inventory.by_kind("anndata")
        if candidate.confidence == "high"
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        issues.append(
            DiscoveryIssue(
                "error",
                "primary_anndata_missing",
                "No AnnData candidate contains all configured spatial columns; "
                "specify hints.anndata explicitly.",
                inventory.root,
            )
        )
    else:
        issues.append(
            DiscoveryIssue(
                "error",
                "primary_anndata_ambiguous",
                f"{len(candidates)} AnnData files contain all configured spatial "
                "columns; specify hints.anndata explicitly.",
                inventory.root,
            )
        )
    return None


def _candidate_name(candidate: AssetCandidate) -> str:
    return _safe_name(candidate.path.name)


def _read_obs_names(path: Path) -> tuple[str, ...]:
    import anndata as ad

    handle = ad.read_h5ad(path, backed="r")
    try:
        return tuple(map(str, handle.obs_names))
    finally:
        if getattr(handle, "file", None) is not None:
            handle.file.close()


def propose_spatialdata_spec(
    inventory: SpatialDataAssetInventory,
) -> SpatialDataAssetProposal:
    """Resolve explicit hints and conservative high-confidence relationships."""

    hints = inventory.hints
    issues = list(inventory.issues)
    selected: dict[str, str] = {}
    modalities: list[Any] = []
    primary = _choose_primary_anndata(inventory, issues)
    if primary is None:
        return SpatialDataAssetProposal(
            inventory=inventory,
            spec=None,
            selected=selected,
            issues=tuple(issues),
        )
    selected["primary_anndata"] = primary.candidate_id
    rois = tuple(str(value) for value in primary.details.get("rois", ()))
    markers = tuple(str(value) for value in primary.details.get("var_names", ()))
    if not rois:
        issues.append(
            DiscoveryIssue(
                "error",
                "primary_anndata_has_no_rois",
                f"Configured ROI key {hints.roi_key!r} produced no ROI values.",
                primary.path,
            )
        )

    explicit_masks = _select_explicit(
        inventory, "raster_collection", hints.cell_masks
    )
    masks = explicit_masks
    if masks is None and rois:
        scored: list[tuple[float, float, float, AssetCandidate]] = []
        for candidate in inventory.by_kind("raster_collection"):
            if candidate.details.get("raster_kind") != "integer_labels":
                continue
            suffix, matched = _infer_suffix(candidate.details.get("stems", ()), rois)
            coverage = len(matched) / len(rois) if rois and suffix is not None else 0.0
            compatibility, instance_support = _mask_compatibility(
                primary.path,
                candidate,
                roi_key=hints.roi_key,
                instance_key=hints.instance_key,
                rois=rois,
            )
            if compatibility == 1.0:
                scored.append(
                    (coverage, compatibility, instance_support, candidate)
                )
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        if scored:
            best = scored[0]
            tied = [item for item in scored if item[:3] == best[:3]]
            if len(tied) == 1:
                masks = best[3]
    if masks is None:
        issues.append(
            DiscoveryIssue(
                "error",
                "cell_masks_unresolved",
                "No unique integer-raster collection contains the configured "
                "AnnData instances; specify hints.cell_masks explicitly.",
                inventory.root,
            )
        )
    else:
        selected["cell_masks"] = masks.candidate_id

    explicit_images = _select_explicit(
        inventory, "roi_image_collection", hints.primary_images
    )
    primary_images = explicit_images
    if primary_images is None and rois and markers:
        scored_panels = []
        for candidate in inventory.by_kind("roi_image_collection"):
            coverage, marker_match = _panel_match_score(candidate, rois, markers)
            if marker_match:
                scored_panels.append((coverage, candidate))
        scored_panels.sort(key=lambda item: item[0], reverse=True)
        if scored_panels:
            best_coverage = scored_panels[0][0]
            tied = [
                candidate
                for coverage, candidate in scored_panels
                if coverage == best_coverage
            ]
            if len(tied) == 1:
                primary_images = tied[0]
    if primary_images is None:
        issues.append(
            DiscoveryIssue(
                "error",
                "primary_images_unresolved",
                "No unique ROI image collection matches every AnnData marker; "
                "specify hints.primary_images explicitly.",
                inventory.root,
            )
        )
    else:
        selected["primary_images"] = primary_images.candidate_id

    if masks is None or primary_images is None:
        return SpatialDataAssetProposal(
            inventory=inventory,
            spec=None,
            selected=selected,
            issues=tuple(issues),
        )

    modalities.extend(
        [
            CellMasks(name=hints.cell_masks_name, folder=masks.path),
            IMCImages(
                name=hints.primary_images_name,
                panel_name=hints.primary_panel_name,
                folder=primary_images.path,
            ),
            IMCAnnData(
                name=hints.primary_table_name,
                panel_name=hints.primary_panel_name,
                adata=primary.path,
                images=hints.primary_images_name,
                masks=hints.cell_masks_name,
                roi_key=hints.roi_key,
                instance_key=hints.instance_key,
                x_key=hints.x_key,
                y_key=hints.y_key,
                copy_adata=hints.copy_adata,
            ),
        ]
    )

    used_paths = {masks.path, primary_images.path, primary.path}
    explicit_panel_paths: set[Path] = set()
    for item in hints.additional_images:
        folder = _resolve(inventory.root, item.folder)
        explicit_panel_paths.add(folder)
        candidate = inventory.candidate_for_path("roi_image_collection", folder)
        if candidate is not None:
            selected[f"image_panel:{item.name}"] = candidate.candidate_id
        channels = (
            tuple(map(str, item.channels))
            if item.channels is not None
            else tuple(
                str(value)
                for value in (candidate.details.get("channels", ()) if candidate else ())
            )
        )
        if not channels:
            issues.append(
                DiscoveryIssue(
                    "error",
                    "explicit_image_panel_channels_missing",
                    f"Could not infer channels for explicit image panel {item.name!r}; "
                    "supply channels.",
                    folder,
                )
            )
            continue
        modalities.append(
            IMCImages(
                name=item.name,
                panel_name=item.panel_name or item.name,
                folder=folder,
                channels=channels,
                reference=item.reference,
                allow_partial=item.allow_partial,
                transformations=item.transformations,
                match_mode=(
                    "exact"
                    if item.channels is None
                    else "exact_or_unique_substring"
                ),
            )
        )
        used_paths.add(folder)

    if hints.discover_unlisted_assets and hints.include_discovered_image_panels:
        for candidate in inventory.by_kind("roi_image_collection"):
            if candidate.path in used_paths or candidate.path in explicit_panel_paths:
                continue
            coverage, marker_match = _panel_match_score(candidate, rois, markers)
            channels = tuple(
                str(value) for value in candidate.details.get("channels", ())
            )
            if (
                coverage <= 0
                or marker_match
                or not channels
                or not candidate.details.get("consistent_channels")
            ):
                continue
            name = f"images_{_candidate_name(candidate)}"
            modalities.append(
                IMCImages(
                    name=name,
                    panel_name=candidate.path.name,
                    folder=candidate.path,
                    channels=channels,
                    reference=hints.cell_masks_name,
                    allow_partial=True,
                    match_mode="exact",
                )
            )
            selected[f"image_panel:{name}"] = candidate.candidate_id
            used_paths.add(candidate.path)

    explicit_histology_paths: set[Path] = set()
    for item in hints.histology:
        folder = _resolve(inventory.root, item.folder)
        explicit_histology_paths.add(folder)
        candidate = inventory.candidate_for_path("raster_collection", folder)
        if candidate is not None:
            selected[f"histology:{item.name}"] = candidate.candidate_id
        modalities.append(
            HistologyImages(
                name=item.name,
                folder=folder,
                reference=item.reference,
                suffix=item.suffix,
                allow_partial=item.allow_partial,
                drop_alpha=item.drop_alpha,
                transformations=item.transformations,
            )
        )
        used_paths.add(folder)

    if hints.discover_unlisted_assets and hints.include_discovered_histology:
        for candidate in inventory.by_kind("raster_collection"):
            if candidate.path in used_paths or candidate.path in explicit_histology_paths:
                continue
            if candidate.details.get("raster_kind") != "rgb":
                continue
            suffix, matched = _infer_suffix(candidate.details.get("stems", ()), rois)
            if suffix is None or not matched:
                continue
            name = f"histology_{_candidate_name(candidate)}"
            modalities.append(
                HistologyImages(
                    name=name,
                    folder=candidate.path,
                    reference=hints.cell_masks_name,
                    suffix=suffix,
                    allow_partial=True,
                )
            )
            selected[f"histology:{name}"] = candidate.candidate_id
            used_paths.add(candidate.path)

    for item in hints.region_labels:
        folder = _resolve(inventory.root, item.folder)
        value_names = (
            _resolve(inventory.root, item.value_names)
            if isinstance(item.value_names, (str, Path))
            else item.value_names
        )
        candidate = inventory.candidate_for_path("raster_collection", folder)
        if candidate is not None:
            selected[f"region_labels:{item.name}"] = candidate.candidate_id
        modalities.append(
            RegionLabels(
                name=item.name,
                folder=folder,
                suffix=item.suffix,
                value_names=value_names,
                reference=item.reference,
                allow_partial=item.allow_partial,
                value_key=item.value_key,
                name_key=item.name_key,
                mapping_roi_key=item.mapping_roi_key,
                transformations=item.transformations,
            )
        )
        used_paths.add(folder)

    for candidate in inventory.by_kind("raster_collection"):
        if (
            candidate.path not in used_paths
            and candidate.details.get("raster_kind") == "integer_labels"
        ):
            suffix, matched = _infer_suffix(candidate.details.get("stems", ()), rois)
            if suffix is not None and matched:
                issues.append(
                    DiscoveryIssue(
                        "info",
                        "unselected_region_labels",
                        "Likely categorical label rasters were discovered but require "
                        "an explicit semantic value-name mapping before inclusion.",
                        candidate.path,
                    )
                )

    explicit_maxfuse_paths: set[Path] = set()
    for item in hints.maxfuse:
        source = (
            _resolve(inventory.root, item.adata)
            if isinstance(item.adata, (str, Path))
            else item.adata
        )
        if isinstance(source, Path):
            explicit_maxfuse_paths.add(source)
            candidate = inventory.candidate_for_path("anndata", source)
            if candidate is not None:
                selected[f"maxfuse:{item.name}"] = candidate.candidate_id
        modalities.append(
            MaxFuseSCRNASeq(
                name=item.name,
                adata=source,
                imc_table=item.imc_table,
                table_name=item.table_name,
                copy_adata=item.copy_adata,
            )
        )

    if hints.discover_unlisted_assets and hints.include_discovered_maxfuse:
        primary_obs = set(_read_obs_names(primary.path))
        for candidate in inventory.by_kind("anndata"):
            if candidate.path == primary.path or candidate.path in explicit_maxfuse_paths:
                continue
            likely_maxfuse = (
                "maxfuse" in {str(value).casefold() for value in candidate.details.get("uns_keys", ())}
                or int(candidate.details.get("n_vars", 0)) > int(primary.details.get("n_vars", 0))
            )
            if not likely_maxfuse:
                continue
            source_obs = set(_read_obs_names(candidate.path))
            if source_obs and source_obs.issubset(primary_obs):
                name = f"maxfuse_{_candidate_name(candidate)}"
                modalities.append(
                    MaxFuseSCRNASeq(
                        name=name,
                        adata=candidate.path,
                        imc_table=hints.primary_table_name,
                    )
                )
                selected[f"maxfuse:{name}"] = candidate.candidate_id

    if any(issue.severity == "error" for issue in issues):
        return SpatialDataAssetProposal(
            inventory=inventory,
            spec=None,
            selected=selected,
            issues=tuple(issues),
        )
    spec = SpatialDataSpec(
        modalities=modalities,
        attrs=hints.attrs,
        raster_chunks=hints.raster_chunks,
        scale_factors=hints.scale_factors,
    )
    return SpatialDataAssetProposal(
        inventory=inventory,
        spec=spec,
        selected=selected,
        issues=tuple(issues),
    )


def plan_spatialdata_from_assets(
    root: PathLike,
    *,
    hints: SpatialDataAssetHints | None = None,
    options: SpatialDataDiscoveryOptions | None = None,
) -> SpatialDataAssetPlan:
    """Discover assets, propose a specification, and run the strict planner."""

    from .core import plan_spatialdata

    inventory = discover_spatialdata_assets(root, hints=hints, options=options)
    proposal = propose_spatialdata_spec(inventory)
    spatialdata_plan = (
        None if proposal.spec is None else plan_spatialdata(proposal.spec)
    )
    return SpatialDataAssetPlan(
        inventory=inventory,
        proposal=proposal,
        spatialdata_plan=spatialdata_plan,
    )


def build_spatialdata_from_assets(
    root: PathLike,
    output_path: PathLike,
    *,
    hints: SpatialDataAssetHints | None = None,
    options: SpatialDataDiscoveryOptions | None = None,
    asset_plan: SpatialDataAssetPlan | None = None,
) -> SpatialDataBuildResult:
    """Create and write a new SpatialData Zarr from a validated asset plan."""

    from ._legacy import write_spatialdata
    from .core import create_spatialdata

    selected_plan = asset_plan or plan_spatialdata_from_assets(
        root, hints=hints, options=options
    )
    selected_plan.raise_for_errors()
    assert selected_plan.spatialdata_plan is not None
    resolved_root = Path(root).expanduser().resolve(strict=False)
    output = _resolve(resolved_root, output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing SpatialData store: {output}")
    sdata = create_spatialdata(selected_plan.spatialdata_plan)
    written = write_spatialdata(sdata, output)
    return SpatialDataBuildResult(
        output_path=written,
        asset_plan=selected_plan,
        element_counts={
            "images": len(sdata.images),
            "labels": len(sdata.labels),
            "points": len(sdata.points),
            "shapes": len(sdata.shapes),
            "tables": len(sdata.tables),
        },
    )


__all__ = [
    "AssetCandidate",
    "DiscoveryIssue",
    "HistologyAssetHint",
    "IMCImageAssetHint",
    "MaxFuseAssetHint",
    "RegionLabelsAssetHint",
    "SpatialDataAssetHints",
    "SpatialDataAssetInventory",
    "SpatialDataAssetPlan",
    "SpatialDataAssetProposal",
    "SpatialDataBuildResult",
    "SpatialDataDiscoveryOptions",
    "build_spatialdata_from_assets",
    "discover_spatialdata_assets",
    "plan_spatialdata_from_assets",
    "propose_spatialdata_spec",
]
