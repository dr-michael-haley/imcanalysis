"""Synchronized, previewable maintenance operations for NapariSBT datasets."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from .anndata_io import write_h5ad_compat
from .colour_helper import categorical_colour_collisions, normalise_hex_colour

ReadinessLevel = Literal["ready", "warning", "blocked", "optional"]
MaskRebuildMode = Literal["preserve", "compact"]
CellFilterMode = Literal[
    "keep_values",
    "remove_values",
    "keep_range",
    "remove_range",
    "keep_missing",
    "remove_missing",
]


class MaintenanceCheck(BaseModel):
    """One concise readiness result for the Dataset Maintenance dashboard."""

    key: str
    label: str
    level: ReadinessLevel
    detail: str


class MaintenancePreview(BaseModel):
    """Side-effect-free preview shared by maintenance operations."""

    operation: str
    summary: str
    checks: list[MaintenanceCheck] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return not any(check.level == "blocked" for check in self.checks)


class CellFilterRequest(BaseModel):
    """A cell filter which can be previewed before slicing AnnData."""

    observation: str
    mode: CellFilterMode
    values: list[str] = Field(default_factory=list)
    lower: float | None = None
    upper: float | None = None

    @model_validator(mode="after")
    def validate_filter(self) -> CellFilterRequest:
        self.observation = str(self.observation).strip()
        self.values = list(dict.fromkeys(str(value) for value in self.values))
        if not self.observation:
            raise ValueError("Choose an AnnData observation to filter.")
        if self.mode in {"keep_values", "remove_values"} and not self.values:
            raise ValueError("Select at least one observation value.")
        if self.mode in {"keep_range", "remove_range"}:
            if self.lower is None or self.upper is None:
                raise ValueError("Enter both ends of the numeric range.")
            if float(self.lower) > float(self.upper):
                raise ValueError(
                    "The numeric lower limit cannot exceed the upper limit."
                )
        return self


class ImageRenameItem(BaseModel):
    """One deterministic image copy/rename action."""

    roi: str
    channel_before: str
    channel_after: str
    source: Path
    destination: Path


class ImageRenamePlan(BaseModel):
    """A validated collection of image copy/rename actions."""

    output_root: Path
    items: list[ImageRenameItem] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    collisions: list[str] = Field(default_factory=list)

    @property
    def ready(self) -> bool:
        return bool(self.items) and not self.unresolved and not self.collisions


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_write_anndata(
    adata,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the current AnnData atomically, optionally replacing one exact file."""

    output = Path(destination).expanduser().resolve(strict=False)
    if output.suffix.lower() != ".h5ad":
        raise ValueError("The AnnData destination must end in .h5ad.")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"AnnData already exists: {output}. Choose a new file or explicitly "
            "allow replacement."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{uuid4().hex}.tmp{output.suffix}")
    try:
        write_h5ad_compat(adata, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def remap_categorical_observation(
    adata,
    source: str,
    destination: str,
    mapping: Mapping[str, str],
    colours: Mapping[str, str],
    *,
    overwrite: bool = False,
):
    """Return a copy with a renamed/merged categorical observation and palette.

    Source values are matched through their displayed string representation, as
    they are in the Dataset Maintenance table. Missing values remain missing.
    Repeating a proposed name is an explicit merge. Distinct output categories
    may not silently share one colour.
    """

    source = str(source).strip()
    destination = str(destination).strip()
    if source not in adata.obs:
        raise ValueError(f"Observation does not exist: {source!r}.")
    if not destination:
        raise ValueError("Enter an output observation name.")
    if destination in adata.obs and not overwrite:
        raise ValueError(
            f"Observation already exists: {destination!r}. Enable overwrite only "
            "when replacing that exact live column is intended."
        )

    source_values = adata.obs[source].astype("string")
    available = source_values.dropna().astype(str).drop_duplicates().tolist()
    cleaned_mapping = {
        str(key).strip(): str(value).strip() for key, value in mapping.items()
    }
    missing = [value for value in available if value not in cleaned_mapping]
    blank = [value for value in available if not cleaned_mapping.get(value, "")]
    if missing or blank:
        affected = list(dict.fromkeys([*missing, *blank]))
        raise ValueError(
            "Every source value needs a proposed name. Missing: "
            + ", ".join(repr(value) for value in affected[:12])
        )

    category_order = list(
        dict.fromkeys(cleaned_mapping[value] for value in available)
    )
    cleaned_colours = {
        str(label).strip(): normalise_hex_colour(colour)
        for label, colour in colours.items()
    }
    missing_colours = [
        label for label in category_order if not cleaned_colours.get(label, "")
    ]
    if missing_colours:
        raise ValueError(
            "Every final population needs a valid #RRGGBB colour. Missing: "
            + ", ".join(repr(value) for value in missing_colours[:12])
        )
    collisions = categorical_colour_collisions(
        category_order,
        [cleaned_colours[label] for label in category_order],
    )
    if collisions:
        details = "; ".join(
            f"{colour}: {', '.join(labels)}"
            for colour, labels in collisions.items()
        )
        raise ValueError(
            "Different final populations cannot share one colour. " + details
        )

    mapped = source_values.map(cleaned_mapping)
    result = adata.copy()
    result.obs[destination] = pd.Categorical(mapped, categories=category_order)
    result.uns[f"{destination}_colors"] = [
        cleaned_colours[label] for label in category_order
    ]
    return result


def normalise_var_rename_mapping(
    adata,
    mapping: Mapping[str, str],
) -> dict[str, str]:
    """Validate and normalize a variable rename mapping."""

    available = [str(value) for value in adata.var_names]
    available_set = set(available)
    cleaned: dict[str, str] = {}
    for source, destination in mapping.items():
        source_text = str(source).strip()
        destination_text = str(destination).strip()
        if not source_text or not destination_text or source_text == destination_text:
            continue
        if source_text not in available_set:
            raise ValueError(f"AnnData variable does not exist: {source_text!r}.")
        cleaned[source_text] = destination_text
    if not cleaned:
        raise ValueError("Enter at least one variable rename.")
    resulting = [cleaned.get(value, value) for value in available]
    duplicated = pd.Index(resulting)[pd.Index(resulting).duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(
            "Variable renaming would create duplicate names: "
            + ", ".join(map(str, duplicated[:10]))
        )
    return cleaned


def preview_var_rename(
    adata,
    mapping: Mapping[str, str],
    *,
    image_index: Mapping[str, Mapping[str, Path]] | None = None,
) -> MaintenancePreview:
    cleaned = normalise_var_rename_mapping(adata, mapping)
    matched_images = 0
    represented = set()
    for roi_images in (image_index or {}).values():
        for channel in roi_images:
            logical_channel = str(channel).split(" [", 1)[0]
            if logical_channel in cleaned:
                matched_images += 1
                represented.add(logical_channel)
    missing_images = sorted(set(cleaned) - represented)
    checks = [
        MaintenanceCheck(
            key="variables",
            label="Variable mapping",
            level="ready",
            detail=f"{len(cleaned):,} AnnData variable(s) will be renamed.",
        ),
        MaintenanceCheck(
            key="images",
            label="Indexed staining images",
            level="warning" if missing_images else "ready",
            detail=(
                f"{matched_images:,} indexed image(s) match the renamed variables."
                + (
                    " No indexed image matched: " + ", ".join(missing_images[:8]) + "."
                    if missing_images
                    else ""
                )
            ),
        ),
    ]
    return MaintenancePreview(
        operation="rename_variables",
        summary=f"Rename {len(cleaned):,} variables and {matched_images:,} indexed images.",
        checks=checks,
        details={"mapping": cleaned, "matched_images": matched_images},
    )


def apply_var_rename(
    adata,
    mapping: Mapping[str, str],
    *,
    update_raw: bool = False,
):
    """Return a copied AnnData with validated variable and marker metadata renames."""

    cleaned = normalise_var_rename_mapping(adata, mapping)
    result = adata.copy()
    result.var_names = [
        cleaned.get(str(value), str(value)) for value in result.var_names
    ]
    for column in ("channel_label", "marker", "marker_name"):
        if column not in result.var:
            continue
        values = result.var[column].astype("string")
        result.var[column] = values.map(lambda value: cleaned.get(str(value), value))
    if update_raw and result.raw is not None:
        raw = result.raw.to_adata()
        renamed_raw_vars = [
            cleaned.get(str(value), str(value)) for value in raw.var_names
        ]
        duplicated_raw = (
            pd.Index(renamed_raw_vars)[pd.Index(renamed_raw_vars).duplicated()]
            .unique()
            .tolist()
        )
        if duplicated_raw:
            raise ValueError(
                "Applying the mapping to adata.raw would create duplicate names: "
                + ", ".join(map(str, duplicated_raw[:10]))
            )
        raw.var_names = renamed_raw_vars
        for column in ("channel_label", "marker", "marker_name"):
            if column in raw.var:
                values = raw.var[column].astype("string")
                raw.var[column] = values.map(
                    lambda value: cleaned.get(str(value), value)
                )
        result.raw = raw
    return result


def remove_anndata_vars(
    adata,
    variables: Sequence[str],
    *,
    subset_raw: bool = False,
):
    """Return an AnnData copy without selected variables; images are untouched."""

    requested = list(dict.fromkeys(str(value) for value in variables))
    available = pd.Index(adata.var_names.astype(str))
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError("Unknown AnnData variables: " + ", ".join(unknown[:10]))
    if not requested:
        raise ValueError("Select at least one variable to remove.")
    keep = ~available.isin(requested)
    if not bool(keep.any()):
        raise ValueError("Refusing to remove every AnnData variable.")
    result = adata[:, keep].copy()
    if subset_raw and adata.raw is not None:
        raw = adata.raw.to_adata()
        raw_keep_names = [name for name in result.var_names if name in raw.var_names]
        result.raw = raw[:, raw_keep_names].copy()
    return result


def resolve_cell_filter(adata, request: CellFilterRequest) -> np.ndarray:
    """Resolve one filter to a retained-cell Boolean mask."""

    if request.observation not in adata.obs:
        raise ValueError(
            f"AnnData observation does not exist: {request.observation!r}."
        )
    series = adata.obs[request.observation]
    if request.mode in {"keep_values", "remove_values"}:
        selected = (
            series.astype("string").isin(request.values).fillna(False).to_numpy(bool)
        )
        return selected if request.mode == "keep_values" else ~selected
    if request.mode in {"keep_missing", "remove_missing"}:
        missing = series.isna().to_numpy(bool)
        return missing if request.mode == "keep_missing" else ~missing
    numeric = pd.to_numeric(series, errors="coerce")
    if int(numeric.notna().sum()) == 0:
        raise ValueError(
            f"adata.obs[{request.observation!r}] contains no numeric values."
        )
    within = numeric.between(
        float(request.lower), float(request.upper), inclusive="both"
    ).fillna(False)
    mask = within.to_numpy(bool)
    return mask if request.mode == "keep_range" else ~mask


def preview_cell_filter(
    adata, request: CellFilterRequest, *, roi_obs: str
) -> MaintenancePreview:
    mask = resolve_cell_filter(adata, request)
    retained = int(mask.sum())
    removed = int(len(mask) - retained)
    represented_rois = (
        int(adata.obs.loc[mask, roi_obs].astype(str).nunique())
        if roi_obs in adata.obs and retained
        else 0
    )
    checks = [
        MaintenanceCheck(
            key="retained_cells",
            label="Retained cells",
            level="blocked" if retained == 0 else "ready",
            detail=f"{retained:,} retained; {removed:,} removed.",
        ),
        MaintenanceCheck(
            key="mask_sync",
            label="Mask synchronization",
            level="warning" if removed else "optional",
            detail=(
                "Create derived masks after applying this filter if mask files should "
                "contain only retained cells."
                if removed
                else "No cells would be removed from masks."
            ),
        ),
    ]
    return MaintenancePreview(
        operation="filter_cells",
        summary=(
            f"Retain {retained:,}/{len(mask):,} cells across {represented_rois:,} ROIs."
        ),
        checks=checks,
        details={
            "retained_cells": retained,
            "removed_cells": removed,
            "represented_rois": represented_rois,
        },
    )


def apply_cell_filter(adata, request: CellFilterRequest):
    mask = resolve_cell_filter(adata, request)
    if not bool(mask.any()):
        raise ValueError("The requested filter would remove every cell.")
    return adata[mask].copy()


def _replace_filename_token(path: Path, source: str, destination: str) -> str | None:
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(source)}(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    matches = list(pattern.finditer(path.stem))
    if len(matches) != 1:
        return None
    match = matches[0]
    renamed_stem = path.stem[: match.start()] + destination + path.stem[match.end() :]
    return renamed_stem + path.suffix


def plan_image_renames(
    image_index: Mapping[str, Mapping[str, Path]],
    mapping: Mapping[str, str],
    *,
    image_roots: Sequence[str | Path],
    output_root: str | Path,
) -> ImageRenamePlan:
    """Plan copy-on-write image renames without rescanning image folders."""

    roots = [Path(root).expanduser().resolve(strict=False) for root in image_roots]
    output = Path(output_root).expanduser().resolve(strict=False)
    items: list[ImageRenameItem] = []
    unresolved: list[str] = []
    destinations: dict[Path, Path] = {}
    collisions: list[str] = []
    invalid_filename = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
    for roi, channels in image_index.items():
        for channel, raw_source in channels.items():
            logical_channel = str(channel).split(" [", 1)[0]
            replacement = str(mapping.get(logical_channel, logical_channel))
            renamed = logical_channel in mapping
            if renamed and invalid_filename.search(replacement):
                unresolved.append(
                    f"{roi}/{logical_channel}: {replacement!r} is not a safe filename component"
                )
                continue
            source = Path(raw_source).expanduser().resolve(strict=False)
            if not source.is_file():
                unresolved.append(f"{roi}/{channel}: indexed source no longer exists")
                continue
            renamed_name = (
                _replace_filename_token(source, logical_channel, replacement)
                if renamed
                else source.name
            )
            if renamed_name is None:
                unresolved.append(
                    f"{roi}/{logical_channel}: {source.name} does not contain exactly "
                    "one safe old-name token"
                )
                continue
            root_index = None
            relative = None
            for index, root in enumerate(roots, start=1):
                try:
                    relative = source.relative_to(root)
                    root_index = index
                    break
                except ValueError:
                    continue
            if root_index is None or relative is None:
                unresolved.append(
                    f"{roi}/{channel}: source is outside configured folders"
                )
                continue
            folder_name = roots[root_index - 1].name or f"images_{root_index}"
            destination_path = (
                output
                / f"{root_index:02d}_{folder_name}"
                / relative.parent
                / renamed_name
            )
            existing = destinations.get(destination_path)
            if existing is not None and existing != source:
                collisions.append(f"{existing} and {source} -> {destination_path}")
                continue
            if existing == source:
                continue
            destinations[destination_path] = source
            items.append(
                ImageRenameItem(
                    roi=str(roi),
                    channel_before=logical_channel,
                    channel_after=replacement,
                    source=source,
                    destination=destination_path,
                )
            )
    return ImageRenamePlan(
        output_root=output,
        items=items,
        unresolved=unresolved,
        collisions=collisions,
    )


def copy_renamed_images(plan: ImageRenamePlan) -> Path:
    """Materialize one image rename plan into a new output folder."""

    if not plan.ready:
        raise ValueError(
            "The image rename plan is not ready. Resolve unmatched files or collisions."
        )
    output = plan.output_root
    if output.exists():
        raise FileExistsError(
            f"Refusing to merge into an existing image output folder: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for item in plan.items:
            relative = item.destination.relative_to(output)
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.source, destination)
        mapping = pd.DataFrame(
            [
                {
                    "ROI": item.roi,
                    "channel_before": item.channel_before,
                    "channel_after": item.channel_after,
                    "source": str(item.source),
                    "destination": str(item.destination),
                }
                for item in plan.items
            ]
        )
        mapping.to_csv(temporary / "image_rename_crosswalk.csv", index=False)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def identity_frame(adata, *, roi_obs: str, object_obs: str) -> pd.DataFrame:
    """Return validated observation identities for maintenance operations."""

    missing = [column for column in (roi_obs, object_obs) if column not in adata.obs]
    if missing:
        raise ValueError("Missing identity observations: " + ", ".join(missing))
    roi_values = adata.obs[roi_obs]
    invalid_roi = roi_values.isna() | roi_values.astype("string").str.strip().eq("")
    if invalid_roi.any():
        raise ValueError(
            f"adata.obs[{roi_obs!r}] must not contain missing or blank ROI values."
        )
    object_ids = pd.to_numeric(adata.obs[object_obs], errors="coerce")
    invalid = object_ids.isna() | object_ids.le(0) | object_ids.mod(1).ne(0)
    if invalid.any():
        raise ValueError(
            f"adata.obs[{object_obs!r}] must contain positive integer mask labels."
        )
    frame = pd.DataFrame(
        {
            "obs_name": adata.obs_names.astype(str),
            "ROI": roi_values.astype("string"),
            "ObjectNumber": object_ids.astype("int64"),
        }
    )
    duplicates = frame.duplicated(["ROI", "ObjectNumber"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Each cell must have a unique (ROI, ObjectNumber) identity before masks "
            "can be rebuilt."
        )
    return frame


def preview_mask_rebuild(
    adata,
    mask_paths: Mapping[str, Path],
    *,
    roi_obs: str,
    object_obs: str,
    mode: MaskRebuildMode,
) -> MaintenancePreview:
    """Inspect identity and mask-label coverage; mask reads occur only on request."""

    if mode not in {"preserve", "compact"}:
        raise ValueError("Mask rebuild mode must be 'preserve' or 'compact'.")
    identities = identity_frame(adata, roi_obs=roi_obs, object_obs=object_obs)
    represented_rois = identities["ROI"].astype(str).drop_duplicates().tolist()
    missing_masks = [roi for roi in represented_rois if roi not in mask_paths]
    missing_labels: list[str] = []
    extra_label_count = 0
    from tifffile import imread

    for roi, rows in identities.groupby("ROI", observed=True, sort=False):
        mask_path = mask_paths.get(str(roi))
        if mask_path is None:
            continue
        mask = np.asarray(imread(mask_path))
        if mask.ndim != 2:
            raise ValueError(f"Mask for ROI {roi!r} is not two-dimensional.")
        mask_ids = set(np.unique(mask).astype(np.int64).tolist()) - {0}
        object_ids = set(rows["ObjectNumber"].astype(int).tolist())
        absent = sorted(object_ids - mask_ids)
        if absent:
            missing_labels.append(f"{roi}: {absent[:8]}")
        extra_label_count += len(mask_ids - object_ids)
    checks = [
        MaintenanceCheck(
            key="identity",
            label="Cell identity",
            level="ready",
            detail=f"{len(identities):,} unique cells across {len(represented_rois):,} ROIs.",
        ),
        MaintenanceCheck(
            key="masks",
            label="Mask files",
            level="blocked" if missing_masks else "ready",
            detail=(
                "Missing masks: " + ", ".join(missing_masks[:10])
                if missing_masks
                else f"Masks were found for all {len(represented_rois):,} ROIs."
            ),
        ),
        MaintenanceCheck(
            key="labels",
            label="ObjectNumber coverage",
            level="blocked" if missing_labels else "ready",
            detail=(
                "AnnData labels absent from masks: " + "; ".join(missing_labels[:5])
                if missing_labels
                else "Every AnnData ObjectNumber is present in its ROI mask."
            ),
        ),
        MaintenanceCheck(
            key="excluded_mask_labels",
            label="Mask-only objects",
            level="warning" if extra_label_count else "ready",
            detail=(
                f"{extra_label_count:,} mask object(s) not represented in AnnData will "
                "be set to background."
                if extra_label_count
                else "Masks contain no extra objects outside the current AnnData."
            ),
        ),
    ]
    mode_text = "preserve existing IDs" if mode == "preserve" else "compact IDs per ROI"
    return MaintenancePreview(
        operation="rebuild_masks",
        summary=(
            f"Write {len(represented_rois):,} derived masks and {mode_text}; original "
            "masks remain untouched."
        ),
        checks=checks,
        details={
            "roi_count": len(represented_rois),
            "cell_count": len(identities),
            "extra_mask_labels": extra_label_count,
            "mode": mode,
        },
    )


def _remap_mask(mask: np.ndarray, mapping: Mapping[int, int]) -> np.ndarray:
    unique, inverse = np.unique(mask.astype(np.int64, copy=False), return_inverse=True)
    translated = np.asarray([mapping.get(int(value), 0) for value in unique])
    maximum = int(translated.max(initial=0))
    dtype = np.uint16 if maximum <= np.iinfo(np.uint16).max else np.uint32
    return translated[inverse].reshape(mask.shape).astype(dtype, copy=False)


def rebuild_masks_and_object_numbers(
    adata,
    mask_paths: Mapping[str, Path],
    output_folder: str | Path,
    *,
    roi_obs: str,
    object_obs: str,
    mode: MaskRebuildMode,
):
    """Write derived masks and return AnnData with exactly aligned ObjectNumbers."""

    preview = preview_mask_rebuild(
        adata,
        mask_paths,
        roi_obs=roi_obs,
        object_obs=object_obs,
        mode=mode,
    )
    if not preview.ready:
        blocked = [check.detail for check in preview.checks if check.level == "blocked"]
        raise ValueError("Mask rebuilding is blocked: " + " ".join(blocked))
    output = Path(output_folder).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(
            f"Refusing to merge into an existing mask output folder: {output}"
        )
    identities = identity_frame(adata, roi_obs=roi_obs, object_obs=object_obs)
    updated = adata.copy()
    updated_ids = pd.Series(index=identities["obs_name"], dtype="int64")
    crosswalk_rows: list[dict[str, Any]] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        from tifffile import imread, imwrite

        for roi, rows in identities.groupby("ROI", observed=True, sort=False):
            old_ids = sorted(rows["ObjectNumber"].astype(int).tolist())
            mapping = (
                {old: old for old in old_ids}
                if mode == "preserve"
                else {old: index for index, old in enumerate(old_ids, start=1)}
            )
            source = Path(mask_paths[str(roi)])
            mask = np.asarray(imread(source))
            rebuilt = _remap_mask(mask, mapping)
            destination = temporary / source.name
            imwrite(destination, rebuilt, photometric="minisblack")
            for row in rows.itertuples(index=False):
                new_id = int(mapping[int(row.ObjectNumber)])
                updated_ids.loc[str(row.obs_name)] = new_id
                crosswalk_rows.append(
                    {
                        "obs_name": str(row.obs_name),
                        "ROI": str(roi),
                        "ObjectNumber_before": int(row.ObjectNumber),
                        "ObjectNumber_after": new_id,
                    }
                )
        aligned = updated_ids.reindex(updated.obs_names.astype(str))
        if aligned.isna().any():
            raise RuntimeError(
                "ObjectNumber remapping did not cover every AnnData cell."
            )
        updated.obs[object_obs] = aligned.to_numpy(dtype=np.int64)
        crosswalk = pd.DataFrame(crosswalk_rows)
        crosswalk.to_csv(temporary / "object_number_crosswalk.csv", index=False)
        (temporary / "maintenance_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "operation": "rebuild_masks",
                    "timestamp": utc_timestamp(),
                    "mode": mode,
                    "roi_obs": roi_obs,
                    "object_obs": object_obs,
                    "cell_count": int(updated.n_obs),
                    "roi_count": int(crosswalk["ROI"].nunique()),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return updated, crosswalk, output


def dataset_readiness(
    adata,
    *,
    roi_obs: str,
    object_obs: str,
    mask_paths: Mapping[str, Path] | None = None,
    image_index: Mapping[str, Mapping[str, Path]] | None = None,
    expect_masks: bool = True,
    expect_images: bool = True,
) -> list[MaintenanceCheck]:
    """Build a cheap dashboard from already indexed assets; never rescan folders."""

    checks = [
        MaintenanceCheck(
            key="anndata",
            label="Live AnnData",
            level="ready",
            detail=f"{adata.n_obs:,} cells and {adata.n_vars:,} variables are in memory.",
        )
    ]
    try:
        identities = identity_frame(adata, roi_obs=roi_obs, object_obs=object_obs)
    except ValueError as error:
        checks.append(
            MaintenanceCheck(
                key="identity",
                label="Cell identity",
                level="blocked",
                detail=str(error),
            )
        )
        return checks
    rois = identities["ROI"].astype(str).drop_duplicates().tolist()
    checks.append(
        MaintenanceCheck(
            key="identity",
            label="Cell identity",
            level="ready",
            detail=f"Unique (ROI, ObjectNumber) identities across {len(rois):,} ROIs.",
        )
    )
    indexed_masks = mask_paths or {}
    missing_masks = [roi for roi in rois if roi not in indexed_masks]
    mask_level: ReadinessLevel = (
        "optional"
        if not indexed_masks and not expect_masks
        else ("warning" if missing_masks else "ready")
    )
    checks.append(
        MaintenanceCheck(
            key="mask_index",
            label="Indexed masks",
            level=mask_level,
            detail=(
                "No mask folder is configured; mask operations remain optional."
                if mask_level == "optional"
                else f"{len(indexed_masks):,} indexed; missing current ROIs: "
                + ", ".join(missing_masks[:8])
                if missing_masks
                else f"All {len(rois):,} current ROIs have indexed masks."
            ),
        )
    )
    indexed_images = image_index or {}
    image_rois = sum(bool(indexed_images.get(roi)) for roi in rois)
    image_level: ReadinessLevel = (
        "optional"
        if not indexed_images and not expect_images
        else ("warning" if image_rois < len(rois) else "ready")
    )
    checks.append(
        MaintenanceCheck(
            key="image_index",
            label="Indexed images",
            level=image_level,
            detail=(
                "No image folder is configured; image operations remain optional."
                if image_level == "optional"
                else (
                    f"Images are indexed for {image_rois:,}/{len(rois):,} current ROIs."
                )
            ),
        )
    )
    indexed_channels = {
        str(channel).split(" [", 1)[0]
        for channels in indexed_images.values()
        for channel in channels
    }
    variables = set(adata.var_names.astype(str))
    missing_channel_images = sorted(variables - indexed_channels)
    orphan_images = sorted(indexed_channels - variables)
    if indexed_channels:
        alignment_detail = (
            f"{len(variables & indexed_channels):,} shared names; "
            f"{len(missing_channel_images):,} AnnData-only variable(s); "
            f"{len(orphan_images):,} image-only channel(s)."
        )
        examples = [
            *(f"AnnData only: {value}" for value in missing_channel_images[:4]),
            *(f"image only: {value}" for value in orphan_images[:4]),
        ]
        if examples:
            alignment_detail += " " + "; ".join(examples) + "."
        checks.append(
            MaintenanceCheck(
                key="channel_alignment",
                label="Variable/image names",
                level=(
                    "warning" if missing_channel_images or orphan_images else "ready"
                ),
                detail=alignment_detail,
            )
        )
    return checks


def append_maintenance_audit(
    path: str | Path, *, action: str, details: dict[str, Any]
) -> Path:
    """Append one compact machine-readable maintenance event."""

    destination = Path(path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "timestamp": utc_timestamp(),
        "action": str(action),
        "details": details,
    }
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return destination


__all__ = [
    "CellFilterRequest",
    "ImageRenameItem",
    "ImageRenamePlan",
    "MaintenanceCheck",
    "MaintenancePreview",
    "append_maintenance_audit",
    "apply_cell_filter",
    "apply_var_rename",
    "atomic_write_anndata",
    "copy_renamed_images",
    "dataset_readiness",
    "identity_frame",
    "normalise_var_rename_mapping",
    "plan_image_renames",
    "preview_cell_filter",
    "preview_mask_rebuild",
    "preview_var_rename",
    "rebuild_masks_and_object_numbers",
    "remove_anndata_vars",
    "resolve_cell_filter",
]
