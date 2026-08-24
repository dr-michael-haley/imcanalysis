"""Pure helpers for the novice-facing NapariSBT Setup workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from SpatialBiologyToolkit.pipeline.manifests import read_model

from .models import ExperimentManifest, WorkflowMode, slugify

ReadinessLevel = Literal["ready", "check", "blocked", "optional"]


@dataclass(frozen=True)
class WorkflowPresentation:
    """Plain-language presentation for one persisted workflow mode."""

    mode: WorkflowMode
    title: str
    summary: str
    requirements: str
    advanced: bool = False


WORKFLOW_PRESENTATIONS = (
    WorkflowPresentation(
        "data_exploration",
        "Explore my images and cells",
        "View staining, cell overlays, saved image recipes, and tissue regions.",
        "Needs processed cell data, masks, and at least one image folder.",
    ),
    WorkflowPresentation(
        "population_qc",
        "Check existing cell populations",
        "Review clusters or named populations in tissue with guided RGB views.",
        "Needs a population column, masks, and staining images.",
    ),
    WorkflowPresentation(
        "classification",
        "Train a cell classifier",
        "Label examples, build or import features, train a model, and export identities.",
        "Needs a defined cell cohort, 2–8 classes, masks, and images.",
    ),
    WorkflowPresentation(
        "cell_labeling",
        "Manually collect labelled cells",
        "Click cells to build reusable identity lists without training a model.",
        "Needs processed cell data, masks, and images for visual context.",
    ),
    WorkflowPresentation(
        "population_curation",
        "Rename, merge, or subcluster populations",
        "Turn numbered Leiden clusters into named, auditable populations.",
        "Needs processed cell data, masks, and images for tissue review.",
    ),
    WorkflowPresentation(
        "dataset_maintenance",
        "Maintain or realign dataset assets",
        "Save, filter, rename, and synchronize AnnData, image, and mask assets.",
        "Needs AnnData; masks and images are required only for matching operations.",
    ),
    WorkflowPresentation(
        "full_workspace",
        "Show every tool",
        "Expose every NapariSBT tab for a combined advanced workflow.",
        "Requires the complete dataset and shows the most controls.",
        advanced=True,
    ),
)


@dataclass(frozen=True)
class SetupCheck:
    """One colour-and-text readiness result shown in Setup."""

    key: str
    label: str
    level: ReadinessLevel
    detail: str


@dataclass(frozen=True)
class WorkspaceSummary:
    """A bounded-discovery result for one experiment manifest."""

    manifest_path: Path
    root: Path
    name: str
    workflow_mode: str | None
    eligible_cells: int | None
    represented_rois: int | None
    modified_at: datetime | None
    issue: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def loadable(self) -> bool:
        return self.issue is None

    @property
    def level(self) -> ReadinessLevel:
        if self.issue:
            return "blocked"
        return "check" if self.warnings else "ready"


@dataclass(frozen=True)
class DatasetAssetSuggestions:
    """Cheap, bounded suggestions for the dataset inputs shown in Setup."""

    anndata_candidates: tuple[Path, ...] = ()
    masks_candidates: tuple[Path, ...] = ()
    image_candidates: tuple[Path, ...] = ()


_ANNDATA_FOLDER_NAMES = {
    "adata",
    "anndata",
    "cell_data",
    "celldata",
    "data",
    "processed_data",
    "processeddata",
}
_MASK_FOLDER_NAMES = {
    "cell_mask",
    "cell_masks",
    "cellmask",
    "cellmasks",
    "mask",
    "masks",
    "segmentation_mask",
    "segmentation_masks",
    "segmentationmask",
    "segmentationmasks",
}
_IMAGE_FOLDER_NAMES = {
    "channel_images",
    "channelimages",
    "denoised_images",
    "denoisedimages",
    "image",
    "images",
    "imc_images",
    "imcimages",
    "processed",
    "staining_images",
    "stainingimages",
    "tiff",
    "tiffs",
}
_DISCOVERY_EXCLUDED_FOLDERS = {
    ".git",
    ".sbt",
    "napari_sbt",
    "outputs",
    "slurm_logs",
}


def _asset_folder_key(path: Path) -> str:
    """Normalize a folder name for conservative convention matching."""

    return "".join(
        character if character.isalnum() else "_" for character in path.name.casefold()
    ).strip("_")


def discover_dataset_assets(project_root: str | Path) -> DatasetAssetSuggestions:
    """Suggest conventional dataset assets without recursively scanning a project.

    AnnData lookup is limited to the project root and conventional data folders
    directly below it. Mask and image suggestions use immediate folder names only;
    image contents are deliberately not inspected here because that potentially
    expensive work belongs to the explicit dataset-integrity action.
    """

    root = Path(project_root).expanduser().resolve(strict=False)
    if not root.is_dir():
        return DatasetAssetSuggestions()

    try:
        children = sorted(root.iterdir(), key=lambda path: path.name.casefold())
    except OSError:
        return DatasetAssetSuggestions()
    child_folders = [path for path in children if path.is_dir()]
    anndata_candidates = [
        path.resolve(strict=False)
        for path in children
        if path.is_file() and path.suffix.casefold() == ".h5ad"
    ]
    masks_candidates: list[Path] = []
    image_candidates: list[Path] = []

    for folder in child_folders:
        key = _asset_folder_key(folder)
        compact_key = key.replace("_", "")
        if key in _ANNDATA_FOLDER_NAMES or compact_key in _ANNDATA_FOLDER_NAMES:
            try:
                anndata_candidates.extend(
                    path.resolve(strict=False)
                    for path in sorted(
                        folder.iterdir(), key=lambda path: path.name.casefold()
                    )
                    if path.is_file() and path.suffix.casefold() == ".h5ad"
                )
            except OSError:
                pass
        if key in _DISCOVERY_EXCLUDED_FOLDERS:
            continue
        if (
            key in _MASK_FOLDER_NAMES
            or compact_key in _MASK_FOLDER_NAMES
            or "mask" in compact_key
        ):
            masks_candidates.append(folder.resolve(strict=False))
            continue
        if (
            key in _IMAGE_FOLDER_NAMES
            or compact_key in _IMAGE_FOLDER_NAMES
            or "image" in compact_key
            or compact_key.startswith("tiff")
        ):
            image_candidates.append(folder.resolve(strict=False))

    return DatasetAssetSuggestions(
        anndata_candidates=tuple(dict.fromkeys(anndata_candidates)),
        masks_candidates=tuple(dict.fromkeys(masks_candidates)),
        image_candidates=tuple(dict.fromkeys(image_candidates)),
    )


def workflow_presentation(mode: str | None) -> WorkflowPresentation | None:
    """Return the plain-language definition for a workflow value."""

    return next((item for item in WORKFLOW_PRESENTATIONS if item.mode == mode), None)


def suggest_identity_columns(columns: list[str]) -> tuple[str | None, str | None]:
    """Suggest conventional ROI and mask-label observations without guessing ambiguities."""

    by_key = {
        "".join(
            character for character in str(column) if character.isalnum()
        ).casefold(): str(column)
        for column in columns
    }
    roi = next(
        (
            by_key[key]
            for key in ("roi", "roiid", "image", "imageid", "region", "regionid")
            if key in by_key
        ),
        None,
    )
    object_id = next(
        (
            by_key[key]
            for key in (
                "objectnumber",
                "objectid",
                "masklabel",
                "label",
                "cellid",
            )
            if key in by_key
        ),
        None,
    )
    return roi, object_id


def workspace_folder(project_root: str | Path, configured: str = "napari_sbt") -> Path:
    """Resolve the configured workspace container without creating it."""

    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = Path(project_root).expanduser() / candidate
    return candidate.resolve(strict=False)


def workspace_destination(container: str | Path, name: str) -> Path:
    """Derive a safe workspace path from a human-readable name."""

    identifier = slugify(name)
    return Path(container).expanduser().resolve(strict=False) / identifier


def _manifest_candidates(container: Path) -> list[Path]:
    """Return exact and one-level-deep manifests; never recurse through a project."""

    candidates: list[Path] = []
    direct = container / "experiment.yaml"
    if direct.is_file():
        candidates.append(direct)
    if container.is_dir():
        for child in sorted(container.iterdir(), key=lambda path: path.name.casefold()):
            manifest = child / "experiment.yaml"
            if child.is_dir() and manifest.is_file():
                candidates.append(manifest)
    return candidates


def _manifest_source_warnings(
    manifest: ExperimentManifest, workspace_root: Path
) -> tuple[str, ...]:
    """Return cheap missing-source warnings without opening or scanning assets."""

    base = (
        Path(manifest.project_root).expanduser()
        if manifest.project_root
        else workspace_root
    )

    def resolved(value: str | None) -> Path | None:
        if not value:
            return None
        path = Path(value).expanduser()
        return path if path.is_absolute() else base / path

    warnings: list[str] = []
    adata = resolved(manifest.anndata_path)
    if adata is None or not adata.is_file():
        warnings.append("processed cell data is missing")
    masks = resolved(manifest.masks_folder)
    if masks is None or not masks.is_dir():
        warnings.append("cell masks folder is missing")
    missing_images = sum(
        not path.is_dir()
        for path in (
            resolved(value)
            for value in [*manifest.images_folders, *manifest.extra_images_folders]
        )
        if path is not None
    )
    if not manifest.images_folders:
        warnings.append("no staining image folder is configured")
    elif missing_images:
        warnings.append(f"{missing_images} image folder(s) are missing")
    return tuple(warnings)


def discover_workspaces(
    container: str | Path,
    *,
    explicit: str | Path | None = None,
) -> list[WorkspaceSummary]:
    """Discover loadable and broken workspaces with a bounded directory lookup."""

    container = Path(container).expanduser().resolve(strict=False)
    candidates = _manifest_candidates(container)
    if explicit:
        candidate = Path(explicit).expanduser().resolve(strict=False)
        manifest = (
            candidate
            if candidate.name == "experiment.yaml"
            else candidate / "experiment.yaml"
        )
        if manifest.is_file() and manifest not in candidates:
            candidates.insert(0, manifest)

    summaries: list[WorkspaceSummary] = []
    for manifest_path in candidates:
        modified = None
        try:
            modified = datetime.fromtimestamp(manifest_path.stat().st_mtime)
        except OSError:
            pass
        try:
            manifest = read_model(manifest_path, ExperimentManifest)
            summaries.append(
                WorkspaceSummary(
                    manifest_path=manifest_path,
                    root=manifest_path.parent,
                    name=manifest.name,
                    workflow_mode=manifest.workflow_mode,
                    eligible_cells=manifest.cell_scope.eligible_cell_count,
                    represented_rois=manifest.cell_scope.represented_roi_count,
                    modified_at=modified,
                    warnings=_manifest_source_warnings(manifest, manifest_path.parent),
                )
            )
        except Exception as exc:  # noqa: BLE001 - broken entries remain visible
            summaries.append(
                WorkspaceSummary(
                    manifest_path=manifest_path,
                    root=manifest_path.parent,
                    name=manifest_path.parent.name,
                    workflow_mode=None,
                    eligible_cells=None,
                    represented_rois=None,
                    modified_at=modified,
                    issue=f"{type(exc).__name__}: {exc}",
                )
            )
    return sorted(
        summaries,
        key=lambda item: (
            not item.loadable,
            -(item.modified_at.timestamp() if item.modified_at else 0),
            item.name.casefold(),
        ),
    )


def path_check(
    key: str,
    label: str,
    value: str | Path | None,
    *,
    kind: Literal["file", "directory"],
    required: bool = True,
) -> SetupCheck:
    """Perform a cheap, non-recursive path readiness check."""

    text = str(value or "").strip()
    if not text:
        return SetupCheck(
            key,
            label,
            "blocked" if required else "optional",
            "Choose this item to continue."
            if required
            else "Optional; nothing selected.",
        )
    path = Path(text).expanduser()
    exists = path.is_file() if kind == "file" else path.is_dir()
    if not exists:
        expected = "file" if kind == "file" else "folder"
        return SetupCheck(
            key, label, "blocked", f"The selected {expected} was not found."
        )
    return SetupCheck(key, label, "check", "Found; run the dataset integrity check.")


def setup_checks(
    *,
    workspace_name: str,
    workspace_path: str | Path,
    workflow_mode: str | None,
    anndata_path: str | Path | None,
    has_in_memory_anndata: bool,
    masks_folder: str | Path | None,
    image_folders: list[str],
    extra_image_folders: list[str],
    roi_obs: str,
    object_id_obs: str,
    normalization_path: str | Path | None,
    integrity_current: bool,
) -> tuple[SetupCheck, ...]:
    """Return the common Setup readiness checks in user-facing order."""

    checks: list[SetupCheck] = []
    name = workspace_name.strip()
    destination = (
        Path(workspace_path).expanduser() if str(workspace_path).strip() else None
    )
    if not name:
        checks.append(
            SetupCheck("workspace", "Workspace", "blocked", "Enter a workspace name.")
        )
    elif destination is None:
        checks.append(
            SetupCheck(
                "workspace",
                "Workspace",
                "blocked",
                "Choose where the workspace will be stored.",
            )
        )
    elif (destination / "experiment.yaml").exists():
        checks.append(
            SetupCheck(
                "workspace",
                "Workspace",
                "blocked",
                "A workspace already exists here; open it instead or choose another name.",
            )
        )
    else:
        checks.append(
            SetupCheck(
                "workspace", "Workspace", "ready", f"New workspace: {destination.name}"
            )
        )

    presentation = workflow_presentation(workflow_mode)
    maintenance_only = workflow_mode == "dataset_maintenance"
    checks.append(
        SetupCheck(
            "workflow",
            "Workflow",
            "ready" if presentation else "blocked",
            presentation.title if presentation else "Choose what you want to do.",
        )
    )

    if has_in_memory_anndata:
        checks.append(
            SetupCheck(
                "anndata",
                "Processed cell data",
                "ready",
                "Using the AnnData object already loaded in the notebook.",
            )
        )
    else:
        checks.append(
            path_check("anndata", "Processed cell data", anndata_path, kind="file")
        )
    checks.append(
        path_check(
            "masks",
            "Cell masks",
            masks_folder,
            kind="directory",
            required=not maintenance_only,
        )
    )

    usable_images = [path for path in image_folders if Path(path).expanduser().is_dir()]
    if not image_folders:
        checks.append(
            SetupCheck(
                "images",
                "Staining images",
                "optional" if maintenance_only else "blocked",
                (
                    "Optional for AnnData-only maintenance; add folders before "
                    "renaming or validating images."
                    if maintenance_only
                    else "Add at least one staining-image folder."
                ),
            )
        )
    elif len(usable_images) != len(image_folders):
        checks.append(
            SetupCheck(
                "images",
                "Staining images",
                "blocked",
                "One or more selected image folders were not found.",
            )
        )
    else:
        checks.append(
            SetupCheck(
                "images",
                "Staining images",
                "check",
                f"Found {len(usable_images)} folder(s); run the dataset integrity check.",
            )
        )

    usable_extra = [
        path for path in extra_image_folders if Path(path).expanduser().is_dir()
    ]
    if not extra_image_folders:
        checks.append(
            SetupCheck(
                "extra_images",
                "Additional images",
                "optional",
                "Optional; no additional image folders selected.",
            )
        )
    elif len(usable_extra) != len(extra_image_folders):
        checks.append(
            SetupCheck(
                "extra_images",
                "Additional images",
                "blocked",
                "One or more optional image folders were not found; remove or replace them.",
            )
        )
    else:
        checks.append(
            SetupCheck(
                "extra_images",
                "Additional images",
                "ready",
                f"Found {len(usable_extra)} optional image folder(s).",
            )
        )

    identity_detail = (
        f"Match image names with '{roi_obs}' and mask labels with '{object_id_obs}'."
        if roi_obs.strip() and object_id_obs.strip()
        else "Choose the AnnData columns that identify images and mask labels."
    )
    checks.append(
        SetupCheck(
            "identity",
            "Cell-to-image matching",
            "ready" if integrity_current else "check",
            identity_detail
            if integrity_current
            else identity_detail + " The integrity check will confirm them.",
        )
    )

    checks.append(
        path_check(
            "normalization",
            "Image brightness settings",
            normalization_path,
            kind="file",
            required=False,
        )
    )
    if integrity_current:
        for index, check in enumerate(checks):
            if check.key in {"anndata", "masks", "images"} and check.level == "check":
                checks[index] = SetupCheck(
                    check.key,
                    check.label,
                    "ready",
                    "Validated for this workspace setup.",
                )
    else:
        checks.append(
            SetupCheck(
                "integrity",
                "Dataset integrity",
                "check",
                "Run the full check after selecting all required inputs.",
            )
        )
    return tuple(checks)


def setup_is_ready(checks: tuple[SetupCheck, ...]) -> bool:
    """Return whether no blocking or unvalidated required item remains."""

    blocking_keys = {
        "workspace",
        "workflow",
        "anndata",
        "masks",
        "images",
        "identity",
        "integrity",
    }
    return all(
        check.level != "blocked"
        and (check.level in {"ready", "optional"} or check.key not in blocking_keys)
        for check in checks
    )


__all__ = [
    "WORKFLOW_PRESENTATIONS",
    "DatasetAssetSuggestions",
    "SetupCheck",
    "WorkflowPresentation",
    "WorkspaceSummary",
    "discover_dataset_assets",
    "discover_workspaces",
    "path_check",
    "setup_checks",
    "setup_is_ready",
    "suggest_identity_columns",
    "workflow_presentation",
    "workspace_destination",
    "workspace_folder",
]
