"""Auditable population naming, merging, and subcluster composition.

The population-curation workspace deliberately keeps the original AnnData
observation immutable.  A draft combines a one-to-one base remap with optional
cell-level split components.  Giving two components the same proposed label is
an explicit merge; split memberships always override the base remap.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from SpatialBiologyToolkit.pipeline.manifests import (
    read_model,
    utc_now,
    write_json,
    write_text,
    write_yaml,
)

from .colour_helper import categorical_colour_collisions, normalise_hex_colour
from .models import slugify
from .storage import dataframe_sha256, read_dataframe, write_dataframe

CURATION_SCHEMA_VERSION = 1
BASE_MAPPING_COLUMNS = [
    "source_value",
    "cell_count",
    "proposed_label",
    "color",
    "notes",
]
COMPONENT_COLUMNS = [
    "component_id",
    "parent_source_value",
    "method",
    "run_id",
    "component_value",
    "cell_count",
    "proposed_label",
    "color",
    "notes",
]
MEMBERSHIP_COLUMNS = ["obs_name", "component_id"]
OBS_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")

_FALLBACK_COLOURS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
    "#31a354",
    "#756bb1",
    "#636363",
    "#e6550d",
)


def normalise_source_label(value: Any) -> str | None:
    """Return the canonical form used to match population source labels.

    AnnData category labels occasionally contain accidental leading or trailing
    whitespace. Population mappings are edited as clean display strings, so all
    matching paths use the same stripped representation. The source fingerprint
    deliberately remains byte-for-byte faithful to the AnnData values.
    """

    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip()
    return cleaned or None


class PopulationWorkspace(BaseModel):
    """One immutable source observation and its sibling derived drafts."""

    schema_version: Literal[CURATION_SCHEMA_VERSION] = CURATION_SCHEMA_VERSION
    workspace_id: str = Field(default_factory=lambda: str(uuid4()))
    source_obs: str
    source_fingerprint: str
    source_cell_count: int = Field(ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    draft_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_workspace(self) -> "PopulationWorkspace":
        self.source_obs = str(self.source_obs).strip()
        if not self.source_obs:
            raise ValueError("A source AnnData observation is required.")
        return self


class PopulationDraft(BaseModel):
    """Persisted recipe for one synthesized AnnData observation."""

    schema_version: Literal[CURATION_SCHEMA_VERSION] = CURATION_SCHEMA_VERSION
    draft_id: str = Field(default_factory=lambda: str(uuid4()))
    revision: int = Field(default=1, ge=1)
    name: str
    source_obs: str
    derived_obs: str
    source_fingerprint: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    status: Literal["draft", "applied", "exported"] = "draft"
    base_mapping_path: str = "base_mapping.csv"
    components_path: str = "components.csv"
    membership_path: str = "component_membership.csv"
    latest_run_id: str | None = None

    @model_validator(mode="after")
    def validate_draft(self) -> "PopulationDraft":
        self.name = str(self.name).strip()
        self.source_obs = str(self.source_obs).strip()
        self.derived_obs = str(self.derived_obs).strip()
        if not self.name:
            raise ValueError("A population-draft name is required.")
        if not OBS_KEY_PATTERN.fullmatch(self.derived_obs):
            raise ValueError(
                "The derived observation must start with a letter or underscore and "
                "contain only letters, numbers, underscores, dots, or hyphens."
            )
        if self.derived_obs == self.source_obs:
            raise ValueError("The derived observation must not overwrite its source.")
        return self


class GraphSubclusterRequest(BaseModel):
    """Portable request consumed by the monitored Scanpy worker."""

    schema_version: Literal[CURATION_SCHEMA_VERSION] = CURATION_SCHEMA_VERSION
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    anndata_path: str
    source_obs: str
    source_fingerprint: str
    selected_values: list[str]
    neighbor_source: Literal["rebuild_from_rep", "existing_graph"] = (
        "rebuild_from_rep"
    )
    representation_key: str | None = "X_biobatchnet"
    n_neighbors: int = Field(default=15, ge=2, le=1000)
    adjacency_key: str | None = None
    resolution: float = Field(default=0.5, gt=0)
    random_state: int = 0
    mode: Literal["within_each", "together"] = "within_each"
    output_folder: str

    @model_validator(mode="after")
    def validate_request(self) -> "GraphSubclusterRequest":
        self.selected_values = list(
            dict.fromkeys(
                cleaned
                for value in self.selected_values
                if (cleaned := normalise_source_label(value)) is not None
            )
        )
        if not self.selected_values:
            raise ValueError("Select at least one population to subcluster.")
        self.representation_key = (
            str(self.representation_key).strip()
            if self.representation_key is not None
            else None
        )
        self.adjacency_key = (
            str(self.adjacency_key).strip()
            if self.adjacency_key is not None
            else None
        )
        if self.neighbor_source == "rebuild_from_rep":
            if not self.representation_key:
                raise ValueError(
                    "Choose an existing corrected AnnData obsm representation."
                )
        elif not self.adjacency_key:
            raise ValueError("Choose an existing AnnData connectivity graph.")
        return self


@dataclass(frozen=True)
class PopulationWorkspacePaths:
    root: Path
    manifest: Path
    drafts: Path
    runs: Path
    inputs: Path
    audit: Path


@dataclass(frozen=True)
class PopulationDraftPaths:
    root: Path
    manifest: Path
    base_mapping: Path
    components: Path
    membership: Path
    runs: Path


def source_obs_fingerprint(adata: Any, source_obs: str) -> str:
    """Hash cell identity plus source labels without depending on row order."""

    source_obs = str(source_obs).strip()
    if source_obs not in adata.obs:
        raise KeyError(f"AnnData observation {source_obs!r} does not exist.")
    obs_names = pd.Index(adata.obs_names.astype(str))
    if not obs_names.is_unique:
        duplicates = obs_names[obs_names.duplicated()].unique().tolist()[:10]
        raise ValueError(f"AnnData obs_names must be unique; duplicates: {duplicates}")
    frame = pd.DataFrame(
        {
            "obs_name": obs_names,
            "source_value": adata.obs[source_obs]
            .astype("string")
            .fillna("<NA>")
            .to_numpy(),
        }
    )
    return dataframe_sha256(frame, ["obs_name", "source_value"])


def population_workspace_paths(
    root: str | Path,
    source_obs: str,
) -> PopulationWorkspacePaths:
    base = Path(root).expanduser().resolve(strict=False)
    workspace_root = base / slugify(source_obs)
    return PopulationWorkspacePaths(
        root=workspace_root,
        manifest=workspace_root / "workspace.yaml",
        drafts=workspace_root / "drafts",
        runs=workspace_root / "runs",
        inputs=workspace_root / "inputs",
        audit=workspace_root / "provenance.jsonl",
    )


def population_draft_paths(
    workspace_paths: PopulationWorkspacePaths,
    draft: PopulationDraft | str,
) -> PopulationDraftPaths:
    draft_id = draft.draft_id if isinstance(draft, PopulationDraft) else str(draft)
    root = workspace_paths.drafts / draft_id
    if isinstance(draft, PopulationDraft):
        base_mapping = root / draft.base_mapping_path
        components = root / draft.components_path
        membership = root / draft.membership_path
    else:
        base_mapping = root / "base_mapping.csv"
        components = root / "components.csv"
        membership = root / "component_membership.csv"
    return PopulationDraftPaths(
        root=root,
        manifest=root / "draft.yaml",
        base_mapping=base_mapping,
        components=components,
        membership=membership,
        runs=root / "runs",
    )


def ensure_population_workspace(
    adata: Any,
    root: str | Path,
    source_obs: str,
) -> tuple[PopulationWorkspace, PopulationWorkspacePaths]:
    """Create or validate a source-locked curation workspace."""

    fingerprint = source_obs_fingerprint(adata, source_obs)
    paths = population_workspace_paths(root, source_obs)
    if paths.manifest.is_file():
        workspace = read_model(paths.manifest, PopulationWorkspace)
        if workspace.source_obs != source_obs:
            raise ValueError(
                f"Workspace source is {workspace.source_obs!r}, not {source_obs!r}."
            )
        if workspace.source_fingerprint != fingerprint:
            raise ValueError(
                "The source observation or cell identities changed after this "
                "population workspace was created. Start a new revision/workspace "
                "instead of silently changing the remapping universe."
            )
        return workspace, paths

    non_null = int(adata.obs[source_obs].notna().sum())
    if non_null == 0:
        raise ValueError(f"AnnData observation {source_obs!r} has no usable labels.")
    workspace = PopulationWorkspace(
        source_obs=source_obs,
        source_fingerprint=fingerprint,
        source_cell_count=int(adata.n_obs),
    )
    for folder in (paths.root, paths.drafts, paths.runs, paths.inputs):
        folder.mkdir(parents=True, exist_ok=True)
    write_yaml(paths.manifest, workspace)
    append_population_audit(
        paths,
        action="create_workspace",
        details={
            "workspace_id": workspace.workspace_id,
            "source_obs": source_obs,
            "source_fingerprint": fingerprint,
            "cell_count": int(adata.n_obs),
        },
    )
    return workspace, paths


def ordered_source_labels(series: pd.Series) -> list[str]:
    """Return source labels in display order using canonical whitespace.

    Distinct source categories must remain distinguishable after conversion to
    strings and trimming. Rejecting collisions is safer than silently merging
    two populations that only differ by whitespace.
    """

    if isinstance(series.dtype, pd.CategoricalDtype):
        raw = list(series.cat.categories)
    else:
        raw = list(pd.unique(series.dropna()))
    ordered: list[str] = []
    raw_by_cleaned: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for value in raw:
        cleaned = normalise_source_label(value)
        if cleaned is None:
            continue
        raw_text = str(value)
        previous = raw_by_cleaned.get(cleaned)
        if previous is not None and previous != raw_text:
            collisions.setdefault(cleaned, [previous]).append(raw_text)
            continue
        if previous is None:
            raw_by_cleaned[cleaned] = raw_text
            ordered.append(cleaned)
    if collisions:
        examples = {
            cleaned: list(dict.fromkeys(raw_values))
            for cleaned, raw_values in list(collisions.items())[:10]
        }
        raise ValueError(
            "Source observation contains labels that become identical after "
            f"trimming whitespace/string conversion: {examples}. Clean or rename "
            "those source labels before population curation."
        )
    return ordered


def _source_colour_map(adata: Any, source_obs: str) -> dict[str, str]:
    values = ordered_source_labels(adata.obs[source_obs])
    stored = adata.uns.get(f"{source_obs}_colors")
    if isinstance(stored, dict):
        result = {
            cleaned: str(value)
            for key, value in stored.items()
            if (cleaned := normalise_source_label(key)) is not None
        }
    elif isinstance(stored, (list, tuple, np.ndarray)):
        result = {
            value: str(colour)
            for value, colour in zip(values, list(stored))
            if str(colour).strip()
        }
    else:
        result = {}
    for index, value in enumerate(values):
        result.setdefault(value, _FALLBACK_COLOURS[index % len(_FALLBACK_COLOURS)])
    return result


def build_base_mapping(adata: Any, source_obs: str) -> pd.DataFrame:
    """Build an identity-preserving initial mapping for every source label."""

    if source_obs not in adata.obs:
        raise KeyError(f"AnnData observation {source_obs!r} does not exist.")
    source = (
        adata.obs[source_obs]
        .astype(object)
        .map(normalise_source_label)
        .astype("string")
    )
    counts = source.value_counts(dropna=True).to_dict()
    colours = _source_colour_map(adata, source_obs)
    rows = []
    for value in ordered_source_labels(adata.obs[source_obs]):
        rows.append(
            {
                "source_value": value,
                "cell_count": int(counts.get(value, 0)),
                "proposed_label": value,
                "color": colours[value],
                "notes": "",
            }
        )
    return pd.DataFrame(rows, columns=BASE_MAPPING_COLUMNS)


def empty_components() -> pd.DataFrame:
    return pd.DataFrame(columns=COMPONENT_COLUMNS)


def empty_membership() -> pd.DataFrame:
    return pd.DataFrame(columns=MEMBERSHIP_COLUMNS)


def _clean_text(value: Any) -> str:
    return normalise_source_label(value) or ""


def validate_base_mapping(
    frame: pd.DataFrame,
    *,
    expected_source_values: Sequence[str] | None = None,
) -> pd.DataFrame:
    missing = sorted(set(BASE_MAPPING_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Population mapping is missing columns: {missing}")
    result = frame.loc[:, BASE_MAPPING_COLUMNS].copy()
    for column in ("source_value", "proposed_label", "color", "notes"):
        result[column] = result[column].map(_clean_text)
    result["cell_count"] = pd.to_numeric(
        result["cell_count"], errors="raise"
    ).astype(int)
    if result["source_value"].eq("").any():
        raise ValueError("Source population values must not be blank.")
    duplicates = result.loc[
        result["source_value"].duplicated(keep=False), "source_value"
    ].unique()
    if len(duplicates):
        raise ValueError(
            f"Population mapping contains duplicate source values: {duplicates.tolist()}"
        )
    if result["proposed_label"].eq("").any():
        missing_labels = result.loc[
            result["proposed_label"].eq(""), "source_value"
        ].tolist()
        raise ValueError(
            "Every source population needs a proposed name; missing for "
            f"{missing_labels[:10]}."
        )
    if expected_source_values is not None:
        expected = {
            cleaned
            for value in expected_source_values
            if (cleaned := normalise_source_label(value)) is not None
        }
        actual = set(result["source_value"])
        missing_values = sorted(expected - actual)
        unknown_values = sorted(actual - expected)
        if missing_values or unknown_values:
            raise ValueError(
                "Population mapping does not match the frozen source labels: "
                f"missing={missing_values[:10]}, unknown={unknown_values[:10]}."
            )
    return result.reset_index(drop=True)


def validate_component_tables(
    components: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    adata: Any | None = None,
    source_obs: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if components.empty and membership.empty:
        return empty_components(), empty_membership()
    missing_components = sorted(set(COMPONENT_COLUMNS) - set(components.columns))
    missing_membership = sorted(set(MEMBERSHIP_COLUMNS) - set(membership.columns))
    if missing_components or missing_membership:
        raise ValueError(
            "Subcluster tables have an invalid schema: "
            f"component columns missing={missing_components}; "
            f"membership columns missing={missing_membership}."
        )
    clean_components = components.loc[:, COMPONENT_COLUMNS].copy()
    for column in (
        "component_id",
        "parent_source_value",
        "method",
        "run_id",
        "component_value",
        "proposed_label",
        "color",
        "notes",
    ):
        clean_components[column] = clean_components[column].map(_clean_text)
    clean_components["cell_count"] = pd.to_numeric(
        clean_components["cell_count"], errors="raise"
    ).astype(int)
    if clean_components["component_id"].eq("").any():
        raise ValueError("Subcluster component IDs must not be blank.")
    if clean_components["component_id"].duplicated().any():
        raise ValueError("Subcluster component IDs must be unique.")
    if clean_components["proposed_label"].eq("").any():
        raise ValueError("Every subcluster component needs a proposed name.")

    clean_membership = membership.loc[:, MEMBERSHIP_COLUMNS].copy()
    clean_membership["obs_name"] = clean_membership["obs_name"].map(_clean_text)
    clean_membership["component_id"] = clean_membership["component_id"].map(
        _clean_text
    )
    if clean_membership["obs_name"].duplicated().any():
        duplicates = clean_membership.loc[
            clean_membership["obs_name"].duplicated(keep=False), "obs_name"
        ].unique()
        raise ValueError(
            "A cell can belong to only one active split component in a draft; "
            f"duplicates: {duplicates.tolist()[:10]}."
        )
    unknown_components = sorted(
        set(clean_membership["component_id"])
        - set(clean_components["component_id"])
    )
    if unknown_components:
        raise ValueError(
            f"Membership references unknown components: {unknown_components[:10]}"
        )
    counts = clean_membership["component_id"].value_counts().to_dict()
    clean_components["cell_count"] = clean_components["component_id"].map(
        counts
    ).fillna(0).astype(int)
    clean_components = clean_components.loc[clean_components["cell_count"] > 0].copy()
    clean_membership = clean_membership.loc[
        clean_membership["component_id"].isin(clean_components["component_id"])
    ].copy()

    if adata is not None:
        if source_obs is None or source_obs not in adata.obs:
            raise KeyError(f"Source observation {source_obs!r} is unavailable.")
        known = set(adata.obs_names.astype(str))
        missing_cells = sorted(set(clean_membership["obs_name"]) - known)
        if missing_cells:
            raise ValueError(
                f"Subcluster membership contains unknown cells: {missing_cells[:10]}"
            )
        source_lookup = (
            adata.obs[source_obs]
            .astype(object)
            .map(normalise_source_label)
            .astype("string")
        )
        source_lookup.index = source_lookup.index.astype(str)
        component_parent = clean_components.set_index("component_id")[
            "parent_source_value"
        ].to_dict()
        expected_parent = clean_membership["component_id"].map(component_parent)
        actual_parent = clean_membership["obs_name"].map(source_lookup)
        mismatch = actual_parent.astype("string") != expected_parent.astype("string")
        if bool(mismatch.any()):
            examples = clean_membership.loc[mismatch, "obs_name"].tolist()[:10]
            raise ValueError(
                "Subcluster membership does not match its parent source population; "
                f"example cells: {examples}."
            )
    return (
        clean_components.reset_index(drop=True),
        clean_membership.reset_index(drop=True),
    )


def append_population_audit(
    paths: PopulationWorkspacePaths,
    *,
    action: str,
    details: dict[str, Any] | None = None,
    draft_id: str | None = None,
    actor: str | None = None,
) -> Path:
    """Append a durable event through an atomic whole-file rewrite."""

    event = {
        "timestamp": utc_now().isoformat(),
        "action": str(action),
        "actor": actor or getpass.getuser(),
        "draft_id": draft_id,
        "details": details or {},
    }
    existing = paths.audit.read_text(encoding="utf-8") if paths.audit.exists() else ""
    line = json.dumps(event, ensure_ascii=False, sort_keys=True, default=str)
    return write_text(paths.audit, existing + line + "\n")


def read_population_audit(
    paths: PopulationWorkspacePaths,
) -> list[dict[str, Any]]:
    if not paths.audit.is_file():
        return []
    events = []
    for line in paths.audit.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _mapping_changes(before: pd.DataFrame, after: pd.DataFrame) -> list[dict[str, Any]]:
    if before.empty:
        return []
    old = before.set_index("source_value")
    new = after.set_index("source_value")
    changes: list[dict[str, Any]] = []
    for source_value in new.index:
        for column in ("proposed_label", "color", "notes"):
            old_value = _clean_text(old.at[source_value, column])
            new_value = _clean_text(new.at[source_value, column])
            if old_value != new_value:
                changes.append(
                    {
                        "source_value": str(source_value),
                        "field": column,
                        "before": old_value,
                        "after": new_value,
                    }
                )
    return changes


def _component_changes(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> list[dict[str, Any]]:
    if before.empty and after.empty:
        return []
    old = before.set_index("component_id") if not before.empty else pd.DataFrame()
    new = after.set_index("component_id") if not after.empty else pd.DataFrame()
    old_ids = set(old.index) if not old.empty else set()
    new_ids = set(new.index) if not new.empty else set()
    changes: list[dict[str, Any]] = [
        {"component_id": component_id, "change": "added"}
        for component_id in sorted(new_ids - old_ids)
    ]
    changes.extend(
        {"component_id": component_id, "change": "removed"}
        for component_id in sorted(old_ids - new_ids)
    )
    for component_id in sorted(old_ids & new_ids):
        for column in ("proposed_label", "color", "notes"):
            old_value = _clean_text(old.at[component_id, column])
            new_value = _clean_text(new.at[component_id, column])
            if old_value != new_value:
                changes.append(
                    {
                        "component_id": component_id,
                        "field": column,
                        "before": old_value,
                        "after": new_value,
                    }
                )
    return changes


def create_population_draft(
    adata: Any,
    root: str | Path,
    *,
    source_obs: str,
    name: str,
    derived_obs: str,
) -> tuple[
    PopulationWorkspace,
    PopulationWorkspacePaths,
    PopulationDraft,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    workspace, workspace_paths = ensure_population_workspace(
        adata, root, source_obs
    )
    existing = [
        draft
        for draft in list_population_drafts(workspace_paths)
        if draft.derived_obs == derived_obs
    ]
    if existing:
        raise ValueError(
            f"A population draft already targets adata.obs[{derived_obs!r}]."
        )
    if derived_obs in adata.obs:
        raise ValueError(
            f"AnnData already contains obs[{derived_obs!r}]. Choose a new name."
        )
    draft = PopulationDraft(
        name=name,
        source_obs=source_obs,
        derived_obs=derived_obs,
        source_fingerprint=workspace.source_fingerprint,
    )
    base = build_base_mapping(adata, source_obs)
    components = empty_components()
    membership = empty_membership()
    draft_paths = population_draft_paths(workspace_paths, draft)
    draft_paths.root.mkdir(parents=True, exist_ok=False)
    draft_paths.runs.mkdir(parents=True, exist_ok=True)
    write_dataframe(draft_paths.base_mapping, base)
    write_dataframe(draft_paths.components, components)
    write_dataframe(draft_paths.membership, membership)
    write_yaml(draft_paths.manifest, draft)
    workspace.draft_ids.append(draft.draft_id)
    workspace.updated_at = utc_now()
    write_yaml(workspace_paths.manifest, workspace)
    append_population_audit(
        workspace_paths,
        action="create_draft",
        draft_id=draft.draft_id,
        details={
            "name": draft.name,
            "source_obs": source_obs,
            "derived_obs": derived_obs,
            "source_populations": len(base),
        },
    )
    return workspace, workspace_paths, draft, base, components, membership


def list_population_drafts(
    workspace_paths: PopulationWorkspacePaths,
) -> list[PopulationDraft]:
    if not workspace_paths.drafts.is_dir():
        return []
    drafts = []
    for manifest in sorted(workspace_paths.drafts.glob("*/draft.yaml")):
        drafts.append(read_model(manifest, PopulationDraft))
    return sorted(drafts, key=lambda item: (item.created_at, item.name))


def load_population_draft(
    workspace_paths: PopulationWorkspacePaths,
    draft_id: str,
) -> tuple[PopulationDraft, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    provisional = population_draft_paths(workspace_paths, draft_id)
    draft = read_model(provisional.manifest, PopulationDraft)
    paths = population_draft_paths(workspace_paths, draft)
    base = validate_base_mapping(read_dataframe(paths.base_mapping))
    components = (
        read_dataframe(paths.components)
        if paths.components.is_file()
        else empty_components()
    )
    membership = (
        read_dataframe(paths.membership)
        if paths.membership.is_file()
        else empty_membership()
    )
    components, membership = validate_component_tables(components, membership)
    return draft, base, components, membership


def merge_groups(
    base_mapping: pd.DataFrame,
    components: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    """Return final names deliberately shared by two or more contributors."""

    contributors: dict[str, list[str]] = {}
    for row in base_mapping.itertuples(index=False):
        contributors.setdefault(str(row.proposed_label), []).append(
            f"source:{row.source_value}"
        )
    if components is not None:
        for row in components.itertuples(index=False):
            contributors.setdefault(str(row.proposed_label), []).append(
                f"{row.method}:{row.parent_source_value}/{row.component_value}"
            )
    return {
        label: values
        for label, values in contributors.items()
        if len(set(values)) > 1
    }


def harmonize_merge_colours(
    base_mapping: pd.DataFrame,
    components: pd.DataFrame | None = None,
    *,
    merge_labels: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Give every contributor to one merge one distinct, shared colour.

    Existing contributor colours are preferred in table order. If that colour is
    already owned by another merge, another existing colour or the fallback
    palette is used. Non-merge rows are never changed.
    """

    base = base_mapping.copy()
    component_frame = (
        components.copy() if components is not None else empty_components()
    )
    labels = list(
        dict.fromkeys(
            str(label)
            for label in (
                merge_labels
                if merge_labels is not None
                else merge_groups(base, component_frame)
            )
        )
    )
    assignments: dict[str, str] = {}
    merge_label_set = set(labels)
    non_merge_colours = [
        *base.loc[
            ~base["proposed_label"].astype(str).isin(merge_label_set), "color"
        ].map(_clean_text),
        *component_frame.loc[
            ~component_frame["proposed_label"].astype(str).isin(merge_label_set),
            "color",
        ].map(_clean_text),
    ]
    # A newly harmonized merge must not inherit a colour already used by an
    # unrelated population. Existing unrelated collisions are reported by the UI.
    used_colours: set[str] = {
        colour.casefold() for colour in non_merge_colours if colour
    }
    palette_cursor = 0

    for label in labels:
        base_match = base["proposed_label"].astype(str).eq(label)
        component_match = (
            component_frame["proposed_label"].astype(str).eq(label)
            if not component_frame.empty
            else pd.Series(False, index=component_frame.index, dtype=bool)
        )
        existing = [
            colour
            for colour in dict.fromkeys(
                [
                    *base.loc[base_match, "color"].map(_clean_text).tolist(),
                    *component_frame.loc[component_match, "color"]
                    .map(_clean_text)
                    .tolist(),
                ]
            )
            if colour
        ]
        chosen = next(
            (
                colour
                for colour in existing
                if colour.casefold() not in used_colours
            ),
            None,
        )
        while chosen is None:
            candidate = _FALLBACK_COLOURS[palette_cursor % len(_FALLBACK_COLOURS)]
            palette_cursor += 1
            if candidate.casefold() not in used_colours:
                chosen = candidate
            elif palette_cursor > len(_FALLBACK_COLOURS) * 2:
                # More simultaneous merges than the standard categorical palette
                # is unusual; deterministic RGB candidates still prevent reuse.
                digest = hashlib.sha256(
                    f"napari-sbt-merge-{label}-{palette_cursor}".encode()
                ).hexdigest()
                candidate = f"#{digest[:6]}"
                if candidate.casefold() not in used_colours:
                    chosen = candidate
        assignments[label] = chosen
        used_colours.add(chosen.casefold())
        base.loc[base_match, "color"] = chosen
        if not component_frame.empty:
            component_frame.loc[component_match, "color"] = chosen

    return base, component_frame, assignments


def validate_distinct_population_colours(
    base_mapping: pd.DataFrame,
    components: pd.DataFrame | None = None,
) -> None:
    """Reject invalid colours and reuse across different final populations."""

    frames = [base_mapping[["proposed_label", "color"]]]
    if components is not None and not components.empty:
        frames.append(components[["proposed_label", "color"]])
    rows = pd.concat(frames, ignore_index=True)
    invalid = (
        rows.loc[
            rows["color"].map(normalise_hex_colour).eq(""),
            "proposed_label",
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if invalid:
        raise ValueError(
            "Every final population needs a valid #RRGGBB colour. Invalid for: "
            + ", ".join(invalid[:12])
        )
    collisions = categorical_colour_collisions(
        rows["proposed_label"].astype(str).tolist(),
        rows["color"].astype(str).tolist(),
    )
    if collisions:
        detail = "; ".join(
            f"{colour}: {', '.join(labels)}"
            for colour, labels in collisions.items()
        )
        raise ValueError(
            "Different final populations cannot share one colour. " + detail
        )


def save_population_draft(
    workspace_paths: PopulationWorkspacePaths,
    draft: PopulationDraft,
    base_mapping: pd.DataFrame,
    components: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    adata: Any,
    action: str = "save_draft",
    details: dict[str, Any] | None = None,
) -> PopulationDraft:
    """Validate and atomically save one draft, logging semantic edits."""

    draft = PopulationDraft.model_validate(draft.model_dump(mode="python"))
    if source_obs_fingerprint(adata, draft.source_obs) != draft.source_fingerprint:
        raise ValueError(
            "The draft source observation has changed; start an explicit new "
            "population workspace."
        )
    conflicting = [
        sibling
        for sibling in list_population_drafts(workspace_paths)
        if sibling.draft_id != draft.draft_id
        and sibling.derived_obs == draft.derived_obs
    ]
    if conflicting:
        raise ValueError(
            f"Another sibling draft already targets obs[{draft.derived_obs!r}]."
        )
    expected = ordered_source_labels(adata.obs[draft.source_obs])
    base = validate_base_mapping(
        base_mapping, expected_source_values=expected
    )
    clean_components, clean_membership = validate_component_tables(
        components,
        membership,
        adata=adata,
        source_obs=draft.source_obs,
    )
    _labels, effective_summary = synthesize_population_labels(
        adata,
        source_obs=draft.source_obs,
        base_mapping=base,
        components=clean_components,
        membership=clean_membership,
    )
    base, clean_components, _merge_colours = harmonize_merge_colours(
        base,
        clean_components,
        merge_labels=effective_summary["merge_groups"],
    )
    validate_distinct_population_colours(base, clean_components)
    paths = population_draft_paths(workspace_paths, draft)
    previous_draft = (
        read_model(paths.manifest, PopulationDraft)
        if paths.manifest.is_file()
        else None
    )
    previous = (
        validate_base_mapping(read_dataframe(paths.base_mapping))
        if paths.base_mapping.is_file()
        else pd.DataFrame(columns=BASE_MAPPING_COLUMNS)
    )
    previous_components = (
        read_dataframe(paths.components)
        if paths.components.is_file()
        else empty_components()
    )
    changes = _mapping_changes(previous, base)
    component_changes = _component_changes(
        previous_components,
        clean_components,
    )
    draft_changes = []
    if previous_draft is not None:
        for field in ("name", "derived_obs", "status", "latest_run_id"):
            before_value = getattr(previous_draft, field)
            after_value = getattr(draft, field)
            if before_value != after_value:
                draft_changes.append(
                    {
                        "field": field,
                        "before": before_value,
                        "after": after_value,
                    }
                )
    write_dataframe(paths.base_mapping, base)
    write_dataframe(paths.components, clean_components)
    write_dataframe(paths.membership, clean_membership)
    updated = draft.model_copy(deep=True)
    updated.revision += 1
    updated.updated_at = utc_now()
    write_yaml(paths.manifest, updated)
    workspace = read_model(workspace_paths.manifest, PopulationWorkspace)
    workspace.updated_at = utc_now()
    if updated.draft_id not in workspace.draft_ids:
        workspace.draft_ids.append(updated.draft_id)
    write_yaml(workspace_paths.manifest, workspace)
    payload = {
        "draft_revision": updated.revision,
        "draft_changes": draft_changes,
        "mapping_changes": changes,
        "component_changes": component_changes,
        "merge_groups": effective_summary["merge_groups"],
        "component_count": len(clean_components),
        "split_cell_count": len(clean_membership),
        **(details or {}),
    }
    append_population_audit(
        workspace_paths,
        action=action,
        draft_id=updated.draft_id,
        details=payload,
    )
    return updated


def import_base_mapping_csv(
    path: str | Path,
    current_mapping: pd.DataFrame,
    *,
    source_obs: str,
    derived_obs: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge a flexible preliminary-label CSV into the current base table."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Population mapping CSV not found: {source}")
    imported = pd.read_csv(source)
    if imported.empty:
        raise ValueError(f"Population mapping CSV is empty: {source}")
    source_candidates = [
        source_obs,
        "source_value",
        "source_population",
        "cluster",
        "leiden",
    ]
    source_column = next(
        (column for column in source_candidates if column in imported.columns),
        str(imported.columns[0]),
    )
    label_candidates = [
        derived_obs,
        "proposed_label",
        "final_population",
        "label",
        "name",
        "population",
    ]
    label_column = next(
        (
            column
            for column in label_candidates
            if column in imported.columns and column != source_column
        ),
        None,
    )
    if label_column is None:
        remaining = [column for column in imported.columns if column != source_column]
        if not remaining:
            raise ValueError(
                "The mapping CSV needs a source-cluster column and a proposed-label column."
            )
        label_column = str(remaining[0])
    working = imported[[source_column, label_column]].copy()
    working.columns = ["source_value", "proposed_label"]
    working["source_value"] = working["source_value"].map(_clean_text)
    working["proposed_label"] = working["proposed_label"].map(_clean_text)
    working = working.loc[
        working["source_value"].ne("") & working["proposed_label"].ne("")
    ]
    if working["source_value"].duplicated().any():
        duplicates = working.loc[
            working["source_value"].duplicated(keep=False), "source_value"
        ].unique()
        raise ValueError(
            f"Imported mapping contains duplicate source values: {duplicates.tolist()}"
        )
    current = validate_base_mapping(current_mapping)
    unknown = sorted(set(working["source_value"]) - set(current["source_value"]))
    if unknown:
        raise ValueError(
            f"Imported mapping contains unknown source populations: {unknown[:10]}"
        )
    lookup = working.set_index("source_value")["proposed_label"].to_dict()
    updated = current.copy()
    selected = updated["source_value"].isin(lookup)
    updated.loc[selected, "proposed_label"] = updated.loc[
        selected, "source_value"
    ].map(lookup)
    if "color" in imported.columns:
        colour_working = imported[[source_column, "color"]].copy()
        colour_working.columns = ["source_value", "color"]
        colour_lookup = {
            _clean_text(key): _clean_text(value)
            for key, value in colour_working.itertuples(index=False, name=None)
            if _clean_text(key) and _clean_text(value)
        }
        selected = updated["source_value"].isin(colour_lookup)
        updated.loc[selected, "color"] = updated.loc[
            selected, "source_value"
        ].map(colour_lookup)
    summary = {
        "path": str(source.resolve(strict=False)),
        "source_column": source_column,
        "label_column": label_column,
        "updated_population_count": len(working),
        "unmapped_source_count": int(len(updated) - len(working)),
    }
    return validate_base_mapping(updated), summary


def _stable_component_id(
    method: str,
    run_id: str,
    parent: str,
    value: str,
) -> str:
    payload = "\0".join((method, run_id, parent, value)).encode("utf-8")
    suffix = hashlib.sha256(payload).hexdigest()[:12]
    return f"{slugify(method)}_{suffix}"


def component_tables_from_assignments(
    adata: Any,
    *,
    source_obs: str,
    assignments: pd.DataFrame,
    method: str,
    run_id: str | None = None,
    obs_name_column: str | None = None,
    label_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Normalize Scanpy/image classifier cell assignments into split components."""

    if source_obs not in adata.obs:
        raise KeyError(f"AnnData observation {source_obs!r} does not exist.")
    if assignments.empty:
        raise ValueError("The subcluster assignment table is empty.")
    obs_candidates = [
        obs_name_column,
        "obs_name",
        "cell_id",
        "index",
        "master_index",
    ]
    resolved_obs = next(
        (column for column in obs_candidates if column and column in assignments),
        None,
    )
    label_candidates = [
        label_column,
        "proposed_label",
        "final_population",
        "subclass",
        "class_name",
        "class_id",
        "predicted_class",
        "component_value",
        "cluster",
        "leiden",
        "label",
    ]
    resolved_label = next(
        (
            column
            for column in label_candidates
            if column and column in assignments and column != resolved_obs
        ),
        None,
    )
    if resolved_obs is None or resolved_label is None:
        raise ValueError(
            "Cell-level split results need an identity column (for example "
            "obs_name) and a label/class column."
        )
    resolved_run = run_id or str(uuid4())
    working = assignments[[resolved_obs, resolved_label]].copy()
    working.columns = ["obs_name", "component_value"]
    working["obs_name"] = working["obs_name"].map(_clean_text)
    working["component_value"] = working["component_value"].map(_clean_text)
    working = working.loc[
        working["obs_name"].ne("") & working["component_value"].ne("")
    ].drop_duplicates(subset=["obs_name"], keep="last")
    known = set(adata.obs_names.astype(str))
    unknown = sorted(set(working["obs_name"]) - known)
    if unknown:
        raise ValueError(
            f"Subcluster assignments contain unknown AnnData cells: {unknown[:10]}"
        )
    source_lookup = (
        adata.obs[source_obs]
        .astype(object)
        .map(normalise_source_label)
        .astype("string")
    )
    source_lookup.index = source_lookup.index.astype(str)
    working["parent_source_value"] = working["obs_name"].map(source_lookup)
    working = working.dropna(subset=["parent_source_value"])
    working["parent_source_value"] = working["parent_source_value"].astype(str)
    component_rows = []
    membership_rows = []
    colour_index = 0
    for (parent, value), group in working.groupby(
        ["parent_source_value", "component_value"], sort=True, observed=True
    ):
        component_id = _stable_component_id(
            str(method), resolved_run, str(parent), str(value)
        )
        component_rows.append(
            {
                "component_id": component_id,
                "parent_source_value": str(parent),
                "method": str(method),
                "run_id": resolved_run,
                "component_value": str(value),
                "cell_count": len(group),
                "proposed_label": str(value),
                "color": _FALLBACK_COLOURS[
                    colour_index % len(_FALLBACK_COLOURS)
                ],
                "notes": "",
            }
        )
        colour_index += 1
        membership_rows.extend(
            {"obs_name": obs_name, "component_id": component_id}
            for obs_name in group["obs_name"]
        )
    components = pd.DataFrame(component_rows, columns=COMPONENT_COLUMNS)
    membership = pd.DataFrame(membership_rows, columns=MEMBERSHIP_COLUMNS)
    components, membership = validate_component_tables(
        components,
        membership,
        adata=adata,
        source_obs=source_obs,
    )
    return components, membership, {
        "method": str(method),
        "run_id": resolved_run,
        "obs_name_column": resolved_obs,
        "label_column": resolved_label,
        "component_count": len(components),
        "assigned_cell_count": len(membership),
    }


def integrate_component_tables(
    existing_components: pd.DataFrame,
    existing_membership: pd.DataFrame,
    new_components: pd.DataFrame,
    new_membership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Replace overlapping cell assignments and retain unaffected components."""

    existing_components, existing_membership = validate_component_tables(
        existing_components, existing_membership
    )
    new_components, new_membership = validate_component_tables(
        new_components, new_membership
    )
    replacing = set(new_membership["obs_name"])
    retained_membership = existing_membership.loc[
        ~existing_membership["obs_name"].isin(replacing)
    ].copy()
    retained_ids = set(retained_membership["component_id"])
    retained_components = existing_components.loc[
        existing_components["component_id"].isin(retained_ids)
    ].copy()
    duplicate_ids = set(retained_components["component_id"]) & set(
        new_components["component_id"]
    )
    if duplicate_ids:
        raise ValueError(f"Duplicate component IDs during import: {sorted(duplicate_ids)}")
    combined_components = pd.concat(
        [retained_components, new_components], ignore_index=True
    )
    combined_membership = pd.concat(
        [retained_membership, new_membership], ignore_index=True
    )
    combined_components, combined_membership = validate_component_tables(
        combined_components, combined_membership
    )
    return combined_components, combined_membership, {
        "replaced_cell_count": len(replacing & set(existing_membership["obs_name"])),
        "retained_component_count": len(retained_components),
        "new_component_count": len(new_components),
    }


def synthesize_population_labels(
    adata: Any,
    *,
    source_obs: str,
    base_mapping: pd.DataFrame,
    components: pd.DataFrame | None = None,
    membership: pd.DataFrame | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Create final labels without mutating AnnData."""

    if source_obs not in adata.obs:
        raise KeyError(f"AnnData observation {source_obs!r} does not exist.")
    expected = ordered_source_labels(adata.obs[source_obs])
    base = validate_base_mapping(
        base_mapping, expected_source_values=expected
    )
    clean_components, clean_membership = validate_component_tables(
        components if components is not None else empty_components(),
        membership if membership is not None else empty_membership(),
        adata=adata,
        source_obs=source_obs,
    )
    lookup = base.set_index("source_value")["proposed_label"].to_dict()
    source_values = (
        adata.obs[source_obs]
        .astype(object)
        .map(normalise_source_label)
        .astype("string")
    )
    labels = source_values.map(lookup).astype("string")
    labels.index = adata.obs_names
    if not clean_membership.empty:
        component_labels = clean_components.set_index("component_id")[
            "proposed_label"
        ].to_dict()
        overrides = clean_membership.copy()
        overrides["proposed_label"] = overrides["component_id"].map(
            component_labels
        )
        labels.loc[overrides["obs_name"].astype(str)] = overrides[
            "proposed_label"
        ].to_numpy()
    categories = list(
        dict.fromkeys(
            [
                *base["proposed_label"].tolist(),
                *clean_components["proposed_label"].tolist(),
            ]
        )
    )
    labels = pd.Series(
        pd.Categorical(labels, categories=categories),
        index=adata.obs_names,
        name="population",
    )
    counts = labels.value_counts(dropna=False).to_dict()
    effective_contributors: dict[str, list[str]] = {}
    split_obs_names = set(clean_membership["obs_name"].astype(str))
    source_with_string_index = (
        adata.obs[source_obs]
        .astype(object)
        .map(normalise_source_label)
        .astype("string")
    )
    source_with_string_index.index = source_with_string_index.index.astype(str)
    for row in base.itertuples(index=False):
        source_cells = set(
            source_with_string_index.index[
                source_with_string_index.eq(str(row.source_value)).fillna(False)
            ]
        )
        if source_cells - split_obs_names:
            effective_contributors.setdefault(str(row.proposed_label), []).append(
                f"source:{row.source_value}"
            )
    for row in clean_components.itertuples(index=False):
        effective_contributors.setdefault(str(row.proposed_label), []).append(
            f"{row.method}:{row.parent_source_value}/{row.component_value}"
        )
    effective_merges = {
        label: contributors
        for label, contributors in effective_contributors.items()
        if len(set(contributors)) > 1
    }
    summary = {
        "cell_count": int(len(labels)),
        "label_count": int(labels.nunique(dropna=True)),
        "missing_source_cells": int(labels.isna().sum()),
        "split_cell_count": int(len(clean_membership)),
        "merge_groups": effective_merges,
        "label_counts": {
            str(label): int(count) for label, count in counts.items()
        },
    }
    return labels, summary


def _mapping_fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return hashlib.sha256(b"").hexdigest()
    return dataframe_sha256(frame, columns)


def apply_population_draft(
    adata: Any,
    *,
    draft: PopulationDraft,
    base_mapping: pd.DataFrame,
    components: pd.DataFrame,
    membership: pd.DataFrame,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Apply a draft to the live working AnnData with explicit overwrite control."""

    current_fingerprint = source_obs_fingerprint(adata, draft.source_obs)
    if current_fingerprint != draft.source_fingerprint:
        raise ValueError(
            "The draft source fingerprint no longer matches the live AnnData."
        )
    if draft.derived_obs in adata.obs and not overwrite:
        raise ValueError(
            f"AnnData already contains obs[{draft.derived_obs!r}]. Enable explicit "
            "overwrite or choose a new derived observation."
        )
    labels, summary = synthesize_population_labels(
        adata,
        source_obs=draft.source_obs,
        base_mapping=base_mapping,
        components=components,
        membership=membership,
    )
    harmonized_base, harmonized_components, _merge_colours = (
        harmonize_merge_colours(
            base_mapping,
            components,
            merge_labels=summary["merge_groups"],
        )
    )
    validate_distinct_population_colours(
        harmonized_base,
        harmonized_components,
    )
    adata.obs[draft.derived_obs] = labels

    colour_rows = pd.concat(
        [
            harmonized_base[["proposed_label", "color"]],
            harmonized_components[["proposed_label", "color"]]
            if not harmonized_components.empty
            else pd.DataFrame(columns=["proposed_label", "color"]),
        ],
        ignore_index=True,
    )
    colour_lookup: dict[str, str] = {}
    colour_conflicts: dict[str, list[str]] = {}
    for label, group in colour_rows.groupby("proposed_label", sort=False):
        colours = [colour for colour in group["color"].map(_clean_text) if colour]
        unique = list(dict.fromkeys(colours))
        colour_lookup[str(label)] = unique[0] if unique else "#ffffff"
        if len(unique) > 1:
            colour_conflicts[str(label)] = unique
    categories = [str(value) for value in adata.obs[draft.derived_obs].cat.categories]
    adata.uns[f"{draft.derived_obs}_colors"] = np.asarray(
        [colour_lookup.get(value, "#ffffff") for value in categories],
        dtype=str,
    )

    applied_at = utc_now().isoformat()
    provenance = {
        "draft_id": draft.draft_id,
        "draft_revision": draft.revision,
        "source_obs": draft.source_obs,
        "derived_obs": draft.derived_obs,
        "source_fingerprint": draft.source_fingerprint,
        "base_mapping_fingerprint": _mapping_fingerprint(
            validate_base_mapping(harmonized_base),
            ["source_value", "proposed_label", "color"],
        ),
        "component_fingerprint": _mapping_fingerprint(
            harmonized_components,
            ["component_id", "parent_source_value", "proposed_label"],
        )
        if not harmonized_components.empty
        else hashlib.sha256(b"").hexdigest(),
        "membership_fingerprint": _mapping_fingerprint(
            membership, ["obs_name", "component_id"]
        )
        if not membership.empty
        else hashlib.sha256(b"").hexdigest(),
        "applied_at": applied_at,
        "merge_groups": summary["merge_groups"],
        "colour_conflicts": colour_conflicts,
        "label_counts": summary["label_counts"],
    }
    napari_uns = dict(adata.uns.get("napari_sbt", {}) or {})
    population_uns = dict(napari_uns.get("population_curation", {}) or {})
    population_uns[draft.derived_obs] = provenance
    napari_uns["population_curation"] = population_uns
    adata.uns["napari_sbt"] = napari_uns
    return {**summary, **provenance}


def population_draft_sync_state(
    adata: Any,
    draft: PopulationDraft,
) -> Literal["missing", "stale", "synced", "conflict"]:
    """Describe whether a saved draft revision owns the live derived obs."""

    if draft.derived_obs not in adata.obs:
        return "missing"
    napari_uns = adata.uns.get("napari_sbt", {})
    population_uns = (
        napari_uns.get("population_curation", {})
        if isinstance(napari_uns, dict)
        else {}
    )
    provenance = (
        population_uns.get(draft.derived_obs, {})
        if isinstance(population_uns, dict)
        else {}
    )
    if not isinstance(provenance, dict) or provenance.get("draft_id") != draft.draft_id:
        return "conflict"
    try:
        applied_revision = int(provenance.get("draft_revision", -1))
    except (TypeError, ValueError):
        return "stale"
    return "synced" if applied_revision == draft.revision else "stale"


def atomic_write_curated_anndata(adata: Any, destination: str | Path) -> Path:
    """Write a new AnnData copy and refuse to replace any existing object."""

    output = Path(destination).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing curated AnnData: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    try:
        adata.write_h5ad(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def save_graph_subcluster_request(
    request: GraphSubclusterRequest,
    destination: str | Path,
) -> Path:
    return write_json(destination, request)


__all__ = [
    "BASE_MAPPING_COLUMNS",
    "COMPONENT_COLUMNS",
    "CURATION_SCHEMA_VERSION",
    "GraphSubclusterRequest",
    "MEMBERSHIP_COLUMNS",
    "PopulationDraft",
    "PopulationDraftPaths",
    "PopulationWorkspace",
    "PopulationWorkspacePaths",
    "append_population_audit",
    "apply_population_draft",
    "atomic_write_curated_anndata",
    "build_base_mapping",
    "component_tables_from_assignments",
    "create_population_draft",
    "empty_components",
    "empty_membership",
    "ensure_population_workspace",
    "harmonize_merge_colours",
    "import_base_mapping_csv",
    "integrate_component_tables",
    "list_population_drafts",
    "load_population_draft",
    "merge_groups",
    "normalise_source_label",
    "ordered_source_labels",
    "population_draft_paths",
    "population_draft_sync_state",
    "population_workspace_paths",
    "read_population_audit",
    "save_graph_subcluster_request",
    "save_population_draft",
    "source_obs_fingerprint",
    "synthesize_population_labels",
    "validate_base_mapping",
    "validate_component_tables",
]
