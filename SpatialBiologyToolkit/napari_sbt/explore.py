"""Pure helpers for reproducible cohort-aware Explore views."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

EXPLORE_STATE_VERSION = 3
EXPLORE_RECIPE_FUNCTION_KEYS = tuple(f"F{index}" for index in range(1, 13))
SIX_COLOUR_COLORMAPS = ("red", "green", "blue", "cyan", "yellow", "magenta")
FALLBACK_CATEGORY_COLOURS = (
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


def identity_value_map(
    mask: np.ndarray,
    values: pd.Series,
    *,
    dtype=np.float32,
    background_value: float = 0,
) -> np.ndarray:
    """Map object identities to pixels with one vectorized mask traversal."""

    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Identity maps require a 2-D mask, got {mask.shape}.")
    if values.empty:
        return np.full(mask.shape, background_value, dtype=dtype)

    # A small Python pass over identities is intentional: it preserves the
    # previous last-value-wins behaviour for duplicate IDs without ever scanning
    # the image once per cell.
    mapped_values: dict[int, Any] = {}
    for object_id, value in values.items():
        if pd.notna(value):
            mapped_values[int(object_id)] = value
    if not mapped_values:
        return np.full(mask.shape, background_value, dtype=dtype)

    object_ids = np.fromiter(mapped_values, dtype=np.int64)
    pixel_values = np.asarray(list(mapped_values.values()), dtype=dtype)
    maximum_label = int(mask.max(initial=0))
    usable = (object_ids >= 0) & (object_ids <= maximum_label)
    object_ids = object_ids[usable]
    pixel_values = pixel_values[usable]
    if object_ids.size == 0:
        return np.full(mask.shape, background_value, dtype=dtype)

    if bool(np.all(pixel_values == pixel_values[0])):
        output = np.full(mask.shape, background_value, dtype=dtype)
        output[np.isin(mask, object_ids)] = pixel_values[0]
        return output

    output_dtype = np.dtype(dtype)
    lookup_bytes = (maximum_label + 1) * output_dtype.itemsize
    lookup_limit = max(64 * 1024 * 1024, int(mask.nbytes) * 4)
    if int(mask.min(initial=0)) >= 0 and lookup_bytes <= lookup_limit:
        lookup = np.full(maximum_label + 1, background_value, dtype=output_dtype)
        lookup[object_ids] = pixel_values
        return lookup[mask]

    # Extremely sparse masks can have a very large maximum label. A sorted
    # search remains vectorized without allocating a maximum-label-sized table.
    order = np.argsort(object_ids)
    sorted_ids = object_ids[order]
    sorted_values = pixel_values[order]
    flat_mask = mask.ravel()
    positions = np.searchsorted(sorted_ids, flat_mask)
    candidates = positions < sorted_ids.size
    matched = np.zeros(flat_mask.size, dtype=bool)
    matched[candidates] = sorted_ids[positions[candidates]] == flat_mask[candidates]
    output = np.full(mask.shape, background_value, dtype=dtype)
    output_flat = output.ravel()
    output_flat[matched] = sorted_values[positions[matched]]
    return output


def population_identity_map(
    mask: np.ndarray,
    object_ids,
    *,
    dtype=np.int32,
) -> np.ndarray:
    """Retain selected object IDs so touching population cells stay distinct."""

    identities = pd.to_numeric(pd.Series(object_ids, copy=False), errors="coerce")
    identities = identities.dropna().astype(np.int64)
    identities = identities.loc[identities.gt(0)]
    mapping = pd.Series(
        identities.to_numpy(dtype=dtype, copy=False),
        index=identities.to_numpy(dtype=np.int64, copy=False),
    )
    return identity_value_map(mask, mapping, dtype=dtype)


def categorical_object_categories(
    object_ids,
    values,
) -> pd.Series:
    """Return one category per valid object ID, preserving identity labels.

    The returned series is indexed by ``ObjectNumber`` and contains category
    text.  Missing categories and non-positive/non-numeric identities are
    omitted.  Duplicate identities retain the final value, matching
    :func:`identity_value_map` and the historical overlay behaviour.
    """

    identities = pd.to_numeric(pd.Series(object_ids, copy=False), errors="coerce")
    categories = pd.Series(values, copy=False)
    if len(identities) != len(categories):
        raise ValueError(
            "Categorical overlays require one observation value per object ID."
        )
    usable = identities.notna() & identities.gt(0) & categories.notna()
    if not bool(usable.any()):
        return pd.Series(dtype="string", index=pd.Index([], dtype=np.int64))
    result = pd.Series(
        categories.loc[usable].astype(str).to_numpy(),
        index=identities.loc[usable].astype(np.int64).to_numpy(),
        dtype="string",
    )
    return result.loc[~result.index.duplicated(keep="last")]


def _colour_text(value: Any) -> str | None:
    """Normalize common Scanpy/Matplotlib colour representations."""

    if isinstance(value, str):
        text = value.strip()
        return text or None
    try:
        channels = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if channels.size not in {3, 4} or not np.isfinite(channels).all():
        return None
    if float(channels.max(initial=0)) <= 1:
        channels = channels * 255
    channels = np.clip(np.rint(channels), 0, 255).astype(np.uint8)
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def observation_categories(adata, observation: str) -> list[str]:
    """Return observation categories in the order used by AnnData colours."""

    if observation not in adata.obs:
        raise KeyError(f"AnnData observation does not exist: {observation}")
    series = adata.obs[observation]
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [str(value) for value in series.cat.categories]
    return sorted(series.dropna().astype(str).unique().tolist())


def _uns_colour_candidate(adata, observation: str) -> Any:
    for key in (
        f"{observation}_colors",
        f"{observation}_colour_map",
        f"{observation}_color_map",
        f"{observation}_colormap",
    ):
        if key in adata.uns:
            return adata.uns[key]
    for container_key in ("colors", "colours", "color_maps", "colormaps"):
        container = adata.uns.get(container_key)
        if isinstance(container, dict) and observation in container:
            return container[observation]
    return None


def categorical_colour_map(adata, observation: str) -> dict[str, str]:
    """
    Return category-to-colour mappings, preferring any palette in ``adata.uns``.

    Scanpy's conventional ``uns["<obs>_colors"]`` list is aligned to categorical
    order. Mapping-style palettes and a few common container/key spellings are
    accepted for interoperability.
    """

    categories = observation_categories(adata, observation)
    candidate = _uns_colour_candidate(adata, observation)
    colour_map: dict[str, str] = {}
    if isinstance(candidate, dict):
        for category in categories:
            colour = _colour_text(
                candidate.get(category, candidate.get(str(category)))
            )
            if colour:
                colour_map[category] = colour
    elif candidate is not None:
        try:
            candidate_values = list(candidate)
        except TypeError:
            candidate_values = []
        for category, value in zip(categories, candidate_values):
            colour = _colour_text(value)
            if colour:
                colour_map[category] = colour
    for index, category in enumerate(categories):
        colour_map.setdefault(
            category,
            FALLBACK_CATEGORY_COLOURS[index % len(FALLBACK_CATEGORY_COLOURS)],
        )
    return colour_map


def marker_values(adata, marker: str) -> np.ndarray:
    """Return one dense, one-dimensional ``adata.X`` marker vector."""

    positions = pd.Index(adata.var_names.astype(str)).get_indexer([str(marker)])
    if positions[0] < 0:
        raise KeyError(f"AnnData marker does not exist: {marker}")
    matrix = adata.X[:, positions[0]]
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    elif hasattr(matrix, "A"):
        matrix = matrix.A
    values = np.asarray(matrix).reshape(-1)
    if values.size != adata.n_obs:
        raise ValueError(
            f"Marker {marker!r} returned {values.size} values for "
            f"{adata.n_obs} AnnData observations."
        )
    return values


def rank_marker_rois(
    adata,
    *,
    marker: str,
    roi_obs: str,
    eligible_rois: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[tuple[str, float]]:
    """Rank ROIs by mean ``adata.X`` signal quantified inside segmented cells."""

    if roi_obs not in adata.obs:
        raise KeyError(f"AnnData ROI observation does not exist: {roi_obs}")
    expression = np.asarray(marker_values(adata, marker), dtype=float)
    roi_values = adata.obs[roi_obs].astype("string")
    usable = roi_values.notna().to_numpy() & np.isfinite(expression)
    if not bool(usable.any()):
        return []
    working = pd.DataFrame(
        {
            "roi": roi_values.loc[usable].astype(str).to_numpy(),
            "expression": expression[usable],
        }
    )
    means = working.groupby("roi", sort=False, observed=True)["expression"].mean()
    allowed = (
        None
        if eligible_rois is None
        else {str(roi) for roi in eligible_rois}
    )
    ranked = [
        (str(roi), float(mean))
        for roi, mean in means.items()
        if allowed is None or str(roi) in allowed
    ]
    return sorted(ranked, key=lambda item: (-item[1], item[0].casefold(), item[0]))


def roi_level_metadata(
    obs: pd.DataFrame,
    *,
    roi_obs: str,
    exclude_columns: tuple[str, ...] = (),
) -> dict[str, dict[str, Any]]:
    """Return scalar obs fields which are constant within every represented ROI.

    Missing values participate in the constancy check. A column containing both a
    value and a missing entry in one ROI is therefore not presented as reliable
    sample metadata. Unhashable object-valued columns are ignored.
    """

    if roi_obs not in obs:
        raise KeyError(f"AnnData ROI observation does not exist: {roi_obs}")
    if obs.empty:
        return {}
    roi_values = obs[roi_obs].astype("string")
    usable = roi_values.notna()
    if not bool(usable.any()):
        return {}
    working = obs.loc[usable]
    working_rois = roi_values.loc[usable]
    grouped = working.groupby(working_rois, sort=False, observed=True)
    excluded = {str(roi_obs), *(str(column) for column in exclude_columns)}
    metadata_columns: list[Any] = []
    for column in working.columns:
        name = str(column)
        if name in excluded or not bool(working[column].notna().any()):
            continue
        try:
            per_roi_values = grouped[column].nunique(dropna=False)
        except (TypeError, ValueError):
            continue
        if not per_roi_values.empty and bool(per_roi_values.le(1).all()):
            metadata_columns.append(column)

    result: dict[str, dict[str, Any]] = {}
    roi_array = working_rois.astype(str).to_numpy()
    for roi in dict.fromkeys(roi_array.tolist()):
        first_position = int(np.flatnonzero(roi_array == roi)[0])
        row = working.iloc[first_position]
        result[str(roi)] = {
            str(column): row[column] for column in metadata_columns
        }
    return result


def cell_level_observations(
    obs: pd.DataFrame,
    *,
    roi_obs: str,
    object_obs: str,
) -> list[str]:
    """Return obs fields which vary within at least one represented ROI."""

    metadata = roi_level_metadata(
        obs,
        roi_obs=roi_obs,
        exclude_columns=(object_obs,),
    )
    roi_level = {
        str(column)
        for values in metadata.values()
        for column in values
    }
    identity = {str(roi_obs), str(object_obs)}
    return [
        str(column)
        for column in obs.columns
        if str(column) not in identity and str(column) not in roi_level
    ]


def format_roi_metadata_value(value: Any) -> str:
    """Format common AnnData obs scalar types for compact read-only display."""

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return "Missing"
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        return str(pd.Timedelta(value))
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return f"{numeric:.8g}" if np.isfinite(numeric) else str(numeric)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class ExploreViewRecipe(BaseModel):
    """The ROI-independent set of layers used for one visual assessment."""

    schema_version: int = EXPLORE_STATE_VERSION
    image_mode: Literal["none", "grayscale", "six_colour", "rgb"] = "none"
    image_channels: list[str] = Field(default_factory=list)
    observation_overlay: str | None = None
    observation_overlay_full_dataset: bool = False
    population_observation: str | None = None
    populations: list[str] = Field(default_factory=list)
    marker_overlays: list[str] = Field(default_factory=list)
    layer_colormaps: dict[str, str] = Field(default_factory=dict)
    layer_colormap_specs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    layer_visibility: dict[str, bool] = Field(default_factory=dict)
    layer_opacities: dict[str, float] = Field(default_factory=dict)
    layer_contours: dict[str, int] = Field(default_factory=dict)
    layer_contrast_limits: dict[str, tuple[float, float]] = Field(
        default_factory=dict
    )

    @property
    def has_content(self) -> bool:
        return bool(
            self.image_channels
            or self.observation_overlay
            or self.populations
            or self.marker_overlays
            or self.layer_colormap_specs
            or self.layer_visibility
            or self.layer_opacities
            or self.layer_contours
            or self.layer_contrast_limits
        )

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def recipe_layer_data_is_current(
    metadata: Any,
    *,
    name: str,
    roi: str,
    reload_descriptor: dict[str, Any],
) -> bool:
    """Return whether a displayed layer already contains the requested data.

    Display-only settings such as an image's greyscale/six-colour mode are
    deliberately excluded.  They can be changed without re-reading the image.
    Older layer metadata without an ROI is treated as stale once, after which
    newly written metadata can participate in reuse.
    """

    if not isinstance(metadata, dict):
        return False
    stored = metadata.get("napari_sbt_reload")
    if not isinstance(stored, dict):
        return False
    if str(stored.get("name", "")) != str(name):
        return False
    if str(stored.get("roi", "")) != str(roi):
        return False
    display_only_keys = {"mode"}
    return all(
        stored.get(key) == value
        for key, value in reload_descriptor.items()
        if key not in display_only_keys
    )


class ExploreRecipePreset(BaseModel):
    """One explicitly named, optionally keyboard-addressable Explore recipe."""

    preset_id: str
    name: str
    shortcut: str | None = None
    recipe: ExploreViewRecipe

    @field_validator("preset_id", "name")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Explore recipe preset IDs and names cannot be empty.")
        return text

    @field_validator("shortcut")
    @classmethod
    def _function_key(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        key = str(value).strip().upper()
        if key not in EXPLORE_RECIPE_FUNCTION_KEYS:
            raise ValueError("Explore recipe shortcuts must be F1 through F12.")
        return key


class ExploreReviewState(BaseModel):
    """Persisted named/population recipes and viewed-ROI sets for an experiment."""

    schema_version: int = EXPLORE_STATE_VERSION
    population_recipes: dict[str, ExploreViewRecipe] = Field(default_factory=dict)
    recipe_presets: dict[str, ExploreRecipePreset] = Field(default_factory=dict)
    active_recipe_id: str | None = None
    viewed_rois: dict[str, list[str]] = Field(default_factory=dict)
    population_qc_contour_width: int = Field(default=1, ge=0, le=20)
    cell_properties_configured: bool = False
    cell_properties_tracking_enabled: bool = True
    cell_properties_observations: list[str] = Field(default_factory=list)
    cell_properties_outline_enabled: bool = False
    cell_properties_outline_colour: str = "#facc15"
    cell_properties_outline_width: int = Field(default=2, ge=1, le=20)

    @field_validator("cell_properties_observations")
    @classmethod
    def _unique_cell_properties(cls, values: list[str]) -> list[str]:
        return list(
            dict.fromkeys(
                text
                for value in values
                if (text := str(value).strip())
            )
        )

    @field_validator("cell_properties_outline_colour")
    @classmethod
    def _valid_cell_outline_colour(cls, value: str) -> str:
        text = str(value).strip()
        try:
            valid = len(text) == 7 and text.startswith("#") and int(text[1:], 16) >= 0
        except ValueError:
            valid = False
        return text if valid else "#facc15"

    @model_validator(mode="after")
    def _validate_recipe_presets(self):
        for key, preset in self.recipe_presets.items():
            if str(key) != preset.preset_id:
                raise ValueError(
                    "Explore recipe preset mapping keys must match preset_id."
                )
        names = [preset.name.casefold() for preset in self.recipe_presets.values()]
        if len(names) != len(set(names)):
            raise ValueError("Explore recipe preset names must be unique.")
        shortcuts = [
            preset.shortcut
            for preset in self.recipe_presets.values()
            if preset.shortcut is not None
        ]
        if len(shortcuts) != len(set(shortcuts)):
            raise ValueError("Each Explore recipe F-key may be assigned only once.")
        if (
            self.active_recipe_id is not None
            and self.active_recipe_id not in self.recipe_presets
        ):
            raise ValueError("The active Explore recipe preset no longer exists.")
        return self


def population_recipe_key(observation: str, population: str) -> str:
    """Return an unambiguous JSON key for one population selector."""

    return json.dumps(
        [str(observation), str(population)],
        ensure_ascii=False,
        separators=(",", ":"),
    )


__all__ = [
    "EXPLORE_RECIPE_FUNCTION_KEYS",
    "EXPLORE_STATE_VERSION",
    "ExploreRecipePreset",
    "ExploreReviewState",
    "ExploreViewRecipe",
    "SIX_COLOUR_COLORMAPS",
    "categorical_colour_map",
    "cell_level_observations",
    "format_roi_metadata_value",
    "identity_value_map",
    "marker_values",
    "observation_categories",
    "population_identity_map",
    "population_recipe_key",
    "rank_marker_rois",
    "recipe_layer_data_is_current",
    "roi_level_metadata",
]
