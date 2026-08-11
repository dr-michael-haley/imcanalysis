"""Pure helpers for reproducible cohort-aware Explore views."""

from __future__ import annotations

import hashlib
import json
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
    "EXPLORE_STATE_VERSION",
    "EXPLORE_RECIPE_FUNCTION_KEYS",
    "ExploreRecipePreset",
    "ExploreReviewState",
    "ExploreViewRecipe",
    "SIX_COLOUR_COLORMAPS",
    "categorical_colour_map",
    "marker_values",
    "observation_categories",
    "population_recipe_key",
    "recipe_layer_data_is_current",
]
