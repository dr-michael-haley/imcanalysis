from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from SpatialBiologyToolkit.napari_sbt.explore import (
    ExploreReviewState,
    ExploreViewRecipe,
    categorical_colour_map,
    marker_values,
    population_recipe_key,
)


def _adata_like(*, colours=None):
    obs = pd.DataFrame(
        {
            "population": pd.Categorical(
                ["B cell", "T cell", "B cell"],
                categories=["T cell", "B cell", "Myeloid"],
            )
        }
    )
    uns = {} if colours is None else {"population_colors": colours}
    return SimpleNamespace(
        obs=obs,
        uns=uns,
        var_names=np.asarray(["CD3", "CD20"]),
        X=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        n_obs=3,
    )


def test_categorical_colour_map_uses_anndata_uns_category_order():
    adata = _adata_like(colours=["#111111", "#222222", "#333333"])

    colours = categorical_colour_map(adata, "population")

    assert colours == {
        "T cell": "#111111",
        "B cell": "#222222",
        "Myeloid": "#333333",
    }


def test_categorical_colour_map_accepts_mapping_and_fills_missing_values():
    adata = _adata_like(colours={"B cell": "#abcdef"})

    colours = categorical_colour_map(adata, "population")

    assert colours["B cell"] == "#abcdef"
    assert colours["T cell"].startswith("#")
    assert colours["Myeloid"].startswith("#")


def test_marker_values_extracts_one_dense_anndata_x_column():
    adata = _adata_like()

    assert marker_values(adata, "CD20").tolist() == [2.0, 4.0, 6.0]


def test_explore_recipe_fingerprint_tracks_colour_assignment_order():
    red_green = ExploreViewRecipe(
        image_mode="six_colour",
        image_channels=["CD3", "CD20"],
    )
    green_red = ExploreViewRecipe(
        image_mode="six_colour",
        image_channels=["CD20", "CD3"],
    )

    assert red_green.has_content
    assert red_green.fingerprint != green_red.fingerprint


def test_population_recipes_and_viewed_rois_round_trip():
    key = population_recipe_key("leiden", "3")
    recipe = ExploreViewRecipe(
        image_mode="grayscale",
        image_channels=["CD3"],
        population_observation="leiden",
        populations=["3"],
    )
    state = ExploreReviewState(
        population_recipes={key: recipe},
        viewed_rois={recipe.fingerprint: ["ROI_1", "ROI_2"]},
    )

    restored = ExploreReviewState.model_validate(state.model_dump(mode="json"))

    assert restored.population_recipes[key] == recipe
    assert restored.viewed_rois[recipe.fingerprint] == ["ROI_1", "ROI_2"]
