from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from SpatialBiologyToolkit.napari_sbt.explore import (
    ExploreRecipePreset,
    ExploreReviewState,
    ExploreViewRecipe,
    categorical_colour_map,
    categorical_object_categories,
    cell_level_observations,
    format_roi_metadata_value,
    identity_value_map,
    marker_values,
    population_identity_map,
    population_recipe_key,
    recipe_layer_data_is_current,
    roi_level_metadata,
)


def test_identity_value_map_vectorizes_non_contiguous_ids_and_background():
    mask = np.asarray([[0, 2, 2], [100, 100, 7]], dtype=np.int32)
    values = pd.Series([0.25, 0.75], index=[2, 100])

    mapped = identity_value_map(mask, values, background_value=-1)

    np.testing.assert_allclose(
        mapped,
        [[-1, 0.25, 0.25], [0.75, 0.75, -1]],
    )


def test_identity_value_map_supports_constant_population_masks():
    mask = np.asarray([[0, 2, 5], [9, 5, 2]], dtype=np.int32)
    values = pd.Series([1, 1], index=[2, 9])

    mapped = identity_value_map(mask, values, dtype=np.uint8)

    np.testing.assert_array_equal(mapped, [[0, 1, 0], [1, 0, 1]])


def test_population_identity_map_keeps_touching_population_cells_distinct():
    mask = np.asarray([[0, 2, 2], [9, 9, 5]], dtype=np.int32)

    mapped = population_identity_map(mask, [2, 9])

    np.testing.assert_array_equal(mapped, [[0, 2, 2], [9, 9, 0]])


def test_categorical_object_categories_keeps_same_class_cells_as_distinct_ids():
    mask = np.asarray([[0, 2, 2], [9, 9, 5]], dtype=np.int32)
    object_categories = categorical_object_categories(
        pd.Series([2, 9, 5]),
        pd.Series(pd.Categorical(["T cell", "T cell", pd.NA])),
    )

    mapped = population_identity_map(mask, object_categories.index)

    assert object_categories.to_dict() == {2: "T cell", 9: "T cell"}
    np.testing.assert_array_equal(mapped, [[0, 2, 2], [9, 9, 0]])


def test_categorical_object_categories_rejects_unpaired_inputs():
    with pytest.raises(ValueError, match="one observation value per object ID"):
        categorical_object_categories([1, 2], ["T cell"])


def test_identity_value_map_avoids_large_lookup_for_sparse_high_ids():
    mask = np.asarray([[0, 3], [1_000_000_000, 3]], dtype=np.int64)
    values = pd.Series([2.5, 8.5], index=[3, 1_000_000_000])

    mapped = identity_value_map(mask, values)

    np.testing.assert_allclose(mapped, [[0, 2.5], [8.5, 2.5]])


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


def test_roi_level_metadata_detects_constant_mixed_type_obs_fields():
    obs = pd.DataFrame(
        {
            "ROI": ["A", "A", "B", "B"],
            "ObjectNumber": [1, 2, 1, 2],
            "patient": pd.Categorical(["P1", "P1", "P2", "P2"]),
            "age": pd.Series([61, 61, 48, 48], dtype="Int64"),
            "treated": [True, True, False, False],
            "acquired": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-02-03", "2026-02-03"]
            ),
            "population": ["T", "B", "T", "T"],
            "partly_missing": ["x", None, "y", "y"],
        }
    )

    metadata = roi_level_metadata(
        obs,
        roi_obs="ROI",
        exclude_columns=("ObjectNumber",),
    )

    assert list(metadata["A"]) == ["patient", "age", "treated", "acquired"]
    assert metadata["B"]["patient"] == "P2"
    assert format_roi_metadata_value(metadata["A"]["age"]) == "61"
    assert format_roi_metadata_value(metadata["B"]["treated"]) == "False"
    assert format_roi_metadata_value(metadata["A"]["acquired"]).startswith(
        "2026-01-01"
    )


def test_cell_level_observations_exclude_roi_metadata_and_identity_columns():
    obs = pd.DataFrame(
        {
            "ROI": ["A", "A", "B", "B"],
            "ObjectNumber": [1, 2, 1, 2],
            "patient": ["P1", "P1", "P2", "P2"],
            "population": pd.Categorical(["T", "B", "T", "T"]),
            "intensity": [0.1, 0.5, 0.2, 0.8],
        }
    )

    fields = cell_level_observations(
        obs,
        roi_obs="ROI",
        object_obs="ObjectNumber",
    )

    assert fields == ["population", "intensity"]


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


def test_explore_recipe_round_trips_actual_layer_colormap_specs():
    recipe = ExploreViewRecipe(
        image_mode="six_colour",
        image_channels=["CD3"],
        layer_colormap_specs={
            "image::CD3": {
                "kind": "continuous",
                "name": "red",
                "colours": [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
                "controls": [0.0, 1.0],
                "interpolation": "linear",
            }
        },
    )

    restored = ExploreViewRecipe.model_validate(recipe.model_dump(mode="json"))

    assert restored.layer_colormap_specs == recipe.layer_colormap_specs


def test_recipe_layer_reuse_ignores_display_mode_but_checks_roi_and_source():
    metadata = {
        "napari_sbt_reload": {
            "name": "image::CD3",
            "roi": "ROI_1",
            "kind": "image",
            "channel": "CD3",
            "mode": "six_colour",
            "source": {"path": "CD3.tiff", "size": 10, "mtime_ns": 20},
        }
    }
    requested = {
        "kind": "image",
        "channel": "CD3",
        "mode": "grayscale",
        "source": {"path": "CD3.tiff", "size": 10, "mtime_ns": 20},
    }

    assert recipe_layer_data_is_current(
        metadata,
        name="image::CD3",
        roi="ROI_1",
        reload_descriptor=requested,
    )
    assert not recipe_layer_data_is_current(
        metadata,
        name="image::CD3",
        roi="ROI_2",
        reload_descriptor=requested,
    )
    requested["source"] = {"path": "CD3.tiff", "size": 11, "mtime_ns": 20}
    assert not recipe_layer_data_is_current(
        metadata,
        name="image::CD3",
        roi="ROI_1",
        reload_descriptor=requested,
    )


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
        population_qc_contour_width=4,
        cell_properties_configured=True,
        cell_properties_observations=["population", "leiden"],
        cell_properties_outline_enabled=True,
        cell_properties_outline_colour="#abcdef",
        cell_properties_outline_width=3,
    )

    restored = ExploreReviewState.model_validate(state.model_dump(mode="json"))

    assert restored.population_recipes[key] == recipe
    assert restored.viewed_rois[recipe.fingerprint] == ["ROI_1", "ROI_2"]
    assert restored.population_qc_contour_width == 4
    assert restored.cell_properties_observations == ["population", "leiden"]
    assert restored.cell_properties_outline_colour == "#abcdef"
    assert restored.cell_properties_outline_width == 3


def test_named_recipe_presets_and_function_keys_round_trip():
    recipe = ExploreViewRecipe(
        image_mode="six_colour",
        image_channels=["CD3", "CD20"],
        population_observation="population",
        populations=["T cell"],
    )
    preset = ExploreRecipePreset(
        preset_id="t-cell-view",
        name="T-cell verification",
        shortcut="f3",
        recipe=recipe,
    )
    state = ExploreReviewState(
        recipe_presets={preset.preset_id: preset},
        active_recipe_id=preset.preset_id,
    )

    restored = ExploreReviewState.model_validate(state.model_dump(mode="json"))

    assert restored.active_recipe_id == "t-cell-view"
    assert restored.recipe_presets["t-cell-view"].shortcut == "F3"
    assert restored.recipe_presets["t-cell-view"].recipe == recipe


def test_named_recipe_presets_reject_duplicate_names_and_function_keys():
    recipe = ExploreViewRecipe(image_mode="grayscale", image_channels=["CD3"])
    first = ExploreRecipePreset(
        preset_id="first",
        name="T cells",
        shortcut="F2",
        recipe=recipe,
    )
    duplicate_key = ExploreRecipePreset(
        preset_id="second",
        name="Myeloid",
        shortcut="F2",
        recipe=recipe,
    )
    with pytest.raises(ValueError, match="F-key"):
        ExploreReviewState(
            recipe_presets={"first": first, "second": duplicate_key}
        )

    duplicate_name = duplicate_key.model_copy(
        update={"name": "t CELLS", "shortcut": "F4"}
    )
    with pytest.raises(ValueError, match="names"):
        ExploreReviewState(
            recipe_presets={"first": first, "second": duplicate_name}
        )


def test_legacy_explore_review_state_defaults_to_no_named_presets():
    restored = ExploreReviewState.model_validate(
        {
            "schema_version": 1,
            "population_recipes": {},
            "viewed_rois": {},
        }
    )

    assert restored.recipe_presets == {}
    assert restored.active_recipe_id is None
    assert restored.population_qc_contour_width == 1
    assert restored.cell_properties_tracking_enabled
    assert restored.cell_properties_observations == []
