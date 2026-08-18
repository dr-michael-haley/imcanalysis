from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from SpatialBiologyToolkit.napari_sbt.models import DisplaySettings
from SpatialBiologyToolkit.napari_sbt.population_qc import (
    build_population_qc_recipe,
    inherit_setup_contrast_limits,
    rank_population_rois,
    top_population_markers,
)


def _adata() -> SimpleNamespace:
    return SimpleNamespace(
        obs=pd.DataFrame(
            {
                "ROI": ["A", "A", "B", "C", "C"],
                "leiden": pd.Categorical(["1", "0", "1", "0", "0"]),
            }
        ),
        var_names=pd.Index(["CD3", "CD68", "PanCK"]),
        X=np.asarray(
            [
                [8.0, 1.0, 0.0],
                [1.0, 2.0, 3.0],
                [6.0, 4.0, 0.0],
                [1.0, 1.0, 8.0],
                [0.0, 1.0, 7.0],
            ]
        ),
    )


def test_population_roi_ranking_supports_top_bottom_and_seeded_random():
    adata = _adata()

    top = rank_population_rois(
        adata,
        observation="leiden",
        population="1",
        roi_obs="ROI",
        eligible_rois=["A", "B", "C", "D"],
        ordering="top",
        limit=None,
    )
    bottom = rank_population_rois(
        adata,
        observation="leiden",
        population="1",
        roi_obs="ROI",
        eligible_rois=["A", "B", "C", "D"],
        ordering="bottom",
        limit=None,
    )
    random_one = rank_population_rois(
        adata,
        observation="leiden",
        population="1",
        roi_obs="ROI",
        eligible_rois=["A", "B", "C", "D"],
        ordering="random",
        limit=3,
        random_seed=17,
    )
    random_two = rank_population_rois(
        adata,
        observation="leiden",
        population="1",
        roi_obs="ROI",
        eligible_rois=["A", "B", "C", "D"],
        ordering="random",
        limit=3,
        random_seed=17,
    )

    assert top == [("A", 1), ("B", 1), ("C", 0), ("D", 0)]
    assert bottom == [("C", 0), ("D", 0), ("A", 1), ("B", 1)]
    assert random_one == random_two


def test_population_marker_suggestions_use_population_mean_expression():
    markers = top_population_markers(
        _adata(),
        observation="leiden",
        population="1",
        candidates=[("CD68 image", "CD68"), ("CD3 image", "CD3")],
        top_n=2,
    )

    assert markers == ["CD3 image", "CD68 image"]


def test_population_qc_recipe_persists_rgb_colours_and_ranges():
    recipe = build_population_qc_recipe(
        observation="leiden",
        population="1",
        channels=["CD3", "CD68", "PanCK"],
        contrast_limits=[(0.05, 0.8), (0.0, 0.6), (0.1, 1.0)],
    )

    assert recipe.image_mode == "six_colour"
    assert recipe.layer_colormaps == {
        "image::CD3": "red",
        "image::CD68": "green",
        "image::PanCK": "blue",
    }
    assert recipe.layer_contrast_limits["image::CD68"] == (0.0, 0.6)
    assert recipe.populations == ["1"]


def test_display_settings_require_normalized_ordered_contrast_limits():
    settings = DisplaySettings(default_contrast_limits=(0.1, 0.9))
    assert settings.default_contrast_limits == (0.1, 0.9)

    with pytest.raises(ValueError, match="lower < upper"):
        DisplaySettings(default_contrast_limits=(0.9, 0.1))


def test_setup_contrast_updates_only_an_unmodified_unsaved_population_range():
    assert inherit_setup_contrast_limits(
        (0.0, 1.0),
        (0.0, 1.0),
        (0.1, 0.8),
        has_saved_recipe=False,
    ) == (0.1, 0.8)
    assert inherit_setup_contrast_limits(
        (0.2, 0.7),
        (0.0, 1.0),
        (0.1, 0.8),
        has_saved_recipe=False,
    ) == (0.2, 0.7)
    assert inherit_setup_contrast_limits(
        (0.0, 1.0),
        (0.0, 1.0),
        (0.1, 0.8),
        has_saved_recipe=True,
    ) == (0.0, 1.0)
