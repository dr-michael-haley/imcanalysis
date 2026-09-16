import numpy as np
import pandas as pd
import pytest
from SpatialBiologyToolkit.diversity import (
    population_counts,
    population_diversity,
    diversity_from_counts,
)


def test_known_values_and_empty():
    result = diversity_from_counts(
        pd.DataFrame(
            [[5, 5], [10, 0], [0, 0], [1, 0]], index=["even", "single", "empty", "one"]
        )
    )
    assert result.loc["even", "simpson_dominance"] == 0.5
    assert result.loc["even", "simpson_diversity"] == 0.5
    assert result.loc["even", "shannon"] == 1
    assert result.loc["single", "inverse_simpson"] == 1
    assert np.isnan(result.loc["empty", "simpson_diversity"])
    assert result.loc["empty", "richness"] == 0
    unbiased = diversity_from_counts([[1, 1], [1, 0]], estimator="unbiased")
    assert unbiased.loc[0, "simpson_diversity"] == 1
    assert np.isnan(unbiased.loc[1, "simpson_diversity"])


def test_dataframe_anndata_parity_and_no_mutation():
    ad = pytest.importorskip("anndata")
    obs = pd.DataFrame(
        {
            "case": ["a", "a", "b", "c"],
            "roi": ["r1", "r1", "r2", "r3"],
            "pop": pd.Categorical(
                ["x", "y", "x", None], categories=["x", "y", "unused"]
            ),
        },
        index=list("abcd"),
    )
    before = obs.copy(deep=True)
    obj = ad.AnnData(obs=obs.copy())
    pd.testing.assert_frame_equal(
        population_counts(obj, "pop", "case"), population_counts(obs, "pop", "case")
    )
    result = population_diversity(obj, "pop", ["case", "roi"])
    assert result.loc[("a", "r1"), "simpson_diversity"] == 0.5
    assert np.isnan(result.loc[("c", "r3"), "simpson_diversity"])
    assert (
        population_counts(obs, "pop", "case", dropna=False).loc["c", "Unlabelled"] == 1
    )
    assert population_counts(obs, "pop").values.sum() == 3
    pd.testing.assert_frame_equal(obs, before)
    pd.testing.assert_frame_equal(obj.obs, before)


@pytest.mark.parametrize("values", [[-1, 2], [np.nan, 1], [np.inf, 1]])
def test_invalid_counts(values):
    with pytest.raises(ValueError):
        diversity_from_counts(values)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        diversity_from_counts([0.2, 0.8], estimator="unbiased")
    with pytest.raises(ValueError):
        diversity_from_counts([1, 2], base=1)
    with pytest.raises(ValueError):
        diversity_from_counts([1, 2], estimator="bad")
    with pytest.raises(ValueError):
        population_counts(pd.DataFrame({"case": [None], "pop": ["x"]}), "pop", "case")
    with pytest.raises(KeyError):
        population_counts(pd.DataFrame({"pop": ["x"]}), "pop", "absent")


def test_roi_average_is_not_pooled():
    obs = pd.DataFrame({"roi": ["a"] * 2 + ["b"] * 8, "pop": ["x", "y"] + ["x"] * 8})
    assert population_diversity(obs, "pop", "roi").simpson_diversity.mean() == 0.25
    assert population_diversity(obs, "pop").simpson_diversity.iloc[0] == pytest.approx(
        0.18
    )
