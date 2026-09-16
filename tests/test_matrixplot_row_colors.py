import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")
from SpatialBiologyToolkit.plotting import matrixplot_with_row_colors  # noqa: E402


def fixture_data():
    obj = ad.AnnData(
        np.array(
            [
                [0.1, 0.8, 0.2],
                [0.11, 0.79, 0.21],
                [0.9, 0.1, 0.7],
                [0.89, 0.11, 0.71],
                [0.3, 0.4, 0.9],
                [0.31, 0.41, 0.89],
            ]
        ),
        obs=pd.DataFrame(
            {"pop": pd.Categorical(["a", "a", "b", "b", "c", "c"])},
            index=list("abcdef"),
        ),
        var=pd.DataFrame(index=["x", "y", "z"]),
    )
    obj.uns["pop_colors"] = ["red", "blue", "green"]
    return obj


def test_dendrogram_and_extra_strip_alignment():
    obj = fixture_data()
    family = {"a": "black", "b": "grey", "c": "orange"}
    mp, fig = matrixplot_with_row_colors(
        obj,
        groupby_key="pop",
        dendrogram=True,
        additional_row_colors={"Family": family},
        row_color_labels=True,
        figsize=(7, 4),
    )
    axes = mp.get_axes()
    labels = [t.get_text() for t in axes["mainplot_ax"].get_yticklabels()]
    assert "group_extra_ax" in axes
    population_colors = dict(zip(obj.obs["pop"].cat.categories, obj.uns["pop_colors"]))
    for label, patch in zip(labels, axes["row_colors_population"].patches):
        assert np.allclose(patch.get_facecolor(), to_rgba(population_colors[label]))
    for label, patch in zip(labels, axes["row_colors_Family"].patches):
        assert np.allclose(patch.get_facecolor(), to_rgba(family[label]))
    plt.close(fig)


def test_original_call_and_both_secondary_modes():
    obj = fixture_data()
    for secondary in [
        None,
        {"a": "red", "b": "blue", "c": "green"},
        {"values": [0, 1], "label": "Score"},
    ]:
        mp, fig = matrixplot_with_row_colors(
            obj, groupby_key="pop", dendrogram=False, secondary_colorbar=secondary
        )
        assert mp.get_axes()["row_colors_population"].patches
        assert not any(key.startswith("row_colors_Family") for key in mp.get_axes())
        plt.close(fig)


def test_missing_extra_colors_fail():
    with pytest.raises(ValueError, match="missing"):
        matrixplot_with_row_colors(
            fixture_data(),
            groupby_key="pop",
            dendrogram=False,
            additional_row_colors={"Family": {"a": "red"}},
        )
    plt.close("all")
