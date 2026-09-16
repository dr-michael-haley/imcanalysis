import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SpatialBiologyToolkit.plotting import plot_stacked_graphs


def test_missing_continuous_and_case_order():
    counts = pd.DataFrame({"A": [30, 20], "B": [70, 80]}, index=["case2", "case1"])
    missing = pd.DataFrame({"Age": [np.nan, np.nan]}, index=counts.index)
    fig = plot_stacked_graphs(
        [counts, missing],
        [{"A": "red", "B": "blue"}, "viridis"],
        ["stacked_bar", "bar"],
        create_legends=[True, False],
        bar_colorbars=False,
        show_case_labels=True,
    )
    assert len(fig.axes) == 2
    assert [t.get_text() for t in fig.axes[-1].get_xticklabels()] == ["case2", "case1"]
    plt.close(fig)


def test_default_colorbar_retained():
    frame = pd.DataFrame({"value": [1, 2]}, index=["a", "b"])
    fig = plot_stacked_graphs([frame], ["viridis"], ["bar"], create_legends=False)
    assert len(fig.axes) == 2
    plt.close(fig)
