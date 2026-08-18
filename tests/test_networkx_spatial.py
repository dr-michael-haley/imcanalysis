from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pandas as pd
import pytest


@pytest.fixture
def networkx_spatial_module(monkeypatch: pytest.MonkeyPatch):
    module_name = "SpatialBiologyToolkit.scripts.networkx_spatial"
    previous_module = sys.modules.pop(module_name, None)
    monkeypatch.setitem(sys.modules, "anndata", ModuleType("anndata"))
    monkeypatch.setitem(sys.modules, "squidpy", ModuleType("squidpy"))
    module = importlib.import_module(module_name)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module


def test_ungrouped_population_barplot_resolves_default_errorbar(
    networkx_spatial_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    def capture_barplot(**kwargs):
        captured.update(kwargs)
        return kwargs["ax"]

    monkeypatch.setattr(networkx_spatial_module.sns, "barplot", capture_barplot)
    output_path = tmp_path / "ungrouped.png"

    networkx_spatial_module._plot_all_populations_no_group(
        pd.DataFrame(
            {
                "population": ["A", "A", "B", "B"],
                "observed": [0.1, 0.2, 0.3, 0.4],
            }
        ),
        pop_order=["A", "B"],
        pop_palette={"A": "#111111", "B": "#222222"},
        value_col="observed",
        ylabel="Observed",
        title="Ungrouped",
        save_path=output_path,
        fixed_figsize=None,
        base_figsize=(4.0, 3.0),
        width_scale=0.45,
        dpi=72,
        add_points=False,
        plot_kind="barplot",
    )

    assert captured["errorbar"] == "se"
    assert output_path.is_file()


def test_grouped_population_barplot_forwards_errorbar_override(
    networkx_spatial_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    def capture_barplot(**kwargs):
        captured.update(kwargs)
        return kwargs["ax"]

    monkeypatch.setattr(networkx_spatial_module.sns, "barplot", capture_barplot)

    networkx_spatial_module._plot_all_populations_by_group(
        pd.DataFrame(
            {
                "population": ["A", "A"],
                "group": ["control", "treated"],
                "observed": [0.1, 0.2],
            }
        ),
        pop_order=["A"],
        group_col="group",
        group_order=["control", "treated"],
        group_palette={"control": "#111111", "treated": "#222222"},
        value_col="observed",
        ylabel="Observed",
        title="Grouped",
        save_path=tmp_path / "grouped.png",
        fixed_figsize=None,
        base_figsize=(4.0, 3.0),
        width_scale=0.45,
        dpi=72,
        add_points=False,
        plot_kind="barplot",
        errorbar=("ci", 95),
    )

    assert captured["errorbar"] == ("ci", 95)
