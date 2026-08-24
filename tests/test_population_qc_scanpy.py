from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from SpatialBiologyToolkit.population_qc import (
    plot_population_scanpy_abundance,
    plot_population_scanpy_matrixplot,
    plot_population_scanpy_umap,
    temporary_numba_cache_dir,
)


def _adata() -> ad.AnnData:
    labels = pd.Categorical(["A", "A", "B", "B", "C", "C"])
    adata = ad.AnnData(
        X=np.asarray(
            [
                [0.1, 0.8],
                [0.2, 0.7],
                [0.3, 0.6],
                [0.4, 0.5],
                [0.5, 0.4],
                [0.6, 0.3],
            ],
            dtype=float,
        ),
        obs=pd.DataFrame(
            {
                "population": labels,
                "Patient": ["P1", "P1", "P2", "P2", "P3", "P3"],
                "ROI": ["R1", "R1", "R2", "R2", "R3", "R3"],
                "SampleType": ["clear", "clear", "tumour", "tumour", "tumour", "tumour"],
            },
            index=[f"cell-{index}" for index in range(6)],
        ),
        var=pd.DataFrame(index=["CD3", "CD20"]),
    )
    adata.obsm["X_umap"] = np.arange(12, dtype=float).reshape(6, 2)
    return adata


def test_temporary_numba_cache_dir_is_project_local_and_restores_environment(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.setenv("NUMBA_CACHE_DIR", "previous-cache")

    with temporary_numba_cache_dir(project_dir) as cache_dir:
        assert cache_dir.parent == project_dir
        assert cache_dir.is_dir()
        assert os.environ["NUMBA_CACHE_DIR"] == str(cache_dir)
        (cache_dir / "sentinel.txt").write_text("cache", encoding="utf-8")

    assert not cache_dir.exists()
    assert os.environ["NUMBA_CACHE_DIR"] == "previous-cache"


class _FakeMatrixPlot:
    def __init__(self) -> None:
        self.fig = plt.figure()
        self.ax_dict = {"mainplot_ax": self.fig.add_subplot(111)}

    def make_figure(self) -> None:
        return None


@pytest.fixture
def fake_scanpy(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, object]]]:
    calls: dict[str, list[dict[str, object]]] = {"umap": [], "matrixplot": [], "dendrogram": []}

    def umap(_adata: ad.AnnData, **kwargs: object) -> None:
        calls["umap"].append({"adata": _adata, **kwargs})
        plt.figure()

    def matrixplot(_adata: ad.AnnData, **kwargs: object) -> _FakeMatrixPlot:
        calls["matrixplot"].append({"adata": _adata, **kwargs})
        return _FakeMatrixPlot()

    def dendrogram(_adata: ad.AnnData, **kwargs: object) -> None:
        calls["dendrogram"].append({"adata": _adata, **kwargs})

    monkeypatch.setitem(
        sys.modules,
        "scanpy",
        SimpleNamespace(
            pl=SimpleNamespace(umap=umap, matrixplot=matrixplot),
            tl=SimpleNamespace(dendrogram=dendrogram),
        ),
    )
    return calls


def test_scanpy_umap_uses_native_renderer_and_does_not_mutate_source(
    fake_scanpy: dict[str, list[dict[str, object]]],
) -> None:
    adata = _adata()
    result = plot_population_scanpy_umap(
        adata,
        "population",
        population="A",
        competitors=["B"],
        max_cells=4,
        random_state=4,
    )

    assert len(fake_scanpy["umap"]) == 1
    call = fake_scanpy["umap"][0]
    assert call["color"] == "_sbt_population_qc_focus"
    assert set(result.data["role"]) <= {"background", "competitor: B", "target"}
    assert "_sbt_population_qc_focus" not in adata.obs
    plt.close(result.figure)


def test_scanpy_matrixplot_uses_requested_dendrogram_and_vmax(
    fake_scanpy: dict[str, list[dict[str, object]]],
) -> None:
    result = plot_population_scanpy_matrixplot(
        _adata(),
        "population",
        markers=["CD3", "CD20"],
        max_cells_per_population=1,
        vmax=0.6,
        random_state=9,
    )

    assert len(fake_scanpy["dendrogram"]) == 1
    assert len(fake_scanpy["matrixplot"]) == 1
    call = fake_scanpy["matrixplot"][0]
    assert call["dendrogram"] is True
    assert call["vmax"] == 0.6
    assert list(result.data.columns) == ["cells", "sampled_cells", "CD3", "CD20"]
    assert result.data["sampled_cells"].tolist() == [1, 1, 1]
    plt.close(result.figure)


def test_scanpy_abundance_uses_sample_level_fractions(
    fake_scanpy: dict[str, list[dict[str, object]]],
) -> None:
    result = plot_population_scanpy_abundance(
        _adata(),
        "population",
        case_key="Patient",
        group_key="SampleType",
        roi_key="ROI",
    )

    assert len(fake_scanpy["dendrogram"]) == 1
    call = fake_scanpy["matrixplot"][0]
    assert call["groupby"] == "SampleType"
    assert call["dendrogram"] is True
    assert set(result.display_data.index) == {"clear", "tumour"}
    assert {"sample_id", "Patient", "ROI", "SampleType", "A", "B", "C"}.issubset(
        result.data.columns
    )
    plt.close(result.figure)
