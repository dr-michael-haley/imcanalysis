from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad

from SpatialBiologyToolkit.annotation import (
    AnnotationSession,
    CohortFilter,
    annotate_embedding,
    apply_annotation_recipe,
)


def data():
    adata = AnnData(
        np.arange(12, dtype=float).reshape(6, 2),
        obs=pd.DataFrame(
            {"population": pd.Categorical(["A", "A", "B", "B", "C", "C"])},
            index=[f"c{i}" for i in range(6)],
        ),
        var=pd.DataFrame(index=["CD3", "CD68"]),
    )
    adata.obsm["X_umap"] = np.array(
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [np.nan, 5]]
    )
    return adata


REGION = [(-0.5, -0.5), (2.5, -0.5), (2.5, 2.5), (-0.5, 2.5)]


@pytest.mark.parametrize("sparse", [False, True])
def test_recipe_replays_shapes_filters_overlap_undo_and_parent_labels(tmp_path, sparse):
    from scipy.sparse import csr_matrix
    import json

    original = data()
    original.obs["score"] = [0, 1, 2, 3, 4, 5]
    if sparse:
        original.X = csr_matrix(original.X)
    replayed = original[[4, 2, 0, 3, 1, 5]].copy()
    session = AnnotationSession(
        original, source_obs="population", obs_names=["c0", "c1", "c2", "c3"]
    )
    session.set_filters([
        CohortFilter("obs", "population", values=("A", "B")),
        CohortFilter("obs", "score", minimum=1, maximum=3),
        CohortFilter("X", "CD3", minimum=2),
    ])
    session.select_polygon(REGION)
    session.assign("Split", inherit_parent=True)
    session.set_filters([])
    second = [(1.5, 1.5), (5, 1.5), (5, 5), (1.5, 5)]
    session.select_polygon(second)
    session.assign("Override")
    session.assign("Undone")
    session.undo()
    session.set_filters([CohortFilter("obs", "population", values=("C",))])
    session.select_polygon(second)  # Pending shape must not assign anything.
    path = session.save_recipe(tmp_path / "regions.json", "manual")
    recipe = json.loads(path.read_text(encoding="utf-8"))
    assert len(recipe["regions"]) == 2
    assert recipe["regions"][0]["vertices"] == [list(x) for x in REGION]
    assert recipe["selection_vertices"] == [list(x) for x in second]
    session.apply("manual")
    assert apply_annotation_recipe(replayed, path) == "manual"
    pd.testing.assert_series_equal(
        original.obs["manual"], replayed.obs["manual"].reindex(original.obs_names)
    )
    np.testing.assert_array_equal(
        original.uns["manual_colors"], replayed.uns["manual_colors"]
    )
    assert replayed.obs.loc["c1", "manual"] == "A / Split"
    assert replayed.obs.loc["c2", "manual"] == "Override"
    assert replayed.obs.loc["c4", "manual"] == "C"
    with pytest.raises(ValueError, match="already exists"):
        apply_annotation_recipe(replayed, path)
    apply_annotation_recipe(replayed, path, overwrite=True)
    apply_annotation_recipe(replayed, path, key_added="another")
    pd.testing.assert_series_equal(
        replayed.obs["manual"], replayed.obs["another"], check_names=False
    )


def test_recipe_no_matches_reset_and_missing_filters(tmp_path):
    original = data()
    original.obs["optional"] = [None, 1, 1, 1, 1, 1]
    session = AnnotationSession(original)
    session.set_filters([
        CohortFilter("obs", "optional", values=(np.nan, np.int64(1)))
    ])
    session.select_polygon(REGION)
    session.assign("Gate")
    saved = session.recipe()
    session.save_recipe(tmp_path / "missing.json")
    session.reset_labels()
    assert not session.recipe()["regions"]
    assert len(saved["regions"]) == 1  # Recipes are snapshots, not live references.
    shifted = original.copy()
    shifted.obsm["X_umap"] += 100
    apply_annotation_recipe(shifted, saved)
    assert set(shifted.obs["manual_population"]) == {"Unassigned"}
    # Invalid later regions must not leave a partially applied output column.
    saved["regions"].append({"filters": [], "vertices": [[0, 0]]})
    with pytest.raises(ValueError):
        apply_annotation_recipe(original, saved)
    assert "manual_population" not in original.obs


def test_popup_saves_recipe_for_headless_replay(qtapp, monkeypatch, tmp_path):
    import json
    from qtpy.QtWidgets import QFileDialog

    original = data()
    replayed = original.copy()
    window = annotate_embedding(original, source_obs="population", block=False)
    try:
        window._selected(REGION)
        window.label_edit.setText("Drawn")
        window.parent_prefix_check.setChecked(True)
        window.assign_button.click()
        path = tmp_path / "popup.json"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(path), ""))
        window.save_recipe_button.click()
        assert "manual_population" not in original.obs
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["display"]["inherit_parent"] is True
        assert saved["display"]["tool"] == "Lasso"
        assert saved["source_obs"] == "population"
        assert len(saved["regions"]) == 1
        window.apply()
        apply_annotation_recipe(replayed, path)
        pd.testing.assert_frame_equal(original.obs, replayed.obs)
        assert window.save_recipe(tmp_path / "programmatic.json").exists()
    finally:
        window.session.dirty = False
        window.close()


def test_recipe_replay_does_not_import_qt_and_cli_writes_copy(tmp_path):
    import subprocess
    import sys

    original = data()
    source = tmp_path / "input.h5ad"
    original.write_h5ad(source)
    session = AnnotationSession(original)
    session.select_polygon(REGION)
    session.assign("Gate")
    path = session.save_recipe(tmp_path / "recipe.json")
    output = tmp_path / "output.h5ad"
    script = """
import runpy, sys
runpy.run_module('SpatialBiologyToolkit.annotation', run_name='__main__')
assert not any(name in sys.modules for name in ('qtpy', 'PyQt5', 'PyQt6', 'PySide6'))
assert 'SpatialBiologyToolkit._annotation_popup' not in sys.modules
"""
    command = [
        sys.executable, "-c", script, "--anndata", str(source),
        "--recipe", str(path), "--output", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert read_h5ad(output).obs["manual_population"].tolist() == (
        ["Gate"] * 3 + ["Unassigned"] * 3
    )
    assert "manual_population" not in read_h5ad(source).obs
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "Refusing to overwrite" in result.stderr


def test_scoped_draft_overlap_undo_and_identity_alignment():
    adata = data()
    session = AnnotationSession(
        adata, obs_names=["c1", "c2", "c5"], source_obs="population"
    )
    assert session.select_polygon(REGION) == 2
    session.assign("Region A")
    session.assign("Region B")
    assert session.labels.loc["c1"] == "Region B"
    session.undo()
    assert session.labels.loc["c1"] == "Region A"
    assert "manual" not in adata.obs
    # Reorder all arrays just as a caller may do between opening and applying.
    adata._inplace_subset_obs([4, 3, 2, 1, 0, 5])
    session.apply("manual")
    assert adata.obs["manual"].to_dict() == {
        "c0": "A",
        "c1": "Region A",
        "c2": "Region A",
        "c3": "B",
        "c4": "C",
        "c5": "C",
    }
    assert isinstance(adata.obs["manual"].dtype, pd.CategoricalDtype)
    with pytest.raises(ValueError, match="already exists"):
        session.apply("manual")
    session.apply("manual", overwrite=True)
    adata._inplace_subset_obs([0, 1])
    with pytest.raises(ValueError, match="cell set changed"):
        session.apply("other")


def test_invalid_inputs_and_missing_coordinates():
    adata = data()
    with pytest.raises(ValueError, match="no embedding"):
        AnnotationSession(adata, basis="missing")
    with pytest.raises(ValueError, match="two different"):
        AnnotationSession(adata, components=(0, 0))
    with pytest.raises(ValueError, match="missing from"):
        AnnotationSession(adata, obs_names=["absent"])
    with pytest.raises(ValueError, match="No cells"):
        AnnotationSession(adata, obs_names=["c5"])
    session = AnnotationSession(adata)
    with pytest.raises(ValueError, match="three distinct"):
        session.select_polygon([(0, 0), (1, 1)])
    assert session.select_polygon(REGION) == 3
    with pytest.raises(ValueError, match="category label"):
        session.assign(" ")
    adata.obs_names = ["duplicate"] * 6
    with pytest.raises(ValueError, match="unique obs_names"):
        AnnotationSession(adata)


@pytest.fixture
def qtapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application
    application.processEvents()


def test_popup_full_selection_apply_and_save_round_trip(qtapp, monkeypatch, tmp_path):
    from qtpy.QtWidgets import QFileDialog
    from SpatialBiologyToolkit._annotation_popup import _WINDOWS

    adata = data()
    calls = []
    dialog = annotate_embedding(
        adata, point_limit=1, color="CD3", block=False, on_apply=calls.append
    )
    try:
        assert len(dialog.display_positions) == 1
        dialog._selected(REGION)
        assert dialog.session.selected.sum() == 3
        dialog.label_edit.setText("Selected")
        dialog.assign_button.click()
        assert "manual_population" not in adata.obs
        dialog.apply_button.click()
        assert calls == ["manual_population"]
        assert (
            adata.obs["manual_population"].tolist()
            == ["Selected"] * 3 + ["Unassigned"] * 3
        )
        assert len(adata.uns["manual_population_colors"]) == 2
        path = tmp_path / "annotated.h5ad"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(path), ""))
        dialog.save_button.click()
        restored = read_h5ad(path)
        pd.testing.assert_frame_equal(adata.obs, restored.obs)
        np.testing.assert_array_equal(
            adata.uns["manual_population_colors"],
            restored.uns["manual_population_colors"],
        )
        dialog.save_button.click()
        assert "Refusing to overwrite" in dialog.status_label.text()
    finally:
        dialog.session.dirty = False
        dialog.close()
    assert dialog not in _WINDOWS
    assert not dialog.isVisible()


def test_popup_polygon_undo_and_layer_colouring(qtapp, monkeypatch):
    from matplotlib.widgets import PolygonSelector
    from qtpy.QtWidgets import QMessageBox

    adata = data()
    adata.layers["scaled"] = adata.X * 2
    dialog = annotate_embedding(adata, color="CD3", layer="scaled", block=False)
    try:
        np.testing.assert_array_equal(
            dialog.points.get_array(), adata.layers["scaled"][:5, 0]
        )
        dialog.tool_combo.setCurrentText("Polygon")
        assert isinstance(dialog.selector, PolygonSelector)
        dialog._selected(REGION)
        dialog.label_edit.setText("A")
        dialog.assign_button.click()
        dialog.undo_button.click()
        assert set(dialog.session.labels) == {"Unassigned"}
        monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.Yes)
        dialog.source_combo.setCurrentIndex(dialog.source_combo.findData("population"))
        assert dialog.session.labels.loc["c4"] == "C"
        assert dialog.session.dirty
    finally:
        dialog.session.dirty = False
        dialog.close()


def test_napari_opens_full_scope_and_refreshes_labels(monkeypatch):
    from SpatialBiologyToolkit import annotation
    from SpatialBiologyToolkit.napari_sbt.app import NapariSBTController
    from SpatialBiologyToolkit.napari_sbt.scanpy_plotting import ScanpyPlotRequest

    controller = object.__new__(NapariSBTController)
    controller.adata = data()
    controller.root = object()
    request = ScanpyPlotRequest(
        groupby="population", embedding_key="X_umap", matrix_source="layer::scaled"
    )
    controller.scanpy_plot_windows = {
        "plot": {
            "adata": controller.adata,
            "request": request,
            "artifact": SimpleNamespace(
                annotation_obs_names=pd.Index(["c0", "c1", "c2"])
            ),
        }
    }
    calls = []
    controller._populate_anndata_selectors = lambda **kw: calls.append(kw)
    controller.refresh_scanpy_plotting_choices = lambda **kw: calls.append(kw)
    controller.set_status = lambda message: None
    captured = {}
    monkeypatch.setattr(
        annotation, "annotate_embedding", lambda adata, **kw: captured.update(kw)
    )
    controller.annotate_scanpy_plot("plot")
    assert captured["layer"] == "scaled"
    assert captured["source_obs"] == "population"
    assert list(captured["obs_names"]) == ["c0", "c1", "c2"]
    captured["on_apply"]("manual")
    assert calls[-1] == {"preferred_groupby": "manual"}
    controller.adata = data()
    with pytest.raises(ValueError, match="loaded AnnData changed"):
        captured["before_apply"]()


def test_napari_popup_applies_to_live_observation_selectors(qtapp, tmp_path):
    napari = pytest.importorskip("napari")
    from matplotlib.figure import Figure
    from SpatialBiologyToolkit.napari_sbt import launch_notebook
    from SpatialBiologyToolkit.napari_sbt.scanpy_plotting import (
        ScanpyPlotArtifact,
        ScanpyPlotRequest,
    )

    adata = data()
    adata.obs["ROI"] = pd.Categorical(["r1"] * 6)
    adata.obs["ObjectNumber"] = np.arange(1, 7)
    viewer = napari.Viewer(show=False)
    popup = None
    controller = None
    try:
        _, controller, _ = launch_notebook(
            adata=adata, viewer=viewer, project_root=tmp_path
        )
        figure = Figure()
        figure.add_subplot(111).scatter([0, 1], [0, 1])
        artifact = ScanpyPlotArtifact(
            figure, "Embedding", pd.DataFrame(), 6, "Six cells", adata.obs_names
        )
        request = ScanpyPlotRequest(groupby="population", embedding_key="X_umap")
        controller._show_scanpy_plot_artifact(artifact, request)
        popup = controller.annotate_scanpy_plot(
            next(iter(controller.scanpy_plot_windows))
        )
        popup._selected(REGION)
        popup.label_edit.setText("Manual group")
        popup.assign_button.click()
        popup.apply_button.click()
        assert "manual_population" in adata.obs
        assert controller.overlay_obs_combo.findText("manual_population") >= 0
        assert (
            controller.scanpy_plotting_panel.groupby_combo.currentText()
            == "manual_population"
        )
        assert "Applied to adata.obs" in popup.status_label.text()
        assert adata.obs.loc["c4", "manual_population"] == "C"
    finally:
        if popup is not None:
            popup.session.dirty = False
            popup.close()
        if controller is not None:
            from qtpy.QtCore import QCoreApplication, QEvent

            controller.close_all_scanpy_plot_windows()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        viewer.close()


@pytest.mark.parametrize("sparse", [False, True])
def test_cohort_filters_combine_obs_and_x_and_keep_parent_labels(sparse):
    from scipy.sparse import csr_matrix

    adata = data()
    adata.obs["sample"] = ["one", "two", "two", "two", "two", "two"]
    adata.obs["score"] = pd.array([1, 2, 3, 4, 5, None], dtype="Float64")
    if sparse:
        adata.X = csr_matrix(adata.X)
    session = AnnotationSession(adata, source_obs="population")
    rules = [
        CohortFilter("obs", "population", values=("A", "B")),
        CohortFilter("obs", "sample", values=("two",)),
        CohortFilter("obs", "score", minimum=2, maximum=3),
        CohortFilter("X", "CD3", minimum=4),
    ]
    # Filtering still aligns X by cell identity after a row reorder.
    adata._inplace_subset_obs([4, 3, 2, 1, 0, 5])
    session.set_filters(rules)
    assert session.obs_names[session.eligible].tolist() == ["c2"]
    assert session.select_polygon(REGION) == 1
    session.assign("Activated", inherit_parent=True)
    session.set_filters([CohortFilter("obs", "population", values=("A",))])
    assert not session.selected.any()
    assert session.labels.loc["c2"] == "B / Activated"
    assert len(session.history) == 1
    session.select_polygon(REGION)
    session.assign("Resting", inherit_parent=True)
    session.apply("split")
    assert adata.obs["split"].to_dict() == {
        "c0": "A / Resting",
        "c1": "A / Resting",
        "c2": "B / Activated",
        "c3": "B",
        "c4": "C",
        "c5": "C",
    }
    # Undo works across cohort changes without losing earlier splits.
    session.undo()
    assert session.labels.loc["c1"] == "A"
    assert session.labels.loc["c2"] == "B / Activated"


def test_filter_empty_cohort_invalid_ranges_and_missing_values():
    adata = data()
    adata.obs["optional"] = pd.Categorical([None, "x", "x", "x", "x", "x"])
    session = AnnotationSession(adata, obs_names=["c0", "c1", "c2"])
    session.set_filters([CohortFilter("obs", "optional", values=(None,))])
    assert session.obs_names[session.eligible].tolist() == ["c0"]
    session.set_filters([CohortFilter("X", "CD3", minimum=100)])
    assert not session.eligible.any()
    assert session.select_polygon(REGION) == 0
    with pytest.raises(ValueError, match="Draw around"):
        session.assign("empty")
    session.set_filters([])
    assert session.eligible.sum() == 3  # the original scope still applies
    for rule in [
        CohortFilter("X", "CD3", minimum=9, maximum=2),
        CohortFilter("X", "CD3", minimum=np.nan),
        CohortFilter("obs", "optional", minimum=1),
        CohortFilter("obs", "optional", values=()),
    ]:
        with pytest.raises(ValueError):
            session.set_filters([rule])
        assert session.eligible.sum() == 3


def test_popup_filters_grey_context_and_parent_prefix(qtapp):
    from qtpy.QtCore import Qt

    adata = data()
    # X filters must use X even if the colour display uses a different layer.
    adata.layers["scaled"] = adata.X * 100
    dialog = annotate_embedding(
        adata,
        source_obs="population",
        color="CD3",
        layer="scaled",
        block=False,
    )
    try:
        dialog.filter_field_combo.setCurrentText("population")
        for i in range(dialog.filter_values_list.count()):
            item = dialog.filter_values_list.item(i)
            if item.text() == "B":
                item.setCheckState(Qt.Checked)
        dialog.add_filter_button.click()
        assert dialog.session.obs_names[dialog.session.eligible].tolist() == [
            "c2",
            "c3",
        ]
        np.testing.assert_array_equal(
            dialog.background.get_offsets(), adata.obsm["X_umap"][[0, 1, 4]]
        )
        np.testing.assert_array_equal(dialog.points.get_array(), [400, 600])
        dialog.filter_kind_combo.setCurrentText("X range")
        dialog.filter_field_combo.setCurrentText("CD3")
        dialog.filter_min_edit.setText("5")
        dialog.add_filter_button.click()
        assert dialog.session.obs_names[dialog.session.eligible].tolist() == ["c3"]
        dialog._selected([(-1, -1), (5, -1), (5, 5), (-1, 5)])
        assert dialog.session.selected.sum() == 1
        dialog.parent_prefix_check.setChecked(True)
        dialog.label_edit.setText("High")
        dialog.assign_button.click()
        dialog.apply_button.click()
        assert adata.obs["manual_population"].tolist() == [
            "A",
            "A",
            "B",
            "B / High",
            "C",
            "C",
        ]
        dialog.clear_filters_button.click()
        assert dialog.session.eligible.sum() == 5
        assert dialog.session.labels.loc["c3"] == "B / High"
        dialog.filter_min_edit.setText("999")
        dialog.add_filter_button.click()
        assert len(dialog.display_positions) == 0
        assert len(dialog.background_positions) == 5
        assert "No cells match" in dialog.status_label.text()
        assert not dialog.assign_button.isEnabled()
    finally:
        dialog.session.dirty = False
        dialog.close()


def test_small_cohort_is_shown_and_parent_colours_are_stable(qtapp):
    adata = data()
    adata.uns["population_colors"] = ["#ff0000", "#00ff00", "#0000ff"]
    dialog = annotate_embedding(
        adata,
        source_obs="population",
        point_limit=2,
        color="population",
        block=False,
        filters=[CohortFilter("obs", "population", values=("C",))],
    )
    try:
        assert dialog.display_positions.tolist() == [4]
        assert len(dialog.background_positions) == 1
        np.testing.assert_allclose(dialog.points.get_facecolors(), [[0, 0, 1, 1]])
        assert not dialog.session.eligible[dialog.background_positions].any()
        dialog.color_combo.setCurrentIndex(0)
        np.testing.assert_allclose(dialog.points.get_facecolors(), [[0, 0, 1, 1]])
    finally:
        dialog.close()


def test_parent_prefix_uses_original_parent_and_keeps_unassigned_colours():
    session = AnnotationSession(data(), source_obs="population")
    parent_palette = session.observation_palette("population")
    assert session.palette()["B"] == parent_palette["B"]
    session.select_polygon(REGION)
    session.assign("First", inherit_parent=True)
    session.assign("Second", inherit_parent=True)
    assert session.labels.tolist() == [
        "A / Second",
        "A / Second",
        "B / Second",
        "B",
        "C",
        "C",
    ]
    assert session.palette()["B"] == parent_palette["B"]
