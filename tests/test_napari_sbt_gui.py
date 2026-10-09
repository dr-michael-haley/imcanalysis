from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
napari = pytest.importorskip("napari")
ad = pytest.importorskip("anndata")
pytest.importorskip("qtpy")

from SpatialBiologyToolkit.napari_sbt import launch_notebook
from SpatialBiologyToolkit.napari_sbt.app import _write_anndata_snapshot, launch
from SpatialBiologyToolkit.napari_sbt.cohort import resolve_cohort
from SpatialBiologyToolkit.napari_sbt.models import (
    ExperimentManifest,
    segmentation_qc_classes,
)
from SpatialBiologyToolkit.napari_sbt.storage import save_experiment


def _live_adata():
    obs = pd.DataFrame(
        {
            "ROI": pd.Categorical(["r1", "r1"]),
            "ObjectNumber": [1, 2],
            "population": pd.Categorical(["target", "other"]),
        },
        index=["cell-1", "cell-2"],
    )
    return ad.AnnData(
        np.zeros((2, 1), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["CD3"]),
    )


def test_launch_accepts_live_anndata_in_anndata_path_argument(tmp_path: Path):
    data = _live_adata()
    viewer = napari.Viewer(show=False)
    try:
        _, controller, _dock = launch(
            viewer=viewer,
            project_root=tmp_path,
            anndata_path=data,
            masks_folder=tmp_path / "masks",
        )
        assert controller.adata is data
        assert controller.anndata_edit.text() == ""
        assert (
            "In-memory AnnData (2 cells)"
            in controller.anndata_edit.placeholderText()
        )
        assert controller.obs_combo.findText("population") >= 0
        assert controller.curation_source_combo.findText("population") >= 0
        assert (
            controller.population_neighbor_source_combo.currentData()
            == "rebuild_from_rep"
        )
        assert controller.population_n_neighbors_spin.value() == 15
        assert controller.marker_overlay_list.item(0).text() == "CD3"
    finally:
        viewer.close()


def test_notebook_launcher_uses_live_anndata(tmp_path: Path):
    data = _live_adata()
    viewer = napari.Viewer(show=False)
    try:
        _, controller, _dock = launch_notebook(
            adata=data,
            viewer=viewer,
            project_root=tmp_path,
            masks_folder=tmp_path / "masks",
        )
        assert controller.adata is data
    finally:
        viewer.close()


def test_in_memory_anndata_snapshot_is_atomic_and_never_overwritten(tmp_path: Path):
    destination = tmp_path / "experiment" / "inputs" / "anndata.h5ad"
    written = _write_anndata_snapshot(_live_adata(), destination)
    assert written == destination.resolve()
    assert ad.read_h5ad(written).obs_names.tolist() == ["cell-1", "cell-2"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _write_anndata_snapshot(_live_adata(), destination)


def test_unified_dock_is_cohort_gated_and_rejects_context_clicks(tmp_path: Path):
    import tifffile

    obs = pd.DataFrame(
        {
            "ROI": pd.Categorical(["r1", "r1", "r2"]),
            "ObjectNumber": [1, 2, 1],
            "leiden": pd.Categorical(["target", "other", "other"]),
        },
        index=["a", "b", "c"],
    )
    data = ad.AnnData(np.zeros((3, 1), dtype=np.float32), obs=obs)
    adata_path = tmp_path / "cells.h5ad"
    data.write_h5ad(adata_path)
    masks = tmp_path / "masks"
    masks.mkdir()
    tifffile.imwrite(
        masks / "r1.tiff",
        np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int32),
    )
    tifffile.imwrite(
        masks / "r2.tiff",
        np.array([[1, 1], [0, 0]], dtype=np.int32),
    )
    preview = resolve_cohort(
        data,
        roi_obs="ROI",
        object_id_obs="ObjectNumber",
        mode="obs_values",
        obs_column="leiden",
        obs_values=["target"],
    )
    root = tmp_path / "experiment"
    (root / "cohort").mkdir(parents=True)
    preview.eligible_cells.to_parquet(
        root / "cohort" / "eligible_cells.parquet", index=False
    )
    manifest = ExperimentManifest(
        name="GUI cohort",
        anndata_path=str(adata_path),
        masks_folder=str(masks),
        cell_scope=preview.scope(
            mode="obs_values", obs_column="leiden", obs_values=["target"]
        ),
        classes=segmentation_qc_classes(),
    )
    save_experiment(manifest, root)

    viewer = napari.Viewer(show=False)
    try:
        _, controller, _dock = launch(viewer=viewer, experiment=root)
        assert [
            controller.tabs.tabText(index).split(" ", 1)[-1]
            for index in range(controller.tabs.count())
        ] == [
            "Home",
            "Workspace",
            "Feature Building",
            "Feature Refinement",
            "Explore",
            "Population QC",
            "Population naming",
            "Scanpy plotting",
            "Dataset Maintenance",
            "Classify",
            "Labeler",
            "Regions & Export",
            "Layers & Status",
        ]
        assert set(np.unique(viewer.layers["classification_cohort"].data)) == {0, 1}
        assert not viewer.layers["excluded_segmentation_context"].visible
        assert controller.roi_combo.count() == 1
        assert "1 eligible cells / 3 total cells" in controller.scope_label.text()
        assert "LIMITED CELL SCOPE" in controller.population_qc_scope_banner.text()
        assert "1 of 3 cells" in controller.population_qc_scope_banner.text()
        assert [
            controller.population_qc_population_combo.itemText(index)
            for index in range(controller.population_qc_population_combo.count())
        ] == ["target"]

        controller._on_cohort_click(
            viewer.layers["classification_cohort"],
            SimpleNamespace(type="mouse_press", position=(0, 0)),
        )
        assert controller.current_selected_object is None
        assert "outside this experiment" in controller.selected_cell_label.text()

        # Empty-cohort regions need known asset paths; navigation never scans
        # the whole mask directory simply to populate this selector.
        controller._mask_path_index["r2"] = masks / "r2.tiff"
        controller.show_empty_rois.setChecked(True)
        controller.refresh_rois()
        assert {controller.roi_combo.itemText(index) for index in range(2)} == {
            "r1",
            "r2",
        }

        controller.start_new_workspace(confirm=False)
        assert controller.scope_combo.currentData() == "all_cells"
        assert not controller.value_list.selectedItems()
        assert "SETUP MODE" in controller.population_qc_scope_banner.text()
    finally:
        viewer.close()


def test_classify_clicks_after_labeler_selection(tmp_path: Path, monkeypatch):
    import tifffile
    from napari.utils.interactions import mouse_press_callbacks

    from SpatialBiologyToolkit.napari_sbt.app import (
        LABELER_SELECTED_CELL_LAYER_NAME,
        SELECTED_CELL_LAYER_NAME,
    )

    data = _live_adata()
    adata_path = tmp_path / "cells.h5ad"
    data.write_h5ad(adata_path)
    masks = tmp_path / "masks"
    masks.mkdir()
    tifffile.imwrite(
        masks / "r1.tiff", np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int32)
    )
    preview = resolve_cohort(
        data, roi_obs="ROI", object_id_obs="ObjectNumber", mode="all_cells"
    )
    root = tmp_path / "experiment"
    (root / "cohort").mkdir(parents=True)
    preview.eligible_cells.to_parquet(
        root / "cohort" / "eligible_cells.parquet", index=False
    )
    save_experiment(
        ExperimentManifest(
            name="Click regression",
            anndata_path=str(adata_path),
            masks_folder=str(masks),
            cell_scope=preview.scope(
                mode="all_cells", obs_column=None, obs_values=[]
            ),
            classes=segmentation_qc_classes(),
        ),
        root,
    )

    viewer = napari.Viewer(show=False)
    try:
        _, controller, _dock = launch(viewer=viewer, experiment=root)

        def fail_on_dialog(_parent, _title, message):
            pytest.fail(message)

        monkeypatch.setattr(controller.QMessageBox, "critical", fail_on_dialog)
        # Locate the visible tabs independently of the controller's routing.
        classify_index = next(
            i for i in range(controller.tabs.count())
            if controller.tabs.tabText(i).endswith("Classify")
        )
        labeler_index = next(
            i for i in range(controller.tabs.count())
            if controller.tabs.tabText(i).endswith("Labeler")
        )
        assert controller.classify_tab_index == classify_index
        assert controller.labeler_tab_index == labeler_index
        for topic, title in (
            ("explore", "Explore"), ("population_qc", "Population QC"),
            ("scanpy_plotting", "Scanpy plotting"),
            ("dataset_maintenance", "Dataset Maintenance"),
        ):
            assert controller.tabs.tabText(
                getattr(controller, f"{topic}_tab_index")
            ).endswith(title)
        # Expose both tabs just as the advanced workflow does.
        for index in (classify_index, labeler_index):
            controller.tabs.setTabVisible(index, True)
        controller.tabs.setCurrentIndex(labeler_index)
        for button in controller.labeler_click_behavior_group.buttons():
            if button.property("napari_sbt_labeler_click_behavior") == "select":
                button.setChecked(True)
        cohort_layer = viewer.layers["classification_cohort"]
        viewer.layers.selection.active = cohort_layer
        event = SimpleNamespace(type="mouse_press", button=1, position=(0, 1))
        mouse_press_callbacks(viewer, event)
        assert controller.current_labeler_object == 1
        outline = viewer.layers[LABELER_SELECTED_CELL_LAYER_NAME]
        assert viewer.layers.selection.active is cohort_layer
        assert not outline.editable

        # Selecting the outline manually must not prevent Classify clicks.
        viewer.layers.selection.active = outline
        controller.tabs.setCurrentIndex(classify_index)
        assert LABELER_SELECTED_CELL_LAYER_NAME not in viewer.layers
        controller.refresh_labeler_layers()
        assert LABELER_SELECTED_CELL_LAYER_NAME not in viewer.layers
        viewer.layers.selection.active = cohort_layer
        for index, state in enumerate(("proposed", "confirmed")):
            controller.class_combo.setCurrentIndex(index)
            controller.click_behavior_radios[state].setChecked(True)
            mouse_press_callbacks(viewer, event)
            saved = pd.read_parquet(controller.paths.labels)
            assert len(saved) == 1
            assert saved.iloc[0]["state"] == state
            assert saved.iloc[0]["class_id"] == controller.selected_class_id()
            assert viewer.layers.selection.active is cohort_layer
            assert not viewer.layers[SELECTED_CELL_LAYER_NAME].editable
            assert controller.labeler_records.empty

        controller.tabs.setCurrentIndex(labeler_index)
        assert SELECTED_CELL_LAYER_NAME not in viewer.layers
        assert LABELER_SELECTED_CELL_LAYER_NAME in viewer.layers
        event.position = (1, 1)
        mouse_press_callbacks(viewer, event)
        assert controller.current_labeler_object == 2
        assert len(pd.read_parquet(controller.paths.labels)) == 1
        controller.assign_selected_labeler_cell()
        labeler_records = controller.labeler_records.copy(deep=True)
        assert len(labeler_records) == 1
        controller.labeler_enabled_check.setChecked(False)
        assert not controller.labeler_enabled
        assert "labeler_assignments" not in viewer.layers
        assert LABELER_SELECTED_CELL_LAYER_NAME not in viewer.layers
        controller.refresh_labeler_layers()
        mouse_press_callbacks(viewer, event)
        assert controller.current_labeler_object is None
        assert LABELER_SELECTED_CELL_LAYER_NAME not in viewer.layers
        pd.testing.assert_frame_equal(controller.labeler_records, labeler_records)
        controller.tabs.setCurrentIndex(classify_index)
        mouse_press_callbacks(viewer, event)
        assert len(pd.read_parquet(controller.paths.labels)) == 2
        pd.testing.assert_frame_equal(controller.labeler_records, labeler_records)
        controller.set_labeler_enabled(True)
        assert controller.labeler_enabled_check.isChecked()
        assert "labeler_assignments" in viewer.layers
        pd.testing.assert_frame_equal(controller.labeler_records, labeler_records)
    finally:
        viewer.close()
