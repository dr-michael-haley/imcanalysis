"""Guided startup and quick-open contracts on tiny, disposable datasets."""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import pytest
import tifffile

from SpatialBiologyToolkit.napari_sbt.setup import setup_checks, setup_is_ready
from SpatialBiologyToolkit.napari_sbt.validation import check_assets


def assets(tmp_path):
    masks = tmp_path / "masks"
    images = tmp_path / "images"
    masks.mkdir()
    (images / "r1").mkdir(parents=True)
    tifffile.imwrite(masks / "r1.tiff", np.array([[0, 1], [2, 2]], dtype=np.int32))
    tifffile.imwrite(images / "r1" / "CD3.tiff", np.ones((2, 2), dtype=np.uint16))
    cohort = pd.DataFrame(
        {"obs_name": ["a", "b"], "ROI": ["r1", "r1"], "ObjectNumber": [1, 2]}
    )
    return masks, images, cohort


@pytest.mark.parametrize(
    "mode,ready",
    [
        ("data_exploration", True),
        ("population_qc", True),
        ("classification", False),
        ("full_workspace", False),
    ],
)
def test_quick_open_is_scoped_to_workflow(tmp_path, mode, ready):
    masks, images, _ = assets(tmp_path)
    cell_file = tmp_path / "cells.h5ad"
    cell_file.touch()
    checks = setup_checks(
        workspace_name="Review",
        workspace_path=tmp_path / "review",
        workflow_mode=mode,
        anndata_path=cell_file,
        has_in_memory_anndata=False,
        masks_folder=masks,
        image_folders=[str(images)],
        extra_image_folders=[],
        roi_obs="ROI",
        object_id_obs="ObjectNumber",
        normalization_path=None,
        integrity_current=False,
        quick_open=True,
    )
    assert setup_is_ready(checks) is ready


def test_asset_check_reports_missing_ids_and_images(tmp_path):
    masks, images, cohort = assets(tmp_path)
    assert check_assets(cohort, masks, [images]).ok
    cohort.loc[1, "ObjectNumber"] = 99
    result = check_assets(cohort, masks, [])
    assert not result.ok
    assert any("missing from the mask" in issue for issue in result.issues)
    assert any("staining images" in issue for issue in result.issues)
    assert result.summary().startswith("Needs review")


def test_cancelled_check_never_claims_success(tmp_path):
    masks, images, cohort = assets(tmp_path)
    result = check_assets(cohort, masks, [images], cancelled=lambda: True)
    assert result.cancelled and not result.ok
    assert result.checked_rois == 0


def test_indexing_does_not_read_masks_or_claim_validation(tmp_path, monkeypatch):
    import SpatialBiologyToolkit.napari_sbt.validation as validation

    masks, images, cohort = assets(tmp_path)
    monkeypatch.setattr(
        validation, "load_mask", lambda *_: pytest.fail("Indexing read a mask")
    )
    result = check_assets(cohort, masks, [images], index_only=True)
    assert result.indexed_only and not result.ok
    assert "r1" in result.masks and "r1" in result.images


def test_mask_export_validates_before_writing(tmp_path):
    from SpatialBiologyToolkit.napari_sbt.exports import materialize_cohort_masks

    masks, _, cohort = assets(tmp_path)
    cohort.loc[1, "ObjectNumber"] = 99
    destination = tmp_path / "export"
    with pytest.raises(ValueError, match="missing"):
        materialize_cohort_masks({"r1": masks / "r1.tiff"}, cohort, destination)
    assert not destination.exists()


def test_welcome_flag_bypasses_launch_folder_discovery(monkeypatch):
    from SpatialBiologyToolkit.napari_sbt import __main__ as entry

    monkeypatch.setattr(
        entry,
        "_resolve_project_context",
        lambda *_: pytest.fail("Unexpected launch-folder discovery"),
    )
    assert entry._project_defaults(entry.build_parser().parse_args(["--welcome"])) == {}
    assert entry._project_defaults(
        entry.build_parser().parse_args(["--dataset", "study"])
    ) == {"project_root": Path("study")}


@pytest.mark.parametrize("options", [["--welcome"], ["--dataset", "study"]])
def test_sbt_cli_forwards_desktop_startup_options(monkeypatch, options):
    from types import SimpleNamespace
    from typer.testing import CliRunner
    import SpatialBiologyToolkit.cli.main as cli

    calls = []
    monkeypatch.setattr(
        cli,
        "_napari_gui_command",
        lambda *_: ["python", "-m", "SpatialBiologyToolkit.napari_sbt"],
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command)
        or SimpleNamespace(returncode=0),
    )
    result = CliRunner().invoke(cli.app, ["gui", "napari", *options])
    assert result.exit_code == 0, result.output
    assert calls[0][-len(options) :] == options


def test_feature_worker_rejects_missing_cell_ids(tmp_path):
    from SpatialBiologyToolkit.napari_sbt.worker import _roi_task

    masks, _, _ = assets(tmp_path)
    with pytest.raises(ValueError, match="eligible cell IDs are missing"):
        _roi_task({"roi": "r1", "mask_path": masks / "r1.tiff", "eligible_ids": [99]})


def test_training_does_not_silently_drop_confirmed_cells():
    from SpatialBiologyToolkit.napari_sbt.classifier import train_multiclass_classifier
    from SpatialBiologyToolkit.napari_sbt.labels import empty_labels, set_label

    labels = set_label(
        empty_labels(), roi="r1", object_number=2, class_id="a", state="confirmed"
    )
    features = pd.DataFrame({"ROI": ["r1"], "ObjectNumber": [1], "signal": [1.0]})
    with pytest.raises(ValueError, match="no feature row"):
        train_multiclass_classifier(features, labels, class_ids=["a", "b"])


def test_program_home_never_discovers_the_launch_folder(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    napari = pytest.importorskip("napari")
    import SpatialBiologyToolkit.napari_sbt.app as app
    from SpatialBiologyToolkit.napari_sbt import onboarding
    from qtpy.QtCore import QSettings

    monkeypatch.setattr(
        onboarding,
        "QSettings",
        lambda *_: QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat),
    )
    monkeypatch.setattr(
        app,
        "discover_dataset_assets",
        lambda *_: pytest.fail("Home scanned for datasets"),
    )
    monkeypatch.chdir(tmp_path)
    viewer = napari.Viewer(show=False)
    try:
        _, c, _ = app.launch(viewer=viewer, welcome=True)
        assert c.tabs.tabText(c.tabs.currentIndex()) == "Home"
        assert [
            c.tabs.tabText(i) for i in range(c.tabs.count()) if c.tabs.isTabVisible(i)
        ] == ["Home", "Workspace"]
        assert c.project_edit.text() == ""
        assert not (tmp_path / "napari_sbt").exists()
    finally:
        viewer.close()


def test_guided_quick_open_and_strict_switch(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    napari = pytest.importorskip("napari")
    ad = pytest.importorskip("anndata")
    from qtpy.QtCore import QSettings
    from SpatialBiologyToolkit.napari_sbt import onboarding
    from SpatialBiologyToolkit.napari_sbt.app import launch
    import SpatialBiologyToolkit.napari_sbt.validation as validation

    monkeypatch.setattr(
        onboarding,
        "QSettings",
        lambda *_: QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat),
    )
    masks, images, cohort = assets(tmp_path)
    data = ad.AnnData(
        np.zeros((2, 1)),
        obs=cohort.set_index("obs_name"),
        var=pd.DataFrame(index=["CD3"]),
    )
    viewer = napari.Viewer(show=False)
    try:
        _, c, _ = launch(
            viewer=viewer,
            project_root=tmp_path,
            anndata=data,
            masks_folder=masks,
            images_folders=[images],
        )
        flow = c.workspace_flow
        assert c.tabs.tabText(0) == "Home"
        assert c.tabs.currentWidget() is flow.root
        assert c.adata is data
        assert flow.pages.count() == 5
        c.workflow_combo.setCurrentIndex(c.workflow_combo.findData("data_exploration"))
        c.name_edit.setText("Quick review")
        monkeypatch.setattr(c.QMessageBox, "question", lambda *_: c.QMessageBox.Yes)
        monkeypatch.setattr(
            validation,
            "check_assets",
            lambda **_: pytest.fail("Quick opening performed a full scan"),
        )
        c.create_experiment()
        assert c.manifest is not None
        assert flow.step == 4
        assert c.current_mask is not None
        assert c.paths.manifest.exists()
        report = (c.paths.root / "inputs" / "asset_validation.json").read_text()
        assert '"not_checked"' in report
        assert '"complete"' not in report
        assert c.tabs.isTabVisible(c._workflow_tab_indices["explore"])
        c.tabs.setCurrentIndex(0)
        assert c.manifest.name == "Quick review"

        # A failed lazy read leaves the workspace usable and a persistent warning.
        tifffile.imwrite(masks / "r1.tiff", np.zeros((2, 2), dtype=np.int32))
        c.load_existing_experiment(c.paths.root)
        assert c.current_mask is None
        assert c.manifest.name == "Quick review"
        assert "Region could not open" in flow.validation_badge.text()
        tifffile.imwrite(masks / "r1.tiff", np.array([[0, 1], [2, 2]], dtype=np.int32))

        c.start_new_workspace(confirm=False)
        c.workflow_combo.setCurrentIndex(c.workflow_combo.findData("classification"))
        c.name_edit.setText("Classifier")
        with pytest.raises(ValueError, match="Check the dataset"):
            c.create_experiment()
        assert not flow.quick_open

        # Explicit checking runs off the GUI thread and unlocks strict creation.
        import time

        flow.show_step(2)
        flow.start_check()
        deadline = time.monotonic() + 10
        while flow.checking and time.monotonic() < deadline:
            c.QApplication.processEvents()
            time.sleep(0.01)
        c.QApplication.processEvents()
        assert not flow.checking
        assert c.integrity_is_current()
        assert c._validation_result.ok
        c.create_experiment()
        report = (c.paths.root / "inputs" / "asset_validation.json").read_text()
        assert '"complete"' in report
        assert "Previously checked" in c.integrity_status_label.text()
    finally:
        if "flow" in locals() and flow.worker is not None:
            flow.cancel_validation()
            flow.worker.wait(10000)
        viewer.close()
