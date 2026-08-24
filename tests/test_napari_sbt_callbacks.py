from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from SpatialBiologyToolkit.napari_sbt.app import (
    NapariSBTController,
    _population_observation_columns,
    _preferred_population_observation,
)
from SpatialBiologyToolkit.napari_sbt.explore import ExploreViewRecipe


class _ComboStub:
    def __init__(self):
        self.items: list[tuple[str, object]] = []

    def clear(self):
        self.items.clear()

    def addItem(self, text, data=None):
        self.items.append((str(text), data))

    def addItems(self, values):
        self.items.extend((str(value), None) for value in values)


def test_guard_discards_surplus_qt_signal_arguments_by_default():
    controller = object.__new__(NapariSBTController)
    calls: list[str] = []

    controller._guard(lambda: calls.append("called"))("signal payload")

    assert calls == ["called"]


def test_guard_can_forward_signal_arguments_explicitly():
    controller = object.__new__(NapariSBTController)
    calls: list[str] = []

    controller._guard(calls.append, pass_signal_args=True)("r1")

    assert calls == ["r1"]


def test_classification_label_layers_use_colormap_keyword():
    controller = object.__new__(NapariSBTController)
    controller.current_mask = np.zeros((2, 2), dtype=np.int32)
    controller.current_roi = "r1"
    controller.manifest = SimpleNamespace(
        classes=[SimpleNamespace(class_id="good", color="#1b7837")]
    )
    controller.labels = pd.DataFrame(
        columns=["ROI", "ObjectNumber", "class_id", "state"]
    )
    controller.scores = pd.DataFrame()
    layer_calls: list[dict] = []
    controller._class_colormap = lambda: "direct-label-colormap"
    controller._replace_layer = (
        lambda _name, _data, _layer_type, **kwargs: layer_calls.append(kwargs)
    )

    controller.refresh_classification_layers()

    assert len(layer_calls) == 2
    assert all(call["colormap"] == "direct-label-colormap" for call in layer_calls)
    assert all("color" not in call for call in layer_calls)


def test_replace_layer_defaults_new_layers_to_full_opacity():
    calls: list[dict] = []
    controller = object.__new__(NapariSBTController)
    controller.viewer = SimpleNamespace(
        layers={},
        add_image=lambda data, **kwargs: calls.append(
            {"data": data, **kwargs}
        ),
    )

    controller._replace_layer("marker", np.ones((2, 2)), "image")

    assert calls[0]["opacity"] == 1.0


def test_cached_scalar_recipe_channel_is_restored_with_additive_blending():
    controller = object.__new__(NapariSBTController)
    controller.explore_recipe = ExploreViewRecipe(
        image_mode="six_colour",
        image_channels=["CD3"],
    )
    controller.current_roi = "r1"
    controller.current_image_paths = {"CD3": Path("CD3.tiff")}
    controller.viewer = SimpleNamespace(layers={})
    controller.set_status = lambda _message: None
    controller._display_image_load_kwargs = lambda _channel: {}
    controller._image_source_identity = lambda _path: {"path": "CD3.tiff"}
    controller._recipe_display_settings = lambda *_args, **_kwargs: {"visible": True}
    controller._reuse_explore_layer = lambda *_args, **_kwargs: None
    restored_settings: dict[str, object] = {}

    def restore(_name, _descriptor, _layer_type, **settings):
        restored_settings.update(settings)
        return object()

    controller._restore_cached_explore_layer = restore

    assert controller._render_recipe_images() == 1
    assert restored_settings["blending"] == "additive"


def test_class_controls_allow_manifest_before_cohort_snapshot_is_loaded():
    controller = object.__new__(NapariSBTController)
    controller.manifest = SimpleNamespace(
        classes=[],
        experiment_mode="full",
        feature_trial=None,
    )
    controller.cohort = pd.DataFrame()
    controller.class_combo = _ComboStub()
    controller.probability_class_combo = _ComboStub()
    controller.queue_class_combo = _ComboStub()
    controller.queue_roi_combo = _ComboStub()
    controller._class_shortcuts = []
    controller.viewer = SimpleNamespace(bind_key=lambda *_args, **_kwargs: None)
    controller._refresh_class_tally = lambda: None
    controller._refresh_model_storage_label = lambda: None
    controller._refresh_queue_if_scored = lambda: None

    controller.refresh_class_controls()

    assert controller.queue_roi_combo.items == [
        ("All current experiment ROIs", None)
    ]


def test_population_selectors_exclude_identity_and_prefer_leiden_labels():
    obs = pd.DataFrame(
        {
            "ROI": pd.Categorical(["r1", "r2"]),
            "ObjectNumber": [1, 1],
            "SampleType": pd.Categorical(["A", "B"]),
            "leiden_1.0": pd.Categorical(["0", "1"]),
        }
    )

    candidates = _population_observation_columns(
        obs.columns,
        roi_obs="ROI",
        object_obs="ObjectNumber",
    )

    assert candidates == ["SampleType", "leiden_1.0"]
    assert _preferred_population_observation(
        obs,
        candidates,
        prefer_leiden=True,
    ) == "leiden_1.0"


def test_empty_normalization_editor_does_not_create_invalid_workspace_file(tmp_path):
    controller = object.__new__(NapariSBTController)
    controller._normalization_from_editor = lambda: {}
    controller.normalization_edit = SimpleNamespace(clear=lambda: None)

    result = controller._write_experiment_normalization(tmp_path)

    assert result is None
    assert not (tmp_path / "display" / "normalization.json").exists()
