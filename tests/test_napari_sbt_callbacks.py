from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from SpatialBiologyToolkit._napari_imc_normalization import (
    prepare_normalization_parameters,
)
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


def test_workspace_normalization_preserves_structured_bounds(tmp_path):
    controller = object.__new__(NapariSBTController)
    parameters = prepare_normalization_parameters(
        {"CD3": {"vmax": 10, "lower_threshold": 0.8}}
    )
    controller._normalization_from_editor = lambda: parameters
    controller.normalization_edit = SimpleNamespace(setText=lambda _value: None)

    result = controller._write_experiment_normalization(tmp_path)

    payload = json.loads(Path(result).read_text(encoding="utf-8"))
    assert payload == {
        "normalization_dict": {
            "CD3": {"vmax": 10.0, "lower_threshold": 0.8}
        }
    }


def test_display_image_kwargs_include_marker_lower_threshold():
    controller = object.__new__(NapariSBTController)
    controller.display_normalization = prepare_normalization_parameters(
        {"CD3": {"vmax": 10, "lower_threshold": 0.8}}
    )
    controller.current_image_paths = {}
    controller._display_image_settings = lambda: SimpleNamespace(
        fallback_quantile=0.999,
        minimum_pixel_counts=0.1,
    )

    kwargs = controller._display_image_load_kwargs("CD3")

    assert kwargs["normalization_value"] == 10.0
    assert kwargs["normalization_lower_threshold"] == 0.8
