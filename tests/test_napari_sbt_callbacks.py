from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from SpatialBiologyToolkit.napari_sbt.app import NapariSBTController


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
