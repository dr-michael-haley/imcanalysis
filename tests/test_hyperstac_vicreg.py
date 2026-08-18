from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import patch


class _FakeKerasModel:
    """Reproduce the Keras 2.15 attribute that collided with our helper."""

    def __init__(self, **kwargs):
        del kwargs
        self._losses = []


class _FakeMean:
    def __init__(self, *, name: str):
        self.name = name


class HyperstacVICRegCompatibilityTests(unittest.TestCase):
    def test_loss_helper_does_not_collide_with_keras_internal_losses_list(self):
        module_name = "SpatialBiologyToolkit.hyperstac.vicreg"
        previous_module = sys.modules.pop(module_name, None)
        fake_tensorflow = types.SimpleNamespace(
            keras=types.SimpleNamespace(
                Model=_FakeKerasModel,
                metrics=types.SimpleNamespace(Mean=_FakeMean),
            )
        )
        try:
            with patch.dict(sys.modules, {"tensorflow": fake_tensorflow}):
                module = importlib.import_module(module_name)
                model = module.VICReg(
                    encoder=object(),
                    projector=object(),
                )
                self.assertIsInstance(model._losses, list)
                self.assertTrue(callable(model._compute_vicreg_losses))
        finally:
            sys.modules.pop(module_name, None)
            if previous_module is not None:
                sys.modules[module_name] = previous_module


if __name__ == "__main__":
    unittest.main()
