from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch
import weakref

import numpy as np
import pytest


@pytest.fixture
def permutation():
    """Exercise the real permutation code without requiring a GPU environment."""
    name = "SpatialBiologyToolkit.hyperstac._permutation_memory_test"
    source = Path(__file__).resolve().parents[1] / "SpatialBiologyToolkit/hyperstac/permutation.py"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    fake_model_module = types.ModuleType("SpatialBiologyToolkit.hyperstac.model")
    fake_model_module.build_encoder = None
    fake_model_module.patch_obs_index = None
    with patch.dict(sys.modules, {
        name: module,
        "tensorflow": types.ModuleType("tensorflow"),
        "SpatialBiologyToolkit.hyperstac.model": fake_model_module,
    }):
        spec.loader.exec_module(module)
        yield module


def embed(batch):
    # Spatial weights make these embeddings sensitive to pixel shuffling.
    weights = np.arange(1, batch.shape[1] * batch.shape[2] + 1, dtype=np.float32)
    return (batch * weights.reshape(1, batch.shape[1], batch.shape[2], 1)).sum(axis=(1, 2))


class InferenceModel:
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, batch, *, training):
        assert training is False
        self.batch_sizes.append(len(batch))
        result = embed(batch)
        return types.SimpleNamespace(numpy=lambda: result)

    def predict(self, *args, **kwargs):
        raise AssertionError("Permutation must not construct a Keras prediction pipeline")


KINDS = [
    "zero_channel", "zero_all_channels", "shuffle_channel",
    "shuffle_all_channels_independent", "shuffle_all_channels_shared",
]


def condition_for(module, kind):
    return module.PerturbationCondition(
        condition_id=kind, perturbation_type=kind, channel="marker", channel_index=1,
        replicate=1, shuffle_scope="test", preserves_channel_histogram=False,
        preserves_cross_channel_colocalization=False, description="Test condition",
    )


def patch_data(count):
    batch = np.random.default_rng(13).integers(0, 8, size=(count, 4, 4, 3)).astype(np.float32)
    batch[0] = 0
    return batch


@pytest.mark.parametrize("count,expected_sizes", [(7, [7]), (32, [32]), (65, [32, 32, 1])])
def test_direct_inference_preserves_order_and_bounds_minibatches(permutation, count, expected_sizes):
    batch = patch_data(count)
    model = InferenceModel()
    actual = permutation.predict_patch_batch(model, batch)
    np.testing.assert_array_equal(actual, embed(batch))
    assert actual.dtype == np.float32
    assert model.batch_sizes == expected_sizes


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shuffle_pixels", ["all", "nonzero"])
def test_in_place_perturbations_match_copying_and_preserve_default_api(permutation, kind, shuffle_pixels):
    original = patch_data(5)
    batch = original.copy()
    condition = condition_for(permutation, kind)
    copied = permutation.perturb_batch(batch, condition, np.random.default_rng(42), shuffle_pixels)
    np.testing.assert_array_equal(batch, original)
    assert not np.shares_memory(copied, batch)
    in_place = permutation.perturb_batch(
        batch, condition, np.random.default_rng(42), shuffle_pixels, copy=False,
    )
    assert in_place is batch
    np.testing.assert_array_equal(in_place, copied)


@pytest.fixture
def patches(tmp_path):
    batch = patch_data(67)
    paths = []
    for index, value in enumerate(batch):
        path = tmp_path / f"patch_{index}.npy"
        np.save(path, value)
        paths.append(str(path))
    return paths, batch


def track_loading_buffers(monkeypatch, module):
    original_load = module.load_patch_batch
    buffers = []

    def load(*args):
        # The previous buffer must be gone before the next one is allocated.
        assert all(ref() is None for ref in buffers)
        result = original_load(*args)
        buffers.append(weakref.ref(result))
        return result

    monkeypatch.setattr(module, "load_patch_batch", load)
    return buffers


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shuffle_pixels", ["all", "nonzero"])
def test_condition_matches_seeded_copying_reference(permutation, patches, monkeypatch, kind, shuffle_pixels):
    paths, images = patches
    condition = condition_for(permutation, kind)
    originals = embed(images)
    seed, condition_index, batch_size = 71, 3, 40
    rng = np.random.default_rng(seed + condition_index + 1)
    expected_parts = []
    for start, end, _ in permutation.iter_batches(paths, batch_size):
        perturbed = permutation.perturb_batch(images[start:end], condition, rng, shuffle_pixels)
        expected_parts.append(permutation.cosine_metrics(originals[start:end], embed(perturbed)))
    expected = [np.concatenate(parts) for parts in zip(*expected_parts)]

    buffers = track_loading_buffers(monkeypatch, permutation)
    original_perturb = permutation.perturb_batch

    def perturb(batch, *args, **kwargs):
        assert kwargs.get("copy") is False
        assert buffers[-1]() is batch
        return original_perturb(batch, *args, **kwargs)

    monkeypatch.setattr(permutation, "perturb_batch", perturb)
    model = InferenceModel()
    actual = permutation.run_condition(
        model, condition, condition_index, paths, originals, 4, 3, batch_size, seed, shuffle_pixels,
    )
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, rtol=0, atol=0, equal_nan=True)
    assert model.batch_sizes == [32, 8, 27]
    assert all(ref() is None for ref in buffers)
    np.testing.assert_array_equal(originals, embed(images))
    np.testing.assert_array_equal(np.stack([np.load(path) for path in paths]), images)


def test_recomputed_embeddings_release_images_between_batches(permutation, patches, monkeypatch):
    paths, images = patches
    buffers = track_loading_buffers(monkeypatch, permutation)
    model = InferenceModel()
    actual = permutation.predict_original_embeddings(model, paths, 4, 3, 40)
    np.testing.assert_array_equal(actual, embed(images))
    assert model.batch_sizes == [32, 8, 27]
    assert all(ref() is None for ref in buffers)


def test_tensorflow_inference_matches_predict_with_batchnorm(permutation):
    # Optional integration check when run in sbt-tensorflow. The fixture's stub
    # is replaced temporarily so the real TensorFlow installation can be used.
    with patch.dict(sys.modules):
        sys.modules.pop("tensorflow", None)
        tf = pytest.importorskip("tensorflow")
        inputs = tf.keras.Input(shape=(4, 4, 3))
        values = tf.keras.layers.BatchNormalization()(inputs)
        outputs = tf.keras.layers.GlobalAveragePooling2D()(values)
        model = tf.keras.Model(inputs, outputs)
        batch = patch_data(67)
        model(batch[:8], training=True)
        before = [weight.numpy().copy() for weight in model.non_trainable_weights]
        expected = model.predict(batch, batch_size=32, verbose=0)
        actual = permutation.predict_patch_batch(model, batch)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        for previous, weight in zip(before, model.non_trainable_weights):
            np.testing.assert_array_equal(previous, weight.numpy())
