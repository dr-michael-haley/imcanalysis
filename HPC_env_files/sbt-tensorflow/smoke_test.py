"""Small CPU-only runtime check for the joint SBT TensorFlow environment."""

from __future__ import annotations

import os

# This is deliberately a CPU-only login-node check. TensorFlow initializes all
# visible devices when its eager context is created, even inside tf.device().
# Hide scheduler-inaccessible GPU devices before TensorFlow is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from IMC_Denoise.checkpoints import validate_weights_name


with tf.device("/CPU:0"):
    left = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    right = tf.constant([[2.0], [1.0]])
    result = tf.linalg.matmul(left, right).numpy()

np.testing.assert_allclose(result, np.array([[4.0], [10.0]]))
assert (
    validate_weights_name("smoke.weights.h5", loading=False)
    == "smoke.weights.h5"
)
print(f"TENSORFLOW_CPU_SMOKE_PASS {tf.__version__}")
