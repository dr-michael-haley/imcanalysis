"""Small CPU-only runtime check for the joint SBT TensorFlow environment."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


with tf.device("/CPU:0"):
    left = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    right = tf.constant([[2.0], [1.0]])
    result = tf.linalg.matmul(left, right).numpy()

np.testing.assert_allclose(result, np.array([[4.0], [10.0]]))
print(f"TENSORFLOW_CPU_SMOKE_PASS {tf.__version__}")
