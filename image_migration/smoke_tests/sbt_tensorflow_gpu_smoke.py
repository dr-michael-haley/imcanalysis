"""GPU acceptance test for the joint ``sbt-tensorflow`` environment."""

from __future__ import annotations

import os
import sys
from importlib.metadata import version

import numpy as np
import tensorflow as tf


def main() -> None:
    """Require a real GPU and execute a deterministic TensorFlow calculation."""

    observed_version = version("tensorflow")
    assert observed_version == "2.15.1", observed_version

    physical_gpus = tf.config.list_physical_devices("GPU")
    assert physical_gpus, "TensorFlow cannot see an allocated GPU"
    for device in physical_gpus:
        tf.config.experimental.set_memory_growth(device, True)

    # A missing or incompatible GPU must fail rather than silently using the CPU.
    tf.config.set_soft_device_placement(False)
    with tf.device("/GPU:0"):
        values = tf.reshape(tf.range(16, dtype=tf.float32), (4, 4))
        result = tf.linalg.matmul(values, tf.eye(4, dtype=tf.float32))
        total = tf.reduce_sum(result)

    np.testing.assert_allclose(result.numpy(), values.numpy(), rtol=0.0, atol=0.0)
    assert float(total.numpy()) == 120.0
    assert "GPU:0" in result.device.upper(), result.device

    build_info = tf.sysconfig.get_build_info()
    print(f"python={sys.version.split()[0]}")
    print(f"tensorflow={tf.__version__}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"physical_gpus={[device.name for device in physical_gpus]}")
    print(f"cuda_version={build_info.get('cuda_version', '<unknown>')}")
    print(f"cudnn_version={build_info.get('cudnn_version', '<unknown>')}")
    print(f"result_device={result.device}")
    print(f"matrix_sum={float(total.numpy())}")
    print("TENSORFLOW_GPU_SMOKE_PASS")


if __name__ == "__main__":
    main()
