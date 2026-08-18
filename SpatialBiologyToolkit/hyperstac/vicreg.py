"""Small TensorFlow VICReg training model used by the HyPERSTAC adaptation.

This implementation follows the variance-invariance-covariance objective used
by HyPERSTAC while avoiding a runtime dependency on the un-packaged upstream
repository helper classes.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf


class VICReg(tf.keras.Model):
    """Train one encoder/projector pair from two augmented image views."""

    def __init__(
        self,
        encoder: tf.keras.Model,
        projector: tf.keras.Model,
        *,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        variance_target: float = 1.0,
        epsilon: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.projector = projector
        self.invariance_weight = float(invariance_weight)
        self.variance_weight = float(variance_weight)
        self.covariance_weight = float(covariance_weight)
        self.variance_target = float(variance_target)
        self.epsilon = float(epsilon)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.invariance_tracker = tf.keras.metrics.Mean(name="invariance_loss")
        self.variance_tracker = tf.keras.metrics.Mean(name="variance_loss")
        self.covariance_tracker = tf.keras.metrics.Mean(name="covariance_loss")

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [
            self.loss_tracker,
            self.invariance_tracker,
            self.variance_tracker,
            self.covariance_tracker,
        ]

    def _project(self, image: tf.Tensor, *, training: bool) -> tf.Tensor:
        representation = self.encoder(image, training=training)
        return self.projector(representation, training=training)

    def _compute_vicreg_losses(
        self,
        first: tf.Tensor,
        second: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Calculate the VICReg objective without shadowing Keras ``_losses``."""

        invariance = tf.reduce_mean(tf.square(first - second))

        variance = tf.add_n(
            [
                tf.reduce_mean(
                    tf.nn.relu(
                        self.variance_target
                        - tf.sqrt(tf.math.reduce_variance(view, axis=0) + self.epsilon)
                    )
                )
                for view in (first, second)
            ]
        )

        covariance = tf.constant(0.0, dtype=first.dtype)
        for view in (first, second):
            centred = view - tf.reduce_mean(view, axis=0)
            batch_size = tf.cast(tf.shape(centred)[0], centred.dtype)
            covariance_matrix = tf.matmul(centred, centred, transpose_a=True) / batch_size
            off_diagonal = covariance_matrix - tf.linalg.diag(
                tf.linalg.diag_part(covariance_matrix)
            )
            covariance += tf.reduce_sum(tf.square(off_diagonal)) / tf.cast(
                tf.shape(centred)[1], centred.dtype
            )

        total = (
            self.invariance_weight * invariance
            + self.variance_weight * variance
            + self.covariance_weight * covariance
        )
        return total, invariance, variance, covariance

    def train_step(self, data: tuple[tf.Tensor, tf.Tensor]) -> dict[str, tf.Tensor]:
        first_view, second_view = data
        with tf.GradientTape() as tape:
            first = self._project(first_view, training=True)
            second = self._project(second_view, training=True)
            total, invariance, variance, covariance = self._compute_vicreg_losses(
                first,
                second,
            )

        gradients = tape.gradient(total, self.trainable_variables)
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, self.trainable_variables)
            if gradient is not None
        ]
        self.optimizer.apply_gradients(pairs)
        self.loss_tracker.update_state(total)
        self.invariance_tracker.update_state(invariance)
        self.variance_tracker.update_state(variance)
        self.covariance_tracker.update_state(covariance)
        return {metric.name: metric.result() for metric in self.metrics}
