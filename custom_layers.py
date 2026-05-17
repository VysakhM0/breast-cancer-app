"""
Custom Keras layers and loss functions for the Breast Cancer Classifier.

Contains:
    - LearnedPositionalEncoding: Learnable positional embeddings for Vision Transformer
    - focal_loss: Focal loss function for handling class imbalance
"""

import tensorflow as tf
from tensorflow import keras


class LearnedPositionalEncoding(keras.layers.Layer):
    """
    Learned positional encoding layer for Vision Transformer architectures.

    Adds learnable position embeddings to input token sequences, allowing
    the model to encode spatial information about patch positions.

    Args:
        sequence_length (int): Maximum number of positions (patches + CLS token).
        embedding_dim (int): Dimensionality of the embedding space.
    """

    def __init__(self, sequence_length, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(self.sequence_length, self.embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.position_embeddings[:seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "embedding_dim": self.embedding_dim,
            }
        )
        return config


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for addressing class imbalance in binary classification.

    Reduces the loss contribution from easy examples and focuses training
    on hard-to-classify samples. Particularly useful for medical imaging
    where positive cases may be rare.

    Args:
        gamma (float): Focusing parameter. Higher values increase focus on
                       hard examples. Default: 2.0
        alpha (float): Balancing factor for the positive class. Default: 0.25

    Returns:
        Callable loss function compatible with Keras model.compile().
    """

    @tf.function
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Compute focal weight
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        focal_weight = alpha_t * tf.pow(1.0 - pt, gamma)

        # Binary cross-entropy
        bce = -(
            y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )

        loss = focal_weight * bce
        return tf.reduce_mean(loss)

    focal_loss_fn.__name__ = "focal_loss_fn"
    return focal_loss_fn
