import tensorflow as tf


class LearnedPositionalEncoding(tf.keras.layers.Layer):
    """Learned positional embedding added to patch token sequences."""

    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_embedding = tf.keras.layers.Embedding(
            input_dim=seq_len, output_dim=d_model
        )

    def call(self, x):
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        return x + self.pos_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return config


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for binary classification."""

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(
            1 - y_pred
        )
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1 - p_t, gamma) * bce)

    loss_fn.__name__ = "focal_loss"
    return loss_fn