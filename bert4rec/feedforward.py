import tensorflow as tf

from typing import Optional
from .utils import get_activation


class FeedForward(tf.keras.layers.Layer):
    """Feed forward layer for transformer encoder.

    Args:
        d_model (int): The dimensionality of the embedding.
        feed_forward_size (Optional[int], optional): hidden size of the feed forward layer. Defaults to None.
        activation_string (str, optional): activation function. Defaults to 'gelu'.
        dropout_rate (float, optional): dropout rate. Defaults to 0.1.
    """

    def __init__(
        self,
        d_model: int,
        feed_forward_size: Optional[int] = None,
        activation_string: str = "gelu",
        dropout_rate=0.1,
    ):
        super().__init__()
        dff = feed_forward_size if feed_forward_size else 4 * d_model
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    dff, activation=get_activation(activation_string)
                ),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x
