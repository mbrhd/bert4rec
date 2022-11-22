import numpy as np
import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
    """
    Class implementing a learnable positional embedding layer.

    Args:
        d_model (int): The dimensionality of the embedding.
        embedding (tf.keras.layers.Embedding): The embedding layer.
        pos_encoding (np.array): The positional encoding.
    """

    def __init__(
        self, d_model: int = 512, vocab_size: int = 40857, max_seq_len: int = 512
    ):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model,
            input_length=max_seq_len,
            mask_zero=True,
        )
        self.pos_encoding = tf.keras.layers.Embedding(
            input_dim=max_seq_len, output_dim=d_model, mask_zero=True
        )

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        embed = self.embedding(x)
        position = self.pos_encoding(tf.range(start=0, limit=length, delta=1))
        return embed + position
