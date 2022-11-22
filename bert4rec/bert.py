from typing import Optional

import tensorflow as tf

from .pos_embedding import PositionalEmbedding
from .transformer import TransformerEncoderLayer
from .utils import get_activation


class BertModel(tf.keras.Model):
    def __init__(
        self,
        num_layers: int = 12,
        num_heads: int = 12,
        d_model: int = 128,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        activation: str = "gelu",
        feed_forward_size: Optional[int] = None,
        vocab_size: int = 40857,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.activation = activation
        self.feed_forward_size = feed_forward_size
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.embeddings = PositionalEmbedding(
            d_model=self.d_model,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
        )
        self.embeddings_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps
        )
        self.enc_layers = [
            TransformerEncoderLayer(
                num_heads=self.num_heads,
                d_model=self.d_model,
                dropout=self.dropout,
                layer_norm_eps=self.layer_norm_eps,
                activation=self.activation,
                feed_forward_size=self.feed_forward_size,
            )
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout)
        self.output1 = tf.keras.layers.Dense(
            self.d_model, activation=get_activation(self.activation), use_bias=True
        )
        self.output2 = tf.keras.layers.Dense(
            self.vocab_size, activation="softmax", use_bias=True
        )

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        inputs = tf.keras.layers.Input((self.max_seq_len), dtype=tf.int64)
        x = self.embeddings(inputs)  # Shape `(batch_size, seq_len, d_model)`.
        x = self.embeddings_layer_norm(x)

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        x = self.output1(x)
        x = self.output2(x)

        return x  # Shape `(batch_size, seq_len, vocab_size)`.
