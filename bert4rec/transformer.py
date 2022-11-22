from typing import Optional

import tensorflow as tf

from .feedforward import FeedForward


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads: int = 12,
        d_model: int = 512,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        activation: str = "gelu",
        feed_forward_size: Optional[int] = None,
    ):
        """Create transformer encoder layer.

        Args:
            hidden_size (int, optional): hidden size of the transformer encoder layer. Defaults to 64.
            num_heads (int, optional): number of heads in the multi head attention. Defaults to 12.
            d_model (int): The dimensionality of the embedding. Defaults to 512.
            dropout (float, optional): dropout rate. Defaults to 0.1.
            layer_norm_eps (float, optional): epsilon for layer normalization. Defaults to 1e-12.
            activation (str, optional): activation function. Defaults to 'gelu'.
            feed_forward_size (Optional[int], optional): hidden size of the feed forward layer. Defaults to None.
        """
        super().__init__()
        self.feed_forward_size = feed_forward_size
        self.dropout = dropout
        key_value_dim = d_model // num_heads
        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_value_dim,
            dropout=dropout,
        )
        self.add = tf.keras.layers.Add()
        self.self_attention_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps
        )
        self.feed_forward_layer = FeedForward(
            d_model=d_model,
            feed_forward_size=self.feed_forward_size,
            activation_string=activation,
            dropout_rate=dropout,
        )
        self.output_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps
        )

    def call(
        self, x: tf.Tensor, attention_mask: Optional[tf.Tensor] = None, training=False
    ):
        """Forward pass of the transformer encoder layer.

        Args:
            x (Tensor): embedding tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask (Optional[Tensor], optional): attention mask of shape (batch_size, seq_len, value_dim). Defaults to None.
            training (bool, optional): checks if training. Defaults to False.

        Returns:
            Tensor: output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # self attention
        self_attention_output = self.self_attention(
            query=x, value=x, key=x, attention_mask=attention_mask, training=training
        )
        self_attention_output = tf.keras.layers.Dropout(self.dropout)(
            self_attention_output, training=training
        )
        self_attention_output = self.add([x, self_attention_output])
        self_attention_output = self.self_attention_norm(self_attention_output)

        pff_output = self.feed_forward_layer(self_attention_output)
        pff_output = tf.keras.layers.Dropout(self.dropout)(
            pff_output, training=training
        )
        pff_output = self.add([self_attention_output, pff_output])
        y = self.output_layer_norm(pff_output)

        return y
