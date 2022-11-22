from collections import Counter
from typing import List

import tensorflow as tf


def get_activation(activation: str):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
        activation: String name of the activation function.

    Returns:
        A Python function corresponding to the activation function. If
        `activation` is None, empty, or "linear", this will return None.
        If `activation` is not a string, it will return `activation`.

    Raises:
        ValueError: The `activation` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not activation:
        return None

    act = activation.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.keras.activations.relu
    elif act == "gelu":
        return tf.keras.activations.gelu
    elif act == "tanh":
        return tf.keras.activations.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


class FreqVocab(object):
    """Count the vocabulary and popularity of each item."""

    def __init__(self, user_index, movieItems: List[int]):
        self.counter = Counter(movieItems)
        self.user_set = set(["user_" + str(user_index)])

    def update_vocab(self, user_index, movieItems):
        self.counter.update(movieItems)
        self.user_set.add("user_" + str(user_index))

    def get_top_tokens(self, top_k=100):
        """get the top k tokens with the highest frequency.

        Args:
            top_k (int, optional): the top k tokens. Defaults to 100.

        Returns:
            List[int]: the top k tokens.
        """
        return [item[0] for item in self.counter.most_common(top_k)]
