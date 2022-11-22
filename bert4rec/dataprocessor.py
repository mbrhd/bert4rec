import logging
import pickle
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from bert4rec.utils import FreqVocab

warnings.simplefilter(action="ignore")
logger = logging.getLogger("deep learning daily main function")


class DataProcessor(object):
    """Data processor for BERT4Rec.

    Convert raw data to tf.data.Dataset and create training and evaluation datasets.
    """

    def __init__(
        self,
        max_seq_length: int = 200,
        mask_prob: float = 0.15,
        vocab_size: int = 40857,
    ):
        self.max_seq_length = max_seq_length
        self.mask_prob = mask_prob
        self.vocab_size = vocab_size

    def __parse_tf_records(self, element):
        """Parse a single record which is expected to be a TensorFlow example.

        Args:
            element: a single record which can be parsed as a string containing a serialized example.

        Returns:
            Parsed tf.train.Example
        """
        schema = {
            "userIndex": tf.io.FixedLenFeature([], tf.int64),
            "movieIndices": tf.io.RaggedFeature(tf.int64, row_splits_dtype=tf.int64),
            "timestamps": tf.io.RaggedFeature(tf.int64, row_splits_dtype=tf.int64),
        }
        return tf.io.parse_single_example(element, schema)

    def __read_tfrecord_data(self, input_data_path):
        """Reads data from a TFRecord file."""
        raw_dataset = tf.data.TFRecordDataset(input_data_path)
        raw_dataset = raw_dataset.map(self.__parse_tf_records)
        return raw_dataset

    def __create_int_features(self, values):
        """Creates a tf.train.Feature with int64_list from a python list."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

    def __create_tf_records(self, features):
        """Creates a tf.train.Example from features."""
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        return tf_example

    def __masking(
        self,
        input_id: np.array,
        mask_prob: float,
        mask_value: int,
        mask_last=False,
        vocab_size=40857,
    ):
        """Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        Identify `mask_prob` fraction of inputs to be corrupted. 80% of those are [MASK] replaced, 10% are
        corrupted with an alternate word, 10% remain the same.

        Args:
            input_id: list of token ids to be masked.
            mask_prob: probability of masking each token. A fraction to corrupt (usually 15%)
            mask_value: id of the token MASK.

        Returns:
            x_masked_tokens: list of masked tokens.
            y_tokens : list of original input tokens.
            mask_layer: list of 0s and 1s. 1s indicate tokens that are masked.
        """
        x_masked_tokens = input_id.copy()
        y_tokens = input_id.copy()
        mask_layer = np.ones_like(input_id)

        if mask_last:
            x_masked_tokens[-1] = mask_value
            mask_layer[-1] = 0
            return x_masked_tokens, y_tokens, mask_layer

        masked_indices = np.random.binomial(size=len(mask_layer), n=1, p=mask_prob)

        # Of the masked items, mask 80% of them with [MASK]
        indices_replaced = np.random.binomial(size=len(mask_layer), n=1, p=0.8)
        indices_replaced = indices_replaced & masked_indices
        x_masked_tokens[indices_replaced == 1] = mask_value

        indices_random = np.random.binomial(size=len(mask_layer), n=1, p=0.5)
        # Replace 10% of them with random words, rest preserved for auto-encoding
        indices_random = indices_random & masked_indices & ~indices_replaced
        random_words = np.random.randint(low=0, high=vocab_size, size=len(mask_layer))
        x_masked_tokens[indices_random == 1] = random_words[indices_random == 1]

        masked_positions = np.where(masked_indices == 1)[0]
        x_masked_tokens[masked_positions] = mask_value
        mask_layer = masked_indices

        return x_masked_tokens, y_tokens, mask_layer

    def __padding(
        self,
        x_masked_tokens: np.array,
        y_tokens: np.array,
        mask_layer: np.array,
        max_seq_length: int,
    ):
        """Padding or truncating list to `max_seq_length` items."""
        x_masked_tokens = np.pad(
            x_masked_tokens,
            (0, max_seq_length - len(x_masked_tokens)),
            "constant",
            constant_values=0,
        )
        y_tokens = np.pad(
            y_tokens, (0, max_seq_length - len(y_tokens)), "constant", constant_values=0
        )
        mask_layer = np.pad(
            mask_layer,
            (0, max_seq_length - len(mask_layer)),
            "constant",
            constant_values=1,
        )

        return x_masked_tokens, y_tokens, mask_layer

    def make_train_test_data(
        self,
        input_data_path: str,
        output_data_path: str,
        is_training: bool = True,
    ):
        """Create `TrainingTFrecord` from raw data."""
        mask_value = 40857
        raw_dataset = self.__read_tfrecord_data(input_data_path)
        vocab = FreqVocab("mask", [mask_value])
        if not is_training:
            self.max_seq_length = self.max_seq_length + 1

        logger.info("Creating data")

        with tf.io.TFRecordWriter(output_data_path) as writer:
            for raw_record in tqdm(raw_dataset):
                user_index = raw_record["userIndex"]
                movie_indices = raw_record["movieIndices"]
                input_tokens = movie_indices.numpy()

                if is_training:
                    if len(input_tokens) > self.max_seq_length:
                        input_tokens = input_tokens[-self.max_seq_length :]
                    (x_masked_tokens, y_tokens, mask_layer) = self.__masking(
                        input_tokens,
                        self.mask_prob,
                        mask_value,
                        mask_last=False,
                        vocab_size=self.vocab_size,
                    )

                    if len(x_masked_tokens) < self.max_seq_length:
                        x_masked_tokens, y_tokens, mask_layer = self.__padding(
                            x_masked_tokens, y_tokens, mask_layer, self.max_seq_length
                        )
                    features = OrderedDict()
                    features["info"] = self.__create_int_features([user_index])
                    features["x_masked_tokens"] = self.__create_int_features(
                        x_masked_tokens
                    )
                    features["y_tokens"] = self.__create_int_features(y_tokens)
                    features["mask_layer"] = self.__create_int_features(mask_layer)

                    tf_example = self.__create_tf_records(features)
                    writer.write(tf_example.SerializeToString())
                else:
                    if len(input_tokens) > self.max_seq_length:
                        input_tokens = input_tokens[-self.max_seq_length :]

                    vocab.update_vocab(user_index, input_tokens)

                    (x_masked_tokens, y_tokens, mask_layer) = self.__masking(
                        input_tokens,
                        self.mask_prob,
                        mask_value,
                        mask_last=True,
                        vocab_size=self.vocab_size,
                    )
                    if len(x_masked_tokens) < self.max_seq_length:
                        x_masked_tokens, y_tokens, mask_layer = self.__padding(
                            x_masked_tokens, y_tokens, mask_layer, self.max_seq_length
                        )

                    features = OrderedDict()
                    features["info"] = self.__create_int_features([user_index])
                    features["x_masked_tokens"] = self.__create_int_features(
                        x_masked_tokens
                    )
                    features["y_tokens"] = self.__create_int_features(y_tokens)
                    features["mask_layer"] = self.__create_int_features(mask_layer)

                    tf_example = self.__create_tf_records(features)
                    writer.write(tf_example.SerializeToString())

        logger.info(f"Saved successfully to {output_data_path}")
        logger.info("Saving vocab")

        if not is_training:
            vocab_file_path = (
                output_data_path.replace("test.tfrecords", "") + "vocab.pkl"
            )
            with open(vocab_file_path, "wb") as f:
                pickle.dump(vocab, f)

            logger.info(f"Saved successfully to {vocab_file_path}")


class DataLoader(object):
    def __init__(
        self,
        input_file_path,
        is_training: bool = True,
        batch_size: int = 32,
        max_seq_length: int = 200,
    ):
        self.input_file_path = input_file_path
        is_training = is_training
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

    def build_input_function(self, num_cpu_threads=4):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""

        def input_fn(self):
            """The actual input function."""

            name_to_features = {
                "info": tf.io.FixedLenFeature([1], tf.int64),  # [user]
                "x_masked_tokens": tf.io.FixedLenFeature(
                    [self.max_seq_length], tf.int64
                ),
                "y_tokens": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
                "mask_layer": tf.io.FixedLenFeature([], tf.int64),
            }

            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            if self.is_training:
                dataset = tf.data.TFRecordDataset(self.input_file_path)
                dataset = dataset.repeat()
                dataset = dataset.shuffle(buffer_size=100)
            else:
                dataset = tf.data.TFRecordDataset(self.input_file_path)

            dataset = dataset.map(
                lambda record: self._decode_record(record, name_to_features),
                num_parallel_calls=num_cpu_threads,
            )
            dataset = dataset.batch(batch_size=self.batch_size)
            return dataset

        return input_fn


if __name__ == "__main__":
    data = DataProcessor()
    data.make_train_test_data(
        "data/interim/raw_no_duplicates.tfrecords",
        "data/processed/training.tfrecords",
        is_training=True,
    )
    data.make_train_test_data(
        "data/interim/raw_no_duplicates.tfrecords",
        "data/processed/test.tfrecords",
        is_training=False,
    )

    # data_loader = DataLoader("../data/interim/training.tfrecords")
    # data_loader.build_input_function()
