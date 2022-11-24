import tensorflow as tf


class BertTrainer(tf.keras.Model):
    def __init__(self, model):
        super(BertTrainer, self).__init__()
        self.model = model
        self.acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def compile(self, optimizer, loss):
        super(BertTrainer, self).compile()
        self.optimizer = optimizer
        self.loss = loss

    def train_step(self, inputs):
        if len(inputs) == 3:
            x_masked_tokens, y_masked_tokens, masked_layer = inputs
        else:
            x_masked_tokens, y_masked_tokens = inputs
            masked_layer = None

        with tf.GradientTape() as tape:
            predictions = self.model(x_masked_tokens, training=True)
            loss = self.loss(y_masked_tokens, predictions)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.acc_metric.update_state(
            y_masked_tokens, predictions
        )

        # Return a dict mapping metric names to current value
        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.acc_metric.result(),
        }

    def test_step(self, inputs):
        x_masked_tokens, y_masked_tokens, masked_layer = inputs

        predictions = self.model(x_masked_tokens, training=False)
        loss = self.loss(x_masked_tokens, predictions)
        acc_metric = self.acc_metric.update_state(
            y_masked_tokens, predictions
        )
        return {"loss": loss, "accuracy": acc_metric.result()}

    # @property
    # def metrics(self):
    #     # `reset_states()` yourself at the time of your choosing.
    #     return [self.loss_tracker, self.acc_metric]

    def call(self, inputs):
        return self.model(inputs)
