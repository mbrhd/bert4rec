import tensorflow as tf

# def masked_loss(label, pred):
#     mask = label != 0
#     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#         from_logits=True, reduction="none"
#     )
#     loss = loss_object(label, pred)

#     mask = tf.cast(mask, dtype=loss.dtype)
#     loss *= mask

#     loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
#     return loss


# def masked_accuracy(label, pred):
#     pred = tf.argmax(pred, axis=2)
#     label = tf.cast(label, pred.dtype)
#     match = label == pred

#     mask = label != 0

#     match = match & mask

#     match = tf.cast(match, dtype=tf.float32)
#     mask = tf.cast(mask, dtype=tf.float32)
#     return tf.reduce_sum(match) / tf.reduce_sum(mask)


# class CustomFit(tf.keras.Model):
#     def __init__(self, model):
#         super(CustomFit, self).__init__()
#         self.model = model
#         self.acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

#     def compile(self, optimizer, loss):
#         super(CustomFit, self).compile()
#         self.optimizer = optimizer
#         self.loss = loss

#     def train_step(self, data):
#         # Unpack the data
#         x_masked_tokens, y_masked_tokens, masked_layer = data

#         with tf.GradientTape() as tape:
#             # Caclulate predictions
#             y_pred = self.model(x_masked_tokens, training=True)

#             # Loss
#             loss = self.loss(y_pred, y_masked_tokens, sample_weight=masked_layer)

#         # Gradients
#         training_vars = self.trainable_variables
#         gradients = tape.gradient(loss, training_vars)

#         # Step with optimizer
#         self.optimizer.apply_gradients(zip(gradients, training_vars))
#         self.acc_metric.update_state(y, y_pred)

#         return {"loss": loss, "accuracy": self.acc_metric.result()}

#     def test_step(self, data):
#         # Unpack the data
#         x, y = data

#         # Compute predictions
#         y_pred = self.model(x, training=False)

#         # Updates the metrics tracking the loss
#         loss = self.loss(y, y_pred)

#         # Update the metrics.
#         self.acc_metric.update_state(y, y_pred)
#         return {"loss": loss, "accuracy": self.acc_metric.result()}

class BertTrainer(tf.keras.Model):
    def __init__(self, model):
        super(BertTrainer, self).__init__()
        self.model = model

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
            loss = self.loss(
                y_masked_tokens, predictions, sample_weight=masked_layer
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss, masked_layer=masked_layer)

        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_metric.result()}

    def test_step(self, inputs):
        if len(inputs) == 3:
            x_masked_tokens, y_masked_tokens, masked_layer = inputs
        else:
            x_masked_tokens, y_masked_tokens = inputs
            masked_layer = None

        predictions = self.model(x_masked_tokens, training=False)
        loss = self.loss(x_masked_tokens, predictions, sample_weight=masked_layer)
        acc_metric = self.acc_metric(y_masked_tokens, predictions, sample_weight=masked_layer)
        return {"loss": loss.result(), "accuracy": acc_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.acc_metric]

    def call(self, inputs):
        return self.model(inputs)

