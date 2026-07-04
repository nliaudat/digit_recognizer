# utils/optimizers.py
"""
utils/optimizers.py — Advanced optimizers and training step wrappers.
Includes Sharpness-Aware Minimization (SAM).
"""

import tensorflow as tf

class SAMModelWrapper(tf.keras.Model):
    """
    Wraps a Keras model to use Sharpness-Aware Minimization (SAM).
    SAM requires two forward-backward passes per step:
    1. Compute gradient of loss w.r.t weights -> step weights in gradient direction (e_w)
    2. Compute gradient of loss at new weights -> actual update
    3. Revert e_w
    """
    def __init__(self, base_model, rho=0.05, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.rho = rho

    def call(self, inputs, training=None, **kwargs):
        return self.base_model(inputs, training=training, **kwargs)

    def train_step(self, data):
        # Unpack the data
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.base_model(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
            
        # 1. First backward pass: Compute gradients
        trainable_vars = self.base_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Compute the scale for e_w
        grad_norm = tf.linalg.global_norm(gradients)
        scale = self.rho / (grad_norm + 1e-12)

        # 2. Step weights to the local maximum
        e_w_list = []
        for v, g in zip(trainable_vars, gradients):
            if g is not None:
                e_w = tf.math.multiply(g, scale)
                v.assign_add(e_w)
                e_w_list.append(e_w)
            else:
                e_w_list.append(None)

        # 3. Second forward-backward pass at local maximum
        with tf.GradientTape() as tape:
            y_pred = self.base_model(x, training=True)
            sam_loss = self.compute_loss(x, y, y_pred, sample_weight)

        sam_gradients = tape.gradient(sam_loss, trainable_vars)

        # 4. Revert weights back to original
        for v, e_w in zip(trainable_vars, e_w_list):
            if e_w is not None:
                v.assign_sub(e_w)

        # 5. Update weights using SAM gradients
        self.optimizer.apply_gradients(zip(sam_gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)
                
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return self.base_model.metrics

    # Expose necessary properties for Keras
    @property
    def output_shape(self):
        return self.base_model.output_shape

    def get_config(self):
        return self.base_model.get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)
