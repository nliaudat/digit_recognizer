"""
Shared ConvNeXt building blocks used by V33 and V34 super student models.

ConvNeXt design:
  - LayerNorm + GELU + large 7x7 depthwise kernels
  - Inverted bottleneck (expand ratio = 4)
  - DropPath (stochastic depth) for regularisation
  - LayerScale (learnable per-channel scaling)

These blocks are pure-convolutional (2022-2025 SOTA) and designed for
PC/GPU training — not intended for IoT deployment.
"""

import tensorflow as tf
from typing import Optional


class DropPath(tf.keras.layers.Layer):
    """
    DropPath (Stochastic Depth) — drops entire residual paths per sample.
    
    Unlike element-wise Dropout which zeros individual activations, DropPath
    randomly drops the *entire* residual branch for a given sample with
    probability ``drop_prob``, and scales survivors by ``1 / (1 - drop_prob)``
    to maintain expected output magnitude.  This is the correct implementation
    of stochastic depth as used in ConvNeXt, ResNeXt, etc.
    
    Uses ``tf.keras.backend.in_train_phase`` to handle symbolic ``training``
    tensors — a Python ``if not training:`` check is unreliable because TF
    symbolic tensors are always truthy and would cause DropPath to be applied
    during inference, severely degrading accuracy.
    
    Reference:
        "Deep Networks with Stochastic Depth" (Huang et al., 2016)
        https://arxiv.org/abs/1603.09382
    """
    def __init__(self, drop_prob: float, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, inputs, training=None):
        if self.drop_prob == 0.0:
            return inputs

        def dropped():
            keep_prob = 1.0 - self.drop_prob
            # Use tf.stack for graph-mode safety — mixing Python scalars
            # with tf.shape in a tuple can be fragile inside tf.function.
            shape = tf.stack(
                [tf.shape(inputs)[0]] + [1] * (len(inputs.shape) - 1)
            )
            random_tensor = keep_prob + tf.random.uniform(
                shape, dtype=inputs.dtype
            )
            binary_tensor = tf.floor(random_tensor)
            return (inputs / tf.cast(keep_prob, inputs.dtype)) * binary_tensor

        # Use tf.cond instead of legacy tf.keras.backend.in_train_phase for
        # Keras 3 forward-compatibility (in_train_phase is being phased out).
        # Also explicitly restore static shape info that can be lost inside
        # the cond branches — needed to avoid compilation failures downstream.
        if training is None:
            training = False
        output = tf.cond(
            tf.cast(training, tf.bool), dropped, lambda: inputs
        )
        output.set_shape(inputs.shape)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config


def layer_norm_2d(x: tf.Tensor, epsilon: float = 1e-6, name: str = "ln") -> tf.Tensor:
    """LayerNorm for 2D feature maps (applied per-channel)."""
    return tf.keras.layers.LayerNormalization(
        epsilon=epsilon, name=name
    )(x)


def convnext_block(
    x: tf.Tensor,
    dim: int,
    drop_path_rate: float = 0.0,
    layer_scale_init: float = 1e-6,
    name: str = "convnext_block",
) -> tf.Tensor:
    """
    ConvNeXt block:
      1. Depthwise 7x7 conv (large receptive field)
      2. LayerNorm
      3. Dense (expand x4) + GELU
      4. Dense (project back to dim)
      5. LayerScale (learnable per-channel scaling)
      6. DropPath (stochastic depth)
      7. Residual connection
    """
    shortcut = x

    # 1. Depthwise 7x7 conv
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=7,
        padding="same",
        depthwise_initializer="he_normal",
        name=f"{name}_dw_conv",
    )(x)

    # 2. LayerNorm (after DW conv, before mixing)
    x = layer_norm_2d(x, name=f"{name}_ln_1")

    # 3. Inverted bottleneck: expand x4
    x = tf.keras.layers.Dense(
        dim * 4,
        name=f"{name}_fc1",
    )(x)
    x = tf.keras.layers.Activation("gelu", name=f"{name}_gelu")(x)

    # 4. Project back to dim
    x = tf.keras.layers.Dense(
        dim,
        name=f"{name}_fc2",
    )(x)

    # 5. LayerScale (learnable per-channel scaling)
    # Use 1x1 DepthwiseConv2D so weights are properly tracked by Keras
    if layer_scale_init > 0:
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=1,
            depthwise_initializer=tf.keras.initializers.Constant(layer_scale_init),
            use_bias=False,
            trainable=True,
            name=f"{name}_layer_scale",
        )(x)

    # 6. DropPath (stochastic depth) — drop entire residual path per sample
    if drop_path_rate > 0:
        x = DropPath(
            drop_prob=drop_path_rate, name=f"{name}_droppath"
        )(x)

    # 7. Residual
    x = shortcut + x
    return x


def convnext_downsample(
    x: tf.Tensor,
    out_dim: int,
    name: str = "downsample",
) -> tf.Tensor:
    """
    ConvNeXt downsampling block:
      LayerNorm -> Conv2D 2x2 stride 2
    """
    x = layer_norm_2d(x, name=f"{name}_ln")
    x = tf.keras.layers.Conv2D(
        out_dim,
        kernel_size=2,
        strides=2,
        padding="same",
        use_bias=True,
        name=f"{name}_conv",
    )(x)
    return x