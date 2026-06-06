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

    # 6. DropPath (stochastic depth)
    if drop_path_rate > 0:
        x = tf.keras.layers.Dropout(
            rate=drop_path_rate, name=f"{name}_droppath"
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