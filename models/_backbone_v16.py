"""
models/_backbone_v16.py — Shared v16 MobileNetV2 Backbone
==========================================================
Extracted from digit_recognizer_v16/v41/v42 to avoid duplication.
Backbone improvements propagate automatically to all architectures.

Architecture:
  Input → Conv2D(16) → BN → ReLU6 → 5× Inverted Residual → Conv2D(96)
        → BN → ReLU6 → GAP → Flatten → 96-dim feature vector
"""

import tensorflow as tf


def _inv_res(x, filters_out, expand_ratio, stride, name_prefix):
    """MobileNetV2-style inverted residual."""
    ch_in = x.shape[-1]
    ch_exp = ch_in * expand_ratio
    use_shortcut = (stride == 1 and ch_in == filters_out)

    # 1. Pointwise expansion
    y = tf.keras.layers.Conv2D(
        ch_exp, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_expand'
    )(x)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_exp_bn')(y)
    y = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_exp_relu6')(y)

    # 2. Depthwise conv
    y = tf.keras.layers.DepthwiseConv2D(
        (3, 3), strides=stride, padding='same',
        depthwise_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_dw'
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_dw_bn')(y)
    y = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_dw_relu6')(y)

    # 3. Pointwise projection (linear bottleneck)
    y = tf.keras.layers.Conv2D(
        filters_out, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_project'
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_proj_bn')(y)

    # 4. Shortcut
    if use_shortcut:
        y = tf.keras.layers.Add(name=f'{name_prefix}_add')([x, y])

    return y


def create_v16_backbone(inputs, name=''):
    """
    Create the shared v16 feature extractor.
    
    Args:
        inputs: Keras Input tensor [batch, H, W, C]
        name: Scope prefix for layer names.  When '' (default), uses bare names
              for backward compatibility with existing v16/v41 checkpoints.
    
    Returns:
        96-dim flattened feature vector [batch, 96]
    """
    prefix = f"{name}_" if name else ""
    
    # Entry conv
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{prefix}entry_conv'
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name=f'{prefix}entry_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name=f'{prefix}entry_relu6')(x)

    # Inverted residual stages
    inv_res_config = [
        (24,  4, 2),
        (24,  4, 1),
        (40,  4, 2),
        (40,  6, 1),
        (56,  6, 1),
    ]
    for i, (out_ch, t, s) in enumerate(inv_res_config):
        x = _inv_res(x, filters_out=out_ch, expand_ratio=t, stride=s,
                     name_prefix=f'{prefix}ir{i+1}')

    # Head conv
    x = tf.keras.layers.Conv2D(
        96, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{prefix}head_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f'{prefix}head_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name=f'{prefix}head_relu6')(x)

    # Global average pooling → shared feature vector
    x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name=f'{prefix}gap')(x)
    shared = tf.keras.layers.Flatten(name=f'{prefix}flatten')(x)

    return shared
