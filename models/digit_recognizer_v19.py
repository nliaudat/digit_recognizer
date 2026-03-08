# models/digit_recognizer_v19.py
"""
digit_recognizer_v19 – Scaled GhostNet for Maximum Performance (10cls RGB)
==========================================================================
Design: A carefully scaled GhostNet architecture designed to surpass v18
while staying within the 100KB INT8 budget and maintaining high inference
speed on ESP32.

Key improvements over v18:
1. Increased width of early ghost modules to capture richer low-level features.
2. Optimized bottleneck structure: (24, 32, 40, 56, 64, 80).
3. Widened final projection layer to 128 to improve classification capacity.
4. Strictly TFLite Micro / ESP-NN compatible ops.
5. Fully QAT-compatible.

Targeting ~98K parameters.
"""

import tensorflow as tf
import parameters as params

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Ghost Module (TFLite Micro Compatible)
# ---------------------------------------------------------------------------

def _ghost_module(x, out_channels, ratio=2, dw_kernel=3, name_prefix='gm'):
    intrinsic_ch = out_channels // ratio
    ghost_ch = out_channels - intrinsic_ch

    primary = tf.keras.layers.Conv2D(
        intrinsic_ch, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_primary_conv'
    )(x)
    primary = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_primary_bn')(primary)
    primary = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_primary_relu6')(primary)

    ghost = tf.keras.layers.DepthwiseConv2D(
        (dw_kernel, dw_kernel), padding='same',
        depthwise_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_ghost_dw'
    )(primary)
    ghost = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_ghost_bn')(ghost)
    ghost = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_ghost_relu6')(ghost)

    if ghost_ch != intrinsic_ch:
        ghost = tf.keras.layers.Conv2D(
            ghost_ch, (1, 1), padding='same',
            kernel_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_ghost_adjust'
        )(ghost)

    return tf.keras.layers.Concatenate(name=f'{name_prefix}_concat')([primary, ghost])


# ---------------------------------------------------------------------------
# Ghost Bottleneck Block (TFLite Micro Compatible)
# ---------------------------------------------------------------------------

def _ghost_block(x, out_channels, stride=1, name_prefix='gb'):
    ch_in = x.shape[-1]
    mid_channels = out_channels

    y = _ghost_module(x, mid_channels, name_prefix=f'{name_prefix}_gm1')

    if stride > 1:
        y = tf.keras.layers.DepthwiseConv2D(
            (3, 3), strides=stride, padding='same',
            depthwise_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_stride_dw'
        )(y)
        y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_stride_bn')(y)

    y = _ghost_module(y, out_channels, name_prefix=f'{name_prefix}_gm2')
    
    y = tf.keras.layers.Conv2D(
        out_channels, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_proj'
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_proj_bn')(y)

    if stride == 1 and ch_in == out_channels:
        shortcut = x
    else:
        shortcut = tf.keras.layers.DepthwiseConv2D(
            (3, 3), strides=stride, padding='same',
            depthwise_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_sc_dw'
        )(x)
        shortcut = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_sc_dw_bn')(shortcut)
        shortcut = tf.keras.layers.Conv2D(
            out_channels, (1, 1), padding='same',
            kernel_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_sc_conv'
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_sc_bn')(shortcut)

    y = tf.keras.layers.Add(name=f'{name_prefix}_add')([shortcut, y])
    y = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_relu6')(y)
    return y


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def create_digit_recognizer_v19():
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')

    # Entry conv - Scaled to 20 filters
    x = tf.keras.layers.Conv2D(
        20, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='entry_conv'
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name='entry_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='entry_relu6')(x)

    # V19 Ghost blocks - Optimized configuration
    # (out_channels, stride)
    ghost_config = [
        (32, 2),   # spatial /2, 20→32
        (40, 1),   # residual,   32→40
        (56, 2),   # spatial /4, 40→56
        (64, 1),   # residual,   56→64
        (80, 1),   # NEW residual, 64→80 (further capacity for 100cls/RGB)
    ]
    for i, (out_ch, s) in enumerate(ghost_config):
        x = _ghost_block(x, out_channels=out_ch, stride=s,
                         name_prefix=f'gb{i+1}')

    # Final 1×1 expansion - Wide 128 to capture features before GAP
    x = tf.keras.layers.Conv2D(
        128, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='head_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='head_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='head_relu6')(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)

    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, activation='softmax', name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name='digit_recognizer_v19')


# ---------------------------------------------------------------------------
# QAT wrapper
# ---------------------------------------------------------------------------

def create_qat_model(base_model=None):
    if base_model is None:
        base_model = create_digit_recognizer_v19()
    if not QAT_AVAILABLE:
        print("Warning: QAT not available. Returning base model.")
        return base_model
    try:
        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_model(base_model)
        print("QAT model created for digit_recognizer_v19")
        return qat_model
    except Exception as e:
        print(f"QAT failed ({e}) – returning base model.")
        return base_model


if __name__ == "__main__":
    # Add parent directory to path so parameters.py can be found when run directly
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    m = create_digit_recognizer_v19()
    m.summary()
    p = m.count_params()
    print(f"\nTotal parameters : {p:,}")
    print(f"Estimated INT8 KB: ~{p * 1.1 / 1024:.1f}")
