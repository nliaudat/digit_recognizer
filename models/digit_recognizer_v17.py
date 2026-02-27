# models/digit_recognizer_v17.py
"""
digit_recognizer_v17 – GhostNet-Inspired Ultra-Efficient IoT Model
==================================================================
Design: Generate half the feature maps via cheap depthwise operations
instead of full convolutions — same representation, ~50% fewer FLOPs.

Reference: Han et al. "GhostNet: More Features from Cheap Operations" CVPR 2020

Ghost Module:
  - Compute n//2 "intrinsic" features via Conv2D
  - Compute n//2 "ghost" features via DepthwiseConv2D on intrinsic features
  - Concatenate → n total features at ~50% the parameter cost

All ops are TFLite Micro safe:
  Conv2D, DepthwiseConv2D, Concatenate, Add, GlobalAveragePooling2D,
  Dense, BatchNorm (folded), ReLU6, Softmax

Targeting ~50KB INT8. Smallest model in the lineup.
Fully QAT-compatible.
"""

import tensorflow as tf
import parameters as params

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Ghost Module
# ---------------------------------------------------------------------------

def _ghost_module(x, out_channels, ratio=2, dw_kernel=3, name_prefix='gm'):
    """Ghost Module: cheap features from cheap operations.

    Args:
        out_channels: total output channels
        ratio: ghost ratio — intrinsic channels = out_channels // ratio
        dw_kernel: kernel size for the cheap depthwise operation
    """
    intrinsic_ch = out_channels // ratio
    ghost_ch = out_channels - intrinsic_ch  # remaining channels

    # Primary: standard Conv2D for intrinsic features
    primary = tf.keras.layers.Conv2D(
        intrinsic_ch, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_primary_conv'
    )(x)
    primary = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_primary_bn')(primary)
    primary = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_primary_relu6')(primary)

    # Ghost: cheap depthwise on the primary features
    ghost = tf.keras.layers.DepthwiseConv2D(
        (dw_kernel, dw_kernel), padding='same',
        depthwise_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_ghost_dw'
    )(primary)
    ghost = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_ghost_bn')(ghost)
    ghost = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_ghost_relu6')(ghost)

    # Handle channel mismatch if out_channels is odd
    if ghost_ch != intrinsic_ch:
        ghost = tf.keras.layers.Conv2D(
            ghost_ch, (1, 1), padding='same',
            kernel_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_ghost_adjust'
        )(ghost)

    # Concatenate → out_channels total
    return tf.keras.layers.Concatenate(name=f'{name_prefix}_concat')([primary, ghost])


# ---------------------------------------------------------------------------
# Ghost Bottleneck Block
# ---------------------------------------------------------------------------

def _ghost_block(x, out_channels, stride=1, name_prefix='gb'):
    """Ghost bottleneck: two Ghost Modules + optional depthwise stride.

    Structure:
        GhostModule (expand) → [DW stride if stride>1] → GhostModule (project)
        + shortcut (1×1 Conv if channels change or stride>1)
    """
    ch_in = x.shape[-1]
    mid_channels = out_channels  # we keep mid = out for simplicity

    # First Ghost Module (expand features)
    y = _ghost_module(x, mid_channels, name_prefix=f'{name_prefix}_gm1')

    # Depthwise stride (only if needed for spatial reduction)
    if stride > 1:
        y = tf.keras.layers.DepthwiseConv2D(
            (3, 3), strides=stride, padding='same',
            depthwise_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_stride_dw'
        )(y)
        y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_stride_bn')(y)

    # Second Ghost Module (project)
    y = _ghost_module(y, out_channels, name_prefix=f'{name_prefix}_gm2')
    # Linear output (no activation before add)
    y = tf.keras.layers.Conv2D(
        out_channels, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_proj'
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_proj_bn')(y)

    # Shortcut
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

def create_digit_recognizer_v17():
    """
    GhostNet-inspired ultra-efficient IoT digit recognizer.
    Generates 'ghost' feature maps from cheap depthwise operations,
    targeting ~50KB INT8 with competitive accuracy.
    All ops are TFLite Micro / ESP32 built-in ops safe.
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')

    # Entry conv
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='entry_conv'
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name='entry_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='entry_relu6')(x)

    # Ghost blocks
    # (out_channels, stride)
    ghost_config = [
        (24, 2),   # spatial /2,  16→24 ch
        (32, 1),   # residual,    24→32
        (40, 2),   # spatial /4,  32→40
        (48, 1),   # residual,    40→48
    ]
    for i, (out_ch, s) in enumerate(ghost_config):
        x = _ghost_block(x, out_channels=out_ch, stride=s,
                         name_prefix=f'gb{i+1}')

    # Final 1×1 expansion before GAP
    x = tf.keras.layers.Conv2D(
        80, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='head_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='head_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='head_relu6')(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)

    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, activation='softmax', name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name='digit_recognizer_v17')


# ---------------------------------------------------------------------------
# QAT wrapper
# ---------------------------------------------------------------------------

def create_qat_model(base_model=None):
    if base_model is None:
        base_model = create_digit_recognizer_v17()
    if not QAT_AVAILABLE:
        print("Warning: QAT not available. Returning base model.")
        return base_model
    try:
        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_model(base_model)
        print("QAT model created for digit_recognizer_v17")
        return qat_model
    except Exception as e:
        print(f"QAT failed ({e}) – returning base model.")
        return base_model


if __name__ == "__main__":
    m = create_digit_recognizer_v17()
    m.summary()
    p = m.count_params()
    print(f"\nTotal parameters : {p:,}")
    print(f"Estimated INT8 KB: ~{p * 1.1 / 1024:.1f}")
