# models/digit_recognizer_v36.py
"""
digit_recognizer_v36 – GhostNet + Squeeze-and-Excitation (IoT Model)
====================================================================
Design: GhostNet architecture upgraded with Squeeze-and-Excitation.
Generates 'ghost' features via depthwise ops, then uses channel attention
to reweigh importance. Highly parameter-efficient.

Scope / Target:
  - TARGET: ESP32 Edge Deployment (Ultra-efficient IoT target, ~100 KB INT8 budget).
  - DATASETS: Primarily 10-class (for ultra low size), but also scales to 100-class.
  - RECOMMENDED COMMANDS:
    - 10-Class: python train.py --model digit_recognizer_v36 --classes 10 --color rgb --no-qat --tqt
    - 100-Class: python train.py --model digit_recognizer_v36 --classes 100 --color rgb --no-qat --tqt

All ops map safely to TFLite Micro / ESP-DL.
Fully QAT-compatible.
"""

import tensorflow as tf
import config as params

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False


def _squeeze_excite_block(x, reduction=8, name_prefix='se'):
    channels = x.shape[-1]
    y = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name=f'{name_prefix}_gap')(x)
    y = tf.keras.layers.Conv2D(
        max(4, channels // reduction), (1, 1), padding='same',
        activation='relu', kernel_initializer='he_normal',
        name=f'{name_prefix}_reduce'
    )(y)
    y = tf.keras.layers.Conv2D(
        channels, (1, 1), padding='same',
        activation='sigmoid', kernel_initializer='he_normal',
        name=f'{name_prefix}_expand'
    )(y)
    return tf.keras.layers.Multiply(name=f'{name_prefix}_scale')([x, y])


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


def _ghost_block_se(x, out_channels, stride=1, name_prefix='gb'):
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

    # Apply SE before second ghost module
    y = _squeeze_excite_block(y, reduction=8, name_prefix=f'{name_prefix}_se')

    y = _ghost_module(y, out_channels, name_prefix=f'{name_prefix}_gm2')
    
    # Linear projection
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


def create_digit_recognizer_v36(num_classes=None, input_shape=None, **kwargs):
    """
    GhostNet+SE IoT digit recognizer.
    """
    if num_classes is None:
        num_classes = params.NB_CLASSES
    if input_shape is None:
        input_shape = params.INPUT_SHAPE

    inputs = tf.keras.Input(shape=input_shape, name='input')

    x = tf.keras.layers.Conv2D(
        20, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='entry_conv'
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name='entry_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='entry_relu6')(x)

    ghost_config = [
        (32, 2),
        (40, 1),
        (56, 2),
        (56, 1),
        (72, 1),
    ]
    for i, (out_ch, s) in enumerate(ghost_config):
        x = _ghost_block_se(x, out_channels=out_ch, stride=s,
                            name_prefix=f'gb{i+1}')

    x = tf.keras.layers.Conv2D(
        120, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='head_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='head_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='head_relu6')(x)

    x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name='gap')(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)

    if params.USE_LOGITS:
        outputs = tf.keras.layers.Dense(
            num_classes, activation=None, name='logits'
        )(x)
    else:
        outputs = tf.keras.layers.Dense(
            num_classes, activation='softmax', name='output'
        )(x)

    return tf.keras.Model(inputs, outputs, name='digit_recognizer_v36')


def create_qat_model(base_model=None):
    if base_model is None:
        base_model = create_digit_recognizer_v36()
    if not QAT_AVAILABLE:
        print("Warning: QAT not available. Returning base model.")
        return base_model
    try:
        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_model(base_model)
        print("QAT model created for digit_recognizer_v36")
        return qat_model
    except Exception as e:
        print(f"QAT failed ({e}) – returning base model.")
        return base_model


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    m = create_digit_recognizer_v36()
    m.summary()
    p = m.count_params()
    print(f"\nTotal parameters : {p:,}")
    print(f"Estimated INT8 KB: ~{p * 1.1 / 1024:.1f}")
