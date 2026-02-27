# models/digit_recognizer_v15.py
"""
digit_recognizer_v15 – Lightweight Residual IoT Model
=======================================================
Design goal: Beat v4 (99.0%) and v3 (98.5%) accuracy while staying
firmly under 100 KB after INT8 quantization.

Architecture:
  - SeparableConv2D entry layer  → reduces initial parameter cost
  - Two residual blocks (skip connections) → preserve spatial features lost
    in v4's plain linear stack, yielding better accuracy for the same size
  - ReLU6 throughout              → TFLite / ESP32 / ESP-DL safe
  - BatchNormalization            → training stability (same as v12 but lighter)
  - GlobalAveragePooling          → no Dense blowup from Flatten
  - Tiny Dense(48) head           → keeps param count low

Estimated: ~80-90K parameters → ~70-80 KB after INT8 quantization.
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
# Helpers
# ---------------------------------------------------------------------------

def _res_block_relu6(x, filters, name_prefix):
    """Residual block with two Conv2D + BN + ReLU6, compatible with TFLite/ESP.

    The shortcut uses a 1×1 Conv if channel count changes, otherwise identity.
    Uses explicit ReLU6 layers (not activation= argument) for QAT compatibility.
    """
    shortcut = x

    y = tf.keras.layers.Conv2D(
        filters, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_conv_a'
    )(x)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_bn_a')(y)
    y = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_relu6_a')(y)

    y = tf.keras.layers.Conv2D(
        filters, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_conv_b'
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_bn_b')(y)

    # Adjust shortcut channel count if needed
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, (1, 1), padding='same',
            kernel_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_shortcut_conv'
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(
            name=f'{name_prefix}_shortcut_bn'
        )(shortcut)

    y = tf.keras.layers.Add(name=f'{name_prefix}_add')([shortcut, y])
    y = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_relu6_out')(y)
    return y


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def create_digit_recognizer_v15():
    """
    Dispatch to the grayscale or RGB variant based on params.INPUT_CHANNELS.
    """
    ch = params.INPUT_SHAPE[-1]
    if ch == 1:
        print("Creating digit_recognizer_v15 (grayscale)")
        return _build_v15(entry_filters=16, res1_filters=32, res2_filters=56,
                          dense_units=48, model_name="digit_recognizer_v15_gray")
    else:
        # RGB: slightly wider entry to handle the extra channel cost
        print("Creating digit_recognizer_v15 (RGB)")
        return _build_v15_rgb(entry_filters=20, res1_filters=36, res2_filters=64,
                              dense_units=56, model_name="digit_recognizer_v15_rgb")


def _build_v15(entry_filters, res1_filters, res2_filters, dense_units, model_name):
    """Grayscale variant – uses plain Conv2D entry (1-channel is cheap enough)."""
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')

    # Entry block
    x = tf.keras.layers.Conv2D(
        entry_filters, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='entry_conv'
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name='entry_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='entry_relu6')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='entry_pool')(x)

    # Residual stage 1
    x = _res_block_relu6(x, res1_filters, name_prefix='res1')
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='res1_pool')(x)

    # Residual stage 2
    x = _res_block_relu6(x, res2_filters, name_prefix='res2')

    # Classifier
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    x = tf.keras.layers.Dense(dense_units, use_bias=True, name='fc')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='fc_relu6')(x)
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, activation='softmax', name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name=model_name)


def _build_v15_rgb(entry_filters, res1_filters, res2_filters, dense_units, model_name):
    """RGB variant – uses SeparableConv2D in the entry to keep params low."""
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')

    # Entry: separable conv is ~3× cheaper for RGB input
    x = tf.keras.layers.SeparableConv2D(
        entry_filters, (3, 3), padding='same',
        depthwise_initializer='he_normal', pointwise_initializer='he_normal',
        use_bias=False, name='entry_sep_conv'
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name='entry_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='entry_relu6')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='entry_pool')(x)

    # Residual stage 1
    x = _res_block_relu6(x, res1_filters, name_prefix='res1')
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='res1_pool')(x)

    # Residual stage 2
    x = _res_block_relu6(x, res2_filters, name_prefix='res2')

    # Classifier
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    x = tf.keras.layers.Dense(dense_units, use_bias=True, name='fc')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='fc_relu6')(x)
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, activation='softmax', name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name=model_name)


# ---------------------------------------------------------------------------
# QAT wrapper  (standard pattern used by all project models)
# ---------------------------------------------------------------------------

def create_qat_model(base_model=None):
    """
    Wrap the base model with QAT annotations if tensorflow_model_optimization
    is available, otherwise return the base model unchanged.
    """
    if base_model is None:
        base_model = create_digit_recognizer_v15()

    if not QAT_AVAILABLE:
        print("⚠️  QAT not available – returning base model.")
        return base_model

    try:
        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_model(base_model)
        print("✅ QAT model created for digit_recognizer_v15")
        return qat_model
    except Exception as e:
        print(f"⚠️  QAT failed ({e}) – returning base model.")
        return base_model


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = create_digit_recognizer_v15()
    model.summary()
    total = model.count_params()
    print(f"\nTotal parameters : {total:,}")
    # Rough INT8 size estimate: 1 byte per param + ~10% overhead
    print(f"Estimated INT8 size: ~{total * 1.1 / 1024:.1f} KB")
