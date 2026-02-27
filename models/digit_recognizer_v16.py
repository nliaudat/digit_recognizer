# models/digit_recognizer_v16.py
"""
digit_recognizer_v16 – MobileNetV2 Inverted Residual IoT Model
===============================================================
Design: Espressif-recommended architecture for TFLite Micro on ESP32.

Key principles:
  - Inverted residual bottleneck: 1×1 expand → 3×3 depthwise → 1×1 project
  - ReLU6 only (fused by ESP-NN into DEPTHWISE_CONV_2D for maximum speed)
  - BatchNorm folded into Conv after INT8 quantisation
  - No Lambda, no SE, no custom ops — strict TFLite Micro built-in ops only
  - Stride embedded in dw conv (no separate MaxPool) → gradient-friendly

Estimated: ~80K params / ~70KB INT8. Fully QAT-compatible.
"""

import tensorflow as tf
import parameters as params

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Inverted residual bottleneck  (MobileNetV2 core block)
# ---------------------------------------------------------------------------

def _inv_res(x, filters_out, expand_ratio, stride, name_prefix):
    """MobileNetV2-style inverted residual.

    expand_ratio: pointwise expansion factor (typically 4–6)
    stride: applied in the depthwise conv; stride=2 halves spatial dims
    Shortcut is only added when stride==1 and channels match.
    """
    ch_in = x.shape[-1]
    ch_exp = ch_in * expand_ratio

    use_shortcut = (stride == 1 and ch_in == filters_out)

    # 1. Pointwise expansion (1×1 Conv + BN + ReLU6)
    y = tf.keras.layers.Conv2D(
        ch_exp, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_expand'
    )(x)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_exp_bn')(y)
    y = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_exp_relu6')(y)

    # 2. Depthwise conv (3×3 + BN + ReLU6)
    y = tf.keras.layers.DepthwiseConv2D(
        (3, 3), strides=stride, padding='same',
        depthwise_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_dw'
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_dw_bn')(y)
    y = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_dw_relu6')(y)

    # 3. Pointwise projection (1×1 Conv + BN, NO activation — linear bottleneck)
    y = tf.keras.layers.Conv2D(
        filters_out, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_project'
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_proj_bn')(y)

    # 4. Shortcut (identity or skip)
    if use_shortcut:
        y = tf.keras.layers.Add(name=f'{name_prefix}_add')([x, y])

    return y


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def create_digit_recognizer_v16():
    """
    MobileNetV2-style IoT digit recognizer optimised for ESP32 TFLite Micro.
    All ops map to ESP-NN hardware-accelerated kernels.
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

    # Inverted residual stages
    # (out_ch, expand_ratio, stride)
    inv_res_config = [
        (24,  4, 2),   # spatial: /2,  channels: 16→24
        (24,  4, 1),   # residual pass
        (40,  4, 2),   # spatial: /4,  channels: 24→40
        (40,  6, 1),   # residual pass with wider expansion
        (56,  6, 1),   # deepen without downsampling
    ]
    for i, (out_ch, t, s) in enumerate(inv_res_config):
        x = _inv_res(x, filters_out=out_ch, expand_ratio=t, stride=s,
                     name_prefix=f'ir{i+1}')

    # Final 1×1 Conv to widen representation before GAP
    x = tf.keras.layers.Conv2D(
        96, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='head_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='head_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='head_relu6')(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)

    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, activation='softmax', name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name='digit_recognizer_v16')


# ---------------------------------------------------------------------------
# QAT wrapper
# ---------------------------------------------------------------------------

def create_qat_model(base_model=None):
    if base_model is None:
        base_model = create_digit_recognizer_v16()
    if not QAT_AVAILABLE:
        print("Warning: QAT not available. Returning base model.")
        return base_model
    try:
        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_model(base_model)
        print("QAT model created for digit_recognizer_v16")
        return qat_model
    except Exception as e:
        print(f"QAT failed ({e}) – returning base model.")
        return base_model


if __name__ == "__main__":
    m = create_digit_recognizer_v16()
    m.summary()
    p = m.count_params()
    print(f"\nTotal parameters : {p:,}")
    print(f"Estimated INT8 KB: ~{p * 1.1 / 1024:.1f}")
