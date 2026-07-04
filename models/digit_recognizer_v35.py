# models/digit_recognizer_v35.py
"""
digit_recognizer_v35 – MobileNetV2 + Squeeze-and-Excitation (IoT Model)
=======================================================================
Design: Upgrade of v16, optimized for ESP32 with TFLite Micro.
Adds SE channel attention inside inverted residual blocks to boost
accuracy while remaining strictly within TFLite built-in operations.

Scope / Target:
  - TARGET: ESP32 Edge Deployment (IoT Target, <200 KB INT8 budget).
  - DATASETS: Suitable for both 10-class and 100-class tasks.
  - RECOMMENDED COMMANDS:
    - 10-Class: python train.py --model digit_recognizer_v35 --classes 10 --color rgb --no-qat --tqt
    - 100-Class: python train.py --model digit_recognizer_v35 --classes 100 --color rgb --no-qat --tqt

Key principles:
  - Inverted residual bottleneck + Squeeze-and-Excitation.
  - ReLU6 only (maps to hardware accelerators).
  - GAP(keepdims=True) to avoid dynamic shape issues in TFLite.
  - No custom ops. Fully QAT-compatible.
"""

import tensorflow as tf
import config as params

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False


def _squeeze_excite_block(x, reduction=8, name_prefix='se'):
    """TFLite-safe Squeeze-and-Excitation block."""
    channels = x.shape[-1]
    
    # keepdims=True is crucial for TFLite to avoid Reshape ops
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


def _inv_res_se(x, filters_out, expand_ratio, stride, name_prefix):
    """MobileNetV2-style inverted residual with SE."""
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

    # 3. Squeeze-and-Excitation
    y = _squeeze_excite_block(y, reduction=8, name_prefix=f'{name_prefix}_se')

    # 4. Pointwise projection (linear bottleneck)
    y = tf.keras.layers.Conv2D(
        filters_out, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_project'
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_proj_bn')(y)

    # 5. Shortcut
    if use_shortcut:
        y = tf.keras.layers.Add(name=f'{name_prefix}_add')([x, y])

    return y


def create_digit_recognizer_v35(num_classes=None, input_shape=None, **kwargs):
    """
    MobileNetV2+SE IoT digit recognizer.
    """
    if num_classes is None:
        num_classes = params.NB_CLASSES
    if input_shape is None:
        input_shape = params.INPUT_SHAPE
        
    inputs = tf.keras.Input(shape=input_shape, name='input')

    # Entry conv (wider than v16's 16ch)
    x = tf.keras.layers.Conv2D(
        20, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='entry_conv'
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name='entry_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='entry_relu6')(x)

    # Inverted residual + SE stages
    # (out_ch, expand_ratio, stride)
    inv_res_config = [
        (28,  4, 2),   # spatial: /2,  channels: 20→28
        (28,  4, 1),   # residual
        (48,  4, 2),   # spatial: /4,  channels: 28→48
        (48,  6, 1),   # residual (wider expansion)
        (64,  6, 1),   # deepen
        (64,  6, 1),   # extra block for capacity
    ]
    
    for i, (out_ch, t, s) in enumerate(inv_res_config):
        x = _inv_res_se(x, filters_out=out_ch, expand_ratio=t, stride=s,
                        name_prefix=f'ir{i+1}')

    # Final head expansion
    x = tf.keras.layers.Conv2D(
        128, (1, 1), padding='same',
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

    return tf.keras.Model(inputs, outputs, name='digit_recognizer_v35')


def create_qat_model(base_model=None):
    if base_model is None:
        base_model = create_digit_recognizer_v35()
    if not QAT_AVAILABLE:
        print("Warning: QAT not available. Returning base model.")
        return base_model
    try:
        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_model(base_model)
        print("QAT model created for digit_recognizer_v35")
        return qat_model
    except Exception as e:
        print(f"QAT failed ({e}) – returning base model.")
        return base_model


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    m = create_digit_recognizer_v35()
    m.summary()
    p = m.count_params()
    print(f"\nTotal parameters : {p:,}")
    print(f"Estimated INT8 KB: ~{p * 1.1 / 1024:.1f}")
