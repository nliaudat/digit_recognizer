# models/digit_recognizer_v37_teacher.py
"""
digit_recognizer_v37_teacher – Wide MobileNetV3-style Teacher
=============================================================
Design: PC-only teacher model for distillation.
Large capacity (~5M params) to achieve very high accuracy on 10 and 100 classes.
Uses inverted residuals with Squeeze-and-Excitation, but much wider and deeper
than IoT deployable models.

Not intended for ESP32 deployment (too large).
"""

import tensorflow as tf
import config as params

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False


def _squeeze_excite_block(x, reduction=4, name_prefix='se'):
    channels = x.shape[-1]
    y = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name=f'{name_prefix}_gap')(x)
    y = tf.keras.layers.Conv2D(
        max(8, channels // reduction), (1, 1), padding='same',
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
    y = _squeeze_excite_block(y, reduction=4, name_prefix=f'{name_prefix}_se')

    # 4. Pointwise projection
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


def create_digit_recognizer_v37_teacher(num_classes=None, input_shape=None, **kwargs):
    if num_classes is None:
        num_classes = params.NB_CLASSES
    if input_shape is None:
        input_shape = params.INPUT_SHAPE

    inputs = tf.keras.Input(shape=input_shape, name='input')

    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='entry_conv'
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name='entry_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='entry_relu6')(x)

    # Wide configuration for teacher
    # (out_ch, expand_ratio, stride)
    inv_res_config = [
        (64,  4, 2),
        (64,  4, 1),
        (96,  6, 2),
        (96,  6, 1),
        (96,  6, 1),
        (128, 6, 2),
        (128, 6, 1),
        (128, 6, 1),
        (192, 6, 1),
        (192, 6, 1),
        (192, 6, 1),
    ]
    
    for i, (out_ch, t, s) in enumerate(inv_res_config):
        x = _inv_res_se(x, filters_out=out_ch, expand_ratio=t, stride=s,
                        name_prefix=f'ir{i+1}')

    x = tf.keras.layers.Conv2D(
        512, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='head_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='head_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='head_relu6')(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    
    x = tf.keras.layers.Dense(1024, activation='relu', name='dense_1')(x)
    x = tf.keras.layers.Dropout(0.4, name='dropout_1')(x)
    x = tf.keras.layers.Dense(512, activation='relu', name='dense_2')(x)
    
    if params.USE_LOGITS:
        outputs = tf.keras.layers.Dense(num_classes, activation=None, name='logits')(x)
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)

    return tf.keras.Model(inputs, outputs, name='digit_recognizer_v37_teacher')


def create_qat_model(base_model=None):
    if base_model is None:
        base_model = create_digit_recognizer_v37_teacher()
    if not QAT_AVAILABLE:
        print("Warning: QAT not available. Returning base model.")
        return base_model
    try:
        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_model(base_model)
        print("QAT model created for digit_recognizer_v37_teacher")
        return qat_model
    except Exception as e:
        print(f"QAT failed ({e}) – returning base model.")
        return base_model


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    m = create_digit_recognizer_v37_teacher()
    m.summary()
    p = m.count_params()
    print(f"\nTotal parameters : {p:,}")
    print(f"Estimated FP32 MB: ~{p * 4 / (1024*1024):.1f}")
