# models/digit_recognizer_v2.py
"""
digit_recognizer_v2 – ReLU6 Clip Baseline (TF QAT Example Style)
=================================================================
Design goal: Minimal, quantization-compatible baseline following the
TensorFlow model optimization training example pattern.

Architecture:
  - Three stacked Conv2D blocks (32 → 64 → 64) + ReLU6 (via clip)
  - MaxPooling2D after first two conv blocks
  - GlobalAveragePooling2D  → replaces Flatten, quantization-friendly
  - Direct Dense(NB_CLASSES) + Softmax output (no hidden Dense)

Notes:
  - ReLU6 implemented as tf.clip_by_value(relu(x), 0, 6) for broad
    ESP-DL and TFLite Micro compatibility without a Lambda layer
  - No BatchNormalization → simpler, but less stable training
  - No bias when not needed for quantization

Reference: https://www.tensorflow.org/model_optimization/guide/quantization/training_example
Estimated: ~50K parameters → ~50 KB after INT8 quantization.
"""

import tensorflow as tf
import parameters as params

def create_digit_recognizer_v2():
    """
    Based on: https://www.tensorflow.org/model_optimization/guide/quantization/training_example
    
    Key points:
    - Avoids ReLU6 (use ReLU + Clip instead) for esp-dl and tflite-micro compat
    - No BatchNorm or Flatten
    - Uses GlobalAveragePooling for dimensionality reduction
    - Compatible with integer quantization (int8)
    """
    def relu6(x, name=None):
        # Equivalent to tf.nn.relu6(x) but ensures ops are supported everywhere
        return tf.clip_by_value(tf.nn.relu(x), 0.0, 6.0, name=name)

    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')

    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv1'
    )(inputs)
    x = relu6(x, name='relu6_1')
    x = tf.keras.layers.MaxPooling2D((2, 2), name='pool1')(x)

    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv2'
    )(x)
    x = relu6(x, name='relu6_2')
    x = tf.keras.layers.MaxPooling2D((2, 2), name='pool2')(x)

    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv3'
    )(x)
    x = relu6(x, name='relu6_3')

    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    outputs = tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')(x)

    model = tf.keras.Model(inputs, outputs, name="digit_recognizer_v2")

    return model