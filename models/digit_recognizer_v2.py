# models/digit_recognizer_v2.py
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