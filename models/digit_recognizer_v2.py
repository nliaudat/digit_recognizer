# models/digit_recognizer_v2.py
import tensorflow as tf
import parameters as params

def create_digit_recognizer_v2():
    """
    Model specifically designed for smooth INT8 quantization
    - No BatchNormalization
    - All activations compatible with quantization
    - Symmetric weight distributions
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First conv with explicit activation
        tf.keras.layers.Conv2D(
            32, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,  # Important for quantization
            name='conv1'
        ),
        tf.keras.layers.ReLU(name='relu1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second conv
        # tf.keras.layers.Conv2D(
            # 64, (3, 3), padding='same',
            # kernel_initializer='he_normal',
            # use_bias=True,
            # name='conv2'
        # ),
        tf.keras.layers.DepthwiseConv2D(
            (3, 3), padding='same',
            depthwise_initializer='he_normal',
            use_bias=True,
            name='depthwise0'
        ),
        tf.keras.layers.ReLU(name='relu2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        
        # Third conv - depthwise for efficiency
        tf.keras.layers.DepthwiseConv2D(
            (3, 3), padding='same',
            depthwise_initializer='he_normal',
            use_bias=True,
            name='depthwise1'
        ),
        tf.keras.layers.ReLU(name='depthwise_relu1'),
        tf.keras.layers.Conv2D(
            64, (1, 1),
            kernel_initializer='he_normal',
            use_bias=True,
            name='pointwise1'
        ),
        tf.keras.layers.ReLU(name='relu3'),
        
        # Global average pooling - quantization friendly
        tf.keras.layers.GlobalAveragePooling2D(name='gap'),
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model