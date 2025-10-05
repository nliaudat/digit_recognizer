# models/esp_optimized_cnn.py
import tensorflow as tf
import parameters as params

def create_esp_optimized_cnn():
    """
    ESP-DL optimized model with guaranteed operator compatibility
    and small footprint (<100KB quantized)
    """
    model = tf.keras.Sequential([
        # Input
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First conv - replace BatchNorm with simple scaling
        tf.keras.layers.Conv2D(
            16, (3, 3), padding='same',
            kernel_initializer='he_normal',
            name='conv1'
        ),
        tf.keras.layers.ReLU(name='relu1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Depthwise separable conv 1
        tf.keras.layers.DepthwiseConv2D(
            (3, 3), padding='same',
            depthwise_initializer='he_normal',
            name='depthwise1'
        ),
        tf.keras.layers.ReLU(name='depthwise_relu1'),
        tf.keras.layers.Conv2D(
            32, (1, 1),
            kernel_initializer='he_normal',
            name='pointwise1'
        ),
        tf.keras.layers.ReLU(name='relu2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Depthwise separable conv 2
        tf.keras.layers.DepthwiseConv2D(
            (3, 3), padding='same',
            depthwise_initializer='he_normal',
            name='depthwise2'
        ),
        tf.keras.layers.ReLU(name='depthwise_relu2'),
        tf.keras.layers.Conv2D(
            64, (1, 1),
            kernel_initializer='he_normal',
            name='pointwise2'
        ),
        tf.keras.layers.ReLU(name='relu3'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool3'),
        
        # Classification head - reduced size for ESP32
        tf.keras.layers.GlobalAveragePooling2D(name='gap'),
        tf.keras.layers.Dense(64, activation='relu', 
                             kernel_initializer='he_normal',
                             name='dense1'),
        tf.keras.layers.Dense(params.NB_CLASSES, name='output', activation='softmax')
    ])
    
    return model