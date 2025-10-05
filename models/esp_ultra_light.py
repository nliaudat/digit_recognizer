# models/esp_ultra_light.py
import tensorflow as tf
import parameters as params

def create_esp_ultra_light():
    """
    Ultra-light model for ESP32 with minimal memory footprint
    Target: <50KB quantized
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # Tiny feature extractor
        tf.keras.layers.Conv2D(
            8, (3, 3), padding='same', activation='relu',
            kernel_initializer='he_normal',
            name='conv1'
        ),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        tf.keras.layers.Conv2D(
            16, (3, 3), padding='same', activation='relu', 
            kernel_initializer='he_normal',
            name='conv2'
        ),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Global average pooling instead of flatten + dense
        tf.keras.layers.GlobalAveragePooling2D(name='gap'),
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model
