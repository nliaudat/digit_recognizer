# models/esp_haverland_compatible.py
import tensorflow as tf
import parameters as params

def create_esp_haverland_compatible():
    """
    Modified Haverland architecture that maintains accuracy 
    while being ESP-DL compatible
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First conv block - keep structure but simplify
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', 
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Second conv block  
        tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Third conv block - reduced but sufficient
        tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Classification head - reduced but adequate
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu',
                             kernel_initializer='he_normal'),
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model