# models/esp_high_capacity.py
import tensorflow as tf
import parameters as params

def create_esp_high_capacity():
    """
    Higher capacity model that can actually learn the task
    while maintaining ESP-DL compatibility
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # Expanded first layer
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', 
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Maintain capacity in middle layers
        tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Final conv layer
        tf.keras.layers.Conv2D(256, (3, 3), padding='same',
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Adequate dense layer
        tf.keras.layers.Dense(256, activation='relu',
                             kernel_initializer='he_normal'),
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model
