# models/simple_cnn.py
import tensorflow as tf
import parameters as params

def create_simple_cnn():
    """Simple CNN model for testing"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First conv block
        tf.keras.layers.Conv2D(params.SIMPLE_CNN_FILTERS[0], (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Second conv block  
        tf.keras.layers.Conv2D(params.SIMPLE_CNN_FILTERS[1], (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Classification head
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(params.SIMPLE_CNN_DENSE_UNITS, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax')
    ])
    
    return model