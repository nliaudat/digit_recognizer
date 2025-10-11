# models/simple_cnn_v2.py
import tensorflow as tf
import parameters as params

def create_simple_cnn_v2():
    """TFLite-optimized CNN with better quantization support"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First conv block - replaced BatchNorm with simpler normalization
        tf.keras.layers.Conv2D(params.SIMPLE_CNN_FILTERS[0], (3, 3), activation='relu', padding='same'),
        tf.keras.layers.LayerNormalization(),  # Better for quantization than BatchNorm
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.1),  # Small dropout instead of heavy BatchNorm
        
        # Second conv block
        tf.keras.layers.Conv2D(params.SIMPLE_CNN_FILTERS[1], (3, 3), activation='relu', padding='same'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.1),
        
        # Optional third conv block for better feature extraction
        tf.keras.layers.Conv2D(params.SIMPLE_CNN_FILTERS[1] * 2, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        
        # Classification head
        tf.keras.layers.Dense(params.SIMPLE_CNN_DENSE_UNITS, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model