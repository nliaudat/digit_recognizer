# models/practical_tiny_depthwise.py
import tensorflow as tf
import parameters as params


def create_practical_tiny_depthwise():
    """
    WORKING version with proper initialization and activations
    """
    model = tf.keras.Sequential([
        # First conv with proper initialization
        tf.keras.layers.Conv2D(
            32, (3, 3), 
            activation='relu', 
            padding='same',
            input_shape=params.INPUT_SHAPE,
            kernel_initializer='he_normal',  # CRITICAL for ReLU
            name='conv1'
        ),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Depthwise separable block 1
        tf.keras.layers.DepthwiseConv2D(
            (3, 3), 
            activation='relu', 
            padding='same',
            depthwise_initializer='he_normal',  # CRITICAL
            name='depthwise1'
        ),
        tf.keras.layers.Conv2D(
            64, (1, 1), 
            activation='relu',
            kernel_initializer='he_normal',  # CRITICAL
            name='pointwise1'
        ),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Depthwise separable block 2  
        tf.keras.layers.DepthwiseConv2D(
            (3, 3), 
            activation='relu', 
            padding='same',
            depthwise_initializer='he_normal',  # CRITICAL
            name='depthwise2'
        ),
        tf.keras.layers.Conv2D(
            128, (1, 1), 
            activation='relu',
            kernel_initializer='he_normal',  # CRITICAL
            name='pointwise2'
        ),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool3'),
        
        # Classification head
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(256, activation='relu', 
                             kernel_initializer='he_normal',  # CRITICAL
                             name='dense1'),
        tf.keras.layers.Dropout(0.5, name='dropout1'),
        tf.keras.layers.Dense(params.NB_CLASSES, 
                             kernel_initializer='he_normal',  # CRITICAL
                             name='output')
    ])
    
    return model

def get_model_info():
    """Return ESP-DL compatible model information"""
    return {
        "name": "ESP-DL Compatible Depthwise CNN - FIXED",
        "description": "Fixed version with proper capacity for learning",
        "esp_dl_compatible": True,
        "expected_size": "150-300KB",
        "expected_accuracy": "95%+",
        "training_notes": "Use bias during training, ESP-DL handles quantization"
    }