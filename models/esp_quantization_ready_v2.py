# models/esp_quantization_ready_v2.py
import tensorflow as tf
import parameters as params

def create_esp_quantization_ready_v2():
    """
    Enhanced quantization-ready model with better accuracy
    while maintaining ESP-DL and TFLite Micro compatibility
    
    Target: >95% accuracy with <50KB quantized size
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # Enhanced first conv block - more filters for better feature extraction
        tf.keras.layers.Conv2D(
            48, (3, 3), padding='same',  # Increased from 32 to 48
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv1'
        ),
        tf.keras.layers.ReLU(name='relu1'),
        tf.keras.layers.Conv2D(
            48, (3, 3), padding='same',  # Additional conv layer
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv2'
        ),
        tf.keras.layers.ReLU(name='relu2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second conv block - balanced capacity
        tf.keras.layers.Conv2D(
            96, (3, 3), padding='same',  # Increased from 64 to 96
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv3'
        ),
        tf.keras.layers.ReLU(name='relu3'),
        tf.keras.layers.Conv2D(
            96, (3, 3), padding='same',  # Additional conv layer
            kernel_initializer='he_normal', 
            use_bias=True,
            name='conv4'
        ),
        tf.keras.layers.ReLU(name='relu4'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Depthwise separable for efficiency - increased capacity
        tf.keras.layers.DepthwiseConv2D(
            (3, 3), padding='same',
            depthwise_initializer='he_normal',
            use_bias=True,
            name='depthwise1'
        ),
        tf.keras.layers.ReLU(name='depthwise_relu1'),
        tf.keras.layers.Conv2D(
            128, (1, 1),  # Increased from 64 to 128
            kernel_initializer='he_normal',
            use_bias=True,
            name='pointwise1'
        ),
        tf.keras.layers.ReLU(name='relu5'),
        
        # Global context before classification
        tf.keras.layers.GlobalAveragePooling2D(name='gap'),
        
        # Enhanced classification head
        tf.keras.layers.Dense(192, activation='relu',  # Increased from 128 to 192
                             kernel_initializer='he_normal',
                             use_bias=True,
                             name='dense1'),
        tf.keras.layers.Dropout(0.2, name='dropout1'),  # Light dropout for regularization
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model

def get_model_info():
    """Return model information for documentation"""
    return {
        "name": "ESP-DL Quantization Ready V2",
        "description": "Enhanced version with better accuracy while maintaining ESP-DL compatibility",
        "target_accuracy": ">95%",
        "target_size": "<50KB quantized", 
        "esp_dl_compatible": True,
        "tflite_micro_compatible": True,
        "features": [
            "Increased filter counts (48, 96, 128)",
            "Additional conv layers for better feature extraction",
            "Depthwise separable convolution for efficiency",
            "GlobalAveragePooling for better quantization",
            "Light dropout for regularization"
        ]
    }