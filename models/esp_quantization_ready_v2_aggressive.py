# models/esp_quantization_ready_v2_aggressive.py
import tensorflow as tf
import parameters as params

def create_esp_quantization_ready_v2_aggressive():
    """
    Maximum accuracy version - targets >97% accuracy
    while maintaining ESP-DL and TFLite Micro compatibility
    
    Target: >97% accuracy with <80KB quantized size
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # Block 1 - High capacity
        tf.keras.layers.Conv2D(
            64, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv1'
        ),
        tf.keras.layers.ReLU(name='relu1'),
        tf.keras.layers.Conv2D(
            64, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv2'
        ),
        tf.keras.layers.ReLU(name='relu2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Block 2 - High capacity  
        tf.keras.layers.Conv2D(
            128, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv3'
        ),
        tf.keras.layers.ReLU(name='relu3'),
        tf.keras.layers.Conv2D(
            128, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv4'
        ),
        tf.keras.layers.ReLU(name='relu4'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Block 3 - Depthwise for efficiency
        tf.keras.layers.DepthwiseConv2D(
            (3, 3), padding='same',
            depthwise_initializer='he_normal',
            use_bias=True,
            name='depthwise1'
        ),
        tf.keras.layers.ReLU(name='depthwise_relu1'),
        tf.keras.layers.Conv2D(
            256, (1, 1),
            kernel_initializer='he_normal',
            use_bias=True,
            name='pointwise1'
        ),
        tf.keras.layers.ReLU(name='relu5'),
        
        # Classification head
        tf.keras.layers.GlobalAveragePooling2D(name='gap'),
        tf.keras.layers.Dense(
            256, activation='relu',
            kernel_initializer='he_normal',
            use_bias=True,
            name='dense1'
        ),
        tf.keras.layers.Dropout(0.3, name='dropout1'),
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model

def get_model_info():
    """Return model information for documentation"""
    return {
        "name": "ESP-DL Quantization Ready V2 Aggressive",
        "description": "Maximum accuracy version targeting >97% while maintaining ESP-DL compatibility",
        "target_accuracy": ">97%",
        "target_size": "<80KB quantized", 
        "esp_dl_compatible": True,
        "tflite_micro_compatible": True,
        "features": [
            "High capacity conv blocks (64, 128 filters)",
            "Multiple conv layers per block",
            "Depthwise separable convolution in final block",
            "Large dense layer (256 units)",
            "Moderate dropout (0.3) for regularization",
            "GlobalAveragePooling for better quantization"
        ],
        "expected_improvement": "Significant accuracy boost over v1, slightly larger model size"
    }