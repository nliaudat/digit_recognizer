# models/esp_quantization_ready_v2.py
"""
esp_quantization_ready_v2 – Enhanced ESP-DL Quantization-Ready CNN (>95% target)
==================================================================================
Design goal: Improve accuracy over v1 by increasing filter counts and adding
extra conv layers, while keeping the graph strictly ESP-DL and TFLite Micro
compatible. Targets >95% accuracy within <50 KB quantized.

Architecture:
  - Conv2D(48) × 2 + ReLU + MaxPool             → Block 1 (wider than v1)
  - Conv2D(96) × 2 + ReLU + MaxPool             → Block 2 (high capacity)
  - DepthwiseConv2D(3×3) + Conv2D(128, 1×1) + ReLU → DW-Sep block (final)
  - GlobalAveragePooling2D
  - Dense(192) + ReLU + Dropout(0.2) → Dense(NB_CLASSES) Softmax

Notes:
  - No BatchNorm (ESP-DL operator compatibility)
  - Standard ReLU throughout
  - Light Dropout(0.2) for regularization
  - No QAT wrapper

Estimated: ~350K parameters → ~350 KB after INT8 quantization.
"""

import parameters as params
from utils.keras_helper import keras

def create_esp_quantization_ready_v2():
    """
    Enhanced quantization-ready model with better accuracy
    while maintaining ESP-DL and TFLite Micro compatibility
    
    Target: >95% accuracy with <50KB quantized size
    """
    model = keras.Sequential([
        keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # Enhanced first conv block - more filters for better feature extraction
        keras.layers.Conv2D(
            48, (3, 3), padding='same',  # Increased from 32 to 48
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv1'
        ),
        keras.layers.ReLU(name='relu1'),
        keras.layers.Conv2D(
            48, (3, 3), padding='same',  # Additional conv layer
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv2'
        ),
        keras.layers.ReLU(name='relu2'),
        keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second conv block - balanced capacity
        keras.layers.Conv2D(
            96, (3, 3), padding='same',  # Increased from 64 to 96
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv3'
        ),
        keras.layers.ReLU(name='relu3'),
        keras.layers.Conv2D(
            96, (3, 3), padding='same',  # Additional conv layer
            kernel_initializer='he_normal', 
            use_bias=True,
            name='conv4'
        ),
        keras.layers.ReLU(name='relu4'),
        keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Depthwise separable for efficiency - increased capacity
        keras.layers.DepthwiseConv2D(
            (3, 3), padding='same',
            depthwise_initializer='he_normal',
            use_bias=True,
            name='depthwise1'
        ),
        keras.layers.ReLU(name='depthwise_relu1'),
        keras.layers.Conv2D(
            128, (1, 1),  # Increased from 64 to 128
            kernel_initializer='he_normal',
            use_bias=True,
            name='pointwise1'
        ),
        keras.layers.ReLU(name='relu5'),
        
        # Global context before classification
        keras.layers.GlobalAveragePooling2D(name='gap'),
        
        # Enhanced classification head
        keras.layers.Dense(192, activation='relu',  # Increased from 128 to 192
                             kernel_initializer='he_normal',
                             use_bias=True,
                             name='dense1'),
        keras.layers.Dropout(0.2, name='dropout1'),  # Light dropout for regularization
        keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
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