# models/esp_optimized_cnn.py
"""
esp_optimized_cnn – Depthwise Separable CNN for ESP-DL (<100KB)
================================================================
Design goal: Achieve <100 KB after INT8 quantization with guaranteed
ESP-DL operator compatibility. Uses depthwise separable mid-blocks to
reduce parameters, without BatchNorm for broadest hardware support.

Architecture:
  - Conv2D(16) + ReLU + MaxPool                         → entry block
  - DepthwiseConv2D(3×3) + ReLU + Conv2D(32, 1×1) + ReLU + MaxPool → DW-Sep block 1
  - DepthwiseConv2D(3×3) + ReLU + Conv2D(64, 1×1) + ReLU + MaxPool → DW-Sep block 2
  - GlobalAveragePooling2D
  - Dense(64) + ReLU → Dense(NB_CLASSES) Softmax

Notes:
  - No BatchNorm (explicit design choice for ESP-DL compatibility)
  - Standard ReLU (no ReLU6 clip)
  - No QAT wrapper

Estimated: ~50–70K parameters → ~50 KB after INT8 quantization.
"""

import parameters as params
from utils.keras_helper import keras

def create_esp_optimized_cnn():
    """
    ESP-DL optimized model with guaranteed operator compatibility
    and small footprint (<100KB quantized)
    """
    model = keras.Sequential([
        # Input
        keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First conv - replace BatchNorm with simple scaling
        keras.layers.Conv2D(
            16, (3, 3), padding='same',
            kernel_initializer='he_normal',
            name='conv1'
        ),
        keras.layers.ReLU(name='relu1'),
        keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Depthwise separable conv 1
        keras.layers.DepthwiseConv2D(
            (3, 3), padding='same',
            depthwise_initializer='he_normal',
            name='depthwise1'
        ),
        keras.layers.ReLU(name='depthwise_relu1'),
        keras.layers.Conv2D(
            32, (1, 1),
            kernel_initializer='he_normal',
            name='pointwise1'
        ),
        keras.layers.ReLU(name='relu2'),
        keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Depthwise separable conv 2
        keras.layers.DepthwiseConv2D(
            (3, 3), padding='same',
            depthwise_initializer='he_normal',
            name='depthwise2'
        ),
        keras.layers.ReLU(name='depthwise_relu2'),
        keras.layers.Conv2D(
            64, (1, 1),
            kernel_initializer='he_normal',
            name='pointwise2'
        ),
        keras.layers.ReLU(name='relu3'),
        keras.layers.MaxPooling2D((2, 2), name='pool3'),
        
        # Classification head - reduced size for ESP32
        keras.layers.GlobalAveragePooling2D(name='gap'),
        keras.layers.Dense(64, activation='relu', 
                             kernel_initializer='he_normal',
                             name='dense1'),
        keras.layers.Dense(params.NB_CLASSES, name='output', activation='softmax')
    ])
    
    return model