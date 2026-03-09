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

import tensorflow as tf
import parameters as params

def create_esp_optimized_cnn():
    """
    ESP-DL optimized model with guaranteed operator compatibility
    and small footprint (<100KB quantized)
    """
    model = tf.keras.Sequential([
        # Input
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First conv - replace BatchNorm with simple scaling
        tf.keras.layers.Conv2D(
            16, (3, 3), padding='same',
            kernel_initializer='he_normal',
            name='conv1'
        ),
        tf.keras.layers.ReLU(name='relu1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Depthwise separable conv 1
        tf.keras.layers.DepthwiseConv2D(
            (3, 3), padding='same',
            depthwise_initializer='he_normal',
            name='depthwise1'
        ),
        tf.keras.layers.ReLU(name='depthwise_relu1'),
        tf.keras.layers.Conv2D(
            32, (1, 1),
            kernel_initializer='he_normal',
            name='pointwise1'
        ),
        tf.keras.layers.ReLU(name='relu2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Depthwise separable conv 2
        tf.keras.layers.DepthwiseConv2D(
            (3, 3), padding='same',
            depthwise_initializer='he_normal',
            name='depthwise2'
        ),
        tf.keras.layers.ReLU(name='depthwise_relu2'),
        tf.keras.layers.Conv2D(
            64, (1, 1),
            kernel_initializer='he_normal',
            name='pointwise2'
        ),
        tf.keras.layers.ReLU(name='relu3'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool3'),
        
        # Classification head - reduced size for ESP32
        tf.keras.layers.GlobalAveragePooling2D(name='gap'),
        tf.keras.layers.Dense(64, activation='relu', 
                             kernel_initializer='he_normal',
                             name='dense1'),
        tf.keras.layers.Dense(params.NB_CLASSES, name='output', activation='softmax')
    ])
    
    return model