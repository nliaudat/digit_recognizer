# models/esp_quantization_ready.py
"""
esp_quantization_ready – Minimal Depthwise CNN for Smooth INT8 Quantization
============================================================================
Design goal: Guarantee clean INT8 quantization by eliminating BatchNorm
and using only quantization-safe ops: explicit ReLU layers, he_normal init,
explicit bias, and GlobalAveragePooling. Smallest viable quantizable model.

Architecture:
  - Conv2D(32, 3×3) + ReLU + MaxPool         → Block 1
  - Conv2D(64, 3×3) + ReLU + MaxPool         → Block 2
  - DepthwiseConv2D(3×3) + Conv2D(64, 1×1) + ReLU → DW-Sep block (final)
  - GlobalAveragePooling2D → Dense(NB_CLASSES) Softmax  (no hidden Dense)

Notes:
  - No BatchNorm (intentionally removed for quantization graph cleanliness)
  - Explicit bias=True on all conv layers (important for quantizer)
  - Standard ReLU (not ReLU6); use v4/v15 for ReLU6-based models
  - No Dropout, no QAT wrapper

Estimated: ~70K parameters → ~70 KB after INT8 quantization.
"""

import tensorflow as tf
import parameters as params

def create_esp_quantization_ready():
    """
    Model specifically designed for smooth INT8 quantization
    - No BatchNormalization
    - All activations compatible with quantization
    - Symmetric weight distributions
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First conv with explicit activation
        tf.keras.layers.Conv2D(
            32, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,  # Important for quantization
            name='conv1'
        ),
        tf.keras.layers.ReLU(name='relu1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second conv
        tf.keras.layers.Conv2D(
            64, (3, 3), padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            name='conv2'
        ),
        tf.keras.layers.ReLU(name='relu2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Third conv - depthwise for efficiency
        tf.keras.layers.DepthwiseConv2D(
            (3, 3), padding='same',
            depthwise_initializer='he_normal',
            use_bias=True,
            name='depthwise1'
        ),
        tf.keras.layers.ReLU(name='depthwise_relu1'),
        tf.keras.layers.Conv2D(
            64, (1, 1),
            kernel_initializer='he_normal',
            use_bias=True,
            name='pointwise1'
        ),
        tf.keras.layers.ReLU(name='relu3'),
        
        # Global average pooling - quantization friendly
        tf.keras.layers.GlobalAveragePooling2D(name='gap'),
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model