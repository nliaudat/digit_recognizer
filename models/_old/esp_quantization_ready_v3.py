# models/esp_quantization_ready_v3.py
"""
esp_quantization_ready_v3 – L2-Regularized Depthwise CNN for ESP-DL
=====================================================================
Design goal: Add L2 kernel regularization to prevent overfitting while
keeping the depthwise separable architecture strictly compatible with
ESP-DL. Reduces Dense layers to a direct output after GAP for smaller size.

Architecture:
  - Conv2D(64, L2=0.001) + ReLU + MaxPool              → Block 1
  - Conv2D(128, L2=0.001) + ReLU + MaxPool             → Block 2
  - DepthwiseConv2D(L2=0.001) + Conv2D(128, 1×1, L2) + ReLU → DW-Sep block
  - GlobalAveragePooling2D
  - Dropout(0.2) → Dense(NB_CLASSES) Softmax  (no hidden Dense)

Notes:
  - L2 regularization on every conv/dw layer prevents overfitting on small datasets
  - No BatchNorm (ESP-DL compatible)
  - Standard ReLU throughout
  - No QAT wrapper; no hidden Dense layer (size-efficient head)

Estimated: ~250K parameters → ~250 KB after INT8 quantization.
"""

import tensorflow as tf
import parameters as params

def create_esp_quantization_ready_v3():
    """
    Improved version with better training characteristics
    - More filters for better feature extraction
    - L2 regularization to prevent overfitting
    - Dropout for better generalization
    - Still quantization-friendly
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First conv layer - increased filters
        tf.keras.layers.Conv2D(
            64, (3, 3), padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            use_bias=True,
            name='conv1'
        ),
        tf.keras.layers.ReLU(name='relu1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second conv layer
        tf.keras.layers.Conv2D(
            128, (3, 3), padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            use_bias=True,
            name='conv2'
        ),
        tf.keras.layers.ReLU(name='relu2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Third conv layer - depthwise separable for efficiency
        tf.keras.layers.DepthwiseConv2D(
            (3, 3), padding='same',
            depthwise_initializer='he_normal',
            depthwise_regularizer=tf.keras.regularizers.l2(0.001),
            use_bias=True,
            name='depthwise1'
        ),
        tf.keras.layers.ReLU(name='depthwise_relu1'),
        tf.keras.layers.Conv2D(
            128, (1, 1),
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            use_bias=True,
            name='pointwise1'
        ),
        tf.keras.layers.ReLU(name='relu3'),
        
        # Global average pooling
        tf.keras.layers.GlobalAveragePooling2D(name='gap'),
        
        # Dropout for regularization (removed during quantization)
        tf.keras.layers.Dropout(0.2, name='dropout1'),
        
        # Output layer
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model