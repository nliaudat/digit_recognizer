# models/mobilenet_style.py
"""
mobilenet_style – Depthwise Separable CNN (MobileNet-Inspired)
==============================================================
Design goal: Approximate MobileNet V1's depthwise separable structure with
a minimal Sequential implementation, using parameterized filter counts from
`params.SIMPLE_CNN_FILTERS`. Exploratory/baseline variant.

Architecture:
  - DepthwiseConv2D(3×3) + ReLU + Conv2D(FILTERS[0], 1×1) + ReLU + MaxPool + Dropout(0.1)
  - DepthwiseConv2D(3×3) + ReLU + Conv2D(FILTERS[1], 1×1) + ReLU + MaxPool + Dropout(0.1)
  - GlobalAveragePooling2D
  - Dense(SIMPLE_CNN_DENSE_UNITS) + ReLU + Dropout(0.3) → Dense(NB_CLASSES) Softmax

Notes:
  - No BatchNorm (simpler graph)
  - Standard relu activation (not ReLU6)
  - Fully parameterized via params.SIMPLE_CNN_FILTERS + SIMPLE_CNN_DENSE_UNITS
  - No QAT wrapper

Estimated: ~20–80K parameters depending on filter config.
"""

import tensorflow as tf
import parameters as params


def create_mobilenet_style():
    """MobileNet-style with depthwise separable convolutions"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First block - depthwise separable
        tf.keras.layers.DepthwiseConv2D((3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(params.SIMPLE_CNN_FILTERS[0], (1, 1), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.1),
        
        # Second block - depthwise separable
        tf.keras.layers.DepthwiseConv2D((3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(params.SIMPLE_CNN_FILTERS[1], (1, 1), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.1),
        
        # Classification head
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(params.SIMPLE_CNN_DENSE_UNITS, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model