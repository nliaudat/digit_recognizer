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

import parameters as params
from utils.keras_helper import keras

def create_mobilenet_style():
    """MobileNet-style with depthwise separable convolutions"""
    model = keras.Sequential([
        keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First block - depthwise separable
        keras.layers.DepthwiseConv2D((3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(params.SIMPLE_CNN_FILTERS[0], (1, 1), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.1),
        
        # Second block - depthwise separable
        keras.layers.DepthwiseConv2D((3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(params.SIMPLE_CNN_FILTERS[1], (1, 1), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.1),
        
        # Classification head
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(params.SIMPLE_CNN_DENSE_UNITS, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model