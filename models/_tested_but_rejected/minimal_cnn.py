# models/minimal_cnn.py
"""
minimal_cnn – Bare-Minimum CNN Without Normalization
=====================================================
Design goal: Simplest possible CNN for quick iteration and ablation.
No BatchNorm, no complex activations — just Conv + ReLU + Pool + Dropout
directly to GlobalAveragePooling, used as a sanity-check baseline.

Architecture:
  - Conv2D(SIMPLE_CNN_FILTERS[0], 3×3) + ReLU + MaxPool + Dropout(0.2)
  - Conv2D(SIMPLE_CNN_FILTERS[1], 3×3) + ReLU + MaxPool + Dropout(0.2)
  - GlobalAveragePooling2D
  - Dense(SIMPLE_CNN_DENSE_UNITS) + ReLU + Dropout(0.3) → Dense(NB_CLASSES) Softmax

Notes:
  - Fully driven by params.SIMPLE_CNN_FILTERS and params.SIMPLE_CNN_DENSE_UNITS
  - No BatchNorm → simpler graph, but less stable training
  - Standard relu activation
  - No QAT wrapper

Estimated: ~20–60K parameters depending on filter config.
"""

import parameters as params
from utils.keras_helper import keras

def create_minimal_cnn():
    """Minimalist CNN without normalization layers"""
    model = keras.Sequential([
        keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First conv block - no BatchNorm
        keras.layers.Conv2D(params.SIMPLE_CNN_FILTERS[0], (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.2),
        
        # Second conv block - no BatchNorm  
        keras.layers.Conv2D(params.SIMPLE_CNN_FILTERS[1], (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.2),
        
        # Classification head
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(params.SIMPLE_CNN_DENSE_UNITS, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model