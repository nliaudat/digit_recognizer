# models/simple_cnn.py
"""
simple_cnn – BatchNorm CNN Baseline for Testing
================================================
Design goal: Minimal CNN with BatchNormalization for quick testing of
training pipeline integrity. Two conv+BN blocks followed by GAP + Dense.
Serves as the simplest BN-based sanity check.

Architecture:
  - Conv2D(SIMPLE_CNN_FILTERS[0], 3×3) + ReLU + BN + MaxPool
  - Conv2D(SIMPLE_CNN_FILTERS[1], 3×3) + ReLU + BN + MaxPool
  - GlobalAveragePooling2D
  - Dense(SIMPLE_CNN_DENSE_UNITS) + ReLU + Dropout(0.3) → Dense(NB_CLASSES) Softmax

Notes:
  - BatchNorm after ReLU (unusual order; consider BN before ReLU for QAT)
  - Standard relu inline activation
  - Fully parameterized via params.SIMPLE_CNN_FILTERS + SIMPLE_CNN_DENSE_UNITS
  - No QAT wrapper

Estimated: ~20–60K parameters depending on filter config.
"""

import parameters as params
from utils.keras_helper import keras

def create_simple_cnn():
    """Simple CNN model for testing"""
    model = keras.Sequential([
        keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First conv block
        keras.layers.Conv2D(params.SIMPLE_CNN_FILTERS[0], (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Second conv block  
        keras.layers.Conv2D(params.SIMPLE_CNN_FILTERS[1], (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Classification head
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(params.SIMPLE_CNN_DENSE_UNITS, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model