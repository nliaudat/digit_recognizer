# models/simple_cnn_v2.py
"""
simple_cnn_v2 – LayerNorm CNN (Quantization-Friendlier Baseline)
=================================================================
Design goal: Replace BatchNormalization with LayerNormalization to improve
quantization stability on small batch sizes, while adding a third conv block
for deeper feature extraction than simple_cnn. Exploratory baseline.

Architecture:
  - Conv2D(FILTERS[0], 3×3) + ReLU + LayerNorm + MaxPool + Dropout(0.1)
  - Conv2D(FILTERS[1], 3×3) + ReLU + LayerNorm + MaxPool + Dropout(0.1)
  - Conv2D(FILTERS[1]×2, 3×3) + ReLU + GlobalAveragePooling2D + Dropout(0.2)
  - Dense(SIMPLE_CNN_DENSE_UNITS) + ReLU + Dropout(0.3) → Dense(NB_CLASSES) Softmax

Notes:
  - LayerNorm is more stable than BatchNorm at small batch sizes
  - However LayerNorm is NOT TFLite Micro / ESP-DL safe — prefer BN for IoT
  - Standard relu inline activation
  - Fully parameterized via params.SIMPLE_CNN_FILTERS + SIMPLE_CNN_DENSE_UNITS
  - No QAT wrapper

Estimated: ~30–100K parameters depending on filter config.
"""

import parameters as params
from utils.keras_helper import keras

def create_simple_cnn_v2():
    """TFLite-optimized CNN with better quantization support"""
    model = keras.Sequential([
        keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First conv block - replaced BatchNorm with simpler normalization
        keras.layers.Conv2D(params.SIMPLE_CNN_FILTERS[0], (3, 3), activation='relu', padding='same'),
        keras.layers.LayerNormalization(),  # Better for quantization than BatchNorm
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.1),  # Small dropout instead of heavy BatchNorm
        
        # Second conv block
        keras.layers.Conv2D(params.SIMPLE_CNN_FILTERS[1], (3, 3), activation='relu', padding='same'),
        keras.layers.LayerNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.1),
        
        # Optional third conv block for better feature extraction
        keras.layers.Conv2D(params.SIMPLE_CNN_FILTERS[1] * 2, (3, 3), activation='relu', padding='same'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.2),
        
        # Classification head
        keras.layers.Dense(params.SIMPLE_CNN_DENSE_UNITS, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model