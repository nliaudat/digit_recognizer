# models/dig_class100_s2.py
"""
dig_class100_s2 – Haverland Reference Architecture (100-Class)
===============================================================
Design goal: Replicate the architecture from the Haverland meter-digit
notebook for 100-class (two-digit) recognition. Used as a reference
baseline to compare against custom-built models.

Reference:
  https://github.com/haverland/Tenth-of-step-of-a-meter-digit/blob/master/dig-class100-s2.ipynb

Architecture:
  - Conv2D × 2 (DIG_CLASS100_FILTERS[0]) + BN + ReLU + MaxPool + Dropout
  - Conv2D × 2 (DIG_CLASS100_FILTERS[1]) + BN + ReLU + MaxPool + Dropout
  - Conv2D × 2 (DIG_CLASS100_FILTERS[2]) + BN + ReLU + MaxPool + Dropout
  - Flatten → Dense(DIG_CLASS100_DENSE_UNITS) + BN + ReLU + Dropout
  - Dense(NB_CLASSES) Softmax

Notes:
  - Filter counts and Dense units driven by params.DIG_CLASS100_FILTERS / DENSE_UNITS
  - Uses Flatten (not GAP) — not quantization-optimal but follows the reference exactly
  - No QAT wrapper

Estimated: ~500K+ parameters depending on filter config → reference only, not IoT.
"""

import parameters as params
from utils.keras_helper import keras

def create_dig_class100_s2():
    """
    Model architecture from the referenced notebook:
    https://github.com/haverland/Tenth-of-step-of-a-meter-digit/blob/master/dig-class100-s2.ipynb
    """
    model = keras.Sequential([
        keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First conv block
        keras.layers.Conv2D(params.DIG_CLASS100_FILTERS[0], (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(params.DIG_CLASS100_FILTERS[0], (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(params.DROPOUT_RATE),
        
        # Second conv block
        keras.layers.Conv2D(params.DIG_CLASS100_FILTERS[1], (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(params.DIG_CLASS100_FILTERS[1], (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(params.DROPOUT_RATE),
        
        # Third conv block
        keras.layers.Conv2D(params.DIG_CLASS100_FILTERS[2], (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(params.DIG_CLASS100_FILTERS[2], (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(params.DROPOUT_RATE),
        
        # Flatten and dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(params.DIG_CLASS100_DENSE_UNITS, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(params.DROPOUT_RATE),
        keras.layers.Dense(params.NB_CLASSES, activation='softmax')
    ])
    
    return model