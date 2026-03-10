# models/esp_haverland_compatible.py
"""
esp_haverland_compatible – ESP-DL Compatible Haverland Variant
==============================================================
Design goal: Adapt the classic Haverland meter-digit architecture to pass
the ESP-DL operator whitelist. Removes BatchNorm and complex activations;
keeps the dual-conv-block structure that made the original accurate.

Architecture:
  - Conv2D(32) × 2 + ReLU + MaxPool   → Block 1
  - Conv2D(64) × 2 + ReLU + MaxPool   → Block 2
  - Conv2D(64) + ReLU + MaxPool        → Block 3 (single conv, reduced)
  - Flatten → Dense(128) + ReLU → Dense(NB_CLASSES) Softmax

Notes:
  - No BatchNorm (removed for ESP-DL compatibility)
  - Standard ReLU (not ReLU6) for broadest ESP-DL support
  - Uses Flatten → Dense head (larger than GAP; not ideal for quantization)
  - No QAT wrapper

Estimated: ~150–250K parameters → ~150 KB after INT8 quantization.
"""

import tensorflow as tf
import parameters as params

def create_esp_haverland_compatible():
    """
    Modified Haverland architecture that maintains accuracy 
    while being ESP-DL compatible
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First conv block - keep structure but simplify
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', 
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Second conv block  
        tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Third conv block - reduced but sufficient
        tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Classification head - reduced but adequate
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu',
                             kernel_initializer='he_normal'),
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model
