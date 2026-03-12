# models/esp_high_capacity.py
"""
esp_high_capacity – High-Capacity ESP-DL Compatible CNN
========================================================
Design goal: Maximize accuracy within the ESP-DL operator set by using
large filter counts (64 → 128 → 256) and standard ReLU, trading model
size for improved feature learning. Not aimed at IoT size constraints.

Architecture:
  - Conv2D(64) × 2 + ReLU + MaxPool    → Block 1 (wide entry)
  - Conv2D(128) × 2 + ReLU + MaxPool   → Block 2 (double capacity)
  - Conv2D(256) + ReLU                 → Block 3 (deep features)
  - GlobalAveragePooling2D
  - Dense(256) + ReLU → Dense(NB_CLASSES) Softmax

Notes:
  - No BatchNorm (ESP-DL compatible)
  - Standard ReLU throughout
  - Uses GAP instead of Flatten for better quantization compatibility
  - No QAT wrapper

Estimated: ~600K+ parameters → large; not intended for ESP32 deployment.
"""

import tensorflow as tf
import parameters as params

def create_esp_high_capacity():
    """
    Higher capacity model that can actually learn the task
    while maintaining ESP-DL compatibility
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # Expanded first layer
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', 
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Maintain capacity in middle layers
        tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Final conv layer
        tf.keras.layers.Conv2D(256, (3, 3), padding='same',
                              kernel_initializer='he_normal'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Adequate dense layer
        tf.keras.layers.Dense(256, activation='relu',
                             kernel_initializer='he_normal'),
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model
