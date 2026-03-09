# models/esp_ultra_light.py
"""
esp_ultra_light – Sub-50KB Ultra-Minimal ESP32 CNN
===================================================
Design goal: Smallest possible model that still produces a valid classification.
Two tiny conv blocks (8 and 16 filters) feed directly into GlobalAveragePooling
→ output, with no Dense hidden layer at all.

Architecture:
  - Conv2D(8, 3×3) + ReLU + MaxPool    → entry (minimal feature extraction)
  - Conv2D(16, 3×3) + ReLU + MaxPool   → additional feature refinement
  - GlobalAveragePooling2D → Dense(NB_CLASSES) Softmax

Notes:
  - No BatchNorm, no Dropout (absolute minimum graph complexity)
  - Standard ReLU (ESP-DL safe)
  - Very low capacity; expected to underfit on most tasks
  - Useful as a size lower-bound reference for benchmarking
  - No QAT wrapper

Estimated: ~5K parameters → ~5 KB after INT8 quantization.
"""

import tensorflow as tf
import parameters as params

def create_esp_ultra_light():
    """
    Ultra-light model for ESP32 with minimal memory footprint
    Target: <50KB quantized
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # Tiny feature extractor
        tf.keras.layers.Conv2D(
            8, (3, 3), padding='same', activation='relu',
            kernel_initializer='he_normal',
            name='conv1'
        ),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        tf.keras.layers.Conv2D(
            16, (3, 3), padding='same', activation='relu', 
            kernel_initializer='he_normal',
            name='conv2'
        ),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Global average pooling instead of flatten + dense
        tf.keras.layers.GlobalAveragePooling2D(name='gap'),
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax', name='output')
    ])
    
    return model
