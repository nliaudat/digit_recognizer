# models/practical_tiny_depthwise.py
"""
practical_tiny_depthwise – Fixed Depthwise Separable CNN (~150–300KB)
======================================================================
Design goal: Practical baseline showing correct depthwise separable usage
(DepthwiseConv2D → ReLU → 1×1 Pointwise Conv). Earlier "tiny" variants had
too few filters and underfitted; this version fixes capacity at 32→64→128 filters.

Architecture:
  - Conv2D(32, 3×3) + ReLU + MaxPool                        → entry block
  - DepthwiseConv2D(3×3) + ReLU + Conv2D(64, 1×1) + ReLU + MaxPool → DW-Sep block 1
  - DepthwiseConv2D(3×3) + ReLU + Conv2D(128, 1×1) + ReLU + MaxPool → DW-Sep block 2
  - Flatten → Dense(256) + ReLU + Dropout(0.3) → Dense(NB_CLASSES) Softmax

Notes:
  - Uses Flatten (not GAP) → larger Dense head but simpler output shape
  - Standard relu activations (ESP-DL compatible)
  - No BatchNorm
  - No QAT wrapper

Estimated: ~200–350K parameters → suitable for benchmarking, not strict IoT.
"""

import parameters as params
from utils.keras_helper import keras

def create_practical_tiny_depthwise():
    """
    FIXED VERSION - Proper architecture that can actually learn
    """
    model = keras.Sequential([
        keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # First conv with sufficient capacity
        keras.layers.Conv2D(
            32, (3, 3), padding='same', activation='relu',
            kernel_initializer='he_normal',
        ),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Depthwise separable block 1 - increased capacity
        keras.layers.DepthwiseConv2D(
            (3, 3), padding='same', activation='relu',
            depthwise_initializer='he_normal',
        ),
        keras.layers.Conv2D(
            64, (1, 1), activation='relu',
            kernel_initializer='he_normal',
        ),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Depthwise separable block 2 - increased capacity  
        keras.layers.DepthwiseConv2D(
            (3, 3), padding='same', activation='relu',
            depthwise_initializer='he_normal',
        ),
        keras.layers.Conv2D(
            128, (1, 1), activation='relu',
            kernel_initializer='he_normal',
        ),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Adequate classification head
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu', 
                             kernel_initializer='he_normal'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(params.NB_CLASSES, activation='softmax')
    ])
    
    return model

def get_model_info():
    """Return ESP-DL compatible model information"""
    return {
        "name": "ESP-DL Compatible Depthwise CNN - FIXED",
        "description": "Fixed version with proper capacity for learning",
        "esp_dl_compatible": True,
        "expected_size": "150-300KB",
        "expected_accuracy": "95%+",
        "training_notes": "Use bias during training, ESP-DL handles quantization"
    }