# models/digit_recognizer_v14.py
"""
digit_recognizer_v14 – Ultra-Tiny Bottleneck Model (Pareto Frontier)
=====================================================================
Design goal: Push the absolute minimum parameter count for ESP32 while
retaining enough capacity to learn digit features. Successor to v13 with
an even more aggressive bottleneck ratio (0.25 vs 0.5) and no intermediate
Dense layer — GAP maps directly to output logits.

Architecture:
  - Conv2D(8) + BN + ReLU + MaxPool              → minimal entry (8 filters)
  - BottleneckBlock(16, ratio=0.5) + MaxPool      → Stage 1 (≥8-ch bottleneck)
  - BottleneckBlock(32, ratio=0.25) + MaxPool     → Stage 2 (8-ch bottleneck)
  - GlobalAveragePooling2D → Dense(NB_CLASSES) Softmax  (no hidden Dense)

Notes:
  - Removing the intermediate Dense layer saves ~16K params vs v13
  - Very low capacity; may underfit on harder datasets or 100-class tasks
  - Standard ReLU + BN throughout (ESP-DL / TFLite Micro safe)
  - QAT wrapper provided via create_qat_model()

Estimated: ~8–12K parameters → ~10 KB after INT8 quantization.
"""

import parameters as params
from utils.keras_helper import keras

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError as e:
    QAT_AVAILABLE = False
    print(f"QAT not available: {e}")

def bottleneck_block(x, filters, reduction_ratio=0.25, strides=1):
    """
    Ultra-compressed ESP32-optimized Bottleneck block.
    """
    shortcut = x
    reduced_filters = max(int(filters * reduction_ratio), 8)
    
    # Compress channels
    y = keras.layers.Conv2D(reduced_filters, (1, 1), strides=1, padding='same', kernel_initializer='he_normal')(x)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.ReLU()(y)
    
    # Process features
    y = keras.layers.Conv2D(reduced_filters, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.ReLU()(y)
    
    # Expand channels
    y = keras.layers.Conv2D(filters, (1, 1), strides=1, padding='same', kernel_initializer='he_normal')(y)
    y = keras.layers.BatchNormalization()(y)
    
    # Identity shortcut
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(filters, (1, 1), strides=strides, padding='same', kernel_initializer='he_normal')(shortcut)
        shortcut = keras.layers.BatchNormalization()(shortcut)
        
    y = keras.layers.add([shortcut, y])
    y = keras.layers.ReLU()(y)
    return y

def create_digit_recognizer_v14():
    """
    Ultra-Tiny variant pushing the absolute Pareto frontier for ESP32 constraints.
    Focuses purely on the minimal possible parameter count utilizing global pooling.
    """
    inputs = keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Initial Conv layer - minimal filters
    x = keras.layers.Conv2D(8, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    
    # Minimal Bottleneck Blocks - Stage 1
    x = bottleneck_block(x, filters=16, reduction_ratio=0.5, strides=1)
    x = keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    
    # Minimal Bottleneck Blocks - Stage 2
    x = bottleneck_block(x, filters=32, reduction_ratio=0.25, strides=1)
    
    # Global Average Pooling directly to logits
    x = keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # No intermediate dense layer! Maps directly to output.
    outputs = keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return keras.Model(inputs, outputs, name="digit_recognizer_v14")

def create_qat_model(base_model=None):
    """
    Create QAT model using explicit annotations compatible with ESP-DL.
    """
    if not QAT_AVAILABLE:
        print("Warning: QAT not available for v14. Returning base model.")
        return base_model if base_model else create_digit_recognizer_v14()
    
    if base_model is None:
        base_model = create_digit_recognizer_v14()
    
    try:
        quantize_annotate = tfmot.quantization.keras.quantize_annotate
        quantize_apply = tfmot.quantization.keras.quantize_apply
        quantize_scope = tfmot.quantization.keras.quantize_scope
        
        with quantize_scope():
            annotated_model = quantize_annotate(base_model)
            qat_model = quantize_apply(
                annotated_model,
                tfmot.experimental.combine.Default8BitClusterPreset()
            )
            
        return qat_model
        
    except Exception as e:
        print(f"QAT failed on v14: {e}")
        return base_model

if __name__ == "__main__":
    model = create_digit_recognizer_v14()
    print(f"Created model: {model.name}")
    model.summary()
