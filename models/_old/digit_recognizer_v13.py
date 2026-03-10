# models/digit_recognizer_v13.py
"""
digit_recognizer_v13 – Micro-Bottleneck Residual IoT Model
===========================================================
Design goal: Drastically reduce MACs and parameter count vs v12 by replacing
plain residual blocks with 1×1-compress → 3×3-process → 1×1-expand
micro-bottleneck blocks (SqueezeNet/ResNet-50 style). ESP-DL + QAT ready.

Architecture:
  - Conv2D(16) + BN + ReLU + MaxPool            → lightweight entry
  - BottleneckBlock(32, ratio=0.5) × 2 + MaxPool → Stage 1 (16-ch bottleneck)
  - BottleneckBlock(64, ratio=0.25) × 2          → Stage 2 (16-ch bottleneck)
  - GlobalAveragePooling2D
  - Dense(32) + ReLU + Dropout(0.2) → Dense(NB_CLASSES) Softmax

BottleneckBlock: Conv2D(1×1 compress) → Conv2D(3×3 process) → Conv2D(1×1 expand)
  + identity shortcut with 1×1 projection when channels change.
  All layers followed by BN + ReLU; add shortcut → final ReLU.

Notes:
  - Standard ReLU throughout (ESP-DL compatible)
  - BatchNorm folds into Conv during INT8 quantization
  - QAT wrapper provided via create_qat_model()

Estimated: ~30K parameters → ~30 KB after INT8 quantization.
"""

import tensorflow as tf
import parameters as params

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError as e:
    QAT_AVAILABLE = False
    print(f"QAT not available: {e}")

def bottleneck_block(x, filters, reduction_ratio=0.5, strides=1):
    """
    ESP32-optimized Bottleneck block.
    Uses 1x1 convs to reduce channels before the expensive 3x3 conv,
    saving thousands of MACs without using unsupported SeparableConv2D.
    """
    shortcut = x
    reduced_filters = max(int(filters * reduction_ratio), 8)
    
    # Compress channels
    y = tf.keras.layers.Conv2D(reduced_filters, (1, 1), strides=1, padding='same', kernel_initializer='he_normal')(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    
    # Process features
    y = tf.keras.layers.Conv2D(reduced_filters, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    
    # Expand channels
    y = tf.keras.layers.Conv2D(filters, (1, 1), strides=1, padding='same', kernel_initializer='he_normal')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    
    # Identity shortcut
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), strides=strides, padding='same', kernel_initializer='he_normal')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        
    y = tf.keras.layers.add([shortcut, y])
    y = tf.keras.layers.ReLU()(y)
    return y

def create_digit_recognizer_v13():
    """
    Highly optimized IoT model utilizing Micro-Bottleneck blocks.
    Limits MACs and parameter-count drastically.
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Initial Conv layer
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    
    # Bottleneck Blocks - Stage 1
    x = bottleneck_block(x, filters=32, reduction_ratio=0.5, strides=1)
    x = bottleneck_block(x, filters=32, reduction_ratio=0.5, strides=1)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    
    # Bottleneck Blocks - Stage 2
    x = bottleneck_block(x, filters=64, reduction_ratio=0.25, strides=1)
    x = bottleneck_block(x, filters=64, reduction_ratio=0.25, strides=1)
    
    # Global Average Pooling mapped directly to classes (saves huge Dense layers)
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Small single dense projection
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, 
        activation='softmax', 
        name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="digit_recognizer_v13")

def create_qat_model(base_model=None):
    """
    Create QAT model using explicit annotations compatible with ESP-DL.
    """
    if not QAT_AVAILABLE:
        print("Warning: QAT not available for v13. Returning base model.")
        return base_model if base_model else create_digit_recognizer_v13()
    
    if base_model is None:
        base_model = create_digit_recognizer_v13()
    
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
        print(f"QAT failed on v13: {e}")
        return base_model

if __name__ == "__main__":
    model = create_digit_recognizer_v13()
    print(f"Created model: {model.name}")
    model.summary()
