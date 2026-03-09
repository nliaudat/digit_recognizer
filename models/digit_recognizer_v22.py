# models/digit_recognizer_v22.py
"""
digit_recognizer_v22 – Spatial-Aware MobileNetV2 for IoT (10-Class RGB)
========================================================================
Design: Based on the highly efficient v16 ESP-NN compatible model, upgraded with
Positional Encoding to preserve spatial coordinates for rotating dial recognition.

Key Features:
1. PositionalEncoding2D layer explicitly feeds X/Y grid coordinates.
2. Inverted Residual Bottlenecks (MobileNetV2 style) optimized for TFLite Micro.
3. Spatial Flattening instead of GAP to retain coordinate/rotation awareness.
4. Parameters fiercely contained to stay under 200KB INT8 capacity (~200K params limit).

Target: 10-Class RGB rotating dials.
"""

import os
import sys

# Support running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
import parameters as params

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Positional Encoding Layer (TFLite Micro/ESP-NN Compatible)
# ---------------------------------------------------------------------------

class PositionalEncoding2D(tf.keras.layers.Layer):
    """Explicitly injects X, Y coordinates as extra channels."""
    def __init__(self, name='pos_encoding', **kwargs):
        super().__init__(name=name, **kwargs)
        
    def build(self, input_shape):
        h, w = input_shape[1], input_shape[2]
        
        y_grid, x_grid = tf.meshgrid(
            tf.linspace(-1.0, 1.0, h),
            tf.linspace(-1.0, 1.0, w),
            indexing='ij'
        )
        
        y_grid = tf.expand_dims(tf.expand_dims(y_grid, 0), -1)  
        x_grid = tf.expand_dims(tf.expand_dims(x_grid, 0), -1)  
        
        self.y_coord = tf.cast(y_grid, tf.float32)
        self.x_coord = tf.cast(x_grid, tf.float32)
        
        self.y_scale = self.add_weight(name='y_scale', shape=(1,), initializer='ones', trainable=True)
        self.x_scale = self.add_weight(name='x_scale', shape=(1,), initializer='ones', trainable=True)
        
        super().build(input_shape)
    
    def call(self, inputs):
        y_coord = self.y_coord * self.y_scale
        x_coord = self.x_coord * self.x_scale
        
        batch_size = tf.shape(inputs)[0]
        y_coord = tf.tile(y_coord, [batch_size, 1, 1, 1])
        x_coord = tf.tile(x_coord, [batch_size, 1, 1, 1])
        
        return tf.concat([inputs, y_coord, x_coord], axis=-1)
    
    def get_config(self):
        return super().get_config()


# ---------------------------------------------------------------------------
# Inverted residual bottleneck (MobileNetV2 core block)
# ---------------------------------------------------------------------------

def _inv_res(x, filters_out, expand_ratio, stride, name_prefix):
    """MobileNetV2-style inverted residual block."""
    ch_in = x.shape[-1]
    ch_exp = ch_in * expand_ratio

    use_shortcut = (stride == 1 and ch_in == filters_out)

    # 1. Expand
    y = tf.keras.layers.Conv2D(
        ch_exp, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_expand'
    )(x)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_exp_bn')(y)
    y = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_exp_relu6')(y)

    # 2. Depthwise
    y = tf.keras.layers.DepthwiseConv2D(
        (3, 3), strides=stride, padding='same',
        depthwise_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_dw'
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_dw_bn')(y)
    y = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_dw_relu6')(y)

    # 3. Project
    y = tf.keras.layers.Conv2D(
        filters_out, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_project'
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_proj_bn')(y)

    # 4. Shortcut
    if use_shortcut:
        y = tf.keras.layers.Add(name=f'{name_prefix}_add')([x, y])

    return y


# ---------------------------------------------------------------------------
# Main Model Builder
# ---------------------------------------------------------------------------

def create_digit_recognizer_v22():
    """
    Builds the spatial-aware v16 iteration for IoT hardware limits (<200KB).
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # ── Positional Injection ──
    x = PositionalEncoding2D(name='pos_encoding')(inputs)

    # Stem
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='entry_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='entry_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='entry_relu6')(x)

    # Inverted residual stages
    # Carefully configured to drop spatial map down to 4x4 while expanding features
    # config: (out_ch, expand_ratio, stride)
    inv_res_config = [
        (16,  4, 2),   # 32x32 -> 16x16
        (16,  4, 1),   # 16x16
        (24,  4, 2),   # 16x16 -> 8x8
        (24,  4, 1),   # 8x8
        (40,  4, 2),   # 8x8 -> 4x4
        (40,  6, 1),   # 4x4
        (64,  6, 1),   # 4x4 spatial preservation
    ]
    
    for i, (out_ch, expand, stride) in enumerate(inv_res_config):
        x = _inv_res(x, filters_out=out_ch, expand_ratio=expand, stride=stride,
                     name_prefix=f'ir{i+1}')

    # Final spatial fusion prior to flatten
    x = tf.keras.layers.Conv2D(
        64, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='head_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='head_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='head_relu6')(x)

    # ── CRITICAL: FLATTEN SPATIAL MAP ──
    # The map is 4x4x64 at this point = 1024 features.
    x = tf.keras.layers.Flatten(name='spatial_flatten')(x)
    
    # Tiny bottleneck classifier
    x = tf.keras.layers.Dropout(0.3, name='dropout_pre')(x)
    x = tf.keras.layers.Dense(32, use_bias=False, kernel_initializer='he_normal', name='fc1')(x)
    x = tf.keras.layers.BatchNormalization(name='fc1_bn')(x)
    x = tf.keras.layers.ReLU(name='fc1_relu')(x)
    x = tf.keras.layers.Dropout(0.2, name='dropout_post')(x)

    # Output (ensure default 10 classes)
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, activation='softmax', name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name='digit_recognizer_v22')


# ---------------------------------------------------------------------------
# QAT wrapper
# ---------------------------------------------------------------------------

def create_qat_model(base_model=None):
    if base_model is None:
        base_model = create_digit_recognizer_v22()
    if not QAT_AVAILABLE:
        print("Warning: QAT not available.")
        return base_model
    try:
        # Quantize injecting the custom layer scope
        with tfmot.quantization.keras.quantize_scope(
            {'PositionalEncoding2D': PositionalEncoding2D}
        ):
            from tensorflow_model_optimization.quantization.keras import quantize_annotate_model
            from tensorflow_model_optimization.quantization.keras import quantize_apply
            annotated = quantize_annotate_model(base_model)
            qat_model = quantize_apply(annotated)
            
        print("✓ QAT model created successfully.")
        return qat_model
    except Exception as e:
        print(f"✗ QAT failed ({e}) – returning base model.")
        return base_model


if __name__ == "__main__":
    # Ensure parameter alignment for testing
    if not hasattr(params, 'NB_CLASSES'):
        params.NB_CLASSES = 10
    if not hasattr(params, 'INPUT_SHAPE'):
        params.INPUT_SHAPE = (32, 32, 3)
        
    print("=" * 70)
    print("DIGIT RECOGNIZER V22 - IoT SPATIAL MOBILE-NET (<200KB INT8 target)")
    print("=" * 70)
    
    m = create_digit_recognizer_v22()
    m.summary()
    
    p = m.count_params()
    print("\n" + "=" * 70)
    print("PARAMETER SIZE CHECK")
    print("=" * 70)
    print(f"Total parameters : {p:,}")
    print(f"Estimated INT8   : {p * 1.1 / 1024:.2f} KB")
    
    print("\n" + "=" * 70)
    print("TFLITE ESP-NN COMPATIBILITY")
    print("=" * 70)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(m)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        def representative_dataset():
            for _ in range(10):
                yield [np.random.randn(1, *params.INPUT_SHAPE).astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        tflite_model = converter.convert()
        print("✓ Model successfully quantized to TFLite INT8!")
        print(f"  Final Output Size: {len(tflite_model) / 1024:.2f} KB")
    except Exception as e:
        print(f"✗ TFLite conversion failed: {e}")
