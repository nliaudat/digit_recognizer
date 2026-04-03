# models/digit_recognizer_v20.py
"""
digit_recognizer_v20 – Spatial-Aware SE-GhostNet with Positional Encoding
===========================================================================
Design: Maximizes accuracy by explicitly preserving and enhancing spatial
information critical for rotating digit recognition.

CRITICAL INSIGHT:
For digits 0-99 rotating through a circle, the Y-coordinate of the digit
in the image is directly correlated with the class. GAP destroys this.
Flatten() preserves it, allowing the network to memorize geometric offsets.
This version explicitly injects 2D coordinate maps to guarantee the filters
can "see" their location.

Target: >99% accuracy on 100-class rotating digits.
Size Limit: <1.5MB INT8 footprint
"""

import os
import sys

# Support running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import parameters as params
from utils.keras_helper import keras

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Positional Encoding Layer (TFLite Compatible)
# ---------------------------------------------------------------------------

class PositionalEncoding2D(keras.layers.Layer):
    """
    Add explicit positional information to feature maps.
    This helps the network understand "where" in the image features are located.
    Designed exclusively with TFLite Micro/ESP-NN compatible tensor operations.
    """
    def __init__(self, name='pos_encoding', **kwargs):
        super().__init__(name=name, **kwargs)
        
    def build(self, input_shape):
        h, w, c = input_shape[1], input_shape[2], input_shape[3]
        
        # Create coordinate grids
        y_grid, x_grid = keras.ops.meshgrid(
            keras.ops.linspace(-1.0, 1.0, h),
            keras.ops.linspace(-1.0, 1.0, w),
            indexing='ij'
        )
        
        # Expand dimensions for broadcasting: 1, h, w, 1
        y_grid = keras.ops.expand_dims(keras.ops.expand_dims(y_grid, 0), -1)  
        x_grid = keras.ops.expand_dims(keras.ops.expand_dims(x_grid, 0), -1)  
        
        # Register as non-trainable buffers
        self.y_coord = keras.ops.cast(y_grid, 'float32')
        self.x_coord = keras.ops.cast(x_grid, 'float32')
        
        # Learnable scaling factors for coordinates
        self.y_scale = self.add_weight(
            name='y_scale',
            shape=(1,),
            initializer='ones',
            trainable=True
        )
        self.x_scale = self.add_weight(
            name='x_scale',
            shape=(1,),
            initializer='ones',
            trainable=True
        )
        
        super().build(input_shape)
    
    def call(self, inputs):
        # Scale coordinates
        y_coord = self.y_coord * self.y_scale
        x_coord = self.x_coord * self.x_scale
        
        # Tile coordinates to match batch size
        batch_size = keras.ops.shape(inputs)[0]
        y_coord = keras.ops.tile(y_coord, [batch_size, 1, 1, 1])
        x_coord = keras.ops.tile(x_coord, [batch_size, 1, 1, 1])
        
        # Concatenate coordinates with input features
        return keras.ops.concatenate([inputs, y_coord, x_coord], axis=-1)
    
    def get_config(self):
        config = super().get_config()
        return config


# ---------------------------------------------------------------------------
# Enhanced SE Block with Spatial Attention
# ---------------------------------------------------------------------------

def _se_block_spatial(x, reduction=8, name_prefix='se'):
    """SE block with both channel and spatial attention."""
    channels = x.shape[-1]
    
    # — Channel Attention (Squeeze and Excitation) —
    channel_se = keras.layers.GlobalAveragePooling2D(name=f'{name_prefix}_gap')(x)
    channel_se = keras.layers.Dense(
        max(4, channels // reduction),
        activation='relu',
        name=f'{name_prefix}_fc1'
    )(channel_se)
    channel_se = keras.layers.Dense(
        channels,
        activation='sigmoid',
        name=f'{name_prefix}_fc2'
    )(channel_se)
    channel_se = keras.layers.Reshape((1, 1, channels), name=f'{name_prefix}_reshape')(channel_se)
    
    # Apply channel attention
    x = keras.layers.Multiply(name=f'{name_prefix}_channel_scale')([x, channel_se])
    
    # — Spatial Attention —
    # Focuses on WHERE the important features are in the receptive field
    spatial = keras.layers.Conv2D(
        1, (1, 1), padding='same',
        activation='sigmoid',
        name=f'{name_prefix}_spatial_conv'
    )(x)
    
    # Apply spatial attention
    x = keras.layers.Multiply(name=f'{name_prefix}_spatial_scale')([x, spatial])
    
    return x


# ---------------------------------------------------------------------------
# Spatial-Preserving Ghost Module
# ---------------------------------------------------------------------------

def _ghost_module_spatial(x, out_channels, ratio=2, dw_kernel=3, use_se=True, 
                          use_position=False, name_prefix='gm'):
    """Ghost module with optional dual-attention and positional awareness."""
    intrinsic_ch = out_channels // ratio
    ghost_ch = out_channels - intrinsic_ch
    
    # Inject spatial awareness directly into the block via coordinates
    if use_position:
        x = PositionalEncoding2D(name=f'{name_prefix}_pos')(x)
        # Adapt channels back down immediately to save parameters
        current_channels = x.shape[-1]
        x = keras.layers.Conv2D(
            current_channels - 2, (1, 1), padding='same',
            kernel_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_pos_adapt'
        )(x)
        x = keras.layers.BatchNormalization(name=f'{name_prefix}_pos_bn')(x)
    
    # Primary convolution
    primary = keras.layers.Conv2D(
        intrinsic_ch, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_primary_conv'
    )(x)
    primary = keras.layers.BatchNormalization(name=f'{name_prefix}_primary_bn')(primary)
    primary = keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_primary_relu')(primary)
    
    # Ghost features
    ghost = keras.layers.DepthwiseConv2D(
        (dw_kernel, dw_kernel), padding='same',
        depthwise_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_ghost_dw'
    )(primary)
    ghost = keras.layers.BatchNormalization(name=f'{name_prefix}_ghost_bn')(ghost)
    ghost = keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_ghost_relu')(ghost)
    
    if ghost_ch != intrinsic_ch:
        ghost = keras.layers.Conv2D(
            ghost_ch, (1, 1), padding='same',
            kernel_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_ghost_adjust'
        )(ghost)
    
    concat = keras.layers.Concatenate(name=f'{name_prefix}_concat')([primary, ghost])
    
    if use_se:
        concat = _se_block_spatial(concat, reduction=8, name_prefix=f'{name_prefix}_se')
    
    return concat


# ---------------------------------------------------------------------------
# Spatial Ghost Bottleneck
# ---------------------------------------------------------------------------

def _ghost_block_spatial(x, out_channels, stride=1, expansion=2, use_se=True,
                         use_position=False, name_prefix='gb'):
    ch_in = x.shape[-1]
    hidden_dim = ch_in * expansion
    
    y = _ghost_module_spatial(
        x, hidden_dim, use_se=False,
        use_position=use_position,
        name_prefix=f'{name_prefix}_expand'
    )
    
    if stride > 1:
        y = keras.layers.DepthwiseConv2D(
            (3, 3), strides=stride, padding='same',
            depthwise_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_stride_dw'
        )(y)
        y = keras.layers.BatchNormalization(name=f'{name_prefix}_stride_bn')(y)
        y = keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_stride_relu')(y)
    
    y = _ghost_module_spatial(
        y, out_channels, use_se=use_se,
        use_position=False,  # Position already added in expand
        name_prefix=f'{name_prefix}_project'
    )
    
    if stride == 1 and ch_in == out_channels:
        shortcut = x
    else:
        shortcut = keras.layers.DepthwiseConv2D(
            (3, 3), strides=stride, padding='same',
            depthwise_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_sc_dw'
        )(x)
        shortcut = keras.layers.BatchNormalization(name=f'{name_prefix}_sc_bn')(shortcut)
        shortcut = keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_sc_relu')(shortcut)
        shortcut = keras.layers.Conv2D(
            out_channels, (1, 1), padding='same',
            kernel_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_sc_conv'
        )(shortcut)
        shortcut = keras.layers.BatchNormalization(name=f'{name_prefix}_sc_conv_bn')(shortcut)
    
    y = keras.layers.Add(name=f'{name_prefix}_add')([shortcut, y])
    y = keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_relu')(y)
    return y


# ---------------------------------------------------------------------------
# Spatial Feature Pyramid for Multi-Scale Preservation (ESP-NN Fixed)
# ---------------------------------------------------------------------------

def _spatial_pyramid(x, name_prefix='pyramid'):
    # Branch 1: Original scale
    b1 = keras.layers.Conv2D(
        64, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_b1_conv'
    )(x)
    b1 = keras.layers.BatchNormalization(name=f'{name_prefix}_b1_bn')(b1)
    
    # Branch 2: 3x3 context
    b2 = keras.layers.Conv2D(
        64, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_b2_conv'
    )(x)
    b2 = keras.layers.BatchNormalization(name=f'{name_prefix}_b2_bn')(b2)
    
    # Branch 3: Broad context (Fixed: standard 5x5 instead of dilated 3x3 for ESP-NN capability)
    b3 = keras.layers.DepthwiseConv2D(
        (5, 5), padding='same',
        depthwise_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_b3_dw'
    )(x)
    b3 = keras.layers.BatchNormalization(name=f'{name_prefix}_b3_bn')(b3)
    b3 = keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_b3_relu')(b3)
    b3 = keras.layers.Conv2D(
        64, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_b3_conv'
    )(b3)
    b3 = keras.layers.BatchNormalization(name=f'{name_prefix}_b3_conv_bn')(b3)
    
    concat = keras.layers.Concatenate(name=f'{name_prefix}_concat')([b1, b2, b3])
    concat = keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_output_relu')(concat)
    
    return concat


# ---------------------------------------------------------------------------
# Main Model Builder
# ---------------------------------------------------------------------------

def create_digit_recognizer_v20():
    inputs = keras.Input(shape=params.INPUT_SHAPE, name='input')
    
    # Inject coordinate maps explicitly into the image!
    x = PositionalEncoding2D(name='input_pos')(inputs)
    
    x = keras.layers.Conv2D(
        32, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='stem_conv'
    )(x)
    x = keras.layers.BatchNormalization(name='stem_bn')(x)
    x = keras.layers.ReLU(max_value=6.0, name='stem_relu')(x)
    
    # (out_channels, stride, expansion, use_se, use_position)
    blocks = [
        (32, 1, 2, False, True),  
        (32, 2, 2, False, True),  
        (48, 1, 3, True, True),   
        (64, 2, 3, True, True),   
        (64, 1, 3, True, True),   
        (80, 1, 4, True, True),   
        (96, 2, 4, True, True),   
        (96, 1, 4, True, True),   
        (128, 1, 4, True, True),  
    ]
    
    for i, (out_ch, stride, expansion, use_se, use_pos) in enumerate(blocks):
        x = _ghost_block_spatial(
            x, out_channels=out_ch,
            stride=stride,
            expansion=expansion,
            use_se=use_se,
            use_position=use_pos,
            name_prefix=f'block{i+1}'
        )
    
    # Multi-scale spatial pyramid before flattening
    x = _spatial_pyramid(x, name_prefix='pyramid')
    
    # Final 1x1 convolution
    x = keras.layers.Conv2D(
        128, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='spatial_fusion'
    )(x)
    x = keras.layers.BatchNormalization(name='spatial_fusion_bn')(x)
    x = keras.layers.ReLU(max_value=6.0, name='spatial_fusion_relu')(x)
    
    # — CRITICAL: FLATTEN TO PRESERVE SPATIAL COORDINATES —
    x = keras.layers.Flatten(name='spatial_flatten')(x)
    
    # — CLASSIFIER —
    # Single resilient bottleneck sized for the 1.5MB headroom
    x = keras.layers.Dropout(0.4, name='dropout_pre')(x)
    x = keras.layers.Dense(256, use_bias=False, kernel_initializer='he_normal', name='fc1')(x)
    x = keras.layers.BatchNormalization(name='fc1_bn')(x)
    x = keras.layers.ReLU(name='fc1_relu')(x)
    x = keras.layers.Dropout(0.3, name='dropout_post')(x)
    
    outputs = keras.layers.Dense(
        params.NB_CLASSES, activation='softmax', name='output'
    )(x)
    
    model = keras.Model(inputs, outputs, name='digit_recognizer_v20')
    return model


def create_qat_model(base_model=None):
    if base_model is None:
        base_model = create_digit_recognizer_v20()
    if not QAT_AVAILABLE:
        print("Warning: QAT not available")
        return base_model
    try:
        # Quantize the model
        with tfmot.quantization.keras.quantize_scope(
            {'PositionalEncoding2D': PositionalEncoding2D}
        ):
            from tensorflow_model_optimization.quantization.keras import quantize_annotate_model
            from tensorflow_model_optimization.quantization.keras import quantize_apply
            annotated_model = quantize_annotate_model(base_model)
            qat_model = quantize_apply(annotated_model)
        
        print("✓ QAT model created successfully")
        return qat_model
    except Exception as e:
        print(f"✗ QAT failed: {e}")
        return base_model

if __name__ == "__main__":
    
    # Create model
    m = create_digit_recognizer_v20()
    m.summary()
    
    # Parameter count
    p = m.count_params()
    
    print("\n" + "=" * 70)
    print("PARAMETER SIZE CHECK")
    print("=" * 70)
    print(f"Total parameters      : {p:,}")
    print(f"INT8 footprint        : {p * 1.1 / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 70)
    print("DIGIT RECOGNIZER V20 - SPATIAL-AWARE WITH POSITIONAL ENCODING")
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
        print(f"  Final Output Size: {len(tflite_model) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"✗ TFLite conversion failed: {e}")
