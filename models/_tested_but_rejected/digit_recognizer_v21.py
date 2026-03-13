# models/digit_recognizer_v21.py
"""
digit_recognizer_v21 – Advanced Spatial-Aware SE-GhostNet with Rotary Position Encoding
=======================================================================================
Design: Enhanced version with rotary position encoding and adaptive spatial attention
for maximum accuracy on 100-class rotating digits. Optimized for PC/GPU usage 
(not bound by ESP32 TFLite constraints).

Key innovations over v20:
1. Rotary position encoding (better for rotational invariance)
2. Adaptive spatial attention with learnable temperature
3. Multi-scale feature aggregation before flattening
4. Progressive channel expansion with spatial preservation

Target: >99.5% accuracy
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
# Rotary Positional Encoding (Better for Rotational Tasks)
# ---------------------------------------------------------------------------

class RotaryPositionalEncoding2D(tf.keras.layers.Layer):
    """
    Rotary position encoding - better for rotational invariance than Cartesian.
    Encodes angles and radii relative to image center.
    """
    def __init__(self, name='rotary_pos', **kwargs):
        super().__init__(name=name, **kwargs)
        
    def build(self, input_shape):
        h, w, c = input_shape[1], input_shape[2], input_shape[3]
        
        # Create coordinate grids centered at 0
        y_grid, x_grid = tf.meshgrid(
            tf.linspace(-1.0, 1.0, h),
            tf.linspace(-1.0, 1.0, w),
            indexing='ij'
        )
        
        # Convert to polar coordinates (better for rotation)
        radius = tf.sqrt(x_grid**2 + y_grid**2)
        angle = tf.atan2(y_grid, x_grid) / np.pi  # Normalized to [-1, 1]
        
        # Expand dimensions
        radius = tf.expand_dims(tf.expand_dims(radius, 0), -1)
        angle = tf.expand_dims(tf.expand_dims(angle, 0), -1)
        
        # Register buffers
        self.radius = tf.cast(radius, tf.float32)
        self.angle = tf.cast(angle, tf.float32)
        
        # Learnable scaling factors
        self.r_scale = self.add_weight(name='r_scale', shape=(1,), initializer='ones', trainable=True)
        self.a_scale = self.add_weight(name='a_scale', shape=(1,), initializer='ones', trainable=True)
        
        super().build(input_shape)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        # Scale and tile
        r = tf.tile(self.radius * self.r_scale, [batch_size, 1, 1, 1])
        a = tf.tile(self.angle * self.a_scale, [batch_size, 1, 1, 1])
        
        # Also add sin/cos encoding for better angular resolution
        sin_angle = tf.sin(a * np.pi)
        cos_angle = tf.cos(a * np.pi)
        
        return tf.concat([inputs, r, a, sin_angle, cos_angle], axis=-1)
        
    def get_config(self):
        config = super().get_config()
        return config


# ---------------------------------------------------------------------------
# Adaptive Spatial Attention with Learnable Temperature
# ---------------------------------------------------------------------------

class AdaptiveSpatialAttention(tf.keras.layers.Layer):
    """
    Spatial attention with learnable temperature for sharper or smoother focus.
    Properly wrapped as a Keras Layer to support weight creation.
    """
    def __init__(self, name_prefix='ada_spatial', **kwargs):
        super().__init__(name=name_prefix, **kwargs)
        self.name_prefix = name_prefix

    def build(self, input_shape):
        # Generate attention map
        self.conv = tf.keras.layers.Conv2D(
            1, (3, 3), padding='same',
            kernel_initializer='he_normal', use_bias=False,
            name=f'{self.name_prefix}_conv'
        )
        self.bn = tf.keras.layers.BatchNormalization(name=f'{self.name_prefix}_bn')
        
        # Learnable temperature parameter
        self.temperature = self.add_weight(
            name=f'{self.name_prefix}_temp',
            shape=(1,),
            initializer=tf.constant_initializer(1.0),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        attention = self.conv(inputs)
        attention = self.bn(attention)
        
        # Apply temperature to attention logits
        attention = attention / tf.maximum(self.temperature, 0.1)
        attention = tf.keras.activations.sigmoid(attention)
        
        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({'name_prefix': self.name_prefix})
        return config


# ---------------------------------------------------------------------------
# Enhanced SE Block with Adaptive Attention
# ---------------------------------------------------------------------------

def _se_block_adaptive(x, reduction=8, name_prefix='se'):
    """SE block with adaptive channel and spatial attention."""
    channels = x.shape[-1]
    
    # Channel attention with adaptive squeeze
    squeezed = tf.keras.layers.GlobalAveragePooling2D(name=f'{name_prefix}_gap')(x)
    
    # Adaptive bottleneck dimension
    bottleneck = tf.keras.layers.Dense(
        max(4, channels // reduction),
        activation='relu',
        name=f'{name_prefix}_bottleneck'
    )(squeezed)
    
    # Channel weights
    channel_weights = tf.keras.layers.Dense(
        channels,
        activation='sigmoid',
        name=f'{name_prefix}_weights'
    )(bottleneck)
    channel_weights = tf.keras.layers.Reshape((1, 1, channels))(channel_weights)
    
    # Apply channel attention
    x = tf.keras.layers.Multiply(name=f'{name_prefix}_channel')([x, channel_weights])
    
    # Adaptive spatial attention
    x = AdaptiveSpatialAttention(name_prefix=f'{name_prefix}_spatial')(x)
    
    return x


# ---------------------------------------------------------------------------
# Enhanced Ghost Module with Adaptive Attention
# ---------------------------------------------------------------------------

def _ghost_module_adaptive(x, out_channels, ratio=2, dw_kernel=3, use_attention=True,
                          use_rotary=False, name_prefix='gm'):
    """Ghost module with adaptive attention and rotary encoding."""
    intrinsic_ch = out_channels // ratio
    ghost_ch = out_channels - intrinsic_ch
    
    # Inject rotary position encoding if requested
    if use_rotary:
        x = RotaryPositionalEncoding2D(name=f'{name_prefix}_rotary')(x)
        # Compress back
        curr_ch = x.shape[-1]
        x = tf.keras.layers.Conv2D(
            curr_ch - 4, (1, 1), padding='same',
            kernel_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_rotary_adapt'
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_rotary_bn')(x)
    
    # Primary path
    primary = tf.keras.layers.Conv2D(
        intrinsic_ch, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_primary'
    )(x)
    primary = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_primary_bn')(primary)
    primary = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_primary_relu')(primary)
    
    # Ghost path with larger kernel for rotation
    ghost = tf.keras.layers.DepthwiseConv2D(
        (dw_kernel, dw_kernel), padding='same',
        depthwise_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_ghost_dw'
    )(primary)
    ghost = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_ghost_bn')(ghost)
    ghost = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_ghost_relu')(ghost)
    
    if ghost_ch != intrinsic_ch:
        ghost = tf.keras.layers.Conv2D(
            ghost_ch, (1, 1), padding='same',
            kernel_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_ghost_adapt'
        )(ghost)
    
    concat = tf.keras.layers.Concatenate(name=f'{name_prefix}_concat')([primary, ghost])
    
    if use_attention:
        concat = _se_block_adaptive(concat, reduction=8, name_prefix=f'{name_prefix}_attention')
    
    return concat


# ---------------------------------------------------------------------------
# Enhanced Ghost Block
# ---------------------------------------------------------------------------

def _ghost_block_adaptive(x, out_channels, stride=1, expansion=2, use_attention=True,
                         use_rotary=False, name_prefix='gb'):
    """Ghost bottleneck with adaptive features."""
    ch_in = x.shape[-1]
    hidden = ch_in * expansion
    
    # Expansion
    y = _ghost_module_adaptive(
        x, hidden, use_attention=False,
        use_rotary=use_rotary,
        name_prefix=f'{name_prefix}_expand'
    )
    
    # Spatial downsampling
    if stride > 1:
        y = tf.keras.layers.DepthwiseConv2D(
            (3, 3), strides=stride, padding='same',
            depthwise_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_stride'
        )(y)
        y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_stride_bn')(y)
        y = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_stride_relu')(y)
    
    # Projection
    y = _ghost_module_adaptive(
        y, out_channels, use_attention=use_attention,
        use_rotary=False,  # Rotary already in expand
        name_prefix=f'{name_prefix}_project'
    )
    
    # Shortcut
    if stride == 1 and ch_in == out_channels:
        shortcut = x
    else:
        shortcut = tf.keras.layers.DepthwiseConv2D(
            (3, 3), strides=stride, padding='same',
            depthwise_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_sc_dw'
        )(x)
        shortcut = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_sc_bn')(shortcut)
        shortcut = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_sc_relu')(shortcut)
        shortcut = tf.keras.layers.Conv2D(
            out_channels, (1, 1), padding='same',
            kernel_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_sc_conv'
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_sc_conv_bn')(shortcut)
    
    y = tf.keras.layers.Add(name=f'{name_prefix}_add')([shortcut, y])
    y = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_relu')(y)
    return y


# ---------------------------------------------------------------------------
# Multi-Resolution Spatial Pyramid
# ---------------------------------------------------------------------------

class ResizeToMatchTarget(tf.keras.layers.Layer):
    """
    Resizes a spatial tensor to match the spatial dimensions of another tensor.
    Useful for dynamically sized feature maps.
    """
    def __init__(self, method='bilinear', **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def call(self, inputs):
        source, target = inputs
        target_shape = tf.shape(target)
        return tf.image.resize(source, [target_shape[1], target_shape[2]], method=self.method)

    def get_config(self):
        config = super().get_config()
        config.update({'method': self.method})
        return config


def _multi_res_pyramid(x, name_prefix='mrp'):
    """Extract features at multiple resolutions."""
    # Original resolution
    r1 = tf.keras.layers.Conv2D(
        48, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_r1_conv'
    )(x)
    r1 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_r1_bn')(r1)
    
    # Medium resolution (3x3)
    r2 = tf.keras.layers.Conv2D(
        48, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_r2_conv'
    )(x)
    r2 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_r2_bn')(r2)
    
    # Large receptive field (5x5 depthwise + pointwise)
    r3 = tf.keras.layers.DepthwiseConv2D(
        (5, 5), padding='same',
        depthwise_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_r3_dw'
    )(x)
    r3 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_r3_bn')(r3)
    r3 = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_r3_relu')(r3)
    r3 = tf.keras.layers.Conv2D(
        48, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_r3_conv'
    )(r3)
    r3 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_r3_conv_bn')(r3)
    
    # Global context via average pooling + dynamic upsampling
    r4 = tf.keras.layers.GlobalAveragePooling2D(name=f'{name_prefix}_r4_gap')(x)
    r4 = tf.keras.layers.Reshape((1, 1, r4.shape[-1]))(r4)
    r4 = tf.keras.layers.Conv2D(
        48, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_r4_conv'
    )(r4)
    r4 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_r4_bn')(r4)
    r4 = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_r4_relu')(r4)
    r4 = ResizeToMatchTarget(method='bilinear', name=f'{name_prefix}_resize')([r4, x])
    
    # Concatenate all resolutions
    concat = tf.keras.layers.Concatenate(name=f'{name_prefix}_concat')([r1, r2, r3, r4])
    concat = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_output')(concat)
    
    return concat


# ---------------------------------------------------------------------------
# Main Model Builder
# ---------------------------------------------------------------------------

def create_digit_recognizer_v21(input_shape=None, nb_classes=None):
    """Create advanced spatial-aware model for >99.5% accuracy PC operations."""
    
    # Use provided arguments, otherwise fallback to params
    shape_to_use = input_shape if input_shape is not None else params.INPUT_SHAPE
    classes_to_use = nb_classes if nb_classes is not None else params.NB_CLASSES

    inputs = tf.keras.Input(shape=shape_to_use, name='input')
    
    # Rotary position encoding at input
    x = RotaryPositionalEncoding2D(name='input_rotary')(inputs)
    
    # Stem with increased capacity
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='stem_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='stem_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='stem_relu')(x)
    
    # Enhanced block configuration
    # (out_channels, stride, expansion, use_attention, use_rotary)
    blocks = [
        # Early - build spatial awareness
        (32, 1, 2, False, True),   # Learn positions
        (32, 2, 2, False, True),   # Downsample, keep positions
        
        # Mid - add attention
        (48, 1, 3, True, True),    # First attention
        (64, 2, 3, True, True),    # Downsample with attention
        (64, 1, 3, True, True),    # Refine
        (80, 1, 4, True, True),    # Expand
        
        # High - full attention
        (96, 2, 4, True, True),    # Downsample
        (96, 1, 4, True, True),    # Refine
        (112, 1, 4, True, True),   # Expand
        (128, 1, 4, True, True),   # Final features
    ]
    
    # Apply blocks
    for i, (out_ch, stride, expansion, use_attn, use_rotary) in enumerate(blocks):
        x = _ghost_block_adaptive(
            x, out_channels=out_ch,
            stride=stride,
            expansion=expansion,
            use_attention=use_attn,
            use_rotary=use_rotary,
            name_prefix=f'block{i+1}'
        )
    
    # Multi-resolution feature extraction
    x = _multi_res_pyramid(x, name_prefix='pyramid')
    
    # Final projection
    x = tf.keras.layers.Conv2D(
        160, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='fusion_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='fusion_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='fusion_relu')(x)
    
    # CRITICAL: Flatten to preserve spatial coordinates
    x = tf.keras.layers.Flatten(name='spatial_flatten')(x)
    
    # Efficient classifier with residual connection
    x = tf.keras.layers.Dropout(0.3, name='dropout1')(x)
    
    # First dense layer with batch norm
    shortcut = x
    x = tf.keras.layers.Dense(384, use_bias=False, kernel_initializer='he_normal', name='fc1')(x)
    x = tf.keras.layers.BatchNormalization(name='fc1_bn')(x)
    x = tf.keras.layers.ReLU(name='fc1_relu')(x)
    
    # Project shortcut if dimensions differ
    shortcut = tf.keras.layers.Dense(384, use_bias=False, name='shortcut')(shortcut)
    shortcut = tf.keras.layers.BatchNormalization(name='shortcut_bn')(shortcut)
    
    # Residual connection
    x = tf.keras.layers.Add(name='residual_add')([shortcut, x])
    x = tf.keras.layers.ReLU(name='residual_relu')(x)
    
    x = tf.keras.layers.Dropout(0.2, name='dropout2')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        classes_to_use, activation='softmax', name='output'
    )(x)
    
    return tf.keras.Model(inputs, outputs, name='digit_recognizer_v21')


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

def get_training_config():
    """Return optimized training config."""
    return {
        'optimizer': tf.keras.optimizers.AdamW(
            learning_rate=tf.keras.optimizers.schedules.CosineDecay(
                0.001, decay_steps=30000, alpha=0.001
            ),
            weight_decay=1e-4
        ),
        'loss': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        'metrics': ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)],
        'callbacks': [
            tf.keras.callbacks.ModelCheckpoint(
                'best_v21.h5', monitor='val_accuracy',
                save_best_only=True, mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=5, min_lr=1e-6
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=20,
                restore_best_weights=True
            )
        ]
    }


if __name__ == "__main__":
    print("=" * 70)
    print("DIGIT RECOGNIZER V21 - ROTARY POSITIONAL ENCODING (PC OPTIMIZED)")
    print("=" * 70)
    
    # Create model explicitly with test parameters instead of mutating params
    model = create_digit_recognizer_v21(input_shape=(32, 32, 3), nb_classes=100)
    model.summary()
    
    # Parameter count
    params_count = model.count_params()
    print("\n" + "=" * 70)
    print(f"Total parameters: {params_count:,}")
    print(f"Estimated Size (FP32 Memory Map): {params_count * 4 / 1024 / 1024:.2f} MB")
