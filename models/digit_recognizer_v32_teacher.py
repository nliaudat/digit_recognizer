"""
Teacher Model V32: Super-Teacher trained via ensemble distillation from multiple models.

This teacher is designed to be:
1. Trained normally via train.py (standard training)
2. Trained via ensemble distillation from multiple teachers (v16, v4, v7, v30, etc.)
3. Used as a high-quality teacher for student distillation

Architecture: MobileNetV2-inspired with enhanced capacity
- Uses depthwise separable convolutions (ESP32-friendly)
- Optional Squeeze-and-Excitation for better feature extraction
- Designed to be larger than v16 but still efficient
- Can be quantized for deployment if needed

Why this design:
- v16 is the best model → we want to surpass it
- Combine knowledge from multiple models → ensemble distillation
- Maintain compatibility with existing training pipeline
"""

import tensorflow as tf
from typing import Tuple, Optional, Dict, Any, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False

import parameters as params


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

def squeeze_excite_block(
    x: tf.Tensor,
    reduction: int = 16,
    name_prefix: str = "se"
) -> tf.Tensor:
    """
    Squeeze-and-Excitation block for channel attention.

    Uses keepdims=True in GlobalAveragePooling2D to output shape
    (batch, 1, 1, channels) directly — avoids the separate Reshape layer
    that causes TFLite to emit SHAPE + STRIDED_SLICE + PACK dynamic-shape
    ops, which break XNNPACK INT8 and add unnecessary overhead.
    """
    channels = int(x.shape[-1])
    # keepdims=True → output is (batch, 1, 1, C), no Reshape needed
    se = tf.keras.layers.GlobalAveragePooling2D(
        keepdims=True, name=f"{name_prefix}_gap"
    )(x)
    se = tf.keras.layers.Conv2D(
        max(1, channels // reduction), 1, activation="relu",
        kernel_initializer="he_normal", name=f"{name_prefix}_reduce"
    )(se)
    se = tf.keras.layers.Conv2D(
        channels, 1, activation="sigmoid",
        kernel_initializer="he_normal", name=f"{name_prefix}_expand"
    )(se)
    return tf.keras.layers.Multiply(name=f"{name_prefix}_scale")([x, se])



def inverted_residual_block(
    x: tf.Tensor,
    filters_out: int,
    expansion: int = 6,
    stride: int = 1,
    use_se: bool = True,
    name_prefix: str = "irb"
) -> tf.Tensor:
    """
    MobileNetV2-style inverted residual block.
    
    Args:
        x: Input tensor
        filters_out: Output filters
        expansion: Expansion factor
        stride: Stride for depthwise convolution
        use_se: Use squeeze-and-excitation
        name_prefix: Prefix for layer names
    """
    in_channels = x.shape[-1]
    hidden_channels = in_channels * expansion
    use_skip = (stride == 1 and in_channels == filters_out)
    
    # Expansion
    if expansion > 1:
        y = tf.keras.layers.Conv2D(
            hidden_channels, 1, padding="same",
            kernel_initializer="he_normal", use_bias=False,
            name=f"{name_prefix}_expand"
        )(x)
        y = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_exp_bn")(y)
        y = tf.keras.layers.ReLU(max_value=6.0, name=f"{name_prefix}_exp_relu6")(y)
    else:
        y = x
    
    # Depthwise convolution
    y = tf.keras.layers.DepthwiseConv2D(
        3, strides=stride, padding="same",
        depthwise_initializer="he_normal", use_bias=False,
        name=f"{name_prefix}_dwconv"
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_dw_bn")(y)
    y = tf.keras.layers.ReLU(max_value=6.0, name=f"{name_prefix}_dw_relu6")(y)
    
    # Squeeze-and-Excitation
    if use_se:
        y = squeeze_excite_block(y, reduction=16, name_prefix=f"{name_prefix}_se")
    
    # Projection
    y = tf.keras.layers.Conv2D(
        filters_out, 1, padding="same",
        kernel_initializer="he_normal", use_bias=False,
        name=f"{name_prefix}_project"
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_proj_bn")(y)
    
    # Skip connection
    if use_skip:
        y = tf.keras.layers.Add(name=f"{name_prefix}_add")([x, y])
    
    return y


# ---------------------------------------------------------------------------
# V32 Teacher Architecture
# ---------------------------------------------------------------------------

def create_digit_recognizer_v32_teacher(
    num_classes: int = None,
    input_shape: Tuple[int, int, int] = None,
    pretrained: bool = False,  # No pretrained weights available for custom arch
    freeze_backbone: bool = False,  # Kept for API compatibility
    width_multiplier: float = 1.0,
    use_se: bool = True,
    **kwargs
) -> tf.keras.Model:
    """
    Create V32 Teacher model - enhanced version designed to surpass v16.
    
    Architecture:
        - Initial Conv (32 → 64 filters with width multiplier)
        - 6 inverted residual blocks with increasing channels
        - Optional Squeeze-and-Excitation for channel attention
        - Global Average Pooling
        - Dense layers: 512 → 256 → num_classes
        - Softmax output (compatible with existing pipeline)
    
    Args:
        num_classes: Number of output classes (10 or 100)
        input_shape: Input shape (H, W, C)
        pretrained: Not used (kept for API compatibility)
        freeze_backbone: Not used (kept for API compatibility)
        width_multiplier: Width multiplier for scaling channels (0.5-1.5)
        use_se: Use Squeeze-and-Excitation blocks
    
    Returns:
        Keras functional model with softmax output
    """
    # Use project parameters as defaults
    if num_classes is None:
        num_classes = params.NB_CLASSES
    if input_shape is None:
        input_shape = params.INPUT_SHAPE
    
    inputs = tf.keras.Input(shape=input_shape, name="input")
    
    # Scale channels based on width multiplier
    def scale_channels(c):
        return max(8, int(c * width_multiplier))
    
    # ── Initial convolution ─────────────────────────────────────────────────
    x = tf.keras.layers.Conv2D(
        scale_channels(32), 3, padding="same",
        kernel_initializer="he_normal", use_bias=False,
        name="init_conv"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="init_bn")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="init_relu6")(x)
    
    # ── Inverted residual blocks (progressive expansion) ───────────────────
    # Block configuration: (filters, expansion, stride, use_se)
    block_configs = [
        # Stage 1: low-level features
        (scale_channels(16), 1, 1, use_se),   # 32x20
        (scale_channels(24), 4, 2, use_se),   # 16x10
        (scale_channels(24), 4, 1, use_se),   # 16x10
        
        # Stage 2: mid-level features
        (scale_channels(32), 4, 2, use_se),   # 8x5
        (scale_channels(32), 4, 1, use_se),   # 8x5
        (scale_channels(48), 6, 1, use_se),   # 8x5
        (scale_channels(48), 6, 1, use_se),   # 8x5
        
        # Stage 3: high-level features
        (scale_channels(64), 6, 1, use_se),   # 8x5
        (scale_channels(96), 6, 1, use_se),   # 8x5
        (scale_channels(96), 6, 1, use_se),   # 8x5
        (scale_channels(128), 6, 1, use_se),  # 8x5
    ]
    
    for i, (filters, expansion, stride, se) in enumerate(block_configs):
        x = inverted_residual_block(
            x, filters, expansion=expansion, stride=stride,
            use_se=se, name_prefix=f"irb_{i+1}"
        )
    
    # ── Final pooling and classification ────────────────────────────────────
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    
    # Dense head (larger than v16 for better capacity)
    x = tf.keras.layers.Dense(
        scale_channels(512), kernel_initializer="he_normal", name="dense_1"
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="relu6_1")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout_1")(x)
    
    x = tf.keras.layers.Dense(
        scale_channels(256), kernel_initializer="he_normal", name="dense_2"
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="relu6_2")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_2")(x)
    
    # Output (softmax, compatible with existing pipeline)
    if params.USE_LOGITS:
        outputs = tf.keras.layers.Dense(
            num_classes, activation=None, name="logits"
        )(x)
    else:
        outputs = tf.keras.layers.Dense(
            num_classes, activation="softmax", name="output"
        )(x)
    
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name=f"teacher_v32_w{width_multiplier}"
    )
    
    # Calculate model size
    total_params = model.count_params()
    size_kb = (total_params * 4) / 1024  # FP32 size
    size_kb_quantized = size_kb / 4  # INT8 size
    
    print(f"✅ V32 Teacher (width={width_multiplier}, SE={use_se}):")
    print(f"   Parameters: {total_params:,}")
    print(f"   Size (FP32): {size_kb:.1f} KB")
    print(f"   Size (INT8): {size_kb_quantized:.1f} KB")
    print(f"   Input: {input_shape}, Classes: {num_classes}")
    
    return model


# ---------------------------------------------------------------------------
# Convenience functions for different sizes
# ---------------------------------------------------------------------------

def create_v32_teacher_small(
    num_classes: int = None,
    input_shape: Tuple[int, int, int] = None,
    **kwargs
) -> tf.keras.Model:
    """Small V32 teacher (~v16 size but better architecture)."""
    return create_digit_recognizer_v32_teacher(
        num_classes=num_classes,
        input_shape=input_shape,
        width_multiplier=0.75,
        use_se=False,
        **kwargs
    )


def create_v32_teacher_medium(
    num_classes: int = None,
    input_shape: Tuple[int, int, int] = None,
    **kwargs
) -> tf.keras.Model:
    """Medium V32 teacher (default, ~2x v16 size)."""
    return create_digit_recognizer_v32_teacher(
        num_classes=num_classes,
        input_shape=input_shape,
        width_multiplier=1.0,
        use_se=True,
        **kwargs
    )


def create_v32_teacher_large(
    num_classes: int = None,
    input_shape: Tuple[int, int, int] = None,
    **kwargs
) -> tf.keras.Model:
    """Large V32 teacher (high capacity)."""
    return create_digit_recognizer_v32_teacher(
        num_classes=num_classes,
        input_shape=input_shape,
        width_multiplier=1.25,
        use_se=True,
        **kwargs
    )


def create_v32_teacher_xl(
    num_classes: int = None,
    input_shape: Tuple[int, int, int] = None,
    **kwargs
) -> tf.keras.Model:
    """Extra large V32 teacher (maximum capacity)."""
    return create_digit_recognizer_v32_teacher(
        num_classes=num_classes,
        input_shape=input_shape,
        width_multiplier=1.5,
        use_se=True,
        **kwargs
    )


# Aliases for compatibility
create_v32_teacher = create_v32_teacher_medium


# ---------------------------------------------------------------------------
# QAT Wrapper
# ---------------------------------------------------------------------------

def create_qat_model(base_model: Optional[tf.keras.Model] = None) -> tf.keras.Model:
    """
    Wrap V32 teacher for Quantization-Aware Training.
    """
    if base_model is None:
        base_model = create_v32_teacher_medium()
    
    if not QAT_AVAILABLE:
        print("⚠️  tensorflow-model-optimization not available — returning base model.")
        return base_model
    
    try:
        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_model(base_model)
        print("✅ QAT model created for V32 teacher")
        return qat_model
    except Exception as exc:
        print(f"⚠️  QAT wrapping failed ({exc}) — returning base model.")
        return base_model


# ---------------------------------------------------------------------------
# Note: To train this model via ensemble distillation, please use
# train_distill.py. Example:
# python train_distill.py --phase student --teachers v16 v4 --student digit_recognizer_v32_teacher
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    
    print("\n" + "="*60)
    print("V32 Teacher Model Test")
    print("="*60)
    
    # Test different variants
    for variant, width, se in [
        ("small", 0.75, False),
        ("medium", 1.0, True),
        ("large", 1.25, True),
        ("xl", 1.5, True),
    ]:
        print(f"\n--- V32 {variant.upper()} (width={width}, SE={se}) ---")
        model = create_digit_recognizer_v32_teacher(
            num_classes=10,
            input_shape=(32, 20, 3),
            width_multiplier=width,
            use_se=se
        )
        
        # Test forward pass
        dummy = tf.zeros((2, 32, 20, 3))
        output = model(dummy, training=False)
        assert output.shape == (2, 10)
        assert np.allclose(output.numpy().sum(axis=1), 1.0, atol=1e-4)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
    
    print("\n✅ All tests passed!")