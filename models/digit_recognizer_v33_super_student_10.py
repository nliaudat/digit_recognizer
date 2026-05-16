"""
Super Student V33 — ConvNeXt-Tiny for 10-class RGB digit recognition.

Architecture: ConvNeXt-T (Tiny)
  - Modern pure-convolutional design (2022-2025 SOTA)
  - LayerNorm + GELU + large 7×7 depthwise kernels
  - Inverted bottleneck (expand ratio = 4)
  - DropPath (stochastic depth) for regularisation
  - Designed for PC/GPU only — maximum accuracy for 10-class rotating digits

Input:  RGB (32 × 20 × 3)
Output: Softmax over 10 classes

This model is trained via multi-teacher ensemble distillation
(train_super_student.py) and then used as the ultimate teacher
for distilling into tiny IoT models via distill_best.py.

Estimated params: ~10-15M
"""

import tensorflow as tf
from typing import Tuple, Optional, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import parameters as params


# ---------------------------------------------------------------------------
# ConvNeXt building blocks
# ---------------------------------------------------------------------------

def layer_norm_2d(x: tf.Tensor, epsilon: float = 1e-6, name: str = "ln") -> tf.Tensor:
    """LayerNorm for 2D feature maps (applied per-channel)."""
    return tf.keras.layers.LayerNormalization(
        epsilon=epsilon, name=name
    )(x)


def convnext_block(
    x: tf.Tensor,
    dim: int,
    drop_path_rate: float = 0.0,
    layer_scale_init: float = 1e-6,
    name: str = "convnext_block",
) -> tf.Tensor:
    """
    ConvNeXt block:
      1. Depthwise 7×7 conv (large receptive field)
      2. LayerNorm
      3. Dense (expand ×4) + GELU
      4. Dense (project back to dim)
      5. LayerScale (learnable per-channel scaling)
      6. DropPath (stochastic depth)
      7. Residual connection
    """
    shortcut = x

    # 1. Depthwise 7×7 conv
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=7,
        padding="same",
        depthwise_initializer="he_normal",
        name=f"{name}_dw_conv",
    )(x)

    # 2. LayerNorm (after DW conv, before mixing)
    x = layer_norm_2d(x, name=f"{name}_ln_1")

    # 3. Inverted bottleneck: expand ×4
    x = tf.keras.layers.Dense(
        dim * 4,
        name=f"{name}_fc1",
    )(x)
    x = tf.keras.layers.Activation("gelu", name=f"{name}_gelu")(x)

    # 4. Project back to dim
    x = tf.keras.layers.Dense(
        dim,
        name=f"{name}_fc2",
    )(x)

    # 5. LayerScale (learnable per-channel scaling)
    if layer_scale_init > 0:
        gamma = tf.Variable(
            initial_value=tf.ones((dim,)) * layer_scale_init,
            trainable=True,
            name=f"{name}_gamma",
        )
        # Reshape gamma to (1, 1, 1, dim) for broadcasting
        x = x * tf.reshape(gamma, (1, 1, 1, dim))

    # 6. DropPath (stochastic depth)
    if drop_path_rate > 0:
        x = tf.keras.layers.Dropout(
            rate=drop_path_rate, name=f"{name}_droppath"
        )(x)

    # 7. Residual
    x = shortcut + x
    return x


def convnext_downsample(
    x: tf.Tensor,
    out_dim: int,
    name: str = "downsample",
) -> tf.Tensor:
    """
    ConvNeXt downsampling block:
      LayerNorm → Conv2D 2×2 stride 2
    """
    x = layer_norm_2d(x, name=f"{name}_ln")
    x = tf.keras.layers.Conv2D(
        out_dim,
        kernel_size=2,
        strides=2,
        padding="same",
        use_bias=True,
        name=f"{name}_conv",
    )(x)
    return x


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_digit_recognizer_v33_super_student_10(
    num_classes: int = None,
    input_shape: Tuple[int, int, int] = None,
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> tf.keras.Model:
    """
    Create the V33 Super Student (ConvNeXt-Tiny) for 10-class RGB.

    Args:
        num_classes:  Output classes (default: 10).
        input_shape:  (H, W, C) — (32, 20, 3) for RGB.
        pretrained:   Ignored (no ImageNet weights for custom ConvNeXt).
        freeze_backbone: Ignored (no pretrained backbone).

    Returns:
        Keras functional model with softmax output.
    """
    if num_classes is None:
        num_classes = params.NB_CLASSES
    if input_shape is None:
        input_shape = params.INPUT_SHAPE

    h, w, c = input_shape
    inputs = tf.keras.Input(shape=input_shape, name="input")

    # ── Stem: Patchify with 4×4 conv (stride 4) ──────────────────────────
    # Input 32×20 → 8×5 after stem
    x = tf.keras.layers.Conv2D(
        96,
        kernel_size=4,
        strides=4,
        padding="same",
        use_bias=True,
        name="stem_conv",
    )(inputs)
    x = layer_norm_2d(x, name="stem_ln")

    # ── Stage 1: dim=96, 3 blocks ────────────────────────────────────────
    # Spatial: 8×5
    for i in range(3):
        x = convnext_block(
            x, dim=96,
            drop_path_rate=0.0,
            name=f"stage1_block{i}",
        )

    # ── Downsample 1: 96 → 192, 8×5 → 4×3 ───────────────────────────────
    x = convnext_downsample(x, out_dim=192, name="downsample1")

    # ── Stage 2: dim=192, 3 blocks ───────────────────────────────────────
    for i in range(3):
        x = convnext_block(
            x, dim=192,
            drop_path_rate=0.1,
            name=f"stage2_block{i}",
        )

    # ── Downsample 2: 192 → 384, 4×3 → 2×2 ──────────────────────────────
    x = convnext_downsample(x, out_dim=384, name="downsample2")

    # ── Stage 3: dim=384, 9 blocks ───────────────────────────────────────
    for i in range(9):
        x = convnext_block(
            x, dim=384,
            drop_path_rate=0.2,
            name=f"stage3_block{i}",
        )

    # ── Downsample 3: 384 → 768, 2×2 → 1×1 ──────────────────────────────
    x = convnext_downsample(x, out_dim=768, name="downsample3")

    # ── Stage 4: dim=768, 3 blocks ───────────────────────────────────────
    for i in range(3):
        x = convnext_block(
            x, dim=768,
            drop_path_rate=0.3,
            name=f"stage4_block{i}",
        )

    # ── Head ─────────────────────────────────────────────────────────────
    x = layer_norm_2d(x, name="head_ln")
    x = tf.keras.layers.GlobalAveragePooling2D(name="head_gap")(x)
    x = tf.keras.layers.Flatten(name="head_flatten")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="output"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="digit_recognizer_v33_super_student_10",
    )

    print(
        f"✅ V33 Super Student (ConvNeXt-T, 10cls): "
        f"{model.count_params():,} params | "
        f"input={input_shape}"
    )
    return model


# ---------------------------------------------------------------------------
# QAT wrapper
# ---------------------------------------------------------------------------

def create_qat_model(base_model: Optional[tf.keras.Model] = None) -> tf.keras.Model:
    """
    Wrap the V33 for Quantization-Aware Training.
    (Primarily for testing — the super student is a teacher, not for IoT.)
    """
    try:
        import tensorflow_model_optimization as tfmot
        QAT_AVAILABLE = True
    except ImportError:
        QAT_AVAILABLE = False

    if base_model is None:
        base_model = create_digit_recognizer_v33_super_student_10()

    if not QAT_AVAILABLE:
        print("⚠️  tensorflow-model-optimization not available — returning base model.")
        return base_model

    try:
        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_model(base_model)
        print("✅ QAT model created for digit_recognizer_v33_super_student_10")
        return qat_model
    except Exception as exc:
        print(f"⚠️  QAT wrapping failed ({exc}) — returning base model.")
        return base_model


# ---------------------------------------------------------------------------
# Convenience alias
# ---------------------------------------------------------------------------

def create_v33_super_student(
    num_classes: int = 10,
    input_shape: Tuple[int, int, int] = (32, 20, 3),
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> tf.keras.Model:
    """Alias used by the distillation pipeline."""
    return create_digit_recognizer_v33_super_student_10(
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    for channels, name in [(1, "GRAY"), (3, "RGB")]:
        shape = (params.INPUT_HEIGHT, params.INPUT_WIDTH, channels)
        m = create_digit_recognizer_v33_super_student_10(
            num_classes=10, input_shape=shape, pretrained=False
        )
        dummy = tf.zeros((2, *shape))
        out = m(dummy, training=False)
        assert out.shape == (2, 10), f"Bad output shape: {out.shape}"
        assert np.allclose(out.numpy().sum(axis=1), 1.0, atol=1e-4)
        print(f"  [{name}/10cls] output={out.shape} ✓ softmax sums=1")
