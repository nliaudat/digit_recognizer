"""
Super Student V34 — ConvNeXt-Small for 100-class RGB digit recognition.

Architecture: ConvNeXt-S (Small)
  - Modern pure-convolutional design (2022-2025 SOTA)
  - LayerNorm + GELU + large 7×7 depthwise kernels
  - Inverted bottleneck (expand ratio = 4)
  - DropPath (stochastic depth) for regularisation
  - Wider and deeper than V33 for the harder 100-class task
  - Designed for PC/GPU only — maximum accuracy for 100-class rotating digits

Input:  RGB (32 × 20 × 3)
Output: Softmax over 100 classes

This model is trained via multi-teacher ensemble distillation
(train_super_student.py) and then used as the ultimate teacher
for distilling into tiny IoT models via distill_best.py.

Estimated params: ~30-50M
"""

import tensorflow as tf
from typing import Tuple, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config as params

# Shared ConvNeXt building blocks (extracted to avoid duplication with V33)
from .convnext_blocks import convnext_block, convnext_downsample, layer_norm_2d


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_digit_recognizer_v34_super_student_100(
    num_classes: int = None,
    input_shape: Tuple[int, int, int] = None,
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> tf.keras.Model:
    """
    Create the V34 Super Student (ConvNeXt-Small) for 100-class RGB.

    Args:
        num_classes:  Output classes (default: 100).
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
        128,
        kernel_size=4,
        strides=4,
        padding="same",
        use_bias=True,
        name="stem_conv",
    )(inputs)
    x = layer_norm_2d(x, name="stem_ln")

    # ── Stage 1: dim=128, 3 blocks ──────────────────────────────────────
    # Spatial: 8×5
    for i in range(3):
        x = convnext_block(
            x, dim=128,
            drop_path_rate=0.0,
            name=f"stage1_block{i}",
        )

    # ── Downsample 1: 128 → 256, 8×5 → 4×3 ──────────────────────────────
    x = convnext_downsample(x, out_dim=256, name="downsample1")

    # ── Stage 2: dim=256, 3 blocks ──────────────────────────────────────
    for i in range(3):
        x = convnext_block(
            x, dim=256,
            drop_path_rate=0.1,
            name=f"stage2_block{i}",
        )

    # ── Downsample 2: 256 → 512, 4×3 → 2×2 ──────────────────────────────
    x = convnext_downsample(x, out_dim=512, name="downsample2")

    # ── Stage 3: dim=512, 27 blocks ──────────────────────────────────────
    # ConvNeXt-S has 27 blocks in stage 3 (vs 9 for Tiny)
    for i in range(27):
        x = convnext_block(
            x, dim=512,
            drop_path_rate=0.2,
            name=f"stage3_block{i}",
        )

    # ── Downsample 3: 512 → 1024, 2×2 → 1×1 ─────────────────────────────
    x = convnext_downsample(x, out_dim=1024, name="downsample3")

    # ── Stage 4: dim=1024, 3 blocks ──────────────────────────────────────
    for i in range(3):
        x = convnext_block(
            x, dim=1024,
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
        name="digit_recognizer_v34_super_student_100",
    )

    print(
        f"✅ V34 Super Student (ConvNeXt-S, 100cls): "
        f"{model.count_params():,} params | "
        f"input={input_shape}"
    )
    return model


# ---------------------------------------------------------------------------
# QAT wrapper
# ---------------------------------------------------------------------------

def create_qat_model(base_model: Optional[tf.keras.Model] = None) -> tf.keras.Model:
    """
    Wrap the V34 for Quantization-Aware Training.
    (Primarily for testing — the super student is a teacher, not for IoT.)
    """
    try:
        import tensorflow_model_optimization as tfmot
        QAT_AVAILABLE = True
    except ImportError:
        QAT_AVAILABLE = False

    if base_model is None:
        base_model = create_digit_recognizer_v34_super_student_100()

    if not QAT_AVAILABLE:
        print("⚠️  tensorflow-model-optimization not available — returning base model.")
        return base_model

    try:
        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_model(base_model)
        print("✅ QAT model created for digit_recognizer_v34_super_student_100")
        return qat_model
    except Exception as exc:
        print(f"⚠️  QAT wrapping failed ({exc}) — returning base model.")
        return base_model


# ---------------------------------------------------------------------------
# Convenience alias
# ---------------------------------------------------------------------------

def create_v34_super_student(
    num_classes: int = 100,
    input_shape: Tuple[int, int, int] = (32, 20, 3),
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> tf.keras.Model:
    """Alias used by the distillation pipeline."""
    return create_digit_recognizer_v34_super_student_100(
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
        m = create_digit_recognizer_v34_super_student_100(
            num_classes=100, input_shape=shape, pretrained=False
        )
        dummy = tf.zeros((2, *shape))
        out = m(dummy, training=False)
        assert out.shape == (2, 100), f"Bad output shape: {out.shape}"
        assert np.allclose(out.numpy().sum(axis=1), 1.0, atol=1e-4)
        print(f"  [{name}/100cls] output={out.shape} ✓ softmax sums=1")