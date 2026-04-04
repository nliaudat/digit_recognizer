"""
Student Model V30: Lightweight depthwise separable CNN for edge deployment.
Trained via distillation from a V30 or V31 Teacher.

Follows the same coding conventions as v4/v16:
  - ReLU6 activations (ESP-NN fused, QAT-safe)
  - activation='softmax' on the output Dense
  - SparseCategoricalCrossentropy(from_logits=False) for compilation
  - BatchNorm kept (folded during INT8 conversion)
  - QAT wrapper via create_qat_model()

Supports:
    - NB_CLASSES: 10 or 100
    - INPUT_CHANNELS: 1 (grayscale) or 3 (RGB)
    - Standard project image dimensions (32 x 20)

Size variants (estimated INT8 size after quantization):
    micro  → < 30 KB  (ESP32-C3)
    small  → < 50 KB  (ESP32)
    medium → < 100 KB (ESP32-S3, balanced)
    large  → < 200 KB (Raspberry Pi, high accuracy)
"""

import tensorflow as tf
from typing import Tuple, Dict, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Architecture configurations per variant
# ---------------------------------------------------------------------------
_VARIANT_CONFIGS: Dict[str, Dict] = {
    "micro": {
        "initial_filters": 8,
        "blocks": [
            {"filters": 16, "stride": 2},
            {"filters": 16, "stride": 1},
            {"filters": 32, "stride": 2},
            {"filters": 32, "stride": 1},
        ],
        "dense_units": 32,
        "dropout": 0.0,
    },
    "small": {
        "initial_filters": 16,
        "blocks": [
            {"filters": 32, "stride": 1},
            {"filters": 64, "stride": 1},
            {"filters": 64, "stride": 1},
        ],
        "dense_units": 64,
        "dropout": 0.3,
    },
    "medium": {
        "initial_filters": 16,
        "blocks": [
            {"filters": 32, "stride": 2},
            {"filters": 32, "stride": 1},
            {"filters": 64, "stride": 2},
            {"filters": 64, "stride": 1},
            {"filters": 64, "stride": 1},
            {"filters": 96, "stride": 1},
            {"filters": 96, "stride": 1},
        ],
        "dense_units": 128,
        "dropout": 0.3,
    },
    "large": {
        "initial_filters": 32,
        "blocks": [
            {"filters": 64, "stride": 1},
            {"filters": 64, "stride": 1},
            {"filters": 128, "stride": 1},
            {"filters": 128, "stride": 1},
            {"filters": 128, "stride": 1},
            {"filters": 256, "stride": 1},
        ],
        "dense_units": 256,
        "dropout": 0.4,
    },
}


# ---------------------------------------------------------------------------
# Functional model builder
# ---------------------------------------------------------------------------

def _build_v30_student(
    num_classes: int,
    input_shape: Tuple[int, int, int],
    variant: str,
    name: str,
) -> tf.keras.Model:
    """
    Build V30 student as a Keras functional model.

    Design choices (v4/v16 compatible):
    - DepthwiseConv2D + pointwise Conv2D (separable, ESP-NN accelerated)
    - BatchNormalization (folded during INT8 conversion)
    - ReLU6 activations (fused by ESP-NN into DEPTHWISE_CONV_2D)
    - Softmax output → SparseCategoricalCrossentropy(from_logits=False)
    - No Lambda / custom ops → strict TFLite Micro built-in ops only
    """
    config = _VARIANT_CONFIGS.get(variant, _VARIANT_CONFIGS["medium"])

    inputs = tf.keras.Input(shape=input_shape, name="input")

    # ── Initial conv ──────────────────────────────────────────────────────────
    x = tf.keras.layers.Conv2D(
        config["initial_filters"], 3, padding="same",
        kernel_initializer="he_normal", use_bias=False, name="init_conv"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="init_bn")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="init_relu6")(x)

    # ── Depthwise-separable blocks ────────────────────────────────────────────
    for i, block_cfg in enumerate(config["blocks"]):
        prefix = f"block{i + 1}"
        # Depthwise
        x = tf.keras.layers.DepthwiseConv2D(
            3, strides=block_cfg["stride"], padding="same",
            depthwise_initializer="he_normal", use_bias=False,
            name=f"{prefix}_dwconv"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{prefix}_dw_bn")(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name=f"{prefix}_dw_relu6")(x)
        # Pointwise
        x = tf.keras.layers.Conv2D(
            block_cfg["filters"], 1, padding="same",
            kernel_initializer="he_normal", use_bias=False,
            name=f"{prefix}_pwconv"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{prefix}_pw_bn")(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name=f"{prefix}_pw_relu6")(x)

    # ── Head ─────────────────────────────────────────────────────────────────
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(
        config["dense_units"], kernel_initializer="he_normal", name="dense"
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="dense_relu6")(x)
    if config["dropout"] > 0:
        x = tf.keras.layers.Dropout(config["dropout"], name="dropout")(x)

    # Softmax output — same convention as v4 / v16
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="output"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


# ---------------------------------------------------------------------------
# QAT wrapper  (same pattern as v16)
# ---------------------------------------------------------------------------

def create_qat_model(base_model: Optional[tf.keras.Model] = None) -> tf.keras.Model:
    """
    Wrap a V30 student for Quantization-Aware Training.

    Args:
        base_model: Pre-built student model (created if None → medium variant).

    Returns:
        QAT-ready model (or plain base model if tfmot is unavailable).
    """
    if base_model is None:
        base_model = create_v30_student_medium()

    if not QAT_AVAILABLE:
        print("⚠️  tensorflow-model-optimization not available — returning base model.")
        return base_model

    try:
        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_model(base_model)
        print(f"✅ QAT model created for {base_model.name}")
        return qat_model
    except Exception as exc:
        print(f"⚠️  QAT wrapping failed ({exc}) — returning base model.")
        return base_model


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def create_v30_student(
    num_classes: int = 10,
    input_shape: Tuple[int, int, int] = (32, 20, 1),
    variant: str = "medium",
) -> tf.keras.Model:
    """
    Create a V30 student model.

    Args:
        num_classes:  10 or 100.
        input_shape:  (H, W, C) — (32, 20, 1) for gray, (32, 20, 3) for RGB.
        variant:      'micro' | 'small' | 'medium' | 'large'.

    Returns:
        Keras functional model, softmax output.
        Compile with SparseCategoricalCrossentropy(from_logits=False).
    """
    model = _build_v30_student(
        num_classes=num_classes,
        input_shape=input_shape,
        variant=variant,
        name=f"student_v30_{variant}",
    )

    total_params = model.count_params()
    # Post-INT8 quantization: ~4× smaller than float32
    size_kb = (total_params * 4 / 4) / 1024

    print(
        f"✅ V30 Student ({variant}): "
        f"{total_params:,} params | ~{size_kb:.0f} KB INT8 | "
        f"input={input_shape} | classes={num_classes}"
    )
    return model


# Convenience wrappers ---

def create_v30_student_micro(
    num_classes: int = 10,
    input_shape: Tuple[int, int, int] = (32, 20, 1),
) -> tf.keras.Model:
    """Ultra-light student for ESP32-C3 (< 30 KB)."""
    return create_v30_student(num_classes, input_shape=input_shape, variant="micro")


def create_v30_student_small(
    num_classes: int = 10,
    input_shape: Tuple[int, int, int] = (32, 20, 1),
) -> tf.keras.Model:
    """Small student for ESP32 (< 50 KB)."""
    return create_v30_student(num_classes, input_shape=input_shape, variant="small")


def create_v30_student_medium(
    num_classes: int = 10,
    input_shape: Tuple[int, int, int] = (32, 20, 1),
) -> tf.keras.Model:
    """Medium student for balanced accuracy/size (< 100 KB)."""
    return create_v30_student(num_classes, input_shape=input_shape, variant="medium")


def create_v30_student_large(
    num_classes: int = 10,
    input_shape: Tuple[int, int, int] = (32, 20, 1),
) -> tf.keras.Model:
    """Large student for Raspberry Pi (< 200 KB)."""
    return create_v30_student(num_classes, input_shape=input_shape, variant="large")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import parameters as params
    import numpy as np

    for channels, color in [(1, "GRAY"), (3, "RGB")]:
        for nb_cls in [10, 100]:
            shape = (params.INPUT_HEIGHT, params.INPUT_WIDTH, channels)
            for var in ["micro", "small", "medium", "large"]:
                m = create_v30_student(num_classes=nb_cls, input_shape=shape, variant=var)
                dummy = tf.zeros((2, *shape))
                out = m(dummy, training=False)
                assert out.shape == (2, nb_cls)
                assert np.allclose(out.numpy().sum(axis=1), 1.0, atol=1e-4)
                print(f"  [{color}/{nb_cls}cls/{var}] output={out.shape} ✓ softmax")

    # QAT test
    base = create_v30_student_medium(num_classes=10, input_shape=(32, 20, 1))
    qat = create_qat_model(base)
    print(f"  QAT model type: {type(qat).__name__}")