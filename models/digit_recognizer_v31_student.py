"""
Student Model V31: MobileNetV2-style inverted residual student for edge deployment.
Alternative to V30 (depthwise separable) with different inductive bias.

Follows the same coding conventions as v4/v16:
  - ReLU6 activations (ESP-NN fused, QAT-safe)
  - activation='softmax' on the output Dense
  - SparseCategoricalCrossentropy(from_logits=False) for compilation
  - BatchNorm folded during INT8 conversion
  - QAT wrapper via create_qat_model()

Built as a Keras Functional model (required for tfmot.quantize_model QAT).

Optionally uses Squeeze-and-Excitation blocks for channel attention.

Size variants:
    micro  → < 30 KB  (ESP32-C3)
    small  → < 50 KB  (ESP32)
    medium → < 100 KB (ESP32-S3, balanced)
    large  → < 200 KB (Raspberry Pi)
"""

import tensorflow as tf
from utils.keras_helper import keras
from typing import Tuple, Optional, Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Architecture variant configurations
# ---------------------------------------------------------------------------
_VARIANT_CONFIGS: Dict[str, Dict] = {
    "micro": {
        "initial_filters": 8,
        "blocks": [
            {"filters": 16, "expansion": 4, "stride": 2},
            {"filters": 16, "expansion": 4, "stride": 1},
            {"filters": 32, "expansion": 4, "stride": 2},
            {"filters": 32, "expansion": 4, "stride": 1},
        ],
        "dense_units": 32,
        "dropout": 0.0,
    },
    "small": {
        "initial_filters": 16,
        "blocks": [
            {"filters": 32, "expansion": 4, "stride": 1},
            {"filters": 64, "expansion": 4, "stride": 1},
            {"filters": 64, "expansion": 4, "stride": 1},
        ],
        "dense_units": 64,
        "dropout": 0.3,
    },
    "medium": {
        "initial_filters": 16,
        "blocks": [
            {"filters": 32, "expansion": 4, "stride": 2},
            {"filters": 32, "expansion": 4, "stride": 1},
            {"filters": 64, "expansion": 6, "stride": 2},
            {"filters": 64, "expansion": 6, "stride": 1},
            {"filters": 64, "expansion": 6, "stride": 1},
            {"filters": 96, "expansion": 6, "stride": 1},
            {"filters": 96, "expansion": 6, "stride": 1},
        ],
        "dense_units": 128,
        "dropout": 0.3,
    },
    "large": {
        "initial_filters": 32,
        "blocks": [
            {"filters": 64,  "expansion": 4, "stride": 1},
            {"filters": 64,  "expansion": 4, "stride": 1},
            {"filters": 128, "expansion": 6, "stride": 1},
            {"filters": 128, "expansion": 6, "stride": 1},
            {"filters": 128, "expansion": 6, "stride": 1},
            {"filters": 256, "expansion": 6, "stride": 1},
        ],
        "dense_units": 256,
        "dropout": 0.4,
    },
}


# ---------------------------------------------------------------------------
# Block builders (functional helpers, not Layer subclasses → QAT-safe)
# ---------------------------------------------------------------------------

def _se_block(x: tf.Tensor, prefix: str) -> tf.Tensor:
    """Squeeze-and-Excitation block for channel attention."""
    channels = x.shape[-1]
    reduction = max(1, channels // 16)

    se = keras.layers.GlobalAveragePooling2D(name=f"{prefix}_se_gap")(x)
    se = keras.layers.Reshape((1, 1, channels), name=f"{prefix}_se_reshape")(se)
    se = keras.layers.Conv2D(
        reduction, 1, activation="relu",
        kernel_initializer="he_normal", name=f"{prefix}_se_squeeze"
    )(se)
    se = keras.layers.Conv2D(
        channels, 1, activation="sigmoid",
        kernel_initializer="he_normal", name=f"{prefix}_se_excite"
    )(se)
    return keras.layers.Multiply(name=f"{prefix}_se_scale")([x, se])


def _inv_res_block(
    x: tf.Tensor,
    filters_out: int,
    expansion: int,
    stride: int,
    use_se: bool,
    prefix: str,
) -> tf.Tensor:
    """
    MobileNetV2-style inverted residual block (functional).

    expand → depthwise → [SE] → project (linear bottleneck)
    Shortcut only when stride==1 and in_channels==out_channels.

    All activations are ReLU6 (fused by ESP-NN into DEPTHWISE_CONV_2D).
    """
    in_channels = x.shape[-1]
    hidden      = in_channels * expansion
    use_skip    = (stride == 1 and in_channels == filters_out)

    # 1. Pointwise expansion
    y = keras.layers.Conv2D(
        hidden, 1, padding="same",
        kernel_initializer="he_normal", use_bias=False,
        name=f"{prefix}_expand"
    )(x)
    y = keras.layers.BatchNormalization(name=f"{prefix}_exp_bn")(y)
    y = keras.layers.ReLU(max_value=6.0, name=f"{prefix}_exp_relu6")(y)

    # 2. Depthwise conv
    y = keras.layers.DepthwiseConv2D(
        3, strides=stride, padding="same",
        depthwise_initializer="he_normal", use_bias=False,
        name=f"{prefix}_dw"
    )(y)
    y = keras.layers.BatchNormalization(name=f"{prefix}_dw_bn")(y)
    y = keras.layers.ReLU(max_value=6.0, name=f"{prefix}_dw_relu6")(y)

    # 3. Optional SE attention
    if use_se:
        y = _se_block(y, prefix)

    # 4. Pointwise projection (linear — no activation)
    y = keras.layers.Conv2D(
        filters_out, 1, padding="same",
        kernel_initializer="he_normal", use_bias=False,
        name=f"{prefix}_project"
    )(y)
    y = keras.layers.BatchNormalization(name=f"{prefix}_proj_bn")(y)

    # 5. Skip connection
    if use_skip:
        y = keras.layers.Add(name=f"{prefix}_add")([x, y])

    return y


# ---------------------------------------------------------------------------
# Model builder (Functional API)
# ---------------------------------------------------------------------------

def _build_v31_student(
    num_classes: int,
    input_shape: Tuple[int, int, int],
    variant: str,
    use_se: bool,
    name: str,
) -> keras.Model:
    """Build V31 student as Keras Functional model (QAT-compatible)."""
    config = _VARIANT_CONFIGS.get(variant, _VARIANT_CONFIGS["medium"])

    inputs = keras.Input(shape=input_shape, name="input")

    # Entry conv
    x = keras.layers.Conv2D(
        config["initial_filters"], 3, padding="same",
        kernel_initializer="he_normal", use_bias=False, name="init_conv"
    )(inputs)
    x = keras.layers.BatchNormalization(name="init_bn")(x)
    x = keras.layers.ReLU(max_value=6.0, name="init_relu6")(x)

    # Inverted residual stages
    for i, blk in enumerate(config["blocks"]):
        x = _inv_res_block(
            x,
            filters_out=blk["filters"],
            expansion=blk["expansion"],
            stride=blk["stride"],
            use_se=use_se,
            prefix=f"ir{i + 1}",
        )

    # Head
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(
        config["dense_units"], kernel_initializer="he_normal", name="dense"
    )(x)
    x = keras.layers.ReLU(max_value=6.0, name="dense_relu6")(x)
    if config["dropout"] > 0:
        x = keras.layers.Dropout(config["dropout"], name="dropout")(x)

    # Softmax output — same convention as v4 / v16
    outputs = keras.layers.Dense(
        num_classes, activation="softmax", name="output"
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


# ---------------------------------------------------------------------------
# QAT wrapper  (same pattern as v16)
# ---------------------------------------------------------------------------

def create_qat_model(base_model: Optional[keras.Model] = None) -> keras.Model:
    """
    Wrap a V31 student for Quantization-Aware Training.

    Args:
        base_model: Pre-built student model (created if None → medium variant).

    Returns:
        QAT-ready Functional model (or plain base model if tfmot unavailable).
    """
    if base_model is None:
        base_model = create_v31_student_medium()

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

def create_v31_student(
    num_classes: int = 10,
    input_shape: Tuple[int, int, int] = (32, 20, 1),
    variant: str = "medium",
    use_se: bool = False,
) -> keras.Model:
    """
    Create a V31 student model.

    Args:
        num_classes:  10 or 100.
        input_shape:  (H, W, C) — (32, 20, 1) for gray, (32, 20, 3) for RGB.
        variant:      'micro' | 'small' | 'medium' | 'large'.
        use_se:       Enable Squeeze-and-Excitation attention blocks.

    Returns:
        Keras Functional model, softmax output.
        Compile with SparseCategoricalCrossentropy(from_logits=False).
    """
    model = _build_v31_student(
        num_classes=num_classes,
        input_shape=input_shape,
        variant=variant,
        use_se=use_se,
        name=f"student_v31_{variant}{'_se' if use_se else ''}",
    )

    total_params = model.count_params()
    size_kb = (total_params * 4 / 4) / 1024  # float32 → INT8 ≈ 4× smaller

    print(
        f"✅ V31 Student ({variant}, SE={use_se}): "
        f"{total_params:,} params | ~{size_kb:.0f} KB INT8 | "
        f"input={input_shape} | classes={num_classes}"
    )
    return model


# Convenience wrappers ---

def create_v31_student_micro(
    num_classes: int = 10,
    input_shape: Tuple[int, int, int] = (32, 20, 1),
    use_se: bool = False,
) -> keras.Model:
    """Ultra-light student for ESP32-C3 (< 30 KB)."""
    return create_v31_student(num_classes, input_shape=input_shape, variant="micro", use_se=use_se)


def create_v31_student_small(
    num_classes: int = 10,
    input_shape: Tuple[int, int, int] = (32, 20, 1),
    use_se: bool = False,
) -> keras.Model:
    """Small student for ESP32 (< 50 KB)."""
    return create_v31_student(num_classes, input_shape=input_shape, variant="small", use_se=use_se)


def create_v31_student_medium(
    num_classes: int = 10,
    input_shape: Tuple[int, int, int] = (32, 20, 1),
    use_se: bool = False,
) -> keras.Model:
    """Medium student for balanced accuracy/size (< 100 KB)."""
    return create_v31_student(num_classes, input_shape=input_shape, variant="medium", use_se=use_se)


def create_v31_student_large(
    num_classes: int = 10,
    input_shape: Tuple[int, int, int] = (32, 20, 1),
    use_se: bool = False,
) -> keras.Model:
    """Large student for high accuracy deployment (< 200 KB)."""
    return create_v31_student(num_classes, input_shape=input_shape, variant="large", use_se=use_se)


# Aliases for backward compatibility
build_v31_student = create_v31_student


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
                for se in [False, True]:
                    m = create_v31_student(num_classes=nb_cls, input_shape=shape, variant=var, use_se=se)
                    dummy = tf.zeros((2, *shape))
                    out = m(dummy, training=False)
                    assert out.shape == (2, nb_cls)
                    assert np.allclose(out.numpy().sum(axis=1), 1.0, atol=1e-4)
                    print(f"  [{color}/{nb_cls}cls/{var}/SE={se}] output={out.shape} ✓ softmax")

    # QAT test
    print("\nTesting QAT wrapper ...")
    base = create_v31_student_medium(num_classes=10, input_shape=(32, 20, 1))
    qat  = create_qat_model(base)
    print(f"  QAT model type: {type(qat).__name__}")