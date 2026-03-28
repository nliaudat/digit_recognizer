"""
Teacher Model V30: EfficientNetB0-based high-accuracy teacher for distillation.

Designed to be trained normally (via train.py) with full project pipeline support.
Provides rich soft targets for student distillation.

Supports:
    - NB_CLASSES: 10 or 100
    - INPUT_CHANNELS: 1 (grayscale) or 3 (RGB)
    - Standard project image dimensions (32 x 20)

Output: softmax probabilities (like v4/v16) — QAT compatible.
Loss:   SparseCategoricalCrossentropy(from_logits=False)

Note on EfficientNetB0 ImageNet weights:
    EfficientNetB0 requires at least 3 channels for ImageNet pretraining.
    When color_mode='gray' we insert a learned 1→3 channel expansion Conv
    (1×1, no bias) before the backbone so pretrained weights are reusable.
"""

import tensorflow as tf
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
# Model factory
# ---------------------------------------------------------------------------

def create_digit_recognizer_v30_teacher(
    num_classes: int = 10,
    input_shape: Tuple[int, int, int] = (32, 20, 1),
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> tf.keras.Model:
    """
    Factory for the V30 Teacher (EfficientNetB0).

    The function name follows the project convention so model_factory can
    discover it automatically as ``create_digit_recognizer_v30_teacher()``.

    Args:
        num_classes:     Output classes — 10 or 100.
        input_shape:     (H, W, C) — (32, 20, 1) for gray, (32, 20, 3) for RGB.
        pretrained:      Load ImageNet weights for EfficientNetB0 backbone.
        freeze_backbone: Freeze backbone during the first training phase.

    Returns:
        Keras functional model. Output is softmax probabilities.
        Use SparseCategoricalCrossentropy(from_logits=False) to compile.
    """
    h, w, c = input_shape
    inputs = tf.keras.Input(shape=input_shape, name="input")

    # ── Channel expansion for grayscale ──────────────────────────────────────
    # EfficientNetB0 expects 3-channel input.
    # Learned 1×1 conv: no bias, no activation — pure linear projection.
    if c == 1:
        x = tf.keras.layers.Conv2D(
            3, 1, padding="same", use_bias=False, name="gray_to_rgb"
        )(inputs)
    else:
        x = inputs

    # ── Upsample to satisfy EfficientNetB0 minimum resolution ────────────────
    if h < 32 or w < 32:
        x = tf.keras.layers.Resizing(
            max(h, 32), max(w, 32), name="spatial_upscale"
        )(x)

    # ── EfficientNetB0 backbone ───────────────────────────────────────────────
    # We instantiate the backbone independently and call it on x to prevent
    # Keras from walking up the graph and inferring the wrong input shape.
    effnet_shape = (max(h, 32), max(w, 32), 3)
    print(f"DEBUG inside v30_teacher: incoming c={c}, effnet_shape={effnet_shape}")
    try:
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet" if pretrained else None,
            input_shape=effnet_shape,
            pooling=None,
        )
    except ValueError as e:
        if "stem_conv" in str(e) and pretrained:
            print("⚠️ TFMoT / Keras 3 conflict detected. Applying tf_keras fallback...")
            import tf_keras
            fallback_model = tf_keras.applications.EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_shape=effnet_shape,
                pooling=None,
            )
            backbone = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=effnet_shape,
                pooling=None,
            )
            backbone.set_weights(fallback_model.get_weights())
            print("✅ Fallback weights loaded securely.")
        else:
            raise
    backbone.trainable = not freeze_backbone
    features = backbone(x)

    # ── Classification head ───────────────────────────────────────────────────
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(features)
    x = tf.keras.layers.Dense(512, name="dense_1")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="relu6_1")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout_1")(x)
    x = tf.keras.layers.Dense(256, name="dense_2")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="relu6_2")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_2")(x)

    # Softmax output — compatible with SparseCategoricalCrossentropy(from_logits=False)
    # and with the temperature-scaled distillation in Distiller.
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="output"
    )(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="digit_recognizer_v30_teacher"
    )

    print(
        f"✅ V30 Teacher (EfficientNetB0): "
        f"{model.count_params():,} params | "
        f"input={input_shape} | classes={num_classes} | "
        f"pretrained={pretrained} | freeze={freeze_backbone}"
    )
    return model


# ---------------------------------------------------------------------------
# QAT wrapper  (same pattern as v16)
# ---------------------------------------------------------------------------

def create_qat_model(base_model: Optional[tf.keras.Model] = None) -> tf.keras.Model:
    """
    Wrap the V30 teacher for Quantization-Aware Training.

    Args:
        base_model: Pre-built base model (will be created if None).

    Returns:
        QAT-wrapped model, or plain base model if tfmot is unavailable.
    """
    if base_model is None:
        base_model = create_digit_recognizer_v30_teacher()

    if not QAT_AVAILABLE:
        print("⚠️  tensorflow-model-optimization not available — returning base model.")
        return base_model

    try:
        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_model(base_model)
        print("✅ QAT model created for digit_recognizer_v30_teacher")
        return qat_model
    except Exception as exc:
        print(f"⚠️  QAT wrapping failed ({exc}) — returning base model.")
        return base_model


# ---------------------------------------------------------------------------
# Convenience alias used by train_distillation.py
# ---------------------------------------------------------------------------

def create_v30_teacher(
    num_classes: int = 10,
    input_shape: Tuple[int, int, int] = (32, 20, 3),
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> tf.keras.Model:
    """Alias used by the distillation pipeline."""
    return create_digit_recognizer_v30_teacher(
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import parameters as params

    for channels, name in [(1, "GRAY"), (3, "RGB")]:
        for nb_cls in [10, 100]:
            shape = (params.INPUT_HEIGHT, params.INPUT_WIDTH, channels)
            m = create_digit_recognizer_v30_teacher(
                num_classes=nb_cls, input_shape=shape, pretrained=False
            )
            dummy = tf.zeros((2, *shape))
            out = m(dummy, training=False)
            assert out.shape == (2, nb_cls), f"Bad output shape: {out.shape}"
            # Softmax outputs must sum to 1
            import numpy as np
            assert np.allclose(out.numpy().sum(axis=1), 1.0, atol=1e-4)
            print(f"  [{name}/{nb_cls}cls] output={out.shape} ✓ softmax sums=1")