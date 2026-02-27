# models/super_high_accuracy_validator.py
"""
super_high_accuracy_validator – GPU-Only Deep Accuracy Model (2026 SOTA)
=========================================================================
Design goal: Maximise classification accuracy for 100-class RGB rotating-digit
recognition on GPU/PC hardware. No IoT constraints. Targets >95% val accuracy.

Architecture:
  MultiScaleStem:
    Parallel Conv2D(32, 3×3) ‖ Conv2D(32, 5×5) ‖ Conv2D(32, 7×7)
    → Concat(96) → Conv2D(64, 1×1) → BN → ReLU → MaxPool(2×2)

  Stage 1: SEResBlock(64,  ratio=4) × 3   stride=1
  Stage 2: SEResBlock(128, ratio=4) × 4   stride=2  (spatial /2)
  Stage 3: SEResBlock(256, ratio=8) × 6   stride=2  (spatial /2)
  Stage 4: SEResBlock(512, ratio=16) × 3  stride=2  (spatial /2)

  DualAttention after Stage-4 output:
    - Channel SE (cSE): GAP → FC(C/16) ReLU → FC(C) Sigmoid → Scale
    - Spatial SE (sSE): Conv(1, 1×1) Sigmoid → Scale

  Classifier:
    GAP → Dense(1024, BN, ReLU, Dropout(0.5))
        → Dense(512, BN, ReLU, Dropout(0.3))
        → Dense(NB_CLASSES, Softmax)

Key features (SOTA 2026):
  - Multi-scale 3/5/7 parallel stem for sub-pixel feature capture in 20×32 imgs
  - Dual (spatial+channel) SE attention for strong digit discrimination
  - Label smoothing in training (pass smoothing=0.05)
  - TTA helper: tta_predict() averages over multiple rotation augmentations
  - Mixed-precision compatible (call enable_mixed_precision() before building)
  - PC/GPU-only — TFLite NOT supported (SE uses Reshape, not TFLite Micro safe)

Estimated: ~15–25M parameters → GPU float32/float16 only.

Hard classes addressed (from benchmark data):
  class 1  (80.2%), class 22 (78.8%), class 31 (79.2%)
  class 41 (82.8%), class 50 (81.6%), class 69 (83.3%)
  → Handled via per-class focal-loss weights in train.py
"""

import os
import tensorflow as tf
import numpy as np

# ---------------------------------------------------------------------------
# Optional: environment-based config (compatible with parameters.py env vars)
# ---------------------------------------------------------------------------
_NB_CLASSES = int(os.environ.get("DIGIT_NB_CLASSES", 100))
_INPUT_CHANNELS = int(os.environ.get("DIGIT_INPUT_CHANNELS", 3))
_INPUT_SHAPE = (32, 20, _INPUT_CHANNELS)  # H × W × C  (32, 20, 3) for RGB

# Try to use parameters.py if available (for train.py integration)
try:
    import parameters as _params
    _NB_CLASSES = _params.NB_CLASSES
    _INPUT_SHAPE = _params.INPUT_SHAPE
except Exception:
    pass

# ---------------------------------------------------------------------------
# Mixed Precision Helper
# ---------------------------------------------------------------------------

def enable_mixed_precision():
    """Enable float16 mixed precision for ~2× GPU throughput on Ampere+ GPUs."""
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("✅ Mixed precision enabled: float16 compute / float32 weights")


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

def _channel_se(x, ratio=16, name="cse"):
    """Channel Squeeze-and-Excitation (cSE) block."""
    filters = x.shape[-1]
    squeezed = tf.keras.layers.GlobalAveragePooling2D(
        name=f"{name}_gap", keepdims=True)(x)
    squeezed = tf.keras.layers.Conv2D(
        max(1, filters // ratio), 1, padding='same', use_bias=True,
        activation='relu', kernel_initializer='he_normal',
        name=f"{name}_fc1")(squeezed)
    squeezed = tf.keras.layers.Conv2D(
        filters, 1, padding='same', use_bias=True,
        activation='sigmoid', kernel_initializer='glorot_uniform',
        name=f"{name}_fc2")(squeezed)
    return tf.keras.layers.Multiply(name=f"{name}_scale")([x, squeezed])


def _spatial_se(x, name="sse"):
    """Spatial Squeeze-and-Excitation (sSE) block."""
    squeezed = tf.keras.layers.Conv2D(
        1, 1, padding='same', use_bias=True,
        activation='sigmoid', kernel_initializer='glorot_uniform',
        name=f"{name}_conv")(x)
    return tf.keras.layers.Multiply(name=f"{name}_scale")([x, squeezed])


def _dual_se(x, ratio=16, name="dse"):
    """Concurrent Spatial-and-Channel SE (scSE) block — best of both worlds."""
    cse_out = _channel_se(x, ratio=ratio, name=f"{name}_c")
    sse_out = _spatial_se(x, name=f"{name}_s")
    return tf.keras.layers.Add(name=f"{name}_add")([cse_out, sse_out])


def _bn_relu(x, name=None):
    """Fused BN + ReLU shorthand."""
    x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3,
                                           name=f"{name}_bn" if name else None)(x)
    x = tf.keras.layers.ReLU(name=f"{name}_relu" if name else None)(x)
    return x


def _se_res_block(x, filters, strides=1, se_ratio=16, name="srb"):
    """
    Full pre-activation SE-Residual block (ResNet-v2 style):
      BN-ReLU-Conv → BN-ReLU-Conv → dual-SE → Add(shortcut)
    """
    shortcut = x

    # Pre-activation path
    y = _bn_relu(x, name=f"{name}_pre")
    y = tf.keras.layers.Conv2D(
        filters, 3, strides=strides, padding='same', use_bias=False,
        kernel_initializer='he_normal', name=f"{name}_conv1")(y)

    y = _bn_relu(y, name=f"{name}_mid")
    y = tf.keras.layers.Conv2D(
        filters, 3, strides=1, padding='same', use_bias=False,
        kernel_initializer='he_normal', name=f"{name}_conv2")(y)

    # Dual-attention SE
    y = _dual_se(y, ratio=se_ratio, name=f"{name}_dse")

    # Adjust shortcut if dimensions change
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, 1, strides=strides, padding='same', use_bias=False,
            kernel_initializer='he_normal', name=f"{name}_skip")(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(
            momentum=0.99, epsilon=1e-3, name=f"{name}_skip_bn")(shortcut)

    return tf.keras.layers.Add(name=f"{name}_add")([shortcut, y])


def _multi_scale_stem(inputs, stem_base=32, out_ch=64, name="stem"):
    """
    Parallel multi-scale convolutions: 3×3, 5×5, 7×7
    Concatenated and projected with 1×1 conv.
    Captures both fine texture (3×3) and broad structure (7×7) simultaneously.
    """
    b3 = tf.keras.layers.Conv2D(
        stem_base, 3, padding='same', use_bias=False,
        kernel_initializer='he_normal', name=f"{name}_3x3")(inputs)

    b5 = tf.keras.layers.Conv2D(
        stem_base, 5, padding='same', use_bias=False,
        kernel_initializer='he_normal', name=f"{name}_5x5")(inputs)

    b7 = tf.keras.layers.Conv2D(
        stem_base, 7, padding='same', use_bias=False,
        kernel_initializer='he_normal', name=f"{name}_7x7")(inputs)

    x = tf.keras.layers.Concatenate(name=f"{name}_concat")([b3, b5, b7])
    x = tf.keras.layers.Conv2D(
        out_ch, 1, padding='same', use_bias=False,
        kernel_initializer='he_normal', name=f"{name}_proj")(x)
    x = _bn_relu(x, name=f"{name}_proj")
    x = tf.keras.layers.MaxPooling2D(2, strides=2, name=f"{name}_pool")(x)
    return x


# ---------------------------------------------------------------------------
# Main Model Builder
# ---------------------------------------------------------------------------

def create_super_high_accuracy_validator(
    nb_classes=None,
    input_shape=None,
    stem_base=32,
    stages=(3, 4, 6, 3),           # ResNet-50 stage depths
    filters=(64, 128, 256, 512),
    se_ratios=(4, 8, 8, 16),
    dense_units=(1024, 512),
    dropout_rates=(0.5, 0.3),
    label_smoothing=0.05,
):
    """
    Build the super_high_accuracy_validator model.

    Args:
        nb_classes:      Number of output classes (default: env DIGIT_NB_CLASSES or 100)
        input_shape:     (H, W, C) tuple (default: (32, 20, 3))
        stem_base:       Channels per branch in multi-scale stem (default: 32)
        stages:          Number of SE-ResBlocks per stage (ResNet-50: 3,4,6,3)
        filters:         Output filters per stage
        se_ratios:       SE reduction ratio per stage
        dense_units:     Dense layer widths in classifier head
        dropout_rates:   Dropout rates in classifier head
        label_smoothing: Label smoothing for compile() CategoricalCE
    Returns:
        tf.keras.Model
    """
    if nb_classes is None:
        nb_classes = _NB_CLASSES
    if input_shape is None:
        input_shape = _INPUT_SHAPE

    inputs = tf.keras.Input(shape=input_shape, name='input')

    # ── Multi-Scale Stem ────────────────────────────────────────────────────
    x = _multi_scale_stem(inputs, stem_base=stem_base, out_ch=filters[0], name="stem")

    # ── SE-ResNet Stages ────────────────────────────────────────────────────
    for s_idx, (n_blocks, filt, se_r) in enumerate(
            zip(stages, filters, se_ratios)):
        stride = 1 if s_idx == 0 else 2
        for b_idx in range(n_blocks):
            block_stride = stride if b_idx == 0 else 1
            n = f"s{s_idx+1}b{b_idx+1}"
            x = _se_res_block(
                x, filters=filt, strides=block_stride,
                se_ratio=se_r, name=n
            )

    # ── Final Dual-Attention ────────────────────────────────────────────────
    x = _dual_se(x, ratio=se_ratios[-1], name="final_dse")

    # ── Classifier Head ─────────────────────────────────────────────────────
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)

    for i, (units, drop) in enumerate(zip(dense_units, dropout_rates)):
        x = tf.keras.layers.Dense(
            units, use_bias=False, kernel_initializer='he_normal',
            name=f'fc{i+1}')(x)
        x = tf.keras.layers.BatchNormalization(
            momentum=0.99, epsilon=1e-3, name=f'fc{i+1}_bn')(x)
        x = tf.keras.layers.ReLU(name=f'fc{i+1}_relu')(x)
        x = tf.keras.layers.Dropout(drop, name=f'fc{i+1}_drop')(x)

    # Output — cast to float32 so mixed-precision loss is stable
    logits = tf.keras.layers.Dense(
        nb_classes, kernel_initializer='glorot_uniform', name='logits')(x)
    outputs = tf.keras.layers.Activation(
        'softmax', dtype='float32', name='output')(logits)

    model = tf.keras.Model(inputs, outputs, name='super_high_accuracy_validator')
    return model


# ---------------------------------------------------------------------------
# TTA (Test-Time Augmentation) wrapper
# ---------------------------------------------------------------------------

def tta_predict(model, images, angles_deg=(-10, -5, 0, 5, 10), verbose=False):
    """
    Rotation Test-Time Augmentation.
    Averages model predictions over several small rotations for better
    accuracy on rotating digit images.

    Args:
        model:      Keras model with softmax output
        images:     numpy array shape (N, H, W, C)  float32 [0,1]
        angles_deg: rotation angles to average (degrees)
        verbose:    if True, print shape info
    Returns:
        numpy array shape (N, nb_classes) — averaged probabilities
    """
    import math

    all_probs = []
    for angle in angles_deg:
        if angle == 0:
            aug_imgs = images
        else:
            rad = angle * math.pi / 180.0
            aug_imgs = tf.keras.ops.image.affine_transform(
                images,
                transform=_rotation_matrix(rad, images.shape[1], images.shape[2]),
                interpolation='bilinear',
                fill_mode='constant',
                fill_value=0.0,
            )
            if hasattr(aug_imgs, 'numpy'):
                aug_imgs = aug_imgs.numpy()

        probs = model.predict(aug_imgs, verbose=0)
        all_probs.append(probs)
        if verbose:
            print(f"  TTA angle {angle:+d}° → avg conf {probs.max(axis=1).mean():.4f}")

    return np.mean(all_probs, axis=0)


def _rotation_matrix(angle_rad, height, width):
    """Returns a batch affine transform matrix for image rotation around center."""
    cx, cy = width / 2.0, height / 2.0
    cos_a, sin_a = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    # Affine transform: [a0,a1,a2,b0,b1,b2,c0,c1] for tf.keras.ops.image.affine_transform
    a0, a1 = cos_a, -sin_a
    a2 = cx - cx * cos_a + cy * sin_a
    b0, b1 = sin_a, cos_a
    b2 = cy - cx * sin_a - cy * cos_a
    return [[a0, a1, a2, b0, b1, b2, 0.0, 0.0]]


# Simpler TTA using tf.raw_ops not available on all backends; fallback version:
def tta_predict_tfa(model, images, angles_deg=(-10, -5, 0, 5, 10)):
    """
    Fallback TTA using tf.keras.preprocessing for environments without
    tf.keras.ops.image.affine_transform.
    """
    all_probs = []
    for angle in angles_deg:
        if angle == 0:
            batch = images
        else:
            batch = np.stack([
                _rotate_image_scipy(img, angle) for img in images
            ])
        probs = model.predict(batch, verbose=0)
        all_probs.append(probs)
    return np.mean(all_probs, axis=0)


def _rotate_image_scipy(image, angle_deg):
    """Rotate a single image using scipy (best quality, anti-aliased)."""
    try:
        from scipy.ndimage import rotate as sci_rotate
        rotated = sci_rotate(image, angle_deg, axes=(0, 1),
                             reshape=False, order=1, mode='nearest')
        return np.clip(rotated, 0.0, 1.0).astype(np.float32)
    except ImportError:
        # pygame-style fallback: no-op if scipy not installed
        return image


# ---------------------------------------------------------------------------
# QAT stub (not supported for this validator — returns float32 model)
# ---------------------------------------------------------------------------

def create_qat_model(base_model=None):
    """
    QAT is NOT supported for super_high_accuracy_validator.
    Returns the float32 model as-is (PC/GPU execution only).
    """
    print("=" * 62)
    print("⚠  WARNING: super_high_accuracy_validator skips QAT")
    print("   This model is GPU/PC only — NOT for TFLite or IoT.")
    print("=" * 62)
    if base_model is None:
        base_model = create_super_high_accuracy_validator()
    return base_model


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*62}")
    print("  super_high_accuracy_validator — self-test")
    print(f"{'='*62}")
    print(f"  Input shape  : {_INPUT_SHAPE}")
    print(f"  Classes      : {_NB_CLASSES}")

    m = create_super_high_accuracy_validator()
    m.summary(line_length=90)

    total = m.count_params()
    print(f"\n✅ Total parameters : {total:,}")
    print(f"   Float32 size est : ~{total * 4 / 1024**2:.1f} MB")
    print(f"   Float16 size est : ~{total * 2 / 1024**2:.1f} MB")

    # Forward-pass sanity check
    dummy = np.random.rand(4, *_INPUT_SHAPE).astype("float32")
    out = m(dummy, training=False)
    print(f"\nForward pass output : {out.shape}")
    print(f"Sum of probs (each) : {out.numpy().sum(axis=1)}")
    print("✅ Model looks good!\n")
