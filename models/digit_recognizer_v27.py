# models/digit_recognizer_v27.py
"""
digit_recognizer_v27 – Improved 100-class Adaptive Contrast Model
==================================================================
v24 backbone + 4 fixes based on benchmark analysis (v23 -9.2pp, v24 -10.5pp drop at 100cls).

Fix 1 — Adaptive backbone: filters scale with NB_CLASSES
    NB_CLASSES=10  → [20,36,48,56] dense=64  (~91K params, same as v24)
    NB_CLASSES=100 → [32,58,77,90] dense=102 (~195K params, near v4/v15 range)

Fix 2 — Polarity inversion moved to training augmentation (PolarityInversionAugmentation)
    Removes unreliable inference-time detection from v24.
    Weights learn contrast invariance naturally—no inference-time flip ever needed.

Fix 3 — Soft contrast normalization (SoftContrastNormalization)
    Replaces hard min-max stretch_contrast=True from v24.
    z-score + sigmoid squash: output = sigmoid((x - mean)/(std + eps))
    Preserves relative intensity gradients between pixels.

Fix 4 — Learnable soft binarization (LearnableSoftBinarization)
    From v26: output = sigmoid(sharpness × (x − threshold))
    sharpness=10 for 10cls (more binary), sharpness=5 for 100cls (preserves blend gradient)
    threshold is trainable — adapts to water meter images.

TFLite Micro / ESP32 IDF:
    All ops: Conv2D, Dense, ReLU6, MaxPool, GAP, Sigmoid, Softmax — standard TFLite ops.
    PolarityInversionAugmentation: training-only, no-op at inference = zero inference risk.

Output: single softmax [NB_CLASSES]. Apply transition rule in C++ or Python:
    int digit = cls / 10;
    if (cls % 10 >= 7) digit = (digit + 1) % 10;
"""

import tensorflow as tf
import parameters as params

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False

try:
    from .digit_recognizer_v24 import AdaptiveContrastNormalization  # noqa: F401 kept for reference
except ImportError:
    pass  # Not used directly in v27; SoftContrastNormalization replaces it

try:
    from utils.augmentation import PolarityInversionAugmentation
except ImportError:
    # Fallback when models/ is not on sys.path (e.g. standalone run)
    class PolarityInversionAugmentation(tf.keras.layers.Layer):
        """Fallback inline definition — prefer utils.augmentation version."""
        def __init__(self, probability=0.5, **kwargs):
            super().__init__(**kwargs)
            self.probability = probability

        def call(self, inputs, training=None):
            if not training:
                return inputs
            input_rank = inputs.shape.rank
            if input_rank == 3:
                flip = tf.cast(tf.random.uniform(()) < self.probability, tf.float32)
                return (1.0 - flip) * inputs + flip * (1.0 - inputs)
            batch_size = tf.shape(inputs)[0]
            mask = tf.cast(
                tf.random.uniform([batch_size, 1, 1, 1]) < self.probability,
                tf.float32
            )
            return (1.0 - mask) * inputs + mask * (1.0 - inputs)

        def get_config(self):
            config = super().get_config()
            config.update({'probability': self.probability})
            return config


# ============================================================================
# FIX 3 — SOFT CONTRAST NORMALIZATION
# ============================================================================

class SoftContrastNormalization(tf.keras.layers.Layer):
    """
    Per-image z-score normalization + sigmoid squash.
        output = sigmoid((x − mean) / (std + eps))

    Replaces hard min-max stretch_contrast from v24:
    - Preserves relative pixel intensity differences
    - No hard clipping artifacts
    - Differentiable, numerically stable
    - TFLite Micro compatible: reduce_mean, reduce_std, sub, div, sigmoid
    """

    def __init__(self, eps=1e-2, **kwargs):
        super().__init__(**kwargs)
        self.eps_val = eps

    def build(self, input_shape):
        self.eps = self.add_weight(
            name='eps', shape=(), trainable=False,
            initializer=tf.keras.initializers.Constant(self.eps_val)
        )
        super().build(input_shape)

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True)
        std  = tf.math.reduce_std(inputs, axis=[1, 2, 3], keepdims=True)
        z    = (inputs - mean) / (std + self.eps)
        z    = tf.clip_by_value(z, -5.0, 5.0)  # sigmoid saturates at ±5; clipping preserves gradients
        return tf.sigmoid(z)

    def get_config(self):
        config = super().get_config()
        config.update({'eps': self.eps_val})
        return config


# ============================================================================
# FIX 4 — LEARNABLE SOFT BINARIZATION (from v26)
# ============================================================================

class LearnableSoftBinarization(tf.keras.layers.Layer):
    """
    Differentiable binary thresholding.
        output = sigmoid(sharpness × (x − threshold))

    - threshold (trainable): adapts to optimal foreground/background split
    - sharpness (fixed):     10 for ≤10cls (near-binary), 5 for 100cls (preserves blend)
    - TFLite Micro: subtract + multiply + sigmoid (all standard ops)
    """

    def __init__(self, initial_threshold=0.5, sharpness=10.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_threshold = initial_threshold
        self.sharpness = float(sharpness)

    def build(self, input_shape):
        self.threshold = self.add_weight(
            name='threshold', shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_threshold),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        threshold = tf.clip_by_value(self.threshold, 0.1, 0.9)
        return tf.sigmoid(self.sharpness * (inputs - threshold))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'initial_threshold': self.initial_threshold,
                       'sharpness': self.sharpness})
        return config


# ============================================================================
# FIX 1 — ADAPTIVE BACKBONE
# ============================================================================

def _get_adaptive_config():
    """Scale backbone capacity with NB_CLASSES."""
    scale = max(1.0, (params.NB_CLASSES / 10) ** 0.45)
    filters    = [max(int(f * scale), f) for f in [20, 36, 48, 56]]
    dense      = max(int(64 * scale), 64)
    sharpness  = 10.0 if params.NB_CLASSES <= 10 else 5.0
    return filters, dense, sharpness


def _build_v27_backbone(x, filters, dense_units, use_batch_norm=False):
    with tf.name_scope('backbone'):
        x = tf.keras.layers.Conv2D(filters[0], (3, 3), padding='same',
                                   kernel_initializer='he_normal', name='conv1')(x)
        if use_batch_norm: x = tf.keras.layers.BatchNormalization(name='bn1')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_1')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)

        x = tf.keras.layers.Conv2D(filters[1], (3, 3), padding='same',
                                   kernel_initializer='he_normal', name='conv2')(x)
        if use_batch_norm: x = tf.keras.layers.BatchNormalization(name='bn2')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_2')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)

        x = tf.keras.layers.Conv2D(filters[2], (3, 3), padding='same',
                                   kernel_initializer='he_normal', name='conv3')(x)
        if use_batch_norm: x = tf.keras.layers.BatchNormalization(name='bn3')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_3')(x)

        x = tf.keras.layers.Conv2D(filters[3], (3, 3), padding='same',
                                   kernel_initializer='he_normal', name='conv4')(x)
        if use_batch_norm: x = tf.keras.layers.BatchNormalization(name='bn4')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_4')(x)

        x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = tf.keras.layers.Dense(dense_units, activation=None, name='feature_dense')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_dense')(x)
        x = tf.keras.layers.Dropout(0.25, name='dropout')(x)
    return x


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_digit_recognizer_v27(use_batch_norm=False):
    """
    Create v27 model. Works for NB_CLASSES=10 and NB_CLASSES=100.

    Pipeline:
        Input → Luminance → PolarityInversionAug → SoftContrastNorm
              → SoftBinarization → AdaptiveBackbone → Dense(NB_CLASSES, softmax)

    Frozen layers (trainable=False): luminance_grayscale, soft_contrast
    Trainable preprocessing: soft_binarization.threshold
    """
    if params.NB_CLASSES not in (10, 100):
        print(f"⚠️  v27 optimized for NB_CLASSES=10 or 100 (got {params.NB_CLASSES})")

    filters, dense_units, sharpness = _get_adaptive_config()
    print(f"v27 config: classes={params.NB_CLASSES}  filters={filters}  "
          f"dense={dense_units}  binarization_sharpness={sharpness}")

    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')

    # 1. Luminance conversion (frozen: fixed BT.601 constants)
    input_channels = params.INPUT_SHAPE[-1]
    if input_channels is not None and input_channels == 3:
        x = tf.keras.layers.Lambda(
            lambda t: 0.299 * t[..., 0:1] + 0.587 * t[..., 1:2] + 0.114 * t[..., 2:3],
            name='luminance_grayscale'
        )(inputs)
    else:
        x = inputs

    # 2. Polarity inversion augmentation (training-only, no inference risk)
    x = PolarityInversionAugmentation(probability=0.5, name='polarity_aug')(x)

    # 3. Soft z-score contrast normalization (frozen: no trainable weights)
    x = SoftContrastNormalization(name='soft_contrast')(x)

    # 4. Learnable soft binarization (threshold trains, sharpness fixed)
    x = LearnableSoftBinarization(
        initial_threshold=0.5, sharpness=sharpness, name='soft_binarization'
    )(x)

    # 5. Adaptive backbone
    x = _build_v27_backbone(x, filters, dense_units, use_batch_norm=use_batch_norm)

    # 6. Classification output
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, activation='softmax', name='output'
    )(x)

    model = tf.keras.Model(inputs, outputs, name='digit_recognizer_v27')

    # Lock fixed preprocessing layers
    frozen = 0
    for layer in model.layers:
        if layer.name in ('luminance_grayscale', 'soft_contrast'):
            layer.trainable = False
            frozen += 1
            print(f"✓ Locked '{layer.name}'")
    # soft_binarization.threshold IS trainable (learns dataset split point)
    # polarity_aug has no trainable weights

    print(f"v27 params: {model.count_params():,}  frozen_preprocessing={frozen}")
    return model


# ============================================================================
# QAT
# ============================================================================

def create_qat_model_v27(model=None):
    """QAT-ready v27 with preprocessing layers frozen."""
    if model is None:
        model = create_digit_recognizer_v27()
    if not QAT_AVAILABLE:
        print("Warning: QAT not available. Returning base model.")
        return model
    try:
        qat_model = tfmot.quantization.keras.quantize_model(model)
        for layer in qat_model.layers:
            if layer.name in ('luminance_grayscale', 'soft_contrast'):
                layer.trainable = False
                print(f"✓ Frozen '{layer.name}'")
        print("QAT model created successfully")
        return qat_model
    except Exception as e:
        print(f"QAT creation failed: {e}")
        return model


# ============================================================================
# INFERENCE HELPERS
# ============================================================================

def apply_transition_rule(class_100):
    """
    Water meter rounding rule for 100-class model.

    class_100 = digit * 10 + decimal_pos  (0-99)

    Examples:
        5.5 (class=55) → decimal_pos=5 (<7)  → digit=5  (no change)
        5.7 (class=57) → decimal_pos=7 (>=7) → digit=6  (round up)
        9.9 (class=99) → decimal_pos=9 (>=7) → digit=0  (wrap-around)

    ESP32 C++:
        int digit = cls / 10;
        if (cls % 10 >= 7) digit = (digit + 1) % 10;
    """
    digit       = int(class_100) // 10
    decimal_pos = int(class_100) % 10
    rounded_up  = decimal_pos >= 7
    if rounded_up:
        digit = (digit + 1) % 10
    return digit, decimal_pos, rounded_up


def estimate_esp32_memory(model):
    """
    Estimate ESP32 RAM usage after INT8 quantization.

    arena_kb: size of the largest peak simultaneous activations in the graph.
    Computed from input shape + max filters in the backbone, NOT a hardcoded constant.
    Rule of thumb: peak is typically at the first Conv2D output before pooling.
    """
    params_count  = model.count_params()
    weights_kb    = params_count / 1024          # INT8: 1 byte per param

    # Peak arena: first post-conv activation (before pool) is usually the largest
    h, w, _  = params.INPUT_SHAPE
    filters, _, _ = _get_adaptive_config()
    filters_0 = filters[0]
    # After conv1 (no pool yet): H x W x filters_0 bytes
    peak_activation_kb = (h * w * filters_0) / 1024
    # Arena must hold ~2-3 simultaneous activations (input, output, scratch)
    arena_kb = peak_activation_kb * 3

    total_kb = weights_kb + arena_kb
    print(f"\nESP32 Memory Estimate (INT8):")
    print(f"  Weights:      {weights_kb:.1f} KB  ({params_count:,} params ×1 byte)")
    print(f"  Tensor arena: {arena_kb:.1f} KB  (peak conv1 output ×3)")
    print(f"  Total RAM:    {total_kb:.1f} KB")
    if total_kb > 320:
        print(f"  ⚠️  Exceeds typical ESP32 SRAM (320 KB)")
    return {'weights_kb': weights_kb, 'arena_kb': arena_kb, 'total_kb': total_kb}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import numpy as np
    print("=== Digit Recognizer v27 ===\n")

    for nb_cls in [10, 100]:
        print(f"\n{'─'*55}")
        # Use project standards
        params.NB_CLASSES  = nb_cls

        model = create_digit_recognizer_v27()
        model.summary()

        x    = np.random.rand(2, *params.INPUT_SHAPE).astype(np.float32)
        pred = model.predict(x, verbose=0)
        cls  = int(np.argmax(pred[0]))

        if nb_cls == 100:
            digit, dec, up = apply_transition_rule(cls)
            print(f"  Prediction: class={cls} ({cls//10}.{cls%10}) → digit={digit} rounded_up={up}")
        else:
            print(f"  Prediction: digit={cls}  conf={pred[0, cls]:.4f}")

    print("\n✓ TFLite Micro: Conv2D, Dense, ReLU6, MaxPool, GAP, Sigmoid, Softmax")
    print("  Fix1: adaptive filters  Fix2: train-only inversion  Fix3: soft norm  Fix4: soft binary")
