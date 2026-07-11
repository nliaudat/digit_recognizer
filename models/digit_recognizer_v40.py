# models/digit_recognizer_v40.py
"""
digit_recognizer_v40 — Soft Binarization Preprocessor (v24 Successor)
======================================================================
Design: Adds a learnable preprocessing frontend to v24's proven backbone.
Two-stage pipeline before the classic v24 conv stack:

  1. Learned Luminance: 1×1 Conv + sigmoid (RGB→Grayscale with trainable weights)
  2. Adaptive Soft Binarization: sigmoid((x - threshold) × sharpness)
       - threshold: trainable (init=0.5)
       - sharpness:  trainable (init=10.0, clamped [1.0, 15.0])

The v24 backbone is used as-is (MaxPool + no BN by default) to keep the
experiment clean — we measure only the preprocessing delta.

All ops map to TFLite built-ins.  Fully QAT-compatible.

Scope / Target:
  - TARGET: ESP32 Edge Deployment (IoT, ~71 KB INT8 budget).
  - DATASETS: 10-class only (v24 backbone scales to 100 but preproc is 10-optimised).
  - RECOMMENDED:
      python train.py --model digit_recognizer_v40 --classes 10 --color rgb --no-qat --tqt

Hyperparameters (config/models.py):
  - PREPROC_V40_SHARPNESS_INIT = 10.0
  - PREPROC_V40_SHARPNESS_MIN = 1.0
  - PREPROC_V40_SHARPNESS_MAX = 15.0

Ablation:
  - Set PREPROC_V40_BINARIZE_SHARPNESS_TRAINABLE = False and retrain
    to test whether adaptivity matters.  Log sharpness trajectory to
    compare convergence.
"""

import os
import sys
import tensorflow as tf
import numpy as np

# Ensure project root is in path (needed for direct execution)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config as params

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False


# ============================================================================
# CUSTOM LAYERS
# ============================================================================

class AdaptiveBinarization(tf.keras.layers.Layer):
    """
    Soft binarization via learnable threshold and clamped sharpness.

    output = sigmoid((x - threshold) * clip(sharpness, min, max))

    config reads from PREPROC_V40_* constants in config/models.py.
    """
    def __init__(self,
                 sharpness_init=None,
                 threshold_init=None,
                 sharpness_min=None,
                 sharpness_max=None,
                 **kwargs):
        super().__init__(**kwargs)

        self._sharpness_init = (
            sharpness_init if sharpness_init is not None
            else getattr(params, 'PREPROC_V40_SHARPNESS_INIT', 10.0)
        )
        self._threshold_init = (
            threshold_init if threshold_init is not None
            else 0.5
        )
        self.sharpness_min = (
            sharpness_min if sharpness_min is not None
            else getattr(params, 'PREPROC_V40_SHARPNESS_MIN', 1.0)
        )
        self.sharpness_max = (
            sharpness_max if sharpness_max is not None
            else getattr(params, 'PREPROC_V40_SHARPNESS_MAX', 15.0)
        )

    def build(self, input_shape):
        self.threshold = self.add_weight(
            name='threshold',
            shape=(),
            initializer=tf.keras.initializers.Constant(self._threshold_init),
            trainable=True,
        )
        self.sharpness = self.add_weight(
            name='sharpness',
            shape=(),
            initializer=tf.keras.initializers.Constant(self._sharpness_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        s = tf.clip_by_value(self.sharpness, self.sharpness_min, self.sharpness_max)
        return tf.sigmoid((inputs - self.threshold) * s)

    def get_config(self):
        config = super().get_config()
        config.update({
            'sharpness_init': float(self._sharpness_init),
            'threshold_init': float(self._threshold_init),
            'sharpness_min': float(self.sharpness_min),
            'sharpness_max': float(self.sharpness_max),
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        return input_shape


# ============================================================================
# V24 BACKBONE (unchanged, adapted from v24's _build_v24_backbone)
# ============================================================================

def _build_v24_backbone(x, use_batch_norm=False):
    """
    v24 backbone with adaptive capacity based on NB_CLASSES.
    10cls: [20,36,48,56] dense=64  |  100cls: [32,58,77,90] dense=102
    """
    # Adaptive capacity
    scale = max(1.0, (params.NB_CLASSES / 10) ** 0.45)
    f     = [max(int(fi * scale), fi) for fi in [20, 36, 48, 56]]
    d     = max(int(64 * scale), 64)

    # Layer 1
    x = tf.keras.layers.Conv2D(
        f[0], (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv1_{}f'.format(f[0])
    )(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)

    # Layer 2
    x = tf.keras.layers.Conv2D(
        f[1], (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv2_{}f'.format(f[1])
    )(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(name='bn2')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)

    # Layer 3
    x = tf.keras.layers.Conv2D(
        f[2], (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv3_{}f'.format(f[2])
    )(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(name='bn3')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_3')(x)

    # Layer 4
    x = tf.keras.layers.Conv2D(
        f[3], (3, 3), padding='same',
        kernel_initializer='he_normal',
        use_bias=True,
        name='conv4_{}f'.format(f[3])
    )(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(name='bn4')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_4')(x)

    # Global pooling + dense
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = tf.keras.layers.Dense(d, activation=None, name='feature_dense')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_dense')(x)
    x = tf.keras.layers.Dropout(0.25, name='dropout')(x)

    return x


# ============================================================================
# MODEL CREATOR
# ============================================================================

def create_digit_recognizer_v40():
    """
    v40a: Soft Binarization Preprocessor + v24 backbone.

    Input(RGB) → LearnedLuminance → AdaptiveBinarization → v24_backbone → Dense(N)
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')

    # Stage 1: Learned luminance (trainable RGB→gray weights)
    x = tf.keras.layers.Conv2D(
        1, (1, 1), padding='same',
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.Constant(
            [[[[0.299], [0.587], [0.114]]]]  # BT.601 init, shape (1,1,3,1)
        ),
        bias_initializer='zeros',
        use_bias=True,
        name='luminance_conv'
    )(inputs)

    # Stage 2: Adaptive soft binarization
    x = AdaptiveBinarization(name='adaptive_binarization')(x)

    # Stage 3: v24 backbone (no BN, like v24 default)
    x = _build_v24_backbone(x, use_batch_norm=False)

    # Output
    if params.USE_LOGITS:
        outputs = tf.keras.layers.Dense(
            params.NB_CLASSES, activation=None, name='logits'
        )(x)
    else:
        outputs = tf.keras.layers.Dense(
            params.NB_CLASSES, activation='softmax', name='output'
        )(x)

    return tf.keras.Model(inputs, outputs, name='digit_recognizer_v40')


# ============================================================================
# QAT WRAPPER
# ============================================================================

def create_qat_model(base_model=None):
    if base_model is None:
        base_model = create_digit_recognizer_v40()
    if not QAT_AVAILABLE:
        print("Warning: QAT not available. Returning base model.")
        return base_model
    try:
        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_model(base_model)
        print("QAT model created for digit_recognizer_v40")
        return qat_model
    except Exception as e:
        print(f"QAT failed ({e}) – returning base model.")
        return base_model


# ============================================================================
# MAIN — self-test
# ============================================================================

if __name__ == "__main__":
    m = create_digit_recognizer_v40()
    m.summary()
    p = m.count_params()
    print(f"\nTotal parameters  : {p:,}")
    print(f"Estimated INT8 KB  : ~{p * 1.1 / 1024:.1f}")

    # Forward pass test
    dummy = tf.random.uniform((1, params.INPUT_HEIGHT, params.INPUT_WIDTH, params.INPUT_CHANNELS))
    y = m(dummy, training=False)
    print(f"Forward pass shape : {y.shape}")
    print(f"Forward pass OK    : {tf.reduce_all(tf.math.is_finite(y)).numpy()}")