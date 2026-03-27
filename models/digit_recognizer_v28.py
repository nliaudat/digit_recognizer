# models/digit_recognizer_v28.py
"""
digit_recognizer_v28 – Direct Binarization + Polarity Normalization
===================================================================
Replaces v27's multi-stage preprocessing (Augment + SoftNorm + Binarize)
with a direct, deterministic binary pipeline.

Pipeline:
  Input → Luminance → AdaptiveMeanBinarization → PolarityNormalization 
       [→ Optional: DoGEdgeDetection] → AdaptiveBackbone → Dense

1. AdaptiveMeanBinarization:
   Thresholds each image at its own mean pixel value. 
   Uses Straight-Through Estimator (STE) for differentiable training.
   
2. PolarityNormalization:
   Ensures digits are always white (1.0) and background is black (0.0).
   If the binary image is mostly white (>50% mean), it flips it `1-x`.
   Removes the need for random polarity augmentation.

Benefits:
  - 100% deterministic at inference
  - Zero reliance on training augmentation for contrast invariance
  - Cleaner graph (fewer ops, no sigmoids)
  - Purely binary input forces the CNN to learn topology, not lighting

Edge Detection:
  Optional Difference of Gaussians (DoG) fusion available via `use_edge_fusion=True`.
"""

import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import parameters as params

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False

try:
    from .digit_recognizer_v27 import _get_adaptive_config, _build_v27_backbone
except ImportError:
    from digit_recognizer_v27 import _get_adaptive_config, _build_v27_backbone


# ============================================================================
# DIRECT BINARIZATION PIPELINE
# ============================================================================

class AdaptiveMeanBinarization(tf.keras.layers.Layer):
    """
    Thresholds each image at its own mean pixel value using STE.
    Forward:  binary = float(x > mean(x))
    Backward: identity gradient
    """
    def call(self, inputs):
        # Compute mean per image
        mean = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True)
        # Hard threshold
        binary_hard = tf.cast(inputs > mean, tf.float32)
        # Straight-Through Estimator trick
        return inputs + tf.stop_gradient(binary_hard - inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()


class PolarityNormalization(tf.keras.layers.Layer):
    """
    Ensures digits are white (1.0) and background is black (0.0).
    Water meter digits take up <50% of the image area. 
    If >50% of the binary image is white (mean > 0.5), it is an inverted image, so we flip it.
    """
    def call(self, inputs):
        # Mean of binary image indicates background polarity
        mean = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True)
        # 1.0 if inverted (mostly white), 0.0 otherwise
        flip = tf.cast(mean > 0.5, tf.float32)
        # Flip pixels if needed: if flip=1 -> 1-x; if flip=0 -> x
        return (1.0 - flip) * inputs + flip * (1.0 - inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()


# ============================================================================
# DIFFERENCE OF GAUSSIANS EDGE DETECTOR (frozen, TFLite Micro compatible)
# ============================================================================

def _make_gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel."""
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    g  = np.exp(-0.5 * (ax / sigma) ** 2)
    g2d = np.outer(g, g)
    return (g2d / g2d.sum()).astype(np.float32)


class DoGEdgeDetection(tf.keras.layers.Layer):
    """
    Difference of Gaussians (DoG) edge detector.
    DoG ≈ Laplacian of Gaussian — highlights digit stroke outlines.

    Two frozen Conv2D operations:
        G1 = Gaussian(sigma=0.8, size=3)  — fine detail
        G2 = Gaussian(sigma=1.6, size=5)  — coarse detail
        DoG = G1(image) - G2(image)       — edges at scale between sigma1 and sigma2

    Handles curved strokes equally in all directions (unlike axis-aligned Sobel).
    Output: magnitude of DoG, normalized to [0, 1].

    TFLite Micro: Conv2D + Sub + Abs or relu — all standard ops.
    trainable=False on all weights.
    """

    def __init__(self, sigma1=0.8, sigma2=1.6, **kwargs):
        super().__init__(**kwargs)
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def build(self, input_shape):
        in_ch = input_shape[-1] or 1

        # Gaussian 1 (fine): 3×3
        g1 = _make_gaussian_kernel(3, self.sigma1)  # (3,3)
        k1 = np.tile(g1[:, :, np.newaxis, np.newaxis], [1, 1, in_ch, 1])  # (3,3,in_ch,1)
        self.kernel1 = self.add_weight(
            name='dog_kernel1', shape=k1.shape,
            initializer=tf.keras.initializers.Constant(k1),
            trainable=False,
        )

        # Gaussian 2 (coarse): 5×5
        g2 = _make_gaussian_kernel(5, self.sigma2)  # (5,5)
        k2 = np.tile(g2[:, :, np.newaxis, np.newaxis], [1, 1, in_ch, 1])  # (5,5,in_ch,1)
        self.kernel2 = self.add_weight(
            name='dog_kernel2', shape=k2.shape,
            initializer=tf.keras.initializers.Constant(k2),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs):
        # Apply two Gaussian blurs via depthwise-style conv (groups=in_ch not available in standard TFLite)
        # Use standard Conv2D with per-channel kernels — shape (K,K,in_ch,1) is valid
        g1_out = tf.nn.conv2d(inputs, self.kernel1, strides=[1, 1, 1, 1], padding='SAME')
        g2_out = tf.nn.conv2d(inputs, self.kernel2, strides=[1, 1, 1, 1], padding='SAME')
        dog    = g1_out - g2_out       # edges
        mag    = tf.abs(dog)           # unsigned magnitude
        # Normalize per-image to [0, 1]
        max_val = tf.reduce_max(mag, axis=[1, 2, 3], keepdims=True)
        return mag / (max_val + 1e-2)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'sigma1': self.sigma1, 'sigma2': self.sigma2})
        return config


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_digit_recognizer_v28(use_batch_norm=False, use_edge_fusion=False):
    """
    Constructs the v28 model with direct binarization and polarity normalization.
    
    Pipeline:
      1. Input: (32, 20, N) image (RGB, Grayscale, or Hybrid)
      2. Preprocessing: Luminance -> AdaptiveMeanBinarization -> PolarityNormalization
      3. Optional: DoGEdgeDetection (Edge Fusion)
      4. Multi-branch feature extraction with shared weights
      5. Softmax output for 10 or 100 classes
      
    Args:
        use_batch_norm:  Add BatchNorm after each Conv (default False for ESP32)
        use_edge_fusion: Concatenate DoG edge map with binary image before backbone.
    """
    filters, dense_units, _ = _get_adaptive_config()
    print(f"v28 config: classes={params.NB_CLASSES}  filters={filters}  "
          f"dense={dense_units}  edge_fusion={use_edge_fusion}")

    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')

    # 1. Luminance (frozen)
    input_channels = params.INPUT_SHAPE[-1]
    if input_channels is not None and input_channels == 3:
        x = tf.keras.layers.Lambda(
            lambda t: 0.299 * t[..., 0:1] + 0.587 * t[..., 1:2] + 0.114 * t[..., 2:3],
            name='luminance_grayscale'
        )(inputs)
    else:
        x = inputs

    # 2. Adaptive Mean Binarization (STE)
    binary_raw = AdaptiveMeanBinarization(name='adaptive_binarization')(x)

    # 3. Polarity Normalization (always foreground=white, background=black)
    binary = PolarityNormalization(name='polarity_norm')(binary_raw)

    # 4. Optional: fuse binary with DoG edge map
    if use_edge_fusion:
        edges = DoGEdgeDetection(sigma1=0.8, sigma2=1.6, name='dog_edges')(binary)
        x = tf.keras.layers.Concatenate(name='binary_edge_fusion')([binary, edges])
    else:
        x = binary

    # 5. Adaptive backbone (reused from v27)
    x = _build_v27_backbone(x, filters, dense_units, use_batch_norm=use_batch_norm)

    # 6. Output
    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, activation='softmax', name='output'
    )(x)

    model = tf.keras.Model(inputs, outputs, name='digit_recognizer_v28')

    # Freeze fixed preprocessing layers
    frozen = 0
    freeze_names = {'luminance_grayscale', 'adaptive_binarization', 'polarity_norm', 'dog_edges'}
    for layer in model.layers:
        if layer.name in freeze_names:
            layer.trainable = False
            frozen += 1
            print(f"✓ Locked '{layer.name}'")

    print(f"v28 params: {model.count_params():,}  frozen={frozen}")
    return model


def create_qat_model_v28(model=None, use_edge_fusion=False):
    """QAT-ready v28."""
    if model is None:
        model = create_digit_recognizer_v28(use_edge_fusion=use_edge_fusion)
    if not QAT_AVAILABLE:
        return model
    try:
        qat_model = tfmot.quantization.keras.quantize_model(model)
        for layer in qat_model.layers:
            if layer.name in ('luminance_grayscale', 'adaptive_binarization', 'polarity_norm', 'dog_edges'):
                layer.trainable = False
                print(f"✓ Frozen '{layer.name}'")
        return qat_model
    except Exception as e:
        print(f"QAT creation failed: {e}")
        return model


# ============================================================================
# INFERENCE HELPERS (same as v27)
# ============================================================================

def apply_transition_rule(class_100):
    """
    Water meter rounding rule for 100-class model.
        5.5 (class=55) → decimal_pos=5 (<7)  → digit=5  (no change)
        5.7 (class=57) → decimal_pos=7 (>=7) → digit=6  (round up)
        9.9 (class=99) → decimal_pos=9 (>=7) → digit=0  (wrap-around)
    ESP32 C++: if (cls % 10 >= 7) digit = (digit + 1) % 10;
    """
    digit = int(class_100) // 10
    decimal_pos = int(class_100) % 10
    rounded_up = decimal_pos >= 7
    if rounded_up:
        digit = (digit + 1) % 10
    return digit, decimal_pos, rounded_up


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=== Digit Recognizer v28 — Hard Binarization ===\n")
    params.NB_CLASSES  = 100

    for edge_fusion in [False, True]:
        print(f"\n{'─'*55}")
        print(f"edge_fusion={edge_fusion}")
        model = create_digit_recognizer_v28(use_edge_fusion=edge_fusion)
        model.summary()

        # Verify binarization: output should be ~{0,1}
        x = np.random.rand(2, *params.INPUT_SHAPE).astype(np.float32)
        pred = model.predict(x, verbose=0)
        cls  = int(np.argmax(pred[0]))
        digit, dec, up = apply_transition_rule(cls)
        print(f"  Prediction: class={cls} ({cls//10}.{cls%10}) → digit={digit}")

        # To check binarization we can just run the pre-processing part of the model
        binary_model = tf.keras.Model(model.input, model.get_layer('polarity_norm').output)
        out_bin = binary_model.predict(x, verbose=0)
        mean_val = float(np.mean(out_bin[0]))
        print(f"  Binarized mean: {mean_val:.3f} (expect < 0.5 since background is black)")

    print("\n✓ TFLite Micro compatible: Mean, Cast, Greater, Multiply, Sub")
    print("  AdaptiveMeanBinarization: x > mean(x)")
    print("  PolarityNormalization: flip if mean > 0.5")
    print("  Both deterministic at inference, zero trainable params in preprocessing")
