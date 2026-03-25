# models/digit_recognizer_v25.py
"""
digit_recognizer_v25 – 10 classes only ! Transition-Aware Water Meter Digit Recognizer
====================================================================
Design goal: Recognize digits from water meters where the wheel can be
between two numbers (e.g., 5.5 showing both 5 and 6). Uses transition
detection and weighted prediction for accurate reading.

Problem:
  - Standard MNIST models fail on transitional digits (5.5 classified as 5 or 6)
  - Water meter wheels show partial digits during rotation
  - Need to round based on position: x.0-x.6 → x, x.7-x.9 → x+1


Water Meter Wheel Positions:
[5.0] [5.1] [5.2] [5.3] [5.4] [5.5] [5.6] [5.7] [5.8] [5.9] [6.0]
  |_____ Lower Range (round down) _____|  |__ Upper Range (round up) __|
  
Rule: x.0 - x.6 → digit x (lower)
      x.7 - x.9 → digit x+1 (upper)
      9.9 → 0 (carry to next digit)

Solution:
  - Multi-head architecture sharing a common feature extractor
  - Digit head     → 10-class softmax
  - Transition head→ transition probability + direction (lower/upper)
  - Transition rule applied in C++ on ESP32 side (not in the model graph)

TFLite Micro / ESP32 IDF Compatibility:
  - All outputs are float32 (softmax / sigmoid) — standard TFLite ops only
  - No tf.nn.quantile, no stateful ops, no custom ops
  - Fully stateless inference
  - No dict outputs needed — four named output tensors, accessed by index on device
  - Transition rule (argmax + conditional +1) runs in C++ on the ESP32 side
  - QAT-compatible

Output tensor order (TFLite index → name):
  0: digit_probs      [batch, 10]  softmax  — digit class probabilities
  1: digit_confidence [batch, 1]   sigmoid  — model certainty in digit
  2: transition_prob  [batch, 1]   sigmoid  — is digit in transition?
  3: transition_dir   [batch, 1]   sigmoid  — 0=lower(x.0–x.6), 1=upper(x.7–x.9)

TFLite Output Tensor Indices:
    index 0: digit_probs      shape=[batch, 10]  float32  — class probabilities
    index 1: digit_confidence shape=[batch, 1]   float32  — digit certainty
    index 2: transition_prob  shape=[batch, 1]   float32  — 0=stable, 1=transition
    index 3: transition_dir   shape=[batch, 1]   float32  — 0=lower, 1=upper

ESP32 C++ transition rule:
    int digit = argmax(digit_probs);
    if (transition_prob > 0.5f && transition_dir > 0.5f)
        digit = (digit + 1) % 10;
"""

import tensorflow as tf
import numpy as np
import parameters as params

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False

# Support both: imported as 'models.digit_recognizer_v25' (model_factory / Docker)
# and run directly as 'python digit_recognizer_v25.py' (local dev)
try:
    from .digit_recognizer_v24 import AdaptiveContrastNormalization
except ImportError:
    from digit_recognizer_v24 import AdaptiveContrastNormalization


# ============================================================================
# LUMINANCE HELPER
# ============================================================================

def _luminance_layer(inputs):
    """RGB → grayscale using BT.601 luminance weights."""
    channels = inputs.shape[-1]
    if channels is not None and channels == 3:
        return tf.keras.layers.Lambda(
            lambda t: 0.299 * t[..., 0:1] + 0.587 * t[..., 1:2] + 0.114 * t[..., 2:3],
            name='luminance_grayscale'
        )(inputs)
    return inputs  # Already grayscale


# ============================================================================
# SHARED BACKBONE
# ============================================================================

def _build_v25_backbone(x, use_batch_norm=False):
    """
    4-layer CNN backbone. Conv1 uses 24 filters (vs v24's 20) for better
    detection of partially-visible digit edges in transitional states.
    All ops are standard TFLite Micro ops.
    """
    with tf.name_scope('backbone'):
        x = tf.keras.layers.Conv2D(24, (3, 3), padding='same',
                                   kernel_initializer='he_normal', name='conv1_24f')(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name='bn1')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_1')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool1')(x)

        x = tf.keras.layers.Conv2D(36, (3, 3), padding='same',
                                   kernel_initializer='he_normal', name='conv2_36f')(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name='bn2')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_2')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')(x)

        x = tf.keras.layers.Conv2D(48, (3, 3), padding='same',
                                   kernel_initializer='he_normal', name='conv3_48f')(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name='bn3')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_3')(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                                   kernel_initializer='he_normal', name='conv4_64f')(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name='bn4')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_4')(x)

        x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = tf.keras.layers.Dense(96, activation=None, name='shared_dense')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_shared')(x)
        x = tf.keras.layers.Dropout(0.25, name='shared_dropout')(x)
    return x


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_digit_recognizer_v25(use_batch_norm=False):
    """
    Create the v25 multi-head model using functional API.
    All outputs are float32 — fully TFLite Micro compatible.

    Compile example:
        model.compile(
            optimizer='adam',
            loss={
                'digit_probs':      'sparse_categorical_crossentropy',
                'digit_confidence': 'binary_crossentropy',
                'transition_prob':  'binary_crossentropy',
                'transition_dir':   'binary_crossentropy',
            },
            loss_weights={'digit_probs': 1.0, 'transition_prob': 0.5,
                          'transition_dir': 0.5, 'digit_confidence': 0.1},
        )
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')

    # Luminance conversion (no-op for grayscale input)
    x = _luminance_layer(inputs)

    # Adaptive contrast normalization (from v24, TFLite Micro compatible)
    x = AdaptiveContrastNormalization(
        invert_threshold=0.5, stretch_contrast=True, name='adaptive_contrast'
    )(x)

    # Shared backbone
    features = _build_v25_backbone(x, use_batch_norm=use_batch_norm)

    # ── Digit classification head ──────────────────────────────────────────
    d = tf.keras.layers.Dense(64, activation=None, name='digit_dense')(features)
    d = tf.keras.layers.ReLU(max_value=6.0, name='relu6_digit')(d)
    digit_probs = tf.keras.layers.Dense(
        params.NB_CLASSES, activation='softmax', name='digit_probs'
    )(d)
    digit_confidence = tf.keras.layers.Dense(
        1, activation='sigmoid', name='digit_confidence'
    )(d)

    # ── Transition detection head ──────────────────────────────────────────
    t = tf.keras.layers.Dense(32, activation=None, name='trans_dense')(features)
    t = tf.keras.layers.ReLU(max_value=6.0, name='relu6_trans')(t)
    transition_prob = tf.keras.layers.Dense(
        1, activation='sigmoid', name='transition_prob'
    )(t)
    transition_dir = tf.keras.layers.Dense(
        1, activation='sigmoid', name='transition_dir'
    )(t)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=[digit_probs, digit_confidence, transition_prob, transition_dir],
        name='digit_recognizer_v25'
    )

    print(f"v25 parameters: {model.count_params():,}")
    return model


def create_digit_recognizer_v25_lightweight():
    """ESP32-optimized version (no BatchNorm)."""
    return create_digit_recognizer_v25(use_batch_norm=False)


# ============================================================================
# QAT
# ============================================================================

def create_qat_model_v25(model=None):
    """Create QAT-ready v25 model with contrast layers frozen."""
    if model is None:
        model = create_digit_recognizer_v25_lightweight()

    if not QAT_AVAILABLE:
        print("Warning: QAT not available. Returning base model.")
        return model

    try:
        qat_model = tfmot.quantization.keras.quantize_model(model)
        for layer in qat_model.layers:
            if 'contrast' in layer.name or 'luminance' in layer.name:
                layer.trainable = False
                print(f"✓ Frozen '{layer.name}'")
        print("QAT model created successfully")
        return qat_model
    except Exception as e:
        print(f"QAT creation failed: {e}")
        return model


# ============================================================================
# CUSTOM MULTI-TASK LOSS
# ============================================================================

class TransitionAwareLoss(tf.keras.losses.Loss):
    """
    Multi-task loss for v25.

    y_true shape: [batch, 3]
        col 0: digit label (int)
        col 1: transition_state — 1 if in transition, 0 if stable
        col 2: transition_dir   — 1 if upper range (x.7–x.9), 0 if lower (x.0–x.6)

    y_pred: list [digit_probs, digit_confidence, transition_prob, transition_dir]

    Note: Use this loss only when training with TransitionAwareLoss directly.
    For Keras .compile(), use separate losses per output (see docstring above).
    """

    def __init__(self, digit_weight=1.0, transition_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.digit_weight = digit_weight
        self.transition_weight = transition_weight

    def call(self, y_true, y_pred):
        digit_probs, _, transition_prob, transition_dir = y_pred

        digit_target = tf.cast(y_true[..., 0], tf.int32)
        trans_target = y_true[..., 1]
        dir_target   = y_true[..., 2]

        digit_loss = tf.keras.losses.sparse_categorical_crossentropy(
            digit_target, digit_probs
        )
        trans_loss = tf.keras.losses.binary_crossentropy(
            tf.expand_dims(trans_target, -1), transition_prob
        )
        # Direction loss only counts when sample is actually in transition
        dir_loss = tf.keras.losses.binary_crossentropy(
            tf.expand_dims(dir_target, -1), transition_dir
        )
        dir_loss = dir_loss * trans_target  # Mask non-transitional samples

        return (self.digit_weight   * digit_loss +
                self.transition_weight * trans_loss +
                self.transition_weight * dir_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            'digit_weight': self.digit_weight,
            'transition_weight': self.transition_weight,
        })
        return config


# ============================================================================
# TRANSITIONAL DATA GENERATION
# ============================================================================

def generate_transitional_mnist(original_images, original_labels):
    """
    Generate blended digit pairs to simulate water-meter wheel transitions.

    Args:
        original_images: float32 array [N, H, W, C] in [0, 1]
        original_labels: int array [N] — digit class 0-9

    Returns:
        blended_images: float32 [M, H, W, C]
        blended_labels: float32 [M, 3] — [digit, transition_state, transition_dir]

    Label convention:
        blend_weight ≤ 0.6 → lower range → digit = src, trans_state=1, dir=0
        blend_weight  > 0.6 → upper range → digit = dst, trans_state=1, dir=1
    """
    # Group indices by label for efficient pairing
    label_to_idx = {}
    for i, lbl in enumerate(original_labels):
        label_to_idx.setdefault(int(lbl), []).append(i)

    blended_images, blended_labels = [], []

    for digit in range(10):
        next_digit = (digit + 1) % 10
        if digit not in label_to_idx or next_digit not in label_to_idx:
            continue

        src_idx = label_to_idx[digit]
        dst_idx = label_to_idx[next_digit]
        n_pairs = min(len(src_idx), len(dst_idx), 200)

        for i in range(n_pairs):
            img1 = original_images[src_idx[i]]
            img2 = original_images[dst_idx[i]]

            # 9 blend weights covering the full transition arc
            for w in np.linspace(0.1, 0.9, 9):
                blended = (1.0 - w) * img1 + w * img2
                if w <= 0.6:
                    label = [digit,      1, 0]   # lower range
                else:
                    label = [next_digit, 1, 1]   # upper range
                blended_images.append(blended)
                blended_labels.append(label)

    return (np.array(blended_images, dtype=np.float32),
            np.array(blended_labels,  dtype=np.float32))


def create_transition_dataset(x_train, y_train, batch_size=32):
    """
    Dataset combining standard stable digits and generated transitional blends.

    Stable samples get label [digit, 0, 0] (not in transition).
    Transitional samples get label [digit, 1, 0/1] from generate_transitional_mnist.
    """
    # Stable labels: [digit, 0, 0]
    y_stable = np.stack([
        y_train.astype(np.float32),
        np.zeros(len(y_train), dtype=np.float32),
        np.zeros(len(y_train), dtype=np.float32),
    ], axis=-1)

    # Generate transitional examples
    x_trans, y_trans = generate_transitional_mnist(x_train, y_train.astype(int))

    x_all = np.concatenate([x_train, x_trans], axis=0)
    y_all = np.concatenate([y_stable, y_trans], axis=0)

    dataset = tf.data.Dataset.from_tensor_slices((x_all, y_all))
    dataset = dataset.shuffle(20000)

    def augment(image, label):
        # Per-sample random contrast inversion — tf.cond for graph compatibility
        image = tf.cond(
            tf.random.uniform(()) > 0.5,
            lambda: 1.0 - image,
            lambda: image,
        )
        # Slight brightness jitter
        image = tf.clip_by_value(image + tf.random.uniform((), -0.1, 0.1), 0.0, 1.0)
        return image, label

    dataset = (dataset
               .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE))
    return dataset


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=== Digit Recognizer v25 — TFLite Micro / ESP32 IDF ===")

    params.INPUT_SHAPE = (28, 28, 1)
    params.NB_CLASSES = 10

    model = create_digit_recognizer_v25()
    model.summary()

    # Smoke test
    test_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
    test_input[0, 10:18, 10:18, 0] = 1.0
    digit_probs, digit_conf, trans_prob, trans_dir = model.predict(test_input, verbose=0)

    digit = int(np.argmax(digit_probs[0]))
    t_prob = float(trans_prob[0, 0])
    t_dir  = float(trans_dir[0, 0])

    print(f"\nDigit: {digit}  confidence: {float(digit_conf[0,0]):.3f}")
    print(f"Transition: {t_prob:.3f}  direction: {'upper' if t_dir > 0.5 else 'lower'}")

    # Transition rule (this runs in C++ on the ESP32 — shown here for reference only)
    if t_prob > 0.5 and t_dir > 0.5:
        digit = (digit + 1) % 10
    print(f"Final digit (after rule): {digit}")

    print("\n✓ TFLite Micro Compatibility:")
    print("  - Ops: Conv2D, Dense, ReLU6, MaxPool, GAP, Sigmoid, Softmax")
    print("  - No stateful ops, no quantile, no custom ops")
    print("  - 4 float32 output tensors — accessed by index on device")
    print("  - Transition rule applied in C++ on ESP32 side")