# models/digit_recognizer_v26.py
"""
digit_recognizer_v26 – Soft-Binary Transition-Aware Water Meter Recognizer
===========================================================================
Extends v25 with a learnable soft-binarization layer inserted after adaptive
contrast normalization.

Why not hard binary?
  - Stable digits:       binary wins — lighting-invariant, clean shapes
  - Transitional digits: binary loses — the pixel blend (digit 5.5 = 50/50 blend
                         of 5 and 6) carries how-far-along the wheel is. Hard
                         thresholding destroys that ratio, making transition
                         detection harder.

Compromise: Soft Binarization (LearnableSoftBinarization)
    output = sigmoid(sharpness × (x − threshold))
  - sharpness=10  → ~90% binary visually, but still differentiable (no dead grads)
  - threshold     → trainable weight, model finds optimal foreground/background split
  - At high sharpness, behavior at inference ≈ hard binarization
  - All ops: subtract, multiply, sigmoid — 100% TFLite Micro standard ops

Architecture (same multi-head as v25 + binarization):
    Input
     └─ Luminance (BT.601)
         └─ AdaptiveContrastNormalization  (from v24)
             └─ LearnableSoftBinarization  ← NEW
                 └─ Backbone (4× Conv + GAP + Dense96)
                     ├─ Digit head      → digit_probs [10], digit_confidence [1]
                     └─ Transition head → transition_prob [1], transition_dir [1]

TFLite Output Tensor Indices:
    index 0: digit_probs      [batch, 10]  float32  — class probabilities
    index 1: digit_confidence [batch, 1]   float32  — digit certainty
    index 2: transition_prob  [batch, 1]   float32  — 0=stable, 1=transition
    index 3: transition_dir   [batch, 1]   float32  — 0=lower(x.0–x.6), 1=upper(x.7–x.9)

ESP32 C++ transition rule (same as v25):
    int digit = argmax(digit_probs, 10);
    if (transition_prob[0] > 0.5f && transition_dir[0] > 0.5f)
        digit = (digit + 1) % 10;
"""

import parameters as params
from utils.keras_helper import keras

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False

# Support both: imported as 'models.digit_recognizer_v26' (model_factory / Docker)
# and run directly as 'python digit_recognizer_v26.py' (local dev)
try:
    from .digit_recognizer_v24 import AdaptiveContrastNormalization
    from .digit_recognizer_v25 import (
        _luminance_layer,
        _build_v25_backbone,
        TransitionAwareLoss,
        generate_transitional_mnist,
        create_transition_dataset,
    )
except ImportError:
    from digit_recognizer_v24 import AdaptiveContrastNormalization
    from digit_recognizer_v25 import (
        _luminance_layer,
        _build_v25_backbone,
        TransitionAwareLoss,
        generate_transitional_mnist,
        create_transition_dataset,
    )


# ============================================================================
# LEARNABLE SOFT BINARIZATION LAYER
# ============================================================================

class LearnableSoftBinarization(keras.layers.Layer):
    """
    Differentiable approximation of binary thresholding.

        output = sigmoid(sharpness × (x − threshold))

    Properties:
    - threshold  (trainable): model finds optimal foreground/background split.
                              Clipped to [0.1, 0.9] to stay in valid range.
    - sharpness  (fixed):     controls how binary the output is.
                              10  → ~90% binary, good gradients
                              50  → ~99% binary, vanishing gradients
                              Recommended: keep at 10 for QAT stability.
    - At sharpness=10:
        x = 0.3 → output ≈ 0.07  (background)
        x = 0.5 → output = 0.50  (threshold)
        x = 0.7 → output ≈ 0.93  (foreground)

    TFLite Micro compatible: subtract + multiply + sigmoid (all standard ops).
    """

    def __init__(self, initial_threshold=0.5, sharpness=10.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_threshold = initial_threshold
        self.sharpness = float(sharpness)

    def build(self, input_shape):
        self.threshold = self.add_weight(
            name='threshold',
            shape=(),
            initializer=keras.initializers.Constant(self.initial_threshold),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # Clip threshold to valid range in the forward pass (no-op at save/load)
        threshold = keras.ops.clip(self.threshold, 0.1, 0.9)
        return keras.activations.sigmoid(self.sharpness * (inputs - threshold))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'initial_threshold': self.initial_threshold,
            'sharpness': self.sharpness,
        })
        return config


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_digit_recognizer_v26(use_batch_norm=False):
    """
    Create v26 multi-head model with learnable soft binarization.
    Fully TFLite Micro / ESP32 IDF compatible.

    Outputs (float32, same as v25):
        [digit_probs, digit_confidence, transition_prob, transition_dir]

    Compile example (same as v25):
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
    inputs = keras.Input(shape=params.INPUT_SHAPE, name='input')

    # Step 1: Luminance conversion (RGB → gray, no-op for 1-channel)
    x = _luminance_layer(inputs)

    # Step 2: Adaptive contrast normalization (contrast inversion + stretching)
    x = AdaptiveContrastNormalization(
        invert_threshold=0.5, stretch_contrast=True, name='adaptive_contrast'
    )(x)

    # Step 3: Soft binarization — NEW in v26
    x = LearnableSoftBinarization(
        initial_threshold=0.5, sharpness=10.0, name='soft_binarization'
    )(x)

    # Step 4: Shared backbone (identical to v25)
    features = _build_v25_backbone(x, use_batch_norm=use_batch_norm)

    # Step 5: Digit classification head
    d = keras.layers.Dense(64, activation=None, name='digit_dense')(features)
    d = keras.layers.ReLU(max_value=6.0, name='relu6_digit')(d)
    digit_probs = keras.layers.Dense(
        params.NB_CLASSES, activation='softmax', name='digit_probs'
    )(d)
    digit_confidence = keras.layers.Dense(
        1, activation='sigmoid', name='digit_confidence'
    )(d)

    # Step 6: Transition detection head
    t = keras.layers.Dense(32, activation=None, name='trans_dense')(features)
    t = keras.layers.ReLU(max_value=6.0, name='relu6_trans')(t)
    transition_prob = keras.layers.Dense(
        1, activation='sigmoid', name='transition_prob'
    )(t)
    transition_dir = keras.layers.Dense(
        1, activation='sigmoid', name='transition_dir'
    )(t)

    model = keras.Model(
        inputs=inputs,
        outputs=[digit_probs, digit_confidence, transition_prob, transition_dir],
        name='digit_recognizer_v26',
    )

    print(f"v26 parameters: {model.count_params():,}")
    return model


def create_digit_recognizer_v26_lightweight():
    """ESP32-optimized version (no BatchNorm)."""
    return create_digit_recognizer_v26(use_batch_norm=False)


# ============================================================================
# QAT
# ============================================================================

def create_qat_model_v26(model=None):
    """Create QAT-ready v26 model with preprocessing layers frozen."""
    if model is None:
        model = create_digit_recognizer_v26_lightweight()

    if not QAT_AVAILABLE:
        print("Warning: QAT not available. Returning base model.")
        return model

    try:
        qat_model = tfmot.quantization.keras.quantize_model(model)
        # Freeze preprocessing: contrast normalization and binarization
        # (their weights are fixed/near-fixed and should not be quantized-trained)
        for layer in qat_model.layers:
            if any(k in layer.name for k in ['contrast', 'luminance', 'binarization']):
                layer.trainable = False
                print(f"✓ Frozen '{layer.name}'")
        print("QAT model created successfully")
        return qat_model
    except Exception as e:
        print(f"QAT creation failed: {e}")
        return model


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=== Digit Recognizer v26 — Soft-Binary (TFLite Micro Ready) ===")

    params.INPUT_SHAPE = (28, 28, 1)
    params.NB_CLASSES = 10

    model = create_digit_recognizer_v26()
    model.summary()

    # Inspect the learned threshold (initialized at 0.5)
    for layer in model.layers:
        if 'binarization' in layer.name:
            print(f"\nSoft binarization threshold (initial): "
                  f"{layer.threshold.numpy():.4f}")
            print(f"Sharpness: {layer.sharpness}")
            break

    # Smoke test: compare binary-like vs grayscale output
    test_gray = np.full((1, 28, 28, 1), 0.4, dtype=np.float32)   # mid-gray
    test_gray[0, 10:18, 10:18, 0] = 0.9                            # bright digit

    digit_probs, digit_conf, trans_prob, trans_dir = model.predict(
        test_gray, verbose=0
    )
    digit = int(np.argmax(digit_probs[0]))
    print(f"\nDigit: {digit}  confidence: {float(digit_conf[0,0]):.3f}")
    print(f"Transition: {float(trans_prob[0,0]):.3f}  "
          f"direction: {'upper' if trans_dir[0,0] > 0.5 else 'lower'}")

    print("\n✓ TFLite Micro Compatibility:")
    print("  - Ops: Conv2D, Dense, ReLU6, MaxPool, GAP, Sigmoid, Softmax, Subtract")
    print("  - LearnableSoftBinarization → subtract + multiply + sigmoid (standard ops)")
    print("  - No stateful ops, no quantile ops")
    print("  - Transition rule applied in C++ on ESP32 side")
    print("\nCurrent vs v25:")
    print("  + LearnableSoftBinarization after contrast normalization")
    print("  + Learnable threshold adapts to dataset foreground/background split")
    print("  + ~same parameter count (+1 weight for threshold)")
