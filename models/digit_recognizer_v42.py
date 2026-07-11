# models/digit_recognizer_v42.py
"""
digit_recognizer_v42 — Full Soft Conditioning Hierarchical Model
===================================================================
Design: Two-stage hierarchical recognition with soft conditioning.

Stage 1: Integer classifier (10 classes) - ~98% accuracy
Stage 2: 10 conditional decimal heads, one per integer
         Each head learns to predict decimal (0-9) for its specific integer
         Final output is probability-weighted average across all heads

Key Advantages:
  - No hard argmax decisions → gradient flows through all heads
  - All 10 decimal heads trained simultaneously
  - Graceful handling of ambiguous integer predictions
  - Better confidence calibration
  - Prevents confusion between 2.1 and 5.2

Architecture:
  [v16 Backbone]
       └── Flatten (96-dim)
             ├── Integer Head: Dense(64) → ReLU6 → Dropout → Dense(10, softmax)
             └── For each integer i (0-9):
                   └── Decimal Head i:
                         Dense(32) → ReLU6 → Dropout → Dense(10, softmax)
                         (conditioned on integer=i via scalar concat)
             └── Weighted average: Σ P(integer=i) × DecimalHead_i(decimal)

Model Size: ~165 KB INT8 (vs ~135 KB for hard conditioning)

Hyperparameters (config/models.py):
  - V42_INTEGER_DENSE_UNITS = 64
  - V42_SHARED_DECIMAL_DENSE_UNITS = 64
  - V42_HEAD_DENSE_UNITS = 32
  - V42_DROPOUT = 0.1
  - V42_LOSS_WEIGHT_INTEGER = 0.7
  - V42_LOSS_WEIGHT_DECIMAL = 0.3
"""

import tensorflow as tf
import config as params

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False


# ---------------------------------------------------------------------------
# NoOpQuantizeConfig — tells QAT to pass through this layer unchanged.
# The soft conditioning combine layer does simple arithmetic (stack, multiply,
# sum) that needs no quantization.
# ---------------------------------------------------------------------------

if QAT_AVAILABLE:
    class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
        def get_weights_and_quantizers(self, layer): return []
        def get_activations_and_quantizers(self, layer): return []
        def set_quantize_weights(self, layer, quantizers): pass
        def set_quantize_activations(self, layer, quantizers): pass
        def get_output_quantizers(self, layer): return []
        def get_config(self): return {}
else:
    class NoOpQuantizeConfig:
        pass


# ---------------------------------------------------------------------------
# Custom Keras layer for the soft conditioning combination.
# A Lambda layer is invisible to tfmot.quantize_model() and causes QAT to
# silently fail.  A real Layer + NoOpQuantizeConfig solves this.
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package='Custom')
class SoftConditioningCombine(tf.keras.layers.Layer):
    """Stack 10 decimal heads and compute Σ P(integer=i) × P(decimal|integer=i)."""

    def call(self, inputs):
        # inputs: list of 11 tensors — 10 decimal heads + 1 integer_probs
        decimal_heads = inputs[:-1]
        integer_probs = inputs[-1]
        return tf.reduce_sum(
            tf.stack(decimal_heads, axis=1) * tf.expand_dims(integer_probs, axis=2),
            axis=1
        )

    def get_config(self):
        return super().get_config()


# ---------------------------------------------------------------------------
# Config helpers (convention: safe defaults, values live in config/models.py)
# ---------------------------------------------------------------------------

def _get_integer_dense_units():
    return getattr(params, 'V42_INTEGER_DENSE_UNITS', 64)

def _get_shared_decimal_dense_units():
    return getattr(params, 'V42_SHARED_DECIMAL_DENSE_UNITS', 64)

def _get_head_dense_units():
    return getattr(params, 'V42_HEAD_DENSE_UNITS', 32)

def _get_dropout():
    return getattr(params, 'V42_DROPOUT', 0.1)


# ---------------------------------------------------------------------------
# Inverted residual bottleneck (identical to v16)
# ---------------------------------------------------------------------------

def _inv_res(x, filters_out, expand_ratio, stride, name_prefix):
    """MobileNetV2-style inverted residual."""
    ch_in = x.shape[-1]
    ch_exp = ch_in * expand_ratio
    use_shortcut = (stride == 1 and ch_in == filters_out)

    # 1. Pointwise expansion
    y = tf.keras.layers.Conv2D(
        ch_exp, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_expand'
    )(x)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_exp_bn')(y)
    y = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_exp_relu6')(y)

    # 2. Depthwise conv
    y = tf.keras.layers.DepthwiseConv2D(
        (3, 3), strides=stride, padding='same',
        depthwise_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_dw'
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_dw_bn')(y)
    y = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_dw_relu6')(y)

    # 3. Pointwise projection (linear bottleneck)
    y = tf.keras.layers.Conv2D(
        filters_out, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_project'
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_proj_bn')(y)

    # 4. Shortcut
    if use_shortcut:
        y = tf.keras.layers.Add(name=f'{name_prefix}_add')([x, y])

    return y


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def create_digit_recognizer_v42():
    """
    Full Soft Conditioning hierarchical model.

    Outputs:
      - integer_probs: [batch, 10] - probability of each integer (0-9)
      - decimal_probs: [batch, 10] - probability-weighted decimal prediction
    """
    # Soft conditioning requires probabilities for weighted combination.
    if params.USE_LOGITS:
        raise ValueError(
            "v42 soft conditioning requires USE_LOGITS=False "
            "(probability-weighted combination needs softmax, not logits)"
        )

    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')

    # ==================================================================
    # Shared Backbone (identical to v16)
    # ==================================================================

    # Entry conv
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='entry_conv'
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name='entry_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='entry_relu6')(x)

    # Inverted residual stages
    inv_res_config = [
        (24,  4, 2),   # spatial: /2
        (24,  4, 1),   # residual pass
        (40,  4, 2),   # spatial: /4
        (40,  6, 1),   # residual pass with wider expansion
        (56,  6, 1),   # deepen without downsampling
    ]
    for i, (out_ch, t, s) in enumerate(inv_res_config):
        x = _inv_res(x, filters_out=out_ch, expand_ratio=t, stride=s,
                     name_prefix=f'ir{i+1}')

    # Head conv
    x = tf.keras.layers.Conv2D(
        96, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='head_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='head_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='head_relu6')(x)

    # Global average pooling → shared feature vector
    x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name='gap')(x)
    shared_features = tf.keras.layers.Flatten(name='flatten')(x)  # [batch, 96]

    # ==================================================================
    # Stage 1: Integer Classifier
    # ==================================================================
    int_dense_units = _get_integer_dense_units()
    dec_dense_units = _get_shared_decimal_dense_units()
    head_units = _get_head_dense_units()
    dropout_rate = _get_dropout()

    integer_dense = tf.keras.layers.Dense(
        int_dense_units, activation=None,
        kernel_initializer='he_normal',
        name='integer_dense'
    )(shared_features)
    integer_dense = tf.keras.layers.ReLU(max_value=6.0, name='integer_relu6')(integer_dense)
    integer_drop = tf.keras.layers.Dropout(dropout_rate, name='integer_dropout')(integer_dense)

    integer_act = None if params.USE_LOGITS else 'softmax'
    integer_probs = tf.keras.layers.Dense(
        10, activation=integer_act, name='integer_probs'
    )(integer_drop)

    # ==================================================================
    # Stage 2: 10 Conditional Decimal Heads (Soft Conditioning)
    # ==================================================================

    # Shared decimal feature extractor (all heads share this)
    shared_decimal = tf.keras.layers.Dense(
        dec_dense_units, activation=None,
        kernel_initializer='he_normal',
        name='shared_decimal_dense'
    )(shared_features)
    shared_decimal = tf.keras.layers.ReLU(
        max_value=6.0, name='shared_decimal_relu6'
    )(shared_decimal)
    shared_decimal = tf.keras.layers.Dropout(
        dropout_rate, name='shared_decimal_dropout'
    )(shared_decimal)

    # Create 10 decimal heads, one for each integer (0-9)
    decimal_heads = []

    for i in range(10):
        # Each head is an independent Dense layer, so concatenating a constant
        # scalar to the input would only add a constant to the bias — redundant.
        # Feed shared_decimal directly into each head's Dense layer.
        head_dense = tf.keras.layers.Dense(
            head_units, activation=None,
            kernel_initializer='he_normal',
            name=f'decimal_head_{i}_dense'
        )(shared_decimal)
        head_dense = tf.keras.layers.ReLU(
            max_value=6.0, name=f'decimal_head_{i}_relu6'
        )(head_dense)
        head_drop = tf.keras.layers.Dropout(
            dropout_rate, name=f'decimal_head_{i}_dropout'
        )(head_dense)

        head_act = None if params.USE_LOGITS else 'softmax'
        head_probs = tf.keras.layers.Dense(
            10, activation=head_act,
            name=f'decimal_head_{i}_probs'
        )(head_drop)

        decimal_heads.append(head_probs)

    # ==================================================================
    # Weighted Combination (Soft Conditioning)
    # ==================================================================

    # Weighted combination: Σ P(integer=i) × P(decimal|integer=i)
    # Custom Layer (not Lambda) so tfmot.quantize_model() can see it.
    combine_layer = SoftConditioningCombine(name='decimal_probs')
    if QAT_AVAILABLE:
        combine_layer = tfmot.quantization.keras.quantize_annotate_layer(
            combine_layer, NoOpQuantizeConfig()
        )
    decimal_probs = combine_layer(decimal_heads + [integer_probs])

    # ==================================================================
    # Model Construction
    # ==================================================================

    model = tf.keras.Model(
        inputs=inputs,
        outputs=[integer_probs, decimal_probs],
        name='digit_recognizer_v42'
    )

    p = model.count_params()
    print(f"✅ v42 Full Soft Conditioning model created — {p:,} params")
    print(f"   Estimated INT8: ~{p * 1.1 / 1024:.1f} KB")
    print(f"   Outputs: integer_probs [batch,10], decimal_probs [batch,10]")
    print(f"   Head dims: int={int_dense_units} shared_dec={dec_dense_units} head={head_units}  dropout={dropout_rate}")

    return model


# ---------------------------------------------------------------------------
# QAT wrapper
# ---------------------------------------------------------------------------

def create_qat_model(base_model=None):
    if base_model is None:
        base_model = create_digit_recognizer_v42()
    if not QAT_AVAILABLE:
        print("Warning: QAT not available. Returning base model.")
        return base_model
    try:
        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_model(base_model)
        print("✅ QAT model created for digit_recognizer_v42")
        return qat_model
    except Exception as e:
        print(f"⚠️ QAT failed ({e}) – returning base model.")
        return base_model


# ---------------------------------------------------------------------------
# Main — self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("v42 Full Soft Conditioning Model")
    print("=" * 60)

    model = create_digit_recognizer_v42()
    model.summary()

    dummy = tf.random.uniform((4, params.INPUT_HEIGHT, params.INPUT_WIDTH, params.INPUT_CHANNELS))
    outputs = model(dummy, training=False)

    print(f"\n📊 Output shapes:")
    print(f"   integer_probs: {outputs[0].shape}")
    print(f"   decimal_probs: {outputs[1].shape}")

    # Demonstrate soft conditioning
    integer_probs_np = outputs[0].numpy()
    decimal_probs_np = outputs[1].numpy()

    print(f"\n🎯 Soft Conditioning Demo:")
    for i in range(4):
        int_pred = np.argmax(integer_probs_np[i])
        dec_pred = np.argmax(decimal_probs_np[i])
        combined = int_pred * 10 + dec_pred
        print(f"   Sample {i}:")
        print(f"     Integer probs: {integer_probs_np[i].round(3)}")
        print(f"     Integer pred: {int_pred} (conf={integer_probs_np[i].max():.3f})")
        print(f"     Decimal probs: {decimal_probs_np[i].round(3)}")
        print(f"     Decimal pred: {dec_pred} (conf={decimal_probs_np[i].max():.3f})")
        print(f"     Combined: {combined}")

    p = model.count_params()
    print(f"\n📈 Model Statistics:")
    print(f"   Total parameters: {p:,}")
    print(f"   Estimated INT8 size: ~{p * 1.1 / 1024:.1f} KB")
    print(f"   Input shape: {params.INPUT_SHAPE}")