# models/digit_recognizer_v41.py
"""
digit_recognizer_v41 — Multi-Head (Tens + Units) based on v16 Backbone
======================================================================
Design: Decomposes 100-class (0-99 = 0.0-9.9) into two independent
10-class problems: a **tens head** and a **units head**.

The key insight is that 100-way softmax is fundamentally harder to
optimize than 10-way.  By predicting each digit position independently,
we convert the problem into two 10-way tasks — exactly where the v16
backbone already achieves ~99% accuracy.

Architecture:
  [v16 Backbone — identical to digit_recognizer_v16]
       └── Flatten (shared 96-dim feature vector)
             ├── Tens head:  Dense(32) → ReLU6 → Dropout → Dense(10, softmax)
             └── Units head: Dense(32) → ReLU6 → Dropout → Dense(10, softmax)

  Combined prediction:  digit = argmax(tens_probs) × 10 + argmax(units_probs)

Scope / Target:
  - TARGET: 100-class only (for 10-class use standard v16).
  - ESP32 Edge Deployment (~135 KB INT8, same footprint as v16 100cls).
  - All ops map to TFLite built-ins.  Fully QAT-compatible.

Training:
  Labels are decomposed: tens_label = label // 10, units_label = label % 10.
  Loss: sparse_categorical_crossentropy on each head, equal weights (1.0).

Hyperparameters (config/models.py):
  - V41_HEAD_DENSE_UNITS = 32
  - V41_HEAD_DROPOUT = 0.2
"""

import tensorflow as tf
import config as params

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Inverted residual bottleneck  (identical to v16)
# ---------------------------------------------------------------------------

def _inv_res(x, filters_out, expand_ratio, stride, name_prefix):
    """MobileNetV2-style inverted residual.  Exact copy from v16."""
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

def _get_head_dense_units():
    """Read head dense size from config, with safe fallback."""
    return getattr(params, 'V41_HEAD_DENSE_UNITS', 32)


def _get_head_dropout():
    """Read head dropout from config, with safe fallback."""
    return getattr(params, 'V41_HEAD_DROPOUT', 0.2)


def create_digit_recognizer_v41():
    """
    v41 multi-head model: v16 backbone + two parallel 10-class heads.

    Produces two outputs:
      - tens_probs   [batch, 10]  softmax  — tens digit (0-9)
      - units_probs  [batch, 10]  softmax  — units digit (0-9)

    For 10-class mode (NB_CLASSES ≤ 10), falls back to a single-head
    standard v16 output named 'output'.
    """
    # ── 10-class fallback: behave like standard v16 single-head ──
    if params.NB_CLASSES <= 10:
        return _create_single_head_v41()

    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')

    # ==================================================================
    # Shared backbone — identical to v16
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
        (24,  4, 2),
        (24,  4, 1),
        (40,  4, 2),
        (40,  6, 1),
        (56,  6, 1),
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
    shared = tf.keras.layers.Flatten(name='flatten')(x)

    # ==================================================================
    # Two independent 10-class heads
    # ==================================================================

    head_units = _get_head_dense_units()
    head_drop = _get_head_dropout()

    # ── Tens head ──
    t = tf.keras.layers.Dense(head_units, activation=None,
                              kernel_initializer='he_normal',
                              name='tens_dense')(shared)
    t = tf.keras.layers.ReLU(max_value=6.0, name='tens_relu6')(t)
    t = tf.keras.layers.Dropout(head_drop, name='tens_dropout')(t)
    tens_act = None if params.USE_LOGITS else 'softmax'
    tens_probs = tf.keras.layers.Dense(
        10, activation=tens_act, name='tens_probs'
    )(t)

    # ── Units head ──
    u = tf.keras.layers.Dense(head_units, activation=None,
                              kernel_initializer='he_normal',
                              name='units_dense')(shared)
    u = tf.keras.layers.ReLU(max_value=6.0, name='units_relu6')(u)
    u = tf.keras.layers.Dropout(head_drop, name='units_dropout')(u)
    units_act = None if params.USE_LOGITS else 'softmax'
    units_probs = tf.keras.layers.Dense(
        10, activation=units_act, name='units_probs'
    )(u)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=[tens_probs, units_probs],
        name='digit_recognizer_v41'
    )

    p = model.count_params()
    print(f"✅ v41 multi-head model created — {p:,} params")
    print(f"   Heads: tens(10) + units(10)  |  head_dense={head_units}  dropout={head_drop}")

    return model


def _create_single_head_v41():
    """
    Fallback for NB_CLASSES <= 10: produce a single 'output' head
    (identical to v16) so the training pipeline doesn't need special
    handling for 10-class mode.
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')

    # Shared backbone — identical to v16
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='entry_conv'
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name='entry_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='entry_relu6')(x)

    inv_res_config = [
        (24,  4, 2), (24,  4, 1), (40,  4, 2), (40,  6, 1), (56,  6, 1),
    ]
    for i, (out_ch, t, s) in enumerate(inv_res_config):
        x = _inv_res(x, filters_out=out_ch, expand_ratio=t, stride=s,
                     name_prefix=f'ir{i+1}')

    x = tf.keras.layers.Conv2D(
        96, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='head_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='head_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='head_relu6')(x)
    x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name='gap')(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)

    if params.USE_LOGITS:
        outputs = tf.keras.layers.Dense(
            params.NB_CLASSES, activation=None, name='logits'
        )(x)
    else:
        outputs = tf.keras.layers.Dense(
            params.NB_CLASSES, activation='softmax', name='output'
        )(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='digit_recognizer_v41'
    )
    print(f"✅ v41 single-head fallback ({params.NB_CLASSES}cls) — {model.count_params():,} params")
    return model


# ---------------------------------------------------------------------------
# QAT wrapper
# ---------------------------------------------------------------------------

def create_qat_model(base_model=None):
    if base_model is None:
        base_model = create_digit_recognizer_v41()
    if not QAT_AVAILABLE:
        print("Warning: QAT not available. Returning base model.")
        return base_model
    try:
        with tfmot.quantization.keras.quantize_scope():
            qat_model = tfmot.quantization.keras.quantize_model(base_model)
        print("QAT model created for digit_recognizer_v41")
        return qat_model
    except Exception as e:
        print(f"QAT failed ({e}) – returning base model.")
        return base_model


# ---------------------------------------------------------------------------
# Main — self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test multi-head mode (100-class)
    print("=" * 60)
    print("v41 Multi-Head (100-class)")
    print("=" * 60)
    params.NB_CLASSES = 100
    params.update_derived_parameters()
    m = create_digit_recognizer_v41()
    m.summary()
    p = m.count_params()
    print(f"\nTotal parameters  : {p:,}")
    print(f"Estimated INT8 KB  : ~{p * 1.1 / 1024:.1f}")

    dummy = tf.random.uniform((1, params.INPUT_HEIGHT, params.INPUT_WIDTH, params.INPUT_CHANNELS))
    outs = m(dummy, training=False)
    print(f"Number of outputs : {len(outs)}")
    for i, o in enumerate(outs):
        print(f"  Output {i}: shape={o.shape}  sum≈{o.numpy().sum():.3f}")

    # Combine to get full 100-class prediction
    tens = tf.argmax(outs[0], axis=-1).numpy()
    units = tf.argmax(outs[1], axis=-1).numpy()
    combined = tens * 10 + units
    print(f"Combined prediction: {combined}  (tens={tens}, units={units})")

    # Test single-head fallback (10-class)
    print("\n" + "=" * 60)
    print("v41 Single-Head Fallback (10-class)")
    print("=" * 60)
    params.NB_CLASSES = 10
    params.update_derived_parameters()
    m10 = create_digit_recognizer_v41()
    m10.summary()
    p10 = m10.count_params()
    print(f"\nTotal parameters  : {p10:,}")
    dummy10 = tf.random.uniform((1, params.INPUT_HEIGHT, params.INPUT_WIDTH, params.INPUT_CHANNELS))
    y10 = m10(dummy10, training=False)
    print(f"Output shape      : {y10.shape}")