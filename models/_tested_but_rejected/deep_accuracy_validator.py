# models/deep_accuracy_validator.py
"""
deep_accuracy_validator – State-of-the-Art PC-Only Validator (2024/2025)
=========================================================================
Purpose: Reclassify ambiguous / low-quality meter images that the IoT models
fail on.  Accuracy is the only objective.  No IoT constraints apply.

Architecture incorporates 2024/2025 SOTA operators:

  1. GELU activation (Gaussian Error Linear Unit)
     – smooth, non-monotonic; stronger gradient signal than ReLU.
     – used in ConvNeXt, BERT, GPT. Consistently outperforms ReLU on
       image classification benchmarks.

  2. SiLU / Swish activation (dense head only)
     – self-gated smooth activation; empirically best for classification heads.
     – used in EfficientNet family.

  3. ConvNeXt-style large-kernel depthwise block (7×7 dw + 1×1 pw × 2)
     – expands receptive field without quadratic cost of standard Conv
     – LayerNorm instead of BatchNorm: more stable across batch sizes
     – 4× inverted bottleneck pointwise projections (from ConvNeXt paper)

  4. CBAM – Convolutional Block Attention Module
     – Dual axis: channel attention (what) + spatial attention (where)
     – Proven SOTA on fine-grained classification and noisy-image benchmarks
     – Both implemented with sigmoid gating for multiplicative feature recalibration

  5. Stochastic Depth (Drop Path)
     – Randomly drops entire residual branches during training
     – Acts as strong regulariser — reduces overfitting on small datasets
     – Used in DeiT, ConvNeXt, EfficientNetV2

  6. Label Smoothing friendly head
     – Softmax output — use parameters.LABEL_SMOOTHING = 0.05 for training

float32 only. No QAT, no quantization, no ReLU6.
"""

import tensorflow as tf
import parameters as params


# ---------------------------------------------------------------------------
#  Activation helpers
# ---------------------------------------------------------------------------

def _gelu(x, name=None):
    """Gaussian Error Linear Unit – smooth, non-monotonic."""
    return tf.keras.layers.Activation('gelu', name=name)(x)


def _silu(x, name=None):
    """SiLU / Swish: x * sigmoid(x).  Built-in since TF 2.12."""
    return tf.keras.layers.Activation('swish', name=name)(x)


# ---------------------------------------------------------------------------
#  Stochastic Depth (Drop Path)
# ---------------------------------------------------------------------------

class StochasticDepth(tf.keras.layers.Layer):
    """Randomly drop entire residual branch during training.

    Args:
        drop_rate: Probability of dropping the residual path (0 = no drop).
    Reference: "Deep Networks with Stochastic Depth", Huang et al. 2016.
    """

    def __init__(self, drop_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate

    def call(self, x, training=None):
        if (not training) or self.drop_rate == 0.0:
            return x
        keep_prob = 1.0 - self.drop_rate
        batch = tf.shape(x)[0]
        noise = tf.random.uniform([batch, 1, 1, 1], dtype=x.dtype)
        mask = tf.floor(noise + keep_prob)
        return x * mask / keep_prob   # scale so E[output] = input

    def get_config(self):
        cfg = super().get_config()
        cfg['drop_rate'] = self.drop_rate
        return cfg


# ---------------------------------------------------------------------------
#  CBAM – Convolutional Block Attention Module
# ---------------------------------------------------------------------------

def _cbam(x, ratio=8, name_prefix='cbam'):
    """Dual channel + spatial attention (CBAM, Woo et al. 2018 / widely used 2024).

    Channel attention: tells the network *what* features matter.
    Spatial attention: tells the network *where* to look.
    """
    ch = x.shape[-1]

    # ── Channel attention ──────────────────────────────────────────────────
    avg_pool = tf.keras.layers.GlobalAveragePooling2D(
        name=f'{name_prefix}_ch_avg')(x)
    max_pool = tf.keras.layers.GlobalMaxPooling2D(
        name=f'{name_prefix}_ch_max')(x)

    # Shared MLP
    shared_fc1 = tf.keras.layers.Dense(
        max(1, ch // ratio), activation='relu',
        kernel_initializer='he_normal',
        name=f'{name_prefix}_ch_fc1'
    )
    shared_fc2 = tf.keras.layers.Dense(
        ch, kernel_initializer='glorot_uniform',
        name=f'{name_prefix}_ch_fc2'
    )

    avg_out = shared_fc2(shared_fc1(avg_pool))
    max_out = shared_fc2(shared_fc1(max_pool))
    ch_att = tf.keras.layers.Add(name=f'{name_prefix}_ch_add')([avg_out, max_out])
    ch_att = tf.keras.layers.Activation(
        'sigmoid', name=f'{name_prefix}_ch_sig')(ch_att)
    ch_att = tf.keras.layers.Reshape(
        (1, 1, ch), name=f'{name_prefix}_ch_reshape')(ch_att)
    x = tf.keras.layers.Multiply(name=f'{name_prefix}_ch_mul')([x, ch_att])

    # ── Spatial attention ──────────────────────────────────────────────────
    # Compress channel axis with avg and max → concat → Conv2D(1)
    sp_avg = tf.keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
        name=f'{name_prefix}_sp_avg')(x)
    sp_max = tf.keras.layers.Lambda(
        lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
        name=f'{name_prefix}_sp_max')(x)
    sp_concat = tf.keras.layers.Concatenate(
        axis=-1, name=f'{name_prefix}_sp_concat')([sp_avg, sp_max])
    sp_att = tf.keras.layers.Conv2D(
        1, (7, 7), padding='same', activation='sigmoid',
        kernel_initializer='glorot_uniform', use_bias=False,
        name=f'{name_prefix}_sp_conv')(sp_concat)
    x = tf.keras.layers.Multiply(name=f'{name_prefix}_sp_mul')([x, sp_att])

    return x


# ---------------------------------------------------------------------------
#  ConvNeXt-style block
# ---------------------------------------------------------------------------

def _convnext_block(x, dim, drop_path_rate=0.1, name_prefix='cnx'):
    """ConvNeXt V2-inspired block:
      depth-wise 7×7 conv → LayerNorm → 4× pw expansion → GELU → pw proj
      + stochastic-depth shortcut.

    References:
      Liu et al. "A ConvNet for the 2020s" (ConvNeXt, CVPR 2022)
      Woo et al. "ConvNeXt V2" (arXiv 2023)
    """
    shortcut = x

    # Depthwise 7×7 (large receptive field with only ch^2 params × 49)
    y = tf.keras.layers.DepthwiseConv2D(
        (7, 7), padding='same', use_bias=False,
        depthwise_initializer='he_normal',
        name=f'{name_prefix}_dw'
    )(x)
    # LayerNorm (on the channel axis) – more stable than BN at variable batch sizes
    y = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-6, name=f'{name_prefix}_ln')(y)

    # Inverted bottleneck: expand × 4, GELU, project back
    y = tf.keras.layers.Conv2D(
        dim * 4, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=True,
        name=f'{name_prefix}_pw_expand'
    )(y)
    y = tf.keras.layers.Activation('gelu', name=f'{name_prefix}_gelu')(y)
    y = tf.keras.layers.Conv2D(
        dim, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=True,
        name=f'{name_prefix}_pw_proj'
    )(y)

    # Stochastic depth regularisation
    if drop_path_rate > 0.0:
        y = StochasticDepth(drop_rate=drop_path_rate, name=f'{name_prefix}_sdepth')(y)

    y = tf.keras.layers.Add(name=f'{name_prefix}_add')([shortcut, y])
    return y


# ---------------------------------------------------------------------------
#  Classic Residual + CBAM block  (for the deeper stages)
# ---------------------------------------------------------------------------

def _res_cbam_block(x, filters, strides=1, drop_path_rate=0.1, name_prefix='rcb'):
    """Standard pre-act residual + CBAM attention + stochastic depth."""
    shortcut = x

    y = tf.keras.layers.Conv2D(
        filters, (3, 3), strides=strides, padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_conv_a'
    )(x)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_bn_a')(y)
    y = tf.keras.layers.Activation('gelu', name=f'{name_prefix}_gelu_a')(y)

    y = tf.keras.layers.Conv2D(
        filters, (3, 3), strides=1, padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_conv_b'
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_bn_b')(y)

    # CBAM dual attention
    y = _cbam(y, ratio=8, name_prefix=f'{name_prefix}_cbam')

    # Stochastic depth
    if drop_path_rate > 0.0:
        y = StochasticDepth(drop_rate=drop_path_rate, name=f'{name_prefix}_sdepth')(y)

    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, (1, 1), strides=strides, padding='same',
            kernel_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_shortcut_conv'
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(
            name=f'{name_prefix}_shortcut_bn')(shortcut)

    y = tf.keras.layers.Add(name=f'{name_prefix}_add')([shortcut, y])
    y = tf.keras.layers.Activation('gelu', name=f'{name_prefix}_gelu_out')(y)
    return y


# ---------------------------------------------------------------------------
#  Inception multi-scale entry block
# ---------------------------------------------------------------------------

def _inception_entry(x, f1x1, f3x3, f5x5, name_prefix='inc'):
    """Three parallel branches concatenated.
    Captures fine texture (1×1), local features (3×3), global shape (5×5 via 3×3×2).
    All activations use GELU.
    """
    # Branch 1×1
    b1 = tf.keras.layers.Conv2D(
        f1x1, (1, 1), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_b1'
    )(x)
    b1 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_b1_bn')(b1)
    b1 = tf.keras.layers.Activation('gelu', name=f'{name_prefix}_b1_gelu')(b1)

    # Branch 3×3
    b2 = tf.keras.layers.Conv2D(
        f3x3, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_b2'
    )(x)
    b2 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_b2_bn')(b2)
    b2 = tf.keras.layers.Activation('gelu', name=f'{name_prefix}_b2_gelu')(b2)

    # Branch "5×5" via two 3×3 (parameter efficient)
    b3 = tf.keras.layers.Conv2D(
        f5x5, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_b3a'
    )(x)
    b3 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_b3a_bn')(b3)
    b3 = tf.keras.layers.Activation('gelu', name=f'{name_prefix}_b3a_gelu')(b3)
    b3 = tf.keras.layers.Conv2D(
        f5x5, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name=f'{name_prefix}_b3b'
    )(b3)
    b3 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_b3b_bn')(b3)
    b3 = tf.keras.layers.Activation('gelu', name=f'{name_prefix}_b3b_gelu')(b3)

    return tf.keras.layers.Concatenate(axis=-1, name=f'{name_prefix}_concat')([b1, b2, b3])


# ---------------------------------------------------------------------------
#  Model builder
# ---------------------------------------------------------------------------

def create_deep_accuracy_validator():
    """
    State-of-the-art PC-only classifier for ambiguous meter image reclassification.

    Stage 0  – Entry Conv(64) + BN + GELU
    Stage 1  – Multi-scale Inception entry → CBAM attention
    Stage 2  – 2× ConvNeXt-style blocks (large-kernel dw + GELU + LayerNorm)
    Stage 3  – 2× Res+CBAM blocks(128) with stochastic depth
    Stage 4  – 2× Res+CBAM blocks(256) with stochastic depth
    Head     – GAP → Dense(512, SiLU) → BN → Dropout(0.4)
                   → Dense(256, SiLU) → BN → Dropout(0.3)
                   → Dense(NB_CLASSES, Softmax)
    """
    inputs = tf.keras.Input(shape=params.INPUT_SHAPE, name='input')

    # ------------------------------------------------------------------
    # Stage 0 – Entry
    # ------------------------------------------------------------------
    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same',
        kernel_initializer='he_normal', use_bias=False,
        name='entry_conv'
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name='entry_bn')(x)
    x = tf.keras.layers.Activation('gelu', name='entry_gelu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='entry_pool')(x)

    # ------------------------------------------------------------------
    # Stage 1 – Multi-scale Inception + CBAM
    # (32 + 48 + 16 = 96 channels out)
    # ------------------------------------------------------------------
    x = _inception_entry(x, f1x1=32, f3x3=48, f5x5=16, name_prefix='inc1')
    x = _cbam(x, ratio=8, name_prefix='inc1_cbam')

    # ------------------------------------------------------------------
    # Stage 2 – Two ConvNeXt blocks at 96-ch (large-kernel + GELU + LayerNorm)
    # ------------------------------------------------------------------
    x = _convnext_block(x, dim=96, drop_path_rate=0.05, name_prefix='cnx1')
    x = _convnext_block(x, dim=96, drop_path_rate=0.10, name_prefix='cnx2')
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool_cnx')(x)

    # ------------------------------------------------------------------
    # Stage 3 – Residual + CBAM at 128 ch
    # ------------------------------------------------------------------
    x = _res_cbam_block(x, filters=128, strides=1, drop_path_rate=0.10, name_prefix='rs3a')
    x = _res_cbam_block(x, filters=128, strides=1, drop_path_rate=0.15, name_prefix='rs3b')
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2, name='pool_rs3')(x)

    # ------------------------------------------------------------------
    # Stage 4 – Residual + CBAM at 256 ch
    # ------------------------------------------------------------------
    x = _res_cbam_block(x, filters=256, strides=1, drop_path_rate=0.15, name_prefix='rs4a')
    x = _res_cbam_block(x, filters=256, strides=1, drop_path_rate=0.20, name_prefix='rs4b')

    # ------------------------------------------------------------------
    # Classifier head  (SiLU / Swish for smoother probability calibration)
    # ------------------------------------------------------------------
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)

    x = tf.keras.layers.Dense(512, use_bias=True, name='fc1')(x)
    x = tf.keras.layers.BatchNormalization(name='fc1_bn')(x)
    x = tf.keras.layers.Activation('swish', name='fc1_swish')(x)
    x = tf.keras.layers.Dropout(0.40, name='fc1_drop')(x)

    x = tf.keras.layers.Dense(256, use_bias=True, name='fc2')(x)
    x = tf.keras.layers.BatchNormalization(name='fc2_bn')(x)
    x = tf.keras.layers.Activation('swish', name='fc2_swish')(x)
    x = tf.keras.layers.Dropout(0.30, name='fc2_drop')(x)

    outputs = tf.keras.layers.Dense(
        params.NB_CLASSES, activation='softmax', name='output'
    )(x)

    return tf.keras.Model(inputs, outputs, name="deep_accuracy_validator")


# ---------------------------------------------------------------------------
#  QAT wrapper – explicitly disabled
# ---------------------------------------------------------------------------

def create_qat_model(base_model=None):
    """
    deep_accuracy_validator is PC-only float32.
    No quantization is applied. Function exists for API compatibility only.
    """
    print("=" * 60)
    print("deep_accuracy_validator: QAT disabled (PC-only float32 model)")
    print("=" * 60)
    if base_model is None:
        base_model = create_deep_accuracy_validator()
    return base_model


# ---------------------------------------------------------------------------
#  Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = create_deep_accuracy_validator()
    model.summary()
    total = model.count_params()
    print(f"\nTotal parameters : {total:,}")
    print(f"Float32 size     : ~{total * 4 / 1024 / 1024:.1f} MB")
    print()
    print("Operators used:")
    print("  - GELU activation (entry, inception, ConvNeXt, residual stages)")
    print("  - SiLU/Swish (dense classifier head)")
    print("  - ConvNeXt 7x7 depthwise conv + LayerNorm (Stage 2)")
    print("  - CBAM dual channel+spatial attention (Stage 1, 3, 4)")
    print("  - Stochastic Depth Drop-Path (stages 2-4)")
    print("  - Multi-scale Inception entry (Stage 1)")
