"""
digit_recognizer_v39: Gated Multi-Scale Depthwise Fusion

Gated multi-scale fusion: 3×3 and 5×5 each get half the channels, then a
learned 1×1 conv produces per-channel sigmoid gates that blend the two
branches. Works well for 10-class compact models.

Architecture fingerprint:
  j:`1`  (Conv2D 3x3 kernel, not 5x5)
  l:`2`  (2-layer InvertedResidualGated, not regular InvertedResidual)
    - Expansion → split half channels
    - 3×3 DW + BN + ReLU6 on first half
    - 5×5 DW + BN + ReLU6 on second half
    - 1×1 conv → sigmoid gating → weighted sum
    - 1×1 projection
  e:`2`  (expansion factor 2 in intermediate layers)
  n:`0.0`  (no dropout)
  u:`64`  (64 base filters, Conv2D stem + 5 levels)
  c:`0.0`  (0 prior-weights on classification head)
  d:    (image order: HWC)
  m:`0`  (ModelNorm: no extra output head normalization)
"""

from typing import Callable

import tensorflow as tf

import config as params
import config.models as _models_cfg

_w3 = 0.75  # width_multiplier for digit_recognizer_v39
MSDW_FUSION_DUAL_KERNEL_MIN_WIDTH = _models_cfg.MSDW_FUSION_DUAL_KERNEL_MIN_WIDTH


def _make_divisible(v: float, divisor: int = 8) -> int:
    """Round `v` up to the nearest multiple of `divisor`."""
    vv = int(v + divisor // 2) // divisor * divisor
    return max(vv, divisor)


# ---------------------------------------------------------------------------
def _conv_stem(x, filters: int, name_prefix: str = 'stem'):
    """Standard Conv2D stem (3×3 conv -> BN -> ReLU6)."""
    x = tf.keras.layers.Conv2D(
        filters,
        (3, 3),
        strides=(2, 2),
        padding='same',
        use_bias=False,
        name=f'{name_prefix}_conv',
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_bn')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_relu6')(x)
    return x


# ---------------------------------------------------------------------------
def _inv_res_multi(
    x,
    expansion_factor: int,
    output_filters: int,
    stride: int,
    spatial_w: int,
    name_prefix: str,
):
    """Gated multi-scale depthwise inverted residual block.

    Expansion → split channels into two halves.
    Half 1: 3×3 DW + BN + ReLU6
    Half 2: 5×5 DW + BN + ReLU6
    Gated fusion: 1×1 conv → sigmoid gate → weighted sum of branches
    Project to output_filters.
    """
    input_filters = x.shape[-1]
    ch_exp = _make_divisible(input_filters * expansion_factor)

    # Expansion
    y = tf.keras.layers.Conv2D(
        ch_exp,
        (1, 1),
        padding='same',
        use_bias=False,
        name=f'{name_prefix}_expand',
    )(x)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_expand_bn')(y)
    y = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_expand_relu6')(y)

    # Spatial mixing (only if spatial dims are large enough)
    min_w = MSDW_FUSION_DUAL_KERNEL_MIN_WIDTH

    if spatial_w is not None and spatial_w >= min_w:
        # --- Multi-scale depthwise (split-channel) ---
        half = ch_exp // 2

        # 3×3 branch: BN + ReLU6 after DW
        b3 = tf.keras.layers.DepthwiseConv2D(
            (3, 3), strides=stride, padding='same',
            depthwise_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_dw_3'
        )(y[:, :, :, :half])
        b3 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_dw_3_bn')(b3)
        b3 = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_dw_3_relu6')(b3)

        # 5×5 branch: BN + ReLU6 after DW
        b5 = tf.keras.layers.DepthwiseConv2D(
            (5, 5), strides=stride, padding='same',
            depthwise_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_dw_5'
        )(y[:, :, :, half:])
        b5 = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_dw_5_bn')(b5)
        b5 = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_dw_5_relu6')(b5)

        # --- Gated fusion: 1×1 conv → sigmoid → weighted sum ---
        concat = tf.keras.layers.Concatenate(axis=-1, name=f'{name_prefix}_dw_concat')([b3, b5])
        gates = tf.keras.layers.Conv2D(
            ch_exp, (1, 1), padding='same',
            activation='sigmoid',
            kernel_initializer=tf.keras.initializers.Constant(0.5),
            use_bias=True,
            bias_initializer='zeros',
            name=f'{name_prefix}_gate'
        )(concat)
        # Weighted sum: gates * concat (element-wise)
        y = tf.keras.layers.Multiply(name=f'{name_prefix}_gated_fusion')([gates, concat])
    else:
        # Fallback: regular depthwise conv
        y = tf.keras.layers.DepthwiseConv2D(
            (3, 3), strides=stride, padding='same',
            depthwise_initializer='he_normal', use_bias=False,
            name=f'{name_prefix}_dw'
        )(y)
        y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_dw_bn')(y)
        y = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_dw_relu6')(y)

    # Project
    y = tf.keras.layers.Conv2D(
        output_filters, (1, 1), padding='same', use_bias=False, name=f'{name_prefix}_project'
    )(y)
    y = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_project_bn')(y)

    # Residual
    if stride == 1 and input_filters == output_filters:
        y = tf.keras.layers.Add(name=f'{name_prefix}_add')([x, y])

    return y

# ---------------------------------------------------------------------------
#  Factory entry point (required by model_factory.py)
# ---------------------------------------------------------------------------
def create_digit_recognizer_v39(num_classes=None, input_shape=None, **kwargs):
    """Factory entry point for digit_recognizer_v39.

    Args:
        num_classes: Optional override for NB_CLASSES (10 or 100).
        input_shape: Optional override for INPUT_SHAPE.
        **kwargs: Ignored additional keyword arguments (compatibility).
    """
    cls = num_classes if num_classes is not None else params.NB_CLASSES
    shape = input_shape if input_shape is not None else params.INPUT_SHAPE
    # Determine activation based on USE_LOGITS
    activation_fn = None if params.USE_LOGITS else tf.keras.activations.softmax
    return build_model(input_shape=shape, num_classes=cls, activation_fn=activation_fn)


def build_model(
    input_shape: tuple,
    num_classes: int,
    width_mult: float = _w3,
    activation_fn: Callable | None = None,
) -> tf.keras.Model:
    """Build digit_recognizer_v39 with gated multi-scale depthwise blocks."""
    _w = lambda f: _make_divisible(f * width_mult)

    inputs = tf.keras.layers.Input(shape=input_shape, name='input')

    filters = [16, 24, 32, 48, 64]
    filters = [_w(f) for f in filters]
    expansion_factors = [1, 2, 2, 2, 1]
    strides = [1, 2, 2, 1, 1]
    spatial_w = [None, 14, 7, 4, 4]

    # Stem
    x = inputs
    x = _conv_stem(x, filters[0], name_prefix='stem')

    # InvertedResidual blocks
    for i, (f, e, s, sw) in enumerate(zip(filters, expansion_factors, strides, spatial_w)):
        x = _inv_res_multi(x, e, f, s, sw, name_prefix=f'block_{i}')

    # Head
    x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name='gap')(x)
    x = tf.keras.layers.Dropout(0.0, name='dropout')(x)
    outputs = tf.keras.layers.Conv2D(
        num_classes, (1, 1), padding='same', name='logits'
    )(x)
    outputs = tf.keras.layers.Reshape((num_classes,), name='output')(outputs)

    if activation_fn is not None:
        outputs = activation_fn(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='digit_recognizer_v39')
    return model