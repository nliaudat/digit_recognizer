"""
digit_recognizer_v40: Soft Binarization Preprocessor

A trainable image preprocessing frontend that learns to binarize input images
before classification.

Preprocessor pipeline (end-to-end trainable):
  1. Learnable luminance conversion — 1×1 conv 3→1 with BT.601 init (linear)
  2. AdaptiveBinarization layer — learnable threshold + sigmoid with learnable
     sharpness
  3. Stack to 3-channel (repeat grayscale)
  4. 3×3 conv → BN → ReLU6 stem

Architecture fingerprint:
  j:`3`  (Conv2D 3x3 kernel, not 5x5)
  l:`0`  (standard Conv2D, not InvertedResidual)
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

_w3 = 0.5  # width_multiplier for digit_recognizer_v40
PREPROC_V40_THRESHOLD_INIT = _models_cfg.PREPROC_V40_THRESHOLD_INIT
PREPROC_V40_SHARPNESS_INIT = _models_cfg.PREPROC_V40_SHARPNESS_INIT
PREPROC_V40_SHARPNESS_MIN = _models_cfg.PREPROC_V40_SHARPNESS_MIN
PREPROC_V40_SHARPNESS_MAX = _models_cfg.PREPROC_V40_SHARPNESS_MAX


def _make_divisible(v: float, divisor: int = 8) -> int:
    """Round `v` up to the nearest multiple of `divisor`."""
    vv = int(v + divisor // 2) // divisor * divisor
    return max(vv, divisor)


# ---------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package='digit_recognizer')
class _ClipConstraint(tf.keras.constraints.Constraint):
    """Keras-compatible constraint that clips weight to [min_val, max_val]."""

    def __init__(self, min_val: float, max_val: float):
        self._min = min_val
        self._max = max_val

    def __call__(self, w):
        return tf.clip_by_value(w, self._min, self._max)

    def get_config(self):
        return {'min_val': self._min, 'max_val': self._max}


@tf.keras.utils.register_keras_serializable(package='digit_recognizer')
class AdaptiveBinarization(tf.keras.layers.Layer):  # pylint: disable=abstract-method
    """Adaptive binarization with learnable threshold and sharpness.

    Applies a sigmoid to soft-binarize around a learnable threshold, with a
    learnable sharpness parameter controlling the steepness.

    ``output = sigmoid((x - threshold) * sharpness)``

    Sharpness is constrained to [sharpness_min, sharpness_max] via a weight
    constraint, so gradients never vanish during backprop (unlike
    tf.clip_by_value in the call method).
    """

    def __init__(
        self,
        threshold_init: float = PREPROC_V40_THRESHOLD_INIT,
        sharpness_init: float = PREPROC_V40_SHARPNESS_INIT,
        sharpness_min: float = PREPROC_V40_SHARPNESS_MIN,
        sharpness_max: float = PREPROC_V40_SHARPNESS_MAX,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._threshold_init = threshold_init
        self._sharpness_init = sharpness_init
        self._sharpness_min = sharpness_min
        self._sharpness_max = sharpness_max

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
            # Use constraint instead of tf.clip_by_value in call()
            # to avoid dead gradients when weight goes out of bounds.
            constraint=_ClipConstraint(self._sharpness_min, self._sharpness_max),
        )
        super().build(input_shape)

    def call(self, inputs):
        return tf.sigmoid((inputs - self.threshold) * self.sharpness)

    def get_config(self):
        config = super().get_config()
        config.update({
            'threshold_init': self._threshold_init,
            'sharpness_init': self._sharpness_init,
            'sharpness_min': self._sharpness_min,
            'sharpness_max': self._sharpness_max,
        })
        return config

    # --- QAT compatibility: provide a default QuantizeConfig so tfmot
    #     doesn't reject or silently skip this layer ---
    def get_quantize_config(self):
        """Return a Default8BitQuantizeConfig for the trainable scalars.

        This prevents ``tfmot.quantization.keras.quantize_model()`` from either
        rejecting the custom layer (which forces a fallback to the base model)
        or silently skipping it.  The two scalars (threshold, sharpness) are
        trivially cheap to quantize; the real quantisation value is in the
        classification backbone's convolutions.
        """
        try:
            import tensorflow_model_optimization as tfmot  # pylint: disable=import-outside-toplevel
        except ImportError:
            return None  # tfmot not available — nothing to configure
        # Treat the scalar weights as 8-bit quantizable
        return tfmot.quantization.keras.Default8BitQuantizeConfig(
            weight_keys=['threshold', 'sharpness'],
            act_key=None,      # no activation quantisation needed for a sigmoid
            quantize_output=False,
        )


# ---------------------------------------------------------------------------
def _conv_block(x, filters: int, stride: int, expansion_factor: int, name_prefix: str):
    """Standard Conv2D block with optional expansion."""
    input_filters = x.shape[-1]

    if expansion_factor > 1:
        ch_exp = _make_divisible(input_filters * expansion_factor)
        x = tf.keras.layers.Conv2D(ch_exp, (1, 1), padding='same', use_bias=False,
                                   name=f'{name_prefix}_expand')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_expand_bn')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_expand_relu6')(x)

    x = tf.keras.layers.Conv2D(filters, (3, 3), strides=stride, padding='same',
                               use_bias=False, name=f'{name_prefix}_conv3')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_bn3')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name=f'{name_prefix}_relu6_3')(x)
    return x


# ---------------------------------------------------------------------------
#  Factory entry point (required by model_factory.py)
# ---------------------------------------------------------------------------
def create_digit_recognizer_v40(num_classes=None, input_shape=None, **kwargs):
    """Factory entry point for digit_recognizer_v40.

    Args:
        num_classes: Optional override for NB_CLASSES (10 or 100).
        input_shape: Optional override for INPUT_SHAPE.
        **kwargs: Ignored additional keyword arguments (compatibility).
    """
    cls = num_classes if num_classes is not None else params.NB_CLASSES
    shape = input_shape if input_shape is not None else params.INPUT_SHAPE
    activation_fn = None if params.USE_LOGITS else tf.keras.activations.softmax
    return build_model(input_shape=shape, num_classes=cls, activation_fn=activation_fn)


def build_model(
    input_shape: tuple,
    num_classes: int,
    width_mult: float = _w3,
    activation_fn: Callable | None = None,
) -> tf.keras.Model:
    """Build digit_recognizer_v40 with soft binarization preprocessor."""
    _w = lambda f: _make_divisible(f * width_mult)

    inputs = tf.keras.layers.Input(shape=input_shape, name='input')

    # --- Trainable preprocessing frontend ---
    # 1. Learnable luminance (BT.601 init): Input 3-channel → grayscale
    #    Linear activation preserves full dynamic range (no sigmoid squashing).
    x = tf.keras.layers.Conv2D(
        1, (1, 1), padding='same',
        activation=None,
        kernel_initializer=tf.keras.initializers.Constant(
            [[[[0.299], [0.587], [0.114]]]]  # BT.601 init, shape (1,1,3,1)
        ),
        bias_initializer='zeros',
        use_bias=True,
        name='luminance_conv'
    )(inputs)

    # 2. Adaptive binarization (learnable threshold + sharpness)
    x = AdaptiveBinarization(
        name='adaptive_binarization'
    )(x)

    # 3. Stack to 3-channel (repeat grayscale across 3 channels)
    x = tf.keras.layers.Concatenate(name='stack_to_3ch')([x, x, x])

    # 4. Stem: 3×3 conv (standard Conv2D, not depthwise)
    x = tf.keras.layers.Conv2D(
        _w(16), (3, 3), strides=(2, 2), padding='same',
        use_bias=False, name='stem_conv3',
    )(x)
    x = tf.keras.layers.BatchNormalization(name='stem_bn3')(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name='stem_relu6_3')(x)

    # --- Classification backbone ---
    filters = [_w(24), _w(32), _w(48), _w(64), _w(96)]
    expansion_factors = [2, 2, 2, 1, 1]
    strides = [1, 2, 2, 1, 1]

    for i, (f, e, s) in enumerate(zip(filters, expansion_factors, strides)):
        x = _conv_block(x, f, s, e, name_prefix=f'block_{i}')

    # Head
    x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name='gap')(x)
    x = tf.keras.layers.Dropout(0.0, name='dropout')(x)
    outputs = tf.keras.layers.Conv2D(
        num_classes, (1, 1), padding='same', name='logits'
    )(x)
    outputs = tf.keras.layers.Reshape((num_classes,), name='output')(outputs)

    if activation_fn is not None:
        outputs = activation_fn(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='digit_recognizer_v40')
    return model