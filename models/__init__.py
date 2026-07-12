# models/__init__.py
# Minimal imports to avoid circular dependencies
from .model_factory import (
    create_model, 
    compile_model, 
    model_summary, 
    get_available_models, 
    get_model_info,
    get_hyperparameter_summary,
    print_hyperparameter_summary,
    create_learning_rate_scheduler,
    get_initializer,
    get_regularizer,
    get_training_callbacks
)

import tensorflow as tf
import numpy as np

def combine_multiheads(model_outputs, model=None):
    """
    Combine head outputs into joint 100-class probability distribution.

    For v41 multi-head models: output is list [tens_probs, units_probs]
    each of shape (N, 10).  Joint probability P(digit = t*10+u) = P(tens=t) * P(units=u).

    For v42 with 12 outputs: output is [integer_probs, decimal_probs, decimal_head_0..9_probs].
    Uses the individual decimal heads to compute the correct joint prediction:
        digit = argmax(integer_probs) * 10 + argmax(decimal_head_{int_pred}_probs)

    For v42 with 2 outputs: falls back to argmax(integer_probs) * 10 + argmax(decimal_probs)
    (marginal approximation; minor accuracy impact for training callbacks).

    Args:
        model_outputs: tensor, numpy array, or list of 2+ tensors/arrays.
        model: optional Keras model (used to detect v42 by output names).
    Returns:
        Single tensor/array of shape (N, 100) containing joint probabilities.
        Returns input unchanged if not a list/tuple (single-head models).
    """
    if not isinstance(model_outputs, (list, tuple)):
        return model_outputs

    # ── Detect v42 12-output model by checking output count and names ──
    is_v42_12 = (
        len(model_outputs) >= 12
        and model is not None
        and hasattr(model, 'output_names')
        and 'integer_probs' in model.output_names
        and 'decimal_head_0_probs' in model.output_names
    )

    if is_v42_12:
        # v42 12-output: integer_probs[0], decimal_probs[1], heads[2:12]
        int_head = model_outputs[0]
        dec_heads_list = model_outputs[2:12]  # 10 arrays/tensors each (N, 10)

        is_tf = tf.is_tensor(int_head)
        if is_tf:
            int_pred = tf.cast(tf.argmax(int_head, axis=-1), tf.int32)  # (N,)
            stacked = tf.stack(dec_heads_list, axis=1)  # (N, 10, 10)
            batch_size = tf.shape(stacked)[0]
            idx = tf.stack([tf.range(batch_size), int_pred], axis=1)  # (N, 2)
            selected = tf.gather_nd(stacked, idx)  # (N, 10)
            dec_pred = tf.cast(tf.argmax(selected, axis=-1), tf.int32)
            combined = int_pred * 10 + dec_pred
            return tf.one_hot(combined, 100, dtype=tf.float32)
        else:
            int_pred = np.atleast_1d(np.argmax(int_head, axis=-1))  # (N,)
            stacked = np.stack(dec_heads_list, axis=1)  # (N, 10, 10)
            selected = stacked[np.arange(len(int_pred)), int_pred]  # (N, 10)
            dec_pred = np.atleast_1d(np.argmax(selected, axis=-1))
            combined = int_pred * 10 + dec_pred
            joint = np.zeros((len(combined), 100), dtype=np.float32)
            joint[np.arange(len(combined)), combined] = 1.0
            return joint

    # ── 2-output models (v41, v42 2-out, or unknown multi-head) ──
    if len(model_outputs) != 2:
        return model_outputs

    head0, head1 = model_outputs

    # Auto-detect v42 by checking output names if model is provided.
    is_v42 = False
    if model is not None and hasattr(model, 'output_names') and len(model.output_names) >= 2:
        is_v42 = 'integer_probs' in model.output_names and 'decimal_probs' in model.output_names

    if is_v42:
        # decimal_probs is the marginal Σ P(int=i)×P(dec|int=i), so its argmax may
        # come from a different integer than argmax(integer_probs).  This means the
        # combined prediction can be *invalid* (e.g. integer=2, decimal=7 producing 27
        # when the model's own internal heads would choose head 2's best decimal).
        #
        # For precise v42 evaluation use _evaluate_keras_multihead() which extracts
        # individual decimal_head_{i}_probs.  During training callbacks the ~1%
        # discrepancy from the marginal approximation is acceptable for early stopping.
        is_tf = tf.is_tensor(head0)
        if is_tf:
            int_pred = tf.cast(tf.argmax(head0, axis=-1), tf.int32)
            dec_pred = tf.cast(tf.argmax(head1, axis=-1), tf.int32)
            combined = int_pred * 10 + dec_pred
            return tf.one_hot(combined, 100, dtype=tf.float32)
        else:
            int_pred = np.atleast_1d(np.argmax(head0, axis=-1))
            dec_pred = np.atleast_1d(np.argmax(head1, axis=-1))
            combined = int_pred * 10 + dec_pred
            joint = np.zeros((len(combined), 100), dtype=np.float32)
            joint[np.arange(len(combined)), combined] = 1.0
            return joint

    # v41 (and default fallback): outer product joint distribution.
    if tf.is_tensor(head0):
        joint = head0[..., :, tf.newaxis] * head1[..., tf.newaxis, :]
        return tf.reshape(joint, (-1, 100))
    else:
        joint = head0[..., :, np.newaxis] * head1[..., np.newaxis, :]
        return joint.reshape((-1, 100))

# Core models will be loaded dynamically via model_factory.py to respect run-time parameters
__all__ = [
    'create_model',
    'compile_model', 
    'model_summary',
    'get_available_models',
    'get_model_info',
    'get_hyperparameter_summary',
    'print_hyperparameter_summary',
    'create_learning_rate_scheduler',
    'get_initializer',
    'get_regularizer',
    'get_training_callbacks',
    'combine_multiheads',
]

# ---------------------------------------------------------------------------
# Global Custom Object Registration for Keras load_model
# ---------------------------------------------------------------------------
try:
    from models.digit_recognizer_v42 import SoftConditioningCombine, NoOpQuantizeConfig
    tf.keras.utils.get_custom_objects().update({
        'SoftConditioningCombine': SoftConditioningCombine,
        'NoOpQuantizeConfig': NoOpQuantizeConfig
    })
except Exception:
    pass

try:
    from models.convnext_blocks import DropPath
    tf.keras.utils.get_custom_objects().update({
        'DropPath': DropPath
    })
except Exception:
    pass

try:
    from models.digit_recognizer_v38 import RepVGGBlock, RepVGGModel
    tf.keras.utils.get_custom_objects().update({
        'RepVGGBlock': RepVGGBlock,
        'RepVGGModel': RepVGGModel
    })
except Exception:
    pass

try:
    from models.digit_recognizer_v40 import AdaptiveBinarization, _ClipConstraint
    tf.keras.utils.get_custom_objects().update({
        'AdaptiveBinarization': AdaptiveBinarization,
        '_ClipConstraint': _ClipConstraint
    })
except Exception:
    pass
