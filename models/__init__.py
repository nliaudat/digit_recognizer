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

def combine_multiheads(model_outputs):
    """
    Combine tens and units head outputs into joint 100-class probability distribution.

    For v41 multi-head models: output is list [tens_probs, units_probs]
    each of shape (N, 10).  Joint probability P(digit = t*10+u) = P(tens=t) * P(units=u).

    Args:
        model_outputs: tensor, numpy array, or list of 2 tensors/arrays.
    Returns:
        Single tensor/array of shape (N, 100) containing joint probabilities.
        Returns input unchanged if not a 2-element list (single-head models).
    """
    if not isinstance(model_outputs, (list, tuple)) or len(model_outputs) != 2:
        return model_outputs

    tens, units = model_outputs
    # Compute outer product: (N, 10, 10) then flatten to (N, 100)
    # Works for both TF tensors and numpy arrays
    if hasattr(tens, 'shape'):
        # TF tensors
        joint = tens[..., :, tf.newaxis] * units[..., tf.newaxis, :]
        return tf.reshape(joint, (-1, 100))
    else:
        # numpy arrays
        joint = tens[..., :, np.newaxis] * units[..., np.newaxis, :]
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
