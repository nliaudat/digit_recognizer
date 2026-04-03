# utils/keras_helper.py
import tensorflow as tf
import os

# --------------------------------------------------------------------------- #
#  Keras 2 / 3 Compatibility Helper
# --------------------------------------------------------------------------- #
# This module provides a unified way to access Keras and TFMoT across the 
# project, handling the transition between Keras 2 (tf_keras) and Keras 3 (tf.keras).
#
# Rule:
# If QAT is enabled, we MUST use tf_keras (Keras 2) because tfmot does not
# yet support Keras 3.
# --------------------------------------------------------------------------- #

def get_keras_backend():
    """
    Returns the appropriate Keras module based on project parameters.
    """
    import parameters as params
    
    # Check if QAT is requested or if we are forced into Keras 2
    use_keras2 = getattr(params, 'USE_QAT', False) or os.environ.get('FORCE_KERAS2', '0') == '1'
    
    if use_keras2:
        try:
            import tf_keras as keras
            # print("🛠️  Using Keras 2 (tf_keras) backend for QAT compatibility")
            return keras
        except ImportError:
            # print("⚠️  tf_keras not found, falling back to tf.keras (Keras 3)")
            return tf.keras
    else:
        # Default to modern Keras (Keras 3 in TF 2.16+)
        return tf.keras

def get_tfmot():
    """
    Returns the TFMoT module, or None if unavailable.
    """
    try:
        import tensorflow_model_optimization as tfmot
        return tfmot
    except ImportError:
        return None

def is_keras3():
    """Returns True if the active Keras is Keras 3."""
    k = get_keras_backend()
    return hasattr(k, '__version__') and k.__version__.startswith('3')

# Convenience exports
keras = get_keras_backend()
tfmot = get_tfmot()
