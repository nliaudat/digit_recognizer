# utils/keras_helper.py
import tensorflow as tf
import os
import sys
import importlib

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

# --------------------------------------------------------------------------- #
#  Apply Backend Fixes & Patches
# --------------------------------------------------------------------------- #
# These patches resolve internal Keras module mismatches and serialization
# issues (e.g., Functional class deserialization error in modern tf-keras).
# --------------------------------------------------------------------------- #
def _apply_backend_patches(k):
    if k.__name__ == 'tf_keras':
        try:
            # Map tf_keras.src.models.functional to its engine location if missing
            if 'tf_keras.src.models.functional' not in sys.modules:
                try:
                    import tf_keras.src.engine.functional as functional
                    sys.modules['tf_keras.src.models.functional'] = functional
                except ImportError:
                    pass
            
            # Map other common internal paths
            alias_map = {
                'tf_keras.src.models': 'tf_keras.models',
                'tf_keras.src.layers': 'tf_keras.layers',
                'tf_keras.src.losses': 'tf_keras.losses',
                'tf_keras.src.metrics': 'tf_keras.metrics',
                'tf_keras.src.optimizers': 'tf_keras.optimizers',
                'tf_keras.src.initializers': 'tf_keras.initializers',
                'tf_keras.src.regularizers': 'tf_keras.regularizers',
                'tf_keras.src.constraints': 'tf_keras.constraints',
                'tf_keras.src.activations': 'tf_keras.activations',
            }
            
            for target, source in alias_map.items():
                if target not in sys.modules:
                    try:
                        sys.modules[target] = importlib.import_module(source)
                    except (ImportError, ModuleNotFoundError):
                        pass

            # Map save_model internal module if needed
            if 'tf_keras.src.saving.serialization_lib' not in sys.modules:
                try:
                    import tf_keras.src.saving.legacy.serialization as legacy_serialization
                    sys.modules['tf_keras.src.saving.serialization_lib'] = legacy_serialization
                except ImportError:
                    pass

            # ── HELPER: Recursive Config Cleaner ───────────────────────────
            def clean_keras3_config(config):
                """Recursively clean Keras 3 configuration dictionaries for Keras 2."""
                if not isinstance(config, dict):
                    return config
                
                # Check for Keras 3 DTypePolicy dictionary
                if 'dtype' in config and isinstance(config['dtype'], dict):
                    d = config['dtype']
                    if d.get('class_name') == 'DTypePolicy':
                        config['dtype'] = d.get('config', {}).get('name', 'float32')
                
                # Check for Keras 3 nested class dictionaries (initializers, etc)
                for key, value in config.items():
                    if isinstance(value, dict) and 'module' in value and 'class_name' in value:
                        value.pop('module', None)
                        value.pop('registered_name', None)
                        config[key] = clean_keras3_config(value)
                    elif isinstance(value, dict):
                        config[key] = clean_keras3_config(value)
                    elif isinstance(value, list):
                        # Fix for Functional inbound_nodes string error
                        # In Keras 3, nodes are simplified; Keras 2 expects a list with indices.
                        if key == 'inbound_nodes' and len(value) > 0:
                            new_nodes = []
                            for node in value:
                                # Keras 2 Expects: [[layer_name, node_index, tensor_index, metadata], ...]
                                if isinstance(node, list):
                                    new_node = []
                                    for entry in node:
                                        if isinstance(entry, str):
                                            # Convert simplified string ref back to 4-tuple
                                            new_node.append([entry, 0, 0, {}])
                                        else:
                                            new_node.append(entry)
                                    new_nodes.append(new_node)
                                elif isinstance(node, str):
                                    new_nodes.append([[node, 0, 0, {}]])
                                else:
                                    new_nodes.append(node)
                            config[key] = new_nodes
                        else:
                            config[key] = [clean_keras3_config(i) if isinstance(i, dict) else i for i in value]
                
                # Standard parameter mapping
                if 'batch_shape' in config and 'batch_input_shape' not in config:
                    config['batch_input_shape'] = config.pop('batch_shape')
                config.pop('sparse', None)
                config.pop('ragged', None)
                
                return config

            # ── NUCLEAR PATCH for Layer ─────────────────────────────────────
            from tf_keras.src.engine.base_layer import Layer as BaseLayer
            original_layer_from_config = BaseLayer.from_config
            @classmethod
            def robust_layer_from_config(cls, config):
                config = clean_keras3_config(config)
                try:
                    return original_layer_from_config(config)
                except Exception:
                    try:
                        return cls(**config)
                    except Exception:
                        raise
            BaseLayer.from_config = robust_layer_from_config

            # ── NUCLEAR PATCH for Model (Functional compatibility) ──────────
            from tf_keras.src.engine.training import Model as BaseModel
            original_model_from_config = BaseModel.from_config
            @classmethod
            def robust_model_from_config(cls, config, custom_objects=None):
                # Clean layers and connections recursively
                if 'layers' in config:
                    config['layers'] = [clean_keras3_config(l) for l in config['layers']]
                
                # Resolve common Keras 3 to 2 incompatibilities in input/output nodes
                for k_node in ['input_layers', 'output_layers']:
                    if k_node in config:
                        new_node_list = []
                        for entry in config[k_node]:
                            if isinstance(entry, str):
                                new_node_list.append([entry, 0, 0])
                            else:
                                new_node_list.append(entry)
                        config[k_node] = new_node_list
                
                return original_model_from_config(config, custom_objects=custom_objects)
            
            BaseModel.from_config = robust_model_from_config

        except Exception:
            pass
    return k

# Convenience exports
keras = _apply_backend_patches(get_keras_backend())
tfmot = get_tfmot()

# --------------------------------------------------------------------------- #
#  Smart Model Loader (Keras 2 & 3 Support)
# --------------------------------------------------------------------------- #
def robust_load_model(filepath, custom_objects=None, **kwargs):
    """
    Attempts to load a model using current backend.
    """
    return keras.models.load_model(filepath, custom_objects=custom_objects, **kwargs)

# convenience export
keras.robust_load_model = robust_load_model

# --------------------------------------------------------------------------- #
#  Tensor / KerasTensor Type Alias
# --------------------------------------------------------------------------- #
if not hasattr(keras, 'Tensor'):
    keras.Tensor = tf.Tensor

# --------------------------------------------------------------------------- #
#  Essential Keras 3 Ops Shim for Keras 2 
# --------------------------------------------------------------------------- #
if not hasattr(keras, 'ops'):
    class KerasOpsShim:
        """Shim to provide Keras 3 style '.ops' in Keras 2."""
        def __init__(self):
            # Essential mathematical ops
            self.cast = tf.cast
            self.expand_dims = tf.expand_dims
            self.squeeze = tf.squeeze
            self.reshape = tf.reshape
            self.shape = tf.shape
            self.stack = tf.stack
            self.concatenate = tf.concat
            self.tile = tf.tile
            self.split = tf.split
            self.unstack = tf.unstack
            self.meshgrid = tf.meshgrid
            self.linspace = tf.linspace
            self.arange = tf.range
            self.sum = tf.reduce_sum
            self.mean = tf.reduce_mean
            self.max = tf.reduce_max
            self.min = tf.reduce_min
            self.prod = tf.reduce_prod
            self.abs = tf.abs
            self.sqrt = tf.sqrt
            self.square = tf.square
            self.log = tf.math.log
            self.exp = tf.exp
            self.pow = tf.pow
            self.clip = tf.clip_by_value
            self.round = tf.round
            self.matmul = tf.matmul
            self.transpose = tf.transpose
            self.logical_and = tf.logical_and
            self.logical_or = tf.logical_or
            self.logical_not = tf.logical_not
            self.where = tf.where
            self.relu6 = tf.nn.relu6
            self.relu = tf.nn.relu
            self.sigmoid = tf.nn.sigmoid
            self.tanh = tf.nn.tanh
            self.softmax = tf.nn.softmax
            self.image = type('ImageOps', (), {
                'resize': tf.image.resize,
                'affine_transform': lambda x, transform, **kwargs: tf.raw_ops.ImageProjectiveTransformV2(
                images=x, transform=transform, interpolation=kwargs.get('interpolation', 'BILINEAR').upper(),
                output_shape=tf.shape(x)[1:3], fill_mode=kwargs.get('fill_mode', 'CONSTANT').upper()
            ) if hasattr(tf.raw_ops, 'ImageProjectiveTransformV2') else None
        })

    keras.ops = KerasOpsShim()

if not hasattr(keras, 'random'):
    class KerasRandomShim:
        """Shim to provide Keras 3 style '.random' in Keras 2."""
        def __init__(self):
            self.uniform = tf.random.uniform
            self.normal = tf.random.normal
            self.seed = tf.random.set_seed
    
    keras.random = KerasRandomShim()

if not hasattr(keras, 'activations'):
    try:
        import tf_keras.activations as activations
        keras.activations = activations
    except ImportError:
        keras.activations = keras.backend
else:
    if not hasattr(keras.activations, 'swish'):
        if hasattr(tf.nn, 'swish'):
            keras.activations.swish = tf.nn.swish
        else:
            keras.activations.swish = lambda x: x * tf.nn.sigmoid(x)
