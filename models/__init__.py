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

# Import core models that are always available
from .practical_tiny_depthwise import create_practical_tiny_depthwise
from .simple_cnn import create_simple_cnn
from .dig_class100_s2 import create_dig_class100_s2
from .original_haverland import create_original_haverland
from .cnn32 import CNN_s2

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
    'create_practical_tiny_depthwise',
    'create_simple_cnn',
    'create_dig_class100_s2',
    'create_original_haverland',
    'CNN_s2',
]