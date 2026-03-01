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
    'get_training_callbacks'
]
