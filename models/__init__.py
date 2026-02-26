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
# from .practical_tiny_depthwise import create_practical_tiny_depthwise
# from .simple_cnn import create_simple_cnn
# from .dig_class100_s2 import create_dig_class100_s2
from .original_haverland import create_original_haverland
from .cnn32 import CNN_s2
from .digit_recognizer_v1 import create_digit_recognizer_v1
from .digit_recognizer_v2 import create_digit_recognizer_v2
from .digit_recognizer_v3 import create_digit_recognizer_v3
from .digit_recognizer_v4 import create_digit_recognizer_v4
from .digit_recognizer_v5 import create_digit_recognizer_v5
from .digit_recognizer_v6 import create_digit_recognizer_v6
from .digit_recognizer_v7 import create_digit_recognizer_v7
from .digit_recognizer_v8 import create_digit_recognizer_v8
from .digit_recognizer_v9 import create_digit_recognizer_v9
from .digit_recognizer_v10 import create_digit_recognizer_v10
from .digit_recognizer_v11 import create_digit_recognizer_v11
from .digit_recognizer_v12 import create_digit_recognizer_v12
from .digit_recognizer_v15 import create_digit_recognizer_v15
from .digit_recognizer_v16 import create_digit_recognizer_v16
from .digit_recognizer_v17 import create_digit_recognizer_v17
from .esp_quantization_ready import create_esp_quantization_ready
from .mnist_quantization import create_mnist_quantization
from .high_accuracy_validator import create_high_accuracy_validator

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
    'create_digit_recognizer_v1',
    'create_digit_recognizer_v2',
    'create_digit_recognizer_v3',
    'create_digit_recognizer_v4',
    'create_digit_recognizer_v5',
    'create_digit_recognizer_v6',
    'create_digit_recognizer_v7',
    'create_digit_recognizer_v8',
    'create_digit_recognizer_v9',
    'create_digit_recognizer_v10',
    'create_digit_recognizer_v11',
    'create_digit_recognizer_v12',
    'create_digit_recognizer_v15',
    'create_digit_recognizer_v16',
    'create_digit_recognizer_v17',
    'create_esp_quantization_ready',
    'create_mnist_quantization',
    'create_high_accuracy_validator',
]