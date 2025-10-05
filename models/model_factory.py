# models/model_factory.py
import tensorflow as tf
import importlib
import parameters as params

# Define MODEL_CREATORS here to avoid circular imports
MODEL_CREATORS = {
    "practical_tiny_depthwise": None,  # Will be populated dynamically
    "simple_cnn": None,
    "dig_class100_s2": None, 
    "original_haverland": None,
    "esp_optimized_cnn": None,
    "esp_ultra_light": None,
    "esp_high_capacity": None,
    "esp_quantization_ready": None,
    "esp_haverland_compatible": None,
}

def _import_model_creator(model_name):
    """Dynamically import a model creator function"""
    try:
        if model_name == "practical_tiny_depthwise":
            from .practical_tiny_depthwise import create_practical_tiny_depthwise
            return create_practical_tiny_depthwise
        elif model_name == "simple_cnn":
            from .simple_cnn import create_simple_cnn
            return create_simple_cnn
        elif model_name == "dig_class100_s2":
            from .dig_class100_s2 import create_dig_class100_s2
            return create_dig_class100_s2
        elif model_name == "original_haverland":
            from .original_haverland import create_original_haverland
            return create_original_haverland
        else:
            # For ESP models, try dynamic import
            module_name = f"models.{model_name}"
            module = importlib.import_module(module_name)
            creator_func_name = f"create_{model_name}"
            return getattr(module, creator_func_name)
    except (ImportError, AttributeError) as e:
        print(f"❌ Failed to import {model_name}: {e}")
        return None

def _get_model_creator(model_name):
    """Get or import a model creator"""
    if MODEL_CREATORS.get(model_name) is None:
        # Import the creator
        creator = _import_model_creator(model_name)
        MODEL_CREATORS[model_name] = creator
    return MODEL_CREATORS[model_name]

def create_model():
    """Factory function to automatically create model based on parameters"""
    model_name = params.MODEL_ARCHITECTURE
    
    print(f"🏗️ Creating model: {model_name}")
    
    # Get the model creator function
    creator = _get_model_creator(model_name)
    
    if not creator:
        available_models = get_available_models()
        raise ValueError(
            f"Unknown model architecture: '{model_name}'\n"
            f"Available models: {available_models}\n"
            f"Check parameters.py MODEL_ARCHITECTURE and ensure the model file exists"
        )
    
    # Create the model
    model = creator()
    
    # CRITICAL: Build the model with input shape
    model.build((None,) + params.INPUT_SHAPE)
    print(f"✅ Model '{model_name}' built successfully with input shape: {params.INPUT_SHAPE}")
    
    return model


def model_summary(model):
    """Print model summary"""
    model.build((None,) + params.INPUT_SHAPE)
    model.summary()

def get_available_models():
    """Return list of all available model architectures"""
    available = []
    for model_name in MODEL_CREATORS.keys():
        creator = _get_model_creator(model_name)
        if creator is not None:
            available.append(model_name)
    return available

def get_model_info(model_name=None):
    """Get information about a specific model or all models"""
    if model_name:
        if model_name not in get_available_models():
            return f"Model '{model_name}' not found. Available: {get_available_models()}"
        
        creator = _get_model_creator(model_name)
        model = creator()
        model.build((None,) + params.INPUT_SHAPE)
        
        info = {
            "name": model_name,
            "total_parameters": model.count_params(),
            "input_shape": params.INPUT_SHAPE,
            "output_shape": model.output_shape,
            "layers": len(model.layers),
            "available": True
        }
        
        return info
    else:
        # Return info for all models
        all_info = {}
        for model_name in get_available_models():
            all_info[model_name] = get_model_info(model_name)
        return all_info

def get_model_creators():
    """Get the MODEL_CREATORS dictionary"""
    return {k: v for k, v in MODEL_CREATORS.items() if v is not None}


def compile_model(model):
    """Compile model with settings that actually work"""
    
    # Use the SAME optimizer as Haverland for all models
    # RMSprop was key to Haverland's success
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=params.LEARNING_RATE,
        rho=0.9,  # Same as Haverland
        momentum=0.0,  # Same as Haverland
        epsilon=1e-07  # Same as Haverland
    )
    
    # Use categorical crossentropy for all models during training
    # (ESP-DL will handle quantization to int8 later)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',  # Consistent loss
        metrics=['accuracy']
    )
    
    return model