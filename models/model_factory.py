# models/model_factory.py
import tensorflow as tf
import parameters as params

# Import all model creation functions
from .practical_tiny_depthwise import create_practical_tiny_depthwise
from .simple_cnn import create_simple_cnn
from .dig_class100_s2 import create_dig_class100_s2
from .original_haverland import create_original_haverland
from .cnn32 import CNN_s2

def create_model():
    """Factory function to create model based on parameters"""
    print(f"🏗️ Creating model: {params.MODEL_ARCHITECTURE}")
    
    if params.MODEL_ARCHITECTURE == "practical_tiny_depthwise":
        model = create_practical_tiny_depthwise()
    elif params.MODEL_ARCHITECTURE == "simple_cnn":
        model = create_simple_cnn()
    elif params.MODEL_ARCHITECTURE == "dig_class100_s2":
        model = create_dig_class100_s2()
    elif params.MODEL_ARCHITECTURE == "original_haverland":
        model = create_original_haverland()
    else:
        raise ValueError(f"Unknown model architecture: {params.MODEL_ARCHITECTURE}")
    
    # CRITICAL: Build the model with input shape
    model.build((None,) + params.INPUT_SHAPE)
    print(f"✅ Model built successfully with input shape: {params.INPUT_SHAPE}")
    
    return model

def compile_model(model):
    """Compile model with standard settings"""
    # Use RMSprop like the original notebook for Haverland model
    if params.MODEL_ARCHITECTURE == "original_haverland":
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            # FIX: Change from_logits=True to from_logits=False
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
    else:
        # For practical_tiny_depthwise and others
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Explicit learning rate
            loss='sparse_categorical_crossentropy',  # Use string for automatic from_logits
            metrics=['accuracy']
        )
    return model

def model_summary(model):
    """Print model summary"""
    model.build((None,) + params.INPUT_SHAPE)
    model.summary()