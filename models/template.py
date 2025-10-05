# models/template.py
import tensorflow as tf
import parameters as params

def create_template_model():
    """
    Template for creating new models.
    Copy this file and modify the architecture.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=params.INPUT_SHAPE),
        
        # Add your layers here
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Classification head
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(params.NB_CLASSES, activation='softmax')
    ])
    
    return model

# Optional: Add model-specific parameters
def get_model_parameters():
    """Return model-specific parameters for documentation"""
    return {
        "input_shape": params.INPUT_SHAPE,
        "num_classes": params.NB_CLASSES,
        "description": "Template model structure"
    }