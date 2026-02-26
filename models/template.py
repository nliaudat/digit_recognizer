# models/template.py
"""
template – Starter Template for New Model Architectures
========================================================
Design goal: Minimal boilerplate to copy-paste when creating a new model.
Replace the placeholder layers with your actual architecture and update
this docstring with the design goal, layer list, notes, and estimated size.

Architecture (placeholder):
  - Conv2D(32, 3×3) + ReLU + MaxPool  → replace with your layers
  - GlobalAveragePooling2D → Dense(NB_CLASSES) Softmax

Notes:
  - Copy this file and rename to digit_recognizer_vXX.py
  - Register the new model in models/__init__.py and parameters.py
  - Update the header docstring before committing
"""

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