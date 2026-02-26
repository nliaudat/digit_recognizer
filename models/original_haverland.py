# models/original_haverland.py
"""
original_haverland – Thin Wrapper for the Haverland CNN_s2 Model
=================================================================
Design goal: Expose the Haverland CNN_s2 architecture under the standard
model factory naming convention (create_original_haverland) by re-exporting
from cnn32.py. Also provides model metadata (source, compiler config, etc.).

Architecture:
  See cnn32.py — delegates entirely to CNN_s2() / create_original_haverland().
  Trained with: RMSprop(lr=0.001), CategoricalCrossentropy(from_logits=True).

Reference:
  https://github.com/haverland/Tenth-of-step-of-a-meter-digit/blob/master/dig-class100-s2.ipynb
"""

import parameters as params
from .cnn32 import create_original_haverland, CNN_s2

def get_model_info():
    """Return model information matching the original notebook"""
    return {
        "name": "CNN_s2 - Exact Haverland Original",
        "source": "https://github.com/haverland/Tenth-of-step-of-a-meter-digit/blob/master/dig-class100-s2.ipynb",
        "implementation": "Exact replica of b2n.models.cnn32.CNN_s2",
        "input_shape": params.INPUT_SHAPE,
        "nb_classes": params.NB_CLASSES,
        "compiler": "RMSprop(lr=0.001)",
        "loss": "CategoricalCrossentropy(from_logits=True)",
    }