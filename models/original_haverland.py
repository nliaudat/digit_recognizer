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
  Haverland's original notebook (repository no longer available as of 20.06.2026)
"""

import config as params
from .cnn32 import create_original_haverland, CNN_s2

def get_model_info():
    """Return model information matching the original notebook"""
    return {
        "name": "CNN_s2 - Exact Haverland Original",
        "source": "Haverland's original notebook (repository no longer available as of 20.06.2026)",
        "implementation": "Exact replica of b2n.models.cnn32.CNN_s2",
        "input_shape": params.INPUT_SHAPE,
        "nb_classes": params.NB_CLASSES,
        "compiler": "RMSprop(lr=0.001)",
        "loss": "CategoricalCrossentropy(from_logits=True)",
    }