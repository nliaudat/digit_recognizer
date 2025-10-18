# models/original_haverland.py
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
        "loss": "CategoricalCrossentropy(from_logits=True)"
    }