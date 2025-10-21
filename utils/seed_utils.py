import random
import numpy as np
import tensorflow as tf

def set_all_seeds(seed):
    """Set random seeds for reproducibility across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
