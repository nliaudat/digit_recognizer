import os
import tensorflow as tf

def setup_tensorflow_logging(verbose=0):
    """Configure TensorFlow logging level based on verbosity level."""
    if verbose >= 2:
        tf.get_logger().setLevel('INFO')
        tf.autograph.set_verbosity(2)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:  # verbose < 2
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
