# utils/__init__.py
"""
Make `utils` a proper package and expose the helpers that other modules
(expect to import directly from `utils`).  

The original project only exported `get_data_splits`, `load_combined_dataset`,
`preprocess_images`, and `predict_single_image`.  Several other modules
(`train_modelmanager.py`, `train_progressbar.py`, etc.) also import
`get_calibration_data` and `suppress_all_output` directly from `utils`.
To avoid circular imports we define those helpers here as thin wrappers that
delegate to the implementations that already exist in the codebase.
"""

# --------------------------------------------------------------------------- #
#  Core data‑loading utilities (already part of the original package)
# --------------------------------------------------------------------------- #
from .multi_source_loader import get_data_splits, load_combined_dataset
from .preprocess import preprocess_images, predict_single_image

# --------------------------------------------------------------------------- #
#  Calibration helper (used by PTQ conversion)
# --------------------------------------------------------------------------- #
import numpy as np
import tensorflow as tf
from .preprocess import preprocess_images  # reuse the existing preprocessing

def get_calibration_data(x_raw: np.ndarray) -> np.ndarray:
    """
    Return a float32 [0, 1] calibration set suitable for
    ``tf.lite.TFLiteConverter``.

    The function uses the *inference* preprocessing path
    (``for_training=False``) and guarantees that the returned array is
    ``float32`` in the range ``[0, 1]``.
    """
    cal = preprocess_images(x_raw, for_training=False)
    if cal.dtype != np.float32:
        cal = cal.astype(np.float32) / 255.0
    elif cal.max() > 1.0:
        cal = cal / 255.0
    return cal


# --------------------------------------------------------------------------- #
#  Silent‑output context manager (used while converting to TFLite)
# --------------------------------------------------------------------------- #
import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_all_output(debug: bool = False):
    """
    Redirect ``stdout`` / ``stderr`` to ``os.devnull`` unless ``debug`` is True.

    This is primarily used during the TFLite conversion step to hide the
    massive amount of TensorFlow logging that would otherwise clutter the
    console.
    """
    if debug:
        yield
        return

    original_stdout, original_stderr = sys.stdout, sys.stderr
    with open(os.devnull, "w") as fnull:
        sys.stdout, sys.stderr = fnull, fnull
        try:
            if hasattr(sys, "__stdout__"):
                stdout_fd = os.dup(sys.__stdout__.fileno())
                stderr_fd = os.dup(sys.__stderr__.fileno())
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull_fd, sys.__stdout__.fileno())
                os.dup2(devnull_fd, sys.__stderr__.fileno())
                os.close(devnull_fd)
        except Exception:
            pass

        try:
            yield
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            try:
                if "stdout_fd" in locals() and "stderr_fd" in locals():
                    os.dup2(stdout_fd, sys.__stdout__.fileno())
                    os.dup2(stderr_fd, sys.__stderr__.fileno())
                    os.close(stdout_fd)
                    os.close(stderr_fd)
            except Exception:
                pass