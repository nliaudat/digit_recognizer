# utils/train_qat_helper.py
"""
Utility functions that deal specifically with Quantization‑Aware Training (QAT).

These helpers are grouped together so that QAT‑related logic lives in a single
module, making the codebase easier to navigate and maintain.

Functions provided:
* `create_qat_model` – builds a model wrapped for QAT (uses tfmot if available).
* `create_qat_representative_dataset` – calibration data generator that matches
  the preprocessing used during QAT training.
* `validate_qat_data_flow` – quick sanity‑check that a QAT model can process a
  sample batch without errors.
* `is_qat_model` – thin wrapper that forwards to the implementation in
  `utils.preprocess` (kept for backward compatibility).

All functions rely only on the public API of the project (`parameters`,
`utils.preprocess`, `utils` helpers, etc.) and raise clear exceptions when
something goes wrong.
"""

from typing import Tuple, Callable, Optional

import numpy as np
import tensorflow as tf

# --------------------------------------------------------------------------- #
#  Imports from the rest of the project
# --------------------------------------------------------------------------- #
import parameters as params
from utils import preprocess_images
# from utils.preprocess import is_qat_model as _is_qat_model  # core detection logic


# --------------------------------------------------------------------------- #
#  QAT model creation
# --------------------------------------------------------------------------- #
def create_qat_model() -> tf.keras.Model:
    """
    Wrap a freshly‑created model with Quantization‑Aware Training (QAT).

    Returns
    -------
    tf.keras.Model
        Either a QAT‑enabled model (if `tensorflow_model_optimization` is
        available) or a plain model (fallback).
    """
    try:
        import tensorflow_model_optimization as tfmot
    except Exception as exc:  # pragma: no cover
        # QAT library not installed – fall back to a regular model
        print("⚠️  QAT library not available – building a standard model")
        from models import create_model
        return create_model()

    # Build the base architecture first
    from models import create_model
    base_model = create_model()

    # Apply QAT transformation – the high‑level helper automatically annotates
    # and wraps the model for quantization‑aware training.
    qat_model = tfmot.quantization.keras.quantize_model(base_model)

    print("✅ QAT model created successfully")
    return qat_model


# --------------------------------------------------------------------------- #
#  Representative dataset for QAT (must match training preprocessing)
# --------------------------------------------------------------------------- #
def create_qat_representative_dataset(
    x_train_raw: np.ndarray,
    num_samples: int = params.QUANTIZE_NUM_SAMPLES,
) -> Callable[[], Tuple[np.ndarray, ...]]:
    """
    Build a calibration generator that reproduces the exact preprocessing used
    during QAT training.

    Parameters
    ----------
    x_train_raw : np.ndarray
        The *raw* training images (before any preprocessing).
    num_samples : int, optional
        Number of samples to include in the calibration set (default taken from
        ``params.QUANTIZE_NUM_SAMPLES``).

    Returns
    -------
    Callable[[], Tuple[np.ndarray, ...]]
        A generator function yielding one‑sample batches suitable for the
        ``representative_dataset`` argument of ``tf.lite.TFLiteConverter``.
    """
    def representative_dataset():
        # -----------------------------------------------------------------
        #  1️⃣  Select a slice of the raw data
        # -----------------------------------------------------------------
        raw_slice = x_train_raw[:num_samples]

        # -----------------------------------------------------------------
        #  2️⃣  Apply the SAME preprocessing that was used during QAT training
        #      (the original code used ``for_training=False`` – keep that)
        # -----------------------------------------------------------------
        processed = preprocess_images(raw_slice, for_training=False)

        # -----------------------------------------------------------------
        #  3️⃣  Ensure the data is ``float32`` in the range [0, 1]
        # -----------------------------------------------------------------
        if processed.dtype != np.float32:
            processed = processed.astype(np.float32)
        if processed.max() > 1.0:
            processed = processed / 255.0

        # -----------------------------------------------------------------
        #  4️⃣  Yield one‑sample batches (required by the TFLite API)
        # -----------------------------------------------------------------
        for i in range(len(processed)):
            # Each yielded element must be a list/tuple containing a single batch
            yield [processed[i : i + 1]]

    return representative_dataset


# --------------------------------------------------------------------------- #
#  QAT data‑flow validation (optional sanity check)
# --------------------------------------------------------------------------- #
def validate_qat_data_flow(
    model: tf.keras.Model,
    x_train_sample: np.ndarray,
    debug: bool = False,
) -> Tuple[bool, str]:
    """
    Perform a quick forward‑pass on a tiny batch to ensure a QAT‑wrapped model
    accepts the data without raising errors.

    Parameters
    ----------
    model : tf.keras.Model
        The model to test (should be QAT‑enabled).
    x_train_sample : np.ndarray
        A small slice of the training data (raw, before preprocessing).
    debug : bool, optional
        If ``True`` additional diagnostic information is printed.

    Returns
    -------
    Tuple[bool, str]
        ``(True, "...")`` if the forward pass succeeded, otherwise
        ``(False, error_message)``.
    """
    if not params.USE_QAT or not params.QUANTIZE_MODEL:
        return True, "QAT not enabled"

    if debug:
        print("\n🔍 VALIDATING QAT DATA FLOW")
        print("=" * 50)

    # Use a single sample (batch dimension required)
    sample = x_train_sample[:1]

    if debug:
        print(
            f"   Sample – dtype:{sample.dtype} "
            f"range:[{sample.min():.3f}, {sample.max():.3f}]"
        )

    try:
        # Forward pass – the model should already contain the QAT wrappers
        output = model(sample)

        # Report the numeric range of the output (helps catch NaNs)
        out_min, out_max = output.numpy().min(), output.numpy().max()
        if debug:
            print(
                f"✅ Forward pass succeeded – output range "
                f"[{out_min:.3f}, {out_max:.3f}]"
            )
        return True, "QAT data flow OK"
    except Exception as exc:  # pragma: no cover
        err_msg = f"QAT forward pass failed: {exc}"
        if debug:
            print(f"❌ {err_msg}")
        return False, err_msg


# --------------------------------------------------------------------------- #
#  QAT model detection
# --------------------------------------------------------------------------- #
def _is_qat_model(model: tf.keras.Model) -> bool:
    """
    Heuristic detection of a QAT‑wrapped model.
    
    Parameters
    ----------
    model : tf.keras.Model
        The model to check for QAT wrappers.
        
    Returns
    -------
    bool
        True if the model appears to be QAT-wrapped, False otherwise.
    """
    # Check for quantization layers
    for layer in model.layers:
        layer_name = layer.name.lower()
        layer_class = layer.__class__.__name__.lower()
        
        # Check for quantization indicators
        if (hasattr(layer, 'quantize_config') or 
            'quant' in layer_name or 
            'qat' in layer_name or
            'quantize' in layer_class):
            return True
    
    # Check model name and attributes
    model_name = model.name.lower() if hasattr(model, 'name') else ''
    if 'qat' in model_name or 'quant' in model_name:
        return True
    
    # Check if model was created within quantize_scope
    if hasattr(model, '_quantize_scope'):
        return True
        
    return False


# --------------------------------------------------------------------------- #
#  Public API list (helps static analysers & IDEs)
# --------------------------------------------------------------------------- #
__all__ = [
    "create_qat_model",
    "create_qat_representative_dataset",
    "validate_qat_data_flow",
    "_is_qat_model",
]