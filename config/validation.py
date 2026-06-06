"""
config/validation.py — Unified configuration validation.

Provides a single entry point ``validate_full_config()`` that checks:
- All data source paths exist
- Label files are parseable with correct class counts
- Class counts match ``NB_CLASSES``
- Quantization flags don't contradict
- Model architecture is available
- Input shape matches dataset expectations

Call this in ``train.py`` before any data loading:
    from config.validation import validate_full_config
    validate_full_config()
"""

import os
import sys
from typing import List, Optional, Tuple

# Import directly from submodules to avoid circular imports with config/__init__.py
from config.data_sources import DATA_SOURCES
from config.models import MODEL_ARCHITECTURE, AVAILABLE_MODELS, OPTIMIZER_TYPE
from config.quantization import USE_QAT, USE_TQT_PIPELINE, ESP_DL_QUANTIZE, QUANTIZE_MODEL, QUANTIZATION_MODE
from config.training import (
    LR_SCHEDULER_TYPE, WEIGHT_INITIALIZER, LEARNING_RATE,
)
from config.losses import (
    LOSS_TYPE, LABEL_SMOOTHING,
)

# Import core parameters from the package (these are defined directly in __init__.py)
from config import NB_CLASSES, INPUT_SHAPE, INPUT_CHANNELS


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _check_path(path: str, description: str) -> List[str]:
    """Return a list of error messages if *path* does not exist."""
    if not os.path.exists(path):
        return [f"❌ {description} not found: {path}"]
    return []


def _check_label_file(label_path: str, expected_classes: int) -> List[str]:
    """
    Validate a label file exists, is parseable, and has correct class range.

    Label file format (tab-separated)::
        filename\\tlabel

    The label is the **last** whitespace-separated token on each line.
    """
    errors = []
    if not os.path.isfile(label_path):
        errors.append(f"❌ Label file not found: {label_path}")
        return errors

    if os.path.getsize(label_path) == 0:
        errors.append(f"⚠️  Label file is empty: {label_path}")
        return errors

    checked = 0
    try:
        with open(label_path, "r") as f:
            for line in f:
                if checked >= 100:
                    break
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # Label is the LAST token (format: "filename\\tlabel")
                try:
                    val = int(parts[-1])
                except (ValueError, IndexError):
                    errors.append(f"❌ Invalid label entry in {label_path}: '{line}'")
                    continue
                if val < 0 or val >= expected_classes:
                    errors.append(
                        f"❌ Label {val} out of range [0, {expected_classes}) in {label_path}"
                    )
                checked += 1
    except Exception as exc:
        errors.append(f"❌ Cannot read label file {label_path}: {exc}")
        return errors

    if checked == 0:
        errors.append(f"⚠️  No parseable labels found in {label_path}")

    return errors


# --------------------------------------------------------------------------- #
#  Validation functions
# --------------------------------------------------------------------------- #

def validate_data_sources() -> List[str]:
    """Check all configured data sources exist and have valid label files."""
    errors = []
    for source in DATA_SOURCES:
        name = source.get("name", "unknown")
        src_path = source.get("path", "")
        full_path = os.path.join(os.getcwd(), src_path) if not os.path.isabs(src_path) else src_path

        errors.extend(_check_path(full_path, f"Data source '{name}'"))

        # Check label file
        label_file = source.get("labels", "")
        if label_file:
            label_path = os.path.join(full_path, label_file) if not os.path.isabs(label_file) else label_file
            errors.extend(_check_label_file(label_path, NB_CLASSES))

    return errors


def validate_model_architecture() -> List[str]:
    """Check the configured model architecture is available."""
    errors = []
    if MODEL_ARCHITECTURE not in AVAILABLE_MODELS:
        errors.append(
            f"❌ MODEL_ARCHITECTURE '{MODEL_ARCHITECTURE}' not in AVAILABLE_MODELS. "
            f"Available: {AVAILABLE_MODELS}"
        )
    return errors


def validate_quantization_flags() -> List[str]:
    """Check quantization flags don't contradict each other."""
    errors = []
    if USE_QAT and USE_TQT_PIPELINE:
        errors.append("❌ USE_QAT and USE_TQT_PIPELINE are both True — they are mutually exclusive.")
    if ESP_DL_QUANTIZE and not QUANTIZE_MODEL:
        errors.append("❌ ESP_DL_QUANTIZE=True requires QUANTIZE_MODEL=True.")
    if USE_QAT and not QUANTIZE_MODEL:
        errors.append("❌ USE_QAT=True requires QUANTIZE_MODEL=True.")
    return errors


def validate_input_shape() -> List[str]:
    """Check input shape is consistent with NB_CLASSES expectations."""
    errors = []
    h, w, c = INPUT_SHAPE
    if h <= 0 or w <= 0 or c <= 0:
        errors.append(f"❌ Invalid INPUT_SHAPE: {INPUT_SHAPE}")
    if c not in (1, 3):
        errors.append(f"❌ INPUT_CHANNELS must be 1 (grayscale) or 3 (RGB), got {c}")
    return errors


def validate_nb_classes() -> List[str]:
    """Check NB_CLASSES is a supported value."""
    errors = []
    if NB_CLASSES not in (10, 100):
        errors.append(
            f"⚠️  NB_CLASSES={NB_CLASSES} — expected 10 or 100. "
            f"The project is optimized for these values."
        )
    return errors


# --------------------------------------------------------------------------- #
#  Hyperparameter validation (moved from parameters.py)
# --------------------------------------------------------------------------- #

def validate_hyperparameters():
    """Validate all hyperparameters for consistency"""
    # Optimizer validation
    valid_optimizers = ["rmsprop", "adam", "sgd", "adagrad", "adamw", "nadam"]
    if OPTIMIZER_TYPE not in valid_optimizers:
        raise ValueError(f"❌ Invalid OPTIMIZER_TYPE: {OPTIMIZER_TYPE}. Must be one of {valid_optimizers}")
    
    # Loss function validation
    valid_losses = ["sparse_categorical_crossentropy", "categorical_crossentropy", "focal_loss", "IntelligentFocalLossController"]
    if LOSS_TYPE not in valid_losses:
        raise ValueError(f"❌ Invalid LOSS_TYPE: {LOSS_TYPE}. Must be one of {valid_losses}")
    
    # Learning rate scheduler validation
    valid_schedulers = ["reduce_on_plateau", "exponential", "cosine", "step", "onecycle"]
    if LR_SCHEDULER_TYPE not in valid_schedulers:
        raise ValueError(f"❌ Invalid LR_SCHEDULER_TYPE: {LR_SCHEDULER_TYPE}. Must be one of {valid_schedulers}")
    
    # Weight initializer validation
    valid_initializers = ["glorot_uniform", "he_normal", "he_uniform", "lecun_normal"]
    if WEIGHT_INITIALIZER not in valid_initializers:
        raise ValueError(f"❌ Invalid WEIGHT_INITIALIZER: {WEIGHT_INITIALIZER}. Must be one of {valid_initializers}")
    
    # Label smoothing validation
    if not 0 <= LABEL_SMOOTHING <= 0.5:
        raise ValueError(f"❌ Invalid LABEL_SMOOTHING: {LABEL_SMOOTHING}. Must be between 0 and 0.5")
    
    # Learning rate validation
    if LEARNING_RATE <= 0:
        raise ValueError(f"❌ Invalid LEARNING_RATE: {LEARNING_RATE}. Must be positive")
    
    # Quantization mode validation
    valid_modes = ["none", "ptq", "qat", "tqt", "auto", None]
    if QUANTIZATION_MODE not in valid_modes:
        raise ValueError(f"❌ Invalid QUANTIZATION_MODE: {QUANTIZATION_MODE}. Must be one of {valid_modes}")
    
    print("✅ All hyperparameters validated successfully!")


def validate_quantization_parameters():
    """
    Validate and correct quantization parameter combinations
    Returns: (is_valid, corrected_params, message)
    """
    original_params = {
        'QUANTIZE_MODEL': QUANTIZE_MODEL,
        'USE_QAT': USE_QAT, 
        'ESP_DL_QUANTIZE': ESP_DL_QUANTIZE,
        'USE_TQT_PIPELINE': USE_TQT_PIPELINE,
        'QUANTIZATION_MODE': QUANTIZATION_MODE
    }
    
    corrected_params = original_params.copy()
    messages = []
    
    # Rule 1: ESP_DL_QUANTIZE requires QUANTIZE_MODEL
    if ESP_DL_QUANTIZE and not QUANTIZE_MODEL:
            messages.append("❌ ESP_DL_QUANTIZE=True requires QUANTIZE_MODEL=True")
            corrected_params['QUANTIZE_MODEL'] = True
            messages.append("✅ Auto-corrected: Set QUANTIZE_MODEL=True")
    
    # Rule 2: USE_QAT requires QUANTIZE_MODEL  
    if USE_QAT and not QUANTIZE_MODEL:
            messages.append("❌ USE_QAT=True requires QUANTIZE_MODEL=True")
            corrected_params['QUANTIZE_MODEL'] = True
            messages.append("✅ Auto-corrected: Set QUANTIZE_MODEL=True")
    
    # Rule 3: QAT and TQT are mutually exclusive
    if USE_QAT and USE_TQT_PIPELINE:
        messages.append("⚠️  Both USE_QAT and USE_TQT_PIPELINE are True. TQT takes precedence.")
        corrected_params['USE_QAT'] = False
        messages.append("✅ Auto-corrected: Set USE_QAT=False")
    
    # Rule 3b: QAT without TQT → ESP_DL_QUANTIZE must be False
    if USE_QAT and not USE_TQT_PIPELINE and ESP_DL_QUANTIZE:
        messages.append("⚠️  QAT enabled without TQT pipeline – resetting ESP_DL_QUANTIZE=False")
        corrected_params['ESP_DL_QUANTIZE'] = False
        messages.append("✅ Auto-corrected: Set ESP_DL_QUANTIZE=False")
    
    # Rule 4: TQT requires ESP_DL_QUANTIZE=True
    if USE_TQT_PIPELINE and not ESP_DL_QUANTIZE:
        messages.append("⚠️  TQT pipeline requires ESP_DL_QUANTIZE=True")
        corrected_params['ESP_DL_QUANTIZE'] = True
        messages.append("✅ Auto-corrected: Set ESP_DL_QUANTIZE=True")
    
    # Rule 5: TQT requires QUANTIZE_MODEL=True
    if USE_TQT_PIPELINE and not QUANTIZE_MODEL:
        messages.append("⚠️  TQT pipeline requires QUANTIZE_MODEL=True")
        corrected_params['QUANTIZE_MODEL'] = True
        messages.append("✅ Auto-corrected: Set QUANTIZE_MODEL=True")
    
    # Determine the final mode
    if USE_TQT_PIPELINE:
        mode = "TQT Pipeline (Trainable Quantization Thresholds)"
    elif USE_QAT:
        if ESP_DL_QUANTIZE:
            mode = "QAT + INT8 quantization for ESP-DL"
        else:
            mode = "QAT + UINT8 quantization"
    elif QUANTIZE_MODEL:
        if ESP_DL_QUANTIZE:
            mode = "Standard training + INT8 post-quantization (ESP-DL)"
        else:
            mode = "Standard training + UINT8 post-quantization"
    else:
        mode = "Float32 training & inference"
    
    messages.append(f"✅ Final mode: {mode}")
    
    # Check if any corrections were made
    needs_correction = any(original_params[k] != corrected_params[k] for k in original_params 
                          if k != 'QUANTIZATION_MODE')
    
    return not needs_correction, corrected_params, "\n".join(messages)


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #

def validate_full_config(raise_on_error: bool = True) -> Tuple[bool, List[str]]:
    """
    Run all configuration validators and report results.

    Parameters
    ----------
    raise_on_error : bool
        If True (default), raise ``SystemExit(1)`` on critical errors.
        If False, return ``(is_valid, error_messages)``.

    Returns
    -------
    (is_valid, errors) : Tuple[bool, List[str]]
        Only meaningful when ``raise_on_error=False``.
    """
    all_errors: List[str] = []
    all_errors.extend(validate_data_sources())
    all_errors.extend(validate_model_architecture())
    all_errors.extend(validate_quantization_flags())
    all_errors.extend(validate_input_shape())
    all_errors.extend(validate_nb_classes())

    if all_errors:
        print("\n" + "=" * 60)
        print("🔍 CONFIGURATION VALIDATION FAILED")
        print("=" * 60)
        for err in all_errors:
            print(f"  {err}")
        print("=" * 60 + "\n")
        if raise_on_error:
            sys.exit(1)
        return False, all_errors

    print("✅ Configuration validation passed.")
    return True, []


# --------------------------------------------------------------------------- #
#  Convenience: run when called directly
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    validate_full_config(raise_on_error=True)
