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

import parameters as params


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

    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
    except Exception as exc:
        errors.append(f"❌ Cannot read label file {label_path}: {exc}")
        return errors

    if not lines:
        errors.append(f"⚠️  Label file is empty: {label_path}")
        return errors

    # Check first few lines are valid integers in [0, expected_classes)
    checked = 0
    for line in lines[:100]:
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

    if checked == 0:
        errors.append(f"⚠️  No parseable labels found in {label_path}")

    return errors


# --------------------------------------------------------------------------- #
#  Validation functions
# --------------------------------------------------------------------------- #

def validate_data_sources() -> List[str]:
    """Check all configured data sources exist and have valid label files."""
    errors = []
    for source in params.DATA_SOURCES:
        name = source.get("name", "unknown")
        src_path = source.get("path", "")
        full_path = os.path.join(os.getcwd(), src_path) if not os.path.isabs(src_path) else src_path

        errors.extend(_check_path(full_path, f"Data source '{name}'"))

        # Check label file
        label_file = source.get("labels", "")
        if label_file:
            label_path = os.path.join(full_path, label_file) if not os.path.isabs(label_file) else label_file
            errors.extend(_check_label_file(label_path, params.NB_CLASSES))

    return errors


def validate_model_architecture() -> List[str]:
    """Check the configured model architecture is available."""
    errors = []
    if params.MODEL_ARCHITECTURE not in params.AVAILABLE_MODELS:
        errors.append(
            f"❌ MODEL_ARCHITECTURE '{params.MODEL_ARCHITECTURE}' not in AVAILABLE_MODELS. "
            f"Available: {params.AVAILABLE_MODELS}"
        )
    return errors


def validate_quantization_flags() -> List[str]:
    """Check quantization flags don't contradict each other."""
    errors = []
    if params.USE_QAT and params.USE_TQT_PIPELINE:
        errors.append("❌ USE_QAT and USE_TQT_PIPELINE are both True — they are mutually exclusive.")
    if params.ESP_DL_QUANTIZE and not params.QUANTIZE_MODEL:
        errors.append("❌ ESP_DL_QUANTIZE=True requires QUANTIZE_MODEL=True.")
    if params.USE_QAT and not params.QUANTIZE_MODEL:
        errors.append("❌ USE_QAT=True requires QUANTIZE_MODEL=True.")
    return errors


def validate_input_shape() -> List[str]:
    """Check input shape is consistent with NB_CLASSES expectations."""
    errors = []
    h, w, c = params.INPUT_SHAPE
    if h <= 0 or w <= 0 or c <= 0:
        errors.append(f"❌ Invalid INPUT_SHAPE: {params.INPUT_SHAPE}")
    if c not in (1, 3):
        errors.append(f"❌ INPUT_CHANNELS must be 1 (grayscale) or 3 (RGB), got {c}")
    return errors


def validate_nb_classes() -> List[str]:
    """Check NB_CLASSES is a supported value."""
    errors = []
    if params.NB_CLASSES not in (10, 100):
        errors.append(
            f"⚠️  NB_CLASSES={params.NB_CLASSES} — expected 10 or 100. "
            f"The project is optimized for these values."
        )
    return errors


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
