# config/__init__.py
"""
Configuration package for Digit Recognizer.

This package provides a modular replacement for the monolithic ``parameters.py``.
For backward compatibility, ``parameters.py`` re-exports everything from this
package so existing imports continue to work.

Usage (new code):
    from config import NB_CLASSES, BATCH_SIZE, ...
    from config.validation import validate_full_config

Usage (legacy, still works):
    import parameters as params
    params.NB_CLASSES  # still works via re-export
"""
import warnings
import sys
import os

# --------------------------------------------------------------------------- #
#  Pre-parse --classes / --color from CLI args before submodule imports
#  so that NB_CLASSES / INPUT_CHANNELS resolve correctly from the start.
#  Uses direct assignment to override any defaults set by scripts
#  (e.g. train.py sets DIGIT_NB_CLASSES=10, DIGIT_INPUT_CHANNELS=1
#   before importing parameters). Externally-set env vars from
#  retrain_all.py subprocesses are also overridden by CLI args.
# --------------------------------------------------------------------------- #
for _i, _arg in enumerate(sys.argv):
    if _arg == '--classes' and _i + 1 < len(sys.argv):
        os.environ["DIGIT_NB_CLASSES"] = sys.argv[_i + 1]
    if _arg == '--color' and _i + 1 < len(sys.argv):
        _val = sys.argv[_i + 1]
        os.environ["DIGIT_INPUT_CHANNELS"] = "3" if _val == "rgb" else "1"

# --------------------------------------------------------------------------- #
#  Force UTF-8 output on Windows to support emojis
# --------------------------------------------------------------------------- #
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --------------------------------------------------------------------------- #
#  Core parameters (defined directly here for import-time safety)
# --------------------------------------------------------------------------- #

# --- NB_CLASSES Logic ---
MANUAL_NB_CLASSES = None  # 10 or 100 — set to override env var
MANUAL_INPUT_CHANNELS = None  # 1 (Gray) or 3 (RGB) — set to override env var

_nb_classes_env = os.environ.get("DIGIT_NB_CLASSES")
if MANUAL_NB_CLASSES is not None:
    NB_CLASSES = MANUAL_NB_CLASSES
elif _nb_classes_env is not None:
    NB_CLASSES = int(_nb_classes_env)
elif "-h" in sys.argv or "--help" in sys.argv:
    NB_CLASSES = 100
else:
    NB_CLASSES = 100
    print("WARNING: DIGIT_NB_CLASSES not set — defaulting to 100. "
          "Set the env var or use --classes to specify.")
del _nb_classes_env

# --- INPUT_CHANNELS Logic ---
_input_channels_env = os.environ.get("DIGIT_INPUT_CHANNELS")
if MANUAL_INPUT_CHANNELS is not None:
    INPUT_CHANNELS = MANUAL_INPUT_CHANNELS
elif _input_channels_env is not None:
    INPUT_CHANNELS = int(_input_channels_env)
elif "-h" in sys.argv or "--help" in sys.argv:
    INPUT_CHANNELS = 1
else:
    INPUT_CHANNELS = 3
    print("WARNING: DIGIT_INPUT_CHANNELS not set — defaulting to RGB (3). "
          "Set the env var or use --color to specify.")
del _input_channels_env

# --- Image Parameters ---
INPUT_WIDTH = 20
INPUT_HEIGHT = 32

# --- Derived parameters ---
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
USE_GRAYSCALE = (INPUT_CHANNELS == 1)
_color_suffix = "GRAY" if USE_GRAYSCALE else "RGB"
OUTPUT_DIR = f"exported_models/{NB_CLASSES}cls_{_color_suffix}"


def update_derived_parameters():
    """Recompute derived parameters after NB_CLASSES / INPUT_CHANNELS changes.
    
    Must be called after CLI or programmatic overrides so that derived values
    like INPUT_SHAPE, USE_GRAYSCALE, and OUTPUT_DIR stay in sync across both 
    the ``config`` and ``parameters`` modules (``parameters`` re-imports via
    ``from config import ...`` which creates a snapshot at import time).
    
    Uses direct module-level name access — ``NB_CLASSES``, ``INPUT_CHANNELS``,
    etc. are resolved dynamically at runtime, so external mutations like
    ``config.NB_CLASSES = 100`` are correctly reflected.
    """
    global INPUT_SHAPE, USE_GRAYSCALE, OUTPUT_DIR
    INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
    USE_GRAYSCALE = (INPUT_CHANNELS == 1)
    _color_suffix = "GRAY" if USE_GRAYSCALE else "RGB"
    OUTPUT_DIR = f"exported_models/{NB_CLASSES}cls_{_color_suffix}"

    # Also sync the ``parameters`` module's bindings, which are snapshots
    # from ``from config import ...`` at import time and would otherwise
    # remain stale after this function runs.
    import sys
    _parameters_mod = sys.modules.get('parameters')
    if _parameters_mod is not None:
        _parameters_mod.INPUT_SHAPE    = INPUT_SHAPE
        _parameters_mod.USE_GRAYSCALE  = USE_GRAYSCALE
        _parameters_mod.OUTPUT_DIR     = OUTPUT_DIR
        _parameters_mod.INPUT_CHANNELS = INPUT_CHANNELS
        _parameters_mod.NB_CLASSES     = NB_CLASSES

# --------------------------------------------------------------------------- #
#  Import from submodules
# --------------------------------------------------------------------------- #

# Import submodules so their names are available
from . import validation
from . import models
from . import data_sources
from . import quantization
from . import training
from . import tuner
from . import augmentation
from . import distillation
from . import losses

# Re-export all public names from submodules
_submodules = [models, data_sources, quantization, training, tuner, augmentation, distillation, losses]
for _mod in _submodules:
    for _name in dir(_mod):
        if not _name.startswith('_'):
            globals()[_name] = getattr(_mod, _name)

# Explicitly import private names needed for backward compatibility
from .quantization import _TQT_DEFAULTS

# Also re-export validation functions for backward compatibility
from .validation import validate_hyperparameters, validate_quantization_parameters, validate_full_config

# Override DATA_SOURCES with the one from data_sources module
from .data_sources import DATA_SOURCES

# --------------------------------------------------------------------------- #
#  Validation
# --------------------------------------------------------------------------- #

# Validate parameters on import
try:
    from .validation import validate_hyperparameters
    validate_hyperparameters()
except Exception as e:
    print(f"❌ Parameter validation failed: {e}")

# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

__all__ = [
    # Core
    'NB_CLASSES', 'INPUT_CHANNELS', 'INPUT_WIDTH', 'INPUT_HEIGHT',
    'INPUT_SHAPE', 'USE_GRAYSCALE', 'OUTPUT_DIR',
    # Submodules
    'validation', 'models', 'data_sources', 'quantization', 'training', 'tuner',
]
