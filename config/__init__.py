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

# --------------------------------------------------------------------------- #
#  Backward-compatible utility functions (moved from parameters.py)
# --------------------------------------------------------------------------- #
# These are kept here so that ``from config import get_hyperparameter_summary``
# works and so that ``parameters.py`` (which re-exports everything from config)
# continues to provide them for legacy code.

def get_hyperparameter_summary():
    """Return a comprehensive summary of all hyperparameter settings."""
    _live_arch = models.get_model_filename()
    summary = {
        'model': {
            'architecture': _live_arch,
            'input_shape': INPUT_SHAPE,
            'num_classes': NB_CLASSES,
        },
        'optimizer': {
            'type': OPTIMIZER_TYPE,
            'learning_rate': LEARNING_RATE,
        },
        'loss': {
            'type': LOSS_TYPE,
            'label_smoothing': LABEL_SMOOTHING,
            'focal_gamma': FOCAL_GAMMA,
            'focal_alpha': FOCAL_ALPHA,
        },
        'training': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'training_percentage': TRAINING_PERCENTAGE,
            'validation_split': VALIDATION_SPLIT,
            'use_lr_scheduler': USE_LEARNING_RATE_SCHEDULER,
            'lr_scheduler_type': LR_SCHEDULER_TYPE,
            'use_dynamic_scheduler': USE_DYNAMIC_SCHEDULER,
            'use_dynamic_optimizer': USE_DYNAMIC_OPTIMIZER,
            'use_dynamic_weights': USE_DYNAMIC_WEIGHTS,
            'use_lr_warmup': USE_LR_WARMUP,
            'weight_initializer': WEIGHT_INITIALIZER,
            'use_gradient_clipping': USE_GRADIENT_CLIPPING,
            'use_early_stopping': USE_EARLY_STOPPING,
            'use_tensorboard': USE_TENSORBOARD,
            'save_checkpoints': SAVE_CHECKPOINTS,
            'restore_best_weights': RESTORE_BEST_WEIGHTS,
        },
        'regularization': {
            'use_batch_norm': USE_BATCH_NORM,
            'l1_regularization': L1_REGULARIZATION,
            'l2_regularization': L2_REGULARIZATION,
            'dropout_rate': DEFAULT_DROPOUT_RATE,
        },
        'augmentation': {
            'use_data_augmentation': USE_DATA_AUGMENTATION,
            'use_mixup': USE_MIXUP,
            'use_cutmix': USE_CUTMIX,
            'use_random_erasing': USE_RANDOM_ERASING,
        },
        'quantization': {
            'quantization_mode': QUANTIZATION_MODE,
            'quantize_model': QUANTIZE_MODEL,
            'use_qat': USE_QAT,
            'use_tqt_pipeline': USE_TQT_PIPELINE,
            'esp_dl_quantize': ESP_DL_QUANTIZE,
        },
        'advanced': {
            'use_mixed_precision': USE_MIXED_PRECISION,
            'use_gradient_accumulation': USE_GRADIENT_ACCUMULATION,
            'use_stochastic_weight_averaging': USE_STOCHASTIC_WEIGHT_AVERAGING,
            'use_cyclical_lr': USE_CYCLICAL_LEARNING_RATE,
            'use_lr_finder': USE_LEARNING_RATE_FINDER,
            'use_model_ensemble': USE_MODEL_ENSEMBLE,
        },
        'hardware': {
            'use_gpu': USE_GPU,
            'output_dir': OUTPUT_DIR,
            'grayscale': USE_GRAYSCALE,
            'input_channels': INPUT_CHANNELS,
            'input_width': INPUT_WIDTH,
            'input_height': INPUT_HEIGHT,
        },
    }
    return summary


def print_hyperparameter_summary():
    """Print a formatted summary of all hyperparameter settings."""
    summary = get_hyperparameter_summary()
    print("\n" + "=" * 60)
    print("HYPERPARAMETER CONFIGURATION SUMMARY")
    print("=" * 60)
    for category, settings in summary.items():
        print(f"\n{category.upper()}:")
        for key, value in settings.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    print("=" * 60 + "\n")


def get_hyperparameter_summary_text():
    """Return hyperparameter summary as formatted text for file export."""
    summary = get_hyperparameter_summary()
    lines = []
    lines.append("HYPERPARAMETER CONFIGURATION SUMMARY")
    lines.append("=" * 50)
    for category, settings in summary.items():
        lines.append(f"\n{category.upper()}:")
        for key, value in settings.items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for k, v in value.items():
                    lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  {key}: {value}")
    return "\n".join(lines)


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
