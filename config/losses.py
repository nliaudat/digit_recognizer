"""
config/losses.py — Loss & dynamic alpha parameters for 10 vs 100 classes.

All loss-related configuration that used to live in ``config/training.py`` has been
moved here for modularity.  Parameters that differ between 10-class and 100-class
training use ``if NB_CLASSES <= 10:`` / ``else:`` branching.

Import this module after ``NB_CLASSES`` has been resolved (e.g. via
``from config import NB_CLASSES``).
"""

import sys

# Get NB_CLASSES from the parent config module (avoids circular import)
_config_mod = sys.modules.get('config')
if _config_mod is not None and hasattr(_config_mod, 'NB_CLASSES'):
    NB_CLASSES = _config_mod.NB_CLASSES
else:
    # Fallback: read from environment
    import os
    NB_CLASSES = int(os.environ.get("DIGIT_NB_CLASSES", "100"))
del _config_mod

# ==============================================================================
# LOSS TYPE & LABEL SMOOTHING
# ==============================================================================

# Options:
#   - "IntelligentFocalLossController":  (DEFAULT) Adaptive Focal Loss
#   - "focal_loss":                       Standard Focal Loss, fixed γ/α
#   - "sparse_categorical_crossentropy":  Standard CrossEntropy for integer labels
#   - "categorical_crossentropy":         One-hot CrossEntropy (original_haverland)

LOSS_TYPE = "IntelligentFocalLossController"

if NB_CLASSES <= 10:
    LABEL_SMOOTHING = 0.01
else:
    LABEL_SMOOTHING = 0.05

# ==============================================================================
# FOCAL LOSS — FIXED PARAMETERS (base values, overridden by Controller)
# ==============================================================================

FOCAL_GAMMA = 2.0

if NB_CLASSES <= 10:
    FOCAL_ALPHA = 0.45
elif NB_CLASSES <= 20:
    FOCAL_ALPHA = 0.38
elif NB_CLASSES <= 50:
    FOCAL_ALPHA = 0.32
else:  # 100 classes
    FOCAL_ALPHA = 0.27

# ==============================================================================
# INTELLIGENT FOCAL LOSS CONTROLLER — THRESHOLD-BASED γ SWITCHING
# ==============================================================================

if NB_CLASSES <= 10:
    FOCAL_ACCURACY_THRESHOLDS = [0.992, 0.995, 0.997]
else:
    FOCAL_ACCURACY_THRESHOLDS = [0.88, 0.93, 0.97]

if NB_CLASSES <= 10:
    FOCAL_GAMMA_VALUES = [0.5, 1.0, 2.0]
else:
    FOCAL_GAMMA_VALUES = [1.2, 2.0, 3.5]

# Smooth γ transition: ramp linearly over N epochs instead of an instant step.
FOCAL_GAMMA_RAMP_EPOCHS = 3

# Plateau detection
FOCAL_PLATEAU_PATIENCE = 8
FOCAL_PLATEAU_MIN_DELTA = 0.0005

# ==============================================================================
# DYNAMIC PER-CLASS WEIGHTING (IntelligentFocalLossController)
# ==============================================================================

USE_DYNAMIC_WEIGHTS = True
DYNAMIC_WEIGHTS_EPOCHS = 5

# ---------------------------------------------------------------------- #
#  Per-class alpha adjustment parameters
# ---------------------------------------------------------------------- #
# These control how the controller boosts the loss for bad classes during
# plateau (or periodic) recalibration steps.
#
#   DYNAMIC_ALPHA_EPSILON   — additive constant in 1/(acc+ε) to limit max ratio
#   DYNAMIC_ALPHA_CAP_MIN   — lower bound for the normalised weight (no class gets
#                             ignored completely)
#   DYNAMIC_ALPHA_CAP_MAX   — upper bound (no class dominates the gradient)
#   DYNAMIC_ALPHA_TRIGGER   - "plateau"  → recalc only when stuck (10cls)
#                             "periodic" → recalc every DYNAMIC_WEIGHTS_EPOCHS (100cls)
if NB_CLASSES <= 10:
    DYNAMIC_ALPHA_EPSILON   = 0.01
    DYNAMIC_ALPHA_CAP_MIN   = 0.5
    DYNAMIC_ALPHA_CAP_MAX   = 2.0
    DYNAMIC_ALPHA_TRIGGER   = "plateau"
else:
    DYNAMIC_ALPHA_EPSILON   = 0.05
    DYNAMIC_ALPHA_CAP_MIN   = 0.1
    DYNAMIC_ALPHA_CAP_MAX   = 8.0
    DYNAMIC_ALPHA_TRIGGER   = "periodic"

# ---------------------------------------------------------------------- #
#  Base alpha formula parameters
# ---------------------------------------------------------------------- #
# Used in IntelligentFocalLossController:
#   base_alpha = min(DYNAMIC_ALPHA_BASE_MAX,
#                    max(DYNAMIC_ALPHA_BASE_MIN,
#                        DYNAMIC_ALPHA_BASE_MAX * (10/nb_classes)**DYNAMIC_ALPHA_BASE_EXP))
# NOTE: Both 10-class and 100-class use the same base formula parameters.
DYNAMIC_ALPHA_BASE_MAX   = 0.45
DYNAMIC_ALPHA_BASE_MIN   = 0.25
DYNAMIC_ALPHA_BASE_EXP   = 0.3
