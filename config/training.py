"""
config/training.py — Training hyperparameters separated for 10 vs 100 classes.

All parameters that differ between 10-class and 100-class training are
defined here with ``if NB_CLASSES <= 10:`` / ``else:`` branching.

Import this module after ``NB_CLASSES`` has been resolved (e.g. via
``from config import NB_CLASSES``).
"""

import sys

# Get NB_CLASSES from the parent config module (avoids circular import)
# config/__init__.py defines NB_CLASSES before importing submodules
_config_mod = sys.modules.get('config')
if _config_mod is not None and hasattr(_config_mod, 'NB_CLASSES'):
    NB_CLASSES = _config_mod.NB_CLASSES
else:
    # Fallback: read from environment
    import os
    NB_CLASSES = int(os.environ.get("DIGIT_NB_CLASSES", "100"))
del _config_mod

# ==============================================================================
# LOSS CONFIGURATION
# ==============================================================================

LOSS_TYPE = "IntelligentFocalLossController"

if NB_CLASSES <= 10:
    LABEL_SMOOTHING = 0.01  # reduced: at >0.999 ceiling, 0.02 caps final accuracy
else:  # 100 classes
    LABEL_SMOOTHING = 0.05  # stronger smoothing — 100cls softmax easily collapses

# Focal Loss Parameters
FOCAL_GAMMA = 2.0      # Robust standard focus parameter

if NB_CLASSES <= 10:
    FOCAL_ALPHA = 0.45  # Your current value for 10 classes
elif NB_CLASSES <= 20:
    FOCAL_ALPHA = 0.38  # Sweet spot for 15-20 classes
elif NB_CLASSES <= 50:
    FOCAL_ALPHA = 0.32  # For medium-sized datasets
else:  # 100 classes
    FOCAL_ALPHA = 0.27  # Optimal for 100 classes

# Intelligent Focal Loss Controller Settings
if NB_CLASSES <= 10:
    # --- 10-class: delay focal until model is genuinely stuck near ceiling ---
    FOCAL_ACCURACY_THRESHOLDS = [0.985, 0.991, 0.995]
else:  # 100 classes or more
    # --- 100-class: wait longer; model needs to learn easy examples first ---
    FOCAL_ACCURACY_THRESHOLDS = [0.88, 0.93, 0.97]

if NB_CLASSES <= 10:
    # Gentler γ ramp: at 0.985+ the model only needs a soft push, not heavy focus
    FOCAL_GAMMA_VALUES = [1.0, 2.0, 3.5]
else:
    FOCAL_GAMMA_VALUES = [1.2, 2.0, 3.5]   # gentler ramp for 100cls

# Smooth γ transition: ramp linearly over N epochs instead of an instant step.
FOCAL_GAMMA_RAMP_EPOCHS = 3

# Plateau detection for IntelligentFocalLossController
FOCAL_PLATEAU_PATIENCE = 8
FOCAL_PLATEAU_MIN_DELTA = 0.0005

# Dynamic Class Weighting
USE_DYNAMIC_WEIGHTS = True
DYNAMIC_WEIGHTS_EPOCHS = 5

# ==============================================================================
# BASIC TRAINING HYPERPARAMETERS
# ==============================================================================

if NB_CLASSES <= 10:
    # 64 better for high-accuracy refinement on 10cls — smoother gradients
    # for the final 0.990→0.9994 squeeze without cold-start risk.
    BATCH_SIZE = 64
else:
    # 100-class: smaller batches help with class imbalance and gradient diversity
    BATCH_SIZE = 32

EPOCHS = 250
LEARNING_RATE = 0.001  # Robust default for cold-start
TRAINING_PERCENTAGE = 1.0  # Use 100% of available data
VALIDATION_SPLIT = 0.2     # 20% of training for validation

# ==============================================================================
# LEARNING RATE SCHEDULING
# ==============================================================================

USE_LEARNING_RATE_SCHEDULER = True
LR_SCHEDULER_TYPE = "reduce_on_plateau"  # best for 100cls per training analysis

# ReduceLROnPlateau Parameters
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_MIN_LR = 1e-7
LR_SCHEDULER_FACTOR = 0.4
LR_SCHEDULER_MONITOR = 'val_loss'

# LR Warm-up (only active when USE_LR_WARMUP=True)
USE_LR_WARMUP = False
LR_WARMUP_INITIAL_SCALE = 0.1

# Exponential Decay Parameters
EXPONENTIAL_DECAY_STEPS = 1000
EXPONENTIAL_DECAY_RATE = 0.96

# Cosine Decay Parameters
COSINE_DECAY_ALPHA = 1e-6
LR_WARMUP_EPOCHS = 15

# Step Decay Parameters
STEP_DECAY_STEP_SIZE = 10
STEP_DECAY_GAMMA = 0.1

# Dynamic Scheduler Controller
USE_DYNAMIC_SCHEDULER = True

if NB_CLASSES <= 10:
    LR_SCHEDULER_THRESHOLDS = [0.990, 0.995]
else:
    LR_SCHEDULER_THRESHOLDS = [0.75, 0.82]

LR_SCHEDULER_SEQUENCE = ["reduce_on_plateau", "reduce_on_plateau", "cosine"]
LR_SCHEDULER_RESET_FRACTION = 0.5

# Dynamic Optimizer (disabled by default — resets momentum state)
USE_DYNAMIC_OPTIMIZER = False
if NB_CLASSES <= 10:
    OPTIMIZER_SEQUENCE = ["rmsprop", "rmsprop", "adamw"]
else:
    OPTIMIZER_SEQUENCE = ["rmsprop", "rmsprop", "adamw"]

# ==============================================================================
# REGULARIZATION HYPERPARAMETERS
# ==============================================================================

L1_REGULARIZATION = 0.0
L2_REGULARIZATION = 0.0
DEFAULT_DROPOUT_RATE = 0.5
USE_BATCH_NORM = True
BATCH_NORM_MOMENTUM = 0.99
BATCH_NORM_EPSILON = 0.001

# ==============================================================================
# GRADIENT & TRAINING HYPERPARAMETERS
# ==============================================================================

USE_GRADIENT_CLIPPING = False
GRADIENT_CLIP_VALUE = 1.0
GRADIENT_CLIP_NORM = 1.0
WEIGHT_INITIALIZER = "he_normal"

# ==============================================================================
# CALLBACK HYPERPARAMETERS
# ==============================================================================

USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MONITOR = 'val_accuracy'
EARLY_STOPPING_MIN_DELTA = 0.0002
RESTORE_BEST_WEIGHTS = True

SAVE_CHECKPOINTS = True
CHECKPOINT_FREQUENCY = 5
SAVE_BEST_ONLY = True
CHECKPOINT_MONITOR = 'val_accuracy'

USE_TENSORBOARD = False
TENSORBOARD_UPDATE_FREQ = 'epoch'
TENSORBOARD_WRITE_GRAPHS = True

# Resume Training Support
RESUME_MODEL_PATH = ""
INITIAL_EPOCH = 0

# Training verbosity
VERBOSE = 1

# Training visualization
SAVE_TRAINING_PLOTS = True

# Data shuffling
SHUFFLE_SEED = 42

# NOTE: Advanced training configuration (USE_MIXED_PRECISION, USE_GRADIENT_ACCUMULATION,
# USE_LEARNING_RATE_FINDER, USE_STOCHASTIC_WEIGHT_AVERAGING, USE_CYCLICAL_LEARNING_RATE,
# USE_MODEL_ENSEMBLE, ENSEMBLE_MODEL_COUNT) has been moved to config/augmentation.py
# to avoid duplication.
