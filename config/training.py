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
# BASIC TRAINING HYPERPARAMETERS
# ==============================================================================

if NB_CLASSES <= 10:
    # 64 better for high-accuracy refinement on 10cls — smoother gradients
    # for the final 0.990→0.9994 squeeze without cold-start risk.
    BATCH_SIZE = 32
else:
    # 100-class: smaller batches help with class imbalance and gradient diversity
    BATCH_SIZE = 32

TRAINING_PERCENTAGE = 1.0  # Use 100% of available data

# ---------------------------------------------------------------------- #
#  Per-Percentage EPOCHS & LEARNING_RATE
# ---------------------------------------------------------------------- #
# 10cls converges at ~30 epochs with 100% data — scale epochs down
# proportionally for smaller fractions so training doesn't waste compute.
# LR is also scaled down gently to avoid overfitting on limited samples.
EPOCHS = 250  # Upper ceiling — early stopping decides actual duration

if TRAINING_PERCENTAGE <= 0.1:
    LEARNING_RATE = 5e-4
elif TRAINING_PERCENTAGE <= 0.25:
    LEARNING_RATE = 6e-4
elif TRAINING_PERCENTAGE <= 0.5:
    LEARNING_RATE = 8e-4
else:
    LEARNING_RATE = 1e-3

VALIDATION_SPLIT = 0.2     # 20% of training for validation

# ---------------------------------------------------------------------- #
#  OneCycle / Cosine LR helpers (used by DynamicSchedulerController and model_factory)
# ---------------------------------------------------------------------- #
ONECYCLE_WARMUP_FRACTION = 0.3       # % of total epochs used for linear warm-up
ONECYCLE_INITIAL_LR_FRACTION = 0.1   # initial LR = peak LR × this fraction

# ---------------------------------------------------------------------- #
#  QAT LR multiplier (used by train_qat_helper)
# ---------------------------------------------------------------------- #
QAT_LR_MULTIPLIER = 2.0              # QAT fine-tuning LR = LEARNING_RATE × this

# ==============================================================================
# LEARNING RATE SCHEDULING
# ==============================================================================


# Learning Rate Scheduler
# Options:
#   - "reduce_on_plateau":
#       Halves LR each time val_loss stops improving (factor × current LR).
#       ✅ Best proven for 100cls — stable, well-understood staircase decay.
#       ✅ Best default for cold-start QAT models (safe, no epoch-count dependency).
#       ⚠️  Eventually decays LR to the noise floor; model can get stranded 20–30 eps.
#       → Use with USE_DYNAMIC_SCHEDULER=True + cosine phase to escape the ceiling.
#
#   - "onecycle":
#       Linear LR warm-up (30% of epochs) → cosine decay to near-zero.
#       ✅ Super-convergence: reaches high accuracy faster on well-trained small models.
#       ❌ Cold-start QAT + 100cls: too aggressive — early stopping kills training at ep~6.
#       → Safe for 10cls fine-tuning or as Phase-0 in DynamicSchedulerController.
#
#   - "cosine":
#       CosineDecayRestarts: LR periodically resets (with shrinking peaks) to escape minima.
#       ✅ Good escape mechanism after a plateau — avoids the noise-floor trap.
#       ⚠️  Requires tuning of LR_WARMUP_EPOCHS (= first_decay_steps). Too short = chaotic.
#       → Ideal as the final phase in LR_SCHEDULER_SEQUENCE (after reduce_on_plateau).
#
#   - "exponential":
#       Smooth exponential decay: LR × EXPONENTIAL_DECAY_RATE every EXPONENTIAL_DECAY_STEPS.
#       ✅ Predictable, no plateaus. Good when you know convergence speed in advance.
#       ⚠️  Decays even when model is still improving (wastes capacity early on).
#
#   - "step":
#       Drops LR by STEP_DECAY_GAMMA every STEP_DECAY_STEP_SIZE epochs.
#       ✅ Simple and interpretable. Useful for manual LR design.
#       ⚠️  Coarse — sudden drops can destabilize training if steps are too small.

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
    LR_SCHEDULER_THRESHOLDS = [0.985, 0.992] #[0.990, 0.995]
else:
    LR_SCHEDULER_THRESHOLDS = [0.75, 0.82]

LR_SCHEDULER_SEQUENCE = ["reduce_on_plateau", "reduce_on_plateau", "cosine"]

# Per-phase LR reset fractions — gentler reset for later phases near
# the accuracy ceiling.  Each element corresponds to the transition
# from phase N to phase N+1 (len = len(LR_SCHEDULER_THRESHOLDS)).
# A scalar value is also accepted for backward compatibility.
if NB_CLASSES <= 10:
    LR_SCHEDULER_RESET_FRACTION = [0.5, 0.4, 0.3]
else:
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
L2_REGULARIZATION = 1e-5 # was 0
DEFAULT_DROPOUT_RATE = 0.2
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
