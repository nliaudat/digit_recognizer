"""
Global parameters for Digit Recognition
Comprehensive hyperparameter configuration for neural network training
"""

import os
import sys

# Force UTF-8 output on Windows to support emojis
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf

# ==============================================================================
# MODEL SELECTION
# ==============================================================================

AVAILABLE_MODELS = [
    # "practical_tiny_depthwise", # 150-300kB / 95%+ (10cls) | Fixed Depthwise Separable
    # "simple_cnn", # Est: ~20-60K params | BatchNorm CNN Baseline
    # "dig_class100_s2", # Est: ~500K+ params | Haverland Reference (100-Class)
    # "cnn32", # Est: ~200-300K params | Haverland Original Reference (Matches original_haverland)
    # "digit_recognizer_v1", # ~130kB | ESP-DL Compatible Baseline (Est: ~135K params)
    # "digit_recognizer_v2", # ~50kB | ReLU6 Clip Baseline (Est: ~50K params)
    "digit_recognizer_v3", # 45.1kB / 75.34%
    "digit_recognizer_v4", # 87.1kB / 80.80%
    # "digit_recognizer_v5", # ~55-130kB | Class-Count Adaptive CNN
    "digit_recognizer_v6", # 160.8kB / 80.56%
    # "digit_recognizer_v7", # 56.0kB / 73.35%
    # "digit_recognizer_v8", # Est: ~300K+ params | SOTA Residual CNN with SE
    # "digit_recognizer_v9", # Est: ~150-200K params | EfficientNet-Style MBConv
    # "digit_recognizer_v10", # Est: ~700K+ params | Hybrid Transformer-Style CNN
    # "digit_recognizer_v11", # Est: ~1M+ params | Modern SOTA CNN with GELU/Swish/SE/MBConv
    # "digit_recognizer_v12", # 415.4kB / 84.46%
    "digit_recognizer_v15", # 107.4kB / 79.78% | IoT residual model — beats v4 accuracy at <100KB
    "digit_recognizer_v16", # 139.7kB / 82.81% | IoT MobileNetV2 inverted residual — ESP-NN optimised
    "digit_recognizer_v17", # 80.5kB / 80.69% | IoT GhostNet-inspired — ultra-efficient ~50KB
    "digit_recognizer_v18", # GhostNet 10-Class IoT optimized model pushing >90% accuracy <100KB INT8
    "digit_recognizer_v19", # GhostNet scale-up hitting 92% accurate baseline for 100-class
    "digit_recognizer_v20", # GhostNet 100-Class IoT limit-pusher (<1.5MB) with 2D Positional Encoding & Dual Attention
    "digit_recognizer_v21", # PC/GPU-Only Rotary Positional Encoding & Adaptive Attention limit pusher (>99.5% target)
    "digit_recognizer_v22", # IoT Spatial MobileNetV2 with 2D Positional Encoding (10-Class RGB <200KB limit)
    # "esp_quantization_ready", # ~70kB | Minimal Depthwise CNN for smooth INT8
    # "high_accuracy_validator", # strictly for PC validation (not for ESP32)
    # "super_high_accuracy_validator", # GPU-only deep SE-ResNet validator (2026 SOTA)
    # "mnist_quantization", # 72.2kB / 76.55%
    # "original_haverland", # 228.8kB / 79.10% | baseline
]

MODEL_ARCHITECTURE = "digit_recognizer_v22" # one of the models in AVAILABLE_MODELS


# ==============================================================================
# GENERAL PARAMETERS
# ==============================================================================

### MANUAL OVERRIDES (Set to None to use Environment Variables or Defaults)
# If set, these will supersede environment variables.
MANUAL_NB_CLASSES = None # 10 or 100
MANUAL_INPUT_CHANNELS = None # 1 (Gray) or 3 (RGB)

# --- NB_CLASSES Logic ---
_nb_classes_env = os.environ.get("DIGIT_NB_CLASSES")
if MANUAL_NB_CLASSES is not None:
    NB_CLASSES = MANUAL_NB_CLASSES
elif _nb_classes_env is not None:
    NB_CLASSES = int(_nb_classes_env)
elif "-h" in sys.argv or "--help" in sys.argv:
    # Avoid interactive prompt when just showing help
    NB_CLASSES = 100
else:
    # Not set via environment – ask the user to avoid silently using a wrong default
    if sys.stdin.isatty():
        while True:
            try:
                _user_input = input("Enter number of classes [10 or 100]: ").strip()
                if _user_input in ("10", "100"):
                    NB_CLASSES = int(_user_input)
                    break
                print("  Please enter 10 or 100.")
            except EOFError:
                NB_CLASSES = 100
                break
    else:
        # Non-interactive context (subprocess, CI, etc.) – keep a safe default and warn
        NB_CLASSES = 100
        print("WARNING: DIGIT_NB_CLASSES not set and no interactive terminal – defaulting to 100. "
              "Set the env var explicitly to avoid this.")
del _nb_classes_env

# --- INPUT_CHANNELS Logic ---
_input_channels_env = os.environ.get("DIGIT_INPUT_CHANNELS")
if MANUAL_INPUT_CHANNELS is not None:
    INPUT_CHANNELS = MANUAL_INPUT_CHANNELS
elif _input_channels_env is not None:
    INPUT_CHANNELS = int(_input_channels_env)
elif "-h" in sys.argv or "--help" in sys.argv:
    # Avoid interactive prompt when just showing help
    INPUT_CHANNELS = 1
else:
    # Not set via environment – ask the user to avoid silently using a wrong default
    if sys.stdin.isatty():
        while True:
            try:
                _user_input = input("Enter color mode [gray or rgb]: ").strip().lower()
                if _user_input == "gray":
                    INPUT_CHANNELS = 1
                    break
                elif _user_input == "rgb":
                    INPUT_CHANNELS = 3
                    break
                print("  Please enter 'gray' or 'rgb'.")
            except EOFError:
                INPUT_CHANNELS = 3
                break
    else:
        # Non-interactive context (subprocess, CI, etc.) – keep a safe default and warn
        INPUT_CHANNELS = 3
        print("WARNING: DIGIT_INPUT_CHANNELS not set and no interactive terminal – defaulting to 3 (RGB). "
              "Set the env var explicitly to avoid this.")
del _input_channels_env

# ==============================================================================
# INPUT IMAGES 
# ==============================================================================


# Image Parameters
INPUT_WIDTH = 20
INPUT_HEIGHT = 32

def update_derived_parameters():
    """Refresh parameters that depend on NB_CLASSES or INPUT_CHANNELS"""
    global INPUT_SHAPE, USE_GRAYSCALE, OUTPUT_DIR, DATA_SOURCES
    INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
    USE_GRAYSCALE = (INPUT_CHANNELS == 1)
    _color_suffix = "GRAY" if USE_GRAYSCALE else "RGB"
    OUTPUT_DIR = f"exported_models/{NB_CLASSES}cls_{_color_suffix}"
    
    # Refresh DATA_SOURCES to use correct NB_CLASSES for label files
    DATA_SOURCES = [
        {
            'name': 'Tenth-of-step-of-a-meter-digit',
            'type': 'label_file', 
            'labels': f'labels_{NB_CLASSES}_shuffle.txt',
            'path': 'datasets/Tenth-of-step-of-a-meter-digit', 
            'weight': 1.0,
        },
        {
            'name': 'real_integra_bad_predictions',
            'type': 'label_file', 
            'labels': f'labels_{NB_CLASSES}_shuffle.txt',  
            'path': 'datasets/real_integra_bad_predictions', 
            'weight': 1.9,
        },
        {
            'name': 'real_integra',
            'type': 'label_file', 
            'labels': f'labels_{NB_CLASSES}_shuffle.txt',  
            'path': 'datasets/real_integra', 
            'weight': 0.7,
        },
        {
            'name': f'failed_predictions_{NB_CLASSES}',
            'type': 'label_file', 
            'labels': f'labels_{NB_CLASSES}_shuffle.txt',  
            'path': f'datasets/failed_predictions_{NB_CLASSES}', 
            'weight': 1.3,
        },
        {
            'name': 'static_augmentation',
            'type': 'label_file', 
            'labels': f'labels_{NB_CLASSES}_shuffle.txt',  
            'path': 'datasets/static_augmentation', 
            'weight': 0.6,
        },
    ]

# Initial call to set defaults
update_derived_parameters()

# ==============================================================================
# DATA SOURCES
# ==============================================================================

# This is far better to use labels that have been shuffled for training, folder_structure shuffle by batch TF_DATA_SHUFFLE_BUFFER and SHUFFLE_SEED

# (Initial DATA_SOURCES is set via update_derived_parameters() below)

# ==============================================================================
# QUANTIZATION PARAMETERS
# ==============================================================================

# QUANTIZATION MODES (9 possible combinations):

# 1. QUANTIZE_MODEL=False, USE_QAT=False, ESP_DL_QUANTIZE=False
   # Float32 training & inference

# 2. QUANTIZE_MODEL=False, USE_QAT=False, ESP_DL_QUANTIZE=True  
   # INVALID (ESP_DL requires quantization)

# 3. QUANTIZE_MODEL=False, USE_QAT=True, ESP_DL_QUANTIZE=False
   # QAT training, float32 inference

# 4. QUANTIZE_MODEL=False, USE_QAT=True, ESP_DL_QUANTIZE=True
   # INVALID (ESP_DL requires quantization)

# 5. QUANTIZE_MODEL=True, USE_QAT=False, ESP_DL_QUANTIZE=False
   # Standard training, UINT8 post-quantization

# 6. QUANTIZE_MODEL=True, USE_QAT=False, ESP_DL_QUANTIZE=True
   # Standard training, INT8 post-quantization (ESP-DL)

# 7. QUANTIZE_MODEL=True, USE_QAT=True, ESP_DL_QUANTIZE=False  
   # QAT training, UINT8 quantization

# 8. QUANTIZE_MODEL=True, USE_QAT=True, ESP_DL_QUANTIZE=True
   # QAT training, INT8 quantization (ESP-DL)

# TFLite Conversion Parameters
QUANTIZE_MODEL = True # Enable post-training quantization for the TFLite model
# ESP-DL specific quantization (only applies if QUANTIZE_MODEL = True)
ESP_DL_QUANTIZE = False  # Quantize to int8 range [-128, 127] for ESP-DL
                         # If False: quantize to uint8 range [0, 255] (default)
                         
# Quantization Aware Training
USE_QAT = True  # Enable Quantization Aware Training
QAT_QUANTIZE_ALL = True  # Quantize all layers
QAT_SCHEME = '8bit'  # Options: '8bit', 'float16'

# Automatically disable quantization flags for PC-only validator models
# PC_ONLY_MODELS = {"high_accuracy_validator", "super_high_accuracy_validator"}
# if MODEL_ARCHITECTURE in PC_ONLY_MODELS:
#     QUANTIZE_MODEL = False
#     ESP_DL_QUANTIZE = False
#     USE_QAT = False

# Dataset disk cache directory
DATASET_CACHE_DIR = os.environ.get("DATASET_CACHE_DIR", ".dataset_cache")

# Data pipeline configuration
USE_TF_DATA_PIPELINE = False
TF_DATA_PARALLEL_CALLS = tf.data.AUTOTUNE
TF_DATA_SHUFFLE_BUFFER = 1000
TF_DATA_PREFETCH_SIZE = tf.data.AUTOTUNE
QUANTIZE_NUM_SAMPLES = 22000
# TFLITE_FILENAME = f"{MODEL_FILENAME}.tflite"
# FLOAT_TFLITE_FILENAME = f"{MODEL_FILENAME}_float.tflite"

# Debug and Logging
VERBOSE = 1
SAVE_TRAINING_PLOTS = True
SHUFFLE_SEED = 42

# Post training analyse
# ANALYSE_SAMPLES = 25000

# ==============================================================================
# MODEL-SPECIFIC PARAMETERS
# ==============================================================================

# For practical_tiny_depthwise
DEPTHWISE_FILTERS = [32, 64]
POINTWISE_FILTERS = [32, 64]

# For simple_cnn  
SIMPLE_CNN_FILTERS = [32, 64]
SIMPLE_CNN_DENSE_UNITS = 128

# For dig_class100_s2 
DIG_CLASS100_FILTERS = [32, 64, 128]
DIG_CLASS100_DENSE_UNITS = 512
DROPOUT_RATE = 0.5

# For original_haverland (exact replica)
ORIGINAL_HAVERLAND_FILTERS = [32, 64, 128]  # Fixed values from notebook
ORIGINAL_HAVERLAND_DENSE_UNITS = 512        # Fixed value from notebook
ORIGINAL_HAVERLAND_DROPOUT_RATES = [0.25, 0.25, 0.25, 0.5]  # Fixed from notebook


# ==============================================================================
# OPTIMIZER HYPERPARAMETERS
# ==============================================================================

# Optimizer Selection
# Options:
#   - "rmsprop":
#       Adaptive per-parameter LR (rho=0.9). Handles noisy gradients well.
#       ✅ Best proven for 100cls QAT — all successful runs used RMSprop.
#       ✅ Robust to the gradient noise introduced by fake-quantization.
#       ⚠️  No weight decay — can overfit on long runs (use L2_REGULARIZATION).
#
#   - "adam":
#       Adaptive moment estimation (β1=0.9, β2=0.999). Fast, popular default.
#       ✅ Good general baseline for most tasks.
#       ⚠️  Weight decay in Adam is incorrect (decoupled in AdamW) — prefer AdamW.
#       ⚠️  Slightly worse than RMSprop on 100cls historically — test with tuner.
#
#   - "adamw":
#       Adam with proper decoupled weight decay. 2024-2026 standard for fine-tuning.
#       ✅ Best for regularised fine-tuning and escaping the ceiling (Phase 2 switch).
#       ⚠️  Cold-start QAT + 100cls: slower initial climb than RMSprop.
#       → Best used as Phase 2 in OPTIMIZER_SEQUENCE after RMSprop climb.
#
#   - "nadam":
#       Adam + Nesterov momentum. Often converges faster than plain Adam.
#       ✅ Worth testing if Adam plateaus — lookahead corrects overshoot.
#       ⚠️  Untested on this project's 100cls QAT — add to Group A config_runner run.
#
#   - "sgd":
#       Classic stochastic gradient descent with momentum (momentum=0.9, Nesterov=True).
#       ✅ Best final-layer fine-tuning convergence when combined with cosine annealing.
#       ❌ Slow cold-start — needs many epochs to settle without a warm-up.
#       → Never use alone for cold-start 100cls; pair with CosineDecayRestarts.
OPTIMIZER_TYPE = "nadam"            # Best default for cold-start 100cls
# OPTIMIZER_TYPE = "rmsprop"            # Best default for cold-start 100cls
# OPTIMIZER_TYPE = "sgd"              # (Optimized for v17 fine-tuning: best balance with 0.0003 LR / 128 BS)
# OPTIMIZER_TYPE = "adamw"            # ← restore to roll back

# RMSprop Hyperparameters
RMSPROP_RHO = 0.9
RMSPROP_MOMENTUM = 0.0
RMSPROP_EPSILON = 1e-07

# Adam Hyperparameters
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1e-07
ADAM_AMSGRAD = False

# SGD Hyperparameters  
SGD_MOMENTUM = 0.9
SGD_NESTEROV = True

# AdaGrad Hyperparameters
ADAGRAD_INITIAL_ACCUMULATOR = 0.1
ADAGRAD_EPSILON = 1e-07

# AdamW Hyperparameters (Adam with Weight Decay)
ADAMW_WEIGHT_DECAY = 0.01
ADAMW_BETA_1 = 0.9
ADAMW_BETA_2 = 0.999
ADAMW_EPSILON = 1e-07


# ==============================================================================
# --------------------------------------------------------------------------- #
#  Training & Loss Configuration
# --------------------------------------------------------------------------- #
# Options:
#   - "IntelligentFocalLossController":
#       Adaptive Focal Loss that starts as CrossEntropy (γ=0) and gradually
#       increases γ at val_acc thresholds (FOCAL_ACCURACY_THRESHOLDS).
#       Also detects plateaus and adjusts α per-class when stuck.
#       ✅ Best for most tasks — zero config needed, self-tuning.
#       ⚠️  Adds ~15% training overhead (plateau detection / α recompute).
#
#   - "focal_loss":
#       Standard Focal Loss with fixed γ=FOCAL_GAMMA and α=FOCAL_ALPHA.
#       Down-weights well-classified examples so the model focuses on hard ones.
#       ✅ Good when class imbalance is the main problem.
#       ⚠️  Requires manual tuning of γ (start 1.0–2.0) and α (0.25–0.45).
#       ⚠️  High γ (>3) can destabilize early training on 100-class tasks.
#
#   - "sparse_categorical_crossentropy":
#       Standard CrossEntropy for integer labels (all models except haverland).
#       Fast, stable, no hyperparameters.
#       ✅ Best baseline; use to diagnose if focal/controller is hurting accuracy.
#       ❌ No focus on hard examples; hits a ceiling earlier on complex tasks.
#
#   - "categorical_crossentropy":
#       Standard CrossEntropy for one-hot labels.
#       ✅ Required for original_haverland model (uses softmax + one-hot).
#       ❌ Do not use with other models (label format mismatch).
LOSS_TYPE = "IntelligentFocalLossController"
if NB_CLASSES <= 10:
    LABEL_SMOOTHING = 0.02  # mild smoothing — 10cls rarely overconfident
else:  # 100 classes
    LABEL_SMOOTHING = 0.05  # stronger smoothing — 100cls softmax easily collapses

# Focal Loss Parameters
FOCAL_GAMMA = 2.0      # Robust standard focus parameter
# FOCAL_GAMMA = 0.7     # (Optimized for v17 fine-tuning)
# FOCAL_ALPHA = 0.45     # Class balancing (0.25 recommended for binary, 0.5 for multi-class)
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
    # --- 10-class: delay until SCCE natural ceiling ---
    # (ep13 analysis: controller fired at val_acc=0.9656, causing a −0.006 dip)
    FOCAL_ACCURACY_THRESHOLDS = [0.96, 0.975, 0.988]  # was [0.95, 0.97, 0.985]
else:  # 100 classes or more
    # --- 100-class: wait longer; model needs to learn easy examples first ---
    # (ep38 analysis: γ fired at val_acc≈0.786, too early for 100-class task)
    FOCAL_ACCURACY_THRESHOLDS = [0.88, 0.93, 0.97]   # was [0.85, 0.92, 0.96]

if NB_CLASSES <= 10:
    FOCAL_GAMMA_VALUES = [1.5, 3.0, 4.5]   # unchanged for 10cls (already proven)
else:
    FOCAL_GAMMA_VALUES = [1.2, 2.0, 3.5]   # gentler ramp for 100cls (was [1.5, 3.0, 4.5])

# Smooth γ transition: ramp linearly over N epochs instead of an instant step.
# Set to 0 to keep the original hard-switch behaviour.
FOCAL_GAMMA_RAMP_EPOCHS = 5             # NEW — eliminates the γ-activation accuracy dip

FOCAL_PLATEAU_PATIENCE = 5
FOCAL_PLATEAU_MIN_DELTA = 0.001

# Advanced Training Options (defaulting to True as requested in prev sessions)
USE_EARLY_STOPPING = True
# 2.0      # Focus parameter for Focal Loss (0 = CrossEntropy) # This line was a duplicate and commented out.

# Dynamic Class Weighting
USE_DYNAMIC_WEIGHTS = True   # Update loss weights based on validation accuracy
DYNAMIC_WEIGHTS_EPOCHS = 5  # Frequency of dynamic weight updates (in epochs)

# Resume Training Support
RESUME_MODEL_PATH = ""  # Path to best_model.keras to resume from (empty = start fresh)
INITIAL_EPOCH = 0       # Epoch to resume from (auto-detected by retrain_all.py)

# ==============================================================================
# HYPERPARAMETER TUNING (tuner.py)
# ==============================================================================

TUNER_MAX_TRIALS = 30       # Maximum number of unique combinations to test
TUNER_EPOCHS = 10           # Epochs per trial (short training to find best parameters)
TUNER_EARLY_STOPPING_PATIENCE = 3

# Search Space
TUNER_OPTIMIZERS = ['adam', 'rmsprop', 'sgd', 'nadam', 'adamw']  # adamw added
TUNER_LEARNING_RATES = [1e-3, 5e-4, 2e-4, 1e-4]  # Refined precision for high-accuracy tasks
TUNER_BATCH_SIZES = [32, 64]
# Added 1.2 and 3.5 based on 100-class log analysis (gentler focal ramp)
TUNER_GAMMAS = [0.0, 1.2, 1.5, 2.0, 3.0, 3.5, 4.5]
TUNER_ALPHAS = [0.25, 0.45]

# Fine-Tune Tuner Search Space (used by: python tuner.py --finetune)
# Loads a pre-trained best_model.keras and searches only post-plateau decisions:
TUNER_FINETUNE_EPOCHS = 15                                        # epochs per fine-tune trial
TUNER_FINETUNE_LRS = [5e-5, 1e-4, 3e-4, 5e-4]                  # small LRs only
TUNER_FINETUNE_OPTIMIZERS = ['adam', 'rmsprop', 'nadam', 'sgd', 'adamw']  # adamw added
TUNER_LR_FACTORS = [0.3, 0.5, 0.7]                              # ReduceLROnPlateau factor to sweep
FINETUNE_UNFREEZE_LAST_N = 0                                      # 0 = unfreeze all layers

# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================

# Basic Training Parameters
BATCH_SIZE = 32      # Robust default for cold-start (use 128 only after initial 50-80% accuracy)
EPOCHS = 250
LEARNING_RATE = 0.001 # Robust default for cold-start
# BATCH_SIZE = 128      # (Optimized for v17 fine-tuning speed)
# LEARNING_RATE = 0.0003 # (Optimized for v17 fine-tuning stability)
TRAINING_PERCENTAGE = 1.0  # Use 100% of available data
VALIDATION_SPLIT = 0.2     # 20% of training for validation



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
# LR_SCHEDULER_TYPE = "onecycle"         # ← restore to roll back

# ReduceLROnPlateau Parameters
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_MIN_LR = 1e-7
# Nudged from 0.5 →0.4: slightly faster decay avoids 20-epoch plateau-creep
# (both 10cls and 100cls runs showed ~15-25 wasted epochs at near-identical val_acc)
LR_SCHEDULER_FACTOR = 0.4
LR_SCHEDULER_MONITOR = 'val_loss'

# LR Warm-up (only active when USE_LR_WARMUP=True).
# Ramps LR from LEARNING_RATE × scale → LEARNING_RATE over LR_WARMUP_EPOCHS.
# Disabled for reduce_on_plateau: the first few steps already use full LR and
# ReduceLROnPlateau handles the decay naturally. Useful only for onecycle/cosine.
USE_LR_WARMUP = False         # not needed with reduce_on_plateau
LR_WARMUP_INITIAL_SCALE = 0.1 # start LR fraction when USE_LR_WARMUP=True

# Exponential Decay Parameters
EXPONENTIAL_DECAY_STEPS = 1000
EXPONENTIAL_DECAY_RATE = 0.96

# Cosine Decay Parameters
COSINE_DECAY_ALPHA = 1e-6    # Minimum LR floor (never decays below this)
# LR_WARMUP_EPOCHS doubles as `first_decay_steps` for CosineDecayRestarts:
# 15 epochs gives the model time to settle before each restart kick.
LR_WARMUP_EPOCHS = 15        # was 5 — longer period makes cosine restarts more useful

# Step Decay Parameters
STEP_DECAY_STEP_SIZE = 10
STEP_DECAY_GAMMA = 0.1

# Dynamic Scheduler Controller
# Switches the active LR scheduler at val_accuracy thresholds, similar to how
# IntelligentFocalLossController switches γ. Requires USE_LEARNING_RATE_SCHEDULER=True.
USE_DYNAMIC_SCHEDULER = True    # enabled: switches scheduler at val_acc thresholds

# val_accuracy thresholds that trigger a phase switch.
# Must have exactly len(LR_SCHEDULER_SEQUENCE) - 1 values.
if NB_CLASSES <= 10:
    LR_SCHEDULER_THRESHOLDS = [0.95, 0.975]   # 10cls: switch near the accuracy ceiling
else:
    LR_SCHEDULER_THRESHOLDS = [0.75, 0.82]    # 100cls: switch at mid-plateau and ceiling

# Scheduler per phase: Phase 0 = start, each threshold triggers the next.
# Phase 0+1: reduce_on_plateau (proven fast-climb); Phase 2: cosine to escape ceiling.
# Options: 'reduce_on_plateau' | 'cosine' | 'exponential' | 'step' | 'onecycle'
LR_SCHEDULER_SEQUENCE = ["reduce_on_plateau", "reduce_on_plateau", "cosine"]

# On phase switch, reset LR to this fraction of LEARNING_RATE.
# E.g. 0.5 → restores to 5e-4 when cosine kicks in after ReduceLROnPlateau has decayed.
# Set to None to keep the current (decayed) LR as the starting point for the new phase.
LR_SCHEDULER_RESET_FRACTION = 0.5   # restore to 50% of base LR on each phase switch

# Optional: also switch optimizer at the same thresholds (risky — resets momentum state).
# Disabled by default; use only after validating with USE_DYNAMIC_SCHEDULER first.
# Each entry corresponds to a phase in LR_SCHEDULER_SEQUENCE.
USE_DYNAMIC_OPTIMIZER = False        # ⚠️ NOT SAFE YET: resets momentum state mid-training
                                     # enable only after validating USE_DYNAMIC_SCHEDULER first
if NB_CLASSES <= 10:
    # Phase 0+1: rmsprop for fast initial climb; Phase 2: adamw for fine-tuning near ceiling
    OPTIMIZER_SEQUENCE = ["rmsprop", "rmsprop", "adamw"]
else:
    # Same logic for 100cls — rmsprop proven best for climb, adamw for plateau phase
    OPTIMIZER_SEQUENCE = ["rmsprop", "rmsprop", "adamw"]


# ==============================================================================
# REGULARIZATION HYPERPARAMETERS
# ==============================================================================

# L1/L2 Regularization
L1_REGULARIZATION = 0.0  # L1 regularization factor
L2_REGULARIZATION = 0.0  # L2 regularization factor

# Dropout Rates (can be overridden by model-specific parameters)
DEFAULT_DROPOUT_RATE = 0.5

# Batch Normalization
USE_BATCH_NORM = True
BATCH_NORM_MOMENTUM = 0.99
BATCH_NORM_EPSILON = 0.001

# ==============================================================================
# GRADIENT & TRAINING HYPERPARAMETERS
# ==============================================================================

# Gradient Clipping
USE_GRADIENT_CLIPPING = False
GRADIENT_CLIP_VALUE = 1.0  # Clip by value
GRADIENT_CLIP_NORM = 1.0   # Clip by norm

# Weight Initialization
WEIGHT_INITIALIZER = "he_normal"  # Options: "glorot_uniform", "he_normal", "he_uniform", "lecun_normal"

# ==============================================================================
# CALLBACK HYPERPARAMETERS
# ==============================================================================

# Early Stopping
USE_EARLY_STOPPING = True
# Tightened from 30→20: both runs wasted 15-25 epochs with no real improvement
# (10cls plateau @ep46, 100cls plateau @ep68 — both ran 15-25 epochs beyond the peak)
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MONITOR = 'val_accuracy'
# Raised from 0.0001→0.0002: more realistic threshold at the noise floor
EARLY_STOPPING_MIN_DELTA = 0.0002
RESTORE_BEST_WEIGHTS = True

# Model Checkpoint
SAVE_CHECKPOINTS = True
CHECKPOINT_FREQUENCY = 5
SAVE_BEST_ONLY = True
CHECKPOINT_MONITOR = 'val_accuracy'

# TensorBoard
USE_TENSORBOARD = False
TENSORBOARD_UPDATE_FREQ = 'epoch'
TENSORBOARD_WRITE_GRAPHS = True

# ==============================================================================
# DATA AUGMENTATION
# ==============================================================================

USE_DATA_AUGMENTATION = True
AUGMENTATION_ZOOM_RANGE = 0.1           # ±10% zoom
AUGMENTATION_ROTATION_RANGE = 1.15      # ±1.15 degrees (converted from ±0.02 radians)
AUGMENTATION_CONTRAST_RANGE = 0.1       # ±10% contrast
AUGMENTATION_BRIGHTNESS_RANGE = [0.9, 1.1]  # ±10% brightness
# Disabled
AUGMENTATION_WIDTH_SHIFT_RANGE = 0.0
AUGMENTATION_HEIGHT_SHIFT_RANGE = 0.0
AUGMENTATION_HORIZONTAL_FLIP = False
AUGMENTATION_VERTICAL_FLIP = False

# Advanced Augmentations (used by super_high_accuracy_validator)
USE_MIXUP = False           # Disabled by default for stability (destructive on small images)
USE_CUTMIX = False
USE_RANDOM_ERASING = False # Disabled by default for benchmark stability

# not implemented yet

# AUGMENTATION_SHEAR_RANGE = 0.0
# AUGMENTATION_SATURATION_RANGE = [1.0, 1.0]
# AUGMENTATION_HUE_RANGE = 0.0
# AUGMENTATION_BLUR_RANGE = 0.0
# AUGMENTATION_NOISE_STDDEV = 0.0


# ==============================================================================
# GPU CONFIGURATION
# ==============================================================================

USE_GPU = True  # Set to False to force CPU usage
GPU_MEMORY_GROWTH = True  # Gradually allocate GPU memory instead of all at once
GPU_MEMORY_LIMIT = None  # Set specific memory limit in MB, or None for no limit




# ==============================================================================
# ADVANCED TRAINING CONFIGURATION
# ==============================================================================

# TF.DATA CONFIGURATION
# USE_TF_DATA = True
# TF_DATA_PREFETCH_SIZE = tf.data.AUTOTUNE
# TF_DATA_SHUFFLE_BUFFER = 1000
# TF_DATA_PARALLEL_CALLS = tf.data.AUTOTUNE

# ADVANCED TRAINING
USE_MIXED_PRECISION = False
USE_GRADIENT_ACCUMULATION = False
ACCUMULATION_STEPS = 4

# ADVANCED CALLBACKS
USE_LEARNING_RATE_FINDER = False
USE_STOCHASTIC_WEIGHT_AVERAGING = False
USE_CYCLICAL_LEARNING_RATE = False

# MODEL ENSEMBLING
USE_MODEL_ENSEMBLE = False
ENSEMBLE_MODEL_COUNT = 3

# ==============================================================================
# HYPERPARAMETER TUNING
# ==============================================================================

USE_KERAS_TUNER = True
TUNER_PROJECT_NAME = "digit_recognizer_tuning"
TUNER_MAX_TRIALS = 150
TUNER_EXECUTIONS_PER_TRIAL = 15
TUNER_OBJECTIVE = "val_accuracy"
TUNER_NUM_TRIAL = 10
TUNER_EPOCHS = 15

# Search Space Configuration
TUNER_OPTIMIZERS = ['adam', 'rmsprop', 'sgd', 'nadam', 'adamw'] # Limit to best performers ["rmsprop", "adam", "sgd", "adagrad", "adamw", "nadam"]
TUNER_LEARNING_RATES = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]  # Wider range
TUNER_BATCH_SIZES = [16, 32, 64, 128]  # More options

# Early Stopping for Tuning
TUNER_EARLY_STOPPING_PATIENCE = 10
TUNER_MIN_DELTA = 0.001

# ==============================================================================
# output FUNCTIONS
# ==============================================================================


def get_model_filename():
    return MODEL_ARCHITECTURE

def get_tflite_filename():
    return f"{get_model_filename()}.tflite"

def get_float_tflite_filename():
    return f"{get_model_filename()}_float.tflite"


# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

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
    
    print("✅ All hyperparameters validated successfully!")

def validate_quantization_parameters():
    """
    Validate and correct quantization parameter combinations
    Returns: (is_valid, corrected_params, message)
    """
    original_params = {
        'QUANTIZE_MODEL': QUANTIZE_MODEL,
        'USE_QAT': USE_QAT, 
        'ESP_DL_QUANTIZE': ESP_DL_QUANTIZE
    }
    
    corrected_params = original_params.copy()
    messages = []
    
    # Rule 0: high_accuracy_validator does not support quantization
    # if MODEL_ARCHITECTURE == "high_accuracy_validator":
    #     if QUANTIZE_MODEL or USE_QAT or ESP_DL_QUANTIZE:
    #         messages.append("❌ 'high_accuracy_validator' selected: Disabling all quantization flags (QUANTIZE_MODEL, USE_QAT, ESP_DL_QUANTIZE)")
    #         corrected_params['QUANTIZE_MODEL'] = False
    #         corrected_params['USE_QAT'] = False
    #         corrected_params['ESP_DL_QUANTIZE'] = False
    #         messages.append("✅ Auto-corrected: Set all quantization flags to False for 'high_accuracy_validator'")
    
    # Rule 1: ESP_DL_QUANTIZE requires QUANTIZE_MODEL
    if ESP_DL_QUANTIZE and not QUANTIZE_MODEL:
            messages.append("❌ ESP_DL_QUANTIZE=True requires QUANTIZE_MODEL=True")
            # Auto-correct: Enable quantization
            corrected_params['QUANTIZE_MODEL'] = True
            messages.append("✅ Auto-corrected: Set QUANTIZE_MODEL=True")
    
    # Rule 2: USE_QAT requires QUANTIZE_MODEL  
    if USE_QAT and not QUANTIZE_MODEL:
            messages.append("❌ USE_QAT=True requires QUANTIZE_MODEL=True")
            # Auto-correct: Enable quantization
            corrected_params['QUANTIZE_MODEL'] = True
            messages.append("✅ Auto-corrected: Set QUANTIZE_MODEL=True")
    
    # Rule 3: QAT + ESP-DL is valid but needs special handling
    if USE_QAT and ESP_DL_QUANTIZE:
        messages.append("💡 Using QAT + ESP-DL quantization (INT8)")
    
    # Determine the final mode
    if not corrected_params['QUANTIZE_MODEL']:
        mode = "Float32 training & inference"
    else:
        if corrected_params['USE_QAT']:
            if corrected_params['ESP_DL_QUANTIZE']:
                mode = "QAT + INT8 quantization for ESP-DL"
            else:
                mode = "QAT + UINT8 quantization"
        else:
            if corrected_params['ESP_DL_QUANTIZE']:
                mode = "Standard training + INT8 post-quantization (ESP-DL)"
            else:
                mode = "Standard training + UINT8 post-quantization"
    
    messages.append(f"✅ Final mode: {mode}")
    
    # Check if any corrections were made
    needs_correction = any(original_params[k] != corrected_params[k] for k in original_params)
    
    return not needs_correction, corrected_params, "\n".join(messages)

def get_hyperparameter_summary():
    """Return a comprehensive summary of all hyperparameter settings"""
    summary = {
        'model': {
            'architecture': MODEL_ARCHITECTURE,
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
        },
        'training': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'validation_split': VALIDATION_SPLIT,
            'training_percentage': TRAINING_PERCENTAGE,
        },
        'regularization': {
            'l1': L1_REGULARIZATION,
            'l2': L2_REGULARIZATION,
            'dropout': DEFAULT_DROPOUT_RATE,
            'batch_norm': USE_BATCH_NORM,
            'gradient_clipping': USE_GRADIENT_CLIPPING,
        },
        'callbacks': {
            'early_stopping': USE_EARLY_STOPPING,
            'checkpoints': SAVE_CHECKPOINTS,
            'lr_scheduler': USE_LEARNING_RATE_SCHEDULER,
            'tensorboard': USE_TENSORBOARD,
        },
        'data': {
            'augmentation': USE_DATA_AUGMENTATION,
            'sources': [source['name'] for source in DATA_SOURCES],
        },
        'quantization': {
            'qat': USE_QAT,
            'post_training': QUANTIZE_MODEL,
            'esp_dl': ESP_DL_QUANTIZE,
        }
    }
    
    # Add optimizer-specific parameters
    if OPTIMIZER_TYPE == "rmsprop":
        summary['optimizer'].update({
            'rho': RMSPROP_RHO,
            'momentum': RMSPROP_MOMENTUM,
            'epsilon': RMSPROP_EPSILON
        })
    elif OPTIMIZER_TYPE == "adam":
        summary['optimizer'].update({
            'beta_1': ADAM_BETA_1,
            'beta_2': ADAM_BETA_2,
            'epsilon': ADAM_EPSILON,
            'amsgrad': ADAM_AMSGRAD
        })
    elif OPTIMIZER_TYPE == "sgd":
        summary['optimizer'].update({
            'momentum': SGD_MOMENTUM,
            'nesterov': SGD_NESTEROV
        })
    
    # Add learning rate scheduler details
    if USE_LEARNING_RATE_SCHEDULER:
        summary['callbacks']['lr_scheduler_type'] = LR_SCHEDULER_TYPE
        summary['callbacks']['lr_scheduler_patience'] = LR_SCHEDULER_PATIENCE
        summary['callbacks']['lr_scheduler_factor'] = LR_SCHEDULER_FACTOR
    
    return summary

def print_hyperparameter_summary():
    """Print a formatted summary of all hyperparameters"""
    summary = get_hyperparameter_summary()
    
    print("=" * 60)
    print("📊 HYPERPARAMETER CONFIGURATION SUMMARY")
    print("=" * 60)
    
    for category, settings in summary.items():
        print(f"\n{category.upper()}:")

        for key, value in settings.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    
def get_hyperparameter_summary_text():
    """Return hyperparameter summary as formatted text for file export"""
    summary = get_hyperparameter_summary()
    
    lines = []
    lines.append("HYPERPARAMETER CONFIGURATION SUMMARY")
    lines.append("=" * 50)
    
    for category, settings in summary.items():
        lines.append(f"\n{category.upper()}:")
        for key, value in settings.items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"    {sub_key}: {sub_value}")
            else:
                lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)
# ==============================================================================
# INITIALIZATION
# ==============================================================================

# Validate parameters on import
try:
    validate_hyperparameters()
    validate_quantization_parameters()
except Exception as e:
    print(f"❌ Parameter validation failed: {e}")

# Print summary when module is imported
if __name__ != "__main__":
    print_hyperparameter_summary()