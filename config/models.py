"""
config/models.py — Model selection and model-specific parameters.

Defines AVAILABLE_MODELS, MODEL_ARCHITECTURE, and per-model constants.
"""

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
    "digit_recognizer_v7", # 56.0kB / 73.35%
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
    # "digit_recognizer_v20", # GhostNet 100-Class IoT limit-pusher (<1.5MB) with 2D Positional Encoding & Dual Attention
    # "digit_recognizer_v21", # PC/GPU-Only Rotary Positional Encoding & Adaptive Attention limit pusher (>99.5% target)
    # "digit_recognizer_v22", # IoT Spatial MobileNetV2 with 2D Positional Encoding (10-Class RGB <200KB limit)
    "digit_recognizer_v23", # Luminance Grayscale with Fixed Conv2D Weights (Auto convert to grayscale) (Train with RGB images)
    "digit_recognizer_v24", # v23 + Adaptive Contrast Normalization for Light/Dark Backgrounds
    "digit_recognizer_v27", # v24 improvements: adaptive backbone + soft norm + soft binarize + polarity augmentation
    "digit_recognizer_v28", # v27 + hard binarization (STE) + optional DoG edge fusion
    "digit_recognizer_v29", # v28 + 2-channel hybrid processing (hard binarization + soft gradient preservation)
    # "digit_recognizer_v25", # 10 classes only ! v24 + Multi-head Transition-Aware (need to change the transition rule in C++ code)
    # "digit_recognizer_v26", # 10 classes only ! v25 + Learnable Soft-Binarization (threshold trained, sharpness=10, TFLite Micro compatible)
    # "esp_quantization_ready", # ~70kB | Minimal Depthwise CNN for smooth INT8
    # "high_accuracy_validator", # strictly for PC validation (not for ESP32)
    # "super_high_accuracy_validator", # GPU-only deep SE-ResNet validator (2026 SOTA)
    # "mnist_quantization", # 72.2kB / 76.55%
    # "original_haverland", # 228.8kB / 79.10% | baseline
    # ── Distillation: Teacher models (PC-only, large backbone, not for ESP32) ──
    # "digit_recognizer_v30_teacher", # EfficientNetB0 teacher (train first, then distill students)
    # "digit_recognizer_v31_teacher", # ResNet50 teacher (alternative backbone for ensemble distillation)
    # "digit_recognizer_v32_teacher", # Super-Teacher: ensemble distillation from multiple teachers
]

MODEL_ARCHITECTURE = "digit_recognizer_v16" # one of the models in AVAILABLE_MODELS
USE_LOGITS = False # else softmax

# ==============================================================================
# OPTIMIZER CONFIGURATION
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
# OPTIMIZER_TYPE = "rmsprop"        # Best default for cold-start 100cls
# OPTIMIZER_TYPE = "sgd"            # (Optimized for v17 fine-tuning)
# OPTIMIZER_TYPE = "adamw"          # Restore to roll back

# RMSprop parameters
RMSPROP_RHO = 0.9
RMSPROP_MOMENTUM = 0.0
RMSPROP_EPSILON = 1e-07

# Adam parameters
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1e-07
ADAM_AMSGRAD = False

# SGD parameters
SGD_MOMENTUM = 0.9
SGD_NESTEROV = True

# Adagrad parameters
ADAGRAD_INITIAL_ACCUMULATOR = 0.1
ADAGRAD_EPSILON = 1e-07

# AdamW parameters
ADAMW_WEIGHT_DECAY = 0.01
ADAMW_BETA_1 = 0.9
ADAMW_BETA_2 = 0.999
ADAMW_EPSILON = 1e-07

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
ORIGINAL_HAVERLAND_FILTERS = [32, 64, 128]
ORIGINAL_HAVERLAND_DENSE_UNITS = 512
ORIGINAL_HAVERLAND_DROPOUT_RATES = [0.25, 0.25, 0.25, 0.5]

# ==============================================================================
# OUTPUT FUNCTIONS
# ==============================================================================

def get_model_filename():
    return MODEL_ARCHITECTURE

def get_tflite_filename():
    return f"{get_model_filename()}.tflite"

def get_float_tflite_filename():
    return f"{get_model_filename()}_float.tflite"

def get_available_model_names():
    """Return a list of all available model names, including short versions (e.g., 'v16' for 'digit_recognizer_v16')."""
    names = []
    for m in AVAILABLE_MODELS:
        names.append(m)
        if m.startswith("digit_recognizer_"):
            short = m.replace("digit_recognizer_", "")
            if short.endswith("_teacher"):
                short = short.replace("_teacher", "")
            if short not in names:
                names.append(short)
    return sorted(list(set(names)))
