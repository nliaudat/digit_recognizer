"""
Global parameters for Digit Recognition
Comprehensive hyperparameter configuration for neural network training
"""

import tensorflow as tf

# ==============================================================================
# MODEL SELECTION
# ==============================================================================

AVAILABLE_MODELS = [
    # "practical_tiny_depthwise",
    # "simple_cnn", 
    # "dig_class100_s2",
    "original_haverland", #203.3	0.9822
    # "esp_optimized_cnn",
    # "esp_ultra_light", 
    # "esp_high_capacity",
    # "esp_quantization_ready",
    # "esp_haverland_compatible",
    # "esp_quantization_ready_v2",
    # "esp_quantization_ready_v2_aggressive",
    # "esp_quantization_ready_v3",
    "mnist_quantization", #64.2	0.9645
    # "digit_recognizer_v1",
    # "simple_cnn_v2",
    # "minimal_cnn",
    # "mobilenet_style",
    # "digit_recognizer_v2",
    "digit_recognizer_v3", #69.4	0.9804
    "digit_recognizer_v4", # 61.4	0.9855
    # "digit_recognizer_v5", #37.4	0.9502
    "digit_recognizer_v6", # 36.5	0.9652
    "digit_recognizer_v7", #46.7	0.9673
    # "digit_recognizer_v8", #not for IOT #396.4	0.9915
    "digit_recognizer_v9", #not for IOT #148.6	0.9907
    # "digit_recognizer_v10", #not for IOT #1392.3	0.9917 (5h30 training)
    # "digit_recognizer_v11", #not for IOT # 1370.8	0.9897
    "digit_recognizer_v12", #406.7	0.9925
]

MODEL_ARCHITECTURE = "digit_recognizer_v4" # one of the models in AVAILABLE_MODELS


# ==============================================================================
# INPUT IMAGES 
# ==============================================================================

# Image Parameters
INPUT_WIDTH = 20
INPUT_HEIGHT = 32
INPUT_CHANNELS = 3  # 1 for grayscale, 3 for RGB
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
USE_GRAYSCALE = (INPUT_CHANNELS == 1) 



# ==============================================================================
# DATA SOURCES
# ==============================================================================

# This is far better to use labels that have been shuffled for training, folder_structure shuffle by batch TF_DATA_SHUFFLE_BUFFER and SHUFFLE_SEED

    # label_file_path = os.path.join(params.DATASET_PATH, 'labels.txt')
    # images_dir = os.path.join(params.DATASET_PATH, 'images')

# Multiple Data Sources Configuration
DATA_SOURCES = [ 
########### class 10 training #############
    {
        'name': 'Tenth-of-step-of-a-meter-digit',
        'type': 'label_file', 
        'labels': 'labels_10_shuffle.txt',  # Optional: specify label file name (default: 'labels.txt' - tab separated)
        'path': 'datasets/Tenth-of-step-of-a-meter-digit', 
        'weight': 1.0, # undersample if weight < 1.0
    },
    {
        'name': 'real_integra_bad_predictions',
        'type': 'label_file', 
        'labels': 'labels_10_shuffle.txt',  
        'path': 'datasets/real_integra_bad_predictions', 
        'weight': 1.0,
    },
    {
        'name': 'real_integra',
        'type': 'label_file', 
        'labels': 'labels_10_shuffle.txt',  
        'path': 'datasets/real_integra', 
        'weight': 0.7,
    },
    {
        'name': 'static_augmentation',
        'type': 'label_file', 
        'labels': 'labels_10_shuffle.txt',  
        'path': 'datasets/static_augmentation', 
        'weight': 0.6,
    },
########### class 100 training #############
    # {
        # 'name': 'Tenth-of-step-of-a-meter-digit',
        # 'type': 'label_file', 
        # 'labels': 'labels_100_shuffle.txt',  # Optional: specify label file name (default: 'labels.txt' - tab separated)
        # 'path': 'datasets/Tenth-of-step-of-a-meter-digit', 
        # 'weight': 1.0, # undersample if weight < 1.0
    # },
    # {
        # 'name': 'real_integra_bad_predictions',
        # 'type': 'label_file', 
        # 'labels': 'labels_100_shuffle.txt',  
        # 'path': 'datasets/real_integra_bad_predictions', 
        # 'weight': 1.0,
    # },
    # {
        # 'name': 'real_integra',
        # 'type': 'label_file', 
        # 'labels': 'labels_100_shuffle.txt',  
        # 'path': 'datasets/real_integra', 
        # 'weight': 0.7,
    # },
    # {
        # 'name': 'static_augmentation',
        # 'type': 'label_file', 
        # 'labels': 'labels_100_shuffle.txt',  
        # 'path': 'datasets/static_augmentation', 
        # 'weight': 0.6,
    # },
########### testing #############
    # {
        # 'name': 'meterdigits_100',
        # 'type': 'folder_structure',
        # 'path': 'datasets/meterdigits_100',
        # 'weight': 1.0,
    # },
    # {
        # 'name': 'meterdigits_100_augmented',
        # 'type': 'folder_structure',
        # 'path': 'datasets/meterdigits_10_augmented',
        # 'weight': 0.3,
    # },
    # {
        # 'name': 'meterdigits_10',
        # 'type': 'folder_structure',
        # 'path': 'datasets/meterdigits_10',
        # 'weight': 1.0,
    # },
    # {
        # 'name': 'meterdigits_10_augmented',
        # 'type': 'folder_structure',
        # 'path': 'datasets/meterdigits_10_augmented',
        # 'weight': 0.3,
    # },
    # {
    #     'name': 'MNIST',
    #     'type': 'folder_structure',
    #     'path': 'mnist_dataset_folders',
    #     'weight': 0.2,
    # },
    # {
    #     'name': 'QMNIST',
    #     'type': 'folder_structure',
    #     'path': 'qmnist_dataset_folders',
    #     'weight': 0.3,
    # },
    # {
    #     'name': 'MR-AMR Dataset',
    #     'type': 'folder_structure',
    #     'path': 'MR-AMR Dataset',
    #     'weight': 0.15,
    # },
]

# ==============================================================================
# MODEL GENERAL PARAMETERS
# ==============================================================================

# Model Parameters
NB_CLASSES = 10  # [0-9]

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

# Data pipeline configuration
USE_TF_DATA_PIPELINE = False
TF_DATA_PARALLEL_CALLS = tf.data.AUTOTUNE
TF_DATA_SHUFFLE_BUFFER = 1000
TF_DATA_PREFETCH_SIZE = tf.data.AUTOTUNE

# File Paths
# MODEL_FILENAME = MODEL_ARCHITECTURE
OUTPUT_DIR = "exported_models"
# OUTPUT_DIR = "exported_models/10cls_GRAY"
# OUTPUT_DIR = "exported_models/10cls_RGB"
# OUTPUT_DIR = "exported_models/100cls_GRAY"
# OUTPUT_DIR = "exported_models/100cls_RGB"
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
OPTIMIZER_TYPE = "rmsprop"  # Options: "rmsprop", "adam", "sgd", "adagrad", "adamw", "nadam"
# OPTIMIZER_TYPE = "nadam"  # for digit_recognizer_v4 100cls RGB

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
# LOSS FUNCTION HYPERPARAMETERS
# ==============================================================================

LOSS_TYPE = "sparse_categorical_crossentropy"  # Options: "sparse_categorical_crossentropy", "categorical_crossentropy"
LABEL_SMOOTHING = 0.0  # Apply label smoothing if > 0

# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================

# Basic Training Parameters
BATCH_SIZE = 32 # 32
EPOCHS = 200
LEARNING_RATE = 0.001
TRAINING_PERCENTAGE = 1.0  # Use 100% of available data
VALIDATION_SPLIT = 0.2     # 20% of training for validation



# ==============================================================================
# LEARNING RATE SCHEDULING
# ==============================================================================

# Learning Rate Scheduler
USE_LEARNING_RATE_SCHEDULER = True
LR_SCHEDULER_TYPE = "reduce_on_plateau"  # Options: "reduce_on_plateau", "exponential", "cosine", "step"

# ReduceLROnPlateau Parameters
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_MIN_LR = 1e-7
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_MONITOR = 'val_loss'

# Exponential Decay Parameters
EXPONENTIAL_DECAY_STEPS = 1000
EXPONENTIAL_DECAY_RATE = 0.96

# Cosine Decay Parameters
COSINE_DECAY_ALPHA = 0.0  # Minimum learning rate as fraction of initial

# Step Decay Parameters
STEP_DECAY_STEP_SIZE = 10
STEP_DECAY_GAMMA = 0.1

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
EARLY_STOPPING_PATIENCE = 30
EARLY_STOPPING_MONITOR = 'val_accuracy'
EARLY_STOPPING_MIN_DELTA = 0.0001
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

# not implemented yet

# AUGMENTATION_SHEAR_RANGE = 0.0
# AUGMENTATION_SATURATION_RANGE = [1.0, 1.0]
# AUGMENTATION_HUE_RANGE = 0.0
# AUGMENTATION_BLUR_RANGE = 0.0
# AUGMENTATION_NOISE_STDDEV = 0.0


# ==============================================================================
# GPU CONFIGURATION
# ==============================================================================

USE_GPU = False  # Set to False to force CPU usage
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
TUNER_OPTIMIZERS = ["adam", "rmsprop", "nadam", "sgd"]  # Limit to best performers ["rmsprop", "adam", "sgd", "adagrad", "adamw", "nadam"]
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
    valid_losses = ["sparse_categorical_crossentropy", "categorical_crossentropy"]
    if LOSS_TYPE not in valid_losses:
        raise ValueError(f"❌ Invalid LOSS_TYPE: {LOSS_TYPE}. Must be one of {valid_losses}")
    
    # Learning rate scheduler validation
    valid_schedulers = ["reduce_on_plateau", "exponential", "cosine", "step"]
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