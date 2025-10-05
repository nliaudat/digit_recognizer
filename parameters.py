"""
Global parameters for Digit Recognition
"""

# ==============================================================================
# MODEL SELECTION
# ==============================================================================

AVAILABLE_MODELS = [
    "practical_tiny_depthwise",
    "simple_cnn", 
    "dig_class100_s2",
    "original_haverland",
    "esp_optimized_cnn",
    "esp_ultra_light", 
    "esp_high_capacity",
    "esp_quantization_ready",
    "esp_haverland_compatible",
    "esp_quantization_ready_v2",
    "esp_quantization_ready_v2_aggressive"
]

MODEL_ARCHITECTURE = "esp_quantization_ready"  # Options: practical_tiny_depthwise, simple_cnn, dig_class100_s2, original_haverland, esp_optimized_cnn, esp_ultra_light, esp_quantization_ready, esp_high_capacity, esp_haverland_compatible

# ==============================================================================
# MODEL-SPECIFIC PARAMETERS
# ==============================================================================

# For practical_tiny_depthwise
# DEPTHWISE_FILTERS = [16, 32]
# POINTWISE_FILTERS = [16, 32]
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
# INPUT IMAGES 
# ==============================================================================

# Image Parameters
INPUT_WIDTH = 20
INPUT_HEIGHT = 32
INPUT_CHANNELS =1  # 1 for grayscale, 3 for RGB
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
USE_GRAYSCALE = (INPUT_CHANNELS == 1) 

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
TRAINING_PERCENTAGE = 1.0  # Use 100% of available data
VALIDATION_SPLIT = 0.2     # 20% of training for validation

USE_EARLY_STOPPING = True  # Set to False to disable early stopping completely
EARLY_STOPPING_PATIENCE = 30  # Number of epochs with no improvement after which training will be stopped
EARLY_STOPPING_MONITOR = 'val_accuracy'  # Metric to monitor: 'val_loss', 'val_accuracy
EARLY_STOPPING_MIN_DELTA = 0.0001  # Minimum change to qualify as an improvement
RESTORE_BEST_WEIGHTS = True  # Whether to restore weights from the best epoch

# Learning rate scheduler configuration
LR_SCHEDULER_PATIENCE = 3  # Number of epochs with no improvement after which learning rate will be reduced
LR_SCHEDULER_MIN_LR = 1e-7  # Lower bound on the learning rate
LR_SCHEDULER_FACTOR = 0.5   # Factor by which the learning rate will be reduced (new LR = old LR * factor)
LR_SCHEDULER_MONITOR = 'val_loss'  # Metric to monitor for learning rate reduction

# Model Parameters
NB_CLASSES = 10  # [0-9]


# Model Architecture Parameters
DEPTHWISE_FILTERS_1 = 6
POINTWISE_FILTERS_1 = 8
DEPTHWISE_FILTERS_2 = 8
POINTWISE_FILTERS_2 = 10

# ==============================================================================
# GPU CONFIGURATION
# ==============================================================================
USE_GPU = True  # Set to False to force CPU usage
GPU_MEMORY_GROWTH = True  # Gradually allocate GPU memory instead of all at once
GPU_MEMORY_LIMIT = None  # Set specific memory limit in MB, or None for no limit


# ==============================================================================
# DATA SOURCES
# ==============================================================================

# Multiple Data Sources Configuration
DATA_SOURCES = [
    {
        'name': 'meterdigits',
        'type': 'folder_structure',
        'path': 'datasets/meterdigits',
        'weight': 1.0,
    },
    {
        'name': 'meterdigits_augmented',
        'type': 'folder_structure',
        'path': 'datasets/meterdigits_augmented',
        'weight': 0.6,
    },
    # {
        # 'name': 'MNIST',
        # 'type': 'folder_structure',
        # 'path': 'mnist_dataset_folders',
        # 'weight': 0.15,
    # },
        # {
        # 'name': 'MR-AMR Dataset',
        # 'type': 'folder_structure',
        # 'path': 'MR-AMR Dataset',
        # 'weight': 0.15,
    # },
]

# ==============================================================================
# others
# ==============================================================================


# File Paths
MODEL_FILENAME = MODEL_ARCHITECTURE
OUTPUT_DIR = "exported_models"

# TFLite Conversion Parameters
QUANTIZE_MODEL = True
# ESP-DL specific quantization (only applies if QUANTIZE_MODEL = True)
ESP_DL_QUANTIZE = False  # Quantize to int8 range [-128, 127] for ESP-DL
                         # If False: quantize to uint8 range [0, 255] (default)
QUANTIZE_NUM_SAMPLES=1000
TFLITE_FILENAME = f"{MODEL_FILENAME}.tflite"
FLOAT_TFLITE_FILENAME = f"{MODEL_FILENAME}_float.tflite"

# Debug and Logging
VERBOSE = 1
SAVE_TRAINING_PLOTS = True
SHUFFLE_SEED = 42


# ESP-DL specific parameters
# ESP_DL_COMPATIBLE_OPS = [
    # 'CONV_2D',
    # 'DEPTHWISE_CONV_2D', 
    # 'FULLY_CONNECTED',
    # 'MAX_POOL_2D',
    # 'RESHAPE',
    # 'SOFTMAX',
    # 'LOGISTIC',
    # 'RELU'
# ]


# # ESP memory constraints
# ESP32_FLASH_SIZE = 4096  # 4MB typical
# ESP32_PSRAM_SIZE = 0     # No PSRAM on most boards
# ESP32_RAM_SIZE = 320     # 320KB SRAM

# # Target model size for ESP32
# TARGET_MODEL_SIZE_KB = 100  # Aim for <100KB quantized

