"""
config/augmentation.py — Data augmentation and GPU configuration.
"""

# ==============================================================================
# DATA AUGMENTATION
# ==============================================================================

## all augmentation is done in datasets/tools/generate_augmented_dataset.py for efficiency and caching
## The important part is image invertion, it permit to train the model for both light-on-dark and dark-on-light digits

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
AUGMENTATION_POLARITY_INVERSION = False

# Advanced Augmentations (used by super_high_accuracy_validator)
USE_MIXUP = False
USE_CUTMIX = False
USE_RANDOM_ERASING = False

# ==============================================================================
# GPU CONFIGURATION
# ==============================================================================

USE_GPU = True
GPU_MEMORY_GROWTH = True
GPU_MEMORY_LIMIT = None

# ==============================================================================
# ADVANCED TRAINING CONFIGURATION
# ==============================================================================

USE_MIXED_PRECISION = False
USE_GRADIENT_ACCUMULATION = False
ACCUMULATION_STEPS = 4

USE_LEARNING_RATE_FINDER = False
USE_STOCHASTIC_WEIGHT_AVERAGING = False
USE_CYCLICAL_LEARNING_RATE = False

USE_MODEL_ENSEMBLE = False
ENSEMBLE_MODEL_COUNT = 3
