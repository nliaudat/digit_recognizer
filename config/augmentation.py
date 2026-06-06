"""
config/augmentation.py — Data augmentation and GPU configuration.
"""

# ==============================================================================
# DATA AUGMENTATION
# ==============================================================================

## all augmentation is done in datasets/tools/generate_augmented_dataset.py for efficiency and caching
## The important part is image invertion, it permit to train the model for both light-on-dark and dark-on-light digits

USE_DATA_AUGMENTATION = True
AUGMENTATION_PROBABILITY = 0.3          # Fraction of images augmented per epoch (0.0-1.0)
                                          # Reduces overhead while keeping epoch-to-epoch
                                          # re-randomization benefit. Static datasets (70k)
                                          # handle the heavy diversity; inline adds variation.
AUGMENTATION_ZOOM_RANGE = 0.1           # ±10% zoom
AUGMENTATION_ROTATION_RANGE = 3.0       # ±3 degrees (was ±1.15° — increased parity with static aug's ±5°)
AUGMENTATION_CONTRAST_RANGE = 0.1       # ±10% contrast
AUGMENTATION_BRIGHTNESS_RANGE = [0.9, 1.1]  # ±10% brightness
# Enabled — small shift for positional robustness (was 0.0)
AUGMENTATION_WIDTH_SHIFT_RANGE = 0.03   # ±3%
AUGMENTATION_HEIGHT_SHIFT_RANGE = 0.03  # ±3%
AUGMENTATION_HORIZONTAL_FLIP = False
AUGMENTATION_VERTICAL_FLIP = False
AUGMENTATION_POLARITY_INVERSION = False # done in static augmentation for efficiency and caching

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
