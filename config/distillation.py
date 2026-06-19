"""
config/distillation.py — Knowledge distillation hyperparameters.

Used by ``train_distill.py`` and ``utils/retrain_with_teacher.py``.
"""

# ==============================================================================
# Core distillation hyperparameters
# ==============================================================================

DISTILLATION_TEMPERATURE = 4.0          # Softmax temperature for teacher logits (fallback)
DISTILLATION_ALPHA = 0.5                # Weight of distillation loss vs. student CE loss (fallback)

# Class-specific defaults: 100-class needs lower T (sharper teacher) and more
# teacher weight (lower alpha) because the probability mass is spread over
# 100 classes; at T=6.0 the distribution becomes ~uniform and KL≈0.
DISTILLATION_TEMPERATURE_100CLS = 2.5   # Sharper teacher targets for 100 classes
DISTILLATION_TEMPERATURE_10CLS = 4.0    # 10 classes works well at T=4.0
DISTILLATION_ALPHA_100CLS = 0.3         # More teacher weight for 100 classes
DISTILLATION_ALPHA_10CLS = 0.5          # Balanced for 10 classes
DISTILLATION_BETA = 0.3                 # Weight of intermediate feature loss (hints)
DISTILLATION_LOSS_WEIGHT = 0.7          # Overall distillation loss weight
DISTILLATION_STRATEGY = "logit"         # "logit" | "feature" | "hybrid"
DISTILLATION_USE_FOCAL_LOSS = True     # Use DynamicSparseFocalLoss for hard-label branch
DISTILLATION_MODE = "soft"              # "soft" | "hard" | "hybrid"
USE_PROGRESSIVE_DISTILLATION = False    # Dynamic temperature + alpha scheduling

# Training hyperparameters
DISTILLATION_EPOCHS = 150
DISTILLATION_LEARNING_RATE = 0.001
DISTILLATION_BATCH_SIZE = 64
DISTILLATION_VALIDATION_SPLIT = 0.2
DISTILLATION_EARLY_STOPPING_PATIENCE = 20
DISTILLATION_MIN_DELTA = 0.0001

# Advanced options
DISTILLATION_USE_MIXED_PRECISION = False
DISTILLATION_USE_GRADIENT_CLIPPING = False
DISTILLATION_GRADIENT_CLIP_VALUE = 1.0
DISTILLATION_GRADIENT_CLIP_NORM = 1.0
DISTILLATION_USE_LABEL_SMOOTHING = False
DISTILLATION_LABEL_SMOOTHING = 0.0
DISTILLATION_USE_WEIGHT_DECAY = False
DISTILLATION_WEIGHT_DECAY = 0.0
DISTILLATION_USE_DROPOUT = False
DISTILLATION_DROPOUT_RATE = 0.0
DISTILLATION_USE_BATCH_NORM = True
DISTILLATION_BATCH_NORM_MOMENTUM = 0.99
DISTILLATION_BATCH_NORM_EPSILON = 0.001

# Learning rate scheduling
DISTILLATION_USE_LEARNING_RATE_SCHEDULER = True
DISTILLATION_LR_SCHEDULER_TYPE = "reduce_on_plateau"
DISTILLATION_LR_SCHEDULER_PATIENCE = 5
DISTILLATION_LR_SCHEDULER_MIN_LR = 1e-7
DISTILLATION_LR_SCHEDULER_FACTOR = 0.5
DISTILLATION_LR_SCHEDULER_MONITOR = 'val_loss'

# Early stopping
DISTILLATION_USE_EARLY_STOPPING = True
DISTILLATION_EARLY_STOPPING_PATIENCE = 20
DISTILLATION_EARLY_STOPPING_MIN_DELTA = 0.0001
DISTILLATION_RESTORE_BEST_WEIGHTS = True

# Checkpointing
DISTILLATION_SAVE_CHECKPOINTS = True
DISTILLATION_CHECKPOINT_FREQUENCY = 5
DISTILLATION_SAVE_BEST_ONLY = True
DISTILLATION_CHECKPOINT_MONITOR = 'val_accuracy'

# TensorBoard
DISTILLATION_USE_TENSORBOARD = False
DISTILLATION_TENSORBOARD_UPDATE_FREQ = 'epoch'
DISTILLATION_TENSORBOARD_WRITE_GRAPHS = True

# Ensemble
DISTILLATION_USE_MODEL_ENSEMBLE = False
DISTILLATION_ENSEMBLE_MODEL_COUNT = 3

# Augmentation
DISTILLATION_USE_MIXUP = False
DISTILLATION_USE_CUTMIX = False
DISTILLATION_USE_RANDOM_ERASING = False
DISTILLATION_USE_DATA_AUGMENTATION = True
DISTILLATION_AUGMENTATION_ZOOM_RANGE = 0.1
DISTILLATION_AUGMENTATION_ROTATION_RANGE = 1.15
DISTILLATION_AUGMENTATION_CONTRAST_RANGE = 0.1
DISTILLATION_AUGMENTATION_BRIGHTNESS_RANGE = [0.9, 1.1]
DISTILLATION_AUGMENTATION_WIDTH_SHIFT_RANGE = 0.0
DISTILLATION_AUGMENTATION_HEIGHT_SHIFT_RANGE = 0.0
DISTILLATION_AUGMENTATION_HORIZONTAL_FLIP = False
DISTILLATION_AUGMENTATION_VERTICAL_FLIP = False
DISTILLATION_AUGMENTATION_POLARITY_INVERSION = False

# Gradient accumulation
DISTILLATION_USE_GRADIENT_ACCUMULATION = False
DISTILLATION_ACCUMULATION_STEPS = 4

# Stochastic Weight Averaging
DISTILLATION_USE_STOCHASTIC_WEIGHT_AVERAGING = False
DISTILLATION_USE_CYCLICAL_LEARNING_RATE = False
DISTILLATION_USE_LEARNING_RATE_FINDER = False

# ==============================================================================
# Progressive distillation schedule
# ==============================================================================

PROGRESSIVE_FINAL_TEMP_RATIO = 0.5       # final_T = initial_T × this
PROGRESSIVE_MIN_FINAL_TEMP = 1.0         # floor for final_T
PROGRESSIVE_FINAL_ALPHA_SHIFT = 0.3      # final_alpha = initial_alpha + this
PROGRESSIVE_MAX_FINAL_ALPHA = 0.9        # ceiling for final_alpha

# ==============================================================================
# Super-student training (train_super_student.py)
# ==============================================================================

SUPER_STUDENT_INITIAL_TEMP_100CLS = 4.0  # Starting T for 100-class super student
SUPER_STUDENT_INITIAL_ALPHA_100CLS = 0.5 # Starting α for 100-class
SUPER_STUDENT_INITIAL_TEMP_10CLS = 8.0   # Starting T for 10-class super student
SUPER_STUDENT_INITIAL_ALPHA_10CLS = 0.3  # Starting α for 10-class
SUPER_STUDENT_WEIGHT_DECAY = 0.01        # AdamW weight decay
SUPER_STUDENT_LR_PATIENCE = 10           # ReduceLROnPlateau patience
SUPER_STUDENT_LR_FACTOR = 0.5            # LR reduction factor
SUPER_STUDENT_LR_MIN = 1e-7              # Minimum LR
SUPER_STUDENT_EARLY_STOP_PATIENCE = 30   # EarlyStopping patience
SUPER_STUDENT_EARLY_STOP_MIN_DELTA = 5e-4  # EarlyStopping min delta
