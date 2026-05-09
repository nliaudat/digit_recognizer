"""
config/tuner.py — Hyperparameter tuning configuration.

Two sets of tuner parameters:
1. ``tuner.py`` CLI tuner (TUNER_*)
2. Keras Tuner integration (KERAS_TUNER_*) — renamed with prefix to avoid collision
"""

# ==============================================================================
# HYPERPARAMETER TUNING (tuner.py)
# ==============================================================================

TUNER_MAX_TRIALS = 30
TUNER_EPOCHS = 10
TUNER_EARLY_STOPPING_PATIENCE = 3

# Search Space
TUNER_OPTIMIZERS = ['adam', 'rmsprop', 'sgd', 'nadam', 'adamw']
TUNER_LEARNING_RATES = [1e-3, 5e-4, 2e-4, 1e-4]
TUNER_BATCH_SIZES = [32, 64]
TUNER_GAMMAS = [0.0, 1.2, 1.5, 2.0, 3.0, 3.5, 4.5]
TUNER_ALPHAS = [0.25, 0.45]

# Fine-Tune Tuner Search Space (used by: python tuner.py --finetune)
TUNER_FINETUNE_EPOCHS = 15
TUNER_FINETUNE_LRS = [5e-5, 1e-4, 3e-4, 5e-4]
TUNER_FINETUNE_OPTIMIZERS = ['adam', 'rmsprop', 'nadam', 'sgd', 'adamw']
TUNER_LR_FACTORS = [0.3, 0.5, 0.7]
FINETUNE_UNFREEZE_LAST_N = 0

# ==============================================================================
# KERAS TUNER (keras_tuner integration)
# ==============================================================================

USE_KERAS_TUNER = True
KERAS_TUNER_PROJECT_NAME = "digit_recognizer_tuning"
KERAS_TUNER_MAX_TRIALS = 150
KERAS_TUNER_EXECUTIONS_PER_TRIAL = 15
KERAS_TUNER_OBJECTIVE = "val_accuracy"
KERAS_TUNER_NUM_TRIAL = 10
KERAS_TUNER_EPOCHS = 15

# Search Space Configuration
KERAS_TUNER_OPTIMIZERS = ['adam', 'rmsprop', 'sgd', 'nadam', 'adamw']
KERAS_TUNER_LEARNING_RATES = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
KERAS_TUNER_BATCH_SIZES = [16, 32, 64, 128]

# Early Stopping for Tuning
KERAS_TUNER_EARLY_STOPPING_PATIENCE = 10
KERAS_TUNER_MIN_DELTA = 0.001
