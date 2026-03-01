# utils/train_callbacks.py
"""
Centralized callback creation for training.
Handles: early stopping, model checkpointing, LR scheduling, CSV logging, etc.
"""

import os
import tensorflow as tf
from utils.train_checkpoint import TFLiteCheckpoint
from utils.train_progressbar import TQDMProgressBar

import parameters as params

def create_callbacks(output_dir, tflite_manager, representative_data, total_epochs, monitor, debug=False, validation_data=None, x_train_raw=None):
    """Create comprehensive training callbacks with robust error handling"""
    
    callbacks = []
    
    # Early stopping
    if params.USE_EARLY_STOPPING:
        mode = 'max' if 'accuracy' in params.EARLY_STOPPING_MONITOR else 'min'
            
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=params.EARLY_STOPPING_MONITOR,
                patience=params.EARLY_STOPPING_PATIENCE,
                min_delta=params.EARLY_STOPPING_MIN_DELTA,
                restore_best_weights=params.RESTORE_BEST_WEIGHTS,
                mode=mode,
                verbose=1 if debug else 0
            )
        )
    
    # Augmentation Safety Monitor
    if params.USE_DATA_AUGMENTATION and validation_data is not None:
        from utils.augmentation import create_augmentation_safety_monitor
        safety_monitor = create_augmentation_safety_monitor(
            validation_data=validation_data,
            debug=debug
        )
        callbacks.append(safety_monitor)
        if debug:
            print("🔒 AugmentationSafetyMonitor callback added")
    
    # Best model checkpoint
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1 if debug else 0,
            save_weights_only=False,
            save_freq='epoch'
        )
    )

    # TFLite model checkpoint - pass x_train_raw if available
    callbacks.append(
        TFLiteCheckpoint(tflite_manager, representative_data, x_train_raw)
    )
    
    # Learning rate scheduler
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=params.LR_SCHEDULER_MONITOR,
            factor=params.LR_SCHEDULER_FACTOR,
            patience=params.LR_SCHEDULER_PATIENCE,
            min_lr=params.LR_SCHEDULER_MIN_LR,
            verbose=1 if debug else 0
        )
    )
    
    # TQDM progress bar
    callbacks.append(
        TQDMProgressBar(total_epochs, monitor, debug)
    )
    
    # CSV Logger with robust error handling
    csv_path = os.path.join(output_dir, 'training_log.csv')
    os.makedirs(output_dir, exist_ok=True)
    
    # Test directory writability
    try:
        test_file = os.path.join(output_dir, 'test_write.tmp')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        if debug:
            print("✅ Output directory is writable")
    except Exception as e:
        print(f"❌ Output directory not writable: {e}")
        csv_path = 'training_log_fallback.csv'
        if debug:
            print(f"🔄 Using fallback path: {csv_path}")
    
    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=csv_path,
        separator=',',
        append=False
    )
    callbacks.append(csv_logger)
    
    # Intelligent Focal Loss Controller
    if params.LOSS_TYPE in ["focal_loss", "IntelligentFocalLossController"]:
        from utils.train_helpers import IntelligentFocalLossController
        callbacks.append(
            IntelligentFocalLossController(
                accuracy_thresholds=params.FOCAL_ACCURACY_THRESHOLDS,
                gamma_values=params.FOCAL_GAMMA_VALUES,
                alpha=params.FOCAL_ALPHA,
                plateau_patience=params.FOCAL_PLATEAU_PATIENCE,
                plateau_min_delta=params.FOCAL_PLATEAU_MIN_DELTA,
                val_ds=validation_data,
                debug=debug
            )
        )
        if debug:
            print("🎯 IntelligentFocalLossController callback added")

    # Per-Class Accuracy Callback
    if validation_data is not None:
        from utils.train_helpers import PerClassAccuracyCallback
        callbacks.append(
            PerClassAccuracyCallback(
                val_ds=validation_data,
                every_n_epochs=5,
                debug=debug
            )
        )
        if debug:
            print("📊 PerClassAccuracyCallback callback added")

    # TensorBoard Logger
    tb_log_dir = os.path.join(output_dir, 'tensorboard_logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tb_log_dir,
        histogram_freq=1 if debug else 0,
        update_freq='epoch'
    )
    callbacks.append(tensorboard_callback)
    
    # Create checkpoints directory
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    if debug:
        print("🔍 Callbacks created:")
        for i, callback in enumerate(callbacks):
            print(f"   {i+1}. {callback.__class__.__name__}")
    
    return callbacks