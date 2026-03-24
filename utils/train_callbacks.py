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
        TFLiteCheckpoint(tflite_manager, representative_data, x_train_raw, save_frequency=params.CHECKPOINT_FREQUENCY)
    )
    
    # Learning rate scheduler — respects LR_SCHEDULER_TYPE from parameters.py
    scheduler_type = getattr(params, 'LR_SCHEDULER_TYPE', 'reduce_on_plateau')
    use_dynamic_scheduler = getattr(params, 'USE_DYNAMIC_SCHEDULER', False)

    if use_dynamic_scheduler:
        # Dynamic mode: a proxy object holds the current schedule function.
        # DynamicSchedulerController swaps it at runtime when thresholds are crossed.
        from utils.train_helpers import DynamicLRProxy
        lr_proxy = DynamicLRProxy()
        # Phase-0 scheduler is activated in DynamicSchedulerController.on_train_begin;
        # register the proxy as a LearningRateScheduler now so Keras picks it up.
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(lr_proxy, verbose=0)
        )
        # Also keep ReduceLROnPlateau if the sequence contains it, so it still operates
        # as an additional layer of decay within the reduce_on_plateau phases.
        if 'reduce_on_plateau' in getattr(params, 'LR_SCHEDULER_SEQUENCE', []):
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=params.LR_SCHEDULER_MONITOR,
                    factor=params.LR_SCHEDULER_FACTOR,
                    patience=params.LR_SCHEDULER_PATIENCE,
                    min_lr=params.LR_SCHEDULER_MIN_LR,
                    verbose=1 if debug else 0,
                )
            )
        if debug:
            print(f"🔀 DynamicSchedulerController mode (proxy registered, LR phases: "
                  f"{getattr(params, 'LR_SCHEDULER_SEQUENCE', [])})")

    elif scheduler_type == 'onecycle':
        # OneCycleLR: warm up to peak LR over 30% of total epochs,
        # then cosine-decay to near-zero over the remaining 70%.
        warmup_frac = 0.3
        warmup_steps = max(1, int(total_epochs * warmup_frac))
        decay_steps  = max(1, total_epochs - warmup_steps)
        min_lr = getattr(params, 'COSINE_DECAY_ALPHA', 1e-6)
        peak_lr = params.LEARNING_RATE
        warmup_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=peak_lr * 0.1, decay_steps=warmup_steps,
            end_learning_rate=peak_lr, power=1.0,
        )
        cosine_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=peak_lr, decay_steps=decay_steps, alpha=min_lr / peak_lr,
        )
        def _onecycle_lr(epoch):
            return float(warmup_schedule(epoch) if epoch < warmup_steps
                         else cosine_schedule(epoch - warmup_steps))
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(_onecycle_lr, verbose=0))
        if debug:
            print(f"🔁 OneCycleLR scheduler added (warmup={warmup_steps} epochs → "
                  f"peak {peak_lr:.2e}, then cosine → {min_lr:.0e})")

    elif scheduler_type == 'cosine':
        # CosineDecayRestarts: cyclic warm restarts to escape local minima.
        first_decay_steps = max(1, getattr(params, 'LR_WARMUP_EPOCHS', 10))
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=params.LEARNING_RATE,
            first_decay_steps=first_decay_steps,
            t_mul=2.0, m_mul=0.9,
            alpha=getattr(params, 'COSINE_DECAY_ALPHA', 1e-6),
        )
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: float(lr_schedule(epoch)), verbose=0,
            )
        )
        if debug:
            print(f"🌀 CosineDecayRestarts LR scheduler added "
                  f"(first_decay={first_decay_steps} epochs)")

    elif scheduler_type == 'exponential':
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: params.LEARNING_RATE * (params.EXPONENTIAL_DECAY_RATE ** (epoch // params.EXPONENTIAL_DECAY_STEPS)),
                verbose=0,
            )
        )
        if debug:
            print("📉 Exponential decay LR scheduler added")

    elif scheduler_type == 'step':
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: params.LEARNING_RATE * (params.STEP_DECAY_GAMMA ** (epoch // params.STEP_DECAY_STEP_SIZE)),
                verbose=0,
            )
        )
        if debug:
            print("📉 Step decay LR scheduler added")

    else:  # 'reduce_on_plateau' (default / fallback)
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=params.LR_SCHEDULER_MONITOR,
                factor=params.LR_SCHEDULER_FACTOR,
                patience=params.LR_SCHEDULER_PATIENCE,
                min_lr=params.LR_SCHEDULER_MIN_LR,
                verbose=1 if debug else 0,
            )
        )
        if debug:
            print("📉 ReduceLROnPlateau LR scheduler added")

    # LR Warm-up — ramps LR from a small fraction up to LEARNING_RATE over
    # LR_WARMUP_EPOCHS epochs; deactivates itself once warm-up is done.
    # Must come AFTER ReduceLROnPlateau in the list so that on_epoch_begin
    # (warm-up) and on_epoch_end (plateau scheduler) don't interfere.
    if getattr(params, 'USE_LR_WARMUP', False):
        from utils.train_helpers import LRWarmupCallback
        callbacks.append(
            LRWarmupCallback(
                initial_lr=params.LEARNING_RATE,
                warmup_epochs=getattr(params, 'LR_WARMUP_EPOCHS', 5),
                initial_scale=getattr(params, 'LR_WARMUP_INITIAL_SCALE', 0.1),
            )
        )
        if debug:
            print("🌡️  LRWarmupCallback callback added")
    
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
    is_focal = params.LOSS_TYPE in ["focal_loss", "IntelligentFocalLossController"]
    if is_focal and getattr(params, 'MODEL_ARCHITECTURE', '') != "original_haverland":
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

    # Dynamic Scheduler Controller — must come AFTER FocalLoss controller so
    # both can read val_accuracy from the same logs dict in on_epoch_end.
    if use_dynamic_scheduler:
        from utils.train_helpers import DynamicSchedulerController
        callbacks.append(
            DynamicSchedulerController(lr_proxy=lr_proxy, debug=debug)
        )
        if debug:
            print("🔀 DynamicSchedulerController callback added")

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