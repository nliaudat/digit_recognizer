import importlib
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.optimizers.schedules import (
    CosineDecayRestarts, ExponentialDecay
)

# Optional/Third-party Keras-related imports
try:
    import tensorflow_addons as tfa
except ImportError:
    tfa = None

# Project imports
import parameters as params

try:
    from utils.losses import (
        DynamicFocalLoss, DynamicSparseFocalLoss, focal_loss, sparse_focal_loss
    )
except ImportError:
    sparse_focal_loss, focal_loss, DynamicSparseFocalLoss, DynamicFocalLoss = (
        None, None, None, None
    )

def create_model():
    """Factory function to automatically create model based on parameters"""
    model_name = params.MODEL_ARCHITECTURE
    
    print(f"🏗️ Creating model: {model_name}")
    
    # Validate model exists in available models
    if model_name not in params.AVAILABLE_MODELS:
        available_models = get_available_models()
        raise ValueError(
            f"Unknown model architecture: '{model_name}'\n"
            f"Available models: {available_models}\n"
            f"Check parameters.py MODEL_ARCHITECTURE and ensure the model file exists"
        )
    
    # Dynamically import and get the model creator function
    creator = _import_model_creator(model_name)
    
    if not creator:
        available_models = get_available_models()
        raise ValueError(
            f"Failed to import model creator for: '{model_name}'\n"
            f"Available models: {available_models}\n"
            f"Ensure the model file exists and has create_{model_name} function"
        )
    
    # Create the model
    model = creator()
    
    # CRITICAL: Build the model with input shape
    model.build((None,) + params.INPUT_SHAPE)
    print(f"✅ Model '{model_name}' built successfully with input shape: {params.INPUT_SHAPE}")
    
    return model

def _import_model_creator(model_name):
    """Dynamically import a model creator function"""
    # 1. Try primary models directory
    try:
        module_name = f"models.{model_name}"
        module = importlib.import_module(module_name)
        creator_func_name = f"create_{model_name}"
        return getattr(module, creator_func_name)
    except (ImportError, AttributeError):
        # 2. Try rejected models directory if not found in primary
        try:
            module_name = f"models._tested_but_rejected.{model_name}"
            module = importlib.import_module(module_name)
            creator_func_name = f"create_{model_name}"
            return getattr(module, creator_func_name)
        except (ImportError, AttributeError) as e:
            print(f"❌ Failed to import {model_name} from primary or rejected: {e}")
            return None

def model_summary(model):
    """Print model summary"""
    model.build((None,) + params.INPUT_SHAPE)
    model.summary()

def get_available_models():
    """Return list of all available model architectures"""
    available = []
    for model_name in params.AVAILABLE_MODELS:
        creator = _import_model_creator(model_name)
        if creator is not None:
            available.append(model_name)
    return available

def resolve_model_name(name):
    """Map short names (v16) to full available model names (digit_recognizer_v16)."""
    # 1. Try exact match in parameters
    if name in params.AVAILABLE_MODELS:
        return name
    
    # 2. Try common prefixes/suffixes
    lookups = [
        name,
        f"digit_recognizer_{name}",
        f"digit_recognizer_{name}_teacher",
        f"digit_recognizer_{name}_student"
    ]
    
    for l in lookups:
        if l in params.AVAILABLE_MODELS:
            return l

    # 3. Search the filesystem if not in AVAILABLE_MODELS
    # This avoids having to hardcode every single experimental model
    project_root = Path(__file__).resolve().parent.parent
    model_dirs = [
        project_root / "models",
        project_root / "models" / "_tested_but_rejected"
    ]
    
    for model_dir in model_dirs:
        if not model_dir.exists():
            continue
        for f in model_dir.glob("*.py"):
            fname = f.stem
            if fname == "__init__": continue
            
            # Check if this file matches our requested name
            # e.g. "v16" matches "digit_recognizer_v16"
            clean_fname = fname.replace("digit_recognizer_", "").replace("_teacher", "").replace("_student", "")
            if clean_fname == name or fname == name or f"digit_recognizer_{name}" == fname:
                return fname

    # Fallback to name
    return name

def create_model_by_name(model_name, num_classes=None, input_shape=None, **kwargs):
    """
    Flexibly create any registered model by name.
    
    Args:
        model_name:  Short (v16) or full (digit_recognizer_v16) name.
        num_classes: Optional override for NB_CLASSES.
        input_shape: Optional override for INPUT_SHAPE.
        **kwargs:    Passed to the model creator function.
    """
    full_name = resolve_model_name(model_name)
    creator   = _import_model_creator(full_name)
    
    if not creator:
        raise ValueError(f"Unknown model: {model_name} (resolved to {full_name})")
    
    # Temporarily override globals because many models still use params.NB_CLASSES
    _prev_classes = params.NB_CLASSES
    _prev_shape   = params.INPUT_SHAPE
    _prev_channels = params.INPUT_CHANNELS
    
    try:
        if num_classes is not None:
            params.NB_CLASSES = num_classes
        if input_shape is not None:
            params.INPUT_SHAPE = input_shape
            params.INPUT_CHANNELS = input_shape[-1]
        
        params.update_derived_parameters()
        
        print(f"🏗️ Building model: {full_name} | {params.NB_CLASSES} classes | {params.INPUT_SHAPE}")
        
        # Call creator. Handle functions that may or may not accept args.
        try:
            # First try passing num_classes and input_shape explicitly
            model = creator(num_classes=params.NB_CLASSES, input_shape=params.INPUT_SHAPE, **kwargs)
        except TypeError:
            # Fallback to no-arg call (it will use the overridden params.NB_CLASSES)
            try:
                model = creator(**kwargs)
            except TypeError:
                 model = creator()
        
        # Build to ensure input shape is fixed
        model.build((None,) + params.INPUT_SHAPE)
        return model
        
    finally:
        # Restore globals
        params.NB_CLASSES     = _prev_classes
        params.INPUT_SHAPE    = _prev_shape
        params.INPUT_CHANNELS = _prev_channels
        params.update_derived_parameters()

def get_model_info(model_name=None):
    """Get information about a specific model or all models"""
    if model_name:
        if model_name not in get_available_models():
            return f"Model '{model_name}' not found. Available: {get_available_models()}"
        
        creator = _import_model_creator(model_name)
        model = creator()
        model.build((None,) + params.INPUT_SHAPE)
        
        info = {
            "name": model_name,
            "total_parameters": model.count_params(),
            "input_shape": params.INPUT_SHAPE,
            "output_shape": model.output_shape,
            "layers": len(model.layers),
            "available": True
        }
        
        return info
    else:
        # Return info for all models
        all_info = {}
        for model_name in get_available_models():
            all_info[model_name] = get_model_info(model_name)
        return all_info

def _compile_multihead_model(model, optimizer):
    """
    Compile a multi-head transition model (v25, v26).
    Detected automatically when the model has more than one output tensor.

    Expected output names (by Keras layer name):
        digit_probs      → sparse_categorical_crossentropy  (weight 1.0)
        digit_confidence → binary_crossentropy              (weight 0.1)
        transition_prob  → binary_crossentropy              (weight 0.5)
        transition_dir   → binary_crossentropy              (weight 0.5)

    Any output not in the map above gets binary_crossentropy with weight 0.1.
    """
    _LOSS_MAP = {
        'digit_probs':      ('sparse_categorical_crossentropy', 1.0),
        'digit_confidence': ('binary_crossentropy',             0.1),
        'transition_prob':  ('binary_crossentropy',             0.5),
        'transition_dir':   ('binary_crossentropy',             0.5),
    }

    output_names = model.output_names          # e.g. ['digit_probs', 'digit_confidence', ...]
    loss_dict    = {}
    weight_dict  = {}
    for name in output_names:
        loss_fn, weight = _LOSS_MAP.get(name, ('binary_crossentropy', 0.1))
        loss_dict[name]   = loss_fn
        weight_dict[name] = weight

    model.compile(
        optimizer=optimizer,
        loss=loss_dict,
        loss_weights=weight_dict,
        metrics={'digit_probs': ['accuracy']},
    )

    print("✅ Multi-head model compiled:")
    for name in output_names:
        print(f"   - {name}: loss={loss_dict[name]}, weight={weight_dict[name]}")
    return model


def compile_model(model, loss_type='sparse'):
    """Compile model with comprehensive hyperparameter support"""
    
    # Validate hyperparameters first
    params.validate_hyperparameters()
    
    # ==========================================================================
    # OPTIMIZER SELECTION
    # ==========================================================================
    
    optimizer = None
    
    if params.OPTIMIZER_TYPE == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=params.LEARNING_RATE,
            rho=params.RMSPROP_RHO,
            momentum=params.RMSPROP_MOMENTUM,
            epsilon=params.RMSPROP_EPSILON
        )
        print(f"🔧 Using RMSprop optimizer (rho={params.RMSPROP_RHO}, momentum={params.RMSPROP_MOMENTUM})")
        
    elif params.OPTIMIZER_TYPE == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params.LEARNING_RATE,
            beta_1=params.ADAM_BETA_1,
            beta_2=params.ADAM_BETA_2,
            epsilon=params.ADAM_EPSILON,
            amsgrad=params.ADAM_AMSGRAD
        )
        print(f"🔧 Using Adam optimizer (beta1={params.ADAM_BETA_1}, beta2={params.ADAM_BETA_2})")
        
    elif params.OPTIMIZER_TYPE == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=params.LEARNING_RATE,
            momentum=params.SGD_MOMENTUM,
            nesterov=params.SGD_NESTEROV
        )
        print(f"🔧 Using SGD optimizer (momentum={params.SGD_MOMENTUM}, nesterov={params.SGD_NESTEROV})")
        
    elif params.OPTIMIZER_TYPE == "adagrad":
        optimizer = tf.keras.optimizers.Adagrad(
            learning_rate=params.LEARNING_RATE,
            initial_accumulator_value=params.ADAGRAD_INITIAL_ACCUMULATOR,
            epsilon=params.ADAGRAD_EPSILON
        )
        print(f"🔧 Using AdaGrad optimizer")
        
    elif params.OPTIMIZER_TYPE == "nadam":
        optimizer = tf.keras.optimizers.Nadam(
            learning_rate=params.LEARNING_RATE,
            beta_1=params.ADAM_BETA_1,  # Use Adam parameters for consistency
            beta_2=params.ADAM_BETA_2,
            epsilon=params.ADAM_EPSILON
        )
        print(f"🔧 Using Nadam optimizer (beta1={params.ADAM_BETA_1}, beta2={params.ADAM_BETA_2})")
        
    elif params.OPTIMIZER_TYPE == "adamw":
        # Note: AdamW might require tensorflow-addons or newer TF version
        if tfa is not None:
            optimizer = tfa.optimizers.AdamW(
                learning_rate=params.LEARNING_RATE,
                weight_decay=params.ADAMW_WEIGHT_DECAY,
                beta_1=params.ADAMW_BETA_1,
                beta_2=params.ADAMW_BETA_2,
                epsilon=params.ADAMW_EPSILON
            )
            print(f"🔧 Using AdamW optimizer (weight_decay={params.ADAMW_WEIGHT_DECAY})")
        else:
            print("⚠️  tensorflow-addons not available, falling back to Adam")
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=params.LEARNING_RATE,
                beta_1=params.ADAM_BETA_1,
                beta_2=params.ADAM_BETA_2,
                epsilon=params.ADAM_EPSILON
            )
    else:
        raise ValueError(f"❌ Unsupported optimizer type: {params.OPTIMIZER_TYPE}")
    
    # Apply gradient clipping if enabled
    if params.USE_GRADIENT_CLIPPING:
        if params.GRADIENT_CLIP_VALUE is not None:
            optimizer = tf.keras.optimizers.get({
                'class_name': type(optimizer).__name__,
                'config': {**optimizer.get_config(), 'clipvalue': params.GRADIENT_CLIP_VALUE}
            })
            print(f"🔧 Applied gradient clipping by value: {params.GRADIENT_CLIP_VALUE}")
        elif params.GRADIENT_CLIP_NORM is not None:
            optimizer = tf.keras.optimizers.get({
                'class_name': type(optimizer).__name__,
                'config': {**optimizer.get_config(), 'clipnorm': params.GRADIENT_CLIP_NORM}
            })
            print(f"🔧 Applied gradient clipping by norm: {params.GRADIENT_CLIP_NORM}")
    
    # ==========================================================================
    # LOSS FUNCTION SELECTION
    # ==========================================================================
    
    # Start with the configured loss type
    loss = params.LOSS_TYPE
    
    # Override with model-specific loss type if provided (backward compatibility)
    if loss_type == 'categorical':
        loss = 'categorical_crossentropy'
        print("🔧 Override: Using categorical crossentropy loss (for Haverland model)")
    elif loss_type == 'sparse':
        # Don't override if it's already a specialized loss like focal_loss
        if loss not in ["focal_loss", "IntelligentFocalLossController"]:
            loss = 'sparse_categorical_crossentropy'
            print("🔧 Override: Using sparse categorical crossentropy loss")

    # Handle Focal Loss
    if loss in ["focal_loss", "IntelligentFocalLossController"]:
        
        if loss == "IntelligentFocalLossController":
             # We start with gamma=0.0 (equivalent to CrossEntropy)
             # and let the controller update gamma/alpha dynamically
             print(f"🔧 IntelligentFocalLossController active: Using Dynamic Focal Loss (starting with γ=0.0, from_logits={params.USE_LOGITS})")
             if params.MODEL_ARCHITECTURE == "original_haverland":
                 loss = DynamicFocalLoss(gamma=0.0, alpha=params.FOCAL_ALPHA, from_logits=params.USE_LOGITS)
             else:
                 loss = DynamicSparseFocalLoss(gamma=0.0, alpha=params.FOCAL_ALPHA, from_logits=params.USE_LOGITS)
        else:
            if params.MODEL_ARCHITECTURE == "original_haverland":
                loss = focal_loss(gamma=params.FOCAL_GAMMA, alpha=params.FOCAL_ALPHA, from_logits=params.USE_LOGITS)
                print(f"🔧 Using focal_loss (one-hot) with gamma={params.FOCAL_GAMMA}, from_logits={params.USE_LOGITS}")
            else:
                loss = sparse_focal_loss(gamma=params.FOCAL_GAMMA, alpha=params.FOCAL_ALPHA, from_logits=params.USE_LOGITS)
                print(f"🔧 Using sparse_focal_loss with gamma={params.FOCAL_GAMMA}, from_logits={params.USE_LOGITS}")
    
    # Handle standard crossentropy with label smoothing
    elif params.LABEL_SMOOTHING > 0:
        if loss == "categorical_crossentropy":
            loss = tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=params.LABEL_SMOOTHING,
                from_logits=params.USE_LOGITS
            )
        elif loss == "sparse_categorical_crossentropy":
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                label_smoothing=params.LABEL_SMOOTHING,
                from_logits=params.USE_LOGITS
            )
        print(f"🔧 Applied label smoothing: {params.LABEL_SMOOTHING}")
    else:
        if loss == "categorical_crossentropy":
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=params.USE_LOGITS)
        elif loss == "sparse_categorical_crossentropy":
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=params.USE_LOGITS
            )
    
    # ==========================================================================
    # METRICS CONFIGURATION
    # ==========================================================================
    
    metrics = ['accuracy']
    
    # Add additional metrics for binary classification
    if params.NB_CLASSES == 2:
        metrics.extend(['precision', 'recall'])
        print("🔧 Added precision and recall metrics for binary classification")
    
    # ==========================================================================
    # COMPILE MODEL
    # ==========================================================================
    
    # --- Multi-head detection (v25, v26 and future transition models) ---
    if hasattr(model, 'outputs') and len(model.outputs) > 1:
        print("🔀 Multi-head model detected — using per-output loss compilation")
        _compile_multihead_model(model, optimizer)
    else:
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    print(f"✅ Model compiled successfully with:")
    print(f"   - Optimizer: {params.OPTIMIZER_TYPE}")
    print(f"   - Loss: {loss}")
    print(f"   - Learning Rate: {params.LEARNING_RATE}")
    print(f"   - Metrics: {metrics}")
    if params.LABEL_SMOOTHING > 0:
        print(f"   - Label Smoothing: {params.LABEL_SMOOTHING}")
    if params.USE_GRADIENT_CLIPPING:
        print(f"   - Gradient Clipping: Enabled")
    
    return model

def get_hyperparameter_summary():
    """Return a summary of all current hyperparameter settings"""
    return params.get_hyperparameter_summary()

def print_hyperparameter_summary():
    """Print a formatted summary of all hyperparameters"""
    params.print_hyperparameter_summary()

def create_learning_rate_scheduler():
    """Create learning rate scheduler based on configuration"""
    if not params.USE_LEARNING_RATE_SCHEDULER:
        return None
    
    if params.LR_SCHEDULER_TYPE == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            monitor=params.LR_SCHEDULER_MONITOR,
            factor=params.LR_SCHEDULER_FACTOR,
            patience=params.LR_SCHEDULER_PATIENCE,
            min_lr=params.LR_SCHEDULER_MIN_LR,
            verbose=1
        )
        print(f"🔧 Using ReduceLROnPlateau scheduler (patience={params.LR_SCHEDULER_PATIENCE})")
        
    elif params.LR_SCHEDULER_TYPE == "exponential":
        lr_schedule = ExponentialDecay(
            initial_learning_rate=params.LEARNING_RATE,
            decay_steps=params.EXPONENTIAL_DECAY_STEPS,
            decay_rate=params.EXPONENTIAL_DECAY_RATE
        )
        print(f"🔧 Using ExponentialDecay scheduler")
        return lr_schedule
    
    elif params.LR_SCHEDULER_TYPE == "cosine":
        lr_schedule = CosineDecayRestarts(
            initial_learning_rate=params.LEARNING_RATE,
            first_decay_steps=1000,
            alpha=params.COSINE_DECAY_ALPHA
        )
        print(f"🔧 Using CosineDecayRestarts scheduler")
        return lr_schedule
    
    elif params.LR_SCHEDULER_TYPE == "onecycle":
        # OneCycleLR: warm up to peak LR over 30% of total epochs,
        # then cosine-decay to near-zero over the remaining 70%.
        total_epochs = params.EPOCHS
        warmup_frac = 0.3
        warmup_steps = max(1, int(total_epochs * warmup_frac))
        decay_steps  = max(1, total_epochs - warmup_steps)

        min_lr = getattr(params, 'COSINE_DECAY_ALPHA', 1e-6)
        peak_lr = params.LEARNING_RATE

        # Phase 1: linear warm-up
        warmup_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=peak_lr * 0.1,
            decay_steps=warmup_steps,
            end_learning_rate=peak_lr,
            power=1.0,
        )
        # Phase 2: cosine decay from peak to floor
        cosine_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=peak_lr,
            decay_steps=decay_steps,
            alpha=min_lr / peak_lr,
        )

        def _onecycle_lr(epoch):
            if epoch < warmup_steps:
                return float(warmup_schedule(epoch))
            else:
                return float(cosine_schedule(epoch - warmup_steps))

        scheduler = LearningRateScheduler(_onecycle_lr, verbose=1)
        print(f"🔧 Using OneCycleLR scheduler (warmup={warmup_steps} epochs)")

    elif params.LR_SCHEDULER_TYPE == "step":
        # Custom step decay implementation
        def step_decay(epoch):
            initial_lr = params.LEARNING_RATE
            drop = params.STEP_DECAY_GAMMA
            epochs_drop = params.STEP_DECAY_STEP_SIZE
            lr = initial_lr * (drop ** (epoch // epochs_drop))
            return lr
        
        scheduler = LearningRateScheduler(step_decay, verbose=1)
        print(f"🔧 Using Step Decay scheduler (step_size={params.STEP_DECAY_STEP_SIZE})")
    
    else:
        print(f"⚠️  Unknown scheduler type: {params.LR_SCHEDULER_TYPE}")
        return None
    
    return scheduler

def get_initializer():
    """Get weight initializer based on configuration"""
    if params.WEIGHT_INITIALIZER == "glorot_uniform":
        return tf.keras.initializers.GlorotUniform()
    elif params.WEIGHT_INITIALIZER == "he_normal":
        return tf.keras.initializers.HeNormal()
    elif params.WEIGHT_INITIALIZER == "he_uniform":
        return tf.keras.initializers.HeUniform()
    elif params.WEIGHT_INITIALIZER == "lecun_normal":
        return tf.keras.initializers.LecunNormal()
    else:
        print(f"⚠️  Unknown initializer: {params.WEIGHT_INITIALIZER}, using he_normal")
        return tf.keras.initializers.HeNormal()

def get_regularizer():
    """Get regularizer based on configuration"""
    if params.L1_REGULARIZATION > 0 and params.L2_REGULARIZATION > 0:
        return tf.keras.regularizers.l1_l2(
            l1=params.L1_REGULARIZATION, 
            l2=params.L2_REGULARIZATION
        )
    elif params.L1_REGULARIZATION > 0:
        return tf.keras.regularizers.l1(params.L1_REGULARIZATION)
    elif params.L2_REGULARIZATION > 0:
        return tf.keras.regularizers.l2(params.L2_REGULARIZATION)
    else:
        return None

def get_training_callbacks():
    """Create training callbacks based on configuration"""
    callbacks = []
    
    # Early Stopping
    if params.USE_EARLY_STOPPING:
        early_stopping = EarlyStopping(
            monitor=params.EARLY_STOPPING_MONITOR,
            patience=params.EARLY_STOPPING_PATIENCE,
            min_delta=params.EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=params.RESTORE_BEST_WEIGHTS,
            verbose=1
        )
        callbacks.append(early_stopping)
        print(f"🔧 Added EarlyStopping (patience={params.EARLY_STOPPING_PATIENCE})")
    
    # Model Checkpoint
    if params.SAVE_CHECKPOINTS:
        checkpoint = ModelCheckpoint(
            filepath=f"checkpoints/{params.MODEL_ARCHITECTURE}_epoch_{{epoch:02d}}.h5",
            monitor=params.CHECKPOINT_MONITOR,
            save_best_only=params.SAVE_BEST_ONLY,
            save_weights_only=False,
            verbose=1,
            period=params.CHECKPOINT_FREQUENCY
        )
        callbacks.append(checkpoint)
        print(f"🔧 Added ModelCheckpoint (frequency={params.CHECKPOINT_FREQUENCY})")
    
    # Learning Rate Scheduler
    lr_scheduler = create_learning_rate_scheduler()
    if lr_scheduler:
        callbacks.append(lr_scheduler)
    
    # TensorBoard
    if params.USE_TENSORBOARD:
        tensorboard = TensorBoard(
            log_dir=f"logs/{params.MODEL_ARCHITECTURE}",
            update_freq=params.TENSORBOARD_UPDATE_FREQ,
            write_graph=params.TENSORBOARD_WRITE_GRAPHS
        )
        callbacks.append(tensorboard)
        print("🔧 Added TensorBoard callback")
    
    return callbacks