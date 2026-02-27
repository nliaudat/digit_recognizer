# models/model_factory.py
import tensorflow as tf
import importlib
import parameters as params

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
    try:
        # For all models, try dynamic import
        module_name = f"models.{model_name}"
        module = importlib.import_module(module_name)
        creator_func_name = f"create_{model_name}"
        return getattr(module, creator_func_name)
    except (ImportError, AttributeError) as e:
        print(f"❌ Failed to import {model_name}: {e}")
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

def compile_model(model, loss_type='sparse', class_weights=None):
    """Compile model with comprehensive hyperparameter support"""
    from utils.dynamic_weighting import FocalLoss
    
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
        # Or tensorflow > 2.10 supports it natively via tf.keras.optimizers.AdamW
        try:
            # Try native Keras first (TF >= 2.11)
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=params.LEARNING_RATE,
                weight_decay=params.ADAMW_WEIGHT_DECAY,
                beta_1=params.ADAMW_BETA_1,
                beta_2=params.ADAMW_BETA_2,
                epsilon=params.ADAMW_EPSILON
            )
            print(f"🔧 Using native AdamW optimizer (weight_decay={params.ADAMW_WEIGHT_DECAY})")
        except AttributeError:
            try:
                import tensorflow_addons as tfa
                optimizer = tfa.optimizers.AdamW(
                    learning_rate=params.LEARNING_RATE,
                    weight_decay=params.ADAMW_WEIGHT_DECAY,
                    beta_1=params.ADAMW_BETA_1,
                    beta_2=params.ADAMW_BETA_2,
                    epsilon=params.ADAMW_EPSILON
                )
                print(f"🔧 Using TFA AdamW optimizer (weight_decay={params.ADAMW_WEIGHT_DECAY})")
            except ImportError:
                print("⚠️  TF < 2.11 and tensorflow-addons not available, falling back to Adam")
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=params.LEARNING_RATE,
                    beta_1=params.ADAM_BETA_1,
                    beta_2=params.ADAM_BETA_2,
                    epsilon=params.ADAM_EPSILON
                )
        except ImportError:
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
        loss = 'sparse_categorical_crossentropy'
        print("🔧 Override: Using sparse categorical crossentropy loss (for other models)")

    # Convert string loss to object form to prevent deprecation warnings
    if params.LABEL_SMOOTHING > 0:
        if loss == "categorical_crossentropy":
            loss = tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=params.LABEL_SMOOTHING
            )
        elif loss == "sparse_categorical_crossentropy":
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                label_smoothing=params.LABEL_SMOOTHING
            )
        print(f"🔧 Applied label smoothing: {params.LABEL_SMOOTHING}")
    else:
        if loss == "categorical_crossentropy":
            loss = tf.keras.losses.CategoricalCrossentropy()
        elif loss == "sparse_categorical_crossentropy":
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # --- Advanced Loss: Focal Loss ---
    if params.USE_FOCAL_LOSS:
        print(f"🔧 Using Focal Loss (gamma={params.FOCAL_GAMMA}, nb_classes={params.NB_CLASSES})")
        loss = FocalLoss(
            gamma=params.FOCAL_GAMMA,
            label_smoothing=params.LABEL_SMOOTHING,
            class_weights=class_weights,
            nb_classes=params.NB_CLASSES,
            name='focal_loss'
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
        from tensorflow.keras.callbacks import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            monitor=params.LR_SCHEDULER_MONITOR,
            factor=params.LR_SCHEDULER_FACTOR,
            patience=params.LR_SCHEDULER_PATIENCE,
            min_lr=params.LR_SCHEDULER_MIN_LR,
            verbose=1
        )
        print(f"🔧 Using ReduceLROnPlateau scheduler (patience={params.LR_SCHEDULER_PATIENCE})")
        
    elif params.LR_SCHEDULER_TYPE == "exponential":
        from tensorflow.keras.optimizers.schedules import ExponentialDecay
        lr_schedule = ExponentialDecay(
            initial_learning_rate=params.LEARNING_RATE,
            decay_steps=params.EXPONENTIAL_DECAY_STEPS,
            decay_rate=params.EXPONENTIAL_DECAY_RATE
        )
        print(f"🔧 Using ExponentialDecay scheduler")
        return lr_schedule
    
    elif params.LR_SCHEDULER_TYPE == "cosine":
        from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
        lr_schedule = CosineDecayRestarts(
            initial_learning_rate=params.LEARNING_RATE,
            first_decay_steps=params.LR_WARMUP_EPOCHS,
            alpha=params.COSINE_DECAY_ALPHA
        )
        print(f"🔧 Using CosineDecayRestarts scheduler")
        return lr_schedule
    
    elif params.LR_SCHEDULER_TYPE == "step":
        # Custom step decay implementation
        def step_decay(epoch):
            initial_lr = params.LEARNING_RATE
            drop = params.STEP_DECAY_GAMMA
            epochs_drop = params.STEP_DECAY_STEP_SIZE
            lr = initial_lr * (drop ** (epoch // epochs_drop))
            return lr
        
        from tensorflow.keras.callbacks import LearningRateScheduler
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
        from tensorflow.keras.callbacks import EarlyStopping
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
        from tensorflow.keras.callbacks import ModelCheckpoint
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
        from tensorflow.keras.callbacks import TensorBoard
        tensorboard = TensorBoard(
            log_dir=f"logs/{params.MODEL_ARCHITECTURE}",
            update_freq=params.TENSORBOARD_UPDATE_FREQ,
            write_graph=params.TENSORBOARD_WRITE_GRAPHS
        )
        callbacks.append(tensorboard)
        print("🔧 Added TensorBoard callback")
    
    return callbacks