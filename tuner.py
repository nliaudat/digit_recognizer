# tuner.py
import tensorflow as tf
import keras_tuner as kt
import parameters as params
from models import create_model
import os
from datetime import datetime

def build_model(hp):
    """Build model with hyperparameters for tuning - TF-Keras compatible"""
    
    # Tune optimizer type, learning rate, and batch size
    optimizer_type = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd', 'nadam', 'adagrad', 'adadelta'])
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    batch_size = hp.Choice('batch_size', values=[32, 64, 128])
    
    print(f"🏗️ Testing optimizer: {optimizer_type}, learning_rate: {learning_rate}, batch_size: {batch_size} for {params.MODEL_ARCHITECTURE}")
    
    # Create model with current architecture
    model = create_model()
    
    # Select optimizer based on tuned choice
    if optimizer_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = 'sparse_categorical_crossentropy'
    elif optimizer_type == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        loss = 'categorical_crossentropy'
    elif optimizer_type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        loss = 'sparse_categorical_crossentropy'
    elif optimizer_type == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
        loss = 'sparse_categorical_crossentropy'
    elif optimizer_type == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        loss = 'sparse_categorical_crossentropy'
    elif optimizer_type == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
        loss = 'sparse_categorical_crossentropy'
    else:
        # Default fallback
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = 'sparse_categorical_crossentropy'
    
    # Compile with tuned optimizer and learning rate
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    # Ensure the model is built
    if not model.built:
        model.build(input_shape=(None,) + params.INPUT_SHAPE)
    
    return model

class TFTuner(kt.RandomSearch):
    """Custom tuner that handles TF-Keras models properly"""
    
    def _build_model(self, hp):
        """Override build model to ensure TF-Keras compatibility"""
        try:
            model = build_model(hp)
            # Force model to be built and return as is
            if not model.built:
                model.build(input_shape=(None,) + params.INPUT_SHAPE)
            return model
        except Exception as e:
            print(f"❌ Model building failed: {e}")
            raise

def run_architecture_tuning(x_train, y_train, x_val, y_val, num_trials=10, debug=False):
    """Tune hyperparameters for the current MODEL_ARCHITECTURE including optimizer type"""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(params.OUTPUT_DIR, f"tuning_{params.MODEL_ARCHITECTURE}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting Hyperparameter Tuning")
    print("=" * 60)
    print(f"🎯 Tuning model: {params.MODEL_ARCHITECTURE}")
    print(f"🔬 Trials: {num_trials}")
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Create custom tuner that handles TF-Keras
        tuner = TFTuner(
            hypermodel=build_model,
            objective='val_accuracy',
            max_trials=num_trials,
            executions_per_trial=1,
            directory=output_dir,
            project_name=f'tune_{params.MODEL_ARCHITECTURE}',
            overwrite=True
        )
        
        # Display search space
        print("\n🔍 Search space summary:")
        print(f"   Optimizers: [adam, rmsprop, sgd, nadam, adagrad, adadelta]")
        print(f"   Learning rate: [0.01, 0.001, 0.0001]")
        print(f"   Batch size: [32, 64, 128]")
        print(f"   Architecture: FIXED ({params.MODEL_ARCHITECTURE})")
        print(f"   Total combinations: 6 optimizers × 3 learning rates × 3 batch sizes = 54")
        print(f"   Testing {num_trials} random combinations")
        
        # Run search
        print("\n🎯 Starting hyperparameter search...")
        tuner.search(
            x_train, y_train,
            epochs=10,  # Reduced epochs for faster tuning
            validation_data=(x_val, y_val),
            batch_size=32,  # Fixed during search for fairness
            verbose=1 if debug else 0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=3,
                    restore_best_weights=True,
                    verbose=1 if debug else 0
                )
            ]
        )
        
        # Get best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_optimizer = best_hps.get('optimizer')
        best_lr = best_hps.get('learning_rate')
        best_batch_size = best_hps.get('batch_size')
        
        print(f"\n🏆 Best Hyperparameters Found for {params.MODEL_ARCHITECTURE}:")
        print("=" * 50)
        print(f"  Optimizer: {best_optimizer}")
        print(f"  Learning rate: {best_lr}")
        print(f"  Batch size: {best_batch_size}")
        
        # Build best model with the optimal optimizer
        print("🔧 Building best model with optimized hyperparameters...")
        
        best_model = create_model()
        
        # Recreate the best optimizer configuration
        if best_optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=best_lr)
            loss = 'sparse_categorical_crossentropy'
        elif best_optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=best_lr)
            loss = 'categorical_crossentropy'
        elif best_optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=best_lr, momentum=0.9)
            loss = 'sparse_categorical_crossentropy'
        elif best_optimizer == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=best_lr)
            loss = 'sparse_categorical_crossentropy'
        elif best_optimizer == 'adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=best_lr)
            loss = 'sparse_categorical_crossentropy'
        elif best_optimizer == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=best_lr)
            loss = 'sparse_categorical_crossentropy'
        
        best_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        # Train best model with full epochs
        print("🚀 Training best model with full epochs...")
        history = best_model.fit(
            x_train, y_train,
            epochs=min(50, params.EPOCHS),  # Use shorter training for tuning
            batch_size=best_batch_size,
            validation_data=(x_val, y_val),
            verbose=1 if debug else 0,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    verbose=1 if debug else 0
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1 if debug else 0
                )
            ]
        )
        
        # Save results
        results_path = os.path.join(output_dir, "tuning_results.txt")
        with open(results_path, 'w') as f:
            f.write(f"Hyperparameter Tuning Results - {params.MODEL_ARCHITECTURE}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {params.MODEL_ARCHITECTURE}\n")
            f.write(f"Trials: {num_trials}\n")
            f.write(f"Best optimizer: {best_optimizer}\n")
            f.write(f"Best learning rate: {best_lr}\n")
            f.write(f"Best batch size: {best_batch_size}\n")
            f.write(f"Final val accuracy: {history.history['val_accuracy'][-1]:.4f}\n\n")
            
            # Save all trials
            trials = tuner.oracle.get_best_trials(num_trials=num_trials)
            f.write("All Trials:\n")
            for i, trial in enumerate(trials):
                f.write(f"Trial {i+1}: Optimizer={trial.hyperparameters.get('optimizer')}, "
                       f"LR={trial.hyperparameters.get('learning_rate')}, "
                       f"BS={trial.hyperparameters.get('batch_size')}, "
                       f"Score={trial.score:.4f}\n")
        
        print(f"💾 Tuning results saved to: {results_path}")
        
        return best_model, best_hps, history, tuner
        
    except Exception as e:
        print(f"❌ Hyperparameter tuning failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return None, None, None, None

def run_simple_tuning(x_train, y_train, x_val, y_val, num_trials=5, debug=False):
    """Simple tuning focusing on optimizer and learning rate - TF-Keras compatible"""
    
    def build_quick_model(hp):
        optimizer_type = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd', 'nadam'])
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
        
        print(f"🔬 Testing optimizer: {optimizer_type}, learning_rate: {learning_rate} for {params.MODEL_ARCHITECTURE}")
        
        model = create_model()
        
        # Select optimizer based on tuned choice
        if optimizer_type == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            loss = 'sparse_categorical_crossentropy'
        elif optimizer_type == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            loss = 'categorical_crossentropy'
        elif optimizer_type == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
            loss = 'sparse_categorical_crossentropy'
        elif optimizer_type == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
            loss = 'sparse_categorical_crossentropy'
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        # Ensure model is built
        if not model.built:
            model.build(input_shape=(None,) + params.INPUT_SHAPE)
        
        return model
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(params.OUTPUT_DIR, f"quick_tune_{params.MODEL_ARCHITECTURE}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting Quick Optimizer & Learning Rate Tuning")
    print(f"🎯 Model: {params.MODEL_ARCHITECTURE}")
    print(f"🔬 Trials: {num_trials}")
    
    try:
        # Use the custom tuner
        tuner = TFTuner(
            build_quick_model,
            objective='val_accuracy',
            max_trials=num_trials,
            directory=output_dir,
            project_name='quick_tune'
        )
        
        tuner.search(
            x_train, y_train,
            epochs=8,  # Very short epochs for quick tuning
            validation_data=(x_val, y_val),
            batch_size=32,
            verbose=1 if debug else 0
        )
        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_optimizer = best_hps.get('optimizer')
        best_lr = best_hps.get('learning_rate')
        
        print(f"🏆 Best optimizer: {best_optimizer}")
        print(f"🏆 Best learning rate: {best_lr}")
        
        return best_optimizer, best_lr
        
    except Exception as e:
        print(f"❌ Quick tuning failed: {e}")
        return None, None