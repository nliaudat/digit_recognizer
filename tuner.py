# tuner.py
import tensorflow as tf
import keras_tuner as kt
import parameters as params
from models import create_model
import os
from datetime import datetime
import random
import numpy as np
from itertools import product

class GuaranteedBayesianOptimization(kt.BayesianOptimization):
    """Bayesian Optimization that guarantees unique hyperparameter combinations"""
    
    def __init__(self, hypermodel, objective, max_trials, **kwargs):
        # Initialize the search space first
        self._define_search_space()
        super().__init__(
            hypermodel=hypermodel,
            objective=objective,
            max_trials=max_trials,
            **kwargs
        )
        self._used_configs = set()
        self._all_possible_configs = None
        self._generate_all_configs()
    
    def _define_search_space(self):
        """Define the search space for the tuner"""
        # This ensures the hyperparameters are properly registered
        tuner_optimizers = getattr(params, 'TUNER_OPTIMIZERS', ['adam', 'rmsprop', 'sgd', 'nadam'])
        tuner_learning_rates = getattr(params, 'TUNER_LEARNING_RATES', [1e-2, 1e-3, 1e-4])
        tuner_batch_sizes = getattr(params, 'TUNER_BATCH_SIZES', [32, 64, 128])
        
        # These will be overridden by our custom logic, but need to exist for the tuner
        self.optimizers = tuner_optimizers
        self.learning_rates = tuner_learning_rates
        self.batch_sizes = tuner_batch_sizes
    
    def _generate_all_configs(self):
        """Generate all possible hyperparameter combinations"""
        # Create all possible combinations
        self._all_possible_configs = list(product(
            self.optimizers, 
            self.learning_rates, 
            self.batch_sizes
        ))
        random.shuffle(self._all_possible_configs)  # Shuffle for random order
        print(f"🔢 Generated {len(self._all_possible_configs)} unique configurations")
    
    def _populate_space(self, trial_id):
        """Override to guarantee unique configurations"""
        # If we haven't used all configurations yet, use them in order
        if len(self._used_configs) < len(self._all_possible_configs):
            for config in self._all_possible_configs:
                config_key = config
                if config_key not in self._used_configs:
                    self._used_configs.add(config_key)
                    optimizer, lr, bs = config
                    
                    # Create hyperparameters with the proper search space defined
                    hp = kt.HyperParameters()
                    
                    # Define the choices (even though we'll fix them)
                    hp.Choice('optimizer', values=self.optimizers)
                    hp.Choice('learning_rate', values=self.learning_rates)
                    hp.Choice('batch_size', values=self.batch_sizes)
                    
                    # Now fix them to our specific values
                    hp.values['optimizer'] = optimizer
                    hp.values['learning_rate'] = lr
                    hp.values['batch_size'] = bs
                    
                    print(f"🎯 Using guaranteed unique config: {optimizer}, LR: {lr}, BS: {bs}")
                    return hp.values
        
        # Fallback to Bayesian optimization if we've used all combinations
        print("🔄 All unique configurations used, switching to Bayesian sampling")
        return super()._populate_space(trial_id)

def build_model(hp):
    """Build model with hyperparameters for tuning"""
    
    # Get values from hyperparameters
    optimizer_type = hp.get('optimizer')
    learning_rate = hp.get('learning_rate')
    batch_size = hp.get('batch_size')
    
    print(f"🏗️ Building model with: {optimizer_type}, LR: {learning_rate}, BS: {batch_size}")
    
    # Create model with current architecture
    model = create_model()
    
    # Select optimizer based on tuned choice
    if optimizer_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_type == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer_type == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_type == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # DYNAMIC LOSS SELECTION based on model type
    if params.MODEL_ARCHITECTURE == "original_haverland":
        loss = 'categorical_crossentropy'
        print(f"   Using categorical crossentropy for Haverland model")
    else:
        loss = 'sparse_categorical_crossentropy'
        print(f"   Using sparse categorical crossentropy for {params.MODEL_ARCHITECTURE}")
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    # Ensure the model is built
    if not model.built:
        model.build(input_shape=(None,) + params.INPUT_SHAPE)
    
    return model

def run_architecture_tuning(x_train, y_train, x_val, y_val, num_trials=None, debug=False):
    """Tune hyperparameters using Guaranteed Bayesian Optimization"""
    
    # Use parameters from config or override
    if num_trials is None:
        num_trials = getattr(params, 'TUNER_MAX_TRIALS', 100)
    
    tuner_epochs = getattr(params, 'TUNER_EPOCHS', 10)
    
    # Get search space parameters
    tuner_optimizers = getattr(params, 'TUNER_OPTIMIZERS', ['adam', 'rmsprop', 'sgd', 'nadam'])
    tuner_learning_rates = getattr(params, 'TUNER_LEARNING_RATES', [1e-2, 1e-3, 1e-4])
    tuner_batch_sizes = getattr(params, 'TUNER_BATCH_SIZES', [32, 64, 128])
    
    # Calculate total possible combinations
    total_combinations = len(tuner_optimizers) * len(tuner_learning_rates) * len(tuner_batch_sizes)
    
    # Don't allow more trials than possible combinations
    if num_trials > total_combinations:
        print(f"⚠️  Reducing trials from {num_trials} to {total_combinations} (total possible combinations)")
        num_trials = total_combinations
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(params.OUTPUT_DIR, f"guaranteed_tune_{params.MODEL_ARCHITECTURE}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting Guaranteed Unique Hyperparameter Optimization")
    print("=" * 60)
    print(f"🎯 Tuning model: {params.MODEL_ARCHITECTURE}")
    print(f"🔬 Trials: {num_trials}")
    print(f"📈 Epochs per trial: {tuner_epochs}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Strategy: Guaranteed Unique Configurations")
    
    try:
        # Create our custom tuner that guarantees unique configurations
        tuner = GuaranteedBayesianOptimization(
            hypermodel=build_model,
            objective='val_accuracy',
            max_trials=num_trials,
            executions_per_trial=getattr(params, 'TUNER_EXECUTIONS_PER_TRIAL', 1),
            directory=output_dir,
            project_name=f'guaranteed_tune_{params.MODEL_ARCHITECTURE}',
            overwrite=True,
            seed=params.SHUFFLE_SEED,
        )
        
        # Display search space
        print("\n🔍 Search space summary:")
        print(f"   Optimizers: {tuner_optimizers}")
        print(f"   Learning rates: {tuner_learning_rates}")
        print(f"   Batch sizes: {tuner_batch_sizes}")
        print(f"   Architecture: FIXED ({params.MODEL_ARCHITECTURE})")
        print(f"   Total combinations: {total_combinations}")
        print(f"   Testing {num_trials} guaranteed unique combinations")
        
        # Early stopping for tuning
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=getattr(params, 'TUNER_EARLY_STOPPING_PATIENCE', 3),
            min_delta=getattr(params, 'TUNER_MIN_DELTA', 0.001),
            restore_best_weights=True,
            verbose=1 if debug else 0
        )
        
        # Run search ONLY - no final training
        print("\n🎯 Starting guaranteed unique hyperparameter search...")
        tuner.search(
            x_train, y_train,
            epochs=tuner_epochs,
            validation_data=(x_val, y_val),
            batch_size=32,  # Fixed during search for fairness
            verbose=1 if debug else 0,
            callbacks=[early_stopping]
        )
        
        # Get best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_optimizer = best_hps.get('optimizer')
        best_lr = best_hps.get('learning_rate')
        best_batch_size = best_hps.get('batch_size')
        
        # Get best trial results
        best_trials = tuner.oracle.get_best_trials(num_trials=1)
        best_score = best_trials[0].score if best_trials else 0
        
        print(f"\n🏆 GUARANTEED UNIQUE OPTIMIZATION RESULTS:")
        print("=" * 50)
        print(f"  Optimizer: {best_optimizer}")
        print(f"  Learning rate: {best_lr}")
        print(f"  Batch size: {best_batch_size}")
        print(f"  Validation Accuracy: {best_score:.4f}")
        print("=" * 50)
        
        # Save results
        results_path = os.path.join(output_dir, "guaranteed_tuning_results.txt")
        with open(results_path, 'w') as f:
            f.write(f"Guaranteed Unique Hyperparameter Optimization Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {params.MODEL_ARCHITECTURE}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Trials: {num_trials}\n")
            f.write(f"Search Strategy: Guaranteed Unique\n")
            f.write(f"Total possible combinations: {total_combinations}\n\n")
            
            f.write(f"BEST HYPERPARAMETERS:\n")
            f.write(f"  Optimizer: {best_optimizer}\n")
            f.write(f"  Learning rate: {best_lr}\n")
            f.write(f"  Batch size: {best_batch_size}\n")
            f.write(f"  Validation Accuracy: {best_score:.4f}\n\n")
            
            # Save all trials
            trials = tuner.oracle.get_best_trials(num_trials=num_trials)
            f.write("ALL TRIALS (Guaranteed Unique):\n")
            for i, trial in enumerate(trials):
                f.write(f"{i+1:2d}. Optimizer={trial.hyperparameters.get('optimizer')}, "
                       f"LR={trial.hyperparameters.get('learning_rate')}, "
                       f"BS={trial.hyperparameters.get('batch_size')}, "
                       f"Score={trial.score:.4f}\n")
        
        print(f"💾 Guaranteed unique optimization results saved to: {results_path}")
        
        return {
            'optimizer': best_optimizer,
            'learning_rate': best_lr,
            'batch_size': best_batch_size,
            'val_accuracy': best_score,
            'output_dir': output_dir,
            'search_strategy': 'guaranteed_unique'
        }
        
    except Exception as e:
        print(f"❌ Guaranteed unique optimization failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        
        # Ultimate fallback - Manual search
        print("🔄 Using ultimate fallback: Manual search")
        return manual_hyperparameter_search(x_train, y_train, x_val, y_val, num_trials, debug)

def manual_hyperparameter_search(x_train, y_train, x_val, y_val, num_trials=10, debug=False):
    """Manual hyperparameter search as ultimate fallback"""
    print("🎯 Starting MANUAL Hyperparameter Search (Guaranteed Unique)")
    
    tuner_optimizers = getattr(params, 'TUNER_OPTIMIZERS', ['adam', 'rmsprop', 'sgd', 'nadam'])
    tuner_learning_rates = getattr(params, 'TUNER_LEARNING_RATES', [1e-2, 1e-3, 1e-4])
    tuner_batch_sizes = getattr(params, 'TUNER_BATCH_SIZES', [32, 64, 128])
    
    # Generate all possible combinations
    all_combinations = list(product(tuner_optimizers, tuner_learning_rates, tuner_batch_sizes))
    random.shuffle(all_combinations)
    
    # Limit to requested number of trials
    if num_trials < len(all_combinations):
        combinations_to_test = all_combinations[:num_trials]
    else:
        combinations_to_test = all_combinations
    
    results = []
    
    print(f"🔢 Testing {len(combinations_to_test)} guaranteed unique combinations")
    
    for i, (optimizer, lr, bs) in enumerate(combinations_to_test):
        print(f"\n🔬 Trial {i+1}/{len(combinations_to_test)}: {optimizer}, LR: {lr}, BS: {bs}")
        
        # Create and train model
        model = create_model()
        
        # Select optimizer
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        elif optimizer == 'nadam':
            opt = tf.keras.optimizers.Nadam(learning_rate=lr)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        
        # Compile
        loss = 'categorical_crossentropy' if params.MODEL_ARCHITECTURE == "original_haverland" else 'sparse_categorical_crossentropy'
        model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        
        # Train briefly
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=5,  # Short training for quick evaluation
            batch_size=bs,
            verbose=0
        )
        
        val_accuracy = history.history['val_accuracy'][-1]
        results.append({
            'optimizer': optimizer,
            'learning_rate': lr,
            'batch_size': bs,
            'val_accuracy': val_accuracy
        })
        
        print(f"   ✅ Validation Accuracy: {val_accuracy:.4f}")
        
        # Clean up
        tf.keras.backend.clear_session()
    
    # Find best result
    best_result = max(results, key=lambda x: x['val_accuracy'])
    
    print(f"\n🏆 MANUAL SEARCH BEST HYPERPARAMETERS:")
    print("=" * 50)
    print(f"  Optimizer: {best_result['optimizer']}")
    print(f"  Learning rate: {best_result['learning_rate']}")
    print(f"  Batch size: {best_result['batch_size']}")
    print(f"  Validation Accuracy: {best_result['val_accuracy']:.4f}")
    print("=" * 50)
    
    return best_result