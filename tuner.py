# tuner.py
import tensorflow as tf
try:
    import keras_tuner as kt
except ImportError:
    kt = None  # keras_tuner is optional — not used by SimpleGuaranteedTuner / FineTuneTuner
import parameters as params
from models import create_model, compile_model
from utils.losses import focal_loss, sparse_focal_loss
import os
from datetime import datetime
import random
import numpy as np
from itertools import product
import json
import pandas as pd

class SimpleGuaranteedTuner:
    """Simple tuner that guarantees unique hyperparameter combinations"""
    
    def __init__(self, hypermodel, objective, max_trials, directory, project_name):
        self.hypermodel = hypermodel
        self.objective = objective
        self.max_trials = max_trials
        self.directory = directory
        self.project_name = project_name
        
        # Get search space parameters from params file
        self.tuner_optimizers = getattr(params, 'TUNER_OPTIMIZERS', ['adam', 'rmsprop', 'sgd', 'nadam'])
        self.tuner_learning_rates = getattr(params, 'TUNER_LEARNING_RATES', [1e-3, 5e-4, 1e-4])
        self.tuner_batch_sizes = getattr(params, 'TUNER_BATCH_SIZES', [32, 64])
        self.tuner_gammas = getattr(params, 'TUNER_GAMMAS', [0.0, 2.0])
        self.tuner_alphas = getattr(params, 'TUNER_ALPHAS', [0.25, 0.45])
        
        # Generate all possible configurations
        # Now including Gamma and Alpha for Focal Loss tuning
        self.all_configs = list(product(
            self.tuner_optimizers, 
            self.tuner_learning_rates, 
            self.tuner_batch_sizes,
            self.tuner_gammas,
            self.tuner_alphas
        ))
        random.shuffle(self.all_configs)
        
        # Limit to max_trials
        if self.max_trials < len(self.all_configs):
            self.all_configs = self.all_configs[:self.max_trials]
        
        self.trials = []
        self.best_score = -float('inf')
        self.best_config = None
        
        print(f"🔢 Generated {len(self.all_configs)} unique configurations (Opt x LR x BS x Gamma x Alpha)")
    
    def search(self, x_train, y_train, validation_data, epochs, verbose=0, callbacks=None):
        """Run the search"""
        x_val, y_val = validation_data
        
        for i, (optimizer, lr, bs, gamma, alpha) in enumerate(self.all_configs):
            gamma_str = f", Gamma: {gamma}" if gamma > 0 else ", Loss: SCCE"
            print(f"\n🎯 Trial {i+1}/{len(self.all_configs)}: {optimizer}, LR: {lr}, BS: {bs}{gamma_str}")
            
            try:
                # Build model with current config
                model = self._build_model_with_config(optimizer, lr, bs, gamma, alpha)
                
                # Train model
                history = model.fit(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=epochs,
                    batch_size=bs,  # Now using the tuned batch size
                    verbose=verbose,
                    callbacks=callbacks
                )
                
                # Get best validation accuracy
                val_accuracy = max(history.history['val_accuracy'])
                
                # Store trial results
                trial_result = {
                    'trial_id': i + 1,
                    'optimizer': optimizer,
                    'learning_rate': lr,
                    'batch_size': bs,
                    'gamma': gamma,
                    'alpha': alpha,
                    'val_accuracy': val_accuracy,
                    'status': 'COMPLETED',
                    'score': val_accuracy
                }
                self.trials.append(trial_result)
                
                print(f"   ✅ Validation Accuracy: {val_accuracy:.4f}")
                
                # Update best result
                if val_accuracy > self.best_score:
                    self.best_score = val_accuracy
                    self.best_config = trial_result
                    print(f"   🏆 New best configuration!")
                
            except Exception as e:
                print(f"   ❌ Trial failed: {e}")
                trial_result = {
                    'trial_id': i + 1,
                    'optimizer': optimizer,
                    'learning_rate': lr,
                    'batch_size': bs,
                    'gamma': gamma,
                    'alpha': alpha,
                    'val_accuracy': 0.0,
                    'status': 'FAILED',
                    'score': 0.0
                }
                self.trials.append(trial_result)
            
            # Clean up
            tf.keras.backend.clear_session()
    
    def _build_model_with_config(self, optimizer, learning_rate, batch_size, gamma=0.0, alpha=0.45):
        """Build model with specific configuration including Focal Loss support"""
        print(f"🏗️ Building model with: {optimizer}, LR: {learning_rate}, BS: {batch_size}, Gamma: {gamma}, Alpha: {alpha}")
        
        # Create model with current architecture
        model = create_model()
        
        # Select optimizer
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'nadam':
            opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Determine Loss Function
        is_haverland = (params.MODEL_ARCHITECTURE == "original_haverland")
        
        if gamma > 0:
            # Use Focal Loss if gamma is enabled in this trial
            if is_haverland:
                loss_fn = focal_loss(gamma=gamma, alpha=alpha)
                print(f"   Using Categorical Focal Loss (γ={gamma}, α={alpha})")
            else:
                loss_fn = sparse_focal_loss(gamma=gamma, alpha=alpha)
                print(f"   Using Sparse Focal Loss (γ={gamma}, α={alpha})")
        else:
            # Revert to standard CrossEntropy
            loss_fn = 'categorical_crossentropy' if is_haverland else 'sparse_categorical_crossentropy'
            print(f"   Using standard {'Categorical' if is_haverland else 'Sparse Categorical'} Crossentropy")
        
        model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=['accuracy']
        )
        
        # Ensure the model is built
        if not model.built:
            model.build(input_shape=(None,) + params.INPUT_SHAPE)
        
        return model
    
    def get_best_hyperparameters(self, num_trials=1):
        """Get best hyperparameters"""
        class SimpleHyperParameters:
            def __init__(self, config):
                self.config = config
            
            def get(self, key):
                return self.config[key]
        
        return [SimpleHyperParameters(self.best_config)]
    
    def oracle(self):
        """Provide oracle-like interface for compatibility"""
        class SimpleOracle:
            def __init__(self, trials):
                self.trials = trials
            
            def get_best_trials(self, num_trials):
                sorted_trials = sorted(self.trials, key=lambda x: x['score'], reverse=True)
                return sorted_trials[:num_trials]
        
        return SimpleOracle(self.trials)

def save_tuning_results_csv(trials, output_dir, search_type="guaranteed_unique"):
    """Save detailed tuning results to CSV file"""
    try:
        # Prepare CSV data
        csv_data = []
        
        for trial in trials:
            csv_data.append({
                'trial_id': trial['trial_id'],
                'optimizer': trial['optimizer'],
                'learning_rate': trial['learning_rate'],
                'batch_size': trial['batch_size'],
                'gamma': trial.get('gamma', 0.0),
                'alpha': trial.get('alpha', 0.45),
                'val_accuracy': trial['val_accuracy'],
                'status': trial['status'],
                'score': trial['score'],
                'search_type': search_type,
                'model_architecture': params.MODEL_ARCHITECTURE,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Create CSV file path
        csv_path = os.path.join(output_dir, "detailed_tuning_results.csv")
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            
            # Sort by validation accuracy (descending)
            df = df.sort_values('val_accuracy', ascending=False)
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            
            print(f"📊 Detailed tuning results saved to CSV: {csv_path}")
            print(f"   📈 Total trials recorded: {len(csv_data)}")
            print(f"   🏆 Best accuracy: {df['val_accuracy'].max():.4f}")
            print(f"   📉 Worst accuracy: {df['val_accuracy'].min():.4f}")
            print(f"   📊 Average accuracy: {df['val_accuracy'].mean():.4f}")
            
            return csv_path
        else:
            print("⚠️  No trial data available for CSV export")
            return None
            
    except Exception as e:
        print(f"❌ Error saving CSV results: {e}")
        return None

def save_best_hyperparameters_json(best_params, output_dir):
    """Save best hyperparameters to JSON file for use in parameters.py"""
    try:
        json_data = {
            "BEST_OPTIMIZER": best_params['optimizer'],
            "BEST_LEARNING_RATE": best_params['learning_rate'],
            "BEST_BATCH_SIZE": best_params['batch_size'],
            "BEST_FOCAL_GAMMA": best_params.get('gamma', 0.0),
            "BEST_FOCAL_ALPHA": best_params.get('alpha', 0.45),
            "BEST_VAL_ACCURACY": float(best_params['val_accuracy']),
            "TUNING_TIMESTAMP": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "MODEL_ARCHITECTURE": params.MODEL_ARCHITECTURE,
            "SEARCH_STRATEGY": best_params.get('search_strategy', 'guaranteed_unique')
        }
        
        json_path = os.path.join(output_dir, "best_hyperparameters.json")
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        print(f"💾 Best hyperparameters saved to JSON: {json_path}")
        
        # Also create a Python-friendly version for easy copying to parameters.py
        py_path = os.path.join(output_dir, "best_hyperparameters_for_parameters.py")
        with open(py_path, 'w') as f:
            f.write("# Best Hyperparameters for parameters.py\n")
            f.write("# Copy these values to your parameters.py file\n\n")
            f.write(f"BEST_OPTIMIZER = '{best_params['optimizer']}'\n")
            f.write(f"BEST_LEARNING_RATE = {best_params['learning_rate']}\n")
            f.write(f"BEST_BATCH_SIZE = {best_params['batch_size']}\n")
            f.write(f"FOCAL_GAMMA = {best_params.get('gamma', 0.0)}\n")
            f.write(f"FOCAL_ALPHA = {best_params.get('alpha', 0.45)}\n")
            f.write(f"# Best validation accuracy: {best_params['val_accuracy']:.4f}\n")
            f.write(f"# Tuning completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Model architecture: {params.MODEL_ARCHITECTURE}\n")
        
        print(f"🐍 Python version saved to: {py_path}")
        
        return json_path, py_path
        
    except Exception as e:
        print(f"❌ Error saving JSON results: {e}")
        return None, None

def create_tuning_summary(trials, best_params, output_dir, search_type="guaranteed_unique"):
    """Create a comprehensive tuning summary"""
    try:
        summary_path = os.path.join(output_dir, "tuning_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("HYPERPARAMETER TUNING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Model Architecture: {params.MODEL_ARCHITECTURE}\n")
            f.write(f"Search Type: {search_type}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Shape: {params.INPUT_SHAPE}\n")
            f.write(f"Output Directory: {output_dir}\n\n")
            
            f.write("BEST HYPERPARAMETERS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Optimizer: {best_params['optimizer']}\n")
            f.write(f"Learning Rate: {best_params['learning_rate']}\n")
            f.write(f"Batch Size: {best_params['batch_size']}\n")
            f.write(f"Validation Accuracy: {best_params['val_accuracy']:.4f}\n\n")
            
            # Trial statistics
            valid_trials = [t for t in trials if t['status'] == 'COMPLETED']
            
            if valid_trials:
                accuracies = [t['val_accuracy'] for t in valid_trials]
                f.write("TRIAL STATISTICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Trials: {len(valid_trials)}\n")
                f.write(f"Best Accuracy: {max(accuracies):.4f}\n")
                f.write(f"Worst Accuracy: {min(accuracies):.4f}\n")
                f.write(f"Average Accuracy: {np.mean(accuracies):.4f}\n")
                f.write(f"Standard Deviation: {np.std(accuracies):.4f}\n\n")
                
                # Top 5 configurations
                f.write("TOP 5 CONFIGURATIONS:\n")
                f.write("-" * 40 + "\n")
                sorted_trials = sorted(valid_trials, key=lambda x: x['val_accuracy'], reverse=True)[:5]
                for i, trial in enumerate(sorted_trials):
                    f.write(f"{i+1}. Optimizer: {trial['optimizer']}, "
                           f"LR: {trial['learning_rate']}, "
                           f"BS: {trial['batch_size']}, "
                           f"Accuracy: {trial['val_accuracy']:.4f}\n")
        
        print(f"📋 Tuning summary saved to: {summary_path}")
        return summary_path
        
    except Exception as e:
        print(f"❌ Error creating tuning summary: {e}")
        return None

def run_architecture_tuning(x_train, y_train, x_val, y_val, num_trials=None, debug=False):
    """Tune hyperparameters using Simple Guaranteed Optimization"""
    
    # Use parameters from config or override
    if num_trials is None:
        num_trials = getattr(params, 'TUNER_MAX_TRIALS', 100)
    
    tuner_epochs = getattr(params, 'TUNER_EPOCHS', 10)
    
    # Get search space parameters
    tuner_optimizers = getattr(params, 'TUNER_OPTIMIZERS', ['adam', 'rmsprop', 'sgd', 'nadam'])
    tuner_learning_rates = getattr(params, 'TUNER_LEARNING_RATES', [1e-3, 5e-4, 2e-4, 1e-4])
    tuner_batch_sizes = getattr(params, 'TUNER_BATCH_SIZES', [32, 64])
    tuner_gammas = getattr(params, 'TUNER_GAMMAS', [0.0, 1.5, 2.0, 3.0, 4.5])
    tuner_alphas = getattr(params, 'TUNER_ALPHAS', [0.25, 0.45])
    
    # Calculate total possible combinations
    total_combinations = len(tuner_optimizers) * len(tuner_learning_rates) * len(tuner_batch_sizes) * len(tuner_gammas) * len(tuner_alphas)
    
    # Don't allow more trials than possible combinations
    if num_trials > total_combinations:
        print(f"⚠️  Reducing trials from {num_trials} to {total_combinations} (total possible combinations)")
        num_trials = total_combinations
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(params.OUTPUT_DIR, f"guaranteed_tune_{params.MODEL_ARCHITECTURE}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting Simple Guaranteed Hyperparameter Optimization")
    print("=" * 60)
    print(f"🎯 Tuning model: {params.MODEL_ARCHITECTURE}")
    print(f"🔬 Trials: {num_trials}")
    print(f"📈 Epochs per trial: {tuner_epochs}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Strategy: Guaranteed Unique Configurations")
    
    try:
        # Create our simple tuner
        tuner = SimpleGuaranteedTuner(
            hypermodel=None,  # We handle model building ourselves
            objective='val_accuracy',
            max_trials=num_trials,
            directory=output_dir,
            project_name=f'guaranteed_tune_{params.MODEL_ARCHITECTURE}'
        )
        
        # Display search space
        print("\n🔍 Search space summary:")
        print(f"   Optimizers: {tuner_optimizers}")
        print(f"   Learning rates: {tuner_learning_rates}")
        print(f"   Batch sizes: {tuner_batch_sizes}")
        print(f"   Gammas (Focal): {tuner_gammas}")
        print(f"   Alphas (Focal): {tuner_alphas}")
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
        
        # Run search
        print("\n🎯 Starting guaranteed unique hyperparameter search...")
        tuner.search(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=tuner_epochs,
            verbose=1 if debug else 0,
            callbacks=[early_stopping]
        )
        
        # Get best results
        best_config = tuner.best_config
        
        print(f"\n🏆 GUARANTEED UNIQUE OPTIMIZATION RESULTS:")
        print("=" * 50)
        print(f"  Optimizer: {best_config['optimizer']}")
        print(f"  Learning rate: {best_config['learning_rate']}")
        print(f"  Batch size: {best_config['batch_size']}")
        print(f"  Validation Accuracy: {best_config['val_accuracy']:.4f}")
        print("=" * 50)
        
        # Create best parameters dictionary
        best_params = {
            'optimizer': best_config['optimizer'],
            'learning_rate': best_config['learning_rate'],
            'batch_size': best_config['batch_size'],
            'gamma': best_config.get('gamma', 0.0),
            'alpha': best_config.get('alpha', 0.45),
            'val_accuracy': best_config['val_accuracy'],
            'output_dir': output_dir,
            'search_strategy': 'guaranteed_unique'
        }
        
        # Save detailed CSV results
        csv_path = save_tuning_results_csv(tuner.trials, output_dir, "guaranteed_unique")
        
        # Create comprehensive summary
        summary_path = create_tuning_summary(tuner.trials, best_params, output_dir, "guaranteed_unique")
        
        # Save to JSON for use in parameters.py
        json_path, py_path = save_best_hyperparameters_json(best_params, output_dir)
        
        print(f"\n🎉 Tuning completed successfully!")
        print(f"📁 All results saved in: {output_dir}")
        print(f"   📊 CSV: detailed_tuning_results.csv")
        print(f"   📋 Summary: tuning_summary.txt")
        print(f"   🏆 Best params: best_hyperparameters.json")
        print(f"   🐍 Python ready: best_hyperparameters_for_parameters.py")
        
        return best_params
        
    except Exception as e:
        print(f"❌ Guaranteed unique optimization failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        
        # Ultimate fallback - Manual search
        print("🔄 Using ultimate fallback: Manual search")
        return manual_hyperparameter_search(x_train, y_train, x_val, y_val, num_trials, debug)


# ==============================================================================
# Fine-Tune Tuner  (python tuner.py --finetune [--model best_model.keras])
# ==============================================================================

class FineTuneTuner(SimpleGuaranteedTuner):
    """
    Variant of SimpleGuaranteedTuner that loads a pre-trained model instead of
    building one from scratch.  Searches only the post-plateau decision space:
      - Optimizer (adam / rmsprop / nadam / sgd)
      - Small learning rates (5e-5 … 5e-4)
      - ReduceLROnPlateau decay factor

    All layers are unfrozen by default (FINETUNE_UNFREEZE_LAST_N = 0).
    Set FINETUNE_UNFREEZE_LAST_N > 0 to freeze the first N-from-last layers.
    """

    def __init__(self, model_path, hypermodel, objective, max_trials,
                 directory, project_name):
        # Build a narrowed search space from fine-tune params
        self.model_path = model_path
        self.unfreeze_last_n = getattr(params, 'FINETUNE_UNFREEZE_LAST_N', 0)

        # Override search space before calling super()
        import parameters as _p
        _p.TUNER_OPTIMIZERS   = getattr(_p, 'TUNER_FINETUNE_OPTIMIZERS',
                                        ['adam', 'rmsprop', 'nadam', 'sgd'])
        _p.TUNER_LEARNING_RATES = getattr(_p, 'TUNER_FINETUNE_LRS',
                                          [5e-5, 1e-4, 3e-4, 5e-4])
        _p.TUNER_BATCH_SIZES  = getattr(_p, 'TUNER_BATCH_SIZES', [32, 64])
        # Fine-tune over LR factors; store them as the "gamma" axis (reuse Cartesian product)
        _p.TUNER_GAMMAS       = getattr(_p, 'TUNER_LR_FACTORS', [0.3, 0.5, 0.7])
        _p.TUNER_ALPHAS       = [params.FOCAL_ALPHA]  # single value — not tuned here

        super().__init__(hypermodel, objective, max_trials, directory, project_name)
        print(f"🔁 FineTuneTuner: loading weights from {model_path}")
        print(f"   Unfreeze last N layers: {'ALL' if self.unfreeze_last_n == 0 else self.unfreeze_last_n}")

    def _build_model_with_config(self, optimizer, learning_rate, batch_size,
                                  gamma=0.5, alpha=None):
        """Load the pre-trained model and recompile with the trial hyperparameters.
        `gamma` here is reused as the ReduceLROnPlateau factor."""
        lr_factor = gamma  # semantic rename
        print(f"🏗️  Fine-tune trial: {optimizer}, LR={learning_rate}, "
              f"BS={batch_size}, LR-factor={lr_factor:.2f}")

        # Load pre-trained model (weights + architecture)
        model = tf.keras.models.load_model(
            self.model_path,
            compile=False,  # We'll recompile with the trial optimizer
        )

        # Selective layer freezing
        if self.unfreeze_last_n > 0:
            for layer in model.layers[:-self.unfreeze_last_n]:
                layer.trainable = False
            print(f"   ❄️  Froze {len(model.layers) - self.unfreeze_last_n} layers, "
                  f"training last {self.unfreeze_last_n}")
        else:
            for layer in model.layers:
                layer.trainable = True

        # Build optimizer
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                          momentum=0.9, nesterov=True)
        elif optimizer == 'nadam':
            opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Recompile with the existing loss type
        is_haverland = (params.MODEL_ARCHITECTURE == "original_haverland")
        if params.LOSS_TYPE in ["focal_loss", "IntelligentFocalLossController"]:
            from utils.losses import DynamicSparseFocalLoss, DynamicFocalLoss
            loss_fn = (DynamicFocalLoss(gamma=params.FOCAL_GAMMA, alpha=params.FOCAL_ALPHA)
                       if is_haverland else
                       DynamicSparseFocalLoss(gamma=params.FOCAL_GAMMA, alpha=params.FOCAL_ALPHA))
        else:
            loss_fn = ('categorical_crossentropy' if is_haverland
                       else 'sparse_categorical_crossentropy')

        model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
        return model

    def search(self, x_train, y_train, validation_data, epochs, verbose=0,
               callbacks=None, lr_factors=None):
        """Override search to inject ReduceLROnPlateau with the trial lr_factor."""
        x_val, y_val = validation_data

        for i, (optimizer, lr, bs, lr_factor, alpha) in enumerate(self.all_configs):
            print(f"\n🎯 Fine-Tune Trial {i+1}/{len(self.all_configs)}: "
                  f"{optimizer}, LR={lr}, BS={bs}, LR-factor={lr_factor:.2f}")
            try:
                model = self._build_model_with_config(optimizer, lr, bs, lr_factor, alpha)

                # Per-trial ReduceLROnPlateau with the swept factor
                trial_callbacks = [
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=lr_factor,
                        patience=params.LR_SCHEDULER_PATIENCE,
                        min_lr=params.LR_SCHEDULER_MIN_LR,
                        verbose=0,
                    ),
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_accuracy',
                        patience=getattr(params, 'TUNER_EARLY_STOPPING_PATIENCE', 5),
                        min_delta=0.0002,
                        restore_best_weights=True,
                        verbose=0,
                    ),
                ]

                history = model.fit(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=epochs,
                    batch_size=bs,
                    verbose=verbose,
                    callbacks=trial_callbacks,
                )

                val_accuracy = max(history.history['val_accuracy'])
                trial_result = {
                    'trial_id': i + 1,
                    'optimizer': optimizer,
                    'learning_rate': lr,
                    'batch_size': bs,
                    'gamma': lr_factor,   # stored as gamma for CSV compatibility
                    'lr_factor': lr_factor,
                    'alpha': alpha,
                    'val_accuracy': val_accuracy,
                    'status': 'COMPLETED',
                    'score': val_accuracy,
                }
                self.trials.append(trial_result)
                print(f"   ✅ Val Accuracy: {val_accuracy:.4f}")

                if val_accuracy > self.best_score:
                    self.best_score = val_accuracy
                    self.best_config = trial_result
                    print(f"   🏆 New best!")

            except Exception as e:
                print(f"   ❌ Trial failed: {e}")
                import traceback; traceback.print_exc()
                self.trials.append({
                    'trial_id': i + 1, 'optimizer': optimizer,
                    'learning_rate': lr, 'batch_size': bs,
                    'gamma': lr_factor, 'lr_factor': lr_factor, 'alpha': alpha,
                    'val_accuracy': 0.0, 'status': 'FAILED', 'score': 0.0,
                })

            tf.keras.backend.clear_session()


def _discover_best_model(nb_classes=None, input_channels=None):
    """Auto-discover the most recent best_model.keras in exported_models."""
    nb = nb_classes or params.NB_CLASSES
    ch = input_channels or params.INPUT_CHANNELS
    color = 'GRAY' if ch == 1 else 'RGB'
    base = os.path.join(params.OUTPUT_DIR)  # exported_models/{N}cls_{C}

    candidates = []
    if os.path.isdir(base):
        for run_dir in os.listdir(base):
            candidate = os.path.join(base, run_dir, 'best_model.keras')
            if os.path.isfile(candidate):
                candidates.append(candidate)

    if not candidates:
        return None

    # Return the most recently modified
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def run_finetune_tuning(x_train, y_train, x_val, y_val,
                        model_path=None, num_trials=None, debug=False):
    """
    Load a pre-trained best_model.keras and search for the best fine-tuning
    configuration (optimizer, small LR, LR-decay factor).

    Args:
        model_path: explicit path to best_model.keras; auto-discovered if None.
        num_trials:  overrides TUNER_MAX_TRIALS when supplied.
    """
    # --- Resolve model path ---
    if model_path is None:
        model_path = _discover_best_model()
        if model_path is None:
            print("❌ No best_model.keras found in OUTPUT_DIR. "
                  "Specify --model or run a full training first.")
            return None
    if not os.path.isfile(model_path):
        print(f"❌ Model not found: {model_path}")
        return None

    finetune_epochs = getattr(params, 'TUNER_FINETUNE_EPOCHS', 15)
    if num_trials is None:
        # default: sweep all fine-tune optimizer × LR × BS × factor combos
        n_opt = len(getattr(params, 'TUNER_FINETUNE_OPTIMIZERS', ['adam','rmsprop','nadam','sgd']))
        n_lr  = len(getattr(params, 'TUNER_FINETUNE_LRS', [5e-5, 1e-4, 3e-4, 5e-4]))
        n_bs  = len(getattr(params, 'TUNER_BATCH_SIZES', [32, 64]))
        n_fac = len(getattr(params, 'TUNER_LR_FACTORS', [0.3, 0.5, 0.7]))
        num_trials = n_opt * n_lr * n_bs * n_fac

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(params.OUTPUT_DIR,
                              f"finetune_{params.MODEL_ARCHITECTURE}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("🔁 FINE-TUNE HYPERPARAMETER SEARCH")
    print("=" * 60)
    print(f"  Model      : {model_path}")
    print(f"  Trials     : {num_trials}")
    print(f"  Epochs/trial: {finetune_epochs}")
    print(f"  Output dir : {output_dir}")
    print(f"  Optimizers : {getattr(params, 'TUNER_FINETUNE_OPTIMIZERS', [])}")
    print(f"  LRs        : {getattr(params, 'TUNER_FINETUNE_LRS', [])}")
    print(f"  LR factors : {getattr(params, 'TUNER_LR_FACTORS', [])}")

    tuner = FineTuneTuner(
        model_path=model_path,
        hypermodel=None,
        objective='val_accuracy',
        max_trials=num_trials,
        directory=output_dir,
        project_name=f'finetune_{params.MODEL_ARCHITECTURE}',
    )

    tuner.search(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=finetune_epochs,
        verbose=1 if debug else 0,
    )

    best = tuner.best_config
    if best is None:
        print("❌ All fine-tune trials failed.")
        return None

    print(f"\n🏆 FINE-TUNE RESULTS:")
    print("=" * 50)
    print(f"  Optimizer  : {best['optimizer']}")
    print(f"  LR         : {best['learning_rate']}")
    print(f"  Batch size : {best['batch_size']}")
    print(f"  LR factor  : {best.get('lr_factor', best.get('gamma', '?'))}")
    print(f"  Val accuracy: {best['val_accuracy']:.4f}")
    print("=" * 50)

    best_params_out = {
        'optimizer': best['optimizer'],
        'learning_rate': best['learning_rate'],
        'batch_size': best['batch_size'],
        'gamma': best.get('lr_factor', 0.0),
        'alpha': best.get('alpha', params.FOCAL_ALPHA),
        'val_accuracy': best['val_accuracy'],
        'output_dir': output_dir,
        'search_strategy': 'finetune',
    }

    save_tuning_results_csv(tuner.trials, output_dir, "finetune")
    create_tuning_summary(tuner.trials, best_params_out, output_dir, "finetune")
    save_best_hyperparameters_json(best_params_out, output_dir)

    print(f"\n🎉 Fine-tune search complete! Results in: {output_dir}")
    return best_params_out


# ==============================================================================
# CLI entry-point
# ==============================================================================

if __name__ == '__main__':
    import argparse
    import os
    os.environ.setdefault('DIGIT_NB_CLASSES', str(params.NB_CLASSES))
    os.environ.setdefault('DIGIT_INPUT_CHANNELS', str(params.INPUT_CHANNELS))

    parser = argparse.ArgumentParser(
        description='Hyperparameter tuner for digit-recognizer models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  # Full search from scratch:\n'
            '  python tuner.py\n\n'
            '  # Fine-tune an existing model (auto-discover latest):\n'
            '  python tuner.py --finetune\n\n'
            '  # Fine-tune a specific model:\n'
            '  python tuner.py --finetune --model path/to/best_model.keras\n'
        ),
    )
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune an existing model instead of training from scratch')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to best_model.keras (only used with --finetune)')
    parser.add_argument('--trials', type=int, default=None,
                        help='Number of trials (overrides TUNER_MAX_TRIALS / auto-count)')
    parser.add_argument('--debug', action='store_true',
                        help='Verbose output during tuning')
    args = parser.parse_args()

    # Load data (common to both modes) — mirrors train.py's loading pattern
    from utils import get_data_splits, preprocess_for_training
    print("📦 Loading dataset...")
    try:
        (x_train_raw, y_train_raw), (x_val_raw, y_val_raw), _ = get_data_splits()
        x_train = preprocess_for_training(x_train_raw)
        x_val   = preprocess_for_training(x_val_raw)
        # For haverland (one-hot labels); all other architectures use sparse integer labels
        if params.MODEL_ARCHITECTURE == "original_haverland":
            import tensorflow as _tf
            y_train = _tf.keras.utils.to_categorical(y_train_raw, params.NB_CLASSES)
            y_val   = _tf.keras.utils.to_categorical(y_val_raw,   params.NB_CLASSES)
        else:
            y_train = y_train_raw.copy()
            y_val   = y_val_raw.copy()
        print(f"   ✅ x_train {x_train.shape}, x_val {x_val.shape}")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        print("   Make sure DIGIT_NB_CLASSES and DIGIT_INPUT_CHANNELS env vars are set.")
        raise

    if args.finetune:
        run_finetune_tuning(
            x_train, y_train, x_val, y_val,
            model_path=args.model,
            num_trials=args.trials,
            debug=args.debug,
        )
    else:
        run_architecture_tuning(
            x_train, y_train, x_val, y_val,
            num_trials=args.trials,
            debug=args.debug,
        )


def manual_hyperparameter_search(x_train, y_train, x_val, y_val, num_trials=10, debug=False):
    """Manual hyperparameter search as ultimate fallback"""
    print("🎯 Starting MANUAL Hyperparameter Search (Guaranteed Unique)")
    
    tuner_optimizers = getattr(params, 'TUNER_OPTIMIZERS', ['adam', 'rmsprop', 'sgd', 'nadam'])
    tuner_learning_rates = getattr(params, 'TUNER_LEARNING_RATES', [1e-3, 5e-4, 2e-4, 1e-4])
    tuner_batch_sizes = getattr(params, 'TUNER_BATCH_SIZES', [32, 64])
    tuner_gammas = getattr(params, 'TUNER_GAMMAS', [0.0, 1.5, 2.0, 3.0, 4.5])
    tuner_alphas = getattr(params, 'TUNER_ALPHAS', [0.25, 0.45])
    
    # Generate all possible combinations
    all_combinations = list(product(
        tuner_optimizers, 
        tuner_learning_rates, 
        tuner_batch_sizes,
        tuner_gammas,
        tuner_alphas
    ))
    random.shuffle(all_combinations)
    
    # Limit to requested number of trials
    if num_trials < len(all_combinations):
        combinations_to_test = all_combinations[:num_trials]
    else:
        combinations_to_test = all_combinations
    
    results = []
    
    print(f"🔢 Testing {len(combinations_to_test)} guaranteed unique combinations")
    
    for i, (optimizer, lr, bs, gamma, alpha) in enumerate(combinations_to_test):
        gamma_str = f", Gamma: {gamma}" if gamma > 0 else ", Loss: SCCE"
        print(f"\n🔬 Trial {i+1}/{len(combinations_to_test)}: {optimizer}, LR: {lr}, BS: {bs}{gamma_str}")
        
        try:
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
            
            # Determine Loss Function
            is_haverland = (params.MODEL_ARCHITECTURE == "original_haverland")
            if gamma > 0:
                if is_haverland:
                    loss_fn = focal_loss(gamma=gamma, alpha=alpha)
                else:
                    loss_fn = sparse_focal_loss(gamma=gamma, alpha=alpha)
            else:
                loss_fn = 'categorical_crossentropy' if is_haverland else 'sparse_categorical_crossentropy'
            
            model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
            
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
                'trial_id': i + 1,
                'optimizer': optimizer,
                'learning_rate': lr,
                'batch_size': bs,
                'gamma': gamma,
                'alpha': alpha,
                'val_accuracy': val_accuracy,
                'status': 'COMPLETED',
                'score': val_accuracy
            })
            
            print(f"   ✅ Validation Accuracy: {val_accuracy:.4f}")
            
        except Exception as e:
            print(f"   ❌ Trial failed: {e}")
            results.append({
                'trial_id': i + 1,
                'optimizer': optimizer,
                'learning_rate': lr,
                'batch_size': bs,
                'gamma': gamma,
                'alpha': alpha,
                'val_accuracy': 0.0,
                'status': 'FAILED',
                'score': 0.0
            })
        
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
    
    # Create output directory for manual search results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(params.OUTPUT_DIR, f"manual_tune_{params.MODEL_ARCHITECTURE}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    best_params = {
        'optimizer': best_result['optimizer'],
        'learning_rate': best_result['learning_rate'],
        'batch_size': best_result['batch_size'],
        'gamma': best_result.get('gamma', 0.0),
        'alpha': best_result.get('alpha', 0.45),
        'val_accuracy': best_result['val_accuracy'],
        'output_dir': output_dir,
        'search_strategy': 'manual'
    }
    
    # Save detailed CSV results
    csv_path = save_tuning_results_csv(results, output_dir, "manual")
    
    # Save to JSON for use in parameters.py
    json_path, py_path = save_best_hyperparameters_json(best_params, output_dir)
    
    # Create comprehensive summary for manual search
    try:
        summary_path = create_tuning_summary(results, best_params, output_dir, "manual")
        print(f"📋 Manual search summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"❌ Error creating manual search summary: {e}")
    
    print(f"\n🎉 Manual search completed successfully!")
    print(f"📁 All results saved in: {output_dir}")
    
    return best_params