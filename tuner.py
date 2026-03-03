# tuner.py
import tensorflow as tf
import keras_tuner as kt
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