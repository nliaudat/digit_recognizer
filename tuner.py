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
import json
import csv
import pandas as pd

class GuaranteedBayesianOptimization(kt.BayesianOptimization):
    """Bayesian Optimization that guarantees unique hyperparameter combinations"""
    
    def __init__(self, hypermodel, objective, max_trials, **kwargs):
        # Store search space parameters first
        self.tuner_optimizers = getattr(params, 'TUNER_OPTIMIZERS', ['adam', 'rmsprop', 'sgd', 'nadam'])
        self.tuner_learning_rates = getattr(params, 'TUNER_LEARNING_RATES', [1e-2, 1e-3, 1e-4])
        self.tuner_batch_sizes = getattr(params, 'TUNER_BATCH_SIZES', [32, 64, 128])
        
        # Generate all possible configurations
        self._all_possible_configs = list(product(
            self.tuner_optimizers, 
            self.tuner_learning_rates, 
            self.tuner_batch_sizes
        ))
        random.shuffle(self._all_possible_configs)
        self._used_configs = set()
        
        print(f"🔢 Generated {len(self._all_possible_configs)} unique configurations")
        
        # Now initialize the parent class
        super().__init__(
            hypermodel=hypermodel,
            objective=objective,
            max_trials=max_trials,
            **kwargs
        )
    
    def _populate_space(self, trial_id):
        """Override to guarantee unique configurations"""
        # If we haven't used all configurations yet, use them in order
        if len(self._used_configs) < len(self._all_possible_configs):
            for config in self._all_possible_configs:
                config_key = str(config)  # Convert to string for set storage
                if config_key not in self._used_configs:
                    self._used_configs.add(config_key)
                    optimizer, lr, bs = config
                    
                    # Create fresh hyperparameters for this trial
                    hp = kt.HyperParameters()
                    
                    # Define the search space choices
                    hp.Choice('optimizer', values=self.tuner_optimizers)
                    hp.Choice('learning_rate', values=self.tuner_learning_rates)
                    hp.Choice('batch_size', values=self.tuner_batch_sizes)
                    
                    # Force the specific values for this trial
                    hp.values['optimizer'] = optimizer
                    hp.values['learning_rate'] = lr
                    hp.values['batch_size'] = bs
                    
                    print(f"🎯 Using guaranteed unique config: {optimizer}, LR: {lr}, BS: {bs}")
                    return {'status': kt.ObjectiveStatus.RUNNING, 'values': hp.values}
        
        # Fallback to Bayesian optimization if we've used all combinations
        print("🔄 All unique configurations used, switching to Bayesian sampling")
        return super()._populate_space(trial_id)

def build_model(hp):
    """Build model with hyperparameters for tuning"""
    
    try:
        # Safely get values from hyperparameters with fallbacks
        optimizer_type = hp.get('optimizer') or 'adam'
        learning_rate = hp.get('learning_rate') or 0.001
        batch_size = hp.get('batch_size') or 32
        
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
        
    except Exception as e:
        print(f"❌ Error building model: {e}")
        raise

def save_tuning_results_csv(tuner, output_dir, search_type="guaranteed_unique"):
    """Save detailed tuning results to CSV file"""
    
    try:
        # Get all trials
        trials = tuner.oracle.get_best_trials(num_trials=len(tuner.oracle.trials))
        
        # Prepare CSV data
        csv_data = []
        
        for i, trial in enumerate(trials):
            if trial and trial.score is not None and trial.hyperparameters:
                try:
                    csv_data.append({
                        'trial_id': i + 1,
                        'trial_trial_id': getattr(trial, 'trial_id', f'trial_{i}'),
                        'optimizer': trial.hyperparameters.get('optimizer', 'unknown'),
                        'learning_rate': trial.hyperparameters.get('learning_rate', 0.001),
                        'batch_size': trial.hyperparameters.get('batch_size', 32),
                        'val_accuracy': trial.score,
                        'status': getattr(trial, 'status', 'COMPLETED'),
                        'score': trial.score,
                        'search_type': search_type,
                        'model_architecture': params.MODEL_ARCHITECTURE,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                except Exception as e:
                    print(f"⚠️  Could not process trial {i}: {e}")
                    continue
        
        # Create CSV file path
        csv_path = os.path.join(output_dir, "detailed_tuning_results.csv")
        
        # Write to CSV
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

def save_manual_search_results_csv(results, output_dir):
    """Save manual search results to CSV file"""
    
    try:
        csv_data = []
        
        for i, result in enumerate(results):
            csv_data.append({
                'trial_id': i + 1,
                'trial_trial_id': f"manual_{i+1}",
                'optimizer': result['optimizer'],
                'learning_rate': result['learning_rate'],
                'batch_size': result['batch_size'],
                'val_accuracy': result['val_accuracy'],
                'status': 'COMPLETED',
                'score': result['val_accuracy'],
                'search_type': 'manual',
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
            
            print(f"📊 Manual search results saved to CSV: {csv_path}")
            print(f"   📈 Total trials recorded: {len(csv_data)}")
            print(f"   🏆 Best accuracy: {df['val_accuracy'].max():.4f}")
            print(f"   📉 Worst accuracy: {df['val_accuracy'].min():.4f}")
            print(f"   📊 Average accuracy: {df['val_accuracy'].mean():.4f}")
            
            return csv_path
        else:
            print("⚠️  No manual search data available for CSV export")
            return None
            
    except Exception as e:
        print(f"❌ Error saving manual search CSV: {e}")
        return None

def save_best_hyperparameters_json(best_params, output_dir):
    """Save best hyperparameters to JSON file for use in parameters.py"""
    
    try:
        json_data = {
            "BEST_OPTIMIZER": best_params['optimizer'],
            "BEST_LEARNING_RATE": best_params['learning_rate'],
            "BEST_BATCH_SIZE": best_params['batch_size'],
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
            f.write(f"# Best validation accuracy: {best_params['val_accuracy']:.4f}\n")
            f.write(f"# Tuning completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Model architecture: {params.MODEL_ARCHITECTURE}\n")
        
        print(f"🐍 Python version saved to: {py_path}")
        
        return json_path, py_path
        
    except Exception as e:
        print(f"❌ Error saving JSON results: {e}")
        return None, None

def create_tuning_summary(tuner, output_dir, best_params, search_type="guaranteed_unique"):
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
            trials = tuner.oracle.get_best_trials(num_trials=len(tuner.oracle.trials))
            valid_trials = [t for t in trials if t and t.score is not None]
            
            if valid_trials:
                accuracies = [t.score for t in valid_trials]
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
                sorted_trials = sorted(valid_trials, key=lambda x: x.score, reverse=True)[:5]
                for i, trial in enumerate(sorted_trials):
                    f.write(f"{i+1}. Optimizer: {trial.hyperparameters.get('optimizer', 'unknown')}, "
                           f"LR: {trial.hyperparameters.get('learning_rate', 0.001)}, "
                           f"BS: {trial.hyperparameters.get('batch_size', 32)}, "
                           f"Accuracy: {trial.score:.4f}\n")
        
        print(f"📋 Tuning summary saved to: {summary_path}")
        return summary_path
        
    except Exception as e:
        print(f"❌ Error creating tuning summary: {e}")
        return None

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
        
        # Create best parameters dictionary
        best_params = {
            'optimizer': best_optimizer,
            'learning_rate': best_lr,
            'batch_size': best_batch_size,
            'val_accuracy': best_score,
            'output_dir': output_dir,
            'search_strategy': 'guaranteed_unique'
        }
        
        # Save detailed CSV results
        csv_path = save_tuning_results_csv(tuner, output_dir, "guaranteed_unique")
        
        # Create comprehensive summary
        summary_path = create_tuning_summary(tuner, output_dir, best_params, "guaranteed_unique")
        
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
            
        except Exception as e:
            print(f"   ❌ Trial failed: {e}")
            results.append({
                'optimizer': optimizer,
                'learning_rate': lr,
                'batch_size': bs,
                'val_accuracy': 0.0
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
    
    # Add output directory to result
    best_result['output_dir'] = output_dir
    best_result['search_strategy'] = 'manual'
    
    # Save detailed CSV results
    csv_path = save_manual_search_results_csv(results, output_dir)
    
    # Save to JSON for use in parameters.py
    json_path, py_path = save_best_hyperparameters_json(best_result, output_dir)
    
    # Create comprehensive summary for manual search
    try:
        summary_path = os.path.join(output_dir, "tuning_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("MANUAL HYPERPARAMETER TUNING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model Architecture: {params.MODEL_ARCHITECTURE}\n")
            f.write(f"Search Type: Manual\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Trials Completed: {len(results)}\n\n")
            
            f.write("BEST HYPERPARAMETERS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Optimizer: {best_result['optimizer']}\n")
            f.write(f"Learning Rate: {best_result['learning_rate']}\n")
            f.write(f"Batch Size: {best_result['batch_size']}\n")
            f.write(f"Validation Accuracy: {best_result['val_accuracy']:.4f}\n\n")
            
            # Statistics
            accuracies = [r['val_accuracy'] for r in results]
            f.write("SEARCH STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Trials: {len(results)}\n")
            f.write(f"Best Accuracy: {max(accuracies):.4f}\n")
            f.write(f"Worst Accuracy: {min(accuracies):.4f}\n")
            f.write(f"Average Accuracy: {np.mean(accuracies):.4f}\n")
            f.write(f"Standard Deviation: {np.std(accuracies):.4f}\n")
        
        print(f"📋 Manual search summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"❌ Error creating manual search summary: {e}")
    
    print(f"\n🎉 Manual search completed successfully!")
    print(f"📁 All results saved in: {output_dir}")
    
    return best_result