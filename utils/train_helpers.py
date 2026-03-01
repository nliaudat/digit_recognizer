# utils/train_helpers.py
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.keras import backend as K
import parameters as params

def print_training_summary(model, x_train, x_val, x_test, debug=False):
    """Print comprehensive training summary"""
    print("\n" + "="*60)
    print("🎯 TRAINING SUMMARY")
    print("="*60)
    
    print(f"📊 Data Shapes:")
    print(f"   Training data:   {x_train.shape}")
    print(f"   Validation data: {x_val.shape}")
    print(f"   Test data:       {x_test.shape}")
    
    print(f"\n🔧 Model Architecture: {params.MODEL_ARCHITECTURE}")
    print(f"   Input shape: {params.INPUT_SHAPE}")
    print(f"   Number of classes: {params.NB_CLASSES}")
    print(f"   Total parameters: {model.count_params():,}")
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs: {params.EPOCHS}")
    print(f"   Batch size: {params.BATCH_SIZE}")
    print(f"   Learning rate: {params.LEARNING_RATE}")
    print(f"   Quantization: {params.QUANTIZE_MODEL}")
    print(f"   QAT: {params.USE_QAT}")
    print(f"   ESP-DL: {params.ESP_DL_QUANTIZE}")
    
    print(f"\n🎯 Quantization Configuration:")
    print(f"   Quantization: {params.QUANTIZE_MODEL}")
    print(f"   QAT: {params.USE_QAT}")
    print(f"   ESP-DL: {params.ESP_DL_QUANTIZE}")
    
    if params.USE_QAT and params.QUANTIZE_MODEL:
        print(f"   QAT Data Format: UINT8 [0, 255]")
        print(f"   Training matches inference format: ✅")
    elif params.QUANTIZE_MODEL:
        print(f"   PTQ Data Format: Train=Float32 [0,1], Infer=UINT8 [0,255]")
    else:
        print(f"   Data Format: Float32 [0, 1]")
    
    if debug:
        print(f"\n🔍 Debug Info:")
        print(f"   Model layers: {len(model.layers)}")
        print(f"   Model built: {model.built}")

def save_model_summary_to_file(model, training_dir):
    """Save model summary to file with proper encoding"""
    summary_path = os.path.join(training_dir, "model_summary.txt")
    
    # Method 1: Try with UTF-8 encoding first
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            # Save string representation
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"💾 Model summary saved to: {summary_path}")
        return
    except Exception as e:
        print(f"⚠️  UTF-8 encoding failed: {e}")
    
    # Method 2: Try to capture summary as string and clean it
    try:
        # Capture summary as string
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        
        # Clean any problematic characters
        cleaned_lines = []
        for line in summary_lines:
            # Remove or replace problematic Unicode characters
            cleaned_line = line.replace('─', '-').replace('┌', '+').replace('┐', '+').replace('└', '+').replace('┘', '+').replace('├', '+').replace('┤', '+').replace('│', '|')
            cleaned_lines.append(cleaned_line)
        
        # Write cleaned summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))
        print(f"💾 Model summary (cleaned) saved to: {summary_path}")
        return
    except Exception as e:
        print(f"⚠️  Cleaned summary failed: {e}")
    
    # Method 3: Fallback - just save basic model info
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {params.MODEL_ARCHITECTURE}\n")
            f.write(f"Input Shape: {params.INPUT_SHAPE}\n")
            f.write(f"Number of Classes: {params.NB_CLASSES}\n")
            f.write(f"Total Parameters: {model.count_params():,}\n")
            f.write(f"Number of Layers: {len(model.layers)}\n")
            f.write("\nLayer Information:\n")
            for i, layer in enumerate(model.layers):
                f.write(f"Layer {i}: {layer.name} - {type(layer).__name__}\n")
                f.write(f"  Output Shape: {layer.output_shape}\n")
                f.write(f"  Parameters: {layer.count_params()}\n")
        print(f"💾 Basic model info saved to: {summary_path}")
    except Exception as e:
        print(f"❌ Failed to save model summary: {e}")

def save_training_config(training_dir, tflite_size, keras_size, tflite_manager, 
                        test_accuracy, tflite_accuracy, training_time, debug, model=None):
    """Save training configuration and results to file"""
    config_path = os.path.join(training_dir, "training_config.txt")
    
    config_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_duration': str(training_time),
        'model_architecture': params.MODEL_ARCHITECTURE,
        'input_shape': params.INPUT_SHAPE,
        'num_classes': params.NB_CLASSES,
        'epochs': params.EPOCHS,
        'batch_size': params.BATCH_SIZE,
        'learning_rate': params.LEARNING_RATE,
        'quantization_enabled': params.QUANTIZE_MODEL,
        'qat_enabled': params.USE_QAT,
        'esp_dl_enabled': params.ESP_DL_QUANTIZE,
        'data_augmentation': params.USE_DATA_AUGMENTATION,
        'grayscale': params.USE_GRAYSCALE,
        'best_validation_accuracy': float(tflite_manager.best_accuracy),
        'test_accuracy_keras': float(test_accuracy),
        'test_accuracy_tflite': float(tflite_accuracy),
        'model_size_keras_kb': float(keras_size),
        'model_size_tflite_kb': float(tflite_size),
        'total_parameters': model.count_params() if model else 0
    }
    
    # Calculate size reduction if quantization is enabled
    if params.QUANTIZE_MODEL and keras_size > 0:
        size_reduction = ((keras_size - tflite_size) / keras_size) * 100
        config_data['size_reduction_percent'] = float(size_reduction)
    
    # Write to file with UTF-8 encoding
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("TRAINING CONFIGURATION AND RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("📅 Training Information:\n")
            f.write(f"   Timestamp: {config_data['timestamp']}\n")
            f.write(f"   Duration: {config_data['training_duration']}\n\n")
            
            f.write("🔧 Model Configuration:\n")
            f.write(f"   Architecture: {config_data['model_architecture']}\n")
            f.write(f"   Input shape: {config_data['input_shape']}\n")
            f.write(f"   Number of classes: {config_data['num_classes']}\n")
            f.write(f"   Total parameters: {config_data['total_parameters']:,}\n\n")
            
            f.write("⚙️  Training Parameters:\n")
            f.write(f"   Epochs: {config_data['epochs']}\n")
            f.write(f"   Batch size: {config_data['batch_size']}\n")
            f.write(f"   Learning rate: {config_data['learning_rate']}\n")
            f.write(f"   Data augmentation: {config_data['data_augmentation']}\n")
            f.write(f"   Grayscale: {config_data['grayscale']}\n\n")
            
            f.write("🎯 Quantization Settings:\n")
            f.write(f"   Quantization enabled: {config_data['quantization_enabled']}\n")
            f.write(f"   QAT enabled: {config_data['qat_enabled']}\n")
            f.write(f"   ESP-DL enabled: {config_data['esp_dl_enabled']}\n\n")
            
            f.write("📊 Results:\n")
            f.write(f"   Best validation accuracy: {config_data['best_validation_accuracy']:.4f}\n")
            f.write(f"   Test accuracy (Keras): {config_data['test_accuracy_keras']:.4f}\n")
            f.write(f"   Test accuracy (TFLite): {config_data['test_accuracy_tflite']:.4f}\n")
            f.write(f"   Keras model size: {config_data['model_size_keras_kb']:.1f} KB\n")
            f.write(f"   TFLite model size: {config_data['model_size_tflite_kb']:.1f} KB\n")
            
            if 'size_reduction_percent' in config_data:
                f.write(f"   Size reduction: {config_data['size_reduction_percent']:.1f}%\n")
        
        print(f"💾 Training configuration saved to: {config_path}")
        
        # Also save as JSON for programmatic access
        import json
        json_path = os.path.join(training_dir, "training_config.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
    except UnicodeEncodeError:
        # Fallback: write without emojis
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("TRAINING CONFIGURATION AND RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Training Information:\n")
            f.write(f"   Timestamp: {config_data['timestamp']}\n")
            f.write(f"   Duration: {config_data['training_duration']}\n\n")
            
            f.write("Model Configuration:\n")
            f.write(f"   Architecture: {config_data['model_architecture']}\n")
            f.write(f"   Input shape: {config_data['input_shape']}\n")
            f.write(f"   Number of classes: {config_data['num_classes']}\n")
            f.write(f"   Total parameters: {config_data['total_parameters']:,}\n\n")
            
            f.write("Training Parameters:\n")
            f.write(f"   Epochs: {config_data['epochs']}\n")
            f.write(f"   Batch size: {config_data['batch_size']}\n")
            f.write(f"   Learning rate: {config_data['learning_rate']}\n")
            f.write(f"   Data augmentation: {config_data['data_augmentation']}\n")
            f.write(f"   Grayscale: {config_data['grayscale']}\n\n")
            
            f.write("Quantization Settings:\n")
            f.write(f"   Quantization enabled: {config_data['quantization_enabled']}\n")
            f.write(f"   QAT enabled: {config_data['qat_enabled']}\n")
            f.write(f"   ESP-DL enabled: {config_data['esp_dl_enabled']}\n\n")
            
            f.write("Results:\n")
            f.write(f"   Best validation accuracy: {config_data['best_validation_accuracy']:.4f}\n")
            f.write(f"   Test accuracy (Keras): {config_data['test_accuracy_keras']:.4f}\n")
            f.write(f"   Test accuracy (TFLite): {config_data['test_accuracy_tflite']:.4f}\n")
            f.write(f"   Keras model size: {config_data['model_size_keras_kb']:.1f} KB\n")
            f.write(f"   TFLite model size: {config_data['model_size_tflite_kb']:.1f} KB\n")
            
            if 'size_reduction_percent' in config_data:
                f.write(f"   Size reduction: {config_data['size_reduction_percent']:.1f}%\n")
        
        print(f"💾 Training configuration (no emojis) saved to: {config_path}")
    
    return config_data

def save_training_csv(training_dir, history):
    """Save training history to CSV file"""
    import pandas as pd
    
    if history and hasattr(history, 'history'):
        csv_path = os.path.join(training_dir, "training_log.csv")
        
        # Create DataFrame from history
        history_df = pd.DataFrame(history.history)
        history_df['epoch'] = range(1, len(history_df) + 1)
        
        # Reorder columns to have epoch first
        cols = ['epoch'] + [col for col in history_df.columns if col != 'epoch']
        history_df = history_df[cols]
        
        history_df.to_csv(csv_path, index=False)
        print(f"💾 Training log saved to: {csv_path}")
        
        return history_df
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Adaptive Loss Controllers
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveFocalLossController(tf.keras.callbacks.Callback):
    """
    Dynamically adjust Focal Loss based on validation accuracy thresholds.
    """
    def __init__(self, 
                 accuracy_thresholds=[0.80, 0.90, 0.95],
                 gamma_values=[1.0, 1.5, 2.0],
                 alpha=0.45,
                 min_epochs_between_switches=5,
                 patience=3,
                 debug=False,
                 **kwargs):
        # Intercept val_ds if passed in kwargs before calling super
        self.val_ds = kwargs.pop('val_ds', None)
        super().__init__() # Don't pass kwargs to Keras Callback init
        self.thresholds = accuracy_thresholds
        self.gammas = gamma_values
        self.alpha = alpha
        self.min_epochs = min_epochs_between_switches
        self.patience = patience
        self.debug = debug
        
        # State tracking
        self.current_gamma = 0.0  # Start with CrossEntropy (effectively)
        self.last_switch_epoch = 0
        self.threshold_reached_epoch = {}
        
    def on_train_begin(self, logs=None):
        print("\n🎯 Adaptive Focal Loss Controller initialized")
        print(f"   Thresholds: {self.thresholds}")
        print(f"   Gamma values: {self.gammas}")
        print(f"   Starting with current model loss")
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
            
        val_acc = logs.get('val_accuracy', 0)
        
        # Check each threshold
        for i, threshold in enumerate(self.thresholds):
            target_gamma = self.gammas[i]
            
            # If we've reached this threshold and not yet switched to this or higher gamma
            if val_acc >= threshold and self.current_gamma < target_gamma:
                
                # Initialize tracking for this threshold
                if threshold not in self.threshold_reached_epoch:
                    self.threshold_reached_epoch[threshold] = epoch
                    if self.debug:
                        print(f"\n📈 Epoch {epoch}: Reached {threshold:.2f} val_acc! "
                              f"Waiting {self.patience} epochs to confirm...")
                    continue
                
                # Check if we've waited long enough
                epochs_since_reached = epoch - self.threshold_reached_epoch[threshold]
                
                if epochs_since_reached >= self.patience:
                    # Check minimum epochs since last switch
                    if epoch - self.last_switch_epoch >= self.min_epochs:
                        self._switch_to_focal_loss(epoch, target_gamma, threshold)
                        break
        
        # Log current state
        if self.debug and epoch % 5 == 0:
            print(f"   Epoch {epoch}: val_acc={val_acc:.3f}, current_gamma={self.current_gamma}")
    
    def _switch_to_focal_loss(self, epoch, new_gamma, threshold):
        """Perform the actual loss function switch"""
        print(f"\n🔄 Epoch {epoch}: Switching to Focal Loss (γ={new_gamma:.1f})")
        print(f"   Trigger: val_acc reached {threshold:.2f}")
        
        # Store old optimizer config (to preserve LR and state)
        old_lr = K.get_value(self.model.optimizer.learning_rate)
        
        # Create new loss based on architecture
        from utils.losses import sparse_focal_loss, focal_loss
        if params.MODEL_ARCHITECTURE == "original_haverland":
            new_loss_fn = focal_loss(gamma=new_gamma, alpha=self.alpha)
        else:
            new_loss_fn = sparse_focal_loss(gamma=new_gamma, alpha=self.alpha)
        
        # Recompile with new loss but keep optimizer instance
        self.model.compile(
            optimizer=self.model.optimizer,
            loss=new_loss_fn,
            metrics=['accuracy']
        )
        
        # Keras 3: optimizer state should be preserved if we pass the same instance
        # but manual LR set is safer
        K.set_value(self.model.optimizer.learning_rate, old_lr)
        
        # Update state
        self.current_gamma = new_gamma
        self.last_switch_epoch = epoch
        
        print(f"   ✅ Successfully switched to γ={new_gamma:.1f}")
        print(f"   📊 Learning rate maintained at {old_lr:.6f}")

class IntelligentFocalLossController(AdaptiveFocalLossController):
    """
    Enhanced controller that detects plateaus AND adjusts gamma & alpha dynamically.
    """
    def __init__(self, 
                 accuracy_thresholds=[0.80, 0.90, 0.95],
                 gamma_values=[1.0, 1.5, 2.0],
                 plateau_patience=5,
                 plateau_min_delta=0.001,
                 **kwargs):
        super().__init__(accuracy_thresholds, gamma_values, **kwargs)
        self.plateau_patience = plateau_patience
        self.plateau_min_delta = plateau_min_delta
        self.best_acc = 0
        self.plateau_count = 0
        self.gamma_history = []
        
        # val_ds is already handled by super().__init__ which pops it from kwargs
        
        # Dynamic Alpha Scaling - Base alpha on class count
        nb_classes = params.NB_CLASSES
        self.base_alpha = min(0.45, max(0.25, 0.45 * (10/nb_classes)**0.3))
        self.current_alpha = self.base_alpha
        
        if self.debug:
            print(f"🎯 Dynamic Alpha Base: {self.base_alpha:.4f} (for {nb_classes} classes)")

    def _get_per_class_accuracy(self):
        """Helper to compute per-class accuracy for alpha adjustment"""
        # Note: This is a heavy operation, we only do it on plateau or major switch
        y_true_all, y_pred_all = [], []
        # We need validation data here - if not passed to controller, we can't do per-class
        if not hasattr(self, 'val_ds') or self.val_ds is None:
            return None

        for x_batch, y_batch in self.val_ds:
            preds = self.model.predict(x_batch, verbose=0)
            y_pred_all.append(np.argmax(preds, axis=-1))
            if len(y_batch.shape) > 1 and y_batch.shape[-1] > 1:
                y_true_all.append(np.argmax(y_batch, axis=-1))
            else:
                y_true_all.append(np.cast[np.int32](y_batch)) # Simplified
                
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        
        per_class_acc = []
        for i in range(params.NB_CLASSES):
            mask = (y_true == i)
            if np.sum(mask) > 0:
                per_class_acc.append(np.mean(y_pred[mask] == y_true[mask]))
            else:
                per_class_acc.append(1.0)
        return np.array(per_class_acc)

    def on_epoch_end(self, epoch, logs=None):
        # First do standard threshold check
        super().on_epoch_end(epoch, logs)
        
        if logs is None:
            return
            
        val_acc = logs.get('val_accuracy', 0)
        
        # Track gamma for debugging
        self.gamma_history.append((epoch, self.current_gamma))
        
        # Check for plateau
        if val_acc > self.best_acc + self.plateau_min_delta:
            self.best_acc = val_acc
            self.plateau_count = 0
        else:
            self.plateau_count += 1
        
        # If plateau detected AND we haven't switched recently
        if (self.plateau_count >= self.plateau_patience and 
            epoch - self.last_switch_epoch >= self.min_epochs):
            
            if self.debug:
                print(f"\n⚠️ Plateau detected at {val_acc:.3f} for {self.plateau_count} epochs")
            
            # 1. Try to increase gamma if thresholds allow
            switched = False
            for i, threshold in enumerate(self.thresholds):
                if val_acc >= threshold and self.current_gamma < self.gammas[i]:
                    self._switch_to_focal_loss(epoch, self.gammas[i], 
                                              f"plateau at {val_acc:.3f}")
                    self.plateau_count = 0
                    switched = True
                    break
            
            # 2. If no gamma switch possible, try to adjust Alpha per-class
            if not switched and hasattr(self, 'val_ds') and self.val_ds is not None:
                per_class_acc = self._get_per_class_accuracy()
                if per_class_acc is not None:
                    # Inverse accuracy weights
                    class_weights = 1.0 / (per_class_acc + 0.001)
                    class_weights = class_weights / class_weights.mean()
                    
                    # Adjust alpha per class
                    self.current_alpha = self.base_alpha * class_weights
                    if self.debug:
                        print(f"⚖️  Adjusting Alpha per-class (max weight: {np.max(class_weights):.2f}x)")
                    
                    # Force recompilation with same gamma but new alpha
                    self._switch_to_focal_loss(epoch, self.current_gamma, "alpha adjustment")
                    self.plateau_count = 0

# ──────────────────────────────────────────────────────────────────────────────
# Per-Class Accuracy Callback
# ──────────────────────────────────────────────────────────────────────────────

class PerClassAccuracyCallback(tf.keras.callbacks.Callback):
    """Prints per-class accuracy on validation set and updates dynamic loss weights."""

    def __init__(self, val_ds, every_n_epochs=5, debug=False):
        super().__init__()
        self.val_ds = val_ds
        self.every_n = every_n_epochs
        self.debug = debug
        self.nb_classes = params.NB_CLASSES

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n != 0:
            return

        print(f"\n📊 [Epoch {epoch+1}] Detailed Validation Report:")
        
        y_true_all = []
        y_pred_all = []
        
        for x_batch, y_batch in self.val_ds:
            preds = self.model.predict(x_batch, verbose=0)
            y_pred_all.append(np.argmax(preds, axis=-1))
            
            # Handle both sparse and one-hot labels
            if len(y_batch.shape) > 1 and y_batch.shape[-1] > 1:
                y_true_all.append(np.argmax(y_batch, axis=-1))
            else:
                y_true_all.append(y_batch.numpy().flatten())
                
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        
        # Header
        print(f"{'Class':<5} | {'Name':<15} | {'Samples':<8} | {'Accuracy':<10}")
        print("-" * 50)
        
        # Load labels if available
        label_names = params.get_label_names()
        
        accuracies = []
        for i in range(self.nb_classes):
            mask = (y_true == i)
            samples = np.sum(mask)
            if samples > 0:
                acc = np.mean(y_pred[mask] == y_true[mask])
                accuracies.append(acc)
                name = label_names[i] if i < len(label_names) else f"Class {i}"
                print(f"{i:<5} | {name[:15]:<15} | {int(samples):<8} | {acc:.4f}")
            else:
                accuracies.append(1.0) # No samples, don't penalize
        
        overall_acc = np.mean(y_pred == y_true)
        print("-" * 50)
        print(f"OVERALL ACCURACY: {overall_acc:.4f}")