# utils/train_helpers.py
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.keras import backend as K
import parameters as params

def normalize_validation_data(data, batch_size=32):
    """
    Ensure validation data is a tf.data.Dataset for consistent iteration.
    Handles (x, y) tuples, lists, and existing datasets.
    """
    if data is None:
        return None
    if isinstance(data, tf.data.Dataset):
        return data
    
    # If it's a tuple/list of (x, y)
    if isinstance(data, (tuple, list)) and len(data) >= 2:
        try:
            # from_tensor_slices expects a tuple of tensors
            ds = tf.data.Dataset.from_tensor_slices((data[0], data[1]))
            return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        except Exception as e:
            print(f"⚠️  Failed to convert validation tuple to dataset: {e}")
            return data
            
    return data

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
        
        # --- Expanded Learning Configuration ---
        'loss_type': params.LOSS_TYPE,
        'label_smoothing': params.LABEL_SMOOTHING,
        'optimizer_type': params.OPTIMIZER_TYPE,
        'weight_decay': getattr(params, f"{params.OPTIMIZER_TYPE.upper()}_WEIGHT_DECAY", 0.0) if hasattr(params, f"{params.OPTIMIZER_TYPE.upper()}_WEIGHT_DECAY") else 0.0,
        'l1_reg': params.L1_REGULARIZATION,
        'l2_reg': params.L2_REGULARIZATION,
        'dropout_rate': params.DEFAULT_DROPOUT_RATE,
        'use_batch_norm': params.USE_BATCH_NORM,
        'lr_scheduler': params.LR_SCHEDULER_TYPE if params.USE_LEARNING_RATE_SCHEDULER else "None",
        'dynamic_weighting': params.USE_DYNAMIC_WEIGHTS,
        # ----------------------------------------

        'best_validation_accuracy': float(tflite_manager.best_accuracy),
        'test_accuracy_keras': float(test_accuracy),
        'test_accuracy_tflite': float(tflite_accuracy),
        'model_size_keras_kb': float(keras_size),
        'model_size_tflite_kb': float(tflite_size),
        'total_parameters': model.count_params() if model else 0
    }
    
    # Add Focal Loss specific params if applicable
    if "focal" in params.LOSS_TYPE.lower():
        config_data['focal_gamma'] = params.FOCAL_GAMMA
        config_data['focal_alpha'] = params.FOCAL_ALPHA
        if hasattr(params, 'FOCAL_ACCURACY_THRESHOLDS'):
            config_data['focal_thresholds'] = params.FOCAL_ACCURACY_THRESHOLDS

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
            
            f.write("🧠 Learning Mode & Parameters:\n")
            f.write(f"   Loss Type: {config_data['loss_type']}\n")
            f.write(f"   Optimizer: {config_data['optimizer_type']}\n")
            if config_data['label_smoothing'] > 0:
                f.write(f"   Label Smoothing: {config_data['label_smoothing']}\n")
            if config_data['weight_decay'] > 0:
                f.write(f"   Weight Decay: {config_data['weight_decay']}\n")
            
            if 'focal_gamma' in config_data:
                f.write(f"   Focal Gamma: {config_data['focal_gamma']}\n")
                f.write(f"   Focal Alpha: {config_data['focal_alpha']}\n")
                if 'focal_thresholds' in config_data:
                    f.write(f"   Thresholds: {config_data['focal_thresholds']}\n")
            
            f.write(f"   Learning Rate: {config_data['learning_rate']}\n")
            f.write(f"   LR Scheduler: {config_data['lr_scheduler']}\n")
            f.write(f"   L1/L2 Reg: {config_data['l1_reg']}/{config_data['l2_reg']}\n")
            f.write(f"   Dropout: {config_data['dropout_rate']}\n")
            f.write(f"   Batch Norm: {config_data['use_batch_norm']}\n")
            f.write(f"   Dynamic Weights: {config_data['dynamic_weighting']}\n\n")

            f.write("⚙️  Training Context:\n")
            f.write(f"   Epochs: {config_data['epochs']}\n")
            f.write(f"   Batch size: {config_data['batch_size']}\n")
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
            
            f.write("Learning Mode & Parameters:\n")
            f.write(f"   Loss Type: {config_data['loss_type']}\n")
            f.write(f"   Optimizer: {config_data['optimizer_type']}\n")
            if config_data['label_smoothing'] > 0:
                f.write(f"   Label Smoothing: {config_data['label_smoothing']}\n")
            if config_data['weight_decay'] > 0:
                f.write(f"   Weight Decay: {config_data['weight_decay']}\n")
            
            if 'focal_gamma' in config_data:
                f.write(f"   Focal Gamma: {config_data['focal_gamma']}\n")
                f.write(f"   Focal Alpha: {config_data['focal_alpha']}\n")
                if 'focal_thresholds' in config_data:
                    f.write(f"   Thresholds: {config_data['focal_thresholds']}\n")
            
            f.write(f"   Learning Rate: {config_data['learning_rate']}\n")
            f.write(f"   LR Scheduler: {config_data['lr_scheduler']}\n")
            f.write(f"   L1/L2 Reg: {config_data['l1_reg']}/{config_data['l2_reg']}\n")
            f.write(f"   Dropout: {config_data['dropout_rate']}\n")
            f.write(f"   Batch Norm: {config_data['use_batch_norm']}\n")
            f.write(f"   Dynamic Weights: {config_data['dynamic_weighting']}\n\n")

            f.write("Training Context:\n")
            f.write(f"   Epochs: {config_data['epochs']}\n")
            f.write(f"   Batch size: {config_data['batch_size']}\n")
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
# LR Warm-up Callback
# ──────────────────────────────────────────────────────────────────────────────

class LRWarmupCallback(tf.keras.callbacks.Callback):
    """
    Linearly ramps the learning rate from `initial_lr * initial_scale` up to
    `initial_lr` over `warmup_epochs` epochs, then deactivates itself.

    Prevents high-variance gradient updates on cold-start when LR=1e-3 hits
    freshly-initialised weights.  After the warm-up window, ReduceLROnPlateau
    (or whichever scheduler is configured) takes full control.

    Parameters read from `parameters.py` when not supplied explicitly:
      USE_LR_WARMUP, LR_WARMUP_EPOCHS, LR_WARMUP_INITIAL_SCALE
    """

    def __init__(self, initial_lr=None, warmup_epochs=None, initial_scale=None):
        super().__init__()
        self.initial_lr = initial_lr or getattr(params, 'LEARNING_RATE', 1e-3)
        self.warmup_epochs = warmup_epochs or getattr(params, 'LR_WARMUP_EPOCHS', 5)
        self.initial_scale = initial_scale or getattr(params, 'LR_WARMUP_INITIAL_SCALE', 0.1)
        self._active = True  # disabled after warm-up completes

    def on_train_begin(self, logs=None):
        # Set to scaled start value immediately
        start_lr = self.initial_lr * self.initial_scale
        self._set_lr(start_lr)
        print(f"\n🌡️  LR Warm-up active: {start_lr:.2e} → {self.initial_lr:.2e} "
              f"over {self.warmup_epochs} epochs")

    def on_epoch_begin(self, epoch, logs=None):
        if not self._active:
            return
        if epoch < self.warmup_epochs:
            # Linear interpolation: epoch 0 → initial_scale, epoch warmup_epochs → 1.0
            progress = epoch / self.warmup_epochs  # 0.0 … <1.0
            scale = self.initial_scale + (1.0 - self.initial_scale) * progress
            lr = self.initial_lr * scale
            self._set_lr(lr)
        else:
            # Warm-up complete — restore full LR and deactivate
            self._set_lr(self.initial_lr)
            self._active = False
            print(f"\n✅ LR Warm-up complete at epoch {epoch}. "
                  f"LR restored to {self.initial_lr:.2e}")

    def _set_lr(self, lr):
        if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
            if hasattr(self.model.optimizer.learning_rate, 'assign'):
                self.model.optimizer.learning_rate.assign(float(lr))
            else:
                self.model.optimizer.learning_rate = float(lr)


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
        self.val_ds = normalize_validation_data(kwargs.pop('val_ds', None), batch_size=params.BATCH_SIZE)
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

        # γ ramp state — smooth transition instead of hard step
        self.gamma_ramp_epochs = getattr(params, 'FOCAL_GAMMA_RAMP_EPOCHS', 0)
        self._ramp_start_gamma = 0.0
        self._ramp_target_gamma = 0.0
        self._ramp_start_epoch = -1  # -1 means no ramp in progress
        
    def on_train_begin(self, logs=None):
        print("\n🎯 Focal Loss Base Controller initialized")
        print(f"   Thresholds: {self.thresholds}")
        print(f"   Gamma values: {self.gammas}")
        print(f"   Starting with current model loss")
        print("   ⏳ Loading dataset and optimizing computation graph...")
        print("      (This may take a minute for the first epoch)")
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        # Advance any in-progress γ ramp first (safe no-op when no ramp)
        self._tick_gamma_ramp(epoch)

        val_acc = logs.get('val_accuracy', 0)

        # While a ramp is in progress, skip threshold checks to avoid cascading switches
        if self._ramp_start_epoch >= 0:
            if self.debug:
                print(f"   Epoch {epoch}: γ ramp in progress, skipping threshold check")
            return
        
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
            print(f"   Epoch {epoch}: val_acc={val_acc:.3f}, current_gamma={self.current_gamma:.3f}")
    
    def _tick_gamma_ramp(self, epoch):
        """Called every epoch to advance an in-progress γ ramp."""
        if self._ramp_start_epoch < 0:
            return  # No ramp in progress

        elapsed = epoch - self._ramp_start_epoch
        if elapsed >= self.gamma_ramp_epochs:
            # Ramp complete — snap to target
            interp_gamma = self._ramp_target_gamma
            self._ramp_start_epoch = -1  # Mark ramp done
        else:
            # Linear interpolation
            progress = elapsed / self.gamma_ramp_epochs
            interp_gamma = self._ramp_start_gamma + progress * (self._ramp_target_gamma - self._ramp_start_gamma)

        # Apply to the live loss variable (no recompile)
        from utils.losses import DynamicSparseFocalLoss, DynamicFocalLoss
        loss_obj = self.model.loss
        if isinstance(loss_obj, (DynamicSparseFocalLoss, DynamicFocalLoss)):
            loss_obj.gamma.assign(float(interp_gamma))
            self.current_gamma = interp_gamma
            if self.debug:
                print(f"   🎚️  γ ramp: epoch {epoch}, γ={interp_gamma:.3f} "
                      f"({elapsed}/{self.gamma_ramp_epochs} steps)")

    def _switch_to_focal_loss(self, epoch, new_gamma, threshold):
        """Perform the actual loss function switch by updating variables.
        When gamma_ramp_epochs > 0, kicks off a linear ramp instead of a hard step."""
        if self.gamma_ramp_epochs > 0:
            print(f"\n🔄 Epoch {epoch}: Starting γ ramp → {new_gamma:.1f} "
                  f"(over {self.gamma_ramp_epochs} epochs)")
        else:
            print(f"\n🔄 Epoch {epoch}: Switching to Focal Loss (γ={new_gamma:.1f})")
        
        # Link to the configuration parameters for transparency
        idx = -1
        try:
            # Check internal lists (self.thresholds) for a match
            t_list = getattr(self, 'thresholds', [])
            for i, val in enumerate(t_list):
                if abs(float(val) - float(threshold)) < 0.0001:
                    idx = i
                    break
        except:
            pass

        if idx != -1:
            why = f"val_acc {threshold:.2f} met FOCAL_ACCURACY_THRESHOLDS[{idx}]"
            how = f"Picked FOCAL_GAMMA_VALUES[{idx}] (γ={new_gamma:.1f})"
        elif isinstance(threshold, (int, float)):
            why = f"Validation accuracy reached {threshold:.2f}"
            how = f"Switching to γ={new_gamma:.1f}"
        else:
            why = f"Trigger: {threshold}"
            how = f"Adjusting γ to {new_gamma:.1f}"
            
        print(f"   Why: {why}")
        print(f"   How: {how}")
        
        # Check if model has the dynamic loss object
        # Note: In Keras 3, model.loss might be the object or a wrapper
        loss_obj = self.model.loss
        
        # We need to find the dynamic loss object if it's wrapped or in a list
        # Check for our dynamic loss types
        from utils.losses import DynamicSparseFocalLoss, DynamicFocalLoss
        
        target_loss = None
        if isinstance(loss_obj, (DynamicSparseFocalLoss, DynamicFocalLoss)):
            target_loss = loss_obj
        
        if target_loss:
            if self.gamma_ramp_epochs > 0:
                # Kick off a smooth ramp — don't assign yet, let _tick_gamma_ramp drive it
                self._ramp_start_gamma = float(self.current_gamma)
                self._ramp_target_gamma = float(new_gamma)
                self._ramp_start_epoch = epoch
                # We still need to update alpha immediately
                if isinstance(self.alpha, (list, tuple, np.ndarray)):
                    target_loss.alpha.assign(np.array(self.alpha, dtype=np.float32))
                else:
                    nb_classes = params.NB_CLASSES
                    target_loss.alpha.assign(np.ones(nb_classes, dtype=np.float32) * float(self.alpha))
                print(f"   🎚️  γ ramp initiated: {self._ramp_start_gamma:.2f} → {new_gamma:.1f} "
                      f"over {self.gamma_ramp_epochs} epochs")
            else:
                # Hard switch (original behaviour, ramp_epochs=0)
                target_loss.gamma.assign(float(new_gamma))
                
                # Update alpha (handle both scalar and per-class vector)
                if isinstance(self.alpha, (list, tuple, np.ndarray)):
                    target_loss.alpha.assign(np.array(self.alpha, dtype=np.float32))
                else:
                    nb_classes = params.NB_CLASSES
                    target_loss.alpha.assign(np.ones(nb_classes, dtype=np.float32) * float(self.alpha))
                
                print(f"   ✅ Successfully updated γ to {new_gamma:.1f} (No model re-compile)")
        else:
            print(f"   ⚠️  Model loss is not a DynamicFocalLoss instance ({type(loss_obj)}).")
            print("      Falling back to legacy recompile method (WARNING: may crash in Keras 3)")
            
            # Store old learning rate
            if hasattr(self.model.optimizer.learning_rate, 'numpy'):
                old_lr = self.model.optimizer.learning_rate.numpy()
            else:
                old_lr = float(self.model.optimizer.learning_rate)
            
            # Create new loss based on architecture
            from utils.losses import sparse_focal_loss, focal_loss
            if params.MODEL_ARCHITECTURE == "original_haverland":
                new_loss_fn = focal_loss(gamma=new_gamma, alpha=self.alpha)
            else:
                new_loss_fn = sparse_focal_loss(gamma=new_gamma, alpha=self.alpha)
            
            # Recompile
            self.model.compile(
                optimizer=self.model.optimizer,
                loss=new_loss_fn,
                metrics=['accuracy']
            )
            
            # Restore learning rate
            if hasattr(self.model.optimizer.learning_rate, 'assign'):
                self.model.optimizer.learning_rate.assign(old_lr)
            else:
                self.model.optimizer.learning_rate = old_lr
            
            print(f"   ✅ Legacy recompile successful (γ={new_gamma:.1f})")

        # Update state (for ramp, current_gamma will be updated each epoch by _tick_gamma_ramp)
        if self.gamma_ramp_epochs == 0:
            self.current_gamma = new_gamma
        self.last_switch_epoch = epoch

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
        self.alpha = self.base_alpha # Ensure base class uses our dynamic alpha
        self.current_alpha = self.base_alpha
        
        if self.debug:
            print(f"🎯 Intelligent Focal Loss initialized:")
            print(f"   Dynamic Alpha Base: {self.base_alpha:.4f} (for {nb_classes} classes)")
            print(f"   Plateau Patience  : {self.plateau_patience}")

    def on_train_begin(self, logs=None):
        print("\n🎯 Intelligent Focal Loss Controller initialized")
        print(f"   Thresholds      : {self.thresholds}")
        print(f"   Gamma values    : {self.gammas}")
        print(f"   Plateau Patience: {self.plateau_patience}")
        print(f"   Dynamic Alpha   : Enabled (Base={self.current_alpha:.4f})")
        ramp_info = f"{self.gamma_ramp_epochs} epochs" if self.gamma_ramp_epochs > 0 else "disabled (hard switch)"
        print(f"   γ Ramp          : {ramp_info}")
        print("   ⏳ Optimizing computation graph for adaptive weighting...")

    def _get_per_class_accuracy(self):
        """Helper to compute per-class accuracy for alpha adjustment"""
        # Note: This is a heavy operation, we only do it on plateau or major switch
        y_true_all, y_pred_all = [], []
        # We need validation data here - if not passed to controller, we can't do per-class
        if not hasattr(self, 'val_ds') or self.val_ds is None:
            return None

        from tqdm import tqdm
        print(f"📊 Analyzing per-class accuracy for {params.NB_CLASSES} classes...")
        # Try to get data length for tqdm
        data_len = None
        if hasattr(self.val_ds, "cardinality"):
            card = self.val_ds.cardinality().numpy()
            if card > 0:
                data_len = card
        elif hasattr(self.val_ds, "__len__"):
            data_len = len(self.val_ds)
        
        with tqdm(total=data_len, desc="Evaluating classes", leave=False) as pbar:
            for x_batch, y_batch in self.val_ds:
                preds = self.model.predict(x_batch, verbose=0)
                y_pred_all.append(np.argmax(preds, axis=-1))
                if len(y_batch.shape) > 1 and y_batch.shape[-1] > 1:
                    y_true_all.append(np.argmax(y_batch, axis=-1))
                else:
                    y_true_all.append(y_batch.numpy().astype(np.int32)) # Simplified
                pbar.update(1)
                
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
                    self.alpha = self.current_alpha # Sync for base class switch method
                    
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
        self.val_ds = normalize_validation_data(val_ds, batch_size=params.BATCH_SIZE)
        self.every_n = every_n_epochs
        self.debug = debug
        self.nb_classes = params.NB_CLASSES

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n != 0:
            return

        print(f"\n📊 [Epoch {epoch+1}] Detailed Validation Report:")
        
        y_true_all = []
        y_pred_all = []
        
        from tqdm import tqdm
        # Try to get data length for tqdm
        data_len = None
        if hasattr(self.val_ds, "cardinality"):
            card = self.val_ds.cardinality().numpy()
            if card > 0:
                data_len = card
        elif hasattr(self.val_ds, "__len__"):
            data_len = len(self.val_ds)

        with tqdm(total=data_len, desc="Validation Report", leave=False) as pbar:
            for x_batch, y_batch in self.val_ds:
                preds = self.model.predict(x_batch, verbose=0)
                y_pred_all.append(np.argmax(preds, axis=-1))
                
                # Handle both sparse and one-hot labels
                if len(y_batch.shape) > 1 and y_batch.shape[-1] > 1:
                    y_true_all.append(np.argmax(y_batch, axis=-1))
                else:
                    # Convert to numpy and flatten
                    if hasattr(y_batch, "numpy"):
                        y_true_all.append(y_batch.numpy().flatten())
                    else:
                        y_true_all.append(np.array(y_batch).flatten())
                pbar.update(1)
                
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        
        # Header
        print(f"{'Class':<5} | {'Name':<15} | {'Validator Samples':<18} | {'Accuracy':<10}")
        print("-" * 60)
        
        accuracies = []
        for i in range(self.nb_classes):
            mask = (y_true == i)
            samples = np.sum(mask)
            if samples > 0:
                acc = np.mean(y_pred[mask] == y_true[mask])
                accuracies.append(acc)
                # Use index directly as name for simplicity (0-9 or 0-99)
                print(f"{i:<5} | {i:<15} | {int(samples):<18} | {acc:.4f}")
            else:
                accuracies.append(1.0) # No samples, don't penalize
        
        overall_acc = np.mean(y_pred == y_true)
        print("-" * 50)
        print(f"OVERALL ACCURACY: {overall_acc:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic LR Scheduler Controller
# ──────────────────────────────────────────────────────────────────────────────

class DynamicLRProxy:
    """
    Mutable wrapper around an LR schedule function.

    A single LearningRateScheduler callback is registered with this proxy at
    training start.  When DynamicSchedulerController triggers a phase switch,
    it simply updates `self.schedule` to a new callable — the Keras callback
    picks up the change on the very next epoch_begin without any structural
    modification to the callback list.

    `self.epoch_offset` is reset on each switch so that the new schedule
    always counts from epoch 0 regardless of how many epochs have elapsed.
    """
    def __init__(self):
        self._schedule = lambda epoch, lr: lr  # pass-through (no-op)
        self.epoch_offset = 0                  # subtracted from global epoch

    def __call__(self, epoch, lr):
        return self._schedule(epoch - self.epoch_offset, lr)

    def update(self, new_schedule_fn, current_epoch):
        """Replace the active schedule and reset the epoch origin."""
        self._schedule = new_schedule_fn
        self.epoch_offset = current_epoch


class DynamicSchedulerController(tf.keras.callbacks.Callback):
    """
    Switches the LR scheduler (and optionally the optimizer) at val_accuracy
    thresholds, mirroring how IntelligentFocalLossController switches γ.

    Parameters are read from parameters.py:
        LR_SCHEDULER_THRESHOLDS  – list of val_acc values that trigger a switch
        LR_SCHEDULER_SEQUENCE    – scheduler name for each phase (len = thresholds+1)
        LR_SCHEDULER_RESET_FRACTION – restore LR to this × LEARNING_RATE on switch
                                      (None = keep the current decayed LR)
        USE_DYNAMIC_OPTIMIZER    – also swap optimizer on each switch (experimental)
        OPTIMIZER_SEQUENCE       – optimizer name per phase (same length as sequence)
    """

    def __init__(self, lr_proxy, debug=False):
        super().__init__()
        self.lr_proxy = lr_proxy
        self.debug = debug

        self.thresholds = list(getattr(params, 'LR_SCHEDULER_THRESHOLDS', []))
        self.scheduler_sequence = list(getattr(params, 'LR_SCHEDULER_SEQUENCE', []))
        self.reset_fraction = getattr(params, 'LR_SCHEDULER_RESET_FRACTION', None)
        self.use_dynamic_optimizer = getattr(params, 'USE_DYNAMIC_OPTIMIZER', False)
        self.optimizer_sequence = list(getattr(params, 'OPTIMIZER_SEQUENCE', []))

        self.current_phase = 0
        self.phase_switched_epoch = {}   # threshold → epoch it fired

    # ------------------------------------------------------------------
    def on_train_begin(self, logs=None):
        print("\n🔀 DynamicSchedulerController initialized")
        print(f"   Thresholds      : {self.thresholds}")
        print(f"   Scheduler phases: {self.scheduler_sequence}")
        if self.reset_fraction is not None:
            print(f"   LR reset        : {self.reset_fraction} × LEARNING_RATE on switch")
        if self.use_dynamic_optimizer:
            print(f"   Optimizer phases: {self.optimizer_sequence}")
        # Activate phase-0 schedule immediately
        self._activate_phase(0, epoch=0)

    # ------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        val_acc = logs.get('val_accuracy', 0.0)

        # Walk thresholds in order; only fire if still in the corresponding phase
        for i, threshold in enumerate(self.thresholds):
            target_phase = i + 1
            if val_acc >= threshold and self.current_phase < target_phase:
                if threshold not in self.phase_switched_epoch:
                    self.phase_switched_epoch[threshold] = epoch
                    print(f"\n🔀 Epoch {epoch+1}: val_acc {val_acc:.4f} ≥ {threshold} "
                          f"→ switching to phase {target_phase} "
                          f"({self.scheduler_sequence[target_phase]})")
                    self._activate_phase(target_phase, epoch + 1)
                break   # only one switch per epoch

    # ------------------------------------------------------------------
    def _activate_phase(self, phase, epoch):
        """Build + install the scheduler for the given phase."""
        if phase >= len(self.scheduler_sequence):
            return

        sched_type = self.scheduler_sequence[phase]
        peak_lr = params.LEARNING_RATE

        # Optionally restore LR before building the new schedule
        if self.reset_fraction is not None and phase > 0:
            new_lr = float(peak_lr * self.reset_fraction)
            self._set_lr(new_lr)
            print(f"   ↺ LR reset to {new_lr:.2e} ({self.reset_fraction} × {peak_lr:.2e})")
        else:
            new_lr = self._get_lr()

        # Build the schedule function for this phase
        sched_fn = self._build_schedule(sched_type, new_lr, epoch)

        # Hotswap via the proxy
        self.lr_proxy.update(sched_fn, epoch)
        self.current_phase = phase
        print(f"   ✅ Phase {phase}: '{sched_type}' scheduler active (base_lr={new_lr:.2e})")

        # Optional optimizer switch
        if self.use_dynamic_optimizer and phase < len(self.optimizer_sequence):
            self._switch_optimizer(self.optimizer_sequence[phase], new_lr)

    # ------------------------------------------------------------------
    def _build_schedule(self, sched_type, base_lr, start_epoch):
        """Return an (epoch, lr) → lr callable for the requested scheduler."""
        if sched_type == 'reduce_on_plateau':
            # ReduceLROnPlateau is a Keras callback, not a schedule function.
            # We approximate it here as a pass-through; the actual ReduceLROnPlateau
            # callback in the list continues to operate unimpeded.
            return lambda epoch, lr: lr

        elif sched_type == 'cosine':
            remaining = max(1, params.EPOCHS - start_epoch)
            first_decay = max(1, getattr(params, 'LR_WARMUP_EPOCHS', 15))
            schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=base_lr,
                first_decay_steps=first_decay,
                t_mul=2.0,
                m_mul=0.9,
                alpha=getattr(params, 'COSINE_DECAY_ALPHA', 1e-6),
            )
            return lambda epoch, lr: float(schedule(epoch))

        elif sched_type == 'exponential':
            schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=base_lr,
                decay_steps=getattr(params, 'EXPONENTIAL_DECAY_STEPS', 1000),
                decay_rate=getattr(params, 'EXPONENTIAL_DECAY_RATE', 0.96),
            )
            return lambda epoch, lr: float(schedule(epoch))

        elif sched_type == 'onecycle':
            total = max(1, params.EPOCHS - start_epoch)
            warmup = max(1, int(total * 0.3))
            decay = max(1, total - warmup)
            min_lr = getattr(params, 'COSINE_DECAY_ALPHA', 1e-6)
            warmup_sched = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=base_lr * 0.1,
                decay_steps=warmup, end_learning_rate=base_lr, power=1.0,
            )
            cosine_sched = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=base_lr,
                decay_steps=decay,
                alpha=min_lr / base_lr,
            )
            def _onecycle(epoch, lr):
                return float(warmup_sched(epoch) if epoch < warmup else cosine_sched(epoch - warmup))
            return _onecycle

        elif sched_type == 'step':
            step_size = getattr(params, 'STEP_DECAY_STEP_SIZE', 10)
            gamma = getattr(params, 'STEP_DECAY_GAMMA', 0.1)
            return lambda epoch, lr: base_lr * (gamma ** (epoch // step_size))

        else:
            print(f"   ⚠️  Unknown scheduler type '{sched_type}' — keeping current LR")
            return lambda epoch, lr: lr

    # ------------------------------------------------------------------
    def _switch_optimizer(self, opt_type, lr):
        """Recompile the model with a new optimizer (experimental)."""
        print(f"   🔧 Switching optimizer → {opt_type} (lr={lr:.2e})")
        try:
            from models.model_factory import compile_model
            # Temporarily override params for recompile
            old_opt = params.OPTIMIZER_TYPE
            old_lr = params.LEARNING_RATE
            params.OPTIMIZER_TYPE = opt_type
            params.LEARNING_RATE = lr
            loss_type = 'categorical' if params.MODEL_ARCHITECTURE == 'original_haverland' else 'sparse'
            compile_model(self.model, loss_type=loss_type)
            params.OPTIMIZER_TYPE = old_opt
            params.LEARNING_RATE = old_lr
            print(f"   ✅ Optimizer switched to {opt_type}")
        except Exception as e:
            print(f"   ❌ Optimizer switch failed: {e}")

    # ------------------------------------------------------------------
    def _get_lr(self):
        try:
            lr = self.model.optimizer.learning_rate
            return float(lr.numpy() if hasattr(lr, 'numpy') else lr)
        except Exception:
            return params.LEARNING_RATE

    def _set_lr(self, lr):
        try:
            opt_lr = self.model.optimizer.learning_rate
            if hasattr(opt_lr, 'assign'):
                opt_lr.assign(float(lr))
            else:
                self.model.optimizer.learning_rate = float(lr)
        except Exception as e:
            print(f"   ⚠️  Could not set LR: {e}")