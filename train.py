# train.py
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys
from tqdm.auto import tqdm
import logging
from contextlib import contextmanager
from models import create_model, compile_model, model_summary
from utils import get_data_splits, preprocess_images
import parameters as params

# QAT imports
try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    print("⚠️  tensorflow-model-optimization not available. Install with: pip install tensorflow-model-optimization")
    QAT_AVAILABLE = False

def parse_arguments():
    parser = argparse.ArgumentParser(description='Digit Recognition Training')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable TensorFlow debug logs and verbose output')
    parser.add_argument('--test_all_models', action='store_true',
                       help='Test all available model architectures and compare performance')
    parser.add_argument('--train', nargs='+', choices=params.AVAILABLE_MODELS,
                       help='Train specific model architectures from AVAILABLE_MODELS')
    parser.add_argument('--train_all', action='store_true',
                       help='Train all available model architectures sequentially')
    parser.add_argument('--use_tuner', action='store_true',
                       help='Enable hyperparameter tuning before training')
    parser.add_argument('--advanced', action='store_true',
                       help='Enable advanced training features')
    parser.add_argument('--num_trials', type=int, default=5,
                       help='Number of tuning trials (default: 5)')
    return parser.parse_args()
    
def set_all_seeds(seed=params.SHUFFLE_SEED):
    """Set all random seeds for complete reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.experimental.enable_op_determinism()

def setup_tensorflow_logging(debug=False):
    if debug:
        tf.get_logger().setLevel('INFO')
        tf.autograph.set_verbosity(3)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)

@contextmanager
def suppress_all_output(debug=False):
    """Completely suppress all output during TFLite conversion"""
    if debug:
        yield
        return
    
    # ADD DATASET SUPPRESSION AT THE START
    original_tf_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    with open(os.devnull, 'w') as fnull:
        sys.stdout = fnull
        sys.stderr = fnull
        
        try:
            if hasattr(sys, '__stdout__'):
                original_stdout_fd = os.dup(sys.__stdout__.fileno())
                original_stderr_fd = os.dup(sys.__stderr__.fileno())
                
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull_fd, sys.__stdout__.fileno())
                os.dup2(devnull_fd, sys.__stderr__.fileno())
                os.close(devnull_fd)
        except (AttributeError, OSError):
            pass
        
        logging.disable(logging.CRITICAL)
        
        # ADD WARNING SUPPRESSION
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                yield
            finally:
                logging.disable(logging.NOTSET)
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                
                try:
                    if original_stdout_fd is not None and original_stderr_fd is not None:
                        os.dup2(original_stdout_fd, sys.__stdout__.fileno())
                        os.dup2(original_stderr_fd, sys.__stderr__.fileno())
                        os.close(original_stdout_fd)
                        os.close(original_stderr_fd)
                except (AttributeError, OSError):
                    pass
                
                # RESTORE TF LOG LEVEL
                if original_tf_log_level:
                    os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_tf_log_level
                else:
                    os.environ.pop('TF_CPP_MIN_LOG_LEVEL', None)

def apply_qat(model):
    """Apply Quantization Aware Training using modern TF API"""
    if not QAT_AVAILABLE:
        print("❌ QAT not available - install tensorflow-model-optimization")
        return model
    
    try:
        print("🎯 Applying Quantization Aware Training...")
        qat_model = tfmot.quantization.keras.quantize_model(model)
        print("✅ QAT applied successfully")
        return qat_model
    except Exception as e:
        print(f"❌ QAT failed: {e}")
        return model

def create_qat_model():
    """Create a model with Quantization Aware Training from scratch"""
    if not QAT_AVAILABLE:
        return create_model()
    
    try:
        print("🎯 Building model with Quantization Aware Training...")
        model = create_model()
        qat_model = tfmot.quantization.keras.quantize_model(model)
        print("✅ QAT model created successfully")
        return qat_model
    except Exception as e:
        print(f"❌ QAT model creation failed: {e}")
        return create_model()

class TFLiteModelManager:
    def __init__(self, output_dir, debug=False):
        self.output_dir = output_dir
        self.best_accuracy = 0.0
        self.debug = debug
        
    def verify_model_for_conversion(self, model):
        """Verify model is compatible with TFLite conversion"""
        try:
            test_input = tf.random.normal([1] + list(params.INPUT_SHAPE))
            test_output = model(test_input)
            
            expected_output_shape = (1, params.NB_CLASSES)
            if test_output.shape != expected_output_shape:
                print(f"⚠️  Output shape mismatch: {test_output.shape} vs {expected_output_shape}")
            
            if tf.reduce_any(tf.math.is_nan(test_output)):
                print("❌ Model output contains NaN values")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Model verification failed: {e}")
            return False
            
    def save_trainable_checkpoint(self, model, accuracy, epoch):
        """Save model in trainable format"""
        timestamp = datetime.now().strftime("%H%M%S")
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch:03d}_acc_{accuracy:.4f}_{timestamp}.keras")
        model.save(checkpoint_path)
        
        if self.debug:
            print(f"💾 Saved trainable checkpoint: {checkpoint_path}")
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            best_checkpoint_path = os.path.join(self.output_dir, "best_model.keras")
            model.save(best_checkpoint_path)
            if self.debug:
                print(f"🏆 New best model saved: {best_checkpoint_path}")
        
        return checkpoint_path
        
    def _is_qat_model(self, model):
        """Check if model is a QAT model"""
        for layer in model.layers:
            if hasattr(layer, 'quantize_config') or 'quant' in layer.name.lower():
                return True
        return False

    def _convert_qat_model(self, model, filename, representative_data=None):
        """Convert QAT model to TFLite with proper representative dataset"""
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # CRITICAL: Provide representative dataset for full integer quantization
            if representative_data is None:
                print("⚠️  No representative dataset provided for QAT conversion")
                # Create a simple representative dataset from the model input shape
                def default_representative_dataset():
                    for _ in range(100):
                        data = np.random.rand(1, *params.INPUT_SHAPE).astype(np.float32)
                        yield [data]
                converter.representative_dataset = default_representative_dataset
            else:
                converter.representative_dataset = representative_data
            
            if params.ESP_DL_QUANTIZE:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            else:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
            
            with suppress_all_output(self.debug):
                tflite_model = converter.convert()
            
            return self._save_tflite_file(tflite_model, filename, True)
            
        except Exception as e:
            print(f"❌ QAT conversion failed: {e}")
            # Fallback: try without full integer quantization
            return self._convert_qat_model_fallback(model, filename)
            
    def _convert_qat_model_fallback(self, model, filename):
        """Fallback conversion for QAT model without full integer quantization"""
        try:
            print("🔄 Trying fallback QAT conversion (dynamic range quantization)...")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # Don't set representative dataset - use dynamic range quantization
            
            with suppress_all_output(self.debug):
                tflite_model = converter.convert()
            
            return self._save_tflite_file(tflite_model, filename, True)
            
        except Exception as e:
            print(f"❌ Fallback QAT conversion failed: {e}")
            raise          
            
            
    def save_as_tflite(self, model, filename, quantize=False, representative_data=None):
        """Save model as TFLite with proper QAT handling"""
        try:
            if not model.built:
                dummy_input = tf.zeros([1] + list(params.INPUT_SHAPE), dtype=tf.float32)
                _ = model(dummy_input)
            
            if quantize and self._is_qat_model(model):
                # print("🎯 Converting QAT model to quantized TFLite...")
                return self._convert_qat_model(model, filename, representative_data)
            
            # print(f"🔧 Converting {filename} to TFLite...")
            return self.save_as_tflite_savedmodel(model, filename, quantize, representative_data)
            
        except Exception as e:
            print(f"❌ TFLite conversion failed: {e}")
            return self.save_as_tflite_savedmodel(model, filename, quantize, representative_data)

    def save_as_tflite_savedmodel(self, model, filename, quantize=False, representative_data=None):
        """Use SavedModel approach for conversion"""
        try:
            import tempfile
            
            with tempfile.TemporaryDirectory() as temp_dir:
                model_dir = os.path.join(temp_dir, "saved_model")
                model.save(model_dir, save_format='tf')
                
                converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
                
                if quantize:
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    if representative_data is not None:
                        converter.representative_dataset = representative_data
                    else:
                        # Create default representative dataset if none provided
                        def default_representative_dataset():
                            for _ in range(100):
                                data = np.random.rand(1, *params.INPUT_SHAPE).astype(np.float32)
                                yield [data]
                        converter.representative_dataset = default_representative_dataset
                    
                    if params.ESP_DL_QUANTIZE:
                        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                        converter.inference_input_type = tf.int8
                        converter.inference_output_type = tf.int8
                    else:
                        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
                        converter.inference_input_type = tf.uint8
                        converter.inference_output_type = tf.uint8
                
                with suppress_all_output(self.debug):
                    tflite_model = converter.convert()
                
                return self._save_tflite_file(tflite_model, filename, quantize)
                    
        except Exception as e:
            print(f"❌ SavedModel conversion failed: {e}")
            raise

    def _save_tflite_file(self, tflite_model, filename, quantize):
        """Save TFLite model to file"""
        model_path = os.path.join(self.output_dir, filename)
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size_kb = len(tflite_model) / 1024
        if self.debug:
            quant_type = "INT8" if (quantize and params.ESP_DL_QUANTIZE) else "UINT8" if quantize else "Float32"
            print(f"💾 Saved {filename} ({quant_type}): {model_size_kb:.1f} KB")
        
        return tflite_model, model_size_kb
    
    def save_best_model(self, model, accuracy, representative_data=None):
        """Save model if it's the best so far"""
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            
            if not self.verify_model_for_conversion(model):
                print("⚠️  Skipping TFLite conversion due to model issues")
                return None
                
            if self.debug:
                print(f"🎯 New best accuracy: {accuracy:.4f}, saving TFLite model...")
            
            try:
                tflite_model, size_kb = self.save_as_tflite(
                    model, 
                    params.TFLITE_FILENAME, 
                    quantize=True, 
                    representative_data=representative_data
                )
                
                self.save_as_tflite(
                    model, 
                    params.FLOAT_TFLITE_FILENAME, 
                    quantize=False
                )
                
                if self.debug:
                    quantized_path = os.path.join(self.output_dir, params.TFLITE_FILENAME)
                    float_path = os.path.join(self.output_dir, params.FLOAT_TFLITE_FILENAME)
                    if os.path.exists(quantized_path):
                        quantized_size = os.path.getsize(quantized_path) / 1024
                        float_size = os.path.getsize(float_path) / 1024
                        print(f"💾 Models saved - Quantized: {quantized_size:.1f} KB, Float: {float_size:.1f} KB")
                
                return size_kb
            except Exception as e:
                print(f"❌ TFLite conversion failed: {e}")
                return None
        return None

class TrainingMonitor:
    def __init__(self, output_dir, debug=False):
        self.output_dir = output_dir
        self.debug = debug
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.lr_history = []
        
    def on_epoch_end(self, epoch, logs):
        """Record training metrics"""
        self.train_loss.append(logs.get('loss', 0))
        self.val_loss.append(logs.get('val_loss', 0))
        self.train_acc.append(logs.get('accuracy', 0))
        self.val_acc.append(logs.get('val_accuracy', 0))
        
        try:
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except AttributeError:
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        
        self.lr_history.append(current_lr)

    def set_model(self, model):
        self.model = model

    def save_training_plots(self):
        """Save training history plots"""
        if not params.SAVE_TRAINING_PLOTS:
            return
            
        epochs = range(1, len(self.train_loss) + 1)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.plot(epochs, self.train_loss, 'b-', label='Training Loss', alpha=0.7)
        ax1.plot(epochs, self.val_loss, 'r-', label='Validation Loss', alpha=0.7)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, self.train_acc, 'b-', label='Training Accuracy', alpha=0.7)
        ax2.plot(epochs, self.val_acc, 'r-', label='Validation Accuracy', alpha=0.7)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(epochs, self.lr_history, 'g-', label='Learning Rate', alpha=0.7)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training plots saved to: {plot_path}")

class TFLiteCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, tflite_manager, representative_data, save_frequency=10):
        super().__init__()
        self.tflite_manager = tflite_manager
        self.representative_data = representative_data
        self.save_frequency = save_frequency
        self.last_save_epoch = -1
        
    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy', 0)
        
        if val_accuracy > getattr(self.tflite_manager, 'best_accuracy', 0):
            try:
                self.tflite_manager.save_best_model(
                    self.model, 
                    val_accuracy, 
                    self.representative_data
                )
            except Exception as e:
                if self.tflite_manager.debug:
                    print(f"⚠️  TFLite save failed: {e}")
        
        if epoch % self.save_frequency == 0 and epoch != self.last_save_epoch:
            try:
                checkpoint_path = self.tflite_manager.save_trainable_checkpoint(self.model, val_accuracy, epoch)
                if checkpoint_path and self.tflite_manager.debug:
                    print(f"💾 Saved checkpoint: {checkpoint_path}")
                self.last_save_epoch = epoch
            except Exception as e:
                if self.tflite_manager.debug:
                    print(f"⚠️  Checkpoint save failed: {e}")

class TQDMProgressBar(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, monitor, tflite_manager, debug=False):
        super().__init__()
        self.total_epochs = total_epochs
        self.monitor = monitor
        self.tflite_manager = tflite_manager
        self.debug = debug
        self.epoch_times = []
        self.pbar = None
        
    def on_train_begin(self, logs=None):
        self.pbar = tqdm(total=self.total_epochs, desc='Training', unit='epoch',
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                        position=0, leave=True)
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = datetime.now()
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = datetime.now() - self.epoch_start
        self.epoch_times.append(epoch_time)
        avg_time = np.mean(self.epoch_times) if self.epoch_times else epoch_time
        
        self.monitor.on_epoch_end(epoch, logs)
        
        train_loss = logs.get('loss', 0)
        train_acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        
        remaining_epochs = self.total_epochs - epoch - 1
        remaining_time = avg_time * remaining_epochs
        
        desc = (f"Epoch {epoch+1}/{self.total_epochs} | "
                f"loss: {train_loss:.4f} | acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | "
                f"ETA: {str(remaining_time).split('.')[0]}")
        
        self.pbar.set_description(desc)
        self.pbar.update(1)
        
    def on_train_end(self, logs=None):
        if self.pbar:
            self.pbar.close()

def create_callbacks(output_dir, tflite_manager, representative_data, total_epochs, monitor, debug=False):
    """Create training callbacks for Keras 3"""
    
    callbacks = []
    
    if params.USE_EARLY_STOPPING:
        mode = 'auto'
        if 'accuracy' in params.EARLY_STOPPING_MONITOR:
            mode = 'max'
        elif 'loss' in params.EARLY_STOPPING_MONITOR:
            mode = 'min'
            
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=params.EARLY_STOPPING_MONITOR,
                patience=params.EARLY_STOPPING_PATIENCE,
                min_delta=params.EARLY_STOPPING_MIN_DELTA,
                restore_best_weights=getattr(params, 'RESTORE_BEST_WEIGHTS', True),
                mode=mode,
                verbose=1 if debug else 0
            )
        )
    
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "checkpoints", "epoch_{epoch:03d}.keras"),
            save_freq='epoch',
            save_best_only=False,
            verbose=1 if debug else 0
        )
    )
    
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1 if debug else 0
        )
    )
    
    callbacks.extend([
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=getattr(params, 'LR_SCHEDULER_MONITOR', 'val_loss'),
            factor=getattr(params, 'LR_SCHEDULER_FACTOR', 0.5),
            patience=getattr(params, 'LR_SCHEDULER_PATIENCE', 3),
            min_lr=getattr(params, 'LR_SCHEDULER_MIN_LR', 1e-7),
            verbose=1 if debug else 0
        ),
        
        TFLiteCheckpoint(tflite_manager, representative_data),
        
        TQDMProgressBar(total_epochs, monitor, tflite_manager, debug),
        
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'training_log.csv'),
            separator=',',
            append=False
        )
    ])
    
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    return callbacks

def create_qat_representative_dataset(x_train, num_samples=100):
    """Create representative dataset for QAT model conversion"""
    def representative_dataset():
        # Use actual training data for better calibration
        for i in range(min(num_samples, len(x_train))):
            yield [x_train[i:i+1].astype(np.float32)]
    return representative_dataset
    
def setup_gpu():
    """Comprehensive GPU configuration"""
    print("🔧 Configuring hardware...")
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    cpus = tf.config.experimental.list_physical_devices('CPU')
    
    print(f"📋 Device inventory:")
    print(f"   CPUs found: {len(cpus)}")
    print(f"   GPUs found: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
    
    if not params.USE_GPU:
        print("🔧 GPU usage disabled in parameters - using CPU")
        return None
    
    if not gpus:
        print("❌ No GPUs detected by TensorFlow")
        return None
    
    try:
        print(f"🎮 Configuring {len(gpus)} GPU(s)...")
        
        for gpu in gpus:
            if params.GPU_MEMORY_GROWTH:
                tf.config.experimental.set_memory_growth(gpu, True)
                print("   ✅ Memory growth enabled")
            
            if params.GPU_MEMORY_LIMIT is not None:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=params.GPU_MEMORY_LIMIT
                    )]
                )
                print(f"   ✅ Memory limit set to {params.GPU_MEMORY_LIMIT} MB")
        
        print("   🧪 Testing GPU functionality...")
        with tf.device('/GPU:0'):
            test_tensor = tf.constant([1.0, 2.0, 3.0])
            result = test_tensor + 1.0
            print(f"   ✅ GPU computation test passed: {result.numpy()}")
        
        if len(gpus) > 1:
            print(f"   🚀 Using {len(gpus)} GPUs with MirroredStrategy")
            strategy = tf.distribute.MirroredStrategy()
            return strategy
        else:
            print("   ✅ Single GPU configured successfully")
            return None
            
    except Exception as e:
        print(f"⚠️  GPU configuration failed: {e}")
        print("   Falling back to CPU")
        return None
   
def print_training_summary(model, x_train, x_val, x_test, debug=False):
    """Print comprehensive training summary"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"Hardware:")
    print(f"  GPU: {'Available' if gpus else 'Not available'}")
    print(f"  GPU Usage: {'Enabled' if params.USE_GPU else 'Disabled'}")
    if gpus and params.USE_GPU:
        print(f"  GPU Count: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"    GPU {i}: {gpu.name}")
    
    print(f"\nModel Architecture:")
    print(f"  Model: {params.MODEL_ARCHITECTURE}")
    print(f"  Input shape: {params.INPUT_SHAPE}")
    print(f"  Classes: {params.NB_CLASSES}")
    print(f"  Total parameters: {model.count_params():,}")
    
    print(f"\nDataset Information:")
    print(f"  Training samples: {len(x_train):,}")
    print(f"  Validation samples: {len(x_val):,}")
    print(f"  Test samples: {len(x_test):,}")
    print(f"  Data sources: {len(params.DATA_SOURCES)}")
    
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {params.BATCH_SIZE}")
    print(f"  Epochs: {params.EPOCHS}")
    print(f"  Learning rate: {params.LEARNING_RATE}")
    print(f"  Early stopping: {'Enabled' if params.USE_EARLY_STOPPING else 'Disabled'}")
    print(f"  Quantization: {params.QUANTIZE_MODEL}")
    print(f"  QAT: {'Enabled' if (params.QUANTIZE_MODEL and getattr(params, 'USE_QAT', False)) else 'Disabled'}")
    print(f"  ESP-DL Quantization: {params.ESP_DL_QUANTIZE}")
    print(f"  Debug mode: {'Enabled' if debug else 'Disabled'}")

def train_model(debug=False):
    """Main training function with proper QAT workflow"""
    setup_tensorflow_logging(debug)
    set_all_seeds(params.SHUFFLE_SEED)
    
    print("🔧 Configuring hardware...")
    strategy = setup_gpu()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_dir = os.path.join(params.OUTPUT_DIR, f"{params.MODEL_ARCHITECTURE}_{timestamp}")
    os.makedirs(training_dir, exist_ok=True)
    
    print("🚀 Starting Digit Recognition Training")
    if debug:
        print("🔍 DEBUG MODE ENABLED - Verbose logging active")
    print("="*60)
    
    print("📊 Loading dataset from multiple sources...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_data_splits()
    
    print("🔄 Preprocessing images...")
    x_train = preprocess_images(x_train, for_training=True)
    x_val = preprocess_images(x_val, for_training=True)  
    x_test = preprocess_images(x_test, for_training=True)
    
    print(f"✅ Preprocessing complete - range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    
    if params.MODEL_ARCHITECTURE != "original_haverland":
        y_train = tf.keras.utils.to_categorical(y_train, params.NB_CLASSES)
        y_val = tf.keras.utils.to_categorical(y_val, params.NB_CLASSES) 
        y_test = tf.keras.utils.to_categorical(y_test, params.NB_CLASSES)
    
    representative_data = create_qat_representative_dataset(x_train)
    
    use_qat = params.QUANTIZE_MODEL and getattr(params, 'USE_QAT', False) and QAT_AVAILABLE
    
    if use_qat:
        if strategy:
            with strategy.scope():
                model = create_qat_model()
                model = compile_model(model)
        else:
            model = create_qat_model()
            model = compile_model(model)
    else:
        if strategy:
            with strategy.scope():
                model = create_model()
                model = compile_model(model)
        else:
            model = create_model()
            model = compile_model(model)
    
    print("🔧 Building model with explicit input shape...")
    model.build(input_shape=(None,) + params.INPUT_SHAPE)
    print(f"✅ Model built with input shape: {model.input_shape}")

    print("🔍 Verifying model can forward pass...")
    try:
        test_input = tf.random.normal([1] + list(params.INPUT_SHAPE))
        test_output = model(test_input)
        print(f"✅ Model verification passed: input {test_input.shape} -> output {test_output.shape}")
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        raise
    
    tflite_manager = TFLiteModelManager(training_dir, debug)
    monitor = TrainingMonitor(training_dir, debug)
    monitor.set_model(model)
    
    print_training_summary(model, x_train, x_val, x_test, debug)
    model_summary(model)
    
    callbacks = create_callbacks(training_dir, tflite_manager, representative_data, params.EPOCHS, monitor, debug)
    
    print("\n🎯 Starting training...")
    print("-" * 60)
    
    start_time = datetime.now()
    
    history = model.fit(
        x_train, y_train,
        batch_size=params.BATCH_SIZE,
        epochs=params.EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=0,
        shuffle=True
    )
    
    training_time = datetime.now() - start_time
    
    if debug:
        print("\n💾 Saving final TFLite models...")
    
    try:
        final_quantized, quantized_size = tflite_manager.save_as_tflite(
            model, "final_quantized.tflite", quantize=True, representative_data=representative_data
        )
    except Exception as e:
        print(f"❌ Quantized TFLite save failed: {e}")
        quantized_size = 0
    
    try:
        final_float, float_size = tflite_manager.save_as_tflite(
            model, "final_float.tflite", quantize=False
        )
    except Exception as e:
        print(f"❌ Float TFLite save failed: {e}")
        float_size = 0
    
    print("\n📈 Evaluating models...")
    
    try:
        from analyse import evaluate_tflite_model, analyze_quantization_impact, debug_tflite_model
    except ImportError:
        print("⚠️  Analysis module not available")
        evaluate_tflite_model = lambda *args: 0.0
        analyze_quantization_impact = lambda *args: None
        debug_tflite_model = lambda *args: None
    
    if debug:
        quantized_tflite_path = os.path.join(training_dir, params.TFLITE_FILENAME)
        if os.path.exists(quantized_tflite_path):
            debug_tflite_model(quantized_tflite_path, x_test[:1])
    
    try:
        train_accuracy = model.evaluate(x_train, y_train, verbose=0)[1]
        val_accuracy = model.evaluate(x_val, y_val, verbose=0)[1]
        test_accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
    except Exception as e:
        print(f"❌ Keras model evaluation failed: {e}")
        train_accuracy = val_accuracy = test_accuracy = 0.0
    
    tflite_accuracy = 0.0
    quantized_tflite_path = os.path.join(training_dir, params.TFLITE_FILENAME)
    if os.path.exists(quantized_tflite_path):
        try:
            tflite_accuracy = evaluate_tflite_model(quantized_tflite_path, x_test, y_test)
            analyze_quantization_impact(model, x_test, y_test, quantized_tflite_path)
        except Exception as e:
            print(f"❌ TFLite evaluation failed: {e}")
    
    monitor.save_training_plots()
    
    try:
        from analyse import training_diagnostics, verify_model_predictions, debug_model_architecture
        training_diagnostics(model, x_train, y_train, x_val, y_val, debug=debug)
        verify_model_predictions(model, x_train[:100], y_train[:100])
        
        if debug:
            debug_model_architecture(model, x_train[:10])
    except Exception as e:
        print(f"⚠️  Diagnostics failed: {e}")
    
    print("\n" + "="*60)
    print("🏁 TRAINING COMPLETED")
    print("="*60)
    print(f"⏱️  Training time: {training_time}")
    print(f"📊 Final Results:")
    print(f"   Keras Model - Test Accuracy: {test_accuracy:.4f}")
    print(f"   TFLite Model - Test Accuracy: {tflite_accuracy:.4f}")
    print(f"   Best Validation Accuracy: {tflite_manager.best_accuracy:.4f}")
    print(f"   Quantized Model Size: {quantized_size:.1f} KB")
    print(f"   Float Model Size: {float_size:.1f} KB")
    
    print(f"\n💾 Models saved to: {training_dir}")
    print(f"   Best quantized: {params.TFLITE_FILENAME}")
    print(f"   Final quantized: final_quantized.tflite")
    print(f"   Final float: final_float.tflite")
    print(f"   Training log: training_log.csv")
    print(f"   Training plot: training_history.png")
    
    save_training_config(training_dir, quantized_size, float_size, tflite_manager,
                        test_accuracy, tflite_accuracy, training_time, debug)
                        
    print("💾 Saving final model checkpoint...")
    final_checkpoint_path = os.path.join(training_dir, "final_model.keras")
    try:
        model.save(final_checkpoint_path)
        print(f"✅ Final model saved: {final_checkpoint_path}")
    except Exception as e:
        print(f"❌ Final model save failed: {e}")
    
    return model, history, training_dir

def save_training_config(training_dir, quantized_size, float_size, tflite_manager, 
                        test_accuracy, tflite_accuracy, training_time, debug=False):
    """Save training configuration and results to file"""
    config_path = os.path.join(training_dir, "training_config.txt")
    
    with open(config_path, 'w') as f:
        f.write("Digit Recognition Training Configuration\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("FINAL RESULTS:\n")
        f.write(f"  Keras Model Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"  TFLite Model Test Accuracy: {tflite_accuracy:.4f}\n")
        f.write(f"  Best Validation Accuracy: {tflite_manager.best_accuracy:.4f}\n")
        f.write(f"  Quantized Model Size: {quantized_size:.1f} KB\n")
        f.write(f"  Float Model Size: {float_size:.1f} KB\n")
        f.write(f"  Training Time: {training_time}\n\n")
        
        f.write("MODEL ARCHITECTURE:\n")
        f.write(f"  Model: {params.MODEL_ARCHITECTURE}\n")
        f.write(f"  Input shape: {params.INPUT_SHAPE}\n")
        f.write(f"  Classes: {params.NB_CLASSES}\n")
        
        f.write(f"\nTRAINING CONFIG:\n")
        f.write(f"  Batch size: {params.BATCH_SIZE}\n")
        f.write(f"  Epochs: {params.EPOCHS}\n")
        f.write(f"  Learning rate: {params.LEARNING_RATE}\n")
        f.write(f"  Early stopping: {'Enabled' if params.USE_EARLY_STOPPING else 'Disabled'}\n")
        f.write(f"  Quantization: {params.QUANTIZE_MODEL}\n")
        f.write(f"  QAT: {'Enabled' if (params.QUANTIZE_MODEL and getattr(params, 'USE_QAT', False)) else 'Disabled'}\n")
        f.write(f"  ESP-DL Quantization: {params.ESP_DL_QUANTIZE}\n")
        
        f.write(f"\nGENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def test_all_models(x_train, y_train, x_val, y_val, models_to_test=None, debug=False):
    """Test all available model architectures or specific models"""
    original_model = params.MODEL_ARCHITECTURE
    results = {}
    
    if models_to_test is None:
        test_models = params.AVAILABLE_MODELS
    else:
        test_models = models_to_test
    
    print(f"\n🧪 TESTING {len(test_models)} MODELS")
    print("=" * 60)
    
    for model_name in test_models:
        print(f"\n🔍 Testing: {model_name}")
        print("-" * 40)
        
        params.MODEL_ARCHITECTURE = model_name
        
        try:
            model = create_model()
            
            if model_name == "original_haverland":
                loss_fn = 'categorical_crossentropy'
                y_train_cat = tf.keras.utils.to_categorical(y_train, params.NB_CLASSES)
                y_val_cat = tf.keras.utils.to_categorical(y_val, params.NB_CLASSES)
            else:
                loss_fn = 'sparse_categorical_crossentropy'
                y_train_cat = y_train
                y_val_cat = y_val
            
            model.compile(
                optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy']
            )
            
            train_samples = min(1000, len(x_train))
            val_samples = min(200, len(x_val))
            
            history = model.fit(
                x_train[:train_samples], y_train_cat[:train_samples],
                validation_data=(x_val[:val_samples], y_val_cat[:val_samples]),
                epochs=5,
                batch_size=32,
                verbose=1 if debug else 0
            )
            
            final_val_acc = history.history['val_accuracy'][-1]
            final_train_acc = history.history['accuracy'][-1]
            results[model_name] = {
                'val_accuracy': final_val_acc,
                'train_accuracy': final_train_acc,
                'params': model.count_params()
            }
            print(f"✅ {model_name}:")
            print(f"   Train Accuracy: {final_train_acc:.4f}")
            print(f"   Val Accuracy: {final_val_acc:.4f}")
            print(f"   Parameters: {model.count_params():,}")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            results[model_name] = {
                'val_accuracy': 0.0,
                'train_accuracy': 0.0,
                'params': 0,
                'error': str(e)
            }
    
    params.MODEL_ARCHITECTURE = original_model
    
    print("\n" + "="*60)
    print("🏆 MODEL COMPARISON RESULTS:")
    print("="*60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['val_accuracy'], reverse=True)
    
    for i, (model_name, metrics) in enumerate(sorted_results, 1):
        if 'error' in metrics:
            print(f"{i:2d}. {model_name:35} -> ERROR: {metrics['error']}")
        else:
            print(f"{i:2d}. {model_name:35} -> Val: {metrics['val_accuracy']:.4f} | Train: {metrics['train_accuracy']:.4f} | Params: {metrics['params']:,}")
    
    return results

def main():
    """Main entry point"""
    args = parse_arguments()
    
    try:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_data_splits()
        
        if args.test_all_models:
            test_all_models(x_train, y_train, x_val, y_val, debug=args.debug)
        else:
            model, history, output_dir = train_model(debug=args.debug)
            print(f"\n✅ Training completed successfully!")
            print(f"📁 Output directory: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()