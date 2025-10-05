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


# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Digit Recognition Training')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable TensorFlow debug logs and verbose output')
    parser.add_argument('--test_all_models', action='store_true',
                       help='Test all available model architectures and compare performance')
    return parser.parse_args()

# Set TensorFlow logging level based on debug flag
def setup_tensorflow_logging(debug=False):
    if debug:
        # Enable all TensorFlow logs
        tf.get_logger().setLevel('INFO')
        tf.autograph.set_verbosity(3)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        # Suppress TensorFlow info and warning messages
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        # Also suppress other noisy libraries
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # Suppress absl logging
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
        
def analyze_quantization_impact(model, x_test, y_test, tflite_path):
    """Analyze why quantization reduces accuracy - handles both label types"""
    print("\n🔍 QUANTIZATION ANALYSIS")
    print("=" * 50)
    
    sample_indices = np.random.choice(len(x_test), 100, replace=False)
    x_sample = x_test[sample_indices]
    y_sample = y_test[sample_indices]
    
    # Keras model predictions
    keras_predictions = model.predict(x_sample, verbose=0)
    keras_classes = np.argmax(keras_predictions, axis=1)
    
    # Handle both categorical and sparse labels
    if len(y_sample.shape) == 2 and y_sample.shape[1] == params.NB_CLASSES:
        # Categorical labels (one-hot) - convert to class indices
        true_classes = np.argmax(y_sample, axis=1)
    else:
        # Sparse labels (already class indices)
        true_classes = y_sample
    
    keras_accuracy = np.mean(keras_classes == true_classes)
    
    # TFLite model predictions WITH PROPER DEQUANTIZATION
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    tflite_predictions = []
    
    for i in range(len(x_sample)):
        test_image = x_sample[i:i+1]
        
        # Handle input quantization
        if input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = input_details[0]['quantization']
            test_image = test_image / input_scale + input_zero_point
            test_image = test_image.astype(np.int8)
        else:
            test_image = test_image.astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # PROPER dequantization
        if output_details[0]['dtype'] == np.int8:
            output_scale, output_zero_point = output_details[0]['quantization']
            output = output.astype(np.float32)
            output = (output - output_zero_point) * output_scale
        
        tflite_predictions.append(output[0])
    
    tflite_classes = np.argmax(tflite_predictions, axis=1)
    tflite_accuracy = np.mean(tflite_classes == true_classes)
    
    print(f"Sample set accuracy:")
    print(f"  Keras:  {keras_accuracy:.4f}")
    print(f"  TFLite: {tflite_accuracy:.4f}")
    print(f"  Difference: {keras_accuracy - tflite_accuracy:.4f}")
    
    # Check where predictions differ
    differing_indices = np.where(keras_classes != tflite_classes)[0]
    print(f"  Differing predictions: {len(differing_indices)}/{len(x_sample)}")
    
    if len(differing_indices) > 0:
        print(f"  Example differences:")
        for i in differing_indices[:3]:  # Show first 3 differences
            keras_conf = np.max(keras_predictions[i])
            tflite_conf = np.max(tflite_predictions[i])
            print(f"    Sample {i}: Keras={keras_classes[i]}, TFLite={tflite_classes[i]}, True={true_classes[i]}")
            print(f"    Keras conf: {keras_conf:.3f}, TFLite conf: {tflite_conf:.3f}")
            
def evaluate_tflite_model_universal(tflite_path, x_test, y_test):
    """Universal TFLite evaluation that handles both label formats and proper dequantization"""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    predictions = []
    
    for i in tqdm(range(len(x_test)), desc="Evaluating TFLite model", leave=False):
        test_image = x_test[i:i+1]
        
        # Handle quantization
        if input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = input_details[0]['quantization']
            test_image = test_image / input_scale + input_zero_point
            test_image = test_image.astype(np.int8)
        else:
            test_image = test_image.astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # FIX: PROPER output dequantization
        if output_details[0]['dtype'] == np.int8:
            output_scale, output_zero_point = output_details[0]['quantization']
            # Convert INT8 output back to float
            output = output.astype(np.float32)
            output = (output - output_zero_point) * output_scale
        
        predictions.append(np.argmax(output[0]))
    
    # Convert y_test to sparse labels if it's categorical
    if len(y_test.shape) == 2 and y_test.shape[1] == params.NB_CLASSES:
        # It's categorical (one-hot), convert to sparse
        true_labels = np.argmax(y_test, axis=1)
    else:
        # It's already sparse
        true_labels = y_test
    
    accuracy = np.mean(np.array(predictions) == true_labels)
    return accuracy

@contextmanager
def suppress_all_output(debug=False):
    """Completely suppress all output during TFLite conversion"""
    if debug:
        # Don't suppress anything in debug mode
        yield
        return
    
    # Redirect all possible output streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Open null devices for stdout and stderr
    with open(os.devnull, 'w') as fnull:
        sys.stdout = fnull
        sys.stderr = fnull
        
        # Also suppress C-level stdout/stderr
        original_stdout_fd = None
        original_stderr_fd = None
        
        try:
            # For Unix-like systems
            if hasattr(sys, '__stdout__'):
                original_stdout_fd = os.dup(sys.__stdout__.fileno())
                original_stderr_fd = os.dup(sys.__stderr__.fileno())
                
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull_fd, sys.__stdout__.fileno())
                os.dup2(devnull_fd, sys.__stderr__.fileno())
                os.close(devnull_fd)
        except (AttributeError, OSError):
            # Fallback for Windows or if above fails
            pass
        
        # Suppress all Python logging
        logging.disable(logging.CRITICAL)
        
        try:
            yield
        finally:
            # Restore everything
            logging.disable(logging.NOTSET)
            
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            # Restore C-level stdout/stderr
            try:
                if original_stdout_fd is not None and original_stderr_fd is not None:
                    os.dup2(original_stdout_fd, sys.__stdout__.fileno())
                    os.dup2(original_stderr_fd, sys.__stderr__.fileno())
                    os.close(original_stdout_fd)
                    os.close(original_stderr_fd)
            except (AttributeError, OSError):
                pass

class TFLiteModelManager:
    def __init__(self, output_dir, debug=False):
        self.output_dir = output_dir
        self.best_accuracy = 0.0
        self.debug = debug
        
    def save_as_tflite(self, model, filename, quantize=False, representative_data=None):
        """Save model directly as TFLite with compatibility fix"""
        try:
            # Build model with input shape first
            model.build((None,) + params.INPUT_SHAPE)
            
            # FIX: Use tf.function for conversion to avoid _get_save_spec issue
            @tf.function
            def model_call(x):
                return model(x)
            
            # Create concrete function from the model
            concrete_func = model_call.get_concrete_function(
                tf.TensorSpec([1] + list(params.INPUT_SHAPE), tf.float32)
            )
            
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model_call)
            
            if quantize:
                if self.debug:
                    print(f"🔧 Converting {filename} with INT8 quantization...")
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                if representative_data is not None:
                    converter.representative_dataset = representative_data
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                
                # Enable experimental features for better quantization
                converter.experimental_new_quantizer = True
                converter._experimental_disable_per_channel = False  # Enable per-channel quantization
                
            else:
                if self.debug:
                    print(f"🔧 Converting {filename} as float32...")
            
            converter.allow_custom_ops = False
            
            # Use comprehensive suppression context manager
            with suppress_all_output(self.debug):
                tflite_model = converter.convert()
            
            model_path = os.path.join(self.output_dir, filename)
            with open(model_path, 'wb') as f:
                f.write(tflite_model)
            
            model_size_kb = len(tflite_model) / 1024
            if self.debug:
                print(f"💾 Saved {filename}: {model_size_kb:.1f} KB")
            
            return tflite_model, model_size_kb
            
        except Exception as e:
            if self.debug:
                print(f"❌ TFLite conversion failed: {e}")
            # Fallback: try traditional conversion
            return self.save_as_tflite_fallback(model, filename, quantize, representative_data)
    
    def save_as_tflite_fallback(self, model, filename, quantize=False, representative_data=None):
        """Fallback conversion method"""
        try:
            # Alternative approach: save to SavedModel then convert
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save as SavedModel first
                tf.saved_model.save(model, temp_dir)
                
                # Convert from SavedModel
                converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
                
                if quantize:
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    if representative_data is not None:
                        converter.representative_dataset = representative_data
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
                
                # Use comprehensive suppression context manager
                with suppress_all_output(self.debug):
                    tflite_model = converter.convert()
                
                model_path = os.path.join(self.output_dir, filename)
                with open(model_path, 'wb') as f:
                    f.write(tflite_model)
                
                model_size_kb = len(tflite_model) / 1024
                if self.debug:
                    print(f"💾 Saved {filename} (fallback): {model_size_kb:.1f} KB")
                
                return tflite_model, model_size_kb
                
        except Exception as e:
            if self.debug:
                print(f"❌ Fallback conversion also failed: {e}")
            raise
    
    def save_best_model(self, model, accuracy, representative_data=None):
        """Save model if it's the best so far - completely silent in non-debug mode"""
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            
            # Only show messages in debug mode
            if self.debug:
                print(f"🎯 New best accuracy: {accuracy:.4f}, saving TFLite model...")
            
            # Save quantized model (for ESP-DL)
            tflite_model, size_kb = self.save_as_tflite(
                model, 
                params.TFLITE_FILENAME, 
                quantize=True, 
                representative_data=representative_data
            )
            
            # Also save float model for comparison
            self.save_as_tflite(
                model, 
                params.FLOAT_TFLITE_FILENAME, 
                quantize=False
            )
            
            # Only show model sizes in debug mode
            if self.debug:
                quantized_path = os.path.join(self.output_dir, params.TFLITE_FILENAME)
                float_path = os.path.join(self.output_dir, params.FLOAT_TFLITE_FILENAME)
                if os.path.exists(quantized_path):
                    quantized_size = os.path.getsize(quantized_path) / 1024
                    float_size = os.path.getsize(float_path) / 1024
                    print(f"💾 Models saved - Quantized: {quantized_size:.1f} KB, Float: {float_size:.1f} KB")
            
            return size_kb
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
        
        # FIX: Use learning_rate instead of lr for TF 2.13+
        try:
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except AttributeError:
            # Fallback for older TF versions
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        
        self.lr_history.append(current_lr)

    def set_model(self, model):
        self.model = model

    def save_training_plots(self):
        """Save training history plots"""
        if not params.SAVE_TRAINING_PLOTS:
            return
            
        epochs = range(1, len(self.train_loss) + 1)
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(epochs, self.train_loss, 'b-', label='Training Loss', alpha=0.7)
        ax1.plot(epochs, self.val_loss, 'r-', label='Validation Loss', alpha=0.7)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(epochs, self.train_acc, 'b-', label='Training Accuracy', alpha=0.7)
        ax2.plot(epochs, self.val_acc, 'r-', label='Validation Accuracy', alpha=0.7)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot learning rate
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
    def __init__(self, tflite_manager, representative_data):
        super().__init__()
        self.tflite_manager = tflite_manager
        self.representative_data = representative_data
        
    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy', 0)
        self.tflite_manager.save_best_model(
            self.model, 
            val_accuracy, 
            self.representative_data
        )

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
        # Initialize tqdm progress bar
        self.pbar = tqdm(total=self.total_epochs, desc='Training', unit='epoch',
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                        position=0, leave=True)
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = datetime.now()
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = datetime.now() - self.epoch_start
        self.epoch_times.append(epoch_time)
        avg_time = np.mean(self.epoch_times) if self.epoch_times else epoch_time
        
        # Call the monitor to record metrics
        self.monitor.on_epoch_end(epoch, logs)
        
        # Get current metrics
        train_loss = logs.get('loss', 0)
        train_acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        
        # Calculate remaining time
        remaining_epochs = self.total_epochs - epoch - 1
        remaining_time = avg_time * remaining_epochs
        
        # Update progress bar description with current metrics
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
    """Create training callbacks"""
    
    callbacks = []
    
    # Early stopping
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
    
    # Learning rate scheduler with configurable parameters
    callbacks.extend([
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=getattr(params, 'LR_SCHEDULER_MONITOR', 'val_loss'),
            factor=getattr(params, 'LR_SCHEDULER_FACTOR', 0.5),
            patience=getattr(params, 'LR_SCHEDULER_PATIENCE', 3),
            min_lr=getattr(params, 'LR_SCHEDULER_MIN_LR', 1e-7),
            verbose=1 if debug else 0
        ),
        
        # TFLite model checkpoint
        TFLiteCheckpoint(tflite_manager, representative_data),
        
        # TQDM progress bar
        TQDMProgressBar(total_epochs, monitor, tflite_manager, debug),
        
        # CSV logger
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'training_log.csv'),
            separator=',',
            append=False
        )
    ])
    
    return callbacks

def create_representative_dataset(x_train, num_samples=params.QUANTIZE_NUM_SAMPLES):
    """Create representative dataset for quantization"""
    def representative_data_gen():
        for i in range(min(num_samples, len(x_train))):
            yield [x_train[i:i+1]]
    return representative_data_gen
    
def setup_gpu():
    """Comprehensive GPU configuration with better error reporting"""
    print("🔧 Configuring hardware...")
    
    # First, let's see what's available
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
        print("💡 Troubleshooting tips:")
        print("   1. Check if tensorflow-gpu is installed: pip list | grep tensorflow")
        print("   2. Verify CUDA/cuDNN installation")
        print("   3. Check nvidia-smi output")
        print("   4. Try: pip install tensorflow-gpu")
        return None
    
    try:
        print(f"🎮 Configuring {len(gpus)} GPU(s)...")
        
        # Configure memory settings
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
        
        # Test GPU functionality
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

def evaluate_tflite_model(tflite_path, x_test, y_test):
    """Universal TFLite evaluation that handles both label formats"""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    predictions = []
    
    for i in tqdm(range(len(x_test)), desc="Evaluating TFLite model", leave=False):
        test_image = x_test[i:i+1]
        
        # Handle quantization
        if input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = input_details[0]['quantization']
            test_image = test_image / input_scale + input_zero_point
            test_image = test_image.astype(np.int8)
        else:
            test_image = test_image.astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(np.argmax(output[0]))
    
    # Convert y_test to sparse labels if it's categorical
    if len(y_test.shape) == 2 and y_test.shape[1] == params.NB_CLASSES:
        # It's categorical (one-hot), convert to sparse
        true_labels = np.argmax(y_test, axis=1)
    else:
        # It's already sparse
        true_labels = y_test
    
    accuracy = np.mean(np.array(predictions) == true_labels)
    return accuracy
    

def print_training_summary(model, x_train, x_val, x_test, debug=False):
    """Print comprehensive training summary"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    print(f"Model Architecture:")
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
    print(f"  Validation split: {params.VALIDATION_SPLIT}")
    print(f"  Training percentage: {params.TRAINING_PERCENTAGE*100}%")
    print(f"  Quantization: {params.QUANTIZE_MODEL}")
    print(f"  Debug mode: {'Enabled' if debug else 'Disabled'}")

def train_model(debug=False):
    """Main training function"""
    # Set up TensorFlow logging based on debug flag
    setup_tensorflow_logging(debug)
    
    # Set up GPU
    print("🔧 Configuring hardware...")
    strategy = setup_gpu()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_dir = os.path.join(params.OUTPUT_DIR, f"training_{timestamp}")
    os.makedirs(training_dir, exist_ok=True)
    
    print("🚀 Starting Digit Recognition Training")
    if debug:
        print("🔍 DEBUG MODE ENABLED - Verbose logging active")
    print("="*60)
    
    # Load and preprocess data from multiple sources
    print("📊 Loading dataset from multiple sources...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_data_splits()
    
    # Ensure data is properly normalized
    print("🔍 Checking data normalization...")
    print(f"   Data range before preprocessing: [{x_train.min():.3f}, {x_train.max():.3f}]")
    
    
    print("🔄 Preprocessing images...")
    x_train = preprocess_images(x_train)
    x_val = preprocess_images(x_val) 
    x_test = preprocess_images(x_test)
    
    print(f"✅ Preprocessing complete:")
    print(f"   x_train: {x_train.shape}")
    print(f"   x_val: {x_val.shape}")
    print(f"   x_test: {x_test.shape}")
    
    print(f"   Data range after preprocessing: [{x_train.min():.3f}, {x_train.max():.3f}]")
    # If data isn't normalized to reasonable range, force it
    if x_train.max() > 5.0 or x_train.min() < -5.0:
        print("⚠️  Data range too large - applying normalization...")
        x_train = (x_train - x_train.mean()) / (x_train.std() + 1e-8)
        x_val = (x_val - x_val.mean()) / (x_val.std() + 1e-8)
        x_test = (x_test - x_test.mean()) / (x_test.std() + 1e-8)
        print(f"   Data range after normalization: [{x_train.min():.3f}, {x_train.max():.3f}]")
    
    
    # Convert labels for Haverland model (categorical instead of sparse)
    if params.MODEL_ARCHITECTURE == "original_haverland":
        y_train = tf.keras.utils.to_categorical(y_train, params.NB_CLASSES)
        y_val = tf.keras.utils.to_categorical(y_val, params.NB_CLASSES)
        y_test = tf.keras.utils.to_categorical(y_test, params.NB_CLASSES)
        print("🔧 Using categorical crossentropy for Haverland model")
    
    # Create representative dataset for quantization
    representative_data = create_representative_dataset(x_train)
    
    # Create and compile model with GPU strategy if available
    if strategy:
        print(f"🧠 Creating {params.MODEL_ARCHITECTURE} model with multi-GPU strategy...")
        with strategy.scope():
            model = create_model()
            model = compile_model(model)
    else:
        print(f"🧠 Creating {params.MODEL_ARCHITECTURE} model...")
        model = create_model()
        model = compile_model(model)
    
    # CRITICAL: Verify model is built and can forward pass
    print("🔍 Verifying model can forward pass...")
    try:
        test_input = tf.random.normal([1] + list(params.INPUT_SHAPE))
        test_output = model(test_input)
        print(f"✅ Model verification passed: input {test_input.shape} -> output {test_output.shape}")
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        raise
    
    # Initialize TFLite manager with debug flag
    tflite_manager = TFLiteModelManager(training_dir, debug)
    
    # Print comprehensive summary
    print_training_summary(model, x_train, x_val, x_test, debug)
    model_summary(model)
    
    # Create monitor with debug flag
    monitor = TrainingMonitor(training_dir, debug)
    monitor.set_model(model)
    
    # Create callbacks with tqdm progress bar
    callbacks = create_callbacks(training_dir, tflite_manager, representative_data, params.EPOCHS, monitor, debug)
    
    # Train model
    print("\n🎯 Starting training...")
    print("-" * 60)
    
    start_time = datetime.now()
    
    # Use the original training approach to maintain accuracy
    history = model.fit(
        x_train, y_train,
        batch_size=params.BATCH_SIZE,
        epochs=params.EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=0,  # We handle progress with tqdm
        shuffle=True
    )
    
    training_time = datetime.now() - start_time
    
    # Save final TFLite models (silently in non-debug mode)
    if debug:
        print("\n💾 Saving final TFLite models...")
    final_quantized, quantized_size = tflite_manager.save_as_tflite(
        model, "final_quantized.tflite", quantize=True, representative_data=representative_data
    )
    final_float, float_size = tflite_manager.save_as_tflite(
        model, "final_float.tflite", quantize=False
    )
    
    # Evaluate models
    print("\n📈 Evaluating models...")
    
    # Keras model evaluation with tqdm
    if debug:
        print("Evaluating Keras model...")
    train_accuracy = model.evaluate(x_train, y_train, verbose=0)[1]
    val_accuracy = model.evaluate(x_val, y_val, verbose=0)[1]
    test_accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
    
    # TFLite model evaluation
    quantized_tflite_path = os.path.join(training_dir, params.TFLITE_FILENAME)
    if os.path.exists(quantized_tflite_path):
        tflite_accuracy = evaluate_tflite_model(quantized_tflite_path, x_test, y_test)   
        # Analyze quantization impact
        analyze_quantization_impact(model, x_test, y_test, quantized_tflite_path)
    
    else:
        tflite_accuracy = 0.0
    
    # Save training plots
    monitor.save_training_plots()
    
    
    ## temporary debugs
    training_diagnostics(model, x_train, y_train, x_val, y_val, debug=args.debug)
    verify_model_predictions(model, x_train[:100], y_train[:100])
    debug_model_architecture(model, x_train[:10])
    
    # Print final results
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
    
    # Save training configuration
    save_training_config(training_dir, quantized_size, float_size, tflite_manager,
                        test_accuracy, tflite_accuracy, training_time, debug)
    
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
        
        f.write("MODEL OUTPUT:\n")
        f.write(f"  Quantized TFLite: {quantized_size:.1f} KB\n")
        f.write(f"  Float TFLite: {float_size:.1f} KB\n")
        f.write(f"  Best accuracy: {tflite_manager.best_accuracy:.4f}\n\n")
        
        f.write("DATA SOURCES:\n")
        for i, source in enumerate(params.DATA_SOURCES):
            f.write(f"  {i+1}. {source['name']} ({source['type']}) - weight: {source.get('weight', 1.0)}\n")
        
        f.write(f"\nMODEL ARCHITECTURE:\n")
        f.write(f"  Model: {params.MODEL_ARCHITECTURE}\n")
        f.write(f"  Input shape: {params.INPUT_SHAPE}\n")
        f.write(f"  Classes: {params.NB_CLASSES}\n")
        
        f.write(f"\nTRAINING CONFIG:\n")
        f.write(f"  Batch size: {params.BATCH_SIZE}\n")
        f.write(f"  Epochs: {params.EPOCHS}\n")
        f.write(f"  Learning rate: {params.LEARNING_RATE}\n")
        f.write(f"  Quantization: {params.QUANTIZE_MODEL}\n")
        f.write(f"  Debug mode: {'Enabled' if debug else 'Disabled'}\n")
        
        f.write(f"\nGENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
def training_diagnostics(model, x_train, y_train, x_val, y_val, debug=False):
    """Comprehensive training diagnostics - FIXED version"""
    print("\n🔬 TRAINING DIAGNOSTICS")
    print("=" * 50)
    
    # 1. Check if model can overfit a small dataset
    print("1. Testing if model can overfit small dataset...")
    small_x = x_train[:100]
    small_y = y_train[:100]
    
    # FIX: Determine label type and handle appropriately
    if len(small_y.shape) == 2 and small_y.shape[1] == params.NB_CLASSES:
        # Categorical labels (one-hot)
        label_type = "categorical"
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        # Sparse labels
        label_type = "sparse" 
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print(f"   Label type: {label_type}")
    
    # Train on small dataset
    history = model.fit(small_x, small_y, epochs=10, batch_size=32, verbose=0)
    small_acc = history.history['accuracy'][-1]
    print(f"   Small dataset accuracy after 10 epochs: {small_acc:.3f}")
    
    if small_acc < 0.8:
        print("   ⚠️  Model cannot overfit small dataset - architecture issue!")
    else:
        print("   ✅ Model can overfit small dataset - architecture OK")
    
    # 2. Check gradient flow - FIXED for both label types
    print("2. Checking gradient flow...")
    with tf.GradientTape() as tape:
        predictions = model(small_x)
        
        # FIX: Use appropriate loss calculation based on label type
        if label_type == "categorical":
            loss = tf.keras.losses.categorical_crossentropy(small_y, predictions)
        else:
            loss = tf.keras.losses.sparse_categorical_crossentropy(small_y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    grad_norms = [tf.norm(g).numpy() if g is not None else 0 for g in gradients]
    
    print(f"   Loss: {tf.reduce_mean(loss).numpy():.3f}")
    print(f"   Gradient norms - Min: {min(grad_norms):.2e}, Max: {max(grad_norms):.2e}")
    
    if max(grad_norms) < 1e-7:
        print("   ⚠️  Vanishing gradients detected!")
    elif min(grad_norms) > 1e3:
        print("   ⚠️  Exploding gradients detected!")
    else:
        print("   ✅ Gradient flow looks OK")
    
    # 3. Check data preprocessing
    print("3. Data preprocessing check...")
    print(f"   Input shape: {x_train.shape}")
    print(f"   Data type: {x_train.dtype}")
    print(f"   Value range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"   Mean: {x_train.mean():.3f}, Std: {x_train.std():.3f}")


def verify_model_predictions(model, x_sample, y_sample):
    """Verify model can learn at all - FIXED version"""
    print("\n🧪 MODEL VERIFICATION:")
    
    # Test on a small batch
    sample_idx = np.random.choice(len(x_sample), 10, replace=False)
    x_batch = x_sample[sample_idx]
    y_batch = y_sample[sample_idx]
    
    # Forward pass
    predictions = model.predict(x_batch, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    # FIX: Handle both categorical and sparse labels
    if len(y_batch.shape) == 2 and y_batch.shape[1] == params.NB_CLASSES:
        # Categorical labels - convert to class indices
        true_classes = np.argmax(y_batch, axis=1)
    else:
        # Sparse labels
        true_classes = y_batch
    
    accuracy = np.mean(pred_classes == true_classes)
    print(f"Sample batch accuracy: {accuracy:.3f}")
    
    # Check prediction distribution
    print(f"Prediction distribution: {np.bincount(pred_classes, minlength=params.NB_CLASSES)}")
    print(f"True distribution: {np.bincount(true_classes.astype(int), minlength=params.NB_CLASSES)}")
    
def debug_model_architecture(model, x_sample):
    """Debug the current model architecture - FIXED VERSION"""
    print("\n🔧 MODEL ARCHITECTURE DEBUG")
    print("=" * 40)
    
    # Ensure model is built
    if not model.built:
        print("🔄 Building model...")
        model.build((None,) + params.INPUT_SHAPE)
    
    # Test forward pass
    print("Testing forward pass...")
    try:
        sample_output = model(x_sample[:1])
        print(f"✅ Forward pass works")
        print(f"   Input shape: {x_sample[:1].shape}")
        print(f"   Output shape: {sample_output.shape}")
        print(f"   Output sum: {tf.reduce_sum(sample_output).numpy():.4f}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return
    
    # Layer by layer debugging - FIXED VERSION
    print("\nLayer-by-layer output shapes:")
    for i, layer in enumerate(model.layers):
        try:
            # Create temporary model that outputs from this layer
            temp_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
            layer_out = temp_model(x_sample[:1])
            print(f"   {layer.name:20} -> {layer_out.shape}")
        except Exception as e:
            print(f"   {layer.name:20} -> ERROR: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Set random seeds for reproducibility (same as original)
    tf.random.set_seed(params.SHUFFLE_SEED)
    np.random.seed(params.SHUFFLE_SEED)
    
    try:
        model, history, training_dir = train_model(debug=args.debug)
        
        print(f"\n✅ Training successful!")
        print(f"📁 Output directory: {training_dir}")
        print(f"🚀 Your model is ready for ESP-DL deployment!")
        print(f"   Use: {os.path.join(training_dir, params.TFLITE_FILENAME)}")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
def test_all_models(x_train, y_train, x_val, y_val):
    """Test all available model architectures"""
    original_model = params.MODEL_ARCHITECTURE
    results = {}
    
    test_models = ["practical_tiny_depthwise", "simple_cnn", "dig_class100_s2", "original_haverland"]
    
    for model_name in test_models:
        print(f"\n🧪 TESTING {model_name.upper()}")
        print("=" * 50)
        
        # Temporarily change model architecture
        params.MODEL_ARCHITECTURE = model_name
        
        try:
            # Create and compile model
            model = create_model()
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train briefly
            history = model.fit(
                x_train[:500], y_train[:500],
                validation_data=(x_val[:100], y_val[:100]),
                epochs=10,
                batch_size=32,
                verbose=0
            )
            
            final_val_acc = history.history['val_accuracy'][-1]
            results[model_name] = final_val_acc
            print(f"✅ {model_name}: {final_val_acc:.3f} validation accuracy")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            results[model_name] = 0.0
    
    # Restore original model
    params.MODEL_ARCHITECTURE = original_model
    
    # Print results
    print("\n🏆 MODEL COMPARISON RESULTS:")
    for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"   {model_name:25} -> {accuracy:.3f}")
    
    return results



# ADD GPU INFO TO TRAINING SUMMARY
def print_training_summary(model, x_train, x_val, x_test, debug=False):
    """Print comprehensive training summary"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    # GPU Information
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
    print(f"  Debug mode: {'Enabled' if debug else 'Disabled'}")
    
def get_gpu_memory_usage():
    """Get current GPU memory usage"""
    try:
        import subprocess
        result = subprocess.check_output([
            'nvidia-smi', '--query-gpu=memory.used,memory.total', 
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
        
        gpu_memory = []
        for line in result.strip().split('\n'):
            used, total = map(int, line.split(', '))
            gpu_memory.append({'used_mb': used, 'total_mb': total, 'usage_percent': (used/total)*100})
        
        return gpu_memory
    except:
        return None
        
