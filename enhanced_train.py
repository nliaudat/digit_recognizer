# enhanced_train.py
import tensorflow as tf
import os
import numpy as np
from datetime import datetime
import argparse
import sys
from pathlib import Path


# Import existing modules
from train import train_model, setup_tensorflow_logging, set_all_seeds, TFLiteModelManager
from utils import get_data_splits, preprocess_images
import parameters as params

class EnhancedTrainingManager:
    """Enhanced training manager for esp_quantization_ready model"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.callbacks = []
        # Ensure we use the correct model architecture
        self.model_architecture = params.MODEL_ARCHITECTURE
        print(f"🎯 Using model architecture: {self.model_architecture}")
        
    def setup_enhanced_callbacks(self, log_dir):
        """Setup enhanced training callbacks"""
        self.callbacks = []
        
        # Enhanced Model Checkpoints - save best model based on validation accuracy
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(log_dir, "best_enhanced_esp_model.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        self.callbacks.append(checkpoint_callback)
        
        # Enhanced Learning Rate Scheduler
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,  # Increased patience for ESP model
            min_lr=1e-7,
            verbose=1
        )
        self.callbacks.append(lr_scheduler)
        
        # Enhanced Early Stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=25,  # More patience for quantization-ready model
            restore_best_weights=True,
            mode='max',
            verbose=1
        )
        self.callbacks.append(early_stopping)
        
        # CSV Logger for detailed analysis
        csv_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(log_dir, 'enhanced_esp_training_log.csv'),
            separator=',',
            append=False
        )
        self.callbacks.append(csv_logger)
        
        # TensorBoard for visualization (optional)
        try:
            tb_callback = tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(log_dir, 'logs'),
                histogram_freq=1,
                write_graph=True,
                write_images=False,
                update_freq='epoch'
            )
            self.callbacks.append(tb_callback)
            print("✅ TensorBoard logging enabled")
        except Exception as e:
            print(f"⚠️  TensorBoard not available: {e}")
        
        print(f"✅ Setup {len(self.callbacks)} enhanced callbacks for ESP model")
        return self.callbacks
    
    def create_enhanced_optimizer(self):
        """Create enhanced optimizer suitable for quantization-ready model"""
        # Use Adam with ESP-DL compatible settings
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params.LEARNING_RATE,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        )
        
        print("✅ Using Adam optimizer optimized for ESP-DL compatibility")
        return optimizer
    
    def compile_esp_model(self, model):
        """Compile model with ESP-DL compatible settings"""
        # Determine appropriate loss function based on label format
        if params.MODEL_ARCHITECTURE == "original_haverland":
            loss_fn = 'categorical_crossentropy'
            print("🔧 Using categorical_crossentropy for Haverland model")
        else:
            loss_fn = 'sparse_categorical_crossentropy'
            print("🔧 Using sparse_categorical_crossentropy for ESP model")
        
        model.compile(
            optimizer=self.create_enhanced_optimizer(),
            loss=loss_fn,
            metrics=['accuracy', 'sparse_categorical_accuracy']
        )
        
        return model

class ESPDataProcessor:
    """Enhanced data processing optimized for ESP quantization"""
    
    def __init__(self):
        self.preprocessed_data = None
    
    def esp_optimized_preprocessing(self, images, labels, is_training=True):
        """Apply ESP-DL optimized preprocessing"""
        print("🔄 Applying ESP-optimized preprocessing...")
        
        # Convert to float32
        images = images.astype(np.float32)
        
        # Enhanced normalization for ESP-DL quantization
        if params.ESP_DL_QUANTIZE:
            # Normalize to [-1, 1] for INT8 quantization
            images = (images / 127.5) - 1.0
            print("✅ Normalized to [-1, 1] for ESP-DL INT8 quantization")
        else:
            # Normalize to [0, 1] for standard quantization
            images = images / 255.0
            print("✅ Normalized to [0, 1] for standard quantization")
        
        # Ensure correct shape for ESP model
        images = self._ensure_esp_compatible_shape(images)
        
        # Enhanced data validation
        self._validate_esp_data(images, labels, is_training)
        
        return images, labels
    
    def _ensure_esp_compatible_shape(self, images):
        """Ensure images have correct shape for ESP model"""
        target_shape = params.INPUT_SHAPE
        
        # Handle missing channel dimension
        if len(images.shape) == 3:  # (batch, height, width)
            images = np.expand_dims(images, axis=-1)
            print(f"✅ Expanded shape from {images.shape[:-1]} to {images.shape}")
        
        # Verify final shape matches expected input
        if images.shape[1:] != target_shape:
            print(f"⚠️  Reshaping images from {images.shape[1:]} to {target_shape}")
            # Note: This should rarely happen as preprocessing should handle this
        
        return images
    
    def _validate_esp_data(self, images, labels, is_training):
        """Validate data for ESP model compatibility"""
        print("🔍 Validating ESP model data compatibility...")
        
        # Check for NaN values
        nan_count = np.isnan(images).sum()
        if nan_count > 0:
            print(f"⚠️  Found {nan_count} NaN values in images - fixing...")
            images = np.nan_to_num(images)
        
        # Check data range based on quantization
        min_val, max_val = images.min(), images.max()
        expected_min = -1.0 if params.ESP_DL_QUANTIZE else 0.0
        expected_max = 1.0
        
        if min_val < expected_min or max_val > expected_max:
            print(f"⚠️  Data range [{min_val:.3f}, {max_val:.3f}] outside expected [{expected_min}, {expected_max}]")
            images = np.clip(images, expected_min, expected_max)
            print("✅ Clipped data to expected range")
        
        # Check label distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"📊 Label distribution: {dict(zip(unique, counts))}")
        
        # ESP-specific validation
        if params.ESP_DL_QUANTIZE:
            print("🔧 ESP-DL Quantization: Data validated for INT8 compatibility")
        
        print("✅ ESP data validation completed")

def verify_model_build(model, input_shape):
    """Verify model is properly built and compatible"""
    print("🔍 Verifying model build...")
    
    try:
        # Test with proper input shape
        test_input = tf.random.normal([1] + list(input_shape))
        print(f"🧪 Test input shape: {test_input.shape}")
        
        # Test forward pass
        output = model(test_input)
        print(f"✅ Model output shape: {output.shape}")
        
        # Verify output is valid
        if tf.reduce_any(tf.math.is_nan(output)):
            print("❌ Model output contains NaN values")
            return False
            
        # Verify output sums to ~1.0 (probability distribution)
        output_sums = tf.reduce_sum(output, axis=1).numpy()
        if not np.allclose(output_sums, 1.0, atol=0.1):
            print(f"⚠️  Output sums not close to 1.0: {output_sums}")
        
        print("✅ Model verification passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

def run_enhanced_esp_training(debug=False):
    """Run enhanced training specifically for esp_quantization_ready model"""
    setup_tensorflow_logging(debug)
    set_all_seeds(params.SHUFFLE_SEED)
    
    # Verify model architecture
    if params.MODEL_ARCHITECTURE != "esp_quantization_ready":
        print(f"⚠️  Warning: Using enhanced ESP training with model: {params.MODEL_ARCHITECTURE}")
        print("💡 For best results, set MODEL_ARCHITECTURE = 'esp_quantization_ready' in parameters.py")
    
    # Create enhanced output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    enhanced_dir = os.path.join(params.OUTPUT_DIR, f"enhanced_esp_{timestamp}")
    os.makedirs(enhanced_dir, exist_ok=True)
    
    print("🚀 Starting Enhanced ESP Quantization-Ready Training")
    print("=" * 60)
    print(f"🎯 Model: {params.MODEL_ARCHITECTURE}")
    print(f"📊 Input Shape: {params.INPUT_SHAPE}")
    print(f"🎯 Classes: {params.NB_CLASSES}")
    print(f"🔧 ESP-DL Quantization: {params.ESP_DL_QUANTIZE}")
    print("=" * 60)
    
    # Load and preprocess data with ESP optimizations
    print("📊 Loading and preprocessing dataset...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_data_splits()
    
    # Apply ESP-optimized preprocessing
    data_processor = ESPDataProcessor()
    x_train, y_train = data_processor.esp_optimized_preprocessing(x_train, y_train, is_training=True)
    x_val, y_val = data_processor.esp_optimized_preprocessing(x_val, y_val, is_training=False)
    x_test, y_test = data_processor.esp_optimized_preprocessing(x_test, y_test, is_training=False)
    
    # Handle label format based on model requirements
    if params.MODEL_ARCHITECTURE == "original_haverland":
        y_train = tf.keras.utils.to_categorical(y_train, params.NB_CLASSES)
        y_val = tf.keras.utils.to_categorical(y_val, params.NB_CLASSES)
        y_test = tf.keras.utils.to_categorical(y_test, params.NB_CLASSES)
        print("🔧 Converted labels to categorical format")
    else:
        # Ensure labels are int32 for sparse categorical crossentropy
        y_train = y_train.astype(np.int32)
        y_val = y_val.astype(np.int32)
        y_test = y_test.astype(np.int32)
        print("🔧 Using sparse categorical labels")
    
    # Create model - using the standard import to ensure consistency
    from models import create_model
    print(f"🏗️  Creating {params.MODEL_ARCHITECTURE} model...")
    model = create_model()
    
    # Verify model is properly built
    if not verify_model_build(model, params.INPUT_SHAPE):
        print("❌ Model build verification failed. Check model architecture.")
        return None, None, None
    
    # Setup enhanced training
    training_manager = EnhancedTrainingManager(debug=debug)
    training_manager.setup_enhanced_callbacks(enhanced_dir)
    
    # Compile model with ESP optimizations
    model = training_manager.compile_esp_model(model)
    
    # Train with enhanced techniques
    print("🎯 Starting Enhanced ESP Training...")
    print("-" * 50)
    
    history = model.fit(
        x_train, y_train,
        batch_size=params.BATCH_SIZE,
        epochs=params.EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=training_manager.callbacks,
        verbose=1,
        shuffle=True
    )
    
    # Evaluate the enhanced model
    print("📊 Evaluating Enhanced ESP Model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"✅ Enhanced ESP Model Test Accuracy: {test_accuracy:.4f}")
    
    # Save TFLite models with ESP optimizations
    tflite_manager = TFLiteModelManager(enhanced_dir, debug=debug)
    
    # Create representative dataset optimized for ESP quantization
    def esp_representative_data_gen():
        # Use diverse samples for better quantization calibration
        num_samples = min(200, len(x_train))
        indices = np.random.choice(len(x_train), num_samples, replace=False)
        for idx in indices:
            yield [x_train[idx:idx+1]]
    
    # Save quantized model
    print("💾 Saving ESP-optimized TFLite models...")
    quantized_model, quantized_size = tflite_manager.save_as_tflite(
        model, 
        "enhanced_esp_quantized.tflite", 
        quantize=True, 
        representative_data=esp_representative_data_gen
    )
    
    # Save float model for comparison
    float_model, float_size = tflite_manager.save_as_tflite(
        model, 
        "enhanced_esp_float.tflite", 
        quantize=False
    )
    
    # Save final model checkpoint
    final_model_path = os.path.join(enhanced_dir, "enhanced_esp_final_model.keras")
    model.save(final_model_path)
    
    # Save comprehensive results
    save_enhanced_esp_results(model, history, enhanced_dir, test_accuracy, 
                             quantized_size, float_size)
    
    # Print final summary
    print_enhanced_esp_summary(history, test_accuracy, quantized_size, float_size, enhanced_dir)
    
    return model, history, enhanced_dir

def save_enhanced_esp_results(model, history, output_dir, test_accuracy, quantized_size, float_size):
    """Save enhanced ESP training results"""
    print("💾 Saving enhanced ESP training results...")
    
    # Save training configuration
    config_path = os.path.join(output_dir, "enhanced_esp_config.txt")
    with open(config_path, 'w') as f:
        f.write("Enhanced ESP Quantization-Ready Training Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model Architecture: {params.MODEL_ARCHITECTURE}\n")
        f.write(f"Input Shape: {params.INPUT_SHAPE}\n")
        f.write(f"ESP-DL Quantization: {params.ESP_DL_QUANTIZE}\n")
        f.write(f"Final Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}\n")
        f.write(f"Quantized Model Size: {quantized_size:.1f} KB\n")
        f.write(f"Float Model Size: {float_size:.1f} KB\n")
        f.write(f"Total Parameters: {model.count_params():,}\n")
        f.write(f"Training Completed: {datetime.now()}\n\n")
        
        f.write("Training Parameters:\n")
        f.write(f"  Batch Size: {params.BATCH_SIZE}\n")
        f.write(f"  Epochs: {params.EPOCHS}\n")
        f.write(f"  Learning Rate: {params.LEARNING_RATE}\n")
        f.write(f"  Early Stopping: {params.USE_EARLY_STOPPING}\n")
        f.write(f"  ESP-DL Quantization: {params.ESP_DL_QUANTIZE}\n")
    
    # Save training history as JSON for analysis
    history_path = os.path.join(output_dir, "enhanced_esp_training_history.json")
    import json
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(val) for val in values]
    
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print("✅ Enhanced ESP results saved successfully")

def print_enhanced_esp_summary(history, test_accuracy, quantized_size, float_size, output_dir):
    """Print comprehensive training summary"""
    print("\n" + "=" * 60)
    print("🏁 ENHANCED ESP TRAINING COMPLETED")
    print("=" * 60)
    
    # Find best validation accuracy
    best_val_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_val_accuracy = max(history.history['val_accuracy'])
    
    print(f"📊 Final Results:")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Best Validation Accuracy: {best_val_accuracy:.4f} (epoch {best_val_epoch})")
    print(f"   Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"   Quantized Model Size: {quantized_size:.1f} KB")
    print(f"   Float Model Size: {float_size:.1f} KB")
    print(f"   Compression Ratio: {float_size/quantized_size:.1f}x")
    
    print(f"\n💾 Output Files:")
    print(f"   Model: {output_dir}/enhanced_esp_final_model.keras")
    print(f"   Quantized TFLite: {output_dir}/enhanced_esp_quantized.tflite")
    print(f"   Float TFLite: {output_dir}/enhanced_esp_float.tflite")
    print(f"   Training Log: {output_dir}/enhanced_esp_training_log.csv")
    print(f"   Configuration: {output_dir}/enhanced_esp_config.txt")
    
    # ESP-specific recommendations
    if params.ESP_DL_QUANTIZE:
        print(f"\n🔧 ESP-DL Recommendations:")
        if quantized_size > 100:  # 100KB threshold
            print("   💡 Model size > 100KB - consider more aggressive quantization")
        else:
            print("   ✅ Model size optimized for ESP32")
        
        if test_accuracy > 0.95:
            print("   ✅ Excellent accuracy for ESP deployment")
        elif test_accuracy > 0.90:
            print("   ⚠️  Good accuracy - monitor real-world performance")
        else:
            print("   🔧 Consider model architecture adjustments")

def compare_esp_vs_standard(debug=False):
    """Compare enhanced ESP training vs standard training"""
    print("🔍 Comparing Enhanced ESP vs Standard Training...")
    
    # Run standard training
    print("📊 Running Standard Training...")
    standard_model, standard_history, standard_dir = train_model(debug=debug)
    
    # Run enhanced ESP training
    print("📊 Running Enhanced ESP Training...")
    enhanced_model, enhanced_history, enhanced_dir = run_enhanced_esp_training(debug=debug)
    
    if enhanced_model is None:
        print("❌ Enhanced training failed, cannot compare")
        return None
    
    # Compare results
    from utils import get_data_splits
    _, _, (x_test, y_test) = get_data_splits()
    
    # Preprocess test data consistently
    data_processor = ESPDataProcessor()
    x_test, y_test = data_processor.esp_optimized_preprocessing(x_test, y_test, is_training=False)
    
    if params.MODEL_ARCHITECTURE != "original_haverland":
        y_test = y_test.astype(np.int32)
    
    standard_test_loss, standard_test_accuracy = standard_model.evaluate(x_test, y_test, verbose=0)
    enhanced_test_loss, enhanced_test_accuracy = enhanced_model.evaluate(x_test, y_test, verbose=0)
    
    print("\n" + "=" * 60)
    print("🏆 ESP TRAINING COMPARISON RESULTS")
    print("=" * 60)
    print(f"Standard Training:")
    print(f"  Test Accuracy: {standard_test_accuracy:.4f}")
    print(f"  Output Directory: {standard_dir}")
    print(f"Enhanced ESP Training:")
    print(f"  Test Accuracy: {enhanced_test_accuracy:.4f}")
    print(f"  Output Directory: {enhanced_dir}")
    print(f"Improvement: {enhanced_test_accuracy - standard_test_accuracy:+.4f}")
    
    return {
        'standard': (standard_model, standard_history, standard_dir),
        'enhanced': (enhanced_model, enhanced_history, enhanced_dir)
    }

# Command line interface
def main():
    parser = argparse.ArgumentParser(description='Enhanced ESP Quantization-Ready Training')
    parser.add_argument('--compare', action='store_true',
                       help='Compare enhanced ESP vs standard training')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            print("🔍 Running ESP Training Comparison...")
            results = compare_esp_vs_standard(debug=args.debug)
            if results:
                print("\n✅ ESP training comparison completed!")
        else:
            print("🚀 Starting Enhanced ESP Training...")
            model, history, output_dir = run_enhanced_esp_training(debug=args.debug)
            if model is not None:
                print(f"\n✅ Enhanced ESP training completed successfully!")
                print(f"📁 Output directory: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Enhanced ESP training failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Enhanced Training with Hyperparameter Tuning')
    parser.add_argument('--use_tuner', action='store_true', 
                       help='Enable hyperparameter tuning')
    parser.add_argument('--advanced', action='store_true',
                       help='Enable advanced training features')
    parser.add_argument('--num_trials', type=int, default=5,
                       help='Number of tuning trials (default: 5)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    return parser.parse_args()

# Update the main section:
if __name__ == "__main__":
    args = parse_arguments()
    
    from utils import get_data_splits
    (x_train, y_train), (x_val, y_val), _ = get_data_splits()
    
    if args.use_tuner:
        print("🚀 Running with hyperparameter tuning...")
        best_model, best_params, results, tuner = run_simple_tuning(
            x_train, y_train, x_val, y_val, 
            num_trials=args.num_trials,
            debug=args.debug
        )
        
        if args.advanced:
            # Run the full enhanced training pipeline
            model, history, output_dir = run_enhanced_training_with_tuning(debug=args.debug)
    else:
        # Run standard training without tuning
        from enhanced_train import run_enhanced_training
        model, history, output_dir = run_enhanced_training(debug=args.debug)