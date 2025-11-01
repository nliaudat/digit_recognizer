import argparse
# train.py
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import os
# Suppress TensorFlow C++ backend logs (INFO/WARNING) only if not in debug mode
import sys
def suppress_tf_logs_if_needed():
    # Check for --debug in sys.argv before importing TensorFlow
    if '--debug' not in sys.argv:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
suppress_tf_logs_if_needed()
import sys
from tqdm.auto import tqdm
import logging
from contextlib import contextmanager
from models import create_model, compile_model, model_summary
# from models.model_factory import print_hyperparameter_summary
from utils import get_data_splits, preprocess_images
# from utils.preprocess import *
# from utils.data_pipeline import *
# from analyse import *
# from tuner import *
from utils.preprocess import validate_quantization_combination, validate_preprocessing_consistency, check_qat_compatibility, debug_preprocessing_flow, diagnose_quantization_settings
from utils.data_pipeline import create_tf_dataset_from_arrays
from analyse import evaluate_tflite_model, analyze_quantization_impact, training_diagnostics, verify_model_predictions, debug_model_architecture
from tuner import run_architecture_tuning
from parameters import get_hyperparameter_summary_text, validate_quantization_parameters
import parameters as params
from utils.logging import log_print
from utils.multi_source_loader import clear_cache
from utils.augmentation import (
    create_augmentation_pipeline, 
    apply_augmentation_to_dataset, 
    test_augmentation_pipeline, 
    print_augmentation_summary,
    create_augmentation_safety_monitor,
    setup_augmentation_for_training
)

# import tensorflow_model_optimization as tfmot

# QAT imports
try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    log_print("⚠️  tensorflow-model-optimization not available. Install with: pip install tensorflow-model-optimization", level=1)
    QAT_AVAILABLE = False
    tfmot = None 
    
# try:
    # import onnx
    # import tf2onnx
    # from tf2onnx import tf_loader
    # ONNX_AVAILABLE = True
# except ImportError:
    # print("⚠️  ONNX export not available. Install with: pip install onnx tf2onnx")
    # ONNX_AVAILABLE = False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Digit Recognition Training')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable TensorFlow debug logs and verbose output')
    parser.add_argument('--test_all_models', action='store_true',
                       help='Test all available model architectures and compare performance')
    parser.add_argument('--train', nargs='+', choices=params.AVAILABLE_MODELS,
                       help='Train specific model architectures from AVAILABLE_MODELS')      
    parser.add_argument('--use_tuner', action='store_true',
                       help='Enable hyperparameter tuning before training')
    parser.add_argument('--train_all', action='store_true',
                       help='Train all available model architectures sequentially')
    parser.add_argument('--advanced', action='store_true',
                       help='Enable advanced training features')
    parser.add_argument('--num_trials', type=int, default=100,
                       help='Number of tuning trials (default: 5)')
    return parser.parse_args()
    
def set_all_seeds(seed=params.SHUFFLE_SEED):
    """Set all random seeds for complete reproducibility"""
    # Python built-in random
    import random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # TensorFlow
    tf.random.set_seed(seed)
    
    # Set environment variables for CUDA (if using GPU)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # Configure TensorFlow for deterministic operations
    tf.config.experimental.enable_op_determinism()

# Set TensorFlow logging level based on debug flag
def setup_tensorflow_logging(debug=False):
    """Configure TensorFlow logging verbosity - enhanced version"""
    if debug:
        # Enable all TensorFlow logs
        tf.get_logger().setLevel('INFO')
        tf.autograph.set_verbosity(3)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        
        # Enable other logging
        import logging
        logging.getLogger().setLevel(logging.INFO)
    else:
        # MAXIMUM SUPPRESSION - Completely silent
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        
        # Suppress ALL TensorFlow C++ logs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '0'
        
        # Suppress absl logging
        try:
            import absl.logging
            absl.logging.set_verbosity(absl.logging.ERROR)
        except ImportError:
            pass
        
        # Suppress all Python warnings
        import warnings
        # warnings.filterwarnings('ignore')
        warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
        warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        
        # Suppress other loggers
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('h5py').setLevel(logging.ERROR)
        logging.getLogger('numexpr').setLevel(logging.ERROR)

@contextmanager
def suppress_all_output(debug=False):
    """Completely suppress all output during TFLite conversion and other noisy operations"""
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
        
        # Additional TensorFlow-specific suppression
        import warnings
        original_warnings = warnings.showwarning
        warnings.showwarning = lambda *args, **kwargs: None
        
        # Set TensorFlow to maximum suppression
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('ERROR')
        
        try:
            import absl.logging
            absl.logging.set_verbosity(absl.logging.ERROR)
        except ImportError:
            pass
        
        try:
            yield
        finally:
            # Restore everything
            logging.disable(logging.NOTSET)
            warnings.showwarning = original_warnings
            
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

# def apply_qat(model):
    # """Apply Quantization Aware Training using modern TF API"""
    # if not QAT_AVAILABLE:
        # print("❌ QAT not available - install tensorflow-model-optimization")
        # return model
    
    # try:
        # print("🎯 Applying Quantization Aware Training...")
        # qat_model = tfmot.quantization.keras.quantize_model(model)
        # print("✅ QAT applied successfully")
        # return qat_model
    # except Exception as e:
        # print(f"❌ QAT failed: {e}")
        # return model

def create_qat_model():
    """Create a model with Quantization Aware Training using the correct API pattern"""
    if not QAT_AVAILABLE:
        print(f"❌ QAT not available")
        return create_model()
    
    try:
        print("🎯 Building model with Quantization Aware Training...")
        model = create_model()
        
        print(f"🔍 Model type: {type(model)}")
        print(f"🔍 Model class: {model.__class__.__name__}")
        
        # Use the annotate/apply pattern directly
        print("🔄 Using quantize_annotate_model + quantize_apply...")
        
        # Method 1: Direct annotate/apply
        # annotated_model = tfmot.quantization.keras.quantize_annotate_model(model)
        # qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)
        qat_model = tfmot.quantization.keras.quantize_model(model)
        
        print("✅ QAT model created successfully")
        return qat_model
        
    except Exception as e:
        print(f"❌ QAT model creation failed: {e}")
        print("🔄 Using standard model without QAT")
        return create_model()

class TFLiteModelManager:
    def __init__(self, output_dir, debug=False):
        self.output_dir = output_dir
        self.best_accuracy = 0.0
        self.debug = debug
 
    def _completely_suppress_output(self):
        """Completely suppress all output during TFLite conversion"""
        import os
        import sys
        from contextlib import contextmanager
        
        @contextmanager
        def suppress_tflite_output():
            # Maximum TensorFlow log suppression
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            
            # Suppress TensorFlow Python logs
            tf_logger = tf.get_logger()
            original_tf_level = tf_logger.level
            tf_logger.setLevel('ERROR')
            
            # Suppress absl logging
            try:
                import absl.logging
                original_absl_level = absl.logging.get_verbosity()
                absl.logging.set_verbosity(absl.logging.ERROR)
            except ImportError:
                pass
            
            # Redirect stdout/stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            try:
                # For Windows compatibility
                with open(os.devnull, 'w') as fnull:
                    sys.stdout = fnull
                    sys.stderr = fnull
                    
                    # Also suppress C-level output
                    try:
                        # This handles the low-level C++ output that bypasses Python redirection
                        original_stdout_fd = os.dup(1)
                        original_stderr_fd = os.dup(2)
                        
                        devnull_fd = os.open(os.devnull, os.O_WRONLY)
                        os.dup2(devnull_fd, 1)
                        os.dup2(devnull_fd, 2)
                        os.close(devnull_fd)
                    except (OSError, AttributeError):
                        # Fallback if fd manipulation fails
                        pass
                    
                    try:
                        yield
                    finally:
                        # Restore stdout/stderr
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr
                        
                        # Restore file descriptors
                        try:
                            os.dup2(original_stdout_fd, 1)
                            os.dup2(original_stderr_fd, 2)
                            os.close(original_stdout_fd)
                            os.close(original_stderr_fd)
                        except (OSError, NameError):
                            pass
            finally:
                # Restore logging levels
                tf_logger.setLevel(original_tf_level)
                try:
                    import absl.logging
                    absl.logging.set_verbosity(original_absl_level)
                except ImportError:
                    pass
        
        return suppress_tflite_output()
 
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
        """Save model in trainable format - Keras 3 compatible"""
        timestamp = datetime.now().strftime("%H%M%S")
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch:03d}_acc_{accuracy:.4f}_{timestamp}.keras")
        
        # Keras 3 compatible saving
        model.save(checkpoint_path)
        
        if self.debug:
            print(f"💾 Saved trainable checkpoint: {checkpoint_path}")
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            best_checkpoint_path = os.path.join(self.output_dir, "best_model.keras")
            model.save(best_checkpoint_path)  # Remove save_format argument
            if self.debug:
                print(f"🏆 New best model saved: {best_checkpoint_path}")
        
        return checkpoint_path
        
    def _is_qat_model(self, model):
        """Check if model is a QAT model with better detection"""
        # Check for quantization layers
        for layer in model.layers:
            layer_name = layer.name.lower()
            layer_class = layer.__class__.__name__.lower()
            
            # Check for quantization indicators
            if (hasattr(layer, 'quantize_config') or 
                'quant' in layer_name or 
                'qat' in layer_name or
                'quantize' in layer_class):
                return True
        
        # Check model name and attributes
        model_name = model.name.lower() if hasattr(model, 'name') else ''
        if 'qat' in model_name or 'quant' in model_name:
            return True
        
        # Check if model was created within quantize_scope
        if hasattr(model, '_quantize_scope'):
            return True
            
        return False


    def _convert_qat_model(self, model, filename, representative_data=None):
        """Convert QAT model to TFLite with proper representative dataset"""
        try:
            if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                log_print("🎯 Converting QAT model to TFLite...", level=2)

            # Use direct conversion for QAT models
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # QAT models need representative dataset with EXACTLY the same preprocessing as training
            if representative_data is None:
                def qat_representative_dataset():
                    from utils import get_data_splits, preprocess_images
                    
                    # Get raw training data
                    (x_train_raw, y_train_raw), _, _ = get_data_splits()
                    
                    # Use a subset for calibration - ensure we have enough samples
                    num_samples = min(100, len(x_train_raw), params.QUANTIZE_NUM_SAMPLES)
                    calibration_data = x_train_raw[:num_samples]
                    
                    # CRITICAL: Use the SAME preprocessing as during QAT training
                    calibration_processed = preprocess_images(calibration_data, for_training=False)
                    
                    # Ensure proper data type and range for QAT
                    if calibration_processed.dtype != np.float32:
                        calibration_processed = calibration_processed.astype(np.float32)
                    
                    # For QAT, data should be in the same range as during training
                    # If your QAT training used [0, 1] range, ensure calibration data is also [0, 1]
                    if calibration_processed.max() > 1.0:
                        calibration_processed = calibration_processed / 255.0
                    
                    if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                        print(f"🔧 QAT Calibration: {len(calibration_processed)} samples, "
                              f"dtype: {calibration_processed.dtype}, "
                              f"range: [{calibration_processed.min():.3f}, {calibration_processed.max():.3f}]")
                    
                    # Verify data matches QAT training expectations
                    if calibration_processed.dtype != np.float32:
                        raise ValueError(f"QAT calibration data must be float32, got {calibration_processed.dtype}")
                    
                    # Yield data in the correct format
                    for i in range(len(calibration_processed)):
                        yield [calibration_processed[i:i+1]]
                
                converter.representative_dataset = qat_representative_dataset
            else:
                converter.representative_dataset = representative_data

            # QAT-specific conversion settings
            if params.ESP_DL_QUANTIZE:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                if self.debug:
                    print("🎯 ESP-DL INT8 quantization for QAT")
            else:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
                if self.debug:
                    print("🎯 Standard UINT8 quantization for QAT")

            # Additional QAT-specific settings
            converter.experimental_new_quantizer = True  # Use new quantizer for better QAT support
            converter._experimental_disable_per_channel = False  # Enable per-channel quantization
            
            # Convert with output suppression
            with suppress_all_output(self.debug):
                tflite_model = converter.convert()
                
            return self._save_tflite_file(tflite_model, filename, True)
            
        except Exception as e:
            log_print(f"❌ QAT conversion failed: {e}", level=1)
            # Enhanced fallback with better debugging
            return self._convert_qat_model_fallback_enhanced(model, filename)
            
    def _convert_qat_model_fallback_enhanced(self, model, filename):
        """Enhanced fallback conversion for QAT model with comprehensive debugging"""
        try:
            if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                log_print("🔄 Trying enhanced QAT fallback conversion...", level=2)
                log_print("🔍 Diagnosing QAT conversion issue...", level=2)
            
            # Test model with sample input
            test_input = tf.random.normal([1] + list(params.INPUT_SHAPE), dtype=tf.float32)
            try:
                test_output = model(test_input)
                if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                    log_print(f"✅ Model accepts float32 inputs: output shape {test_output.shape}", level=2)
            except Exception as e:
                if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                    log_print(f"❌ Model input test failed: {e}", level=1)
            
            # Strategy 1: Dynamic range quantization only (most reliable)
            if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                log_print("🔧 Strategy 1: Dynamic range quantization only...", level=2)
            
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Don't use representative dataset in fallback
            # This often causes issues with QAT models
            
            if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                tflite_model = converter.convert()
            else:
                with self._completely_suppress_output():
                    tflite_model = converter.convert()
            
            if self.debug or getattr(params, 'VERBOSE', 2) >= 2:
                log_print("✅ Dynamic range quantization successful", level=2)
            
            return self._save_tflite_file(tflite_model, filename, True)
                
        except Exception as e:
            log_print(f"❌ Enhanced QAT fallback failed: {e}", level=1)
            
            # Final fallback: Just save the model without quantization
            log_print("🔄 Final fallback: Saving without quantization...", level=1)
            return self.save_as_tflite(model, filename, quantize=False)
            


    def _convert_qat_model_fallback(self, model, filename):
        """Legacy fallback - redirect to enhanced version"""
        return self._convert_qat_model_fallback_enhanced(model, filename)
        
            
    def save_as_tflite_direct(self, model, filename, quantize=False, representative_data=None):
        """Direct Keras model conversion - Keras 3 compatible with REAL data"""
        try:
            # Ensure model is built
            if not model.built:
                dummy_input = tf.zeros([1] + list(params.INPUT_SHAPE), dtype=tf.float32)
                _ = model(dummy_input)
            
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                # Use provided representative data or create REAL default
                if representative_data is not None:
                    converter.representative_dataset = representative_data
                else:
                    def real_representative_dataset():
                        from utils import get_data_splits, preprocess_images
                        
                        # Use real data instead of random
                        (x_train_raw, _), _, _ = get_data_splits()
                        num_samples = min(100, len(x_train_raw), params.QUANTIZE_NUM_SAMPLES)
                        calibration_data = x_train_raw[:num_samples]
                        
                        # Preprocess properly
                        calibration_processed = preprocess_images(calibration_data, for_training=False)
                        
                        # Ensure proper format
                        if calibration_processed.dtype != np.float32:
                            calibration_processed = calibration_processed.astype(np.float32)
                        if calibration_processed.max() > 1.0:
                            calibration_processed = calibration_processed / 255.0
                        
                        if self.debug:
                            print(f"🔧 Direct conversion - Real calibration: {calibration_processed.dtype}")
                        
                        for i in range(len(calibration_processed)):
                            yield [calibration_processed[i:i+1]]
                    
                    converter.representative_dataset = real_representative_dataset
                
                print("🎯 Using dynamic range quantization with REAL data (Keras 3 safe)")
            
            # Convert model
            with suppress_all_output(self.debug):
                tflite_model = converter.convert()
            
            return self._save_tflite_file(tflite_model, filename, quantize)
                
        except Exception as e:
            print(f"❌ Direct TFLite conversion failed: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise

    def save_as_tflite(self, model, filename, quantize=False, representative_data=None):
        """Save model as TFLite with proper QAT handling and debug control"""
        try:
            if self.debug:
                print(f"🔧 Converting {filename} to TFLite...")
                print(f"   Quantize: {quantize}, QAT Model: {self._is_qat_model(model)}")
            
            # Ensure model is built
            if not model.built:
                dummy_input = tf.zeros([1] + list(params.INPUT_SHAPE), dtype=tf.float32)
                _ = model(dummy_input)
            
            # Handle QAT models specifically
            if quantize and self._is_qat_model(model):
                if self.debug:
                    print("🎯 Converting QAT model to quantized TFLite...")
                return self._convert_qat_model(model, filename, representative_data)
            
            # For non-QAT models, use the simple Keras 3 approach but ensure quantization works
            if quantize:
                if self.debug:
                    print("🎯 Converting standard model to quantized TFLite...")
                return self._convert_standard_quantized(model, filename, representative_data)
            else:
                if self.debug:
                    print("🔧 Converting to float TFLite...")
                return self.save_as_tflite_simple_keras3(model, filename, quantize=False)
                
        except Exception as e:
            if self.debug:
                print(f"❌ Primary TFLite conversion failed: {e}")
                print("🔄 Falling back to saved model approach...")
            
            # Fallback to saved model approach
            return self.save_as_tflite_savedmodel(model, filename, quantize, representative_data)
            
    def _convert_standard_quantized(self, model, filename, representative_data=None):
        """Convert standard (non-QAT) model to quantized TFLite"""
        try:
            if self.debug:
                print("🔧 Converting standard model with quantization...")
            
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Use representative dataset for proper quantization
            if representative_data is None:
                def default_representative_dataset():
                    from utils import get_data_splits, preprocess_images
                    (x_train_raw, _), _, _ = get_data_splits()
                    calibration_data = x_train_raw[:params.QUANTIZE_NUM_SAMPLES]
                    calibration_processed = preprocess_images(calibration_data, for_training=False)
                    
                    # Ensure proper format for quantization
                    if calibration_processed.dtype != np.float32:
                        calibration_processed = calibration_processed.astype(np.float32)
                    if calibration_processed.max() > 1.0:
                        calibration_processed = calibration_processed / 255.0
                    
                    if self.debug:
                        print(f"🔧 Calibration data: {calibration_processed.dtype}, "
                              f"range: [{calibration_processed.min():.3f}, {calibration_processed.max():.3f}]")
                    
                    for i in range(len(calibration_processed)):
                        yield [calibration_processed[i:i+1]]
                
                converter.representative_dataset = default_representative_dataset
            else:
                converter.representative_dataset = representative_data
            
            # Set quantization based on ESP-DL setting
            if params.ESP_DL_QUANTIZE:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                if self.debug:
                    print("🎯 ESP-DL INT8 quantization")
            else:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
                if self.debug:
                    print("🎯 Standard UINT8 quantization")
            
            # Convert with output suppression
            with suppress_all_output(self.debug):
                tflite_model = converter.convert()
            
            return self._save_tflite_file(tflite_model, filename, True)
            
        except Exception as e:
            if self.debug:
                print(f"❌ Standard quantization failed: {e}")
                print("🔄 Falling back to dynamic range quantization...")
            
            # Fallback to dynamic range only
            return self.save_as_tflite_simple_keras3(model, filename, quantize=True)

    def save_as_tflite_savedmodel(self, model, filename, quantize=False, representative_data=None):
        """Use SavedModel approach for conversion - Keras 3 FIXED"""
        try:
            import tempfile
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Keras 3: Use model.export() for SavedModel format
                model_dir = os.path.join(temp_dir, "saved_model")
                
                # Use export() method for SavedModel in Keras 3
                model.export(model_dir)
                
                # Verify the SavedModel was created
                saved_model_pb = os.path.join(model_dir, "saved_model.pb")
                if not os.path.exists(saved_model_pb):
                    raise FileNotFoundError(f"SavedModel not properly created at {model_dir}")
                
                print(f"✅ SavedModel created at: {model_dir}")
                
                # Load from SavedModel
                converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
                
                if quantize:
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    
                    # Use proper representative dataset
                    if representative_data is not None:
                        converter.representative_dataset = representative_data
                    else:
                        def default_representative_dataset():
                            # Use actual data instead of random data
                            from utils import get_data_splits, preprocess_images
                            (x_train_raw, _), _, _ = get_data_splits()
                            calib_data = x_train_raw[:min(100, params.QUANTIZE_NUM_SAMPLES)]
                            
                            # Use inference preprocessing
                            calib_processed = preprocess_images(calib_data, for_training=False)
                            
                            # Ensure float32 for calibration
                            if calib_processed.dtype != np.float32:
                                calib_processed = calib_processed.astype(np.float32)
                            if calib_processed.max() > 1.0:
                                calib_processed = calib_processed / 255.0
                                
                            print(f"🔧 SavedModel Calibration: {calib_processed.dtype}, "
                                  f"range: [{calib_processed.min():.3f}, {calib_processed.max():.3f}]")
                            
                            for i in range(len(calib_processed)):
                                yield [calib_processed[i:i+1]]
                        
                        converter.representative_dataset = default_representative_dataset
                    
                    # Set quantization settings
                    if params.ESP_DL_QUANTIZE:
                        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                        converter.inference_input_type = tf.int8
                        converter.inference_output_type = tf.int8
                        print("🎯 ESP-DL INT8 quantization for SavedModel")
                    else:
                        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                        converter.inference_input_type = tf.uint8
                        converter.inference_output_type = tf.uint8
                        print("🎯 Standard UINT8 quantization for SavedModel")
                
                # Convert model
                with suppress_all_output(self.debug):
                    tflite_model = converter.convert()
                
                return self._save_tflite_file(tflite_model, filename, quantize)
                    
        except Exception as e:
            print(f"❌ SavedModel conversion failed: {e}")
            print("🔄 Trying Keras 3 alternative approach...")
            return self.save_as_tflite_keras3_alternative(model, filename, quantize, representative_data)
            
    def save_as_tflite_keras3_alternative(self, model, filename, quantize=False, representative_data=None):
        """Keras 3 alternative conversion method"""
        try:
            print("🔧 Using Keras 3 alternative conversion...")
            
            # Method 1: Direct conversion (should work in Keras 3)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                # Use representative dataset
                if representative_data is None:
                    def representative_dataset():
                        from utils import get_data_splits, preprocess_images
                        (x_train_raw, _), _, _ = get_data_splits()
                        calib_data = x_train_raw[:min(100, params.QUANTIZE_NUM_SAMPLES)]
                        calib_processed = preprocess_images(calib_data, for_training=False)
                        
                        # Ensure float32
                        if calib_processed.dtype != np.float32:
                            calib_processed = calib_processed.astype(np.float32)
                        if calib_processed.max() > 1.0:
                            calib_processed = calib_processed / 255.0
                        
                        print(f"🔧 Keras3 Calibration: {calib_processed.dtype}, "
                              f"range: [{calib_processed.min():.3f}, {calib_processed.max():.3f}]")
                        
                        for i in range(len(calib_processed)):
                            yield [calib_processed[i:i+1]]
                    
                    converter.representative_dataset = representative_dataset
                
                # For Keras 3, start with dynamic range only (more reliable)
                print("🎯 Using dynamic range quantization (Keras 3 compatible)")
            
            # Convert
            tflite_model = converter.convert()
            
            return self._save_tflite_file(tflite_model, filename, quantize)
            
        except Exception as e:
            print(f"❌ Keras 3 alternative failed: {e}")
            raise
            
    def save_as_tflite_simple_keras3(self, model, filename, quantize=False):
        """Simple reliable method for Keras 3 - with REAL representative data"""
        try:
            if self.debug:
                print("🔧 Using simple Keras 3 conversion...")
            
            # Always use direct conversion in Keras 3
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            if quantize:
                # Use ONLY dynamic range quantization (most reliable in Keras 3)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                if self.debug:
                    print("🎯 Dynamic range quantization only (Keras 3 safe)")
                
                # Use REAL representative dataset instead of random data
                def real_representative_dataset():
                    from utils import get_data_splits, preprocess_images
                    
                    # Get real training data
                    (x_train_raw, _), _, _ = get_data_splits()
                    
                    # Use a subset for calibration
                    num_calibration_samples = min(100, len(x_train_raw), params.QUANTIZE_NUM_SAMPLES)
                    calibration_data = x_train_raw[:num_calibration_samples]
                    
                    # Preprocess the same way as inference
                    calibration_processed = preprocess_images(calibration_data, for_training=False)
                    
                    # Ensure proper format for quantization
                    if calibration_processed.dtype != np.float32:
                        calibration_processed = calibration_processed.astype(np.float32)
                    if calibration_processed.max() > 1.0:
                        calibration_processed = calibration_processed / 255.0
                    
                    if self.debug:
                        print(f"🔧 Real calibration data: {len(calibration_processed)} samples")
                        print(f"   Data type: {calibration_processed.dtype}")
                        print(f"   Range: [{calibration_processed.min():.3f}, {calibration_processed.max():.3f}]")
                        print(f"   Shape: {calibration_processed.shape}")
                    
                    for i in range(len(calibration_processed)):
                        yield [calibration_processed[i:i+1]]
                
                converter.representative_dataset = real_representative_dataset
            
            # Convert with complete output suppression
            with suppress_all_output(self.debug):
                tflite_model = converter.convert()
            
            # Save
            model_path = os.path.join(self.output_dir, filename)
            with open(model_path, 'wb') as f:
                f.write(tflite_model)
            
            model_size_kb = len(tflite_model) / 1024
            quant_type = "DynamicQuant" if quantize else "Float32"
            
            # Only print if debug is enabled
            if self.debug:
                print(f"💾 Saved {filename} ({quant_type}): {model_size_kb:.1f} KB")
                
                # Test the converted model (only in debug mode)
                self.test_tflite_model(model_path)
            else:
                # Silent mode: just do a basic check without output
                try:
                    interpreter = tf.lite.Interpreter(model_path=model_path)
                    interpreter.allocate_tensors()
                except Exception:
                    # If basic check fails, we still want to know even in non-debug mode
                    print(f"❌ TFLite model verification failed for {filename}")
                    return None, 0
            
            return tflite_model, model_size_kb
            
        except Exception as e:
            # Always show errors, even in non-debug mode
            print(f"❌ Simple Keras 3 conversion failed: {e}")
            return None, 0

    def test_tflite_model(self, tflite_path):
        """Quick test of TFLite model - only outputs in debug mode"""
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            if self.debug:
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                print(f"✅ TFLite model loaded successfully:")
                print(f"   Input: {input_details[0]['dtype']}, shape: {input_details[0]['shape']}")
                print(f"   Output: {output_details[0]['dtype']}, shape: {output_details[0]['shape']}")
            
            return True
        except Exception as e:
            # Always show errors, even in non-debug mode
            print(f"❌ TFLite model test failed: {e}")
            return False

    def _save_tflite_file(self, tflite_model, filename, quantize=False):
        """Save TFLite model to file with debug control"""
        try:
            model_path = os.path.join(self.output_dir, filename)
            with open(model_path, 'wb') as f:
                f.write(tflite_model)
            
            model_size_kb = len(tflite_model) / 1024
            quant_type = "Quantized" if quantize else "Float32"
            
            # Only print in debug mode
            if self.debug:
                print(f"💾 Saved {filename} ({quant_type}): {model_size_kb:.1f} KB")
                
                # Test the model only in debug mode
                self.test_tflite_model(model_path)
            else:
                # Silent verification
                try:
                    interpreter = tf.lite.Interpreter(model_path=model_path)
                    interpreter.allocate_tensors()
                except Exception as e:
                    print(f"❌ TFLite model verification failed: {e}")
                    return None, 0
            
            return tflite_model, model_size_kb
            
        except Exception as e:
            print(f"❌ Failed to save TFLite file: {e}")
            return None, 0

    
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
                print(f"❌ TFLite conversion failed in save best model: {e}")
                return None
        return None
        
    # def export_to_onnx(self, model, filename="best_model.onnx"):
        # """Export model to ONNX format - programmatic approach"""
        # if not ONNX_AVAILABLE:
            # print("❌ ONNX export not available - install onnx and tf2onnx")
            # return None
        
        # try:
            # print("🔄 Converting to ONNX format...")
            
            # # Build model if not built
            # if not model.built:
                # dummy_input = tf.zeros([1] + list(params.INPUT_SHAPE), dtype=tf.float32)
                # _ = model(dummy_input)
            
            # onnx_path = os.path.join(self.output_dir, filename)
            
            # # Get the concrete function from the Keras model
            # input_spec = tf.TensorSpec((None,) + params.INPUT_SHAPE, tf.float32, name="input")
            
            # # Use the newer tf2onnx API
            # from tf2onnx import convert
            # from tf2onnx import tf_loader
            # import onnxruntime as ort
            
            # # Save as SavedModel first
            # temp_model_path = os.path.join(self.output_dir, "temp_saved_model")
            # tf.saved_model.save(model, temp_model_path)
            
            # # Convert using tf2onnx
            # convert.from_saved_model(
                # temp_model_path,
                # input_signature=[input_spec],
                # output_path=onnx_path,
                # opset=13
            # )
            
            # # Clean up temporary model
            # import shutil
            # shutil.rmtree(temp_model_path, ignore_errors=True)
            
            # # Verify the ONNX model
            # try:
                # ort_session = ort.InferenceSession(onnx_path)
                # onnx_size_kb = os.path.getsize(onnx_path) / 1024
                # print(f"✅ ONNX model saved and verified: {onnx_path} ({onnx_size_kb:.1f} KB)")
                # print(f"   ONNX Inputs: {[input.name for input in ort_session.get_inputs()]}")
                # print(f"   ONNX Outputs: {[output.name for output in ort_session.get_outputs()]}")
                # return onnx_path, onnx_size_kb
            # except Exception as verification_error:
                # print(f"⚠️  ONNX model saved but verification failed: {verification_error}")
                # onnx_size_kb = os.path.getsize(onnx_path) / 1024
                # return onnx_path, onnx_size_kb
                
        # except Exception as e:
            # print(f"❌ ONNX conversion failed: {e}")
            # if self.debug:
                # import traceback
                # traceback.print_exc()
            
            # # Clean up on error
            # try:
                # import shutil
                # temp_model_path = os.path.join(self.output_dir, "temp_saved_model")
                # if os.path.exists(temp_model_path):
                    # shutil.rmtree(temp_model_path, ignore_errors=True)
            # except:
                # pass
            
            # return None

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
        
        print(f"Training plots saved to: {plot_path}")

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
### end class TQDMProgressBa           
            
def create_callbacks(output_dir, tflite_manager, representative_data, total_epochs, monitor, debug=False, validation_data=None):
    """Create training callbacks with robust CSV logging"""
    
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
                restore_best_weights=params.RESTORE_BEST_WEIGHTS,
                mode=mode,
                verbose=1 if debug else 0
            )
        )
    
    # Add Augmentation Safety Monitor if using augmentation and validation data is provided
    if params.USE_DATA_AUGMENTATION and validation_data is not None:
        safety_monitor = create_augmentation_safety_monitor(
            validation_data=validation_data,
            debug=debug
        )
        callbacks.append(safety_monitor)
        if debug:
            print("🔒 AugmentationSafetyMonitor callback added")
    
    # Regular checkpoint every epoch
    # callbacks.append(
        # tf.keras.callbacks.ModelCheckpoint(
            # filepath=os.path.join(output_dir, "checkpoints", "epoch_{epoch:03d}.keras"),
            # save_freq='epoch',
            # save_best_only=False,
            # verbose=1 if debug else 0
        # )
    # )
    
    # Best model checkpoint
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1 if debug else 0
        )
    )
    
    # Learning rate scheduler
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=params.LR_SCHEDULER_MONITOR,
            factor=params.LR_SCHEDULER_FACTOR,
            patience=params.LR_SCHEDULER_PATIENCE,
            min_lr=params.LR_SCHEDULER_MIN_LR,
            verbose=1 if debug else 0
        )
    )
    
    # TFLite model checkpoint
    callbacks.append(
        TFLiteCheckpoint(tflite_manager, representative_data)
    )
    
    # TQDM progress bar
    callbacks.append(
        TQDMProgressBar(total_epochs, monitor, debug)
    )
    
    # ROBUST CSV Logger with proper error handling
    csv_path = os.path.join(output_dir, 'training_log.csv')
    print(f"📄 CSV Logger will save to: {csv_path}")
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Test if we can write to the directory
    try:
        test_file = os.path.join(output_dir, 'test_write.tmp')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("✅ Output directory is writable")
    except Exception as e:
        print(f"❌ Output directory not writable: {e}")
        # Try to create CSV in current directory as fallback
        csv_path = 'training_log_fallback.csv'
        print(f"🔄 Using fallback path: {csv_path}")
    
    # Create CSV logger with explicit configuration
    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=csv_path,
        separator=',',
        append=False
    )
    callbacks.append(csv_logger)
    
    # Create checkpoints directory
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # Debug: Print all callbacks
    if debug:
        print("🔍 Callbacks created:")
        for i, callback in enumerate(callbacks):
            print(f"   {i+1}. {callback.__class__.__name__}")
            if hasattr(callback, 'filename'):
                print(f"      File: {getattr(callback, 'filename', 'N/A')}")
    
    return callbacks
    
def create_qat_representative_dataset(x_train_raw, num_samples=params.QUANTIZE_NUM_SAMPLES):
    """Create representative dataset that exactly matches QAT training conditions"""
    def representative_dataset():
        from utils import get_data_splits, preprocess_images
        
        # Use the sophisticated preprocess_images with for_training=False
        # BUT ensure it matches what was used during QAT training
        x_calibration = preprocess_images(x_train_raw[:num_samples], for_training=False)
        
        # CRITICAL: FOR QAT, data must match training conditions exactly
        # If QAT training used specific normalization, replicate it here
        
        # Convert to float32 (required for calibration)
        if x_calibration.dtype != np.float32:
            x_calibration = x_calibration.astype(np.float32)
            
        # Ensure normalization consistency with QAT training
        # If your QAT model was trained with [0, 1] data, ensure calibration data is [0, 1]
        if x_calibration.max() > 1.0:
            x_calibration = x_calibration / 255.0
        
        print(f"QAT Representative: {x_calibration.dtype}, "
              f"range: [{x_calibration.min():.3f}, {x_calibration.max():.3f}], "
              f"samples: {len(x_calibration)}")
        
        # Verify we have valid data
        if np.any(np.isnan(x_calibration)) or np.any(np.isinf(x_calibration)):
            raise ValueError("QAT calibration data contains NaN or Inf values")
        
        for i in range(len(x_calibration)):
            yield [x_calibration[i:i+1]]  # Keep as float32 for QAT
    
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
    print(f"  Quantization: {'Enabled' if params.QUANTIZE_MODEL else 'Disabled'}")
    print(f"  QAT: {'Enabled' if params.USE_QAT else 'Disabled'}")
    print(f"  ESP-DL Quantization: {params.ESP_DL_QUANTIZE}")
    print(f"  Debug mode: {'Enabled' if debug else 'Disabled'}")
    
def save_model_summary_to_file(model, output_dir):
    """Save model summary to a text file directly"""
    try:
        summary_path = os.path.join(output_dir, "model_summary.txt")
        
        # Redirect stdout to capture the summary
        original_stdout = sys.stdout
        with open(summary_path, 'w') as f:
            sys.stdout = f
            model.summary(print_fn=lambda x: print(x))
            sys.stdout = original_stdout
        
        print(f"✅ Model summary saved to: {summary_path}")
        
        # Also print summary to console in debug mode
        if getattr(params, 'VERBOSE', 2) >= 2:
            print("\n📋 MODEL SUMMARY:")
            model.summary()
            
        return True
    except Exception as e:
        print(f"⚠️  Could not save model summary: {e}")
        return False
    

def train_model(debug=False, best_hps=None):
    """Main training function with comprehensive handling of all 9 quantization cases"""
    setup_tensorflow_logging(debug)
    set_all_seeds(params.SHUFFLE_SEED)
    
    # Apply tuned hyperparameters if provided
    # if best_hps is not None:
        # print("🎯 APPLYING TUNED HYPERPARAMETERS:")
        # print("=" * 50)
        
        # # Extract best hyperparameters
        # best_optimizer = best_hps.get('optimizer')
        # best_lr = best_hps.get('learning_rate')
        # best_batch_size = best_hps.get('batch_size')
        
        # # Apply to parameters
        # params.OPTIMIZER_TYPE = best_optimizer
        # params.LEARNING_RATE = best_lr
        # params.BATCH_SIZE = best_batch_size
        
        # print(f"   Optimizer: {best_optimizer}")
        # print(f"   Learning Rate: {best_lr}")
        # print(f"   Batch Size: {best_batch_size}")
        # print("=" * 50)
    
    # VALIDATE AND CORRECT QUANTIZATION PARAMETERS FIRST
    print("🎯 VALIDATING QUANTIZATION PARAMETERS...")
    is_valid, corrected_params, message = validate_quantization_parameters()
    
    print(message)
    
    # Apply corrections if needed
    if not is_valid:
        print("🔄 Applying parameter corrections...")
        params.QUANTIZE_MODEL = corrected_params['QUANTIZE_MODEL']
        params.USE_QAT = corrected_params['USE_QAT']
        params.ESP_DL_QUANTIZE = corrected_params['ESP_DL_QUANTIZE']
        print(f"✅ Corrected parameters applied")
        
    # # Check training/inference alignment
    # alignment_ok = check_training_inference_alignment()
    # if not alignment_ok and params.USE_QAT:
        # print("🚨 CRITICAL: QAT training/inference misalignment detected!")
        # print("   This will cause quantization errors!")
    
    print("🎯 TRAINING CONFIGURATION:")
    print("=" * 60)
    print(f"   MODEL_ARCHITECTURE: {params.MODEL_ARCHITECTURE}")
    print(f"   QUANTIZE_MODEL: {params.QUANTIZE_MODEL}")
    print(f"   USE_QAT: {params.USE_QAT}")
    print(f"   ESP_DL_QUANTIZE: {params.ESP_DL_QUANTIZE}")
    
    # Validate quantization combination first
    is_valid, msg = validate_quantization_combination()
    if not is_valid:
        print(f"❌ {msg}")
        print("💡 Fix the quantization parameters in parameters.py")
        return None, None, None
    
    print(f"✅ {msg}")
    
    # Setup hardware
    print("🔧 Configuring hardware...")
    strategy = setup_gpu()
    
    # Create output directory
    color_mode = "GRAY" if params.USE_GRAYSCALE else "RGB"
    quantization_mode = ""
    if params.USE_QAT:
        quantization_mode = "_QAT"
    if params.ESP_DL_QUANTIZE:
        quantization_mode += "_ESP-DL"
    elif params.QUANTIZE_MODEL:
        quantization_mode += "_QUANT"
    
    training_dir = os.path.join(
        params.OUTPUT_DIR, 
        f"{params.MODEL_ARCHITECTURE}_{params.NB_CLASSES}cls{quantization_mode}_{color_mode}_{datetime.now().strftime('%m%d_%H%M')}"
    )
    os.makedirs(training_dir, exist_ok=True)
    print(f"📁 Output directory: {training_dir}")
    
    # QAT Compatibility Check
    if params.USE_QAT:
        qat_ok, qat_warnings, qat_errors, qat_info = check_qat_compatibility(QAT_AVAILABLE)
        
        if not qat_ok:
            print("❌ QAT Compatibility Errors:")
            for error in qat_errors:
                print(f"   - {error}")
            print("🔄 Disabling QAT...")
            params.USE_QAT = False
            # Re-validate after disabling QAT
            is_valid, msg = validate_quantization_combination()
            print(f"🔄 New configuration: {msg}")
        else:
            # Show warnings but don't disable QAT
            if qat_warnings:
                print("⚠️  QAT Warnings:")
                for warning in qat_warnings:
                    print(f"   - {warning}")
            
            # Show info messages
            if qat_info:
                print("💡 QAT Info:")
                for info_msg in qat_info:
                    print(f"   - {info_msg}")
    
    # Validate preprocessing consistency
    if not validate_preprocessing_consistency():
        print("❌ Preprocessing validation failed!")
        return None, None, None
    
    # LOAD AND PREPROCESS DATA
    print("\nLoading dataset from multiple sources...")
    (x_train_raw, y_train_raw), (x_val_raw, y_val_raw), (x_test_raw, y_test_raw) = get_data_splits()
    
    
    # Test the raw data
    print(f"Raw data - dtype: {x_train_raw.dtype}, range: [{x_train_raw.min()}, {x_train_raw.max()}]")

    # Test preprocessing output
    # x_train_test = preprocess_images(x_train_raw[:10], for_training=True)
    # print(f"After preprocessing - dtype: {x_train_test.dtype}, range: [{x_train_test.min():.3f}, {x_train_test.max():.3f}]")

    # Check if tf.data pipeline modifies it
    # if params.USE_TF_DATA_PIPELINE:
        # from utils.data_pipeline import create_tf_dataset_from_arrays
        # test_dataset = create_tf_dataset_from_arrays(x_train_test, y_train_raw[:10], training=True)
        # for batch_x, batch_y in test_dataset.take(1):
            # print(f"After tf.data - dtype: {batch_x.dtype}, range: [{batch_x.numpy().min():.3f}, {batch_x.numpy().max():.3f}]")
    
    print("🔄 Preprocessing images...")
    # Process each split ONLY ONCE
    x_train = preprocess_images(x_train_raw, for_training=True)
    x_val = preprocess_images(x_val_raw, for_training=True)  
    x_test = preprocess_images(x_test_raw, for_training=True)
    
    print(f"✅ Preprocessing complete:")
    print(f"   Train range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"   Val range: [{x_val.min():.3f}, {x_val.max():.3f}]")
    print(f"   Test range: [{x_test.min():.3f}, {x_test.max():.3f}]")
    print(f"   Shapes - Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    
    print("\n🔍 CHECKING TRAINING/INFERENCE ALIGNMENT WITH REAL DATA")
    alignment_ok = check_training_inference_alignment(x_train_raw)
    if not alignment_ok and params.USE_QAT:
        print("🚨 CRITICAL: QAT training/inference misalignment detected!")
        print("   This will cause quantization errors!")
    
    # NORMALIZE TO [0,1] FOR TRAINING AND AUGMENTATION
    # print("🔄 Normalizing data to [0,1] range for training...")
    # if x_train.dtype != np.float32 or x_train.max() > 1.0:
        # x_train = x_train.astype(np.float32) / 255.0
        # x_val = x_val.astype(np.float32) / 255.0
        # x_test = x_test.astype(np.float32) / 255.0
    
    # print(f"✅ Preprocessing complete:")
    # print(f"   Train range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    # print(f"   Val range: [{x_val.min():.3f}, {x_val.max():.3f}]")
    
    # print("✅ Data preprocessing complete - using preprocessed ranges directly")
    # # Ensure data types are correct but DON'T re-normalize
    # if x_train.dtype != np.float32:
        # x_train = x_train.astype(np.float32)
    # if x_val.dtype != np.float32:
        # x_val = x_val.astype(np.float32) 
    # if x_test.dtype != np.float32:
        # x_test = x_test.astype(np.float32)

    # print(f"✅ Using preprocessed ranges directly:")
    # print(f"   Train range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    # print(f"   Val range: [{x_val.min():.3f}, {x_val.max():.3f}]")    
    # print(f"   Shapes - Train: {x_train.shape}, Val: {x_val.shape}")
    
    # Handle labels based on model type
    if params.MODEL_ARCHITECTURE == "original_haverland":
        # Haverland model needs categorical labels (one-hot encoded)
        y_train_final = tf.keras.utils.to_categorical(y_train_raw, params.NB_CLASSES)
        y_val_final = tf.keras.utils.to_categorical(y_val_raw, params.NB_CLASSES) 
        y_test_final = tf.keras.utils.to_categorical(y_test_raw, params.NB_CLASSES)
        loss_type = 'categorical_crossentropy'
        print(f"✅ Using categorical labels (one-hot) for Haverland model")
    else:
        # Other models use sparse categorical labels (integer labels)
        y_train_final = y_train_raw.copy()
        y_val_final = y_val_raw.copy()
        y_test_final = y_test_raw.copy()
        loss_type = 'sparse_categorical_crossentropy'
        print(f"✅ Using sparse categorical labels for {params.MODEL_ARCHITECTURE}")
    
    print(f"   Loss function: {loss_type}")
    print(f"   y_train shape: {y_train_final.shape}")
    
    # VERIFY DATA BEFORE TRAINING
    print("\n🔍 Verifying data consistency...")
    sample_image = x_train[0]
    print(f"   Sample image - Range: [{sample_image.min():.3f}, {sample_image.max():.3f}], Shape: {sample_image.shape}")

    # Check for double preprocessing
    if sample_image.max() <= 0.1:
        print("❌ WARNING: Data appears to be over-normalized! Check for double preprocessing.")
        print("   This might indicate double preprocessing. Data should be in appropriate range:")
        if params.QUANTIZE_MODEL and params.ESP_DL_QUANTIZE:
            print("   Expected: UINT8 [0, 255] for ESP-DL quantization")
        elif params.QUANTIZE_MODEL:
            print("   Expected: Float32 [0, 1] for standard quantization")  
        else:
            print("   Expected: Float32 [0, 1] for float32 training")

    print(f"✅ Data verification:")
    print(f"   Train range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"   Mean: {x_train.mean():.3f}, Std: {x_train.std():.3f}")

    if x_train.std() < 0.01:
        print("❌ WARNING: Data has very low variance - might be over-normalized!")
    
    # Create representative dataset for quantization
    representative_data = create_qat_representative_dataset(x_train_raw) ## This should use for_training=False internally
    
    # MODEL CREATION WITH QUANTIZATION AWARENESS
    use_qat = params.QUANTIZE_MODEL and params.USE_QAT and QAT_AVAILABLE
    
    print(f"\n🔧 Creating model...")
    print(f"   Using QAT: {use_qat}")
    print(f"   Strategy: {'Multi-GPU' if strategy else 'Single device'}")
    
    if use_qat:
        print("🎯 Creating model with Quantization Aware Training...")
        if strategy:
            with strategy.scope():
                model = create_qat_model()
                loss_type = 'categorical' if params.MODEL_ARCHITECTURE == "original_haverland" else 'sparse'                                                                                            
                model = compile_model(model, loss_type=loss_type)
        else:
            model = create_qat_model()
            loss_type = 'categorical' if params.MODEL_ARCHITECTURE == "original_haverland" else 'sparse'                                                                                            
            model = compile_model(model, loss_type=loss_type)
    else:
        print("🔧 Creating standard model...")
        if strategy:
            with strategy.scope():
                model = create_model()
                loss_type = 'categorical' if params.MODEL_ARCHITECTURE == "original_haverland" else 'sparse'                                                                                            
                model = compile_model(model, loss_type=loss_type)
        else:
            model = create_model()
            loss_type = 'categorical' if params.MODEL_ARCHITECTURE == "original_haverland" else 'sparse'                                                                                            
            model = compile_model(model, loss_type=loss_type)
    
    # Build model with explicit input shape
    print("🔧 Building model with explicit input shape...")
    model.build(input_shape=(None,) + params.INPUT_SHAPE)
    print(f"✅ Model built with input shape: {model.input_shape}")
    
    # Verify model can forward pass
    print("🔍 Verifying model can forward pass...")
    try:
        test_input = tf.random.normal([1] + list(params.INPUT_SHAPE))
        test_output = model(test_input)
        print(f"✅ Model verification passed: input {test_input.shape} -> output {test_output.shape}")
        
        # Also test with actual data
        real_test = tf.convert_to_tensor(x_train[:1], dtype=tf.float32)
        real_output = model(real_test)
        print(f"✅ Real data test - Output range: [{real_output.numpy().min():.3f}, {real_output.numpy().max():.3f}]")
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        raise
        
    # QAT data flow validation
    if params.USE_QAT and params.QUANTIZE_MODEL:
        qat_flow_ok, qat_msg = validate_qat_data_flow(model, x_train[:1])
        if not qat_flow_ok:
            print(f"❌ {qat_msg}")    
    
    # Setup training components
    tflite_manager = TFLiteModelManager(training_dir, debug)
    monitor = TrainingMonitor(training_dir, debug)
    monitor.set_model(model)
    
    # Print comprehensive training summary
    print_training_summary(model, x_train, x_val, x_test, debug)
    model_summary(model)
    save_model_summary_to_file(model, training_dir)
    
    # from utils.preprocess import debug_preprocessing_flow
    # Debug the preprocessing flow
    debug_preprocessing_flow()
    
    # FINAL VERIFICATION - Check for any remaining double processing
    print("\n🔍 FINAL DOUBLE PROCESSING CHECK:")
    print("=" * 40)

    # Check if data was accidentally processed twice
    if x_train.max() < 0.1:
        print("❌ DOUBLE PROCESSING DETECTED!")
        print("   Data range indicates double normalization")
        print("   Expected: [0, 1], Got: [{:.4f}, {:.4f}]".format(x_train.min(), x_train.max()))
    else:
        print("✅ No double processing detected")
        print("   Data range: [{:.3f}, {:.3f}]".format(x_train.min(), x_train.max()))

    # Check data pipeline doesn't reprocess
    print("\n🔍 Checking data pipeline...")
    from utils.data_pipeline import create_tf_dataset_from_arrays
    sample_batch = next(iter(create_tf_dataset_from_arrays(x_train[:1], y_train_final[:1], training=False)))
    pipeline_image, pipeline_label = sample_batch
    print("   Data pipeline output range: [{:.3f}, {:.3f}]".format(
        pipeline_image.numpy().min(), pipeline_image.numpy().max()))
    
    # Create callbacks
    callbacks = create_callbacks(
        training_dir, 
        tflite_manager, 
        representative_data, 
        params.EPOCHS, 
        monitor, 
        debug, 
        validation_data=(x_val, y_val_final)
    )
        
    print("\n🎯 Starting training...")
    print("-" * 60)
    
    start_time = datetime.now()
    
    # DATA AUGMENTATION PIPELINE
    if params.USE_DATA_AUGMENTATION:
        # Use the centralized augmentation setup
        train_dataset, val_dataset, augmentation_pipeline = setup_augmentation_for_training(
            x_train, y_train_final, x_val, y_val_final, debug=debug
        )
        
        # Train with augmented dataset
        history = model.fit(
            train_dataset,
            epochs=params.EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=0
        )
        
    else:
        # Training without augmentation
        print("ℹ️  Training without data augmentation")
        
        if getattr(params, 'USE_TF_DATA_PIPELINE', False):
            print("🔧 Using tf.data pipeline without augmentation...")
            from utils.data_pipeline import create_tf_dataset_from_arrays
            train_dataset = create_tf_dataset_from_arrays(x_train, y_train_final, training=True)
            val_dataset = create_tf_dataset_from_arrays(x_val, y_val_final, training=False)
            
            # Test data pipeline
            print("🧪 Testing data pipeline...")
            sample_batch = next(iter(train_dataset))
            sample_x, sample_y = sample_batch
            print(f"   Sample batch - X range: [{sample_x.numpy().min():.3f}, {sample_x.numpy().max():.3f}]")
            print(f"   Sample batch - Y shape: {sample_y.numpy().shape}")         
            
            history = model.fit(
                train_dataset,
                epochs=params.EPOCHS,
                validation_data=val_dataset,
                callbacks=callbacks,
                verbose=0
            )
        else:
            print("🔧 Using standard arrays without augmentation...")
            # Test data
            print("🧪 Testing data pipeline...")
            sample_x, sample_y = x_train[:1], y_train_final[:1]
            print(f"   Sample batch - X range: [{sample_x.min():.3f}, {sample_x.max():.3f}]")
            print(f"   Sample batch - Y shape: {sample_y.shape}")                                    
            history = model.fit(
                x_train, y_train_final,
                batch_size=params.BATCH_SIZE,
                epochs=params.EPOCHS,
                validation_data=(x_val, y_val_final),
                callbacks=callbacks,
                verbose=0,
                shuffle=True
            )
    
        
    training_time = datetime.now() - start_time
    
    # FINAL MODEL SAVING AND EVALUATION
    print("\n💾 Saving final models...")
    
    # Save final TFLite models based on quantization settings
    final_quantized_size = 0
    final_float_size = 0
    
    if params.QUANTIZE_MODEL:
        try:
            final_quantized, final_quantized_size = tflite_manager.save_as_tflite(
                model, "final_quantized.tflite", quantize=True, representative_data=representative_data
            )
            print(f"✅ Final quantized model saved: {final_quantized_size:.1f} KB")
        except Exception as e:
            print(f"❌ Final quantized TFLite save failed: {e}")
    
    try:
        final_float, final_float_size = tflite_manager.save_as_tflite(
            model, "final_float.tflite", quantize=False
        )
        print(f"✅ Final float model saved: {final_float_size:.1f} KB")
    except Exception as e:
        print(f"❌ Final float TFLite save failed: {e}")
    
    # MODEL EVALUATION
    print("\n📈 Evaluating models...")
    
    try:
        # Evaluate Keras model
        train_accuracy = model.evaluate(x_train, y_train_final, verbose=0)[1]
        val_accuracy = model.evaluate(x_val, y_val_final, verbose=0)[1]
        test_accuracy = model.evaluate(x_test, y_test_final, verbose=0)[1]
        
        print(f"✅ Keras Model Evaluation:")
        print(f"   Train Accuracy: {train_accuracy:.4f}")
        print(f"   Val Accuracy: {val_accuracy:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        
    except Exception as e:
        print(f"❌ Keras model evaluation failed: {e}")
        train_accuracy = val_accuracy = test_accuracy = 0.0
    
    # TFLite model evaluation
    tflite_accuracy = 0.0
    quantized_tflite_path = os.path.join(training_dir, params.TFLITE_FILENAME)
    if os.path.exists(quantized_tflite_path) and params.QUANTIZE_MODEL:
        try:
            tflite_accuracy = evaluate_tflite_model(quantized_tflite_path, x_test, y_test_final)
            print(f"✅ TFLite Model Evaluation:")
            print(f"   Test Accuracy: {tflite_accuracy:.4f}")
            
            # Analyze quantization impact
            analyze_quantization_impact(model, x_test, y_test_final, quantized_tflite_path)
        except Exception as e:
            print(f"❌ TFLite evaluation failed: {e}")
    
    # Save training plots
    monitor.save_training_plots()
    
    # Run diagnostics
    try:
        training_diagnostics(model, x_train, y_train_final, x_val, y_val_final, debug=debug)
        verify_model_predictions(model, x_train[:100], y_train_final[:100])
        
        if debug:
            debug_model_architecture(model, x_train[:10])
    except Exception as e:
        print(f"⚠️  Diagnostics failed: {e}")
    
    # FINAL SUMMARY
    print("\n" + "="*60)
    print("🏁 TRAINING COMPLETED")
    print("="*60)
    print(f"⏱️  Training time: {training_time}")
    print(f"📊 Final Results:")
    print(f"   Best Validation Accuracy: {tflite_manager.best_accuracy:.4f}")
    print(f"   Test Accuracy - Keras: {test_accuracy:.4f}")
    print(f"   Test Accuracy - TFLite: {tflite_accuracy:.4f}")
    print(f"   Quantized Model Size: {final_quantized_size:.1f} KB")
    print(f"   Float Model Size: {final_float_size:.1f} KB")
    
    print(f"\n💾 Models saved to: {training_dir}")
    if params.QUANTIZE_MODEL:
        print(f"   Best quantized: {params.TFLITE_FILENAME}")
        print(f"   Final quantized: final_quantized.tflite")
    print(f"   Final float: final_float.tflite")
    print(f"   Training log: training_log.csv")
    print(f"   Training plot: training_history.png")
    
    # Save training configuration
    save_training_config(training_dir, final_quantized_size, final_float_size, tflite_manager,
                        test_accuracy, tflite_accuracy, training_time, debug, model=model)
                        
    # Save final model checkpoint
    print("💾 Saving final model checkpoint...")
    final_checkpoint_path = os.path.join(training_dir, "final_model.keras")
    try:
        model.save(final_checkpoint_path)
        print(f"✅ Final model saved: {final_checkpoint_path}")
    except Exception as e:
        print(f"❌ Final model save failed: {e}")
    
    # Export to ONNX if available
    try:
        onnx_path, onnx_size = tflite_manager.export_to_onnx(model, "final_model.onnx")
        if onnx_path:
            print(f"✅ Final model exported to ONNX: {onnx_path} ({onnx_size:.1f} KB)")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
    
    # Print quantization summary
    print(f"\n🎯 Quantization Summary:")
    print(f"   QAT Used: {use_qat}")
    print(f"   ESP-DL Mode: {params.ESP_DL_QUANTIZE}")
    print(f"   Final Quantization: {params.QUANTIZE_MODEL}")
    
    if use_qat and params.QUANTIZE_MODEL:
        accuracy_drop = test_accuracy - tflite_accuracy if tflite_accuracy > 0 else 0
        print(f"   QAT Effectiveness - Accuracy drop: {accuracy_drop:.4f}")
        if accuracy_drop < 0.02:
            print("   ✅ QAT performed well (minimal accuracy drop)")
        else:
            print("   ⚠️  QAT may need tuning (significant accuracy drop)")
    
    return model, history, training_dir
    

def save_training_config(training_dir, quantized_size, float_size, tflite_manager, 
                        test_accuracy, tflite_accuracy, training_time, debug=False, model=None):
    """Save training configuration and results to file with enhanced parameters"""
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
        f.write(f"  Input shape: [{params.INPUT_SHAPE}]\n")
        f.write(f"  Classes: {params.NB_CLASSES}\n")
        
        f.write(f"\nTRAINING CONFIG:\n")
        f.write(f"  Batch size: {params.BATCH_SIZE}\n")
        f.write(f"  Epochs: {params.EPOCHS}\n")
        f.write(f"  Learning rate: {params.LEARNING_RATE}\n")
        f.write(f"  Early stopping: {'Enabled' if params.USE_EARLY_STOPPING else 'Disabled'}\n")
        if params.USE_EARLY_STOPPING:
            f.write(f"    Monitor: {params.EARLY_STOPPING_MONITOR}\n")
            f.write(f"    Patience: {params.EARLY_STOPPING_PATIENCE}\n")
            f.write(f"    Min delta: {params.EARLY_STOPPING_MIN_DELTA}\n")
        
        f.write(f"  Learning rate scheduler:\n")
        f.write(f"    Monitor: {params.LR_SCHEDULER_MONITOR}\n")
        f.write(f"    Patience: {params.LR_SCHEDULER_PATIENCE}\n")
        f.write(f"    Factor: {params.LR_SCHEDULER_FACTOR}\n")
        f.write(f"    Min LR: {params.LR_SCHEDULER_MIN_LR}\n")
        
        f.write(f"  Quantization: {params.QUANTIZE_MODEL}\n")
        if params.QUANTIZE_MODEL:
            f.write(f"    ESP-DL Quantization: {params.ESP_DL_QUANTIZE}\n")
            f.write(f"    Num samples: {params.QUANTIZE_NUM_SAMPLES}\n")
        
        f.write(f"  Debug mode: {'Enabled' if debug else 'Disabled'}\n")
        
        f.write(f"\nHARDWARE CONFIG:\n")
        f.write(f"  GPU Usage: {'Enabled' if params.USE_GPU else 'Disabled'}\n")
        if params.USE_GPU:
            f.write(f"  Memory growth: {params.GPU_MEMORY_GROWTH}\n")
            f.write(f"  Memory limit: {params.GPU_MEMORY_LIMIT} MB\n")
        
        # Add hyperparameter summary
        try:
            from parameters import get_hyperparameter_summary_text
            hyperparam_text = get_hyperparameter_summary_text()
            f.write(hyperparam_text)
            f.write("\n\n" + "=" * 50 + "\n\n")
        except ImportError:
            print("⚠️  Could not import hyperparameter summary function")
            
        f.write(f"\nMODEL SUMMARY:\n")    
        # Try to read from the saved model summary file
        summary_file_path = os.path.join(training_dir, "model_summary.txt")
        if os.path.exists(summary_file_path):
            try:
                with open(summary_file_path, 'r') as summary_file:
                    model_summary_content = summary_file.read()
                f.write(model_summary_content)
                f.write("\n")  # Add a newline after the summary
            except Exception as e:
                f.write(f"Could not read model summary file: {e}\n")
        else:
            # Fallback to the model_summary function
            model_summary_text = model_summary(model)
            if model_summary_text is not None:
                f.write(model_summary_text)
            else:
                f.write("Model summary not available\n")
            
        f.write(f"\nGENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Also save as CSV for benchmarking
    save_training_csv(training_dir, quantized_size, float_size, tflite_manager,
                     test_accuracy, tflite_accuracy, training_time)

def save_training_csv(training_dir, quantized_size, float_size, tflite_manager,
                     test_accuracy, tflite_accuracy, training_time):
    """Save training results to CSV for benchmarking"""
    csv_path = os.path.join(training_dir, "training_results.csv")
    
    # Extract data source information
    data_sources_str = ";".join([f"{src['name']}({src.get('weight', 1.0)})" 
                               for src in params.DATA_SOURCES])
    
    with open(csv_path, 'w') as f:
        f.write("parameter,value\n")
        f.write(f"timestamp,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"model_architecture,{params.MODEL_ARCHITECTURE}\n")
        f.write(f"input_shape,{params.INPUT_SHAPE}\n")
        f.write(f"nb_classes,{params.NB_CLASSES}\n")
        f.write(f"data_sources,{data_sources_str}\n")
        f.write(f"batch_size,{params.BATCH_SIZE}\n")
        f.write(f"epochs,{params.EPOCHS}\n")
        f.write(f"learning_rate,{params.LEARNING_RATE}\n")
        f.write(f"use_early_stopping,{params.USE_EARLY_STOPPING}\n")
        f.write(f"early_stopping_monitor,{params.EARLY_STOPPING_MONITOR}\n")
        f.write(f"early_stopping_patience,{params.EARLY_STOPPING_PATIENCE}\n")
        f.write(f"lr_scheduler_monitor,{params.LR_SCHEDULER_MONITOR}\n")
        f.write(f"lr_scheduler_patience,{params.LR_SCHEDULER_PATIENCE}\n")
        f.write(f"lr_scheduler_factor,{params.LR_SCHEDULER_FACTOR}\n")
        f.write(f"QUANTIZE_MODEL,{params.QUANTIZE_MODEL}\n")
        f.write(f"esp_dl_quantize,{params.ESP_DL_QUANTIZE}\n")
        f.write(f"quantize_num_samples,{params.QUANTIZE_NUM_SAMPLES}\n")
        f.write(f"use_gpu,{params.USE_GPU}\n")
        f.write(f"keras_test_accuracy,{test_accuracy:.4f}\n")
        f.write(f"tflite_test_accuracy,{tflite_accuracy:.4f}\n")
        f.write(f"best_val_accuracy,{tflite_manager.best_accuracy:.4f}\n")
        f.write(f"quantized_model_size_kb,{quantized_size:.1f}\n")
        f.write(f"float_model_size_kb,{float_size:.1f}\n")
        f.write(f"training_time,{training_time}\n")
        f.write(f"optimizer,{params.OPTIMIZER_TYPE}\n")
        #f.write(f"model_parameters,{model.count_params()}\n")

def test_all_models(x_train_raw, y_train_raw, x_val_raw, y_val_raw, models_to_test=None, debug=False):
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
        
        # Set current model - IMPORTANT: Update the actual parameter
        params.MODEL_ARCHITECTURE = model_name
        
        try:
            model = create_model()
            
            if params.MODEL_ARCHITECTURE == "original_haverland":  # Use current model name
                loss_fn = 'categorical_crossentropy'
                y_train_current = tf.keras.utils.to_categorical(y_train_raw, params.NB_CLASSES)  
                y_val_current = tf.keras.utils.to_categorical(y_val_raw, params.NB_CLASSES)      
            else:
                loss_fn = 'sparse_categorical_crossentropy'
                y_train_current = y_train_raw  
                y_val_current = y_val_raw      
            
            model.compile(
                optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy']
            )
            
            train_samples = min(1000, len(x_train_raw))
            val_samples = min(200, len(x_val_raw))
            
            history = model.fit(
                x_train_raw[:train_samples], y_train_current[:train_samples],  
                validation_data=(x_val_raw[:val_samples], y_val_current[:val_samples]),  
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
        
        # Clear session between models
        tf.keras.backend.clear_session()
    
    # Restore original model - IMPORTANT: Reset the parameter
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

def train_specific_models(models_to_train, debug=False):
    """Train specific model architectures with full training"""
    original_model = params.MODEL_ARCHITECTURE
    results = {}
    
    print(f"\n🚀 TRAINING {len(models_to_train)} MODELS")
    print("=" * 60)
    
    for model_name in models_to_train:
        print(f"\n🎯 Training: {model_name}")
        print("=" * 50)
        
        # Set current model - IMPORTANT: Update the actual parameter
        params.MODEL_ARCHITECTURE = model_name
        
        try:
            # Train model with full configuration
            model, history, output_dir = train_model(debug=debug)
            
            if model is not None:  # Only process if training was successful
                # Extract results
                from analyse import evaluate_tflite_model
                quantized_tflite_path = os.path.join(output_dir, params.TFLITE_FILENAME)
                
                # Load test data for evaluation
                (x_train_raw, y_train_raw), (x_val_raw, y_val_raw), (x_test_raw, y_test_raw) = get_data_splits()
                x_test = preprocess_images(x_test_raw, for_training=False)
                
                # Handle labels based on current model type
                if params.MODEL_ARCHITECTURE == "original_haverland":
                    y_test_current = tf.keras.utils.to_categorical(y_test_raw, params.NB_CLASSES) 
                else:
                    y_test_current = y_test_raw 
                
                # Evaluate models
                keras_test_accuracy = model.evaluate(x_test, y_test_current, verbose=0)[1]
                
                tflite_accuracy = 0.0
                if os.path.exists(quantized_tflite_path) and params.QUANTIZE_MODEL:
                    tflite_accuracy = evaluate_tflite_model(quantized_tflite_path, x_test, y_test_current)
                
                results[model_name] = {
                    'keras_test_accuracy': keras_test_accuracy,
                    'tflite_accuracy': tflite_accuracy,
                    'output_dir': output_dir,
                    'params': model.count_params(),
                    'model': model
                }
                
                print(f"✅ {model_name} completed:")
                print(f"   Keras Test Accuracy: {keras_test_accuracy:.4f}")
                print(f"   TFLite Accuracy: {tflite_accuracy:.4f}")
                print(f"   Output: {output_dir}")
            else:
                print(f"❌ {model_name} training failed (returned None)")
                results[model_name] = {
                    'keras_test_accuracy': 0.0,
                    'tflite_accuracy': 0.0,
                    'error': 'Training returned None'
                }
                
        except Exception as e:
            print(f"❌ {model_name} training failed: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            results[model_name] = {
                'keras_test_accuracy': 0.0,
                'tflite_accuracy': 0.0,
                'error': str(e)
            }
        
        # Clear session to free memory between models
        tf.keras.backend.clear_session()
    
    # Restore original model - IMPORTANT: Reset the parameter
    params.MODEL_ARCHITECTURE = original_model
    
    # Print summary
    print("\n" + "="*60)
    print("🏁 ALL MODELS TRAINING COMPLETED")
    print("="*60)
    
    successful_models = {k: v for k, v in results.items() if 'error' not in v}
    if successful_models:
        sorted_results = sorted(successful_models.items(), 
                              key=lambda x: x[1]['keras_test_accuracy'], 
                              reverse=True)
        
        print("📊 FINAL RANKINGS:")
        for i, (model_name, metrics) in enumerate(sorted_results, 1):
            print(f"{i:2d}. {model_name:35} -> Keras: {metrics['keras_test_accuracy']:.4f} | TFLite: {metrics['tflite_accuracy']:.4f} | Params: {metrics['params']:,}")
    
    return results
    
    
def validate_qat_data_flow(model, x_train_sample, debug=False):
    """
    Validate that QAT data flow is consistent between training and inference
    """
    if not params.USE_QAT or not params.QUANTIZE_MODEL:
        return True, "QAT not enabled"
    
    print("\n🔍 VALIDATING QAT DATA FLOW")
    print("=" * 50)
    
    # Get a sample batch for testing
    sample_batch = x_train_sample[:1]
    
    print(f"Sample batch - dtype: {sample_batch.dtype}, range: [{sample_batch.min():.3f}, {sample_batch.max():.3f}]")
    
    # Test model forward pass
    try:
        output = model(sample_batch)
        print(f"✅ Model forward pass successful")
        print(f"   Output range: [{output.numpy().min():.3f}, {output.numpy().max():.3f}]")
        
        # Check for quantization layers
        quant_layers = [layer for layer in model.layers if any(quant_term in layer.name for quant_term in ['quant', 'qat'])]
        print(f"   Quantization layers found: {len(quant_layers)}")
        
        return True, "QAT data flow validated"
        
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        return False, f"QAT data flow failed: {e}"

def check_training_inference_alignment(x_train_sample=None):
    """
    Check if training and inference preprocessing are aligned
    FIXED: Recognizes that training≠inference is correct for standard quantization
    """
    print("\n🔍 CHECKING TRAINING/INFERENCE ALIGNMENT")
    print("=" * 50)
    
    from utils.preprocess import get_qat_training_format, preprocess_images
    
    # Get expected formats
    train_dtype, train_min, train_max, train_desc = get_qat_training_format()
    
    # Use provided sample or create test data
    if x_train_sample is not None:
        test_data = x_train_sample[:5]
        print("   Using real training data for alignment check")
    else:
        test_data = np.random.randint(0, 255, (5, 28, 28, 1), dtype=np.uint8)
        print("   Using synthetic data for alignment check")
    
    # Process for training and inference
    train_processed = preprocess_images(test_data, for_training=True)
    infer_processed = preprocess_images(test_data, for_training=False)
    
    print(f"Expected training format: {train_desc}")
    print(f"Actual training:   {train_processed.dtype} [{train_processed.min():.1f}, {train_processed.max():.1f}]")
    print(f"Actual inference:  {infer_processed.dtype} [{infer_processed.min():.1f}, {infer_processed.max():.1f}]")
    
    # FIXED: Different behavior based on quantization mode
    if params.QUANTIZE_MODEL and not params.USE_QAT:
        # Standard quantization: Training uses float32, inference uses uint8 - THIS IS CORRECT
        print("✅ STANDARD QUANTIZATION: Training≠Inference is EXPECTED")
        print("   - Training: Float32 [0,1] for stable training")
        print("   - Inference: UINT8 [0,255] for TFLite quantization")
        print("   - This prevents double quantization during conversion")
        return True
    elif params.USE_QAT and params.QUANTIZE_MODEL:
        # QAT: Training and inference should be identical
        aligned = (train_processed.dtype == infer_processed.dtype)
        if aligned:
            print("✅ QAT ALIGNMENT: PERFECT - training matches inference")
            return True
        else:
            print("❌ QAT ALIGNMENT: MISMATCH - training should match inference for QAT")
            return False
    else:
        # No quantization: Both should be float32
        aligned = (train_processed.dtype == infer_processed.dtype)
        if aligned:
            print("✅ FLOAT32 ALIGNMENT: PERFECT")
            return True
        else:
            print("❌ FLOAT32 ALIGNMENT: MISMATCH")
            return False

def main():
    """Main entry point"""
    args = parse_arguments()
    tflite_checkpoint_callback = None  # Store reference to the callback
    
    diagnose_quantization_settings()

    # Validate and correct parameters
    is_valid, corrected_params, message = validate_quantization_parameters()
    print(message)
    
    if not is_valid:
        # Apply corrections
        params.QUANTIZE_MODEL = corrected_params['QUANTIZE_MODEL']
        params.USE_QAT = corrected_params['USE_QAT'] 
        params.ESP_DL_QUANTIZE = corrected_params['ESP_DL_QUANTIZE']
        print("✅ Parameters corrected automatically")
    
    try:
       
        # DEBUG: Print arguments
        if args.debug:
            print("🔍 Command line arguments:")
            print(f"   debug: {args.debug}")
            print(f"   use_tuner: {getattr(args, 'use_tuner', False)}")
            print(f"   num_trials: {getattr(args, 'num_trials', 5)}")
            print(f"   advanced: {getattr(args, 'advanced', False)}")
            print(f"   test_all_models: {args.test_all_models}")
            print(f"   train: {getattr(args, 'train', None)}")
            print(f"   train_all: {getattr(args, 'train_all', False)}")
            print(f"   Current MODEL_ARCHITECTURE: {params.MODEL_ARCHITECTURE}")
        
        # Handle different operation modes
        if getattr(args, 'use_tuner', False):
            # HYPERPARAMETER TUNING MODE
            print("🚀 Starting hyperparameter tuning...")
            
            # Load data for tuning
            print("📊 Loading dataset for tuning...")
            (x_train_raw, y_train_raw), (x_val_raw, y_val_raw), (x_test_raw, y_test_raw) = get_data_splits()
            
            # Preprocess data
            print("🔄 Preprocessing images for tuning...")
            x_train = preprocess_images(x_train_raw, for_training=True)
            x_val = preprocess_images(x_val_raw, for_training=True)
            
            # Handle labels based on model type
            if params.MODEL_ARCHITECTURE == "original_haverland":
                y_train = tf.keras.utils.to_categorical(y_train_raw, params.NB_CLASSES)
                y_val = tf.keras.utils.to_categorical(y_val_raw, params.NB_CLASSES)
            else:
                y_train = y_train_raw
                y_val = y_val_raw
            
            from tuner import run_architecture_tuning
            
            # Run tuning to find best hyperparameters
            tuning_results = run_architecture_tuning(
                x_train=x_train,
                y_train=y_train, 
                x_val=x_val,
                y_val=y_val,
                num_trials=getattr(args, 'num_trials', params.TUNER_MAX_TRIALS),
                debug=args.debug
            )
            
            # if tuning_results:
                # print(f"\n🎯 APPLYING TUNED HYPERPARAMETERS TO MAIN TRAINING:")
                # print("=" * 50)
                
                # # Apply tuned parameters
                # params.OPTIMIZER_TYPE = tuning_results['optimizer']
                # params.LEARNING_RATE = tuning_results['learning_rate']
                # params.BATCH_SIZE = tuning_results['batch_size']
                
                # print(f"   Optimizer: {tuning_results['optimizer']}")
                # print(f"   Learning Rate: {tuning_results['learning_rate']}")
                # print(f"   Batch Size: {tuning_results['batch_size']}")
                # print(f"   Tuned Val Accuracy: {tuning_results['val_accuracy']:.4f}")
                # print("=" * 50)
                
                # # Now run the main training with tuned parameters
                # print("🚀 Starting main training with tuned hyperparameters...")
                # model, history, output_dir = train_model(debug=args.debug)
                
                # if model is not None:
                    # print(f"\n✅ Training completed successfully!")
                    # print(f"📁 Output directory: {output_dir}")
                    
                    # # Compare tuned vs final results
                    # if hasattr(history, 'history') and history.history:
                        # final_val_acc = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0
                        # improvement = final_val_acc - tuning_results['val_accuracy']
                        # print(f"📈 Tuning vs Final: {tuning_results['val_accuracy']:.4f} → {final_val_acc:.4f} (Δ{improvement:+.4f})")
                # else:
                    # print("❌ Main training failed after tuning")
            # else:
                # print("❌ Hyperparameter tuning failed, using default parameters")
                # model, history, output_dir = train_model(debug=args.debug)
                
        elif args.test_all_models:
            # TEST ALL MODELS MODE
            print("🧪 Testing all available models...")
            test_all_models(debug=args.debug)  # Let test_all_models handle its own data loading
            
        elif getattr(args, 'train', None) is not None:
            # TRAIN SPECIFIC MODELS MODE
            models_to_train = args.train
            print(f"🚀 Training specific models: {models_to_train}")
            results = train_specific_models(models_to_train, debug=args.debug)
            
            # Print summary
            successful_models = {k: v for k, v in results.items() if 'error' not in v}
            if successful_models:
                print(f"\n🏁 Successfully trained {len(successful_models)} models")
                for model_name, metrics in successful_models.items():
                    print(f"   {model_name}: {metrics.get('keras_test_accuracy', 0):.4f}")
            
        elif getattr(args, 'train_all', False):
            # TRAIN ALL MODELS MODE
            print(f"🚀 Training all available models: {params.AVAILABLE_MODELS}")
            results = train_specific_models(params.AVAILABLE_MODELS, debug=args.debug)
            
            # Print summary
            successful_models = {k: v for k, v in results.items() if 'error' not in v}
            if successful_models:
                print(f"\n🏁 Successfully trained {len(successful_models)} models")
                sorted_results = sorted(successful_models.items(), 
                                      key=lambda x: x[1].get('keras_test_accuracy', 0), 
                                      reverse=True)
                for i, (model_name, metrics) in enumerate(sorted_results, 1):
                    print(f"   {i}. {model_name}: {metrics.get('keras_test_accuracy', 0):.4f}")
        
        else:
            # NORMAL SINGLE MODEL TRAINING MODE
            print(f"🚀 Training single model: {params.MODEL_ARCHITECTURE}")
            
            # We need to modify train_model to return the TFLiteCheckpoint callback
            # For now, let's handle cleanup differently
            model, history, output_dir = train_model(debug=args.debug)
            print(f"\n✅ Training completed successfully!")
            print(f"📁 Output directory: {output_dir}")
            
            # Display final results
            if hasattr(history, 'history') and history.history:
                final_val_acc = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0
                final_train_acc = history.history['accuracy'][-1] if 'accuracy' in history.history else 0
                print(f"📊 Final metrics - Train: {final_train_acc:.4f}, Val: {final_val_acc:.4f}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
        
    except Exception as e:
        print(f"\n❌ MAIN - Operation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        
        # Provide helpful error information
        if "CUDA" in str(e) or "GPU" in str(e):
            print("\n💡 GPU-related error detected. Try:")
            print("   - Setting USE_GPU = False in parameters.py")
            print("   - Checking CUDA/cuDNN installation")
            print("   - Reducing batch size")
        
        elif "memory" in str(e).lower():
            print("\n💡 Memory error detected. Try:")
            print("   - Reducing batch size")
            print("   - Setting GPU_MEMORY_LIMIT in parameters.py")
            print("   - Using a smaller model architecture")
        
        elif "shape" in str(e).lower():
            print("\n💡 Shape mismatch error. Check:")
            print("   - Input shape in parameters.py matches your data")
            print("   - Model architecture compatibility")
            print("   - Data preprocessing steps")
    
    finally:
        # CLEANUP: Delete checkpoints if not in debug mode
        if not args.debug:
            print("\n🧹 Cleaning up intermediate checkpoints...")
            try:
                # Clean up checkpoint directories and files
                if 'output_dir' in locals() and os.path.exists(output_dir):
                    checkpoints_dir = os.path.join(output_dir, "checkpoints")
                    if os.path.exists(checkpoints_dir):
                        import shutil
                        shutil.rmtree(checkpoints_dir)
                        print("🗑️  Deleted checkpoints directory")
                    
                    # Also delete individual checkpoint files in the main directory
                    for file in os.listdir(output_dir):
                        if file.startswith("checkpoint_epoch_") and file.endswith(".keras"):
                            file_path = os.path.join(output_dir, file)
                            os.remove(file_path)
                            print(f"🗑️  Deleted checkpoint: {file}")
                
                # Also clean up from any training that happened in train_specific_models
                if 'results' in locals():
                    for model_name, metrics in results.items():
                        if 'output_dir' in metrics and os.path.exists(metrics['output_dir']):
                            checkpoints_dir = os.path.join(metrics['output_dir'], "checkpoints")
                            if os.path.exists(checkpoints_dir):
                                import shutil
                                shutil.rmtree(checkpoints_dir)
                            
                            # Delete individual checkpoint files
                            for file in os.listdir(metrics['output_dir']):
                                if file.startswith("checkpoint_epoch_") and file.endswith(".keras"):
                                    file_path = os.path.join(metrics['output_dir'], file)
                                    os.remove(file_path)
            except Exception as e:
                print(f"⚠️  Cleanup failed: {e}")
        
        # Cleanup and final message
        print("\n" + "="*60)
        print("🏁 Program finished")
        print("="*60)

if __name__ == "__main__":
    main()
    clear_cache()